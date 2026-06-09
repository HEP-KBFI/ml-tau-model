import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    """Multi-class Focal Loss"""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # Class weights (tensor of size num_classes)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - raw logits
            targets: (batch_size,) - class indices
        """
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Get predicted probabilities for true classes
        pt = torch.exp(-ce_loss)

        # Apply focal loss formula
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# Usage in your model:
# self.focal_loss = FocalLoss(alpha=torch.tensor([0.1, 1.0, 1.0, 1.0]), gamma=2.0)  # Lower weight for background


class SigmoidFocalLoss(nn.Module):
    """Wrapper to make sigmoid_focal_loss behave like a module for consistency."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(
            inputs,
            targets.float(),  # targets need to be float for sigmoid
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )


class TauLoss(nn.Module):
    """Unified loss module for Tau tagging, charge, decay mode, and kinematics."""

    def __init__(self, l_m=0.2, label_smoothing=0.1):
        super().__init__()
        self.l_m = l_m
        # Tagging: all jets (background=0, signal=1)
        self.tag_loss_fn = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=label_smoothing
        )
        # Charge: signal taus only
        self.charge_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        # Decay Mode: signal taus only
        self.dm_loss_fn = nn.CrossEntropyLoss(reduction="none")
        # Kinematics: signal taus only
        self.kin_loss_fn = nn.HuberLoss(reduction="none", delta=1.0)

    def compute_tagging_loss(self, predictions, targets, weights):
        """CrossEntropy loss for background vs signal classification."""
        loss = self.tag_loss_fn(predictions, targets.long())
        return (loss * weights).mean()

    def compute_charge_loss(self, predictions, targets, weights):
        """BCE loss for charge classification (+1 vs -1)."""
        # Map physical charges {-1, 1} to binary labels {0, 1}.
        # (targets == 1) maps +1 -> 1 and -1 -> 0.
        binary_targets = (targets == 1).float()
        loss = self.charge_loss_fn(predictions, binary_targets)
        return (loss * weights).mean()

    def compute_decay_mode_loss(self, predictions, targets, weights):
        """CrossEntropy loss for decay mode classification."""
        loss = self.dm_loss_fn(predictions, targets.long())
        return (loss * weights).mean()

    def _compute_kinematics_loss_per_sample(self, predictions, targets):
        """Internal helper to compute per-sample Huber loss for (log pt, deta, phi_chord, log m)."""
        log_pt_loss = self.kin_loss_fn(predictions[:, 0], targets[:, 0])
        delta_eta_loss = self.kin_loss_fn(predictions[:, 1], targets[:, 1])
        # Phi chord loss: treat (sin, cos) as a 2D unit-vector difference
        phi_chord_loss = torch.sqrt(
            (predictions[:, 2] - targets[:, 2]) ** 2
            + (predictions[:, 3] - targets[:, 3]) ** 2
            + 1e-8
        )
        log_mass_loss = self.kin_loss_fn(predictions[:, 4], targets[:, 4])

        # Combined per-sample loss
        per_sample_loss = (
            log_pt_loss + delta_eta_loss + phi_chord_loss + self.l_m * log_mass_loss
        ) / (3.0 + self.l_m)

        return per_sample_loss, {
            "log_pt": log_pt_loss,
            "delta_eta": delta_eta_loss,
            "phi_chord": phi_chord_loss,
            "log_mass": log_mass_loss,
        }

    def compute_kinematics_loss(self, predictions, targets, weights):
        """Huber loss for (log pt, deta, phi_chord, log m)."""
        per_sample_loss, components_per_sample = self._compute_kinematics_loss_per_sample(
            predictions, targets
        )

        components = {
            k: (v * weights).mean() for k, v in components_per_sample.items()
        }

        return (per_sample_loss * weights).mean(), components

    def compute_multi_task_losses(self, predictions_dict, targets_dict, sample_weights):
        """Helper for MultiParTau to compute all 4 task losses at once with masking."""
        is_tau_mask = targets_dict["is_tau"].bool()

        # 1. Tagging loss — all jets
        tag_loss = self.compute_tagging_loss(
            predictions_dict["is_tau"], targets_dict["is_tau"], sample_weights
        )

        if not is_tau_mask.any():
            zero = tag_loss.new_zeros(())
            return torch.stack([tag_loss, zero, zero, zero]), {}

        tau_weights = sample_weights[is_tau_mask]

        # 2. Decay Mode loss — signal only
        dm_loss = self.compute_decay_mode_loss(
            predictions_dict["decay_mode"][is_tau_mask],
            targets_dict["decay_mode"][is_tau_mask],
            tau_weights,
        )

        # 3. Charge loss — signal only
        charge_loss = self.compute_charge_loss(
            predictions_dict["charge"][is_tau_mask],
            targets_dict["charge"][is_tau_mask],
            tau_weights,
        )

        # 4. Kinematics loss — signal only
        kin_loss, kin_components = self.compute_kinematics_loss(
            predictions_dict["kinematics"][is_tau_mask],
            targets_dict["kinematics"][is_tau_mask],
            tau_weights,
        )

        return torch.stack([tag_loss, dm_loss, charge_loss, kin_loss]), kin_components

    def compute_combined_loss(
        self,
        predictions,
        targets,
        weights,
        w_tag=1.0,
        w_dm=1.0,
        w_charge=1.0,
        w_kin=1.0,
    ):
        """
        Compute a single scalar loss representing the weighted average of per-jet
        combined losses. This is the logic used for validation monitoring and
        combined-loss training.
        """
        is_tau_mask = targets["is_tau"].bool()

        # 1. Tagging loss — all jets
        tag_per_jet = self.tag_loss_fn(predictions["is_tau"], targets["is_tau"].long())
        combined_per_jet = w_tag * tag_per_jet

        if is_tau_mask.any():
            # 2. Decay Mode loss — signal only
            dm_per_jet = self.dm_loss_fn(
                predictions["decay_mode"][is_tau_mask],
                targets["decay_mode"][is_tau_mask].long(),
            )

            # 3. Charge loss — signal only
            charge_targets = (targets["charge"][is_tau_mask] == 1).float()
            charge_per_jet = self.charge_loss_fn(
                predictions["charge"][is_tau_mask],
                charge_targets,
            )

            # 4. Kinematics loss — signal only
            kin_per_jet, _ = self._compute_kinematics_loss_per_sample(
                predictions["kinematics"][is_tau_mask],
                targets["kinematics"][is_tau_mask],
            )

            # Add signal-only terms into combined per-jet loss
            combined_per_jet[is_tau_mask] += (
                w_dm * dm_per_jet + w_charge * charge_per_jet + w_kin * kin_per_jet
            )

        # Multiply each jet's combined loss by its weight, then average
        return (combined_per_jet * weights).mean()
