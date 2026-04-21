import torch
import numpy as np
import awkward as ak
import torch.nn as nn
import lightning as L
from omegaconf import DictConfig
from mltau.tools import general as g

# from mltau.tools.optimizers.lookahead import Lookahead
from mltau.tools.io.general import BatchInputs
from mltau.tools.losses import SigmoidFocalLoss
from mltau.tools.logging import logger
from mltau.models.MultiParTau import ParTau


class ParTauModule(L.LightningModule):
    def __init__(self, cfg: DictConfig, input_dim: int, num_dm_classes: int):
        super().__init__()
        self.cfg = cfg
        self.ParTau = ParTau(
            input_dim=input_dim,
            num_dm_classes=num_dm_classes,  # Number of decay modes we wish to classify
            num_layers=2,  # cfg.models.ParticleTransformer.hyperparameters.num_layers,
            embed_dims=[
                256,
                512,
                256,
            ],  # cfg.models.ParticleTransformer.hyperparameters.embed_dims,
            use_pre_activation_pair=False,
            for_inference=False,
            use_amp=False,
            metric="eta-phi",
        )

        # Initialize loss functions once to avoid memory allocation overhead
        self.charge_loss = nn.BCEWithLogitsLoss(
            reduction="none"
        )  # For raw logits - avoids bias from double sigmoid
        # self.charge_loss = SigmoidFocalLoss(
        #     reduction="none", gamma=0.0, alpha=0.5
        # )  # class balance, so one could use BCE with sigmoid also.
        self.tagging_loss = SigmoidFocalLoss(
            alpha=0.2, gamma=2.0, reduction="none"
        )  # class imbalance
        self.decay_mode_loss = nn.CrossEntropyLoss(reduction="none")
        self.kinematics_loss = nn.HuberLoss(reduction="none", delta=1.0)

        # Uncertainty-based weighting parameters (learnable log variances)
        # Following Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
        # Initialize log variance parameters to 0 (corresponding to variance = 1)
        self.log_var_tagging = nn.Parameter(torch.zeros(1))
        self.log_var_charge = nn.Parameter(torch.zeros(1))
        self.log_var_decay_mode = nn.Parameter(torch.zeros(1))
        self.log_var_kinematics = nn.Parameter(torch.zeros(1))

    def training_step(self, batch, batch_idx):
        predictions, targets, weights = self.forward(batch)

        metrics = self.calculate_metrics(
            targets=targets, predictions=predictions, weights=weights
        )
        for key, value in metrics.items():
            self.training_loss_accumulator[key].append(value.detach())

        self.log(
            "LR",
            self.optimizers().param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return metrics["loss"]

    def predict_step(self, batch, _batch_idx):
        """
        Runs inference for a batch during prediction (trainer.predict()).
        Returns only the predictions dict (no losses/metrics).
        """
        # Unpack batch if needed (BatchInputs or tuple)
        if isinstance(batch, (list, tuple)):
            # Standard tuple from DataLoader
            predictions, _, _ = self.forward(batch)
        else:
            # Already a BatchInputs or similar
            predictions, _, _ = self.forward((batch,))
        return predictions

    def test_step(self, batch, _batch_idx):
        return self.forward(batch)[0]

    def configure_optimizers(self):
        # AdamW is generally preferred for transformer architectures
        # Use a 10x lower LR for log_var parameters to prevent them from
        # racing ahead of the main network and collapsing tasks
        log_var_params = [
            self.log_var_tagging,
            self.log_var_charge,
            self.log_var_decay_mode,
            self.log_var_kinematics,
        ]
        log_var_param_ids = {id(p) for p in log_var_params}
        main_params = [p for p in self.parameters() if id(p) not in log_var_param_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": main_params, "lr": self.cfg.training.lr},
                {"params": log_var_params, "lr": self.cfg.training.lr * 0.1},
            ],
        )
        # if self.cfg.training.optimizer.use_lookahead:
        #     optimizer = Lookahead(base_optimizer=optimizer, k=6, alpha=0.5)

        # Use a more reliable method to calculate T_max
        # Check if estimated_stepping_batches is available and valid
        estimated_steps = getattr(self.trainer, "estimated_stepping_batches", None)

        if estimated_steps is None or estimated_steps <= 0:
            # Fallback: calculate based on config (will be approximate but functional)
            max_epochs = self.cfg.training.trainer.max_epochs
            # Use a conservative estimate of steps per epoch
            # This will be less precise but the scheduler will still work
            estimated_steps_per_epoch = 500  # Reasonable default for most datasets
            T_max = max_epochs * estimated_steps_per_epoch
            print(
                f"Warning: Using estimated T_max={T_max} (estimated_stepping_batches not available)"
            )
        else:
            T_max = estimated_steps
            print(f"Using calculated T_max={T_max} from estimated_stepping_batches")

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=self.cfg.training.lr * 0.01,
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def _convert_logits_to_predictions(self, logits_dict):
        """Convert model logits to probabilities/predictions for evaluation and logging."""
        predictions = {}

        # Convert awkward arrays to tensors if needed, apply activations, then convert back
        for key, logits in logits_dict.items():
            # Convert awkward array to tensor if necessary
            if hasattr(logits, "to_numpy"):  # awkward array
                logits_tensor = torch.from_numpy(ak.to_numpy(logits))
            else:  # already a tensor
                logits_tensor = logits

            # Apply appropriate activation
            if key in ["is_tau", "charge"]:  # Binary classification heads
                pred_tensor = torch.sigmoid(logits_tensor)
            elif key == "decay_mode":  # Multiclass classification
                pred_tensor = torch.softmax(logits_tensor, dim=-1)
            else:  # Regression (kinematics)
                pred_tensor = logits_tensor

            # Convert back to awkward array to match the expected format
            if hasattr(logits, "to_numpy"):  # Input was awkward array
                predictions[key] = ak.from_numpy(pred_tensor.detach().cpu().numpy())
            else:  # Input was tensor
                predictions[key] = pred_tensor

        return predictions

    def _calculate_baseline_charges(
        self, inputs
    ):  # TODO: Should maybe add as a separate "algorithm" under evaluation and merge with inference?
        """Calculate baseline jet charge using Q*kappa weighting."""
        # Extract candidate data
        cand_charges = inputs.cand_features[:, 7, :]  # Feature index 7 is charge
        cand_mask = inputs.cand_mask[:, 0, :]  # Remove singleton dimension

        # Calculate candidate pTs from px, py
        px = inputs.cand_kinematics_pxpypze[:, 0, :]
        py = inputs.cand_kinematics_pxpypze[:, 1, :]
        cand_pts = torch.sqrt(px**2 + py**2)

        # Get jet pTs - first convert dict to awkward array, then reinitialize p4
        try:
            # Convert dict to awkward array
            reco_jet_p4s_ak = ak.Array(inputs.reco_jet_p4s)
            reco_jet_p4s = g.reinitialize_p4(reco_jet_p4s_ak)

            pt_values = reco_jet_p4s.pt

            if hasattr(pt_values, "to_numpy"):
                pt_numpy = pt_values.to_numpy()
            else:
                pt_numpy = ak.to_numpy(pt_values)

            if pt_numpy.ndim == 0:
                pt_numpy = np.array([pt_numpy])
            elif pt_numpy.ndim > 1:
                pt_numpy = pt_numpy.flatten()[: len(cand_charges)]

            jet_pts = torch.tensor(
                pt_numpy, dtype=torch.float32, device=cand_charges.device
            )

            if len(jet_pts) != len(cand_charges):
                if len(jet_pts) == 1:
                    jet_pts = jet_pts.repeat(len(cand_charges))
                else:
                    jet_pts = jet_pts[: len(cand_charges)]

        except Exception as e:
            print(f"Warning: Error processing reco_jet_p4s.pt: {e}")
            jet_pts = torch.sum(cand_pts * cand_mask, dim=1)

        cand_charges_masked = cand_charges * cand_mask
        cand_pts_masked = cand_pts * cand_mask

        kappa = 0.2
        numer = torch.sum(cand_charges_masked * (cand_pts_masked**kappa), dim=1)
        denom = jet_pts**kappa
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)

        baseline_charges = numer / denom
        return baseline_charges.detach().cpu().numpy()

    def forward(self, batch):
        """Both `predictions` and `targets` are defined for the multiple heads"""
        # Unpack batch components
        inputs = BatchInputs(*batch)

        predictions = self.ParTau(
            cand_features=inputs.cand_features,
            cand_kinematics_pxpypze=inputs.cand_kinematics_pxpypze,
            cand_mask=inputs.cand_mask,
        )
        return predictions, inputs.target, inputs.weight

    def charge_loss_fn(self, predictions, targets):
        return self.charge_loss(predictions, targets)

    def tagging_loss_fn(self, predictions, targets):
        return self.tagging_loss(predictions, targets)

    def decay_mode_loss_fn(self, predictions, targets):
        return self.decay_mode_loss(predictions, targets)

    def kinematics_loss_fn(self, predictions, targets, l_m: float = 0.2):
        # Log-ratio terms: independent Huber in log space
        log_pt_loss = self.kinematics_loss(predictions[:, 0], targets[:, 0])  # log pT
        log_m_loss = self.kinematics_loss(predictions[:, 4], targets[:, 4])  # log mass

        # Delta eta term
        deta_loss = self.kinematics_loss(predictions[:, 1], targets[:, 1])  # delta eta

        # Angular terms using sin/cos representation
        sin_dphi_loss = self.kinematics_loss(
            predictions[:, 2], targets[:, 2]
        )  # sin(dphi)
        cos_dphi_loss = self.kinematics_loss(
            predictions[:, 3], targets[:, 3]
        )  # cos(dphi)

        # Average all components weighted by their importance
        return (
            log_pt_loss + deta_loss + sin_dphi_loss + cos_dphi_loss + l_m * log_m_loss
        ) / (
            4.0 + l_m
        )  # Normalize by sum of weights: 4 * 1.0 + l_m

    def calculate_metrics(self, targets, predictions, weights):
        is_tau_mask = targets["is_tau"].bool()

        # Per-jet losses — shape [N]
        tau_id_loss_per_jet = self.tagging_loss_fn(
            predictions["is_tau"], targets["is_tau"]
        )

        if not is_tau_mask.any():
            # Only tagging loss when no tau jets present
            # Apply uncertainty weighting: L = (1/(2*exp(log_var))) * loss + 1.0 * log_var
            log_var_tag_clamped = self.log_var_tagging.clamp(min=-4.0, max=4.0)
            precision = torch.exp(-log_var_tag_clamped)
            weighted_tagging_loss = (
                0.5 * precision * tau_id_loss_per_jet * weights
            ).mean() + 1.0 * log_var_tag_clamped

            return {
                "tau_id_loss": tau_id_loss_per_jet.mean(),
                "charge_loss": tau_id_loss_per_jet.new_zeros(()),
                "decay_mode_loss": tau_id_loss_per_jet.new_zeros(()),
                "kinematics_loss": tau_id_loss_per_jet.new_zeros(()),
                "tau_id_loss_weighted": weighted_tagging_loss.squeeze(),
                "charge_loss_weighted": tau_id_loss_per_jet.new_zeros(()),
                "decay_mode_loss_weighted": tau_id_loss_per_jet.new_zeros(()),
                "kinematics_loss_weighted": tau_id_loss_per_jet.new_zeros(()),
                "loss": weighted_tagging_loss.squeeze(),
            }

        # Per-jet losses for signal-only heads — shape [N_signal]
        dm_loss_per_jet = self.decay_mode_loss_fn(
            predictions["decay_mode"][is_tau_mask], targets["decay_mode"][is_tau_mask]
        )
        charge_loss_per_jet = self.charge_loss_fn(
            predictions["charge"][is_tau_mask], targets["charge"][is_tau_mask]
        )
        kin_loss_per_jet = self.kinematics_loss_fn(
            predictions["kinematics"][is_tau_mask], targets["kinematics"][is_tau_mask]
        )

        # Apply uncertainty-based weighting following Kendall et al.
        # L = (1/(2*exp(log_var))) * loss + 1.0 * log_var  (beta=1.0 to penalise task collapse)
        # Clamp log_var to prevent task collapse (network zeroing out hard tasks)
        # Tagging is learning well so allow wider range; signal tasks clamped tighter
        # to enforce minimum precision floor: exp(-1.5) ≈ 0.22 for signal tasks
        log_var_tagging = self.log_var_tagging.clamp(min=-4.0, max=4.0)
        log_var_charge = self.log_var_charge.clamp(min=-4.0, max=1.5)
        log_var_decay_mode = self.log_var_decay_mode.clamp(min=-4.0, max=1.5)
        log_var_kinematics = self.log_var_kinematics.clamp(min=-4.0, max=1.5)

        # Calculate precisions (inverse variances)
        precision_tagging = torch.exp(-log_var_tagging)
        precision_charge = torch.exp(-log_var_charge)
        precision_dm = torch.exp(-log_var_decay_mode)
        precision_kin = torch.exp(-log_var_kinematics)

        # Apply uncertainty weighting to tagging task (with classification weights)
        weighted_tagging_loss = (
            0.5 * precision_tagging * tau_id_loss_per_jet * weights
        ).mean() + 1.0 * log_var_tagging

        # Apply uncertainty weighting to signal tasks (no classification weights)
        weighted_charge_loss = (
            0.5 * precision_charge * charge_loss_per_jet
        ).mean() + 1.0 * log_var_charge
        weighted_dm_loss = (
            0.5 * precision_dm * dm_loss_per_jet
        ).mean() + 1.0 * log_var_decay_mode
        weighted_kin_loss = (
            0.5 * precision_kin * kin_loss_per_jet
        ).mean() + 1.0 * log_var_kinematics

        signal_losses = weighted_charge_loss + weighted_dm_loss + weighted_kin_loss
        loss = weighted_tagging_loss + signal_losses

        # No degradation correction: pure uncertainty weighting with clamped log_var

        return {
            "tau_id_loss": tau_id_loss_per_jet.mean(),
            "charge_loss": charge_loss_per_jet.mean(),
            "decay_mode_loss": dm_loss_per_jet.mean(),
            "kinematics_loss": kin_loss_per_jet.mean(),
            "tau_id_loss_weighted": weighted_tagging_loss.squeeze(),
            "charge_loss_weighted": weighted_charge_loss.squeeze(),
            "decay_mode_loss_weighted": weighted_dm_loss.squeeze(),
            "kinematics_loss_weighted": weighted_kin_loss.squeeze(),
            "loss": loss.squeeze(),
        }

    def validation_step(self, batch, _batch_idx):
        predictions, targets, weights = self.forward(batch)

        # Store inputs for baseline calculation at epoch end
        inputs = BatchInputs(*batch)

        metrics = self.calculate_metrics(
            targets=targets, predictions=predictions, weights=weights
        )

        output = {
            "predictions": predictions,
            "targets": targets,
            # "weights": weights,
            "inputs": inputs,  # Store inputs for baseline calculation and p4s extraction at epoch end
        }
        self.validation_outputs.append(output)
        for key, value in metrics.items():
            self.validation_loss_accumulator[key].append(value.detach())
        return metrics["loss"]

    def on_validation_epoch_start(self):
        """Initialize storage for validation outputs."""
        self.validation_outputs = []
        self.validation_loss_accumulator = {
            key: []
            for key in [
                "loss",
                "tau_id_loss",
                "charge_loss",
                "decay_mode_loss",
                "kinematics_loss",
                "tau_id_loss_weighted",
                "charge_loss_weighted",
                "decay_mode_loss_weighted",
                "kinematics_loss_weighted",
            ]
        }

    def _log_at_epoch_end(self, dataset: str):
        if dataset == "val" and self.trainer.sanity_checking:
            return

        dataset_outputs = (
            self.validation_outputs if dataset == "val" else self.training_outputs
        )

        if dataset_outputs:
            # Aggregate all predictions, targets, and weights
            all_predictions = {}
            all_targets = {}
            # all_weights = []
            all_gen_jet_p4s = {}
            all_gen_jet_tau_p4s = {}
            all_reco_jet_p4s = {}
            all_inputs = []  # Store all inputs for baseline calculation

            for output in dataset_outputs:
                # Concatenate predictions for each head
                for key, pred in output["predictions"].items():
                    if key not in all_predictions:
                        all_predictions[key] = []
                    all_predictions[key].append(pred.detach().cpu())

                # Concatenate targets
                for key, target in output["targets"].items():
                    if key not in all_targets:
                        all_targets[key] = []
                    all_targets[key].append(target.detach().cpu())

                # Store inputs for baseline calculation and p4s extraction
                inputs = output["inputs"]
                all_inputs.append(inputs)

                # Extract p4s from inputs
                for key, value in inputs.gen_jet_p4s.items():
                    if key not in all_gen_jet_p4s:
                        all_gen_jet_p4s[key] = []
                    all_gen_jet_p4s[key].append(ak.Array(value.detach().cpu()))

                for key, value in inputs.reco_jet_p4s.items():
                    if key not in all_reco_jet_p4s:
                        all_reco_jet_p4s[key] = []
                    all_reco_jet_p4s[key].append(ak.Array(value.detach().cpu()))

                for key, value in inputs.gen_jet_tau_p4s.items():
                    if key not in all_gen_jet_tau_p4s:
                        all_gen_jet_tau_p4s[key] = []
                    all_gen_jet_tau_p4s[key].append(ak.Array(value.detach().cpu()))

                # Concatenate weights
                # all_weights.append(output["weights"].detach().cpu())

            # Convert lists to tensors
            for key in all_predictions:
                all_predictions[key] = ak.concatenate(all_predictions[key], axis=0)
            for key in all_targets:
                all_targets[key] = ak.concatenate(all_targets[key], axis=0)

            # Convert logits to probabilities for logging (evaluation functions expect probabilities)
            all_predictions_for_logging = self._convert_logits_to_predictions(
                all_predictions
            )
            for key in all_gen_jet_p4s:
                all_gen_jet_p4s[key] = ak.concatenate(all_gen_jet_p4s[key], axis=0)
            for key in all_reco_jet_p4s:
                all_reco_jet_p4s[key] = ak.concatenate(all_reco_jet_p4s[key], axis=0)
            for key in all_gen_jet_tau_p4s:
                all_gen_jet_tau_p4s[key] = ak.concatenate(
                    all_gen_jet_tau_p4s[key], axis=0
                )

            # Convert dictionaries back to awkward arrays with fields for reinitialize_p4
            gen_jet_p4s = ak.Array(all_gen_jet_p4s)
            reco_jet_p4s = ak.Array(all_reco_jet_p4s)
            gen_jet_tau_p4s = ak.Array(all_gen_jet_tau_p4s)

            all_baseline_charges = []
            for inputs in all_inputs:
                baseline_charges = self._calculate_baseline_charges(inputs)
                all_baseline_charges.append(baseline_charges)
            all_baseline_charges = np.concatenate(all_baseline_charges, axis=0)

            # all_weights = ak.concatenate(all_weights, dim=0)

            # Log comprehensive metrics with full validation dataset
            current_epoch = self.current_epoch
            tb_logger = self.logger.experiment
            logger.log_all(
                targets=all_targets,
                gen_jet_p4s=gen_jet_p4s,
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                reco_jet_p4s=reco_jet_p4s,
                predictions=all_predictions_for_logging,  # Use probabilities for logging
                cfg=self.cfg,
                tb_logger=tb_logger,
                current_epoch=current_epoch,
                dataset=dataset,
                baseline_charges=all_baseline_charges,
            )

            # Clear outputs to free memory
            dataset_outputs.clear()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            epoch_metrics = {
                k: torch.stack(v).mean()
                for k, v in self.validation_loss_accumulator.items()
                if v
            }
            for k, v in epoch_metrics.items():
                self.log(f"val_losses/{k}", v)
        self._log_at_epoch_end(dataset="val")

    def on_train_epoch_start(self):
        self.training_loss_accumulator = {
            key: []
            for key in [
                "loss",
                "tau_id_loss",
                "charge_loss",
                "decay_mode_loss",
                "kinematics_loss",
                "tau_id_loss_weighted",
                "charge_loss_weighted",
                "decay_mode_loss_weighted",
                "kinematics_loss_weighted",
            ]
        }

    def on_train_epoch_end(self):
        epoch_metrics = {
            k: torch.stack(v).mean()
            for k, v in self.training_loss_accumulator.items()
            if v
        }
        for k, v in epoch_metrics.items():
            self.log(f"train_losses/{k}", v)
