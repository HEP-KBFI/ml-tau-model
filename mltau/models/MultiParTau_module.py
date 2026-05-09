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
        self.charge_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.tagging_loss = SigmoidFocalLoss(
            alpha=0.2, gamma=2.0, reduction="none"
        )  # class imbalance
        self.decay_mode_loss = nn.CrossEntropyLoss(reduction="none")
        self.kinematics_loss = nn.HuberLoss(reduction="none", delta=1.0)

        self.num_tasks = 4
        # Task order: [tagging, decay_mode, charge, kinematics]
        # Kinematics is the hardest regression task and benefits from a higher
        # initial weight so it competes with the classification heads early on.
        self.task_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 2.0]))
        # Store initial losses for GradNorm
        self.initial_losses = None
        # GradNorm hyperparameter: higher alpha → faster equalization of training rates
        self.gradnorm_alpha = 1.5
        # Only recompute gradient norms every N steps; task weights are slow-moving
        # so there is no accuracy benefit to running GradNorm every step.
        self.gradnorm_update_freq = 20
        # Disable automatic optimization so we can run two separate backward passes
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        net_opt, w_opt = self.optimizers()

        predictions, targets, sample_weights = self.forward(batch)

        # Per-task scalar losses with gradient graph: [tag, dm, charge, kin]
        task_losses = self._compute_per_task_losses(
            predictions, targets, sample_weights
        )

        # Initialise reference losses on the very first training step
        if self.initial_losses is None:
            self.initial_losses = task_losses.detach().clone()

        # Normalise task weights: non-negative, sum == num_tasks
        task_w = torch.relu(self.task_weights)
        task_w = task_w * (self.num_tasks / task_w.sum().clamp(min=1e-6))

        # Combined weighted task loss (network update — task_w detached so gradients
        # flow only through network params).
        combined_loss = (task_w.detach() * task_losses).sum()

        run_gradnorm = batch_idx % self.gradnorm_update_freq == 0

        if run_gradnorm:
            # ---------------------------------------------------------------- #
            # GradNorm: ||∇_W L_i|| measured on a SINGLE representative param  #
            # (fc2.weight of the last shared block) — avoids iterating over all #
            # block parameters while still capturing relative task magnitudes.  #
            # retain_graph=True on all calls so the graph survives for          #
            # combined_loss.backward() below.                                   #
            # ---------------------------------------------------------------- #
            rep_param = [self.ParTau.blocks[-1].fc2.weight]
            raw_grad_norms = []
            for i in range(self.num_tasks):
                (gj,) = torch.autograd.grad(
                    task_losses[i],
                    rep_param,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                if gj is not None:
                    norm = gj.float().norm().detach()
                else:
                    norm = task_losses.new_zeros(())
                raw_grad_norms.append(norm)
            raw_grad_norms = torch.stack(raw_grad_norms)  # [4], detached

            # Weighted norms G_i = w_i * ||∇_W L_i||  (differentiable through task_w)
            weighted_grad_norms = task_w * raw_grad_norms

            G_bar = weighted_grad_norms.detach().mean()
            loss_ratios = task_losses.detach() / (self.initial_losses + 1e-12)
            r_i = loss_ratios / (loss_ratios.mean() + 1e-12)
            G_targets = (G_bar * r_i.pow(self.gradnorm_alpha)).detach()

            gradnorm_loss = (weighted_grad_norms - G_targets).abs().sum()

        # ---- Update network weights (one backward pass) ----
        net_opt.zero_grad()
        self.manual_backward(combined_loss)
        torch.nn.utils.clip_grad_norm_(self.ParTau.parameters(), 1.0)
        net_opt.step()

        if run_gradnorm:
            # ---- Update task weights ----
            w_opt.zero_grad()
            self.manual_backward(gradnorm_loss)
            w_opt.step()

            # Renormalise task weights so they sum to num_tasks.
            # Floor at 0.1 so GradNorm can never fully suppress a task
            # (with weight → 0, AdamW weight decay would destroy that head).
            with torch.no_grad():
                self.task_weights.clamp_(min=0.1)
                s = self.task_weights.sum()
                if s > 1e-6:
                    self.task_weights.mul_(self.num_tasks / s)

        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        # Accumulate individual task losses for epoch-level logging
        with torch.no_grad():
            loss_dict = {
                "loss": combined_loss,
                "tau_id_loss": task_losses[0],
                "decay_mode_loss": task_losses[1],
                "charge_loss": task_losses[2],
                "kinematics_loss": task_losses[3],
            }
        for key, value in loss_dict.items():
            self.training_loss_accumulator[key].append(value.detach())

        # Step-level logs
        self.log(
            "LR",
            net_opt.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        (
            self.log(
                "train/gradnorm_loss",
                gradnorm_loss.detach(),
                on_step=True,
                on_epoch=False,
            )
            if run_gradnorm
            else None
        )
        for i, name in enumerate(["tagging", "decay_mode", "charge", "kinematics"]):
            self.log(
                f"task_weights/{name}", task_w[i].detach(), on_step=True, on_epoch=False
            )

        return combined_loss

    def predict_step(self, batch, _batch_idx):
        """
        Runs inference for a batch during prediction (trainer.predict()).
        Returns only the predictions dict (no losses/metrics).
        """
        # Unpack batch if needed (BatchInputs or tuple)
        if isinstance(batch, (list, tuple)):
            # Standard tuple from DataLoader
            logits, _, _ = self.forward(batch)
        else:
            # Already a BatchInputs or similar
            logits, _, _ = self.forward((batch,))
        predictions = self._convert_logits_to_predictions(logits)
        return predictions

    def test_step(self, batch, _batch_idx):
        return self.forward(batch)[0]

    def configure_optimizers(self):
        # AdamW is generally preferred for transformer architectures
        net_optimizer = torch.optim.AdamW(
            params=self.ParTau.parameters(),
            lr=self.cfg.training.lr,
        )
        # Separate Adam optimizer for GradNorm task weights (small, fixed LR)
        w_optimizer = torch.optim.Adam(
            params=[self.task_weights],
            lr=0.025,
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
            net_optimizer,
            T_max=T_max,
            eta_min=self.cfg.training.lr * 0.01,
        )
        return [net_optimizer, w_optimizer], [
            {"scheduler": lr_scheduler, "interval": "step"}
        ]

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

        # Phi chord loss: treat (sin, cos) as a 2D unit-vector difference so that
        # the gradient corrects the angular direction jointly rather than pushing
        # sin and cos independently.  The chord distance equals 2|sin(Δφ/2)|.
        phi_chord_loss = torch.sqrt(
            (predictions[:, 2] - targets[:, 2]) ** 2
            + (predictions[:, 3] - targets[:, 3]) ** 2
            + 1e-8
        )  # shape [N], differentiable everywhere

        # 3 independent components + mass, normalise by sum of weights
        return (log_pt_loss + deta_loss + phi_chord_loss + l_m * log_m_loss) / (
            3.0 + l_m
        )

    def _compute_per_task_losses(self, predictions, targets, sample_weights):
        """
        Return a [4] tensor [tag_loss, dm_loss, charge_loss, kin_loss] — each a
        sample-weighted scalar with a gradient graph attached (for GradNorm).

        The three signal-only tasks (dm, charge, kin) require at least one tau jet
        in the batch; if there are none they are returned as zero tensors so that
        GradNorm can still run without crashing (their gradient norm will be 0 and
        their target G_target will be pulled toward 0 as well).
        """
        is_tau_mask = targets["is_tau"].bool()

        # Tagging loss — all jets
        tag_per_jet = self.tagging_loss_fn(predictions["is_tau"], targets["is_tau"])
        tag_loss = (tag_per_jet * sample_weights).mean()

        if not is_tau_mask.any():
            zero = tag_loss.new_zeros(())
            return torch.stack([tag_loss, zero, zero, zero])

        tau_weights = sample_weights[is_tau_mask]

        dm_loss = (
            self.decay_mode_loss_fn(
                predictions["decay_mode"][is_tau_mask],
                targets["decay_mode"][is_tau_mask],
            )
            * tau_weights
        ).mean()

        charge_loss = (
            self.charge_loss_fn(
                predictions["charge"][is_tau_mask],
                targets["charge"][is_tau_mask],
            )
            * tau_weights
        ).mean()

        kin_loss = (
            self.kinematics_loss_fn(
                predictions["kinematics"][is_tau_mask],
                targets["kinematics"][is_tau_mask],
            )
            * tau_weights
        ).mean()

        return torch.stack([tag_loss, dm_loss, charge_loss, kin_loss])

    def calculate_metrics(
        self, targets, predictions, weights, w_kin=1, w_dm=1, w_tag=1, w_charge=1
    ):
        is_tau_mask = targets["is_tau"].bool()

        # Per-jet losses — shape [N]
        tau_id_loss_per_jet = self.tagging_loss_fn(
            predictions["is_tau"], targets["is_tau"]
        )

        # Start combined per-jet loss with tagging term
        combined_per_jet = w_tag * tau_id_loss_per_jet

        if not is_tau_mask.any():
            return {
                "tau_id_loss": tau_id_loss_per_jet.mean(),
                "charge_loss": combined_per_jet.new_zeros(()),
                "decay_mode_loss": combined_per_jet.new_zeros(()),
                "kinematics_loss": combined_per_jet.new_zeros(()),
                "loss": (combined_per_jet * weights).mean(),
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

        # loss = torch.stack(
        #     [
        #         tau_id_loss_per_jet,
        #         dm_loss_per_jet,
        #         charge_loss_per_jet,
        #         kin_loss_per_jet,
        #     ]
        # )

        # Add signal-only terms into combined per-jet loss
        combined_per_jet[is_tau_mask] += (
            w_dm * dm_loss_per_jet
            + w_charge * charge_loss_per_jet
            + w_kin * kin_loss_per_jet
        )

        # Multiply each jet's combined loss by its cls_weight, then average
        loss = (combined_per_jet * weights).mean()

        return {
            "tau_id_loss": tau_id_loss_per_jet.mean(),
            "charge_loss": charge_loss_per_jet.mean(),
            "decay_mode_loss": dm_loss_per_jet.mean(),
            "kinematics_loss": kin_loss_per_jet.mean(),
            "loss": loss,
        }

    def validation_step(self, batch, _batch_idx):
        predictions, targets, weights = self.forward(batch)

        inputs = BatchInputs(*batch)

        metrics = self.calculate_metrics(
            targets=targets, predictions=predictions, weights=weights
        )

        output = {
            "predictions": predictions,
            "targets": targets,
            # "weights": weights,
            "inputs": inputs,  # Store inputs for p4s extraction at epoch end
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
