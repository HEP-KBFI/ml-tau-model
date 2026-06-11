import torch
import numpy as np
import awkward as ak
import torch.nn as nn
import lightning as L
from omegaconf import DictConfig
from mltau.tools import general as g

# from mltau.tools.optimizers.lookahead import Lookahead
from mltau.tools.io.general import BatchInputs

from mltau.tools.logging import logger
from mltau.tools.losses import TauLoss
from mltau.models.MultiParTau import ParTau


class ParTauModule(L.LightningModule):
    def __init__(self, cfg: DictConfig, input_dim: int, num_dm_classes: int):
        super().__init__()
        self.cfg = cfg
        m_cfg = cfg.training.model
        self.ParTau = ParTau(
            input_dim=input_dim,
            num_dm_classes=num_dm_classes,
            num_layers=m_cfg.get("num_layers", 8),
            num_heads=m_cfg.get("num_heads", 8),
            num_cls_layers=m_cfg.get("num_cls_layers", 2),
            embed_dims=m_cfg.get("embed_dims", [256, 512, 256]),
            pair_embed_dims=m_cfg.get("pair_embed_dims", [64, 64, 64]),
            use_pre_activation_pair=False,
            for_inference=False,
            use_amp=False,
            metric="theta-phi",
        )

        # Unified loss module handles all task-specific functions and weighting logic.
        self.tau_loss = TauLoss(l_m=0.2, label_smoothing=0.1)

        self.num_tasks = 4
        # Disable automatic optimization so PCGrad can do per-task backward passes
        self.automatic_optimization = False

        # Task weight scheduler state
        tw = self.cfg.training.task_weights
        self.current_task_weights = {
            "tau_id": tw.tau_id,
            "decay_mode": tw.decay_mode,
            "charge": tw.charge,
            "kinematics": tw.kinematics,
        }

        self.scheduler_state = {}
        if self.cfg.training.task_weight_scheduler.enabled:
            for task in self.cfg.training.task_weight_scheduler.monitor_tasks:
                self.scheduler_state[task] = {
                    "best_loss": float("inf"),
                    "patience_counter": 0,
                }

    def training_step(self, batch, batch_idx):
        net_opt = self.optimizers()

        predictions, targets, sample_weights = self.forward(batch)

        # Per-task scalar losses: [tag, dm, charge, kin]
        task_losses, kin_components = self._compute_per_task_losses(
            predictions, targets, sample_weights
        )

        # ------------------------------------------------------------------ #
        # PCGrad: for each task i, compute its gradient then subtract the     #
        # projection onto any task j whose gradient conflicts (dot < 0).      #
        # ------------------------------------------------------------------ #
        params = [p for p in self.ParTau.parameters() if p.requires_grad]

        tw = self.current_task_weights
        task_weight_tensor = task_losses.new_tensor(
            [tw["tau_id"], tw["decay_mode"], tw["charge"], tw["kinematics"]]
        )
        weighted_task_losses = task_losses * task_weight_tensor

        # Log current task weights
        for task_name, weight in tw.items():
            self.log(f"task_weights/{task_name}", weight, on_step=True, on_epoch=False)

        # 1. Collect per-task gradient vectors.
        task_grads = []
        for i in range(self.num_tasks):
            net_opt.zero_grad()
            self.manual_backward(
                weighted_task_losses[i], retain_graph=(i < self.num_tasks - 1)
            )
            grads = [
                (
                    p.grad.float().clone()
                    if p.grad is not None
                    else torch.zeros_like(p, dtype=torch.float32)
                )
                for p in params
            ]
            task_grads.append(grads)

        # 2. Project conflicting gradients (asymmetric: kinematics gradient is never
        # projected away from other tasks).
        KIN_IDX = self.num_tasks - 1  # kinematics is the last task
        pc_grads = [list(g) for g in task_grads]  # mutable copy
        for i in range(self.num_tasks):
            if i == KIN_IDX:  # never reduce the kinematics gradient
                continue
            for j in range(self.num_tasks):
                if i == j:
                    continue
                for k, (gi, gj) in enumerate(zip(pc_grads[i], task_grads[j])):
                    dot = (gi * gj).sum()
                    if dot < 0:  # conflict: remove component along gj
                        pc_grads[i][k] = gi - dot / (gj.norm().pow(2) + 1e-12) * gj

        # 3. Sum projected gradients and write into .grad
        for k, p in enumerate(params):
            merged = sum(pc_grads[i][k] for i in range(self.num_tasks))
            p.grad = merged.to(p.dtype)

        # 4. Clip -> step.
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        net_opt.step()

        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        combined_loss = task_losses.sum().detach()

        # Accumulate individual task losses for epoch-level logging
        with torch.no_grad():
            loss_dict = {
                "loss": combined_loss,
                "tau_id_loss": task_losses[0],
                "decay_mode_loss": task_losses[1],
                "charge_loss": task_losses[2],
                "kinematics_loss": task_losses[3],
            }
            for k, v in kin_components.items():
                loss_dict[f"kinematics_{k}_loss"] = v

        for key, value in loss_dict.items():
            self.training_loss_accumulator[key].append(value.detach())

        self.log(
            "LR",
            net_opt.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
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
        # AdamW is generally preferred for transformer architectures.
        base_lr = self.cfg.training.lr
        net_optimizer = torch.optim.AdamW(self.ParTau.parameters(), lr=base_lr, weight_decay=1e-2)

        # Check if estimated_stepping_batches is available and valid
        estimated_steps = getattr(self.trainer, "estimated_stepping_batches", None)

        if estimated_steps is None or estimated_steps <= 0:
            # Fallback: calculate based on config (will be approximate but functional)
            max_epochs = self.cfg.training.trainer.max_epochs
            estimated_steps_per_epoch = 500  # Reasonable default for most datasets
            T_max = max_epochs * estimated_steps_per_epoch
            print(
                f"Warning: Using estimated T_max={T_max} (estimated_stepping_batches not available)"
            )
        else:
            T_max = estimated_steps
            print(f"Using calculated T_max={T_max} from estimated_stepping_batches")

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            net_optimizer,
            max_lr=base_lr,
            total_steps=T_max,
            anneal_strategy="cos",
        )
        return [net_optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

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
            if key == "is_tau":  # Two-class head; return signal probability (class 1)
                pred_tensor = torch.softmax(logits_tensor, dim=-1)[:, 1]
            elif key == "charge":  # Binary classification head
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

    def _compute_per_task_losses(self, predictions, targets, sample_weights):
        """
        Return (task_losses, kin_components).
        task_losses: [4] tensor [tag_loss, dm_loss, charge_loss, kin_loss]
        """
        return self.tau_loss.compute_multi_task_losses(
            predictions, targets, sample_weights
        )

    def calculate_metrics(
        self, targets, predictions, weights, w_kin=1, w_dm=1, w_tag=1, w_charge=1
    ):
        task_losses, kin_components = self._compute_per_task_losses(
            predictions, targets, weights
        )

        # Aggregation: Sum of Averages (consistent with training and SingleParTau)
        combined_loss = task_losses.sum()

        metrics = {
            "loss": combined_loss,
            "tau_id_loss": task_losses[0],
            "decay_mode_loss": task_losses[1],
            "charge_loss": task_losses[2],
            "kinematics_loss": task_losses[3],
        }
        for k, v in kin_components.items():
            metrics[f"kinematics_{k}_loss"] = v

        return metrics

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
        keys = ["loss", "tau_id_loss", "charge_loss", "decay_mode_loss", "kinematics_loss"]
        keys.extend(
            [
                "kinematics_log_pt_loss",
                "kinematics_delta_eta_loss",
                "kinematics_phi_chord_loss",
                "kinematics_log_mass_loss",
            ]
        )
        self.validation_loss_accumulator = {key: [] for key in keys}

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

            # --- Task Weight Scheduler Logic ---
            sch_cfg = self.cfg.training.task_weight_scheduler
            if sch_cfg.enabled:
                for task in sch_cfg.monitor_tasks:
                    loss_key = f"{task}_loss"
                    if loss_key in epoch_metrics:
                        current_loss = epoch_metrics[loss_key].item()
                        state = self.scheduler_state[task]

                        if current_loss < state["best_loss"]:
                            state["best_loss"] = current_loss
                            state["patience_counter"] = 0
                        else:
                            state["patience_counter"] += 1

                        if state["patience_counter"] >= sch_cfg.patience:
                            old_weight = self.current_task_weights[task]
                            new_weight = max(
                                sch_cfg.min_weight, old_weight * sch_cfg.factor
                            )
                            if new_weight < old_weight:
                                self.current_task_weights[task] = new_weight
                                print(
                                    f"[Scheduler] Reducing {task} weight: "
                                    f"{old_weight:.4f} -> {new_weight:.4f}"
                                )
                            # Reset counter after reduction (standard ReduceLROnPlateau behavior)
                            state["patience_counter"] = 0

        self._log_at_epoch_end(dataset="val")

    def on_train_epoch_start(self):
        keys = ["loss", "tau_id_loss", "charge_loss", "decay_mode_loss", "kinematics_loss"]
        keys.extend(
            [
                "kinematics_log_pt_loss",
                "kinematics_delta_eta_loss",
                "kinematics_phi_chord_loss",
                "kinematics_log_mass_loss",
            ]
        )
        self.training_loss_accumulator = {key: [] for key in keys}

    def on_train_epoch_end(self):
        epoch_metrics = {
            k: torch.stack(v).mean()
            for k, v in self.training_loss_accumulator.items()
            if v
        }
        for k, v in epoch_metrics.items():
            self.log(f"train_losses/{k}", v)

