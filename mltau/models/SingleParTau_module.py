import torch
import awkward as ak
import numpy as np
import torch.nn as nn
import lightning as L
from omegaconf import DictConfig

from mltau.tools.io.general import BatchInputs
from mltau.tools import general as g
from mltau.tools.losses import FocalLoss, TauLoss
from mltau.tools.logging import tagging, kinematics, decay_mode, charge_id
from mltau.models.SingleParTau import ParTau
from mltau.models.MixerTau import MixerTau

VALID_TASKS = {"is_tau", "charge", "decay_mode", "kinematics"}


class ParTauModule(L.LightningModule):
    def __init__(self, cfg: DictConfig, input_dim: int, num_dm_classes: int, task: str):
        super().__init__()
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got '{task}'")
        self.cfg = cfg
        self.task = task
        
        model_type = cfg.training.model.get("backbone", "ParT")
        if model_type == "Mixer":
            self.ParTau = MixerTau(
                input_dim=input_dim,
                task=task,
                n_constituents=cfg.dataset.get("max_cands", 20),
                num_dm_classes=num_dm_classes,
                embed_dim=cfg.training.model.get("embed_dim", 128),
            )
        else:
            self.ParTau = ParTau(
                input_dim=input_dim,
                task=task,
                num_dm_classes=num_dm_classes,
                num_layers=cfg.training.model.get("num_layers", 2),
                embed_dims=cfg.training.model.get("embed_dims", [256, 512, 256]),
                use_pre_activation_pair=False,
                for_inference=False,
                use_amp=False,
                metric="theta-phi",
            )
        self.tau_loss = TauLoss(l_m=0.2, label_smoothing=0.1)

    def _loss_key(self):
        task_name = "tau_id" if self.task == "is_tau" else self.task
        return f"{task_name}_loss"

    def _make_accumulator(self):
        # Original aggregate-only accumulator kept for reference.
        # return {key: [] for key in ["loss", self._loss_key()]}
        keys = ["loss", self._loss_key()]
        if self.task == "kinematics":
            keys.extend(
                [
                    "kinematics_log_pt_loss",
                    "kinematics_delta_eta_loss",
                    "kinematics_phi_chord_loss",
                    "kinematics_log_mass_loss",
                ]
            )
        return {key: [] for key in keys}

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
        return self.forward(batch)[0]

    def test_step(self, batch, _batch_idx):
        return self.forward(batch)[0]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.ParTau.parameters(), lr=self.cfg.training.lr, weight_decay=1e-2
        )

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

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=T_max,
        #     eta_min=self.cfg.training.lr * 0.01,
        # )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.training.lr,
            total_steps=T_max,
            anneal_strategy="cos",
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def forward(self, batch):
        inputs = BatchInputs(*batch)
        model_output = self.ParTau(
            cand_features=inputs.cand_features,
            cand_kinematics_pxpypze=inputs.cand_kinematics_pxpypze,
            cand_mask=inputs.cand_mask,
        )
        # Keep logging inputs stable while using task-specific tensors for the loss.
        if self.task == "charge":
            charge_logits = model_output[0]
            predictions = {
                self.task: torch.sigmoid(charge_logits),
                "charge_logits": charge_logits,
            }
        elif self.task == "decay_mode":
            decay_mode_logits = model_output[0]
            predictions = {
                self.task: torch.softmax(decay_mode_logits, dim=-1),
                "decay_mode_logits": decay_mode_logits,
            }
        elif self.task == "is_tau":
            tau_logits = model_output[0]
            predictions = {
                self.task: torch.softmax(tau_logits, dim=-1)[:, 1],
                "is_tau_logits": tau_logits,
            }
        else:
            predictions = {self.task: model_output[0]}
        return predictions, inputs.target, inputs.weight

    def calculate_metrics(self, targets, predictions, weights):
        pred = predictions[self.task]
        target = targets[self.task]
        is_tau_mask = targets["is_tau"].bool()

        if self.task == "kinematics":
            # Signal-only task: mask inputs
            if not is_tau_mask.any():
                return {
                    "loss": pred.new_zeros(()),
                    self._loss_key(): pred.new_zeros(()),
                    "kinematics_log_pt_loss": pred.new_zeros(()),
                    "kinematics_delta_eta_loss": pred.new_zeros(()),
                    "kinematics_phi_chord_loss": pred.new_zeros(()),
                    "kinematics_log_mass_loss": pred.new_zeros(()),
                }

            pred_tau = pred[is_tau_mask]
            tgt_tau = target[is_tau_mask]
            weights_tau = weights[is_tau_mask]

            loss, components = self.tau_loss.compute_kinematics_loss(
                pred_tau, tgt_tau, weights_tau
            )

            metrics = {
                "loss": loss,
                self._loss_key(): loss,
                "kinematics_log_pt_loss": components["log_pt"],
                "kinematics_delta_eta_loss": components["delta_eta"],
                "kinematics_phi_chord_loss": components["phi_chord"],
                "kinematics_log_mass_loss": components["log_mass"],
            }
            return metrics
        elif self.task == "is_tau":
            # Tagging is for all jets
            loss = self.tau_loss.compute_tagging_loss(
                predictions["is_tau_logits"], target, weights
            )
        elif self.task == "charge":
            # Signal-only task: mask inputs
            if not is_tau_mask.any():
                return {"loss": pred.new_zeros(()), self._loss_key(): pred.new_zeros(())}
            loss = self.tau_loss.compute_charge_loss(
                predictions["charge_logits"][is_tau_mask],
                target[is_tau_mask],
                weights[is_tau_mask],
            )
        else:  # "decay_mode" — only meaningful for signal taus
            if not is_tau_mask.any():
                return {"loss": pred.new_zeros(()), self._loss_key(): pred.new_zeros(())}
            # Apply weights to the signal-only loss
            loss = self.tau_loss.compute_decay_mode_loss(
                predictions["decay_mode_logits"][is_tau_mask],
                target[is_tau_mask],
                weights[is_tau_mask],
            )

        return {"loss": loss, self._loss_key(): loss}

    def validation_step(self, batch, _batch_idx):
        predictions, targets, weights = self.forward(batch)
        metrics = self.calculate_metrics(
            targets=targets, predictions=predictions, weights=weights
        )
        inputs = BatchInputs(*batch)
        self.validation_outputs.append(
            {
                "predictions": predictions,
                "targets": targets,
                "gen_jet_p4s": inputs.gen_jet_p4s,
                "reco_jet_p4s": inputs.reco_jet_p4s,
                "gen_jet_tau_p4s": inputs.gen_jet_tau_p4s,
                "inputs": inputs if self.task == "charge" else None,
            }
        )
        for key, value in metrics.items():
            self.validation_loss_accumulator[key].append(value.detach())
        return metrics["loss"]

    def on_validation_epoch_start(self):
        self.validation_outputs = []
        self.validation_loss_accumulator = self._make_accumulator()

    def _log_task_metrics(
        self,
        targets,
        predictions,
        gen_jet_p4s,
        gen_jet_tau_p4s,
        reco_jet_p4s,
        tb_logger,
        current_epoch,
        dataset,
    ):
        kwargs = dict(
            targets=targets,
            predictions=predictions,
            tb_logger=tb_logger,
            current_epoch=current_epoch,
        )
        if self.task == "is_tau":
            tagging.log_all_tagging_metrics(
                gen_jet_p4s=gen_jet_p4s,
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                reco_jet_p4s=reco_jet_p4s,
                cfg=self.cfg,
                dataset=dataset,
                **kwargs,
            )
        elif self.task == "charge":
            charge_id.log_charge_id_performance(
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                reco_jet_p4s=reco_jet_p4s,
                cfg=self.cfg,
                dataset=dataset,
                **kwargs,
            )
        elif self.task == "decay_mode":
            decay_mode.log_all_decay_mode_metrics(**kwargs)
        elif self.task == "kinematics":
            kinematics.log_all_kinematics_metrics(
                reco_jet_p4s=reco_jet_p4s,
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                cfg=self.cfg,
                dataset=dataset,
                **kwargs,
            )

    def _log_at_epoch_end(self, dataset: str):
        if dataset == "val" and self.trainer.sanity_checking:
            return

        dataset_outputs = self.validation_outputs if dataset == "val" else []

        if dataset_outputs:
            all_predictions = {}
            all_targets = {}
            all_gen_jet_p4s = {}
            all_gen_jet_tau_p4s = {}
            all_reco_jet_p4s = {}
            all_inputs = []

            for output in dataset_outputs:
                for key, pred in output["predictions"].items():
                    if key not in all_predictions:
                        all_predictions[key] = []
                    all_predictions[key].append(pred.detach().cpu())

                for key, target in output["targets"].items():
                    if key not in all_targets:
                        all_targets[key] = []
                    all_targets[key].append(target.detach().cpu())

                for key, value in output["gen_jet_p4s"].items():
                    if key not in all_gen_jet_p4s:
                        all_gen_jet_p4s[key] = []
                    all_gen_jet_p4s[key].append(ak.Array(value.detach().cpu()))

                for key, value in output["reco_jet_p4s"].items():
                    if key not in all_reco_jet_p4s:
                        all_reco_jet_p4s[key] = []
                    all_reco_jet_p4s[key].append(ak.Array(value.detach().cpu()))

                for key, value in output["gen_jet_tau_p4s"].items():
                    if key not in all_gen_jet_tau_p4s:
                        all_gen_jet_tau_p4s[key] = []
                    all_gen_jet_tau_p4s[key].append(ak.Array(value.detach().cpu()))

                if output.get("inputs") is not None:
                    all_inputs.append(output["inputs"])

            for key in all_predictions:
                all_predictions[key] = ak.concatenate(all_predictions[key], axis=0)
            for key in all_targets:
                all_targets[key] = ak.concatenate(all_targets[key], axis=0)
            for key in all_gen_jet_p4s:
                all_gen_jet_p4s[key] = ak.concatenate(all_gen_jet_p4s[key], axis=0)
            for key in all_reco_jet_p4s:
                all_reco_jet_p4s[key] = ak.concatenate(all_reco_jet_p4s[key], axis=0)
            for key in all_gen_jet_tau_p4s:
                all_gen_jet_tau_p4s[key] = ak.concatenate(
                    all_gen_jet_tau_p4s[key], axis=0
                )

            gen_jet_p4s = ak.Array(all_gen_jet_p4s)
            reco_jet_p4s = ak.Array(all_reco_jet_p4s)
            gen_jet_tau_p4s = ak.Array(all_gen_jet_tau_p4s)

            self._log_task_metrics(
                targets=all_targets,
                predictions=all_predictions,
                gen_jet_p4s=gen_jet_p4s,
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                reco_jet_p4s=reco_jet_p4s,
                tb_logger=self.logger.experiment,
                current_epoch=self.current_epoch,
                dataset=dataset,
            )

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
        self.training_loss_accumulator = self._make_accumulator()

    def on_train_epoch_end(self):
        epoch_metrics = {
            k: torch.stack(v).mean()
            for k, v in self.training_loss_accumulator.items()
            if v
        }
        for k, v in epoch_metrics.items():
            self.log(f"train_losses/{k}", v)
