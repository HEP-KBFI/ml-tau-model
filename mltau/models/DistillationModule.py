import torch
import torch.nn as nn
import lightning as L
from omegaconf import DictConfig

from mltau.tools.io.general import BatchInputs
from mltau.tools.losses import TauLoss
from mltau.models.SingleParTau import ParTau as TeacherParT
from mltau.models.MixerTau import MixerTau as StudentMixer

from mltau.tools.logging import tagging, kinematics, decay_mode, charge_id
import awkward as ak

class DistillationModule(L.LightningModule):
    """
    LightningModule for Knowledge Distillation from ParT to MLP-Mixer.
    Performs Feature Distillation (MSE on embeddings) and Logit Distillation.
    """
    def __init__(
        self, 
        cfg: DictConfig, 
        teacher_checkpoint: str, 
        input_dim: int, 
        num_dm_classes: int, 
        task: str,
        distill_alpha: float = 0.5, # Weight for distillation loss
        temperature: float = 2.0
    ):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.distill_alpha = distill_alpha
        self.temperature = temperature
        
        # 1. Initialize Student (Mixer)
        self.student = StudentMixer(
            input_dim=input_dim,
            task=task,
            n_constituents=cfg.dataset.get("max_cands", 20),
            num_dm_classes=num_dm_classes,
            embed_dim=cfg.training.model.get("embed_dim", 128),
        )
        
        # 2. Initialize and Load Teacher (ParT)
        # We assume the teacher was trained with the same input_dim and task
        self.teacher = TeacherParT(
            input_dim=input_dim,
            task=task,
            num_dm_classes=num_dm_classes,
            num_layers=cfg.training.model.get("num_layers", 2), # Or load from teacher config
            embed_dims=cfg.training.model.get("embed_dims", [256, 512, 256]),
            metric="theta-phi",
        )
        print(f"Loading teacher from {teacher_checkpoint}")
        # Teacher checkpoint is likely a Lightning checkpoint, extract state_dict
        checkpoint = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        # Remove "ParTau." prefix from state_dict keys if it exists
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("ParTau."):
                new_state_dict[k[len("ParTau."):]] = v
            else:
                new_state_dict[k] = v
        self.teacher.load_state_dict(new_state_dict)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # 3. Projection layer to match Student (128) to Teacher (256) embedding size
        student_embed_dim = cfg.training.model.get("embed_dim", 128)
        teacher_embed_dim = cfg.training.model.get("embed_dims", [256, 512, 256])[-1]
        self.projection = nn.Linear(student_embed_dim, teacher_embed_dim)
        
        self.tau_loss = TauLoss(l_m=0.2, label_smoothing=0.1)
        self.mse_loss = nn.MSELoss()
        
    def _loss_key(self):
        task_name = "tau_id" if self.task == "is_tau" else self.task
        return f"{task_name}_loss"

    def _make_accumulator(self):
        keys = ["loss", "task_loss", "distill_loss", self._loss_key()]
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

    def on_validation_epoch_start(self):
        self.validation_outputs = []
        self.validation_loss_accumulator = self._make_accumulator()

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

    def forward(self, batch):
        inputs = BatchInputs(*batch)
        
        # Student forward
        student_output, student_embed = self.student(
            cand_features=inputs.cand_features,
            cand_kinematics_pxpypze=inputs.cand_kinematics_pxpypze,
            cand_mask=inputs.cand_mask,
            return_embedding=True
        )
        
        # Teacher forward (in eval mode)
        with torch.no_grad():
            teacher_output, teacher_embed = self.teacher(
                cand_features=inputs.cand_features,
                cand_kinematics_pxpypze=inputs.cand_kinematics_pxpypze,
                cand_mask=inputs.cand_mask,
                return_embedding=True
            )
            
        return student_output, teacher_output, student_embed, teacher_embed, inputs

    def training_step(self, batch, batch_idx):
        student_output, teacher_output, student_embed, teacher_embed, inputs = self.forward(batch)
        
        # 1. Task Loss (Student vs Ground Truth)
        predictions = self._get_predictions_dict(student_output)
        task_metrics = self._calculate_task_metrics(inputs.target, predictions, inputs.weight)
        task_loss = task_metrics["loss"]
        
        # 2. Distillation Loss (Embedding MSE)
        projected_student_embed = self.projection(student_embed)
        distill_loss = self.mse_loss(projected_student_embed, teacher_embed)
        
        # 3. Combined Loss
        total_loss = (1 - self.distill_alpha) * task_loss + self.distill_alpha * distill_loss
        
        self.training_loss_accumulator["loss"].append(total_loss.detach())
        self.training_loss_accumulator["task_loss"].append(task_loss.detach())
        self.training_loss_accumulator["distill_loss"].append(distill_loss.detach())
        for key, value in task_metrics.items():
            if key != "loss":
                self.training_loss_accumulator[key].append(value.detach())
                
        self.log("LR", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=True)
        
        return total_loss

    def _get_predictions_dict(self, model_output):
        if self.task == "charge":
            charge_logits = model_output[0]
            return {self.task: torch.sigmoid(charge_logits), "charge_logits": charge_logits}
        elif self.task == "decay_mode":
            decay_mode_logits = model_output[0]
            return {self.task: torch.softmax(decay_mode_logits, dim=-1), "decay_mode_logits": decay_mode_logits}
        elif self.task == "is_tau":
            tau_logits = model_output[0]
            return {self.task: torch.softmax(tau_logits, dim=-1)[:, 1], "is_tau_logits": tau_logits}
        else:
            return {self.task: model_output[0]}

    def _calculate_task_metrics(self, targets, predictions, weights):
        pred = predictions[self.task]
        target = targets[self.task]
        is_tau_mask = targets["is_tau"].bool()

        if self.task == "kinematics":
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
            loss = self.tau_loss.compute_tagging_loss(
                predictions["is_tau_logits"], target, weights
            )
        elif self.task == "charge":
            if not is_tau_mask.any():
                return {"loss": pred.new_zeros(()), self._loss_key(): pred.new_zeros(())}
            loss = self.tau_loss.compute_charge_loss(
                predictions["charge_logits"][is_tau_mask],
                target[is_tau_mask],
                weights[is_tau_mask],
            )
        else:  # "decay_mode"
            if not is_tau_mask.any():
                return {"loss": pred.new_zeros(()), self._loss_key(): pred.new_zeros(())}
            loss = self.tau_loss.compute_decay_mode_loss(
                predictions["decay_mode_logits"][is_tau_mask],
                target[is_tau_mask],
                weights[is_tau_mask],
            )

        return {"loss": loss, self._loss_key(): loss}

    def validation_step(self, batch, batch_idx):
        student_output, teacher_output, student_embed, teacher_embed, inputs = self.forward(batch)
        predictions = self._get_predictions_dict(student_output)
        task_metrics = self._calculate_task_metrics(inputs.target, predictions, inputs.weight)
        task_loss = task_metrics["loss"]
        
        projected_student_embed = self.projection(student_embed)
        distill_loss = self.mse_loss(projected_student_embed, teacher_embed)
        
        total_loss = (1 - self.distill_alpha) * task_loss + self.distill_alpha * distill_loss
        
        self.validation_loss_accumulator["loss"].append(total_loss.detach())
        self.validation_loss_accumulator["task_loss"].append(task_loss.detach())
        self.validation_loss_accumulator["distill_loss"].append(distill_loss.detach())
        for key, value in task_metrics.items():
            if key != "loss":
                self.validation_loss_accumulator[key].append(value.detach())
        
        self.validation_outputs.append(
            {
                "predictions": predictions,
                "targets": inputs.target,
                "gen_jet_p4s": inputs.gen_jet_p4s,
                "reco_jet_p4s": inputs.reco_jet_p4s,
                "gen_jet_tau_p4s": inputs.gen_jet_tau_p4s,
                "inputs": inputs if self.task == "charge" else None,
            }
        )
        return total_loss

    def _log_at_epoch_end(self, dataset: str):
        dataset_outputs = self.validation_outputs if dataset == "val" else []

        if dataset_outputs:
            all_predictions = {}
            all_targets = {}
            all_gen_jet_p4s = {}
            all_gen_jet_tau_p4s = {}
            all_reco_jet_p4s = {}

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

            for key in all_predictions:
                all_predictions[key] = ak.concatenate(all_predictions[key], axis=0)
            for key in all_targets:
                all_targets[key] = ak.concatenate(all_targets[key], axis=0)
            for key in all_gen_jet_p4s:
                all_gen_jet_p4s[key] = ak.concatenate(all_gen_jet_p4s[key], axis=0)
            for key in all_reco_jet_p4s:
                all_reco_jet_p4s[key] = ak.concatenate(all_reco_jet_p4s[key], axis=0)
            for key in all_gen_jet_tau_p4s:
                all_gen_jet_tau_p4s[key] = ak.concatenate(all_gen_jet_tau_p4s[key], axis=0)

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

    def _log_task_metrics(self, targets, predictions, gen_jet_p4s, gen_jet_tau_p4s, reco_jet_p4s, tb_logger, current_epoch, dataset):
        kwargs = dict(targets=targets, predictions=predictions, tb_logger=tb_logger, current_epoch=current_epoch)
        if self.task == "is_tau":
            tagging.log_all_tagging_metrics(gen_jet_p4s=gen_jet_p4s, gen_jet_tau_p4s=gen_jet_tau_p4s, reco_jet_p4s=reco_jet_p4s, cfg=self.cfg, dataset=dataset, **kwargs)
        elif self.task == "charge":
            charge_id.log_charge_id_performance(gen_jet_tau_p4s=gen_jet_tau_p4s, reco_jet_p4s=reco_jet_p4s, cfg=self.cfg, dataset=dataset, **kwargs)
        elif self.task == "decay_mode":
            decay_mode.log_all_decay_mode_metrics(**kwargs)
        elif self.task == "kinematics":
            kinematics.log_all_kinematics_metrics(reco_jet_p4s=reco_jet_p4s, gen_jet_tau_p4s=gen_jet_tau_p4s, cfg=self.cfg, dataset=dataset, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.student.parameters()) + list(self.projection.parameters()), 
            lr=self.cfg.training.lr, 
            weight_decay=1e-2
        )
        
        estimated_steps = getattr(self.trainer, "estimated_stepping_batches", 500 * self.cfg.training.trainer.max_epochs)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.training.lr,
            total_steps=estimated_steps,
            anneal_strategy="cos",
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
