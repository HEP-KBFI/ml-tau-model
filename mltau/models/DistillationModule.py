import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig

from mltau.tools.io.general import BatchInputs
from mltau.tools.io import scaling
from mltau.tools.losses import TauLoss
from mltau.models.SingleParTau import ParTau as TeacherParT
from mltau.models.MixerTau import MixerTau as StudentMixer

from mltau.tools.logging import tagging, kinematics, decay_mode, charge_id
import awkward as ak


class QuerySoftDistillation(nn.Module):
    """Use student tokens as queries into teacher-token relations."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.query = nn.Sequential(
            nn.LayerNorm(student_dim), nn.Linear(student_dim, teacher_dim)
        )
        self.value = nn.Sequential(
            nn.LayerNorm(student_dim), nn.Linear(student_dim, teacher_dim)
        )
        self.scale = teacher_dim**-0.5

    def forward(
        self,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = self.query(student_tokens)
        values = self.value(student_tokens)
        attention = torch.matmul(queries, teacher_tokens.transpose(-2, -1))
        attention = attention * self.scale
        if token_mask is not None:
            attention = attention.masked_fill(
                ~token_mask[:, None, :], torch.finfo(attention.dtype).min
            )
        attention = attention.softmax(dim=-1)
        return torch.matmul(attention, values)


class DistillationModule(L.LightningModule):
    """
    LightningModule for Knowledge Distillation from ParT to MLP-Mixer.

    The schedule first learns the student representation without task labels,
    then warms up the task head, and finally fine-tunes backbone and head with
    separate learning rates.
    """
    def __init__(
        self, 
        cfg: DictConfig, 
        teacher_checkpoint: str, 
        input_dim: int, 
        num_dm_classes: int, 
        task: str,
        distill_alpha: float = 0.5,
        temperature: float = 2.0,
        representation_epochs: int = 20,
        head_warmup_epochs: int = 5,
        token_loss_weight: float = 1.0,
        global_loss_weight: float = 1.0,
        masked_token_loss_weight: float = 1.0,
        mask_probability: float = 0.3,
        output_distill_loss_weight: float = 1.0,
        adaptation_feature_loss_weight: float = 0.1,
        representation_lr: float = 1e-3,
        head_warmup_lr: float = 1e-4,
        joint_head_lr: float = 3e-4,
        joint_backbone_lr: float = 2e-5,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.cfg = cfg
        self.task = task
        self.distill_alpha = distill_alpha
        self.temperature = temperature
        self.representation_epochs = representation_epochs
        self.head_warmup_epochs = head_warmup_epochs
        self.token_loss_weight = token_loss_weight
        self.global_loss_weight = global_loss_weight
        self.masked_token_loss_weight = masked_token_loss_weight
        self.mask_probability = mask_probability
        self.output_distill_loss_weight = output_distill_loss_weight
        self.adaptation_feature_loss_weight = adaptation_feature_loss_weight
        self.representation_lr = representation_lr
        self.head_warmup_lr = head_warmup_lr
        self.joint_head_lr = joint_head_lr
        self.joint_backbone_lr = joint_backbone_lr
        self.input_dim = input_dim
        
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
            num_layers=cfg.training.model.get("num_layers", 2),
            num_heads=cfg.training.model.get("num_heads", 8),
            num_cls_layers=cfg.training.model.get("num_cls_layers", 2),
            embed_dims=cfg.training.model.get("embed_dims", [256, 512, 256]),
            pair_embed_dims=cfg.training.model.get(
                "pair_embed_dims", [64, 64, 64]
            ),
            use_pre_activation_pair=False,
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
        self.global_projection = nn.Sequential(
            nn.LayerNorm(student_embed_dim),
            nn.Linear(student_embed_dim, teacher_embed_dim),
        )
        # Mixer constituent tokens are the 16-dimensional stage-2 activations.
        self.token_projection = QuerySoftDistillation(16, teacher_embed_dim)
        
        self.tau_loss = TauLoss(l_m=0.2, label_smoothing=0.1)
        self.mse_loss = nn.MSELoss()
        self.register_buffer("_input_scaler_mean", torch.zeros(input_dim))
        self.register_buffer("_input_scaler_std", torch.ones(input_dim))
        self.register_buffer(
            "_input_scaler_mask", torch.zeros(input_dim, dtype=torch.bool)
        )

    @property
    def in_representation_stage(self):
        return self.current_epoch < self.representation_epochs

    @property
    def in_head_warmup_stage(self):
        return (
            self.representation_epochs
            <= self.current_epoch
            < self.representation_epochs + self.head_warmup_epochs
        )

    @property
    def training_stage(self):
        if self.in_representation_stage:
            return "representation"
        if self.in_head_warmup_stage:
            return "head_warmup"
        return "joint"

    def on_fit_start(self):
        """Load scaler statistics after the data module has fitted them."""
        if not scaling.input_scaling_enabled(self.cfg):
            return

        scaler = scaling.load_saved_scaler(self.cfg)
        indices = torch.as_tensor(
            scaler["feature_indices"], dtype=torch.long, device=self.device
        )
        self._input_scaler_mean.zero_()
        self._input_scaler_std.fill_(1)
        self._input_scaler_mask.zero_()
        self._input_scaler_mean[indices] = torch.as_tensor(
            scaler["mean"], dtype=self._input_scaler_mean.dtype, device=self.device
        )
        self._input_scaler_std[indices] = torch.as_tensor(
            scaler["std"], dtype=self._input_scaler_std.dtype, device=self.device
        )
        self._input_scaler_mask[indices] = True

    def _teacher_features(self, student_features, cand_mask):
        """Undo data-loader scaling because the frozen teacher expects raw inputs."""
        if not scaling.input_scaling_enabled(self.cfg):
            return student_features

        mean = self._input_scaler_mean.view(1, -1, 1)
        std = self._input_scaler_std.view(1, -1, 1)
        scaled_channels = self._input_scaler_mask.view(1, -1, 1)
        teacher_features = torch.where(
            scaled_channels,
            student_features * std + mean,
            student_features,
        )
        return teacher_features * cand_mask.to(dtype=teacher_features.dtype)

    def _set_stage_trainability(self):
        representation_stage = self.in_representation_stage
        joint_stage = self.training_stage == "joint"
        for parameter in self.student.backbone.parameters():
            parameter.requires_grad = representation_stage or joint_stage
        for parameter in self.global_projection.parameters():
            parameter.requires_grad = representation_stage or joint_stage
        for parameter in self.token_projection.parameters():
            parameter.requires_grad = representation_stage or joint_stage

        head = self._student_head()
        for parameter in head.parameters():
            parameter.requires_grad = not representation_stage

    def _student_head(self):
        if self.task in {"decay_mode"}:
            return self.student.classification_head
        if self.task == "kinematics":
            return self.student.regression_head
        return self.student.binary_head
        
    def _loss_key(self):
        task_name = "tau_id" if self.task == "is_tau" else self.task
        return f"{task_name}_loss"

    def _make_accumulator(self):
        keys = [
            "loss",
            "task_loss",
            "distill_loss",
            "global_distill_loss",
            "token_distill_loss",
            "masked_token_distill_loss",
            "output_distill_loss",
            self._loss_key(),
        ]
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
        self.teacher.eval()
        self._set_stage_trainability()
        self.training_loss_accumulator = self._make_accumulator()
        self.log(
            "training_stage",
            {"representation": 0, "head_warmup": 1, "joint": 2}[
                self.training_stage
            ],
        )

    def on_train_epoch_end(self):
        epoch_metrics = {
            k: torch.stack(v).mean()
            for k, v in self.training_loss_accumulator.items()
            if v
        }
        for k, v in epoch_metrics.items():
            self.log(f"train_losses/{k}", v)
        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        stage_index = {"representation": 0, "head_warmup": 1, "joint": 2}[
            self.training_stage
        ]
        schedulers[stage_index].step()

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
            checkpoint_loss = epoch_metrics["task_loss"]
            if self.in_representation_stage:
                checkpoint_loss = checkpoint_loss + 1_000.0
            self.log("val_losses/checkpoint_loss", checkpoint_loss)
        self._log_at_epoch_end(dataset="val")

    def forward(self, batch):
        inputs = BatchInputs(*batch)
        
        # Student forward
        student_output, student_embed, student_tokens = self.student(
            cand_features=inputs.cand_features,
            cand_kinematics_pxpypze=inputs.cand_kinematics_pxpypze,
            cand_mask=inputs.cand_mask,
            return_tokens=True,
        )
        masked_student_tokens = None
        if self.in_representation_stage and self.mask_probability > 0:
            valid_tokens = inputs.cand_mask.squeeze(1).bool()
            masked_positions = (
                torch.rand_like(valid_tokens, dtype=torch.float32)
                < self.mask_probability
            ) & valid_tokens
            masked_features = inputs.cand_features.masked_fill(
                masked_positions.unsqueeze(1), 0
            )
            _, _, masked_student_tokens = self.student(
                cand_features=masked_features,
                cand_kinematics_pxpypze=inputs.cand_kinematics_pxpypze,
                cand_mask=inputs.cand_mask,
                return_tokens=True,
            )
        
        # Teacher forward (in eval mode)
        with torch.no_grad():
            teacher_features = self._teacher_features(
                inputs.cand_features, inputs.cand_mask
            )
            teacher_output, teacher_embed, teacher_tokens = self.teacher(
                cand_features=teacher_features,
                cand_kinematics_pxpypze=inputs.cand_kinematics_pxpypze,
                cand_mask=inputs.cand_mask,
                return_tokens=True,
            )
            
        return (
            student_output,
            teacher_output,
            student_embed,
            teacher_embed,
            student_tokens,
            masked_student_tokens,
            teacher_tokens,
            inputs,
        )

    def _token_distillation_loss(
        self, student_tokens, teacher_tokens, valid_tokens
    ):
        projected_tokens = self.token_projection(
            student_tokens, teacher_tokens, valid_tokens
        )
        if not valid_tokens.any():
            return projected_tokens.new_zeros(())
        projected_tokens = F.layer_norm(
            projected_tokens, projected_tokens.shape[-1:]
        )
        normalized_teacher_tokens = F.layer_norm(
            teacher_tokens, teacher_tokens.shape[-1:]
        )
        return F.mse_loss(
            projected_tokens[valid_tokens],
            normalized_teacher_tokens[valid_tokens],
        )

    def _distillation_losses(
        self,
        student_embed,
        teacher_embed,
        student_tokens,
        masked_student_tokens,
        teacher_tokens,
        cand_mask,
    ):
        projected_global = self.global_projection(student_embed)
        global_loss = self.mse_loss(
            F.layer_norm(projected_global, projected_global.shape[-1:]),
            F.layer_norm(teacher_embed, teacher_embed.shape[-1:]),
        )

        valid_tokens = cand_mask.squeeze(1).bool()
        token_loss = self._token_distillation_loss(
            student_tokens, teacher_tokens, valid_tokens
        )
        if masked_student_tokens is not None:
            masked_token_loss = self._token_distillation_loss(
                masked_student_tokens, teacher_tokens, valid_tokens
            )
        else:
            masked_token_loss = global_loss.new_zeros(())

        distill_loss = (
            self.global_loss_weight * global_loss
            + self.token_loss_weight * token_loss
            + self.masked_token_loss_weight * masked_token_loss
        )
        return distill_loss, global_loss, token_loss, masked_token_loss

    def _output_distillation_loss(
        self, student_output, teacher_output, targets, weights
    ):
        if self.task != "kinematics":
            return student_output[0].new_zeros(())

        is_tau_mask = targets["is_tau"].bool()
        if not is_tau_mask.any():
            return student_output[0].new_zeros(())

        loss, _ = self.tau_loss.compute_kinematics_loss(
            student_output[0][is_tau_mask],
            teacher_output[0][is_tau_mask],
            weights[is_tau_mask],
        )
        return loss

    def _adaptation_loss(self, task_loss, output_distill_loss, distill_loss):
        return (
            task_loss
            + self.output_distill_loss_weight * output_distill_loss
            + self.adaptation_feature_loss_weight * distill_loss
        )

    def training_step(self, batch, batch_idx):
        (
            student_output,
            teacher_output,
            student_embed,
            teacher_embed,
            student_tokens,
            masked_student_tokens,
            teacher_tokens,
            inputs,
        ) = self.forward(batch)
        
        # 1. Task Loss (Student vs Ground Truth)
        predictions = self._get_predictions_dict(student_output)
        task_metrics = self._calculate_task_metrics(inputs.target, predictions, inputs.weight)
        task_loss = task_metrics["loss"]
        output_distill_loss = self._output_distillation_loss(
            student_output, teacher_output, inputs.target, inputs.weight
        )
        
        (
            distill_loss,
            global_loss,
            token_loss,
            masked_token_loss,
        ) = self._distillation_losses(
            student_embed,
            teacher_embed,
            student_tokens,
            masked_student_tokens,
            teacher_tokens,
            inputs.cand_mask,
        )
        
        total_loss = (
            distill_loss
            if self.in_representation_stage
            else self._adaptation_loss(
                task_loss, output_distill_loss, distill_loss
            )
        )
        
        self.training_loss_accumulator["loss"].append(total_loss.detach().cpu())
        self.training_loss_accumulator["task_loss"].append(task_loss.detach().cpu())
        self.training_loss_accumulator["distill_loss"].append(
            distill_loss.detach().cpu()
        )
        self.training_loss_accumulator["global_distill_loss"].append(
            global_loss.detach().cpu()
        )
        self.training_loss_accumulator["token_distill_loss"].append(
            token_loss.detach().cpu()
        )
        self.training_loss_accumulator["masked_token_distill_loss"].append(
            masked_token_loss.detach().cpu()
        )
        self.training_loss_accumulator["output_distill_loss"].append(
            output_distill_loss.detach().cpu()
        )
        for key, value in task_metrics.items():
            if key != "loss":
                self.training_loss_accumulator[key].append(value.detach().cpu())
                
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        stage_index = {"representation": 0, "head_warmup": 1, "joint": 2}[
            self.training_stage
        ]
        optimizer = optimizers[stage_index]
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        optimizer.step()

        self.log(
            "LR",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return total_loss.detach()

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
        (
            student_output,
            teacher_output,
            student_embed,
            teacher_embed,
            student_tokens,
            masked_student_tokens,
            teacher_tokens,
            inputs,
        ) = self.forward(batch)
        predictions = self._get_predictions_dict(student_output)
        task_metrics = self._calculate_task_metrics(inputs.target, predictions, inputs.weight)
        task_loss = task_metrics["loss"]
        output_distill_loss = self._output_distillation_loss(
            student_output, teacher_output, inputs.target, inputs.weight
        )
        
        (
            distill_loss,
            global_loss,
            token_loss,
            masked_token_loss,
        ) = self._distillation_losses(
            student_embed,
            teacher_embed,
            student_tokens,
            masked_student_tokens,
            teacher_tokens,
            inputs.cand_mask,
        )
        total_loss = (
            distill_loss
            if self.in_representation_stage
            else self._adaptation_loss(
                task_loss, output_distill_loss, distill_loss
            )
        )
        
        self.validation_loss_accumulator["loss"].append(total_loss.detach().cpu())
        self.validation_loss_accumulator["task_loss"].append(task_loss.detach().cpu())
        self.validation_loss_accumulator["distill_loss"].append(
            distill_loss.detach().cpu()
        )
        self.validation_loss_accumulator["global_distill_loss"].append(
            global_loss.detach().cpu()
        )
        self.validation_loss_accumulator["token_distill_loss"].append(
            token_loss.detach().cpu()
        )
        self.validation_loss_accumulator["masked_token_distill_loss"].append(
            masked_token_loss.detach().cpu()
        )
        self.validation_loss_accumulator["output_distill_loss"].append(
            output_distill_loss.detach().cpu()
        )
        for key, value in task_metrics.items():
            if key != "loss":
                self.validation_loss_accumulator[key].append(value.detach().cpu())

        # Awkward cannot consume bfloat16 tensors through DLPack. Move validation
        # outputs to CPU immediately and promote floating-point values to float32.
        cpu_predictions = {
            key: value.detach().float().cpu()
            if value.is_floating_point()
            else value.detach().cpu()
            for key, value in predictions.items()
        }
        cpu_targets = {
            key: value.detach().float().cpu()
            if value.is_floating_point()
            else value.detach().cpu()
            for key, value in inputs.target.items()
        }
        cpu_gen_jet_p4s = {
            key: value.detach().float().cpu()
            for key, value in inputs.gen_jet_p4s.items()
        }
        cpu_reco_jet_p4s = {
            key: value.detach().float().cpu()
            for key, value in inputs.reco_jet_p4s.items()
        }
        cpu_gen_jet_tau_p4s = {
            key: value.detach().float().cpu()
            for key, value in inputs.gen_jet_tau_p4s.items()
        }
        
        self.validation_outputs.append(
            {
                "predictions": cpu_predictions,
                "targets": cpu_targets,
                "gen_jet_p4s": cpu_gen_jet_p4s,
                "reco_jet_p4s": cpu_reco_jet_p4s,
                "gen_jet_tau_p4s": cpu_gen_jet_tau_p4s,
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
        representation_optimizer = torch.optim.AdamW(
            list(self.student.backbone.parameters())
            + list(self.global_projection.parameters())
            + list(self.token_projection.parameters()),
            lr=self.representation_lr,
            weight_decay=1e-2,
        )
        head_optimizer = torch.optim.AdamW(
            self._student_head().parameters(),
            lr=self.head_warmup_lr,
            weight_decay=1e-2,
        )
        joint_optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.student.backbone.parameters(),
                    "lr": self.joint_backbone_lr,
                },
                {
                    "params": self._student_head().parameters(),
                    "lr": self.joint_head_lr,
                },
                {
                    "params": list(self.global_projection.parameters())
                    + list(self.token_projection.parameters()),
                    "lr": self.joint_backbone_lr,
                },
            ],
            weight_decay=1e-2,
        )

        total_epochs = self.cfg.training.trainer.max_epochs
        joint_epochs = max(
            1,
            total_epochs - self.representation_epochs - self.head_warmup_epochs,
        )
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                representation_optimizer, T_max=max(1, self.representation_epochs)
            ),
            torch.optim.lr_scheduler.LinearLR(
                head_optimizer,
                start_factor=0.2,
                end_factor=1.0,
                total_iters=max(1, self.head_warmup_epochs),
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                joint_optimizer, T_max=joint_epochs
            ),
        ]
        return (
            [representation_optimizer, head_optimizer, joint_optimizer],
            schedulers,
        )
