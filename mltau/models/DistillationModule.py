import torch
import torch.nn as nn
import lightning as L
from omegaconf import DictConfig

from mltau.tools.io.general import BatchInputs
from mltau.tools.losses import TauLoss
from mltau.models.SingleParTau import ParTau as TeacherParT
from mltau.models.MixerTau import MixerTau as StudentMixer

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
        checkpoint = torch.load(teacher_checkpoint, map_location="cpu")
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
        
        self.validation_outputs = []

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
        # Re-using the logic from ParTauModule
        predictions = self._get_predictions_dict(student_output)
        task_metrics = self._calculate_task_metrics(inputs.target, predictions, inputs.weight)
        task_loss = task_metrics["loss"]
        
        # 2. Distillation Loss (Embedding MSE)
        projected_student_embed = self.projection(student_embed)
        distill_loss = self.mse_loss(projected_student_embed, teacher_embed)
        
        # 3. Combined Loss
        total_loss = (1 - self.distill_alpha) * task_loss + self.distill_alpha * distill_loss
        
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/task_loss", task_loss, on_step=True, on_epoch=True)
        self.log("train/distill_loss", distill_loss, on_step=True, on_epoch=True)
        
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
        # Implementation copied/adapted from ParTauModule
        pred = predictions[self.task]
        target = targets[self.task]
        is_tau_mask = targets["is_tau"].bool()

        if self.task == "kinematics":
            if not is_tau_mask.any():
                return {"loss": pred.new_zeros(())}
            loss, _ = self.tau_loss.compute_kinematics_loss(pred[is_tau_mask], target[is_tau_mask], weights[is_tau_mask])
        elif self.task == "is_tau":
            loss = self.tau_loss.compute_tagging_loss(predictions["is_tau_logits"], target, weights)
        elif self.task == "charge":
            if not is_tau_mask.any():
                return {"loss": pred.new_zeros(())}
            loss = self.tau_loss.compute_charge_loss(predictions["charge_logits"][is_tau_mask], target[is_tau_mask], weights[is_tau_mask])
        else: # decay_mode
            if not is_tau_mask.any():
                return {"loss": pred.new_zeros(())}
            loss = self.tau_loss.compute_decay_mode_loss(predictions["decay_mode_logits"][is_tau_mask], target[is_tau_mask], weights[is_tau_mask])
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        student_output, teacher_output, student_embed, teacher_embed, inputs = self.forward(batch)
        predictions = self._get_predictions_dict(student_output)
        task_metrics = self._calculate_task_metrics(inputs.target, predictions, inputs.weight)
        
        projected_student_embed = self.projection(student_embed)
        distill_loss = self.mse_loss(projected_student_embed, teacher_embed)
        
        self.log("val/task_loss", task_metrics["loss"], on_epoch=True)
        self.log("val/distill_loss", distill_loss, on_epoch=True)
        
        return task_metrics["loss"]

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
