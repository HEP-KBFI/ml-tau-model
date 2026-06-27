"""
### MLP-Mixer Distillation

The Particle Transformer teacher is distilled into the smaller MLP-Mixer using
a three-stage, GKD-inspired procedure:

1. **Representation distillation:** the task head is frozen while the Mixer
   backbone learns from the frozen teacher. The loss combines normalized global
   embedding alignment, query-based constituent-token alignment, and the same
   token objective with randomly masked input constituents.
2. **Head warmup:** the distilled backbone is frozen briefly while the newly
   initialized task head is trained with a low learning rate against labels and
   teacher regression outputs.
3. **Joint adaptation:** the backbone and head are fine-tuned together using
   separate learning rates. The loss combines task supervision, output-level
   distillation, and a reduced representation-distillation term.

By default, the first 20 of 100 epochs are used for representation
distillation and the next 5 for head warmup. Continuous Mixer inputs are
standardized using statistics fitted on the training split; inputs to the
frozen teacher are converted back to their original scale. Schedule, learning
rates, masking, and loss weights are configured under `distillation` in
`mltau/config/training.yaml`.

`submit_mixer.sh` launches scratch and distilled Mixer jobs for tau
identification, charge, decay mode, and kinematics.
"""

import os
import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mltau.tools.io import ParT_dataloader as dl
from mltau.models.DistillationModule import DistillationModule

@hydra.main(config_path="../config", config_name="main", version_base=None)
def distill(cfg: DictConfig):
    # Ensure pT sorting is enabled in the config for the student
    # Use OmegaConf.set_struct to allow adding new keys if needed, 
    # but here we just want to force the value.
    cfg.dataset.sort_by_pt = True

    distillation_cfg = cfg.get("distillation", {})
    if distillation_cfg.get("enable_input_scaling", True):
        cfg.training.input_scaling.enabled = True

    datamodule = dl.ParTDataModule(cfg=cfg, debug_run=cfg.training.debug_run)
    # Fit/load the input scaler and construct the datasets before the model's
    # on_fit_start hook attempts to load the scaler statistics.
    datamodule.setup("fit")
    
    # Check for teacher checkpoint
    teacher_checkpoint = cfg.get("teacher_checkpoint", None)
    if not teacher_checkpoint:
        raise ValueError("Please provide 'teacher_checkpoint' path in config or via CLI.")

    representation_epochs = distillation_cfg.get("representation_epochs", 20)
    head_warmup_epochs = distillation_cfg.get("head_warmup_epochs", 5)
    if (
        representation_epochs + head_warmup_epochs
        >= cfg.training.trainer.max_epochs
    ):
        raise ValueError(
            "distillation representation and head-warmup epochs must leave "
            "at least one epoch for joint fine-tuning."
        )

    model = DistillationModule(
        cfg=cfg,
        teacher_checkpoint=teacher_checkpoint,
        input_dim=17,
        num_dm_classes=6,
        task=cfg.training.model.task,
        distill_alpha=cfg.get("distill_alpha", 0.5),
        representation_epochs=representation_epochs,
        head_warmup_epochs=head_warmup_epochs,
        token_loss_weight=distillation_cfg.get("token_loss_weight", 1.0),
        global_loss_weight=distillation_cfg.get("global_loss_weight", 1.0),
        masked_token_loss_weight=distillation_cfg.get(
            "masked_token_loss_weight", 1.0
        ),
        mask_probability=distillation_cfg.get("mask_probability", 0.3),
        output_distill_loss_weight=distillation_cfg.get(
            "output_distill_loss_weight", 1.0
        ),
        adaptation_feature_loss_weight=distillation_cfg.get(
            "adaptation_feature_loss_weight", 0.1
        ),
        representation_lr=distillation_cfg.get("representation_lr", 1e-3),
        head_warmup_lr=distillation_cfg.get("head_warmup_lr", 1e-4),
        joint_head_lr=distillation_cfg.get("joint_head_lr", 3e-4),
        joint_backbone_lr=distillation_cfg.get("joint_backbone_lr", 2e-5),
    )

    models_dir = os.path.join(cfg.output_dir, "models")
    tb_log_dir = os.path.join(cfg.output_dir, "tensorboard")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    callbacks = [
        TQDMProgressBar(refresh_rate=100),
        ModelCheckpoint(
            dirpath=models_dir,
            monitor="val_losses/checkpoint_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            filename="ParT-model_best",
        ),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        accelerator="auto",
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=[
            TensorBoardLogger(
                save_dir=tb_log_dir,
                name="ParTau_distill_experiment",
                log_graph=False,
                default_hp_metric=False,
            ),
        ],
        default_root_dir=cfg.output_dir,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    distill()
