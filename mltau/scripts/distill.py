import os
import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from mltau.tools.io import ParT_dataloader as dl
from mltau.models.DistillationModule import DistillationModule

@hydra.main(config_path="../config", config_name="main", version_base=None)
def distill(cfg: DictConfig):
    # Ensure pT sorting is enabled in the config for the student
    # Use OmegaConf.set_struct to allow adding new keys if needed, 
    # but here we just want to force the value.
    cfg.dataset.sort_by_pt = True
    
    datamodule = dl.ParTDataModule(cfg=cfg, debug_run=cfg.training.debug_run)
    
    # Check for teacher checkpoint
    teacher_checkpoint = cfg.get("teacher_checkpoint", None)
    if not teacher_checkpoint:
        raise ValueError("Please provide 'teacher_checkpoint' path in config or via CLI.")

    distillation_cfg = cfg.get("distillation", {})
    representation_epochs = distillation_cfg.get("representation_epochs", 20)
    if representation_epochs >= cfg.training.trainer.max_epochs:
        raise ValueError(
            "distillation.representation_epochs must be smaller than "
            "training.trainer.max_epochs so the regression head is trained."
        )

    model = DistillationModule(
        cfg=cfg,
        teacher_checkpoint=teacher_checkpoint,
        input_dim=17,
        num_dm_classes=6,
        task=cfg.training.model.task,
        distill_alpha=cfg.get("distill_alpha", 0.5),
        representation_epochs=representation_epochs,
        token_loss_weight=distillation_cfg.get("token_loss_weight", 1.0),
        global_loss_weight=distillation_cfg.get("global_loss_weight", 1.0),
        masked_token_loss_weight=distillation_cfg.get(
            "masked_token_loss_weight", 1.0
        ),
        mask_probability=distillation_cfg.get("mask_probability", 0.3),
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
