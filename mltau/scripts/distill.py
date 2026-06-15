import os
import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

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

    model = DistillationModule(
        cfg=cfg,
        teacher_checkpoint=teacher_checkpoint,
        input_dim=17,
        num_dm_classes=6,
        task=cfg.training.model.task,
        distill_alpha=cfg.get("distill_alpha", 0.5)
    )

    log_dir = os.path.join(cfg.output_dir, "distill_logs")
    os.makedirs(log_dir, exist_ok=True)

    trainer = L.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        accelerator="auto",
        precision="16-mixed",
        default_root_dir=cfg.output_dir,
    )

    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    distill()
