import os

import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from mltau.models import ParTauDETR_module
from mltau.tools.io import ParTauDETR_dataloader as dl


@hydra.main(config_path="../config", config_name="main_ParTauDETR", version_base=None)
def train(cfg: DictConfig):
    # Ensure the datamodule follows the signal-only path by default.
    cfg.training.model.name = "ParTauDETR"
    cfg.training.model.task = "set2set"

    # Safety belt: after Hydra composes configs, `cfg.dataset` must be a DictConfig,
    # not an overridden string.
    if not isinstance(cfg.dataset, DictConfig):
        raise TypeError(
            f"'cfg.dataset' is expected to be a DictConfig but got {type(cfg.dataset)}. "
            f"Check that Hydra composed configs correctly (main_ParTauDETR.yaml defaults)."
        )

    datamodule = dl.ParTauDETRDataModule(cfg=cfg, debug_run=cfg.training.debug_run)


    model = ParTauDETR_module.ParTauDETRModule(
        cfg=cfg,
        input_dim=17,
        num_queries=cfg.model.num_queries,
        num_charge_classes=cfg.model.num_charge_classes,
    )

    models_dir = os.path.join(cfg.output_dir, "models")
    tb_log_dir = os.path.join(cfg.output_dir, "tensorboard")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    print(f"[ParTauDETR] output_dir: {cfg.output_dir}")
    print(f"[ParTauDETR] checkpoints dir: {models_dir}")

    callbacks = [
        TQDMProgressBar(refresh_rate=100),
        # Best by validation loss
        ModelCheckpoint(
            dirpath=models_dir,
            monitor="val_losses/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
            filename="ParTauDETR-model_best",
        ),
        # Fallback: best by train loss (useful if val metric is unavailable)
        ModelCheckpoint(
            dirpath=models_dir,
            monitor="train_losses/loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            filename="ParTauDETR-model_best_train",
        ),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        callbacks=callbacks,
        logger=[
            TensorBoardLogger(
                save_dir=tb_log_dir,
                name="ParTauDETR_experiment",
                log_graph=False,
                default_hp_metric=False,
            )
        ],
        accelerator="auto",
        precision="16-mixed",
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
