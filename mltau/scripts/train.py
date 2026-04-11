import os
import hydra
import lightning as L

from omegaconf import DictConfig
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger  # , CometLogger
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint

from mltau.tools.io import ParT_dataloader as dl
from mltau.tools.io import preprocessed_ParTau_dataloader as dl
from mltau.models import MultiParTau_module, SingleParTau_module


@hydra.main(config_path="../config", config_name="main", version_base=None)
def train(cfg: DictConfig):
    datamodule = dl.ParTDataModule(cfg=cfg, debug_run=cfg.training.debug_run)
    model_name = cfg.training.model.name
    if model_name == "MultiParTau":
        model = MultiParTau_module.ParTauModule(cfg=cfg, input_dim=17, num_dm_classes=6)
    elif model_name == "SingleParTau":
        model = SingleParTau_module.ParTauModule(
            cfg=cfg, input_dim=17, num_dm_classes=6, task=cfg.training.model.task
        )
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose 'MultiParTau' or 'SingleParTau'."
        )
    models_dir = os.path.join(cfg.output_dir, "models")
    log_dir = os.path.join(cfg.output_dir, "logs")
    tb_log_dir = os.path.join(cfg.output_dir, "tensorboard")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # Configure callbacks
    callbacks = [
        TQDMProgressBar(refresh_rate=100),  # Reduced refresh rate for CPU
        ModelCheckpoint(
            dirpath=models_dir,
            monitor="val_losses/loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            filename="ParT-model_best",
        ),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        callbacks=callbacks,
        logger=[
            TensorBoardLogger(
                save_dir=tb_log_dir,
                name="ParTau_experiment",
                log_graph=False,
                default_hp_metric=False,
            ),
        ],
        accelerator="auto",  # Automatically detect GPU/CPU
        precision="16-mixed",  # fp16 activations: halves GPU memory, ~30% faster
        num_sanity_val_steps=0,  # Skip sanity validation for faster startup
        enable_progress_bar=True,  # Keep enabled for monitoring
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
