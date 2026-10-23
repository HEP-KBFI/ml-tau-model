import os

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger  # , CometLogger
from omegaconf import DictConfig, ListConfig


# Checkpoints carry the OmegaConf config in their hyperparameters; torch >= 2.6
# refuses to unpickle those under its weights_only default unless registered.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([DictConfig, ListConfig])

import matplotlib

matplotlib.use("Agg")

# TF32 matmuls on Ampere and later GPUs: Lightning prints this hint on every
# run; it is free throughput and does not touch mixed-precision numerics.
torch.set_float32_matmul_precision("high")

from mltau.models import MultiParTau_module, SingleParTau_module
from mltau.tools.evaluation import inference
from mltau.tools.io import ParT_dataloader as dl
from mltau.tools.io import RareDecays_dataloader as rare_dl

# DataModule to use, selected by training.datamodule (default "ParT"). A
# subclass swaps in a dataset with different targets while reusing all of
# ParTDataModule's file discovery, splitting and batching; see
# RareDecaysDataModule for the decay_mode.decay_mode_scheme: rare case.
DATAMODULE_REGISTRY = {
    "ParT": dl.ParTDataModule,
    "RareDecays": rare_dl.RareDecaysDataModule,
}


@hydra.main(config_path="../config", config_name="main", version_base=None)
def train(cfg: DictConfig):
    datamodule_name = cfg.training.get("datamodule", "ParT")
    if datamodule_name not in DATAMODULE_REGISTRY:
        raise ValueError(
            f"Unknown training.datamodule '{datamodule_name}'. Choose one of "
            f"{sorted(DATAMODULE_REGISTRY)}."
        )
    datamodule = DATAMODULE_REGISTRY[datamodule_name](
        cfg=cfg, debug_run=cfg.training.debug_run
    )
    model_name = cfg.training.model.name
    num_dm_classes = cfg.training.model.get("num_dm_classes", 6)
    if model_name == "MultiParTau":
        model = MultiParTau_module.ParTauModule(cfg=cfg, input_dim=17, num_dm_classes=6)
    elif model_name == "SingleParTau":
        model = SingleParTau_module.ParTauModule(
            cfg=cfg,
            input_dim=17,
            num_dm_classes=num_dm_classes,
            task=cfg.training.model.task,
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

    # Log dataset size
    datamodule.setup("fit")
    train_ds = datamodule.train_dataloader().dataset
    val_ds = datamodule.val_dataloader().dataset

    def get_ds_size(ds):
        if hasattr(ds, "cand_features"):
            return len(ds.cand_features)
        elif hasattr(ds, "num_rows"):
            return ds.num_rows
        return 0

    n_train = get_ds_size(train_ds)
    n_val = get_ds_size(val_ds)
    with open(os.path.join(cfg.output_dir, "dataset_size.txt"), "w") as f:
        f.write(f"train: {n_train}\nval: {n_val}\ntotal: {n_train + n_val}\n")
    print(f"[INFO] Dataset size saved to {cfg.output_dir}/dataset_size.txt")

    # Configure callbacks
    callbacks = [
        TQDMProgressBar(refresh_rate=100),
        ModelCheckpoint(
            dirpath=models_dir,
            monitor="val_losses/loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            filename="ParT-model_best",
        ),
        # last.ckpt with optimizer and scheduler state (see ParTauDETR_module's
        # equivalent in train_ParTauDETR.py), so a run killed by a dataloader
        # worker crash, preemption or a wall clock can resume instead of losing
        # everything. Kept apart from the best-model checkpoint above because a
        # ModelCheckpoint has one save_weights_only flag for both its top-k and
        # its last file.
        ModelCheckpoint(
            dirpath=models_dir,
            save_top_k=0,
            save_last=True,
            save_weights_only=False,
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
        accelerator="auto",
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    # With training.resume, continue from this output_dir's last.ckpt if it has
    # one -- e.g. a resubmission after a dataloader worker crash or a preempted
    # job -- instead of restarting from epoch 0. Off by default, so a new run
    # in a reused output_dir does not silently pick up an old run's state.
    last_ckpt_path = os.path.join(models_dir, "last.ckpt")
    ckpt_path = None
    if cfg.training.get("resume", False) and os.path.exists(last_ckpt_path):
        ckpt_path = last_ckpt_path
    if ckpt_path is not None:
        print(f"[INFO] Resuming from {ckpt_path}")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    # --- Inference on test set using best checkpoint ---
    best_ckpt_path = os.path.join(models_dir, "ParT-model_best.ckpt")
    if os.path.exists(best_ckpt_path):
        print(f"\n[INFO] Running inference on test set using {best_ckpt_path}")
        # Reload the best model
        if model_name == "MultiParTau":
            best_model = MultiParTau_module.ParTauModule.load_from_checkpoint(
                best_ckpt_path, cfg=cfg, input_dim=17, num_dm_classes=6
            )
        elif model_name == "SingleParTau":
            best_model = SingleParTau_module.ParTauModule.load_from_checkpoint(
                best_ckpt_path,
                cfg=cfg,
                input_dim=17,
                num_dm_classes=num_dm_classes,
                task=cfg.training.model.task,
                weights_only=False,
            )
        else:
            raise ValueError(f"Unknown model '{model_name}' for prediction.")

        inference.create_predictions_files(
            best_model=best_model,
            model_name=model_name,
            cfg=cfg,
            dataset_cls=DATAMODULE_REGISTRY[datamodule_name].dataset_cls,
        )

    else:
        print(
            f"[WARNING] Best checkpoint not found at {best_ckpt_path}. Skipping inference."
        )


if __name__ == "__main__":
    train()
