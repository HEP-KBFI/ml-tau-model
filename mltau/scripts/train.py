import os
import hydra
import lightning as L
import numpy as np

from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger  # , CometLogger
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, Callback

from mltau.tools.io import preprocessed_ParTau_dataloader as dl
from mltau.models import MultiParTau_module, SingleParTau_module
from mltau.tools.evaluation import inference


class KinematicsFinetuneCallback(Callback):
    """
    Two-phase training for MultiParTau:

    Phase 1  — all parameters train normally.
    Phase 2  — triggered when val_losses/loss has not improved for
               `patience` epochs: freeze everything except regression_head,
               reset the patience counter for kinematics-only fine-tuning.
    Stop     — triggered when val_losses/kinematics_loss has not improved for
               `patience` epochs while in phase 2.

    Both the overall val loss AND the kinematics val loss are monitored so the
    callback is robust to the overall loss stagnating for reasons unrelated to
    kinematics.
    """

    def __init__(self, patience: int = 10):
        super().__init__()
        self.patience = patience
        self.phase = 1
        self.best_val_loss = float("inf")
        self.best_kin_loss = float("inf")
        self.wait = 0

    def _freeze_all_except_regression_head(self, model):
        part = model.ParTau
        for module in [
            part.embed,
            part.pair_embed,
            part.blocks,
            part.cls_blocks,
            part.norm,
        ]:
            for p in module.parameters():
                p.requires_grad_(False)
        for token in [part.cls_token_shared, part.cls_token_kinematics]:
            token.requires_grad_(False)
        for head in [part.tau_id_head, part.tau_charge_head, part.classification_head]:
            for p in head.parameters():
                p.requires_grad_(False)
        # regression_head stays requires_grad=True (default)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics

        if self.phase == 1:
            val_loss = metrics.get("val_losses/loss", None)
            if val_loss is None:
                return
            val_loss = val_loss.item()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.wait = 0
            else:
                self.wait += 1
                print(
                    f"[Phase 1] No improvement for {self.wait}/{self.patience} epochs "
                    f"(best={self.best_val_loss:.6f}, current={val_loss:.6f})"
                )
                if self.wait >= self.patience:
                    print(
                        "[Phase 1 → Phase 2] Switching to kinematics-only fine-tuning."
                    )
                    self.phase = 2
                    self.wait = 0
                    self.best_kin_loss = float("inf")
                    self._freeze_all_except_regression_head(pl_module)
                    pl_module.reinitialize_optimizer_for_phase2()

        elif self.phase == 2:
            kin_loss = metrics.get("val_losses/kinematics_loss", None)
            if kin_loss is None:
                return
            kin_loss = kin_loss.item()
            if kin_loss < self.best_kin_loss:
                self.best_kin_loss = kin_loss
                self.wait = 0
            else:
                self.wait += 1
                print(
                    f"[Phase 2] Kinematics no improvement for {self.wait}/{self.patience} epochs "
                    f"(best={self.best_kin_loss:.6f}, current={kin_loss:.6f})"
                )
                if self.wait >= self.patience:
                    print("[Phase 2] Kinematics converged. Stopping training.")
                    trainer.should_stop = True


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
    if model_name == "MultiParTau":
        callbacks.append(KinematicsFinetuneCallback(patience=5))

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
                num_dm_classes=6,
                task=cfg.training.model.task,
            )
        else:
            raise ValueError(f"Unknown model '{model_name}' for prediction.")

        inference.create_predictions_files(
            best_model=best_model, model_name=model_name, cfg=cfg
        )

    else:
        print(
            f"[WARNING] Best checkpoint not found at {best_ckpt_path}. Skipping inference."
        )


if __name__ == "__main__":
    train()
