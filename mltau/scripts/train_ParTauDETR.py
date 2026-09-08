# comet_ml monkey-patches the frameworks it auto-instruments, so it has to be
# imported before torch/lightning to log correctly. Guarded so that this script
# still runs on an environment without comet installed.
try:
    import comet_ml  # noqa: F401

    _COMET_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    comet_ml = None
    _COMET_IMPORT_ERROR = exc

import inspect
import os
import warnings

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from mltau.models import ParTauDETR_module
from mltau.tools.io import ParTauDETR_dataloader as dl


# Every experiment from this repository belongs to one Comet project. This is a
# constant rather than a config key on purpose: a stale or overridden value
# silently creates a second project (Comet slugifies names to lowercase, so
# "ml-tau-ParTauDETR" showed up as "ml-tau-partaudetr").
COMET_PROJECT_NAME = "ml-tau"

# Lightning >= 2.5 injects ExperimentConfig(disabled=True) whenever `online`
# is the False singleton, which produces an OfflineExperiment that records
# nothing and writes no uploadable archive. A falsy value that is not that
# singleton (0 is not False) keeps comet_ml's real offline path while dodging
# the coercion. Verified against lightning 2.5.4 / comet_ml 3.52.
COMET_OFFLINE = 0


def build_comet_logger(cfg: DictConfig, save_dir: str):
    """
    Build a CometLogger from cfg.logging.comet, or return None if disabled.

    Lightning reworked CometLogger in 2.5 to wrap `comet_ml.start()`, which moved
    the project, name and offline-directory arguments around. Both spellings are
    supported, selected by inspecting the installed signature.
    """
    comet_cfg = cfg.logging.comet
    if not comet_cfg.enabled:
        return None

    if comet_ml is None:
        raise RuntimeError(
            "logging.comet.enabled is true but comet_ml is not installed "
            f"({_COMET_IMPORT_ERROR}). Install it with `pip install comet_ml`, "
            "or set logging.comet.enabled=false."
        )

    from lightning.pytorch.loggers import CometLogger

    accepted = inspect.signature(CometLogger.__init__).parameters
    online = bool(comet_cfg.online)
    experiment_name = comet_cfg.experiment_name
    experiment_name = None if experiment_name is None else str(experiment_name)
    experiment_key = comet_cfg.experiment_key
    experiment_key = None if experiment_key is None else str(experiment_key)
    workspace = None if comet_cfg.workspace is None else str(comet_cfg.workspace)
    # Forwarded into comet_ml.ExperimentConfig; drives the System Metrics tab.
    # Lightning sets COMET_DISABLE_AUTO_LOGGING=1, but that only gates framework
    # auto-instrumentation, not the system/environment monitor.
    env_kwargs = {
        "log_env_details": bool(comet_cfg.log_env_details),
        "log_env_gpu": bool(comet_cfg.log_env_gpu),
        "log_env_cpu": bool(comet_cfg.log_env_cpu),
    }

    if "online" in accepted:
        # Lightning >= 2.5. Two traps here:
        #  - `mode` is comet_ml's get/create/get_or_create selector, NOT
        #    online-vs-offline. That is `online`.
        #  - every unrecognised kwarg is forwarded into comet_ml.ExperimentConfig,
        #    so the name and the offline directory are passed flat, not as a
        #    prebuilt ExperimentConfig object.
        # Leaving `name` unset lets Comet generate one itself.
        kwargs = {
            "project": COMET_PROJECT_NAME,
            "online": True if online else COMET_OFFLINE,
            "offline_directory": save_dir,
            "log_code": bool(comet_cfg.log_code),
            "log_graph": bool(comet_cfg.log_graph),
            **env_kwargs,
        }
        if experiment_name is not None:
            kwargs["name"] = experiment_name
        if workspace is not None:
            kwargs["workspace"] = workspace
        if experiment_key is not None:
            # Resume the run carrying this key, creating it if it is new.
            kwargs["experiment_key"] = experiment_key
            kwargs["mode"] = "get_or_create"
        return CometLogger(**kwargs)

    # Lightning < 2.5
    kwargs = {
        "project_name": COMET_PROJECT_NAME,
        "save_dir": save_dir,
        "offline": not online,
    }
    if workspace is not None:
        kwargs["workspace"] = workspace
    if experiment_name is not None:
        kwargs["experiment_name"] = experiment_name
    if experiment_key is not None:
        kwargs["experiment_key"] = experiment_key
    # Older CometLogger forwards unknown kwargs to the comet Experiment, so the
    # env_* flags reach it even though they are not named in the signature.
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return CometLogger(**kwargs, **env_kwargs)


def _cuda_visible_devices_hint() -> str:
    """
    Detect a CUDA_VISIBLE_DEVICES that indexes past the devices actually present.

    Slurm hands out node-global GPU indices while its cgroup renumbers the
    job's view from 0, so an allocation of the node's second GPU arrives as
    CUDA_VISIBLE_DEVICES=1 against a single visible device. CUDA then finds
    nothing. pynvml is used because it reports the physical devices regardless
    of the masking variable, and unlike torch it does not need a working
    CUDA context.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return ""
    indices = [part for part in visible.split(",") if part.strip().isdigit()]
    if not indices:
        return ""
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
    except Exception:
        return ""
    if count and max(int(i) for i in indices) >= count:
        return (
            f"CUDA_VISIBLE_DEVICES={visible} indexes past the {count} device(s) "
            "this process can actually see.\n"
            "Slurm sets the node-global GPU index while its cgroup renumbers the\n"
            "job's view from 0, so the variable is stale. Renumber it densely:\n"
            f"  export CUDA_VISIBLE_DEVICES={','.join(str(i) for i in range(count))}\n"
            "train-gpu-DETR.sh now does this automatically before launching.\n\n"
        )
    return ""


def check_accelerator(cfg: DictConfig) -> str:
    """
    Validate the requested accelerator before any data is touched.

    Lightning's `accelerator="auto"` quietly selects CPU when CUDA is not
    visible, so a misconfigured job trains ~100x too slowly instead of failing.
    """
    requested = str(cfg.training.trainer.accelerator)
    if requested in ("gpu", "cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            _cuda_visible_devices_hint()
            + "training.trainer.accelerator is "
            f"{requested!r} but no CUDA device is visible.\n"
            f"  torch.cuda.is_available() : False\n"
            f"  torch.version.cuda        : {torch.version.cuda}\n"
            "  CUDA_VISIBLE_DEVICES      : "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n"
            "  SLURM_JOB_GPUS            : "
            f"{os.environ.get('SLURM_JOB_GPUS', '<unset>')}\n"
            "  SLURM_JOB_ID              : "
            f"{os.environ.get('SLURM_JOB_ID', '<unset>')}\n"
            "Check that (a) the job actually requested a GPU -- note that #SBATCH\n"
            "lines only count before the first command in the submit script -- and\n"
            "(b) run.sh passes --nv to apptainer. To run without a GPU on purpose,\n"
            "set training.trainer.accelerator=cpu training.trainer.precision=bf16-mixed."
        )
    if requested in ("gpu", "cuda"):
        print(
            f"[ParTauDETR] CUDA devices: {torch.cuda.device_count()} "
            f"({torch.cuda.get_device_name(0)}), "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GiB"
        )
    return requested


def build_loggers(cfg: DictConfig, tb_log_dir: str) -> list:
    loggers = []
    if cfg.logging.tensorboard.enabled:
        loggers.append(
            TensorBoardLogger(
                save_dir=tb_log_dir,
                name="ParTauDETR_experiment",
                log_graph=False,
                default_hp_metric=False,
            )
        )
    comet_logger = build_comet_logger(
        cfg, save_dir=os.path.join(cfg.output_dir, "comet")
    )
    if comet_logger is not None:
        loggers.append(comet_logger)
    if not loggers:
        warnings.warn("No logger is enabled under cfg.logging; metrics are dropped.")
    return loggers


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


    model = ParTauDETR_module.ParTauDETRModule(cfg=cfg)

    models_dir = os.path.join(cfg.output_dir, "models")
    tb_log_dir = os.path.join(cfg.output_dir, "tensorboard")
    comet_dir = os.path.join(cfg.output_dir, "comet")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(comet_dir, exist_ok=True)

    print(f"[ParTauDETR] output_dir: {cfg.output_dir}")
    print(f"[ParTauDETR] checkpoints dir: {models_dir}")

    loggers = build_loggers(cfg, tb_log_dir=tb_log_dir)
    for logger in loggers:
        url = getattr(getattr(logger, "experiment", None), "url", None)
        print(f"[ParTauDETR] logger: {type(logger).__name__}{f' -> {url}' if url else ''}")

    callbacks = [
        TQDMProgressBar(refresh_rate=10),
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
        logger=loggers,
        accelerator=check_accelerator(cfg),
        devices=cfg.training.trainer.devices,
        precision=str(cfg.training.trainer.precision),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    try:
        trainer.fit(model=model, datamodule=datamodule)
    finally:
        # CometLogger.finalize() only flushes; an OfflineExperiment writes its
        # uploadable .zip on end(). Without this an offline run leaves nothing
        # behind, so end the experiment even if training raised.
        for logger in loggers:
            experiment = getattr(logger, "_experiment", None)
            if experiment is not None and hasattr(experiment, "end"):
                experiment.end()


if __name__ == "__main__":
    train()
