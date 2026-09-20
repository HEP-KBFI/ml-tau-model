"""
Per-feature standardisation of the candidate inputs.

These helpers used to live in the preprocessed (.pt) dataloader, which loaded a
whole split into memory as one tensor tuple and could therefore fit a scaler
over it in one pass. That loader is gone -- training reads parquet and builds
tensors per chunk -- so `fit_and_apply_input_scaling` currently has no caller:
fitting a scaler now needs a pass over the parquet, which nothing does yet.

`apply_saved_input_scaling_from_cfg` is still used at inference time and works
on any tensor tuple, one batch at a time included, because the transform is a
per-feature affine map. `make_input_scaler` is the form to prefer in a loop: it
reads the .npz once and returns a callable, rather than re-loading it per batch.

Everything is a no-op unless `training.input_scaling.enabled` is true.
"""

import os
import warnings

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


# Metadata for recording the input features order in the .npz file
_CAND_FEATURE_NAMES = np.array(
    [
        "cand_deta",
        "cand_dphi",
        "cand_logpt",
        "cand_loge",
        "cand_logptrel",
        "cand_logerel",
        "cand_deltaR",
        "cand_charge",
        "isElectron",
        "isMuon",
        "isPhoton",
        "isChargedHadron",
        "isNeutralHadron",
        "cand_dz",
        "cand_dz_error",
        "cand_dxy",
        "cand_dxy_error",
    ]
)


def scaling_enabled(cfg: DictConfig) -> bool:
    """Checks whether training.input_scaling.enabled exists and is true. (False by default so that old configs still run normally)."""
    scaling_cfg = OmegaConf.select(cfg, "training.input_scaling")
    return scaling_cfg is not None and bool(scaling_cfg.enabled)


def scaler_path(cfg: DictConfig) -> str:
    """Turns the configured path into an absolute path. This is where the scaler gets saved and later loaded."""
    return os.path.abspath(
        os.path.expanduser(str(cfg.training.input_scaling.scaler_path))
    )


def _feature_indices(cfg: DictConfig) -> list[int]:
    """Reads the list of continuous feature indices from config. This lets us scale only the features we list in the config."""
    return [int(i) for i in cfg.training.input_scaling.continuous_feature_indices]


class ScalerFitter:
    """
    Accumulate per-feature mean and std over batches, in one pass.

    The old whole-split fitter could hold every jet in memory at once because
    the .pt loader had already done so. Reading parquet, the tensors arrive a
    chunk at a time, so the sums are accumulated instead and the mean and std
    are formed at the end:

        mean = sum / count
        std  = sqrt(sum_x2 / count - mean^2)

    Padded candidates are excluded via the mask, and the accumulators are
    float64 regardless of the input dtype: summing ~1e6 candidate values in
    float32 loses the low bits of the second moment, which is exactly where a
    std comes from.
    """

    def __init__(self, feature_indices):
        self.idx = torch.as_tensor(list(feature_indices), dtype=torch.long)
        self.total = torch.zeros(len(self.idx), dtype=torch.float64)
        self.total_sq = torch.zeros(len(self.idx), dtype=torch.float64)
        self.count = 0

    def update(self, cand_features, mask) -> int:
        """Add one chunk; returns the number of valid candidates it contributed."""
        x = cand_features.permute(0, 2, 1)  # [N, P, C]
        valid = mask.squeeze(1).bool()
        vals = x[..., self.idx][valid].to(torch.float64)
        if vals.numel() == 0:
            return 0
        self.total += vals.sum(dim=0)
        self.total_sq += (vals**2).sum(dim=0)
        self.count += vals.shape[0]
        return vals.shape[0]

    def result(self, eps: float = 1e-6):
        if self.count == 0:
            raise RuntimeError("Input scaling: no valid candidates to fit the scaler on.")
        mean = self.total / self.count
        var = torch.clamp(self.total_sq / self.count - mean**2, min=0.0)
        std = torch.sqrt(var).clamp_min(eps)
        return mean.numpy(), std.numpy()


def fit_scaler(chunks, cfg: DictConfig, max_jets: int | None = None) -> str:
    """
    Fit a scaler over `chunks` -- an iterable of build_tensors outputs -- and save it.

    Fitting on a bounded subsample rather than the whole split is deliberate: a
    per-feature mean and std converge long before ten million jets, and a full
    extra pass over the training data before every run is a real cost. The jet
    count actually used is stored in the .npz so a scaler can be audited later.
    """
    feature_indices = _feature_indices(cfg)
    fitter = ScalerFitter(feature_indices)
    jets = 0
    for tensors in chunks:
        cand_features, mask = tensors[0], tensors[3]
        fitter.update(cand_features, mask)
        jets += cand_features.shape[0]
        if max_jets is not None and jets >= max_jets:
            break
    mean, std = fitter.result()

    path = scaler_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        mean=mean,
        std=std,
        feature_indices=np.asarray(feature_indices, dtype=np.int64),
        feature_names=_CAND_FEATURE_NAMES,
        jets_fit=np.asarray(jets, dtype=np.int64),
        data_dir=np.asarray(str(cfg.dataset.data_dir)),
    )
    print(
        f"[input scaling] Fitted on {jets:,} jets "
        f"({fitter.count:,} candidates), saved to {path}",
        flush=True,
    )
    return path


def _apply_feature_scaler(tensors, mean, std, feature_indices):
    """Applies x = (x - mean) / std only to selected cand_features channels.
    It leaves cand_kinematics, targets, weights, and p4 dictionaries untouched.
    cf.mul_(msk.to(dtype=cf.dtype)) resets padded candidates back to exactly zero after scaling.
    """
    cf, ck, tgt, msk, wt, gt, rc, gj = tensors
    idx = torch.as_tensor(feature_indices, dtype=torch.long)
    mean_t = torch.as_tensor(mean, dtype=cf.dtype).view(1, -1, 1)
    std_t = torch.as_tensor(std, dtype=cf.dtype).view(1, -1, 1)

    # cf[:, idx, :] = (cf[:, idx, :] - mean_t) / std_t
    cf_scaled = (cf[:, idx, :] - mean_t) / std_t
    cf[:, idx, :] = cf_scaled.clone()
    cf.mul_(msk.to(dtype=cf.dtype))  # keep padded candidates exactly zero
    return cf, ck, tgt, msk, wt, gt, rc, gj


def _warn_on_foreign_scaler(scaler, cfg: DictConfig, path: str) -> None:
    """
    Say so when a scaler was fitted on different data than is being read now.

    The scaler lives under output_dir, not under data_dir, so pointing an
    existing output_dir at a new production silently reuses the old constants --
    which is wrong for training and invisible in any loss curve. It is only a
    warning because applying a training scaler to another directory is exactly
    what evaluation does, and that is correct.
    """
    fitted_on = str(scaler["data_dir"]) if "data_dir" in scaler.files else None
    current = str(cfg.dataset.data_dir)
    if fitted_on is not None and os.path.normpath(fitted_on) != os.path.normpath(current):
        warnings.warn(
            f"Input scaler {path} was fitted on {fitted_on} but dataset.data_dir "
            f"is {current}. Delete the scaler to refit, or ignore this if you are "
            "deliberately applying a training scaler to another sample.",
            stacklevel=3,
        )


def apply_saved_input_scaling_from_cfg(tensors, cfg: DictConfig):
    """This is the test/inference-time entry point:
    1. If scaling is disabled, return tensors unchanged.
    2. Load the saved .npz.
    3. Apply the same train-derived scaler to test/prediction tensors.
    """
    if not scaling_enabled(cfg):
        return tensors

    path = scaler_path(cfg)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Input scaling is enabled, but scaler was not found: {path}"
        )

    scaler = np.load(path)
    mean = scaler["mean"]
    std = scaler["std"]
    feature_indices = scaler["feature_indices"].astype(np.int64).tolist()
    _warn_on_foreign_scaler(scaler, cfg, path)
    print(f"[input scaling] Loaded scaler from {path}", flush=True)

    return _apply_feature_scaler(tensors, mean, std, feature_indices)


def load_saved_scaler(cfg: DictConfig) -> dict:
    """
    The fitted constants as a dict: mean, std (per selected feature) and
    feature_indices. For code that needs the numbers themselves rather than a
    tensors -> tensors transform -- the distillation module folds them into
    buffers so it can un-scale the student's inputs for the frozen teacher.
    """
    path = scaler_path(cfg)
    if not os.path.exists(path):
        raise RuntimeError(f"Input scaling is enabled, but scaler was not found: {path}")
    scaler = np.load(path)
    _warn_on_foreign_scaler(scaler, cfg, path)
    return {
        "mean": scaler["mean"],
        "std": scaler["std"],
        "feature_indices": scaler["feature_indices"].astype(np.int64).tolist(),
    }


def make_input_scaler(cfg: DictConfig):
    """
    Return a `tensors -> tensors` callable, reading the scaler .npz once.

    Equivalent to calling `apply_saved_input_scaling_from_cfg` per batch, but
    without re-reading the file every time; the transform is per-feature affine,
    so applying it batch by batch gives exactly the same result as applying it
    to a whole split at once.
    """
    if not scaling_enabled(cfg):
        return lambda tensors: tensors

    path = scaler_path(cfg)
    if not os.path.exists(path):
        raise RuntimeError(f"Input scaling is enabled, but scaler was not found: {path}")

    scaler = np.load(path)
    mean = scaler["mean"]
    std = scaler["std"]
    feature_indices = scaler["feature_indices"].astype(np.int64).tolist()
    _warn_on_foreign_scaler(scaler, cfg, path)
    print(f"[input scaling] Loaded scaler from {path}", flush=True)

    def scale(tensors):
        return _apply_feature_scaler(tensors, mean, std, feature_indices)

    return scale
