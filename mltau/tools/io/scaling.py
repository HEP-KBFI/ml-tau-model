import os
import torch
import numpy as np
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


def input_scaling_enabled(cfg: DictConfig) -> bool:
    """Checks whether training.input_scaling.enabled exists and is true."""
    scaling_cfg = OmegaConf.select(cfg, "training.input_scaling")
    return scaling_cfg is not None and bool(scaling_cfg.enabled)


def get_scaler_path(cfg: DictConfig) -> str:
    """Turns the configured path into an absolute path."""
    return os.path.abspath(
        os.path.expanduser(str(cfg.training.input_scaling.scaler_path))
    )


def get_feature_indices(cfg: DictConfig) -> list[int]:
    """Reads the list of continuous feature indices from config."""
    return [int(i) for i in cfg.training.input_scaling.continuous_feature_indices]


def fit_feature_scaler(
    cand_features, mask, feature_indices, eps=1e-6, chunk_size=200_000
):
    """Computes mean and std from the actual training slice only.
    Expects cand_features in shape [N, C, P].
    """
    idx = torch.as_tensor(feature_indices, dtype=torch.long)
    valid = mask.squeeze(1).bool()
    total = torch.zeros(len(feature_indices), dtype=torch.float64)
    total_sq = torch.zeros(len(feature_indices), dtype=torch.float64)
    count = 0

    for start in range(0, cand_features.shape[0], chunk_size):
        end = min(start + chunk_size, cand_features.shape[0])
        x = cand_features[start:end].permute(0, 2, 1)  # [N, P, C] for indexing
        m = valid[start:end]
        vals = x[..., idx][m].to(torch.float64)
        if vals.numel() == 0:
            continue
        total += vals.sum(dim=0)
        total_sq += (vals * vals).sum(dim=0)
        count += vals.shape[0]

    if count == 0:
        raise RuntimeError("No valid candidates found while fitting input scaler.")

    mean = total / count
    var = torch.clamp(total_sq / count - mean * mean, min=0.0)
    std = torch.sqrt(var)
    std = torch.where(std < eps, torch.ones_like(std), std)
    return mean.float().numpy(), std.float().numpy()


def apply_feature_scaler(tensors, mean, std, feature_indices):
    """Applies x = (x - mean) / std only to selected cand_features channels.
    Expects cand_features in shape [N, C, P].
    """
    cf, ck, tgt, msk, wt, gt, rc, gj = tensors
    idx = torch.as_tensor(feature_indices, dtype=torch.long)
    mean_t = torch.as_tensor(mean, dtype=cf.dtype).view(1, -1, 1)
    std_t = torch.as_tensor(std, dtype=cf.dtype).view(1, -1, 1)

    cf_scaled = (cf[:, idx, :] - mean_t) / std_t
    cf[:, idx, :] = cf_scaled.clone()
    cf.mul_(msk.to(dtype=cf.dtype))  # keep padded candidates exactly zero
    return cf, ck, tgt, msk, wt, gt, rc, gj


def fit_and_apply_input_scaling(train_tensors, val_tensors, cfg: DictConfig):
    if not input_scaling_enabled(cfg):
        return train_tensors, val_tensors

    feature_indices = get_feature_indices(cfg)
    mean, std = fit_feature_scaler(train_tensors[0], train_tensors[3], feature_indices)

    path = get_scaler_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        mean=mean,
        std=std,
        feature_indices=np.asarray(feature_indices, dtype=np.int64),
        feature_names=_CAND_FEATURE_NAMES,
    )
    print(f"[input scaling] Saved scaler to {path}", flush=True)

    return (
        apply_feature_scaler(train_tensors, mean, std, feature_indices),
        apply_feature_scaler(val_tensors, mean, std, feature_indices),
    )


def load_saved_scaler(cfg: DictConfig):
    path = get_scaler_path(cfg)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Input scaling is enabled, but scaler was not found: {path}"
        )

    scaler = np.load(path)
    return {
        "mean": scaler["mean"],
        "std": scaler["std"],
        "feature_indices": scaler["feature_indices"].astype(np.int64).tolist(),
    }


def apply_scaler(tensors, scaler):
    return apply_feature_scaler(
        tensors, scaler["mean"], scaler["std"], scaler["feature_indices"]
    )


def apply_saved_input_scaling_from_cfg(tensors, cfg: DictConfig):
    if not input_scaling_enabled(cfg):
        return tensors
    scaler = load_saved_scaler(cfg)
    print(f"[input scaling] Loaded scaler from {get_scaler_path(cfg)}", flush=True)
    return apply_scaler(tensors, scaler)
