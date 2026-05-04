import os
import glob
import math
import torch
import numpy as np

from torch.utils.data import DataLoader, IterableDataset
from omegaconf import DictConfig, OmegaConf
from lightning import LightningDataModule


class ParticleTransformerDataset(IterableDataset):
    """Wraps a pre-loaded slice of jet tensors.  Workers share the same
    physical memory via share_memory_(); no file I/O happens in __iter__."""

    def __init__(self, tensors: tuple, batch_size: int, shuffle: bool = True):
        super().__init__()
        (
            self.cand_features,
            self.cand_kinematics,
            self.targets,
            self.mask,
            self.weights,
            self.gen_tau,
            self.reco,
            self.gen_jet,
        ) = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(self.cand_features.shape[0] / self.batch_size)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # N = 5000
        N = self.cand_features.shape[0]
        # Strided partition — each worker gets every num_workers-th index.
        # Because the dataset is pre-shuffled (see _load_and_split), consecutive
        # indices are already mixed z/qq, so the stride preserves that mix.
        worker_indices = torch.arange(worker_id, N, num_workers)
        # Per-epoch local shuffle within this worker's slice.
        if self.shuffle:
            worker_indices = worker_indices[torch.randperm(len(worker_indices))]

        for i in range(0, len(worker_indices), self.batch_size):
            idx = worker_indices[i : i + self.batch_size]
            yield (
                self.cand_features[idx],
                self.cand_kinematics[idx],
                {k: v[idx] for k, v in self.targets.items()},
                self.mask[idx],
                self.weights[idx],
                {k: v[idx] for k, v in self.gen_tau.items()},
                {k: v[idx] for k, v in self.reco.items()},
                {k: v[idx] for k, v in self.gen_jet.items()},
            )


def _load_and_split(pt_paths: list[str], train_frac: float) -> tuple:
    """Load all .pt files once, globally shuffle, then return (train_tensors, val_tensors).
    Both slices are views of the same shared-memory storage — no extra RAM used."""
    print(f"Loading {len(pt_paths)} .pt file(s) into shared memory...", flush=True)
    parts = [torch.load(p, weights_only=True) for p in pt_paths]

    def _cat(tensors):
        return torch.cat(tensors, dim=0)

    cf = _cat([p[0].transpose(1, 2) for p in parts])
    ck = _cat([p[1].transpose(1, 2) for p in parts])
    tgt = {k: _cat([p[2][k] for p in parts]) for k in parts[0][2]}
    msk = _cat([p[3] for p in parts])
    wt = _cat([p[4] for p in parts])
    gt = {k: _cat([p[5][k] for p in parts]) for k in parts[0][5]}
    rc = {k: _cat([p[6][k] for p in parts]) for k in parts[0][6]}
    gj = {k: _cat([p[7][k] for p in parts]) for k in parts[0][7]}
    del parts

    # Global shuffle with fixed seed so train/val split is reproducible.
    N = cf.shape[0]
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(42))
    cf = cf[perm].share_memory_()
    ck = ck[perm].share_memory_()
    tgt = {k: v[perm].share_memory_() for k, v in tgt.items()}
    msk = msk[perm].share_memory_()
    wt = wt[perm].share_memory_()
    gt = {k: v[perm].share_memory_() for k, v in gt.items()}
    rc = {k: v[perm].share_memory_() for k, v in rc.items()}
    gj = {k: v[perm].share_memory_() for k, v in gj.items()}

    n_train = int(N * train_frac)
    print(
        f"Shared memory ready: {N:,} jets total  "
        f"(train {n_train:,} | val {N - n_train:,})",
        flush=True,
    )

    def _slice(start, end):
        return (
            cf[start:end],
            ck[start:end],
            {k: v[start:end] for k, v in tgt.items()},
            msk[start:end],
            wt[start:end],
            {k: v[start:end] for k, v in gt.items()},
            {k: v[start:end] for k, v in rc.items()},
            {k: v[start:end] for k, v in gj.items()},
        )

    return _slice(0, n_train), _slice(n_train, N)

#
### Add helper functions for input feature scaling
#

# Metadata for recording the input features order in the .npz file
_CAND_FEATURE_NAMES = np.array([
    "cand_deta", "cand_dphi", "cand_logpt", "cand_loge",
    "cand_logptrel", "cand_logerel", "cand_deltaR", "cand_charge",
    "isElectron", "isMuon", "isPhoton", "isChargedHadron",
    "isNeutralHadron", "cand_dz", "cand_dz_error", "cand_dxy",
    "cand_dxy_error",
])

def _input_scaling_enabled(cfg: DictConfig) -> bool:
    """ Checks whether training.input_scaling.enabled exists and is true. (False by default so that old configs still run normally). """
    scaling_cfg = OmegaConf.select(cfg, "training.input_scaling")
    return scaling_cfg is not None and bool(scaling_cfg.enabled)


def _scaler_path(cfg: DictConfig) -> str:
    """ Turns the configured path into an absolute path. This is where the scaler gets saved and later loaded. """
    return os.path.abspath(os.path.expanduser(str(cfg.training.input_scaling.scaler_path)))


def _feature_indices(cfg: DictConfig) -> list[int]:
    """ Reads the list of continuous feature indices from config. This lets us scale only the features we list in the config. """
    return [int(i) for i in cfg.training.input_scaling.continuous_feature_indices]

def _fit_feature_scaler(cand_features, mask, feature_indices, eps=1e-6, chunk_size=200_000):
    """ Computes mean and std from the actual training slice only. It uses valid = mask.squeeze(1).bool() so that padded candidates are ignored. 
    It loops in chunks to avoid making a huge flattened copy of all candidates at once. 
    mean = sum / count
    std = sqrt(sum_x2 / count - mean^2) 
    """
    idx = torch.as_tensor(feature_indices, dtype=torch.long)
    valid = mask.squeeze(1).bool()
    total = torch.zeros(len(feature_indices), dtype=torch.float64)
    total_sq = torch.zeros(len(feature_indices), dtype=torch.float64)
    count = 0

    for start in range(0, cand_features.shape[0], chunk_size):
        end = min(start + chunk_size, cand_features.shape[0])
        x = cand_features[start:end].permute(0, 2, 1)  # [N, P, C]
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


def _apply_feature_scaler(tensors, mean, std, feature_indices):
    """ Applies x = (x - mean) / std only to selected cand_features channels.
        It leaves cand_kinematics, targets, weights, and p4 dictionaries untouched.
        cf.mul_(msk.to(dtype=cf.dtype)) resets padded candidates back to exactly zero after scaling.
    """
    cf, ck, tgt, msk, wt, gt, rc, gj = tensors
    idx = torch.as_tensor(feature_indices, dtype=torch.long)
    mean_t = torch.as_tensor(mean, dtype=cf.dtype).view(1, -1, 1)
    std_t = torch.as_tensor(std, dtype=cf.dtype).view(1, -1, 1)

    cf[:, idx, :] = (cf[:, idx, :] - mean_t) / std_t
    cf.mul_(msk.to(dtype=cf.dtype))  # keep padded candidates exactly zero
    return cf, ck, tgt, msk, wt, gt, rc, gj


def fit_and_apply_input_scaling(train_tensors, val_tensors, cfg: DictConfig):
    """ This is the training-time entry point:
        1. If scaling is disabled, return tensors unchanged.
        2. Fit scaler on train_tensors only.
        3. Save mean, std, feature_indices, and feature_names to .npz.
        4. Apply the same scaler to both train and val.
    """
    if not _input_scaling_enabled(cfg):
        return train_tensors, val_tensors

    feature_indices = _feature_indices(cfg)
    mean, std = _fit_feature_scaler(train_tensors[0], train_tensors[3], feature_indices)

    path = _scaler_path(cfg)
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
        _apply_feature_scaler(train_tensors, mean, std, feature_indices),
        _apply_feature_scaler(val_tensors, mean, std, feature_indices),
    )


def apply_saved_input_scaling_from_cfg(tensors, cfg: DictConfig):
    """ This is the test/inference-time entry point:
        1. If scaling is disabled, return tensors unchanged.
        2. Load the saved .npz.
        3. Apply the same train-derived scaler to test/prediction tensors.
    """
    if not _input_scaling_enabled(cfg):
        return tensors

    path = _scaler_path(cfg)
    if not os.path.exists(path):
        raise RuntimeError(f"Input scaling is enabled, but scaler was not found: {path}")

    scaler = np.load(path)
    mean = scaler["mean"]
    std = scaler["std"]
    feature_indices = scaler["feature_indices"].astype(np.int64).tolist()
    print(f"[input scaling] Loaded scaler from {path}", flush=True)

    return _apply_feature_scaler(tensors, mean, std, feature_indices)


class ParTDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig, debug_run: bool = False):
        self.cfg = cfg
        self.debug_run = debug_run
        use_bkg = (cfg.training.model.task == "is_tau") or (
            cfg.training.model.name == "MultiParTau"
        )
        self.sample = "*" if use_bkg else "z"
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.save_hyperparameters()
        super().__init__()

    def _get_pt_paths(self, split: str) -> list[str]:
        pattern = os.path.join(self.cfg.dataset.data_dir, f"{self.sample}_{split}.pt")
        paths = sorted(glob.glob(pattern))
        print(
            f"[ParTDataModule] Found {len(paths)} {split} .pt files: {pattern}",
            flush=True,
        )
        return paths

    def _make_loader(self, tensors: tuple, batch_size: int) -> DataLoader:
        dataset = ParticleTransformerDataset(tensors=tensors, batch_size=batch_size)
        num_workers = (
            0 if self.debug_run else self.cfg.training.dataloader.num_dataloader_workers
        )
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(
                self.cfg.training.dataloader.prefetch_factor
                if num_workers > 0
                else None
            ),
            pin_memory=True,
            multiprocessing_context="forkserver" if num_workers > 1 else None,
        )

    def setup(self, stage: str) -> None:
        batch_size = (
            self.cfg.training.dataloader.batch_size if not self.debug_run else 512
        )
        if stage == "fit":
            all_train_paths = self._get_pt_paths("train")
            total = sum(self.cfg.dataset.relative_sizes[s] for s in ["train", "val"])
            train_frac = self.cfg.dataset.relative_sizes["train"] / total
            train_tensors, val_tensors = _load_and_split(all_train_paths, train_frac)
            # Add the scaling call
            train_tensors, val_tensors = fit_and_apply_input_scaling(
                train_tensors, val_tensors, self.cfg
            )
            self.train_loader = self._make_loader(train_tensors, batch_size)
            self.val_loader = self._make_loader(val_tensors, batch_size)
        elif stage == "test" or stage == "predict":
            test_paths = self._get_pt_paths("test")
            # For test, use all data (no split needed)
            test_tensors, _ = _load_and_split(test_paths, train_frac=1.0)
            # Add the scaling call
            test_tensors = apply_saved_input_scaling_from_cfg(test_tensors, self.cfg)
            self.test_loader = self._make_loader(test_tensors, batch_size)
        else:
            raise ValueError(f"Unexpected stage: {stage}")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_loader
