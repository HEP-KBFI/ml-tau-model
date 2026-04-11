import os
import glob
import math
import torch

from torch.utils.data import DataLoader, IterableDataset
from omegaconf import DictConfig
from lightning import LightningDataModule


class ParticleTransformerDataset(IterableDataset):
    """Wraps a pre-loaded slice of jet tensors.  Workers share the same
    physical memory via share_memory_(); no file I/O happens in __iter__."""

    def __init__(self, tensors: tuple, batch_size: int):
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

    def __len__(self):
        return math.ceil(self.cand_features.shape[0] / self.batch_size)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        N = self.cand_features.shape[0]
        # Strided partition — each worker gets every num_workers-th index.
        # Because the dataset is pre-shuffled (see _load_and_split), consecutive
        # indices are already mixed z/qq, so the stride preserves that mix.
        worker_indices = torch.arange(worker_id, N, num_workers)
        # Per-epoch local shuffle within this worker's slice.
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
            self.train_loader = self._make_loader(train_tensors, batch_size)
            self.val_loader = self._make_loader(val_tensors, batch_size)
        elif stage == "test":
            test_paths = self._get_pt_paths("test")
            # For test, use all data (no split needed)
            test_tensors, _ = _load_and_split(test_paths, train_frac=1.0)
            self.test_loader = self._make_loader(test_tensors, batch_size)
        else:
            raise ValueError(f"Unexpected stage: {stage}")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
