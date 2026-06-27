import os
import glob
import math
import sys
import torch
import numpy as np
import awkward as ak

from pathlib import Path
from collections.abc import Sequence
from torch.utils.data import DataLoader, IterableDataset
from omegaconf import DictConfig
from lightning import LightningDataModule

from mltau.tools.io import general as ig  # RowGroupDataset
from mltau.tools import general as g
from mltau.tools import features as f
from mltau.tools.io import scaling

# Add ml-tau-data submodule to path to allow importing from ntupelizer
submodule_path = Path(__file__).resolve().parents[3] / "ml-tau-data"
if str(submodule_path) not in sys.path:
    sys.path.insert(0, str(submodule_path))

from ntupelizer.scripts.preprocess_torch import build_tensors_from_data

np.random.seed(42)


def build_tensors(data: ak.Array, cfg: DictConfig):
    # Sort candidates by pT if requested.
    # This is critical for MLP-Mixer which is not permutation invariant.
    if cfg.dataset.get("sort_by_pt", False):
        # Ensure we can access .pt
        p4 = g.reinitialize_p4(data.reco_cand_p4s)
        indices = ak.argsort(p4.pt, ascending=False)
        
        # Consistent sorting for all constituent-related fields
        # Note: 'reco_jet_p4', 'gen_jet_tau_p4', 'gen_jet_p4', 'gen_jet_tau_decaymode', 
        # 'gen_jet_tau_charge', 'cls_weight' are per-jet and don't need sorting.
        data = ak.Array({
            "reco_cand_p4s": data.reco_cand_p4s[indices],
            "reco_cand_charges": data.reco_cand_charges[indices],
            "reco_cand_pdgs": data.reco_cand_pdgs[indices],
            "reco_cand_dz": data.reco_cand_dz[indices],
            "reco_cand_dz_error": data.reco_cand_dz_error[indices],
            "reco_cand_dxy": data.reco_cand_dxy[indices],
            "reco_cand_dxy_error": data.reco_cand_dxy_error[indices],
            "reco_jet_p4": data.reco_jet_p4,
            "gen_jet_tau_p4": data.gen_jet_tau_p4,
            "gen_jet_p4": data.gen_jet_p4,
            "gen_jet_tau_decaymode": data.gen_jet_tau_decaymode,
            "gen_jet_tau_charge": data.gen_jet_tau_charge,
            "cls_weight": data.cls_weight,
        })

    tensors = build_tensors_from_data(data, cfg.dataset.max_cands)
    # Transpose to [N, C, P] to match model expectations and preprocessed dataloader
    return (
        tensors[0].transpose(1, 2),
        tensors[1].transpose(1, 2),
        tensors[2],
        tensors[3],
        tensors[4],
        tensors[5],
        tensors[6],
        tensors[7],
    )


class ParticleTransformerDataset(IterableDataset):
    def __init__(
        self,
        row_groups: Sequence[ig.RowGroup],
        cfg: DictConfig,
        batch_size: int = 1,
        shuffle: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.row_groups = row_groups
        self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        self.shuffle = shuffle
        self._file_cache = {}

        # Pre-load scaler once if enabled
        self.scaler = None
        if scaling.input_scaling_enabled(self.cfg):
            try:
                self.scaler = scaling.load_saved_scaler(self.cfg)
            except RuntimeError as e:
                # If fitting is expected to happen elsewhere or we're in setup,
                # we might not have it yet.
                print(f"[Warning] {e}")

        print(f"There are {'{:,}'.format(self.num_rows)} jets in the dataset.")

    def __len__(self):
        return math.ceil(self.num_rows / self.batch_size)

    def _get_parquet_file(self, filename):
        if filename not in self._file_cache:
            import pyarrow.parquet as pq

            self._file_cache[filename] = pq.ParquetFile(filename)
        return self._file_cache[filename]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            row_groups_to_process = self.row_groups
        else:
            per_worker = int(
                math.ceil(float(len(self.row_groups)) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            row_groups_start = worker_id * per_worker
            row_groups_end = row_groups_start + per_worker
            row_groups_to_process = self.row_groups[row_groups_start:row_groups_end]

        # Only load columns actually used by build_tensors
        _NEEDED_COLUMNS = [
            "reco_cand_p4s",
            "reco_cand_charges",
            "reco_cand_pdgs",
            "reco_cand_dz",
            "reco_cand_dz_error",
            "reco_cand_dxy",
            "reco_cand_dxy_error",
            "reco_jet_p4",
            "gen_jet_tau_p4",
            "gen_jet_p4",
            "gen_jet_tau_decaymode",
            "gen_jet_tau_charge",
            "cls_weight",
        ]

        # Batch row group reading to amortize metadata parsing overhead
        row_groups_per_read = self.cfg.training.dataloader.get("row_groups_per_read", 16)

        for i in range(0, len(row_groups_to_process), row_groups_per_read):
            chunk = row_groups_to_process[i : i + row_groups_per_read]

            # Group by filename to utilize Parquet batch reading
            from collections import defaultdict

            file_batches = defaultdict(list)
            for rg in chunk:
                file_batches[rg.filename].append(rg.row_group)

            all_data = []
            for filename, indices in file_batches.items():
                pf = self._get_parquet_file(filename)
                table = pf.read_row_groups(indices, columns=_NEEDED_COLUMNS)
                data = ak.from_arrow(table)
                all_data.append(data)

            if len(all_data) > 1:
                data = ak.concatenate(all_data)
            else:
                data = all_data[0]

            tensors = build_tensors(data, self.cfg)

            if self.scaler is not None:
                tensors = scaling.apply_scaler(tensors, self.scaler)

            N = tensors[0].shape[0]

            if self.shuffle:
                # Intra-chunk shuffle: ensures that even if a read chunk contains both signal
                # and background (from different row groups), they are mixed within batches.
                perm = torch.randperm(N)

                def _apply_perm(t):
                    if isinstance(t, torch.Tensor):
                        return t[perm]
                    if isinstance(t, dict):
                        return {k: v[perm] for k, v in t.items()}
                    return t

                tensors = tuple(_apply_perm(t) for t in tensors)

            # Yield pre-batched slices — bypasses PyTorch per-sample collation entirely
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                yield (
                    tensors[0][start:end],  # cand_features
                    tensors[1][start:end],  # cand_kinematics
                    {k: v[start:end] for k, v in tensors[2].items()},  # targets
                    tensors[3][start:end],  # mask
                    tensors[4][start:end],  # weights
                    {k: v[start:end] for k, v in tensors[5].items()},  # gen_jet_tau_p4s
                    {k: v[start:end] for k, v in tensors[6].items()},  # reco_jet_p4s
                    {k: v[start:end] for k, v in tensors[7].items()},  # gen_jet_p4s
                )


class ParTDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        debug_run: bool = False,
    ):
        """Base data module class to be used for different types of trainings.
        Parameters:
            cfg : DictConfig
                The configuration file used to set up the data module.

        """
        self.cfg = cfg
        use_bkg = (cfg.training.model.task == "is_tau") or (
            cfg.training.model.name == "MultiParTau"
        )
        self.debug_run = debug_run
        self.sample = "z" if not use_bkg else "*"
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.num_row_groups = 2 if debug_run else None
        self.save_hyperparameters()
        super().__init__()

    def get_dataset_rowgroups(self, dataset_type: str):
        if dataset_type == "test":
            test_paths_wcp = os.path.join(
                self.cfg.dataset.data_dir, f"{self.sample}_test.parquet"
            )
            test_paths = sorted(list(glob.glob(test_paths_wcp)))
            test_rowgroups = ig.get_row_groups(input_paths=test_paths)
            return test_rowgroups
        elif dataset_type == "train":
            total = sum(
                [
                    self.cfg.dataset.relative_sizes[dataset]
                    for dataset in ["train", "val"]
                ]
            )
            fractions = {
                dataset: self.cfg.dataset.relative_sizes[dataset] / total
                for dataset in ["train", "val"]
            }
            
            # Separate paths by signal (z) and background (qq) to allow interleaving
            train_paths_wcp = os.path.join(
                self.cfg.dataset.data_dir, f"{self.sample}_train.parquet"
            )
            all_train_paths = sorted(list(glob.glob(train_paths_wcp)))
            
            sig_paths = [p for p in all_train_paths if "z_train" in os.path.basename(p)]
            bkg_paths = [p for p in all_train_paths if "qq_train" in os.path.basename(p)]
            other_paths = [p for p in all_train_paths if p not in sig_paths and p not in bkg_paths]

            def _get_and_split(paths):
                rgs = ig.get_row_groups(input_paths=paths)
                np.random.shuffle(rgs)
                n_train = int(len(rgs) * fractions["train"])
                return rgs[:n_train], rgs[n_train:]

            sig_train, sig_val = _get_and_split(sig_paths) if sig_paths else ([], [])
            bkg_train, bkg_val = _get_and_split(bkg_paths) if bkg_paths else ([], [])
            other_train, other_val = _get_and_split(other_paths) if other_paths else ([], [])

            def _interleave(sig, bkg, other):
                combined = []
                # Calculate ratio to maintain balance throughout the dataset
                ratio = len(bkg) // len(sig) if sig else 0
                isig, ibkg = 0, 0
                while isig < len(sig) or ibkg < len(bkg):
                    if isig < len(sig):
                        combined.append(sig[isig])
                        isig += 1
                    for _ in range(ratio):
                        if ibkg < len(bkg):
                            combined.append(bkg[ibkg])
                            ibkg += 1
                combined.extend(bkg[ibkg:])
                combined.extend(other)
                return combined

            return _interleave(sig_train, bkg_train, other_train), _interleave(sig_val, bkg_val, other_val)
        else:
            return []

    def fit_scaler(self, row_groups: Sequence[ig.RowGroup], n_row_groups: int = 200):
        print(
            f"[input scaling] Fitting scaler on a sample of {n_row_groups} row groups..."
        )
        sample_rgs = row_groups[:n_row_groups]

        # Group by filename
        from collections import defaultdict

        file_batches = defaultdict(list)
        for rg in sample_rgs:
            file_batches[rg.filename].append(rg.row_group)

        all_cf = []
        all_msk = []

        _NEEDED_COLUMNS = [
            "reco_cand_p4s",
            "reco_cand_charges",
            "reco_cand_pdgs",
            "reco_cand_dz",
            "reco_cand_dz_error",
            "reco_cand_dxy",
            "reco_cand_dxy_error",
            "reco_jet_p4",
            "gen_jet_tau_p4",
            "gen_jet_p4",
            "gen_jet_tau_decaymode",
            "gen_jet_tau_charge",
            "cls_weight",
        ]

        for filename, indices in file_batches.items():
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(filename)
            table = pf.read_row_groups(indices, columns=_NEEDED_COLUMNS)
            data = ak.from_arrow(table)
            tensors = build_tensors(data, self.cfg)
            all_cf.append(tensors[0])
            all_msk.append(tensors[3])

        cf = torch.cat(all_cf, dim=0)
        msk = torch.cat(all_msk, dim=0)

        feature_indices = scaling.get_feature_indices(self.cfg)
        mean, std = scaling.fit_feature_scaler(cf, msk, feature_indices)

        path = scaling.get_scaler_path(self.cfg)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            mean=mean,
            std=std,
            feature_indices=np.asarray(feature_indices, dtype=np.int64),
            feature_names=scaling._CAND_FEATURE_NAMES,
        )
        print(f"[input scaling] Saved scaler to {path}")

    def setup(self, stage: str) -> None:
        # For debug runs, use smaller but reasonable batch size for speed
        batch_size = (
            self.cfg.training.dataloader.batch_size if not self.debug_run else 512
        )
        if stage == "fit":
            # distill.py initializes the data module eagerly so scaler
            # statistics exist before the model's on_fit_start hook. Lightning
            # calls setup("fit") again; retain the same split and loaders.
            if self.train_loader is not None and self.val_loader is not None:
                return

            train_row_groups, val_row_groups = self.get_dataset_rowgroups(
                dataset_type="train"
            )

            if scaling.input_scaling_enabled(self.cfg):
                try:
                    scaling.load_saved_scaler(self.cfg)
                except RuntimeError:
                    self.fit_scaler(train_row_groups)

            self.train_dataset = ParticleTransformerDataset(
                row_groups=train_row_groups,
                cfg=self.cfg,
                batch_size=batch_size,
                shuffle=True,
            )
            self.val_dataset = ParticleTransformerDataset(
                row_groups=val_row_groups,
                cfg=self.cfg,
                batch_size=batch_size,
                shuffle=True,
            )
            # batch_size=None: dataset yields pre-batched slices, skip collation entirely
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=None,
                persistent_workers=False if self.debug_run else True,
                num_workers=(
                    0
                    if self.debug_run
                    else self.cfg.training.dataloader.num_dataloader_workers
                ),
                multiprocessing_context=(
                    "forkserver"
                    if self.cfg.training.dataloader.num_dataloader_workers > 1
                    else None
                ),
                prefetch_factor=(
                    None
                    if self.debug_run
                    else self.cfg.training.dataloader.prefetch_factor
                ),
                pin_memory=True,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=None,
                persistent_workers=False if self.debug_run else True,
                num_workers=(
                    0
                    if self.debug_run
                    else self.cfg.training.dataloader.num_dataloader_workers
                ),
                multiprocessing_context=(
                    "forkserver"
                    if self.cfg.training.dataloader.num_dataloader_workers > 1
                    else None
                ),
                prefetch_factor=(
                    None
                    if self.debug_run
                    else self.cfg.training.dataloader.prefetch_factor
                ),
                pin_memory=True,
            )
        elif stage == "test" or stage == "predict":
            test_row_groups = self.get_dataset_rowgroups(dataset_type="test")
            self.test_dataset = ParticleTransformerDataset(
                row_groups=test_row_groups, cfg=self.cfg, batch_size=batch_size, shuffle=True
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=None,
                persistent_workers=self.cfg.training.dataloader.num_dataloader_workers > 0,
                num_workers=self.cfg.training.dataloader.num_dataloader_workers,
                prefetch_factor=(
                    self.cfg.training.dataloader.prefetch_factor
                    if self.cfg.training.dataloader.num_dataloader_workers > 0
                    else None
                ),
                pin_memory=True,
            )
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
