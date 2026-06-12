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


class ParticleTransformerDataset(IterableDataset):
    def __init__(
        self, row_groups: Sequence[ig.RowGroup], cfg: DictConfig, batch_size: int = 1
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.row_groups = row_groups
        self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        self._file_cache = {}
        print(f"There are {'{:,}'.format(self.num_rows)} jets in the dataset.")

    def __len__(self):
        return math.ceil(self.num_rows / self.batch_size)

    def _get_parquet_file(self, filename):
        if filename not in self._file_cache:
            import pyarrow.parquet as pq

            self._file_cache[filename] = pq.ParquetFile(filename)
        return self._file_cache[filename]

    def build_tensors(self, data: ak.Array):
        tensors = build_tensors_from_data(data, self.cfg.dataset.max_cands)
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

            tensors = self.build_tensors(data)

            if scaling.input_scaling_enabled(self.cfg):
                # Apply saved scaler (usually derived from training set)
                tensors = scaling.apply_saved_input_scaling_from_cfg(tensors, self.cfg)

            N = tensors[0].shape[0]

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
            test_paths = list(glob.glob(test_paths_wcp))
            test_rowgroups = ig.get_row_groups(input_paths=test_paths)
            np.random.shuffle(test_rowgroups)
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
            train_paths_wcp = os.path.join(
                self.cfg.dataset.data_dir, f"{self.sample}_train.parquet"
            )
            train_paths = list(glob.glob(train_paths_wcp))
            all_train_rowgroups = ig.get_row_groups(input_paths=train_paths)
            np.random.shuffle(all_train_rowgroups)
            n_train_rowgroups = int(len(all_train_rowgroups) * fractions["train"])
            train_rowgroups = all_train_rowgroups[:n_train_rowgroups]
            val_rowgroups = all_train_rowgroups[n_train_rowgroups:]
            return train_rowgroups, val_rowgroups
        else:
            return []

    def setup(self, stage: str) -> None:
        # For debug runs, use smaller but reasonable batch size for speed
        batch_size = (
            self.cfg.training.dataloader.batch_size if not self.debug_run else 512
        )
        if stage == "fit":
            train_row_groups, val_row_groups = self.get_dataset_rowgroups(
                dataset_type="train"
            )
            self.train_dataset = ParticleTransformerDataset(
                row_groups=train_row_groups, cfg=self.cfg, batch_size=batch_size
            )
            self.val_dataset = ParticleTransformerDataset(
                row_groups=val_row_groups, cfg=self.cfg, batch_size=batch_size
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
                row_groups=test_row_groups, cfg=self.cfg, batch_size=batch_size
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
