import os
import glob
import math
import torch
import numpy as np
import awkward as ak

from collections.abc import Sequence
from torch.utils.data import DataLoader, IterableDataset
from omegaconf import DictConfig
from lightning import LightningDataModule

from mltau.tools.io import general as ig  # RowGroupDataset
from mltau.tools import general as g
from mltau.tools import features as f

np.random.seed(42)


class ParticleTransformerDataset(IterableDataset):
    def __init__(self, row_groups: Sequence[ig.RowGroup], cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.row_groups = row_groups
        self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        print(f"There are {'{:,}'.format(self.num_rows)} jets in the dataset.")

    def __len__(self):
        return self.num_rows

    def _pad_and_convert_to_tensor(
        self, ak_array, dtype=torch.float32, fill_value=0, unsqueeze_dim=None
    ):
        """Helper function to pad awkward arrays and convert to torch tensors."""
        padded_array = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(
                    ak_array,
                    self.cfg.dataset.max_cands,
                    clip=True,
                ),
                fill_value,
            )
        )
        tensor = torch.tensor(padded_array, dtype=dtype)
        if unsqueeze_dim is not None:
            tensor = torch.unsqueeze(tensor, dim=unsqueeze_dim)
        return tensor

    def build_tensors(self, data: ak.Array):
        jet_constituent_p4s = g.reinitialize_p4(data.reco_cand_p4s)
        gen_jet_tau_p4s = g.reinitialize_p4(data.gen_jet_tau_p4)
        jet_p4s = g.reinitialize_p4(data.reco_jet_p4)
        gen_jet_p4s = g.reinitialize_p4(data.gen_jet_p4)

        # ParticleTransformer features from https://arxiv.org/pdf/2202.03772, table 2
        # Add small epsilon to avoid log(0) issues
        eps = 1e-6
        cand_features = ak.Array(
            {
                "cand_deta": f.deltaEta(jet_constituent_p4s.eta, jet_p4s.eta),
                "cand_dphi": f.deltaPhi(jet_constituent_p4s.phi, jet_p4s.phi),
                "cand_logpt": np.log(np.maximum(jet_constituent_p4s.pt, eps)),
                "cand_loge": np.log(np.maximum(jet_constituent_p4s.energy, eps)),
                "cand_logptrel": np.log(
                    np.maximum(jet_constituent_p4s.pt / jet_p4s.pt, eps)
                ),
                "cand_logerel": np.log(
                    np.maximum(jet_constituent_p4s.energy / jet_p4s.energy, eps)
                ),
                "cand_deltaR": f.deltaR_etaPhi(
                    jet_constituent_p4s.eta,
                    jet_constituent_p4s.phi,
                    jet_p4s.eta,
                    jet_p4s.phi,
                ),
                "cand_charge": data.reco_cand_charges,
                "isElectron": ak.values_astype(
                    abs(data.reco_cand_pdgs) == 11, np.float32
                ),
                "isMuon": ak.values_astype(abs(data.reco_cand_pdgs) == 13, np.float32),
                "isPhoton": ak.values_astype(
                    abs(data.reco_cand_pdgs) == 22, np.float32
                ),
                "isChargedHadron": ak.values_astype(
                    abs(data.reco_cand_pdgs) == 211, np.float32
                ),
                "isNeutralHadron": ak.values_astype(
                    abs(data.reco_cand_pdgs) == 130, np.float32
                ),
                "cand_dz": data.reco_cand_dz,
                "cand_dz_error": data.reco_cand_dz_error,
                "cand_dxy": data.reco_cand_dxy,
                "cand_dxy_error": data.reco_cand_dxy_error,
            }
        )

        cand_kinematics = ak.Array(
            {
                "cand_px": jet_constituent_p4s.px,
                "cand_py": jet_constituent_p4s.py,
                "cand_pz": jet_constituent_p4s.pz,
                "cand_en": jet_constituent_p4s.energy,
            }
        )

        if not "cls_weight" in data.fields:
            weight_tensors = torch.tensor(
                ak.ones_like(data.gen_jet_tau_decaymode), dtype=torch.float32
            )
        else:
            weight_tensors = torch.tensor(
                ak.to_numpy(data.cls_weight), dtype=torch.float32
            )

        gen_jet_tau_decaymode = ak.to_numpy(data.gen_jet_tau_decaymode)
        reduced_gen_decay_modes = g.get_reduced_decaymodes(gen_jet_tau_decaymode)
        ohe_prepared_decay_modes = g.prepare_one_hot_encoding(reduced_gen_decay_modes)
        gen_jet_tau_decaymode_reduced = torch.tensor(ohe_prepared_decay_modes).long()
        gen_jet_tau_decaymode_ohe = torch.nn.functional.one_hot(
            gen_jet_tau_decaymode_reduced, 6
        ).float()

        gen_jet_tau_decaymode_exists = (
            torch.tensor(ak.to_numpy(data.gen_jet_tau_decaymode)) != -1
        ).long()

        charge_tensor = (
            torch.tensor(ak.to_numpy(data.gen_jet_tau_charge)) == 1
        ).float()

        # dtheta = f.deltaTheta(gen_jet_tau_p4s.theta, jet_p4s.theta)

        deta = f.signedDeltaEta(gen_jet_tau_p4s.eta, jet_p4s.eta)
        dphi = f.signedDeltaPhi(gen_jet_tau_p4s.phi, jet_p4s.phi)
        # Add epsilon and clamp to avoid log(0) or log(negative)
        vis_pt_ratio = torch.tensor(gen_jet_tau_p4s.pt / jet_p4s.pt)
        vis_pt_ratio_safe = torch.clamp(vis_pt_ratio, min=eps)

        vis_m_ratio = torch.tensor(gen_jet_tau_p4s.mass / jet_p4s.mass)
        vis_m_ratio_safe = torch.clamp(vis_m_ratio, min=eps)
        # Stack kinematic variables into a single tensor [N, 5] for [pT_vis, eta, sin(phi), cos(phi), m_vis]
        kinematics_tensor = torch.stack(
            [
                torch.log(vis_pt_ratio_safe),  # pT_vis (log of pT ratio) - safe version
                torch.tensor(deta),  # delta eta between gen_vis_tau and reco_jet
                torch.sin(
                    torch.tensor(dphi)
                ),  # sin(delta phi) between gen_vis_tau and reco_jet
                torch.cos(
                    torch.tensor(dphi)
                ),  # cos(delta phi) between gen_vis_tau and reco_jet
                torch.log(vis_m_ratio_safe),  # m_vis
            ],
            dim=-1,
        )  # Stack along last dimension to get [N, 5]

        # Pad and convert cand_kinematics to tensor
        cand_kinematics_tensor = torch.stack(
            [
                self._pad_and_convert_to_tensor(cand_kinematics[feat])
                for feat in cand_kinematics.fields
            ],
            dim=-1,
        ).transpose(
            1, 2
        )  # [batch, max_cands, 4] -> [batch, 4, max_cands]

        # Pad and convert cand_features to tensor
        cand_features_tensor = torch.stack(
            [
                self._pad_and_convert_to_tensor(cand_features[feat])
                for feat in cand_features.fields
            ],
            dim=-1,
        ).transpose(
            1, 2
        )  # [batch, max_cands, 13] -> [batch, 13, max_cands]

        # Create padded mask
        mask = self._pad_and_convert_to_tensor(
            ak.ones_like(data.reco_cand_pdgs),
            dtype=torch.bool,
            fill_value=0,
            unsqueeze_dim=1,
        )

        return (
            cand_features_tensor,
            cand_kinematics_tensor,
            {
                "kinematics": kinematics_tensor.float(),
                "decay_mode": gen_jet_tau_decaymode_ohe.float(),
                "charge": charge_tensor.float(),
                "is_tau": gen_jet_tau_decaymode_exists.long(),
            },
            mask,
            weight_tensors.float(),
            gen_jet_tau_p4s,
            jet_p4s,
            gen_jet_p4s,
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

        for row_group in row_groups_to_process:
            # load one chunk from one file
            data = ak.from_parquet(row_group.filename, row_groups=[row_group.row_group])
            tensors = self.build_tensors(data)

            # return individual jets from the dataset
            # Note: Let PyTorch Lightning handle device placement automatically
            for ijet in range(len(data)):
                yield (
                    tensors[0][ijet],  # cand_features
                    tensors[1][ijet],  # cand_kinematics
                    {k: v[ijet] for k, v in tensors[2].items()},  # targets
                    tensors[3][ijet],  # mask
                    tensors[4][ijet],  # weights
                    {
                        field: tensors[5][field][ijet] for field in tensors[5].fields
                    },  # gen_jet_tau_p4s
                    {
                        field: tensors[6][field][ijet] for field in tensors[6].fields
                    },  # reco_jet_p4s
                    {
                        field: tensors[7][field][ijet] for field in tensors[7].fields
                    },  # gen_jet_p4s
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
                row_groups=train_row_groups, cfg=self.cfg
            )
            self.val_dataset = ParticleTransformerDataset(
                row_groups=val_row_groups, cfg=self.cfg
            )
            # Use conservative prefetch_factor to avoid memory issues with IterableDataset
            # IterableDatasets with complex data (awkward arrays) can cause OOM with high prefetch
            safe_prefetch = min(self.cfg.training.dataloader.prefetch_factor, 4)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
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
                prefetch_factor=safe_prefetch,  # Limited to prevent memory explosion
                pin_memory=True,  # Enable for faster GPU transfers
            )
            safe_prefetch = min(self.cfg.training.dataloader.prefetch_factor, 4)
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
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
                prefetch_factor=safe_prefetch,  # Limited to prevent memory explosion
                pin_memory=True,  # Enable for faster GPU transfers
            )
        elif stage == "test":
            test_row_groups = self.get_dataset_rowgroups(dataset_type="test")
            self.test_dataset = ParticleTransformerDataset(
                row_groups=test_row_groups, cfg=self.cfg
            )
            safe_prefetch = min(self.cfg.training.dataloader.prefetch_factor, 4)
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                persistent_workers=True,
                num_workers=self.cfg.training.dataloader.num_dataloader_workers,
                prefetch_factor=safe_prefetch,  # Limited to prevent memory explosion
                pin_memory=True,  # Enable for faster GPU transfers
            )
        else:
            raise ValueError(f"Unexpected stage: {stage}")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
