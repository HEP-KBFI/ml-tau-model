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
    def __init__(
        self, row_groups: Sequence[ig.RowGroup], cfg: DictConfig, batch_size: int = 1
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.row_groups = row_groups
        self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        print(f"There are {'{:,}'.format(self.num_rows)} jets in the dataset.")

    def __len__(self):
        return math.ceil(self.num_rows / self.batch_size)

    def build_tensors(self, data: ak.Array):
        max_cands = self.cfg.dataset.max_cands
        eps = 1e-6

        # ------------------------------------------------------------------
        # Helper: pad jagged awkward array → dense float32 [N, max_cands]
        # ------------------------------------------------------------------
        def pad_cand(arr, fill=0.0):
            return ak.to_numpy(
                ak.fill_none(ak.pad_none(arr, max_cands, clip=True), fill)
            ).astype(np.float32)

        # ------------------------------------------------------------------
        # Candidate p4 components: stored as (rho=pt, eta, phi, t=energy)
        # All other candidate fields — one padded extraction each
        # ------------------------------------------------------------------
        cand_pt = pad_cand(data.reco_cand_p4s["rho"])  # [N, max_cands]
        cand_eta = pad_cand(data.reco_cand_p4s["eta"])
        cand_phi = pad_cand(data.reco_cand_p4s["phi"])
        cand_en = pad_cand(data.reco_cand_p4s["t"])  # energy
        cand_charge = pad_cand(data.reco_cand_charges)
        cand_pdg_abs = pad_cand(abs(data.reco_cand_pdgs))
        cand_dz = pad_cand(data.reco_cand_dz)
        cand_dz_err = pad_cand(data.reco_cand_dz_error)
        cand_dxy = pad_cand(data.reco_cand_dxy)
        cand_dxy_err = pad_cand(data.reco_cand_dxy_error)

        # Mask: True = real particle, False = padding  [N, max_cands]
        lengths = np.minimum(ak.to_numpy(ak.num(data.reco_cand_pdgs)), max_cands)
        mask_np = np.arange(max_cands)[None, :] < lengths[:, None]

        # Scalar jet p4s — read raw fields directly, no reinitialize_p4
        jet_pt = ak.to_numpy(data.reco_jet_p4["rho"]).astype(np.float32)  # [N]
        jet_eta = ak.to_numpy(data.reco_jet_p4["eta"]).astype(np.float32)
        jet_phi = ak.to_numpy(data.reco_jet_p4["phi"]).astype(np.float32)
        jet_en = ak.to_numpy(data.reco_jet_p4["t"]).astype(np.float32)

        _pt_gen = ak.to_numpy(data.gen_jet_tau_p4["rho"]).astype(np.float32)
        _eta_gen = ak.to_numpy(data.gen_jet_tau_p4["eta"]).astype(np.float32)
        _phi_gen = ak.to_numpy(data.gen_jet_tau_p4["phi"]).astype(np.float32)
        _energy_gen = ak.to_numpy(data.gen_jet_tau_p4["t"]).astype(np.float32)

        _pt_gen_jet = ak.to_numpy(data.gen_jet_p4["rho"]).astype(np.float32)
        _eta_gen_jet = ak.to_numpy(data.gen_jet_p4["eta"]).astype(np.float32)
        _phi_gen_jet = ak.to_numpy(data.gen_jet_p4["phi"]).astype(np.float32)
        _energy_gen_jet = ak.to_numpy(data.gen_jet_p4["t"]).astype(np.float32)

        # ------------------------------------------------------------------
        # Compute 17 ParticleTransformer features in numpy (zero awkward)
        # ParticleTransformer features from https://arxiv.org/pdf/2202.03772, table 2
        # Broadcast jet scalars [N] → [N, 1] against candidates [N, max_cands]
        # ------------------------------------------------------------------
        jpt = jet_pt[:, None]
        jeta = jet_eta[:, None]
        jphi = jet_phi[:, None]
        jen = jet_en[:, None]

        cand_deta = np.abs(cand_eta - jeta)
        dphi_raw = cand_phi - jphi
        cand_dphi = np.abs(np.arctan2(np.sin(dphi_raw), np.cos(dphi_raw)))
        cand_logpt = np.log(np.maximum(cand_pt, eps))
        cand_loge = np.log(np.maximum(cand_en, eps))
        cand_logptrel = np.log(np.maximum(cand_pt / np.maximum(jpt, eps), eps))
        cand_logerel = np.log(np.maximum(cand_en / np.maximum(jen, eps), eps))
        cand_dR = np.sqrt(cand_deta**2 + cand_dphi**2)

        isElectron = (cand_pdg_abs == 11).astype(np.float32)
        isMuon = (cand_pdg_abs == 13).astype(np.float32)
        isPhoton = (cand_pdg_abs == 22).astype(np.float32)
        isChargedHadron = (cand_pdg_abs == 211).astype(np.float32)
        isNeutralHadron = (cand_pdg_abs == 130).astype(np.float32)

        # Stack → [N, 17, max_cands], zero padded slots, fix nan/inf
        cand_features_np = np.stack(
            [
                cand_deta,
                cand_dphi,
                cand_logpt,
                cand_loge,
                cand_logptrel,
                cand_logerel,
                cand_dR,
                cand_charge,
                isElectron,
                isMuon,
                isPhoton,
                isChargedHadron,
                isNeutralHadron,
                cand_dz,
                cand_dz_err,
                cand_dxy,
                cand_dxy_err,
            ],
            axis=1,
        )  # [N, 17, max_cands]
        cand_features_np *= mask_np[:, None, :]
        np.nan_to_num(cand_features_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Cand kinematics: (px, py, pz, energy) → [N, 4, max_cands]
        cand_px = cand_pt * np.cos(cand_phi)
        cand_py = cand_pt * np.sin(cand_phi)
        cand_pz = cand_pt * np.sinh(cand_eta)
        cand_kinematics_np = np.stack([cand_px, cand_py, cand_pz, cand_en], axis=1)
        cand_kinematics_np *= mask_np[:, None, :]

        # ------------------------------------------------------------------
        # Weights, decay mode, charge
        # ------------------------------------------------------------------
        if "cls_weight" not in data.fields:
            weight_tensors = torch.ones(len(data), dtype=torch.float32)
        else:
            weight_tensors = torch.from_numpy(
                ak.to_numpy(data.cls_weight).astype(np.float32)
            )

        gen_jet_tau_decaymode = ak.to_numpy(data.gen_jet_tau_decaymode)
        reduced_gen_decay_modes = g.get_reduced_decaymodes(gen_jet_tau_decaymode)
        ohe_prepared_decay_modes = g.prepare_one_hot_encoding(reduced_gen_decay_modes)
        gen_jet_tau_decaymode_reduced = torch.from_numpy(
            ohe_prepared_decay_modes.astype(np.int64)
        )
        gen_jet_tau_decaymode_ohe = torch.nn.functional.one_hot(
            gen_jet_tau_decaymode_reduced, 6
        ).float()
        gen_jet_tau_decaymode_exists = torch.from_numpy(
            (gen_jet_tau_decaymode != -1).astype(np.int64)
        )
        charge_tensor = torch.from_numpy(
            (ak.to_numpy(data.gen_jet_tau_charge).astype(np.int32) == 1).astype(
                np.float32
            )
        )

        # ------------------------------------------------------------------
        # Kinematics regression targets (pure numpy, no reinitialize_p4)
        # ------------------------------------------------------------------
        _deta = _eta_gen - jet_eta
        _dphi_raw = _phi_gen - jet_phi
        _dphi = np.arctan2(np.sin(_dphi_raw), np.cos(_dphi_raw))
        _vis_pt_ratio = np.maximum(_pt_gen / np.maximum(jet_pt, eps), eps)
        # m^2 = E^2 - pt^2 * cosh^2(eta)
        _mass_gen = np.sqrt(
            np.maximum(_energy_gen**2 - (_pt_gen * np.cosh(_eta_gen)) ** 2, 0.0)
        )
        _mass_reco = np.sqrt(
            np.maximum(jet_en**2 - (jet_pt * np.cosh(jet_eta)) ** 2, 0.0)
        )
        _vis_m_ratio = np.maximum(_mass_gen / np.maximum(_mass_reco, eps), eps)
        # Clamp log-ratio targets to ±5 (≈ factor-of-150 correction).
        # Without this, massless reco jets give log(_mass_gen/eps) ≈ 14, which
        # dominates the loss and causes GradNorm to suppress the kin head weight.
        _LOG_CLAMP = 5.0
        kinematics_tensor = torch.from_numpy(
            np.stack(
                [
                    np.clip(np.log(_vis_pt_ratio), -_LOG_CLAMP, _LOG_CLAMP),
                    _deta,
                    np.sin(_dphi),
                    np.cos(_dphi),
                    np.clip(np.log(_vis_m_ratio), -_LOG_CLAMP, _LOG_CLAMP),
                ],
                axis=-1,
            )
        )

        return (
            torch.from_numpy(cand_features_np),
            torch.from_numpy(cand_kinematics_np),
            {
                "kinematics": kinematics_tensor.float(),
                "decay_mode": gen_jet_tau_decaymode_ohe.float(),
                "charge": charge_tensor.float(),
                "is_tau": gen_jet_tau_decaymode_exists.long(),
            },
            torch.from_numpy(mask_np).unsqueeze(1),  # [N, 1, max_cands]
            weight_tensors.float(),
            {
                "pt": torch.from_numpy(_pt_gen),
                "eta": torch.from_numpy(_eta_gen),
                "phi": torch.from_numpy(_phi_gen),
                "energy": torch.from_numpy(_energy_gen),
            },
            {
                "pt": torch.from_numpy(jet_pt),
                "eta": torch.from_numpy(jet_eta),
                "phi": torch.from_numpy(jet_phi),
                "energy": torch.from_numpy(jet_en),
            },
            {
                "pt": torch.from_numpy(_pt_gen_jet),
                "eta": torch.from_numpy(_eta_gen_jet),
                "phi": torch.from_numpy(_phi_gen_jet),
                "energy": torch.from_numpy(_energy_gen_jet),
            },
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

        for row_group in row_groups_to_process:
            data = ak.from_parquet(
                row_group.filename,
                row_groups=[row_group.row_group],
                columns=_NEEDED_COLUMNS,
            )
            tensors = self.build_tensors(data)
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
                persistent_workers=True,
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
