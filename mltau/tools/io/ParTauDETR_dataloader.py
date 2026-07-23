import math

import awkward as ak
import numpy as np
import torch
from torch.utils.data import DataLoader

from mltau.tools.io.ParT_dataloader import ParTDataModule, ParticleTransformerDataset


class ParticleTransformerDETRDataset(ParticleTransformerDataset):
    """
    ParT-style dataset for DETR set-to-set training.

    Inputs are kept identical to ParticleTransformerDataset:
      - cand_features: [N, 17, max_cands]
      - cand_kinematics: [N, 4, max_cands]
      - cand_mask: [N, 1, max_cands]

    Targets are replaced with daughter-level set targets:
      - particles_mask: [N, T] (True for valid daughter)
      - particles_kinematics: [N, T, 5] =
          [log(pt_dau/pt_jet), delta_eta(dau-jet), sin(delta_phi), cos(delta_phi), log(m_dau/m_jet)]
      - particles_charge_ohe: [N, T, 3] one-hot for charges [-1, 0, +1]
      - particles_pdg_ohe: [N, T, N_PDG] one-hot over PDG_CLASS_IDS map

    where T = cfg.dataset.max_tau_daughters if provided, otherwise inferred from
    the currently loaded row-group.
    """

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
        "gen_jet_tau_vis_daughter_p4s",
        "gen_jet_tau_vis_daughter_pdgs",
        "gen_jet_tau_vis_daughter_charges",
        "cls_weight",
    ]

    # Fixed PDG class map for one-hot targets.
    # Charged particle sign is handled by charge target; here we map by abs(PDG).
    PDG_CLASS_IDS = [
        211,
        111,
        321,
        311,
        310,
        130,
        11,
        13,
        22,
        2212,
        2112,
        221,
        323,
        223,
    ]
    PDG_TO_CLASS = {pdg: i for i, pdg in enumerate(PDG_CLASS_IDS)}
    CHARGE_CLASS_VALUES = [-1, 0, 1]
    CHARGE_TO_CLASS = {q: i for i, q in enumerate(CHARGE_CLASS_VALUES)}

    @staticmethod
    def _pad_jagged(arr, max_len: int, fill=0.0, dtype=None):
        out = ak.to_numpy(ak.fill_none(ak.pad_none(arr, max_len, clip=True), fill))
        return out.astype(dtype) if dtype is not None else out

    @staticmethod
    def _get_record_field(record_array, names: list[str]):
        for name in names:
            if name in record_array.fields:
                return record_array[name]
        raise KeyError(
            f"Could not find any of fields {names} in {record_array.fields}."
        )

    def _get_max_tau_daughters(self, n_daughters: np.ndarray) -> int:
        configured = self.cfg.dataset.get("max_tau_daughters", None)
        if configured is not None:
            return int(configured)
        if n_daughters.size == 0:
            return 0
        return int(np.max(n_daughters))

    @classmethod
    def _charges_to_class_indices(cls, raw_charge: np.ndarray) -> np.ndarray:
        out = np.full(raw_charge.shape, -1, dtype=np.int64)
        q = np.rint(raw_charge).astype(np.int64)
        for val, idx in cls.CHARGE_TO_CLASS.items():
            out[q == val] = idx
        return out

    @classmethod
    def _pdg_to_class_indices(cls, raw_pdg: np.ndarray) -> np.ndarray:
        # Map by absolute PDG so that sign is represented by charge target.
        out = np.full(raw_pdg.shape, -1, dtype=np.int64)
        p_abs = np.abs(raw_pdg.astype(np.int64))
        for pdg, idx in cls.PDG_TO_CLASS.items():
            out[p_abs == pdg] = idx
        return out

    def build_tensors(self, data: ak.Array):
        # -------------------------
        # Inputs (unchanged)
        # -------------------------
        max_cands = self.cfg.dataset.max_cands
        eps = 1e-6

        def pad_cand(arr, fill=0.0):
            return self._pad_jagged(arr, max_cands, fill=fill, dtype=np.float32)

        # Candidate-level quantities
        cand_pt = pad_cand(data.reco_cand_p4s["rho"])
        cand_eta = pad_cand(data.reco_cand_p4s["eta"])
        cand_phi = pad_cand(data.reco_cand_p4s["phi"])
        cand_en = pad_cand(data.reco_cand_p4s["t"])
        cand_charge = pad_cand(data.reco_cand_charges)
        cand_pdg_abs = pad_cand(abs(data.reco_cand_pdgs))
        cand_dz = pad_cand(data.reco_cand_dz)
        cand_dz_err = pad_cand(data.reco_cand_dz_error)
        cand_dxy = pad_cand(data.reco_cand_dxy)
        cand_dxy_err = pad_cand(data.reco_cand_dxy_error)

        lengths = np.minimum(ak.to_numpy(ak.num(data.reco_cand_pdgs)), max_cands)
        mask_np = np.arange(max_cands)[None, :] < lengths[:, None]

        # Jet-level p4 for feature engineering and bookkeeping
        jet_pt = ak.to_numpy(data.reco_jet_p4["rho"]).astype(np.float32)
        jet_eta = ak.to_numpy(data.reco_jet_p4["eta"]).astype(np.float32)
        jet_phi = ak.to_numpy(data.reco_jet_p4["phi"]).astype(np.float32)
        jet_en = ak.to_numpy(data.reco_jet_p4["t"]).astype(np.float32)

        gen_tau_pt = ak.to_numpy(data.gen_jet_tau_p4["rho"]).astype(np.float32)
        gen_tau_eta = ak.to_numpy(data.gen_jet_tau_p4["eta"]).astype(np.float32)
        gen_tau_phi = ak.to_numpy(data.gen_jet_tau_p4["phi"]).astype(np.float32)
        gen_tau_energy = ak.to_numpy(data.gen_jet_tau_p4["t"]).astype(np.float32)

        gen_jet_pt = ak.to_numpy(data.gen_jet_p4["rho"]).astype(np.float32)
        gen_jet_eta = ak.to_numpy(data.gen_jet_p4["eta"]).astype(np.float32)
        gen_jet_phi = ak.to_numpy(data.gen_jet_p4["phi"]).astype(np.float32)
        gen_jet_energy = ak.to_numpy(data.gen_jet_p4["t"]).astype(np.float32)

        # 17 ParticleTransformer features
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

        is_electron = (cand_pdg_abs == 11).astype(np.float32)
        is_muon = (cand_pdg_abs == 13).astype(np.float32)
        is_photon = (cand_pdg_abs == 22).astype(np.float32)
        is_charged_hadron = (cand_pdg_abs == 211).astype(np.float32)
        is_neutral_hadron = (cand_pdg_abs == 130).astype(np.float32)

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
                is_electron,
                is_muon,
                is_photon,
                is_charged_hadron,
                is_neutral_hadron,
                cand_dz,
                cand_dz_err,
                cand_dxy,
                cand_dxy_err,
            ],
            axis=1,
        )
        cand_features_np *= mask_np[:, None, :]
        np.nan_to_num(cand_features_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Candidate kinematics [px, py, pz, E]
        cand_px = cand_pt * np.cos(cand_phi)
        cand_py = cand_pt * np.sin(cand_phi)
        cand_pz = cand_pt * np.sinh(cand_eta)
        cand_kinematics_np = np.stack([cand_px, cand_py, cand_pz, cand_en], axis=1)
        cand_kinematics_np *= mask_np[:, None, :]

        # Optional per-jet training weight
        if "cls_weight" not in data.fields:
            weight_tensors = torch.ones(len(data), dtype=torch.float32)
        else:
            weight_tensors = torch.from_numpy(
                ak.to_numpy(data.cls_weight).astype(np.float32)
            )

        # -------------------------
        # DETR set targets
        # -------------------------
        daughter_p4 = data.gen_jet_tau_vis_daughter_p4s
        daughter_pdg_jag = data.gen_jet_tau_vis_daughter_pdgs
        daughter_charge_jag = data.gen_jet_tau_vis_daughter_charges

        daughter_counts = ak.to_numpy(ak.num(daughter_pdg_jag)).astype(np.int64)
        max_tau_daughters = self._get_max_tau_daughters(daughter_counts)

        if max_tau_daughters > 0:
            dau_pt = self._pad_jagged(
                self._get_record_field(daughter_p4, ["pt", "rho"]),
                max_tau_daughters,
                fill=0.0,
                dtype=np.float32,
            )
            dau_eta = self._pad_jagged(
                self._get_record_field(daughter_p4, ["eta"]),
                max_tau_daughters,
                fill=0.0,
                dtype=np.float32,
            )
            dau_phi = self._pad_jagged(
                self._get_record_field(daughter_p4, ["phi"]),
                max_tau_daughters,
                fill=0.0,
                dtype=np.float32,
            )

            if any(name in daughter_p4.fields for name in ["t", "energy", "E", "e"]):
                dau_energy = self._pad_jagged(
                    self._get_record_field(daughter_p4, ["t", "energy", "E", "e"]),
                    max_tau_daughters,
                    fill=0.0,
                    dtype=np.float32,
                )
            else:
                # If only mass is present, reconstruct energy from pt, eta, mass.
                if any(name in daughter_p4.fields for name in ["mass", "m"]):
                    dau_mass = self._pad_jagged(
                        self._get_record_field(daughter_p4, ["mass", "m"]),
                        max_tau_daughters,
                        fill=0.0,
                        dtype=np.float32,
                    )
                else:
                    dau_mass = np.zeros_like(dau_pt, dtype=np.float32)
                dau_energy = np.sqrt(
                    np.maximum((dau_pt * np.cosh(dau_eta)) ** 2 + dau_mass**2, 0.0)
                )

            daughter_charge = self._pad_jagged(
                daughter_charge_jag,
                max_tau_daughters,
                fill=0,
                dtype=np.int64,
            )
            daughter_pdg = self._pad_jagged(
                daughter_pdg_jag,
                max_tau_daughters,
                fill=0,
                dtype=np.int64,
            )

            clipped_counts = np.minimum(daughter_counts, max_tau_daughters)
            daughter_mask_np = (
                np.arange(max_tau_daughters)[None, :] < clipped_counts[:, None]
            )

            dau_px = dau_pt * np.cos(dau_phi)
            dau_py = dau_pt * np.sin(dau_phi)
            dau_pz = dau_pt * np.sinh(dau_eta)
            daughter_p4_np = np.stack([dau_px, dau_py, dau_pz, dau_energy], axis=-1)

            # Daughter kinematic targets in the same spirit as ParT kinematics_tensor.
            _LOG_CLAMP = 5.0
            jet_pt_2d = np.maximum(jet_pt[:, None], eps)
            jet_eta_2d = jet_eta[:, None]
            jet_phi_2d = jet_phi[:, None]
            jet_mass = np.sqrt(
                np.maximum(jet_en**2 - (jet_pt * np.cosh(jet_eta)) ** 2, 0.0)
            )
            jet_mass_2d = np.maximum(jet_mass[:, None], eps)

            daughter_deta = dau_eta - jet_eta_2d
            daughter_dphi_raw = dau_phi - jet_phi_2d
            daughter_dphi = np.arctan2(
                np.sin(daughter_dphi_raw), np.cos(daughter_dphi_raw)
            )
            daughter_log_pt_ratio = np.clip(
                np.log(np.maximum(dau_pt / jet_pt_2d, eps)), -_LOG_CLAMP, _LOG_CLAMP
            )
            daughter_mass = np.sqrt(
                np.maximum(dau_energy**2 - (dau_pt * np.cosh(dau_eta)) ** 2, 0.0)
            )
            daughter_log_mass_ratio = np.clip(
                np.log(np.maximum(daughter_mass / jet_mass_2d, eps)),
                -_LOG_CLAMP,
                _LOG_CLAMP,
            )

            daughter_kinematics_np = np.stack(
                [
                    daughter_log_pt_ratio,
                    daughter_deta,
                    np.sin(daughter_dphi),
                    np.cos(daughter_dphi),
                    daughter_log_mass_ratio,
                ],
                axis=-1,
            )

            daughter_p4_np *= daughter_mask_np[..., None]
            daughter_kinematics_np *= daughter_mask_np[..., None]
            np.nan_to_num(
                daughter_kinematics_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            n_jets = len(data)
            daughter_mask_np = np.zeros((n_jets, 0), dtype=bool)
            daughter_p4_np = np.zeros((n_jets, 0, 4), dtype=np.float32)
            daughter_kinematics_np = np.zeros((n_jets, 0, 5), dtype=np.float32)
            daughter_charge = np.zeros((n_jets, 0), dtype=np.int64)
            daughter_pdg = np.zeros((n_jets, 0), dtype=np.int64)

        charge_cls = self._charges_to_class_indices(daughter_charge)
        pdg_cls = self._pdg_to_class_indices(daughter_pdg)

        # Prepare one-hot targets; unknown classes stay all-zero.
        n_charge = len(self.CHARGE_CLASS_VALUES)
        n_pdg = len(self.PDG_CLASS_IDS)
        charge_ohe = np.zeros((*charge_cls.shape, n_charge), dtype=np.float32)
        pdg_ohe = np.zeros((*pdg_cls.shape, n_pdg), dtype=np.float32)

        valid_charge = charge_cls >= 0
        valid_pdg = pdg_cls >= 0
        if np.any(valid_charge):
            rows, cols = np.where(valid_charge)
            charge_ohe[rows, cols, charge_cls[rows, cols]] = 1.0
        if np.any(valid_pdg):
            rows, cols = np.where(valid_pdg)
            pdg_ohe[rows, cols, pdg_cls[rows, cols]] = 1.0

        # Zero out padded daughters in one-hot tensors too.
        charge_ohe *= daughter_mask_np[..., None]
        pdg_ohe *= daughter_mask_np[..., None]

        targets = {
            "particles_mask": torch.from_numpy(daughter_mask_np).bool(),
            "particles_kinematics": torch.from_numpy(daughter_kinematics_np).float(),
            "particles_charge_ohe": torch.from_numpy(charge_ohe).float(),
            "particles_pdg_ohe": torch.from_numpy(pdg_ohe).float(),
        }

        return (
            torch.from_numpy(cand_features_np),
            torch.from_numpy(cand_kinematics_np),
            targets,
            torch.from_numpy(mask_np).unsqueeze(1),
            weight_tensors.float(),
            {
                "pt": torch.from_numpy(gen_tau_pt),
                "eta": torch.from_numpy(gen_tau_eta),
                "phi": torch.from_numpy(gen_tau_phi),
                "energy": torch.from_numpy(gen_tau_energy),
            },
            {
                "pt": torch.from_numpy(jet_pt),
                "eta": torch.from_numpy(jet_eta),
                "phi": torch.from_numpy(jet_phi),
                "energy": torch.from_numpy(jet_en),
            },
            {
                "pt": torch.from_numpy(gen_jet_pt),
                "eta": torch.from_numpy(gen_jet_eta),
                "phi": torch.from_numpy(gen_jet_phi),
                "energy": torch.from_numpy(gen_jet_energy),
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

        for row_group in row_groups_to_process:
            data = ak.from_parquet(
                row_group.filename,
                row_groups=[row_group.row_group],
                columns=self._NEEDED_COLUMNS,
            )
            tensors = self.build_tensors(data)
            n_rows = tensors[0].shape[0]

            for start in range(0, n_rows, self.batch_size):
                end = min(start + self.batch_size, n_rows)
                yield (
                    tensors[0][start:end],
                    tensors[1][start:end],
                    {k: v[start:end] for k, v in tensors[2].items()},
                    tensors[3][start:end],
                    tensors[4][start:end],
                    {k: v[start:end] for k, v in tensors[5].items()},
                    {k: v[start:end] for k, v in tensors[6].items()},
                    {k: v[start:end] for k, v in tensors[7].items()},
                )


class ParTauDETRDataModule(ParTDataModule):
    """
    DataModule variant using ParticleTransformerDETRDataset.

    File discovery intentionally follows ParTDataModule behavior, i.e.
    `{sample}_train.parquet` and `{sample}_test.parquet` under
    `cfg.dataset.data_dir`.
    """

    def setup(self, stage: str) -> None:
        batch_size = (
            self.cfg.training.dataloader.batch_size if not self.debug_run else 512
        )
        if stage == "fit":
            train_row_groups, val_row_groups = self.get_dataset_rowgroups(
                dataset_type="train"
            )
            self.train_dataset = ParticleTransformerDETRDataset(
                row_groups=train_row_groups, cfg=self.cfg, batch_size=batch_size
            )
            self.val_dataset = ParticleTransformerDETRDataset(
                row_groups=val_row_groups, cfg=self.cfg, batch_size=batch_size
            )
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
            if isinstance(test_row_groups, tuple):
                test_row_groups = test_row_groups[0]
            self.test_dataset = ParticleTransformerDETRDataset(
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


# Backward-compatible alias
ParTDETRDataModule = ParTauDETRDataModule
