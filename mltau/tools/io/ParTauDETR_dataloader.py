import math
import os
import warnings
from collections.abc import Sequence

import awkward as ak
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mltau.tools.io import general as ig

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
      - particles_pdg_ohe: [N, T, N_PDG] one-hot over
          cfg.dataset.tau_daughter_pdg_ids

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
        "gen_jet_tau_decaymode",
        "cls_weight",
    ]

    # Charge classes are a structural constant, not configuration: the model
    # rejects anything but three classes and ParTauDETRModule.predict_step maps
    # them back through a fixed [-1, 0, +1] lookup table.
    CHARGE_CLASS_VALUES = [-1, 0, 1]
    CHARGE_TO_CLASS = {q: i for i, q in enumerate(CHARGE_CLASS_VALUES)}

    @property
    def pdg_class_ids(self) -> list[int]:
        """
        PDG ids defining the one-hot target classes, in class-index order.

        Read from `cfg.dataset.tau_daughter_pdg_ids`, which is the same key
        ParTauDETRModule uses to size its PDG head and to build the lookup table
        in predict_step. Keeping one source of truth means the targets, the head
        width and the decoded predictions cannot silently disagree.
        """
        return [int(x) for x in self.cfg.dataset.tau_daughter_pdg_ids]

    @property
    def pdg_to_class(self) -> dict[int, int]:
        """abs(PDG) -> class index. Sign is carried by the charge target."""
        return {pdg: i for i, pdg in enumerate(self.pdg_class_ids)}

    def __init__(
        self,
        row_groups: Sequence[ig.RowGroup],
        cfg: DictConfig,
        batch_size: int = 1,
        shuffle: bool = False,
        row_groups_per_read: int = 1,
        mixing_reads: int = 1,
        cache_parquet_handles: bool = True,
        num_workers: int = 0,
    ):
        """
        Args:
            shuffle: reshuffle the read order and the jets inside each loaded
                chunk on every epoch.
            row_groups_per_read: number of consecutive row groups pulled in a
                single `ak.from_parquet` call. Each such call re-opens the file
                and re-parses the whole Parquet footer (every row group x every
                column), so with small row groups that fixed cost dominates the
                actual payload and scales as O(n_row_groups^2) per epoch.
                Coalescing divides the number of footer parses by this factor.
            mixing_reads: number of reads held in memory at once. Values > 1 mix
                signal and background into the same batch, at the cost of
                proportionally more worker memory. A read covers one file and
                therefore one class, so with mixing_reads=2 about half of all
                chunks are still single-class; 4 brings that to ~12%.
        """
        super().__init__(row_groups=row_groups, cfg=cfg, batch_size=batch_size)
        self.shuffle = shuffle
        self.cache_parquet_handles = bool(cache_parquet_handles)
        # Filled lazily inside the worker; see _parquet_handle.
        self._handles = None
        self.mixing_reads = max(1, int(mixing_reads))
        # Needed for an exact __len__: batches are counted per worker shard.
        self.num_workers = max(0, int(num_workers))
        self.read_units = self._build_read_units(
            row_groups, max(1, int(row_groups_per_read))
        )
        print(
            f"Grouped {len(row_groups):,} row groups into "
            f"{len(self.read_units):,} parquet read(s)."
        )

    @staticmethod
    def _build_read_units(
        row_groups: Sequence[ig.RowGroup], row_groups_per_read: int
    ) -> list[tuple[str, list[int], int]]:
        """
        Group row groups into (filename, row_group_indices, num_rows) reads.

        Row groups are batched per file in ascending index order, up to
        `row_groups_per_read` each. Contiguity is deliberately NOT required:
        `get_dataset_rowgroups` shuffles and then splits train/val, so a train
        shard is a random ~87% subset whose indices have gaps every ~8 entries.
        Insisting on consecutive runs would cap reads at that gap spacing and
        undo the coalescing entirely. pyarrow accepts an arbitrary index list,
        and ascending order keeps enough locality; the cost being amortised here
        is the per-call Parquet footer parse, not seek time.
        """
        by_file: dict[str, list[ig.RowGroup]] = {}
        for rg in row_groups:
            by_file.setdefault(rg.filename, []).append(rg)

        units: list[tuple[str, list[int], int]] = []
        for filename, groups in by_file.items():
            groups.sort(key=lambda rg: rg.row_group)
            for start in range(0, len(groups), row_groups_per_read):
                block = groups[start : start + row_groups_per_read]
                units.append(
                    (
                        filename,
                        [rg.row_group for rg in block],
                        sum(rg.num_rows for rg in block),
                    )
                )
        return units

    def __len__(self):
        """
        Exact number of batches this dataset yields.

        `__iter__` carries leftover rows across chunks, so a worker emits
        ceil(rows_in_its_shard / batch_size) batches regardless of how the
        reads happen to be grouped or shuffled. Sharding is strided and
        therefore fixed, so this is deterministic.

        Exactness matters beyond cosmetics. With `val_check_interval` unset,
        Lightning sets val_check_batch = len(dataloader) and triggers
        end-of-epoch validation via `(batch_idx + 1) % val_check_batch == 0`,
        overwriting its own is_last_batch default. An over-estimate here means
        that condition never fires and validation is silently skipped forever.
        """
        num_workers = max(1, self.num_workers)
        total = 0
        for worker in range(num_workers):
            rows = sum(
                num_rows
                for _, _, num_rows in self.read_units[worker::num_workers]
            )
            if rows:
                total += math.ceil(rows / self.batch_size)
        return total

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

    def _pdg_to_class_indices(self, raw_pdg: np.ndarray) -> np.ndarray:
        # Map by absolute PDG so that sign is represented by charge target.
        out = np.full(raw_pdg.shape, -1, dtype=np.int64)
        p_abs = np.abs(raw_pdg.astype(np.int64))
        for pdg, idx in self.pdg_to_class.items():
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
        cand_pt = pad_cand(data.reco_cand_p4s["pt"])
        cand_eta = pad_cand(data.reco_cand_p4s["eta"])
        cand_phi = pad_cand(data.reco_cand_p4s["phi"])
        cand_en = pad_cand(data.reco_cand_p4s["energy"])
        cand_charge = pad_cand(data.reco_cand_charges)
        cand_pdg_abs = pad_cand(abs(data.reco_cand_pdgs))
        cand_dz = pad_cand(data.reco_cand_dz)
        cand_dz_err = pad_cand(data.reco_cand_dz_error)
        cand_dxy = pad_cand(data.reco_cand_dxy)
        cand_dxy_err = pad_cand(data.reco_cand_dxy_error)

        lengths = np.minimum(ak.to_numpy(ak.num(data.reco_cand_pdgs)), max_cands)
        mask_np = np.arange(max_cands)[None, :] < lengths[:, None]

        # Jet-level p4 for feature engineering and bookkeeping
        jet_pt = ak.to_numpy(data.reco_jet_p4["pt"]).astype(np.float32)
        jet_eta = ak.to_numpy(data.reco_jet_p4["eta"]).astype(np.float32)
        jet_phi = ak.to_numpy(data.reco_jet_p4["phi"]).astype(np.float32)
        jet_en = ak.to_numpy(data.reco_jet_p4["energy"]).astype(np.float32)

        gen_tau_pt = ak.to_numpy(data.gen_jet_tau_p4["pt"]).astype(np.float32)
        gen_tau_eta = ak.to_numpy(data.gen_jet_tau_p4["eta"]).astype(np.float32)
        gen_tau_phi = ak.to_numpy(data.gen_jet_tau_p4["phi"]).astype(np.float32)
        gen_tau_energy = ak.to_numpy(data.gen_jet_tau_p4["energy"]).astype(np.float32)

        gen_jet_pt = ak.to_numpy(data.gen_jet_p4["pt"]).astype(np.float32)
        gen_jet_eta = ak.to_numpy(data.gen_jet_p4["eta"]).astype(np.float32)
        gen_jet_phi = ak.to_numpy(data.gen_jet_p4["phi"]).astype(np.float32)
        gen_jet_energy = ak.to_numpy(data.gen_jet_p4["energy"]).astype(np.float32)

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

        # Failsafe: background samples (and any jet with no visible daughters)
        # store daughter_p4 as an empty/unknown-typed array with no fields, so
        # there is nothing to extract.  Fall through to the zero-filled branch
        # instead of raising KeyError in _get_record_field.
        if max_tau_daughters > 0 and len(daughter_p4.fields) > 0:
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
            # Keep the daughter axis at max_tau_daughters even with nothing to
            # put in it. A background read that emitted T=0 while a signal read
            # emitted T=8 cannot be concatenated, which breaks batches that mix
            # the two, and would make the target shape depend on which file a
            # batch happened to come from. The all-False mask already tells the
            # criterion that none of these slots carry supervision.
            n_jets = len(data)
            n_slots = max(int(max_tau_daughters), 0)
            daughter_mask_np = np.zeros((n_jets, n_slots), dtype=bool)
            daughter_p4_np = np.zeros((n_jets, n_slots, 4), dtype=np.float32)
            daughter_kinematics_np = np.zeros((n_jets, n_slots, 5), dtype=np.float32)
            daughter_charge = np.zeros((n_jets, n_slots), dtype=np.int64)
            daughter_pdg = np.zeros((n_jets, n_slots), dtype=np.int64)

        charge_cls = self._charges_to_class_indices(daughter_charge)
        pdg_cls = self._pdg_to_class_indices(daughter_pdg)

        # Prepare one-hot targets; unknown classes stay all-zero.
        n_charge = len(self.CHARGE_CLASS_VALUES)
        n_pdg = len(self.pdg_class_ids)
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
            # Jet-level tau-tagging label, following ParticleTransformerDataset:
            # -1 -> no genuine tau (background), >= 0 -> genuine tau (signal).
            "is_tau": torch.from_numpy(
                (ak.to_numpy(data.gen_jet_tau_decaymode) != -1).astype(np.int64)
            ),
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

    def _parquet_handle(self, filename: str):
        """
        Return a cached pyarrow handle for `filename`.

        `ak.from_parquet(path, row_groups=...)` re-opens the file and re-parses
        the entire Parquet footer on every call, which with tens of thousands of
        row groups costs far more than the rows being read. A ParquetFile holds
        the parsed footer, so keeping one per file turns that into a one-off cost
        per worker.

        Handles are opened lazily here rather than in __init__ because __init__
        runs in the parent process and the dataset is pickled out to the workers;
        an open file handle must not cross that boundary.
        """
        import pyarrow.parquet as pq

        if self._handles is None:
            self._handles = {}
        handle = self._handles.get(filename)
        if handle is None:
            handle = pq.ParquetFile(filename)
            self._handles[filename] = handle
        return handle

    def _load_read_unit(self, read_unit):
        filename, row_group_indices, _ = read_unit
        if self.cache_parquet_handles:
            table = self._parquet_handle(filename).read_row_groups(
                row_group_indices, columns=self._NEEDED_COLUMNS
            )
            data = ak.from_arrow(table)
            del table
        else:
            data = ak.from_parquet(
                filename,
                row_groups=row_group_indices,
                columns=self._NEEDED_COLUMNS,
            )
        tensors = self.build_tensors(data)
        del data
        return tensors

    @staticmethod
    def _concat_tensors(parts: list[tuple]):
        """Concatenate several build_tensors() outputs along the jet axis."""
        if len(parts) == 1:
            return parts[0]
        def _cat(tensors, label):
            shapes = {t.shape[1:] for t in tensors}
            if len(shapes) > 1:
                raise RuntimeError(
                    f"Cannot concatenate '{label}' across reads: trailing shapes "
                    f"differ ({sorted(str(s) for s in shapes)}). All reads must "
                    "agree on every axis but the jet axis; check that "
                    "dataset.max_tau_daughters is set so signal and background "
                    "produce the same number of daughter slots."
                )
            return torch.cat(tensors, dim=0)

        out = []
        for field in range(len(parts[0])):
            if isinstance(parts[0][field], dict):
                out.append(
                    {
                        k: _cat([p[field][k] for p in parts], f"{field}.{k}")
                        for k in parts[0][field]
                    }
                )
            else:
                out.append(_cat([p[field] for p in parts], str(field)))
        return tuple(out)

    @staticmethod
    def _take(tensors: tuple, idx):
        return tuple(
            {k: v[idx] for k, v in t.items()} if isinstance(t, dict) else t[idx]
            for t in tensors
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            reads_to_process = list(self.read_units)
        else:
            # Strided instead of contiguous sharding: contiguous slicing with
            # ceil() hands the last worker a short (or empty) shard while the
            # first ones do a full share, so the epoch is paced by the slowest.
            reads_to_process = list(
                self.read_units[worker_info.id :: worker_info.num_workers]
            )

        if self.shuffle:
            np.random.default_rng().shuffle(reads_to_process)

        # A read covers one file, hence one class. Emitting one read at a time
        # therefore yields runs of pure-signal followed by runs of pure-background
        # batches. Draining several reads at once and permuting across them
        # restores a mixed class composition per batch.
        #
        # Rows left over from a chunk are carried into the next one instead of
        # being emitted as a short batch. That keeps every batch full except the
        # last of the shard, which is what makes __len__ exact, and it lets a
        # batch straddle two chunks so the class mixing improves slightly.
        carry = None
        for start_read in range(0, len(reads_to_process), self.mixing_reads):
            chunk = reads_to_process[start_read : start_read + self.mixing_reads]
            tensors = self._concat_tensors([self._load_read_unit(u) for u in chunk])
            if carry is not None:
                tensors = self._concat_tensors([carry, tensors])
                carry = None
            n_rows = tensors[0].shape[0]

            if self.shuffle:
                tensors = self._take(tensors, torch.randperm(n_rows))

            n_full = (n_rows // self.batch_size) * self.batch_size
            for start in range(0, n_full, self.batch_size):
                yield self._take(tensors, slice(start, start + self.batch_size))
            if n_full < n_rows:
                carry = self._take(tensors, slice(n_full, n_rows))

        if carry is not None and carry[0].shape[0] > 0:
            yield carry


def resolve_num_workers(requested: int) -> int:
    """
    Clamp the worker count to the CPUs this process may actually use.

    `os.sched_getaffinity` reflects the Slurm cpuset, so this catches a job that
    asked for one cpu but configured several workers -- they would otherwise
    timeshare a single core and stall the first batch for minutes.
    """
    requested = int(requested)
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        available = os.cpu_count() or 1
    # Leave one core for the main process that feeds the GPU.
    usable = max(1, available - 1) if available > 1 else 1
    if requested > usable:
        warnings.warn(
            f"training.dataloader.num_dataloader_workers={requested} but only "
            f"{available} cpu(s) are available to this process; using {usable}. "
            "Request more cpus (e.g. #SBATCH --cpus-per-task=8) to use more "
            "workers.",
            stacklevel=2,
        )
        return usable
    return requested


class ParTauDETRDataModule(ParTDataModule):
    """
    DataModule variant using ParticleTransformerDETRDataset.

    File discovery intentionally follows ParTDataModule behavior, i.e.
    `{sample}_train.parquet` and `{sample}_test.parquet` under
    `cfg.dataset.data_dir`.
    """

    def __init__(self, cfg: DictConfig, debug_run: bool = False):
        super().__init__(cfg=cfg, debug_run=debug_run)
        # Tau tagging is a binary signal-vs-background task, so we need the
        # background samples in addition to the signal samples. The base class
        # otherwise defaults to signal-only (`sample = "z"`) for set-to-set.
        if cfg.model.detr.tau_id_head:
            self.sample = "*"

    def setup(self, stage: str) -> None:
        batch_size = (
            self.cfg.training.dataloader.batch_size if not self.debug_run else 512
        )
        if stage == "fit":
            train_row_groups, val_row_groups = self.get_dataset_rowgroups(
                dataset_type="train"
            )
            row_groups_per_read = self.cfg.training.dataloader.get(
                "row_groups_per_read", 1
            )
            mixing_reads = self.cfg.training.dataloader.get("mixing_reads", 1)
            cache_handles = self.cfg.training.dataloader.get(
                "cache_parquet_handles", True
            )
            # The dataset needs the count actually used by the DataLoader, so
            # __len__ matches how the shards are really split.
            n_workers = (
                0
                if self.debug_run
                else resolve_num_workers(
                    self.cfg.training.dataloader.num_dataloader_workers
                )
            )
            self.train_dataset = ParticleTransformerDETRDataset(
                row_groups=train_row_groups,
                cfg=self.cfg,
                batch_size=batch_size,
                shuffle=True,
                row_groups_per_read=row_groups_per_read,
                mixing_reads=mixing_reads,
                cache_parquet_handles=cache_handles,
                num_workers=n_workers,
            )
            self.val_dataset = ParticleTransformerDETRDataset(
                row_groups=val_row_groups,
                cfg=self.cfg,
                batch_size=batch_size,
                shuffle=False,
                row_groups_per_read=row_groups_per_read,
                mixing_reads=mixing_reads,
                cache_parquet_handles=cache_handles,
                num_workers=n_workers,
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=None,
                persistent_workers=False if self.debug_run else True,
                num_workers=n_workers,
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
                num_workers=n_workers,
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
                row_groups=test_row_groups,
                cfg=self.cfg,
                batch_size=batch_size,
                shuffle=False,
                row_groups_per_read=self.cfg.training.dataloader.get(
                    "row_groups_per_read", 1
                ),
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=None,
                persistent_workers=True,
                num_workers=resolve_num_workers(
                    self.cfg.training.dataloader.num_dataloader_workers
                ),
                prefetch_factor=(
                    self.cfg.training.dataloader.prefetch_factor
                    if self.cfg.training.dataloader.num_dataloader_workers > 0
                    else None
                ),
                pin_memory=True,
            )
        else:
            raise ValueError(f"Unexpected stage: {stage}")
