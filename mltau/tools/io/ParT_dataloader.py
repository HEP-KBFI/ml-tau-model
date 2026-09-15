import glob
import math
import os
import warnings
from collections.abc import Sequence

import awkward as ak
import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset

from mltau.tools import features as f
from mltau.tools import general as g
from mltau.tools.io import general as ig  # RowGroupDataset

np.random.seed(42)

# Four-vectors reach us under several different field namings depending on how
# the ntuple was written: vector's Momentum4D storage uses (rho, eta, phi, t),
# other producers use (pt, eta, phi, energy), and some write a Cartesian
# (px, py, pz, energy) record instead. Awkward's `arr["pt"]` looks up a FIELD,
# not vector's `pt` property, so the wrong naming raises FieldNotFoundError
# rather than being resolved by the behaviour.
#
# ParT_dataloader used to hardcode rho/t and ParTauDETR_dataloader pt/energy, so
# whichever layout the data had, one of them broke. Go through p4_field instead.
P4_FIELD_ALIASES = {
    "pt": ("pt", "rho"),
    "eta": ("eta",),
    "phi": ("phi",),
    "energy": ("energy", "t", "E", "e"),
    "mass": ("mass", "m", "tau"),
}


def has_p4_field(record_array, quantity: str) -> bool:
    """True if `quantity` is stored outright under one of its aliases.

    Only asks about stored fields, so it stays a cheap question about the
    layout: a Cartesian record has no `pt` field even though p4_field can hand
    one back.
    """
    fields = record_array.fields
    return any(alias in fields for alias in P4_FIELD_ALIASES[quantity])


def p4_field(record_array, quantity: str):
    """Read `quantity` from a p4 record, whatever layout it was written in.

    A stored field wins, because reading one is free and exact. Otherwise the
    record is in a basis that does not carry `quantity` at all -- Cartesian
    (px, py, pz, energy) has none of pt/eta/phi, and a (pt, eta, phi, mass)
    record has no energy -- so hand it to vector via `reinitialize_p4`, which
    picks whichever complete basis is present and derives the rest.
    """
    aliases = P4_FIELD_ALIASES[quantity]
    fields = record_array.fields
    for alias in aliases:
        if alias in fields:
            return record_array[alias]

    try:
        return getattr(g.reinitialize_p4(record_array), quantity)
    except Exception as exc:
        raise KeyError(
            f"no field for {quantity!r} in p4 record: tried {list(aliases)} and "
            f"deriving it from the stored basis; record has {fields}"
        ) from exc


def sample_name(path: str) -> str:
    """
    Sample label from a `{sample}_train*.parquet` / `{sample}_test*.parquet` path.

    Module level rather than a DataModule method because the dataset needs the
    same rule: a read covers one file and therefore one sample, and the batch
    composition is built on knowing which.
    """
    base = os.path.basename(path)
    for split in ("_train", "_test"):
        if split in base:
            return base.split(split)[0]
    return os.path.splitext(base)[0]


def loader_kwargs(num_workers: int, prefetch_factor, debug_run: bool) -> dict:
    """
    DataLoader arguments that are only valid for a given worker count.

    With num_workers=0 PyTorch rejects both `persistent_workers=True` and
    `prefetch_factor`, so passing them unconditionally made
    `num_dataloader_workers=0` raise. That is the one configuration that
    isolates worker startup from everything else, so it has to work -- it is
    the first thing to try when a job hangs before the first batch.

    `multiprocessing_context` is decided from the RESOLVED worker count, not the
    configured one: a request of 6 clamped to 1 by the cpuset should not still
    select forkserver.
    """
    kwargs = {"num_workers": int(num_workers), "pin_memory": True}
    if num_workers > 0:
        kwargs["persistent_workers"] = not debug_run
        kwargs["prefetch_factor"] = None if debug_run else prefetch_factor
        if num_workers > 1:
            kwargs["multiprocessing_context"] = "forkserver"
    return kwargs


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

class ParticleTransformerDataset(IterableDataset):
    def __init__(
        self, row_groups: Sequence[ig.RowGroup], cfg: DictConfig, batch_size: int = 1
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.row_groups = row_groups
        self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        if self.row_groups:
            print(
                f"There are {'{:,}'.format(self.num_rows)} jets in the dataset.",
                flush=True,
            )

    @classmethod
    def for_arrays(cls, cfg: DictConfig):
        """
        Dataset bound to `cfg` alone, for running `build_tensors` on arrays that
        are already in memory.

        Inference reads its parquet file itself and needs only the
        tensor-building half of the dataset, so there are no row groups to plan
        reads over. Constructing that with `row_groups=[]` used to make the
        dataset announce "There are 0 jets in the dataset" about a file it had
        just read in full, which reads like data loss and is why the counts are
        now printed only when there is actually something to read.
        """
        return cls(row_groups=[], cfg=cfg, batch_size=1)

    def __len__(self):
        # A batch never spans two row groups, so every row group contributes its
        # own trailing partial batch. Using ceil(num_rows / batch_size) here
        # under-counts, which makes the progress bar wrong and -- worse -- makes
        # Trainer.estimated_stepping_batches too small, so OneCycleLR runs out of
        # schedule and raises part-way through training.
        return sum(
            math.ceil(rg.num_rows / self.batch_size) for rg in self.row_groups
        )

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
        cand_pt = pad_cand(p4_field(data.reco_cand_p4s, "pt"))  # [N, max_cands]
        cand_eta = pad_cand(p4_field(data.reco_cand_p4s, "eta"))
        cand_phi = pad_cand(p4_field(data.reco_cand_p4s, "phi"))
        cand_en = pad_cand(p4_field(data.reco_cand_p4s, "energy"))  # energy
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
        jet_pt = ak.to_numpy(p4_field(data.reco_jet_p4, "pt")).astype(np.float32)  # [N]
        jet_eta = ak.to_numpy(p4_field(data.reco_jet_p4, "eta")).astype(np.float32)
        jet_phi = ak.to_numpy(p4_field(data.reco_jet_p4, "phi")).astype(np.float32)
        jet_en = ak.to_numpy(p4_field(data.reco_jet_p4, "energy")).astype(np.float32)

        _pt_gen = ak.to_numpy(p4_field(data.gen_jet_tau_p4, "pt")).astype(np.float32)
        _eta_gen = ak.to_numpy(p4_field(data.gen_jet_tau_p4, "eta")).astype(np.float32)
        _phi_gen = ak.to_numpy(p4_field(data.gen_jet_tau_p4, "phi")).astype(np.float32)
        _energy_gen = ak.to_numpy(p4_field(data.gen_jet_tau_p4, "energy")).astype(np.float32)

        _pt_gen_jet = ak.to_numpy(p4_field(data.gen_jet_p4, "pt")).astype(np.float32)
        _eta_gen_jet = ak.to_numpy(p4_field(data.gen_jet_p4, "eta")).astype(np.float32)
        _phi_gen_jet = ak.to_numpy(p4_field(data.gen_jet_p4, "phi")).astype(np.float32)
        _energy_gen_jet = ak.to_numpy(p4_field(data.gen_jet_p4, "energy")).astype(np.float32)

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

    _sample_name = staticmethod(sample_name)

    def _select_row_groups(
        self, row_groups: list, rng: np.random.Generator
    ) -> list:
        """
        Apply `cfg.dataset.max_jets_per_sample` to a flat row-group list.

        Selection is deterministic given `cfg.dataset.selection_seed`: row groups
        are ordered canonically by (filename, row-group index) before being
        shuffled by the seeded generator, so the same seed and the same files
        always yield the same subset regardless of glob order.

        Row groups are the smallest readable unit, so the kept jet count
        overshoots the limit by at most one row group.
        """
        limits = self.cfg.dataset.get("max_jets_per_sample", None)

        by_sample: dict[str, list] = {}
        for row_group in row_groups:
            by_sample.setdefault(self._sample_name(row_group.filename), []).append(
                row_group
            )

        selected = []
        for sample in sorted(by_sample):
            groups = sorted(
                by_sample[sample], key=lambda rg: (rg.filename, rg.row_group)
            )
            rng.shuffle(groups)

            limit = None if limits is None else limits.get(sample, None)
            available = sum(rg.num_rows for rg in groups)
            if limit is None:
                kept, n_jets = groups, available
                note = "all"
            else:
                kept, n_jets = [], 0
                for row_group in groups:
                    if n_jets >= int(limit):
                        break
                    kept.append(row_group)
                    n_jets += row_group.num_rows
                note = f"limit {int(limit):,}"
            print(
                f"[dataset] {sample}: {n_jets:,} / {available:,} jets "
                f"({len(kept):,} / {len(groups):,} row groups, {note})",
                flush=True,
            )
            selected.extend(kept)
        return selected

    def _resolve_input_paths(self, pattern: str, dataset_type: str) -> list:
        """
        Glob `pattern` and fail loudly if it matches nothing.

        An empty match previously produced a dataset of 0 jets and training
        continued, so the only symptom was "There are 0 jets in the dataset"
        followed by a job that appeared to hang. The three usual causes are a
        data_dir that is not bind-mounted into the container, filenames that do
        not contain "_train"/"_test", and files sitting one directory deeper.
        """
        paths = sorted(glob.glob(pattern))
        if paths:
            # Logged on success too: a silently wrong data_dir is otherwise only
            # visible as a surprising jet count much further down.
            print(
                f"[dataset] {dataset_type}: {len(paths):,} file(s) matched "
                f"{pattern}",
                flush=True,
            )
            return paths

        data_dir = os.path.dirname(pattern)
        lines = [
            f"No {dataset_type} files matched: {pattern}",
            f"  dataset.data_dir      : {self.cfg.dataset.data_dir}",
            f"  directory exists      : {os.path.isdir(data_dir)}",
        ]
        if os.path.isdir(data_dir):
            everything = sorted(os.listdir(data_dir))
            parquet = [f for f in everything if f.endswith(".parquet")]
            subdirs = [f for f in everything if os.path.isdir(os.path.join(data_dir, f))]
            lines += [
                f"  entries in directory  : {len(everything)}",
                f"  .parquet files there  : {len(parquet)}",
                f"  first few names       : {everything[:8]}",
                f"  subdirectories        : {subdirs[:8]}",
                "",
                "Files are matched as '{sample}_" + dataset_type + "*.parquet' with "
                f"sample='{self.sample}', so a name must contain '_{dataset_type}'.",
                "The sample label is the part before '_" + dataset_type + "', and it is "
                "what dataset.max_jets_per_sample keys on.",
            ]
        else:
            lines += [
                "",
                "The directory is not visible from inside the container. Check that "
                "it is covered by a -B bind mount in run-lumi.sh / run.sh.",
            ]
        raise FileNotFoundError("\n".join(lines))

    def get_dataset_rowgroups(self, dataset_type: str):
        if dataset_type == "test":
            test_paths_wcp = os.path.join(
                os.path.expanduser(os.path.expandvars(self.cfg.dataset.data_dir)),
                f"{self.sample}_test*.parquet",
            )
            test_paths = self._resolve_input_paths(test_paths_wcp, "test")
            test_rowgroups = ig.get_row_groups(input_paths=test_paths)
            # max_jets_per_sample is deliberately NOT applied here: truncating the
            # evaluation set would silently change every reported metric.
            np.random.default_rng(
                int(self.cfg.dataset.get("selection_seed", 42))
            ).shuffle(test_rowgroups)
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
                os.path.expanduser(os.path.expandvars(self.cfg.dataset.data_dir)),
                f"{self.sample}_train*.parquet",
            )
            train_paths = self._resolve_input_paths(train_paths_wcp, "train")
            # A dedicated generator rather than the global numpy state, so the
            # train/val split and the per-sample subsampling cannot be perturbed
            # by unrelated random draws elsewhere in the process.
            rng = np.random.default_rng(
                int(self.cfg.dataset.get("selection_seed", 42))
            )
            all_train_rowgroups = self._select_row_groups(
                ig.get_row_groups(input_paths=train_paths), rng
            )
            rng.shuffle(all_train_rowgroups)
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
            # batch_size=None: dataset yields pre-batched slices, skip collation
            # entirely. The loader arguments go through loader_kwargs because
            # several of them are only legal for num_workers > 0: passing
            # prefetch_factor or persistent_workers with 0 workers raises, and 0
            # workers is the configuration to reach for when a job hangs before
            # the first batch.
            n_workers = (
                0
                if self.debug_run
                else resolve_num_workers(
                    self.cfg.training.dataloader.num_dataloader_workers
                )
            )
            kwargs = loader_kwargs(
                n_workers,
                self.cfg.training.dataloader.prefetch_factor,
                self.debug_run,
            )
            self.train_loader = DataLoader(self.train_dataset, batch_size=None, **kwargs)
            self.val_loader = DataLoader(self.val_dataset, batch_size=None, **kwargs)
        elif stage == "test" or stage == "predict":
            test_row_groups = self.get_dataset_rowgroups(dataset_type="test")
            self.test_dataset = ParticleTransformerDataset(
                row_groups=test_row_groups, cfg=self.cfg, batch_size=batch_size
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=None,
                **loader_kwargs(
                    resolve_num_workers(
                        self.cfg.training.dataloader.num_dataloader_workers
                    ),
                    self.cfg.training.dataloader.prefetch_factor,
                    self.debug_run,
                ),
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
