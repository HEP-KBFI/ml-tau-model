"""
Measure the residual scales for the kinematics regression loss (TauLoss).

TauLoss divides each kinematics residual by a scale before the Huber, so every
component enters the loss as "error in units of the spread of its own target"
(see mltau/tools/losses.py, _compute_kinematics_loss_per_sample). The scales
are properties of the data, not tuning knobs: this script reads the training
targets exactly as the dataloader produces them and prints their spreads, plus
a block to paste into tau_loss.kinematics_scales.

    ./run.sh python3 mltau/scripts/measure_kinematics_scales.py --model ParT
    ./run.sh python3 mltau/scripts/measure_kinematics_scales.py --model DETR
    # extra Hydra overrides go after --, e.g. another production:
    ./run.sh python3 mltau/scripts/measure_kinematics_scales.py --model ParT -- \\
        dataset.data_dir=/scratch/persistent/laurits/ml-tau/<production>/

Targets (same 5-vector for both models, built in the dataloaders):
    [log(pt / pt_jet), eta - eta_jet, sin(dphi), cos(dphi), log(m / m_jet)]
  ParT:  the visible tau of signal jets (is_tau == 1), cfg main.yaml.
  DETR:  every visible tau daughter of signal jets, cfg main_ParTauDETR.yaml.
The two differ by an order of magnitude in the angular spreads, so they need
separate scales.

The phi component of the loss is the chord between the predicted and the true
(sin dphi, cos dphi). For the trivial prediction "along the jet axis" the chord
is 2|sin(dphi/2)| ~ |dphi|, so its natural scale is the spread of dphi itself;
the script also prints sigma(sin dphi) and the RMS chord, since earlier configs
quoted those.

Several spread estimators are printed per component because the targets have
tails (the log ratios are clamped at +-5 by the dataloader): std, a robust
sigma (half the 16-84 % interquantile range, equal to the std for a Gaussian)
and the RMS. The YAML block uses the estimator chosen with --estimator.
"""

import argparse
import os
import sys

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig

# Checkpoints/configs carry OmegaConf objects; harmless here, mirrors train.py.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([DictConfig, ListConfig])

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
LOG_CLAMP = 5.0  # the dataloaders clip the log-ratio targets to +-LOG_CLAMP


def load_targets(model: str, split: str, max_entries: int, overrides: list) -> np.ndarray:
    """(N, 5) kinematics targets for signal taus (ParT) or tau daughters (DETR)."""
    config_name = "main" if model == "ParT" else "main_ParTauDETR"
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    if model == "ParT":
        from mltau.tools.io.ParT_dataloader import ParTDataModule as DataModule
    else:
        from mltau.tools.io.ParTauDETR_dataloader import ParTauDETRDataModule as DataModule

    datamodule = DataModule(cfg=cfg, debug_run=False)
    datamodule.setup("fit")
    loader = {"train": datamodule.train_dataloader, "val": datamodule.val_dataloader}[split]()

    chunks, n = [], 0
    for batch in loader:
        targets = batch[2]
        if model == "ParT":
            keep = targets["is_tau"].bool()
            kin = targets["kinematics"][keep]
        else:
            # Only signal jets have daughters; padded daughter slots are masked.
            kin = targets["particles_kinematics"][targets["particles_mask"]]
        chunks.append(kin.double().numpy())
        n += len(kin)
        print(f"\r  read {n:,} targets", end="", flush=True)
        if n >= max_entries:
            break
    print()
    if not chunks or n == 0:
        sys.exit("No targets read; check dataset.data_dir and the sample selection.")
    return np.concatenate(chunks)[:max_entries]


def spreads(x: np.ndarray) -> dict:
    q16, q50, q84 = np.percentile(x, [16, 50, 84])
    return {
        "mean": x.mean(),
        "median": q50,
        "std": x.std(),
        "robust": 0.5 * (q84 - q16),
        "rms": np.sqrt(np.mean(x**2)),
    }


def summarize(kin: np.ndarray) -> dict:
    """Spreads of every quantity the loss components are built from."""
    dphi = np.arctan2(kin[:, 2], kin[:, 3])
    quantities = {
        "log_pt": kin[:, 0],
        "delta_eta": kin[:, 1],
        "dphi": dphi,
        "sin_dphi": kin[:, 2],
        "chord_to_axis": 2.0 * np.abs(np.sin(dphi / 2.0)),  # chord of the trivial prediction
        "log_mass": kin[:, 4],
    }
    table = {name: spreads(x) for name, x in quantities.items()}
    for name in ("log_pt", "log_mass"):
        table[name]["clamped"] = float(np.mean(np.abs(quantities[name]) >= LOG_CLAMP - 1e-6))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["ParT", "DETR"], required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--max-entries", type=int, default=1_000_000)
    parser.add_argument("--estimator", choices=["std", "robust", "rms"], default="std",
                        help="estimator used for the printed YAML block")
    args, overrides = parser.parse_known_args()
    overrides = [o for o in overrides if o != "--"]

    print(f"[{args.model}] reading {args.split} targets (max {args.max_entries:,})")
    kin = load_targets(args.model, args.split, args.max_entries, overrides)
    table = summarize(kin)

    what = "visible taus" if args.model == "ParT" else "tau daughters"
    print(f"\n{len(kin):,} {what}\n")
    print(f"{'quantity':15s} {'mean':>9s} {'median':>9s} {'std':>9s} {'robust':>9s} {'rms':>9s}  at clamp")
    for name, s in table.items():
        clamp = f"{100 * s['clamped']:.2f}%" if "clamped" in s else ""
        print(f"{name:15s} {s['mean']:9.4f} {s['median']:9.4f} {s['std']:9.4f} "
              f"{s['robust']:9.4f} {s['rms']:9.4f}  {clamp}")

    e = args.estimator
    print(f"\n# tau_loss.kinematics_scales ({what}, {args.split}, {len(kin):,} entries, estimator: {e})")
    print("kinematics_scales:")
    print(f"    log_pt: {table['log_pt'][e]:.3g}")
    print(f"    delta_eta: {table['delta_eta'][e]:.3g}")
    print(f"    phi_chord: {table['dphi'][e]:.3g}    # spread of dphi; chord ~ |dphi| for collimated targets")
    print(f"    log_mass: {table['log_mass'][e]:.3g}")


if __name__ == "__main__":
    main()
