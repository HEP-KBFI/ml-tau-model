"""
Evaluate a RareDecays SingleParTau decay-mode checkpoint on the held-out
z_test signal set.

Runs inference with the given checkpoint, then writes:
  - {sample}_confusion_matrix.png / .pdf        (normalized per true class:
                                                  recall/efficiency view)
  - {sample}_confusion_matrix_normPred.png / .pdf (normalized per predicted
                                                  class: precision/purity view)
  - {sample}_{algorithm}_class_metrics.json
  - {sample}_confusion_matrix_K0_split.png / .pdf        (true axis split into
                                                  K0_S / K0_L for the two
                                                  K0-bearing classes, true-norm)
  - {sample}_confusion_matrix_K0_split_normPred.png / .pdf (same, pred-norm)

All labels are the physical decay-mode names (pi, piK0, 3pi, ...), not the
raw integer class codes.

Usage:
    ./run.sh python3 mltau/scripts/evaluate_RareDecays.py \
        --checkpoint /path/to/ParT-model_best.ckpt

Optional:
    --output-dir /path/to/evaluation/z_test   (default: <run_dir>/evaluation/<sample>,
                                                where <run_dir> is the checkpoint's
                                                grandparent directory, i.e. output_dir)
    --data-dir /scratch/.../some_rare_decays_dataset  (default: dataset_RareDecays.yaml's)
    --sample z_test                            (default: "z_test"; used only for
                                                labeling output files)
    --skip-k0-split                            (skip the K0_S/K0_L-split plots)
"""

import argparse
import glob
import os

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig

torch.serialization.add_safe_globals([DictConfig, ListConfig])
torch.set_float32_matmul_precision("high")

from mltau.models import SingleParTau_module
from mltau.tools.evaluation import decay_mode as dm
from mltau.tools.evaluation import inference
from mltau.tools.io.RareDecays_dataloader import RareDecaysDataModule

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The two K0-bearing classes in the 13-class rare scheme (see
# mltau.tools.general.RARE_DECAY_MODE_NAME_MAPPING): piK0 and pipi0K0. Every
# other class has no neutral kaon to split.
K0_BEARING_CODES = {6, 10}
K0S_PDG, K0L_PDG = 310, 130


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="Path to the .ckpt to evaluate")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write evaluation outputs. Defaults to "
        "<checkpoint's output_dir>/evaluation/<sample>",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Override dataset.data_dir (defaults to dataset_RareDecays.yaml's)",
    )
    p.add_argument(
        "--sample",
        default="z_test",
        help="Label used in output filenames and printed metrics (default: z_test)",
    )
    p.add_argument(
        "--skip-k0-split",
        action="store_true",
        help="Skip the K0_S/K0_L true-axis-split confusion matrix",
    )
    p.add_argument(
        "--skip-loss-curves",
        action="store_true",
        help="Skip the train/val loss curve plot",
    )
    return p.parse_args()


def load_model(cfg, ckpt_path):
    num_dm_classes = cfg.training.model.num_dm_classes
    model = SingleParTau_module.ParTauModule.load_from_checkpoint(
        ckpt_path,
        cfg=cfg,
        input_dim=17,
        num_dm_classes=num_dm_classes,
        task=cfg.training.model.task,
        weights_only=False,
        map_location="cpu",
    )
    model.eval()
    return model


def run_inference(model, cfg, output_dir):
    print(f"[INFO] Running inference on z_test signal parquet files from {cfg.dataset.data_dir}")
    inference.create_predictions_files(
        best_model=model,
        cfg=cfg,
        model_name="SingleParTau",
        test_only=True,
        dataset_cls=RareDecaysDataModule.dataset_cls,
    )
    pred_dir = os.path.join(output_dir, "predictions")
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "z_test*.parquet")))
    if not pred_files:
        raise RuntimeError(f"No prediction files found in {pred_dir}")
    data = ak.concatenate([ak.from_parquet(f) for f in pred_files])
    print(f"[INFO] Total test signal jets: {len(data)}")
    return data, pred_files


def run_basic_evaluation(data, output_dir, sample, algorithm):
    evaluator = dm.DecayModeEvaluator(
        pred_proba=data.tau_decay_mode_probs,
        truth=data.gen_jet_tau_decaymode,
        output_dir=output_dir,
        sample=sample,
        algorithm=algorithm,
        decay_mode_name_mapping=dm.RARE_DECAY_MODE_NAME_MAPPING,
    )
    evaluator.print_performance()
    # Writes both {sample}_{algorithm}_confusion_matrix.pdf (per-truth) and
    # {sample}_{algorithm}_confusion_matrix_normPred.pdf (per-prediction), plus
    # {sample}_{algorithm}_class_metrics.json.
    evaluator.save_performance()

    for normalize, suffix in (("true", ""), ("pred", "_normPred")):
        fig, _ = evaluator.plot_confusion_matrix(normalize=normalize)
        fig.savefig(
            os.path.join(output_dir, f"{sample}_confusion_matrix{suffix}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
    return evaluator


def run_k0_split_evaluation(data_dir, pred_files, output_dir, sample):
    """
    Split the true axis of the two K0-bearing classes (piK0, pipi0K0) into
    K0_S / K0_L using the per-jet PDG list already in the raw input dataset
    (gen_jet_tau_vis_daughter_pdgs_rare: 310 = K0_S, 130 = K0_L). The
    predicted axis stays the model's native 13 classes, since it was never
    trained to tell K0_S from K0_L apart.
    """
    pred_categories = list(dm.RARE_DECAY_MODE_NAME_MAPPING.values())
    pred_code_to_col = {
        code: i for i, code in enumerate(dm.RARE_DECAY_MODE_NAME_MAPPING.keys())
    }

    true_row_keys, true_categories = [], []
    for code, label in dm.RARE_DECAY_MODE_NAME_MAPPING.items():
        if code == 6:
            true_row_keys += ["6S", "6L"]
            true_categories += [r"$\pi K^0_S$", r"$\pi K^0_L$"]
        elif code == 10:
            true_row_keys += ["10S", "10L"]
            true_categories += [r"$\pi\pi^0 K^0_S$", r"$\pi\pi^0 K^0_L$"]
        else:
            true_row_keys.append(code)
            true_categories.append(label)
    true_key_to_row = {key: i for i, key in enumerate(true_row_keys)}

    true_rows, pred_cols = [], []
    n_ambiguous = n_no_flag = 0
    for pred_path in pred_files:
        raw_path = os.path.join(data_dir, os.path.basename(pred_path))
        if not os.path.exists(raw_path):
            print(f"[WARNING] Raw input {raw_path} not found; skipping K0 split for this file")
            continue
        raw = ak.from_parquet(raw_path)
        pred = ak.from_parquet(pred_path)
        if len(raw) != len(pred):
            print(f"[WARNING] Row count mismatch for {os.path.basename(pred_path)}; skipping")
            continue

        dm_rare = np.asarray(raw.gen_jet_tau_decaymode_rare).astype(int)
        pred_code = np.asarray(pred.tau_decay_mode).astype(int)
        pdgs_rare = raw.gen_jet_tau_vis_daughter_pdgs_rare

        for i in range(len(raw)):
            code = int(dm_rare[i])
            p_col = pred_code_to_col[int(pred_code[i])]
            if code in K0_BEARING_CODES:
                abs_pdgs = set(int(abs(p)) for p in pdgs_rare[i])
                has_s, has_l = K0S_PDG in abs_pdgs, K0L_PDG in abs_pdgs
                if has_s and has_l:
                    n_ambiguous += 1
                    continue
                elif has_s:
                    row_key = f"{code}S"
                elif has_l:
                    row_key = f"{code}L"
                else:
                    n_no_flag += 1
                    continue
                true_rows.append(true_key_to_row[row_key])
            else:
                true_rows.append(true_key_to_row[code])
            pred_cols.append(p_col)

    if n_ambiguous or n_no_flag:
        print(
            f"[INFO] K0 split: dropped {n_ambiguous} double-K0 jets, "
            f"{n_no_flag} K0-bearing jets with no S/L flag found"
        )

    true_rows = np.asarray(true_rows)
    pred_cols = np.asarray(pred_cols)
    n_true, n_pred = len(true_categories), len(pred_categories)
    counts = np.zeros((n_true, n_pred), dtype=np.int64)
    np.add.at(counts, (true_rows, pred_cols), 1)

    row_sums = counts.sum(axis=1, keepdims=True)
    normalized_true = np.divide(
        counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums > 0
    )
    col_sums = counts.sum(axis=0, keepdims=True)
    normalized_pred = np.divide(
        counts, col_sums, out=np.zeros_like(counts, dtype=float), where=col_sums > 0
    )

    print("[INFO] Support per true (K0-split) category:")
    for key, label, n in zip(true_row_keys, true_categories, row_sums[:, 0]):
        print(f"  {key:>4} {label:>18}  n={int(n)}")

    for normalize, histogram, suffix in [
        ("true", normalized_true, ""),
        ("pred", normalized_pred, "_normPred"),
    ]:
        fig, _ = dm.visualize_confusion_matrix(
            histogram=histogram,
            categories=true_categories,
            y_categories=pred_categories,
            x_label=r"True decay modes ($K^0$ split by $S$/$L$)",
            y_label="Predicted decay modes",
        )
        for ext, kwargs in (
            ("png", dict(dpi=150, bbox_inches="tight")),
            ("pdf", dict(format="pdf", bbox_inches="tight")),
        ):
            out_path = os.path.join(output_dir, f"{sample}_confusion_matrix_K0_split{suffix}.{ext}")
            fig.savefig(out_path, **kwargs)
            print(f"[INFO] wrote {out_path}")
        plt.close(fig)


def plot_loss_curves(run_dir, output_dir, sample):
    """
    Plot train/val loss across the WHOLE training history, concatenating every
    TensorBoard version directory under run_dir/tensorboard/ParTau_experiment
    (a crash-and-resume creates a new version_N each time trainer.fit restarts,
    but global_step is preserved across the resume, so sorting by step
    reconstructs one continuous curve).
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb_root = os.path.join(run_dir, "tensorboard", "ParTau_experiment")
    version_dirs = sorted(
        glob.glob(os.path.join(tb_root, "version_*")),
        key=lambda p: int(p.rsplit("_", 1)[-1]),
    )
    if not version_dirs:
        print(f"[WARNING] No tensorboard logs found under {tb_root}; skipping loss curves")
        return

    def collect(tag):
        points = []
        for v in version_dirs:
            ea = EventAccumulator(v, size_guidance={"scalars": 0})
            ea.Reload()
            if tag in ea.Tags().get("scalars", []):
                points.extend((e.step, e.value) for e in ea.Scalars(tag))
        points.sort(key=lambda p: p[0])
        return points

    train_loss = collect("train_losses/loss")
    val_loss = collect("val_losses/loss")
    if not train_loss and not val_loss:
        print(f"[WARNING] No loss scalars found under {tb_root}; skipping loss curves")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    if train_loss:
        steps, values = zip(*train_loss)
        ax.plot(steps, values, label="train loss", color="tab:blue", alpha=0.5, linewidth=1)
    if val_loss:
        steps, values = zip(*val_loss)
        ax.plot(steps, values, label="val loss", color="tab:orange", linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.set_title("Training / validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    for ext, kwargs in (
        ("png", dict(dpi=150, bbox_inches="tight")),
        ("pdf", dict(format="pdf", bbox_inches="tight")),
    ):
        out_path = os.path.join(output_dir, f"{sample}_loss_curves.{ext}")
        fig.savefig(out_path, **kwargs)
        print(f"[INFO] wrote {out_path}")
    plt.close(fig)


def main():
    args = parse_args()
    ckpt_path = os.path.abspath(args.checkpoint)
    # <run_dir>/models/some.ckpt -> run_dir is the training run's output_dir,
    # needed for the tensorboard logs regardless of where --output-dir points.
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(run_dir, "evaluation", args.sample)
    os.makedirs(output_dir, exist_ok=True)

    overrides = [f"output_dir={output_dir}"]
    if args.data_dir is not None:
        overrides.append(f"dataset.data_dir={args.data_dir}")

    # training.input_scaling.scaler_path defaults to ${output_dir}/scaler/..., but
    # output_dir is overridden above to the EVALUATION dir, not the training run's
    # -- so that interpolation silently points at a nonexistent file, and a
    # checkpoint trained with input scaling enabled gets evaluated with scaling
    # silently OFF (unscaled features it was never trained on). Auto-detect from
    # the training run's own scaler file instead of trusting the composed config.
    run_dir_scaler_path = os.path.join(run_dir, "scaler", "cand_feature_scaler.npz")
    if os.path.exists(run_dir_scaler_path):
        overrides.append("training.input_scaling.enabled=true")
        overrides.append(f"training.input_scaling.scaler_path={run_dir_scaler_path}")
        print(f"[INFO] Detected scaler at {run_dir_scaler_path}; enabling input scaling")

    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(REPO_ROOT, "mltau/config")
    ):
        cfg = compose(config_name="main_RareDecays", overrides=overrides)

    model = load_model(cfg, ckpt_path)
    print(f"[INFO] Loaded checkpoint {ckpt_path}")

    data, pred_files = run_inference(model, cfg, output_dir)

    algorithm = os.path.splitext(os.path.basename(ckpt_path))[0]
    run_basic_evaluation(data, output_dir, args.sample, algorithm)

    if not args.skip_k0_split:
        run_k0_split_evaluation(cfg.dataset.data_dir, pred_files, output_dir, args.sample)

    if not args.skip_loss_curves:
        plot_loss_curves(run_dir, output_dir, args.sample)

    print(f"[INFO] Wrote evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
