import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from omegaconf import DictConfig

from mltau.tools.evaluation import charge_id as c
from mltau.tools.evaluation.general import binary_classifier_metrics
from mltau.tools.logging.general import log_metrics_dict


def log_charge_id_performance(
    targets: np.array,
    gen_jet_tau_p4s: np.array,
    reco_jet_p4s: np.array,
    predictions: np.array,
    cfg: DictConfig,
    tb_logger,
    current_epoch: int,
    dataset="train",
    baseline_charges: np.array = None,
):
    # Charge is only meaningful for signal taus — exclude background jets
    signal_mask = targets["is_tau"] == 1
    predictions = predictions["charge"][signal_mask]
    targets = targets["charge"][signal_mask]
    gen_jet_tau_p4s = gen_jet_tau_p4s[signal_mask]
    reco_jet_p4s = reco_jet_p4s[signal_mask]

    # Apply signal mask to baseline charges if provided.
    # The current evaluator path does not consume it, but keeping the argument
    # avoids breaking callers that still pass it.
    if baseline_charges is not None:
        baseline_charges = baseline_charges[signal_mask]

    evaluator = c.ChargeIdEvaluator(
        predicted=predictions,
        truth=targets,
        gen_jet_tau_p4s=gen_jet_tau_p4s,
        reco_jet_p4s=reco_jet_p4s,
        cfg=cfg,
        sample="all",
        algorithm="all",
    )

    # Classifier plot
    classifier_plot = c.ChargeClassifierPlot()
    classifier_plot.add_line(evaluator, dataset)
    tb_logger.add_figure("charge_id/classifier", classifier_plot.fig, current_epoch)
    plt.close(classifier_plot.fig)

    # ROC plot
    roc_plot = c.ROCPlot(cfg)
    roc_plot.add_line(evaluator)
    tb_logger.add_figure("charge_id/ROC", roc_plot.fig, current_epoch)
    plt.close(roc_plot.fig)

    # Confusion matrix plot
    confusion_plot = c.ConfusionMatrixPlot()
    confusion_plot.add_data(evaluator)
    tb_logger.add_figure(
        "charge_id/confusion_matrix", confusion_plot.fig, current_epoch
    )
    plt.close(confusion_plot.fig)

    # Per-metric efficiency and fakerate plots
    metrics = list(cfg.metrics.charge.metrics.keys())
    for metric in metrics:
        eff_plot = c.EfficiencyPlot(cfg, metric)
        eff_plot.add_line(evaluator)
        tb_logger.add_figure(
            f"charge_id/{metric}_efficiency", eff_plot.fig, current_epoch
        )
        plt.close(eff_plot.fig)

        fr_plot = c.FakeRatePlot(cfg, metric)
        fr_plot.add_line(evaluator)
        tb_logger.add_figure(f"charge_id/{metric}_fakerate", fr_plot.fig, current_epoch)
        plt.close(fr_plot.fig)

    # Scalar classification metrics at the 95% average efficiency working point
    charge_scalars = binary_classifier_metrics(
        evaluator.predicted, evaluator.truth, evaluator.wp_pos
    )
    charge_scalars["wp_pos"] = evaluator.wp_pos
    charge_scalars["wp_neg"] = evaluator.wp_neg

    # Add confusion matrix metrics
    confusion_matrix = evaluator.confusion_matrix
    charge_scalars["TP"] = confusion_matrix["TP"]
    charge_scalars["FP"] = confusion_matrix["FP"]

    # Calculate and add derived metrics from confusion matrix
    total = (
        confusion_matrix["TP"]
        + confusion_matrix["TN"]
        + confusion_matrix["FP"]
        + confusion_matrix["FN"]
    )
    if total > 0:
        charge_scalars["confusion_accuracy"] = (
            confusion_matrix["TP"] + confusion_matrix["TN"]
        ) / total
    else:
        charge_scalars["confusion_accuracy"] = 0.0

    log_metrics_dict(tb_logger, charge_scalars, "charge_id", current_epoch)
