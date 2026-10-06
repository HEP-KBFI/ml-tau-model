import numpy as np
import matplotlib.pyplot as plt

from mltau.tools.evaluation import decay_mode as dm
from mltau.tools.logging.general import log_metrics_dict


def log_all_decay_mode_metrics(
    targets: np.array,
    predictions: np.array,
    tb_logger,
    # output_dir: str,
    current_epoch: int,
    decay_mode_scheme: str = "standard",
):
    # DM loss is trained only on signal taus; background jets have DM=-1 mapped
    # to "Rare"/"Other" (the last class) in the target, which the model never
    # learns to predict. Filter to signal only so the confusion matrix matches
    # inference evaluation.
    signal_mask = np.asarray(targets["is_tau"]) == 1
    predictions_proba = np.asarray(predictions["decay_mode"])[signal_mask]

    evaluator = dm.DecayModeEvaluator(
        pred_proba=predictions_proba,
        truth=np.asarray(targets["decay_mode"])[signal_mask],
        output_dir="",
        sample="all",
        algorithm="all",
        decay_mode_name_mapping=dm.DECAY_MODE_NAME_MAPPINGS[decay_mode_scheme],
    )
    cm_true = dm.ConfusionMatrix(evaluator=evaluator, normalize="true")
    tb_logger.add_figure("decay_mode/confusion_matrix", cm_true.fig, current_epoch)
    plt.close(cm_true.fig)

    cm_pred = dm.ConfusionMatrix(evaluator=evaluator, normalize="pred")
    tb_logger.add_figure("decay_mode/confusion_matrix_normPred", cm_pred.fig, current_epoch)
    plt.close(cm_pred.fig)

    log_metrics_dict(tb_logger, evaluator.general_metrics, "decay_mode", current_epoch)

    roc_plot = dm.DecayModeROCPlot(evaluator=evaluator)
    tb_logger.add_figure("decay_mode/ROC", roc_plot.fig, current_epoch)
    plt.close(roc_plot.fig)
