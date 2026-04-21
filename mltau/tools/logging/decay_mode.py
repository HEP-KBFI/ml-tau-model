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
):
    predictions_proba = np.asarray(predictions["decay_mode"])
    targets_class = np.argmax(np.asarray(targets["decay_mode"]), axis=-1)

    evaluator = dm.DecayModeEvaluator(
        pred_proba=predictions_proba,
        truth=targets_class,
        output_dir="",
        sample="all",
        algorithm="all",
    )
    cm = dm.ConfusionMatrix(evaluator=evaluator)
    tb_logger.add_figure("decay_mode/confusion_matrix", cm.fig, current_epoch)
    plt.close(cm.fig)

    log_metrics_dict(tb_logger, evaluator.general_metrics, "decay_mode", current_epoch)

    roc_plot = dm.DecayModeROCPlot(evaluator=evaluator)
    tb_logger.add_figure("decay_mode/ROC", roc_plot.fig, current_epoch)
    plt.close(roc_plot.fig)
