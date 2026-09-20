import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from sklearn import metrics


def dm_from_reco_daughters(hps_data: ak.Array):
    """HPS has also a decay mode -1, where no tau is built as it does not satisfy some conditions.
    So far we have treated this decay mode as a Rare decay mode, but did not use these entries for calculating
    p4 related variables. I think it makes sense to pivot back to not including them at all to make the comparisons
    a bit fairer, as we can just say that our ML model recovers XYZ number of these taus"""
    hps_data = ak.from_parquet(hps_pred_path)
    n_charged = ak.sum(hps_data.tauSigCand_pdgIds == 211, axis=1)
    n_pions = ak.num(hps_data.tauStrip_p4s)
    decay_mode = 5 * (n_charged - 1) + n_pions
    return decay_mode


def visualize_hps_confusion_matrix(
    histogram: np.ndarray,
    categories: list,
    predicted_categories: list,
    cmap: str = "GnBu",
    bin_text_color: str = "black",
    y_label: str = "Predicted decay modes",
    x_label: str = "True decay modes",
    figsize: tuple = (12, 12),
):
    """Plot a rectangular, true-normalized confusion matrix.

    histogram:
        Normalized confusion matrix in sklearn convention:
        (true, predicted).

    categories:
        True-class labels.

    predicted_categories:
        Predicted-class labels, including the extra -1 class.
    """

    n_true = len(categories)
    n_pred = len(predicted_categories)

    if histogram.shape != (n_true, n_pred):
        raise ValueError(
            f"Expected histogram shape ({n_true}, {n_pred}), but got {histogram.shape}."
        )

    fig, ax = plt.subplots(figsize=figsize)

    # sklearn:       (true, predicted)
    # plotting:      (predicted, true)
    histogram_plot = histogram

    xbins = np.arange(n_true + 1)
    ybins = np.arange(n_pred + 1)
    x_centers = 0.5 * (xbins[:-1] + xbins[1:])
    y_centers = 0.5 * (ybins[:-1] + ybins[1:])

    x_tick_values = np.arange(n_true) + 0.5
    y_tick_values = np.arange(n_pred) + 0.5

    hep.hist2dplot(
        histogram_plot,
        xbins,
        ybins,
        cmap=cmap,
        cbar=False,
        flow=None,
    )

    ax.grid(False)

    plt.xticks(
        x_tick_values,
        categories,
        fontsize=30,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        y_tick_values,
        predicted_categories,
        fontsize=30,
        rotation=45,
        va="top",
    )

    plt.xlabel(
        x_label,
        fontdict={"size": 36},
    )

    plt.ylabel(
        y_label,
        fontdict={"size": 36},
    )

    ax.tick_params(
        axis="both",
        which="both",
        length=0,
    )

    text_fontsize = 14

    for i in range(n_pred):
        for j in range(n_true):
            # histogram is (true, predicted)
            # i = predicted class
            # j = true class
            value = histogram[j, i]

            # Upper-left
            ax.text(
                x_centers[j],
                y_centers[i],
                f"{value:.2f}",
                color=bin_text_color,
                ha="center",
                va="center",
                fontsize=text_fontsize,
                fontweight="bold",
            )

    fig.subplots_adjust(
        left=0.20,
        bottom=0.22,
        right=0.90,
        top=0.95,
    )

    return fig, ax


def construct_hps_confusion_matrix(truth, predicted, categories):
    confusion_matrix_counts = metrics.confusion_matrix(
        truth,
        predicted,
        labels=categories,
    )
    # Add the unclassified prediction column
    unclassified_counts = np.array(
        [np.sum((truth == category) & (predicted == -1)) for category in categories]
    )[:, None]

    confusion_matrix_counts = np.hstack([confusion_matrix_counts, unclassified_counts])

    # Normalize each true-class row
    confusion_matrix_normalized = confusion_matrix_counts / confusion_matrix_counts.sum(
        axis=1, keepdims=True
    )
    return confusion_matrix_normalized


def evaluate_hps_decaymode_classification(truth, predicted, categories):
    confusion_matrix_normalized = construct_hps_confusion_matrix(
        truth, predicted, categories
    )
    fig, ax = visualize_hps_confusion_matrix(
        histogram=confusion_matrix_normalized,
        categories=categories,
        predicted_categories=list(categories) + [-1],
    )
    return fig, ax
