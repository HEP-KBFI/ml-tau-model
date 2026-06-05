import os
import json
import numpy as np
import mplhep as hep
from sklearn import metrics
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from mltau.tools.io.general import NpEncoder
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import precision_score, f1_score

hep.style.use(hep.styles.CMS)
plt.rcParams["mathtext.fontset"] = "stix"


def visualize_confusion_matrix(
    histogram: np.array,
    categories: list,
    cmap: str = "GnBu",
    bin_text_color: str = "black",
    y_label: str = "Predicted decay modes",
    x_label: str = "True decay modes",
    figsize: tuple = (12, 12),
):
    """Plots the confusion matrix for the classification task. Confusion
    matrix functions has the categories in the other way in order to have the
    truth on the x-axis.
    Args:
        histogram : np.array
            Histogram produced by the sklearn.metrics.confusion_matrix.
        categories : list
            Category labels in the correct order.
        cmap : str
            [default: "gray"] The colormap to be used.
        bin_text_color : str
            [default: "r"] The color of the text on bins.
        y_label : str
            [default: "Predicted"] The label for the y-axis.
        x_label : str
            [default: "Truth"] The label for the x-axis.
        figsize : tuple
            The size of the figure drawn.
    """
    fig, ax = plt.subplots(figsize=figsize)
    xbins = ybins = np.arange(len(categories) + 1)
    tick_values = np.arange(len(categories)) + 0.5
    hep.hist2dplot(histogram, xbins, ybins, cmap=cmap, cbar=False, flow=None)
    ax.grid(False)
    plt.xticks(tick_values, categories, fontsize=30, rotation=45, ha="right")
    plt.yticks(tick_values + 0.2, categories, fontsize=30, rotation=45, va="top")
    plt.xlabel(f"{x_label}", fontdict={"size": 36})
    plt.ylabel(f"{y_label}", fontdict={"size": 36})
    ax.tick_params(axis="both", which="both", length=0)
    for i in range(len(ybins) - 1):
        for j in range(len(xbins) - 1):
            bin_value = histogram.T[i, j]
            ax.text(
                float(xbins[j] + 0.5),
                float(ybins[i] + 0.5),
                f"{bin_value:.2f}",
                color=bin_text_color,
                ha="center",
                va="center",
                fontsize=26,
                fontweight="bold",
            )
    fig.subplots_adjust(left=0.20, bottom=0.22, right=0.90, top=0.95)
    return fig, ax


class BaseDecayModeEvaluator:
    """Shared functionality for decay mode evaluation.

    Subclasses must call super().__init__() with already-prepared
    ``predicted`` (class-index array) and ``pred_proba`` ((N, n_classes) array).
    """

    def __init__(
        self,
        predicted: np.array,
        pred_proba: np.array,
        truth: np.array,
        output_dir: str = "",
        sample: str = "all",
        algorithm: str = "all",
    ):
        self.output_dir = output_dir
        if output_dir != "":
            os.makedirs(self.output_dir, exist_ok=True)
        self.sample = sample
        self.algorithm = algorithm
        self.pred_proba = np.asarray(pred_proba)
        self.predicted = np.asarray(predicted)
        self._decay_mode_name_mapping = {
            0: r"$h^{\pm}$",
            1: r"$h^{\pm}\pi^0$",
            2: r"$h^\pm+\geq2\pi^0$",
            10: r"$h^{\pm}h^{\mp}h^{\pm}$",
            11: r"$h^\pm h^\mp h^\pm$" "\n" r"$+\geq\pi^0$",
            15: "Rare",
        }
        self.inverse_mapping = {
            i: key for i, key in enumerate(self._decay_mode_name_mapping.keys())
        }
        self.forward_mapping = {key: i for i, key in self.inverse_mapping.items()}

        truth = np.asarray(truth)
        if truth.ndim == 1:
            truth = truth.astype(int)
            # Map -1 → 15 (Rare) before applying forward_mapping, so that
            # algorithms outputting -1 for unclassified taus are handled correctly.
            truth = np.where(truth == -1, 15, truth)
            if np.any(truth > 5):
                # Map known DM labels to class indices; unknown labels → 5 (Rare)
                self.truth = np.vectorize(lambda x: self.forward_mapping.get(x, 5))(
                    truth
                )
            else:
                self.truth = truth
        else:
            self.truth = np.argmax(truth, axis=-1)
        self.confusion_matrix = metrics.confusion_matrix(self.truth, self.predicted)
        self.normalized_confusion_matrix = metrics.confusion_matrix(
            self.truth, self.predicted, normalize="true"
        )
        self.categories = list(self._decay_mode_name_mapping.values())
        self.general_metrics, self.class_metrics = self._calculate_performance_metrics()
        self.class_performances = self.calculate_class_wise_metrics()

    def plot_confusion_matrix(self, output_path: str = ""):
        fig, ax = visualize_confusion_matrix(
            histogram=self.normalized_confusion_matrix,
            categories=self.categories,
        )
        if output_path != "":
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close("all")
        else:
            return fig, ax

    def _calculate_performance_metrics(self):
        cm = self.confusion_matrix
        total = cm.sum()

        class_TP = np.diag(cm).astype(float)
        class_FP = (cm.sum(axis=0) - np.diag(cm)).astype(float)
        class_FN = (cm.sum(axis=1) - np.diag(cm)).astype(float)
        class_TN = (total - class_TP - class_FP - class_FN).astype(float)

        with np.errstate(divide="ignore", invalid="ignore"):
            class_TPR = np.where(
                class_TP + class_FN > 0, class_TP / (class_TP + class_FN), 0.0
            )
            class_FPR = np.where(
                class_FP + class_TN > 0, class_FP / (class_FP + class_TN), 0.0
            )
            class_FNR = np.where(
                class_TP + class_FN > 0, class_FN / (class_TP + class_FN), 0.0
            )
            class_TNR = np.where(
                class_TN + class_FP > 0, class_TN / (class_TN + class_FP), 0.0
            )
            class_precision = np.where(
                class_TP + class_FP > 0, class_TP / (class_TP + class_FP), 0.0
            )
            class_recall = class_TPR
            denom_f1 = class_precision + class_recall
            class_F1 = np.where(
                denom_f1 > 0, 2 * class_precision * class_recall / denom_f1, 0.0
            )
            class_accuracy = (class_TP + class_TN) / total

        TPR = float(np.mean(class_TPR))
        FPR = float(np.mean(class_FPR))
        FNR = float(np.mean(class_FNR))
        TNR = float(np.mean(class_TNR))
        precision = float(np.mean(class_precision))
        recall = float(np.mean(class_recall))
        denom_f1 = precision + recall
        F1 = 2 * precision * recall / denom_f1 if denom_f1 > 0 else 0.0
        accuracy = float(np.sum(class_TP) / total) if total > 0 else 0.0

        class_metrics = {
            "class_FPR": class_FPR,
            "class_FNR": class_FNR,
            "class_TPR": class_TPR,
            "class_TNR": class_TNR,
            "class_precision": class_precision,
            "class_accuracy": class_accuracy,
            "class_recall": class_recall,
            "class_F1": class_F1,
        }
        general_metrics = {
            "FPR": FPR,
            "FNR": FNR,
            "TPR": TPR,
            "TNR": TNR,
            "precision": precision,
            "accuracy": accuracy,
            "recall": recall,
            "F1": F1,
        }
        return general_metrics, class_metrics

    def print_performance(self):
        print("----------------------------------------")
        print("------------ Class metrics -------------")
        print("----------------------------------------")
        print(json.dumps(self.class_metrics, indent=4, cls=NpEncoder))
        print("----------------------------------------")
        print("------------ General metrics -----------")
        print("----------------------------------------")
        print(json.dumps(self.general_metrics, indent=4, cls=NpEncoder))

    def save_performance(self):
        class_metrics_output_path = os.path.join(
            self.output_dir, f"{self.sample}_{self.algorithm}_class_metrics.json"
        )
        with open(class_metrics_output_path, "wt") as out_file:
            json.dump(self.class_metrics, out_file, indent=4, cls=NpEncoder)
        confusion_matrix_output_path = os.path.join(
            self.output_dir, f"{self.sample}_{self.algorithm}_confusion_matrix.pdf"
        )
        self.plot_confusion_matrix(output_path=confusion_matrix_output_path)

    def calculate_class_wise_metrics(self):
        return {
            "F1": f1_score(y_true=self.truth, y_pred=self.predicted, average=None),
            "precision": precision_score(
                y_true=self.truth, y_pred=self.predicted, average=None
            ),
        }


class DecayModeEvaluator(BaseDecayModeEvaluator):
    """Decay mode evaluator for soft probability predictions."""

    def __init__(
        self,
        pred_proba: np.array,
        truth: np.array,
        output_dir: str = "",
        sample: str = "all",
        algorithm: str = "all",
    ):
        pred_proba = np.asarray(pred_proba)
        predicted = np.argmax(pred_proba, axis=-1)
        super().__init__(predicted, pred_proba, truth, output_dir, sample, algorithm)


class HardLabelDecayModeEvaluator(BaseDecayModeEvaluator):
    """Decay mode evaluator for hard-label predictions ({-1, 0, 1, 2, 10, 11, 15}).

    Labels of -1 are mapped to 15 (Rare). A one-hot array is constructed as
    ``pred_proba`` so that ``DecayModeROCPlot`` produces a single point per
    class on the ROC plane rather than a curve.
    """

    # Define the mapping as a class constant so it's available before super().__init__
    _FORWARD_MAPPING = {0: 0, 1: 1, 2: 2, 10: 3, 11: 4, 15: 5}
    _N_CLASSES = 6

    def __init__(
        self,
        predicted: np.array,
        truth: np.array,
        output_dir: str = "",
        sample: str = "all",
        algorithm: str = "all",
    ):
        predicted = np.asarray(predicted, dtype=int)
        predicted = np.where(predicted == -1, 15, predicted)
        predicted_idx = np.vectorize(self._FORWARD_MAPPING.get)(predicted)
        pred_proba = np.eye(self._N_CLASSES)[predicted_idx]
        super().__init__(
            predicted_idx, pred_proba, truth, output_dir, sample, algorithm
        )


class ConfusionMatrix:
    def __init__(self, evaluator: DecayModeEvaluator):
        self.evaluator = evaluator
        self.fig, self.ax = self.plot()

    def plot(self):
        fig, ax = visualize_confusion_matrix(
            histogram=self.evaluator.normalized_confusion_matrix,
            categories=self.evaluator.categories,
        )
        return fig, ax

    def save(self, output_dir):
        output_path = os.path.join(
            output_dir, f"decay_mode_cm_{self.evaluator.algorithm}.pdf"
        )
        self.fig.tight_layout(pad=1.5)
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


# Example use:
#     dm_evaluator = DecayModeEvaluator(true_classes, pred_classes, '/path/to/output')
#     dm_evaluator.print_performance()


class DecayModeROCPlot:
    """ROC curve plot for decay mode classification.

    For each decay mode class the class is treated as signal and all other
    classes as background, then a ROC curve (TPR vs FPR) is drawn.
    """

    COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    def __init__(self, evaluator: DecayModeEvaluator):
        """Args:
        predictions_proba: (N, num_classes) softmax probability array.
        targets: (N,) integer class-index array.
        categories: list of class label strings in class-index order.
        """
        self.evaluator = evaluator
        self.predictions_proba = np.asarray(evaluator.pred_proba)
        self.targets = np.asarray(evaluator.truth)
        self.categories = evaluator.categories
        self.fig, self.ax = self._make_axes()
        self._plot_roc_curves()

    def _make_axes(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel("False Positive Rate", fontsize=20)
        ax.set_ylabel("True Positive Rate", fontsize=20)
        ax.tick_params(axis="both", labelsize=16)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
        plt.grid()
        return fig, ax

    def _plot_roc_curves(self):
        num_classes = self.predictions_proba.shape[1]
        for cls_idx in range(num_classes):
            binary_targets = (self.targets == cls_idx).astype(int)
            if binary_targets.sum() == 0:
                continue
            fpr, tpr, _ = metrics.roc_curve(
                binary_targets, self.predictions_proba[:, cls_idx]
            )
            auc = metrics.auc(fpr, tpr)
            label = f"{self.categories[cls_idx]} (AUC={auc:.3f})"
            color = self.COLORS[cls_idx % len(self.COLORS)]
            self.ax.plot(fpr, tpr, label=label, color=color, lw=2)
        self.ax.legend(prop={"size": 14}, loc="lower right")

    def save(self, output_dir: str):
        output_path = os.path.join(
            output_dir, f"decay_mode_ROC_{self.evaluator.algorithm}.pdf"
        )
        self.fig.savefig(output_path, bbox_inches="tight")
        plt.close("all")


class DecayModeComparisonPlot:
    def __init__(
        self,
        cfg,
        ymin: float = 0.0,
        ymax: float = 1.0,
        figsize: tuple = (10, 10),
        metric: str = "F1",
    ):
        self.cfg = cfg
        self.ymin = ymin
        self.ymax = ymax
        self.figsize = figsize
        self.metric = metric
        self.decay_modes = {
            0: {"label": r"$h^\pm$", "PDG_ratio": 0.1777},
            1: {"label": r"$h^\pm+\pi^0$", "PDG_ratio": 0.4002},
            2: {"label": r"$h^\pm+\geq2\pi^0$", "PDG_ratio": 0.1668},
            3: {"label": r"$h^\pm h^\mp h^\pm$", "PDG_ratio": 0.1513},
            4: {
                "label": r"$h^\pm h^\mp h^\pm$" "\n" r"$+\geq\pi^0$",
                "PDG_ratio": 0.0816,
            },
            5: {"label": "Rare", "PDG_ratio": 0.0224},
        }
        self.fig, self.ax = self.plot()

    def plot(self):
        x = range(len(self.decay_modes.keys()) + 1)
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlabel("Decay Modes")
        ax.set_ylabel(self.metric, x=1.05)
        # ax.set_title(f"Classification Precision of the DMs for {dataset}", y=1.05)
        ax.set_xticks(x)
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=True
        )
        ax.tick_params(axis="y", which="both", left=True, right=False, labelleft=True)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xticklabels(
            [info["label"] for info in self.decay_modes.values()] + ["Overall"],
            rotation=45,
        )
        # Vertical lines between the DMs
        for i in range(len(x)):
            ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=0.5, zorder=0)
        # Explicit margins: same bottom/top as confusion matrix so axes boxes
        # share the same height; smaller left since no tilted tick labels.
        fig.subplots_adjust(left=0.12, bottom=0.22, right=0.83, top=0.95)
        return fig, ax

    def _calculate_overall_performance(self, evaluator):
        metric_value = np.array(evaluator.class_performances[self.metric])
        brs = np.array([value["PDG_ratio"] for value in self.decay_modes.values()])
        overall = np.sum(metric_value * brs)
        return overall

    def _annotate_points(self, x, y, offset):
        # Function to add labels to the datapoints
        for ii, (i, j) in enumerate(zip(x, y)):
            self.ax.annotate(
                f"{j:.3f}",
                (i, j),
                textcoords="offset points",
                xytext=offset,
                ha="center",
                va="bottom",
                fontsize=12,
            )

    def add_line(self, evaluator, offset, annotation_offset=(-18, -4)):
        location = np.arange(len(self.decay_modes.keys()) + 1) + offset
        overall_score = self._calculate_overall_performance(evaluator)
        metric_values = list(evaluator.class_performances[self.metric]) + [
            overall_score
        ]
        self.ax.scatter(
            location,
            metric_values,
            label=evaluator.algorithm,
            color=self.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm].color,
            marker=self.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm].marker,
            s=100,
        )
        self._annotate_points(location, metric_values, annotation_offset)
        self.ax.legend(
            loc="lower left",
            shadow=True,
            fancybox=True,
            framealpha=1,
            borderpad=1,
            fontsize=20,
        )

    def save(self, output_dir: str):
        self.fig.savefig(
            os.path.join(output_dir, f"decay_mode_{self.metric}.pdf"),
            bbox_inches="tight",
            format="pdf",
        )
        plt.close("all")


def get_offsets(n: int):
    """Calculates the offsets for n nuber of algorithms"""
    if n < 6:
        delta_offset = 0.2
    elif n < 12:
        delta_offset = 0.1
    else:
        raise ValueError("Having more than 12 algorithms will bee too crowded")
    left_most_bin = (1 - n) * (delta_offset / 2)
    offsets = [left_most_bin + (i * delta_offset) for i in range(n)]
    return offsets

annotation_offsets = [(0, 9), (0, -18)]

class DecayModeMultiEvaluator:
    def __init__(self, output_dir: str, cfg: DictConfig, sample: str):
        self.output_dir = output_dir
        self.cfg = cfg
        self.sample = sample
        self.cfg = cfg
        self.dmrps = []
        self.cms = []
        self.dmcp = DecayModeComparisonPlot(
            cfg=self.cfg, ymin=0.0, ymax=1.0, figsize=(10, 10), metric="F1"
        )

    def combine_results(self, evaluators: list):
        offsets = get_offsets(len(evaluators))
        for i, evaluator in enumerate(evaluators):
            self.cms.append(ConfusionMatrix(evaluator=evaluator))
            self.dmrps.append(DecayModeROCPlot(evaluator=evaluator))
            # self.dmcp.add_line(evaluator=evaluator, offset=offsets[i])
            self.dmcp.add_line(evaluator=evaluator, offset=offsets[i], annotation_offset=annotation_offsets[i % len(annotation_offsets)])

    def save(self):
        for drmp in self.dmrps:
            drmp.save(output_dir=self.output_dir)
        for cm in self.cms:
            cm.save(output_dir=self.output_dir)
        self.dmcp.save(output_dir=self.output_dir)
