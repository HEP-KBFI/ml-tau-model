import os
import json
from mltau.tools.io.general import NpEncoder
import numpy as np
import awkward as ak
import mplhep as hep
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

from omegaconf import DictConfig
from mltau.tools import general as g
from mltau.tools.evaluation.histogram import Histogram

plt.rcParams["mathtext.fontset"] = "stix"

RESULT_FIGSIZE = (10, 10)
RESULT_SUBPLOT_ADJUST = {"left": 0.16, "bottom": 0.14, "right": 0.96, "top": 0.96}
STACK_SUBPLOT_HSPACE = 0
STACK_YLIM_LOG_PADDING = 0.30
STACK_AXIS_LABEL_SIZE = 30
STACK_TICK_LABEL_SIZE = 30
STACK_Y_TICK_LABEL_X = -0.11
ROC_LEGEND_FONT_SIZE = 30
ROC_MARKER_SIZE = 12
ROC_MAX_MARKERS = 45
STACK_LEGEND_FONT_SIZE = ROC_LEGEND_FONT_SIZE


def _sparse_marker_step(values, max_markers=ROC_MAX_MARKERS):
    values = np.asarray(values)
    if len(values) <= max_markers:
        return 1
    return int(np.ceil(len(values) / max_markers))


def _lighten_color(color, amount=0.55):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1 - amount) * rgb + amount * np.ones_like(rgb))


def _get_charge_colors(cfg, algorithm):
    style = cfg.metrics.ALGORITHM_PLOT_STYLES[algorithm]
    base = style.color
    light = getattr(style, "light_color", None)
    if light is None:
        light = _lighten_color(base)
    return base, light


def _get_charge_label(cfg, algorithm):
    return cfg.metrics.ALGORITHM_PLOT_STYLES[algorithm].name


def _get_metric_plot_range(metric_cfg):
    if "x_plot_range" in metric_cfg:
        return tuple(metric_cfg.x_plot_range)
    return tuple(metric_cfg.x_range)


class BaseChargeIdEvaluator:
    """Shared infrastructure for charge ID evaluation.

    Subclasses must implement ``_get_wp_mask(charge)`` to define which events
    are selected at the working point, and must populate
    ``eff_denominator_masks`` and ``fake_denominator_masks`` before calling
    ``_build_wp_metrics()``.
    """

    def __init__(
        self,
        predicted: np.array,
        truth: np.array,
        gen_jet_tau_p4s: ak.Array,
        reco_jet_p4s: ak.Array,
        cfg: DictConfig,
        output_dir: str = "",
        sample: str = "",
        algorithm: str = "",
    ):
        self.output_dir = output_dir
        if self.output_dir != "":
            os.makedirs(self.output_dir, exist_ok=True)
        self.sample = sample
        self.algorithm = algorithm
        truth = np.asarray(truth)
        predicted = np.asarray(predicted)
        # Normalize truth to binary {0, 1}: 1 = positive charge, 0 = negative charge.
        # Accept either physical signed charge {-1, +1} or already-binary {0, 1}.
        if np.any(truth < 0):
            truth = (truth == 1).astype(int)
        else:
            truth = truth.astype(int)
        self.truth = truth
        self.predicted = predicted
        self.gen_jet_tau_p4s = g.reinitialize_p4(gen_jet_tau_p4s)
        self.reco_jet_p4s = g.reinitialize_p4(reco_jet_p4s)
        self.cfg = cfg
        self.true_positive_charge_mask = self.truth == 1
        self.true_negative_charge_mask = self.truth == 0

    def _base_mask(self):
        """Kinematic acceptance cuts applied to both gen and reco jets."""
        ref_var_pt_mask = self.gen_jet_tau_p4s.pt > self.cfg.metrics.charge.cuts.min_pt
        ref_var_theta_mask1 = (
            abs(np.rad2deg(self.gen_jet_tau_p4s.theta))
            < self.cfg.metrics.charge.cuts.max_theta
        )
        ref_var_theta_mask2 = (
            abs(np.rad2deg(self.gen_jet_tau_p4s.theta))
            > self.cfg.metrics.charge.cuts.min_theta
        )
        gen_denominator_mask = (
            ref_var_pt_mask * ref_var_theta_mask1 * ref_var_theta_mask2
        )
        # As we are only assigning charge for tau (candidates) that are tagged, then to have the correct total
        # number of jets, we need to add the cuts on the reco jet also to the denominator
        tau_pt_mask = self.reco_jet_p4s.pt > self.cfg.metrics.charge.cuts.min_pt
        tau_theta_mask1 = (
            abs(np.rad2deg(self.reco_jet_p4s.theta))
            < self.cfg.metrics.charge.cuts.max_theta
        )
        tau_theta_mask2 = (
            abs(np.rad2deg(self.reco_jet_p4s.theta))
            > self.cfg.metrics.charge.cuts.min_theta
        )
        reco_denominator_mask = tau_pt_mask * tau_theta_mask1 * tau_theta_mask2
        return gen_denominator_mask * reco_denominator_mask

    def _compute_denominator_masks(self, eff_fake: str):
        positive_mask = (
            self.true_positive_charge_mask
            if eff_fake == "eff"
            else self.true_negative_charge_mask
        )
        negative_mask = (
            self.true_negative_charge_mask
            if eff_fake == "eff"
            else self.true_positive_charge_mask
        )
        base = self._base_mask()
        return {
            "positive": base * positive_mask,
            "negative": base * negative_mask,
        }

    def _metric_values(self, metric: str):
        values = getattr(self.gen_jet_tau_p4s, metric).to_numpy()
        if metric == "theta":
            values = np.rad2deg(values)
        return np.asarray(values)

    def _get_wp_mask(self, charge: str) -> np.ndarray:
        """Return boolean mask for events selected at the working point.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _get_working_point_eff_fakes(self, name, metric, eff_fake="eff"):
        eff_fake_mask = (
            self.eff_denominator_masks
            if eff_fake == "eff"
            else self.fake_denominator_masks
        )
        return_values = {}
        var_values = getattr(self.gen_jet_tau_p4s, name).to_numpy()
        if name == "theta":
            var_values = np.rad2deg(var_values)
        for charge in ["positive", "negative"]:
            wp_mask = self._get_wp_mask(charge)
            eff_var_denom = var_values[eff_fake_mask[charge]]
            eff_var_num = var_values[wp_mask * eff_fake_mask[charge]]
            bin_edges = np.linspace(
                min(eff_var_denom), max(eff_var_denom), metric.n_bins + 1
            )
            denom_hist = Histogram(eff_var_denom, bin_edges, "denominator")
            num_hist = Histogram(eff_var_num, bin_edges, "numerator")
            efficiencies = num_hist / denom_hist
            return_values[charge] = (
                efficiencies.bin_centers,
                efficiencies.data,
                efficiencies.uncertainties,
                efficiencies.bin_halfwidths,
            )
        return return_values

    def _build_wp_metrics(self):
        self.wp_metrics = {}
        for name, metric in self.cfg.metrics.charge.metrics.items():
            self.wp_metrics[name] = {}
            charge_fr = self._get_working_point_eff_fakes(name, metric, eff_fake="fake")
            charge_eff = self._get_working_point_eff_fakes(name, metric, eff_fake="eff")
            for charge in ["negative", "positive"]:
                fr_bin_centers, fr_data, fr_yerr, fr_xerr = charge_fr[charge]
                eff_bin_centers, eff_data, eff_yerr, eff_xerr = charge_eff[charge]
                self.wp_metrics[name][charge] = {
                    "fakerates": fr_data,
                    "fr_bin_centers": fr_bin_centers,
                    "fr_yerr": fr_yerr,
                    "fr_xerr": fr_xerr,
                    "efficiencies": eff_data,
                    "eff_bin_centers": eff_bin_centers,
                    "eff_yerr": eff_yerr,
                    "eff_xerr": eff_xerr,
                }

    def compute_binned_efficiencies_for_target_eff(
        self, metric: str, target_eff: float = 0.8
    ):
        """Binned efficiencies at the working point.

        The default implementation uses the single hard-label working point
        (``target_eff`` is ignored).  Soft-score subclasses should override
        this to threshold at the requested efficiency.
        """
        charge_eff = self._get_working_point_eff_fakes(
            metric, self.cfg.metrics.charge.metrics[metric], eff_fake="eff"
        )
        x, pos, _, _ = charge_eff["positive"]
        _, neg, _, _ = charge_eff["negative"]
        return x, np.asarray(pos), np.asarray(neg)

    def compute_binned_fake_rates_for_target_eff(
        self, metric: str, target_eff: float = 0.8
    ):
        """Binned fake rates at the working point.

        See ``compute_binned_efficiencies_for_target_eff`` for notes on the
        default implementation.
        """
        charge_fake = self._get_working_point_eff_fakes(
            metric, self.cfg.metrics.charge.metrics[metric], eff_fake="fake"
        )
        x, pos, _, _ = charge_fake["positive"]
        _, neg, _, _ = charge_fake["negative"]
        return x, np.asarray(pos), np.asarray(neg)

    def compute_combined_roc(self):
        """Charge-combined ROC points using denominator-weighted averages."""
        eff_pos = np.asarray(self.efficiencies["positive"], dtype=float)
        eff_neg = np.asarray(self.efficiencies["negative"], dtype=float)
        fake_pos = np.asarray(self.fakerates["positive"], dtype=float)
        fake_neg = np.asarray(self.fakerates["negative"], dtype=float)

        eff_pos_den = np.sum(self.eff_denominator_masks["positive"])
        eff_neg_den = np.sum(self.eff_denominator_masks["negative"])
        fake_pos_den = np.sum(self.fake_denominator_masks["positive"])
        fake_neg_den = np.sum(self.fake_denominator_masks["negative"])

        efficiencies = np.divide(
            eff_pos * eff_pos_den + eff_neg * eff_neg_den,
            eff_pos_den + eff_neg_den,
            out=np.full_like(eff_pos, np.nan, dtype=float),
            where=(eff_pos_den + eff_neg_den) > 0,
        )
        fakerates = np.divide(
            fake_pos * fake_pos_den + fake_neg * fake_neg_den,
            fake_pos_den + fake_neg_den,
            out=np.full_like(fake_pos, np.nan, dtype=float),
            where=(fake_pos_den + fake_neg_den) > 0,
        )
        return efficiencies, fakerates

    def compute_combined_binned_fake_rates_for_target_eff(
        self, metric: str, target_eff: float = 0.8
    ):
        """Charge-combined binned fake rate at the evaluator's working point.

        For hard-label evaluators there is only one operating point, so
        ``target_eff`` is intentionally ignored.
        """
        values = self._metric_values(metric)
        metric_cfg = self.cfg.metrics.charge.metrics[metric]
        bin_edges = np.linspace(
            metric_cfg.x_range[0], metric_cfg.x_range[1], metric_cfg.n_bins + 1
        )
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        pred_charge_pos = self._get_wp_mask("positive")
        pred_charge_neg = self._get_wp_mask("negative")
        base_mask = self._base_mask()
        pos_truth = self.true_positive_charge_mask & base_mask
        neg_truth = self.true_negative_charge_mask & base_mask

        fake_rates = []
        for left, right in zip(bin_edges[:-1], bin_edges[1:]):
            metric_mask = (values >= left) & (values < right)
            pos_sel = metric_mask & pos_truth
            neg_sel = metric_mask & neg_truth
            numerator = np.sum(pred_charge_neg[pos_sel]) + np.sum(
                pred_charge_pos[neg_sel]
            )
            denominator = np.sum(pos_sel) + np.sum(neg_sel)
            fake_rates.append(numerator / denominator if denominator > 0 else np.nan)

        return centers, np.asarray(fake_rates)


class ChargeIdEvaluator(BaseChargeIdEvaluator):
    """Evaluates charge ID performance for continuous probability scores."""

    def __init__(
        self,
        predicted: np.array,
        truth: np.array,
        gen_jet_tau_p4s: ak.Array,
        reco_jet_p4s: ak.Array,
        cfg: DictConfig,
        output_dir: str = "",
        sample: str = "",
        algorithm: str = "",
    ):
        super().__init__(
            predicted,
            truth,
            gen_jet_tau_p4s,
            reco_jet_p4s,
            cfg,
            output_dir,
            sample,
            algorithm,
        )
        # Use quantile-based thresholds: dense where scores concentrate (near 0/1),
        # sparse in the middle. Capped at 1000 points for performance.
        self.tagging_cuts = np.unique(
            np.concatenate(
                [[0], np.quantile(self.predicted, np.linspace(0, 1, 1000)), [1]]
            )
        )
        self.efficiencies, self.eff_denominator_masks = self._calculate_eff_fake(
            eff_fake="eff"
        )
        self.fakerates, self.fake_denominator_masks = self._calculate_eff_fake(
            eff_fake="fake"
        )
        self.pos_charge_predictions = self.predicted[self.true_positive_charge_mask]
        self.neg_charge_predictions = self.predicted[self.true_negative_charge_mask]
        # Find the working point where average efficiency is 95%
        self.wp_idx = self._find_95_percent_average_efficiency_working_point()
        self.wp_pos = float(self.tagging_cuts[self.wp_idx])
        self.wp_neg = float(1.0 - self.tagging_cuts[self.wp_idx])
        # Calculate confusion matrix at working point
        self.confusion_matrix = self._calculate_confusion_matrix()
        self._build_wp_metrics()

    def _get_wp_mask(self, charge: str) -> np.ndarray:
        if charge == "positive":
            return self.predicted >= self.wp_pos
        else:
            return self.predicted <= self.wp_neg

    def choose_threshold_for_target_eff_per_class(
        self, target_eff: float = 0.8, which: str = "positive"
    ) -> float:
        eff = np.asarray(
            self.efficiencies["positive" if which == "positive" else "negative"],
            dtype=float,
        )
        thr = np.asarray(self.tagging_cuts, dtype=float)
        mask = np.isfinite(eff) & np.isfinite(thr)
        eff = eff[mask]
        thr = thr[mask]
        if eff.size == 0:
            raise RuntimeError("No finite ROC points found for threshold selection")
        order = np.argsort(eff)
        eff_sorted = eff[order]
        thr_sorted = thr[order]
        eff_unique, unique_idx = np.unique(eff_sorted, return_index=True)
        thr_unique = thr_sorted[unique_idx]
        if eff_unique.size == 1:
            threshold = float(thr_unique[0])
            return threshold if which == "positive" else float(1.0 - threshold)
        clipped_eff = np.clip(target_eff, eff_unique[0], eff_unique[-1])
        threshold = float(np.interp(clipped_eff, eff_unique, thr_unique))
        return threshold if which == "positive" else float(1.0 - threshold)

    def choose_threshold_for_target_eff(self, target_eff: float = 0.8) -> float:
        """Symmetric score threshold closest to a charge-combined efficiency."""
        efficiencies, _ = self.compute_combined_roc()
        efficiencies = np.asarray(efficiencies, dtype=float)
        mask = np.isfinite(efficiencies)
        if not np.any(mask):
            raise RuntimeError("No finite ROC points found for threshold selection")
        valid_indices = np.flatnonzero(mask)
        idx = valid_indices[np.argmin(np.abs(efficiencies[mask] - target_eff))]
        return float(self.tagging_cuts[idx])

    def _calculate_eff_fake(self, eff_fake: str = "eff"):
        _eff_fake = {"positive": [], "negative": []}
        denominator_masks = self._compute_denominator_masks(eff_fake)
        pos_denominator_mask = denominator_masks["positive"]
        neg_denominator_mask = denominator_masks["negative"]
        pos_all = np.sum(pos_denominator_mask)
        neg_all = np.sum(neg_denominator_mask)
        for cut in self.tagging_cuts:
            pos_passing_cut = np.sum(self.predicted[pos_denominator_mask] >= cut)
            # Alternative 1: Use same threshold for both (current method)
            neg_passing_cut = np.sum((1 - self.predicted[neg_denominator_mask]) >= cut)

            # Alternative 2: Use direct thresholding (uncomment to test)
            # Treat negative charge as predictions < threshold (more natural)
            # neg_passing_cut = np.sum(self.predicted[neg_denominator_mask] <= (1 - cut))

            # Alternative 3: Use calibrated thresholding around model's natural bias
            # model_bias = np.mean(self.predicted)  # Estimated model bias
            # neg_threshold = 2 * model_bias - cut  # Symmetric around bias point
            # neg_passing_cut = np.sum(self.predicted[neg_denominator_mask] <= neg_threshold)

            _eff_fake["positive"].append(
                pos_passing_cut / pos_all if pos_all > 0 else 0.0
            )
            _eff_fake["negative"].append(
                neg_passing_cut / neg_all if neg_all > 0 else 0.0
            )
        return _eff_fake, denominator_masks

    def _find_95_percent_average_efficiency_working_point(
        self, target_avg_efficiency: float = 0.95
    ) -> int:
        """Return the index into tagging_cuts where average efficiency is closest to target."""
        eff_pos = np.array(self.efficiencies["positive"])
        eff_neg = np.array(self.efficiencies["negative"])
        avg_efficiencies = (eff_pos + eff_neg) / 2.0
        return int(np.argmin(np.abs(avg_efficiencies - target_avg_efficiency)))

    def _calculate_confusion_matrix(self):
        positive_predictions = self.predicted >= self.wp_pos
        tp = np.sum(positive_predictions & (self.truth == 1))
        tn = np.sum(~positive_predictions & (self.truth == 0))
        fp = np.sum(positive_predictions & (self.truth == 0))
        fn = np.sum(~positive_predictions & (self.truth == 1))
        return {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}

    def compute_binned_fake_rates_for_target_eff(
        self, metric: str, target_eff: float = 0.8
    ):
        values = self._metric_values(metric)
        metric_cfg = self.cfg.metrics.charge.metrics[metric]
        bin_edges = np.linspace(
            metric_cfg.x_range[0], metric_cfg.x_range[1], metric_cfg.n_bins + 1
        )
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        threshold_pos = self.choose_threshold_for_target_eff_per_class(
            target_eff, "positive"
        )
        threshold_neg = self.choose_threshold_for_target_eff_per_class(
            target_eff, "negative"
        )

        pred_charge_pos = self.predicted >= threshold_pos
        pred_charge_neg = self.predicted < threshold_neg

        base_mask = self._base_mask()
        pos_truth = self.true_positive_charge_mask & base_mask
        neg_truth = self.true_negative_charge_mask & base_mask

        fake_pos = []
        fake_neg = []
        for left, right in zip(bin_edges[:-1], bin_edges[1:]):
            metric_mask = (values >= left) & (values < right)
            neg_sel = metric_mask & neg_truth
            pos_sel = metric_mask & pos_truth
            fake_pos.append(
                np.mean(pred_charge_pos[neg_sel]) if np.any(neg_sel) else np.nan
            )
            fake_neg.append(
                np.mean(pred_charge_neg[pos_sel]) if np.any(pos_sel) else np.nan
            )

        return centers, np.asarray(fake_pos), np.asarray(fake_neg)

    def compute_binned_efficiencies_for_target_eff(
        self, metric: str, target_eff: float = 0.8
    ):
        values = self._metric_values(metric)
        metric_cfg = self.cfg.metrics.charge.metrics[metric]
        bin_edges = np.linspace(
            metric_cfg.x_range[0], metric_cfg.x_range[1], metric_cfg.n_bins + 1
        )
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        threshold_pos = self.choose_threshold_for_target_eff_per_class(
            target_eff, "positive"
        )
        threshold_neg = self.choose_threshold_for_target_eff_per_class(
            target_eff, "negative"
        )

        pred_charge_pos = self.predicted >= threshold_pos
        pred_charge_neg = self.predicted < threshold_neg

        base_mask = self._base_mask()
        pos_truth = self.true_positive_charge_mask & base_mask
        neg_truth = self.true_negative_charge_mask & base_mask

        eff_pos = []
        eff_neg = []
        for left, right in zip(bin_edges[:-1], bin_edges[1:]):
            metric_mask = (values >= left) & (values < right)
            pos_sel = metric_mask & pos_truth
            neg_sel = metric_mask & neg_truth
            eff_pos.append(
                np.mean(pred_charge_pos[pos_sel]) if np.any(pos_sel) else np.nan
            )
            eff_neg.append(
                np.mean(pred_charge_neg[neg_sel]) if np.any(neg_sel) else np.nan
            )

        return centers, np.asarray(eff_pos), np.asarray(eff_neg)

    def compute_combined_binned_fake_rates_for_target_eff(
        self, metric: str, target_eff: float = 0.8
    ):
        values = self._metric_values(metric)
        metric_cfg = self.cfg.metrics.charge.metrics[metric]
        bin_edges = np.linspace(
            metric_cfg.x_range[0], metric_cfg.x_range[1], metric_cfg.n_bins + 1
        )
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        threshold = self.choose_threshold_for_target_eff(target_eff)

        pred_charge_pos = self.predicted >= threshold
        pred_charge_neg = self.predicted <= (1.0 - threshold)
        base_mask = self._base_mask()
        pos_truth = self.true_positive_charge_mask & base_mask
        neg_truth = self.true_negative_charge_mask & base_mask

        fake_rates = []
        for left, right in zip(bin_edges[:-1], bin_edges[1:]):
            metric_mask = (values >= left) & (values < right)
            pos_sel = metric_mask & pos_truth
            neg_sel = metric_mask & neg_truth
            numerator = np.sum(pred_charge_neg[pos_sel]) + np.sum(
                pred_charge_pos[neg_sel]
            )
            denominator = np.sum(pos_sel) + np.sum(neg_sel)
            fake_rates.append(numerator / denominator if denominator > 0 else np.nan)

        return centers, np.asarray(fake_rates)


class HardLabelChargeIdEvaluator(BaseChargeIdEvaluator):
    """Evaluates charge ID performance for hard-label predictions ({-1, 0, +1}).

    Unlike ChargeIdEvaluator, there is no threshold scanning.  Efficiencies and
    fakerates are single-element arrays representing the one operating point
    defined directly by the hard labels, so they can be plotted as a single
    marker on a ROC curve alongside continuous-score evaluators.
    """

    def __init__(
        self,
        predicted: np.array,
        truth: np.array,
        gen_jet_tau_p4s: ak.Array,
        reco_jet_p4s: ak.Array,
        cfg: DictConfig,
        output_dir: str = "",
        sample: str = "",
        algorithm: str = "",
    ):
        super().__init__(
            predicted,
            truth,
            gen_jet_tau_p4s,
            reco_jet_p4s,
            cfg,
            output_dir,
            sample,
            algorithm,
        )
        # Hard prediction masks: +1 → predicted positive charge, -1 → predicted negative
        self._pred_positive_mask = self.predicted == 1
        self._pred_negative_mask = self.predicted == -1
        # For interface compatibility with ChargeClassifierPlot
        self.pos_charge_predictions = self.predicted[self.true_positive_charge_mask]
        self.neg_charge_predictions = self.predicted[self.true_negative_charge_mask]
        self.eff_denominator_masks = self._compute_denominator_masks(eff_fake="eff")
        self.fake_denominator_masks = self._compute_denominator_masks(eff_fake="fake")
        # Single-point efficiencies and fakerates as 1-element arrays
        self.efficiencies = self._calculate_single_eff_fake(eff_fake="eff")
        self.fakerates = self._calculate_single_eff_fake(eff_fake="fake")
        # Confusion matrix at the single operating point
        self.confusion_matrix = self._calculate_confusion_matrix()
        self._build_wp_metrics()

    def _get_wp_mask(self, charge: str) -> np.ndarray:
        return (
            self._pred_positive_mask
            if charge == "positive"
            else self._pred_negative_mask
        )

    def _calculate_single_eff_fake(self, eff_fake: str):
        denom_masks = (
            self.eff_denominator_masks
            if eff_fake == "eff"
            else self.fake_denominator_masks
        )
        pos_denom = denom_masks["positive"]
        neg_denom = denom_masks["negative"]
        pos_all = np.sum(pos_denom)
        neg_all = np.sum(neg_denom)
        pos_passing = np.sum(self._pred_positive_mask[pos_denom])
        neg_passing = np.sum(self._pred_negative_mask[neg_denom])
        eff_pos = float(pos_passing) / float(pos_all) if pos_all > 0 else 0.0
        eff_neg = float(neg_passing) / float(neg_all) if neg_all > 0 else 0.0
        return {
            "positive": np.array([eff_pos]),
            "negative": np.array([eff_neg]),
        }

    def _calculate_confusion_matrix(self):
        pred_neutral = self.predicted == 0
        # truth is binary {0, 1} after base class normalization
        tp = int(np.sum(self._pred_positive_mask & self.true_positive_charge_mask))
        tn = int(np.sum(self._pred_negative_mask & self.true_negative_charge_mask))
        fp = int(np.sum(self._pred_positive_mask & self.true_negative_charge_mask))
        fn = int(np.sum(self._pred_negative_mask & self.true_positive_charge_mask))
        # predicted==0 is an efficiency loss for both classes
        neutral_pos = int(np.sum(pred_neutral & self.true_positive_charge_mask))
        neutral_neg = int(np.sum(pred_neutral & self.true_negative_charge_mask))
        return {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "neutral_pos": neutral_pos,
            "neutral_neg": neutral_neg,
        }


class ChargeClassifierPlot:
    def __init__(self):
        self.bin_edges = np.linspace(start=0, stop=1, num=21)
        self.fig, self.ax = self.plot()

    def add_line(self, evaluator, dataset: str):
        linestyle = "solid" if dataset == "test" else "dashed"
        pos_color, neg_color = _get_charge_colors(evaluator.cfg, evaluator.algorithm)
        algorithm_label = _get_charge_label(evaluator.cfg, evaluator.algorithm)
        neg_histogram = np.histogram(
            evaluator.neg_charge_predictions, bins=self.bin_edges
        )[0]
        neg_histogram = neg_histogram / np.sum(neg_histogram)
        pos_histogram = np.histogram(
            evaluator.pos_charge_predictions, bins=self.bin_edges
        )[0]
        pos_histogram = pos_histogram / np.sum(pos_histogram)
        hep.histplot(
            pos_histogram,
            bins=self.bin_edges,
            histtype="fill",
            label=rf"{algorithm_label} $\tau^{{+}}$",
            color=pos_color,
            alpha=0.6,
            hatch="\\",
            edgecolor=pos_color,
            ax=self.ax,
        )
        hep.histplot(
            neg_histogram,
            bins=self.bin_edges,
            histtype="fill",
            label=rf"{algorithm_label} $\tau^{{-}}$",
            color=neg_color,
            alpha=0.6,
            hatch="//",
            edgecolor=neg_color,
            ax=self.ax,
        )
        self.ax.legend(prop={"size": 28}, loc="upper center")

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel(r"$\mathcal{D}$", fontdict={"size": 28})
        ax.set_yscale("log")
        ax.set_ylabel("Relative yield / bin")
        return fig, ax

    def save(self, output_path: str):
        self.fig.tight_layout(pad=1.5)
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class ROCPlot:
    def __init__(self, cfg, split_by_charge: bool = False):
        self.fig, self.ax = self.plot()
        self.cfg = cfg
        self.comparison_algos = []
        self.split_by_charge = split_by_charge

    def plot(self):
        fig, ax = plt.subplots(figsize=RESULT_FIGSIZE)
        ax.set_ylabel(r"$P_{misid}$", fontsize=30)
        ax.set_xlabel(r"$\varepsilon_{\tau}$", fontsize=30)
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=30)
        ax.set_ylim((1e-5, 1))
        ax.set_xlim((0, 1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.set_yscale("log")
        return fig, ax

    def add_line(self, evaluator):
        self.comparison_algos.append(evaluator.algorithm)
        pos_color, neg_color = _get_charge_colors(evaluator.cfg, evaluator.algorithm)
        style = evaluator.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm]
        marker = style.marker
        algorithm_label = _get_charge_label(evaluator.cfg, evaluator.algorithm)
        if self.split_by_charge:
            # ML prediction curves
            self.ax.plot(
                evaluator.efficiencies["positive"],
                evaluator.fakerates["positive"],
                color=pos_color,
                marker=marker,
                label=None,
                ms=ROC_MARKER_SIZE,
                markevery=_sparse_marker_step(evaluator.efficiencies["positive"]),
                ls="",
            )
            self.ax.plot(
                evaluator.efficiencies["negative"],
                evaluator.fakerates["negative"],
                color=neg_color,
                marker=marker,
                label=None,
                ms=ROC_MARKER_SIZE,
                markevery=_sparse_marker_step(evaluator.efficiencies["negative"]),
                ls="",
            )
        else:
            efficiencies, fakerates = evaluator.compute_combined_roc()
            self.ax.plot(
                efficiencies,
                fakerates,
                color=pos_color,
                marker=marker,
                label=algorithm_label,
                ms=ROC_MARKER_SIZE,
                markevery=_sparse_marker_step(efficiencies),
                ls="",
            )

    # Create a custom legend for the ROC
    def finalize_legend(self):
        handles = []

        # --- Models (ONLY positive colors)

        for algo in self.comparison_algos:
            style = self.cfg.metrics.ALGORITHM_PLOT_STYLES[algo]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style.color,
                    marker=style.marker,
                    linestyle="",
                    label=style.name,
                    markersize=ROC_MARKER_SIZE,
                )
            )

        self.ax.legend(
            handles=handles, loc="upper left", prop={"size": ROC_LEGEND_FONT_SIZE}
        )

    def save(self, output_path):
        self.fig.subplots_adjust(**RESULT_SUBPLOT_ADJUST)
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class TargetEfficiencyFakeRateStackPlot:
    def __init__(self, cfg: DictConfig, metric: str, target_efficiencies):
        self.cfg = cfg
        self.metric = metric
        self.target_efficiencies = list(target_efficiencies)
        self._panel_y_values = [[] for _ in self.target_efficiencies]
        self.fig, self.axes = self.plot()

    def add_line(self, evaluator):
        style = evaluator.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm]
        algorithm_label = _get_charge_label(evaluator.cfg, evaluator.algorithm)
        color = style.color
        marker = style.marker
        linestyle = getattr(style, "ls", "solid")
        linewidth = getattr(style, "lw", 2)
        marker_size = getattr(style, "marker_size", 8)

        for panel_idx, (ax, target_eff) in enumerate(
            zip(self.axes, self.target_efficiencies)
        ):
            x, fakerates = evaluator.compute_combined_binned_fake_rates_for_target_eff(
                self.metric, target_eff=target_eff
            )
            valid = np.isfinite(x) & np.isfinite(fakerates)
            plotted_fakerates = np.asarray(fakerates)[valid]
            self._panel_y_values[panel_idx].extend(plotted_fakerates)
            ax.plot(
                np.asarray(x)[valid],
                plotted_fakerates,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=linewidth,
                markersize=marker_size,
                label=algorithm_label,
            )

    def plot(self):
        fig, axes = plt.subplots(
            len(self.target_efficiencies),
            1,
            figsize=RESULT_FIGSIZE,
            sharex=True,
            gridspec_kw={"hspace": STACK_SUBPLOT_HSPACE},
        )
        if len(self.target_efficiencies) == 1:
            axes = [axes]

        locator = ticker.MultipleLocator(
            self.cfg.metrics.charge.metrics[self.metric].x_maj_tick_spacing
        )
        xmin, xmax = _get_metric_plot_range(
            self.cfg.metrics.charge.metrics[self.metric]
        )

        for ax, target_eff in zip(axes, self.target_efficiencies):
            ax.xaxis.set_major_locator(locator)
            ax.set_xlim(xmin, xmax)
            ax.set_ylabel("")
            ax.set_yscale(self.cfg.metrics.charge.performances.fakerate.yscale)
            ax.tick_params(axis="x", labelsize=STACK_TICK_LABEL_SIZE)
            ax.tick_params(axis="y", labelsize=STACK_TICK_LABEL_SIZE)
            ax.text(
                0.02,
                0.92,
                rf"$\varepsilon_\tau = {100 * target_eff:.0f}\%$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=20,
            )

        axes[-1].set_xlabel(
            rf"{self.cfg.metrics.charge.performances.fakerate.xlabel[self.metric]}",
            fontsize=STACK_AXIS_LABEL_SIZE,
        )
        axes[0].set_ylabel(
            rf"{self.cfg.metrics.charge.performances.fakerate.ylabel}",
            fontsize=STACK_AXIS_LABEL_SIZE,
        )
        fig.subplots_adjust(**RESULT_SUBPLOT_ADJUST, hspace=STACK_SUBPLOT_HSPACE)
        return fig, axes

    def finalize_legend(self):
        # Legend intentionally disabled for the stacked target-efficiency fake-rate plot.
        # self.axes[0].legend(
        #     loc="upper left", frameon=False, prop={"size": STACK_LEGEND_FONT_SIZE}
        # )
        pass

    def _set_auto_ylims(self):
        for ax, y_values in zip(self.axes, self._panel_y_values):
            y_values = np.asarray(y_values, dtype=float)
            valid = y_values[np.isfinite(y_values) & (y_values > 0)]
            if len(valid) == 0:
                continue
            ymin = 10 ** (
                np.floor(np.log10(np.min(valid))) - STACK_YLIM_LOG_PADDING
            )
            ymax = 10 ** (
                np.ceil(np.log10(np.max(valid))) + STACK_YLIM_LOG_PADDING
            )
            if ymin == ymax:
                ymax *= 10
            ax.set_ylim(ymin, ymax)

    def _align_y_tick_labels(self):
        for ax in self.axes:
            for label in ax.get_yticklabels(which="major"):
                label.set_horizontalalignment("left")
                label.set_x(STACK_Y_TICK_LABEL_X)

    def save(self, output_path):
        self._set_auto_ylims()
        self._align_y_tick_labels()
        self.fig.subplots_adjust(**RESULT_SUBPLOT_ADJUST, hspace=STACK_SUBPLOT_HSPACE)
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class EfficiencyPlot:
    def __init__(self, cfg: DictConfig, metric: str):
        self.cfg = cfg
        self.metric = metric
        self.fig, self.ax = self.plot()

    def add_line(self, evaluator):
        pos_color, neg_color = _get_charge_colors(evaluator.cfg, evaluator.algorithm)
        marker = evaluator.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm].marker
        algorithm_label = _get_charge_label(evaluator.cfg, evaluator.algorithm)
        self.ax.errorbar(
            evaluator.wp_metrics[self.metric]["negative"]["eff_bin_centers"],
            evaluator.wp_metrics[self.metric]["negative"]["efficiencies"],
            xerr=evaluator.wp_metrics[self.metric]["negative"]["eff_xerr"],
            yerr=evaluator.wp_metrics[self.metric]["negative"]["eff_yerr"],
            ms=20,
            color=neg_color,
            marker=marker,
            linestyle="",
            label=rf"{algorithm_label} $\tau^{{-}}$",
        )
        self.ax.errorbar(
            evaluator.wp_metrics[self.metric]["positive"]["eff_bin_centers"],
            evaluator.wp_metrics[self.metric]["positive"]["efficiencies"],
            xerr=evaluator.wp_metrics[self.metric]["positive"]["eff_xerr"],
            yerr=evaluator.wp_metrics[self.metric]["positive"]["eff_yerr"],
            ms=20,
            color=pos_color,
            marker=marker,
            linestyle="",
            label=rf"{algorithm_label} $\tau^{{+}}$",
        )
        self.ax.legend(loc="upper right", prop={"size": 30})

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(
            _get_metric_plot_range(self.cfg.metrics.charge.metrics[self.metric])
        )
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(
                self.cfg.metrics.charge.metrics[self.metric].x_maj_tick_spacing
            )
        )
        ax.set_xlabel(
            rf"{self.cfg.metrics.charge.performances.efficiency.xlabel[self.metric]}",
            fontsize=30,
        )
        ax.set_ylabel(
            rf"{self.cfg.metrics.charge.performances.efficiency.ylabel}",
            fontsize=30,
        )
        ax.set_yscale(self.cfg.metrics.charge.performances.efficiency.yscale)
        if self.cfg.metrics.charge.performances.efficiency.ylim is not None:
            ylim = tuple(self.cfg.metrics.charge.performances.efficiency.ylim)
        else:
            ylim = self.cfg.metrics.charge.performances.efficiency.ylim
        ax.set_ylim(tuple(ylim))
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=30)
        return fig, ax

    def save(self, output_path):
        self.fig.tight_layout(pad=1.5)
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class FakeRatePlot:
    def __init__(self, cfg: DictConfig, metric: str):
        self.cfg = cfg
        self.metric = metric
        self.fig, self.ax = self.plot()

    def add_line(self, evaluator):
        pos_color, neg_color = _get_charge_colors(evaluator.cfg, evaluator.algorithm)
        marker = evaluator.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm].marker
        algorithm_label = _get_charge_label(evaluator.cfg, evaluator.algorithm)
        self.ax.errorbar(
            evaluator.wp_metrics[self.metric]["negative"]["fr_bin_centers"],
            evaluator.wp_metrics[self.metric]["negative"]["fakerates"],
            xerr=evaluator.wp_metrics[self.metric]["negative"]["fr_xerr"],
            yerr=evaluator.wp_metrics[self.metric]["negative"]["fr_yerr"],
            ms=20,
            color=neg_color,
            marker=marker,
            linestyle="",
            label=rf"{algorithm_label} $\tau^{{-}}$",
        )
        self.ax.errorbar(
            evaluator.wp_metrics[self.metric]["positive"]["fr_bin_centers"],
            evaluator.wp_metrics[self.metric]["positive"]["fakerates"],
            xerr=evaluator.wp_metrics[self.metric]["positive"]["fr_xerr"],
            yerr=evaluator.wp_metrics[self.metric]["positive"]["fr_yerr"],
            ms=20,
            color=pos_color,
            marker=marker,
            linestyle="",
            label=rf"{algorithm_label} $\tau^{{+}}$",
        )
        self.ax.legend(loc="upper right", prop={"size": 30})

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(
            _get_metric_plot_range(self.cfg.metrics.charge.metrics[self.metric])
        )
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(
                self.cfg.metrics.charge.metrics[self.metric].x_maj_tick_spacing
            )
        )
        ax.set_xlabel(
            rf"{self.cfg.metrics.charge.performances.fakerate.xlabel[self.metric]}",
            fontsize=30,
        )
        ax.set_ylabel(
            rf"{self.cfg.metrics.charge.performances.fakerate.ylabel}", fontsize=30
        )
        ax.set_yscale(self.cfg.metrics.charge.performances.fakerate.yscale)
        if self.cfg.metrics.charge.performances.fakerate.ylim is not None:
            ylim = tuple(self.cfg.metrics.charge.performances.fakerate.ylim)
        else:
            ylim = self.cfg.metrics.charge.performances.fakerate.ylim
        ax.set_ylim(ylim)
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=30)
        return fig, ax

    def save(self, output_path):
        self.fig.tight_layout(pad=1.5)
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class AsymmetryPlot:
    def __init__(self, cfg: DictConfig, metric: str, quantity: str):
        self.cfg = cfg
        self.metric = metric
        self.quantity = quantity
        self.target_eff = 0.8
        self.fig, (self.ax, self.ax_ratio) = self.plot()

    def _series(self, evaluator):
        if self.quantity == "efficiency":
            x, pos, neg = evaluator.compute_binned_efficiencies_for_target_eff(
                self.metric, target_eff=self.target_eff
            )
        else:
            x, pos, neg = evaluator.compute_binned_fake_rates_for_target_eff(
                self.metric, target_eff=self.target_eff
            )
        return np.asarray(x), np.asarray(pos), np.asarray(neg)

    def add_line(self, evaluator):
        x, pos, neg = self._series(evaluator)
        pos_color, neg_color = _get_charge_colors(evaluator.cfg, evaluator.algorithm)
        marker = evaluator.cfg.metrics.ALGORITHM_PLOT_STYLES[evaluator.algorithm].marker
        algorithm_label = _get_charge_label(evaluator.cfg, evaluator.algorithm)

        valid = np.isfinite(pos) & np.isfinite(neg)
        x = x[valid]
        pos = pos[valid]
        neg = neg[valid]

        self.ax.plot(
            x,
            pos,
            color=pos_color,
            marker=marker,
            linewidth=1.8,
            markersize=10,
            markevery=1,
            label=rf"{algorithm_label} $\tau^{{+}}$",
        )
        self.ax.plot(
            x,
            neg,
            color=neg_color,
            marker=marker,
            linewidth=1.8,
            markersize=10,
            markevery=1,
            label=rf"{algorithm_label} $\tau^{{-}}$",
        )

        ratio = np.divide(
            pos,
            neg,
            out=np.full_like(pos, np.nan, dtype=float),
            where=np.isfinite(neg) & (neg > 0),
        )
        self.ax_ratio.plot(
            x,
            ratio,
            color=pos_color,
            marker=marker,
            linewidth=1.6,
            markersize=10,
            markevery=1,
            label=algorithm_label,
        )

    def plot(self):
        fig, (ax, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(10, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
        )

        locator = ticker.MultipleLocator(
            self.cfg.metrics.charge.metrics[self.metric].x_maj_tick_spacing
        )
        ax.xaxis.set_major_locator(locator)
        ax_ratio.xaxis.set_major_locator(locator)

        xmin, xmax = _get_metric_plot_range(
            self.cfg.metrics.charge.metrics[self.metric]
        )
        ax.set_xlim(xmin, xmax)
        ax_ratio.set_xlim(xmin, xmax)

        if self.quantity == "efficiency":
            ylabel = self.cfg.metrics.charge.performances.efficiency.ylabel
            yscale = self.cfg.metrics.charge.performances.efficiency.yscale
            ylim = self.cfg.metrics.charge.performances.efficiency.ylim
        else:
            ylabel = self.cfg.metrics.charge.performances.fakerate.ylabel
            yscale = self.cfg.metrics.charge.performances.fakerate.yscale
            ylim = self.cfg.metrics.charge.performances.fakerate.ylim

        ax.set_ylabel(rf"{ylabel}", fontsize=24)
        ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(tuple(ylim))
        ax.tick_params(axis="x", labelsize=24)
        ax.tick_params(axis="y", labelsize=24)
        ax.tick_params(labelbottom=False)
        ax.text(
            0.98,
            0.95,
            rf"$\varepsilon_\tau = {self.target_eff:.1f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=18,
        )

        ax_ratio.set_xlabel(
            rf"{self.cfg.metrics.charge.performances.fakerate.xlabel[self.metric]}",
            fontsize=24,
        )
        ratio_label = (
            r"$\varepsilon(\tau^{+}) / \varepsilon(\tau^{-})$"
            if self.quantity == "efficiency"
            else r"$P_{\mathrm{misid}}(\tau^{+}) / P_{\mathrm{misid}}(\tau^{-})$"
        )
        ax_ratio.set_ylabel(ratio_label, fontsize=16)
        ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        ax_ratio.tick_params(axis="x", labelsize=24)
        ax_ratio.tick_params(axis="y", labelsize=20)
        return fig, (ax, ax_ratio)

    def save(self, output_path):
        self.fig.tight_layout(pad=1.5)
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class ConfusionMatrixPlot:
    """Plot confusion matrix for charge ID classification."""

    def __init__(self):
        self.fig, self.ax = self.plot()

    def add_data(self, evaluator):
        """Add confusion matrix data from evaluator."""
        cm = evaluator.confusion_matrix

        # Create 2x2 confusion matrix
        confusion_matrix = np.array(
            [
                [cm["TN"], cm["FP"]],  # Predicted Negative row
                [cm["FN"], cm["TP"]],  # Predicted Positive row
            ]
        )

        # Normalize confusion matrix (values sum to 1)
        total_sum = confusion_matrix.sum()
        if total_sum > 0:
            confusion_matrix_normalized = confusion_matrix / total_sum
        else:
            confusion_matrix_normalized = confusion_matrix

        # Create heatmap with matplotlib
        im = self.ax.imshow(confusion_matrix_normalized, cmap="Blues", aspect="auto")

        # Add text annotations with both normalized and raw counts
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                # Show both normalized (percentage) and raw count
                text = self.ax.text(
                    j,
                    i,
                    f"{confusion_matrix_normalized[i, j]:.3f}\n({confusion_matrix[i, j]})",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if confusion_matrix_normalized[i, j]
                        > confusion_matrix_normalized.max() / 2
                        else "black"
                    ),
                    fontsize=14,
                    fontweight="bold",
                )

        # Set labels and title
        self.ax.set_xticks([0, 1])
        self.ax.set_yticks([0, 1])
        self.ax.set_xticklabels(["Negative", "Positive"], fontsize=14)
        self.ax.set_yticklabels(["Negative", "Positive"], fontsize=14)
        self.ax.set_xlabel("Predicted Charge", fontsize=14)
        self.ax.set_ylabel("True Charge", fontsize=14)
        self.ax.set_title("Charge ID Confusion Matrix (Normalized)", fontsize=16)

        # Add colorbar
        cbar = plt.colorbar(im, ax=self.ax)
        cbar.set_label("Fraction", fontsize=14)

    def plot(self):
        """Create the basic plot structure."""
        fig, ax = plt.subplots(figsize=(8, 6))
        return fig, ax

    def save(self, output_path):
        self.fig.tight_layout(pad=1.5)
        self.fig.savefig(output_path, format="pdf")
        plt.close(self.fig)


class ChargeMultiEvaluator:
    def __init__(
        self,
        output_dir: str,
        cfg: DictConfig,
        target_efficiencies=(0.99, 0.80, 0.70),
    ):
        self.output_dir = output_dir
        self.cfg = cfg
        self.target_efficiencies = list(target_efficiencies)
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics = list(self.cfg.metrics.charge.metrics.keys())

        self.tagging_plots = {}
        self.efficiency_plots = {
            metric: EfficiencyPlot(self.cfg, metric) for metric in self.metrics
        }
        self.fakerate_plots = {
            metric: FakeRatePlot(self.cfg, metric) for metric in self.metrics
        }
        self.efficiency_asymmetry_plots = {
            metric: AsymmetryPlot(self.cfg, metric, "efficiency")
            for metric in self.metrics
        }
        self.fakerate_asymmetry_plots = {
            metric: AsymmetryPlot(self.cfg, metric, "fakerate")
            for metric in self.metrics
        }
        self.roc_plot = ROCPlot(self.cfg)
        self.pt_fakerate_target_efficiencies_plot = TargetEfficiencyFakeRateStackPlot(
            self.cfg, "pt", self.target_efficiencies
        )
        self.wp_values = {}

    def combine_results(self, evaluators: list):
        for evaluator in evaluators:
            self.tagging_plots[evaluator.algorithm] = ChargeClassifierPlot()
            self.tagging_plots[evaluator.algorithm].add_line(evaluator, "test")
            self.roc_plot.add_line(evaluator)
            self.pt_fakerate_target_efficiencies_plot.add_line(evaluator)
            for metric in self.metrics:
                self.efficiency_plots[metric].add_line(evaluator)
                self.fakerate_plots[metric].add_line(evaluator)
                self.efficiency_asymmetry_plots[metric].add_line(evaluator)
                self.fakerate_asymmetry_plots[metric].add_line(evaluator)
        self.roc_plot.finalize_legend()
        self.pt_fakerate_target_efficiencies_plot.finalize_legend()

    def save(self):
        for metric in self.metrics:
            fr_output_path = os.path.join(self.output_dir, f"{metric}_fakerates.pdf")
            self.fakerate_plots[metric].save(fr_output_path)
            fr_asym_output_path = os.path.join(
                self.output_dir, f"{metric}_fakerates_asymmetry.pdf"
            )
            self.fakerate_asymmetry_plots[metric].save(fr_asym_output_path)
            eff_output_path = os.path.join(
                self.output_dir, f"{metric}_efficiencies.pdf"
            )
            self.efficiency_plots[metric].save(eff_output_path)
            eff_asym_output_path = os.path.join(
                self.output_dir, f"{metric}_efficiencies_asymmetry.pdf"
            )
            self.efficiency_asymmetry_plots[metric].save(eff_asym_output_path)
        roc_output_path = os.path.join(self.output_dir, f"ROC.pdf")
        self.roc_plot.save(roc_output_path)
        target_eff_fr_output_path = os.path.join(
            self.output_dir, "pt_fakerates_target_efficiencies.pdf"
        )
        self.pt_fakerate_target_efficiencies_plot.save(target_eff_fr_output_path)
        for algorithm in self.tagging_plots.keys():
            cls_output_path = os.path.join(
                self.output_dir, f"classifier_scores_{algorithm}.pdf"
            )
            self.tagging_plots[algorithm].save(cls_output_path)
