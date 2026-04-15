import os
import numpy as np
import awkward as ak
import mplhep as hep
import matplotlib.pyplot as plt
from matplotlib import ticker

from omegaconf import DictConfig
from mltau.tools import general as g
from mltau.tools.evaluation.histogram import Histogram


plt.rcParams["mathtext.fontset"] = "stix"


def jet_charge_qkappa(cand_charges, cand_pts, jet_pts, kappa=0.5):
    """Calculate jet charge using Q*kappa weighting.

    Args:
        cand_charges: Candidate particle charges [batch, max_cands]
        cand_pts: Candidate particle pTs [batch, max_cands]
        jet_pts: Jet pTs [batch]
        kappa: Weighting exponent

    Returns:
        Jet charge values ranging from -1 to +1
    """
    numer = ak.sum(cand_charges * (cand_pts**kappa), axis=1)
    denom = jet_pts**kappa
    denom = ak.where(denom == 0, 1.0, denom)
    return ak.to_numpy(numer / denom)


class ChargeIdEvaluator:
    """Class for evaluating charge ID performance."""

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
        baseline_charges: np.array = None,
    ):
        self.output_dir = output_dir
        if self.output_dir != "":
            os.makedirs(self.output_dir, exist_ok=True)
        self.sample = sample
        self.algorithm = algorithm
        self.predicted = predicted
        self.gen_jet_tau_p4s = g.reinitialize_p4(gen_jet_tau_p4s)
        self.reco_jet_p4s = g.reinitialize_p4(reco_jet_p4s)
        self.cfg = cfg
        self.truth = truth
        # Use quantile-based thresholds: dense where scores concentrate (near 0/1),
        # sparse in the middle. Capped at 1000 points for performance.
        self.tagging_cuts = np.unique(
            np.concatenate(
                [[0], np.quantile(self.predicted, np.linspace(0, 1, 1000)), [1]]
            )
        )

        # Define charge masks early - needed by baseline calculations
        self.true_positive_charge_mask = self.truth == 1
        self.true_negative_charge_mask = self.truth == 0

        # Handle baseline charges if provided (range -1 to +1)
        self.baseline_charges = baseline_charges
        if self.baseline_charges is not None:
            # Convert baseline charges from [-1, +1] to [0, 1] for compatibility
            self.baseline_charges_normalized = (self.baseline_charges + 1) / 2
            # Create thresholds for baseline charges
            self.baseline_cuts = np.unique(
                np.concatenate(
                    [
                        [0],
                        np.quantile(
                            self.baseline_charges_normalized, np.linspace(0, 1, 1000)
                        ),
                        [1],
                    ]
                )
            )
            # Calculate baseline efficiency and fake rates
            self.baseline_efficiencies, self.baseline_eff_masks = (
                self._calculate_baseline_eff_fake(eff_fake="eff")
            )
            self.baseline_fakerates, self.baseline_fake_masks = (
                self._calculate_baseline_eff_fake(eff_fake="fake")
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
        self.wp_metrics = {}
        for name, metric in cfg.metrics.charge.metrics.items():
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

    def _calculate_eff_fake(self, eff_fake: str = "eff"):
        _eff_fake = {"positive": [], "negative": []}
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
        ref_var_pt_mask = self.gen_jet_tau_p4s.pt > self.cfg.metrics.tagging.cuts.min_pt
        ref_var_theta_mask1 = (
            abs(np.rad2deg(self.gen_jet_tau_p4s.theta))
            < self.cfg.metrics.tagging.cuts.max_theta
        )
        ref_var_theta_mask2 = (
            abs(np.rad2deg(self.gen_jet_tau_p4s.theta))
            > self.cfg.metrics.tagging.cuts.min_theta
        )
        gen_denominator_mask = (
            ref_var_pt_mask * ref_var_theta_mask1 * ref_var_theta_mask2
        )
        # As we are only assigning charge for tau (candidates) that are tagged, then to have the correct total
        # number of jets, we need to add the cuts on the reco jet also to the denominator
        tau_pt_mask = self.reco_jet_p4s.pt > self.cfg.metrics.tagging.cuts.min_pt
        tau_theta_mask1 = (
            abs(np.rad2deg(self.reco_jet_p4s.theta))
            < self.cfg.metrics.tagging.cuts.max_theta
        )
        tau_theta_mask2 = (
            abs(np.rad2deg(self.reco_jet_p4s.theta))
            > self.cfg.metrics.tagging.cuts.min_theta
        )
        reco_denominator_mask = tau_pt_mask * tau_theta_mask1 * tau_theta_mask2
        base_denominator_mask = gen_denominator_mask * reco_denominator_mask

        pos_denominator_mask = base_denominator_mask * positive_mask
        neg_denominator_mask = base_denominator_mask * negative_mask
        denominator_masks = {
            "positive": pos_denominator_mask,
            "negative": neg_denominator_mask,
        }
        neg_all = np.sum(neg_denominator_mask)
        pos_all = np.sum(pos_denominator_mask)
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

            if pos_all > 0:
                _eff_fake["positive"].append(pos_passing_cut / pos_all)
            else:
                _eff_fake["positive"].append(0.0)

            if neg_all > 0:
                _eff_fake["negative"].append(neg_passing_cut / neg_all)
            else:
                _eff_fake["negative"].append(0.0)
        return _eff_fake, denominator_masks

    def _calculate_baseline_eff_fake(self, eff_fake: str = "eff"):
        """Calculate efficiency/fake rates for baseline charge method."""
        _eff_fake = {"positive": [], "negative": []}
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
        ref_var_pt_mask = self.gen_jet_tau_p4s.pt > self.cfg.metrics.tagging.cuts.min_pt
        ref_var_theta_mask1 = (
            abs(np.rad2deg(self.gen_jet_tau_p4s.theta))
            < self.cfg.metrics.tagging.cuts.max_theta
        )
        ref_var_theta_mask2 = (
            abs(np.rad2deg(self.gen_jet_tau_p4s.theta))
            > self.cfg.metrics.tagging.cuts.min_theta
        )
        gen_denominator_mask = (
            ref_var_pt_mask * ref_var_theta_mask1 * ref_var_theta_mask2
        )
        # As we are only assigning charge for tau (candidates) that are tagged, then to have the correct total
        # number of jets, we need to add the cuts on the reco jet also to the denominator
        tau_pt_mask = self.reco_jet_p4s.pt > self.cfg.metrics.tagging.cuts.min_pt
        tau_theta_mask1 = (
            abs(np.rad2deg(self.reco_jet_p4s.theta))
            < self.cfg.metrics.tagging.cuts.max_theta
        )
        tau_theta_mask2 = (
            abs(np.rad2deg(self.reco_jet_p4s.theta))
            > self.cfg.metrics.tagging.cuts.min_theta
        )
        reco_denominator_mask = tau_pt_mask * tau_theta_mask1 * tau_theta_mask2
        base_denominator_mask = gen_denominator_mask * reco_denominator_mask

        pos_denominator_mask = base_denominator_mask * positive_mask
        neg_denominator_mask = base_denominator_mask * negative_mask
        denominator_masks = {
            "positive": pos_denominator_mask,
            "negative": neg_denominator_mask,
        }
        neg_all = np.sum(neg_denominator_mask)
        pos_all = np.sum(pos_denominator_mask)
        for cut in self.baseline_cuts:  # Use baseline_cuts for baseline method
            # Use baseline charges instead of ML model predictions
            pos_passing_cut = np.sum(
                self.baseline_charges_normalized[pos_denominator_mask] >= cut
            )
            # Alternative 1: Use same threshold for both (current method)
            neg_passing_cut = np.sum(
                (1 - self.baseline_charges_normalized[neg_denominator_mask]) >= cut
            )

            # Alternative 2: Use direct thresholding (uncomment to test)
            # neg_passing_cut = np.sum(
            #     self.baseline_charges_normalized[neg_denominator_mask] <= (1 - cut)
            # )

            # Add zero-division protection
            if pos_all > 0:
                _eff_fake["positive"].append(pos_passing_cut / pos_all)
            else:
                _eff_fake["positive"].append(0.0)

            if neg_all > 0:
                _eff_fake["negative"].append(neg_passing_cut / neg_all)
            else:
                _eff_fake["negative"].append(0.0)
        return _eff_fake, denominator_masks

    ###################################
    ###################################
    ###################################
    ###################################

    def _find_95_percent_average_efficiency_working_point(
        self, target_avg_efficiency: float = 0.95
    ) -> int:
        """Return the index into tagging_cuts where average efficiency is closest to target.

        Finds the working point where (positive_efficiency + negative_efficiency) / 2 = target_avg_efficiency
        """
        eff_pos = np.array(self.efficiencies["positive"])
        eff_neg = np.array(self.efficiencies["negative"])

        # Calculate average efficiency for each threshold
        avg_efficiencies = (eff_pos + eff_neg) / 2.0

        # Find the threshold closest to target average efficiency
        return int(np.argmin(np.abs(avg_efficiencies - target_avg_efficiency)))

    def _calculate_confusion_matrix(self):
        """Calculate confusion matrix at the working point threshold.

        Returns:
            dict: Confusion matrix with keys 'TP', 'TN', 'FP', 'FN'
        """
        # Use positive threshold for predictions (negative threshold is 1 - positive)
        positive_predictions = self.predicted >= self.wp_pos

        # Calculate confusion matrix elements
        # True Positive: predicted positive (1), actual positive (1)
        tp = np.sum(positive_predictions & (self.truth == 1))
        # True Negative: predicted negative (0), actual negative (0)
        tn = np.sum(~positive_predictions & (self.truth == 0))
        # False Positive: predicted positive (1), actual negative (0)
        fp = np.sum(positive_predictions & (self.truth == 0))
        # False Negative: predicted negative (0), actual positive (1)
        fn = np.sum(~positive_predictions & (self.truth == 1))

        return {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}

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
            if charge == "positive":
                wp_mask = self.predicted >= self.wp_pos
            else:
                wp_mask = self.predicted <= self.wp_neg
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


class ChargeClassifierPlot:
    def __init__(self):
        self.bin_edges = np.linspace(start=0, stop=1, num=21)
        self.fig, self.ax = self.plot()

    def add_line(self, evaluator, dataset: str):
        linestyle = "solid" if dataset == "test" else "dashed"
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
            histtype="step",
            label=r"$\tau^{+}$",
            ls=linestyle,
            color="red",
            ax=self.ax,
        )
        hep.histplot(
            neg_histogram,
            bins=self.bin_edges,
            histtype="step",
            label=r"$\tau^{-}$",
            ls=linestyle,
            color="blue",
            ax=self.ax,
        )
        self.ax.legend(prop={"size": 28})

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel(r"$\mathcal{D}$", fontdict={"size": 28})
        ax.set_yscale("log")
        ax.set_ylabel("Relative yield / bin")
        return fig, ax

    def save(self, output_path: str):
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class ROCPlot:
    def __init__(self, cfg):
        self.fig, self.ax = self.plot()
        self.cfg = cfg

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_ylabel(r"$P_{misid}$", fontsize=30)
        ax.set_xlabel(r"$\varepsilon_{\tau}$", fontsize=30)
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=30)
        ax.set_ylim((1e-5, 1))
        ax.set_xlim((0, 1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.set_yscale("log")
        plt.grid()
        return fig, ax

    def add_line(self, evaluator):
        # ML prediction curves
        self.ax.plot(
            evaluator.efficiencies["positive"],
            evaluator.fakerates["positive"],
            color="r",
            marker="s",
            label=r"$\tau^{+}$ (ML)",
            ms=8,
            ls="",
        )
        self.ax.plot(
            evaluator.efficiencies["negative"],
            evaluator.fakerates["negative"],
            color="b",
            marker="^",
            label=r"$\tau^{-}$ (ML)",
            ms=8,
            ls="",
        )

        # Baseline charge curves (if available)
        if (
            hasattr(evaluator, "baseline_efficiencies")
            and evaluator.baseline_efficiencies is not None
        ):
            self.ax.plot(
                evaluator.baseline_efficiencies["positive"],
                evaluator.baseline_fakerates["positive"],
                color="r",
                marker="o",
                label=r"$\tau^{+}$ (Baseline)",
                ms=6,
                ls="--",
                alpha=0.7,
            )
            self.ax.plot(
                evaluator.baseline_efficiencies["negative"],
                evaluator.baseline_fakerates["negative"],
                color="b",
                marker="v",
                label=r"$\tau^{-}$ (Baseline)",
                ms=6,
                ls="--",
                alpha=0.7,
            )
        # Mark the 95% average efficiency working point on both curves
        idx = evaluator.wp_idx
        for eff_key, fake_key in [
            ("positive", "positive"),
            ("negative", "negative"),
        ]:
            self.ax.plot(
                evaluator.efficiencies[eff_key][idx],
                evaluator.fakerates[fake_key][idx],
                marker="*",
                color="k",
                ms=25,
                zorder=5,
                linestyle="",
            )
        self.ax.legend(prop={"size": 30})

    def save(self, output_path):
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class EfficiencyPlot:
    def __init__(self, cfg: DictConfig, metric: str):
        self.cfg = cfg
        self.metric = metric
        self.fig, self.ax = self.plot()

    def add_line(self, evaluator):
        self.ax.errorbar(
            evaluator.wp_metrics[self.metric]["negative"]["eff_bin_centers"],
            evaluator.wp_metrics[self.metric]["negative"]["efficiencies"],
            xerr=evaluator.wp_metrics[self.metric]["negative"]["eff_xerr"],
            yerr=evaluator.wp_metrics[self.metric]["negative"]["eff_yerr"],
            ms=20,
            color="b",
            marker="^",
            linestyle="",
            label=r"$\tau^{-}$",
        )
        self.ax.errorbar(
            evaluator.wp_metrics[self.metric]["positive"]["eff_bin_centers"],
            evaluator.wp_metrics[self.metric]["positive"]["efficiencies"],
            xerr=evaluator.wp_metrics[self.metric]["positive"]["eff_xerr"],
            yerr=evaluator.wp_metrics[self.metric]["positive"]["eff_yerr"],
            ms=20,
            color="r",
            marker="s",
            linestyle="",
            label=r"$\tau^{+}$",
        )
        self.ax.legend(loc="upper right", prop={"size": 30})

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(
                self.cfg.metrics.tagging.metrics[self.metric].x_maj_tick_spacing
            )
        )
        ax.set_xlabel(
            rf"{self.cfg.metrics.tagging.performances.efficiency.xlabel[self.metric]}",
            fontsize=30,
        )
        ax.set_ylabel(
            rf"{self.cfg.metrics.tagging.performances.efficiency.ylabel}",
            fontsize=30,
        )
        ax.set_yscale(self.cfg.metrics.tagging.performances.efficiency.yscale)
        if self.cfg.metrics.tagging.performances.efficiency.ylim is not None:
            ylim = tuple(self.cfg.metrics.tagging.performances.efficiency.ylim)
        else:
            ylim = self.cfg.metrics.tagging.performances.efficiency.ylim
        ax.set_ylim(tuple(ylim))
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=30)
        plt.grid()
        return fig, ax

    def save(self, output_path):
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class FakeRatePlot:
    def __init__(self, cfg: DictConfig, metric: str):
        self.cfg = cfg
        self.metric = metric
        self.fig, self.ax = self.plot()

    def add_line(self, evaluator):
        self.ax.errorbar(
            evaluator.wp_metrics[self.metric]["negative"]["fr_bin_centers"],
            evaluator.wp_metrics[self.metric]["negative"]["fakerates"],
            xerr=evaluator.wp_metrics[self.metric]["negative"]["fr_xerr"],
            yerr=evaluator.wp_metrics[self.metric]["negative"]["fr_yerr"],
            ms=20,
            color="b",
            marker="^",
            linestyle="",
            label=r"$\tau^{-}$",
        )
        self.ax.errorbar(
            evaluator.wp_metrics[self.metric]["positive"]["fr_bin_centers"],
            evaluator.wp_metrics[self.metric]["positive"]["fakerates"],
            xerr=evaluator.wp_metrics[self.metric]["positive"]["fr_xerr"],
            yerr=evaluator.wp_metrics[self.metric]["positive"]["fr_yerr"],
            ms=20,
            color="r",
            marker="s",
            linestyle="",
            label=r"$\tau^{+}$",
        )
        self.ax.legend(loc="upper right", prop={"size": 30})

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(
                self.cfg.metrics.tagging.metrics[self.metric].x_maj_tick_spacing
            )
        )
        ax.set_xlabel(
            rf"{self.cfg.metrics.tagging.performances.fakerate.xlabel[self.metric]}",
            fontsize=30,
        )
        ax.set_ylabel(
            rf"{self.cfg.metrics.tagging.performances.fakerate.ylabel}", fontsize=30
        )
        ax.set_yscale(self.cfg.metrics.tagging.performances.fakerate.yscale)
        if self.cfg.metrics.tagging.performances.fakerate.ylim is not None:
            ylim = tuple(self.cfg.metrics.tagging.performances.fakerate.ylim)
        else:
            ylim = self.cfg.metrics.tagging.performances.fakerate.ylim
        ax.set_ylim(ylim)
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=30)
        plt.grid()
        return fig, ax

    def save(self, output_path):
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
        self.fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(self.fig)
