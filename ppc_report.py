"""Posterior predictive checks and validation report for the Kd Bayesian fits.

Runs five complementary internal-validation analyses against the already-
computed outputs in okl_concentration_pair_datasets/, without re-fitting
anything:

1. Posterior predictive checks (PPC): for each compound/target pair, forward-
   simulate the fitted Hill-equation model (using stored posterior draws from
   all_eligible_posteriors.zarr) at the pair's actually observed doses, and
   compute a Bayesian p-value / standardized residual per dose point. Results
   are stratified by potency bin to surface any subclass of interaction that
   the model captures poorly, and additionally broken down by potency bin x
   tested dose, to check whether miscalibration is uniform across the dose-
   response curve or concentrated at specific concentrations.

2. Credible-interval / extrapolation summary: using the already-computed
   kd_log HDI columns in all_eligible_concentrations.csv.gz, reports CI width
   separately for pairs whose Kd point estimate falls inside vs. outside the
   measured dose range (interpolation vs. extrapolation).

3. Leave-doses-out stability check: compares each 2-point concentration-pair
   fit's Kd posterior against the corresponding 4-point ("all_eligible") fit
   for the same compound/target pair, as a robustness/sensitivity check on
   which doses were used.

4. Prior sensitivity: compares the default-prior all_eligible fit against the
   wide-prior subsample fit in prior_sensitivity_wide_kd_prior.csv.gz.

5. Prior predictive checks: same PPC machinery as (1), but driven by i.i.d.
   draws from the model's prior (no fitting, no stored posterior) via
   PriorDraws, to check whether the prior itself implies a plausible spread
   of dose-response behavior before any data is seen.

Outputs CSV summary tables plus publication-ready vector PDF and 300-dpi PNG
figures into ppc_report_output/.
"""

import argparse
import itertools
import pathlib

import numpy as np
import pandas as pd
import zarr
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PAIR_DATASET_PATH = pathlib.Path("okl_concentration_pair_datasets")
OUTPUT_PATH = pathlib.Path("ppc_report_output")

ELIGIBLE_CONCENTRATIONS = [12.5, 100, 1000, 10000]
POSSIBLE_PAIRS = list(itertools.combinations(ELIGIBLE_CONCENTRATIONS, 2))

# log-Kd (molar) bin edges for stratification, from < 1 nM to > 10 uM.
POTENCY_BIN_EDGES_LOG_M = np.log(
    [1e-12, 1e-9, 1e-8, 1e-7, 1e-5, 1e-3]
)
POTENCY_BIN_EDGES_LOG_M[[0, -1]] = [-np.inf, np.inf]
POTENCY_BIN_LABELS = [
    "<1 nM",
    "1–10 nM",
    "10–100 nM",
    "100 nM–10 µM",
    ">10 µM",
]

# Color-blind-safe Okabe–Ito palette, following the visual language used in
# the Andrew Heiss reference supplied with the analysis request.
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
INK = "#262626"
MID_GREY = "#777777"
GRID_GREY = "#E6E6E6"
TAIL_GREY = "#F1F1F1"


def set_publication_style():
    """Set a compact, journal-friendly Matplotlib style.

    PDF/PS font type 42 keeps text as embedded TrueType glyphs rather than
    Type 3 bitmap-like paths, which is accepted by most journal workflows.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.labelweight": "bold",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.titlesize": 10,
            "figure.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def style_axis(ax, grid_axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID_GREY, linewidth=0.6)
    ax.tick_params(length=3, width=0.6)


def facet_title(ax, text, fontsize=None):
    """Add an unboxed facet label above an axes."""
    ax.set_title(
        text,
        loc="left",
        pad=5,
        fontweight="bold",
        fontsize=fontsize,
    )


def figure_title(fig, title, subtitle=None):
    """Intentionally omit figure titles; captions carry this information."""
    return None


def save_figure(fig, out_path):
    """Write a vector PDF and a high-resolution PNG preview."""
    stem = pathlib.Path(out_path).with_suffix("")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


set_publication_style()


def _error_model_np(mu):
    """Plain-numpy equivalent of fit_kd_cli._error_model."""
    return np.maximum(0.15 * mu - 0.043, 1)


def _model_predict_np(kd, hill_slope, doses):
    """Plain-numpy equivalent of fit_kd_cli._model_predict.

    kd, hill_slope: (n_pairs, n_draws) arrays (molar units for kd)
    doses: (n_pairs, n_doses) array (molar units), broadcast per pair
    Returns mu with shape (n_pairs, n_draws, n_doses).
    """
    kd = kd[:, :, None]
    hill_slope = hill_slope[:, :, None]
    doses = doses[:, None, :]
    return 100 - 100 / (1 + (kd / doses) ** hill_slope)


def load_observed_data():
    """Load raw single-dose CSV, restricted to the OKL library / replaced-repeat
    dataset used by fit_kd_dose_pairs.py, grouped by compound/target pair."""
    df = pd.read_csv("okl_single_dose_datasets.csv.gz")
    df = df[(df["dataset"] == "original_repeat_replaced") & (df["library"] == "OKL")]
    return df


def filter_eligible_pairs(observed_df):
    """Restrict to pairs tested at all 4 ELIGIBLE_CONCENTRATIONS, at exactly
    those concentrations - the same "all_eligible" subset fit_kd_dose_pairs.py
    fits, matching what all_eligible_posteriors.zarr covers. Needed for draw
    sources like PriorDraws that don't naturally restrict to this subset the
    way a zarr lookup keyed on those pairs does.
    """
    eligible = set(ELIGIBLE_CONCENTRATIONS)
    df = observed_df[observed_df["Compound Concentration (nM)"].isin(eligible)]
    group_keys = ["hmsl_id", "DiscoveRx Gene Symbol"]
    complete = df.groupby(group_keys)["Compound Concentration (nM)"].apply(
        lambda s: eligible.issubset(s)
    )
    keep = complete[complete].index
    return df.set_index(group_keys).loc[keep].reset_index()


def potency_bin(kd_log_molar):
    idx = np.digitize(kd_log_molar, POTENCY_BIN_EDGES_LOG_M[1:-1])
    return np.array(POTENCY_BIN_LABELS)[idx]


# ---------------------------------------------------------------------------
# 1. Posterior / prior predictive checks
# ---------------------------------------------------------------------------


class ZarrDraws:
    """Draw source backed by a zarr group of per-pair posterior draws (as
    written by fit_kd_dose_pairs.py's extract_posterior_matrices/zarr export).

    Each real (compound_id, target) pair has its own row of stored draws.
    """

    def __init__(self, zarr_group_path, group_name):
        root = zarr.open_group(str(zarr_group_path), mode="r")
        grp = root[group_name]
        compound_ids = list(grp.attrs["compound_ids"])
        targets = list(grp.attrs["targets"])
        self._kd_log_draws = grp["kd"][:]  # (n_pairs, n_draws), log-Kd (molar)
        self._hill_slope_log_draws = grp["hill_slope"][:]  # (n_pairs, n_draws), log-Hill-slope
        self._row_by_key = {
            (cid, tgt): i for i, (cid, tgt) in enumerate(zip(compound_ids, targets))
        }

    def keys(self):
        return self._row_by_key.keys()

    def get(self, compound_id, target):
        i = self._row_by_key.get((compound_id, target))
        if i is None:
            return None
        return self._kd_log_draws[i], self._hill_slope_log_draws[i]


class PriorDraws:
    """Draw source that ignores compound_id/target entirely and returns i.i.d.
    samples from the model's prior (fit_kd_cli._create_model's kd_log/
    hill_slope_log priors) for every pair. Since the prior doesn't depend on
    the pair, one shared pool of draws is sampled once and reused for all
    pairs (this is a prior predictive check, not a per-pair posterior).
    """

    def __init__(self, n_draws=8000, kd_log_sigma=3, hill_slope_log_sigma=0.5, seed=0):
        rng = np.random.default_rng(seed)
        self._kd_log_draws = rng.normal(np.log(1e-6), kd_log_sigma, size=n_draws)
        self._hill_slope_log_draws = rng.normal(0, hill_slope_log_sigma, size=n_draws)

    def get(self, compound_id, target):
        return self.sample()

    def sample(self):
        """Return the shared (kd_log_draws, hill_slope_log_draws) pool directly, without needing a (compound_id, target) key."""
        return self._kd_log_draws, self._hill_slope_log_draws


def _ppc_for_pair(kd, hill_slope, doses_molar, responses):
    """Per-dose PPC diagnostics for one pair, given already-exponentiated
    kd/hill_slope draws (shape (n_draws,)) and that pair's observed doses/
    responses. Returns arrays (p_values, std_resid, mean_mu, mean_sigma).
    """
    mu = _model_predict_np(kd[None, :], hill_slope[None, :], doses_molar[None, :])[0]
    sigma = _error_model_np(mu)

    # Bayesian predictive p-value per dose, averaged over draws:
    # P(replicated response >= observed | draws), analytically via Normal CDF.
    p_values = stats.norm.sf(responses[:, None], loc=mu.T, scale=sigma.T).mean(axis=1)

    mean_mu = mu.mean(axis=0)
    mean_sigma = sigma.mean(axis=0)
    std_resid = (responses - mean_mu) / mean_sigma
    return p_values, std_resid, mean_mu, mean_sigma


def compute_ppc(observed_df, draw_source):
    """Compute per-pair, per-dose PPC diagnostics for every real
    (compound_id, target) pair in observed_df, using kd_log/hill_slope_log
    draws supplied by draw_source.get(compound_id, target).

    draw_source may be a ZarrDraws (one draw set per real pair, i.e. a
    posterior predictive check) or a PriorDraws (a shared draw set reused for
    every pair, i.e. a prior predictive check) - compute_ppc doesn't care
    which, it only needs .get(compound_id, target) -> (kd_log_draws,
    hill_slope_log_draws) | None.

    Returns a DataFrame with one row per (compound_id, target, dose) with
    columns: ppc_p_value (Bayesian predictive p-value), std_resid
    (standardized residual using mean mu/sigma across draws).
    """
    obs_grouped = observed_df.groupby(["hmsl_id", "DiscoveRx Gene Symbol"])

    records = []
    for key, pair_obs in obs_grouped:
        draws = draw_source.get(*key)
        if draws is None:
            continue
        kd_log_draws, hill_slope_log_draws = draws
        doses_molar = pair_obs["Compound Concentration (nM)"].to_numpy() * 1e-9
        responses = pair_obs["Percent Control"].to_numpy()

        kd = np.exp(kd_log_draws).astype(np.float64)
        hill_slope = np.exp(hill_slope_log_draws).astype(np.float64)

        p_values, std_resid, _, _ = _ppc_for_pair(kd, hill_slope, doses_molar, responses)

        kd_log_mean_molar = kd_log_draws.mean()
        for dose, resp, p, sr in zip(doses_molar, responses, p_values, std_resid):
            records.append(
                {
                    "compound_id": key[0],
                    "target": key[1],
                    "dose_molar": dose,
                    "observed_response": resp,
                    "ppc_p_value": p,
                    "std_resid": sr,
                    "kd_log_mean_molar": kd_log_mean_molar,
                }
            )

    return pd.DataFrame.from_records(records)


def summarize_ppc_by_potency(ppc_df):
    ppc_df = ppc_df.copy()
    ppc_df["extreme_tail"] = (ppc_df["ppc_p_value"] < 0.025) | (
        ppc_df["ppc_p_value"] > 0.975
    )
    per_pair = (
        ppc_df.groupby(["compound_id", "target"])
        .agg(
            kd_log_mean_molar=("kd_log_mean_molar", "first"),
            mean_abs_std_resid=("std_resid", lambda s: s.abs().mean()),
            max_abs_std_resid=("std_resid", lambda s: s.abs().max()),
            min_ppc_p_value=("ppc_p_value", "min"),
            max_ppc_p_value=("ppc_p_value", "max"),
            flagged_extreme_tail=("extreme_tail", "any"),
        )
        .reset_index()
    )
    per_pair["potency_bin"] = potency_bin(per_pair["kd_log_mean_molar"].to_numpy())
    # Retain the original standardized-residual flag for continuity. The
    # predictive-tail flag above is the primary PPC diagnostic because it
    # integrates over uncertainty in Kd and Hill slope.
    per_pair["flagged_outlier"] = per_pair["max_abs_std_resid"] > 3

    summary = (
        per_pair.groupby("potency_bin")
        .agg(
            n_pairs=("compound_id", "size"),
            frac_extreme_predictive_tail=("flagged_extreme_tail", "mean"),
            frac_flagged=("flagged_outlier", "mean"),
            median_mean_abs_std_resid=("mean_abs_std_resid", "median"),
        )
        .reindex(POTENCY_BIN_LABELS)
    )
    return per_pair, summary


def plot_ppc_calibration(ppc_df, out_path):
    """Histogram of per-dose-point PPC p-values, pooled across all dose points
    in each potency bin.

    Each panel is normalized to a density (area = 1) rather than raw counts.
    Posterior predictive tail probabilities need not be uniform because the
    observations are used both to fit and to check the model; concentration
    near 0.5 is expected. Shading marks the pre-specified extreme tails.
    """
    ppc_df = ppc_df.copy()
    ppc_df["potency_bin"] = potency_bin(ppc_df["kd_log_mean_molar"].to_numpy())
    fig, axes = plt.subplots(
        1, len(POTENCY_BIN_LABELS), figsize=(6.0, 2.05), sharex=True, sharey=True
    )
    for ax, label in zip(axes, POTENCY_BIN_LABELS):
        sub = ppc_df[ppc_df["potency_bin"] == label]
        n_pairs = sub[["compound_id", "target"]].drop_duplicates().shape[0]
        ax.axvspan(0, 0.025, color=TAIL_GREY, zorder=0)
        ax.axvspan(0.975, 1, color=TAIL_GREY, zorder=0)
        ax.hist(
            sub["ppc_p_value"], bins=20, range=(0, 1), color=BLUE,
            alpha=0.88, density=True, edgecolor="white", linewidth=0.25,
        )
        ax.axvline(0.5, color=INK, linestyle=(0, (2, 2)), linewidth=0.8)
        facet_title(ax, f"{label}\nn = {n_pairs:,}", fontsize=7)
        style_axis(ax)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Density")
    fig.supxlabel("Posterior predictive tail probability, P(yrep ≥ yobs | y)", y=0.01)
    figure_title(
        fig,
        "Posterior predictive checks by inferred potency",
        "Each observation is compared with its fitted posterior predictive distribution; grey bands mark p < 0.025 or p > 0.975.",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1], w_pad=0.6)
    save_figure(fig, out_path)


def plot_ppc_calibration_by_dose(ppc_df, out_path):
    """Grid of PPC p-value distributions broken down by potency bin (rows)
    AND by the actual tested dose (columns), to check whether miscalibration
    is uniform across the dose-response curve or concentrated at specific
    concentrations (e.g. the lowest/highest dose, where the Hill curve is
    flattest and the error model's sigma floor dominates).

    Each panel is normalized to a density. Grey bands mark the pre-specified
    extreme tails; a dotted line marks 0.5.
    """
    ppc_df = ppc_df.copy()
    ppc_df["potency_bin"] = potency_bin(ppc_df["kd_log_mean_molar"].to_numpy())
    doses_nm = np.sort(ppc_df["dose_molar"].unique())

    fig, axes = plt.subplots(
        len(POTENCY_BIN_LABELS),
        len(doses_nm),
        figsize=(6.0, 6.5),
        sharex=True,
    )
    for row, potency_label in enumerate(POTENCY_BIN_LABELS):
        for col, dose in enumerate(doses_nm):
            ax = axes[row, col]
            sub = ppc_df[(ppc_df["potency_bin"] == potency_label) & (ppc_df["dose_molar"] == dose)]
            n_pairs = sub[["compound_id", "target"]].drop_duplicates().shape[0]
            ax.axvspan(0, 0.025, color=TAIL_GREY, zorder=0)
            ax.axvspan(0.975, 1, color=TAIL_GREY, zorder=0)
            if n_pairs > 0:
                ax.hist(
                    sub["ppc_p_value"], bins=20, range=(0, 1), color=BLUE,
                    alpha=0.88, density=True, edgecolor="white", linewidth=0.2,
                )
            ax.axvline(0.5, color=INK, linestyle=(0, (2, 2)), linewidth=0.7)
            style_axis(ax)
            if row == 0:
                facet_title(ax, f"{dose * 1e9:g} nM", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{potency_label}\n(n = {n_pairs:,})", fontweight="bold")
            ax.set_xlim(0, 1)
    fig.supxlabel("Posterior predictive tail probability, P(yrep ≥ yobs | y)", y=0.01)
    fig.supylabel("Density", x=0.005)
    figure_title(
        fig,
        "Posterior predictive checks by potency and tested dose",
        "Rows stratify fitted Kd; columns stratify the four assayed concentrations.",
    )
    fig.tight_layout(rect=[0.035, 0.04, 1, 1], h_pad=0.7, w_pad=0.6)
    save_figure(fig, out_path)


def plot_ppc_calibration_by_dose_only(ppc_df, out_path):
    """PPC p-value distribution broken down by tested dose only, pooling
    across all potency bins. Complements plot_ppc_calibration (potency only)
    and plot_ppc_calibration_by_dose (potency x dose): this isolates whether
    a given dose is miscalibrated on average, independent of potency bin.
    """
    doses_nm = np.sort(ppc_df["dose_molar"].unique())
    fig, axes = plt.subplots(1, len(doses_nm), figsize=(6.0, 2.05), sharey=True)
    for ax, dose in zip(axes, doses_nm):
        sub = ppc_df[ppc_df["dose_molar"] == dose]
        n_pairs = sub[["compound_id", "target"]].drop_duplicates().shape[0]
        ax.axvspan(0, 0.025, color=TAIL_GREY, zorder=0)
        ax.axvspan(0.975, 1, color=TAIL_GREY, zorder=0)
        ax.hist(
            sub["ppc_p_value"], bins=20, range=(0, 1), color=BLUE,
            alpha=0.88, density=True, edgecolor="white", linewidth=0.25,
        )
        ax.axvline(0.5, color=INK, linestyle=(0, (2, 2)), linewidth=0.8)
        facet_title(ax, f"{dose * 1e9:g} nM\nn = {n_pairs:,}", fontsize=7.5)
        style_axis(ax)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Density")
    fig.supxlabel("Posterior predictive tail probability, P(yrep ≥ yobs | y)", y=0.01)
    figure_title(
        fig,
        "Posterior predictive checks by tested dose",
        "Grey bands mark p < 0.025 or p > 0.975; the dotted line marks 0.5.",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1], w_pad=0.65)
    save_figure(fig, out_path)


def plot_ppc_extreme_rate_heatmap(ppc_df, out_path):
    """Compact map of where extreme posterior predictive tails occur."""
    df = ppc_df.copy()
    df["potency_bin"] = potency_bin(df["kd_log_mean_molar"].to_numpy())
    df["extreme_tail"] = (df["ppc_p_value"] < 0.025) | (df["ppc_p_value"] > 0.975)
    df["dose_nm"] = df["dose_molar"] * 1e9
    rates = (
        df.groupby(["potency_bin", "dose_nm"])["extreme_tail"]
        .mean()
        .unstack()
        .reindex(POTENCY_BIN_LABELS)
    )

    cmap = LinearSegmentedColormap.from_list("extreme_rate", ["#FFFFFF", ORANGE, VERMILLION])
    fig, ax = plt.subplots(figsize=(5.5, 2.65))
    image = ax.pcolormesh(
        np.arange(rates.shape[1] + 1),
        np.arange(rates.shape[0] + 1),
        rates.to_numpy(),
        cmap=cmap,
        vmin=0,
        vmax=max(0.25, rates.max().max()),
        shading="flat",
    )
    ax.set_xlim(0, rates.shape[1])
    ax.set_ylim(rates.shape[0], 0)
    ax.set_xticks(np.arange(len(rates.columns)) + 0.5, [f"{d:g}" for d in rates.columns])
    ax.set_yticks(np.arange(len(rates.index)) + 0.5, rates.index)
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            value = rates.iloc[i, j]
            color = "white" if value > 0.16 else INK
            ax.text(j + 0.5, i + 0.5, f"{value:.1%}", ha="center", va="center", color=color, fontsize=8)
    ax.set_xlabel("Tested dose (nM)")
    ax.set_ylabel("Inferred potency")
    figure_title(
        fig,
        "Location of posterior predictive discrepancies",
        "Cells report the fraction with p < 0.025 or p > 0.975; these are model checks, not multiplicity-adjusted tests.",
    )
    fig.tight_layout()
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Credible-interval / extrapolation summary
# ---------------------------------------------------------------------------


def summarize_ci_extrapolation(all_eligible_csv_path, observed_df):
    df = pd.read_csv(all_eligible_csv_path)
    df = df[df["status"] == "ok"].copy()

    dose_range = (
        observed_df.groupby(["hmsl_id", "DiscoveRx Gene Symbol"])["Compound Concentration (nM)"]
        .agg(min_dose_nm="min", max_dose_nm="max")
        .reset_index()
        .rename(columns={"hmsl_id": "compound_id", "DiscoveRx Gene Symbol": "target"})
    )
    df = df.merge(dose_range, on=["compound_id", "target"], how="left")

    # kd_mean (linear-scale, molar) is rounded to 3 decimals in the exported
    # CSV and underflows to 0 for essentially all sub-micromolar Kds, so use
    # kd_log_mean (log-scale, retains precision) for anything downstream.
    df["kd_nm"] = np.exp(df["kd_log_mean"]) * 1e9
    df["extrapolated"] = (df["kd_nm"] < df["min_dose_nm"]) | (df["kd_nm"] > df["max_dose_nm"])
    df["ci_width_log"] = df["kd_log_hdi_97%"] - df["kd_log_hdi_3%"]
    df["ci_span_fold"] = np.exp(df["ci_width_log"])
    df["hill_slope_ci_width_log"] = (
        df["hill_slope_log_hdi_97%"] - df["hill_slope_log_hdi_3%"]
    )
    df["hill_slope_ci_span_fold"] = np.exp(df["hill_slope_ci_width_log"])
    df["potency_bin"] = potency_bin(df["kd_log_mean"].to_numpy())

    summary = (
        df.groupby(["potency_bin", "extrapolated"])
        .agg(
            n_pairs=("compound_id", "size"),
            median_ci_width_log=("ci_width_log", "median"),
            median_ci_span_fold=("ci_span_fold", "median"),
            median_hill_slope_ci_span_fold=("hill_slope_ci_span_fold", "median"),
        )
        .reindex(pd.MultiIndex.from_product([POTENCY_BIN_LABELS, [False, True]], names=["potency_bin", "extrapolated"]))
    )
    return df, summary


def plot_ci_width_vs_potency(df, out_path):
    df = df.copy()
    df["potency_bin"] = potency_bin(df["kd_log_mean"].to_numpy())
    df["ci_span_fold"] = np.exp(df["ci_width_log"])
    fig, ax = plt.subplots(figsize=(4, 1.8))
    y_base = np.arange(len(POTENCY_BIN_LABELS))[::-1]
    specs = [(False, -0.12, BLUE, "Inside tested range"), (True, 0.12, VERMILLION, "Outside tested range")]
    for extrap, offset, color, label in specs:
        first = True
        for y, potency in zip(y_base, POTENCY_BIN_LABELS):
            values = df.loc[
                (df["potency_bin"] == potency) & (df["extrapolated"] == extrap),
                "ci_span_fold",
            ].dropna()
            if values.empty:
                continue
            q03, q25, q50, q75, q97 = values.quantile([0.03, 0.25, 0.5, 0.75, 0.97])
            yy = y + offset
            ax.plot([q03, q97], [yy, yy], color=color, lw=1.0, alpha=0.7)
            ax.plot([q25, q75], [yy, yy], color=color, lw=4.0, solid_capstyle="round")
            ax.scatter(q50, yy, s=25, color=color, edgecolor="white", linewidth=0.5,
                       zorder=3, label=label if first else None)
            first = False
    ax.axvline(1, color=INK, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.set_xscale("log")
    ax.set_yticks(y_base, POTENCY_BIN_LABELS)
    ax.set_xlabel("Multiplicative span of the 94% Kd HDI")
    ax.set_ylabel("Inferred potency")
    ax.legend(loc="upper left")
    style_axis(ax, grid_axis="x")
    figure_title(
        fig,
        "Kd uncertainty increases sharply beyond the assayed range",
        "Dots are medians; thick and thin intervals contain 50% and 94% of pair-specific HDI spans, respectively.",
    )
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_hill_slope_uncertainty(df, out_path):
    """Distribution of Hill-slope posterior uncertainty by potency class."""
    df = df.copy()
    df["potency_bin"] = potency_bin(df["kd_log_mean"].to_numpy())
    df["hill_slope_ci_span_fold"] = np.exp(
        df["hill_slope_log_hdi_97%"] - df["hill_slope_log_hdi_3%"]
    )
    fig, ax = plt.subplots(figsize=(5.9, 2.9))
    y_base = np.arange(len(POTENCY_BIN_LABELS))[::-1]
    specs = [(False, -0.12, BLUE, "Inside tested range"), (True, 0.12, VERMILLION, "Outside tested range")]
    for extrap, offset, color, label in specs:
        first = True
        for y, potency in zip(y_base, POTENCY_BIN_LABELS):
            values = df.loc[
                (df["potency_bin"] == potency) & (df["extrapolated"] == extrap),
                "hill_slope_ci_span_fold",
            ].dropna()
            if values.empty:
                continue
            q03, q25, q50, q75, q97 = values.quantile([0.03, 0.25, 0.5, 0.75, 0.97])
            yy = y + offset
            ax.plot([q03, q97], [yy, yy], color=color, lw=1.0, alpha=0.7)
            ax.plot([q25, q75], [yy, yy], color=color, lw=4.0, solid_capstyle="round")
            ax.scatter(q50, yy, s=25, color=color, edgecolor="white", linewidth=0.5,
                       zorder=3, label=label if first else None)
            first = False
    ax.axvline(1, color=INK, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.set_xscale("log")
    ax.set_yticks(y_base, POTENCY_BIN_LABELS)
    ax.set_xlabel("Multiplicative span of the 94% Hill-slope HDI")
    ax.set_ylabel("Inferred potency")
    ax.legend(loc="upper left")
    style_axis(ax, grid_axis="x")
    figure_title(
        fig,
        "Hill-slope uncertainty is greatest for out-of-range interactions",
        "Dots are medians; thick and thin intervals contain 50% and 94% of pair-specific HDI spans, respectively.",
    )
    fig.tight_layout()
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Leave-doses-out stability check (2-point vs. 4-point fits)
# ---------------------------------------------------------------------------


def summarize_leave_doses_out(all_eligible_csv_path):
    all_eligible = pd.read_csv(all_eligible_csv_path)
    all_eligible = all_eligible[all_eligible["status"] == "ok"]
    all_eligible = all_eligible[["compound_id", "target", "kd_log_mean", "kd_log_hdi_3%", "kd_log_hdi_97%"]]
    all_eligible = all_eligible.rename(
        columns={c: f"full_{c}" for c in all_eligible.columns if c not in ("compound_id", "target")}
    )
    all_eligible["potency_bin"] = potency_bin(all_eligible["full_kd_log_mean"].to_numpy())

    rows = []
    for conc_pair in POSSIBLE_PAIRS:
        pair_name = f"{conc_pair[0]}_{conc_pair[1]}"
        csv_path = PAIR_DATASET_PATH / f"{pair_name}.csv.gz"
        if not csv_path.exists():
            continue
        pair_df = pd.read_csv(csv_path)
        pair_df = pair_df[pair_df["status"] == "ok"]
        pair_df = pair_df[["compound_id", "target", "kd_log_mean", "kd_log_hdi_3%", "kd_log_hdi_97%"]]
        pair_df = pair_df.rename(
            columns={c: f"pair_{c}" for c in pair_df.columns if c not in ("compound_id", "target")}
        )
        merged = pair_df.merge(all_eligible, on=["compound_id", "target"], how="inner")
        merged["pair_name"] = pair_name
        merged["full_within_pair_ci"] = (
            merged["full_kd_log_mean"] >= merged["pair_kd_log_hdi_3%"]
        ) & (merged["full_kd_log_mean"] <= merged["pair_kd_log_hdi_97%"])
        rows.append(merged)

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if combined.empty:
        return combined, pd.DataFrame()

    summary = (
        combined.groupby(["pair_name", "potency_bin"])
        .agg(
            n_pairs=("compound_id", "size"),
            corr_kd_log=("pair_kd_log_mean", lambda s: s.corr(combined.loc[s.index, "full_kd_log_mean"])),
            frac_full_within_pair_ci=("full_within_pair_ci", "mean"),
        )
        .reset_index()
    )
    return combined, summary


# ---------------------------------------------------------------------------
# 4. Prior sensitivity
# ---------------------------------------------------------------------------


def summarize_prior_sensitivity(all_eligible_csv_path, wide_prior_csv_path):
    default = pd.read_csv(all_eligible_csv_path)
    default = default[default["status"] == "ok"][["compound_id", "target", "kd_log_mean"]]
    default = default.rename(columns={"kd_log_mean": "default_kd_log_mean"})

    if not wide_prior_csv_path.exists():
        return None, None

    wide = pd.read_csv(wide_prior_csv_path)
    wide = wide[wide["status"] == "ok"][["compound_id", "target", "kd_log_mean"]]
    wide = wide.rename(columns={"kd_log_mean": "wide_prior_kd_log_mean"})

    merged = default.merge(wide, on=["compound_id", "target"], how="inner")
    merged["shift_log"] = merged["wide_prior_kd_log_mean"] - merged["default_kd_log_mean"]
    merged["kd_ratio_widened_vs_default"] = np.exp(merged["shift_log"])
    merged["abs_shift_log"] = merged["shift_log"].abs()
    merged["shift_fold"] = np.exp(merged["abs_shift_log"])
    merged["potency_bin"] = potency_bin(merged["default_kd_log_mean"].to_numpy())

    summary = (
        merged.groupby("potency_bin")
        .agg(
            n_pairs=("compound_id", "size"),
            median_abs_shift_log=("abs_shift_log", "median"),
            median_shift_fold=("shift_fold", "median"),
            median_kd_ratio_widened_vs_default=("kd_ratio_widened_vs_default", "median"),
            corr=("default_kd_log_mean", lambda s: s.corr(merged.loc[s.index, "wide_prior_kd_log_mean"])),
        )
        .reindex(POTENCY_BIN_LABELS)
    )
    return merged, summary


def plot_prior_sensitivity(merged, out_path):
    """Interval summary of the widened-prior/default-prior Kd ratio."""
    fig, ax = plt.subplots(figsize=(4, 1.8))
    y_positions = np.arange(len(POTENCY_BIN_LABELS))[::-1]
    for y, potency in zip(y_positions, POTENCY_BIN_LABELS):
        values = merged.loc[
            merged["potency_bin"] == potency, "kd_ratio_widened_vs_default"
        ].dropna()
        if values.empty:
            continue
        q03, q25, q50, q75, q97 = values.quantile([0.03, 0.25, 0.5, 0.75, 0.97])
        ax.plot([q03, q97], [y, y], color=PURPLE, lw=1.1, alpha=0.75)
        ax.plot([q25, q75], [y, y], color=PURPLE, lw=4.5, solid_capstyle="round")
        ax.scatter(q50, y, s=28, color=PURPLE, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(q97 * 1.06, y, f"n = {len(values):,}", va="center", fontsize=7, color=MID_GREY)
    ax.axvline(1, color=INK, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.set_xscale("log")
    ax.set_yticks(y_positions, POTENCY_BIN_LABELS)
    ax.set_xlabel("Posterior mean Kd ratio (widened prior / default prior)")
    ax.set_ylabel("Default-prior potency")
    style_axis(ax, grid_axis="x")
    figure_title(
        fig,
        "Prior sensitivity is concentrated in extrapolated potency classes",
        "The log-Kd prior SD was widened from 3 to 6; dots are medians and intervals contain 50%/94% of pairs.",
    )
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_leave_doses_out(summary, out_path):
    """Heatmaps of agreement between every two-dose fit and the four-dose fit."""
    pair_order = [f"{a:g}–{b:g}" for a, b in POSSIBLE_PAIRS]
    work = summary.copy()
    work["dose_pair"] = work["pair_name"].str.replace("_", "–", regex=False)
    corr = work.pivot(index="potency_bin", columns="dose_pair", values="corr_kd_log").reindex(
        index=POTENCY_BIN_LABELS, columns=pair_order
    )
    coverage = work.pivot(index="potency_bin", columns="dose_pair", values="frac_full_within_pair_ci").reindex(
        index=POTENCY_BIN_LABELS, columns=pair_order
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), sharey=True)
    panels = [
        (axes[0], corr, "Pearson correlation of log Kd", "coolwarm", -1, 1, lambda x: f"{x:.2f}"),
        (axes[1], coverage, "Four-dose mean inside two-dose 94% HDI", "YlGnBu", 0, 1, lambda x: f"{x:.0%}"),
    ]
    for ax, matrix, title, cmap, vmin, vmax, formatter in panels:
        im = ax.pcolormesh(
            np.arange(matrix.shape[1] + 1),
            np.arange(matrix.shape[0] + 1),
            matrix.to_numpy(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )
        ax.set_xlim(0, matrix.shape[1])
        ax.set_ylim(matrix.shape[0], 0)
        ax.set_xticks(np.arange(len(matrix.columns)) + 0.5, matrix.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(matrix.index)) + 0.5, matrix.index)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j + 0.5, i + 0.5, formatter(value), ha="center", va="center", fontsize=7,
                            color="white" if abs(value) > 0.65 else INK)
        facet_title(ax, title)
        ax.set_xlabel("Retained dose pair (nM)")
    axes[0].set_ylabel("Four-dose inferred potency")
    figure_title(
        fig,
        "Leave-two-doses-out stability",
        "Each cell compares a two-dose refit with the corresponding four-dose fit within a potency class.",
    )
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Prior predictive checks
# ---------------------------------------------------------------------------


def summarize_prior_ppc_by_dose(prior_ppc_df):
    """Per-dose summary of the prior predictive p-value distribution, pooled
    across all pairs (there is no meaningful potency-bin stratification here:
    every pair is checked against the same shared prior draws, so
    kd_log_mean_molar is ~constant across pairs by construction).
    """
    return prior_ppc_df.groupby("dose_molar").agg(
        n_pairs=("compound_id", "size"),
        mean_ppc_p_value=("ppc_p_value", "mean"),
        std_ppc_p_value=("ppc_p_value", "std"),
    )


def plot_prior_ppc_by_dose(prior_ppc_df, observed_df, out_path):
    """Prior predictive check: distribution of the *simulated* response at
    each tested dose, drawn purely from the prior (before seeing any data).

    Unlike the posterior check, the healthy/expected result here is a wide,
    non-degenerate spread across the observable 0-100% range at every dose -
    a prior that instead piles up responses at an implausible extreme (e.g.
    all mass pinned at 0% or 100%) would indicate the prior itself, not just
    the fitted posterior, encodes an unreasonable assumption.
    """
    doses_nm = np.sort(prior_ppc_df["dose_molar"].unique())
    fig, axes = plt.subplots(1, len(doses_nm), figsize=(6.0, 2), sharey=True)
    for ax, dose in zip(axes, doses_nm):
        sub = prior_ppc_df[prior_ppc_df["dose_molar"] == dose]
        obs = observed_df.loc[
            np.isclose(observed_df["Compound Concentration (nM)"], dose * 1e9),
            "Percent Control",
        ]
        lower_bound, upper_bound = sub["simulated_response"].quantile([0.025, 0.975])
        observed_below = (obs < lower_bound).mean()
        observed_above = (obs > upper_bound).mean()
        bins = np.linspace(-30, 135, 34)
        ax.axvspan(-30, lower_bound, color=TAIL_GREY, zorder=0)
        ax.axvspan(upper_bound, 135, color=TAIL_GREY, zorder=0)
        ax.hist(
            sub["simulated_response"], bins=bins, density=True, color=VERMILLION,
            alpha=0.55, edgecolor="white", linewidth=0.25, label="Prior predictive",
        )
        ax.hist(
            obs, bins=bins, density=True, histtype="step", color=BLUE,
            linewidth=1.2, label="Observed",
        )
        ax.set_title(
            f"{dose * 1e9:g} nM",
            loc="left",
            pad=20,
            fontweight="bold",
            fontsize=7.2,
        )
        ax.text(
            0.01,
            1.01,
            f"< prior q2.5\n{observed_below:.1%} observed",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.7,
            color=INK,
            clip_on=False,
        )
        ax.text(
            0.99,
            1.01,
            f"> prior q97.5\n{observed_above:.1%} observed",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.7,
            color=INK,
            clip_on=False,
        )
        style_axis(ax)
        ax.set_xlim(-30, 135)
    axes[0].set_ylabel("Density")
    fig.supxlabel("Percent control", y=0.01)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 0.80))
    figure_title(
        fig,
        "Prior predictive response distributions",
        "Grey regions lie outside the dose-specific 2.5th and 97.5th percentiles of the prior predictive distribution.",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1], w_pad=0.65)
    save_figure(fig, out_path)


def plot_prior_ppc_pvalues_by_dose(prior_ppc_df, out_path):
    """Bayesian p-value distribution for the prior predictive check: at each
    tested dose, P(prior-predicted response >= observed | prior) for every
    real pair's actual observed response, pooled across pairs.

    This is the prior-predictive analogue of plot_ppc_calibration_by_dose_only,
    but the interpretation differs: since every pair is checked against the
    *same* shared prior (not its own fitted posterior), a flat/uniform-ish
    distribution here would be a coincidence, not the goal. What matters is
    whether real observed responses fall in the extreme tails of the prior
    (p near 0 or 1) at a concerning rate - that would mean the prior assigns
    real, commonly-observed data very low prior predictive density, i.e. the
    prior is too narrow/informative relative to what the data actually looks
    like. A wide, non-extreme spread (mass away from 0/1) is the healthy result.
    """
    doses_nm = np.sort(prior_ppc_df["dose_molar"].unique())
    fig, axes = plt.subplots(1, len(doses_nm), figsize=(6.0, 2.05), sharey=True)
    for ax, dose in zip(axes, doses_nm):
        sub = prior_ppc_df[prior_ppc_df["dose_molar"] == dose]
        n_pairs = sub[["compound_id", "target"]].drop_duplicates().shape[0]
        ax.axvspan(0, 0.025, color=TAIL_GREY, zorder=0)
        ax.axvspan(0.975, 1, color=TAIL_GREY, zorder=0)
        ax.hist(
            sub["ppc_p_value"], bins=20, range=(0, 1), color=VERMILLION,
            alpha=0.82, density=True, edgecolor="white", linewidth=0.25,
        )
        ax.axvline(0.5, color=INK, linestyle=(0, (2, 2)), linewidth=0.8)
        facet_title(ax, f"{dose * 1e9:g} nM\nn = {n_pairs:,}", fontsize=7.5)
        style_axis(ax)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Density")
    fig.supxlabel("Prior predictive tail probability, P(yrep ≥ yobs)", y=0.01)
    figure_title(
        fig,
        "Observed responses under the prior predictive distribution",
        "Grey bands mark observations in the lower or upper 2.5% prior predictive tail; the distribution is not expected to be uniform.",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1], w_pad=0.65)
    save_figure(fig, out_path)


def sample_prior_predictive_responses(draw_source, doses_molar, seed=0):
    """Draw one simulated response per prior draw at each dose, for plotting
    the raw prior predictive distribution (not just a p-value against real
    data). Returns a long DataFrame with columns dose_molar, simulated_response.
    """
    kd_log_draws, hill_slope_log_draws = draw_source.sample()
    kd = np.exp(kd_log_draws).astype(np.float64)
    hill_slope = np.exp(hill_slope_log_draws).astype(np.float64)

    mu = _model_predict_np(kd[None, :], hill_slope[None, :], doses_molar[None, :])[0]
    sigma = _error_model_np(mu)

    rng = np.random.default_rng(seed)
    simulated = rng.normal(mu, sigma)  # (n_draws, n_doses)

    records = []
    for j, dose in enumerate(doses_molar):
        records.extend({"dose_molar": dose, "simulated_response": r} for r in simulated[:, j])
    return pd.DataFrame.from_records(records)


def summarize_prior_responses(prior_response_df):
    """Quantify prior predictive support, boundary mass, and response extremes."""
    return prior_response_df.groupby("dose_molar")["simulated_response"].agg(
        n_draws="size",
        q025=lambda s: s.quantile(0.025),
        median="median",
        q975=lambda s: s.quantile(0.975),
        frac_below_zero=lambda s: (s < 0).mean(),
        frac_above_100=lambda s: (s > 100).mean(),
        frac_outside_physical_range=lambda s: ((s < 0) | (s > 100)).mean(),
        frac_extreme_response=lambda s: ((s < 2.5) | (s > 97.5)).mean(),
    )


def summarize_sampling_diagnostics(fit_df):
    """Aggregate first-line MCMC diagnostics for the two inferred parameters."""
    rows = []
    for parameter in ("kd_log", "hill_slope_log"):
        rhat = fit_df[f"{parameter}_r_hat"]
        ess_bulk = fit_df[f"{parameter}_ess_bulk"]
        ess_tail = fit_df[f"{parameter}_ess_tail"]
        rows.append(
            {
                "parameter": parameter,
                "n_fits": len(fit_df),
                "frac_rhat_le_1.01": (rhat <= 1.01).mean(),
                "frac_bulk_ess_ge_400": (ess_bulk >= 400).mean(),
                "frac_tail_ess_ge_400": (ess_tail >= 400).mean(),
                "max_rhat": rhat.max(),
                "min_bulk_ess": ess_bulk.min(),
                "min_tail_ess": ess_tail.min(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def stratified_enrichment(population, successes, selected, observed):
    """Exact upper tail of independent conditional hypergeometric counts.

    Strata fix their total event counts and group membership counts. Independence
    and exchangeability within these strata are assumptions of this diagnostic.
    """
    distribution = np.array([1.0])
    offset = 0
    expected = 0.0
    for n, k, m in zip(population, successes, selected):
        n, k, m = int(n), int(k), int(m)
        if not (0 <= k <= n and 0 <= m <= n and n > 0):
            raise ValueError("Invalid hypergeometric margins")
        expected += m * k / n
        lo, hi = max(0, m - (n - k)), min(m, k)
        if lo == hi:
            offset += lo
            continue
        probabilities = (np.array([1 - k / n, k / n]) if m == 1 else
                         stats.hypergeom.pmf(np.arange(lo, hi + 1), n, k, m))
        distribution = np.convolve(distribution, probabilities)
        offset += lo
    start = max(0, int(observed) - offset)
    probability = distribution[start:].sum() / distribution.sum()
    return float(np.clip(probability, 0, 1)), expected


def bh_adjust(probabilities):
    """Benjamini–Hochberg adjustment, preserving input order."""
    p = np.asarray(probabilities, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def validate_subclass_data(ppc_df):
    """Return complete, ordered four-dose data; reject ambiguous input."""
    df = ppc_df.copy()
    keys = ["compound_id", "target"]
    required = keys + ["dose_molar", "ppc_p_value", "kd_log_mean_molar"]
    if df.empty or df[required].isna().any().any():
        raise ValueError("Missing or empty PPC data")
    numeric = df[["dose_molar", "ppc_p_value", "kd_log_mean_molar"]].to_numpy()
    if not np.isfinite(numeric).all() or not df.ppc_p_value.between(0, 1).all():
        raise ValueError("PPC probabilities and log Kd must be finite and probabilities in [0, 1]")
    dose = df.dose_molar.to_numpy() * 1e9
    match = np.isclose(dose[:, None], ELIGIBLE_CONCENTRATIONS, rtol=1e-8, atol=1e-8)
    if not (match.sum(axis=1) == 1).all():
        raise ValueError("Unexpected assay dose")
    df["dose_nm"] = np.asarray(ELIGIBLE_CONCENTRATIONS)[match.argmax(axis=1)]
    if df.duplicated(keys + ["dose_nm"]).any():
        raise ValueError("Duplicate interaction/dose key")
    grouped = df.groupby(keys)
    if not grouped.size().eq(4).all():
        raise ValueError("Every interaction must have all four doses")
    df["potency_bin"] = potency_bin(df.kd_log_mean_molar.to_numpy())
    if not df.groupby(keys).potency_bin.nunique().eq(1).all():
        raise ValueError("Inconsistent potency assignments within an interaction")
    df["lower"] = df.ppc_p_value < 0.025
    df["upper"] = df.ppc_p_value > 0.975
    df["combined"] = df.lower | df.upper
    return df.sort_values(keys + ["dose_nm"]).reset_index(drop=True)


def subclass_enrichment_tables(df):
    """Curve-level and per-dose tests; one joint multiplicity correction."""
    tails = ["combined", "lower", "upper"]
    pairs = df.groupby(["compound_id", "target", "potency_bin"], as_index=False)[tails].max()
    outcomes = [("any_dose", pairs)] + [
        (f"{dose:g}_nM", df[df.dose_nm == dose]) for dose in ELIGIBLE_CONCENTRATIONS
    ]
    rows = []
    for entity, counterpart in [("compound_id", "target"), ("target", "compound_id")]:
        counts = df.groupby(entity)[tails].agg(["sum", "size"])
        for outcome, work in outcomes:
            strata = [counterpart, "potency_bin"]
            margins = work.groupby(strata)[tails].agg(["sum", "size"])
            for name, group in work.groupby(entity):
                # Each entity has at most one interaction in each counterpart/potency stratum.
                indices = pd.MultiIndex.from_frame(group[strata])
                background = margins.loc[indices]
                for tail in tails:
                    observed = int(group[tail].sum())
                    p, expected = stratified_enrichment(
                        background[(tail, "size")], background[(tail, "sum")],
                        np.ones(len(group), dtype=int), observed,
                    )
                    n_interactions = int(counts.loc[name, (tail, "size")] // 4)
                    rows.append(dict(
                        entity_type=entity, entity=name, outcome=outcome, tail=tail,
                        n_interactions=n_interactions, n_measurements=4 * n_interactions,
                        pointwise_abnormal_count=int(counts.loc[name, (tail, "sum")]),
                        pointwise_abnormal_rate=counts.loc[name, (tail, "sum")] / (4 * n_interactions),
                        n_outcome_interactions=len(group), observed_count=observed,
                        observed_rate=observed / len(group), expected_count=expected,
                        expected_rate=expected / len(group),
                        observed_expected_ratio=observed / expected if expected > 0 else np.nan,
                        excess_rate=(observed - expected) / len(group),
                        enrichment_p_value=p, sparse=n_interactions < 20,
                    ))
    result = pd.DataFrame(rows)
    result["enrichment_q_value"] = bh_adjust(result.enrichment_p_value)
    result["enriched"] = (result.enrichment_q_value < 0.05) & (result.excess_rate > 0)
    return result


def bootstrap_dose_rates(df, n_boot=10000, seed=0):
    """Compound cluster bootstrap with simultaneous max-standardized-error CIs.

    Contrasts are differences in interaction-weighted rates, second dose minus
    first dose. The 18 contrasts share one 95% simultaneous critical value.
    """
    tails = ["combined", "lower", "upper"]
    columns = pd.MultiIndex.from_product([tails, ELIGIBLE_CONCENTRATIONS])
    totals = df.groupby(["compound_id", "dose_nm"])[tails].sum().unstack().reindex(columns=columns)
    sizes = df.groupby("compound_id").size().reindex(totals.index).to_numpy() / 4
    events = totals.to_numpy(dtype=float)
    rates = events.sum(axis=0) / sizes.sum()
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(columns)))
    for start in range(0, n_boot, 250):
        stop = min(start + 250, n_boot)
        weights = rng.multinomial(len(sizes), np.full(len(sizes), 1 / len(sizes)), size=stop-start)
        draws[start:stop] = (weights @ events) / (weights @ sizes)[:, None]
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    rate_table = pd.DataFrame([
        dict(tail=tail, dose_nm=dose, n_interactions=int(sizes.sum()),
             abnormal_count=int(events[:, i].sum()), abnormal_rate=rates[i],
             ci_lower=lo[i], ci_upper=hi[i], n_compounds=len(sizes))
        for i, (tail, dose) in enumerate(columns)
    ])
    contrast_indices = [(t * 4 + a, t * 4 + b) for t in range(3)
                        for a, b in itertools.combinations(range(4), 2)]
    estimates = np.array([rates[b] - rates[a] for a, b in contrast_indices])
    replicates = np.column_stack([draws[:, b] - draws[:, a] for a, b in contrast_indices])
    se = replicates.std(axis=0, ddof=1)
    standardized = np.divide(np.abs(replicates - estimates), se,
                             out=np.zeros_like(replicates), where=se > 0)
    critical = np.quantile(standardized.max(axis=1), 0.95)
    contrasts = pd.DataFrame([
        dict(tail=columns[a][0], dose_a_nm=columns[a][1], dose_b_nm=columns[b][1],
             difference_b_minus_a=estimates[i], bootstrap_se=se[i],
             simultaneous_ci_lower=max(-1, estimates[i] - critical * se[i]),
             simultaneous_ci_upper=min(1, estimates[i] + critical * se[i]),
             bootstrap_replicates=n_boot, seed=seed, simultaneous_critical_value=critical)
        for i, (a, b) in enumerate(contrast_indices)
    ])
    return rate_table, contrasts


def plot_subclass_results(enrichment, rates, out_dir):
    fig, ax = plt.subplots(figsize=(4.5, 2.7))
    for offset, tail, color, label in [(-0.13, "combined", INK, "Either tail"),
                                     (0, "lower", BLUE, "Low p: higher response"),
                                     (0.13, "upper", VERMILLION, "High p: lower response")]:
        sub = rates[rates["tail"].eq(tail)]
        x = np.arange(4) + offset
        ax.vlines(x, sub.ci_lower * 100, sub.ci_upper * 100, color=color)
        ax.scatter(x, sub.abnormal_rate * 100, s=16, color=color, label=label)
    ax.set_xticks(range(4), [f"{d:g}" for d in ELIGIBLE_CONCENTRATIONS])
    ax.set_xlabel("Tested dose (nM)")
    ax.set_ylabel("Abnormal observations (%)")
    ax.legend(fontsize=6, loc="best")
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, out_dir / "ppc_subclass_dose_rates.pdf")
    for entity, filename in [("compound_id", "compound"), ("target", "target")]:
        sub = enrichment[(enrichment.entity_type == entity) &
                         (enrichment.outcome == "any_dose") &
                         (enrichment["tail"] == "combined") & ~enrichment["sparse"]]
        sub = sub.sort_values(["enrichment_q_value", "excess_rate", "entity"],
                              ascending=[True, False, True]).head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.0, 4.1))
        y = np.arange(len(sub))
        ax.hlines(y, sub.expected_rate * 100, sub.observed_rate * 100, color=GRID_GREY, lw=2)
        ax.scatter(sub.expected_rate * 100, y, marker="|", color=INK, s=45, label="Conditional expectation")
        for significant, color, label in [(True, VERMILLION, "Observed: q < 0.05"),
                                           (False, MID_GREY, "Observed: q ≥ 0.05")]:
            keep = sub.enriched.to_numpy() == significant
            if keep.any():
                ax.scatter(sub.observed_rate.to_numpy()[keep] * 100, y[keep],
                           color=color, s=20, label=label)
        ax.set_yticks(y, [f"{r.entity} (n={r.n_interactions})" for r in sub.itertuples()])
        ax.set_xlabel("Interactions with ≥1 abnormal dose (%)")
        ax.set_ylim(-0.7, len(sub) + 1.1)
        ax.legend(loc="upper right", fontsize=6)
        style_axis(ax, "x")
        fig.tight_layout()
        save_figure(fig, out_dir / f"ppc_subclass_{filename}_enrichment.pdf")


def run_subclass_analysis():
    """Run independently from cached PPCs, without refitting binding models."""
    df = validate_subclass_data(pd.read_csv(OUTPUT_PATH / "ppc_per_dose.csv.gz"))
    print("Testing subclass enrichment...", flush=True)
    enrichment = subclass_enrichment_tables(df)
    for entity, name in [("compound_id", "compound"), ("target", "target")]:
        enrichment[enrichment.entity_type == entity].to_csv(
            OUTPUT_PATH / f"ppc_subclass_{name}_enrichment.csv", index=False)
    rates, contrasts = bootstrap_dose_rates(df)
    rates.to_csv(OUTPUT_PATH / "ppc_subclass_dose_rates.csv", index=False)
    contrasts.to_csv(OUTPUT_PATH / "ppc_subclass_dose_contrasts.csv", index=False)
    plot_subclass_results(enrichment, rates, OUTPUT_PATH)
    print(enrichment.groupby(["entity_type", "outcome", "tail"]).enriched.sum().to_string())
    print(rates.to_string(index=False))
    return enrichment, rates, contrasts


def main(overwrite=False):
    OUTPUT_PATH.mkdir(exist_ok=True)

    ppc_per_dose_path = OUTPUT_PATH / "ppc_per_dose.csv.gz"
    ppc_per_pair_path = OUTPUT_PATH / "ppc_per_pair.csv.gz"
    ci_per_pair_path = OUTPUT_PATH / "ci_extrapolation_per_pair.csv.gz"

    observed_df = None

    def get_observed_df():
        nonlocal observed_df
        if observed_df is None:
            observed_df = load_observed_data()
        return observed_df

    if not overwrite and ppc_per_dose_path.exists():
        print(f"Loading cached {ppc_per_dose_path}...")
        ppc_df = pd.read_csv(ppc_per_dose_path)
    else:
        print("Computing posterior predictive checks...")
        draw_source = ZarrDraws(
            PAIR_DATASET_PATH / "all_eligible_posteriors.zarr", "all_eligible_concentrations"
        )
        ppc_df = compute_ppc(get_observed_df(), draw_source)
        ppc_df.to_csv(ppc_per_dose_path, index=False)

    # Always rebuild this inexpensive summary from the canonical per-dose
    # cache. This prevents stale categorical labels from silently dropping a
    # potency class when plotting code or labels change.
    per_pair, ppc_summary = summarize_ppc_by_potency(ppc_df)
    per_pair.to_csv(ppc_per_pair_path, index=False)
    ppc_summary.to_csv(OUTPUT_PATH / "ppc_summary_by_potency.csv")
    plot_ppc_calibration(ppc_df, OUTPUT_PATH / "ppc_calibration_by_potency.png")
    plot_ppc_calibration_by_dose(ppc_df, OUTPUT_PATH / "ppc_calibration_by_potency_and_dose.png")
    plot_ppc_calibration_by_dose_only(ppc_df, OUTPUT_PATH / "ppc_calibration_by_dose.png")
    plot_ppc_extreme_rate_heatmap(ppc_df, OUTPUT_PATH / "ppc_extreme_rate_by_potency_and_dose.png")
    print(ppc_summary)

    print("Summarizing credible intervals / extrapolation...")
    if not overwrite and ci_per_pair_path.exists():
        print(f"Loading cached {ci_per_pair_path}...")
        ci_df = pd.read_csv(ci_per_pair_path)
        ci_df["potency_bin"] = potency_bin(ci_df["kd_log_mean"].to_numpy())
        ci_df["ci_span_fold"] = np.exp(ci_df["ci_width_log"])
        ci_df["hill_slope_ci_width_log"] = (
            ci_df["hill_slope_log_hdi_97%"] - ci_df["hill_slope_log_hdi_3%"]
        )
        ci_df["hill_slope_ci_span_fold"] = np.exp(ci_df["hill_slope_ci_width_log"])
        ci_summary = (
            ci_df.groupby(["potency_bin", "extrapolated"])
            .agg(
                n_pairs=("compound_id", "size"),
                median_ci_width_log=("ci_width_log", "median"),
                median_ci_span_fold=("ci_span_fold", "median"),
                median_hill_slope_ci_span_fold=("hill_slope_ci_span_fold", "median"),
            )
            .reindex(pd.MultiIndex.from_product([POTENCY_BIN_LABELS, [False, True]], names=["potency_bin", "extrapolated"]))
        )
    else:
        ci_df, ci_summary = summarize_ci_extrapolation(
            PAIR_DATASET_PATH / "all_eligible_concentrations.csv.gz", get_observed_df()
        )
        ci_df.to_csv(ci_per_pair_path, index=False)
    ci_summary.to_csv(OUTPUT_PATH / "ci_summary_by_potency.csv")
    ci_df.to_csv(ci_per_pair_path, index=False)
    plot_ci_width_vs_potency(ci_df, OUTPUT_PATH / "ci_width_vs_potency.png")
    plot_hill_slope_uncertainty(ci_df, OUTPUT_PATH / "hill_slope_uncertainty_by_potency.png")
    sampling_diagnostics = summarize_sampling_diagnostics(ci_df)
    sampling_diagnostics.to_csv(OUTPUT_PATH / "sampling_diagnostics_summary.csv", index=False)
    print(ci_summary)
    print(sampling_diagnostics)

    print("Summarizing leave-doses-out stability...")
    ldo_combined, ldo_summary = summarize_leave_doses_out(
        PAIR_DATASET_PATH / "all_eligible_concentrations.csv.gz"
    )
    if not ldo_summary.empty:
        ldo_summary.to_csv(OUTPUT_PATH / "leave_doses_out_summary.csv", index=False)
        plot_leave_doses_out(ldo_summary, OUTPUT_PATH / "leave_doses_out_stability.png")
    print(ldo_summary)

    print("Summarizing prior sensitivity...")
    prior_merged, prior_summary = summarize_prior_sensitivity(
        PAIR_DATASET_PATH / "all_eligible_concentrations.csv.gz",
        PAIR_DATASET_PATH / "prior_sensitivity_wide_kd_prior.csv.gz",
    )
    if prior_summary is not None:
        prior_summary.to_csv(OUTPUT_PATH / "prior_sensitivity_summary.csv")
        plot_prior_sensitivity(prior_merged, OUTPUT_PATH / "prior_sensitivity.png")
        print(prior_summary)
    else:
        print("prior_sensitivity_wide_kd_prior.csv.gz not found yet; skipping.")

    print("Computing prior predictive checks...")
    prior_draws = PriorDraws()
    prior_ppc_per_dose_path = OUTPUT_PATH / "prior_ppc_per_dose.csv.gz"
    if not overwrite and prior_ppc_per_dose_path.exists():
        print(f"Loading cached {prior_ppc_per_dose_path}...")
        prior_ppc_df = pd.read_csv(prior_ppc_per_dose_path)
    else:
        prior_ppc_df = compute_ppc(filter_eligible_pairs(get_observed_df()), prior_draws)
        prior_ppc_df.to_csv(prior_ppc_per_dose_path, index=False)
    prior_ppc_summary = summarize_prior_ppc_by_dose(prior_ppc_df)
    prior_ppc_summary.to_csv(OUTPUT_PATH / "prior_ppc_summary_by_dose.csv")
    print(prior_ppc_summary)

    plot_prior_ppc_pvalues_by_dose(prior_ppc_df, OUTPUT_PATH / "prior_ppc_pvalues_by_dose.png")

    doses_molar = np.sort(prior_ppc_df["dose_molar"].unique())
    prior_predictive_responses = sample_prior_predictive_responses(prior_draws, doses_molar)
    prior_response_summary = summarize_prior_responses(prior_predictive_responses)
    prior_response_summary.to_csv(OUTPUT_PATH / "prior_response_summary_by_dose.csv")
    plot_prior_ppc_by_dose(
        prior_predictive_responses,
        filter_eligible_pairs(get_observed_df()),
        OUTPUT_PATH / "prior_ppc_by_dose.png",
    )
    print(prior_response_summary)

    print(f"Done. Outputs written to {OUTPUT_PATH}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute PPC/CI per-pair results from scratch instead of reusing cached CSVs in ppc_report_output/.",
    )
    parser.add_argument("--subclasses-only", action="store_true",
                        help="Analyze subclass enrichment using the existing per-dose PPC cache only.")
    args = parser.parse_args()
    if args.subclasses_only:
        run_subclass_analysis()
    else:
        main(overwrite=args.overwrite)
        run_subclass_analysis()
