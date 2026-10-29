"""Posterior predictive checks and validation report for the Kd Bayesian fits.

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

3. Prior sensitivity: compares the default-prior all_eligible fit against the
   wide-prior subsample fit in prior_sensitivity_wide_kd_prior.csv.gz.

4. Prior predictive checks: same PPC machinery as (1), but driven by i.i.d.
   draws from the model's prior (no fitting, no stored posterior) via
   PriorDraws, to check whether the prior itself implies a plausible spread
   of dose-response behavior before any data is seen.

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


def save_figure(fig, out_path):
    """Write a PDF and a high-resolution PNG"""
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
    dataset used by fit_kd_dose_pairs.py"""
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

    summary = (
        per_pair.groupby("potency_bin")
        .agg(
            n_pairs=("compound_id", "size"),
            frac_extreme_predictive_tail=("flagged_extreme_tail", "mean"),
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
    fig.tight_layout(rect=[0, 0.07, 1, 1], w_pad=0.65)
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
    fig.tight_layout()
    save_figure(fig, out_path)

# ---------------------------------------------------------------------------
# 3. Prior sensitivity
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
    fig.tight_layout()
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Prior predictive checks
# ---------------------------------------------------------------------------


def summarize_prior_ppc_by_dose(prior_ppc_df):
    """Per-dose summary of the prior predictive p-value distribution, pooled
    across all pairs (there is no meaningful potency-bin stratification here:
    every pair is checked against the same shared prior draws, so
    kd_log_mean_molar is ~constant across pairs.
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
    sampling_diagnostics = summarize_sampling_diagnostics(ci_df)
    sampling_diagnostics.to_csv(OUTPUT_PATH / "sampling_diagnostics_summary.csv", index=False)
    print(ci_summary)
    print(sampling_diagnostics)

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
    args = parser.parse_args()
    main(overwrite=args.overwrite)
