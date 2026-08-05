"""Posterior predictive checks and validation report for the Kd Bayesian fits.

Runs three complementary internal-validation analyses against the already-
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

Outputs CSV summary tables and PNG plots into ppc_report_output/.
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
POTENCY_BIN_LABELS = [
    "<1 nM",
    "1-10 nM",
    "10-100 nM",
    "100 nM-10 uM",
    ">10 uM",
]


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
    per_pair = (
        ppc_df.groupby(["compound_id", "target"])
        .agg(
            kd_log_mean_molar=("kd_log_mean_molar", "first"),
            mean_abs_std_resid=("std_resid", lambda s: s.abs().mean()),
            max_abs_std_resid=("std_resid", lambda s: s.abs().max()),
            min_ppc_p_value=("ppc_p_value", "min"),
        )
        .reset_index()
    )
    per_pair["potency_bin"] = potency_bin(per_pair["kd_log_mean_molar"].to_numpy())
    # A pair is flagged as poorly modeled if any dose point is an extreme
    # two-sided outlier under the fitted Normal error model.
    per_pair["flagged_outlier"] = per_pair["max_abs_std_resid"] > 3

    summary = (
        per_pair.groupby("potency_bin")
        .agg(
            n_pairs=("compound_id", "size"),
            frac_flagged=("flagged_outlier", "mean"),
            median_mean_abs_std_resid=("mean_abs_std_resid", "median"),
        )
        .reindex(POTENCY_BIN_LABELS)
    )
    return per_pair, summary


def plot_ppc_calibration(ppc_df, out_path):
    """Histogram of per-dose-point PPC p-values, pooled across all dose points
    in each potency bin.

    Each panel is normalized to a density (area = 1) rather than raw counts,
    since bin sizes range from ~500 to ~50,000 pairs and a shared count axis
    would make the small bins invisible. A dashed line at density = 1 marks
    the expected level under perfect calibration.
    """
    ppc_df = ppc_df.copy()
    ppc_df["potency_bin"] = potency_bin(ppc_df["kd_log_mean_molar"].to_numpy())
    fig, axes = plt.subplots(1, len(POTENCY_BIN_LABELS), figsize=(4 * len(POTENCY_BIN_LABELS), 3), sharey=True)
    for ax, label in zip(axes, POTENCY_BIN_LABELS):
        sub = ppc_df[ppc_df["potency_bin"] == label]
        n_pairs = sub[["compound_id", "target"]].drop_duplicates().shape[0]
        ax.hist(sub["ppc_p_value"], bins=20, range=(0, 1), color="steelblue", density=True)
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{label}\n(n={n_pairs} pairs)", fontsize=9)
        ax.set_xlabel("PPC p-value")
    axes[0].set_ylabel("density (per dose point)")
    fig.suptitle("Per-dose PPC p-value distribution by potency bin")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ppc_calibration_by_dose(ppc_df, out_path):
    """Grid of PPC p-value distributions broken down by potency bin (rows)
    AND by the actual tested dose (columns), to check whether miscalibration
    is uniform across the dose-response curve or concentrated at specific
    concentrations (e.g. the lowest/highest dose, where the Hill curve is
    flattest and the error model's sigma floor dominates).

    Each panel is normalized to a density, with a dashed reference line at
    density = 1 for perfect calibration, same convention as
    plot_ppc_calibration.
    """
    ppc_df = ppc_df.copy()
    ppc_df["potency_bin"] = potency_bin(ppc_df["kd_log_mean_molar"].to_numpy())
    doses_nm = np.sort(ppc_df["dose_molar"].unique())

    fig, axes = plt.subplots(
        len(POTENCY_BIN_LABELS),
        len(doses_nm),
        figsize=(3.2 * len(doses_nm), 2.6 * len(POTENCY_BIN_LABELS)),
        sharex=True,
    )
    for row, potency_label in enumerate(POTENCY_BIN_LABELS):
        for col, dose in enumerate(doses_nm):
            ax = axes[row, col]
            sub = ppc_df[(ppc_df["potency_bin"] == potency_label) & (ppc_df["dose_molar"] == dose)]
            n_pairs = sub[["compound_id", "target"]].drop_duplicates().shape[0]
            if n_pairs > 0:
                ax.hist(sub["ppc_p_value"], bins=20, range=(0, 1), color="steelblue", density=True)
            ax.axhline(1, color="black", linestyle="--", linewidth=1)
            if row == 0:
                ax.set_title(f"dose = {dose * 1e9:.0f} nM", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{potency_label}\n(n={n_pairs})", fontsize=8)
            if row == len(POTENCY_BIN_LABELS) - 1:
                ax.set_xlabel("PPC p-value")
    fig.suptitle(
        "Per-dose PPC p-value distribution by potency bin x dose"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ppc_calibration_by_dose_only(ppc_df, out_path):
    """PPC p-value distribution broken down by tested dose only, pooling
    across all potency bins. Complements plot_ppc_calibration (potency only)
    and plot_ppc_calibration_by_dose (potency x dose): this isolates whether
    a given dose is miscalibrated on average, independent of potency bin.
    """
    doses_nm = np.sort(ppc_df["dose_molar"].unique())
    fig, axes = plt.subplots(1, len(doses_nm), figsize=(4 * len(doses_nm), 3), sharey=True)
    for ax, dose in zip(axes, doses_nm):
        sub = ppc_df[ppc_df["dose_molar"] == dose]
        n_pairs = sub[["compound_id", "target"]].drop_duplicates().shape[0]
        ax.hist(sub["ppc_p_value"], bins=20, range=(0, 1), color="steelblue", density=True)
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"dose = {dose * 1e9:.0f} nM\n(n={n_pairs} curves)", fontsize=9)
        ax.set_xlabel("PPC p-value")
    axes[0].set_ylabel("density (per dose point)")
    fig.suptitle("Per-dose PPC p-value distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    df["potency_bin"] = potency_bin(df["kd_log_mean"].to_numpy())

    summary = (
        df.groupby(["potency_bin", "extrapolated"])
        .agg(n_pairs=("compound_id", "size"), median_ci_width_log=("ci_width_log", "median"))
        .reindex(pd.MultiIndex.from_product([POTENCY_BIN_LABELS, [False, True]], names=["potency_bin", "extrapolated"]))
    )
    return df, summary


def plot_ci_width_vs_potency(df, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for extrap, color, label in [(False, "steelblue", "interpolated"), (True, "indianred", "extrapolated")]:
        sub = df[df["extrapolated"] == extrap]
        means = sub.groupby("potency_bin")["ci_width_log"].median().reindex(POTENCY_BIN_LABELS)
        ax.plot(POTENCY_BIN_LABELS, means, marker="o", color=color, label=label)
    ax.set_ylabel("median log-Kd 94% CI width")
    ax.set_xlabel("potency bin")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    merged["abs_shift_log"] = (merged["wide_prior_kd_log_mean"] - merged["default_kd_log_mean"]).abs()
    merged["potency_bin"] = potency_bin(merged["default_kd_log_mean"].to_numpy())

    summary = (
        merged.groupby("potency_bin")
        .agg(
            n_pairs=("compound_id", "size"),
            median_abs_shift_log=("abs_shift_log", "median"),
            corr=("default_kd_log_mean", lambda s: s.corr(merged.loc[s.index, "wide_prior_kd_log_mean"])),
        )
        .reindex(POTENCY_BIN_LABELS)
    )
    return merged, summary


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


def plot_prior_ppc_by_dose(prior_ppc_df, out_path):
    """Prior predictive check: distribution of the *simulated* response at
    each tested dose, drawn purely from the prior (before seeing any data).

    Unlike the posterior check, the healthy/expected result here is a wide,
    non-degenerate spread across the observable 0-100% range at every dose -
    a prior that instead piles up responses at an implausible extreme (e.g.
    all mass pinned at 0% or 100%) would indicate the prior itself, not just
    the fitted posterior, encodes an unreasonable assumption.
    """
    doses_nm = np.sort(prior_ppc_df["dose_molar"].unique())
    fig, axes = plt.subplots(1, len(doses_nm), figsize=(4 * len(doses_nm), 3), sharey=True)
    for ax, dose in zip(axes, doses_nm):
        sub = prior_ppc_df[prior_ppc_df["dose_molar"] == dose]
        ax.hist(sub["simulated_response"], bins=30, range=(0, 100), color="indianred")
        ax.set_title(f"dose = {dose * 1e9:.0f} nM", fontsize=9)
        ax.set_xlabel("simulated % control")
    axes[0].set_ylabel("count (prior draws)")
    fig.suptitle("Prior predictive response distribution by dose")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    fig, axes = plt.subplots(1, len(doses_nm), figsize=(4 * len(doses_nm), 3), sharey=True)
    for ax, dose in zip(axes, doses_nm):
        sub = prior_ppc_df[prior_ppc_df["dose_molar"] == dose]
        n_pairs = sub[["compound_id", "target"]].drop_duplicates().shape[0]
        ax.hist(sub["ppc_p_value"], bins=20, range=(0, 1), color="indianred", density=True)
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"dose = {dose * 1e9:.0f} nM\n(n={n_pairs} pairs)", fontsize=9)
        ax.set_xlabel("prior predictive p-value")
    axes[0].set_ylabel("density (per dose point)")
    fig.suptitle(
        "Prior predictive Bayesian p-value by dose: P(prior-simulated response >= observed)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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

    if not overwrite and ppc_per_pair_path.exists():
        print(f"Loading cached {ppc_per_pair_path}...")
        per_pair = pd.read_csv(ppc_per_pair_path)
        ppc_summary = (
            per_pair.groupby("potency_bin")
            .agg(
                n_pairs=("compound_id", "size"),
                frac_flagged=("flagged_outlier", "mean"),
                median_mean_abs_std_resid=("mean_abs_std_resid", "median"),
            )
            .reindex(POTENCY_BIN_LABELS)
        )
    else:
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
        ci_summary = (
            ci_df.groupby(["potency_bin", "extrapolated"])
            .agg(n_pairs=("compound_id", "size"), median_ci_width_log=("ci_width_log", "median"))
            .reindex(pd.MultiIndex.from_product([POTENCY_BIN_LABELS, [False, True]], names=["potency_bin", "extrapolated"]))
        )
    else:
        ci_df, ci_summary = summarize_ci_extrapolation(
            PAIR_DATASET_PATH / "all_eligible_concentrations.csv.gz", get_observed_df()
        )
        ci_df.to_csv(ci_per_pair_path, index=False)
    ci_summary.to_csv(OUTPUT_PATH / "ci_summary_by_potency.csv")
    plot_ci_width_vs_potency(ci_df, OUTPUT_PATH / "ci_width_vs_potency.png")
    print(ci_summary)

    print("Summarizing leave-doses-out stability...")
    ldo_combined, ldo_summary = summarize_leave_doses_out(
        PAIR_DATASET_PATH / "all_eligible_concentrations.csv.gz"
    )
    if not ldo_summary.empty:
        ldo_summary.to_csv(OUTPUT_PATH / "leave_doses_out_summary.csv", index=False)
    print(ldo_summary)

    print("Summarizing prior sensitivity...")
    prior_merged, prior_summary = summarize_prior_sensitivity(
        PAIR_DATASET_PATH / "all_eligible_concentrations.csv.gz",
        PAIR_DATASET_PATH / "prior_sensitivity_wide_kd_prior.csv.gz",
    )
    if prior_summary is not None:
        prior_summary.to_csv(OUTPUT_PATH / "prior_sensitivity_summary.csv")
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
    plot_prior_ppc_by_dose(prior_predictive_responses, OUTPUT_PATH / "prior_ppc_by_dose.png")

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
