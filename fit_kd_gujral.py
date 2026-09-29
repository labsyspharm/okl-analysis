"""Fit Kd and Hill slope for the Gujral dose-response data (percent inhibition)."""

import numpy as np
import pandas as pd
import synapseclient

import fit_kd_cli as fit_kd

DB_PATH = "data/gujral_kd_fits.db"
CSV_OUTPUT = "data/gujral_kd_fits.csv.gz"


def load_gujral():
    syn = synapseclient.login()
    # File is Mac Roman encoded; "PHA-848125" carries a trailing non-breaking space (0xCA)
    gujral_long = pd.read_csv(
        syn.get("syn77577944").path,
        encoding="mac_roman",
    )
    gujral_long["Name"] = gujral_long["Name"].str.replace("\xa0", "").str.strip()
    return gujral_long


def prepare_for_fitting(gujral_long):
    """Map Gujral columns onto the ones expected by fit_kd_cli.

    Most drugs are included twice with identical values; those exact duplicates
    are dropped. Staurosporine has two distinct replicate curves, both are kept.
    The "IC50" column is percent inhibition (0 -> 100 with dose).
    """
    df = gujral_long.drop_duplicates(["Name", "Kinase", "Dose", "IC50"])
    return pd.DataFrame(
        {
            "hmsl_id": df["Name"],
            "lspci_id": df["lspci_id"],
            "DiscoveRx Gene Symbol": df["Kinase"],
            "Compound Concentration (nM)": df["Dose"]
            .str.removesuffix("uM")
            .astype(float)
            * 1e3,
            "Percent Control": np.clip(df["IC50"], 0, 100),
        }
    )


if __name__ == "__main__":
    gujral_fit_input = prepare_for_fitting(load_gujral())

    # Curves are nearly noise-free, so the KINOMEscan error model does not
    # apply; learn a constant sigma per curve instead.
    res = fit_kd.fit_kd_from_dataframe(
        gujral_fit_input,
        db_path=DB_PATH,
        direction="increasing",
        error_model="homoscedastic",
        sigma_prior_scale=10.0,
    )

    compound_map = gujral_fit_input[["hmsl_id", "lspci_id"]].drop_duplicates()
    res = res.merge(
        compound_map.rename(columns={"hmsl_id": "compound_id"}),
        on="compound_id",
        how="left",
    )
    res.to_csv(CSV_OUTPUT, index=False)
    print(f"Results written to {CSV_OUTPUT}")
