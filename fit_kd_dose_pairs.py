import itertools
import pathlib

import pandas as pd
import synapseclient

import fit_kd_cli as fit_kd

all_okl = pd.read_csv("okl_single_dose_datasets.csv.gz")

okl_dataset = all_okl[
    (all_okl["dataset"] == "original_repeat_replaced") & (all_okl["library"] == "OKL")
]

okl_dataset["Compound Concentration (nM)"].value_counts()

eligible_concentrations = [12.5, 100, 1000, 10000]

possible_combinations = list(itertools.combinations(eligible_concentrations, 2))


def filter_dataset_conc(dataset, concentrations):
    return dataset[dataset["Compound Concentration (nM)"].isin(concentrations)]


pair_dataset_path = pathlib.Path("okl_concentration_pair_datasets")
pair_dataset_path.mkdir(exist_ok=True)

for conc_pair in possible_combinations:
    filtered = filter_dataset_conc(okl_dataset, conc_pair)
    if len(filtered) == 0:
        continue
    pair_name = f"{conc_pair[0]}_{conc_pair[1]}"
    _, _, df = fit_kd.fit_kd_from_dataframe(
        filtered,
        db_path=pair_dataset_path / f"{pair_name}.db",
        cxx="/usr/bin/g++",
    )
    if df is not None:
        df.to_csv(pair_dataset_path / f"{pair_name}.csv.gz", index=False)

syn = synapseclient.login()

for f in pair_dataset_path.glob("*.csv.gz"):
    syn.store(synapseclient.File(str(f), parent="syn73756839"), forceVersion=False)
