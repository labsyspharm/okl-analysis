import itertools
import pathlib
import sqlite3
import contextlib
import zipfile

import numpy as np
import pandas as pd
import synapseclient
import pytensor
import zarr

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

pytensor.config.cxx = "g++"
for conc_pair in possible_combinations:
    filtered = filter_dataset_conc(okl_dataset, conc_pair)
    if len(filtered) == 0:
        continue
    pair_name = f"{conc_pair[0]}_{conc_pair[1]}"
    df = fit_kd.fit_kd_from_dataframe(
        filtered,
        db_path=pair_dataset_path / f"{pair_name}.db"
    )
    if df is not None:
        df.to_csv(pair_dataset_path / f"{pair_name}.csv.gz", index=False)

okl_all_eligible = filter_dataset_conc(okl_dataset, eligible_concentrations)
kd_all_eligible = fit_kd.fit_kd_from_dataframe(
    okl_all_eligible,
    db_path=pair_dataset_path / f"all_eligible_concentrations.db"
)

def extract_posterior_matrices(db_path: pathlib.Path, n_samples: int):
    """Stream posterior samples into preallocated float32 matrices.

    Returns
    -------
    compound_ids : list[str]
    targets      : list[str]
    kd_matrix    : (n, N_SAMPLES) float32  – log Kd
    slope_matrix : (n, N_SAMPLES) float32  – log Hill slope
    """
    with contextlib.closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        cur = conn.execute(
            "SELECT rowid, compound_id, target FROM fits WHERE status = 'ok' ORDER BY rowid"
        )
        rows = cur.fetchall()  # metadata only – no blobs
        n = len(rows)
        if n == 0:
            return [], [], None, None

        compound_ids = [r[1] for r in rows]
        targets = [r[2] for r in rows]
        idx_map = {(cid, tgt): i for i, (_, cid, tgt) in enumerate(rows)}

        kd_matrix = np.empty((n, n_samples), dtype=np.float32)
        slope_matrix = np.empty((n, n_samples), dtype=np.float32)

        for row_id, cid, tgt in rows:
            i = idx_map[(cid, tgt)]
            with conn.blobopen("fits", "kd_log_samples", row_id, readonly=True) as b:
                kd_matrix[i] = np.frombuffer(b.read(), dtype=np.float32)
            with conn.blobopen("fits", "hill_slope_log_samples", row_id, readonly=True) as b:
                slope_matrix[i] = np.frombuffer(b.read(), dtype=np.float32)
    return compound_ids, targets, kd_matrix, slope_matrix

_N_SAMPLES = 4 * 2000  # chains * draws
for conc_pair in possible_combinations:
    pair_name = f"{conc_pair[0]}_{conc_pair[1]}"
    db_path = pair_dataset_path / f"{pair_name}.db"
    if not db_path.exists():
        continue
    compound_ids, targets, kd_matrix, slope_matrix = extract_posterior_matrices(db_path, _N_SAMPLES)
    if kd_matrix is None:
        continue
    np.savez_compressed(
        pair_dataset_path / f"{pair_name}_posteriors.npz",
        compound_ids=compound_ids,
        targets=targets,
        kd=kd_matrix,
        hill_slope=slope_matrix,
    )

zarr_path = pair_dataset_path / "all_pairs_posteriors.zarr"
blosc_codec = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
root = zarr.open_group(str(zarr_path), mode="w")
for conc_pair in possible_combinations:
    pair_name = f"{conc_pair[0]}_{conc_pair[1]}"
    print(f"Processing pair {pair_name}...")
    db_path = pair_dataset_path / f"{pair_name}.db"
    if not db_path.exists():
        continue
    # compound_ids, targets, kd_matrix, slope_matrix = extract_posterior_matrices(db_path, _N_SAMPLES)
    with np.load(pair_dataset_path / f"{pair_name}_posteriors.npz") as data:
        compound_ids = data["compound_ids"].tolist()
        targets = data["targets"].tolist()
        kd_matrix = data["kd"]
        slope_matrix = data["hill_slope"]
    if kd_matrix is None:
        continue
    grp = root.require_group(pair_name)
    grp.create_array("kd", data=kd_matrix, chunks=(1, _N_SAMPLES), shards=(100, _N_SAMPLES), overwrite=True, compressors=[blosc_codec])
    grp.create_array("hill_slope", data=slope_matrix, chunks=(1, _N_SAMPLES), shards=(100, _N_SAMPLES), overwrite=True, compressors=[blosc_codec])
    grp.attrs["compound_ids"] = compound_ids
    grp.attrs["targets"] = targets

zarr_zip_path = pair_dataset_path / "all_pairs_posteriors.zarr.zip"
with zipfile.ZipFile(zarr_zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
    for file in zarr_path.rglob("*"):
        zf.write(file, file.relative_to(zarr_path))

zarr_path = pair_dataset_path / "all_eligible_posteriors.zarr"
blosc_codec = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
root = zarr.open_group(str(zarr_path), mode="w")
db_path = pair_dataset_path / "all_eligible_concentrations.db"
compound_ids, targets, kd_matrix, slope_matrix = extract_posterior_matrices(db_path, _N_SAMPLES)
# with np.load(pair_dataset_path / f"{pair_name}_posteriors.npz") as data:
#     compound_ids = data["compound_ids"].tolist()
#     targets = data["targets"].tolist()
#     kd_matrix = data["kd"]
#     slope_matrix = data["hill_slope"]
grp = root.require_group("all_eligible_concentrations")
grp.create_array("kd", data=kd_matrix, chunks=(1, _N_SAMPLES), shards=(100, _N_SAMPLES), overwrite=True, compressors=[blosc_codec])
grp.create_array("hill_slope", data=slope_matrix, chunks=(1, _N_SAMPLES), shards=(100, _N_SAMPLES), overwrite=True, compressors=[blosc_codec])
grp.attrs["compound_ids"] = compound_ids
grp.attrs["targets"] = targets

zarr_zip_path = pair_dataset_path / "all_eligible_posteriors.zarr.zip"
with zipfile.ZipFile(zarr_zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
    for file in zarr_path.rglob("*"):
        zf.write(file, file.relative_to(zarr_path))

x = zarr.open(zarr.storage.ZipStore(zarr_zip_path, mode="r"))

syn = synapseclient.login()

for f in pair_dataset_path.glob("*.csv.gz"):
    syn.store(synapseclient.File(str(f), parent="syn73756839"), forceVersion=False)
