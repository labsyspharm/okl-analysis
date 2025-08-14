from collections import defaultdict
import multiprocessing as mp
from multiprocessing import Queue, Process
import os
import json
import contextlib
import gzip
import pickle
import time
import sqlite3

import pytensor
import numpy as np
import pandas as pd
import pymc as pm
import pymc.math
import nutpie
import pystow
import tqdm
import matplotlib
import matplotlib.pyplot as plt
import synapseclient

matplotlib.use("Agg")

np.random.seed(42)

BASE = pystow.module("cipn", "kinomescan")
syn = synapseclient.login()

PARALLELIZE = True
BATCH_SIZE = 10  # Number of drug/target pairs per batch
OKL_SINGLE_DOSE_DATA = syn.get("syn52504516").path
OKL_PSEUDO_KD_DATA = pd.read_csv(syn.get("syn51080578").path)
OKL_PSEUDO_KD_DATA = OKL_PSEUDO_KD_DATA[
    OKL_PSEUDO_KD_DATA["dataset"] == "original_repeat_replaced"
]
DB_PATH = BASE.join("results", name="fits.sqlite").as_posix()


pytensor.config.cxx = "/usr/bin/g++"
# pytensor.config.cxx = "/usr/bin/clang++"


def model_predict(kd, hill_slope, doses):
    return 100 - 100 / (1 + (kd / doses) ** hill_slope)


# (Intercept) -0.0429160  0.0188197   -2.28   0.0458 *
# V2           0.1499235  0.0002977  503.52   <2e-16 ***
# From KINOMEscan General Introduction.pptx
def error_model(mu):
    # constrain to minimum of 1
    return pymc.math.maximum(0.15 * mu - 0.043, 1)


def simulated_data(kd, hill_slope, doses, error_std):
    # Model for data generation
    mu_true = model_predict(kd, hill_slope, doses)
    responses = np.random.normal(mu_true, scale=error_std)
    data_errors = np.full_like(responses, error_std)
    return doses, responses, data_errors


def create_model_template():
    """Create a PyMC model template that can handle varying numbers of data points."""
    with pm.Model() as model:
        # Priors for kd and hill_slope using lognormal (via normal on log)
        kd_log = pm.Normal("kd_log", mu=np.log(1e-6), sigma=3)
        kd = pm.Deterministic("kd", pm.math.exp(kd_log))

        hill_slope_log = pm.Normal("hill_slope_log", mu=0, sigma=0.5)
        hill_slope = pm.Deterministic("hill_slope", pm.math.exp(hill_slope_log))

        # Use mutable data containers that can be updated with different shapes
        doses_data = pm.Data("doses_data", np.array([1e-9]))
        responses_data = pm.Data("responses_data", np.array([0.0]))

        mu = model_predict(kd, hill_slope, doses_data)

        # Likelihood of observed data:
        sigma = error_model(mu)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=responses_data)

    return model


def train_model(doses, responses, compiled_sampler=None):
    """Train the model using a pre-compiled sampler to avoid recompilation.

    Args:
        doses: Array of dose values
        responses: Array of response values
        compiled_sampler: Pre-compiled nutpie sampler, if None will create new one

    Returns:
        trace: PyMC trace object
        compiled_sampler: The compiled sampler (for reuse)
    """
    if compiled_sampler is None:
        # Create the model template and compile it
        model = create_model_template()
        compiled_sampler = nutpie.compile_pymc_model(model)

    # Sample using the compiled sampler
    trace = nutpie.sample(
        compiled_sampler.with_data(doses_data=doses, responses_data=responses),
        draws=2000,
        tune=1000,
        chains=4,
        cores=1,
        progress_bar=False,
    )

    return trace, compiled_sampler


def _get_db_connection(db_path: str) -> sqlite3.Connection:
    """Create a SQLite connection and ensure schema/PRAGMAs.

    Single connection is used in the main process to avoid multi-writer issues.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60)
    # Improve concurrency and durability
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fits (
            compound_id TEXT NOT NULL,
            target TEXT NOT NULL,
            kd REAL,
            kd_std REAL,
            hill_slope REAL,
            hill_slope_std REAL,
            trace_path TEXT,
            plot_path TEXT,
            status TEXT,
            summary_json TEXT,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (compound_id, target)
        )
        """
    )
    return conn


def _fetch_existing_pairs(conn: sqlite3.Connection) -> set:
    """Return set of (compound_id, target) already present in DB."""
    cur = conn.execute("SELECT compound_id, target FROM fits")
    return set(cur.fetchall())


def _upsert_fit(conn: sqlite3.Connection, r: dict) -> None:
    """Insert or update a fit result row."""
    conn.execute(
        """
        INSERT INTO fits (
            compound_id, target, kd, kd_std, hill_slope, hill_slope_std,
            trace_path, plot_path, status, summary_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(compound_id, target) DO UPDATE SET
            kd=excluded.kd,
            kd_std=excluded.kd_std,
            hill_slope=excluded.hill_slope,
            hill_slope_std=excluded.hill_slope_std,
            trace_path=excluded.trace_path,
            plot_path=excluded.plot_path,
            status=excluded.status,
            summary_json=excluded.summary_json,
            updated_at=excluded.updated_at
        """,
        (
            r["compound_id"],
            r["target"],
            r.get("kd"),
            r.get("kd_std"),
            r.get("hill_slope"),
            r.get("hill_slope_std"),
            r.get("trace_path"),
            r.get("plot_path"),
            r.get("status", "ok"),
            r.get("summary_json"),
            int(time.time()),
        ),
    )


def load_data(fname, okl_only=True):
    df = pd.read_csv(fname)
    df = df[df["dataset"] == "original_repeat_replaced"]
    if okl_only:
        df = df[df["library"] == "OKL"]
    return df


def process_data(fname, okl_only=True):
    df = load_data(fname, okl_only=okl_only)
    # Organize data by DiscoveRx Gene Symbol and then dose
    # into a dict of two lists
    data = {}
    for _, row in df.iterrows():
        compound_id = row["hmsl_id"]
        gene = row["DiscoveRx Gene Symbol"]
        dose = row["Compound Concentration (nM)"]
        response = row["Percent Control"]
        if compound_id not in data:
            data[compound_id] = {}
        if gene not in data[compound_id]:
            data[compound_id][gene] = {"doses": [], "responses": []}

        data[compound_id][gene]["doses"].append(dose)
        data[compound_id][gene]["responses"].append(response)
    # Build mapping from DiscoveRx gene symbol to HGNC symbol
    mappings = dict(zip(df["DiscoveRx Gene Symbol"], df["hgnc_symbol"]))
    return data, mappings


def get_data_at_dose(data, dose):
    data_at_dose = defaultdict(dict)
    for drug_id, drug_data in data.items():
        for target, target_data in drug_data.items():
            for drug_dose, response in zip(
                target_data["doses"], target_data["responses"]
            ):
                if drug_dose == dose:
                    data_at_dose[drug_id][target] = response
    return dict(data_at_dose)


def plot_results(trace, dose_range, doses, responses, pseudo_kd, fname):
    # Extract posterior samples of Kd and HillSlope
    # These have shape (chains, draws), so flatten them into a single dimension.
    kd_samples = trace.posterior["kd"].values.reshape(-1)
    hill_slope_samples = trace.posterior["hill_slope"].values.reshape(-1)

    # For each posterior sample, compute the predicted response at each dose in dose_range
    # pred_matrix will have shape (n_samples, len(dose_range))
    pred_matrix = []
    for kd, hill_slope in zip(kd_samples, hill_slope_samples):
        mu = model_predict(kd, hill_slope, dose_range)
        pred_matrix.append(mu)
    pred_matrix = np.array(pred_matrix)  # shape: (n_samples, n_dose_points)

    # Compute mean and 95% credible intervals
    pred_mean = np.mean(pred_matrix, axis=0)
    pred_lower = np.percentile(pred_matrix, 2.5, axis=0)
    pred_upper = np.percentile(pred_matrix, 97.5, axis=0)

    kd_mean = np.mean(kd_samples)
    kd_std = np.std(kd_samples)

    # Use explicit fig, ax API and ensure figures are closed to avoid memory leaks
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot the measurements
    ax.scatter(doses, responses, color="black", label="Observed Data")

    # Plot the posterior predictive mean and credible interval over the specified dose range
    ax.plot(dose_range, pred_mean, color="red", label="Posterior Predictive Mean")
    ax.fill_between(
        dose_range,
        pred_lower,
        pred_upper,
        color="red",
        alpha=0.3,
        label="95% Credible Interval",
    )

    # Put the Kd estimate in the plot as a solid vertical line with dashed lines for +/- 1 SD
    ax.axvline(kd_mean, color="green", linestyle="-", label="Estimated Kd")
    ax.axvline(
        kd_mean - kd_std, color="green", linestyle="--", label="Estimated Kd +/- 1 SD"
    )
    ax.axvline(kd_mean + kd_std, color="green", linestyle="--")

    # Plot pseudo Kd in blue
    ax.axvline(pseudo_kd, color="blue", linestyle="-", label="Pseudo Kd")

    ax.set_xscale("log")
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Dose")
    ax.set_ylabel("Response")
    ax.set_title("Dose-Response with Posterior Predictive")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def process_target(
    drug_id, target, target_data, pseudo_kd, dose_range, compiled_sampler=None
):
    pseudo_kd *= 1e-9
    doses = np.array([d * 1e-9 for d in target_data["doses"]])
    responses = np.array(target_data["responses"])

    # Use compiled sampler
    trace, compiled_sampler = train_model(doses, responses, compiled_sampler)

    # import ipdb; ipdb.set_trace()
    # sigma_obs = np.mean(trace.posterior.sigma.values.reshape(-1))
    kd_samples = trace.posterior["kd"].values.reshape(-1)
    hill_slope_samples = trace.posterior["hill_slope"].values.reshape(-1)
    kd_means = np.mean(kd_samples)
    kd_std = np.std(kd_samples)
    hill_slope_means = np.mean(hill_slope_samples)
    hill_slope_std = np.std(hill_slope_samples)
    summary = pm.summary(trace)
    # Serialize summary DataFrame to JSON for persistence
    try:
        summary_json = summary.to_json(orient="table")
    except Exception:
        # Fallback: convert to dict then to json
        summary_json = json.dumps(summary.to_dict())
    # data_errors = np.full_like(responses, sigma_obs)
    plot_fname = BASE.join(
        "results", drug_id, name=f"{drug_id}_{target}_dose_response.png"
    ).as_posix()
    plot_results(trace, dose_range, doses, responses, pseudo_kd, plot_fname)
    trace_fname = BASE.join(
        "results", drug_id, name=f"{drug_id}_{target}_trace.pkl.gz"
    ).as_posix()
    with gzip.open(trace_fname, "wb") as fh:
        pickle.dump(trace, fh)
    return (
        drug_id,
        target,
        {
            "kd": kd_means,
            "kd_std": kd_std,
            "hill_slope": hill_slope_means,
            "hill_slope_std": hill_slope_std,
            # Keep paths for trace/plot for quick lookup
            "trace_path": trace_fname,
            "plot_path": plot_fname,
            "summary_json": summary_json,
        },
        compiled_sampler,
    )


def get_drugs_by_dose(data):
    """Return a dict of drug identifiers by dose for which there is data."""
    drugs_by_dose = defaultdict(set)
    for drug, drug_data in data.items():
        for gene, gene_data in drug_data.items():
            for dose in gene_data["doses"]:
                drugs_by_dose[dose].add(drug)
    return dict(drugs_by_dose)


def filter_to_dose_missing(data, drugs_by_dose, dose_missing):
    """Filter data to drugs missing a given dose."""
    return {
        drug_id: drug_data
        for drug_id, drug_data in data.items()
        if drug_id not in drugs_by_dose[dose_missing]
    }


pseudo_kd_map = {}
for _, row in OKL_PSEUDO_KD_DATA.iterrows():
    pseudo_kd_map[(row["hmsl_id"], row["DiscoveRx Gene Symbol"])] = row["pseudo_kd"]

# Dose range for plotting purposes
dose_range = np.logspace(-10, 0, 50)  # from 1 nM to 1 mM

# Load and process the data
data, gene_mappings = process_data(OKL_SINGLE_DOSE_DATA)

data_to_train = data

# Initialize SQLite persistence and pre-load existing (compound_id, target) pairs
conn = _get_db_connection(DB_PATH)
existing_pairs = _fetch_existing_pairs(conn)
print(f"Found {len(existing_pairs)} existing (compound_id, target) pairs in DB.")
_writes_since_commit = 0

# Pre-compile the sampler once using dummy data to avoid recompilation
print("Pre-compiling the Bayesian model...")
dummy_doses = np.array([1e-9, 1e-8, 1e-7])
dummy_responses = np.array([10.0, 50.0, 90.0])
model_template = create_model_template()
global_compiled_sampler = nutpie.compile_pymc_model(model_template, backend="numba")
print("Model compilation complete.")


def worker_process(work_queue, result_queue, pseudo_kd_map, dose_range):
    """Persistent worker process that compiles nutpie model once and reuses it."""
    # Each worker compiles its own model to avoid sharing compiled objects across processes
    try:
        model_template = create_model_template()
        compiled_sampler = nutpie.compile_pymc_model(model_template, backend="numba")

        while True:
            try:
                # Get work item from queue (blocks until available)
                work_item = work_queue.get(timeout=1)

                # Poison pill to terminate worker
                if work_item is None:
                    break

                drug_id, target, target_data = work_item

                # Process the drug/target pair
                with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(
                    fnull
                ), contextlib.redirect_stderr(fnull):

                    try:
                        _, _, result, _ = process_target(
                            drug_id,
                            target,
                            target_data,
                            pseudo_kd_map[(drug_id, target)],
                            dose_range,
                            compiled_sampler,
                        )
                        record = {
                            "compound_id": drug_id,
                            "target": target,
                            "status": "ok",
                            **result,
                        }
                    except Exception as e:
                        # Log error and continue
                        record = {
                            "compound_id": drug_id,
                            "target": target,
                            "status": f"error: {str(e)}",
                            "kd": None,
                            "kd_std": None,
                            "hill_slope": None,
                            "hill_slope_std": None,
                            "trace_path": None,
                            "plot_path": None,
                            "summary_json": None,
                        }

                    # Send result back
                    result_queue.put(record)

            except mp.TimeoutError:
                # Check for more work
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                break

    except Exception as e:
        print(f"Worker initialization error: {e}")
        return


if PARALLELIZE:
    # Collect all drug/target pairs that need processing
    items_to_process = []
    for drug_id, drug_data in data_to_train.items():
        for target, target_data in drug_data.items():
            # Skip if already in DB
            if (drug_id, target) in existing_pairs:
                continue
            items_to_process.append((drug_id, target, target_data))

    print(f"Found {len(items_to_process)} drug/target pairs to process")

    if items_to_process:
        # Create queues for work distribution and result collection
        num_workers = min(mp.cpu_count(), 18)  # Limit to 18 workers
        work_queue = Queue()
        result_queue = Queue()

        # Start worker processes
        workers = []
        for i in range(num_workers):
            worker = Process(
                target=worker_process,
                args=(work_queue, result_queue, pseudo_kd_map, dose_range),
            )
            worker.start()
            workers.append(worker)

        # Add all work items to the queue
        for item in items_to_process:
            work_queue.put(item)

        # Add poison pills to terminate workers
        for _ in range(num_workers):
            work_queue.put(None)

        # Collect results
        collected_results = 0
        total_items = len(items_to_process)

        with tqdm.tqdm(total=total_items, desc="Processing drug/target pairs") as pbar:
            while collected_results < total_items:
                try:
                    record = result_queue.get(timeout=300)
                    _upsert_fit(conn, record)
                    _writes_since_commit += 1
                    if _writes_since_commit % 100 == 0:
                        conn.commit()
                    collected_results += 1
                    pbar.update(1)
                except mp.TimeoutError:
                    print("Warning: Timeout waiting for results")
                    break

        # Wait for all workers to finish
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                print(f"Warning: Worker {worker.pid} still running, terminating...")
                worker.terminate()
                worker.join()

        conn.commit()
        print(f"Completed processing {collected_results}/{total_items} items")
    else:
        print("No new drug/target pairs to process")

else:
    for drug_id, drug_data in tqdm.tqdm(data_to_train.items(), desc="Processing drugs"):
        for target, target_data in drug_data.items():
            if (drug_id, target) in existing_pairs:
                continue
            _, _, result, _ = process_target(
                drug_id,
                target,
                target_data,
                pseudo_kd_map[(drug_id, target)],
                dose_range,
                global_compiled_sampler,
            )
            record = {
                "compound_id": drug_id,
                "target": target,
                **result,
            }
            _upsert_fit(conn, record)
            _writes_since_commit += 1
            if _writes_since_commit % 50 == 0:
                conn.commit()
    conn.commit()

    # Query for drug_id, target pairs with errors
error_query = """
SELECT compound_id, target, status
FROM fits
WHERE status LIKE 'error:%'
ORDER BY compound_id, target
"""

error_pairs = conn.execute(error_query).fetchall()
print(f"Found {len(error_pairs)} drug/target pairs with errors:")
for compound_id, target, status in error_pairs:
    print(f"  {compound_id}, {target}: {status}")

# Remove all errored combinations from the database
delete_query = """
DELETE FROM fits
WHERE status LIKE 'error:%'
"""

deleted_count = conn.execute(delete_query).rowcount
conn.commit()
print(f"Removed {deleted_count} errored drug/target combinations from the database.")

# Query for drug_id, target pairs created within the last two hours
two_hours_ago = int(time.time()) - 1000

recent_query = """
SELECT compound_id, target, updated_at, status
FROM fits
WHERE updated_at > ?
ORDER BY updated_at DESC
"""

recent_pairs = conn.execute(recent_query, (two_hours_ago,)).fetchall()
print(f"Found {len(recent_pairs)} drug/target pairs created within the last two hours:")
for compound_id, target, updated_at, status in recent_pairs:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(updated_at))
    print(f"  {compound_id}, {target}: {status} (created at {timestamp})")
