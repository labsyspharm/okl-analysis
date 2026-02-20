"""Fit Kd and Hill slope for compound/target pairs from single-dose KINOMEscan data."""

import contextlib
import json
import multiprocessing as mp
from multiprocessing import Process, Queue
import os
import sqlite3
import time

import click
import numpy as np
import pandas as pd
import pymc as pm
import pymc.math
import nutpie
import pytensor
import tqdm

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _model_predict(kd, hill_slope, doses):
    return 100 - 100 / (1 + (kd / doses) ** hill_slope)


def _error_model(mu):
    """KINOMEscan error model (see General Introduction.pptx)."""
    return pymc.math.maximum(0.15 * mu - 0.043, 1)


def _create_model():
    """Return a PyMC model with mutable data containers for doses/responses."""
    with pm.Model() as model:
        kd_log = pm.Normal("kd_log", mu=np.log(1e-6), sigma=3)
        kd = pm.Deterministic("kd", pm.math.exp(kd_log))

        hill_slope_log = pm.Normal("hill_slope_log", mu=0, sigma=0.5)
        hill_slope = pm.Deterministic("hill_slope", pm.math.exp(hill_slope_log))

        doses_data = pm.Data("doses_data", np.array([1e-9]))
        responses_data = pm.Data("responses_data", np.array([0.0]))

        mu = _model_predict(kd, hill_slope, doses_data)
        sigma = _error_model(mu)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=responses_data)
    return model


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _fit_one(
    drug_id, target, target_data, compiled_sampler, draws, tune, chains, seed=42
):
    """Fit a single compound/target pair. Returns a result dict."""
    doses = np.array(target_data["doses"]) * 1e-9
    responses = np.array(target_data["responses"])

    trace = nutpie.sample(
        compiled_sampler.with_data(doses_data=doses, responses_data=responses),
        draws=draws,
        tune=tune,
        chains=chains,
        cores=1,
        progress_bar=False,
        seed=seed,
    )

    kd_samples = trace.posterior["kd"].values.reshape(-1)
    hs_samples = trace.posterior["hill_slope"].values.reshape(-1)

    summary = pm.summary(trace)
    try:
        summary_json = summary.to_json(orient="table")
    except Exception:
        summary_json = json.dumps(summary.to_dict())

    return {
        "compound_id": drug_id,
        "target": target,
        "kd": float(np.mean(kd_samples)),
        "kd_std": float(np.std(kd_samples)),
        "hill_slope": float(np.mean(hs_samples)),
        "hill_slope_std": float(np.std(hs_samples)),
        "status": "ok",
        "summary_json": summary_json,
    }


def _error_record(drug_id, target, exc):
    return {
        "compound_id": drug_id,
        "target": target,
        "kd": None,
        "kd_std": None,
        "hill_slope": None,
        "hill_slope_std": None,
        "status": f"error: {exc}",
        "summary_json": None,
    }


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------


def _get_db(db_path):
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fits (
            compound_id  TEXT NOT NULL,
            target       TEXT NOT NULL,
            kd           REAL,
            kd_std       REAL,
            hill_slope   REAL,
            hill_slope_std REAL,
            status       TEXT,
            summary_json TEXT,
            updated_at   INTEGER NOT NULL,
            PRIMARY KEY (compound_id, target)
        )
        """
    )
    return conn


def _existing_pairs(conn):
    return set(conn.execute("SELECT compound_id, target FROM fits").fetchall())


def _upsert(conn, r):
    conn.execute(
        """
        INSERT INTO fits (compound_id, target, kd, kd_std,
                          hill_slope, hill_slope_std,
                          status, summary_json, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(compound_id, target) DO UPDATE SET
            kd=excluded.kd, kd_std=excluded.kd_std,
            hill_slope=excluded.hill_slope, hill_slope_std=excluded.hill_slope_std,
            status=excluded.status, summary_json=excluded.summary_json,
            updated_at=excluded.updated_at
        """,
        (
            r["compound_id"],
            r["target"],
            r.get("kd"),
            r.get("kd_std"),
            r.get("hill_slope"),
            r.get("hill_slope_std"),
            r.get("status", "ok"),
            r.get("summary_json"),
            int(time.time()),
        ),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

REQUIRED_COLS = [
    "hmsl_id",
    "DiscoveRx Gene Symbol",
    "Compound Concentration (nM)",
    "Percent Control",
]


def _load_data(csv_path):
    """Read the single-dose CSV and group by compound / target."""
    df = pd.read_csv(csv_path)
    return _load_data_from_df(df)


def _load_data_from_df(df):
    """Group a validated DataFrame by compound / target."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise click.ClickException(f"Missing columns: {', '.join(missing)}")

    data = {}
    for _, row in df.iterrows():
        cid = str(row["hmsl_id"])
        tgt = str(row["DiscoveRx Gene Symbol"])
        data.setdefault(cid, {}).setdefault(tgt, {"doses": [], "responses": []})
        data[cid][tgt]["doses"].append(float(row["Compound Concentration (nM)"]))
        data[cid][tgt]["responses"].append(float(row["Percent Control"]))
    return data


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


def _worker(work_q, result_q, cfg):
    """Long-lived worker: compiles the model once, processes items from the queue."""
    try:
        compiled = nutpie.compile_pymc_model(_create_model(), backend=cfg["backend"])
    except Exception as exc:
        result_q.put(_error_record(None, None, f"worker init: {exc}"))
        return

    while True:
        try:
            item = work_q.get(timeout=2)
        except Exception:
            continue
        if item is None:  # poison pill
            break

        drug_id, target, target_data = item
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(
            devnull
        ), contextlib.redirect_stderr(devnull):
            try:
                record = _fit_one(
                    drug_id,
                    target,
                    target_data,
                    compiled,
                    cfg["draws"],
                    cfg["tune"],
                    cfg["chains"],
                )
            except Exception as exc:
                record = _error_record(drug_id, target, exc)
        result_q.put(record)


def _run_parallel(items, conn, workers, commit_every, cfg):
    work_q, result_q = Queue(), Queue()
    n_procs = min(mp.cpu_count(), workers)

    procs = []
    for _ in range(n_procs):
        p = Process(target=_worker, args=(work_q, result_q, cfg))
        p.start()
        procs.append(p)

    for item in items:
        work_q.put(item)
    for _ in range(n_procs):
        work_q.put(None)

    total = len(items)
    done = errors = writes = 0
    with tqdm.tqdm(total=total, desc="Fitting") as pbar:
        while done < total:
            rec = result_q.get(timeout=600)
            if rec["compound_id"] is None:
                click.echo(rec["status"], err=True)
                continue
            _upsert(conn, rec)
            writes += 1
            done += 1
            if str(rec.get("status", "")).startswith("error"):
                errors += 1
            if writes % commit_every == 0:
                conn.commit()
            pbar.update(1)
        conn.commit()

    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join()

    return done, errors


def _run_serial(items, conn, commit_every, cfg):
    compiled = nutpie.compile_pymc_model(_create_model(), backend=cfg["backend"])
    done = errors = writes = 0

    for drug_id, target, target_data in tqdm.tqdm(items, desc="Fitting"):
        try:
            rec = _fit_one(
                drug_id,
                target,
                target_data,
                compiled,
                cfg["draws"],
                cfg["tune"],
                cfg["chains"],
            )
        except Exception as exc:
            rec = _error_record(drug_id, target, exc)
            errors += 1
        _upsert(conn, rec)
        writes += 1
        done += 1
        if writes % commit_every == 0:
            conn.commit()

    conn.commit()
    return done, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def fit_kd_from_dataframe(
    df,
    db_path,
    skip_existing=True,
    parallel=True,
    workers=8,
    draws=2000,
    tune=1000,
    chains=4,
    commit_every=100,
    backend="numba",
    cxx=None,
):
    """Fit Kd and Hill slope for all compound/target pairs in a DataFrame.

    DataFrame must contain columns: hmsl_id, DiscoveRx Gene Symbol,
    Compound Concentration (nM), Percent Control.
    """
    if cxx:
        pytensor.config.cxx = cxx

    data = _load_data_from_df(df)

    with contextlib.closing(_get_db(db_path)) as conn:
        existing = _existing_pairs(conn) if skip_existing else set()
        click.echo(f"Existing pairs in DB: {len(existing)}")

        items = [
            (cid, tgt, tdata)
            for cid, cdata in data.items()
            for tgt, tdata in cdata.items()
            if (cid, tgt) not in existing
        ]
        click.echo(f"Pairs to fit: {len(items)}")
        if not items:
            return 0, 0, None

        cfg = {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "backend": backend.lower(),
        }

        if parallel:
            done, errs = _run_parallel(items, conn, workers, commit_every, cfg)
        else:
            done, errs = _run_serial(items, conn, commit_every, cfg)

        click.echo(f"Done: {done}/{len(items)} fitted, {errs} errors")

        df = pd.read_sql_query("SELECT * FROM fits", conn)
        return done, errs, df


@click.command(context_settings={"show_default": True})
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False))
@click.argument("db_path", type=click.Path(dir_okay=False))
@click.option(
    "--csv-output",
    type=click.Path(dir_okay=False),
    help="CSV output path for results.",
)
@click.option(
    "--skip-existing/--recompute",
    default=True,
    help="Skip compound/target pairs already in the DB.",
)
@click.option(
    "--parallel/--no-parallel", default=True, help="Use multiprocessing workers."
)
@click.option("--workers", default=8, type=int, help="Max worker processes.")
@click.option("--draws", default=2000, type=int, help="Posterior draws per chain.")
@click.option("--tune", default=1000, type=int, help="Tuning steps per chain.")
@click.option("--chains", default=4, type=int, help="MCMC chains.")
@click.option("--commit-every", default=100, type=int, help="DB commit interval.")
@click.option(
    "--backend",
    default="numba",
    type=click.Choice(["numba", "jax"], case_sensitive=False),
    help="nutpie compilation backend.",
)
@click.option(
    "--cxx",
    default=None,
    type=str,
    help="C++ compiler path for pytensor (e.g. /usr/bin/g++).",
)
def main(
    input_csv,
    db_path,
    csv_output,
    skip_existing,
    parallel,
    workers,
    draws,
    tune,
    chains,
    commit_every,
    backend,
    cxx,
):
    """Fit Kd and Hill slope for every compound/target pair in INPUT_CSV
    and store them in DB_PATH.

    INPUT_CSV must contain columns: hmsl_id, DiscoveRx Gene Symbol,
    Compound Concentration (nM), Percent Control.
    """
    df = pd.read_csv(input_csv)
    _, _, df = fit_kd_from_dataframe(
        df=df,
        db_path=db_path,
        skip_existing=skip_existing,
        parallel=parallel,
        workers=workers,
        draws=draws,
        tune=tune,
        chains=chains,
        commit_every=commit_every,
        backend=backend,
        cxx=cxx,
    )

    if csv_output and df is not None:
        df.to_csv(csv_output, index=False)
        click.echo(f"Results written to {csv_output}")


if __name__ == "__main__":
    main()
