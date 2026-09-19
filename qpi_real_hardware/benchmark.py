from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.decomposition import PCA

BACKENDS = ["ibm_fez", "ibm_kingston", "ibm_marrakesh", "ibm_torino"]
PROPERTIES = ["T1", "T2", "readout_error", "sx_error"]
BUDGETS = [4, 8, 16, 32, 64]
COMMON_QUBITS = 133
MAX_EVAL_SNAPSHOTS = 60
MASK_REPEATS = 5
PCA_RANK = 12
PCA_OBS_NOISE = 0.50
SEED = 20260919


def log_transform(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(x, dtype=float), 1e-12, None))


def load_data() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id="phanerozoic/qiskit-calibration-drift",
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    cols = [
        "backend", "property_family", "qubit_a", "qubit_b", "value",
        "is_failure_ceiling", "is_new_measurement", "observed_time",
        "calibrated_time", "scope",
    ]
    # Push simple predicates into the Parquet scan: the public dataset contains
    # many heartbeat replicas, while QPI-0E is about actual calibration updates.
    df = pd.read_parquet(
        path,
        columns=cols,
        filters=[
            ("backend", "in", BACKENDS),
            ("property_family", "in", PROPERTIES),
            ("is_new_measurement", "=", True),
            ("is_failure_ceiling", "=", False),
        ],
    )
    df = df[df["qubit_b"].isna()].copy()
    df = df[df["qubit_a"].notna()].copy()
    df["qubit_a"] = df["qubit_a"].astype(int)
    df = df[(df["qubit_a"] >= 0) & (df["qubit_a"] < COMMON_QUBITS)].copy()
    df = df[~df["is_failure_ceiling"].fillna(False)].copy()
    df = df[np.isfinite(df["value"].astype(float)) & (df["value"].astype(float) > 0)].copy()
    df["observed_time"] = pd.to_datetime(df["observed_time"], utc=True)
    df["day"] = df["observed_time"].dt.floor("D")
    return df


def build_state_history(df: pd.DataFrame, backend: str, prop: str) -> pd.DataFrame:
    """Reconstruct the reported per-qubit state on each actual update day.

    The dataset stores deduplicated calibration events, not full snapshots.
    We therefore forward-fill the latest published value for every qubit and
    retain only days on which this property had at least one new measurement.
    """
    sub = df[(df.backend == backend) & (df.property_family == prop)].copy()
    if sub.empty:
        return pd.DataFrame()

    # If duplicate rows for a qubit/day exist, keep the last one observed that day.
    sub = sub.sort_values(["day", "observed_time"])
    daily_updates = (
        sub.groupby(["day", "qubit_a"], as_index=False)
           .tail(1)
           [["day", "qubit_a", "value"]]
    )
    update_days = pd.DatetimeIndex(sorted(daily_updates["day"].unique()))

    wide_updates = daily_updates.pivot(index="day", columns="qubit_a", values="value")
    all_days = pd.date_range(update_days.min(), update_days.max(), freq="D")
    wide = wide_updates.reindex(all_days).sort_index().ffill()
    wide = wide.reindex(columns=np.arange(COMMON_QUBITS))

    # Keep actual update days only and require nearly complete state.
    wide = wide.loc[wide.index.intersection(update_days)]
    coverage = wide.notna().mean(axis=1)
    wide = wide.loc[coverage >= 0.97].copy()
    if wide.empty:
        return wide

    # Fill the very small residual missing set without target leakage across time:
    # first with each column's past median, then the row median as a last resort.
    for col in wide.columns:
        vals = wide[col]
        if vals.isna().any():
            med = vals.expanding(min_periods=1).median()
            wide[col] = vals.fillna(med)
    row_med = wide.median(axis=1)
    for col in wide.columns:
        wide[col] = wide[col].fillna(row_med)

    # Remove exact consecutive replicas after forward filling.
    changed = wide.ne(wide.shift()).any(axis=1)
    wide = wide.loc[changed].copy()
    return wide


def source_training_matrix(
    histories: Dict[Tuple[str, str], pd.DataFrame],
    heldout: str,
    prop: str,
    cutoff: pd.Timestamp,
) -> np.ndarray:
    mats = []
    for b in BACKENDS:
        if b == heldout:
            continue
        h = histories.get((b, prop))
        if h is None or h.empty:
            continue
        hh = h[h.index <= cutoff]
        if len(hh):
            mats.append(log_transform(hh.to_numpy(dtype=float)))
    if not mats:
        return np.empty((0, COMMON_QUBITS))
    return np.concatenate(mats, axis=0)


def fit_population_model(Xlog: np.ndarray) -> Dict[str, np.ndarray | PCA]:
    mu = np.mean(Xlog, axis=0)
    sd = np.std(Xlog, axis=0, ddof=1)
    finite_sd = sd[np.isfinite(sd) & (sd > 1e-6)]
    floor = max(0.05, float(np.quantile(finite_sd, 0.15))) if len(finite_sd) else 0.05
    sd = np.where(np.isfinite(sd) & (sd > floor), sd, floor)

    Xz = (Xlog - mu) / sd
    rank = min(PCA_RANK, Xz.shape[0] - 1, Xz.shape[1])
    if rank < 1:
        raise ValueError("not enough source snapshots for PCA")
    pca = PCA(n_components=rank, svd_solver="full")
    pca.fit(Xz)
    return {"mu": mu, "sd": sd, "pca": pca}


def pca_reconstruct(model: Dict[str, np.ndarray | PCA], ylog: np.ndarray, obs: np.ndarray) -> np.ndarray:
    mu = model["mu"]
    sd = model["sd"]
    pca: PCA = model["pca"]  # type: ignore[assignment]
    ystd = (ylog - mu) / sd

    # PCA representation: ystd ≈ pca.mean_ + C @ z.
    C = pca.components_[:, obs].T
    rhs = ystd[obs] - pca.mean_[obs]
    ev = np.clip(pca.explained_variance_, 1e-6, None)
    prior_prec = np.diag(1.0 / ev)
    noise_var = PCA_OBS_NOISE ** 2
    A = (C.T @ C) / noise_var + prior_prec
    b = (C.T @ rhs) / noise_var
    z = np.linalg.solve(A, b)
    recon_std = pca.mean_ + pca.components_.T @ z
    return mu + sd * recon_std


def source_shift_reconstruct(model: Dict[str, np.ndarray | PCA], ylog: np.ndarray, obs: np.ndarray) -> np.ndarray:
    mu = model["mu"]
    shift = float(np.median(ylog[obs] - mu[obs]))
    return mu + shift


def target_only_linear(ylog: np.ndarray, obs: np.ndarray) -> np.ndarray:
    x = obs.astype(float)
    if len(obs) >= 2 and np.ptp(x) > 0:
        coef = np.polyfit(x, ylog[obs], deg=1)
        pred = np.polyval(coef, np.arange(len(ylog), dtype=float))
    else:
        pred = np.full_like(ylog, float(np.mean(ylog[obs])))
    # Guard pathological extrapolation from tiny K by clipping to observed range
    # with a generous margin.
    lo, hi = np.quantile(ylog[obs], [0.05, 0.95]) if len(obs) >= 4 else (np.min(ylog[obs]), np.max(ylog[obs]))
    margin = max(0.25, 0.5 * (hi - lo))
    return np.clip(pred, lo - margin, hi + margin)


def target_only_constant(ylog: np.ndarray, obs: np.ndarray) -> np.ndarray:
    return np.full_like(ylog, float(np.mean(ylog[obs])))


def standardized_rmse(
    pred_log: np.ndarray,
    true_log: np.ndarray,
    hidden: np.ndarray,
    source_sd: np.ndarray,
) -> float:
    e = (pred_log[hidden] - true_log[hidden]) / source_sd[hidden]
    return float(np.sqrt(np.mean(e * e)))


def median_relative_error(pred_log: np.ndarray, true_log: np.ndarray, hidden: np.ndarray) -> float:
    pred = np.exp(pred_log[hidden])
    true = np.exp(true_log[hidden])
    return float(np.median(np.abs(pred - true) / np.clip(np.abs(true), 1e-12, None)))


def bootstrap_paired(diff: np.ndarray, seed: int, nboot: int = 3000) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(diff) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(diff), size=(nboot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print("rows after filter:", len(df), flush=True)
    print("property counts:", df.property_family.value_counts().to_dict(), flush=True)
    print("backend counts:", df.backend.value_counts().to_dict(), flush=True)

    histories: Dict[Tuple[str, str], pd.DataFrame] = {}
    hist_meta = []
    for b in BACKENDS:
        for prop in PROPERTIES:
            h = build_state_history(df, b, prop)
            histories[(b, prop)] = h
            hist_meta.append({
                "backend": b, "property": prop, "snapshots": len(h),
                "start": str(h.index.min()) if len(h) else None,
                "end": str(h.index.max()) if len(h) else None,
            })
            print(f"history {b} {prop}: {len(h)}", flush=True)
    pd.DataFrame(hist_meta).to_csv(out_dir / "history_inventory.csv", index=False)

    rows: List[dict] = []
    paired_rows: List[dict] = []
    methods = ["target_constant", "target_linear", "source_mean", "source_shift", "pca_transfer"]

    for heldout in BACKENDS:
        for prop in PROPERTIES:
            target = histories[(heldout, prop)]
            if len(target) < 12:
                print(f"skip {heldout} {prop}: only {len(target)} snapshots", flush=True)
                continue

            # Strict temporal split on the held-out QPU. Source population model sees
            # only source-QPU snapshots available up to the target cutoff.
            cutoff_idx = max(1, int(np.floor(0.70 * len(target))) - 1)
            cutoff = target.index[cutoff_idx]
            target_eval = target[target.index > cutoff]
            if len(target_eval) > MAX_EVAL_SNAPSHOTS:
                take = np.linspace(0, len(target_eval) - 1, MAX_EVAL_SNAPSHOTS).round().astype(int)
                target_eval = target_eval.iloc[np.unique(take)]

            Xsrc = source_training_matrix(histories, heldout, prop, cutoff)
            if len(Xsrc) < 20:
                print(f"skip {heldout} {prop}: only {len(Xsrc)} source snapshots", flush=True)
                continue
            model = fit_population_model(Xsrc)
            mu = model["mu"]
            sd = model["sd"]

            print(
                f"LOQPO {heldout} {prop}: cutoff={cutoff.date()} "
                f"source={len(Xsrc)} eval={len(target_eval)}",
                flush=True,
            )

            for K in BUDGETS:
                if K >= COMMON_QUBITS:
                    continue
                for snap_i, (date, row) in enumerate(target_eval.iterrows()):
                    true_log = log_transform(row.to_numpy(dtype=float))
                    for rep in range(MASK_REPEATS):
                        rng = np.random.default_rng(
                            SEED
                            + BACKENDS.index(heldout) * 1_000_003
                            + PROPERTIES.index(prop) * 100_003
                            + K * 1009
                            + snap_i * 53
                            + rep
                        )
                        obs = np.sort(rng.choice(COMMON_QUBITS, size=K, replace=False))
                        hidden = np.setdiff1d(np.arange(COMMON_QUBITS), obs, assume_unique=True)

                        preds = {
                            "target_constant": target_only_constant(true_log, obs),
                            "target_linear": target_only_linear(true_log, obs),
                            "source_mean": np.asarray(mu),
                            "source_shift": source_shift_reconstruct(model, true_log, obs),
                            "pca_transfer": pca_reconstruct(model, true_log, obs),
                        }
                        for method, pred in preds.items():
                            rows.append({
                                "heldout": heldout,
                                "property": prop,
                                "date": str(date),
                                "K": K,
                                "repeat": rep,
                                "method": method,
                                "nrmse": standardized_rmse(pred, true_log, hidden, sd),
                                "median_relative_error": median_relative_error(pred, true_log, hidden),
                            })

    per = pd.DataFrame(rows)
    per.to_csv(out_dir / "per_snapshot.csv", index=False)

    # Aggregate by snapshot first so mask repeats do not inflate confidence.
    snap = (
        per.groupby(["heldout", "property", "date", "K", "method"], as_index=False)
           .agg(nrmse=("nrmse", "mean"), median_relative_error=("median_relative_error", "mean"))
    )
    snap.to_csv(out_dir / "per_snapshot_averaged.csv", index=False)

    summary_rows = []
    for (heldout, prop, K, method), g in snap.groupby(["heldout", "property", "K", "method"]):
        summary_rows.append({
            "heldout": heldout,
            "property": prop,
            "K": int(K),
            "method": method,
            "snapshots": len(g),
            "mean_nrmse": float(g.nrmse.mean()),
            "median_nrmse": float(g.nrmse.median()),
            "mean_median_relative_error": float(g.median_relative_error.mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    # Paired QPI-vs-target-only statistics.
    for heldout in sorted(snap.heldout.unique()):
        for prop in sorted(snap[snap.heldout == heldout].property.unique()):
            for K in BUDGETS:
                ss = snap[(snap.heldout == heldout) & (snap.property == prop) & (snap.K == K)]
                if ss.empty:
                    continue
                wide = ss.pivot(index="date", columns="method", values="nrmse").dropna()
                if "pca_transfer" not in wide or "target_linear" not in wide:
                    continue
                diff = (wide["pca_transfer"] - wide["target_linear"]).to_numpy()
                lo, hi = bootstrap_paired(
                    diff,
                    SEED + K * 43 + BACKENDS.index(heldout) * 701 + PROPERTIES.index(prop) * 101,
                )
                paired_rows.append({
                    "heldout": heldout,
                    "property": prop,
                    "K": K,
                    "snapshots": len(diff),
                    "target_linear_nrmse": float(wide["target_linear"].mean()),
                    "pca_transfer_nrmse": float(wide["pca_transfer"].mean()),
                    "relative_improvement": float(
                        1.0 - wide["pca_transfer"].mean() / wide["target_linear"].mean()
                    ),
                    "pca_win_rate": float(np.mean(diff < 0)),
                    "paired_mean_diff": float(np.mean(diff)),
                    "paired_ci95_low": lo,
                    "paired_ci95_high": hi,
                })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out_dir / "paired_stats.csv", index=False)

    # Overall macro-average: every heldout-QPU/property pair receives equal weight.
    macro = (
        summary.groupby(["heldout", "property", "K", "method"], as_index=False)
               .mean(numeric_only=True)
               .groupby(["K", "method"], as_index=False)
               .mean(numeric_only=True)
    )
    macro.to_csv(out_dir / "macro_summary.csv", index=False)

    # Human-readable tables.
    md = [
        "# QPI-0E real-hardware sparse-calibration transfer",
        "",
        "Dataset: `phanerozoic/qiskit-calibration-drift` (IBM Quantum backend.properties history).",
        "",
        "Protocol: strict leave-one-QPU-out + temporal source cutoff; reveal K of 133 common qubit calibration values on the held-out QPU and reconstruct the remaining values.",
        "",
        "Primary metric: RMSE in source-standardized log-parameter space on hidden qubits (lower is better).",
        "",
    ]
    for heldout in BACKENDS:
        md += [f"## {heldout}", ""]
        for prop in PROPERTIES:
            s = summary[(summary.heldout == heldout) & (summary.property == prop)]
            if s.empty:
                continue
            md += [f"### {prop}", "", "| K | target-linear | source-shift | PCA transfer |", "|---:|---:|---:|---:|"]
            for K in BUDGETS:
                x = s[s.K == K].set_index("method")
                if x.empty:
                    continue
                md.append(
                    f"| {K} | {x.loc['target_linear','mean_nrmse']:.4f} | "
                    f"{x.loc['source_shift','mean_nrmse']:.4f} | "
                    f"{x.loc['pca_transfer','mean_nrmse']:.4f} |"
                )
            md.append("")

    md += ["## Macro average across held-out QPU/property tasks", "", "| K | target-linear | source-shift | PCA transfer |", "|---:|---:|---:|---:|"]
    for K in BUDGETS:
        x = macro[macro.K == K].set_index("method")
        if x.empty:
            continue
        md.append(
            f"| {K} | {x.loc['target_linear','mean_nrmse']:.4f} | "
            f"{x.loc['source_shift','mean_nrmse']:.4f} | "
            f"{x.loc['pca_transfer','mean_nrmse']:.4f} |"
        )

    (out_dir / "RESULTS.md").write_text("\n".join(md))
    print("\n".join(md[-12:]), flush=True)

    metadata = {
        "seed": SEED,
        "backends": BACKENDS,
        "properties": PROPERTIES,
        "common_qubits": COMMON_QUBITS,
        "budgets": BUDGETS,
        "max_eval_snapshots": MAX_EVAL_SNAPSHOTS,
        "mask_repeats": MASK_REPEATS,
        "pca_rank": PCA_RANK,
        "pca_observation_noise": PCA_OBS_NOISE,
        "protocol_note": (
            "K counts revealed published calibration values, not raw hardware shots. "
            "This is a retrospective masked-measurement proxy."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results_real"))
    args = ap.parse_args()
    run(args.out)
