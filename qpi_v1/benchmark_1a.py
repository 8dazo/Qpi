from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from qpi_real_hardware.benchmark import (
    BACKENDS, PROPERTIES, BUDGETS, COMMON_QUBITS, MAX_EVAL_SNAPSHOTS,
    MASK_REPEATS, SEED, load_data, build_state_history, source_training_matrix,
    fit_population_model, log_transform, target_only_linear, source_shift_reconstruct,
    pca_reconstruct, standardized_rmse,
)

# QPI-1A uses a graph smoother over qubit topology. The public calibration
# parquet does not expose a backend coupling map, so this first graph baseline
# uses the processor's indexed 1-D adjacency as a deliberately weak topology
# prior. It must beat PCA before we justify fetching richer topology metadata.
LAMBDA_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
T2_TRANSFER_ENABLED = False


def chain_laplacian(n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    i = np.arange(n - 1)
    A[i, i + 1] = 1.0
    A[i + 1, i] = 1.0
    D = np.diag(A.sum(axis=1))
    return D - A


LAPLACIAN = chain_laplacian(COMMON_QUBITS)


def graph_reconstruct(mu: np.ndarray, sd: np.ndarray, ylog: np.ndarray, obs: np.ndarray, lam: float) -> np.ndarray:
    """MAP reconstruction: source prior + graph smoothness + exact revealed values."""
    n = len(mu)
    prior_prec = np.diag(1.0 / np.square(np.clip(sd, 0.05, None)))
    H = np.zeros((len(obs), n))
    H[np.arange(len(obs)), obs] = 1.0
    obs_var = 0.05 ** 2
    A = prior_prec + lam * LAPLACIAN + (H.T @ H) / obs_var
    b = prior_prec @ mu + (H.T @ ylog[obs]) / obs_var
    return np.linalg.solve(A, b)


def choose_lambda(mu: np.ndarray, sd: np.ndarray, ylog: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    """LOO/CV on revealed target values only."""
    if len(obs) < 3:
        return 1.0, np.inf
    folds = np.array_split(obs, min(4, len(obs)))
    scores = []
    for lam in LAMBDA_GRID:
        se, count = 0.0, 0
        for val in folds:
            train = np.setdiff1d(obs, val, assume_unique=True)
            if len(train) < 2 or len(val) == 0:
                continue
            pred = graph_reconstruct(mu, sd, ylog, train, lam)
            e = (pred[val] - ylog[val]) / sd[val]
            se += float(np.sum(e * e)); count += len(val)
        scores.append(se / max(count, 1))
    j = int(np.argmin(scores))
    return float(LAMBDA_GRID[j]), float(scores[j])


def target_cv(ylog: np.ndarray, obs: np.ndarray, sd: np.ndarray) -> float:
    folds = np.array_split(obs, min(4, len(obs)))
    se, count = 0.0, 0
    for val in folds:
        train = np.setdiff1d(obs, val, assume_unique=True)
        if len(train) < 2 or len(val) == 0:
            continue
        pred = target_only_linear(ylog, train)
        e = (pred[val] - ylog[val]) / sd[val]
        se += float(np.sum(e * e)); count += len(val)
    return se / max(count, 1)


def gated_blend(target: np.ndarray, transfer: np.ndarray, k: int, target_err: float, transfer_err: float, enabled: bool) -> tuple[np.ndarray, float]:
    if not enabled:
        return target, 0.0
    advantage = np.log(max(target_err, 1e-8) / max(transfer_err, 1e-8))
    trust = 1.0 / (1.0 + np.exp(-advantage))
    # QPI-0F showed persistent high-K negative transfer. QPI-1A freezes a
    # conservative prior-decay baseline before learning a gate.
    alpha = float(trust * np.exp(-k / 12.0))
    return (1 - alpha) * target + alpha * transfer, alpha


def bootstrap(diff: np.ndarray, seed: int, nboot: int = 3000):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(nboot, len(diff)))
    x = diff[idx].mean(axis=1)
    return float(np.quantile(x, .025)), float(np.quantile(x, .975))


def run(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    df = load_data()
    histories = {(b,p): build_state_history(df,b,p) for b in BACKENDS for p in PROPERTIES}
    rows=[]

    for heldout in BACKENDS:
        for prop in PROPERTIES:
            target=histories[(heldout,prop)]
            if len(target)<12: continue
            cutoff_idx=max(1,int(np.floor(.70*len(target)))-1)
            cutoff=target.index[cutoff_idx]
            ev=target[target.index>cutoff]
            if len(ev)>MAX_EVAL_SNAPSHOTS:
                take=np.linspace(0,len(ev)-1,MAX_EVAL_SNAPSHOTS).round().astype(int)
                ev=ev.iloc[np.unique(take)]
            X=source_training_matrix(histories,heldout,prop,cutoff)
            if len(X)<20: continue
            model=fit_population_model(X); mu=np.asarray(model["mu"]); sd=np.asarray(model["sd"])
            print(f"QPI-1A {heldout} {prop}: source={len(X)} eval={len(ev)}",flush=True)

            for K in BUDGETS:
                for si,(date,state) in enumerate(ev.iterrows()):
                    y=log_transform(state.to_numpy(float))
                    for rep in range(MASK_REPEATS):
                        rng=np.random.default_rng(SEED+BACKENDS.index(heldout)*1000003+PROPERTIES.index(prop)*100003+K*1009+si*53+rep)
                        obs=np.sort(rng.choice(COMMON_QUBITS,K,replace=False))
                        hidden=np.setdiff1d(np.arange(COMMON_QUBITS),obs,assume_unique=True)
                        target_pred=target_only_linear(y,obs)
                        shift=source_shift_reconstruct(model,y,obs)
                        pca=pca_reconstruct(model,y,obs)
                        lam,gcv=choose_lambda(mu,sd,y,obs)
                        graph=graph_reconstruct(mu,sd,y,obs,lam)
                        tcv=target_cv(y,obs,sd)
                        enabled=(prop!="T2" or T2_TRANSFER_ENABLED)
                        gated,alpha=gated_blend(target_pred,graph,K,tcv,gcv,enabled)
                        for method,pred in {"target_linear":target_pred,"source_shift":shift,"pca_transfer":pca,"graph_transfer":graph,"graph_gated":gated}.items():
                            rows.append({"heldout":heldout,"property":prop,"date":str(date),"K":K,"repeat":rep,"method":method,
                                         "nrmse":standardized_rmse(pred,y,hidden,sd),"lambda":lam,"alpha":alpha})

    per=pd.DataFrame(rows); per.to_csv(out/"per_mask.csv",index=False)
    snap=per.groupby(["heldout","property","date","K","method"],as_index=False).agg(nrmse=("nrmse","mean"),alpha=("alpha","mean"))
    snap.to_csv(out/"per_snapshot.csv",index=False)
    summary=snap.groupby(["heldout","property","K","method"],as_index=False).agg(snapshots=("date","nunique"),mean_nrmse=("nrmse","mean"),mean_alpha=("alpha","mean"))
    summary.to_csv(out/"summary.csv",index=False)
    macro=summary.groupby(["K","method"],as_index=False).mean(numeric_only=True)
    macro.to_csv(out/"macro_summary.csv",index=False)

    stats=[]
    for (h,p,k),g in snap.groupby(["heldout","property","K"]):
        w=g.pivot(index="date",columns="method",values="nrmse").dropna()
        for m in ["source_shift","pca_transfer","graph_transfer","graph_gated"]:
            d=(w[m]-w["target_linear"]).to_numpy(); lo,hi=bootstrap(d,SEED+int(k)*31+len(stats))
            stats.append({"heldout":h,"property":p,"K":int(k),"method":m,"target":float(w.target_linear.mean()),
                          "method_nrmse":float(w[m].mean()),"relative_improvement":float(1-w[m].mean()/w.target_linear.mean()),
                          "win_rate":float(np.mean(d<0)),"ci95_low":lo,"ci95_high":hi})
    pd.DataFrame(stats).to_csv(out/"paired_stats.csv",index=False)

    mm=macro.pivot(index="K",columns="method",values="mean_nrmse")
    lines=["# QPI-1A graph conditional-transfer benchmark","",
           "Primary metric: hidden-qubit source-standardized log-space RMSE; lower is better.","",
           "| K | target | shift | PCA | graph | graph+gate |","|---:|---:|---:|---:|---:|---:|"]
    for k,r in mm.iterrows():
        lines.append(f"| {k} | {r.target_linear:.4f} | {r.source_shift:.4f} | {r.pca_transfer:.4f} | {r.graph_transfer:.4f} | {r.graph_gated:.4f} |")
    lines += ["","T2 transfer is disabled in graph+gate by default because QPI-0E showed systematic negative transfer.",
              "K is a masked published-calibration-value budget, not raw QPU shots."]
    (out/"RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=Path("qpi_v1/results_1a"))
    run(ap.parse_args().out)
