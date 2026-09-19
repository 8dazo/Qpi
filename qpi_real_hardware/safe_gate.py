from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from benchmark import (
    BACKENDS, PROPERTIES, BUDGETS, COMMON_QUBITS, MAX_EVAL_SNAPSHOTS,
    MASK_REPEATS, SEED, load_data, build_state_history, source_training_matrix,
    fit_population_model, log_transform, target_only_linear,
    source_shift_reconstruct, pca_reconstruct, standardized_rmse,
)

CANDIDATES = ["target_linear", "source_shift", "pca_transfer"]


def candidate_predictions(model, ylog: np.ndarray, obs: np.ndarray):
    return {
        "target_linear": target_only_linear(ylog, obs),
        "source_shift": source_shift_reconstruct(model, ylog, obs),
        "pca_transfer": pca_reconstruct(model, ylog, obs),
    }


def observed_cv_scores(model, ylog: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    """Cross-validate candidates using only revealed target calibration values."""
    sd = np.asarray(model["sd"])
    total = {n: 0.0 for n in CANDIDATES}
    count = {n: 0 for n in CANDIDATES}
    for val in np.array_split(obs, min(4, len(obs))):
        train = np.setdiff1d(obs, val, assume_unique=True)
        if len(train) < 2 or len(val) == 0:
            continue
        preds = candidate_predictions(model, ylog, train)
        for name, pred in preds.items():
            e = (pred[val] - ylog[val]) / sd[val]
            total[name] += float(np.sum(e * e))
            count[name] += len(val)
    return {n: total[n] / max(count[n], 1) for n in CANDIDATES}


def safe_predictions(model, ylog: np.ndarray, obs: np.ndarray):
    full = candidate_predictions(model, ylog, obs)
    scores = observed_cv_scores(model, ylog, obs)
    chosen = min(scores, key=scores.get)

    # Inverse predictive-MSE model averaging. Every weight comes only from
    # observed target qubits; hidden qubits never participate in selection.
    inv = np.array([1.0 / max(scores[n], 1e-4) for n in CANDIDATES])
    cap = np.quantile(inv, 0.95)
    inv = np.minimum(inv, cap)
    weights = inv / inv.sum()
    blend = sum(w * full[n] for w, n in zip(weights, CANDIDATES))
    return full, full[chosen], blend, chosen, scores, weights


def bootstrap_paired(diff: np.ndarray, seed: int, nboot: int = 3000) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(nboot, len(diff)))
    boot = diff[idx].mean(axis=1)
    return float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def run(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    df = load_data()
    histories = {}
    for b in BACKENDS:
        for prop in PROPERTIES:
            histories[(b, prop)] = build_state_history(df, b, prop)

    rows = []
    for heldout in BACKENDS:
        for prop in PROPERTIES:
            target = histories[(heldout, prop)]
            if len(target) < 12:
                continue
            cutoff_idx = max(1, int(np.floor(.70 * len(target))) - 1)
            cutoff = target.index[cutoff_idx]
            target_eval = target[target.index > cutoff]
            if len(target_eval) > MAX_EVAL_SNAPSHOTS:
                take = np.linspace(0, len(target_eval)-1, MAX_EVAL_SNAPSHOTS).round().astype(int)
                target_eval = target_eval.iloc[np.unique(take)]
            Xsrc = source_training_matrix(histories, heldout, prop, cutoff)
            if len(Xsrc) < 20:
                continue
            model = fit_population_model(Xsrc)
            print(f"safe {heldout} {prop}: source={len(Xsrc)} eval={len(target_eval)}", flush=True)

            for K in BUDGETS:
                for snap_i, (date, state) in enumerate(target_eval.iterrows()):
                    ylog = log_transform(state.to_numpy(float))
                    for rep in range(MASK_REPEATS):
                        rng = np.random.default_rng(
                            SEED + BACKENDS.index(heldout)*1_000_003
                            + PROPERTIES.index(prop)*100_003 + K*1009 + snap_i*53 + rep
                        )
                        obs = np.sort(rng.choice(COMMON_QUBITS, K, replace=False))
                        hidden = np.setdiff1d(np.arange(COMMON_QUBITS), obs, assume_unique=True)
                        base, selected, blend, choice, scores, weights = safe_predictions(model, ylog, obs)
                        preds = {**base, "cv_select": selected, "cv_blend": blend}
                        for method, pred in preds.items():
                            rows.append({
                                "heldout": heldout, "property": prop, "date": str(date),
                                "K": K, "repeat": rep, "method": method,
                                "nrmse": standardized_rmse(pred, ylog, hidden, np.asarray(model["sd"])),
                                "cv_choice": choice,
                                "w_target": weights[0], "w_shift": weights[1], "w_pca": weights[2],
                            })

    per = pd.DataFrame(rows)
    per.to_csv(out/"per_mask.csv", index=False)
    snap = per.groupby(["heldout","property","date","K","method"], as_index=False).nrmse.mean()
    snap.to_csv(out/"per_snapshot.csv", index=False)

    summary = snap.groupby(["heldout","property","K","method"], as_index=False).agg(
        snapshots=("date","nunique"), mean_nrmse=("nrmse","mean")
    )
    summary.to_csv(out/"summary.csv", index=False)

    macro = summary.groupby(["K","method"], as_index=False).mean(numeric_only=True)
    macro.to_csv(out/"macro_summary.csv", index=False)

    stats=[]
    for (heldout, prop, K), g in snap.groupby(["heldout","property","K"]):
        w=g.pivot(index="date",columns="method",values="nrmse").dropna()
        for m in ["source_shift","pca_transfer","cv_select","cv_blend"]:
            diff=(w[m]-w["target_linear"]).to_numpy()
            lo,hi=bootstrap_paired(diff, SEED+int(K)*31+len(stats))
            stats.append({
                "heldout":heldout,"property":prop,"K":int(K),"method":m,
                "target":float(w.target_linear.mean()),"method_nrmse":float(w[m].mean()),
                "relative_improvement":float(1-w[m].mean()/w.target_linear.mean()),
                "win_rate":float(np.mean(diff<0)),"ci95_low":lo,"ci95_high":hi,
            })
    pd.DataFrame(stats).to_csv(out/"paired_stats.csv",index=False)

    choices=(per[per.method=="cv_select"].groupby(["K","cv_choice"]).size().rename("count").reset_index())
    choices["fraction"] = choices["count"] / choices.groupby("K")["count"].transform("sum")
    choices.to_csv(out/"gate_choices.csv",index=False)

    mm=macro.pivot(index="K",columns="method",values="mean_nrmse")
    lines=["# QPI-0F target-evidence safety gate","", "Lower NRMSE is better.","",
           "| K | target | shift | PCA | CV select | CV blend |",
           "|---:|---:|---:|---:|---:|---:|"]
    for K,row in mm.iterrows():
        lines.append(f"| {K} | {row.target_linear:.4f} | {row.source_shift:.4f} | {row.pca_transfer:.4f} | {row.cv_select:.4f} | {row.cv_blend:.4f} |")
    (out/"RESULTS.md").write_text("\n".join(lines))
    (out/"metadata.json").write_text(json.dumps({
        "selection_rule":"4-fold predictive CV over only revealed target qubits",
        "hidden_qubits_used_for_selection":False,
        "note":"K counts published calibration values, not raw hardware shots",
    },indent=2))
    print("\n".join(lines),flush=True)


if __name__ == "__main__":
    run(Path("qpi_real_hardware/results_safe"))
