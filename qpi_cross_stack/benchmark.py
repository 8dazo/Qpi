from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

OMEGA0 = 2 * np.pi * 5.0  # rad / us, nominal resonant Rabi rate
ALPHA = 2 * np.pi * (-200.0)  # rad / us, transmon anharmonicity for Qiskit family
EPS = 1e-9
PARAM_NAMES = ["delta", "log_t1", "log_t2", "log_drive", "raw_readout"]


@dataclass
class Device:
    delta: float
    t1: float
    tphi: float
    drive_scale: float
    readout: float
    qs_sigma: float = 0.0

    @property
    def t2(self) -> float:
        return 1.0 / (1.0 / (2.0 * self.t1) + 1.0 / self.tphi)


@dataclass(frozen=True)
class Setting:
    kind: str
    t: float


@dataclass
class DeviceData:
    backend: str
    device_id: int
    support_settings: List[Setting]
    support_probs: np.ndarray
    eval_settings: List[Setting]
    eval_probs: np.ndarray


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def readout_from_raw(raw: float) -> float:
    return float(0.12 * sigmoid(raw))


def raw_from_readout(e: float) -> float:
    x = np.clip(e / 0.12, 1e-5, 1 - 1e-5)
    return float(np.log(x / (1 - x)))


def apply_readout(p: np.ndarray, e: float) -> np.ndarray:
    return np.clip(e + (1.0 - 2.0 * e) * p, 1e-6, 1 - 1e-6)


def make_protocol() -> Tuple[List[Setting], List[Setting]]:
    support: List[Setting] = []
    support += [Setting("t1", float(t)) for t in np.geomspace(0.5, 80.0, 18)]
    support += [Setting("ramsey", float(t)) for t in np.linspace(0.10, 12.0, 22)]
    support += [Setting("rabi", float(t)) for t in np.linspace(0.005, 0.35, 20)]

    evals: List[Setting] = []
    evals += [Setting("t1", float(t)) for t in np.geomspace(0.65, 72.0, 24)]
    evals += [Setting("ramsey", float(t)) for t in np.linspace(0.15, 11.75, 30)]
    evals += [Setting("rabi", float(t)) for t in np.linspace(0.009, 0.343, 30)]
    return support, evals


def family_device(rng: np.random.Generator, backend: str) -> Device:
    if backend == "qutip":
        mu_delta_mhz, sd_delta_mhz = 0.060, 0.035
        t1_mu, tphi_mu = 45.0, 85.0
        drive_mu, drive_sd = 1.00, 0.045
        ro_mu, ro_sd = 0.025, 0.007
        qs_sigma = 0.0
    elif backend == "qiskit":
        mu_delta_mhz, sd_delta_mhz = 0.080, 0.040
        t1_mu, tphi_mu = 38.0, 65.0
        drive_mu, drive_sd = 0.965, 0.050
        ro_mu, ro_sd = 0.034, 0.008
        qs_sigma = 0.0
    elif backend == "scipy_qs":
        mu_delta_mhz, sd_delta_mhz = 0.105, 0.050
        t1_mu, tphi_mu = 52.0, 115.0
        drive_mu, drive_sd = 1.035, 0.055
        ro_mu, ro_sd = 0.030, 0.008
        qs_sigma = 2 * np.pi * max(0.005, rng.normal(0.022, 0.006))
    else:
        raise ValueError(backend)

    delta = 2 * np.pi * rng.normal(mu_delta_mhz, sd_delta_mhz)
    t1 = float(np.exp(rng.normal(np.log(t1_mu), 0.20)))
    tphi = float(np.exp(rng.normal(np.log(tphi_mu), 0.23)))
    drive = float(np.clip(rng.normal(drive_mu, drive_sd), 0.78, 1.22))
    ro = float(np.clip(rng.normal(ro_mu, ro_sd), 0.008, 0.070))
    return Device(delta=delta, t1=t1, tphi=tphi, drive_scale=drive, readout=ro, qs_sigma=qs_sigma)


def _group_times(settings: List[Setting]) -> Dict[str, np.ndarray]:
    return {
        kind: np.array([s.t for s in settings if s.kind == kind], dtype=float)
        for kind in ["t1", "ramsey", "rabi"]
    }


def _reassemble(settings: List[Setting], values: Dict[str, np.ndarray]) -> np.ndarray:
    cursor = {k: 0 for k in values}
    out = []
    for s in settings:
        out.append(values[s.kind][cursor[s.kind]])
        cursor[s.kind] += 1
    return np.asarray(out, dtype=float)


def simulate_qutip(device: Device, settings: List[Setting]) -> np.ndarray:
    import qutip as qt

    times = _group_times(settings)
    sx, sz = qt.sigmax(), qt.sigmaz()
    sm = qt.Qobj([[0.0, 1.0], [0.0, 0.0]])  # |0><1| in the chosen basis
    p1 = qt.basis(2, 1) * qt.basis(2, 1).dag()
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    pplus = plus * plus.dag()
    rho1 = p1
    rhop = pplus
    rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()

    gamma1 = 1.0 / device.t1
    gphi = 1.0 / device.tphi
    c_ops = [np.sqrt(gamma1) * sm, np.sqrt(gphi / 2.0) * sz]

    vals: Dict[str, np.ndarray] = {}
    for kind, ts in times.items():
        if len(ts) == 0:
            vals[kind] = np.array([])
            continue
        order = np.argsort(ts)
        t_sorted = ts[order]
        tlist = np.r_[0.0, t_sorted]
        if kind == "t1":
            H = 0.5 * device.delta * sz
            res = qt.mesolve(H, rho1, tlist, c_ops, e_ops=[p1], options={"nsteps": 10000})
            y = np.asarray(res.expect[0][1:], dtype=float)
        elif kind == "ramsey":
            H = 0.5 * device.delta * sz
            res = qt.mesolve(H, rhop, tlist, c_ops, e_ops=[pplus], options={"nsteps": 10000})
            y = np.asarray(res.expect[0][1:], dtype=float)
        else:
            H = 0.5 * device.delta * sz + 0.5 * OMEGA0 * device.drive_scale * sx
            res = qt.mesolve(H, rho0, tlist, c_ops, e_ops=[p1], options={"nsteps": 10000})
            y = np.asarray(res.expect[0][1:], dtype=float)
        unsorted = np.empty_like(y)
        unsorted[order] = y
        vals[kind] = unsorted
    return apply_readout(_reassemble(settings, vals), device.readout)


def _qiskit_solve(H: np.ndarray, c_ops: List[np.ndarray], rho0: np.ndarray, ts: np.ndarray, op: np.ndarray) -> np.ndarray:
    from qiskit_dynamics import Solver

    if len(ts) == 0:
        return np.array([])
    order = np.argsort(ts)
    t_sorted = ts[order]
    solver = Solver(static_hamiltonian=H, static_dissipators=c_ops, vectorized=False)
    res = solver.solve(
        t_span=[0.0, float(t_sorted[-1])],
        y0=rho0,
        t_eval=np.r_[0.0, t_sorted],
        atol=1e-9,
        rtol=1e-9,
        method="DOP853",
    )
    arr = np.asarray(res.y)
    y = np.array([np.real(np.trace(op @ np.asarray(rho))) for rho in arr[1:]], dtype=float)
    unsorted = np.empty_like(y)
    unsorted[order] = y
    return unsorted


def simulate_qiskit(device: Device, settings: List[Setting]) -> np.ndarray:
    dim = 3
    a = np.diag(np.sqrt(np.arange(1, dim)), 1).astype(complex)
    adag = a.conj().T
    N = np.diag(np.arange(dim)).astype(complex)
    I = np.eye(dim, dtype=complex)
    H0 = device.delta * N + 0.5 * ALPHA * (N @ (N - I))
    Hdrive = H0 + 0.5 * OMEGA0 * device.drive_scale * (a + adag)

    gamma1 = 1.0 / device.t1
    gphi = 1.0 / device.tphi
    c_ops = [np.sqrt(gamma1) * a, np.sqrt(2.0 * gphi) * N]

    ket0 = np.array([1.0, 0.0, 0.0], dtype=complex)
    ket1 = np.array([0.0, 1.0, 0.0], dtype=complex)
    ketp = (ket0 + ket1) / np.sqrt(2.0)
    rho0 = np.outer(ket0, ket0.conj())
    rho1 = np.outer(ket1, ket1.conj())
    rhop = np.outer(ketp, ketp.conj())
    P1 = rho1.copy()
    Pplus = rhop.copy()

    times = _group_times(settings)
    vals = {
        "t1": _qiskit_solve(H0, c_ops, rho1, times["t1"], P1),
        "ramsey": _qiskit_solve(H0, c_ops, rhop, times["ramsey"], Pplus),
        "rabi": _qiskit_solve(Hdrive, c_ops, rho0, times["rabi"], P1),
    }
    return apply_readout(_reassemble(settings, vals), device.readout)


def liouvillian(H: np.ndarray, c_ops: Iterable[np.ndarray]) -> np.ndarray:
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for c in c_ops:
        cd_c = c.conj().T @ c
        L += np.kron(c.conj(), c)
        L -= 0.5 * np.kron(I, cd_c)
        L -= 0.5 * np.kron(cd_c.T, I)
    return L


def evolve_expm(H: np.ndarray, c_ops: List[np.ndarray], rho0: np.ndarray, ts: np.ndarray, op: np.ndarray) -> np.ndarray:
    L = liouvillian(H, c_ops)
    v0 = rho0.reshape(-1, order="F")
    out = []
    for t in ts:
        vt = expm(L * float(t)) @ v0
        rho = vt.reshape(rho0.shape, order="F")
        out.append(float(np.real(np.trace(op @ rho))))
    return np.asarray(out)


def simulate_scipy_qs(device: Device, settings: List[Setting]) -> np.ndarray:
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    ket0 = np.array([1.0, 0.0], dtype=complex)
    ket1 = np.array([0.0, 1.0], dtype=complex)
    ketp = (ket0 + ket1) / np.sqrt(2.0)
    rho0, rho1, rhop = [np.outer(k, k.conj()) for k in (ket0, ket1, ketp)]
    P1, Pplus = rho1, rhop
    g1 = 1.0 / device.t1
    gphi = 1.0 / device.tphi
    c_ops = [np.sqrt(g1) * sm, np.sqrt(gphi / 2.0) * sz]
    times = _group_times(settings)

    nodes, weights = np.polynomial.hermite.hermgauss(7)
    weights = weights / np.sqrt(np.pi)
    offsets = np.sqrt(2.0) * device.qs_sigma * nodes

    vals: Dict[str, np.ndarray] = {}
    vals["t1"] = evolve_expm(0.5 * device.delta * sz, c_ops, rho1, times["t1"], P1)
    for kind, rho_init, op in [("ramsey", rhop, Pplus), ("rabi", rho0, P1)]:
        accum = np.zeros(len(times[kind]), dtype=float)
        for off, w in zip(offsets, weights):
            H = 0.5 * (device.delta + off) * sz
            if kind == "rabi":
                H = H + 0.5 * OMEGA0 * device.drive_scale * sx
            accum += w * evolve_expm(H, c_ops, rho_init, times[kind], op)
        vals[kind] = accum
    return apply_readout(_reassemble(settings, vals), device.readout)


def simulate(backend: str, device: Device, settings: List[Setting]) -> np.ndarray:
    if backend == "qutip":
        return simulate_qutip(device, settings)
    if backend == "qiskit":
        return simulate_qiskit(device, settings)
    if backend == "scipy_qs":
        return simulate_scipy_qs(device, settings)
    raise ValueError(backend)


def surrogate_probs(theta: np.ndarray, settings: List[Setting]) -> np.ndarray:
    delta, log_t1, log_t2, log_drive, raw_ro = theta
    t1 = np.exp(log_t1)
    t2 = np.exp(log_t2)
    drive = np.exp(log_drive)
    ro = readout_from_raw(raw_ro)
    omega = OMEGA0 * drive
    out = []
    for s in settings:
        t = s.t
        if s.kind == "t1":
            p = np.exp(-t / t1)
        elif s.kind == "ramsey":
            p = 0.5 * (1.0 + np.exp(-t / t2) * np.cos(delta * t))
        else:
            Om = np.sqrt(omega * omega + delta * delta)
            amp = (omega * omega) / (Om * Om + EPS)
            p_ideal = amp * np.sin(0.5 * Om * t) ** 2
            p = 0.5 + (p_ideal - 0.5) * np.exp(-t / max(t2, 1e-3))
        out.append(p)
    return apply_readout(np.asarray(out), ro)


def broad_prior() -> Tuple[np.ndarray, np.ndarray]:
    mean = np.array([
        2 * np.pi * 0.075,
        np.log(45.0),
        np.log(35.0),
        np.log(1.0),
        raw_from_readout(0.03),
    ])
    sd = np.array([2 * np.pi * 0.18, 0.9, 1.0, 0.32, 1.6])
    return mean, np.diag(sd * sd)


def safe_inv(cov: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(cov + 1e-8 * np.eye(cov.shape[0]))


def fit_map(
    settings: List[Setting],
    counts: np.ndarray,
    shots: int,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    init: np.ndarray | None = None,
) -> np.ndarray:
    inv = safe_inv(prior_cov)
    bounds = [
        (-2.8, 2.8),
        (np.log(4.0), np.log(220.0)),
        (np.log(1.5), np.log(220.0)),
        (np.log(0.55), np.log(1.45)),
        (-6.0, 3.0),
    ]
    x0 = np.asarray(prior_mean if init is None else init, dtype=float).copy()
    for i, (lo, hi) in enumerate(bounds):
        x0[i] = np.clip(x0[i], lo + 1e-5, hi - 1e-5)

    def objective(x: np.ndarray) -> float:
        p = surrogate_probs(x, settings)
        nll = -np.sum(counts * np.log(p) + (shots - counts) * np.log(1.0 - p))
        d = x - prior_mean
        return float(nll + 0.5 * d @ inv @ d)

    best = None
    starts = [x0, prior_mean]
    for d0 in [2 * np.pi * 0.02, 2 * np.pi * 0.08, 2 * np.pi * 0.14]:
        s = x0.copy()
        s[0] = d0
        starts.append(s)
    for s in starts:
        res = minimize(objective, s, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    return np.asarray(best.x, dtype=float)


def balanced_order(settings: List[Setting], rng: np.random.Generator) -> List[int]:
    by_kind = {}
    for kind in ["t1", "ramsey", "rabi"]:
        idx = [i for i, s in enumerate(settings) if s.kind == kind]
        rng.shuffle(idx)
        by_kind[kind] = idx
    out = []
    j = 0
    while len(out) < len(settings):
        for kind in ["t1", "ramsey", "rabi"]:
            if j < len(by_kind[kind]):
                out.append(by_kind[kind][j])
        j += 1
    return out


def observe(probs: np.ndarray, idx: List[int], shots: int, rng: np.random.Generator) -> np.ndarray:
    return rng.binomial(shots, probs[idx])


def covariance_regularized(samples: np.ndarray, floor_sd: np.ndarray, inflate: float = 1.0) -> np.ndarray:
    if len(samples) <= 1:
        cov = np.diag(floor_sd ** 2)
    else:
        cov = np.cov(samples.T, ddof=1)
    cov = np.atleast_2d(cov)
    cov = inflate * cov + np.diag(floor_sd ** 2)
    return 0.5 * (cov + cov.T)


def learn_hierarchy(source: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    family_means = {k: np.mean(v, axis=0) for k, v in source.items()}
    all_samples = np.concatenate(list(source.values()), axis=0)
    m0 = np.mean(np.stack(list(family_means.values())), axis=0)

    centered = np.concatenate([source[k] - family_means[k] for k in source], axis=0)
    W = covariance_regularized(centered, np.array([0.05, 0.06, 0.07, 0.025, 0.18]), inflate=1.20)

    means = np.stack(list(family_means.values()))
    if len(means) > 1:
        B_emp = np.cov(means.T, ddof=1)
    else:
        B_emp = np.zeros((len(PARAM_NAMES), len(PARAM_NAMES)))
    B = np.atleast_2d(B_emp) + np.diag(np.array([0.18, 0.16, 0.18, 0.07, 0.35]) ** 2)
    B = 0.5 * (B + B.T)

    pooled_cov = covariance_regularized(all_samples, np.array([0.08, 0.08, 0.10, 0.035, 0.22]), inflate=1.35)
    return {"m0": m0, "W": W, "B": B, "pooled_mean": np.mean(all_samples, axis=0), "pooled_cov": pooled_cov}


def anchor_prior(h: Dict[str, np.ndarray], anchors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m0, W, B = h["m0"], h["W"], h["B"]
    if len(anchors) == 0:
        return m0, W + B
    Bi, Wi = safe_inv(B), safe_inv(W)
    C = safe_inv(Bi + len(anchors) * Wi)
    mf = C @ (Bi @ m0 + Wi @ np.sum(anchors, axis=0))
    return mf, W + C


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n: int = 2000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    boots = np.mean(rng.choice(values, size=(n, len(values)), replace=True), axis=1)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def gate_weight(
    settings: List[Setting], counts: np.ndarray, shots: int,
    broad: Tuple[np.ndarray, np.ndarray], transfer: Tuple[np.ndarray, np.ndarray]
) -> float:
    if len(settings) < 4:
        return 0.5
    val_n = max(1, len(settings) // 4)
    train_settings, val_settings = settings[:-val_n], settings[-val_n:]
    train_counts, val_counts = counts[:-val_n], counts[-val_n:]
    tb = fit_map(train_settings, train_counts, shots, broad[0], broad[1])
    tt = fit_map(train_settings, train_counts, shots, transfer[0], transfer[1])
    pb = surrogate_probs(tb, val_settings)
    pt = surrogate_probs(tt, val_settings)
    nll_b = -np.sum(val_counts * np.log(pb) + (shots - val_counts) * np.log(1 - pb))
    nll_t = -np.sum(val_counts * np.log(pt) + (shots - val_counts) * np.log(1 - pt))
    score = np.clip((nll_b - nll_t) / max(1.0, val_n), -6.0, 6.0)
    return float(sigmoid(score))


def run(seed: int, out_dir: Path, n_source: int, n_target: int) -> None:
    rng = np.random.default_rng(seed)
    support, evals = make_protocol()
    backends = ["qutip", "qiskit", "scipy_qs"]

    bank: Dict[str, List[DeviceData]] = {}
    params_truth = []
    total_per_backend = n_source + 4 + n_target
    for b in backends:
        bank[b] = []
        for i in range(total_per_backend):
            d = family_device(rng, b)
            sp = simulate(b, d, support)
            ep = simulate(b, d, evals)
            bank[b].append(DeviceData(b, i, support, sp, evals, ep))
            params_truth.append({
                "backend": b, "device_id": i, "delta": d.delta, "t1": d.t1,
                "t2": d.t2, "drive_scale": d.drive_scale, "readout": d.readout,
                "qs_sigma": d.qs_sigma,
            })
        print(f"generated {b}: {total_per_backend} devices", flush=True)

    pd.DataFrame(params_truth).to_csv(out_dir / "device_truth.csv", index=False)

    broad = broad_prior()
    budgets = [4, 8, 16, 32]
    source_k = 48
    source_shots = 256
    target_shots = 64

    rows = []
    per_device_rows = []

    for heldout in backends:
        source_backends = [b for b in backends if b != heldout]
        source_est: Dict[str, np.ndarray] = {}
        for b in source_backends:
            est = []
            for dd in bank[b][:n_source]:
                order = balanced_order(support, np.random.default_rng(seed * 100003 + dd.device_id * 97 + backends.index(b)))
                idx = order[:source_k]
                counts = observe(dd.support_probs, idx, source_shots, np.random.default_rng(seed * 9001 + dd.device_id * 31 + backends.index(b)))
                st = [support[j] for j in idx]
                est.append(fit_map(st, counts, source_shots, broad[0], broad[1]))
            source_est[b] = np.stack(est)
        h = learn_hierarchy(source_est)
        pooled = (h["pooled_mean"], h["pooled_cov"])
        zero = anchor_prior(h, np.empty((0, len(PARAM_NAMES))))

        anchor_est = []
        for a, dd in enumerate(bank[heldout][n_source:n_source + 4]):
            order = balanced_order(support, np.random.default_rng(seed * 70001 + a * 113 + backends.index(heldout)))
            idx = order[:source_k]
            counts = observe(dd.support_probs, idx, source_shots, np.random.default_rng(seed * 17011 + a * 53 + backends.index(heldout)))
            st = [support[j] for j in idx]
            anchor_est.append(fit_map(st, counts, source_shots, broad[0], broad[1]))
        anchor_est = np.stack(anchor_est)
        one_anchor = anchor_prior(h, anchor_est[:1])
        two_anchor = anchor_prior(h, anchor_est[:2])

        target_bank = bank[heldout][n_source + 4:]
        for K in budgets:
            method_errors: Dict[str, List[float]] = {m: [] for m in [
                "target_only", "pooled", "hier_zero", "hier_one_anchor", "hier_two_anchor", "gated_zero"
            ]}
            for j, dd in enumerate(target_bank):
                order = balanced_order(support, np.random.default_rng(seed * 300007 + j * 131 + K * 17 + backends.index(heldout)))
                idx = order[:K]
                st = [support[x] for x in idx]
                counts = observe(dd.support_probs, idx, target_shots, np.random.default_rng(seed * 51001 + j * 211 + K * 19 + backends.index(heldout)))

                priors = {
                    "target_only": broad,
                    "pooled": pooled,
                    "hier_zero": zero,
                    "hier_one_anchor": one_anchor,
                    "hier_two_anchor": two_anchor,
                }
                pred = {}
                for m, pr in priors.items():
                    th = fit_map(st, counts, target_shots, pr[0], pr[1])
                    pred[m] = surrogate_probs(th, evals)
                    err = float(np.sqrt(np.mean((pred[m] - dd.eval_probs) ** 2)))
                    method_errors[m].append(err)
                    per_device_rows.append({"heldout": heldout, "K": K, "device": j, "method": m, "rmse": err})

                w = gate_weight(st, counts, target_shots, broad, zero)
                gated_pred = w * pred["hier_zero"] + (1.0 - w) * pred["target_only"]
                ge = float(np.sqrt(np.mean((gated_pred - dd.eval_probs) ** 2)))
                method_errors["gated_zero"].append(ge)
                per_device_rows.append({"heldout": heldout, "K": K, "device": j, "method": "gated_zero", "rmse": ge, "transfer_weight": w})

            for m, errs in method_errors.items():
                arr = np.asarray(errs)
                lo, hi = bootstrap_mean_ci(arr, np.random.default_rng(seed + K * 1009 + len(m) * 37 + backends.index(heldout)))
                rows.append({
                    "heldout": heldout, "K": K, "method": m,
                    "mean_rmse": float(np.mean(arr)), "median_rmse": float(np.median(arr)),
                    "ci95_low": lo, "ci95_high": hi,
                })
            print(f"LOFO heldout={heldout} K={K} complete", flush=True)

    summary = pd.DataFrame(rows)
    per_device = pd.DataFrame(per_device_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_device.to_csv(out_dir / "per_device.csv", index=False)

    headline = {}
    for heldout in backends:
        headline[heldout] = {}
        for K in budgets:
            sub = summary[(summary.heldout == heldout) & (summary.K == K)].set_index("method")
            base = float(sub.loc["target_only", "mean_rmse"])
            headline[heldout][str(K)] = {m: float(sub.loc[m, "mean_rmse"]) for m in sub.index}
            headline[heldout][str(K)]["one_anchor_relative_improvement"] = float(
                1.0 - sub.loc["hier_one_anchor", "mean_rmse"] / base
            )
    (out_dir / "headline.json").write_text(json.dumps(headline, indent=2))

    md = ["# QPI-0C cross-stack leave-one-family-out results", "", f"Seed: `{seed}`", ""]
    for heldout in backends:
        md += [f"## Held out: {heldout}", "", "| K | target-only | pooled | hier-zero | 1-anchor | 2-anchor | gated-zero |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for K in budgets:
            sub = summary[(summary.heldout == heldout) & (summary.K == K)].set_index("method")
            md.append(
                f"| {K} | {sub.loc['target_only','mean_rmse']:.4f} | {sub.loc['pooled','mean_rmse']:.4f} | "
                f"{sub.loc['hier_zero','mean_rmse']:.4f} | {sub.loc['hier_one_anchor','mean_rmse']:.4f} | "
                f"{sub.loc['hier_two_anchor','mean_rmse']:.4f} | {sub.loc['gated_zero','mean_rmse']:.4f} |"
            )
        md.append("")
    (out_dir / "RESULTS.md").write_text("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--n-source", type=int, default=12)
    ap.add_argument("--n-target", type=int, default=24)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.seed, args.out, args.n_source, args.n_target)
