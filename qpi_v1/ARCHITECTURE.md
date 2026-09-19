# QPI-1 — restructured architecture

QPI-0 established that reusable cross-processor structure exists, but a single universal PCA latent is not a safe model of real hardware. QPI-1 therefore treats transfer as a conditional physical prior that must earn trust from target evidence.

## Frozen scientific claim

Given a population of source processors and a completely unseen target processor, learn reusable, observable-specific physical structure that reduces the number of target measurements needed to predict hidden physical state, while detecting and suppressing negative transfer.

This is intentionally narrower than "quantum foundation model". QPI-1 succeeds only through equal-budget leave-one-processor/family-out experiments.

## Model

For processor graph G, observable/property p, time t, and sparse target context C:

  shared physics -> property expert -> hardware-family latent -> processor latent -> temporal state
                                      |
                                      +-> graph/spatial decoder

The predictive distribution is a mixture of a target-only model and transferred experts:

  p(y_hidden | C,G,p,t)
    = (1-alpha(C)) p_target(y_hidden | C,G,p,t)
      + alpha(C) sum_e pi_e(C,p,G) p_transfer,e(y_hidden | C,G,p,t)

alpha is not fixed. It is inferred only from revealed target evidence and must decay toward zero when source and target disagree or when target evidence becomes sufficient.

## Required components

1. Property-specific experts
   - Separate transfer heads/priors for T1, T2, readout error, gate error, and later pulse/Hamiltonian observables.
   - No forced shared output latent across properties.
   - T2 is initially transfer-disabled unless validation demonstrates positive transfer.

2. Graph-structured processor encoder
   - Node state: qubit-local measurements and metadata.
   - Edge state: coupling/gate measurements when available.
   - Permutation-aware message passing rather than qubit-index PCA.
   - Architecture must support different processor sizes/topologies.

3. Hierarchical population prior
   - global latent z_global
   - family latent z_family
   - processor latent z_device
   - temporal latent z_t
   - posterior is conditioned on sparse target observations.

4. Temporal dynamics
   - Source history is censored at each target cutoff.
   - Temporal state models calibration drift instead of pooling historical snapshots as IID samples.

5. Evidence gate / prior decay
   - Inputs may use only revealed target values, source-model uncertainty, predictive residuals on held-in target observations, K, property identity, topology metadata, and time.
   - Hidden target values are forbidden.
   - Gate outputs alpha in [0,1] and expert weights pi.
   - Strong target evidence or source-target disagreement must reduce alpha.

6. Uncertainty and abstention
   - Produce predictive intervals, not only point estimates.
   - If transfer uncertainty or target disagreement is high, fall back to target-only.
   - Calibration quality (coverage/error) is a primary metric.

## QPI-1 benchmark ladder

QPI-1A — graph reconstruction on recorded IBM calibration history.
Compare target-only, source-shift, PCA, graph target-only, graph transfer, and graph transfer + evidence gate under identical masks.

QPI-1B — property specialization.
Train/evaluate separate property experts and a shared-trunk/multi-head ablation. Explicitly test whether T2 should transfer at all.

QPI-1C — temporal transfer.
Predict future hidden target state from sparse current measurements with strict time censoring. Compare static vs temporal priors.

QPI-1D — cross-stack physics.
Port the graph/hierarchical model back to QuTiP, Qiskit Dynamics, and the independent quasi-static/SciPy family. Hold an entire simulator family out.

QPI-1E — active measurement selection.
Replace random K masks with an acquisition policy that chooses the next physical measurement by expected information gain / posterior uncertainty reduction. Compare at equal K.

QPI-1F — real-QPU experiment.
Use actual circuits/shots on an unseen processor. K becomes physical experiment cost rather than a masked-calibration proxy. This is required before claiming real measurement reduction.

## Baselines

Every experiment must include:
- target-only constant/linear or appropriate physical fit
- source mean/shift where meaningful
- QPI-0 PCA
- target-only graph model
- transfer graph model without gate
- transfer graph model with gate
- oracle selector only as an upper bound, never as a deployable result

## Primary metrics

1. hidden-state prediction error at fixed target measurement budget K
2. target measurements required to reach a fixed error threshold
3. paired per-target improvement and bootstrap confidence interval
4. negative-transfer rate
5. predictive interval coverage/calibration
6. transfer weight alpha versus K
7. active-acquisition area under error-vs-measurement curve

Macro averages must give equal weight to each held-out processor/property task. Mask repeats are averaged within a snapshot before confidence intervals.

## Hard validity rules

- Completely held-out target processor/family during population training.
- No future source data beyond target cutoff.
- No hidden target labels for gating, hyperparameter selection, or acquisition.
- Equal target measurement/query/shot budgets across methods.
- Simulator-family holdout for cross-stack claims.
- Report failures and negative transfer.
- Do not call masked published calibration values "shots".
- Do not claim a universal quantum foundation model from these experiments.

## Kill criteria

The QPI-1 thesis is weakened or rejected if, after graph/property/temporal restructuring:
- transfer does not beat strong target-only baselines in the low-measurement regime across multiple held-out processors/families;
- gains disappear under equal physical measurement cost;
- uncertainty cannot identify harmful transfer better than a simple K-based fallback;
- active measurement selection, rather than population knowledge, explains essentially all measurement savings.

## Current evidence carried forward

QPI-0C: transfer survived independent simulator stacks, but gains were smaller than the original synthetic result and negative transfer appeared as target evidence increased.

QPI-0E: on recorded IBM calibration histories, a universal PCA latent failed overall. Some low-K T1/readout/sx tasks transferred positively; T2 showed strong negative transfer.

QPI-0F: target-evidence blending improved the low-K macro result, but retained a negative-transfer tail at larger K. This motivates learned prior decay and property-specific experts.

These are architecture constraints, not claims to optimize away.
