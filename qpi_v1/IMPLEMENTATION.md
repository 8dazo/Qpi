# QPI-1 implementation plan

The implementation order is evidence-driven. Do not jump to a large foundation model.

## Stage 1: data contract

Represent every observation as:
- processor/family identifier
- timestamp
- property/observable
- node or edge identity
- value + uncertainty when available
- topology metadata
- measurement cost

Create chronological source/target splits before model fitting.

## Stage 2: graph baseline

Start with a small message-passing encoder over processor topology. The first benchmark must answer only one question: does topology-aware transfer beat QPI-0 PCA and target-only at K=4/8 without worsening the high-K tail?

No temporal model yet.

## Stage 3: property experts + gate

Use a shared graph encoder only where useful, then independent property heads. Fit a target-evidence gate that predicts transfer reliability from revealed-target cross-validation residuals, model disagreement, uncertainty, K, and property identity.

T2 begins with alpha=0 and must earn re-entry through held-out validation.

## Stage 4: temporal latent

After graph transfer is validated, add a state-space/recurrent temporal component. Train on source history only up to each target cutoff. Evaluate current-state reconstruction and future-state prediction separately.

## Stage 5: active physical measurement

Given posterior uncertainty over hidden state, choose the next node/edge/experiment. Compare random, uncertainty sampling, information-gain acquisition, and target-only acquisition at equal measurement cost.

## Stage 6: actual QPU

Translate K from masked calibration values into actual circuits/shots/query cost. Freeze model/hyperparameters before touching the held-out hardware target.

## Promotion rule

A component is promoted only when it improves paired held-out results or uncertainty calibration without violating the equal-budget protocol. Architectural complexity is not itself progress.
