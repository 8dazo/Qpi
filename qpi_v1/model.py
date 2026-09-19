from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Protocol

import numpy as np

PROPERTY_TRANSFER_DEFAULTS: Dict[str, bool] = {
    "T1": True,
    "T2": False,
    "readout_error": True,
    "sx_error": True,
}


@dataclass(frozen=True)
class SparseTargetContext:
    property_name: str
    observed_nodes: np.ndarray
    observed_values: np.ndarray
    measurement_budget: int
    time_index: float | None = None


@dataclass
class PredictiveState:
    mean: np.ndarray
    variance: np.ndarray


class TargetModel(Protocol):
    def predict(self, context: SparseTargetContext) -> PredictiveState: ...


class TransferExpert(Protocol):
    def predict(self, context: SparseTargetContext) -> PredictiveState: ...


@dataclass
class GateDiagnostics:
    alpha: float
    target_cv_error: float
    transfer_cv_error: float
    disagreement: float
    property_transfer_enabled: bool


class EvidenceGate:
    """Conservative QPI-1 transfer gate.

    This scaffold deliberately uses only revealed target evidence. It is a
    baseline for the learned gate, not the final architecture.
    """

    def __init__(self, k_decay: float = 12.0, disagreement_scale: float = 1.0):
        self.k_decay = float(k_decay)
        self.disagreement_scale = float(disagreement_scale)

    def weight(
        self,
        context: SparseTargetContext,
        target_cv_error: float,
        transfer_cv_error: float,
        disagreement: float,
    ) -> GateDiagnostics:
        enabled = PROPERTY_TRANSFER_DEFAULTS.get(context.property_name, True)
        if not enabled:
            alpha = 0.0
        else:
            evidence_advantage = np.log(
                max(target_cv_error, 1e-8) / max(transfer_cv_error, 1e-8)
            )
            trust = 1.0 / (1.0 + np.exp(-evidence_advantage))
            budget_decay = np.exp(-context.measurement_budget / self.k_decay)
            mismatch_decay = np.exp(
                -max(disagreement, 0.0) / max(self.disagreement_scale, 1e-8)
            )
            alpha = float(np.clip(trust * budget_decay * mismatch_decay, 0.0, 1.0))
        return GateDiagnostics(
            alpha=alpha,
            target_cv_error=float(target_cv_error),
            transfer_cv_error=float(transfer_cv_error),
            disagreement=float(disagreement),
            property_transfer_enabled=enabled,
        )


def precision_blend(
    target: PredictiveState,
    transfer: PredictiveState,
    alpha: float,
) -> PredictiveState:
    """Blend distributions while retaining a disagreement uncertainty term."""
    a = float(np.clip(alpha, 0.0, 1.0))
    mean = (1.0 - a) * target.mean + a * transfer.mean
    within = (1.0 - a) * target.variance + a * transfer.variance
    between = a * (1.0 - a) * np.square(target.mean - transfer.mean)
    return PredictiveState(mean=mean, variance=within + between)


class QPI1Model:
    def __init__(
        self,
        target_model: TargetModel,
        experts: Mapping[str, TransferExpert],
        gate: EvidenceGate,
    ):
        self.target_model = target_model
        self.experts = dict(experts)
        self.gate = gate

    def predict(
        self,
        context: SparseTargetContext,
        *,
        target_cv_error: float,
        transfer_cv_error: float,
        disagreement: float,
    ) -> tuple[PredictiveState, GateDiagnostics]:
        target = self.target_model.predict(context)
        expert = self.experts.get(context.property_name)
        if expert is None:
            diag = GateDiagnostics(
                alpha=0.0,
                target_cv_error=target_cv_error,
                transfer_cv_error=transfer_cv_error,
                disagreement=disagreement,
                property_transfer_enabled=False,
            )
            return target, diag

        transfer = expert.predict(context)
        diag = self.gate.weight(
            context,
            target_cv_error=target_cv_error,
            transfer_cv_error=transfer_cv_error,
            disagreement=disagreement,
        )
        return precision_blend(target, transfer, diag.alpha), diag
