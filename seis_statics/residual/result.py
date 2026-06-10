"""Public result objects for first-break residual statics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seis_statics.residual.robust import (
    ResidualStaticRobustIterationSummary,
    ResidualStaticRobustOptions,
    ResidualStaticRobustStopReason,
)
from seis_statics.residual.solver import (
    ResidualStaticLsmrDiagnostics,
    ResidualStaticMinimumDataSummary,
    ResidualStaticObservationGraphSummary,
    ResidualStaticStabilizationOptions,
)
from seis_statics.residual.types import MoveoutModel


@dataclass(frozen=True)
class FirstBreakResidualStaticsResult:
    """Stable package-level result for first-break residual statics."""

    moveout_model: MoveoutModel
    source_id: np.ndarray
    receiver_id: np.ndarray

    intercept_s: float
    slowness_s_per_offset_unit: float | None
    source_delay_s: np.ndarray
    receiver_delay_s: np.ndarray

    moveout_model_time_s: np.ndarray
    estimated_trace_delay_s: np.ndarray
    modeled_pick_time_s: np.ndarray
    residual_s: np.ndarray
    residual_valid_mask: np.ndarray

    initial_used_pick_mask: np.ndarray
    used_pick_mask: np.ndarray
    rejected_pick_mask: np.ndarray
    rejected_iteration: np.ndarray

    diagnostics: ResidualStaticLsmrDiagnostics
    minimum_data: ResidualStaticMinimumDataSummary
    graph: ResidualStaticObservationGraphSummary
    stabilization_options: ResidualStaticStabilizationOptions
    robust_options: ResidualStaticRobustOptions
    robust_iteration_summaries: tuple[ResidualStaticRobustIterationSummary, ...]
    robust_stop_reason: ResidualStaticRobustStopReason

    n_initial_used_picks: int
    n_final_used_picks: int
    n_rejected_total: int
    n_observations: int
    n_model_parameters: int
    n_gauge_rows: int
    n_damping_rows: int
    max_abs_estimated_delay_s: float


__all__ = ['FirstBreakResidualStaticsResult']
