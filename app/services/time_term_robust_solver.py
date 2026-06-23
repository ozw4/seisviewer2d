"""Compatibility exports for time-term robust solver core computations."""

from __future__ import annotations

from seis_statics.time_term.robust_solver import (
    ROBUST_SCALE_FLOOR_S,
    TimeTermRobustIterationSummary,
    TimeTermRobustMethod,
    TimeTermRobustOptions,
    TimeTermRobustSolveResult,
    TimeTermRobustStopReason,
    build_time_term_outlier_mask,
    compute_time_term_robust_center_scale,
    solve_time_term_robust_least_squares,
    summarize_time_term_robust_solver_result,
    validate_time_term_robust_options,
)

__all__ = [
    'ROBUST_SCALE_FLOOR_S',
    'TimeTermRobustIterationSummary',
    'TimeTermRobustMethod',
    'TimeTermRobustOptions',
    'TimeTermRobustSolveResult',
    'TimeTermRobustStopReason',
    'build_time_term_outlier_mask',
    'compute_time_term_robust_center_scale',
    'solve_time_term_robust_least_squares',
    'summarize_time_term_robust_solver_result',
    'validate_time_term_robust_options',
]
