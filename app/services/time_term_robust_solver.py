"""Compatibility exports for time-term robust solver core computations."""

from __future__ import annotations

from seis_statics.time_term.robust_solver import (
    ROBUST_SCALE_FLOOR_S,
    TimeTermRobustIteration,
    TimeTermRobustMethod,
    TimeTermRobustSolverOptions,
    TimeTermRobustSolverResult,
    TimeTermRobustStopReason,
    compute_time_term_robust_scores,
    solve_time_term_robust_least_squares,
    subset_time_term_design_matrix_rows,
    summarize_time_term_robust_solver_result,
    validate_time_term_robust_solver_options,
)

__all__ = [
    'ROBUST_SCALE_FLOOR_S',
    'TimeTermRobustIteration',
    'TimeTermRobustMethod',
    'TimeTermRobustSolverOptions',
    'TimeTermRobustSolverResult',
    'TimeTermRobustStopReason',
    'compute_time_term_robust_scores',
    'solve_time_term_robust_least_squares',
    'subset_time_term_design_matrix_rows',
    'summarize_time_term_robust_solver_result',
    'validate_time_term_robust_solver_options',
]
