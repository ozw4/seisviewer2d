"""Compatibility exports for residual static robust solver helpers."""

from __future__ import annotations

from seis_statics.residual.robust import (
    ROBUST_SCALE_FLOOR_S,
    ResidualStaticRobustIterationSummary,
    ResidualStaticRobustMethod,
    ResidualStaticRobustOptions,
    ResidualStaticRobustSolveResult,
    ResidualStaticRobustStopReason,
    build_residual_static_outlier_mask,
    compute_residual_static_robust_center_scale,
    robust_options_from_request_robust,
    solve_residual_static_robust_least_squares,
    validate_residual_static_robust_options,
)


__all__ = [
    'ROBUST_SCALE_FLOOR_S',
    'ResidualStaticRobustIterationSummary',
    'ResidualStaticRobustMethod',
    'ResidualStaticRobustOptions',
    'ResidualStaticRobustSolveResult',
    'ResidualStaticRobustStopReason',
    'build_residual_static_outlier_mask',
    'compute_residual_static_robust_center_scale',
    'robust_options_from_request_robust',
    'solve_residual_static_robust_least_squares',
    'validate_residual_static_robust_options',
]
