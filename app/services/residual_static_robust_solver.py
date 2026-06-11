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
    solve_residual_static_robust_least_squares,
    validate_residual_static_robust_options,
)


def robust_options_from_request_robust(
    request_robust: object,
) -> ResidualStaticRobustOptions:
    """Convert an app request robust object into package robust options."""
    values: dict[str, object] = {}
    for field in (
        'enabled',
        'method',
        'max_iterations',
        'threshold',
        'min_used_fraction',
    ):
        try:
            values[field] = getattr(request_robust, field)
        except AttributeError as exc:
            raise ValueError(f'robust missing field: {field}') from exc
    return validate_residual_static_robust_options(
        ResidualStaticRobustOptions(**values)  # type: ignore[arg-type]
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
