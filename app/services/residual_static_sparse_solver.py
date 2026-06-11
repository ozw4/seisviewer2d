"""Compatibility exports for residual static sparse solver helpers."""

from __future__ import annotations

from seis_statics.residual.solver import (
    ResidualStaticAugmentedSystem,
    ResidualStaticGauge,
    ResidualStaticLeastSquaresResult,
    ResidualStaticLsmrDiagnostics,
    ResidualStaticLsmrOptions,
    ResidualStaticMinimumDataSummary,
    ResidualStaticObservationGraphSummary,
    ResidualStaticRawLsmrResult,
    ResidualStaticStabilizationOptions,
    ResidualStaticStabilizedLeastSquaresResult,
    build_csr_matrix_from_residual_static_triplets,
    build_delay_damping_rows,
    build_residual_static_observation_graph_summary,
    build_stabilized_residual_static_augmented_system,
    build_zero_mean_gauge_rows,
    run_sparse_lsmr,
    solve_residual_static_least_squares,
    solve_residual_static_stabilized_least_squares,
    validate_lsmr_options,
    validate_minimum_residual_static_data,
    validate_residual_static_estimated_delay_limit,
    validate_residual_static_stabilization_options,
    validate_residual_static_used_mask,
)


def stabilization_options_from_request_solver(
    solver: object,
) -> ResidualStaticStabilizationOptions:
    """Convert an app request solver object into package stabilization options."""
    values: dict[str, object] = {}
    for field in (
        'gauge',
        'damping_lambda',
        'min_valid_picks',
        'min_picks_per_source',
        'min_picks_per_receiver',
        'max_abs_estimated_delay_ms',
    ):
        try:
            values[field] = getattr(solver, field)
        except AttributeError as exc:
            raise ValueError(f'solver missing stabilization field: {field}') from exc
    return validate_residual_static_stabilization_options(
        ResidualStaticStabilizationOptions(**values)  # type: ignore[arg-type]
    )


__all__ = [
    'ResidualStaticAugmentedSystem',
    'ResidualStaticGauge',
    'ResidualStaticLeastSquaresResult',
    'ResidualStaticLsmrDiagnostics',
    'ResidualStaticLsmrOptions',
    'ResidualStaticMinimumDataSummary',
    'ResidualStaticObservationGraphSummary',
    'ResidualStaticRawLsmrResult',
    'ResidualStaticStabilizationOptions',
    'ResidualStaticStabilizedLeastSquaresResult',
    'build_delay_damping_rows',
    'build_csr_matrix_from_residual_static_triplets',
    'build_residual_static_observation_graph_summary',
    'build_stabilized_residual_static_augmented_system',
    'build_zero_mean_gauge_rows',
    'run_sparse_lsmr',
    'solve_residual_static_least_squares',
    'solve_residual_static_stabilized_least_squares',
    'stabilization_options_from_request_solver',
    'validate_minimum_residual_static_data',
    'validate_lsmr_options',
    'validate_residual_static_estimated_delay_limit',
    'validate_residual_static_stabilization_options',
    'validate_residual_static_used_mask',
]
