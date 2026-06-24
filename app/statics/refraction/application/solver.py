"""Application adapter for external refraction static least-squares solving."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from scipy import sparse
from seis_statics.refraction.solver import (
    RefractionStaticSolverError as CoreRefractionStaticSolverError,
    solve_refraction_static_design_least_squares,
)

from app.statics.refraction.application.core_options import (
    model_options_from_request,
    resolve_weathering_velocity_from_model_request as resolve_weathering_velocity_m_s,
    solver_options_from_request,
)
from app.statics.refraction.contracts.model import RefractionStaticModelRequest
from app.statics.refraction.contracts.options import RefractionStaticSolverRequest
from app.statics.refraction.contracts.result_types import (
    RefractionStaticDesignMatrix,
    RefractionStaticSolverResult,
    ResolvedRefractionFirstLayer,
)

_CELL_THRESHOLD_QC_KEYS = (
    'min_observations_per_cell',
    'n_low_fold_cells',
    'n_observations_rejected_by_low_fold_cell',
    'low_fold_cell_rejection_reason',
    'low_fold_cell_id',
    'cell_observation_count',
)


class RefractionStaticSolverError(ValueError):
    """Raised when the external refraction static solve cannot be completed."""


def solve_refraction_static_bounded_ls(
    *,
    design_matrix: RefractionStaticDesignMatrix,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> RefractionStaticSolverResult:
    """Solve a refraction static design matrix through the external core."""
    _validate_app_model_for_external_solve(
        model=model,
        design_matrix=design_matrix,
        resolved_first_layer=resolved_first_layer,
    )
    _validate_design_matrix_has_observable_active_columns(design_matrix)
    try:
        core_result = solve_refraction_static_design_least_squares(
            design_matrix,
            model=model_options_from_request(model),
            solver_options=solver_options_from_request(solver),
        )
    except CoreRefractionStaticSolverError as exc:
        raise RefractionStaticSolverError(str(exc)) from exc
    except ValueError as exc:
        raise RefractionStaticSolverError(str(exc)) from exc

    result = _app_solver_result_from_core(
        design_matrix=design_matrix,
        core_result=core_result,
        model=model,
        solver=solver,
        resolved_first_layer=resolved_first_layer,
    )
    if design_matrix.bedrock_velocity_mode != 'solve_cell':
        return result
    extra_qc = {
        key: design_matrix.qc[key]
        for key in _CELL_THRESHOLD_QC_KEYS
        if key in design_matrix.qc
    }
    if not extra_qc:
        return result
    return replace(result, qc={**result.qc, **extra_qc})


def _app_solver_result_from_core(
    *,
    design_matrix: RefractionStaticDesignMatrix,
    core_result: Any,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None,
) -> RefractionStaticSolverResult:
    row_trace = np.asarray(design_matrix.row_trace_index_sorted, dtype=np.int64)
    used_row_mask = _trace_indexed_mask_to_row_mask(
        core_result.used_observation_mask_sorted,
        row_trace_index_sorted=row_trace,
        name='core_result.used_observation_mask_sorted',
    )
    rejected_row_mask = _trace_indexed_mask_to_row_mask(
        core_result.rejected_observation_mask_sorted,
        row_trace_index_sorted=row_trace,
        name='core_result.rejected_observation_mask_sorted',
    )
    active_node_id = np.asarray(design_matrix.active_node_id, dtype=np.int64)
    parameter_vector = np.ascontiguousarray(
        core_result.parameter_vector,
        dtype=np.float64,
    )
    active_half = np.ascontiguousarray(
        parameter_vector[: active_node_id.shape[0]],
        dtype=np.float64,
    )
    lower_bounds = np.ascontiguousarray(
        core_result.system.lower_bounds,
        dtype=np.float64,
    )
    upper_bounds = np.ascontiguousarray(
        core_result.system.upper_bounds,
        dtype=np.float64,
    )
    active_cell_id = _optional_int_array(getattr(design_matrix, 'active_cell_id', None))
    inactive_cell_id = _optional_int_array(
        getattr(design_matrix, 'inactive_cell_id', None),
    )
    cell_slowness, cell_velocity, cell_status = _active_cell_solution_arrays(
        core_result=core_result,
        active_cell_id=active_cell_id,
    )
    bedrock_slowness, bedrock_velocity = _summary_bedrock_values(
        core_result=core_result,
        cell_slowness=cell_slowness,
    )
    row_midpoint_cell_id = _optional_int_array(core_result.row_midpoint_cell_id)
    row_midpoint_velocity = _optional_float_array(
        core_result.row_midpoint_bedrock_velocity_m_s,
    )
    qc = _app_qc_from_core(
        design_matrix=design_matrix,
        core_result=core_result,
        model=model,
        solver=solver,
        bedrock_slowness_s_per_m=bedrock_slowness,
        bedrock_velocity_m_s=bedrock_velocity,
        used_row_mask=used_row_mask,
        rejected_row_mask=rejected_row_mask,
        resolved_first_layer=resolved_first_layer,
    )
    return RefractionStaticSolverResult(
        parameter_vector=parameter_vector,
        active_node_id=np.ascontiguousarray(active_node_id, dtype=np.int64),
        active_node_half_intercept_time_s=active_half,
        node_id=np.ascontiguousarray(core_result.node_id, dtype=np.int64),
        node_half_intercept_time_s=np.ascontiguousarray(
            core_result.node_half_intercept_time_s,
            dtype=np.float64,
        ),
        node_solution_status=np.ascontiguousarray(core_result.node_solution_status),
        bedrock_velocity_mode=core_result.bedrock_velocity_mode,
        bedrock_slowness_s_per_m=bedrock_slowness,
        bedrock_velocity_m_s=bedrock_velocity,
        row_trace_index_sorted=np.ascontiguousarray(row_trace, dtype=np.int64),
        row_source_node_id=np.ascontiguousarray(
            design_matrix.row_source_node_id,
            dtype=np.int64,
        ),
        row_receiver_node_id=np.ascontiguousarray(
            design_matrix.row_receiver_node_id,
            dtype=np.int64,
        ),
        row_distance_m=np.ascontiguousarray(
            design_matrix.row_distance_m,
            dtype=np.float64,
        ),
        observed_pick_time_s=np.ascontiguousarray(
            design_matrix.observed_pick_time_s,
            dtype=np.float64,
        ),
        modeled_pick_time_s=np.ascontiguousarray(
            core_result.row_modeled_pick_time_s,
            dtype=np.float64,
        ),
        residual_time_s=np.ascontiguousarray(
            core_result.row_residual_s,
            dtype=np.float64,
        ),
        used_row_mask=used_row_mask,
        rejected_by_robust_mask=rejected_row_mask,
        solver_status='success' if core_result.solver_success else 'failed',
        solver_message=str(core_result.solver_message),
        solver_cost=float(core_result.solver_cost),
        solver_optimality=float(core_result.solver_optimality),
        solver_nit=int(core_result.solver_iterations),
        robust_iteration_count=len(core_result.robust_iteration_summaries),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        qc=qc,
        active_cell_id=active_cell_id,
        inactive_cell_id=inactive_cell_id,
        cell_bedrock_slowness_s_per_m=cell_slowness,
        cell_bedrock_velocity_m_s=cell_velocity,
        cell_velocity_status=cell_status,
        row_midpoint_cell_id=row_midpoint_cell_id,
        row_midpoint_bedrock_velocity_m_s=row_midpoint_velocity,
    )


def _trace_indexed_mask_to_row_mask(
    mask_sorted: np.ndarray,
    *,
    row_trace_index_sorted: np.ndarray,
    name: str,
) -> np.ndarray:
    mask = np.asarray(mask_sorted, dtype=bool)
    row_trace = np.asarray(row_trace_index_sorted, dtype=np.int64)
    if mask.ndim != 1:
        raise RefractionStaticSolverError(f'{name} must be one-dimensional')
    if row_trace.ndim != 1:
        raise RefractionStaticSolverError('row_trace_index_sorted must be one-dimensional')
    if row_trace.size == 0:
        return np.empty(0, dtype=bool)
    min_trace = int(np.min(row_trace))
    max_trace = int(np.max(row_trace))
    if min_trace < 0 or max_trace >= mask.shape[0]:
        raise RefractionStaticSolverError(
            f'{name} does not cover row_trace_index_sorted'
        )
    return np.ascontiguousarray(np.take(mask, row_trace), dtype=bool)


def _app_qc_from_core(
    *,
    design_matrix: RefractionStaticDesignMatrix,
    core_result: Any,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    bedrock_slowness_s_per_m: float,
    bedrock_velocity_m_s: float,
    used_row_mask: np.ndarray,
    rejected_row_mask: np.ndarray,
    resolved_first_layer: ResolvedRefractionFirstLayer | None,
) -> dict[str, Any]:
    residual_s = np.asarray(core_result.row_residual_s, dtype=np.float64)
    node_status = np.asarray(core_result.node_solution_status).astype(str, copy=False)
    system = core_result.system
    qc: dict[str, Any] = dict(core_result.qc)
    qc.update(
        {
            'method': 'gli_variable_thickness',
            'bedrock_velocity_mode': str(core_result.bedrock_velocity_mode),
            'weathering_velocity_m_s': float(
                resolve_weathering_velocity_m_s(
                    model=model,
                    resolved_first_layer=resolved_first_layer,
                    name='model.weathering_velocity_m_s',
                )
            ),
            'bedrock_slowness_s_per_m': float(bedrock_slowness_s_per_m),
            'bedrock_velocity_m_s': float(bedrock_velocity_m_s),
            'bedrock_velocity_status': str(core_result.bedrock_velocity_status),
            'fixed_bedrock_velocity_m_s': (
                None
                if core_result.bedrock_velocity_mode != 'fixed_global'
                else float(bedrock_velocity_m_s)
            ),
            'n_observations': int(design_matrix.n_observations),
            'n_used_observations': int(np.count_nonzero(used_row_mask)),
            'n_rejected_by_robust': int(np.count_nonzero(rejected_row_mask)),
            'robust_enabled': bool(solver.robust.enabled),
            'robust_method': str(solver.robust.method),
            'robust_iteration_count': len(core_result.robust_iteration_summaries),
            'robust_stop_reason': _viewer_robust_stop_reason(
                str(core_result.robust_stop_reason)
            ),
            'solver_status': 'success' if core_result.solver_success else 'failed',
            'solver_message': str(core_result.solver_message),
            'solver_cost': float(core_result.solver_cost),
            'solver_optimality': float(core_result.solver_optimality),
            'solver_nit': int(core_result.solver_iterations),
            'n_damping_rows': int(system.n_damping_rows),
            'n_source_receiver_gauge_rows': int(system.n_gauge_rows),
            'n_cell_smoothing_rows': int(system.n_smoothing_rows),
            'n_augmented_rows': int(system.n_augmented_rows),
            'damping_applied_to': 'half_intercept_time_columns',
            'half_intercept_time_clipped_lower_count': int(
                np.count_nonzero(node_status == 'clipped_lower')
            ),
            'half_intercept_time_clipped_upper_count': int(
                np.count_nonzero(node_status == 'clipped_upper')
            ),
            'bedrock_slowness_clipped': str(core_result.bedrock_velocity_status)
            in {'clipped_lower', 'clipped_upper'},
            'bedrock_slowness_clipped_lower': str(core_result.bedrock_velocity_status)
            == 'clipped_upper',
            'bedrock_slowness_clipped_upper': str(core_result.bedrock_velocity_status)
            == 'clipped_lower',
        }
    )
    qc.update(_residual_stats_ms(residual_s))
    row_type_counts = dict(qc.get('row_type_counts', {}))
    if system.n_damping_rows:
        row_type_counts['damping'] = int(system.n_damping_rows)
    if system.n_gauge_rows:
        row_type_counts['source_receiver_gauge'] = int(system.n_gauge_rows)
        qc['source_receiver_gauge'] = 'mean_source_equals_mean_receiver'
    if system.n_smoothing_rows:
        row_type_counts['cell_smoothing'] = int(system.n_smoothing_rows)
    if row_type_counts:
        qc['row_type_counts'] = row_type_counts
    if system.smoothing_rows is not None:
        smoothing_qc = dict(getattr(system.smoothing_rows, 'qc', {}))
        qc.update(smoothing_qc)
        if 'smoothing_row_scale' not in qc and 'row_scale' in smoothing_qc:
            qc['smoothing_row_scale'] = float(smoothing_qc['row_scale'])
    elif system.n_smoothing_rows == 0:
        qc.setdefault('smoothing_row_scale', 0.0)
    if core_result.bedrock_velocity_mode == 'solve_cell':
        qc.update(_cell_qc(core_result))
    return qc


def _active_cell_solution_arrays(
    *,
    core_result: Any,
    active_cell_id: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if active_cell_id is None:
        return None, None, None
    if active_cell_id.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype='<U16'),
        )
    cell_id = np.asarray(core_result.cell_id, dtype=np.int64)
    slowness = np.asarray(
        core_result.cell_bedrock_slowness_s_per_m,
        dtype=np.float64,
    )
    velocity = np.asarray(core_result.cell_bedrock_velocity_m_s, dtype=np.float64)
    status = np.asarray(core_result.cell_velocity_status)
    if (
        cell_id.ndim != 1
        or slowness.shape != cell_id.shape
        or velocity.shape != cell_id.shape
        or status.shape != cell_id.shape
        or np.unique(cell_id).shape[0] != cell_id.shape[0]
    ):
        raise RefractionStaticSolverError(
            'solve_cell result active cell arrays length mismatch'
        )
    cell_pos_by_id = {int(cell): idx for idx, cell in enumerate(cell_id.tolist())}
    try:
        active_pos = np.asarray(
            [cell_pos_by_id[int(cell)] for cell in active_cell_id],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise RefractionStaticSolverError(
            'solve_cell result is missing an active cell solution'
        ) from exc
    return (
        np.ascontiguousarray(
            slowness[active_pos],
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            velocity[active_pos],
            dtype=np.float64,
        ),
        np.ascontiguousarray(status[active_pos]),
    )


def _summary_bedrock_values(
    *,
    core_result: Any,
    cell_slowness: np.ndarray | None,
) -> tuple[float, float]:
    if core_result.bedrock_velocity_mode != 'solve_cell':
        return (
            float(core_result.bedrock_slowness_s_per_m),
            float(core_result.bedrock_velocity_m_s),
        )
    if cell_slowness is None or cell_slowness.size == 0:
        raise RefractionStaticSolverError(
            'solve_cell result requires active cell slowness values'
        )
    finite = np.asarray(cell_slowness, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RefractionStaticSolverError(
            'solve_cell result requires finite active cell slowness values'
        )
    slowness = float(np.median(finite))
    if slowness <= 0.0:
        raise RefractionStaticSolverError(
            'solve_cell summary bedrock slowness must be positive'
        )
    return slowness, float(1.0 / slowness)


def _cell_qc(core_result: Any) -> dict[str, Any]:
    velocity = np.asarray(core_result.cell_bedrock_velocity_m_s, dtype=np.float64)
    slowness = np.asarray(core_result.cell_bedrock_slowness_s_per_m, dtype=np.float64)
    status = np.asarray(core_result.cell_velocity_status).astype(str, copy=False)
    solved = np.isfinite(velocity)
    if not np.any(solved):
        return {'bedrock_velocity_solution_kind': 'per_cell'}
    return {
        'bedrock_velocity_solution_kind': 'per_cell',
        'cell_bedrock_velocity_min_m_s': float(np.nanmin(velocity)),
        'cell_bedrock_velocity_median_m_s': float(np.nanmedian(velocity)),
        'cell_bedrock_velocity_max_m_s': float(np.nanmax(velocity)),
        'cell_bedrock_slowness_min_s_per_m': float(np.nanmin(slowness)),
        'cell_bedrock_slowness_median_s_per_m': float(np.nanmedian(slowness)),
        'cell_bedrock_slowness_max_s_per_m': float(np.nanmax(slowness)),
        'n_solved_cells': int(np.count_nonzero(status == 'solved')),
        'n_cell_velocity_clipped_lower': int(np.count_nonzero(status == 'clipped_lower')),
        'n_cell_velocity_clipped_upper': int(np.count_nonzero(status == 'clipped_upper')),
    }


def _validate_app_model_for_external_solve(
    *,
    model: RefractionStaticModelRequest,
    design_matrix: RefractionStaticDesignMatrix,
    resolved_first_layer: ResolvedRefractionFirstLayer | None,
) -> None:
    if getattr(model, 'method', None) != 'gli_variable_thickness':
        raise RefractionStaticSolverError('model.method must be gli_variable_thickness')
    mode = getattr(model, 'bedrock_velocity_mode', None)
    if mode not in {'solve_global', 'fixed_global', 'solve_cell'}:
        raise RefractionStaticSolverError(
            'bedrock_velocity_mode must be solve_global, fixed_global, or solve_cell'
        )
    if mode != getattr(design_matrix, 'bedrock_velocity_mode', None):
        raise RefractionStaticSolverError(
            'design matrix bedrock_velocity_mode does not match model'
        )
    weathering_velocity = resolve_weathering_velocity_m_s(
        model=model,
        resolved_first_layer=resolved_first_layer,
        name='model.weathering_velocity_m_s',
    )
    if float(model.min_bedrock_velocity_m_s) <= float(weathering_velocity):
        raise RefractionStaticSolverError(
            'model.min_bedrock_velocity_m_s must be greater than '
            'model.weathering_velocity_m_s'
        )


def _validate_design_matrix_has_observable_active_columns(
    design_matrix: RefractionStaticDesignMatrix,
) -> None:
    matrix = design_matrix.matrix
    if not sparse.isspmatrix_csr(matrix):
        raise RefractionStaticSolverError('refraction design matrix must be CSR')
    if np.any(~np.isfinite(matrix.data)):
        raise RefractionStaticSolverError(
            'refraction design matrix values must be finite'
        )
    row_abs_sum = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
    if np.any(row_abs_sum == 0.0):
        bad_row = int(np.flatnonzero(row_abs_sum == 0.0)[0])
        raise RefractionStaticSolverError(
            f'refraction design matrix contains all-zero rows: row={bad_row}'
        )
    n_active_nodes = int(getattr(design_matrix, 'n_active_nodes', 0))
    if n_active_nodes <= 0:
        raise RefractionStaticSolverError(
            'refraction design matrix requires active-node columns'
        )
    col_abs_sum = np.asarray(np.abs(matrix[:, :n_active_nodes]).sum(axis=0)).ravel()
    zero_cols = np.flatnonzero(col_abs_sum == 0.0)
    if zero_cols.size == 0:
        return
    details = []
    diagnostics = {
        int(item.matrix_column): item
        for item in getattr(design_matrix, 'node_diagnostics', ())
    }
    active_node_id = np.asarray(design_matrix.active_node_id, dtype=np.int64)
    for raw_col in zero_cols.tolist():
        col = int(raw_col)
        diagnostic = diagnostics.get(col)
        if diagnostic is not None:
            details.append(
                'node_id={node_id}, endpoint_key={endpoint_key}, column={column}'.format(
                    node_id=int(diagnostic.node_id),
                    endpoint_key=str(diagnostic.endpoint_key),
                    column=col,
                )
            )
            continue
        node_id = int(active_node_id[col]) if col < active_node_id.shape[0] else col
        details.append(f'node_id={node_id}, column={col}')
    raise RefractionStaticSolverError(
        'refraction design matrix contains all-zero active-node columns: '
        + '; '.join(details)
    )


def _viewer_robust_stop_reason(reason: str) -> str:
    if reason == 'safe_rejection':
        return 'coverage_guard_no_safe_rejections'
    return reason


def _residual_stats_ms(residual_time_s: np.ndarray) -> dict[str, float]:
    residual_ms = np.asarray(residual_time_s, dtype=np.float64) * 1000.0
    median = float(np.median(residual_ms))
    return {
        'residual_rms_ms': float(np.sqrt(np.mean(residual_ms * residual_ms))),
        'residual_mad_ms': float(1.4826 * np.median(np.abs(residual_ms - median))),
        'residual_mean_ms': float(np.mean(residual_ms)),
        'residual_median_ms': median,
        'residual_p95_abs_ms': float(np.percentile(np.abs(residual_ms), 95.0)),
        'residual_max_abs_ms': float(np.max(np.abs(residual_ms))),
    }


def _optional_int_array(values: object) -> np.ndarray | None:
    if values is None:
        return None
    return np.ascontiguousarray(np.asarray(values, dtype=np.int64), dtype=np.int64)


def _optional_float_array(values: object) -> np.ndarray | None:
    if values is None:
        return None
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64), dtype=np.float64)


__all__ = [
    'RefractionStaticSolverError',
    'RefractionStaticSolverResult',
    'solve_refraction_static_bounded_ls',
]
