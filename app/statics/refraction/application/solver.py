"""Application adapter for external refraction static least-squares solving."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np
from scipy import sparse
from seis_statics.refraction.design_matrix import (
    RefractionDesignMatrixNodeDiagnostics as CoreNodeDiagnostics,
    RefractionStaticDesignMatrix as CoreDesignMatrix,
)
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
    RefractionDesignMatrixNodeDiagnostics,
    RefractionStaticDesignMatrix,
    RefractionStaticSolverResult,
    ResolvedRefractionFirstLayer,
)

BedrockVelocityMode = Literal['solve_global', 'fixed_global', 'solve_cell']

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


def solve_refraction_static_bounded_ls_from_matrix(
    *,
    matrix: sparse.csr_matrix,
    rhs_s: np.ndarray,
    active_node_id: np.ndarray,
    bedrock_slowness_col: int | None,
    row_distance_m: np.ndarray,
    observed_pick_time_s: np.ndarray,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    row_trace_index_sorted: np.ndarray | None = None,
    row_source_node_id: np.ndarray | None = None,
    row_receiver_node_id: np.ndarray | None = None,
    inactive_node_id: np.ndarray | None = None,
    bedrock_velocity_mode: BedrockVelocityMode | None = None,
    fixed_bedrock_velocity_m_s: float | None = None,
    fixed_bedrock_slowness_s_per_m: float | None = None,
    bedrock_slowness_cell_col_start: int | None = None,
    active_cell_id: np.ndarray | None = None,
    inactive_cell_id: np.ndarray | None = None,
    n_total_cells: int | None = None,
    number_of_cell_x: int | None = None,
    number_of_cell_y: int | None = None,
    row_midpoint_cell_id: np.ndarray | None = None,
    row_midpoint_cell_col: np.ndarray | None = None,
    node_diagnostics: tuple[RefractionDesignMatrixNodeDiagnostics, ...] = (),
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> RefractionStaticSolverResult:
    """Solve a pre-built sparse system by wrapping it as an external design."""
    mode = _resolve_mode(model=model, design_mode=bedrock_velocity_mode)
    design = _core_design_from_matrix(
        matrix=matrix,
        rhs_s=rhs_s,
        active_node_id=active_node_id,
        inactive_node_id=inactive_node_id,
        bedrock_slowness_col=bedrock_slowness_col,
        row_distance_m=row_distance_m,
        observed_pick_time_s=observed_pick_time_s,
        row_trace_index_sorted=row_trace_index_sorted,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        mode=mode,
        fixed_bedrock_velocity_m_s=fixed_bedrock_velocity_m_s,
        fixed_bedrock_slowness_s_per_m=fixed_bedrock_slowness_s_per_m,
        bedrock_slowness_cell_col_start=bedrock_slowness_cell_col_start,
        active_cell_id=active_cell_id,
        inactive_cell_id=inactive_cell_id,
        n_total_cells=n_total_cells,
        number_of_cell_x=number_of_cell_x,
        number_of_cell_y=number_of_cell_y,
        row_midpoint_cell_id=row_midpoint_cell_id,
        row_midpoint_cell_col=row_midpoint_cell_col,
        node_diagnostics=node_diagnostics,
        model=model,
    )
    return solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=model,
        solver=solver,
        resolved_first_layer=resolved_first_layer,
    )


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


def _core_design_from_matrix(
    *,
    matrix: sparse.csr_matrix,
    rhs_s: np.ndarray,
    active_node_id: np.ndarray,
    inactive_node_id: np.ndarray | None,
    bedrock_slowness_col: int | None,
    row_distance_m: np.ndarray,
    observed_pick_time_s: np.ndarray,
    row_trace_index_sorted: np.ndarray | None,
    row_source_node_id: np.ndarray | None,
    row_receiver_node_id: np.ndarray | None,
    mode: BedrockVelocityMode,
    fixed_bedrock_velocity_m_s: float | None,
    fixed_bedrock_slowness_s_per_m: float | None,
    bedrock_slowness_cell_col_start: int | None,
    active_cell_id: np.ndarray | None,
    inactive_cell_id: np.ndarray | None,
    n_total_cells: int | None,
    number_of_cell_x: int | None,
    number_of_cell_y: int | None,
    row_midpoint_cell_id: np.ndarray | None,
    row_midpoint_cell_col: np.ndarray | None,
    node_diagnostics: tuple[RefractionDesignMatrixNodeDiagnostics, ...],
    model: RefractionStaticModelRequest,
) -> CoreDesignMatrix:
    if mode == 'solve_global' and bedrock_slowness_col is None:
        raise RefractionStaticSolverError(
            'solve_global mode requires a bedrock slowness column'
        )
    if mode != 'solve_global' and bedrock_slowness_col is not None:
        raise RefractionStaticSolverError(
            f'{mode} mode must not include a global bedrock slowness column'
        )
    if not sparse.isspmatrix_csr(matrix):
        raise RefractionStaticSolverError('refraction design matrix must be CSR')
    matrix = matrix.copy()
    matrix.sort_indices()
    if np.any(~np.isfinite(matrix.data)):
        raise RefractionStaticSolverError('refraction design matrix values must be finite')
    rhs = _coerce_1d_float(rhs_s, name='rhs_s')
    observed = _coerce_1d_float(observed_pick_time_s, name='observed_pick_time_s')
    distance = _coerce_1d_float(row_distance_m, name='row_distance_m')
    n_rows, n_cols = matrix.shape
    if rhs.shape != (n_rows,):
        raise RefractionStaticSolverError('rhs_s shape mismatch')
    if observed.shape != (n_rows,):
        raise RefractionStaticSolverError('observed_pick_time_s shape mismatch')
    if distance.shape != (n_rows,):
        raise RefractionStaticSolverError('row_distance_m shape mismatch')
    if n_rows <= 0:
        raise RefractionStaticSolverError(
            'refraction solver system requires at least one observation row'
        )
    nodes = _coerce_1d_int(active_node_id, name='active_node_id')
    inactive_nodes = (
        np.empty(0, dtype=np.int64)
        if inactive_node_id is None
        else _coerce_1d_int(inactive_node_id, name='inactive_node_id')
    )
    source_node = _coerce_optional_row_node(
        row_source_node_id,
        fallback_nodes=nodes,
        n_rows=n_rows,
        name='row_source_node_id',
    )
    receiver_node = _coerce_optional_row_node(
        row_receiver_node_id,
        fallback_nodes=nodes,
        n_rows=n_rows,
        name='row_receiver_node_id',
    )
    trace_index = (
        np.arange(n_rows, dtype=np.int64)
        if row_trace_index_sorted is None
        else _coerce_1d_int(
            row_trace_index_sorted,
            name='row_trace_index_sorted',
            expected_shape=(n_rows,),
        )
    )
    if np.unique(trace_index).shape[0] != n_rows:
        raise RefractionStaticSolverError('row_trace_index_sorted must be unique')
    n_traces = int(np.max(trace_index)) + 1 if trace_index.size else 0
    node_id_to_col = {int(node): index for index, node in enumerate(nodes.tolist())}
    source_col = _node_cols(source_node, node_id_to_col=node_id_to_col)
    receiver_col = _node_cols(receiver_node, node_id_to_col=node_id_to_col)
    design_qc: dict[str, Any] = {
        'n_traces': n_traces,
        'bedrock_velocity_mode': mode,
    }
    active_cells = _optional_int_array(active_cell_id)
    inactive_cells = _optional_int_array(inactive_cell_id)
    cell_id_to_col = None
    if mode == 'solve_cell':
        if active_cells is None:
            raise RefractionStaticSolverError('solve_cell mode requires active_cell_id')
        if n_total_cells is None:
            raise RefractionStaticSolverError('solve_cell mode requires n_total_cells')
        if bedrock_slowness_cell_col_start is None:
            raise RefractionStaticSolverError(
                'solve_cell mode requires cell slowness columns'
            )
        cell_id_to_col = {
            int(cell_id): int(bedrock_slowness_cell_col_start) + index
            for index, cell_id in enumerate(active_cells.tolist())
            }
    if mode == 'fixed_global':
        if fixed_bedrock_velocity_m_s is None:
            fixed_bedrock_velocity_m_s = float(model.bedrock_velocity_m_s)
        if fixed_bedrock_slowness_s_per_m is None:
            fixed_bedrock_slowness_s_per_m = float(1.0 / fixed_bedrock_velocity_m_s)
    return CoreDesignMatrix(
        matrix=matrix,
        rhs_s=np.ascontiguousarray(rhs, dtype=np.float64),
        observed_pick_time_s=np.ascontiguousarray(observed, dtype=np.float64),
        row_trace_index_sorted=np.ascontiguousarray(trace_index, dtype=np.int64),
        row_source_node_id=np.ascontiguousarray(source_node, dtype=np.int64),
        row_receiver_node_id=np.ascontiguousarray(receiver_node, dtype=np.int64),
        row_distance_m=np.ascontiguousarray(distance, dtype=np.float64),
        active_node_id=np.ascontiguousarray(nodes, dtype=np.int64),
        inactive_node_id=np.ascontiguousarray(inactive_nodes, dtype=np.int64),
        node_id_to_col=node_id_to_col,
        source_node_col=np.ascontiguousarray(source_col, dtype=np.int64),
        receiver_node_col=np.ascontiguousarray(receiver_col, dtype=np.int64),
        bedrock_slowness_col=bedrock_slowness_col,
        bedrock_velocity_mode=mode,
        fixed_bedrock_velocity_m_s=fixed_bedrock_velocity_m_s,
        fixed_bedrock_slowness_s_per_m=fixed_bedrock_slowness_s_per_m,
        n_total_nodes=int(nodes.shape[0] + inactive_nodes.shape[0]),
        n_active_nodes=int(nodes.shape[0]),
        min_observations_per_node=1,
        node_observation_count=np.zeros(nodes.shape, dtype=np.int64),
        low_fold_node_id=np.empty(0, dtype=np.int64),
        n_observations_rejected_by_low_fold_node=0,
        n_observations=n_rows,
        n_parameters=n_cols,
        qc=design_qc,
        node_diagnostics=tuple(_core_node_diagnostics(node_diagnostics)),
        design_matrix_qc=dict(design_qc),
        diagnostics_context=None,
        bedrock_slowness_cell_col_start=bedrock_slowness_cell_col_start,
        active_cell_id=active_cells,
        inactive_cell_id=inactive_cells,
        cell_id_to_col=cell_id_to_col,
        row_midpoint_cell_id=_optional_int_array(row_midpoint_cell_id),
        row_midpoint_cell_col=_optional_int_array(row_midpoint_cell_col),
        cell_assignment_mode='midpoint' if mode == 'solve_cell' else None,
        n_total_cells=n_total_cells,
        n_active_cells=None if active_cells is None else int(active_cells.shape[0]),
        n_inactive_cells=None if inactive_cells is None else int(inactive_cells.shape[0]),
        number_of_cell_x=number_of_cell_x,
        number_of_cell_y=number_of_cell_y,
        rejection_reason_sorted=None,
    )


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


def _resolve_mode(
    *,
    model: RefractionStaticModelRequest,
    design_mode: BedrockVelocityMode | None,
) -> BedrockVelocityMode:
    mode = getattr(model, 'bedrock_velocity_mode', None)
    if mode not in {'solve_global', 'fixed_global', 'solve_cell'}:
        raise RefractionStaticSolverError(
            'bedrock_velocity_mode must be solve_global, fixed_global, or solve_cell'
        )
    if design_mode is not None and design_mode != mode:
        raise RefractionStaticSolverError(
            'design matrix bedrock_velocity_mode does not match model'
        )
    return mode


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


def _coerce_1d_float(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticSolverError(f'{name} must be numeric') from exc
    if out.ndim != 1:
        raise RefractionStaticSolverError(f'{name} must be one-dimensional')
    if expected_shape is not None and out.shape != expected_shape:
        raise RefractionStaticSolverError(f'{name} shape mismatch')
    if np.any(~np.isfinite(out)):
        raise RefractionStaticSolverError(f'{name} must contain only finite values')
    return np.ascontiguousarray(out, dtype=np.float64)


def _coerce_1d_int(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticSolverError(f'{name} must contain integers') from exc
    if out.ndim != 1:
        raise RefractionStaticSolverError(f'{name} must be one-dimensional')
    if expected_shape is not None and out.shape != expected_shape:
        raise RefractionStaticSolverError(f'{name} shape mismatch')
    return np.ascontiguousarray(out, dtype=np.int64)


def _optional_int_array(values: object) -> np.ndarray | None:
    if values is None:
        return None
    return np.ascontiguousarray(np.asarray(values, dtype=np.int64), dtype=np.int64)


def _optional_float_array(values: object) -> np.ndarray | None:
    if values is None:
        return None
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64), dtype=np.float64)


def _coerce_optional_row_node(
    values: object,
    *,
    fallback_nodes: np.ndarray,
    n_rows: int,
    name: str,
) -> np.ndarray:
    if values is not None:
        return _coerce_1d_int(values, name=name, expected_shape=(n_rows,))
    if fallback_nodes.size == 0:
        raise RefractionStaticSolverError(f'{name} fallback requires active nodes')
    index = np.minimum(np.arange(n_rows), fallback_nodes.shape[0] - 1)
    return np.ascontiguousarray(fallback_nodes[index], dtype=np.int64)


def _node_cols(
    row_node_id: np.ndarray,
    *,
    node_id_to_col: dict[int, int],
) -> np.ndarray:
    cols = np.full(row_node_id.shape, -1, dtype=np.int64)
    for index, raw_node in enumerate(row_node_id.tolist()):
        cols[index] = int(node_id_to_col.get(int(raw_node), -1))
    return cols


def _core_node_diagnostics(
    diagnostics: tuple[RefractionDesignMatrixNodeDiagnostics, ...],
) -> tuple[CoreNodeDiagnostics, ...]:
    out: list[CoreNodeDiagnostics] = []
    for item in diagnostics:
        if isinstance(item, CoreNodeDiagnostics):
            out.append(item)
            continue
        out.append(
            CoreNodeDiagnostics(
                node_id=item.node_id,
                matrix_column=item.matrix_column,
                endpoint_kind=item.endpoint_kind,
                endpoint_key=item.endpoint_key,
                source_endpoint_key=item.source_endpoint_key,
                receiver_endpoint_key=item.receiver_endpoint_key,
                n_rows_pre_filter=item.n_rows_pre_filter,
                n_rows_post_filter=item.n_rows_post_filter,
                n_nonzero_entries=item.n_nonzero_entries,
                active=item.active,
                status=item.status,
                reason=item.reason,
                first_trace_indices_pre_filter=item.first_trace_indices_pre_filter,
            )
        )
    return tuple(out)


__all__ = [
    'BedrockVelocityMode',
    'RefractionStaticSolverError',
    'RefractionStaticSolverResult',
    'solve_refraction_static_bounded_ls',
    'solve_refraction_static_bounded_ls_from_matrix',
]
