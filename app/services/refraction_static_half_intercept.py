"""Source/receiver half-intercept time model for GLI refraction statics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.core.state import AppState
from app.services.refraction_static_bedrock import (
    RefractionBedrockSlownessResult,
    estimate_global_bedrock_slowness_from_input_model,
)
from app.services.refraction_static_design_matrix import (
    RefractionStaticDesignMatrix,
    build_refraction_static_design_matrix,
)
from app.services.refraction_static_inputs import (
    RefractionStaticInputModel,
    build_refraction_static_input_model,
)
from app.services.refraction_static_solver import (
    RefractionStaticSolverResult,
    solve_refraction_static_bounded_ls,
)

REFRACTION_HALF_INTERCEPT_QC_JSON_NAME = 'refraction_half_intercept_qc.json'
REFRACTION_HALF_INTERCEPT_NODES_CSV_NAME = 'refraction_half_intercept_nodes.csv'
REFRACTION_HALF_INTERCEPT_SOURCES_CSV_NAME = 'refraction_half_intercept_sources.csv'
REFRACTION_HALF_INTERCEPT_RECEIVERS_CSV_NAME = (
    'refraction_half_intercept_receivers.csv'
)
REFRACTION_HALF_INTERCEPT_TRACE_PREVIEW_CSV_NAME = (
    'refraction_half_intercept_trace_preview.csv'
)

_BOUND_TOL = 1.0e-10
_STATUS_DTYPE = '<U16'
_ENDPOINT_KEY_DTYPE = object

_NODE_COLUMNS = (
    'node_id',
    'node_kind',
    'x_m',
    'y_m',
    'elevation_m',
    'half_intercept_time_ms',
    'solution_status',
    'pick_count',
    'used_pick_count',
    'rejected_pick_count',
    'residual_mean_ms',
    'residual_median_ms',
    'residual_rms_ms',
    'residual_mad_ms',
    'residual_max_abs_ms',
)
_SOURCE_COLUMNS = (
    'source_endpoint_key',
    'source_id',
    'source_node_id',
    'source_x_m',
    'source_y_m',
    'source_elevation_m',
    'half_intercept_time_ms',
    'solution_status',
    'pick_count',
    'residual_rms_ms',
)
_RECEIVER_COLUMNS = (
    'receiver_endpoint_key',
    'receiver_id',
    'receiver_node_id',
    'receiver_x_m',
    'receiver_y_m',
    'receiver_elevation_m',
    'half_intercept_time_ms',
    'solution_status',
    'pick_count',
    'residual_rms_ms',
)
_TRACE_PREVIEW_COLUMNS = (
    'sorted_trace_index',
    'valid_observation',
    'used_observation',
    'source_node_id',
    'receiver_node_id',
    'source_half_intercept_time_ms',
    'receiver_half_intercept_time_ms',
    'intercept_time_sum_ms',
    'bedrock_moveout_time_ms',
    'observed_pick_time_ms',
    'estimated_first_break_time_ms',
    'first_break_residual_ms',
)


class RefractionHalfInterceptTimeError(ValueError):
    """Raised when half-intercept time outputs cannot be built."""


@dataclass(frozen=True)
class RefractionHalfInterceptTimeResult:
    """Full node, endpoint, trace-order, and QC output for GLI half-intercepts."""

    bedrock_velocity_mode: Literal['solve_global', 'fixed_global']
    bedrock_slowness_s_per_m: float
    bedrock_velocity_m_s: float
    weathering_velocity_m_s: float

    node_id: np.ndarray
    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_elevation_m: np.ndarray
    node_kind: np.ndarray

    node_half_intercept_time_s: np.ndarray
    node_half_intercept_time_ms: np.ndarray
    node_solution_status: np.ndarray

    node_pick_count: np.ndarray
    node_used_pick_count: np.ndarray
    node_rejected_pick_count: np.ndarray
    node_residual_mean_s: np.ndarray
    node_residual_median_s: np.ndarray
    node_residual_rms_s: np.ndarray
    node_residual_mad_s: np.ndarray
    node_residual_max_abs_s: np.ndarray

    source_endpoint_key: np.ndarray
    source_id: np.ndarray
    source_node_id: np.ndarray
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    source_elevation_m: np.ndarray
    source_half_intercept_time_s: np.ndarray
    source_solution_status: np.ndarray
    source_pick_count: np.ndarray
    source_residual_rms_s: np.ndarray

    receiver_endpoint_key: np.ndarray
    receiver_id: np.ndarray
    receiver_node_id: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    receiver_elevation_m: np.ndarray
    receiver_half_intercept_time_s: np.ndarray
    receiver_solution_status: np.ndarray
    receiver_pick_count: np.ndarray
    receiver_residual_rms_s: np.ndarray

    sorted_trace_index: np.ndarray
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_elevation_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    source_half_intercept_time_s_sorted: np.ndarray
    receiver_half_intercept_time_s_sorted: np.ndarray
    estimated_intercept_time_sum_s_sorted: np.ndarray
    estimated_bedrock_moveout_time_s_sorted: np.ndarray
    estimated_first_break_time_s_sorted: np.ndarray
    first_break_residual_s_sorted: np.ndarray
    valid_observation_mask_sorted: np.ndarray
    used_observation_mask_sorted: np.ndarray

    row_trace_index_sorted: np.ndarray
    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    row_distance_m: np.ndarray
    observed_pick_time_s: np.ndarray
    modeled_pick_time_s: np.ndarray
    residual_time_s: np.ndarray
    used_row_mask: np.ndarray
    rejected_by_robust_mask: np.ndarray

    qc: dict[str, Any]


@dataclass(frozen=True)
class _ValidatedInputs:
    n_traces: int
    node_id: np.ndarray
    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_elevation_m: np.ndarray
    node_kind: np.ndarray
    sorted_trace_index: np.ndarray
    pick_time_s_sorted: np.ndarray
    valid_observation_mask_sorted: np.ndarray
    source_id_sorted: np.ndarray
    receiver_id_sorted: np.ndarray
    source_x_m_sorted: np.ndarray
    source_y_m_sorted: np.ndarray
    source_elevation_m_sorted: np.ndarray
    receiver_x_m_sorted: np.ndarray
    receiver_y_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray
    distance_m_sorted: np.ndarray
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray


@dataclass(frozen=True)
class _ValidatedRows:
    row_trace_index_sorted: np.ndarray
    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    row_distance_m: np.ndarray
    observed_pick_time_s: np.ndarray
    modeled_pick_time_s: np.ndarray
    residual_time_s: np.ndarray
    used_row_mask: np.ndarray
    rejected_by_robust_mask: np.ndarray


@dataclass(frozen=True)
class _NodeAggregation:
    pick_count: np.ndarray
    used_pick_count: np.ndarray
    rejected_pick_count: np.ndarray
    residual_mean_s: np.ndarray
    residual_median_s: np.ndarray
    residual_rms_s: np.ndarray
    residual_mad_s: np.ndarray
    residual_max_abs_s: np.ndarray


@dataclass(frozen=True)
class _EndpointResult:
    endpoint_key: np.ndarray
    endpoint_id: np.ndarray
    node_id: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    elevation_m: np.ndarray
    half_intercept_time_s: np.ndarray
    solution_status: np.ndarray
    pick_count: np.ndarray
    residual_rms_s: np.ndarray


def estimate_refraction_half_intercept_times_from_first_breaks(
    *,
    req: RefractionStaticApplyRequest,
    state: AppState,
    job_dir: Path | None = None,
) -> RefractionHalfInterceptTimeResult:
    """Build inputs, solve the GLI system, and emit the half-intercept model."""
    try:
        input_model = build_refraction_static_input_model(
            req=req,
            state=state,
            job_dir=job_dir,
        )
        if req.model.bedrock_velocity_mode == 'solve_global':
            bedrock_result = estimate_global_bedrock_slowness_from_input_model(
                input_model=input_model,
                model=req.model,
                solver=req.solver,
                job_dir=job_dir,
                include_debug_objects=True,
            )
            return build_refraction_half_intercept_time_model_from_bedrock_result(
                bedrock_result=bedrock_result,
                job_dir=job_dir,
            )
        design_matrix = build_refraction_static_design_matrix(
            input_model=input_model,
            model=req.model,
        )
        solver_result = solve_refraction_static_bounded_ls(
            design_matrix=design_matrix,
            model=req.model,
            solver=req.solver,
        )
        return build_refraction_half_intercept_time_model(
            input_model=input_model,
            design_matrix=design_matrix,
            solver_result=solver_result,
            weathering_velocity_m_s=req.model.weathering_velocity_m_s,
            min_picks_per_node=req.solver.min_picks_per_node,
            job_dir=job_dir,
        )
    except RefractionHalfInterceptTimeError:
        raise
    except ValueError as exc:
        raise RefractionHalfInterceptTimeError(str(exc)) from exc


def build_refraction_half_intercept_time_model_from_bedrock_result(
    *,
    bedrock_result: RefractionBedrockSlownessResult,
    job_dir: Path | None = None,
) -> RefractionHalfInterceptTimeResult:
    """Build the half-intercept model from a P0-06 bedrock slowness result."""
    if bedrock_result.input_model is None:
        raise RefractionHalfInterceptTimeError(
            'bedrock_result.input_model is required'
        )
    if bedrock_result.design_matrix is None:
        raise RefractionHalfInterceptTimeError(
            'bedrock_result.design_matrix is required'
        )
    return build_refraction_half_intercept_time_model(
        input_model=bedrock_result.input_model,
        design_matrix=bedrock_result.design_matrix,
        solver_result=bedrock_result.solver_result,
        weathering_velocity_m_s=bedrock_result.weathering_velocity_m_s,
        job_dir=job_dir,
    )


def build_refraction_half_intercept_time_model(
    *,
    input_model: RefractionStaticInputModel,
    design_matrix: RefractionStaticDesignMatrix,
    solver_result: RefractionStaticSolverResult,
    weathering_velocity_m_s: float | None = None,
    min_picks_per_node: int | None = None,
    job_dir: Path | None = None,
) -> RefractionHalfInterceptTimeResult:
    """Build full node, endpoint, trace-order, and QC half-intercept outputs."""
    inputs = _validate_input_model(input_model)
    rows = _validate_rows(
        inputs=inputs,
        design_matrix=design_matrix,
        solver_result=solver_result,
    )
    mode = _validate_velocity_mode(solver_result.bedrock_velocity_mode)
    bedrock_slowness = _positive_finite(
        solver_result.bedrock_slowness_s_per_m,
        name='bedrock_slowness_s_per_m',
    )
    bedrock_velocity = _positive_finite(
        solver_result.bedrock_velocity_m_s,
        name='bedrock_velocity_m_s',
    )
    derived_velocity = 1.0 / bedrock_slowness
    velocity_tol = max(1.0e-6, abs(derived_velocity) * 1.0e-6)
    if abs(derived_velocity - bedrock_velocity) > velocity_tol:
        raise RefractionHalfInterceptTimeError(
            'bedrock_velocity_m_s does not match bedrock_slowness_s_per_m'
        )
    weathering_velocity = _resolve_weathering_velocity(
        weathering_velocity_m_s,
        solver_result=solver_result,
    )
    if bedrock_velocity <= weathering_velocity:
        raise RefractionHalfInterceptTimeError(
            'bedrock_velocity_m_s must be greater than weathering_velocity_m_s'
        )
    min_fold = _resolve_min_picks_per_node(
        min_picks_per_node,
        solver_result=solver_result,
    )

    node_pos = {int(node): idx for idx, node in enumerate(inputs.node_id.tolist())}
    _validate_node_references(
        inputs=inputs,
        rows=rows,
        design_matrix=design_matrix,
        solver_result=solver_result,
        node_pos=node_pos,
    )

    node_half_s, node_status = _build_node_solution_arrays(
        inputs=inputs,
        design_matrix=design_matrix,
        solver_result=solver_result,
        rows=rows,
        node_pos=node_pos,
        min_picks_per_node=min_fold,
    )
    node_aggregation = _aggregate_node_rows(
        inputs.node_id,
        rows=rows,
        node_pos=node_pos,
    )
    source_endpoint = _build_endpoint_result(
        side='source',
        inputs=inputs,
        rows=rows,
        node_pos=node_pos,
        node_half_s=node_half_s,
        node_status=node_status,
    )
    receiver_endpoint = _build_endpoint_result(
        side='receiver',
        inputs=inputs,
        rows=rows,
        node_pos=node_pos,
        node_half_s=node_half_s,
        node_status=node_status,
    )
    (
        source_half_sorted,
        receiver_half_sorted,
        intercept_sum_sorted,
        moveout_sorted,
        estimated_first_break_sorted,
        residual_sorted,
        used_observation_sorted,
    ) = _build_trace_order_arrays(
        inputs=inputs,
        rows=rows,
        node_pos=node_pos,
        node_half_s=node_half_s,
        bedrock_velocity_m_s=bedrock_velocity,
    )

    qc = _build_qc(
        mode=mode,
        bedrock_slowness_s_per_m=bedrock_slowness,
        bedrock_velocity_m_s=bedrock_velocity,
        weathering_velocity_m_s=weathering_velocity,
        inputs=inputs,
        rows=rows,
        node_half_s=node_half_s,
        node_status=node_status,
        node_pick_count=node_aggregation.pick_count,
        node_used_pick_count=node_aggregation.used_pick_count,
        source_endpoint=source_endpoint,
        receiver_endpoint=receiver_endpoint,
        solver_result=solver_result,
        source_receiver_linkage_used=bool(
            getattr(input_model, 'qc', {}).get('linkage_used', False)
        ),
    )

    result = RefractionHalfInterceptTimeResult(
        bedrock_velocity_mode=mode,
        bedrock_slowness_s_per_m=bedrock_slowness,
        bedrock_velocity_m_s=bedrock_velocity,
        weathering_velocity_m_s=weathering_velocity,
        node_id=inputs.node_id,
        node_x_m=inputs.node_x_m,
        node_y_m=inputs.node_y_m,
        node_elevation_m=inputs.node_elevation_m,
        node_kind=inputs.node_kind,
        node_half_intercept_time_s=node_half_s,
        node_half_intercept_time_ms=np.ascontiguousarray(
            node_half_s * 1000.0,
            dtype=np.float64,
        ),
        node_solution_status=node_status,
        node_pick_count=node_aggregation.pick_count,
        node_used_pick_count=node_aggregation.used_pick_count,
        node_rejected_pick_count=node_aggregation.rejected_pick_count,
        node_residual_mean_s=node_aggregation.residual_mean_s,
        node_residual_median_s=node_aggregation.residual_median_s,
        node_residual_rms_s=node_aggregation.residual_rms_s,
        node_residual_mad_s=node_aggregation.residual_mad_s,
        node_residual_max_abs_s=node_aggregation.residual_max_abs_s,
        source_endpoint_key=source_endpoint.endpoint_key,
        source_id=source_endpoint.endpoint_id,
        source_node_id=source_endpoint.node_id,
        source_x_m=source_endpoint.x_m,
        source_y_m=source_endpoint.y_m,
        source_elevation_m=source_endpoint.elevation_m,
        source_half_intercept_time_s=source_endpoint.half_intercept_time_s,
        source_solution_status=source_endpoint.solution_status,
        source_pick_count=source_endpoint.pick_count,
        source_residual_rms_s=source_endpoint.residual_rms_s,
        receiver_endpoint_key=receiver_endpoint.endpoint_key,
        receiver_id=receiver_endpoint.endpoint_id,
        receiver_node_id=receiver_endpoint.node_id,
        receiver_x_m=receiver_endpoint.x_m,
        receiver_y_m=receiver_endpoint.y_m,
        receiver_elevation_m=receiver_endpoint.elevation_m,
        receiver_half_intercept_time_s=receiver_endpoint.half_intercept_time_s,
        receiver_solution_status=receiver_endpoint.solution_status,
        receiver_pick_count=receiver_endpoint.pick_count,
        receiver_residual_rms_s=receiver_endpoint.residual_rms_s,
        sorted_trace_index=inputs.sorted_trace_index,
        source_endpoint_key_sorted=inputs.source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=inputs.receiver_endpoint_key_sorted,
        source_elevation_m_sorted=inputs.source_elevation_m_sorted,
        receiver_elevation_m_sorted=inputs.receiver_elevation_m_sorted,
        source_node_id_sorted=inputs.source_node_id_sorted,
        receiver_node_id_sorted=inputs.receiver_node_id_sorted,
        source_half_intercept_time_s_sorted=source_half_sorted,
        receiver_half_intercept_time_s_sorted=receiver_half_sorted,
        estimated_intercept_time_sum_s_sorted=intercept_sum_sorted,
        estimated_bedrock_moveout_time_s_sorted=moveout_sorted,
        estimated_first_break_time_s_sorted=estimated_first_break_sorted,
        first_break_residual_s_sorted=residual_sorted,
        valid_observation_mask_sorted=inputs.valid_observation_mask_sorted,
        used_observation_mask_sorted=used_observation_sorted,
        row_trace_index_sorted=rows.row_trace_index_sorted,
        row_source_node_id=rows.row_source_node_id,
        row_receiver_node_id=rows.row_receiver_node_id,
        row_distance_m=rows.row_distance_m,
        observed_pick_time_s=rows.observed_pick_time_s,
        modeled_pick_time_s=rows.modeled_pick_time_s,
        residual_time_s=rows.residual_time_s,
        used_row_mask=rows.used_row_mask,
        rejected_by_robust_mask=rows.rejected_by_robust_mask,
        qc=qc,
    )
    if job_dir is not None:
        write_refraction_half_intercept_artifacts(Path(job_dir), result)
    return result


def write_refraction_half_intercept_artifacts(
    job_dir: Path,
    result: RefractionHalfInterceptTimeResult,
) -> dict[str, Path]:
    """Write half-intercept QC JSON and CSV tables."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / REFRACTION_HALF_INTERCEPT_QC_JSON_NAME
    node_path = root / REFRACTION_HALF_INTERCEPT_NODES_CSV_NAME
    source_path = root / REFRACTION_HALF_INTERCEPT_SOURCES_CSV_NAME
    receiver_path = root / REFRACTION_HALF_INTERCEPT_RECEIVERS_CSV_NAME
    trace_path = root / REFRACTION_HALF_INTERCEPT_TRACE_PREVIEW_CSV_NAME
    _write_json_atomic(qc_path, result.qc)
    _write_csv_atomic(node_path, _node_rows(result), _NODE_COLUMNS)
    _write_csv_atomic(source_path, _source_rows(result), _SOURCE_COLUMNS)
    _write_csv_atomic(receiver_path, _receiver_rows(result), _RECEIVER_COLUMNS)
    _write_csv_atomic(trace_path, _trace_preview_rows(result), _TRACE_PREVIEW_COLUMNS)
    return {
        'qc_json': qc_path,
        'nodes_csv': node_path,
        'sources_csv': source_path,
        'receivers_csv': receiver_path,
        'trace_preview_csv': trace_path,
    }


def _validate_input_model(input_model: RefractionStaticInputModel) -> _ValidatedInputs:
    n_traces = _positive_int(_required(input_model, 'n_traces'), name='input_model.n_traces')
    endpoint_table = _required(input_model, 'endpoint_table')
    node_id = _coerce_1d_integer(
        _required(endpoint_table, 'node_id'),
        name='input_model.endpoint_table.node_id',
    )
    n_nodes = int(node_id.shape[0])
    if n_nodes <= 0:
        raise RefractionHalfInterceptTimeError(
            'input_model.endpoint_table.node_id must contain at least one node'
        )
    if np.unique(node_id).shape[0] != n_nodes:
        raise RefractionHalfInterceptTimeError('node_id values must be unique')

    trace_shape = (n_traces,)
    node_shape = (n_nodes,)
    return _ValidatedInputs(
        n_traces=n_traces,
        node_id=node_id,
        node_x_m=_coerce_1d_float(
            _required(input_model, 'node_x_m'),
            name='input_model.node_x_m',
            expected_shape=node_shape,
            allow_nan=True,
        ),
        node_y_m=_coerce_1d_float(
            _required(input_model, 'node_y_m'),
            name='input_model.node_y_m',
            expected_shape=node_shape,
            allow_nan=True,
        ),
        node_elevation_m=_coerce_1d_float(
            _required(input_model, 'node_elevation_m'),
            name='input_model.node_elevation_m',
            expected_shape=node_shape,
            allow_nan=True,
        ),
        node_kind=_coerce_1d_string(
            _required(input_model, 'node_kind'),
            name='input_model.node_kind',
            expected_shape=node_shape,
        ),
        sorted_trace_index=_coerce_1d_integer(
            _required(input_model, 'sorted_trace_index'),
            name='input_model.sorted_trace_index',
            expected_shape=trace_shape,
        ),
        pick_time_s_sorted=_coerce_1d_float(
            _required(input_model, 'pick_time_s_sorted'),
            name='input_model.pick_time_s_sorted',
            expected_shape=trace_shape,
            allow_nan=True,
        ),
        valid_observation_mask_sorted=_coerce_1d_bool(
            _required(input_model, 'valid_observation_mask_sorted'),
            name='input_model.valid_observation_mask_sorted',
            expected_shape=trace_shape,
        ),
        source_id_sorted=_coerce_1d_integer(
            _required(input_model, 'source_id_sorted'),
            name='input_model.source_id_sorted',
            expected_shape=trace_shape,
        ),
        receiver_id_sorted=_coerce_1d_integer(
            _required(input_model, 'receiver_id_sorted'),
            name='input_model.receiver_id_sorted',
            expected_shape=trace_shape,
        ),
        source_x_m_sorted=_coerce_1d_float(
            _required(input_model, 'source_x_m_sorted'),
            name='input_model.source_x_m_sorted',
            expected_shape=trace_shape,
            allow_nan=True,
        ),
        source_y_m_sorted=_coerce_1d_float(
            _required(input_model, 'source_y_m_sorted'),
            name='input_model.source_y_m_sorted',
            expected_shape=trace_shape,
            allow_nan=True,
        ),
        source_elevation_m_sorted=_coerce_1d_float(
            _required(input_model, 'source_elevation_m_sorted'),
            name='input_model.source_elevation_m_sorted',
            expected_shape=trace_shape,
            allow_nan=True,
        ),
        receiver_x_m_sorted=_coerce_1d_float(
            _required(input_model, 'receiver_x_m_sorted'),
            name='input_model.receiver_x_m_sorted',
            expected_shape=trace_shape,
            allow_nan=True,
        ),
        receiver_y_m_sorted=_coerce_1d_float(
            _required(input_model, 'receiver_y_m_sorted'),
            name='input_model.receiver_y_m_sorted',
            expected_shape=trace_shape,
            allow_nan=True,
        ),
        receiver_elevation_m_sorted=_coerce_1d_float(
            _required(input_model, 'receiver_elevation_m_sorted'),
            name='input_model.receiver_elevation_m_sorted',
            expected_shape=trace_shape,
            allow_nan=True,
        ),
        distance_m_sorted=_coerce_1d_float(
            _required(input_model, 'distance_m_sorted'),
            name='input_model.distance_m_sorted',
            expected_shape=trace_shape,
            allow_nan=True,
        ),
        source_endpoint_key_sorted=_coerce_1d_string(
            _required(input_model, 'source_endpoint_key_sorted'),
            name='input_model.source_endpoint_key_sorted',
            expected_shape=trace_shape,
        ),
        receiver_endpoint_key_sorted=_coerce_1d_string(
            _required(input_model, 'receiver_endpoint_key_sorted'),
            name='input_model.receiver_endpoint_key_sorted',
            expected_shape=trace_shape,
        ),
        source_node_id_sorted=_coerce_1d_integer(
            _required(input_model, 'source_node_id_sorted'),
            name='input_model.source_node_id_sorted',
            expected_shape=trace_shape,
        ),
        receiver_node_id_sorted=_coerce_1d_integer(
            _required(input_model, 'receiver_node_id_sorted'),
            name='input_model.receiver_node_id_sorted',
            expected_shape=trace_shape,
        ),
    )


def _validate_rows(
    *,
    inputs: _ValidatedInputs,
    design_matrix: RefractionStaticDesignMatrix,
    solver_result: RefractionStaticSolverResult,
) -> _ValidatedRows:
    observed_design = _coerce_1d_float(
        _required(design_matrix, 'observed_pick_time_s'),
        name='design_matrix.observed_pick_time_s',
    )
    n_rows = int(observed_design.shape[0])
    expected_shape = (n_rows,)
    design_trace = _coerce_1d_integer(
        _required(design_matrix, 'row_trace_index_sorted'),
        name='design_matrix.row_trace_index_sorted',
        expected_shape=expected_shape,
    )
    if np.any((design_trace < 0) | (design_trace >= inputs.n_traces)):
        raise RefractionHalfInterceptTimeError(
            'design_matrix.row_trace_index_sorted contains out-of-range trace indices'
        )
    if np.unique(design_trace).shape[0] != n_rows:
        raise RefractionHalfInterceptTimeError(
            'design_matrix.row_trace_index_sorted must be unique'
        )
    expected_trace = np.flatnonzero(inputs.valid_observation_mask_sorted).astype(
        np.int64,
        copy=False,
    )
    if not np.array_equal(design_trace, expected_trace):
        raise RefractionHalfInterceptTimeError(
            'design_matrix rows do not match input_model.valid_observation_mask_sorted'
        )

    design_source = _coerce_1d_integer(
        _required(design_matrix, 'row_source_node_id'),
        name='design_matrix.row_source_node_id',
        expected_shape=expected_shape,
    )
    design_receiver = _coerce_1d_integer(
        _required(design_matrix, 'row_receiver_node_id'),
        name='design_matrix.row_receiver_node_id',
        expected_shape=expected_shape,
    )
    design_distance = _coerce_1d_float(
        _required(design_matrix, 'row_distance_m'),
        name='design_matrix.row_distance_m',
        expected_shape=expected_shape,
    )
    if not np.array_equal(design_source, inputs.source_node_id_sorted[design_trace]):
        raise RefractionHalfInterceptTimeError(
            'design_matrix source rows do not match input_model.source_node_id_sorted'
        )
    if not np.array_equal(design_receiver, inputs.receiver_node_id_sorted[design_trace]):
        raise RefractionHalfInterceptTimeError(
            'design_matrix receiver rows do not match input_model.receiver_node_id_sorted'
        )
    if not np.allclose(
        design_distance,
        inputs.distance_m_sorted[design_trace],
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RefractionHalfInterceptTimeError(
            'design_matrix distances do not match input_model.distance_m_sorted'
        )
    if not np.allclose(
        observed_design,
        inputs.pick_time_s_sorted[design_trace],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RefractionHalfInterceptTimeError(
            'design_matrix observed picks do not match input_model.pick_time_s_sorted'
        )

    solver_trace = _coerce_1d_integer(
        _required(solver_result, 'row_trace_index_sorted'),
        name='solver_result.row_trace_index_sorted',
        expected_shape=expected_shape,
    )
    solver_source = _coerce_1d_integer(
        _required(solver_result, 'row_source_node_id'),
        name='solver_result.row_source_node_id',
        expected_shape=expected_shape,
    )
    solver_receiver = _coerce_1d_integer(
        _required(solver_result, 'row_receiver_node_id'),
        name='solver_result.row_receiver_node_id',
        expected_shape=expected_shape,
    )
    solver_distance = _coerce_1d_float(
        _required(solver_result, 'row_distance_m'),
        name='solver_result.row_distance_m',
        expected_shape=expected_shape,
    )
    solver_observed = _coerce_1d_float(
        _required(solver_result, 'observed_pick_time_s'),
        name='solver_result.observed_pick_time_s',
        expected_shape=expected_shape,
    )
    modeled = _coerce_1d_float(
        _required(solver_result, 'modeled_pick_time_s'),
        name='solver_result.modeled_pick_time_s',
        expected_shape=expected_shape,
    )
    residual = _coerce_1d_float(
        _required(solver_result, 'residual_time_s'),
        name='solver_result.residual_time_s',
        expected_shape=expected_shape,
    )
    used = _coerce_1d_bool(
        _required(solver_result, 'used_row_mask'),
        name='solver_result.used_row_mask',
        expected_shape=expected_shape,
    )
    rejected = _coerce_1d_bool(
        _required(solver_result, 'rejected_by_robust_mask'),
        name='solver_result.rejected_by_robust_mask',
        expected_shape=expected_shape,
    )

    if not np.array_equal(solver_trace, design_trace):
        raise RefractionHalfInterceptTimeError('solver row trace indices mismatch')
    if not np.array_equal(solver_source, design_source):
        raise RefractionHalfInterceptTimeError('solver source node rows mismatch')
    if not np.array_equal(solver_receiver, design_receiver):
        raise RefractionHalfInterceptTimeError('solver receiver node rows mismatch')
    if not np.allclose(solver_distance, design_distance, rtol=0.0, atol=1.0e-9):
        raise RefractionHalfInterceptTimeError('solver row distances mismatch')
    if not np.allclose(solver_observed, observed_design, rtol=0.0, atol=1.0e-12):
        raise RefractionHalfInterceptTimeError('solver observed pick rows mismatch')
    if np.any(used & rejected):
        raise RefractionHalfInterceptTimeError(
            'used_row_mask and rejected_by_robust_mask overlap'
        )

    return _ValidatedRows(
        row_trace_index_sorted=solver_trace,
        row_source_node_id=solver_source,
        row_receiver_node_id=solver_receiver,
        row_distance_m=solver_distance,
        observed_pick_time_s=solver_observed,
        modeled_pick_time_s=modeled,
        residual_time_s=residual,
        used_row_mask=used,
        rejected_by_robust_mask=rejected,
    )


def _validate_node_references(
    *,
    inputs: _ValidatedInputs,
    rows: _ValidatedRows,
    design_matrix: RefractionStaticDesignMatrix,
    solver_result: RefractionStaticSolverResult,
    node_pos: dict[int, int],
) -> None:
    design_active = _coerce_1d_integer(
        _required(design_matrix, 'active_node_id'),
        name='design_matrix.active_node_id',
    )
    for node in design_active.tolist():
        if int(node) not in node_pos:
            raise RefractionHalfInterceptTimeError(
                f'unknown active node ID in design matrix: {int(node)}'
            )
    solver_active = _coerce_1d_integer(
        _required(solver_result, 'active_node_id'),
        name='solver_result.active_node_id',
    )
    for node in solver_active.tolist():
        if int(node) not in node_pos:
            raise RefractionHalfInterceptTimeError(
                f'unknown active node ID in solver result: {int(node)}'
            )

    _validate_trace_node_ids(
        values=inputs.source_node_id_sorted,
        mask=inputs.valid_observation_mask_sorted,
        node_pos=node_pos,
        name='source_node_id_sorted',
    )
    _validate_trace_node_ids(
        values=inputs.receiver_node_id_sorted,
        mask=inputs.valid_observation_mask_sorted,
        node_pos=node_pos,
        name='receiver_node_id_sorted',
    )
    for node in np.concatenate((rows.row_source_node_id, rows.row_receiver_node_id)):
        if int(node) not in node_pos:
            raise RefractionHalfInterceptTimeError(
                f'unknown node ID in design rows: {int(node)}'
            )


def _build_node_solution_arrays(
    *,
    inputs: _ValidatedInputs,
    design_matrix: RefractionStaticDesignMatrix,
    solver_result: RefractionStaticSolverResult,
    rows: _ValidatedRows,
    node_pos: dict[int, int],
    min_picks_per_node: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = int(inputs.node_id.shape[0])
    node_half = np.full(n_nodes, np.nan, dtype=np.float64)
    status = np.full(n_nodes, 'inactive', dtype=_STATUS_DTYPE)

    design_active = _coerce_1d_integer(
        _required(design_matrix, 'active_node_id'),
        name='design_matrix.active_node_id',
    )
    active_set = {int(node) for node in design_active.tolist()}
    for node in active_set:
        status[node_pos[node]] = 'missing_solution'

    solver_active = _coerce_1d_integer(
        _required(solver_result, 'active_node_id'),
        name='solver_result.active_node_id',
    )
    active_half = _coerce_1d_float(
        _required(solver_result, 'active_node_half_intercept_time_s'),
        name='solver_result.active_node_half_intercept_time_s',
        expected_shape=(int(solver_active.shape[0]),),
    )
    if np.any(~np.isfinite(active_half)):
        raise RefractionHalfInterceptTimeError(
            'active_node_half_intercept_time_s must be finite'
        )
    if np.any(active_half < 0.0):
        raise RefractionHalfInterceptTimeError(
            'active_node_half_intercept_time_s must be non-negative'
        )

    lower = _coerce_1d_float(
        _required(solver_result, 'lower_bounds'),
        name='solver_result.lower_bounds',
    )
    upper = _coerce_1d_float(
        _required(solver_result, 'upper_bounds'),
        name='solver_result.upper_bounds',
    )
    if lower.shape[0] < solver_active.shape[0] or upper.shape[0] < solver_active.shape[0]:
        raise RefractionHalfInterceptTimeError(
            'solver_result bounds are shorter than active_node_id'
        )

    aggregation = _aggregate_node_rows(inputs.node_id, rows=rows, node_pos=node_pos)
    for active_index, node in enumerate(solver_active.tolist()):
        node_int = int(node)
        idx = node_pos[node_int]
        value = float(active_half[active_index])
        node_half[idx] = value
        if aggregation.used_pick_count[idx] < min_picks_per_node:
            status[idx] = 'low_fold'
        elif value <= float(lower[active_index]) + _BOUND_TOL:
            status[idx] = 'clipped_lower'
        elif value >= float(upper[active_index]) - _BOUND_TOL:
            status[idx] = 'clipped_upper'
        else:
            status[idx] = 'solved'

    return (
        np.ascontiguousarray(node_half, dtype=np.float64),
        np.ascontiguousarray(status, dtype=_STATUS_DTYPE),
    )


def _aggregate_node_rows(
    node_id: np.ndarray,
    *,
    rows: _ValidatedRows,
    node_pos: dict[int, int],
) -> _NodeAggregation:
    n_nodes = int(node_id.shape[0])
    pick_count = np.zeros(n_nodes, dtype=np.int64)
    used_count = np.zeros(n_nodes, dtype=np.int64)
    rejected_count = np.zeros(n_nodes, dtype=np.int64)
    residuals: list[list[float]] = [[] for _ in range(n_nodes)]

    for row_index in range(int(rows.row_trace_index_sorted.shape[0])):
        participants = {
            int(rows.row_source_node_id[row_index]),
            int(rows.row_receiver_node_id[row_index]),
        }
        for node in participants:
            idx = node_pos[node]
            pick_count[idx] += 1
            if rows.used_row_mask[row_index]:
                used_count[idx] += 1
                residuals[idx].append(float(rows.residual_time_s[row_index]))
            if rows.rejected_by_robust_mask[row_index]:
                rejected_count[idx] += 1

    mean = np.full(n_nodes, np.nan, dtype=np.float64)
    median = np.full(n_nodes, np.nan, dtype=np.float64)
    rms = np.full(n_nodes, np.nan, dtype=np.float64)
    mad = np.full(n_nodes, np.nan, dtype=np.float64)
    max_abs = np.full(n_nodes, np.nan, dtype=np.float64)
    for idx, values in enumerate(residuals):
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        center = float(np.median(arr))
        mean[idx] = float(np.mean(arr))
        median[idx] = center
        rms[idx] = float(np.sqrt(np.mean(arr * arr)))
        mad[idx] = float(1.4826 * np.median(np.abs(arr - center)))
        max_abs[idx] = float(np.max(np.abs(arr)))

    return _NodeAggregation(
        pick_count=np.ascontiguousarray(pick_count, dtype=np.int64),
        used_pick_count=np.ascontiguousarray(used_count, dtype=np.int64),
        rejected_pick_count=np.ascontiguousarray(rejected_count, dtype=np.int64),
        residual_mean_s=np.ascontiguousarray(mean, dtype=np.float64),
        residual_median_s=np.ascontiguousarray(median, dtype=np.float64),
        residual_rms_s=np.ascontiguousarray(rms, dtype=np.float64),
        residual_mad_s=np.ascontiguousarray(mad, dtype=np.float64),
        residual_max_abs_s=np.ascontiguousarray(max_abs, dtype=np.float64),
    )


def _build_endpoint_result(
    *,
    side: Literal['source', 'receiver'],
    inputs: _ValidatedInputs,
    rows: _ValidatedRows,
    node_pos: dict[int, int],
    node_half_s: np.ndarray,
    node_status: np.ndarray,
) -> _EndpointResult:
    if side == 'source':
        key_sorted = inputs.source_endpoint_key_sorted
        endpoint_id_sorted = inputs.source_id_sorted
        node_id_sorted = inputs.source_node_id_sorted
        x_sorted = inputs.source_x_m_sorted
        y_sorted = inputs.source_y_m_sorted
        elevation_sorted = inputs.source_elevation_m_sorted
        row_key = inputs.source_endpoint_key_sorted[rows.row_trace_index_sorted]
    else:
        key_sorted = inputs.receiver_endpoint_key_sorted
        endpoint_id_sorted = inputs.receiver_id_sorted
        node_id_sorted = inputs.receiver_node_id_sorted
        x_sorted = inputs.receiver_x_m_sorted
        y_sorted = inputs.receiver_y_m_sorted
        elevation_sorted = inputs.receiver_elevation_m_sorted
        row_key = inputs.receiver_endpoint_key_sorted[rows.row_trace_index_sorted]

    known_node_mask = np.fromiter(
        (int(node) in node_pos for node in node_id_sorted.tolist()),
        dtype=bool,
        count=int(node_id_sorted.shape[0]),
    )
    candidate_positions = np.flatnonzero(known_node_mask)
    positions = candidate_positions[
        _first_occurrence_positions(key_sorted[candidate_positions])
    ]
    n_endpoints = int(positions.shape[0])
    endpoint_key = key_sorted[positions].astype(_ENDPOINT_KEY_DTYPE, copy=False)
    endpoint_id = endpoint_id_sorted[positions].astype(np.int64, copy=False)
    endpoint_node = node_id_sorted[positions].astype(np.int64, copy=False)
    x_m = x_sorted[positions].astype(np.float64, copy=False)
    y_m = y_sorted[positions].astype(np.float64, copy=False)
    elevation_m = elevation_sorted[positions].astype(np.float64, copy=False)

    half = np.full(n_endpoints, np.nan, dtype=np.float64)
    status = np.full(n_endpoints, 'inactive', dtype=_STATUS_DTYPE)
    for idx, node in enumerate(endpoint_node.tolist()):
        node_int = int(node)
        node_index = node_pos.get(node_int)
        if node_index is None:
            continue
        half[idx] = float(node_half_s[node_index])
        status[idx] = str(node_status[node_index])

    pick_count = np.zeros(n_endpoints, dtype=np.int64)
    residuals: list[list[float]] = [[] for _ in range(n_endpoints)]
    endpoint_pos = {str(key): idx for idx, key in enumerate(endpoint_key.tolist())}
    for row_index, _trace_index in enumerate(rows.row_trace_index_sorted.tolist()):
        key = str(row_key[row_index])
        endpoint_index = endpoint_pos.get(key)
        if endpoint_index is None:
            continue
        pick_count[endpoint_index] += 1
        if rows.used_row_mask[row_index]:
            residuals[endpoint_index].append(float(rows.residual_time_s[row_index]))

    residual_rms = np.full(n_endpoints, np.nan, dtype=np.float64)
    for idx, values in enumerate(residuals):
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        residual_rms[idx] = float(np.sqrt(np.mean(arr * arr)))

    return _EndpointResult(
        endpoint_key=np.ascontiguousarray(endpoint_key, dtype=_ENDPOINT_KEY_DTYPE),
        endpoint_id=np.ascontiguousarray(endpoint_id, dtype=np.int64),
        node_id=np.ascontiguousarray(endpoint_node, dtype=np.int64),
        x_m=np.ascontiguousarray(x_m, dtype=np.float64),
        y_m=np.ascontiguousarray(y_m, dtype=np.float64),
        elevation_m=np.ascontiguousarray(elevation_m, dtype=np.float64),
        half_intercept_time_s=np.ascontiguousarray(half, dtype=np.float64),
        solution_status=np.ascontiguousarray(status, dtype=_STATUS_DTYPE),
        pick_count=np.ascontiguousarray(pick_count, dtype=np.int64),
        residual_rms_s=np.ascontiguousarray(residual_rms, dtype=np.float64),
    )


def _build_trace_order_arrays(
    *,
    inputs: _ValidatedInputs,
    rows: _ValidatedRows,
    node_pos: dict[int, int],
    node_half_s: np.ndarray,
    bedrock_velocity_m_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_traces = inputs.n_traces
    source_half = np.full(n_traces, np.nan, dtype=np.float64)
    receiver_half = np.full(n_traces, np.nan, dtype=np.float64)
    intercept_sum = np.full(n_traces, np.nan, dtype=np.float64)
    moveout = np.full(n_traces, np.nan, dtype=np.float64)
    estimated = np.full(n_traces, np.nan, dtype=np.float64)
    residual = np.full(n_traces, np.nan, dtype=np.float64)
    used = np.zeros(n_traces, dtype=bool)

    valid_indices = np.flatnonzero(inputs.valid_observation_mask_sorted)
    for trace_index in valid_indices.tolist():
        source_node = int(inputs.source_node_id_sorted[trace_index])
        receiver_node = int(inputs.receiver_node_id_sorted[trace_index])
        source_idx = node_pos[source_node]
        receiver_idx = node_pos[receiver_node]
        source_value = float(node_half_s[source_idx])
        receiver_value = float(node_half_s[receiver_idx])
        if not (np.isfinite(source_value) and np.isfinite(receiver_value)):
            continue
        distance = float(inputs.distance_m_sorted[trace_index])
        pick_time = float(inputs.pick_time_s_sorted[trace_index])
        if not (np.isfinite(distance) and np.isfinite(pick_time)):
            continue
        source_half[trace_index] = source_value
        receiver_half[trace_index] = receiver_value
        intercept_sum[trace_index] = source_value + receiver_value
        moveout[trace_index] = distance / bedrock_velocity_m_s
        estimated[trace_index] = intercept_sum[trace_index] + moveout[trace_index]
        residual[trace_index] = pick_time - estimated[trace_index]

    for row_index, trace_index in enumerate(rows.row_trace_index_sorted.tolist()):
        used[trace_index] = bool(rows.used_row_mask[row_index])

    finite_required = inputs.valid_observation_mask_sorted & np.isfinite(source_half)
    finite_required &= np.isfinite(receiver_half)
    if np.any(~np.isfinite(estimated[finite_required])):
        raise RefractionHalfInterceptTimeError(
            'estimated_first_break_time_s_sorted must be finite for solved valid observations'
        )

    return (
        np.ascontiguousarray(source_half, dtype=np.float64),
        np.ascontiguousarray(receiver_half, dtype=np.float64),
        np.ascontiguousarray(intercept_sum, dtype=np.float64),
        np.ascontiguousarray(moveout, dtype=np.float64),
        np.ascontiguousarray(estimated, dtype=np.float64),
        np.ascontiguousarray(residual, dtype=np.float64),
        np.ascontiguousarray(used, dtype=bool),
    )


def _build_qc(
    *,
    mode: Literal['solve_global', 'fixed_global'],
    bedrock_slowness_s_per_m: float,
    bedrock_velocity_m_s: float,
    weathering_velocity_m_s: float,
    inputs: _ValidatedInputs,
    rows: _ValidatedRows,
    node_half_s: np.ndarray,
    node_status: np.ndarray,
    node_pick_count: np.ndarray,
    node_used_pick_count: np.ndarray,
    source_endpoint: _EndpointResult,
    receiver_endpoint: _EndpointResult,
    solver_result: RefractionStaticSolverResult,
    source_receiver_linkage_used: bool,
) -> dict[str, Any]:
    finite_half_ms = node_half_s[np.isfinite(node_half_s)] * 1000.0
    used_residual = rows.residual_time_s[rows.used_row_mask]
    residual_ms = used_residual * 1000.0
    qc: dict[str, Any] = {
        'method': 'gli_variable_thickness',
        'bedrock_velocity_mode': mode,
        'bedrock_velocity_m_s': float(bedrock_velocity_m_s),
        'bedrock_slowness_s_per_m': float(bedrock_slowness_s_per_m),
        'weathering_velocity_m_s': float(weathering_velocity_m_s),
        'n_traces': int(inputs.n_traces),
        'n_valid_observations': int(rows.row_trace_index_sorted.shape[0]),
        'n_used_observations': int(np.count_nonzero(rows.used_row_mask)),
        'n_rejected_by_robust': int(np.count_nonzero(rows.rejected_by_robust_mask)),
        'n_nodes': int(inputs.node_id.shape[0]),
        'n_active_nodes': int(np.count_nonzero(node_status != 'inactive')),
        'n_inactive_nodes': int(np.count_nonzero(node_status == 'inactive')),
        'n_source_endpoints': int(source_endpoint.endpoint_key.shape[0]),
        'n_receiver_endpoints': int(receiver_endpoint.endpoint_key.shape[0]),
        'half_intercept_time_min_ms': _json_stat(finite_half_ms, 'min'),
        'half_intercept_time_max_ms': _json_stat(finite_half_ms, 'max'),
        'half_intercept_time_median_ms': _json_stat(finite_half_ms, 'median'),
        'half_intercept_time_p95_ms': _json_stat(finite_half_ms, 'p95'),
        'node_pick_count_min': _json_stat(node_pick_count, 'min'),
        'node_pick_count_max': _json_stat(node_pick_count, 'max'),
        'node_pick_count_median': _json_stat(node_pick_count, 'median'),
        'node_used_pick_count_min': _json_stat(node_used_pick_count, 'min'),
        'node_used_pick_count_max': _json_stat(node_used_pick_count, 'max'),
        'node_used_pick_count_median': _json_stat(node_used_pick_count, 'median'),
        'low_fold_node_count': int(np.count_nonzero(node_status == 'low_fold')),
        'clipped_lower_node_count': int(np.count_nonzero(node_status == 'clipped_lower')),
        'clipped_upper_node_count': int(np.count_nonzero(node_status == 'clipped_upper')),
        'inactive_node_count': int(np.count_nonzero(node_status == 'inactive')),
        'invalid_solution_node_count': int(
            np.count_nonzero(node_status == 'invalid_solution')
        ),
        'missing_solution_node_count': int(
            np.count_nonzero(node_status == 'missing_solution')
        ),
        'residual_rms_ms': _residual_stat(residual_ms, 'rms'),
        'residual_mad_ms': _residual_stat(residual_ms, 'mad'),
        'residual_mean_ms': _residual_stat(residual_ms, 'mean'),
        'residual_median_ms': _residual_stat(residual_ms, 'median'),
        'residual_p95_abs_ms': _residual_stat(residual_ms, 'p95_abs'),
        'residual_max_abs_ms': _residual_stat(residual_ms, 'max_abs'),
        'robust_enabled': bool(getattr(solver_result, 'qc', {}).get('robust_enabled', False)),
        'robust_method': str(getattr(solver_result, 'qc', {}).get('robust_method', '')),
        'robust_iteration_count': int(
            getattr(solver_result, 'robust_iteration_count', 0)
        ),
        'source_receiver_linkage_used': bool(source_receiver_linkage_used),
    }
    _assert_json_safe(qc)
    return qc


def _first_occurrence_positions(values: np.ndarray) -> np.ndarray:
    seen: set[str] = set()
    positions: list[int] = []
    for index, raw in enumerate(values.tolist()):
        value = str(raw)
        if not value or value in seen:
            continue
        seen.add(value)
        positions.append(index)
    return np.ascontiguousarray(positions, dtype=np.int64)


def _validate_trace_node_ids(
    *,
    values: np.ndarray,
    mask: np.ndarray,
    node_pos: dict[int, int],
    name: str,
) -> None:
    for node in values[mask].tolist():
        if int(node) not in node_pos:
            raise RefractionHalfInterceptTimeError(f'unknown {name}: {int(node)}')


def _resolve_weathering_velocity(
    value: float | None,
    *,
    solver_result: RefractionStaticSolverResult,
) -> float:
    if value is not None:
        return _positive_finite(value, name='weathering_velocity_m_s')
    qc = getattr(solver_result, 'qc', {})
    if isinstance(qc, dict) and qc.get('weathering_velocity_m_s') is not None:
        return _positive_finite(qc['weathering_velocity_m_s'], name='weathering_velocity_m_s')
    raise RefractionHalfInterceptTimeError('weathering_velocity_m_s is required')


def _resolve_min_picks_per_node(
    value: int | None,
    *,
    solver_result: RefractionStaticSolverResult,
) -> int:
    if value is not None:
        return _positive_int(value, name='min_picks_per_node')
    qc = getattr(solver_result, 'qc', {})
    if isinstance(qc, dict) and qc.get('min_picks_per_node') is not None:
        return _positive_int(qc['min_picks_per_node'], name='min_picks_per_node')
    return 1


def _validate_velocity_mode(value: object) -> Literal['solve_global', 'fixed_global']:
    if value == 'solve_global':
        return 'solve_global'
    if value == 'fixed_global':
        return 'fixed_global'
    raise RefractionHalfInterceptTimeError(
        'bedrock_velocity_mode must be solve_global or fixed_global'
    )


def _required(owner: object, field: str) -> object:
    try:
        value = getattr(owner, field)
    except AttributeError as exc:
        raise RefractionHalfInterceptTimeError(f'{field} is required') from exc
    if value is None:
        raise RefractionHalfInterceptTimeError(f'{field} is required')
    return value


def _coerce_1d_float(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    allow_nan: bool = False,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionHalfInterceptTimeError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise RefractionHalfInterceptTimeError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if np.issubdtype(arr.dtype, np.bool_) or not _is_real_numeric_dtype(arr.dtype):
        raise RefractionHalfInterceptTimeError(f'{name} must have a real numeric dtype')
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if allow_nan:
        if np.any(np.isinf(out)):
            raise RefractionHalfInterceptTimeError(f'{name} must not contain infinity')
    elif np.any(~np.isfinite(out)):
        raise RefractionHalfInterceptTimeError(f'{name} must contain only finite values')
    return out


def _coerce_1d_integer(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionHalfInterceptTimeError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise RefractionHalfInterceptTimeError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if np.issubdtype(arr.dtype, np.bool_):
        raise RefractionHalfInterceptTimeError(f'{name} must contain integer values')
    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr, dtype=np.int64)
    if not _is_real_numeric_dtype(arr.dtype):
        raise RefractionHalfInterceptTimeError(f'{name} must contain integer values')
    arr_f64 = arr.astype(np.float64, copy=False)
    if np.any(~np.isfinite(arr_f64)):
        raise RefractionHalfInterceptTimeError(f'{name} must contain finite values')
    if not np.all(arr_f64 == np.rint(arr_f64)):
        raise RefractionHalfInterceptTimeError(f'{name} must contain integer values')
    return np.ascontiguousarray(arr_f64, dtype=np.int64)


def _coerce_1d_bool(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionHalfInterceptTimeError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise RefractionHalfInterceptTimeError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if arr.dtype != np.bool_:
        raise RefractionHalfInterceptTimeError(f'{name} must have bool dtype')
    return np.ascontiguousarray(arr, dtype=bool)


def _coerce_1d_string(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionHalfInterceptTimeError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise RefractionHalfInterceptTimeError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    return np.ascontiguousarray(arr.astype(_ENDPOINT_KEY_DTYPE, copy=False))


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise RefractionHalfInterceptTimeError(f'{name} must be finite and positive')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionHalfInterceptTimeError(
            f'{name} must be finite and positive'
        ) from exc
    if not np.isfinite(out):
        raise RefractionHalfInterceptTimeError(f'{name} must be finite')
    if out <= 0.0:
        raise RefractionHalfInterceptTimeError(f'{name} must be positive')
    return out


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise RefractionHalfInterceptTimeError(f'{name} must be an integer')
    out = int(value)
    if out <= 0:
        raise RefractionHalfInterceptTimeError(f'{name} must be greater than 0')
    return out


def _json_stat(values: np.ndarray, stat: str) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if stat == 'min':
        return float(np.min(arr))
    if stat == 'max':
        return float(np.max(arr))
    if stat == 'median':
        return float(np.median(arr))
    if stat == 'p95':
        return float(np.percentile(arr, 95.0))
    raise RefractionHalfInterceptTimeError(f'unsupported statistic: {stat}')


def _residual_stat(values_ms: np.ndarray, stat: str) -> float | None:
    arr = np.asarray(values_ms, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if stat == 'rms':
        return float(np.sqrt(np.mean(arr * arr)))
    if stat == 'mean':
        return float(np.mean(arr))
    if stat == 'median':
        return float(np.median(arr))
    if stat == 'mad':
        center = float(np.median(arr))
        return float(1.4826 * np.median(np.abs(arr - center)))
    if stat == 'p95_abs':
        return float(np.percentile(np.abs(arr), 95.0))
    if stat == 'max_abs':
        return float(np.max(np.abs(arr)))
    raise RefractionHalfInterceptTimeError(f'unsupported residual statistic: {stat}')


def _node_rows(result: RefractionHalfInterceptTimeResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(result.node_id.shape[0])):
        rows.append(
            {
                'node_id': int(result.node_id[index]),
                'node_kind': str(result.node_kind[index]),
                'x_m': _csv_float(result.node_x_m[index]),
                'y_m': _csv_float(result.node_y_m[index]),
                'elevation_m': _csv_float(result.node_elevation_m[index]),
                'half_intercept_time_ms': _csv_float(
                    result.node_half_intercept_time_ms[index]
                ),
                'solution_status': str(result.node_solution_status[index]),
                'pick_count': int(result.node_pick_count[index]),
                'used_pick_count': int(result.node_used_pick_count[index]),
                'rejected_pick_count': int(result.node_rejected_pick_count[index]),
                'residual_mean_ms': _csv_float(
                    result.node_residual_mean_s[index] * 1000.0
                ),
                'residual_median_ms': _csv_float(
                    result.node_residual_median_s[index] * 1000.0
                ),
                'residual_rms_ms': _csv_float(
                    result.node_residual_rms_s[index] * 1000.0
                ),
                'residual_mad_ms': _csv_float(
                    result.node_residual_mad_s[index] * 1000.0
                ),
                'residual_max_abs_ms': _csv_float(
                    result.node_residual_max_abs_s[index] * 1000.0
                ),
            }
        )
    return rows


def _source_rows(result: RefractionHalfInterceptTimeResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(result.source_endpoint_key.shape[0])):
        rows.append(
            {
                'source_endpoint_key': str(result.source_endpoint_key[index]),
                'source_id': int(result.source_id[index]),
                'source_node_id': int(result.source_node_id[index]),
                'source_x_m': _csv_float(result.source_x_m[index]),
                'source_y_m': _csv_float(result.source_y_m[index]),
                'source_elevation_m': _csv_float(result.source_elevation_m[index]),
                'half_intercept_time_ms': _csv_float(
                    result.source_half_intercept_time_s[index] * 1000.0
                ),
                'solution_status': str(result.source_solution_status[index]),
                'pick_count': int(result.source_pick_count[index]),
                'residual_rms_ms': _csv_float(
                    result.source_residual_rms_s[index] * 1000.0
                ),
            }
        )
    return rows


def _receiver_rows(result: RefractionHalfInterceptTimeResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(result.receiver_endpoint_key.shape[0])):
        rows.append(
            {
                'receiver_endpoint_key': str(result.receiver_endpoint_key[index]),
                'receiver_id': int(result.receiver_id[index]),
                'receiver_node_id': int(result.receiver_node_id[index]),
                'receiver_x_m': _csv_float(result.receiver_x_m[index]),
                'receiver_y_m': _csv_float(result.receiver_y_m[index]),
                'receiver_elevation_m': _csv_float(
                    result.receiver_elevation_m[index]
                ),
                'half_intercept_time_ms': _csv_float(
                    result.receiver_half_intercept_time_s[index] * 1000.0
                ),
                'solution_status': str(result.receiver_solution_status[index]),
                'pick_count': int(result.receiver_pick_count[index]),
                'residual_rms_ms': _csv_float(
                    result.receiver_residual_rms_s[index] * 1000.0
                ),
            }
        )
    return rows


def _trace_preview_rows(result: RefractionHalfInterceptTimeResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(result.sorted_trace_index.shape[0])):
        rows.append(
            {
                'sorted_trace_index': int(result.sorted_trace_index[index]),
                'valid_observation': bool(result.valid_observation_mask_sorted[index]),
                'used_observation': bool(result.used_observation_mask_sorted[index]),
                'source_node_id': int(result.source_node_id_sorted[index]),
                'receiver_node_id': int(result.receiver_node_id_sorted[index]),
                'source_half_intercept_time_ms': _csv_float(
                    result.source_half_intercept_time_s_sorted[index] * 1000.0
                ),
                'receiver_half_intercept_time_ms': _csv_float(
                    result.receiver_half_intercept_time_s_sorted[index] * 1000.0
                ),
                'intercept_time_sum_ms': _csv_float(
                    result.estimated_intercept_time_sum_s_sorted[index] * 1000.0
                ),
                'bedrock_moveout_time_ms': _csv_float(
                    result.estimated_bedrock_moveout_time_s_sorted[index] * 1000.0
                ),
                'observed_pick_time_ms': _csv_float(
                    np.nan
                    if not result.valid_observation_mask_sorted[index]
                    else (
                        result.estimated_first_break_time_s_sorted[index]
                        + result.first_break_residual_s_sorted[index]
                    )
                    * 1000.0
                ),
                'estimated_first_break_time_ms': _csv_float(
                    result.estimated_first_break_time_s_sorted[index] * 1000.0
                ),
                'first_break_residual_ms': _csv_float(
                    result.first_break_residual_s_sorted[index] * 1000.0
                ),
            }
        )
    return rows


def _csv_float(value: object) -> str | float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ''
    if not np.isfinite(out):
        return ''
    return out


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        tmp_path.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
            ),
            encoding='utf-8',
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
) -> None:
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _assert_json_safe(payload: dict[str, Any]) -> None:
    json.dumps(payload, allow_nan=False)


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype,
        np.complexfloating,
    )


__all__ = [
    'REFRACTION_HALF_INTERCEPT_NODES_CSV_NAME',
    'REFRACTION_HALF_INTERCEPT_QC_JSON_NAME',
    'REFRACTION_HALF_INTERCEPT_RECEIVERS_CSV_NAME',
    'REFRACTION_HALF_INTERCEPT_SOURCES_CSV_NAME',
    'REFRACTION_HALF_INTERCEPT_TRACE_PREVIEW_CSV_NAME',
    'RefractionHalfInterceptTimeError',
    'RefractionHalfInterceptTimeResult',
    'build_refraction_half_intercept_time_model',
    'build_refraction_half_intercept_time_model_from_bedrock_result',
    'estimate_refraction_half_intercept_times_from_first_breaks',
    'write_refraction_half_intercept_artifacts',
]
