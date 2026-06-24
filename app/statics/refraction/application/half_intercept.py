"""Source/receiver half-intercept time model for GLI refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from seis_statics.refraction.half_intercept import (
    RefractionHalfInterceptError as CoreRefractionHalfInterceptError,
)
from seis_statics.refraction.half_intercept import (
    RefractionHalfInterceptResult as CoreRefractionHalfInterceptResult,
)
from seis_statics.refraction.half_intercept import (
    build_refraction_half_intercept_result_from_bedrock_result as core_build_refraction_half_intercept_result_from_bedrock_result,
)
from seis_statics.refraction.half_intercept import (
    estimate_refraction_half_intercept_from_input_model as core_estimate_refraction_half_intercept_from_input_model,
)
from seis_statics.refraction.types import (
    RefractionStaticInputModel as CoreRefractionStaticInputModel,
)

from app.statics.refraction.contracts.apply import RefractionStaticApplyRequest
from app.services.common.artifact_io import write_csv_atomic, write_json_atomic
from app.statics.refraction.application.bedrock import (
    estimate_global_bedrock_slowness_from_input_model,
)
from app.statics.refraction.ports.runtime import RefractionRuntime
from app.statics.refraction.application.design_matrix import (
    write_refraction_design_matrix_diagnostics_artifacts,
)
from app.statics.refraction.application.core_options import (
    core_input_model_from_app,
    model_options_from_request,
    resolve_weathering_velocity_from_model_request as resolve_weathering_velocity_m_s,
    solver_options_from_request,
)
from app.statics.refraction.application.trace_order import (
    sorted_positions_for_original_trace_ids,
)
from app.statics.refraction.contracts.result_types import (
    RefractionBedrockSlownessResult,
    RefractionHalfInterceptTimeResult,
    RefractionStaticInputModel,
    ResolvedRefractionFirstLayer,
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

_STATUS_DTYPE = '<U16'

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
class _HalfInterceptCoreContext:
    app_input_model: RefractionStaticInputModel
    core_input_model: CoreRefractionStaticInputModel
    core_result: CoreRefractionHalfInterceptResult
    app_result: RefractionHalfInterceptTimeResult


def estimate_refraction_half_intercept_times_from_first_breaks(
    *,
    req: RefractionStaticApplyRequest,
    runtime: RefractionRuntime | None = None,
    state: object | None = None,
    job_dir: Path | None = None,
    input_model: RefractionStaticInputModel | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> RefractionHalfInterceptTimeResult:
    """Build inputs, solve the GLI system, and emit the half-intercept model."""
    return estimate_refraction_half_intercept_core_context_from_first_breaks(
        req=req,
        runtime=runtime,
        state=state,
        job_dir=job_dir,
        input_model=input_model,
        resolved_first_layer=resolved_first_layer,
    ).app_result


def estimate_refraction_half_intercept_core_context_from_first_breaks(
    *,
    req: RefractionStaticApplyRequest,
    runtime: RefractionRuntime | None = None,
    state: object | None = None,
    job_dir: Path | None = None,
    input_model: RefractionStaticInputModel | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> _HalfInterceptCoreContext:
    """Build inputs, solve externally, and retain the external half-intercept result."""
    if runtime is None and input_model is None:
        if state is None:
            raise TypeError('runtime is required')
        raise TypeError('runtime is required; AppState adaptation belongs in adapters')
    try:
        if input_model is None:
            from app.statics.refraction.application.input_model import (
                build_refraction_static_input_model,
            )

            input_model = build_refraction_static_input_model(
                req=req,
                runtime=runtime,
                job_dir=job_dir,
            )
        if req.model.bedrock_velocity_mode == 'solve_global':
            bedrock_result = estimate_global_bedrock_slowness_from_input_model(
                input_model=input_model,
                model=req.model,
                solver=req.solver,
                job_dir=job_dir,
                include_debug_objects=True,
                resolved_first_layer=resolved_first_layer,
            )
            return build_refraction_half_intercept_core_context_from_bedrock_result(
                bedrock_result=bedrock_result,
                job_dir=job_dir,
            )
        core_input_model = core_input_model_from_app(input_model)
        core_result = core_estimate_refraction_half_intercept_from_input_model(
            input_model=core_input_model,
            model=model_options_from_request(req.model),
            solver_options=solver_options_from_request(req.solver),
            resolved_first_layer=resolved_first_layer,
            include_diagnostics=job_dir is not None,
            include_debug_objects=True,
        )
        result = _app_half_intercept_result_from_core(
            input_model=input_model,
            weathering_velocity_m_s=resolve_weathering_velocity_m_s(
                model=req.model,
                resolved_first_layer=resolved_first_layer,
                name='model.weathering_velocity_m_s',
            ),
            core_result=core_result,
        )
        if job_dir is not None:
            if core_result.debug_design is not None:
                write_refraction_design_matrix_diagnostics_artifacts(
                    Path(job_dir),
                    core_result.debug_design,
            )
            write_refraction_half_intercept_artifacts(Path(job_dir), result)
        return _HalfInterceptCoreContext(
            app_input_model=input_model,
            core_input_model=core_input_model,
            core_result=core_result,
            app_result=result,
        )
    except RefractionHalfInterceptTimeError:
        raise
    except (CoreRefractionHalfInterceptError, ValueError) as exc:
        raise RefractionHalfInterceptTimeError(str(exc)) from exc


def build_refraction_half_intercept_time_model_from_bedrock_result(
    *,
    bedrock_result: RefractionBedrockSlownessResult,
    job_dir: Path | None = None,
) -> RefractionHalfInterceptTimeResult:
    """Build the half-intercept model from a P0-06 bedrock slowness result."""
    return build_refraction_half_intercept_core_context_from_bedrock_result(
        bedrock_result=bedrock_result,
        job_dir=job_dir,
    ).app_result


def build_refraction_half_intercept_core_context_from_bedrock_result(
    *,
    bedrock_result: RefractionBedrockSlownessResult,
    job_dir: Path | None = None,
) -> _HalfInterceptCoreContext:
    """Build app and external half-intercept results from a core-backed bedrock result."""
    if bedrock_result.input_model is None:
        raise RefractionHalfInterceptTimeError(
            'bedrock_result.input_model is required'
        )
    if bedrock_result.design_matrix is None:
        raise RefractionHalfInterceptTimeError(
            'bedrock_result.design_matrix is required'
        )
    if bedrock_result.core_result is None:
        raise RefractionHalfInterceptTimeError('bedrock_result.core_result is required')
    core_input_model = core_input_model_from_app(bedrock_result.input_model)
    core_result = core_build_refraction_half_intercept_result_from_bedrock_result(
        input_model=core_input_model,
        bedrock_result=bedrock_result.core_result,
        include_debug_objects=True,
    )
    result = _app_half_intercept_result_from_core(
        input_model=bedrock_result.input_model,
        weathering_velocity_m_s=bedrock_result.weathering_velocity_m_s,
        core_result=core_result,
    )
    if job_dir is not None:
        if core_result.debug_design is not None:
            write_refraction_design_matrix_diagnostics_artifacts(
                Path(job_dir),
                core_result.debug_design,
            )
        write_refraction_half_intercept_artifacts(Path(job_dir), result)
    return _HalfInterceptCoreContext(
        app_input_model=bedrock_result.input_model,
        core_input_model=core_input_model,
        core_result=core_result,
        app_result=result,
    )


def _app_half_intercept_result_from_core(
    *,
    input_model: RefractionStaticInputModel,
    weathering_velocity_m_s: float,
    core_result: Any,
) -> RefractionHalfInterceptTimeResult:
    design = core_result.debug_design
    solve_result = core_result.debug_solve_result
    if design is None or solve_result is None:
        raise RefractionHalfInterceptTimeError(
            'external half-intercept result is missing debug solve output'
        )

    row_trace = np.ascontiguousarray(design.row_trace_index_sorted, dtype=np.int64)
    row_sorted_position = sorted_positions_for_original_trace_ids(
        sorted_to_original=input_model.sorted_trace_index,
        original_trace_id=row_trace,
    )
    used_sorted = np.ascontiguousarray(
        core_result.used_observation_mask_sorted,
        dtype=bool,
    )
    rejected_sorted = np.ascontiguousarray(
        core_result.rejected_observation_mask_sorted,
        dtype=bool,
    )
    used_row = np.ascontiguousarray(used_sorted[row_sorted_position], dtype=bool)
    rejected_row = np.ascontiguousarray(rejected_sorted[row_sorted_position], dtype=bool)
    row_residual = np.ascontiguousarray(
        solve_result.row_residual_s,
        dtype=np.float64,
    )
    node_stats = _node_residual_stats_from_rows(
        node_id=np.asarray(input_model.endpoint_table.node_id, dtype=np.int64),
        row_source_node_id=np.asarray(design.row_source_node_id, dtype=np.int64),
        row_receiver_node_id=np.asarray(design.row_receiver_node_id, dtype=np.int64),
        row_residual_s=row_residual,
        used_row_mask=used_row,
    )
    source_geometry = _endpoint_geometry_from_input(
        input_model=input_model,
        side='source',
        endpoint_key=core_result.source_endpoint.endpoint_key,
    )
    receiver_geometry = _endpoint_geometry_from_input(
        input_model=input_model,
        side='receiver',
        endpoint_key=core_result.receiver_endpoint.endpoint_key,
    )
    source_residual_rms = _endpoint_residual_rms_from_rows(
        row_endpoint_key=np.asarray(
            input_model.source_endpoint_key_sorted,
            dtype=object,
        )[row_sorted_position],
        endpoint_key=core_result.source_endpoint.endpoint_key,
        row_residual_s=row_residual,
        used_row_mask=used_row,
    )
    receiver_residual_rms = _endpoint_residual_rms_from_rows(
        row_endpoint_key=np.asarray(
            input_model.receiver_endpoint_key_sorted,
            dtype=object,
        )[row_sorted_position],
        endpoint_key=core_result.receiver_endpoint.endpoint_key,
        row_residual_s=row_residual,
        used_row_mask=used_row,
    )
    intercept_sum = np.ascontiguousarray(
        core_result.trace_half_intercept_time_s_sorted,
        dtype=np.float64,
    )
    estimated_first_break = np.ascontiguousarray(
        core_result.modeled_pick_time_s_sorted,
        dtype=np.float64,
    )
    moveout = np.ascontiguousarray(estimated_first_break - intercept_sum)
    qc = _core_half_intercept_qc(
        input_model=input_model,
        core_result=core_result,
        weathering_velocity_m_s=weathering_velocity_m_s,
        row_trace=row_trace,
        used_row_mask=used_row,
        rejected_row_mask=rejected_row,
    )
    active_cell_id = _optional_int_array(getattr(design, 'active_cell_id', None))
    cell_slowness, cell_velocity, cell_status = _active_cell_arrays_from_core(
        core_result=core_result,
        active_cell_id=active_cell_id,
    )
    bedrock_slowness, bedrock_velocity = _summary_bedrock_values_from_core(
        core_result=core_result,
        cell_slowness=cell_slowness,
        cell_velocity=cell_velocity,
    )
    qc['bedrock_slowness_s_per_m'] = float(bedrock_slowness)
    qc['bedrock_velocity_m_s'] = float(bedrock_velocity)
    qc.update(
        _cell_velocity_summary_qc(
            cell_slowness=cell_slowness,
            cell_velocity=cell_velocity,
        )
    )
    _assert_json_safe(qc)
    return RefractionHalfInterceptTimeResult(
        bedrock_velocity_mode=core_result.bedrock_velocity_mode,
        bedrock_slowness_s_per_m=bedrock_slowness,
        bedrock_velocity_m_s=bedrock_velocity,
        weathering_velocity_m_s=float(weathering_velocity_m_s),
        node_id=np.ascontiguousarray(input_model.endpoint_table.node_id, dtype=np.int64),
        node_x_m=np.ascontiguousarray(input_model.node_x_m, dtype=np.float64),
        node_y_m=np.ascontiguousarray(input_model.node_y_m, dtype=np.float64),
        node_elevation_m=np.ascontiguousarray(
            input_model.node_elevation_m,
            dtype=np.float64,
        ),
        node_kind=np.ascontiguousarray(input_model.node_kind),
        node_half_intercept_time_s=np.ascontiguousarray(
            core_result.node_half_intercept_time_s,
            dtype=np.float64,
        ),
        node_half_intercept_time_ms=np.ascontiguousarray(
            np.asarray(core_result.node_half_intercept_time_s, dtype=np.float64)
            * 1000.0,
            dtype=np.float64,
        ),
        node_solution_status=np.ascontiguousarray(core_result.node_solution_status),
        node_pick_count=np.ascontiguousarray(core_result.node_pick_count, dtype=np.int64),
        node_used_pick_count=np.ascontiguousarray(
            core_result.node_used_observation_count,
            dtype=np.int64,
        ),
        node_rejected_pick_count=np.ascontiguousarray(
            core_result.node_rejected_observation_count,
            dtype=np.int64,
        ),
        node_residual_mean_s=node_stats['mean'],
        node_residual_median_s=node_stats['median'],
        node_residual_rms_s=node_stats['rms'],
        node_residual_mad_s=node_stats['mad'],
        node_residual_max_abs_s=node_stats['max_abs'],
        source_endpoint_key=np.ascontiguousarray(
            core_result.source_endpoint.endpoint_key,
            dtype=object,
        ),
        source_id=source_geometry['id'],
        source_node_id=np.ascontiguousarray(
            core_result.source_endpoint.node_id,
            dtype=np.int64,
        ),
        source_x_m=source_geometry['x'],
        source_y_m=source_geometry['y'],
        source_elevation_m=source_geometry['elevation'],
        source_half_intercept_time_s=np.ascontiguousarray(
            core_result.source_endpoint.half_intercept_time_s,
            dtype=np.float64,
        ),
        source_solution_status=np.ascontiguousarray(
            core_result.source_endpoint.solution_status,
        ),
        source_pick_count=np.ascontiguousarray(
            core_result.source_endpoint.pick_count,
            dtype=np.int64,
        ),
        source_residual_rms_s=source_residual_rms,
        receiver_endpoint_key=np.ascontiguousarray(
            core_result.receiver_endpoint.endpoint_key,
            dtype=object,
        ),
        receiver_id=receiver_geometry['id'],
        receiver_node_id=np.ascontiguousarray(
            core_result.receiver_endpoint.node_id,
            dtype=np.int64,
        ),
        receiver_x_m=receiver_geometry['x'],
        receiver_y_m=receiver_geometry['y'],
        receiver_elevation_m=receiver_geometry['elevation'],
        receiver_half_intercept_time_s=np.ascontiguousarray(
            core_result.receiver_endpoint.half_intercept_time_s,
            dtype=np.float64,
        ),
        receiver_solution_status=np.ascontiguousarray(
            core_result.receiver_endpoint.solution_status,
        ),
        receiver_pick_count=np.ascontiguousarray(
            core_result.receiver_endpoint.pick_count,
            dtype=np.int64,
        ),
        receiver_residual_rms_s=receiver_residual_rms,
        sorted_trace_index=np.ascontiguousarray(
            input_model.sorted_trace_index,
            dtype=np.int64,
        ),
        source_endpoint_key_sorted=np.ascontiguousarray(
            input_model.source_endpoint_key_sorted,
        ),
        receiver_endpoint_key_sorted=np.ascontiguousarray(
            input_model.receiver_endpoint_key_sorted,
        ),
        source_elevation_m_sorted=np.ascontiguousarray(
            input_model.source_elevation_m_sorted,
            dtype=np.float64,
        ),
        receiver_elevation_m_sorted=np.ascontiguousarray(
            input_model.receiver_elevation_m_sorted,
            dtype=np.float64,
        ),
        source_node_id_sorted=np.ascontiguousarray(
            input_model.source_node_id_sorted,
            dtype=np.int64,
        ),
        receiver_node_id_sorted=np.ascontiguousarray(
            input_model.receiver_node_id_sorted,
            dtype=np.int64,
        ),
        source_half_intercept_time_s_sorted=np.ascontiguousarray(
            core_result.source_half_intercept_time_s_sorted,
            dtype=np.float64,
        ),
        receiver_half_intercept_time_s_sorted=np.ascontiguousarray(
            core_result.receiver_half_intercept_time_s_sorted,
            dtype=np.float64,
        ),
        estimated_intercept_time_sum_s_sorted=intercept_sum,
        estimated_bedrock_moveout_time_s_sorted=moveout,
        estimated_first_break_time_s_sorted=estimated_first_break,
        first_break_residual_s_sorted=np.ascontiguousarray(
            core_result.residual_s_sorted,
            dtype=np.float64,
        ),
        valid_observation_mask_sorted=np.ascontiguousarray(
            input_model.valid_observation_mask_sorted,
            dtype=bool,
        ),
        used_observation_mask_sorted=used_sorted,
        row_trace_index_sorted=row_trace,
        row_source_node_id=np.ascontiguousarray(
            design.row_source_node_id,
            dtype=np.int64,
        ),
        row_receiver_node_id=np.ascontiguousarray(
            design.row_receiver_node_id,
            dtype=np.int64,
        ),
        row_distance_m=np.ascontiguousarray(design.row_distance_m, dtype=np.float64),
        observed_pick_time_s=np.ascontiguousarray(
            design.observed_pick_time_s,
            dtype=np.float64,
        ),
        modeled_pick_time_s=np.ascontiguousarray(
            solve_result.row_modeled_pick_time_s,
            dtype=np.float64,
        ),
        residual_time_s=row_residual,
        used_row_mask=used_row,
        rejected_by_robust_mask=rejected_row,
        qc=qc,
        active_cell_id=active_cell_id,
        inactive_cell_id=_optional_int_array(getattr(design, 'inactive_cell_id', None)),
        cell_bedrock_slowness_s_per_m=cell_slowness,
        cell_bedrock_velocity_m_s=cell_velocity,
        cell_velocity_status=cell_status,
        row_midpoint_cell_id=_optional_int_array(
            getattr(design, 'row_midpoint_cell_id', None),
        ),
        row_midpoint_bedrock_velocity_m_s=_optional_float_array(
            getattr(solve_result, 'row_midpoint_bedrock_velocity_m_s', None),
        ),
    )


def _optional_int_array(values: object) -> np.ndarray | None:
    if values is None:
        return None
    return np.ascontiguousarray(values, dtype=np.int64)


def _optional_float_array(values: object) -> np.ndarray | None:
    if values is None:
        return None
    return np.ascontiguousarray(values, dtype=np.float64)


def _active_cell_arrays_from_core(
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
            np.empty(0, dtype=_STATUS_DTYPE),
        )
    cell_id = np.asarray(core_result.cell_id, dtype=np.int64)
    slowness = np.asarray(
        core_result.cell_bedrock_slowness_s_per_m,
        dtype=np.float64,
    )
    velocity = np.asarray(core_result.cell_bedrock_velocity_m_s, dtype=np.float64)
    status = np.asarray(core_result.cell_velocity_status, dtype=_STATUS_DTYPE)
    if (
        cell_id.ndim != 1
        or slowness.shape != cell_id.shape
        or velocity.shape != cell_id.shape
        or status.shape != cell_id.shape
        or np.unique(cell_id).shape[0] != cell_id.shape[0]
    ):
        raise RefractionHalfInterceptTimeError(
            'solve_cell result active cell arrays length mismatch'
        )
    cell_pos_by_id = {int(cell): idx for idx, cell in enumerate(cell_id.tolist())}
    try:
        active_pos = np.asarray(
            [cell_pos_by_id[int(cell)] for cell in active_cell_id],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise RefractionHalfInterceptTimeError(
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
        np.ascontiguousarray(status[active_pos], dtype=_STATUS_DTYPE),
    )


def _summary_bedrock_values_from_core(
    *,
    core_result: Any,
    cell_slowness: np.ndarray | None,
    cell_velocity: np.ndarray | None,
) -> tuple[float, float]:
    if str(core_result.bedrock_velocity_mode) != 'solve_cell':
        return (
            float(core_result.bedrock_slowness_s_per_m),
            float(core_result.bedrock_velocity_m_s),
        )
    if cell_slowness is None or cell_slowness.size == 0:
        return float('nan'), float('nan')
    finite = np.asarray(cell_slowness, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float('nan'), float('nan')
    slowness = float(np.median(finite))
    return slowness, float(1.0 / slowness)


def _cell_velocity_summary_qc(
    *,
    cell_slowness: np.ndarray | None,
    cell_velocity: np.ndarray | None,
) -> dict[str, float]:
    if cell_slowness is None or cell_velocity is None:
        return {}
    velocity = np.asarray(cell_velocity, dtype=np.float64)
    slowness = np.asarray(cell_slowness, dtype=np.float64)
    velocity = velocity[np.isfinite(velocity)]
    slowness = slowness[np.isfinite(slowness)]
    if velocity.size == 0 or slowness.size == 0:
        return {}
    return {
        'cell_bedrock_velocity_min_m_s': float(np.min(velocity)),
        'cell_bedrock_velocity_median_m_s': float(np.median(velocity)),
        'cell_bedrock_velocity_max_m_s': float(np.max(velocity)),
        'cell_bedrock_slowness_min_s_per_m': float(np.min(slowness)),
        'cell_bedrock_slowness_median_s_per_m': float(np.median(slowness)),
        'cell_bedrock_slowness_max_s_per_m': float(np.max(slowness)),
    }


def _node_residual_stats_from_rows(
    *,
    node_id: np.ndarray,
    row_source_node_id: np.ndarray,
    row_receiver_node_id: np.ndarray,
    row_residual_s: np.ndarray,
    used_row_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    node_pos = {int(node): index for index, node in enumerate(node_id.tolist())}
    residuals: list[list[float]] = [[] for _ in range(int(node_id.shape[0]))]
    for row_index, used in enumerate(used_row_mask.tolist()):
        if not used:
            continue
        residual = float(row_residual_s[row_index])
        for node in {
            int(row_source_node_id[row_index]),
            int(row_receiver_node_id[row_index]),
        }:
            index = node_pos.get(node)
            if index is not None:
                residuals[index].append(residual)
    return _residual_summary_arrays(residuals)


def _endpoint_residual_rms_from_rows(
    *,
    row_endpoint_key: np.ndarray,
    endpoint_key: np.ndarray,
    row_residual_s: np.ndarray,
    used_row_mask: np.ndarray,
) -> np.ndarray:
    endpoint_pos = {str(key): index for index, key in enumerate(endpoint_key.tolist())}
    residuals: list[list[float]] = [[] for _ in range(int(endpoint_key.shape[0]))]
    for row_index, used in enumerate(used_row_mask.tolist()):
        if not used:
            continue
        index = endpoint_pos.get(str(row_endpoint_key[row_index]))
        if index is not None:
            residuals[index].append(float(row_residual_s[row_index]))
    return _residual_summary_arrays(residuals)['rms']


def _residual_summary_arrays(residuals: list[list[float]]) -> dict[str, np.ndarray]:
    n_items = len(residuals)
    mean = np.full(n_items, np.nan, dtype=np.float64)
    median = np.full(n_items, np.nan, dtype=np.float64)
    rms = np.full(n_items, np.nan, dtype=np.float64)
    mad = np.full(n_items, np.nan, dtype=np.float64)
    max_abs = np.full(n_items, np.nan, dtype=np.float64)
    for index, values in enumerate(residuals):
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        center = float(np.median(arr))
        mean[index] = float(np.mean(arr))
        median[index] = center
        rms[index] = float(np.sqrt(np.mean(arr * arr)))
        mad[index] = float(1.4826 * np.median(np.abs(arr - center)))
        max_abs[index] = float(np.max(np.abs(arr)))
    return {
        'mean': np.ascontiguousarray(mean, dtype=np.float64),
        'median': np.ascontiguousarray(median, dtype=np.float64),
        'rms': np.ascontiguousarray(rms, dtype=np.float64),
        'mad': np.ascontiguousarray(mad, dtype=np.float64),
        'max_abs': np.ascontiguousarray(max_abs, dtype=np.float64),
    }


def _endpoint_geometry_from_input(
    *,
    input_model: RefractionStaticInputModel,
    side: Literal['source', 'receiver'],
    endpoint_key: np.ndarray,
) -> dict[str, np.ndarray]:
    if side == 'source':
        key_sorted = np.asarray(input_model.source_endpoint_key_sorted, dtype=object)
        id_sorted = np.asarray(input_model.source_id_sorted, dtype=np.int64)
        x_sorted = np.asarray(input_model.source_x_m_sorted, dtype=np.float64)
        y_sorted = np.asarray(input_model.source_y_m_sorted, dtype=np.float64)
        elevation_sorted = np.asarray(
            input_model.source_elevation_m_sorted,
            dtype=np.float64,
        )
    else:
        key_sorted = np.asarray(input_model.receiver_endpoint_key_sorted, dtype=object)
        id_sorted = np.asarray(input_model.receiver_id_sorted, dtype=np.int64)
        x_sorted = np.asarray(input_model.receiver_x_m_sorted, dtype=np.float64)
        y_sorted = np.asarray(input_model.receiver_y_m_sorted, dtype=np.float64)
        elevation_sorted = np.asarray(
            input_model.receiver_elevation_m_sorted,
            dtype=np.float64,
        )
    first_index: dict[str, int] = {}
    for index, raw_key in enumerate(key_sorted.tolist()):
        first_index.setdefault(str(raw_key), index)
    n_endpoints = int(endpoint_key.shape[0])
    endpoint_id = np.full(n_endpoints, -1, dtype=np.int64)
    x_m = np.full(n_endpoints, np.nan, dtype=np.float64)
    y_m = np.full(n_endpoints, np.nan, dtype=np.float64)
    elevation_m = np.full(n_endpoints, np.nan, dtype=np.float64)
    for index, raw_key in enumerate(endpoint_key.tolist()):
        source_index = first_index.get(str(raw_key))
        if source_index is None:
            continue
        endpoint_id[index] = int(id_sorted[source_index])
        x_m[index] = float(x_sorted[source_index])
        y_m[index] = float(y_sorted[source_index])
        elevation_m[index] = float(elevation_sorted[source_index])
    return {
        'id': np.ascontiguousarray(endpoint_id, dtype=np.int64),
        'x': np.ascontiguousarray(x_m, dtype=np.float64),
        'y': np.ascontiguousarray(y_m, dtype=np.float64),
        'elevation': np.ascontiguousarray(elevation_m, dtype=np.float64),
    }


def _core_half_intercept_qc(
    *,
    input_model: RefractionStaticInputModel,
    core_result: Any,
    weathering_velocity_m_s: float,
    row_trace: np.ndarray,
    used_row_mask: np.ndarray,
    rejected_row_mask: np.ndarray,
) -> dict[str, Any]:
    qc = _json_safe_copy(core_result.qc)
    half_ms = np.asarray(core_result.node_half_intercept_time_s, dtype=np.float64)
    half_ms = half_ms[np.isfinite(half_ms)] * 1000.0
    residual_ms = np.asarray(core_result.residual_s_sorted, dtype=np.float64)[
        np.asarray(core_result.used_observation_mask_sorted, dtype=bool)
    ]
    residual_ms = residual_ms * 1000.0
    qc.update(
        {
            'method': 'gli_variable_thickness',
            'bedrock_velocity_mode': str(core_result.bedrock_velocity_mode),
            'bedrock_velocity_m_s': float(core_result.bedrock_velocity_m_s),
            'bedrock_slowness_s_per_m': float(core_result.bedrock_slowness_s_per_m),
            'weathering_velocity_m_s': float(weathering_velocity_m_s),
            'n_traces': int(input_model.n_traces),
            'n_valid_observations': int(row_trace.shape[0]),
            'n_used_observations': int(np.count_nonzero(used_row_mask)),
            'n_rejected_by_robust': int(np.count_nonzero(rejected_row_mask)),
            'n_nodes': int(np.asarray(input_model.endpoint_table.node_id).shape[0]),
            'n_active_nodes': int(
                np.count_nonzero(
                    np.asarray(core_result.node_solution_status).astype(str)
                    != 'inactive'
                )
            ),
            'n_inactive_nodes': int(
                np.count_nonzero(
                    np.asarray(core_result.node_solution_status).astype(str)
                    == 'inactive'
                )
            ),
            'n_source_endpoints': int(core_result.source_endpoint.endpoint_key.shape[0]),
            'n_receiver_endpoints': int(
                core_result.receiver_endpoint.endpoint_key.shape[0]
            ),
            'half_intercept_time_min_ms': _json_stat(half_ms, 'min'),
            'half_intercept_time_max_ms': _json_stat(half_ms, 'max'),
            'half_intercept_time_median_ms': _json_stat(half_ms, 'median'),
            'half_intercept_time_p95_ms': _json_stat(half_ms, 'p95'),
            'residual_rms_ms': _residual_stat(residual_ms, 'rms'),
            'residual_mad_ms': _residual_stat(residual_ms, 'mad'),
            'residual_mean_ms': _residual_stat(residual_ms, 'mean'),
            'residual_median_ms': _residual_stat(residual_ms, 'median'),
            'residual_p95_abs_ms': _residual_stat(residual_ms, 'p95_abs'),
            'residual_max_abs_ms': _residual_stat(residual_ms, 'max_abs'),
            'robust_enabled': bool(core_result.robust_enabled),
            'robust_method': str(
                getattr(getattr(core_result, 'debug_solve_result', None), 'qc', {}).get(
                    'robust_method',
                    '',
                )
            ),
            'robust_iteration_count': int(
                len(getattr(core_result, 'robust_iteration_summaries', ()))
            ),
            'source_receiver_linkage_used': bool(
                getattr(input_model, 'qc', {}).get('linkage_used', False)
            ),
        }
    )
    layer_qc = getattr(input_model, 'qc', {}).get('layers')
    if isinstance(layer_qc, dict):
        qc['layers'] = layer_qc
    debug_design = getattr(core_result, 'debug_design', None)
    design_qc = getattr(debug_design, 'qc', {}) if debug_design is not None else {}
    if isinstance(design_qc, dict):
        for key in (
            'min_observations_per_cell',
            'n_low_fold_cells',
            'n_observations_rejected_by_low_fold_cell',
            'low_fold_cell_rejection_reason',
            'low_fold_cell_id',
            'cell_observation_count',
        ):
            if key in design_qc:
                qc[key] = _json_safe_copy(design_qc[key])
    return qc


def _json_safe_copy(value: object) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_copy(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe_copy(value.tolist())
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


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
    write_json_atomic(
        qc_path,
        result.qc,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
    )
    write_csv_atomic(
        node_path,
        columns=_NODE_COLUMNS,
        rows=_node_rows(result),
        lineterminator='\r\n',
    )
    write_csv_atomic(
        source_path,
        columns=_SOURCE_COLUMNS,
        rows=_source_rows(result),
        lineterminator='\r\n',
    )
    write_csv_atomic(
        receiver_path,
        columns=_RECEIVER_COLUMNS,
        rows=_receiver_rows(result),
        lineterminator='\r\n',
    )
    write_csv_atomic(
        trace_path,
        columns=_TRACE_PREVIEW_COLUMNS,
        rows=_trace_preview_rows(result),
        lineterminator='\r\n',
    )
    return {
        'qc_json': qc_path,
        'nodes_csv': node_path,
        'sources_csv': source_path,
        'receivers_csv': receiver_path,
        'trace_preview_csv': trace_path,
    }


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


def _assert_json_safe(payload: dict[str, Any]) -> None:
    json.dumps(payload, allow_nan=False)


__all__ = [
    'REFRACTION_HALF_INTERCEPT_NODES_CSV_NAME',
    'REFRACTION_HALF_INTERCEPT_QC_JSON_NAME',
    'REFRACTION_HALF_INTERCEPT_RECEIVERS_CSV_NAME',
    'REFRACTION_HALF_INTERCEPT_SOURCES_CSV_NAME',
    'REFRACTION_HALF_INTERCEPT_TRACE_PREVIEW_CSV_NAME',
    'RefractionHalfInterceptTimeError',
    'RefractionHalfInterceptTimeResult',
    'build_refraction_half_intercept_core_context_from_bedrock_result',
    'build_refraction_half_intercept_time_model_from_bedrock_result',
    'estimate_refraction_half_intercept_core_context_from_first_breaks',
    'estimate_refraction_half_intercept_times_from_first_breaks',
    'write_refraction_half_intercept_artifacts',
]
