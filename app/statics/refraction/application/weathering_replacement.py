"""Weathering-replacement static shifts for GLI refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app.statics.refraction.contracts.options import RefractionStaticApplyOptions
from app.statics.refraction.contracts.apply import RefractionStaticApplyRequest
from app.services.common.array_validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_integer_int64 as _coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_1d_string_array as _coerce_1d_string_array,
    coerce_finite_float as _coerce_finite_float,
)
from app.services.common.artifact_io import write_csv_atomic, write_json_atomic
from seis_statics.refraction.first_layer import (
    validate_resolved_first_layer_velocity_match,
)
from seis_statics.refraction.weathering import (
    RefractionWeatheringEndpointComponents as CoreRefractionWeatheringEndpointComponents,
)
from seis_statics.refraction.weathering import (
    RefractionWeatheringModel as CoreRefractionWeatheringModel,
)
from app.statics.refraction.contracts.result_types import (
    RefractionStaticInputModel,
    RefractionWeatheringReplacementStaticsResult,
    RefractionWeatheringThicknessResult,
    ResolvedRefractionFirstLayer,
)
from app.statics.refraction.application.weathering import (
    estimate_weathering_thickness_core_context_from_first_breaks,
)
from app.statics.refraction.ports.runtime import RefractionRuntime
from seis_statics.refraction.weathering_replacement import (
    RefractionWeatheringReplacementError as CoreRefractionWeatheringReplacementError,
)
from seis_statics.refraction.weathering_replacement import (
    RefractionWeatheringReplacementResult as CoreRefractionWeatheringReplacementResult,
)
from seis_statics.refraction.weathering_replacement import (
    build_refraction_weathering_replacement_statics as core_build_refraction_weathering_replacement_statics,
)
from seis_statics.refraction.weathering_replacement import (
    compute_weathering_replacement_shift_s as core_compute_weathering_replacement_shift_s,
)

REFRACTION_WEATHERING_REPLACEMENT_QC_JSON_NAME = (
    'refraction_weathering_replacement_qc.json'
)
REFRACTION_WEATHERING_REPLACEMENT_NODES_CSV_NAME = (
    'refraction_weathering_replacement_nodes.csv'
)
REFRACTION_WEATHERING_REPLACEMENT_SOURCES_CSV_NAME = (
    'refraction_weathering_replacement_sources.csv'
)
REFRACTION_WEATHERING_REPLACEMENT_RECEIVERS_CSV_NAME = (
    'refraction_weathering_replacement_receivers.csv'
)
REFRACTION_WEATHERING_REPLACEMENT_TRACE_PREVIEW_CSV_NAME = (
    'refraction_weathering_replacement_trace_preview.csv'
)

_STATUS_DTYPE = '<U32'
_ENDPOINT_KEY_DTYPE = object
_SLOWNESS_RTOL = 1.0e-8

_NODE_COLUMNS = (
    'node_id',
    'node_kind',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'weathering_thickness_m',
    'refractor_elevation_m',
    'weathering_status',
    'weathering_replacement_shift_ms',
    'static_status',
    'pick_count',
    'used_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
)
_SOURCE_COLUMNS = (
    'source_endpoint_key',
    'source_id',
    'source_node_id',
    'source_x_m',
    'source_y_m',
    'source_surface_elevation_m',
    'source_weathering_thickness_m',
    'source_weathering_replacement_shift_ms',
    'source_static_status',
)
_RECEIVER_COLUMNS = (
    'receiver_endpoint_key',
    'receiver_id',
    'receiver_node_id',
    'receiver_x_m',
    'receiver_y_m',
    'receiver_surface_elevation_m',
    'receiver_weathering_thickness_m',
    'receiver_weathering_replacement_shift_ms',
    'receiver_static_status',
)
_TRACE_PREVIEW_COLUMNS = (
    'sorted_trace_index',
    'valid_observation',
    'used_observation',
    'trace_static_valid',
    'source_node_id',
    'receiver_node_id',
    'source_weathering_thickness_m',
    'receiver_weathering_thickness_m',
    'source_weathering_replacement_shift_ms',
    'receiver_weathering_replacement_shift_ms',
    'weathering_replacement_trace_shift_ms',
    'source_static_status',
    'receiver_static_status',
    'trace_static_status',
    'estimated_first_break_time_ms',
    'first_break_residual_ms',
)


class RefractionWeatheringReplacementStaticsError(ValueError):
    """Raised when weathering-replacement static outputs cannot be built."""


@dataclass(frozen=True)
class _WeatheringReplacementCoreContext:
    core_replacement_result: CoreRefractionWeatheringReplacementResult
    app_replacement_result: RefractionWeatheringReplacementStaticsResult


@dataclass(frozen=True)
class _VelocityContext:
    mode: Literal['solve_global', 'fixed_global', 'solve_cell']
    bedrock_slowness_s_per_m: float
    bedrock_velocity_m_s: float
    weathering_velocity_m_s: float
    replacement_slowness_delta_s_per_m: float


@dataclass(frozen=True)
class _ValidatedWeathering:
    node_id: np.ndarray
    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_surface_elevation_m: np.ndarray
    node_kind: np.ndarray
    node_weathering_thickness_m: np.ndarray
    node_refractor_elevation_m: np.ndarray
    node_half_intercept_time_s: np.ndarray
    node_solution_status: np.ndarray
    node_weathering_status: np.ndarray
    node_pick_count: np.ndarray
    node_used_pick_count: np.ndarray
    node_rejected_pick_count: np.ndarray
    node_residual_rms_s: np.ndarray
    node_residual_mad_s: np.ndarray
    active_cell_id: np.ndarray | None
    inactive_cell_id: np.ndarray | None
    cell_bedrock_slowness_s_per_m: np.ndarray | None
    cell_bedrock_velocity_m_s: np.ndarray | None
    cell_velocity_status: np.ndarray | None
    row_midpoint_cell_id: np.ndarray | None
    node_v2_cell_id: np.ndarray | None
    node_v2_m_s: np.ndarray | None
    node_v2_status: np.ndarray | None
    source_endpoint_key: np.ndarray
    source_id: np.ndarray
    source_node_id: np.ndarray
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    source_surface_elevation_m: np.ndarray
    source_half_intercept_time_s: np.ndarray
    source_weathering_thickness_m: np.ndarray
    source_refractor_elevation_m: np.ndarray
    source_weathering_status: np.ndarray
    source_v2_cell_id: np.ndarray | None
    source_v2_m_s: np.ndarray | None
    source_v2_status: np.ndarray | None
    receiver_endpoint_key: np.ndarray
    receiver_id: np.ndarray
    receiver_node_id: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    receiver_surface_elevation_m: np.ndarray
    receiver_half_intercept_time_s: np.ndarray
    receiver_weathering_thickness_m: np.ndarray
    receiver_refractor_elevation_m: np.ndarray
    receiver_weathering_status: np.ndarray
    receiver_v2_cell_id: np.ndarray | None
    receiver_v2_m_s: np.ndarray | None
    receiver_v2_status: np.ndarray | None
    sorted_trace_index: np.ndarray
    valid_observation_mask_sorted: np.ndarray
    used_observation_mask_sorted: np.ndarray
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    source_half_intercept_time_s_sorted: np.ndarray
    receiver_half_intercept_time_s_sorted: np.ndarray
    source_weathering_thickness_m_sorted: np.ndarray
    receiver_weathering_thickness_m_sorted: np.ndarray
    source_refractor_elevation_m_sorted: np.ndarray
    receiver_refractor_elevation_m_sorted: np.ndarray
    source_weathering_status_sorted: np.ndarray
    receiver_weathering_status_sorted: np.ndarray
    source_v2_cell_id_sorted: np.ndarray | None
    source_v2_m_s_sorted: np.ndarray | None
    source_v2_status_sorted: np.ndarray | None
    receiver_v2_cell_id_sorted: np.ndarray | None
    receiver_v2_m_s_sorted: np.ndarray | None
    receiver_v2_status_sorted: np.ndarray | None
    estimated_first_break_time_s_sorted: np.ndarray
    first_break_residual_s_sorted: np.ndarray
    row_trace_index_sorted: np.ndarray
    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    row_distance_m: np.ndarray
    observed_pick_time_s: np.ndarray
    modeled_pick_time_s: np.ndarray
    residual_time_s: np.ndarray
    used_row_mask: np.ndarray
    rejected_by_robust_mask: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.node_id.shape[0])

    @property
    def n_traces(self) -> int:
        return int(self.sorted_trace_index.shape[0])


def compute_weathering_replacement_statics_from_first_breaks(
    *,
    req: RefractionStaticApplyRequest,
    runtime: RefractionRuntime | None = None,
    state: object | None = None,
    job_dir: Path | None = None,
    input_model: RefractionStaticInputModel | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> RefractionWeatheringReplacementStaticsResult:
    """Run the GLI weathering model, then compute replacement static shifts."""
    return compute_weathering_replacement_core_context_from_first_breaks(
        req=req,
        runtime=runtime,
        state=state,
        job_dir=job_dir,
        input_model=input_model,
        resolved_first_layer=resolved_first_layer,
    ).app_replacement_result


def compute_weathering_replacement_core_context_from_first_breaks(
    *,
    req: RefractionStaticApplyRequest,
    runtime: RefractionRuntime | None = None,
    state: object | None = None,
    job_dir: Path | None = None,
    input_model: RefractionStaticInputModel | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> _WeatheringReplacementCoreContext:
    """Run weathering replacement while retaining the external replacement result."""
    try:
        weathering_kwargs: dict[str, Any] = {
            'req': req,
            'job_dir': job_dir,
        }
        if runtime is not None:
            weathering_kwargs['runtime'] = runtime
        elif state is not None:
            weathering_kwargs['state'] = state
        if input_model is not None:
            weathering_kwargs['input_model'] = input_model
        if resolved_first_layer is not None:
            weathering_kwargs['resolved_first_layer'] = resolved_first_layer
        weathering_context = estimate_weathering_thickness_core_context_from_first_breaks(
            **weathering_kwargs
        )
        return build_refraction_weathering_replacement_core_context(
            weathering_result=weathering_context.app_weathering_result,
            core_weathering_model=weathering_context.core_weathering_model,
            apply_options=req.apply,
            job_dir=job_dir,
            resolved_first_layer=resolved_first_layer,
        )
    except RefractionWeatheringReplacementStaticsError:
        raise
    except (CoreRefractionWeatheringReplacementError, ValueError) as exc:
        raise RefractionWeatheringReplacementStaticsError(str(exc)) from exc


def build_refraction_weathering_replacement_statics(
    *,
    weathering_result: RefractionWeatheringThicknessResult,
    apply_options: RefractionStaticApplyOptions | None = None,
    job_dir: Path | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> RefractionWeatheringReplacementStaticsResult:
    """Compute weathering-replacement statics from weathering thickness."""
    return build_refraction_weathering_replacement_core_context(
        weathering_result=weathering_result,
        apply_options=apply_options,
        job_dir=job_dir,
        resolved_first_layer=resolved_first_layer,
    ).app_replacement_result


def build_refraction_weathering_replacement_core_context(
    *,
    weathering_result: RefractionWeatheringThicknessResult,
    core_weathering_model: CoreRefractionWeatheringModel | None = None,
    apply_options: RefractionStaticApplyOptions | None = None,
    job_dir: Path | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> _WeatheringReplacementCoreContext:
    """Call the external high-level replacement builder and map to the app DTO."""
    try:
        max_abs_shift_ms = _resolve_max_abs_shift_ms(apply_options)
        core_weathering = (
            core_weathering_model
            if core_weathering_model is not None
            else _core_weathering_model_from_app_result(weathering_result)
        )
        if resolved_first_layer is not None:
            validate_resolved_first_layer_velocity_match(
                weathering_velocity_m_s=float(core_weathering.weathering_velocity_m_s),
                resolved_first_layer=resolved_first_layer,
                name='weathering_result.weathering_velocity_m_s',
            )
        core_result = core_build_refraction_weathering_replacement_statics(
            weathering_model=core_weathering,
            max_abs_shift_ms=max_abs_shift_ms,
        )
        result = _app_replacement_result_from_core(
            core_result=core_result,
            weathering_result=weathering_result,
        )
        if job_dir is not None:
            write_refraction_weathering_replacement_artifacts(Path(job_dir), result)
        return _WeatheringReplacementCoreContext(
            core_replacement_result=core_result,
            app_replacement_result=result,
        )
    except RefractionWeatheringReplacementStaticsError:
        raise
    except (CoreRefractionWeatheringReplacementError, ValueError) as exc:
        raise RefractionWeatheringReplacementStaticsError(str(exc)) from exc


def _build_refraction_weathering_replacement_contract_result(
    *,
    weathering_result: RefractionWeatheringThicknessResult,
    apply_options: RefractionStaticApplyOptions | None = None,
    job_dir: Path | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> RefractionWeatheringReplacementStaticsResult:
    return build_refraction_weathering_replacement_core_context(
        weathering_result=weathering_result,
        apply_options=apply_options,
        job_dir=job_dir,
        resolved_first_layer=resolved_first_layer,
    ).app_replacement_result


def _app_replacement_result_from_core(
    *,
    core_result: CoreRefractionWeatheringReplacementResult,
    weathering_result: RefractionWeatheringThicknessResult,
) -> RefractionWeatheringReplacementStaticsResult:
    data = _validate_weathering_result(weathering_result)
    velocity = _validate_velocity_context(weathering_result, resolved_first_layer=None)
    result = RefractionWeatheringReplacementStaticsResult(
        bedrock_velocity_mode=_validate_velocity_mode(core_result.bedrock_velocity_mode),
        bedrock_slowness_s_per_m=velocity.bedrock_slowness_s_per_m,
        bedrock_velocity_m_s=velocity.bedrock_velocity_m_s,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        replacement_slowness_delta_s_per_m=(
            velocity.replacement_slowness_delta_s_per_m
        ),
        node_id=data.node_id,
        node_x_m=data.node_x_m,
        node_y_m=data.node_y_m,
        node_surface_elevation_m=data.node_surface_elevation_m,
        node_kind=data.node_kind,
        node_weathering_thickness_m=data.node_weathering_thickness_m,
        node_refractor_elevation_m=data.node_refractor_elevation_m,
        node_half_intercept_time_s=data.node_half_intercept_time_s,
        node_solution_status=data.node_solution_status,
        node_weathering_status=data.node_weathering_status,
        node_weathering_replacement_shift_s=_core_f64(
            core_result.node_weathering_replacement_shift_s
        ),
        node_weathering_replacement_shift_ms=_core_f64(
            core_result.node_weathering_replacement_shift_ms
        ),
        node_static_status=_core_status(core_result.node_static_status),
        node_pick_count=data.node_pick_count,
        node_used_pick_count=data.node_used_pick_count,
        node_rejected_pick_count=data.node_rejected_pick_count,
        node_residual_rms_s=data.node_residual_rms_s,
        node_residual_mad_s=data.node_residual_mad_s,
        source_endpoint_key=data.source_endpoint_key,
        source_id=data.source_id,
        source_node_id=data.source_node_id,
        source_x_m=data.source_x_m,
        source_y_m=data.source_y_m,
        source_surface_elevation_m=data.source_surface_elevation_m,
        source_half_intercept_time_s=data.source_half_intercept_time_s,
        source_weathering_thickness_m=data.source_weathering_thickness_m,
        source_refractor_elevation_m=data.source_refractor_elevation_m,
        source_weathering_replacement_shift_s=_core_f64(
            core_result.source_weathering_replacement_shift_s
        ),
        source_static_status=_core_status(core_result.source_static_status),
        receiver_endpoint_key=data.receiver_endpoint_key,
        receiver_id=data.receiver_id,
        receiver_node_id=data.receiver_node_id,
        receiver_x_m=data.receiver_x_m,
        receiver_y_m=data.receiver_y_m,
        receiver_surface_elevation_m=data.receiver_surface_elevation_m,
        receiver_half_intercept_time_s=data.receiver_half_intercept_time_s,
        receiver_weathering_thickness_m=data.receiver_weathering_thickness_m,
        receiver_refractor_elevation_m=data.receiver_refractor_elevation_m,
        receiver_weathering_replacement_shift_s=_core_f64(
            core_result.receiver_weathering_replacement_shift_s
        ),
        receiver_static_status=_core_status(core_result.receiver_static_status),
        sorted_trace_index=data.sorted_trace_index,
        valid_observation_mask_sorted=data.valid_observation_mask_sorted,
        used_observation_mask_sorted=data.used_observation_mask_sorted,
        source_endpoint_key_sorted=data.source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=data.receiver_endpoint_key_sorted,
        source_node_id_sorted=data.source_node_id_sorted,
        receiver_node_id_sorted=data.receiver_node_id_sorted,
        source_half_intercept_time_s_sorted=(
            data.source_half_intercept_time_s_sorted
        ),
        receiver_half_intercept_time_s_sorted=(
            data.receiver_half_intercept_time_s_sorted
        ),
        source_weathering_thickness_m_sorted=(
            data.source_weathering_thickness_m_sorted
        ),
        receiver_weathering_thickness_m_sorted=(
            data.receiver_weathering_thickness_m_sorted
        ),
        source_refractor_elevation_m_sorted=data.source_refractor_elevation_m_sorted,
        receiver_refractor_elevation_m_sorted=(
            data.receiver_refractor_elevation_m_sorted
        ),
        source_weathering_replacement_shift_s_sorted=_core_f64(
            core_result.source_weathering_replacement_shift_s_sorted
        ),
        receiver_weathering_replacement_shift_s_sorted=_core_f64(
            core_result.receiver_weathering_replacement_shift_s_sorted
        ),
        weathering_replacement_trace_shift_s_sorted=_core_f64(
            core_result.weathering_replacement_trace_shift_s_sorted
        ),
        source_static_status_sorted=_core_status(core_result.source_static_status_sorted),
        receiver_static_status_sorted=_core_status(
            core_result.receiver_static_status_sorted
        ),
        trace_static_status_sorted=_core_status(core_result.trace_static_status_sorted),
        trace_static_valid_mask_sorted=np.ascontiguousarray(
            core_result.trace_static_valid_mask_sorted,
            dtype=bool,
        ),
        estimated_first_break_time_s_sorted=(
            data.estimated_first_break_time_s_sorted
        ),
        first_break_residual_s_sorted=data.first_break_residual_s_sorted,
        row_trace_index_sorted=data.row_trace_index_sorted,
        row_source_node_id=data.row_source_node_id,
        row_receiver_node_id=data.row_receiver_node_id,
        row_distance_m=data.row_distance_m,
        observed_pick_time_s=data.observed_pick_time_s,
        modeled_pick_time_s=data.modeled_pick_time_s,
        residual_time_s=data.residual_time_s,
        used_row_mask=data.used_row_mask,
        rejected_by_robust_mask=data.rejected_by_robust_mask,
        qc=_app_replacement_qc_from_core(core_result.qc, velocity=velocity),
        active_cell_id=data.active_cell_id,
        inactive_cell_id=data.inactive_cell_id,
        cell_bedrock_slowness_s_per_m=data.cell_bedrock_slowness_s_per_m,
        cell_bedrock_velocity_m_s=data.cell_bedrock_velocity_m_s,
        cell_velocity_status=data.cell_velocity_status,
        row_midpoint_cell_id=data.row_midpoint_cell_id,
        node_v2_cell_id=data.node_v2_cell_id,
        node_v2_m_s=data.node_v2_m_s,
        node_v2_status=data.node_v2_status,
        source_v2_cell_id=data.source_v2_cell_id,
        source_v2_m_s=data.source_v2_m_s,
        source_v2_status=data.source_v2_status,
        receiver_v2_cell_id=data.receiver_v2_cell_id,
        receiver_v2_m_s=data.receiver_v2_m_s,
        receiver_v2_status=data.receiver_v2_status,
        source_v2_cell_id_sorted=data.source_v2_cell_id_sorted,
        source_v2_m_s_sorted=data.source_v2_m_s_sorted,
        source_v2_status_sorted=data.source_v2_status_sorted,
        receiver_v2_cell_id_sorted=data.receiver_v2_cell_id_sorted,
        receiver_v2_m_s_sorted=data.receiver_v2_m_s_sorted,
        receiver_v2_status_sorted=data.receiver_v2_status_sorted,
    )
    _validate_core_replacement_shapes(result)
    return result


def _core_weathering_model_from_app_result(
    weathering_result: RefractionWeatheringThicknessResult,
) -> CoreRefractionWeatheringModel:
    data = _validate_weathering_result(weathering_result)
    velocity = _validate_velocity_context(weathering_result, resolved_first_layer=None)
    node_v2 = _local_or_scalar_v2(
        data.node_v2_m_s,
        shape=data.node_weathering_thickness_m.shape,
        scalar_v2_m_s=velocity.bedrock_velocity_m_s,
    )
    source_v2 = _local_or_scalar_v2(
        data.source_v2_m_s,
        shape=data.source_weathering_thickness_m.shape,
        scalar_v2_m_s=velocity.bedrock_velocity_m_s,
    )
    receiver_v2 = _local_or_scalar_v2(
        data.receiver_v2_m_s,
        shape=data.receiver_weathering_thickness_m.shape,
        scalar_v2_m_s=velocity.bedrock_velocity_m_s,
    )
    node_v2_status = _local_or_ok_status(data.node_v2_status, data.node_id.shape)
    source_v2_status = _local_or_ok_status(
        data.source_v2_status,
        data.source_endpoint_key.shape,
    )
    receiver_v2_status = _local_or_ok_status(
        data.receiver_v2_status,
        data.receiver_endpoint_key.shape,
    )
    return CoreRefractionWeatheringModel(
        file_id='',
        n_traces=data.n_traces,
        bedrock_velocity_mode=velocity.mode,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        bedrock_velocity_m_s=velocity.bedrock_velocity_m_s,
        bedrock_slowness_s_per_m=velocity.bedrock_slowness_s_per_m,
        bedrock_velocity_status='ok',
        v2_m_s=velocity.bedrock_velocity_m_s,
        node_id=data.node_id,
        node_x_m=data.node_x_m,
        node_y_m=data.node_y_m,
        node_surface_elevation_m=data.node_surface_elevation_m,
        node_half_intercept_time_s=data.node_half_intercept_time_s,
        node_v1_m_s=np.full(
            data.node_id.shape,
            velocity.weathering_velocity_m_s,
            dtype=np.float64,
        ),
        node_v2_m_s=node_v2,
        node_weathering_thickness_m=data.node_weathering_thickness_m,
        node_refractor_elevation_m=data.node_refractor_elevation_m,
        node_solution_status=data.node_solution_status,
        node_local_v2_status=node_v2_status,
        node_weathering_status=data.node_weathering_status,
        node_pick_count=data.node_pick_count,
        node_used_observation_count=data.node_used_pick_count,
        node_rejected_observation_count=data.node_rejected_pick_count,
        source_endpoint=CoreRefractionWeatheringEndpointComponents(
            endpoint_key=data.source_endpoint_key,
            endpoint_id=data.source_id,
            node_id=data.source_node_id,
            x_m=data.source_x_m,
            y_m=data.source_y_m,
            surface_elevation_m=data.source_surface_elevation_m,
            half_intercept_time_s=data.source_half_intercept_time_s,
            v1_m_s=np.full(
                data.source_endpoint_key.shape,
                velocity.weathering_velocity_m_s,
                dtype=np.float64,
            ),
            v2_m_s=source_v2,
            weathering_thickness_m=data.source_weathering_thickness_m,
            refractor_elevation_m=data.source_refractor_elevation_m,
            solution_status=data.source_weathering_status,
            local_v2_status=source_v2_status,
            weathering_status=data.source_weathering_status,
            pick_count=np.zeros(data.source_endpoint_key.shape, dtype=np.int64),
            used_observation_count=np.zeros(
                data.source_endpoint_key.shape,
                dtype=np.int64,
            ),
            rejected_observation_count=np.zeros(
                data.source_endpoint_key.shape,
                dtype=np.int64,
            ),
        ),
        receiver_endpoint=CoreRefractionWeatheringEndpointComponents(
            endpoint_key=data.receiver_endpoint_key,
            endpoint_id=data.receiver_id,
            node_id=data.receiver_node_id,
            x_m=data.receiver_x_m,
            y_m=data.receiver_y_m,
            surface_elevation_m=data.receiver_surface_elevation_m,
            half_intercept_time_s=data.receiver_half_intercept_time_s,
            v1_m_s=np.full(
                data.receiver_endpoint_key.shape,
                velocity.weathering_velocity_m_s,
                dtype=np.float64,
            ),
            v2_m_s=receiver_v2,
            weathering_thickness_m=data.receiver_weathering_thickness_m,
            refractor_elevation_m=data.receiver_refractor_elevation_m,
            solution_status=data.receiver_weathering_status,
            local_v2_status=receiver_v2_status,
            weathering_status=data.receiver_weathering_status,
            pick_count=np.zeros(data.receiver_endpoint_key.shape, dtype=np.int64),
            used_observation_count=np.zeros(
                data.receiver_endpoint_key.shape,
                dtype=np.int64,
            ),
            rejected_observation_count=np.zeros(
                data.receiver_endpoint_key.shape,
                dtype=np.int64,
            ),
        ),
        trace_index_sorted=data.sorted_trace_index,
        source_endpoint_key_sorted=data.source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=data.receiver_endpoint_key_sorted,
        source_node_id_sorted=data.source_node_id_sorted,
        receiver_node_id_sorted=data.receiver_node_id_sorted,
        source_weathering_thickness_m_sorted=(
            data.source_weathering_thickness_m_sorted
        ),
        receiver_weathering_thickness_m_sorted=(
            data.receiver_weathering_thickness_m_sorted
        ),
        source_refractor_elevation_m_sorted=data.source_refractor_elevation_m_sorted,
        receiver_refractor_elevation_m_sorted=(
            data.receiver_refractor_elevation_m_sorted
        ),
        source_weathering_status_sorted=data.source_weathering_status_sorted,
        receiver_weathering_status_sorted=data.receiver_weathering_status_sorted,
        trace_weathering_thickness_m_sorted=np.full(
            data.sorted_trace_index.shape,
            np.nan,
            dtype=np.float64,
        ),
        trace_weathering_status_sorted=np.full(
            data.sorted_trace_index.shape,
            'ok',
            dtype=_STATUS_DTYPE,
        ),
        cell_id=_optional_or_empty_i64(data.active_cell_id),
        cell_v2_m_s=_optional_or_empty_f64(data.cell_bedrock_velocity_m_s),
        cell_velocity_status=_optional_or_empty_status(data.cell_velocity_status),
        cell_observation_count=_cell_observation_count_from_qc(weathering_result.qc),
        qc=_core_qc(getattr(weathering_result, 'qc', {})),
    )


def _local_or_ok_status(
    local_status: np.ndarray | None,
    shape: tuple[int, ...],
) -> np.ndarray:
    if local_status is None:
        return np.full(shape, 'ok', dtype=_STATUS_DTYPE)
    return np.ascontiguousarray(local_status, dtype=_STATUS_DTYPE)


def _optional_or_empty_i64(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.int64)
    return np.ascontiguousarray(values, dtype=np.int64)


def _optional_or_empty_f64(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.float64)
    return np.ascontiguousarray(values, dtype=np.float64)


def _optional_or_empty_status(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=_STATUS_DTYPE)
    return np.ascontiguousarray(values, dtype=_STATUS_DTYPE)


def _cell_observation_count_from_qc(qc: dict[str, Any]) -> np.ndarray:
    values = qc.get('cell_observation_count')
    if values is None:
        return np.empty(0, dtype=np.int64)
    return np.ascontiguousarray(values, dtype=np.int64)


def _core_f64(values: object) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _core_status(values: object) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=_STATUS_DTYPE)


def _core_qc(values: object) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise RefractionWeatheringReplacementStaticsError('core qc must be a dict')
    return dict(values)


def _app_replacement_qc_from_core(
    values: object,
    *,
    velocity: _VelocityContext,
) -> dict[str, Any]:
    qc = _core_qc(values)
    qc['bedrock_velocity_m_s'] = velocity.bedrock_velocity_m_s
    qc['bedrock_slowness_s_per_m'] = velocity.bedrock_slowness_s_per_m
    qc['weathering_velocity_m_s'] = velocity.weathering_velocity_m_s
    qc['replacement_slowness_delta_s_per_m'] = (
        velocity.replacement_slowness_delta_s_per_m
    )
    return qc


def _validate_core_replacement_shapes(
    result: RefractionWeatheringReplacementStaticsResult,
) -> None:
    n_nodes = int(result.node_id.shape[0])
    n_sources = int(result.source_endpoint_key.shape[0])
    n_receivers = int(result.receiver_endpoint_key.shape[0])
    n_traces = int(result.sorted_trace_index.shape[0])
    _require_shape(
        result.node_weathering_replacement_shift_s,
        (n_nodes,),
        'core_result.node_weathering_replacement_shift_s',
    )
    _require_shape(result.node_static_status, (n_nodes,), 'core_result.node_static_status')
    _require_shape(
        result.source_weathering_replacement_shift_s,
        (n_sources,),
        'core_result.source_weathering_replacement_shift_s',
    )
    _require_shape(
        result.source_static_status,
        (n_sources,),
        'core_result.source_static_status',
    )
    _require_shape(
        result.receiver_weathering_replacement_shift_s,
        (n_receivers,),
        'core_result.receiver_weathering_replacement_shift_s',
    )
    _require_shape(
        result.receiver_static_status,
        (n_receivers,),
        'core_result.receiver_static_status',
    )
    _require_shape(
        result.weathering_replacement_trace_shift_s_sorted,
        (n_traces,),
        'core_result.weathering_replacement_trace_shift_s_sorted',
    )
    _require_shape(
        result.trace_static_status_sorted,
        (n_traces,),
        'core_result.trace_static_status_sorted',
    )
    _require_shape(
        result.trace_static_valid_mask_sorted,
        (n_traces,),
        'core_result.trace_static_valid_mask_sorted',
    )


def _require_shape(values: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if np.asarray(values).shape != shape:
        raise RefractionWeatheringReplacementStaticsError(
            f'{name} shape mismatch: expected {shape}'
        )


def compute_weathering_replacement_shift_s(
    *,
    weathering_thickness_m: np.ndarray,
    weathering_velocity_m_s: float,
    bedrock_velocity_m_s: float | np.ndarray,
) -> np.ndarray:
    """Compute ``shift = z * (1/vb - 1/vw)`` in seconds."""
    try:
        return core_compute_weathering_replacement_shift_s(
            weathering_thickness_m=weathering_thickness_m,
            weathering_velocity_m_s=weathering_velocity_m_s,
            bedrock_velocity_m_s=bedrock_velocity_m_s,
        )
    except ValueError as exc:
        raise RefractionWeatheringReplacementStaticsError(str(exc)) from exc


def compute_weathering_replacement_shift_scalar_s(
    *,
    weathering_thickness_m: float,
    weathering_velocity_m_s: float,
    bedrock_velocity_m_s: float,
) -> float:
    """Scalar wrapper around ``compute_weathering_replacement_shift_s``."""
    value = compute_weathering_replacement_shift_s(
        weathering_thickness_m=np.asarray([weathering_thickness_m], dtype=np.float64),
        weathering_velocity_m_s=weathering_velocity_m_s,
        bedrock_velocity_m_s=bedrock_velocity_m_s,
    )
    return float(value[0])


def _local_or_scalar_v2(
    local_v2_m_s: np.ndarray | None,
    *,
    shape: tuple[int, ...],
    scalar_v2_m_s: float,
) -> np.ndarray:
    if local_v2_m_s is None:
        return np.full(shape, scalar_v2_m_s, dtype=np.float64)
    return np.ascontiguousarray(local_v2_m_s, dtype=np.float64)


def write_refraction_weathering_replacement_artifacts(
    job_dir: Path,
    result: RefractionWeatheringReplacementStaticsResult,
) -> dict[str, Path]:
    """Write weathering-replacement QC JSON and CSV tables."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / REFRACTION_WEATHERING_REPLACEMENT_QC_JSON_NAME
    node_path = root / REFRACTION_WEATHERING_REPLACEMENT_NODES_CSV_NAME
    source_path = root / REFRACTION_WEATHERING_REPLACEMENT_SOURCES_CSV_NAME
    receiver_path = root / REFRACTION_WEATHERING_REPLACEMENT_RECEIVERS_CSV_NAME
    trace_path = root / REFRACTION_WEATHERING_REPLACEMENT_TRACE_PREVIEW_CSV_NAME
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


def _validate_weathering_result(
    weathering_result: RefractionWeatheringThicknessResult,
) -> _ValidatedWeathering:
    node_id = _coerce_1d_integer(
        _required(weathering_result, 'node_id'),
        name='weathering_result.node_id',
    )
    n_nodes = int(node_id.shape[0])
    if n_nodes <= 0:
        raise RefractionWeatheringReplacementStaticsError(
            'node_id must contain at least one node'
        )
    if np.unique(node_id).shape[0] != n_nodes:
        raise RefractionWeatheringReplacementStaticsError(
            'node_id values must be unique'
        )
    node_shape = (n_nodes,)

    sorted_trace_index = _coerce_1d_integer(
        _required(weathering_result, 'sorted_trace_index'),
        name='weathering_result.sorted_trace_index',
    )
    n_traces = int(sorted_trace_index.shape[0])
    trace_shape = (n_traces,)

    source_endpoint_key = _coerce_1d_string(
        _required(weathering_result, 'source_endpoint_key'),
        name='weathering_result.source_endpoint_key',
    )
    receiver_endpoint_key = _coerce_1d_string(
        _required(weathering_result, 'receiver_endpoint_key'),
        name='weathering_result.receiver_endpoint_key',
    )
    source_shape = (int(source_endpoint_key.shape[0]),)
    receiver_shape = (int(receiver_endpoint_key.shape[0]),)
    source_node_id = _coerce_1d_integer(
        _required(weathering_result, 'source_node_id'),
        name='weathering_result.source_node_id',
        expected_shape=source_shape,
    )
    receiver_node_id = _coerce_1d_integer(
        _required(weathering_result, 'receiver_node_id'),
        name='weathering_result.receiver_node_id',
        expected_shape=receiver_shape,
    )
    _validate_endpoint_nodes(source_node_id, node_id, name='source_node_id')
    _validate_endpoint_nodes(receiver_node_id, node_id, name='receiver_node_id')

    return _ValidatedWeathering(
        node_id=node_id,
        node_x_m=_coerce_1d_float(
            _required(weathering_result, 'node_x_m'),
            name='weathering_result.node_x_m',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_y_m=_coerce_1d_float(
            _required(weathering_result, 'node_y_m'),
            name='weathering_result.node_y_m',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_surface_elevation_m=_coerce_1d_float(
            _required(weathering_result, 'node_surface_elevation_m'),
            name='weathering_result.node_surface_elevation_m',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_kind=_coerce_1d_string(
            _required(weathering_result, 'node_kind'),
            name='weathering_result.node_kind',
            expected_shape=node_shape,
            dtype=_STATUS_DTYPE,
        ),
        node_weathering_thickness_m=_coerce_1d_float(
            _required(weathering_result, 'node_weathering_thickness_m'),
            name='weathering_result.node_weathering_thickness_m',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_refractor_elevation_m=_coerce_1d_float(
            _required(weathering_result, 'node_refractor_elevation_m'),
            name='weathering_result.node_refractor_elevation_m',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_half_intercept_time_s=_coerce_1d_float(
            _required(weathering_result, 'node_half_intercept_time_s'),
            name='weathering_result.node_half_intercept_time_s',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_solution_status=_coerce_1d_string(
            _required(weathering_result, 'node_solution_status'),
            name='weathering_result.node_solution_status',
            expected_shape=node_shape,
            dtype=_STATUS_DTYPE,
        ),
        node_weathering_status=_coerce_1d_string(
            _required(weathering_result, 'node_weathering_status'),
            name='weathering_result.node_weathering_status',
            expected_shape=node_shape,
            dtype=_STATUS_DTYPE,
        ),
        node_pick_count=_coerce_1d_integer(
            _required(weathering_result, 'node_pick_count'),
            name='weathering_result.node_pick_count',
            expected_shape=node_shape,
        ),
        node_used_pick_count=_coerce_1d_integer(
            _required(weathering_result, 'node_used_pick_count'),
            name='weathering_result.node_used_pick_count',
            expected_shape=node_shape,
        ),
        node_rejected_pick_count=_coerce_1d_integer(
            _required(weathering_result, 'node_rejected_pick_count'),
            name='weathering_result.node_rejected_pick_count',
            expected_shape=node_shape,
        ),
        node_residual_rms_s=_coerce_1d_float(
            _required(weathering_result, 'node_residual_rms_s'),
            name='weathering_result.node_residual_rms_s',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_residual_mad_s=_coerce_1d_float(
            _required(weathering_result, 'node_residual_mad_s'),
            name='weathering_result.node_residual_mad_s',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        active_cell_id=_optional_1d_integer(
            weathering_result,
            'active_cell_id',
            name='weathering_result.active_cell_id',
        ),
        inactive_cell_id=_optional_1d_integer(
            weathering_result,
            'inactive_cell_id',
            name='weathering_result.inactive_cell_id',
        ),
        cell_bedrock_slowness_s_per_m=_optional_1d_float(
            weathering_result,
            'cell_bedrock_slowness_s_per_m',
            name='weathering_result.cell_bedrock_slowness_s_per_m',
        ),
        cell_bedrock_velocity_m_s=_optional_1d_float(
            weathering_result,
            'cell_bedrock_velocity_m_s',
            name='weathering_result.cell_bedrock_velocity_m_s',
        ),
        cell_velocity_status=_optional_1d_string(
            weathering_result,
            'cell_velocity_status',
            name='weathering_result.cell_velocity_status',
            dtype=_STATUS_DTYPE,
        ),
        row_midpoint_cell_id=_optional_1d_integer(
            weathering_result,
            'row_midpoint_cell_id',
            name='weathering_result.row_midpoint_cell_id',
        ),
        node_v2_cell_id=_optional_1d_integer(
            weathering_result,
            'node_v2_cell_id',
            name='weathering_result.node_v2_cell_id',
            expected_shape=node_shape,
        ),
        node_v2_m_s=_optional_1d_float(
            weathering_result,
            'node_v2_m_s',
            name='weathering_result.node_v2_m_s',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_v2_status=_optional_1d_string(
            weathering_result,
            'node_v2_status',
            name='weathering_result.node_v2_status',
            expected_shape=node_shape,
            dtype=_STATUS_DTYPE,
        ),
        source_endpoint_key=source_endpoint_key,
        source_id=_coerce_1d_integer(
            _required(weathering_result, 'source_id'),
            name='weathering_result.source_id',
            expected_shape=source_shape,
        ),
        source_node_id=source_node_id,
        source_x_m=_coerce_1d_float(
            _required(weathering_result, 'source_x_m'),
            name='weathering_result.source_x_m',
            expected_shape=source_shape,
            allow_nonfinite=True,
        ),
        source_y_m=_coerce_1d_float(
            _required(weathering_result, 'source_y_m'),
            name='weathering_result.source_y_m',
            expected_shape=source_shape,
            allow_nonfinite=True,
        ),
        source_surface_elevation_m=_coerce_1d_float(
            _required(weathering_result, 'source_surface_elevation_m'),
            name='weathering_result.source_surface_elevation_m',
            expected_shape=source_shape,
            allow_nonfinite=True,
        ),
        source_half_intercept_time_s=_coerce_1d_float(
            _required(weathering_result, 'source_half_intercept_time_s'),
            name='weathering_result.source_half_intercept_time_s',
            expected_shape=source_shape,
            allow_nonfinite=True,
        ),
        source_weathering_thickness_m=_coerce_1d_float(
            _required(weathering_result, 'source_weathering_thickness_m'),
            name='weathering_result.source_weathering_thickness_m',
            expected_shape=source_shape,
            allow_nonfinite=True,
        ),
        source_refractor_elevation_m=_coerce_1d_float(
            _required(weathering_result, 'source_refractor_elevation_m'),
            name='weathering_result.source_refractor_elevation_m',
            expected_shape=source_shape,
            allow_nonfinite=True,
        ),
        source_weathering_status=_coerce_1d_string(
            _required(weathering_result, 'source_weathering_status'),
            name='weathering_result.source_weathering_status',
            expected_shape=source_shape,
            dtype=_STATUS_DTYPE,
        ),
        source_v2_cell_id=_optional_1d_integer(
            weathering_result,
            'source_v2_cell_id',
            name='weathering_result.source_v2_cell_id',
            expected_shape=source_shape,
        ),
        source_v2_m_s=_optional_1d_float(
            weathering_result,
            'source_v2_m_s',
            name='weathering_result.source_v2_m_s',
            expected_shape=source_shape,
            allow_nonfinite=True,
        ),
        source_v2_status=_optional_1d_string(
            weathering_result,
            'source_v2_status',
            name='weathering_result.source_v2_status',
            expected_shape=source_shape,
            dtype=_STATUS_DTYPE,
        ),
        receiver_endpoint_key=receiver_endpoint_key,
        receiver_id=_coerce_1d_integer(
            _required(weathering_result, 'receiver_id'),
            name='weathering_result.receiver_id',
            expected_shape=receiver_shape,
        ),
        receiver_node_id=receiver_node_id,
        receiver_x_m=_coerce_1d_float(
            _required(weathering_result, 'receiver_x_m'),
            name='weathering_result.receiver_x_m',
            expected_shape=receiver_shape,
            allow_nonfinite=True,
        ),
        receiver_y_m=_coerce_1d_float(
            _required(weathering_result, 'receiver_y_m'),
            name='weathering_result.receiver_y_m',
            expected_shape=receiver_shape,
            allow_nonfinite=True,
        ),
        receiver_surface_elevation_m=_coerce_1d_float(
            _required(weathering_result, 'receiver_surface_elevation_m'),
            name='weathering_result.receiver_surface_elevation_m',
            expected_shape=receiver_shape,
            allow_nonfinite=True,
        ),
        receiver_half_intercept_time_s=_coerce_1d_float(
            _required(weathering_result, 'receiver_half_intercept_time_s'),
            name='weathering_result.receiver_half_intercept_time_s',
            expected_shape=receiver_shape,
            allow_nonfinite=True,
        ),
        receiver_weathering_thickness_m=_coerce_1d_float(
            _required(weathering_result, 'receiver_weathering_thickness_m'),
            name='weathering_result.receiver_weathering_thickness_m',
            expected_shape=receiver_shape,
            allow_nonfinite=True,
        ),
        receiver_refractor_elevation_m=_coerce_1d_float(
            _required(weathering_result, 'receiver_refractor_elevation_m'),
            name='weathering_result.receiver_refractor_elevation_m',
            expected_shape=receiver_shape,
            allow_nonfinite=True,
        ),
        receiver_weathering_status=_coerce_1d_string(
            _required(weathering_result, 'receiver_weathering_status'),
            name='weathering_result.receiver_weathering_status',
            expected_shape=receiver_shape,
            dtype=_STATUS_DTYPE,
        ),
        receiver_v2_cell_id=_optional_1d_integer(
            weathering_result,
            'receiver_v2_cell_id',
            name='weathering_result.receiver_v2_cell_id',
            expected_shape=receiver_shape,
        ),
        receiver_v2_m_s=_optional_1d_float(
            weathering_result,
            'receiver_v2_m_s',
            name='weathering_result.receiver_v2_m_s',
            expected_shape=receiver_shape,
            allow_nonfinite=True,
        ),
        receiver_v2_status=_optional_1d_string(
            weathering_result,
            'receiver_v2_status',
            name='weathering_result.receiver_v2_status',
            expected_shape=receiver_shape,
            dtype=_STATUS_DTYPE,
        ),
        sorted_trace_index=sorted_trace_index,
        valid_observation_mask_sorted=_coerce_1d_bool(
            _required(weathering_result, 'valid_observation_mask_sorted'),
            name='weathering_result.valid_observation_mask_sorted',
            expected_shape=trace_shape,
        ),
        used_observation_mask_sorted=_coerce_1d_bool(
            _required(weathering_result, 'used_observation_mask_sorted'),
            name='weathering_result.used_observation_mask_sorted',
            expected_shape=trace_shape,
        ),
        source_endpoint_key_sorted=_coerce_1d_string(
            _required(weathering_result, 'source_endpoint_key_sorted'),
            name='weathering_result.source_endpoint_key_sorted',
            expected_shape=trace_shape,
        ),
        receiver_endpoint_key_sorted=_coerce_1d_string(
            _required(weathering_result, 'receiver_endpoint_key_sorted'),
            name='weathering_result.receiver_endpoint_key_sorted',
            expected_shape=trace_shape,
        ),
        source_node_id_sorted=_coerce_1d_integer(
            _required(weathering_result, 'source_node_id_sorted'),
            name='weathering_result.source_node_id_sorted',
            expected_shape=trace_shape,
        ),
        receiver_node_id_sorted=_coerce_1d_integer(
            _required(weathering_result, 'receiver_node_id_sorted'),
            name='weathering_result.receiver_node_id_sorted',
            expected_shape=trace_shape,
        ),
        source_half_intercept_time_s_sorted=_coerce_1d_float(
            _required(weathering_result, 'source_half_intercept_time_s_sorted'),
            name='weathering_result.source_half_intercept_time_s_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        receiver_half_intercept_time_s_sorted=_coerce_1d_float(
            _required(weathering_result, 'receiver_half_intercept_time_s_sorted'),
            name='weathering_result.receiver_half_intercept_time_s_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        source_weathering_thickness_m_sorted=_coerce_1d_float(
            _required(weathering_result, 'source_weathering_thickness_m_sorted'),
            name='weathering_result.source_weathering_thickness_m_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        receiver_weathering_thickness_m_sorted=_coerce_1d_float(
            _required(weathering_result, 'receiver_weathering_thickness_m_sorted'),
            name='weathering_result.receiver_weathering_thickness_m_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        source_refractor_elevation_m_sorted=_coerce_1d_float(
            _required(weathering_result, 'source_refractor_elevation_m_sorted'),
            name='weathering_result.source_refractor_elevation_m_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        receiver_refractor_elevation_m_sorted=_coerce_1d_float(
            _required(weathering_result, 'receiver_refractor_elevation_m_sorted'),
            name='weathering_result.receiver_refractor_elevation_m_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        source_weathering_status_sorted=_coerce_1d_string(
            _required(weathering_result, 'source_weathering_status_sorted'),
            name='weathering_result.source_weathering_status_sorted',
            expected_shape=trace_shape,
            dtype=_STATUS_DTYPE,
        ),
        receiver_weathering_status_sorted=_coerce_1d_string(
            _required(weathering_result, 'receiver_weathering_status_sorted'),
            name='weathering_result.receiver_weathering_status_sorted',
            expected_shape=trace_shape,
            dtype=_STATUS_DTYPE,
        ),
        source_v2_cell_id_sorted=_optional_1d_integer(
            weathering_result,
            'source_v2_cell_id_sorted',
            name='weathering_result.source_v2_cell_id_sorted',
            expected_shape=trace_shape,
        ),
        source_v2_m_s_sorted=_optional_1d_float(
            weathering_result,
            'source_v2_m_s_sorted',
            name='weathering_result.source_v2_m_s_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        source_v2_status_sorted=_optional_1d_string(
            weathering_result,
            'source_v2_status_sorted',
            name='weathering_result.source_v2_status_sorted',
            expected_shape=trace_shape,
            dtype=_STATUS_DTYPE,
        ),
        receiver_v2_cell_id_sorted=_optional_1d_integer(
            weathering_result,
            'receiver_v2_cell_id_sorted',
            name='weathering_result.receiver_v2_cell_id_sorted',
            expected_shape=trace_shape,
        ),
        receiver_v2_m_s_sorted=_optional_1d_float(
            weathering_result,
            'receiver_v2_m_s_sorted',
            name='weathering_result.receiver_v2_m_s_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        receiver_v2_status_sorted=_optional_1d_string(
            weathering_result,
            'receiver_v2_status_sorted',
            name='weathering_result.receiver_v2_status_sorted',
            expected_shape=trace_shape,
            dtype=_STATUS_DTYPE,
        ),
        estimated_first_break_time_s_sorted=_coerce_1d_float(
            _required(weathering_result, 'estimated_first_break_time_s_sorted'),
            name='weathering_result.estimated_first_break_time_s_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        first_break_residual_s_sorted=_coerce_1d_float(
            _required(weathering_result, 'first_break_residual_s_sorted'),
            name='weathering_result.first_break_residual_s_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        row_trace_index_sorted=_coerce_1d_integer(
            _required(weathering_result, 'row_trace_index_sorted'),
            name='weathering_result.row_trace_index_sorted',
        ),
        row_source_node_id=_coerce_1d_integer(
            _required(weathering_result, 'row_source_node_id'),
            name='weathering_result.row_source_node_id',
        ),
        row_receiver_node_id=_coerce_1d_integer(
            _required(weathering_result, 'row_receiver_node_id'),
            name='weathering_result.row_receiver_node_id',
        ),
        row_distance_m=_coerce_1d_float(
            _required(weathering_result, 'row_distance_m'),
            name='weathering_result.row_distance_m',
        ),
        observed_pick_time_s=_coerce_1d_float(
            _required(weathering_result, 'observed_pick_time_s'),
            name='weathering_result.observed_pick_time_s',
        ),
        modeled_pick_time_s=_coerce_1d_float(
            _required(weathering_result, 'modeled_pick_time_s'),
            name='weathering_result.modeled_pick_time_s',
        ),
        residual_time_s=_coerce_1d_float(
            _required(weathering_result, 'residual_time_s'),
            name='weathering_result.residual_time_s',
        ),
        used_row_mask=_coerce_1d_bool(
            _required(weathering_result, 'used_row_mask'),
            name='weathering_result.used_row_mask',
        ),
        rejected_by_robust_mask=_coerce_1d_bool(
            _required(weathering_result, 'rejected_by_robust_mask'),
            name='weathering_result.rejected_by_robust_mask',
        ),
    )


def _validate_endpoint_nodes(
    endpoint_node_id: np.ndarray,
    node_id: np.ndarray,
    *,
    name: str,
) -> None:
    known = {int(node) for node in np.asarray(node_id, dtype=np.int64).tolist()}
    missing = [
        int(node)
        for node in np.asarray(endpoint_node_id, dtype=np.int64).reshape(-1).tolist()
        if int(node) not in known
    ]
    if missing:
        raise RefractionWeatheringReplacementStaticsError(
            f'{name} references unknown node_id {missing[0]}'
        )


def _validate_velocity_context(
    weathering_result: RefractionWeatheringThicknessResult,
    *,
    resolved_first_layer: ResolvedRefractionFirstLayer | None,
) -> _VelocityContext:
    mode = _validate_velocity_mode(_required(weathering_result, 'bedrock_velocity_mode'))
    weathering_velocity = _positive_finite(
        _required(weathering_result, 'weathering_velocity_m_s'),
        name='weathering_result.weathering_velocity_m_s',
    )
    weathering_velocity = validate_resolved_first_layer_velocity_match(
        weathering_velocity_m_s=weathering_velocity,
        resolved_first_layer=resolved_first_layer,
        name='weathering_result.weathering_velocity_m_s',
    )
    bedrock_slowness = _positive_finite(
        _required(weathering_result, 'bedrock_slowness_s_per_m'),
        name='weathering_result.bedrock_slowness_s_per_m',
    )
    bedrock_velocity = _positive_finite(
        _required(weathering_result, 'bedrock_velocity_m_s'),
        name='weathering_result.bedrock_velocity_m_s',
    )
    if bedrock_velocity <= weathering_velocity:
        raise RefractionWeatheringReplacementStaticsError(
            'bedrock_velocity_m_s must be greater than weathering_velocity_m_s'
        )
    derived_slowness = 1.0 / bedrock_velocity
    slowness_tol = max(1.0e-12, abs(bedrock_slowness) * _SLOWNESS_RTOL)
    if abs(derived_slowness - bedrock_slowness) > slowness_tol:
        raise RefractionWeatheringReplacementStaticsError(
            'bedrock_velocity_m_s does not match bedrock_slowness_s_per_m'
        )
    return _VelocityContext(
        mode=mode,
        bedrock_slowness_s_per_m=bedrock_slowness,
        bedrock_velocity_m_s=bedrock_velocity,
        weathering_velocity_m_s=weathering_velocity,
        replacement_slowness_delta_s_per_m=(
            1.0 / bedrock_velocity - 1.0 / weathering_velocity
        ),
    )


def _resolve_max_abs_shift_ms(
    apply_options: RefractionStaticApplyOptions | None,
) -> float | None:
    if apply_options is None:
        return None
    value = getattr(apply_options, 'max_abs_shift_ms', None)
    if value is None:
        return None
    return _positive_finite(value, name='apply.max_abs_shift_ms')


def _required(owner: object, field: str) -> object:
    try:
        value = getattr(owner, field)
    except AttributeError as exc:
        raise RefractionWeatheringReplacementStaticsError(
            f'{field} is required'
        ) from exc
    if value is None:
        raise RefractionWeatheringReplacementStaticsError(f'{field} is required')
    return value


def _coerce_1d_float(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    allow_nonfinite: bool = False,
) -> np.ndarray:
    return _coerce_1d_real_numeric_float64(
        values,
        name=name,
        expected_shape=expected_shape,
        allow_nonfinite=allow_nonfinite,
        error_type=RefractionWeatheringReplacementStaticsError,
    )


def _optional_1d_float(
    owner: object,
    field: str,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    allow_nonfinite: bool = False,
) -> np.ndarray | None:
    value = getattr(owner, field, None)
    if value is None:
        return None
    return _coerce_1d_float(
        value,
        name=name,
        expected_shape=expected_shape,
        allow_nonfinite=allow_nonfinite,
    )


def _coerce_1d_integer(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    return _coerce_1d_integer_int64(
        values,
        name=name,
        expected_shape=expected_shape,
        nonfinite_message='must contain finite values',
        error_type=RefractionWeatheringReplacementStaticsError,
    )


def _optional_1d_integer(
    owner: object,
    field: str,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray | None:
    value = getattr(owner, field, None)
    if value is None:
        return None
    return _coerce_1d_integer(value, name=name, expected_shape=expected_shape)


def _coerce_1d_bool(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    return _coerce_1d_bool_array(
        values,
        name=name,
        expected_shape=expected_shape,
        error_type=RefractionWeatheringReplacementStaticsError,
    )


def _coerce_1d_string(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    dtype: object = _ENDPOINT_KEY_DTYPE,
) -> np.ndarray:
    return _coerce_1d_string_array(
        values,
        name=name,
        expected_shape=expected_shape,
        allow_non_string_dtype=True,
        output_dtype=dtype,
        error_type=RefractionWeatheringReplacementStaticsError,
    )


def _optional_1d_string(
    owner: object,
    field: str,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    dtype: object = _ENDPOINT_KEY_DTYPE,
) -> np.ndarray | None:
    value = getattr(owner, field, None)
    if value is None:
        return None
    return _coerce_1d_string(
        value,
        name=name,
        expected_shape=expected_shape,
        dtype=dtype,
    )


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise RefractionWeatheringReplacementStaticsError(
            f'{name} must be finite and positive'
        )
    out = _coerce_finite_float(
        value,
        name=name,
        error_type=RefractionWeatheringReplacementStaticsError,
    )
    if not np.isfinite(out):
        raise RefractionWeatheringReplacementStaticsError(f'{name} must be finite')
    if out <= 0.0:
        raise RefractionWeatheringReplacementStaticsError(f'{name} must be positive')
    return out


def _validate_velocity_mode(
    value: object,
) -> Literal['solve_global', 'fixed_global', 'solve_cell']:
    if value == 'solve_global':
        return 'solve_global'
    if value == 'fixed_global':
        return 'fixed_global'
    if value == 'solve_cell':
        return 'solve_cell'
    raise RefractionWeatheringReplacementStaticsError(
        'bedrock_velocity_mode must be solve_global, fixed_global, or solve_cell'
    )


def _node_rows(
    result: RefractionWeatheringReplacementStaticsResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(result.node_id.shape[0])):
        rows.append(
            {
                'node_id': int(result.node_id[index]),
                'node_kind': str(result.node_kind[index]),
                'x_m': _csv_float(result.node_x_m[index]),
                'y_m': _csv_float(result.node_y_m[index]),
                'surface_elevation_m': _csv_float(
                    result.node_surface_elevation_m[index]
                ),
                'weathering_thickness_m': _csv_float(
                    result.node_weathering_thickness_m[index]
                ),
                'refractor_elevation_m': _csv_float(
                    result.node_refractor_elevation_m[index]
                ),
                'weathering_status': str(result.node_weathering_status[index]),
                'weathering_replacement_shift_ms': _csv_float(
                    result.node_weathering_replacement_shift_s[index] * 1000.0
                ),
                'static_status': str(result.node_static_status[index]),
                'pick_count': int(result.node_pick_count[index]),
                'used_pick_count': int(result.node_used_pick_count[index]),
                'residual_rms_ms': _csv_float(
                    result.node_residual_rms_s[index] * 1000.0
                ),
                'residual_mad_ms': _csv_float(
                    result.node_residual_mad_s[index] * 1000.0
                ),
            }
        )
    return rows


def _source_rows(
    result: RefractionWeatheringReplacementStaticsResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(result.source_endpoint_key.shape[0])):
        rows.append(
            {
                'source_endpoint_key': str(result.source_endpoint_key[index]),
                'source_id': int(result.source_id[index]),
                'source_node_id': int(result.source_node_id[index]),
                'source_x_m': _csv_float(result.source_x_m[index]),
                'source_y_m': _csv_float(result.source_y_m[index]),
                'source_surface_elevation_m': _csv_float(
                    result.source_surface_elevation_m[index]
                ),
                'source_weathering_thickness_m': _csv_float(
                    result.source_weathering_thickness_m[index]
                ),
                'source_weathering_replacement_shift_ms': _csv_float(
                    result.source_weathering_replacement_shift_s[index] * 1000.0
                ),
                'source_static_status': str(result.source_static_status[index]),
            }
        )
    return rows


def _receiver_rows(
    result: RefractionWeatheringReplacementStaticsResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(result.receiver_endpoint_key.shape[0])):
        rows.append(
            {
                'receiver_endpoint_key': str(result.receiver_endpoint_key[index]),
                'receiver_id': int(result.receiver_id[index]),
                'receiver_node_id': int(result.receiver_node_id[index]),
                'receiver_x_m': _csv_float(result.receiver_x_m[index]),
                'receiver_y_m': _csv_float(result.receiver_y_m[index]),
                'receiver_surface_elevation_m': _csv_float(
                    result.receiver_surface_elevation_m[index]
                ),
                'receiver_weathering_thickness_m': _csv_float(
                    result.receiver_weathering_thickness_m[index]
                ),
                'receiver_weathering_replacement_shift_ms': _csv_float(
                    result.receiver_weathering_replacement_shift_s[index] * 1000.0
                ),
                'receiver_static_status': str(result.receiver_static_status[index]),
            }
        )
    return rows


def _trace_preview_rows(
    result: RefractionWeatheringReplacementStaticsResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(result.sorted_trace_index.shape[0])):
        rows.append(
            {
                'sorted_trace_index': int(result.sorted_trace_index[index]),
                'valid_observation': bool(result.valid_observation_mask_sorted[index]),
                'used_observation': bool(result.used_observation_mask_sorted[index]),
                'trace_static_valid': bool(
                    result.trace_static_valid_mask_sorted[index]
                ),
                'source_node_id': int(result.source_node_id_sorted[index]),
                'receiver_node_id': int(result.receiver_node_id_sorted[index]),
                'source_weathering_thickness_m': _csv_float(
                    result.source_weathering_thickness_m_sorted[index]
                ),
                'receiver_weathering_thickness_m': _csv_float(
                    result.receiver_weathering_thickness_m_sorted[index]
                ),
                'source_weathering_replacement_shift_ms': _csv_float(
                    result.source_weathering_replacement_shift_s_sorted[index] * 1000.0
                ),
                'receiver_weathering_replacement_shift_ms': _csv_float(
                    result.receiver_weathering_replacement_shift_s_sorted[index]
                    * 1000.0
                ),
                'weathering_replacement_trace_shift_ms': _csv_float(
                    result.weathering_replacement_trace_shift_s_sorted[index] * 1000.0
                ),
                'source_static_status': str(result.source_static_status_sorted[index]),
                'receiver_static_status': str(
                    result.receiver_static_status_sorted[index]
                ),
                'trace_static_status': str(result.trace_static_status_sorted[index]),
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


__all__ = [
    'REFRACTION_WEATHERING_REPLACEMENT_NODES_CSV_NAME',
    'REFRACTION_WEATHERING_REPLACEMENT_QC_JSON_NAME',
    'REFRACTION_WEATHERING_REPLACEMENT_RECEIVERS_CSV_NAME',
    'REFRACTION_WEATHERING_REPLACEMENT_SOURCES_CSV_NAME',
    'REFRACTION_WEATHERING_REPLACEMENT_TRACE_PREVIEW_CSV_NAME',
    'RefractionWeatheringReplacementStaticsError',
    'RefractionWeatheringReplacementStaticsResult',
    'build_refraction_weathering_replacement_core_context',
    'build_refraction_weathering_replacement_statics',
    'compute_weathering_replacement_core_context_from_first_breaks',
    'compute_weathering_replacement_shift_s',
    'compute_weathering_replacement_shift_scalar_s',
    'compute_weathering_replacement_statics_from_first_breaks',
    'write_refraction_weathering_replacement_artifacts',
]
