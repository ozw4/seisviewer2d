"""Weathering-thickness model conversion for GLI refraction statics."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
from seis_statics.refraction.half_intercept import (
    RefractionHalfInterceptEndpointResult as CoreRefractionHalfInterceptEndpointResult,
)
from seis_statics.refraction.half_intercept import (
    RefractionHalfInterceptResult as CoreRefractionHalfInterceptResult,
)
from seis_statics.refraction.weathering import (
    RefractionWeatheringError as CoreRefractionWeatheringError,
)
from seis_statics.refraction.weathering import (
    RefractionWeatheringModel as CoreRefractionWeatheringModel,
)
from seis_statics.refraction.weathering import (
    build_refraction_weathering_model_from_half_intercept_result as core_build_refraction_weathering_model_from_half_intercept_result,
)
from seis_statics.refraction.weathering import (
    compute_weathering_thickness_from_half_intercept_time as core_compute_weathering_thickness,
)
from seis_statics.refraction.types import (
    RefractionEndpointTable as CoreRefractionEndpointTable,
)
from seis_statics.refraction.types import (
    RefractionStaticInputModel as CoreRefractionStaticInputModel,
)

from app.services.common.artifact_io import write_csv_atomic, write_json_atomic
from app.statics.refraction.application.cell_v2_metadata import (
    _CELL_ID_PROBE_V2_BASE_M_S,
    cell_v2_metadata_from_core_weathering,
)
from app.statics.refraction.application.core_options import (
    core_input_model_from_app,
    model_options_from_request,
)
from app.statics.refraction.application.core_options import (
    resolve_weathering_velocity_from_model_request as resolve_weathering_velocity_m_s,
)
from app.statics.refraction.application.half_intercept import (
    _HalfInterceptCoreContext,
    estimate_refraction_half_intercept_core_context_from_first_breaks,
)
from app.statics.refraction.application.trace_order import (
    sorted_positions_for_original_trace_ids,
)
from app.statics.refraction.contracts.apply import RefractionStaticApplyRequest
from app.statics.refraction.contracts.model import RefractionStaticModelRequest
from app.statics.refraction.contracts.result_types import (
    RefractionHalfInterceptTimeResult,
    RefractionStaticInputModel,
    RefractionWeatheringThicknessResult,
    ResolvedRefractionFirstLayer,
)
from app.statics.refraction.ports.runtime import RefractionRuntime

REFRACTION_WEATHERING_QC_JSON_NAME = 'refraction_weathering_thickness_qc.json'
REFRACTION_WEATHERING_NODES_CSV_NAME = 'refraction_weathering_nodes.csv'
REFRACTION_WEATHERING_SOURCES_CSV_NAME = 'refraction_weathering_sources.csv'
REFRACTION_WEATHERING_RECEIVERS_CSV_NAME = 'refraction_weathering_receivers.csv'
REFRACTION_WEATHERING_TRACE_PREVIEW_CSV_NAME = (
    'refraction_weathering_trace_preview.csv'
)

_STATUS_DTYPE = '<U32'
_FORMULA_TEXT = 'z = T * vb * vw / sqrt(vb^2 - vw^2)'
_CELL_THRESHOLD_QC_KEYS = (
    'min_observations_per_cell',
    'n_low_fold_cells',
    'n_observations_rejected_by_low_fold_cell',
    'low_fold_cell_rejection_reason',
    'low_fold_cell_id',
    'cell_observation_count',
)

_NODE_COLUMNS = (
    'node_id',
    'node_kind',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'half_intercept_time_ms',
    'weathering_thickness_m',
    'refractor_elevation_m',
    'solution_status',
    'weathering_status',
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
    'source_half_intercept_time_ms',
    'source_weathering_thickness_m',
    'source_refractor_elevation_m',
    'source_weathering_status',
)
_RECEIVER_COLUMNS = (
    'receiver_endpoint_key',
    'receiver_id',
    'receiver_node_id',
    'receiver_x_m',
    'receiver_y_m',
    'receiver_surface_elevation_m',
    'receiver_half_intercept_time_ms',
    'receiver_weathering_thickness_m',
    'receiver_refractor_elevation_m',
    'receiver_weathering_status',
)
_TRACE_PREVIEW_COLUMNS = (
    'sorted_trace_index',
    'valid_observation',
    'used_observation',
    'source_node_id',
    'receiver_node_id',
    'source_half_intercept_time_ms',
    'receiver_half_intercept_time_ms',
    'source_weathering_thickness_m',
    'receiver_weathering_thickness_m',
    'source_refractor_elevation_m',
    'receiver_refractor_elevation_m',
    'estimated_first_break_time_ms',
    'first_break_residual_ms',
)


class RefractionWeatheringThicknessError(ValueError):
    """Raised when weathering-thickness outputs cannot be built."""


@dataclass(frozen=True)
class _WeatheringCoreContext:
    core_weathering_model: CoreRefractionWeatheringModel
    app_weathering_result: RefractionWeatheringThicknessResult


def estimate_weathering_thickness_from_first_breaks(
    *,
    req: RefractionStaticApplyRequest,
    runtime: RefractionRuntime | None = None,
    state: object | None = None,
    job_dir: Path | None = None,
    input_model: RefractionStaticInputModel | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> RefractionWeatheringThicknessResult:
    """Estimate half-intercepts externally, then build the core weathering model."""
    return estimate_weathering_thickness_core_context_from_first_breaks(
        req=req,
        runtime=runtime,
        state=state,
        job_dir=job_dir,
        input_model=input_model,
        resolved_first_layer=resolved_first_layer,
    ).app_weathering_result


def estimate_weathering_thickness_core_context_from_first_breaks(
    *,
    req: RefractionStaticApplyRequest,
    runtime: RefractionRuntime | None = None,
    state: object | None = None,
    job_dir: Path | None = None,
    input_model: RefractionStaticInputModel | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> _WeatheringCoreContext:
    """Estimate half-intercepts and retain the external weathering model."""
    try:
        context = estimate_refraction_half_intercept_core_context_from_first_breaks(
            req=req,
            runtime=runtime,
            state=state,
            job_dir=job_dir,
            input_model=input_model,
            resolved_first_layer=resolved_first_layer,
        )
        return build_refraction_weathering_core_context(
            half_intercept_context=context,
            model=req.model,
            job_dir=job_dir,
            resolved_first_layer=resolved_first_layer,
        )
    except RefractionWeatheringThicknessError:
        raise
    except (CoreRefractionWeatheringError, ValueError) as exc:
        raise RefractionWeatheringThicknessError(str(exc)) from exc


def build_refraction_weathering_thickness_model(
    *,
    half_intercept_result: RefractionHalfInterceptTimeResult,
    model: RefractionStaticModelRequest,
    job_dir: Path | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    input_model: RefractionStaticInputModel | None = None,
    half_intercept_context: _HalfInterceptCoreContext | None = None,
) -> RefractionWeatheringThicknessResult:
    """Convert a half-intercept result to the existing app weathering DTO."""
    try:
        if half_intercept_context is not None:
            if half_intercept_context.app_result is not half_intercept_result:
                raise RefractionWeatheringThicknessError(
                    'half_intercept_context.app_result must match half_intercept_result'
                )
            _validate_weathering_velocity_matches_half_intercept_result(
                half_intercept_result=half_intercept_result,
                model=model,
                resolved_first_layer=resolved_first_layer,
            )
            return build_refraction_weathering_core_context(
                half_intercept_context=half_intercept_context,
                model=model,
                job_dir=job_dir,
                resolved_first_layer=resolved_first_layer,
            ).app_weathering_result

        _validate_weathering_velocity_matches_half_intercept_result(
            half_intercept_result=half_intercept_result,
            model=model,
            resolved_first_layer=resolved_first_layer,
        )
        core_input_model = (
            core_input_model_from_app(input_model)
            if input_model is not None
            else _core_input_model_from_app_half_intercept_result(
                half_intercept_result
            )
        )
        core_result = _core_half_intercept_result_from_app_result(
            half_intercept_result=half_intercept_result,
            model=model,
        )
        core_model = _model_options_for_weathering_core(
            model,
            resolved_first_layer=resolved_first_layer,
        )
        core_weathering = core_build_refraction_weathering_model_from_half_intercept_result(
            input_model=core_input_model,
            half_intercept_result=core_result,
            model=core_model,
        )
        cell_id_probe = _core_weathering_cell_id_probe(
            input_model=core_input_model,
            half_intercept_result=core_result,
            app_half_intercept_result=half_intercept_result,
            model=core_model,
        )
        core_weathering = _with_side_specific_endpoint_weathering(
            core_weathering=core_weathering,
            half_intercept_result=half_intercept_result,
            core_result=core_result,
            core_model=core_model,
        )
        result = _app_weathering_result_from_core(
            core_weathering=core_weathering,
            cell_id_probe=cell_id_probe,
            half_intercept_result=half_intercept_result,
            model=model,
            resolved_first_layer=resolved_first_layer,
        )
        if job_dir is not None:
            write_refraction_weathering_thickness_artifacts(Path(job_dir), result)
        return result
    except RefractionWeatheringThicknessError:
        raise
    except (CoreRefractionWeatheringError, ValueError) as exc:
        raise RefractionWeatheringThicknessError(str(exc)) from exc


def _build_refraction_weathering_thickness_from_core_context(
    *,
    half_intercept_context: _HalfInterceptCoreContext,
    model: RefractionStaticModelRequest,
    job_dir: Path | None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None,
) -> RefractionWeatheringThicknessResult:
    return build_refraction_weathering_core_context(
        half_intercept_context=half_intercept_context,
        model=model,
        job_dir=job_dir,
        resolved_first_layer=resolved_first_layer,
    ).app_weathering_result


def build_refraction_weathering_core_context(
    *,
    half_intercept_context: _HalfInterceptCoreContext,
    model: RefractionStaticModelRequest,
    job_dir: Path | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> _WeatheringCoreContext:
    """Build app weathering output and retain the external model for downstream stages."""
    _validate_weathering_velocity_matches_half_intercept_result(
        half_intercept_result=half_intercept_context.app_result,
        model=model,
        resolved_first_layer=resolved_first_layer,
    )
    core_model = _model_options_for_weathering_core(
        model,
        resolved_first_layer=resolved_first_layer,
    )
    core_weathering = core_build_refraction_weathering_model_from_half_intercept_result(
        input_model=half_intercept_context.core_input_model,
        half_intercept_result=half_intercept_context.core_result,
        model=core_model,
    )
    cell_id_probe = _core_weathering_cell_id_probe(
        input_model=half_intercept_context.core_input_model,
        half_intercept_result=half_intercept_context.core_result,
        app_half_intercept_result=half_intercept_context.app_result,
        model=core_model,
    )
    core_weathering = _with_side_specific_endpoint_weathering(
        core_weathering=core_weathering,
        half_intercept_result=half_intercept_context.app_result,
        core_result=half_intercept_context.core_result,
        core_model=core_model,
    )
    result = _app_weathering_result_from_core(
        core_weathering=core_weathering,
        cell_id_probe=cell_id_probe,
        half_intercept_result=half_intercept_context.app_result,
        model=model,
        resolved_first_layer=resolved_first_layer,
    )
    if job_dir is not None:
        write_refraction_weathering_thickness_artifacts(Path(job_dir), result)
    return _WeatheringCoreContext(
        core_weathering_model=core_weathering,
        app_weathering_result=result,
    )


def compute_weathering_thickness_from_half_intercept_time(
    *,
    half_intercept_time_s: np.ndarray,
    weathering_velocity_m_s: float,
    bedrock_velocity_m_s: float | np.ndarray,
) -> np.ndarray:
    """Convert half-intercept time to weathering thickness with the core primitive."""
    try:
        half = np.asarray(half_intercept_time_s, dtype=np.float64)
        bedrock = np.asarray(bedrock_velocity_m_s, dtype=np.float64)
        half, bedrock = np.broadcast_arrays(half, bedrock)
        out = np.full(half.shape, np.nan, dtype=np.float64)
        valid = np.isfinite(half) & np.isfinite(bedrock)
        if np.any(valid):
            out[valid] = core_compute_weathering_thickness(
                half_intercept_time_s=half[valid],
                v1_m_s=weathering_velocity_m_s,
                v2_m_s=bedrock[valid],
            )
        return np.ascontiguousarray(out, dtype=np.float64)
    except ValueError as exc:
        if str(exc) == 'v2_m_s must be greater than v1_m_s':
            raise RefractionWeatheringThicknessError(
                'bedrock_velocity_m_s must be greater than weathering_velocity_m_s'
            ) from exc
        raise RefractionWeatheringThicknessError(
            str(exc)
            .replace('t1_s', 'half_intercept_time_s')
            .replace('v1_m_s', 'weathering_velocity_m_s')
            .replace('v2_m_s', 'bedrock_velocity_m_s')
        ) from exc


def compute_weathering_thickness_scalar(
    *,
    half_intercept_time_s: float,
    weathering_velocity_m_s: float,
    bedrock_velocity_m_s: float,
) -> float:
    """Scalar wrapper around ``compute_weathering_thickness_from_half_intercept_time``."""
    value = compute_weathering_thickness_from_half_intercept_time(
        half_intercept_time_s=np.asarray([half_intercept_time_s], dtype=np.float64),
        weathering_velocity_m_s=weathering_velocity_m_s,
        bedrock_velocity_m_s=bedrock_velocity_m_s,
    )
    return float(value[0])


def _model_options_for_weathering_core(
    model: RefractionStaticModelRequest,
    *,
    resolved_first_layer: ResolvedRefractionFirstLayer | None,
) -> object:
    options = model_options_from_request(model)
    if resolved_first_layer is None or getattr(options, 'first_layer', None) is None:
        return options
    if options.first_layer.mode != 'estimate_direct_arrival':
        return options
    return replace(
        options,
        first_layer=replace(
            options.first_layer,
            mode='constant',
            weathering_velocity_m_s=resolved_first_layer.weathering_velocity_m_s,
        ),
    )


def _validate_weathering_velocity_matches_half_intercept_result(
    *,
    half_intercept_result: RefractionHalfInterceptTimeResult,
    model: RefractionStaticModelRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None,
) -> float:
    requested_velocity = float(
        resolve_weathering_velocity_m_s(
            model=model,
            resolved_first_layer=resolved_first_layer,
            name='model.weathering_velocity_m_s',
        )
    )
    result_velocity = float(half_intercept_result.weathering_velocity_m_s)
    if not np.isfinite(result_velocity) or result_velocity <= 0.0:
        raise ValueError(
            'half_intercept_result.weathering_velocity_m_s must be positive finite'
        )
    if not np.isclose(
        requested_velocity,
        result_velocity,
        rtol=1.0e-6,
        atol=1.0e-6,
    ):
        raise ValueError(
            'model.weathering_velocity_m_s does not match '
            'half_intercept_result.weathering_velocity_m_s'
        )
    return requested_velocity


def _core_weathering_cell_id_probe(
    *,
    input_model: CoreRefractionStaticInputModel,
    half_intercept_result: CoreRefractionHalfInterceptResult,
    app_half_intercept_result: RefractionHalfInterceptTimeResult,
    model: object,
) -> CoreRefractionWeatheringModel | None:
    if half_intercept_result.bedrock_velocity_mode != 'solve_cell':
        return None
    cell_id = np.asarray(half_intercept_result.cell_id, dtype=np.int64)
    if cell_id.size == 0:
        return None
    probe_v2 = np.ascontiguousarray(
        _CELL_ID_PROBE_V2_BASE_M_S + cell_id.astype(np.float64),
        dtype=np.float64,
    )
    probe_result = replace(
        half_intercept_result,
        cell_bedrock_velocity_m_s=probe_v2,
        cell_bedrock_slowness_s_per_m=np.divide(
            1.0,
            probe_v2,
            out=np.full(probe_v2.shape, np.nan, dtype=np.float64),
            where=probe_v2 > 0.0,
        ),
        cell_v2_m_s=probe_v2,
    )
    probe_weathering = core_build_refraction_weathering_model_from_half_intercept_result(
        input_model=input_model,
        half_intercept_result=probe_result,
        model=model,
    )
    return _with_side_specific_endpoint_weathering(
        core_weathering=probe_weathering,
        half_intercept_result=app_half_intercept_result,
        core_result=probe_result,
        core_model=model,
    )


def _core_input_model_from_app_half_intercept_result(
    result: RefractionHalfInterceptTimeResult,
    *,
    endpoint_geometry_side: str | None = None,
) -> CoreRefractionStaticInputModel:
    n_traces = int(np.asarray(result.sorted_trace_index).shape[0])
    source_x_sorted = _node_values_sorted(
        node_id_sorted=result.source_node_id_sorted,
        node_id=result.node_id,
        values=result.node_x_m,
        fill_value=np.nan,
    )
    source_y_sorted = _node_values_sorted(
        node_id_sorted=result.source_node_id_sorted,
        node_id=result.node_id,
        values=result.node_y_m,
        fill_value=np.nan,
    )
    receiver_x_sorted = _node_values_sorted(
        node_id_sorted=result.receiver_node_id_sorted,
        node_id=result.node_id,
        values=result.node_x_m,
        fill_value=np.nan,
    )
    receiver_y_sorted = _node_values_sorted(
        node_id_sorted=result.receiver_node_id_sorted,
        node_id=result.node_id,
        values=result.node_y_m,
        fill_value=np.nan,
    )
    pick_time_s_sorted = np.full(n_traces, np.nan, dtype=np.float64)
    row_index = np.asarray(result.row_trace_index_sorted, dtype=np.int64).reshape(-1)
    row_sorted_position = sorted_positions_for_original_trace_ids(
        sorted_to_original=result.sorted_trace_index,
        original_trace_id=row_index,
    )
    observed = np.asarray(result.observed_pick_time_s, dtype=np.float64).reshape(-1)
    if row_index.shape == observed.shape:
        pick_time_s_sorted[row_sorted_position] = observed
    elif observed.shape == (n_traces,):
        pick_time_s_sorted[:] = observed
    distance_m_sorted = np.full(n_traces, np.nan, dtype=np.float64)
    row_distance = np.asarray(result.row_distance_m, dtype=np.float64).reshape(-1)
    if row_index.shape == row_distance.shape:
        distance_m_sorted[row_sorted_position] = row_distance
    elif row_distance.shape == (n_traces,):
        distance_m_sorted[:] = row_distance
    endpoint_table = _endpoint_table_from_app_half_intercept_result(
        result,
        endpoint_geometry_side=endpoint_geometry_side,
    )
    return CoreRefractionStaticInputModel(
        file_id=str(result.qc.get('file_id', '')),
        n_traces=n_traces,
        sorted_trace_index=_i64(result.sorted_trace_index),
        pick_time_s_sorted=np.ascontiguousarray(pick_time_s_sorted, dtype=np.float64),
        valid_pick_mask_sorted=np.isfinite(pick_time_s_sorted),
        valid_observation_mask_sorted=np.ascontiguousarray(
            result.valid_observation_mask_sorted,
            dtype=bool,
        ),
        source_id_sorted=_i64(result.source_node_id_sorted),
        receiver_id_sorted=_i64(result.receiver_node_id_sorted),
        source_x_m_sorted=source_x_sorted,
        source_y_m_sorted=source_y_sorted,
        receiver_x_m_sorted=receiver_x_sorted,
        receiver_y_m_sorted=receiver_y_sorted,
        source_elevation_m_sorted=_f64(result.source_elevation_m_sorted),
        receiver_elevation_m_sorted=_f64(result.receiver_elevation_m_sorted),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=np.nan_to_num(
            distance_m_sorted,
            nan=0.0,
        ),
        offset_m_sorted=None,
        distance_m_sorted=np.ascontiguousarray(distance_m_sorted, dtype=np.float64),
        source_endpoint_key_sorted=np.ascontiguousarray(
            result.source_endpoint_key_sorted,
        ),
        receiver_endpoint_key_sorted=np.ascontiguousarray(
            result.receiver_endpoint_key_sorted,
        ),
        source_node_id_sorted=_i64(result.source_node_id_sorted),
        receiver_node_id_sorted=_i64(result.receiver_node_id_sorted),
        node_x_m=_f64(result.node_x_m),
        node_y_m=_f64(result.node_y_m),
        node_elevation_m=_f64(result.node_elevation_m),
        node_kind=np.ascontiguousarray(result.node_kind),
        rejection_reason_sorted=np.full(n_traces, '', dtype=_STATUS_DTYPE),
        qc=dict(result.qc),
        endpoint_table=endpoint_table,
        metadata={},
        layer_observation_masks=None,
        source_endpoint_id_sorted=None,
        receiver_endpoint_id_sorted=None,
    )


def _with_side_specific_endpoint_weathering(
    *,
    core_weathering: CoreRefractionWeatheringModel,
    half_intercept_result: RefractionHalfInterceptTimeResult,
    core_result: CoreRefractionHalfInterceptResult,
    core_model: object,
) -> CoreRefractionWeatheringModel:
    if not _has_shared_endpoint_geometry_conflict(half_intercept_result):
        return core_weathering
    source_core = core_build_refraction_weathering_model_from_half_intercept_result(
        input_model=_core_input_model_from_app_half_intercept_result(
            half_intercept_result,
            endpoint_geometry_side='source',
        ),
        half_intercept_result=core_result,
        model=core_model,
    )
    receiver_core = core_build_refraction_weathering_model_from_half_intercept_result(
        input_model=_core_input_model_from_app_half_intercept_result(
            half_intercept_result,
            endpoint_geometry_side='receiver',
        ),
        half_intercept_result=core_result,
        model=core_model,
    )
    trace_thickness = np.ascontiguousarray(
        source_core.source_weathering_thickness_m_sorted
        + receiver_core.receiver_weathering_thickness_m_sorted,
        dtype=np.float64,
    )
    trace_status = _trace_weathering_status_from_endpoint_status(
        source_status=source_core.source_weathering_status_sorted,
        receiver_status=receiver_core.receiver_weathering_status_sorted,
        source_thickness=source_core.source_weathering_thickness_m_sorted,
        receiver_thickness=receiver_core.receiver_weathering_thickness_m_sorted,
    )
    qc = dict(getattr(core_weathering, 'qc', {}))
    qc['source_weathering_status_counts'] = _status_counts(
        np.asarray(source_core.source_endpoint.weathering_status)
    )
    qc['receiver_weathering_status_counts'] = _status_counts(
        np.asarray(receiver_core.receiver_endpoint.weathering_status)
    )
    qc['trace_weathering_status_counts'] = _status_counts(trace_status)
    return replace(
        core_weathering,
        source_endpoint=source_core.source_endpoint,
        receiver_endpoint=receiver_core.receiver_endpoint,
        source_weathering_thickness_m_sorted=(
            source_core.source_weathering_thickness_m_sorted
        ),
        receiver_weathering_thickness_m_sorted=(
            receiver_core.receiver_weathering_thickness_m_sorted
        ),
        source_refractor_elevation_m_sorted=(
            source_core.source_refractor_elevation_m_sorted
        ),
        receiver_refractor_elevation_m_sorted=(
            receiver_core.receiver_refractor_elevation_m_sorted
        ),
        source_weathering_status_sorted=source_core.source_weathering_status_sorted,
        receiver_weathering_status_sorted=(
            receiver_core.receiver_weathering_status_sorted
        ),
        trace_weathering_thickness_m_sorted=trace_thickness,
        trace_weathering_status_sorted=trace_status,
        qc=qc,
    )


def _trace_weathering_status_from_endpoint_status(
    *,
    source_status: np.ndarray,
    receiver_status: np.ndarray,
    source_thickness: np.ndarray,
    receiver_thickness: np.ndarray,
) -> np.ndarray:
    source = _status(source_status)
    receiver = _status(receiver_status)
    source_values = _f64(source_thickness)
    receiver_values = _f64(receiver_thickness)
    if not (
        source.shape
        == receiver.shape
        == source_values.shape
        == receiver_values.shape
    ):
        raise RefractionWeatheringThicknessError(
            'source/receiver trace weathering shape mismatch'
        )
    status = np.full(source.shape, 'ok', dtype=_STATUS_DTYPE)
    for index, (src_status, rec_status) in enumerate(
        zip(source.tolist(), receiver.tolist(), strict=True)
    ):
        src = str(src_status)
        rec = str(rec_status)
        if src == rec:
            status[index] = src
        elif src == 'ok':
            status[index] = rec
        elif rec == 'ok':
            status[index] = src
        else:
            status[index] = 'mixed'
    invalid_value = ~(np.isfinite(source_values) & np.isfinite(receiver_values))
    status[invalid_value & (status == 'ok')] = 'invalid_weathering_thickness'
    return np.ascontiguousarray(status, dtype=_STATUS_DTYPE)


def _has_shared_endpoint_geometry_conflict(
    result: RefractionHalfInterceptTimeResult,
) -> bool:
    source_by_node = _endpoint_geometry_by_node(result, 'source')
    receiver_by_node = _endpoint_geometry_by_node(result, 'receiver')
    for node, source_geometry in source_by_node.items():
        receiver_geometry = receiver_by_node.get(node)
        if receiver_geometry is None:
            continue
        if not all(
            _same_geometry_value(left, right)
            for left, right in zip(source_geometry, receiver_geometry, strict=True)
        ):
            return True
    return False


def _endpoint_geometry_by_node(
    result: RefractionHalfInterceptTimeResult,
    side: str,
) -> dict[int, tuple[float, float, float]]:
    node_id = _i64(getattr(result, f'{side}_node_id'))
    x_m = _f64(getattr(result, f'{side}_x_m'))
    y_m = _f64(getattr(result, f'{side}_y_m'))
    elevation_m = _f64(getattr(result, f'{side}_elevation_m'))
    if not (node_id.shape == x_m.shape == y_m.shape == elevation_m.shape):
        raise RefractionWeatheringThicknessError(
            f'{side} endpoint geometry shape mismatch'
        )
    return {
        int(node): (float(x), float(y), float(elevation))
        for node, x, y, elevation in zip(
            node_id.tolist(),
            x_m.tolist(),
            y_m.tolist(),
            elevation_m.tolist(),
            strict=True,
        )
    }


def _same_geometry_value(left: float, right: float) -> bool:
    if np.isnan(left) and np.isnan(right):
        return True
    return bool(left == right)


def _endpoint_table_from_app_half_intercept_result(
    result: RefractionHalfInterceptTimeResult,
    *,
    endpoint_geometry_side: str | None = None,
) -> CoreRefractionEndpointTable:
    node_id = _i64(result.node_id)
    x_m = _f64(result.node_x_m).copy()
    y_m = _f64(result.node_y_m).copy()
    elevation_m = _f64(result.node_elevation_m).copy()
    pick_count = _i64(result.node_pick_count).copy()
    if endpoint_geometry_side is None:
        return CoreRefractionEndpointTable(
            node_id=node_id,
            endpoint_id=node_id.copy(),
            x_m=np.ascontiguousarray(x_m, dtype=np.float64),
            y_m=np.ascontiguousarray(y_m, dtype=np.float64),
            elevation_m=np.ascontiguousarray(elevation_m, dtype=np.float64),
            kind=np.ascontiguousarray(result.node_kind),
            pick_count=np.ascontiguousarray(pick_count, dtype=np.int64),
        )
    if endpoint_geometry_side not in {'source', 'receiver'}:
        raise RefractionWeatheringThicknessError(
            'endpoint_geometry_side must be source, receiver, or None'
        )
    node_pos = {int(node): index for index, node in enumerate(node_id.tolist())}
    endpoint_node_id = _i64(getattr(result, f'{endpoint_geometry_side}_node_id'))
    endpoint_x = _f64(getattr(result, f'{endpoint_geometry_side}_x_m'))
    endpoint_y = _f64(getattr(result, f'{endpoint_geometry_side}_y_m'))
    endpoint_elevation = _f64(getattr(result, f'{endpoint_geometry_side}_elevation_m'))
    endpoint_pick_count = _i64(getattr(result, f'{endpoint_geometry_side}_pick_count'))
    if not (
        endpoint_node_id.shape
        == endpoint_x.shape
        == endpoint_y.shape
        == endpoint_elevation.shape
        == endpoint_pick_count.shape
    ):
        raise RefractionWeatheringThicknessError(
            f'{endpoint_geometry_side} endpoint geometry shape mismatch'
        )
    for index, raw_node in enumerate(endpoint_node_id.tolist()):
        position = node_pos.get(int(raw_node))
        if position is None:
            continue
        x_m[position] = float(endpoint_x[index])
        y_m[position] = float(endpoint_y[index])
        elevation_m[position] = float(endpoint_elevation[index])
        pick_count[position] = max(
            int(pick_count[position]),
            int(endpoint_pick_count[index]),
        )
    return CoreRefractionEndpointTable(
        node_id=node_id,
        endpoint_id=node_id.copy(),
        x_m=np.ascontiguousarray(x_m, dtype=np.float64),
        y_m=np.ascontiguousarray(y_m, dtype=np.float64),
        elevation_m=np.ascontiguousarray(elevation_m, dtype=np.float64),
        kind=np.ascontiguousarray(result.node_kind),
        pick_count=np.ascontiguousarray(pick_count, dtype=np.int64),
    )


def _node_values_sorted(
    *,
    node_id_sorted: np.ndarray,
    node_id: np.ndarray,
    values: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    lookup = {
        int(node): float(value)
        for node, value in zip(
            np.asarray(node_id, dtype=np.int64).tolist(),
            np.asarray(values, dtype=np.float64).tolist(),
            strict=True,
        )
    }
    return np.ascontiguousarray(
        np.asarray(
            [
                lookup.get(int(node), fill_value)
                for node in np.asarray(node_id_sorted, dtype=np.int64).tolist()
            ],
            dtype=np.float64,
        )
    )


def _app_weathering_result_from_core(
    *,
    core_weathering: CoreRefractionWeatheringModel,
    cell_id_probe: CoreRefractionWeatheringModel | None,
    half_intercept_result: RefractionHalfInterceptTimeResult,
    model: RefractionStaticModelRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None,
) -> RefractionWeatheringThicknessResult:
    source = core_weathering.source_endpoint
    receiver = core_weathering.receiver_endpoint
    node_thickness = _f64(core_weathering.node_weathering_thickness_m)
    node_refractor = _f64(core_weathering.node_refractor_elevation_m)
    node_status = _weathering_status(core_weathering.node_weathering_status)
    source_thickness = _f64(source.weathering_thickness_m)
    receiver_thickness = _f64(receiver.weathering_thickness_m)
    source_refractor = _f64(source.refractor_elevation_m)
    receiver_refractor = _f64(receiver.refractor_elevation_m)
    source_status = _weathering_status(source.weathering_status)
    receiver_status = _weathering_status(receiver.weathering_status)
    source_thickness_sorted = _f64(
        core_weathering.source_weathering_thickness_m_sorted
    )
    receiver_thickness_sorted = _f64(
        core_weathering.receiver_weathering_thickness_m_sorted
    )
    source_refractor_sorted = _f64(core_weathering.source_refractor_elevation_m_sorted)
    receiver_refractor_sorted = _f64(
        core_weathering.receiver_refractor_elevation_m_sorted
    )
    source_status_sorted = _weathering_status(
        core_weathering.source_weathering_status_sorted
    )
    receiver_status_sorted = _weathering_status(
        core_weathering.receiver_weathering_status_sorted
    )
    cell_v2 = cell_v2_metadata_from_core_weathering(
        core_weathering=core_weathering,
        model=model,
        cell_id_probe=cell_id_probe,
    )
    qc = _app_weathering_qc_from_core(
        core_weathering=core_weathering,
        half_intercept_result=half_intercept_result,
        model=model,
    )
    return RefractionWeatheringThicknessResult(
        bedrock_velocity_mode=core_weathering.bedrock_velocity_mode,
        bedrock_slowness_s_per_m=float(
            half_intercept_result.bedrock_slowness_s_per_m
        ),
        bedrock_velocity_m_s=float(half_intercept_result.bedrock_velocity_m_s),
        weathering_velocity_m_s=float(core_weathering.weathering_velocity_m_s),
        node_id=_i64(core_weathering.node_id),
        node_x_m=_f64(core_weathering.node_x_m),
        node_y_m=_f64(core_weathering.node_y_m),
        node_surface_elevation_m=_f64(core_weathering.node_surface_elevation_m),
        node_kind=np.ascontiguousarray(half_intercept_result.node_kind),
        node_half_intercept_time_s=_f64(core_weathering.node_half_intercept_time_s),
        node_half_intercept_time_ms=_f64(
            core_weathering.node_half_intercept_time_s
        )
        * 1000.0,
        node_weathering_thickness_m=node_thickness,
        node_refractor_elevation_m=node_refractor,
        node_solution_status=_status(core_weathering.node_solution_status),
        node_weathering_status=node_status,
        node_pick_count=_i64(core_weathering.node_pick_count),
        node_used_pick_count=_i64(core_weathering.node_used_observation_count),
        node_rejected_pick_count=_i64(core_weathering.node_rejected_observation_count),
        node_residual_rms_s=_f64(half_intercept_result.node_residual_rms_s),
        node_residual_mad_s=_f64(half_intercept_result.node_residual_mad_s),
        source_endpoint_key=np.ascontiguousarray(source.endpoint_key, dtype=object),
        source_id=_i64(half_intercept_result.source_id),
        source_node_id=_i64(source.node_id),
        source_x_m=_f64(source.x_m),
        source_y_m=_f64(source.y_m),
        source_surface_elevation_m=_f64(source.surface_elevation_m),
        source_half_intercept_time_s=_f64(source.half_intercept_time_s),
        source_weathering_thickness_m=source_thickness,
        source_refractor_elevation_m=source_refractor,
        source_weathering_status=source_status,
        receiver_endpoint_key=np.ascontiguousarray(receiver.endpoint_key, dtype=object),
        receiver_id=_i64(half_intercept_result.receiver_id),
        receiver_node_id=_i64(receiver.node_id),
        receiver_x_m=_f64(receiver.x_m),
        receiver_y_m=_f64(receiver.y_m),
        receiver_surface_elevation_m=_f64(receiver.surface_elevation_m),
        receiver_half_intercept_time_s=_f64(receiver.half_intercept_time_s),
        receiver_weathering_thickness_m=receiver_thickness,
        receiver_refractor_elevation_m=receiver_refractor,
        receiver_weathering_status=receiver_status,
        sorted_trace_index=_i64(core_weathering.trace_index_sorted),
        valid_observation_mask_sorted=np.ascontiguousarray(
            half_intercept_result.valid_observation_mask_sorted,
            dtype=bool,
        ),
        used_observation_mask_sorted=np.ascontiguousarray(
            half_intercept_result.used_observation_mask_sorted,
            dtype=bool,
        ),
        source_endpoint_key_sorted=np.ascontiguousarray(
            core_weathering.source_endpoint_key_sorted,
        ),
        receiver_endpoint_key_sorted=np.ascontiguousarray(
            core_weathering.receiver_endpoint_key_sorted,
        ),
        source_node_id_sorted=_i64(core_weathering.source_node_id_sorted),
        receiver_node_id_sorted=_i64(core_weathering.receiver_node_id_sorted),
        source_half_intercept_time_s_sorted=_f64(
            half_intercept_result.source_half_intercept_time_s_sorted
        ),
        receiver_half_intercept_time_s_sorted=_f64(
            half_intercept_result.receiver_half_intercept_time_s_sorted
        ),
        source_weathering_thickness_m_sorted=source_thickness_sorted,
        receiver_weathering_thickness_m_sorted=receiver_thickness_sorted,
        source_refractor_elevation_m_sorted=source_refractor_sorted,
        receiver_refractor_elevation_m_sorted=receiver_refractor_sorted,
        source_weathering_status_sorted=source_status_sorted,
        receiver_weathering_status_sorted=receiver_status_sorted,
        estimated_first_break_time_s_sorted=_f64(
            half_intercept_result.estimated_first_break_time_s_sorted
        ),
        first_break_residual_s_sorted=_f64(
            half_intercept_result.first_break_residual_s_sorted
        ),
        row_trace_index_sorted=_i64(half_intercept_result.row_trace_index_sorted),
        row_source_node_id=_i64(half_intercept_result.row_source_node_id),
        row_receiver_node_id=_i64(half_intercept_result.row_receiver_node_id),
        row_distance_m=_f64(half_intercept_result.row_distance_m),
        observed_pick_time_s=_f64(half_intercept_result.observed_pick_time_s),
        modeled_pick_time_s=_f64(half_intercept_result.modeled_pick_time_s),
        residual_time_s=_f64(half_intercept_result.residual_time_s),
        used_row_mask=np.ascontiguousarray(half_intercept_result.used_row_mask, dtype=bool),
        rejected_by_robust_mask=np.ascontiguousarray(
            half_intercept_result.rejected_by_robust_mask,
            dtype=bool,
        ),
        qc=qc,
        active_cell_id=_optional_i64(half_intercept_result.active_cell_id),
        inactive_cell_id=_optional_i64(half_intercept_result.inactive_cell_id),
        cell_bedrock_slowness_s_per_m=_optional_f64(
            half_intercept_result.cell_bedrock_slowness_s_per_m
        ),
        cell_bedrock_velocity_m_s=_optional_f64(
            half_intercept_result.cell_bedrock_velocity_m_s
        ),
        cell_velocity_status=_optional_status(half_intercept_result.cell_velocity_status),
        row_midpoint_cell_id=_optional_i64(half_intercept_result.row_midpoint_cell_id),
        node_v2_cell_id=cell_v2.node_cell_id,
        node_v2_m_s=_optional_cell_array(core_weathering.node_v2_m_s, model),
        node_v2_status=_optional_cell_status(core_weathering.node_local_v2_status, model),
        source_v2_cell_id=cell_v2.source_cell_id,
        source_v2_m_s=_optional_cell_array(source.v2_m_s, model),
        source_v2_status=_optional_cell_status(source.local_v2_status, model),
        receiver_v2_cell_id=cell_v2.receiver_cell_id,
        receiver_v2_m_s=_optional_cell_array(receiver.v2_m_s, model),
        receiver_v2_status=_optional_cell_status(receiver.local_v2_status, model),
        source_v2_cell_id_sorted=cell_v2.source_cell_id_sorted,
        source_v2_m_s_sorted=cell_v2.source_v2_m_s_sorted,
        source_v2_status_sorted=cell_v2.source_v2_status_sorted,
        receiver_v2_cell_id_sorted=cell_v2.receiver_cell_id_sorted,
        receiver_v2_m_s_sorted=cell_v2.receiver_v2_m_s_sorted,
        receiver_v2_status_sorted=cell_v2.receiver_v2_status_sorted,
    )


def _app_weathering_qc_from_core(
    *,
    core_weathering: CoreRefractionWeatheringModel,
    half_intercept_result: RefractionHalfInterceptTimeResult,
    model: RefractionStaticModelRequest,
) -> dict[str, Any]:
    qc = _core_qc(getattr(core_weathering, 'qc', {}))
    source = core_weathering.source_endpoint
    receiver = core_weathering.receiver_endpoint
    node_status = _weathering_status(core_weathering.node_weathering_status)
    source_status = _weathering_status(source.weathering_status)
    receiver_status = _weathering_status(receiver.weathering_status)
    node_counts = _qc_counts(qc.get('node_weathering_status_counts')) or _status_counts(
        node_status
    )
    source_counts = _qc_counts(
        qc.get('source_weathering_status_counts')
    ) or _status_counts(source_status)
    receiver_counts = _qc_counts(
        qc.get('receiver_weathering_status_counts')
    ) or _status_counts(receiver_status)
    ok_nodes = node_status == 'ok'
    source_ok = source_status == 'ok'
    receiver_ok = receiver_status == 'ok'
    node_thickness = _f64(core_weathering.node_weathering_thickness_m)
    node_refractor = _f64(core_weathering.node_refractor_elevation_m)
    source_thickness = _f64(source.weathering_thickness_m)
    receiver_thickness = _f64(receiver.weathering_thickness_m)
    finite_ok_thickness = node_thickness[ok_nodes & np.isfinite(node_thickness)]
    finite_ok_refractor = node_refractor[ok_nodes & np.isfinite(node_refractor)]
    max_thickness = model.max_weathering_thickness_m
    qc.setdefault('method', 'gli_variable_thickness')
    qc.setdefault('bedrock_velocity_mode', str(core_weathering.bedrock_velocity_mode))
    qc.setdefault(
        'bedrock_velocity_m_s',
        _json_float_or_none(half_intercept_result.bedrock_velocity_m_s),
    )
    qc.setdefault(
        'bedrock_slowness_s_per_m',
        _json_float_or_none(half_intercept_result.bedrock_slowness_s_per_m),
    )
    qc.setdefault(
        'weathering_velocity_m_s',
        _json_float_or_none(core_weathering.weathering_velocity_m_s),
    )
    qc.setdefault('n_traces', int(core_weathering.n_traces))
    qc.setdefault(
        'n_valid_observations',
        int(np.count_nonzero(half_intercept_result.valid_observation_mask_sorted)),
    )
    qc.setdefault(
        'n_used_observations',
        int(np.count_nonzero(half_intercept_result.used_observation_mask_sorted)),
    )
    qc.setdefault('n_nodes', int(core_weathering.node_id.shape[0]))
    qc.setdefault(
        'n_source_endpoints',
        int(core_weathering.source_endpoint.endpoint_key.shape[0]),
    )
    qc.setdefault(
        'n_receiver_endpoints',
        int(core_weathering.receiver_endpoint.endpoint_key.shape[0]),
    )
    qc.setdefault('weathering_status_counts', node_counts)
    qc.setdefault('source_weathering_status_counts', source_counts)
    qc.setdefault('receiver_weathering_status_counts', receiver_counts)
    qc.setdefault('n_inactive_nodes', int(np.count_nonzero(node_status == 'inactive')))
    qc.setdefault('n_active_nodes', int(np.count_nonzero(node_status != 'inactive')))
    qc.setdefault('weathering_thickness_min_m', _json_stat(finite_ok_thickness, 'min'))
    qc.setdefault('weathering_thickness_max_m', _json_stat(finite_ok_thickness, 'max'))
    qc.setdefault(
        'weathering_thickness_median_m',
        _json_stat(finite_ok_thickness, 'median'),
    )
    qc.setdefault('weathering_thickness_p95_m', _json_stat(finite_ok_thickness, 'p95'))
    qc.setdefault(
        'source_weathering_thickness_median_m',
        _json_stat(
            source_thickness[source_ok & np.isfinite(source_thickness)],
            'median',
        ),
    )
    qc.setdefault(
        'receiver_weathering_thickness_median_m',
        _json_stat(
            receiver_thickness[receiver_ok & np.isfinite(receiver_thickness)],
            'median',
        ),
    )
    qc.setdefault('refractor_elevation_min_m', _json_stat(finite_ok_refractor, 'min'))
    qc.setdefault('refractor_elevation_max_m', _json_stat(finite_ok_refractor, 'max'))
    qc.setdefault(
        'refractor_elevation_median_m',
        _json_stat(finite_ok_refractor, 'median'),
    )
    qc.setdefault(
        'max_weathering_thickness_m',
        None if max_thickness is None else float(max_thickness),
    )
    qc.setdefault('inactive_node_count', int(np.count_nonzero(node_status == 'inactive')))
    qc.setdefault('low_fold_node_count', int(np.count_nonzero(node_status == 'low_fold')))
    qc.setdefault(
        'exceeds_max_thickness_count',
        int(np.count_nonzero(node_status == 'exceeds_max_thickness')),
    )
    qc.setdefault(
        'clipped_half_intercept_node_count',
        int(
            np.count_nonzero(
                (node_status == 'clipped_half_intercept_lower')
                | (node_status == 'clipped_half_intercept_upper')
                | (node_status == 'clipped_lower')
                | (node_status == 'clipped_upper')
            )
        ),
    )
    qc.setdefault(
        'invalid_half_intercept_node_count',
        int(np.count_nonzero(node_status == 'invalid_half_intercept')),
    )
    qc.setdefault(
        'negative_thickness_node_count',
        int(np.count_nonzero(node_status == 'negative_thickness')),
    )
    qc.setdefault(
        'zero_thickness_node_count',
        int(np.count_nonzero(node_status == 'zero_thickness')),
    )
    qc.setdefault(
        'invalid_surface_elevation_count',
        int(np.count_nonzero(node_status == 'invalid_surface_elevation')),
    )
    qc.setdefault(
        'invalid_refractor_elevation_count',
        int(np.count_nonzero(node_status == 'invalid_refractor_elevation')),
    )
    qc.setdefault('thickness_formula', _FORMULA_TEXT)
    _copy_cell_threshold_qc(qc, getattr(half_intercept_result, 'qc', {}))
    _assert_json_safe(qc)
    return qc


def _core_qc(values: object) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise RefractionWeatheringThicknessError('core qc must be a dict')
    return dict(values)


def _qc_counts(values: object) -> dict[str, int]:
    if not isinstance(values, dict):
        return {}
    return {str(key): int(value) for key, value in values.items()}


def _core_half_intercept_result_from_app_result(
    *,
    half_intercept_result: RefractionHalfInterceptTimeResult,
    model: RefractionStaticModelRequest,
) -> CoreRefractionHalfInterceptResult:
    _validate_app_half_intercept_for_core_conversion(half_intercept_result)
    n_traces = int(half_intercept_result.sorted_trace_index.shape[0])
    cell_id, cell_v2, cell_status, cell_count = _core_cell_arrays_from_app_result(
        half_intercept_result,
        model=model,
    )
    return CoreRefractionHalfInterceptResult(
        file_id=str(half_intercept_result.qc.get('file_id', '')),
        n_traces=n_traces,
        bedrock_velocity_mode=str(half_intercept_result.bedrock_velocity_mode),
        bedrock_velocity_m_s=float(half_intercept_result.bedrock_velocity_m_s),
        bedrock_slowness_s_per_m=float(
            half_intercept_result.bedrock_slowness_s_per_m
        ),
        bedrock_velocity_status=(
            'per_cell'
            if half_intercept_result.bedrock_velocity_mode == 'solve_cell'
            else 'solved'
        ),
        v2_m_s=float(half_intercept_result.bedrock_velocity_m_s),
        node_id=_i64(half_intercept_result.node_id),
        node_half_intercept_time_s=_f64(
            half_intercept_result.node_half_intercept_time_s
        ),
        node_solution_status=_status(half_intercept_result.node_solution_status),
        node_pick_count=_i64(half_intercept_result.node_pick_count),
        node_used_observation_count=_i64(half_intercept_result.node_used_pick_count),
        node_rejected_observation_count=_i64(
            half_intercept_result.node_rejected_pick_count
        ),
        source_endpoint=_core_endpoint_from_app(half_intercept_result, 'source'),
        receiver_endpoint=_core_endpoint_from_app(half_intercept_result, 'receiver'),
        trace_index_sorted=_i64(half_intercept_result.sorted_trace_index),
        source_endpoint_key_sorted=np.ascontiguousarray(
            half_intercept_result.source_endpoint_key_sorted
        ),
        receiver_endpoint_key_sorted=np.ascontiguousarray(
            half_intercept_result.receiver_endpoint_key_sorted
        ),
        source_endpoint_id_sorted=None,
        receiver_endpoint_id_sorted=None,
        source_node_id_sorted=_i64(half_intercept_result.source_node_id_sorted),
        receiver_node_id_sorted=_i64(half_intercept_result.receiver_node_id_sorted),
        source_half_intercept_time_s_sorted=_f64(
            half_intercept_result.source_half_intercept_time_s_sorted
        ),
        receiver_half_intercept_time_s_sorted=_f64(
            half_intercept_result.receiver_half_intercept_time_s_sorted
        ),
        trace_half_intercept_time_s_sorted=_f64(
            half_intercept_result.source_half_intercept_time_s_sorted
        )
        + _f64(half_intercept_result.receiver_half_intercept_time_s_sorted),
        trace_half_intercept_status_sorted=np.full(n_traces, 'ok', dtype=_STATUS_DTYPE),
        pick_time_s_sorted=np.full(n_traces, np.nan, dtype=np.float64),
        modeled_pick_time_s_sorted=_f64(
            half_intercept_result.estimated_first_break_time_s_sorted
        ),
        residual_s_sorted=_f64(half_intercept_result.first_break_residual_s_sorted),
        residual_ms_sorted=_f64(half_intercept_result.first_break_residual_s_sorted)
        * 1000.0,
        used_observation_mask_sorted=np.ascontiguousarray(
            half_intercept_result.used_observation_mask_sorted,
            dtype=bool,
        ),
        rejected_observation_mask_sorted=np.zeros(n_traces, dtype=bool),
        rejected_iteration_sorted=np.full(n_traces, -1, dtype=np.int64),
        cell_id=cell_id,
        cell_bedrock_slowness_s_per_m=np.divide(
            1.0,
            cell_v2,
            out=np.full(cell_v2.shape, np.nan, dtype=np.float64),
            where=np.isfinite(cell_v2) & (cell_v2 != 0.0),
        ),
        cell_bedrock_velocity_m_s=cell_v2,
        cell_velocity_status=cell_status,
        cell_observation_count=cell_count,
        row_midpoint_cell_id=np.asarray([], dtype=np.int64)
        if half_intercept_result.row_midpoint_cell_id is None
        else _i64(half_intercept_result.row_midpoint_cell_id),
        row_midpoint_bedrock_slowness_s_per_m=np.asarray([], dtype=np.float64),
        row_midpoint_bedrock_velocity_m_s=np.asarray([], dtype=np.float64),
        row_midpoint_v2_m_s=np.asarray([], dtype=np.float64),
        cell_v2_m_s=cell_v2,
        rms_residual_s=float(_nan_stat(half_intercept_result.residual_time_s, 'rms')),
        rms_residual_ms=float(
            _nan_stat(half_intercept_result.residual_time_s, 'rms') * 1000.0
        ),
        residual_mean_s=float(_nan_stat(half_intercept_result.residual_time_s, 'mean')),
        residual_median_s=float(
            _nan_stat(half_intercept_result.residual_time_s, 'median')
        ),
        residual_mad_s=float(_nan_stat(half_intercept_result.residual_time_s, 'mad')),
        residual_max_abs_s=float(
            _nan_stat(half_intercept_result.residual_time_s, 'max_abs')
        ),
        solver_success=True,
        solver_status=0,
        solver_message='app DTO conversion',
        solver_cost=float('nan'),
        solver_optimality=float('nan'),
        solver_iterations=0,
        robust_enabled=False,
        robust_stop_reason='disabled',
        robust_iteration_summaries=(),
        n_initial_used_observations=int(
            np.count_nonzero(half_intercept_result.used_observation_mask_sorted)
        ),
        n_final_used_observations=int(
            np.count_nonzero(half_intercept_result.used_observation_mask_sorted)
        ),
        n_rejected_observations=0,
        qc=dict(half_intercept_result.qc),
        debug_design=None,
        debug_solve_result=None,
    )


def _validate_app_half_intercept_for_core_conversion(
    result: RefractionHalfInterceptTimeResult,
) -> None:
    bedrock_slowness = float(result.bedrock_slowness_s_per_m)
    bedrock_velocity = float(result.bedrock_velocity_m_s)
    if not np.isfinite(bedrock_slowness) or bedrock_slowness <= 0.0:
        raise RefractionWeatheringThicknessError(
            'bedrock_slowness_s_per_m must be positive and finite'
        )
    if not np.isfinite(bedrock_velocity) or bedrock_velocity <= 0.0:
        raise RefractionWeatheringThicknessError(
            'bedrock_velocity_m_s must be positive and finite'
        )
    if abs((1.0 / bedrock_slowness) - bedrock_velocity) > max(
        1.0e-6,
        abs(bedrock_velocity) * 1.0e-9,
    ):
        raise RefractionWeatheringThicknessError(
            'bedrock_velocity_m_s does not match bedrock_slowness_s_per_m'
        )
    node_id = _i64(result.node_id)
    valid = np.asarray(result.valid_observation_mask_sorted, dtype=bool)
    _validate_endpoint_nodes_used_by_valid_observations(
        result,
        node_id,
        side='source',
        valid_observation_mask_sorted=valid,
    )
    _validate_endpoint_nodes_used_by_valid_observations(
        result,
        node_id,
        side='receiver',
        valid_observation_mask_sorted=valid,
    )
    _validate_known_nodes(
        _i64(result.source_node_id_sorted)[valid],
        node_id,
        name='source_node_id_sorted',
    )
    _validate_known_nodes(
        _i64(result.receiver_node_id_sorted)[valid],
        node_id,
        name='receiver_node_id_sorted',
    )


def _validate_endpoint_nodes_used_by_valid_observations(
    result: RefractionHalfInterceptTimeResult,
    node_id: np.ndarray,
    *,
    side: str,
    valid_observation_mask_sorted: np.ndarray,
) -> None:
    endpoint_key = np.asarray(getattr(result, f'{side}_endpoint_key'), dtype=object)
    endpoint_node_id = _i64(getattr(result, f'{side}_node_id'))
    sorted_key = np.asarray(
        getattr(result, f'{side}_endpoint_key_sorted'),
        dtype=object,
    )
    if endpoint_key.shape != endpoint_node_id.shape:
        raise RefractionWeatheringThicknessError(
            f'{side}_endpoint_key and {side}_node_id shape mismatch'
        )
    if sorted_key.shape != valid_observation_mask_sorted.shape:
        raise RefractionWeatheringThicknessError(
            f'{side}_endpoint_key_sorted shape mismatch'
        )
    valid_keys = {
        str(key)
        for key in sorted_key[valid_observation_mask_sorted].reshape(-1).tolist()
    }
    referenced_nodes = [
        int(node)
        for key, node in zip(
            endpoint_key.reshape(-1).tolist(),
            endpoint_node_id.reshape(-1).tolist(),
            strict=True,
        )
        if str(key) in valid_keys
    ]
    _validate_known_nodes(
        np.asarray(referenced_nodes, dtype=np.int64),
        node_id,
        name=f'{side}_node_id',
    )


def _validate_known_nodes(values: np.ndarray, node_id: np.ndarray, *, name: str) -> None:
    known = {int(node) for node in np.asarray(node_id, dtype=np.int64).tolist()}
    missing = [
        int(node)
        for node in np.asarray(values, dtype=np.int64).reshape(-1).tolist()
        if int(node) not in known
    ]
    if missing:
        raise RefractionWeatheringThicknessError(
            f'{name} references unknown node_id {missing[0]}'
        )


def _core_endpoint_from_app(
    result: RefractionHalfInterceptTimeResult,
    side: str,
) -> CoreRefractionHalfInterceptEndpointResult:
    prefix = 'source' if side == 'source' else 'receiver'
    endpoint_key = getattr(result, f'{prefix}_endpoint_key')
    endpoint_id = getattr(result, f'{prefix}_id')
    node_id = getattr(result, f'{prefix}_node_id')
    pick_count = getattr(result, f'{prefix}_pick_count', None)
    if pick_count is None:
        pick_count = np.zeros(np.asarray(endpoint_key).shape, dtype=np.int64)
    half_intercept_time_s = _values_by_node(
        node_id=node_id,
        table_node_id=result.node_id,
        table_values=result.node_half_intercept_time_s,
        fill_value=np.nan,
    )
    solution_status = _status_by_node(
        node_id=node_id,
        table_node_id=result.node_id,
        table_values=result.node_solution_status,
        fill_value='missing_node',
    )
    return CoreRefractionHalfInterceptEndpointResult(
        endpoint_key=np.ascontiguousarray(endpoint_key),
        endpoint_id=_i64(endpoint_id),
        node_id=_i64(node_id),
        half_intercept_time_s=half_intercept_time_s,
        solution_status=solution_status,
        pick_count=_i64(pick_count),
        used_observation_count=_i64(pick_count),
        rejected_observation_count=np.zeros(np.asarray(endpoint_key).shape, dtype=np.int64),
    )


def _core_cell_arrays_from_app_result(
    result: RefractionHalfInterceptTimeResult,
    *,
    model: RefractionStaticModelRequest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if model.bedrock_velocity_mode != 'solve_cell' or model.refractor_cell is None:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=_STATUS_DTYPE),
            np.asarray([], dtype=np.int64),
        )
    n_cells = (
        int(model.refractor_cell.number_of_cell_x)
        * int(model.refractor_cell.number_of_cell_y)
    )
    cell_id = np.arange(n_cells, dtype=np.int64)
    velocity = np.full(n_cells, np.nan, dtype=np.float64)
    status = np.full(n_cells, 'inactive', dtype=_STATUS_DTYPE)
    count = np.zeros(n_cells, dtype=np.int64)
    active = result.active_cell_id
    active_velocity = result.cell_bedrock_velocity_m_s
    active_status = result.cell_velocity_status
    if active is not None and active_velocity is not None:
        for index, raw_cell in enumerate(np.asarray(active, dtype=np.int64).tolist()):
            if 0 <= int(raw_cell) < n_cells:
                velocity[int(raw_cell)] = float(active_velocity[index])
                status[int(raw_cell)] = (
                    'solved'
                    if active_status is None
                    else str(np.asarray(active_status)[index])
                )
    low_fold = result.qc.get('low_fold_cell_id', [])
    for raw_cell in np.asarray(low_fold, dtype=np.int64).reshape(-1).tolist():
        if 0 <= int(raw_cell) < n_cells and not np.isfinite(velocity[int(raw_cell)]):
            status[int(raw_cell)] = 'low_fold'
    raw_count = result.qc.get('cell_observation_count')
    if raw_count is not None:
        raw = np.asarray(raw_count, dtype=np.int64).reshape(-1)
        count[: min(n_cells, raw.shape[0])] = raw[:n_cells]
    return (
        np.ascontiguousarray(cell_id, dtype=np.int64),
        np.ascontiguousarray(velocity, dtype=np.float64),
        np.ascontiguousarray(status, dtype=_STATUS_DTYPE),
        np.ascontiguousarray(count, dtype=np.int64),
    )


def _optional_cell_array(
    values: np.ndarray | None,
    model: RefractionStaticModelRequest,
) -> np.ndarray | None:
    if values is None or model.bedrock_velocity_mode != 'solve_cell':
        return None
    return _f64(values)


def _optional_cell_status(
    values: np.ndarray | None,
    model: RefractionStaticModelRequest,
) -> np.ndarray | None:
    if values is None or model.bedrock_velocity_mode != 'solve_cell':
        return None
    return _status(values)


def _values_by_node(
    *,
    node_id: np.ndarray,
    table_node_id: np.ndarray,
    table_values: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    lookup = {
        int(node): float(value)
        for node, value in zip(table_node_id, table_values, strict=True)
    }
    return np.ascontiguousarray(
        np.asarray(
            [lookup.get(int(node), fill_value) for node in node_id],
            dtype=np.float64,
        )
    )


def _status_by_node(
    *,
    node_id: np.ndarray,
    table_node_id: np.ndarray,
    table_values: np.ndarray,
    fill_value: str,
) -> np.ndarray:
    lookup = {
        int(node): str(value)
        for node, value in zip(table_node_id, table_values, strict=True)
    }
    return np.ascontiguousarray(
        np.asarray(
            [lookup.get(int(node), fill_value) for node in node_id],
            dtype=_STATUS_DTYPE,
        )
    )


def write_refraction_weathering_thickness_artifacts(
    job_dir: Path,
    result: RefractionWeatheringThicknessResult,
) -> dict[str, Path]:
    """Write weathering-thickness QC JSON and CSV tables."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / REFRACTION_WEATHERING_QC_JSON_NAME
    node_path = root / REFRACTION_WEATHERING_NODES_CSV_NAME
    source_path = root / REFRACTION_WEATHERING_SOURCES_CSV_NAME
    receiver_path = root / REFRACTION_WEATHERING_RECEIVERS_CSV_NAME
    trace_path = root / REFRACTION_WEATHERING_TRACE_PREVIEW_CSV_NAME
    write_json_atomic(qc_path, result.qc, allow_nan=False, ensure_ascii=True, sort_keys=True)
    write_csv_atomic(node_path, columns=_NODE_COLUMNS, rows=_node_rows(result), lineterminator='\r\n')
    write_csv_atomic(source_path, columns=_SOURCE_COLUMNS, rows=_source_rows(result), lineterminator='\r\n')
    write_csv_atomic(receiver_path, columns=_RECEIVER_COLUMNS, rows=_receiver_rows(result), lineterminator='\r\n')
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


def _node_rows(result: RefractionWeatheringThicknessResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(result.node_id.shape[0])):
        rows.append(
            {
                'node_id': int(result.node_id[index]),
                'node_kind': str(result.node_kind[index]),
                'x_m': _csv_float(result.node_x_m[index]),
                'y_m': _csv_float(result.node_y_m[index]),
                'surface_elevation_m': _csv_float(result.node_surface_elevation_m[index]),
                'half_intercept_time_ms': _csv_float(result.node_half_intercept_time_ms[index]),
                'weathering_thickness_m': _csv_float(result.node_weathering_thickness_m[index]),
                'refractor_elevation_m': _csv_float(result.node_refractor_elevation_m[index]),
                'solution_status': str(result.node_solution_status[index]),
                'weathering_status': str(result.node_weathering_status[index]),
                'pick_count': int(result.node_pick_count[index]),
                'used_pick_count': int(result.node_used_pick_count[index]),
                'residual_rms_ms': _csv_float(result.node_residual_rms_s[index] * 1000.0),
                'residual_mad_ms': _csv_float(result.node_residual_mad_s[index] * 1000.0),
            }
        )
    return rows


def _source_rows(result: RefractionWeatheringThicknessResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(result.source_endpoint_key.shape[0])):
        rows.append(
            {
                'source_endpoint_key': str(result.source_endpoint_key[index]),
                'source_id': int(result.source_id[index]),
                'source_node_id': int(result.source_node_id[index]),
                'source_x_m': _csv_float(result.source_x_m[index]),
                'source_y_m': _csv_float(result.source_y_m[index]),
                'source_surface_elevation_m': _csv_float(result.source_surface_elevation_m[index]),
                'source_half_intercept_time_ms': _csv_float(
                    result.source_half_intercept_time_s[index] * 1000.0
                ),
                'source_weathering_thickness_m': _csv_float(
                    result.source_weathering_thickness_m[index]
                ),
                'source_refractor_elevation_m': _csv_float(
                    result.source_refractor_elevation_m[index]
                ),
                'source_weathering_status': str(result.source_weathering_status[index]),
            }
        )
    return rows


def _receiver_rows(result: RefractionWeatheringThicknessResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(result.receiver_endpoint_key.shape[0])):
        rows.append(
            {
                'receiver_endpoint_key': str(result.receiver_endpoint_key[index]),
                'receiver_id': int(result.receiver_id[index]),
                'receiver_node_id': int(result.receiver_node_id[index]),
                'receiver_x_m': _csv_float(result.receiver_x_m[index]),
                'receiver_y_m': _csv_float(result.receiver_y_m[index]),
                'receiver_surface_elevation_m': _csv_float(result.receiver_surface_elevation_m[index]),
                'receiver_half_intercept_time_ms': _csv_float(
                    result.receiver_half_intercept_time_s[index] * 1000.0
                ),
                'receiver_weathering_thickness_m': _csv_float(
                    result.receiver_weathering_thickness_m[index]
                ),
                'receiver_refractor_elevation_m': _csv_float(
                    result.receiver_refractor_elevation_m[index]
                ),
                'receiver_weathering_status': str(result.receiver_weathering_status[index]),
            }
        )
    return rows


def _trace_preview_rows(
    result: RefractionWeatheringThicknessResult,
) -> list[dict[str, Any]]:
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
                'source_weathering_thickness_m': _csv_float(
                    result.source_weathering_thickness_m_sorted[index]
                ),
                'receiver_weathering_thickness_m': _csv_float(
                    result.receiver_weathering_thickness_m_sorted[index]
                ),
                'source_refractor_elevation_m': _csv_float(
                    result.source_refractor_elevation_m_sorted[index]
                ),
                'receiver_refractor_elevation_m': _csv_float(
                    result.receiver_refractor_elevation_m_sorted[index]
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


def _weathering_status(values: np.ndarray) -> np.ndarray:
    status = _status(values)
    mapping = {
        'clipped_lower': 'clipped_half_intercept_lower',
        'clipped_upper': 'clipped_half_intercept_upper',
        'invalid_nonfinite_input': 'invalid_half_intercept',
        'invalid_weathering_thickness': 'invalid_refractor_elevation',
        'negative_weathering_thickness': 'negative_thickness',
    }
    out = status.copy()
    for old, new in mapping.items():
        out[status == old] = new
    return np.ascontiguousarray(out, dtype=_STATUS_DTYPE)


def _status(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=_STATUS_DTYPE)


def _optional_status(values: np.ndarray | None) -> np.ndarray | None:
    return None if values is None else _status(values)


def _f64(values: object) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _optional_f64(values: object | None) -> np.ndarray | None:
    return None if values is None else _f64(values)


def _i64(values: object) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def _optional_i64(values: object | None) -> np.ndarray | None:
    return None if values is None else _i64(values)


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
    raise RefractionWeatheringThicknessError(f'unsupported statistic: {stat}')


def _json_float_or_none(value: object) -> float | None:
    out = float(value)
    return out if np.isfinite(out) else None


def _nan_stat(values: object, stat: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan')
    if stat == 'mean':
        return float(np.mean(arr))
    if stat == 'median':
        return float(np.median(arr))
    if stat == 'rms':
        return float(np.sqrt(np.mean(arr * arr)))
    if stat == 'mad':
        center = float(np.median(arr))
        return float(1.4826 * np.median(np.abs(arr - center)))
    if stat == 'max_abs':
        return float(np.max(np.abs(arr)))
    raise RefractionWeatheringThicknessError(f'unsupported statistic: {stat}')


def _status_counts(values: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in values.tolist():
        key = str(raw)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _copy_cell_threshold_qc(payload: dict[str, Any], upstream: dict[str, Any]) -> None:
    for key in _CELL_THRESHOLD_QC_KEYS:
        if key in upstream:
            payload[key] = upstream[key]
    layer_qc = upstream.get('layers')
    if isinstance(layer_qc, dict):
        payload['layers'] = layer_qc


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
    'REFRACTION_WEATHERING_NODES_CSV_NAME',
    'REFRACTION_WEATHERING_QC_JSON_NAME',
    'REFRACTION_WEATHERING_RECEIVERS_CSV_NAME',
    'REFRACTION_WEATHERING_SOURCES_CSV_NAME',
    'REFRACTION_WEATHERING_TRACE_PREVIEW_CSV_NAME',
    'RefractionWeatheringThicknessError',
    'RefractionWeatheringThicknessResult',
    'build_refraction_weathering_thickness_model',
    'build_refraction_weathering_core_context',
    'compute_weathering_thickness_from_half_intercept_time',
    'compute_weathering_thickness_scalar',
    'estimate_weathering_thickness_core_context_from_first_breaks',
    'estimate_weathering_thickness_from_first_breaks',
    'write_refraction_weathering_thickness_artifacts',
]
