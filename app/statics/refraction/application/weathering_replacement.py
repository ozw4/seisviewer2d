"""Weathering-replacement static shifts for GLI refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app.statics.refraction.contracts.options import RefractionStaticApplyOptions
from app.statics.refraction.contracts.apply import RefractionStaticApplyRequest
from app.core.state import AppState
from app.services.common.array_validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_integer_int64 as _coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_1d_string_array as _coerce_1d_string_array,
    coerce_finite_float as _coerce_finite_float,
)
from app.services.common.artifact_io import write_csv_atomic, write_json_atomic
from app.statics.refraction.domain.first_layer import (
    validate_resolved_first_layer_velocity_match,
)
from app.statics.refraction.domain.status import LOCAL_V2_STATUS_VALUES
from app.statics.refraction.domain.t1lsst import (
    RefractionT1LSSTError,
    compute_t1lsst_1layer_weathering_correction,
)
from app.statics.refraction.domain.types import (
    RefractionStaticInputModel,
    RefractionWeatheringReplacementStaticsResult,
    RefractionWeatheringThicknessResult,
    ResolvedRefractionFirstLayer,
)
from app.statics.refraction.application.weathering import (
    estimate_weathering_thickness_from_first_breaks,
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
_ZERO_SHIFT_ATOL_S = 1.0e-12
_FORMULA_TEXT = 'shift = z * (1/vb - 1/vw)'
_SIGN_CONVENTION_TEXT = 'corrected(t) = raw(t - shift_s)'
_CELL_THRESHOLD_QC_KEYS = (
    'min_observations_per_cell',
    'n_low_fold_cells',
    'n_observations_rejected_by_low_fold_cell',
    'low_fold_cell_rejection_reason',
    'low_fold_cell_id',
    'cell_observation_count',
)

_STATUS_PRIORITY = {
    'ok': 0,
    'not_observed': 1,
    'clipped_half_intercept_lower': 2,
    'clipped_half_intercept_upper': 3,
    'low_fold': 4,
    'exceeds_max_thickness': 5,
    'exceeds_max_abs_shift': 6,
    'invalid_shift': 7,
    'negative_weathering_thickness': 8,
    'invalid_weathering_thickness': 9,
    'inactive': 10,
    'invalid_velocity': 11,
    'outside_refractor_cell_grid': 12,
    'inactive_v2_cell': 13,
    'low_fold_v2_cell': 14,
    'invalid_local_v2': 15,
    'v2_not_greater_than_v1': 16,
    'missing_endpoint': 17,
    'missing_node': 18,
}

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
    state: AppState,
    job_dir: Path | None = None,
    input_model: RefractionStaticInputModel | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> RefractionWeatheringReplacementStaticsResult:
    """Run the GLI weathering model, then compute replacement static shifts."""
    try:
        weathering_kwargs: dict[str, Any] = {
            'req': req,
            'state': state,
            'job_dir': job_dir,
        }
        if input_model is not None:
            weathering_kwargs['input_model'] = input_model
        if resolved_first_layer is not None:
            weathering_kwargs['resolved_first_layer'] = resolved_first_layer
        weathering_result = estimate_weathering_thickness_from_first_breaks(
            **weathering_kwargs
        )
        return build_refraction_weathering_replacement_statics(
            weathering_result=weathering_result,
            apply_options=req.apply,
            job_dir=job_dir,
            resolved_first_layer=resolved_first_layer,
        )
    except RefractionWeatheringReplacementStaticsError:
        raise
    except ValueError as exc:
        raise RefractionWeatheringReplacementStaticsError(str(exc)) from exc


def build_refraction_weathering_replacement_statics(
    *,
    weathering_result: RefractionWeatheringThicknessResult,
    apply_options: RefractionStaticApplyOptions | None = None,
    job_dir: Path | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> RefractionWeatheringReplacementStaticsResult:
    """Compute weathering-replacement statics from weathering thickness."""
    data = _validate_weathering_result(weathering_result)
    velocity = _validate_velocity_context(
        weathering_result,
        resolved_first_layer=resolved_first_layer,
    )
    max_abs_shift_ms = _resolve_max_abs_shift_ms(apply_options)

    node_pos = {int(node): index for index, node in enumerate(data.node_id.tolist())}
    _validate_endpoint_nodes(data.source_node_id, node_pos, name='source_node_id')
    _validate_endpoint_nodes(data.receiver_node_id, node_pos, name='receiver_node_id')

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
    source_v2_sorted = _local_or_scalar_v2(
        data.source_v2_m_s_sorted,
        shape=data.source_weathering_thickness_m_sorted.shape,
        scalar_v2_m_s=velocity.bedrock_velocity_m_s,
    )
    receiver_v2_sorted = _local_or_scalar_v2(
        data.receiver_v2_m_s_sorted,
        shape=data.receiver_weathering_thickness_m_sorted.shape,
        scalar_v2_m_s=velocity.bedrock_velocity_m_s,
    )
    node_raw_shift = _compute_weathering_replacement_shift_with_local_v2(
        weathering_thickness_m=data.node_weathering_thickness_m,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        bedrock_velocity_m_s=node_v2,
    )
    node_shift, node_static_status = _classify_component_status(
        weathering_status=data.node_weathering_status,
        weathering_thickness_m=data.node_weathering_thickness_m,
        raw_shift_s=node_raw_shift,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    if data.node_v2_status is not None:
        node_static_status = _overlay_local_v2_status(
            node_static_status,
            data.node_v2_status,
        )
        node_shift = _nan_for_status(node_shift, node_static_status)

    source_raw_shift = _compute_weathering_replacement_shift_with_local_v2(
        weathering_thickness_m=data.source_weathering_thickness_m,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        bedrock_velocity_m_s=source_v2,
    )
    source_shift, source_static_status = _classify_component_status(
        weathering_status=data.source_weathering_status,
        weathering_thickness_m=data.source_weathering_thickness_m,
        raw_shift_s=source_raw_shift,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    if data.source_v2_status is not None:
        source_static_status = _overlay_local_v2_status(
            source_static_status,
            data.source_v2_status,
        )
        source_shift = _nan_for_status(source_shift, source_static_status)

    receiver_raw_shift = _compute_weathering_replacement_shift_with_local_v2(
        weathering_thickness_m=data.receiver_weathering_thickness_m,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        bedrock_velocity_m_s=receiver_v2,
    )
    receiver_shift, receiver_static_status = _classify_component_status(
        weathering_status=data.receiver_weathering_status,
        weathering_thickness_m=data.receiver_weathering_thickness_m,
        raw_shift_s=receiver_raw_shift,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    if data.receiver_v2_status is not None:
        receiver_static_status = _overlay_local_v2_status(
            receiver_static_status,
            data.receiver_v2_status,
        )
        receiver_shift = _nan_for_status(receiver_shift, receiver_static_status)

    source_raw_shift_sorted = _compute_weathering_replacement_shift_with_local_v2(
        weathering_thickness_m=data.source_weathering_thickness_m_sorted,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        bedrock_velocity_m_s=source_v2_sorted,
    )
    source_shift_sorted, source_static_status_sorted = _classify_component_status(
        weathering_status=data.source_weathering_status_sorted,
        weathering_thickness_m=data.source_weathering_thickness_m_sorted,
        raw_shift_s=source_raw_shift_sorted,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    if data.source_v2_status_sorted is not None:
        source_static_status_sorted = _overlay_local_v2_status(
            source_static_status_sorted,
            data.source_v2_status_sorted,
        )
        source_shift_sorted = _nan_for_status(
            source_shift_sorted,
            source_static_status_sorted,
        )

    receiver_raw_shift_sorted = _compute_weathering_replacement_shift_with_local_v2(
        weathering_thickness_m=data.receiver_weathering_thickness_m_sorted,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        bedrock_velocity_m_s=receiver_v2_sorted,
    )
    receiver_shift_sorted, receiver_static_status_sorted = _classify_component_status(
        weathering_status=data.receiver_weathering_status_sorted,
        weathering_thickness_m=data.receiver_weathering_thickness_m_sorted,
        raw_shift_s=receiver_raw_shift_sorted,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    if data.receiver_v2_status_sorted is not None:
        receiver_static_status_sorted = _overlay_local_v2_status(
            receiver_static_status_sorted,
            data.receiver_v2_status_sorted,
        )
        receiver_shift_sorted = _nan_for_status(
            receiver_shift_sorted,
            receiver_static_status_sorted,
        )
    trace_shift = _combine_trace_shifts(source_shift_sorted, receiver_shift_sorted)
    trace_static_status = _classify_trace_status(
        source_status=source_static_status_sorted,
        receiver_status=receiver_static_status_sorted,
        trace_shift_s=trace_shift,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    trace_static_valid = _trace_valid_mask(
        trace_shift_s=trace_shift,
        max_abs_shift_ms=max_abs_shift_ms,
    )

    qc = _build_qc(
        velocity=velocity,
        data=data,
        node_shift_s=node_shift,
        node_static_status=node_static_status,
        source_shift_s=source_shift,
        source_static_status=source_static_status,
        receiver_shift_s=receiver_shift,
        receiver_static_status=receiver_static_status,
        trace_shift_s=trace_shift,
        trace_static_status=trace_static_status,
        trace_static_valid_mask=trace_static_valid,
        max_abs_shift_ms=max_abs_shift_ms,
        upstream_qc=getattr(weathering_result, 'qc', {}),
    )

    result = RefractionWeatheringReplacementStaticsResult(
        bedrock_velocity_mode=velocity.mode,
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
        node_weathering_replacement_shift_s=node_shift,
        node_weathering_replacement_shift_ms=np.ascontiguousarray(
            node_shift * 1000.0,
            dtype=np.float64,
        ),
        node_static_status=node_static_status,
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
        source_weathering_replacement_shift_s=source_shift,
        source_static_status=source_static_status,
        receiver_endpoint_key=data.receiver_endpoint_key,
        receiver_id=data.receiver_id,
        receiver_node_id=data.receiver_node_id,
        receiver_x_m=data.receiver_x_m,
        receiver_y_m=data.receiver_y_m,
        receiver_surface_elevation_m=data.receiver_surface_elevation_m,
        receiver_half_intercept_time_s=data.receiver_half_intercept_time_s,
        receiver_weathering_thickness_m=data.receiver_weathering_thickness_m,
        receiver_refractor_elevation_m=data.receiver_refractor_elevation_m,
        receiver_weathering_replacement_shift_s=receiver_shift,
        receiver_static_status=receiver_static_status,
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
        source_weathering_replacement_shift_s_sorted=source_shift_sorted,
        receiver_weathering_replacement_shift_s_sorted=receiver_shift_sorted,
        weathering_replacement_trace_shift_s_sorted=trace_shift,
        source_static_status_sorted=source_static_status_sorted,
        receiver_static_status_sorted=receiver_static_status_sorted,
        trace_static_status_sorted=trace_static_status,
        trace_static_valid_mask_sorted=trace_static_valid,
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
        qc=qc,
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
    if job_dir is not None:
        write_refraction_weathering_replacement_artifacts(Path(job_dir), result)
    return result


def compute_weathering_replacement_shift_s(
    *,
    weathering_thickness_m: np.ndarray,
    weathering_velocity_m_s: float,
    bedrock_velocity_m_s: float | np.ndarray,
) -> np.ndarray:
    """Compute ``shift = z * (1/vb - 1/vw)`` in seconds."""
    try:
        return compute_t1lsst_1layer_weathering_correction(
            sh1_m=weathering_thickness_m,
            v1_m_s=weathering_velocity_m_s,
            v2_m_s=bedrock_velocity_m_s,
        )
    except RefractionT1LSSTError as exc:
        if str(exc) == 'v2_m_s must be greater than v1_m_s':
            raise RefractionWeatheringReplacementStaticsError(
                'bedrock_velocity_m_s must be greater than weathering_velocity_m_s'
            ) from exc
        raise RefractionWeatheringReplacementStaticsError(
            str(exc)
            .replace('sh1_m', 'weathering_thickness_m')
            .replace('v1_m_s', 'weathering_velocity_m_s')
            .replace('v2_m_s', 'bedrock_velocity_m_s')
        ) from exc


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


def _compute_weathering_replacement_shift_with_local_v2(
    *,
    weathering_thickness_m: np.ndarray,
    weathering_velocity_m_s: float,
    bedrock_velocity_m_s: np.ndarray,
) -> np.ndarray:
    thickness = np.asarray(weathering_thickness_m, dtype=np.float64)
    v2 = np.asarray(bedrock_velocity_m_s, dtype=np.float64)
    if thickness.shape != v2.shape:
        raise RefractionWeatheringReplacementStaticsError(
            'weathering_thickness_m and bedrock_velocity_m_s shape mismatch'
        )
    out = np.full(thickness.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(thickness) & np.isfinite(v2) & (v2 > weathering_velocity_m_s)
    if np.any(valid):
        out[valid] = compute_weathering_replacement_shift_s(
            weathering_thickness_m=thickness[valid],
            weathering_velocity_m_s=weathering_velocity_m_s,
            bedrock_velocity_m_s=v2[valid],
        )
    return np.ascontiguousarray(out, dtype=np.float64)


def _local_or_scalar_v2(
    local_v2_m_s: np.ndarray | None,
    *,
    shape: tuple[int, ...],
    scalar_v2_m_s: float,
) -> np.ndarray:
    if local_v2_m_s is None:
        return np.full(shape, scalar_v2_m_s, dtype=np.float64)
    return np.ascontiguousarray(local_v2_m_s, dtype=np.float64)


def _overlay_local_v2_status(
    component_status: np.ndarray,
    local_v2_status: np.ndarray,
) -> np.ndarray:
    status = np.asarray(component_status).astype(_STATUS_DTYPE, copy=True)
    local = np.asarray(local_v2_status).astype(str, copy=False)
    invalid_local = np.isin(local, list(LOCAL_V2_STATUS_VALUES))
    status[invalid_local] = local[invalid_local]
    return np.ascontiguousarray(status, dtype=_STATUS_DTYPE)


def _nan_for_status(values: np.ndarray, status: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(values, dtype=np.float64).copy()
    text = np.asarray(status).astype(str, copy=False)
    out[np.isin(text, list(LOCAL_V2_STATUS_VALUES))] = np.nan
    return out


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
        source_node_id=_coerce_1d_integer(
            _required(weathering_result, 'source_node_id'),
            name='weathering_result.source_node_id',
            expected_shape=source_shape,
        ),
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
        receiver_node_id=_coerce_1d_integer(
            _required(weathering_result, 'receiver_node_id'),
            name='weathering_result.receiver_node_id',
            expected_shape=receiver_shape,
        ),
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


def _validate_endpoint_nodes(
    endpoint_node_id: np.ndarray,
    node_pos: dict[int, int],
    *,
    name: str,
) -> None:
    missing = sorted({int(node) for node in endpoint_node_id.tolist()} - node_pos.keys())
    if missing:
        raise RefractionWeatheringReplacementStaticsError(
            f'{name} references unknown node_id values: {missing}'
        )


def _classify_component_status(
    *,
    weathering_status: np.ndarray,
    weathering_thickness_m: np.ndarray,
    raw_shift_s: np.ndarray,
    max_abs_shift_ms: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    n_items = int(weathering_thickness_m.shape[0])
    status = np.full(n_items, 'ok', dtype=_STATUS_DTYPE)
    shift = np.ascontiguousarray(raw_shift_s, dtype=np.float64).copy()

    inherited = np.asarray(weathering_status).astype(str, copy=False)
    thickness = np.asarray(weathering_thickness_m, dtype=np.float64)

    _assign(status, inherited == 'clipped_half_intercept_lower', 'clipped_half_intercept_lower')
    _assign(status, inherited == 'clipped_half_intercept_upper', 'clipped_half_intercept_upper')
    _assign(status, inherited == 'low_fold', 'low_fold')
    _assign(status, inherited == 'exceeds_max_thickness', 'exceeds_max_thickness')

    negative_thickness = np.isfinite(thickness) & (thickness < 0.0)
    invalid_thickness = (~np.isfinite(thickness)) & (inherited != 'inactive')
    invalid_thickness &= inherited != 'missing_node'
    invalid_thickness |= inherited == 'invalid_weathering_thickness'
    negative_thickness |= (inherited == 'negative_thickness') | (
        inherited == 'negative_weathering_thickness'
    )
    unknown_invalid = _unknown_invalid_weathering_status(inherited)
    invalid_shift = np.isfinite(thickness) & (thickness >= 0.0) & (~np.isfinite(shift))
    exceeds_max_abs_shift = np.zeros(n_items, dtype=bool)
    if max_abs_shift_ms is not None:
        exceeds_max_abs_shift = np.isfinite(shift) & (
            np.abs(shift) * 1000.0 > max_abs_shift_ms
        )

    _assign(status, exceeds_max_abs_shift, 'exceeds_max_abs_shift')
    _assign(status, invalid_shift, 'invalid_shift')
    _assign(status, negative_thickness, 'negative_weathering_thickness')
    _assign(status, invalid_thickness | unknown_invalid, 'invalid_weathering_thickness')
    for local_status in LOCAL_V2_STATUS_VALUES:
        _assign(status, inherited == local_status, local_status)
    _assign(status, inherited == 'inactive', 'inactive')
    _assign(status, inherited == 'missing_node', 'missing_node')

    invalid_output = np.isin(
        status.astype(str),
        [
            'missing_node',
            'missing_endpoint',
            'invalid_velocity',
            'outside_refractor_cell_grid',
            'inactive_v2_cell',
            'low_fold_v2_cell',
            'invalid_local_v2',
            'v2_not_greater_than_v1',
            'inactive',
            'invalid_weathering_thickness',
            'negative_weathering_thickness',
            'invalid_shift',
        ],
    )
    shift[invalid_output] = np.nan
    return (
        np.ascontiguousarray(shift, dtype=np.float64),
        np.ascontiguousarray(status, dtype=_STATUS_DTYPE),
    )


def _copy_cell_threshold_qc(payload: dict[str, Any], upstream: dict[str, Any]) -> None:
    for key in _CELL_THRESHOLD_QC_KEYS:
        if key in upstream:
            payload[key] = upstream[key]
    layer_qc = upstream.get('layers')
    if isinstance(layer_qc, dict):
        payload['layers'] = layer_qc


def _map_endpoint_component(
    *,
    endpoint_node_id: np.ndarray,
    endpoint_weathering_thickness_m: np.ndarray,
    endpoint_weathering_status: np.ndarray,
    node_pos: dict[int, int],
    node_shift_s: np.ndarray,
    max_abs_shift_ms: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    raw_shift = np.full(endpoint_node_id.shape, np.nan, dtype=np.float64)
    for index, raw_node in enumerate(endpoint_node_id.tolist()):
        raw_shift[index] = node_shift_s[node_pos[int(raw_node)]]
    return _classify_component_status(
        weathering_status=endpoint_weathering_status,
        weathering_thickness_m=endpoint_weathering_thickness_m,
        raw_shift_s=raw_shift,
        max_abs_shift_ms=max_abs_shift_ms,
    )


def _map_trace_component(
    *,
    node_id_sorted: np.ndarray,
    thickness_m_sorted: np.ndarray,
    weathering_status_sorted: np.ndarray,
    node_pos: dict[int, int],
    node_shift_s: np.ndarray,
    max_abs_shift_ms: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    n_traces = int(node_id_sorted.shape[0])
    raw_shift = np.full(n_traces, np.nan, dtype=np.float64)
    status = np.asarray(weathering_status_sorted).astype(_STATUS_DTYPE, copy=True)
    for index, raw_node in enumerate(node_id_sorted.tolist()):
        node_index = node_pos.get(int(raw_node))
        if node_index is None:
            status[index] = 'missing_node'
            continue
        raw_shift[index] = node_shift_s[node_index]
    return _classify_component_status(
        weathering_status=status,
        weathering_thickness_m=thickness_m_sorted,
        raw_shift_s=raw_shift,
        max_abs_shift_ms=max_abs_shift_ms,
    )


def _combine_trace_shifts(
    source_shift_s: np.ndarray,
    receiver_shift_s: np.ndarray,
) -> np.ndarray:
    out = np.full(source_shift_s.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(source_shift_s) & np.isfinite(receiver_shift_s)
    out[finite] = source_shift_s[finite] + receiver_shift_s[finite]
    return np.ascontiguousarray(out, dtype=np.float64)


def _classify_trace_status(
    *,
    source_status: np.ndarray,
    receiver_status: np.ndarray,
    trace_shift_s: np.ndarray,
    max_abs_shift_ms: float | None,
) -> np.ndarray:
    n_traces = int(trace_shift_s.shape[0])
    status = np.full(n_traces, 'ok', dtype=_STATUS_DTYPE)
    source = np.asarray(source_status).astype(str, copy=False)
    receiver = np.asarray(receiver_status).astype(str, copy=False)
    for index in range(n_traces):
        status[index] = _highest_priority_status(source[index], receiver[index])
    _assign(status, ~np.isfinite(trace_shift_s), 'invalid_shift')
    if max_abs_shift_ms is not None:
        exceeds = np.isfinite(trace_shift_s) & (
            np.abs(trace_shift_s) * 1000.0 > max_abs_shift_ms
        )
        _assign(status, exceeds, 'exceeds_max_abs_shift')
    return np.ascontiguousarray(status, dtype=_STATUS_DTYPE)


def _trace_valid_mask(
    *,
    trace_shift_s: np.ndarray,
    max_abs_shift_ms: float | None,
) -> np.ndarray:
    valid = np.isfinite(trace_shift_s)
    if max_abs_shift_ms is not None:
        valid &= np.abs(trace_shift_s) * 1000.0 <= max_abs_shift_ms
    return np.ascontiguousarray(valid, dtype=bool)


def _build_qc(
    *,
    velocity: _VelocityContext,
    data: _ValidatedWeathering,
    node_shift_s: np.ndarray,
    node_static_status: np.ndarray,
    source_shift_s: np.ndarray,
    source_static_status: np.ndarray,
    receiver_shift_s: np.ndarray,
    receiver_static_status: np.ndarray,
    trace_shift_s: np.ndarray,
    trace_static_status: np.ndarray,
    trace_static_valid_mask: np.ndarray,
    max_abs_shift_ms: float | None,
    upstream_qc: dict[str, Any],
) -> dict[str, Any]:
    valid_node_shift_ms = _valid_shift_ms(node_shift_s, node_static_status)
    valid_source_shift_ms = _valid_shift_ms(source_shift_s, source_static_status)
    valid_receiver_shift_ms = _valid_shift_ms(receiver_shift_s, receiver_static_status)
    valid_trace_shift_ms = trace_shift_s[trace_static_valid_mask] * 1000.0
    finite_trace_shift = trace_shift_s[np.isfinite(trace_shift_s)]

    qc: dict[str, Any] = {
        'method': 'gli_variable_thickness',
        'static_component': 'weathering_replacement',
        'bedrock_velocity_mode': velocity.mode,
        'bedrock_velocity_m_s': float(velocity.bedrock_velocity_m_s),
        'bedrock_slowness_s_per_m': float(velocity.bedrock_slowness_s_per_m),
        'weathering_velocity_m_s': float(velocity.weathering_velocity_m_s),
        'replacement_slowness_delta_s_per_m': float(
            velocity.replacement_slowness_delta_s_per_m
        ),
        'n_traces': int(data.n_traces),
        'n_valid_observations': int(
            np.count_nonzero(data.valid_observation_mask_sorted)
        ),
        'n_used_observations': int(np.count_nonzero(data.used_observation_mask_sorted)),
        'n_nodes': int(data.n_nodes),
        'n_source_endpoints': int(data.source_endpoint_key.shape[0]),
        'n_receiver_endpoints': int(data.receiver_endpoint_key.shape[0]),
        'node_shift_min_ms': _json_stat(valid_node_shift_ms, 'min'),
        'node_shift_max_ms': _json_stat(valid_node_shift_ms, 'max'),
        'node_shift_median_ms': _json_stat(valid_node_shift_ms, 'median'),
        'node_shift_p95_abs_ms': _json_stat(np.abs(valid_node_shift_ms), 'p95'),
        'source_shift_min_ms': _json_stat(valid_source_shift_ms, 'min'),
        'source_shift_max_ms': _json_stat(valid_source_shift_ms, 'max'),
        'source_shift_median_ms': _json_stat(valid_source_shift_ms, 'median'),
        'receiver_shift_min_ms': _json_stat(valid_receiver_shift_ms, 'min'),
        'receiver_shift_max_ms': _json_stat(valid_receiver_shift_ms, 'max'),
        'receiver_shift_median_ms': _json_stat(valid_receiver_shift_ms, 'median'),
        'trace_shift_min_ms': _json_stat(valid_trace_shift_ms, 'min'),
        'trace_shift_max_ms': _json_stat(valid_trace_shift_ms, 'max'),
        'trace_shift_median_ms': _json_stat(valid_trace_shift_ms, 'median'),
        'trace_shift_p95_abs_ms': _json_stat(np.abs(valid_trace_shift_ms), 'p95'),
        'trace_shift_max_abs_ms': _json_stat(np.abs(valid_trace_shift_ms), 'max'),
        'negative_trace_shift_count': int(
            np.count_nonzero(valid_trace_shift_ms < -_ZERO_SHIFT_ATOL_S * 1000.0)
        ),
        'positive_trace_shift_count': int(
            np.count_nonzero(valid_trace_shift_ms > _ZERO_SHIFT_ATOL_S * 1000.0)
        ),
        'zero_trace_shift_count': int(
            np.count_nonzero(np.abs(valid_trace_shift_ms) <= _ZERO_SHIFT_ATOL_S * 1000.0)
        ),
        'invalid_trace_shift_count': int(
            np.count_nonzero(~np.isfinite(trace_shift_s))
        ),
        'max_abs_shift_ms': _json_optional_float(max_abs_shift_ms),
        'exceeds_max_abs_shift_count': int(
            np.count_nonzero(trace_static_status == 'exceeds_max_abs_shift')
        ),
        'inactive_node_count': int(np.count_nonzero(node_static_status == 'inactive')),
        'low_fold_node_count': int(np.count_nonzero(node_static_status == 'low_fold')),
        'invalid_weathering_thickness_count': int(
            np.count_nonzero(node_static_status == 'invalid_weathering_thickness')
        ),
        'exceeds_max_thickness_count': int(
            np.count_nonzero(node_static_status == 'exceeds_max_thickness')
        ),
        'finite_trace_shift_count': int(finite_trace_shift.shape[0]),
        'node_static_status_counts': _status_counts(node_static_status),
        'source_static_status_counts': _status_counts(source_static_status),
        'receiver_static_status_counts': _status_counts(receiver_static_status),
        'trace_static_status_counts': _status_counts(trace_static_status),
        'sign_convention': _SIGN_CONVENTION_TEXT,
        'formula': _FORMULA_TEXT,
    }
    _copy_cell_threshold_qc(qc, upstream_qc)
    _assert_json_safe(qc)
    return qc


def _valid_shift_ms(shift_s: np.ndarray, status: np.ndarray) -> np.ndarray:
    arr = np.asarray(shift_s, dtype=np.float64)
    status_arr = np.asarray(status).astype(str, copy=False)
    return arr[(status_arr == 'ok') & np.isfinite(arr)] * 1000.0


def _resolve_max_abs_shift_ms(
    apply_options: RefractionStaticApplyOptions | None,
) -> float | None:
    if apply_options is None:
        return None
    value = getattr(apply_options, 'max_abs_shift_ms', None)
    if value is None:
        return None
    return _positive_finite(value, name='apply.max_abs_shift_ms')


def _unknown_invalid_weathering_status(status: np.ndarray) -> np.ndarray:
    text = np.asarray(status).astype(str, copy=False)
    known = np.isin(
        text,
        [
            'ok',
            'zero_thickness',
            'inactive',
            'invalid_weathering_thickness',
            'negative_thickness',
            'negative_weathering_thickness',
            'low_fold',
            'clipped_half_intercept_lower',
            'clipped_half_intercept_upper',
            'exceeds_max_thickness',
            'invalid_surface_elevation',
            'invalid_refractor_elevation',
            'missing_node',
            'outside_refractor_cell_grid',
            'inactive_v2_cell',
            'low_fold_v2_cell',
            'invalid_local_v2',
            'v2_not_greater_than_v1',
        ],
    )
    return ~known


def _assign(status: np.ndarray, mask: np.ndarray, value: str) -> None:
    current = np.asarray(status).astype(str, copy=False)
    priority = _STATUS_PRIORITY[value]
    should_assign = np.asarray(mask, dtype=bool) & np.asarray(
        [_STATUS_PRIORITY.get(str(item), -1) <= priority for item in current],
        dtype=bool,
    )
    status[should_assign] = value


def _highest_priority_status(left: str, right: str) -> str:
    if _STATUS_PRIORITY.get(left, -1) >= _STATUS_PRIORITY.get(right, -1):
        return left
    return right


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
    raise RefractionWeatheringReplacementStaticsError(
        f'unsupported statistic: {stat}'
    )


def _status_counts(values: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in values.tolist():
        key = str(raw)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _json_optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


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


def _assert_json_safe(payload: dict[str, Any]) -> None:
    json.dumps(payload, allow_nan=False)


__all__ = [
    'REFRACTION_WEATHERING_REPLACEMENT_NODES_CSV_NAME',
    'REFRACTION_WEATHERING_REPLACEMENT_QC_JSON_NAME',
    'REFRACTION_WEATHERING_REPLACEMENT_RECEIVERS_CSV_NAME',
    'REFRACTION_WEATHERING_REPLACEMENT_SOURCES_CSV_NAME',
    'REFRACTION_WEATHERING_REPLACEMENT_TRACE_PREVIEW_CSV_NAME',
    'RefractionWeatheringReplacementStaticsError',
    'RefractionWeatheringReplacementStaticsResult',
    'build_refraction_weathering_replacement_statics',
    'compute_weathering_replacement_shift_s',
    'compute_weathering_replacement_shift_scalar_s',
    'compute_weathering_replacement_statics_from_first_breaks',
    'write_refraction_weathering_replacement_artifacts',
]
