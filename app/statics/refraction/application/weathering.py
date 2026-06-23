"""Weathering-thickness model conversion for GLI refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app.statics.refraction.contracts.apply import RefractionStaticApplyRequest
from app.statics.refraction.contracts.model import RefractionStaticModelRequest
from app.services.common.array_validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_integer_int64 as _coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_1d_string_array as _coerce_1d_string_array,
    coerce_finite_float as _coerce_finite_float,
)
from app.services.common.artifact_io import write_csv_atomic, write_json_atomic
from seis_statics.refraction.cell_grid import (
    assign_points_to_refraction_cells,
    build_refraction_cell_grid,
)
from seis_statics.refraction.cell_coordinates import (
    effective_refraction_cell_grid_config,
    project_refraction_cell_points,
)
from app.statics.refraction.application.core_options import (
    resolve_weathering_velocity_from_model_request as resolve_weathering_velocity_m_s,
)
from app.statics.refraction.application.core_options import (
    refractor_cell_options_from_request,
)
from app.statics.refraction.application.half_intercept import (
    estimate_refraction_half_intercept_times_from_first_breaks,
)
from app.statics.refraction.ports.runtime import RefractionRuntime
from seis_statics.refraction.t1lsst import (
    RefractionT1LSSTError,
    compute_t1lsst_1layer_thickness,
)
from app.statics.refraction.domain.types import (
    RefractionHalfInterceptTimeResult,
    RefractionStaticInputModel,
    RefractionWeatheringThicknessResult,
    ResolvedRefractionFirstLayer,
)

REFRACTION_WEATHERING_QC_JSON_NAME = 'refraction_weathering_thickness_qc.json'
REFRACTION_WEATHERING_NODES_CSV_NAME = 'refraction_weathering_nodes.csv'
REFRACTION_WEATHERING_SOURCES_CSV_NAME = 'refraction_weathering_sources.csv'
REFRACTION_WEATHERING_RECEIVERS_CSV_NAME = 'refraction_weathering_receivers.csv'
REFRACTION_WEATHERING_TRACE_PREVIEW_CSV_NAME = (
    'refraction_weathering_trace_preview.csv'
)

_STATUS_DTYPE = '<U32'
_ENDPOINT_KEY_DTYPE = object
_ZERO_THICKNESS_ATOL_M = 1.0e-9
_SLOWNESS_RTOL = 1.0e-6
_VELOCITY_RTOL = 1.0e-6
_FORMULA_TEXT = 'z = T * vb * vw / sqrt(vb^2 - vw^2)'
_LOW_FOLD_V2_CELL_STATUS = 'low_fold_v2_cell'
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
    'solved': 0,
    'zero_thickness': 1,
    'clipped_lower': 2,
    'clipped_half_intercept_lower': 2,
    'clipped_upper': 3,
    'clipped_half_intercept_upper': 3,
    'low_fold': 4,
    'invalid_refractor_elevation': 5,
    'invalid_surface_elevation': 6,
    'exceeds_max_thickness': 7,
    'negative_thickness': 8,
    'invalid_solution': 9,
    'missing_solution': 9,
    'invalid_half_intercept': 9,
    'inactive': 10,
    'outside_refractor_cell_grid': 11,
    'inactive_v2_cell': 12,
    'low_fold_v2_cell': 13,
    'invalid_local_v2': 14,
    'v2_not_greater_than_v1': 15,
    'missing_endpoint': 16,
    'missing_node': 16,
}

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
class _VelocityContext:
    mode: Literal['solve_global', 'fixed_global', 'solve_cell']
    bedrock_slowness_s_per_m: float
    bedrock_velocity_m_s: float
    weathering_velocity_m_s: float
    max_weathering_thickness_m: float | None


@dataclass(frozen=True)
class _LocalV2Model:
    node_cell_id: np.ndarray
    node_v2_m_s: np.ndarray
    node_v2_status: np.ndarray
    source_cell_id: np.ndarray
    source_v2_m_s: np.ndarray
    source_v2_status: np.ndarray
    receiver_cell_id: np.ndarray
    receiver_v2_m_s: np.ndarray
    receiver_v2_status: np.ndarray
    source_cell_id_sorted: np.ndarray
    source_v2_m_s_sorted: np.ndarray
    source_v2_status_sorted: np.ndarray
    receiver_cell_id_sorted: np.ndarray
    receiver_v2_m_s_sorted: np.ndarray
    receiver_v2_status_sorted: np.ndarray


@dataclass(frozen=True)
class _ValidatedHalfIntercept:
    node_id: np.ndarray
    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_surface_elevation_m: np.ndarray
    node_kind: np.ndarray
    node_half_intercept_time_s: np.ndarray
    node_solution_status: np.ndarray
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
    source_endpoint_key: np.ndarray
    source_id: np.ndarray
    source_node_id: np.ndarray
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    source_surface_elevation_m: np.ndarray
    receiver_endpoint_key: np.ndarray
    receiver_id: np.ndarray
    receiver_node_id: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    receiver_surface_elevation_m: np.ndarray
    sorted_trace_index: np.ndarray
    valid_observation_mask_sorted: np.ndarray
    used_observation_mask_sorted: np.ndarray
    source_surface_elevation_m_sorted: np.ndarray
    receiver_surface_elevation_m_sorted: np.ndarray
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
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


def estimate_weathering_thickness_from_first_breaks(
    *,
    req: RefractionStaticApplyRequest,
    runtime: RefractionRuntime | None = None,
    state: object | None = None,
    job_dir: Path | None = None,
    input_model: RefractionStaticInputModel | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> RefractionWeatheringThicknessResult:
    """Estimate half-intercepts from first breaks, then convert to thickness."""
    try:
        half_intercept_kwargs: dict[str, Any] = {
            'req': req,
            'job_dir': job_dir,
        }
        if runtime is not None:
            half_intercept_kwargs['runtime'] = runtime
        elif state is not None:
            half_intercept_kwargs['state'] = state
        if input_model is not None:
            half_intercept_kwargs['input_model'] = input_model
        if resolved_first_layer is not None:
            half_intercept_kwargs['resolved_first_layer'] = resolved_first_layer
        half_intercept_result = (
            estimate_refraction_half_intercept_times_from_first_breaks(
                **half_intercept_kwargs
            )
        )
        return build_refraction_weathering_thickness_model(
            half_intercept_result=half_intercept_result,
            model=req.model,
            job_dir=job_dir,
            resolved_first_layer=resolved_first_layer,
        )
    except RefractionWeatheringThicknessError:
        raise
    except ValueError as exc:
        raise RefractionWeatheringThicknessError(str(exc)) from exc


def build_refraction_weathering_thickness_model(
    *,
    half_intercept_result: RefractionHalfInterceptTimeResult,
    model: RefractionStaticModelRequest,
    job_dir: Path | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> RefractionWeatheringThicknessResult:
    """Convert GLI half-intercept times to a weathering-thickness model."""
    data = _validate_half_intercept_result(half_intercept_result)
    velocity = _validate_velocity_context(
        half_intercept_result=half_intercept_result,
        model=model,
        resolved_first_layer=resolved_first_layer,
    )
    node_pos = {int(node): index for index, node in enumerate(data.node_id.tolist())}
    _validate_endpoint_nodes(data.source_node_id, node_pos, name='source_node_id')
    _validate_endpoint_nodes(data.receiver_node_id, node_pos, name='receiver_node_id')
    _validate_valid_observation_nodes(
        data.source_node_id_sorted,
        data.valid_observation_mask_sorted,
        node_pos,
        name='source_node_id_sorted',
    )
    _validate_valid_observation_nodes(
        data.receiver_node_id_sorted,
        data.valid_observation_mask_sorted,
        node_pos,
        name='receiver_node_id_sorted',
    )

    local_v2 = _build_local_v2_model(
        data=data,
        model=model,
        velocity=velocity,
        low_fold_cell_id=_low_fold_cell_id_from_qc(
            getattr(half_intercept_result, 'qc', {})
        ),
    )
    node_bedrock_velocity = (
        local_v2.node_v2_m_s
        if local_v2 is not None
        else np.full(data.node_half_intercept_time_s.shape, velocity.bedrock_velocity_m_s)
    )
    source_bedrock_velocity = (
        local_v2.source_v2_m_s
        if local_v2 is not None
        else np.full(data.source_node_id.shape, velocity.bedrock_velocity_m_s)
    )
    receiver_bedrock_velocity = (
        local_v2.receiver_v2_m_s
        if local_v2 is not None
        else np.full(data.receiver_node_id.shape, velocity.bedrock_velocity_m_s)
    )
    node_thickness = _compute_weathering_thickness_with_local_v2(
        half_intercept_time_s=data.node_half_intercept_time_s,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        bedrock_velocity_m_s=node_bedrock_velocity,
    )
    node_refractor = _compute_refractor_elevation(
        data.node_surface_elevation_m,
        node_thickness,
    )
    node_status = _classify_weathering_status(
        solution_status=data.node_solution_status,
        half_intercept_time_s=data.node_half_intercept_time_s,
        thickness_m=node_thickness,
        surface_elevation_m=data.node_surface_elevation_m,
        refractor_elevation_m=node_refractor,
        max_weathering_thickness_m=velocity.max_weathering_thickness_m,
    )
    if local_v2 is not None:
        node_status = _overlay_local_v2_status(
            node_status,
            local_v2.node_v2_status,
        )

    source_half, _source_node_thickness, source_status, source_missing = _map_node_table(
        node_id=data.source_node_id,
        node_pos=node_pos,
        node_half_intercept_time_s=data.node_half_intercept_time_s,
        node_weathering_thickness_m=node_thickness,
        node_solution_status=data.node_solution_status,
    )
    source_thickness = _compute_weathering_thickness_with_local_v2(
        half_intercept_time_s=source_half,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        bedrock_velocity_m_s=source_bedrock_velocity,
    )
    source_refractor = _compute_refractor_elevation(
        data.source_surface_elevation_m,
        source_thickness,
    )
    source_status = _classify_weathering_status(
        solution_status=source_status,
        half_intercept_time_s=source_half,
        thickness_m=source_thickness,
        surface_elevation_m=data.source_surface_elevation_m,
        refractor_elevation_m=source_refractor,
        max_weathering_thickness_m=velocity.max_weathering_thickness_m,
        missing_node_mask=source_missing,
    )
    if local_v2 is not None:
        source_status = _overlay_local_v2_status(
            source_status,
            local_v2.source_v2_status,
        )

    (
        receiver_half,
        _receiver_node_thickness,
        receiver_status,
        receiver_missing,
    ) = _map_node_table(
        node_id=data.receiver_node_id,
        node_pos=node_pos,
        node_half_intercept_time_s=data.node_half_intercept_time_s,
        node_weathering_thickness_m=node_thickness,
        node_solution_status=data.node_solution_status,
    )
    receiver_thickness = _compute_weathering_thickness_with_local_v2(
        half_intercept_time_s=receiver_half,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        bedrock_velocity_m_s=receiver_bedrock_velocity,
    )
    receiver_refractor = _compute_refractor_elevation(
        data.receiver_surface_elevation_m,
        receiver_thickness,
    )
    receiver_status = _classify_weathering_status(
        solution_status=receiver_status,
        half_intercept_time_s=receiver_half,
        thickness_m=receiver_thickness,
        surface_elevation_m=data.receiver_surface_elevation_m,
        refractor_elevation_m=receiver_refractor,
        max_weathering_thickness_m=velocity.max_weathering_thickness_m,
        missing_node_mask=receiver_missing,
    )
    if local_v2 is not None:
        receiver_status = _overlay_local_v2_status(
            receiver_status,
            local_v2.receiver_v2_status,
        )

    source_half_sorted, source_thickness_sorted, source_refractor_sorted, source_status_sorted = (
        _map_trace_table_from_endpoints(
            endpoint_key_sorted=data.source_endpoint_key_sorted,
            endpoint_key=data.source_endpoint_key,
            node_id_sorted=data.source_node_id_sorted,
            node_pos=node_pos,
            endpoint_half_intercept_time_s=source_half,
            endpoint_weathering_thickness_m=source_thickness,
            endpoint_solution_status=source_status,
            surface_elevation_m_sorted=data.source_surface_elevation_m_sorted,
            max_weathering_thickness_m=velocity.max_weathering_thickness_m,
        )
    )
    receiver_half_sorted, receiver_thickness_sorted, receiver_refractor_sorted, receiver_status_sorted = (
        _map_trace_table_from_endpoints(
            endpoint_key_sorted=data.receiver_endpoint_key_sorted,
            endpoint_key=data.receiver_endpoint_key,
            node_id_sorted=data.receiver_node_id_sorted,
            node_pos=node_pos,
            endpoint_half_intercept_time_s=receiver_half,
            endpoint_weathering_thickness_m=receiver_thickness,
            endpoint_solution_status=receiver_status,
            surface_elevation_m_sorted=data.receiver_surface_elevation_m_sorted,
            max_weathering_thickness_m=velocity.max_weathering_thickness_m,
        )
    )
    if local_v2 is not None:
        source_status_sorted = _overlay_local_v2_status(
            source_status_sorted,
            local_v2.source_v2_status_sorted,
        )
        receiver_status_sorted = _overlay_local_v2_status(
            receiver_status_sorted,
            local_v2.receiver_v2_status_sorted,
        )

    qc = _build_qc(
        velocity=velocity,
        data=data,
        node_weathering_thickness_m=node_thickness,
        node_refractor_elevation_m=node_refractor,
        node_weathering_status=node_status,
        source_weathering_thickness_m=source_thickness,
        source_weathering_status=source_status,
        receiver_weathering_thickness_m=receiver_thickness,
        receiver_weathering_status=receiver_status,
        upstream_qc=getattr(half_intercept_result, 'qc', {}),
    )

    result = RefractionWeatheringThicknessResult(
        bedrock_velocity_mode=velocity.mode,
        bedrock_slowness_s_per_m=velocity.bedrock_slowness_s_per_m,
        bedrock_velocity_m_s=velocity.bedrock_velocity_m_s,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        node_id=data.node_id,
        node_x_m=data.node_x_m,
        node_y_m=data.node_y_m,
        node_surface_elevation_m=data.node_surface_elevation_m,
        node_kind=data.node_kind,
        node_half_intercept_time_s=data.node_half_intercept_time_s,
        node_half_intercept_time_ms=np.ascontiguousarray(
            data.node_half_intercept_time_s * 1000.0,
            dtype=np.float64,
        ),
        node_weathering_thickness_m=node_thickness,
        node_refractor_elevation_m=np.ascontiguousarray(
            node_refractor,
            dtype=np.float64,
        ),
        node_solution_status=data.node_solution_status,
        node_weathering_status=node_status,
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
        source_half_intercept_time_s=source_half,
        source_weathering_thickness_m=source_thickness,
        source_refractor_elevation_m=np.ascontiguousarray(
            source_refractor,
            dtype=np.float64,
        ),
        source_weathering_status=source_status,
        receiver_endpoint_key=data.receiver_endpoint_key,
        receiver_id=data.receiver_id,
        receiver_node_id=data.receiver_node_id,
        receiver_x_m=data.receiver_x_m,
        receiver_y_m=data.receiver_y_m,
        receiver_surface_elevation_m=data.receiver_surface_elevation_m,
        receiver_half_intercept_time_s=receiver_half,
        receiver_weathering_thickness_m=receiver_thickness,
        receiver_refractor_elevation_m=np.ascontiguousarray(
            receiver_refractor,
            dtype=np.float64,
        ),
        receiver_weathering_status=receiver_status,
        sorted_trace_index=data.sorted_trace_index,
        valid_observation_mask_sorted=data.valid_observation_mask_sorted,
        used_observation_mask_sorted=data.used_observation_mask_sorted,
        source_endpoint_key_sorted=data.source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=data.receiver_endpoint_key_sorted,
        source_node_id_sorted=data.source_node_id_sorted,
        receiver_node_id_sorted=data.receiver_node_id_sorted,
        source_half_intercept_time_s_sorted=source_half_sorted,
        receiver_half_intercept_time_s_sorted=receiver_half_sorted,
        source_weathering_thickness_m_sorted=source_thickness_sorted,
        receiver_weathering_thickness_m_sorted=receiver_thickness_sorted,
        source_refractor_elevation_m_sorted=source_refractor_sorted,
        receiver_refractor_elevation_m_sorted=receiver_refractor_sorted,
        source_weathering_status_sorted=source_status_sorted,
        receiver_weathering_status_sorted=receiver_status_sorted,
        estimated_first_break_time_s_sorted=data.estimated_first_break_time_s_sorted,
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
        node_v2_cell_id=(
            None if local_v2 is None else local_v2.node_cell_id
        ),
        node_v2_m_s=None if local_v2 is None else local_v2.node_v2_m_s,
        node_v2_status=None if local_v2 is None else local_v2.node_v2_status,
        source_v2_cell_id=(
            None if local_v2 is None else local_v2.source_cell_id
        ),
        source_v2_m_s=None if local_v2 is None else local_v2.source_v2_m_s,
        source_v2_status=None if local_v2 is None else local_v2.source_v2_status,
        receiver_v2_cell_id=(
            None if local_v2 is None else local_v2.receiver_cell_id
        ),
        receiver_v2_m_s=None if local_v2 is None else local_v2.receiver_v2_m_s,
        receiver_v2_status=(
            None if local_v2 is None else local_v2.receiver_v2_status
        ),
        source_v2_cell_id_sorted=(
            None if local_v2 is None else local_v2.source_cell_id_sorted
        ),
        source_v2_m_s_sorted=(
            None if local_v2 is None else local_v2.source_v2_m_s_sorted
        ),
        source_v2_status_sorted=(
            None if local_v2 is None else local_v2.source_v2_status_sorted
        ),
        receiver_v2_cell_id_sorted=(
            None if local_v2 is None else local_v2.receiver_cell_id_sorted
        ),
        receiver_v2_m_s_sorted=(
            None if local_v2 is None else local_v2.receiver_v2_m_s_sorted
        ),
        receiver_v2_status_sorted=(
            None if local_v2 is None else local_v2.receiver_v2_status_sorted
        ),
    )
    if job_dir is not None:
        write_refraction_weathering_thickness_artifacts(Path(job_dir), result)
    return result


def compute_weathering_thickness_from_half_intercept_time(
    *,
    half_intercept_time_s: np.ndarray,
    weathering_velocity_m_s: float,
    bedrock_velocity_m_s: float | np.ndarray,
) -> np.ndarray:
    """Convert half-intercept time to weathering thickness with the GLI relation."""
    try:
        return compute_t1lsst_1layer_thickness(
            t1_s=half_intercept_time_s,
            v1_m_s=weathering_velocity_m_s,
            v2_m_s=bedrock_velocity_m_s,
        )
    except RefractionT1LSSTError as exc:
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


def _build_local_v2_model(
    *,
    data: _ValidatedHalfIntercept,
    model: RefractionStaticModelRequest,
    velocity: _VelocityContext,
    low_fold_cell_id: np.ndarray,
) -> _LocalV2Model | None:
    if velocity.mode != 'solve_cell':
        return None
    refractor_cell = getattr(model, 'refractor_cell', None)
    if refractor_cell is None:
        raise RefractionWeatheringThicknessError(
            'model.refractor_cell is required when model.bedrock_velocity_mode is solve_cell'
        )
    if data.active_cell_id is None or data.cell_bedrock_velocity_m_s is None:
        raise RefractionWeatheringThicknessError(
            'solve_cell half-intercept result requires active cell velocities'
        )
    cell_options = refractor_cell_options_from_request(refractor_cell)
    grid = build_refraction_cell_grid(
        effective_refraction_cell_grid_config(cell_options)
    )
    node = _project_local_v2(
        grid=grid,
        refractor_cell=cell_options,
        x_m=data.node_x_m,
        y_m=data.node_y_m,
        active_cell_id=data.active_cell_id,
        cell_bedrock_velocity_m_s=data.cell_bedrock_velocity_m_s,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        low_fold_cell_id=low_fold_cell_id,
    )
    source = _project_local_v2(
        grid=grid,
        refractor_cell=cell_options,
        x_m=data.source_x_m,
        y_m=data.source_y_m,
        active_cell_id=data.active_cell_id,
        cell_bedrock_velocity_m_s=data.cell_bedrock_velocity_m_s,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        low_fold_cell_id=low_fold_cell_id,
    )
    receiver = _project_local_v2(
        grid=grid,
        refractor_cell=cell_options,
        x_m=data.receiver_x_m,
        y_m=data.receiver_y_m,
        active_cell_id=data.active_cell_id,
        cell_bedrock_velocity_m_s=data.cell_bedrock_velocity_m_s,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        low_fold_cell_id=low_fold_cell_id,
    )
    source_sorted = _map_endpoint_local_v2_to_trace_order(
        endpoint_key_sorted=data.source_endpoint_key_sorted,
        endpoint_key=data.source_endpoint_key,
        endpoint_cell_id=source[0],
        endpoint_v2_m_s=source[1],
        endpoint_v2_status=source[2],
    )
    receiver_sorted = _map_endpoint_local_v2_to_trace_order(
        endpoint_key_sorted=data.receiver_endpoint_key_sorted,
        endpoint_key=data.receiver_endpoint_key,
        endpoint_cell_id=receiver[0],
        endpoint_v2_m_s=receiver[1],
        endpoint_v2_status=receiver[2],
    )
    return _LocalV2Model(
        node_cell_id=node[0],
        node_v2_m_s=node[1],
        node_v2_status=node[2],
        source_cell_id=source[0],
        source_v2_m_s=source[1],
        source_v2_status=source[2],
        receiver_cell_id=receiver[0],
        receiver_v2_m_s=receiver[1],
        receiver_v2_status=receiver[2],
        source_cell_id_sorted=source_sorted[0],
        source_v2_m_s_sorted=source_sorted[1],
        source_v2_status_sorted=source_sorted[2],
        receiver_cell_id_sorted=receiver_sorted[0],
        receiver_v2_m_s_sorted=receiver_sorted[1],
        receiver_v2_status_sorted=receiver_sorted[2],
    )


def _project_local_v2(
    *,
    grid: object,
    refractor_cell: object,
    x_m: np.ndarray,
    y_m: np.ndarray,
    active_cell_id: np.ndarray,
    cell_bedrock_velocity_m_s: np.ndarray,
    weathering_velocity_m_s: float,
    low_fold_cell_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = project_refraction_cell_points(
        x_m=x_m,
        y_m=y_m,
        mode=getattr(refractor_cell, 'coordinate_mode', 'grid_3d'),
        line_origin_x_m=getattr(refractor_cell, 'line_origin_x_m', None),
        line_origin_y_m=getattr(refractor_cell, 'line_origin_y_m', None),
        line_azimuth_deg=getattr(refractor_cell, 'line_azimuth_deg', None),
    )
    assignment = assign_points_to_refraction_cells(
        grid,
        x_m=projected.x_m,
        y_m=projected.y_m,
    )
    cell_id = np.ascontiguousarray(assignment.cell_id, dtype=np.int64)
    v2 = np.full(cell_id.shape, np.nan, dtype=np.float64)
    status = np.full(cell_id.shape, 'ok', dtype=_STATUS_DTYPE)
    active_cells = _coerce_1d_integer(active_cell_id, name='active_cell_id')
    cell_velocity = _coerce_1d_float(
        cell_bedrock_velocity_m_s,
        name='cell_bedrock_velocity_m_s',
        expected_shape=active_cells.shape,
    )
    velocity_by_cell = {
        int(raw_cell): float(cell_velocity[index])
        for index, raw_cell in enumerate(active_cells.tolist())
    }
    low_fold_cells = {
        int(value)
        for value in _coerce_1d_integer(
            low_fold_cell_id,
            name='low_fold_cell_id',
        ).tolist()
    }
    for index, raw_cell in enumerate(cell_id.tolist()):
        cell = int(raw_cell)
        if cell < 0:
            status[index] = 'outside_refractor_cell_grid'
            continue
        value = velocity_by_cell.get(cell)
        if value is None:
            status[index] = (
                _LOW_FOLD_V2_CELL_STATUS
                if cell in low_fold_cells
                else 'inactive_v2_cell'
            )
            continue
        v2[index] = value
        if not np.isfinite(value) or value <= 0.0:
            status[index] = 'invalid_local_v2'
        elif value <= weathering_velocity_m_s:
            status[index] = 'v2_not_greater_than_v1'
    return (
        np.ascontiguousarray(cell_id, dtype=np.int64),
        np.ascontiguousarray(v2, dtype=np.float64),
        np.ascontiguousarray(status, dtype=_STATUS_DTYPE),
    )


def _map_endpoint_local_v2_to_trace_order(
    *,
    endpoint_key_sorted: np.ndarray,
    endpoint_key: np.ndarray,
    endpoint_cell_id: np.ndarray,
    endpoint_v2_m_s: np.ndarray,
    endpoint_v2_status: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_to_index = _endpoint_key_to_index(endpoint_key)
    n_traces = int(endpoint_key_sorted.shape[0])
    cell_id = np.full(n_traces, -1, dtype=np.int64)
    v2 = np.full(n_traces, np.nan, dtype=np.float64)
    status = np.full(n_traces, 'missing_endpoint', dtype=_STATUS_DTYPE)
    for index, raw_key in enumerate(endpoint_key_sorted.tolist()):
        endpoint_index = key_to_index.get(str(raw_key))
        if endpoint_index is None:
            continue
        cell_id[index] = int(endpoint_cell_id[endpoint_index])
        v2[index] = float(endpoint_v2_m_s[endpoint_index])
        status[index] = str(endpoint_v2_status[endpoint_index])
    return (
        np.ascontiguousarray(cell_id, dtype=np.int64),
        np.ascontiguousarray(v2, dtype=np.float64),
        np.ascontiguousarray(status, dtype=_STATUS_DTYPE),
    )


def _compute_weathering_thickness_with_local_v2(
    *,
    half_intercept_time_s: np.ndarray,
    weathering_velocity_m_s: float,
    bedrock_velocity_m_s: np.ndarray,
) -> np.ndarray:
    half = np.asarray(half_intercept_time_s, dtype=np.float64)
    v2 = np.asarray(bedrock_velocity_m_s, dtype=np.float64)
    if half.shape != v2.shape:
        raise RefractionWeatheringThicknessError(
            'half_intercept_time_s and bedrock_velocity_m_s shape mismatch'
        )
    out = np.full(half.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(half) & np.isfinite(v2) & (v2 > weathering_velocity_m_s)
    if np.any(valid):
        out[valid] = compute_weathering_thickness_from_half_intercept_time(
            half_intercept_time_s=half[valid],
            weathering_velocity_m_s=weathering_velocity_m_s,
            bedrock_velocity_m_s=v2[valid],
        )
    return np.ascontiguousarray(out, dtype=np.float64)


def _overlay_local_v2_status(
    weathering_status: np.ndarray,
    local_v2_status: np.ndarray,
) -> np.ndarray:
    status = np.asarray(weathering_status).astype(_STATUS_DTYPE, copy=True)
    local = np.asarray(local_v2_status).astype(str, copy=False)
    invalid_local = local != 'ok'
    status[invalid_local] = local[invalid_local]
    return np.ascontiguousarray(status, dtype=_STATUS_DTYPE)


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


def _validate_half_intercept_result(
    half_intercept_result: RefractionHalfInterceptTimeResult,
) -> _ValidatedHalfIntercept:
    node_id = _coerce_1d_integer(
        _required(half_intercept_result, 'node_id'),
        name='half_intercept_result.node_id',
    )
    n_nodes = int(node_id.shape[0])
    if n_nodes <= 0:
        raise RefractionWeatheringThicknessError('node_id must contain at least one node')
    if np.unique(node_id).shape[0] != n_nodes:
        raise RefractionWeatheringThicknessError('node_id values must be unique')
    node_shape = (n_nodes,)

    sorted_trace_index = _coerce_1d_integer(
        _required(half_intercept_result, 'sorted_trace_index'),
        name='half_intercept_result.sorted_trace_index',
    )
    n_traces = int(sorted_trace_index.shape[0])
    trace_shape = (n_traces,)

    source_endpoint_key = _coerce_1d_string(
        _required(half_intercept_result, 'source_endpoint_key'),
        name='half_intercept_result.source_endpoint_key',
    )
    receiver_endpoint_key = _coerce_1d_string(
        _required(half_intercept_result, 'receiver_endpoint_key'),
        name='half_intercept_result.receiver_endpoint_key',
    )
    source_shape = (int(source_endpoint_key.shape[0]),)
    receiver_shape = (int(receiver_endpoint_key.shape[0]),)

    return _ValidatedHalfIntercept(
        node_id=node_id,
        node_x_m=_coerce_1d_float(
            _required(half_intercept_result, 'node_x_m'),
            name='half_intercept_result.node_x_m',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_y_m=_coerce_1d_float(
            _required(half_intercept_result, 'node_y_m'),
            name='half_intercept_result.node_y_m',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_surface_elevation_m=_coerce_1d_float(
            _required(half_intercept_result, 'node_elevation_m'),
            name='half_intercept_result.node_elevation_m',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_kind=_coerce_1d_string(
            _required(half_intercept_result, 'node_kind'),
            name='half_intercept_result.node_kind',
            expected_shape=node_shape,
        ),
        node_half_intercept_time_s=_coerce_1d_float(
            _required(half_intercept_result, 'node_half_intercept_time_s'),
            name='half_intercept_result.node_half_intercept_time_s',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_solution_status=_coerce_1d_string(
            _required(half_intercept_result, 'node_solution_status'),
            name='half_intercept_result.node_solution_status',
            expected_shape=node_shape,
            dtype=_STATUS_DTYPE,
        ),
        node_pick_count=_coerce_1d_integer(
            _required(half_intercept_result, 'node_pick_count'),
            name='half_intercept_result.node_pick_count',
            expected_shape=node_shape,
        ),
        node_used_pick_count=_coerce_1d_integer(
            _required(half_intercept_result, 'node_used_pick_count'),
            name='half_intercept_result.node_used_pick_count',
            expected_shape=node_shape,
        ),
        node_rejected_pick_count=_coerce_1d_integer(
            _required(half_intercept_result, 'node_rejected_pick_count'),
            name='half_intercept_result.node_rejected_pick_count',
            expected_shape=node_shape,
        ),
        node_residual_rms_s=_coerce_1d_float(
            _required(half_intercept_result, 'node_residual_rms_s'),
            name='half_intercept_result.node_residual_rms_s',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_residual_mad_s=_coerce_1d_float(
            _required(half_intercept_result, 'node_residual_mad_s'),
            name='half_intercept_result.node_residual_mad_s',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        active_cell_id=_optional_1d_integer(
            half_intercept_result,
            'active_cell_id',
            name='half_intercept_result.active_cell_id',
        ),
        inactive_cell_id=_optional_1d_integer(
            half_intercept_result,
            'inactive_cell_id',
            name='half_intercept_result.inactive_cell_id',
        ),
        cell_bedrock_slowness_s_per_m=_optional_1d_float(
            half_intercept_result,
            'cell_bedrock_slowness_s_per_m',
            name='half_intercept_result.cell_bedrock_slowness_s_per_m',
        ),
        cell_bedrock_velocity_m_s=_optional_1d_float(
            half_intercept_result,
            'cell_bedrock_velocity_m_s',
            name='half_intercept_result.cell_bedrock_velocity_m_s',
        ),
        cell_velocity_status=_optional_1d_string(
            half_intercept_result,
            'cell_velocity_status',
            name='half_intercept_result.cell_velocity_status',
            dtype=_STATUS_DTYPE,
        ),
        row_midpoint_cell_id=_optional_1d_integer(
            half_intercept_result,
            'row_midpoint_cell_id',
            name='half_intercept_result.row_midpoint_cell_id',
        ),
        source_endpoint_key=source_endpoint_key,
        source_id=_coerce_1d_integer(
            _required(half_intercept_result, 'source_id'),
            name='half_intercept_result.source_id',
            expected_shape=source_shape,
        ),
        source_node_id=_coerce_1d_integer(
            _required(half_intercept_result, 'source_node_id'),
            name='half_intercept_result.source_node_id',
            expected_shape=source_shape,
        ),
        source_x_m=_coerce_1d_float(
            _required(half_intercept_result, 'source_x_m'),
            name='half_intercept_result.source_x_m',
            expected_shape=source_shape,
            allow_nonfinite=True,
        ),
        source_y_m=_coerce_1d_float(
            _required(half_intercept_result, 'source_y_m'),
            name='half_intercept_result.source_y_m',
            expected_shape=source_shape,
            allow_nonfinite=True,
        ),
        source_surface_elevation_m=_coerce_1d_float(
            _required(half_intercept_result, 'source_elevation_m'),
            name='half_intercept_result.source_elevation_m',
            expected_shape=source_shape,
            allow_nonfinite=True,
        ),
        receiver_endpoint_key=receiver_endpoint_key,
        receiver_id=_coerce_1d_integer(
            _required(half_intercept_result, 'receiver_id'),
            name='half_intercept_result.receiver_id',
            expected_shape=receiver_shape,
        ),
        receiver_node_id=_coerce_1d_integer(
            _required(half_intercept_result, 'receiver_node_id'),
            name='half_intercept_result.receiver_node_id',
            expected_shape=receiver_shape,
        ),
        receiver_x_m=_coerce_1d_float(
            _required(half_intercept_result, 'receiver_x_m'),
            name='half_intercept_result.receiver_x_m',
            expected_shape=receiver_shape,
            allow_nonfinite=True,
        ),
        receiver_y_m=_coerce_1d_float(
            _required(half_intercept_result, 'receiver_y_m'),
            name='half_intercept_result.receiver_y_m',
            expected_shape=receiver_shape,
            allow_nonfinite=True,
        ),
        receiver_surface_elevation_m=_coerce_1d_float(
            _required(half_intercept_result, 'receiver_elevation_m'),
            name='half_intercept_result.receiver_elevation_m',
            expected_shape=receiver_shape,
            allow_nonfinite=True,
        ),
        sorted_trace_index=sorted_trace_index,
        valid_observation_mask_sorted=_coerce_1d_bool(
            _required(half_intercept_result, 'valid_observation_mask_sorted'),
            name='half_intercept_result.valid_observation_mask_sorted',
            expected_shape=trace_shape,
        ),
        used_observation_mask_sorted=_coerce_1d_bool(
            _required(half_intercept_result, 'used_observation_mask_sorted'),
            name='half_intercept_result.used_observation_mask_sorted',
            expected_shape=trace_shape,
        ),
        source_surface_elevation_m_sorted=_coerce_1d_float(
            _required(half_intercept_result, 'source_elevation_m_sorted'),
            name='half_intercept_result.source_elevation_m_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        receiver_surface_elevation_m_sorted=_coerce_1d_float(
            _required(half_intercept_result, 'receiver_elevation_m_sorted'),
            name='half_intercept_result.receiver_elevation_m_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        source_endpoint_key_sorted=_coerce_1d_string(
            _required(half_intercept_result, 'source_endpoint_key_sorted'),
            name='half_intercept_result.source_endpoint_key_sorted',
            expected_shape=trace_shape,
        ),
        receiver_endpoint_key_sorted=_coerce_1d_string(
            _required(half_intercept_result, 'receiver_endpoint_key_sorted'),
            name='half_intercept_result.receiver_endpoint_key_sorted',
            expected_shape=trace_shape,
        ),
        source_node_id_sorted=_coerce_1d_integer(
            _required(half_intercept_result, 'source_node_id_sorted'),
            name='half_intercept_result.source_node_id_sorted',
            expected_shape=trace_shape,
        ),
        receiver_node_id_sorted=_coerce_1d_integer(
            _required(half_intercept_result, 'receiver_node_id_sorted'),
            name='half_intercept_result.receiver_node_id_sorted',
            expected_shape=trace_shape,
        ),
        estimated_first_break_time_s_sorted=_coerce_1d_float(
            _required(half_intercept_result, 'estimated_first_break_time_s_sorted'),
            name='half_intercept_result.estimated_first_break_time_s_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        first_break_residual_s_sorted=_coerce_1d_float(
            _required(half_intercept_result, 'first_break_residual_s_sorted'),
            name='half_intercept_result.first_break_residual_s_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        row_trace_index_sorted=_coerce_1d_integer(
            _required(half_intercept_result, 'row_trace_index_sorted'),
            name='half_intercept_result.row_trace_index_sorted',
        ),
        row_source_node_id=_coerce_1d_integer(
            _required(half_intercept_result, 'row_source_node_id'),
            name='half_intercept_result.row_source_node_id',
        ),
        row_receiver_node_id=_coerce_1d_integer(
            _required(half_intercept_result, 'row_receiver_node_id'),
            name='half_intercept_result.row_receiver_node_id',
        ),
        row_distance_m=_coerce_1d_float(
            _required(half_intercept_result, 'row_distance_m'),
            name='half_intercept_result.row_distance_m',
        ),
        observed_pick_time_s=_coerce_1d_float(
            _required(half_intercept_result, 'observed_pick_time_s'),
            name='half_intercept_result.observed_pick_time_s',
        ),
        modeled_pick_time_s=_coerce_1d_float(
            _required(half_intercept_result, 'modeled_pick_time_s'),
            name='half_intercept_result.modeled_pick_time_s',
        ),
        residual_time_s=_coerce_1d_float(
            _required(half_intercept_result, 'residual_time_s'),
            name='half_intercept_result.residual_time_s',
        ),
        used_row_mask=_coerce_1d_bool(
            _required(half_intercept_result, 'used_row_mask'),
            name='half_intercept_result.used_row_mask',
        ),
        rejected_by_robust_mask=_coerce_1d_bool(
            _required(half_intercept_result, 'rejected_by_robust_mask'),
            name='half_intercept_result.rejected_by_robust_mask',
        ),
    )


def _validate_velocity_context(
    *,
    half_intercept_result: RefractionHalfInterceptTimeResult,
    model: RefractionStaticModelRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None,
) -> _VelocityContext:
    mode = _validate_velocity_mode(
        _required(half_intercept_result, 'bedrock_velocity_mode')
    )
    model_mode = _validate_velocity_mode(_required(model, 'bedrock_velocity_mode'))
    if mode != model_mode:
        raise RefractionWeatheringThicknessError(
            'bedrock_velocity_mode does not match the half-intercept result'
        )
    weathering_velocity = _positive_finite(
        resolve_weathering_velocity_m_s(
            model=model,
            resolved_first_layer=resolved_first_layer,
            name='model.weathering_velocity_m_s',
        ),
        name='model.weathering_velocity_m_s',
    )
    result_weathering_velocity = _positive_finite(
        _required(half_intercept_result, 'weathering_velocity_m_s'),
        name='half_intercept_result.weathering_velocity_m_s',
    )
    if not _close_velocity(weathering_velocity, result_weathering_velocity):
        raise RefractionWeatheringThicknessError(
            'model.weathering_velocity_m_s does not match the half-intercept result'
        )
    bedrock_slowness = _positive_finite(
        _required(half_intercept_result, 'bedrock_slowness_s_per_m'),
        name='half_intercept_result.bedrock_slowness_s_per_m',
    )
    bedrock_velocity = _positive_finite(
        _required(half_intercept_result, 'bedrock_velocity_m_s'),
        name='half_intercept_result.bedrock_velocity_m_s',
    )
    if bedrock_velocity <= weathering_velocity:
        raise RefractionWeatheringThicknessError(
            'bedrock_velocity_m_s must be greater than weathering_velocity_m_s'
        )
    derived_slowness = 1.0 / bedrock_velocity
    slowness_tol = max(1.0e-12, abs(bedrock_slowness) * _SLOWNESS_RTOL)
    if abs(derived_slowness - bedrock_slowness) > slowness_tol:
        raise RefractionWeatheringThicknessError(
            'bedrock_velocity_m_s does not match bedrock_slowness_s_per_m'
        )
    if mode == 'fixed_global':
        model_bedrock_velocity = _positive_finite(
            _required(model, 'bedrock_velocity_m_s'),
            name='model.bedrock_velocity_m_s',
        )
        if not _close_velocity(model_bedrock_velocity, bedrock_velocity):
            raise RefractionWeatheringThicknessError(
                'model.bedrock_velocity_m_s does not match the half-intercept result'
            )
    max_thickness = _optional_positive_finite(
        _required(model, 'max_weathering_thickness_m', allow_none=True),
        name='model.max_weathering_thickness_m',
    )
    return _VelocityContext(
        mode=mode,
        bedrock_slowness_s_per_m=bedrock_slowness,
        bedrock_velocity_m_s=bedrock_velocity,
        weathering_velocity_m_s=weathering_velocity,
        max_weathering_thickness_m=max_thickness,
    )


def _map_node_table(
    *,
    node_id: np.ndarray,
    node_pos: dict[int, int],
    node_half_intercept_time_s: np.ndarray,
    node_weathering_thickness_m: np.ndarray,
    node_solution_status: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_items = int(node_id.shape[0])
    half = np.full(n_items, np.nan, dtype=np.float64)
    thickness = np.full(n_items, np.nan, dtype=np.float64)
    solution_status = np.full(n_items, 'missing_solution', dtype=_STATUS_DTYPE)
    missing = np.zeros(n_items, dtype=bool)
    for index, raw_node in enumerate(node_id.tolist()):
        node_index = node_pos.get(int(raw_node))
        if node_index is None:
            missing[index] = True
            continue
        half[index] = float(node_half_intercept_time_s[node_index])
        thickness[index] = float(node_weathering_thickness_m[node_index])
        solution_status[index] = str(node_solution_status[node_index])
    return (
        np.ascontiguousarray(half, dtype=np.float64),
        np.ascontiguousarray(thickness, dtype=np.float64),
        np.ascontiguousarray(solution_status, dtype=_STATUS_DTYPE),
        np.ascontiguousarray(missing, dtype=bool),
    )


def _validate_endpoint_nodes(
    endpoint_node_id: np.ndarray,
    node_pos: dict[int, int],
    *,
    name: str,
) -> None:
    missing = sorted({int(node) for node in endpoint_node_id.tolist()} - node_pos.keys())
    if missing:
        raise RefractionWeatheringThicknessError(
            f'{name} references unknown node_id values: {missing}'
        )


def _validate_valid_observation_nodes(
    node_id_sorted: np.ndarray,
    valid_observation_mask_sorted: np.ndarray,
    node_pos: dict[int, int],
    *,
    name: str,
) -> None:
    node_ids = np.asarray(node_id_sorted, dtype=np.int64)
    valid = np.asarray(valid_observation_mask_sorted, dtype=bool)
    if node_ids.shape != valid.shape:
        raise RefractionWeatheringThicknessError(
            f'{name} and valid_observation_mask_sorted shape mismatch'
        )
    known = np.fromiter(
        (int(node) in node_pos for node in node_ids.tolist()),
        dtype=bool,
        count=int(node_ids.shape[0]),
    )
    missing = sorted({int(node) for node in node_ids[valid & ~known].tolist()})
    if missing:
        raise RefractionWeatheringThicknessError(
            f'{name} references unknown node_id values for valid observations: {missing}'
        )


def _map_trace_table(
    *,
    node_id_sorted: np.ndarray,
    node_pos: dict[int, int],
    node_half_intercept_time_s: np.ndarray,
    node_weathering_thickness_m: np.ndarray,
    node_solution_status: np.ndarray,
    surface_elevation_m_sorted: np.ndarray,
    max_weathering_thickness_m: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_traces = int(node_id_sorted.shape[0])
    half = np.full(n_traces, np.nan, dtype=np.float64)
    thickness = np.full(n_traces, np.nan, dtype=np.float64)
    surface = np.full(n_traces, np.nan, dtype=np.float64)
    solution_status = np.full(n_traces, 'missing_solution', dtype=_STATUS_DTYPE)
    missing = np.zeros(n_traces, dtype=bool)
    for index, raw_node in enumerate(node_id_sorted.tolist()):
        node_index = node_pos.get(int(raw_node))
        if node_index is None:
            missing[index] = True
            continue
        half[index] = float(node_half_intercept_time_s[node_index])
        thickness[index] = float(node_weathering_thickness_m[node_index])
        surface[index] = float(surface_elevation_m_sorted[index])
        solution_status[index] = str(node_solution_status[node_index])
    refractor = _compute_refractor_elevation(surface, thickness)
    status = _classify_weathering_status(
        solution_status=solution_status,
        half_intercept_time_s=half,
        thickness_m=thickness,
        surface_elevation_m=surface,
        refractor_elevation_m=refractor,
        max_weathering_thickness_m=max_weathering_thickness_m,
        missing_node_mask=missing,
    )
    inherited = solution_status.astype(str, copy=False)
    inherited_mask = ~np.isin(inherited, ['ok', 'solved'])
    status[inherited_mask] = inherited[inherited_mask]
    return (
        np.ascontiguousarray(half, dtype=np.float64),
        np.ascontiguousarray(thickness, dtype=np.float64),
        np.ascontiguousarray(refractor, dtype=np.float64),
        status,
    )


def _map_trace_table_from_endpoints(
    *,
    endpoint_key_sorted: np.ndarray,
    endpoint_key: np.ndarray,
    node_id_sorted: np.ndarray,
    node_pos: dict[int, int],
    endpoint_half_intercept_time_s: np.ndarray,
    endpoint_weathering_thickness_m: np.ndarray,
    endpoint_solution_status: np.ndarray,
    surface_elevation_m_sorted: np.ndarray,
    max_weathering_thickness_m: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key_to_index = _endpoint_key_to_index(endpoint_key)
    n_traces = int(endpoint_key_sorted.shape[0])
    half = np.full(n_traces, np.nan, dtype=np.float64)
    thickness = np.full(n_traces, np.nan, dtype=np.float64)
    solution_status = np.full(n_traces, 'missing_endpoint', dtype=_STATUS_DTYPE)
    missing = np.zeros(n_traces, dtype=bool)
    for index, raw_key in enumerate(endpoint_key_sorted.tolist()):
        if int(node_id_sorted[index]) not in node_pos:
            missing[index] = True
            continue
        endpoint_index = key_to_index.get(str(raw_key))
        if endpoint_index is None:
            missing[index] = True
            continue
        half[index] = float(endpoint_half_intercept_time_s[endpoint_index])
        thickness[index] = float(endpoint_weathering_thickness_m[endpoint_index])
        solution_status[index] = str(endpoint_solution_status[endpoint_index])
    surface = np.asarray(surface_elevation_m_sorted, dtype=np.float64)
    refractor = _compute_refractor_elevation(surface, thickness)
    status = _classify_weathering_status(
        solution_status=solution_status,
        half_intercept_time_s=half,
        thickness_m=thickness,
        surface_elevation_m=surface,
        refractor_elevation_m=refractor,
        max_weathering_thickness_m=max_weathering_thickness_m,
        missing_node_mask=missing,
    )
    status = _merge_weathering_status(
        status,
        solution_status,
    )
    return (
        np.ascontiguousarray(half, dtype=np.float64),
        np.ascontiguousarray(thickness, dtype=np.float64),
        np.ascontiguousarray(refractor, dtype=np.float64),
        status,
    )


def _endpoint_key_to_index(endpoint_key: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for index, raw_key in enumerate(endpoint_key.tolist()):
        key = str(raw_key)
        if not key:
            raise RefractionWeatheringThicknessError(
                'endpoint_key contains an empty value'
            )
        if key in out:
            raise RefractionWeatheringThicknessError(
                f'endpoint_key contains duplicate value: {key}'
            )
        out[key] = index
    return out


def _classify_weathering_status(
    *,
    solution_status: np.ndarray,
    half_intercept_time_s: np.ndarray,
    thickness_m: np.ndarray,
    surface_elevation_m: np.ndarray,
    refractor_elevation_m: np.ndarray,
    max_weathering_thickness_m: float | None,
    missing_node_mask: np.ndarray | None = None,
) -> np.ndarray:
    n_items = int(half_intercept_time_s.shape[0])
    status = np.full(n_items, 'ok', dtype=_STATUS_DTYPE)
    missing = (
        np.zeros(n_items, dtype=bool)
        if missing_node_mask is None
        else np.asarray(missing_node_mask, dtype=bool)
    )
    if missing.shape != (n_items,):
        raise RefractionWeatheringThicknessError('missing_node_mask shape mismatch')

    solution = np.asarray(solution_status).astype(str, copy=False)
    half = np.asarray(half_intercept_time_s, dtype=np.float64)
    thickness = np.asarray(thickness_m, dtype=np.float64)
    surface = np.asarray(surface_elevation_m, dtype=np.float64)
    refractor = np.asarray(refractor_elevation_m, dtype=np.float64)

    inactive = solution == 'inactive'
    invalid_half = (~np.isfinite(half)) | (half < 0.0)
    invalid_half |= (solution == 'invalid_solution') | (solution == 'missing_solution')
    negative_thickness = np.isfinite(thickness) & (thickness < 0.0)
    exceeds_max = np.zeros(n_items, dtype=bool)
    if max_weathering_thickness_m is not None:
        exceeds_max = np.isfinite(thickness) & (thickness > max_weathering_thickness_m)
    invalid_surface = ~np.isfinite(surface)
    invalid_refractor = ~np.isfinite(refractor)
    low_fold = solution == 'low_fold'
    clipped_upper = solution == 'clipped_upper'
    clipped_lower = solution == 'clipped_lower'
    zero_thickness = np.isfinite(thickness) & (
        np.abs(thickness) <= _ZERO_THICKNESS_ATOL_M
    )

    # Assign from lowest to highest priority so stronger statuses cannot be
    # hidden by inherited solver statuses such as low_fold or clipped_*.
    status[zero_thickness] = 'zero_thickness'
    status[clipped_lower] = 'clipped_half_intercept_lower'
    status[clipped_upper] = 'clipped_half_intercept_upper'
    status[low_fold] = 'low_fold'
    status[invalid_refractor] = 'invalid_refractor_elevation'
    status[invalid_surface] = 'invalid_surface_elevation'
    status[exceeds_max] = 'exceeds_max_thickness'
    status[negative_thickness] = 'negative_thickness'
    status[invalid_half] = 'invalid_half_intercept'
    status[inactive] = 'inactive'
    status[missing] = 'missing_node'
    return np.ascontiguousarray(status, dtype=_STATUS_DTYPE)


def _merge_weathering_status(base: np.ndarray, inherited: np.ndarray) -> np.ndarray:
    status = np.asarray(base).astype(_STATUS_DTYPE, copy=True)
    inherited_text = np.asarray(inherited).astype(str, copy=False)
    if status.shape != inherited_text.shape:
        raise RefractionWeatheringThicknessError('weathering status shape mismatch')
    for index, raw_value in enumerate(inherited_text.tolist()):
        value = str(raw_value)
        normalized = {
            'clipped_lower': 'clipped_half_intercept_lower',
            'clipped_upper': 'clipped_half_intercept_upper',
            'invalid_solution': 'invalid_half_intercept',
            'missing_solution': 'invalid_half_intercept',
            'missing_endpoint': 'missing_node',
        }.get(value, value)
        if _STATUS_PRIORITY.get(normalized, 0) > _STATUS_PRIORITY.get(
            str(status[index]),
            0,
        ):
            status[index] = normalized
    return np.ascontiguousarray(status, dtype=_STATUS_DTYPE)


def _compute_refractor_elevation(
    surface_elevation_m: np.ndarray,
    weathering_thickness_m: np.ndarray,
) -> np.ndarray:
    surface = np.asarray(surface_elevation_m, dtype=np.float64)
    thickness = np.asarray(weathering_thickness_m, dtype=np.float64)
    if surface.shape != thickness.shape:
        raise RefractionWeatheringThicknessError(
            'surface and thickness shape mismatch'
        )
    out = np.full(surface.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(surface) & np.isfinite(thickness) & (thickness >= 0.0)
    out[valid] = surface[valid] - thickness[valid]
    return np.ascontiguousarray(out, dtype=np.float64)


def _build_qc(
    *,
    velocity: _VelocityContext,
    data: _ValidatedHalfIntercept,
    node_weathering_thickness_m: np.ndarray,
    node_refractor_elevation_m: np.ndarray,
    node_weathering_status: np.ndarray,
    source_weathering_thickness_m: np.ndarray,
    source_weathering_status: np.ndarray,
    receiver_weathering_thickness_m: np.ndarray,
    receiver_weathering_status: np.ndarray,
    upstream_qc: dict[str, Any],
) -> dict[str, Any]:
    ok_nodes = node_weathering_status == 'ok'
    finite_ok_thickness = node_weathering_thickness_m[
        ok_nodes & np.isfinite(node_weathering_thickness_m)
    ]
    finite_ok_refractor = node_refractor_elevation_m[
        ok_nodes & np.isfinite(node_refractor_elevation_m)
    ]
    ok_sources = source_weathering_status == 'ok'
    ok_receivers = receiver_weathering_status == 'ok'
    source_ok_thickness = source_weathering_thickness_m[
        ok_sources & np.isfinite(source_weathering_thickness_m)
    ]
    receiver_ok_thickness = receiver_weathering_thickness_m[
        ok_receivers & np.isfinite(receiver_weathering_thickness_m)
    ]
    clipped_count = int(
        np.count_nonzero(
            (node_weathering_status == 'clipped_half_intercept_lower')
            | (node_weathering_status == 'clipped_half_intercept_upper')
        )
    )
    qc: dict[str, Any] = {
        'method': 'gli_variable_thickness',
        'bedrock_velocity_mode': velocity.mode,
        'bedrock_velocity_m_s': float(velocity.bedrock_velocity_m_s),
        'bedrock_slowness_s_per_m': float(velocity.bedrock_slowness_s_per_m),
        'weathering_velocity_m_s': float(velocity.weathering_velocity_m_s),
        'n_traces': int(data.n_traces),
        'n_valid_observations': int(
            np.count_nonzero(data.valid_observation_mask_sorted)
        ),
        'n_used_observations': int(np.count_nonzero(data.used_observation_mask_sorted)),
        'n_nodes': int(data.n_nodes),
        'n_active_nodes': int(np.count_nonzero(data.node_solution_status != 'inactive')),
        'n_inactive_nodes': int(np.count_nonzero(data.node_solution_status == 'inactive')),
        'n_source_endpoints': int(data.source_endpoint_key.shape[0]),
        'n_receiver_endpoints': int(data.receiver_endpoint_key.shape[0]),
        'weathering_thickness_min_m': _json_stat(finite_ok_thickness, 'min'),
        'weathering_thickness_max_m': _json_stat(finite_ok_thickness, 'max'),
        'weathering_thickness_median_m': _json_stat(finite_ok_thickness, 'median'),
        'weathering_thickness_p95_m': _json_stat(finite_ok_thickness, 'p95'),
        'source_weathering_thickness_median_m': _json_stat(
            source_ok_thickness,
            'median',
        ),
        'receiver_weathering_thickness_median_m': _json_stat(
            receiver_ok_thickness,
            'median',
        ),
        'refractor_elevation_min_m': _json_stat(finite_ok_refractor, 'min'),
        'refractor_elevation_max_m': _json_stat(finite_ok_refractor, 'max'),
        'refractor_elevation_median_m': _json_stat(finite_ok_refractor, 'median'),
        'max_weathering_thickness_m': _json_optional_float(
            velocity.max_weathering_thickness_m
        ),
        'exceeds_max_thickness_count': int(
            np.count_nonzero(node_weathering_status == 'exceeds_max_thickness')
        ),
        'inactive_node_count': int(
            np.count_nonzero(node_weathering_status == 'inactive')
        ),
        'low_fold_node_count': int(
            np.count_nonzero(node_weathering_status == 'low_fold')
        ),
        'clipped_half_intercept_node_count': clipped_count,
        'invalid_half_intercept_node_count': int(
            np.count_nonzero(node_weathering_status == 'invalid_half_intercept')
        ),
        'negative_thickness_node_count': int(
            np.count_nonzero(node_weathering_status == 'negative_thickness')
        ),
        'zero_thickness_node_count': int(
            np.count_nonzero(node_weathering_status == 'zero_thickness')
        ),
        'invalid_surface_elevation_count': int(
            np.count_nonzero(node_weathering_status == 'invalid_surface_elevation')
        ),
        'invalid_refractor_elevation_count': int(
            np.count_nonzero(node_weathering_status == 'invalid_refractor_elevation')
        ),
        'weathering_status_counts': _status_counts(node_weathering_status),
        'source_weathering_status_counts': _status_counts(source_weathering_status),
        'receiver_weathering_status_counts': _status_counts(receiver_weathering_status),
        'thickness_formula': _FORMULA_TEXT,
    }
    _copy_cell_threshold_qc(qc, upstream_qc)
    _assert_json_safe(qc)
    return qc


def _low_fold_cell_id_from_qc(qc: dict[str, Any]) -> np.ndarray:
    values = qc.get('low_fold_cell_id', [])
    return _coerce_1d_integer(
        np.asarray(values, dtype=np.int64),
        name='low_fold_cell_id',
    )


def _copy_cell_threshold_qc(payload: dict[str, Any], upstream: dict[str, Any]) -> None:
    for key in _CELL_THRESHOLD_QC_KEYS:
        if key in upstream:
            payload[key] = upstream[key]
    layer_qc = upstream.get('layers')
    if isinstance(layer_qc, dict):
        payload['layers'] = layer_qc


def _required(owner: object, field: str, *, allow_none: bool = False) -> object:
    try:
        value = getattr(owner, field)
    except AttributeError as exc:
        raise RefractionWeatheringThicknessError(f'{field} is required') from exc
    if value is None and not allow_none:
        raise RefractionWeatheringThicknessError(f'{field} is required')
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
        error_type=RefractionWeatheringThicknessError,
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
        error_type=RefractionWeatheringThicknessError,
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
        error_type=RefractionWeatheringThicknessError,
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
        error_type=RefractionWeatheringThicknessError,
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
        raise RefractionWeatheringThicknessError(f'{name} must be finite and positive')
    out = _coerce_finite_float(
        value,
        name=name,
        error_type=RefractionWeatheringThicknessError,
    )
    if not np.isfinite(out):
        raise RefractionWeatheringThicknessError(f'{name} must be finite')
    if out <= 0.0:
        raise RefractionWeatheringThicknessError(f'{name} must be positive')
    return out


def _optional_positive_finite(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _positive_finite(value, name=name)


def _validate_velocity_mode(
    value: object,
) -> Literal['solve_global', 'fixed_global', 'solve_cell']:
    if value == 'solve_global':
        return 'solve_global'
    if value == 'fixed_global':
        return 'fixed_global'
    if value == 'solve_cell':
        return 'solve_cell'
    raise RefractionWeatheringThicknessError(
        'bedrock_velocity_mode must be solve_global, fixed_global, or solve_cell'
    )


def _close_velocity(left: float, right: float) -> bool:
    tolerance = max(1.0e-6, max(abs(left), abs(right)) * _VELOCITY_RTOL)
    return abs(left - right) <= tolerance


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


def _status_counts(values: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in values.tolist():
        key = str(raw)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _json_optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _node_rows(result: RefractionWeatheringThicknessResult) -> list[dict[str, Any]]:
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
                'half_intercept_time_ms': _csv_float(
                    result.node_half_intercept_time_ms[index]
                ),
                'weathering_thickness_m': _csv_float(
                    result.node_weathering_thickness_m[index]
                ),
                'refractor_elevation_m': _csv_float(
                    result.node_refractor_elevation_m[index]
                ),
                'solution_status': str(result.node_solution_status[index]),
                'weathering_status': str(result.node_weathering_status[index]),
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
                'source_surface_elevation_m': _csv_float(
                    result.source_surface_elevation_m[index]
                ),
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
                'receiver_surface_elevation_m': _csv_float(
                    result.receiver_surface_elevation_m[index]
                ),
                'receiver_half_intercept_time_ms': _csv_float(
                    result.receiver_half_intercept_time_s[index] * 1000.0
                ),
                'receiver_weathering_thickness_m': _csv_float(
                    result.receiver_weathering_thickness_m[index]
                ),
                'receiver_refractor_elevation_m': _csv_float(
                    result.receiver_refractor_elevation_m[index]
                ),
                'receiver_weathering_status': str(
                    result.receiver_weathering_status[index]
                ),
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
    'compute_weathering_thickness_from_half_intercept_time',
    'compute_weathering_thickness_scalar',
    'estimate_weathering_thickness_from_first_breaks',
    'write_refraction_weathering_thickness_artifacts',
]
