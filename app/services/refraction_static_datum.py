"""Datum-related refraction static components for GLI statics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from app.api.schemas import (
    RefractionStaticApplyOptions,
    RefractionStaticApplyRequest,
    RefractionStaticDatumRequest,
)
from app.core.state import AppState
from app.services.job_artifact_refs import resolve_job_artifact_path
from app.services.refraction_static_first_layer import (
    validate_resolved_first_layer_velocity_match,
)
from app.services.refraction_static_status import LOCAL_V2_STATUS_VALUES
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
    RefractionWeatheringReplacementStaticsResult,
    ResolvedRefractionFirstLayer,
)
from app.services.refraction_static_weathering_replacement import (
    compute_weathering_replacement_statics_from_first_breaks,
)

REFRACTION_DATUM_STATICS_QC_JSON_NAME = 'refraction_datum_statics_qc.json'
REFRACTION_DATUM_NODES_CSV_NAME = 'refraction_datum_nodes.csv'
REFRACTION_DATUM_SOURCES_CSV_NAME = 'refraction_datum_sources.csv'
REFRACTION_DATUM_RECEIVERS_CSV_NAME = 'refraction_datum_receivers.csv'
REFRACTION_DATUM_TRACE_PREVIEW_CSV_NAME = 'refraction_datum_trace_preview.csv'

_STATUS_DTYPE = '<U48'
_ENDPOINT_KEY_DTYPE = object
_SLOWNESS_RTOL = 1.0e-6
_ZERO_SHIFT_ATOL_S = 1.0e-12
_SIGN_CONVENTION_TEXT = 'corrected(t) = raw(t - shift_s)'
_FORMULA_TO_FLOATING_TEXT = 'weathering + floating_elevation'
_FORMULA_TO_FLAT_TEXT = '2*ED - (EFDs + EFDr) over vb'
_FROM_ARTIFACT_MESSAGE = (
    'floating datum artifact must be provided or resolvable for '
    'floating_datum_mode=from_artifact'
)
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
    'inactive': 2,
    'exceeds_max_abs_shift': 3,
    'invalid_datum_shift': 4,
    'flat_datum_below_refractor': 5,
    'floating_datum_below_refractor': 6,
    'invalid_weathering_replacement': 7,
    'invalid_nonfinite_input': 8,
    'invalid_velocity_order': 9,
    'outside_refractor_cell_grid': 10,
    'inactive_v2_cell': 11,
    'low_fold_v2_cell': 12,
    'invalid_local_v2': 13,
    'v2_not_greater_than_v1': 14,
    'invalid_flat_datum_elevation': 15,
    'invalid_floating_datum_elevation': 16,
    'invalid_surface_elevation': 17,
    'invalid_bedrock_velocity': 18,
    'missing_endpoint': 19,
    'missing_node': 20,
}

_INVALID_TRACE_STATUSES = {
    'missing_node',
    'missing_endpoint',
    'invalid_bedrock_velocity',
    'invalid_surface_elevation',
    'invalid_floating_datum_elevation',
    'invalid_flat_datum_elevation',
    'invalid_weathering_replacement',
    'invalid_nonfinite_input',
    'invalid_velocity_order',
    'outside_refractor_cell_grid',
    'inactive_v2_cell',
    'low_fold_v2_cell',
    'invalid_local_v2',
    'v2_not_greater_than_v1',
    'floating_datum_below_refractor',
    'flat_datum_below_refractor',
    'invalid_datum_shift',
    'exceeds_max_abs_shift',
    'inactive',
}

_UPSTREAM_REPLACEMENT_STATUS_TO_DATUM_STATUS = {
    'missing_node': 'missing_node',
    'missing_endpoint': 'missing_endpoint',
    'invalid_velocity': 'invalid_bedrock_velocity',
    'inactive': 'inactive',
    'exceeds_max_abs_shift': 'invalid_weathering_replacement',
    'invalid_shift': 'invalid_weathering_replacement',
    'negative_weathering_thickness': 'invalid_weathering_replacement',
    'invalid_weathering_thickness': 'invalid_weathering_replacement',
    'invalid_nonfinite_input': 'invalid_nonfinite_input',
    'invalid_velocity_order': 'invalid_velocity_order',
    'outside_refractor_cell_grid': 'outside_refractor_cell_grid',
    'inactive_v2_cell': 'inactive_v2_cell',
    'low_fold_v2_cell': 'low_fold_v2_cell',
    'invalid_local_v2': 'invalid_local_v2',
    'v2_not_greater_than_v1': 'v2_not_greater_than_v1',
}
_UPSTREAM_REPLACEMENT_NON_INVALID_STATUSES = {
    'ok',
    'not_observed',
    'clipped_half_intercept_lower',
    'clipped_half_intercept_upper',
    'low_fold',
    'exceeds_max_thickness',
}

_NODE_COLUMNS = (
    'node_id',
    'node_kind',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'weathering_thickness_m',
    'refractor_elevation_m',
    'floating_datum_elevation_m',
    'datum_status',
    'weathering_status',
    'pick_count',
    'used_pick_count',
    'residual_rms_ms',
)
_SOURCE_COLUMNS = (
    'source_endpoint_key',
    'source_id',
    'source_node_id',
    'source_x_m',
    'source_y_m',
    'source_surface_elevation_m',
    'source_weathering_thickness_m',
    'source_refractor_elevation_m',
    'source_floating_datum_elevation_m',
    'source_weathering_replacement_shift_ms',
    'source_floating_datum_elevation_shift_ms',
    'source_flat_datum_shift_ms',
    'source_refraction_shift_ms',
    'source_datum_status',
)
_RECEIVER_COLUMNS = (
    'receiver_endpoint_key',
    'receiver_id',
    'receiver_node_id',
    'receiver_x_m',
    'receiver_y_m',
    'receiver_surface_elevation_m',
    'receiver_weathering_thickness_m',
    'receiver_refractor_elevation_m',
    'receiver_floating_datum_elevation_m',
    'receiver_weathering_replacement_shift_ms',
    'receiver_floating_datum_elevation_shift_ms',
    'receiver_flat_datum_shift_ms',
    'receiver_refraction_shift_ms',
    'receiver_datum_status',
)
_TRACE_PREVIEW_COLUMNS = (
    'sorted_trace_index',
    'valid_observation',
    'used_observation',
    'trace_static_valid',
    'source_node_id',
    'receiver_node_id',
    'source_surface_elevation_m',
    'receiver_surface_elevation_m',
    'source_floating_datum_elevation_m',
    'receiver_floating_datum_elevation_m',
    'weathering_replacement_trace_shift_ms',
    'floating_datum_elevation_shift_ms',
    'flat_datum_shift_ms',
    'refraction_trace_shift_ms',
    'trace_static_status',
    'estimated_first_break_time_ms',
    'first_break_residual_ms',
)

_FIELD_DISABLED_STATUS = 'not_enabled'
_FIELD_NOT_APPLICABLE_STATUS = 'not_applicable'
_FIELD_TOTAL_VALID_STATUSES = frozenset(
    {'ok', _FIELD_DISABLED_STATUS, _FIELD_NOT_APPLICABLE_STATUS}
)


class RefractionDatumStaticsError(ValueError):
    """Raised when datum refraction static outputs cannot be built."""


@dataclass(frozen=True)
class _VelocityContext:
    mode: Literal['solve_global', 'fixed_global', 'solve_cell']
    bedrock_slowness_s_per_m: float
    bedrock_velocity_m_s: float
    weathering_velocity_m_s: float
    replacement_slowness_delta_s_per_m: float


@dataclass(frozen=True)
class _ValidatedReplacement:
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
    node_weathering_replacement_shift_s: np.ndarray
    node_static_status: np.ndarray
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
    source_weathering_replacement_shift_s: np.ndarray
    source_static_status: np.ndarray
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
    receiver_weathering_replacement_shift_s: np.ndarray
    receiver_static_status: np.ndarray
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
    source_surface_elevation_m_sorted: np.ndarray
    receiver_surface_elevation_m_sorted: np.ndarray
    source_weathering_thickness_m_sorted: np.ndarray
    receiver_weathering_thickness_m_sorted: np.ndarray
    source_refractor_elevation_m_sorted: np.ndarray
    receiver_refractor_elevation_m_sorted: np.ndarray
    source_half_intercept_time_s_sorted: np.ndarray
    receiver_half_intercept_time_s_sorted: np.ndarray
    source_weathering_replacement_shift_s_sorted: np.ndarray
    receiver_weathering_replacement_shift_s_sorted: np.ndarray
    source_missing_node_sorted: np.ndarray
    receiver_missing_node_sorted: np.ndarray
    source_missing_endpoint_sorted: np.ndarray
    receiver_missing_endpoint_sorted: np.ndarray
    weathering_replacement_trace_shift_s_sorted: np.ndarray
    source_static_status_sorted: np.ndarray
    receiver_static_status_sorted: np.ndarray
    source_v2_cell_id_sorted: np.ndarray | None
    source_v2_m_s_sorted: np.ndarray | None
    source_v2_status_sorted: np.ndarray | None
    receiver_v2_cell_id_sorted: np.ndarray | None
    receiver_v2_m_s_sorted: np.ndarray | None
    receiver_v2_status_sorted: np.ndarray | None
    trace_static_status_sorted: np.ndarray
    trace_static_valid_mask_sorted: np.ndarray
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
    row_layer_kind: np.ndarray | None
    row_layer_index: np.ndarray | None
    row_source_endpoint_key: np.ndarray | None
    row_receiver_endpoint_key: np.ndarray | None
    row_rejection_reason: np.ndarray | None
    row_velocity_m_s: np.ndarray | None

    @property
    def n_nodes(self) -> int:
        return int(self.node_id.shape[0])

    @property
    def n_traces(self) -> int:
        return int(self.sorted_trace_index.shape[0])


@dataclass(frozen=True)
class _FloatingDatumModel:
    node_floating_datum_elevation_m: np.ndarray
    source_floating_datum_elevation_m: np.ndarray
    receiver_floating_datum_elevation_m: np.ndarray
    source_floating_datum_elevation_m_sorted: np.ndarray
    receiver_floating_datum_elevation_m_sorted: np.ndarray


def compute_datum_refraction_statics_from_first_breaks(
    *,
    req: RefractionStaticApplyRequest,
    state: AppState,
    job_dir: Path | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> RefractionDatumStaticsResult:
    """Run GLI weathering replacement, then compose datum refraction statics."""
    try:
        replacement_kwargs: dict[str, Any] = {
            'req': req,
            'state': state,
            'job_dir': job_dir,
        }
        if resolved_first_layer is not None:
            replacement_kwargs['resolved_first_layer'] = resolved_first_layer
        replacement = compute_weathering_replacement_statics_from_first_breaks(
            **replacement_kwargs
        )
        return build_refraction_datum_statics(
            weathering_replacement_result=replacement,
            datum=req.datum,
            apply_options=req.apply,
            job_dir=job_dir,
            state=state,
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            resolved_first_layer=resolved_first_layer,
        )
    except RefractionDatumStaticsError:
        raise
    except ValueError as exc:
        raise RefractionDatumStaticsError(str(exc)) from exc


def build_refraction_datum_statics(
    *,
    weathering_replacement_result: RefractionWeatheringReplacementStaticsResult,
    datum: RefractionStaticDatumRequest,
    apply_options: RefractionStaticApplyOptions | None = None,
    job_dir: Path | None = None,
    state: AppState | None = None,
    file_id: str | None = None,
    key1_byte: int | None = None,
    key2_byte: int | None = None,
    floating_datum_artifact_path: Path | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> RefractionDatumStaticsResult:
    """Compose weathering replacement, floating datum, and flat datum shifts."""
    data = _validate_replacement_result(weathering_replacement_result)
    velocity = _validate_velocity_context(
        weathering_replacement_result,
        resolved_first_layer=resolved_first_layer,
    )
    datum_req = _validate_datum_request(datum)
    max_abs_shift_ms = _resolve_max_abs_shift_ms(apply_options)
    artifact_path = _resolve_floating_datum_artifact_path(
        datum=datum_req,
        state=state,
        explicit_path=floating_datum_artifact_path,
        file_id=file_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )

    node_pos = {int(node): index for index, node in enumerate(data.node_id.tolist())}
    floating = _build_floating_datum_model(
        data=data,
        datum=datum_req,
        node_pos=node_pos,
        artifact_path=artifact_path,
    )

    floating_active = datum_req.mode in {'floating_only', 'floating_and_flat'}
    flat_active = datum_req.mode in {'flat_only', 'floating_and_flat'}
    flat_datum = _resolve_flat_datum(datum_req, flat_active=flat_active)

    source_floating_shift = _endpoint_floating_shift(
        surface_elevation_m=data.source_surface_elevation_m,
        floating_datum_elevation_m=floating.source_floating_datum_elevation_m,
        bedrock_velocity_m_s=velocity.bedrock_velocity_m_s,
        active=floating_active,
    )
    receiver_floating_shift = _endpoint_floating_shift(
        surface_elevation_m=data.receiver_surface_elevation_m,
        floating_datum_elevation_m=floating.receiver_floating_datum_elevation_m,
        bedrock_velocity_m_s=velocity.bedrock_velocity_m_s,
        active=floating_active,
    )
    source_flat_shift = _endpoint_flat_shift(
        flat_datum_elevation_m=flat_datum,
        floating_datum_elevation_m=floating.source_floating_datum_elevation_m,
        bedrock_velocity_m_s=velocity.bedrock_velocity_m_s,
        active=flat_active,
        shape=data.source_surface_elevation_m.shape,
    )
    receiver_flat_shift = _endpoint_flat_shift(
        flat_datum_elevation_m=flat_datum,
        floating_datum_elevation_m=floating.receiver_floating_datum_elevation_m,
        bedrock_velocity_m_s=velocity.bedrock_velocity_m_s,
        active=flat_active,
        shape=data.receiver_surface_elevation_m.shape,
    )

    source_status = _classify_endpoint_status(
        inherited_status=data.source_static_status,
        surface_elevation_m=data.source_surface_elevation_m,
        floating_datum_elevation_m=floating.source_floating_datum_elevation_m,
        refractor_elevation_m=data.source_refractor_elevation_m,
        weathering_replacement_shift_s=data.source_weathering_replacement_shift_s,
        flat_datum_elevation_m=flat_datum,
        datum=datum_req,
        max_abs_shift_ms=max_abs_shift_ms,
        refraction_shift_s=(
            data.source_weathering_replacement_shift_s
            + source_floating_shift
            + source_flat_shift
        ),
    )
    receiver_status = _classify_endpoint_status(
        inherited_status=data.receiver_static_status,
        surface_elevation_m=data.receiver_surface_elevation_m,
        floating_datum_elevation_m=floating.receiver_floating_datum_elevation_m,
        refractor_elevation_m=data.receiver_refractor_elevation_m,
        weathering_replacement_shift_s=data.receiver_weathering_replacement_shift_s,
        flat_datum_elevation_m=flat_datum,
        datum=datum_req,
        max_abs_shift_ms=max_abs_shift_ms,
        refraction_shift_s=(
            data.receiver_weathering_replacement_shift_s
            + receiver_floating_shift
            + receiver_flat_shift
        ),
    )
    source_refraction_shift = _compose_endpoint_shift(
        weathering_shift_s=data.source_weathering_replacement_shift_s,
        floating_shift_s=source_floating_shift,
        flat_shift_s=source_flat_shift,
        status=source_status,
    )
    receiver_refraction_shift = _compose_endpoint_shift(
        weathering_shift_s=data.receiver_weathering_replacement_shift_s,
        floating_shift_s=receiver_floating_shift,
        flat_shift_s=receiver_flat_shift,
        status=receiver_status,
    )
    source_floating_shift_sorted, _, _ = _map_trace_endpoint_values(
        node_id_sorted=data.source_node_id_sorted,
        endpoint_key_sorted=data.source_endpoint_key_sorted,
        endpoint_key=data.source_endpoint_key,
        endpoint_node_id=data.source_node_id,
        endpoint_values=source_floating_shift,
        node_pos=node_pos,
        name='source_floating_datum_elevation_shift_s_sorted',
    )
    receiver_floating_shift_sorted, _, _ = _map_trace_endpoint_values(
        node_id_sorted=data.receiver_node_id_sorted,
        endpoint_key_sorted=data.receiver_endpoint_key_sorted,
        endpoint_key=data.receiver_endpoint_key,
        endpoint_node_id=data.receiver_node_id,
        endpoint_values=receiver_floating_shift,
        node_pos=node_pos,
        name='receiver_floating_datum_elevation_shift_s_sorted',
    )
    source_flat_shift_sorted, _, _ = _map_trace_endpoint_values(
        node_id_sorted=data.source_node_id_sorted,
        endpoint_key_sorted=data.source_endpoint_key_sorted,
        endpoint_key=data.source_endpoint_key,
        endpoint_node_id=data.source_node_id,
        endpoint_values=source_flat_shift,
        node_pos=node_pos,
        name='source_flat_datum_shift_s_sorted',
    )
    receiver_flat_shift_sorted, _, _ = _map_trace_endpoint_values(
        node_id_sorted=data.receiver_node_id_sorted,
        endpoint_key_sorted=data.receiver_endpoint_key_sorted,
        endpoint_key=data.receiver_endpoint_key,
        endpoint_node_id=data.receiver_node_id,
        endpoint_values=receiver_flat_shift,
        node_pos=node_pos,
        name='receiver_flat_datum_shift_s_sorted',
    )
    source_refraction_shift_sorted, _, _ = _map_trace_endpoint_values(
        node_id_sorted=data.source_node_id_sorted,
        endpoint_key_sorted=data.source_endpoint_key_sorted,
        endpoint_key=data.source_endpoint_key,
        endpoint_node_id=data.source_node_id,
        endpoint_values=source_refraction_shift,
        node_pos=node_pos,
        name='source_refraction_shift_s_sorted',
    )
    receiver_refraction_shift_sorted, _, _ = _map_trace_endpoint_values(
        node_id_sorted=data.receiver_node_id_sorted,
        endpoint_key_sorted=data.receiver_endpoint_key_sorted,
        endpoint_key=data.receiver_endpoint_key,
        endpoint_node_id=data.receiver_node_id,
        endpoint_values=receiver_refraction_shift,
        node_pos=node_pos,
        name='receiver_refraction_shift_s_sorted',
    )

    node_status = _classify_node_status(
        inherited_status=data.node_static_status,
        surface_elevation_m=data.node_surface_elevation_m,
        floating_datum_elevation_m=floating.node_floating_datum_elevation_m,
        refractor_elevation_m=data.node_refractor_elevation_m,
        weathering_replacement_shift_s=data.node_weathering_replacement_shift_s,
        flat_datum_elevation_m=flat_datum,
        datum=datum_req,
    )

    trace_floating_shift = _trace_floating_shift(
        source_surface_elevation_m=data.source_surface_elevation_m_sorted,
        receiver_surface_elevation_m=data.receiver_surface_elevation_m_sorted,
        source_floating_datum_elevation_m=(
            floating.source_floating_datum_elevation_m_sorted
        ),
        receiver_floating_datum_elevation_m=(
            floating.receiver_floating_datum_elevation_m_sorted
        ),
        bedrock_velocity_m_s=velocity.bedrock_velocity_m_s,
        active=floating_active,
    )
    trace_flat_shift = _trace_flat_shift(
        flat_datum_elevation_m=flat_datum,
        source_floating_datum_elevation_m=(
            floating.source_floating_datum_elevation_m_sorted
        ),
        receiver_floating_datum_elevation_m=(
            floating.receiver_floating_datum_elevation_m_sorted
        ),
        bedrock_velocity_m_s=velocity.bedrock_velocity_m_s,
        active=flat_active,
        shape=data.sorted_trace_index.shape,
    )
    refraction_trace_shift = compose_refraction_trace_shift_s(
        weathering_replacement_trace_shift_s=(
            data.weathering_replacement_trace_shift_s_sorted
        ),
        floating_datum_elevation_shift_s=trace_floating_shift,
        flat_datum_shift_s=trace_flat_shift,
    )
    trace_status = _classify_trace_status(
        data=data,
        datum=datum_req,
        floating=floating,
        flat_datum_elevation_m=flat_datum,
        floating_shift_s=trace_floating_shift,
        flat_shift_s=trace_flat_shift,
        refraction_shift_s=refraction_trace_shift,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    trace_valid = _trace_valid_mask(trace_status, refraction_trace_shift)
    refraction_trace_shift = np.ascontiguousarray(
        np.where(_trace_nan_mask(trace_status), np.nan, refraction_trace_shift),
        dtype=np.float64,
    )
    trace_valid = _trace_valid_mask(trace_status, refraction_trace_shift)

    qc = _build_qc(
        velocity=velocity,
        datum=datum_req,
        data=data,
        floating=floating,
        flat_datum_elevation_m=flat_datum,
        weathering_replacement_trace_shift_s=(
            data.weathering_replacement_trace_shift_s_sorted
        ),
        floating_datum_shift_s=trace_floating_shift,
        flat_datum_shift_s=trace_flat_shift,
        refraction_trace_shift_s=refraction_trace_shift,
        trace_status=trace_status,
        trace_valid_mask=trace_valid,
        node_status=node_status,
        source_status=source_status,
        receiver_status=receiver_status,
        max_abs_shift_ms=max_abs_shift_ms,
        upstream_qc=getattr(weathering_replacement_result, 'qc', {}),
    )

    result = RefractionDatumStaticsResult(
        bedrock_velocity_mode=velocity.mode,
        bedrock_slowness_s_per_m=velocity.bedrock_slowness_s_per_m,
        bedrock_velocity_m_s=velocity.bedrock_velocity_m_s,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        replacement_slowness_delta_s_per_m=(
            velocity.replacement_slowness_delta_s_per_m
        ),
        datum_mode=datum_req.mode,
        floating_datum_mode=datum_req.floating_datum_mode,
        flat_datum_elevation_m=flat_datum,
        node_id=data.node_id,
        node_x_m=data.node_x_m,
        node_y_m=data.node_y_m,
        node_surface_elevation_m=data.node_surface_elevation_m,
        node_kind=data.node_kind,
        node_weathering_thickness_m=data.node_weathering_thickness_m,
        node_refractor_elevation_m=data.node_refractor_elevation_m,
        node_half_intercept_time_s=data.node_half_intercept_time_s,
        node_weathering_replacement_shift_s=data.node_weathering_replacement_shift_s,
        node_floating_datum_elevation_m=floating.node_floating_datum_elevation_m,
        node_solution_status=data.node_solution_status,
        node_datum_status=node_status,
        node_weathering_status=data.node_weathering_status,
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
        source_floating_datum_elevation_m=(
            floating.source_floating_datum_elevation_m
        ),
        source_weathering_replacement_shift_s=(
            data.source_weathering_replacement_shift_s
        ),
        source_floating_datum_elevation_shift_s=source_floating_shift,
        source_flat_datum_shift_s=source_flat_shift,
        source_refraction_shift_s=source_refraction_shift,
        source_datum_status=source_status,
        receiver_endpoint_key=data.receiver_endpoint_key,
        receiver_id=data.receiver_id,
        receiver_node_id=data.receiver_node_id,
        receiver_x_m=data.receiver_x_m,
        receiver_y_m=data.receiver_y_m,
        receiver_surface_elevation_m=data.receiver_surface_elevation_m,
        receiver_half_intercept_time_s=data.receiver_half_intercept_time_s,
        receiver_weathering_thickness_m=data.receiver_weathering_thickness_m,
        receiver_refractor_elevation_m=data.receiver_refractor_elevation_m,
        receiver_floating_datum_elevation_m=(
            floating.receiver_floating_datum_elevation_m
        ),
        receiver_weathering_replacement_shift_s=(
            data.receiver_weathering_replacement_shift_s
        ),
        receiver_floating_datum_elevation_shift_s=receiver_floating_shift,
        receiver_flat_datum_shift_s=receiver_flat_shift,
        receiver_refraction_shift_s=receiver_refraction_shift,
        receiver_datum_status=receiver_status,
        sorted_trace_index=data.sorted_trace_index,
        valid_observation_mask_sorted=data.valid_observation_mask_sorted,
        used_observation_mask_sorted=data.used_observation_mask_sorted,
        source_node_id_sorted=data.source_node_id_sorted,
        receiver_node_id_sorted=data.receiver_node_id_sorted,
        source_surface_elevation_m_sorted=data.source_surface_elevation_m_sorted,
        receiver_surface_elevation_m_sorted=data.receiver_surface_elevation_m_sorted,
        source_floating_datum_elevation_m_sorted=(
            floating.source_floating_datum_elevation_m_sorted
        ),
        receiver_floating_datum_elevation_m_sorted=(
            floating.receiver_floating_datum_elevation_m_sorted
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
        source_half_intercept_time_s_sorted=(
            data.source_half_intercept_time_s_sorted
        ),
        receiver_half_intercept_time_s_sorted=(
            data.receiver_half_intercept_time_s_sorted
        ),
        source_weathering_replacement_shift_s_sorted=(
            data.source_weathering_replacement_shift_s_sorted
        ),
        receiver_weathering_replacement_shift_s_sorted=(
            data.receiver_weathering_replacement_shift_s_sorted
        ),
        source_floating_datum_elevation_shift_s_sorted=source_floating_shift_sorted,
        receiver_floating_datum_elevation_shift_s_sorted=(
            receiver_floating_shift_sorted
        ),
        source_flat_datum_shift_s_sorted=source_flat_shift_sorted,
        receiver_flat_datum_shift_s_sorted=receiver_flat_shift_sorted,
        source_refraction_shift_s_sorted=source_refraction_shift_sorted,
        receiver_refraction_shift_s_sorted=receiver_refraction_shift_sorted,
        source_endpoint_key_sorted=data.source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=data.receiver_endpoint_key_sorted,
        weathering_replacement_trace_shift_s_sorted=(
            data.weathering_replacement_trace_shift_s_sorted
        ),
        floating_datum_elevation_shift_s_sorted=trace_floating_shift,
        flat_datum_shift_s_sorted=trace_flat_shift,
        refraction_trace_shift_s_sorted=refraction_trace_shift,
        trace_static_status_sorted=trace_status,
        trace_static_valid_mask_sorted=trace_valid,
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
        node_sh1_weathering_thickness_m=(
            weathering_replacement_result.node_sh1_weathering_thickness_m
        ),
        node_sh2_weathering_thickness_m=(
            weathering_replacement_result.node_sh2_weathering_thickness_m
        ),
        node_sh3_weathering_thickness_m=(
            weathering_replacement_result.node_sh3_weathering_thickness_m
        ),
        source_t2_time_s=weathering_replacement_result.source_t2_time_s,
        source_t3_time_s=weathering_replacement_result.source_t3_time_s,
        source_v3_m_s=weathering_replacement_result.source_v3_m_s,
        source_vsub_m_s=weathering_replacement_result.source_vsub_m_s,
        source_sh1_weathering_thickness_m=(
            weathering_replacement_result.source_sh1_weathering_thickness_m
        ),
        source_sh2_weathering_thickness_m=(
            weathering_replacement_result.source_sh2_weathering_thickness_m
        ),
        source_sh3_weathering_thickness_m=(
            weathering_replacement_result.source_sh3_weathering_thickness_m
        ),
        receiver_t2_time_s=weathering_replacement_result.receiver_t2_time_s,
        receiver_t3_time_s=weathering_replacement_result.receiver_t3_time_s,
        receiver_v3_m_s=weathering_replacement_result.receiver_v3_m_s,
        receiver_vsub_m_s=weathering_replacement_result.receiver_vsub_m_s,
        receiver_sh1_weathering_thickness_m=(
            weathering_replacement_result.receiver_sh1_weathering_thickness_m
        ),
        receiver_sh2_weathering_thickness_m=(
            weathering_replacement_result.receiver_sh2_weathering_thickness_m
        ),
        receiver_sh3_weathering_thickness_m=(
            weathering_replacement_result.receiver_sh3_weathering_thickness_m
        ),
        row_layer_kind=(
            data.row_layer_kind
            if data.row_layer_kind is not None
            else np.full(data.row_trace_index_sorted.shape, 'v2_t1', dtype='<U32')
        ),
        row_layer_index=(
            data.row_layer_index
            if data.row_layer_index is not None
            else np.ones(data.row_trace_index_sorted.shape, dtype=np.int64)
        ),
        row_source_endpoint_key=(
            data.row_source_endpoint_key
            if data.row_source_endpoint_key is not None
            else data.source_endpoint_key_sorted[data.row_trace_index_sorted]
        ),
        row_receiver_endpoint_key=(
            data.row_receiver_endpoint_key
            if data.row_receiver_endpoint_key is not None
            else data.receiver_endpoint_key_sorted[data.row_trace_index_sorted]
        ),
        row_rejection_reason=data.row_rejection_reason,
        row_velocity_m_s=data.row_velocity_m_s,
    )
    if job_dir is not None:
        write_refraction_datum_statics_artifacts(Path(job_dir), result)
    return result


def compute_floating_datum_elevation_shift_s(
    *,
    true_source_elevation_m: np.ndarray,
    true_receiver_elevation_m: np.ndarray,
    floating_source_elevation_m: np.ndarray,
    floating_receiver_elevation_m: np.ndarray,
    bedrock_velocity_m_s: float,
) -> np.ndarray:
    """Compute ``-((ETs - EFDs) + (ETr - EFDr)) / vb`` in seconds."""
    source_surface = _coerce_1d_float(
        true_source_elevation_m,
        name='true_source_elevation_m',
        allow_nonfinite=True,
    )
    shape = source_surface.shape
    receiver_surface = _coerce_1d_float(
        true_receiver_elevation_m,
        name='true_receiver_elevation_m',
        expected_shape=shape,
        allow_nonfinite=True,
    )
    source_floating = _coerce_1d_float(
        floating_source_elevation_m,
        name='floating_source_elevation_m',
        expected_shape=shape,
        allow_nonfinite=True,
    )
    receiver_floating = _coerce_1d_float(
        floating_receiver_elevation_m,
        name='floating_receiver_elevation_m',
        expected_shape=shape,
        allow_nonfinite=True,
    )
    vb = _positive_finite(bedrock_velocity_m_s, name='bedrock_velocity_m_s')
    return np.ascontiguousarray(
        -(
            (source_surface - source_floating)
            + (receiver_surface - receiver_floating)
        )
        / vb,
        dtype=np.float64,
    )


def compute_flat_datum_shift_s(
    *,
    flat_datum_elevation_m: float,
    floating_source_elevation_m: np.ndarray,
    floating_receiver_elevation_m: np.ndarray,
    bedrock_velocity_m_s: float,
) -> np.ndarray:
    """Compute ``(2*ED - (EFDs + EFDr)) / vb`` in seconds."""
    flat_datum = _finite_float(
        flat_datum_elevation_m,
        name='flat_datum_elevation_m',
    )
    source_floating = _coerce_1d_float(
        floating_source_elevation_m,
        name='floating_source_elevation_m',
        allow_nonfinite=True,
    )
    receiver_floating = _coerce_1d_float(
        floating_receiver_elevation_m,
        name='floating_receiver_elevation_m',
        expected_shape=source_floating.shape,
        allow_nonfinite=True,
    )
    vb = _positive_finite(bedrock_velocity_m_s, name='bedrock_velocity_m_s')
    return np.ascontiguousarray(
        (2.0 * flat_datum - (source_floating + receiver_floating)) / vb,
        dtype=np.float64,
    )


def compose_refraction_trace_shift_s(
    *,
    weathering_replacement_trace_shift_s: np.ndarray,
    floating_datum_elevation_shift_s: np.ndarray | None,
    flat_datum_shift_s: np.ndarray | None,
) -> np.ndarray:
    """Compose datum components without clipping or NaN replacement."""
    out = _coerce_1d_float(
        weathering_replacement_trace_shift_s,
        name='weathering_replacement_trace_shift_s',
        allow_nonfinite=True,
    ).copy()
    if floating_datum_elevation_shift_s is not None:
        floating = _coerce_1d_float(
            floating_datum_elevation_shift_s,
            name='floating_datum_elevation_shift_s',
            expected_shape=out.shape,
            allow_nonfinite=True,
        )
        out += floating
    if flat_datum_shift_s is not None:
        flat = _coerce_1d_float(
            flat_datum_shift_s,
            name='flat_datum_shift_s',
            expected_shape=out.shape,
            allow_nonfinite=True,
        )
        out += flat
    return np.ascontiguousarray(out, dtype=np.float64)


def write_refraction_datum_statics_artifacts(
    job_dir: Path,
    result: RefractionDatumStaticsResult,
) -> dict[str, Path]:
    """Write datum-static QC JSON and CSV preview artifacts."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / REFRACTION_DATUM_STATICS_QC_JSON_NAME
    node_path = root / REFRACTION_DATUM_NODES_CSV_NAME
    source_path = root / REFRACTION_DATUM_SOURCES_CSV_NAME
    receiver_path = root / REFRACTION_DATUM_RECEIVERS_CSV_NAME
    trace_path = root / REFRACTION_DATUM_TRACE_PREVIEW_CSV_NAME
    _write_json_atomic(qc_path, result.qc)
    _write_csv_atomic(node_path, _node_rows(result), _NODE_COLUMNS)
    _write_csv_atomic(source_path, _source_rows(result), _SOURCE_COLUMNS)
    _write_csv_atomic(receiver_path, _receiver_rows(result), _RECEIVER_COLUMNS)
    _write_csv_atomic(
        trace_path,
        _trace_preview_rows(result),
        _trace_preview_columns(result),
    )
    return {
        'qc_json': qc_path,
        'nodes_csv': node_path,
        'sources_csv': source_path,
        'receivers_csv': receiver_path,
        'trace_preview_csv': trace_path,
    }


def _validate_replacement_result(
    result: RefractionWeatheringReplacementStaticsResult,
) -> _ValidatedReplacement:
    node_id = _coerce_1d_integer(
        _required(result, 'node_id'),
        name='weathering_replacement_result.node_id',
    )
    n_nodes = int(node_id.shape[0])
    if n_nodes <= 0:
        raise RefractionDatumStaticsError('node_id must contain at least one node')
    if np.unique(node_id).shape[0] != n_nodes:
        raise RefractionDatumStaticsError('node_id values must be unique')
    node_shape = (n_nodes,)

    sorted_trace_index = _coerce_1d_integer(
        _required(result, 'sorted_trace_index'),
        name='weathering_replacement_result.sorted_trace_index',
    )
    n_traces = int(sorted_trace_index.shape[0])
    trace_shape = (n_traces,)

    source_endpoint_key = _coerce_1d_string(
        _required(result, 'source_endpoint_key'),
        name='weathering_replacement_result.source_endpoint_key',
    )
    receiver_endpoint_key = _coerce_1d_string(
        _required(result, 'receiver_endpoint_key'),
        name='weathering_replacement_result.receiver_endpoint_key',
    )
    source_shape = (int(source_endpoint_key.shape[0]),)
    receiver_shape = (int(receiver_endpoint_key.shape[0]),)

    node_pos = {int(node): index for index, node in enumerate(node_id.tolist())}
    node_surface = _coerce_1d_float(
        _required(result, 'node_surface_elevation_m'),
        name='weathering_replacement_result.node_surface_elevation_m',
        expected_shape=node_shape,
        allow_nonfinite=True,
    )
    node_refractor = _coerce_1d_float(
        _required(result, 'node_refractor_elevation_m'),
        name='weathering_replacement_result.node_refractor_elevation_m',
        expected_shape=node_shape,
        allow_nonfinite=True,
    )
    source_node_id = _coerce_1d_integer(
        _required(result, 'source_node_id'),
        name='weathering_replacement_result.source_node_id',
        expected_shape=source_shape,
    )
    receiver_node_id = _coerce_1d_integer(
        _required(result, 'receiver_node_id'),
        name='weathering_replacement_result.receiver_node_id',
        expected_shape=receiver_shape,
    )
    _validate_endpoint_nodes(source_node_id, node_pos, name='source_node_id')
    _validate_endpoint_nodes(receiver_node_id, node_pos, name='receiver_node_id')

    source_surface = _coerce_1d_float(
        _required(result, 'source_surface_elevation_m'),
        name='weathering_replacement_result.source_surface_elevation_m',
        expected_shape=source_shape,
        allow_nonfinite=True,
    )
    receiver_surface = _coerce_1d_float(
        _required(result, 'receiver_surface_elevation_m'),
        name='weathering_replacement_result.receiver_surface_elevation_m',
        expected_shape=receiver_shape,
        allow_nonfinite=True,
    )
    source_thickness = _coerce_1d_float(
        _required(result, 'source_weathering_thickness_m'),
        name='weathering_replacement_result.source_weathering_thickness_m',
        expected_shape=source_shape,
        allow_nonfinite=True,
    )
    receiver_thickness = _coerce_1d_float(
        _required(result, 'receiver_weathering_thickness_m'),
        name='weathering_replacement_result.receiver_weathering_thickness_m',
        expected_shape=receiver_shape,
        allow_nonfinite=True,
    )
    source_refractor = _optional_or_derived_float(
        result,
        field='source_refractor_elevation_m',
        fallback=source_surface - source_thickness,
        expected_shape=source_shape,
    )
    receiver_refractor = _optional_or_derived_float(
        result,
        field='receiver_refractor_elevation_m',
        fallback=receiver_surface - receiver_thickness,
        expected_shape=receiver_shape,
    )
    source_half_intercept = _coerce_1d_float(
        _required(result, 'source_half_intercept_time_s'),
        name='weathering_replacement_result.source_half_intercept_time_s',
        expected_shape=source_shape,
        allow_nonfinite=True,
    )
    receiver_half_intercept = _coerce_1d_float(
        _required(result, 'receiver_half_intercept_time_s'),
        name='weathering_replacement_result.receiver_half_intercept_time_s',
        expected_shape=receiver_shape,
        allow_nonfinite=True,
    )

    source_node_sorted = _coerce_1d_integer(
        _required(result, 'source_node_id_sorted'),
        name='weathering_replacement_result.source_node_id_sorted',
        expected_shape=trace_shape,
    )
    receiver_node_sorted = _coerce_1d_integer(
        _required(result, 'receiver_node_id_sorted'),
        name='weathering_replacement_result.receiver_node_id_sorted',
        expected_shape=trace_shape,
    )
    source_endpoint_key_sorted = _coerce_1d_string(
        _required(result, 'source_endpoint_key_sorted'),
        name='weathering_replacement_result.source_endpoint_key_sorted',
        expected_shape=trace_shape,
    )
    receiver_endpoint_key_sorted = _coerce_1d_string(
        _required(result, 'receiver_endpoint_key_sorted'),
        name='weathering_replacement_result.receiver_endpoint_key_sorted',
        expected_shape=trace_shape,
    )
    (
        source_surface_sorted,
        source_missing_node,
        source_missing_endpoint,
    ) = _map_trace_endpoint_values(
        node_id_sorted=source_node_sorted,
        endpoint_key_sorted=source_endpoint_key_sorted,
        endpoint_key=source_endpoint_key,
        endpoint_node_id=source_node_id,
        endpoint_values=source_surface,
        node_pos=node_pos,
        name='weathering_replacement_result.source_endpoint_key_sorted',
    )
    (
        receiver_surface_sorted,
        receiver_missing_node,
        receiver_missing_endpoint,
    ) = _map_trace_endpoint_values(
        node_id_sorted=receiver_node_sorted,
        endpoint_key_sorted=receiver_endpoint_key_sorted,
        endpoint_key=receiver_endpoint_key,
        endpoint_node_id=receiver_node_id,
        endpoint_values=receiver_surface,
        node_pos=node_pos,
        name='weathering_replacement_result.receiver_endpoint_key_sorted',
    )
    source_refractor_sorted, _, _ = _map_trace_endpoint_values(
        node_id_sorted=source_node_sorted,
        endpoint_key_sorted=source_endpoint_key_sorted,
        endpoint_key=source_endpoint_key,
        endpoint_node_id=source_node_id,
        endpoint_values=source_refractor,
        node_pos=node_pos,
        name='weathering_replacement_result.source_endpoint_key_sorted',
    )
    receiver_refractor_sorted, _, _ = _map_trace_endpoint_values(
        node_id_sorted=receiver_node_sorted,
        endpoint_key_sorted=receiver_endpoint_key_sorted,
        endpoint_key=receiver_endpoint_key,
        endpoint_node_id=receiver_node_id,
        endpoint_values=receiver_refractor,
        node_pos=node_pos,
        name='weathering_replacement_result.receiver_endpoint_key_sorted',
    )

    return _ValidatedReplacement(
        node_id=node_id,
        node_x_m=_coerce_1d_float(
            _required(result, 'node_x_m'),
            name='weathering_replacement_result.node_x_m',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_y_m=_coerce_1d_float(
            _required(result, 'node_y_m'),
            name='weathering_replacement_result.node_y_m',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_surface_elevation_m=node_surface,
        node_kind=_coerce_1d_string(
            _required(result, 'node_kind'),
            name='weathering_replacement_result.node_kind',
            expected_shape=node_shape,
            dtype=_STATUS_DTYPE,
        ),
        node_weathering_thickness_m=_coerce_1d_float(
            _required(result, 'node_weathering_thickness_m'),
            name='weathering_replacement_result.node_weathering_thickness_m',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_refractor_elevation_m=node_refractor,
        node_half_intercept_time_s=_coerce_1d_float(
            _required(result, 'node_half_intercept_time_s'),
            name='weathering_replacement_result.node_half_intercept_time_s',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_solution_status=_coerce_1d_string(
            _required(result, 'node_solution_status'),
            name='weathering_replacement_result.node_solution_status',
            expected_shape=node_shape,
            dtype=_STATUS_DTYPE,
        ),
        node_weathering_status=_coerce_1d_string(
            _required(result, 'node_weathering_status'),
            name='weathering_replacement_result.node_weathering_status',
            expected_shape=node_shape,
            dtype=_STATUS_DTYPE,
        ),
        node_weathering_replacement_shift_s=_coerce_1d_float(
            _required(result, 'node_weathering_replacement_shift_s'),
            name='weathering_replacement_result.node_weathering_replacement_shift_s',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_static_status=_coerce_1d_string(
            _required(result, 'node_static_status'),
            name='weathering_replacement_result.node_static_status',
            expected_shape=node_shape,
            dtype=_STATUS_DTYPE,
        ),
        node_pick_count=_coerce_1d_integer(
            _required(result, 'node_pick_count'),
            name='weathering_replacement_result.node_pick_count',
            expected_shape=node_shape,
        ),
        node_used_pick_count=_coerce_1d_integer(
            _required(result, 'node_used_pick_count'),
            name='weathering_replacement_result.node_used_pick_count',
            expected_shape=node_shape,
        ),
        node_rejected_pick_count=_coerce_1d_integer(
            _required(result, 'node_rejected_pick_count'),
            name='weathering_replacement_result.node_rejected_pick_count',
            expected_shape=node_shape,
        ),
        node_residual_rms_s=_coerce_1d_float(
            _required(result, 'node_residual_rms_s'),
            name='weathering_replacement_result.node_residual_rms_s',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_residual_mad_s=_coerce_1d_float(
            _required(result, 'node_residual_mad_s'),
            name='weathering_replacement_result.node_residual_mad_s',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        active_cell_id=_optional_1d_integer(
            result,
            'active_cell_id',
            name='weathering_replacement_result.active_cell_id',
        ),
        inactive_cell_id=_optional_1d_integer(
            result,
            'inactive_cell_id',
            name='weathering_replacement_result.inactive_cell_id',
        ),
        cell_bedrock_slowness_s_per_m=_optional_1d_float(
            result,
            'cell_bedrock_slowness_s_per_m',
            name='weathering_replacement_result.cell_bedrock_slowness_s_per_m',
        ),
        cell_bedrock_velocity_m_s=_optional_1d_float(
            result,
            'cell_bedrock_velocity_m_s',
            name='weathering_replacement_result.cell_bedrock_velocity_m_s',
        ),
        cell_velocity_status=_optional_1d_string(
            result,
            'cell_velocity_status',
            name='weathering_replacement_result.cell_velocity_status',
            dtype=_STATUS_DTYPE,
        ),
        row_midpoint_cell_id=_optional_1d_integer(
            result,
            'row_midpoint_cell_id',
            name='weathering_replacement_result.row_midpoint_cell_id',
        ),
        node_v2_cell_id=_optional_1d_integer(
            result,
            'node_v2_cell_id',
            name='weathering_replacement_result.node_v2_cell_id',
            expected_shape=node_shape,
        ),
        node_v2_m_s=_optional_1d_float(
            result,
            'node_v2_m_s',
            name='weathering_replacement_result.node_v2_m_s',
            expected_shape=node_shape,
            allow_nonfinite=True,
        ),
        node_v2_status=_optional_1d_string(
            result,
            'node_v2_status',
            name='weathering_replacement_result.node_v2_status',
            expected_shape=node_shape,
            dtype=_STATUS_DTYPE,
        ),
        source_endpoint_key=source_endpoint_key,
        source_id=_coerce_1d_integer(
            _required(result, 'source_id'),
            name='weathering_replacement_result.source_id',
            expected_shape=source_shape,
        ),
        source_node_id=source_node_id,
        source_x_m=_coerce_1d_float(
            _required(result, 'source_x_m'),
            name='weathering_replacement_result.source_x_m',
            expected_shape=source_shape,
            allow_nonfinite=True,
        ),
        source_y_m=_coerce_1d_float(
            _required(result, 'source_y_m'),
            name='weathering_replacement_result.source_y_m',
            expected_shape=source_shape,
            allow_nonfinite=True,
        ),
        source_surface_elevation_m=source_surface,
        source_half_intercept_time_s=source_half_intercept,
        source_weathering_thickness_m=source_thickness,
        source_refractor_elevation_m=source_refractor,
        source_weathering_replacement_shift_s=_coerce_1d_float(
            _required(result, 'source_weathering_replacement_shift_s'),
            name='weathering_replacement_result.source_weathering_replacement_shift_s',
            expected_shape=source_shape,
            allow_nonfinite=True,
        ),
        source_static_status=_coerce_1d_string(
            _required(result, 'source_static_status'),
            name='weathering_replacement_result.source_static_status',
            expected_shape=source_shape,
            dtype=_STATUS_DTYPE,
        ),
        source_v2_cell_id=_optional_1d_integer(
            result,
            'source_v2_cell_id',
            name='weathering_replacement_result.source_v2_cell_id',
            expected_shape=source_shape,
        ),
        source_v2_m_s=_optional_1d_float(
            result,
            'source_v2_m_s',
            name='weathering_replacement_result.source_v2_m_s',
            expected_shape=source_shape,
            allow_nonfinite=True,
        ),
        source_v2_status=_optional_1d_string(
            result,
            'source_v2_status',
            name='weathering_replacement_result.source_v2_status',
            expected_shape=source_shape,
            dtype=_STATUS_DTYPE,
        ),
        receiver_endpoint_key=receiver_endpoint_key,
        receiver_id=_coerce_1d_integer(
            _required(result, 'receiver_id'),
            name='weathering_replacement_result.receiver_id',
            expected_shape=receiver_shape,
        ),
        receiver_node_id=receiver_node_id,
        receiver_x_m=_coerce_1d_float(
            _required(result, 'receiver_x_m'),
            name='weathering_replacement_result.receiver_x_m',
            expected_shape=receiver_shape,
            allow_nonfinite=True,
        ),
        receiver_y_m=_coerce_1d_float(
            _required(result, 'receiver_y_m'),
            name='weathering_replacement_result.receiver_y_m',
            expected_shape=receiver_shape,
            allow_nonfinite=True,
        ),
        receiver_surface_elevation_m=receiver_surface,
        receiver_half_intercept_time_s=receiver_half_intercept,
        receiver_weathering_thickness_m=receiver_thickness,
        receiver_refractor_elevation_m=receiver_refractor,
        receiver_weathering_replacement_shift_s=_coerce_1d_float(
            _required(result, 'receiver_weathering_replacement_shift_s'),
            name=(
                'weathering_replacement_result.'
                'receiver_weathering_replacement_shift_s'
            ),
            expected_shape=receiver_shape,
            allow_nonfinite=True,
        ),
        receiver_static_status=_coerce_1d_string(
            _required(result, 'receiver_static_status'),
            name='weathering_replacement_result.receiver_static_status',
            expected_shape=receiver_shape,
            dtype=_STATUS_DTYPE,
        ),
        receiver_v2_cell_id=_optional_1d_integer(
            result,
            'receiver_v2_cell_id',
            name='weathering_replacement_result.receiver_v2_cell_id',
            expected_shape=receiver_shape,
        ),
        receiver_v2_m_s=_optional_1d_float(
            result,
            'receiver_v2_m_s',
            name='weathering_replacement_result.receiver_v2_m_s',
            expected_shape=receiver_shape,
            allow_nonfinite=True,
        ),
        receiver_v2_status=_optional_1d_string(
            result,
            'receiver_v2_status',
            name='weathering_replacement_result.receiver_v2_status',
            expected_shape=receiver_shape,
            dtype=_STATUS_DTYPE,
        ),
        sorted_trace_index=sorted_trace_index,
        valid_observation_mask_sorted=_coerce_1d_bool(
            _required(result, 'valid_observation_mask_sorted'),
            name='weathering_replacement_result.valid_observation_mask_sorted',
            expected_shape=trace_shape,
        ),
        used_observation_mask_sorted=_coerce_1d_bool(
            _required(result, 'used_observation_mask_sorted'),
            name='weathering_replacement_result.used_observation_mask_sorted',
            expected_shape=trace_shape,
        ),
        source_endpoint_key_sorted=source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=receiver_endpoint_key_sorted,
        source_node_id_sorted=source_node_sorted,
        receiver_node_id_sorted=receiver_node_sorted,
        source_surface_elevation_m_sorted=source_surface_sorted,
        receiver_surface_elevation_m_sorted=receiver_surface_sorted,
        source_weathering_thickness_m_sorted=_coerce_1d_float(
            _required(result, 'source_weathering_thickness_m_sorted'),
            name=(
                'weathering_replacement_result.'
                'source_weathering_thickness_m_sorted'
            ),
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        receiver_weathering_thickness_m_sorted=_coerce_1d_float(
            _required(result, 'receiver_weathering_thickness_m_sorted'),
            name=(
                'weathering_replacement_result.'
                'receiver_weathering_thickness_m_sorted'
            ),
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        source_refractor_elevation_m_sorted=source_refractor_sorted,
        receiver_refractor_elevation_m_sorted=receiver_refractor_sorted,
        source_half_intercept_time_s_sorted=_coerce_1d_float(
            _required(result, 'source_half_intercept_time_s_sorted'),
            name=(
                'weathering_replacement_result.'
                'source_half_intercept_time_s_sorted'
            ),
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        receiver_half_intercept_time_s_sorted=_coerce_1d_float(
            _required(result, 'receiver_half_intercept_time_s_sorted'),
            name=(
                'weathering_replacement_result.'
                'receiver_half_intercept_time_s_sorted'
            ),
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        source_weathering_replacement_shift_s_sorted=_coerce_1d_float(
            _required(result, 'source_weathering_replacement_shift_s_sorted'),
            name=(
                'weathering_replacement_result.'
                'source_weathering_replacement_shift_s_sorted'
            ),
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        receiver_weathering_replacement_shift_s_sorted=_coerce_1d_float(
            _required(result, 'receiver_weathering_replacement_shift_s_sorted'),
            name=(
                'weathering_replacement_result.'
                'receiver_weathering_replacement_shift_s_sorted'
            ),
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        source_missing_node_sorted=source_missing_node,
        receiver_missing_node_sorted=receiver_missing_node,
        source_missing_endpoint_sorted=source_missing_endpoint,
        receiver_missing_endpoint_sorted=receiver_missing_endpoint,
        weathering_replacement_trace_shift_s_sorted=_coerce_1d_float(
            _required(result, 'weathering_replacement_trace_shift_s_sorted'),
            name=(
                'weathering_replacement_result.'
                'weathering_replacement_trace_shift_s_sorted'
            ),
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        source_static_status_sorted=_coerce_1d_string(
            _required(result, 'source_static_status_sorted'),
            name='weathering_replacement_result.source_static_status_sorted',
            expected_shape=trace_shape,
            dtype=_STATUS_DTYPE,
        ),
        receiver_static_status_sorted=_coerce_1d_string(
            _required(result, 'receiver_static_status_sorted'),
            name='weathering_replacement_result.receiver_static_status_sorted',
            expected_shape=trace_shape,
            dtype=_STATUS_DTYPE,
        ),
        trace_static_status_sorted=_coerce_1d_string(
            _required(result, 'trace_static_status_sorted'),
            name='weathering_replacement_result.trace_static_status_sorted',
            expected_shape=trace_shape,
            dtype=_STATUS_DTYPE,
        ),
        trace_static_valid_mask_sorted=_coerce_1d_bool(
            _required(result, 'trace_static_valid_mask_sorted'),
            name='weathering_replacement_result.trace_static_valid_mask_sorted',
            expected_shape=trace_shape,
        ),
        source_v2_cell_id_sorted=_optional_1d_integer(
            result,
            'source_v2_cell_id_sorted',
            name='weathering_replacement_result.source_v2_cell_id_sorted',
            expected_shape=trace_shape,
        ),
        source_v2_m_s_sorted=_optional_1d_float(
            result,
            'source_v2_m_s_sorted',
            name='weathering_replacement_result.source_v2_m_s_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        source_v2_status_sorted=_optional_1d_string(
            result,
            'source_v2_status_sorted',
            name='weathering_replacement_result.source_v2_status_sorted',
            expected_shape=trace_shape,
            dtype=_STATUS_DTYPE,
        ),
        receiver_v2_cell_id_sorted=_optional_1d_integer(
            result,
            'receiver_v2_cell_id_sorted',
            name='weathering_replacement_result.receiver_v2_cell_id_sorted',
            expected_shape=trace_shape,
        ),
        receiver_v2_m_s_sorted=_optional_1d_float(
            result,
            'receiver_v2_m_s_sorted',
            name='weathering_replacement_result.receiver_v2_m_s_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        receiver_v2_status_sorted=_optional_1d_string(
            result,
            'receiver_v2_status_sorted',
            name='weathering_replacement_result.receiver_v2_status_sorted',
            expected_shape=trace_shape,
            dtype=_STATUS_DTYPE,
        ),
        estimated_first_break_time_s_sorted=_coerce_1d_float(
            _required(result, 'estimated_first_break_time_s_sorted'),
            name='weathering_replacement_result.estimated_first_break_time_s_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        first_break_residual_s_sorted=_coerce_1d_float(
            _required(result, 'first_break_residual_s_sorted'),
            name='weathering_replacement_result.first_break_residual_s_sorted',
            expected_shape=trace_shape,
            allow_nonfinite=True,
        ),
        row_trace_index_sorted=_coerce_1d_integer(
            _required(result, 'row_trace_index_sorted'),
            name='weathering_replacement_result.row_trace_index_sorted',
        ),
        row_source_node_id=_coerce_1d_integer(
            _required(result, 'row_source_node_id'),
            name='weathering_replacement_result.row_source_node_id',
        ),
        row_receiver_node_id=_coerce_1d_integer(
            _required(result, 'row_receiver_node_id'),
            name='weathering_replacement_result.row_receiver_node_id',
        ),
        row_distance_m=_coerce_1d_float(
            _required(result, 'row_distance_m'),
            name='weathering_replacement_result.row_distance_m',
        ),
        observed_pick_time_s=_coerce_1d_float(
            _required(result, 'observed_pick_time_s'),
            name='weathering_replacement_result.observed_pick_time_s',
        ),
        modeled_pick_time_s=_coerce_1d_float(
            _required(result, 'modeled_pick_time_s'),
            name='weathering_replacement_result.modeled_pick_time_s',
        ),
        residual_time_s=_coerce_1d_float(
            _required(result, 'residual_time_s'),
            name='weathering_replacement_result.residual_time_s',
        ),
        used_row_mask=_coerce_1d_bool(
            _required(result, 'used_row_mask'),
            name='weathering_replacement_result.used_row_mask',
        ),
        rejected_by_robust_mask=_coerce_1d_bool(
            _required(result, 'rejected_by_robust_mask'),
            name='weathering_replacement_result.rejected_by_robust_mask',
        ),
        row_layer_kind=_optional_1d_string(
            result,
            'row_layer_kind',
            name='weathering_replacement_result.row_layer_kind',
            dtype=_STATUS_DTYPE,
        ),
        row_layer_index=_optional_1d_integer(
            result,
            'row_layer_index',
            name='weathering_replacement_result.row_layer_index',
        ),
        row_source_endpoint_key=_optional_1d_string(
            result,
            'row_source_endpoint_key',
            name='weathering_replacement_result.row_source_endpoint_key',
            dtype=_ENDPOINT_KEY_DTYPE,
        ),
        row_receiver_endpoint_key=_optional_1d_string(
            result,
            'row_receiver_endpoint_key',
            name='weathering_replacement_result.row_receiver_endpoint_key',
            dtype=_ENDPOINT_KEY_DTYPE,
        ),
        row_rejection_reason=_optional_1d_string(
            result,
            'row_rejection_reason',
            name='weathering_replacement_result.row_rejection_reason',
            dtype=_STATUS_DTYPE,
        ),
        row_velocity_m_s=_optional_1d_float(
            result,
            'row_velocity_m_s',
            name='weathering_replacement_result.row_velocity_m_s',
            allow_nonfinite=True,
        ),
    )


def _validate_velocity_context(
    result: RefractionWeatheringReplacementStaticsResult,
    *,
    resolved_first_layer: ResolvedRefractionFirstLayer | None,
) -> _VelocityContext:
    mode = _validate_velocity_mode(_required(result, 'bedrock_velocity_mode'))
    weathering_velocity = _positive_finite(
        _required(result, 'weathering_velocity_m_s'),
        name='weathering_replacement_result.weathering_velocity_m_s',
    )
    weathering_velocity = validate_resolved_first_layer_velocity_match(
        weathering_velocity_m_s=weathering_velocity,
        resolved_first_layer=resolved_first_layer,
        name='weathering_replacement_result.weathering_velocity_m_s',
    )
    bedrock_slowness = _positive_finite(
        _required(result, 'bedrock_slowness_s_per_m'),
        name='weathering_replacement_result.bedrock_slowness_s_per_m',
    )
    bedrock_velocity = _positive_finite(
        _required(result, 'bedrock_velocity_m_s'),
        name='weathering_replacement_result.bedrock_velocity_m_s',
    )
    if bedrock_velocity <= weathering_velocity:
        raise RefractionDatumStaticsError(
            'bedrock_velocity_m_s must be greater than weathering_velocity_m_s'
        )
    derived_slowness = 1.0 / bedrock_velocity
    slowness_tol = max(1.0e-12, abs(bedrock_slowness) * _SLOWNESS_RTOL)
    if abs(derived_slowness - bedrock_slowness) > slowness_tol:
        raise RefractionDatumStaticsError(
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


def _validate_datum_request(
    datum: RefractionStaticDatumRequest,
) -> RefractionStaticDatumRequest:
    try:
        return RefractionStaticDatumRequest.model_validate(datum)
    except ValueError as exc:
        raise RefractionDatumStaticsError(str(exc)) from exc


def _resolve_flat_datum(
    datum: RefractionStaticDatumRequest,
    *,
    flat_active: bool,
) -> float | None:
    if not flat_active:
        return None
    if datum.flat_datum_elevation_m is None:
        raise RefractionDatumStaticsError(
            'datum.flat_datum_elevation_m is required for flat datum modes'
        )
    return _finite_float(
        datum.flat_datum_elevation_m,
        name='datum.flat_datum_elevation_m',
    )


def _resolve_floating_datum_artifact_path(
    *,
    datum: RefractionStaticDatumRequest,
    state: AppState | None,
    explicit_path: Path | None,
    file_id: str | None,
    key1_byte: int | None,
    key2_byte: int | None,
) -> Path | None:
    if datum.mode == 'none' or datum.floating_datum_mode != 'from_artifact':
        return None
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.is_file():
            raise RefractionDatumStaticsError(
                f'floating datum artifact not found: {path}'
            )
        return path
    if state is None:
        raise RefractionDatumStaticsError(_FROM_ARTIFACT_MESSAGE)
    if not datum.floating_datum_job_id or not datum.floating_datum_artifact_name:
        raise RefractionDatumStaticsError(_FROM_ARTIFACT_MESSAGE)
    try:
        return resolve_job_artifact_path(
            state,
            job_id=datum.floating_datum_job_id,
            name=datum.floating_datum_artifact_name,
            allowed_job_types={'statics'},
            expected_file_id=file_id,
            expected_key1_byte=key1_byte,
            expected_key2_byte=key2_byte,
            reference_label='floating datum',
        )
    except ValueError as exc:
        raise RefractionDatumStaticsError(
            f'floating datum artifact validation failed: {exc}'
        ) from exc


def _build_floating_datum_model(
    *,
    data: _ValidatedReplacement,
    datum: RefractionStaticDatumRequest,
    node_pos: dict[int, int],
    artifact_path: Path | None,
) -> _FloatingDatumModel:
    if datum.mode == 'none':
        return _FloatingDatumModel(
            node_floating_datum_elevation_m=np.full(data.n_nodes, np.nan),
            source_floating_datum_elevation_m=np.full(
                data.source_node_id.shape,
                np.nan,
            ),
            receiver_floating_datum_elevation_m=np.full(
                data.receiver_node_id.shape,
                np.nan,
            ),
            source_floating_datum_elevation_m_sorted=np.full(
                data.sorted_trace_index.shape,
                np.nan,
            ),
            receiver_floating_datum_elevation_m_sorted=np.full(
                data.sorted_trace_index.shape,
                np.nan,
            ),
        )

    if datum.floating_datum_mode == 'from_artifact':
        if artifact_path is None:
            raise RefractionDatumStaticsError(_FROM_ARTIFACT_MESSAGE)
        return _load_floating_datum_artifact_model(
            path=artifact_path,
            data=data,
            node_pos=node_pos,
        )

    if datum.floating_datum_mode == 'constant':
        if datum.floating_datum_elevation_m is None:
            raise RefractionDatumStaticsError(
                'datum.floating_datum_elevation_m is required when '
                'floating_datum_mode is constant'
            )
        constant = _finite_float(
            datum.floating_datum_elevation_m,
            name='datum.floating_datum_elevation_m',
        )
        return _FloatingDatumModel(
            node_floating_datum_elevation_m=np.full(data.n_nodes, constant),
            source_floating_datum_elevation_m=np.full(
                data.source_node_id.shape,
                constant,
            ),
            receiver_floating_datum_elevation_m=np.full(
                data.receiver_node_id.shape,
                constant,
            ),
            source_floating_datum_elevation_m_sorted=np.full(
                data.sorted_trace_index.shape,
                constant,
            ),
            receiver_floating_datum_elevation_m_sorted=np.full(
                data.sorted_trace_index.shape,
                constant,
            ),
        )

    if datum.floating_datum_mode == 'surface':
        return _FloatingDatumModel(
            node_floating_datum_elevation_m=np.ascontiguousarray(
                data.node_surface_elevation_m,
                dtype=np.float64,
            ),
            source_floating_datum_elevation_m=np.ascontiguousarray(
                data.source_surface_elevation_m,
                dtype=np.float64,
            ),
            receiver_floating_datum_elevation_m=np.ascontiguousarray(
                data.receiver_surface_elevation_m,
                dtype=np.float64,
            ),
            source_floating_datum_elevation_m_sorted=np.ascontiguousarray(
                data.source_surface_elevation_m_sorted,
                dtype=np.float64,
            ),
            receiver_floating_datum_elevation_m_sorted=np.ascontiguousarray(
                data.receiver_surface_elevation_m_sorted,
                dtype=np.float64,
            ),
        )

    if datum.floating_datum_mode != 'smoothed_topography':
        raise RefractionDatumStaticsError(
            f'unsupported floating_datum_mode: {datum.floating_datum_mode}'
        )

    node_floating = _smooth_node_surface(data=data, datum=datum)
    source_floating = _map_node_values(
        node_id=data.source_node_id,
        node_pos=node_pos,
        node_values=node_floating,
    )
    receiver_floating = _map_node_values(
        node_id=data.receiver_node_id,
        node_pos=node_pos,
        node_values=node_floating,
    )
    source_floating_sorted = _map_node_values(
        node_id=data.source_node_id_sorted,
        node_pos=node_pos,
        node_values=node_floating,
    )
    receiver_floating_sorted = _map_node_values(
        node_id=data.receiver_node_id_sorted,
        node_pos=node_pos,
        node_values=node_floating,
    )
    return _FloatingDatumModel(
        node_floating_datum_elevation_m=node_floating,
        source_floating_datum_elevation_m=source_floating,
        receiver_floating_datum_elevation_m=receiver_floating,
        source_floating_datum_elevation_m_sorted=source_floating_sorted,
        receiver_floating_datum_elevation_m_sorted=receiver_floating_sorted,
    )


def _load_floating_datum_artifact_model(
    *,
    path: Path,
    data: _ValidatedReplacement,
    node_pos: dict[int, int],
) -> _FloatingDatumModel:
    suffix = path.suffix.lower()
    if suffix == '.npz':
        return _load_floating_datum_npz_model(
            path=path,
            data=data,
            node_pos=node_pos,
        )
    if suffix == '.csv':
        return _load_floating_datum_csv_model(
            path=path,
            data=data,
            node_pos=node_pos,
        )
    raise RefractionDatumStaticsError(
        f'unsupported floating datum artifact format: {path.name}'
    )


def _load_floating_datum_npz_model(
    *,
    path: Path,
    data: _ValidatedReplacement,
    node_pos: dict[int, int],
) -> _FloatingDatumModel:
    try:
        with np.load(path, allow_pickle=False) as artifact:
            keys = set(artifact.files)
            if {'node_id', 'floating_datum_elevation_m'} <= keys:
                return _floating_model_from_node_values(
                    data=data,
                    node_pos=node_pos,
                    node_id=_coerce_1d_integer(
                        artifact['node_id'],
                        name='floating_datum_artifact.node_id',
                    ),
                    floating_datum_elevation_m=_coerce_1d_float(
                        artifact['floating_datum_elevation_m'],
                        name=(
                            'floating_datum_artifact.'
                            'floating_datum_elevation_m'
                        ),
                        allow_nonfinite=True,
                    ),
                )
            endpoint_keys = {
                'source_endpoint_key',
                'source_floating_datum_elevation_m',
                'receiver_endpoint_key',
                'receiver_floating_datum_elevation_m',
            }
            if endpoint_keys <= keys:
                return _floating_model_from_endpoint_values(
                    data=data,
                    node_pos=node_pos,
                    source_endpoint_key=_coerce_1d_string(
                        artifact['source_endpoint_key'],
                        name='floating_datum_artifact.source_endpoint_key',
                    ),
                    source_floating_datum_elevation_m=_coerce_1d_float(
                        artifact['source_floating_datum_elevation_m'],
                        name=(
                            'floating_datum_artifact.'
                            'source_floating_datum_elevation_m'
                        ),
                        allow_nonfinite=True,
                    ),
                    receiver_endpoint_key=_coerce_1d_string(
                        artifact['receiver_endpoint_key'],
                        name='floating_datum_artifact.receiver_endpoint_key',
                    ),
                    receiver_floating_datum_elevation_m=_coerce_1d_float(
                        artifact['receiver_floating_datum_elevation_m'],
                        name=(
                            'floating_datum_artifact.'
                            'receiver_floating_datum_elevation_m'
                        ),
                        allow_nonfinite=True,
                    ),
                )
    except RefractionDatumStaticsError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RefractionDatumStaticsError(
            f'could not read floating datum artifact: {path.name}'
        ) from exc
    raise RefractionDatumStaticsError(
        'floating datum NPZ artifact must contain either node_id and '
        'floating_datum_elevation_m, or source/receiver endpoint keys and '
        'floating datum elevations'
    )


def _load_floating_datum_csv_model(
    *,
    path: Path,
    data: _ValidatedReplacement,
    node_pos: dict[int, int],
) -> _FloatingDatumModel:
    try:
        with path.open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = set(reader.fieldnames or [])
    except Exception as exc:  # noqa: BLE001
        raise RefractionDatumStaticsError(
            f'could not read floating datum artifact: {path.name}'
        ) from exc

    if {'node_id', 'floating_datum_elevation_m'} <= fieldnames:
        node_id: list[int] = []
        elevation: list[float] = []
        for row_index, row in enumerate(rows, start=2):
            raw_node = (row.get('node_id') or '').strip()
            raw_elevation = (row.get('floating_datum_elevation_m') or '').strip()
            if not raw_node and not raw_elevation:
                continue
            node_id.append(
                _csv_int(raw_node, name=f'floating_datum_artifact.csv:{row_index}')
            )
            elevation.append(
                _finite_float(
                    raw_elevation,
                    name=(
                        'floating_datum_artifact.'
                        f'floating_datum_elevation_m:{row_index}'
                    ),
                )
            )
        if node_id:
            return _floating_model_from_node_values(
                data=data,
                node_pos=node_pos,
                node_id=np.asarray(node_id, dtype=np.int64),
                floating_datum_elevation_m=np.asarray(elevation, dtype=np.float64),
            )

    endpoint_fields = {
        'source_endpoint_key',
        'source_floating_datum_elevation_m',
        'receiver_endpoint_key',
        'receiver_floating_datum_elevation_m',
    }
    if endpoint_fields <= fieldnames:
        source_key: list[str] = []
        source_elevation: list[float] = []
        receiver_key: list[str] = []
        receiver_elevation: list[float] = []
        for row_index, row in enumerate(rows, start=2):
            raw_source_key = (row.get('source_endpoint_key') or '').strip()
            raw_source_elevation = (
                row.get('source_floating_datum_elevation_m') or ''
            ).strip()
            if raw_source_key or raw_source_elevation:
                source_key.append(raw_source_key)
                source_elevation.append(
                    _finite_float(
                        raw_source_elevation,
                        name=(
                            'floating_datum_artifact.'
                            f'source_floating_datum_elevation_m:{row_index}'
                        ),
                    )
                )

            raw_receiver_key = (row.get('receiver_endpoint_key') or '').strip()
            raw_receiver_elevation = (
                row.get('receiver_floating_datum_elevation_m') or ''
            ).strip()
            if raw_receiver_key or raw_receiver_elevation:
                receiver_key.append(raw_receiver_key)
                receiver_elevation.append(
                    _finite_float(
                        raw_receiver_elevation,
                        name=(
                            'floating_datum_artifact.'
                            f'receiver_floating_datum_elevation_m:{row_index}'
                        ),
                    )
                )

        if source_key or receiver_key:
            return _floating_model_from_endpoint_values(
                data=data,
                node_pos=node_pos,
                source_endpoint_key=np.asarray(source_key, dtype=object),
                source_floating_datum_elevation_m=np.asarray(
                    source_elevation,
                    dtype=np.float64,
                ),
                receiver_endpoint_key=np.asarray(receiver_key, dtype=object),
                receiver_floating_datum_elevation_m=np.asarray(
                    receiver_elevation,
                    dtype=np.float64,
                ),
            )

    raise RefractionDatumStaticsError(
        'floating datum CSV artifact must contain node_id and '
        'floating_datum_elevation_m, or source/receiver endpoint keys and '
        'floating datum elevations'
    )


def _floating_model_from_node_values(
    *,
    data: _ValidatedReplacement,
    node_pos: dict[int, int],
    node_id: np.ndarray,
    floating_datum_elevation_m: np.ndarray,
) -> _FloatingDatumModel:
    node_values = _coerce_1d_float(
        floating_datum_elevation_m,
        name='floating_datum_artifact.floating_datum_elevation_m',
        expected_shape=node_id.shape,
        allow_nonfinite=True,
    )
    node_floating = np.full(data.n_nodes, np.nan, dtype=np.float64)
    seen: set[int] = set()
    for index, raw_node in enumerate(node_id.tolist()):
        node = int(raw_node)
        if node in seen:
            raise RefractionDatumStaticsError(
                f'floating datum artifact contains duplicate node_id: {node}'
            )
        seen.add(node)
        target_index = node_pos.get(node)
        if target_index is not None:
            node_floating[target_index] = node_values[index]
    if not np.any(np.isfinite(node_floating)):
        raise RefractionDatumStaticsError(
            'floating datum artifact does not contain any matching node_id values'
        )
    return _floating_model_from_node_floating(
        data=data,
        node_pos=node_pos,
        node_floating=node_floating,
    )


def _floating_model_from_node_floating(
    *,
    data: _ValidatedReplacement,
    node_pos: dict[int, int],
    node_floating: np.ndarray,
) -> _FloatingDatumModel:
    source_floating = _map_node_values(
        node_id=data.source_node_id,
        node_pos=node_pos,
        node_values=node_floating,
    )
    receiver_floating = _map_node_values(
        node_id=data.receiver_node_id,
        node_pos=node_pos,
        node_values=node_floating,
    )
    source_floating_sorted = _map_node_values(
        node_id=data.source_node_id_sorted,
        node_pos=node_pos,
        node_values=node_floating,
    )
    receiver_floating_sorted = _map_node_values(
        node_id=data.receiver_node_id_sorted,
        node_pos=node_pos,
        node_values=node_floating,
    )
    return _FloatingDatumModel(
        node_floating_datum_elevation_m=np.ascontiguousarray(
            node_floating,
            dtype=np.float64,
        ),
        source_floating_datum_elevation_m=source_floating,
        receiver_floating_datum_elevation_m=receiver_floating,
        source_floating_datum_elevation_m_sorted=source_floating_sorted,
        receiver_floating_datum_elevation_m_sorted=receiver_floating_sorted,
    )


def _floating_model_from_endpoint_values(
    *,
    data: _ValidatedReplacement,
    node_pos: dict[int, int],
    source_endpoint_key: np.ndarray,
    source_floating_datum_elevation_m: np.ndarray,
    receiver_endpoint_key: np.ndarray,
    receiver_floating_datum_elevation_m: np.ndarray,
) -> _FloatingDatumModel:
    source_floating = _map_endpoint_key_values(
        requested_key=data.source_endpoint_key,
        artifact_key=source_endpoint_key,
        artifact_values=source_floating_datum_elevation_m,
        name='floating_datum_artifact.source_endpoint_key',
    )
    receiver_floating = _map_endpoint_key_values(
        requested_key=data.receiver_endpoint_key,
        artifact_key=receiver_endpoint_key,
        artifact_values=receiver_floating_datum_elevation_m,
        name='floating_datum_artifact.receiver_endpoint_key',
    )
    if not np.any(np.isfinite(source_floating)) and not np.any(
        np.isfinite(receiver_floating)
    ):
        raise RefractionDatumStaticsError(
            'floating datum artifact does not contain any matching endpoint keys'
        )
    source_floating_sorted, _, _ = _map_trace_endpoint_values(
        node_id_sorted=data.source_node_id_sorted,
        endpoint_key_sorted=data.source_endpoint_key_sorted,
        endpoint_key=data.source_endpoint_key,
        endpoint_node_id=data.source_node_id,
        endpoint_values=source_floating,
        node_pos=node_pos,
        name='weathering_replacement_result.source_endpoint_key_sorted',
    )
    receiver_floating_sorted, _, _ = _map_trace_endpoint_values(
        node_id_sorted=data.receiver_node_id_sorted,
        endpoint_key_sorted=data.receiver_endpoint_key_sorted,
        endpoint_key=data.receiver_endpoint_key,
        endpoint_node_id=data.receiver_node_id,
        endpoint_values=receiver_floating,
        node_pos=node_pos,
        name='weathering_replacement_result.receiver_endpoint_key_sorted',
    )
    node_floating = _node_floating_from_endpoint_values(
        data=data,
        node_pos=node_pos,
        source_floating=source_floating,
        receiver_floating=receiver_floating,
    )
    return _FloatingDatumModel(
        node_floating_datum_elevation_m=node_floating,
        source_floating_datum_elevation_m=source_floating,
        receiver_floating_datum_elevation_m=receiver_floating,
        source_floating_datum_elevation_m_sorted=source_floating_sorted,
        receiver_floating_datum_elevation_m_sorted=receiver_floating_sorted,
    )


def _map_endpoint_key_values(
    *,
    requested_key: np.ndarray,
    artifact_key: np.ndarray,
    artifact_values: np.ndarray,
    name: str,
) -> np.ndarray:
    keys = _coerce_1d_string(artifact_key, name=name)
    values = _coerce_1d_float(
        artifact_values,
        name=f'{name}.floating_datum_elevation_m',
        expected_shape=keys.shape,
        allow_nonfinite=True,
    )
    mapping: dict[str, float] = {}
    for index, raw_key in enumerate(keys.tolist()):
        key = str(raw_key)
        if not key:
            raise RefractionDatumStaticsError(f'{name} contains an empty key')
        if key in mapping:
            raise RefractionDatumStaticsError(
                f'{name} contains duplicate endpoint key: {key}'
            )
        mapping[key] = float(values[index])
    out = np.full(requested_key.shape, np.nan, dtype=np.float64)
    for index, raw_key in enumerate(requested_key.tolist()):
        value = mapping.get(str(raw_key))
        if value is not None:
            out[index] = value
    return np.ascontiguousarray(out, dtype=np.float64)


def _node_floating_from_endpoint_values(
    *,
    data: _ValidatedReplacement,
    node_pos: dict[int, int],
    source_floating: np.ndarray,
    receiver_floating: np.ndarray,
) -> np.ndarray:
    sums = np.zeros(data.n_nodes, dtype=np.float64)
    counts = np.zeros(data.n_nodes, dtype=np.int64)
    for endpoint_node_id, values in (
        (data.source_node_id, source_floating),
        (data.receiver_node_id, receiver_floating),
    ):
        for raw_node, raw_value in zip(
            endpoint_node_id.tolist(),
            values.tolist(),
            strict=True,
        ):
            value = float(raw_value)
            if not np.isfinite(value):
                continue
            target_index = node_pos.get(int(raw_node))
            if target_index is None:
                continue
            sums[target_index] += value
            counts[target_index] += 1
    out = np.full(data.n_nodes, np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return np.ascontiguousarray(out, dtype=np.float64)


def _smooth_node_surface(
    *,
    data: _ValidatedReplacement,
    datum: RefractionStaticDatumRequest,
) -> np.ndarray:
    window = datum.smoothing_window_nodes
    if window is None:
        raise RefractionDatumStaticsError('datum.smoothing_window_nodes is required')
    if window <= 0 or window % 2 == 0:
        raise RefractionDatumStaticsError(
            'datum.smoothing_window_nodes must be a positive odd integer'
        )
    order, coordinate = _node_smoothing_order(data)
    surface_sorted = data.node_surface_elevation_m[order]
    smoothed_sorted = np.full(surface_sorted.shape, np.nan, dtype=np.float64)
    half = window // 2
    for sorted_index in range(int(surface_sorted.shape[0])):
        neighbor_values: np.ndarray
        if datum.smoothing_radius_m is not None and np.isfinite(
            coordinate[sorted_index]
        ):
            radius = float(datum.smoothing_radius_m)
            radius_mask = np.isfinite(coordinate) & (
                np.abs(coordinate - coordinate[sorted_index]) <= radius
            )
            neighbor_values = surface_sorted[radius_mask]
        else:
            start = max(0, sorted_index - half)
            stop = min(int(surface_sorted.shape[0]), sorted_index + half + 1)
            neighbor_values = surface_sorted[start:stop]
        finite = neighbor_values[np.isfinite(neighbor_values)]
        if finite.size == 0 and datum.smoothing_radius_m is not None:
            start = max(0, sorted_index - half)
            stop = min(int(surface_sorted.shape[0]), sorted_index + half + 1)
            finite = surface_sorted[start:stop]
            finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        if datum.smoothing_method == 'median':
            smoothed_sorted[sorted_index] = float(np.median(finite))
        else:
            smoothed_sorted[sorted_index] = float(np.mean(finite))
    out = np.full(surface_sorted.shape, np.nan, dtype=np.float64)
    out[order] = smoothed_sorted
    return np.ascontiguousarray(out, dtype=np.float64)


def _node_smoothing_order(data: _ValidatedReplacement) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data.node_x_m, dtype=np.float64)
    y = np.asarray(data.node_y_m, dtype=np.float64)
    finite_xy = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite_xy) >= 2 and np.nanmax(x[finite_xy]) > np.nanmin(
        x[finite_xy]
    ):
        coordinate = x.copy()
    elif np.count_nonzero(finite_xy) >= 2:
        points = np.column_stack([x[finite_xy], y[finite_xy]])
        centered = points - np.mean(points, axis=0)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            projected = centered @ vh[0]
        except np.linalg.LinAlgError:
            projected = np.arange(int(points.shape[0]), dtype=np.float64)
        coordinate = np.full(data.node_id.shape, np.nan, dtype=np.float64)
        coordinate[finite_xy] = projected
    else:
        coordinate = np.arange(data.n_nodes, dtype=np.float64)

    sortable = np.where(np.isfinite(coordinate), coordinate, np.inf)
    order = np.lexsort((data.node_id, sortable))
    return np.ascontiguousarray(order, dtype=np.int64), np.ascontiguousarray(
        coordinate[order],
        dtype=np.float64,
    )


def _endpoint_floating_shift(
    *,
    surface_elevation_m: np.ndarray,
    floating_datum_elevation_m: np.ndarray,
    bedrock_velocity_m_s: float,
    active: bool,
) -> np.ndarray:
    if not active:
        return np.zeros(surface_elevation_m.shape, dtype=np.float64)
    return np.ascontiguousarray(
        -(surface_elevation_m - floating_datum_elevation_m) / bedrock_velocity_m_s,
        dtype=np.float64,
    )


def _endpoint_flat_shift(
    *,
    flat_datum_elevation_m: float | None,
    floating_datum_elevation_m: np.ndarray,
    bedrock_velocity_m_s: float,
    active: bool,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not active:
        return np.zeros(shape, dtype=np.float64)
    if flat_datum_elevation_m is None:
        return np.full(shape, np.nan, dtype=np.float64)
    return np.ascontiguousarray(
        (flat_datum_elevation_m - floating_datum_elevation_m)
        / bedrock_velocity_m_s,
        dtype=np.float64,
    )


def _trace_floating_shift(
    *,
    source_surface_elevation_m: np.ndarray,
    receiver_surface_elevation_m: np.ndarray,
    source_floating_datum_elevation_m: np.ndarray,
    receiver_floating_datum_elevation_m: np.ndarray,
    bedrock_velocity_m_s: float,
    active: bool,
) -> np.ndarray:
    if not active:
        return np.zeros(source_surface_elevation_m.shape, dtype=np.float64)
    return compute_floating_datum_elevation_shift_s(
        true_source_elevation_m=source_surface_elevation_m,
        true_receiver_elevation_m=receiver_surface_elevation_m,
        floating_source_elevation_m=source_floating_datum_elevation_m,
        floating_receiver_elevation_m=receiver_floating_datum_elevation_m,
        bedrock_velocity_m_s=bedrock_velocity_m_s,
    )


def _trace_flat_shift(
    *,
    flat_datum_elevation_m: float | None,
    source_floating_datum_elevation_m: np.ndarray,
    receiver_floating_datum_elevation_m: np.ndarray,
    bedrock_velocity_m_s: float,
    active: bool,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not active:
        return np.zeros(shape, dtype=np.float64)
    if flat_datum_elevation_m is None:
        return np.full(shape, np.nan, dtype=np.float64)
    return compute_flat_datum_shift_s(
        flat_datum_elevation_m=flat_datum_elevation_m,
        floating_source_elevation_m=source_floating_datum_elevation_m,
        floating_receiver_elevation_m=receiver_floating_datum_elevation_m,
        bedrock_velocity_m_s=bedrock_velocity_m_s,
    )


def _classify_node_status(
    *,
    inherited_status: np.ndarray,
    surface_elevation_m: np.ndarray,
    floating_datum_elevation_m: np.ndarray,
    refractor_elevation_m: np.ndarray,
    weathering_replacement_shift_s: np.ndarray,
    flat_datum_elevation_m: float | None,
    datum: RefractionStaticDatumRequest,
) -> np.ndarray:
    if datum.mode == 'none':
        return np.full(surface_elevation_m.shape, 'inactive', dtype=_STATUS_DTYPE)
    status = np.full(surface_elevation_m.shape, 'ok', dtype=_STATUS_DTYPE)
    _assign_inherited_replacement_status(status, inherited_status)
    _assign(
        status,
        ~np.isfinite(weathering_replacement_shift_s),
        'invalid_weathering_replacement',
    )
    _assign(status, ~np.isfinite(surface_elevation_m), 'invalid_surface_elevation')
    _assign(
        status,
        ~np.isfinite(floating_datum_elevation_m),
        'invalid_floating_datum_elevation',
    )
    _assign(
        status,
        _floating_below_refractor_mask(
            floating_datum_elevation_m=floating_datum_elevation_m,
            refractor_elevation_m=refractor_elevation_m,
        ),
        'floating_datum_below_refractor',
    )
    if datum.mode in {'flat_only', 'floating_and_flat'}:
        flat_invalid = flat_datum_elevation_m is None or not np.isfinite(
            flat_datum_elevation_m
        )
        _assign(
            status,
            np.full(surface_elevation_m.shape, flat_invalid, dtype=bool),
            'invalid_flat_datum_elevation',
        )
        if not datum.allow_flat_datum_above_topography:
            _assign(
                status,
                _flat_above_topography_mask(
                    flat_datum_elevation_m=flat_datum_elevation_m,
                    surface_elevation_m=surface_elevation_m,
                ),
                'invalid_flat_datum_elevation',
            )
        if not datum.allow_flat_datum_below_refractor:
            _assign(
                status,
                _flat_below_refractor_mask(
                    flat_datum_elevation_m=flat_datum_elevation_m,
                    refractor_elevation_m=refractor_elevation_m,
                ),
                'flat_datum_below_refractor',
            )
    return np.ascontiguousarray(status, dtype=_STATUS_DTYPE)


def _classify_endpoint_status(
    *,
    inherited_status: np.ndarray,
    surface_elevation_m: np.ndarray,
    floating_datum_elevation_m: np.ndarray,
    refractor_elevation_m: np.ndarray,
    weathering_replacement_shift_s: np.ndarray,
    flat_datum_elevation_m: float | None,
    datum: RefractionStaticDatumRequest,
    max_abs_shift_ms: float | None,
    refraction_shift_s: np.ndarray,
) -> np.ndarray:
    inherited = np.asarray(inherited_status).astype(str, copy=False)
    if datum.mode == 'none':
        status = np.full(surface_elevation_m.shape, 'ok', dtype=_STATUS_DTYPE)
        _assign_inherited_replacement_status(status, inherited)
        _assign(
            status,
            ~np.isfinite(weathering_replacement_shift_s),
            'invalid_weathering_replacement',
        )
        return status
    status = np.full(surface_elevation_m.shape, 'ok', dtype=_STATUS_DTYPE)
    _assign_inherited_replacement_status(status, inherited)
    _assign(
        status,
        ~np.isfinite(weathering_replacement_shift_s),
        'invalid_weathering_replacement',
    )
    _assign(status, ~np.isfinite(surface_elevation_m), 'invalid_surface_elevation')
    _assign(
        status,
        ~np.isfinite(floating_datum_elevation_m),
        'invalid_floating_datum_elevation',
    )
    _assign(
        status,
        _floating_below_refractor_mask(
            floating_datum_elevation_m=floating_datum_elevation_m,
            refractor_elevation_m=refractor_elevation_m,
        ),
        'floating_datum_below_refractor',
    )
    if datum.mode in {'flat_only', 'floating_and_flat'}:
        flat_invalid = flat_datum_elevation_m is None or not np.isfinite(
            flat_datum_elevation_m
        )
        _assign(
            status,
            np.full(surface_elevation_m.shape, flat_invalid, dtype=bool),
            'invalid_flat_datum_elevation',
        )
        if not datum.allow_flat_datum_above_topography:
            _assign(
                status,
                _flat_above_topography_mask(
                    flat_datum_elevation_m=flat_datum_elevation_m,
                    surface_elevation_m=surface_elevation_m,
                ),
                'invalid_flat_datum_elevation',
            )
        if not datum.allow_flat_datum_below_refractor:
            _assign(
                status,
                _flat_below_refractor_mask(
                    flat_datum_elevation_m=flat_datum_elevation_m,
                    refractor_elevation_m=refractor_elevation_m,
                ),
                'flat_datum_below_refractor',
            )
    _assign(status, ~np.isfinite(refraction_shift_s), 'invalid_datum_shift')
    if max_abs_shift_ms is not None:
        exceeds = np.isfinite(refraction_shift_s) & (
            np.abs(refraction_shift_s) * 1000.0 > max_abs_shift_ms
        )
        _assign(status, exceeds, 'exceeds_max_abs_shift')
    return np.ascontiguousarray(status, dtype=_STATUS_DTYPE)


def _classify_trace_status(
    *,
    data: _ValidatedReplacement,
    datum: RefractionStaticDatumRequest,
    floating: _FloatingDatumModel,
    flat_datum_elevation_m: float | None,
    floating_shift_s: np.ndarray,
    flat_shift_s: np.ndarray,
    refraction_shift_s: np.ndarray,
    max_abs_shift_ms: float | None,
) -> np.ndarray:
    status = np.full(data.sorted_trace_index.shape, 'ok', dtype=_STATUS_DTYPE)
    inherited = np.asarray(data.trace_static_status_sorted).astype(str, copy=False)
    _assign(
        status,
        ~np.asarray(data.valid_observation_mask_sorted, dtype=bool),
        'not_observed',
    )
    _assign_inherited_replacement_status(status, inherited)
    _assign_inherited_replacement_status(status, data.source_static_status_sorted)
    _assign_inherited_replacement_status(status, data.receiver_static_status_sorted)
    _assign(
        status,
        ~np.asarray(data.trace_static_valid_mask_sorted, dtype=bool),
        'invalid_weathering_replacement',
    )
    _assign(
        status,
        data.source_missing_endpoint_sorted | data.receiver_missing_endpoint_sorted,
        'missing_endpoint',
    )
    _assign(
        status,
        data.source_missing_node_sorted | data.receiver_missing_node_sorted,
        'missing_node',
    )
    _assign(
        status,
        ~np.isfinite(data.weathering_replacement_trace_shift_s_sorted),
        'invalid_weathering_replacement',
    )
    if datum.mode != 'none':
        _assign(
            status,
            ~np.isfinite(data.source_surface_elevation_m_sorted)
            | ~np.isfinite(data.receiver_surface_elevation_m_sorted),
            'invalid_surface_elevation',
        )
        _assign(
            status,
            ~np.isfinite(floating.source_floating_datum_elevation_m_sorted)
            | ~np.isfinite(floating.receiver_floating_datum_elevation_m_sorted),
            'invalid_floating_datum_elevation',
        )
        _assign(
            status,
            _floating_below_refractor_mask(
                floating_datum_elevation_m=(
                    floating.source_floating_datum_elevation_m_sorted
                ),
                refractor_elevation_m=data.source_refractor_elevation_m_sorted,
            )
            | _floating_below_refractor_mask(
                floating_datum_elevation_m=(
                    floating.receiver_floating_datum_elevation_m_sorted
                ),
                refractor_elevation_m=data.receiver_refractor_elevation_m_sorted,
            ),
            'floating_datum_below_refractor',
        )
    if datum.mode in {'flat_only', 'floating_and_flat'}:
        flat_invalid = flat_datum_elevation_m is None or not np.isfinite(
            flat_datum_elevation_m
        )
        _assign(
            status,
            np.full(data.sorted_trace_index.shape, flat_invalid, dtype=bool),
            'invalid_flat_datum_elevation',
        )
        if not datum.allow_flat_datum_above_topography:
            _assign(
                status,
                _flat_above_topography_mask(
                    flat_datum_elevation_m=flat_datum_elevation_m,
                    surface_elevation_m=data.source_surface_elevation_m_sorted,
                )
                | _flat_above_topography_mask(
                    flat_datum_elevation_m=flat_datum_elevation_m,
                    surface_elevation_m=data.receiver_surface_elevation_m_sorted,
                ),
                'invalid_flat_datum_elevation',
            )
        if not datum.allow_flat_datum_below_refractor:
            _assign(
                status,
                _flat_below_refractor_mask(
                    flat_datum_elevation_m=flat_datum_elevation_m,
                    refractor_elevation_m=data.source_refractor_elevation_m_sorted,
                )
                | _flat_below_refractor_mask(
                    flat_datum_elevation_m=flat_datum_elevation_m,
                    refractor_elevation_m=data.receiver_refractor_elevation_m_sorted,
                ),
                'flat_datum_below_refractor',
            )
    _assign(
        status,
        ~np.isfinite(floating_shift_s) | ~np.isfinite(flat_shift_s),
        'invalid_datum_shift',
    )
    _assign(status, ~np.isfinite(refraction_shift_s), 'invalid_datum_shift')
    if max_abs_shift_ms is not None:
        exceeds = np.isfinite(refraction_shift_s) & (
            np.abs(refraction_shift_s) * 1000.0 > max_abs_shift_ms
        )
        _assign(status, exceeds, 'exceeds_max_abs_shift')
    return np.ascontiguousarray(status, dtype=_STATUS_DTYPE)


def _compose_endpoint_shift(
    *,
    weathering_shift_s: np.ndarray,
    floating_shift_s: np.ndarray,
    flat_shift_s: np.ndarray,
    status: np.ndarray,
) -> np.ndarray:
    out = np.ascontiguousarray(
        weathering_shift_s + floating_shift_s + flat_shift_s,
        dtype=np.float64,
    )
    invalid = _endpoint_nan_mask(status)
    out[invalid] = np.nan
    return out


def _endpoint_nan_mask(status: np.ndarray) -> np.ndarray:
    text = np.asarray(status).astype(str, copy=False)
    return np.isin(
        text,
        [
            'missing_node',
            'missing_endpoint',
            'invalid_bedrock_velocity',
            'invalid_surface_elevation',
            'invalid_floating_datum_elevation',
            'invalid_flat_datum_elevation',
            'invalid_weathering_replacement',
            'invalid_nonfinite_input',
            'invalid_velocity_order',
            'floating_datum_below_refractor',
            'flat_datum_below_refractor',
            'invalid_datum_shift',
            'inactive',
            *LOCAL_V2_STATUS_VALUES,
        ],
    )


def _trace_nan_mask(status: np.ndarray) -> np.ndarray:
    text = np.asarray(status).astype(str, copy=False)
    return np.isin(
        text,
        [
            'missing_node',
            'missing_endpoint',
            'invalid_bedrock_velocity',
            'invalid_surface_elevation',
            'invalid_floating_datum_elevation',
            'invalid_flat_datum_elevation',
            'invalid_weathering_replacement',
            'invalid_nonfinite_input',
            'invalid_velocity_order',
            'floating_datum_below_refractor',
            'flat_datum_below_refractor',
            'invalid_datum_shift',
            'inactive',
            *LOCAL_V2_STATUS_VALUES,
        ],
    )


def _trace_valid_mask(status: np.ndarray, refraction_shift_s: np.ndarray) -> np.ndarray:
    text = np.asarray(status).astype(str, copy=False)
    valid = np.isfinite(refraction_shift_s)
    valid &= ~np.isin(text, list(_INVALID_TRACE_STATUSES))
    return np.ascontiguousarray(valid, dtype=bool)


def _floating_below_refractor_mask(
    *,
    floating_datum_elevation_m: np.ndarray,
    refractor_elevation_m: np.ndarray,
) -> np.ndarray:
    return (
        np.isfinite(floating_datum_elevation_m)
        & np.isfinite(refractor_elevation_m)
        & (floating_datum_elevation_m < refractor_elevation_m)
    )


def _flat_below_refractor_mask(
    *,
    flat_datum_elevation_m: float | None,
    refractor_elevation_m: np.ndarray,
) -> np.ndarray:
    if flat_datum_elevation_m is None or not np.isfinite(flat_datum_elevation_m):
        return np.zeros(refractor_elevation_m.shape, dtype=bool)
    return np.isfinite(refractor_elevation_m) & (
        float(flat_datum_elevation_m) < refractor_elevation_m
    )


def _flat_above_topography_mask(
    *,
    flat_datum_elevation_m: float | None,
    surface_elevation_m: np.ndarray,
) -> np.ndarray:
    if flat_datum_elevation_m is None or not np.isfinite(flat_datum_elevation_m):
        return np.zeros(surface_elevation_m.shape, dtype=bool)
    return np.isfinite(surface_elevation_m) & (
        float(flat_datum_elevation_m) > surface_elevation_m
    )


def _build_qc(
    *,
    velocity: _VelocityContext,
    datum: RefractionStaticDatumRequest,
    data: _ValidatedReplacement,
    floating: _FloatingDatumModel,
    flat_datum_elevation_m: float | None,
    weathering_replacement_trace_shift_s: np.ndarray,
    floating_datum_shift_s: np.ndarray,
    flat_datum_shift_s: np.ndarray,
    refraction_trace_shift_s: np.ndarray,
    trace_status: np.ndarray,
    trace_valid_mask: np.ndarray,
    node_status: np.ndarray,
    source_status: np.ndarray,
    receiver_status: np.ndarray,
    max_abs_shift_ms: float | None,
    upstream_qc: dict[str, Any],
) -> dict[str, Any]:
    valid_weathering_ms = weathering_replacement_trace_shift_s[trace_valid_mask] * 1000.0
    valid_floating_ms = floating_datum_shift_s[trace_valid_mask] * 1000.0
    valid_flat_ms = flat_datum_shift_s[trace_valid_mask] * 1000.0
    valid_refraction_ms = refraction_trace_shift_s[trace_valid_mask] * 1000.0
    finite_refraction = refraction_trace_shift_s[np.isfinite(refraction_trace_shift_s)]
    surface_values = np.concatenate(
        [data.source_surface_elevation_m, data.receiver_surface_elevation_m]
    )
    floating_values = np.concatenate(
        [
            floating.source_floating_datum_elevation_m,
            floating.receiver_floating_datum_elevation_m,
        ]
    )
    refractor_values = np.concatenate(
        [data.source_refractor_elevation_m, data.receiver_refractor_elevation_m]
    )
    qc: dict[str, Any] = {
        'method': 'gli_variable_thickness',
        'static_component': 'datum_composition',
        'bedrock_velocity_mode': velocity.mode,
        'bedrock_velocity_m_s': float(velocity.bedrock_velocity_m_s),
        'bedrock_slowness_s_per_m': float(velocity.bedrock_slowness_s_per_m),
        'weathering_velocity_m_s': float(velocity.weathering_velocity_m_s),
        'replacement_slowness_delta_s_per_m': float(
            velocity.replacement_slowness_delta_s_per_m
        ),
        'datum_mode': datum.mode,
        'floating_datum_mode': datum.floating_datum_mode,
        'flat_datum_elevation_m': _json_optional_float(flat_datum_elevation_m),
        'n_traces': int(data.n_traces),
        'n_valid_observations': int(
            np.count_nonzero(data.valid_observation_mask_sorted)
        ),
        'n_used_observations': int(np.count_nonzero(data.used_observation_mask_sorted)),
        'n_nodes': int(data.n_nodes),
        'n_source_endpoints': int(data.source_endpoint_key.shape[0]),
        'n_receiver_endpoints': int(data.receiver_endpoint_key.shape[0]),
        'surface_elevation_min_m': _json_stat(surface_values, 'min'),
        'surface_elevation_max_m': _json_stat(surface_values, 'max'),
        'floating_datum_elevation_min_m': _json_stat(floating_values, 'min'),
        'floating_datum_elevation_max_m': _json_stat(floating_values, 'max'),
        'floating_datum_elevation_median_m': _json_stat(floating_values, 'median'),
        'refractor_elevation_min_m': _json_stat(refractor_values, 'min'),
        'refractor_elevation_max_m': _json_stat(refractor_values, 'max'),
        'weathering_replacement_shift_min_ms': _json_stat(
            valid_weathering_ms,
            'min',
        ),
        'weathering_replacement_shift_max_ms': _json_stat(
            valid_weathering_ms,
            'max',
        ),
        'floating_datum_shift_min_ms': _json_stat(valid_floating_ms, 'min'),
        'floating_datum_shift_max_ms': _json_stat(valid_floating_ms, 'max'),
        'flat_datum_shift_min_ms': _json_stat(valid_flat_ms, 'min'),
        'flat_datum_shift_max_ms': _json_stat(valid_flat_ms, 'max'),
        'refraction_trace_shift_min_ms': _json_stat(valid_refraction_ms, 'min'),
        'refraction_trace_shift_max_ms': _json_stat(valid_refraction_ms, 'max'),
        'refraction_trace_shift_median_ms': _json_stat(
            valid_refraction_ms,
            'median',
        ),
        'refraction_trace_shift_p95_abs_ms': _json_stat(
            np.abs(valid_refraction_ms),
            'p95',
        ),
        'refraction_trace_shift_max_abs_ms': _json_stat(
            np.abs(valid_refraction_ms),
            'max',
        ),
        'negative_refraction_shift_count': int(
            np.count_nonzero(valid_refraction_ms < -_ZERO_SHIFT_ATOL_S * 1000.0)
        ),
        'positive_refraction_shift_count': int(
            np.count_nonzero(valid_refraction_ms > _ZERO_SHIFT_ATOL_S * 1000.0)
        ),
        'zero_refraction_shift_count': int(
            np.count_nonzero(
                np.abs(valid_refraction_ms) <= _ZERO_SHIFT_ATOL_S * 1000.0
            )
        ),
        'invalid_refraction_shift_count': int(
            np.count_nonzero(~np.isfinite(refraction_trace_shift_s))
        ),
        'finite_refraction_shift_count': int(finite_refraction.shape[0]),
        'floating_datum_below_refractor_count': int(
            np.count_nonzero(trace_status == 'floating_datum_below_refractor')
        ),
        'flat_datum_below_refractor_count': int(
            np.count_nonzero(trace_status == 'flat_datum_below_refractor')
        ),
        'exceeds_max_abs_shift_count': int(
            np.count_nonzero(trace_status == 'exceeds_max_abs_shift')
        ),
        'max_abs_shift_ms': _json_optional_float(max_abs_shift_ms),
        'node_datum_status_counts': _status_counts(node_status),
        'source_datum_status_counts': _status_counts(source_status),
        'receiver_datum_status_counts': _status_counts(receiver_status),
        'trace_static_status_counts': _status_counts(trace_status),
        'sign_convention': _SIGN_CONVENTION_TEXT,
        'formula_to_floating': _FORMULA_TO_FLOATING_TEXT,
        'formula_to_flat': _FORMULA_TO_FLAT_TEXT,
    }
    _copy_cell_threshold_qc(qc, upstream_qc)
    _assert_json_safe(qc)
    return qc


def _validate_endpoint_nodes(
    endpoint_node_id: np.ndarray,
    node_pos: dict[int, int],
    *,
    name: str,
) -> None:
    missing = sorted({int(node) for node in endpoint_node_id.tolist()} - node_pos.keys())
    if missing:
        raise RefractionDatumStaticsError(
            f'{name} references unknown node_id values: {missing}'
        )


def _map_node_values(
    *,
    node_id: np.ndarray,
    node_pos: dict[int, int],
    node_values: np.ndarray,
) -> np.ndarray:
    out = np.full(node_id.shape, np.nan, dtype=np.float64)
    for index, raw_node in enumerate(node_id.tolist()):
        node_index = node_pos.get(int(raw_node))
        if node_index is not None:
            out[index] = node_values[node_index]
    return np.ascontiguousarray(out, dtype=np.float64)


def _map_trace_endpoint_values(
    *,
    node_id_sorted: np.ndarray,
    endpoint_key_sorted: np.ndarray,
    endpoint_key: np.ndarray,
    endpoint_node_id: np.ndarray,
    endpoint_values: np.ndarray,
    node_pos: dict[int, int],
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    endpoint_pos: dict[str, int] = {}
    for index, raw_key in enumerate(endpoint_key.tolist()):
        key = str(raw_key)
        if not key:
            raise RefractionDatumStaticsError(f'{name} contains an empty key')
        if key in endpoint_pos:
            raise RefractionDatumStaticsError(
                f'{name} contains duplicate endpoint key: {key}'
            )
        endpoint_pos[key] = index
    out = np.full(node_id_sorted.shape, np.nan, dtype=np.float64)
    missing_node = np.zeros(node_id_sorted.shape, dtype=bool)
    missing_endpoint = np.zeros(node_id_sorted.shape, dtype=bool)
    for index, (raw_node, raw_key) in enumerate(
        zip(node_id_sorted.tolist(), endpoint_key_sorted.tolist(), strict=True)
    ):
        node = int(raw_node)
        if node not in node_pos:
            missing_node[index] = True
            continue
        endpoint_index = endpoint_pos.get(str(raw_key))
        if endpoint_index is None:
            missing_endpoint[index] = True
            continue
        if int(endpoint_node_id[endpoint_index]) != node:
            raise RefractionDatumStaticsError(
                f'{name} endpoint key {raw_key!s} does not match node_id {node}'
            )
        out[index] = endpoint_values[endpoint_index]
    return (
        np.ascontiguousarray(out, dtype=np.float64),
        np.ascontiguousarray(missing_node, dtype=bool),
        np.ascontiguousarray(missing_endpoint, dtype=bool),
    )


def _optional_or_derived_float(
    owner: object,
    *,
    field: str,
    fallback: np.ndarray,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    value = getattr(owner, field, None)
    if value is None:
        return np.ascontiguousarray(fallback, dtype=np.float64)
    return _coerce_1d_float(
        value,
        name=f'weathering_replacement_result.{field}',
        expected_shape=expected_shape,
        allow_nonfinite=True,
    )


def _coerce_1d_float(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    allow_nonfinite: bool = False,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionDatumStaticsError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise RefractionDatumStaticsError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if np.issubdtype(arr.dtype, np.bool_) or not _is_real_numeric_dtype(arr.dtype):
        raise RefractionDatumStaticsError(f'{name} must have a real numeric dtype')
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if not allow_nonfinite and np.any(~np.isfinite(out)):
        raise RefractionDatumStaticsError(f'{name} must contain only finite values')
    return out


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
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionDatumStaticsError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise RefractionDatumStaticsError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if np.issubdtype(arr.dtype, np.bool_):
        raise RefractionDatumStaticsError(f'{name} must contain integer values')
    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr, dtype=np.int64)
    if not _is_real_numeric_dtype(arr.dtype):
        raise RefractionDatumStaticsError(f'{name} must contain integer values')
    arr_f64 = arr.astype(np.float64, copy=False)
    if np.any(~np.isfinite(arr_f64)):
        raise RefractionDatumStaticsError(f'{name} must contain finite values')
    if not np.all(arr_f64 == np.rint(arr_f64)):
        raise RefractionDatumStaticsError(f'{name} must contain integer values')
    return np.ascontiguousarray(arr_f64, dtype=np.int64)


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
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionDatumStaticsError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise RefractionDatumStaticsError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if arr.dtype != np.bool_:
        raise RefractionDatumStaticsError(f'{name} must have bool dtype')
    return np.ascontiguousarray(arr, dtype=bool)


def _coerce_1d_string(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    dtype: object = _ENDPOINT_KEY_DTYPE,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionDatumStaticsError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise RefractionDatumStaticsError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    return np.ascontiguousarray(arr.astype(dtype, copy=False))


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


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise RefractionDatumStaticsError(f'{name} must be finite')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionDatumStaticsError(f'{name} must be finite') from exc
    if not np.isfinite(out):
        raise RefractionDatumStaticsError(f'{name} must be finite')
    return out


def _csv_int(value: str, *, name: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise RefractionDatumStaticsError(f'{name} must contain an integer') from exc
    return out


def _positive_finite(value: object, *, name: str) -> float:
    out = _finite_float(value, name=name)
    if out <= 0.0:
        raise RefractionDatumStaticsError(f'{name} must be positive')
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
    raise RefractionDatumStaticsError(
        'bedrock_velocity_mode must be solve_global, fixed_global, or solve_cell'
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
        raise RefractionDatumStaticsError(f'{field} is required') from exc
    if value is None:
        raise RefractionDatumStaticsError(f'{field} is required')
    return value


def _assign_inherited_replacement_status(
    status: np.ndarray,
    inherited_status: np.ndarray,
) -> None:
    inherited = np.asarray(inherited_status).astype(str, copy=False)
    for upstream_status, datum_status in (
        _UPSTREAM_REPLACEMENT_STATUS_TO_DATUM_STATUS.items()
    ):
        _assign(status, inherited == upstream_status, datum_status)
    known = set(_UPSTREAM_REPLACEMENT_STATUS_TO_DATUM_STATUS)
    known.update(_UPSTREAM_REPLACEMENT_NON_INVALID_STATUSES)
    unknown = ~np.isin(inherited, list(known))
    _assign(status, unknown, 'invalid_weathering_replacement')


def _copy_cell_threshold_qc(payload: dict[str, Any], upstream: dict[str, Any]) -> None:
    for key in _CELL_THRESHOLD_QC_KEYS:
        if key in upstream:
            payload[key] = upstream[key]
    layer_qc = upstream.get('layers')
    if isinstance(layer_qc, dict):
        payload['layers'] = layer_qc


def _assign(status: np.ndarray, mask: np.ndarray, value: str) -> None:
    current = np.asarray(status).astype(str, copy=False)
    priority = _STATUS_PRIORITY[value]
    should_assign = np.asarray(mask, dtype=bool) & np.asarray(
        [_STATUS_PRIORITY.get(str(item), -1) <= priority for item in current],
        dtype=bool,
    )
    status[should_assign] = value


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
    raise RefractionDatumStaticsError(f'unsupported statistic: {stat}')


def _status_counts(values: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in values.tolist():
        key = str(raw)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _json_optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _node_rows(result: RefractionDatumStaticsResult) -> list[dict[str, Any]]:
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
                'floating_datum_elevation_m': _csv_float(
                    result.node_floating_datum_elevation_m[index]
                ),
                'datum_status': str(result.node_datum_status[index]),
                'weathering_status': str(result.node_weathering_status[index]),
                'pick_count': int(result.node_pick_count[index]),
                'used_pick_count': int(result.node_used_pick_count[index]),
                'residual_rms_ms': _csv_float(
                    result.node_residual_rms_s[index] * 1000.0
                ),
            }
        )
    return rows


def _source_rows(result: RefractionDatumStaticsResult) -> list[dict[str, Any]]:
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
                'source_refractor_elevation_m': _csv_float(
                    result.source_refractor_elevation_m[index]
                ),
                'source_floating_datum_elevation_m': _csv_float(
                    result.source_floating_datum_elevation_m[index]
                ),
                'source_weathering_replacement_shift_ms': _csv_float(
                    result.source_weathering_replacement_shift_s[index] * 1000.0
                ),
                'source_floating_datum_elevation_shift_ms': _csv_float(
                    result.source_floating_datum_elevation_shift_s[index] * 1000.0
                ),
                'source_flat_datum_shift_ms': _csv_float(
                    result.source_flat_datum_shift_s[index] * 1000.0
                ),
                'source_refraction_shift_ms': _csv_float(
                    result.source_refraction_shift_s[index] * 1000.0
                ),
                'source_datum_status': str(result.source_datum_status[index]),
            }
        )
    return rows


def _receiver_rows(result: RefractionDatumStaticsResult) -> list[dict[str, Any]]:
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
                'receiver_refractor_elevation_m': _csv_float(
                    result.receiver_refractor_elevation_m[index]
                ),
                'receiver_floating_datum_elevation_m': _csv_float(
                    result.receiver_floating_datum_elevation_m[index]
                ),
                'receiver_weathering_replacement_shift_ms': _csv_float(
                    result.receiver_weathering_replacement_shift_s[index] * 1000.0
                ),
                'receiver_floating_datum_elevation_shift_ms': _csv_float(
                    result.receiver_floating_datum_elevation_shift_s[index] * 1000.0
                ),
                'receiver_flat_datum_shift_ms': _csv_float(
                    result.receiver_flat_datum_shift_s[index] * 1000.0
                ),
                'receiver_refraction_shift_ms': _csv_float(
                    result.receiver_refraction_shift_s[index] * 1000.0
                ),
                'receiver_datum_status': str(result.receiver_datum_status[index]),
            }
        )
    return rows


def _insert_after(
    columns: tuple[str, ...],
    anchor: str,
    additions: tuple[str, ...],
) -> tuple[str, ...]:
    index = columns.index(anchor)
    return columns[: index + 1] + additions + columns[index + 1 :]


def _has_field_correction_composition(
    result: RefractionDatumStaticsResult,
) -> bool:
    present = (
        result.source_field_shift_s_sorted is not None,
        result.receiver_field_shift_s_sorted is not None,
        result.trace_field_shift_s_sorted is not None,
        result.trace_field_static_status_sorted is not None,
        result.base_refraction_trace_shift_s_sorted is not None,
    )
    if not any(present):
        return False
    if not all(present):
        raise RefractionDatumStaticsError(
            'field-correction composition arrays must be provided together'
        )
    return all(present)


def _source_field_shift_s_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_float_array(int(result.sorted_trace_index.shape[0]))
    return _optional_field_float_array(
        result.source_field_shift_s_sorted,
        int(result.sorted_trace_index.shape[0]),
    )


def _receiver_field_shift_s_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_float_array(int(result.sorted_trace_index.shape[0]))
    return _optional_field_float_array(
        result.receiver_field_shift_s_sorted,
        int(result.sorted_trace_index.shape[0]),
    )


def _trace_field_shift_s_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_float_array(int(result.sorted_trace_index.shape[0]))
    return _optional_field_float_array(
        result.trace_field_shift_s_sorted,
        int(result.sorted_trace_index.shape[0]),
    )


def _trace_field_static_status_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_status_array(int(result.sorted_trace_index.shape[0]))
    return _optional_field_status_array(
        result.trace_field_static_status_sorted,
        int(result.sorted_trace_index.shape[0]),
    )


def _base_refraction_trace_shift_s_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if result.base_refraction_trace_shift_s_sorted is None:
        return np.ascontiguousarray(
            result.refraction_trace_shift_s_sorted,
            dtype=np.float64,
        )
    return np.ascontiguousarray(
        result.base_refraction_trace_shift_s_sorted,
        dtype=np.float64,
    )


def _optional_field_float_array(value: object, shape: int) -> np.ndarray:
    if value is None:
        raise RefractionDatumStaticsError('field correction array is missing')
    arr = np.ascontiguousarray(value, dtype=np.float64)
    if arr.shape != (int(shape),):
        raise RefractionDatumStaticsError(
            'field correction array has unexpected shape'
        )
    return arr


def _optional_field_status_array(value: object, shape: int) -> np.ndarray:
    if value is None:
        raise RefractionDatumStaticsError(
            'field correction status array is missing'
        )
    raw = [str(item) for item in np.asarray(value).tolist()]
    max_len = max([1, *(len(item) for item in raw)])
    arr = np.ascontiguousarray(raw, dtype=f'<U{max_len}')
    if arr.shape != (int(shape),):
        raise RefractionDatumStaticsError(
            'field correction status array has unexpected shape'
        )
    return arr


def _disabled_field_float_array(shape: int) -> np.ndarray:
    return np.zeros(int(shape), dtype=np.float64)


def _disabled_field_status_array(shape: int) -> np.ndarray:
    return np.full(int(shape), _FIELD_DISABLED_STATUS, dtype='<U16')


def _total_with_field_shift_s(
    *,
    refraction_shift_s: np.ndarray,
    field_shift_s: np.ndarray,
    field_status: np.ndarray,
) -> np.ndarray:
    refraction = np.asarray(refraction_shift_s, dtype=np.float64)
    field = np.asarray(field_shift_s, dtype=np.float64)
    status = np.asarray(field_status).astype(str)
    out = np.full(refraction.shape, np.nan, dtype=np.float64)
    valid = (
        np.isin(status, tuple(_FIELD_TOTAL_VALID_STATUSES))
        & np.isfinite(refraction)
        & np.isfinite(field)
    )
    out[valid] = refraction[valid] + field[valid]
    return np.ascontiguousarray(out, dtype=np.float64)


def _final_trace_shift_s_sorted(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    return _total_with_field_shift_s(
        refraction_shift_s=_base_refraction_trace_shift_s_sorted_array(result),
        field_shift_s=_trace_field_shift_s_sorted_array(result),
        field_status=_trace_field_static_status_sorted_array(result),
    )


def _trace_preview_columns(result: RefractionDatumStaticsResult) -> tuple[str, ...]:
    columns = _insert_after(
        _TRACE_PREVIEW_COLUMNS,
        'flat_datum_shift_ms',
        (
            'source_field_shift_ms',
            'receiver_field_shift_ms',
            'trace_field_shift_ms',
        ),
    )
    return _insert_after(
        columns,
        'refraction_trace_shift_ms',
        ('final_trace_shift_ms', 'trace_field_static_status'),
    )


def _trace_preview_rows(result: RefractionDatumStaticsResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_field_shift_s = _source_field_shift_s_sorted_array(result)
    receiver_field_shift_s = _receiver_field_shift_s_sorted_array(result)
    trace_field_shift_s = _trace_field_shift_s_sorted_array(result)
    trace_field_status = _trace_field_static_status_sorted_array(result)
    base_refraction_trace_shift_s = _base_refraction_trace_shift_s_sorted_array(result)
    final_trace_shift_s = _final_trace_shift_s_sorted(result)
    for index in range(int(result.sorted_trace_index.shape[0])):
        row = {
            'sorted_trace_index': int(result.sorted_trace_index[index]),
            'valid_observation': bool(result.valid_observation_mask_sorted[index]),
            'used_observation': bool(result.used_observation_mask_sorted[index]),
            'trace_static_valid': bool(result.trace_static_valid_mask_sorted[index]),
            'source_node_id': int(result.source_node_id_sorted[index]),
            'receiver_node_id': int(result.receiver_node_id_sorted[index]),
            'source_surface_elevation_m': _csv_float(
                result.source_surface_elevation_m_sorted[index]
            ),
            'receiver_surface_elevation_m': _csv_float(
                result.receiver_surface_elevation_m_sorted[index]
            ),
            'source_floating_datum_elevation_m': _csv_float(
                result.source_floating_datum_elevation_m_sorted[index]
            ),
            'receiver_floating_datum_elevation_m': _csv_float(
                result.receiver_floating_datum_elevation_m_sorted[index]
            ),
            'weathering_replacement_trace_shift_ms': _csv_float(
                result.weathering_replacement_trace_shift_s_sorted[index] * 1000.0
            ),
            'floating_datum_elevation_shift_ms': _csv_float(
                result.floating_datum_elevation_shift_s_sorted[index] * 1000.0
            ),
            'flat_datum_shift_ms': _csv_float(
                result.flat_datum_shift_s_sorted[index] * 1000.0
            ),
            'source_field_shift_ms': _csv_float(
                source_field_shift_s[index] * 1000.0
            ),
            'receiver_field_shift_ms': _csv_float(
                receiver_field_shift_s[index] * 1000.0
            ),
            'trace_field_shift_ms': _csv_float(
                trace_field_shift_s[index] * 1000.0
            ),
            'refraction_trace_shift_ms': _csv_float(
                base_refraction_trace_shift_s[index] * 1000.0
            ),
            'final_trace_shift_ms': _csv_float(
                final_trace_shift_s[index] * 1000.0
            ),
            'trace_field_static_status': str(trace_field_status[index]),
            'trace_static_status': str(result.trace_static_status_sorted[index]),
            'estimated_first_break_time_ms': _csv_float(
                result.estimated_first_break_time_s_sorted[index] * 1000.0
            ),
            'first_break_residual_ms': _csv_float(
                result.first_break_residual_s_sorted[index] * 1000.0
            ),
        }
        rows.append(row)
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
    'REFRACTION_DATUM_NODES_CSV_NAME',
    'REFRACTION_DATUM_RECEIVERS_CSV_NAME',
    'REFRACTION_DATUM_SOURCES_CSV_NAME',
    'REFRACTION_DATUM_STATICS_QC_JSON_NAME',
    'REFRACTION_DATUM_TRACE_PREVIEW_CSV_NAME',
    'RefractionDatumStaticsError',
    'RefractionDatumStaticsResult',
    'build_refraction_datum_statics',
    'compose_refraction_trace_shift_s',
    'compute_datum_refraction_statics_from_first_breaks',
    'compute_flat_datum_shift_s',
    'compute_floating_datum_elevation_shift_s',
    'write_refraction_datum_statics_artifacts',
]
