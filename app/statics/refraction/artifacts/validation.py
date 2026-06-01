"""Validation helpers for refraction static artifact writers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from app.contracts.statics.refraction.apply import RefractionStaticApplyRequest
from app.statics.refraction.artifacts.arrays import (
    _length,
    _required_cell_int_array,
    _validate_status_array,
)
from app.statics.refraction.artifacts.contract import (
    _ValidatedResult,
    RefractionStaticArtifactError,
)
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
    ResolvedRefractionFirstLayer,
)

def _validate_job_dir(job_dir: Path) -> Path:
    try:
        root = Path(job_dir)
    except TypeError as exc:
        raise RefractionStaticArtifactError('job_dir must be path-like') from exc
    if not root.exists():
        raise RefractionStaticArtifactError('missing job directory')
    if not root.is_dir():
        raise RefractionStaticArtifactError('job_dir is not a directory')
    if not os.access(root, os.W_OK):
        raise RefractionStaticArtifactError('job directory is not writable')
    return root

def _validate_result(result: RefractionDatumStaticsResult) -> _ValidatedResult:
    if not isinstance(result, RefractionDatumStaticsResult):
        raise RefractionStaticArtifactError(
            'result must be a RefractionDatumStaticsResult instance'
        )
    n_traces = _length(result.sorted_trace_index, name='sorted_trace_index')
    if n_traces <= 0:
        raise RefractionStaticArtifactError('sorted_trace_index must not be empty')
    for name in _TRACE_ARRAY_NAMES:
        if _length(getattr(result, name), name=name) != n_traces:
            raise RefractionStaticArtifactError(
                f'trace-order array length mismatch for {name}'
            )
    n_nodes = _length(result.node_id, name='node_id')
    for name in _NODE_ARRAY_NAMES:
        if _length(getattr(result, name), name=name) != n_nodes:
            raise RefractionStaticArtifactError(f'node array length mismatch for {name}')
    _validate_optional_arrays(
        result=result,
        names=_NODE_2LAYER_STATIC_ARRAY_NAMES,
        expected_length=n_nodes,
        label='node two-layer',
    )
    n_source = _length(result.source_endpoint_key, name='source_endpoint_key')
    for name in _SOURCE_ARRAY_NAMES:
        if _length(getattr(result, name), name=name) != n_source:
            raise RefractionStaticArtifactError(
                f'source endpoint array length mismatch for {name}'
            )
    _validate_optional_arrays(
        result=result,
        names=('source_depth_m', 'source_depth_shift_s', 'source_depth_status'),
        expected_length=n_source,
        label='source-depth endpoint',
    )
    _validate_optional_arrays(
        result=result,
        names=(
            'source_uphole_time_s',
            'source_uphole_shift_s',
            'source_uphole_status',
        ),
        expected_length=n_source,
        label='uphole source endpoint',
    )
    _validate_optional_arrays(
        result=result,
        names=('source_manual_static_shift_s', 'source_manual_static_status'),
        expected_length=n_source,
        label='manual static source endpoint',
    )
    _validate_optional_arrays(
        result=result,
        names=('source_field_shift_s', 'source_field_static_status'),
        expected_length=n_source,
        label='source field-composition endpoint',
    )
    _validate_optional_arrays(
        result=result,
        names=_SOURCE_2LAYER_STATIC_ARRAY_NAMES,
        expected_length=n_source,
        label='source two-layer endpoint',
    )
    n_receiver = _length(result.receiver_endpoint_key, name='receiver_endpoint_key')
    for name in _RECEIVER_ARRAY_NAMES:
        if _length(getattr(result, name), name=name) != n_receiver:
            raise RefractionStaticArtifactError(
                f'receiver endpoint array length mismatch for {name}'
            )
    _validate_optional_arrays(
        result=result,
        names=('receiver_manual_static_shift_s', 'receiver_manual_static_status'),
        expected_length=n_receiver,
        label='manual static receiver endpoint',
    )
    _validate_optional_arrays(
        result=result,
        names=('receiver_field_shift_s', 'receiver_field_static_status'),
        expected_length=n_receiver,
        label='receiver field-composition endpoint',
    )
    _validate_optional_arrays(
        result=result,
        names=_RECEIVER_2LAYER_STATIC_ARRAY_NAMES,
        expected_length=n_receiver,
        label='receiver two-layer endpoint',
    )
    n_rows = _length(result.row_trace_index_sorted, name='row_trace_index_sorted')
    for name in _ROW_ARRAY_NAMES:
        if _length(getattr(result, name), name=name) != n_rows:
            raise RefractionStaticArtifactError(
                f'residual array length mismatch for {name}'
            )
    _validate_optional_arrays(
        result=result,
        names=(
            'source_endpoint_key_sorted',
            'receiver_endpoint_key_sorted',
        ),
        expected_length=n_traces,
        label='trace endpoint key',
    )
    _validate_optional_arrays(
        result=result,
        names=(
            'source_field_shift_s_sorted',
            'receiver_field_shift_s_sorted',
            'trace_field_shift_s_sorted',
            'trace_field_static_status_sorted',
            'trace_field_static_valid_mask_sorted',
            'base_refraction_trace_shift_s_sorted',
        ),
        expected_length=n_traces,
        label='trace field-composition',
    )
    _validate_optional_arrays(
        result=result,
        names=(
            'final_trace_shift_s_sorted',
            'final_trace_static_status_sorted',
            'final_trace_static_valid_mask_sorted',
            'applied_field_shift_s_sorted',
        ),
        expected_length=n_traces,
        label='final trace field-composition',
    )
    if np.any((result.row_trace_index_sorted < 0) | (result.row_trace_index_sorted >= n_traces)):
        raise RefractionStaticArtifactError(
            'row_trace_index_sorted contains out-of-range trace indices'
        )
    for name in (
        'weathering_velocity_m_s',
        'bedrock_velocity_m_s',
        'bedrock_slowness_s_per_m',
        'replacement_slowness_delta_s_per_m',
    ):
        if not np.isfinite(float(getattr(result, name))):
            raise RefractionStaticArtifactError(f'non-finite required scalar {name}')
    for name in _STATUS_ARRAY_NAMES:
        _validate_status_array(getattr(result, name), name=name)
    if result.bedrock_velocity_mode == 'solve_cell':
        _validate_solve_cell_local_v2_arrays(
            result=result,
            n_traces=n_traces,
            n_nodes=n_nodes,
            n_source=n_source,
            n_receiver=n_receiver,
            n_rows=n_rows,
        )
    return _ValidatedResult(
        result=result,
        n_traces=n_traces,
        n_nodes=n_nodes,
        n_source_endpoints=n_source,
        n_receiver_endpoints=n_receiver,
        n_rows=n_rows,
    )

def _validate_optional_arrays(
    *,
    result: RefractionDatumStaticsResult,
    names: tuple[str, ...],
    expected_length: int,
    label: str,
) -> None:
    present = [name for name in names if getattr(result, name) is not None]
    if not present:
        return
    if len(present) != len(names):
        missing = ', '.join(name for name in names if name not in present)
        raise RefractionStaticArtifactError(
            f'{label} arrays must be provided together; missing {missing}'
        )
    for name in names:
        if _length(getattr(result, name), name=name) != expected_length:
            raise RefractionStaticArtifactError(
                f'{label} array length mismatch for {name}'
            )

def _validate_solve_cell_local_v2_arrays(
    *,
    result: RefractionDatumStaticsResult,
    n_traces: int,
    n_nodes: int,
    n_source: int,
    n_receiver: int,
    n_rows: int,
) -> None:
    expected_lengths = {
        'node_v2_cell_id': n_nodes,
        'node_v2_m_s': n_nodes,
        'node_v2_status': n_nodes,
        'source_v2_cell_id': n_source,
        'source_v2_m_s': n_source,
        'source_v2_status': n_source,
        'receiver_v2_cell_id': n_receiver,
        'receiver_v2_m_s': n_receiver,
        'receiver_v2_status': n_receiver,
        'source_v2_cell_id_sorted': n_traces,
        'source_v2_m_s_sorted': n_traces,
        'source_v2_status_sorted': n_traces,
        'receiver_v2_cell_id_sorted': n_traces,
        'receiver_v2_m_s_sorted': n_traces,
        'receiver_v2_status_sorted': n_traces,
    }
    for name, expected_length in expected_lengths.items():
        value = getattr(result, name)
        if value is None:
            raise RefractionStaticArtifactError(
                f'solve_cell result requires {name}'
            )
        if _length(value, name=name) != expected_length:
            raise RefractionStaticArtifactError(
                f'solve_cell local V2 array length mismatch for {name}'
            )
        if name.endswith('_status'):
            _validate_status_array(value, name=name)
    active_cell_id = _required_cell_int_array(
        result.active_cell_id,
        name='active_cell_id',
    )
    for name in (
        'cell_bedrock_slowness_s_per_m',
        'cell_bedrock_velocity_m_s',
        'cell_velocity_status',
    ):
        value = getattr(result, name)
        if value is None:
            raise RefractionStaticArtifactError(f'solve_cell result requires {name}')
        if _length(value, name=name) != int(active_cell_id.shape[0]):
            raise RefractionStaticArtifactError(
                f'solve_cell cell array length mismatch for {name}'
            )
        if name.endswith('_status'):
            _validate_status_array(value, name=name)
    inactive_cell_id = _required_cell_int_array(
        result.inactive_cell_id,
        name='inactive_cell_id',
    )
    if np.intersect1d(active_cell_id, inactive_cell_id).size:
        raise RefractionStaticArtifactError(
            'active_cell_id and inactive_cell_id must not overlap'
        )
    row_midpoint_cell_id = _required_cell_int_array(
        result.row_midpoint_cell_id,
        name='row_midpoint_cell_id',
    )
    if int(row_midpoint_cell_id.shape[0]) != n_rows:
        raise RefractionStaticArtifactError(
            'solve_cell row_midpoint_cell_id length mismatch'
        )

def _validate_resolved_first_layer(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None,
) -> ResolvedRefractionFirstLayer:
    expected_mode = req.model.first_layer_mode
    result_velocity = float(result.weathering_velocity_m_s)
    if resolved_first_layer is None:
        try:
            velocity = float(req.model.resolved_weathering_velocity_m_s)
        except ValueError as exc:
            raise RefractionStaticArtifactError(
                'resolved first-layer weathering velocity is required'
            ) from exc
        status = (
            'estimated'
            if expected_mode == 'estimate_direct_arrival'
            else 'resolved_constant'
        )
        resolved_first_layer = ResolvedRefractionFirstLayer(
            mode=expected_mode,
            weathering_velocity_m_s=velocity,
            status=status,
            qc={
                'v1_mode': expected_mode,
                'weathering_velocity_m_s': velocity,
                'resolved_weathering_velocity_m_s': velocity,
                'v1_status': status,
            },
        )

    if resolved_first_layer.mode != expected_mode:
        raise RefractionStaticArtifactError(
            'resolved first-layer mode does not match request'
        )
    velocity = float(resolved_first_layer.weathering_velocity_m_s)
    if not np.isfinite(velocity) or velocity <= 0.0:
        raise RefractionStaticArtifactError(
            'resolved first-layer weathering velocity must be finite and positive'
        )
    if not _velocities_close(velocity, result_velocity):
        raise RefractionStaticArtifactError(
            'resolved first-layer weathering velocity does not match result'
        )
    return resolved_first_layer

def _velocities_close(left: float, right: float) -> bool:
    return bool(
        np.isclose(
            float(left),
            float(right),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
    )

_TRACE_ARRAY_NAMES = (
    'valid_observation_mask_sorted',
    'used_observation_mask_sorted',
    'trace_static_valid_mask_sorted',
    'source_node_id_sorted',
    'receiver_node_id_sorted',
    'source_surface_elevation_m_sorted',
    'receiver_surface_elevation_m_sorted',
    'source_floating_datum_elevation_m_sorted',
    'receiver_floating_datum_elevation_m_sorted',
    'source_weathering_thickness_m_sorted',
    'receiver_weathering_thickness_m_sorted',
    'source_refractor_elevation_m_sorted',
    'receiver_refractor_elevation_m_sorted',
    'source_half_intercept_time_s_sorted',
    'receiver_half_intercept_time_s_sorted',
    'source_weathering_replacement_shift_s_sorted',
    'receiver_weathering_replacement_shift_s_sorted',
    'source_floating_datum_elevation_shift_s_sorted',
    'receiver_floating_datum_elevation_shift_s_sorted',
    'source_flat_datum_shift_s_sorted',
    'receiver_flat_datum_shift_s_sorted',
    'source_refraction_shift_s_sorted',
    'receiver_refraction_shift_s_sorted',
    'weathering_replacement_trace_shift_s_sorted',
    'floating_datum_elevation_shift_s_sorted',
    'flat_datum_shift_s_sorted',
    'refraction_trace_shift_s_sorted',
    'trace_static_status_sorted',
    'estimated_first_break_time_s_sorted',
    'first_break_residual_s_sorted',
)

_NODE_ARRAY_NAMES = (
    'node_x_m',
    'node_y_m',
    'node_surface_elevation_m',
    'node_kind',
    'node_weathering_thickness_m',
    'node_refractor_elevation_m',
    'node_half_intercept_time_s',
    'node_weathering_replacement_shift_s',
    'node_floating_datum_elevation_m',
    'node_solution_status',
    'node_datum_status',
    'node_weathering_status',
    'node_pick_count',
    'node_used_pick_count',
    'node_rejected_pick_count',
    'node_residual_rms_s',
    'node_residual_mad_s',
)

_NODE_2LAYER_STATIC_ARRAY_NAMES = (
    'node_sh1_weathering_thickness_m',
    'node_sh2_weathering_thickness_m',
)

_NODE_3LAYER_STATIC_ARRAY_NAMES = (
    'node_sh1_weathering_thickness_m',
    'node_sh2_weathering_thickness_m',
    'node_sh3_weathering_thickness_m',
)

_SOURCE_ARRAY_NAMES = (
    'source_id',
    'source_node_id',
    'source_x_m',
    'source_y_m',
    'source_surface_elevation_m',
    'source_half_intercept_time_s',
    'source_weathering_thickness_m',
    'source_refractor_elevation_m',
    'source_floating_datum_elevation_m',
    'source_weathering_replacement_shift_s',
    'source_floating_datum_elevation_shift_s',
    'source_flat_datum_shift_s',
    'source_refraction_shift_s',
    'source_datum_status',
)

_SOURCE_2LAYER_STATIC_ARRAY_NAMES = (
    'source_t2_time_s',
    'source_v3_m_s',
    'source_sh1_weathering_thickness_m',
    'source_sh2_weathering_thickness_m',
)

_SOURCE_3LAYER_STATIC_ARRAY_NAMES = (
    'source_t2_time_s',
    'source_t3_time_s',
    'source_v3_m_s',
    'source_vsub_m_s',
    'source_sh1_weathering_thickness_m',
    'source_sh2_weathering_thickness_m',
    'source_sh3_weathering_thickness_m',
)

_RECEIVER_ARRAY_NAMES = (
    'receiver_id',
    'receiver_node_id',
    'receiver_x_m',
    'receiver_y_m',
    'receiver_surface_elevation_m',
    'receiver_half_intercept_time_s',
    'receiver_weathering_thickness_m',
    'receiver_refractor_elevation_m',
    'receiver_floating_datum_elevation_m',
    'receiver_weathering_replacement_shift_s',
    'receiver_floating_datum_elevation_shift_s',
    'receiver_flat_datum_shift_s',
    'receiver_refraction_shift_s',
    'receiver_datum_status',
)

_RECEIVER_2LAYER_STATIC_ARRAY_NAMES = (
    'receiver_t2_time_s',
    'receiver_v3_m_s',
    'receiver_sh1_weathering_thickness_m',
    'receiver_sh2_weathering_thickness_m',
)

_RECEIVER_3LAYER_STATIC_ARRAY_NAMES = (
    'receiver_t2_time_s',
    'receiver_t3_time_s',
    'receiver_v3_m_s',
    'receiver_vsub_m_s',
    'receiver_sh1_weathering_thickness_m',
    'receiver_sh2_weathering_thickness_m',
    'receiver_sh3_weathering_thickness_m',
)

_ROW_ARRAY_NAMES = (
    'row_source_node_id',
    'row_receiver_node_id',
    'row_distance_m',
    'observed_pick_time_s',
    'modeled_pick_time_s',
    'residual_time_s',
    'used_row_mask',
    'rejected_by_robust_mask',
)

_STATUS_ARRAY_NAMES = (
    'trace_static_status_sorted',
    'node_solution_status',
    'node_weathering_status',
    'node_datum_status',
    'source_datum_status',
    'receiver_datum_status',
)

_FIELD_DISABLED_STATUS = 'not_enabled'
_FIELD_NOT_APPLICABLE_STATUS = 'not_applicable'
_FIELD_TOTAL_VALID_STATUSES = frozenset(
    {'ok', _FIELD_DISABLED_STATUS, _FIELD_NOT_APPLICABLE_STATUS}
)



__all__ = [
    '_validate_job_dir',
    '_validate_result',
    '_validate_resolved_first_layer',
]
