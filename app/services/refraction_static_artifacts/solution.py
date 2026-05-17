"""Core solution artifacts for final refraction statics outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
    ResolvedRefractionFirstLayer,
)
from app.services.refraction_static_artifacts.contract import (
    _NEAR_SURFACE_2LAYER_COLUMNS,
    _NEAR_SURFACE_3LAYER_COLUMNS,
    _NEAR_SURFACE_COLUMNS,
    _TRACE_STATICS_COLUMNS,
    ARTIFACT_VERSION,
    METHOD,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    SIGN_CONVENTION,
    RefractionStaticArtifactError,
)
from app.services.refraction_static_artifacts.formatters import (
    _csv_bool,
    _csv_float,
    _csv_ms,
    _float_or_nan,
    _nan_if_none,
)
from app.services.refraction_static_artifacts.io import (
    _validate_no_object_arrays,
    _write_csv_atomic,
    _write_npz_atomic,
)
from app.services.refraction_static_artifacts.validation import (
    _NODE_2LAYER_STATIC_ARRAY_NAMES,
    _RECEIVER_2LAYER_STATIC_ARRAY_NAMES,
    _SOURCE_2LAYER_STATIC_ARRAY_NAMES,
    _validate_resolved_first_layer,
    _validate_result,
)

_NODE_3LAYER_STATIC_ARRAY_NAMES = (
    'node_sh1_weathering_thickness_m',
    'node_sh2_weathering_thickness_m',
    'node_sh3_weathering_thickness_m',
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

_RECEIVER_3LAYER_STATIC_ARRAY_NAMES = (
    'receiver_t2_time_s',
    'receiver_t3_time_s',
    'receiver_v3_m_s',
    'receiver_vsub_m_s',
    'receiver_sh1_weathering_thickness_m',
    'receiver_sh2_weathering_thickness_m',
    'receiver_sh3_weathering_thickness_m',
)

_FIELD_DISABLED_STATUS = 'not_enabled'
_FIELD_NOT_APPLICABLE_STATUS = 'not_applicable'
_FIELD_TOTAL_VALID_STATUSES = frozenset(
    {'ok', _FIELD_DISABLED_STATUS, _FIELD_NOT_APPLICABLE_STATUS}
)


def write_refraction_static_solution_npz(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> None:
    """Write the compressed, pickle-free machine-readable solution artifact."""
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    payload = build_refraction_static_solution_arrays(
        result=values.result,
        req=request,
        resolved_first_layer=resolved_first_layer,
    )
    _write_npz_atomic(Path(path), payload)


def write_refraction_statics_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    rows = _trace_statics_rows(values.result)
    _write_csv_atomic(Path(path), _trace_statics_columns(values.result), rows)


def write_near_surface_model_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    rows = _near_surface_model_rows(values.result)
    _write_csv_atomic(Path(path), _near_surface_columns(values.result), rows)


def build_refraction_static_solution_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> dict[str, np.ndarray]:
    values = _validate_result(result)
    r = values.result
    first_layer = _validate_resolved_first_layer(
        result=r,
        req=req,
        resolved_first_layer=resolved_first_layer,
    )
    arrays: dict[str, np.ndarray] = {
        'artifact_version': _scalar_str(ARTIFACT_VERSION),
        'method': _scalar_str(METHOD),
        'bedrock_velocity_mode': _scalar_str(r.bedrock_velocity_mode),
        'datum_mode': _scalar_str(r.datum_mode),
        'floating_datum_mode': _scalar_str(r.floating_datum_mode),
        'sign_convention': _scalar_str(SIGN_CONVENTION),
        'n_traces': _scalar_int(values.n_traces),
        'n_nodes': _scalar_int(values.n_nodes),
        'n_source_endpoints': _scalar_int(values.n_source_endpoints),
        'n_receiver_endpoints': _scalar_int(values.n_receiver_endpoints),
        'n_valid_observations': _scalar_int(values.n_rows),
        'n_used_observations': _scalar_int(np.count_nonzero(r.used_row_mask)),
        'n_rejected_by_robust': _scalar_int(
            np.count_nonzero(r.rejected_by_robust_mask)
        ),
        'v1_mode': _scalar_str(first_layer.mode),
        'v1_weathering_velocity_m_s': _scalar_float(
            first_layer.weathering_velocity_m_s
        ),
        'weathering_velocity_m_s': _scalar_float(r.weathering_velocity_m_s),
        'resolved_weathering_velocity_m_s': _scalar_float(
            first_layer.weathering_velocity_m_s
        ),
        'bedrock_velocity_m_s': _scalar_float(r.bedrock_velocity_m_s),
        'v2_refractor_velocity_m_s': _scalar_float(r.bedrock_velocity_m_s),
        'bedrock_slowness_s_per_m': _scalar_float(r.bedrock_slowness_s_per_m),
        'replacement_slowness_delta_s_per_m': _scalar_float(
            r.replacement_slowness_delta_s_per_m
        ),
        'flat_datum_elevation_m': _scalar_float(_nan_if_none(r.flat_datum_elevation_m)),
        'max_abs_shift_ms': _scalar_float(req.apply.max_abs_shift_ms),
        'sorted_trace_index': _int_array(r.sorted_trace_index),
        'source_endpoint_key_sorted': _trace_endpoint_key_sorted_array(
            r,
            endpoint='source',
        ),
        'receiver_endpoint_key_sorted': _trace_endpoint_key_sorted_array(
            r,
            endpoint='receiver',
        ),
        'valid_observation_mask_sorted': _bool_array(
            r.valid_observation_mask_sorted
        ),
        'used_observation_mask_sorted': _bool_array(r.used_observation_mask_sorted),
        'trace_static_valid_mask_sorted': _bool_array(
            r.trace_static_valid_mask_sorted
        ),
        'source_node_id_sorted': _int_array(r.source_node_id_sorted),
        'receiver_node_id_sorted': _int_array(r.receiver_node_id_sorted),
        'source_surface_elevation_m_sorted': _float_array(
            r.source_surface_elevation_m_sorted
        ),
        'receiver_surface_elevation_m_sorted': _float_array(
            r.receiver_surface_elevation_m_sorted
        ),
        'source_floating_datum_elevation_m_sorted': _float_array(
            r.source_floating_datum_elevation_m_sorted
        ),
        'receiver_floating_datum_elevation_m_sorted': _float_array(
            r.receiver_floating_datum_elevation_m_sorted
        ),
        'source_weathering_thickness_m_sorted': _float_array(
            r.source_weathering_thickness_m_sorted
        ),
        'receiver_weathering_thickness_m_sorted': _float_array(
            r.receiver_weathering_thickness_m_sorted
        ),
        'source_refractor_elevation_m_sorted': _float_array(
            r.source_refractor_elevation_m_sorted
        ),
        'receiver_refractor_elevation_m_sorted': _float_array(
            r.receiver_refractor_elevation_m_sorted
        ),
        'source_half_intercept_time_s_sorted': _float_array(
            r.source_half_intercept_time_s_sorted
        ),
        'receiver_half_intercept_time_s_sorted': _float_array(
            r.receiver_half_intercept_time_s_sorted
        ),
        'weathering_replacement_trace_shift_s_sorted': _float_array(
            r.weathering_replacement_trace_shift_s_sorted
        ),
        'floating_datum_elevation_shift_s_sorted': _float_array(
            r.floating_datum_elevation_shift_s_sorted
        ),
        'flat_datum_shift_s_sorted': _float_array(r.flat_datum_shift_s_sorted),
        'refraction_trace_shift_s_sorted': _float_array(
            r.refraction_trace_shift_s_sorted
        ),
        'estimated_first_break_time_s_sorted': _float_array(
            r.estimated_first_break_time_s_sorted
        ),
        'first_break_residual_s_sorted': _float_array(
            r.first_break_residual_s_sorted
        ),
        'trace_static_status_sorted': _string_array(r.trace_static_status_sorted),
        'source_weathering_replacement_shift_s_sorted': _float_array(
            r.source_weathering_replacement_shift_s_sorted
        ),
        'source_v2_cell_id_sorted': _endpoint_cell_id_array(
            r.source_v2_cell_id_sorted,
            values.n_traces,
        ),
        'source_v2_m_s_sorted': _endpoint_v2_m_s(
            r.source_v2_m_s_sorted,
            shape=values.n_traces,
            scalar_v2_m_s=r.bedrock_velocity_m_s,
        ),
        'source_v2_status_sorted': _endpoint_v2_status_array(
            r.source_v2_status_sorted,
            values.n_traces,
        ),
        'receiver_weathering_replacement_shift_s_sorted': _float_array(
            r.receiver_weathering_replacement_shift_s_sorted
        ),
        'receiver_v2_cell_id_sorted': _endpoint_cell_id_array(
            r.receiver_v2_cell_id_sorted,
            values.n_traces,
        ),
        'receiver_v2_m_s_sorted': _endpoint_v2_m_s(
            r.receiver_v2_m_s_sorted,
            shape=values.n_traces,
            scalar_v2_m_s=r.bedrock_velocity_m_s,
        ),
        'receiver_v2_status_sorted': _endpoint_v2_status_array(
            r.receiver_v2_status_sorted,
            values.n_traces,
        ),
        'source_floating_datum_elevation_shift_s_sorted': _float_array(
            r.source_floating_datum_elevation_shift_s_sorted
        ),
        'receiver_floating_datum_elevation_shift_s_sorted': _float_array(
            r.receiver_floating_datum_elevation_shift_s_sorted
        ),
        'source_flat_datum_shift_s_sorted': _float_array(
            r.source_flat_datum_shift_s_sorted
        ),
        'receiver_flat_datum_shift_s_sorted': _float_array(
            r.receiver_flat_datum_shift_s_sorted
        ),
        'source_refraction_shift_s_sorted': _float_array(
            r.source_refraction_shift_s_sorted
        ),
        'receiver_refraction_shift_s_sorted': _float_array(
            r.receiver_refraction_shift_s_sorted
        ),
        'node_id': _int_array(r.node_id),
        'node_x_m': _float_array(r.node_x_m),
        'node_y_m': _float_array(r.node_y_m),
        'node_surface_elevation_m': _float_array(r.node_surface_elevation_m),
        'node_floating_datum_elevation_m': _float_array(
            r.node_floating_datum_elevation_m
        ),
        'node_refractor_elevation_m': _float_array(r.node_refractor_elevation_m),
        'node_weathering_thickness_m': _float_array(
            r.node_weathering_thickness_m
        ),
        'node_half_intercept_time_s': _float_array(
            r.node_half_intercept_time_s
        ),
        'node_weathering_replacement_shift_s': _float_array(
            r.node_weathering_replacement_shift_s
        ),
        'node_v2_cell_id': _endpoint_cell_id_array(r.node_v2_cell_id, values.n_nodes),
        'node_v2_m_s': _endpoint_v2_m_s(
            r.node_v2_m_s,
            shape=values.n_nodes,
            scalar_v2_m_s=r.bedrock_velocity_m_s,
        ),
        'node_v2_status': _endpoint_v2_status_array(
            r.node_v2_status,
            values.n_nodes,
        ),
        'node_t1_time_s': _float_array(r.node_half_intercept_time_s),
        'node_sh1_weathering_thickness_m': _node_sh1_weathering_thickness_m(r),
        'node_weathering_correction_s': _float_array(
            r.node_weathering_replacement_shift_s
        ),
        'node_solution_status': _string_array(r.node_solution_status),
        'node_weathering_status': _string_array(r.node_weathering_status),
        'node_datum_status': _string_array(r.node_datum_status),
        'node_pick_count': _int_array(r.node_pick_count),
        'node_used_pick_count': _int_array(r.node_used_pick_count),
        'node_rejected_pick_count': _int_array(r.node_rejected_pick_count),
        'node_residual_rms_s': _float_array(r.node_residual_rms_s),
        'node_residual_mad_s': _float_array(r.node_residual_mad_s),
        'source_endpoint_key': _string_array(r.source_endpoint_key),
        'source_id': _int_array(r.source_id),
        'source_node_id': _int_array(r.source_node_id),
        'source_v2_cell_id': _endpoint_cell_id_array(
            r.source_v2_cell_id,
            values.n_source_endpoints,
        ),
        'source_v2_m_s': _endpoint_v2_m_s(
            r.source_v2_m_s,
            shape=values.n_source_endpoints,
            scalar_v2_m_s=r.bedrock_velocity_m_s,
        ),
        'source_v2_status': _endpoint_v2_status_array(
            r.source_v2_status,
            values.n_source_endpoints,
        ),
        'source_x_m': _float_array(r.source_x_m),
        'source_y_m': _float_array(r.source_y_m),
        'source_surface_elevation_m': _float_array(r.source_surface_elevation_m),
        'source_floating_datum_elevation_m': _float_array(
            r.source_floating_datum_elevation_m
        ),
        'source_refractor_elevation_m': _float_array(
            r.source_refractor_elevation_m
        ),
        'source_weathering_thickness_m': _float_array(
            r.source_weathering_thickness_m
        ),
        'source_half_intercept_time_s': _float_array(
            r.source_half_intercept_time_s
        ),
        'source_t1_s': _float_array(r.source_half_intercept_time_s),
        'source_weathering_replacement_shift_s': _float_array(
            r.source_weathering_replacement_shift_s
        ),
        'source_weathering_correction_s': _float_array(
            r.source_weathering_replacement_shift_s
        ),
        'source_floating_datum_elevation_shift_s': _float_array(
            r.source_floating_datum_elevation_shift_s
        ),
        'source_flat_datum_shift_s': _float_array(r.source_flat_datum_shift_s),
        'source_refraction_shift_s': _float_array(r.source_refraction_shift_s),
        'source_datum_status': _string_array(r.source_datum_status),
        'source_sh1_m': _source_sh1_weathering_thickness_m(r),
        'receiver_endpoint_key': _string_array(r.receiver_endpoint_key),
        'receiver_id': _int_array(r.receiver_id),
        'receiver_node_id': _int_array(r.receiver_node_id),
        'receiver_v2_cell_id': _endpoint_cell_id_array(
            r.receiver_v2_cell_id,
            values.n_receiver_endpoints,
        ),
        'receiver_v2_m_s': _endpoint_v2_m_s(
            r.receiver_v2_m_s,
            shape=values.n_receiver_endpoints,
            scalar_v2_m_s=r.bedrock_velocity_m_s,
        ),
        'receiver_v2_status': _endpoint_v2_status_array(
            r.receiver_v2_status,
            values.n_receiver_endpoints,
        ),
        'receiver_x_m': _float_array(r.receiver_x_m),
        'receiver_y_m': _float_array(r.receiver_y_m),
        'receiver_surface_elevation_m': _float_array(
            r.receiver_surface_elevation_m
        ),
        'receiver_floating_datum_elevation_m': _float_array(
            r.receiver_floating_datum_elevation_m
        ),
        'receiver_refractor_elevation_m': _float_array(
            r.receiver_refractor_elevation_m
        ),
        'receiver_weathering_thickness_m': _float_array(
            r.receiver_weathering_thickness_m
        ),
        'receiver_half_intercept_time_s': _float_array(
            r.receiver_half_intercept_time_s
        ),
        'receiver_t1_s': _float_array(r.receiver_half_intercept_time_s),
        'receiver_weathering_replacement_shift_s': _float_array(
            r.receiver_weathering_replacement_shift_s
        ),
        'receiver_weathering_correction_s': _float_array(
            r.receiver_weathering_replacement_shift_s
        ),
        'receiver_floating_datum_elevation_shift_s': _float_array(
            r.receiver_floating_datum_elevation_shift_s
        ),
        'receiver_flat_datum_shift_s': _float_array(r.receiver_flat_datum_shift_s),
        'receiver_refraction_shift_s': _float_array(
            r.receiver_refraction_shift_s
        ),
        'receiver_datum_status': _string_array(r.receiver_datum_status),
        'receiver_sh1_m': _receiver_sh1_weathering_thickness_m(r),
        'row_trace_index_sorted': _int_array(r.row_trace_index_sorted),
        'row_source_node_id': _int_array(r.row_source_node_id),
        'row_receiver_node_id': _int_array(r.row_receiver_node_id),
        'row_distance_m': _float_array(r.row_distance_m),
        'observed_pick_time_s': _float_array(r.observed_pick_time_s),
        'modeled_pick_time_s': _float_array(r.modeled_pick_time_s),
        'residual_time_s': _float_array(r.residual_time_s),
        'used_row_mask': _bool_array(r.used_row_mask),
        'rejected_by_robust_mask': _bool_array(r.rejected_by_robust_mask),
    }
    source_field_shift = _source_field_shift_s_array(r)
    source_field_status = _source_field_static_status_array(r)
    receiver_field_shift = _receiver_field_shift_s_array(r)
    receiver_field_status = _receiver_field_static_status_array(r)
    trace_field_shift = _trace_field_shift_s_sorted_array(r)
    trace_field_status = _trace_field_static_status_sorted_array(r)
    arrays.update(
        {
            'source_depth_m': _source_depth_m_array(r),
            'source_depth_shift_s': _source_depth_shift_s_array(r),
            'source_depth_status': _source_depth_status_array(r),
            'source_uphole_time_s': _source_uphole_time_s_array(r),
            'source_uphole_shift_s': _source_uphole_shift_s_array(r),
            'source_uphole_status': _source_uphole_status_array(r),
            'source_manual_static_shift_s': _source_manual_static_shift_s_array(r),
            'source_manual_static_status': _source_manual_static_status_array(r),
            'receiver_manual_static_shift_s': _receiver_manual_static_shift_s_array(r),
            'receiver_manual_static_status': _receiver_manual_static_status_array(r),
            'source_field_shift_s': source_field_shift,
            'source_field_static_status': source_field_status,
            'source_total_with_field_shift_s': _total_with_field_shift_s(
                refraction_shift_s=r.source_refraction_shift_s,
                field_shift_s=source_field_shift,
                field_status=source_field_status,
            ),
            'receiver_field_shift_s': receiver_field_shift,
            'receiver_field_static_status': receiver_field_status,
            'receiver_total_with_field_shift_s': _total_with_field_shift_s(
                refraction_shift_s=r.receiver_refraction_shift_s,
                field_shift_s=receiver_field_shift,
                field_status=receiver_field_status,
            ),
            'source_field_shift_s_sorted': _source_field_shift_s_sorted_array(r),
            'receiver_field_shift_s_sorted': _receiver_field_shift_s_sorted_array(r),
            'trace_field_shift_s_sorted': trace_field_shift,
            'trace_field_static_status_sorted': trace_field_status,
            'trace_field_static_valid_mask_sorted': _field_static_valid_mask(
                shift_s=trace_field_shift,
                status=trace_field_status,
            ),
            'base_refraction_trace_shift_s_sorted': _base_refraction_trace_shift_s_sorted_array(
                r
            ),
            'final_trace_shift_s_sorted': _final_trace_shift_s_sorted(r),
            'final_trace_static_status_sorted': (
                _final_trace_static_status_sorted_array(r)
            ),
            'final_trace_static_valid_mask_sorted': (
                _final_trace_static_valid_mask_sorted_array(r)
            ),
            'applied_field_shift_s_sorted': _applied_field_shift_s_sorted_array(r),
        }
    )
    if _has_node_2layer_static_fields(r):
        assert r.node_sh2_weathering_thickness_m is not None
        node_sh1_m = _node_sh1_weathering_thickness_m(r)
        node_layer1_base = r.node_surface_elevation_m - node_sh1_m
        arrays.update(
            {
                'node_sh2_weathering_thickness_m': _float_array(
                    r.node_sh2_weathering_thickness_m
                ),
                'node_layer1_base_elevation_m': _float_array(
                    node_layer1_base
                ),
                'node_final_refractor_elevation_m': _float_array(
                    r.node_refractor_elevation_m
                ),
            }
        )
        if _has_node_3layer_static_fields(r):
            assert r.node_sh3_weathering_thickness_m is not None
            arrays.update(
                {
                    'node_sh3_weathering_thickness_m': _float_array(
                        r.node_sh3_weathering_thickness_m
                    ),
                    'node_layer2_base_elevation_m': _float_array(
                        node_layer1_base - r.node_sh2_weathering_thickness_m
                    ),
                }
            )
    if _has_source_2layer_static_fields(r):
        assert r.source_t2_time_s is not None
        assert r.source_v3_m_s is not None
        assert r.source_sh2_weathering_thickness_m is not None
        source_sh1_m = _source_sh1_weathering_thickness_m(r)
        source_layer1_base = r.source_surface_elevation_m - source_sh1_m
        arrays.update(
            {
                'source_t2_time_s': _float_array(r.source_t2_time_s),
                'source_t2_s': _float_array(r.source_t2_time_s),
                'source_v3_m_s': _float_array(r.source_v3_m_s),
                'source_sh1_weathering_thickness_m': source_sh1_m,
                'source_sh2_weathering_thickness_m': _float_array(
                    r.source_sh2_weathering_thickness_m
                ),
                'source_sh2_m': _float_array(
                    r.source_sh2_weathering_thickness_m
                ),
                'source_layer1_base_elevation_m': _float_array(
                    source_layer1_base
                ),
                'source_final_refractor_elevation_m': _float_array(
                    r.source_refractor_elevation_m
                ),
            }
        )
        if _has_source_3layer_static_fields(r):
            assert r.source_t3_time_s is not None
            assert r.source_vsub_m_s is not None
            assert r.source_sh3_weathering_thickness_m is not None
            arrays.update(
                {
                    'source_t3_time_s': _float_array(r.source_t3_time_s),
                    'source_t3_s': _float_array(r.source_t3_time_s),
                    'source_vsub_m_s': _float_array(r.source_vsub_m_s),
                    'source_sh3_weathering_thickness_m': _float_array(
                        r.source_sh3_weathering_thickness_m
                    ),
                    'source_sh3_m': _float_array(
                        r.source_sh3_weathering_thickness_m
                    ),
                    'source_layer2_base_elevation_m': _float_array(
                        source_layer1_base - r.source_sh2_weathering_thickness_m
                    ),
                }
            )
    if _has_receiver_2layer_static_fields(r):
        assert r.receiver_t2_time_s is not None
        assert r.receiver_v3_m_s is not None
        assert r.receiver_sh2_weathering_thickness_m is not None
        receiver_sh1_m = _receiver_sh1_weathering_thickness_m(r)
        receiver_layer1_base = r.receiver_surface_elevation_m - receiver_sh1_m
        arrays.update(
            {
                'receiver_t2_time_s': _float_array(r.receiver_t2_time_s),
                'receiver_t2_s': _float_array(r.receiver_t2_time_s),
                'receiver_v3_m_s': _float_array(r.receiver_v3_m_s),
                'receiver_sh1_weathering_thickness_m': receiver_sh1_m,
                'receiver_sh2_weathering_thickness_m': _float_array(
                    r.receiver_sh2_weathering_thickness_m
                ),
                'receiver_sh2_m': _float_array(
                    r.receiver_sh2_weathering_thickness_m
                ),
                'receiver_layer1_base_elevation_m': _float_array(
                    receiver_layer1_base
                ),
                'receiver_final_refractor_elevation_m': _float_array(
                    r.receiver_refractor_elevation_m
                ),
            }
        )
        if _has_receiver_3layer_static_fields(r):
            assert r.receiver_t3_time_s is not None
            assert r.receiver_vsub_m_s is not None
            assert r.receiver_sh3_weathering_thickness_m is not None
            arrays.update(
                {
                    'receiver_t3_time_s': _float_array(r.receiver_t3_time_s),
                    'receiver_t3_s': _float_array(r.receiver_t3_time_s),
                    'receiver_vsub_m_s': _float_array(r.receiver_vsub_m_s),
                    'receiver_sh3_weathering_thickness_m': _float_array(
                        r.receiver_sh3_weathering_thickness_m
                    ),
                    'receiver_sh3_m': _float_array(
                        r.receiver_sh3_weathering_thickness_m
                    ),
                    'receiver_layer2_base_elevation_m': _float_array(
                        receiver_layer1_base - r.receiver_sh2_weathering_thickness_m
                    ),
                }
            )
    _validate_no_object_arrays(arrays, artifact_name=REFRACTION_STATIC_SOLUTION_NPZ_NAME)
    return arrays


def _trace_statics_columns(
    result: RefractionDatumStaticsResult,
) -> tuple[str, ...]:
    columns = _insert_after(
        _TRACE_STATICS_COLUMNS,
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


def _trace_statics_rows(result: RefractionDatumStaticsResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_field_shift_s = _source_field_shift_s_sorted_array(result)
    receiver_field_shift_s = _receiver_field_shift_s_sorted_array(result)
    trace_field_shift_s = _trace_field_shift_s_sorted_array(result)
    trace_field_status = _trace_field_static_status_sorted_array(result)
    base_refraction_trace_shift_s = _base_refraction_trace_shift_s_sorted_array(result)
    final_trace_shift_s = _final_trace_shift_s_sorted(result)
    for index in range(int(result.sorted_trace_index.shape[0])):
        row = {
            'sorted_trace_index': int(result.sorted_trace_index[index]),
            'valid_observation': _csv_bool(result.valid_observation_mask_sorted[index]),
            'used_observation': _csv_bool(result.used_observation_mask_sorted[index]),
            'trace_static_valid': _csv_bool(result.trace_static_valid_mask_sorted[index]),
            'trace_static_status': str(result.trace_static_status_sorted[index]),
            'source_node_id': int(result.source_node_id_sorted[index]),
            'receiver_node_id': int(result.receiver_node_id_sorted[index]),
            'source_surface_elevation_m': _csv_float(result.source_surface_elevation_m_sorted[index]),
            'receiver_surface_elevation_m': _csv_float(result.receiver_surface_elevation_m_sorted[index]),
            'source_floating_datum_elevation_m': _csv_float(result.source_floating_datum_elevation_m_sorted[index]),
            'receiver_floating_datum_elevation_m': _csv_float(result.receiver_floating_datum_elevation_m_sorted[index]),
            'source_weathering_thickness_m': _csv_float(result.source_weathering_thickness_m_sorted[index]),
            'receiver_weathering_thickness_m': _csv_float(result.receiver_weathering_thickness_m_sorted[index]),
            'source_refractor_elevation_m': _csv_float(result.source_refractor_elevation_m_sorted[index]),
            'receiver_refractor_elevation_m': _csv_float(result.receiver_refractor_elevation_m_sorted[index]),
            'source_half_intercept_time_ms': _csv_ms(result.source_half_intercept_time_s_sorted[index]),
            'receiver_half_intercept_time_ms': _csv_ms(result.receiver_half_intercept_time_s_sorted[index]),
            'weathering_replacement_trace_shift_ms': _csv_ms(result.weathering_replacement_trace_shift_s_sorted[index]),
            'floating_datum_elevation_shift_ms': _csv_ms(result.floating_datum_elevation_shift_s_sorted[index]),
            'flat_datum_shift_ms': _csv_ms(result.flat_datum_shift_s_sorted[index]),
            'source_field_shift_ms': _csv_ms(source_field_shift_s[index]),
            'receiver_field_shift_ms': _csv_ms(receiver_field_shift_s[index]),
            'trace_field_shift_ms': _csv_ms(trace_field_shift_s[index]),
            'refraction_trace_shift_ms': _csv_ms(base_refraction_trace_shift_s[index]),
            'final_trace_shift_ms': _csv_ms(final_trace_shift_s[index]),
            'trace_field_static_status': str(trace_field_status[index]),
            'estimated_first_break_time_ms': _csv_ms(result.estimated_first_break_time_s_sorted[index]),
            'first_break_residual_ms': _csv_ms(result.first_break_residual_s_sorted[index]),
            'source_weathering_replacement_shift_ms': _csv_ms(result.source_weathering_replacement_shift_s_sorted[index]),
            'receiver_weathering_replacement_shift_ms': _csv_ms(result.receiver_weathering_replacement_shift_s_sorted[index]),
            'source_floating_datum_elevation_shift_ms': _csv_ms(result.source_floating_datum_elevation_shift_s_sorted[index]),
            'receiver_floating_datum_elevation_shift_ms': _csv_ms(result.receiver_floating_datum_elevation_shift_s_sorted[index]),
            'source_flat_datum_shift_ms': _csv_ms(result.source_flat_datum_shift_s_sorted[index]),
            'receiver_flat_datum_shift_ms': _csv_ms(result.receiver_flat_datum_shift_s_sorted[index]),
            'source_refraction_shift_ms': _csv_ms(result.source_refraction_shift_s_sorted[index]),
            'receiver_refraction_shift_ms': _csv_ms(result.receiver_refraction_shift_s_sorted[index]),
        }
        rows.append(row)
    return rows


def _near_surface_model_rows(result: RefractionDatumStaticsResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    node_sh1_m = _node_sh1_weathering_thickness_m(result)
    node_sh2_m = result.node_sh2_weathering_thickness_m
    node_sh3_m = result.node_sh3_weathering_thickness_m
    has_3layer_fields = _has_node_3layer_static_fields(result)
    has_2layer_fields = _has_node_2layer_static_fields(result)
    for index in range(int(result.node_id.shape[0])):
        row = {
            'node_id': int(result.node_id[index]),
            'node_kind': str(result.node_kind[index]),
            'x_m': _csv_float(result.node_x_m[index]),
            'y_m': _csv_float(result.node_y_m[index]),
            'surface_elevation_m': _csv_float(result.node_surface_elevation_m[index]),
            'floating_datum_elevation_m': _csv_float(result.node_floating_datum_elevation_m[index]),
            'refractor_elevation_m': _csv_float(result.node_refractor_elevation_m[index]),
            'weathering_thickness_m': _csv_float(result.node_weathering_thickness_m[index]),
            'half_intercept_time_ms': _csv_ms(result.node_half_intercept_time_s[index]),
            'weathering_replacement_shift_ms': _csv_ms(result.node_weathering_replacement_shift_s[index]),
            'solution_status': str(result.node_solution_status[index]),
            'weathering_status': str(result.node_weathering_status[index]),
            'datum_status': str(result.node_datum_status[index]),
            'pick_count': int(result.node_pick_count[index]),
            'used_pick_count': int(result.node_used_pick_count[index]),
            'rejected_pick_count': int(result.node_rejected_pick_count[index]),
            'residual_rms_ms': _csv_ms(result.node_residual_rms_s[index]),
            'residual_mad_ms': _csv_ms(result.node_residual_mad_s[index]),
        }
        if has_2layer_fields:
            assert node_sh2_m is not None
            layer1_base = result.node_surface_elevation_m[index] - node_sh1_m[index]
            row.update(
                {
                    'sh1_weathering_thickness_m': _csv_float(node_sh1_m[index]),
                    'sh2_weathering_thickness_m': _csv_float(node_sh2_m[index]),
                    'layer1_base_elevation_m': _csv_float(layer1_base),
                    'final_refractor_elevation_m': _csv_float(
                        result.node_refractor_elevation_m[index]
                    ),
                }
            )
            if has_3layer_fields:
                assert node_sh3_m is not None
                layer2_base = layer1_base - node_sh2_m[index]
                row.update(
                    {
                        'sh3_weathering_thickness_m': _csv_float(
                            node_sh3_m[index]
                        ),
                        'layer2_base_elevation_m': _csv_float(layer2_base),
                    }
                )
        rows.append(row)
    return rows


def _near_surface_columns(result: RefractionDatumStaticsResult) -> tuple[str, ...]:
    if _has_node_3layer_static_fields(result):
        return _NEAR_SURFACE_3LAYER_COLUMNS
    if _has_node_2layer_static_fields(result):
        return _NEAR_SURFACE_2LAYER_COLUMNS
    return _NEAR_SURFACE_COLUMNS


def _insert_after(
    columns: tuple[str, ...],
    anchor: str,
    additions: tuple[str, ...],
) -> tuple[str, ...]:
    try:
        index = columns.index(anchor)
    except ValueError as exc:
        raise RefractionStaticArtifactError(
            f'column anchor not found: {anchor}'
        ) from exc
    return columns[: index + 1] + additions + columns[index + 1 :]


def _has_source_depth_field_correction(
    result: RefractionDatumStaticsResult,
) -> bool:
    present = (
        result.source_depth_m is not None,
        result.source_depth_shift_s is not None,
        result.source_depth_status is not None,
    )
    if not any(present):
        return False
    if not all(present):
        raise RefractionStaticArtifactError(
            'source-depth field correction arrays must be provided together'
        )
    return True


def _has_uphole_field_correction(
    result: RefractionDatumStaticsResult,
) -> bool:
    present = (
        result.source_uphole_time_s is not None,
        result.source_uphole_shift_s is not None,
        result.source_uphole_status is not None,
    )
    if not any(present):
        return False
    if not all(present):
        raise RefractionStaticArtifactError(
            'uphole field correction arrays must be provided together'
        )
    return True


def _has_manual_static_field_correction(
    result: RefractionDatumStaticsResult,
) -> bool:
    present = (
        result.source_manual_static_shift_s is not None,
        result.source_manual_static_status is not None,
        result.receiver_manual_static_shift_s is not None,
        result.receiver_manual_static_status is not None,
    )
    if not any(present):
        return False
    if not all(present):
        raise RefractionStaticArtifactError(
            'manual static field correction arrays must be provided together'
        )
    return True


def _has_field_correction_composition(
    result: RefractionDatumStaticsResult,
) -> bool:
    present = (
        result.source_field_shift_s is not None,
        result.source_field_static_status is not None,
        result.receiver_field_shift_s is not None,
        result.receiver_field_static_status is not None,
        result.source_field_shift_s_sorted is not None,
        result.receiver_field_shift_s_sorted is not None,
        result.trace_field_shift_s_sorted is not None,
        result.trace_field_static_status_sorted is not None,
        result.trace_field_static_valid_mask_sorted is not None,
        result.base_refraction_trace_shift_s_sorted is not None,
        result.field_composition_qc is not None,
    )
    if not any(present):
        return False
    if not all(present):
        raise RefractionStaticArtifactError(
            'field-correction composition arrays must be provided together'
        )
    return True


def _source_depth_m_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_source_depth_field_correction(result):
        return _disabled_field_float_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.source_depth_m,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_depth_shift_s_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_source_depth_field_correction(result):
        return _disabled_field_float_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.source_depth_shift_s,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_depth_status_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_source_depth_field_correction(result):
        return _disabled_field_status_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_status_array(
        result.source_depth_status,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_uphole_time_s_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_uphole_field_correction(result):
        return _disabled_field_float_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.source_uphole_time_s,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_uphole_shift_s_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_uphole_field_correction(result):
        return _disabled_field_float_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.source_uphole_shift_s,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_uphole_status_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_uphole_field_correction(result):
        return _disabled_field_status_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_status_array(
        result.source_uphole_status,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_manual_static_shift_s_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_manual_static_field_correction(result):
        return _disabled_field_float_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.source_manual_static_shift_s,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_manual_static_status_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_manual_static_field_correction(result):
        return _disabled_field_status_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_status_array(
        result.source_manual_static_status,
        int(result.source_endpoint_key.shape[0]),
    )


def _receiver_manual_static_shift_s_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_manual_static_field_correction(result):
        return _disabled_field_float_array(int(result.receiver_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.receiver_manual_static_shift_s,
        int(result.receiver_endpoint_key.shape[0]),
    )


def _receiver_manual_static_status_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_manual_static_field_correction(result):
        return _disabled_field_status_array(int(result.receiver_endpoint_key.shape[0]))
    return _optional_field_status_array(
        result.receiver_manual_static_status,
        int(result.receiver_endpoint_key.shape[0]),
    )


def _source_field_shift_s_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_float_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.source_field_shift_s,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_field_static_status_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_status_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_status_array(
        result.source_field_static_status,
        int(result.source_endpoint_key.shape[0]),
    )


def _receiver_field_shift_s_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_float_array(int(result.receiver_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.receiver_field_shift_s,
        int(result.receiver_endpoint_key.shape[0]),
    )


def _receiver_field_static_status_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_status_array(int(result.receiver_endpoint_key.shape[0]))
    return _optional_field_status_array(
        result.receiver_field_static_status,
        int(result.receiver_endpoint_key.shape[0]),
    )


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
        return _float_array(result.refraction_trace_shift_s_sorted)
    return _float_array(result.base_refraction_trace_shift_s_sorted)


def _optional_field_float_array(value: object, shape: int) -> np.ndarray:
    if value is None:
        raise RefractionStaticArtifactError('field correction array is missing')
    arr = _float_array(value)
    if arr.shape != (int(shape),):
        raise RefractionStaticArtifactError(
            'field correction array has unexpected shape'
        )
    return arr


def _optional_field_status_array(value: object, shape: int) -> np.ndarray:
    if value is None:
        raise RefractionStaticArtifactError('field correction status array is missing')
    arr = _string_array(value)
    if arr.shape != (int(shape),):
        raise RefractionStaticArtifactError(
            'field correction status array has unexpected shape'
        )
    return arr


def _disabled_field_float_array(shape: int) -> np.ndarray:
    return np.zeros(int(shape), dtype=np.float64)


def _disabled_field_status_array(shape: int) -> np.ndarray:
    return np.full(int(shape), _FIELD_DISABLED_STATUS, dtype='<U16')


def _field_static_valid_mask(
    *,
    shift_s: np.ndarray,
    status: np.ndarray,
) -> np.ndarray:
    status_text = np.asarray(status).astype(str)
    valid_status = np.isin(status_text, tuple(_FIELD_TOTAL_VALID_STATUSES))
    return np.ascontiguousarray(valid_status & np.isfinite(shift_s), dtype=bool)


def _total_with_field_shift_s(
    *,
    refraction_shift_s: np.ndarray,
    field_shift_s: np.ndarray,
    field_status: np.ndarray,
) -> np.ndarray:
    refraction = np.asarray(refraction_shift_s, dtype=np.float64)
    field = np.asarray(field_shift_s, dtype=np.float64)
    status = np.asarray(field_status).astype(str)
    if refraction.shape != field.shape or refraction.shape != status.shape:
        raise RefractionStaticArtifactError(
            'field total shift arrays must have matching shapes'
        )
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
    if result.final_trace_shift_s_sorted is not None:
        return _float_array(result.final_trace_shift_s_sorted)
    return _total_with_field_shift_s(
        refraction_shift_s=_base_refraction_trace_shift_s_sorted_array(result),
        field_shift_s=_trace_field_shift_s_sorted_array(result),
        field_status=_trace_field_static_status_sorted_array(result),
    )


def _final_trace_static_status_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if result.final_trace_static_status_sorted is not None:
        return _string_array(result.final_trace_static_status_sorted)
    return _string_array(result.trace_static_status_sorted)


def _final_trace_static_valid_mask_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if result.final_trace_static_valid_mask_sorted is not None:
        return _bool_array(result.final_trace_static_valid_mask_sorted)
    return _bool_array(result.trace_static_valid_mask_sorted)


def _applied_field_shift_s_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if result.applied_field_shift_s_sorted is not None:
        return _float_array(result.applied_field_shift_s_sorted)
    return np.zeros(int(result.sorted_trace_index.shape[0]), dtype=np.float64)


def _trace_endpoint_key_sorted_array(
    result: RefractionDatumStaticsResult,
    *,
    endpoint: str,
) -> np.ndarray:
    if endpoint == 'source':
        raw = result.source_endpoint_key_sorted
    elif endpoint == 'receiver':
        raw = result.receiver_endpoint_key_sorted
    else:
        raise RefractionStaticArtifactError(f'unsupported endpoint kind: {endpoint}')

    expected_shape = result.sorted_trace_index.shape
    if raw is None:
        raise RefractionStaticArtifactError(f'{endpoint}_endpoint_key_sorted is required')
    out = _string_array(raw)
    if out.shape != expected_shape:
        raise RefractionStaticArtifactError(
            f'{endpoint}_endpoint_key_sorted shape mismatch'
        )
    return out


def _has_node_3layer_static_fields(result: RefractionDatumStaticsResult) -> bool:
    return all(
        getattr(result, name) is not None for name in _NODE_3LAYER_STATIC_ARRAY_NAMES
    )


def _has_node_2layer_static_fields(result: RefractionDatumStaticsResult) -> bool:
    return all(
        getattr(result, name) is not None for name in _NODE_2LAYER_STATIC_ARRAY_NAMES
    )


def _has_source_3layer_static_fields(result: RefractionDatumStaticsResult) -> bool:
    return all(
        getattr(result, name) is not None for name in _SOURCE_3LAYER_STATIC_ARRAY_NAMES
    )


def _has_source_2layer_static_fields(result: RefractionDatumStaticsResult) -> bool:
    return all(
        getattr(result, name) is not None for name in _SOURCE_2LAYER_STATIC_ARRAY_NAMES
    )


def _has_receiver_3layer_static_fields(result: RefractionDatumStaticsResult) -> bool:
    return all(
        getattr(result, name) is not None
        for name in _RECEIVER_3LAYER_STATIC_ARRAY_NAMES
    )


def _has_receiver_2layer_static_fields(result: RefractionDatumStaticsResult) -> bool:
    return all(
        getattr(result, name) is not None
        for name in _RECEIVER_2LAYER_STATIC_ARRAY_NAMES
    )


def _node_sh1_weathering_thickness_m(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    node_sh1 = result.node_sh1_weathering_thickness_m
    if node_sh1 is not None:
        return _float_array(node_sh1)
    node_sh2 = result.node_sh2_weathering_thickness_m
    if node_sh2 is None:
        return _float_array(result.node_weathering_thickness_m)
    raise RefractionStaticArtifactError(
        'node_sh1_weathering_thickness_m is required with '
        'node_sh2_weathering_thickness_m'
    )


def _source_sh1_weathering_thickness_m(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    source_sh1 = result.source_sh1_weathering_thickness_m
    if source_sh1 is not None:
        return _float_array(source_sh1)
    source_sh2 = result.source_sh2_weathering_thickness_m
    if source_sh2 is None:
        return _float_array(result.source_weathering_thickness_m)
    raise RefractionStaticArtifactError(
        'source_sh1_weathering_thickness_m is required with '
        'source_sh2_weathering_thickness_m'
    )


def _receiver_sh1_weathering_thickness_m(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    receiver_sh1 = result.receiver_sh1_weathering_thickness_m
    if receiver_sh1 is not None:
        return _float_array(receiver_sh1)
    receiver_sh2 = result.receiver_sh2_weathering_thickness_m
    if receiver_sh2 is None:
        return _float_array(result.receiver_weathering_thickness_m)
    raise RefractionStaticArtifactError(
        'receiver_sh1_weathering_thickness_m is required with '
        'receiver_sh2_weathering_thickness_m'
    )


def _scalar_str(value: object) -> np.ndarray:
    text = '' if value is None else str(value)
    return np.asarray(text, dtype=f'<U{max(1, len(text))}')


def _scalar_int(value: object) -> np.ndarray:
    return np.asarray(int(value), dtype=np.int64)


def _scalar_float(value: object) -> np.ndarray:
    return np.asarray(float(value), dtype=np.float64)


def _int_array(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.int64)


def _float_array(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64)


def _filled_float_array(value: object, shape: int) -> np.ndarray:
    return np.full(int(shape), float(value), dtype=np.float64)


def _endpoint_v2_m_s(
    value: object,
    *,
    shape: int,
    scalar_v2_m_s: float,
) -> np.ndarray:
    if value is None:
        return _filled_float_array(scalar_v2_m_s, shape)
    return np.ascontiguousarray(value, dtype=np.float64)


def _endpoint_cell_id_array(value: object, shape: int) -> np.ndarray:
    if value is None:
        return np.full(int(shape), -1, dtype=np.int64)
    return np.ascontiguousarray(value, dtype=np.int64)


def _cell_id_float_array(value: object) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).copy()
    out[out < 0] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)


def _endpoint_v2_status_array(value: object, shape: int) -> np.ndarray:
    if value is None:
        return _string_array(np.full(int(shape), 'ok', dtype='<U2'))
    return _string_array(value)


def _sum_float_arrays(left: object, right: object) -> np.ndarray:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    out = np.full(left_arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(left_arr) & np.isfinite(right_arr)
    out[finite] = left_arr[finite] + right_arr[finite]
    return np.ascontiguousarray(out, dtype=np.float64)


def _bool_array(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=bool)


def _string_array(value: object) -> np.ndarray:
    raw = [str(item) for item in np.asarray(value).tolist()]
    max_len = max([1, *(len(item) for item in raw)])
    return np.ascontiguousarray(raw, dtype=f'<U{max_len}')


def _sum_correction_s(left: object, right: object) -> float:
    left_value = _float_or_nan(left)
    right_value = _float_or_nan(right)
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return float('nan')
    return float(left_value + right_value)


__all__ = [
    'build_refraction_static_solution_arrays',
    'write_near_surface_model_csv',
    'write_refraction_static_solution_npz',
    'write_refraction_statics_csv',
]
