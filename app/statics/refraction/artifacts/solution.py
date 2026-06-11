"""Machine-readable solution artifact builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.statics.refraction.contracts.apply import RefractionStaticApplyRequest
from app.statics.refraction.domain.types import (
    RefractionDatumStaticsResult,
    ResolvedRefractionFirstLayer,
)

from app.statics.refraction.artifacts.arrays import (
    _bool_array,
    _endpoint_cell_id_array,
    _endpoint_v2_m_s,
    _endpoint_v2_status_array,
    _float_array,
    _int_array,
    _scalar_float,
    _scalar_int,
    _scalar_str,
    _string_array,
)
from app.statics.refraction.artifacts.contract import (
    ARTIFACT_VERSION,
    METHOD,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    SIGN_CONVENTION,
)
from app.statics.refraction.artifacts.field_corrections import (
    _applied_field_shift_s_sorted_array,
    _base_refraction_trace_shift_s_sorted_array,
    _field_static_valid_mask,
    _final_trace_shift_s_sorted,
    _final_trace_static_status_sorted_array,
    _final_trace_static_valid_mask_sorted_array,
    _receiver_field_shift_s_array,
    _receiver_field_shift_s_sorted_array,
    _receiver_field_static_status_array,
    _receiver_manual_static_shift_s_array,
    _receiver_manual_static_status_array,
    _source_depth_m_array,
    _source_depth_shift_s_array,
    _source_depth_status_array,
    _source_field_shift_s_array,
    _source_field_shift_s_sorted_array,
    _source_field_static_status_array,
    _source_manual_static_shift_s_array,
    _source_manual_static_status_array,
    _source_uphole_shift_s_array,
    _source_uphole_status_array,
    _source_uphole_time_s_array,
    _total_with_field_shift_s,
    _trace_endpoint_key_sorted_array,
    _trace_field_shift_s_sorted_array,
    _trace_field_static_status_sorted_array,
)
from app.statics.refraction.artifacts.formatters import _nan_if_none
from app.statics.refraction.artifacts.io import (
    _validate_no_object_arrays,
    _write_npz_atomic,
)
from app.statics.refraction.artifacts.static_tables import (
    _has_node_2layer_static_fields,
    _has_node_3layer_static_fields,
    _has_receiver_2layer_static_fields,
    _has_receiver_3layer_static_fields,
    _has_source_2layer_static_fields,
    _has_source_3layer_static_fields,
    _node_sh1_weathering_thickness_m,
    _receiver_sh1_weathering_thickness_m,
    _source_sh1_weathering_thickness_m,
)
from app.statics.refraction.artifacts.validation import (
    _validate_resolved_first_layer,
    _validate_result,
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



__all__ = [
    'build_refraction_static_solution_arrays',
    'write_refraction_static_solution_npz',
]
