"""Trace statics and near-surface CSV artifact writers."""

from __future__ import annotations

from pathlib import Path

from app.statics.refraction.artifacts.contract import (
    _NEAR_SURFACE_2LAYER_COLUMNS,
    _NEAR_SURFACE_3LAYER_COLUMNS,
    _NEAR_SURFACE_COLUMNS,
    _TRACE_STATICS_COLUMNS,
)
from app.statics.refraction.artifacts.field_corrections import (
    _base_refraction_trace_shift_s_sorted_array,
    _final_trace_shift_s_sorted,
    _receiver_field_shift_s_sorted_array,
    _source_field_shift_s_sorted_array,
    _trace_field_shift_s_sorted_array,
    _trace_field_static_status_sorted_array,
)
from app.statics.refraction.artifacts.formatters import (
    _csv_bool,
    _csv_float,
    _csv_ms,
)
from app.statics.refraction.artifacts.io import _write_csv_atomic
from app.statics.refraction.artifacts.static_tables import (
    _has_node_2layer_static_fields,
    _has_node_3layer_static_fields,
    _insert_after,
    _node_sh1_weathering_thickness_m,
)
from app.statics.refraction.artifacts.validation import _validate_result
from app.services.refraction_static_types import RefractionDatumStaticsResult

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
        row = (
            {
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
        )
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



__all__ = [
    'write_near_surface_model_csv',
    'write_refraction_statics_csv',
]
