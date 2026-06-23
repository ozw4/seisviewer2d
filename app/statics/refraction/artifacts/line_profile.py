"""Line-profile QC artifacts for refraction statics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from app.statics.refraction.contracts.apply import RefractionStaticApplyRequest
from app.statics.refraction.artifacts.contract import (
    _LINE_PROFILE_QC_COLUMNS,
    ARTIFACT_VERSION,
    LINE_PROFILE_QC_SCHEMA_VERSION,
    REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_JSON_NAME,
    REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
    SIGN_CONVENTION,
    WORKFLOW,
    RefractionStaticArtifactError,
)
from app.statics.refraction.artifacts.first_break import (
    _residual_row_layer_context,
    _residual_row_string_context,
)
from app.statics.refraction.artifacts.validation import _validate_result
from app.statics.refraction.artifacts.field_corrections import (
    _receiver_field_shift_s_array,
    _receiver_field_static_status_array,
    _receiver_manual_static_shift_s_array,
    _source_field_shift_s_array,
    _source_field_static_status_array,
    _source_manual_static_shift_s_array,
    _total_with_field_shift_s,
)
from app.statics.refraction.artifacts.formatters import (
    _csv_float,
    _csv_int,
    _csv_json_object,
    _float_or_nan,
)
from app.statics.refraction.artifacts.io import (
    _assert_strict_json,
    _validate_no_object_arrays,
    _write_csv_atomic,
    _write_json_atomic,
    _write_npz_atomic,
)
from app.statics.refraction.artifacts.stats import _stat, _status_counts
from seis_statics.refraction.cell_coordinates import (
    project_refraction_cell_points,
    refraction_cell_coordinate_metadata_from_config,
)
from seis_statics.refraction.status import (
    classify_refraction_endpoint_static_status,
)
from app.statics.refraction.contracts.core_options import (
    refractor_cell_options_from_request,
)
from app.statics.refraction.contracts.result_types import RefractionDatumStaticsResult

_LINE_PROFILE_STRING_COLUMNS = frozenset(
    {'endpoint_kind', 'endpoint_key', 'static_status', 'solution_status'}
)
_LINE_PROFILE_INT_COLUMNS = frozenset(
    {'node_id', 'pick_count', 'used_pick_count'}
)


def write_refraction_line_profile_qc_artifacts(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    source_csv_path: Path,
    receiver_csv_path: Path,
    combined_csv_path: Path,
    npz_path: Path,
    json_path: Path,
) -> dict[str, Any]:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    arrays = build_refraction_line_profile_qc_arrays(
        result=values.result,
        req=request,
    )
    _write_csv_atomic(
        Path(source_csv_path),
        _LINE_PROFILE_QC_COLUMNS,
        _line_profile_qc_rows(arrays, endpoint_kind='source'),
    )
    _write_csv_atomic(
        Path(receiver_csv_path),
        _LINE_PROFILE_QC_COLUMNS,
        _line_profile_qc_rows(arrays, endpoint_kind='receiver'),
    )
    _write_csv_atomic(
        Path(combined_csv_path),
        _LINE_PROFILE_QC_COLUMNS,
        _line_profile_qc_rows(arrays),
    )
    _write_npz_atomic(Path(npz_path), arrays)
    payload = build_refraction_line_profile_qc_payload(
        result=values.result,
        req=request,
        arrays=arrays,
    )
    _write_json_atomic(Path(json_path), payload)
    return payload


def build_refraction_line_profile_qc_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, np.ndarray]:
    """Build combined source/receiver endpoint arrays for line-profile QC."""
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    if not _line_profile_qc_available(request):
        return _empty_line_profile_qc_arrays()

    source = _line_profile_endpoint_arrays(
        values.result,
        request,
        endpoint='source',
    )
    receiver = _line_profile_endpoint_arrays(
        values.result,
        request,
        endpoint='receiver',
    )
    arrays = {
        column: np.concatenate((source[column], receiver[column]))
        for column in _LINE_PROFILE_QC_COLUMNS
    }
    order = np.lexsort(
        (
            np.asarray(arrays['endpoint_key']).astype(str, copy=False),
            np.asarray(arrays['inline_m'], dtype=np.float64),
            np.asarray(arrays['endpoint_kind']).astype(str, copy=False),
        )
    )
    out = {
        column: np.ascontiguousarray(np.asarray(values)[order])
        for column, values in arrays.items()
    }
    _validate_no_object_arrays(out, artifact_name=REFRACTION_LINE_PROFILE_QC_NPZ_NAME)
    return out


def build_refraction_line_profile_qc_payload(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    profile_arrays = (
        build_refraction_line_profile_qc_arrays(result=values.result, req=request)
        if arrays is None
        else arrays
    )
    endpoint_kind = np.asarray(profile_arrays['endpoint_kind']).astype(str, copy=False)
    inline_m = np.asarray(profile_arrays['inline_m'], dtype=np.float64)
    crossline_m = np.asarray(profile_arrays['crossline_m'], dtype=np.float64)
    status = 'available' if _line_profile_qc_available(request) else 'unavailable'
    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'schema_version': LINE_PROFILE_QC_SCHEMA_VERSION,
        'kind': 'refraction_line_profile_qc',
        'workflow': WORKFLOW,
        'status': status,
        'availability_reason': _line_profile_qc_availability_reason(request),
        'sign_convention': SIGN_CONVENTION,
        **_line_profile_coordinate_metadata(request),
        'sort_order': ['endpoint_kind', 'inline_m', 'endpoint_key'],
        'columns': list(_LINE_PROFILE_QC_COLUMNS),
        'row_count': int(endpoint_kind.shape[0]),
        'source_row_count': int(np.count_nonzero(endpoint_kind == 'source')),
        'receiver_row_count': int(np.count_nonzero(endpoint_kind == 'receiver')),
        'endpoint_kind_counts': _status_counts(endpoint_kind),
        'static_status_counts': _status_counts(profile_arrays['static_status']),
        'solution_status_counts': _status_counts(profile_arrays['solution_status']),
        'inline_m_min': _stat(inline_m, 'min'),
        'inline_m_max': _stat(inline_m, 'max'),
        'crossline_m_min': _stat(crossline_m, 'min'),
        'crossline_m_max': _stat(crossline_m, 'max'),
        'artifacts': {
            'source_csv': REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
            'receiver_csv': REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
            'combined_csv': REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
            'npz': REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
            'json': REFRACTION_LINE_PROFILE_QC_JSON_NAME,
        },
    }
    _assert_strict_json(payload, artifact_name=REFRACTION_LINE_PROFILE_QC_JSON_NAME)
    return payload


def _line_profile_qc_available(req: RefractionStaticApplyRequest) -> bool:
    refractor_cell = req.model.refractor_cell
    return (
        refractor_cell is not None
        and refractor_cell.coordinate_mode == 'line_2d_projected'
    )


def _line_profile_qc_availability_reason(
    req: RefractionStaticApplyRequest,
) -> str:
    refractor_cell = req.model.refractor_cell
    if refractor_cell is None:
        return 'no_projected_inline_coordinate_model'
    if refractor_cell.coordinate_mode != 'line_2d_projected':
        return 'projected_inline_coordinates_unavailable_for_grid_3d'
    return 'line_2d_projected'


def _line_profile_coordinate_metadata(
    req: RefractionStaticApplyRequest,
) -> dict[str, Any]:
    refractor_cell = req.model.refractor_cell
    if refractor_cell is None:
        return {
            'coordinate_mode': 'grid_3d',
            'line_origin_x_m': None,
            'line_origin_y_m': None,
            'line_azimuth_deg': None,
        }
    return refraction_cell_coordinate_metadata_from_config(
        refractor_cell_options_from_request(refractor_cell)
    )


def _empty_line_profile_qc_arrays() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for column in _LINE_PROFILE_QC_COLUMNS:
        if column in _LINE_PROFILE_STRING_COLUMNS:
            arrays[column] = np.asarray([], dtype='<U1')
        elif column in _LINE_PROFILE_INT_COLUMNS:
            arrays[column] = np.asarray([], dtype=np.int64)
        else:
            arrays[column] = np.asarray([], dtype=np.float64)
    return arrays


def _line_profile_endpoint_arrays(
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    *,
    endpoint: str,
) -> dict[str, np.ndarray]:
    if endpoint == 'source':
        endpoint_key = _string_array(result.source_endpoint_key)
        node_id = _int_array(result.source_node_id)
        x_m = _float_array(result.source_x_m)
        y_m = _float_array(result.source_y_m)
        surface_elevation_m = _float_array(result.source_surface_elevation_m)
        t1_s = _float_array(result.source_half_intercept_time_s)
        t2_s = _optional_line_profile_float_array(
            result.source_t2_time_s,
            shape=endpoint_key.shape[0],
        )
        t3_s = _optional_line_profile_float_array(
            result.source_t3_time_s,
            shape=endpoint_key.shape[0],
        )
        v2_m_s = _endpoint_v2_m_s(
            result.source_v2_m_s,
            shape=endpoint_key.shape[0],
            scalar_v2_m_s=result.bedrock_velocity_m_s,
        )
        v3_m_s = _optional_line_profile_float_array(
            result.source_v3_m_s,
            shape=endpoint_key.shape[0],
        )
        vsub_m_s = _optional_line_profile_float_array(
            result.source_vsub_m_s,
            shape=endpoint_key.shape[0],
        )
        sh1_m = _source_sh1_weathering_thickness_m(result)
        sh2_m = _optional_line_profile_float_array(
            result.source_sh2_weathering_thickness_m,
            shape=endpoint_key.shape[0],
        )
        sh3_m = _optional_line_profile_float_array(
            result.source_sh3_weathering_thickness_m,
            shape=endpoint_key.shape[0],
        )
        weathering_correction_s = _float_array(
            result.source_weathering_replacement_shift_s
        )
        elevation_correction_s = _sum_float_arrays(
            result.source_floating_datum_elevation_shift_s,
            result.source_flat_datum_shift_s,
        )
        field_correction_s = _source_field_shift_s_array(result)
        manual_static_s = _source_manual_static_shift_s_array(result)
        total_static_s = _float_array(result.source_refraction_shift_s)
        source_field_shift_s = field_correction_s
        receiver_field_shift_s = np.full_like(field_correction_s, np.nan)
        source_total_with_field_shift_s = _total_with_field_shift_s(
            refraction_shift_s=result.source_refraction_shift_s,
            field_shift_s=_source_field_shift_s_array(result),
            field_status=_source_field_static_status_array(result),
        )
        receiver_total_with_field_shift_s = np.full_like(
            source_total_with_field_shift_s,
            np.nan,
        )
        static_status = _source_static_status_array(result)
        kind = 'source'
    elif endpoint == 'receiver':
        endpoint_key = _string_array(result.receiver_endpoint_key)
        node_id = _int_array(result.receiver_node_id)
        x_m = _float_array(result.receiver_x_m)
        y_m = _float_array(result.receiver_y_m)
        surface_elevation_m = _float_array(result.receiver_surface_elevation_m)
        t1_s = _float_array(result.receiver_half_intercept_time_s)
        t2_s = _optional_line_profile_float_array(
            result.receiver_t2_time_s,
            shape=endpoint_key.shape[0],
        )
        t3_s = _optional_line_profile_float_array(
            result.receiver_t3_time_s,
            shape=endpoint_key.shape[0],
        )
        v2_m_s = _endpoint_v2_m_s(
            result.receiver_v2_m_s,
            shape=endpoint_key.shape[0],
            scalar_v2_m_s=result.bedrock_velocity_m_s,
        )
        v3_m_s = _optional_line_profile_float_array(
            result.receiver_v3_m_s,
            shape=endpoint_key.shape[0],
        )
        vsub_m_s = _optional_line_profile_float_array(
            result.receiver_vsub_m_s,
            shape=endpoint_key.shape[0],
        )
        sh1_m = _receiver_sh1_weathering_thickness_m(result)
        sh2_m = _optional_line_profile_float_array(
            result.receiver_sh2_weathering_thickness_m,
            shape=endpoint_key.shape[0],
        )
        sh3_m = _optional_line_profile_float_array(
            result.receiver_sh3_weathering_thickness_m,
            shape=endpoint_key.shape[0],
        )
        weathering_correction_s = _float_array(
            result.receiver_weathering_replacement_shift_s
        )
        elevation_correction_s = _sum_float_arrays(
            result.receiver_floating_datum_elevation_shift_s,
            result.receiver_flat_datum_shift_s,
        )
        field_correction_s = _receiver_field_shift_s_array(result)
        manual_static_s = _receiver_manual_static_shift_s_array(result)
        total_static_s = _float_array(result.receiver_refraction_shift_s)
        source_field_shift_s = np.full_like(field_correction_s, np.nan)
        receiver_field_shift_s = field_correction_s
        source_total_with_field_shift_s = np.full_like(field_correction_s, np.nan)
        receiver_total_with_field_shift_s = _total_with_field_shift_s(
            refraction_shift_s=result.receiver_refraction_shift_s,
            field_shift_s=_receiver_field_shift_s_array(result),
            field_status=_receiver_field_static_status_array(result),
        )
        static_status = _receiver_static_status_array(result)
        kind = 'receiver'
    else:
        raise RefractionStaticArtifactError(f'unsupported endpoint kind: {endpoint}')

    projected = _line_profile_projected_coordinates(
        req=req,
        x_m=x_m,
        y_m=y_m,
    )
    node_context = _node_context(result)
    layer1_base = _line_profile_layer1_base_elevation(
        surface_elevation_m=surface_elevation_m,
        sh1_m=sh1_m,
    )
    layer2_base = _line_profile_layer2_base_elevation(
        layer1_base_elevation_m=layer1_base,
        sh2_m=sh2_m,
        has_3layer=bool(np.any(np.isfinite(sh3_m))),
    )
    n_endpoints = int(endpoint_key.shape[0])
    return {
        'endpoint_kind': _string_array(np.full(n_endpoints, kind, dtype=f'<U{len(kind)}')),
        'endpoint_key': endpoint_key,
        'node_id': node_id,
        'inline_m': projected['inline_m'],
        'crossline_m': projected['crossline_m'],
        'x_m': x_m,
        'y_m': y_m,
        'surface_elevation_m': surface_elevation_m,
        'pick_count': _line_profile_node_int_values(
            node_id,
            node_context['pick_count'],
        ),
        'used_pick_count': _line_profile_node_int_values(
            node_id,
            node_context['used_pick_count'],
        ),
        'residual_rms_ms': _line_profile_seconds_to_ms(
            _line_profile_node_float_values(node_id, node_context['residual_rms'])
        ),
        'residual_mad_ms': _line_profile_seconds_to_ms(
            _line_profile_node_float_values(node_id, node_context['residual_mad'])
        ),
        'v1_m_s': _filled_float_array(result.weathering_velocity_m_s, n_endpoints),
        'v2_m_s': v2_m_s,
        'v3_m_s': v3_m_s,
        'vsub_m_s': vsub_m_s,
        't1_ms': _line_profile_seconds_to_ms(t1_s),
        't2_ms': _line_profile_seconds_to_ms(t2_s),
        't3_ms': _line_profile_seconds_to_ms(t3_s),
        'sh1_m': sh1_m,
        'sh2_m': sh2_m,
        'sh3_m': sh3_m,
        'layer1_base_elevation_m': layer1_base,
        'layer2_base_elevation_m': layer2_base,
        'final_refractor_elevation_m': _float_array(
            result.source_refractor_elevation_m
            if endpoint == 'source'
            else result.receiver_refractor_elevation_m
        ),
        'weathering_correction_ms': _line_profile_seconds_to_ms(
            weathering_correction_s
        ),
        'elevation_correction_ms': _line_profile_seconds_to_ms(
            elevation_correction_s
        ),
        'source_field_shift_ms': _line_profile_seconds_to_ms(source_field_shift_s),
        'receiver_field_shift_ms': _line_profile_seconds_to_ms(
            receiver_field_shift_s
        ),
        'field_correction_ms': _line_profile_seconds_to_ms(field_correction_s),
        'manual_static_shift_ms': _line_profile_seconds_to_ms(manual_static_s),
        'manual_static_ms': _line_profile_seconds_to_ms(manual_static_s),
        'total_static_ms': _line_profile_seconds_to_ms(total_static_s),
        'total_applied_shift_ms': _line_profile_seconds_to_ms(total_static_s),
        'source_total_with_field_shift_ms': _line_profile_seconds_to_ms(
            source_total_with_field_shift_s
        ),
        'receiver_total_with_field_shift_ms': _line_profile_seconds_to_ms(
            receiver_total_with_field_shift_s
        ),
        'static_status': _string_array(static_status),
        'solution_status': _line_profile_node_string_values(
            node_id,
            node_context['solution_status'],
            default='missing_solution',
        ),
    }


def _line_profile_projected_coordinates(
    *,
    req: RefractionStaticApplyRequest,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> dict[str, np.ndarray]:
    refractor_cell = req.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'line-profile QC requires model.refractor_cell'
        )
    projected = project_refraction_cell_points(
        x_m=x_m,
        y_m=y_m,
        mode=refractor_cell.coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    if (
        projected.projected_inline_m is None
        or projected.projected_crossline_m is None
    ):
        raise RefractionStaticArtifactError(
            'line-profile QC requires projected inline/crossline coordinates'
        )
    return {
        'inline_m': np.ascontiguousarray(
            projected.projected_inline_m,
            dtype=np.float64,
        ),
        'crossline_m': np.ascontiguousarray(
            projected.projected_crossline_m,
            dtype=np.float64,
        ),
    }


def _optional_line_profile_float_array(
    value: object,
    *,
    shape: int,
) -> np.ndarray:
    if value is None:
        return np.full(int(shape), np.nan, dtype=np.float64)
    array = _float_array(value)
    if array.shape != (int(shape),):
        raise RefractionStaticArtifactError(
            'line-profile optional endpoint array length mismatch'
        )
    return array


def _line_profile_seconds_to_ms(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64) * 1000.0)


def _line_profile_layer1_base_elevation(
    *,
    surface_elevation_m: np.ndarray,
    sh1_m: np.ndarray,
) -> np.ndarray:
    surface = np.asarray(surface_elevation_m, dtype=np.float64)
    sh1 = np.asarray(sh1_m, dtype=np.float64)
    out = np.full(surface.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(surface) & np.isfinite(sh1)
    out[finite] = surface[finite] - sh1[finite]
    return np.ascontiguousarray(out, dtype=np.float64)


def _line_profile_layer2_base_elevation(
    *,
    layer1_base_elevation_m: np.ndarray,
    sh2_m: np.ndarray,
    has_3layer: bool,
) -> np.ndarray:
    out = np.full(np.asarray(layer1_base_elevation_m).shape, np.nan, dtype=np.float64)
    if not has_3layer:
        return out
    layer1 = np.asarray(layer1_base_elevation_m, dtype=np.float64)
    sh2 = np.asarray(sh2_m, dtype=np.float64)
    finite = np.isfinite(layer1) & np.isfinite(sh2)
    out[finite] = layer1[finite] - sh2[finite]
    return np.ascontiguousarray(out, dtype=np.float64)


def _line_profile_node_int_values(
    node_id: np.ndarray,
    lookup: Mapping[int, Any],
) -> np.ndarray:
    out = np.full(np.asarray(node_id).shape, -1, dtype=np.int64)
    for index, raw_node_id in enumerate(np.asarray(node_id).tolist()):
        value = lookup.get(int(raw_node_id))
        if value is not None:
            out[index] = int(value)
    return np.ascontiguousarray(out, dtype=np.int64)


def _line_profile_node_float_values(
    node_id: np.ndarray,
    lookup: Mapping[int, Any],
) -> np.ndarray:
    out = np.full(np.asarray(node_id).shape, np.nan, dtype=np.float64)
    for index, raw_node_id in enumerate(np.asarray(node_id).tolist()):
        value = lookup.get(int(raw_node_id))
        if value is not None:
            out[index] = _float_or_nan(value)
    return np.ascontiguousarray(out, dtype=np.float64)


def _line_profile_node_string_values(
    node_id: np.ndarray,
    lookup: Mapping[int, Any],
    *,
    default: str,
) -> np.ndarray:
    values = [
        str(lookup.get(int(raw_node_id), default))
        for raw_node_id in np.asarray(node_id).tolist()
    ]
    return _string_array(values)


def _line_profile_qc_rows(
    arrays: Mapping[str, np.ndarray],
    *,
    endpoint_kind: str | None = None,
) -> list[dict[str, object]]:
    kind = np.asarray(arrays['endpoint_kind']).astype(str, copy=False)
    selected = np.arange(kind.shape[0], dtype=np.int64)
    if endpoint_kind is not None:
        selected = selected[kind == endpoint_kind]
    rows: list[dict[str, object]] = []
    for index in selected.tolist():
        rows.append(
            {
                column: _line_profile_csv_value(column, arrays[column][index])
                for column in _LINE_PROFILE_QC_COLUMNS
            }
        )
    return rows


def _line_profile_csv_value(column: str, value: object) -> object:
    if column in _LINE_PROFILE_STRING_COLUMNS:
        return str(value)
    if column in _LINE_PROFILE_INT_COLUMNS:
        out = _csv_int(value)
        return '' if out != '' and int(out) < 0 else out
    return _csv_float(value)


def _node_lookup(node_id: np.ndarray, values: np.ndarray) -> dict[int, Any]:
    return {
        int(raw_node): values[index]
        for index, raw_node in enumerate(np.asarray(node_id).tolist())
    }


def _endpoint_node_values(
    endpoint_node_id: np.ndarray,
    node_id: np.ndarray,
    node_values: np.ndarray,
) -> np.ndarray:
    lookup = _node_lookup(node_id, node_values)
    out = np.full(np.asarray(endpoint_node_id).shape, np.nan, dtype=np.float64)
    for index, raw_node in enumerate(np.asarray(endpoint_node_id).tolist()):
        value = lookup.get(int(raw_node))
        if value is not None:
            out[index] = _float_or_nan(value)
    return np.ascontiguousarray(out, dtype=np.float64)


def _node_context(result: RefractionDatumStaticsResult) -> dict[str, dict[int, Any]]:
    return {
        'solution_status': _node_lookup(result.node_id, result.node_solution_status),
        'weathering_status': _node_lookup(result.node_id, result.node_weathering_status),
        't1_s': _node_lookup(result.node_id, result.node_half_intercept_time_s),
        'weathering_thickness': _node_lookup(
            result.node_id,
            result.node_weathering_thickness_m,
        ),
        'weathering_correction': _node_lookup(
            result.node_id,
            result.node_weathering_replacement_shift_s,
        ),
        'pick_count': _node_lookup(result.node_id, result.node_pick_count),
        'used_pick_count': _node_lookup(result.node_id, result.node_used_pick_count),
        'residual_rms': _node_lookup(result.node_id, result.node_residual_rms_s),
        'residual_mad': _node_lookup(result.node_id, result.node_residual_mad_s),
    }


def _endpoint_layer_qc_context(
    result: RefractionDatumStaticsResult,
    *,
    endpoint: str,
) -> dict[str, dict[str, dict[str, int | float]]]:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    layer_kind, _layer_index = _residual_row_layer_context(result)
    endpoint_field = (
        'row_source_endpoint_key'
        if endpoint == 'source'
        else 'row_receiver_endpoint_key'
    )
    endpoint_key = _residual_row_string_context(result, endpoint_field)
    used = np.asarray(result.used_row_mask, dtype=bool)
    residual_s = np.asarray(result.residual_time_s, dtype=np.float64)
    context: dict[str, dict[str, dict[str, Any]]] = {
        'pick_count': {},
        'used_pick_count': {},
        'residual_values_ms': {},
    }
    for row_index in range(n_rows):
        kind = str(layer_kind[row_index])
        key = str(endpoint_key[row_index])
        if not kind or not key:
            continue
        _increment_layer_count(context['pick_count'], key, kind)
        if bool(used[row_index]):
            _increment_layer_count(context['used_pick_count'], key, kind)
            residual = residual_s[row_index]
            if np.isfinite(residual):
                values = context['residual_values_ms'].setdefault(key, {}).setdefault(
                    kind,
                    [],
                )
                values.append(float(residual) * 1000.0)
    return {
        'pick_count': context['pick_count'],
        'used_pick_count': context['used_pick_count'],
        'residual_rms_ms': _endpoint_layer_residual_stat(
            context['residual_values_ms'],
            stat='rms',
        ),
        'residual_mad_ms': _endpoint_layer_residual_stat(
            context['residual_values_ms'],
            stat='mad',
        ),
    }


def _endpoint_layer_qc_row_fields(
    layer_context: dict[str, dict[str, dict[str, int | float]]],
    endpoint_key: str,
) -> dict[str, str]:
    return {
        'pick_count_by_layer': _csv_json_object(
            layer_context['pick_count'].get(endpoint_key, {})
        ),
        'used_pick_count_by_layer': _csv_json_object(
            layer_context['used_pick_count'].get(endpoint_key, {})
        ),
        'residual_rms_by_layer_ms': _csv_json_object(
            layer_context['residual_rms_ms'].get(endpoint_key, {})
        ),
        'residual_mad_by_layer_ms': _csv_json_object(
            layer_context['residual_mad_ms'].get(endpoint_key, {})
        ),
    }


def _increment_layer_count(
    target: dict[str, dict[str, int]],
    endpoint_key: str,
    layer_kind: str,
) -> None:
    by_layer = target.setdefault(endpoint_key, {})
    by_layer[layer_kind] = int(by_layer.get(layer_kind, 0)) + 1


def _endpoint_layer_residual_stat(
    values_by_endpoint: dict[str, dict[str, list[float]]],
    *,
    stat: str,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for endpoint_key, values_by_layer in values_by_endpoint.items():
        by_layer: dict[str, float] = {}
        for layer_kind, values in values_by_layer.items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                continue
            if stat == 'rms':
                by_layer[layer_kind] = float(np.sqrt(np.mean(arr * arr)))
            elif stat == 'mad':
                by_layer[layer_kind] = float(np.median(np.abs(arr - np.median(arr))))
            else:
                raise RefractionStaticArtifactError(
                    f'unsupported endpoint layer residual stat: {stat}'
                )
        out[endpoint_key] = by_layer
    return out


def _source_static_status_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    node_context = _node_context(result)
    return _endpoint_static_status_array(
        node_id=result.source_node_id,
        x_m=result.source_x_m,
        y_m=result.source_y_m,
        surface_elevation_m=result.source_surface_elevation_m,
        t1_s=result.source_half_intercept_time_s,
        weathering_thickness_m=result.source_weathering_thickness_m,
        total_shift_s=result.source_refraction_shift_s,
        datum_status=result.source_datum_status,
        node_solution_status=node_context['solution_status'],
        node_weathering_status=node_context['weathering_status'],
    )


def _receiver_static_status_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    node_context = _node_context(result)
    return _endpoint_static_status_array(
        node_id=result.receiver_node_id,
        x_m=result.receiver_x_m,
        y_m=result.receiver_y_m,
        surface_elevation_m=result.receiver_surface_elevation_m,
        t1_s=result.receiver_half_intercept_time_s,
        weathering_thickness_m=result.receiver_weathering_thickness_m,
        total_shift_s=result.receiver_refraction_shift_s,
        datum_status=result.receiver_datum_status,
        node_solution_status=node_context['solution_status'],
        node_weathering_status=node_context['weathering_status'],
    )


def _endpoint_static_status_array(
    *,
    node_id: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    surface_elevation_m: np.ndarray,
    t1_s: np.ndarray,
    weathering_thickness_m: np.ndarray,
    total_shift_s: np.ndarray,
    datum_status: np.ndarray,
    node_solution_status: dict[int, Any],
    node_weathering_status: dict[int, Any],
) -> np.ndarray:
    statuses: list[str] = []
    for index, raw_node_id in enumerate(np.asarray(node_id).tolist()):
        endpoint_node_id = int(raw_node_id)
        solution_status = str(
            node_solution_status.get(endpoint_node_id, 'missing_solution')
        )
        weathering_status = str(
            node_weathering_status.get(endpoint_node_id, 'missing_node')
        )
        statuses.append(
            _endpoint_static_status(
                node_missing=endpoint_node_id not in node_solution_status,
                x_m=x_m[index],
                y_m=y_m[index],
                surface_elevation_m=surface_elevation_m[index],
                t1_s=t1_s[index],
                weathering_thickness_m=weathering_thickness_m[index],
                total_shift_s=total_shift_s[index],
                solution_status=solution_status,
                weathering_status=weathering_status,
                datum_status=datum_status[index],
            )
        )
    return _string_array(statuses)


def _endpoint_static_status(
    *,
    node_missing: bool,
    x_m: object,
    y_m: object,
    surface_elevation_m: object,
    t1_s: object,
    weathering_thickness_m: object,
    total_shift_s: object,
    solution_status: object,
    weathering_status: object,
    datum_status: object,
) -> str:
    return classify_refraction_endpoint_static_status(
        node_missing=node_missing,
        x_m=x_m,
        y_m=y_m,
        surface_elevation_m=surface_elevation_m,
        t1_s=t1_s,
        weathering_thickness_m=weathering_thickness_m,
        total_shift_s=total_shift_s,
        solution_status=solution_status,
        weathering_status=weathering_status,
        datum_status=datum_status,
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


def _sum_float_arrays(left: object, right: object) -> np.ndarray:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    out = np.full(left_arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(left_arr) & np.isfinite(right_arr)
    out[finite] = left_arr[finite] + right_arr[finite]
    return np.ascontiguousarray(out, dtype=np.float64)


def _string_array(value: object) -> np.ndarray:
    raw = [str(item) for item in np.asarray(value).tolist()]
    max_len = max([1, *(len(item) for item in raw)])
    return np.ascontiguousarray(raw, dtype=f'<U{max_len}')


__all__ = [
    'build_refraction_line_profile_qc_arrays',
    'build_refraction_line_profile_qc_payload',
    'write_refraction_line_profile_qc_artifacts',
]
