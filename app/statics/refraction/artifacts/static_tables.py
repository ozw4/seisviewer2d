"""Source/receiver static table and time-term spreadsheet artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from app.statics.refraction.artifacts.arrays import (
    _endpoint_cell_id_array,
    _endpoint_v2_m_s,
    _endpoint_v2_status_array,
    _filled_float_array,
    _float_array,
    _int_array,
    _scalar_str,
    _string_array,
    _sum_correction_s,
    _sum_float_arrays,
)
from app.statics.refraction.artifacts.line_profile import (
    _endpoint_layer_qc_context,
    _endpoint_layer_qc_row_fields,
    _node_context,
    _receiver_static_status_array,
    _source_static_status_array,
)
from app.statics.refraction.artifacts.validation import _validate_result
from app.statics.refraction.artifacts.contract import (
    _RECEIVER_STATIC_TABLE_2LAYER_COLUMNS,
    _RECEIVER_STATIC_TABLE_3LAYER_COLUMNS,
    _RECEIVER_STATIC_TABLE_COLUMNS,
    _SOURCE_STATIC_TABLE_2LAYER_COLUMNS,
    _SOURCE_STATIC_TABLE_3LAYER_COLUMNS,
    _SOURCE_STATIC_TABLE_COLUMNS,
    _TIME_TERM_SPREADSHEET_COLUMNS,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    SIGN_CONVENTION,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    TIME_TERM_SPREADSHEET_FORMAT_NAME,
    TIME_TERM_SPREADSHEET_FORMAT_VERSION,
    TIME_TERM_SPREADSHEET_SCHEMA_VERSION,
    RefractionStaticArtifactError,
)
from app.statics.refraction.artifacts.field_corrections import (
    _FIELD_NOT_APPLICABLE_STATUS,
    _receiver_field_shift_s_array,
    _receiver_field_static_status_array,
    _receiver_manual_static_shift_s_array,
    _receiver_manual_static_status_array,
    _source_depth_m_array,
    _source_depth_shift_s_array,
    _source_depth_status_array,
    _source_field_shift_s_array,
    _source_field_static_status_array,
    _source_manual_static_shift_s_array,
    _source_manual_static_status_array,
    _source_uphole_shift_s_array,
    _source_uphole_status_array,
    _source_uphole_time_s_array,
    _total_with_field_shift_s,
)
from app.statics.refraction.artifacts.formatters import (
    _csv_cell_id,
    _csv_float,
    _csv_int,
    _csv_ms,
    _nan_if_none,
    _spreadsheet_int,
    _spreadsheet_m,
    _spreadsheet_ms,
    _spreadsheet_text,
    _spreadsheet_velocity,
)
from app.statics.refraction.artifacts.io import (
    _validate_no_object_arrays,
    _write_csv_atomic,
    _write_npz_atomic,
)
from app.statics.refraction.contracts.result_types import RefractionDatumStaticsResult

_NODE_2LAYER_STATIC_ARRAY_NAMES = (
    'node_sh1_weathering_thickness_m',
    'node_sh2_weathering_thickness_m',
)

_NODE_3LAYER_STATIC_ARRAY_NAMES = (
    'node_sh1_weathering_thickness_m',
    'node_sh2_weathering_thickness_m',
    'node_sh3_weathering_thickness_m',
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


def write_source_static_table_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    rows = _source_static_table_rows(values.result)
    _write_csv_atomic(Path(path), _source_static_table_columns(values.result), rows)


def write_receiver_static_table_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    rows = _receiver_static_table_rows(values.result)
    _write_csv_atomic(Path(path), _receiver_static_table_columns(values.result), rows)


def write_source_receiver_static_table_npz(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    payload = build_source_receiver_static_table_arrays(result=values.result)
    _write_npz_atomic(Path(path), payload)


def write_refraction_time_term_spreadsheet_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
    source_job_id: str | None = None,
) -> None:
    values = _validate_result(result)
    rows = _time_term_spreadsheet_rows(
        values.result,
        source_job_id=source_job_id,
    )
    _write_csv_atomic(Path(path), _TIME_TERM_SPREADSHEET_COLUMNS, rows)


def write_refraction_time_term_spreadsheet_csv_from_static_tables(
    *,
    source_rows: Iterable[Mapping[str, object]],
    receiver_rows: Iterable[Mapping[str, object]],
    path: Path,
    source_job_id: str | None = None,
    include_inactive_endpoints: bool = True,
) -> None:
    rows = _time_term_spreadsheet_rows_from_static_tables(
        source_rows=source_rows,
        receiver_rows=receiver_rows,
        source_job_id=source_job_id,
        include_inactive_endpoints=include_inactive_endpoints,
    )
    _write_csv_atomic(Path(path), _TIME_TERM_SPREADSHEET_COLUMNS, rows)


def build_source_receiver_static_table_arrays(
    *,
    result: RefractionDatumStaticsResult,
) -> dict[str, np.ndarray]:
    values = _validate_result(result)
    r = values.result
    source_t1_s = _float_array(r.source_half_intercept_time_s)
    source_sh1_m = _source_sh1_weathering_thickness_m(r)
    source_weathering_correction_s = _float_array(
        r.source_weathering_replacement_shift_s
    )
    receiver_t1_s = _float_array(r.receiver_half_intercept_time_s)
    receiver_sh1_m = _receiver_sh1_weathering_thickness_m(r)
    receiver_weathering_correction_s = _float_array(
        r.receiver_weathering_replacement_shift_s
    )
    source_static_status = _source_static_status_array(r)
    receiver_static_status = _receiver_static_status_array(r)
    source_v2 = _endpoint_v2_m_s(
        r.source_v2_m_s,
        shape=values.n_source_endpoints,
        scalar_v2_m_s=r.bedrock_velocity_m_s,
    )
    receiver_v2 = _endpoint_v2_m_s(
        r.receiver_v2_m_s,
        shape=values.n_receiver_endpoints,
        scalar_v2_m_s=r.bedrock_velocity_m_s,
    )
    arrays: dict[str, np.ndarray] = {
        'sign_convention': _scalar_str(SIGN_CONVENTION),
        'source_endpoint_key': _string_array(r.source_endpoint_key),
        'source_id': _int_array(r.source_id),
        'source_node_id': _int_array(r.source_node_id),
        'source_v2_cell_id': _endpoint_cell_id_array(
            r.source_v2_cell_id,
            values.n_source_endpoints,
        ),
        'source_v2_status': _endpoint_v2_status_array(
            r.source_v2_status,
            values.n_source_endpoints,
        ),
        'source_x_m': _float_array(r.source_x_m),
        'source_y_m': _float_array(r.source_y_m),
        'source_surface_elevation_m': _float_array(
            r.source_surface_elevation_m
        ),
        'source_t1_s': source_t1_s,
        'source_v1_m_s': _filled_float_array(
            r.weathering_velocity_m_s,
            values.n_source_endpoints,
        ),
        'source_v2_m_s': source_v2,
        'source_sh1_m': source_sh1_m,
        'source_total_weathering_thickness_m': _float_array(
            r.source_weathering_thickness_m
        ),
        'source_weathering_correction_s': source_weathering_correction_s,
        'source_elevation_correction_s': _sum_float_arrays(
            r.source_floating_datum_elevation_shift_s,
            r.source_flat_datum_shift_s,
        ),
        'source_total_static_s': _float_array(r.source_refraction_shift_s),
        'source_total_applied_shift_s': _float_array(
            r.source_refraction_shift_s
        ),
        'source_static_status': source_static_status,
        'receiver_endpoint_key': _string_array(r.receiver_endpoint_key),
        'receiver_id': _int_array(r.receiver_id),
        'receiver_node_id': _int_array(r.receiver_node_id),
        'receiver_v2_cell_id': _endpoint_cell_id_array(
            r.receiver_v2_cell_id,
            values.n_receiver_endpoints,
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
        'receiver_t1_s': receiver_t1_s,
        'receiver_v1_m_s': _filled_float_array(
            r.weathering_velocity_m_s,
            values.n_receiver_endpoints,
        ),
        'receiver_v2_m_s': receiver_v2,
        'receiver_sh1_m': receiver_sh1_m,
        'receiver_total_weathering_thickness_m': _float_array(
            r.receiver_weathering_thickness_m
        ),
        'receiver_weathering_correction_s': receiver_weathering_correction_s,
        'receiver_elevation_correction_s': _sum_float_arrays(
            r.receiver_floating_datum_elevation_shift_s,
            r.receiver_flat_datum_shift_s,
        ),
        'receiver_total_static_s': _float_array(
            r.receiver_refraction_shift_s
        ),
        'receiver_total_applied_shift_s': _float_array(
            r.receiver_refraction_shift_s
        ),
        'receiver_static_status': receiver_static_status,
    }
    source_field_shift = _source_field_shift_s_array(r)
    source_field_status = _source_field_static_status_array(r)
    receiver_field_shift = _receiver_field_shift_s_array(r)
    receiver_field_status = _receiver_field_static_status_array(r)
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
        }
    )
    if _has_source_2layer_static_fields(r):
        assert r.source_t2_time_s is not None
        assert r.source_v3_m_s is not None
        assert r.source_sh2_weathering_thickness_m is not None
        source_layer1_base = r.source_surface_elevation_m - source_sh1_m
        arrays.update(
            {
                'source_t2_s': _float_array(r.source_t2_time_s),
                'source_v3_m_s': _float_array(r.source_v3_m_s),
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
                    'source_t3_s': _float_array(r.source_t3_time_s),
                    'source_vsub_m_s': _float_array(r.source_vsub_m_s),
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
        receiver_layer1_base = r.receiver_surface_elevation_m - receiver_sh1_m
        arrays.update(
            {
                'receiver_t2_s': _float_array(r.receiver_t2_time_s),
                'receiver_v3_m_s': _float_array(r.receiver_v3_m_s),
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
                    'receiver_t3_s': _float_array(r.receiver_t3_time_s),
                    'receiver_vsub_m_s': _float_array(r.receiver_vsub_m_s),
                    'receiver_sh3_m': _float_array(
                        r.receiver_sh3_weathering_thickness_m
                    ),
                    'receiver_layer2_base_elevation_m': _float_array(
                        receiver_layer1_base
                        - r.receiver_sh2_weathering_thickness_m
                    ),
                }
            )
    _validate_no_object_arrays(
        arrays,
        artifact_name=SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    )
    return arrays


def _source_static_table_columns(
    result: RefractionDatumStaticsResult,
) -> tuple[str, ...]:
    if _has_source_3layer_static_fields(result):
        columns = _SOURCE_STATIC_TABLE_3LAYER_COLUMNS
    elif _has_source_2layer_static_fields(result):
        columns = _SOURCE_STATIC_TABLE_2LAYER_COLUMNS
    else:
        columns = _SOURCE_STATIC_TABLE_COLUMNS
    columns = _insert_after(
        columns,
        'weathering_correction_ms',
        ('source_depth_m', 'source_depth_shift_ms', 'source_depth_status'),
    )
    columns = _insert_after(
        columns,
        'source_depth_status',
        ('uphole_time_ms', 'uphole_shift_ms', 'uphole_status'),
    )
    columns = _insert_after(
        columns,
        'uphole_status',
        ('manual_static_shift_ms', 'manual_static_status'),
    )
    columns = _insert_after(
        columns,
        'manual_static_status',
        (
            'source_field_shift_ms',
            'source_field_status',
            'source_field_static_status',
            'source_total_with_field_shift_ms',
        ),
    )
    return columns


def _receiver_static_table_columns(
    result: RefractionDatumStaticsResult,
) -> tuple[str, ...]:
    if _has_receiver_3layer_static_fields(result):
        columns = _RECEIVER_STATIC_TABLE_3LAYER_COLUMNS
    elif _has_receiver_2layer_static_fields(result):
        columns = _RECEIVER_STATIC_TABLE_2LAYER_COLUMNS
    else:
        columns = _RECEIVER_STATIC_TABLE_COLUMNS
    columns = _insert_after(
        columns,
        'weathering_correction_ms',
        ('manual_static_shift_ms', 'manual_static_status'),
    )
    columns = _insert_after(
        columns,
        'manual_static_status',
        (
            'receiver_field_shift_ms',
            'receiver_field_status',
            'receiver_field_static_status',
            'receiver_total_with_field_shift_ms',
        ),
    )
    return columns


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


def _source_static_table_rows(
    result: RefractionDatumStaticsResult,
) -> list[dict[str, object]]:
    node_context = _node_context(result)
    layer_context = _endpoint_layer_qc_context(result, endpoint='source')
    static_status = _source_static_status_array(result)
    flat_datum = _nan_if_none(result.flat_datum_elevation_m)
    source_v2 = _endpoint_v2_m_s(
        result.source_v2_m_s,
        shape=int(result.source_endpoint_key.shape[0]),
        scalar_v2_m_s=result.bedrock_velocity_m_s,
    )
    source_v2_cell_id = _endpoint_cell_id_array(
        result.source_v2_cell_id,
        int(result.source_endpoint_key.shape[0]),
    )
    source_v2_status = _endpoint_v2_status_array(
        result.source_v2_status,
        int(result.source_endpoint_key.shape[0]),
    )
    has_2layer_fields = _has_source_2layer_static_fields(result)
    has_3layer_fields = _has_source_3layer_static_fields(result)
    source_t2_time_s = result.source_t2_time_s
    source_t3_time_s = result.source_t3_time_s
    source_v3_m_s = result.source_v3_m_s
    source_vsub_m_s = result.source_vsub_m_s
    source_sh2_m = result.source_sh2_weathering_thickness_m
    source_sh3_m = result.source_sh3_weathering_thickness_m
    source_sh1_m = _source_sh1_weathering_thickness_m(result)
    source_depth_m = _source_depth_m_array(result)
    source_depth_shift_s = _source_depth_shift_s_array(result)
    source_depth_status = _source_depth_status_array(result)
    source_uphole_time_s = _source_uphole_time_s_array(result)
    source_uphole_shift_s = _source_uphole_shift_s_array(result)
    source_uphole_status = _source_uphole_status_array(result)
    source_manual_static_shift_s = _source_manual_static_shift_s_array(result)
    source_manual_static_status = _source_manual_static_status_array(result)
    source_field_shift_s = _source_field_shift_s_array(result)
    source_field_status = _source_field_static_status_array(result)
    source_total_with_field_s = _total_with_field_shift_s(
        refraction_shift_s=result.source_refraction_shift_s,
        field_shift_s=source_field_shift_s,
        field_status=source_field_status,
    )
    rows: list[dict[str, object]] = []
    for index in range(int(result.source_endpoint_key.shape[0])):
        node_id = int(result.source_node_id[index])
        t1_s = result.source_half_intercept_time_s[index]
        weathering_correction_s = result.source_weathering_replacement_shift_s[index]
        elevation_correction_s = _sum_correction_s(
            result.source_floating_datum_elevation_shift_s[index],
            result.source_flat_datum_shift_s[index],
        )
        rows.append(
            {
                'endpoint_kind': 'source',
                'source_endpoint_key': str(result.source_endpoint_key[index]),
                'source_id': int(result.source_id[index]),
                'source_node_id': node_id,
                'source_v2_cell_id': _csv_cell_id(source_v2_cell_id[index]),
                'x_m': _csv_float(result.source_x_m[index]),
                'y_m': _csv_float(result.source_y_m[index]),
                'surface_elevation_m': _csv_float(
                    result.source_surface_elevation_m[index]
                ),
                'floating_datum_elevation_m': _csv_float(
                    result.source_floating_datum_elevation_m[index]
                ),
                'flat_datum_elevation_m': _csv_float(flat_datum),
                't1_ms': _csv_ms(t1_s),
                'v1_m_s': _csv_float(result.weathering_velocity_m_s),
                'v2_m_s': _csv_float(source_v2[index]),
                'v2_status': str(source_v2_status[index]),
                'sh1_weathering_thickness_m': _csv_float(source_sh1_m[index]),
                'total_weathering_thickness_m': _csv_float(
                    result.source_weathering_thickness_m[index]
                ),
                'refractor_elevation_m': _csv_float(
                    result.source_refractor_elevation_m[index]
                ),
                'weathering_correction_ms': _csv_ms(weathering_correction_s),
                'floating_datum_correction_ms': _csv_ms(
                    result.source_floating_datum_elevation_shift_s[index]
                ),
                'flat_datum_correction_ms': _csv_ms(
                    result.source_flat_datum_shift_s[index]
                ),
                'elevation_correction_ms': _csv_ms(elevation_correction_s),
                'total_static_ms': _csv_ms(result.source_refraction_shift_s[index]),
                'total_applied_shift_ms': _csv_ms(
                    result.source_refraction_shift_s[index]
                ),
                'solution_status': str(
                    node_context['solution_status'].get(node_id, 'missing_solution')
                ),
                'weathering_status': str(
                    node_context['weathering_status'].get(node_id, 'missing_node')
                ),
                'datum_status': str(result.source_datum_status[index]),
                'static_status': str(static_status[index]),
                'sign_convention': SIGN_CONVENTION,
                'pick_count': _csv_int(node_context['pick_count'].get(node_id)),
                'used_pick_count': _csv_int(
                    node_context['used_pick_count'].get(node_id)
                ),
                'residual_rms_ms': _csv_ms(node_context['residual_rms'].get(node_id)),
                'residual_mad_ms': _csv_ms(node_context['residual_mad'].get(node_id)),
            }
        )
        rows[-1].update(
            {
                'source_depth_m': _csv_float(source_depth_m[index]),
                'source_depth_shift_ms': _csv_ms(source_depth_shift_s[index]),
                'source_depth_status': str(source_depth_status[index]),
                'uphole_time_ms': _csv_ms(source_uphole_time_s[index]),
                'uphole_shift_ms': _csv_ms(source_uphole_shift_s[index]),
                'uphole_status': str(source_uphole_status[index]),
                'manual_static_shift_ms': _csv_ms(
                    source_manual_static_shift_s[index]
                ),
                'manual_static_status': str(source_manual_static_status[index]),
                'source_field_shift_ms': _csv_ms(source_field_shift_s[index]),
                'source_field_status': str(source_field_status[index]),
                'source_field_static_status': str(source_field_status[index]),
                'source_total_with_field_shift_ms': _csv_ms(
                    source_total_with_field_s[index]
                ),
            }
        )
        if has_2layer_fields:
            assert source_t2_time_s is not None
            assert source_v3_m_s is not None
            assert source_sh2_m is not None
            layer1_base = result.source_surface_elevation_m[index] - source_sh1_m[index]
            rows[-1].update(
                {
                    **_endpoint_layer_qc_row_fields(
                        layer_context,
                        str(result.source_endpoint_key[index]),
                    ),
                    't2_ms': _csv_ms(source_t2_time_s[index]),
                    'v3_m_s': _csv_float(source_v3_m_s[index]),
                    'sh2_weathering_thickness_m': _csv_float(source_sh2_m[index]),
                    'layer1_base_elevation_m': _csv_float(layer1_base),
                    'final_refractor_elevation_m': _csv_float(
                        result.source_refractor_elevation_m[index]
                    ),
                }
            )
            if has_3layer_fields:
                assert source_t3_time_s is not None
                assert source_vsub_m_s is not None
                assert source_sh3_m is not None
                rows[-1].update(
                    {
                        't3_ms': _csv_ms(source_t3_time_s[index]),
                        'vsub_m_s': _csv_float(source_vsub_m_s[index]),
                        'sh3_weathering_thickness_m': _csv_float(
                            source_sh3_m[index]
                        ),
                        'layer2_base_elevation_m': _csv_float(
                            layer1_base - source_sh2_m[index]
                        ),
                    }
                )
    return rows


def _receiver_static_table_rows(
    result: RefractionDatumStaticsResult,
) -> list[dict[str, object]]:
    node_context = _node_context(result)
    layer_context = _endpoint_layer_qc_context(result, endpoint='receiver')
    static_status = _receiver_static_status_array(result)
    flat_datum = _nan_if_none(result.flat_datum_elevation_m)
    receiver_v2 = _endpoint_v2_m_s(
        result.receiver_v2_m_s,
        shape=int(result.receiver_endpoint_key.shape[0]),
        scalar_v2_m_s=result.bedrock_velocity_m_s,
    )
    receiver_v2_cell_id = _endpoint_cell_id_array(
        result.receiver_v2_cell_id,
        int(result.receiver_endpoint_key.shape[0]),
    )
    receiver_v2_status = _endpoint_v2_status_array(
        result.receiver_v2_status,
        int(result.receiver_endpoint_key.shape[0]),
    )
    has_2layer_fields = _has_receiver_2layer_static_fields(result)
    has_3layer_fields = _has_receiver_3layer_static_fields(result)
    receiver_t2_time_s = result.receiver_t2_time_s
    receiver_t3_time_s = result.receiver_t3_time_s
    receiver_v3_m_s = result.receiver_v3_m_s
    receiver_vsub_m_s = result.receiver_vsub_m_s
    receiver_sh2_m = result.receiver_sh2_weathering_thickness_m
    receiver_sh3_m = result.receiver_sh3_weathering_thickness_m
    receiver_sh1_m = _receiver_sh1_weathering_thickness_m(result)
    receiver_manual_static_shift_s = _receiver_manual_static_shift_s_array(result)
    receiver_manual_static_status = _receiver_manual_static_status_array(result)
    receiver_field_shift_s = _receiver_field_shift_s_array(result)
    receiver_field_status = _receiver_field_static_status_array(result)
    receiver_total_with_field_s = _total_with_field_shift_s(
        refraction_shift_s=result.receiver_refraction_shift_s,
        field_shift_s=receiver_field_shift_s,
        field_status=receiver_field_status,
    )
    rows: list[dict[str, object]] = []
    for index in range(int(result.receiver_endpoint_key.shape[0])):
        node_id = int(result.receiver_node_id[index])
        t1_s = result.receiver_half_intercept_time_s[index]
        weathering_correction_s = result.receiver_weathering_replacement_shift_s[index]
        elevation_correction_s = _sum_correction_s(
            result.receiver_floating_datum_elevation_shift_s[index],
            result.receiver_flat_datum_shift_s[index],
        )
        rows.append(
            {
                'endpoint_kind': 'receiver',
                'receiver_endpoint_key': str(result.receiver_endpoint_key[index]),
                'receiver_id': int(result.receiver_id[index]),
                'receiver_node_id': node_id,
                'receiver_v2_cell_id': _csv_cell_id(receiver_v2_cell_id[index]),
                'x_m': _csv_float(result.receiver_x_m[index]),
                'y_m': _csv_float(result.receiver_y_m[index]),
                'surface_elevation_m': _csv_float(
                    result.receiver_surface_elevation_m[index]
                ),
                'floating_datum_elevation_m': _csv_float(
                    result.receiver_floating_datum_elevation_m[index]
                ),
                'flat_datum_elevation_m': _csv_float(flat_datum),
                't1_ms': _csv_ms(t1_s),
                'v1_m_s': _csv_float(result.weathering_velocity_m_s),
                'v2_m_s': _csv_float(receiver_v2[index]),
                'v2_status': str(receiver_v2_status[index]),
                'sh1_weathering_thickness_m': _csv_float(receiver_sh1_m[index]),
                'total_weathering_thickness_m': _csv_float(
                    result.receiver_weathering_thickness_m[index]
                ),
                'refractor_elevation_m': _csv_float(
                    result.receiver_refractor_elevation_m[index]
                ),
                'weathering_correction_ms': _csv_ms(weathering_correction_s),
                'floating_datum_correction_ms': _csv_ms(
                    result.receiver_floating_datum_elevation_shift_s[index]
                ),
                'flat_datum_correction_ms': _csv_ms(
                    result.receiver_flat_datum_shift_s[index]
                ),
                'elevation_correction_ms': _csv_ms(elevation_correction_s),
                'total_static_ms': _csv_ms(result.receiver_refraction_shift_s[index]),
                'total_applied_shift_ms': _csv_ms(
                    result.receiver_refraction_shift_s[index]
                ),
                'solution_status': str(
                    node_context['solution_status'].get(node_id, 'missing_solution')
                ),
                'weathering_status': str(
                    node_context['weathering_status'].get(node_id, 'missing_node')
                ),
                'datum_status': str(result.receiver_datum_status[index]),
                'static_status': str(static_status[index]),
                'sign_convention': SIGN_CONVENTION,
                'pick_count': _csv_int(node_context['pick_count'].get(node_id)),
                'used_pick_count': _csv_int(
                    node_context['used_pick_count'].get(node_id)
                ),
                'residual_rms_ms': _csv_ms(node_context['residual_rms'].get(node_id)),
                'residual_mad_ms': _csv_ms(node_context['residual_mad'].get(node_id)),
            }
        )
        rows[-1].update(
            {
                'manual_static_shift_ms': _csv_ms(
                    receiver_manual_static_shift_s[index]
                ),
                'manual_static_status': str(receiver_manual_static_status[index]),
                'receiver_field_shift_ms': _csv_ms(receiver_field_shift_s[index]),
                'receiver_field_status': str(receiver_field_status[index]),
                'receiver_field_static_status': str(receiver_field_status[index]),
                'receiver_total_with_field_shift_ms': _csv_ms(
                    receiver_total_with_field_s[index]
                ),
            }
        )
        if has_2layer_fields:
            assert receiver_t2_time_s is not None
            assert receiver_v3_m_s is not None
            assert receiver_sh2_m is not None
            layer1_base = (
                result.receiver_surface_elevation_m[index] - receiver_sh1_m[index]
            )
            rows[-1].update(
                {
                    **_endpoint_layer_qc_row_fields(
                        layer_context,
                        str(result.receiver_endpoint_key[index]),
                    ),
                    't2_ms': _csv_ms(receiver_t2_time_s[index]),
                    'v3_m_s': _csv_float(receiver_v3_m_s[index]),
                    'sh2_weathering_thickness_m': _csv_float(receiver_sh2_m[index]),
                    'layer1_base_elevation_m': _csv_float(layer1_base),
                    'final_refractor_elevation_m': _csv_float(
                        result.receiver_refractor_elevation_m[index]
                    ),
                }
            )
            if has_3layer_fields:
                assert receiver_t3_time_s is not None
                assert receiver_vsub_m_s is not None
                assert receiver_sh3_m is not None
                rows[-1].update(
                    {
                        't3_ms': _csv_ms(receiver_t3_time_s[index]),
                        'vsub_m_s': _csv_float(receiver_vsub_m_s[index]),
                        'sh3_weathering_thickness_m': _csv_float(
                            receiver_sh3_m[index]
                        ),
                        'layer2_base_elevation_m': _csv_float(
                            layer1_base - receiver_sh2_m[index]
                        ),
                    }
                )
    return rows


def _time_term_spreadsheet_rows(
    result: RefractionDatumStaticsResult,
    *,
    source_job_id: str | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    job_id = '' if source_job_id is None else str(source_job_id)
    for row in _source_static_table_rows(result):
        rows.append(
            _time_term_spreadsheet_endpoint_row(
                row,
                endpoint_prefix='source',
                source_job_id=job_id,
            )
        )
    for row in _receiver_static_table_rows(result):
        rows.append(
            _time_term_spreadsheet_endpoint_row(
                row,
                endpoint_prefix='receiver',
                source_job_id=job_id,
            )
        )
    return rows


def _time_term_spreadsheet_rows_from_static_tables(
    *,
    source_rows: Iterable[Mapping[str, object]],
    receiver_rows: Iterable[Mapping[str, object]],
    source_job_id: str | None,
    include_inactive_endpoints: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    job_id = '' if source_job_id is None else str(source_job_id)
    for row in source_rows:
        _validate_time_term_static_table_row(row, endpoint_kind='source')
        if not _include_time_term_static_table_row(
            row,
            include_inactive_endpoints=include_inactive_endpoints,
        ):
            continue
        rows.append(
            _time_term_spreadsheet_endpoint_row(
                row,
                endpoint_prefix='source',
                source_job_id=job_id,
            )
        )
    for row in receiver_rows:
        _validate_time_term_static_table_row(row, endpoint_kind='receiver')
        if not _include_time_term_static_table_row(
            row,
            include_inactive_endpoints=include_inactive_endpoints,
        ):
            continue
        rows.append(
            _time_term_spreadsheet_endpoint_row(
                row,
                endpoint_prefix='receiver',
                source_job_id=job_id,
            )
        )
    return rows


def _validate_time_term_static_table_row(
    row: Mapping[str, object],
    *,
    endpoint_kind: str,
) -> None:
    if row.get('sign_convention') != SIGN_CONVENTION:
        raise RefractionStaticArtifactError(
            f'{endpoint_kind} static table sign_convention must be {SIGN_CONVENTION!r}'
        )
    if row.get('endpoint_kind') != endpoint_kind:
        raise RefractionStaticArtifactError(
            f'{endpoint_kind} static table contains endpoint_kind '
            f'{row.get("endpoint_kind")!r}'
        )


def _include_time_term_static_table_row(
    row: Mapping[str, object],
    *,
    include_inactive_endpoints: bool,
) -> bool:
    status = str(row.get('static_status', '')).strip()
    return status == 'ok' or bool(include_inactive_endpoints)


def _time_term_spreadsheet_endpoint_row(
    row: Mapping[str, object],
    *,
    endpoint_prefix: str,
    source_job_id: str,
) -> dict[str, object]:
    if endpoint_prefix not in {'source', 'receiver'}:
        raise RefractionStaticArtifactError(
            f'unsupported spreadsheet endpoint prefix: {endpoint_prefix}'
        )
    endpoint_kind = str(row.get('endpoint_kind', ''))
    endpoint_key = row.get(f'{endpoint_prefix}_endpoint_key')
    endpoint_id = row.get(f'{endpoint_prefix}_id')
    node_id = row.get(f'{endpoint_prefix}_node_id')
    layer1_base = row.get('layer1_base_elevation_m')
    if layer1_base in (None, ''):
        layer1_base = row.get('refractor_elevation_m')
    final_refractor = row.get('final_refractor_elevation_m')
    if final_refractor in (None, ''):
        final_refractor = row.get('refractor_elevation_m')
    field_shift_column = f'{endpoint_prefix}_field_shift_ms'
    field_status_column = f'{endpoint_prefix}_field_static_status'

    source_depth_correction = ''
    source_depth_status = _FIELD_NOT_APPLICABLE_STATUS
    uphole_correction = ''
    uphole_status = _FIELD_NOT_APPLICABLE_STATUS
    if endpoint_prefix == 'source':
        source_depth_correction = row.get('source_depth_shift_ms', '')
        source_depth_status = str(row.get('source_depth_status', ''))
        uphole_correction = row.get('uphole_shift_ms', '')
        uphole_status = str(row.get('uphole_status', ''))

    out = {
        'schema_version': str(TIME_TERM_SPREADSHEET_SCHEMA_VERSION),
        'format_name': TIME_TERM_SPREADSHEET_FORMAT_NAME,
        'format_version': str(TIME_TERM_SPREADSHEET_FORMAT_VERSION),
        'source_job_id': source_job_id,
        'endpoint_kind': endpoint_kind,
        'endpoint_key': _spreadsheet_text(endpoint_key),
        'endpoint_id': _spreadsheet_text(endpoint_id),
        'station_id': _spreadsheet_text(endpoint_id),
        'node_id': _spreadsheet_int(node_id),
        'x_m': _spreadsheet_m(row.get('x_m')),
        'y_m': _spreadsheet_m(row.get('y_m')),
        'elevation_m': _spreadsheet_m(row.get('surface_elevation_m')),
        'surface_elevation_m': _spreadsheet_m(row.get('surface_elevation_m')),
        't1_ms': _spreadsheet_ms(row.get('t1_ms')),
        't2_ms': _spreadsheet_ms(row.get('t2_ms')),
        't3_ms': _spreadsheet_ms(row.get('t3_ms')),
        'v1_m_s': _spreadsheet_velocity(row.get('v1_m_s')),
        'v2_m_s': _spreadsheet_velocity(row.get('v2_m_s')),
        'v3_m_s': _spreadsheet_velocity(row.get('v3_m_s')),
        'vsub_m_s': _spreadsheet_velocity(row.get('vsub_m_s')),
        'sh1_m': _spreadsheet_m(row.get('sh1_weathering_thickness_m')),
        'sh2_m': _spreadsheet_m(row.get('sh2_weathering_thickness_m')),
        'sh3_m': _spreadsheet_m(row.get('sh3_weathering_thickness_m')),
        'layer1_base_elevation_m': _spreadsheet_m(layer1_base),
        'layer2_base_elevation_m': _spreadsheet_m(
            row.get('layer2_base_elevation_m')
        ),
        'final_refractor_elevation_m': _spreadsheet_m(final_refractor),
        'weathering_correction_ms': _spreadsheet_ms(
            row.get('weathering_correction_ms')
        ),
        'elevation_correction_ms': _spreadsheet_ms(
            row.get('elevation_correction_ms')
        ),
        'source_depth_correction_ms': _spreadsheet_ms(source_depth_correction),
        'uphole_correction_ms': _spreadsheet_ms(uphole_correction),
        'manual_static_ms': _spreadsheet_ms(row.get('manual_static_shift_ms')),
        'field_correction_ms': _spreadsheet_ms(row.get(field_shift_column)),
        'total_applied_shift_ms': _spreadsheet_ms(
            row.get('total_applied_shift_ms')
        ),
        'pick_count': _spreadsheet_int(row.get('pick_count')),
        'used_pick_count': _spreadsheet_int(row.get('used_pick_count')),
        'pick_count_by_layer': _spreadsheet_text(row.get('pick_count_by_layer')),
        'used_pick_count_by_layer': _spreadsheet_text(
            row.get('used_pick_count_by_layer')
        ),
        'residual_rms_ms': _spreadsheet_ms(row.get('residual_rms_ms')),
        'residual_mad_ms': _spreadsheet_ms(row.get('residual_mad_ms')),
        'residual_rms_by_layer_ms': _spreadsheet_text(
            row.get('residual_rms_by_layer_ms')
        ),
        'residual_mad_by_layer_ms': _spreadsheet_text(
            row.get('residual_mad_by_layer_ms')
        ),
        'solution_status': _spreadsheet_text(row.get('solution_status')),
        'weathering_status': _spreadsheet_text(row.get('weathering_status')),
        'datum_status': _spreadsheet_text(row.get('datum_status')),
        'source_depth_status': source_depth_status,
        'uphole_status': uphole_status,
        'manual_static_status': _spreadsheet_text(row.get('manual_static_status')),
        'field_static_status': _spreadsheet_text(row.get(field_status_column)),
        'static_status': _spreadsheet_text(row.get('static_status')),
        'sign_convention': SIGN_CONVENTION,
    }
    missing = set(_TIME_TERM_SPREADSHEET_COLUMNS) - set(out)
    if missing:
        raise RefractionStaticArtifactError(
            'time-term spreadsheet row missing columns: ' + ', '.join(sorted(missing))
        )
    return out


__all__ = [
    'RECEIVER_STATIC_TABLE_CSV_NAME',
    'REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME',
    'SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME',
    'SOURCE_STATIC_TABLE_CSV_NAME',
    '_has_node_2layer_static_fields',
    '_has_node_3layer_static_fields',
    '_has_receiver_2layer_static_fields',
    '_has_receiver_3layer_static_fields',
    '_has_source_2layer_static_fields',
    '_has_source_3layer_static_fields',
    '_include_time_term_static_table_row',
    '_insert_after',
    '_node_sh1_weathering_thickness_m',
    '_receiver_sh1_weathering_thickness_m',
    '_receiver_static_table_columns',
    '_receiver_static_table_rows',
    '_source_sh1_weathering_thickness_m',
    '_source_static_table_columns',
    '_source_static_table_rows',
    '_time_term_spreadsheet_endpoint_row',
    '_time_term_spreadsheet_rows',
    '_time_term_spreadsheet_rows_from_static_tables',
    '_validate_time_term_static_table_row',
    'build_source_receiver_static_table_arrays',
    'write_receiver_static_table_csv',
    'write_refraction_time_term_spreadsheet_csv',
    'write_refraction_time_term_spreadsheet_csv_from_static_tables',
    'write_source_receiver_static_table_npz',
    'write_source_static_table_csv',
]
