"""Static component CSV and component-QC artifact helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_artifacts.contract import (
    _COMPONENT_COLUMNS,
    _STATIC_COMPONENT_QC_ENDPOINT_COLUMNS,
    _STATIC_COMPONENT_QC_TRACE_COLUMNS,
    ARTIFACT_VERSION,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_JSON_NAME,
    REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME,
    REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    SIGN_CONVENTION,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    RefractionStaticArtifactError,
)
from app.services.refraction_static_artifacts.field_corrections import (
    _FIELD_NOT_APPLICABLE_STATUS,
    _applied_endpoint_field_shift_s,
    _applied_field_shift_s_sorted_array,
    _base_refraction_trace_shift_s_sorted_array,
    _endpoint_shift_to_trace_order,
    _final_trace_shift_s_sorted,
    _final_trace_static_status_sorted_array,
    _has_field_correction_composition,
    _has_manual_static_field_correction,
    _has_source_depth_field_correction,
    _has_uphole_field_correction,
    _receiver_field_shift_s_array,
    _receiver_field_static_status_array,
    _receiver_manual_static_shift_s_array,
    _receiver_manual_static_status_array,
    _source_depth_shift_s_array,
    _source_depth_status_array,
    _source_field_shift_s_array,
    _source_field_static_status_array,
    _source_manual_static_shift_s_array,
    _source_manual_static_status_array,
    _source_uphole_shift_s_array,
    _source_uphole_status_array,
    _total_with_field_shift_s,
    _trace_endpoint_key_sorted_array,
    _trace_field_shift_s_sorted_array,
    _trace_field_static_status_sorted_array,
)
from app.services.refraction_static_artifacts.formatters import (
    _csv_bool,
    _csv_float,
    _csv_int,
    _csv_ms,
)
from app.services.refraction_static_artifacts.io import (
    _assert_strict_json,
    _write_csv_atomic,
    _write_json_atomic,
    _write_npz_atomic,
)
from app.services.refraction_static_artifacts.stats import _stat, _status_counts
from app.services.refraction_static_status import (
    REFRACTION_STATIC_STATUSES,
    classify_refraction_endpoint_static_status,
)
from app.services.refraction_static_types import RefractionDatumStaticsResult


def write_refraction_static_components_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    r = _validate_component_result(result)
    rows = _component_rows(r)
    _write_csv_atomic(Path(path), _component_columns(r), rows)


def write_refraction_static_component_qc_artifacts(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    trace_csv_path: Path,
    endpoint_csv_path: Path,
    npz_path: Path,
    json_path: Path,
) -> dict[str, Any]:
    r = _validate_component_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    arrays = build_refraction_static_component_qc_arrays(result=r, req=request)
    _write_csv_atomic(
        Path(trace_csv_path),
        _STATIC_COMPONENT_QC_TRACE_COLUMNS,
        _static_component_qc_trace_rows(arrays),
    )
    _write_csv_atomic(
        Path(endpoint_csv_path),
        _STATIC_COMPONENT_QC_ENDPOINT_COLUMNS,
        _static_component_qc_endpoint_rows(arrays),
    )
    _write_npz_atomic(Path(npz_path), arrays)
    payload = build_refraction_static_component_qc_payload(arrays=arrays)
    _write_json_atomic(Path(json_path), payload)
    return payload


def build_refraction_static_component_qc_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, np.ndarray]:
    """Build trace and endpoint static component waterfall QC arrays."""
    r = _validate_component_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    apply_to_trace_shift = bool(
        request.field_corrections.composition.apply_to_trace_shift
    )

    source_endpoint_key_sorted = _trace_endpoint_key_sorted_array(
        r,
        endpoint='source',
    )
    receiver_endpoint_key_sorted = _trace_endpoint_key_sorted_array(
        r,
        endpoint='receiver',
    )
    source_depth_shift_s_sorted = _endpoint_shift_to_trace_order(
        endpoint_key=r.source_endpoint_key,
        endpoint_shift_s=_source_depth_shift_s_array(r),
        endpoint_key_sorted=source_endpoint_key_sorted,
        label='source_depth_shift_s',
    )
    uphole_shift_s_sorted = _endpoint_shift_to_trace_order(
        endpoint_key=r.source_endpoint_key,
        endpoint_shift_s=_source_uphole_shift_s_array(r),
        endpoint_key_sorted=source_endpoint_key_sorted,
        label='uphole_shift_s',
    )
    source_manual_static_shift_s_sorted = _endpoint_shift_to_trace_order(
        endpoint_key=r.source_endpoint_key,
        endpoint_shift_s=_source_manual_static_shift_s_array(r),
        endpoint_key_sorted=source_endpoint_key_sorted,
        label='source_manual_static_shift_s',
    )
    receiver_manual_static_shift_s_sorted = _endpoint_shift_to_trace_order(
        endpoint_key=r.receiver_endpoint_key,
        endpoint_shift_s=_receiver_manual_static_shift_s_array(r),
        endpoint_key_sorted=receiver_endpoint_key_sorted,
        label='receiver_manual_static_shift_s',
    )
    manual_static_shift_s_sorted = _sum_float_arrays(
        source_manual_static_shift_s_sorted,
        receiver_manual_static_shift_s_sorted,
    )
    datum_shift_s_sorted = _sum_float_arrays(
        r.floating_datum_elevation_shift_s_sorted,
        r.flat_datum_shift_s_sorted,
    )
    trace_field_shift_s = _trace_field_shift_s_sorted_array(r)
    trace_field_status = _trace_field_static_status_sorted_array(r)
    applied_field_shift_s = _applied_field_shift_s_sorted_array(r)
    final_trace_shift_s = _final_trace_shift_s_sorted(r)
    source_field_shift_s = _source_field_shift_s_array(r)
    source_field_status = _source_field_static_status_array(r)
    receiver_field_shift_s = _receiver_field_shift_s_array(r)
    receiver_field_status = _receiver_field_static_status_array(r)
    source_depth_status = _source_depth_status_array(r)
    source_uphole_status = _source_uphole_status_array(r)
    source_manual_status = _source_manual_static_status_array(r)
    receiver_manual_status = _receiver_manual_static_status_array(r)
    source_total_with_field_s = _total_with_field_shift_s(
        refraction_shift_s=r.source_refraction_shift_s,
        field_shift_s=source_field_shift_s,
        field_status=source_field_status,
    )
    receiver_total_with_field_s = _total_with_field_shift_s(
        refraction_shift_s=r.receiver_refraction_shift_s,
        field_shift_s=receiver_field_shift_s,
        field_status=receiver_field_status,
    )
    source_applied_field_s = _applied_endpoint_field_shift_s(
        field_shift_s=source_field_shift_s,
        field_status=source_field_status,
        apply_to_trace_shift=apply_to_trace_shift,
    )
    receiver_applied_field_s = _applied_endpoint_field_shift_s(
        field_shift_s=receiver_field_shift_s,
        field_status=receiver_field_status,
        apply_to_trace_shift=apply_to_trace_shift,
    )

    source_count = int(r.source_endpoint_key.shape[0])
    receiver_count = int(r.receiver_endpoint_key.shape[0])
    trace_count = int(r.sorted_trace_index.shape[0])
    endpoint_count = source_count + receiver_count

    endpoint_kind = _string_array(
        np.concatenate(
            (
                np.full(source_count, 'source', dtype='<U8'),
                np.full(receiver_count, 'receiver', dtype='<U8'),
            )
        )
    )
    endpoint_key = _string_array(
        np.concatenate(
            (
                _string_array(r.source_endpoint_key),
                _string_array(r.receiver_endpoint_key),
            )
        )
    )
    endpoint_weathering_correction_s = np.concatenate(
        (
            _float_array(r.source_weathering_replacement_shift_s),
            _float_array(r.receiver_weathering_replacement_shift_s),
        )
    )
    endpoint_elevation_correction_s = np.concatenate(
        (
            _sum_float_arrays(
                r.source_floating_datum_elevation_shift_s,
                r.source_flat_datum_shift_s,
            ),
            _sum_float_arrays(
                r.receiver_floating_datum_elevation_shift_s,
                r.receiver_flat_datum_shift_s,
            ),
        )
    )
    endpoint_source_depth_correction_s = np.concatenate(
        (
            _source_depth_shift_s_array(r),
            np.full(receiver_count, np.nan, dtype=np.float64),
        )
    )
    endpoint_source_depth_status = _string_array(
        np.concatenate(
            (
                source_depth_status,
                np.full(
                    receiver_count,
                    _FIELD_NOT_APPLICABLE_STATUS,
                    dtype='<U48',
                ),
            )
        )
    )
    endpoint_uphole_correction_s = np.concatenate(
        (
            _source_uphole_shift_s_array(r),
            np.full(receiver_count, np.nan, dtype=np.float64),
        )
    )
    endpoint_uphole_status = _string_array(
        np.concatenate(
            (
                source_uphole_status,
                np.full(
                    receiver_count,
                    _FIELD_NOT_APPLICABLE_STATUS,
                    dtype='<U48',
                ),
            )
        )
    )
    endpoint_manual_static_s = np.concatenate(
        (
            _source_manual_static_shift_s_array(r),
            _receiver_manual_static_shift_s_array(r),
        )
    )
    endpoint_manual_static_status = _string_array(
        np.concatenate((source_manual_status, receiver_manual_status))
    )
    endpoint_field_correction_s = np.concatenate(
        (source_field_shift_s, receiver_field_shift_s)
    )
    endpoint_source_field_shift_s = np.concatenate(
        (
            source_field_shift_s,
            np.full(receiver_count, np.nan, dtype=np.float64),
        )
    )
    endpoint_receiver_field_shift_s = np.concatenate(
        (
            np.full(source_count, np.nan, dtype=np.float64),
            receiver_field_shift_s,
        )
    )
    endpoint_source_field_static_status = _string_array(
        np.concatenate(
            (
                source_field_status,
                np.full(
                    receiver_count,
                    _FIELD_NOT_APPLICABLE_STATUS,
                    dtype='<U48',
                ),
            )
        )
    )
    endpoint_receiver_field_static_status = _string_array(
        np.concatenate(
            (
                np.full(
                    source_count,
                    _FIELD_NOT_APPLICABLE_STATUS,
                    dtype='<U48',
                ),
                receiver_field_status,
            )
        )
    )
    endpoint_applied_field_correction_s = np.concatenate(
        (source_applied_field_s, receiver_applied_field_s)
    )
    endpoint_total_static_s = np.concatenate(
        (
            _float_array(r.source_refraction_shift_s),
            _float_array(r.receiver_refraction_shift_s),
        )
    )
    endpoint_total_with_field_shift_s = np.concatenate(
        (source_total_with_field_s, receiver_total_with_field_s)
    )
    endpoint_source_total_with_field_shift_s = np.concatenate(
        (
            source_total_with_field_s,
            np.full(receiver_count, np.nan, dtype=np.float64),
        )
    )
    endpoint_receiver_total_with_field_shift_s = np.concatenate(
        (
            np.full(source_count, np.nan, dtype=np.float64),
            receiver_total_with_field_s,
        )
    )
    endpoint_static_status = _string_array(
        np.concatenate(
            (
                _source_static_status_array(r),
                _receiver_static_status_array(r),
            )
        )
    )

    return {
        'artifact_version': _scalar_str(ARTIFACT_VERSION),
        'sign_convention': _scalar_str(SIGN_CONVENTION),
        'apply_to_trace_shift': np.asarray(apply_to_trace_shift, dtype=bool),
        'trace_index_sorted': _int_array(r.sorted_trace_index),
        'source_endpoint_key': source_endpoint_key_sorted,
        'receiver_endpoint_key': receiver_endpoint_key_sorted,
        'refraction_shift_s': _base_refraction_trace_shift_s_sorted_array(r),
        'weathering_shift_s': _float_array(
            r.weathering_replacement_trace_shift_s_sorted
        ),
        'datum_shift_s': datum_shift_s_sorted,
        'field_shift_s': trace_field_shift_s,
        'trace_field_shift_s': trace_field_shift_s,
        'computed_field_shift_s': trace_field_shift_s,
        'applied_field_shift_s': applied_field_shift_s,
        'trace_field_static_status': trace_field_status,
        'manual_static_shift_s': manual_static_shift_s_sorted,
        'source_depth_shift_s': source_depth_shift_s_sorted,
        'uphole_shift_s': uphole_shift_s_sorted,
        'final_trace_shift_s': final_trace_shift_s,
        'applied_trace_shift_s': final_trace_shift_s,
        'trace_apply_to_trace_shift': np.full(
            trace_count,
            apply_to_trace_shift,
            dtype=bool,
        ),
        'trace_static_status': _final_trace_static_status_sorted_array(r),
        'endpoint_kind': endpoint_kind,
        'endpoint_key': endpoint_key,
        'endpoint_weathering_correction_s': np.ascontiguousarray(
            endpoint_weathering_correction_s,
            dtype=np.float64,
        ),
        'endpoint_elevation_correction_s': np.ascontiguousarray(
            endpoint_elevation_correction_s,
            dtype=np.float64,
        ),
        'endpoint_source_depth_correction_s': np.ascontiguousarray(
            endpoint_source_depth_correction_s,
            dtype=np.float64,
        ),
        'endpoint_source_depth_status': endpoint_source_depth_status,
        'endpoint_uphole_correction_s': np.ascontiguousarray(
            endpoint_uphole_correction_s,
            dtype=np.float64,
        ),
        'endpoint_uphole_status': endpoint_uphole_status,
        'endpoint_manual_static_s': np.ascontiguousarray(
            endpoint_manual_static_s,
            dtype=np.float64,
        ),
        'endpoint_manual_static_status': endpoint_manual_static_status,
        'endpoint_source_field_shift_s': np.ascontiguousarray(
            endpoint_source_field_shift_s,
            dtype=np.float64,
        ),
        'endpoint_source_field_static_status': endpoint_source_field_static_status,
        'endpoint_receiver_field_shift_s': np.ascontiguousarray(
            endpoint_receiver_field_shift_s,
            dtype=np.float64,
        ),
        'endpoint_receiver_field_static_status': endpoint_receiver_field_static_status,
        'endpoint_field_correction_s': np.ascontiguousarray(
            endpoint_field_correction_s,
            dtype=np.float64,
        ),
        'endpoint_computed_field_correction_s': np.ascontiguousarray(
            endpoint_field_correction_s,
            dtype=np.float64,
        ),
        'endpoint_applied_field_correction_s': np.ascontiguousarray(
            endpoint_applied_field_correction_s,
            dtype=np.float64,
        ),
        'endpoint_total_static_s': np.ascontiguousarray(
            endpoint_total_static_s,
            dtype=np.float64,
        ),
        'endpoint_total_applied_shift_s': np.ascontiguousarray(
            endpoint_total_static_s,
            dtype=np.float64,
        ),
        'endpoint_total_with_field_shift_s': np.ascontiguousarray(
            endpoint_total_with_field_shift_s,
            dtype=np.float64,
        ),
        'endpoint_source_total_with_field_shift_s': np.ascontiguousarray(
            endpoint_source_total_with_field_shift_s,
            dtype=np.float64,
        ),
        'endpoint_receiver_total_with_field_shift_s': np.ascontiguousarray(
            endpoint_receiver_total_with_field_shift_s,
            dtype=np.float64,
        ),
        'endpoint_apply_to_trace_shift': np.full(
            endpoint_count,
            apply_to_trace_shift,
            dtype=bool,
        ),
        'endpoint_static_status': endpoint_static_status,
    }


def build_refraction_static_component_qc_payload(
    *,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Build strict-JSON static component waterfall QC summary."""
    apply_to_trace_shift = bool(np.asarray(arrays['apply_to_trace_shift']).item())
    payload = {
        'artifact_version': ARTIFACT_VERSION,
        'kind': 'refraction_static_component_qc',
        'sign_convention': SIGN_CONVENTION,
        'units': {
            'csv_time_shift_columns': 'milliseconds',
            'npz_time_shift_arrays': 'seconds',
        },
        'apply_to_trace_shift': apply_to_trace_shift,
        'trace': {
            'row_count': int(arrays['trace_index_sorted'].shape[0]),
            'component_summary_ms': _component_qc_stats_ms(
                {
                    'refraction_trace_shift_ms': arrays['refraction_shift_s'],
                    'refraction_shift_ms': arrays['refraction_shift_s'],
                    'weathering_shift_ms': arrays['weathering_shift_s'],
                    'datum_shift_ms': arrays['datum_shift_s'],
                    'trace_field_shift_ms': arrays['trace_field_shift_s'],
                    'field_shift_ms': arrays['field_shift_s'],
                    'computed_field_shift_ms': arrays['computed_field_shift_s'],
                    'applied_field_shift_ms': arrays['applied_field_shift_s'],
                    'manual_static_shift_ms': arrays['manual_static_shift_s'],
                    'source_depth_shift_ms': arrays['source_depth_shift_s'],
                    'uphole_shift_ms': arrays['uphole_shift_s'],
                    'final_trace_shift_ms': arrays['final_trace_shift_s'],
                    'applied_trace_shift_ms': arrays['applied_trace_shift_s'],
                }
            ),
            'status_counts': _status_counts(arrays['trace_static_status']),
        },
        'endpoint': {
            'row_count': int(arrays['endpoint_key'].shape[0]),
            'component_summary_ms': _component_qc_stats_ms(
                {
                    'weathering_correction_ms': arrays[
                        'endpoint_weathering_correction_s'
                    ],
                    'elevation_correction_ms': arrays[
                        'endpoint_elevation_correction_s'
                    ],
                    'source_depth_correction_ms': arrays[
                        'endpoint_source_depth_correction_s'
                    ],
                    'uphole_correction_ms': arrays['endpoint_uphole_correction_s'],
                    'manual_static_shift_ms': arrays['endpoint_manual_static_s'],
                    'manual_static_ms': arrays['endpoint_manual_static_s'],
                    'source_field_shift_ms': arrays[
                        'endpoint_source_field_shift_s'
                    ],
                    'receiver_field_shift_ms': arrays[
                        'endpoint_receiver_field_shift_s'
                    ],
                    'field_correction_ms': arrays['endpoint_field_correction_s'],
                    'computed_field_correction_ms': arrays[
                        'endpoint_computed_field_correction_s'
                    ],
                    'applied_field_correction_ms': arrays[
                        'endpoint_applied_field_correction_s'
                    ],
                    'total_static_ms': arrays['endpoint_total_static_s'],
                    'total_applied_shift_ms': arrays[
                        'endpoint_total_applied_shift_s'
                    ],
                    'source_total_with_field_shift_ms': arrays[
                        'endpoint_source_total_with_field_shift_s'
                    ],
                    'receiver_total_with_field_shift_ms': arrays[
                        'endpoint_receiver_total_with_field_shift_s'
                    ],
                    'total_with_field_shift_ms': arrays[
                        'endpoint_total_with_field_shift_s'
                    ],
                }
            ),
            'status_counts': _status_counts(arrays['endpoint_static_status']),
        },
        'artifacts': {
            'trace_csv': REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
            'endpoint_csv': REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
            'npz': REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME,
            'json': REFRACTION_STATIC_COMPONENT_QC_JSON_NAME,
        },
        'source_artifacts': {
            'solution_npz': REFRACTION_STATIC_SOLUTION_NPZ_NAME,
            'source_static_table_csv': SOURCE_STATIC_TABLE_CSV_NAME,
            'receiver_static_table_csv': RECEIVER_STATIC_TABLE_CSV_NAME,
            'source_receiver_static_table_npz': SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        },
    }
    _assert_strict_json(
        payload,
        artifact_name=REFRACTION_STATIC_COMPONENT_QC_JSON_NAME,
    )
    return payload


def _static_component_qc_trace_rows(
    arrays: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_rows = int(arrays['trace_index_sorted'].shape[0])
    for index in range(n_rows):
        rows.append(
            {
                'trace_index_sorted': int(arrays['trace_index_sorted'][index]),
                'source_endpoint_key': str(arrays['source_endpoint_key'][index]),
                'receiver_endpoint_key': str(arrays['receiver_endpoint_key'][index]),
                'refraction_trace_shift_ms': _csv_ms(
                    arrays['refraction_shift_s'][index]
                ),
                'refraction_shift_ms': _csv_ms(arrays['refraction_shift_s'][index]),
                'weathering_shift_ms': _csv_ms(arrays['weathering_shift_s'][index]),
                'datum_shift_ms': _csv_ms(arrays['datum_shift_s'][index]),
                'trace_field_shift_ms': _csv_ms(
                    arrays['trace_field_shift_s'][index]
                ),
                'field_shift_ms': _csv_ms(arrays['field_shift_s'][index]),
                'computed_field_shift_ms': _csv_ms(
                    arrays['computed_field_shift_s'][index]
                ),
                'applied_field_shift_ms': _csv_ms(
                    arrays['applied_field_shift_s'][index]
                ),
                'trace_field_static_status': str(
                    arrays['trace_field_static_status'][index]
                ),
                'manual_static_shift_ms': _csv_ms(
                    arrays['manual_static_shift_s'][index]
                ),
                'source_depth_shift_ms': _csv_ms(
                    arrays['source_depth_shift_s'][index]
                ),
                'uphole_shift_ms': _csv_ms(arrays['uphole_shift_s'][index]),
                'final_trace_shift_ms': _csv_ms(
                    arrays['final_trace_shift_s'][index]
                ),
                'applied_trace_shift_ms': _csv_ms(
                    arrays['applied_trace_shift_s'][index]
                ),
                'apply_to_trace_shift': _csv_bool(
                    arrays['trace_apply_to_trace_shift'][index]
                ),
                'static_status': str(arrays['trace_static_status'][index]),
                'sign_convention': SIGN_CONVENTION,
            }
        )
    return rows


def _static_component_qc_endpoint_rows(
    arrays: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_rows = int(arrays['endpoint_key'].shape[0])
    for index in range(n_rows):
        rows.append(
            {
                'endpoint_kind': str(arrays['endpoint_kind'][index]),
                'endpoint_key': str(arrays['endpoint_key'][index]),
                'weathering_correction_ms': _csv_ms(
                    arrays['endpoint_weathering_correction_s'][index]
                ),
                'elevation_correction_ms': _csv_ms(
                    arrays['endpoint_elevation_correction_s'][index]
                ),
                'source_depth_correction_ms': _csv_ms(
                    arrays['endpoint_source_depth_correction_s'][index]
                ),
                'source_depth_status': str(
                    arrays['endpoint_source_depth_status'][index]
                ),
                'uphole_correction_ms': _csv_ms(
                    arrays['endpoint_uphole_correction_s'][index]
                ),
                'uphole_status': str(arrays['endpoint_uphole_status'][index]),
                'manual_static_shift_ms': _csv_ms(
                    arrays['endpoint_manual_static_s'][index]
                ),
                'manual_static_ms': _csv_ms(
                    arrays['endpoint_manual_static_s'][index]
                ),
                'manual_static_status': str(
                    arrays['endpoint_manual_static_status'][index]
                ),
                'source_field_shift_ms': _csv_ms(
                    arrays['endpoint_source_field_shift_s'][index]
                ),
                'source_field_static_status': str(
                    arrays['endpoint_source_field_static_status'][index]
                ),
                'receiver_field_shift_ms': _csv_ms(
                    arrays['endpoint_receiver_field_shift_s'][index]
                ),
                'receiver_field_static_status': str(
                    arrays['endpoint_receiver_field_static_status'][index]
                ),
                'field_correction_ms': _csv_ms(
                    arrays['endpoint_field_correction_s'][index]
                ),
                'computed_field_correction_ms': _csv_ms(
                    arrays['endpoint_computed_field_correction_s'][index]
                ),
                'applied_field_correction_ms': _csv_ms(
                    arrays['endpoint_applied_field_correction_s'][index]
                ),
                'total_static_ms': _csv_ms(arrays['endpoint_total_static_s'][index]),
                'total_applied_shift_ms': _csv_ms(
                    arrays['endpoint_total_applied_shift_s'][index]
                ),
                'source_total_with_field_shift_ms': _csv_ms(
                    arrays['endpoint_source_total_with_field_shift_s'][index]
                ),
                'receiver_total_with_field_shift_ms': _csv_ms(
                    arrays['endpoint_receiver_total_with_field_shift_s'][index]
                ),
                'total_with_field_shift_ms': _csv_ms(
                    arrays['endpoint_total_with_field_shift_s'][index]
                ),
                'apply_to_trace_shift': _csv_bool(
                    arrays['endpoint_apply_to_trace_shift'][index]
                ),
                'static_status': str(arrays['endpoint_static_status'][index]),
                'sign_convention': SIGN_CONVENTION,
            }
        )
    return rows


def _component_rows(result: RefractionDatumStaticsResult) -> list[dict[str, object]]:
    node_pick_count = _node_lookup(result.node_id, result.node_pick_count)
    node_residual_rms = _node_lookup(result.node_id, result.node_residual_rms_s)
    has_source_depth = _has_source_depth_field_correction(result)
    has_uphole = _has_uphole_field_correction(result)
    has_manual_static = _has_manual_static_field_correction(result)
    has_field_composition = _has_field_correction_composition(result)
    rows: list[dict[str, object]] = []
    for index in range(int(result.source_endpoint_key.shape[0])):
        node_id = int(result.source_node_id[index])
        row = {
            'kind': 'source',
            'endpoint_key': str(result.source_endpoint_key[index]),
            'station_id': int(result.source_id[index]),
            'node_id': node_id,
            'x_m': _csv_float(result.source_x_m[index]),
            'y_m': _csv_float(result.source_y_m[index]),
            'surface_elevation_m': _csv_float(
                result.source_surface_elevation_m[index]
            ),
            'floating_datum_elevation_m': _csv_float(
                result.source_floating_datum_elevation_m[index]
            ),
            'refractor_elevation_m': _csv_float(
                result.source_refractor_elevation_m[index]
            ),
            'weathering_thickness_m': _csv_float(
                result.source_weathering_thickness_m[index]
            ),
            'half_intercept_time_ms': _csv_ms(
                result.source_half_intercept_time_s[index]
            ),
            'weathering_replacement_shift_ms': _csv_ms(
                result.source_weathering_replacement_shift_s[index]
            ),
            'floating_datum_elevation_shift_ms': _csv_ms(
                result.source_floating_datum_elevation_shift_s[index]
            ),
            'flat_datum_shift_ms': _csv_ms(result.source_flat_datum_shift_s[index]),
            'refraction_shift_ms': _csv_ms(result.source_refraction_shift_s[index]),
            'datum_status': str(result.source_datum_status[index]),
            'pick_count': _csv_int(node_pick_count.get(node_id)),
            'residual_rms_ms': _csv_ms(node_residual_rms.get(node_id)),
        }
        if has_source_depth:
            assert result.source_depth_shift_s is not None
            assert result.source_depth_status is not None
            row.update(
                {
                    'source_depth_shift_ms': _csv_ms(
                        result.source_depth_shift_s[index]
                    ),
                    'source_depth_status': str(result.source_depth_status[index]),
                }
            )
        if has_uphole:
            assert result.source_uphole_shift_s is not None
            assert result.source_uphole_status is not None
            row.update(
                {
                    'uphole_shift_ms': _csv_ms(result.source_uphole_shift_s[index]),
                    'uphole_status': str(result.source_uphole_status[index]),
                }
            )
        if has_manual_static:
            assert result.source_manual_static_shift_s is not None
            assert result.source_manual_static_status is not None
            row.update(
                {
                    'manual_static_shift_ms': _csv_ms(
                        result.source_manual_static_shift_s[index]
                    ),
                    'manual_static_status': str(
                        result.source_manual_static_status[index]
                    ),
                }
            )
        if has_field_composition:
            assert result.source_field_shift_s is not None
            assert result.source_field_static_status is not None
            row.update(
                {
                    'field_shift_ms': _csv_ms(result.source_field_shift_s[index]),
                    'field_status': str(result.source_field_static_status[index]),
                }
            )
        rows.append(row)
    for index in range(int(result.receiver_endpoint_key.shape[0])):
        node_id = int(result.receiver_node_id[index])
        row = {
            'kind': 'receiver',
            'endpoint_key': str(result.receiver_endpoint_key[index]),
            'station_id': int(result.receiver_id[index]),
            'node_id': node_id,
            'x_m': _csv_float(result.receiver_x_m[index]),
            'y_m': _csv_float(result.receiver_y_m[index]),
            'surface_elevation_m': _csv_float(
                result.receiver_surface_elevation_m[index]
            ),
            'floating_datum_elevation_m': _csv_float(
                result.receiver_floating_datum_elevation_m[index]
            ),
            'refractor_elevation_m': _csv_float(
                result.receiver_refractor_elevation_m[index]
            ),
            'weathering_thickness_m': _csv_float(
                result.receiver_weathering_thickness_m[index]
            ),
            'half_intercept_time_ms': _csv_ms(
                result.receiver_half_intercept_time_s[index]
            ),
            'weathering_replacement_shift_ms': _csv_ms(
                result.receiver_weathering_replacement_shift_s[index]
            ),
            'floating_datum_elevation_shift_ms': _csv_ms(
                result.receiver_floating_datum_elevation_shift_s[index]
            ),
            'flat_datum_shift_ms': _csv_ms(result.receiver_flat_datum_shift_s[index]),
            'refraction_shift_ms': _csv_ms(
                result.receiver_refraction_shift_s[index]
            ),
            'datum_status': str(result.receiver_datum_status[index]),
            'pick_count': _csv_int(node_pick_count.get(node_id)),
            'residual_rms_ms': _csv_ms(node_residual_rms.get(node_id)),
        }
        if has_source_depth:
            row.update(
                {
                    'source_depth_shift_ms': '',
                    'source_depth_status': 'not_applicable',
                }
            )
        if has_uphole:
            row.update(
                {
                    'uphole_shift_ms': '',
                    'uphole_status': 'not_applicable',
                }
            )
        if has_manual_static:
            assert result.receiver_manual_static_shift_s is not None
            assert result.receiver_manual_static_status is not None
            row.update(
                {
                    'manual_static_shift_ms': _csv_ms(
                        result.receiver_manual_static_shift_s[index]
                    ),
                    'manual_static_status': str(
                        result.receiver_manual_static_status[index]
                    ),
                }
            )
        if has_field_composition:
            assert result.receiver_field_shift_s is not None
            assert result.receiver_field_static_status is not None
            row.update(
                {
                    'field_shift_ms': _csv_ms(result.receiver_field_shift_s[index]),
                    'field_status': str(result.receiver_field_static_status[index]),
                }
            )
        rows.append(row)
    return rows


def _component_columns(result: RefractionDatumStaticsResult) -> tuple[str, ...]:
    columns = _COMPONENT_COLUMNS
    if _has_source_depth_field_correction(result):
        columns = _insert_after(
            columns,
            'flat_datum_shift_ms',
            ('source_depth_shift_ms', 'source_depth_status'),
        )
    if _has_uphole_field_correction(result):
        anchor = (
            'source_depth_status'
            if 'source_depth_status' in columns
            else 'flat_datum_shift_ms'
        )
        columns = _insert_after(
            columns,
            anchor,
            ('uphole_shift_ms', 'uphole_status'),
        )
    if _has_manual_static_field_correction(result):
        if 'uphole_status' in columns:
            anchor = 'uphole_status'
        elif 'source_depth_status' in columns:
            anchor = 'source_depth_status'
        else:
            anchor = 'flat_datum_shift_ms'
        columns = _insert_after(
            columns,
            anchor,
            ('manual_static_shift_ms', 'manual_static_status'),
        )
    if _has_field_correction_composition(result):
        anchor = (
            'manual_static_status'
            if 'manual_static_status' in columns
            else (
                'uphole_status'
                if 'uphole_status' in columns
                else (
                    'source_depth_status'
                    if 'source_depth_status' in columns
                    else 'flat_datum_shift_ms'
                )
            )
        )
        columns = _insert_after(columns, anchor, ('field_shift_ms', 'field_status'))
    return columns


def _component_qc_stats_ms(
    components_s: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float | None]]:
    return {
        name: {
            'min': _stat(values * 1000.0, 'min'),
            'median': _stat(values * 1000.0, 'median'),
            'max': _stat(values * 1000.0, 'max'),
        }
        for name, values in components_s.items()
    }


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
            classify_refraction_endpoint_static_status(
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


def _node_context(result: RefractionDatumStaticsResult) -> dict[str, dict[int, Any]]:
    return {
        'solution_status': _node_lookup(result.node_id, result.node_solution_status),
        'weathering_status': _node_lookup(
            result.node_id,
            result.node_weathering_status,
        ),
    }


def _node_lookup(node_id: np.ndarray, values: np.ndarray) -> dict[int, Any]:
    return {
        int(raw_node): values[index]
        for index, raw_node in enumerate(np.asarray(node_id).tolist())
    }


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


def _validate_component_result(
    result: RefractionDatumStaticsResult,
) -> RefractionDatumStaticsResult:
    if not isinstance(result, RefractionDatumStaticsResult):
        raise RefractionStaticArtifactError(
            'result must be a RefractionDatumStaticsResult instance'
        )
    n_traces = _length(result.sorted_trace_index, name='sorted_trace_index')
    if n_traces <= 0:
        raise RefractionStaticArtifactError('sorted_trace_index must not be empty')
    for name in _COMPONENT_TRACE_ARRAY_NAMES:
        if _length(getattr(result, name), name=name) != n_traces:
            raise RefractionStaticArtifactError(
                f'trace-order array length mismatch for {name}'
            )
    n_nodes = _length(result.node_id, name='node_id')
    for name in _COMPONENT_NODE_ARRAY_NAMES:
        if _length(getattr(result, name), name=name) != n_nodes:
            raise RefractionStaticArtifactError(f'node array length mismatch for {name}')
    n_source = _length(result.source_endpoint_key, name='source_endpoint_key')
    for name in _COMPONENT_SOURCE_ARRAY_NAMES:
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
    n_receiver = _length(result.receiver_endpoint_key, name='receiver_endpoint_key')
    for name in _COMPONENT_RECEIVER_ARRAY_NAMES:
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
        names=('source_endpoint_key_sorted', 'receiver_endpoint_key_sorted'),
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
    for name in _COMPONENT_STATUS_ARRAY_NAMES:
        _validate_status_array(getattr(result, name), name=name)
    return result


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


def _length(value: object, *, name: str) -> int:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise RefractionStaticArtifactError(f'{name} must be one-dimensional')
    return int(arr.shape[0])


def _validate_status_array(value: object, *, name: str) -> None:
    unknown = sorted(
        {
            str(item)
            for item in np.asarray(value).tolist()
            if str(item) not in REFRACTION_STATIC_STATUSES
        }
    )
    if unknown:
        raise RefractionStaticArtifactError(
            f'unknown status array values in {name}: {unknown}'
        )


def _scalar_str(value: object) -> np.ndarray:
    return np.asarray(str(value), dtype='<U128')


def _int_array(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.int64)


def _float_array(value: object) -> np.ndarray:
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


_COMPONENT_TRACE_ARRAY_NAMES = (
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

_COMPONENT_NODE_ARRAY_NAMES = (
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

_COMPONENT_SOURCE_ARRAY_NAMES = (
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

_COMPONENT_RECEIVER_ARRAY_NAMES = (
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

_COMPONENT_STATUS_ARRAY_NAMES = (
    'trace_static_status_sorted',
    'node_solution_status',
    'node_weathering_status',
    'node_datum_status',
    'source_datum_status',
    'receiver_datum_status',
)


__all__ = [
    '_component_columns',
    '_component_qc_stats_ms',
    '_component_rows',
    '_static_component_qc_endpoint_rows',
    '_static_component_qc_trace_rows',
    'build_refraction_static_component_qc_arrays',
    'build_refraction_static_component_qc_payload',
    'write_refraction_static_component_qc_artifacts',
    'write_refraction_static_components_csv',
]
