"""First-break residual, time-export, and fit-QC artifact writers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_types import RefractionDatumStaticsResult
from app.services.refraction_static_artifacts.contract import (
    _FIRST_BREAK_FIT_QC_COLUMNS,
    _FIRST_BREAK_TIME_EXPORT_COLUMNS,
    _RESIDUAL_COLUMNS,
    ARTIFACT_VERSION,
    FIRST_BREAK_FIT_QC_RESIDUAL_SIGN,
    FIRST_BREAK_TIME_EXPORT_FORMAT_NAME,
    FIRST_BREAK_TIME_EXPORT_FORMAT_VERSION,
    FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    SIGN_CONVENTION,
    WORKFLOW,
)
from app.services.refraction_static_artifacts.formatters import (
    _csv_bool,
    _csv_cell_id,
    _csv_float,
    _csv_identifier,
    _csv_layer_index,
    _csv_meters,
    _csv_ms,
)
from app.services.refraction_static_artifacts.io import (
    _assert_strict_json,
    _validate_no_object_arrays,
    _write_csv_atomic,
    _write_json_atomic,
    _write_npz_atomic,
)
from app.services.refraction_static_artifacts.row_context import (
    _bool_array,
    _cell_id_float_array,
    _first_break_export_layer_context,
    _first_break_fit_inline_crossline,
    _float_array,
    _int_array,
    _midpoint_coordinate,
    _residual_rejection_reason,
    _residual_row_cell_context,
    _residual_row_layer_context,
    _residual_row_string_context,
    _residual_row_velocity_context,
    _row_endpoint_float_context,
    _row_endpoint_id_context,
    _source_job_id,
    _string_array,
)
from app.services.refraction_static_artifacts.stats import (
    _residual_stat,
    _status_counts,
)
from app.services.refraction_static_artifacts.validation import _validate_result


def write_first_break_residuals_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
    req: RefractionStaticApplyRequest | None = None,
) -> None:
    values = _validate_result(result)
    rows = _first_break_residual_rows(values.result, req=req)
    _write_csv_atomic(Path(path), _RESIDUAL_COLUMNS, rows)


def write_refraction_first_break_time_export_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
    req: RefractionStaticApplyRequest | None = None,
    source_job_id: str | None = None,
) -> None:
    values = _validate_result(result)
    rows = _first_break_time_export_rows(
        values.result,
        req=req,
        source_job_id=source_job_id,
    )
    _write_csv_atomic(Path(path), _FIRST_BREAK_TIME_EXPORT_COLUMNS, rows)


def write_refraction_first_break_fit_qc_csv(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest | None = None,
    path: Path,
) -> None:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req) if req is not None else None
    arrays = build_refraction_first_break_fit_qc_arrays(
        result=values.result,
        req=request,
    )
    rows = _first_break_fit_qc_rows(arrays)
    _write_csv_atomic(Path(path), _FIRST_BREAK_FIT_QC_COLUMNS, rows)


def write_refraction_first_break_fit_qc_npz(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest | None = None,
    path: Path,
) -> None:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req) if req is not None else None
    arrays = build_refraction_first_break_fit_qc_arrays(
        result=values.result,
        req=request,
    )
    _write_npz_atomic(Path(path), arrays)


def write_refraction_first_break_fit_qc_json(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest | None = None,
    path: Path,
) -> dict[str, Any]:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req) if req is not None else None
    payload = build_refraction_first_break_fit_qc_payload(
        result=values.result,
        req=request,
    )
    _write_json_atomic(Path(path), payload)
    return payload


def build_refraction_first_break_fit_qc_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest | None = None,
) -> dict[str, np.ndarray]:
    """Build the viewer-ready observed-modeled first-break fit QC arrays."""
    values = _validate_result(result)
    r = values.result
    request = RefractionStaticApplyRequest.model_validate(req) if req is not None else None
    source_key_by_row = _residual_row_string_context(
        r,
        'row_source_endpoint_key',
    )
    receiver_key_by_row = _residual_row_string_context(
        r,
        'row_receiver_endpoint_key',
    )
    source_id_by_row = _row_endpoint_id_context(
        r,
        endpoint='source',
        row_endpoint_key=source_key_by_row,
        value_field='source_id',
    )
    receiver_id_by_row = _row_endpoint_id_context(
        r,
        endpoint='receiver',
        row_endpoint_key=receiver_key_by_row,
        value_field='receiver_id',
    )
    source_x_m = _row_endpoint_float_context(
        r,
        endpoint='source',
        row_endpoint_key=source_key_by_row,
        value_field='source_x_m',
    )
    source_y_m = _row_endpoint_float_context(
        r,
        endpoint='source',
        row_endpoint_key=source_key_by_row,
        value_field='source_y_m',
    )
    receiver_x_m = _row_endpoint_float_context(
        r,
        endpoint='receiver',
        row_endpoint_key=receiver_key_by_row,
        value_field='receiver_x_m',
    )
    receiver_y_m = _row_endpoint_float_context(
        r,
        endpoint='receiver',
        row_endpoint_key=receiver_key_by_row,
        value_field='receiver_y_m',
    )
    midpoint_x_m = _midpoint_coordinate(source_x_m, receiver_x_m)
    midpoint_y_m = _midpoint_coordinate(source_y_m, receiver_y_m)
    inline_m, crossline_m = _first_break_fit_inline_crossline(
        midpoint_x_m=midpoint_x_m,
        midpoint_y_m=midpoint_y_m,
        req=request,
    )
    layer_kind_by_row, _layer_index_by_row = _residual_row_layer_context(r)
    rejection_reason_by_row = _residual_row_string_context(
        r,
        'row_rejection_reason',
    )
    cell_id_by_row, cell_ix_by_row, cell_iy_by_row = _residual_row_cell_context(
        r,
        req=request,
    )
    used = np.asarray(r.used_row_mask, dtype=bool)
    rejection_reason = np.asarray(
        [
            _residual_rejection_reason(
                used=bool(used[row_index]),
                rejected_by_robust=bool(r.rejected_by_robust_mask[row_index]),
                explicit_reason=rejection_reason_by_row[row_index],
            )
            for row_index in range(values.n_rows)
        ],
        dtype='<U64',
    )
    status = np.where(used, 'ok', 'rejected')
    arrays = {
        'observation_index': np.arange(values.n_rows, dtype=np.int64),
        'sorted_trace_index': _int_array(r.row_trace_index_sorted),
        'trace_index_sorted': _int_array(r.row_trace_index_sorted),
        'source_endpoint_key': _string_array(source_key_by_row),
        'receiver_endpoint_key': _string_array(receiver_key_by_row),
        'source_id': _string_array(source_id_by_row),
        'receiver_id': _string_array(receiver_id_by_row),
        'source_node_id': _int_array(r.row_source_node_id),
        'receiver_node_id': _int_array(r.row_receiver_node_id),
        'source_x_m': _float_array(source_x_m),
        'source_y_m': _float_array(source_y_m),
        'receiver_x_m': _float_array(receiver_x_m),
        'receiver_y_m': _float_array(receiver_y_m),
        'midpoint_x_m': _float_array(midpoint_x_m),
        'midpoint_y_m': _float_array(midpoint_y_m),
        'inline_m': _float_array(inline_m),
        'crossline_m': _float_array(crossline_m),
        'offset_m': _float_array(r.row_distance_m),
        'observed_first_break_time_s': _float_array(r.observed_pick_time_s),
        'modeled_first_break_time_s': _float_array(r.modeled_pick_time_s),
        'residual_time_s': _float_array(r.residual_time_s),
        'residual_s': _float_array(r.residual_time_s),
        'residual_time_ms': _float_array(r.residual_time_s * 1000.0),
        'layer_kind': _string_array(layer_kind_by_row),
        'cell_id': _cell_id_float_array(cell_id_by_row),
        'cell_ix': _cell_id_float_array(cell_ix_by_row),
        'cell_iy': _cell_id_float_array(cell_iy_by_row),
        'used_for_inversion': _bool_array(used),
        'used_in_solve': _bool_array(used),
        'rejection_reason': _string_array(rejection_reason),
        'reject_reason': _string_array(rejection_reason),
        'status': _string_array(status),
        'sign_convention': _string_array(
            np.full(values.n_rows, SIGN_CONVENTION, dtype=f'<U{len(SIGN_CONVENTION)}')
        ),
    }
    _validate_no_object_arrays(
        arrays,
        artifact_name=REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    )
    return arrays


def build_refraction_first_break_fit_qc_payload(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest | None = None,
) -> dict[str, Any]:
    """Build the strict-JSON schema and summary for first-break fit QC."""
    arrays = build_refraction_first_break_fit_qc_arrays(result=result, req=req)
    used = np.asarray(arrays['used_for_inversion'], dtype=bool)
    residual_s = np.asarray(arrays['residual_time_s'], dtype=np.float64)
    used_residual_s = residual_s[used]
    residual_ms = residual_s * 1000.0
    used_residual_ms = residual_ms[used]
    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'schema_version': 1,
        'kind': 'refraction_first_break_fit_qc',
        'workflow': WORKFLOW,
        'sign_convention': SIGN_CONVENTION,
        'residual_sign': FIRST_BREAK_FIT_QC_RESIDUAL_SIGN,
        'residual_definition': (
            'residual_time_s = observed_first_break_time_s - '
            'modeled_first_break_time_s'
        ),
        'modeled_time_definition': {
            'general': (
                'modeled_first_break_time_s = source_time_term_s + '
                'receiver_time_term_s + moveout_or_cell_path_time_s'
            ),
            'midpoint_cell': (
                'moveout_or_cell_path_time_s = offset_m * '
                'cell_slowness_s_per_m'
            ),
            'global_velocity': (
                'moveout_or_cell_path_time_s = offset_m / velocity_m_s'
            ),
        },
        'columns': list(_FIRST_BREAK_FIT_QC_COLUMNS),
        'row_count': int(arrays['observation_index'].shape[0]),
        'used_count': int(np.count_nonzero(used)),
        'rejected_count': int(np.count_nonzero(~used)),
        'status_counts': _status_counts(arrays['status']),
        'rejection_reason_counts': _status_counts(arrays['rejection_reason']),
        'layer_kind_counts': _status_counts(arrays['layer_kind']),
        'residual_summary': {
            'all_rms_s': _residual_stat(residual_s, 'rms'),
            'all_mad_s': _residual_stat(residual_s, 'mad'),
            'all_mean_s': _residual_stat(residual_s, 'mean'),
            'all_p95_abs_s': _residual_stat(residual_s, 'p95_abs'),
            'all_rms_ms': _residual_stat(residual_ms, 'rms'),
            'all_mad_ms': _residual_stat(residual_ms, 'mad'),
            'all_mean_ms': _residual_stat(residual_ms, 'mean'),
            'all_p95_abs_ms': _residual_stat(residual_ms, 'p95_abs'),
            'used_rms_s': _residual_stat(used_residual_s, 'rms'),
            'used_mad_s': _residual_stat(used_residual_s, 'mad'),
            'used_mean_s': _residual_stat(used_residual_s, 'mean'),
            'used_p95_abs_s': _residual_stat(used_residual_s, 'p95_abs'),
            'used_rms_ms': _residual_stat(used_residual_ms, 'rms'),
            'used_mad_ms': _residual_stat(used_residual_ms, 'mad'),
            'used_mean_ms': _residual_stat(used_residual_ms, 'mean'),
            'used_p95_abs_ms': _residual_stat(used_residual_ms, 'p95_abs'),
        },
        'artifacts': {
            'csv': REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
            'npz': REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
            'json': REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
        },
    }
    _assert_strict_json(payload, artifact_name=REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME)
    return payload


def _first_break_residual_rows(
    result: RefractionDatumStaticsResult,
    *,
    req: RefractionStaticApplyRequest | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cell_id_by_row, cell_ix_by_row, cell_iy_by_row = _residual_row_cell_context(
        result,
        req=req,
    )
    layer_kind_by_row, layer_index_by_row = _residual_row_layer_context(result)
    source_key_by_row = _residual_row_string_context(
        result,
        'row_source_endpoint_key',
    )
    receiver_key_by_row = _residual_row_string_context(
        result,
        'row_receiver_endpoint_key',
    )
    rejection_reason_by_row = _residual_row_string_context(
        result,
        'row_rejection_reason',
    )
    row_velocity_m_s = _residual_row_velocity_context(result)
    for row_index in range(int(result.row_trace_index_sorted.shape[0])):
        rejected_by_robust = bool(result.rejected_by_robust_mask[row_index])
        used = bool(result.used_row_mask[row_index])
        rejection_reason = _residual_rejection_reason(
            used=used,
            rejected_by_robust=rejected_by_robust,
            explicit_reason=rejection_reason_by_row[row_index],
        )
        rows.append(
            {
                'row_index': row_index,
                'observation_index': row_index,
                'sorted_trace_index': int(result.row_trace_index_sorted[row_index]),
                'source_node_id': int(result.row_source_node_id[row_index]),
                'receiver_node_id': int(result.row_receiver_node_id[row_index]),
                'distance_m': _csv_float(result.row_distance_m[row_index]),
                'observed_pick_time_ms': _csv_ms(result.observed_pick_time_s[row_index]),
                'observed_pick_time_s': _csv_float(result.observed_pick_time_s[row_index]),
                'modeled_pick_time_ms': _csv_ms(result.modeled_pick_time_s[row_index]),
                'modeled_pick_time_s': _csv_float(result.modeled_pick_time_s[row_index]),
                'residual_ms': _csv_ms(result.residual_time_s[row_index]),
                'residual_s': _csv_float(result.residual_time_s[row_index]),
                'used': _csv_bool(used),
                'used_in_solve': _csv_bool(used),
                'rejected_by_robust': _csv_bool(rejected_by_robust),
                'rejection_reason': rejection_reason,
                'cell_id': _csv_cell_id(cell_id_by_row[row_index]),
                'cell_ix': _csv_cell_id(cell_ix_by_row[row_index]),
                'cell_iy': _csv_cell_id(cell_iy_by_row[row_index]),
                'trace_index_sorted': int(result.row_trace_index_sorted[row_index]),
                'layer_kind': str(layer_kind_by_row[row_index]),
                'layer_index': _csv_layer_index(layer_index_by_row[row_index]),
                'source_endpoint_key': str(source_key_by_row[row_index]),
                'receiver_endpoint_key': str(receiver_key_by_row[row_index]),
                'offset_m': _csv_float(result.row_distance_m[row_index]),
                'residual_time_s': _csv_float(result.residual_time_s[row_index]),
                'midpoint_cell_id': _csv_cell_id(cell_id_by_row[row_index]),
                'row_velocity_m_s': _csv_float(row_velocity_m_s[row_index]),
            }
        )
    return rows


def _first_break_time_export_rows(
    result: RefractionDatumStaticsResult,
    *,
    req: RefractionStaticApplyRequest | None = None,
    source_job_id: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    layer_kind_by_row, _layer_index_by_row = _first_break_export_layer_context(
        result,
    )
    source_key_by_row = _residual_row_string_context(
        result,
        'row_source_endpoint_key',
    )
    receiver_key_by_row = _residual_row_string_context(
        result,
        'row_receiver_endpoint_key',
    )
    rejection_reason_by_row = _residual_row_string_context(
        result,
        'row_rejection_reason',
    )
    source_id_by_row = _row_endpoint_id_context(
        result,
        endpoint='source',
        row_endpoint_key=source_key_by_row,
        value_field='source_id',
    )
    receiver_id_by_row = _row_endpoint_id_context(
        result,
        endpoint='receiver',
        row_endpoint_key=receiver_key_by_row,
        value_field='receiver_id',
    )
    job_id = _source_job_id(source_job_id, req)

    for row_index in range(int(result.row_trace_index_sorted.shape[0])):
        rejected_by_robust = bool(result.rejected_by_robust_mask[row_index])
        used = bool(result.used_row_mask[row_index])
        rejection_reason = _residual_rejection_reason(
            used=used,
            rejected_by_robust=rejected_by_robust,
            explicit_reason=rejection_reason_by_row[row_index],
        )
        rows.append(
            {
                'format_name': FIRST_BREAK_TIME_EXPORT_FORMAT_NAME,
                'format_version': FIRST_BREAK_TIME_EXPORT_FORMAT_VERSION,
                'source_job_id': job_id,
                'observation_index': row_index,
                'sorted_trace_index': int(result.row_trace_index_sorted[row_index]),
                'source_endpoint_key': str(source_key_by_row[row_index]),
                'receiver_endpoint_key': str(receiver_key_by_row[row_index]),
                'source_id': _csv_identifier(source_id_by_row[row_index]),
                'receiver_id': _csv_identifier(receiver_id_by_row[row_index]),
                'offset_m': _csv_meters(result.row_distance_m[row_index]),
                'layer_kind': str(layer_kind_by_row[row_index]),
                'observed_pick_time_ms': _csv_ms(
                    result.observed_pick_time_s[row_index]
                ),
                'modeled_pick_time_ms': _csv_ms(
                    result.modeled_pick_time_s[row_index]
                ),
                'residual_ms': _csv_ms(result.residual_time_s[row_index]),
                'used_in_solve': _csv_bool(used),
                'reject_reason': rejection_reason,
                'sign_convention': FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION,
            }
        )
    return rows


def _first_break_fit_qc_rows(
    arrays: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_rows = int(np.asarray(arrays['observation_index']).shape[0])
    for row_index in range(n_rows):
        rejection_reason = str(arrays['rejection_reason'][row_index])
        rows.append(
            {
                'observation_index': int(arrays['observation_index'][row_index]),
                'sorted_trace_index': int(arrays['sorted_trace_index'][row_index]),
                'trace_index_sorted': int(arrays['trace_index_sorted'][row_index]),
                'source_endpoint_key': str(arrays['source_endpoint_key'][row_index]),
                'receiver_endpoint_key': str(
                    arrays['receiver_endpoint_key'][row_index]
                ),
                'source_id': _csv_identifier(arrays['source_id'][row_index]),
                'receiver_id': _csv_identifier(arrays['receiver_id'][row_index]),
                'source_node_id': int(arrays['source_node_id'][row_index]),
                'receiver_node_id': int(arrays['receiver_node_id'][row_index]),
                'source_x_m': _csv_float(arrays['source_x_m'][row_index]),
                'source_y_m': _csv_float(arrays['source_y_m'][row_index]),
                'receiver_x_m': _csv_float(arrays['receiver_x_m'][row_index]),
                'receiver_y_m': _csv_float(arrays['receiver_y_m'][row_index]),
                'midpoint_x_m': _csv_float(arrays['midpoint_x_m'][row_index]),
                'midpoint_y_m': _csv_float(arrays['midpoint_y_m'][row_index]),
                'inline_m': _csv_float(arrays['inline_m'][row_index]),
                'crossline_m': _csv_float(arrays['crossline_m'][row_index]),
                'offset_m': _csv_float(arrays['offset_m'][row_index]),
                'observed_first_break_time_s': _csv_float(
                    arrays['observed_first_break_time_s'][row_index]
                ),
                'modeled_first_break_time_s': _csv_float(
                    arrays['modeled_first_break_time_s'][row_index]
                ),
                'residual_time_s': _csv_float(arrays['residual_time_s'][row_index]),
                'residual_s': _csv_float(arrays['residual_s'][row_index]),
                'residual_time_ms': _csv_float(arrays['residual_time_ms'][row_index]),
                'layer_kind': str(arrays['layer_kind'][row_index]),
                'cell_id': _csv_cell_id(arrays['cell_id'][row_index]),
                'cell_ix': _csv_cell_id(arrays['cell_ix'][row_index]),
                'cell_iy': _csv_cell_id(arrays['cell_iy'][row_index]),
                'used_for_inversion': _csv_bool(
                    arrays['used_for_inversion'][row_index]
                ),
                'used_in_solve': _csv_bool(arrays['used_in_solve'][row_index]),
                'rejection_reason': rejection_reason,
                'reject_reason': rejection_reason,
                'status': str(arrays['status'][row_index]),
                'sign_convention': str(arrays['sign_convention'][row_index]),
            }
        )
    return rows


__all__ = [
    'build_refraction_first_break_fit_qc_arrays',
    'build_refraction_first_break_fit_qc_payload',
    'write_first_break_residuals_csv',
    'write_refraction_first_break_fit_qc_csv',
    'write_refraction_first_break_fit_qc_json',
    'write_refraction_first_break_fit_qc_npz',
    'write_refraction_first_break_time_export_csv',
]
