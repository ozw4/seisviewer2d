"""First-break residual, fit-QC, and reduced-time artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from app.contracts.statics.refraction.apply import RefractionStaticApplyRequest
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
)

from app.services.refraction_static_cell_coordinates import (
    effective_refraction_cell_grid_config,
    project_refraction_cell_points,
)
from app.statics.refraction.artifacts.arrays import (
    _bool_array,
    _cell_id_float_array,
    _float_array,
    _int_array,
    _required_positive_qc_int,
    _scalar_str,
    _string_array,
)
from app.statics.refraction.artifacts.cell_velocity import (
    _request_has_cell_velocity_layer,
)
from app.statics.refraction.artifacts.contract import (
    _FIRST_BREAK_FIT_QC_COLUMNS,
    _FIRST_BREAK_TIME_EXPORT_COLUMNS,
    _REDUCED_TIME_QC_COLUMNS,
    _RESIDUAL_COLUMNS,
    ARTIFACT_VERSION,
    FIRST_BREAK_FIT_QC_RESIDUAL_SIGN,
    FIRST_BREAK_TIME_EXPORT_FORMAT_NAME,
    FIRST_BREAK_TIME_EXPORT_FORMAT_VERSION,
    FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION,
    REDUCED_TIME_QC_FORMULA,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_REDUCED_TIME_QC_CSV_NAME,
    REFRACTION_REDUCED_TIME_QC_JSON_NAME,
    REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
    RefractionStaticArtifactError,
    SIGN_CONVENTION,
    WORKFLOW,
)
from app.statics.refraction.artifacts.formatters import (
    _csv_bool,
    _csv_cell_id,
    _csv_float,
    _csv_identifier,
    _csv_layer_index,
    _csv_meters,
    _csv_ms,
    _float_or_nan,
    _json_float,
)
from app.statics.refraction.artifacts.io import (
    _assert_strict_json,
    _validate_no_object_arrays,
    _write_csv_atomic,
    _write_json_atomic,
    _write_npz_atomic,
)
from app.statics.refraction.artifacts.registry import (
    _request_cell_velocity_layer_kinds,
)
from app.statics.refraction.artifacts.stats import _residual_stat, _stat, _status_counts
from app.statics.refraction.artifacts.validation import _validate_result
from app.services.refraction_static_layer_config import normalize_refraction_static_layers

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

def write_refraction_reduced_time_qc_csv(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> None:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    arrays = build_refraction_reduced_time_qc_arrays(
        result=values.result,
        req=request,
    )
    rows = _reduced_time_qc_rows(arrays)
    _write_csv_atomic(Path(path), _REDUCED_TIME_QC_COLUMNS, rows)

def write_refraction_reduced_time_qc_npz(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> None:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    arrays = build_refraction_reduced_time_qc_arrays(
        result=values.result,
        req=request,
    )
    _write_npz_atomic(Path(path), arrays)

def write_refraction_reduced_time_qc_json(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> dict[str, Any]:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    payload = build_refraction_reduced_time_qc_payload(
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

def build_refraction_reduced_time_qc_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, np.ndarray]:
    """Build reduced-time first-break QC arrays without changing raw picks."""
    values = _validate_result(result)
    r = values.result
    request = RefractionStaticApplyRequest.model_validate(req)
    source_key_by_row = _residual_row_string_context(
        r,
        'row_source_endpoint_key',
    )
    receiver_key_by_row = _residual_row_string_context(
        r,
        'row_receiver_endpoint_key',
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
    gate_flags = _reduced_time_layer_gate_flags(request, r.row_distance_m)
    layer_gate_kind = _reduced_time_layer_gate_kind(
        layer_kind_by_row=layer_kind_by_row,
        gate_flags=gate_flags,
    )
    reduction_velocity = _reduced_time_reduction_velocity_by_row(
        result=r,
        req=request,
        layer_gate_kind=layer_gate_kind,
    )
    observed = np.asarray(r.observed_pick_time_s, dtype=np.float64)
    offset = np.asarray(r.row_distance_m, dtype=np.float64)
    reduced_time = np.full(values.n_rows, np.nan, dtype=np.float64)
    status = _reduced_time_status(
        observed_time_s=observed,
        offset_m=offset,
        reduction_velocity_m_s=reduction_velocity,
    )
    ok = status == 'ok'
    reduced_time[ok] = observed[ok] - offset[ok] / reduction_velocity[ok]

    arrays = {
        'trace_index_sorted': _int_array(r.row_trace_index_sorted),
        'source_endpoint_key': _string_array(source_key_by_row),
        'receiver_endpoint_key': _string_array(receiver_key_by_row),
        'offset_m': _float_array(offset),
        'inline_m': _float_array(inline_m),
        'crossline_m': _float_array(crossline_m),
        'observed_first_break_time_s': _float_array(observed),
        'reduction_velocity_m_s': _float_array(reduction_velocity),
        'reduced_time_s': _float_array(reduced_time),
        'reduced_time_ms': _float_array(reduced_time * 1000.0),
        'layer_gate_kind': _string_array(layer_gate_kind),
        'within_v1_gate': _bool_array(gate_flags['v1_direct_arrival']),
        'within_v2_t1_gate': _bool_array(gate_flags['v2_t1']),
        'within_v3_t2_gate': _bool_array(gate_flags['v3_t2']),
        'within_vsub_t3_gate': _bool_array(gate_flags['vsub_t3']),
        'used_for_inversion': _bool_array(r.used_row_mask),
        'status': _string_array(status),
        'reduction_velocity_mode': _scalar_str(
            request.reduced_time_qc.reduction_velocity_mode
        ),
    }
    _validate_no_object_arrays(
        arrays,
        artifact_name=REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
    )
    return arrays

def build_refraction_reduced_time_qc_payload(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, Any]:
    """Build the strict-JSON schema and summary for reduced-time QC."""
    request = RefractionStaticApplyRequest.model_validate(req)
    arrays = build_refraction_reduced_time_qc_arrays(result=result, req=request)
    status = np.asarray(arrays['status']).astype(str, copy=False)
    velocity = np.asarray(arrays['reduction_velocity_m_s'], dtype=np.float64)
    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'schema_version': 1,
        'kind': 'refraction_reduced_time_qc',
        'workflow': WORKFLOW,
        'sign_convention': SIGN_CONVENTION,
        'formula': REDUCED_TIME_QC_FORMULA,
        'reduction_velocity_mode': (
            request.reduced_time_qc.reduction_velocity_mode
        ),
        'fixed_velocity_m_s': _json_float(
            request.reduced_time_qc.fixed_velocity_m_s
        ),
        'columns': list(_REDUCED_TIME_QC_COLUMNS),
        'row_count': int(arrays['trace_index_sorted'].shape[0]),
        'used_count': int(np.count_nonzero(arrays['used_for_inversion'])),
        'status_counts': _status_counts(status),
        'layer_gate_kind_counts': _status_counts(arrays['layer_gate_kind']),
        'missing_velocity_count': int(
            np.count_nonzero(status == 'missing_reduction_velocity')
        ),
        'reduction_velocity_summary': {
            'min_m_s': _stat(velocity, 'min'),
            'max_m_s': _stat(velocity, 'max'),
            'median_m_s': _stat(velocity, 'median'),
        },
        'offset_gates': _reduced_time_gate_qc(request),
        'artifacts': {
            'csv': REFRACTION_REDUCED_TIME_QC_CSV_NAME,
            'npz': REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
            'json': REFRACTION_REDUCED_TIME_QC_JSON_NAME,
        },
    }
    _assert_strict_json(payload, artifact_name=REFRACTION_REDUCED_TIME_QC_JSON_NAME)
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

def _reduced_time_qc_rows(
    arrays: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_rows = int(np.asarray(arrays['trace_index_sorted']).shape[0])
    for row_index in range(n_rows):
        rows.append(
            {
                'trace_index_sorted': int(arrays['trace_index_sorted'][row_index]),
                'source_endpoint_key': str(arrays['source_endpoint_key'][row_index]),
                'receiver_endpoint_key': str(
                    arrays['receiver_endpoint_key'][row_index]
                ),
                'offset_m': _csv_float(arrays['offset_m'][row_index]),
                'inline_m': _csv_float(arrays['inline_m'][row_index]),
                'crossline_m': _csv_float(arrays['crossline_m'][row_index]),
                'observed_first_break_time_s': _csv_float(
                    arrays['observed_first_break_time_s'][row_index]
                ),
                'reduction_velocity_m_s': _csv_float(
                    arrays['reduction_velocity_m_s'][row_index]
                ),
                'reduced_time_s': _csv_float(arrays['reduced_time_s'][row_index]),
                'reduced_time_ms': _csv_float(arrays['reduced_time_ms'][row_index]),
                'layer_gate_kind': str(arrays['layer_gate_kind'][row_index]),
                'within_v1_gate': _csv_bool(arrays['within_v1_gate'][row_index]),
                'within_v2_t1_gate': _csv_bool(
                    arrays['within_v2_t1_gate'][row_index]
                ),
                'within_v3_t2_gate': _csv_bool(
                    arrays['within_v3_t2_gate'][row_index]
                ),
                'within_vsub_t3_gate': _csv_bool(
                    arrays['within_vsub_t3_gate'][row_index]
                ),
                'used_for_inversion': _csv_bool(
                    arrays['used_for_inversion'][row_index]
                ),
                'status': str(arrays['status'][row_index]),
            }
        )
    return rows

def _reduced_time_layer_gate_flags(
    req: RefractionStaticApplyRequest,
    offset_m: np.ndarray,
) -> dict[str, np.ndarray]:
    offset = np.asarray(offset_m, dtype=np.float64)
    flags = {
        'v1_direct_arrival': np.zeros(offset.shape, dtype=bool),
        'v2_t1': np.zeros(offset.shape, dtype=bool),
        'v3_t2': np.zeros(offset.shape, dtype=bool),
        'vsub_t3': np.zeros(offset.shape, dtype=bool),
    }
    first_layer = req.model.first_layer
    if first_layer is not None and first_layer.mode == 'estimate_direct_arrival':
        flags['v1_direct_arrival'] = _offset_gate_mask(
            offset,
            min_offset_m=first_layer.min_direct_offset_m,
            max_offset_m=first_layer.max_direct_offset_m,
            enabled=True,
        )
    for config in normalize_refraction_static_layers(req.model, enabled_only=False):
        if not config.enabled:
            continue
        flags[config.kind] = _offset_gate_mask(
            offset,
            min_offset_m=config.min_offset_m,
            max_offset_m=config.max_offset_m,
            enabled=True,
        )
    return {key: np.ascontiguousarray(value, dtype=bool) for key, value in flags.items()}

def _offset_gate_mask(
    offset_m: np.ndarray,
    *,
    min_offset_m: float | None,
    max_offset_m: float | None,
    enabled: bool,
) -> np.ndarray:
    offset = np.asarray(offset_m, dtype=np.float64)
    mask = np.zeros(offset.shape, dtype=bool)
    if not enabled:
        return mask
    mask = np.isfinite(offset)
    if min_offset_m is not None:
        mask &= offset >= float(min_offset_m)
    if max_offset_m is not None:
        mask &= offset <= float(max_offset_m)
    return np.ascontiguousarray(mask, dtype=bool)

def _reduced_time_layer_gate_kind(
    *,
    layer_kind_by_row: np.ndarray,
    gate_flags: Mapping[str, np.ndarray],
) -> np.ndarray:
    out = np.asarray(layer_kind_by_row).astype('<U32', copy=True)
    empty = out == ''
    for kind in ('v2_t1', 'v3_t2', 'vsub_t3', 'v1_direct_arrival'):
        mask = empty & np.asarray(gate_flags[kind], dtype=bool)
        out[mask] = kind
        empty &= ~mask
    return np.ascontiguousarray(out, dtype='<U32')

def _reduced_time_reduction_velocity_by_row(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    layer_gate_kind: np.ndarray,
) -> np.ndarray:
    mode = req.reduced_time_qc.reduction_velocity_mode
    n_rows = int(result.row_trace_index_sorted.shape[0])
    if mode == 'fixed':
        return np.full(
            n_rows,
            float(req.reduced_time_qc.fixed_velocity_m_s),
            dtype=np.float64,
        )
    if mode == 'initial_velocity':
        return _configured_reduction_velocity_by_row(
            req=req,
            layer_gate_kind=layer_gate_kind,
        )
    velocity = _residual_row_velocity_context(result)
    kind = np.asarray(layer_gate_kind).astype(str, copy=False)
    if np.any(kind == 'v1_direct_arrival'):
        velocity = np.asarray(velocity, dtype=np.float64).copy()
        velocity[kind == 'v1_direct_arrival'] = _float_or_nan(
            result.weathering_velocity_m_s
        )
    return np.ascontiguousarray(velocity, dtype=np.float64)

def _configured_reduction_velocity_by_row(
    *,
    req: RefractionStaticApplyRequest,
    layer_gate_kind: np.ndarray,
) -> np.ndarray:
    kind = np.asarray(layer_gate_kind).astype(str, copy=False)
    out = np.full(kind.shape, np.nan, dtype=np.float64)
    velocity_by_kind = _configured_initial_velocity_by_layer(req)
    for layer_kind, velocity in velocity_by_kind.items():
        out[kind == layer_kind] = velocity
    return np.ascontiguousarray(out, dtype=np.float64)

def _configured_initial_velocity_by_layer(
    req: RefractionStaticApplyRequest,
) -> dict[str, float]:
    values: dict[str, float] = {}
    first_layer_velocity = _configured_v1_velocity(req)
    if first_layer_velocity is not None:
        values['v1_direct_arrival'] = first_layer_velocity
    for config in normalize_refraction_static_layers(req.model, enabled_only=False):
        velocity = config.initial_velocity_m_s
        if velocity is None:
            velocity = config.fixed_velocity_m_s
        if velocity is None:
            continue
        velocity_f = _float_or_nan(velocity)
        if np.isfinite(velocity_f) and velocity_f > 0.0:
            values[config.kind] = velocity_f
    return values

def _configured_v1_velocity(req: RefractionStaticApplyRequest) -> float | None:
    first_layer = req.model.first_layer
    if first_layer is not None and first_layer.weathering_velocity_m_s is not None:
        return float(first_layer.weathering_velocity_m_s)
    if req.model.weathering_velocity_m_s is not None:
        return float(req.model.weathering_velocity_m_s)
    return None

def _reduced_time_status(
    *,
    observed_time_s: np.ndarray,
    offset_m: np.ndarray,
    reduction_velocity_m_s: np.ndarray,
) -> np.ndarray:
    observed = np.asarray(observed_time_s, dtype=np.float64)
    offset = np.asarray(offset_m, dtype=np.float64)
    velocity = np.asarray(reduction_velocity_m_s, dtype=np.float64)
    status = np.full(observed.shape, 'ok', dtype='<U32')
    missing_observed = ~np.isfinite(observed)
    missing_offset = np.isfinite(observed) & ~np.isfinite(offset)
    missing_velocity = (
        np.isfinite(observed)
        & np.isfinite(offset)
        & (~np.isfinite(velocity) | (velocity <= 0.0))
    )
    status[missing_observed] = 'missing_observed_time'
    status[missing_offset] = 'missing_offset'
    status[missing_velocity] = 'missing_reduction_velocity'
    return np.ascontiguousarray(status, dtype='<U32')

def _reduced_time_gate_qc(req: RefractionStaticApplyRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'v1_direct_arrival': {
            'enabled': False,
            'min_offset_m': None,
            'max_offset_m': None,
        },
        'v2_t1': {'enabled': False, 'min_offset_m': None, 'max_offset_m': None},
        'v3_t2': {'enabled': False, 'min_offset_m': None, 'max_offset_m': None},
        'vsub_t3': {'enabled': False, 'min_offset_m': None, 'max_offset_m': None},
    }
    first_layer = req.model.first_layer
    if first_layer is not None and first_layer.mode == 'estimate_direct_arrival':
        payload['v1_direct_arrival'] = {
            'enabled': True,
            'min_offset_m': _json_float(first_layer.min_direct_offset_m),
            'max_offset_m': _json_float(first_layer.max_direct_offset_m),
        }
    for config in normalize_refraction_static_layers(req.model, enabled_only=False):
        payload[config.kind] = {
            'enabled': bool(config.enabled),
            'min_offset_m': _json_float(config.min_offset_m),
            'max_offset_m': _json_float(config.max_offset_m),
        }
    return payload

def _source_job_id(
    source_job_id: str | None,
    req: RefractionStaticApplyRequest | None,
) -> str:
    raw = source_job_id
    if raw is None and req is not None:
        raw = getattr(req, 'source_job_id', None)
    return '' if raw is None else str(raw)

def _first_break_export_layer_context(
    result: RefractionDatumStaticsResult,
) -> tuple[np.ndarray, np.ndarray]:
    return _residual_row_layer_context(result)

def _row_endpoint_id_context(
    result: RefractionDatumStaticsResult,
    *,
    endpoint: str,
    row_endpoint_key: np.ndarray,
    value_field: str,
) -> np.ndarray:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    keys = (
        result.source_endpoint_key
        if endpoint == 'source'
        else result.receiver_endpoint_key
    )
    values = getattr(result, value_field)
    lookup = {
        str(key): value
        for key, value in zip(
            np.asarray(keys).tolist(),
            np.asarray(values).tolist(),
            strict=True,
        )
    }
    out = np.full(n_rows, '', dtype=object)
    for index, raw_key in enumerate(row_endpoint_key.tolist()):
        value = lookup.get(str(raw_key))
        if value is not None:
            out[index] = value
    return np.ascontiguousarray(out, dtype=object)

def _row_endpoint_float_context(
    result: RefractionDatumStaticsResult,
    *,
    endpoint: str,
    row_endpoint_key: np.ndarray,
    value_field: str,
) -> np.ndarray:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    keys = (
        result.source_endpoint_key
        if endpoint == 'source'
        else result.receiver_endpoint_key
    )
    values = getattr(result, value_field)
    lookup = {
        str(key): float(value)
        for key, value in zip(
            np.asarray(keys).tolist(),
            np.asarray(values).tolist(),
            strict=True,
        )
    }
    out = np.full(n_rows, np.nan, dtype=np.float64)
    for index, raw_key in enumerate(row_endpoint_key.tolist()):
        value = lookup.get(str(raw_key))
        if value is not None:
            out[index] = value
    return np.ascontiguousarray(out, dtype=np.float64)

def _first_break_row_time_terms(
    result: RefractionDatumStaticsResult,
    *,
    layer_kind_by_row: np.ndarray,
    source_key_by_row: np.ndarray,
    receiver_key_by_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    source = np.full(n_rows, np.nan, dtype=np.float64)
    receiver = np.full(n_rows, np.nan, dtype=np.float64)
    source_t1_sorted = np.asarray(
        result.source_half_intercept_time_s_sorted,
        dtype=np.float64,
    )
    receiver_t1_sorted = np.asarray(
        result.receiver_half_intercept_time_s_sorted,
        dtype=np.float64,
    )
    source_t2 = _endpoint_time_lookup(
        result.source_endpoint_key,
        result.source_t2_time_s,
    )
    receiver_t2 = _endpoint_time_lookup(
        result.receiver_endpoint_key,
        result.receiver_t2_time_s,
    )
    source_t3 = _endpoint_time_lookup(
        result.source_endpoint_key,
        result.source_t3_time_s,
    )
    receiver_t3 = _endpoint_time_lookup(
        result.receiver_endpoint_key,
        result.receiver_t3_time_s,
    )
    for row_index, trace_index in enumerate(result.row_trace_index_sorted.tolist()):
        kind = str(layer_kind_by_row[row_index])
        if kind == 'v3_t2':
            source[row_index] = source_t2.get(
                str(source_key_by_row[row_index]),
                np.nan,
            )
            receiver[row_index] = receiver_t2.get(
                str(receiver_key_by_row[row_index]),
                np.nan,
            )
        elif kind == 'vsub_t3':
            source[row_index] = source_t3.get(
                str(source_key_by_row[row_index]),
                np.nan,
            )
            receiver[row_index] = receiver_t3.get(
                str(receiver_key_by_row[row_index]),
                np.nan,
            )
        elif kind == 'v2_t1':
            source[row_index] = source_t1_sorted[int(trace_index)]
            receiver[row_index] = receiver_t1_sorted[int(trace_index)]
    return (
        np.ascontiguousarray(source, dtype=np.float64),
        np.ascontiguousarray(receiver, dtype=np.float64),
    )

def _endpoint_time_lookup(
    endpoint_key: np.ndarray,
    values: np.ndarray | None,
) -> dict[str, float]:
    if values is None:
        return {}
    return {
        str(key): float(value)
        for key, value in zip(
            np.asarray(endpoint_key).tolist(),
            np.asarray(values, dtype=np.float64).tolist(),
            strict=True,
        )
    }

def _first_break_moveout_time_s(
    *,
    modeled_pick_time_s: np.ndarray,
    source_time_term_s: np.ndarray,
    receiver_time_term_s: np.ndarray,
) -> np.ndarray:
    modeled = np.asarray(modeled_pick_time_s, dtype=np.float64)
    source = np.asarray(source_time_term_s, dtype=np.float64)
    receiver = np.asarray(receiver_time_term_s, dtype=np.float64)
    out = modeled - source - receiver
    invalid = ~np.isfinite(modeled) | ~np.isfinite(source) | ~np.isfinite(receiver)
    out[invalid] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)

def _midpoint_coordinate(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    out = 0.5 * (left_arr + right_arr)
    out[~np.isfinite(left_arr) | ~np.isfinite(right_arr)] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)

def _first_break_fit_inline_crossline(
    *,
    midpoint_x_m: np.ndarray,
    midpoint_y_m: np.ndarray,
    req: RefractionStaticApplyRequest | None,
) -> tuple[np.ndarray, np.ndarray]:
    midpoint_x = np.asarray(midpoint_x_m, dtype=np.float64)
    midpoint_y = np.asarray(midpoint_y_m, dtype=np.float64)
    inline = np.full(midpoint_x.shape, np.nan, dtype=np.float64)
    crossline = np.full(midpoint_x.shape, np.nan, dtype=np.float64)
    if req is None or req.model.refractor_cell is None:
        return inline, crossline
    refractor_cell = req.model.refractor_cell
    if refractor_cell.coordinate_mode != 'line_2d_projected':
        return inline, crossline
    projected = project_refraction_cell_points(
        x_m=midpoint_x,
        y_m=midpoint_y,
        mode=refractor_cell.coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    if projected.projected_inline_m is not None:
        inline = projected.projected_inline_m
    if projected.projected_crossline_m is not None:
        crossline = projected.projected_crossline_m
    return (
        np.ascontiguousarray(inline, dtype=np.float64),
        np.ascontiguousarray(crossline, dtype=np.float64),
    )

def _residual_row_layer_context(
    result: RefractionDatumStaticsResult,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    kind = np.full(n_rows, '', dtype='<U16')
    index = np.zeros(n_rows, dtype=np.int64)
    raw_kind = getattr(result, 'row_layer_kind', None)
    raw_index = getattr(result, 'row_layer_index', None)
    if raw_kind is not None:
        kind = np.asarray(raw_kind).astype('<U16', copy=False)
        if kind.shape != (n_rows,):
            raise RefractionStaticArtifactError(
                'row_layer_kind length must match residual rows'
            )
    if raw_index is not None:
        index = np.asarray(raw_index, dtype=np.int64)
        if index.shape != (n_rows,):
            raise RefractionStaticArtifactError(
                'row_layer_index length must match residual rows'
            )
    return np.ascontiguousarray(kind), np.ascontiguousarray(index)

def _residual_row_string_context(
    result: RefractionDatumStaticsResult,
    field: str,
) -> np.ndarray:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    raw = getattr(result, field, None)
    if raw is None:
        return np.full(n_rows, '', dtype=object)
    out = np.asarray(raw, dtype=object)
    if out.shape != (n_rows,):
        raise RefractionStaticArtifactError(
            f'{field} length must match residual rows'
        )
    return np.ascontiguousarray(out, dtype=object)

def _residual_row_velocity_context(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    raw = getattr(result, 'row_velocity_m_s', None)
    if raw is not None:
        out = np.asarray(raw, dtype=np.float64)
        if out.shape != (n_rows,):
            raise RefractionStaticArtifactError(
                'row_velocity_m_s length must match residual rows'
            )
        return np.ascontiguousarray(out, dtype=np.float64)
    return _row_velocity_from_cell_or_scalar(result, n_rows)

def _row_velocity_from_cell_or_scalar(
    result: RefractionDatumStaticsResult,
    n_rows: int,
) -> np.ndarray:
    out = np.full(n_rows, _float_or_nan(result.bedrock_velocity_m_s), dtype=np.float64)
    if result.row_midpoint_cell_id is None or result.cell_bedrock_velocity_m_s is None:
        return np.ascontiguousarray(out, dtype=np.float64)
    cell_id = np.asarray(result.row_midpoint_cell_id, dtype=np.int64)
    if cell_id.shape != (n_rows,):
        return np.ascontiguousarray(out, dtype=np.float64)
    velocity = np.asarray(result.cell_bedrock_velocity_m_s, dtype=np.float64)
    active_cell_id = result.active_cell_id
    if active_cell_id is not None:
        active = np.asarray(active_cell_id, dtype=np.int64)
        if active.shape == velocity.shape:
            out.fill(np.nan)
            for raw_cell, raw_velocity in zip(active.tolist(), velocity.tolist(), strict=True):
                rows = cell_id == int(raw_cell)
                out[rows] = float(raw_velocity)
            return np.ascontiguousarray(out, dtype=np.float64)
    if velocity.ndim == 1 and velocity.size > int(np.max(cell_id, initial=-1)):
        valid = (cell_id >= 0) & (cell_id < int(velocity.size))
        out[valid] = velocity[cell_id[valid]]
    return np.ascontiguousarray(out, dtype=np.float64)

def _residual_row_cell_context(
    result: RefractionDatumStaticsResult,
    *,
    req: RefractionStaticApplyRequest | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    empty = np.full(n_rows, -1, dtype=np.int64)
    request = RefractionStaticApplyRequest.model_validate(req) if req is not None else None
    request_cell_layer_kinds = (
        _request_cell_velocity_layer_kinds(request) if request is not None else ()
    )
    request_has_cell_velocity = bool(request_cell_layer_kinds)
    if (
        not request_has_cell_velocity
        and result.bedrock_velocity_mode != 'solve_cell'
    ):
        return empty, empty.copy(), empty.copy()

    if result.row_midpoint_cell_id is None:
        raise RefractionStaticArtifactError(
            'cell velocity residual rows require row_midpoint_cell_id'
        )
    cell_id = np.ascontiguousarray(result.row_midpoint_cell_id, dtype=np.int64)
    if cell_id.shape != (n_rows,):
        raise RefractionStaticArtifactError(
            'row_midpoint_cell_id length must match residual rows'
        )
    if request is not None and request_has_cell_velocity:
        layer_kind_by_row, _layer_index_by_row = _residual_row_layer_context(result)
        if np.any(layer_kind_by_row.astype(str, copy=False) != ''):
            supported_layer = np.isin(
                layer_kind_by_row.astype(str, copy=False),
                np.asarray(request_cell_layer_kinds, dtype=str),
            )
            cell_id = np.where(supported_layer, cell_id, -1)
    number_of_cell_x = _residual_cell_x_count(result=result, req=req)
    cell_ix = np.full(n_rows, -1, dtype=np.int64)
    cell_iy = np.full(n_rows, -1, dtype=np.int64)
    if number_of_cell_x is not None:
        valid = cell_id >= 0
        cell_ix[valid] = cell_id[valid] % number_of_cell_x
        cell_iy[valid] = cell_id[valid] // number_of_cell_x
    return (
        np.ascontiguousarray(cell_id, dtype=np.int64),
        np.ascontiguousarray(cell_ix, dtype=np.int64),
        np.ascontiguousarray(cell_iy, dtype=np.int64),
    )

def _residual_cell_x_count(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest | None,
) -> int | None:
    if req is not None:
        request = RefractionStaticApplyRequest.model_validate(req)
        refractor_cell = request.model.refractor_cell
        if _request_has_cell_velocity_layer(request) and refractor_cell is not None:
            return int(
                effective_refraction_cell_grid_config(
                    refractor_cell
                ).number_of_cell_x
            )
    raw = result.qc.get('number_of_cell_x')
    if raw is None:
        return None
    return _required_positive_qc_int(result.qc, 'number_of_cell_x')

def _residual_rejection_reason(
    *,
    used: bool,
    rejected_by_robust: bool,
    explicit_reason: object = '',
) -> str:
    if rejected_by_robust:
        return 'robust_outlier'
    reason = str(explicit_reason)
    if reason:
        return reason
    if not used:
        return 'not_used'
    return 'ok'



__all__ = [
    'build_refraction_first_break_fit_qc_arrays',
    'build_refraction_first_break_fit_qc_payload',
    'build_refraction_reduced_time_qc_arrays',
    'build_refraction_reduced_time_qc_payload',
    'write_first_break_residuals_csv',
    'write_refraction_first_break_fit_qc_csv',
    'write_refraction_first_break_fit_qc_json',
    'write_refraction_first_break_fit_qc_npz',
    'write_refraction_first_break_time_export_csv',
    'write_refraction_reduced_time_qc_csv',
    'write_refraction_reduced_time_qc_json',
    'write_refraction_reduced_time_qc_npz',
]
