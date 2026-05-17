"""Reduced-time first-break QC artifact writers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_layer_config import (
    normalize_refraction_static_layers,
)
from app.services.refraction_static_types import RefractionDatumStaticsResult
from app.services.refraction_static_artifacts.contract import (
    _REDUCED_TIME_QC_COLUMNS,
    ARTIFACT_VERSION,
    REDUCED_TIME_QC_FORMULA,
    REFRACTION_REDUCED_TIME_QC_CSV_NAME,
    REFRACTION_REDUCED_TIME_QC_JSON_NAME,
    REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
    SIGN_CONVENTION,
    WORKFLOW,
)
from app.services.refraction_static_artifacts.formatters import (
    _csv_bool,
    _csv_float,
    _float_or_nan,
    _json_float,
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
    _first_break_fit_inline_crossline,
    _float_array,
    _int_array,
    _midpoint_coordinate,
    _residual_row_layer_context,
    _residual_row_string_context,
    _residual_row_velocity_context,
    _row_endpoint_float_context,
    _scalar_str,
    _string_array,
)
from app.services.refraction_static_artifacts.stats import _stat, _status_counts
from app.services.refraction_static_artifacts.validation import _validate_result


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


__all__ = [
    'build_refraction_reduced_time_qc_arrays',
    'build_refraction_reduced_time_qc_payload',
    'write_refraction_reduced_time_qc_csv',
    'write_refraction_reduced_time_qc_json',
    'write_refraction_reduced_time_qc_npz',
]
