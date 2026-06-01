"""Field-correction accessors for refraction static artifacts."""

from __future__ import annotations

import numpy as np

from app.statics.refraction.artifacts.contract import (
    RefractionStaticArtifactError,
)
from app.services.refraction_static_types import RefractionDatumStaticsResult

_FIELD_DISABLED_STATUS = 'not_enabled'
_FIELD_NOT_APPLICABLE_STATUS = 'not_applicable'
_FIELD_TOTAL_VALID_STATUSES = frozenset(
    {'ok', _FIELD_DISABLED_STATUS, _FIELD_NOT_APPLICABLE_STATUS}
)


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


def _endpoint_shift_to_trace_order(
    *,
    endpoint_key: np.ndarray,
    endpoint_shift_s: np.ndarray,
    endpoint_key_sorted: np.ndarray,
    label: str,
) -> np.ndarray:
    shift_by_key: dict[str, float] = {}
    for key, shift in zip(endpoint_key.tolist(), endpoint_shift_s.tolist()):
        text = str(key)
        value = float(shift)
        existing = shift_by_key.get(text)
        if existing is not None and not (
            existing == value or (np.isnan(existing) and np.isnan(value))
        ):
            raise RefractionStaticArtifactError(
                f'{label} cannot be mapped to trace order; duplicate endpoint '
                f'key {text!r} has conflicting shifts'
            )
        shift_by_key[text] = value

    out = np.full(endpoint_key_sorted.shape, np.nan, dtype=np.float64)
    for index, key in enumerate(endpoint_key_sorted.tolist()):
        text = str(key)
        if text not in shift_by_key:
            raise RefractionStaticArtifactError(
                f'{label} cannot be mapped to trace order; endpoint key '
                f'{text!r} is missing'
            )
        out[index] = shift_by_key[text]
    return np.ascontiguousarray(out, dtype=np.float64)


def _applied_endpoint_field_shift_s(
    *,
    field_shift_s: np.ndarray,
    field_status: np.ndarray,
    apply_to_trace_shift: bool,
) -> np.ndarray:
    field = np.asarray(field_shift_s, dtype=np.float64)
    if not apply_to_trace_shift:
        return np.zeros(field.shape, dtype=np.float64)
    status = np.asarray(field_status).astype(str)
    out = np.full(field.shape, np.nan, dtype=np.float64)
    valid = np.isin(status, tuple(_FIELD_TOTAL_VALID_STATUSES)) & np.isfinite(field)
    out[valid] = field[valid]
    return np.ascontiguousarray(out, dtype=np.float64)


def _float_array(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64)


def _bool_array(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=bool)


def _string_array(value: object) -> np.ndarray:
    raw = [str(item) for item in np.asarray(value).tolist()]
    max_len = max([1, *(len(item) for item in raw)])
    return np.ascontiguousarray(raw, dtype=f'<U{max_len}')


__all__ = [
    '_FIELD_DISABLED_STATUS',
    '_FIELD_NOT_APPLICABLE_STATUS',
    '_FIELD_TOTAL_VALID_STATUSES',
    '_applied_endpoint_field_shift_s',
    '_applied_field_shift_s_sorted_array',
    '_base_refraction_trace_shift_s_sorted_array',
    '_disabled_field_float_array',
    '_disabled_field_status_array',
    '_endpoint_shift_to_trace_order',
    '_field_static_valid_mask',
    '_final_trace_shift_s_sorted',
    '_final_trace_static_status_sorted_array',
    '_final_trace_static_valid_mask_sorted_array',
    '_has_field_correction_composition',
    '_has_manual_static_field_correction',
    '_has_source_depth_field_correction',
    '_has_uphole_field_correction',
    '_optional_field_float_array',
    '_optional_field_status_array',
    '_receiver_field_shift_s_array',
    '_receiver_field_shift_s_sorted_array',
    '_receiver_field_static_status_array',
    '_receiver_manual_static_shift_s_array',
    '_receiver_manual_static_status_array',
    '_source_depth_m_array',
    '_source_depth_shift_s_array',
    '_source_depth_status_array',
    '_source_field_shift_s_array',
    '_source_field_shift_s_sorted_array',
    '_source_field_static_status_array',
    '_source_manual_static_shift_s_array',
    '_source_manual_static_status_array',
    '_source_uphole_shift_s_array',
    '_source_uphole_status_array',
    '_source_uphole_time_s_array',
    '_total_with_field_shift_s',
    '_trace_endpoint_key_sorted_array',
    '_trace_field_shift_s_sorted_array',
    '_trace_field_static_status_sorted_array',
]
