"""Array coercion helpers for refraction static artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.services.refraction_static_artifacts.contract import RefractionStaticArtifactError
from app.services.refraction_static_artifacts.formatters import _float_or_nan, _json_float
from app.services.refraction_static_status import REFRACTION_STATIC_STATUSES

def _required_layer_cell_id_array(value: object, *, name: str) -> np.ndarray:
    if value is None:
        raise RefractionStaticArtifactError(f'{name} is required')
    array = np.asarray(value, dtype=np.int64)
    if array.ndim != 1:
        raise RefractionStaticArtifactError(f'{name} must be one-dimensional')
    return np.ascontiguousarray(array, dtype=np.int64)

def _required_cell_int_array(value: object, *, name: str) -> np.ndarray:
    if value is None:
        raise RefractionStaticArtifactError(f'solve_cell result requires {name}')
    return _int_array(value)

def _required_cell_float_array(value: object, *, name: str) -> np.ndarray:
    if value is None:
        raise RefractionStaticArtifactError(f'solve_cell result requires {name}')
    return _float_array(value)

def _required_cell_status_array(value: object, *, name: str) -> np.ndarray:
    if value is None:
        raise RefractionStaticArtifactError(f'solve_cell result requires {name}')
    status = _string_array(value)
    _validate_status_array(status, name=name)
    return status

def _validate_refractor_velocity_cell_ids(
    *,
    grid_cell_id: np.ndarray,
    active_cell_id: np.ndarray,
    inactive_cell_id: np.ndarray,
) -> None:
    grid_ids = {int(value) for value in np.asarray(grid_cell_id).tolist()}
    active_ids = [int(value) for value in np.asarray(active_cell_id).tolist()]
    inactive_ids = [int(value) for value in np.asarray(inactive_cell_id).tolist()]
    combined = active_ids + inactive_ids
    if len(combined) != len(set(combined)):
        raise RefractionStaticArtifactError(
            'active and inactive refractor cell IDs must be unique'
        )
    combined_ids = set(combined)
    if combined_ids != grid_ids:
        missing = sorted(grid_ids - combined_ids)
        extra = sorted(combined_ids - grid_ids)
        raise RefractionStaticArtifactError(
            'solve_cell refractor cell IDs do not cover the configured grid: '
            f'missing={missing}, extra={extra}'
        )

def _qc_int(qc: dict[str, Any], key: str, *, default: int) -> int:
    raw = qc.get(key)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticArtifactError(
            f'QC field {key} must be an integer'
        ) from exc

def _required_positive_qc_int(qc: dict[str, Any], key: str) -> int:
    raw = qc.get(key)
    if raw is None:
        raise RefractionStaticArtifactError(f'QC field {key} is required')
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticArtifactError(
            f'QC field {key} must be an integer'
        ) from exc
    if value <= 0:
        raise RefractionStaticArtifactError(
            f'QC field {key} must be a positive integer'
        )
    return value

def _qc_cell_id_array(
    qc: dict[str, Any],
    key: str,
    *,
    n_total_cells: int,
) -> np.ndarray:
    raw = qc.get(key, [])
    arr = np.asarray(raw)
    if arr.ndim != 1:
        raise RefractionStaticArtifactError(f'QC field {key} must be one-dimensional')
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    try:
        out = np.ascontiguousarray(arr, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticArtifactError(
            f'QC field {key} must contain integer cell IDs'
        ) from exc
    if np.any(out < 0) or np.any(out >= n_total_cells):
        raise RefractionStaticArtifactError(
            f'QC field {key} contains out-of-range cell IDs'
        )
    return out

def _qc_cell_count_array(
    qc: dict[str, Any],
    key: str,
    *,
    n_total_cells: int,
) -> np.ndarray | None:
    raw = qc.get(key)
    if raw is None:
        return None
    arr = np.asarray(raw)
    if arr.shape != (n_total_cells,):
        raise RefractionStaticArtifactError(
            f'QC field {key} must have one count per refractor cell'
        )
    try:
        out = np.ascontiguousarray(arr, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticArtifactError(
            f'QC field {key} must contain integer counts'
        ) from exc
    if np.any(out < 0):
        raise RefractionStaticArtifactError(
            f'QC field {key} must not contain negative counts'
        )
    return out

def _qc_optional_float(
    qc: dict[str, Any],
    key: str,
    *,
    default: float | None,
) -> float | None:
    raw = qc.get(key, default)
    return _json_float(raw)

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



__all__ = [name for name in globals() if name.startswith('_') and not name.startswith('__')]
