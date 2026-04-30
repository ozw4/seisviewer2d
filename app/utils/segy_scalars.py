"""SEG-Y header scalar utilities."""

from __future__ import annotations

import numpy as np


def apply_segy_scalar(values: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    """Apply SEG-Y scalar rules to raw header values."""
    arr = _coerce_numeric_float64(values, name='values')
    scalar_arr = _validate_integer_scalars(scalars)
    if arr.shape != scalar_arr.shape:
        raise ValueError('values and scalars must have the same shape')

    scalar_f64 = scalar_arr.astype(np.float64, copy=False)
    scale = np.ones(scalar_arr.shape, dtype=np.float64)
    positive = scalar_f64 > 0.0
    negative = scalar_f64 < 0.0
    scale[positive] = scalar_f64[positive]
    scale[negative] = 1.0 / np.abs(scalar_f64[negative])

    result = arr * scale
    if not np.all(np.isfinite(result)):
        raise ValueError('scaled values contain NaN or Inf')
    return np.asarray(result, dtype=np.float64)


def count_zero_segy_scalars(scalars: np.ndarray) -> int:
    """Return the number of zero SEG-Y scalar header values."""
    scalar_arr = _validate_integer_scalars(scalars)
    return int(np.count_nonzero(scalar_arr == 0))


def normalize_elevation_unit(values: np.ndarray, unit: str) -> np.ndarray:
    """Normalize elevation values to meters."""
    arr = _coerce_numeric_float64(values, name='values')
    if unit == 'm':
        result = arr
    elif unit == 'ft':
        result = arr * 0.3048
    else:
        raise ValueError('unit must be "m" or "ft"')

    if not np.all(np.isfinite(result)):
        raise ValueError('normalized elevations contain NaN or Inf')
    return np.asarray(result, dtype=np.float64)


def _coerce_numeric_float64(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    try:
        arr_f64 = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    return arr_f64


def _validate_integer_scalars(scalars: np.ndarray) -> np.ndarray:
    scalar_arr = np.asarray(scalars)
    if not np.issubdtype(scalar_arr.dtype, np.integer):
        raise ValueError('scalars must have an integer dtype')
    return scalar_arr


__all__ = [
    'apply_segy_scalar',
    'count_zero_segy_scalars',
    'normalize_elevation_unit',
]
