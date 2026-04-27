"""Linear moveout calculation helpers."""

from __future__ import annotations

import numpy as np


def compute_lmo_shift_seconds(
    offsets: np.ndarray,
    *,
    velocity_mps: float,
    offset_scale: float,
    offset_mode: str,
    ref_mode: str,
    ref_trace: int | None,
    polarity: int,
) -> np.ndarray:
    """Return per-trace linear moveout shifts in seconds."""
    offsets_f64 = _validate_offsets(offsets)
    velocity = _validate_velocity_mps(velocity_mps)
    scale = _validate_offset_scale(offset_scale)
    _validate_offset_mode(offset_mode)
    _validate_ref_mode(ref_mode)
    polarity_int = _validate_polarity(polarity)

    scaled_offsets = offsets_f64 * scale
    if not np.all(np.isfinite(scaled_offsets)):
        raise ValueError('Scaled offsets contain NaN or Inf')

    if offset_mode == 'absolute':
        x = np.abs(scaled_offsets)
    else:
        x = scaled_offsets

    x_ref = _reference_offset(x, ref_mode=ref_mode, ref_trace=ref_trace)
    shifts = polarity_int * (x - x_ref) / velocity
    if not np.all(np.isfinite(shifts)):
        raise ValueError('LMO shifts contain NaN or Inf')
    return np.asarray(shifts, dtype=np.float64)


def _validate_offsets(offsets: np.ndarray) -> np.ndarray:
    arr = np.asarray(offsets)
    if arr.ndim != 1:
        raise ValueError('offsets must be a 1D array')
    if arr.size == 0:
        raise ValueError('offsets must not be empty')
    try:
        arr_f64 = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('offsets must be numeric') from exc
    if not np.all(np.isfinite(arr_f64)):
        raise ValueError('offsets contain NaN or Inf')
    return arr_f64


def _validate_velocity_mps(velocity_mps: float) -> float:
    velocity = float(velocity_mps)
    if not np.isfinite(velocity) or velocity <= 0.0:
        raise ValueError('velocity_mps must be finite and greater than 0')
    return velocity


def _validate_offset_scale(offset_scale: float) -> float:
    scale = float(offset_scale)
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError('offset_scale must be finite and non-zero')
    return scale


def _validate_offset_mode(offset_mode: str) -> None:
    if offset_mode not in {'absolute', 'signed'}:
        raise ValueError('offset_mode must be "absolute" or "signed"')


def _validate_ref_mode(ref_mode: str) -> None:
    if ref_mode not in {'min', 'first', 'center', 'trace', 'zero'}:
        raise ValueError('ref_mode must be one of "min", "first", "center", "trace", "zero"')


def _validate_polarity(polarity: int) -> int:
    if (
        isinstance(polarity, bool)
        or not isinstance(polarity, (int, np.integer))
        or int(polarity) not in {1, -1}
    ):
        raise ValueError('polarity must be 1 or -1')
    return int(polarity)


def _reference_offset(
    x: np.ndarray,
    *,
    ref_mode: str,
    ref_trace: int | None,
) -> float:
    if ref_mode == 'min':
        return float(np.min(x))
    if ref_mode == 'first':
        return float(x[0])
    if ref_mode == 'center':
        return float(x[x.shape[0] // 2])
    if ref_mode == 'zero':
        return 0.0
    if ref_trace is None:
        raise ValueError('ref_trace is required when ref_mode is "trace"')
    if (
        isinstance(ref_trace, bool)
        or not isinstance(ref_trace, (int, np.integer))
        or int(ref_trace) < 0
        or int(ref_trace) >= x.shape[0]
    ):
        raise ValueError('ref_trace is out of range')
    return float(x[int(ref_trace)])


__all__ = ['compute_lmo_shift_seconds']
