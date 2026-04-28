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


def compute_lmo_raw_sample_bounds(
    *,
    y0: int,
    y1: int,
    shift_samples: np.ndarray,
    n_samples: int,
) -> tuple[int, int]:
    """Return the clamped raw sample range needed for an LMO display window."""
    shifts = np.asarray(shift_samples, dtype=np.float64)
    if shifts.ndim != 1 or shifts.size == 0:
        raise ValueError('shift_samples must be a non-empty 1D array')
    if not np.all(np.isfinite(shifts)):
        raise ValueError('shift_samples contain NaN or Inf')
    sample_count = int(n_samples)
    if sample_count <= 0:
        raise ValueError('n_samples must be greater than 0')

    raw_y0 = int(np.floor(int(y0) + float(np.min(shifts)))) - 1
    raw_y1 = int(np.ceil(int(y1) + float(np.max(shifts)))) + 1
    raw_y0 = min(sample_count - 1, max(0, raw_y0))
    raw_y1 = min(sample_count - 1, max(0, raw_y1))
    return raw_y0, raw_y1


def resample_lmo_window(
    expanded: np.ndarray,
    *,
    y0: int,
    y1: int,
    step_y: int,
    raw_y0: int,
    shift_samples: np.ndarray,
) -> np.ndarray:
    """Interpolate an expanded raw window onto the LMO display sample grid."""
    arr = np.asarray(expanded, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError('expanded window must be 2D')
    if int(step_y) < 1:
        raise ValueError('step_y must be >= 1')

    shifts = np.asarray(shift_samples, dtype=np.float64)
    if shifts.ndim != 1:
        raise ValueError('shift_samples must be a 1D array')
    if shifts.shape[0] != arr.shape[0]:
        raise ValueError('shift_samples length must match trace count')
    if not np.all(np.isfinite(shifts)):
        raise ValueError('shift_samples contain NaN or Inf')

    display_samples = np.arange(int(y0), int(y1) + 1, int(step_y), dtype=np.float64)
    if display_samples.size == 0:
        raise ValueError('Requested window is empty')

    out = np.zeros((arr.shape[0], display_samples.size), dtype=np.float32)
    raw_width = int(arr.shape[1])
    if raw_width == 0:
        return out

    max_source = float(raw_width - 1)
    for trace_idx, shift in enumerate(shifts):
        source = display_samples + float(shift) - float(raw_y0)
        valid = (source >= 0.0) & (source <= max_source)
        if not np.any(valid):
            continue
        source_valid = source[valid]
        lo = np.floor(source_valid).astype(np.int64)
        hi = np.minimum(lo + 1, raw_width - 1)
        frac = (source_valid - lo).astype(np.float32, copy=False)
        row = arr[trace_idx]
        out[trace_idx, valid] = row[lo] * (1.0 - frac) + row[hi] * frac
    return out


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


__all__ = [
    'compute_lmo_raw_sample_bounds',
    'compute_lmo_shift_seconds',
    'resample_lmo_window',
]
