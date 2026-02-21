"""Pick snapping utilities ported from viewer legacy implementation."""

from __future__ import annotations

import math

import numpy as np


def _js_round(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _as_trace_1d(trace: np.ndarray) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError('trace must be 1D')
    return arr


def parabolic_refine(arr: np.ndarray, i: int) -> float:
    """3-point parabolic interpolation around ``i`` for peak/trough modes."""
    trace = _as_trace_1d(arr)
    n = int(trace.shape[0])
    if n == 0:
        return float(i)
    if n < 3:
        return float(_clamp(float(i), 0.0, float(n - 1)))

    ii = max(1, min(n - 2, int(i)))
    y1 = float(trace[ii - 1])
    y2 = float(trace[ii])
    y3 = float(trace[ii + 1])
    denom = y1 - (2.0 * y2) + y3
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return float(ii)

    delta = 0.5 * (y1 - y3) / denom
    if not np.isfinite(delta) or abs(delta) > 0.6:
        return float(ii)

    xhat = float(ii) + float(delta)
    if not np.isfinite(xhat):
        return float(ii)
    return float(_clamp(xhat, 0.0, float(n - 1)))


def zero_cross_refine(arr: np.ndarray, i: int) -> float:
    """Linear interpolation of an upward zero crossing near ``i``."""
    trace = _as_trace_1d(arr)
    n = int(trace.shape[0])
    if n < 2:
        return float(i)

    i0 = max(0, min(n - 2, int(i)))
    i1 = i0 + 1

    if not (trace[i0] <= 0.0 and trace[i1] > 0.0):
        if i0 > 0 and (trace[i0 - 1] <= 0.0 and trace[i0] > 0.0):
            i0 = i0 - 1
            i1 = i0 + 1
        elif i1 < (n - 1) and (trace[i1] <= 0.0 and trace[i1 + 1] > 0.0):
            i0 = i1
            i1 = i0 + 1
        else:
            return float(i)

    dy = float(trace[i1] - trace[i0])
    if not np.isfinite(dy) or abs(dy) < 1e-12:
        return float(i)

    frac = (0.0 - float(trace[i0])) / dy
    xhat = float(i0) + float(frac)
    if not np.isfinite(xhat):
        return float(i)
    return float(_clamp(xhat, 0.0, float(n - 1)))


def snap_pick_index(
    trace: np.ndarray,
    idx0: float,
    *,
    mode: str,
    refine: str,
    window_samples: int,
) -> float:
    """Snap one pick index to a local feature near ``idx0``."""
    arr = _as_trace_1d(trace)
    mode_norm = str(mode).lower()
    refine_norm = str(refine).lower()

    if mode_norm not in {'none', 'peak', 'trough', 'rise'}:
        raise ValueError('Unsupported mode')
    if refine_norm not in {'none', 'parabolic', 'zc'}:
        raise ValueError('Unsupported refine mode')
    if mode_norm == 'none':
        return float(idx0)

    n = int(arr.shape[0])
    if n < 3:
        return float(idx0)

    i0 = _js_round(float(idx0))
    rad = max(1, int(window_samples))
    lo = max(1, i0 - rad)
    hi = min(n - 2, i0 + rad)
    if lo > hi:
        return float(idx0)

    idx = i0

    if mode_norm == 'peak':
        best: int | None = None
        best_dist = math.inf
        for i in range(lo, hi + 1):
            if arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
                dist = abs(i - i0)
                if dist < best_dist:
                    best_dist = dist
                    best = i
        if best is not None:
            idx = best
        else:
            vmax = -math.inf
            for i in range(lo, hi + 1):
                value = float(arr[i])
                if value > vmax:
                    vmax = value
                    idx = i
    elif mode_norm == 'trough':
        best = None
        best_dist = math.inf
        for i in range(lo, hi + 1):
            if arr[i] <= arr[i - 1] and arr[i] <= arr[i + 1]:
                dist = abs(i - i0)
                if dist < best_dist:
                    best_dist = dist
                    best = i
        if best is not None:
            idx = best
        else:
            vmin = math.inf
            for i in range(lo, hi + 1):
                value = float(arr[i])
                if value < vmin:
                    vmin = value
                    idx = i
    else:
        best = None
        best_dist = math.inf
        for i in range(lo, hi):
            if arr[i] <= 0.0 and arr[i + 1] > 0.0:
                cand = i if abs(arr[i]) < abs(arr[i + 1]) else (i + 1)
                dist = abs(cand - i0)
                if dist < best_dist:
                    best_dist = dist
                    best = cand
        if best is not None:
            idx = best
        else:
            smax = -math.inf
            for i in range(lo, hi + 1):
                slope = float(arr[i + 1] - arr[i - 1])
                if slope > 0.0 and slope > smax:
                    smax = slope
                    idx = i

    idx_float = float(idx)
    if mode_norm in {'peak', 'trough'} and refine_norm == 'parabolic':
        idx_float = parabolic_refine(arr, idx)
    elif mode_norm == 'rise' and refine_norm == 'zc':
        idx_float = zero_cross_refine(arr, idx)

    return float(_clamp(float(idx_float), 0.0, float(n - 1)))


def snap_pick_time_s(
    trace: np.ndarray,
    time_s: float,
    *,
    dt: float,
    mode: str,
    refine: str,
    window_ms: float,
) -> float:
    """Snap one pick time (seconds) to the requested local feature."""
    if dt <= 0:
        raise ValueError('dt must be > 0')

    idx0 = float(_js_round(float(time_s) / float(dt)))
    window_samples = _js_round((float(window_ms) / 1000.0) / float(dt))
    idx = snap_pick_index(
        trace,
        idx0,
        mode=mode,
        refine=refine,
        window_samples=window_samples,
    )
    return float(idx) * float(dt)


__all__ = [
    'parabolic_refine',
    'snap_pick_index',
    'snap_pick_time_s',
    'zero_cross_refine',
]
