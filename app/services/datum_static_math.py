"""Datum static correction math helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatumStaticResult:
    source_shift_s_sorted: np.ndarray
    receiver_shift_s_sorted: np.ndarray
    trace_shift_s_sorted: np.ndarray
    source_surface_elevation_m_sorted: np.ndarray
    source_depth_m_sorted: np.ndarray
    source_depth_used_sorted: np.ndarray
    source_elevation_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray


def compute_datum_static_shifts(
    *,
    source_surface_elevation_m_sorted: np.ndarray,
    receiver_elevation_m_sorted: np.ndarray,
    datum_elevation_m: float,
    replacement_velocity_m_s: float,
    source_depth_m_sorted: np.ndarray | None = None,
) -> DatumStaticResult:
    """Compute sorted per-trace datum static shifts in seconds."""
    source_surface = _coerce_1d_finite_float64(
        source_surface_elevation_m_sorted,
        name='source_surface_elevation_m_sorted',
    )
    receiver = _coerce_1d_finite_float64(
        receiver_elevation_m_sorted,
        name='receiver_elevation_m_sorted',
    )
    if source_surface.shape != receiver.shape:
        raise ValueError(
            'source_surface_elevation_m_sorted and receiver_elevation_m_sorted must have the same shape'
        )

    if source_depth_m_sorted is None:
        source_depth = np.zeros(source_surface.shape, dtype=np.float64)
        source_depth_used = np.zeros(source_surface.shape, dtype=bool)
    else:
        source_depth = _coerce_1d_finite_float64(
            source_depth_m_sorted,
            name='source_depth_m_sorted',
        )
        if source_depth.shape != source_surface.shape:
            raise ValueError('source_depth_m_sorted must have the same shape as elevations')
        source_depth_used = np.ones(source_surface.shape, dtype=bool)

    datum = _coerce_finite_float(datum_elevation_m, name='datum_elevation_m')
    velocity = _coerce_positive_finite_float(
        replacement_velocity_m_s,
        name='replacement_velocity_m_s',
    )

    source_elevation = source_surface - source_depth
    if not np.all(np.isfinite(source_elevation)):
        raise ValueError('source_elevation_m_sorted contains NaN or Inf')

    source_shift = (datum - source_elevation) / velocity
    receiver_shift = (datum - receiver) / velocity
    trace_shift = source_shift + receiver_shift
    if (
        not np.all(np.isfinite(source_shift))
        or not np.all(np.isfinite(receiver_shift))
        or not np.all(np.isfinite(trace_shift))
    ):
        raise ValueError('datum static shifts contain NaN or Inf')

    return DatumStaticResult(
        source_shift_s_sorted=np.asarray(source_shift, dtype=np.float64),
        receiver_shift_s_sorted=np.asarray(receiver_shift, dtype=np.float64),
        trace_shift_s_sorted=np.asarray(trace_shift, dtype=np.float64),
        source_surface_elevation_m_sorted=np.asarray(source_surface, dtype=np.float64),
        source_depth_m_sorted=np.asarray(source_depth, dtype=np.float64),
        source_depth_used_sorted=np.asarray(source_depth_used, dtype=bool),
        source_elevation_m_sorted=np.asarray(source_elevation, dtype=np.float64),
        receiver_elevation_m_sorted=np.asarray(receiver, dtype=np.float64),
    )


def _coerce_1d_finite_float64(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    try:
        arr_f64 = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not np.all(np.isfinite(arr_f64)):
        raise ValueError(f'{name} must contain only finite values')
    return arr_f64


def _coerce_finite_float(value: float, *, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not np.isfinite(scalar):
        raise ValueError(f'{name} must be finite')
    return scalar


def _coerce_positive_finite_float(value: float, *, name: str) -> float:
    scalar = _coerce_finite_float(value, name=name)
    if scalar <= 0.0:
        raise ValueError(f'{name} must be greater than 0')
    return scalar


__all__ = ['DatumStaticResult', 'compute_datum_static_shifts']
