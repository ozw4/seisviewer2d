"""Statistics helpers shared by refraction static artifact writers."""

from __future__ import annotations

import numpy as np

from app.services.refraction_static_artifacts.contract import (
    RefractionStaticArtifactError,
)


def _stat(values: object, stat: str) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if stat == 'min':
        return float(np.min(arr))
    if stat == 'max':
        return float(np.max(arr))
    if stat == 'median':
        return float(np.median(arr))
    if stat == 'p95':
        return float(np.percentile(arr, 95.0))
    raise RefractionStaticArtifactError(f'unsupported statistic: {stat}')


def _residual_stat(values_ms: np.ndarray, stat: str) -> float | None:
    arr = np.asarray(values_ms, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if stat == 'rms':
        return float(np.sqrt(np.mean(arr * arr)))
    if stat == 'mad':
        median = float(np.median(arr))
        return float(np.median(np.abs(arr - median)))
    if stat == 'mean':
        return float(np.mean(arr))
    if stat == 'median':
        return float(np.median(arr))
    if stat == 'p95_abs':
        return float(np.percentile(np.abs(arr), 95.0))
    if stat == 'max_abs':
        return float(np.max(np.abs(arr)))
    raise RefractionStaticArtifactError(f'unsupported residual statistic: {stat}')


def _fraction(numerator: int | np.integer, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _status_counts(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in np.asarray(values).tolist():
        key = str(item)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    '_fraction',
    '_residual_stat',
    '_stat',
    '_status_counts',
]
