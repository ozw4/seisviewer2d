"""Math helpers for converting FB-pick probabilities into pick indices."""

from __future__ import annotations

import numpy as np


def _expectation_moments(
    prob: np.ndarray,
    *,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    sums, sum_i, sum_i2 = _chunked_expectations(prob, chunk=chunk)
    if not np.all(np.isfinite(sums)):
        raise ValueError('Probability sum invalid')
    if np.any(sums <= 0):
        raise ValueError('Probability mass is zero for a trace')
    mu = sum_i / sums
    second_moment = sum_i2 / sums
    var = np.maximum(second_moment - mu * mu, 0.0)
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(var)):
        raise ValueError('Expectation calculation failed')
    return mu, var


def _chunked_expectations(
    prob: np.ndarray,
    *,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_traces, n_samples = prob.shape
    indices = np.arange(n_samples, dtype=np.float64)
    indices_sq = indices * indices
    sums = np.empty(n_traces, dtype=np.float64)
    sum_i = np.empty(n_traces, dtype=np.float64)
    sum_i2 = np.empty(n_traces, dtype=np.float64)
    for start in range(0, n_traces, chunk):
        end = min(start + chunk, n_traces)
        block = prob[start:end]
        if not np.all(np.isfinite(block)):
            raise ValueError('Probability map contains NaN or Inf')
        sums[start:end] = np.sum(block, axis=1, dtype=np.float64)
        sum_i[start:end] = np.einsum(
            'ij,j->i', block, indices, dtype=np.float64, optimize=True
        )
        sum_i2[start:end] = np.einsum(
            'ij,j->i', block, indices_sq, dtype=np.float64, optimize=True
        )
    return sums, sum_i, sum_i2


def pick_index_from_prob(
    prob: np.ndarray, *, method: str, chunk: int = 4096
) -> np.ndarray:
    """Return one pick index per trace from probability map ``prob``."""
    if prob.ndim != 2:
        raise ValueError('Probability map must be 2D')
    n_traces, n_samples = prob.shape
    if n_traces == 0 or n_samples == 0:
        return np.empty(n_traces, dtype=np.float64)

    method_norm = method.lower()
    if method_norm not in {'argmax', 'expectation'}:
        raise ValueError('Unsupported method')

    if method_norm == 'expectation':
        mu, _ = _expectation_moments(prob, chunk=chunk)
        return np.asarray(mu, dtype=np.float64)

    idx = np.empty(n_traces, dtype=np.float64)
    for start in range(0, n_traces, chunk):
        end = min(start + chunk, n_traces)
        block = prob[start:end]
        idx[start:end] = np.argmax(block, axis=1, keepdims=False)
    return idx


def sigma_ms_from_prob(prob: np.ndarray, *, dt: float, chunk: int = 4096) -> np.ndarray:
    """Return per-trace pick uncertainty (standard deviation) in milliseconds."""
    if prob.ndim != 2:
        raise ValueError('Probability map must be 2D')
    n_traces, n_samples = prob.shape
    if n_traces == 0 or n_samples == 0:
        return np.empty(n_traces, dtype=np.float64)
    if dt <= 0:
        raise ValueError('Non-positive dt value')

    _, var = _expectation_moments(prob, chunk=chunk)
    sigma = np.sqrt(var)
    return sigma * float(dt) * 1000.0


def expectation_idx_and_sigma_ms(
    prob: np.ndarray, *, dt: float, chunk: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Return expectation-based index and sigma_ms with one probability pass."""
    if prob.ndim != 2:
        raise ValueError('Probability map must be 2D')
    n_traces, n_samples = prob.shape
    if n_traces == 0 or n_samples == 0:
        empty = np.empty(n_traces, dtype=np.float64)
        return empty, empty.copy()
    if dt <= 0:
        raise ValueError('Non-positive dt value')

    mu, var = _expectation_moments(prob, chunk=chunk)
    sigma_ms = np.sqrt(var) * float(dt) * 1000.0
    return np.asarray(mu, dtype=np.float64), np.asarray(sigma_ms, dtype=np.float64)


def apply_sigma_gate(
    idx: np.ndarray, sigma_ms: np.ndarray, *, sigma_ms_max: float | None
) -> np.ndarray:
    """Replace picks whose sigma exceeds threshold with NaN."""
    idx_f64 = np.asarray(idx, dtype=np.float64)
    sigma_f64 = np.asarray(sigma_ms, dtype=np.float64)
    if idx_f64.ndim != 1 or sigma_f64.ndim != 1:
        raise ValueError('idx and sigma_ms must be 1D arrays')
    if idx_f64.shape != sigma_f64.shape:
        raise ValueError('idx and sigma_ms length mismatch')
    if sigma_ms_max is None:
        return idx_f64.copy()

    gated = idx_f64.copy()
    gated[sigma_f64 > float(sigma_ms_max)] = np.nan
    return gated


__all__ = [
    'apply_sigma_gate',
    'expectation_idx_and_sigma_ms',
    'pick_index_from_prob',
    'sigma_ms_from_prob',
]
