"""Math helpers for converting FB-pick probabilities into pick indices."""

from __future__ import annotations

import numpy as np


def normalize_prob_time(
    prob: np.ndarray,
    *,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each trace over time and return valid-trace mask."""
    if prob.ndim != 2:
        raise ValueError('Probability map must be 2D')

    prob_f32 = np.ascontiguousarray(prob, dtype=np.float32)
    n_traces, _n_samples = prob_f32.shape
    valid_mask = np.zeros(n_traces, dtype=bool)
    for start in range(0, n_traces, chunk):
        end = min(start + chunk, n_traces)
        block = prob_f32[start:end]
        if not np.all(np.isfinite(block)):
            raise ValueError('Probability map contains NaN or Inf')
        masses = np.sum(block, axis=1, dtype=np.float64)
        valid_block = masses > 0.0
        valid_mask[start:end] = valid_block
        if np.any(valid_block):
            block[valid_block] /= masses[valid_block][:, None]
        if np.any(~valid_block):
            block[~valid_block] = 0.0
    return prob_f32, valid_mask


def _expectation_moments(
    prob: np.ndarray,
    *,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    sums, sum_i, sum_i2 = _chunked_expectations(prob, chunk=chunk)
    if not np.all(np.isfinite(sums)):
        raise ValueError('Probability sum invalid')
    valid = sums > 0.0
    mu = np.full(sums.shape, np.nan, dtype=np.float64)
    var = np.full(sums.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        valid_sums = sums[valid]
        valid_mu = sum_i[valid] / valid_sums
        second_moment = sum_i2[valid] / valid_sums
        valid_var = np.maximum(second_moment - valid_mu * valid_mu, 0.0)
        mu[valid] = valid_mu
        var[valid] = valid_var
    if np.any(valid) and (
        not np.all(np.isfinite(mu[valid])) or not np.all(np.isfinite(var[valid]))
    ):
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

    prob_time, valid = normalize_prob_time(prob, chunk=chunk)
    if method_norm == 'expectation':
        mu, _ = _expectation_moments(prob_time, chunk=chunk)
        return np.asarray(mu, dtype=np.float64)

    idx = np.empty(n_traces, dtype=np.float64)
    for start in range(0, n_traces, chunk):
        end = min(start + chunk, n_traces)
        block = prob_time[start:end]
        idx[start:end] = np.argmax(block, axis=1, keepdims=False)
    idx[~valid] = np.nan
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

    prob_time, _ = normalize_prob_time(prob, chunk=chunk)
    _, var = _expectation_moments(prob_time, chunk=chunk)
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

    prob_time, _ = normalize_prob_time(prob, chunk=chunk)
    mu, var = _expectation_moments(prob_time, chunk=chunk)
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
    gated = idx_f64.copy()
    gated[~np.isfinite(sigma_f64)] = np.nan
    if sigma_ms_max is not None:
        gated[sigma_f64 > float(sigma_ms_max)] = np.nan
    return gated


__all__ = [
    'apply_sigma_gate',
    'expectation_idx_and_sigma_ms',
    'normalize_prob_time',
    'pick_index_from_prob',
    'sigma_ms_from_prob',
]
