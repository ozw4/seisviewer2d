"""Baseline scaling helpers."""

from __future__ import annotations

import contextlib
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.services.errors import BadRequestError, InternalError, UnprocessableError
from app.utils.segy_meta import NORM_EPS, load_baseline

logger = logging.getLogger(__name__)


def apply_scaling_from_baseline(
    arr: NDArray[np.float32],
    scaling: str | None,
    file_id: str,
    key1: int,
    store_dir: str | Path,
    *,
    key1_byte: int,
    key2_byte: int,
    trace_stats_cache: Any,
    trace_stats_lock: threading.RLock | None = None,
    x0: int,
    x1: int,
    step_x: int,
) -> NDArray[np.float32]:
    """Normalize ``arr`` in-place using baseline statistics."""
    mode = 'amax' if scaling is None else str(scaling)
    mode = mode.lower()
    if mode not in {'amax', 'tracewise'}:
        raise BadRequestError('Unsupported scaling mode')
    if not np.all(np.isfinite(arr)):
        raise UnprocessableError('Section contains non-finite values')
    try:
        baseline = load_baseline(
            store_dir,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
        )
    except FileNotFoundError as exc:
        raise InternalError('Baseline statistics not found') from exc
    except ValueError as exc:
        raise UnprocessableError(str(exc)) from exc
    except Exception as exc:
        raise InternalError('Failed to load baseline statistics') from exc
    key1_int = int(key1)
    index = baseline['key1_index'].get(key1_int)
    if index is None:
        raise UnprocessableError('Baseline statistics missing for key1 value')
    if mode == 'amax':
        mean = baseline['section_mean'][index]
        inv_std = baseline['section_inv_std'][index]
        if not (np.isfinite(mean) and np.isfinite(inv_std)):
            raise UnprocessableError('Baseline statistics contain non-finite values')
        arr -= mean
        arr *= inv_std
        if bool(baseline['section_clamp_mask'][index]):
            logger.info(
                'Section std clamped to eps for file_id=%s key1=%s', file_id, key1_int
            )
        return arr
    spans = baseline['trace_spans'].get(key1_int)
    if spans is None:
        raise UnprocessableError('Baseline trace spans missing for key1 value')
    lock_ctx = (
        trace_stats_lock if trace_stats_lock is not None else contextlib.nullcontext()
    )
    store_key = baseline['store_key']
    idx_full_key = (store_key, key1_int, 'idx_full')
    with lock_ctx:
        idx_full_payload = trace_stats_cache.get(idx_full_key)
    if idx_full_payload is None:
        idx_full_candidate = np.concatenate(
            [np.arange(int(s), int(e), dtype=np.int64) for (s, e) in spans]
        )
        with lock_ctx:
            idx_full_payload = trace_stats_cache.setdefault(
                idx_full_key, (idx_full_candidate, None, 0)
            )
        idx_full = idx_full_payload[0]
    else:
        idx_full = idx_full_payload[0]  # unpack tuple

    # Window/stride selection within the section
    n_traces = int(arr.shape[0])
    if x0 < 0 or x1 < x0 or step_x < 1:
        raise BadRequestError('Invalid window parameters')
    sel = np.arange(0, idx_full.shape[0], dtype=np.int64)[x0 : x1 + 1 : step_x]
    if sel.shape[0] != n_traces:
        raise UnprocessableError('Trace count mismatch for window')
    sel_global = idx_full[sel]

    # Gather per-trace stats for the selected window (cache by x0,x1,step_x)
    trace_key = (store_key, key1_int, x0, x1, step_x)
    with lock_ctx:
        cached = trace_stats_cache.get(trace_key)
    if cached is None:
        trace_mean = baseline['trace_mean']
        trace_inv = baseline['trace_inv_std']
        mean_vec = trace_mean[sel_global].astype(np.float32, copy=False)
        inv_vec = trace_inv[sel_global].astype(np.float32, copy=False)
        with lock_ctx:
            cached = trace_stats_cache.setdefault(trace_key, (mean_vec, inv_vec, 0))
    mean_vec, inv_vec, _ = cached
    if inv_vec is None:
        raise InternalError('Trace statistics cache entry is invalid')
    if not np.all(np.isfinite(mean_vec)) or not np.all(np.isfinite(inv_vec)):
        raise UnprocessableError('Baseline trace statistics contain non-finite values')
    arr -= mean_vec[:, None]
    arr *= inv_vec[:, None]
    return arr


def apply_scaling_from_reference_section(
    arr: NDArray[np.float32],
    reference: NDArray[np.float32],
    scaling: str | None,
    *,
    x0: int,
    x1: int,
    step_x: int,
) -> NDArray[np.float32]:
    """Normalize ``arr`` in-place using statistics from ``reference``."""
    mode = 'amax' if scaling is None else str(scaling)
    mode = mode.lower()
    if mode not in {'amax', 'tracewise'}:
        raise BadRequestError('Unsupported scaling mode')
    if arr.ndim != 2 or reference.ndim != 2:
        raise UnprocessableError('Reference source must be 2D')
    if not np.all(np.isfinite(arr)) or not np.all(np.isfinite(reference)):
        raise UnprocessableError('Section contains non-finite values')
    if x0 < 0 or x1 < x0 or step_x < 1:
        raise BadRequestError('Invalid window parameters')
    if x1 >= reference.shape[0]:
        raise UnprocessableError('Reference source trace range out of bounds')

    if mode == 'amax':
        mean = np.float32(np.mean(reference, dtype=np.float64))
        std = np.float32(np.std(reference, dtype=np.float64))
        if not (np.isfinite(mean) and np.isfinite(std)):
            raise UnprocessableError('Reference source statistics contain non-finite values')
        inv_std = np.float32(1.0) / np.maximum(std, NORM_EPS)
        arr -= mean
        arr *= inv_std
        return arr

    n_traces = int(arr.shape[0])
    sel = np.arange(0, reference.shape[0], dtype=np.int64)[x0 : x1 + 1 : step_x]
    if sel.shape[0] != n_traces:
        raise UnprocessableError('Trace count mismatch for reference source')
    reference_window = reference[sel, :]
    mean_vec = np.mean(reference_window, axis=1, dtype=np.float64).astype(
        np.float32,
        copy=False,
    )
    std_vec = np.std(reference_window, axis=1, dtype=np.float64).astype(
        np.float32,
        copy=False,
    )
    if not np.all(np.isfinite(mean_vec)) or not np.all(np.isfinite(std_vec)):
        raise UnprocessableError('Reference source trace statistics contain non-finite values')
    np.maximum(std_vec, NORM_EPS, out=std_vec)
    inv_vec = np.empty_like(std_vec, dtype=np.float32)
    np.reciprocal(std_vec, out=inv_vec)
    arr -= mean_vec[:, None]
    arr *= inv_vec[:, None]
    return arr


__all__ = ['apply_scaling_from_baseline', 'apply_scaling_from_reference_section']
