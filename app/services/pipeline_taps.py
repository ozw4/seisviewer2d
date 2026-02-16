"""Pipeline tap cache helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.state import AppState
from app.services.reader import EXPECTED_SECTION_NDIM


class PipelineTapNotFoundError(LookupError):
    """Raised when a requested pipeline tap output is unavailable."""


def _pipeline_payload_to_array(payload: object, *, tap_label: str) -> np.ndarray:
    """Convert a cached pipeline payload into a 2D ``float32`` array."""
    data_obj = payload
    if isinstance(payload, dict):
        for key in ('data', 'prob', 'values'):
            if key in payload:
                data_obj = payload[key]
                break
        else:
            msg = f'Pipeline tap {tap_label!r} payload missing data field'
            raise ValueError(msg)

    arr = np.asarray(data_obj, dtype=np.float32)
    if arr.ndim != EXPECTED_SECTION_NDIM:
        msg = f'Pipeline tap {tap_label!r} expected 2D data, got {arr.ndim}D'
        raise ValueError(msg)
    return np.ascontiguousarray(arr)


def build_pipeline_tap_cache_base_key(
    *,
    file_id: str,
    key1: int,
    key1_byte: int,
    key2_byte: int,
    pipeline_key: str,
    window_hash: str | None,
    offset_byte: int | None,
) -> tuple[str, int, int, int, str, str | None, int | None]:
    """Build the canonical base key for ``pipeline_tap_cache``."""
    return (
        file_id,
        int(key1),
        int(key1_byte),
        int(key2_byte),
        pipeline_key,
        window_hash,
        offset_byte,
    )


def build_pipeline_tap_cache_key(
    *,
    file_id: str,
    key1: int,
    key1_byte: int,
    key2_byte: int,
    pipeline_key: str,
    window_hash: str | None,
    offset_byte: int | None,
    tap_label: str,
) -> tuple[str, int, int, int, str, str | None, int | None, str]:
    """Build the canonical full key for ``pipeline_tap_cache``."""
    base_key = build_pipeline_tap_cache_base_key(
        file_id=file_id,
        key1=key1,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        pipeline_key=pipeline_key,
        window_hash=window_hash,
        offset_byte=offset_byte,
    )
    return (*base_key, tap_label)


def get_section_from_pipeline_tap(
    *,
    file_id: str,
    key1: int,
    key1_byte: int,
    key2_byte: int,
    pipeline_key: str,
    tap_label: str,
    state: AppState,
    offset_byte: int | None = None,
) -> np.ndarray:
    """Return the cached pipeline tap output as a ``float32`` array."""
    arr, _ = get_section_and_meta_from_pipeline_tap(
        file_id=file_id,
        key1=key1,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        pipeline_key=pipeline_key,
        tap_label=tap_label,
        offset_byte=offset_byte,
        state=state,
    )
    return arr


def get_section_and_meta_from_pipeline_tap(
    *,
    file_id: str,
    key1: int,
    key1_byte: int,
    key2_byte: int,
    pipeline_key: str,
    tap_label: str,
    state: AppState,
    offset_byte: int | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Return tap payload as ``float32`` along with optional metadata."""
    cache_key = build_pipeline_tap_cache_key(
        file_id=file_id,
        key1=key1,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        pipeline_key=pipeline_key,
        window_hash=None,
        offset_byte=offset_byte,
        tap_label=tap_label,
    )
    with state.lock:
        payload = state.pipeline_tap_cache.get(cache_key)
    if payload is None:
        msg = (
            f'Pipeline tap {tap_label!r} for pipeline {pipeline_key!r} '
            f'and key1={key1} is not available. '
            'Please re-run the pipeline.'
        )
        raise PipelineTapNotFoundError(msg)
    arr = _pipeline_payload_to_array(payload, tap_label=tap_label)
    meta = None
    if isinstance(payload, dict):
        meta_obj = payload.get('meta')
        if isinstance(meta_obj, dict):
            meta = meta_obj
    return arr, meta


__all__ = [
    'PipelineTapNotFoundError',
    '_pipeline_payload_to_array',
    'build_pipeline_tap_cache_base_key',
    'build_pipeline_tap_cache_key',
    'get_section_and_meta_from_pipeline_tap',
    'get_section_from_pipeline_tap',
]
