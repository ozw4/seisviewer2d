"""Helpers for SEG-Y metadata such as sampling interval."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.utils.baseline_artifacts import (
    BASELINE_STAGE_RAW,
    SplitBaselineArtifactsError,
    build_baseline_manifest_path,
    build_legacy_baseline_path,
    read_split_baseline_payload,
)

HEADER_SAMPLE_INTERVAL_OFFSET = 3200 + 16
SAMPLE_INTERVAL_BYTES = 2
MICROSECONDS_PER_SECOND = 1_000_000.0

logger = logging.getLogger(__name__)

NORM_EPS = np.float32(float(os.getenv('NORM_EPS', '1e-6')))

_BASELINE_CACHE: dict[str, dict[str, Any]] = {}


def read_segy_dt_seconds(path: str) -> float | None:
    """Return the SEG-Y sampling interval in seconds, if available."""
    sample_path = Path(path)
    try:
        with sample_path.open('rb') as f:
            f.seek(HEADER_SAMPLE_INTERVAL_OFFSET)
            raw = f.read(SAMPLE_INTERVAL_BYTES)
        if len(raw) != SAMPLE_INTERVAL_BYTES:
            return None
        us = int.from_bytes(raw, byteorder='big', signed=False)
        if us <= 0:
            return None
        return us / MICROSECONDS_PER_SECOND
    except Exception:  # noqa: BLE001
        return None


def _load_json_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('Baseline payload must be a JSON object')
    return payload


def _payload_matches_key_bytes(
    payload: dict[str, Any], *, key1_byte: int, key2_byte: int
) -> bool:
    if payload.get('stage') != BASELINE_STAGE_RAW:
        return False
    try:
        stored_key1 = int(payload['key1_byte'])
        stored_key2 = int(payload['key2_byte'])
    except (KeyError, TypeError, ValueError):
        return False
    return stored_key1 == int(key1_byte) and stored_key2 == int(key2_byte)


def _load_legacy_baseline_payload(
    store_path: Path,
    *,
    key1_byte: int,
    key2_byte: int,
) -> tuple[dict[str, Any], str] | None:
    legacy_path = build_legacy_baseline_path(store_path)
    if not legacy_path.is_file():
        return None
    payload = _load_json_payload(legacy_path)
    if not _payload_matches_key_bytes(
        payload,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    ):
        return None
    cache_key = f'legacy-json|{legacy_path.resolve()}|{legacy_path.stat().st_mtime_ns}'
    return payload, cache_key


def _build_baseline_entry(
    *,
    payload: dict[str, Any],
    store_path: Path,
    key1_byte: int,
    key2_byte: int,
) -> dict[str, Any]:
    key1_values = np.ascontiguousarray(
        np.asarray(payload.get('key1_values'), dtype=np.int64)
    )
    mu_section = np.ascontiguousarray(
        np.asarray(payload.get('mu_section_by_key1'), dtype=np.float32)
    )
    sigma_section = np.ascontiguousarray(
        np.asarray(payload.get('sigma_section_by_key1'), dtype=np.float32)
    )
    if (
        key1_values.shape[0] != mu_section.shape[0]
        or mu_section.shape != sigma_section.shape
    ):
        raise ValueError('Baseline section statistics are inconsistent')
    if not np.all(np.isfinite(mu_section)):
        raise ValueError('Baseline section mean contains non-finite values')
    if not np.all(np.isfinite(sigma_section)):
        raise ValueError('Baseline section std contains non-finite values')
    section_clamp_mask = np.ascontiguousarray(sigma_section <= NORM_EPS, dtype=bool)
    if section_clamp_mask.any():
        logger.info(
            'Clamped %d section std values to eps (%s)',
            int(section_clamp_mask.sum()),
            store_path,
        )
    safe_sigma_section = sigma_section.copy()
    np.maximum(safe_sigma_section, NORM_EPS, out=safe_sigma_section)
    inv_sigma_section = np.empty_like(safe_sigma_section, dtype=np.float32)
    np.reciprocal(safe_sigma_section, out=inv_sigma_section)
    mu_traces = np.ascontiguousarray(
        np.asarray(payload.get('mu_traces'), dtype=np.float32)
    )
    sigma_traces = np.ascontiguousarray(
        np.asarray(payload.get('sigma_traces'), dtype=np.float32)
    )
    if mu_traces.shape != sigma_traces.shape:
        raise ValueError('Baseline trace statistics are inconsistent')
    if not np.all(np.isfinite(mu_traces)):
        raise ValueError('Baseline trace mean contains non-finite values')
    if not np.all(np.isfinite(sigma_traces)):
        raise ValueError('Baseline trace std contains non-finite values')
    trace_clamp_mask = np.ascontiguousarray(sigma_traces <= NORM_EPS, dtype=bool)
    if trace_clamp_mask.any():
        logger.info(
            'Clamped %d trace std values to eps (%s)',
            int(trace_clamp_mask.sum()),
            store_path,
        )
    safe_sigma_traces = sigma_traces.copy()
    np.maximum(safe_sigma_traces, NORM_EPS, out=safe_sigma_traces)
    inv_sigma_traces = np.empty_like(safe_sigma_traces, dtype=np.float32)
    np.reciprocal(safe_sigma_traces, out=inv_sigma_traces)
    raw_spans = payload.get('trace_spans_by_key1') or {}
    trace_spans: dict[int, list[tuple[int, int]]] = {}
    for key_str, ranges in raw_spans.items():
        key_int = int(key_str)
        span_list: list[tuple[int, int]] = []
        for span in ranges:
            if not isinstance(span, list) or len(span) != 2:
                raise ValueError('Baseline trace span entry malformed')
            start, end = int(span[0]), int(span[1])
            if start < 0 or end < start or end > mu_traces.shape[0]:
                raise ValueError('Baseline trace span is out of bounds')
            span_list.append((start, end))
        trace_spans[key_int] = span_list
    return {
        'store_key': f'{store_path.resolve()}::{int(key1_byte)}::{int(key2_byte)}',
        'key1_values': key1_values,
        'key1_index': {int(val): idx for idx, val in enumerate(key1_values.tolist())},
        'section_mean': mu_section,
        'section_inv_std': inv_sigma_section,
        'section_clamp_mask': section_clamp_mask,
        'trace_mean': mu_traces,
        'trace_inv_std': inv_sigma_traces,
        'trace_clamp_mask': trace_clamp_mask,
        'trace_spans': trace_spans,
    }


def load_baseline(
    store_dir: str | Path, *, key1_byte: int, key2_byte: int
) -> dict[str, Any]:
    """Return cached baseline statistics for ``store_dir`` and key bytes."""
    store_path = Path(store_dir)
    try:
        resolved = read_split_baseline_payload(
            store_path,
            stage=BASELINE_STAGE_RAW,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            include_arrays=True,
        )
    except SplitBaselineArtifactsError:
        resolved = None
    if resolved is None:
        resolved = _load_legacy_baseline_payload(
            store_path,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
        )
    if resolved is None:
        manifest_path = build_baseline_manifest_path(
            store_path,
            stage=BASELINE_STAGE_RAW,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
        )
        legacy_path = build_legacy_baseline_path(store_path)
        raise FileNotFoundError(
            f'baseline payload not found: {manifest_path} or {legacy_path}'
        )
    payload, cache_key = resolved
    entry = _BASELINE_CACHE.get(cache_key)
    if entry is not None:
        return entry
    entry = _build_baseline_entry(
        payload=payload,
        store_path=store_path,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    _BASELINE_CACHE[cache_key] = entry
    return entry
