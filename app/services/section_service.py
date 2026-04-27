"""Section-window construction service."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from app.api.binary_codec import pack_quantized_array_gzip
from app.services.linear_moveout import (
    compute_lmo_raw_sample_bounds,
    compute_lmo_shift_seconds,
    resample_lmo_window,
)
from app.services.reader import EXPECTED_SECTION_NDIM, coerce_section_f32
from app.services.scaling import (
    apply_scaling_from_baseline,
    apply_scaling_from_reference_section,
)
from app.trace_store.reader import TraceStoreSectionReader
from app.trace_store.types import SectionView


class SectionServiceInternalError(RuntimeError):
    """Raised when service detects an internal data/state inconsistency."""


def _load_section_view(
    *,
    file_id: str,
    key1: int,
    key1_byte: int,
    key2_byte: int,
    offset_byte: int | None,
    pipeline_key: str | None,
    tap_label: str | None,
    reader_getter: Callable[[str, int, int], TraceStoreSectionReader],
    pipeline_section_getter: Callable[..., np.ndarray],
) -> tuple[SectionView, TraceStoreSectionReader | None]:
    if pipeline_key and tap_label:
        section_arr = pipeline_section_getter(
            file_id=file_id,
            key1=key1,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            pipeline_key=pipeline_key,
            tap_label=tap_label,
            offset_byte=offset_byte,
        )
        return (
            SectionView(arr=section_arr, dtype=section_arr.dtype, scale=None),
            None,
        )
    reader = reader_getter(file_id, key1_byte, key2_byte)
    return reader.get_section(key1), reader


def _resolve_store_dir(
    *,
    file_id: str,
    reader: TraceStoreSectionReader | None,
    store_dir_resolver: Callable[[str], str],
) -> str:
    store_dir: str | None = None
    if reader is not None:
        maybe_store = getattr(reader, 'store_dir', None)
        if isinstance(maybe_store, (str, Path)):
            store_dir = str(maybe_store)
    if store_dir is None:
        resolved_store = store_dir_resolver(file_id)
        if isinstance(resolved_store, str) and resolved_store:
            store_dir = resolved_store
    if store_dir is None:
        raise SectionServiceInternalError('Trace store path unavailable')
    return store_dir


def _validate_reference_source(
    reference_pipeline_key: str | None,
    reference_tap_label: str | None,
) -> tuple[str | None, str | None]:
    has_pipeline_key = bool(reference_pipeline_key)
    has_tap_label = bool(reference_tap_label)
    if has_pipeline_key != has_tap_label:
        raise ValueError(
            'reference_pipeline_key and reference_tap_label must be provided together'
        )
    if not has_pipeline_key:
        return None, None
    return reference_pipeline_key, reference_tap_label


def _source_matches(
    *,
    pipeline_key: str | None,
    tap_label: str | None,
    reference_pipeline_key: str | None,
    reference_tap_label: str | None,
) -> bool:
    return (
        bool(pipeline_key)
        and bool(tap_label)
        and pipeline_key == reference_pipeline_key
        and tap_label == reference_tap_label
    )


def _load_lmo_shift_samples(
    *,
    reader: TraceStoreSectionReader,
    key1: int,
    n_traces: int,
    dt: float,
    velocity_mps: float | None,
    offset_byte: int,
    offset_scale: float,
    offset_mode: str,
    ref_mode: str,
    ref_trace: int | None,
    polarity: int,
) -> np.ndarray:
    """Load raw offsets and return per-section LMO shifts in sample units."""
    if velocity_mps is None:
        raise ValueError('lmo_velocity_mps is required when lmo_enabled=true')
    dt_val = float(dt)
    if not np.isfinite(dt_val) or dt_val <= 0.0:
        raise ValueError('dt must be finite and greater than 0 for LMO')

    get_offsets = getattr(reader, 'get_offsets_for_section', None)
    if not callable(get_offsets):
        raise ValueError('Offset header unavailable for LMO')
    try:
        offsets_raw = get_offsets(key1, int(offset_byte))
    except Exception as exc:  # noqa: BLE001
        raise ValueError('Failed to read offsets for LMO') from exc
    if offsets_raw is None:
        raise ValueError('Offsets are required for LMO')

    offsets = np.asarray(offsets_raw)
    if offsets.ndim == 1 and offsets.size == 0:
        raise ValueError('offsets must not be empty')
    if offsets.ndim == 1 and offsets.shape[0] != int(n_traces):
        raise ValueError('Offsets length does not match section trace count')

    shift_seconds = compute_lmo_shift_seconds(
        offsets,
        velocity_mps=velocity_mps,
        offset_scale=offset_scale,
        offset_mode=offset_mode,
        ref_mode=ref_mode,
        ref_trace=ref_trace,
        polarity=polarity,
    )
    if shift_seconds.shape[0] != int(n_traces):
        raise ValueError('Offsets length does not match section trace count')
    return shift_seconds / dt_val


def build_section_window_payload(
    *,
    file_id: str,
    key1: int,
    key1_byte: int,
    key2_byte: int,
    offset_byte: int | None,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    step_x: int,
    step_y: int,
    transpose: bool,
    pipeline_key: str | None,
    tap_label: str | None,
    scaling_mode: Literal['amax', 'tracewise'],
    trace_stats_cache: Any,
    reader_getter: Callable[[str, int, int], TraceStoreSectionReader],
    pipeline_section_getter: Callable[..., np.ndarray],
    store_dir_resolver: Callable[[str], str],
    trace_stats_lock: threading.RLock | None = None,
    dt_resolver: Callable[[str], float] | None = None,
    reference_pipeline_key: str | None = None,
    reference_tap_label: str | None = None,
    lmo_enabled: bool = False,
    lmo_velocity_mps: float | None = None,
    lmo_offset_byte: int = 37,
    lmo_offset_scale: float = 1.0,
    lmo_offset_mode: str = 'absolute',
    lmo_ref_mode: str = 'min',
    lmo_ref_trace: int | None = None,
    lmo_polarity: int = 1,
    perf_timings_ms: dict[str, float] | None = None,
) -> bytes:
    """Build the compressed binary payload for a section window."""
    build_started = time.perf_counter()
    mode = scaling_mode.lower()
    if mode not in {'amax', 'tracewise'}:
        raise ValueError('Unsupported scaling mode')
    reference_pipeline_key, reference_tap_label = _validate_reference_source(
        reference_pipeline_key,
        reference_tap_label,
    )

    section_view, reader = _load_section_view(
        file_id=file_id,
        key1=key1,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        offset_byte=offset_byte,
        pipeline_key=pipeline_key,
        tap_label=tap_label,
        reader_getter=reader_getter,
        pipeline_section_getter=pipeline_section_getter,
    )
    base = section_view.arr
    if base.ndim != EXPECTED_SECTION_NDIM:
        raise SectionServiceInternalError('Section data must be 2D')

    n_traces, n_samples = base.shape
    if not (0 <= x0 <= x1 < n_traces):
        raise ValueError('Trace range out of bounds')
    if not (0 <= y0 <= y1 < n_samples):
        raise ValueError('Sample range out of bounds')
    if step_x < 1 or step_y < 1:
        raise ValueError('Steps must be >= 1')
    if dt_resolver is None:
        raise SectionServiceInternalError('dt resolver is required')
    dt_val = dt_resolver(file_id)

    raw_reader = reader
    selected_shift_samples: np.ndarray | None = None
    raw_y0 = int(y0)
    raw_y1 = int(y1)
    if bool(lmo_enabled):
        if raw_reader is None:
            raw_reader = reader_getter(file_id, key1_byte, key2_byte)
        shift_samples = _load_lmo_shift_samples(
            reader=raw_reader,
            key1=key1,
            n_traces=n_traces,
            dt=dt_val,
            velocity_mps=lmo_velocity_mps,
            offset_byte=lmo_offset_byte,
            offset_scale=lmo_offset_scale,
            offset_mode=lmo_offset_mode,
            ref_mode=lmo_ref_mode,
            ref_trace=lmo_ref_trace,
            polarity=lmo_polarity,
        )
        selected_shift_samples = shift_samples[x0 : x1 + 1 : step_x]
        raw_y0, raw_y1 = compute_lmo_raw_sample_bounds(
            y0=y0,
            y1=y1,
            shift_samples=selected_shift_samples,
            n_samples=n_samples,
        )

    sample_slice = (
        slice(raw_y0, raw_y1 + 1)
        if selected_shift_samples is not None
        else slice(y0, y1 + 1, step_y)
    )
    sub = base[x0 : x1 + 1 : step_x, sample_slice]
    if sub.size == 0:
        raise ValueError('Requested window is empty')

    prepared = coerce_section_f32(sub, section_view.scale)
    if reference_pipeline_key and reference_tap_label:
        if _source_matches(
            pipeline_key=pipeline_key,
            tap_label=tap_label,
            reference_pipeline_key=reference_pipeline_key,
            reference_tap_label=reference_tap_label,
        ):
            reference_base = base
            reference_scale = section_view.scale
        else:
            reference_view, _ = _load_section_view(
                file_id=file_id,
                key1=key1,
                key1_byte=key1_byte,
                key2_byte=key2_byte,
                offset_byte=offset_byte,
                pipeline_key=reference_pipeline_key,
                tap_label=reference_tap_label,
                reader_getter=reader_getter,
                pipeline_section_getter=pipeline_section_getter,
            )
            reference_base = reference_view.arr
            reference_scale = reference_view.scale
        if reference_base.ndim != EXPECTED_SECTION_NDIM:
            raise SectionServiceInternalError('Reference source data must be 2D')
        reference_prepared = coerce_section_f32(reference_base, reference_scale)
        prepared = apply_scaling_from_reference_section(
            prepared,
            reference_prepared,
            scaling=mode,
            x0=x0,
            x1=x1,
            step_x=step_x,
        )
    else:
        store_dir = _resolve_store_dir(
            file_id=file_id,
            reader=raw_reader if raw_reader is not None else reader,
            store_dir_resolver=store_dir_resolver,
        )
        prepared = apply_scaling_from_baseline(
            prepared,
            scaling=mode,
            file_id=file_id,
            key1=key1,
            store_dir=store_dir,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            trace_stats_cache=trace_stats_cache,
            trace_stats_lock=trace_stats_lock,
            x0=x0,
            x1=x1,
            step_x=step_x,
        )
    if selected_shift_samples is not None:
        prepared = resample_lmo_window(
            prepared,
            y0=y0,
            y1=y1,
            step_y=step_y,
            raw_y0=raw_y0,
            shift_samples=selected_shift_samples,
        )
    build_ms = (time.perf_counter() - build_started) * 1000.0
    pack_started = time.perf_counter()
    payload = pack_quantized_array_gzip(
        prepared,
        scale=None,
        dt=dt_val,
        extra=None,
        transpose=transpose,
    )
    if perf_timings_ms is not None:
        perf_timings_ms['pack_ms'] = (time.perf_counter() - pack_started) * 1000.0
        perf_timings_ms['build_ms'] = build_ms
    return payload


__all__ = ['SectionServiceInternalError', 'build_section_window_payload']
