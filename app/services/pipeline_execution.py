"""Shared helpers for resolving and executing pipeline sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

from app.api.schemas import PipelineSpec
from app.core.state import AppState
from app.services.errors import (
    ConflictError,
    InternalError,
    NotFoundError,
    UnprocessableError,
)
from app.services.fbpick_support import (
    OFFSET_BYTE_FIXED,
    _maybe_attach_fbpick_offsets,
    _spec_uses_offset,
)
from app.services.pipeline_taps import (
    PipelineTapNotFoundError,
    get_section_and_meta_from_pipeline_tap,
)
from app.services.reader import EXPECTED_SECTION_NDIM, coerce_section_f32, get_reader
from app.trace_store.reader import TraceStoreSectionReader
from app.utils.pipeline import apply_pipeline


@dataclass(slots=True)
class SectionSourceSpec:
    file_id: str
    key1: int
    key1_byte: int
    key2_byte: int
    pipeline_key: str | None = None
    tap_label: str | None = None
    offset_byte: int | None = None
    window: dict[str, int | float] | None = None


@dataclass(slots=True)
class ResolvedSectionSource:
    section: np.ndarray
    reader: TraceStoreSectionReader | None
    source_kind: Literal['raw', 'pipeline_tap']
    source_meta: dict[str, Any] | None
    trace_slice: slice | None
    window_bounds: dict[str, int] | None


@dataclass(slots=True)
class PipelineExecutionContext:
    spec: PipelineSpec
    file_id: str
    key1: int
    key1_byte: int
    key2_byte: int
    section: np.ndarray
    reader: TraceStoreSectionReader | None
    source_kind: Literal['raw', 'pipeline_tap']
    source_meta: dict[str, Any] | None
    trace_slice: slice | None
    window_bounds: dict[str, int] | None
    dt: float | None
    effective_offset_byte: int | None
    meta: dict[str, Any]


ReaderGetter = Callable[..., TraceStoreSectionReader]
PipelineTapGetter = Callable[..., tuple[np.ndarray, dict[str, Any] | None]]
OffsetAttacher = Callable[..., dict[str, Any]]
ApplyPipelineFn = Callable[..., dict[str, Any]]


def _coerce_source_section(
    section: np.ndarray,
    *,
    detail_prefix: str,
) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(section, dtype=np.float32))
    if arr.ndim != EXPECTED_SECTION_NDIM:
        raise UnprocessableError(f'{detail_prefix} expected 2D data, got {arr.ndim}D')
    return arr


def resolve_effective_offset_byte(
    spec: PipelineSpec,
    offset_byte: int | None,
) -> int | None:
    """Return the offset byte that should be used for pipeline execution."""
    if _spec_uses_offset(spec) and offset_byte is None:
        return OFFSET_BYTE_FIXED
    return offset_byte


def resolve_section_source(
    source: SectionSourceSpec,
    *,
    state: AppState,
    reader: TraceStoreSectionReader | None = None,
    offset_byte: int | None = None,
    reader_getter: ReaderGetter | None = None,
    pipeline_tap_getter: PipelineTapGetter | None = None,
) -> ResolvedSectionSource:
    """Resolve either a raw section or a cached pipeline tap into a 2D array."""
    resolved_reader_getter = get_reader if reader_getter is None else reader_getter
    resolved_tap_getter = (
        get_section_and_meta_from_pipeline_tap
        if pipeline_tap_getter is None
        else pipeline_tap_getter
    )
    normalized_window = source.window or None
    resolved_offset_byte = (
        source.offset_byte if offset_byte is None else int(offset_byte)
    )
    has_pipeline_source = (
        source.pipeline_key is not None or source.tap_label is not None
    )
    if has_pipeline_source:
        if source.pipeline_key is None or source.tap_label is None:
            raise UnprocessableError(
                'pipeline_key and tap_label must be provided together'
            )
        if normalized_window is not None:
            raise UnprocessableError('window is not supported for pipeline tap source')
        try:
            section, tap_meta = resolved_tap_getter(
                file_id=source.file_id,
                key1=source.key1,
                key1_byte=source.key1_byte,
                key2_byte=source.key2_byte,
                pipeline_key=source.pipeline_key,
                tap_label=source.tap_label,
                offset_byte=resolved_offset_byte,
                state=state,
            )
        except PipelineTapNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        arr = _coerce_source_section(
            section,
            detail_prefix=f'Pipeline tap {source.tap_label!r}',
        )
        return ResolvedSectionSource(
            section=arr,
            reader=None,
            source_kind='pipeline_tap',
            source_meta=tap_meta if isinstance(tap_meta, dict) else None,
            trace_slice=None,
            window_bounds=None,
        )

    reader_obj = (
        reader
        if reader is not None
        else resolved_reader_getter(
            source.file_id, source.key1_byte, source.key2_byte, state=state
        )
    )
    view = reader_obj.get_section(source.key1)
    arr = coerce_section_f32(view.arr, view.scale)
    if arr.ndim != EXPECTED_SECTION_NDIM:
        raise UnprocessableError(f'Raw section expected 2D data, got {arr.ndim}D')

    trace_slice: slice | None = None
    window_bounds: dict[str, int] | None = None
    if normalized_window is not None:
        tr_min = int(normalized_window.get('tr_min', 0))
        tr_max = int(normalized_window.get('tr_max', arr.shape[0]))
        t_min = int(normalized_window.get('t_min', 0))
        t_max = int(normalized_window.get('t_max', arr.shape[1]))
        arr = arr[tr_min:tr_max, t_min:t_max]
        trace_slice = slice(tr_min, tr_max)
        window_bounds = {
            'tr_min': tr_min,
            'tr_max': tr_max,
            't_min': t_min,
            't_max': t_max,
        }
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return ResolvedSectionSource(
        section=arr,
        reader=reader_obj,
        source_kind='raw',
        source_meta=None,
        trace_slice=trace_slice,
        window_bounds=window_bounds,
    )


def resolve_execution_dt(
    file_id: str,
    source_meta: dict[str, Any] | None,
    *,
    state: AppState,
    validate_source_dt: bool = False,
) -> float:
    """Resolve execution ``dt`` from file registry and optionally validate source metadata."""
    dt_file = float(state.file_registry.get_dt(file_id))
    if dt_file <= 0:
        raise UnprocessableError('Non-positive dt value')
    if isinstance(source_meta, dict) and 'dt' in source_meta:
        dt_meta = source_meta.get('dt')
        if not isinstance(dt_meta, (int, float)):
            raise UnprocessableError('Invalid dt metadata')
        if dt_meta <= 0:
            raise UnprocessableError('Non-positive dt metadata')
        if validate_source_dt and abs(float(dt_meta) - dt_file) > 1e-9:
            raise ConflictError('dt mismatch between tap and source')
    return dt_file


def build_execution_meta(
    *,
    spec: PipelineSpec,
    file_id: str,
    key1: int,
    key1_byte: int,
    key2_byte: int,
    source_meta: dict[str, Any] | None,
    reader: TraceStoreSectionReader | None,
    trace_slice: slice | None,
    section_shape: tuple[int, int],
    offset_byte: int | None,
    state: AppState,
    validate_source_dt: bool = False,
    reader_getter: ReaderGetter | None = None,
    offset_attacher: OffsetAttacher | None = None,
    include_dt: bool = True,
) -> tuple[float | None, dict[str, Any], TraceStoreSectionReader | None]:
    """Build ``meta`` for ``apply_pipeline`` and ensure offsets are attached when required."""
    resolved_reader_getter = get_reader if reader_getter is None else reader_getter
    resolved_offset_attacher = (
        _maybe_attach_fbpick_offsets if offset_attacher is None else offset_attacher
    )
    dt: float | None = None
    meta: dict[str, Any] = {}
    if include_dt or validate_source_dt:
        dt = resolve_execution_dt(
            file_id,
            source_meta,
            state=state,
            validate_source_dt=validate_source_dt,
        )
        if include_dt:
            meta['dt'] = dt
    reader_obj = reader
    if reader_obj is None and _spec_uses_offset(spec) and offset_byte is not None:
        reader_obj = resolved_reader_getter(file_id, key1_byte, key2_byte, state=state)
    if reader_obj is not None:
        meta = resolved_offset_attacher(
            meta,
            spec=spec,
            reader=reader_obj,
            key1=key1,
            offset_byte=offset_byte,
            trace_slice=trace_slice,
            section_shape=section_shape,
        )
    return dt, meta, reader_obj


def prepare_pipeline_execution(
    *,
    spec: PipelineSpec,
    source: SectionSourceSpec,
    state: AppState,
    reader: TraceStoreSectionReader | None = None,
    validate_source_dt: bool = False,
    reader_getter: ReaderGetter | None = None,
    pipeline_tap_getter: PipelineTapGetter | None = None,
    offset_attacher: OffsetAttacher | None = None,
    include_dt: bool = True,
) -> PipelineExecutionContext:
    """Resolve a source section and assemble the metadata required to run a pipeline."""
    effective_offset_byte = resolve_effective_offset_byte(spec, source.offset_byte)
    resolved_source = resolve_section_source(
        source,
        state=state,
        reader=reader,
        offset_byte=effective_offset_byte,
        reader_getter=reader_getter,
        pipeline_tap_getter=pipeline_tap_getter,
    )
    dt, meta, reader_obj = build_execution_meta(
        spec=spec,
        file_id=source.file_id,
        key1=source.key1,
        key1_byte=source.key1_byte,
        key2_byte=source.key2_byte,
        source_meta=resolved_source.source_meta,
        reader=resolved_source.reader,
        trace_slice=resolved_source.trace_slice,
        section_shape=(
            int(resolved_source.section.shape[0]),
            int(resolved_source.section.shape[1]),
        ),
        offset_byte=effective_offset_byte,
        state=state,
        validate_source_dt=validate_source_dt,
        reader_getter=reader_getter,
        offset_attacher=offset_attacher,
        include_dt=include_dt,
    )
    return PipelineExecutionContext(
        spec=spec,
        file_id=source.file_id,
        key1=source.key1,
        key1_byte=source.key1_byte,
        key2_byte=source.key2_byte,
        section=resolved_source.section,
        reader=reader_obj,
        source_kind=resolved_source.source_kind,
        source_meta=resolved_source.source_meta,
        trace_slice=resolved_source.trace_slice,
        window_bounds=resolved_source.window_bounds,
        dt=dt,
        effective_offset_byte=effective_offset_byte,
        meta=meta,
    )


def run_pipeline_execution(
    context: PipelineExecutionContext,
    *,
    taps: list[str] | None = None,
    apply_fn: ApplyPipelineFn | None = None,
) -> dict[str, Any]:
    """Execute ``context.spec`` with the pre-resolved section and metadata."""
    resolved_apply_fn = apply_pipeline if apply_fn is None else apply_fn
    return resolved_apply_fn(
        context.section,
        spec=context.spec,
        meta=context.meta,
        taps=taps,
    )


def build_fbpick_spec(
    *,
    model_id: str | None,
    channel: str | None,
    tile: tuple[int, int] | list[int],
    overlap: int | tuple[int, int] | list[int],
    amp: bool,
) -> PipelineSpec:
    """Build a one-step fbpick analyzer spec."""
    return PipelineSpec(
        steps=[
            {
                'kind': 'analyzer',
                'name': 'fbpick',
                'params': {
                    'tile': tuple(tile),
                    'overlap': overlap,
                    'amp': amp,
                    'model_id': model_id,
                    'channel': channel,
                },
            }
        ]
    )


def extract_fbpick_probability_map(
    outputs: dict[str, Any],
    *,
    label: str = 'fbpick',
) -> np.ndarray:
    """Extract a 2D probability map from analyzer outputs."""
    fbpick_out = outputs.get(label) or outputs.get('final')
    if not isinstance(fbpick_out, dict):
        raise InternalError('fbpick analyzer output missing')
    if 'prob' not in fbpick_out:
        raise InternalError('fbpick analyzer output missing')
    prob = np.asarray(fbpick_out['prob'], dtype=np.float32)
    if prob.ndim != EXPECTED_SECTION_NDIM:
        raise UnprocessableError('Probability map must be 2D')
    return np.ascontiguousarray(prob)


def extract_pipeline_outputs(
    outputs: dict[str, Any],
    *,
    denoise_taps: list[str],
    fbpick_label: str | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract denoise and fbpick outputs from one pipeline execution result."""
    denoise = None
    if denoise_taps:
        denoise_tap = denoise_taps[-1]
        tap_payload = outputs.get(denoise_tap)
        if not isinstance(tap_payload, dict) or 'data' not in tap_payload:
            raise ValueError(f'Denoise tap output missing: {denoise_tap}')
        denoise = np.asarray(tap_payload['data'], dtype=np.float32, order='C')

    prob = None
    if fbpick_label is not None:
        prob = extract_fbpick_probability_map(outputs, label=fbpick_label)

    return denoise, prob


__all__ = [
    'PipelineExecutionContext',
    'ResolvedSectionSource',
    'SectionSourceSpec',
    'build_execution_meta',
    'build_fbpick_spec',
    'extract_fbpick_probability_map',
    'extract_pipeline_outputs',
    'prepare_pipeline_execution',
    'resolve_effective_offset_byte',
    'resolve_execution_dt',
    'resolve_section_source',
    'run_pipeline_execution',
]
