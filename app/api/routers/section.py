"""Section retrieval and binary I/O endpoints."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

if TYPE_CHECKING:
    from numpy.typing import NDArray
else:  # pragma: no cover - runtime alias for type checkers
    NDArray = np.ndarray

from app.api._helpers import (
    OFFSET_BYTE_FIXED,
    USE_FBPICK_OFFSET,
    PipelineTapNotFoundError,
    get_reader,
    reject_legacy_key1_query_params,
    get_state,
    get_section_from_pipeline_tap,
)
from app.api.baselines import (
    BASELINE_STAGE_RAW,
    BaselineComputationError,
    get_or_create_raw_baseline,
)
from app.core.state import AppState
from app.services.section_service import (
    SectionServiceInternalError,
    build_section_window_payload,
)
from app.utils.segy_meta import FILE_REGISTRY, get_dt_for_file

router = APIRouter()


class SectionMeta(BaseModel):
    shape: list[int]
    dt: float
    dtype: str | None = None
    scale: float | None = None


def _build_window_section_cache_key(
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
    scaling_mode: str,
) -> tuple[
    str,
    int,
    int,
    int,
    int | None,
    int,
    int,
    int,
    int,
    int,
    int,
    bool,
    str | None,
    str | None,
    str,
]:
    """Build the canonical cache key for section-window binary payloads."""
    return (
        file_id,
        int(key1),
        int(key1_byte),
        int(key2_byte),
        offset_byte,
        int(x0),
        int(x1),
        int(y0),
        int(y1),
        int(step_x),
        int(step_y),
        bool(transpose),
        pipeline_key,
        tap_label,
        str(scaling_mode),
    )


def get_ntraces_for(
    file_id: str,
    key1_byte: int | None = None,
    key2_byte: int | None = None,
    *,
    state: AppState,
) -> int:
    """Return total number of traces for ``file_id``.
    Always uses ``get_reader`` (lazy-safe). Falls back to registry meta if needed.
    """
    ent = FILE_REGISTRY.get(file_id)
    if ent is None:
        raise KeyError(f'file_id not found: {file_id}')

    kb1 = 189 if key1_byte is None else int(key1_byte)
    kb2 = 193 if key2_byte is None else int(key2_byte)

    # Prefer an actual reader (lazy open). If it fails, try registry meta.
    with contextlib.suppress(Exception):
        reader = get_reader(file_id, kb1, kb2, state=state)
        ntraces = getattr(reader, 'ntraces', None)
        if ntraces is None:
            meta = getattr(reader, 'meta', None)
            if isinstance(meta, dict):
                ntraces = meta.get('n_traces')
        if ntraces is None and hasattr(reader, 'traces'):
            ntraces = getattr(reader.traces, 'shape', (None,))[0]
        if ntraces is None and hasattr(reader, 'key1s'):
            ntraces = len(reader.key1s)
        if ntraces is not None:
            return int(ntraces)

    # Fallback: registry meta (if present)
    meta = getattr(ent, 'meta', None)
    if isinstance(ent, dict) and meta is None:
        meta = ent.get('meta')
    if isinstance(meta, dict) and 'n_traces' in meta:
        return int(meta['n_traces'])

    raise AttributeError('Unable to determine number of traces for file')


def get_trace_seq_for_value(
    file_id: str,
    key1: int,
    key1_byte: int,
    *,
    state: AppState,
) -> NDArray[np.int64]:
    """Return display-aligned trace ordering for ``key1`` of ``file_id``."""
    key2_byte = 193
    ent = FILE_REGISTRY.get(file_id)
    if ent is None:
        raise KeyError(f'file_id not found: {file_id}')
    maybe_reader = getattr(ent, 'reader', None)
    if maybe_reader is None and isinstance(ent, dict):
        maybe_reader = ent.get('reader')
    if maybe_reader is not None:
        key2_byte = int(getattr(maybe_reader, 'key2_byte', 193))

    reader = get_reader(file_id, int(key1_byte), key2_byte, state=state)
    target_val = int(key1)

    get_trace_seq = getattr(reader, 'get_trace_seq_for_section', None)
    if callable(get_trace_seq):
        seq = get_trace_seq(target_val, align_to='display')
        return np.asarray(seq, dtype=np.int64)

    get_header = getattr(reader, 'get_header', None)
    if callable(get_header):
        key1_headers = np.asarray(get_header(int(key1_byte)), dtype=np.int64)
        indices = np.flatnonzero(key1_headers == target_val)
        if indices.size == 0:
            msg = f'Key1 value {target_val} not found'
            raise ValueError(msg)
        key2_src = int(getattr(reader, 'key2_byte', key2_byte))
        key2_headers = np.asarray(get_header(key2_src), dtype=np.int64)
        order = np.argsort(key2_headers[indices], kind='stable')
        return np.asarray(indices[order], dtype=np.int64)

    msg = 'Reader cannot provide trace sequence information'
    raise AttributeError(msg)


@router.get('/get_key1_values')
def get_key1_values(
    request: Request,
    file_id: Annotated[str, Query(...)],
    key1_byte: Annotated[int, Query()] = 189,
    key2_byte: Annotated[int, Query()] = 193,
) -> JSONResponse:
    """Return the available key1 header values for ``file_id``."""
    state = get_state(request.app)
    reader = get_reader(file_id, key1_byte, key2_byte, state=state)
    values = reader.get_key1_values()
    payload = values.tolist() if isinstance(values, np.ndarray) else list(values)
    return JSONResponse(content={'values': payload})


@router.get('/get_section', dependencies=[Depends(reject_legacy_key1_query_params)])
def get_section(
    request: Request,
    file_id: Annotated[str, Query(...)],
    key1: Annotated[int, Query(...)],
    key1_byte: Annotated[int, Query()] = 189,
    key2_byte: Annotated[int, Query()] = 193,
) -> JSONResponse:
    """Return the section for the ``key1`` trace grouping."""
    try:
        state = get_state(request.app)
        reader = get_reader(file_id, key1_byte, key2_byte, state=state)
        view = reader.get_section(key1)
        payload = view.arr.tolist()
        return JSONResponse(content={'section': payload})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get('/section/stats', dependencies=[Depends(reject_legacy_key1_query_params)])
def get_section_stats(
    request: Request,
    file_id: Annotated[str, Query(...)],
    baseline: Annotated[str, Query(...)],
    key1: Annotated[int | None, Query()] = None,
    key1_byte: Annotated[int, Query()] = 189,
    key2_byte: Annotated[int, Query()] = 193,
    step_x: Annotated[int | None, Query()] = None,
    step_y: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    baseline_value = baseline.lower().strip()
    if baseline_value != BASELINE_STAGE_RAW:
        raise HTTPException(status_code=400, detail='Only baseline=raw is supported')
    for name, value in (('step_x', step_x), ('step_y', step_y)):
        if value is not None and int(value) != 1:
            raise HTTPException(
                status_code=400,
                detail=f'{name} must equal 1 for raw baseline',
            )
    try:
        payload = get_or_create_raw_baseline(
            file_id=file_id,
            key1_byte=int(key1_byte),
            key2_byte=int(key2_byte),
            app=request.app,
        )
    except BaselineComputationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    response_payload = dict(payload)
    if key1 is not None:
        key1_value = int(key1)
        key1_values = response_payload.get('key1_values') or []
        try:
            pos = key1_values.index(key1_value)
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail=f'key1 {key1_value} not found in baseline',
            ) from exc
        trace_spans_map = response_payload.get('trace_spans_by_key1') or {}
        trace_spans = trace_spans_map.get(str(key1_value))
        if trace_spans is None:
            trace_spans = []
        selected = {
            'key1': key1_value,
            'mu_section': response_payload['mu_section_by_key1'][pos],
            'sigma_section': response_payload['sigma_section_by_key1'][pos],
            'trace_spans': trace_spans,
        }
        if len(trace_spans) == 1:
            selected['trace_range'] = trace_spans[0]
        response_payload['selected_key1'] = selected
    return JSONResponse(content=response_payload)


@router.get('/get_section_meta', response_model=SectionMeta)
def get_section_meta(
    request: Request,
    file_id: Annotated[str, Query(...)],
    key1_byte: Annotated[int, Query()] = 189,
    key2_byte: Annotated[int, Query()] = 193,
) -> SectionMeta:
    state = get_state(request.app)
    reader = get_reader(file_id, key1_byte, key2_byte, state=state)

    # セクション形状を実データから最短経路で確定
    values = reader.get_key1_values()
    first_val = int(values[0])
    n_traces = int(reader.get_trace_seq_for_value(first_val, align_to='display').size)
    n_samples = int(reader.get_n_samples())

    dtype = str(reader.dtype) if reader.dtype is not None else None
    scale = float(reader.scale) if isinstance(reader.scale, (int, float)) else None
    dt_val = float(get_dt_for_file(file_id))
    _ = get_or_create_raw_baseline(
        file_id=file_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        app=request.app,
    )

    return SectionMeta(
        shape=[n_traces, n_samples],  # セクション内 [traces, samples]
        dt=dt_val,
        dtype=dtype,
        scale=scale,
    )


@router.get(
    '/get_section_window_bin', dependencies=[Depends(reject_legacy_key1_query_params)]
)
def get_section_window_bin(
    request: Request,
    file_id: Annotated[str, Query(...)],
    key1: Annotated[int, Query(...)],
    x0: Annotated[int, Query(...)],
    x1: Annotated[int, Query(...)],
    y0: Annotated[int, Query(...)],
    y1: Annotated[int, Query(...)],
    key1_byte: Annotated[int, Query()] = 189,
    key2_byte: Annotated[int, Query()] = 193,
    offset_byte: Annotated[int | None, Query()] = None,
    step_x: Annotated[int, Query(ge=1)] = 1,
    step_y: Annotated[int, Query(ge=1)] = 1,
    transpose: Annotated[bool, Query()] = True,
    pipeline_key: Annotated[str | None, Query()] = None,
    tap_label: Annotated[str | None, Query()] = None,
    scaling: Annotated[
        Literal['amax', 'tracewise'] | None, Query(description='Normalization mode')
    ] = None,
) -> Response:
    """Return a quantized window of a section, optionally via a pipeline tap."""
    state = get_state(request.app)
    mode = 'amax' if scaling is None else scaling
    mode = mode.lower()
    if mode not in {'amax', 'tracewise'}:
        raise HTTPException(status_code=400, detail='Unsupported scaling mode')
    forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else offset_byte
    cache_key = _build_window_section_cache_key(
        file_id=file_id,
        key1=key1,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        offset_byte=forced_offset_byte,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        step_x=step_x,
        step_y=step_y,
        transpose=transpose,
        pipeline_key=pipeline_key,
        tap_label=tap_label,
        scaling_mode=mode,
    )

    cached_payload = state.window_section_cache.get(cache_key)
    if cached_payload is not None:
        return Response(
            cached_payload,
            media_type='application/octet-stream',
            headers={'Content-Encoding': 'gzip'},
        )

    try:
        compressed = build_section_window_payload(
            file_id=file_id,
            key1=key1,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            offset_byte=forced_offset_byte,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            step_x=step_x,
            step_y=step_y,
            transpose=transpose,
            pipeline_key=pipeline_key,
            tap_label=tap_label,
            scaling_mode=mode,
            trace_stats_cache=state.trace_stats_cache,
            reader_getter=lambda fid, kb1, kb2: get_reader(fid, kb1, kb2, state=state),
            pipeline_section_getter=lambda **kwargs: get_section_from_pipeline_tap(
                **kwargs, state=state
            ),
            dt_resolver=get_dt_for_file,
        )
    except SectionServiceInternalError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
    except PipelineTapNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    state.window_section_cache.set(cache_key, compressed)
    return Response(
        compressed,
        media_type='application/octet-stream',
        headers={'Content-Encoding': 'gzip'},
    )
