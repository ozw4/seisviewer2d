"""Section retrieval and binary I/O endpoints."""

from __future__ import annotations

import time
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from app.api._helpers import reject_legacy_key1_query_params, get_state
from app.api.baselines import (
    BASELINE_STAGE_RAW,
    BaselineComputationError,
    get_or_create_raw_baseline,
)
from app.services.errors import DomainError
from app.services.fbpick_support import DEFAULT_FBPICK_MODEL_ID, OFFSET_BYTE_FIXED
from app.services.pipeline_taps import (
    PipelineTapNotFoundError,
    get_section_from_pipeline_tap,
)
from app.services.reader import get_reader
from app.services.section_service import (
    SectionServiceInternalError,
    build_section_window_payload,
)

router = APIRouter()


class SectionMeta(BaseModel):
    shape: list[int]
    dt: float
    dtype: str | None = None
    scale: float | None = None


def _format_ms(value: float) -> str:
    """Format a duration in milliseconds for response headers."""
    return f'{max(float(value), 0.0):.3f}'


def _build_window_perf_headers(
    *,
    cache_state: str,
    total_ms: float,
    build_ms: float,
    pack_ms: float,
    payload_bytes: int,
) -> dict[str, str]:
    total_str = _format_ms(total_ms)
    build_str = _format_ms(build_ms)
    pack_str = _format_ms(pack_ms)
    return {
        'Content-Encoding': 'gzip',
        'Server-Timing': (
            f'sv_total;dur={total_str}, '
            f'sv_build;dur={build_str}, '
            f'sv_pack;dur={pack_str}, '
            f'sv_cache;desc="{cache_state}"'
        ),
        'X-SV-Cache': cache_state,
        'X-SV-Server-Ms': total_str,
        'X-SV-Build-Ms': build_str,
        'X-SV-Pack-Ms': pack_str,
        'X-SV-Bytes': str(int(payload_bytes)),
    }


def _build_meta_perf_headers(
    *,
    total_ms: float,
    baseline_ms: float,
    baseline_source: str,
) -> dict[str, str]:
    total_str = _format_ms(total_ms)
    baseline_str = _format_ms(baseline_ms)
    return {
        'Server-Timing': (
            f'sv_total;dur={total_str}, '
            f'sv_baseline;dur={baseline_str}, '
            f'sv_baseline_source;desc="{baseline_source}"'
        ),
        'X-SV-Server-Ms': total_str,
        'X-SV-Baseline-Ms': baseline_str,
        'X-SV-Baseline-Source': baseline_source,
    }


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
    reference_pipeline_key: str | None,
    reference_tap_label: str | None,
    scaling_mode: str,
    lmo_enabled: bool = False,
    lmo_velocity_mps: float | None = None,
    lmo_offset_byte: int = 37,
    lmo_offset_scale: float = 1.0,
    lmo_offset_mode: str = 'absolute',
    lmo_ref_mode: str = 'min',
    lmo_ref_trace: int | None = None,
    lmo_polarity: int = 1,
) -> tuple[object, ...]:
    """Build the canonical cache key for section-window binary payloads."""
    base_key: tuple[object, ...] = (
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
        reference_pipeline_key,
        reference_tap_label,
        str(scaling_mode),
    )
    if not bool(lmo_enabled):
        return base_key
    return (
        *base_key,
        'lmo',
        True,
        None if lmo_velocity_mps is None else float(lmo_velocity_mps),
        int(lmo_offset_byte),
        float(lmo_offset_scale),
        str(lmo_offset_mode),
        str(lmo_ref_mode),
        None if lmo_ref_trace is None else int(lmo_ref_trace),
        int(lmo_polarity),
    )


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
    except DomainError:
        raise
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
) -> JSONResponse:
    route_started = time.perf_counter()
    state = get_state(request.app)
    reader = get_reader(file_id, key1_byte, key2_byte, state=state)

    # セクション形状を実データから最短経路で確定
    values = reader.get_key1_values()
    first_val = int(values[0])
    n_traces = int(reader.get_trace_seq_for_value(first_val, align_to='display').size)
    n_samples = int(reader.get_n_samples())

    dtype = str(reader.dtype) if reader.dtype is not None else None
    scale = float(reader.scale) if isinstance(reader.scale, (int, float)) else None
    dt_val = float(state.file_registry.get_dt(file_id))
    baseline_status: dict[str, str] = {}
    baseline_started = time.perf_counter()
    get_or_create_raw_baseline(
        file_id=file_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        app=request.app,
        include_arrays=False,
        status=baseline_status,
    )
    baseline_ms = (time.perf_counter() - baseline_started) * 1000.0
    payload = SectionMeta(
        shape=[n_traces, n_samples],  # セクション内 [traces, samples]
        dt=dt_val,
        dtype=dtype,
        scale=scale,
    )
    return JSONResponse(
        content=payload.model_dump(),
        headers=_build_meta_perf_headers(
            total_ms=(time.perf_counter() - route_started) * 1000.0,
            baseline_ms=baseline_ms,
            baseline_source=baseline_status.get('source', 'unknown'),
        ),
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
    reference_pipeline_key: Annotated[str | None, Query()] = None,
    reference_tap_label: Annotated[str | None, Query()] = None,
    scaling: Annotated[
        Literal['amax', 'tracewise'] | None, Query(description='Normalization mode')
    ] = None,
    lmo_enabled: Annotated[bool, Query()] = False,
    lmo_velocity_mps: Annotated[float | None, Query()] = None,
    lmo_offset_byte: Annotated[int, Query()] = 37,
    lmo_offset_scale: Annotated[float, Query()] = 1.0,
    lmo_offset_mode: Annotated[str, Query()] = 'absolute',
    lmo_ref_mode: Annotated[str, Query()] = 'min',
    lmo_ref_trace: Annotated[int | None, Query()] = None,
    lmo_polarity: Annotated[int, Query()] = 1,
) -> Response:
    """Return a quantized window of a section, optionally via a pipeline tap."""
    route_started = time.perf_counter()
    state = get_state(request.app)
    mode = 'amax' if scaling is None else scaling
    mode = mode.lower()
    if mode not in {'amax', 'tracewise'}:
        raise HTTPException(status_code=400, detail='Unsupported scaling mode')
    uses_offset = "offset" in DEFAULT_FBPICK_MODEL_ID.lower()
    forced_offset_byte = OFFSET_BYTE_FIXED if uses_offset else offset_byte
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
        reference_pipeline_key=reference_pipeline_key,
        reference_tap_label=reference_tap_label,
        scaling_mode=mode,
        lmo_enabled=lmo_enabled,
        lmo_velocity_mps=lmo_velocity_mps,
        lmo_offset_byte=lmo_offset_byte,
        lmo_offset_scale=lmo_offset_scale,
        lmo_offset_mode=lmo_offset_mode,
        lmo_ref_mode=lmo_ref_mode,
        lmo_ref_trace=lmo_ref_trace,
        lmo_polarity=lmo_polarity,
    )

    with state.lock:
        cached_payload = state.window_section_cache.get(cache_key)
    if cached_payload is not None:
        return Response(
            cached_payload,
            media_type='application/octet-stream',
            headers=_build_window_perf_headers(
                cache_state='hit',
                total_ms=(time.perf_counter() - route_started) * 1000.0,
                build_ms=0.0,
                pack_ms=0.0,
                payload_bytes=len(cached_payload),
            ),
        )

    perf_timings_ms: dict[str, float] = {}
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
            reference_pipeline_key=reference_pipeline_key,
            reference_tap_label=reference_tap_label,
            scaling_mode=mode,
            lmo_enabled=lmo_enabled,
            lmo_velocity_mps=lmo_velocity_mps,
            lmo_offset_byte=lmo_offset_byte,
            lmo_offset_scale=lmo_offset_scale,
            lmo_offset_mode=lmo_offset_mode,
            lmo_ref_mode=lmo_ref_mode,
            lmo_ref_trace=lmo_ref_trace,
            lmo_polarity=lmo_polarity,
            trace_stats_cache=state.trace_stats_cache,
            trace_stats_lock=state.lock,
            reader_getter=lambda fid, kb1, kb2: get_reader(fid, kb1, kb2, state=state),
            pipeline_section_getter=lambda **kwargs: get_section_from_pipeline_tap(
                **kwargs, state=state
            ),
            dt_resolver=lambda fid: state.file_registry.get_dt(fid),
            store_dir_resolver=lambda fid: state.file_registry.get_store_path(fid),
            perf_timings_ms=perf_timings_ms,
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

    with state.lock:
        cached_payload = state.window_section_cache.get(cache_key)
        if cached_payload is None:
            state.window_section_cache.set(cache_key, compressed)
            cache_state = 'miss'
        else:
            compressed = cached_payload
            cache_state = 'hit-after-build'
    return Response(
        compressed,
        media_type='application/octet-stream',
        headers=_build_window_perf_headers(
            cache_state=cache_state,
            total_ms=(time.perf_counter() - route_started) * 1000.0,
            build_ms=perf_timings_ms.get('build_ms', 0.0),
            pack_ms=perf_timings_ms.get('pack_ms', 0.0),
            payload_bytes=len(compressed),
        ),
    )
