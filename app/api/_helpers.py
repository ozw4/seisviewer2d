"""Shared helpers and state for API routers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import HTTPException, Request
from numpy.typing import NDArray

from app.core.state import AppState, LRUCache
from app.services.errors import DomainError
from app.services.pipeline_taps import (
    PipelineTapNotFoundError,
    _pipeline_payload_to_array,
    build_pipeline_tap_cache_base_key,
    build_pipeline_tap_cache_key,
    get_section_and_meta_from_pipeline_tap as _service_get_section_and_meta_from_pipeline_tap,
    get_section_from_pipeline_tap as _service_get_section_from_pipeline_tap,
)
from app.services.reader import (
    EXPECTED_SECTION_NDIM,
    coerce_section_f32,
    get_raw_section as _service_get_raw_section,
    get_reader as _service_get_reader,
)
from app.services.registry import _filename_for_file_id, _update_file_registry
from app.services.scaling import (
    apply_scaling_from_baseline as _service_apply_scaling_from_baseline,
)
from app.utils.fbpick import _MODEL_PATH as FBPICK_MODEL_PATH
from app.utils.utils import TraceStoreSectionReader

if TYPE_CHECKING:
    from fastapi import FastAPI

    from app.api.schemas import PipelineSpec

USE_FBPICK_OFFSET = 'offset' in FBPICK_MODEL_PATH.name.lower()
OFFSET_BYTE_FIXED: int = 37

logger = logging.getLogger(__name__)


def reject_legacy_key1_query_params(request: Request) -> None:
    """Reject legacy key1 query parameters for non-compatible endpoints."""
    legacy_val = 'key1' + '_val'
    legacy_idx = 'key1' + '_idx'
    present = [
        name for name in (legacy_val, legacy_idx) if name in request.query_params
    ]
    if present:
        names = ', '.join(sorted(present))
        raise HTTPException(
            status_code=422,
            detail=f'Legacy query parameter(s) are not supported: {names}; use key1',
        )


def get_state(app: FastAPI) -> AppState:
    """Return app-scoped state from ``app.state.sv``."""
    sv = getattr(getattr(app, 'state', None), 'sv', None)
    if isinstance(sv, AppState):
        return sv
    msg = 'Application state is not initialized (app.state.sv)'
    logger.error(msg)
    raise RuntimeError(msg)


def _resolve_state(
    *, app: FastAPI | None = None, state: AppState | None = None
) -> AppState:
    if state is not None:
        return state
    if app is None:
        raise RuntimeError('Either app or state must be provided')
    return get_state(app)


def _domain_error_to_http(exc: DomainError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


def _spec_uses_fbpick(spec: PipelineSpec) -> bool:
    """Return ``True`` when ``spec`` contains an fbpick analyzer step."""
    return any(step.kind == 'analyzer' and step.name == 'fbpick' for step in spec.steps)


def _maybe_attach_fbpick_offsets(
    meta: dict[str, Any],
    *,
    spec: PipelineSpec,
    reader: TraceStoreSectionReader,
    key1: int,
    offset_byte: int | None,
    trace_slice: slice | None = None,
) -> dict[str, Any]:
    """Add offset metadata when the fbpick model expects it."""
    if not USE_FBPICK_OFFSET or offset_byte is None:
        return meta
    if not _spec_uses_fbpick(spec):
        return meta
    get_offsets = getattr(reader, 'get_offsets_for_section', None)
    if get_offsets is None:
        return meta
    offsets = get_offsets(key1, offset_byte)
    if trace_slice is not None:
        offsets = offsets[trace_slice]
    offsets = np.ascontiguousarray(offsets, dtype=np.float32)
    if not meta:
        return {'offsets': offsets}
    meta_with_offsets = dict(meta)
    meta_with_offsets['offsets'] = offsets
    return meta_with_offsets


def apply_scaling_from_baseline(
    arr: NDArray[np.float32],
    scaling: str | None,
    file_id: str,
    key1: int,
    store_dir: str | Path,
    *,
    trace_stats_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray | None, int]],
    x0: int,
    x1: int,
    step_x: int,
) -> NDArray[np.float32]:
    return _service_apply_scaling_from_baseline(
        arr,
        scaling=scaling,
        file_id=file_id,
        key1=key1,
        store_dir=store_dir,
        trace_stats_cache=trace_stats_cache,
        x0=x0,
        x1=x1,
        step_x=step_x,
    )


def get_section_from_pipeline_tap(
    *,
    file_id: str,
    key1: int,
    key1_byte: int,
    key2_byte: int,
    pipeline_key: str,
    tap_label: str,
    offset_byte: int | None = None,
    app: FastAPI | None = None,
    state: AppState | None = None,
) -> np.ndarray:
    """Return the cached pipeline tap output as a ``float32`` array."""
    sv = _resolve_state(app=app, state=state)
    return _service_get_section_from_pipeline_tap(
        file_id=file_id,
        key1=key1,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        pipeline_key=pipeline_key,
        tap_label=tap_label,
        offset_byte=offset_byte,
        state=sv,
    )


def get_section_and_meta_from_pipeline_tap(
    *,
    file_id: str,
    key1: int,
    key1_byte: int,
    key2_byte: int,
    pipeline_key: str,
    tap_label: str,
    offset_byte: int | None = None,
    app: FastAPI | None = None,
    state: AppState | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Return tap payload as ``float32`` along with optional metadata."""
    sv = _resolve_state(app=app, state=state)
    return _service_get_section_and_meta_from_pipeline_tap(
        file_id=file_id,
        key1=key1,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        pipeline_key=pipeline_key,
        tap_label=tap_label,
        offset_byte=offset_byte,
        state=sv,
    )


def get_reader(
    file_id: str,
    key1_byte: int,
    key2_byte: int,
    *,
    app: FastAPI | None = None,
    state: AppState | None = None,
) -> TraceStoreSectionReader:
    sv = _resolve_state(app=app, state=state)
    return _service_get_reader(file_id, key1_byte, key2_byte, state=sv)


def get_raw_section(
    *,
    file_id: str,
    key1: int,
    key1_byte: int,
    key2_byte: int,
    app: FastAPI | None = None,
    state: AppState | None = None,
) -> np.ndarray:
    """Load the RAW seismic section as ``float32``."""
    sv = _resolve_state(app=app, state=state)
    return _service_get_raw_section(
        file_id=file_id,
        key1=key1,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        state=sv,
    )


__all__ = [
    'AppState',
    'EXPECTED_SECTION_NDIM',
    'OFFSET_BYTE_FIXED',
    'USE_FBPICK_OFFSET',
    'LRUCache',
    'PipelineTapNotFoundError',
    '_filename_for_file_id',
    '_maybe_attach_fbpick_offsets',
    '_pipeline_payload_to_array',
    '_spec_uses_fbpick',
    '_update_file_registry',
    'apply_scaling_from_baseline',
    'build_pipeline_tap_cache_base_key',
    'build_pipeline_tap_cache_key',
    'coerce_section_f32',
    'get_raw_section',
    'get_reader',
    'get_section_and_meta_from_pipeline_tap',
    'get_section_from_pipeline_tap',
    'get_state',
    'reject_legacy_key1_query_params',
]
