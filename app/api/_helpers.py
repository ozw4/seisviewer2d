"""Shared helpers and state for API routers."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

from app.api.schemas import PipelineSpec
from app.utils.fbpick import _MODEL_PATH as FBPICK_MODEL_PATH
from app.utils.segy_meta import FILE_REGISTRY, get_dt_for_file
from app.utils.utils import SegySectionReader, TraceStoreSectionReader

USE_FBPICK_OFFSET = 'offset' in FBPICK_MODEL_PATH.name.lower()
OFFSET_BYTE_FIXED: int = 37

EXPECTED_SECTION_NDIM = 2


class LRUCache(OrderedDict):
        """A tiny ordered cache used for in-memory tap storage."""

        def __init__(self, capacity: int = 16):
                super().__init__()
                self.capacity = capacity

        def get(self, key):
                if key in self:
                        self.move_to_end(key)
                        return super().__getitem__(key)
                return None

        def set(self, key, value):
                if key in self:
                        self.move_to_end(key)
                super().__setitem__(key, value)
                if len(self) > self.capacity:
                        self.popitem(last=False)


cached_readers: dict[str, SegySectionReader | TraceStoreSectionReader] = {}
SEGYS: dict[str, str] = {}
fbpick_cache: dict[tuple, bytes] = {}
jobs: dict[str, dict[str, object]] = {}
pipeline_tap_cache = LRUCache(16)
window_section_cache = LRUCache(32)


class PipelineTapNotFoundError(LookupError):
        """Raised when a requested pipeline tap output is unavailable."""


def _spec_uses_fbpick(spec: PipelineSpec) -> bool:
        """Return ``True`` when ``spec`` contains an fbpick analyzer step."""
        return any(step.kind == 'analyzer' and step.name == 'fbpick' for step in spec.steps)


def _maybe_attach_fbpick_offsets(
        meta: dict[str, Any],
        *,
        spec: PipelineSpec,
        reader: SegySectionReader | TraceStoreSectionReader,
        key1_val: int,
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
        offsets = get_offsets(key1_val, offset_byte)
        if trace_slice is not None:
                offsets = offsets[trace_slice]
        offsets = np.ascontiguousarray(offsets, dtype=np.float32)
        if not meta:
                return {'offsets': offsets}
        meta_with_offsets = dict(meta)
        meta_with_offsets['offsets'] = offsets
        return meta_with_offsets


def _update_file_registry(
        file_id: str,
        *,
        path: str | None = None,
        store_path: str | None = None,
        dt: float | None = None,
) -> None:
        rec = FILE_REGISTRY.get(file_id) or {}
        if path:
                rec['path'] = path
        if store_path:
                rec['store_path'] = store_path
        if isinstance(dt, (int, float)) and dt > 0:
                rec['dt'] = float(dt)
        FILE_REGISTRY[file_id] = rec


def _filename_for_file_id(file_id: str) -> str | None:
        rec = FILE_REGISTRY.get(file_id) or {}
        path = rec.get('path') or rec.get('store_path')
        return Path(path).name if path else None


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


def get_section_from_pipeline_tap(
        *,
        file_id: str,
        key1_val: int,
        key1_byte: int,
        pipeline_key: str,
        tap_label: str,
        offset_byte: int | None = None,
) -> np.ndarray:
        """Return the cached pipeline tap output as a ``float32`` array."""
        base_key = (file_id, key1_val, key1_byte, pipeline_key, None, offset_byte)
        payload = pipeline_tap_cache.get((*base_key, tap_label))
        if payload is None:
                msg = (
                        f'Pipeline tap {tap_label!r} for pipeline {pipeline_key!r} '
                        f'and key1={key1_val} is not available. '
                        'Please re-run the pipeline.'
                )
                raise PipelineTapNotFoundError(msg)
        return _pipeline_payload_to_array(payload, tap_label=tap_label)


def get_reader(
        file_id: str, key1_byte: int, key2_byte: int
) -> SegySectionReader | TraceStoreSectionReader:
        cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
        if cache_key not in cached_readers:
                if file_id not in SEGYS:
                        raise HTTPException(status_code=404, detail='File ID not found')
                path = SEGYS[file_id]
                p = Path(path)
                if p.is_dir():
                        reader = TraceStoreSectionReader(p, key1_byte, key2_byte)
                else:
                        reader = SegySectionReader(path, key1_byte, key2_byte)
                cached_readers[cache_key] = reader
        reader = cached_readers[cache_key]
        dt_val = get_dt_for_file(file_id)
        meta_attr = getattr(reader, 'meta', None)
        if isinstance(meta_attr, dict):
                if not isinstance(meta_attr.get('dt'), (int, float)) or meta_attr['dt'] <= 0:
                        meta_attr['dt'] = dt_val
        else:
                try:
                        reader.meta = {'dt': dt_val}
                except Exception:  # noqa: BLE001
                        pass
        return reader


def get_raw_section(
        *, file_id: str, key1_val: int, key1_byte: int, key2_byte: int
) -> np.ndarray:
        """Load the RAW seismic section as ``float32``."""
        reader = get_reader(file_id, key1_byte, key2_byte)
        section = reader.get_section(key1_val)
        arr = np.asarray(section, dtype=np.float32)
        if arr.ndim != EXPECTED_SECTION_NDIM:
                msg = f'Raw section expected 2D data, got {arr.ndim}D'
                raise ValueError(msg)
        return np.ascontiguousarray(arr)


__all__ = [
        'EXPECTED_SECTION_NDIM',
        'OFFSET_BYTE_FIXED',
        'SEGYS',
        'USE_FBPICK_OFFSET',
        'LRUCache',
        'PipelineTapNotFoundError',
        '_filename_for_file_id',
        '_maybe_attach_fbpick_offsets',
        '_pipeline_payload_to_array',
        '_spec_uses_fbpick',
        '_update_file_registry',
        'cached_readers',
        'fbpick_cache',
        'get_raw_section',
        'get_reader',
        'get_section_from_pipeline_tap',
        'jobs',
        'pipeline_tap_cache',
        'window_section_cache',
]

