"""Common helpers for building and registering static-corrected TraceStores."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np

from app.core.state import AppState
from app.services.corrected_trace_store import (
    TimeShiftedTraceStoreResult,
    build_time_shifted_trace_store,
)
from app.services.trace_store_registration import (
    register_trace_store,
    trace_store_cache_key,
)

_SAFE_STORE_NAME_RE = re.compile(r'[^A-Za-z0-9_.-]+')


RegisterTraceStoreFn = Callable[..., object]


def corrected_store_path(
    *,
    source_store_path: Path,
    statics_kind: str,
    suffix: str,
    error_type: type[Exception] = ValueError,
) -> Path:
    """Return a non-existing sibling path for a static-corrected TraceStore."""
    source_name = safe_store_name_component(
        source_store_path.name,
        error_type=error_type,
    )
    safe_kind = safe_store_name_component(statics_kind, error_type=error_type)
    safe_suffix = safe_store_name_component(str(suffix), error_type=error_type)
    output_path = source_store_path.parent / f'{source_name}.statics.{safe_kind}.{safe_suffix}'
    if output_path.exists() or output_path.is_symlink():
        raise error_type(f'corrected output path already exists: {output_path}')
    return output_path


def safe_store_name_component(
    value: str,
    *,
    error_type: type[Exception] = ValueError,
) -> str:
    safe = _SAFE_STORE_NAME_RE.sub('_', str(value))
    if safe in {'', '.', '..'}:
        raise error_type('TraceStore name cannot be made filesystem-safe')
    return safe


def build_and_register_time_shifted_trace_store(
    *,
    state: AppState,
    corrected_file_id: str,
    source_store_path: Path,
    output_store_path: Path,
    trace_shift_s_sorted: np.ndarray,
    fill_value: float,
    output_dtype: str,
    derived_metadata: Mapping[str, Any],
    from_file_id: str,
    original_segy_path: str | None,
    key1_byte: int,
    key2_byte: int,
    header_bytes_to_materialize: Iterable[int] = (),
    register_fn: RegisterTraceStoreFn = register_trace_store,
    update_registry: bool = True,
    touch_meta: bool = True,
    preload_header_bytes: Iterable[int] = (),
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> TimeShiftedTraceStoreResult:
    """Build, register, and smoke-check a time-shifted corrected TraceStore."""
    build_result = build_time_shifted_trace_store(
        source_store_path=source_store_path,
        output_store_path=output_store_path,
        trace_shift_s_sorted=trace_shift_s_sorted,
        fill_value=fill_value,
        output_dtype=output_dtype,
        derived_metadata=derived_metadata,
        from_file_id=from_file_id,
        original_segy_path=original_segy_path,
        header_bytes_to_materialize=header_bytes_to_materialize,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    reader = register_fn(
        state=state,
        file_id=corrected_file_id,
        store_dir=build_result.store_path,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        dt=build_result.dt,
        update_registry=update_registry,
        touch_meta=touch_meta,
        preload_header_bytes=preload_header_bytes,
    )
    verify_registered_trace_store(
        state=state,
        file_id=corrected_file_id,
        store_path=build_result.store_path,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        reader=reader,
    )
    return build_result


def verify_registered_trace_store(
    *,
    state: AppState,
    file_id: str,
    store_path: Path,
    key1_byte: int,
    key2_byte: int,
    reader: object,
) -> None:
    registered_path = Path(state.file_registry.get_store_path(file_id))
    if registered_path.resolve() != store_path.resolve():
        raise RuntimeError('registered corrected TraceStore path mismatch')
    cache_key = trace_store_cache_key(file_id, key1_byte, key2_byte)
    with state.lock:
        if cache_key not in state.cached_readers:
            raise RuntimeError('registered corrected TraceStore reader is missing')
    key1_values = np.asarray(reader.get_key1_values())
    if key1_values.size == 0:
        raise RuntimeError('registered corrected TraceStore has no key1 values')
    reader.get_section(int(key1_values[0]))


def cleanup_registration(
    state: object,
    *,
    file_id: str,
    key1_byte: int,
    key2_byte: int,
) -> None:
    lock = getattr(state, 'lock', None)
    context = lock if lock is not None else nullcontext()
    with context:
        state.file_registry.pop(file_id, None)
        state.cached_readers.pop(trace_store_cache_key(file_id, key1_byte, key2_byte), None)


def cleanup_store(output_path: Path) -> None:
    for tmp_path in output_path.parent.glob(f'{output_path.name}.tmp-*'):
        if tmp_path.is_dir():
            shutil.rmtree(tmp_path, ignore_errors=True)
    if output_path.exists():
        shutil.rmtree(output_path, ignore_errors=True)


def cleanup_artifact(path: Path) -> None:
    path.unlink(missing_ok=True)
    for tmp_path in path.parent.glob(f'{path.name}.*.tmp'):
        tmp_path.unlink(missing_ok=True)


__all__ = [
    'RegisterTraceStoreFn',
    'build_and_register_time_shifted_trace_store',
    'cleanup_artifact',
    'cleanup_registration',
    'cleanup_store',
    'corrected_store_path',
    'safe_store_name_component',
    'verify_registered_trace_store',
]
