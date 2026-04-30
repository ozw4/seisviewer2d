"""Build derived TraceStores with per-trace time shifts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

import numpy as np

from app.services.reader import coerce_section_f32
from app.services.trace_store_baselines import write_trace_store_raw_baseline_artifacts
from app.services.trace_store_headers import write_header_array_atomic
from app.services.trace_store_index_validation import validate_sorted_to_original
from app.trace_store.reader import TraceStoreSectionReader
from app.utils.time_shift import (
    coerce_finite_float32_fill_value,
    shift_traces_linear,
)

_HEADER_FILENAME_RE = re.compile(r'^headers_byte_(\d+)\.npy$')
_RESERVED_DERIVED_KEYS = {
    'kind',
    'from_store_path',
    'header_source_store_path',
    'sign_convention',
    'static_value_origin',
    'interpolation',
    'fill_value',
    'output_dtype',
    'applied_shift_summary',
}


@dataclass(frozen=True)
class TimeShiftedTraceStoreResult:
    store_path: Path
    source_store_path: Path
    n_traces: int
    n_samples: int
    dt: float
    key1_byte: int
    key2_byte: int
    output_dtype: str
    shift_summary: dict[str, float]


@dataclass(frozen=True)
class _SourceMeta:
    key1_byte: int
    key2_byte: int
    dt: float
    n_traces: int
    n_samples: int


@dataclass(frozen=True)
class _IndexData:
    key1_values: np.ndarray
    key1_offsets: np.ndarray
    key1_counts: np.ndarray
    sorted_to_original: np.ndarray


def _notify(
    progress_callback: Callable[[float, str], None] | None,
    progress: float,
    message: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback(float(progress), message)


def _raise_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise RuntimeError('time-shifted TraceStore build cancelled')


def _require_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        msg = f'{name} must be an integer'
        raise ValueError(msg)
    if isinstance(value, int | np.integer):
        return int(value)
    msg = f'{name} must be an integer'
    raise ValueError(msg)


def _validate_source_paths(
    *,
    source_store_path: str | Path,
    output_store_path: str | Path,
) -> tuple[Path, Path]:
    source_path = Path(source_store_path)
    output_path = Path(output_store_path)

    if not source_path.exists():
        msg = f'source_store_path does not exist: {source_path}'
        raise ValueError(msg)
    if not source_path.is_dir():
        msg = f'source_store_path must be a directory: {source_path}'
        raise ValueError(msg)
    if source_path.resolve() == output_path.resolve():
        msg = 'source_store_path and output_store_path must be different'
        raise ValueError(msg)
    if output_path.exists() or output_path.is_symlink():
        msg = f'output_store_path already exists: {output_path}'
        raise ValueError(msg)

    return source_path, output_path


def _load_source_meta(source_path: Path, traces_shape: tuple[int, int]) -> _SourceMeta:
    meta_path = source_path / 'meta.json'
    if not meta_path.exists():
        msg = f'source meta.json is missing: {meta_path}'
        raise ValueError(msg)

    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        msg = f'source meta.json is invalid JSON: {meta_path}'
        raise ValueError(msg) from exc
    if not isinstance(meta, dict):
        msg = 'source meta.json must be an object'
        raise ValueError(msg)

    try:
        n_traces = _require_int(meta['n_traces'], name='meta.n_traces')
        n_samples = _require_int(meta['n_samples'], name='meta.n_samples')
    except KeyError as exc:
        msg = f'source meta.json is missing {exc.args[0]}'
        raise ValueError(msg) from exc
    if n_traces <= 0:
        msg = 'source meta n_traces must be positive'
        raise ValueError(msg)
    if n_samples <= 0:
        msg = 'source meta n_samples must be positive'
        raise ValueError(msg)
    if (n_traces, n_samples) != traces_shape:
        msg = (
            'source meta trace shape does not match traces.npy: '
            f'meta={(n_traces, n_samples)}, traces={traces_shape}'
        )
        raise ValueError(msg)

    key_bytes = meta.get('key_bytes')
    if not isinstance(key_bytes, dict):
        msg = 'source meta key_bytes must be an object'
        raise ValueError(msg)
    try:
        key1_byte = _require_int(key_bytes['key1'], name='meta.key_bytes.key1')
        key2_byte = _require_int(key_bytes['key2'], name='meta.key_bytes.key2')
    except KeyError as exc:
        msg = f'source meta key_bytes is missing {exc.args[0]}'
        raise ValueError(msg) from exc

    try:
        dt = float(meta['dt'])
    except (KeyError, TypeError, ValueError) as exc:
        msg = 'source meta dt must be finite and greater than 0'
        raise ValueError(msg) from exc
    if not np.isfinite(dt) or dt <= 0.0:
        msg = 'source meta dt must be finite and greater than 0'
        raise ValueError(msg)

    return _SourceMeta(
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        dt=dt,
        n_traces=n_traces,
        n_samples=n_samples,
    )


def _load_source_traces(source_path: Path) -> np.ndarray:
    traces_path = source_path / 'traces.npy'
    if not traces_path.exists():
        msg = f'source traces.npy is missing: {traces_path}'
        raise ValueError(msg)
    try:
        traces = np.load(traces_path, mmap_mode='r')
    except Exception as exc:  # noqa: BLE001
        msg = f'source traces.npy could not be loaded: {traces_path}'
        raise ValueError(msg) from exc
    if traces.ndim != 2:
        msg = 'source traces.npy must be 2-dimensional'
        raise ValueError(msg)
    n_traces, n_samples = (int(traces.shape[0]), int(traces.shape[1]))
    if n_traces <= 0:
        msg = 'source traces.npy must contain at least one trace'
        raise ValueError(msg)
    if n_samples <= 0:
        msg = 'source traces.npy must contain at least one sample'
        raise ValueError(msg)
    return traces


def _ensure_index_i64(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        msg = f'{name} must be 1-dimensional'
        raise ValueError(msg)
    if not np.issubdtype(arr.dtype, np.integer):
        msg = f'{name} must have an integer dtype'
        raise ValueError(msg)
    return np.asarray(arr, dtype=np.int64)


def _load_and_validate_index(source_path: Path, *, n_traces: int) -> _IndexData:
    index_path = source_path / 'index.npz'
    if not index_path.exists():
        msg = f'source index.npz is missing: {index_path}'
        raise ValueError(msg)

    required = {'key1_values', 'key1_offsets', 'key1_counts', 'sorted_to_original'}
    try:
        with np.load(index_path, allow_pickle=False) as index:
            missing = sorted(required.difference(index.files))
            if missing:
                msg = f'source index.npz is missing required arrays: {", ".join(missing)}'
                raise ValueError(msg)
            key1_values = _ensure_index_i64('key1_values', index['key1_values'])
            key1_offsets = _ensure_index_i64('key1_offsets', index['key1_offsets'])
            key1_counts = _ensure_index_i64('key1_counts', index['key1_counts'])
            sorted_to_original = validate_sorted_to_original(
                index['sorted_to_original'],
                expected_n_traces=n_traces,
                role='source',
            )
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        msg = f'source index.npz could not be loaded: {index_path}'
        raise ValueError(msg) from exc

    if (
        key1_values.shape != key1_offsets.shape
        or key1_values.shape != key1_counts.shape
    ):
        msg = 'key1_values, key1_offsets, and key1_counts must have matching shapes'
        raise ValueError(msg)
    if key1_values.size == 0:
        msg = 'source index.npz must contain at least one key1 section'
        raise ValueError(msg)
    if np.any(key1_offsets < 0):
        msg = 'key1_offsets must not contain negative values'
        raise ValueError(msg)
    if np.any(key1_counts <= 0):
        msg = 'key1_counts must contain positive values'
        raise ValueError(msg)

    expected_offsets = np.empty_like(key1_offsets)
    expected_offsets[0] = 0
    if expected_offsets.size > 1:
        expected_offsets[1:] = key1_offsets[:-1] + key1_counts[:-1]
    if not np.array_equal(key1_offsets, expected_offsets):
        msg = 'source index section spans must be contiguous'
        raise ValueError(msg)
    last_end = int(key1_offsets[-1] + key1_counts[-1])
    if last_end != int(n_traces):
        msg = 'source index section spans must cover all traces'
        raise ValueError(msg)

    return _IndexData(
        key1_values=key1_values,
        key1_offsets=key1_offsets,
        key1_counts=key1_counts,
        sorted_to_original=sorted_to_original,
    )


def _validate_shifts(
    trace_shift_s_sorted: np.ndarray,
    *,
    n_traces: int,
) -> np.ndarray:
    shifts_raw = np.asarray(trace_shift_s_sorted)
    if shifts_raw.ndim != 1:
        msg = 'trace_shift_s_sorted must be 1-dimensional'
        raise ValueError(msg)
    if shifts_raw.shape != (int(n_traces),):
        msg = (
            'trace_shift_s_sorted shape mismatch: '
            f'expected {(int(n_traces),)}, got {shifts_raw.shape}'
        )
        raise ValueError(msg)
    try:
        shifts = np.asarray(shifts_raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = 'trace_shift_s_sorted must contain finite numeric values'
        raise ValueError(msg) from exc
    if not np.all(np.isfinite(shifts)):
        msg = 'trace_shift_s_sorted must contain only finite values'
        raise ValueError(msg)
    return np.ascontiguousarray(shifts, dtype=np.float64)


def _shift_summary(shifts: np.ndarray) -> dict[str, float]:
    return {
        'min_s': float(np.min(shifts)),
        'max_s': float(np.max(shifts)),
        'mean_s': float(np.mean(shifts)),
        'max_abs_s': float(np.max(np.abs(shifts))),
    }


def _validate_header_byte(value: Any) -> int:
    if isinstance(value, bool):
        msg = f'header byte must be an integer: {value!r}'
        raise ValueError(msg)
    try:
        header_byte = int(value)
    except (TypeError, ValueError) as exc:
        msg = f'header byte must be an integer: {value!r}'
        raise ValueError(msg) from exc
    if isinstance(value, float | np.floating) and float(value) != float(header_byte):
        msg = f'header byte must be an integer: {value!r}'
        raise ValueError(msg)
    if header_byte < 1 or header_byte > 240:
        msg = f'header byte must be between 1 and 240: {header_byte}'
        raise ValueError(msg)
    return header_byte


def _normalize_header_bytes(header_bytes_to_materialize: Iterable[int]) -> tuple[int, ...]:
    normalized: dict[int, None] = {}
    for value in header_bytes_to_materialize:
        normalized[_validate_header_byte(value)] = None
    return tuple(normalized)


def _header_byte_from_path(path: Path) -> int:
    match = _HEADER_FILENAME_RE.match(path.name)
    if match is None:
        msg = f'invalid header filename: {path.name}'
        raise ValueError(msg)
    return _validate_header_byte(match.group(1))


def _copy_source_headers(
    *,
    source_path: Path,
    tmp_store_path: Path,
    n_traces: int,
) -> None:
    for header_path in sorted(source_path.glob('headers_byte_*.npy')):
        header_byte = _header_byte_from_path(header_path)
        try:
            values = np.load(header_path, mmap_mode='r')
        except Exception as exc:  # noqa: BLE001
            msg = f'source header could not be loaded: {header_path}'
            raise ValueError(msg) from exc
        write_header_array_atomic(
            store_dir=tmp_store_path,
            header_byte=header_byte,
            values=values,
            expected_n_traces=n_traces,
        )


def _build_meta(
    *,
    source_path: Path,
    source_meta: _SourceMeta,
    output_dtype: str,
    fill_value: float,
    shift_summary: dict[str, float],
    derived_metadata: Mapping[str, Any] | None,
    from_file_id: str | None,
) -> dict[str, Any]:
    derived_extra = dict(derived_metadata or {})
    reserved = sorted(_RESERVED_DERIVED_KEYS.intersection(derived_extra))
    if reserved:
        msg = f'derived_metadata cannot override reserved keys: {", ".join(reserved)}'
        raise ValueError(msg)
    if from_file_id is not None and 'from_file_id' in derived_extra:
        msg = 'derived_metadata.from_file_id is only allowed when from_file_id is None'
        raise ValueError(msg)

    derived_from_file_id = (
        derived_extra.pop('from_file_id')
        if from_file_id is None and 'from_file_id' in derived_extra
        else from_file_id
    )
    derived = {
        'kind': 'time_shifted_trace_store',
        'from_store_path': str(source_path.resolve()),
        'header_source_store_path': str(source_path.resolve()),
        'from_file_id': derived_from_file_id,
        'sign_convention': 'corrected(t)=raw(t-shift_s); positive_shift_delays_events',
        'static_value_origin': 'internal_event_time_shift',
        'interpolation': 'linear',
        'fill_value': float(fill_value),
        'output_dtype': output_dtype,
        'applied_shift_summary': dict(shift_summary),
    }
    derived.update(derived_extra)

    return {
        'schema_version': 1,
        'dtype': output_dtype,
        'n_traces': source_meta.n_traces,
        'n_samples': source_meta.n_samples,
        'key_bytes': {
            'key1': source_meta.key1_byte,
            'key2': source_meta.key2_byte,
        },
        'sorted_by': ['key1', 'key2'],
        'dt': source_meta.dt,
        'original_segy_path': None,
        'source_sha256': None,
        'derived': derived,
    }


def _copy_index(source_path: Path, tmp_store_path: Path) -> None:
    shutil.copy2(source_path / 'index.npz', tmp_store_path / 'index.npz')


def _write_shifted_traces(
    *,
    reader: TraceStoreSectionReader,
    source_traces: np.ndarray,
    tmp_store_path: Path,
    shifts: np.ndarray,
    dt: float,
    fill_value: float,
    chunk_size: int,
    progress_callback: Callable[[float, str], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> tuple[np.ndarray, np.ndarray]:
    n_traces, n_samples = (int(source_traces.shape[0]), int(source_traces.shape[1]))
    trace_sum = np.zeros(n_traces, dtype=np.float64)
    trace_sumsq = np.zeros(n_traces, dtype=np.float64)
    out = np.lib.format.open_memmap(
        tmp_store_path / 'traces.npy',
        mode='w+',
        dtype=np.float32,
        shape=(n_traces, n_samples),
    )

    for start in range(0, n_traces, chunk_size):
        _raise_if_cancelled(cancel_check)
        stop = min(start + chunk_size, n_traces)
        chunk = source_traces[start:stop]
        chunk_f32 = coerce_section_f32(chunk, reader.scale)
        if not np.all(np.isfinite(chunk_f32)):
            msg = 'source chunk contains NaN or Inf values'
            raise ValueError(msg)

        corrected = shift_traces_linear(
            chunk_f32,
            shifts[start:stop],
            dt,
            fill_value=fill_value,
        )
        if not np.all(np.isfinite(corrected)):
            msg = 'corrected chunk contains NaN or Inf values'
            raise ValueError(msg)

        out[start:stop] = corrected
        trace_sum[start:stop] = corrected.sum(axis=1, dtype=np.float64)
        trace_sumsq[start:stop] = np.einsum(
            'ij,ij->i',
            corrected,
            corrected,
            dtype=np.float64,
        )
        _raise_if_cancelled(cancel_check)

        progress = 0.2 + (0.65 * (float(stop) / float(n_traces)))
        _notify(progress_callback, progress, 'shifting_traces')

    out.flush()
    del out
    return trace_sum, trace_sumsq


def build_time_shifted_trace_store(
    *,
    source_store_path: str | Path,
    output_store_path: str | Path,
    trace_shift_s_sorted: np.ndarray,
    fill_value: float = 0.0,
    output_dtype: str = 'float32',
    derived_metadata: Mapping[str, Any] | None = None,
    from_file_id: str | None = None,
    header_bytes_to_materialize: Iterable[int] = (),
    chunk_size: int = 512,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> TimeShiftedTraceStoreResult:
    """Build an atomic float32 TraceStore from sorted per-trace event-time shifts."""
    _notify(progress_callback, 0.0, 'validating')
    source_path, output_path = _validate_source_paths(
        source_store_path=source_store_path,
        output_store_path=output_store_path,
    )
    if output_dtype != 'float32':
        msg = 'output_dtype must be "float32"'
        raise ValueError(msg)
    fill = coerce_finite_float32_fill_value(fill_value)
    if isinstance(chunk_size, bool):
        msg = 'chunk_size must be positive'
        raise ValueError(msg)
    try:
        chunk_size_int = int(chunk_size)
    except (TypeError, ValueError) as exc:
        msg = 'chunk_size must be positive'
        raise ValueError(msg) from exc
    if chunk_size_int != chunk_size or chunk_size_int <= 0:
        msg = 'chunk_size must be positive'
        raise ValueError(msg)

    source_traces = _load_source_traces(source_path)
    source_meta = _load_source_meta(
        source_path,
        traces_shape=(int(source_traces.shape[0]), int(source_traces.shape[1])),
    )
    index_data = _load_and_validate_index(source_path, n_traces=source_meta.n_traces)
    shifts = _validate_shifts(
        trace_shift_s_sorted,
        n_traces=source_meta.n_traces,
    )
    shift_summary = _shift_summary(shifts)
    normalized_header_bytes = _normalize_header_bytes(header_bytes_to_materialize)
    meta = _build_meta(
        source_path=source_path,
        source_meta=source_meta,
        output_dtype=output_dtype,
        fill_value=fill,
        shift_summary=shift_summary,
        derived_metadata=derived_metadata,
        from_file_id=from_file_id,
    )
    _raise_if_cancelled(cancel_check)

    reader = TraceStoreSectionReader(
        source_path,
        key1_byte=source_meta.key1_byte,
        key2_byte=source_meta.key2_byte,
    )
    _notify(progress_callback, 0.1, 'materializing_headers')
    _raise_if_cancelled(cancel_check)
    for header_byte in normalized_header_bytes:
        reader.ensure_header(header_byte)
    _raise_if_cancelled(cancel_check)

    tmp_store_path = output_path.with_name(f'{output_path.name}.tmp-{uuid4().hex}')
    try:
        tmp_store_path.mkdir()
        _notify(progress_callback, 0.2, 'shifting_traces')
        trace_sum, trace_sumsq = _write_shifted_traces(
            reader=reader,
            source_traces=source_traces,
            tmp_store_path=tmp_store_path,
            shifts=shifts,
            dt=source_meta.dt,
            fill_value=fill,
            chunk_size=chunk_size_int,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

        _copy_index(source_path, tmp_store_path)
        _copy_source_headers(
            source_path=source_path,
            tmp_store_path=tmp_store_path,
            n_traces=source_meta.n_traces,
        )
        (tmp_store_path / 'meta.json').write_text(
            json.dumps(meta),
            encoding='utf-8',
        )

        _notify(progress_callback, 0.9, 'writing_baselines')
        _raise_if_cancelled(cancel_check)
        write_trace_store_raw_baseline_artifacts(
            store_path=tmp_store_path,
            key1_byte=source_meta.key1_byte,
            key2_byte=source_meta.key2_byte,
            dtype_base='float32',
            dt=source_meta.dt,
            key1_values=index_data.key1_values,
            key1_offsets=index_data.key1_offsets,
            key1_counts=index_data.key1_counts,
            trace_sum=trace_sum,
            trace_sumsq=trace_sumsq,
            n_samples=source_meta.n_samples,
            source_sha256=None,
        )

        _notify(progress_callback, 0.98, 'finalizing')
        _raise_if_cancelled(cancel_check)
        if output_path.exists() or output_path.is_symlink():
            msg = f'output_store_path already exists: {output_path}'
            raise ValueError(msg)
        tmp_store_path.rename(output_path)
    except Exception:
        shutil.rmtree(tmp_store_path, ignore_errors=True)
        raise

    return TimeShiftedTraceStoreResult(
        store_path=output_path,
        source_store_path=source_path,
        n_traces=source_meta.n_traces,
        n_samples=source_meta.n_samples,
        dt=source_meta.dt,
        key1_byte=source_meta.key1_byte,
        key2_byte=source_meta.key2_byte,
        output_dtype=output_dtype,
        shift_summary=shift_summary,
    )


__all__ = [
    'TimeShiftedTraceStoreResult',
    'build_time_shifted_trace_store',
]
