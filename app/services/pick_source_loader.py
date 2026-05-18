"""First-break pick source loading and sorted-order normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from app.core.state import AppState
from app.services.reader import get_reader
from app.services.trace_store_index_validation import validate_sorted_to_original
from app.trace_store.reader import TraceStoreSectionReader
from app.utils.pick_cache_file1d_mem import path_for_file

NpzPickSourceKind = Literal['batch_npz', 'manual_npz']
PickSourceKind = Literal['batch_npz', 'manual_npz', 'manual_memmap']


@dataclass(frozen=True)
class LoadedPickSource:
    picks_time_s_sorted: np.ndarray
    valid_mask_sorted: np.ndarray
    source_kind: PickSourceKind
    n_traces: int
    n_samples: int
    dt: float
    n_valid: int
    n_nan: int
    metadata: dict[str, object]


def _coerce_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = f'{name} must be an integer'
        raise ValueError(msg)
    out = int(value)
    if out < 0:
        msg = f'{name} must be non-negative'
        raise ValueError(msg)
    return out


def _coerce_positive_int(value: object, *, name: str) -> int:
    out = _coerce_nonnegative_int(value, name=name)
    if out <= 0:
        msg = f'{name} must be greater than 0'
        raise ValueError(msg)
    return out


def _coerce_positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        msg = f'{name} must be a positive finite float'
        raise ValueError(msg)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        msg = f'{name} must be a positive finite float'
        raise ValueError(msg) from exc
    if not np.isfinite(out) or out <= 0.0:
        msg = f'{name} must be a positive finite float'
        raise ValueError(msg)
    return out


def _reader_n_traces(reader: TraceStoreSectionReader) -> int:
    n_traces = getattr(reader, 'ntraces', None)
    if n_traces is None:
        meta = getattr(reader, 'meta', None)
        if isinstance(meta, dict):
            n_traces = meta.get('n_traces')
    if n_traces is None and hasattr(reader, 'traces'):
        n_traces = getattr(reader.traces, 'shape', (None,))[0]
    if n_traces is None:
        msg = 'reader cannot provide number of traces'
        raise ValueError(msg)
    return _coerce_nonnegative_int(n_traces, name='reader n_traces')


def _reader_n_samples(reader: TraceStoreSectionReader) -> int:
    getter = getattr(reader, 'get_n_samples', None)
    if callable(getter):
        return _coerce_positive_int(getter(), name='reader n_samples')
    if hasattr(reader, 'traces'):
        shape = getattr(reader.traces, 'shape', ())
        if len(shape) >= 2:
            return _coerce_positive_int(shape[-1], name='reader n_samples')
    msg = 'reader cannot provide number of samples'
    raise ValueError(msg)


def _read_int_scalar(npz: np.lib.npyio.NpzFile, key: str) -> int:
    arr = np.asarray(npz[key])
    if arr.size != 1:
        msg = f'{key} must be a scalar'
        raise ValueError(msg)
    if not np.issubdtype(arr.dtype, np.integer):
        msg = f'{key} must have an integer dtype'
        raise ValueError(msg)
    return int(arr.reshape(-1)[0])


def _read_float_scalar(npz: np.lib.npyio.NpzFile, key: str) -> float:
    arr = np.asarray(npz[key])
    if arr.size != 1:
        msg = f'{key} must be a scalar'
        raise ValueError(msg)
    if not np.issubdtype(arr.dtype, np.number):
        msg = f'{key} must be numeric'
        raise ValueError(msg)
    return float(arr.reshape(-1)[0])


def _validate_npz_metadata(
    npz: np.lib.npyio.NpzFile,
    *,
    n_traces: int,
    n_samples: int,
    dt: float,
) -> dict[str, object]:
    metadata: dict[str, object] = {'npz_keys': tuple(npz.files)}
    if 'n_traces' in npz.files:
        npz_n_traces = _read_int_scalar(npz, 'n_traces')
        metadata['n_traces'] = npz_n_traces
        if npz_n_traces != n_traces:
            msg = f'n_traces mismatch: expected {n_traces}, got {npz_n_traces}'
            raise ValueError(msg)
    if 'n_samples' in npz.files:
        npz_n_samples = _read_int_scalar(npz, 'n_samples')
        metadata['n_samples'] = npz_n_samples
        if npz_n_samples != n_samples:
            msg = f'n_samples mismatch: expected {n_samples}, got {npz_n_samples}'
            raise ValueError(msg)
    if 'dt' in npz.files:
        npz_dt = _read_float_scalar(npz, 'dt')
        metadata['dt'] = npz_dt
        if not np.isfinite(npz_dt) or abs(npz_dt - dt) > 1e-9:
            msg = f'dt mismatch: expected {dt}, got {npz_dt}'
            raise ValueError(msg)
    return metadata


def _reader_sorted_to_original(
    reader: TraceStoreSectionReader,
    *,
    n_traces: int,
) -> np.ndarray:
    values = reader.get_sorted_to_original()
    return validate_sorted_to_original(
        np.asarray(values),
        expected_n_traces=n_traces,
        role='reader',
    )


def _build_loaded_pick_source(
    picks_time_s_sorted: np.ndarray,
    *,
    source_kind: PickSourceKind,
    n_traces: int,
    n_samples: int,
    dt: float,
    metadata: dict[str, object],
) -> LoadedPickSource:
    picks, valid_mask = validate_picks_time_s(
        picks_time_s_sorted,
        n_traces=n_traces,
        n_samples=n_samples,
        dt=dt,
    )
    return LoadedPickSource(
        picks_time_s_sorted=picks,
        valid_mask_sorted=valid_mask,
        source_kind=source_kind,
        n_traces=n_traces,
        n_samples=n_samples,
        dt=dt,
        n_valid=int(np.count_nonzero(valid_mask)),
        n_nan=int(np.count_nonzero(np.isnan(picks))),
        metadata=metadata,
    )


def normalize_picks_original_to_sorted(
    picks_time_s_original: np.ndarray,
    *,
    sorted_to_original: np.ndarray,
) -> np.ndarray:
    """Return picks reindexed from original trace order to sorted trace order."""
    picks = np.asarray(picks_time_s_original)
    if picks.ndim != 1:
        msg = 'picks_time_s_original must be 1D'
        raise ValueError(msg)
    order = validate_sorted_to_original(
        np.asarray(sorted_to_original),
        expected_n_traces=int(picks.shape[0]),
        role='sorted_to_original',
    )
    return np.ascontiguousarray(picks[order])


def validate_picks_time_s(
    picks_time_s: np.ndarray,
    *,
    n_traces: int,
    n_samples: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate pick times and return a float64 copy plus finite-value mask."""
    expected_n_traces = _coerce_nonnegative_int(n_traces, name='n_traces')
    expected_n_samples = _coerce_positive_int(n_samples, name='n_samples')
    sample_interval = _coerce_positive_float(dt, name='dt')

    arr = np.asarray(picks_time_s)
    if arr.ndim != 1:
        msg = 'picks_time_s must be 1D'
        raise ValueError(msg)
    if arr.shape != (expected_n_traces,):
        msg = (
            'picks_time_s shape mismatch: '
            f'expected {(expected_n_traces,)}, got {arr.shape}'
        )
        raise ValueError(msg)
    if not np.issubdtype(arr.dtype, np.floating):
        msg = 'picks_time_s must have a floating dtype'
        raise ValueError(msg)

    out = np.ascontiguousarray(arr, dtype=np.float64)
    if np.any(np.isinf(out)):
        msg = 'picks_time_s contains inf'
        raise ValueError(msg)

    valid_mask = np.isfinite(out)
    finite = out[valid_mask]
    if finite.size:
        if np.any(finite < 0.0):
            msg = 'picks_time_s contains negative pick times'
            raise ValueError(msg)
        max_time = float(expected_n_samples - 1) * sample_interval
        if np.any(finite > max_time + 1e-9):
            msg = 'picks_time_s contains pick times beyond n_samples'
            raise ValueError(msg)
    return out, valid_mask


def load_npz_pick_source(
    npz_path: Path,
    *,
    reader: TraceStoreSectionReader,
    expected_dt: float,
    expected_n_samples: int,
    source_kind: NpzPickSourceKind,
) -> LoadedPickSource:
    """Load a batch/manual NPZ pick source and normalize it to sorted order."""
    if source_kind not in ('batch_npz', 'manual_npz'):
        msg = "source_kind must be 'batch_npz' or 'manual_npz'"
        raise ValueError(msg)

    path = Path(npz_path)
    n_traces = _reader_n_traces(reader)
    n_samples = _coerce_positive_int(expected_n_samples, name='expected_n_samples')
    reader_n_samples = _reader_n_samples(reader)
    if reader_n_samples != n_samples:
        msg = f'expected_n_samples mismatch: reader has {reader_n_samples}, got {n_samples}'
        raise ValueError(msg)
    dt = _coerce_positive_float(expected_dt, name='expected_dt')
    reader_sorted_to_original = _reader_sorted_to_original(
        reader,
        n_traces=n_traces,
    )

    try:
        npz_file = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        msg = f'Could not read npz pick source: {path}'
        raise ValueError(msg) from exc

    with npz_file as npz:
        if 'picks_time_s' not in npz.files:
            msg = 'Missing key: picks_time_s'
            raise ValueError(msg)

        metadata = _validate_npz_metadata(
            npz,
            n_traces=n_traces,
            n_samples=n_samples,
            dt=dt,
        )
        metadata['npz_path'] = str(path)
        metadata['has_sorted_to_original'] = 'sorted_to_original' in npz.files

        if 'sorted_to_original' in npz.files:
            npz_sorted_to_original = validate_sorted_to_original(
                np.asarray(npz['sorted_to_original']),
                expected_n_traces=n_traces,
                role='npz',
            )
            if not np.array_equal(npz_sorted_to_original, reader_sorted_to_original):
                msg = 'sorted_to_original mismatch'
                raise ValueError(msg)

        picks_time_s_original, _ = validate_picks_time_s(
            np.asarray(npz['picks_time_s']),
            n_traces=n_traces,
            n_samples=n_samples,
            dt=dt,
        )

    picks_time_s_sorted = normalize_picks_original_to_sorted(
        picks_time_s_original,
        sorted_to_original=reader_sorted_to_original,
    )
    return _build_loaded_pick_source(
        picks_time_s_sorted,
        source_kind=source_kind,
        n_traces=n_traces,
        n_samples=n_samples,
        dt=dt,
        metadata=metadata,
    )


def load_manual_memmap_pick_source(
    *,
    file_id: str,
    key1_byte: int,
    key2_byte: int,
    state: AppState,
) -> LoadedPickSource:
    """Load manual picks from the sorted-order memmap without reindexing."""
    file_name = state.file_registry.filename(file_id)
    if not file_name:
        msg = f'manual memmap file_id could not be resolved: {file_id}'
        raise ValueError(msg)

    try:
        reader = get_reader(file_id, key1_byte, key2_byte, state=state)
    except Exception as exc:  # noqa: BLE001
        msg = f'manual memmap reader could not be resolved: {file_id}'
        raise ValueError(msg) from exc

    n_traces = _reader_n_traces(reader)
    n_samples = _reader_n_samples(reader)
    dt = _coerce_positive_float(state.file_registry.get_dt(file_id), name='dt')
    memmap_path = path_for_file(file_name)
    if not memmap_path.is_file():
        msg = f'manual pick memmap not found: {memmap_path}'
        raise ValueError(msg)

    try:
        memmap = np.load(memmap_path, mmap_mode='r', allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        msg = f'Could not read manual pick memmap: {memmap_path}'
        raise ValueError(msg) from exc

    try:
        picks_time_s_sorted = np.asarray(memmap)
        return _build_loaded_pick_source(
            picks_time_s_sorted,
            source_kind='manual_memmap',
            n_traces=n_traces,
            n_samples=n_samples,
            dt=dt,
            metadata={
                'file_id': file_id,
                'file_name': file_name,
                'memmap_path': str(memmap_path),
                'key1_byte': int(key1_byte),
                'key2_byte': int(key2_byte),
            },
        )
    finally:
        del memmap


__all__ = [
    'LoadedPickSource',
    'NpzPickSourceKind',
    'PickSourceKind',
    'load_manual_memmap_pick_source',
    'load_npz_pick_source',
    'normalize_picks_original_to_sorted',
    'validate_picks_time_s',
]
