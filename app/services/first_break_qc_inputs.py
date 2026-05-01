"""Validated inputs for first-break QC after datum static correction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.pick_source_loader import LoadedPickSource
from app.trace_store.reader import TraceStoreSectionReader


_REQUIRED_SOLUTION_KEYS = frozenset(
    {
        'trace_shift_s_sorted',
        'source_shift_s_sorted',
        'receiver_shift_s_sorted',
        'source_elevation_m_sorted',
        'receiver_elevation_m_sorted',
        'key1_sorted',
        'key2_sorted',
        'datum_elevation_m',
        'replacement_velocity_m_s',
        'dt',
        'n_traces',
        'key1_byte',
        'key2_byte',
        'source_elevation_byte',
        'receiver_elevation_byte',
    }
)

_OPTIONAL_METADATA_KEYS = (
    'source_surface_elevation_m_sorted',
    'source_depth_m_sorted',
    'source_depth_used_sorted',
    'elevation_scalar_byte',
    'source_depth_byte',
    'source_depth_enabled',
    'elevation_unit',
    'elevation_scalar_zero_count',
    'header_source_segy_path',
)

_DT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class DatumStaticSolution:
    trace_shift_s_sorted: np.ndarray
    source_shift_s_sorted: np.ndarray
    receiver_shift_s_sorted: np.ndarray
    source_elevation_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray
    key1_sorted: np.ndarray
    key2_sorted: np.ndarray
    datum_elevation_m: float
    replacement_velocity_m_s: float
    dt: float
    n_traces: int
    key1_byte: int
    key2_byte: int
    source_elevation_byte: int
    receiver_elevation_byte: int
    metadata: dict[str, object]


@dataclass(frozen=True)
class FirstBreakQcInputs:
    picks_time_s_sorted: np.ndarray
    valid_pick_mask_sorted: np.ndarray
    datum_trace_shift_s_sorted: np.ndarray
    source_elevation_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray
    offset_sorted: np.ndarray
    key1_sorted: np.ndarray
    key2_sorted: np.ndarray
    dt: float
    n_traces: int
    n_samples: int
    offset_byte: int
    source_kind: str
    metadata: dict[str, Any]


def load_datum_static_solution_npz(
    solution_npz_path: Path,
    *,
    expected_n_traces: int,
    expected_dt: float,
    expected_key1_byte: int,
    expected_key2_byte: int,
) -> DatumStaticSolution:
    """Load and validate a PR2 ``datum_static_solution.npz`` artifact."""
    path = Path(solution_npz_path)
    if not path.exists():
        msg = f'datum static solution npz not found: {path}'
        raise ValueError(msg)
    if not path.is_file():
        msg = f'datum static solution npz is not a file: {path}'
        raise ValueError(msg)

    n_traces = _coerce_positive_int(expected_n_traces, name='expected_n_traces')
    dt = _coerce_positive_finite_float(expected_dt, name='expected_dt')
    key1_byte = _validate_header_byte(
        expected_key1_byte,
        name='expected_key1_byte',
    )
    key2_byte = _validate_header_byte(
        expected_key2_byte,
        name='expected_key2_byte',
    )

    try:
        npz_file = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        msg = f'Could not read datum static solution npz: {path}'
        raise ValueError(msg) from exc

    with npz_file as npz:
        missing = sorted(_REQUIRED_SOLUTION_KEYS.difference(npz.files))
        if missing:
            msg = f'Missing required datum static solution key: {missing[0]}'
            raise ValueError(msg)

        expected_shape = (n_traces,)
        trace_shift = _coerce_1d_finite_float64(
            npz['trace_shift_s_sorted'],
            name='trace_shift_s_sorted',
            expected_shape=expected_shape,
        )
        source_shift = _coerce_1d_finite_float64(
            npz['source_shift_s_sorted'],
            name='source_shift_s_sorted',
            expected_shape=expected_shape,
        )
        receiver_shift = _coerce_1d_finite_float64(
            npz['receiver_shift_s_sorted'],
            name='receiver_shift_s_sorted',
            expected_shape=expected_shape,
        )
        if not np.allclose(trace_shift, source_shift + receiver_shift):
            msg = (
                'trace_shift_s_sorted must match '
                'source_shift_s_sorted + receiver_shift_s_sorted'
            )
            raise ValueError(msg)

        source_elevation = _coerce_1d_finite_float64(
            npz['source_elevation_m_sorted'],
            name='source_elevation_m_sorted',
            expected_shape=expected_shape,
        )
        receiver_elevation = _coerce_1d_finite_float64(
            npz['receiver_elevation_m_sorted'],
            name='receiver_elevation_m_sorted',
            expected_shape=expected_shape,
        )
        key1 = _coerce_1d_integer_int64(
            npz['key1_sorted'],
            name='key1_sorted',
            expected_shape=expected_shape,
        )
        key2 = _coerce_1d_integer_int64(
            npz['key2_sorted'],
            name='key2_sorted',
            expected_shape=expected_shape,
        )

        solution_n_traces = _read_int_scalar(npz, 'n_traces')
        if solution_n_traces != n_traces:
            msg = f'n_traces mismatch: expected {n_traces}, got {solution_n_traces}'
            raise ValueError(msg)

        solution_dt = _coerce_positive_finite_float(
            _read_float_scalar(npz, 'dt'),
            name='datum_static_solution.npz dt',
        )
        if abs(solution_dt - dt) > _DT_TOLERANCE:
            msg = f'dt mismatch: expected {dt}, got {solution_dt}'
            raise ValueError(msg)

        solution_key1_byte = _read_int_scalar(npz, 'key1_byte')
        if solution_key1_byte != key1_byte:
            msg = f'key1_byte mismatch: expected {key1_byte}, got {solution_key1_byte}'
            raise ValueError(msg)
        solution_key2_byte = _read_int_scalar(npz, 'key2_byte')
        if solution_key2_byte != key2_byte:
            msg = f'key2_byte mismatch: expected {key2_byte}, got {solution_key2_byte}'
            raise ValueError(msg)

        source_elevation_byte = _validate_header_byte(
            _read_int_scalar(npz, 'source_elevation_byte'),
            name='source_elevation_byte',
        )
        receiver_elevation_byte = _validate_header_byte(
            _read_int_scalar(npz, 'receiver_elevation_byte'),
            name='receiver_elevation_byte',
        )
        datum_elevation = _coerce_finite_float(
            _read_float_scalar(npz, 'datum_elevation_m'),
            name='datum_elevation_m',
        )
        replacement_velocity = _coerce_positive_finite_float(
            _read_float_scalar(npz, 'replacement_velocity_m_s'),
            name='replacement_velocity_m_s',
        )
        metadata = _solution_metadata(npz=npz, path=path)

    return DatumStaticSolution(
        trace_shift_s_sorted=trace_shift,
        source_shift_s_sorted=source_shift,
        receiver_shift_s_sorted=receiver_shift,
        source_elevation_m_sorted=source_elevation,
        receiver_elevation_m_sorted=receiver_elevation,
        key1_sorted=key1,
        key2_sorted=key2,
        datum_elevation_m=datum_elevation,
        replacement_velocity_m_s=replacement_velocity,
        dt=solution_dt,
        n_traces=solution_n_traces,
        key1_byte=solution_key1_byte,
        key2_byte=solution_key2_byte,
        source_elevation_byte=source_elevation_byte,
        receiver_elevation_byte=receiver_elevation_byte,
        metadata=metadata,
    )


def load_offset_header_sorted(
    reader: TraceStoreSectionReader,
    *,
    offset_byte: int,
    expected_n_traces: int,
) -> np.ndarray:
    """Read a signed offset header in TraceStore sorted order."""
    byte = _validate_header_byte(offset_byte, name='offset_byte')
    n_traces = _coerce_positive_int(expected_n_traces, name='expected_n_traces')
    values = _read_reader_header(reader, byte=byte, role='offset')
    return _coerce_1d_finite_float64(
        values,
        name=f'offset header byte {byte}',
        expected_shape=(n_traces,),
    )


def build_first_break_qc_inputs(
    *,
    pick_source: LoadedPickSource,
    solution_npz_path: Path,
    reader: TraceStoreSectionReader,
    offset_byte: int = 37,
    expected_dt: float,
    expected_n_samples: int,
    expected_key1_byte: int,
    expected_key2_byte: int,
) -> FirstBreakQcInputs:
    """Build the sorted-order validated input object for first-break QC."""
    dt = _coerce_positive_finite_float(expected_dt, name='expected_dt')
    n_samples = _coerce_positive_int(
        expected_n_samples,
        name='expected_n_samples',
    )
    key1_byte = _validate_header_byte(
        expected_key1_byte,
        name='expected_key1_byte',
    )
    key2_byte = _validate_header_byte(
        expected_key2_byte,
        name='expected_key2_byte',
    )
    _validate_reader_key_bytes(
        reader,
        expected_key1_byte=key1_byte,
        expected_key2_byte=key2_byte,
    )
    _validate_reader_dt(reader, expected_dt=dt)

    n_traces = _reader_n_traces(reader)
    reader_n_samples = _reader_n_samples(reader)
    if reader_n_samples != n_samples:
        msg = f'n_samples mismatch: expected {n_samples}, reader has {reader_n_samples}'
        raise ValueError(msg)

    solution = load_datum_static_solution_npz(
        Path(solution_npz_path),
        expected_n_traces=n_traces,
        expected_dt=dt,
        expected_key1_byte=key1_byte,
        expected_key2_byte=key2_byte,
    )

    _validate_pick_source(
        pick_source,
        expected_n_traces=n_traces,
        expected_n_samples=n_samples,
        expected_dt=dt,
    )
    picks = _coerce_1d_pick_times(
        pick_source.picks_time_s_sorted,
        valid_mask=pick_source.valid_mask_sorted,
        expected_n_traces=n_traces,
        n_samples=n_samples,
        dt=dt,
    )
    valid_mask = _coerce_1d_bool_array(
        pick_source.valid_mask_sorted,
        name='valid_mask_sorted',
        expected_shape=(n_traces,),
    )

    key1_header = _coerce_1d_integer_int64(
        _read_reader_header(reader, byte=key1_byte, role='key1'),
        name=f'reader header byte {key1_byte}',
        expected_shape=(n_traces,),
    )
    key2_header = _coerce_1d_integer_int64(
        _read_reader_header(reader, byte=key2_byte, role='key2'),
        name=f'reader header byte {key2_byte}',
        expected_shape=(n_traces,),
    )
    if not np.array_equal(solution.key1_sorted, key1_header):
        msg = f'solution key1_sorted does not match reader header byte {key1_byte}'
        raise ValueError(msg)
    if not np.array_equal(solution.key2_sorted, key2_header):
        msg = f'solution key2_sorted does not match reader header byte {key2_byte}'
        raise ValueError(msg)

    offset = load_offset_header_sorted(
        reader,
        offset_byte=offset_byte,
        expected_n_traces=n_traces,
    )

    source_kind = _pick_source_kind(pick_source)
    pick_metadata = _pick_source_metadata(pick_source)
    solution_path = Path(solution_npz_path)
    metadata: dict[str, Any] = {
        'solution_artifact': solution_path.name,
        'solution_npz_path': str(solution_path),
        'datum_elevation_m': solution.datum_elevation_m,
        'replacement_velocity_m_s': solution.replacement_velocity_m_s,
        'key1_byte': key1_byte,
        'key2_byte': key2_byte,
        'offset_byte': int(offset_byte),
        'sign_convention': (
            'pick_time_after_datum_s = pick_time_raw_s + datum_trace_shift_s'
        ),
        'order': 'trace_store_sorted',
        'pick_source_metadata': pick_metadata,
        'datum_static_solution_metadata': solution.metadata,
    }

    return FirstBreakQcInputs(
        picks_time_s_sorted=picks,
        valid_pick_mask_sorted=valid_mask,
        datum_trace_shift_s_sorted=solution.trace_shift_s_sorted,
        source_elevation_m_sorted=solution.source_elevation_m_sorted,
        receiver_elevation_m_sorted=solution.receiver_elevation_m_sorted,
        offset_sorted=offset,
        key1_sorted=solution.key1_sorted,
        key2_sorted=solution.key2_sorted,
        dt=dt,
        n_traces=n_traces,
        n_samples=n_samples,
        offset_byte=int(offset_byte),
        source_kind=source_kind,
        metadata=metadata,
    )


def _solution_metadata(
    *,
    npz: np.lib.npyio.NpzFile,
    path: Path,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        'npz_path': str(path),
        'npz_keys': tuple(npz.files),
    }
    for key in _OPTIONAL_METADATA_KEYS:
        if key not in npz.files:
            continue
        arr = np.asarray(npz[key])
        if key == 'source_depth_byte':
            value = _read_int_scalar(npz, key)
            if value != -1:
                _validate_header_byte(value, name=key)
            metadata[key] = value
            continue
        if arr.size == 1:
            metadata[key] = arr.reshape(-1)[0].item()
        else:
            metadata[key] = np.ascontiguousarray(arr)
    return metadata


def _read_reader_header(
    reader: TraceStoreSectionReader,
    *,
    byte: int,
    role: str,
) -> np.ndarray:
    get_header = getattr(reader, 'get_header', None)
    ensure_header = getattr(reader, 'ensure_header', None)
    reader_header = get_header if callable(get_header) else ensure_header
    if not callable(reader_header):
        msg = f'reader cannot read {role} header byte {byte}'
        raise ValueError(msg)
    try:
        return reader_header(byte)
    except Exception as exc:  # noqa: BLE001
        msg = f'failed to read {role} header byte {byte}: {exc}'
        raise ValueError(msg) from exc


def _validate_reader_key_bytes(
    reader: TraceStoreSectionReader,
    *,
    expected_key1_byte: int,
    expected_key2_byte: int,
) -> None:
    if not hasattr(reader, 'key1_byte'):
        msg = 'reader key1_byte is required'
        raise ValueError(msg)
    if not hasattr(reader, 'key2_byte'):
        msg = 'reader key2_byte is required'
        raise ValueError(msg)
    reader_key1 = _validate_header_byte(reader.key1_byte, name='reader key1_byte')
    reader_key2 = _validate_header_byte(reader.key2_byte, name='reader key2_byte')
    if reader_key1 != expected_key1_byte:
        msg = (
            f'reader key1_byte mismatch: expected {expected_key1_byte}, '
            f'got {reader_key1}'
        )
        raise ValueError(msg)
    if reader_key2 != expected_key2_byte:
        msg = (
            f'reader key2_byte mismatch: expected {expected_key2_byte}, '
            f'got {reader_key2}'
        )
        raise ValueError(msg)


def _validate_reader_dt(
    reader: TraceStoreSectionReader,
    *,
    expected_dt: float,
) -> None:
    meta = getattr(reader, 'meta', None)
    if not isinstance(meta, Mapping):
        msg = 'reader meta must be a mapping containing dt'
        raise ValueError(msg)
    if 'dt' not in meta:
        msg = 'reader meta missing dt'
        raise ValueError(msg)
    reader_dt = _coerce_positive_finite_float(meta['dt'], name='reader dt')
    if abs(reader_dt - expected_dt) > _DT_TOLERANCE:
        msg = f'reader dt mismatch: expected {expected_dt}, got {reader_dt}'
        raise ValueError(msg)


def _validate_pick_source(
    pick_source: LoadedPickSource,
    *,
    expected_n_traces: int,
    expected_n_samples: int,
    expected_dt: float,
) -> None:
    pick_n_traces = _coerce_nonnegative_int(
        getattr(pick_source, 'n_traces', None),
        name='pick source n_traces',
    )
    if pick_n_traces != expected_n_traces:
        msg = f'pick source n_traces mismatch: expected {expected_n_traces}, got {pick_n_traces}'
        raise ValueError(msg)

    pick_n_samples = _coerce_positive_int(
        getattr(pick_source, 'n_samples', None),
        name='pick source n_samples',
    )
    if pick_n_samples != expected_n_samples:
        msg = f'pick source n_samples mismatch: expected {expected_n_samples}, got {pick_n_samples}'
        raise ValueError(msg)

    pick_dt = _coerce_positive_finite_float(
        getattr(pick_source, 'dt', None),
        name='pick source dt',
    )
    if abs(pick_dt - expected_dt) > _DT_TOLERANCE:
        msg = f'pick source dt mismatch: expected {expected_dt}, got {pick_dt}'
        raise ValueError(msg)

    _pick_source_kind(pick_source)
    _pick_source_metadata(pick_source)


def _coerce_1d_pick_times(
    values: np.ndarray,
    *,
    valid_mask: np.ndarray,
    expected_n_traces: int,
    n_samples: int,
    dt: float,
) -> np.ndarray:
    expected_shape = (expected_n_traces,)
    picks = _coerce_1d_real_numeric_float64(
        values,
        name='picks_time_s_sorted',
        expected_shape=expected_shape,
    )
    mask = _coerce_1d_bool_array(
        valid_mask,
        name='valid_mask_sorted',
        expected_shape=expected_shape,
    )
    if np.any(np.isinf(picks)):
        msg = 'picks_time_s_sorted contains inf'
        raise ValueError(msg)
    if np.any(~np.isfinite(picks[mask])):
        msg = 'valid picks must be finite'
        raise ValueError(msg)
    if np.any(~np.isnan(picks[~mask])):
        msg = 'invalid picks must be NaN'
        raise ValueError(msg)
    finite = picks[mask]
    if finite.size:
        if np.any(finite < 0.0):
            msg = 'valid picks must be non-negative'
            raise ValueError(msg)
        max_time = float(n_samples - 1) * dt
        if np.any(finite > max_time + _DT_TOLERANCE):
            msg = 'valid picks must not exceed the trace sample range'
            raise ValueError(msg)
    return picks


def _pick_source_kind(pick_source: LoadedPickSource) -> str:
    source_kind = getattr(pick_source, 'source_kind', None)
    if not isinstance(source_kind, str) or not source_kind:
        msg = 'pick source source_kind must be a non-empty string'
        raise ValueError(msg)
    return source_kind


def _pick_source_metadata(pick_source: LoadedPickSource) -> dict[str, object]:
    metadata = getattr(pick_source, 'metadata', None)
    if not isinstance(metadata, Mapping):
        msg = 'pick source metadata must be a mapping'
        raise ValueError(msg)
    return dict(metadata)


def _reader_n_traces(reader: TraceStoreSectionReader) -> int:
    if hasattr(reader, 'traces'):
        shape = getattr(reader.traces, 'shape', ())
        if shape:
            return _coerce_positive_int(shape[0], name='reader n_traces')
    meta = getattr(reader, 'meta', None)
    if isinstance(meta, Mapping) and 'n_traces' in meta:
        return _coerce_positive_int(meta['n_traces'], name='reader n_traces')
    msg = 'reader cannot provide number of traces'
    raise ValueError(msg)


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


def _coerce_1d_finite_float64(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = _coerce_1d_real_numeric_float64(
        values,
        name=name,
        expected_shape=expected_shape,
    )
    if not np.all(np.isfinite(arr)):
        msg = f'{name} must contain only finite values'
        raise ValueError(msg)
    return arr


def _coerce_1d_real_numeric_float64(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        msg = f'{name} must be a 1D array'
        raise ValueError(msg)
    if arr.shape != expected_shape:
        msg = f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        raise ValueError(msg)
    if not _is_real_numeric_dtype(arr.dtype):
        msg = f'{name} must have a numeric dtype'
        raise ValueError(msg)
    return np.ascontiguousarray(arr, dtype=np.float64)


def _coerce_1d_integer_int64(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        msg = f'{name} must be a 1D array'
        raise ValueError(msg)
    if arr.shape != expected_shape:
        msg = f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        raise ValueError(msg)
    if np.issubdtype(arr.dtype, np.bool_):
        msg = f'{name} must contain integer values'
        raise ValueError(msg)
    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr, dtype=np.int64)
    if not _is_real_numeric_dtype(arr.dtype):
        msg = f'{name} must contain integer values'
        raise ValueError(msg)
    arr_f64 = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr_f64)):
        msg = f'{name} must contain only finite values'
        raise ValueError(msg)
    if not np.all(arr_f64 == np.rint(arr_f64)):
        msg = f'{name} must contain integer values'
        raise ValueError(msg)
    return np.ascontiguousarray(arr_f64, dtype=np.int64)


def _coerce_1d_bool_array(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        msg = f'{name} must be a 1D array'
        raise ValueError(msg)
    if arr.shape != expected_shape:
        msg = f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        raise ValueError(msg)
    if not np.issubdtype(arr.dtype, np.bool_):
        msg = f'{name} must have bool dtype'
        raise ValueError(msg)
    return np.ascontiguousarray(arr, dtype=bool)


def _read_int_scalar(npz: np.lib.npyio.NpzFile, key: str) -> int:
    arr = np.asarray(npz[key])
    if arr.size != 1:
        msg = f'{key} must be a scalar'
        raise ValueError(msg)
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(
        arr.dtype,
        np.integer,
    ):
        msg = f'{key} must have an integer dtype'
        raise ValueError(msg)
    return int(arr.reshape(-1)[0])


def _read_float_scalar(npz: np.lib.npyio.NpzFile, key: str) -> float:
    arr = np.asarray(npz[key])
    if arr.size != 1:
        msg = f'{key} must be a scalar'
        raise ValueError(msg)
    if not _is_real_numeric_dtype(arr.dtype):
        msg = f'{key} must be numeric'
        raise ValueError(msg)
    return float(arr.reshape(-1)[0])


def _validate_header_byte(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        msg = f'{name} must be an integer SEG-Y trace header byte'
        raise ValueError(msg)
    byte = int(value)
    if byte < 1 or byte > 240:
        msg = f'{name} must be between 1 and 240'
        raise ValueError(msg)
    return byte


def _coerce_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        msg = f'{name} must be an integer'
        raise ValueError(msg)
    out = int(value)
    if out < 0:
        msg = f'{name} must be greater than or equal to 0'
        raise ValueError(msg)
    return out


def _coerce_positive_int(value: object, *, name: str) -> int:
    out = _coerce_nonnegative_int(value, name=name)
    if out <= 0:
        msg = f'{name} must be greater than 0'
        raise ValueError(msg)
    return out


def _coerce_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        msg = f'{name} must be finite'
        raise ValueError(msg)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        msg = f'{name} must be finite'
        raise ValueError(msg) from exc
    if not np.isfinite(out):
        msg = f'{name} must be finite'
        raise ValueError(msg)
    return out


def _coerce_positive_finite_float(value: object, *, name: str) -> float:
    out = _coerce_finite_float(value, name=name)
    if out <= 0.0:
        msg = f'{name} must be finite and greater than 0'
        raise ValueError(msg)
    return out


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype,
        np.complexfloating,
    )


__all__ = [
    'DatumStaticSolution',
    'FirstBreakQcInputs',
    'build_first_break_qc_inputs',
    'load_datum_static_solution_npz',
    'load_offset_header_sorted',
]
