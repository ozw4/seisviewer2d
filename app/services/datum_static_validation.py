"""Validation helpers for datum static correction inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.trace_store.reader import TraceStoreSectionReader


@dataclass(frozen=True)
class ExistingStaticHeaderConfig:
    policy: Literal['fail_if_nonzero'] = 'fail_if_nonzero'
    source_static_byte: int | None = 99
    receiver_static_byte: int | None = 101
    total_static_byte: int | None = 103

    def __post_init__(self) -> None:
        _validate_existing_static_config(self)


@dataclass(frozen=True)
class ExistingStaticHeaderCheck:
    policy: str
    checked: bool
    source_static_byte: int | None
    receiver_static_byte: int | None
    total_static_byte: int | None
    nonzero_source_static_count: int
    nonzero_receiver_static_count: int
    nonzero_total_static_count: int
    nonzero_any_count: int
    checked_bytes: tuple[int, ...]


@dataclass(frozen=True)
class TraceShiftValidationResult:
    n_traces: int
    max_abs_shift_ms: float
    min_shift_ms: float
    max_shift_ms: float
    mean_shift_ms: float
    max_abs_observed_shift_ms: float


def validate_existing_static_headers(
    *,
    reader: TraceStoreSectionReader,
    config: ExistingStaticHeaderConfig,
) -> ExistingStaticHeaderCheck:
    """Validate that configured existing SEG-Y static headers are all zero."""
    _validate_existing_static_config(config)
    n_traces = int(reader.traces.shape[0])
    masks_by_name: dict[str, np.ndarray] = {}
    counts_by_name = {
        'source_static_byte': 0,
        'receiver_static_byte': 0,
        'total_static_byte': 0,
    }
    checked_bytes: list[int] = []

    for field_name, byte in _configured_static_header_bytes(config):
        if byte is None:
            continue
        values = _read_static_header(reader=reader, byte=byte)
        arr = _validate_static_header_array(
            values,
            name=field_name,
            byte=byte,
            n_traces=n_traces,
        )
        nonzero_mask = np.asarray(arr != 0, dtype=bool)
        masks_by_name[field_name] = nonzero_mask
        counts_by_name[field_name] = int(np.count_nonzero(nonzero_mask))
        checked_bytes.append(byte)

    if masks_by_name:
        nonzero_any_count = int(
            np.count_nonzero(np.logical_or.reduce(tuple(masks_by_name.values())))
        )
    else:
        nonzero_any_count = 0

    if nonzero_any_count:
        parts = [
            f'{field_name}={getattr(config, field_name)} count={count}'
            for field_name, count in counts_by_name.items()
            if count
        ]
        msg = 'existing SEG-Y static headers contain nonzero values: ' + ', '.join(
            parts
        )
        raise ValueError(msg)

    return ExistingStaticHeaderCheck(
        policy=config.policy,
        checked=bool(checked_bytes),
        source_static_byte=config.source_static_byte,
        receiver_static_byte=config.receiver_static_byte,
        total_static_byte=config.total_static_byte,
        nonzero_source_static_count=counts_by_name['source_static_byte'],
        nonzero_receiver_static_count=counts_by_name['receiver_static_byte'],
        nonzero_total_static_count=counts_by_name['total_static_byte'],
        nonzero_any_count=nonzero_any_count,
        checked_bytes=tuple(checked_bytes),
    )


def validate_trace_shift_limits(
    *,
    trace_shift_s_sorted: np.ndarray,
    max_abs_shift_ms: float,
    expected_n_traces: int | None = None,
) -> TraceShiftValidationResult:
    """Validate sorted per-trace shifts against a maximum absolute shift limit."""
    shifts_s = np.asarray(trace_shift_s_sorted)
    if shifts_s.ndim != 1:
        msg = 'trace_shift_s_sorted must be a 1D array'
        raise ValueError(msg)
    if expected_n_traces is not None and shifts_s.shape != (expected_n_traces,):
        msg = (
            'trace_shift_s_sorted shape mismatch: '
            f'expected {(expected_n_traces,)}, got {shifts_s.shape}'
        )
        raise ValueError(msg)
    if not _is_real_numeric_dtype(shifts_s.dtype):
        msg = 'trace_shift_s_sorted must have a numeric dtype'
        raise ValueError(msg)

    shifts_ms = shifts_s.astype(np.float64, copy=False) * 1000.0
    if not np.all(np.isfinite(shifts_ms)):
        msg = 'trace_shift_s_sorted must contain only finite values'
        raise ValueError(msg)

    try:
        limit_ms = float(max_abs_shift_ms)
    except (TypeError, ValueError) as exc:
        msg = 'max_abs_shift_ms must be finite'
        raise ValueError(msg) from exc
    if not np.isfinite(limit_ms) or limit_ms <= 0.0:
        msg = 'max_abs_shift_ms must be finite and greater than 0'
        raise ValueError(msg)

    abs_shift_ms = np.abs(shifts_ms)
    observed_ms = float(abs_shift_ms.max()) if abs_shift_ms.size else 0.0
    exceeding_mask = abs_shift_ms > limit_ms
    exceeding_count = int(np.count_nonzero(exceeding_mask))
    if exceeding_count:
        msg = (
            'trace_shift_s_sorted exceeds max_abs_shift_ms: '
            f'limit={limit_ms} ms, observed={observed_ms} ms, '
            f'count={exceeding_count}'
        )
        raise ValueError(msg)

    return TraceShiftValidationResult(
        n_traces=int(shifts_ms.size),
        max_abs_shift_ms=limit_ms,
        min_shift_ms=float(shifts_ms.min()) if shifts_ms.size else 0.0,
        max_shift_ms=float(shifts_ms.max()) if shifts_ms.size else 0.0,
        mean_shift_ms=float(shifts_ms.mean()) if shifts_ms.size else 0.0,
        max_abs_observed_shift_ms=observed_ms,
    )


def _validate_existing_static_config(config: ExistingStaticHeaderConfig) -> None:
    if config.policy != 'fail_if_nonzero':
        msg = "existing static header policy must be 'fail_if_nonzero'"
        raise ValueError(msg)

    header_bytes = [
        _validate_optional_static_header_byte(value, name=name)
        for name, value in _configured_static_header_bytes(config)
    ]
    specified = [byte for byte in header_bytes if byte is not None]
    if not specified:
        msg = 'at least one existing static header byte must be specified'
        raise ValueError(msg)
    if len(set(specified)) != len(specified):
        msg = 'existing static header bytes must be unique'
        raise ValueError(msg)


def _configured_static_header_bytes(
    config: ExistingStaticHeaderConfig,
) -> tuple[tuple[str, int | None], ...]:
    return (
        ('source_static_byte', config.source_static_byte),
        ('receiver_static_byte', config.receiver_static_byte),
        ('total_static_byte', config.total_static_byte),
    )


def _validate_optional_static_header_byte(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f'{name} must be an integer SEG-Y trace header byte or None'
        raise ValueError(msg)
    if value < 1 or value > 240:
        msg = f'{name} must be between 1 and 240'
        raise ValueError(msg)
    return value


def _read_static_header(
    *,
    reader: TraceStoreSectionReader,
    byte: int,
) -> np.ndarray:
    try:
        reader.ensure_header(byte)
        return reader.get_header(byte)
    except Exception as exc:
        msg = f'failed to read existing static header byte {byte}: {exc}'
        raise ValueError(msg) from exc


def _validate_static_header_array(
    values: np.ndarray,
    *,
    name: str,
    byte: int,
    n_traces: int,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        msg = f'{name} header byte {byte} must be a 1D array'
        raise ValueError(msg)
    expected_shape = (n_traces,)
    if arr.shape != expected_shape:
        msg = (
            f'{name} header byte {byte} shape mismatch: '
            f'expected {expected_shape}, got {arr.shape}'
        )
        raise ValueError(msg)
    if not _is_real_numeric_dtype(arr.dtype):
        msg = f'{name} header byte {byte} must have a numeric dtype'
        raise ValueError(msg)

    if np.issubdtype(arr.dtype, np.integer):
        return arr

    arr_f64 = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr_f64)):
        msg = f'{name} header byte {byte} must contain only finite values'
        raise ValueError(msg)
    if not np.all(arr_f64 == np.rint(arr_f64)):
        msg = f'{name} header byte {byte} must contain only integer values'
        raise ValueError(msg)
    return arr


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype,
        np.complexfloating,
    )


__all__ = [
    'ExistingStaticHeaderCheck',
    'ExistingStaticHeaderConfig',
    'TraceShiftValidationResult',
    'validate_existing_static_headers',
    'validate_trace_shift_limits',
]
