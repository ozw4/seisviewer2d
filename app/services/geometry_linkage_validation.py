"""Validation helpers for static linkage TraceStore headers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.common.array_validation import (
    is_real_numeric_dtype as _is_real_numeric_dtype,
)
from app.trace_store.reader import TraceStoreSectionReader


@dataclass(frozen=True)
class GeometryLinkageHeaderConfig:
    source_x_byte: int = 73
    source_y_byte: int = 77
    receiver_x_byte: int = 81
    receiver_y_byte: int = 85
    coordinate_scalar_byte: int = 71

    def __post_init__(self) -> None:
        _validate_config(self)


@dataclass(frozen=True)
class GeometryLinkageHeaders:
    source_x: np.ndarray
    source_y: np.ndarray
    receiver_x: np.ndarray
    receiver_y: np.ndarray
    coordinate_scalar: np.ndarray
    checked_bytes: tuple[int, ...]


def validate_geometry_linkage_headers(
    *,
    reader: TraceStoreSectionReader,
    config: GeometryLinkageHeaderConfig,
) -> GeometryLinkageHeaders:
    """Materialize and validate static linkage headers in TraceStore order."""
    n_traces = int(reader.traces.shape[0])
    raw_headers: dict[str, np.ndarray] = {}
    checked_bytes: list[int] = []

    for name, byte in _required_headers(config):
        values = _read_header(reader=reader, byte=byte)
        raw_headers[name] = np.asarray(values)
        checked_bytes.append(byte)

    coordinate_scalar = _validate_coordinate_scalar_header(
        raw_headers['coordinate_scalar'],
        name='coordinate_scalar',
        byte=config.coordinate_scalar_byte,
        n_traces=n_traces,
    )
    source_x = _validate_coordinate_header(
        raw_headers['source_x'],
        name='source_x',
        byte=config.source_x_byte,
        n_traces=n_traces,
    )
    source_y = _validate_coordinate_header(
        raw_headers['source_y'],
        name='source_y',
        byte=config.source_y_byte,
        n_traces=n_traces,
    )
    receiver_x = _validate_coordinate_header(
        raw_headers['receiver_x'],
        name='receiver_x',
        byte=config.receiver_x_byte,
        n_traces=n_traces,
    )
    receiver_y = _validate_coordinate_header(
        raw_headers['receiver_y'],
        name='receiver_y',
        byte=config.receiver_y_byte,
        n_traces=n_traces,
    )

    return GeometryLinkageHeaders(
        source_x=source_x,
        source_y=source_y,
        receiver_x=receiver_x,
        receiver_y=receiver_y,
        coordinate_scalar=coordinate_scalar,
        checked_bytes=tuple(checked_bytes),
    )


def _required_headers(
    config: GeometryLinkageHeaderConfig,
) -> tuple[tuple[str, int], ...]:
    return (
        ('coordinate_scalar', config.coordinate_scalar_byte),
        ('source_x', config.source_x_byte),
        ('source_y', config.source_y_byte),
        ('receiver_x', config.receiver_x_byte),
        ('receiver_y', config.receiver_y_byte),
    )


def _validate_config(config: GeometryLinkageHeaderConfig) -> None:
    header_bytes = [
        _validate_header_byte(value, name=name)
        for name, value in (
            ('source_x_byte', config.source_x_byte),
            ('source_y_byte', config.source_y_byte),
            ('receiver_x_byte', config.receiver_x_byte),
            ('receiver_y_byte', config.receiver_y_byte),
            ('coordinate_scalar_byte', config.coordinate_scalar_byte),
        )
    ]
    if len(set(header_bytes)) != len(header_bytes):
        msg = 'geometry header bytes must be unique'
        raise ValueError(msg)


def _validate_header_byte(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f'{name} must be an integer SEG-Y trace header byte'
        raise ValueError(msg)
    if value < 1 or value > 240:
        msg = f'{name} must be in the range 1..240'
        raise ValueError(msg)
    return value


def _read_header(
    *,
    reader: TraceStoreSectionReader,
    byte: int,
) -> np.ndarray:
    try:
        return reader.ensure_header(byte)
    except Exception as exc:
        msg = f'failed to read geometry linkage header byte {byte}: {exc}'
        raise ValueError(msg) from exc


def _validate_coordinate_header(
    values: np.ndarray,
    *,
    name: str,
    byte: int,
    n_traces: int,
) -> np.ndarray:
    arr = _validate_header_shape(values, name=name, byte=byte, n_traces=n_traces)
    if not _is_real_numeric_dtype(arr.dtype):
        msg = f'{name} header byte {byte} must have a real numeric dtype'
        raise ValueError(msg)
    if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
        msg = f'{name} header byte {byte} must contain only finite values'
        raise ValueError(msg)
    return arr


def _validate_coordinate_scalar_header(
    values: np.ndarray,
    *,
    name: str,
    byte: int,
    n_traces: int,
) -> np.ndarray:
    arr = _validate_header_shape(values, name=name, byte=byte, n_traces=n_traces)
    if np.issubdtype(arr.dtype, np.integer):
        return arr
    if not np.issubdtype(arr.dtype, np.floating):
        msg = (
            f'{name} header byte {byte} must have an integer dtype '
            'or integer-valued float dtype'
        )
        raise ValueError(msg)

    arr_f64 = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr_f64)):
        msg = f'{name} header byte {byte} must contain only finite values'
        raise ValueError(msg)
    if not np.all(arr_f64 == np.rint(arr_f64)):
        msg = f'{name} header byte {byte} must contain only integer values'
        raise ValueError(msg)
    return arr


def _validate_header_shape(
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
    return arr


__all__ = [
    'GeometryLinkageHeaderConfig',
    'GeometryLinkageHeaders',
    'validate_geometry_linkage_headers',
]
