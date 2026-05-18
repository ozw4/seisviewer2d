"""Datum static geometry loading from TraceStore headers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.trace_store.reader import TraceStoreSectionReader
from app.utils.segy_scalars import (
    apply_segy_scalar,
    count_zero_segy_scalars,
    normalize_elevation_unit,
)


@dataclass(frozen=True)
class DatumStaticGeometryConfig:
    source_elevation_byte: int = 45
    receiver_elevation_byte: int = 41
    elevation_scalar_byte: int = 69
    source_depth_byte: int | None = None
    elevation_unit: Literal['m', 'ft'] = 'm'


@dataclass(frozen=True)
class DatumStaticGeometry:
    source_surface_elevation_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray
    source_depth_m_sorted: np.ndarray
    source_depth_used_sorted: np.ndarray
    elevation_scalar_sorted: np.ndarray
    elevation_scalar_zero_count: int
    source_elevation_byte: int
    receiver_elevation_byte: int
    elevation_scalar_byte: int
    source_depth_byte: int | None
    elevation_unit: str
    n_traces: int


def load_datum_static_geometry(
    *,
    reader: TraceStoreSectionReader,
    config: DatumStaticGeometryConfig,
) -> DatumStaticGeometry:
    """Load datum static geometry arrays in TraceStore sorted trace order."""
    _validate_config(config)
    n_traces = int(reader.traces.shape[0])

    source_surface_raw = _validate_header_array(
        reader.ensure_header(config.source_elevation_byte),
        name='source_elevation',
        byte=config.source_elevation_byte,
        n_traces=n_traces,
    )
    receiver_raw = _validate_header_array(
        reader.ensure_header(config.receiver_elevation_byte),
        name='receiver_elevation',
        byte=config.receiver_elevation_byte,
        n_traces=n_traces,
    )
    elevation_scalar_raw = _validate_header_array(
        reader.ensure_header(config.elevation_scalar_byte),
        name='elevation_scalar',
        byte=config.elevation_scalar_byte,
        n_traces=n_traces,
    )
    elevation_scalar = _coerce_integer_scalar_header(elevation_scalar_raw)

    source_surface = _apply_scalar_and_normalize(
        source_surface_raw,
        elevation_scalar,
        unit=config.elevation_unit,
    )
    receiver = _apply_scalar_and_normalize(
        receiver_raw,
        elevation_scalar,
        unit=config.elevation_unit,
    )

    if config.source_depth_byte is None:
        source_depth = np.zeros(n_traces, dtype=np.float64)
        source_depth_used = np.zeros(n_traces, dtype=bool)
    else:
        source_depth_raw = _validate_header_array(
            reader.ensure_header(config.source_depth_byte),
            name='source_depth',
            byte=config.source_depth_byte,
            n_traces=n_traces,
        )
        source_depth = _apply_scalar_and_normalize(
            source_depth_raw,
            elevation_scalar,
            unit=config.elevation_unit,
        )
        source_depth_used = np.ones(n_traces, dtype=bool)

    return DatumStaticGeometry(
        source_surface_elevation_m_sorted=np.ascontiguousarray(
            source_surface,
            dtype=np.float64,
        ),
        receiver_elevation_m_sorted=np.ascontiguousarray(
            receiver,
            dtype=np.float64,
        ),
        source_depth_m_sorted=np.ascontiguousarray(source_depth, dtype=np.float64),
        source_depth_used_sorted=np.ascontiguousarray(source_depth_used, dtype=bool),
        elevation_scalar_sorted=np.ascontiguousarray(elevation_scalar, dtype=np.int64),
        elevation_scalar_zero_count=count_zero_segy_scalars(elevation_scalar),
        source_elevation_byte=config.source_elevation_byte,
        receiver_elevation_byte=config.receiver_elevation_byte,
        elevation_scalar_byte=config.elevation_scalar_byte,
        source_depth_byte=config.source_depth_byte,
        elevation_unit=config.elevation_unit,
        n_traces=n_traces,
    )


def _validate_config(config: DatumStaticGeometryConfig) -> None:
    header_bytes = [
        _validate_required_header_byte(
            config.source_elevation_byte,
            name='source_elevation_byte',
        ),
        _validate_required_header_byte(
            config.receiver_elevation_byte,
            name='receiver_elevation_byte',
        ),
        _validate_required_header_byte(
            config.elevation_scalar_byte,
            name='elevation_scalar_byte',
        ),
    ]
    if config.source_depth_byte is not None:
        header_bytes.append(
            _validate_required_header_byte(
                config.source_depth_byte,
                name='source_depth_byte',
            )
        )
    if len(set(header_bytes)) != len(header_bytes):
        msg = 'datum static geometry header bytes must be unique'
        raise ValueError(msg)


def _validate_required_header_byte(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f'{name} must be an integer SEG-Y trace header byte'
        raise ValueError(msg)
    if value < 1 or value > 240:
        msg = f'{name} must be between 1 and 240'
        raise ValueError(msg)
    return value


def _validate_header_array(
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

    if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
        msg = f'{name} header byte {byte} must contain only finite values'
        raise ValueError(msg)
    return arr


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype,
        np.complexfloating,
    )


def _coerce_integer_scalar_header(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer):
        return np.asarray(arr, dtype=np.int64)

    arr_f64 = arr.astype(np.float64, copy=False)
    if not np.all(arr_f64 == np.rint(arr_f64)):
        msg = 'elevation_scalar header values must be integers'
        raise ValueError(msg)

    int64_info = np.iinfo(np.int64)
    if arr_f64.size and (
        float(arr_f64.min()) < int64_info.min
        or float(arr_f64.max()) > int64_info.max
    ):
        msg = 'elevation_scalar header values are outside the int64 range'
        raise ValueError(msg)
    return np.asarray(arr_f64, dtype=np.int64)


def _apply_scalar_and_normalize(
    values: np.ndarray,
    scalars: np.ndarray,
    *,
    unit: str,
) -> np.ndarray:
    scaled = apply_segy_scalar(values, scalars)
    return normalize_elevation_unit(scaled, unit)


__all__ = [
    'DatumStaticGeometry',
    'DatumStaticGeometryConfig',
    'load_datum_static_geometry',
]
