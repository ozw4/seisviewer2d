"""Artifact writer for datum static correction results."""

from __future__ import annotations

import csv
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.datum_static_validation import (
    ExistingStaticHeaderCheck,
    TraceShiftValidationResult,
)

SOLUTION_NPZ_NAME = 'datum_static_solution.npz'
QC_JSON_NAME = 'datum_static_qc.json'
STATICS_CSV_NAME = 'datum_statics.csv'

_CSV_COLUMNS = [
    'sorted_trace_index',
    'key1',
    'key2',
    'source_surface_elevation_m',
    'source_depth_m',
    'source_depth_used',
    'source_elevation_m',
    'receiver_elevation_m',
    'source_shift_ms',
    'receiver_shift_ms',
    'trace_shift_ms',
]


@dataclass(frozen=True)
class DatumStaticArtifactPaths:
    solution_npz: Path
    qc_json: Path
    statics_csv: Path


@dataclass(frozen=True)
class _ValidatedInputs:
    job_dir: Path
    trace_shift_s_sorted: np.ndarray
    source_shift_s_sorted: np.ndarray
    receiver_shift_s_sorted: np.ndarray
    source_surface_elevation_m_sorted: np.ndarray
    source_depth_m_sorted: np.ndarray
    source_depth_used_sorted: np.ndarray
    source_elevation_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray
    key1_sorted: np.ndarray
    key2_sorted: np.ndarray
    datum_elevation_m: float
    replacement_velocity_m_s: float
    dt: float
    key1_byte: int
    key2_byte: int
    source_elevation_byte: int
    receiver_elevation_byte: int
    elevation_scalar_byte: int
    source_depth_byte: int | None
    source_depth_enabled: bool
    elevation_unit: str
    elevation_scalar_zero_count: int
    header_source_segy_path: str

    @property
    def n_traces(self) -> int:
        return int(self.trace_shift_s_sorted.shape[0])


def write_datum_static_artifacts(
    *,
    job_dir: Path,
    trace_shift_s_sorted: np.ndarray,
    source_shift_s_sorted: np.ndarray,
    receiver_shift_s_sorted: np.ndarray,
    source_surface_elevation_m_sorted: np.ndarray,
    source_depth_m_sorted: np.ndarray,
    source_depth_used_sorted: np.ndarray,
    source_elevation_m_sorted: np.ndarray,
    receiver_elevation_m_sorted: np.ndarray,
    key1_sorted: np.ndarray,
    key2_sorted: np.ndarray,
    datum_elevation_m: float,
    replacement_velocity_m_s: float,
    dt: float,
    key1_byte: int,
    key2_byte: int,
    source_elevation_byte: int,
    receiver_elevation_byte: int,
    elevation_scalar_byte: int,
    source_depth_byte: int | None,
    source_depth_enabled: bool,
    elevation_unit: str,
    elevation_scalar_zero_count: int,
    existing_static_check: ExistingStaticHeaderCheck | None,
    trace_shift_validation: TraceShiftValidationResult | None,
    header_source_segy_path: str | None = None,
) -> DatumStaticArtifactPaths:
    """Write datum static solution, QC, and CSV artifacts atomically."""
    values = _validate_inputs(
        job_dir=job_dir,
        trace_shift_s_sorted=trace_shift_s_sorted,
        source_shift_s_sorted=source_shift_s_sorted,
        receiver_shift_s_sorted=receiver_shift_s_sorted,
        source_surface_elevation_m_sorted=source_surface_elevation_m_sorted,
        source_depth_m_sorted=source_depth_m_sorted,
        source_depth_used_sorted=source_depth_used_sorted,
        source_elevation_m_sorted=source_elevation_m_sorted,
        receiver_elevation_m_sorted=receiver_elevation_m_sorted,
        key1_sorted=key1_sorted,
        key2_sorted=key2_sorted,
        datum_elevation_m=datum_elevation_m,
        replacement_velocity_m_s=replacement_velocity_m_s,
        dt=dt,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        source_elevation_byte=source_elevation_byte,
        receiver_elevation_byte=receiver_elevation_byte,
        elevation_scalar_byte=elevation_scalar_byte,
        source_depth_byte=source_depth_byte,
        source_depth_enabled=source_depth_enabled,
        elevation_unit=elevation_unit,
        elevation_scalar_zero_count=elevation_scalar_zero_count,
        header_source_segy_path=header_source_segy_path,
    )

    solution_payload = _build_solution_payload(values)
    qc_payload = _build_qc_payload(
        values=values,
        existing_static_check=existing_static_check,
        trace_shift_validation=trace_shift_validation,
    )

    values.job_dir.mkdir(parents=True, exist_ok=True)
    paths = DatumStaticArtifactPaths(
        solution_npz=values.job_dir / SOLUTION_NPZ_NAME,
        qc_json=values.job_dir / QC_JSON_NAME,
        statics_csv=values.job_dir / STATICS_CSV_NAME,
    )

    _write_npz_atomic(paths.solution_npz, solution_payload)
    _write_json_atomic(paths.qc_json, qc_payload)
    _write_csv_atomic(paths.statics_csv, values)
    return paths


def _validate_inputs(
    *,
    job_dir: Path,
    trace_shift_s_sorted: np.ndarray,
    source_shift_s_sorted: np.ndarray,
    receiver_shift_s_sorted: np.ndarray,
    source_surface_elevation_m_sorted: np.ndarray,
    source_depth_m_sorted: np.ndarray,
    source_depth_used_sorted: np.ndarray,
    source_elevation_m_sorted: np.ndarray,
    receiver_elevation_m_sorted: np.ndarray,
    key1_sorted: np.ndarray,
    key2_sorted: np.ndarray,
    datum_elevation_m: float,
    replacement_velocity_m_s: float,
    dt: float,
    key1_byte: int,
    key2_byte: int,
    source_elevation_byte: int,
    receiver_elevation_byte: int,
    elevation_scalar_byte: int,
    source_depth_byte: int | None,
    source_depth_enabled: bool,
    elevation_unit: str,
    elevation_scalar_zero_count: int,
    header_source_segy_path: str | None,
) -> _ValidatedInputs:
    try:
        job_dir_path = Path(job_dir)
    except TypeError as exc:
        msg = 'job_dir must be path-like'
        raise ValueError(msg) from exc

    trace_shift = _coerce_1d_finite_float64(
        trace_shift_s_sorted,
        name='trace_shift_s_sorted',
    )
    n_traces = int(trace_shift.shape[0])
    if n_traces <= 0:
        msg = 'trace_shift_s_sorted must contain at least one trace'
        raise ValueError(msg)
    expected_shape = trace_shift.shape

    source_shift = _coerce_1d_finite_float64(
        source_shift_s_sorted,
        name='source_shift_s_sorted',
        expected_shape=expected_shape,
    )
    receiver_shift = _coerce_1d_finite_float64(
        receiver_shift_s_sorted,
        name='receiver_shift_s_sorted',
        expected_shape=expected_shape,
    )
    if not np.allclose(trace_shift, source_shift + receiver_shift):
        msg = (
            'trace_shift_s_sorted must match '
            'source_shift_s_sorted + receiver_shift_s_sorted'
        )
        raise ValueError(msg)

    source_surface = _coerce_1d_finite_float64(
        source_surface_elevation_m_sorted,
        name='source_surface_elevation_m_sorted',
        expected_shape=expected_shape,
    )
    source_depth = _coerce_1d_finite_float64(
        source_depth_m_sorted,
        name='source_depth_m_sorted',
        expected_shape=expected_shape,
    )
    source_depth_used = _coerce_1d_bool(
        source_depth_used_sorted,
        name='source_depth_used_sorted',
        expected_shape=expected_shape,
    )
    source_elevation = _coerce_1d_finite_float64(
        source_elevation_m_sorted,
        name='source_elevation_m_sorted',
        expected_shape=expected_shape,
    )
    receiver_elevation = _coerce_1d_finite_float64(
        receiver_elevation_m_sorted,
        name='receiver_elevation_m_sorted',
        expected_shape=expected_shape,
    )
    key1 = _coerce_1d_integer_int64(
        key1_sorted,
        name='key1_sorted',
        expected_shape=expected_shape,
    )
    key2 = _coerce_1d_integer_int64(
        key2_sorted,
        name='key2_sorted',
        expected_shape=expected_shape,
    )

    source_depth_enabled_bool = _coerce_bool_scalar(
        source_depth_enabled,
        name='source_depth_enabled',
    )
    if not source_depth_enabled_bool:
        if not np.all(source_depth == 0.0):
            msg = (
                'source_depth_m_sorted must be all 0.0 when '
                'source_depth_enabled is False'
            )
            raise ValueError(msg)
        if np.any(source_depth_used):
            msg = (
                'source_depth_used_sorted must be all False when '
                'source_depth_enabled is False'
            )
            raise ValueError(msg)

    return _ValidatedInputs(
        job_dir=job_dir_path,
        trace_shift_s_sorted=trace_shift,
        source_shift_s_sorted=source_shift,
        receiver_shift_s_sorted=receiver_shift,
        source_surface_elevation_m_sorted=source_surface,
        source_depth_m_sorted=source_depth,
        source_depth_used_sorted=source_depth_used,
        source_elevation_m_sorted=source_elevation,
        receiver_elevation_m_sorted=receiver_elevation,
        key1_sorted=key1,
        key2_sorted=key2,
        datum_elevation_m=_coerce_finite_float(
            datum_elevation_m,
            name='datum_elevation_m',
        ),
        replacement_velocity_m_s=_coerce_positive_finite_float(
            replacement_velocity_m_s,
            name='replacement_velocity_m_s',
        ),
        dt=_coerce_positive_finite_float(dt, name='dt'),
        key1_byte=_validate_header_byte(key1_byte, name='key1_byte'),
        key2_byte=_validate_header_byte(key2_byte, name='key2_byte'),
        source_elevation_byte=_validate_header_byte(
            source_elevation_byte,
            name='source_elevation_byte',
        ),
        receiver_elevation_byte=_validate_header_byte(
            receiver_elevation_byte,
            name='receiver_elevation_byte',
        ),
        elevation_scalar_byte=_validate_header_byte(
            elevation_scalar_byte,
            name='elevation_scalar_byte',
        ),
        source_depth_byte=_validate_optional_header_byte(
            source_depth_byte,
            name='source_depth_byte',
        ),
        source_depth_enabled=source_depth_enabled_bool,
        elevation_unit=_validate_elevation_unit(elevation_unit),
        elevation_scalar_zero_count=_validate_nonnegative_int(
            elevation_scalar_zero_count,
            name='elevation_scalar_zero_count',
        ),
        header_source_segy_path=''
        if header_source_segy_path is None
        else str(header_source_segy_path),
    )


def _coerce_1d_finite_float64(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        msg = f'{name} must be a 1D array'
        raise ValueError(msg)
    if expected_shape is not None and arr.shape != expected_shape:
        msg = f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        raise ValueError(msg)
    try:
        arr_f64 = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        msg = f'{name} must be numeric'
        raise ValueError(msg) from exc
    if not np.all(np.isfinite(arr_f64)):
        msg = f'{name} must contain only finite values'
        raise ValueError(msg)
    return np.asarray(arr_f64, dtype=np.float64)


def _coerce_1d_bool(
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
        return np.asarray(arr, dtype=bool)
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):
        arr_f64 = arr.astype(np.float64, copy=False)
        if not np.all(np.isfinite(arr_f64)):
            msg = f'{name} must contain only finite values'
            raise ValueError(msg)
        if np.all((arr_f64 == 0.0) | (arr_f64 == 1.0)):
            return np.asarray(arr_f64, dtype=bool)
    if arr.dtype == object and all(
        isinstance(value, (bool, np.bool_)) for value in arr
    ):
        return np.asarray(arr, dtype=bool)
    msg = f'{name} must be bool dtype or safely convertible to bool'
    raise ValueError(msg)


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
        return np.asarray(arr, dtype=np.int64)
    try:
        arr_f64 = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        msg = f'{name} must contain integer values'
        raise ValueError(msg) from exc
    if not np.all(np.isfinite(arr_f64)):
        msg = f'{name} must contain only finite values'
        raise ValueError(msg)
    if not np.all(arr_f64 == np.rint(arr_f64)):
        msg = f'{name} must contain integer values'
        raise ValueError(msg)
    return np.asarray(arr_f64, dtype=np.int64)


def _coerce_finite_float(value: float, *, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        msg = f'{name} must be finite'
        raise ValueError(msg) from exc
    if not np.isfinite(scalar):
        msg = f'{name} must be finite'
        raise ValueError(msg)
    return scalar


def _coerce_positive_finite_float(value: float, *, name: str) -> float:
    scalar = _coerce_finite_float(value, name=name)
    if scalar <= 0.0:
        msg = f'{name} must be finite and greater than 0'
        raise ValueError(msg)
    return scalar


def _coerce_bool_scalar(value: bool, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    msg = f'{name} must be a bool'
    raise ValueError(msg)


def _validate_header_byte(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        msg = f'{name} must be an integer SEG-Y trace header byte'
        raise ValueError(msg)
    value = int(value)
    if value < 1 or value > 240:
        msg = f'{name} must be between 1 and 240'
        raise ValueError(msg)
    return value


def _validate_optional_header_byte(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    return _validate_header_byte(value, name=name)


def _validate_elevation_unit(value: str) -> str:
    if value not in {'m', 'ft'}:
        msg = "elevation_unit must be 'm' or 'ft'"
        raise ValueError(msg)
    return value


def _validate_nonnegative_int(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        msg = f'{name} must be an integer'
        raise ValueError(msg)
    value = int(value)
    if value < 0:
        msg = f'{name} must be greater than or equal to 0'
        raise ValueError(msg)
    return value


def _build_solution_payload(values: _ValidatedInputs) -> dict[str, np.ndarray]:
    source_depth_byte = (
        -1 if values.source_depth_byte is None else values.source_depth_byte
    )
    return {
        'trace_shift_s_sorted': values.trace_shift_s_sorted,
        'source_shift_s_sorted': values.source_shift_s_sorted,
        'receiver_shift_s_sorted': values.receiver_shift_s_sorted,
        'source_surface_elevation_m_sorted': values.source_surface_elevation_m_sorted,
        'source_depth_m_sorted': values.source_depth_m_sorted,
        'source_depth_used_sorted': values.source_depth_used_sorted,
        'source_elevation_m_sorted': values.source_elevation_m_sorted,
        'receiver_elevation_m_sorted': values.receiver_elevation_m_sorted,
        'key1_sorted': values.key1_sorted,
        'key2_sorted': values.key2_sorted,
        'datum_elevation_m': np.asarray(values.datum_elevation_m, dtype=np.float64),
        'replacement_velocity_m_s': np.asarray(
            values.replacement_velocity_m_s,
            dtype=np.float64,
        ),
        'dt': np.asarray(values.dt, dtype=np.float64),
        'n_traces': np.asarray(values.n_traces, dtype=np.int64),
        'key1_byte': np.asarray(values.key1_byte, dtype=np.int64),
        'key2_byte': np.asarray(values.key2_byte, dtype=np.int64),
        'source_elevation_byte': np.asarray(
            values.source_elevation_byte,
            dtype=np.int64,
        ),
        'receiver_elevation_byte': np.asarray(
            values.receiver_elevation_byte,
            dtype=np.int64,
        ),
        'elevation_scalar_byte': np.asarray(
            values.elevation_scalar_byte,
            dtype=np.int64,
        ),
        'source_depth_byte': np.asarray(source_depth_byte, dtype=np.int64),
        'source_depth_enabled': np.asarray(
            values.source_depth_enabled,
            dtype=np.bool_,
        ),
        'elevation_unit': np.asarray(values.elevation_unit, dtype=np.str_),
        'elevation_scalar_zero_count': np.asarray(
            values.elevation_scalar_zero_count,
            dtype=np.int64,
        ),
        'header_source_segy_path': np.asarray(
            values.header_source_segy_path,
            dtype=np.str_,
        ),
    }


def _build_qc_payload(
    *,
    values: _ValidatedInputs,
    existing_static_check: ExistingStaticHeaderCheck | None,
    trace_shift_validation: TraceShiftValidationResult | None,
) -> dict[str, Any]:
    scalar_zero_count = int(values.elevation_scalar_zero_count)
    return {
        'datum_elevation_m': float(values.datum_elevation_m),
        'dt': float(values.dt),
        'elevation_unit': values.elevation_unit,
        'existing_statics': _existing_statics_payload(existing_static_check),
        'n_traces': values.n_traces,
        'receiver_elevation_m': _stats_without_max_abs(
            values.receiver_elevation_m_sorted
        ),
        'receiver_shift_ms': _finite_stats_ms(values.receiver_shift_s_sorted),
        'replacement_velocity_m_s': float(values.replacement_velocity_m_s),
        'scalar': {
            'zero_count': scalar_zero_count,
            'zero_fraction': float(scalar_zero_count / values.n_traces),
        },
        'source_depth_enabled': bool(values.source_depth_enabled),
        'source_elevation_m': _stats_without_max_abs(values.source_elevation_m_sorted),
        'source_shift_ms': _finite_stats_ms(values.source_shift_s_sorted),
        'trace_shift_ms': _finite_stats_ms(values.trace_shift_s_sorted),
        'validation': _validation_payload(trace_shift_validation),
    }


def _existing_statics_payload(
    check: ExistingStaticHeaderCheck | None,
) -> dict[str, Any]:
    if check is None:
        return {'checked': False}
    return {
        'checked': bool(check.checked),
        'nonzero_any_count': _coerce_int_from_attr(
            check.nonzero_any_count,
            name='existing_static_check.nonzero_any_count',
        ),
        'nonzero_receiver_static_count': _coerce_int_from_attr(
            check.nonzero_receiver_static_count,
            name='existing_static_check.nonzero_receiver_static_count',
        ),
        'nonzero_source_static_count': _coerce_int_from_attr(
            check.nonzero_source_static_count,
            name='existing_static_check.nonzero_source_static_count',
        ),
        'nonzero_total_static_count': _coerce_int_from_attr(
            check.nonzero_total_static_count,
            name='existing_static_check.nonzero_total_static_count',
        ),
        'policy': str(check.policy),
    }


def _validation_payload(
    result: TraceShiftValidationResult | None,
) -> dict[str, Any]:
    if result is None:
        return {'max_abs_shift_checked': False}
    return {
        'max_abs_observed_shift_ms': _coerce_finite_float(
            result.max_abs_observed_shift_ms,
            name='trace_shift_validation.max_abs_observed_shift_ms',
        ),
        'max_abs_shift_ms': _coerce_finite_float(
            result.max_abs_shift_ms,
            name='trace_shift_validation.max_abs_shift_ms',
        ),
    }


def _coerce_int_from_attr(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        msg = f'{name} must be an integer'
        raise ValueError(msg)
    return int(value)


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        msg = 'stats input must be a 1D array'
        raise ValueError(msg)
    if arr.size == 0:
        msg = 'stats input must not be empty'
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = 'stats input must contain only finite values'
        raise ValueError(msg)
    return {
        'max': float(np.max(arr)),
        'max_abs': float(np.max(np.abs(arr))),
        'mean': float(np.mean(arr)),
        'min': float(np.min(arr)),
    }


def _finite_stats_ms(values_s: np.ndarray) -> dict[str, float]:
    return _finite_stats(np.asarray(values_s, dtype=np.float64) * 1000.0)


def _stats_without_max_abs(values: np.ndarray) -> dict[str, float]:
    stats = _finite_stats(values)
    return {
        'max': stats['max'],
        'mean': stats['mean'],
        'min': stats['min'],
    }


def _write_npz_atomic(out_path: Path, payload: dict[str, np.ndarray]) -> None:
    def write(tmp_path: Path) -> None:
        with tmp_path.open('wb') as handle:
            np.savez(handle, **payload)

    _atomic_write(out_path, write)


def _write_json_atomic(out_path: Path, payload: dict[str, Any]) -> None:
    def write(tmp_path: Path) -> None:
        with tmp_path.open('w', encoding='utf-8') as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write('\n')

    _atomic_write(out_path, write)


def _write_csv_atomic(out_path: Path, values: _ValidatedInputs) -> None:
    def write(tmp_path: Path) -> None:
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(_CSV_COLUMNS)
            for sorted_trace_index in range(values.n_traces):
                writer.writerow(
                    [
                        sorted_trace_index,
                        int(values.key1_sorted[sorted_trace_index]),
                        int(values.key2_sorted[sorted_trace_index]),
                        float(
                            values.source_surface_elevation_m_sorted[sorted_trace_index]
                        ),
                        float(values.source_depth_m_sorted[sorted_trace_index]),
                        'true'
                        if bool(values.source_depth_used_sorted[sorted_trace_index])
                        else 'false',
                        float(values.source_elevation_m_sorted[sorted_trace_index]),
                        float(values.receiver_elevation_m_sorted[sorted_trace_index]),
                        float(
                            values.source_shift_s_sorted[sorted_trace_index] * 1000.0
                        ),
                        float(
                            values.receiver_shift_s_sorted[sorted_trace_index] * 1000.0
                        ),
                        float(values.trace_shift_s_sorted[sorted_trace_index] * 1000.0),
                    ]
                )

    _atomic_write(out_path, write)


def _atomic_write(out_path: Path, write: Callable[[Path], None]) -> None:
    tmp_path = out_path.with_name(f'{out_path.name}.tmp-{uuid.uuid4().hex}')
    try:
        write(tmp_path)
        tmp_path.replace(out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


__all__ = [
    'DatumStaticArtifactPaths',
    'write_datum_static_artifacts',
]
