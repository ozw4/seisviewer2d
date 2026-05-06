"""Validation boundary for future time-term static inversion jobs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from app.api.schemas import TimeTermStaticApplyRequest
from app.core.state import AppState
from app.services.errors import DomainError
from app.services.geometry_linkage_loader import load_geometry_linkage_artifact
from app.services.job_artifact_refs import resolve_job_artifact_path
from app.services.reader import get_reader


@dataclass(frozen=True)
class TimeTermValidationResult:
    """Validated shallow inputs for a future time-term solver job."""

    file_id: str
    dt: float
    n_traces: int
    linkage_artifact_path: Path | None


def validate_time_term_request(
    req: TimeTermStaticApplyRequest,
    *,
    state: AppState,
) -> TimeTermValidationResult:
    """Validate request-level preconditions before time-term inversion is implemented."""
    _validate_file_registered(state, req.file_id)
    reader = _resolve_reader(state, req)
    n_traces = _reader_n_traces(reader)
    dt = _resolve_dt(state, req.file_id, reader=reader)
    linkage_path = _validate_linkage_artifact(
        state,
        req=req,
        expected_n_traces=n_traces,
    )
    return TimeTermValidationResult(
        file_id=req.file_id,
        dt=dt,
        n_traces=n_traces,
        linkage_artifact_path=linkage_path,
    )


def _validate_file_registered(state: AppState, file_id: str) -> None:
    if state.file_registry.get_record(file_id) is None:
        raise ValueError(f'file_id not found: {file_id}')


def _resolve_reader(state: AppState, req: TimeTermStaticApplyRequest) -> object:
    try:
        return get_reader(req.file_id, req.key1_byte, req.key2_byte, state=state)
    except DomainError as exc:
        raise ValueError(exc.detail) from exc
    except Exception as exc:  # noqa: BLE001
        msg = f'Could not open TraceStore for file_id {req.file_id}: {exc}'
        raise ValueError(msg) from exc


def _reader_n_traces(reader: object) -> int:
    n_traces = getattr(reader, 'ntraces', None)
    if n_traces is None:
        meta = getattr(reader, 'meta', None)
        if isinstance(meta, dict):
            n_traces = meta.get('n_traces')
    if n_traces is None:
        traces = getattr(reader, 'traces', None)
        shape = getattr(traces, 'shape', ())
        if shape:
            n_traces = shape[0]
    if isinstance(n_traces, bool) or not isinstance(n_traces, (int, np.integer)):
        raise ValueError('TraceStore n_traces must be available')
    n_traces_int = int(n_traces)
    if n_traces_int <= 0:
        raise ValueError('TraceStore n_traces must be greater than 0')
    return n_traces_int


def _resolve_dt(state: AppState, file_id: str, *, reader: object) -> float:
    dt_raw = None
    record = state.file_registry.get_record(file_id)
    if isinstance(record, dict):
        dt_raw = record.get('dt')
    if not _is_positive_finite_number(dt_raw):
        meta = getattr(reader, 'meta', None)
        if isinstance(meta, dict):
            dt_raw = meta.get('dt')
    if not _is_positive_finite_number(dt_raw):
        try:
            dt_raw = state.file_registry.get_dt(file_id)
        except Exception as exc:  # noqa: BLE001
            msg = f'dt must be finite and greater than 0 for file_id {file_id}'
            raise ValueError(msg) from exc
    if not _is_positive_finite_number(dt_raw):
        raise ValueError(f'dt must be finite and greater than 0 for file_id {file_id}')
    return float(dt_raw)


def _is_positive_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _validate_linkage_artifact(
    state: AppState,
    *,
    req: TimeTermStaticApplyRequest,
    expected_n_traces: int,
) -> Path | None:
    linkage = req.linkage
    if linkage.mode == 'none':
        return None
    if linkage.mode == 'optional' and linkage.job_id is None:
        return None
    if not linkage.job_id:
        raise ValueError('linkage.job_id is required when linkage.mode is required')

    try:
        path = resolve_job_artifact_path(
            state,
            job_id=linkage.job_id,
            name=linkage.artifact_name,
            allowed_job_types={'statics'},
            allowed_statics_kinds={'geometry_linkage'},
            expected_file_id=req.file_id,
            expected_key1_byte=req.key1_byte,
            expected_key2_byte=req.key2_byte,
            reference_label='linkage',
        )
    except ValueError as exc:
        raise ValueError(f'linkage artifact validation failed: {exc}') from exc

    _validate_linkage_trace_node_arrays(path, expected_n_traces=expected_n_traces)
    try:
        load_geometry_linkage_artifact(
            path,
            expected_n_traces=expected_n_traces,
            expected_key1_byte=req.key1_byte,
            expected_key2_byte=req.key2_byte,
        )
    except ValueError as exc:
        raise ValueError(f'linkage artifact validation failed: {exc}') from exc
    return path


def _validate_linkage_trace_node_arrays(
    path: Path,
    *,
    expected_n_traces: int,
) -> None:
    try:
        npz_file = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f'Could not read linkage artifact: {path}') from exc

    with npz_file as npz:
        for name in ('source_node_id_sorted', 'receiver_node_id_sorted'):
            if name not in npz.files:
                raise ValueError(f'linkage artifact missing required array: {name}')
            try:
                values = np.asarray(npz[name])
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f'linkage artifact array {name} could not be read') from exc
            _validate_node_id_array(
                values,
                name=name,
                expected_n_traces=expected_n_traces,
            )


def _validate_node_id_array(
    values: np.ndarray,
    *,
    name: str,
    expected_n_traces: int,
) -> None:
    arr = np.asarray(values)
    expected_shape = (int(expected_n_traces),)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if arr.shape != expected_shape:
        raise ValueError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if np.issubdtype(arr.dtype, np.bool_) or not _is_real_numeric_dtype(arr.dtype):
        raise ValueError(f'{name} must have a real numeric dtype')
    if np.issubdtype(arr.dtype, np.floating):
        arr_f64 = arr.astype(np.float64, copy=False)
        if not np.all(np.isfinite(arr_f64)):
            raise ValueError(f'{name} must contain only finite values')
        if not np.all(arr_f64 == np.rint(arr_f64)):
            raise ValueError(f'{name} must contain only integer values')
    if np.any(arr < 0):
        raise ValueError(f'{name} must not contain negative values')


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype,
        np.complexfloating,
    )


__all__ = ['TimeTermValidationResult', 'validate_time_term_request']
