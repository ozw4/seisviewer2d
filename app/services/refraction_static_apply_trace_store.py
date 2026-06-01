"""Apply final refraction static shifts to TraceStores."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import partial
import json
from pathlib import Path
import re
import shutil
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from app.contracts.statics.refraction.apply import RefractionStaticApplyRequest
from app.core.state import AppState
from app.services.common.artifact_io import write_json_atomic
from app.services.common.array_validation import (
    coerce_1d_bool_array,
    coerce_1d_real_numeric_float64,
    coerce_1d_string_array,
    coerce_header_byte,
    coerce_nonnegative_finite_float,
    coerce_positive_finite_float,
    coerce_positive_int,
)
from app.services.corrected_trace_store import (
    TimeShiftedTraceStoreResult,
    build_time_shifted_trace_store,
)
from app.services.reader import get_reader
from app.statics.refraction.artifacts import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    REFRACTION_STATICS_CSV_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_HISTORY_JSON_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    refraction_static_double_application_qc,
)
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
    RefractionStaticApplyTraceStoreResult,
    RefractionTraceShiftValidationResult,
)
from app.services.trace_store_index_validation import validate_sorted_to_original
from app.services.trace_store_registration import (
    register_trace_store,
    trace_store_cache_key,
)
from app.trace_store.reader import TraceStoreSectionReader
from app.utils.time_shift import shift_traces_linear

RefractionStaticApplyMode = Literal['refraction_from_raw']

CORRECTED_FILE_JSON_NAME = 'corrected_file.json'
REFRACTION_STATIC_APPLY_QC_JSON_NAME = 'refraction_static_apply_qc.json'
SIGN_CONVENTION = 'corrected(t) = raw(t - shift_s)'

_SAFE_STORE_NAME_RE = re.compile(r'[^A-Za-z0-9_.-]+')
_BUILDER_SIGN_CONVENTION = (
    'corrected(t)=raw(t-shift_s); positive_shift_delays_events'
)
_OPTIONAL_SOLUTION_METADATA_KEYS = (
    'weathering_replacement_trace_shift_s_sorted',
    'floating_datum_elevation_shift_s_sorted',
    'flat_datum_shift_s_sorted',
    'base_refraction_trace_shift_s_sorted',
    'final_trace_shift_s_sorted',
    'bedrock_velocity_m_s',
    'weathering_velocity_m_s',
    'datum_mode',
    'floating_datum_mode',
    'flat_datum_elevation_m',
    'sign_convention',
)


class RefractionStaticTraceStoreApplyError(ValueError):
    """Raised when refraction statics cannot be applied to a TraceStore."""


_coerce_positive_int = partial(
    coerce_positive_int,
    error_type=RefractionStaticTraceStoreApplyError,
)
_coerce_header_byte = partial(
    coerce_header_byte,
    error_type=RefractionStaticTraceStoreApplyError,
)
_coerce_positive_finite_float = partial(
    coerce_positive_finite_float,
    error_type=RefractionStaticTraceStoreApplyError,
)
_coerce_nonnegative_finite_float = partial(
    coerce_nonnegative_finite_float,
    error_type=RefractionStaticTraceStoreApplyError,
)


@dataclass(frozen=True)
class LoadedRefractionStaticSolutionForApply:
    refraction_trace_shift_s_sorted: np.ndarray
    trace_static_valid_mask_sorted: np.ndarray
    trace_static_status_sorted: np.ndarray
    sorted_trace_index: np.ndarray
    source_solution_artifact: str | None
    metadata: dict[str, object]
    final_trace_shift_s_sorted: np.ndarray | None = None
    final_trace_static_valid_mask_sorted: np.ndarray | None = None
    final_trace_static_status_sorted: np.ndarray | None = None
    trace_field_static_status_sorted: np.ndarray | None = None


@dataclass(frozen=True)
class _SelectedTraceShiftForApplication:
    trace_shift_s_sorted: np.ndarray
    trace_static_valid_mask_sorted: np.ndarray
    trace_static_status_sorted: np.ndarray
    shift_field: str
    static_components_applied: tuple[str, ...]
    field_corrections_applied_to_trace_shift: bool
    requested_field_components: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SourceTraceStoreContext:
    store_path: Path
    reader: TraceStoreSectionReader
    meta: dict[str, object]
    sorted_trace_index: np.ndarray
    n_traces: int
    n_samples: int
    dt: float
    key1_byte: int
    key2_byte: int
    original_segy_path: str | None


def apply_trace_shifts_to_array(
    *,
    traces: np.ndarray,
    sample_interval_s: float,
    trace_shift_s_sorted: np.ndarray,
    interpolation: Literal['linear'] = 'linear',
    fill_value: float = 0.0,
    output_dtype: np.dtype | str = np.float32,
) -> np.ndarray:
    """Apply sorted per-trace shifts using ``corrected(t) = raw(t - shift_s)``."""
    if interpolation != 'linear':
        raise RefractionStaticTraceStoreApplyError('interpolation must be "linear"')
    dtype = np.dtype(output_dtype)
    if dtype != np.dtype('float32'):
        raise RefractionStaticTraceStoreApplyError('output_dtype must be "float32"')

    arr = np.asarray(traces)
    if arr.ndim != 2:
        raise RefractionStaticTraceStoreApplyError('traces must be a 2D array')
    dt = _coerce_positive_finite_float(
        sample_interval_s,
        name='sample_interval_s',
    )
    try:
        shifted = shift_traces_linear(
            arr,
            np.asarray(trace_shift_s_sorted, dtype=np.float64),
            dt,
            fill_value=fill_value,
        )
    except ValueError as exc:
        raise RefractionStaticTraceStoreApplyError(str(exc)) from exc
    return np.ascontiguousarray(shifted, dtype=np.float32)


def validate_refraction_trace_shifts_for_application(
    *,
    trace_shift_s_sorted: np.ndarray,
    trace_static_valid_mask_sorted: np.ndarray,
    trace_static_status_sorted: np.ndarray,
    n_traces: int,
    max_abs_shift_ms: float,
    require_all_traces_valid: bool = True,
) -> RefractionTraceShiftValidationResult:
    """Validate final refraction trace shifts before TraceStore application."""
    expected_shape = (_coerce_positive_int(n_traces, name='n_traces'),)
    shifts = _require_1d_float64(
        trace_shift_s_sorted,
        name='refraction_trace_shift_s_sorted',
        expected_shape=expected_shape,
        allow_nonfinite=True,
    )
    valid_mask = _require_1d_bool(
        trace_static_valid_mask_sorted,
        name='trace_static_valid_mask_sorted',
        expected_shape=expected_shape,
    )
    statuses = _require_1d_string(
        trace_static_status_sorted,
        name='trace_static_status_sorted',
        expected_shape=expected_shape,
    )
    status_counts = _status_counts(statuses)
    max_abs_ms = _coerce_nonnegative_finite_float(
        max_abs_shift_ms,
        name='max_abs_shift_ms',
    )

    if require_all_traces_valid and not bool(np.all(valid_mask)):
        invalid_count = int(np.count_nonzero(~valid_mask))
        raise RefractionStaticTraceStoreApplyError(
            'Refraction statics contain invalid trace shifts; corrected '
            'TraceStore was not created. '
            f'invalid_trace_shift_count={invalid_count}; '
            f'trace_static_status_counts={status_counts}'
        )

    applied_mask = np.ones(expected_shape, dtype=bool) if require_all_traces_valid else valid_mask
    if not np.all(np.isfinite(shifts[applied_mask])):
        raise RefractionStaticTraceStoreApplyError(
            'refraction_trace_shift_s_sorted contains non-finite shifts for '
            'traces selected for application'
        )

    shift_ms = shifts * 1000.0
    exceeds_mask = np.abs(shift_ms) > max_abs_ms
    exceeds_count = int(np.count_nonzero(exceeds_mask & applied_mask))
    max_abs_applied_ms = float(np.max(np.abs(shift_ms[applied_mask])))
    if exceeds_count:
        raise RefractionStaticTraceStoreApplyError(
            'refraction_trace_shift_s_sorted exceeds max_abs_shift_ms: '
            f'{max_abs_applied_ms:.6g} > {max_abs_ms:.6g}; '
            f'exceeds_max_abs_shift_count={exceeds_count}; '
            f'trace_static_status_counts={status_counts}'
        )

    valid_count = int(np.count_nonzero(valid_mask))
    invalid_count = int(valid_mask.size - valid_count)
    return RefractionTraceShiftValidationResult(
        trace_shift_s_sorted=np.ascontiguousarray(shifts, dtype=np.float64),
        trace_static_valid_mask_sorted=np.ascontiguousarray(valid_mask, dtype=bool),
        trace_static_status_sorted=np.ascontiguousarray(statuses),
        trace_static_status_counts=status_counts,
        max_abs_shift_ms=max_abs_ms,
        max_abs_applied_shift_ms=max_abs_applied_ms,
        exceeds_max_abs_shift_count=exceeds_count,
        n_valid_trace_shifts=valid_count,
        n_invalid_trace_shifts=invalid_count,
        n_zero_trace_shifts=int(np.count_nonzero(shift_ms == 0.0)),
        n_positive_trace_shifts=int(np.count_nonzero(shift_ms > 0.0)),
        n_negative_trace_shifts=int(np.count_nonzero(shift_ms < 0.0)),
    )


def load_refraction_static_solution_for_apply(
    solution_npz_path: Path,
) -> LoadedRefractionStaticSolutionForApply:
    """Load the P0 refraction static solution fields needed for application."""
    path = Path(solution_npz_path)
    if not path.exists():
        raise RefractionStaticTraceStoreApplyError(
            f'refraction_static_solution.npz does not exist: {path}'
        )
    if not path.is_file():
        raise RefractionStaticTraceStoreApplyError(
            f'refraction_static_solution.npz is not a file: {path}'
        )

    required = {
        'refraction_trace_shift_s_sorted',
        'trace_static_valid_mask_sorted',
        'trace_static_status_sorted',
        'sorted_trace_index',
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(required.difference(data.files))
            if missing:
                raise RefractionStaticTraceStoreApplyError(
                    'refraction_static_solution.npz is missing required fields: '
                    + ', '.join(missing)
                )
            n_traces = int(np.asarray(data['refraction_trace_shift_s_sorted']).shape[0])
            expected_shape = (n_traces,)
            shifts = _require_1d_float64(
                data['refraction_trace_shift_s_sorted'],
                name='refraction_trace_shift_s_sorted',
                expected_shape=expected_shape,
                allow_nonfinite=True,
            )
            valid_mask = _require_1d_bool(
                data['trace_static_valid_mask_sorted'],
                name='trace_static_valid_mask_sorted',
                expected_shape=expected_shape,
            )
            statuses = _require_1d_string(
                data['trace_static_status_sorted'],
                name='trace_static_status_sorted',
                expected_shape=expected_shape,
            )
            final_shifts = (
                _require_1d_float64(
                    data['final_trace_shift_s_sorted'],
                    name='final_trace_shift_s_sorted',
                    expected_shape=expected_shape,
                    allow_nonfinite=True,
                )
                if 'final_trace_shift_s_sorted' in data.files
                else None
            )
            final_valid_mask = (
                _require_1d_bool(
                    data['final_trace_static_valid_mask_sorted'],
                    name='final_trace_static_valid_mask_sorted',
                    expected_shape=expected_shape,
                )
                if 'final_trace_static_valid_mask_sorted' in data.files
                else None
            )
            final_statuses = (
                _require_1d_string(
                    data['final_trace_static_status_sorted'],
                    name='final_trace_static_status_sorted',
                    expected_shape=expected_shape,
                )
                if 'final_trace_static_status_sorted' in data.files
                else None
            )
            field_statuses = (
                _require_1d_string(
                    data['trace_field_static_status_sorted'],
                    name='trace_field_static_status_sorted',
                    expected_shape=expected_shape,
                )
                if 'trace_field_static_status_sorted' in data.files
                else None
            )
            sorted_trace_index = validate_sorted_to_original(
                data['sorted_trace_index'],
                expected_n_traces=n_traces,
                role='refraction_static_solution.npz sorted_trace_index',
            )
            metadata = _solution_metadata_from_npz(data)
    except RefractionStaticTraceStoreApplyError:
        raise
    except ValueError as exc:
        if 'Object arrays cannot be loaded' in str(exc):
            raise RefractionStaticTraceStoreApplyError(
                'refraction_static_solution.npz contains object dtype arrays'
            ) from exc
        raise RefractionStaticTraceStoreApplyError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise RefractionStaticTraceStoreApplyError(
            f'refraction_static_solution.npz could not be loaded: {path}'
        ) from exc

    return LoadedRefractionStaticSolutionForApply(
        refraction_trace_shift_s_sorted=shifts,
        trace_static_valid_mask_sorted=valid_mask,
        trace_static_status_sorted=statuses,
        sorted_trace_index=sorted_trace_index,
        source_solution_artifact=path.name,
        metadata=metadata,
        final_trace_shift_s_sorted=final_shifts,
        final_trace_static_valid_mask_sorted=final_valid_mask,
        final_trace_static_status_sorted=final_statuses,
        trace_field_static_status_sorted=field_statuses,
    )


def apply_refraction_statics_to_trace_store(
    *,
    req: RefractionStaticApplyRequest,
    result: RefractionDatumStaticsResult,
    state: AppState,
    job_id: str,
    job_dir: Path,
) -> RefractionStaticApplyTraceStoreResult:
    """Apply an in-memory final refraction statics result to the source TraceStore."""
    solution = _solution_from_datum_result(result)
    return _apply_refraction_solution_to_trace_store(
        req=req,
        solution=solution,
        state=state,
        job_id=job_id,
        job_dir=job_dir,
    )


def apply_refraction_statics_from_solution_artifact(
    *,
    req: RefractionStaticApplyRequest,
    solution_npz_path: Path,
    state: AppState,
    job_id: str,
    job_dir: Path,
) -> RefractionStaticApplyTraceStoreResult:
    """Apply final refraction statics from a P0 solution artifact."""
    solution = load_refraction_static_solution_for_apply(solution_npz_path)
    return _apply_refraction_solution_to_trace_store(
        req=req,
        solution=solution,
        state=state,
        job_id=job_id,
        job_dir=job_dir,
    )


def _apply_refraction_solution_to_trace_store(
    *,
    req: RefractionStaticApplyRequest,
    solution: LoadedRefractionStaticSolutionForApply,
    state: AppState,
    job_id: str,
    job_dir: Path,
) -> RefractionStaticApplyTraceStoreResult:
    request = RefractionStaticApplyRequest.model_validate(req)
    _validate_apply_options(request)
    source = _resolve_source_trace_store(
        req=request,
        state=state,
    )
    _validate_sorted_trace_order(
        solution.sorted_trace_index,
        source_sorted_trace_index=source.sorted_trace_index,
        n_traces=source.n_traces,
    )
    selected_shift = _select_trace_shift_for_application(
        req=request,
        solution=solution,
    )
    selected_shift = _with_double_application_policy(
        req=request,
        source_meta=source.meta,
        selected=selected_shift,
    )
    validation = validate_refraction_trace_shifts_for_application(
        trace_shift_s_sorted=selected_shift.trace_shift_s_sorted,
        trace_static_valid_mask_sorted=selected_shift.trace_static_valid_mask_sorted,
        trace_static_status_sorted=selected_shift.trace_static_status_sorted,
        n_traces=source.n_traces,
        max_abs_shift_ms=request.apply.max_abs_shift_ms,
        require_all_traces_valid=True,
    )

    corrected_file_id = str(uuid4())
    output_store_path = _corrected_store_path(
        source_store_path=source.store_path,
        job_id=job_id,
    )
    artifacts_dir = Path(job_dir)
    corrected_file_json_path = artifacts_dir / CORRECTED_FILE_JSON_NAME
    qc_json_path = artifacts_dir / REFRACTION_STATIC_APPLY_QC_JSON_NAME
    derived_metadata = _build_derived_metadata(
        source_meta=source.meta,
        job_id=job_id,
        solution_artifact=solution.source_solution_artifact,
        selected_shift=selected_shift,
    )

    build_result: TimeShiftedTraceStoreResult | None = None
    qc: dict[str, Any] | None = None
    try:
        build_result = build_time_shifted_trace_store(
            source_store_path=source.store_path,
            output_store_path=output_store_path,
            trace_shift_s_sorted=validation.trace_shift_s_sorted,
            fill_value=request.apply.fill_value,
            output_dtype=request.apply.output_dtype,
            derived_metadata=derived_metadata,
            from_file_id=request.file_id,
            original_segy_path=source.original_segy_path,
            header_bytes_to_materialize=(request.key1_byte, request.key2_byte),
        )
        registered_reader = register_trace_store(
            state=state,
            file_id=corrected_file_id,
            store_dir=build_result.store_path,
            key1_byte=request.key1_byte,
            key2_byte=request.key2_byte,
            dt=build_result.dt,
            update_registry=True,
            touch_meta=True,
            preload_header_bytes=(request.key1_byte, request.key2_byte),
        )
        _verify_registered_trace_store(
            state=state,
            file_id=corrected_file_id,
            store_path=build_result.store_path,
            key1_byte=request.key1_byte,
            key2_byte=request.key2_byte,
            reader=registered_reader,
        )
        qc = _build_apply_qc_payload(
            req=request,
            job_id=job_id,
            corrected_file_id=corrected_file_id,
            build_result=build_result,
            validation=validation,
            solution=solution,
            selected_shift=selected_shift,
            corrected_tracestore_path_written=True,
        )
        write_json_atomic(qc_json_path, qc, make_parent=True)
        write_json_atomic(
            corrected_file_json_path,
            _build_corrected_file_payload(
                req=request,
                job_id=job_id,
                corrected_file_id=corrected_file_id,
                build_result=build_result,
                validation=validation,
                solution=solution,
                selected_shift=selected_shift,
            ),
            make_parent=True,
        )
    except Exception:
        _cleanup_registration(
            state,
            file_id=corrected_file_id,
            key1_byte=request.key1_byte,
            key2_byte=request.key2_byte,
        )
        _cleanup_store(output_store_path)
        _cleanup_artifact(corrected_file_json_path)
        _cleanup_artifact(qc_json_path)
        raise

    return RefractionStaticApplyTraceStoreResult(
        source_file_id=request.file_id,
        corrected_file_id=corrected_file_id,
        source_trace_store_path=source.store_path,
        corrected_trace_store_path=build_result.store_path,
        n_traces=build_result.n_traces,
        n_samples=build_result.n_samples,
        sample_interval_s=build_result.dt,
        interpolation=request.apply.interpolation,
        fill_value=request.apply.fill_value,
        output_dtype=request.apply.output_dtype,
        applied_shift_s_sorted=validation.trace_shift_s_sorted,
        applied_shift_ms_sorted=validation.trace_shift_s_sorted * 1000.0,
        trace_static_valid_mask_sorted=validation.trace_static_valid_mask_sorted,
        trace_static_status_sorted=validation.trace_static_status_sorted,
        max_abs_applied_shift_ms=validation.max_abs_applied_shift_ms,
        n_valid_trace_shifts=validation.n_valid_trace_shifts,
        n_invalid_trace_shifts=validation.n_invalid_trace_shifts,
        n_zero_trace_shifts=validation.n_zero_trace_shifts,
        n_positive_trace_shifts=validation.n_positive_trace_shifts,
        n_negative_trace_shifts=validation.n_negative_trace_shifts,
        corrected_file_json=corrected_file_json_path,
        qc_json=qc_json_path,
        qc=qc,
    )


def _validate_apply_options(req: RefractionStaticApplyRequest) -> None:
    if req.apply.mode != 'refraction_from_raw':
        raise RefractionStaticTraceStoreApplyError(
            f'unsupported refraction apply mode: {req.apply.mode}'
        )
    if req.apply.interpolation != 'linear':
        raise RefractionStaticTraceStoreApplyError('interpolation must be "linear"')
    if req.apply.output_dtype != 'float32':
        raise RefractionStaticTraceStoreApplyError('output_dtype must be "float32"')
    if req.apply.register_corrected_file is not True:
        raise RefractionStaticTraceStoreApplyError(
            'apply.register_corrected_file must be true to create a corrected TraceStore'
        )
    _coerce_nonnegative_finite_float(
        req.apply.max_abs_shift_ms,
        name='max_abs_shift_ms',
    )


def _resolve_source_trace_store(
    *,
    req: RefractionStaticApplyRequest,
    state: AppState,
) -> _SourceTraceStoreContext:
    try:
        store_path = Path(state.file_registry.get_store_path(req.file_id))
    except Exception as exc:  # noqa: BLE001
        raise RefractionStaticTraceStoreApplyError(
            f'source file_id is not registered with a TraceStore: {req.file_id}'
        ) from exc
    if not store_path.exists():
        raise RefractionStaticTraceStoreApplyError(
            f'source TraceStore path does not exist: {store_path}'
        )
    if not store_path.is_dir():
        raise RefractionStaticTraceStoreApplyError(
            f'source TraceStore path is not a directory: {store_path}'
        )

    try:
        reader = get_reader(req.file_id, req.key1_byte, req.key2_byte, state=state)
    except Exception as exc:  # noqa: BLE001
        raise RefractionStaticTraceStoreApplyError(
            f'source TraceStore reader could not be resolved: {exc}'
        ) from exc
    reader_store_dir = Path(reader.store_dir)
    if reader_store_dir.resolve() != store_path.resolve():
        raise RefractionStaticTraceStoreApplyError(
            'source TraceStore reader path does not match file registry'
        )
    meta = _read_json_object(store_path / 'meta.json', label='source TraceStore meta.json')
    traces = getattr(reader, 'traces', None)
    if not isinstance(traces, np.ndarray) or traces.ndim != 2:
        raise RefractionStaticTraceStoreApplyError(
            'source TraceStore reader.traces must be a 2D array'
        )
    n_traces = _coerce_positive_int(meta.get('n_traces'), name='meta.n_traces')
    n_samples = _coerce_positive_int(meta.get('n_samples'), name='meta.n_samples')
    if (int(traces.shape[0]), int(traces.shape[1])) != (n_traces, n_samples):
        raise RefractionStaticTraceStoreApplyError(
            'source TraceStore trace shape mismatch: '
            f'meta={(n_traces, n_samples)}, traces={traces.shape}'
        )
    dt = _coerce_positive_finite_float(meta.get('dt'), name='meta.dt')
    key_bytes = meta.get('key_bytes')
    if not isinstance(key_bytes, dict):
        raise RefractionStaticTraceStoreApplyError(
            'source TraceStore meta.key_bytes must be an object'
        )
    key1_byte = _coerce_header_byte(
        key_bytes.get('key1'),
        name='meta.key_bytes.key1',
    )
    key2_byte = _coerce_header_byte(
        key_bytes.get('key2'),
        name='meta.key_bytes.key2',
    )
    if key1_byte != int(req.key1_byte):
        raise RefractionStaticTraceStoreApplyError(
            f'source TraceStore key1_byte mismatch: expected {req.key1_byte}, got {key1_byte}'
        )
    if key2_byte != int(req.key2_byte):
        raise RefractionStaticTraceStoreApplyError(
            f'source TraceStore key2_byte mismatch: expected {req.key2_byte}, got {key2_byte}'
        )
    try:
        sorted_trace_index = validate_sorted_to_original(
            reader.get_sorted_to_original(),
            expected_n_traces=n_traces,
            role='source TraceStore',
        )
    except ValueError as exc:
        raise RefractionStaticTraceStoreApplyError(
            f'source TraceStore sorted trace order cannot be verified: {exc}'
        ) from exc

    original_segy_path = meta.get('original_segy_path')
    if original_segy_path is not None and not isinstance(original_segy_path, str):
        raise RefractionStaticTraceStoreApplyError(
            'source TraceStore original_segy_path must be a string or null'
        )

    return _SourceTraceStoreContext(
        store_path=store_path,
        reader=reader,
        meta=meta,
        sorted_trace_index=sorted_trace_index,
        n_traces=n_traces,
        n_samples=n_samples,
        dt=dt,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        original_segy_path=original_segy_path,
    )


def _validate_sorted_trace_order(
    sorted_trace_index: np.ndarray,
    *,
    source_sorted_trace_index: np.ndarray,
    n_traces: int,
) -> None:
    solution_index = validate_sorted_to_original(
        sorted_trace_index,
        expected_n_traces=n_traces,
        role='refraction sorted_trace_index',
    )
    if not np.array_equal(solution_index, source_sorted_trace_index):
        raise RefractionStaticTraceStoreApplyError(
            'sorted_trace_index mismatch: refraction statics are not aligned '
            'with the source TraceStore sorted order'
        )


def _solution_from_datum_result(
    result: RefractionDatumStaticsResult,
) -> LoadedRefractionStaticSolutionForApply:
    if not isinstance(result, RefractionDatumStaticsResult):
        raise RefractionStaticTraceStoreApplyError(
            'result must be a RefractionDatumStaticsResult instance'
        )
    n_traces = int(np.asarray(result.sorted_trace_index).shape[0])
    expected_shape = (n_traces,)
    return LoadedRefractionStaticSolutionForApply(
        refraction_trace_shift_s_sorted=_require_1d_float64(
            result.refraction_trace_shift_s_sorted,
            name='refraction_trace_shift_s_sorted',
            expected_shape=expected_shape,
            allow_nonfinite=True,
        ),
        trace_static_valid_mask_sorted=_require_1d_bool(
            result.trace_static_valid_mask_sorted,
            name='trace_static_valid_mask_sorted',
            expected_shape=expected_shape,
        ),
        trace_static_status_sorted=_require_1d_string(
            result.trace_static_status_sorted,
            name='trace_static_status_sorted',
            expected_shape=expected_shape,
        ),
        sorted_trace_index=validate_sorted_to_original(
            result.sorted_trace_index,
            expected_n_traces=n_traces,
            role='result.sorted_trace_index',
        ),
        source_solution_artifact=REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        metadata={
            'bedrock_velocity_m_s': float(result.bedrock_velocity_m_s),
            'weathering_velocity_m_s': float(result.weathering_velocity_m_s),
            'datum_mode': str(result.datum_mode),
            'floating_datum_mode': str(result.floating_datum_mode),
            'flat_datum_elevation_m': (
                None
                if result.flat_datum_elevation_m is None
                else float(result.flat_datum_elevation_m)
            ),
            'sign_convention': SIGN_CONVENTION,
        },
        final_trace_shift_s_sorted=(
            None
            if result.final_trace_shift_s_sorted is None
            else _require_1d_float64(
                result.final_trace_shift_s_sorted,
                name='final_trace_shift_s_sorted',
                expected_shape=expected_shape,
                allow_nonfinite=True,
            )
        ),
        final_trace_static_valid_mask_sorted=(
            None
            if result.final_trace_static_valid_mask_sorted is None
            else _require_1d_bool(
                result.final_trace_static_valid_mask_sorted,
                name='final_trace_static_valid_mask_sorted',
                expected_shape=expected_shape,
            )
        ),
        final_trace_static_status_sorted=(
            None
            if result.final_trace_static_status_sorted is None
            else _require_1d_string(
                result.final_trace_static_status_sorted,
                name='final_trace_static_status_sorted',
                expected_shape=expected_shape,
            )
        ),
        trace_field_static_status_sorted=(
            None
            if result.trace_field_static_status_sorted is None
            else _require_1d_string(
                result.trace_field_static_status_sorted,
                name='trace_field_static_status_sorted',
                expected_shape=expected_shape,
            )
        ),
    )


def _select_trace_shift_for_application(
    *,
    req: RefractionStaticApplyRequest,
    solution: LoadedRefractionStaticSolutionForApply,
) -> _SelectedTraceShiftForApplication:
    requested_components = _requested_field_components(req)
    field_corrections_applied = bool(
        requested_components
        and req.field_corrections.composition.enabled
        and req.field_corrections.composition.apply_to_trace_shift
    )
    if not field_corrections_applied:
        return _SelectedTraceShiftForApplication(
            trace_shift_s_sorted=solution.refraction_trace_shift_s_sorted,
            trace_static_valid_mask_sorted=solution.trace_static_valid_mask_sorted,
            trace_static_status_sorted=solution.trace_static_status_sorted,
            shift_field='refraction_trace_shift_s_sorted',
            static_components_applied=('refraction',),
            field_corrections_applied_to_trace_shift=False,
            requested_field_components=requested_components,
        )

    if solution.final_trace_shift_s_sorted is None:
        raise RefractionStaticTraceStoreApplyError(
            'field_corrections.composition.apply_to_trace_shift=true requires '
            'final_trace_shift_s_sorted'
        )
    if (
        solution.final_trace_static_valid_mask_sorted is None
        or solution.final_trace_static_status_sorted is None
    ):
        raise RefractionStaticTraceStoreApplyError(
            'field_corrections.composition.apply_to_trace_shift=true requires '
            'final trace static status arrays'
        )
    _validate_trace_field_policy_for_application(req=req, solution=solution)
    return _SelectedTraceShiftForApplication(
        trace_shift_s_sorted=solution.final_trace_shift_s_sorted,
        trace_static_valid_mask_sorted=solution.final_trace_static_valid_mask_sorted,
        trace_static_status_sorted=solution.final_trace_static_status_sorted,
        shift_field='final_trace_shift_s_sorted',
        static_components_applied=('refraction', *requested_components),
        field_corrections_applied_to_trace_shift=True,
        requested_field_components=requested_components,
    )


def _requested_field_components(
    req: RefractionStaticApplyRequest,
) -> tuple[str, ...]:
    components: list[str] = []
    if req.field_corrections.source_depth.mode != 'none':
        components.append('source_depth')
    if req.field_corrections.uphole.mode != 'none':
        components.append('uphole')
    if req.field_corrections.manual_static.mode != 'none':
        components.append('manual_static')
    return tuple(components)


def _validate_trace_field_policy_for_application(
    *,
    req: RefractionStaticApplyRequest,
    solution: LoadedRefractionStaticSolutionForApply,
) -> None:
    policy = req.field_corrections.composition.invalid_component_policy
    if policy != 'fail':
        return
    statuses = solution.trace_field_static_status_sorted
    if statuses is None:
        raise RefractionStaticTraceStoreApplyError(
            'field_corrections.composition.invalid_component_policy=fail '
            'requires trace_field_static_status_sorted'
        )
    invalid_count = int(np.count_nonzero(np.asarray(statuses, dtype=str) != 'ok'))
    if invalid_count:
        raise RefractionStaticTraceStoreApplyError(
            'invalid field-correction components prevent corrected TraceStore '
            'creation; invalid_trace_field_shift_count='
            f'{invalid_count}; trace_field_static_status_counts='
            f'{_status_counts(statuses)}'
        )


def _with_double_application_policy(
    *,
    req: RefractionStaticApplyRequest,
    source_meta: Mapping[str, object],
    selected: _SelectedTraceShiftForApplication,
) -> _SelectedTraceShiftForApplication:
    qc = refraction_static_double_application_qc(req=req, source_meta=source_meta)
    if qc.get('status') == 'checked':
        return selected
    message = str(
        qc.get('message')
        or 'static history double-application policy rejected the job'
    )
    if qc.get('status') == 'duplicate_rejected':
        raise RefractionStaticTraceStoreApplyError(message)
    warnings = tuple(str(item) for item in qc.get('warnings', []) if item)
    return replace(selected, warnings=(*selected.warnings, *warnings))


def _build_derived_metadata(
    *,
    source_meta: Mapping[str, object],
    job_id: str,
    solution_artifact: str | None,
    selected_shift: _SelectedTraceShiftForApplication,
) -> dict[str, object]:
    source_derived = source_meta.get('derived')
    components = _component_dicts(source_derived if isinstance(source_derived, dict) else {})
    components.append(
        {
            'name': 'refraction_static_correction',
            'job_id': str(job_id),
            'solution_artifact': solution_artifact or REFRACTION_STATIC_SOLUTION_NPZ_NAME,
            'shift_field': selected_shift.shift_field,
            'value_kind': 'applied_event_time_shift_s',
            'apply_mode': 'refraction_from_raw',
            'static_components_applied': list(
                selected_shift.static_components_applied
            ),
            'field_corrections_applied_to_trace_shift': (
                selected_shift.field_corrections_applied_to_trace_shift
            ),
            'sign_convention': SIGN_CONVENTION,
        }
    )
    metadata: dict[str, object] = {
        'statics_kind': 'refraction',
        'derived_by': 'refraction_static_correction',
        'derivation': 'refraction_static_correction',
        'source_job_id': str(job_id),
        'applied_to': 'raw_trace_store',
        'components': components,
        'solution_artifact': solution_artifact or REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        'shift_field': selected_shift.shift_field,
        'value_kind': 'applied_event_time_shift_s',
        'apply_mode': 'refraction_from_raw',
        'static_components_applied': list(selected_shift.static_components_applied),
        'field_corrections_applied_to_trace_shift': (
            selected_shift.field_corrections_applied_to_trace_shift
        ),
        'field_correction_components_requested': list(
            selected_shift.requested_field_components
        ),
        'refraction_sign_convention': SIGN_CONVENTION,
    }
    if selected_shift.warnings:
        metadata['warnings'] = list(selected_shift.warnings)
    return metadata


def _build_corrected_file_payload(
    *,
    req: RefractionStaticApplyRequest,
    job_id: str,
    corrected_file_id: str,
    build_result: TimeShiftedTraceStoreResult,
    validation: RefractionTraceShiftValidationResult,
    solution: LoadedRefractionStaticSolutionForApply,
    selected_shift: _SelectedTraceShiftForApplication,
) -> dict[str, object]:
    payload: dict[str, object] = {
        'schema_version': 1,
        'artifact_kind': 'corrected_file',
        'source_file_id': req.file_id,
        'corrected_file_id': corrected_file_id,
        'file_id': corrected_file_id,
        'statics_kind': 'refraction',
        'apply_mode': req.apply.mode,
        'sign_convention': SIGN_CONVENTION,
        'shift_field': selected_shift.shift_field,
        'static_components_applied': list(selected_shift.static_components_applied),
        'field_corrections_applied_to_trace_shift': (
            selected_shift.field_corrections_applied_to_trace_shift
        ),
        'field_correction_components_requested': list(
            selected_shift.requested_field_components
        ),
        'interpolation': req.apply.interpolation,
        'fill_value': float(req.apply.fill_value),
        'output_dtype': req.apply.output_dtype,
        'store_name': build_result.store_path.name,
        'derived_from_file_id': req.file_id,
        'derived_by': 'refraction_static_correction',
        'derivation': 'refraction_static_correction',
        'source_job_id': job_id,
        'job_id': job_id,
        'key1_byte': int(req.key1_byte),
        'key2_byte': int(req.key2_byte),
        'dt': float(build_result.dt),
        'n_traces': int(build_result.n_traces),
        'n_samples': int(build_result.n_samples),
        'sample_interval_s': float(build_result.dt),
        'max_abs_applied_shift_ms': validation.max_abs_applied_shift_ms,
        'source_solution_artifact': (
            solution.source_solution_artifact or REFRACTION_STATIC_SOLUTION_NPZ_NAME
        ),
        'solution_artifact': (
            solution.source_solution_artifact or REFRACTION_STATIC_SOLUTION_NPZ_NAME
        ),
        'apply_qc_artifact': REFRACTION_STATIC_APPLY_QC_JSON_NAME,
        'artifact_names': _corrected_artifact_names(),
        'solution_metadata': _json_safe_mapping(solution.metadata),
    }
    if selected_shift.warnings:
        payload['warnings'] = list(selected_shift.warnings)
    return payload


def _build_apply_qc_payload(
    *,
    req: RefractionStaticApplyRequest,
    job_id: str,
    corrected_file_id: str,
    build_result: TimeShiftedTraceStoreResult,
    validation: RefractionTraceShiftValidationResult,
    solution: LoadedRefractionStaticSolutionForApply,
    selected_shift: _SelectedTraceShiftForApplication,
    corrected_tracestore_path_written: bool,
) -> dict[str, Any]:
    shift_ms = validation.trace_shift_s_sorted * 1000.0
    payload: dict[str, Any] = {
        'schema_version': 1,
        'artifact_kind': 'refraction_static_apply_qc',
        'statics_kind': 'refraction',
        'apply_mode': req.apply.mode,
        'sign_convention': SIGN_CONVENTION,
        'shift_field': selected_shift.shift_field,
        'static_components_applied': list(selected_shift.static_components_applied),
        'field_corrections_applied_to_trace_shift': (
            selected_shift.field_corrections_applied_to_trace_shift
        ),
        'field_correction_components_requested': list(
            selected_shift.requested_field_components
        ),
        'source_file_id': req.file_id,
        'corrected_file_id': corrected_file_id,
        'job_id': job_id,
        'n_traces': int(build_result.n_traces),
        'n_samples': int(build_result.n_samples),
        'sample_interval_s': float(build_result.dt),
        'interpolation': req.apply.interpolation,
        'fill_value': float(req.apply.fill_value),
        'output_dtype': req.apply.output_dtype,
        'n_valid_trace_shifts': validation.n_valid_trace_shifts,
        'n_invalid_trace_shifts': validation.n_invalid_trace_shifts,
        'n_positive_trace_shifts': validation.n_positive_trace_shifts,
        'n_negative_trace_shifts': validation.n_negative_trace_shifts,
        'n_zero_trace_shifts': validation.n_zero_trace_shifts,
        'applied_shift_min_ms': _stat(shift_ms, 'min'),
        'applied_shift_max_ms': _stat(shift_ms, 'max'),
        'applied_shift_median_ms': _stat(shift_ms, 'median'),
        'applied_shift_p95_abs_ms': _stat(np.abs(shift_ms), 'p95'),
        'max_abs_applied_shift_ms': validation.max_abs_applied_shift_ms,
        'max_abs_shift_ms': validation.max_abs_shift_ms,
        'exceeds_max_abs_shift_count': validation.exceeds_max_abs_shift_count,
        'trace_static_status_counts': validation.trace_static_status_counts,
        'source_solution_artifact': (
            solution.source_solution_artifact or REFRACTION_STATIC_SOLUTION_NPZ_NAME
        ),
        'corrected_store_name': build_result.store_path.name,
        'corrected_tracestore_path_written': bool(corrected_tracestore_path_written),
    }
    if selected_shift.warnings:
        payload['warnings'] = list(selected_shift.warnings)
    return payload


def _corrected_artifact_names() -> list[str]:
    return [
        REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        REFRACTION_STATIC_QC_JSON_NAME,
        REFRACTION_STATICS_CSV_NAME,
        NEAR_SURFACE_MODEL_CSV_NAME,
        FIRST_BREAK_RESIDUALS_CSV_NAME,
        REFRACTION_STATIC_COMPONENTS_CSV_NAME,
        REFRACTION_STATIC_HISTORY_JSON_NAME,
        CORRECTED_FILE_JSON_NAME,
        REFRACTION_STATIC_APPLY_QC_JSON_NAME,
    ]


def _corrected_store_path(*, source_store_path: Path, job_id: str) -> Path:
    source_name = _safe_store_name_component(source_store_path.name)
    safe_job_id = _safe_store_name_component(str(job_id))
    store_name = f'{source_name}.statics.refraction.{safe_job_id}'
    output_path = source_store_path.parent / store_name
    if output_path.exists() or output_path.is_symlink():
        raise RefractionStaticTraceStoreApplyError(
            f'corrected output path already exists: {output_path}'
        )
    return output_path


def _safe_store_name_component(value: str) -> str:
    safe = _SAFE_STORE_NAME_RE.sub('_', str(value))
    if safe in {'', '.', '..'}:
        raise RefractionStaticTraceStoreApplyError(
            'TraceStore name cannot be made filesystem-safe'
        )
    return safe


def _verify_registered_trace_store(
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


def _solution_metadata_from_npz(npz: np.lib.npyio.NpzFile) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key in _OPTIONAL_SOLUTION_METADATA_KEYS:
        if key not in npz.files:
            continue
        arr = np.asarray(npz[key])
        _reject_object_dtype(arr, name=key)
        if arr.shape == ():
            metadata[key] = _scalar_to_json_value(arr.item())
        elif arr.ndim == 1 and arr.size <= 8:
            metadata[key] = [_scalar_to_json_value(item) for item in arr.tolist()]
    return metadata


def _json_safe_mapping(values: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in values.items():
        if isinstance(value, np.generic):
            out[str(key)] = _scalar_to_json_value(value.item())
        elif isinstance(value, float):
            out[str(key)] = value if np.isfinite(value) else None
        elif isinstance(value, int | str | bool) or value is None:
            out[str(key)] = value
        elif isinstance(value, list):
            out[str(key)] = [
                _scalar_to_json_value(item) if isinstance(item, np.generic) else item
                for item in value
            ]
    return out


def _scalar_to_json_value(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.generic):
        return _scalar_to_json_value(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, int | str | bool) or value is None:
        return value
    return str(value)


def _component_dicts(derived: Mapping[str, object]) -> list[dict[str, object]]:
    components = derived.get('components')
    if components is None:
        return []
    if not isinstance(components, list):
        raise RefractionStaticTraceStoreApplyError(
            'source TraceStore derived.components must be a list'
        )
    output: list[dict[str, object]] = []
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise RefractionStaticTraceStoreApplyError(
                f'source TraceStore derived.components[{index}] must be an object'
            )
        output.append(dict(component))
    return output


def _status_counts(statuses: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(statuses, dtype=str), return_counts=True)
    return {str(status): int(count) for status, count in zip(unique, counts, strict=True)}


def _stat(values: np.ndarray, name: str) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if name == 'min':
        return float(np.min(arr))
    if name == 'max':
        return float(np.max(arr))
    if name == 'median':
        return float(np.median(arr))
    if name == 'p95':
        return float(np.percentile(arr, 95.0))
    raise ValueError(f'unsupported stat: {name}')


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    if not path.exists():
        raise RefractionStaticTraceStoreApplyError(f'{label} is missing: {path}')
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise RefractionStaticTraceStoreApplyError(f'{label} is invalid: {path}') from exc
    if not isinstance(payload, dict):
        raise RefractionStaticTraceStoreApplyError(f'{label} must be an object')
    return payload


def _cleanup_registration(
    state: AppState,
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


def _cleanup_store(output_path: Path) -> None:
    for tmp_path in output_path.parent.glob(f'{output_path.name}.tmp-*'):
        if tmp_path.is_dir():
            shutil.rmtree(tmp_path, ignore_errors=True)
    if output_path.exists():
        shutil.rmtree(output_path, ignore_errors=True)


def _cleanup_artifact(path: Path) -> None:
    path.unlink(missing_ok=True)
    for tmp_path in path.parent.glob(f'{path.name}.*.tmp'):
        tmp_path.unlink(missing_ok=True)


def _require_1d_float64(
    value: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    allow_nonfinite: bool = False,
) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        raise RefractionStaticTraceStoreApplyError(f'{name} must not have object dtype')
    if not np.issubdtype(arr.dtype, np.floating):
        raise RefractionStaticTraceStoreApplyError(f'{name} must be a float array')
    return coerce_1d_real_numeric_float64(
        arr,
        name=name,
        expected_shape=expected_shape,
        allow_nonfinite=allow_nonfinite,
        error_type=RefractionStaticTraceStoreApplyError,
    )


def _require_1d_bool(
    value: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    return coerce_1d_bool_array(
        value,
        name=name,
        expected_shape=expected_shape,
        error_type=RefractionStaticTraceStoreApplyError,
    )


def _require_1d_string(
    value: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    return coerce_1d_string_array(
        value,
        name=name,
        expected_shape=expected_shape,
        reject_object_dtype=True,
        output_dtype=str,
        error_type=RefractionStaticTraceStoreApplyError,
    )


def _reject_object_dtype(arr: np.ndarray, *, name: str) -> None:
    if arr.dtype == object:
        raise RefractionStaticTraceStoreApplyError(f'{name} must not have object dtype')


__all__ = [
    'CORRECTED_FILE_JSON_NAME',
    'REFRACTION_STATIC_APPLY_QC_JSON_NAME',
    'LoadedRefractionStaticSolutionForApply',
    'RefractionStaticApplyTraceStoreResult',
    'RefractionStaticTraceStoreApplyError',
    'RefractionTraceShiftValidationResult',
    'apply_refraction_statics_from_solution_artifact',
    'apply_refraction_statics_to_trace_store',
    'apply_trace_shifts_to_array',
    'load_refraction_static_solution_for_apply',
    'validate_refraction_trace_shifts_for_application',
]
