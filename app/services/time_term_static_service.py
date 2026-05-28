"""Validation and orchestration for time-term static inversion jobs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any
from uuid import uuid4

import numpy as np

from app.contracts.statics.time_term import TimeTermStaticApplyRequest
from app.core.state import AppState
from app.services.errors import DomainError
from app.services.geometry_linkage_loader import load_geometry_linkage_artifact
from app.services.job_artifact_refs import resolve_job_artifact_path
from app.services.job_runner import (
    JobCancelledError,
    JobCompletion,
    JobFailure,
    ensure_job_not_cancelled,
    run_job_with_lifecycle,
)
from app.services.pipeline_artifacts import get_job_dir
from app.services.reader import get_reader
from app.services.time_term_apply_shift import (
    TimeTermAppliedShiftOptions,
    build_time_term_applied_shift_result,
)
from app.services.time_term_design_matrix import (
    TimeTermDesignMatrixOptions,
    build_time_term_design_matrix,
)
from app.services.time_term_moveout import (
    TimeTermMoveoutConfig,
    compute_time_term_moveout,
)
from app.services.time_term_robust_solver import (
    TimeTermRobustSolverOptions,
    solve_time_term_robust_least_squares,
)
from app.services.time_term_sparse_solver import TimeTermSparseSolverOptions
from app.services.time_term_static_apply_trace_store import (
    TimeTermTraceStoreApplyOptions,
    apply_time_term_static_correction_to_trace_store,
)
from app.services.time_term_static_artifacts import (
    TIME_TERM_STATIC_QC_JSON_NAME,
    TIME_TERM_STATIC_SOLUTION_NPZ_NAME,
    TIME_TERM_STATICS_CSV_NAME,
    TimeTermStaticArtifactMetadata,
    write_time_term_static_artifacts,
)
from app.services.time_term_static_inputs import build_time_term_inversion_inputs
from app.services.trace_store_registration import trace_store_cache_key

_CORRECTED_FILE_NAME = 'corrected_file.json'


@dataclass(frozen=True)
class TimeTermValidationResult:
    """Validated shallow inputs for a future time-term solver job."""

    file_id: str
    dt: float
    n_traces: int
    linkage_artifact_path: Path | None


@dataclass
class _TimeTermStaticLifecycle:
    created_ts: float
    job_dir: Path | None = None
    corrected_store_path: Path | None = None
    corrected_file_id: str | None = None
    key1_byte: int | None = None
    key2_byte: int | None = None


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


def _resolve_created_ts(state: AppState, job_id: str) -> float:
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            return time.time()
        created_ts_obj = job.get('created_ts')
    return (
        float(created_ts_obj)
        if isinstance(created_ts_obj, (int, float))
        else time.time()
    )


def _resolve_job_dir(state: AppState, job_id: str) -> Path:
    with state.lock:
        job = state.jobs.get(job_id)
        artifacts_dir = job.get('artifacts_dir') if isinstance(job, dict) else None
    if isinstance(artifacts_dir, str) and artifacts_dir:
        return Path(artifacts_dir)
    return get_job_dir(job_id)


def _set_job_progress_message(
    state: AppState,
    job_id: str,
    *,
    progress: float,
    message: str,
) -> None:
    with state.lock:
        if state.jobs.get(job_id) is None:
            return
        state.jobs.set_progress(job_id, progress)
        state.jobs.set_message(job_id, message)


def _is_cancel_requested(state: AppState, job_id: str) -> bool:
    with state.lock:
        return state.jobs.is_cancel_requested(job_id)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        tmp_path.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
            ),
            encoding='utf-8',
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_time_term_job_meta(
    *,
    job_id: str,
    job_dir: Path,
    req: TimeTermStaticApplyRequest,
) -> None:
    _write_json_atomic(
        job_dir / 'job_meta.json',
        {
            'job_id': job_id,
            'job_type': 'statics',
            'statics_kind': 'time_term',
            'source_file_id': req.file_id,
            'key1_byte': req.key1_byte,
            'key2_byte': req.key2_byte,
            'request': req.model_dump(mode='json'),
            'inputs': {
                'pick_source': req.pick_source.model_dump(
                    mode='json',
                    exclude_none=True,
                ),
                'geometry': req.geometry.model_dump(mode='json'),
                'linkage': req.linkage.model_dump(mode='json', exclude_none=True),
                'velocity': req.velocity.model_dump(mode='json'),
                'moveout': req.moveout.model_dump(mode='json'),
                'solver': req.solver.model_dump(mode='json'),
                'apply': req.apply.model_dump(mode='json'),
            },
            'artifacts': {
                'solution_npz': TIME_TERM_STATIC_SOLUTION_NPZ_NAME,
                'qc_json': TIME_TERM_STATIC_QC_JSON_NAME,
                'statics_csv': TIME_TERM_STATICS_CSV_NAME,
                'corrected_file_json': _CORRECTED_FILE_NAME,
            },
        },
    )


def _moveout_config_from_time_term_request(
    req: TimeTermStaticApplyRequest,
) -> TimeTermMoveoutConfig:
    return TimeTermMoveoutConfig(
        model=req.moveout.model,
        refractor_velocity_m_s=req.velocity.refractor_velocity_m_s,
        distance_source=req.moveout.distance_source,
        offset_byte=req.moveout.offset_byte,
        allow_missing_offset=req.moveout.allow_missing_offset,
        max_geometry_offset_mismatch_m=req.moveout.max_geometry_offset_mismatch_m,
    )


def _sparse_solver_options_from_time_term_request(
    req: TimeTermStaticApplyRequest,
) -> TimeTermSparseSolverOptions:
    return TimeTermSparseSolverOptions(
        damping_lambda=req.solver.damping,
        gauge=req.solver.gauge,
        reference_node_id=(
            req.solver.reference_node_id
            if req.solver.gauge == 'reference_node'
            else None
        ),
        min_observations=req.solver.robust.min_used_observations,
        max_abs_node_time_term_ms=req.apply.max_abs_shift_ms,
        max_abs_estimated_trace_delay_ms=req.apply.max_abs_shift_ms,
    )


def _robust_solver_options_from_time_term_request(
    req: TimeTermStaticApplyRequest,
) -> TimeTermRobustSolverOptions:
    robust = req.solver.robust
    return TimeTermRobustSolverOptions(
        enabled=robust.enabled,
        method=robust.method,
        threshold=robust.threshold,
        max_iterations=robust.max_iterations,
        min_used_fraction=robust.min_used_fraction,
        min_used_observations=robust.min_used_observations,
    )


def _applied_shift_options_from_time_term_request(
    req: TimeTermStaticApplyRequest,
) -> TimeTermAppliedShiftOptions:
    return TimeTermAppliedShiftOptions(
        max_abs_weathering_shift_ms=req.apply.max_abs_shift_ms,
        max_abs_final_shift_ms=req.apply.max_abs_shift_ms,
        rejected_trace_policy='use_final_model',
    )


def _trace_store_apply_options_from_time_term_request(
    req: TimeTermStaticApplyRequest,
) -> TimeTermTraceStoreApplyOptions:
    return TimeTermTraceStoreApplyOptions(
        mode=req.apply.mode,
        interpolation=req.apply.interpolation,
        fill_value=req.apply.fill_value,
        output_dtype=req.apply.output_dtype,
        max_abs_shift_ms=req.apply.max_abs_shift_ms,
        register_corrected_file=req.apply.register_corrected_file,
    )


def _cleanup_corrected_outputs(
    state: AppState,
    lifecycle: _TimeTermStaticLifecycle,
) -> None:
    if lifecycle.corrected_file_id is not None:
        with state.lock:
            state.file_registry.pop(lifecycle.corrected_file_id, None)
            if lifecycle.key1_byte is not None and lifecycle.key2_byte is not None:
                state.cached_readers.pop(
                    trace_store_cache_key(
                        lifecycle.corrected_file_id,
                        lifecycle.key1_byte,
                        lifecycle.key2_byte,
                    ),
                    None,
                )
    if lifecycle.corrected_store_path is not None:
        shutil.rmtree(lifecycle.corrected_store_path, ignore_errors=True)
    if lifecycle.job_dir is not None:
        (lifecycle.job_dir / _CORRECTED_FILE_NAME).unlink(missing_ok=True)


def _time_term_artifact_metadata(
    *,
    job_id: str,
    req: TimeTermStaticApplyRequest,
    inputs: object,
) -> TimeTermStaticArtifactMetadata:
    return TimeTermStaticArtifactMetadata(
        job_id=job_id,
        input_file_id=req.file_id,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
        request=req.model_dump(mode='json'),
        pick_source_description=getattr(inputs, 'pick_source_description', None),
        datum_solution_path=(
            str(getattr(inputs, 'datum_solution_path'))
            if getattr(inputs, 'datum_solution_path', None) is not None
            else None
        ),
        residual_solution_path=(
            str(getattr(inputs, 'residual_solution_path'))
            if getattr(inputs, 'residual_solution_path', None) is not None
            else None
        ),
        linkage_artifact_path=(
            str(getattr(inputs, 'linkage_artifact_path'))
            if getattr(inputs, 'linkage_artifact_path', None) is not None
            else None
        ),
    )


def _run_time_term_static_apply_job_body(
    *,
    job_id: str,
    req: TimeTermStaticApplyRequest,
    state: AppState,
    lifecycle: _TimeTermStaticLifecycle,
) -> JobCompletion | None:
    lifecycle.key1_byte = req.key1_byte
    lifecycle.key2_byte = req.key2_byte
    job_dir = _resolve_job_dir(state, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    lifecycle.job_dir = job_dir

    _set_job_progress_message(
        state,
        job_id,
        progress=0.02,
        message='preparing_time_term_static_job',
    )
    ensure_job_not_cancelled(state, job_id)
    _write_time_term_job_meta(job_id=job_id, job_dir=job_dir, req=req)
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.05,
        message='resolving_time_term_inputs',
    )
    inputs = build_time_term_inversion_inputs(request=req, state=state)
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.18,
        message='computing_time_term_moveout',
    )
    moveout = compute_time_term_moveout(
        inputs,
        _moveout_config_from_time_term_request(req),
    )
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.28,
        message='building_time_term_design_matrix',
    )
    design = build_time_term_design_matrix(
        inputs,
        moveout,
        options=TimeTermDesignMatrixOptions(
            min_observations=req.solver.robust.min_used_observations,
        ),
    )
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.40,
        message='solving_time_term_statics',
    )
    solver_result = solve_time_term_robust_least_squares(
        design,
        solver_options=_sparse_solver_options_from_time_term_request(req),
        robust_options=_robust_solver_options_from_time_term_request(req),
    )
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.58,
        message='building_time_term_applied_shifts',
    )
    applied_shift = build_time_term_applied_shift_result(
        inputs=inputs,
        solver_result=solver_result,
        options=_applied_shift_options_from_time_term_request(req),
    )
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.68,
        message='writing_time_term_artifacts',
    )
    artifact_paths = write_time_term_static_artifacts(
        job_dir=job_dir,
        inputs=inputs,
        moveout=moveout,
        design=design,
        solver_result=solver_result,
        applied_shift=applied_shift,
        metadata=_time_term_artifact_metadata(
            job_id=job_id,
            req=req,
            inputs=inputs,
        ),
    )
    ensure_job_not_cancelled(state, job_id)

    if not req.apply.register_corrected_file:
        _set_job_progress_message(state, job_id, progress=1.0, message='done')
        return JobCompletion(finished_ts=time.time())

    def cancel_check() -> bool:
        return _is_cancel_requested(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.80,
        message='applying_time_term_static_trace_shift',
    )
    ensure_job_not_cancelled(state, job_id)
    try:
        corrected_result = apply_time_term_static_correction_to_trace_store(
            source_file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            solution_npz_path=artifact_paths.solution_npz_path,
            artifacts_dir=job_dir,
            state=state,
            options=_trace_store_apply_options_from_time_term_request(req),
            cancel_check=cancel_check,
        )
    except RuntimeError as exc:
        if 'cancelled' in str(exc).lower() and cancel_check():
            raise JobCancelledError() from exc
        raise

    lifecycle.corrected_file_id = corrected_result.file_id
    lifecycle.corrected_store_path = corrected_result.store_path
    _set_job_progress_message(
        state,
        job_id,
        progress=0.93,
        message='registering_time_term_corrected_trace_store',
    )
    with state.lock:
        state.jobs.set_static_corrected_file(
            job_id,
            corrected_file_id=corrected_result.file_id,
            corrected_store_path=str(corrected_result.store_path),
        )
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(state, job_id, progress=1.0, message='done')
    return JobCompletion(finished_ts=time.time())


def _handle_time_term_static_job_error(
    *,
    state: AppState,
    lifecycle: _TimeTermStaticLifecycle,
) -> JobFailure:
    _cleanup_corrected_outputs(state, lifecycle)
    return JobFailure(finished_ts=time.time())


def _handle_time_term_static_job_cancel(
    *,
    state: AppState,
    lifecycle: _TimeTermStaticLifecycle,
    exc: JobCancelledError,
) -> JobCompletion:
    _cleanup_corrected_outputs(state, lifecycle)
    finished_ts = float(exc.finished_ts) if exc.finished_ts is not None else time.time()
    return JobCompletion(finished_ts=finished_ts)


def run_time_term_static_apply_job(
    job_id: str,
    req: TimeTermStaticApplyRequest,
    state: AppState,
) -> None:
    """Run one time-term static correction job."""
    lifecycle = _TimeTermStaticLifecycle(created_ts=_resolve_created_ts(state, job_id))

    def worker() -> JobCompletion | None:
        return _run_time_term_static_apply_job_body(
            job_id=job_id,
            req=req,
            state=state,
            lifecycle=lifecycle,
        )

    run_job_with_lifecycle(
        state=state,
        job_id=job_id,
        worker=worker,
        progress_1_on_done=True,
        start_progress=0.0,
        clear_message_on_start=True,
        on_error=lambda _exc: _handle_time_term_static_job_error(
            state=state,
            lifecycle=lifecycle,
        ),
        on_cancel=lambda exc: _handle_time_term_static_job_cancel(
            state=state,
            lifecycle=lifecycle,
            exc=exc,
        ),
    )
    with state.lock:
        job = state.jobs.get(job_id)
        if job is not None and job.get('status') == 'done':
            state.jobs.set_message(job_id, 'done')


__all__ = [
    'TimeTermValidationResult',
    'run_time_term_static_apply_job',
    'validate_time_term_request',
]
