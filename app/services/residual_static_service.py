"""Residual static correction background job service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from app.contracts.statics.residual import ResidualStaticApplyRequest
from app.services.common.artifact_io import write_json_atomic
from app.core.state import AppState
from app.services.job_runner import (
    JobCancelledError,
    JobCompletion,
    JobFailure,
    ensure_job_not_cancelled,
    run_job_with_lifecycle,
)
from app.services.pipeline_artifacts import get_job_dir
from app.services.reader import get_reader
from app.services.residual_static_artifacts import (
    QC_JSON_NAME,
    SOLUTION_NPZ_NAME,
    STATICS_CSV_NAME,
    ResidualStaticArtifactMetadata,
    write_residual_static_artifacts,
)
from app.services.residual_static_corrected_store import (
    ResidualStaticTraceStoreApplyOptions,
    apply_residual_static_correction_to_trace_store,
)
from app.services.residual_static_inputs import (
    build_residual_static_solver_inputs,
    load_residual_static_pick_source,
    resolve_residual_static_input_artifacts,
)
from app.services.residual_static_robust_solver import (
    robust_options_from_request_robust,
)
from app.services.residual_static_sparse_solver import (
    stabilization_options_from_request_solver,
)
from seis_statics.residual import (
    ResidualStaticRobustSolveResult,
    ResidualStaticSolverInputs,
    solve_first_break_residual_statics,
)

_CORRECTED_FILE_NAME = 'corrected_file.json'


@dataclass
class _ResidualStaticLifecycle:
    created_ts: float
    job_dir: Path | None = None
    corrected_file_id: str | None = None
    corrected_store_path: Path | None = None


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


def _write_job_meta(
    *,
    job_id: str,
    job_dir: Path,
    req: ResidualStaticApplyRequest,
) -> None:
    write_json_atomic(
        job_dir / 'job_meta.json',
        {
            'job_id': job_id,
            'job_type': 'statics',
            'statics_kind': 'residual',
            'source_file_id': req.file_id,
            'key1_byte': req.key1_byte,
            'key2_byte': req.key2_byte,
            'request': req.model_dump(mode='json'),
            'inputs': {
                'datum_solution': req.datum_solution.model_dump(mode='json'),
                'pick_source': req.pick_source.model_dump(
                    mode='json',
                    exclude_none=True,
                ),
                'geometry': req.geometry.model_dump(mode='json'),
                'offset': req.offset.model_dump(mode='json'),
                'moveout': req.moveout.model_dump(mode='json'),
            },
            'solver': req.solver.model_dump(mode='json'),
            'robust': req.robust.model_dump(mode='json'),
            'apply': req.apply.model_dump(mode='json'),
            'artifacts': {
                'solution_npz': SOLUTION_NPZ_NAME,
                'qc_json': QC_JSON_NAME,
                'statics_csv': STATICS_CSV_NAME,
                'corrected_file_json': _CORRECTED_FILE_NAME,
            },
        },
        make_parent=True,
    )


def _reader_n_samples(reader: object) -> int:
    getter = getattr(reader, 'get_n_samples', None)
    if callable(getter):
        n_samples = int(getter())
    else:
        traces = getattr(reader, 'traces', None)
        shape = getattr(traces, 'shape', ())
        if len(shape) < 2:
            raise ValueError('reader cannot provide number of samples')
        n_samples = int(shape[-1])
    if n_samples <= 0:
        raise ValueError('reader n_samples must be greater than 0')
    return n_samples


def _resolve_dt(state: AppState, file_id: str) -> float:
    dt = float(state.file_registry.get_dt(file_id))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError('dt must be finite and greater than 0')
    return dt


def _apply_options_from_request(
    req: ResidualStaticApplyRequest,
) -> ResidualStaticTraceStoreApplyOptions:
    return ResidualStaticTraceStoreApplyOptions(
        interpolation=req.apply.interpolation,
        fill_value=req.apply.fill_value,
        max_abs_shift_ms=req.apply.max_abs_shift_ms,
        output_dtype=req.apply.output_dtype,
        register_corrected_file=req.apply.register_corrected_file,
    )


def _solve_residual_static_with_package_api(
    inputs: ResidualStaticSolverInputs,
    req: ResidualStaticApplyRequest,
) -> ResidualStaticRobustSolveResult:
    result = solve_first_break_residual_statics(
        solver_inputs=inputs,
        stabilization_options=stabilization_options_from_request_solver(req.solver),
        robust_options=robust_options_from_request_robust(req.robust),
    )
    return result.robust_solve_result


def _run_residual_static_apply_job_body(
    job_id: str,
    req: ResidualStaticApplyRequest,
    state: AppState,
    *,
    lifecycle: _ResidualStaticLifecycle,
) -> JobCompletion | None:
    job_dir = _resolve_job_dir(state, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    lifecycle.job_dir = job_dir

    _set_job_progress_message(
        state,
        job_id,
        progress=0.02,
        message='preparing_residual_static_job',
    )
    ensure_job_not_cancelled(state, job_id)
    _write_job_meta(job_id=job_id, job_dir=job_dir, req=req)
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.05,
        message='resolving_residual_static_inputs',
    )
    reader = get_reader(req.file_id, req.key1_byte, req.key2_byte, state=state)
    source_store_path = Path(state.file_registry.get_store_path(req.file_id))
    dt = _resolve_dt(state, req.file_id)
    n_samples = _reader_n_samples(reader)
    artifacts = resolve_residual_static_input_artifacts(state, req)
    ensure_job_not_cancelled(state, job_id)
    pick_source = load_residual_static_pick_source(
        req=req,
        artifacts=artifacts,
        reader=reader,
        expected_dt=dt,
        expected_n_samples=n_samples,
        state=state,
    )
    ensure_job_not_cancelled(state, job_id)
    inputs = build_residual_static_solver_inputs(
        req=req,
        artifacts=artifacts,
        pick_source=pick_source,
        reader=reader,
        expected_dt=dt,
        expected_n_samples=n_samples,
    )
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.20,
        message='solving_residual_static_delays',
    )
    ensure_job_not_cancelled(state, job_id)
    robust_result = _solve_residual_static_with_package_api(inputs, req)
    ensure_job_not_cancelled(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.45,
        message='writing_residual_static_artifacts',
    )
    ensure_job_not_cancelled(state, job_id)
    artifact_paths = write_residual_static_artifacts(
        job_dir,
        inputs,
        robust_result,
        metadata=ResidualStaticArtifactMetadata(
            job_id=job_id,
            input_file_id=req.file_id,
            datum_source_file_id=artifacts.datum_source_file_id,
            datum_job_id=artifacts.datum_job_id,
            datum_solution_artifact=req.datum_solution.name,
            pick_source_kind=inputs.pick_source_kind,
            pick_source_artifact=artifacts.pick_source_artifact_name,
        ),
    )
    ensure_job_not_cancelled(state, job_id)

    def progress_callback(apply_progress: float, message: str) -> None:
        progress_value = max(0.0, min(1.0, float(apply_progress)))
        if message == 'registering_residual_corrected_trace_store':
            mapped_progress = 0.90
            mapped_message = message
        else:
            mapped_progress = 0.60 + (0.30 * progress_value)
            mapped_message = 'applying_residual_static_trace_shift'
        _set_job_progress_message(
            state,
            job_id,
            progress=mapped_progress,
            message=mapped_message,
        )

    def cancel_check() -> bool:
        return _is_cancel_requested(state, job_id)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.60,
        message='applying_residual_static_trace_shift',
    )
    ensure_job_not_cancelled(state, job_id)
    try:
        corrected_result = apply_residual_static_correction_to_trace_store(
            source_file_id=req.file_id,
            source_store_path=source_store_path,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            residual_solution_npz_path=artifact_paths.solution_npz_path,
            artifacts_dir=job_dir,
            job_id=job_id,
            state=state,
            options=_apply_options_from_request(req),
            progress_callback=progress_callback,
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
        progress=0.90,
        message='registering_residual_corrected_trace_store',
    )
    with state.lock:
        state.jobs.set_static_corrected_file(
            job_id,
            corrected_file_id=corrected_result.file_id,
            corrected_store_path=str(corrected_result.store_path),
        )
    _set_job_progress_message(state, job_id, progress=1.0, message='done')
    return JobCompletion(finished_ts=time.time())


def _handle_residual_static_job_error(
    *,
    lifecycle: _ResidualStaticLifecycle,
) -> JobFailure:
    return JobFailure(finished_ts=time.time())


def _handle_residual_static_job_cancel(
    *,
    lifecycle: _ResidualStaticLifecycle,
    exc: JobCancelledError,
) -> JobCompletion:
    finished_ts = float(exc.finished_ts) if exc.finished_ts is not None else time.time()
    return JobCompletion(finished_ts=finished_ts)


def run_residual_static_apply_job(
    job_id: str,
    req: ResidualStaticApplyRequest,
    state: AppState,
) -> None:
    """Run one residual static correction job and register the corrected TraceStore."""
    lifecycle = _ResidualStaticLifecycle(created_ts=_resolve_created_ts(state, job_id))

    def worker() -> JobCompletion | None:
        return _run_residual_static_apply_job_body(
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
        on_error=lambda _exc: _handle_residual_static_job_error(
            lifecycle=lifecycle,
        ),
        on_cancel=lambda exc: _handle_residual_static_job_cancel(
            lifecycle=lifecycle,
            exc=exc,
        ),
    )


__all__ = ['run_residual_static_apply_job']
