"""M5 refraction export request validation and metadata-only job service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from app.api.schemas import (
    REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS,
    RefractionStaticExportFormat,
    RefractionStaticExportJobRequest,
    RefractionStaticExportRequest,
)
from app.core.state import AppState
from app.services.job_manager import JobManager
from app.services.job_runner import JobCompletion, JobFailure, run_job_with_lifecycle
from app.services.refraction_static_artifacts import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_REQUEST_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)
from app.services.refraction_static_export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)

REFRACTION_STATIC_EXPORT_REQUEST_JSON_NAME = 'refraction_static_export_request.json'
REFRACTION_STATIC_EXPORT_JOB_META_JSON_NAME = 'job_meta.json'
REFRACTION_STATIC_EXPORT_DONE_MESSAGE = 'refraction_static_export_contract_recorded'

_BASE_SOURCE_ARTIFACTS = (
    REFRACTION_STATIC_REQUEST_JSON_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
)

_FORMAT_SOURCE_ARTIFACTS: dict[
    RefractionStaticExportFormat,
    tuple[str, ...],
] = {
    'canonical_static_table': (
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
    ),
    'lsst': (
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
    ),
    'lsst_plus': (
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
    ),
    'time_term_spreadsheet': (
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    ),
    'first_break_time': (
        FIRST_BREAK_RESIDUALS_CSV_NAME,
        REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    ),
}


class RefractionStaticExportSourceJobNotFound(ValueError):
    """Raised when a standalone export references an unknown source job."""


class RefractionStaticExportValidationError(ValueError):
    """Raised when a standalone export source job is not export-compatible."""


@dataclass(frozen=True)
class ResolvedRefractionStaticExportSourceJob:
    source_job_id: str
    source_file_id: str
    key1_byte: int
    key2_byte: int
    source_artifacts_dir: Path
    requested_formats: tuple[RefractionStaticExportFormat, ...]
    required_source_artifacts: tuple[str, ...]


def resolve_refraction_static_export_formats(
    export: RefractionStaticExportRequest,
) -> tuple[RefractionStaticExportFormat, ...]:
    """Resolve default M5 export formats without running any formatter."""
    if not bool(export.enabled):
        return ()
    if export.formats:
        return tuple(export.formats)
    return REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS


def required_refraction_static_export_source_artifacts(
    requested_formats: tuple[RefractionStaticExportFormat, ...],
) -> tuple[str, ...]:
    """Return source refraction artifacts required for the requested formats."""
    if not requested_formats:
        return ()
    names: list[str] = list(_BASE_SOURCE_ARTIFACTS)
    for export_format in requested_formats:
        names.extend(_FORMAT_SOURCE_ARTIFACTS[export_format])
    return tuple(dict.fromkeys(names))


def validate_refraction_static_export_source_job(
    *,
    req: RefractionStaticExportJobRequest,
    state: AppState,
) -> ResolvedRefractionStaticExportSourceJob:
    """Validate the completed source refraction job used by standalone export."""
    requested_formats = resolve_refraction_static_export_formats(req.export)
    required_artifacts = required_refraction_static_export_source_artifacts(
        requested_formats,
    )
    with state.lock:
        raw_job = state.jobs.get(req.source_job_id)
        job = dict(raw_job) if isinstance(raw_job, dict) else None
    if job is None:
        raise RefractionStaticExportSourceJobNotFound(
            f'source refraction job not found: {req.source_job_id}'
        )

    if job.get('job_type') != 'statics' or job.get('statics_kind') != 'refraction':
        raise RefractionStaticExportValidationError(
            'source_job_id must reference a refraction static job'
        )
    status = JobManager.normalize_status_value(job.get('status'))
    if status != 'done':
        raise RefractionStaticExportValidationError(
            'source_job_id must reference a completed refraction static job'
        )

    source_file_id = job.get('file_id')
    if not isinstance(source_file_id, str) or not source_file_id:
        raise RefractionStaticExportValidationError(
            'source refraction job metadata is missing file_id'
        )
    key1_byte = _job_int(job, 'key1_byte')
    key2_byte = _job_int(job, 'key2_byte')

    artifacts_dir_raw = job.get('artifacts_dir')
    if not isinstance(artifacts_dir_raw, str) or not artifacts_dir_raw:
        raise RefractionStaticExportValidationError(
            'source refraction job metadata is missing artifacts_dir'
        )
    artifacts_dir = Path(artifacts_dir_raw)
    if not artifacts_dir.is_dir():
        raise RefractionStaticExportValidationError(
            'source refraction job artifacts_dir is missing'
        )

    missing = [
        name for name in required_artifacts if not (artifacts_dir / name).is_file()
    ]
    if missing:
        missing_text = ', '.join(missing)
        raise RefractionStaticExportValidationError(
            'source refraction job is missing required export artifacts: '
            f'{missing_text}'
        )

    return ResolvedRefractionStaticExportSourceJob(
        source_job_id=req.source_job_id,
        source_file_id=source_file_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        source_artifacts_dir=artifacts_dir,
        requested_formats=requested_formats,
        required_source_artifacts=required_artifacts,
    )


def run_refraction_static_export_job(
    job_id: str,
    req: RefractionStaticExportJobRequest,
    state: AppState,
) -> None:
    """Record a standalone M5 export contract job.

    Formatter implementation is intentionally out of scope for issue #499; this
    job validates the source refraction artifacts and persists the public export
    request metadata, including resolved default formats.
    """

    def worker() -> JobCompletion:
        return _run_refraction_static_export_job_body(
            job_id=job_id,
            req=req,
            state=state,
        )

    run_job_with_lifecycle(
        state=state,
        job_id=job_id,
        worker=worker,
        progress_1_on_done=False,
        start_progress=0.0,
        clear_message_on_start=True,
        on_error=_handle_refraction_static_export_job_error,
    )


def _run_refraction_static_export_job_body(
    *,
    job_id: str,
    req: RefractionStaticExportJobRequest,
    state: AppState,
) -> JobCompletion:
    _set_job_progress_message(
        state,
        job_id,
        progress=0.10,
        message='validating_source_refraction_static_job',
    )
    source = validate_refraction_static_export_source_job(req=req, state=state)
    job_dir = _resolve_job_dir(state, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = _refraction_static_export_job_payload(
        job_id=job_id,
        req=req,
        source=source,
    )
    _set_job_progress_message(
        state,
        job_id,
        progress=0.80,
        message='writing_refraction_static_export_request',
    )
    _write_json_atomic(job_dir / REFRACTION_STATIC_EXPORT_JOB_META_JSON_NAME, payload)
    _write_json_atomic(job_dir / REFRACTION_STATIC_EXPORT_REQUEST_JSON_NAME, payload)
    _set_job_progress_message(
        state,
        job_id,
        progress=1.0,
        message=REFRACTION_STATIC_EXPORT_DONE_MESSAGE,
    )
    return JobCompletion(
        finished_ts=time.time(),
        message=REFRACTION_STATIC_EXPORT_DONE_MESSAGE,
    )


def _refraction_static_export_job_payload(
    *,
    job_id: str,
    req: RefractionStaticExportJobRequest,
    source: ResolvedRefractionStaticExportSourceJob,
) -> dict[str, Any]:
    return {
        'job_id': job_id,
        'job_type': 'statics',
        'statics_kind': 'refraction_export',
        'source_job_id': source.source_job_id,
        'source_file_id': source.source_file_id,
        'key1_byte': source.key1_byte,
        'key2_byte': source.key2_byte,
        'request': req.model_dump(mode='json'),
        'export': {
            'enabled': bool(req.export.enabled),
            'requested_formats': list(source.requested_formats),
            'units': req.export.units,
            'rounding_ms': req.export.rounding_ms,
            'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
            'include_inactive_endpoints': bool(req.export.include_inactive_endpoints),
            'include_legacy_alias_columns': bool(
                req.export.include_legacy_alias_columns
            ),
            'fail_on_invalid_static_status': bool(
                req.export.fail_on_invalid_static_status
            ),
        },
        'required_source_artifacts': list(source.required_source_artifacts),
        'source_artifacts_dir': str(source.source_artifacts_dir),
    }


def _job_int(job: dict[str, object], field: str) -> int:
    value = job.get(field)
    if not isinstance(value, int):
        raise RefractionStaticExportValidationError(
            f'source refraction job metadata is missing {field}'
        )
    return int(value)


def _resolve_job_dir(state: AppState, job_id: str) -> Path:
    with state.lock:
        job = state.jobs.get(job_id)
        artifacts_dir = job.get('artifacts_dir') if isinstance(job, dict) else None
    if not isinstance(artifacts_dir, str) or not artifacts_dir:
        raise ValueError('job artifacts_dir is not available')
    return Path(artifacts_dir)


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


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding='utf-8',
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _handle_refraction_static_export_job_error(_exc: Exception) -> JobFailure:
    return JobFailure(finished_ts=time.time())


__all__ = [
    'REFRACTION_STATIC_EXPORT_DONE_MESSAGE',
    'REFRACTION_STATIC_EXPORT_JOB_META_JSON_NAME',
    'REFRACTION_STATIC_EXPORT_REQUEST_JSON_NAME',
    'RefractionStaticExportSourceJobNotFound',
    'RefractionStaticExportValidationError',
    'ResolvedRefractionStaticExportSourceJob',
    'required_refraction_static_export_source_artifacts',
    'resolve_refraction_static_export_formats',
    'run_refraction_static_export_job',
    'validate_refraction_static_export_source_job',
]
