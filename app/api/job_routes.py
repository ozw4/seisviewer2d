"""Shared API helpers for job status and cancellation routes."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import HTTPException

from app.core.state import AppState
from app.services.job_manager import JobManager
from app.services.job_runner import request_job_cancel


def infer_job_type(job: Mapping[str, object]) -> str | None:
    """Infer the API job family from existing job metadata conventions."""
    job_type = job.get('job_type')
    if isinstance(job_type, str):
        return job_type
    if 'pipeline_key' in job:
        return 'pipeline'
    return None


def get_job_or_404(
    state: AppState,
    job_id: str,
    *,
    allowed_job_types: set[str] | None = None,
    missing_detail: str = 'Job ID not found',
) -> dict[str, object]:
    """Return a copied job state or raise the route-compatible 404."""
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=missing_detail)
        job_copy = dict(job)

    if allowed_job_types is not None and infer_job_type(job_copy) not in allowed_job_types:
        raise HTTPException(status_code=404, detail='Job ID not found')
    return job_copy


def job_status_payload(job: Mapping[str, object]) -> dict[str, object]:
    """Build the common status payload used by job lifecycle routes."""
    progress = job.get('progress', 0.0)
    message = job.get('message', '')
    return {
        'state': JobManager.normalize_status_value(job.get('status', 'unknown')),
        'progress': float(progress) if isinstance(progress, (int, float)) else 0.0,
        'message': message if isinstance(message, str) else '',
    }


def cancel_job_and_get_status_payload(
    state: AppState,
    job_id: str,
    *,
    allowed_job_types: set[str] | None = None,
) -> dict[str, object]:
    """Request cancellation after validating the job, then return current status."""
    get_job_or_404(state, job_id, allowed_job_types=allowed_job_types)
    request_job_cancel(state, job_id)
    job = get_job_or_404(state, job_id, allowed_job_types=allowed_job_types)
    return job_status_payload(job)


__all__ = [
    'cancel_job_and_get_status_payload',
    'get_job_or_404',
    'infer_job_type',
    'job_status_payload',
]
