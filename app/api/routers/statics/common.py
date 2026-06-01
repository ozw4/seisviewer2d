"""Shared helpers for static correction routers."""

from __future__ import annotations

from fastapi import HTTPException

from app.core.state import AppState


def _get_static_job_or_404(state: AppState, job_id: str) -> dict[str, object]:
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Job ID not found')
        if job.get('job_type') != 'statics':
            raise HTTPException(status_code=404, detail='Job ID not found')
        return dict(job)
