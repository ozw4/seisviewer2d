"""In-memory job table cleanup helpers."""

from __future__ import annotations

import time

from app.core.state import AppState
from app.services.pipeline_artifacts import pipeline_jobs_ttl_seconds


def _job_timestamp(job: dict[str, object], field: str) -> float | None:
    value = job.get(field)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _job_created_ts(job: dict[str, object]) -> float:
    created_ts = _job_timestamp(job, 'created_ts')
    if created_ts is None:
        return 0.0
    return created_ts


def _job_expiry_ts(job: dict[str, object], now_ts: float) -> float:
    finished_ts = _job_timestamp(job, 'finished_ts')
    if finished_ts is not None:
        return finished_ts
    created_ts = _job_timestamp(job, 'created_ts')
    if created_ts is not None:
        return created_ts
    return now_ts


def _job_kind(job: dict[str, object]) -> str:
    if job.get('job_type') == 'batch_apply':
        return 'pipeline'
    if 'cache_key' in job:
        return 'fbpick'
    if 'pipeline_key' in job:
        return 'pipeline'
    return 'other'


def cleanup_in_memory_state(state: AppState) -> None:
    """Evict terminal jobs by TTL and cap the total in-memory jobs count."""
    now_ts = time.time()
    fbpick_ttl_sec = state.settings.sv_fbpick_job_ttl_sec
    jobs_max = state.settings.sv_jobs_max
    pipeline_ttl_sec = pipeline_jobs_ttl_seconds(settings=state.settings)

    with state.lock:
        jobs = state.jobs
        expired_job_ids: list[str] = []
        for job_id, raw_job in jobs.items():
            if not isinstance(raw_job, dict):
                continue
            status = raw_job.get('status')
            status_value = status.lower() if isinstance(status, str) else ''
            kind = _job_kind(raw_job)

            ttl_sec: int | None = None
            if kind == 'fbpick' and status_value in {'done', 'error', 'expired'}:
                ttl_sec = fbpick_ttl_sec
            elif kind == 'pipeline' and status_value in {'done', 'error'}:
                ttl_sec = pipeline_ttl_sec

            if ttl_sec is None:
                continue

            expiry_base_ts = _job_expiry_ts(raw_job, now_ts)
            if now_ts - expiry_base_ts > float(ttl_sec):
                expired_job_ids.append(job_id)

        for job_id in expired_job_ids:
            jobs.pop(job_id, None)

        overflow = len(jobs) - jobs_max
        if overflow <= 0:
            return

        oldest_first = sorted(jobs.items(), key=lambda item: _job_created_ts(item[1]))
        for job_id, _ in oldest_first[:overflow]:
            jobs.pop(job_id, None)


__all__ = ['cleanup_in_memory_state']
