"""Safe resolution of persisted background-job artifact files."""

from __future__ import annotations

from pathlib import Path

from app.core.state import AppState


def _plain_artifact_name(name: str) -> str:
    if not name or name in {'.', '..'} or Path(name).name != name:
        raise ValueError('artifact name must be a plain file name')
    return name


def _job_type(job: dict[str, object]) -> str | None:
    raw = job.get('job_type')
    if isinstance(raw, str) and raw:
        return raw
    if 'pipeline_key' in job:
        return 'pipeline'
    return None


def resolve_job_artifact_path(
    state: AppState,
    *,
    job_id: str,
    name: str,
    allowed_job_types: set[str] | None = None,
    allowed_statics_kinds: set[str] | None = None,
) -> Path:
    """Resolve a job artifact by job id and file name without accepting paths."""
    artifact_name = _plain_artifact_name(name)
    with state.lock:
        raw_job = state.jobs.get(job_id)
        job = dict(raw_job) if isinstance(raw_job, dict) else None
    if job is None:
        raise ValueError(f'job_id not found: {job_id}')

    job_type = _job_type(job)
    if allowed_job_types is not None and job_type not in allowed_job_types:
        raise ValueError(f'job {job_id} has unsupported job_type: {job_type}')

    if allowed_statics_kinds is not None:
        if job_type != 'statics':
            raise ValueError(f'job {job_id} is not a statics job')
        statics_kind = job.get('statics_kind')
        if statics_kind not in allowed_statics_kinds:
            raise ValueError(
                f'job {job_id} has unsupported statics_kind: {statics_kind}'
            )

    artifacts_dir_raw = job.get('artifacts_dir')
    if not isinstance(artifacts_dir_raw, str) or not artifacts_dir_raw:
        raise ValueError(f'job {job_id} has no artifacts_dir')

    artifacts_dir = Path(artifacts_dir_raw)
    if not artifacts_dir.is_dir():
        raise ValueError(f'job artifacts_dir is not a directory: {artifacts_dir}')

    artifact_path = artifacts_dir / artifact_name
    if not artifact_path.is_file():
        raise ValueError(f'job artifact not found: {artifact_name}')
    return artifact_path


__all__ = ['resolve_job_artifact_path']
