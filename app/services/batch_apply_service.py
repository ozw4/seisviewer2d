"""Background runner for batch apply jobs."""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.api.schemas import BatchApplyRequest
from app.core.state import AppState
from app.services.pipeline_artifacts import get_job_dir


def _write_job_meta(*, job_dir: Path, payload: dict[str, object]) -> Path:
    """Write ``job_meta.json`` directly under the job directory."""
    job_meta_path = job_dir / 'job_meta.json'
    job_meta_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True),
        encoding='utf-8',
    )
    return job_meta_path


def run_batch_apply_job(job_id: str, req: BatchApplyRequest, state: AppState) -> None:
    """Run one batch job with a minimal placeholder artifact output."""
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            return
        created_ts_obj = job.get('created_ts')
        created_ts = (
            float(created_ts_obj)
            if isinstance(created_ts_obj, (int, float))
            else time.time()
        )
        job['status'] = 'running'
        job['progress'] = 0.0

    try:
        job_dir = get_job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        finished_ts = time.time()
        meta_payload: dict[str, object] = {
            'job_id': job_id,
            'created_ts': created_ts,
            'finished_ts': finished_ts,
            'request': req.model_dump(mode='json'),
        }
        _write_job_meta(job_dir=job_dir, payload=meta_payload)
        with state.lock:
            job = state.jobs.get(job_id)
            if job is not None:
                job['status'] = 'done'
                job['progress'] = 1.0
                job['finished_ts'] = finished_ts
    except Exception as e:  # noqa: BLE001
        with state.lock:
            job = state.jobs.get(job_id)
            if job is not None:
                job['status'] = 'error'
                job['message'] = str(e)
                job['finished_ts'] = time.time()


__all__ = ['run_batch_apply_job']
