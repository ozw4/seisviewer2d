"""Shared job runtime context helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time

from app.core.state import AppState
from app.services.job_runner import JobCancelledError, ensure_job_not_cancelled
from app.services.pipeline_artifacts import get_job_dir


@dataclass(frozen=True)
class JobContext:
    """Resolved runtime helpers for a background job."""

    state: AppState
    job_id: str
    job_dir: Path
    created_ts: float

    @classmethod
    def resolve(cls, state: AppState, job_id: str) -> JobContext:
        with state.lock:
            job = state.jobs.get(job_id)
            if job is None:
                created_ts_obj = None
                artifacts_dir = None
            else:
                created_ts_obj = job.get('created_ts')
                artifacts_dir = job.get('artifacts_dir')

        created_ts = (
            float(created_ts_obj)
            if isinstance(created_ts_obj, (int, float))
            else time.time()
        )
        job_dir = (
            Path(artifacts_dir)
            if isinstance(artifacts_dir, str) and artifacts_dir
            else get_job_dir(job_id)
        )
        return cls(
            state=state,
            job_id=job_id,
            job_dir=job_dir,
            created_ts=created_ts,
        )

    def set_progress_message(self, *, progress: float, message: str) -> bool:
        with self.state.lock:
            if self.state.jobs.get(self.job_id) is None:
                return False
            self.state.jobs.set_progress(self.job_id, progress)
            self.state.jobs.set_message(self.job_id, message)
            return True

    def set_progress(self, progress: float) -> bool:
        with self.state.lock:
            if self.state.jobs.get(self.job_id) is None:
                return False
            self.state.jobs.set_progress(self.job_id, progress)
            return True

    def set_message(self, message: str) -> bool:
        with self.state.lock:
            if self.state.jobs.get(self.job_id) is None:
                return False
            self.state.jobs.set_message(self.job_id, message)
            return True

    def is_cancel_requested(self) -> bool:
        with self.state.lock:
            return self.state.jobs.is_cancel_requested(self.job_id)

    def ensure_not_cancelled(
        self,
        *,
        message: str = 'The job was cancelled by the user.',
    ) -> None:
        ensure_job_not_cancelled(self.state, self.job_id, message=message)

    def cancel_check(self) -> Callable[[], bool]:
        return self.is_cancel_requested


__all__ = ['JobContext', 'JobCancelledError']
