"""SeisViewer2D background-job context adapter for refraction workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.state import AppState
from app.services.job_runner import (
    ensure_job_not_cancelled,
    set_job_message,
    set_job_progress,
)
from app.statics.refraction.ports.job_context import RefractionJobContext


@dataclass(frozen=True)
class SeisViewer2DRefractionJobContext(RefractionJobContext):
    """Report progress and cancellation through the existing job runner."""

    state: AppState
    job_id: str
    artifacts_dir: Path

    def set_progress(self, progress: float, message: str) -> None:
        set_job_progress(self.state, self.job_id, progress)
        set_job_message(self.state, self.job_id, message)

    def set_message(self, message: str) -> None:
        set_job_message(self.state, self.job_id, message)

    def ensure_not_cancelled(self) -> None:
        ensure_job_not_cancelled(self.state, self.job_id)


__all__ = ['SeisViewer2DRefractionJobContext']
