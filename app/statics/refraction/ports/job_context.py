"""Background-job execution context port for refraction workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class RefractionJobContext(Protocol):
    """Expose progress reporting and cancellation checks for one job."""

    job_id: str
    artifacts_dir: Path

    def set_progress(self, progress: float, message: str) -> None:
        """Update job progress and status text."""

    def set_message(self, message: str) -> None:
        """Update job status text."""

    def ensure_not_cancelled(self) -> None:
        """Raise when cancellation has been requested."""


__all__ = ['RefractionJobContext']
