"""Runtime dependency port for refraction workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.statics.refraction.ports.artifact_resolver import (
    RefractionArtifactResolver,
)
from app.statics.refraction.ports.corrected_store import (
    RefractionCorrectedStoreProvider,
)
from app.statics.refraction.ports.job_context import RefractionJobContext
from app.statics.refraction.ports.trace_store import RefractionTraceStoreProvider


class RefractionRuntime(Protocol):
    """Container for application services used by refraction orchestration."""

    @property
    def trace_store(self) -> RefractionTraceStoreProvider:
        """TraceStore provider for refraction workflows."""

    @property
    def artifacts(self) -> RefractionArtifactResolver:
        """Artifact resolver for refraction workflows."""

    @property
    def corrected_store(self) -> RefractionCorrectedStoreProvider:
        """Corrected TraceStore provider for refraction workflows."""

    def job_context(self, *, job_id: str, artifacts_dir: Path) -> RefractionJobContext:
        """Return a job context bound to one background job."""

    def job_artifacts_dir(self, job_id: str) -> Path:
        """Return the artifacts directory for a registered job."""

    def get_job_snapshot(self, job_id: str) -> dict[str, object] | None:
        """Return a shallow copy of job metadata if available."""

    def set_static_corrected_file(
        self,
        job_id: str,
        *,
        corrected_file_id: str,
        corrected_store_path: str,
    ) -> None:
        """Record corrected TraceStore metadata on a static job."""


__all__ = ['RefractionRuntime']
