"""Runtime dependency port for refraction workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.statics.refraction.ports.artifact_resolver import (
    RefractionArtifactResolver,
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

    def job_context(self, *, job_id: str, artifacts_dir: Path) -> RefractionJobContext:
        """Return a job context bound to one background job."""


__all__ = ['RefractionRuntime']
