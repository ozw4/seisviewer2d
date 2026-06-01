"""SeisViewer2D runtime adapter for refraction workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.state import AppState
from app.statics.refraction.adapters.seisviewer2d.artifact_resolver import (
    SeisViewer2DRefractionArtifactResolver,
)
from app.statics.refraction.adapters.seisviewer2d.job_context import (
    SeisViewer2DRefractionJobContext,
)
from app.statics.refraction.adapters.seisviewer2d.trace_store import (
    SeisViewer2DRefractionTraceStoreProvider,
)
from app.statics.refraction.ports.artifact_resolver import (
    RefractionArtifactResolver,
)
from app.statics.refraction.ports.job_context import RefractionJobContext
from app.statics.refraction.ports.runtime import RefractionRuntime
from app.statics.refraction.ports.trace_store import RefractionTraceStoreProvider


@dataclass(frozen=True)
class SeisViewer2DRefractionRuntime(RefractionRuntime):
    """AppState-backed runtime dependency bundle for refraction workflows."""

    state: AppState

    @property
    def trace_store(self) -> RefractionTraceStoreProvider:
        return SeisViewer2DRefractionTraceStoreProvider(self.state)

    @property
    def artifacts(self) -> RefractionArtifactResolver:
        return SeisViewer2DRefractionArtifactResolver(self.state)

    def job_context(self, *, job_id: str, artifacts_dir: Path) -> RefractionJobContext:
        return SeisViewer2DRefractionJobContext(
            state=self.state,
            job_id=job_id,
            artifacts_dir=Path(artifacts_dir),
        )


__all__ = ['SeisViewer2DRefractionRuntime']
