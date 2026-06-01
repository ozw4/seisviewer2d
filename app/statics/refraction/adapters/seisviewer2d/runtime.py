"""SeisViewer2D runtime adapter for refraction workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.state import AppState
from app.statics.refraction.adapters.seisviewer2d.artifact_resolver import (
    SeisViewer2DRefractionArtifactResolver,
)
from app.statics.refraction.adapters.seisviewer2d.corrected_store import (
    SeisViewer2DRefractionCorrectedStoreProvider,
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
from app.statics.refraction.ports.corrected_store import (
    RefractionCorrectedStoreProvider,
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

    @property
    def corrected_store(self) -> RefractionCorrectedStoreProvider:
        return SeisViewer2DRefractionCorrectedStoreProvider(self.state)

    def job_context(self, *, job_id: str, artifacts_dir: Path) -> RefractionJobContext:
        return SeisViewer2DRefractionJobContext(
            state=self.state,
            job_id=job_id,
            artifacts_dir=Path(artifacts_dir),
        )

    def job_artifacts_dir(self, job_id: str) -> Path:
        with self.state.lock:
            job = self.state.jobs.get(job_id)
            artifacts_dir = job.get('artifacts_dir') if isinstance(job, dict) else None
        if not isinstance(artifacts_dir, str) or not artifacts_dir:
            raise ValueError('job artifacts_dir is not available')
        return Path(artifacts_dir)

    def get_job_snapshot(self, job_id: str) -> dict[str, object] | None:
        with self.state.lock:
            job = self.state.jobs.get(job_id)
            return dict(job) if isinstance(job, dict) else None

    def set_static_corrected_file(
        self,
        job_id: str,
        *,
        corrected_file_id: str,
        corrected_store_path: str,
    ) -> None:
        with self.state.lock:
            self.state.jobs.set_static_corrected_file(
                job_id,
                corrected_file_id=corrected_file_id,
                corrected_store_path=corrected_store_path,
            )


__all__ = ['SeisViewer2DRefractionRuntime']
