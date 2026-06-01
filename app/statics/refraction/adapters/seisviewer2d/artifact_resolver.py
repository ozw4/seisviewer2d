"""SeisViewer2D job-artifact adapter for refraction workflows."""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path

from app.core.state import AppState
from app.services.job_artifact_refs import resolve_job_artifact_path
from app.statics.refraction.ports.artifact_resolver import (
    RefractionArtifactResolver,
)


@dataclass(frozen=True)
class SeisViewer2DRefractionArtifactResolver(RefractionArtifactResolver):
    """Resolve job artifacts through the existing AppState job table."""

    state: AppState

    def resolve_artifact(
        self,
        *,
        job_id: str,
        name: str,
        allowed_job_types: Set[str] | None = None,
        allowed_statics_kinds: Set[str] | None = None,
        expected_file_id: str | None = None,
        expected_key1_byte: int | None = None,
        expected_key2_byte: int | None = None,
        reference_label: str = 'artifact',
    ) -> Path:
        return resolve_job_artifact_path(
            self.state,
            job_id=job_id,
            name=name,
            allowed_job_types=allowed_job_types,
            allowed_statics_kinds=allowed_statics_kinds,
            expected_file_id=expected_file_id,
            expected_key1_byte=expected_key1_byte,
            expected_key2_byte=expected_key2_byte,
            reference_label=reference_label,
        )


__all__ = ['SeisViewer2DRefractionArtifactResolver']
