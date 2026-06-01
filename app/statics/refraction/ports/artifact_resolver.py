"""Background-job artifact resolution port for refraction workflows."""

from __future__ import annotations

from collections.abc import Set
from pathlib import Path
from typing import Protocol


class RefractionArtifactResolver(Protocol):
    """Resolve artifact file references without exposing application state."""

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
        """Return a validated artifact path for a job artifact reference."""


__all__ = ['RefractionArtifactResolver']
