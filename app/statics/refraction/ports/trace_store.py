"""TraceStore access port for refraction workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class RefractionTraceStoreProvider(Protocol):
    """Provide TraceStore paths, metadata, and readers by file id."""

    def get_store_path(self, file_id: str) -> Path:
        """Return the TraceStore directory for a registered file."""

    def get_reader(
        self,
        file_id: str,
        key1_byte: int,
        key2_byte: int,
    ) -> object:
        """Return a section reader for a registered TraceStore."""

    def get_dt(self, file_id: str) -> float:
        """Return the sample interval in seconds for a registered file."""

    def filename(self, file_id: str) -> str | None:
        """Return the source filename used for manual-pick cache lookup."""


__all__ = ['RefractionTraceStoreProvider']
