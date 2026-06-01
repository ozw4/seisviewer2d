"""SeisViewer2D TraceStore adapter for refraction workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.state import AppState
from app.services.reader import get_reader
from app.statics.refraction.ports.trace_store import RefractionTraceStoreProvider


@dataclass(frozen=True)
class SeisViewer2DRefractionTraceStoreProvider(RefractionTraceStoreProvider):
    """Expose AppState file registry and reader cache through refraction ports."""

    state: AppState

    def get_store_path(self, file_id: str) -> Path:
        return Path(self.state.file_registry.get_store_path(file_id))

    def get_reader(
        self,
        file_id: str,
        key1_byte: int,
        key2_byte: int,
    ) -> object:
        return get_reader(file_id, key1_byte, key2_byte, state=self.state)

    def get_dt(self, file_id: str) -> float:
        return self.state.file_registry.get_dt(file_id)

    def filename(self, file_id: str) -> str | None:
        return self.state.file_registry.filename(file_id)


__all__ = ['SeisViewer2DRefractionTraceStoreProvider']
