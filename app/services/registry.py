"""File-registry helpers."""

from __future__ import annotations

from app.services.file_registry import FileRegistry


def _update_file_registry(
    file_registry: FileRegistry,
    file_id: str,
    *,
    path: str | None = None,
    store_path: str | None = None,
    dt: float | None = None,
) -> None:
    file_registry.update(
        file_id,
        path=path,
        store_path=store_path,
        dt=dt,
    )


def _filename_for_file_id(file_id: str, *, file_registry: FileRegistry) -> str | None:
    return file_registry.filename(file_id)


__all__ = ['_filename_for_file_id', '_update_file_registry']
