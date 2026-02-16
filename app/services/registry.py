"""File-registry helpers."""

from __future__ import annotations

from pathlib import Path

from app.utils.segy_meta import FILE_REGISTRY


def _update_file_registry(
    file_id: str,
    *,
    path: str | None = None,
    store_path: str | None = None,
    dt: float | None = None,
) -> None:
    rec = FILE_REGISTRY.get(file_id) or {}
    if path:
        rec['path'] = path
    if store_path:
        rec['store_path'] = store_path
    if isinstance(dt, (int, float)) and dt > 0:
        rec['dt'] = float(dt)
    FILE_REGISTRY[file_id] = rec


def _filename_for_file_id(file_id: str) -> str | None:
    rec = FILE_REGISTRY.get(file_id) or {}
    path = rec.get('path') or rec.get('store_path')
    return Path(path).name if path else None


__all__ = ['_filename_for_file_id', '_update_file_registry']
