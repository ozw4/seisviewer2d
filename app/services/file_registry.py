"""Thread-safe file registry for SEG-Y and trace-store metadata."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from app.services.errors import ConflictError, NotFoundError
from app.utils.segy_meta import read_segy_dt_seconds


class FileRegistry:
    """Mutable file metadata registry with coarse-grained locking."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.records: dict[str, dict[str, Any]] = {}

    def clear(self) -> None:
        with self.lock:
            self.records.clear()

    def pop(self, file_id: str, default=None):
        with self.lock:
            return self.records.pop(file_id, default)

    def get_record(self, file_id: str) -> dict[str, Any] | None:
        with self.lock:
            rec = self.records.get(file_id)
            if isinstance(rec, dict):
                return rec
            return None

    def set_record(self, file_id: str, rec: dict[str, Any]) -> None:
        with self.lock:
            self.records[file_id] = rec

    def update(
        self,
        file_id: str,
        *,
        path: str | Path | None = None,
        store_path: str | Path | None = None,
        dt: float | None = None,
    ) -> None:
        with self.lock:
            rec = self.records.get(file_id)
            if not isinstance(rec, dict):
                rec = {}
                self.records[file_id] = rec
            path_str = self._coerce_path(path)
            if path_str is not None:
                rec['path'] = path_str
            store_path_str = self._coerce_path(store_path)
            if store_path_str is not None:
                rec['store_path'] = store_path_str
            if isinstance(dt, (int, float)) and dt > 0:
                rec['dt'] = float(dt)

    @staticmethod
    def _coerce_path(value: Any) -> str | None:
        if isinstance(value, Path):
            text = str(value)
        elif isinstance(value, str):
            text = value
        else:
            return None
        return text if text else None

    def get_store_path(self, file_id: str) -> str:
        with self.lock:
            rec = self.records.get(file_id)
            if not isinstance(rec, dict):
                raise NotFoundError('File ID not found')
            store_path = self._coerce_path(rec.get('store_path'))
            if store_path is not None:
                return store_path
        raise ConflictError('trace store not built')

    def _restore_from_meta(self, file_id: str, store_path: str) -> float | None:
        meta_path = Path(store_path) / 'meta.json'
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:  # noqa: BLE001
            meta = None
        if not isinstance(meta, dict):
            return None
        meta_dt = meta.get('dt')
        if isinstance(meta_dt, (int, float)) and meta_dt > 0:
            with self.lock:
                rec = self.records.get(file_id)
                if not isinstance(rec, dict):
                    rec = {}
                    self.records[file_id] = rec
                rec['dt'] = float(meta_dt)
            return float(meta_dt)
        original = meta.get('original_segy_path')
        if isinstance(original, str):
            with self.lock:
                rec = self.records.get(file_id)
                if not isinstance(rec, dict):
                    rec = {}
                    self.records[file_id] = rec
                rec['path'] = original
        return None

    def get_dt(self, file_id: str) -> float:
        with self.lock:
            rec = self.records.get(file_id)
            if not isinstance(rec, dict):
                rec = {}
                self.records[file_id] = rec
            dt_val = rec.get('dt')
            if isinstance(dt_val, (int, float)) and dt_val > 0:
                return float(dt_val)
            path = self._coerce_path(rec.get('path'))
            store_path = self._coerce_path(rec.get('store_path'))

        if path is None and store_path is not None:
            meta_dt = self._restore_from_meta(file_id, store_path)
            if isinstance(meta_dt, (int, float)) and meta_dt > 0:
                return float(meta_dt)
            with self.lock:
                rec = self.records.get(file_id)
                if isinstance(rec, dict):
                    path = self._coerce_path(rec.get('path'))

        dt = read_segy_dt_seconds(path) if path else None
        if not dt:
            raise RuntimeError('dt not found')

        with self.lock:
            rec = self.records.get(file_id)
            if not isinstance(rec, dict):
                rec = {}
                self.records[file_id] = rec
            rec['dt'] = float(dt)
        return float(dt)

    def filename(self, file_id: str) -> str | None:
        with self.lock:
            rec = self.records.get(file_id)
            if not isinstance(rec, dict):
                return None
            path = rec.get('path')
            if not path:
                path = rec.get('store_path')
            if not path:
                return None
            normalized = str(path).replace('\\', '/')
            return Path(normalized).name


__all__ = ['FileRegistry']
