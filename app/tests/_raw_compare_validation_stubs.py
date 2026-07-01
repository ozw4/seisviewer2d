from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


class ValidationReader:
    def __init__(
        self,
        key2_by_key1: Mapping[int, Sequence[int]],
        *,
        n_samples: int = 1500,
    ) -> None:
        self._key1_values = np.asarray(list(key2_by_key1), dtype=np.int64)
        self._n_samples = int(n_samples)
        self._seq_by_key1: dict[int, np.ndarray] = {}
        chunks: list[np.ndarray] = []
        offset = 0
        for key1, key2_values in key2_by_key1.items():
            key2_arr = np.asarray(key2_values, dtype=np.int64)
            chunks.append(key2_arr)
            self._seq_by_key1[int(key1)] = np.arange(
                offset,
                offset + key2_arr.size,
                dtype=np.int64,
            )
            offset += int(key2_arr.size)
        self._key2_header = (
            np.concatenate(chunks)
            if chunks
            else np.asarray([], dtype=np.int64)
        )

    def get_key1_values(self) -> np.ndarray:
        return np.array(self._key1_values, copy=True)

    def get_n_samples(self) -> int:
        return self._n_samples

    def get_header(self, _byte: int) -> np.ndarray:
        return np.array(self._key2_header, copy=True)

    def get_trace_seq_for_value(
        self,
        key1_val: int,
        align_to: str = 'display',
    ) -> np.ndarray:
        if align_to != 'display':
            raise ValueError("align_to must be 'display'")
        return np.array(self._seq_by_key1[int(key1_val)], copy=True)


class DummyFileRegistry:
    def __init__(
        self,
        file_ids: Sequence[str],
        *,
        dts: Mapping[str, float] | None = None,
        base_path: Path,
    ) -> None:
        dt_by_file = dict(dts or {})
        self._records = {
            file_id: {
                'path': str(base_path / f'{file_id}.sgy'),
                'dt': float(dt_by_file.get(file_id, 0.002)),
            }
            for file_id in file_ids
        }

    def get_dt(self, file_id: str) -> float:
        return float(self._records[file_id]['dt'])

    def filename(self, file_id: str) -> str | None:
        path = self._records[file_id].get('path')
        if not path:
            return None
        return Path(str(path)).name


class DummyState:
    def __init__(
        self,
        file_ids: Sequence[str],
        *,
        dts: Mapping[str, float] | None = None,
        base_path: Path,
    ) -> None:
        self.file_registry = DummyFileRegistry(
            file_ids,
            dts=dts,
            base_path=base_path,
        )
