"""Utility classes for SEG-Y section handling."""

import json
from typing import Any

import numpy as np
import segyio


class SegySectionReader:
        """Read and cache SEG-Y sections using Redis."""

        def __init__(
                self,
                path: str,
                file_id: str,
                key1_byte: int = 189,
                key2_byte: int = 193,
                redis_client: Any | None = None,
        ) -> None:
                """Initialize the reader and load key metadata."""
                self.path = path
                self.file_id = file_id
                self.key1_byte = key1_byte
                self.key2_byte = key2_byte
                self.redis = redis_client
                self._initialize_metadata()

        def _initialize_metadata(self) -> None:
                with segyio.open(self.path, "r", ignore_geometry=True) as f:
                        f.mmap()
                        self.key1s = f.attributes(self.key1_byte)[:]
                        self.key2s = f.attributes(self.key2_byte)[:]
                self.unique_key1 = np.unique(self.key1s)

        def _redis_key(self, key1_val: int) -> str:
                return (
                        f"section:{self.file_id}:{self.key1_byte}:"
                        f"{self.key2_byte}:{key1_val}"
                )

        def get_key1_values(self) -> np.ndarray:
                """Return sorted unique values for key1."""
                return self.unique_key1

        def get_section(self, key1_val: int) -> list[list[float]]:
                """Return a normalized section for the given key1 value."""
                if self.redis:
                        cached = self.redis.get(self._redis_key(key1_val))
                        if cached:
                                return json.loads(cached)

                indices = np.where(self.key1s == key1_val)[0]
                print(len(indices), "indices found for key1_val:", key1_val)
                if len(indices) == 0:
                        msg = f"Key1 value {key1_val} not found"
                        raise ValueError(msg)

                key2_vals = self.key2s[indices]
                sorted_indices = indices[np.argsort(key2_vals)]

                with segyio.open(self.path, "r", ignore_geometry=True) as f:
                        f.mmap()
                        traces = np.array(
                                [f.trace[i] for i in sorted_indices],
                                dtype="float32",
                        )
                        max_abs = np.max(np.abs(traces), axis=1, keepdims=True)
                        max_abs[max_abs == 0] = 1.0
                        section = (traces / max_abs).tolist()

                if self.redis:
                        self.redis.set(self._redis_key(key1_val), json.dumps(section))
                return section

        def preload_all_sections(self) -> None:
                """Preload all sections into Redis."""
                for key1_val in self.unique_key1:
                        self.get_section(key1_val)

