"""Utility helpers for working with SEG-Y sections."""

import json

import numpy as np
import redis
import segyio


class SegySectionReader:
    """Read SEG-Y sections and cache them in Redis."""

    def __init__(
        self,
        path: str,
        file_id: str,
        key1_byte: int = 189,
        key2_byte: int = 193,
    ) -> None:
        """Initialize the reader and pre-load header metadata."""
        self.path = path
        self.file_id = file_id
        self.key1_byte = key1_byte
        self.key2_byte = key2_byte
        self.redis = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        self._initialize_metadata()

    def _initialize_metadata(self) -> None:
        with segyio.open(self.path, "r", ignore_geometry=True) as f:
            f.mmap()
            self.key1s = f.attributes(self.key1_byte)[:]
            self.key2s = f.attributes(self.key2_byte)[:]
        self.unique_key1 = np.unique(self.key1s)

    def get_key1_values(self) -> np.ndarray:
        """Return available key1 values."""
        return self.unique_key1

    def get_section(self, key1_val: int) -> list[list[float]]:
        """Return a normalized section for the given key1 value."""
        cache_key = f"section:{self.file_id}:{key1_val}"
        cached = self.redis.get(cache_key)
        if cached is not None:
            return json.loads(cached)

        indices = np.where(self.key1s == key1_val)[0]
        print(len(indices), "indices found for key1_val:", key1_val)
        if len(indices) == 0:
            message = f"Key1 value {key1_val} not found"
            raise ValueError(message)

        key2_vals = self.key2s[indices]
        sorted_indices = indices[np.argsort(key2_vals)]

        with segyio.open(self.path, "r", ignore_geometry=True) as f:
            f.mmap()
            traces = np.array(
                [f.trace[i] for i in sorted_indices], dtype="float32"
            )
            max_abs = np.max(np.abs(traces), axis=1, keepdims=True)
            max_abs[max_abs == 0] = 1.0
            section = (traces / max_abs).tolist()

        self.redis.set(cache_key, json.dumps(section))
        return section

    def preload_all_sections(self) -> None:
        """Eagerly load all sections into the Redis cache."""
        for key1_val in self.unique_key1:
            self.get_section(key1_val)
