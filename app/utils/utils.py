"""Utility helpers for working with SEG-Y sections and cached traces."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import segyio


def to_builtin(obj: object) -> object:
    """Recursively convert numpy containers to built-in Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {key: to_builtin(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(val) for val in obj]
    return obj


def quantize_float32(
    arr: np.ndarray, *, bits: int = 8, fixed_scale: float | None = None
) -> tuple[float, np.ndarray]:
    """Quantize ``arr`` into int8 with an optional externally provided scale."""
    qmax = (1 << (bits - 1)) - 1
    default_scale = float(os.getenv("FIXED_INT8_SCALE", "42.333333"))
    scale = float(fixed_scale) if fixed_scale is not None else default_scale
    q = np.clip(np.round(arr * scale), -qmax, qmax).astype(np.int8)
    return scale, q


class SegySectionReader:
    """Read SEG-Y sections and associated trace headers."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        key1_byte: int = 189,
        key2_byte: int = 193,
    ) -> None:
        """Initialize the SEG-Y reader with header byte settings."""
        self.path = Path(path)
        self.key1_byte = key1_byte
        self.key2_byte = key2_byte
        self.section_cache: dict[int, list[list[float]]] = {}
        self._trace_seq_cache: dict[int, np.ndarray] = {}
        self._trace_seq_disp_cache: dict[int, np.ndarray] = {}
        self.ntraces: int = 0
        self._initialize_metadata()

    def _initialize_metadata(self) -> None:
        with segyio.open(self.path, "r", ignore_geometry=True) as f:
            f.mmap()
            self.key1s = f.attributes(self.key1_byte)[:]
            self.key2s = f.attributes(self.key2_byte)[:]
        self.unique_key1 = np.unique(self.key1s)
        self.ntraces = int(len(self.key1s))

    def _indices_for_key1(self, key1_val: int) -> np.ndarray:
        if key1_val in self._trace_seq_cache:
            return self._trace_seq_cache[key1_val]

        idx = np.flatnonzero(self.key1s == key1_val).astype(np.int64)
        if idx.size == 0:
            msg = f"Key1 value {key1_val} not found"
            raise ValueError(msg)

        self._trace_seq_cache[key1_val] = idx
        return idx

    def _sorted_indices_for_key1(self, key1_val: int) -> np.ndarray:
        if key1_val in self._trace_seq_disp_cache:
            return self._trace_seq_disp_cache[key1_val]

        idx = self._indices_for_key1(key1_val)
        order = np.argsort(self.key2s[idx], kind="stable")
        sorted_idx = idx[order]
        self._trace_seq_disp_cache[key1_val] = sorted_idx
        return sorted_idx

    def get_trace_seq_for_section(
        self, key1_val: int, align_to: str = "display"
    ) -> np.ndarray:
        """Return TraceSeq indices for ``key1_val`` aligned to the requested order."""

        if align_to == "display":
            return self._sorted_indices_for_key1(key1_val)
        if align_to == "original":
            return self._indices_for_key1(key1_val)
        msg = "align_to must be 'display' or 'original'"
        raise ValueError(msg)

    def get_key1_values(self) -> np.ndarray:
        """Return the available values for header ``key1``."""
        return self.unique_key1

    def get_section(self, key1_val: int) -> list[list[float]]:
        """Return the z-scored section for ``key1_val``."""
        if key1_val in self.section_cache:
            return self.section_cache[key1_val]

        sorted_indices = self.get_trace_seq_for_section(key1_val, align_to="display")
        print(len(sorted_indices), "indices found for key1_val:", key1_val)

        with segyio.open(self.path, "r", ignore_geometry=True) as f:
            f.mmap()
            traces = np.array([f.trace[idx] for idx in sorted_indices], dtype="float32")
            mean = traces.mean(axis=1, keepdims=True)
            std = traces.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            section = ((traces - mean) / std).tolist()

        self.section_cache[key1_val] = section
        return section

    def get_offsets_for_section(self, key1_val: int, offset_byte: int) -> np.ndarray:
        """Return ``(W,)`` float32 offsets aligned with :meth:`get_section`."""
        sorted_indices = self.get_trace_seq_for_section(key1_val, align_to="display")
        print(len(sorted_indices), "indices found for key1_val:", key1_val)

        with segyio.open(self.path, "r", ignore_geometry=True) as f:
            f.mmap()
            attr = f.attributes(offset_byte)
            offsets = np.asarray(attr[sorted_indices], dtype=np.float32)
        return np.ascontiguousarray(offsets)

    def preload_all_sections(self) -> None:
        """Populate the section cache for every ``key1`` value."""
        for key1_val in self.unique_key1:
            self.get_section(int(key1_val))


class TraceStoreSectionReader:
    """Read cached traces and headers generated from SEG-Y files."""

    def __init__(
        self,
        store_dir: str | Path,
        key1_byte: int = 189,
        key2_byte: int = 193,
    ) -> None:
        """Initialize the trace-store reader for cached sections."""
        self.store_dir = Path(store_dir)
        self.key1_byte = key1_byte
        self.key2_byte = key2_byte
        meta_path = self.store_dir / "meta.json"
        self.meta = json.loads(meta_path.read_text())
        self.traces = np.load(self.store_dir / "traces.npy", mmap_mode="r")
        self.section_cache: dict[int, list[list[float]]] = {}

    def _header_path(self, byte: int) -> Path:
        return self.store_dir / f"headers_byte_{byte}.npy"

    def ensure_header(self, byte: int) -> np.ndarray:
        """Ensure the header array for ``byte`` exists on disk and return it."""
        path = self._header_path(byte)
        if path.exists():
            return np.load(path, mmap_mode="r")

        print(f"Extracting header byte {byte} for {self.store_dir}")
        with segyio.open(
            self.meta["original_segy_path"],
            "r",
            ignore_geometry=True,
        ) as f:
            f.mmap()
            values = f.attributes(byte)[:].astype(np.int32)

        tmp_path = path.with_name(path.stem + "_tmp.npy")
        np.save(tmp_path, values)
        tmp_path.replace(path)
        return np.load(path, mmap_mode="r")

    def get_header(self, byte: int) -> np.ndarray:
        """Return the header array for ``byte``."""
        return self.ensure_header(byte)

    def get_key1_values(self) -> np.ndarray:
        """Return the available ``key1`` header values."""
        key1s = self.get_header(self.key1_byte)
        return np.unique(key1s)

    def get_section(self, key1_val: int) -> list[list[float]]:
        """Return the cached section for ``key1_val``."""
        if key1_val in self.section_cache:
            return self.section_cache[key1_val]

        key1s = self.get_header(self.key1_byte)
        indices = np.where(key1s == key1_val)[0]
        print(len(indices), "indices found for key1_val:", key1_val)
        if len(indices) == 0:
            msg = f"Key1 value {key1_val} not found"
            raise ValueError(msg)

        key2s = self.get_header(self.key2_byte)[indices]
        sorted_indices = indices[np.argsort(key2s, kind="stable")]
        section = self.traces[sorted_indices].tolist()
        self.section_cache[key1_val] = section
        return section

    def get_offsets_for_section(self, key1_val: int, offset_byte: int) -> np.ndarray:
        """Return ``(W,)`` float32 offsets aligned with :meth:`get_section`."""
        key1s = self.get_header(self.key1_byte)
        indices = np.where(key1s == key1_val)[0]
        print(len(indices), "indices found for key1_val:", key1_val)
        if len(indices) == 0:
            msg = f"Key1 value {key1_val} not found"
            raise ValueError(msg)

        key2s = self.get_header(self.key2_byte)[indices]
        sorted_indices = indices[np.argsort(key2s, kind="stable")]
        header = self.ensure_header(offset_byte)
        offsets = np.asarray(header[sorted_indices], dtype=np.float32)
        return np.ascontiguousarray(offsets)

    def preload_all_sections(self) -> None:
        """Warm caches for the frequently accessed headers."""
        self.get_header(self.key1_byte)
        self.get_header(self.key2_byte)
