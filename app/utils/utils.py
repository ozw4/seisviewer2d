"""Utility helpers for working with SEG-Y sections and cached traces."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import segyio

from app.utils.ingest import SegyIngestor


class SectionView(NamedTuple):
	"""Lightweight wrapper describing a section view."""

	arr: np.ndarray
	dtype: np.dtype
	scale: float | None = None


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
	"""Compatibility wrapper that ingests SEG-Y data into TraceStore."""

	def __init__(
		self,
		path: str | os.PathLike[str],
		key1_byte: int = 189,
		key2_byte: int = 193,
	) -> None:
		self.path = Path(path)
		self.key1_byte = int(key1_byte)
		self.key2_byte = int(key2_byte)
		self.section_cache: dict[int, SectionView] = {}
		self._trace_seq_cache: dict[int, np.ndarray] = {}
		self._trace_seq_disp_cache: dict[int, np.ndarray] = {}
		self._delegate: TraceStoreSectionReader | None = None
		warnings.warn(
			'`SegySectionReader` is deprecated. Please ingest the SEG-Y file '
			'into a TraceStore before reading.',
			DeprecationWarning,
			stacklevel=2,
		)
		store_dir = self._default_store_dir()
		if not self._is_store_complete(store_dir, self.key1_byte, self.key2_byte):
			SegyIngestor.from_segy(
				self.path,
				store_dir,
				self.key1_byte,
				self.key2_byte,
			)
		self.store_dir = store_dir
		self._delegate = TraceStoreSectionReader(store_dir, self.key1_byte, self.key2_byte)
		self.meta = getattr(self._delegate, 'meta', {})
		self.traces = self._delegate.traces
		self.section_cache = self._delegate.section_cache
		self.dtype = self._delegate.dtype
		self.scale = self._delegate.scale
		self._hydrate_headers_from_delegate()
		self.unique_key1 = np.unique(self.key1s)
		self.ntraces = int(self.key1s.shape[0])

	def _default_store_dir(self) -> Path:
		return self.path.with_name(f"{self.path.stem}_trace_store")

	@staticmethod
	def _is_store_complete(store_dir: Path, key1_byte: int, key2_byte: int) -> bool:
		required = [
			store_dir / 'traces.npy',
			store_dir / 'meta.json',
			store_dir / 'index.npz',
			store_dir / f'headers_byte_{key1_byte}.npy',
			store_dir / f'headers_byte_{key2_byte}.npy',
		]
		return all(path.exists() for path in required)

	def _hydrate_headers_from_delegate(self) -> None:
		delegate = self._delegate
		if delegate is None:
			return
		self.key1s = np.asarray(delegate.get_header(self.key1_byte), dtype=np.int32)
		self.key2s = np.asarray(delegate.get_header(self.key2_byte), dtype=np.int32)

	def _indices_for_key1(self, key1_val: int) -> np.ndarray:
		if self._delegate is not None:
			self._hydrate_headers_from_delegate()
		if key1_val in self._trace_seq_cache:
			return self._trace_seq_cache[key1_val]

		idx = np.flatnonzero(self.key1s == key1_val).astype(np.int64)
		if idx.size == 0:
			msg = f"Key1 value {key1_val} not found"
			raise ValueError(msg)

		self._trace_seq_cache[key1_val] = idx
		return idx

	def _sorted_indices_for_key1(self, key1_val: int) -> np.ndarray:
		if self._delegate is not None:
			self._hydrate_headers_from_delegate()
		if key1_val in self._trace_seq_disp_cache:
			return self._trace_seq_disp_cache[key1_val]

		idx = self._indices_for_key1(key1_val)
		order = np.argsort(self.key2s[idx], kind='stable')
		sorted_idx = idx[order]
		self._trace_seq_disp_cache[key1_val] = sorted_idx
		return sorted_idx

	def get_trace_seq_for_section(self, key1_val: int, align_to: str = 'display') -> np.ndarray:
		if align_to == 'display':
			return self._sorted_indices_for_key1(key1_val)
		if align_to == 'original':
			return self._indices_for_key1(key1_val)
		msg = "align_to must be 'display' or 'original'"
		raise ValueError(msg)

	def get_key1_values(self) -> np.ndarray:
		if self._delegate is not None:
			return self._delegate.get_key1_values()
		return self.unique_key1

	def _load_section_array(self, sorted_indices: np.ndarray) -> np.ndarray:
		if sorted_indices.size == 0:
			return np.empty((0, 0), dtype=self.dtype)

		with segyio.open(self.path, 'r', ignore_geometry=True) as f:
			f.mmap()
			trace_len = int(np.asarray(f.trace[int(sorted_indices[0])]).shape[0])
			out = np.empty((sorted_indices.size, trace_len), dtype=self.dtype)
			for out_row, idx in enumerate(sorted_indices):
				out[out_row] = np.asarray(f.trace[int(idx)], dtype=self.dtype)
		return out

	def get_section(self, key1_val: int) -> SectionView:
		if self._delegate is not None:
			return self._delegate.get_section(key1_val)

		cached = self.section_cache.get(key1_val)
		if cached is not None:
			return cached

		sorted_indices = self.get_trace_seq_for_section(key1_val, align_to='display')
		print(len(sorted_indices), 'indices found for key1_val:', key1_val)
		if sorted_indices.size == 0:
			msg = f"Key1 value {key1_val} not found"
			raise ValueError(msg)

		arr = self._load_section_array(sorted_indices)
		view = SectionView(arr=arr, dtype=arr.dtype, scale=None)
		self.section_cache[key1_val] = view
		return view

	def get_offsets_for_section(self, key1_val: int, offset_byte: int) -> np.ndarray:
		if self._delegate is not None:
			return self._delegate.get_offsets_for_section(key1_val, offset_byte)

		sorted_indices = self.get_trace_seq_for_section(key1_val, align_to='display')
		print(len(sorted_indices), 'indices found for key1_val:', key1_val)

		with segyio.open(self.path, 'r', ignore_geometry=True) as f:
			f.mmap()
			attr = f.attributes(offset_byte)
			offsets = np.asarray(attr[sorted_indices], dtype=np.float32)
		return np.ascontiguousarray(offsets)

	def preload_all_sections(self) -> None:
		if self._delegate is not None:
			self._delegate.preload_all_sections()
			return
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
		self.section_cache: dict[int, SectionView] = {}
		self.dtype = self.traces.dtype
		scale_val = self.meta.get("scale") if isinstance(self.meta, dict) else None
		self.scale = float(scale_val) if isinstance(scale_val, (int, float)) else None

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

	def get_section(self, key1_val: int) -> SectionView:
		"""Return the cached section for ``key1_val``."""
		cached = self.section_cache.get(key1_val)
		if cached is not None:
			return cached

		key1s = self.get_header(self.key1_byte)
		indices = np.flatnonzero(key1s == key1_val).astype(np.int64)
		print(len(indices), "indices found for key1_val:", key1_val)
		if indices.size == 0:
			msg = f"Key1 value {key1_val} not found"
			raise ValueError(msg)

		key2s = self.get_header(self.key2_byte)[indices]
		order = np.argsort(key2s, kind="stable")
		sorted_indices = indices[order]
		if sorted_indices.size:
			diffs = np.diff(sorted_indices)
			if np.all(diffs == 1):
				start = int(sorted_indices[0])
				stop = int(sorted_indices[-1]) + 1
				section = self.traces[start:stop]
			else:
				section = self.traces[sorted_indices]
		else:
			section = self.traces[[]]
		view = SectionView(arr=section, dtype=section.dtype, scale=self.scale)
		self.section_cache[key1_val] = view
		return view

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
