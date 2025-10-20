"""Utility helpers for working with SEG-Y sections and cached traces."""

from __future__ import annotations

import json
import os
import re
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
	default_scale = float(os.getenv('FIXED_INT8_SCALE', '42.333333'))
	scale = float(fixed_scale) if fixed_scale is not None else default_scale
	q = np.clip(np.round(arr * scale), -qmax, qmax).astype(np.int8)
	return scale, q


class SegySectionReader:
	"""Deprecated wrapper delegating to :class:`TraceStoreSectionReader`."""

	def __init__(
		self,
		path: str | os.PathLike[str],
		key1_byte: int = 189,
		key2_byte: int = 193,
	) -> None:
		"""Build or reuse a TraceStore for the provided SEG-Y file."""
		self.path = Path(path)
		self.key1_byte = key1_byte
		self.key2_byte = key2_byte
		warnings.warn(
			'SegySectionReader is deprecated; use TraceStoreSectionReader instead.',
			DeprecationWarning,
			stacklevel=2,
		)
		self.section_cache: dict[int, SectionView] = {}
		self._trace_seq_cache: dict[int, np.ndarray] = {}
		self._trace_seq_disp_cache: dict[int, np.ndarray] = {}
		self._store_dir = self._compute_store_dir()
		self._delegate = self._initialize_delegate()
		self.section_cache = self._delegate.section_cache
		self.traces = getattr(self._delegate, 'traces', None)
		self.dtype = getattr(self._delegate, 'dtype', None)
		self.scale = getattr(self._delegate, 'scale', None)
		self.meta = getattr(self._delegate, 'meta', None)

	def _compute_store_dir(self) -> Path:
		safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', self.path.name)
		return self.path.parent / f'{safe_name}.trace_store'

	def _trace_store_complete(self) -> bool:
		required = [
			self._store_dir / 'traces.npy',
			self._store_dir / 'index.npz',
			self._store_dir / 'meta.json',
			self._store_dir / f'headers_byte_{self.key1_byte}.npy',
			self._store_dir / f'headers_byte_{self.key2_byte}.npy',
		]
		if not all(path.exists() for path in required):
			return False
		try:
			meta = json.loads((self._store_dir / 'meta.json').read_text())
		except Exception:  # noqa: BLE001
			return False
		key_meta = meta.get('key_bytes') if isinstance(meta, dict) else None
		if isinstance(key_meta, dict):
			key1_meta = int(key_meta.get('key1', self.key1_byte))
			key2_meta = int(key_meta.get('key2', self.key2_byte))
			if key1_meta != self.key1_byte or key2_meta != self.key2_byte:
				return False
		return True

	def _initialize_delegate(self) -> TraceStoreSectionReader:
		if not self._trace_store_complete():
			SegyIngestor.from_segy(
				self.path,
				self._store_dir,
				key1_byte=self.key1_byte,
				key2_byte=self.key2_byte,
			)
		return TraceStoreSectionReader(self._store_dir, self.key1_byte, self.key2_byte)

	def _compute_indices(self, key1_val: int) -> tuple[np.ndarray, np.ndarray]:
		original = self._trace_seq_cache.get(key1_val)
		display = self._trace_seq_disp_cache.get(key1_val)
		if original is not None and display is not None:
			return original, display

		key1_header = np.asarray(
			self._delegate.get_header(self.key1_byte),
			dtype=np.int64,
		)
		indices = np.flatnonzero(key1_header == key1_val).astype(np.int64)
		if indices.size == 0:
			msg = f'Key1 value {key1_val} not found'
			raise ValueError(msg)
		self._trace_seq_cache[key1_val] = indices

		key2_header = np.asarray(
			self._delegate.get_header(self.key2_byte)[indices],
			dtype=np.int64,
		)
		order = np.argsort(key2_header, kind='stable')
		display_indices = indices[order]
		self._trace_seq_disp_cache[key1_val] = display_indices
		return indices, display_indices

	def get_trace_seq_for_section(
		self, key1_val: int, align_to: str = 'display'
	) -> np.ndarray:
		"""Return TraceSeq indices for ``key1_val`` aligned to the requested order."""
		original, display = self._compute_indices(key1_val)
		if align_to == 'display':
			return display
		if align_to == 'original':
			return original
		msg = "align_to must be 'display' or 'original'"
		raise ValueError(msg)

	def get_key1_values(self) -> np.ndarray:
		"""Return the available values for header ``key1``."""
		return self._delegate.get_key1_values()

	def get_section(self, key1_val: int) -> SectionView:
		"""Return the cached section for ``key1_val``."""
		return self._delegate.get_section(key1_val)

	def get_offsets_for_section(self, key1_val: int, offset_byte: int) -> np.ndarray:
		"""Return ``(W,)`` float32 offsets aligned with :meth:`get_section`."""
		return self._delegate.get_offsets_for_section(key1_val, offset_byte)

	def preload_all_sections(self) -> None:
		"""Warm caches using the delegate TraceStore reader."""
		self._delegate.preload_all_sections()


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
		meta_path = self.store_dir / 'meta.json'
		self.meta = json.loads(meta_path.read_text())
		self.traces = np.load(self.store_dir / 'traces.npy', mmap_mode='r')
		self.section_cache: dict[int, SectionView] = {}
		self._trace_seq_cache: dict[int, np.ndarray] = {}
		self._trace_seq_disp_cache: dict[int, np.ndarray] = {}
		self.dtype = self.traces.dtype
		scale_val = self.meta.get('scale') if isinstance(self.meta, dict) else None
		self.scale = float(scale_val) if isinstance(scale_val, (int, float)) else None
		print('Initialized TraceStoreSectionReader with scale:', self.scale)

	def _header_path(self, byte: int) -> Path:
		return self.store_dir / f'headers_byte_{byte}.npy'

	def ensure_header(self, header_byte: int) -> np.ndarray:
		"""Ensure the header array for ``header_byte`` exists on disk and return it."""
		path = self._header_path(header_byte)
		if path.exists():
			return np.load(path, mmap_mode='r')

		print(f'Extracting header byte {header_byte} for {self.store_dir}')
		with segyio.open(
			self.meta['original_segy_path'],
			'r',
			ignore_geometry=True,
		) as f:
			f.mmap()
			values = f.attributes(header_byte)[:].astype(np.int32)

		tmp_path = path.with_name(path.stem + '_tmp.npy')
		np.save(tmp_path, values)
		tmp_path.replace(path)
		return np.load(path, mmap_mode='r')

	def get_header(self, byte: int) -> np.ndarray:
		"""Return the header array for ``byte``."""
		return self.ensure_header(byte)

	def get_n_samples(self) -> int:
		"""Return the number of samples per trace."""
		return int(self.traces.shape[-1])

	def _compute_indices(self, key1_val: int) -> tuple[np.ndarray, np.ndarray]:
		"""Return original and display-aligned indices for ``key1_val``."""
		original = self._trace_seq_cache.get(key1_val)
		display = self._trace_seq_disp_cache.get(key1_val)
		if original is not None and display is not None:
			return original, display

		key1s = self.get_header(self.key1_byte)
		indices = np.flatnonzero(key1s == key1_val).astype(np.int64)
		print(len(indices), 'indices found for key1_val:', key1_val)
		if indices.size == 0:
			msg = f'Key1 value {key1_val} not found'
			raise ValueError(msg)

		key2s = self.get_header(self.key2_byte)[indices]
		order = np.argsort(key2s, kind='stable')
		display_indices = indices[order]
		self._trace_seq_cache[key1_val] = indices
		self._trace_seq_disp_cache[key1_val] = display_indices
		return indices, display_indices

	def get_key1_values(self) -> np.ndarray:
		"""Return the available ``key1`` header values."""
		key1s = self.get_header(self.key1_byte)
		return np.unique(key1s)

	def get_trace_seq_for_value(
		self, key1_val: int, align_to: str = 'display'
	) -> np.ndarray:
		"""Return TraceSeq indices for ``key1_val`` aligned to ``align_to`` order."""
		original, display = self._compute_indices(key1_val)
		if align_to == 'display':
			return display
		if align_to == 'original':
			return original
		msg = "align_to must be 'display' or 'original'"
		raise ValueError(msg)

	def get_section(self, key1_val: int) -> SectionView:
		"""Return the cached section for ``key1_val``."""
		cached = self.section_cache.get(key1_val)
		if cached is not None:
			return cached

		_, sorted_indices = self._compute_indices(key1_val)
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
		_, sorted_indices = self._compute_indices(key1_val)
		header = self.ensure_header(offset_byte)
		offsets = np.asarray(header[sorted_indices], dtype=np.float32)
		return np.ascontiguousarray(offsets)

	def preload_all_sections(self) -> None:
		"""Warm caches for the frequently accessed headers."""
		self.get_header(self.key1_byte)
		self.get_header(self.key2_byte)
