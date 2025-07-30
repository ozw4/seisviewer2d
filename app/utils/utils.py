"""Helper classes for SEG-Y operations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import segyio


class SegySectionReader:
	"""Read and cache 2D sections from a SEG-Y file."""

	def __init__(
		self,
		path: str | Path,
		key1_byte: int = 189,
		key2_byte: int = 193,
	) -> None:
		"""Initialize reader with target file and trace header bytes."""
		self.path = Path(path)
		self.key1_byte = key1_byte
		self.key2_byte = key2_byte
		self.section_cache: dict[int, list[list[float]]] = {}
		self._initialize_metadata()

	def _initialize_metadata(self) -> None:
		"""Read header information needed for section extraction."""
		with segyio.open(self.path, 'r', ignore_geometry=True) as f:
			f.mmap()
			self.key1s = f.attributes(self.key1_byte)[:]
			self.key2s = f.attributes(self.key2_byte)[:]
		self.unique_key1 = np.unique(self.key1s)

	def get_section(self, key1_val: int) -> list[list[float]]:
		"""Return all traces that share ``key1_val`` sorted by ``key2``."""
		if key1_val in self.section_cache:
			return self.section_cache[key1_val]

		indices = np.where(self.key1s == key1_val)[0]
		if len(indices) == 0:
			msg = f'Key1 value {key1_val} not found'
			raise ValueError(msg)

		key2_vals = self.key2s[indices]
		sorted_indices = indices[np.argsort(key2_vals)]
		with segyio.open(self.path, 'r', ignore_geometry=True) as f:
			f.mmap()
			section = [f.trace[i].tolist() for i in sorted_indices]

		self.section_cache[key1_val] = section
		return section
