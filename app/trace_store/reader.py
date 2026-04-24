"""Trace-store section reader."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import numpy as np
import segyio

from app.trace_store.types import SectionView

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Key1Index:
    """Validated key1 lookup metadata loaded from ``index.npz``."""

    key1_values: np.ndarray
    key1_offsets: np.ndarray
    key1_counts: np.ndarray
    key1_pos_by_value: dict[int, int]


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
        self._sorted_to_original: np.ndarray | None = None
        self._sorted_to_original_loaded = False
        self._key1_index: _Key1Index | None = None
        self._key1_index_loaded = False
        self.dtype = self.traces.dtype
        scale_val = self.meta.get('scale') if isinstance(self.meta, dict) else None
        self.scale = float(scale_val) if isinstance(scale_val, (int, float)) else None
        logger.info('Initialized TraceStoreSectionReader with scale: %s', self.scale)

    def _header_path(self, byte: int) -> Path:
        return self.store_dir / f'headers_byte_{byte}.npy'

    def _meta_key_bytes_match_reader(self) -> bool:
        """Return whether reader key bytes match the TraceStore ingest metadata."""
        key_bytes = self.meta.get('key_bytes') if isinstance(self.meta, dict) else None
        if not isinstance(key_bytes, dict):
            logger.warning(
                '[FALLBACK] meta.json has no key_bytes in %s; section lookup falls back',
                self.store_dir,
            )
            return False
        try:
            ingest_key1 = int(key_bytes['key1'])
            ingest_key2 = int(key_bytes['key2'])
        except (KeyError, TypeError, ValueError):
            logger.warning(
                '[FALLBACK] Invalid key_bytes in %s/meta.json; section lookup falls back',
                self.store_dir,
            )
            return False
        if ingest_key1 != self.key1_byte or ingest_key2 != self.key2_byte:
            logger.warning(
                '[FALLBACK] TraceStore key_bytes (%s, %s) do not match reader (%s, %s) in %s',
                ingest_key1,
                ingest_key2,
                self.key1_byte,
                self.key2_byte,
                self.store_dir,
            )
            return False
        return True

    def _validate_key1_index(
        self,
        key1_values: np.ndarray,
        key1_offsets: np.ndarray,
        key1_counts: np.ndarray,
    ) -> str | None:
        """Return a validation error message for key1 index metadata, if any."""
        if key1_values.ndim != 1 or key1_offsets.ndim != 1 or key1_counts.ndim != 1:
            return 'index arrays must be 1D'
        if key1_values.size != key1_offsets.size or key1_values.size != key1_counts.size:
            return 'index arrays must have matching lengths'
        if np.any(key1_offsets < 0):
            return 'index offsets must be non-negative'
        if np.any(key1_counts <= 0):
            return 'index counts must be positive'
        if key1_offsets.size and np.any(np.diff(key1_offsets) < 0):
            return 'index offsets must be sorted in ascending order'
        if key1_values.size != np.unique(key1_values).size:
            return 'index key1_values must be unique'
        n_traces = int(self.traces.shape[0])
        key1_ends = key1_offsets + key1_counts
        if np.any(key1_ends > n_traces):
            return 'index offsets/counts exceed trace count'
        if key1_offsets.size and key1_offsets[0] != 0:
            return 'index offsets must start at zero'
        if key1_offsets.size and np.any(key1_offsets[1:] != key1_ends[:-1]):
            return 'index offsets/counts must form contiguous non-overlapping ranges'
        if key1_ends.size and key1_ends[-1] != n_traces:
            return 'index offsets/counts must cover all traces'
        return None

    def _get_key1_index(self) -> _Key1Index | None:
        """Lazily load and validate key1 lookup metadata when it is safe to use."""
        if self._key1_index_loaded:
            return self._key1_index

        self._key1_index_loaded = True
        if not self._meta_key_bytes_match_reader():
            return None

        index_path = self.store_dir / 'index.npz'
        if not index_path.exists():
            logger.warning(
                '[FALLBACK] Missing index.npz in %s; section lookup falls back',
                self.store_dir,
            )
            return None

        try:
            with np.load(index_path, allow_pickle=False) as idx:
                required = {'key1_values', 'key1_offsets', 'key1_counts'}
                if not required.issubset(idx.files):
                    missing = sorted(required.difference(idx.files))
                    logger.warning(
                        '[FALLBACK] index.npz missing %s in %s; section lookup falls back',
                        missing,
                        self.store_dir,
                    )
                    return None
                key1_values = np.asarray(idx['key1_values'], dtype=np.int64)
                key1_offsets = np.asarray(idx['key1_offsets'], dtype=np.int64)
                key1_counts = np.asarray(idx['key1_counts'], dtype=np.int64)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                '[FALLBACK] Failed to load index.npz in %s: %s',
                self.store_dir,
                exc,
            )
            return None

        validation_error = self._validate_key1_index(
            key1_values,
            key1_offsets,
            key1_counts,
        )
        if validation_error is not None:
            logger.warning(
                '[FALLBACK] Invalid key1 index in %s: %s',
                self.store_dir,
                validation_error,
            )
            return None

        key1_pos_by_value = {
            int(value): pos for pos, value in enumerate(key1_values.tolist())
        }
        self._key1_index = _Key1Index(
            key1_values=np.ascontiguousarray(key1_values, dtype=np.int64),
            key1_offsets=np.ascontiguousarray(key1_offsets, dtype=np.int64),
            key1_counts=np.ascontiguousarray(key1_counts, dtype=np.int64),
            key1_pos_by_value=key1_pos_by_value,
        )
        return self._key1_index

    def _get_key1_range(self, key1_val: int) -> tuple[int, int] | None:
        """Return ``(offset, count)`` for ``key1_val`` when index metadata is usable."""
        index = self._get_key1_index()
        if index is None:
            return None
        pos = index.key1_pos_by_value.get(int(key1_val))
        if pos is None:
            return None
        return int(index.key1_offsets[pos]), int(index.key1_counts[pos])

    def _get_sorted_to_original(self) -> np.ndarray | None:
        """Lazily load ``sorted_to_original`` from index metadata when available."""
        if self._sorted_to_original_loaded:
            return self._sorted_to_original

        index_path = self.store_dir / 'index.npz'
        if not index_path.exists():
            logger.warning(
                '[FALLBACK] Missing index.npz in %s; new headers keep original SEG-Y order',
                self.store_dir,
            )
            self._sorted_to_original_loaded = True
            return None

        with np.load(index_path) as idx:
            if 'sorted_to_original' not in idx.files:
                logger.warning(
                    '[FALLBACK] index.npz has no sorted_to_original in %s; new headers keep original SEG-Y order',
                    self.store_dir,
                )
                self._sorted_to_original_loaded = True
                return None
            self._sorted_to_original = np.asarray(
                idx['sorted_to_original'], dtype=np.int64
            )
        self._sorted_to_original_loaded = True
        return self._sorted_to_original

    def get_sorted_to_original(self) -> np.ndarray:
        """Return the ``sorted -> original`` trace index mapping."""
        sorted_to_original = self._get_sorted_to_original()
        if sorted_to_original is None:
            msg = f'sorted_to_original is missing in {self.store_dir}/index.npz'
            raise ValueError(msg)
        n_traces = int(self.traces.shape[0])
        if sorted_to_original.shape != (n_traces,):
            msg = (
                'sorted_to_original shape mismatch: '
                f'expected {(n_traces,)}, got {sorted_to_original.shape}'
            )
            raise ValueError(msg)
        return sorted_to_original

    def ensure_header(self, header_byte: int) -> np.ndarray:
        """Ensure the header array for ``header_byte`` exists on disk and return it."""
        path = self._header_path(header_byte)
        if path.exists():
            return np.load(path, mmap_mode='r')

        logger.info('Extracting header byte %s for %s', header_byte, self.store_dir)
        with segyio.open(
            self.meta['original_segy_path'],
            'r',
            ignore_geometry=True,
        ) as f:
            f.mmap()
            values = f.attributes(header_byte)[:].astype(np.int32)

        sorted_to_original = self._get_sorted_to_original()
        if sorted_to_original is not None:
            if values.shape[0] != sorted_to_original.shape[0]:
                msg = (
                    'Header length does not match sorted_to_original: '
                    f'header_byte={header_byte}, '
                    f'header_len={values.shape[0]}, '
                    f'sorted_to_original_len={sorted_to_original.shape[0]}'
                )
                raise ValueError(msg)
            values = values[sorted_to_original]

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

        key1_range = self._get_key1_range(key1_val)
        if key1_range is not None:
            offset, count = key1_range
            indices = np.arange(offset, offset + count, dtype=np.int64)
            self._trace_seq_cache[key1_val] = indices
            self._trace_seq_disp_cache[key1_val] = indices
            return indices, indices

        key1s = self.get_header(self.key1_byte)
        indices = np.flatnonzero(key1s == key1_val).astype(np.int64)
        logger.debug('%s indices found for key1_val: %s', len(indices), key1_val)
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
        index = self._get_key1_index()
        if index is not None:
            return index.key1_values
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

        key1_range = self._get_key1_range(key1_val)
        if key1_range is not None:
            offset, count = key1_range
            section = self.traces[offset : offset + count]
            view = SectionView(arr=section, dtype=section.dtype, scale=self.scale)
            self.section_cache[key1_val] = view
            return view

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


__all__ = ['TraceStoreSectionReader']
