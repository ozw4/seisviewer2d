"""Utilities for ingesting SEG-Y files into the TraceStore layout."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import segyio


class SegyIngestor:
	"""Build TraceStore artifacts from a raw SEG-Y file."""

	@classmethod
	def from_segy(
		cls,
		path: str | Path,
		store_dir: str | Path,
		key1_byte: int = 189,
		key2_byte: int = 193,
		dtype: str = 'float32',
		quantize: bool = False,
		scale: float | None = None,
	) -> dict[str, Any]:
		"""Ingest ``path`` into ``store_dir`` and return the generated metadata."""
		segy_path = Path(path)
		if not segy_path.is_file():
			msg = f'SEG-Y file not found: {segy_path}'
			raise RuntimeError(msg)

		store_path = Path(store_dir)
		store_path.mkdir(parents=True, exist_ok=True)
		lock_path = store_path / '.build.lock'
		try:
			lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
		except FileExistsError as exc:
			msg = f'TraceStore build already in progress for {store_path}'
			raise RuntimeError(msg) from exc

		os.close(lock_fd)
		temp_paths: list[Path] = []

		def _cleanup() -> None:
			for tmp in temp_paths:
				if tmp.exists():
					tmp.unlink(missing_ok=True)

		try:
			with segyio.open(str(segy_path), 'r', ignore_geometry=True) as f:
				f.mmap()
				n_traces = int(f.tracecount)
				n_samples = len(f.samples)
				key1 = np.asarray(f.attributes(key1_byte)[:], dtype=np.int64)
				key2 = np.asarray(f.attributes(key2_byte)[:], dtype=np.int64)
				try:
					raw_dt = f.bin[segyio.BinField.Interval]
				except Exception:  # noqa: BLE001
					raw_dt = None

			if n_traces == 0:
				_cleanup()
				msg = f'SEG-Y file contains no traces: {segy_path}'
				raise RuntimeError(msg)

			original_index = np.arange(n_traces, dtype=np.int64)
			order = np.lexsort([original_index, key2, key1])
			key1_sorted = key1[order]
			key2_sorted = key2[order]
			final_dtype = np.dtype(dtype)

			scale_val: float | None = None
			if quantize:
				if final_dtype != np.dtype('int8'):
					_cleanup()
					msg = "Quantized output requires dtype='int8'"
					raise RuntimeError(msg)
				if scale is not None:
					scale_val = float(scale)
				else:
					max_abs = 0.0
					with segyio.open(
						str(segy_path), 'r', ignore_geometry=True
					) as f_scan:
						f_scan.mmap()
						for idx in range(n_traces):
							trace = np.asarray(f_scan.trace[idx], dtype=np.float32)
							if trace.size == 0:
								continue
							val = float(np.max(np.abs(trace)))
							max_abs = max(max_abs, val)
					if max_abs <= 0:
						scale_val = 1.0
					else:
						scale_val = 127.0 / max_abs

			traces_tmp = store_path / 'traces.npy.tmp'
			temp_paths.append(traces_tmp)
			mm = np.lib.format.open_memmap(
				str(traces_tmp),
				mode='w+',
				dtype=final_dtype,
				shape=(n_traces, n_samples),
			)
			try:
				chunk_size = 512
				with segyio.open(str(segy_path), 'r', ignore_geometry=True) as f_traces:
					f_traces.mmap()
					for chunk_start in range(0, n_traces, chunk_size):
						chunk_end = min(chunk_start + chunk_size, n_traces)
						chunk_indices = order[chunk_start:chunk_end]
						for row_offset, trace_idx in enumerate(chunk_indices):
							trace = np.asarray(
								f_traces.trace[int(trace_idx)], dtype=np.float32
							)
							if trace.shape[0] != n_samples:
								_cleanup()
								msg = 'Inconsistent trace length encountered during ingest'
								raise RuntimeError(msg)
							if quantize:
								if scale_val is None:
									_cleanup()
									msg = 'Quantization scale was not computed'
									raise RuntimeError(msg)
								q = np.clip(
									np.round(trace * scale_val),
									-127,
									127,
								).astype(np.int8)
								mm[chunk_start + row_offset] = q
							else:
								row = trace.astype(final_dtype, copy=False)
								mm[chunk_start + row_offset] = row
			finally:
				mm.flush()
				del mm

			traces_path = store_path / 'traces.npy'
			os.replace(traces_tmp, traces_path)
			temp_paths.remove(traces_tmp)

			headers1_tmp = store_path / f'headers_byte_{key1_byte}.npy.tmp'
			headers2_tmp = store_path / f'headers_byte_{key2_byte}.npy.tmp'
			temp_paths.extend([headers1_tmp, headers2_tmp])
			with open(headers1_tmp, 'wb') as f_headers1:
				np.save(f_headers1, key1_sorted.astype(np.int32, copy=False))
			with open(headers2_tmp, 'wb') as f_headers2:
				np.save(f_headers2, key2_sorted.astype(np.int32, copy=False))
			os.replace(headers1_tmp, store_path / f'headers_byte_{key1_byte}.npy')
			os.replace(headers2_tmp, store_path / f'headers_byte_{key2_byte}.npy')
			temp_paths.remove(headers1_tmp)
			temp_paths.remove(headers2_tmp)

			key1_values, key1_offsets, key1_counts = np.unique(
				key1_sorted,
				return_index=True,
				return_counts=True,
			)
			index_tmp = store_path / 'index.npz.tmp'
			temp_paths.append(index_tmp)
			with open(index_tmp, 'wb') as f_index:
				np.savez(
					f_index,
					key1_values=key1_values.astype(np.int32, copy=False),
					key1_offsets=key1_offsets.astype(np.int64, copy=False),
					key1_counts=key1_counts.astype(np.int64, copy=False),
					sorted_to_original=order.astype(np.int64, copy=False),
				)
			os.replace(index_tmp, store_path / 'index.npz')
			temp_paths.remove(index_tmp)

			meta_tmp = store_path / 'meta.json.tmp'
			temp_paths.append(meta_tmp)
			stat = segy_path.stat()
			dt_seconds = None
			if isinstance(raw_dt, (int, float)) and raw_dt > 0:
				dt_seconds = float(raw_dt) / 1_000_000.0
			meta: dict[str, Any] = {
				'schema_version': 1,
				'dtype': str(final_dtype),
				'n_traces': n_traces,
				'n_samples': n_samples,
				'key_bytes': {'key1': key1_byte, 'key2': key2_byte},
				'sorted_by': ['key1', 'key2'],
				'dt': dt_seconds,
				'original_segy_path': str(segy_path),
				'original_mtime': stat.st_mtime,
				'original_size': stat.st_size,
			}
			if quantize:
				meta['scale'] = scale_val
			with meta_tmp.open('w', encoding='utf-8') as fh:
				json.dump(meta, fh)
			os.replace(meta_tmp, store_path / 'meta.json')
			temp_paths.remove(meta_tmp)

			return meta
		except Exception as exc:
			_cleanup()
			msg = f'Failed to build TraceStore artifacts for {segy_path}: {exc}'
			raise RuntimeError(msg) from exc
		finally:
			_cleanup()
			lock_path.unlink(missing_ok=True)
