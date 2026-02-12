"""Section-window construction service."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import numpy as np

from app.api._helpers import (
	EXPECTED_SECTION_NDIM,
	apply_scaling_from_baseline,
	coerce_section_f32,
	get_reader,
	get_section_from_pipeline_tap,
)
from app.api.binary_codec import pack_quantized_array_gzip
from app.utils.segy_meta import FILE_REGISTRY, get_dt_for_file
from app.utils.utils import SectionView, TraceStoreSectionReader


class SectionServiceInternalError(RuntimeError):
	"""Raised when service detects an internal data/state inconsistency."""


def _load_section_view(
	*,
	file_id: str,
	key1_val: int,
	key1_byte: int,
	key2_byte: int,
	offset_byte: int | None,
	pipeline_key: str | None,
	tap_label: str | None,
	reader_getter: Callable[[str, int, int], TraceStoreSectionReader],
	pipeline_section_getter: Callable[..., np.ndarray],
) -> tuple[SectionView, TraceStoreSectionReader | None]:
	if pipeline_key and tap_label:
		section_arr = pipeline_section_getter(
			file_id=file_id,
			key1_val=key1_val,
			key1_byte=key1_byte,
			pipeline_key=pipeline_key,
			tap_label=tap_label,
			offset_byte=offset_byte,
		)
		return (
			SectionView(arr=section_arr, dtype=section_arr.dtype, scale=None),
			None,
		)
	reader = reader_getter(file_id, key1_byte, key2_byte)
	return reader.get_section(key1_val), reader


def _resolve_store_dir(
	*, file_id: str, reader: TraceStoreSectionReader | None
) -> str:
	registry_entry = FILE_REGISTRY.get(file_id)
	store_dir: str | None = None
	if isinstance(registry_entry, dict):
		maybe_store = registry_entry.get('store_path')
		if isinstance(maybe_store, str):
			store_dir = maybe_store
	else:
		maybe_store = getattr(registry_entry, 'store_path', None)
		if isinstance(maybe_store, (str, Path)):
			store_dir = str(maybe_store)
	if store_dir is None and reader is not None:
		maybe_store = getattr(reader, 'store_dir', None)
		if isinstance(maybe_store, (str, Path)):
			store_dir = str(maybe_store)
	if store_dir is None:
		raise SectionServiceInternalError('Trace store path unavailable')
	return store_dir


def build_section_window_payload(
	*,
	file_id: str,
	key1_val: int,
	key1_byte: int,
	key2_byte: int,
	offset_byte: int | None,
	x0: int,
	x1: int,
	y0: int,
	y1: int,
	step_x: int,
	step_y: int,
	transpose: bool,
	pipeline_key: str | None,
	tap_label: str | None,
	scaling_mode: Literal['amax', 'tracewise'],
	reader_getter: Callable[[str, int, int], TraceStoreSectionReader] = get_reader,
	pipeline_section_getter: Callable[..., np.ndarray] = get_section_from_pipeline_tap,
	dt_resolver: Callable[[str], float] = get_dt_for_file,
) -> bytes:
	"""Build the compressed binary payload for a section window."""
	mode = scaling_mode.lower()
	if mode not in {'amax', 'tracewise'}:
		raise ValueError('Unsupported scaling mode')

	section_view, reader = _load_section_view(
		file_id=file_id,
		key1_val=key1_val,
		key1_byte=key1_byte,
		key2_byte=key2_byte,
		offset_byte=offset_byte,
		pipeline_key=pipeline_key,
		tap_label=tap_label,
		reader_getter=reader_getter,
		pipeline_section_getter=pipeline_section_getter,
	)
	base = section_view.arr
	if base.ndim != EXPECTED_SECTION_NDIM:
		raise SectionServiceInternalError('Section data must be 2D')

	n_traces, n_samples = base.shape
	if not (0 <= x0 <= x1 < n_traces):
		raise ValueError('Trace range out of bounds')
	if not (0 <= y0 <= y1 < n_samples):
		raise ValueError('Sample range out of bounds')
	if step_x < 1 or step_y < 1:
		raise ValueError('Steps must be >= 1')

	sub = base[x0 : x1 + 1 : step_x, y0 : y1 + 1 : step_y]
	if sub.size == 0:
		raise ValueError('Requested window is empty')

	prepared = coerce_section_f32(sub, section_view.scale)
	store_dir = _resolve_store_dir(file_id=file_id, reader=reader)
	prepared = apply_scaling_from_baseline(
		prepared,
		scaling=mode,
		file_id=file_id,
		key1_val=key1_val,
		store_dir=store_dir,
		x0=x0,
		x1=x1,
		step_x=step_x,
	)
	view = prepared.T if transpose else prepared
	window_view = np.ascontiguousarray(view, dtype=np.float32)
	dt_val = dt_resolver(file_id)
	return pack_quantized_array_gzip(
		window_view,
		scale=None,
		dt=dt_val,
		extra=None,
	)


__all__ = ['SectionServiceInternalError', 'build_section_window_payload']
