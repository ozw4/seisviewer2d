"""Helpers for SEG-Y metadata such as sampling interval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HEADER_SAMPLE_INTERVAL_OFFSET = 3200 + 16
SAMPLE_INTERVAL_BYTES = 2
MICROSECONDS_PER_SECOND = 1_000_000.0

# ファイル毎のメタ情報を保持
FILE_REGISTRY: dict[str, dict[str, Any]] = {}


def read_segy_dt_seconds(path: str) -> float | None:
	"""Return the SEG-Y sampling interval in seconds, if available."""
	sample_path = Path(path)
	try:
		with sample_path.open('rb') as f:
			f.seek(HEADER_SAMPLE_INTERVAL_OFFSET)
			raw = f.read(SAMPLE_INTERVAL_BYTES)
		if len(raw) != SAMPLE_INTERVAL_BYTES:
			return None
		us = int.from_bytes(raw, byteorder='big', signed=False)
		if us <= 0:
			return None
		return us / MICROSECONDS_PER_SECOND
	except Exception:  # noqa: BLE001
		return None


def get_dt_for_file(file_id: str) -> float:
	"""Resolve the sampling interval in seconds for ``file_id``."""
	rec = FILE_REGISTRY.get(file_id)
	if not isinstance(rec, dict):
		rec = {}

	dt_val = rec.get('dt')
	if isinstance(dt_val, (int, float)) and dt_val > 0:
		return float(dt_val)

	# 1) 直接パスが分かっていればヘッダから読む
	path = rec.get('path')

	# 2) trace store から meta.json を参照して復元
	if not path:
		store_path = rec.get('store_path')
		if isinstance(store_path, str):
			meta_path = Path(store_path) / 'meta.json'
			try:
				meta = json.loads(meta_path.read_text())
			except Exception:  # noqa: BLE001
				meta = None
			if isinstance(meta, dict):
				meta_dt = meta.get('dt')
				if isinstance(meta_dt, (int, float)) and meta_dt > 0:
					rec['dt'] = float(meta_dt)
					FILE_REGISTRY[file_id] = rec
					return float(meta_dt)
				original = meta.get('original_segy_path')
				if isinstance(original, str):
					path = original
					rec['path'] = path

	dt = read_segy_dt_seconds(path) if path else None
	if not dt:
		raise RuntimeError('dt not found')
	rec['dt'] = dt
	FILE_REGISTRY[file_id] = rec
	return dt
