import json
from pathlib import Path

import numpy as np
import pytest

from app.utils.utils import TraceStoreSectionReader

KEY1 = 189
KEY2 = 193


def _write_min_store(
	tmp_path: Path, key1s: np.ndarray, key2s: np.ndarray, n_samples: int = 4
) -> Path:
	# 作業用ディレクトリ
	store = tmp_path / 'store'
	store.mkdir(parents=True, exist_ok=True)

	# ダミー traces と headers を保存（TraceStoreSectionReader は index.npz を参照しない）
	n_traces = int(key1s.size)
	traces = np.arange(n_traces * n_samples, dtype=np.float32).reshape(
		n_traces, n_samples
	)
	np.save(store / 'traces.npy', traces)
	np.save(store / f'headers_byte_{KEY1}.npy', key1s.astype(np.int32, copy=False))
	np.save(store / f'headers_byte_{KEY2}.npy', key2s.astype(np.int32, copy=False))
	np.savez(
		store / 'index.npz',
		key1_values=np.unique(key1s),
		key1_offsets=np.array([], dtype=np.int32),
		key1_counts=np.array([], dtype=np.int32),
	)

	meta = {
		'dt': 0.004,
		'key_bytes': {'key1': KEY1, 'key2': KEY2},
		'original_segy_path': 'dummy.sgy',
		'original_mtime': 0.0,
		'original_size': 0,
	}
	(store / 'meta.json').write_text(json.dumps(meta))
	return store


def test_trace_seq_display_matches_legacy(tmp_path: Path):
	key1s = np.array([1, 1, 2, 1, 2, 3, 3, 1, 2, 3], dtype=np.int32)
	key2s = np.array([5, 2, 9, 2, 1, 1, 1, 2, 5, 1], dtype=np.int32)
	store = _write_min_store(tmp_path, key1s, key2s)
	reader = TraceStoreSectionReader(store, KEY1, KEY2)

	for v in np.unique(key1s):
		indices = np.where(key1s == v)[0]
		expected = indices[np.argsort(key2s[indices], kind='stable')]
		got = reader.get_trace_seq_for_value(int(v), align_to='display')
		assert np.array_equal(expected, got)


def test_trace_seq_original_matches_indices(tmp_path: Path):
	key1s = np.array([10, 10, 20, 10, 20, 30, 30, 10, 20, 30], dtype=np.int32)
	key2s = np.array([5, 2, 9, 2, 1, 1, 1, 2, 5, 1], dtype=np.int32)
	store = _write_min_store(tmp_path, key1s, key2s)
	reader = TraceStoreSectionReader(store, KEY1, KEY2)

	for v in np.unique(key1s):
		expected = np.where(key1s == v)[0]
		got = reader.get_trace_seq_for_value(int(v), align_to='original')
		assert np.array_equal(expected, got)


def test_trace_seq_raises_for_missing_key1(tmp_path: Path):
	key1s = np.array([1, 1, 2, 3], dtype=np.int32)
	key2s = np.array([0, 1, 2, 3], dtype=np.int32)
	store = _write_min_store(tmp_path, key1s, key2s)
	reader = TraceStoreSectionReader(store, KEY1, KEY2)

	with pytest.raises(ValueError):
		reader.get_trace_seq_for_value(999999, align_to='display')


def test_get_n_samples(tmp_path: Path):
	key1s = np.array([1, 1, 2], dtype=np.int32)
	key2s = np.array([0, 1, 2], dtype=np.int32)
	store = _write_min_store(tmp_path, key1s, key2s, n_samples=7)
	reader = TraceStoreSectionReader(store, KEY1, KEY2)
	assert reader.get_n_samples() == 7
