# app/tests/test_ingest.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.utils.ingest import SegyIngestor
from app.utils.utils import TraceStoreSectionReader


class _Attr:
	def __init__(self, arr: np.ndarray) -> None:
		self._arr = np.asarray(arr)

	def __getitem__(self, _sl):
		# segyio.attributes(... )[:] の呼び出しに対応
		return self._arr


class _FakeBin:
	def __init__(self, interval_us: int | None) -> None:
		self._interval_us = interval_us

	def __getitem__(self, _key):
		# segyio.BinField.Interval で来る
		return self._interval_us


class _TraceAccessor:
	def __init__(self, traces: np.ndarray) -> None:
		self._traces = np.asarray(traces)

	def __getitem__(self, i: int):
		return self._traces[int(i)]


class _FakeSegy:
	def __init__(
		self,
		traces: np.ndarray,  # (n_traces, n_samples) float32
		key1: np.ndarray,  # (n_traces,) int
		key2: np.ndarray,  # (n_traces,) int
		interval_us: int | None,
	) -> None:
		self._traces = np.asarray(traces)
		self._key1 = np.asarray(key1)
		self._key2 = np.asarray(key2)
		self.tracecount = int(self._traces.shape[0])
		self.samples = np.arange(self._traces.shape[1], dtype=np.int32)
		self.bin = _FakeBin(interval_us)
		self.trace = _TraceAccessor(self._traces)

	def mmap(self):
		# no-op（互換のため）
		return None

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		return False

	def attributes(self, byte: int):
		# 189 → key1, 193 → key2 として扱う
		if int(byte) == 189:
			return _Attr(self._key1)
		if int(byte) == 193:
			return _Attr(self._key2)
		# その他のバイト要求はゼロ配列
		return _Attr(np.zeros_like(self._key1))

	def trace(self, i: int):
		return self._traces[int(i)]


def _patch_segyio(
	monkeypatch,
	traces: np.ndarray,
	key1: np.ndarray,
	key2: np.ndarray,
	dt_us: int | None = 2000,
):
	"""Monkeypatch segyio.open と segyio.BinField.Interval を差し替え。"""
	import segyio  # type: ignore

	def _open_stub(_path, _mode='r', ignore_geometry=True):
		# ingest中に複数回 open されるため、都度 fresh なオブジェクトを返す
		return _FakeSegy(traces=traces, key1=key1, key2=key2, interval_us=dt_us)

	monkeypatch.setattr(segyio, 'open', _open_stub, raising=True)

	class _BF:
		Interval = 'Interval'

	monkeypatch.setattr(segyio, 'BinField', _BF, raising=True)


def test_ingest_builds_artifacts_float32(tmp_path: Path, monkeypatch):
	# 合成データ
	traces = np.array(
		[
			[1.0, 0.5, -0.5, 0.0],  # idx 0
			[2.0, 0.0, 1.0, -1.0],  # idx 1
			[0.1, 0.2, 0.3, 0.4],  # idx 2
			[3.0, 3.0, 3.0, 3.0],  # idx 3
			[-1.0, -0.1, 0.0, 0.1],  # idx 4
		],
		dtype=np.float32,
	)
	key1 = np.array([10, 20, 10, 20, 10], dtype=np.int32)
	key2 = np.array([2, 1, 3, 2, 1], dtype=np.int32)
	# 期待ソート順（key1, key2, original_index）
	# (10,1)->4, (10,2)->0, (10,3)->2, (20,1)->1, (20,2)->3
	expected_order = np.array([4, 0, 2, 1, 3], dtype=np.int64)

	_patch_segyio(monkeypatch, traces, key1, key2, dt_us=2000)

	# ダミーseg yファイル（存在チェック用）
	segy_path = tmp_path / 'dummy.segy'
	segy_path.write_bytes(b'stub')

	store_dir = tmp_path / 'store'
	SegyIngestor.from_segy(
		path=segy_path,
		store_dir=store_dir,
		key1_byte=189,
		key2_byte=193,
		dtype='float32',
		quantize=False,
	)

	# 生成物の存在確認
	traces_path = store_dir / 'traces.npy'
	h1 = store_dir / 'headers_byte_189.npy'
	h2 = store_dir / 'headers_byte_193.npy'
	index_path = store_dir / 'index.npz'
	meta_path = store_dir / 'meta.json'
	for p in [traces_path, h1, h2, index_path, meta_path]:
		assert p.exists(), f'missing: {p}'

	# 余計な拡張子の混入がないか（.tmp.npy / .tmp.npz など）
	assert not list(store_dir.glob('*.tmp')), 'Temporary files should be cleaned up'
	assert not list(store_dir.glob('*.tmp.npy')), 'Should not leave *.tmp.npy'
	assert not list(store_dir.glob('*.tmp.npz')), 'Should not leave *.tmp.npz'

	# traces の中身（順序/shape/dtype）
	mm = np.load(traces_path, mmap_mode='r')
	assert mm.shape == traces.shape
	assert mm.dtype == np.float32
	np.testing.assert_allclose(mm[:], traces[expected_order], rtol=0, atol=0)

	# headers
	k1 = np.load(h1, mmap_mode='r')
	k2 = np.load(h2, mmap_mode='r')
	np.testing.assert_array_equal(k1, key1[expected_order].astype(np.int32))
	np.testing.assert_array_equal(k2, key2[expected_order].astype(np.int32))

	# index
	idx = np.load(index_path)
	key1_values = idx['key1_values']
	key1_offsets = idx['key1_offsets']
	key1_counts = idx['key1_counts']
	sorted_to_original = idx['sorted_to_original']
	np.testing.assert_array_equal(sorted_to_original, expected_order)
	np.testing.assert_array_equal(key1_values, np.array([10, 20], dtype=np.int32))
	np.testing.assert_array_equal(key1_offsets, np.array([0, 3], dtype=np.int64))
	np.testing.assert_array_equal(key1_counts, np.array([3, 2], dtype=np.int64))

	# meta
	with meta_path.open('r', encoding='utf-8') as fh:
		m = json.load(fh)
	assert m['dtype'].startswith('float32')
	assert m['n_traces'] == traces.shape[0]
	assert m['n_samples'] == traces.shape[1]
	assert m['key_bytes'] == {'key1': 189, 'key2': 193}
	assert m['sorted_by'] == ['key1', 'key2']
	assert m['dt'] == pytest.approx(0.002, rel=0, abs=1e-12)  # 2000 μs → 0.002 s
	assert 'original_segy_path' in m
	assert 'original_mtime' in m
	assert 'original_size' in m

	# Reader で実読（key1=10 のセクションが 3 本、key2 昇順で並ぶ）
	reader = TraceStoreSectionReader(store_dir, key1_byte=189, key2_byte=193)
	view = reader.get_section(10)
	assert view.arr.shape == (3, traces.shape[1])
	# 期待順の最初の3本は indices [4,0,2]
	np.testing.assert_allclose(view.arr[:], traces[[4, 0, 2]], rtol=0, atol=0)


def test_ingest_quantize_int8_with_auto_scale(tmp_path: Path, monkeypatch):
	# 値幅広めにして量子化の効果を確認
	traces = np.array(
		[
			[0.0, 1.0, -2.0, 3.0],
			[4.0, -5.0, 6.0, -7.0],
		],
		dtype=np.float32,
	)
	key1 = np.array([10, 10], dtype=np.int32)
	key2 = np.array([1, 2], dtype=np.int32)
	expected_order = np.array([0, 1], dtype=np.int64)

	_patch_segyio(monkeypatch, traces, key1, key2, dt_us=1000)

	segy_path = tmp_path / 'dummy2.segy'
	segy_path.write_bytes(b'stub')

	store_dir = tmp_path / 'store_q'
	SegyIngestor.from_segy(
		path=segy_path,
		store_dir=store_dir,
		key1_byte=189,
		key2_byte=193,
		dtype='int8',
		quantize=True,  # 自動スケール計算
	)

	# traces は int8、範囲は [-127,127]
	mm = np.load(store_dir / 'traces.npy', mmap_mode='r')
	assert mm.dtype == np.int8
	assert np.max(mm) <= 127 and np.min(mm) >= -127

	# メタスケールの妥当性（最大絶対値に対応する）
	with (store_dir / 'meta.json').open('r', encoding='utf-8') as fh:
		m = json.load(fh)
	scale = float(m['scale'])
	assert scale > 0

	# 期待量子化値と一致（order はそのまま）
	q_expected = np.clip(np.round(traces * scale), -127, 127).astype(np.int8)
	np.testing.assert_array_equal(mm[:], q_expected[expected_order])


def test_ingest_respects_build_lock(tmp_path: Path, monkeypatch):
	traces = np.zeros((1, 4), dtype=np.float32)
	key1 = np.array([10], dtype=np.int32)
	key2 = np.array([1], dtype=np.int32)
	_patch_segyio(monkeypatch, traces, key1, key2, dt_us=None)

	segy_path = tmp_path / 'dummy3.segy'
	segy_path.write_bytes(b'stub')

	store_dir = tmp_path / 'store_locked'
	store_dir.mkdir(parents=True, exist_ok=True)
	# 既存ロックを配置
	(store_dir / '.build.lock').write_text('lock')

	with pytest.raises(RuntimeError):
		SegyIngestor.from_segy(
			path=segy_path,
			store_dir=store_dir,
			key1_byte=189,
			key2_byte=193,
		)

	# ロックはそのまま残っている（ingest 側は「開始時に作って finally で消す」運用）
	assert (store_dir / '.build.lock').exists()
