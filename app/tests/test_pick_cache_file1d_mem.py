# app/tests/test_pick_cache_file1d_mem.py
import importlib

import numpy as np
import pytest


@pytest.fixture
def cache(tmp_path, monkeypatch):
	# 書き込み先を一時ディレクトリへ
	monkeypatch.setenv('PICKS_NPY_DIR', str(tmp_path))
	# 環境変数を反映させるため再読込
	import app.utils.pick_cache_file1d_mem as mod

	importlib.reload(mod)
	return mod


def test_set_and_read_pairs(cache):
	ntr = 10
	fname = 'LineA.sgy'
	sec = np.array([2, 3, 7, 9], dtype=np.int64)  # セクション→大域TraceSeq

	cache.set_by_traceseq(fname, ntr, 3, 1.23)
	cache.set_by_traceseq(fname, ntr, 7, 4.56)

	pairs = cache.to_pairs_for_section(fname, ntr, sec)
	# ローカルindex: 3はsec内で1、7は2
	assert pairs[0]['trace'] == 1 and pytest.approx(pairs[0]['time'], rel=1e-6) == 1.23
	assert pairs[1]['trace'] == 2 and pytest.approx(pairs[1]['time'], rel=1e-6) == 4.56


def test_clear_by_traceseq(cache):
	ntr = 10
	fname = 'LineA.sgy'
	sec = np.array([2, 3, 7, 9], dtype=np.int64)

	cache.set_by_traceseq(fname, ntr, 3, 1.0)
	cache.set_by_traceseq(fname, ntr, 7, 2.0)
	cache.clear_by_traceseq(fname, ntr, 3)

	pairs = cache.to_pairs_for_section(fname, ntr, sec)
	traces = [p['trace'] for p in pairs]
	assert 1 not in traces  # 3(=local1)が消えている
	assert 2 in traces  # 7(=local2)は残る


def test_clear_section(cache):
	ntr = 10
	fname = 'LineA.sgy'
	sec = np.array([2, 3, 7, 9], dtype=np.int64)

	cache.set_by_traceseq(fname, ntr, 2, 0.1)
	cache.set_by_traceseq(fname, ntr, 3, 0.2)
	cache.clear_section(fname, ntr, sec)

	assert cache.to_pairs_for_section(fname, ntr, sec) == []


def test_resize_grow_then_shrink(cache):
	fname = 'LineB.sgy'
	# まず5本で作成し、index=4に書く
	cache.set_by_traceseq(fname, 5, 4, 9.99)

	# 8本に拡張（既存値は維持される想定）
	pairs = cache.to_pairs_for_section(fname, 8, np.array([4], dtype=np.int64))
	assert len(pairs) == 1 and pytest.approx(pairs[0]['time'], rel=1e-6) == 9.99

	# 3本に縮小（index=4は切り落とされる）
	pairs = cache.to_pairs_for_section(fname, 3, np.array([4], dtype=np.int64))
	assert pairs == []  # もう到達不可


def test_out_of_range_raises(cache):
	fname = 'LineC.sgy'
	with pytest.raises(RuntimeError):
		cache.set_by_traceseq(fname, 3, 99, 0.5)
