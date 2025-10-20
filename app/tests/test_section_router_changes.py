# app/tests/test_section_router_changes.py
import gzip
import json

import msgpack
import numpy as np
import pytest

from app.api.routers import section as sec
from app.utils.utils import SectionView


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
	# Ensure a clean FILE_REGISTRY and harmless dt function for binary endpoints.
	sec.FILE_REGISTRY.clear()
	monkeypatch.setattr(sec, 'get_dt_for_file', lambda _fid: 0.004, raising=True)
	yield
	sec.FILE_REGISTRY.clear()


# -------------------------
# Test helpers (fake readers)
# -------------------------


def _make_stub_reader(key1s: np.ndarray, key2s: np.ndarray):
	"""Minimal reader stub implementing the unified public API used by section router:
	- get_key1_values()
	- get_trace_seq_for_value(...), with legacy alias get_trace_seq_for_section(...)
	- get_section(key1_val)  (returns synthetic data)
	Also exposes .ntraces and optional .traces.shape fallback like legacy tests.
	"""
	r = type('_StubReader', (), {})()
	r.path = None
	r.key1_byte = 189
	r.key2_byte = 193
	r.section_cache = {}
	r._trace_seq_cache = {}
	r._trace_seq_disp_cache = {}
	r.key1s = np.asarray(key1s, dtype=np.int32)
	r.key2s = np.asarray(key2s, dtype=np.int32)
	r.unique_key1 = np.unique(r.key1s)
	r.ntraces = int(r.key1s.size)

	def get_trace_seq_for_value(key1_val: int, align_to: str = 'display'):
		idx = np.flatnonzero(r.key1s == key1_val).astype(np.int64)
		if idx.size == 0:
			raise ValueError(f'Key1 value {key1_val} not found')
		if align_to == 'original':
			return idx
		order = np.argsort(r.key2s[idx], kind='stable')
		return idx[order]

	# Legacy alias kept for older call sites inside router (if any remain)
	def get_trace_seq_for_section(key1_val: int, align_to: str = 'display'):
		return get_trace_seq_for_value(key1_val, align_to=align_to)

	def get_key1_values():
		return r.unique_key1

	def get_section(key1_val: int):
		idx = np.where(r.key1s == key1_val)[0]
		if idx.size == 0:
			raise ValueError('Key1 value not found')
		order = np.argsort(r.key2s[idx], kind='stable')
		sorted_idx = idx[order]
		# synthetic traces: (n_traces, n_samples) = (len, 4)
		arr = (np.arange(sorted_idx.size)[:, None] + np.array([0, 1, 2, 3])).astype(
			np.float32
		)
		# Router expects SectionView(arr, dtype, scale)
		return SectionView(arr=arr, dtype=arr.dtype, scale=None)

	r.get_trace_seq_for_value = get_trace_seq_for_value  # type: ignore[attr-defined]
	r.get_trace_seq_for_section = get_trace_seq_for_section  # type: ignore[attr-defined]
	r.get_key1_values = get_key1_values  # type: ignore[attr-defined]
	r.get_section = get_section  # type: ignore[attr-defined]
	return r


def _make_tracestore_like_reader(key1s: np.ndarray, key2s: np.ndarray):
	"""TraceStore-like stub:
	- Provides public get_header(byte) only (no _get_header),
	- get_key1_values(), get_section(key1_val) returning SectionView,
	- meta['n_traces'] present.
	This verifies the router uses the public API.
	"""
	t = type('_TraceStoreLike', (), {})()
	t.store_dir = None
	t.key1_byte = 189
	t.key2_byte = 193
	t.meta = {'n_traces': int(np.asarray(key1s).size)}
	_k1 = np.asarray(key1s, dtype=np.int32)
	_k2 = np.asarray(key2s, dtype=np.int32)

	def get_header(byte: int):
		if byte == t.key1_byte:
			return _k1
		if byte == t.key2_byte:
			return _k2
		# dummy header
		return np.zeros_like(_k1)

	def get_key1_values():
		return np.unique(_k1)

	def get_section(key1_val: int):
		idx = np.where(_k1 == key1_val)[0]
		if idx.size == 0:
			raise ValueError('Key1 value not found')
		order = np.argsort(_k2[idx], kind='stable')
		sorted_idx = idx[order]
		# synthetic traces: (n_traces, n_samples) = (len, 3)
		arr = (np.arange(sorted_idx.size)[:, None] + np.array([0, 1, 2])).astype(
			np.float32
		)
		return SectionView(arr=arr, dtype=arr.dtype, scale=None)

	t.get_header = get_header  # type: ignore[attr-defined]
	t.get_key1_values = get_key1_values  # type: ignore[attr-defined]
	t.get_section = get_section  # type: ignore[attr-defined]
	# Intentionally no _get_header attribute: test ensures public method is used.
	return t


# -------------------------
# Tests
# -------------------------


def test_get_ntraces_for_prefers_get_reader_and_fallbacks(monkeypatch):
	# Always mutate the same dict object; don't rebind FILE_REGISTRY.
	assert isinstance(sec.FILE_REGISTRY, dict)

	# 1) get_reader が返す reader を優先
	sec.FILE_REGISTRY.clear()
	sec.FILE_REGISTRY.update({'f1': {}})

	r = _make_stub_reader(
		key1s=np.array([1, 1, 2, 3, 3], dtype=np.int32),
		key2s=np.array([5, 2, 9, 1, 1], dtype=np.int32),
	)
	monkeypatch.setattr(sec, 'get_reader', lambda file_id, kb1, kb2: r)
	assert sec.get_ntraces_for('f1') == 5

	# 2) get_reader が失敗した場合は registry.meta にフォールバック
	sec.FILE_REGISTRY.clear()
	sec.FILE_REGISTRY.update({'f2': {'meta': {'n_traces': 8}}})

	def boom(*a, **k):
		raise RuntimeError('no reader')

	monkeypatch.setattr(sec, 'get_reader', boom)
	assert 'f2' in sec.FILE_REGISTRY
	assert sec.get_ntraces_for('f2') == 8

	# 3) reader.meta['n_traces'] / TraceStore 風のフォールバック
	t = _make_tracestore_like_reader(
		key1s=np.array([10, 10, 20, 20, 30], dtype=np.int32),
		key2s=np.array([1, 2, 3, 4, 5], dtype=np.int32),
	)
	sec.FILE_REGISTRY.clear()
	sec.FILE_REGISTRY.update({'f2': {}})
	monkeypatch.setattr(sec, 'get_reader', lambda file_id, kb1, kb2: t)
	assert sec.get_ntraces_for('f2') == 5

	# 4) traces.shape[0] フォールバック
	r2 = _make_stub_reader(
		key1s=np.array([7, 8, 9], dtype=np.int32),
		key2s=np.array([0, 0, 0], dtype=np.int32),
	)
	r2.ntraces = None  # type: ignore[attr-defined]

	class _Traces:
		shape = (3, 100)

	r2.traces = _Traces()  # type: ignore[attr-defined]
	sec.FILE_REGISTRY.clear()
	sec.FILE_REGISTRY.update({'f3': {}})
	monkeypatch.setattr(sec, 'get_reader', lambda file_id, kb1, kb2: r2)
	assert sec.get_ntraces_for('f3') == 3


def test_get_trace_seq_for_with_stub_reader_matches_legacy(monkeypatch):
	key1s = np.array([1, 1, 2, 1, 2, 3, 3, 1, 2, 3], dtype=np.int32)
	key2s = np.array([5, 2, 9, 2, 1, 1, 1, 2, 5, 1], dtype=np.int32)
	r = _make_stub_reader(key1s, key2s)
	sec.FILE_REGISTRY['lineA'] = {'reader': r}

	# ensure per-request reader is used
	monkeypatch.setattr(sec, 'get_reader', lambda fid, kb1, kb2: r, raising=True)

	vals = r.get_key1_values()
	for v in vals:
		seq = sec.get_trace_seq_for('lineA', key1_val=int(v), key1_byte=189)
		indices = np.where(r.key1s == v)[0]
		expected = indices[np.argsort(r.key2s[indices], kind='stable')]
		assert np.array_equal(seq, expected)


def test_get_trace_seq_for_with_tracestore_uses_public_get_header(monkeypatch):
	key1s = np.array([9, 9, 9, 8, 8, 7], dtype=np.int32)
	key2s = np.array([3, 1, 2, 5, 4, 0], dtype=np.int32)
	t = _make_tracestore_like_reader(key1s, key2s)
	sec.FILE_REGISTRY['lineB'] = {'reader': t}

	# ensure per-request reader is used
	monkeypatch.setattr(sec, 'get_reader', lambda fid, kb1, kb2: t, raising=True)

	# unique_key1 = [7,8,9] → value=9
	seq = sec.get_trace_seq_for('lineB', key1_val=9, key1_byte=189)
	idx = np.where(key1s == 9)[0]
	expected = idx[np.argsort(key2s[idx], kind='stable')]
	assert np.array_equal(seq, expected)

	# also verify missing value raises ValueError
	with pytest.raises(ValueError):
		sec.get_trace_seq_for('lineB', key1_val=999, key1_byte=189)


def test_get_section_returns_json_for_value(monkeypatch):
	# Reader returns specific values and captures the key1_val it received.
	received = {'val': None}

	class _StubReader:
		key1_byte = 189
		key2_byte = 193

		def get_key1_values(self):
			return np.array([10, 20, 30], dtype=np.int32)

		def get_section(self, key1_val: int):
			received['val'] = key1_val
			arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
			return SectionView(arr=arr, dtype=arr.dtype, scale=None)

	monkeypatch.setattr(
		sec, 'get_reader', lambda fid, kb1, kb2: _StubReader(), raising=True
	)
	resp = sec.get_section(file_id='f', key1_byte=189, key2_byte=193, key1_val=20)
	data = json.loads(resp.body)
	assert data['section'] == [[1.0, 2.0], [3.0, 4.0]]
	assert received['val'] == 20

	# missing value → HTTPException(400)
	class _StubReader2:
		key1_byte = 189
		key2_byte = 193

		def get_section(self, key1_val: int):
			raise ValueError(f'key1 {key1_val} not found')

	monkeypatch.setattr(
		sec, 'get_reader', lambda fid, kb1, kb2: _StubReader2(), raising=True
	)
	with pytest.raises(Exception) as ei:
		sec.get_section(file_id='f', key1_byte=189, key2_byte=193, key1_val=99)
	# (FastAPI HTTPException) don't overfit type; message check is enough
	assert 'key1 99 not found' in str(ei.value)


def test_get_section_bin_happy_path(monkeypatch):
	class _StubReader:
		key1_byte = 189
		key2_byte = 193

		def get_key1_values(self):
			return np.array([111], dtype=np.int32)

		def get_section(self, key1_val: int):
			# 2 traces x 4 samples
			assert key1_val == 111
			arr = np.array(
				[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32
			)
			return SectionView(arr=arr, dtype=arr.dtype, scale=None)

	monkeypatch.setattr(
		sec, 'get_reader', lambda fid, kb1, kb2: _StubReader(), raising=True
	)
	res = sec.get_section_bin(file_id='f', key1_val=111, key1_byte=189, key2_byte=193)
	assert res.headers.get('Content-Encoding') == 'gzip'

	payload = msgpack.unpackb(gzip.decompress(res.body))
	assert (
		'scale' in payload
		and 'shape' in payload
		and 'data' in payload
		and 'dt' in payload
	)
	# data length equals product of shape (int8 bytes)
	h, w = payload['shape']
	assert len(payload['data']) == (h * w)


def test_get_section_window_bin_happy_path(monkeypatch):
	class _StubReader:
		key1_byte = 189
		key2_byte = 193

		def get_key1_values(self):
			return np.array([7], dtype=np.int32)

		def get_section(self, key1_val: int):
			# 3 traces x 5 samples
			arr = np.array(
				[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [1, 2, 3, 4, 5]], dtype=np.float32
			)
			return SectionView(arr=arr, dtype=arr.dtype, scale=None)

	monkeypatch.setattr(
		sec, 'get_reader', lambda fid, kb1, kb2: _StubReader(), raising=True
	)
	res = sec.get_section_window_bin(
		file_id='f',
		key1_val=7,
		key1_byte=189,
		key2_byte=193,
		offset_byte=None,
		x0=0,
		x1=2,
		y0=1,
		y1=3,
		step_x=1,
		step_y=1,
		pipeline_key=None,
		tap_label=None,
	)
	assert res.headers.get('Content-Encoding') == 'gzip'
	payload = msgpack.unpackb(gzip.decompress(res.body))
	assert (
		'scale' in payload
		and 'shape' in payload
		and 'data' in payload
		and 'dt' in payload
	)
	h, w = payload['shape']
	assert len(payload['data']) == (h * w)
