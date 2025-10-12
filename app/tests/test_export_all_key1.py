import io

import numpy as np
from fastapi.testclient import TestClient

from app.main import app

try:
	# app.main は `from api import endpoints` を使用するため、こちらを優先
	import api.endpoints as ep
except Exception:
	import app.api.endpoints as ep  # フォールバック（環境差異対策）


def test_export_all_key1_basic(monkeypatch):
	"""- key1 values are arbitrary (100, 200)
	- header says: 100→3 traces, 200→2 traces ⇒ width = 3
	- manual picks persisted by *actual key1 value* (not zero-based index)
	- verifies:
	    * matrix shape == (len(key1_values), max_traces)
	    * indices = round(time / dt)
	    * correct rows get filled (and others stay -1)
	    * list_picks was called with key1 values [100, 200]
	"""
	assert any(
		getattr(r, 'path', '') == '/export_manual_picks_all_npy'
		for r in app.router.routes
	), (
		f'route not found. routes={[getattr(r, "path", None) for r in app.router.routes]}'
	)

	class FakeReader:
		key1_byte = 189

		def get_key1_values(self):
			return [100, 200]

		def _get_header(self, byte):
			# 5 traces total: first 3 belong to key1=100, next 2 belong to key1=200
			return np.array([100, 100, 100, 200, 200], dtype=np.int32)

		@property
		def traces(self):
			# provide n_samples for clamping
			return np.zeros((5, 1000), dtype=np.float32)

	# ---- monkeypatch を全部先に当てる ----
	monkeypatch.setattr(
		ep, 'get_reader', lambda file_id, key1_byte, key2_byte: FakeReader()
	)
	monkeypatch.setattr(ep, '_filename_for_file_id', lambda file_id: 'lineA.sgy')
	monkeypatch.setattr(ep, 'get_dt_for_file', lambda file_id: 0.004)  # 4 ms

	calls = []

	def fake_list_picks(file_name, key1_idx, key1_byte):
		# Ensure we are called with *actual key1 value* (100, then 200)
		calls.append(key1_idx)
		if key1_idx == 100:
			return [{'trace': 0, 'time': 0.012}, {'trace': 2, 'time': 0.020}]  # -> 3, 5
		if key1_idx == 200:
			return [{'trace': 1, 'time': 0.0}]  # -> 0
		return []

	monkeypatch.setattr(ep.picks_by_name, 'list_picks', fake_list_picks)

	# ✨ ここがキモ：isinstance チェックを通すためにシンボルを差し替える
	monkeypatch.setattr(ep, 'TraceStoreSectionReader', FakeReader)
	# （SegySectionReader を触る必要はないが、保険で ↓ でも可）
	# monkeypatch.setattr(ep, 'SegySectionReader', FakeReader)

	# ---- その後で TestClient を生成 ----
	client = TestClient(app, raise_server_exceptions=False)

	r = client.get(
		'/export_manual_picks_all_npy',
		params={'file_id': 'X', 'key1_byte': 189, 'key2_byte': 193},
	)
	assert r.status_code == 200

	arr = np.load(io.BytesIO(r.content))
	# width=max(3,2)=3
	assert arr.shape == (2, 3)
	# dt=0.004 ⇒ 0.012/0.004=3, 0.020/0.004=5, 0.0/0.004=0
	assert arr[0].tolist() == [3, -1, 5]  # key1=100 row
	assert arr[1].tolist() == [-1, 0, -1]  # key1=200 row
	assert calls == [100, 200]


def test_export_all_key1_empty_is_all_minus1(monkeypatch):
	"""- 1つの key1 値しかないが、ピックは空
	- 幅は header により width=2
	- 返る行列は -1 埋め
	"""
	assert any(
		getattr(r, 'path', '') == '/export_manual_picks_all_npy'
		for r in app.router.routes
	), (
		f'route not found. routes={[getattr(r, "path", None) for r in app.router.routes]}'
	)

	class FakeReader:
		key1_byte = 189

		def get_key1_values(self):
			return [10]

		def _get_header(self, byte):
			return np.array([10, 10], dtype=np.int32)  # width=2

		@property
		def traces(self):
			return np.zeros((2, 100), dtype=np.float32)

	# ---- monkeypatch を全部先に当てる ----
	monkeypatch.setattr(
		ep, 'get_reader', lambda file_id, key1_byte, key2_byte: FakeReader()
	)
	monkeypatch.setattr(ep, '_filename_for_file_id', lambda file_id: 'lineB.sgy')
	monkeypatch.setattr(ep, 'get_dt_for_file', lambda file_id: 0.002)
	monkeypatch.setattr(
		ep.picks_by_name, 'list_picks', lambda file_name, key1_idx, key1_byte: []
	)

	# ✨ ここも同じく isinstance を通す
	monkeypatch.setattr(ep, 'TraceStoreSectionReader', FakeReader)

	# ---- その後で TestClient を生成 ----
	client = TestClient(app, raise_server_exceptions=False)

	r = client.get(
		'/export_manual_picks_all_npy',
		params={'file_id': 'Y', 'key1_byte': 189, 'key2_byte': 193},
	)
	assert r.status_code == 200

	arr = np.load(io.BytesIO(r.content))
	assert arr.shape == (1, 2)
	assert arr.tolist() == [[-1, -1]]
