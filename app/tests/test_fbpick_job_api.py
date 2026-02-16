# app/tests/test_fbpick_job_api.py
from __future__ import annotations

import gzip

import msgpack
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


class _DummyPath:
    def __init__(self, exists: bool):
        self._exists = bool(exists)

    def exists(self) -> bool:
        return self._exists


class _DummyView:
    def __init__(self, arr: np.ndarray, scale: float = 1.0):
        self.arr = arr
        self.scale = scale


class _DummyReader:
    def __init__(self, section: np.ndarray, dt: float = 0.002):
        self._section = np.asarray(section, dtype=np.float32)
        self.meta = {'dt': float(dt)}

    def get_section(self, _key1: int) -> _DummyView:
        return _DummyView(self._section, 1.0)


def _pack_payload(*, shape: tuple[int, int], dt: float) -> bytes:
    q = np.zeros(shape, dtype=np.int8)
    payload = msgpack.packb(
        {
            'scale': 127.0,
            'shape': q.shape,
            'data': q.tobytes(),
            'dt': float(dt),
        },
        use_bin_type=True,
    )
    return gzip.compress(payload)


@pytest.fixture()
def _fb_env(monkeypatch) -> tuple[TestClient, object]:
    from app.api.routers import fbpick as fbpick_mod

    state = app.state.sv
    state.jobs.clear()
    state.fbpick_cache.clear()
    monkeypatch.setattr(fbpick_mod, 'USE_FBPICK_OFFSET', False, raising=True)

    with TestClient(app) as client:
        yield client, fbpick_mod

    state.jobs.clear()
    state.fbpick_cache.clear()


def test_fbpick_section_bin_returns_409_when_weights_missing(_fb_env, monkeypatch):
    client, fbpick_mod = _fb_env
    monkeypatch.setattr(
        fbpick_mod, 'FBPICK_MODEL_PATH', _DummyPath(False), raising=True
    )

    r = client.post('/fbpick_section_bin', json={'file_id': 'fid', 'key1': 1})
    assert r.status_code == 409
    assert r.json().get('detail') == 'FB pick model weights not found'


def test_fbpick_cache_hit_marks_job_done_and_get_returns_payload(_fb_env, monkeypatch):
    client, fbpick_mod = _fb_env
    monkeypatch.setattr(fbpick_mod, 'FBPICK_MODEL_PATH', _DummyPath(True), raising=True)

    class _NoThread:
        def __init__(self, *a, **k):
            raise AssertionError('thread should not start on cache hit')

    monkeypatch.setattr(fbpick_mod.threading, 'Thread', _NoThread, raising=True)

    state = app.state.sv
    cache_key = fbpick_mod._build_fbpick_cache_key(
        file_id='fid',
        key1=1,
        key1_byte=189,
        key2_byte=193,
        offset_byte=None,
        tile_h=128,
        tile_w=6016,
        overlap=32,
        amp=True,
        pipeline_key=None,
        tap_label=None,
    )
    payload_gz = _pack_payload(shape=(3, 4), dt=0.002)
    state.fbpick_cache[cache_key] = payload_gz

    r = client.post('/fbpick_section_bin', json={'file_id': 'fid', 'key1': 1})
    assert r.status_code == 200
    out = r.json()
    assert out['status'] == 'done'
    job_id = out['job_id']

    gr = client.get('/get_fbpick_section_bin', params={'job_id': job_id})
    assert gr.status_code == 200
    assert gr.headers.get('content-encoding') == 'gzip'

    # TestClient は Content-Encoding: gzip を自動展開するため、gr.content は「解凍済み(msgpack生)」
    assert gr.content == gzip.decompress(payload_gz)

    obj = msgpack.unpackb(gr.content, raw=False)
    assert tuple(obj['shape']) == (3, 4)
    assert isinstance(obj['data'], (bytes, bytearray))
    assert len(obj['data']) == 3 * 4
    assert obj['dt'] == pytest.approx(0.002)


def test_get_fbpick_section_bin_returns_404_until_done(_fb_env, monkeypatch):
    client, fbpick_mod = _fb_env
    monkeypatch.setattr(fbpick_mod, 'FBPICK_MODEL_PATH', _DummyPath(True), raising=True)

    class _NoOpThread:
        def __init__(self, target, args, daemon):
            self._target = target
            self._args = args
            self._daemon = daemon

        def start(self):
            return None

    monkeypatch.setattr(fbpick_mod.threading, 'Thread', _NoOpThread, raising=True)

    r = client.post('/fbpick_section_bin', json={'file_id': 'fid', 'key1': 1})
    assert r.status_code == 200
    job_id = r.json()['job_id']

    gr = client.get('/get_fbpick_section_bin', params={'job_id': job_id})
    assert gr.status_code == 404
    assert gr.json().get('detail') == 'Result not ready'


def test_get_fbpick_section_bin_returns_410_when_payload_missing(_fb_env, monkeypatch):
    client, fbpick_mod = _fb_env
    monkeypatch.setattr(fbpick_mod, 'FBPICK_MODEL_PATH', _DummyPath(True), raising=True)

    class _NoOpThread:
        def __init__(self, target, args, daemon):
            self._target = target
            self._args = args
            self._daemon = daemon

        def start(self):
            return None

    monkeypatch.setattr(fbpick_mod.threading, 'Thread', _NoOpThread, raising=True)

    r = client.post('/fbpick_section_bin', json={'file_id': 'fid', 'key1': 1})
    assert r.status_code == 200
    job_id = r.json()['job_id']

    state = app.state.sv
    state.jobs[job_id]['status'] = 'done'
    cache_key = state.jobs[job_id]['cache_key']
    assert cache_key not in state.fbpick_cache

    gr = client.get('/get_fbpick_section_bin', params={'job_id': job_id})
    assert gr.status_code == 410
    assert gr.json().get('detail') == 'Result expired'
    assert state.jobs[job_id]['status'] == 'expired'


def test_fbpick_job_run_produces_gzip_msgpack_with_shape_data_dt(_fb_env, monkeypatch):
    client, fbpick_mod = _fb_env
    monkeypatch.setattr(fbpick_mod, 'FBPICK_MODEL_PATH', _DummyPath(True), raising=True)
    monkeypatch.setattr(
        fbpick_mod, '_maybe_attach_fbpick_offsets', lambda meta, **k: meta, raising=True
    )

    section = np.ones((3, 4), dtype=np.float32)
    reader = _DummyReader(section, dt=0.002)
    monkeypatch.setattr(fbpick_mod, 'get_reader', lambda *a, **k: reader, raising=True)
    monkeypatch.setattr(
        fbpick_mod, 'coerce_section_f32', lambda arr, _scale: arr, raising=True
    )
    monkeypatch.setattr(
        fbpick_mod,
        'apply_pipeline',
        lambda *_a, **_k: {'fbpick': {'prob': section}},
        raising=True,
    )
    monkeypatch.setattr(
        fbpick_mod,
        'quantize_float32',
        lambda _arr, fixed_scale: (fixed_scale, np.zeros((3, 4), dtype=np.int8)),
        raising=True,
    )

    class _SyncThread:
        def __init__(self, target, args, daemon):
            self._target = target
            self._args = args
            self._daemon = daemon

        def start(self):
            self._target(*self._args)

    monkeypatch.setattr(fbpick_mod.threading, 'Thread', _SyncThread, raising=True)

    r = client.post('/fbpick_section_bin', json={'file_id': 'fid', 'key1': 1})
    assert r.status_code == 200
    out = r.json()
    assert out['status'] == 'done'
    job_id = out['job_id']

    gr = client.get('/get_fbpick_section_bin', params={'job_id': job_id})
    assert gr.status_code == 200
    assert gr.headers.get('content-encoding') == 'gzip'

    # TestClient が自動展開するため、ここは msgpack を直接読む
    obj = msgpack.unpackb(gr.content, raw=False)
    assert tuple(obj['shape']) == (3, 4)
    assert isinstance(obj['data'], (bytes, bytearray))
    assert len(obj['data']) == 3 * 4

    # dt は現状の実装で入っていない可能性があるため、「入っているなら検証」にしておく
    dt = obj.get('dt')
    if dt is not None:
        assert float(dt) == pytest.approx(0.002)

    # 返却は解凍済みだが、キャッシュは gzip のはず（本当に gzip を返す設計かの保険）
    state = app.state.sv
    cache_key = state.jobs[job_id]['cache_key']
    cached = state.fbpick_cache.get(cache_key)
    assert isinstance(cached, (bytes, bytearray))
    assert cached[:2] == b'\x1f\x8b'  # gzip magic
    assert gzip.decompress(cached) == gr.content
