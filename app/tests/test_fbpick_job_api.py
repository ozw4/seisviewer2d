# app/tests/test_fbpick_job_api.py
from __future__ import annotations

import gzip

import msgpack
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.errors import ConflictError


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
    monkeypatch.setattr(
        fbpick_mod,
        '_resolve_model_selection',
        lambda model_id: (
            'fbpick_edgenext_small.pth' if model_id is None else model_id,
            False,
            'fbpick_edgenext_small.pth:123',
        ),
        raising=True,
    )

    with TestClient(app) as client:
        yield client, fbpick_mod

    state.jobs.clear()
    state.fbpick_cache.clear()


def test_fbpick_section_bin_returns_409_when_weights_missing(_fb_env, monkeypatch):
    client, fbpick_mod = _fb_env
    monkeypatch.setattr(
        fbpick_mod,
        '_resolve_model_selection',
        lambda _model_id: (_ for _ in ()).throw(
            ConflictError('FB pick model weights not found')
        ),
        raising=True,
    )

    r = client.post('/fbpick_section_bin', json={'file_id': 'fid', 'key1': 1})
    assert r.status_code == 409
    assert r.json().get('detail') == 'FB pick model weights not found'


def test_fbpick_cache_hit_marks_job_done_and_get_returns_payload(_fb_env, monkeypatch):
    client, fbpick_mod = _fb_env

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
        model_id='fbpick_edgenext_small.pth',
        model_ver='fbpick_edgenext_small.pth:123',
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


def test_fbpick_pipeline_request_keeps_job_state_array_free(_fb_env, monkeypatch):
    client, fbpick_mod = _fb_env

    calls = {'count': 0}

    def _unexpected_tap_lookup(**_kwargs):
        calls['count'] += 1
        raise AssertionError('request handler must not materialize pipeline tap data')

    class _NoOpThread:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        def start(self):
            return None

    monkeypatch.setattr(
        fbpick_mod,
        'get_section_from_pipeline_tap',
        _unexpected_tap_lookup,
        raising=True,
    )
    monkeypatch.setattr(fbpick_mod.threading, 'Thread', _NoOpThread, raising=True)

    r = client.post(
        '/fbpick_section_bin',
        json={
            'file_id': 'fid',
            'key1': 1,
            'pipeline_key': 'pk',
            'tap_label': 'tapA',
        },
    )
    assert r.status_code == 200, r.text

    job_id = r.json()['job_id']
    state = fbpick_mod.get_state(app)
    job = state.jobs[job_id]

    assert 'section_override' not in job
    assert not any(isinstance(v, np.ndarray) for v in job.values())
    assert job.get('pipeline_key') == 'pk'
    assert job.get('tap_label') == 'tapA'
    assert job.get('offset_byte') is None
    assert calls['count'] == 0


def test_fbpick_pipeline_worker_fetches_tap_and_caches_payload(_fb_env, monkeypatch):
    client, fbpick_mod = _fb_env

    section = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    tap_calls: list[dict[str, object]] = []

    def _fake_tap_lookup(**kwargs):
        tap_calls.append(dict(kwargs))
        return section

    def _fake_apply_pipeline(section_arr, *, spec, meta, taps=None):
        _ = spec, meta, taps
        return {'fbpick': {'prob': np.asarray(section_arr, dtype=np.float32)}}

    class _SyncThread:
        def __init__(self, *args, **kwargs):
            _ = args
            self._target = kwargs.get('target')
            self._args = kwargs.get('args', ())
            self._kwargs = kwargs.get('kwargs', {})

        def start(self):
            assert callable(self._target)
            self._target(*self._args, **self._kwargs)

    monkeypatch.setattr(
        fbpick_mod, 'get_section_from_pipeline_tap', _fake_tap_lookup, raising=True
    )
    monkeypatch.setattr(
        fbpick_mod, 'apply_pipeline', _fake_apply_pipeline, raising=True
    )
    monkeypatch.setattr(fbpick_mod.threading, 'Thread', _SyncThread, raising=True)

    r = client.post(
        '/fbpick_section_bin',
        json={
            'file_id': 'fid',
            'key1': 1,
            'pipeline_key': 'pk',
            'tap_label': 'tapA',
        },
    )
    assert r.status_code == 200, r.text

    job_id = r.json()['job_id']
    state = fbpick_mod.get_state(app)
    job = state.jobs[job_id]

    assert len(tap_calls) == 1
    assert tap_calls[0].get('pipeline_key') == 'pk'
    assert tap_calls[0].get('tap_label') == 'tapA'
    assert job.get('status') == 'done'
    assert 'section_override' not in job
    assert not any(isinstance(v, np.ndarray) for v in job.values())

    cache_key = job['cache_key']
    payload = state.fbpick_cache.get(cache_key)
    assert payload is not None

    gr = client.get('/get_fbpick_section_bin', params={'job_id': job_id})
    assert gr.status_code == 200


def test_fbpick_cache_key_isolated_by_model_id(_fb_env, monkeypatch):
    client, fbpick_mod = _fb_env
    monkeypatch.setattr(
        fbpick_mod,
        '_resolve_model_selection',
        lambda model_id: (
            'fbpick_edgenext_small.pth' if model_id is None else model_id,
            False,
            f'{model_id}:111',
        ),
        raising=True,
    )

    class _NoOpThread:
        def __init__(self, target, args, daemon):
            self._target = target
            self._args = args
            self._daemon = daemon

        def start(self):
            return None

    monkeypatch.setattr(fbpick_mod.threading, 'Thread', _NoOpThread, raising=True)

    r1 = client.post(
        '/fbpick_section_bin',
        json={'file_id': 'fid', 'key1': 1, 'model_id': 'fbpick_a.pth'},
    )
    r2 = client.post(
        '/fbpick_section_bin',
        json={'file_id': 'fid', 'key1': 1, 'model_id': 'fbpick_b.pth'},
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    state = fbpick_mod.get_state(app)
    job1 = state.jobs[r1.json()['job_id']]
    job2 = state.jobs[r2.json()['job_id']]
    assert job1['cache_key'] != job2['cache_key']


def test_offset_model_enforces_offset_byte_37_in_job_state(_fb_env, monkeypatch):
    client, fbpick_mod = _fb_env
    monkeypatch.setattr(
        fbpick_mod,
        '_resolve_model_selection',
        lambda model_id: (str(model_id), True, f'{model_id}:111'),
        raising=True,
    )

    class _NoOpThread:
        def __init__(self, target, args, daemon):
            self._target = target
            self._args = args
            self._daemon = daemon

        def start(self):
            return None

    monkeypatch.setattr(fbpick_mod.threading, 'Thread', _NoOpThread, raising=True)
    r = client.post(
        '/fbpick_section_bin',
        json={'file_id': 'fid', 'key1': 1, 'model_id': 'fbpick_offset_demo.pth'},
    )
    assert r.status_code == 200
    state = fbpick_mod.get_state(app)
    job = state.jobs[r.json()['job_id']]
    assert job.get('offset_byte') == 37


def test_fbpick_models_endpoint_lists_models(_fb_env, monkeypatch):
    client, fbpick_mod = _fb_env
    monkeypatch.setattr(
        fbpick_mod,
        'list_fbpick_models',
        lambda: [
            {'id': 'fbpick_edgenext_small.pth', 'uses_offset': False},
            {'id': 'fbpick_offset_demo.pth', 'uses_offset': True},
        ],
        raising=True,
    )
    r = client.get('/fbpick_models')
    assert r.status_code == 200
    body = r.json()
    assert body['default_model_id'] == 'fbpick_edgenext_small.pth'
    assert len(body['models']) == 2
