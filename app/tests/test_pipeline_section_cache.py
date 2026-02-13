# app/tests/test_pipeline_section_cache.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@dataclass
class _View:
    arr: np.ndarray
    scale: float


class _DummyReader:
    def __init__(self, arr: np.ndarray):
        self._arr = arr
        self.meta = {'dt': 0.002}

    def get_section(self, _key1_val: int) -> _View:
        return _View(arr=self._arr, scale=1.0)


def _clean_window_and_hash(
    window: dict[str, int | float], section: np.ndarray
) -> tuple[dict, str]:
    tr_min = int(window.get('tr_min', 0))
    tr_max = int(window.get('tr_max', section.shape[0]))
    t_min = int(window.get('t_min', 0))
    t_max = int(window.get('t_max', section.shape[1]))
    clean = {'tr_min': tr_min, 'tr_max': tr_max, 't_min': t_min, 't_max': t_max}
    w_hash = hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:8]
    return clean, w_hash


@pytest.fixture()
def _pipeline_env(monkeypatch):
    """
    - get_reader: ダミー reader を返す
    - apply_pipeline: 呼び出し回数/引数を記録しつつ、section に依存した値を返す
    - pipeline_key: 固定値にする（spec 由来の差を排除）
    - _maybe_attach_fbpick_offsets / coerce_section_f32: 影響を排除
    """
    from app.api.routers import pipeline as pipeline_mod

    state = app.state.sv
    state.pipeline_tap_cache.clear()

    section = np.arange(50, dtype=np.float32).reshape(10, 5)

    reader = _DummyReader(section)

    monkeypatch.setattr(
        pipeline_mod, 'get_reader', lambda *a, **k: reader, raising=True
    )
    monkeypatch.setattr(pipeline_mod, 'pipeline_key', lambda _spec: 'pk', raising=True)
    monkeypatch.setattr(
        pipeline_mod,
        '_maybe_attach_fbpick_offsets',
        lambda meta, **k: meta,
        raising=True,
    )
    monkeypatch.setattr(
        pipeline_mod,
        'coerce_section_f32',
        lambda arr, _scale: np.asarray(arr, dtype=np.float32),
        raising=True,
    )

    calls = {'n': 0, 'taps_args': []}

    def _fake_apply_pipeline(section_arr, *, spec, meta, taps=None):
        calls['n'] += 1
        calls['taps_args'].append(None if taps is None else list(taps))
        labels = ['final'] if taps is None else list(taps)
        s = int(np.sum(section_arr))
        return {lab: {'sum': s, 'call': calls['n'], 'label': lab} for lab in labels}

    monkeypatch.setattr(
        pipeline_mod, 'apply_pipeline', _fake_apply_pipeline, raising=True
    )

    client = TestClient(app)
    return client, state, calls


def _post_pipeline_section(
    client: TestClient,
    *,
    file_id: str,
    key1_val: int,
    offset_byte: int,
    spec_steps: list[dict] | None = None,
    taps: list[str] | None = None,
    window: dict[str, int | float] | None = None,
    list_only: bool = False,
) -> dict:
    spec_steps = spec_steps or []
    params = {
        'file_id': file_id,
        'key1_val': key1_val,
        'offset_byte': offset_byte,
        'list_only': str(list_only).lower(),
    }
    body: dict = {'spec': {'steps': spec_steps}}
    if taps is not None:
        body['taps'] = taps
    if window is not None:
        body['window'] = window

    r = client.post('/pipeline/section', params=params, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_pipeline_section_cache_hit_same_key_and_tap(_pipeline_env):
    client, state, calls = _pipeline_env

    out1 = _post_pipeline_section(
        client,
        file_id='fid',
        key1_val=1,
        offset_byte=7,
        taps=['tapA'],
    )
    assert calls['n'] == 1
    assert out1['pipeline_key'] == 'pk'
    assert out1['taps']['tapA']['call'] == 1

    out2 = _post_pipeline_section(
        client,
        file_id='fid',
        key1_val=1,
        offset_byte=7,
        taps=['tapA'],
    )
    assert calls['n'] == 1
    assert out2 == out1

    cache_key = ('fid', 1, 189, 'pk', None, 7, 'tapA')
    assert state.pipeline_tap_cache.get(cache_key) == out1['taps']['tapA']


def test_pipeline_section_cache_separates_by_window_hash(_pipeline_env):
    client, state, calls = _pipeline_env

    w1 = {'tr_min': 0, 'tr_max': 5, 't_min': 0, 't_max': 3}
    w2 = {'tr_min': 0, 'tr_max': 6, 't_min': 0, 't_max': 3}

    out1 = _post_pipeline_section(
        client,
        file_id='fid',
        key1_val=1,
        offset_byte=7,
        taps=['tapA'],
        window=w1,
    )
    assert calls['n'] == 1

    out1b = _post_pipeline_section(
        client,
        file_id='fid',
        key1_val=1,
        offset_byte=7,
        taps=['tapA'],
        window=w1,
    )
    assert calls['n'] == 1
    assert out1b == out1

    out2 = _post_pipeline_section(
        client,
        file_id='fid',
        key1_val=1,
        offset_byte=7,
        taps=['tapA'],
        window=w2,
    )
    assert calls['n'] == 2
    assert out2 != out1

    # window_hash が違えば別キーで cache される
    section = np.arange(50, dtype=np.float32).reshape(10, 5)
    _, h1 = _clean_window_and_hash(w1, section)
    _, h2 = _clean_window_and_hash(w2, section)
    assert h1 != h2

    k1 = ('fid', 1, 189, 'pk', h1, 7, 'tapA')
    k2 = ('fid', 1, 189, 'pk', h2, 7, 'tapA')
    assert state.pipeline_tap_cache.get(k1) == out1['taps']['tapA']
    assert state.pipeline_tap_cache.get(k2) == out2['taps']['tapA']


def test_pipeline_section_list_only_hits_cache_and_does_not_compute(_pipeline_env):
    client, state, calls = _pipeline_env

    # list_only の cache キーは window_hash=None 固定
    base = ('fid', 1, 189, 'pk', None, 7)
    state.pipeline_tap_cache.set((*base, 'tapA'), {'cached': True})
    state.pipeline_tap_cache.set((*base, 'tapB'), {'cached': True})

    out = _post_pipeline_section(
        client,
        file_id='fid',
        key1_val=1,
        offset_byte=7,
        taps=['tapA', 'tapB'],
        list_only=True,
    )
    assert calls['n'] == 0
    assert out['pipeline_key'] == 'pk'
    assert out['taps'] == {'tapA': True, 'tapB': True}


def test_pipeline_section_list_only_computes_only_misses(_pipeline_env):
    client, state, calls = _pipeline_env

    base = ('fid', 1, 189, 'pk', None, 7)
    state.pipeline_tap_cache.set((*base, 'tapA'), {'cached': True})

    out = _post_pipeline_section(
        client,
        file_id='fid',
        key1_val=1,
        offset_byte=7,
        taps=['tapA', 'tapB'],
        list_only=True,
    )

    # tapA は hit、tapB だけ miss → apply_pipeline は1回 & taps=['tapB'] のみ
    assert calls['n'] == 1
    assert calls['taps_args'] == [['tapB']]
    assert out['taps'] == {'tapA': True, 'tapB': True}

    cached_b = state.pipeline_tap_cache.get((*base, 'tapB'))
    assert isinstance(cached_b, dict)
    assert cached_b.get('label') == 'tapB'
