from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import sys
import types
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

msgpack_stub = types.ModuleType('msgpack')
msgpack_stub.packb = lambda obj, **kwargs: b''
msgpack_stub.unpackb = lambda data, **kwargs: {}
sys.modules.setdefault('msgpack', msgpack_stub)

PICKS_PATH = Path(__file__).resolve().parents[1] / 'api' / 'routers' / 'picks.py'
PICKS_SPEC = spec_from_file_location('picks_router_for_test', PICKS_PATH)
PICKS_MOD = module_from_spec(PICKS_SPEC)
assert PICKS_SPEC is not None
assert PICKS_SPEC.loader is not None
PICKS_SPEC.loader.exec_module(PICKS_MOD)


class DummyReader:
    def __init__(self, key1_values, source_sha256='abc'):
        self._key1_values = np.asarray(key1_values, dtype=np.int64)
        self.meta = {'source_sha256': source_sha256}

    def get_key1_values(self):
        return self._key1_values


def build_npz_bytes(
    *,
    file_id='f1',
    key1_byte=189,
    key2_byte=193,
    dt=0.004,
    picks_idx=None,
    key1_values=None,
    source_sha256='abc',
):
    if picks_idx is None:
        picks_idx = np.array([[1, -1, 3], [-1, 2, -1]], dtype=np.int32)
    if key1_values is None:
        key1_values = np.array([10, 20], dtype=np.int64)
    buf = BytesIO()
    payload = {
        'picks_idx': np.asarray(picks_idx),
        'key1_values': np.asarray(key1_values),
        'dt': np.float64(dt),
        'key1_byte': np.int32(key1_byte),
        'key2_byte': np.int32(key2_byte),
        'file_id': np.asarray(file_id),
    }
    if source_sha256 is not None:
        payload['source_sha256'] = np.asarray(source_sha256)
    np.savez_compressed(buf, **payload)
    return buf.getvalue()


def make_client():
    app = FastAPI()
    app.include_router(PICKS_MOD.router)
    return TestClient(app)


def test_import_manual_picks_all_npz_replace_success(monkeypatch):
    client = make_client()
    calls = {'clear': 0, 'set': None}

    monkeypatch.setattr(PICKS_MOD, 'get_state', lambda app: None)
    monkeypatch.setattr(PICKS_MOD, '_filename_for_file_id', lambda file_id: 'dummy.sgy')
    monkeypatch.setattr(PICKS_MOD, 'get_dt_for_file', lambda file_id: 0.004)
    monkeypatch.setattr(PICKS_MOD, 'get_ntraces_for', lambda *args, **kwargs: 10)
    monkeypatch.setattr(
        PICKS_MOD,
        'get_reader',
        lambda *args, **kwargs: DummyReader([10, 20], source_sha256='abc'),
    )

    def fake_map(file_id, key1, key1_byte, key2_byte, state=None):
        if int(key1) == 10:
            return np.array([0, 2, 5], dtype=np.int64)
        if int(key1) == 20:
            return np.array([1, 3], dtype=np.int64)
        return np.array([], dtype=np.int64)

    monkeypatch.setattr(PICKS_MOD, 'get_trace_seq_for_value', fake_map)

    def fake_clear_all(file_name, ntraces):
        calls['clear'] += 1

    def fake_set_many(file_name, ntraces, trace_seq_arr, time_s_arr):
        calls['set'] = (
            np.asarray(trace_seq_arr, dtype=np.int64),
            np.asarray(time_s_arr, dtype=np.float32),
        )
        return int(np.asarray(trace_seq_arr).size)

    monkeypatch.setattr(PICKS_MOD, 'clear_all', fake_clear_all)
    monkeypatch.setattr(PICKS_MOD, 'set_many_by_traceseq', fake_set_many)

    data = build_npz_bytes()
    res = client.post(
        '/import_manual_picks_all_npz?file_id=f1&key1_byte=189&key2_byte=193&mode=replace',
        files={'file': ('manual.npz', data, 'application/octet-stream')},
    )
    assert res.status_code == 200
    body = res.json()
    assert body['status'] == 'ok'
    assert body['mode'] == 'replace'
    assert body['sections'] == 2
    assert body['inserted'] == 3
    assert body['skipped_negative'] == 2
    assert body['cleared'] == 'all'
    assert calls['clear'] == 1

    trace_seq_arr, time_s_arr = calls['set']
    np.testing.assert_array_equal(trace_seq_arr, np.array([0, 5, 3], dtype=np.int64))
    np.testing.assert_allclose(
        time_s_arr, np.array([0.004, 0.012, 0.008], dtype=np.float32)
    )


def test_import_manual_picks_all_npz_conflicts(monkeypatch):
    client = make_client()
    monkeypatch.setattr(PICKS_MOD, 'get_state', lambda app: None)
    monkeypatch.setattr(PICKS_MOD, '_filename_for_file_id', lambda file_id: 'dummy.sgy')
    monkeypatch.setattr(PICKS_MOD, 'get_dt_for_file', lambda file_id: 0.004)
    monkeypatch.setattr(PICKS_MOD, 'get_ntraces_for', lambda *args, **kwargs: 10)
    monkeypatch.setattr(
        PICKS_MOD,
        'get_reader',
        lambda *args, **kwargs: DummyReader([10, 20], source_sha256='abc'),
    )
    monkeypatch.setattr(
        PICKS_MOD,
        'get_trace_seq_for_value',
        lambda *args, **kwargs: np.array([0, 1, 2], dtype=np.int64),
    )

    base_url = '/import_manual_picks_all_npz?file_id=f1&key1_byte=189&key2_byte=193&mode=replace'

    res = client.post(
        base_url,
        files={
            'file': (
                'manual.npz',
                build_npz_bytes(file_id='other'),
                'application/octet-stream',
            )
        },
    )
    assert res.status_code == 409

    res = client.post(
        base_url,
        files={
            'file': (
                'manual.npz',
                build_npz_bytes(key1_byte=191),
                'application/octet-stream',
            )
        },
    )
    assert res.status_code == 409

    res = client.post(
        base_url,
        files={
            'file': (
                'manual.npz',
                build_npz_bytes(dt=0.002),
                'application/octet-stream',
            )
        },
    )
    assert res.status_code == 409

    res = client.post(
        base_url,
        files={
            'file': (
                'manual.npz',
                build_npz_bytes(
                    picks_idx=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
                    key1_values=np.array([10], dtype=np.int64),
                ),
                'application/octet-stream',
            )
        },
    )
    assert res.status_code == 409

    monkeypatch.setattr(
        PICKS_MOD,
        'get_trace_seq_for_value',
        lambda *args, **kwargs: np.array([0, 1, 2, 3], dtype=np.int64),
    )
    res = client.post(
        base_url,
        files={
            'file': (
                'manual.npz',
                build_npz_bytes(
                    picks_idx=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
                    key1_values=np.array([10, 20], dtype=np.int64),
                ),
                'application/octet-stream',
            )
        },
    )
    assert res.status_code == 409
