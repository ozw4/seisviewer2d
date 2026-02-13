# app/tests/test_section_window_bin_http_contract.py

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Some tests monkeypatch segyio usage; ensure import succeeds even if segyio
# isn't installed in the test environment.
sys.modules.setdefault('segyio', types.ModuleType('segyio'))

from app.api.routers import section as sec
from app.main import app


def _write_baseline(store_dir: Path, *, key1_val: int, n_traces: int) -> None:
    """Write a minimal baseline_raw.json compatible with load_baseline()."""
    baseline = {
        'key1_values': [int(key1_val)],
        'mu_section_by_key1': [0.0],
        'sigma_section_by_key1': [1.0],
        'mu_traces': [0.0] * int(n_traces),
        'sigma_traces': [1.0] * int(n_traces),
        'trace_spans_by_key1': {str(int(key1_val)): [[0, int(n_traces)]]},
    }
    (store_dir / 'baseline_raw.json').write_text(json.dumps(baseline), encoding='utf-8')


def _decode_payload(resp) -> dict:
    # TestClient は Content-Encoding: gzip を自動展開するため、resp.content は「解凍済み(msgpack生)」
    assert resp.headers.get('content-encoding') == 'gzip'
    return msgpack.unpackb(resp.content, raw=False)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Make quantization deterministic.
    monkeypatch.setenv('FIXED_INT8_SCALE', '1')

    sec.FILE_REGISTRY.clear()
    state = sec.get_state(app)
    state.window_section_cache.clear()
    state.trace_stats_cache.clear()
    state.cached_readers.clear()

    # dt is resolved via sec.get_dt_for_file (imported into the router),
    # so patching sec.get_dt_for_file stabilizes the payload contract.
    monkeypatch.setattr(sec, 'get_dt_for_file', lambda _fid: 0.0125, raising=True)

    yield

    sec.FILE_REGISTRY.clear()
    state.window_section_cache.clear()
    state.trace_stats_cache.clear()
    state.cached_readers.clear()


def test_get_section_window_bin_payload_includes_dt_and_matches_resolver(
    monkeypatch, tmp_path: Path
):
    key1_val = 7

    class _StubReader:
        key1_byte = 189
        key2_byte = 193

        def get_section(self, _key1_val: int):
            arr = np.arange(5 * 6, dtype=np.float32).reshape(5, 6)
            return SimpleNamespace(arr=arr, dtype=arr.dtype, scale=None)

    monkeypatch.setattr(
        sec,
        'get_reader',
        lambda fid, kb1, kb2, state=None: _StubReader(),
        raising=True,
    )

    sec.FILE_REGISTRY['f'] = {'store_path': str(tmp_path)}
    _write_baseline(tmp_path, key1_val=key1_val, n_traces=5)

    with TestClient(app) as client:
        resp = client.get(
            '/get_section_window_bin',
            params={
                'file_id': 'f',
                'key1_val': key1_val,
                'x0': 0,
                'x1': 4,
                'y0': 0,
                'y1': 5,
                'step_x': 1,
                'step_y': 1,
                'transpose': False,
                'scaling': 'amax',
            },
        )

    assert resp.status_code == 200
    payload = _decode_payload(resp)
    assert 'dt' in payload
    assert float(payload['dt']) == pytest.approx(0.0125)
    assert set(payload.keys()) >= {'shape', 'scale', 'data'}


@pytest.mark.parametrize(
    'params, expected_detail',
    [
        (
            {'x0': 0, 'x1': 99, 'y0': 0, 'y1': 5, 'step_x': 1, 'step_y': 1},
            'Trace range out of bounds',
        ),
        (
            {'x0': 0, 'x1': 4, 'y0': 0, 'y1': 999, 'step_x': 1, 'step_y': 1},
            'Sample range out of bounds',
        ),
    ],
)
def test_get_section_window_bin_out_of_bounds_returns_400(
    monkeypatch, tmp_path: Path, params: dict, expected_detail: str
):
    key1_val = 7

    class _StubReader:
        key1_byte = 189
        key2_byte = 193

        def get_section(self, _key1_val: int):
            arr = np.arange(5 * 6, dtype=np.float32).reshape(5, 6)
            return SimpleNamespace(arr=arr, dtype=arr.dtype, scale=None)

    monkeypatch.setattr(
        sec,
        'get_reader',
        lambda fid, kb1, kb2, state=None: _StubReader(),
        raising=True,
    )

    sec.FILE_REGISTRY['f'] = {'store_path': str(tmp_path)}
    _write_baseline(tmp_path, key1_val=key1_val, n_traces=5)

    with TestClient(app) as client:
        resp = client.get(
            '/get_section_window_bin',
            params={
                'file_id': 'f',
                'key1_val': key1_val,
                'transpose': False,
                'scaling': 'amax',
                **params,
            },
        )

    assert resp.status_code == 400
    assert resp.json().get('detail') == expected_detail


def test_get_section_window_bin_step_less_than_one_is_422(tmp_path: Path):
    # step_x/step_y have Query(ge=1), so FastAPI validation rejects them before service.
    sec.FILE_REGISTRY['f'] = {'store_path': str(tmp_path)}
    _write_baseline(tmp_path, key1_val=7, n_traces=1)

    with TestClient(app) as client:
        resp = client.get(
            '/get_section_window_bin',
            params={
                'file_id': 'f',
                'key1_val': 7,
                'x0': 0,
                'x1': 0,
                'y0': 0,
                'y1': 0,
                'step_x': 0,
                'step_y': 1,
                'transpose': False,
                'scaling': 'amax',
            },
        )

    assert resp.status_code == 422


def test_get_section_window_bin_value_error_maps_to_400(monkeypatch, tmp_path: Path):
    # Ensure any ValueError from the service layer becomes a stable 400 response.
    def _raise_value_error(**_k):  # noqa: ANN001
        raise ValueError('Requested window is empty')

    monkeypatch.setattr(
        sec, 'build_section_window_payload', _raise_value_error, raising=True
    )
    sec.FILE_REGISTRY['f'] = {'store_path': str(tmp_path)}

    with TestClient(app) as client:
        resp = client.get(
            '/get_section_window_bin',
            params={
                'file_id': 'f',
                'key1_val': 7,
                'x0': 0,
                'x1': 0,
                'y0': 0,
                'y1': 0,
                'step_x': 1,
                'step_y': 1,
                'transpose': False,
                'scaling': 'amax',
            },
        )

    assert resp.status_code == 400
    assert resp.json().get('detail') == 'Requested window is empty'
