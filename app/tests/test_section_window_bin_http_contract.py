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

from app.api.routers import section as sec  # noqa: E402
from app.main import app  # noqa: E402


def _write_baseline(store_dir: Path, *, key1: int, n_traces: int) -> None:
    """Write a minimal baseline_raw.json compatible with load_baseline()."""
    baseline = {
        'key1_values': [int(key1)],
        'mu_section_by_key1': [0.0],
        'sigma_section_by_key1': [1.0],
        'mu_traces': [0.0] * int(n_traces),
        'sigma_traces': [1.0] * int(n_traces),
        'trace_spans_by_key1': {str(int(key1)): [[0, int(n_traces)]]},
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

    app.state.sv.file_registry.clear()
    state = sec.get_state(app)
    state.window_section_cache.clear()
    state.trace_stats_cache.clear()
    state.cached_readers.clear()

    # dt is resolved via app-scoped file registry.
    monkeypatch.setattr(
        app.state.sv.file_registry,
        'get_dt',
        lambda _fid: 0.0125,
        raising=True,
    )

    yield

    app.state.sv.file_registry.clear()
    state.window_section_cache.clear()
    state.trace_stats_cache.clear()
    state.cached_readers.clear()


def test_get_section_window_bin_payload_includes_dt_and_matches_resolver(
    monkeypatch, tmp_path: Path
):
    key1 = 7

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

    app.state.sv.file_registry.set_record('f', {'store_path': str(tmp_path)})
    _write_baseline(tmp_path, key1=key1, n_traces=5)

    with TestClient(app) as client:
        resp = client.get(
            '/get_section_window_bin',
            params={
                'file_id': 'f',
                'key1': key1,
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
    key1 = 7

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

    app.state.sv.file_registry.set_record('f', {'store_path': str(tmp_path)})
    _write_baseline(tmp_path, key1=key1, n_traces=5)

    with TestClient(app) as client:
        resp = client.get(
            '/get_section_window_bin',
            params={
                'file_id': 'f',
                'key1': key1,
                'transpose': False,
                'scaling': 'amax',
                **params,
            },
        )

    assert resp.status_code == 400
    assert resp.json().get('detail') == expected_detail


def test_get_section_window_bin_step_less_than_one_is_422(tmp_path: Path):
    # step_x/step_y have Query(ge=1), so FastAPI validation rejects them before service.
    app.state.sv.file_registry.set_record('f', {'store_path': str(tmp_path)})
    _write_baseline(tmp_path, key1=7, n_traces=1)

    with TestClient(app) as client:
        resp = client.get(
            '/get_section_window_bin',
            params={
                'file_id': 'f',
                'key1': 7,
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
    app.state.sv.file_registry.set_record('f', {'store_path': str(tmp_path)})

    with TestClient(app) as client:
        resp = client.get(
            '/get_section_window_bin',
            params={
                'file_id': 'f',
                'key1': 7,
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
