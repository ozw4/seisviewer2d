from __future__ import annotations

from typing import Any

import msgpack
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.routers import section as sec
from app.main import app
from app.services.fbpick_support import OFFSET_BYTE_FIXED


def _decode_payload(resp) -> tuple[dict[str, Any], np.ndarray]:
    assert resp.headers.get('content-encoding') == 'gzip'
    payload = msgpack.unpackb(resp.content, raw=False)
    offsets = np.frombuffer(payload['offsets'], dtype=np.float32)
    return payload, offsets


class _OffsetsReader:
    def __init__(
        self,
        *,
        n_traces: int = 3,
        offsets: object = (100.0, 200.0, 350.0),
        missing_key1: bool = False,
        offset_error: Exception | None = None,
    ) -> None:
        self.n_traces = int(n_traces)
        self.offsets = offsets
        self.missing_key1 = bool(missing_key1)
        self.offset_error = offset_error
        self.requested_offset_bytes: list[int] = []

    def get_trace_seq_for_value(self, key1: int, align_to: str = 'display'):
        if align_to != 'display':
            raise ValueError("align_to must be 'display'")
        if self.missing_key1:
            raise ValueError(f'Key1 value {int(key1)} not found')
        return np.arange(self.n_traces, dtype=np.int64)

    def get_offsets_for_section(self, _key1: int, offset_byte: int):
        self.requested_offset_bytes.append(int(offset_byte))
        if self.offset_error is not None:
            raise self.offset_error
        return self.offsets


class _ReaderWithoutOffsets:
    def get_trace_seq_for_value(self, _key1: int, align_to: str = 'display'):
        if align_to != 'display':
            raise ValueError("align_to must be 'display'")
        return np.arange(2, dtype=np.int64)


@pytest.fixture(autouse=True)
def _clean_state():
    app.state.sv.file_registry.clear()
    state = sec.get_state(app)
    state.cached_readers.clear()
    state.window_section_cache.clear()
    state.section_offsets_cache.clear()
    state.trace_stats_cache.clear()
    yield
    app.state.sv.file_registry.clear()
    state.cached_readers.clear()
    state.window_section_cache.clear()
    state.section_offsets_cache.clear()
    state.trace_stats_cache.clear()


def _install_reader(monkeypatch, reader):
    monkeypatch.setattr(
        sec,
        'get_reader',
        lambda _fid, _kb1, _kb2, state=None: reader,
        raising=True,
    )


def test_get_section_offsets_bin_returns_float32_payload(monkeypatch):
    reader = _OffsetsReader(offsets=np.array([10.5, 20.25, 30.0], dtype=np.float64))
    _install_reader(monkeypatch, reader)

    with TestClient(app) as client:
        resp = client.get(
            '/get_section_offsets_bin',
            params={
                'file_id': 'f',
                'key1': 7,
                'key1_byte': 189,
                'key2_byte': 193,
                'offset_byte': 41,
            },
        )

    assert resp.status_code == 200
    assert resp.headers['x-sv-cache'] == 'miss'
    payload, offsets = _decode_payload(resp)
    assert payload['file_id'] == 'f'
    assert payload['key1'] == 7
    assert payload['key1_byte'] == 189
    assert payload['key2_byte'] == 193
    assert payload['offset_byte'] == 41
    assert payload['dtype'] == 'float32'
    assert payload['shape'] == [3]
    assert offsets.dtype == np.float32
    np.testing.assert_allclose(offsets, np.array([10.5, 20.25, 30.0], dtype=np.float32))
    assert reader.requested_offset_bytes == [41]


def test_get_section_offsets_bin_defaults_offset_byte_to_37(monkeypatch):
    reader = _OffsetsReader()
    _install_reader(monkeypatch, reader)

    with TestClient(app) as client:
        resp = client.get(
            '/get_section_offsets_bin',
            params={
                'file_id': 'f',
                'key1': 7,
                'key1_byte': 189,
                'key2_byte': 193,
            },
        )

    assert resp.status_code == 200
    payload, offsets = _decode_payload(resp)
    assert payload['offset_byte'] == OFFSET_BYTE_FIXED == 37
    assert payload['shape'] == [3]
    np.testing.assert_allclose(offsets, np.array([100.0, 200.0, 350.0], dtype=np.float32))
    assert reader.requested_offset_bytes == [OFFSET_BYTE_FIXED]


def test_get_section_offsets_bin_uses_cache_key_metadata(monkeypatch):
    reader = _OffsetsReader(offsets=[1.0, 2.0, 3.0])
    _install_reader(monkeypatch, reader)
    state = sec.get_state(app)
    params = {
        'file_id': 'f',
        'key1': 7,
        'key1_byte': 189,
        'key2_byte': 193,
        'offset_byte': 37,
    }

    with TestClient(app) as client:
        first = client.get('/get_section_offsets_bin', params=params)
        reader.offsets = [9.0, 9.0, 9.0]
        second = client.get('/get_section_offsets_bin', params=params)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers['x-sv-cache'] == 'miss'
    assert second.headers['x-sv-cache'] == 'hit'
    _, first_offsets = _decode_payload(first)
    _, second_offsets = _decode_payload(second)
    np.testing.assert_allclose(first_offsets, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(second_offsets, first_offsets)
    assert reader.requested_offset_bytes == [37]
    assert len(state.section_offsets_cache) == 1
    assert len(state.window_section_cache) == 0


def test_get_section_offsets_bin_unknown_file_id_returns_404():
    with TestClient(app) as client:
        resp = client.get(
            '/get_section_offsets_bin',
            params={'file_id': 'missing', 'key1': 7},
        )

    assert resp.status_code == 404
    assert resp.json()['detail'] == 'File ID not found'


@pytest.mark.parametrize(
    'params',
    [
        {'key1_byte': 0},
        {'key2_byte': 0},
        {'offset_byte': 0},
        {'offset_byte': 241},
    ],
)
def test_get_section_offsets_bin_invalid_byte_query_is_422(monkeypatch, params):
    _install_reader(monkeypatch, _OffsetsReader())
    base_params = {
        'file_id': 'f',
        'key1': 7,
        'key1_byte': 189,
        'key2_byte': 193,
        'offset_byte': 37,
    }
    base_params.update(params)

    with TestClient(app) as client:
        resp = client.get('/get_section_offsets_bin', params=base_params)

    assert resp.status_code == 422


def test_get_section_offsets_bin_missing_key1_returns_400(monkeypatch):
    _install_reader(monkeypatch, _OffsetsReader(missing_key1=True))

    with TestClient(app) as client:
        resp = client.get(
            '/get_section_offsets_bin',
            params={'file_id': 'f', 'key1': 999},
        )

    assert resp.status_code == 400
    assert resp.json()['detail'] == 'Key1 value 999 not found'


def test_get_section_offsets_bin_missing_offset_reader_returns_400(monkeypatch):
    _install_reader(monkeypatch, _ReaderWithoutOffsets())

    with TestClient(app) as client:
        resp = client.get(
            '/get_section_offsets_bin',
            params={'file_id': 'f', 'key1': 7},
        )

    assert resp.status_code == 400
    assert resp.json()['detail'] == 'Offset header unavailable'


def test_get_section_offsets_bin_offset_read_failure_returns_400(monkeypatch):
    _install_reader(
        monkeypatch,
        _OffsetsReader(offset_error=RuntimeError('header load failed')),
    )

    with TestClient(app) as client:
        resp = client.get(
            '/get_section_offsets_bin',
            params={'file_id': 'f', 'key1': 7},
        )

    assert resp.status_code == 400
    assert resp.json()['detail'] == 'Failed to read offsets'


@pytest.mark.parametrize(
    ('offsets', 'detail'),
    [
        ([], 'Offsets must not be empty'),
        ([[1.0, 2.0, 3.0]], 'Offsets must be a 1D array'),
        ([1.0, np.nan, 3.0], 'Offsets contain NaN or Inf'),
        ([1.0, np.inf, 3.0], 'Offsets contain NaN or Inf'),
        ([1.0, 2.0], 'Offsets length does not match section trace count'),
    ],
)
def test_get_section_offsets_bin_invalid_offsets_return_400(
    monkeypatch,
    offsets,
    detail: str,
):
    _install_reader(monkeypatch, _OffsetsReader(offsets=offsets))

    with TestClient(app) as client:
        resp = client.get(
            '/get_section_offsets_bin',
            params={'file_id': 'f', 'key1': 7},
        )

    assert resp.status_code == 400
    assert resp.json()['detail'] == detail
