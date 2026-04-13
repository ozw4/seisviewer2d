# app/tests/test_section_router_changes.py
import gzip
import json
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

from app.api.routers import section as sec
from app.main import app
from app.services import section_index as secidx
from app.services.errors import ConflictError
from app.trace_store.types import SectionView


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    # Ensure a clean file registry and harmless dt function for binary endpoints.
    app.state.sv.file_registry.clear()
    monkeypatch.setattr(
        app.state.sv.file_registry,
        'get_dt',
        lambda _fid: 0.004,
        raising=True,
    )
    yield
    app.state.sv.file_registry.clear()


# -------------------------
# Test helpers (fake readers)
# -------------------------


def _make_stub_reader(key1s: np.ndarray, key2s: np.ndarray):
    """Minimal reader stub implementing the unified public API used by section router:
    - get_key1_values()
    - get_trace_seq_for_value(...), with legacy alias get_trace_seq_for_section(...)
    - get_section(key1)  (returns synthetic data)
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

    def get_trace_seq_for_value(key1: int, align_to: str = 'display'):
        idx = np.flatnonzero(r.key1s == key1).astype(np.int64)
        if idx.size == 0:
            raise ValueError(f'Key1 value {key1} not found')
        if align_to == 'original':
            return idx
        order = np.argsort(r.key2s[idx], kind='stable')
        return idx[order]

    # Legacy alias kept for older call sites inside router (if any remain)
    def get_trace_seq_for_section(key1: int, align_to: str = 'display'):
        return get_trace_seq_for_value(key1, align_to=align_to)

    def get_key1_values():
        return r.unique_key1

    def get_section(key1: int):
        idx = np.where(r.key1s == key1)[0]
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
    - get_key1_values(), get_section(key1) returning SectionView,
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

    def get_section(key1: int):
        idx = np.where(_k1 == key1)[0]
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


def test_get_ntraces_for_uses_reader_and_fails_fast(monkeypatch):
    r = _make_stub_reader(
        key1s=np.array([1, 1, 2, 3, 3], dtype=np.int32),
        key2s=np.array([5, 2, 9, 1, 1], dtype=np.int32),
    )
    monkeypatch.setattr(secidx, 'get_reader', lambda file_id, kb1, kb2, state=None: r)
    assert secidx.get_ntraces_for('f1', 189, 193, state=sec.get_state(app)) == 5

    def boom(*a, **k):
        raise ConflictError('trace store not built')

    monkeypatch.setattr(secidx, 'get_reader', boom)
    with pytest.raises(ConflictError, match='trace store not built'):
        secidx.get_ntraces_for('f2', 189, 193, state=sec.get_state(app))

    r2 = _make_stub_reader(
        key1s=np.array([7, 8, 9], dtype=np.int32),
        key2s=np.array([0, 0, 0], dtype=np.int32),
    )
    r2.ntraces = None  # type: ignore[attr-defined]

    class _Traces:
        shape = (3, 100)

    r2.traces = _Traces()  # type: ignore[attr-defined]
    monkeypatch.setattr(secidx, 'get_reader', lambda file_id, kb1, kb2, state=None: r2)
    assert secidx.get_ntraces_for('f3', 189, 193, state=sec.get_state(app)) == 3


def test_get_trace_seq_for_with_stub_reader_matches_legacy(monkeypatch):
    key1s = np.array([1, 1, 2, 1, 2, 3, 3, 1, 2, 3], dtype=np.int32)
    key2s = np.array([5, 2, 9, 2, 1, 1, 1, 2, 5, 1], dtype=np.int32)
    r = _make_stub_reader(key1s, key2s)
    monkeypatch.setattr(secidx, 'get_reader', lambda fid, kb1, kb2, state=None: r)

    vals = r.get_key1_values()
    for v in vals:
        seq = secidx.get_trace_seq_for_value(
            'lineA',
            key1=int(v),
            key1_byte=189,
            key2_byte=193,
            state=sec.get_state(app),
        )
        indices = np.where(r.key1s == v)[0]
        expected = indices[np.argsort(r.key2s[indices], kind='stable')]
        assert np.array_equal(seq, expected)


def test_get_trace_seq_for_requires_reader_method(monkeypatch):
    key1s = np.array([9, 9, 9, 8, 8, 7], dtype=np.int32)
    key2s = np.array([3, 1, 2, 5, 4, 0], dtype=np.int32)
    t = _make_tracestore_like_reader(key1s, key2s)
    monkeypatch.setattr(secidx, 'get_reader', lambda fid, kb1, kb2, state=None: t)
    with pytest.raises(ConflictError, match='trace sequence'):
        secidx.get_trace_seq_for_value(
            'lineB',
            key1=9,
            key1_byte=189,
            key2_byte=193,
            state=sec.get_state(app),
        )


def test_get_section_returns_json_for_value(monkeypatch):
    # Reader returns specific values and captures the key1 it received.
    received = {'val': None}

    class _StubReader:
        key1_byte = 189
        key2_byte = 193

        def get_key1_values(self):
            return np.array([10, 20, 30], dtype=np.int32)

        def get_section(self, key1: int):
            received['val'] = key1
            arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            return SectionView(arr=arr, dtype=arr.dtype, scale=None)

    monkeypatch.setattr(
        sec,
        'get_reader',
        lambda fid, kb1, kb2, state=None: _StubReader(),
        raising=True,
    )
    resp = sec.get_section(
        file_id='f',
        key1_byte=189,
        key2_byte=193,
        key1=20,
        request=SimpleNamespace(app=app),
    )
    data = json.loads(resp.body)
    assert data['section'] == [[1.0, 2.0], [3.0, 4.0]]
    assert received['val'] == 20

    # missing value → HTTPException(400)
    class _StubReader2:
        key1_byte = 189
        key2_byte = 193

        def get_section(self, key1: int):
            raise ValueError(f'key1 {key1} not found')

    monkeypatch.setattr(
        sec,
        'get_reader',
        lambda fid, kb1, kb2, state=None: _StubReader2(),
        raising=True,
    )
    with pytest.raises(Exception) as ei:
        sec.get_section(
            file_id='f',
            key1_byte=189,
            key2_byte=193,
            key1=99,
            request=SimpleNamespace(app=app),
        )
    # (FastAPI HTTPException) don't overfit type; message check is enough
    assert 'key1 99 not found' in str(ei.value)


def test_get_section_window_bin_happy_path(monkeypatch, tmp_path):
    class _StubReader:
        key1_byte = 189
        key2_byte = 193

        def get_key1_values(self):
            return np.array([7], dtype=np.int32)

        def get_section(self, key1: int):
            # 3 traces x 5 samples
            arr = np.array(
                [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [1, 2, 3, 4, 5]], dtype=np.float32
            )
            return SectionView(arr=arr, dtype=arr.dtype, scale=None)

    monkeypatch.setattr(
        sec,
        'get_reader',
        lambda fid, kb1, kb2, state=None: _StubReader(),
        raising=True,
    )

    # TraceStore の場所と baseline を用意（apply_scaling_from_baseline が参照）
    app.state.sv.file_registry.set_record('f', {'store_path': str(tmp_path)})
    baseline = {
        'key1_values': [7],
        'mu_section_by_key1': [0.0],
        'sigma_section_by_key1': [1.0],
        'mu_traces': [0.0, 0.0, 0.0],
        'sigma_traces': [1.0, 1.0, 1.0],
        'trace_spans_by_key1': {'7': [[0, 3]]},
    }
    (tmp_path / 'baseline_raw.json').write_text(json.dumps(baseline), encoding='utf-8')
    res = sec.get_section_window_bin(
        file_id='f',
        key1=7,
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
        request=SimpleNamespace(app=app),
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
