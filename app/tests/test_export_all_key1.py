import io

import numpy as np
from fastapi.testclient import TestClient

from app.api.routers import picks as ep
from app.main import app
from app.utils.pick_cache_file1d_mem import load_all, open_for_write


class FakeReader:
    def __init__(self, sorted_to_original, n_samples=8):
        self._sorted_to_original = np.asarray(sorted_to_original, dtype=np.int64)
        self._n_samples = int(n_samples)

    def get_n_samples(self):
        return self._n_samples

    def get_sorted_to_original(self):
        return self._sorted_to_original


def _client_with_base_patches(
    monkeypatch, tmp_path, sorted_to_original, dt=0.004, n_samples=1000
):
    file_name = 'lineA.sgy'
    n_traces = int(len(sorted_to_original))
    monkeypatch.setattr(ep, '_filename_for_file_id', lambda file_id: file_name)
    monkeypatch.setattr(ep, 'get_dt_for_file', lambda file_id: dt)
    monkeypatch.setattr(
        ep,
        'get_ntraces_for',
        lambda file_id, key1_byte, key2_byte=193, state=None: n_traces,
    )
    monkeypatch.setattr(
        ep,
        'get_reader',
        lambda file_id, key1_byte, key2_byte, state=None: FakeReader(
            sorted_to_original=sorted_to_original,
            n_samples=n_samples,
        ),
    )
    monkeypatch.setenv('PICKS_NPY_DIR', str(tmp_path))
    return TestClient(app, raise_server_exceptions=False), file_name, n_traces


def test_export_manual_picks_npz_route_exists():
    assert any(
        getattr(r, 'path', '') == '/export_manual_picks_npz' for r in app.router.routes
    )


def test_export_manual_picks_npz_uses_original_order(monkeypatch, tmp_path):
    sorted_to_original = np.array([2, 0, 1], dtype=np.int64)
    client, file_name, n_traces = _client_with_base_patches(
        monkeypatch,
        tmp_path,
        sorted_to_original,
    )
    mm = open_for_write(file_name, n_traces)
    mm[:] = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    mm.flush()
    del mm

    r = client.get('/export_manual_picks_npz', params={'file_id': 'X'})
    assert r.status_code == 200
    with np.load(io.BytesIO(r.content)) as z:
        assert z['picks_time_s'].tolist() == [20.0, 30.0, 10.0]
        assert int(np.asarray(z['n_traces']).item()) == 3
        assert int(np.asarray(z['n_samples']).item()) == 1000
        assert float(np.asarray(z['dt']).item()) == 0.004
        assert int(np.asarray(z['format_version']).item()) == 1


def test_import_manual_picks_npz_replace_writes_sorted_order(monkeypatch, tmp_path):
    sorted_to_original = np.array([2, 0, 1], dtype=np.int64)
    client, file_name, n_traces = _client_with_base_patches(
        monkeypatch,
        tmp_path,
        sorted_to_original,
    )
    buf = io.BytesIO()
    np.savez(
        buf,
        picks_time_s=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        n_traces=np.int64(3),
        n_samples=np.int64(1000),
        dt=np.float64(0.004),
    )
    buf.seek(0)
    r = client.post(
        '/import_manual_picks_npz',
        params={'file_id': 'X', 'mode': 'replace'},
        files={'file': ('picks.npz', buf.getvalue(), 'application/octet-stream')},
    )
    assert r.status_code == 200
    assert r.json()['mode'] == 'replace'
    stored = load_all(file_name, n_traces)
    assert stored.tolist() == [3.0, 1.0, 2.0]


def test_import_manual_picks_npz_merge_overwrites_only_non_nan(monkeypatch, tmp_path):
    sorted_to_original = np.array([2, 0, 1], dtype=np.int64)
    client, file_name, n_traces = _client_with_base_patches(
        monkeypatch,
        tmp_path,
        sorted_to_original,
    )
    mm = open_for_write(file_name, n_traces)
    mm[:] = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    mm.flush()
    del mm

    buf = io.BytesIO()
    np.savez(
        buf,
        picks_time_s=np.array([np.nan, 2.0, 3.0], dtype=np.float32),
        n_traces=np.int64(3),
        n_samples=np.int64(1000),
        dt=np.float64(0.004),
    )
    buf.seek(0)
    r = client.post(
        '/import_manual_picks_npz',
        params={'file_id': 'X', 'mode': 'merge'},
        files={'file': ('picks.npz', buf.getvalue(), 'application/octet-stream')},
    )
    assert r.status_code == 200
    stored = load_all(file_name, n_traces)
    assert stored.tolist() == [3.0, 11.0, 2.0]


def test_import_manual_picks_npz_mismatch_returns_409(monkeypatch, tmp_path):
    sorted_to_original = np.array([0, 1, 2], dtype=np.int64)
    client, _, _ = _client_with_base_patches(monkeypatch, tmp_path, sorted_to_original)

    buf = io.BytesIO()
    np.savez(
        buf,
        picks_time_s=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        n_traces=np.int64(3),
        n_samples=np.int64(999),
        dt=np.float64(0.004),
    )
    buf.seek(0)
    r = client.post(
        '/import_manual_picks_npz',
        params={'file_id': 'X'},
        files={'file': ('picks.npz', buf.getvalue(), 'application/octet-stream')},
    )
    assert r.status_code == 409

    buf = io.BytesIO()
    np.savez(
        buf,
        picks_time_s=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        n_traces=np.int64(99),
        n_samples=np.int64(1000),
        dt=np.float64(0.004),
    )
    buf.seek(0)
    r = client.post(
        '/import_manual_picks_npz',
        params={'file_id': 'X'},
        files={'file': ('picks.npz', buf.getvalue(), 'application/octet-stream')},
    )
    assert r.status_code == 409

    buf = io.BytesIO()
    np.savez(
        buf,
        picks_time_s=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        n_traces=np.int64(3),
        n_samples=np.int64(1000),
        dt=np.float64(0.123),
    )
    buf.seek(0)
    r = client.post(
        '/import_manual_picks_npz',
        params={'file_id': 'X'},
        files={'file': ('picks.npz', buf.getvalue(), 'application/octet-stream')},
    )
    assert r.status_code == 409


def test_export_manual_picks_npz_sorted_to_original_error_is_409(monkeypatch, tmp_path):
    client, _, _ = _client_with_base_patches(
        monkeypatch, tmp_path, np.array([0], dtype=np.int64)
    )

    class BadReader:
        def get_n_samples(self):
            return 1000

        def get_sorted_to_original(self):
            raise ValueError('sorted_to_original is missing')

    monkeypatch.setattr(
        ep,
        'get_reader',
        lambda file_id, key1_byte, key2_byte, state=None: BadReader(),
    )
    r = client.get('/export_manual_picks_npz', params={'file_id': 'X'})
    assert r.status_code == 409


def test_import_manual_picks_npz_invalid_n_samples_is_409(monkeypatch, tmp_path):
    client, _, _ = _client_with_base_patches(
        monkeypatch, tmp_path, np.array([0], dtype=np.int64)
    )

    class BadReader:
        def get_n_samples(self):
            return None

        def get_sorted_to_original(self):
            return np.array([0], dtype=np.int64)

    monkeypatch.setattr(
        ep,
        'get_reader',
        lambda file_id, key1_byte, key2_byte, state=None: BadReader(),
    )
    buf = io.BytesIO()
    np.savez(
        buf,
        picks_time_s=np.array([1.0], dtype=np.float32),
        n_traces=np.int64(1),
        n_samples=np.int64(1),
        dt=np.float64(0.004),
    )
    buf.seek(0)
    r = client.post(
        '/import_manual_picks_npz',
        params={'file_id': 'X'},
        files={'file': ('picks.npz', buf.getvalue(), 'application/octet-stream')},
    )
    assert r.status_code == 409
