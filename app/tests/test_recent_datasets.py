import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.routers import upload as upload_mod
from app.main import app


def _write_store(
    root: Path,
    store_name: str,
    *,
    original_name: str | None = None,
    key1_byte: int = 189,
    key2_byte: int = 193,
    dt: float | None = 0.004,
    n_traces: int = 3,
    n_samples: int = 5,
    original_size: int = 123,
    mtime: float | None = None,
) -> Path:
    store_dir = root / store_name
    store_dir.mkdir(parents=True, exist_ok=True)

    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    np.save(store_dir / 'traces.npy', traces)
    np.savez(
        store_dir / 'index.npz',
        key1_values=np.asarray([1], dtype=np.int32),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([n_traces], dtype=np.int64),
        sorted_to_original=np.arange(n_traces, dtype=np.int64),
    )
    meta = {
        'schema_version': 1,
        'dtype': 'float32',
        'n_traces': n_traces,
        'n_samples': n_samples,
        'key_bytes': {'key1': key1_byte, 'key2': key2_byte},
        'dt': dt,
        'original_size': original_size,
        'original_segy_path': '/secret/input/' + (original_name or store_name),
    }
    (store_dir / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')

    if mtime is not None:
        targets = [
            store_dir,
            store_dir / 'meta.json',
            store_dir / 'traces.npy',
            store_dir / 'index.npz',
        ]
        for target in targets:
            os.utime(target, (mtime, mtime))

    return store_dir


@pytest.fixture()
def client(monkeypatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(upload_mod, 'TRACE_DIR', tmp_path, raising=True)
    with TestClient(app) as test_client:
        yield test_client


def test_recent_datasets_returns_only_valid_stores(client: TestClient, tmp_path: Path):
    _write_store(tmp_path, 'valid-line.sgy', n_traces=4, n_samples=8)

    broken_meta_dir = tmp_path / 'broken-meta.sgy'
    broken_meta_dir.mkdir()
    (broken_meta_dir / 'traces.npy').write_bytes(b'not-a-valid-npy')
    (broken_meta_dir / 'index.npz').write_bytes(b'not-a-valid-npz')
    (broken_meta_dir / 'meta.json').write_text('{broken', encoding='utf-8')

    missing_traces_dir = tmp_path / 'missing-traces.sgy'
    missing_traces_dir.mkdir()
    (missing_traces_dir / 'meta.json').write_text('{}', encoding='utf-8')
    (missing_traces_dir / 'index.npz').write_bytes(b'noop')

    missing_index_dir = tmp_path / 'missing-index.sgy'
    missing_index_dir.mkdir()
    (missing_index_dir / 'meta.json').write_text('{}', encoding='utf-8')
    (missing_index_dir / 'traces.npy').write_bytes(b'noop')

    (tmp_path / 'not-a-store.txt').write_text('skip', encoding='utf-8')

    response = client.get('/recent_datasets')

    assert response.status_code == 200
    datasets = response.json()['datasets']
    assert [item['name'] for item in datasets] == ['valid-line.sgy']
    assert datasets[0]['key1_byte'] == 189
    assert datasets[0]['key2_byte'] == 193
    assert datasets[0]['n_traces'] == 4
    assert datasets[0]['n_samples'] == 8


def test_recent_datasets_excludes_archived_old_store_dirs(client: TestClient, tmp_path: Path):
    _write_store(tmp_path, 'line-a.sgy', mtime=100)
    _write_store(tmp_path, 'line-a.sgy.old-deadbeef', original_name='line-a-archived.sgy', mtime=200)

    response = client.get('/recent_datasets')

    assert response.status_code == 200
    datasets = response.json()['datasets']
    assert [item['name'] for item in datasets] == ['line-a.sgy']


def test_recent_datasets_are_sorted_by_updated_at_desc(client: TestClient, tmp_path: Path):
    older_ts = 1_710_000_000
    newer_ts = 1_720_000_000
    _write_store(tmp_path, 'older-store.sgy', original_name='older.sgy', mtime=older_ts)
    _write_store(tmp_path, 'newer-store.sgy', original_name='newer.sgy', mtime=newer_ts)

    response = client.get('/recent_datasets')

    assert response.status_code == 200
    datasets = response.json()['datasets']
    assert [item['name'] for item in datasets] == ['newer.sgy', 'older.sgy']
    assert datasets[0]['updated_at'] == datetime.fromtimestamp(newer_ts, timezone.utc).isoformat()
    assert datasets[1]['updated_at'] == datetime.fromtimestamp(older_ts, timezone.utc).isoformat()


def test_recent_datasets_returns_only_basenames_not_paths(client: TestClient, tmp_path: Path):
    _write_store(
        tmp_path,
        'stored-name.sgy',
        original_name='nested/path/secret-line.segy',
    )

    response = client.get('/recent_datasets')

    assert response.status_code == 200
    datasets = response.json()['datasets']
    assert datasets[0]['name'] == 'secret-line.segy'
    assert '/' not in datasets[0]['name']
    assert '\\' not in datasets[0]['name']


def test_recent_datasets_ignores_malformed_meta_and_missing_fields(client: TestClient, tmp_path: Path):
    missing_meta_fields = tmp_path / 'missing-fields.sgy'
    missing_meta_fields.mkdir()
    np.save(missing_meta_fields / 'traces.npy', np.zeros((2, 2), dtype=np.float32))
    np.savez(
        missing_meta_fields / 'index.npz',
        key1_values=np.asarray([1], dtype=np.int32),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([2], dtype=np.int64),
        sorted_to_original=np.asarray([0, 1], dtype=np.int64),
    )
    (missing_meta_fields / 'meta.json').write_text(
        json.dumps({'key_bytes': {'key1': 189, 'key2': 193}}),
        encoding='utf-8',
    )

    invalid_meta_dir = tmp_path / 'invalid-meta.sgy'
    invalid_meta_dir.mkdir()
    np.save(invalid_meta_dir / 'traces.npy', np.zeros((2, 2), dtype=np.float32))
    np.savez(
        invalid_meta_dir / 'index.npz',
        key1_values=np.asarray([1], dtype=np.int32),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([2], dtype=np.int64),
        sorted_to_original=np.asarray([0, 1], dtype=np.int64),
    )
    (invalid_meta_dir / 'meta.json').write_text('[]', encoding='utf-8')

    response = client.get('/recent_datasets')

    assert response.status_code == 200
    assert response.json() == {'datasets': []}
