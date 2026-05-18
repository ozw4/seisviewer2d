from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.tests._stubs import write_baseline_raw


def _qc_payload() -> dict:
    return {
        'segy': {'n_traces': 200, 'n_samples': 1500, 'dt': 0.002},
        'recommended_pairs': [
            {
                'key1_byte': 189,
                'key1_name': 'INLINE_3D',
                'key2_byte': 193,
                'key2_name': 'CROSSLINE_3D',
                'score': 0.94,
                'confidence': 'high',
                'reasons': ['key1 has 10 sections with median 20 traces/section'],
                'warnings': [],
            }
        ],
        'headers': [
            {
                'byte': 189,
                'name': 'INLINE_3D',
                'available': True,
                'min': 1,
                'max': 10,
                'unique_count': 10,
                'unique_ratio': 0.05,
                'key1_score': 0.91,
                'group_size': {'min': 20, 'p05': 20, 'p50': 20, 'p95': 20, 'max': 20},
                'warnings': [],
            }
        ],
        'warnings': [],
    }


@pytest.fixture()
def _staged_env(tmp_path: Path, monkeypatch):
    from app.api.routers import upload as upload_mod

    state = app.state.sv
    state.file_registry.clear()
    state.cached_readers.clear()
    state.staged_uploads.clear()

    upload_root = tmp_path / 'uploads'
    processed = upload_root / 'processed'
    trace_dir = processed / 'traces'
    trace_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(upload_mod, 'UPLOAD_DIR', upload_root, raising=True)
    monkeypatch.setattr(upload_mod, 'PROCESSED_DIR', processed, raising=True)
    monkeypatch.setattr(upload_mod, 'TRACE_DIR', trace_dir, raising=True)
    def _fake_register(
        *,
        state,
        file_id,
        store_dir,
        dt=None,
        update_registry=True,
        touch_meta=True,
        **_kwargs,
    ):
        if touch_meta:
            meta_path = Path(store_dir) / 'meta.json'
            if meta_path.exists():
                meta_path.touch()
        if update_registry:
            state.file_registry.update(
                file_id,
                store_path=str(store_dir),
                dt=dt,
            )

    monkeypatch.setattr(upload_mod, 'register_trace_store', _fake_register)

    calls = {'qc': 0, 'ingest': 0}

    def _fake_qc(path: str | Path) -> dict:
        calls['qc'] += 1
        assert Path(path).is_file()
        return _qc_payload()

    def _fake_from_segy(
        path: str,
        store_dir: str | Path,
        key1_byte: int,
        key2_byte: int,
        *,
        source_sha256: str | None = None,
        **_kwargs,
    ) -> dict:
        calls['ingest'] += 1
        store_path = Path(store_dir)
        store_path.mkdir(parents=True, exist_ok=True)
        np.save(store_path / 'traces.npy', np.zeros((2, 4), dtype=np.float32))
        np.savez(
            store_path / 'index.npz',
            key1_values=np.asarray([1], dtype=np.int32),
            key1_offsets=np.asarray([0], dtype=np.int64),
            key1_counts=np.asarray([2], dtype=np.int64),
        )
        meta = {
            'key_bytes': {'key1': int(key1_byte), 'key2': int(key2_byte)},
            'dt': 0.002,
            'original_segy_path': str(path),
            'original_size': Path(path).stat().st_size,
            'source_sha256': source_sha256,
        }
        (store_path / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
        return meta

    monkeypatch.setattr(upload_mod, 'inspect_segy_header_qc', _fake_qc, raising=True)
    monkeypatch.setattr(
        upload_mod.SegyIngestor, 'from_segy', _fake_from_segy, raising=True
    )

    try:
        yield TestClient(app), upload_mod, calls
    finally:
        state.file_registry.clear()
        state.cached_readers.clear()
        state.staged_uploads.clear()


def _stage(client: TestClient, *, name: str = 'line 001.sgy', data: bytes = b'segy') -> dict:
    response = client.post(
        '/stage_segy',
        files={'file': (name, data, 'application/octet-stream')},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_stage_segy_returns_qc_and_stores_metadata(_staged_env):
    client, _upload_mod, calls = _staged_env

    body = _stage(client, data=b'abc123')
    staged_id = body['staged_id']
    staged = app.state.sv.staged_uploads.get(staged_id)

    assert calls['qc'] == 1
    assert body['file']['original_name'] == 'line 001.sgy'
    assert body['file']['safe_name'] == 'line_001.sgy'
    assert body['file']['size'] == 6
    assert body['segy']['n_traces'] == 200
    assert body['headers']
    assert body['recommended_pairs']
    assert isinstance(staged, dict)
    assert staged['original_name'] == 'line 001.sgy'
    assert staged['safe_name'] == 'line_001.sgy'
    assert Path(staged['raw_path']).read_bytes() == b'abc123'
    assert staged['header_qc']['segy']['n_samples'] == 1500


def test_ingest_staged_segy_ingests_valid_stage(_staged_env):
    client, upload_mod, calls = _staged_env
    staged = _stage(client)
    staged_id = staged['staged_id']
    record = app.state.sv.staged_uploads.get(staged_id)
    assert isinstance(record, dict)
    staged_path = Path(record['raw_path'])
    staged_dir = staged_path.parent

    response = client.post(
        '/ingest_staged_segy',
        data={
            'staged_id': staged_id,
            'key1_byte': '189',
            'key2_byte': '193',
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert calls['ingest'] == 1
    assert body['file_id']
    assert body['reused_trace_store'] is False
    assert body['header_qc'] == {
        'selected_pair_score': 0.94,
        'confidence': 'high',
        'warnings': [],
    }
    meta = json.loads(
        (Path(upload_mod.TRACE_DIR) / 'line_001.sgy' / 'meta.json').read_text(
            encoding='utf-8'
        )
    )
    durable_raw_path = (
        Path(upload_mod.UPLOAD_DIR)
        / 'raw'
        / staged['file']['sha256']
        / 'line_001.sgy'
    )
    assert meta['key_bytes'] == {'key1': 189, 'key2': 193}
    assert meta['original_segy_path'] == str(durable_raw_path)
    assert durable_raw_path.read_bytes() == b'segy'
    assert not staged_path.exists()
    assert not staged_dir.exists()
    assert app.state.sv.staged_uploads.get(staged_id) is None


def test_stage_segy_qc_failure_removes_staged_file(_staged_env, monkeypatch):
    client, upload_mod, calls = _staged_env

    def _raise_qc(_path: str | Path) -> dict:
        calls['qc'] += 1
        raise RuntimeError('bad headers')

    monkeypatch.setattr(upload_mod, 'inspect_segy_header_qc', _raise_qc, raising=True)

    response = client.post(
        '/stage_segy',
        files={'file': ('bad.sgy', b'bad-data', 'application/octet-stream')},
    )

    assert response.status_code == 422
    assert calls == {'qc': 1, 'ingest': 0}
    assert len(app.state.sv.staged_uploads) == 0
    staged_root = Path(upload_mod.UPLOAD_DIR) / 'staged'
    assert not staged_root.exists() or list(staged_root.iterdir()) == []


def test_staged_upload_capacity_eviction_removes_staged_file(
    _staged_env,
    monkeypatch,
):
    client, _upload_mod, _calls = _staged_env
    cache = app.state.sv.staged_uploads
    monkeypatch.setattr(cache, 'capacity', 1)

    first = _stage(client, name='first.sgy', data=b'first')
    first_record = app.state.sv.staged_uploads.get(first['staged_id'])
    assert isinstance(first_record, dict)
    first_path = Path(first_record['raw_path'])
    first_dir = first_path.parent

    second = _stage(client, name='second.sgy', data=b'second')
    second_record = app.state.sv.staged_uploads.get(second['staged_id'])
    assert isinstance(second_record, dict)
    second_path = Path(second_record['raw_path'])

    assert app.state.sv.staged_uploads.get(first['staged_id']) is None
    assert not first_path.exists()
    assert not first_dir.exists()
    assert second_path.exists()


def test_staged_upload_ttl_expiry_removes_staged_file(_staged_env, monkeypatch):
    client, _upload_mod, _calls = _staged_env
    cache = app.state.sv.staged_uploads
    clock = {'now': 1000.0}
    monkeypatch.setattr(cache, 'ttl_sec', 1)
    monkeypatch.setattr(cache, '_time_fn', lambda: clock['now'])

    staged = _stage(client)
    record = app.state.sv.staged_uploads.get(staged['staged_id'])
    assert isinstance(record, dict)
    staged_path = Path(record['raw_path'])
    staged_dir = staged_path.parent

    clock['now'] = 1002.0
    assert app.state.sv.staged_uploads.get(staged['staged_id']) is None
    assert not staged_path.exists()
    assert not staged_dir.exists()


def test_stale_staged_upload_dir_cleanup_removes_orphan(_staged_env):
    _client, upload_mod, _calls = _staged_env
    state = app.state.sv
    now = 10_000.0
    staged_root = Path(upload_mod.UPLOAD_DIR) / 'staged'
    stale_dir = staged_root / 'orphaned-stage'
    stale_file = stale_dir / 'orphan.sgy'
    stale_dir.mkdir(parents=True)
    stale_file.write_bytes(b'orphan')
    stale_mtime = now - float(state.staged_uploads.ttl_sec) - 1.0
    os.utime(stale_dir, (stale_mtime, stale_mtime))
    os.utime(stale_file, (stale_mtime, stale_mtime))

    removed = upload_mod.cleanup_staged_uploads(state, force=True, now_ts=now)

    assert removed == 1
    assert not stale_dir.exists()


def test_staged_upload_cleanup_refuses_paths_outside_staged_root(
    _staged_env,
    tmp_path: Path,
):
    _client, upload_mod, _calls = _staged_env
    outside_dir = tmp_path / 'outside'
    outside_path = outside_dir / 'unsafe.sgy'
    outside_dir.mkdir()
    outside_path.write_bytes(b'unsafe')

    upload_mod._cleanup_staged_upload(outside_path)

    assert outside_path.exists()
    assert outside_dir.exists()


def test_ingest_staged_segy_reused_store_metadata_points_to_durable_raw(
    _staged_env,
):
    client, upload_mod, calls = _staged_env
    data = b'reuse-me'
    source_sha256 = hashlib.sha256(data).hexdigest()
    store_dir = Path(upload_mod.TRACE_DIR) / 'line_001.sgy'
    old_staged_path = Path(upload_mod.UPLOAD_DIR) / 'staged' / 'old' / 'line_001.sgy'
    store_dir.mkdir(parents=True, exist_ok=True)
    np.save(store_dir / 'traces.npy', np.zeros((2, 4), dtype=np.float32))
    np.savez(
        store_dir / 'index.npz',
        key1_values=np.asarray([1], dtype=np.int32),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([2], dtype=np.int64),
    )
    (store_dir / 'meta.json').write_text(
        json.dumps(
            {
                'key_bytes': {'key1': 189, 'key2': 193},
                'dt': 0.002,
                'original_segy_path': str(old_staged_path),
                'original_size': len(data),
                'source_sha256': source_sha256,
            }
        ),
        encoding='utf-8',
    )
    write_baseline_raw(
        store_dir,
        key1=1,
        n_traces=2,
        key1_byte=189,
        key2_byte=193,
        source_sha256=source_sha256,
    )
    staged = _stage(client, data=data)

    response = client.post(
        '/ingest_staged_segy',
        data={
            'staged_id': staged['staged_id'],
            'key1_byte': '189',
            'key2_byte': '193',
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()['reused_trace_store'] is True
    assert calls == {'qc': 1, 'ingest': 0}
    durable_raw_path = (
        Path(upload_mod.UPLOAD_DIR) / 'raw' / source_sha256 / 'line_001.sgy'
    )
    meta = json.loads((store_dir / 'meta.json').read_text(encoding='utf-8'))
    assert meta['original_segy_path'] == str(durable_raw_path)
    assert durable_raw_path.read_bytes() == data


def test_ingest_staged_segy_invalid_id_returns_404(_staged_env):
    client, _upload_mod, calls = _staged_env

    response = client.post(
        '/ingest_staged_segy',
        data={'staged_id': 'missing', 'key1_byte': '189', 'key2_byte': '193'},
    )

    assert response.status_code == 404
    assert calls['ingest'] == 0


def test_ingest_staged_segy_identical_key_bytes_return_400(_staged_env):
    client, _upload_mod, calls = _staged_env
    staged = _stage(client)

    response = client.post(
        '/ingest_staged_segy',
        data={
            'staged_id': staged['staged_id'],
            'key1_byte': '189',
            'key2_byte': '189',
        },
    )

    assert response.status_code == 400
    assert calls['ingest'] == 0


def test_ingest_staged_segy_missing_file_returns_410(_staged_env):
    client, _upload_mod, calls = _staged_env
    staged = _stage(client)
    record = app.state.sv.staged_uploads.get(staged['staged_id'])
    assert isinstance(record, dict)
    Path(record['raw_path']).unlink()

    response = client.post(
        '/ingest_staged_segy',
        data={
            'staged_id': staged['staged_id'],
            'key1_byte': '189',
            'key2_byte': '193',
        },
    )

    assert response.status_code == 410
    assert calls['ingest'] == 0


@pytest.mark.parametrize('endpoint', ['/upload_segy', '/stage_segy'])
@pytest.mark.parametrize('filename', ['.', '..'])
def test_segy_upload_rejects_dot_filenames(_staged_env, endpoint: str, filename: str):
    client, _upload_mod, calls = _staged_env
    kwargs = {
        'files': {'file': (filename, b'data', 'application/octet-stream')},
    }
    if endpoint == '/upload_segy':
        kwargs['data'] = {'key1_byte': '189', 'key2_byte': '193'}

    response = client.post(endpoint, **kwargs)

    assert response.status_code == 400
    assert response.json()['detail'] == 'Uploaded file must have a safe filename'
    assert calls == {'qc': 0, 'ingest': 0}


def test_upload_segy_still_returns_existing_basic_shape(_staged_env):
    client, _upload_mod, calls = _staged_env

    response = client.post(
        '/upload_segy',
        files={'file': ('direct.sgy', b'direct', 'application/octet-stream')},
        data={'key1_byte': '189', 'key2_byte': '193'},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {'file_id', 'reused_trace_store'}
    assert body['file_id']
    assert body['reused_trace_store'] is False
    assert calls == {'qc': 0, 'ingest': 1}
