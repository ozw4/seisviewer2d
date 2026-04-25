from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


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
    monkeypatch.setattr(upload_mod, '_register_trace_store', lambda *a, **k: None)

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

    response = client.post(
        '/ingest_staged_segy',
        data={
            'staged_id': staged['staged_id'],
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
    assert meta['key_bytes'] == {'key1': 189, 'key2': 193}


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
