from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from app.main import app
from app.services import segy_upload_storage
from app.tests._stubs import write_baseline_raw
from app.trace_store.naming import (
    CONTENT_ADDRESSED_STORE_NAME_MAX_CHARS,
    DIRECT_IMPORT_RAW_NAME_MAX_CHARS,
    content_addressed_compare_store_name,
)


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
    from app.services import compare_raw_import_service
    from app.services import segy_ingest_service
    from app.services import segy_open_service

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

    monkeypatch.setattr(segy_ingest_service, 'register_trace_store', _fake_register)
    monkeypatch.setattr(segy_open_service, 'register_trace_store', _fake_register)

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
        write_baseline_raw(
            store_path,
            key1=1,
            n_traces=2,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            source_sha256=source_sha256,
        )
        return meta

    monkeypatch.setattr(upload_mod, 'inspect_segy_header_qc', _fake_qc, raising=True)
    monkeypatch.setattr(
        compare_raw_import_service,
        'inspect_segy_header_qc',
        _fake_qc,
        raising=True,
    )
    monkeypatch.setattr(
        segy_ingest_service.SegyIngestor,
        'from_segy',
        _fake_from_segy,
        raising=True,
    )
    monkeypatch.setattr(
        segy_open_service.SegyIngestor,
        'from_segy',
        _fake_from_segy,
        raising=True,
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


def _compare_raw_import(
    client: TestClient,
    *,
    name: str = 'line 001.sgy',
    data: bytes = b'segy',
    key1_byte: int = 189,
    key2_byte: int = 193,
):
    return client.post(
        '/compare/raw/import',
        files={'file': (name, data, 'application/octet-stream')},
        data={'key1_byte': str(key1_byte), 'key2_byte': str(key2_byte)},
    )


def _write_complete_compare_store(
    store_dir: Path,
    *,
    source_sha256: str,
    source_size: int,
    key1_byte: int = 189,
    key2_byte: int = 193,
    original_name: str | None = None,
    dt: float = 0.002,
) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    np.save(store_dir / 'traces.npy', np.zeros((2, 4), dtype=np.float32))
    np.savez(
        store_dir / 'index.npz',
        key1_values=np.asarray([1], dtype=np.int32),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([2], dtype=np.int64),
    )
    meta = {
        'key_bytes': {'key1': int(key1_byte), 'key2': int(key2_byte)},
        'dt': float(dt),
        'original_segy_path': str(store_dir / 'original.sgy'),
        'original_size': int(source_size),
        'source_sha256': source_sha256,
    }
    if original_name is not None:
        meta['original_name'] = original_name
    (store_dir / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    write_baseline_raw(
        store_dir,
        key1=1,
        n_traces=2,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        source_sha256=source_sha256,
    )


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, object] = {}

    def _target() -> None:
        try:
            result['value'] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            result['error'] = exc

    thread = threading.Thread(target=_target)
    thread.start()
    thread.join()
    if 'error' in result:
        raise result['error']
    return result.get('value')


def test_save_upload_file_returns_hash_and_size(tmp_path: Path):
    data = b'saved-upload'
    upload = UploadFile(file=io.BytesIO(data), filename='line.sgy')
    raw_path = tmp_path / 'raw' / 'line.sgy'

    saved = _run_async(
        segy_upload_storage.save_upload_file(
            upload,
            'line.sgy',
            raw_path=raw_path,
        )
    )

    assert saved.original_name == 'line.sgy'
    assert saved.safe_name == 'line.sgy'
    assert saved.raw_path == raw_path
    assert saved.source_sha256 == hashlib.sha256(data).hexdigest()
    assert saved.source_size == len(data)
    assert raw_path.read_bytes() == data


def test_promote_staged_segy_to_raw_reuses_same_hash_raw(tmp_path: Path):
    upload_dir = tmp_path / 'uploads'
    data = b'reusable-raw'
    source_sha256 = hashlib.sha256(data).hexdigest()
    staged_path = upload_dir / 'staged' / 'stage-id' / 'line.sgy'
    staged_path.parent.mkdir(parents=True)
    staged_path.write_bytes(data)
    raw_path = upload_dir / 'raw' / source_sha256 / 'line.sgy'
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(data)

    promoted = segy_upload_storage.promote_staged_segy_to_raw(
        staged_path=staged_path,
        safe_name='line.sgy',
        source_sha256=source_sha256,
        upload_dir=upload_dir,
    )

    assert promoted == raw_path
    assert raw_path.read_bytes() == data


def test_cleanup_staged_upload_refuses_paths_outside_staged_root(tmp_path: Path):
    upload_dir = tmp_path / 'uploads'
    outside_dir = tmp_path / 'outside'
    outside_path = outside_dir / 'unsafe.sgy'
    outside_dir.mkdir()
    outside_path.write_bytes(b'unsafe')

    segy_upload_storage.cleanup_staged_upload(outside_path, upload_dir=upload_dir)

    assert outside_path.exists()
    assert outside_dir.exists()


def test_compare_raw_import_ingests_content_addressed_store(_staged_env):
    client, upload_mod, calls = _staged_env
    data = b'compare-raw'
    source_sha256 = hashlib.sha256(data).hexdigest()

    response = _compare_raw_import(client, data=data)

    assert response.status_code == 200, response.text
    body = response.json()
    expected_store_name = f'line_001__k189_193__sha256_{source_sha256}'
    assert calls == {'qc': 1, 'ingest': 1}
    assert body['file_id']
    assert body['display_name'] == 'line 001.sgy'
    assert body['original_name'] == 'line 001.sgy'
    assert body['safe_name'] == 'line_001.sgy'
    assert body['store_name'] == expected_store_name
    assert body['source_sha256'] == source_sha256
    assert body['source_size'] == len(data)
    assert body['key1_byte'] == 189
    assert body['key2_byte'] == 193
    assert body['reused_trace_store'] is False
    assert body['header_qc'] == {
        'selected_pair_score': 0.94,
        'confidence': 'high',
        'warnings': [],
    }

    store_dir = Path(upload_mod.TRACE_DIR) / expected_store_name
    meta = json.loads((store_dir / 'meta.json').read_text(encoding='utf-8'))
    durable_raw_path = (
        Path(upload_mod.UPLOAD_DIR) / 'raw' / source_sha256 / 'line_001.sgy'
    )
    assert meta['original_segy_path'] == str(durable_raw_path)
    assert meta['original_name'] == 'line 001.sgy'
    assert meta['display_name'] == 'line 001.sgy'
    assert meta['store_name'] == expected_store_name
    assert meta['source_sha256'] == source_sha256
    assert meta['source_size'] == len(data)
    assert durable_raw_path.read_bytes() == data
    staged_root = Path(upload_mod.UPLOAD_DIR) / 'staged'
    assert not staged_root.exists() or list(staged_root.iterdir()) == []


def test_recent_datasets_uses_original_display_name_for_compare_import(_staged_env):
    client, _upload_mod, _calls = _staged_env
    data = b'recent-compare-raw'
    source_sha256 = hashlib.sha256(data).hexdigest()

    imported = _compare_raw_import(client, data=data)
    assert imported.status_code == 200, imported.text
    store_name = imported.json()['store_name']

    response = client.get('/recent_datasets')

    assert response.status_code == 200, response.text
    datasets = response.json()['datasets']
    item = next(dataset for dataset in datasets if dataset['store_name'] == store_name)
    assert item['display_name'] == 'line 001.sgy'
    assert item['original_name'] == 'line 001.sgy'
    assert '__sha256_' not in item['display_name']
    assert item['store_name'] == store_name
    assert item['source_sha256'] == source_sha256
    assert item['key1_byte'] == 189
    assert item['key2_byte'] == 193


def test_compare_raw_import_long_filename_store_name_is_bounded(_staged_env):
    client, upload_mod, _calls = _staged_env
    data = b'long-name-compare-raw'
    source_sha256 = hashlib.sha256(data).hexdigest()
    original_name = f'{"line_" * 70}.sgy'

    response = _compare_raw_import(client, name=original_name, data=data)

    assert response.status_code == 200, response.text
    body = response.json()
    suffix = f'__k189_193__sha256_{source_sha256}'
    assert len(body['store_name']) <= CONTENT_ADDRESSED_STORE_NAME_MAX_CHARS
    assert body['store_name'].endswith(suffix)
    assert '/' not in body['store_name']
    assert '\\' not in body['store_name']
    store_dir = Path(upload_mod.TRACE_DIR) / body['store_name']
    assert store_dir.is_dir()
    assert len(store_dir.name) <= CONTENT_ADDRESSED_STORE_NAME_MAX_CHARS
    meta = json.loads((store_dir / 'meta.json').read_text(encoding='utf-8'))
    raw_path = Path(meta['original_segy_path'])
    assert raw_path.is_file()
    assert raw_path.parent == Path(upload_mod.UPLOAD_DIR) / 'raw' / source_sha256
    assert len(raw_path.name) <= DIRECT_IMPORT_RAW_NAME_MAX_CHARS
    assert meta['original_name'] == original_name
    assert meta['display_name'] == original_name
    assert meta['store_name'] == body['store_name']


def test_compare_raw_import_reuses_same_source_hash_and_key_bytes(_staged_env):
    client, upload_mod, calls = _staged_env
    data = b'reuse-compare-raw'

    first = _compare_raw_import(client, data=data)
    stores_after_first = {
        child.name for child in Path(upload_mod.TRACE_DIR).iterdir() if child.is_dir()
    }
    second = _compare_raw_import(client, data=data)
    stores_after_second = {
        child.name for child in Path(upload_mod.TRACE_DIR).iterdir() if child.is_dir()
    }

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()['reused_trace_store'] is False
    assert second.json()['reused_trace_store'] is True
    assert first.json()['store_name'] == second.json()['store_name']
    assert first.json()['file_id'] != second.json()['file_id']
    assert stores_after_second == stores_after_first
    assert calls == {'qc': 2, 'ingest': 1}


def test_compare_raw_import_same_basename_different_content_creates_distinct_stores(
    _staged_env,
):
    client, _upload_mod, _calls = _staged_env
    first_sha = hashlib.sha256(b'compare-a').hexdigest()
    second_sha = hashlib.sha256(b'compare-b').hexdigest()

    first = _compare_raw_import(client, name='line 001.sgy', data=b'compare-a')
    second = _compare_raw_import(client, name='line 001.sgy', data=b'compare-b')

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    first_body = first.json()
    second_body = second.json()
    assert first_body['store_name'] != second_body['store_name']
    assert first_body['source_sha256'] == first_sha
    assert second_body['source_sha256'] == second_sha

    recent = client.get('/recent_datasets')
    assert recent.status_code == 200, recent.text
    by_sha = {
        item['source_sha256']: item
        for item in recent.json()['datasets']
        if item.get('source_sha256') in {first_sha, second_sha}
    }
    assert set(by_sha) == {first_sha, second_sha}
    assert by_sha[first_sha]['store_name'] == first_body['store_name']
    assert by_sha[second_sha]['store_name'] == second_body['store_name']
    assert by_sha[first_sha]['display_name'] == 'line 001.sgy'
    assert by_sha[second_sha]['display_name'] == 'line 001.sgy'


def test_compare_raw_import_same_basename_does_not_archive_upload_store(
    _staged_env,
):
    client, upload_mod, _calls = _staged_env
    data = b'different-compare-source'
    source_sha256 = hashlib.sha256(data).hexdigest()
    expected_store_name = content_addressed_compare_store_name(
        safe_name='line_001.sgy',
        source_sha256=source_sha256,
        key1_byte=189,
        key2_byte=193,
    )
    normal_store = Path(upload_mod.TRACE_DIR) / 'line_001.sgy'
    normal_store.mkdir(parents=True)
    sentinel = normal_store / 'sentinel.txt'
    sentinel.write_text('keep', encoding='utf-8')

    response = _compare_raw_import(client, data=data)

    assert response.status_code == 200, response.text
    assert response.json()['store_name'] == expected_store_name
    assert sentinel.read_text(encoding='utf-8') == 'keep'
    assert normal_store.is_dir()
    assert (Path(upload_mod.TRACE_DIR) / expected_store_name).is_dir()
    assert not list(Path(upload_mod.TRACE_DIR).glob('line_001.sgy.old-*'))


def test_compare_raw_import_content_addressed_conflict_returns_409_without_archive(
    _staged_env,
):
    client, upload_mod, calls = _staged_env
    data = b'conflicting-compare-source'
    source_sha256 = hashlib.sha256(data).hexdigest()
    store_name = content_addressed_compare_store_name(
        safe_name='line_001.sgy',
        source_sha256=source_sha256,
        key1_byte=189,
        key2_byte=193,
    )
    store_dir = Path(upload_mod.TRACE_DIR) / store_name
    _write_complete_compare_store(
        store_dir,
        source_sha256=hashlib.sha256(b'other-source').hexdigest(),
        source_size=len(data),
    )

    response = _compare_raw_import(client, data=data)

    assert response.status_code == 409
    assert response.json()['detail'] == (
        'Trace store already exists for a different source or key bytes'
    )
    assert store_dir.is_dir()
    assert not list(Path(upload_mod.TRACE_DIR).glob(f'{store_name}.old-*'))
    assert calls == {'qc': 1, 'ingest': 0}


def test_open_segy_store_name_selects_requested_compare_store_with_same_original_name(
    _staged_env,
):
    client, upload_mod, calls = _staged_env
    legacy_store = Path(upload_mod.TRACE_DIR) / 'line_001.sgy'
    target_store_name = 'line_001__k189_193__sha256_bbbbbbbb'
    target_store = Path(upload_mod.TRACE_DIR) / target_store_name
    _write_complete_compare_store(
        legacy_store,
        source_sha256='aaaaaaaa',
        source_size=10,
        original_name='line 001.sgy',
        dt=0.001,
    )
    _write_complete_compare_store(
        target_store,
        source_sha256='bbbbbbbb',
        source_size=20,
        original_name='line 001.sgy',
        dt=0.007,
    )

    response = client.post(
        '/open_segy',
        data={
            'original_name': 'line 001.sgy',
            'store_name': target_store_name,
            'key1_byte': '189',
            'key2_byte': '193',
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['store_name'] == target_store_name
    assert body['original_name'] == 'line 001.sgy'
    assert body['source_sha256'] == 'bbbbbbbb'
    assert app.state.sv.file_registry.get_record(body['file_id'])['store_path'] == str(
        target_store
    )
    assert calls == {'qc': 0, 'ingest': 0}


@pytest.mark.parametrize('store_name', ['../x', 'a/b', '.', '..'])
def test_open_segy_rejects_compare_store_name_traversal(_staged_env, store_name: str):
    client, _upload_mod, calls = _staged_env

    response = client.post(
        '/open_segy',
        data={
            'original_name': 'line 001.sgy',
            'store_name': store_name,
            'key1_byte': '189',
            'key2_byte': '193',
        },
    )

    assert response.status_code == 400
    assert response.json()['detail'] == 'Trace store name is unsafe'
    assert calls == {'qc': 0, 'ingest': 0}


def test_recent_datasets_include_compare_store_name_and_source_sha256(_staged_env):
    client, upload_mod, calls = _staged_env
    store_name = 'line_001__k189_193__sha256_cccccccc'
    _write_complete_compare_store(
        Path(upload_mod.TRACE_DIR) / store_name,
        source_sha256='cccccccc',
        source_size=30,
        original_name='line 001.sgy',
    )

    response = client.get('/recent_datasets')

    assert response.status_code == 200, response.text
    datasets = response.json()['datasets']
    assert datasets[0]['original_name'] == 'line 001.sgy'
    assert datasets[0]['store_name'] == store_name
    assert datasets[0]['source_sha256'] == 'cccccccc'
    assert calls == {'qc': 0, 'ingest': 0}


def test_compare_raw_import_rejects_duplicate_key_bytes(_staged_env):
    client, _upload_mod, calls = _staged_env

    response = _compare_raw_import(client, key1_byte=189, key2_byte=189)

    assert response.status_code == 400
    assert response.json()['detail'] == 'key1_byte and key2_byte must be different'
    assert calls == {'qc': 0, 'ingest': 0}


def test_compare_raw_import_rejects_missing_filename(_staged_env):
    _client, upload_mod, calls = _staged_env
    upload = UploadFile(file=io.BytesIO(b'data'), filename='')

    with pytest.raises(upload_mod.HTTPException) as exc_info:
        _run_async(
            upload_mod.import_compare_raw(
                request=SimpleNamespace(app=app),
                file=upload,
                key1_byte=189,
                key2_byte=193,
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == 'Uploaded file must have a filename'
    assert calls == {'qc': 0, 'ingest': 0}


def test_compare_raw_import_qc_failure_removes_staged_file(
    _staged_env,
    monkeypatch,
):
    client, upload_mod, calls = _staged_env
    from app.services import compare_raw_import_service

    def _raise_qc(_path: str | Path) -> dict:
        calls['qc'] += 1
        raise RuntimeError('bad headers')

    monkeypatch.setattr(
        compare_raw_import_service,
        'inspect_segy_header_qc',
        _raise_qc,
        raising=True,
    )

    response = _compare_raw_import(client, name='bad.sgy', data=b'bad-data')

    assert response.status_code == 422
    assert response.json()['detail'] == 'Unable to inspect SEG-Y headers: bad headers'
    assert calls == {'qc': 1, 'ingest': 0}
    staged_root = Path(upload_mod.UPLOAD_DIR) / 'staged'
    assert not staged_root.exists() or list(staged_root.iterdir()) == []


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
    assert set(body) == {'file_id', 'reused_trace_store', 'store_name'}
    assert body['file_id']
    assert body['reused_trace_store'] is False
    assert body['store_name'] == 'direct.sgy'
    assert calls == {'qc': 0, 'ingest': 1}
