# app/tests/test_upload_segy_reuse_archive.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.tests._stubs import write_baseline_raw
from app.utils.baseline_artifacts import (
    BASELINE_STAGE_RAW,
    build_baseline_manifest_path,
    build_baseline_npz_path,
)


def _write_complete_store(
    store_dir: Path,
    *,
    key1_byte: int,
    key2_byte: int,
    meta_overrides: dict | None = None,
) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    traces = np.zeros((2, 4), dtype=np.float32)
    np.save(store_dir / 'traces.npy', traces)
    np.savez(
        store_dir / 'index.npz',
        key1_values=np.asarray([1], dtype=np.int32),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([2], dtype=np.int64),
    )
    meta: dict = {
        'key_bytes': {'key1': int(key1_byte), 'key2': int(key2_byte)},
        'dt': 0.002,
        'original_segy_path': str(store_dir / 'original.segy'),
        'original_size': 0,
    }
    if meta_overrides:
        meta.update(meta_overrides)
    (store_dir / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    write_baseline_raw(
        store_dir,
        key1=1,
        n_traces=2,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        source_sha256=meta.get('source_sha256'),
    )


@pytest.fixture()
def _upload_env(
    tmp_path: Path, monkeypatch
) -> tuple[TestClient, object, dict[str, int]]:
    """Isolate UPLOAD_DIR/TRACE_DIR into a temp folder and stub ingest."""
    from app.api.routers import upload as upload_mod

    app.state.sv.file_registry.clear()
    state = getattr(getattr(app, 'state', None), 'sv', None)
    if state is not None:
        getattr(state, 'cached_readers', {}).clear()

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

    calls = {'ingest': 0}

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
        traces = np.zeros((2, 4), dtype=np.float32)
        np.save(store_path / 'traces.npy', traces)
        np.savez(
            store_path / 'index.npz',
            key1_values=np.asarray([1], dtype=np.int32),
            key1_offsets=np.asarray([0], dtype=np.int64),
            key1_counts=np.asarray([2], dtype=np.int64),
        )
        sz = Path(path).stat().st_size
        meta = {
            'key_bytes': {'key1': int(key1_byte), 'key2': int(key2_byte)},
            'dt': 0.002,
            'original_segy_path': str(path),
            'original_size': int(sz),
        }
        if source_sha256 is not None:
            meta['source_sha256'] = str(source_sha256)
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

    monkeypatch.setattr(
        upload_mod.SegyIngestor, 'from_segy', _fake_from_segy, raising=True
    )

    return TestClient(app), upload_mod, calls


def _upload(
    client: TestClient,
    *,
    name: str,
    data: bytes,
    key1_byte: int,
    key2_byte: int,
) -> dict:
    r = client.post(
        '/upload_segy',
        files={'file': (name, data, 'application/octet-stream')},
        data={'key1_byte': str(key1_byte), 'key2_byte': str(key2_byte)},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert isinstance(out, dict)
    assert 'file_id' in out
    assert 'reused_trace_store' in out
    return out


def _load_meta(store_dir: Path) -> dict:
    return json.loads((store_dir / 'meta.json').read_text(encoding='utf-8'))


def test_upload_segy_reuses_existing_store_on_same_sha_and_key_bytes(_upload_env):
    client, upload_mod, calls = _upload_env
    data = b'abc123'

    out1 = _upload(client, name='dummy.segy', data=data, key1_byte=189, key2_byte=193)
    assert out1['reused_trace_store'] is False
    assert calls['ingest'] == 1

    out2 = _upload(client, name='dummy.segy', data=data, key1_byte=189, key2_byte=193)
    assert out2['reused_trace_store'] is True
    assert calls['ingest'] == 1

    store_dir = Path(upload_mod.TRACE_DIR) / 'dummy.segy'
    assert store_dir.is_dir()
    archived = list(Path(upload_mod.TRACE_DIR).glob('dummy.segy.old-*'))
    assert not archived

    meta = _load_meta(store_dir)
    assert meta.get('key_bytes') == {'key1': 189, 'key2': 193}
    assert meta.get('source_sha256') == hashlib.sha256(data).hexdigest()


def test_upload_segy_archives_store_when_key_bytes_change(_upload_env):
    client, upload_mod, calls = _upload_env
    data = b'abc123'

    _upload(client, name='dummy.segy', data=data, key1_byte=189, key2_byte=193)
    assert calls['ingest'] == 1
    before_meta = _load_meta(Path(upload_mod.TRACE_DIR) / 'dummy.segy')
    assert before_meta.get('key_bytes') == {'key1': 189, 'key2': 193}

    out = _upload(client, name='dummy.segy', data=data, key1_byte=189, key2_byte=17)
    assert out['reused_trace_store'] is False
    assert calls['ingest'] == 2

    trace_root = Path(upload_mod.TRACE_DIR)
    archived = sorted(trace_root.glob('dummy.segy.old-*'))
    assert len(archived) == 1
    arch_meta = _load_meta(archived[0])
    assert arch_meta.get('key_bytes') == {'key1': 189, 'key2': 193}

    store_dir = trace_root / 'dummy.segy'
    assert store_dir.is_dir()
    new_meta = _load_meta(store_dir)
    assert new_meta.get('key_bytes') == {'key1': 189, 'key2': 17}


def test_upload_segy_legacy_meta_size_fallback_match_and_mismatch(_upload_env):
    client, upload_mod, calls = _upload_env
    data = b'0123456789'
    sz = len(data)
    store_dir = Path(upload_mod.TRACE_DIR) / 'legacy.segy'

    _write_complete_store(
        store_dir,
        key1_byte=189,
        key2_byte=193,
        meta_overrides={
            'original_size': sz,
        },
    )
    meta = _load_meta(store_dir)
    assert 'source_sha256' not in meta or meta.get('source_sha256') is None

    out1 = _upload(client, name='legacy.segy', data=data, key1_byte=189, key2_byte=193)
    assert out1['reused_trace_store'] is True
    assert calls['ingest'] == 0

    meta2 = _load_meta(store_dir)
    meta2['original_size'] = sz + 1
    (store_dir / 'meta.json').write_text(json.dumps(meta2), encoding='utf-8')

    out2 = _upload(client, name='legacy.segy', data=data, key1_byte=189, key2_byte=193)
    assert out2['reused_trace_store'] is False
    assert calls['ingest'] == 1
    archived = list(Path(upload_mod.TRACE_DIR).glob('legacy.segy.old-*'))
    assert len(archived) == 1


@pytest.mark.parametrize('case', ['corrupt_meta', 'missing_index', 'missing_traces'])
def test_upload_segy_bad_store_triggers_rebuild(_upload_env, case: str):
    client, upload_mod, calls = _upload_env
    data = b'abc123'
    store_dir = Path(upload_mod.TRACE_DIR) / 'broken.segy'

    _write_complete_store(store_dir, key1_byte=189, key2_byte=193)
    if case == 'corrupt_meta':
        (store_dir / 'meta.json').write_text('{bad json', encoding='utf-8')
    elif case == 'missing_index':
        (store_dir / 'index.npz').unlink()
    elif case == 'missing_traces':
        (store_dir / 'traces.npy').unlink()
    else:
        raise AssertionError(case)

    out = _upload(client, name='broken.segy', data=data, key1_byte=189, key2_byte=193)
    assert out['reused_trace_store'] is False
    assert calls['ingest'] == 1

    archived = list(Path(upload_mod.TRACE_DIR).glob('broken.segy.old-*'))
    assert len(archived) == 1

    meta = _load_meta(Path(upload_mod.TRACE_DIR) / 'broken.segy')
    assert meta.get('key_bytes') == {'key1': 189, 'key2': 193}
    assert meta.get('source_sha256') == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize('baseline_mode', ['missing', 'legacy_json_only'])
def test_upload_segy_rebuilds_store_without_split_baseline_artifact(
    _upload_env, baseline_mode: str
):
    client, upload_mod, calls = _upload_env
    data = b'abc123'
    store_dir = Path(upload_mod.TRACE_DIR) / 'missing-baseline.segy'
    source_sha256 = hashlib.sha256(data).hexdigest()

    _write_complete_store(
        store_dir,
        key1_byte=189,
        key2_byte=193,
        meta_overrides={'source_sha256': source_sha256},
    )
    if baseline_mode == 'legacy_json_only':
        write_baseline_raw(
            store_dir,
            key1=1,
            n_traces=2,
            key1_byte=189,
            key2_byte=193,
            source_sha256=source_sha256,
            legacy_only=True,
        )
    build_baseline_manifest_path(
        store_dir,
        stage=BASELINE_STAGE_RAW,
        key1_byte=189,
        key2_byte=193,
    ).unlink()
    build_baseline_npz_path(
        store_dir,
        stage=BASELINE_STAGE_RAW,
        key1_byte=189,
        key2_byte=193,
    ).unlink()

    out = _upload(
        client,
        name='missing-baseline.segy',
        data=data,
        key1_byte=189,
        key2_byte=193,
    )
    assert out['reused_trace_store'] is False
    assert calls['ingest'] == 1
    archived = list(Path(upload_mod.TRACE_DIR).glob('missing-baseline.segy.old-*'))
    assert len(archived) == 1


def test_upload_segy_rebuilds_store_with_stale_baseline_artifact(_upload_env):
    client, upload_mod, calls = _upload_env
    data = b'abc123'
    source_sha256 = hashlib.sha256(data).hexdigest()
    store_dir = Path(upload_mod.TRACE_DIR) / 'stale-baseline.segy'

    _write_complete_store(
        store_dir,
        key1_byte=189,
        key2_byte=193,
        meta_overrides={'source_sha256': source_sha256},
    )
    write_baseline_raw(
        store_dir,
        key1=1,
        n_traces=2,
        key1_byte=189,
        key2_byte=193,
        source_sha256='stale-sha',
    )

    out = _upload(
        client,
        name='stale-baseline.segy',
        data=data,
        key1_byte=189,
        key2_byte=193,
    )

    assert out['reused_trace_store'] is False
    assert calls['ingest'] == 1
    archived = list(Path(upload_mod.TRACE_DIR).glob('stale-baseline.segy.old-*'))
    assert len(archived) == 1
