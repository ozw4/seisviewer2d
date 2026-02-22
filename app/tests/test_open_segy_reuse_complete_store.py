from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api._helpers import get_state
from app.main import app

KEY1 = 189
KEY2 = 193


def _write_complete_store(
    store_dir: Path,
    *,
    key1_byte: int,
    key2_byte: int,
    dt: float = 0.002,
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
        'dt': float(dt),
        'original_segy_path': str(store_dir / 'original.segy'),
        'original_size': 0,
    }
    if meta_overrides:
        meta.update(meta_overrides)
    (store_dir / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')


@pytest.fixture()
def _open_env(tmp_path: Path, monkeypatch):
    """Isolate TRACE_DIR and prevent side-effect threads for /open_segy reuse tests."""
    from app.api.routers import upload as upload_mod

    app.state.sv.file_registry.clear()
    get_state(app).cached_readers.clear()

    upload_root = tmp_path / 'uploads'
    processed = upload_root / 'processed'
    trace_dir = processed / 'traces'
    trace_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(upload_mod, 'UPLOAD_DIR', upload_root, raising=True)
    monkeypatch.setattr(upload_mod, 'PROCESSED_DIR', processed, raising=True)
    monkeypatch.setattr(upload_mod, 'TRACE_DIR', trace_dir, raising=True)

    captured: dict[str, object] = {'register': None}

    def _fake_register(
        file_id: str, store_dir: Path, key1_byte: int, key2_byte: int, *, state
    ):
        captured['register'] = (
            str(file_id),
            str(store_dir),
            int(key1_byte),
            int(key2_byte),
        )
        return None

    monkeypatch.setattr(
        upload_mod, '_register_trace_store', _fake_register, raising=True
    )

    calls: dict[str, int] = {'ingest': 0}

    def _should_not_be_called(*_a, **_k):
        calls['ingest'] += 1
        raise AssertionError('SegyIngestor.from_segy must not be called for reuse path')

    monkeypatch.setattr(
        upload_mod.SegyIngestor, 'from_segy', _should_not_be_called, raising=True
    )

    client = TestClient(app)
    return client, upload_mod, captured, calls


def test_open_segy_reuses_complete_store_sets_reused_true(_open_env):
    client, upload_mod, captured, calls = _open_env
    original_name = 'complete.segy'
    store_dir = Path(upload_mod.TRACE_DIR) / original_name
    _write_complete_store(store_dir, key1_byte=KEY1, key2_byte=KEY2, dt=0.004)

    res = client.post(
        '/open_segy',
        data={
            'original_name': original_name,
            'key1_byte': str(KEY1),
            'key2_byte': str(KEY2),
        },
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload['reused_trace_store'] is True
    assert isinstance(payload.get('file_id'), str)
    assert calls['ingest'] == 0

    file_id = payload['file_id']
    rec = app.state.sv.file_registry.get_record(file_id)
    assert isinstance(rec, dict)
    assert rec['store_path'] == str(store_dir)
    assert rec['dt'] == pytest.approx(0.004)

    assert captured['register'] is not None
    reg_file_id, reg_store, reg_k1, reg_k2 = captured['register']
    assert reg_file_id == file_id
    assert reg_store == str(store_dir)
    assert reg_k1 == KEY1
    assert reg_k2 == KEY2


def test_open_segy_sanitizes_original_name_and_prevents_path_traversal(_open_env):
    client, upload_mod, captured, calls = _open_env
    original_name = '../../etc/passwd'
    safe_name = '.._.._etc_passwd'
    store_dir = Path(upload_mod.TRACE_DIR) / safe_name
    _write_complete_store(store_dir, key1_byte=KEY1, key2_byte=KEY2, dt=0.006)

    # サニタイズが無ければ TRACE_DIR / original_name が到達するはずの場所に、
    # 罠（別dt）の完全ストアを置いておく
    outside_dir = (Path(upload_mod.TRACE_DIR) / original_name).resolve()
    _write_complete_store(outside_dir, key1_byte=KEY1, key2_byte=KEY2, dt=0.001)

    res = client.post(
        '/open_segy',
        data={
            'original_name': original_name,
            'key1_byte': str(KEY1),
            'key2_byte': str(KEY2),
        },
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload['reused_trace_store'] is True
    assert calls['ingest'] == 0

    file_id = payload['file_id']
    # 罠ではなく、TRACE_DIR 配下のサニタイズ名ストアが開かれていること
    rec = app.state.sv.file_registry.get_record(file_id)
    assert isinstance(rec, dict)
    assert rec['store_path'] == str(store_dir)
    assert rec['dt'] == pytest.approx(0.006)

    assert captured['register'] is not None
    _reg_file_id, reg_store, _reg_k1, _reg_k2 = captured['register']
    assert reg_store == str(store_dir)


def test_open_segy_path_traversal_target_store_is_not_accessible(_open_env):
    client, upload_mod, _captured, calls = _open_env
    original_name = '../../etc/passwd'

    # traversal 先（TRACE_DIR外）にだけ完全ストアを用意しても、開けないこと
    outside_dir = (Path(upload_mod.TRACE_DIR) / original_name).resolve()
    _write_complete_store(outside_dir, key1_byte=KEY1, key2_byte=KEY2)

    res = client.post(
        '/open_segy',
        data={
            'original_name': original_name,
            'key1_byte': str(KEY1),
            'key2_byte': str(KEY2),
        },
    )
    assert res.status_code == 404
    assert res.json()['detail'] == f'Trace store not found for {original_name}'
    assert calls['ingest'] == 0
