from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api._helpers import get_state
from app.main import app
from app.tests._stubs import write_baseline_raw

KEY1 = 189
KEY2 = 193


def _write_meta_only_store(
    store_dir: Path,
    *,
    original_segy_path: str | None,
    key1_byte: int = KEY1,
    key2_byte: int = KEY2,
    source_sha256: str | None = None,
) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        'key_bytes': {'key1': int(key1_byte), 'key2': int(key2_byte)},
        'dt': 0.002,
        'original_size': 0,
    }
    if original_segy_path is not None:
        meta['original_segy_path'] = str(original_segy_path)
    if source_sha256 is not None:
        meta['source_sha256'] = source_sha256
    (store_dir / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')


def _write_complete_store(
    store_dir: Path,
    *,
    original_segy_path: str,
    key1_byte: int = KEY1,
    key2_byte: int = KEY2,
    source_sha256: str | None = None,
    baseline_source_sha256: str | None = None,
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
    meta = {
        'key_bytes': {'key1': int(key1_byte), 'key2': int(key2_byte)},
        'dt': 0.002,
        'original_segy_path': str(original_segy_path),
        'original_size': int(Path(original_segy_path).stat().st_size),
        'source_sha256': source_sha256,
    }
    (store_dir / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    write_baseline_raw(
        store_dir,
        key1=1,
        n_traces=2,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        source_sha256=baseline_source_sha256,
    )


@pytest.fixture()
def _open_env(tmp_path: Path, monkeypatch):
    """Isolate TRACE_DIR and stub ingest for /open_segy tests."""
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

    # open_segy で segyio を触らないようにする（Thread/preload/ensure_header を止める）
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

    calls: dict[str, object] = {'ingest': 0, 'args': None, 'source_sha256': None}

    def _fake_from_segy(
        segy_path: str | Path,
        store_dir: str | Path,
        key1_byte: int,
        key2_byte: int,
        source_sha256: str | None = None,
    ) -> dict:
        calls['ingest'] = int(calls['ingest']) + 1
        calls['args'] = (
            str(segy_path),
            str(store_dir),
            int(key1_byte),
            int(key2_byte),
        )
        calls['source_sha256'] = source_sha256
        segy_p = Path(segy_path)
        if not segy_p.is_file():
            raise RuntimeError(f'SEG-Y file not found: {segy_p}')

        store_p = Path(store_dir)
        store_p.mkdir(parents=True, exist_ok=True)

        traces = np.zeros((2, 4), dtype=np.float32)
        np.save(store_p / 'traces.npy', traces)
        np.savez(
            store_p / 'index.npz',
            key1_values=np.asarray([1], dtype=np.int32),
            key1_offsets=np.asarray([0], dtype=np.int64),
            key1_counts=np.asarray([2], dtype=np.int64),
        )
        meta = {
            'key_bytes': {'key1': int(key1_byte), 'key2': int(key2_byte)},
            'dt': 0.002,
            'original_segy_path': str(segy_p),
            'original_size': int(segy_p.stat().st_size),
        }
        if source_sha256 is not None:
            meta['source_sha256'] = source_sha256
        (store_p / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
        write_baseline_raw(
            store_p,
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

    client = TestClient(app, raise_server_exceptions=False)
    return client, upload_mod, calls


@pytest.mark.parametrize('missing', ['traces', 'index', 'both'])
def test_open_segy_rebuilds_incomplete_store_from_original_path(
    _open_env, tmp_path: Path, missing: str
):
    client, upload_mod, calls = _open_env
    original_name = 'incomplete.sgy'
    store_dir = Path(upload_mod.TRACE_DIR) / original_name

    segy_path = tmp_path / 'source.sgy'
    segy_path.write_bytes(b'not-a-real-segy')

    _write_meta_only_store(
        store_dir,
        original_segy_path=str(segy_path),
        key1_byte=KEY1,
        key2_byte=KEY2,
    )

    # 途中生成を模擬（meta.json はあるが traces/index が欠損）
    if missing == 'traces':
        np.savez(
            store_dir / 'index.npz',
            key1_values=np.asarray([1], dtype=np.int32),
            key1_offsets=np.asarray([0], dtype=np.int64),
            key1_counts=np.asarray([2], dtype=np.int64),
        )
    elif missing == 'index':
        traces = np.zeros((2, 4), dtype=np.float32)
        np.save(store_dir / 'traces.npy', traces)
    elif missing == 'both':
        pass
    else:
        raise AssertionError(missing)

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
    assert payload['reused_trace_store'] is False

    assert int(calls['ingest']) == 1
    assert calls['args'] == (str(segy_path), str(store_dir), KEY1, KEY2)

    assert (store_dir / 'traces.npy').is_file()
    assert (store_dir / 'index.npz').is_file()
    assert (store_dir / 'meta.json').is_file()
    assert upload_mod._trace_store_complete(store_dir, KEY1, KEY2) is True
    assert not list(Path(upload_mod.TRACE_DIR).glob(f'{original_name}.old-*'))

    file_id = payload['file_id']
    rec = app.state.sv.file_registry.get_record(file_id)
    assert isinstance(rec, dict)
    assert rec['store_path'] == str(store_dir)
    assert rec['dt'] == pytest.approx(0.002)


def test_open_segy_incomplete_store_without_original_path_returns_500(_open_env):
    client, upload_mod, calls = _open_env
    original_name = 'no_origin.sgy'
    store_dir = Path(upload_mod.TRACE_DIR) / original_name

    _write_meta_only_store(
        store_dir,
        original_segy_path=None,
        key1_byte=KEY1,
        key2_byte=KEY2,
    )

    res = client.post(
        '/open_segy',
        data={
            'original_name': original_name,
            'key1_byte': str(KEY1),
            'key2_byte': str(KEY2),
        },
    )
    assert res.status_code == 500
    assert res.json()['detail'] == 'Trace store incomplete and SEG-Y path unavailable'
    assert int(calls['ingest']) == 0


def test_open_segy_incomplete_store_with_missing_original_file_returns_500(
    _open_env, tmp_path: Path
):
    client, upload_mod, calls = _open_env
    original_name = 'bad_origin.sgy'
    store_dir = Path(upload_mod.TRACE_DIR) / original_name
    missing_path = tmp_path / 'does_not_exist.sgy'

    _write_meta_only_store(
        store_dir,
        original_segy_path=str(missing_path),
        key1_byte=KEY1,
        key2_byte=KEY2,
    )

    res = client.post(
        '/open_segy',
        data={
            'original_name': original_name,
            'key1_byte': str(KEY1),
            'key2_byte': str(KEY2),
        },
    )
    assert res.status_code == 500
    assert int(calls['ingest']) == 1


def test_open_segy_rebuild_for_incomplete_store_forwards_known_source_sha256(
    _open_env, tmp_path: Path
):
    client, upload_mod, calls = _open_env
    original_name = 'incomplete-with-sha.sgy'
    store_dir = Path(upload_mod.TRACE_DIR) / original_name

    segy_path = tmp_path / 'source.sgy'
    segy_path.write_bytes(b'not-a-real-segy')

    _write_meta_only_store(
        store_dir,
        original_segy_path=str(segy_path),
        key1_byte=KEY1,
        key2_byte=KEY2,
        source_sha256='known-sha',
    )

    res = client.post(
        '/open_segy',
        data={
            'original_name': original_name,
            'key1_byte': str(KEY1),
            'key2_byte': str(KEY2),
        },
    )

    assert res.status_code == 200, res.text
    assert res.json()['reused_trace_store'] is False
    assert int(calls['ingest']) == 1
    assert calls['source_sha256'] == 'known-sha'

    rebuilt_meta = json.loads((store_dir / 'meta.json').read_text(encoding='utf-8'))
    assert rebuilt_meta['source_sha256'] == 'known-sha'


def test_open_segy_rebuilds_store_when_baseline_artifact_is_stale(
    _open_env, tmp_path: Path
):
    client, upload_mod, calls = _open_env
    original_name = 'stale-baseline.sgy'
    store_dir = Path(upload_mod.TRACE_DIR) / original_name
    segy_path = tmp_path / 'source.sgy'
    segy_path.write_bytes(b'not-a-real-segy')

    _write_complete_store(
        store_dir,
        original_segy_path=str(segy_path),
        source_sha256='store-sha',
        baseline_source_sha256='stale-sha',
    )

    res = client.post(
        '/open_segy',
        data={
            'original_name': original_name,
            'key1_byte': str(KEY1),
            'key2_byte': str(KEY2),
        },
    )

    assert res.status_code == 200, res.text
    assert res.json()['reused_trace_store'] is False
    assert int(calls['ingest']) == 1
