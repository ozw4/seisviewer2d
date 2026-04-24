import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.routers import upload as upload_mod
from app.tests._stubs import write_baseline_raw


def _write_store(
    store_dir: Path,
    *,
    key1: int,
    key2: int,
    dt: float,
    n_traces: int,
    n_samples: int,
    original_name: str,
    meta_text: str | None = None,
    traces: bool = True,
    index: bool = True,
    source_sha256: str | None = None,
    baseline_source_sha256: str | None = None,
    with_baseline: bool = True,
) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    if traces:
        (store_dir / 'traces.npy').write_bytes(b'0')
    if index:
        (store_dir / 'index.npz').write_bytes(b'0')
    meta_path = store_dir / 'meta.json'
    if meta_text is not None:
        meta_path.write_text(meta_text, encoding='utf-8')
        return
    meta_path.write_text(
        json.dumps(
            {
                'original_name': original_name,
                'key_bytes': {'key1': key1, 'key2': key2},
                'dt': dt,
                'n_traces': n_traces,
                'n_samples': n_samples,
                'source_sha256': source_sha256,
            }
        ),
        encoding='utf-8',
    )
    if traces and index and with_baseline:
        write_baseline_raw(
            store_dir,
            key1=1,
            n_traces=2,
            key1_byte=key1,
            key2_byte=key2,
            source_sha256=baseline_source_sha256,
        )


def test_recent_datasets_lists_complete_stores_sorted_by_meta_mtime(
    tmp_path, monkeypatch
):
    trace_dir = tmp_path / 'trace_store'
    monkeypatch.setattr(upload_mod, 'TRACE_DIR', trace_dir, raising=True)

    older = trace_dir / 'older.segy'
    newer = trace_dir / 'newer.segy'
    incomplete = trace_dir / 'incomplete.segy'
    broken = trace_dir / 'broken.segy'

    _write_store(
        older,
        key1=189,
        key2=193,
        dt=0.002,
        n_traces=32,
        n_samples=128,
        original_name='older.segy',
    )
    _write_store(
        newer,
        key1=9,
        key2=13,
        dt=0.004,
        n_traces=16,
        n_samples=64,
        original_name='newer.segy',
    )
    _write_store(
        incomplete,
        key1=189,
        key2=193,
        dt=0.002,
        n_traces=8,
        n_samples=32,
        original_name='incomplete.segy',
        index=False,
    )
    _write_store(
        broken,
        key1=189,
        key2=193,
        dt=0.002,
        n_traces=8,
        n_samples=32,
        original_name='broken.segy',
        meta_text='{bad json',
    )

    older_meta = older / 'meta.json'
    newer_meta = newer / 'meta.json'
    older_meta.touch()
    newer_meta.touch()
    older_mtime = older_meta.stat().st_mtime
    newer_mtime = newer_meta.stat().st_mtime + 5
    older_meta.touch()
    newer_meta.touch()
    older_meta.chmod(0o644)
    newer_meta.chmod(0o644)
    import os

    os.utime(older_meta, (older_mtime, older_mtime))
    os.utime(newer_meta, (newer_mtime, newer_mtime))

    client = TestClient(app)
    response = client.get('/recent_datasets')

    assert response.status_code == 200
    payload = response.json()
    datasets = payload['datasets']
    assert [item['original_name'] for item in datasets] == ['newer.segy', 'older.segy']
    assert datasets[0]['key1_byte'] == 9
    assert datasets[0]['key2_byte'] == 13
    assert datasets[0]['dt'] == 0.004
    assert datasets[0]['n_traces'] == 16
    assert datasets[0]['n_samples'] == 64
    assert datasets[0]['last_used_ts'] == newer_mtime
    assert datasets[1]['last_used_ts'] == older_mtime


@pytest.mark.parametrize('baseline_mode', ['missing', 'legacy_json_only'])
def test_recent_datasets_skips_store_without_split_baseline_artifact(
    tmp_path, monkeypatch, baseline_mode: str
):
    trace_dir = tmp_path / 'trace_store'
    monkeypatch.setattr(upload_mod, 'TRACE_DIR', trace_dir, raising=True)

    legacy = trace_dir / 'legacy.segy'
    _write_store(
        legacy,
        key1=189,
        key2=193,
        dt=0.002,
        n_traces=32,
        n_samples=128,
        original_name='legacy.segy',
        with_baseline=False,
    )
    if baseline_mode == 'legacy_json_only':
        write_baseline_raw(
            legacy,
            key1=1,
            n_traces=2,
            key1_byte=189,
            key2_byte=193,
            legacy_only=True,
        )

    client = TestClient(app)
    response = client.get('/recent_datasets')

    assert response.status_code == 200
    assert response.json()['datasets'] == []


def test_recent_datasets_skips_store_with_stale_baseline_artifact(
    tmp_path, monkeypatch
):
    trace_dir = tmp_path / 'trace_store'
    monkeypatch.setattr(upload_mod, 'TRACE_DIR', trace_dir, raising=True)

    healthy = trace_dir / 'healthy.segy'
    stale = trace_dir / 'stale.segy'

    _write_store(
        healthy,
        key1=189,
        key2=193,
        dt=0.002,
        n_traces=32,
        n_samples=128,
        original_name='healthy.segy',
        source_sha256='sha-healthy',
        baseline_source_sha256='sha-healthy',
    )
    _write_store(
        stale,
        key1=189,
        key2=193,
        dt=0.002,
        n_traces=32,
        n_samples=128,
        original_name='stale.segy',
        source_sha256='sha-store',
        baseline_source_sha256='sha-baseline',
    )

    client = TestClient(app)
    response = client.get('/recent_datasets')

    assert response.status_code == 200
    assert [item['original_name'] for item in response.json()['datasets']] == [
        'healthy.segy'
    ]
