import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.tests._stubs import write_baseline_raw
from app.trace_store.catalog import (
    ensure_trace_store_meta_identity,
    ensure_trace_store_meta_key_bytes,
    list_recent_dataset_summaries,
    load_trace_store_meta,
    trace_store_matches_source,
)


def _write_complete_store(
    store_dir: Path,
    *,
    key1_byte: int = 189,
    key2_byte: int = 193,
    original_name: str = 'line.sgy',
    display_name: str | None = None,
    source_sha256: str | None = None,
    original_size: int | None = None,
    source_size: int | None = None,
    with_baseline: bool = True,
) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / 'traces.npy').write_bytes(b'traces')
    meta: dict[str, object] = {
        'original_name': original_name,
        'key_bytes': {'key1': key1_byte, 'key2': key2_byte},
        'dt': 0.004,
        'n_traces': 2,
        'n_samples': 4,
    }
    if display_name is not None:
        meta['display_name'] = display_name
    if source_sha256 is not None:
        meta['source_sha256'] = source_sha256
    if original_size is not None:
        meta['original_size'] = original_size
    if source_size is not None:
        meta['source_size'] = source_size
    (store_dir / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    write_baseline_raw(
        store_dir,
        key1=1,
        n_traces=2,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        source_sha256=source_sha256 if with_baseline else None,
    )
    if not with_baseline:
        for path in store_dir.glob('baseline_raw.k1_*.k2_*.*'):
            path.unlink()


def test_list_recent_dataset_summaries_lists_complete_stores_only(tmp_path):
    trace_dir = tmp_path / 'trace_store'
    normal = trace_dir / 'line.sgy'
    content_addressed = trace_dir / 'line__k189_193__sha256_abc'
    missing_baseline = trace_dir / 'missing-baseline.sgy'

    _write_complete_store(
        normal,
        original_name='line.sgy',
        source_sha256='sha-normal',
        source_size=10,
    )
    _write_complete_store(
        content_addressed,
        original_name='line.sgy',
        display_name='Line 1',
        source_sha256='abc',
        source_size=20,
    )
    _write_complete_store(
        missing_baseline,
        original_name='missing-baseline.sgy',
        source_sha256='sha-missing',
        with_baseline=False,
    )

    normal_mtime = (normal / 'meta.json').stat().st_mtime
    newer_mtime = normal_mtime + 5
    os.utime(content_addressed / 'meta.json', (newer_mtime, newer_mtime))

    summaries = list_recent_dataset_summaries(trace_dir)

    assert [item['store_name'] for item in summaries] == [
        'line__k189_193__sha256_abc',
        'line.sgy',
    ]
    assert summaries[0]['original_name'] == 'line.sgy'
    assert summaries[0]['display_name'] == 'Line 1'
    assert summaries[0]['source_sha256'] == 'abc'
    assert summaries[0]['source_size'] == 20


def test_ensure_trace_store_meta_key_bytes_rejects_mismatch():
    with pytest.raises(HTTPException) as exc_info:
        ensure_trace_store_meta_key_bytes(
            {'key_bytes': {'key1': 189, 'key2': 193}},
            key1_byte=189,
            key2_byte=17,
        )

    assert exc_info.value.status_code == 409


def test_ensure_trace_store_meta_identity_persists_identity_keys(tmp_path):
    store_dir = tmp_path / 'store'
    store_dir.mkdir()
    raw_path = tmp_path / 'line.sgy'
    raw_path.write_bytes(b'segy')
    meta_path = store_dir / 'meta.json'
    meta_path.write_text(
        json.dumps({'key_bytes': {'key1': 189, 'key2': 193}}),
        encoding='utf-8',
    )

    updated = ensure_trace_store_meta_identity(
        store_dir,
        load_trace_store_meta(meta_path) or {},
        raw_path=raw_path,
        original_name='line.sgy',
        display_name='Line 1',
        store_name='store',
        source_sha256='sha',
        source_size=4,
    )
    persisted = load_trace_store_meta(meta_path)

    assert persisted == updated
    assert persisted is not None
    assert persisted['original_segy_path'] == str(raw_path)
    assert persisted['original_size'] == 4
    assert persisted['source_size'] == 4
    assert persisted['source_sha256'] == 'sha'
    assert persisted['original_name'] == 'line.sgy'
    assert persisted['display_name'] == 'Line 1'
    assert persisted['store_name'] == 'store'
    assert list(store_dir.glob('meta.json.*.tmp')) == []


def test_trace_store_matches_source_prefers_hash_over_size(tmp_path):
    store_dir = tmp_path / 'store'
    _write_complete_store(
        store_dir,
        source_sha256='sha-a',
        original_size=10,
        source_size=10,
    )

    assert (
        trace_store_matches_source(
            store_dir,
            189,
            193,
            source_sha256='sha-b',
            source_size=10,
        )
        is None
    )
    assert trace_store_matches_source(
        store_dir,
        189,
        193,
        source_sha256='sha-a',
        source_size=999,
    ) is not None


def test_trace_store_matches_source_uses_size_for_legacy_store(tmp_path):
    store_dir = tmp_path / 'legacy'
    _write_complete_store(store_dir, source_sha256=None, original_size=10)

    assert trace_store_matches_source(
        store_dir,
        189,
        193,
        source_size=10,
    ) is not None
    assert (
        trace_store_matches_source(
            store_dir,
            189,
            193,
            source_size=9,
        )
        is None
    )
