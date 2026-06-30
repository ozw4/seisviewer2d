from __future__ import annotations

import asyncio
import hashlib
import io
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

from app.core.state import create_app_state
from app.services import compare_raw_import_service
from app.services.compare_raw_import_service import import_compare_raw_source
from app.trace_store.naming import content_addressed_compare_store_name


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


def _upload_file(data: bytes, *, filename: str = 'line 001.sgy') -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


def _header_qc_payload() -> dict:
    return {
        'recommended_pairs': [
            {
                'key1_byte': 189,
                'key2_byte': 193,
                'score': 0.94,
                'confidence': 'high',
                'warnings': [],
            }
        ],
        'warnings': [],
    }


def test_import_compare_raw_source_builds_content_addressed_response(
    tmp_path: Path,
    monkeypatch,
):
    state = create_app_state()
    upload_dir = tmp_path / 'uploads'
    trace_dir = tmp_path / 'traces'
    trace_dir.mkdir()
    data = b'compare-service'
    source_sha256 = hashlib.sha256(data).hexdigest()
    captured: dict[str, object] = {}

    def _inspect(path: Path) -> dict:
        assert path.is_file()
        return _header_qc_payload()

    async def _ingest_saved_segy(**kwargs):
        captured.update(kwargs)
        return {
            'file_id': 'file-1',
            'store_name': kwargs['store_name'],
            'reused_trace_store': False,
        }

    monkeypatch.setattr(
        compare_raw_import_service,
        'inspect_segy_header_qc',
        _inspect,
        raising=True,
    )
    monkeypatch.setattr(
        compare_raw_import_service,
        'ingest_saved_segy',
        _ingest_saved_segy,
        raising=True,
    )

    body = _run_async(
        import_compare_raw_source(
            state=state,
            upload_dir=upload_dir,
            trace_dir=trace_dir,
            file=_upload_file(data),
            key1_byte=189,
            key2_byte=193,
        )
    )

    expected_store_name = content_addressed_compare_store_name(
        safe_name='line_001.sgy',
        source_sha256=source_sha256,
        key1_byte=189,
        key2_byte=193,
    )
    durable_raw_path = upload_dir / 'raw' / source_sha256 / 'line_001.sgy'
    assert body == {
        'file_id': 'file-1',
        'display_name': 'line 001.sgy',
        'original_name': 'line 001.sgy',
        'safe_name': 'line_001.sgy',
        'store_name': expected_store_name,
        'source_sha256': source_sha256,
        'source_size': len(data),
        'key1_byte': 189,
        'key2_byte': 193,
        'reused_trace_store': False,
        'header_qc': {
            'selected_pair_score': 0.94,
            'confidence': 'high',
            'warnings': [],
        },
    }
    assert captured['trace_dir'] == trace_dir
    assert captured['safe_name'] == 'line_001.sgy'
    assert captured['raw_path'] == durable_raw_path
    assert captured['allow_archive_existing'] is False
    assert durable_raw_path.read_bytes() == data
    staged_root = upload_dir / 'staged'
    assert not staged_root.exists() or list(staged_root.iterdir()) == []


def test_import_compare_raw_source_qc_failure_cleans_staged_file(
    tmp_path: Path,
    monkeypatch,
):
    state = create_app_state()
    upload_dir = tmp_path / 'uploads'
    trace_dir = tmp_path / 'traces'
    trace_dir.mkdir()

    def _inspect(_path: Path) -> dict:
        raise RuntimeError('bad headers')

    async def _ingest_saved_segy(**_kwargs):
        raise AssertionError('ingest should not run')

    monkeypatch.setattr(
        compare_raw_import_service,
        'inspect_segy_header_qc',
        _inspect,
        raising=True,
    )
    monkeypatch.setattr(
        compare_raw_import_service,
        'ingest_saved_segy',
        _ingest_saved_segy,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        _run_async(
            import_compare_raw_source(
                state=state,
                upload_dir=upload_dir,
                trace_dir=trace_dir,
                file=_upload_file(b'bad-data', filename='bad.sgy'),
                key1_byte=189,
                key2_byte=193,
            )
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == 'Unable to inspect SEG-Y headers: bad headers'
    staged_root = upload_dir / 'staged'
    assert not staged_root.exists() or list(staged_root.iterdir()) == []
