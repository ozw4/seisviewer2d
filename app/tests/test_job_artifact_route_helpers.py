from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.job_artifacts import (
    job_artifacts_dir_or_404,
    list_job_artifact_files,
    resolve_download_artifact_or_http_error,
)
from app.core.state import create_app_state


def test_list_job_artifact_files_returns_empty_list_for_empty_dir(
    tmp_path: Path,
) -> None:
    artifacts_dir = tmp_path / 'job'
    artifacts_dir.mkdir()

    assert list_job_artifact_files({'artifacts_dir': str(artifacts_dir)}) == {
        'files': []
    }


def test_list_job_artifact_files_ignores_nested_dirs_and_sorts_names(
    tmp_path: Path,
) -> None:
    artifacts_dir = tmp_path / 'job'
    artifacts_dir.mkdir()
    (artifacts_dir / 'z-last.txt').write_bytes(b'last')
    (artifacts_dir / 'a-first.txt').write_bytes(b'first')
    nested_dir = artifacts_dir / 'nested'
    nested_dir.mkdir()
    (nested_dir / 'ignored.txt').write_bytes(b'ignored')

    assert list_job_artifact_files({'artifacts_dir': str(artifacts_dir)}) == {
        'files': [
            {'name': 'a-first.txt', 'size_bytes': 5},
            {'name': 'z-last.txt', 'size_bytes': 4},
        ]
    }


def test_job_artifacts_dir_or_404_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc_info:
        job_artifacts_dir_or_404({'artifacts_dir': str(tmp_path / 'missing')})

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == 'Job artifacts not found'


@pytest.mark.parametrize('name', ['', '.', '..', '../artifact.txt', 'nested/file.txt'])
def test_resolve_download_artifact_or_http_error_rejects_invalid_name(
    tmp_path: Path,
    name: str,
) -> None:
    state = create_app_state()
    artifacts_dir = tmp_path / 'job'
    artifacts_dir.mkdir()
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-1',
            file_id='file-a',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir=str(artifacts_dir),
        )

    with pytest.raises(HTTPException) as exc_info:
        resolve_download_artifact_or_http_error(
            state,
            job_id='job-1',
            name=name,
            allowed_job_types={'batch_apply'},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == 'Invalid file name'


def test_resolve_download_artifact_or_http_error_rejects_missing_dir(
    tmp_path: Path,
) -> None:
    state = create_app_state()
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-1',
            file_id='file-a',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir=str(tmp_path / 'missing'),
        )

    with pytest.raises(HTTPException) as exc_info:
        resolve_download_artifact_or_http_error(
            state,
            job_id='job-1',
            name='artifact.txt',
            allowed_job_types={'batch_apply'},
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == 'Job artifacts not found'


def test_resolve_download_artifact_or_http_error_rejects_missing_file(
    tmp_path: Path,
) -> None:
    state = create_app_state()
    artifacts_dir = tmp_path / 'job'
    artifacts_dir.mkdir()
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-1',
            file_id='file-a',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir=str(artifacts_dir),
        )

    with pytest.raises(HTTPException) as exc_info:
        resolve_download_artifact_or_http_error(
            state,
            job_id='job-1',
            name='artifact.txt',
            allowed_job_types={'batch_apply'},
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == 'File not found'


def test_resolve_download_artifact_or_http_error_rejects_wrong_job_type(
    tmp_path: Path,
) -> None:
    state = create_app_state()
    artifacts_dir = tmp_path / 'job'
    artifacts_dir.mkdir()
    (artifacts_dir / 'artifact.txt').write_bytes(b'data')
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-1',
            file_id='file-a',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir=str(artifacts_dir),
        )

    with pytest.raises(HTTPException) as exc_info:
        resolve_download_artifact_or_http_error(
            state,
            job_id='job-1',
            name='artifact.txt',
            allowed_job_types={'statics'},
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == 'Job ID not found'


def test_resolve_download_artifact_or_http_error_returns_file(
    tmp_path: Path,
) -> None:
    state = create_app_state()
    artifacts_dir = tmp_path / 'job'
    artifacts_dir.mkdir()
    artifact_path = artifacts_dir / 'artifact.txt'
    artifact_path.write_bytes(b'data')
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-1',
            file_id='file-a',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir=str(artifacts_dir),
        )

    assert (
        resolve_download_artifact_or_http_error(
            state,
            job_id='job-1',
            name='artifact.txt',
            allowed_job_types={'batch_apply'},
        )
        == artifact_path
    )
