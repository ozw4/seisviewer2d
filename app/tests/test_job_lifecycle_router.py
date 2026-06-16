from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import job_lifecycle_router
from app.api.job_lifecycle_router import build_job_lifecycle_router
from app.contracts.batch import BatchJobFilesResponse, BatchJobStatusResponse
from app.core.state import create_app_state


@pytest.fixture()
def calls() -> list[Any]:
    return []


@pytest.fixture()
def client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    calls: list[Any],
) -> TestClient:
    artifact_path = tmp_path / 'artifact.txt'
    artifact_path.write_text('artifact payload', encoding='utf-8')

    def cleanup_in_memory_state(state: Any) -> None:
        calls.append('cleanup')

    def maybe_cleanup_expired_jobs() -> int:
        calls.append('maybe_cleanup')
        return 0

    def get_job_or_404(
        state: Any,
        job_id: str,
        *,
        allowed_job_types: set[str] | None = None,
        missing_detail: str = 'Job ID not found',
    ) -> dict[str, object]:
        calls.append(('get_job', job_id, allowed_job_types, missing_detail))
        return {'status': 'running', 'progress': 0.25, 'message': 'Running'}

    def job_status_payload(job: dict[str, object]) -> dict[str, object]:
        calls.append(('status_payload', job))
        return {'state': 'running', 'progress': 0.25, 'message': 'Running'}

    def cancel_job_and_get_status_payload(
        state: Any,
        job_id: str,
        *,
        allowed_job_types: set[str] | None = None,
    ) -> dict[str, object]:
        calls.append(('cancel_payload', job_id, allowed_job_types))
        return {'state': 'cancel_requested', 'progress': 0.25, 'message': 'Cancelling'}

    def list_job_artifact_files(job: dict[str, object]) -> dict[str, object]:
        calls.append(('list_files', job))
        return {'files': [{'name': 'artifact.txt', 'size_bytes': 16}]}

    def resolve_download_artifact_or_http_error(
        state: Any,
        *,
        job_id: str,
        name: str,
        allowed_job_types: set[str],
    ) -> Path:
        calls.append(('resolve_download', job_id, name, allowed_job_types))
        return artifact_path

    monkeypatch.setattr(
        job_lifecycle_router,
        'cleanup_in_memory_state',
        cleanup_in_memory_state,
    )
    monkeypatch.setattr(
        job_lifecycle_router,
        'maybe_cleanup_expired_jobs',
        maybe_cleanup_expired_jobs,
    )
    monkeypatch.setattr(job_lifecycle_router, 'get_job_or_404', get_job_or_404)
    monkeypatch.setattr(job_lifecycle_router, 'job_status_payload', job_status_payload)
    monkeypatch.setattr(
        job_lifecycle_router,
        'cancel_job_and_get_status_payload',
        cancel_job_and_get_status_payload,
    )
    monkeypatch.setattr(
        job_lifecycle_router,
        'list_job_artifact_files',
        list_job_artifact_files,
    )
    monkeypatch.setattr(
        job_lifecycle_router,
        'resolve_download_artifact_or_http_error',
        resolve_download_artifact_or_http_error,
    )

    app = FastAPI()
    app.state.sv = create_app_state()
    app.include_router(
        build_job_lifecycle_router(
            route_prefix='/factory/job',
            allowed_job_types={'batch_apply'},
            status_response_model=BatchJobStatusResponse,
            files_response_model=BatchJobFilesResponse,
        )
    )
    return TestClient(app)


def test_build_job_lifecycle_router_status_route_uses_shared_helpers(
    client: TestClient,
    calls: list[Any],
) -> None:
    response = client.get('/factory/job/job-1/status')

    assert response.status_code == 200
    assert response.json() == {
        'state': 'running',
        'progress': 0.25,
        'message': 'Running',
    }
    assert calls == [
        'cleanup',
        ('get_job', 'job-1', {'batch_apply'}, 'Job ID not found'),
        (
            'status_payload',
            {'status': 'running', 'progress': 0.25, 'message': 'Running'},
        ),
    ]


def test_build_job_lifecycle_router_cancel_route_uses_shared_helpers(
    client: TestClient,
    calls: list[Any],
) -> None:
    response = client.post('/factory/job/job-1/cancel')

    assert response.status_code == 200
    assert response.json() == {
        'state': 'cancel_requested',
        'progress': 0.25,
        'message': 'Cancelling',
    }
    assert calls == [
        'cleanup',
        ('cancel_payload', 'job-1', {'batch_apply'}),
    ]


def test_build_job_lifecycle_router_files_route_uses_shared_helpers(
    client: TestClient,
    calls: list[Any],
) -> None:
    response = client.get('/factory/job/job-1/files')

    assert response.status_code == 200
    assert response.json() == {
        'files': [{'name': 'artifact.txt', 'size_bytes': 16}],
    }
    assert calls == [
        'cleanup',
        ('get_job', 'job-1', {'batch_apply'}, 'Job ID not found'),
        'maybe_cleanup',
        (
            'list_files',
            {'status': 'running', 'progress': 0.25, 'message': 'Running'},
        ),
    ]


def test_build_job_lifecycle_router_download_route_uses_shared_helpers(
    client: TestClient,
    calls: list[Any],
) -> None:
    response = client.get('/factory/job/job-1/download', params={'name': 'artifact.txt'})

    assert response.status_code == 200
    assert response.content == b'artifact payload'
    assert calls == [
        'cleanup',
        ('get_job', 'job-1', {'batch_apply'}, 'Job ID not found'),
        'maybe_cleanup',
        ('resolve_download', 'job-1', 'artifact.txt', {'batch_apply'}),
    ]
