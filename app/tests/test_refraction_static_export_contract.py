from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
from app.api.schemas import (
    REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS,
    RefractionStaticApplyRequest,
    RefractionStaticExportJobRequest,
)
from app.main import app
from app.services.refraction_static_artifacts import (
    RECEIVER_STATIC_TABLE_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_REQUEST_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)
from app.services.refraction_static_export_service import (
    REFRACTION_STATIC_EXPORT_JOB_META_JSON_NAME,
    REFRACTION_STATIC_EXPORT_REQUEST_JSON_NAME,
    run_refraction_static_export_job,
)
from app.tests._refraction_static_synthetic import synthetic_refraction_apply_request


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(tmp_path / 'jobs'))
    state = app.state.sv
    with state.lock:
        state.jobs.clear()
        state.cached_readers.clear()
        state.file_registry.clear()
    with TestClient(app) as test_client:
        yield test_client
    with state.lock:
        state.jobs.clear()
        state.cached_readers.clear()
        state.file_registry.clear()


def _apply_payload() -> dict[str, Any]:
    payload = synthetic_refraction_apply_request().model_dump(mode='json')
    payload.pop('export', None)
    return payload


def _source_artifacts_for_default_export() -> tuple[str, ...]:
    return (
        REFRACTION_STATIC_REQUEST_JSON_NAME,
        REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    )


def _create_source_refraction_job(
    client: TestClient,
    tmp_path: Path,
    *,
    job_id: str = 'source-refraction-job',
    artifact_names: tuple[str, ...] = (),
    status: str = 'done',
) -> Path:
    job_dir = tmp_path / 'jobs' / job_id
    job_dir.mkdir(parents=True)
    for artifact_name in artifact_names:
        (job_dir / artifact_name).write_bytes(b'data')

    state = client.app.state.sv
    with state.lock:
        state.jobs.create_static_job(
            job_id,
            file_id='raw-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction',
            artifacts_dir=str(job_dir),
        )
        if status == 'done':
            state.jobs.mark_done(job_id, progress_1=True)
        else:
            state.jobs.set_status(job_id, status)
    return job_dir


def test_refraction_apply_accepts_export_block(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    payload = _apply_payload()
    payload['export'] = {'enabled': True, 'formats': []}

    response = client.post('/statics/refraction/apply', json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body['requested_formats'] == list(REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS)
    assert len(started) == 1
    req = started[0]['args'][1]
    assert isinstance(req, RefractionStaticApplyRequest)
    assert req.export.enabled is True
    assert req.export.formats == []
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[body['job_id']])
    assert job['export_formats'] == list(REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS)


def test_refraction_apply_legacy_request_without_export_still_valid(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )

    response = client.post('/statics/refraction/apply', json=_apply_payload())

    assert response.status_code == 200
    assert 'requested_formats' not in response.json()
    assert len(started) == 1


def test_refraction_export_job_request_requires_source_job_id(
    client: TestClient,
) -> None:
    response = client.post(
        '/statics/refraction/export',
        json={'export': {'enabled': True}},
    )

    assert response.status_code == 422


def test_refraction_export_rejects_unknown_format(client: TestClient) -> None:
    response = client.post(
        '/statics/refraction/export',
        json={
            'source_job_id': 'source-refraction-job',
            'export': {'enabled': True, 'formats': ['unknown_format']},
        },
    )

    assert response.status_code == 422


def test_refraction_export_rejects_incomplete_source_job(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _create_source_refraction_job(
        client,
        tmp_path,
        artifact_names=(
            REFRACTION_STATIC_REQUEST_JSON_NAME,
            REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
        ),
    )

    response = client.post(
        '/statics/refraction/export',
        json={
            'source_job_id': 'source-refraction-job',
            'export': {'enabled': True, 'formats': []},
        },
    )

    assert response.status_code == 409
    assert SOURCE_STATIC_TABLE_CSV_NAME in response.json()['detail']


def test_refraction_export_endpoint_starts_metadata_job(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    _create_source_refraction_job(
        client,
        tmp_path,
        artifact_names=_source_artifacts_for_default_export(),
    )

    response = client.post(
        '/statics/refraction/export',
        json={
            'source_job_id': 'source-refraction-job',
            'export': {'enabled': True, 'formats': []},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body['source_job_id'] == 'source-refraction-job'
    assert body['requested_formats'] == list(REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS)
    assert len(started) == 1
    assert started[0]['target'] is statics_router_module.run_refraction_static_export_job
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[body['job_id']])
    assert job['statics_kind'] == 'refraction_export'
    assert job['source_job_id'] == 'source-refraction-job'
    assert job['export_formats'] == list(REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS)


def test_run_refraction_static_export_job_writes_requested_format_metadata(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _create_source_refraction_job(
        client,
        tmp_path,
        artifact_names=_source_artifacts_for_default_export(),
    )
    req = RefractionStaticExportJobRequest.model_validate(
        {
            'source_job_id': 'source-refraction-job',
            'export': {'enabled': True, 'formats': []},
        }
    )
    export_job_id = 'export-job'
    export_job_dir = tmp_path / 'jobs' / export_job_id
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            export_job_id,
            file_id='raw-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction_export',
            artifacts_dir=str(export_job_dir),
        )

    run_refraction_static_export_job(export_job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[export_job_id])
    assert job['status'] == 'done'
    meta = json.loads(
        (export_job_dir / REFRACTION_STATIC_EXPORT_JOB_META_JSON_NAME).read_text(
            encoding='utf-8',
        )
    )
    request_meta = json.loads(
        (export_job_dir / REFRACTION_STATIC_EXPORT_REQUEST_JSON_NAME).read_text(
            encoding='utf-8',
        )
    )
    assert meta == request_meta
    assert meta['statics_kind'] == 'refraction_export'
    assert meta['source_job_id'] == 'source-refraction-job'
    assert meta['request']['export']['formats'] == []
    assert meta['export']['requested_formats'] == list(
        REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS
    )
