from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.pipeline_artifacts import get_job_dir


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(tmp_path / 'pipeline_jobs'))
    state = app.state.sv
    with state.lock:
        state.jobs.clear()
    with TestClient(app) as test_client:
        yield test_client
    with state.lock:
        state.jobs.clear()


def _create_batch_job(
    client: TestClient,
    *,
    job_id: str = 'batch-job-1',
    status: str | None = None,
    progress: float | None = None,
    message: str | None = None,
) -> Path:
    artifacts_dir = get_job_dir(job_id)
    state = client.app.state.sv
    with state.lock:
        state.jobs.create_batch_apply_job(
            job_id,
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir=str(artifacts_dir),
        )
        job = state.jobs[job_id]
        if status is not None:
            job['status'] = status
        if progress is not None:
            state.jobs.set_progress(job_id, progress)
        if message is not None:
            state.jobs.set_message(job_id, message)
    return artifacts_dir


def _create_pipeline_job(
    client: TestClient,
    *,
    job_id: str = 'pipeline-job-1',
    status: str | None = None,
    progress: float | None = None,
    message: str | None = None,
) -> Path:
    artifacts_dir = get_job_dir(job_id)
    state = client.app.state.sv
    with state.lock:
        state.jobs.create_pipeline_all_job(
            job_id,
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            pipeline_key='pipeline-key-1',
            offset_byte=None,
            artifacts_dir=str(artifacts_dir),
        )
        job = state.jobs[job_id]
        if status is not None:
            job['status'] = status
        if progress is not None:
            state.jobs.set_progress(job_id, progress)
        if message is not None:
            state.jobs.set_message(job_id, message)
    return artifacts_dir


def _create_static_job(
    client: TestClient,
    *,
    job_id: str = 'static-job-1',
    status: str | None = None,
) -> Path:
    artifacts_dir = get_job_dir(job_id)
    state = client.app.state.sv
    with state.lock:
        state.jobs.create_static_job(
            job_id,
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            statics_kind='datum',
            artifacts_dir=str(artifacts_dir),
        )
        if status is not None:
            state.jobs[job_id]['status'] = status
    return artifacts_dir


@pytest.mark.parametrize(
    ('method', 'path', 'params'),
    [
        ('get', '/batch/job/missing/status', None),
        ('post', '/batch/job/missing/cancel', None),
        ('get', '/batch/job/missing/files', None),
        ('get', '/batch/job/missing/download', {'name': 'job_meta.json'}),
    ],
)
def test_batch_job_routes_reject_unknown_job(
    client: TestClient,
    method: str,
    path: str,
    params: dict[str, str] | None,
) -> None:
    response = client.request(method, path, params=params)

    assert response.status_code == 404
    assert response.json() == {'detail': 'Job ID not found'}


def test_batch_job_status_returns_state_progress_message(client: TestClient) -> None:
    _create_batch_job(
        client,
        status='running',
        progress=0.5,
        message='Applying batch pipeline',
    )

    response = client.get('/batch/job/batch-job-1/status')

    assert response.status_code == 200
    assert response.json() == {
        'state': 'running',
        'progress': 0.5,
        'message': 'Applying batch pipeline',
    }


@pytest.mark.parametrize(
    ('method', 'path'),
    [
        ('get', '/batch/job/pipeline-job-1/status'),
        ('post', '/batch/job/pipeline-job-1/cancel'),
        ('get', '/batch/job/static-job-1/status'),
        ('post', '/batch/job/static-job-1/cancel'),
    ],
)
def test_batch_status_cancel_reject_non_batch_jobs(
    client: TestClient,
    method: str,
    path: str,
) -> None:
    _create_pipeline_job(client)
    _create_static_job(client)

    response = client.request(method, path)

    assert response.status_code == 404
    assert response.json() == {'detail': 'Job ID not found'}


def test_batch_job_cancel_queued_marks_cancelled(client: TestClient) -> None:
    _create_batch_job(client)

    response = client.post('/batch/job/batch-job-1/cancel')

    assert response.status_code == 200
    assert response.json() == {
        'state': 'cancelled',
        'progress': 0.0,
        'message': 'The job was cancelled by the user before it started.',
    }
    state = client.app.state.sv
    with state.lock:
        job = state.jobs['batch-job-1']
        assert job['cancel_requested'] is True
        assert job['finished_ts'] is not None


def test_batch_job_cancel_running_marks_cancel_requested(client: TestClient) -> None:
    _create_batch_job(client, status='running', progress=0.25)

    response = client.post('/batch/job/batch-job-1/cancel')

    assert response.status_code == 200
    assert response.json() == {
        'state': 'cancel_requested',
        'progress': 0.25,
        'message': 'Cancel requested. The job will stop at the next safe point.',
    }
    state = client.app.state.sv
    with state.lock:
        assert state.jobs['batch-job-1']['cancel_requested'] is True


def test_batch_job_files_lists_direct_artifact_files_by_name(
    client: TestClient,
) -> None:
    job_dir = _create_batch_job(client)
    job_dir.mkdir(parents=True)
    (job_dir / 'zeta.txt').write_text('zz', encoding='utf-8')
    (job_dir / 'alpha.txt').write_text('a', encoding='utf-8')
    (job_dir / 'nested').mkdir()
    (job_dir / 'nested' / 'ignored.txt').write_text('ignored', encoding='utf-8')

    response = client.get('/batch/job/batch-job-1/files')

    assert response.status_code == 200
    assert response.json() == {
        'files': [
            {'name': 'alpha.txt', 'size_bytes': 1},
            {'name': 'zeta.txt', 'size_bytes': 2},
        ]
    }


def test_batch_job_download_returns_plain_filename(client: TestClient) -> None:
    job_dir = _create_batch_job(client)
    job_dir.mkdir(parents=True)
    (job_dir / 'job_meta.json').write_text('{"ok": true}', encoding='utf-8')

    response = client.get(
        '/batch/job/batch-job-1/download',
        params={'name': 'job_meta.json'},
    )

    assert response.status_code == 200
    assert response.text == '{"ok": true}'


@pytest.mark.parametrize(
    'name',
    [
        '../job_meta.json',
        'nested/job_meta.json',
        '/tmp/job_meta.json',
    ],
)
def test_batch_job_download_rejects_path_traversal(
    client: TestClient,
    name: str,
) -> None:
    job_dir = _create_batch_job(client)
    job_dir.mkdir(parents=True)

    response = client.get('/batch/job/batch-job-1/download', params={'name': name})

    assert response.status_code == 400
    assert response.json() == {'detail': 'Invalid file name'}


@pytest.mark.parametrize(
    ('method', 'path'),
    [
        ('get', '/pipeline/job/missing/status'),
        ('post', '/pipeline/job/missing/cancel'),
    ],
)
def test_pipeline_job_routes_reject_unknown_job(
    client: TestClient,
    method: str,
    path: str,
) -> None:
    response = client.request(method, path)

    assert response.status_code == 404
    assert response.json() == {'detail': 'Job ID not found'}


def test_pipeline_job_status_returns_state_progress_message(
    client: TestClient,
) -> None:
    _create_pipeline_job(
        client,
        status='running',
        progress=0.75,
        message='Building pipeline artifacts',
    )

    response = client.get('/pipeline/job/pipeline-job-1/status')

    assert response.status_code == 200
    assert response.json() == {
        'state': 'running',
        'progress': 0.75,
        'message': 'Building pipeline artifacts',
    }


@pytest.mark.parametrize(
    ('method', 'path'),
    [
        ('get', '/pipeline/job/batch-job-1/status'),
        ('post', '/pipeline/job/batch-job-1/cancel'),
        ('get', '/pipeline/job/static-job-1/status'),
        ('post', '/pipeline/job/static-job-1/cancel'),
    ],
)
def test_pipeline_status_cancel_reject_non_pipeline_jobs(
    client: TestClient,
    method: str,
    path: str,
) -> None:
    _create_batch_job(client)
    _create_static_job(client)

    response = client.request(method, path)

    assert response.status_code == 404
    assert response.json() == {'detail': 'Job ID not found'}


def test_pipeline_job_cancel_queued_marks_cancelled(client: TestClient) -> None:
    _create_pipeline_job(client)

    response = client.post('/pipeline/job/pipeline-job-1/cancel')

    assert response.status_code == 200
    assert response.json() == {
        'state': 'cancelled',
        'progress': 0.0,
        'message': 'The job was cancelled by the user before it started.',
    }
    state = client.app.state.sv
    with state.lock:
        job = state.jobs['pipeline-job-1']
        assert job['cancel_requested'] is True
        assert job['finished_ts'] is not None


def test_pipeline_job_cancel_running_marks_cancel_requested(
    client: TestClient,
) -> None:
    _create_pipeline_job(client, status='running', progress=0.4)

    response = client.post('/pipeline/job/pipeline-job-1/cancel')

    assert response.status_code == 200
    assert response.json() == {
        'state': 'cancel_requested',
        'progress': 0.4,
        'message': 'Cancel requested. The job will stop at the next safe point.',
    }
    state = client.app.state.sv
    with state.lock:
        assert state.jobs['pipeline-job-1']['cancel_requested'] is True


@pytest.mark.parametrize(
    ('route_prefix', 'create_job', 'raw_status', 'expected_state'),
    [
        ('/batch/job/batch-job-1', _create_batch_job, 'completed', 'done'),
        ('/batch/job/batch-job-1', _create_batch_job, 'failed', 'error'),
        ('/pipeline/job/pipeline-job-1', _create_pipeline_job, 'completed', 'done'),
        ('/pipeline/job/pipeline-job-1', _create_pipeline_job, 'failed', 'error'),
        ('/statics/job/static-job-1', _create_static_job, 'completed', 'done'),
        ('/statics/job/static-job-1', _create_static_job, 'failed', 'error'),
    ],
)
def test_job_status_aliases_are_normalized(
    client: TestClient,
    route_prefix: str,
    create_job: Any,
    raw_status: str,
    expected_state: str,
) -> None:
    create_job(client, status=raw_status)

    response = client.get(f'{route_prefix}/status')

    assert response.status_code == 200
    assert response.json()['state'] == expected_state
