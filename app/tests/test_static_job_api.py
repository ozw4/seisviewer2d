from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.state import create_app_state
from app.main import app
from app.services.job_manager import JobManager
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


def _create_static_job(
    client: TestClient,
    *,
    job_id: str = 'static-job-1',
    artifacts_dir: Path | None = None,
    status: str | None = None,
    progress: float | None = None,
    message: str | None = None,
) -> Path:
    if artifacts_dir is None:
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
            state.jobs.set_status(job_id, status)  # type: ignore[arg-type]
        if progress is not None:
            state.jobs.set_progress(job_id, progress)
        if message is not None:
            state.jobs.set_message(job_id, message)
    return artifacts_dir


def _create_batch_job(client: TestClient, *, job_id: str = 'batch-job-1') -> None:
    state = client.app.state.sv
    with state.lock:
        state.jobs.create_batch_apply_job(
            job_id,
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir=str(get_job_dir(job_id)),
        )


def test_create_static_job_state_defaults() -> None:
    jobs = JobManager()

    job = jobs.create_static_job(
        'static-job-1',
        file_id='source-file-id',
        key1_byte=189,
        key2_byte=193,
        statics_kind='datum',
        artifacts_dir='/tmp/static-job-1',
        created_ts=0.0,
    )

    assert job == {
        'status': 'queued',
        'progress': 0.0,
        'message': '',
        'cancel_requested': False,
        'created_ts': 0.0,
        'finished_ts': None,
        'file_id': 'source-file-id',
        'key1_byte': 189,
        'key2_byte': 193,
        'artifacts_dir': '/tmp/static-job-1',
        'job_type': 'statics',
        'statics_kind': 'datum',
    }
    assert jobs['static-job-1'] is job


@pytest.mark.parametrize(
    ('field', 'overrides'),
    [
        ('file_id', {'file_id': ''}),
        ('statics_kind', {'statics_kind': ''}),
        ('artifacts_dir', {'artifacts_dir': ''}),
    ],
)
def test_create_static_job_rejects_empty_required_strings(
    field: str,
    overrides: dict[str, Any],
) -> None:
    jobs = JobManager()
    kwargs = {
        'file_id': 'file-1',
        'key1_byte': 189,
        'key2_byte': 193,
        'statics_kind': 'datum',
        'artifacts_dir': '/tmp/static-job-1',
        'created_ts': 0.0,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=field):
        jobs.create_static_job('static-job-1', **kwargs)


def test_static_job_kind_uses_pipeline_ttl_cleanup() -> None:
    state = create_app_state()
    with state.lock:
        state.jobs.create_static_job(
            'static-job-1',
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            statics_kind='datum',
            artifacts_dir='/tmp/static-job-1',
            created_ts=0.0,
        )
        state.jobs.mark_done('static-job-1', finished_ts=0.0)

    ttl_sec = state.settings.pipeline_jobs_ttl_hours * 3600
    with state.lock:
        state.jobs.cleanup_in_memory(
            now_ts=float(ttl_sec + 1),
            settings=state.settings,
        )

    assert 'static-job-1' not in state.jobs


def test_static_job_cleanup_keeps_running_jobs() -> None:
    state = create_app_state()
    with state.lock:
        state.jobs.create_static_job(
            'static-job-1',
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            statics_kind='datum',
            artifacts_dir='/tmp/static-job-1',
            created_ts=0.0,
        )
        state.jobs.set_status('static-job-1', 'running')

    ttl_sec = state.settings.pipeline_jobs_ttl_hours * 3600
    with state.lock:
        state.jobs.cleanup_in_memory(
            now_ts=float(ttl_sec + 1),
            settings=state.settings,
        )

    assert 'static-job-1' in state.jobs


def test_static_job_status_returns_state_progress_message(client: TestClient) -> None:
    _create_static_job(
        client,
        status='running',
        progress=0.5,
        message='Applying datum statics',
    )

    response = client.get('/statics/job/static-job-1/status')

    assert response.status_code == 200
    assert response.json() == {
        'state': 'running',
        'progress': 0.5,
        'message': 'Applying datum statics',
    }


@pytest.mark.parametrize(
    ('method', 'path', 'params'),
    [
        ('get', '/statics/job/missing/status', None),
        ('post', '/statics/job/missing/cancel', None),
        ('get', '/statics/job/missing/files', None),
        ('get', '/statics/job/missing/download', {'name': 'job_meta.json'}),
    ],
)
def test_static_job_routes_reject_unknown_job(
    client: TestClient,
    method: str,
    path: str,
    params: dict[str, str] | None,
) -> None:
    response = client.request(method, path, params=params)

    assert response.status_code == 404
    assert response.json() == {'detail': 'Job ID not found'}


@pytest.mark.parametrize(
    ('method', 'suffix', 'params'),
    [
        ('get', 'status', None),
        ('post', 'cancel', None),
        ('get', 'files', None),
        ('get', 'download', {'name': 'job_meta.json'}),
    ],
)
def test_static_job_routes_reject_non_static_job(
    client: TestClient,
    method: str,
    suffix: str,
    params: dict[str, str] | None,
) -> None:
    _create_batch_job(client, job_id='batch-job-1')

    response = client.request(
        method,
        f'/statics/job/batch-job-1/{suffix}',
        params=params,
    )

    assert response.status_code == 404
    assert response.json() == {'detail': 'Job ID not found'}


def test_static_job_cancel_queued_marks_cancelled(client: TestClient) -> None:
    _create_static_job(client)

    response = client.post('/statics/job/static-job-1/cancel')

    assert response.status_code == 200
    payload = response.json()
    assert payload['state'] == 'cancelled'
    assert payload['message'] == 'The job was cancelled by the user before it started.'
    state = client.app.state.sv
    with state.lock:
        job = state.jobs['static-job-1']
        assert job['cancel_requested'] is True
        assert job['finished_ts'] is not None


def test_static_job_cancel_running_marks_cancel_requested(client: TestClient) -> None:
    _create_static_job(client, status='running', progress=0.25)

    response = client.post('/statics/job/static-job-1/cancel')

    assert response.status_code == 200
    assert response.json() == {
        'state': 'cancel_requested',
        'progress': 0.25,
        'message': 'Cancel requested. The job will stop at the next safe point.',
    }
    state = client.app.state.sv
    with state.lock:
        job = state.jobs['static-job-1']
        assert job['cancel_requested'] is True


def test_static_job_files_lists_artifact_files(client: TestClient) -> None:
    job_dir = _create_static_job(client)
    job_dir.mkdir(parents=True)
    (job_dir / 'datum_static_qc.json').write_text('{"ok": true}', encoding='utf-8')
    (job_dir / 'datum_statics.csv').write_text('a,b\n1,2\n', encoding='utf-8')
    (job_dir / 'nested').mkdir()

    response = client.get('/statics/job/static-job-1/files')

    assert response.status_code == 200
    assert response.json() == {
        'files': [
            {'name': 'datum_static_qc.json', 'size_bytes': 12},
            {'name': 'datum_statics.csv', 'size_bytes': 8},
        ]
    }


def test_static_job_files_returns_empty_list_for_empty_dir(
    client: TestClient,
) -> None:
    job_dir = _create_static_job(client)
    job_dir.mkdir(parents=True)

    response = client.get('/statics/job/static-job-1/files')

    assert response.status_code == 200
    assert response.json() == {'files': []}


def test_static_job_files_missing_dir_404(client: TestClient) -> None:
    _create_static_job(client)

    response = client.get('/statics/job/static-job-1/files')

    assert response.status_code == 404
    assert response.json() == {'detail': 'Job artifacts not found'}


def test_static_job_download_returns_file(client: TestClient) -> None:
    job_dir = _create_static_job(client)
    job_dir.mkdir(parents=True)
    (job_dir / 'datum_static_qc.json').write_text('{"ok": true}', encoding='utf-8')

    response = client.get(
        '/statics/job/static-job-1/download',
        params={'name': 'datum_static_qc.json'},
    )

    assert response.status_code == 200
    assert response.text == '{"ok": true}'


@pytest.mark.parametrize(
    'name',
    [
        '',
        '../datum_static_qc.json',
        'nested/file.json',
        '/path/to/file.json',
    ],
)
def test_static_job_download_rejects_invalid_name(
    client: TestClient,
    name: str,
) -> None:
    job_dir = _create_static_job(client)
    job_dir.mkdir(parents=True)

    response = client.get(
        '/statics/job/static-job-1/download',
        params={'name': name},
    )

    assert response.status_code == 400
    assert response.json() == {'detail': 'Invalid file name'}


def test_static_job_download_missing_dir_404(client: TestClient) -> None:
    _create_static_job(client)

    response = client.get(
        '/statics/job/static-job-1/download',
        params={'name': 'datum_static_qc.json'},
    )

    assert response.status_code == 404
    assert response.json() == {'detail': 'Job artifacts not found'}


def test_static_job_download_missing_file_404(client: TestClient) -> None:
    job_dir = _create_static_job(client)
    job_dir.mkdir(parents=True)

    response = client.get(
        '/statics/job/static-job-1/download',
        params={'name': 'datum_static_qc.json'},
    )

    assert response.status_code == 404
    assert response.json() == {'detail': 'File not found'}


def test_statics_router_registered() -> None:
    paths = {getattr(route, 'path', None) for route in app.routes}

    assert '/statics/job/{job_id}/status' in paths
    assert '/statics/job/{job_id}/cancel' in paths
    assert '/statics/job/{job_id}/files' in paths
    assert '/statics/job/{job_id}/download' in paths
    assert '/statics/datum/apply' not in paths
