from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
from app.main import app


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


def _payload() -> dict[str, Any]:
    return {
        'file_id': 'source-file-id',
        'key1_byte': 189,
        'key2_byte': 193,
        'datum_solution': {
            'job_id': 'datum-static-job-id',
            'name': 'datum_static_solution.npz',
        },
        'pick_source': {
            'kind': 'batch_job_artifact',
            'job_id': 'batch-apply-job-id',
            'name': 'predicted_picks_time_s.npz',
        },
        'offset': {'offset_byte': 37},
        'qc': {'require_linear_offset_model': False},
    }


def test_first_break_qc_endpoint_starts_static_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []

    def _capture_start_job_thread(**kwargs: Any) -> object:
        started.append(kwargs)
        return object()

    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        _capture_start_job_thread,
    )

    response = client.post('/statics/first-break/qc', json=_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload['state'] == 'queued'
    assert isinstance(payload['job_id'], str)
    assert len(started) == 1
    assert started[0]['target'] is statics_router_module.run_first_break_qc_job

    state = client.app.state.sv
    with state.lock:
        job = dict(state.jobs[payload['job_id']])

    assert job['status'] == 'queued'
    assert job['job_type'] == 'statics'
    assert job['statics_kind'] == 'first_break_qc'
    assert job['file_id'] == 'source-file-id'
    assert job['key1_byte'] == 189
    assert job['key2_byte'] == 193


@pytest.mark.parametrize(
    'mutator',
    [
        lambda payload: payload.update({'file_id': ''}),
        lambda payload: payload.update({'key1_byte': 0}),
        lambda payload: payload.update({'key2_byte': 0}),
        lambda payload: payload['pick_source'].update({'kind': 'unknown'}),
        lambda payload: payload['pick_source'].pop('job_id'),
        lambda payload: payload['pick_source'].pop('name'),
        lambda payload: payload.update(
            {'pick_source': {'kind': 'manual_memmap', 'job_id': 'job-a'}}
        ),
        lambda payload: payload.update(
            {'pick_source': {'kind': 'manual_memmap', 'name': 'manual.npz'}}
        ),
        lambda payload: payload['datum_solution'].update({'name': '../x.npz'}),
        lambda payload: payload['datum_solution'].update({'name': 'nested/x.npz'}),
        lambda payload: payload['datum_solution'].update({'name': '/tmp/x.npz'}),
        lambda payload: payload['pick_source'].update({'name': '../x.npz'}),
        lambda payload: payload['pick_source'].update({'name': 'nested/x.npz'}),
        lambda payload: payload['pick_source'].update({'name': '/tmp/x.npz'}),
        lambda payload: payload.update({'offset': {'offset_byte': 0}}),
    ],
)
def test_first_break_qc_endpoint_rejects_invalid_request(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    mutator,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    payload = _payload()
    mutator(payload)

    response = client.post('/statics/first-break/qc', json=payload)

    assert response.status_code == 422
    assert started == []


def test_first_break_qc_job_files_and_download_use_static_lifecycle(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'first-break-qc-job'
    job_dir.mkdir()
    (job_dir / 'job_meta.json').write_text('{"job_id":"qc-job"}', encoding='utf-8')
    (job_dir / 'first_break_qc.json').write_text('{"ok":true}', encoding='utf-8')
    (job_dir / 'first_break_qc.csv').write_text('a,b\n', encoding='utf-8')
    (job_dir / 'residual_by_key1.csv').write_text('key1\n', encoding='utf-8')
    state = client.app.state.sv
    with state.lock:
        state.jobs.create_static_job(
            'qc-job',
            file_id='source-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='first_break_qc',
            artifacts_dir=str(job_dir),
        )
        state.jobs.mark_done('qc-job')

    files_response = client.get('/statics/job/qc-job/files')
    download_response = client.get(
        '/statics/job/qc-job/download',
        params={'name': 'first_break_qc.json'},
    )

    assert files_response.status_code == 200
    assert files_response.json() == {
        'files': [
            {'name': 'first_break_qc.csv', 'size_bytes': 4},
            {'name': 'first_break_qc.json', 'size_bytes': 11},
            {'name': 'job_meta.json', 'size_bytes': 19},
            {'name': 'residual_by_key1.csv', 'size_bytes': 5},
        ]
    }
    assert download_response.status_code == 200
    assert download_response.text == '{"ok":true}'
