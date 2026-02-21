from __future__ import annotations

import threading
import time
from typing import Any
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.routers import batch_apply as batch_apply_router_module
from app.main import app
from app.services import batch_apply_service


class CapturedThread:
    instances: list['CapturedThread'] = []

    def __init__(self, *init_args: Any, **init_kwargs: Any):
        del init_args
        self.target = init_kwargs.get('target')
        self.args = init_kwargs.get('args', ())
        kwargs = init_kwargs.get('kwargs')
        self.kwargs = {} if kwargs is None else kwargs
        self.daemon = init_kwargs.get('daemon')
        CapturedThread.instances.append(self)

    def start(self) -> None:
        return None


def _wait_for_status(
    client: TestClient,
    job_id: str,
    expected_state: str,
    *,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_payload: dict[str, Any] = {}
    while time.time() < deadline:
        resp = client.get(f'/batch/job/{job_id}/status')
        assert resp.status_code == 200
        payload = resp.json()
        last_payload = payload
        if payload.get('state') == expected_state:
            return payload
        time.sleep(0.01)
    raise AssertionError(
        f'timeout waiting for state={expected_state}; last_payload={last_payload}'
    )


@pytest.fixture(autouse=True)
def _clear_jobs_table() -> None:
    state = app.state.sv
    with state.lock:
        state.jobs.clear()
    yield
    with state.lock:
        state.jobs.clear()


def test_batch_job_status_unknown_job_id() -> None:
    with TestClient(app) as client:
        resp = client.get('/batch/job/missing/status')

    assert resp.status_code == 404
    assert resp.json() == {'detail': 'Job ID not found'}


def test_batch_apply_lifecycle_files_download_and_path_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(tmp_path))
    CapturedThread.instances.clear()
    monkeypatch.setattr(
        batch_apply_router_module,
        'threading',
        SimpleNamespace(Thread=CapturedThread),
    )

    orig_write_job_meta = batch_apply_service._write_job_meta
    started = threading.Event()
    proceed = threading.Event()

    def _gated_write_job_meta(*, job_dir, payload):
        started.set()
        if not proceed.wait(timeout=5.0):
            raise RuntimeError('test gate timeout')
        return orig_write_job_meta(job_dir=job_dir, payload=payload)

    monkeypatch.setattr(batch_apply_service, '_write_job_meta', _gated_write_job_meta)

    with TestClient(app) as client:
        create_resp = client.post(
            '/batch/apply',
            json={
                'file_id': 'file-a',
                'key1_byte': 189,
                'key2_byte': 193,
                'pipeline_spec': {'steps': []},
            },
        )
        assert create_resp.status_code == 200
        create_payload = create_resp.json()
        job_id = create_payload['job_id']
        assert create_payload['state'] == 'queued'

        queued_resp = client.get(f'/batch/job/{job_id}/status')
        assert queued_resp.status_code == 200
        assert queued_resp.json()['state'] == 'queued'

        assert len(CapturedThread.instances) == 1
        captured = CapturedThread.instances[0]
        worker = threading.Thread(
            target=captured.target,
            args=captured.args,
            kwargs=captured.kwargs,
            daemon=True,
        )
        worker.start()

        assert started.wait(timeout=5.0)
        running_payload = _wait_for_status(client, job_id, 'running')
        assert running_payload['progress'] == 0.0

        proceed.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()

        done_payload = _wait_for_status(client, job_id, 'done')
        assert done_payload['progress'] == 1.0
        assert done_payload['message'] == ''

        files_resp = client.get(f'/batch/job/{job_id}/files')
        assert files_resp.status_code == 200
        files_payload = files_resp.json()
        job_meta_entry = [
            item for item in files_payload['files'] if item['name'] == 'job_meta.json'
        ]
        assert len(job_meta_entry) == 1
        assert job_meta_entry[0]['size_bytes'] > 0

        download_resp = client.get(
            f'/batch/job/{job_id}/download', params={'name': 'job_meta.json'}
        )
        assert download_resp.status_code == 200
        assert len(download_resp.content) > 0

        traversal_resp = client.get(
            f'/batch/job/{job_id}/download', params={'name': '../job_meta.json'}
        )
        assert traversal_resp.status_code == 400
