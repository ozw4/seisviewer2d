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
        'datum': {
            'mode': 'constant',
            'elevation_m': 0.0,
            'replacement_velocity_m_s': 2000.0,
        },
        'apply': {
            'interpolation': 'linear',
            'fill_value': 0.0,
            'max_abs_shift_ms': 250.0,
            'output_dtype': 'float32',
            'register_corrected_file': True,
        },
    }


def test_datum_static_apply_endpoint_starts_job(
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

    response = client.post('/statics/datum/apply', json=_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload['state'] == 'queued'
    assert isinstance(payload['job_id'], str)
    assert len(started) == 1
    assert started[0]['target'] is statics_router_module.run_datum_static_apply_job

    state = client.app.state.sv
    with state.lock:
        job = dict(state.jobs[payload['job_id']])

    assert job['status'] == 'queued'
    assert job['job_type'] == 'statics'
    assert job['statics_kind'] == 'datum'
    assert job['file_id'] == 'source-file-id'
    assert job['key1_byte'] == 189
    assert job['key2_byte'] == 193


@pytest.mark.parametrize(
    'mutator',
    [
        lambda payload: payload['datum'].update({'mode': 'floating'}),
        lambda payload: payload['apply'].update({'interpolation': 'nearest'}),
        lambda payload: payload['apply'].update({'output_dtype': 'float64'}),
        lambda payload: payload['apply'].update({'register_corrected_file': False}),
        lambda payload: payload['apply'].update({'max_abs_shift_ms': 0.0}),
        lambda payload: payload['datum'].update({'replacement_velocity_m_s': 0.0}),
    ],
)
def test_datum_static_apply_endpoint_rejects_unsupported_mvp_options(
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

    response = client.post('/statics/datum/apply', json=payload)

    assert response.status_code == 422
    assert started == []
