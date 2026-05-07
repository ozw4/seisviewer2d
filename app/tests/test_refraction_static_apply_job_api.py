from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
from app.api.schemas import RefractionStaticApplyRequest
from app.main import app
from app.services.refraction_static_service import run_refraction_static_apply_job


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
        'file_id': 'raw-file-id',
        'key1_byte': 189,
        'key2_byte': 193,
        'pick_source': {
            'kind': 'batch_predicted_npz',
            'job_id': 'first-break-job-id',
            'artifact_name': 'predicted_picks_time_s.npz',
        },
        'geometry': {
            'source_id_byte': 9,
            'receiver_id_byte': 13,
            'source_x_byte': 73,
            'source_y_byte': 77,
            'receiver_x_byte': 81,
            'receiver_y_byte': 85,
            'source_elevation_byte': 45,
            'receiver_elevation_byte': 41,
            'source_depth_byte': None,
            'coordinate_scalar_byte': 71,
            'elevation_scalar_byte': 69,
            'coordinate_unit': 'm',
            'elevation_unit': 'm',
        },
        'linkage': {
            'mode': 'required',
            'job_id': 'linkage-job-id',
            'artifact_name': 'geometry_linkage.npz',
        },
        'model': {
            'method': 'gli_variable_thickness',
            'weathering_velocity_m_s': 800.0,
            'bedrock_velocity_mode': 'solve_global',
            'bedrock_velocity_m_s': None,
            'initial_bedrock_velocity_m_s': 2500.0,
            'min_bedrock_velocity_m_s': 1200.0,
            'max_bedrock_velocity_m_s': 6000.0,
        },
        'offset': {
            'distance_source': 'geometry',
            'offset_byte': 37,
            'min_offset_m': None,
            'max_offset_m': None,
            'allow_missing_offset': False,
            'max_geometry_offset_mismatch_m': None,
        },
        'solver': {
            'damping': 0.01,
            'robust': {
                'enabled': True,
                'method': 'mad',
                'threshold': 3.5,
                'max_iterations': 5,
                'min_used_fraction': 0.5,
                'min_used_observations': 1,
            },
        },
        'apply': {
            'mode': 'refraction_from_raw',
            'interpolation': 'linear',
            'fill_value': 0.0,
            'max_abs_shift_ms': 250.0,
            'output_dtype': 'float32',
            'register_corrected_file': False,
        },
    }


def test_refraction_static_apply_endpoint_starts_job(
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

    response = client.post('/statics/refraction/apply', json=_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload['state'] == 'queued'
    assert isinstance(payload['job_id'], str)
    assert len(started) == 1
    assert started[0]['target'] is statics_router_module.run_refraction_static_apply_job
    assert started[0]['args'][0] == payload['job_id']
    assert isinstance(started[0]['args'][1], RefractionStaticApplyRequest)
    assert started[0]['args'][2] is client.app.state.sv

    state = client.app.state.sv
    with state.lock:
        job = dict(state.jobs[payload['job_id']])

    assert job['status'] == 'queued'
    assert job['job_type'] == 'statics'
    assert job['statics_kind'] == 'refraction'
    assert job['file_id'] == 'raw-file-id'
    assert job['key1_byte'] == 189
    assert job['key2_byte'] == 193


def test_refraction_static_apply_endpoint_rejects_invalid_schema_without_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    payload = _payload()
    payload['file_id'] = ''

    response = client.post('/statics/refraction/apply', json=payload)

    assert response.status_code == 422
    assert started == []
    with client.app.state.sv.lock:
        assert len(client.app.state.sv.jobs) == 0


def test_run_refraction_static_apply_job_writes_request_and_fails(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_id = 'refraction-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    req = RefractionStaticApplyRequest.model_validate(_payload())
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            job_id,
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            statics_kind='refraction',
            artifacts_dir=str(job_dir),
        )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    request_json = job_dir / 'refraction_static_request.json'
    assert request_json.is_file()
    request_payload = json.loads(request_json.read_text(encoding='utf-8'))
    assert request_payload['job_id'] == job_id
    assert request_payload['statics_kind'] == 'refraction'
    assert request_payload['request']['file_id'] == 'raw-file-id'
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'error'
    assert job['message'] == 'Refraction statics solver is not implemented yet.'
