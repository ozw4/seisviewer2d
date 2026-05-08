from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
import app.services.refraction_static_service as refraction_service_module
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
            'max_weathering_thickness_m': None,
        },
        'moveout': {
            'model': 'head_wave_linear_offset',
            'distance_source': 'geometry',
            'offset_byte': 37,
            'min_offset_m': None,
            'max_offset_m': None,
            'allow_missing_offset': False,
            'max_geometry_offset_mismatch_m': None,
        },
        'solver': {
            'damping': 0.01,
            'min_picks_per_node': 1,
            'max_abs_half_intercept_time_ms': 500.0,
            'robust': {
                'enabled': True,
                'method': 'mad',
                'threshold': 3.5,
                'max_iterations': 5,
                'min_used_fraction': 0.5,
                'min_used_observations': 1,
            },
        },
        'datum': {
            'mode': 'floating_and_flat',
            'floating_datum_mode': 'constant',
            'flat_datum_elevation_m': 200.0,
            'floating_datum_elevation_m': 100.0,
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


def test_run_refraction_static_apply_job_writes_artifacts_and_completes(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    replacements: list[dict[str, Any]] = []
    datum_builds: list[dict[str, Any]] = []
    artifact_writes: list[dict[str, Any]] = []
    replacement_result = object()
    datum_result = object()

    def _capture_replacement(**kwargs: Any) -> object:
        replacements.append(kwargs)
        return replacement_result

    def _capture_datum_build(**kwargs: Any) -> object:
        datum_builds.append(kwargs)
        return datum_result

    def _capture_artifact_write(**kwargs: Any) -> object:
        artifact_writes.append(kwargs)
        return object()

    monkeypatch.setattr(
        refraction_service_module,
        'compute_weathering_replacement_statics_from_first_breaks',
        _capture_replacement,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_datum_statics',
        _capture_datum_build,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'write_refraction_static_artifacts',
        _capture_artifact_write,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    request_json = job_dir / 'refraction_static_request.json'
    assert request_json.is_file()
    request_payload = json.loads(request_json.read_text(encoding='utf-8'))
    assert request_payload['job_id'] == job_id
    assert request_payload['statics_kind'] == 'refraction'
    assert request_payload['request']['file_id'] == 'raw-file-id'
    assert len(replacements) == 1
    assert replacements[0]['req'] is req
    assert replacements[0]['job_dir'] == job_dir
    assert len(datum_builds) == 1
    assert datum_builds[0]['weathering_replacement_result'] is replacement_result
    assert datum_builds[0]['datum'] is req.datum
    assert datum_builds[0]['apply_options'] is req.apply
    assert datum_builds[0]['job_dir'] == job_dir
    assert datum_builds[0]['state'] is client.app.state.sv
    assert datum_builds[0]['file_id'] == req.file_id
    assert datum_builds[0]['key1_byte'] == req.key1_byte
    assert datum_builds[0]['key2_byte'] == req.key2_byte
    assert len(artifact_writes) == 1
    assert artifact_writes[0]['result'] is datum_result
    assert artifact_writes[0]['req'] is req
    assert artifact_writes[0]['job_dir'] == job_dir
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    assert job['progress'] == pytest.approx(1.0)
    assert job['message'] == 'refraction_static_artifacts_written_artifact_only'


def test_run_refraction_static_apply_job_registers_corrected_file_when_requested(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    payload = _payload()
    payload['apply']['register_corrected_file'] = True
    req = RefractionStaticApplyRequest.model_validate(payload)
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            job_id,
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            statics_kind='refraction',
            artifacts_dir=str(job_dir),
        )
    apply_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        refraction_service_module,
        'compute_weathering_replacement_statics_from_first_breaks',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_datum_statics',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'write_refraction_static_artifacts',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'apply_refraction_statics_to_trace_store',
        lambda **kwargs: (
            apply_calls.append(kwargs)
            or SimpleNamespace(
                corrected_file_id='corrected-refraction-file-id',
                corrected_trace_store_path=job_dir / 'corrected-store',
            )
        ),
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert len(apply_calls) == 1
    assert apply_calls[0]['req'] is req
    assert apply_calls[0]['state'] is client.app.state.sv
    assert apply_calls[0]['job_id'] == job_id
    assert apply_calls[0]['job_dir'] == job_dir
    assert job['status'] == 'done'
    assert job['corrected_file_id'] == 'corrected-refraction-file-id'
    assert job['corrected_store_path'] == str(job_dir / 'corrected-store')
