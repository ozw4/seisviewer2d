from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
import app.services.refraction_static_service as refraction_service_module
from app.api.schemas import RefractionStaticApplyRequest
from app.main import app
from app.services.refraction_static_apply_trace_store import (
    CORRECTED_FILE_JSON_NAME,
    REFRACTION_STATIC_APPLY_QC_JSON_NAME,
)
from app.services.refraction_static_artifacts import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    REFRACTION_STATICS_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)
from app.services.refraction_static_service import run_refraction_static_apply_job
from app.services.refraction_static_v1 import RefractionV1EstimateResult
from app.services.trace_store_registration import trace_store_cache_key
from app.tests.test_refraction_static_apply_trace_store import (
    DT,
    KEY1,
    KEY2,
    SOURCE_FILE_ID,
    _valid_result,
    _write_source_store,
)
from app.tests.test_refraction_static_artifacts import _result as _artifact_result


FINAL_REFRACTION_ARTIFACTS = {
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATICS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
}
REQUEST_JSON_NAME = 'refraction_static_request.json'


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
    replacement_resolved = replacements[0]['resolved_first_layer']
    assert replacement_resolved.mode == 'constant'
    assert replacement_resolved.weathering_velocity_m_s == pytest.approx(800.0)
    assert len(datum_builds) == 1
    assert datum_builds[0]['weathering_replacement_result'] is replacement_result
    assert datum_builds[0]['datum'] is req.datum
    assert datum_builds[0]['apply_options'] is req.apply
    assert datum_builds[0]['job_dir'] == job_dir
    assert datum_builds[0]['state'] is client.app.state.sv
    assert datum_builds[0]['file_id'] == req.file_id
    assert datum_builds[0]['key1_byte'] == req.key1_byte
    assert datum_builds[0]['key2_byte'] == req.key2_byte
    assert datum_builds[0]['resolved_first_layer'] is replacement_resolved
    assert len(artifact_writes) == 1
    assert artifact_writes[0]['result'] is datum_result
    assert artifact_writes[0]['req'] is req
    assert artifact_writes[0]['job_dir'] == job_dir
    resolved_first_layer = artifact_writes[0]['resolved_first_layer']
    assert resolved_first_layer.mode == 'constant'
    assert resolved_first_layer.weathering_velocity_m_s == pytest.approx(800.0)
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    assert job['progress'] == pytest.approx(1.0)
    assert job['message'] == 'refraction_static_artifacts_written_artifact_only'


def test_run_refraction_static_apply_job_estimates_v1_before_downstream_work(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-v1-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    payload = _payload()
    payload['model']['weathering_velocity_m_s'] = None
    payload['model']['first_layer'] = {
        'mode': 'estimate_direct_arrival',
        'min_weathering_velocity_m_s': 500.0,
        'max_weathering_velocity_m_s': 1200.0,
        'min_direct_offset_m': 20.0,
        'max_direct_offset_m': 140.0,
        'min_picks_per_fit': 5,
        'min_groups': 3,
        'robust_enabled': True,
        'robust_threshold': 3.5,
    }
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

    input_model = object()
    replacement_result = object()
    datum_result = object()
    replacement_calls: list[dict[str, Any]] = []
    artifact_calls: list[dict[str, Any]] = []
    v1_write_calls: list[dict[str, Any]] = []
    estimate = RefractionV1EstimateResult(
        mode='estimate_direct_arrival',
        resolved_weathering_velocity_m_s=812.5,
        group_kind='source_endpoint',
        group_key=np.asarray(['source:1']),
        group_v1_m_s=np.asarray([812.5]),
        group_slope_s_per_m=np.asarray([1.0 / 812.5]),
        group_intercept_s=np.asarray([0.01]),
        group_n_candidates=np.asarray([6]),
        group_n_used=np.asarray([6]),
        group_offset_min_m=np.asarray([20.0]),
        group_offset_max_m=np.asarray([120.0]),
        group_residual_rms_s=np.asarray([0.0]),
        group_residual_mad_s=np.asarray([0.0]),
        group_status=np.asarray(['ok']),
        qc={
            'v1_mode': 'estimate_direct_arrival',
            'resolved_weathering_velocity_m_s': 812.5,
            'v1_status': 'estimated',
        },
    )

    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_static_input_model',
        lambda **_kwargs: input_model,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'estimate_global_v1_from_direct_arrivals',
        lambda **_kwargs: estimate,
    )

    def _capture_v1_write(*args: Any, **kwargs: Any) -> object:
        v1_write_calls.append({'args': args, 'kwargs': kwargs})
        return object()

    def _capture_replacement(**kwargs: Any) -> object:
        replacement_calls.append(kwargs)
        return replacement_result

    def _capture_artifact_write(**kwargs: Any) -> object:
        artifact_calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        refraction_service_module,
        'write_refraction_v1_artifacts',
        _capture_v1_write,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'compute_weathering_replacement_statics_from_first_breaks',
        _capture_replacement,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_datum_statics',
        lambda **_kwargs: datum_result,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'write_refraction_static_artifacts',
        _capture_artifact_write,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    assert len(v1_write_calls) == 1
    assert v1_write_calls[0]['args'] == (job_dir, estimate)
    assert len(replacement_calls) == 1
    resolved_req = replacement_calls[0]['req']
    assert resolved_req is not req
    assert replacement_calls[0]['input_model'] is input_model
    assert resolved_req.model.first_layer_mode == 'estimate_direct_arrival'
    assert resolved_req.model.resolved_weathering_velocity_m_s == pytest.approx(812.5)
    replacement_resolved = replacement_calls[0]['resolved_first_layer']
    assert replacement_resolved.mode == 'estimate_direct_arrival'
    assert replacement_resolved.weathering_velocity_m_s == pytest.approx(812.5)
    assert len(artifact_calls) == 1
    assert artifact_calls[0]['req'] is resolved_req
    resolved_first_layer = artifact_calls[0]['resolved_first_layer']
    assert resolved_first_layer is replacement_resolved
    assert resolved_first_layer.mode == 'estimate_direct_arrival'
    assert resolved_first_layer.weathering_velocity_m_s == pytest.approx(812.5)


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


def test_run_refraction_static_apply_job_artifact_only_writes_real_final_artifacts(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-artifact-only-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    req = RefractionStaticApplyRequest.model_validate(_payload())
    datum_result = _artifact_result()
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    _stub_upstream_refraction_result(monkeypatch, datum_result=datum_result)

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
        registry_count = len(client.app.state.sv.file_registry.records)
    assert job['status'] == 'done'
    assert job['message'] == 'refraction_static_artifacts_written_artifact_only'
    assert 'corrected_file_id' not in job
    assert registry_count == 0
    assert FINAL_REFRACTION_ARTIFACTS.issubset(_job_file_names(job_dir))
    assert CORRECTED_FILE_JSON_NAME not in _job_file_names(job_dir)
    assert REFRACTION_STATIC_APPLY_QC_JSON_NAME not in _job_file_names(job_dir)
    assert not _segy_output_files(job_dir)

    with np.load(job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        assert data['sorted_trace_index'].shape == (4,)
        assert data['trace_static_status_sorted'].tolist() == [
            'ok',
            'ok',
            'not_observed',
            'ok',
        ]
        assert data['node_solution_status'].tolist() == [
            'solved',
            'solved',
            'inactive',
        ]
        assert data['node_weathering_status'].tolist() == [
            'ok',
            'zero_thickness',
            'inactive',
        ]
        for key in data.files:
            assert data[key].dtype != object

    qc = json.loads((job_dir / REFRACTION_STATIC_QC_JSON_NAME).read_text(encoding='utf-8'))
    assert qc['workflow'] == 'refraction_statics'
    assert qc['method'] == 'gli_variable_thickness'
    assert qc['sign_convention'] == 'corrected(t) = raw(t - shift_s)'
    assert qc['status_counts']['node_solution_status']['solved'] == 2
    assert qc['status_counts']['node_weathering_status']['zero_thickness'] == 1
    assert qc['status_counts']['trace_static_status']['not_observed'] == 1
    assert {item['name'] for item in qc['artifacts']} == (
        FINAL_REFRACTION_ARTIFACTS - {REFRACTION_STATIC_ARTIFACTS_JSON_NAME}
    )

    assert len(_read_csv(job_dir / REFRACTION_STATICS_CSV_NAME)) == 4
    assert len(_read_csv(job_dir / NEAR_SURFACE_MODEL_CSV_NAME)) == 3
    assert len(_read_csv(job_dir / FIRST_BREAK_RESIDUALS_CSV_NAME)) == 3
    assert len(_read_csv(job_dir / REFRACTION_STATIC_COMPONENTS_CSV_NAME)) == 4

    files_response = client.get(f'/statics/job/{job_id}/files')
    assert files_response.status_code == 200
    listed = {item['name'] for item in files_response.json()['files']}
    assert FINAL_REFRACTION_ARTIFACTS.issubset(listed)
    assert REQUEST_JSON_NAME in listed
    assert CORRECTED_FILE_JSON_NAME not in listed

    download = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': REFRACTION_STATIC_QC_JSON_NAME},
    )
    assert download.status_code == 200
    assert download.json()['status_counts']['node_weathering_status'][
        'zero_thickness'
    ] == 1


def test_run_refraction_static_apply_job_writes_real_artifacts_and_corrected_store(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-corrected-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    source_store = tmp_path / 'trace_stores' / 'line001.sgy'
    source_traces = _write_source_store(source_store)
    state = client.app.state.sv
    state.file_registry.update(SOURCE_FILE_ID, store_path=str(source_store), dt=DT)
    payload = _payload()
    payload['apply']['register_corrected_file'] = True
    req = RefractionStaticApplyRequest.model_validate(payload)
    datum_result = _valid_result()
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    _stub_upstream_refraction_result(monkeypatch, datum_result=datum_result)

    run_refraction_static_apply_job(job_id, req, state)

    with state.lock:
        job = dict(state.jobs[job_id])
    corrected_file_id = job.get('corrected_file_id')
    assert job['status'] == 'done'
    assert isinstance(corrected_file_id, str)
    assert corrected_file_id != SOURCE_FILE_ID
    corrected_store = Path(str(job['corrected_store_path']))
    assert state.file_registry.get_store_path(corrected_file_id) == str(corrected_store)
    with state.lock:
        assert trace_store_cache_key(corrected_file_id, KEY1, KEY2) in state.cached_readers

    assert FINAL_REFRACTION_ARTIFACTS.issubset(_job_file_names(job_dir))
    assert CORRECTED_FILE_JSON_NAME in _job_file_names(job_dir)
    assert REFRACTION_STATIC_APPLY_QC_JSON_NAME in _job_file_names(job_dir)
    assert not _segy_output_files(job_dir)
    assert not _segy_output_files(corrected_store)

    corrected = np.load(corrected_store / 'traces.npy')
    assert [int(np.argmax(corrected[index])) for index in range(3)] == [8, 10, 6]
    np.testing.assert_array_equal(np.load(source_store / 'traces.npy'), source_traces)

    corrected_manifest = json.loads(
        (job_dir / CORRECTED_FILE_JSON_NAME).read_text(encoding='utf-8')
    )
    assert corrected_manifest['derived_from_file_id'] == SOURCE_FILE_ID
    assert corrected_manifest['derivation'] == 'refraction_static_correction'
    assert corrected_manifest['statics_kind'] == 'refraction'
    assert corrected_manifest['sign_convention'] == 'corrected(t) = raw(t - shift_s)'
    assert corrected_manifest['solution_artifact'] == REFRACTION_STATIC_SOLUTION_NPZ_NAME

    apply_qc = json.loads(
        (job_dir / REFRACTION_STATIC_APPLY_QC_JSON_NAME).read_text(encoding='utf-8')
    )
    assert apply_qc['corrected_file_id'] == corrected_file_id
    assert apply_qc['n_positive_trace_shifts'] == 1
    assert apply_qc['n_negative_trace_shifts'] == 1

    meta = json.loads((corrected_store / 'meta.json').read_text(encoding='utf-8'))
    assert meta['derived']['from_file_id'] == SOURCE_FILE_ID
    assert meta['derived']['derivation'] == 'refraction_static_correction'
    assert meta['derived']['solution_artifact'] == REFRACTION_STATIC_SOLUTION_NPZ_NAME

    files_response = client.get(f'/statics/job/{job_id}/files')
    assert files_response.status_code == 200
    listed = {item['name'] for item in files_response.json()['files']}
    assert FINAL_REFRACTION_ARTIFACTS.issubset(listed)
    assert CORRECTED_FILE_JSON_NAME in listed
    assert REFRACTION_STATIC_APPLY_QC_JSON_NAME in listed

    download = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': CORRECTED_FILE_JSON_NAME},
    )
    assert download.status_code == 200
    assert download.json()['corrected_file_id'] == corrected_file_id


def test_run_refraction_static_apply_job_apply_failure_keeps_final_artifacts(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-apply-failure-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    source_store = tmp_path / 'trace_stores' / 'line001.sgy'
    _write_source_store(source_store)
    state = client.app.state.sv
    state.file_registry.update(SOURCE_FILE_ID, store_path=str(source_store), dt=DT)
    payload = _payload()
    payload['apply']['register_corrected_file'] = True
    req = RefractionStaticApplyRequest.model_validate(payload)
    invalid_result = _valid_result(
        shifts=np.asarray([0.0, 0.008, np.nan, 0.0], dtype=np.float64),
        valid_mask=np.asarray([True, True, False, True], dtype=bool),
        statuses=np.asarray(['ok', 'ok', 'not_observed', 'ok'], dtype='<U16'),
    )
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    _stub_upstream_refraction_result(monkeypatch, datum_result=invalid_result)

    run_refraction_static_apply_job(job_id, req, state)

    with state.lock:
        job = dict(state.jobs[job_id])
        registry_ids = set(state.file_registry.records)
    assert job['status'] == 'error'
    assert 'invalid_trace_shift_count=1' in str(job['message'])
    assert 'not_observed' in str(job['message'])
    assert 'corrected_file_id' not in job
    assert registry_ids == {SOURCE_FILE_ID}
    assert FINAL_REFRACTION_ARTIFACTS.issubset(_job_file_names(job_dir))
    assert CORRECTED_FILE_JSON_NAME not in _job_file_names(job_dir)
    assert REFRACTION_STATIC_APPLY_QC_JSON_NAME not in _job_file_names(job_dir)
    assert list(source_store.parent.glob(f'line001.sgy.statics.refraction.{job_id}*')) == []

    files_response = client.get(f'/statics/job/{job_id}/files')
    assert files_response.status_code == 200
    listed = {item['name'] for item in files_response.json()['files']}
    assert FINAL_REFRACTION_ARTIFACTS.issubset(listed)
    assert CORRECTED_FILE_JSON_NAME not in listed


def _create_refraction_job(
    client: TestClient,
    *,
    job_id: str,
    req: RefractionStaticApplyRequest,
    job_dir: Path,
) -> None:
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            job_id,
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            statics_kind='refraction',
            artifacts_dir=str(job_dir),
        )


def _stub_upstream_refraction_result(
    monkeypatch: pytest.MonkeyPatch,
    *,
    datum_result: object,
) -> None:
    monkeypatch.setattr(
        refraction_service_module,
        'compute_weathering_replacement_statics_from_first_breaks',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_datum_statics',
        lambda **_kwargs: datum_result,
    )


def _job_file_names(job_dir: Path) -> set[str]:
    return {path.name for path in job_dir.iterdir() if path.is_file()}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _segy_output_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob('*')
        if path.is_file() and path.suffix.lower() in {'.sgy', '.segy'}
    ]
