from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
import app.services.residual_static_service as residual_service
from app.api.schemas import ResidualStaticApplyRequest
from app.main import app
from app.services.residual_static_artifacts import ResidualStaticArtifactPaths
from app.services.residual_static_corrected_store import (
    ResidualStaticCorrectedStoreResult,
)
from app.services.residual_static_inputs import ResidualStaticResolvedArtifacts

FILE_ID = 'datum-corrected-file-id'
SOURCE_FILE_ID = 'raw-source-file-id'
DATUM_JOB_ID = 'datum-static-job-id'
PICK_JOB_ID = 'batch-pick-job-id'
KEY1_BYTE = 189
KEY2_BYTE = 193
DT = 0.004


class _Reader:
    key1_byte = KEY1_BYTE
    key2_byte = KEY2_BYTE

    def __init__(self) -> None:
        self.traces = np.zeros((4, 16), dtype=np.float32)
        self.meta = {'dt': DT}

    def get_n_samples(self) -> int:
        return int(self.traces.shape[-1])


class _Inputs:
    pick_source_kind = 'batch_npz'


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
        'file_id': FILE_ID,
        'key1_byte': KEY1_BYTE,
        'key2_byte': KEY2_BYTE,
        'datum_solution': {
            'job_id': DATUM_JOB_ID,
            'name': 'datum_static_solution.npz',
        },
        'pick_source': {
            'kind': 'batch_job_artifact',
            'job_id': PICK_JOB_ID,
            'name': 'predicted_picks_time_s.npz',
        },
        'geometry': {
            'source_id_byte': 17,
            'receiver_id_byte': 13,
        },
        'offset': {'offset_byte': 37},
        'moveout': {'model': 'linear_abs_offset'},
        'solver': {
            'gauge': 'zero_mean_source_receiver',
            'damping_lambda': 0.0,
            'min_valid_picks': 10,
            'min_picks_per_source': 1,
            'min_picks_per_receiver': 1,
            'max_abs_estimated_delay_ms': 250.0,
        },
        'robust': {
            'enabled': True,
            'method': 'mad',
            'max_iterations': 3,
            'threshold': 4.0,
            'min_used_fraction': 0.5,
        },
        'apply': {
            'interpolation': 'linear',
            'fill_value': 0.0,
            'max_abs_shift_ms': 250.0,
            'output_dtype': 'float32',
            'register_corrected_file': True,
        },
    }


def _request(**overrides: Any) -> ResidualStaticApplyRequest:
    payload = _payload()
    payload.update(overrides)
    return ResidualStaticApplyRequest.model_validate(payload)


def _create_residual_job(client: TestClient, tmp_path: Path) -> tuple[str, Path, Path]:
    job_id = 'residual-job-id'
    job_dir = tmp_path / 'residual-job'
    source_store = tmp_path / 'datum-corrected-store'
    source_store.mkdir()
    state = client.app.state.sv
    state.file_registry.update(FILE_ID, store_path=source_store, dt=DT)
    with state.lock:
        state.jobs.create_static_job(
            job_id,
            file_id=FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            statics_kind='residual',
            artifacts_dir=str(job_dir),
        )
    return job_id, job_dir, source_store


def _install_success_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    calls: list[str],
) -> None:
    def _get_reader(*args: Any, **kwargs: Any) -> _Reader:
        calls.append('get_reader')
        return _Reader()

    def _resolve_artifacts(*args: Any, **kwargs: Any) -> ResidualStaticResolvedArtifacts:
        calls.append('resolve_artifacts')
        datum_path = tmp_path / 'datum_static_solution.npz'
        pick_path = tmp_path / 'predicted_picks_time_s.npz'
        datum_path.write_bytes(b'datum')
        pick_path.write_bytes(b'picks')
        return ResidualStaticResolvedArtifacts(
            datum_solution_path=datum_path,
            pick_artifact_path=pick_path,
            datum_job_id=DATUM_JOB_ID,
            datum_source_file_id=SOURCE_FILE_ID,
            datum_corrected_file_id=FILE_ID,
            pick_source_artifact_name='predicted_picks_time_s.npz',
        )

    def _load_pick_source(*args: Any, **kwargs: Any) -> object:
        calls.append('load_pick_source')
        return object()

    def _build_inputs(*args: Any, **kwargs: Any) -> _Inputs:
        calls.append('build_inputs')
        return _Inputs()

    def _solve(*args: Any, **kwargs: Any) -> object:
        calls.append('solve')
        return object()

    def _write_artifacts(
        job_dir: Path,
        *args: Any,
        **kwargs: Any,
    ) -> ResidualStaticArtifactPaths:
        calls.append('write_artifacts')
        solution = job_dir / 'residual_static_solution.npz'
        qc = job_dir / 'residual_static_qc.json'
        csv = job_dir / 'residual_statics.csv'
        solution.write_bytes(b'solution')
        qc.write_text('{"ok":true}', encoding='utf-8')
        csv.write_text('trace,shift\n', encoding='utf-8')
        return ResidualStaticArtifactPaths(
            solution_npz_path=solution,
            qc_json_path=qc,
            statics_csv_path=csv,
        )

    def _apply_trace_store(**kwargs: Any) -> ResidualStaticCorrectedStoreResult:
        calls.append('apply_trace_store')
        progress_callback = kwargs.get('progress_callback')
        if callable(progress_callback):
            progress_callback(1.0, 'registering_residual_corrected_trace_store')
        artifacts_dir = Path(kwargs['artifacts_dir'])
        corrected_file = artifacts_dir / 'corrected_file.json'
        corrected_store = tmp_path / 'residual-corrected-store'
        corrected_store.mkdir(exist_ok=True)
        corrected_file.write_text(
            json.dumps({'file_id': 'residual-corrected-file-id'}),
            encoding='utf-8',
        )
        return ResidualStaticCorrectedStoreResult(
            file_id='residual-corrected-file-id',
            store_path=corrected_store,
            store_name=corrected_store.name,
            corrected_file_json_path=corrected_file,
            derived_from_file_id=FILE_ID,
            derived_from_store_path=Path(kwargs['source_store_path']),
            job_id=kwargs['job_id'],
            key1_byte=kwargs['key1_byte'],
            key2_byte=kwargs['key2_byte'],
            dt=DT,
        )

    monkeypatch.setattr(residual_service, 'get_reader', _get_reader)
    monkeypatch.setattr(
        residual_service,
        'resolve_residual_static_input_artifacts',
        _resolve_artifacts,
    )
    monkeypatch.setattr(
        residual_service,
        'load_residual_static_pick_source',
        _load_pick_source,
    )
    monkeypatch.setattr(
        residual_service,
        'build_residual_static_solver_inputs',
        _build_inputs,
    )
    monkeypatch.setattr(
        residual_service,
        'solve_residual_static_robust_least_squares',
        _solve,
    )
    monkeypatch.setattr(
        residual_service,
        'write_residual_static_artifacts',
        _write_artifacts,
    )
    monkeypatch.setattr(
        residual_service,
        'apply_residual_static_correction_to_trace_store',
        _apply_trace_store,
    )


def test_residual_static_apply_endpoint_creates_static_job(
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

    response = client.post('/statics/residual/apply', json=_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload['state'] == 'queued'
    assert isinstance(payload['job_id'], str)
    assert len(started) == 1
    assert started[0]['target'] is statics_router_module.run_residual_static_apply_job

    state = client.app.state.sv
    with state.lock:
        job = dict(state.jobs[payload['job_id']])

    assert job['status'] == 'queued'
    assert job['job_type'] == 'statics'
    assert job['statics_kind'] == 'residual'
    assert job['file_id'] == FILE_ID
    assert job['key1_byte'] == KEY1_BYTE
    assert job['key2_byte'] == KEY2_BYTE


def test_run_residual_static_apply_job_success_lifecycle(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id, job_dir, _source_store = _create_residual_job(client, tmp_path)
    calls: list[str] = []
    _install_success_fakes(monkeypatch, tmp_path, calls)

    residual_service.run_residual_static_apply_job(
        job_id,
        _request(),
        client.app.state.sv,
    )

    assert calls == [
        'get_reader',
        'resolve_artifacts',
        'load_pick_source',
        'build_inputs',
        'solve',
        'write_artifacts',
        'apply_trace_store',
    ]
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    assert job['progress'] == 1.0
    assert job['corrected_file_id'] == 'residual-corrected-file-id'
    assert Path(str(job['corrected_store_path'])).name == 'residual-corrected-store'

    files_response = client.get(f'/statics/job/{job_id}/files')
    assert files_response.status_code == 200
    assert {item['name'] for item in files_response.json()['files']} == {
        'job_meta.json',
        'residual_static_solution.npz',
        'residual_static_qc.json',
        'residual_statics.csv',
        'corrected_file.json',
    }

    meta = json.loads((job_dir / 'job_meta.json').read_text(encoding='utf-8'))
    assert meta['job_type'] == 'statics'
    assert meta['statics_kind'] == 'residual'
    assert meta['source_file_id'] == FILE_ID
    assert meta['inputs']['datum_solution']['job_id'] == DATUM_JOB_ID
    assert meta['inputs']['geometry'] == {
        'source_id_byte': 17,
        'receiver_id_byte': 13,
    }
    assert meta['artifacts']['corrected_file_json'] == 'corrected_file.json'


def test_run_residual_static_apply_job_cancel_before_solver(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id, _job_dir, _source_store = _create_residual_job(client, tmp_path)
    calls: list[str] = []
    _install_success_fakes(monkeypatch, tmp_path, calls)
    with client.app.state.sv.lock:
        client.app.state.sv.jobs[job_id]['cancel_requested'] = True

    residual_service.run_residual_static_apply_job(
        job_id,
        _request(),
        client.app.state.sv,
    )

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'cancelled'
    assert job['message'] == 'The job was cancelled by the user.'
    assert calls == []


def test_run_residual_static_apply_job_cancel_before_apply(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id, _job_dir, _source_store = _create_residual_job(client, tmp_path)
    calls: list[str] = []
    _install_success_fakes(monkeypatch, tmp_path, calls)

    original_writer = residual_service.write_residual_static_artifacts

    def _write_and_cancel(*args: Any, **kwargs: Any) -> ResidualStaticArtifactPaths:
        paths = original_writer(*args, **kwargs)
        with client.app.state.sv.lock:
            client.app.state.sv.jobs.request_cancel(job_id)
        return paths

    monkeypatch.setattr(
        residual_service,
        'write_residual_static_artifacts',
        _write_and_cancel,
    )

    residual_service.run_residual_static_apply_job(
        job_id,
        _request(),
        client.app.state.sv,
    )

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'cancelled'
    assert 'apply_trace_store' not in calls


def test_run_residual_static_apply_job_failure_sets_error_message(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id, _job_dir, _source_store = _create_residual_job(client, tmp_path)
    calls: list[str] = []
    _install_success_fakes(monkeypatch, tmp_path, calls)

    def _raise_input_error(*args: Any, **kwargs: Any) -> object:
        raise ValueError('residual input mismatch')

    monkeypatch.setattr(
        residual_service,
        'build_residual_static_solver_inputs',
        _raise_input_error,
    )

    residual_service.run_residual_static_apply_job(
        job_id,
        _request(),
        client.app.state.sv,
    )

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'error'
    assert job['message'] == 'residual input mismatch'
    assert 'solve' not in calls


def test_residual_static_job_files_and_download_artifact(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_id, job_dir, _source_store = _create_residual_job(client, tmp_path)
    job_dir.mkdir(parents=True)
    (job_dir / 'job_meta.json').write_text('{"job_id":"residual"}', encoding='utf-8')
    (job_dir / 'residual_static_solution.npz').write_bytes(b'npz')
    (job_dir / 'residual_static_qc.json').write_text('{"ok":true}', encoding='utf-8')
    (job_dir / 'residual_statics.csv').write_text('trace,shift\n', encoding='utf-8')
    (job_dir / 'corrected_file.json').write_text(
        '{"file_id":"residual"}',
        encoding='utf-8',
    )

    files_response = client.get(f'/statics/job/{job_id}/files')
    download_response = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': 'residual_static_qc.json'},
    )

    assert files_response.status_code == 200
    assert {item['name'] for item in files_response.json()['files']} == {
        'job_meta.json',
        'residual_static_solution.npz',
        'residual_static_qc.json',
        'residual_statics.csv',
        'corrected_file.json',
    }
    assert download_response.status_code == 200
    assert download_response.text == '{"ok":true}'


def test_residual_static_job_download_rejects_path_traversal(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_id, job_dir, _source_store = _create_residual_job(client, tmp_path)
    job_dir.mkdir(parents=True)

    response = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': '../residual_static_qc.json'},
    )

    assert response.status_code == 400
    assert response.json() == {'detail': 'Invalid file name'}


def test_residual_static_apply_endpoint_schema_validation_422(
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
    payload['moveout'] = {'model': 'none'}

    response = client.post('/statics/residual/apply', json=payload)

    assert response.status_code == 422
    assert started == []
