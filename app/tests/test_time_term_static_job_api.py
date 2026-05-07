from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.services.time_term_static_service as time_term_service
from app.api.schemas import TimeTermStaticApplyRequest
from app.main import app
from app.services.time_term_static_apply_trace_store import (
    TimeTermTraceStoreApplyResult,
)
from app.services.time_term_static_artifacts import TimeTermStaticArtifactPaths

FILE_ID = 'datum-residual-corrected-file-id'
KEY1_BYTE = 189
KEY2_BYTE = 193
DT = 0.004


class _Inputs:
    def __init__(self, tmp_path: Path) -> None:
        self.pick_source_description = 'batch_predicted_npz:predicted_picks_time_s.npz'
        self.datum_solution_path = None
        self.residual_solution_path = None
        self.linkage_artifact_path = tmp_path / 'geometry_linkage.npz'


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


def _payload(
    *,
    register_corrected_file: bool = True,
    moveout_offset_byte: int | None = 37,
) -> dict[str, Any]:
    return {
        'file_id': FILE_ID,
        'key1_byte': KEY1_BYTE,
        'key2_byte': KEY2_BYTE,
        'pick_source': {
            'kind': 'batch_predicted_npz',
            'job_id': 'batch-job-id',
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
        'velocity': {
            'replacement_velocity_m_s': 2000.0,
            'refractor_velocity_m_s': 4500.0,
            'weathering_velocity_m_s': None,
        },
        'moveout': {
            'model': 'head_wave_linear_offset',
            'distance_source': 'geometry',
            'offset_byte': moveout_offset_byte,
            'allow_missing_offset': False,
        },
        'solver': {
            'damping': 0.01,
            'gauge': 'mean_zero',
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
            'interpolation': 'linear',
            'fill_value': 0.0,
            'mode': 'weathering_only',
            'register_corrected_file': register_corrected_file,
            'max_abs_shift_ms': 250.0,
            'output_dtype': 'float32',
        },
    }


def _request(
    *,
    register_corrected_file: bool = True,
    moveout_offset_byte: int | None = 37,
) -> TimeTermStaticApplyRequest:
    return TimeTermStaticApplyRequest.model_validate(
        _payload(
            register_corrected_file=register_corrected_file,
            moveout_offset_byte=moveout_offset_byte,
        )
    )


def _create_time_term_job(
    client: TestClient,
    tmp_path: Path,
    *,
    job_id: str = 'time-term-job-id',
) -> tuple[str, Path]:
    job_dir = tmp_path / job_id
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            job_id,
            file_id=FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            statics_kind='time_term',
            artifacts_dir=str(job_dir),
        )
    return job_id, job_dir


def _install_success_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    calls: list[str],
    *,
    cancel_after_artifacts: tuple[Any, str] | None = None,
    moveout_configs: list[Any] | None = None,
) -> None:
    def _build_inputs(*args: Any, **kwargs: Any) -> _Inputs:
        calls.append('inputs')
        return _Inputs(tmp_path)

    def _moveout(*args: Any, **kwargs: Any) -> object:
        calls.append('moveout')
        if len(args) >= 2 and moveout_configs is not None:
            moveout_configs.append(args[1])
        return object()

    def _design(*args: Any, **kwargs: Any) -> object:
        calls.append('design')
        return object()

    def _solve(*args: Any, **kwargs: Any) -> object:
        calls.append('solver')
        return object()

    def _applied_shift(*args: Any, **kwargs: Any) -> object:
        calls.append('applied_shift')
        return object()

    def _artifacts(*, job_dir: Path, **kwargs: Any) -> TimeTermStaticArtifactPaths:
        calls.append('artifacts')
        solution = job_dir / 'time_term_static_solution.npz'
        qc = job_dir / 'time_term_static_qc.json'
        csv = job_dir / 'time_term_statics.csv'
        np.savez(solution, trace=np.asarray([0.0, 1.0], dtype=np.float64))
        qc.write_text('{"ok": true}', encoding='utf-8')
        csv.write_text('trace_index,time_term_ms\n0,0.0\n', encoding='utf-8')
        if cancel_after_artifacts is not None:
            state, job_id = cancel_after_artifacts
            with state.lock:
                state.jobs.request_cancel(job_id)
        return TimeTermStaticArtifactPaths(
            solution_npz_path=solution,
            qc_json_path=qc,
            statics_csv_path=csv,
        )

    def _apply(**kwargs: Any) -> TimeTermTraceStoreApplyResult:
        calls.append('apply')
        artifacts_dir = Path(kwargs['artifacts_dir'])
        corrected_store = tmp_path / 'time-term-corrected-store'
        corrected_store.mkdir(exist_ok=True)
        corrected_json = artifacts_dir / 'corrected_file.json'
        corrected_json.write_text(
            json.dumps({'file_id': 'time-term-corrected-file-id'}),
            encoding='utf-8',
        )
        return TimeTermTraceStoreApplyResult(
            file_id='time-term-corrected-file-id',
            store_path=corrected_store,
            store_name=corrected_store.name,
            source_file_id=FILE_ID,
            source_store_path=tmp_path / 'source-store',
            solution_npz_path=Path(kwargs['solution_npz_path']),
            job_id='time-term-job-id',
            mode='weathering_only',
            applied_shift_field='applied_weathering_shift_s_sorted',
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            dt=DT,
            n_traces=2,
            n_samples=16,
            shift_min_ms=0.0,
            shift_max_ms=0.0,
            shift_mean_ms=0.0,
            shift_max_abs_ms=0.0,
            corrected_file_json_path=corrected_json,
        )

    monkeypatch.setattr(time_term_service, 'build_time_term_inversion_inputs', _build_inputs)
    monkeypatch.setattr(time_term_service, 'compute_time_term_moveout', _moveout)
    monkeypatch.setattr(time_term_service, 'build_time_term_design_matrix', _design)
    monkeypatch.setattr(
        time_term_service,
        'solve_time_term_robust_least_squares',
        _solve,
    )
    monkeypatch.setattr(
        time_term_service,
        'build_time_term_applied_shift_result',
        _applied_shift,
    )
    monkeypatch.setattr(
        time_term_service,
        'write_time_term_static_artifacts',
        _artifacts,
    )
    monkeypatch.setattr(
        time_term_service,
        'apply_time_term_static_correction_to_trace_store',
        _apply,
    )


def test_run_time_term_static_apply_job_success_with_corrected_file(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id, job_dir = _create_time_term_job(client, tmp_path)
    calls: list[str] = []
    _install_success_fakes(monkeypatch, tmp_path, calls)

    time_term_service.run_time_term_static_apply_job(
        job_id,
        _request(register_corrected_file=True),
        client.app.state.sv,
    )

    assert calls == [
        'inputs',
        'moveout',
        'design',
        'solver',
        'applied_shift',
        'artifacts',
        'apply',
    ]
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    assert job['progress'] == 1.0
    assert job['message'] == 'done'
    assert job['corrected_file_id'] == 'time-term-corrected-file-id'

    files_response = client.get(f'/statics/job/{job_id}/files')
    assert files_response.status_code == 200
    assert {item['name'] for item in files_response.json()['files']} == {
        'job_meta.json',
        'time_term_static_solution.npz',
        'time_term_static_qc.json',
        'time_term_statics.csv',
        'corrected_file.json',
    }

    meta = json.loads((job_dir / 'job_meta.json').read_text(encoding='utf-8'))
    assert meta['job_type'] == 'statics'
    assert meta['statics_kind'] == 'time_term'
    assert meta['source_file_id'] == FILE_ID
    assert meta['artifacts']['solution_npz'] == 'time_term_static_solution.npz'


def test_run_time_term_static_apply_job_success_artifacts_only(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id, _job_dir = _create_time_term_job(client, tmp_path)
    calls: list[str] = []
    _install_success_fakes(monkeypatch, tmp_path, calls)

    time_term_service.run_time_term_static_apply_job(
        job_id,
        _request(register_corrected_file=False),
        client.app.state.sv,
    )

    assert calls == [
        'inputs',
        'moveout',
        'design',
        'solver',
        'applied_shift',
        'artifacts',
    ]
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    assert 'corrected_file_id' not in job
    files_response = client.get(f'/statics/job/{job_id}/files')
    assert {item['name'] for item in files_response.json()['files']} == {
        'job_meta.json',
        'time_term_static_solution.npz',
        'time_term_static_qc.json',
        'time_term_statics.csv',
    }


def test_run_time_term_static_apply_job_passes_moveout_offset_byte(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id, _job_dir = _create_time_term_job(client, tmp_path)
    calls: list[str] = []
    moveout_configs: list[Any] = []
    _install_success_fakes(
        monkeypatch,
        tmp_path,
        calls,
        moveout_configs=moveout_configs,
    )

    time_term_service.run_time_term_static_apply_job(
        job_id,
        _request(register_corrected_file=False, moveout_offset_byte=41),
        client.app.state.sv,
    )

    assert len(moveout_configs) == 1
    assert moveout_configs[0].offset_byte == 41


def test_run_time_term_static_apply_job_cancel_before_solver(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id, _job_dir = _create_time_term_job(client, tmp_path)
    calls: list[str] = []
    _install_success_fakes(monkeypatch, tmp_path, calls)
    with client.app.state.sv.lock:
        client.app.state.sv.jobs[job_id]['cancel_requested'] = True

    time_term_service.run_time_term_static_apply_job(
        job_id,
        _request(),
        client.app.state.sv,
    )

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'cancelled'
    assert job['message'] == 'The job was cancelled by the user.'
    assert calls == []


def test_run_time_term_static_apply_job_cancel_before_apply(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id, _job_dir = _create_time_term_job(client, tmp_path)
    calls: list[str] = []
    _install_success_fakes(
        monkeypatch,
        tmp_path,
        calls,
        cancel_after_artifacts=(client.app.state.sv, job_id),
    )

    time_term_service.run_time_term_static_apply_job(
        job_id,
        _request(register_corrected_file=True),
        client.app.state.sv,
    )

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'cancelled'
    assert 'apply' not in calls


def test_run_time_term_static_apply_job_failure_sets_error_message(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id, _job_dir = _create_time_term_job(client, tmp_path)
    calls: list[str] = []
    _install_success_fakes(monkeypatch, tmp_path, calls)

    def _raise_solver_error(*args: Any, **kwargs: Any) -> object:
        raise ValueError('time-term solver mismatch')

    monkeypatch.setattr(
        time_term_service,
        'solve_time_term_robust_least_squares',
        _raise_solver_error,
    )

    time_term_service.run_time_term_static_apply_job(
        job_id,
        _request(),
        client.app.state.sv,
    )

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'error'
    assert job['message'] == 'time-term solver mismatch'
    assert 'applied_shift' not in calls


def test_time_term_static_job_invalid_file_id_becomes_error_status(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_id, _job_dir = _create_time_term_job(client, tmp_path)

    time_term_service.run_time_term_static_apply_job(
        job_id,
        _request(register_corrected_file=False),
        client.app.state.sv,
    )

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'error'
    assert job['message'] == f'file_id not found: {FILE_ID}'


def test_time_term_static_job_download_artifacts(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id, _job_dir = _create_time_term_job(client, tmp_path)
    calls: list[str] = []
    _install_success_fakes(monkeypatch, tmp_path, calls)
    time_term_service.run_time_term_static_apply_job(
        job_id,
        _request(register_corrected_file=True),
        client.app.state.sv,
    )

    solution_response = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': 'time_term_static_solution.npz'},
    )
    qc_response = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': 'time_term_static_qc.json'},
    )
    csv_response = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': 'time_term_statics.csv'},
    )
    corrected_response = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': 'corrected_file.json'},
    )

    assert solution_response.status_code == 200
    with np.load(BytesIO(solution_response.content), allow_pickle=False) as data:
        np.testing.assert_allclose(data['trace'], np.asarray([0.0, 1.0]))
    assert qc_response.json() == {'ok': True}
    assert csv_response.text.startswith('trace_index,time_term_ms\n')
    assert corrected_response.json() == {'file_id': 'time-term-corrected-file-id'}


def test_time_term_static_job_download_rejects_path_traversal(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_id, job_dir = _create_time_term_job(client, tmp_path)
    job_dir.mkdir(parents=True)

    response = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': '../time_term_static_solution.npz'},
    )

    assert response.status_code == 400
    assert response.json() == {'detail': 'Invalid file name'}
