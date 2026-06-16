from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.services.static_job_targets as static_job_targets
import app.services.residual_static_service as residual_service
from app.api.schemas import ResidualStaticApplyRequest
from app.main import app
from app.services.residual_static_artifacts import (
    ResidualStaticArtifactMetadata,
    ResidualStaticArtifactPaths,
    write_residual_static_artifacts,
)
from app.services.residual_static_corrected_store import (
    ResidualStaticCorrectedStoreResult,
)
from app.services.residual_static_inputs import ResidualStaticResolvedArtifacts
from seis_statics.residual import ResidualStaticSolverInputs

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


def _package_api_request() -> ResidualStaticApplyRequest:
    return ResidualStaticApplyRequest.model_validate(
        {
            'file_id': 'input-file',
            'datum_solution': {'job_id': 'datum-job'},
            'pick_source': {'kind': 'manual_memmap'},
            'geometry': {'source_id_byte': 17, 'receiver_id_byte': 13},
            'offset': {'offset_byte': None},
            'moveout': {'model': 'none'},
            'solver': {
                'min_valid_picks': 10,
                'max_abs_estimated_delay_ms': 1000.0,
            },
            'robust': {
                'enabled': True,
                'method': 'mad',
                'max_iterations': 3,
                'threshold': 3.0,
                'min_used_fraction': 0.5,
            },
        },
    )


def _package_api_solver_inputs() -> ResidualStaticSolverInputs:
    n_sources = 5
    n_receivers = 6
    source_unique_ids = np.arange(101, 101 + n_sources, dtype=np.int64)
    receiver_unique_ids = np.arange(201, 201 + n_receivers, dtype=np.int64)
    source_index = np.repeat(np.arange(n_sources, dtype=np.int64), n_receivers)
    receiver_index = np.tile(np.arange(n_receivers, dtype=np.int64), n_sources)
    n_traces = int(source_index.size)

    source_delay_s = np.linspace(-0.009, 0.007, n_sources, dtype=np.float64)
    source_delay_s -= float(np.mean(source_delay_s))
    receiver_delay_s = np.linspace(-0.004, 0.005, n_receivers, dtype=np.float64)
    receiver_delay_s -= float(np.mean(receiver_delay_s))
    pick_time_after_datum = (
        0.075 + source_delay_s[source_index] + receiver_delay_s[receiver_index]
    )
    sample_index = np.arange(n_traces, dtype=np.float64)
    noise = np.sin(sample_index * 1.7) + 0.5 * np.cos(sample_index * 2.3)
    noise -= float(np.mean(noise))
    pick_time_after_datum = pick_time_after_datum + (0.002 * noise)
    pick_time_after_datum[13] += 0.03

    datum_trace_shift = 0.001 * np.sin(np.arange(n_traces, dtype=np.float64))
    valid_mask = np.ones(n_traces, dtype=bool)

    return ResidualStaticSolverInputs(
        picks_time_s_sorted=pick_time_after_datum - datum_trace_shift,
        valid_pick_mask_sorted=valid_mask,
        pick_time_after_datum_s_sorted=pick_time_after_datum,
        datum_trace_shift_s_sorted=datum_trace_shift,
        source_id_sorted=source_unique_ids[source_index],
        receiver_id_sorted=receiver_unique_ids[receiver_index],
        source_unique_ids=source_unique_ids,
        receiver_unique_ids=receiver_unique_ids,
        source_index_sorted=source_index,
        receiver_index_sorted=receiver_index,
        source_valid_pick_counts=np.bincount(
            source_index,
            minlength=n_sources,
        ).astype(np.int64),
        receiver_valid_pick_counts=np.bincount(
            receiver_index,
            minlength=n_receivers,
        ).astype(np.int64),
        offset_sorted=None,
        abs_offset_sorted=None,
        key1_sorted=source_unique_ids[source_index],
        key2_sorted=receiver_unique_ids[receiver_index],
        source_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        dt=0.004,
        n_traces=n_traces,
        n_samples=96,
        key1_byte=189,
        key2_byte=193,
        source_id_byte=17,
        receiver_id_byte=13,
        offset_byte=None,
        moveout_model='none',
        input_file_id='input-file',
        datum_source_file_id='datum-source-file',
        datum_job_id='datum-job',
        pick_source_kind='manual_memmap',
        metadata={'source': 'residual-static-service-package-api-test'},
    )


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
        '_solve_residual_static_with_package_api',
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


def test_residual_static_service_solve_uses_package_api_and_writes_stable_artifacts(
    tmp_path: Path,
) -> None:
    inputs = _package_api_solver_inputs()
    result = residual_service._solve_residual_static_with_package_api(
        inputs,
        _package_api_request(),
    )
    paths = write_residual_static_artifacts(
        tmp_path,
        inputs,
        result,
        metadata=ResidualStaticArtifactMetadata(
            job_id='job-1',
            input_file_id='input-file',
            datum_source_file_id='datum-source-file',
            datum_job_id='datum-job',
            datum_solution_artifact='datum_static_solution.npz',
            pick_source_kind='manual_memmap',
        ),
    )

    expected_source_delay = np.asarray(
        [-0.00754332, -0.004312, 0.0002385, 0.00401094, 0.00760588],
        dtype=np.float64,
    )
    expected_receiver_delay = np.asarray(
        [-0.00467254, -0.00257925, -0.00073709, 0.00082254, 0.00257405, 0.0045923],
        dtype=np.float64,
    )
    expected_residual = np.asarray(
        [
            0.00071395,
            0.00073771,
            -0.00124474,
            -0.00141704,
            -0.00031926,
            0.00152939,
            -0.00058631,
            -0.00197107,
            0.00276613,
            0.00090501,
            -0.0020196,
            0.00090583,
            0.00115059,
            np.nan,
            -0.00163889,
            -0.00044319,
            0.00226638,
            -0.0013349,
            -0.00214466,
            0.00237348,
            0.00045022,
            -0.00213924,
            0.00046832,
            0.00099188,
            0.00086644,
            -0.00114012,
            -0.00033272,
            0.00309446,
            -0.00039585,
            -0.00209221,
        ],
        dtype=np.float64,
    )
    expected_used = np.ones(inputs.n_traces, dtype=bool)
    expected_used[13] = False
    expected_rejected = np.zeros(inputs.n_traces, dtype=bool)
    expected_rejected[13] = True
    expected_applied_shift = np.asarray(
        [
            0.01221586,
            0.01012257,
            0.00828041,
            0.00672078,
            0.00496927,
            0.00295103,
            0.00898454,
            0.00689125,
            0.00504909,
            0.00348946,
            0.00173795,
            -0.0002803,
            0.00443404,
            0.00234075,
            0.00049858,
            -0.00106104,
            -0.00281255,
            -0.0048308,
            0.00066161,
            -0.00143168,
            -0.00327385,
            -0.00483347,
            -0.00658499,
            -0.00860323,
            -0.00293334,
            -0.00502663,
            -0.0068688,
            -0.00842842,
            -0.01017993,
            -0.01219818,
        ],
        dtype=np.float64,
    )

    with np.load(paths.solution_npz_path) as solution:
        np.testing.assert_allclose(
            solution['source_delay_s'],
            expected_source_delay,
            atol=5.0e-9,
        )
        np.testing.assert_allclose(
            solution['receiver_delay_s'],
            expected_receiver_delay,
            atol=5.0e-9,
        )
        np.testing.assert_allclose(
            solution['residual_after_s'],
            expected_residual,
            atol=5.0e-9,
            equal_nan=True,
        )
        np.testing.assert_array_equal(solution['used_mask_sorted'], expected_used)
        np.testing.assert_array_equal(solution['rejected_mask_sorted'], expected_rejected)
        np.testing.assert_allclose(
            solution['applied_residual_shift_s_sorted'],
            expected_applied_shift,
            atol=5.0e-9,
        )

    qc = json.loads(paths.qc_json_path.read_text())
    assert qc['counts']['n_initial_used_picks'] == 30
    assert qc['counts']['n_final_used_picks'] == 29
    assert qc['counts']['n_rejected_total'] == 1
    assert qc['robust']['stop_reason'] == 'converged'
    assert qc['solver']['final_lsmr']['itn'] == 8

    with paths.statics_csv_path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert rows[13]['final_used'] == 'false'
    assert rows[13]['rejected'] == 'true'
    assert rows[13]['rejected_iteration'] == '0'
    np.testing.assert_allclose(
        float(rows[13]['applied_residual_shift_ms']),
        expected_applied_shift[13] * 1000.0,
        atol=5.0e-6,
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
        static_job_targets,
        'start_static_job_thread',
        _capture_start_job_thread,
    )

    response = client.post('/statics/residual/apply', json=_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload['state'] == 'queued'
    assert isinstance(payload['job_id'], str)
    assert len(started) == 1
    assert started[0]['target'] is static_job_targets.get_static_job_target('residual')

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
        static_job_targets,
        'start_static_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    payload = _payload()
    payload['moveout'] = {'model': 'none'}

    response = client.post('/statics/residual/apply', json=payload)

    assert response.status_code == 422
    assert started == []
