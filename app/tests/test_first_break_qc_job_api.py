from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
from app.main import app

KEY1_SORTED = np.asarray([10, 10, 20, 20], dtype=np.int64)
KEY2_SORTED = np.asarray([1, 2, 1, 2], dtype=np.int64)
OFFSET_SORTED = np.asarray([-1200.0, -400.0, 350.0, 1250.0], dtype=np.float64)
PICKS_TIME_S = np.asarray([0.004, np.nan, 0.012, 0.020], dtype=np.float32)
DT = 0.004
N_SAMPLES = 8


class _Reader:
    key1_byte = 189
    key2_byte = 193

    def __init__(self) -> None:
        self.traces = np.zeros((4, N_SAMPLES), dtype=np.float32)
        self.meta = {'dt': DT}
        self.headers = {
            189: KEY1_SORTED,
            193: KEY2_SORTED,
            37: OFFSET_SORTED,
        }

    def get_n_samples(self) -> int:
        return N_SAMPLES

    def ensure_header(self, byte: int) -> np.ndarray:
        return self.headers[int(byte)]

    def get_sorted_to_original(self) -> np.ndarray:
        return np.arange(4, dtype=np.int64)


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


def _write_real_datum_solution(path: Path) -> Path:
    source_shift = np.zeros(4, dtype=np.float64)
    receiver_shift = np.zeros(4, dtype=np.float64)
    np.savez(
        path,
        trace_shift_s_sorted=source_shift + receiver_shift,
        source_shift_s_sorted=source_shift,
        receiver_shift_s_sorted=receiver_shift,
        source_elevation_m_sorted=np.asarray([100.0, 105.0, 110.0, 115.0]),
        receiver_elevation_m_sorted=np.asarray([90.0, 95.0, 100.0, 105.0]),
        key1_sorted=KEY1_SORTED,
        key2_sorted=KEY2_SORTED,
        datum_elevation_m=np.float64(500.0),
        replacement_velocity_m_s=np.float64(2000.0),
        dt=np.float64(DT),
        n_traces=np.int64(4),
        key1_byte=np.int64(189),
        key2_byte=np.int64(193),
        source_elevation_byte=np.int64(45),
        receiver_elevation_byte=np.int64(41),
    )
    return path


def _write_real_pick_npz(path: Path) -> Path:
    np.savez(
        path,
        picks_time_s=PICKS_TIME_S,
        n_traces=np.int64(4),
        n_samples=np.int64(N_SAMPLES),
        dt=np.float64(DT),
        sorted_to_original=np.arange(4, dtype=np.int64),
    )
    return path


def _setup_real_first_break_qc_inputs(client: TestClient, tmp_path: Path) -> None:
    datum_dir = tmp_path / 'datum-job'
    datum_dir.mkdir()
    batch_dir = tmp_path / 'batch-job'
    batch_dir.mkdir()
    _write_real_datum_solution(datum_dir / 'datum_static_solution.npz')
    _write_real_pick_npz(batch_dir / 'predicted_picks_time_s.npz')

    state = client.app.state.sv
    state.file_registry.set_record(
        'source-file-id',
        {
            'path': str(tmp_path / 'line.sgy'),
            'store_path': str(tmp_path / 'trace-store'),
            'dt': DT,
        },
    )
    with state.lock:
        state.cached_readers['source-file-id_189_193'] = _Reader()
        state.jobs.create_static_job(
            'datum-static-job-id',
            file_id='source-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='datum',
            artifacts_dir=str(datum_dir),
        )
        state.jobs.create_batch_apply_job(
            'batch-apply-job-id',
            file_id='source-file-id',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir=str(batch_dir),
        )


def _run_job_sync(**kwargs: Any) -> object:
    target = kwargs['target']
    args = kwargs.get('args', ())
    extra_kwargs = kwargs.get('kwargs') or {}
    target(*args, **extra_kwargs)
    return object()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


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


def test_first_break_qc_job_real_path_allows_nan_pick(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _setup_real_first_break_qc_inputs(client, tmp_path)
    monkeypatch.setattr(statics_router_module, 'start_job_thread', _run_job_sync)

    response = client.post('/statics/first-break/qc', json=_payload())

    assert response.status_code == 200
    job_id = response.json()['job_id']
    status_response = client.get(f'/statics/job/{job_id}/status')
    assert status_response.status_code == 200
    assert status_response.json()['state'] == 'done'

    state = client.app.state.sv
    with state.lock:
        job = dict(state.jobs[job_id])
    job_dir = Path(str(job['artifacts_dir']))
    payload = json.loads((job_dir / 'first_break_qc.json').read_text(encoding='utf-8'))
    json.dumps(payload, allow_nan=False)
    rows = _read_csv(job_dir / 'first_break_qc.csv')
    invalid = rows[1]
    assert invalid['pick_time_raw_s'] == ''
    assert invalid['pick_time_after_datum_s'] == ''
    assert invalid['linear_moveout_model_s'] == ''
    assert invalid['residual_after_datum_s'] == ''
    assert (job_dir / 'residual_by_key1.csv').is_file()
