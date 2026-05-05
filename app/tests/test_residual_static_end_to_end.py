from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
from app.main import app
from app.services.reader import get_reader

KEY1_BYTE = 189
KEY2_BYTE = 193
SOURCE_ID_BYTE = 17
RECEIVER_ID_BYTE = 13
SOURCE_ELEVATION_BYTE = 45
RECEIVER_ELEVATION_BYTE = 41
DT = 0.004
N_SAMPLES = 128
KEY1_VALUE = 100
SOURCE_FILE_ID = 'synthetic-raw-source-file-id'
DATUM_FILE_ID = 'synthetic-datum-corrected-file-id'
DATUM_JOB_ID = 'synthetic-datum-job-id'
PICK_JOB_ID = 'synthetic-pick-job-id'


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


def _grid_arrays() -> dict[str, np.ndarray]:
    source_unique_ids = np.asarray([101, 102, 103], dtype=np.int64)
    receiver_unique_ids = np.asarray([201, 202, 203, 204], dtype=np.int64)
    source_index = np.repeat(
        np.arange(source_unique_ids.shape[0], dtype=np.int64),
        receiver_unique_ids.shape[0],
    )
    receiver_index = np.tile(
        np.arange(receiver_unique_ids.shape[0], dtype=np.int64),
        source_unique_ids.shape[0],
    )
    source_delay_s = np.asarray([-0.004, 0.0, 0.004], dtype=np.float64)
    receiver_delay_s = np.asarray([-0.008, -0.004, 0.004, 0.008], dtype=np.float64)
    estimated_delay_s = source_delay_s[source_index] + receiver_delay_s[receiver_index]
    n_traces = int(source_index.shape[0])
    return {
        'key1_sorted': np.full(n_traces, KEY1_VALUE, dtype=np.int64),
        'key2_sorted': np.arange(1, n_traces + 1, dtype=np.int64),
        'source_id_sorted': source_unique_ids[source_index],
        'receiver_id_sorted': receiver_unique_ids[receiver_index],
        'estimated_delay_s': estimated_delay_s,
        'sorted_to_original': np.asarray(
            [5, 0, 7, 1, 10, 2, 8, 3, 11, 4, 6, 9],
            dtype=np.int64,
        ),
    }


def _datum_component() -> dict[str, object]:
    return {
        'name': 'datum_static_correction',
        'job_id': DATUM_JOB_ID,
        'solution_artifact': 'datum_static_solution.npz',
        'shift_field': 'trace_shift_s_sorted',
        'value_kind': 'applied_event_time_shift_s',
    }


def _write_datum_corrected_store(store: Path, arrays: dict[str, np.ndarray]) -> None:
    n_traces = int(arrays['key1_sorted'].shape[0])
    base_sample = 40
    traces = np.zeros((n_traces, N_SAMPLES), dtype=np.float32)
    for trace_index, delay_s in enumerate(arrays['estimated_delay_s']):
        sample_index = base_sample + int(round(float(delay_s) / DT))
        traces[trace_index, sample_index] = 1.0

    store.mkdir(parents=True)
    np.save(store / 'traces.npy', traces)
    np.savez(
        store / 'index.npz',
        key1_values=np.asarray([KEY1_VALUE], dtype=np.int64),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([n_traces], dtype=np.int64),
        sorted_to_original=arrays['sorted_to_original'],
    )
    np.save(store / f'headers_byte_{KEY1_BYTE}.npy', arrays['key1_sorted'])
    np.save(store / f'headers_byte_{KEY2_BYTE}.npy', arrays['key2_sorted'])
    np.save(store / f'headers_byte_{SOURCE_ID_BYTE}.npy', arrays['source_id_sorted'])
    np.save(
        store / f'headers_byte_{RECEIVER_ID_BYTE}.npy',
        arrays['receiver_id_sorted'],
    )
    meta = {
        'schema_version': 1,
        'dtype': 'float32',
        'n_traces': n_traces,
        'n_samples': N_SAMPLES,
        'key_bytes': {'key1': KEY1_BYTE, 'key2': KEY2_BYTE},
        'sorted_by': ['key1', 'key2'],
        'dt': DT,
        'original_segy_path': '/synthetic/source.sgy',
        'source_sha256': 'synthetic-datum-source-sha256',
        'derived': {
            'kind': 'time_shifted_trace_store',
            'from_file_id': SOURCE_FILE_ID,
            'components': [_datum_component()],
        },
    }
    (store / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')


def _write_datum_solution(path: Path, arrays: dict[str, np.ndarray]) -> None:
    n_traces = int(arrays['key1_sorted'].shape[0])
    zero_shift = np.zeros(n_traces, dtype=np.float64)
    np.savez(
        path,
        trace_shift_s_sorted=zero_shift,
        source_shift_s_sorted=zero_shift,
        receiver_shift_s_sorted=zero_shift,
        source_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        key1_sorted=arrays['key1_sorted'],
        key2_sorted=arrays['key2_sorted'],
        datum_elevation_m=np.asarray(500.0, dtype=np.float64),
        replacement_velocity_m_s=np.asarray(2000.0, dtype=np.float64),
        dt=np.asarray(DT, dtype=np.float64),
        n_traces=np.asarray(n_traces, dtype=np.int64),
        key1_byte=np.asarray(KEY1_BYTE, dtype=np.int64),
        key2_byte=np.asarray(KEY2_BYTE, dtype=np.int64),
        source_elevation_byte=np.asarray(SOURCE_ELEVATION_BYTE, dtype=np.int64),
        receiver_elevation_byte=np.asarray(RECEIVER_ELEVATION_BYTE, dtype=np.int64),
    )


def _write_pick_artifact(path: Path, arrays: dict[str, np.ndarray]) -> None:
    base_pick_time_s = 40 * DT
    picks_sorted = base_pick_time_s + arrays['estimated_delay_s']
    picks_original = np.empty_like(picks_sorted)
    picks_original[arrays['sorted_to_original']] = picks_sorted
    np.savez(
        path,
        picks_time_s=picks_original,
        sorted_to_original=arrays['sorted_to_original'],
        n_traces=np.asarray(picks_sorted.shape[0], dtype=np.int64),
        n_samples=np.asarray(N_SAMPLES, dtype=np.int64),
        dt=np.asarray(DT, dtype=np.float64),
    )


def _install_synthetic_inputs(client: TestClient, tmp_path: Path) -> dict[str, np.ndarray]:
    arrays = _grid_arrays()
    datum_store = tmp_path / 'datum-corrected-store'
    datum_job_dir = tmp_path / 'datum-job'
    pick_job_dir = tmp_path / 'pick-job'
    datum_job_dir.mkdir()
    pick_job_dir.mkdir()
    _write_datum_corrected_store(datum_store, arrays)
    _write_datum_solution(datum_job_dir / 'datum_static_solution.npz', arrays)
    _write_pick_artifact(pick_job_dir / 'predicted_picks_time_s.npz', arrays)
    (datum_job_dir / 'corrected_file.json').write_text(
        json.dumps(
            {
                'file_id': DATUM_FILE_ID,
                'store_path': str(datum_store),
                'derived_from_file_id': SOURCE_FILE_ID,
                'job_id': DATUM_JOB_ID,
                'key1_byte': KEY1_BYTE,
                'key2_byte': KEY2_BYTE,
            }
        ),
        encoding='utf-8',
    )

    state = client.app.state.sv
    state.file_registry.update(DATUM_FILE_ID, store_path=datum_store, dt=DT)
    with state.lock:
        state.jobs.create_static_job(
            DATUM_JOB_ID,
            file_id=SOURCE_FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            statics_kind='datum',
            artifacts_dir=str(datum_job_dir),
        )
        state.jobs.set_static_corrected_file(
            DATUM_JOB_ID,
            corrected_file_id=DATUM_FILE_ID,
            corrected_store_path=str(datum_store),
        )
        state.jobs.mark_done(DATUM_JOB_ID, progress_1=True)
        state.jobs.create_batch_apply_job(
            PICK_JOB_ID,
            file_id=SOURCE_FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            artifacts_dir=str(pick_job_dir),
        )
        state.jobs.mark_done(PICK_JOB_ID, progress_1=True)
    return arrays


def _request_payload() -> dict[str, Any]:
    return {
        'file_id': DATUM_FILE_ID,
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
            'source_id_byte': SOURCE_ID_BYTE,
            'receiver_id_byte': RECEIVER_ID_BYTE,
        },
        'offset': {'offset_byte': None},
        'moveout': {'model': 'none'},
        'solver': {
            'gauge': 'zero_mean_source_receiver',
            'damping_lambda': 0.0,
            'min_valid_picks': 10,
            'min_picks_per_source': 1,
            'min_picks_per_receiver': 1,
            'max_abs_estimated_delay_ms': 250.0,
        },
        'robust': {
            'enabled': False,
            'method': 'mad',
            'max_iterations': 1,
            'threshold': 4.0,
            'min_used_fraction': 1.0,
        },
        'apply': {
            'interpolation': 'linear',
            'fill_value': 0.0,
            'max_abs_shift_ms': 250.0,
            'output_dtype': 'float32',
            'register_corrected_file': True,
        },
    }


def test_residual_static_apply_job_end_to_end_synthetic_trace_store(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arrays = _install_synthetic_inputs(client, tmp_path)

    def _run_synchronously(**kwargs: Any) -> None:
        target = kwargs['target']
        args = kwargs.get('args', ())
        call_kwargs = kwargs.get('kwargs') or {}
        target(*args, **call_kwargs)

    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        _run_synchronously,
    )

    response = client.post('/statics/residual/apply', json=_request_payload())

    assert response.status_code == 200
    job_id = response.json()['job_id']
    status_response = client.get(f'/statics/job/{job_id}/status')
    assert status_response.status_code == 200
    assert status_response.json()['state'] == 'done'
    assert status_response.json()['progress'] == pytest.approx(1.0)

    state = client.app.state.sv
    with state.lock:
        job = dict(state.jobs[job_id])
    job_dir = Path(str(job['artifacts_dir']))
    with np.load(job_dir / 'residual_static_solution.npz', allow_pickle=False) as data:
        estimated = np.asarray(data['estimated_trace_delay_s_sorted'])
        applied = np.asarray(data['applied_residual_shift_s_sorted'])
    np.testing.assert_allclose(
        estimated,
        arrays['estimated_delay_s'],
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(applied, -estimated, rtol=0.0, atol=1.0e-12)

    corrected_manifest = json.loads(
        (job_dir / 'corrected_file.json').read_text(encoding='utf-8')
    )
    corrected_file_id = corrected_manifest['file_id']
    assert corrected_file_id == job['corrected_file_id']
    assert state.file_registry.get_store_path(corrected_file_id) == (
        corrected_manifest['store_path']
    )

    reader = get_reader(corrected_file_id, KEY1_BYTE, KEY2_BYTE, state=state)
    corrected = np.asarray(reader.get_section(KEY1_VALUE).arr)
    assert corrected.shape == (arrays['estimated_delay_s'].shape[0], N_SAMPLES)
    np.testing.assert_array_equal(np.argmax(corrected, axis=1), np.full(12, 40))

    corrected_meta = json.loads(
        (Path(corrected_manifest['store_path']) / 'meta.json').read_text(
            encoding='utf-8'
        )
    )
    component_names = [
        component['name'] for component in corrected_meta['derived']['components']
    ]
    assert component_names == [
        'datum_static_correction',
        'residual_static_correction',
    ]
