from __future__ import annotations

import csv
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
from app.main import app
from app.services.geometry_linkage_loader import load_geometry_linkage_from_job_dir
from app.services.reader import get_reader
from app.services.trace_store_registration import trace_store_cache_key
from app.tests._stubs import write_baseline_raw

KEY1_BYTE = 189
KEY2_BYTE = 193
KEY1_VALUE = 7001

SOURCE_ID_BYTE = 9
RECEIVER_ID_BYTE = 13
SOURCE_X_BYTE = 73
SOURCE_Y_BYTE = 77
RECEIVER_X_BYTE = 81
RECEIVER_Y_BYTE = 85
SOURCE_ELEVATION_BYTE = 45
RECEIVER_ELEVATION_BYTE = 41
COORDINATE_SCALAR_BYTE = 71
ELEVATION_SCALAR_BYTE = 69
OFFSET_BYTE = 37

DT = 0.004
N_SAMPLES = 128
IMPULSE_SAMPLE = 64
REPEATS_PER_PAIR = 10
REFRACTOR_VELOCITY_M_S = 4000.0
REPLACEMENT_VELOCITY_M_S = 2000.0

FILE_ID = 'synthetic-time-term-source'
PICK_JOB_ID = 'synthetic-time-term-picks'

EXPECTED_TIME_TERM_FILES = {
    'job_meta.json',
    'time_term_static_solution.npz',
    'time_term_static_qc.json',
    'time_term_statics.csv',
    'corrected_file.json',
}


@dataclass(frozen=True)
class SyntheticTimeTermDataset:
    file_id: str
    store_dir: Path
    key1_byte: int
    key2_byte: int
    key1_value: int
    dt: float
    n_traces: int
    n_samples: int
    original_segy_path: str
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    true_node_time_term_s: np.ndarray
    true_trace_delay_s_sorted: np.ndarray
    true_applied_weathering_shift_s_sorted: np.ndarray
    pick_time_raw_s_sorted: np.ndarray
    header_bytes: dict[int, np.ndarray]


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(tmp_path / 'jobs'))
    _clear_state()
    with TestClient(app) as test_client:
        yield test_client
    _clear_state()


@pytest.fixture()
def sync_static_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(statics_router_module, 'start_job_thread', _run_job_sync)


def test_time_term_static_e2e_recovers_known_delays_and_registers_corrected_store(
    client: TestClient,
    sync_static_jobs: None,
    tmp_path: Path,
) -> None:
    dataset = _install_synthetic_time_term_dataset(client, tmp_path)
    linkage_job_id = _run_geometry_linkage_job(client, dataset)

    response = client.post(
        '/statics/time-term/apply',
        json=_time_term_apply_payload(linkage_job_id, robust_enabled=False),
    )

    assert response.status_code == 200
    assert response.json()['state'] == 'queued'
    job_id = response.json()['job_id']
    status = client.get(f'/statics/job/{job_id}/status')
    assert status.status_code == 200
    assert status.json() == {'state': 'done', 'progress': 1.0, 'message': 'done'}

    job_dir = _job_dir(client, job_id)
    solution_path = job_dir / 'time_term_static_solution.npz'
    qc_path = job_dir / 'time_term_static_qc.json'
    csv_path = job_dir / 'time_term_statics.csv'
    corrected_file_path = job_dir / 'corrected_file.json'

    with np.load(solution_path, allow_pickle=False) as solution:
        _assert_no_object_dtype(solution)
        assert str(solution['gauge_mode']) == 'reference_node'
        node_time_term = np.asarray(solution['node_time_term_s'])
        estimated_delay = np.asarray(
            solution['estimated_trace_time_term_delay_s_sorted']
        )
        applied_weathering = np.asarray(
            solution['applied_weathering_shift_s_sorted']
        )
        final_shift = np.asarray(solution['final_trace_shift_s_sorted'])
        np.testing.assert_allclose(
            node_time_term,
            dataset.true_node_time_term_s,
            rtol=0.0,
            atol=1.0e-4,
        )
        np.testing.assert_allclose(
            estimated_delay,
            dataset.true_trace_delay_s_sorted,
            rtol=0.0,
            atol=1.0e-4,
        )
        np.testing.assert_allclose(
            applied_weathering,
            -estimated_delay,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(final_shift, applied_weathering, atol=1.0e-12)
        assert int(solution['n_traces']) == dataset.n_traces
        assert str(solution['robust_stop_reason']) == 'disabled'

    qc = json.loads(qc_path.read_text(encoding='utf-8'))
    json.dumps(qc, allow_nan=False)
    assert qc['counts']['n_rejected_traces'] == 0
    assert qc['robust']['enabled'] is False
    assert qc['time_terms']['estimated_trace_time_term_delay_ms']['count'] == (
        dataset.n_traces
    )
    assert qc['components']['final_trace_shift_ms']['count'] == dataset.n_traces

    rows = _read_time_term_csv(csv_path)
    assert len(rows) == dataset.n_traces
    assert {'sorted_trace_index', 'source_node_id', 'receiver_node_id'}.issubset(
        rows[0]
    )

    _assert_download_round_trip(client, job_id, dataset.n_traces)
    _assert_corrected_trace_store_registered(
        client,
        corrected_file_path,
        dataset,
    )

    assert not Path(dataset.original_segy_path).exists()
    _assert_corrected_impulse_shift_sign(client, corrected_file_path, dataset)


def test_time_term_static_e2e_rejects_outlier_and_updates_qc(
    client: TestClient,
    sync_static_jobs: None,
    tmp_path: Path,
) -> None:
    outlier_trace = 5
    dataset = _install_synthetic_time_term_dataset(
        client,
        tmp_path,
        outlier_trace=outlier_trace,
    )
    linkage_job_id = _run_geometry_linkage_job(client, dataset)

    response = client.post(
        '/statics/time-term/apply',
        json=_time_term_apply_payload(linkage_job_id, robust_enabled=True),
    )

    assert response.status_code == 200
    job_id = response.json()['job_id']
    assert client.get(f'/statics/job/{job_id}/status').json()['state'] == 'done'
    job_dir = _job_dir(client, job_id)

    with np.load(
        job_dir / 'time_term_static_solution.npz',
        allow_pickle=False,
    ) as solution:
        rejected = np.asarray(solution['rejected_trace_mask_sorted'])
        rejected_iteration = np.asarray(solution['rejected_iteration_sorted'])
        final_used = np.asarray(solution['final_used_trace_mask_sorted'])
        estimated_delay = np.asarray(
            solution['estimated_trace_time_term_delay_s_sorted']
        )
        assert bool(rejected[outlier_trace]) is True
        assert int(rejected_iteration[outlier_trace]) >= 0
        assert bool(final_used[outlier_trace]) is False
        assert int(np.count_nonzero(rejected)) == 1
        non_outlier_mask = np.ones(dataset.n_traces, dtype=bool)
        non_outlier_mask[outlier_trace] = False
        np.testing.assert_allclose(
            estimated_delay[non_outlier_mask],
            dataset.true_trace_delay_s_sorted[non_outlier_mask],
            rtol=0.0,
            atol=1.0e-4,
        )
        assert str(solution['robust_stop_reason']) in {
            'converged',
            'zero_scale',
            'max_iterations',
        }

    qc = json.loads(
        (job_dir / 'time_term_static_qc.json').read_text(encoding='utf-8')
    )
    assert qc['robust']['enabled'] is True
    assert qc['counts']['n_rejected_traces'] == 1
    assert qc['robust']['n_iterations'] >= 1
    assert any(
        item['n_rejected_this_iteration'] >= 1
        for item in qc['robust']['iterations']
    )
    assert qc['time_terms']['applied_weathering_shift_ms']['count'] == (
        dataset.n_traces
    )

    rows = _read_time_term_csv(job_dir / 'time_term_statics.csv')
    assert len(rows) == dataset.n_traces
    assert rows[outlier_trace]['rejected'] == 'true'
    assert rows[outlier_trace]['final_used'] == 'false'


def test_time_term_static_e2e_rejects_linkage_n_traces_mismatch(
    client: TestClient,
    sync_static_jobs: None,
    tmp_path: Path,
) -> None:
    dataset = _install_synthetic_time_term_dataset(client, tmp_path)
    linkage_job_id = _run_geometry_linkage_job(client, dataset)
    linkage_path = _job_dir(client, linkage_job_id) / 'geometry_linkage.npz'
    _truncate_linkage_trace_mapping(linkage_path)

    response = client.post(
        '/statics/time-term/apply',
        json=_time_term_apply_payload(linkage_job_id, robust_enabled=False),
    )

    assert response.status_code == 200
    job_id = response.json()['job_id']
    status = client.get(f'/statics/job/{job_id}/status')
    assert status.status_code == 200
    body = status.json()
    assert body['state'] == 'error'
    assert 'source_node_id_sorted shape mismatch' in body['message']

    job_dir = _job_dir(client, job_id)
    assert not (job_dir / 'time_term_static_solution.npz').exists()
    assert not (job_dir / 'corrected_file.json').exists()
    assert not any(dataset.store_dir.parent.glob(f'{dataset.store_dir.name}.statics.time_term.*'))


def _clear_state() -> None:
    state = app.state.sv
    with state.lock:
        state.jobs.clear()
        state.cached_readers.clear()
        state.file_registry.clear()
        state.pipeline_tap_cache.clear()
        state.window_section_cache.clear()
        state.section_offsets_cache.clear()
        state.trace_stats_cache.clear()


def _run_job_sync(**kwargs: Any) -> object:
    target = kwargs['target']
    args = kwargs.get('args', ())
    extra_kwargs = kwargs.get('kwargs') or {}
    target(*args, **extra_kwargs)
    return object()


def _install_synthetic_time_term_dataset(
    client: TestClient,
    tmp_path: Path,
    *,
    outlier_trace: int | None = None,
) -> SyntheticTimeTermDataset:
    dataset = _write_synthetic_time_term_trace_store(
        tmp_path,
        outlier_trace=outlier_trace,
    )
    _write_pick_job_artifact(client, tmp_path, dataset)
    client.app.state.sv.file_registry.update(
        dataset.file_id,
        store_path=dataset.store_dir,
        dt=dataset.dt,
    )
    return dataset


def _write_synthetic_time_term_trace_store(
    tmp_path: Path,
    *,
    outlier_trace: int | None,
) -> SyntheticTimeTermDataset:
    store_dir = tmp_path / 'synthetic-time-term-trace-store'
    store_dir.mkdir(parents=True)

    true_node_time_term_s = np.asarray([0.0, 0.004, 0.008], dtype=np.float64)
    node_x_m = np.asarray([0, 100, 200], dtype=np.int64)
    source_node_id, receiver_node_id = _trace_node_pairs(REPEATS_PER_PAIR)
    n_traces = int(source_node_id.shape[0])
    source_x = node_x_m[source_node_id]
    receiver_x = node_x_m[receiver_node_id]
    moveout_s = np.abs(receiver_x - source_x).astype(np.float64) / (
        REFRACTOR_VELOCITY_M_S
    )
    true_trace_delay_s = (
        true_node_time_term_s[source_node_id]
        + true_node_time_term_s[receiver_node_id]
    )
    pick_time_raw_s = moveout_s + true_trace_delay_s
    if outlier_trace is not None:
        pick_time_raw_s = pick_time_raw_s.copy()
        pick_time_raw_s[int(outlier_trace)] += 0.080

    traces = np.zeros((n_traces, N_SAMPLES), dtype=np.float32)
    traces[:, IMPULSE_SAMPLE] = 1.0
    np.save(store_dir / 'traces.npy', traces)
    np.savez(
        store_dir / 'index.npz',
        key1_values=np.asarray([KEY1_VALUE], dtype=np.int64),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([n_traces], dtype=np.int64),
        sorted_to_original=np.arange(n_traces, dtype=np.int64),
    )

    original_segy_path = str(tmp_path / 'missing-original.sgy')
    (store_dir / 'meta.json').write_text(
        json.dumps(
            {
                'schema_version': 1,
                'dtype': 'float32',
                'n_traces': n_traces,
                'n_samples': N_SAMPLES,
                'key_bytes': {'key1': KEY1_BYTE, 'key2': KEY2_BYTE},
                'sorted_by': ['key1', 'key2'],
                'dt': DT,
                'scale': None,
                'original_name': 'synthetic-time-term.sgy',
                'original_segy_path': original_segy_path,
                'source_sha256': None,
                'derived': {
                    'kind': 'time_shifted_trace_store',
                    'from_file_id': 'synthetic-raw-source',
                    'components': [
                        _datum_component(),
                        _residual_component(),
                    ],
                },
            },
            sort_keys=True,
        ),
        encoding='utf-8',
    )

    header_bytes = {
        KEY1_BYTE: np.full(n_traces, KEY1_VALUE, dtype=np.int64),
        KEY2_BYTE: np.arange(1, n_traces + 1, dtype=np.int64),
        SOURCE_ID_BYTE: 100 + source_node_id,
        RECEIVER_ID_BYTE: 200 + receiver_node_id,
        SOURCE_X_BYTE: source_x,
        SOURCE_Y_BYTE: np.zeros(n_traces, dtype=np.int64),
        RECEIVER_X_BYTE: receiver_x,
        RECEIVER_Y_BYTE: np.zeros(n_traces, dtype=np.int64),
        SOURCE_ELEVATION_BYTE: np.zeros(n_traces, dtype=np.int64),
        RECEIVER_ELEVATION_BYTE: np.zeros(n_traces, dtype=np.int64),
        COORDINATE_SCALAR_BYTE: np.ones(n_traces, dtype=np.int64),
        ELEVATION_SCALAR_BYTE: np.ones(n_traces, dtype=np.int64),
        OFFSET_BYTE: np.abs(receiver_x - source_x).astype(np.int64),
    }
    for byte, values in header_bytes.items():
        np.save(store_dir / f'headers_byte_{byte}.npy', np.asarray(values))

    write_baseline_raw(
        store_dir,
        key1=KEY1_VALUE,
        key1_byte=KEY1_BYTE,
        key2_byte=KEY2_BYTE,
        source_sha256=None,
        n_traces=n_traces,
    )

    return SyntheticTimeTermDataset(
        file_id=FILE_ID,
        store_dir=store_dir,
        key1_byte=KEY1_BYTE,
        key2_byte=KEY2_BYTE,
        key1_value=KEY1_VALUE,
        dt=DT,
        n_traces=n_traces,
        n_samples=N_SAMPLES,
        original_segy_path=original_segy_path,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        true_node_time_term_s=true_node_time_term_s,
        true_trace_delay_s_sorted=true_trace_delay_s,
        true_applied_weathering_shift_s_sorted=-true_trace_delay_s,
        pick_time_raw_s_sorted=pick_time_raw_s,
        header_bytes=header_bytes,
    )


def _trace_node_pairs(repeats: int) -> tuple[np.ndarray, np.ndarray]:
    source_node_id: list[int] = []
    receiver_node_id: list[int] = []
    for _repeat in range(repeats):
        for source_node in range(3):
            for receiver_node in range(3):
                source_node_id.append(source_node)
                receiver_node_id.append(receiver_node)
    return (
        np.asarray(source_node_id, dtype=np.int64),
        np.asarray(receiver_node_id, dtype=np.int64),
    )


def _datum_component() -> dict[str, object]:
    return {
        'name': 'datum_static_correction',
        'job_id': 'synthetic-datum-job',
        'solution_artifact': 'datum_static_solution.npz',
        'shift_field': 'trace_shift_s_sorted',
        'value_kind': 'applied_event_time_shift_s',
    }


def _residual_component() -> dict[str, object]:
    return {
        'name': 'residual_static_correction',
        'job_id': 'synthetic-residual-job',
        'solution_artifact': 'residual_static_solution.npz',
        'shift_field': 'applied_residual_shift_s_sorted',
        'value_kind': 'applied_event_time_shift_s',
    }


def _write_pick_job_artifact(
    client: TestClient,
    tmp_path: Path,
    dataset: SyntheticTimeTermDataset,
) -> None:
    pick_job_dir = tmp_path / PICK_JOB_ID
    pick_job_dir.mkdir()
    np.savez(
        pick_job_dir / 'predicted_picks_time_s.npz',
        picks_time_s=dataset.pick_time_raw_s_sorted.astype(np.float64),
        sorted_to_original=np.arange(dataset.n_traces, dtype=np.int64),
        n_traces=np.asarray(dataset.n_traces, dtype=np.int64),
        n_samples=np.asarray(dataset.n_samples, dtype=np.int64),
        dt=np.asarray(dataset.dt, dtype=np.float64),
    )
    state = client.app.state.sv
    with state.lock:
        state.jobs.create_batch_apply_job(
            PICK_JOB_ID,
            file_id=dataset.file_id,
            key1_byte=dataset.key1_byte,
            key2_byte=dataset.key2_byte,
            artifacts_dir=str(pick_job_dir),
        )
        state.jobs.mark_done(PICK_JOB_ID, progress_1=True)


def _run_geometry_linkage_job(
    client: TestClient,
    dataset: SyntheticTimeTermDataset,
) -> str:
    response = client.post(
        '/statics/linkage/build',
        json={
            'file_id': dataset.file_id,
            'key1_byte': dataset.key1_byte,
            'key2_byte': dataset.key2_byte,
            'geometry': {
                'source_x_byte': SOURCE_X_BYTE,
                'source_y_byte': SOURCE_Y_BYTE,
                'receiver_x_byte': RECEIVER_X_BYTE,
                'receiver_y_byte': RECEIVER_Y_BYTE,
                'coordinate_scalar_byte': COORDINATE_SCALAR_BYTE,
            },
            'linkage': {
                'mode': 'auto_threshold',
                'threshold_m': 1.0,
                'prefer_receiver_anchor': True,
            },
        },
    )
    assert response.status_code == 200
    job_id = response.json()['job_id']
    assert client.get(f'/statics/job/{job_id}/status').json()['state'] == 'done'

    loaded = load_geometry_linkage_from_job_dir(
        _job_dir(client, job_id),
        expected_n_traces=dataset.n_traces,
        expected_key1_byte=dataset.key1_byte,
        expected_key2_byte=dataset.key2_byte,
    )
    np.testing.assert_array_equal(
        loaded.source_node_id_sorted,
        dataset.source_node_id_sorted,
    )
    np.testing.assert_array_equal(
        loaded.receiver_node_id_sorted,
        dataset.receiver_node_id_sorted,
    )
    assert loaded.n_nodes == 3
    return job_id


def _time_term_apply_payload(
    linkage_job_id: str,
    *,
    robust_enabled: bool,
) -> dict[str, Any]:
    return {
        'file_id': FILE_ID,
        'key1_byte': KEY1_BYTE,
        'key2_byte': KEY2_BYTE,
        'pick_source': {
            'kind': 'batch_predicted_npz',
            'job_id': PICK_JOB_ID,
            'artifact_name': 'predicted_picks_time_s.npz',
        },
        'geometry': {
            'source_id_byte': SOURCE_ID_BYTE,
            'receiver_id_byte': RECEIVER_ID_BYTE,
            'source_x_byte': SOURCE_X_BYTE,
            'source_y_byte': SOURCE_Y_BYTE,
            'receiver_x_byte': RECEIVER_X_BYTE,
            'receiver_y_byte': RECEIVER_Y_BYTE,
            'source_elevation_byte': SOURCE_ELEVATION_BYTE,
            'receiver_elevation_byte': RECEIVER_ELEVATION_BYTE,
            'source_depth_byte': None,
            'coordinate_scalar_byte': COORDINATE_SCALAR_BYTE,
            'elevation_scalar_byte': ELEVATION_SCALAR_BYTE,
            'coordinate_unit': 'm',
            'elevation_unit': 'm',
        },
        'linkage': {
            'mode': 'required',
            'job_id': linkage_job_id,
            'artifact_name': 'geometry_linkage.npz',
        },
        'velocity': {
            'replacement_velocity_m_s': REPLACEMENT_VELOCITY_M_S,
            'refractor_velocity_m_s': REFRACTOR_VELOCITY_M_S,
            'weathering_velocity_m_s': None,
        },
        'moveout': {
            'model': 'head_wave_linear_offset',
            'distance_source': 'geometry',
            'offset_byte': OFFSET_BYTE,
            'allow_missing_offset': False,
        },
        'solver': {
            'damping': 0.0,
            'gauge': 'reference_node',
            'reference_node_id': 0,
            'robust': {
                'enabled': robust_enabled,
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
            'register_corrected_file': True,
            'max_abs_shift_ms': 250.0,
            'output_dtype': 'float32',
        },
    }


def _job_dir(client: TestClient, job_id: str) -> Path:
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    return Path(str(job['artifacts_dir']))


def _assert_no_object_dtype(solution: np.lib.npyio.NpzFile) -> None:
    object_arrays = [
        name for name in solution.files if np.asarray(solution[name]).dtype == object
    ]
    assert object_arrays == []


def _truncate_linkage_trace_mapping(linkage_path: Path) -> None:
    with np.load(linkage_path, allow_pickle=False) as linkage:
        arrays = {name: np.asarray(linkage[name]) for name in linkage.files}
    arrays['source_node_id_sorted'] = arrays['source_node_id_sorted'][:-1]
    arrays['receiver_node_id_sorted'] = arrays['receiver_node_id_sorted'][:-1]
    np.savez(linkage_path, **arrays)


def _read_time_term_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _assert_download_round_trip(
    client: TestClient,
    job_id: str,
    expected_rows: int,
) -> None:
    files_response = client.get(f'/statics/job/{job_id}/files')
    assert files_response.status_code == 200
    assert {item['name'] for item in files_response.json()['files']} == (
        EXPECTED_TIME_TERM_FILES
    )

    solution_response = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': 'time_term_static_solution.npz'},
    )
    assert solution_response.status_code == 200
    with np.load(BytesIO(solution_response.content), allow_pickle=False) as solution:
        assert int(solution['n_traces']) == expected_rows
        _assert_no_object_dtype(solution)

    qc_response = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': 'time_term_static_qc.json'},
    )
    assert qc_response.status_code == 200
    json.dumps(qc_response.json(), allow_nan=False)

    csv_response = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': 'time_term_statics.csv'},
    )
    assert csv_response.status_code == 200
    csv_rows = list(csv.DictReader(csv_response.text.splitlines()))
    assert len(csv_rows) == expected_rows
    assert {'sorted_trace_index', 'source_id', 'receiver_id'}.issubset(csv_rows[0])

    corrected_response = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': 'corrected_file.json'},
    )
    assert corrected_response.status_code == 200
    assert corrected_response.json()['derived_by'] == 'time_term_static_correction'

    traversal_response = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': '../time_term_static_solution.npz'},
    )
    assert traversal_response.status_code == 400
    missing_response = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': 'missing.npz'},
    )
    assert missing_response.status_code == 404


def _assert_corrected_trace_store_registered(
    client: TestClient,
    corrected_file_path: Path,
    dataset: SyntheticTimeTermDataset,
) -> None:
    corrected_file = json.loads(corrected_file_path.read_text(encoding='utf-8'))
    corrected_file_id = corrected_file['file_id']
    state = client.app.state.sv

    with state.lock:
        assert state.file_registry.get_record(corrected_file_id) is not None
        assert trace_store_cache_key(
            corrected_file_id,
            dataset.key1_byte,
            dataset.key2_byte,
        ) in state.cached_readers
    assert Path(corrected_file['store_path']).is_dir()
    assert corrected_file['applied_shift_field'] == (
        'applied_weathering_shift_s_sorted'
    )

    reader = get_reader(
        corrected_file_id,
        dataset.key1_byte,
        dataset.key2_byte,
        state=state,
    )
    section = np.asarray(reader.get_section(dataset.key1_value).arr)
    assert section.shape == (dataset.n_traces, dataset.n_samples)

    meta = json.loads(
        (Path(corrected_file['store_path']) / 'meta.json').read_text(
            encoding='utf-8'
        )
    )
    assert meta['source_sha256'] is None
    assert meta['original_segy_path'] == dataset.original_segy_path
    components = meta['derived']['components']
    time_term_components = [
        component
        for component in components
        if component['name'] == 'time_term_static_correction'
    ]
    assert len(time_term_components) == 1
    assert time_term_components[0]['shift_field'] == (
        'applied_weathering_shift_s_sorted'
    )


def _assert_corrected_impulse_shift_sign(
    client: TestClient,
    corrected_file_path: Path,
    dataset: SyntheticTimeTermDataset,
) -> None:
    corrected_file = json.loads(corrected_file_path.read_text(encoding='utf-8'))
    corrected_file_id = corrected_file['file_id']
    reader = get_reader(
        corrected_file_id,
        dataset.key1_byte,
        dataset.key2_byte,
        state=client.app.state.sv,
    )
    section = np.asarray(reader.get_section(dataset.key1_value).arr)

    zero_delay_trace = int(
        np.flatnonzero(np.isclose(dataset.true_trace_delay_s_sorted, 0.0))[0]
    )
    positive_delay_trace = int(
        np.flatnonzero(np.isclose(dataset.true_trace_delay_s_sorted, 0.008))[0]
    )
    assert int(np.argmax(np.abs(section[zero_delay_trace]))) == IMPULSE_SAMPLE
    assert int(np.argmax(np.abs(section[positive_delay_trace]))) == (
        IMPULSE_SAMPLE - 2
    )
