from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.refraction_static_artifacts import (
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
)
from app.services.refraction_static_export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)

KEY1 = 189
KEY2 = 193
FILE_ID = 'pick-map-file'


@pytest.fixture()
def pick_map_client(tmp_path: Path):
    state = app.state.sv
    with state.lock:
        state.jobs.clear()
        state.file_registry.clear()
        state.cached_readers.clear()

    store = tmp_path / 'store'
    _write_trace_store(store)
    with state.lock:
        state.file_registry.update(FILE_ID, store_path=str(store), dt=0.004)

    with TestClient(app) as client:
        yield client, state, tmp_path

    with state.lock:
        state.jobs.clear()
        state.file_registry.clear()
        state.cached_readers.clear()


def test_refraction_qc_pick_map_from_uploaded_npz_before_statics(pick_map_client):
    client, _state, tmp_path = pick_map_client
    pick_path = tmp_path / 'picks.npz'
    np.savez(
        pick_path,
        first_break_time_s=np.asarray([0.084, -1.0, 0.091, 0.102]),
        n_traces=np.asarray(4),
        n_samples=np.asarray(10),
        dt=np.asarray(0.004),
        trace_order=np.asarray('trace_store_sorted'),
    )

    response = client.post(
        '/statics/refraction/qc/pick-map',
        data={
            'request_json': json.dumps(
                {
                    'file_id': FILE_ID,
                    'key1_byte': KEY1,
                    'key2_byte': KEY2,
                    'pick_source': {'kind': 'uploaded_npz'},
                    'geometry': {'receiver_number_mode': 'global_sequential'},
                }
            ),
        },
        files={'pick_npz': ('picks.npz', pick_path.read_bytes(), 'application/x-npz')},
    )

    assert response.status_code == 200
    body = response.json()
    assert body['mode'] == 'pre_statics'
    assert body.get('job_id') is None
    assert body['has_after_statics'] is False
    assert 'After Statics is unavailable' in body['status_message']
    assert body['gather_range'] == {'min': 100, 'max': 101}
    pick_map = body['pick_map']
    assert pick_map['gather_id'] == [100, 101, 101]
    assert pick_map['receiver_number'] == [2000, 2000, 2001]
    assert pick_map['pick_before_ms'] == pytest.approx([84.0, 91.0, 102.0])
    assert pick_map['pick_after_ms'] == [None, None, None]
    assert pick_map['used_in_statics'] == [None, None, None]
    assert pick_map['offset_m'] == pytest.approx([100.0, 100.0, 100.0])


def test_refraction_qc_pick_map_before_statics_does_not_require_job_id(
    pick_map_client,
):
    client, _state, tmp_path = pick_map_client
    pick_path = tmp_path / 'picks.npz'
    np.savez(
        pick_path,
        first_break_time_s=np.asarray([0.084, 0.086, 0.091, 0.102]),
        trace_order=np.asarray('trace_store_sorted'),
    )

    response = client.post(
        '/statics/refraction/qc/pick-map',
        data={
            'request_json': json.dumps(
                {
                    'file_id': FILE_ID,
                    'pick_source': {'kind': 'uploaded_npz'},
                }
            ),
        },
        files={'pick_npz': ('picks.npz', pick_path.read_bytes(), 'application/x-npz')},
    )

    assert response.status_code == 200
    assert response.json()['mode'] == 'pre_statics'


def test_pick_map_pre_statics_accepts_nan_missing_picks_and_plots_valid_only(
    pick_map_client,
):
    client, _state, tmp_path = pick_map_client
    pick_path = tmp_path / 'picks-nan.npz'
    np.savez(
        pick_path,
        first_break_time_s=np.asarray([0.084, np.nan, np.inf, 0.102]),
        trace_order=np.asarray('trace_store_sorted'),
    )

    response = client.post(
        '/statics/refraction/qc/pick-map',
        data={
            'request_json': json.dumps(
                {
                    'file_id': FILE_ID,
                    'pick_source': {'kind': 'uploaded_npz'},
                }
            ),
        },
        files={'pick_npz': ('picks.npz', pick_path.read_bytes(), 'application/x-npz')},
    )

    assert response.status_code == 200
    pick_map = response.json()['pick_map']
    assert pick_map['trace_index'] == [0, 3]
    assert pick_map['pick_before_ms'] == pytest.approx([84.0, 102.0])


def test_pick_map_pre_statics_accepts_negative_sentinel_and_plots_valid_only(
    pick_map_client,
):
    client, _state, tmp_path = pick_map_client
    pick_path = tmp_path / 'picks-sentinel.npz'
    np.savez(
        pick_path,
        first_break_time_s=np.asarray([0.084, -1.0, 0.091, -999.25]),
        trace_order=np.asarray('trace_store_sorted'),
    )

    response = client.post(
        '/statics/refraction/qc/pick-map',
        data={
            'request_json': json.dumps(
                {
                    'file_id': FILE_ID,
                    'pick_source': {'kind': 'uploaded_npz'},
                }
            ),
        },
        files={'pick_npz': ('picks.npz', pick_path.read_bytes(), 'application/x-npz')},
    )

    assert response.status_code == 200
    pick_map = response.json()['pick_map']
    assert pick_map['trace_index'] == [0, 2]
    assert pick_map['pick_before_ms'] == pytest.approx([84.0, 91.0])


def test_pick_map_pre_statics_applies_negative_coordinate_scalar(
    pick_map_client,
):
    client, _state, tmp_path = pick_map_client
    np.save(tmp_path / 'store' / 'headers_byte_71.npy', np.full(4, -10, dtype=np.int32))
    pick_path = tmp_path / 'picks-scaled.npz'
    np.savez(
        pick_path,
        first_break_time_s=np.asarray([0.084, 0.086, 0.091, 0.102]),
        trace_order=np.asarray('trace_store_sorted'),
    )

    response = client.post(
        '/statics/refraction/qc/pick-map',
        data={
            'request_json': json.dumps(
                {
                    'file_id': FILE_ID,
                    'pick_source': {'kind': 'uploaded_npz'},
                    'geometry': {
                        'receiver_number_mode': 'global_sequential',
                        'coordinate_scalar_byte': 71,
                    },
                }
            ),
        },
        files={'pick_npz': ('picks.npz', pick_path.read_bytes(), 'application/x-npz')},
    )

    assert response.status_code == 200
    assert response.json()['pick_map']['offset_m'] == pytest.approx([10.0] * 4)


def test_pick_map_pre_statics_converts_coordinate_feet_to_meters(
    pick_map_client,
):
    client, _state, tmp_path = pick_map_client
    pick_path = tmp_path / 'picks-feet.npz'
    np.savez(
        pick_path,
        first_break_time_s=np.asarray([0.084, 0.086, 0.091, 0.102]),
        trace_order=np.asarray('trace_store_sorted'),
    )

    response = client.post(
        '/statics/refraction/qc/pick-map',
        data={
            'request_json': json.dumps(
                {
                    'file_id': FILE_ID,
                    'pick_source': {'kind': 'uploaded_npz'},
                    'geometry': {
                        'receiver_number_mode': 'global_sequential',
                        'coordinate_unit': 'ft',
                    },
                }
            ),
        },
        files={'pick_npz': ('picks.npz', pick_path.read_bytes(), 'application/x-npz')},
    )

    assert response.status_code == 200
    assert response.json()['pick_map']['offset_m'] == pytest.approx([30.48] * 4)


def test_refraction_qc_pick_map_completed_job_includes_before_and_after_picks(
    pick_map_client,
):
    client, state, tmp_path = pick_map_client
    artifacts_dir = tmp_path / 'artifacts'
    artifacts_dir.mkdir()
    _write_completed_pick_map_artifacts(artifacts_dir)
    job_id = 'pick-map-static-job'
    with state.lock:
        state.jobs.create_static_job(
            job_id,
            file_id=FILE_ID,
            key1_byte=KEY1,
            key2_byte=KEY2,
            statics_kind='refraction',
            artifacts_dir=str(artifacts_dir),
        )
        state.jobs.mark_done(job_id, progress_1=True)

    response = client.post(
        '/statics/refraction/qc/pick-map',
        json={'job_id': job_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body['mode'] == 'completed_job'
    assert body['has_after_statics'] is True
    pick_map = body['pick_map']
    assert pick_map['pick_before_ms'] == pytest.approx([84.0, 91.0, 102.0])
    assert pick_map['receiver_number'] == [2000, 2001, 2000]
    assert pick_map['applied_shift_ms'] == pytest.approx([-10.0, 5.0, 0.0])
    assert pick_map['pick_after_ms'] == pytest.approx([74.0, 96.0, 102.0])
    assert pick_map['used_in_statics'] == [True, False, True]
    assert pick_map['offset_used'] == [120.0, None, 300.0]


def test_pick_map_completed_job_gather_range_reports_endpoint_key_numeric_suffix(
    pick_map_client,
):
    client, state, tmp_path = pick_map_client
    artifacts_dir = tmp_path / 'endpoint-artifacts'
    artifacts_dir.mkdir()
    (artifacts_dir / REFRACTION_STATIC_QC_JSON_NAME).write_text(
        json.dumps({'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION}),
        encoding='utf-8',
    )
    _write_csv(
        artifacts_dir / REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
        [
            {
                'trace_index_sorted': '0',
                'source_endpoint_key': 'source:100',
                'receiver_endpoint_key': 'receiver:2000',
                'offset_m': '120.0',
                'observed_first_break_time_s': '0.084',
                'used_in_solve': 'true',
            },
            {
                'trace_index_sorted': '1',
                'source_endpoint_key': 'source:101:10:20',
                'receiver_endpoint_key': 'receiver:2001',
                'offset_m': '220.0',
                'observed_first_break_time_s': '0.091',
                'used_in_solve': 'true',
            },
        ],
    )
    _write_csv(
        artifacts_dir / REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
        [
            {'trace_index_sorted': '0', 'applied_trace_shift_ms': '0.0'},
            {'trace_index_sorted': '1', 'applied_trace_shift_ms': '0.0'},
        ],
    )
    job_id = 'pick-map-endpoint-job'
    with state.lock:
        state.jobs.create_static_job(
            job_id,
            file_id=FILE_ID,
            key1_byte=KEY1,
            key2_byte=KEY2,
            statics_kind='refraction',
            artifacts_dir=str(artifacts_dir),
        )
        state.jobs.mark_done(job_id, progress_1=True)

    response = client.post('/statics/refraction/qc/pick-map', json={'job_id': job_id})

    assert response.status_code == 200
    body = response.json()
    assert body['gather_range'] == {'min': 100, 'max': 101}
    assert body['pick_map']['gather_id'] == ['source:100', 'source:101:10:20']


def _write_trace_store(store: Path) -> None:
    store.mkdir(parents=True)
    traces = np.zeros((4, 10), dtype=np.float32)
    np.save(store / 'traces.npy', traces)
    np.savez(
        store / 'index.npz',
        key1_values=np.asarray([100, 101], dtype=np.int64),
        key1_offsets=np.asarray([0, 2], dtype=np.int64),
        key1_counts=np.asarray([2, 2], dtype=np.int64),
        sorted_to_original=np.arange(4, dtype=np.int64),
    )
    np.save(store / f'headers_byte_{KEY1}.npy', np.asarray([100, 100, 101, 101]))
    np.save(store / f'headers_byte_{KEY2}.npy', np.asarray([1, 2, 1, 2]))
    np.save(store / 'headers_byte_9.npy', np.asarray([100, 100, 101, 101]))
    np.save(store / 'headers_byte_13.npy', np.asarray([2000, 2001, 2000, 2001]))
    np.save(store / 'headers_byte_73.npy', np.asarray([0, 0, 1000, 1000]))
    np.save(store / 'headers_byte_77.npy', np.asarray([0, 0, 0, 0]))
    np.save(store / 'headers_byte_81.npy', np.asarray([100, 100, 1100, 1100]))
    np.save(store / 'headers_byte_85.npy', np.asarray([0, 0, 0, 0]))
    np.save(store / 'headers_byte_45.npy', np.asarray([10, 10, 10, 10]))
    np.save(store / 'headers_byte_41.npy', np.asarray([20, 20, 20, 20]))
    np.save(store / 'headers_byte_71.npy', np.ones(4, dtype=np.int32))
    np.save(store / 'headers_byte_69.npy', np.ones(4, dtype=np.int32))
    (store / 'meta.json').write_text(
        json.dumps(
            {
                'schema_version': 1,
                'dtype': 'float32',
                'n_traces': 4,
                'n_samples': 10,
                'key_bytes': {'key1': KEY1, 'key2': KEY2},
                'sorted_by': ['key1', 'key2'],
                'dt': 0.004,
            }
        ),
        encoding='utf-8',
    )


def _write_completed_pick_map_artifacts(artifacts_dir: Path) -> None:
    (artifacts_dir / REFRACTION_STATIC_QC_JSON_NAME).write_text(
        json.dumps({'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION}),
        encoding='utf-8',
    )
    _write_csv(
        artifacts_dir / REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
        [
            {
                'trace_index_sorted': '0',
                'source_id': '100',
                'receiver_endpoint_key': 'receiver:2000',
                'offset_m': '120.0',
                'observed_first_break_time_s': '0.084',
                'used_in_solve': 'true',
            },
            {
                'trace_index_sorted': '1',
                'source_id': '100',
                'receiver_endpoint_key': 'receiver:2001:10:0:20',
                'offset_m': '220.0',
                'observed_first_break_time_s': '0.091',
                'used_in_solve': 'false',
            },
            {
                'trace_index_sorted': '2',
                'source_id': '101',
                'receiver_endpoint_key': 'receiver:2000',
                'offset_m': '300.0',
                'observed_first_break_time_s': '0.102',
                'used_in_solve': 'true',
            },
        ],
    )
    _write_csv(
        artifacts_dir / REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
        [
            {'trace_index_sorted': '0', 'applied_trace_shift_ms': '-10.0'},
            {'trace_index_sorted': '1', 'applied_trace_shift_ms': '5.0'},
            {'trace_index_sorted': '2', 'applied_trace_shift_ms': '0.0'},
        ],
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
