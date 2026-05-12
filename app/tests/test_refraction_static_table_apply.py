from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
from app.api.schemas import RefractionStaticTableApplyRequest
from app.core.state import AppState, create_app_state
from app.main import app
from app.services.refraction_static_export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)
from app.services.refraction_static_table_apply_service import (
    STATIC_TABLE_APPLY_HISTORY_JSON_NAME,
    STATIC_TABLE_APPLY_QC_JSON_NAME,
    STATIC_TABLE_APPLY_REFRACTION_HISTORY_JSON_NAME,
    STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME,
    STATIC_TABLE_APPLY_TRACE_SHIFTS_CSV_NAME,
    run_refraction_static_table_apply_job,
)
from app.services.refraction_static_apply_trace_store import CORRECTED_FILE_JSON_NAME
from app.services.refraction_static_artifacts import (
    REFRACTION_STATIC_REQUEST_JSON_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
)
from app.tests.test_refraction_static_apply_trace_store import (
    DT,
    KEY1,
    KEY2,
    SOURCE_FILE_ID,
    _write_source_store,
)

TABLE_JOB_ID = 'static-table-import-job'
APPLY_JOB_ID = 'static-table-apply-job'
SECOND_APPLY_JOB_ID = 'static-table-apply-job-2'


def _endpoint_key(
    endpoint_kind: str,
    endpoint_id: int,
    x_m: float,
    y_m: float,
    elevation_m: float,
) -> str:
    return (
        f'{endpoint_kind}:'
        f'{endpoint_id}:'
        f'{x_m:.17g}:'
        f'{y_m:.17g}:'
        f'{elevation_m:.17g}'
    )


SOURCE_KEYS = {
    100: _endpoint_key('source', 100, 1000.0, 2000.0, 10.0),
    101: _endpoint_key('source', 101, 1010.0, 2000.0, 12.0),
}
RECEIVER_KEYS = {
    200: _endpoint_key('receiver', 200, 1100.0, 2000.0, 20.0),
    201: _endpoint_key('receiver', 201, 1110.0, 2000.0, 22.0),
}
DEFAULT_GEOMETRY = {
    'source_id_byte': 9,
    'receiver_id_byte': 13,
    'source_x_byte': 73,
    'source_y_byte': 77,
    'receiver_x_byte': 81,
    'receiver_y_byte': 85,
    'source_elevation_byte': 45,
    'receiver_elevation_byte': 41,
    'coordinate_scalar_byte': 71,
    'elevation_scalar_byte': 69,
    'coordinate_unit': 'm',
    'elevation_unit': 'm',
}


def _write_header(store: Path, byte: int, values: list[int] | list[float]) -> None:
    np.save(store / f'headers_byte_{byte}.npy', np.asarray(values))


def _write_target_store(
    tmp_path: Path,
    *,
    sorted_to_original: np.ndarray | None = None,
) -> tuple[AppState, Path]:
    store = tmp_path / 'trace_stores' / 'line001.sgy'
    traces = np.zeros((4, 16), dtype=np.float32)
    traces[:, 8] = 1.0
    if sorted_to_original is None:
        sorted_to_original = np.arange(4, dtype=np.int64)
    _write_source_store(store, traces=traces, sorted_to_original=sorted_to_original)

    _write_header(store, 9, [100, 101, 100, 101])
    _write_header(store, 13, [200, 200, 201, 201])
    _write_header(store, 73, [1000, 1010, 1000, 1010])
    _write_header(store, 77, [2000, 2000, 2000, 2000])
    _write_header(store, 81, [1100, 1100, 1110, 1110])
    _write_header(store, 85, [2000, 2000, 2000, 2000])
    _write_header(store, 45, [10, 12, 10, 12])
    _write_header(store, 41, [20, 20, 22, 22])
    _write_header(store, 71, [1, 1, 1, 1])
    _write_header(store, 69, [1, 1, 1, 1])

    state = create_app_state()
    state.file_registry.update(SOURCE_FILE_ID, store_path=str(store), dt=DT)
    return state, store


def _canonical_row(
    *,
    endpoint_kind: str,
    endpoint_key: str,
    endpoint_id: int,
    applied_shift_ms: float,
) -> dict[str, str]:
    return {
        'format_name': 'canonical_static_table',
        'format_version': '1',
        'source_job_id': 'refraction-source-job',
        'endpoint_kind': endpoint_kind,
        'endpoint_key': endpoint_key,
        'endpoint_id': str(endpoint_id),
        'applied_shift_ms': f'{applied_shift_ms:.6f}',
        'static_status': 'ok',
        'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
    }


def _write_table(
    path: Path,
    *,
    include_source_101: bool = True,
    endpoint_id_matching_keys: bool = False,
) -> None:
    source_100_key = 'source-row-100' if endpoint_id_matching_keys else SOURCE_KEYS[100]
    source_101_key = 'source-row-101' if endpoint_id_matching_keys else SOURCE_KEYS[101]
    receiver_200_key = (
        'receiver-row-200' if endpoint_id_matching_keys else RECEIVER_KEYS[200]
    )
    receiver_201_key = (
        'receiver-row-201' if endpoint_id_matching_keys else RECEIVER_KEYS[201]
    )
    rows = [
        _canonical_row(
            endpoint_kind='source',
            endpoint_key=source_100_key,
            endpoint_id=100,
            applied_shift_ms=8.0,
        ),
        _canonical_row(
            endpoint_kind='receiver',
            endpoint_key=receiver_200_key,
            endpoint_id=200,
            applied_shift_ms=0.0,
        ),
        _canonical_row(
            endpoint_kind='receiver',
            endpoint_key=receiver_201_key,
            endpoint_id=201,
            applied_shift_ms=-4.0,
        ),
    ]
    if include_source_101:
        rows.insert(
            1,
            _canonical_row(
                endpoint_kind='source',
                endpoint_key=source_101_key,
                endpoint_id=101,
                applied_shift_ms=4.0,
            ),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _write_source_receiver_static_table_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sign_convention=np.asarray(REFRACTION_STATIC_REPO_SIGN_CONVENTION),
        source_endpoint_key=np.asarray(
            [SOURCE_KEYS[100], SOURCE_KEYS[101]],
            dtype='<U80',
        ),
        source_id=np.asarray([100, 101], dtype=np.int64),
        source_total_applied_shift_s=np.asarray([0.008, 0.004], dtype=np.float64),
        source_static_status=np.asarray(['ok', 'ok'], dtype='<U2'),
        receiver_endpoint_key=np.asarray(
            [RECEIVER_KEYS[200], RECEIVER_KEYS[201]],
            dtype='<U80',
        ),
        receiver_id=np.asarray([200, 201], dtype=np.int64),
        receiver_total_applied_shift_s=np.asarray([0.0, -0.004], dtype=np.float64),
        receiver_static_status=np.asarray(['ok', 'ok'], dtype='<U2'),
    )


def _write_refraction_static_request(
    job_dir: Path,
    *,
    geometry: dict[str, Any] | None = None,
) -> None:
    payload = {
        'job_id': TABLE_JOB_ID,
        'job_type': 'statics',
        'statics_kind': 'refraction',
        'source_file_id': SOURCE_FILE_ID,
        'key1_byte': KEY1,
        'key2_byte': KEY2,
        'request': {'geometry': geometry or DEFAULT_GEOMETRY},
    }
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / REFRACTION_STATIC_REQUEST_JSON_NAME).write_text(
        json.dumps(payload),
        encoding='utf-8',
    )


def _create_table_job(
    state: AppState,
    tmp_path: Path,
    table_path: Path,
    *,
    statics_kind: str = 'refraction_export',
) -> None:
    with state.lock:
        state.jobs.create_static_job(
            TABLE_JOB_ID,
            file_id=SOURCE_FILE_ID,
            key1_byte=KEY1,
            key2_byte=KEY2,
            statics_kind=statics_kind,
            artifacts_dir=str(table_path.parent),
        )


def _create_apply_job(
    state: AppState,
    job_dir: Path,
    *,
    job_id: str = APPLY_JOB_ID,
    file_id: str = SOURCE_FILE_ID,
) -> None:
    with state.lock:
        state.jobs.create_static_job(
            job_id,
            file_id=file_id,
            key1_byte=KEY1,
            key2_byte=KEY2,
            statics_kind='refraction_static_table_apply',
            artifacts_dir=str(job_dir),
        )


def _run_apply(
    tmp_path: Path,
    *,
    include_source_101: bool = True,
    register_corrected_file: bool = False,
    missing_static_policy: str = 'fail',
    allow_missing_source_static: bool = False,
    endpoint_id_matching: bool = False,
    write_producer_geometry: bool = True,
    lineage_components: list[str] | None = None,
    double_application_policy: str = 'warn',
) -> tuple[AppState, Path, Path]:
    state, store = _write_target_store(tmp_path)
    if lineage_components is not None:
        _write_source_lineage(store, lineage_components)
    table_path = tmp_path / 'jobs' / TABLE_JOB_ID / 'canonical_static_table.csv'
    _write_table(
        table_path,
        include_source_101=include_source_101,
        endpoint_id_matching_keys=endpoint_id_matching,
    )
    if write_producer_geometry:
        _write_refraction_static_request(table_path.parent)
    _create_table_job(state, tmp_path, table_path)
    job_dir = tmp_path / 'jobs' / APPLY_JOB_ID
    _create_apply_job(state, job_dir)
    payload: dict[str, Any] = {
        'file_id': SOURCE_FILE_ID,
        'key1_byte': KEY1,
        'key2_byte': KEY2,
        'combined_table_artifact_id': f'{TABLE_JOB_ID}:{table_path.name}',
        'register_corrected_file': register_corrected_file,
        'missing_static_policy': missing_static_policy,
        'allow_missing_source_static': allow_missing_source_static,
        'double_application_policy': double_application_policy,
        'max_abs_shift_ms': 250.0,
    }
    if endpoint_id_matching:
        payload['source_key_header'] = 'endpoint_id'
        payload['receiver_key_header'] = 'endpoint_id'
    req = RefractionStaticTableApplyRequest.model_validate(payload)

    run_refraction_static_table_apply_job(APPLY_JOB_ID, req, state)
    return state, store, job_dir


def _write_source_lineage(store: Path, components: list[str]) -> None:
    meta_path = store / 'meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    meta['derived'] = {
        'components': [
            {
                'name': 'refraction_static_correction',
                'static_components_applied': components,
            }
        ]
    }
    meta_path.write_text(json.dumps(meta), encoding='utf-8')


def _read_static_table_apply_history(job_dir: Path) -> dict[str, Any]:
    return json.loads(
        (job_dir / STATIC_TABLE_APPLY_HISTORY_JSON_NAME).read_text(
            encoding='utf-8',
        )
    )


def _rerun_same_table_against_file(
    *,
    state: AppState,
    tmp_path: Path,
    file_id: str,
    allow_reapply_same_static_table: bool = False,
) -> Path:
    job_dir = tmp_path / 'jobs' / SECOND_APPLY_JOB_ID
    _create_apply_job(state, job_dir, job_id=SECOND_APPLY_JOB_ID, file_id=file_id)
    req = RefractionStaticTableApplyRequest.model_validate(
        {
            'file_id': file_id,
            'key1_byte': KEY1,
            'key2_byte': KEY2,
            'combined_table_artifact_id': (
                f'{TABLE_JOB_ID}:canonical_static_table.csv'
            ),
            'register_corrected_file': False,
            'allow_reapply_same_static_table': allow_reapply_same_static_table,
        }
    )
    run_refraction_static_table_apply_job(SECOND_APPLY_JOB_ID, req, state)
    return job_dir


def test_static_table_apply_maps_source_receiver_to_trace_shift(tmp_path: Path) -> None:
    state, _store, job_dir = _run_apply(tmp_path)

    with state.lock:
        assert state.jobs[APPLY_JOB_ID]['status'] == 'done'
    with np.load(job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        np.testing.assert_allclose(
            data['trace_shift_s_sorted'],
            [0.008, 0.004, 0.004, 0.0],
        )
        np.testing.assert_array_equal(data['trace_static_status_sorted'], ['ok'] * 4)

    rows = list(
        csv.DictReader(
            (job_dir / STATIC_TABLE_APPLY_TRACE_SHIFTS_CSV_NAME).open(
                encoding='utf-8'
            )
        )
    )
    assert [row['trace_shift_ms'] for row in rows] == [
        '8.000000',
        '4.000000',
        '4.000000',
        '0.000000',
    ]


def test_static_table_apply_writes_history(tmp_path: Path) -> None:
    _state, _store, job_dir = _run_apply(tmp_path)

    history = _read_static_table_apply_history(job_dir)
    legacy_history = json.loads(
        (job_dir / STATIC_TABLE_APPLY_REFRACTION_HISTORY_JSON_NAME).read_text(
            encoding='utf-8',
        )
    )
    assert legacy_history == history
    assert history['source_table_digest'] is None
    assert history['receiver_table_digest'] is None
    assert len(history['combined_table_digest']) == 64
    assert history['input_file_id'] == SOURCE_FILE_ID
    assert history['output_file_id'] is None
    assert history['applied_component_name'] == 'refraction'
    assert len(history['trace_shift_s_sorted_digest']) == 64


def test_static_table_apply_rejects_duplicate_table_digest(
    tmp_path: Path,
) -> None:
    state, _store, _job_dir = _run_apply(tmp_path, register_corrected_file=True)
    with state.lock:
        corrected_file_id = str(state.jobs[APPLY_JOB_ID]['corrected_file_id'])

    second_job_dir = _rerun_same_table_against_file(
        state=state,
        tmp_path=tmp_path,
        file_id=corrected_file_id,
    )

    with state.lock:
        job = dict(state.jobs[SECOND_APPLY_JOB_ID])
    assert job['status'] == 'error'
    assert 'allow_reapply_same_static_table=False' in str(job['message'])
    assert not (second_job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME).exists()


def test_static_table_apply_override_allows_reapply(tmp_path: Path) -> None:
    state, _store, _job_dir = _run_apply(tmp_path, register_corrected_file=True)
    with state.lock:
        corrected_file_id = str(state.jobs[APPLY_JOB_ID]['corrected_file_id'])

    second_job_dir = _rerun_same_table_against_file(
        state=state,
        tmp_path=tmp_path,
        file_id=corrected_file_id,
        allow_reapply_same_static_table=True,
    )

    with state.lock:
        job = dict(state.jobs[SECOND_APPLY_JOB_ID])
    assert job['status'] == 'done', job.get('message')
    history = _read_static_table_apply_history(second_job_dir)
    guard = history['static_table_reapply_guard']
    assert history['allow_reapply_same_static_table'] is True
    assert guard['status'] == 'duplicate_allowed_by_override'
    assert guard['override_used'] is True
    assert 'same_table_digest' in guard['duplicate_reasons']


def test_static_table_history_records_source_job_id(tmp_path: Path) -> None:
    _state, _store, job_dir = _run_apply(tmp_path)

    history = _read_static_table_apply_history(job_dir)
    assert history['created_from_refraction_job_id'] == 'refraction-source-job'
    assert history['created_from_export_job_id'] == TABLE_JOB_ID


def test_static_table_history_records_sign_convention(tmp_path: Path) -> None:
    _state, _store, job_dir = _run_apply(tmp_path)

    history = _read_static_table_apply_history(job_dir)
    assert history['sign_convention'] == REFRACTION_STATIC_REPO_SIGN_CONVENTION


def test_static_table_apply_uses_materialized_sorted_headers_for_trace_shift(
    tmp_path: Path,
) -> None:
    sorted_to_original = np.asarray([2, 0, 3, 1], dtype=np.int64)
    state, store = _write_target_store(
        tmp_path,
        sorted_to_original=sorted_to_original,
    )
    _write_header(store, 9, [100, 100, 101, 101])
    _write_header(store, 13, [201, 200, 201, 200])
    _write_header(store, 73, [1000, 1000, 1010, 1010])
    _write_header(store, 77, [2000, 2000, 2000, 2000])
    _write_header(store, 81, [1110, 1100, 1110, 1100])
    _write_header(store, 85, [2000, 2000, 2000, 2000])
    _write_header(store, 45, [10, 10, 12, 12])
    _write_header(store, 41, [22, 20, 22, 20])
    table_path = tmp_path / 'jobs' / TABLE_JOB_ID / 'canonical_static_table.csv'
    _write_table(table_path)
    _write_refraction_static_request(table_path.parent)
    _create_table_job(state, tmp_path, table_path)
    job_dir = tmp_path / 'jobs' / APPLY_JOB_ID
    _create_apply_job(state, job_dir)
    req = RefractionStaticTableApplyRequest.model_validate(
        {
            'file_id': SOURCE_FILE_ID,
            'key1_byte': KEY1,
            'key2_byte': KEY2,
            'combined_table_artifact_id': f'{TABLE_JOB_ID}:{table_path.name}',
            'register_corrected_file': False,
        }
    )

    run_refraction_static_table_apply_job(APPLY_JOB_ID, req, state)

    with state.lock:
        assert state.jobs[APPLY_JOB_ID]['status'] == 'done'
    with np.load(job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['sorted_trace_index'], sorted_to_original)
        np.testing.assert_array_equal(
            data['source_endpoint_id_sorted'],
            ['100', '100', '101', '101'],
        )
        np.testing.assert_array_equal(
            data['receiver_endpoint_id_sorted'],
            ['201', '200', '201', '200'],
        )
        np.testing.assert_allclose(
            data['trace_shift_s_sorted'],
            [0.004, 0.008, 0.0, 0.004],
        )


def test_static_table_apply_rejects_large_endpoint_shift_even_when_trace_sum_cancels(
    tmp_path: Path,
) -> None:
    state, _store = _write_target_store(tmp_path)
    table_path = tmp_path / 'jobs' / TABLE_JOB_ID / 'canonical_static_table.csv'
    rows = [
        _canonical_row(
            endpoint_kind='source',
            endpoint_key=SOURCE_KEYS[100],
            endpoint_id=100,
            applied_shift_ms=1000.0,
        ),
        _canonical_row(
            endpoint_kind='source',
            endpoint_key=SOURCE_KEYS[101],
            endpoint_id=101,
            applied_shift_ms=1000.0,
        ),
        _canonical_row(
            endpoint_kind='receiver',
            endpoint_key=RECEIVER_KEYS[200],
            endpoint_id=200,
            applied_shift_ms=-1000.0,
        ),
        _canonical_row(
            endpoint_kind='receiver',
            endpoint_key=RECEIVER_KEYS[201],
            endpoint_id=201,
            applied_shift_ms=-1000.0,
        ),
    ]
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    _write_refraction_static_request(table_path.parent)
    _create_table_job(state, tmp_path, table_path)
    job_dir = tmp_path / 'jobs' / APPLY_JOB_ID
    _create_apply_job(state, job_dir)
    req = RefractionStaticTableApplyRequest.model_validate(
        {
            'file_id': SOURCE_FILE_ID,
            'key1_byte': KEY1,
            'key2_byte': KEY2,
            'combined_table_artifact_id': f'{TABLE_JOB_ID}:{table_path.name}',
            'register_corrected_file': False,
            'max_abs_shift_ms': 250.0,
        }
    )

    run_refraction_static_table_apply_job(APPLY_JOB_ID, req, state)

    with state.lock:
        job = dict(state.jobs[APPLY_JOB_ID])
    assert job['status'] == 'error'
    assert 'imported endpoint static shift exceeds max_abs_shift_ms' in str(
        job['message']
    )
    assert not (job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME).exists()


def test_static_table_apply_imports_source_receiver_static_table_npz(
    tmp_path: Path,
) -> None:
    state, _store = _write_target_store(tmp_path)
    table_path = tmp_path / 'jobs' / TABLE_JOB_ID / SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME
    _write_source_receiver_static_table_npz(table_path)
    _write_refraction_static_request(table_path.parent)
    _create_table_job(state, tmp_path, table_path, statics_kind='refraction')
    job_dir = tmp_path / 'jobs' / APPLY_JOB_ID
    _create_apply_job(state, job_dir)

    req = RefractionStaticTableApplyRequest.model_validate(
        {
            'file_id': SOURCE_FILE_ID,
            'key1_byte': KEY1,
            'key2_byte': KEY2,
            'combined_table_artifact_id': f'{TABLE_JOB_ID}:{table_path.name}',
            'register_corrected_file': False,
        }
    )

    run_refraction_static_table_apply_job(APPLY_JOB_ID, req, state)

    with state.lock:
        assert state.jobs[APPLY_JOB_ID]['status'] == 'done'
    with np.load(job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        np.testing.assert_allclose(
            data['trace_shift_s_sorted'],
            [0.008, 0.004, 0.004, 0.0],
        )


def test_static_table_apply_does_not_require_producer_geometry(
    tmp_path: Path,
) -> None:
    state, _store, job_dir = _run_apply(tmp_path, write_producer_geometry=False)

    with state.lock:
        job = dict(state.jobs[APPLY_JOB_ID])
    assert job['status'] == 'done'
    assert (job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME).exists()


def test_static_table_apply_uses_producer_geometry_when_request_omits_geometry(
    tmp_path: Path,
) -> None:
    state, store = _write_target_store(tmp_path)
    geometry = {
        'source_id_byte': 109,
        'receiver_id_byte': 113,
        'source_x_byte': 173,
        'source_y_byte': 177,
        'receiver_x_byte': 181,
        'receiver_y_byte': 185,
        'source_elevation_byte': 145,
        'receiver_elevation_byte': 141,
        'coordinate_scalar_byte': 171,
        'elevation_scalar_byte': 169,
        'coordinate_unit': 'm',
        'elevation_unit': 'm',
    }
    _write_header(store, 109, [300, 301, 300, 301])
    _write_header(store, 113, [400, 400, 401, 401])
    _write_header(store, 173, [3000, 3010, 3000, 3010])
    _write_header(store, 177, [5000, 5000, 5000, 5000])
    _write_header(store, 181, [3100, 3100, 3110, 3110])
    _write_header(store, 185, [5000, 5000, 5000, 5000])
    _write_header(store, 145, [30, 32, 30, 32])
    _write_header(store, 141, [40, 40, 42, 42])
    _write_header(store, 171, [1, 1, 1, 1])
    _write_header(store, 169, [1, 1, 1, 1])

    source_keys = {
        300: _endpoint_key('source', 300, 3000.0, 5000.0, 30.0),
        301: _endpoint_key('source', 301, 3010.0, 5000.0, 32.0),
    }
    receiver_keys = {
        400: _endpoint_key('receiver', 400, 3100.0, 5000.0, 40.0),
        401: _endpoint_key('receiver', 401, 3110.0, 5000.0, 42.0),
    }
    table_path = tmp_path / 'jobs' / TABLE_JOB_ID / 'canonical_static_table.csv'
    rows = [
        _canonical_row(
            endpoint_kind='source',
            endpoint_key=source_keys[300],
            endpoint_id=300,
            applied_shift_ms=8.0,
        ),
        _canonical_row(
            endpoint_kind='source',
            endpoint_key=source_keys[301],
            endpoint_id=301,
            applied_shift_ms=4.0,
        ),
        _canonical_row(
            endpoint_kind='receiver',
            endpoint_key=receiver_keys[400],
            endpoint_id=400,
            applied_shift_ms=0.0,
        ),
        _canonical_row(
            endpoint_kind='receiver',
            endpoint_key=receiver_keys[401],
            endpoint_id=401,
            applied_shift_ms=-4.0,
        ),
    ]
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    _write_refraction_static_request(table_path.parent, geometry=geometry)
    _create_table_job(state, tmp_path, table_path, statics_kind='refraction')
    job_dir = tmp_path / 'jobs' / APPLY_JOB_ID
    _create_apply_job(state, job_dir)

    req = RefractionStaticTableApplyRequest.model_validate(
        {
            'file_id': SOURCE_FILE_ID,
            'key1_byte': KEY1,
            'key2_byte': KEY2,
            'combined_table_artifact_id': f'{TABLE_JOB_ID}:{table_path.name}',
            'register_corrected_file': False,
        }
    )

    run_refraction_static_table_apply_job(APPLY_JOB_ID, req, state)

    with state.lock:
        assert state.jobs[APPLY_JOB_ID]['status'] == 'done'
    with np.load(job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        np.testing.assert_array_equal(
            data['source_endpoint_key_sorted'],
            [source_keys[300], source_keys[301], source_keys[300], source_keys[301]],
        )
        np.testing.assert_allclose(
            data['trace_shift_s_sorted'],
            [0.008, 0.004, 0.004, 0.0],
        )


def test_static_table_apply_rejects_missing_source_static_by_default(
    tmp_path: Path,
) -> None:
    state, _store, job_dir = _run_apply(tmp_path, include_source_101=False)

    with state.lock:
        job = dict(state.jobs[APPLY_JOB_ID])
    assert job['status'] == 'error'
    assert 'missing_source_static' in str(job['message'])
    assert not (job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME).exists()


def test_static_table_apply_zero_policy_allows_missing_static(tmp_path: Path) -> None:
    state, _store, job_dir = _run_apply(
        tmp_path,
        include_source_101=False,
        missing_static_policy='zero',
        allow_missing_source_static=True,
    )

    with state.lock:
        assert state.jobs[APPLY_JOB_ID]['status'] == 'done'
    with np.load(job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        np.testing.assert_allclose(
            data['trace_shift_s_sorted'],
            [0.008, 0.0, 0.004, -0.004],
        )
        assert data['trace_static_status_sorted'][1] == 'missing_source_static_zeroed'
    qc = json.loads((job_dir / STATIC_TABLE_APPLY_QC_JSON_NAME).read_text())
    assert qc['n_missing_source_endpoints'] == 1
    assert qc['trace_static_status_counts']['missing_source_static_zeroed'] == 2


def test_static_table_apply_registers_corrected_file(tmp_path: Path) -> None:
    state, _store, job_dir = _run_apply(tmp_path, register_corrected_file=True)

    with state.lock:
        job = dict(state.jobs[APPLY_JOB_ID])
    assert job['status'] == 'done', job.get('message')
    corrected_file_id = job['corrected_file_id']
    corrected_store_path = Path(str(job['corrected_store_path']))
    assert state.file_registry.get_store_path(str(corrected_file_id)) == str(
        corrected_store_path
    )
    assert (job_dir / CORRECTED_FILE_JSON_NAME).exists()
    qc = json.loads((job_dir / STATIC_TABLE_APPLY_QC_JSON_NAME).read_text())
    assert qc['corrected_file_id'] == corrected_file_id
    assert CORRECTED_FILE_JSON_NAME in qc['artifact_names']

    corrected = np.load(corrected_store_path / 'traces.npy')
    assert [int(np.argmax(corrected[index])) for index in range(4)] == [10, 9, 9, 8]

    corrected_meta = json.loads((corrected_store_path / 'meta.json').read_text())
    assert corrected_meta['derived']['static_components_applied'] == ['refraction']


def test_static_table_apply_history_checks_trace_store_lineage(
    tmp_path: Path,
) -> None:
    state, _store, job_dir = _run_apply(
        tmp_path,
        lineage_components=['refraction'],
    )

    with state.lock:
        assert state.jobs[APPLY_JOB_ID]['status'] == 'done'
    qc = json.loads((job_dir / STATIC_TABLE_APPLY_QC_JSON_NAME).read_text())
    history = json.loads((job_dir / 'refraction_static_history.json').read_text())
    assert qc['double_application_policy']['status'] == 'duplicate_warned'
    assert qc['double_application_policy']['duplicate_components'] == ['refraction']
    assert history['double_application_policy']['status'] == 'duplicate_warned'
    assert 'double_application_policy=warn' in history['warnings'][0]


def test_static_table_apply_double_application_policy_fail_rejects(
    tmp_path: Path,
) -> None:
    state, _store, job_dir = _run_apply(
        tmp_path,
        lineage_components=['refraction'],
        double_application_policy='fail',
    )

    with state.lock:
        job = dict(state.jobs[APPLY_JOB_ID])
    assert job['status'] == 'error'
    assert 'double_application_policy=fail' in str(job['message'])
    assert not (job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME).exists()


def test_static_table_apply_trace_shift_matches_expected_synthetic(
    tmp_path: Path,
) -> None:
    state, _store, job_dir = _run_apply(tmp_path, endpoint_id_matching=True)

    with state.lock:
        assert state.jobs[APPLY_JOB_ID]['status'] == 'done'
    with np.load(job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        assert str(data['source_identity_mode']) == 'endpoint_id'
        assert str(data['receiver_identity_mode']) == 'endpoint_id'
        np.testing.assert_allclose(
            data['source_static_shift_s_sorted'],
            [0.008, 0.004, 0.008, 0.004],
        )
        np.testing.assert_allclose(
            data['receiver_static_shift_s_sorted'],
            [0.0, 0.0, -0.004, -0.004],
        )
        np.testing.assert_allclose(
            data['trace_shift_s_sorted'],
            [0.008, 0.004, 0.004, 0.0],
        )


def test_static_table_apply_endpoint_creates_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(tmp_path / 'jobs'))
    captured: dict[str, Any] = {}

    def _capture_start_job_thread(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        _capture_start_job_thread,
    )
    with app.state.sv.lock:
        app.state.sv.jobs.clear()
        app.state.sv.cached_readers.clear()
        app.state.sv.file_registry.clear()
    with TestClient(app) as client:
        response = client.post(
            '/statics/refraction/static-table/apply',
            json={
                'file_id': SOURCE_FILE_ID,
                'combined_table_artifact_id': 'table-job:canonical.csv',
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload['state'] == 'queued'
    assert captured['target'] is run_refraction_static_table_apply_job
    with app.state.sv.lock:
        job = dict(app.state.sv.jobs[payload['job_id']])
    assert job['statics_kind'] == 'refraction_static_table_apply'
    with app.state.sv.lock:
        app.state.sv.jobs.clear()
        app.state.sv.cached_readers.clear()
        app.state.sv.file_registry.clear()
