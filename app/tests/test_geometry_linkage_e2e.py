from __future__ import annotations

import csv
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
from app.main import app
from app.services.geometry_linkage_artifacts import (
    GEOMETRY_LINKAGE_CSV_NAME,
    GEOMETRY_LINKAGE_NPZ_NAME,
    GEOMETRY_LINKAGE_QC_JSON_NAME,
)
from app.services.geometry_linkage_loader import (
    LoadedGeometryLinkageArtifact,
    load_geometry_linkage_from_job_dir,
    load_geometry_linkage_trace_node_mapping,
)
from app.tests._stubs import write_baseline_raw

FILE_ID = 'geometry-linkage-e2e-source'
KEY1_BYTE = 189
KEY2_BYTE = 193
SOURCE_X_BYTE = 73
SOURCE_Y_BYTE = 77
RECEIVER_X_BYTE = 81
RECEIVER_Y_BYTE = 85
COORDINATE_SCALAR_BYTE = 71
DT = 0.004
EXPECTED_ARTIFACTS = {
    GEOMETRY_LINKAGE_NPZ_NAME,
    GEOMETRY_LINKAGE_CSV_NAME,
    GEOMETRY_LINKAGE_QC_JSON_NAME,
    'job_meta.json',
}
CSV_COLUMNS = [
    'endpoint_kind',
    'endpoint_id',
    'x_m',
    'y_m',
    'node_id',
    'linked_to_kind',
    'linked_to_id',
    'distance_m',
    'method',
]


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(tmp_path / 'jobs'))
    _clear_state()
    with TestClient(app) as test_client:
        yield test_client
    _clear_state()


@pytest.fixture()
def sync_linkage_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(statics_router_module, 'start_job_thread', _run_job_sync)


def test_linkage_none_api_lifecycle_artifact_round_trip(
    client: TestClient,
    sync_linkage_jobs: None,
    tmp_path: Path,
) -> None:
    store_dir = _write_synthetic_trace_store(
        client,
        tmp_path,
        source_xy=[(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)],
        receiver_xy=[(10.0, 0.0), (110.0, 0.0), (210.0, 0.0)],
    )

    response = client.post(
        '/statics/linkage/build',
        json=_request_payload(linkage={'mode': 'none', 'prefer_receiver_anchor': False}),
    )

    assert response.status_code == 200
    body = response.json()
    assert body['state'] == 'queued'
    job_id = body['job_id']

    status_response = client.get(f'/statics/job/{job_id}/status')
    assert status_response.json() == {
        'state': 'done',
        'progress': 1.0,
        'message': '',
    }

    files_response = client.get(f'/statics/job/{job_id}/files')
    assert files_response.status_code == 200
    assert {item['name'] for item in files_response.json()['files']} == (
        EXPECTED_ARTIFACTS
    )

    job_dir = _job_dir(client, job_id)
    loaded = load_geometry_linkage_from_job_dir(
        job_dir,
        expected_n_traces=3,
        expected_key1_byte=KEY1_BYTE,
        expected_key2_byte=KEY2_BYTE,
    )
    mapping = load_geometry_linkage_trace_node_mapping(
        job_dir / GEOMETRY_LINKAGE_NPZ_NAME,
        expected_n_traces=3,
    )
    csv_rows = _read_csv_rows(job_dir / GEOMETRY_LINKAGE_CSV_NAME)
    qc = _read_qc(job_dir)

    assert loaded.mode == 'none'
    assert loaded.source_node_id_sorted.shape == (3,)
    assert loaded.receiver_node_id_sorted.shape == (3,)
    np.testing.assert_array_equal(
        loaded.source_node_id_sorted,
        loaded.source_node_id_by_endpoint[loaded.source_endpoint_id_sorted],
    )
    np.testing.assert_array_equal(
        loaded.receiver_node_id_sorted,
        loaded.receiver_node_id_by_endpoint[loaded.receiver_endpoint_id_sorted],
    )
    np.testing.assert_array_equal(
        mapping.source_node_id_sorted,
        loaded.source_node_id_sorted,
    )
    np.testing.assert_array_equal(
        mapping.receiver_node_id_sorted,
        loaded.receiver_node_id_sorted,
    )
    assert set(loaded.source_node_id_by_endpoint).isdisjoint(
        set(loaded.receiver_node_id_by_endpoint)
    )
    np.testing.assert_array_equal(
        np.unique(
            np.concatenate(
                (loaded.source_node_id_by_endpoint, loaded.receiver_node_id_by_endpoint)
            )
        ),
        np.arange(loaded.n_nodes),
    )
    assert loaded.n_receiver_anchor_links == 0
    assert loaded.n_source_fallback_links == 0
    assert loaded.n_independent_source_nodes == loaded.n_source_endpoints

    _assert_csv_matches_loaded(csv_rows, loaded)
    assert qc['mode'] == 'none'
    assert qc['counts']['n_traces'] == 3
    assert qc['counts']['n_nodes'] == loaded.n_nodes
    assert qc['receiver_anchor_distance_m']['count'] == 0
    assert qc['source_fallback_distance_m']['count'] == 0

    _assert_download_artifacts_match_files(
        client,
        job_id,
        job_dir,
        expected_csv_rows=len(csv_rows),
    )
    _assert_no_corrected_trace_store(client, job_dir, store_dir)


def test_auto_threshold_receiver_anchor_e2e(
    client: TestClient,
    sync_linkage_jobs: None,
    tmp_path: Path,
) -> None:
    _write_synthetic_trace_store(
        client,
        tmp_path,
        source_xy=[(100.0, 100.0), (110.0, 100.0), (500.0, 100.0)],
        receiver_xy=[(100.0, 100.0), (100.0, 100.0), (300.0, 100.0)],
    )

    loaded, csv_rows, qc = _run_done_linkage_job(
        client,
        linkage={
            'mode': 'auto_threshold',
            'threshold_m': 25.0,
            'receiver_location_interval_m': 25.0,
            'prefer_receiver_anchor': True,
        },
        expected_n_traces=3,
    )

    r0 = _endpoint_id(loaded, 'receiver', 100.0, 100.0)
    s0 = _endpoint_id(loaded, 'source', 100.0, 100.0)
    s1 = _endpoint_id(loaded, 'source', 110.0, 100.0)
    s2 = _endpoint_id(loaded, 'source', 500.0, 100.0)

    assert loaded.mode == 'auto_threshold'
    assert loaded.n_receiver_anchor_links == 2
    assert loaded.source_node_id_by_endpoint[s0] == loaded.receiver_node_id_by_endpoint[r0]
    assert loaded.source_node_id_by_endpoint[s1] == loaded.receiver_node_id_by_endpoint[r0]
    assert loaded.source_node_id_by_endpoint[s2] not in set(
        loaded.receiver_node_id_by_endpoint
    )
    _assert_source_record(
        loaded,
        source_endpoint_id=s0,
        method='receiver_anchor',
        linked_to_kind='receiver',
        linked_to_id=r0,
        distance_m=0.0,
    )
    _assert_source_record(
        loaded,
        source_endpoint_id=s1,
        method='receiver_anchor',
        linked_to_kind='receiver',
        linked_to_id=r0,
        distance_m=10.0,
    )
    _assert_csv_matches_loaded(csv_rows, loaded)
    assert qc['counts']['n_receiver_anchor_links'] == 2
    assert qc['receiver_anchor_distance_m']['count'] == 2
    assert qc['receiver_anchor_distance_m']['min'] == pytest.approx(0.0)
    assert qc['receiver_anchor_distance_m']['max'] == pytest.approx(10.0)


def test_auto_threshold_converts_ft_coordinates_before_linking(
    client: TestClient,
    sync_linkage_jobs: None,
    tmp_path: Path,
) -> None:
    _write_synthetic_trace_store(
        client,
        tmp_path,
        source_xy=[(0.0, 0.0), (80.0, 0.0)],
        receiver_xy=[(0.0, 0.0), (0.0, 0.0)],
    )

    loaded, csv_rows, qc = _run_done_linkage_job(
        client,
        linkage={
            'mode': 'auto_threshold',
            'threshold_m': 25.0,
            'receiver_location_interval_m': 25.0,
            'prefer_receiver_anchor': True,
        },
        geometry={'coordinate_unit': 'ft'},
        expected_n_traces=2,
    )

    r0 = _endpoint_id(loaded, 'receiver', 0.0, 0.0)
    s0 = _endpoint_id(loaded, 'source', 0.0, 0.0)
    s1 = _endpoint_id(loaded, 'source', 24.384, 0.0)

    assert loaded.n_receiver_anchor_links == 2
    assert (
        loaded.source_node_id_by_endpoint[s0]
        == loaded.receiver_node_id_by_endpoint[r0]
    )
    assert (
        loaded.source_node_id_by_endpoint[s1]
        == loaded.receiver_node_id_by_endpoint[r0]
    )
    _assert_source_record(
        loaded,
        source_endpoint_id=s1,
        method='receiver_anchor',
        linked_to_kind='receiver',
        linked_to_id=r0,
        distance_m=24.384,
    )
    _assert_csv_matches_loaded(csv_rows, loaded)
    assert qc['receiver_anchor_distance_m']['max'] == pytest.approx(24.384)


def test_auto_threshold_source_fallback_e2e(
    client: TestClient,
    sync_linkage_jobs: None,
    tmp_path: Path,
) -> None:
    _write_synthetic_trace_store(
        client,
        tmp_path,
        source_xy=[(1000.0, 0.0), (1010.0, 0.0)],
        receiver_xy=[(0.0, 0.0), (0.0, 0.0)],
    )

    loaded, csv_rows, qc = _run_done_linkage_job(
        client,
        linkage={
            'mode': 'auto_threshold',
            'threshold_m': 25.0,
            'prefer_receiver_anchor': True,
        },
        expected_n_traces=2,
    )

    r0 = _endpoint_id(loaded, 'receiver', 0.0, 0.0)
    s0 = _endpoint_id(loaded, 'source', 1000.0, 0.0)
    s1 = _endpoint_id(loaded, 'source', 1010.0, 0.0)

    assert loaded.n_source_fallback_links == 2
    assert loaded.source_node_id_by_endpoint[s0] == loaded.source_node_id_by_endpoint[s1]
    assert loaded.source_node_id_by_endpoint[s0] != loaded.receiver_node_id_by_endpoint[
        r0
    ]
    _assert_source_record(
        loaded,
        source_endpoint_id=s0,
        method='source_fallback',
        linked_to_kind='source',
        linked_to_id=s1,
        distance_m=10.0,
    )
    _assert_source_record(
        loaded,
        source_endpoint_id=s1,
        method='source_fallback',
        linked_to_kind='source',
        linked_to_id=s0,
        distance_m=10.0,
    )
    assert loaded.record_linked_to_id[_record_index(loaded, 'source', s0)] != s0
    assert loaded.record_linked_to_id[_record_index(loaded, 'source', s1)] != s1
    _assert_csv_matches_loaded(csv_rows, loaded)
    assert qc['counts']['n_source_fallback_links'] == 2
    assert qc['counts']['n_source_only_nodes'] == 1
    assert qc['source_fallback_distance_m']['count'] == 2
    assert qc['source_fallback_distance_m']['min'] == pytest.approx(10.0)
    assert qc['source_fallback_distance_m']['max'] == pytest.approx(10.0)


def test_auto_threshold_receiver_tie_break_is_deterministic_e2e(
    client: TestClient,
    sync_linkage_jobs: None,
    tmp_path: Path,
) -> None:
    _write_synthetic_trace_store(
        client,
        tmp_path,
        source_xy=[(0.0, 0.0), (0.0, 0.0)],
        receiver_xy=[(-10.0, 0.0), (10.0, 0.0)],
    )

    first_loaded, first_csv_rows, first_qc = _run_done_linkage_job(
        client,
        linkage={
            'mode': 'auto_threshold',
            'threshold_m': 20.0,
            'prefer_receiver_anchor': True,
        },
        expected_n_traces=2,
    )
    second_loaded, second_csv_rows, second_qc = _run_done_linkage_job(
        client,
        linkage={
            'mode': 'auto_threshold',
            'threshold_m': 20.0,
            'prefer_receiver_anchor': True,
        },
        expected_n_traces=2,
    )

    r0 = _endpoint_id(first_loaded, 'receiver', -10.0, 0.0)
    s0 = _endpoint_id(first_loaded, 'source', 0.0, 0.0)
    _assert_source_record(
        first_loaded,
        source_endpoint_id=s0,
        method='receiver_anchor',
        linked_to_kind='receiver',
        linked_to_id=r0,
        distance_m=10.0,
    )
    assert first_csv_rows == second_csv_rows
    _assert_record_arrays_equal(first_loaded, second_loaded)
    assert _stable_qc(first_qc) == _stable_qc(second_qc)


def test_coordinate_scalar_applied_in_api_artifacts(
    client: TestClient,
    sync_linkage_jobs: None,
    tmp_path: Path,
) -> None:
    _write_synthetic_trace_store(
        client,
        tmp_path,
        source_xy=[(1000.0, 2000.0), (5000.0, 6000.0)],
        receiver_xy=[(3000.0, 4000.0), (7000.0, 8000.0)],
        scalars=[-10, 0],
    )

    loaded, csv_rows, qc = _run_done_linkage_job(
        client,
        linkage={'mode': 'none', 'prefer_receiver_anchor': False},
        expected_n_traces=2,
    )

    np.testing.assert_allclose(loaded.source_x_m_sorted, [100.0, 5000.0])
    np.testing.assert_allclose(loaded.source_y_m_sorted, [200.0, 6000.0])
    np.testing.assert_allclose(loaded.receiver_x_m_sorted, [300.0, 7000.0])
    np.testing.assert_allclose(loaded.receiver_y_m_sorted, [400.0, 8000.0])
    assert _endpoint_id(loaded, 'source', 100.0, 200.0) == 0
    assert _endpoint_id(loaded, 'receiver', 300.0, 400.0) == 0

    first_source_row = next(
        row for row in csv_rows if row['endpoint_kind'] == 'source'
    )
    assert float(first_source_row['x_m']) == pytest.approx(100.0)
    assert float(first_source_row['y_m']) == pytest.approx(200.0)
    assert loaded.coordinate_scalar_zero_count == 1
    assert qc['coordinate_scalar']['zero_count'] == 1


@pytest.mark.parametrize(
    ('case_name', 'mutator'),
    [
        (
            'auto_threshold_null_threshold',
            lambda payload: payload['linkage'].update({'threshold_m': None}),
        ),
        (
            'none_nonnull_threshold',
            lambda payload: payload.update(
                {'linkage': {'mode': 'none', 'threshold_m': 10.0}}
            ),
        ),
        ('key_byte_zero', lambda payload: payload.update({'key1_byte': 0})),
        ('key_byte_241', lambda payload: payload.update({'key2_byte': 241})),
        (
            'duplicate_geometry_header',
            lambda payload: payload['geometry'].update({'source_y_byte': 73}),
        ),
        ('unknown_field', lambda payload: payload.update({'unknown': True})),
    ],
)
def test_request_schema_failures_do_not_create_jobs(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case_name: str,
    mutator: Any,
) -> None:
    del case_name
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    payload = _request_payload(
        linkage={
            'mode': 'auto_threshold',
            'threshold_m': 25.0,
            'prefer_receiver_anchor': True,
        }
    )
    mutator(payload)

    response = client.post('/statics/linkage/build', json=payload)

    assert response.status_code == 422
    assert 'job_id' not in response.json()
    assert started == []
    with client.app.state.sv.lock:
        assert len(client.app.state.sv.jobs) == 0
    jobs_dir = tmp_path / 'jobs'
    assert not jobs_dir.exists() or not any(jobs_dir.iterdir())


@pytest.mark.parametrize(
    ('case_name', 'store_kwargs', 'message_part'),
    [
        (
            'missing_header',
            {'missing_header_bytes': {SOURCE_X_BYTE}},
            f'failed to read geometry linkage header byte {SOURCE_X_BYTE}',
        ),
        (
            'shape_mismatch',
            {'header_overrides': {SOURCE_X_BYTE: np.asarray([1, 2])}},
            'shape mismatch',
        ),
        (
            'nonfinite_coordinate',
            {'header_overrides': {SOURCE_X_BYTE: np.asarray([np.nan, 2.0, 3.0])}},
            'must contain only finite values',
        ),
        (
            'noninteger_scalar',
            {
                'header_overrides': {
                    COORDINATE_SCALAR_BYTE: np.asarray([1.0, 1.5, 1.0])
                }
            },
            'must contain only integer values',
        ),
        (
            'nonfinite_scalar',
            {
                'header_overrides': {
                    COORDINATE_SCALAR_BYTE: np.asarray([1.0, np.inf, 1.0])
                }
            },
            'must contain only finite values',
        ),
    ],
)
def test_job_time_validation_failures_become_error_jobs(
    client: TestClient,
    sync_linkage_jobs: None,
    tmp_path: Path,
    case_name: str,
    store_kwargs: dict[str, Any],
    message_part: str,
) -> None:
    del case_name
    _write_synthetic_trace_store(
        client,
        tmp_path,
        source_xy=[(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)],
        receiver_xy=[(10.0, 0.0), (110.0, 0.0), (210.0, 0.0)],
        **store_kwargs,
    )

    response = client.post(
        '/statics/linkage/build',
        json=_request_payload(
            linkage={
                'mode': 'auto_threshold',
                'threshold_m': 25.0,
                'prefer_receiver_anchor': True,
            }
        ),
    )

    assert response.status_code == 200
    assert response.json()['state'] == 'queued'
    job_id = response.json()['job_id']
    status_response = client.get(f'/statics/job/{job_id}/status')
    assert status_response.json()['state'] == 'error'
    assert message_part in status_response.json()['message']

    job_dir = _job_dir(client, job_id)
    assert (job_dir / 'job_meta.json').is_file()
    assert not (job_dir / GEOMETRY_LINKAGE_NPZ_NAME).exists()
    assert not (job_dir / GEOMETRY_LINKAGE_CSV_NAME).exists()
    assert not (job_dir / GEOMETRY_LINKAGE_QC_JSON_NAME).exists()
    for name in (
        GEOMETRY_LINKAGE_NPZ_NAME,
        GEOMETRY_LINKAGE_CSV_NAME,
        GEOMETRY_LINKAGE_QC_JSON_NAME,
    ):
        missing_artifact_response = client.get(
            f'/statics/job/{job_id}/download',
            params={'name': name},
        )
        assert missing_artifact_response.status_code == 404
    assert client.get(
        f'/statics/job/{job_id}/download',
        params={'name': 'job_meta.json'},
    ).status_code == 200


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


def _request_payload(
    *,
    linkage: dict[str, Any],
    file_id: str = FILE_ID,
    geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    geometry_payload = {
        'source_x_byte': SOURCE_X_BYTE,
        'source_y_byte': SOURCE_Y_BYTE,
        'receiver_x_byte': RECEIVER_X_BYTE,
        'receiver_y_byte': RECEIVER_Y_BYTE,
        'coordinate_scalar_byte': COORDINATE_SCALAR_BYTE,
    }
    if geometry is not None:
        geometry_payload.update(deepcopy(geometry))
    return {
        'file_id': file_id,
        'key1_byte': KEY1_BYTE,
        'key2_byte': KEY2_BYTE,
        'geometry': geometry_payload,
        'linkage': deepcopy(linkage),
    }


def _write_synthetic_trace_store(
    client: TestClient,
    tmp_path: Path,
    *,
    source_xy: list[tuple[float, float]],
    receiver_xy: list[tuple[float, float]],
    scalars: list[int] | np.ndarray | None = None,
    missing_header_bytes: set[int] | None = None,
    header_overrides: dict[int, np.ndarray] | None = None,
    file_id: str = FILE_ID,
) -> Path:
    source = np.asarray(source_xy, dtype=np.float64)
    receiver = np.asarray(receiver_xy, dtype=np.float64)
    if source.shape != receiver.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError('source_xy and receiver_xy must be matching Nx2 arrays')
    n_traces = int(source.shape[0])
    store_dir = tmp_path / f'trace-store-{file_id}'
    store_dir.mkdir(parents=True, exist_ok=True)

    traces = np.zeros((n_traces, 8), dtype=np.float32)
    np.save(store_dir / 'traces.npy', traces)
    np.savez(
        store_dir / 'index.npz',
        key1_values=np.asarray([101], dtype=np.int64),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([n_traces], dtype=np.int64),
        sorted_to_original=np.arange(n_traces, dtype=np.int64),
    )
    (store_dir / 'meta.json').write_text(
        json.dumps(
            {
                'dt': DT,
                'dtype': 'float32',
                'n_traces': n_traces,
                'n_samples': 8,
                'scale': None,
                'original_name': 'synthetic-line.sgy',
                'original_segy_path': '/synthetic/synthetic-line.sgy',
                'key_bytes': {'key1': KEY1_BYTE, 'key2': KEY2_BYTE},
            },
            sort_keys=True,
        ),
        encoding='utf-8',
    )
    write_baseline_raw(
        store_dir,
        key1=101,
        key1_byte=KEY1_BYTE,
        key2_byte=KEY2_BYTE,
        n_traces=n_traces,
    )

    scalar_arr = (
        np.ones(n_traces, dtype=np.int64)
        if scalars is None
        else np.asarray(scalars)
    )
    headers: dict[int, np.ndarray] = {
        COORDINATE_SCALAR_BYTE: scalar_arr,
        SOURCE_X_BYTE: source[:, 0],
        SOURCE_Y_BYTE: source[:, 1],
        RECEIVER_X_BYTE: receiver[:, 0],
        RECEIVER_Y_BYTE: receiver[:, 1],
        KEY1_BYTE: np.full(n_traces, 101, dtype=np.int64),
        KEY2_BYTE: np.arange(1, n_traces + 1, dtype=np.int64),
    }
    for byte, values in (header_overrides or {}).items():
        headers[int(byte)] = np.asarray(values)

    missing = set() if missing_header_bytes is None else set(missing_header_bytes)
    for byte, values in headers.items():
        if byte in missing:
            continue
        np.save(store_dir / f'headers_byte_{byte}.npy', np.asarray(values))

    state = client.app.state.sv
    state.file_registry.update(file_id, store_path=store_dir, dt=DT)
    return store_dir


def _run_done_linkage_job(
    client: TestClient,
    *,
    linkage: dict[str, Any],
    geometry: dict[str, Any] | None = None,
    expected_n_traces: int,
) -> tuple[LoadedGeometryLinkageArtifact, list[dict[str, str]], dict[str, Any]]:
    response = client.post(
        '/statics/linkage/build',
        json=_request_payload(linkage=linkage, geometry=geometry),
    )
    assert response.status_code == 200
    job_id = response.json()['job_id']
    status_response = client.get(f'/statics/job/{job_id}/status')
    assert status_response.json()['state'] == 'done'
    job_dir = _job_dir(client, job_id)
    loaded = load_geometry_linkage_from_job_dir(
        job_dir,
        expected_n_traces=expected_n_traces,
        expected_key1_byte=KEY1_BYTE,
        expected_key2_byte=KEY2_BYTE,
    )
    csv_rows = _read_csv_rows(job_dir / GEOMETRY_LINKAGE_CSV_NAME)
    qc = _read_qc(job_dir)
    return loaded, csv_rows, qc


def _job_dir(client: TestClient, job_id: str) -> Path:
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    return Path(str(job['artifacts_dir']))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CSV_COLUMNS
        return list(reader)


def _read_qc(job_dir: Path) -> dict[str, Any]:
    payload = json.loads(
        (job_dir / GEOMETRY_LINKAGE_QC_JSON_NAME).read_text(encoding='utf-8')
    )
    json.dumps(payload, allow_nan=False)
    return payload


def _endpoint_id(
    loaded: LoadedGeometryLinkageArtifact,
    kind: str,
    x_m: float,
    y_m: float,
) -> int:
    if kind == 'source':
        endpoint_id = loaded.source_endpoint_id
        xs = loaded.source_endpoint_x_m
        ys = loaded.source_endpoint_y_m
    elif kind == 'receiver':
        endpoint_id = loaded.receiver_endpoint_id
        xs = loaded.receiver_endpoint_x_m
        ys = loaded.receiver_endpoint_y_m
    else:
        raise ValueError(f'unsupported endpoint kind: {kind}')
    matches = np.flatnonzero(np.isclose(xs, x_m) & np.isclose(ys, y_m))
    assert matches.shape == (1,)
    return int(endpoint_id[int(matches[0])])


def _record_index(
    loaded: LoadedGeometryLinkageArtifact,
    kind: str,
    endpoint_id: int,
) -> int:
    matches = np.flatnonzero(
        (loaded.record_endpoint_kind == kind)
        & (loaded.record_endpoint_id == int(endpoint_id))
    )
    assert matches.shape == (1,)
    return int(matches[0])


def _assert_source_record(
    loaded: LoadedGeometryLinkageArtifact,
    *,
    source_endpoint_id: int,
    method: str,
    linked_to_kind: str,
    linked_to_id: int,
    distance_m: float,
) -> None:
    index = _record_index(loaded, 'source', source_endpoint_id)
    assert str(loaded.record_method[index]) == method
    assert str(loaded.record_linked_to_kind[index]) == linked_to_kind
    assert int(loaded.record_linked_to_id[index]) == int(linked_to_id)
    assert float(loaded.record_distance_m[index]) == pytest.approx(distance_m)


def _assert_csv_matches_loaded(
    rows: list[dict[str, str]],
    loaded: LoadedGeometryLinkageArtifact,
) -> None:
    assert len(rows) == loaded.record_method.shape[0]
    for index, row in enumerate(rows):
        assert row['endpoint_kind'] == str(loaded.record_endpoint_kind[index])
        assert int(row['endpoint_id']) == int(loaded.record_endpoint_id[index])
        assert float(row['x_m']) == pytest.approx(float(loaded.record_x_m[index]))
        assert float(row['y_m']) == pytest.approx(float(loaded.record_y_m[index]))
        assert int(row['node_id']) == int(loaded.record_node_id[index])
        assert row['linked_to_kind'] == str(loaded.record_linked_to_kind[index])
        linked_to_id = int(loaded.record_linked_to_id[index])
        assert row['linked_to_id'] == (
            '' if linked_to_id < 0 else str(linked_to_id)
        )
        distance = float(loaded.record_distance_m[index])
        if np.isnan(distance):
            assert row['distance_m'] == ''
        else:
            assert float(row['distance_m']) == pytest.approx(distance)
        assert row['method'] == str(loaded.record_method[index])


def _assert_download_artifacts_match_files(
    client: TestClient,
    job_id: str,
    job_dir: Path,
    *,
    expected_csv_rows: int,
) -> None:
    for name in EXPECTED_ARTIFACTS:
        response = client.get(f'/statics/job/{job_id}/download', params={'name': name})
        assert response.status_code == 200
        assert response.content
        assert response.content == (job_dir / name).read_bytes()

        if name == GEOMETRY_LINKAGE_NPZ_NAME:
            with np.load(BytesIO(response.content), allow_pickle=False) as npz:
                assert 'source_node_id_sorted' in npz.files
        elif name == GEOMETRY_LINKAGE_CSV_NAME:
            rows = list(csv.DictReader(response.text.splitlines()))
            assert rows
            assert len(rows) == expected_csv_rows
        elif name == GEOMETRY_LINKAGE_QC_JSON_NAME:
            qc = json.loads(response.text)
            json.dumps(qc, allow_nan=False)
        elif name == 'job_meta.json':
            meta = json.loads(response.text)
            assert meta['statics_kind'] == 'geometry_linkage'

    traversal = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': f'../{GEOMETRY_LINKAGE_NPZ_NAME}'},
    )
    assert traversal.status_code == 400
    missing = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': 'missing.txt'},
    )
    assert missing.status_code == 404


def _assert_no_corrected_trace_store(
    client: TestClient,
    job_dir: Path,
    source_store_dir: Path,
) -> None:
    assert not (job_dir / 'corrected_file.json').exists()
    with client.app.state.sv.lock:
        registry_keys = set(client.app.state.sv.file_registry.records)
        job_values = list(client.app.state.sv.jobs.values())
    assert registry_keys == {FILE_ID}
    assert all('corrected_file_id' not in job for job in job_values)
    assert not list(source_store_dir.parent.glob('*.statics.*'))


def _assert_record_arrays_equal(
    first: LoadedGeometryLinkageArtifact,
    second: LoadedGeometryLinkageArtifact,
) -> None:
    for name in (
        'record_endpoint_kind',
        'record_endpoint_id',
        'record_x_m',
        'record_y_m',
        'record_node_id',
        'record_linked_to_kind',
        'record_linked_to_id',
        'record_distance_m',
        'record_method',
    ):
        first_arr = getattr(first, name)
        second_arr = getattr(second, name)
        if np.issubdtype(first_arr.dtype, np.floating):
            np.testing.assert_allclose(first_arr, second_arr, equal_nan=True)
        else:
            np.testing.assert_array_equal(first_arr, second_arr)


def _stable_qc(qc: dict[str, Any]) -> dict[str, Any]:
    stable = deepcopy(qc)
    stable['job']['job_id'] = None
    return stable
