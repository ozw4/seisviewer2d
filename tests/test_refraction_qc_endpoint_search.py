from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.refraction_static_artifacts import (
    REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
)


@pytest.fixture()
def refraction_endpoint_search_client(tmp_path: Path):
    artifacts_dir = tmp_path / 'refraction-job'
    artifacts_dir.mkdir()
    _write_refraction_endpoint_search_artifacts(artifacts_dir)

    state = app.state.sv
    job_id = 'refraction-endpoint-search-job'
    with state.lock:
        state.jobs.clear()
        state.jobs.create_static_job(
            job_id,
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction',
            artifacts_dir=str(artifacts_dir),
        )
        state.jobs.mark_done(job_id, progress_1=True)

    with TestClient(app) as client:
        yield client, job_id, artifacts_dir

    with state.lock:
        state.jobs.clear()


def test_refraction_qc_endpoint_search_filters_source_and_receiver(
    refraction_endpoint_search_client,
):
    client, job_id, _artifacts_dir = refraction_endpoint_search_client

    source_response = client.post(
        '/statics/refraction/qc/endpoints',
        json={'job_id': job_id, 'endpoint_kind': 'source', 'sort': 'endpoint_key_asc'},
    )
    receiver_response = client.post(
        '/statics/refraction/qc/endpoints',
        json={
            'job_id': job_id,
            'endpoint_kind': 'receiver',
            'sort': 'endpoint_key_asc',
        },
    )
    both_response = client.post(
        '/statics/refraction/qc/endpoints',
        json={'job_id': job_id, 'endpoint_kind': 'both', 'sort': 'endpoint_key_asc'},
    )

    assert source_response.status_code == 200
    assert [row['endpoint_key'] for row in source_response.json()['records']] == [
        'S001',
        'S001_DUP',
        'S002',
    ]
    assert receiver_response.status_code == 200
    assert [row['endpoint_key'] for row in receiver_response.json()['records']] == [
        'R001',
        'R002',
    ]
    assert both_response.status_code == 200
    assert both_response.json()['total'] == 5


def test_refraction_qc_endpoint_search_query_matches_station_key_and_node(
    refraction_endpoint_search_client,
):
    client, job_id, _artifacts_dir = refraction_endpoint_search_client

    station_response = client.post(
        '/statics/refraction/qc/endpoints',
        json={'job_id': job_id, 'query': '1001', 'sort': 'endpoint_key_asc'},
    )
    key_response = client.post(
        '/statics/refraction/qc/endpoints',
        json={'job_id': job_id, 'query': 'dup', 'sort': 'endpoint_key_asc'},
    )
    node_response = client.post(
        '/statics/refraction/qc/endpoints',
        json={'job_id': job_id, 'query': '45', 'sort': 'endpoint_key_asc'},
    )

    assert station_response.status_code == 200
    station_body = station_response.json()
    assert station_body['total'] == 2
    assert [row['endpoint_key'] for row in station_body['records']] == [
        'S001',
        'S001_DUP',
    ]
    assert key_response.status_code == 200
    assert [row['endpoint_key'] for row in key_response.json()['records']] == [
        'S001_DUP'
    ]
    assert node_response.status_code == 200
    assert [row['endpoint_key'] for row in node_response.json()['records']] == [
        'S001_DUP'
    ]


def test_refraction_qc_endpoint_search_sort_and_pagination(
    refraction_endpoint_search_client,
):
    client, job_id, _artifacts_dir = refraction_endpoint_search_client

    rms_response = client.post(
        '/statics/refraction/qc/endpoints',
        json={
            'job_id': job_id,
            'endpoint_kind': 'both',
            'sort': 'residual_rms_desc',
            'limit': 2,
            'offset': 1,
        },
    )
    pick_response = client.post(
        '/statics/refraction/qc/endpoints',
        json={'job_id': job_id, 'endpoint_kind': 'both', 'sort': 'pick_count_desc'},
    )

    assert rms_response.status_code == 200
    rms_body = rms_response.json()
    assert rms_body['total'] == 5
    assert rms_body['limit'] == 2
    assert rms_body['offset'] == 1
    assert [row['endpoint_key'] for row in rms_body['records']] == ['S002', 'R002']
    assert pick_response.status_code == 200
    assert [row['endpoint_key'] for row in pick_response.json()['records'][:3]] == [
        'R001',
        'S001',
        'R002',
    ]


def test_refraction_qc_endpoint_search_merges_qc_status_and_filters_problem(
    refraction_endpoint_search_client,
):
    client, job_id, _artifacts_dir = refraction_endpoint_search_client

    response = client.post(
        '/statics/refraction/qc/endpoints',
        json={
            'job_id': job_id,
            'endpoint_kind': 'source',
            'status_filter': 'problem',
            'sort': 'endpoint_key_asc',
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert [row['endpoint_key'] for row in body['records']] == ['S002']
    assert body['records'][0]['static_status'] == 'invalid_weathering'
    assert 'label' in body['records'][0]


def test_refraction_qc_endpoint_search_missing_static_components_artifact(
    refraction_endpoint_search_client,
):
    client, job_id, artifacts_dir = refraction_endpoint_search_client
    (artifacts_dir / REFRACTION_STATIC_COMPONENTS_CSV_NAME).unlink()

    response = client.post(
        '/statics/refraction/qc/endpoints',
        json={'job_id': job_id},
    )

    assert response.status_code == 409
    assert REFRACTION_STATIC_COMPONENTS_CSV_NAME in response.json()['detail']


def _write_refraction_endpoint_search_artifacts(artifacts_dir: Path) -> None:
    _write_csv(
        artifacts_dir / REFRACTION_STATIC_COMPONENTS_CSV_NAME,
        [
            {
                'kind': 'source',
                'endpoint_key': 'S001',
                'station_id': '1001',
                'node_id': '1',
                'x_m': '1000',
                'y_m': '2000',
                'surface_elevation_m': '120',
                'pick_count': '8',
                'residual_rms_ms': '3.1',
                'datum_status': 'ok',
            },
            {
                'kind': 'source',
                'endpoint_key': 'S002',
                'station_id': '1002',
                'node_id': '2',
                'x_m': '1100',
                'y_m': '2100',
                'surface_elevation_m': '124',
                'pick_count': '5',
                'residual_rms_ms': '6.2',
                'datum_status': 'ok',
            },
            {
                'kind': 'receiver',
                'endpoint_key': 'R001',
                'station_id': '2001',
                'node_id': '10',
                'x_m': '1020',
                'y_m': '2020',
                'surface_elevation_m': '118',
                'pick_count': '9',
                'residual_rms_ms': '2',
                'datum_status': 'ok',
                'static_status': 'ok',
            },
            {
                'kind': 'receiver',
                'endpoint_key': 'R002',
                'station_id': '2002',
                'node_id': '11',
                'x_m': '1120',
                'y_m': '2120',
                'surface_elevation_m': '121',
                'pick_count': '7',
                'residual_rms_ms': '3.4',
                'datum_status': 'ok',
                'static_status': 'ok',
            },
            {
                'kind': 'source',
                'endpoint_key': 'S001_DUP',
                'station_id': '1001',
                'node_id': '45',
                'x_m': '1001',
                'y_m': '2001',
                'surface_elevation_m': '121',
                'pick_count': '4',
                'residual_rms_ms': '12.4',
                'datum_status': 'ok',
                'static_status': 'ok',
            },
        ],
    )
    _write_csv(
        artifacts_dir / REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
        [
            {
                'endpoint_kind': 'source',
                'endpoint_key': 'S001',
                'static_status': 'ok',
            },
            {
                'endpoint_kind': 'source',
                'endpoint_key': 'S002',
                'static_status': 'invalid_weathering',
            },
        ],
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
