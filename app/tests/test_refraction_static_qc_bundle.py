from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import RefractionStaticQcBundleRequest
from app.main import app
from app.statics.refraction.artifacts import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_GRID_MAP_QC_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_JSON_NAME,
    REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_REQUEST_JSON_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(tmp_path / 'pipeline_jobs'))
    state = app.state.sv
    with state.lock:
        state.jobs.clear()
    with TestClient(app) as test_client:
        yield test_client
    with state.lock:
        state.jobs.clear()


def test_refraction_static_qc_bundle_schema_defaults() -> None:
    req = RefractionStaticQcBundleRequest(job_id='refraction-job-id')

    assert req.include == [
        'summary',
        'first_break',
        'profiles',
        'cells',
        'static_components',
    ]
    assert req.max_points == 20000
    assert req.coordinate_mode == 'auto'


def test_refraction_static_qc_bundle_rejects_unknown_job(
    client: TestClient,
) -> None:
    response = client.post(
        '/statics/refraction/qc',
        json={'job_id': 'missing-refraction-job'},
    )

    assert response.status_code == 404
    assert response.json() == {'detail': 'Job ID not found'}


def test_refraction_static_qc_bundle_rejects_non_refraction_job(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _create_static_job(
        client,
        job_id='datum-job',
        statics_kind='datum',
        job_dir=tmp_path / 'datum-job',
    )

    response = client.post('/statics/refraction/qc', json={'job_id': 'datum-job'})

    assert response.status_code == 409
    assert response.json()['detail'] == 'Job datum-job is not a refraction statics job'


def test_refraction_static_qc_bundle_rejects_incomplete_artifacts(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    job_dir.mkdir()
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc',
        json={'job_id': 'refraction-job'},
    )

    assert response.status_code == 409
    assert (
        response.json()['detail']
        == 'Refraction QC bundle requires artifact refraction_static_artifacts.json'
    )


def test_refraction_static_qc_bundle_returns_summary_and_artifact_refs(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[
            {'trace': '0', 'first_break_residual_ms': '1.25'},
            {'trace': '1', 'first_break_residual_ms': '-0.50'},
        ],
        coordinate_mode='line_2d_projected',
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc',
        json={'job_id': 'refraction-job'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['job_id'] == 'refraction-job'
    assert payload['statics_kind'] == 'refraction'
    assert payload['sign_convention'] == 'corrected(t) = raw(t - shift_s)'
    assert payload['coordinate_mode'] == 'line_2d_projected'
    assert payload['summary']['status'] == 'ok'
    assert payload['summary']['file_id'] == 'file-1'
    assert payload['summary']['key1_byte'] == 189
    assert payload['summary']['key2_byte'] == 193
    assert payload['summary']['workflow'] == 'refraction_statics'
    assert payload['artifacts']['first_break_residuals'] == FIRST_BREAK_RESIDUALS_CSV_NAME
    assert 'summary' in payload['available_views']
    assert 'first_break_residual' in payload['available_views']
    assert payload['views']['first_break_residual']['records'] == [
        {'trace': '0', 'first_break_residual_ms': '1.25'},
        {'trace': '1', 'first_break_residual_ms': '-0.50'},
    ]


def test_refraction_static_qc_bundle_keeps_same_stem_artifact_refs(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[
            REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
            REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
            REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc',
        json={'job_id': 'refraction-job'},
    )

    assert response.status_code == 200
    artifacts = response.json()['artifacts']
    assert artifacts['refraction_first_break_fit_qc_csv'] == (
        REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME
    )
    assert artifacts['refraction_first_break_fit_qc_npz'] == (
        REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME
    )
    assert artifacts['refraction_first_break_fit_qc_json'] == (
        REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME
    )
    assert 'refraction_first_break_fit_qc' not in artifacts


def test_refraction_static_qc_bundle_downsamples_deterministically(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[
            {'trace': str(index), 'first_break_residual_ms': str(index * 10)}
            for index in range(5)
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc',
        json={
            'job_id': 'refraction-job',
            'include': ['first_break'],
            'max_points': 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    view = payload['views']['first_break_residual']
    assert [row['trace'] for row in view['records']] == ['0', '2', '4']
    assert payload['downsampling']['first_break_residual'] == {
        'total_points': 5,
        'returned_points': 3,
        'downsampled': True,
        'method': 'even_index_floor_first_last',
    }


def test_refraction_static_qc_bundle_uses_line_profile_qc_artifact(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[
            REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
            REFRACTION_LINE_PROFILE_QC_JSON_NAME,
        ],
    )
    profile_rows = [
        {
            'endpoint_kind': 'source',
            'endpoint_key': 'S001',
            'inline_m': '10.0',
            't1_ms': '12.5',
            'static_status': 'ok',
            'solution_status': 'ok',
        },
        {
            'endpoint_kind': 'receiver',
            'endpoint_key': 'R001',
            'inline_m': '11.0',
            't1_ms': '13.5',
            'static_status': 'ok',
            'solution_status': 'ok',
        },
    ]
    with (job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME).open(
        'w',
        encoding='utf-8',
        newline='',
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)
    _write_line_profile_qc_metadata(
        job_dir / REFRACTION_LINE_PROFILE_QC_JSON_NAME,
        status='available',
        availability_reason='line_2d_projected',
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc',
        json={'job_id': 'refraction-job', 'include': ['profiles']},
    )

    assert response.status_code == 200
    payload = response.json()
    assert 'line_profiles' in payload['available_views']
    assert payload['views']['line_profiles']['artifact'] == (
        REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME
    )
    assert payload['views']['line_profiles']['records'] == profile_rows
    assert payload['unavailable_view_reasons'] == {}


def test_refraction_static_qc_bundle_marks_unavailable_line_profile_from_json_status(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[
            REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
            REFRACTION_LINE_PROFILE_QC_JSON_NAME,
        ],
    )
    _write_empty_csv(
        job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        ['endpoint_kind', 'endpoint_key', 'inline_m'],
    )
    _write_line_profile_qc_metadata(
        job_dir / REFRACTION_LINE_PROFILE_QC_JSON_NAME,
        status='unavailable',
        availability_reason='no_projected_inline_coordinate_model',
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc',
        json={'job_id': 'refraction-job', 'include': ['profiles']},
    )

    assert response.status_code == 200
    payload = response.json()
    assert 'line_profiles' not in payload['available_views']
    assert 'profiles' in payload['unavailable_views']
    assert 'line_profiles' not in payload['views']
    assert 'line_profiles' not in payload['downsampling']
    assert payload['unavailable_view_reasons'] == {
        'profiles': 'no_projected_inline_coordinate_model',
    }


def test_refraction_static_qc_bundle_preserves_unavailable_profile_artifact_refs(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[
            REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
            REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
            REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
            REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
            REFRACTION_LINE_PROFILE_QC_JSON_NAME,
        ],
    )
    _write_line_profile_qc_metadata(
        job_dir / REFRACTION_LINE_PROFILE_QC_JSON_NAME,
        status='unavailable',
        availability_reason='projected_inline_coordinates_unavailable_for_grid_3d',
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc',
        json={'job_id': 'refraction-job', 'include': ['profiles']},
    )

    assert response.status_code == 200
    artifacts = response.json()['artifacts']
    assert artifacts['refraction_line_profile_qc_source'] == (
        REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME
    )
    assert artifacts['refraction_line_profile_qc_receiver'] == (
        REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME
    )
    assert artifacts['refraction_line_profile_qc_combined'] == (
        REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME
    )
    assert artifacts['refraction_line_profile_qc_npz'] == (
        REFRACTION_LINE_PROFILE_QC_NPZ_NAME
    )
    assert artifacts['refraction_line_profile_qc_json'] == (
        REFRACTION_LINE_PROFILE_QC_JSON_NAME
    )


def test_refraction_static_qc_bundle_uses_grid_map_qc_artifact(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        coordinate_mode='grid_3d',
        extra_artifact_names=[REFRACTION_GRID_MAP_QC_CSV_NAME],
    )
    grid_rows = [
        {
            'layer_kind': 'v2_t1',
            'cell_ix': '0',
            'cell_iy': '0',
            'cell_center_x_m': '25.0',
            'cell_center_y_m': '75.0',
            'velocity_m_s': '2400.0',
            'initial_velocity_m_s': '2300.0',
            'velocity_update_from_initial_m_s': '100.0',
            'n_observations': '8',
            'residual_rms_ms': '4.5',
            'residual_mad_ms': '3.0',
            'status': 'solved',
            'status_reason': 'ok',
        },
    ]
    with (job_dir / REFRACTION_GRID_MAP_QC_CSV_NAME).open(
        'w',
        encoding='utf-8',
        newline='',
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_rows[0]))
        writer.writeheader()
        writer.writerows(grid_rows)
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc',
        json={'job_id': 'refraction-job', 'include': ['cells']},
    )

    assert response.status_code == 200
    payload = response.json()
    assert 'refraction_grid_map_qc' in payload['available_views']
    assert payload['views']['refraction_grid_map_qc']['artifact'] == (
        REFRACTION_GRID_MAP_QC_CSV_NAME
    )
    assert payload['views']['refraction_grid_map_qc']['records'] == grid_rows


def test_refraction_static_qc_bundle_uses_static_component_qc_artifacts(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[
            REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
            REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
        ],
    )
    endpoint_rows = [
        {
            'endpoint_kind': 'source',
            'endpoint_key': 'S001',
            'computed_field_correction_ms': '4.5',
            'applied_field_correction_ms': '0.0',
            'total_applied_shift_ms': '-8.0',
        }
    ]
    trace_rows = [
        {
            'trace_index_sorted': '0',
            'source_endpoint_key': 'S001',
            'receiver_endpoint_key': 'R001',
            'computed_field_shift_ms': '6.5',
            'applied_field_shift_ms': '0.0',
            'final_trace_shift_ms': '-4.0',
        }
    ]
    _write_csv(job_dir / REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME, endpoint_rows)
    _write_csv(job_dir / REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME, trace_rows)
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc',
        json={'job_id': 'refraction-job', 'include': ['static_components']},
    )

    assert response.status_code == 200
    payload = response.json()
    assert 'static_component_qc_endpoint' in payload['available_views']
    assert 'static_component_qc_trace' in payload['available_views']
    assert payload['views']['static_component_qc_endpoint']['artifact'] == (
        REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME
    )
    assert payload['views']['static_component_qc_endpoint']['records'] == endpoint_rows
    assert payload['views']['static_component_qc_trace']['artifact'] == (
        REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME
    )
    assert payload['views']['static_component_qc_trace']['records'] == trace_rows


def test_refraction_static_station_structure_uses_endpoint_side_fields(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[
            REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
            REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
        ],
    )
    endpoint_rows = [
        {
            'endpoint_kind': 'source',
            'endpoint_key': 'source:100',
            'source_id': '100',
            'receiver_id': '',
            'global_receiver_number': '100',
            't1_ms': '8.0',
            'v2_m_s': '2400',
            'sh1_m': '12.5',
            'static_status': 'ok',
        },
        {
            'endpoint_kind': 'source',
            'endpoint_key': 'source:101',
            'source_id': '101',
            'receiver_id': '',
            'global_receiver_number': '101',
            't1_ms': '8.5',
            'v2_m_s': '2450',
            'sh1_m': '13.5',
            'static_status': 'ok',
        },
        {
            'endpoint_kind': 'receiver',
            'endpoint_key': 'receiver:200',
            'source_id': '',
            'receiver_id': '200',
            'global_receiver_number': '200',
            't1_ms': '18.0',
            'v2_m_s': '2600',
            'sh1_m': '22.5',
            'static_status': 'ok',
        },
        {
            'endpoint_kind': 'receiver',
            'endpoint_key': 'receiver:201',
            'source_id': '',
            'receiver_id': '201',
            'global_receiver_number': '201',
            't1_ms': '19.0',
            'v2_m_s': '2650',
            'sh1_m': '23.5',
            'static_status': 'ok',
        },
    ]
    _write_csv(job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME, endpoint_rows)
    _write_csv(
        job_dir / REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
        [
            {'source_id': '100', 'receiver_id': '200'},
            {'source_id': '101', 'receiver_id': '201'},
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={
            'job_id': 'refraction-job',
            'gather_start': 101,
            'gather_end': 101,
            'velocity_field': 'v2',
            'depth_field': 'sh1',
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['view_kind'] == 'station_structure'
    assert payload['x_axis'] == 'global_station_number'
    assert payload['x_axis_label'] == 'Global station number'
    assert payload['x_axis_status'] == 'ok'
    assert payload['station_mapping']['source_method'] == 'explicit_global_receiver_number'
    assert payload['station_mapping']['receiver_method'] == 'explicit_global_receiver_number'
    assert payload['filter_status'] == 'ok'
    assert payload['colors'] == {'source': 'cyan', 'receiver': 'red'}
    assert payload['time_term']['source']['endpoint_key'] == ['source:101']
    assert payload['time_term']['source']['y'] == [8.5]
    assert payload['velocity']['field'] == 'v2'
    assert payload['velocity']['receiver']['endpoint_key'] == ['receiver:201']
    assert payload['velocity']['receiver']['y'] == [2650.0]
    assert payload['depth']['field'] == 'sh1'
    assert payload['depth']['receiver']['y'] == [23.5]


def test_refraction_static_station_structure_maps_source_coordinate_to_receiver_station(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME],
    )
    endpoint_rows: list[dict[str, str]] = []
    for index in range(96):
        endpoint_rows.append(
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': str(2000 + index),
                'node_id': str(index),
                'inline_m': str(index * 25.0),
                'x_m': str(index * 25.0),
                'source_id': '',
                'receiver_id': '',
                't1_ms': str(10.0 + index),
                'v2_m_s': '2400',
                'sh1_m': '12.5',
                'static_status': 'ok',
            }
        )
    for source_id in range(1, 25):
        x_m = float((source_id - 1) * 100.0)
        endpoint_rows.append(
            {
                'endpoint_kind': 'source',
                'endpoint_key': f'source:{source_id}',
                'node_id': str(100 + source_id),
                'inline_m': str(x_m),
                'x_m': str(x_m),
                'source_id': str(source_id),
                'receiver_id': '',
                't1_ms': str(20.0 + source_id),
                'v2_m_s': '2500',
                'sh1_m': '15.5',
                'static_status': 'ok',
            }
        )
    _write_csv(job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME, endpoint_rows)
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job'},
    )

    assert response.status_code == 200
    payload = response.json()
    source_x = payload['time_term']['source']['x']
    assert source_x[:3] == [1, 5, 9]
    assert source_x[18] == 73
    assert source_x[23] == 93
    assert payload['velocity']['source']['x'] == source_x
    assert payload['depth']['source']['x'] == source_x
    assert payload['time_term']['receiver']['x'][:5] == [1, 2, 3, 4, 5]
    assert payload['x_axis'] == 'global_station_number'
    assert payload['x_axis_label'] == 'Global station number'
    assert payload['station_mapping']['source_method'] == 'coordinate_interpolation'
    assert payload['station_mapping']['receiver_method'] == 'coordinate_order'
    assert payload['station_mapping']['coordinate_field'] == 'inline_m'
    assert payload['warnings'] == []


def test_refraction_static_station_structure_uses_source_x_when_receiver_inline_is_primary(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME],
    )
    endpoint_rows: list[dict[str, str]] = []
    for index in range(8):
        endpoint_rows.append(
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': f'receiver:{index + 1}',
                'inline_m': str(index * 25.0),
                'x_m': str(index * 25.0),
                'source_id': '',
                'receiver_id': '',
                't1_ms': str(10.0 + index),
                'v2_m_s': '2400',
                'sh1_m': '12.5',
            }
        )
    endpoint_rows.append(
        {
            'endpoint_kind': 'source',
            'endpoint_key': 'source:2',
            'inline_m': '',
            'x_m': '100.0',
            'source_id': '2',
            'receiver_id': '',
            't1_ms': '20.0',
            'v2_m_s': '2500',
            'sh1_m': '15.5',
        }
    )
    _write_csv(job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME, endpoint_rows)
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['time_term']['source']['x'] == [5]
    assert payload['velocity']['source']['x'] == [5]
    assert payload['depth']['source']['x'] == [5]
    assert payload['station_mapping']['source_method'] == 'coordinate_interpolation'
    assert payload['station_mapping']['receiver_method'] == 'coordinate_order'
    assert payload['station_mapping']['coordinate_field'] == 'mixed:inline_m,x_m'
    assert payload['warnings'] == []


def test_refraction_static_station_structure_does_not_mix_source_id_with_station_axis(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME],
    )
    _write_csv(
        job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        [
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:1',
                'node_id': '1',
                'inline_m': '0.0',
                'source_id': '',
                'receiver_id': '',
                't1_ms': '18.0',
                'v2_m_s': '2600',
                'sh1_m': '22.5',
            },
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:2',
                'node_id': '2',
                'inline_m': '25.0',
                'source_id': '',
                'receiver_id': '',
                't1_ms': '19.0',
                'v2_m_s': '2610',
                'sh1_m': '23.5',
            },
            {
                'endpoint_kind': 'source',
                'endpoint_key': 'source:19',
                'node_id': '19',
                'inline_m': '',
                'source_id': '19',
                'receiver_id': '',
                't1_ms': '8.0',
                'v2_m_s': '2400',
                'sh1_m': '12.5',
            },
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['x_axis'] == 'global_station_number'
    assert payload['x_axis_status'] == 'ok'
    assert payload['time_term']['source']['x'] == []
    assert payload['velocity']['source']['x'] == []
    assert payload['depth']['source']['x'] == []
    assert payload['station_mapping']['source_method'] == 'unavailable'
    assert payload['station_mapping']['receiver_method'] == 'coordinate_order'
    assert any('coordinate was missing or non-finite' in warning for warning in payload['warnings'])
    assert not any('fell back to source_id' in warning for warning in payload['warnings'])


def test_refraction_static_station_structure_infers_receiver_x_from_station_reference(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME],
    )
    _write_csv(
        job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        [
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:1',
                'global_receiver_number': '1',
                'inline_m': '0.0',
                'source_id': '',
                'receiver_id': '',
                't1_ms': '18.0',
                'v2_m_s': '2600',
                'sh1_m': '22.5',
            },
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:2',
                'global_receiver_number': '',
                'inline_m': '25.0',
                'source_id': '',
                'receiver_id': '',
                't1_ms': '19.0',
                'v2_m_s': '2610',
                'sh1_m': '23.5',
            },
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:3',
                'global_receiver_number': '3',
                'inline_m': '50.0',
                'source_id': '',
                'receiver_id': '',
                't1_ms': '20.0',
                'v2_m_s': '2620',
                'sh1_m': '24.5',
            },
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['x_axis'] == 'global_station_number'
    assert payload['time_term']['receiver']['endpoint_key'] == [
        'receiver:1',
        'receiver:2',
        'receiver:3',
    ]
    assert payload['time_term']['receiver']['x'] == [1, 2, 3]
    assert payload['velocity']['receiver']['x'] == [1, 2, 3]
    assert payload['depth']['receiver']['x'] == [1, 2, 3]
    assert payload['station_mapping']['receiver_method'] == (
        'mixed:explicit_global_receiver_number,coordinate_order'
    )


def test_refraction_static_station_structure_infers_receiver_x_from_single_station_anchor(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME],
    )
    _write_csv(
        job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        [
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:101',
                'global_receiver_number': '101',
                'inline_m': '0.0',
                'source_id': '',
                'receiver_id': '',
                't1_ms': '18.0',
                'v2_m_s': '2600',
                'sh1_m': '22.5',
            },
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:missing-102',
                'global_receiver_number': '',
                'inline_m': '25.0',
                'source_id': '',
                'receiver_id': '',
                't1_ms': '19.0',
                'v2_m_s': '2610',
                'sh1_m': '23.5',
            },
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:missing-103',
                'global_receiver_number': '',
                'inline_m': '50.0',
                'source_id': '',
                'receiver_id': '',
                't1_ms': '20.0',
                'v2_m_s': '2620',
                'sh1_m': '24.5',
            },
            {
                'endpoint_kind': 'source',
                'endpoint_key': 'source:3',
                'source_id': '3',
                'inline_m': '50.0',
                't1_ms': '8.0',
                'v2_m_s': '2400',
                'sh1_m': '12.5',
            },
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['x_axis'] == 'global_station_number'
    assert payload['time_term']['receiver']['x'] == [101, 102, 103]
    assert payload['velocity']['receiver']['x'] == [101, 102, 103]
    assert payload['depth']['receiver']['x'] == [101, 102, 103]
    assert payload['time_term']['source']['x'] == [103]
    assert payload['velocity']['source']['x'] == [103]
    assert payload['depth']['source']['x'] == [103]
    assert payload['station_mapping']['source_method'] == 'coordinate_interpolation'
    assert payload['station_mapping']['receiver_method'] == (
        'mixed:explicit_global_receiver_number,coordinate_order'
    )
    assert payload['station_mapping']['coordinate_field'] == 'inline_m'
    assert payload['warnings'] == []


def test_refraction_static_station_structure_prefers_coordinate_order_over_receiver_id(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME],
    )
    endpoint_rows: list[dict[str, str]] = []
    for index in range(4):
        endpoint_rows.append(
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': f'receiver:{201 + index}',
                'inline_m': str(index * 25.0),
                'x_m': str(index * 25.0),
                'source_id': '',
                'receiver_id': str(201 + index),
                't1_ms': str(18.0 + index),
                'v2_m_s': '2600',
                'sh1_m': '22.5',
            }
        )
    endpoint_rows.append(
        {
            'endpoint_kind': 'source',
            'endpoint_key': 'source:3',
            'inline_m': '50.0',
            'x_m': '50.0',
            'source_id': '3',
            'receiver_id': '',
            't1_ms': '8.0',
            'v2_m_s': '2400',
            'sh1_m': '12.5',
        }
    )
    _write_csv(job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME, endpoint_rows)
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['time_term']['receiver']['x'] == [1, 2, 3, 4]
    assert payload['velocity']['receiver']['x'] == [1, 2, 3, 4]
    assert payload['depth']['receiver']['x'] == [1, 2, 3, 4]
    assert payload['time_term']['source']['x'] == [3]
    assert payload['velocity']['source']['x'] == [3]
    assert payload['depth']['source']['x'] == [3]
    assert payload['station_mapping']['receiver_method'] == 'coordinate_order'
    assert payload['station_mapping']['source_method'] == 'coordinate_interpolation'
    assert payload['station_mapping']['coordinate_field'] == 'inline_m'


def test_refraction_static_station_structure_omits_out_of_range_source_coordinate(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME],
    )
    _write_csv(
        job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        [
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:1',
                'inline_m': '0.0',
                'source_id': '',
                'receiver_id': '',
                't1_ms': '18.0',
                'v2_m_s': '2600',
                'sh1_m': '22.5',
            },
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:2',
                'inline_m': '25.0',
                'source_id': '',
                'receiver_id': '',
                't1_ms': '19.0',
                'v2_m_s': '2610',
                'sh1_m': '23.5',
            },
            {
                'endpoint_kind': 'source',
                'endpoint_key': 'source:19',
                'inline_m': '100.0',
                'source_id': '19',
                'receiver_id': '',
                't1_ms': '8.0',
                'v2_m_s': '2400',
                'sh1_m': '12.5',
            },
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['x_axis'] == 'global_station_number'
    assert payload['x_axis_status'] == 'ok'
    assert payload['time_term']['source']['x'] == []
    assert payload['velocity']['source']['x'] == []
    assert payload['depth']['source']['x'] == []
    assert payload['station_mapping']['source_method'] == 'unavailable'
    assert any('omitted 1 source endpoint' in warning for warning in payload['warnings'])
    assert not any('fell back to source_id' in warning for warning in payload['warnings'])


def test_refraction_static_station_structure_uses_linked_receiver_station_number(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME],
    )
    _write_csv(
        job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        [
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:73',
                'node_id': '7',
                'global_receiver_number': '73',
                'source_id': '',
                't1_ms': '18.0',
                'v2_m_s': '2600',
                'sh1_m': '22.5',
            },
            {
                'endpoint_kind': 'source',
                'endpoint_key': 'source:19',
                'node_id': '7',
                'global_receiver_number': '',
                'source_id': '19',
                't1_ms': '8.0',
                'v2_m_s': '2400',
                'sh1_m': '12.5',
            },
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['time_term']['source']['x'] == [73]
    assert payload['time_term']['receiver']['x'] == [73]
    assert payload['station_mapping']['source_method'] == 'linked_receiver_station_number'


def test_refraction_static_station_structure_warns_when_source_id_fallback_is_used(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME],
    )
    _write_csv(
        job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        [
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:200',
                'source_id': '',
                'receiver_id': '200',
                't1_ms': '18.0',
                'v2_m_s': '2600',
                'sh1_m': '22.5',
            },
            {
                'endpoint_kind': 'source',
                'endpoint_key': 'source:19',
                'source_id': '19',
                'receiver_id': '',
                't1_ms': '8.0',
                'v2_m_s': '2400',
                'sh1_m': '12.5',
            },
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['x_axis'] == 'endpoint_id_fallback'
    assert payload['x_axis_status'] == 'fallback'
    assert payload['x_axis_label'] == 'source/receiver endpoint id fallback'
    assert payload['time_term']['source']['x'] == [19]
    assert payload['station_mapping']['source_method'] == 'source_id_fallback'
    assert any('fell back to source_id' in warning for warning in payload['warnings'])


def test_refraction_static_station_structure_labels_mixed_source_id_fallback(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME],
    )
    _write_csv(
        job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        [
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:200',
                'global_station_number': '',
                'source_id': '',
                'receiver_id': '200',
                't1_ms': '18.0',
                'v2_m_s': '2600',
                'sh1_m': '22.5',
            },
            {
                'endpoint_kind': 'source',
                'endpoint_key': 'source:73',
                'global_station_number': '73',
                'source_id': '18',
                't1_ms': '8.0',
                'v2_m_s': '2400',
                'sh1_m': '12.5',
            },
            {
                'endpoint_kind': 'source',
                'endpoint_key': 'source:19',
                'source_id': '19',
                't1_ms': '9.0',
                'v2_m_s': '2450',
                'sh1_m': '13.5',
            },
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['x_axis'] == 'endpoint_id_fallback'
    assert payload['x_axis_status'] == 'fallback'
    assert payload['x_axis_label'] == 'source/receiver endpoint id fallback'
    assert payload['time_term']['source']['x'] == [19, 73]
    assert payload['station_mapping']['source_method'] == (
        'mixed:explicit_global_station_number,source_id_fallback'
    )
    assert any('fell back to source_id' in warning for warning in payload['warnings'])


def test_refraction_static_station_structure_warns_on_linked_velocity_fallback(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME],
    )
    _write_csv(
        job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        [
            {
                'endpoint_kind': 'source',
                'endpoint_key': 'source:100',
                'source_id': '100',
                'receiver_id': '',
                'global_receiver_number': '100',
                't1_ms': '8.0',
                'linked_node_velocity_m_s': '2400',
                'sh1_m': '12.5',
            },
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:100',
                'source_id': '',
                'receiver_id': '100',
                'global_receiver_number': '100',
                't1_ms': '18.0',
                'linked_node_velocity_m_s': '2400',
                'sh1_m': '22.5',
            },
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job', 'velocity_field': 'v2'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['velocity']['source']['endpoint_key'] == ['source:100']
    assert payload['velocity']['receiver']['endpoint_key'] == ['receiver:100']
    assert payload['velocity']['source']['y'] == [2400.0]
    assert payload['velocity']['receiver']['y'] == [2400.0]
    assert any('linked_node_velocity_m_s' in warning for warning in payload['warnings'])


def test_refraction_static_station_structure_filters_receivers_by_station_fields(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[
            REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
            REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
        ],
    )
    _write_csv(
        job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        [
            {
                'endpoint_kind': 'source',
                'endpoint_key': 'source:10',
                'source_id': '10',
                'receiver_id': '',
                'global_receiver_number': '10',
                'station_number': '10',
                'receiver_number': '10',
                't1_ms': '7.5',
                'v2_m_s': '2300',
                'sh1_m': '10.0',
            },
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:A',
                'receiver_id': '',
                'global_receiver_number': '500',
                'station_number': '1500',
                'receiver_number': '2500',
                't1_ms': '17.0',
                'v2_m_s': '2550',
                'sh1_m': '20.0',
            },
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:B',
                'receiver_id': '',
                'global_receiver_number': '501',
                'station_number': '1501',
                'receiver_number': '2501',
                't1_ms': '18.0',
                'v2_m_s': '2650',
                'sh1_m': '21.0',
            },
        ],
    )
    _write_csv(
        job_dir / REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
        [
            {
                'source_id': '10',
                'global_receiver_number': '501',
            }
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job', 'gather_start': 10, 'gather_end': 10},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['filter_status'] == 'ok'
    assert payload['time_term']['receiver']['endpoint_key'] == ['receiver:B']
    assert payload['velocity']['receiver']['x'] == [501]


def test_refraction_static_station_structure_reports_unavailable_receiver_filter(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[
            SOURCE_STATIC_TABLE_CSV_NAME,
            RECEIVER_STATIC_TABLE_CSV_NAME,
        ],
    )
    _write_csv(
        job_dir / SOURCE_STATIC_TABLE_CSV_NAME,
        [
            {
                'endpoint_kind': 'source',
                'source_endpoint_key': 'source:100',
                'source_id': '100',
                't1_ms': '8.0',
                'v2_m_s': '2400',
                'sh1_weathering_thickness_m': '12.5',
                'static_status': 'ok',
            }
        ],
    )
    _write_csv(
        job_dir / RECEIVER_STATIC_TABLE_CSV_NAME,
        [
            {
                'endpoint_kind': 'receiver',
                'receiver_endpoint_key': 'receiver:200',
                'receiver_id': '200',
                't1_ms': '18.0',
                'v2_m_s': '2600',
                'sh1_weathering_thickness_m': '22.5',
                'static_status': 'ok',
            }
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job', 'gather_start': 100, 'gather_end': 100},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['filter_status'] == 'receiver_participation_unavailable'
    assert 'receiver series is unfiltered' in payload['warnings'][0]
    assert payload['velocity']['source']['x'] == [100]
    assert payload['velocity']['receiver']['x'] == [200]


def test_refraction_static_station_structure_reports_unfilterable_observations(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[
            REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
            REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
        ],
    )
    _write_csv(
        job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        [
            {
                'endpoint_kind': 'source',
                'endpoint_key': 'source:100',
                'source_id': '100',
                'receiver_id': '',
                'global_receiver_number': '100',
                't1_ms': '8.0',
                'v2_m_s': '2400',
                'sh1_m': '12.5',
            },
            {
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:200',
                'source_id': '',
                'receiver_id': '200',
                'global_receiver_number': '200',
                't1_ms': '18.0',
                'v2_m_s': '2600',
                'sh1_m': '22.5',
            },
        ],
    )
    _write_csv(
        job_dir / REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
        [
            {
                'endpoint_key': 'receiver:200',
                'receiver_id': '200',
            }
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job', 'gather_start': 100, 'gather_end': 100},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['filter_status'] == 'receiver_participation_unavailable'
    assert 'do not include usable source gather identifiers' in payload['warnings'][0]
    assert payload['time_term']['receiver']['endpoint_key'] == ['receiver:200']


def test_refraction_static_station_structure_falls_back_from_empty_line_profile(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'refraction-job'
    _write_refraction_qc_artifacts(
        job_dir,
        rows=[{'trace': '0', 'first_break_residual_ms': '1.25'}],
        extra_artifact_names=[
            REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
            SOURCE_STATIC_TABLE_CSV_NAME,
            RECEIVER_STATIC_TABLE_CSV_NAME,
        ],
    )
    _write_empty_csv(
        job_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        [
            'endpoint_kind',
            'endpoint_key',
            'global_receiver_number',
            't1_ms',
            'v2_m_s',
            'sh1_m',
        ],
    )
    _write_csv(
        job_dir / SOURCE_STATIC_TABLE_CSV_NAME,
        [
            {
                'source_endpoint_key': 'source:100',
                'source_id': '100',
                't1_ms': '8.0',
                'v2_m_s': '2400',
                'sh1_weathering_thickness_m': '12.5',
                'static_status': 'ok',
            }
        ],
    )
    _write_csv(
        job_dir / RECEIVER_STATIC_TABLE_CSV_NAME,
        [
            {
                'receiver_endpoint_key': 'receiver:200',
                'receiver_id': '200',
                't1_ms': '18.0',
                'v2_m_s': '2600',
                'sh1_weathering_thickness_m': '22.5',
                'static_status': 'ok',
            }
        ],
    )
    _create_static_job(client, job_id='refraction-job', job_dir=job_dir)

    response = client.post(
        '/statics/refraction/qc/station-structure',
        json={'job_id': 'refraction-job'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['time_term']['source']['endpoint_key'] == ['source:100']
    assert payload['time_term']['receiver']['endpoint_key'] == ['receiver:200']
    assert payload['velocity']['source']['y'] == [2400.0]
    assert payload['depth']['receiver']['y'] == [22.5]


def _create_static_job(
    client: TestClient,
    *,
    job_id: str,
    job_dir: Path,
    statics_kind: str = 'refraction',
) -> None:
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            job_id,
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            statics_kind=statics_kind,
            artifacts_dir=str(job_dir),
        )
        client.app.state.sv.jobs.mark_done(job_id)


def _write_refraction_qc_artifacts(
    job_dir: Path,
    *,
    rows: list[dict[str, str]],
    coordinate_mode: str | None = None,
    extra_artifact_names: list[str] | None = None,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    qc_payload: dict[str, object] = {
        'artifact_version': '1.0',
        'method': 'gli_variable_thickness',
        'workflow': 'refraction_statics',
        'conversion_mode': 't1lsst_1layer',
        'sign_convention': {
            'trace_shift_s': 'corrected(t) = raw(t - shift_s)',
            'positive_shift': 'event appears later in corrected data',
            'negative_shift': 'event appears earlier in corrected data',
        },
        'observations': {'n_used_observations': len(rows)},
        'artifacts': [
            {
                'name': FIRST_BREAK_RESIDUALS_CSV_NAME,
                'kind': 'csv',
                'description': 'GLI first-break residual table',
            },
        ],
    }
    if coordinate_mode is not None:
        qc_payload['refractor_velocity_cells'] = {
            'coordinate_mode': coordinate_mode,
        }

    (job_dir / REFRACTION_STATIC_QC_JSON_NAME).write_text(
        json.dumps(qc_payload),
        encoding='utf-8',
    )
    (job_dir / REFRACTION_STATIC_REQUEST_JSON_NAME).write_text(
        json.dumps({'file_id': 'file-1'}),
        encoding='utf-8',
    )
    manifest_artifacts = [
        {
            'name': REFRACTION_STATIC_QC_JSON_NAME,
            'kind': 'json',
            'required': True,
            'origin': 'final',
        },
        {
            'name': FIRST_BREAK_RESIDUALS_CSV_NAME,
            'kind': 'csv',
            'required': True,
            'origin': 'final',
        },
    ]
    for artifact_name in extra_artifact_names or []:
        (job_dir / artifact_name).write_text('', encoding='utf-8')
        manifest_artifacts.append(
            {
                'name': artifact_name,
                'kind': Path(artifact_name).suffix.removeprefix('.'),
                'required': True,
                'origin': 'final',
            },
        )
    manifest_payload = {
        'artifact_version': '1.0',
        'job_kind': 'statics',
        'statics_kind': 'refraction',
        'artifacts': manifest_artifacts,
    }
    (job_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).write_text(
        json.dumps(manifest_payload),
        encoding='utf-8',
    )
    with (job_dir / FIRST_BREAK_RESIDUALS_CSV_NAME).open(
        'w',
        encoding='utf-8',
        newline='',
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    assert rows
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_empty_csv(path: Path, columns: list[str]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()


def _write_line_profile_qc_metadata(
    path: Path,
    *,
    status: str,
    availability_reason: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                'kind': 'refraction_line_profile_qc',
                'status': status,
                'availability_reason': availability_reason,
            }
        ),
        encoding='utf-8',
    )
