from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.statics.refraction.artifacts import (
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)
from app.statics.refraction.domain.export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)


@pytest.fixture()
def refraction_drilldown_client(tmp_path: Path):
    artifacts_dir = tmp_path / 'refraction-job'
    artifacts_dir.mkdir()
    _write_refraction_drilldown_artifacts(artifacts_dir)

    state = app.state.sv
    job_id = 'refraction-drilldown-job'
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
        yield client, job_id

    with state.lock:
        state.jobs.clear()


def test_refraction_qc_endpoint_drilldown_source(refraction_drilldown_client):
    client, job_id = refraction_drilldown_client

    response = client.post(
        '/statics/refraction/qc/drilldown',
        json={
            'job_id': job_id,
            'target': {
                'kind': 'endpoint',
                'endpoint_kind': 'source',
                'endpoint_key': 'source:1001',
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body['sign_convention'] == REFRACTION_STATIC_REPO_SIGN_CONVENTION
    assert body['endpoint']['endpoint_key'] == 'source:1001'
    assert body['static_components']['total_applied_shift_ms'] == -8.0
    assert body['time_terms']['t1_ms'] == 12.5
    assert body['thicknesses']['sh2_weathering_thickness_m'] == 4.0
    assert body['velocities']['v2_m_s'] == 2400.0
    assert body['pick_counts']['used_pick_count'] == 2
    assert body['observations']['total_count'] == 3


def test_refraction_qc_endpoint_drilldown_receiver(refraction_drilldown_client):
    client, job_id = refraction_drilldown_client

    response = client.post(
        '/statics/refraction/qc/drilldown',
        json={
            'job_id': job_id,
            'target': {
                'kind': 'endpoint',
                'endpoint_kind': 'receiver',
                'endpoint_key': 'receiver:2001',
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body['endpoint']['endpoint_kind'] == 'receiver'
    assert body['endpoint']['endpoint_key'] == 'receiver:2001'
    assert body['static_components']['total_applied_shift_ms'] == -6.5
    assert body['observations']['total_count'] == 3


def test_refraction_qc_cell_drilldown_v2_cell(refraction_drilldown_client):
    client, job_id = refraction_drilldown_client

    response = client.post(
        '/statics/refraction/qc/drilldown',
        json={
            'job_id': job_id,
            'target': {
                'kind': 'cell',
                'layer_kind': 'v2_t1',
                'cell_ix': 3,
                'cell_iy': 2,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body['cell']['cell_id'] == 23
    assert body['velocity']['velocity_m_s'] == 2500.0
    assert body['velocity']['velocity_status'] == 'ok'
    assert body['fold']['n_used_observations'] == 2
    assert body['endpoint_counts'] == {'source_count': 1, 'receiver_count': 2}
    assert body['observations']['total_count'] == 3
    assert body['neighbor_velocity_summary']['neighbor_count'] == 2


def test_refraction_qc_drilldown_caps_observations(refraction_drilldown_client):
    client, job_id = refraction_drilldown_client

    response = client.post(
        '/statics/refraction/qc/drilldown',
        json={
            'job_id': job_id,
            'max_observations': 2,
            'target': {
                'kind': 'endpoint',
                'endpoint_kind': 'source',
                'endpoint_key': 'source:1001',
            },
        },
    )

    assert response.status_code == 200
    observations = response.json()['observations']
    assert observations['total_count'] == 3
    assert observations['returned_count'] == 2
    assert observations['capped'] is True
    assert [row['observation_index'] for row in observations['records']] == [0, 1]


def test_refraction_qc_drilldown_rejects_unknown_target(refraction_drilldown_client):
    client, job_id = refraction_drilldown_client

    response = client.post(
        '/statics/refraction/qc/drilldown',
        json={
            'job_id': job_id,
            'target': {
                'kind': 'endpoint',
                'endpoint_kind': 'source',
                'endpoint_key': 'source:missing',
            },
        },
    )

    assert response.status_code == 404
    assert 'source:missing' in response.json()['detail']


def _write_refraction_drilldown_artifacts(artifacts_dir: Path) -> None:
    (artifacts_dir / REFRACTION_STATIC_QC_JSON_NAME).write_text(
        json.dumps({'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION}),
        encoding='utf-8',
    )
    _write_csv(
        artifacts_dir / SOURCE_STATIC_TABLE_CSV_NAME,
        [
            {
                'endpoint_kind': 'source',
                'source_endpoint_key': 'source:1001',
                'source_id': '1001',
                'source_node_id': '11',
                'source_v2_cell_id': '23',
                't1_ms': '12.5',
                't2_ms': '20.0',
                't3_ms': '31.0',
                'v1_m_s': '800.0',
                'v2_m_s': '2400.0',
                'v2_status': 'ok',
                'v3_m_s': '3600.0',
                'vsub_m_s': '5000.0',
                'sh1_weathering_thickness_m': '10.0',
                'sh2_weathering_thickness_m': '4.0',
                'sh3_weathering_thickness_m': '2.0',
                'total_weathering_thickness_m': '16.0',
                'weathering_correction_ms': '-5.0',
                'floating_datum_correction_ms': '-1.0',
                'flat_datum_correction_ms': '-2.0',
                'elevation_correction_ms': '-3.0',
                'total_static_ms': '-8.0',
                'total_applied_shift_ms': '-8.0',
                'solution_status': 'ok',
                'static_status': 'ok',
                'pick_count': '3',
                'used_pick_count': '2',
                'residual_rms_ms': '1.5',
                'residual_mad_ms': '0.8',
            }
        ],
    )
    _write_csv(
        artifacts_dir / RECEIVER_STATIC_TABLE_CSV_NAME,
        [
            {
                'endpoint_kind': 'receiver',
                'receiver_endpoint_key': 'receiver:2001',
                'receiver_id': '2001',
                'receiver_node_id': '21',
                'receiver_v2_cell_id': '23',
                't1_ms': '11.0',
                'v1_m_s': '800.0',
                'v2_m_s': '2420.0',
                'v2_status': 'ok',
                'sh1_weathering_thickness_m': '9.0',
                'total_weathering_thickness_m': '9.0',
                'weathering_correction_ms': '-4.0',
                'floating_datum_correction_ms': '-1.0',
                'flat_datum_correction_ms': '-1.5',
                'elevation_correction_ms': '-2.5',
                'total_static_ms': '-6.5',
                'total_applied_shift_ms': '-6.5',
                'solution_status': 'ok',
                'static_status': 'ok',
                'pick_count': '2',
                'used_pick_count': '2',
                'residual_rms_ms': '1.2',
                'residual_mad_ms': '0.5',
            }
        ],
    )
    _write_csv(
        artifacts_dir / REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
        [
            _observation_row(
                observation_index=2,
                source='source:1001',
                receiver='receiver:2002',
                cell_ix=4,
                residual_ms='3.0',
            ),
            _observation_row(
                observation_index=0,
                source='source:1001',
                receiver='receiver:2001',
                cell_ix=3,
                residual_ms='1.0',
            ),
            _observation_row(
                observation_index=1,
                source='source:1001',
                receiver='receiver:2001',
                cell_ix=3,
                residual_ms='-2.0',
            ),
            _observation_row(
                observation_index=3,
                source='source:1002',
                receiver='receiver:2001',
                cell_ix=3,
                residual_ms='0.5',
            ),
        ],
    )
    _write_csv(
        artifacts_dir / REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
        [
            _cell_row(cell_id=22, ix=2, iy=2, velocity='2400.0'),
            _cell_row(cell_id=23, ix=3, iy=2, velocity='2500.0'),
            _cell_row(cell_id=24, ix=4, iy=2, velocity='2600.0'),
        ],
    )


def _observation_row(
    *,
    observation_index: int,
    source: str,
    receiver: str,
    cell_ix: int,
    residual_ms: str,
) -> dict[str, str]:
    return {
        'observation_index': str(observation_index),
        'trace_index_sorted': str(observation_index + 10),
        'source_endpoint_key': source,
        'receiver_endpoint_key': receiver,
        'offset_m': '500.0',
        'observed_first_break_time_s': '0.200',
        'modeled_first_break_time_s': '0.199',
        'residual_time_s': str(float(residual_ms) / 1000.0),
        'residual_time_ms': residual_ms,
        'layer_kind': 'v2_t1',
        'cell_ix': str(cell_ix),
        'cell_iy': '2',
        'used_in_solve': 'true',
        'rejection_reason': '',
        'status': 'ok',
        'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
    }


def _cell_row(*, cell_id: int, ix: int, iy: int, velocity: str) -> dict[str, str]:
    return {
        'cell_id': str(cell_id),
        'ix': str(ix),
        'iy': str(iy),
        'cell_ix': str(ix),
        'cell_iy': str(iy),
        'cell_velocity_layer_kind': 'v2_t1',
        'cell_velocity_component': 'v2',
        'velocity_m_s': velocity,
        'v2_m_s': velocity,
        'slowness_s_per_m': '0.0004',
        'initial_velocity_m_s': '2450.0',
        'velocity_update_from_initial_m_s': '50.0',
        'velocity_status': 'ok',
        'status_reason': 'ok',
        'active': 'true',
        'n_observations': '3',
        'n_used_observations': '2',
        'n_rejected_observations': '1',
        'n_sources': '1',
        'n_receivers': '2',
        'residual_rms_ms': '1.8',
        'residual_mad_ms': '0.7',
        'residual_mean_ms': '0.1',
        'residual_p95_abs_ms': '2.5',
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    assert rows
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
