from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import RefractionStaticQcBundleRequest
from app.main import app
from app.services.refraction_static_artifacts import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_REQUEST_JSON_NAME,
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
