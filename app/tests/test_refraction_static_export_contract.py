from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
from app.api.schemas import (
    REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS,
    RefractionStaticApplyRequest,
    RefractionStaticExportJobRequest,
)
from app.main import app
from app.services.refraction_static_artifacts import (
    FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_REQUEST_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)
from app.services.refraction_static_export_service import (
    REFRACTION_STATIC_EXPORT_JOB_META_JSON_NAME,
    REFRACTION_STATIC_EXPORT_REQUEST_JSON_NAME,
    run_refraction_static_export_job,
)
from app.services.refraction_static_export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)
from app.services.refraction_static_lsst_export import (
    REFRACTION_LSST_CSV_NAME,
    REFRACTION_LSST_PLUS_CSV_NAME,
)
from app.tests._refraction_static_synthetic import synthetic_refraction_apply_request


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


def _apply_payload() -> dict[str, Any]:
    payload = synthetic_refraction_apply_request().model_dump(mode='json')
    payload.pop('export', None)
    return payload


def _source_artifacts_for_default_export() -> tuple[str, ...]:
    return (
        REFRACTION_STATIC_REQUEST_JSON_NAME,
        REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    )


def _create_source_refraction_job(
    client: TestClient,
    tmp_path: Path,
    *,
    job_id: str = 'source-refraction-job',
    artifact_names: tuple[str, ...] = (),
    status: str = 'done',
) -> Path:
    job_dir = tmp_path / 'jobs' / job_id
    job_dir.mkdir(parents=True)
    for artifact_name in artifact_names:
        _write_source_artifact_stub(job_dir / artifact_name, artifact_name)

    state = client.app.state.sv
    with state.lock:
        state.jobs.create_static_job(
            job_id,
            file_id='raw-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction',
            artifacts_dir=str(job_dir),
        )
        if status == 'done':
            state.jobs.mark_done(job_id, progress_1=True)
        else:
            state.jobs.set_status(job_id, status)
    return job_dir


def test_refraction_apply_accepts_export_block(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    payload = _apply_payload()
    payload['export'] = {'enabled': True, 'formats': []}

    response = client.post('/statics/refraction/apply', json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body['requested_formats'] == list(REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS)
    assert len(started) == 1
    req = started[0]['args'][1]
    assert isinstance(req, RefractionStaticApplyRequest)
    assert req.export.enabled is True
    assert req.export.formats == []
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[body['job_id']])
    assert job['export_formats'] == list(REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS)


def test_refraction_apply_legacy_request_without_export_still_valid(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )

    response = client.post('/statics/refraction/apply', json=_apply_payload())

    assert response.status_code == 200
    assert 'requested_formats' not in response.json()
    assert len(started) == 1


def test_refraction_export_job_request_requires_source_job_id(
    client: TestClient,
) -> None:
    response = client.post(
        '/statics/refraction/export',
        json={'export': {'enabled': True}},
    )

    assert response.status_code == 422


def test_refraction_export_rejects_unknown_format(client: TestClient) -> None:
    response = client.post(
        '/statics/refraction/export',
        json={
            'source_job_id': 'source-refraction-job',
            'export': {'enabled': True, 'formats': ['unknown_format']},
        },
    )

    assert response.status_code == 422


def test_refraction_export_rejects_incomplete_source_job(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _create_source_refraction_job(
        client,
        tmp_path,
        artifact_names=(
            REFRACTION_STATIC_REQUEST_JSON_NAME,
            REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
        ),
    )

    response = client.post(
        '/statics/refraction/export',
        json={
            'source_job_id': 'source-refraction-job',
            'export': {'enabled': True, 'formats': []},
        },
    )

    assert response.status_code == 409
    assert SOURCE_STATIC_TABLE_CSV_NAME in response.json()['detail']


def test_refraction_export_endpoint_starts_metadata_job(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    _create_source_refraction_job(
        client,
        tmp_path,
        artifact_names=_source_artifacts_for_default_export(),
    )

    response = client.post(
        '/statics/refraction/export',
        json={
            'source_job_id': 'source-refraction-job',
            'export': {'enabled': True, 'formats': []},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body['source_job_id'] == 'source-refraction-job'
    assert body['requested_formats'] == list(REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS)
    assert len(started) == 1
    assert started[0]['target'] is statics_router_module.run_refraction_static_export_job
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[body['job_id']])
    assert job['statics_kind'] == 'refraction_export'
    assert job['source_job_id'] == 'source-refraction-job'
    assert job['export_formats'] == list(REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS)


def test_run_refraction_static_export_job_writes_requested_format_metadata(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _create_source_refraction_job(
        client,
        tmp_path,
        artifact_names=_source_artifacts_for_default_export(),
    )
    req = RefractionStaticExportJobRequest.model_validate(
        {
            'source_job_id': 'source-refraction-job',
            'export': {'enabled': True, 'formats': []},
        }
    )
    export_job_id = 'export-job'
    export_job_dir = tmp_path / 'jobs' / export_job_id
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            export_job_id,
            file_id='raw-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction_export',
            artifacts_dir=str(export_job_dir),
        )

    run_refraction_static_export_job(export_job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[export_job_id])
    assert job['status'] == 'done'
    meta = json.loads(
        (export_job_dir / REFRACTION_STATIC_EXPORT_JOB_META_JSON_NAME).read_text(
            encoding='utf-8',
        )
    )
    request_meta = json.loads(
        (export_job_dir / REFRACTION_STATIC_EXPORT_REQUEST_JSON_NAME).read_text(
            encoding='utf-8',
        )
    )
    assert meta == request_meta
    assert meta['statics_kind'] == 'refraction_export'
    assert meta['source_job_id'] == 'source-refraction-job'
    assert meta['request']['export']['formats'] == []
    assert meta['export']['requested_formats'] == list(
        REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS
    )
    assert meta['export']['sign_convention'] == REFRACTION_STATIC_REPO_SIGN_CONVENTION
    assert meta['generated_artifacts'] == [REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME]
    spreadsheet_rows = _read_csv_text(
        (export_job_dir / REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME).read_text(
            encoding='utf-8',
        )
    )
    assert [row['endpoint_kind'] for row in spreadsheet_rows] == [
        'source',
        'receiver',
    ]
    assert spreadsheet_rows[0]['source_job_id'] == 'source-refraction-job'
    assert spreadsheet_rows[0]['endpoint_key'] == 'source:1001'
    assert spreadsheet_rows[0]['total_applied_shift_ms'] == '12.500000'
    assert spreadsheet_rows[1]['endpoint_key'] == 'receiver:2001'
    assert spreadsheet_rows[1]['total_applied_shift_ms'] == '-3.250000'


def test_run_refraction_static_export_job_time_term_spreadsheet_filters_inactive(
    client: TestClient,
    tmp_path: Path,
) -> None:
    source_dir = _create_source_refraction_job(
        client,
        tmp_path,
        artifact_names=_source_artifacts_for_default_export(),
    )
    with (source_dir / SOURCE_STATIC_TABLE_CSV_NAME).open(
        encoding='utf-8',
        newline='',
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    inactive_row = dict(source_rows[0])
    inactive_row.update(
        {
            'source_endpoint_key': 'source:inactive',
            'source_id': '1002',
            'source_node_id': '11',
            'static_status': 'inactive',
        }
    )
    _write_csv_rows(source_dir / SOURCE_STATIC_TABLE_CSV_NAME, source_rows + [inactive_row])

    req = RefractionStaticExportJobRequest.model_validate(
        {
            'source_job_id': 'source-refraction-job',
            'export': {
                'enabled': True,
                'formats': ['time_term_spreadsheet'],
                'include_inactive_endpoints': False,
            },
        }
    )
    export_job_id = 'export-job-filtered'
    export_job_dir = tmp_path / 'jobs' / export_job_id
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            export_job_id,
            file_id='raw-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction_export',
            artifacts_dir=str(export_job_dir),
        )

    run_refraction_static_export_job(export_job_id, req, client.app.state.sv)

    rows = _read_csv_text(
        (export_job_dir / REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME).read_text(
            encoding='utf-8',
        )
    )
    assert [row['endpoint_key'] for row in rows] == ['source:1001', 'receiver:2001']


def test_run_refraction_static_export_job_time_term_spreadsheet_includes_inactive(
    client: TestClient,
    tmp_path: Path,
) -> None:
    source_dir = _create_source_refraction_job(
        client,
        tmp_path,
        artifact_names=_source_artifacts_for_default_export(),
    )
    with (source_dir / SOURCE_STATIC_TABLE_CSV_NAME).open(
        encoding='utf-8',
        newline='',
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    inactive_row = dict(source_rows[0])
    inactive_row.update(
        {
            'source_endpoint_key': 'source:inactive',
            'source_id': '1002',
            'source_node_id': '11',
            'static_status': 'inactive',
        }
    )
    _write_csv_rows(source_dir / SOURCE_STATIC_TABLE_CSV_NAME, source_rows + [inactive_row])

    req = RefractionStaticExportJobRequest.model_validate(
        {
            'source_job_id': 'source-refraction-job',
            'export': {
                'enabled': True,
                'formats': ['time_term_spreadsheet'],
                'include_inactive_endpoints': True,
            },
        }
    )
    export_job_id = 'export-job-with-inactive'
    export_job_dir = tmp_path / 'jobs' / export_job_id
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            export_job_id,
            file_id='raw-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction_export',
            artifacts_dir=str(export_job_dir),
        )

    run_refraction_static_export_job(export_job_id, req, client.app.state.sv)

    rows = _read_csv_text(
        (export_job_dir / REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME).read_text(
            encoding='utf-8',
        )
    )
    assert [row['endpoint_key'] for row in rows] == [
        'source:1001',
        'source:inactive',
        'receiver:2001',
    ]
    assert rows[1]['static_status'] == 'inactive'


def test_run_refraction_static_export_job_writes_lsst_artifacts(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _create_source_refraction_job(
        client,
        tmp_path,
        artifact_names=(
            REFRACTION_STATIC_REQUEST_JSON_NAME,
            REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
            SOURCE_STATIC_TABLE_CSV_NAME,
            RECEIVER_STATIC_TABLE_CSV_NAME,
        ),
    )
    req = RefractionStaticExportJobRequest.model_validate(
        {
            'source_job_id': 'source-refraction-job',
            'export': {'enabled': True, 'formats': ['lsst', 'lsst_plus']},
        }
    )
    export_job_id = 'export-job'
    export_job_dir = tmp_path / 'jobs' / export_job_id
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            export_job_id,
            file_id='raw-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction_export',
            artifacts_dir=str(export_job_dir),
        )

    run_refraction_static_export_job(export_job_id, req, client.app.state.sv)

    lsst_rows = _read_csv_text(
        (export_job_dir / REFRACTION_LSST_CSV_NAME).read_text(encoding='utf-8')
    )
    lsst_plus_rows = _read_csv_text(
        (export_job_dir / REFRACTION_LSST_PLUS_CSV_NAME).read_text(encoding='utf-8')
    )
    assert [row['endpoint_kind'] for row in lsst_rows] == ['source', 'receiver']
    assert lsst_rows[0]['endpoint_key'] == 'source:1001'
    assert lsst_rows[0]['total_applied_shift_ms'] == '12.500000'
    assert lsst_rows[1]['endpoint_key'] == 'receiver:2001'
    assert lsst_rows[1]['total_applied_shift_ms'] == '-3.250000'
    assert lsst_plus_rows[0]['format_name'] == 'lsst_plus'
    assert lsst_plus_rows[0]['source_field_shift_ms'] == '1.500000'
    assert lsst_plus_rows[1]['receiver_field_shift_ms'] == '-0.500000'
    meta = json.loads(
        (export_job_dir / REFRACTION_STATIC_EXPORT_JOB_META_JSON_NAME).read_text(
            encoding='utf-8',
        )
    )
    assert meta['generated_artifacts'] == [
        REFRACTION_LSST_CSV_NAME,
        REFRACTION_LSST_PLUS_CSV_NAME,
    ]


def test_run_refraction_static_export_job_writes_first_break_time_artifact(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _create_source_refraction_job(
        client,
        tmp_path,
        artifact_names=(
            REFRACTION_STATIC_REQUEST_JSON_NAME,
            REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
            REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
        ),
    )
    req = RefractionStaticExportJobRequest.model_validate(
        {
            'source_job_id': 'source-refraction-job',
            'export': {'enabled': True, 'formats': ['first_break_time']},
        }
    )
    export_job_id = 'export-first-break-time-job'
    export_job_dir = tmp_path / 'jobs' / export_job_id
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            export_job_id,
            file_id='raw-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction_export',
            artifacts_dir=str(export_job_dir),
        )

    run_refraction_static_export_job(export_job_id, req, client.app.state.sv)

    rows = _read_csv_text(
        (export_job_dir / REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME).read_text(
            encoding='utf-8'
        )
    )
    assert rows[0]['source_endpoint_key'] == 'source:1001'
    assert rows[0]['source_job_id'] == 'source-refraction-job'
    assert rows[0]['observed_first_break_time_ms'] == '50.0'
    assert rows[0]['modeled_first_break_time_ms'] == '48.5'
    assert rows[0]['residual_ms'] == '1.5'
    meta = json.loads(
        (export_job_dir / REFRACTION_STATIC_EXPORT_JOB_META_JSON_NAME).read_text(
            encoding='utf-8',
        )
    )
    assert meta['generated_artifacts'] == [REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME]


def _write_source_artifact_stub(path: Path, artifact_name: str) -> None:
    if artifact_name == SOURCE_STATIC_TABLE_CSV_NAME:
        _write_csv(
            path,
            {
                'endpoint_kind': 'source',
                'source_endpoint_key': 'source:1001',
                'source_id': '1001',
                'source_node_id': '10',
                'x_m': '1000.0',
                'y_m': '2000.0',
                'surface_elevation_m': '25.0',
                't1_ms': '12.5',
                'v1_m_s': '800.0',
                'v2_m_s': '2400.0',
                'sh1_weathering_thickness_m': '8.0',
                'weathering_correction_ms': '10.0',
                'elevation_correction_ms': '2.5',
                'total_static_ms': '12.5',
                'total_applied_shift_ms': '12.5',
                'static_status': 'ok',
                'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
                'source_field_shift_ms': '1.5',
                'source_field_static_status': 'ok',
                'source_total_with_field_shift_ms': '14.0',
            },
        )
        return
    if artifact_name == RECEIVER_STATIC_TABLE_CSV_NAME:
        _write_csv(
            path,
            {
                'endpoint_kind': 'receiver',
                'receiver_endpoint_key': 'receiver:2001',
                'receiver_id': '2001',
                'receiver_node_id': '20',
                'x_m': '1010.0',
                'y_m': '2010.0',
                'surface_elevation_m': '30.0',
                't1_ms': '8.5',
                'v1_m_s': '800.0',
                'v2_m_s': '2300.0',
                'sh1_weathering_thickness_m': '7.0',
                'weathering_correction_ms': '-4.0',
                'elevation_correction_ms': '0.75',
                'total_static_ms': '-3.25',
                'total_applied_shift_ms': '-3.25',
                'static_status': 'ok',
                'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
                'receiver_field_shift_ms': '-0.5',
                'receiver_field_static_status': 'ok',
                'receiver_total_with_field_shift_ms': '-3.75',
            },
        )
        return
    if artifact_name == REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME:
        _write_csv(
            path,
            {
                'format_name': 'first_break_time',
                'format_version': '1',
                'source_job_id': 'source-refraction-job',
                'observation_index': '0',
                'sorted_trace_index': '0',
                'source_endpoint_key': 'source:1001',
                'receiver_endpoint_key': 'receiver:2001',
                'source_id': '1001',
                'receiver_id': '2001',
                'offset_m': '100.0',
                'layer_kind': 'v2_t1',
                'observed_first_break_time_ms': '50.0',
                'modeled_first_break_time_ms': '48.5',
                'residual_ms': '1.5',
                'used_in_solve': 'true',
                'reject_reason': 'ok',
                'sign_convention': FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION,
            },
        )
        return
    path.write_bytes(b'data')


def _write_csv(path: Path, row: dict[str, str]) -> None:
    _write_csv_rows(path, [row])


def _write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    assert rows
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))
