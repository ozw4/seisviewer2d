from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.api.schemas import (
    RefractionStaticExportJobRequest,
    RefractionStaticTableApplyRequest,
)
from app.core.state import AppState
from app.services.refraction_static_artifacts import (
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_REQUEST_JSON_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)
from app.services.refraction_static_export_service import (
    CANONICAL_SOURCE_RECEIVER_STATIC_TABLE_CSV_NAME,
    run_refraction_static_export_job,
)
from app.services.refraction_static_export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)
from app.services.refraction_static_table_apply_service import (
    STATIC_TABLE_APPLY_HISTORY_JSON_NAME,
    STATIC_TABLE_APPLY_QC_JSON_NAME,
    STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME,
    run_refraction_static_table_apply_job,
)
from app.tests.test_refraction_static_apply_trace_store import (
    KEY1,
    KEY2,
    SOURCE_FILE_ID,
)
from app.tests.test_refraction_static_table_apply import (
    DEFAULT_GEOMETRY,
    RECEIVER_KEYS,
    SOURCE_KEYS,
    _write_target_store,
)

SOURCE_REFRACTION_JOB_ID = 'e2e-source-refraction-job'
EXPORT_JOB_ID = 'e2e-refraction-export-job'
APPLY_JOB_ID = 'e2e-static-table-apply-job'
SECOND_APPLY_JOB_ID = 'e2e-static-table-apply-job-2'
THIRD_APPLY_JOB_ID = 'e2e-static-table-apply-job-3'

SOURCE_SHIFTS_MS = {100: 8.0, 101: 4.0}
RECEIVER_SHIFTS_MS = {200: 0.0, 201: -4.0}


def _endpoint_geometry(endpoint_kind: str, endpoint_id: int) -> tuple[float, float, float]:
    if endpoint_kind == 'source':
        return {
            100: (1000.0, 2000.0, 10.0),
            101: (1010.0, 2000.0, 12.0),
        }[endpoint_id]
    return {
        200: (1100.0, 2000.0, 20.0),
        201: (1110.0, 2000.0, 22.0),
    }[endpoint_id]


def _static_table_row(
    *,
    endpoint_kind: str,
    endpoint_id: int,
    applied_shift_ms: float,
    three_layer_metadata: bool = False,
) -> dict[str, str]:
    prefix = endpoint_kind
    x_m, y_m, elevation_m = _endpoint_geometry(endpoint_kind, endpoint_id)
    endpoint_key = (
        SOURCE_KEYS[endpoint_id]
        if endpoint_kind == 'source'
        else RECEIVER_KEYS[endpoint_id]
    )
    row = {
        'endpoint_kind': endpoint_kind,
        f'{prefix}_endpoint_key': endpoint_key,
        f'{prefix}_id': str(endpoint_id),
        f'{prefix}_node_id': str(endpoint_id),
        'x_m': f'{x_m:.3f}',
        'y_m': f'{y_m:.3f}',
        'surface_elevation_m': f'{elevation_m:.3f}',
        't1_ms': '10.000000',
        'v1_m_s': '800.000',
        'v2_m_s': '2400.000',
        'sh1_weathering_thickness_m': '10.000',
        'weathering_correction_ms': '-5.000000',
        'elevation_correction_ms': f'{applied_shift_ms + 5.0:.6f}',
        'total_static_ms': f'{applied_shift_ms:.6f}',
        'total_applied_shift_ms': f'{applied_shift_ms:.6f}',
        'static_status': 'ok',
        'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
    }
    if three_layer_metadata:
        row.update(_three_layer_metadata(endpoint_kind=endpoint_kind))
    return row


def _three_layer_metadata(*, endpoint_kind: str) -> dict[str, str]:
    if endpoint_kind == 'source':
        return {
            't1_ms': '14.000000',
            't2_ms': '22.000000',
            't3_ms': '31.000000',
            'v1_m_s': '900.000',
            'v2_m_s': '2500.000',
            'v3_m_s': '3700.000',
            'vsub_m_s': '5200.000',
            'sh1_weathering_thickness_m': '13.506',
            'sh2_weathering_thickness_m': '25.246',
            'sh3_weathering_thickness_m': '38.785',
        }
    return {
        't1_ms': '11.000000',
        't2_ms': '19.000000',
        't3_ms': '28.000000',
        'v1_m_s': '900.000',
        'v2_m_s': '2480.000',
        'v3_m_s': '3650.000',
        'vsub_m_s': '5100.000',
        'sh1_weathering_thickness_m': '10.624',
        'sh2_weathering_thickness_m': '25.552',
        'sh3_weathering_thickness_m': '38.555',
    }


def _write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _write_source_refraction_job(
    state: AppState,
    tmp_path: Path,
    *,
    source_ids: tuple[int, ...] = (100, 101),
    receiver_ids: tuple[int, ...] = (200, 201),
    three_layer_metadata: bool = False,
) -> Path:
    job_dir = tmp_path / 'jobs' / SOURCE_REFRACTION_JOB_ID
    source_rows = [
        _static_table_row(
            endpoint_kind='source',
            endpoint_id=source_id,
            applied_shift_ms=SOURCE_SHIFTS_MS[source_id],
            three_layer_metadata=three_layer_metadata,
        )
        for source_id in source_ids
    ]
    receiver_rows = [
        _static_table_row(
            endpoint_kind='receiver',
            endpoint_id=receiver_id,
            applied_shift_ms=RECEIVER_SHIFTS_MS[receiver_id],
            three_layer_metadata=three_layer_metadata,
        )
        for receiver_id in receiver_ids
    ]
    _write_csv_rows(job_dir / SOURCE_STATIC_TABLE_CSV_NAME, source_rows)
    _write_csv_rows(job_dir / RECEIVER_STATIC_TABLE_CSV_NAME, receiver_rows)
    (job_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).write_text(
        json.dumps({'artifact_names': []}),
        encoding='utf-8',
    )
    (job_dir / REFRACTION_STATIC_REQUEST_JSON_NAME).write_text(
        json.dumps(
            {
                'job_id': SOURCE_REFRACTION_JOB_ID,
                'job_type': 'statics',
                'statics_kind': 'refraction',
                'source_file_id': SOURCE_FILE_ID,
                'key1_byte': KEY1,
                'key2_byte': KEY2,
                'request': {'geometry': DEFAULT_GEOMETRY},
            }
        ),
        encoding='utf-8',
    )
    with state.lock:
        state.jobs.create_static_job(
            SOURCE_REFRACTION_JOB_ID,
            file_id=SOURCE_FILE_ID,
            key1_byte=KEY1,
            key2_byte=KEY2,
            statics_kind='refraction',
            artifacts_dir=str(job_dir),
        )
        state.jobs.mark_done(SOURCE_REFRACTION_JOB_ID, progress_1=True)
    return job_dir


def _run_export(state: AppState, tmp_path: Path) -> Path:
    export_job_dir = tmp_path / 'jobs' / EXPORT_JOB_ID
    with state.lock:
        state.jobs.create_static_job(
            EXPORT_JOB_ID,
            file_id=SOURCE_FILE_ID,
            key1_byte=KEY1,
            key2_byte=KEY2,
            statics_kind='refraction_export',
            artifacts_dir=str(export_job_dir),
        )
    req = RefractionStaticExportJobRequest.model_validate(
        {
            'source_job_id': SOURCE_REFRACTION_JOB_ID,
            'export': {
                'enabled': True,
                'formats': ['canonical_static_table'],
            },
        }
    )
    run_refraction_static_export_job(EXPORT_JOB_ID, req, state)
    with state.lock:
        job = dict(state.jobs[EXPORT_JOB_ID])
    assert job['status'] == 'done', job.get('message')
    return export_job_dir / CANONICAL_SOURCE_RECEIVER_STATIC_TABLE_CSV_NAME


def _run_apply(
    state: AppState,
    tmp_path: Path,
    *,
    job_id: str = APPLY_JOB_ID,
    file_id: str = SOURCE_FILE_ID,
    register_corrected_file: bool = False,
    missing_static_policy: str = 'fail',
    allow_missing_source_static: bool = False,
    allow_missing_receiver_static: bool = False,
    allow_reapply_same_static_table: bool = False,
) -> Path:
    job_dir = tmp_path / 'jobs' / job_id
    with state.lock:
        state.jobs.create_static_job(
            job_id,
            file_id=file_id,
            key1_byte=KEY1,
            key2_byte=KEY2,
            statics_kind='refraction_static_table_apply',
            artifacts_dir=str(job_dir),
        )
    req = RefractionStaticTableApplyRequest.model_validate(
        {
            'file_id': file_id,
            'key1_byte': KEY1,
            'key2_byte': KEY2,
            'combined_table_artifact_id': (
                f'{EXPORT_JOB_ID}:{CANONICAL_SOURCE_RECEIVER_STATIC_TABLE_CSV_NAME}'
            ),
            'register_corrected_file': register_corrected_file,
            'missing_static_policy': missing_static_policy,
            'allow_missing_source_static': allow_missing_source_static,
            'allow_missing_receiver_static': allow_missing_receiver_static,
            'allow_reapply_same_static_table': allow_reapply_same_static_table,
            'max_abs_shift_ms': 250.0,
        }
    )
    run_refraction_static_table_apply_job(job_id, req, state)
    return job_dir


def _job(state: AppState, job_id: str) -> dict[str, Any]:
    with state.lock:
        return dict(state.jobs[job_id])


def _history(job_dir: Path) -> dict[str, Any]:
    return json.loads(
        (job_dir / STATIC_TABLE_APPLY_HISTORY_JSON_NAME).read_text(encoding='utf-8')
    )


def _imported_metadata_value(
    data: Any,
    *,
    endpoint_kind: str,
    endpoint_key: str,
    column: str,
) -> str:
    prefix = f'{endpoint_kind}_imported'
    keys = np.asarray(data[f'{prefix}_endpoint_key'], dtype=str)
    matches = np.flatnonzero(keys == endpoint_key)
    assert matches.shape == (1,)
    values = np.asarray(data[f'{prefix}_metadata_{column}'], dtype=str)
    return str(values[int(matches[0])])


def test_static_table_export_import_apply_round_trip_synthetic(
    tmp_path: Path,
) -> None:
    state, _store = _write_target_store(tmp_path)
    _write_source_refraction_job(state, tmp_path)
    _run_export(state, tmp_path)

    job_dir = _run_apply(state, tmp_path)

    assert _job(state, APPLY_JOB_ID)['status'] == 'done'
    with np.load(job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
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
            data['source_static_shift_s_sorted'] + data['receiver_static_shift_s_sorted'],
        )
        np.testing.assert_allclose(
            data['trace_shift_s_sorted'],
            [0.008, 0.004, 0.004, 0.0],
        )
        np.testing.assert_array_equal(data['trace_static_status_sorted'], ['ok'] * 4)

    history = _history(job_dir)
    assert history['source_export_format'] == 'canonical_static_table'
    assert history['created_from_refraction_job_id'] == SOURCE_REFRACTION_JOB_ID
    assert history['created_from_export_job_id'] == EXPORT_JOB_ID
    assert history['cumulative_shift_field'] == 'trace_shift_s_sorted'


def test_static_table_apply_exported_three_layer_metadata_preserved(
    tmp_path: Path,
) -> None:
    state, _store = _write_target_store(tmp_path)
    _write_source_refraction_job(state, tmp_path, three_layer_metadata=True)
    _run_export(state, tmp_path)

    job_dir = _run_apply(state, tmp_path)

    assert _job(state, APPLY_JOB_ID)['status'] == 'done'
    with np.load(job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        assert (
            _imported_metadata_value(
                data,
                endpoint_kind='source',
                endpoint_key=SOURCE_KEYS[100],
                column='t1_ms',
            )
            == '14.000000'
        )
        assert (
            _imported_metadata_value(
                data,
                endpoint_kind='source',
                endpoint_key=SOURCE_KEYS[100],
                column='t2_ms',
            )
            == '22.000000'
        )
        assert (
            _imported_metadata_value(
                data,
                endpoint_kind='source',
                endpoint_key=SOURCE_KEYS[100],
                column='t3_ms',
            )
            == '31.000000'
        )
        assert (
            _imported_metadata_value(
                data,
                endpoint_kind='source',
                endpoint_key=SOURCE_KEYS[100],
                column='v1_m_s',
            )
            == '900.000'
        )
        assert (
            _imported_metadata_value(
                data,
                endpoint_kind='source',
                endpoint_key=SOURCE_KEYS[100],
                column='v2_m_s',
            )
            == '2500.000'
        )
        assert (
            _imported_metadata_value(
                data,
                endpoint_kind='source',
                endpoint_key=SOURCE_KEYS[100],
                column='v3_m_s',
            )
            == '3700.000'
        )
        assert (
            _imported_metadata_value(
                data,
                endpoint_kind='source',
                endpoint_key=SOURCE_KEYS[100],
                column='vsub_m_s',
            )
            == '5200.000'
        )
        assert (
            _imported_metadata_value(
                data,
                endpoint_kind='source',
                endpoint_key=SOURCE_KEYS[100],
                column='sh1_weathering_thickness_m',
            )
            == '13.506'
        )
        assert (
            _imported_metadata_value(
                data,
                endpoint_kind='source',
                endpoint_key=SOURCE_KEYS[100],
                column='sh2_weathering_thickness_m',
            )
            == '25.246'
        )
        assert (
            _imported_metadata_value(
                data,
                endpoint_kind='source',
                endpoint_key=SOURCE_KEYS[100],
                column='sh3_weathering_thickness_m',
            )
            == '38.785'
        )
        assert (
            _imported_metadata_value(
                data,
                endpoint_kind='receiver',
                endpoint_key=RECEIVER_KEYS[201],
                column='t3_ms',
            )
            == '28.000000'
        )
        assert (
            _imported_metadata_value(
                data,
                endpoint_kind='receiver',
                endpoint_key=RECEIVER_KEYS[201],
                column='vsub_m_s',
            )
            == '5100.000'
        )
        assert (
            _imported_metadata_value(
                data,
                endpoint_kind='receiver',
                endpoint_key=RECEIVER_KEYS[201],
                column='sh3_weathering_thickness_m',
            )
            == '38.555'
        )
    history = _history(job_dir)
    assert history['import_schema_name'] == 'canonical_static_table'
    assert history['created_from_refraction_job_id'] == SOURCE_REFRACTION_JOB_ID


def test_static_table_apply_exported_table_missing_endpoint_policy_fail(
    tmp_path: Path,
) -> None:
    state, _store = _write_target_store(tmp_path)
    _write_source_refraction_job(state, tmp_path, source_ids=(100,))
    _run_export(state, tmp_path)

    job_dir = _run_apply(state, tmp_path)

    job = _job(state, APPLY_JOB_ID)
    assert job['status'] == 'error'
    assert 'missing_source_static' in str(job['message'])
    assert not (job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME).exists()


def test_static_table_apply_exported_table_missing_receiver_policy_fail(
    tmp_path: Path,
) -> None:
    state, _store = _write_target_store(tmp_path)
    _write_source_refraction_job(state, tmp_path, receiver_ids=(200,))
    _run_export(state, tmp_path)

    job_dir = _run_apply(state, tmp_path)

    job = _job(state, APPLY_JOB_ID)
    assert job['status'] == 'error'
    assert 'missing_receiver_static' in str(job['message'])
    assert not (job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME).exists()


def test_static_table_apply_exported_table_missing_endpoint_policy_zero(
    tmp_path: Path,
) -> None:
    state, _store = _write_target_store(tmp_path)
    _write_source_refraction_job(state, tmp_path, source_ids=(100,))
    _run_export(state, tmp_path)

    job_dir = _run_apply(
        state,
        tmp_path,
        missing_static_policy='zero',
        allow_missing_source_static=True,
    )

    assert _job(state, APPLY_JOB_ID)['status'] == 'done'
    with np.load(job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        np.testing.assert_allclose(
            data['trace_shift_s_sorted'],
            [0.008, 0.0, 0.004, -0.004],
        )
        assert data['trace_static_status_sorted'][1] == 'missing_source_static_zeroed'
    qc = json.loads((job_dir / STATIC_TABLE_APPLY_QC_JSON_NAME).read_text())
    assert qc['n_missing_source_endpoints'] == 1
    assert qc['trace_static_status_counts']['missing_source_static_zeroed'] == 2


def test_static_table_apply_exported_table_missing_receiver_policy_zero(
    tmp_path: Path,
) -> None:
    state, _store = _write_target_store(tmp_path)
    _write_source_refraction_job(state, tmp_path, receiver_ids=(200,))
    _run_export(state, tmp_path)

    job_dir = _run_apply(
        state,
        tmp_path,
        missing_static_policy='zero',
        allow_missing_receiver_static=True,
    )

    assert _job(state, APPLY_JOB_ID)['status'] == 'done'
    with np.load(job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        np.testing.assert_allclose(
            data['trace_shift_s_sorted'],
            [0.008, 0.004, 0.008, 0.004],
        )
        assert data['trace_static_status_sorted'][2] == 'missing_receiver_static_zeroed'
    qc = json.loads((job_dir / STATIC_TABLE_APPLY_QC_JSON_NAME).read_text())
    assert qc['n_missing_receiver_endpoints'] == 1
    assert qc['trace_static_status_counts']['missing_receiver_static_zeroed'] == 2


def test_static_table_apply_exported_table_double_application_guard(
    tmp_path: Path,
) -> None:
    state, _store = _write_target_store(tmp_path)
    _write_source_refraction_job(state, tmp_path)
    _run_export(state, tmp_path)
    first_job_dir = _run_apply(
        state,
        tmp_path,
        register_corrected_file=True,
    )
    first_job = _job(state, APPLY_JOB_ID)
    assert first_job['status'] == 'done', first_job.get('message')

    corrected_file_id = str(first_job['corrected_file_id'])
    second_job_dir = _run_apply(
        state,
        tmp_path,
        job_id=SECOND_APPLY_JOB_ID,
        file_id=corrected_file_id,
    )

    second_job = _job(state, SECOND_APPLY_JOB_ID)
    assert second_job['status'] == 'error'
    assert 'allow_reapply_same_static_table=False' in str(second_job['message'])
    assert not (second_job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME).exists()

    third_job_dir = _run_apply(
        state,
        tmp_path,
        job_id=THIRD_APPLY_JOB_ID,
        file_id=corrected_file_id,
        allow_reapply_same_static_table=True,
    )

    third_job = _job(state, THIRD_APPLY_JOB_ID)
    assert third_job['status'] == 'done', third_job.get('message')
    guard = _history(third_job_dir)['static_table_reapply_guard']
    assert guard['status'] == 'duplicate_allowed_by_override'
    assert guard['override_used'] is True
    assert 'same_table_digest' in guard['duplicate_reasons']
    assert (first_job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME).exists()
