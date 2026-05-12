from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import app.services.refraction_static_service as refraction_service_module
from app.core.state import AppState, create_app_state
from app.services.refraction_static_apply_trace_store import (
    CORRECTED_FILE_JSON_NAME,
    REFRACTION_STATIC_APPLY_QC_JSON_NAME,
)
from app.services.refraction_static_artifacts import (
    RECEIVER_STATIC_TABLE_CSV_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)
from app.services.refraction_static_source_depth import (
    REFRACTION_SOURCE_DEPTH_QC_JSON_NAME,
)
from app.services.refraction_static_service import run_refraction_static_apply_job
from app.services.refraction_static_uphole import REFRACTION_UPHOLE_QC_JSON_NAME
from app.tests._refraction_static_field_e2e_helpers import (
    FIELD_FILE_ID,
    FIELD_ORIGINAL_NAME,
    clean_field_e2e_fixture,
    create_field_refraction_job,
    field_apply_request,
    FIELD_SAMPLE_INTERVAL_S,
    install_field_job_stubs,
    messy_field_e2e_fixture,
    write_field_manual_picks,
    write_field_trace_store,
)


def test_field_e2e_artifact_only_preserves_refraction_trace_shift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = clean_field_e2e_fixture()
    req = field_apply_request(
        fixture.dataset,
        apply_to_trace_shift=False,
        register_corrected_file=True,
    )
    state, job_dir = _run_field_job(
        tmp_path,
        monkeypatch,
        fixture=fixture,
        req=req,
        job_id='field-artifact-only',
        with_source_store=True,
    )

    with state.lock:
        job = dict(state.jobs['field-artifact-only'])
    assert job['status'] == 'done', job.get('message')

    with np.load(job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        np.testing.assert_allclose(
            data['refraction_trace_shift_s_sorted'],
            fixture.dataset.expected_refraction_trace_shift_s,
        )
        np.testing.assert_allclose(
            data['trace_field_shift_s_sorted'],
            fixture.dataset.expected_trace_field_shift_s,
        )
        np.testing.assert_allclose(
            data['final_trace_shift_s_sorted'],
            fixture.dataset.expected_final_trace_shift_s,
        )
        np.testing.assert_allclose(
            data['applied_field_shift_s_sorted'],
            np.zeros_like(fixture.dataset.expected_refraction_trace_shift_s),
        )

    corrected = json.loads((job_dir / CORRECTED_FILE_JSON_NAME).read_text())
    assert corrected['shift_field'] == 'refraction_trace_shift_s_sorted'
    assert corrected['static_components_applied'] == ['refraction']
    assert corrected['field_corrections_applied_to_trace_shift'] is False

    apply_qc = json.loads((job_dir / REFRACTION_STATIC_APPLY_QC_JSON_NAME).read_text())
    assert apply_qc['max_abs_applied_shift_ms'] == pytest.approx(
        float(np.max(np.abs(fixture.dataset.expected_refraction_trace_shift_s))) * 1000.0
    )


def test_field_e2e_apply_mode_uses_final_trace_shift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = clean_field_e2e_fixture()
    req = field_apply_request(
        fixture.dataset,
        apply_to_trace_shift=True,
        register_corrected_file=True,
    )
    state, job_dir = _run_field_job(
        tmp_path,
        monkeypatch,
        fixture=fixture,
        req=req,
        job_id='field-apply-final',
        with_source_store=True,
    )

    with state.lock:
        job = dict(state.jobs['field-apply-final'])
    assert job['status'] == 'done', job.get('message')

    with np.load(job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        np.testing.assert_allclose(
            data['final_trace_shift_s_sorted'],
            fixture.dataset.expected_final_trace_shift_s,
        )
        np.testing.assert_allclose(
            data['applied_field_shift_s_sorted'],
            fixture.dataset.expected_trace_field_shift_s,
        )

    corrected = json.loads((job_dir / CORRECTED_FILE_JSON_NAME).read_text())
    assert corrected['shift_field'] == 'final_trace_shift_s_sorted'
    assert corrected['static_components_applied'] == [
        'refraction',
        'source_depth',
        'uphole',
        'manual_static',
    ]
    assert corrected['field_corrections_applied_to_trace_shift'] is True

    apply_qc = json.loads((job_dir / REFRACTION_STATIC_APPLY_QC_JSON_NAME).read_text())
    assert apply_qc['max_abs_applied_shift_ms'] == pytest.approx(
        float(np.max(np.abs(fixture.dataset.expected_final_trace_shift_s))) * 1000.0
    )


def test_field_e2e_source_depth_uphole_manual_components_match_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = clean_field_e2e_fixture()
    req = field_apply_request(fixture.dataset, apply_to_trace_shift=True)
    _state, job_dir = _run_field_job(
        tmp_path,
        monkeypatch,
        fixture=fixture,
        req=req,
        job_id='field-components',
    )

    with np.load(job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        np.testing.assert_allclose(data['source_depth_m'], fixture.dataset.source_depth_m)
        np.testing.assert_allclose(
            data['source_depth_shift_s'],
            fixture.dataset.expected_source_depth_shift_s,
        )
        np.testing.assert_allclose(
            data['source_uphole_time_s'],
            fixture.dataset.uphole_time_s,
        )
        np.testing.assert_allclose(
            data['source_uphole_shift_s'],
            fixture.dataset.expected_uphole_shift_s,
        )
        np.testing.assert_allclose(
            data['source_manual_static_shift_s'],
            fixture.dataset.expected_source_manual_static_shift_s,
        )
        np.testing.assert_allclose(
            data['receiver_manual_static_shift_s'],
            fixture.dataset.expected_receiver_manual_static_shift_s,
        )
        np.testing.assert_allclose(
            data['trace_field_shift_s_sorted'],
            fixture.dataset.expected_trace_field_shift_s,
        )
        np.testing.assert_allclose(
            data['final_trace_shift_s_sorted'],
            fixture.dataset.expected_final_trace_shift_s,
        )

    source_depth_qc = json.loads(
        (job_dir / REFRACTION_SOURCE_DEPTH_QC_JSON_NAME).read_text()
    )
    assert source_depth_qc['source_depth_byte'] == 115
    uphole_qc = json.loads((job_dir / REFRACTION_UPHOLE_QC_JSON_NAME).read_text())
    assert uphole_qc['uphole_time_byte'] == 95
    assert uphole_qc['uphole_time_unit'] == 'ms'


def test_field_e2e_source_receiver_tables_match_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = clean_field_e2e_fixture()
    req = field_apply_request(fixture.dataset, apply_to_trace_shift=True)
    _state, job_dir = _run_field_job(
        tmp_path,
        monkeypatch,
        fixture=fixture,
        req=req,
        job_id='field-tables',
    )

    source_rows = _read_csv(job_dir / SOURCE_STATIC_TABLE_CSV_NAME)
    receiver_rows = _read_csv(job_dir / RECEIVER_STATIC_TABLE_CSV_NAME)
    for index, row in enumerate(source_rows):
        assert float(row['source_depth_shift_ms']) == pytest.approx(
            fixture.dataset.expected_source_depth_shift_s[index] * 1000.0
        )
        assert float(row['uphole_shift_ms']) == pytest.approx(
            fixture.dataset.expected_uphole_shift_s[index] * 1000.0
        )
        assert float(row['manual_static_shift_ms']) == pytest.approx(
            fixture.dataset.expected_source_manual_static_shift_s[index] * 1000.0
        )
        assert float(row['source_field_shift_ms']) == pytest.approx(
            fixture.dataset.expected_source_field_shift_s[index] * 1000.0
        )
        assert row['source_field_static_status'] == 'ok'
        assert float(row['source_total_with_field_shift_ms']) == pytest.approx(
            (
                fixture.dataset.source_endpoint_table.refraction_shift_s[index]
                + fixture.dataset.expected_source_field_shift_s[index]
            )
            * 1000.0
        )

    for index, row in enumerate(receiver_rows):
        assert float(row['manual_static_shift_ms']) == pytest.approx(
            fixture.dataset.expected_receiver_manual_static_shift_s[index] * 1000.0
        )
        assert float(row['receiver_field_shift_ms']) == pytest.approx(
            fixture.dataset.expected_receiver_field_shift_s[index] * 1000.0
        )
        assert row['receiver_field_static_status'] == 'ok'
        assert float(row['receiver_total_with_field_shift_ms']) == pytest.approx(
            (
                fixture.dataset.receiver_endpoint_table.refraction_shift_s[index]
                + fixture.dataset.expected_receiver_field_shift_s[index]
            )
            * 1000.0
        )

    with np.load(job_dir / SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME, allow_pickle=False) as data:
        np.testing.assert_allclose(
            data['source_field_shift_s'],
            fixture.dataset.expected_source_field_shift_s,
        )
        np.testing.assert_allclose(
            data['receiver_field_shift_s'],
            fixture.dataset.expected_receiver_field_shift_s,
        )
        np.testing.assert_allclose(
            data['source_total_with_field_shift_s'],
            fixture.dataset.source_endpoint_table.refraction_shift_s
            + fixture.dataset.expected_source_field_shift_s,
        )
        np.testing.assert_allclose(
            data['receiver_total_with_field_shift_s'],
            fixture.dataset.receiver_endpoint_table.refraction_shift_s
            + fixture.dataset.expected_receiver_field_shift_s,
        )


def test_field_e2e_manual_static_delay_positive_ms_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = clean_field_e2e_fixture(manual_static_sign_convention='delay_positive_ms')
    req = field_apply_request(fixture.dataset, apply_to_trace_shift=True)
    _state, job_dir = _run_field_job(
        tmp_path,
        monkeypatch,
        fixture=fixture,
        req=req,
        job_id='field-manual-delay-positive',
    )

    with np.load(job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        np.testing.assert_allclose(
            data['source_manual_static_shift_s'],
            -fixture.dataset.source_manual_static_input_s,
        )
        np.testing.assert_allclose(
            data['receiver_manual_static_shift_s'],
            -fixture.dataset.receiver_manual_static_input_s,
        )

    source_rows = _read_csv(job_dir / SOURCE_STATIC_TABLE_CSV_NAME)
    assert source_rows[0]['sign_convention'] == 'corrected(t) = raw(t - shift_s)'


def test_field_e2e_invalid_policy_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = messy_field_e2e_fixture()
    req = field_apply_request(
        fixture.dataset,
        apply_to_trace_shift=True,
        invalid_component_policy='fail',
    )
    state, job_dir = _run_field_job(
        tmp_path,
        monkeypatch,
        fixture=fixture,
        req=req,
        job_id='field-invalid-fail',
    )

    with state.lock:
        job = dict(state.jobs['field-invalid-fail'])
    assert job['status'] == 'error'
    assert 'invalid_trace_field' in str(job['message'])
    assert not (job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME).exists()


def test_field_e2e_invalid_policy_skip_invalid_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = messy_field_e2e_fixture()
    req = field_apply_request(
        fixture.dataset,
        apply_to_trace_shift=True,
        invalid_component_policy='skip_invalid_traces',
    )
    state, job_dir = _run_field_job(
        tmp_path,
        monkeypatch,
        fixture=fixture,
        req=req,
        job_id='field-invalid-skip',
    )

    with state.lock:
        job = dict(state.jobs['field-invalid-skip'])
    assert job['status'] == 'done', job.get('message')

    with np.load(job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        statuses = data['trace_field_static_status_sorted']
        valid = statuses == 'ok'
        invalid = ~valid
        assert bool(np.any(invalid))
        np.testing.assert_allclose(
            data['final_trace_shift_s_sorted'][invalid],
            fixture.dataset.expected_refraction_trace_shift_s[invalid],
        )
        np.testing.assert_allclose(data['applied_field_shift_s_sorted'][invalid], 0.0)
        np.testing.assert_allclose(
            data['final_trace_shift_s_sorted'][valid],
            data['refraction_trace_shift_s_sorted'][valid]
            + data['applied_field_shift_s_sorted'][valid],
        )


def _run_field_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fixture: Any,
    req: Any,
    job_id: str,
    with_source_store: bool = False,
) -> tuple[AppState, Path]:
    state = create_app_state()
    job_dir = tmp_path / 'jobs' / job_id
    create_field_refraction_job(state, job_id=job_id, req=req, job_dir=job_dir)
    install_field_job_stubs(monkeypatch, refraction_service_module, fixture)
    monkeypatch.setenv('SV_APP_DATA_DIR', str(tmp_path / 'app_data'))
    store = tmp_path / 'trace_stores' / f'{job_id}.sgy'
    write_field_trace_store(store, fixture.dataset)
    write_field_manual_picks(fixture.dataset)
    state.file_registry.update(
        FIELD_FILE_ID,
        path=f'/data/{FIELD_ORIGINAL_NAME}',
        store_path=str(store),
        dt=FIELD_SAMPLE_INTERVAL_S,
    )

    run_refraction_static_apply_job(job_id, req, state)
    return state, job_dir


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _rows_by_key(rows: list[dict[str, str]], key_name: str) -> dict[str, dict[str, str]]:
    return {row[key_name]: row for row in rows}
