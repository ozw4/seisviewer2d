from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from app.services.refraction_static_artifacts import (
    REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
    write_refraction_static_artifacts,
)
from app.tests._refraction_static_synthetic import (
    run_synthetic_cell_refraction_statics,
    synthetic_cell_refracted_arrival_input_model,
    synthetic_cell_refraction_apply_request,
)
from app.tests.test_refraction_static_cell_low_fold import (
    _low_fold_case,
    _run_refraction_statics,
)


def test_cell_solver_history_artifact_exists_for_solve_cell(
    tmp_path: Path,
) -> None:
    result, req = _clean_cell_result()

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    assert paths.refraction_cell_solver_history_csv is not None
    assert paths.refraction_cell_solver_history_csv.name == (
        REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME
    )
    rows = _read_csv(paths.refraction_cell_solver_history_csv)
    assert [row['stage'] for row in rows] == ['initial', 'final']
    assert rows[0]['converged'] == 'false'
    assert rows[-1]['converged'] == 'true'
    assert {
        'median_velocity_m_s',
        'min_velocity_m_s',
        'max_velocity_m_s',
        'max_abs_velocity_update_m_s',
        'median_v2_m_s',
        'min_v2_m_s',
        'max_v2_m_s',
        'max_abs_v2_update_m_s',
    }.issubset(rows[0])

    manifest = json.loads(paths.manifest_json.read_text(encoding='utf-8'))
    artifact_names = {item['name'] for item in manifest['artifacts']}
    assert REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME in artifact_names


def test_cell_solver_history_final_row_matches_qc_summary(
    tmp_path: Path,
) -> None:
    result, req = _clean_cell_result()

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    final = _read_csv(paths.refraction_cell_solver_history_csv)[-1]
    assert int(final['n_candidate_observations']) == (
        qc['observations']['n_valid_observations']
    )
    assert int(final['n_used_observations']) == (
        qc['observations']['n_used_observations']
    )
    assert int(final['n_rejected_observations']) == (
        qc['observations']['n_rejected_by_robust']
    )
    assert float(final['residual_rms_ms']) == pytest.approx(
        qc['first_break_fit']['residual_rms_ms'],
        abs=1.0e-12,
    )
    assert float(final['residual_mad_ms']) == pytest.approx(
        qc['first_break_fit']['residual_mad_ms'],
        abs=1.0e-12,
    )
    assert float(final['max_abs_residual_ms']) == pytest.approx(
        qc['first_break_fit']['residual_max_abs_ms'],
        abs=1.0e-12,
    )
    assert float(final['residual_rms_ms']) == pytest.approx(0.0, abs=1.0e-6)
    assert final['median_velocity_m_s'] == final['median_v2_m_s']
    assert final['min_velocity_m_s'] == final['min_v2_m_s']
    assert final['max_velocity_m_s'] == final['max_v2_m_s']
    assert final['max_abs_velocity_update_m_s'] == final['max_abs_v2_update_m_s']


def test_cell_solver_history_records_robust_rejections(
    tmp_path: Path,
) -> None:
    input_model = _outlier_input_model()
    req = synthetic_cell_refraction_apply_request(robust_enabled=True)
    result = run_synthetic_cell_refraction_statics(
        req=req,
        input_model=input_model,
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    final = _read_csv(paths.refraction_cell_solver_history_csv)[-1]
    rejected_count = int(np.count_nonzero(result.rejected_by_robust_mask))
    assert rejected_count > 0
    assert int(final['n_rejected_observations']) == rejected_count
    assert float(final['robust_threshold']) == pytest.approx(
        req.solver.robust.threshold
    )
    assert final['convergence_reason'] == 'robust_reweight_converged'


def test_cell_solver_history_records_smoothing_settings(
    tmp_path: Path,
) -> None:
    req = synthetic_cell_refraction_apply_request(velocity_smoothing_weight=2.0)
    result = run_synthetic_cell_refraction_statics(req=req)

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    final = _read_csv(paths.refraction_cell_solver_history_csv)[-1]
    assert float(final['smoothing_weight']) == pytest.approx(2.0)
    assert float(final['damping_weight']) == pytest.approx(0.0)
    assert float(final['robust_threshold']) == pytest.approx(
        req.solver.robust.threshold
    )
    assert final['convergence_reason'] == 'smoothed_least_squares_converged'


def test_cell_solver_history_records_low_fold_and_empty_cell_counts(
    tmp_path: Path,
) -> None:
    dataset, req, input_model = _low_fold_case()
    result = _run_refraction_statics(req=req, input_model=input_model)

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    final = _read_csv(paths.refraction_cell_solver_history_csv)[-1]
    assert int(final['n_active_cells']) == int(result.active_cell_id.shape[0])
    assert int(final['n_low_fold_cells']) == 1
    assert int(final['n_empty_cells']) == int(
        np.count_nonzero(dataset.cell_observation_count == 0)
    )
    assert int(final['n_rejected_observations']) == 0
    assert final['convergence_reason'] == 'least_squares_converged'


def _clean_cell_result():
    req = synthetic_cell_refraction_apply_request()
    result = run_synthetic_cell_refraction_statics(req=req)
    return result, req


def _outlier_input_model():
    input_model = synthetic_cell_refracted_arrival_input_model()
    outlier_indices = np.asarray([5, 6, 7, 13, 14, 15], dtype=np.int64)
    outlier_shift_s = np.asarray([0.050, -0.060, 0.100], dtype=np.float64)
    pick_time_s = input_model.pick_time_s_sorted.copy()
    pick_time_s[outlier_indices] += np.resize(
        outlier_shift_s,
        outlier_indices.shape[0],
    )
    return replace(
        input_model,
        pick_time_s_sorted=np.ascontiguousarray(pick_time_s, dtype=np.float64),
    )


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    assert path is not None
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))
