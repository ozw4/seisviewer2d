from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from app.services.refraction_static_artifacts import (
    REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
    write_refraction_static_artifacts,
)
from app.tests._refraction_static_synthetic import (
    SYNTHETIC_CELL_SIZE_X_M,
    SYNTHETIC_CELL_V2_M_S,
    run_synthetic_cell_refraction_statics,
    synthetic_cell_midpoint_cell_id_sorted,
    synthetic_cell_refracted_arrival_input_model,
    synthetic_cell_refraction_apply_request,
)
from app.tests.test_refraction_static_cell_low_fold import (
    EMPTY_CELL_ID,
    LOW_FOLD_CELL_ID,
    _low_fold_case,
    _run_refraction_statics,
)


EXPECTED_CELL_COLUMNS = {
    'cell_ix',
    'cell_iy',
    'cell_id',
    'coordinate_mode',
    'cell_center_x_m',
    'cell_center_y_m',
    'cell_center_inline_m',
    'cell_center_crossline_m',
    'n_observations',
    'n_used_observations',
    'n_rejected_observations',
    'n_sources',
    'n_receivers',
    'cell_velocity_layer_kind',
    'cell_velocity_component',
    'velocity_m_s',
    'v2_m_s',
    'slowness_s_per_m',
    'initial_velocity_m_s',
    'initial_v2_m_s',
    'velocity_update_from_initial_m_s',
    'v2_update_from_initial_m_s',
    'residual_rms_ms',
    'residual_mad_ms',
    'velocity_status',
    'status_reason',
    'smoothing_enabled',
    'smoothing_weight',
}

EXPECTED_RESIDUAL_COLUMNS = {
    'observation_index',
    'cell_id',
    'cell_ix',
    'cell_iy',
    'observed_pick_time_s',
    'modeled_pick_time_s',
    'residual_s',
    'rejection_reason',
    'used_in_solve',
}


def test_cell_velocity_artifact_contains_expected_columns(tmp_path: Path) -> None:
    result, req, _ = _clean_result()

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    assert paths.refraction_refractor_velocity_cells_csv is not None
    rows = _read_csv(paths.refraction_refractor_velocity_cells_csv)
    assert EXPECTED_CELL_COLUMNS.issubset(rows[0].keys())

    assert paths.first_break_residuals_csv is not None
    residual_rows = _read_csv(paths.first_break_residuals_csv)
    assert EXPECTED_RESIDUAL_COLUMNS.issubset(residual_rows[0].keys())


def test_cell_velocity_artifact_counts_observations_correctly(
    tmp_path: Path,
) -> None:
    result, req, input_model = _clean_result()
    expected_cell_id = synthetic_cell_midpoint_cell_id_sorted(input_model)

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    assert paths.refraction_refractor_velocity_cells_csv is not None
    rows = {
        int(row['cell_id']): row
        for row in _read_csv(paths.refraction_refractor_velocity_cells_csv)
    }
    for cell_id in range(SYNTHETIC_CELL_V2_M_S.shape[0]):
        in_cell = expected_cell_id == cell_id
        assert int(rows[cell_id]['n_observations']) == int(np.count_nonzero(in_cell))
        assert int(rows[cell_id]['n_used_observations']) == int(
            np.count_nonzero(in_cell)
        )
        assert int(rows[cell_id]['n_rejected_observations']) == 0
        assert int(rows[cell_id]['n_sources']) == int(
            np.unique(input_model.source_node_id_sorted[in_cell]).shape[0]
        )
        assert int(rows[cell_id]['n_receivers']) == int(
            np.unique(input_model.receiver_node_id_sorted[in_cell]).shape[0]
        )


def test_cell_velocity_artifact_clean_synthetic_values_are_known_truth(
    tmp_path: Path,
) -> None:
    result, req, _ = _clean_result()

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    assert paths.refraction_refractor_velocity_cells_csv is not None
    rows = _read_csv(paths.refraction_refractor_velocity_cells_csv)
    for row in rows:
        cell_id = int(row['cell_id'])
        true_v2 = float(SYNTHETIC_CELL_V2_M_S[cell_id])
        center_x_m = (cell_id + 0.5) * SYNTHETIC_CELL_SIZE_X_M

        assert int(row['cell_ix']) == cell_id
        assert int(row['cell_iy']) == 0
        assert row['coordinate_mode'] == 'grid_3d'
        assert float(row['cell_center_x_m']) == pytest.approx(center_x_m)
        assert float(row['cell_center_y_m']) == pytest.approx(0.0)
        assert row['cell_center_inline_m'] == ''
        assert row['cell_center_crossline_m'] == ''
        assert row['cell_velocity_layer_kind'] == 'v2_t1'
        assert row['cell_velocity_component'] == 'v2'
        assert float(row['velocity_m_s']) == pytest.approx(true_v2, abs=1.0e-3)
        assert float(row['v2_m_s']) == pytest.approx(true_v2, abs=1.0e-3)
        assert float(row['slowness_s_per_m']) == pytest.approx(
            1.0 / true_v2,
            abs=1.0e-12,
        )
        assert float(row['initial_velocity_m_s']) == pytest.approx(2600.0)
        assert float(row['initial_v2_m_s']) == pytest.approx(2600.0)
        assert float(row['velocity_update_from_initial_m_s']) == pytest.approx(
            true_v2 - 2600.0,
            abs=1.0e-3,
        )
        assert float(row['v2_update_from_initial_m_s']) == pytest.approx(
            true_v2 - 2600.0,
            abs=1.0e-3,
        )
        assert float(row['residual_rms_ms']) == pytest.approx(0.0, abs=1.0e-6)
        assert float(row['residual_mad_ms']) == pytest.approx(0.0, abs=1.0e-6)
        assert row['velocity_status'] == 'solved'
        assert row['status_reason'] == 'solved'


def test_cell_velocity_artifact_marks_low_fold_and_empty_cells(
    tmp_path: Path,
) -> None:
    _, req, input_model = _low_fold_case()
    result = _run_refraction_statics(req=req, input_model=input_model)

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    assert paths.refraction_refractor_velocity_cells_csv is not None
    rows = {
        int(row['cell_id']): row
        for row in _read_csv(paths.refraction_refractor_velocity_cells_csv)
    }
    assert rows[LOW_FOLD_CELL_ID]['velocity_status'] == 'low_fold'
    assert rows[LOW_FOLD_CELL_ID]['status_reason'] == (
        'below_min_observations_per_cell'
    )
    assert rows[LOW_FOLD_CELL_ID]['n_observations'] == '19'
    assert rows[LOW_FOLD_CELL_ID]['n_used_observations'] == '0'
    assert rows[LOW_FOLD_CELL_ID]['n_rejected_observations'] == '19'

    assert rows[EMPTY_CELL_ID]['velocity_status'] == 'inactive'
    assert rows[EMPTY_CELL_ID]['status_reason'] == 'no_observations'
    assert rows[EMPTY_CELL_ID]['n_observations'] == '0'
    assert rows[EMPTY_CELL_ID]['velocity_m_s'] == ''
    assert rows[EMPTY_CELL_ID]['v2_m_s'] == ''


def test_observation_residual_artifact_includes_cell_ids_and_rejection_reason(
    tmp_path: Path,
) -> None:
    result, req, input_model = _clean_result()
    expected_cell_id = synthetic_cell_midpoint_cell_id_sorted(input_model)

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    assert paths.first_break_residuals_csv is not None
    rows = _read_csv(paths.first_break_residuals_csv)
    for row in rows:
        observation_index = int(row['observation_index'])
        trace_index = int(row['sorted_trace_index'])
        cell_id = int(row['cell_id'])
        assert observation_index == int(row['row_index'])
        assert cell_id == int(expected_cell_id[trace_index])
        assert int(row['cell_ix']) == cell_id
        assert int(row['cell_iy']) == 0
        assert float(row['observed_pick_time_s']) == pytest.approx(
            float(input_model.pick_time_s_sorted[trace_index])
        )
        assert float(row['modeled_pick_time_s']) == pytest.approx(
            float(input_model.pick_time_s_sorted[trace_index]),
            abs=1.0e-8,
        )
        assert float(row['residual_s']) == pytest.approx(0.0, abs=1.0e-8)
        assert row['used_in_solve'] == 'true'
        assert row['rejection_reason'] == 'ok'


def test_first_break_time_export_contains_cell_indices_for_cell_solve(
    tmp_path: Path,
) -> None:
    result, req, input_model = _clean_result()
    expected_cell_id = synthetic_cell_midpoint_cell_id_sorted(input_model)

    write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    rows = _read_csv(tmp_path / REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME)
    for row in rows:
        trace_index = int(row['trace_index_sorted'])
        cell_id = int(expected_cell_id[trace_index])
        assert int(row['cell_ix']) == cell_id
        assert int(row['cell_iy']) == 0
        assert row['layer_kind'] == 'v2_t1'
        assert row['used_for_layer'] == 'true'


def test_cell_velocity_artifact_records_smoothing_metadata(
    tmp_path: Path,
) -> None:
    input_model = synthetic_cell_refracted_arrival_input_model()
    req = synthetic_cell_refraction_apply_request(velocity_smoothing_weight=2.0)
    result = run_synthetic_cell_refraction_statics(
        req=req,
        input_model=input_model,
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    assert paths.refraction_refractor_velocity_cells_csv is not None
    for row in _read_csv(paths.refraction_refractor_velocity_cells_csv):
        assert row['coordinate_mode'] == 'grid_3d'
        assert row['smoothing_enabled'] == 'true'
        assert float(row['smoothing_weight']) == pytest.approx(2.0)


def _clean_result():
    input_model = synthetic_cell_refracted_arrival_input_model()
    req = synthetic_cell_refraction_apply_request()
    result = run_synthetic_cell_refraction_statics(
        req=req,
        input_model=input_model,
    )
    return result, req, input_model


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))
