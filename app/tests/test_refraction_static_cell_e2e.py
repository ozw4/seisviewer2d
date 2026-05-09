from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_artifacts import (
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    write_refraction_static_artifacts,
)
from app.tests._refraction_static_synthetic import (
    SYNTHETIC_CELL_NODE_SH1_M,
    SYNTHETIC_CELL_NODE_T1_S,
    SYNTHETIC_CELL_NODE_WCOR_S,
    SYNTHETIC_CELL_V2_M_S,
    SYNTHETIC_CELL_V2_TOLERANCE_M_S,
    SYNTHETIC_SH1_TOLERANCE_M,
    SYNTHETIC_T1_TOLERANCE_MS,
    SYNTHETIC_WCOR_TOLERANCE_MS,
    run_synthetic_cell_refraction_statics,
    synthetic_cell_midpoint_cell_id_sorted,
    synthetic_cell_refracted_arrival_input_model,
    synthetic_cell_refraction_apply_request,
)


def test_solve_cell_e2e_recovers_noiseless_v2_t1_sh1_wcor() -> None:
    result = run_synthetic_cell_refraction_statics()

    assert result.bedrock_velocity_mode == 'solve_cell'
    assert result.active_cell_id is not None
    assert result.cell_bedrock_velocity_m_s is not None
    np.testing.assert_array_equal(result.active_cell_id, [0, 1, 2])
    np.testing.assert_allclose(
        result.cell_bedrock_velocity_m_s,
        SYNTHETIC_CELL_V2_M_S,
        atol=SYNTHETIC_CELL_V2_TOLERANCE_M_S,
    )
    np.testing.assert_allclose(
        result.node_half_intercept_time_s * 1000.0,
        SYNTHETIC_CELL_NODE_T1_S * 1000.0,
        atol=SYNTHETIC_T1_TOLERANCE_MS,
    )
    np.testing.assert_allclose(
        result.node_weathering_thickness_m,
        SYNTHETIC_CELL_NODE_SH1_M,
        atol=SYNTHETIC_SH1_TOLERANCE_M,
    )
    np.testing.assert_allclose(
        result.node_weathering_replacement_shift_s * 1000.0,
        SYNTHETIC_CELL_NODE_WCOR_S * 1000.0,
        atol=SYNTHETIC_WCOR_TOLERANCE_MS,
    )
    assert np.all(result.trace_static_valid_mask_sorted)


def test_solve_cell_e2e_even_active_cell_count_does_not_fail() -> None:
    result = run_synthetic_cell_refraction_statics(
        input_model=synthetic_cell_refracted_arrival_input_model(
            allowed_midpoint_cell_ids=(0, 1),
        )
    )

    assert result.bedrock_velocity_mode == 'solve_cell'
    assert result.bedrock_velocity_m_s == pytest.approx(
        1.0 / result.bedrock_slowness_s_per_m
    )
    assert result.active_cell_id is not None
    assert result.cell_bedrock_velocity_m_s is not None
    assert result.cell_bedrock_slowness_s_per_m is not None
    np.testing.assert_array_equal(result.active_cell_id, [0, 1])
    np.testing.assert_allclose(
        result.cell_bedrock_velocity_m_s,
        SYNTHETIC_CELL_V2_M_S[:2],
        atol=SYNTHETIC_CELL_V2_TOLERANCE_M_S,
    )
    np.testing.assert_allclose(
        result.cell_bedrock_slowness_s_per_m,
        1.0 / SYNTHETIC_CELL_V2_M_S[:2],
        atol=1.0e-10,
    )


def test_solve_cell_e2e_robust_rejects_outliers() -> None:
    input_model = synthetic_cell_refracted_arrival_input_model()
    pick_time = input_model.pick_time_s_sorted.copy()
    outlier_trace_indices = np.asarray([5, 48], dtype=np.int64)
    pick_time[outlier_trace_indices] += np.asarray([0.120, -0.090], dtype=np.float64)
    input_model = replace(input_model, pick_time_s_sorted=pick_time)

    result = run_synthetic_cell_refraction_statics(
        req=synthetic_cell_refraction_apply_request(robust_enabled=True),
        input_model=input_model,
    )

    rejected_trace_indices = set(
        result.row_trace_index_sorted[result.rejected_by_robust_mask].tolist()
    )
    assert set(outlier_trace_indices.tolist()).issubset(rejected_trace_indices)
    assert int(np.count_nonzero(result.rejected_by_robust_mask)) >= 2
    assert result.cell_bedrock_velocity_m_s is not None
    np.testing.assert_allclose(
        result.cell_bedrock_velocity_m_s,
        SYNTHETIC_CELL_V2_M_S,
        atol=SYNTHETIC_CELL_V2_TOLERANCE_M_S,
    )


def test_solve_cell_e2e_smoothing_reduces_velocity_contrast() -> None:
    input_model = synthetic_cell_refracted_arrival_input_model()
    midpoint_cell = synthetic_cell_midpoint_cell_id_sorted(input_model)
    perturbation_s = np.where(
        midpoint_cell == 0,
        0.004,
        np.where(midpoint_cell == 2, -0.004, 0.0),
    )
    noisy_input = replace(
        input_model,
        pick_time_s_sorted=input_model.pick_time_s_sorted + perturbation_s,
    )

    unsmoothed = run_synthetic_cell_refraction_statics(
        req=synthetic_cell_refraction_apply_request(),
        input_model=noisy_input,
    )
    smoothed = run_synthetic_cell_refraction_statics(
        req=synthetic_cell_refraction_apply_request(
            velocity_smoothing_weight=5.0,
        ),
        input_model=noisy_input,
    )

    assert unsmoothed.cell_bedrock_velocity_m_s is not None
    assert smoothed.cell_bedrock_velocity_m_s is not None
    assert float(np.ptp(smoothed.cell_bedrock_velocity_m_s)) < float(
        np.ptp(unsmoothed.cell_bedrock_velocity_m_s)
    )


def test_solve_cell_e2e_rejects_outside_grid_observations(
    tmp_path: Path,
) -> None:
    req = synthetic_cell_refraction_apply_request()
    input_model = synthetic_cell_refracted_arrival_input_model(
        include_outside_grid_observation=True,
    )

    result = run_synthetic_cell_refraction_statics(req=req, input_model=input_model)

    outside_trace_index = int(input_model.n_traces - 1)
    assert outside_trace_index not in result.row_trace_index_sorted.tolist()
    assert not bool(result.used_observation_mask_sorted[outside_trace_index])
    assert (
        result.trace_static_status_sorted[outside_trace_index]
        == 'outside_refractor_cell_grid'
    )
    assert result.row_midpoint_cell_id is not None
    assert np.all(result.row_midpoint_cell_id >= 0)

    write_refraction_static_artifacts(result=result, req=req, job_dir=tmp_path)
    cell_qc = json.loads(
        (tmp_path / REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME).read_text(
            encoding='utf-8'
        )
    )
    assert cell_qc['n_observations_outside_grid'] == 1


def test_solve_cell_e2e_rejects_low_fold_cells_and_projects_status(
    tmp_path: Path,
) -> None:
    payload = synthetic_cell_refraction_apply_request().model_dump(mode='json')
    payload['model']['refractor_cell']['min_observations_per_cell'] = 20
    req = RefractionStaticApplyRequest.model_validate(payload)

    result = run_synthetic_cell_refraction_statics(req=req)

    np.testing.assert_array_equal(result.active_cell_id, [1])
    np.testing.assert_array_equal(result.inactive_cell_id, [0, 2])
    assert result.row_midpoint_cell_id is not None
    assert set(result.row_midpoint_cell_id.tolist()) == {1}
    assert result.qc['min_observations_per_cell'] == 20
    assert result.qc['n_low_fold_cells'] == 2
    assert result.qc['low_fold_cell_id'] == [0, 2]
    assert result.qc['n_observations_rejected_by_low_fold_cell'] == 30
    assert 'low_fold_v2_cell' in result.source_v2_status.tolist()
    assert 'low_fold_v2_cell' in result.receiver_v2_status.tolist()

    paths = write_refraction_static_artifacts(result=result, req=req, job_dir=tmp_path)
    cell_rows = _read_csv(paths.refraction_refractor_velocity_cells_csv)
    assert [row['velocity_status'] for row in cell_rows] == [
        'low_fold',
        'solved',
        'low_fold',
    ]
    cell_qc = json.loads(
        paths.refraction_refractor_velocity_qc_json.read_text(encoding='utf-8')
    )
    assert cell_qc['min_observations_per_cell'] == 20
    assert cell_qc['n_low_fold_cells'] == 2
    assert cell_qc['n_observations_rejected_by_low_fold_cell'] == 30


def test_solve_cell_e2e_inactive_endpoint_cell_status(
    tmp_path: Path,
) -> None:
    req = synthetic_cell_refraction_apply_request()
    input_model = synthetic_cell_refracted_arrival_input_model(
        allowed_midpoint_cell_ids=(1,),
    )

    result = run_synthetic_cell_refraction_statics(req=req, input_model=input_model)

    assert result.active_cell_id is not None
    assert result.inactive_cell_id is not None
    np.testing.assert_array_equal(result.active_cell_id, [1])
    np.testing.assert_array_equal(result.inactive_cell_id, [0, 2])

    paths = write_refraction_static_artifacts(result=result, req=req, job_dir=tmp_path)
    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    inactive_source = next(row for row in source_rows if row['source_node_id'] == '6')
    inactive_receiver = next(
        row for row in receiver_rows if row['receiver_node_id'] == '8'
    )

    assert inactive_source['source_v2_cell_id'] == '2'
    assert inactive_source['v2_status'] == 'inactive_v2_cell'
    assert inactive_source['static_status'] == 'inactive_v2_cell'
    assert inactive_receiver['receiver_v2_cell_id'] == '2'
    assert inactive_receiver['v2_status'] == 'inactive_v2_cell'
    assert inactive_receiver['static_status'] == 'inactive_v2_cell'


def test_solve_cell_e2e_static_tables_match_solution_npz(
    tmp_path: Path,
) -> None:
    req = synthetic_cell_refraction_apply_request()
    result = run_synthetic_cell_refraction_statics(req=req)

    paths = write_refraction_static_artifacts(result=result, req=req, job_dir=tmp_path)

    manifest = json.loads(paths.manifest_json.read_text(encoding='utf-8'))
    manifest_names = {item['name'] for item in manifest['artifacts']}
    assert {
        REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
        REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
        REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
    }.issubset(manifest_names)

    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    assert paths.refraction_refractor_velocity_grid_npz is not None
    with (
        np.load(
            paths.refraction_refractor_velocity_grid_npz,
            allow_pickle=False,
        ) as grid,
        np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as table,
        np.load(paths.solution_npz, allow_pickle=False) as solution,
    ):
        assert paths.source_receiver_static_table_npz.name == (
            SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME
        )
        assert paths.solution_npz.name == REFRACTION_STATIC_SOLUTION_NPZ_NAME
        np.testing.assert_allclose(
            grid['v2_m_s'],
            SYNTHETIC_CELL_V2_M_S,
            atol=SYNTHETIC_CELL_V2_TOLERANCE_M_S,
        )
        _assert_source_row_matches_npz_and_solution(source_rows[0], table, solution, 0)
        _assert_receiver_row_matches_npz_and_solution(
            receiver_rows[-1],
            table,
            solution,
            len(receiver_rows) - 1,
        )


def _assert_source_row_matches_npz_and_solution(
    row: dict[str, str],
    table: np.lib.npyio.NpzFile,
    solution: np.lib.npyio.NpzFile,
    table_index: int,
) -> None:
    node_id = int(row['source_node_id'])
    node_index = _node_index(solution, node_id)
    assert int(row['source_node_id']) == int(table['source_node_id'][table_index])
    assert row['source_endpoint_key'] == str(table['source_endpoint_key'][table_index])
    assert int(row['source_v2_cell_id']) == int(
        table['source_v2_cell_id'][table_index]
    )
    assert float(row['v2_m_s']) == pytest.approx(
        float(table['source_v2_m_s'][table_index])
    )
    assert float(row['t1_ms']) == pytest.approx(
        float(solution['node_t1_time_s'][node_index]) * 1000.0
    )
    assert float(row['sh1_weathering_thickness_m']) == pytest.approx(
        float(solution['node_sh1_weathering_thickness_m'][node_index])
    )
    assert float(row['weathering_correction_ms']) == pytest.approx(
        float(solution['node_weathering_correction_s'][node_index]) * 1000.0
    )
    assert row['static_status'] == str(table['source_static_status'][table_index])


def _assert_receiver_row_matches_npz_and_solution(
    row: dict[str, str],
    table: np.lib.npyio.NpzFile,
    solution: np.lib.npyio.NpzFile,
    table_index: int,
) -> None:
    node_id = int(row['receiver_node_id'])
    node_index = _node_index(solution, node_id)
    assert int(row['receiver_node_id']) == int(
        table['receiver_node_id'][table_index]
    )
    assert row['receiver_endpoint_key'] == str(
        table['receiver_endpoint_key'][table_index]
    )
    assert int(row['receiver_v2_cell_id']) == int(
        table['receiver_v2_cell_id'][table_index]
    )
    assert float(row['v2_m_s']) == pytest.approx(
        float(table['receiver_v2_m_s'][table_index])
    )
    assert float(row['t1_ms']) == pytest.approx(
        float(solution['node_t1_time_s'][node_index]) * 1000.0
    )
    assert float(row['sh1_weathering_thickness_m']) == pytest.approx(
        float(solution['node_sh1_weathering_thickness_m'][node_index])
    )
    assert float(row['weathering_correction_ms']) == pytest.approx(
        float(solution['node_weathering_correction_s'][node_index]) * 1000.0
    )
    assert row['static_status'] == str(table['receiver_static_status'][table_index])


def _node_index(solution: np.lib.npyio.NpzFile, node_id: int) -> int:
    matches = np.flatnonzero(solution['node_id'] == node_id)
    assert matches.shape == (1,)
    return int(matches[0])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))
