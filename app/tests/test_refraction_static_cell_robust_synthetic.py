from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from app.statics.refraction.artifacts import write_refraction_static_artifacts
from app.statics.refraction.contracts.result_types import (
    RefractionDatumStaticsResult,
    RefractionStaticInputModel,
)
from app.tests._refraction_static_synthetic import (
    SYNTHETIC_CELL_NODE_T1_S,
    SYNTHETIC_CELL_V2_M_S,
    SYNTHETIC_CELL_V2_TOLERANCE_M_S,
    run_synthetic_cell_refraction_statics,
    synthetic_cell_midpoint_cell_id_sorted,
    synthetic_cell_refracted_arrival_input_model,
    synthetic_cell_refraction_apply_request,
)

OUTLIER_SHIFT_PATTERN_S = np.asarray([0.050, -0.060, 0.100], dtype=np.float64)
RANDOM_OUTLIER_SEED = 432


def test_cell_v2_robust_solver_rejects_random_bad_picks() -> None:
    input_model, outlier_indices = _random_sparse_outlier_input_model(
        seed=RANDOM_OUTLIER_SEED
    )

    first_result = _run_cell_statics(input_model, robust_enabled=True)
    second_result = _run_cell_statics(input_model, robust_enabled=True)

    _assert_rejected_outliers(first_result, outlier_indices)
    np.testing.assert_array_equal(
        first_result.rejected_by_robust_mask,
        second_result.rejected_by_robust_mask,
    )
    np.testing.assert_allclose(
        _required_cell_v2(first_result),
        _required_cell_v2(second_result),
        atol=0.0,
        rtol=0.0,
    )
    _assert_known_cell_v2(first_result)
    _assert_used_t1_pair_sums_close(first_result)


def test_cell_v2_robust_solver_handles_clustered_bad_picks_in_one_cell() -> None:
    base = synthetic_cell_refracted_arrival_input_model()
    outlier_indices = np.asarray([5, 6, 7, 13, 14, 15], dtype=np.int64)
    midpoint_cell = synthetic_cell_midpoint_cell_id_sorted(base)
    assert set(midpoint_cell[outlier_indices].tolist()) == {1}
    input_model = _with_outlier_shifts(
        base,
        outlier_indices=outlier_indices,
        shifts_s=_outlier_shifts(outlier_indices.shape[0]),
    )

    result = _run_cell_statics(input_model, robust_enabled=True)

    _assert_rejected_outliers(result, outlier_indices)
    _assert_known_cell_v2(result)
    _assert_used_t1_pair_sums_close(result)
    cell_v2 = _required_cell_v2(result)
    cell_one_position = _cell_position(result, cell_id=1)
    assert cell_v2[cell_one_position] == pytest.approx(
        SYNTHETIC_CELL_V2_M_S[1],
        abs=SYNTHETIC_CELL_V2_TOLERANCE_M_S,
    )


def test_cell_v2_robust_solver_handles_bad_source_group() -> None:
    base = synthetic_cell_refracted_arrival_input_model()
    outlier_indices = np.asarray([48, 49, 50, 51, 52], dtype=np.int64)
    bad_source_node = int(base.source_node_id_sorted[outlier_indices[0]])
    assert set(base.source_node_id_sorted[outlier_indices].tolist()) == {
        bad_source_node
    }
    input_model = _with_outlier_shifts(
        base,
        outlier_indices=outlier_indices,
        shifts_s=_outlier_shifts(outlier_indices.shape[0]),
    )

    result = _run_cell_statics(input_model, robust_enabled=True)

    _assert_rejected_outliers(result, outlier_indices)
    _assert_known_cell_v2(result)
    _assert_used_t1_pair_sums_close(result)
    node_position = _node_position(result, bad_source_node)
    assert result.node_rejected_pick_count[node_position] >= outlier_indices.shape[0]


def test_cell_v2_robust_qc_reports_rejected_outliers(tmp_path: Path) -> None:
    input_model, outlier_indices = _random_sparse_outlier_input_model(
        seed=RANDOM_OUTLIER_SEED
    )
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

    rejected_count = int(np.count_nonzero(result.rejected_by_robust_mask))
    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    assert qc['observations']['n_rejected_by_robust'] == rejected_count
    assert qc['observations']['n_used_observations'] == int(
        np.count_nonzero(result.used_row_mask)
    )
    assert qc['first_break_fit']['robust_enabled'] is True
    assert qc['first_break_fit']['robust_iteration_count'] >= 1

    residual_by_trace = _rows_by_trace_index(paths.first_break_residuals_csv)
    trace_by_trace = _rows_by_trace_index(paths.refraction_statics_csv)
    for trace_index in outlier_indices.tolist():
        residual_row = residual_by_trace[trace_index]
        assert residual_row['used'] == 'false'
        assert residual_row['rejected_by_robust'] == 'true'
        assert residual_row['rejection_reason'] == 'robust_outlier'
        assert trace_by_trace[trace_index]['used_observation'] == 'false'

    clean_row = residual_by_trace[0]
    assert clean_row['used'] == 'true'
    assert clean_row['rejected_by_robust'] == 'false'
    assert clean_row['rejection_reason'] == 'ok'


def test_cell_v2_solution_degrades_without_robust_rejection_for_large_outliers() -> None:
    input_model, _ = _random_sparse_outlier_input_model(seed=RANDOM_OUTLIER_SEED)

    robust_result = _run_cell_statics(input_model, robust_enabled=True)
    plain_result = _run_cell_statics(input_model, robust_enabled=False)

    _assert_known_cell_v2(robust_result)
    plain_cell_error_m_s = np.max(
        np.abs(_required_cell_v2(plain_result) - SYNTHETIC_CELL_V2_M_S)
    )
    assert plain_cell_error_m_s > 250.0
    assert _used_residual_rms_ms(robust_result) < 1.0e-6
    assert _used_residual_mad_ms(robust_result) < 1.0e-6
    assert _used_residual_rms_ms(plain_result) > 5.0
    assert _used_residual_mad_ms(plain_result) > 1.0


def _random_sparse_outlier_input_model(
    *,
    seed: int,
) -> tuple[RefractionStaticInputModel, np.ndarray]:
    base = synthetic_cell_refracted_arrival_input_model()
    eligible = np.flatnonzero(base.pick_time_s_sorted > 0.061)
    rng = np.random.default_rng(seed)
    outlier_indices = np.sort(rng.choice(eligible, size=6, replace=False))
    midpoint_cell = synthetic_cell_midpoint_cell_id_sorted(base)
    assert {0, 1, 2}.issubset(set(midpoint_cell[outlier_indices].tolist()))
    return (
        _with_outlier_shifts(
            base,
            outlier_indices=outlier_indices,
            shifts_s=_outlier_shifts(outlier_indices.shape[0]),
        ),
        outlier_indices,
    )


def _with_outlier_shifts(
    input_model: RefractionStaticInputModel,
    *,
    outlier_indices: np.ndarray,
    shifts_s: np.ndarray,
) -> RefractionStaticInputModel:
    pick_time_s = input_model.pick_time_s_sorted.copy()
    pick_time_s[outlier_indices] += shifts_s
    if np.any(pick_time_s[outlier_indices] < 0.0):
        raise AssertionError('synthetic outlier shifts produced negative pick times')
    return replace(
        input_model,
        pick_time_s_sorted=np.ascontiguousarray(pick_time_s, dtype=np.float64),
    )


def _outlier_shifts(count: int) -> np.ndarray:
    return np.resize(OUTLIER_SHIFT_PATTERN_S, int(count)).astype(
        np.float64,
        copy=False,
    )


def _run_cell_statics(
    input_model: RefractionStaticInputModel,
    *,
    robust_enabled: bool,
) -> RefractionDatumStaticsResult:
    return run_synthetic_cell_refraction_statics(
        req=synthetic_cell_refraction_apply_request(
            robust_enabled=robust_enabled,
        ),
        input_model=input_model,
    )


def _assert_rejected_outliers(
    result: RefractionDatumStaticsResult,
    outlier_indices: np.ndarray,
) -> None:
    rejected_trace_indices = set(
        result.row_trace_index_sorted[result.rejected_by_robust_mask].tolist()
    )
    assert set(outlier_indices.tolist()).issubset(rejected_trace_indices)
    assert int(np.count_nonzero(result.rejected_by_robust_mask)) >= int(
        outlier_indices.shape[0]
    )
    for trace_index in outlier_indices.tolist():
        row_position = np.flatnonzero(result.row_trace_index_sorted == trace_index)
        assert row_position.shape == (1,)
        assert not bool(result.used_row_mask[int(row_position[0])])


def _assert_known_cell_v2(result: RefractionDatumStaticsResult) -> None:
    np.testing.assert_allclose(
        _required_cell_v2(result),
        SYNTHETIC_CELL_V2_M_S,
        atol=SYNTHETIC_CELL_V2_TOLERANCE_M_S,
        rtol=0.0,
    )


def _assert_used_t1_pair_sums_close(result: RefractionDatumStaticsResult) -> None:
    node_t1_by_id = {
        int(node_id): float(result.node_half_intercept_time_s[index])
        for index, node_id in enumerate(result.node_id.tolist())
    }
    solved_pair_sums: list[float] = []
    true_pair_sums: list[float] = []
    for row_index in np.flatnonzero(result.used_row_mask).tolist():
        source = int(result.row_source_node_id[row_index])
        receiver = int(result.row_receiver_node_id[row_index])
        solved_pair_sums.append(node_t1_by_id[source] + node_t1_by_id[receiver])
        true_pair_sums.append(
            float(SYNTHETIC_CELL_NODE_T1_S[source])
            + float(SYNTHETIC_CELL_NODE_T1_S[receiver])
        )
    np.testing.assert_allclose(
        solved_pair_sums,
        true_pair_sums,
        atol=1.0e-8,
        rtol=0.0,
    )


def _required_cell_v2(result: RefractionDatumStaticsResult) -> np.ndarray:
    assert result.cell_bedrock_velocity_m_s is not None
    return result.cell_bedrock_velocity_m_s


def _cell_position(result: RefractionDatumStaticsResult, *, cell_id: int) -> int:
    assert result.active_cell_id is not None
    matches = np.flatnonzero(result.active_cell_id == cell_id)
    assert matches.shape == (1,)
    return int(matches[0])


def _node_position(result: RefractionDatumStaticsResult, node_id: int) -> int:
    matches = np.flatnonzero(result.node_id == node_id)
    assert matches.shape == (1,)
    return int(matches[0])


def _used_residual_rms_ms(result: RefractionDatumStaticsResult) -> float:
    residual_ms = result.residual_time_s[result.used_row_mask] * 1000.0
    return float(np.sqrt(np.mean(residual_ms * residual_ms)))


def _used_residual_mad_ms(result: RefractionDatumStaticsResult) -> float:
    residual_ms = result.residual_time_s[result.used_row_mask] * 1000.0
    median = float(np.median(residual_ms))
    return float(1.4826 * np.median(np.abs(residual_ms - median)))


def _rows_by_trace_index(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return {
            int(row['sorted_trace_index']): row
            for row in csv.DictReader(handle)
        }
