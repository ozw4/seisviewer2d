from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from scipy import sparse

import app.services.time_term_robust_solver as robust_module
from app.services.time_term_design_matrix import TimeTermDesignMatrix
from app.services.time_term_robust_solver import (
    TimeTermRobustSolverOptions,
    compute_time_term_robust_scores,
    solve_time_term_robust_least_squares,
    subset_time_term_design_matrix_rows,
    summarize_time_term_robust_solver_result,
    validate_time_term_robust_solver_options,
)
from app.services.time_term_sparse_solver import TimeTermSparseSolverOptions

SOURCE_NODE_COUNT = 4
RECEIVER_NODE_COUNT = 5
N_NODES = SOURCE_NODE_COUNT + RECEIVER_NODE_COUNT
SOURCE_NODE_ID_ROWS = np.repeat(
    np.arange(SOURCE_NODE_COUNT, dtype=np.int64),
    RECEIVER_NODE_COUNT,
)
RECEIVER_NODE_ID_ROWS = (
    np.tile(np.arange(RECEIVER_NODE_COUNT, dtype=np.int64), SOURCE_NODE_COUNT)
    + SOURCE_NODE_COUNT
)
N_OBSERVATIONS = int(SOURCE_NODE_ID_ROWS.shape[0])
N_TRACES = N_OBSERVATIONS + 1
TRUE_NODE_TIME_TERM_S = np.asarray(
    [-0.012, -0.004, 0.006, 0.010, -0.006, -0.002, 0.0, 0.003, 0.005],
    dtype=np.float64,
)
NOISE_S = np.asarray(
    [
        -0.0010,
        0.0007,
        -0.0002,
        0.0011,
        -0.0006,
        0.0003,
        -0.0008,
        0.0005,
        0.0010,
        -0.0004,
        0.0009,
        -0.0012,
        0.0004,
        -0.0007,
        0.0002,
        -0.0003,
        0.0012,
        -0.0009,
        0.0006,
        -0.0005,
    ],
    dtype=np.float64,
)


def _observation_matrix() -> sparse.csr_matrix:
    row_index = np.repeat(np.arange(N_OBSERVATIONS, dtype=np.int64), 2)
    col_index = np.empty(N_OBSERVATIONS * 2, dtype=np.int64)
    col_index[0::2] = SOURCE_NODE_ID_ROWS
    col_index[1::2] = RECEIVER_NODE_ID_ROWS
    matrix = sparse.coo_matrix(
        (
            np.ones(N_OBSERVATIONS * 2, dtype=np.float64),
            (row_index, col_index),
        ),
        shape=(N_OBSERVATIONS, N_NODES),
        dtype=np.float64,
    ).tocsr()
    matrix.sort_indices()
    return matrix


def _design(
    *,
    outlier_trace: int | None = None,
    outlier_s: float = 0.080,
    noise_s: np.ndarray | None = NOISE_S,
    **overrides: Any,
) -> TimeTermDesignMatrix:
    source_node_id_sorted = np.concatenate(
        [SOURCE_NODE_ID_ROWS, np.asarray([0], dtype=np.int64)]
    )
    receiver_node_id_sorted = np.concatenate(
        [RECEIVER_NODE_ID_ROWS, np.asarray([SOURCE_NODE_COUNT], dtype=np.int64)]
    )
    data_s = (
        TRUE_NODE_TIME_TERM_S[SOURCE_NODE_ID_ROWS]
        + TRUE_NODE_TIME_TERM_S[RECEIVER_NODE_ID_ROWS]
    )
    if noise_s is not None:
        data_s = data_s + noise_s
    if outlier_trace is not None:
        data_s = data_s.copy()
        data_s[outlier_trace] += outlier_s

    source_count = np.bincount(
        SOURCE_NODE_ID_ROWS,
        minlength=N_NODES,
    ).astype(np.int64)
    receiver_count = np.bincount(
        RECEIVER_NODE_ID_ROWS,
        minlength=N_NODES,
    ).astype(np.int64)
    total_count = source_count + receiver_count
    trace_to_row = np.full(N_TRACES, -1, dtype=np.int64)
    trace_to_row[:N_OBSERVATIONS] = np.arange(N_OBSERVATIONS, dtype=np.int64)

    payload: dict[str, Any] = {
        'matrix': _observation_matrix(),
        'data_s': data_s.copy(),
        'n_traces': N_TRACES,
        'n_observations': N_OBSERVATIONS,
        'n_nodes': N_NODES,
        'used_trace_mask_sorted': np.asarray(
            [True] * N_OBSERVATIONS + [False],
            dtype=bool,
        ),
        'row_trace_index_sorted': np.arange(N_OBSERVATIONS, dtype=np.int64),
        'trace_to_row_index_sorted': trace_to_row,
        'source_node_id_sorted': source_node_id_sorted,
        'receiver_node_id_sorted': receiver_node_id_sorted,
        'row_source_node_id': SOURCE_NODE_ID_ROWS.copy(),
        'row_receiver_node_id': RECEIVER_NODE_ID_ROWS.copy(),
        'row_pick_time_after_static_s': data_s.copy(),
        'row_moveout_time_s': np.zeros(N_OBSERVATIONS, dtype=np.float64),
        'row_data_s': data_s.copy(),
        'source_observation_count_by_node': source_count,
        'receiver_observation_count_by_node': receiver_count,
        'total_observation_count_by_node': total_count,
    }
    payload.update(overrides)
    return TimeTermDesignMatrix(**payload)


def _solver_options(**overrides: Any) -> TimeTermSparseSolverOptions:
    payload: dict[str, Any] = {
        'damping_lambda': 0.0,
        'gauge': 'mean_zero',
        'solver': 'lsmr',
        'atol': 1.0e-12,
        'btol': 1.0e-12,
        'conlim': 1.0e12,
        'maxiter': 1000,
        'max_abs_node_time_term_ms': None,
        'max_abs_estimated_trace_delay_ms': None,
    }
    payload.update(overrides)
    return TimeTermSparseSolverOptions(**payload)


def test_validate_time_term_robust_options_accepts_defaults() -> None:
    options = validate_time_term_robust_solver_options(
        TimeTermRobustSolverOptions()
    )

    assert options == TimeTermRobustSolverOptions()


def test_validate_time_term_robust_options_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match='method'):
        validate_time_term_robust_solver_options(
            TimeTermRobustSolverOptions(method='median')  # type: ignore[arg-type]
        )


def test_validate_time_term_robust_options_rejects_reciprocal_pair_protection() -> None:
    with pytest.raises(ValueError, match='protect_reciprocal_pairs'):
        validate_time_term_robust_solver_options(
            TimeTermRobustSolverOptions(protect_reciprocal_pairs=True)
        )


def test_compute_time_term_robust_scores_mad() -> None:
    center_s, scale_s, score, threshold_s = compute_time_term_robust_scores(
        np.asarray([0.0, 0.001, -0.001, 0.100], dtype=np.float64),
        method='mad',
        threshold=3.5,
    )

    assert center_s == pytest.approx(0.0005)
    assert scale_s > 0.0
    assert threshold_s == pytest.approx(3.5 * scale_s)
    assert int(np.argmax(score)) == 3


def test_compute_time_term_robust_scores_sigma() -> None:
    residual = np.asarray([0.0, 0.001, -0.001, 0.100], dtype=np.float64)

    center_s, scale_s, score, threshold_s = compute_time_term_robust_scores(
        residual,
        method='sigma',
        threshold=2.0,
    )

    assert center_s == pytest.approx(float(np.mean(residual)))
    assert scale_s == pytest.approx(float(np.std(residual, ddof=0)))
    assert threshold_s == pytest.approx(2.0 * scale_s)
    assert int(np.argmax(score)) == 3


def test_compute_time_term_robust_scores_zero_scale_returns_zero_scores() -> None:
    center_s, scale_s, score, threshold_s = compute_time_term_robust_scores(
        np.zeros(3, dtype=np.float64),
        method='mad',
        threshold=3.5,
    )

    assert center_s == 0.0
    assert scale_s == 0.0
    assert threshold_s == 0.0
    np.testing.assert_array_equal(score, np.zeros(3, dtype=np.float64))


def test_subset_time_term_design_matrix_rows_preserves_order_and_recomputes_counts() -> None:
    design = _design()
    row_mask = np.zeros(N_OBSERVATIONS, dtype=np.int64)
    selected_rows = np.asarray([1, 7, 19], dtype=np.int64)
    row_mask[selected_rows] = 1

    subset = subset_time_term_design_matrix_rows(design, row_mask)

    assert sparse.isspmatrix_csr(subset.matrix)
    assert subset.n_observations == 3
    np.testing.assert_array_equal(subset.row_trace_index_sorted, selected_rows)
    np.testing.assert_allclose(
        subset.matrix.toarray(),
        design.matrix[selected_rows, :].toarray(),
    )
    np.testing.assert_allclose(subset.data_s, design.data_s[selected_rows])
    expected_trace_to_row = np.full(N_TRACES, -1, dtype=np.int64)
    expected_trace_to_row[selected_rows] = np.arange(3, dtype=np.int64)
    np.testing.assert_array_equal(
        subset.trace_to_row_index_sorted,
        expected_trace_to_row,
    )
    expected_used_trace_mask = np.zeros(N_TRACES, dtype=bool)
    expected_used_trace_mask[selected_rows] = True
    np.testing.assert_array_equal(
        subset.used_trace_mask_sorted,
        expected_used_trace_mask,
    )
    np.testing.assert_array_equal(
        subset.source_observation_count_by_node,
        np.bincount(SOURCE_NODE_ID_ROWS[selected_rows], minlength=N_NODES),
    )
    np.testing.assert_array_equal(
        subset.receiver_observation_count_by_node,
        np.bincount(RECEIVER_NODE_ID_ROWS[selected_rows], minlength=N_NODES),
    )
    np.testing.assert_array_equal(
        subset.total_observation_count_by_node,
        subset.source_observation_count_by_node
        + subset.receiver_observation_count_by_node,
    )


def test_robust_disabled_runs_single_sparse_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real_solve = robust_module.solve_time_term_sparse_least_squares

    def wrapped_solve(*args: Any, **kwargs: Any):
        calls.append((args, kwargs))
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(
        robust_module,
        'solve_time_term_sparse_least_squares',
        wrapped_solve,
    )

    result = solve_time_term_robust_least_squares(
        _design(outlier_trace=0),
        solver_options=_solver_options(),
        robust_options=TimeTermRobustSolverOptions(enabled=False),
    )

    assert len(calls) == 1
    assert result.stop_reason == 'disabled'
    assert len(result.iterations) == 1
    assert result.final_solver_result is result.iterations[0].solver_result
    np.testing.assert_array_equal(
        result.final_used_trace_mask_sorted,
        result.initial_used_trace_mask_sorted,
    )
    np.testing.assert_array_equal(
        result.final_rejected_trace_mask_sorted,
        np.zeros(N_TRACES, dtype=bool),
    )
    np.testing.assert_array_equal(
        result.rejected_iteration_sorted,
        np.full(N_TRACES, -1, dtype=np.int64),
    )


def test_robust_mad_rejects_time_term_outlier_row() -> None:
    result = solve_time_term_robust_least_squares(
        _design(outlier_trace=0),
        solver_options=_solver_options(),
        robust_options=TimeTermRobustSolverOptions(method='mad', threshold=10.0),
    )

    assert result.n_rejected_traces == 1
    assert not bool(result.final_used_trace_mask_sorted[0])
    assert bool(result.final_rejected_trace_mask_sorted[0])
    assert result.rejected_iteration_sorted[0] == 0
    assert result.row_rejected_iteration[0] == 0
    assert result.stop_reason in {'converged', 'zero_scale'}
    np.testing.assert_array_equal(
        result.final_row_rejected_mask,
        result.initial_row_used_mask & ~result.final_row_used_mask,
    )
    assert result.iterations[0].solver_result.system.n_observation_rows == 20
    assert result.final_solver_result.system.n_observation_rows == 19
    assert result.final_solver_result.used_trace_mask_sorted[0] == np.False_


def test_robust_sigma_rejects_time_term_outlier_row() -> None:
    result = solve_time_term_robust_least_squares(
        _design(outlier_trace=0),
        solver_options=_solver_options(),
        robust_options=TimeTermRobustSolverOptions(method='sigma', threshold=3.0),
    )

    assert result.n_rejected_traces == 1
    assert not bool(result.final_used_trace_mask_sorted[0])
    assert result.stop_reason in {'converged', 'zero_scale'}


def test_robust_final_solver_result_is_recomputed_when_max_iterations_reached() -> None:
    result = solve_time_term_robust_least_squares(
        _design(outlier_trace=0),
        solver_options=_solver_options(),
        robust_options=TimeTermRobustSolverOptions(
            method='mad',
            threshold=10.0,
            max_iterations=1,
        ),
    )

    assert result.stop_reason == 'max_iterations'
    assert len(result.iterations) == 1
    assert result.iterations[0].solver_result.system.n_observation_rows == 20
    assert result.final_solver_result.system.n_observation_rows == 19
    np.testing.assert_array_equal(
        result.final_solver_result.used_trace_mask_sorted,
        result.final_used_trace_mask_sorted,
    )


def test_robust_zero_scale_stops_without_rejection() -> None:
    result = solve_time_term_robust_least_squares(
        _design(noise_s=None),
        solver_options=_solver_options(),
        robust_options=TimeTermRobustSolverOptions(method='mad', threshold=3.5),
    )

    assert result.stop_reason == 'zero_scale'
    assert result.n_rejected_traces == 0
    np.testing.assert_array_equal(
        result.final_used_trace_mask_sorted,
        result.initial_used_trace_mask_sorted,
    )


def test_robust_rejects_min_used_fraction_violation() -> None:
    with pytest.raises(ValueError, match='min_used_fraction'):
        solve_time_term_robust_least_squares(
            _design(outlier_trace=0),
            solver_options=_solver_options(),
            robust_options=TimeTermRobustSolverOptions(
                method='mad',
                threshold=10.0,
                min_used_fraction=1.0,
            ),
        )


def test_robust_rejects_initial_min_used_observations_violation() -> None:
    with pytest.raises(ValueError, match='min_used_observations'):
        solve_time_term_robust_least_squares(
            _design(),
            solver_options=_solver_options(),
            robust_options=TimeTermRobustSolverOptions(min_used_observations=21),
        )


def test_summarize_time_term_robust_solver_result_is_json_safe() -> None:
    result = solve_time_term_robust_least_squares(
        _design(outlier_trace=0),
        solver_options=_solver_options(),
        robust_options=TimeTermRobustSolverOptions(method='mad', threshold=10.0),
    )

    summary = summarize_time_term_robust_solver_result(result)

    json.dumps(summary, allow_nan=False)
    assert summary['enabled'] is True
    assert summary['method'] == 'mad'
    assert summary['threshold'] == 10.0
    assert summary['n_rejected_traces'] == 1
    assert summary['final']['node_time_term_ms']['count'] == N_NODES
