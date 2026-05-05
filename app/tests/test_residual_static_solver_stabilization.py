from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import sparse

from app.services.residual_static_design_matrix import (
    build_residual_static_column_layout,
)
from app.services.residual_static_inputs import ResidualStaticSolverInputs
from app.services.residual_static_sparse_solver import (
    ResidualStaticLsmrOptions,
    ResidualStaticStabilizationOptions,
    build_delay_damping_rows,
    build_residual_static_observation_graph_summary,
    build_stabilized_residual_static_augmented_system,
    build_zero_mean_gauge_rows,
    solve_residual_static_stabilized_least_squares,
    stabilization_options_from_request_solver,
    validate_minimum_residual_static_data,
    validate_residual_static_stabilization_options,
    validate_residual_static_used_mask,
)

SOURCE_DELAY_S = np.asarray([-0.012, 0.004, 0.008], dtype=np.float64)
RECEIVER_DELAY_S = np.asarray([-0.006, -0.002, 0.003, 0.005], dtype=np.float64)
INTERCEPT_S = 0.080
SLOWNESS_S_PER_OFFSET_UNIT = 2.5e-5


def _synthetic_inputs(
    *,
    moveout_model: str = 'linear_abs_offset',
    **overrides: Any,
) -> ResidualStaticSolverInputs:
    source_unique_ids = np.asarray([101, 102, 103], dtype=np.int32)
    receiver_unique_ids = np.asarray([201, 202, 203, 204], dtype=np.int32)
    n_sources = int(source_unique_ids.shape[0])
    n_receivers = int(receiver_unique_ids.shape[0])
    source_index = np.repeat(np.arange(n_sources, dtype=np.int64), n_receivers)
    receiver_index = np.tile(np.arange(n_receivers, dtype=np.int64), n_sources)
    n_traces = int(source_index.shape[0])

    abs_offset = np.asarray(
        [
            [100.0, 185.0, 260.0, 415.0],
            [150.0, 245.0, 395.0, 525.0],
            [230.0, 315.0, 475.0, 655.0],
        ],
        dtype=np.float64,
    ).reshape(-1)
    offset_sign = np.where(np.arange(n_traces) % 2 == 0, -1.0, 1.0)
    offset = abs_offset * offset_sign

    if moveout_model == 'linear_abs_offset':
        moveout_s = INTERCEPT_S + SLOWNESS_S_PER_OFFSET_UNIT * abs_offset
        offset_sorted = offset
        abs_offset_sorted = abs_offset
        offset_byte = 37
    else:
        moveout_s = np.full(n_traces, INTERCEPT_S, dtype=np.float64)
        offset_sorted = None
        abs_offset_sorted = None
        offset_byte = None

    pick_time_after_datum = (
        moveout_s
        + SOURCE_DELAY_S[source_index]
        + RECEIVER_DELAY_S[receiver_index]
    )
    valid_pick_mask = np.ones(n_traces, dtype=bool)
    payload: dict[str, Any] = {
        'picks_time_s_sorted': pick_time_after_datum.copy(),
        'valid_pick_mask_sorted': valid_pick_mask,
        'pick_time_after_datum_s_sorted': pick_time_after_datum.copy(),
        'datum_trace_shift_s_sorted': np.zeros(n_traces, dtype=np.float64),
        'source_id_sorted': source_unique_ids[source_index],
        'receiver_id_sorted': receiver_unique_ids[receiver_index],
        'source_unique_ids': source_unique_ids,
        'receiver_unique_ids': receiver_unique_ids,
        'source_index_sorted': source_index,
        'receiver_index_sorted': receiver_index,
        'source_valid_pick_counts': np.bincount(
            source_index,
            minlength=n_sources,
        ).astype(np.int64),
        'receiver_valid_pick_counts': np.bincount(
            receiver_index,
            minlength=n_receivers,
        ).astype(np.int64),
        'offset_sorted': offset_sorted,
        'abs_offset_sorted': abs_offset_sorted,
        'key1_sorted': np.repeat([10, 20, 30], n_receivers).astype(np.int64),
        'key2_sorted': np.tile([1, 2, 3, 4], n_sources).astype(np.int64),
        'source_elevation_m_sorted': np.zeros(n_traces, dtype=np.float64),
        'receiver_elevation_m_sorted': np.zeros(n_traces, dtype=np.float64),
        'dt': 0.004,
        'n_traces': n_traces,
        'n_samples': 64,
        'key1_byte': 189,
        'key2_byte': 193,
        'source_id_byte': 17,
        'receiver_id_byte': 13,
        'offset_byte': offset_byte,
        'moveout_model': moveout_model,
        'input_file_id': 'corrected-file-id',
        'datum_source_file_id': 'source-file-id',
        'datum_job_id': 'datum-job',
        'pick_source_kind': 'batch_npz',
        'metadata': {'source': 'stabilization-test'},
    }
    payload.update(overrides)
    return ResidualStaticSolverInputs(**payload)


def _options(**overrides: Any) -> ResidualStaticStabilizationOptions:
    payload: dict[str, Any] = {
        'gauge': 'zero_mean_source_receiver',
        'damping_lambda': 0.0,
        'min_valid_picks': 10,
        'min_picks_per_source': 1,
        'min_picks_per_receiver': 1,
        'max_abs_estimated_delay_ms': 250.0,
    }
    payload.update(overrides)
    return ResidualStaticStabilizationOptions(**payload)


def test_validate_stabilization_options_accepts_request_like_object() -> None:
    class RequestSolver:
        gauge = 'zero_mean_source_receiver'
        damping_lambda = 0.25
        min_valid_picks = 12
        min_picks_per_source = 2
        min_picks_per_receiver = 3
        max_abs_estimated_delay_ms = 50.0

    options = stabilization_options_from_request_solver(RequestSolver())

    assert options == ResidualStaticStabilizationOptions(
        gauge='zero_mean_source_receiver',
        damping_lambda=0.25,
        min_valid_picks=12,
        min_picks_per_source=2,
        min_picks_per_receiver=3,
        max_abs_estimated_delay_ms=50.0,
    )


def test_validate_stabilization_options_rejects_bool_numeric_fields() -> None:
    with pytest.raises(ValueError, match='min_valid_picks'):
        validate_residual_static_stabilization_options(
            _options(min_valid_picks=True)
        )

    with pytest.raises(ValueError, match='damping_lambda'):
        validate_residual_static_stabilization_options(
            _options(damping_lambda=False)
        )


def test_validate_stabilization_options_rejects_unsupported_gauge() -> None:
    with pytest.raises(ValueError, match='gauge'):
        validate_residual_static_stabilization_options(_options(gauge='free'))


def test_validate_used_mask_defaults_to_valid_pick_mask_and_allows_nan_invalid() -> None:
    valid_mask = np.ones(12, dtype=bool)
    valid_mask[3] = False
    pick_time = _synthetic_inputs().pick_time_after_datum_s_sorted.copy()
    pick_time[3] = np.nan
    inputs = _synthetic_inputs(
        valid_pick_mask_sorted=valid_mask,
        pick_time_after_datum_s_sorted=pick_time,
    )

    used_mask = validate_residual_static_used_mask(inputs, None)

    np.testing.assert_array_equal(used_mask, valid_mask)


def test_validate_used_mask_rejects_invalid_pick_subset_violation() -> None:
    valid_mask = np.ones(12, dtype=bool)
    valid_mask[3] = False
    inputs = _synthetic_inputs(valid_pick_mask_sorted=valid_mask)
    used_mask = np.ones(12, dtype=bool)

    with pytest.raises(ValueError, match='subset'):
        validate_residual_static_used_mask(inputs, used_mask)


def test_validate_used_mask_rejects_nonfinite_used_pick_time() -> None:
    pick_time = _synthetic_inputs().pick_time_after_datum_s_sorted.copy()
    pick_time[0] = np.nan

    with pytest.raises(ValueError, match='pick_time_after_datum'):
        validate_residual_static_used_mask(
            _synthetic_inputs(pick_time_after_datum_s_sorted=pick_time),
            None,
        )


def test_observation_graph_summary_connected_complete_grid() -> None:
    inputs = _synthetic_inputs()
    graph = build_residual_static_observation_graph_summary(
        inputs,
        used_mask_sorted=np.ones(inputs.n_traces, dtype=bool),
    )

    assert graph.n_components == 1
    np.testing.assert_array_equal(graph.source_component_index, [0, 0, 0])
    np.testing.assert_array_equal(graph.receiver_component_index, [0, 0, 0, 0])
    np.testing.assert_array_equal(graph.component_observation_counts, [12])
    np.testing.assert_array_equal(graph.component_source_counts, [3])
    np.testing.assert_array_equal(graph.component_receiver_counts, [4])


def test_minimum_data_summary_counts_and_offset_span() -> None:
    inputs = _synthetic_inputs()

    summary = validate_minimum_residual_static_data(
        inputs,
        used_mask_sorted=np.ones(inputs.n_traces, dtype=bool),
        options=_options(),
    )

    assert summary.n_used_picks == 12
    assert summary.n_sources == 3
    assert summary.n_receivers == 4
    assert summary.n_effective_parameters == 7
    np.testing.assert_array_equal(summary.source_used_pick_counts, [4, 4, 4])
    np.testing.assert_array_equal(summary.receiver_used_pick_counts, [3, 3, 3, 3])
    assert summary.underconstrained_source_ids.dtype == np.int32
    assert summary.underconstrained_receiver_ids.dtype == np.int32
    assert summary.abs_offset_min == 100.0
    assert summary.abs_offset_max == 655.0
    assert summary.abs_offset_span == 555.0


def test_minimum_data_rejects_source_with_insufficient_used_picks() -> None:
    inputs = _synthetic_inputs()
    used_mask = inputs.source_index_sorted != 2

    with pytest.raises(ValueError, match='min_picks_per_source'):
        validate_minimum_residual_static_data(
            inputs,
            used_mask_sorted=used_mask,
            options=_options(min_valid_picks=1),
        )


def test_minimum_data_rejects_disconnected_observation_graph() -> None:
    inputs = _synthetic_inputs(moveout_model='none')
    used_mask = np.asarray(
        [
            True,
            True,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
        ],
        dtype=bool,
    )

    with pytest.raises(ValueError, match='connected'):
        validate_minimum_residual_static_data(
            inputs,
            used_mask_sorted=used_mask,
            options=_options(min_valid_picks=1),
        )


def test_minimum_data_rejects_linear_abs_offset_without_offset_span() -> None:
    inputs = _synthetic_inputs(
        abs_offset_sorted=np.full(12, 100.0, dtype=np.float64),
        offset_sorted=np.full(12, 100.0, dtype=np.float64),
    )

    with pytest.raises(ValueError, match='span'):
        validate_minimum_residual_static_data(
            inputs,
            used_mask_sorted=np.ones(inputs.n_traces, dtype=bool),
            options=_options(),
        )


def test_gauge_and_damping_rows_target_only_delay_columns() -> None:
    layout = build_residual_static_column_layout(_synthetic_inputs())

    gauge_rows, gauge_cols, gauge_data, gauge_rhs = build_zero_mean_gauge_rows(
        layout
    )
    gauge_matrix = sparse.coo_matrix(
        (gauge_data, (gauge_rows, gauge_cols)),
        shape=(2, layout.n_model_parameters),
    ).toarray()

    assert gauge_rhs.tolist() == [0.0, 0.0]
    assert gauge_matrix[0, layout.intercept_col] == 0.0
    assert layout.slowness_col is not None
    assert gauge_matrix[0, layout.slowness_col] == 0.0
    np.testing.assert_allclose(gauge_matrix[0, layout.source_delay_cols], 1.0 / 3.0)
    np.testing.assert_allclose(gauge_matrix[1, layout.receiver_delay_cols], 1.0 / 4.0)

    damping_rows, damping_cols, damping_data, damping_rhs = build_delay_damping_rows(
        layout,
        damping_lambda=0.2,
    )

    np.testing.assert_array_equal(damping_rows, np.arange(7, dtype=np.int64))
    np.testing.assert_array_equal(
        damping_cols,
        np.concatenate([layout.source_delay_cols, layout.receiver_delay_cols]),
    )
    np.testing.assert_allclose(damping_data, np.full(7, 0.2, dtype=np.float64))
    np.testing.assert_allclose(damping_rhs, np.zeros(7, dtype=np.float64))

    empty_rows = build_delay_damping_rows(layout, damping_lambda=0.0)
    assert all(part.size == 0 for part in empty_rows)


def test_augmented_system_includes_observation_gauge_and_damping_rows() -> None:
    inputs = _synthetic_inputs()

    augmented = build_stabilized_residual_static_augmented_system(
        inputs,
        options=_options(damping_lambda=0.5),
    )

    assert sparse.isspmatrix_csr(augmented.matrix)
    assert augmented.n_observation_rows == 12
    assert augmented.n_gauge_rows == 2
    assert augmented.n_damping_rows == 7
    assert augmented.n_rows == 21
    assert augmented.n_cols == 9
    assert augmented.rhs_s.shape == (21,)
    np.testing.assert_array_equal(
        augmented.observation_row_to_sorted_trace_index,
        np.arange(12, dtype=np.int64),
    )


def test_stabilized_solver_recovers_zero_mean_synthetic_delays() -> None:
    inputs = _synthetic_inputs()

    result = solve_residual_static_stabilized_least_squares(
        inputs,
        stabilization_options=_options(),
        lsmr_options=ResidualStaticLsmrOptions(
            atol=1.0e-12,
            btol=1.0e-12,
            conlim=1.0e12,
            maxiter=10000,
        ),
    )

    assert result.rank_deficient_possible is False
    assert result.n_observations == 12
    assert result.n_model_parameters == 9
    assert result.n_gauge_rows == 2
    assert result.n_damping_rows == 0
    np.testing.assert_allclose(
        result.parameter_parts.source_delay_s,
        SOURCE_DELAY_S,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        result.parameter_parts.receiver_delay_s,
        RECEIVER_DELAY_S,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        np.mean(result.parameter_parts.source_delay_s),
        0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.mean(result.parameter_parts.receiver_delay_s),
        0.0,
        atol=1.0e-10,
    )
    assert result.parameter_parts.intercept_s == pytest.approx(INTERCEPT_S, abs=1e-8)
    assert result.parameter_parts.slowness_s_per_offset_unit == pytest.approx(
        SLOWNESS_S_PER_OFFSET_UNIT,
        abs=1e-10,
    )
    np.testing.assert_allclose(
        result.model_evaluation.residual_s_sorted[result.used_mask_sorted],
        np.zeros(inputs.n_traces, dtype=np.float64),
        atol=1.0e-8,
    )
    assert not hasattr(result, 'applied_residual_shift_s_sorted')


def test_stabilized_solver_rejects_estimated_delay_limit_exceedance() -> None:
    with pytest.raises(ValueError, match='max_abs_estimated_delay_ms'):
        solve_residual_static_stabilized_least_squares(
            _synthetic_inputs(),
            stabilization_options=_options(max_abs_estimated_delay_ms=1.0),
            lsmr_options=ResidualStaticLsmrOptions(
                atol=1.0e-12,
                btol=1.0e-12,
                conlim=1.0e12,
                maxiter=10000,
            ),
        )
