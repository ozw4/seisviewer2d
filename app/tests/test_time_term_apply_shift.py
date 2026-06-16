from __future__ import annotations

import json
from dataclasses import fields
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from seis_statics.time_term.apply_shift import (
    DELAY_TO_SHIFT_CONVENTION,
    FINAL_SHIFT_CONVENTION,
    SIGN_CONVENTION,
    TimeTermAppliedShiftOptions,
    TimeTermAppliedShiftResult,
    build_time_term_applied_shift_result,
    delay_to_applied_shift_s,
    summarize_time_term_applied_shift_result,
)
from seis_statics.time_term import (
    ORDER,
    TimeTermInversionInputs,
    TimeTermRobustSolverOptions,
    TimeTermRobustSolverResult,
    TimeTermSolverSystem,
    TimeTermSparseSolverResult,
)

N_TRACES = 3
N_NODES = 3
DT = 0.002
SOURCE_NODE_ID_SORTED = np.asarray([0, 1, 2], dtype=np.int64)
RECEIVER_NODE_ID_SORTED = np.asarray([1, 2, 2], dtype=np.int64)
NODE_TIME_TERM_S = np.asarray([0.010, 0.005, -0.002], dtype=np.float64)
ESTIMATED_DELAY_S = np.asarray([0.015, 0.003, -0.004], dtype=np.float64)
APPLIED_WEATHERING_SHIFT_S = np.asarray([-0.015, -0.003, 0.004], dtype=np.float64)
DATUM_SHIFT_S = np.asarray([-0.020, -0.020, -0.020], dtype=np.float64)
RESIDUAL_SHIFT_S = np.asarray([0.001, 0.002, 0.003], dtype=np.float64)
FINAL_SHIFT_S = np.asarray([-0.034, -0.021, -0.013], dtype=np.float64)
VALID_PICK_MASK = np.asarray([True, True, False], dtype=bool)
USED_TRACE_MASK = np.asarray([True, True, False], dtype=bool)


def _inputs(**overrides: Any) -> TimeTermInversionInputs:
    zeros = np.zeros(N_TRACES, dtype=np.float64)
    payload: dict[str, Any] = {
        'n_traces': N_TRACES,
        'n_samples': 200,
        'dt': DT,
        'key1_byte': 189,
        'key2_byte': 193,
        'pick_time_raw_s_sorted': np.asarray([0.10, 0.12, 0.14], dtype=np.float64),
        'valid_pick_mask_sorted': VALID_PICK_MASK.copy(),
        'datum_trace_shift_s_sorted': DATUM_SHIFT_S.copy(),
        'residual_applied_shift_s_sorted': RESIDUAL_SHIFT_S.copy(),
        'pick_time_after_static_s_sorted': np.asarray(
            [0.081, 0.102, 0.123],
            dtype=np.float64,
        ),
        'source_node_id_sorted': SOURCE_NODE_ID_SORTED.copy(),
        'receiver_node_id_sorted': RECEIVER_NODE_ID_SORTED.copy(),
        'n_nodes': N_NODES,
        'source_id_sorted': np.asarray([100, 101, 102], dtype=np.int64),
        'receiver_id_sorted': np.asarray([200, 201, 202], dtype=np.int64),
        'offset_sorted': np.asarray([1000.0, 1200.0, 1400.0], dtype=np.float64),
        'source_x_m_sorted': zeros.copy(),
        'source_y_m_sorted': zeros.copy(),
        'receiver_x_m_sorted': zeros.copy(),
        'receiver_y_m_sorted': zeros.copy(),
        'source_elevation_m_sorted': zeros.copy(),
        'receiver_elevation_m_sorted': zeros.copy(),
        'source_depth_m_sorted': zeros.copy(),
        'input_file_id': 'synthetic-file',
        'pick_source_description': 'synthetic picks',
        'datum_solution_path': None,
        'residual_solution_path': None,
        'linkage_artifact_path': None,
    }
    payload.update(overrides)
    return TimeTermInversionInputs(**payload)


def _system(*, n_nodes: int = N_NODES, n_observations: int = 2) -> TimeTermSolverSystem:
    return TimeTermSolverSystem(
        augmented_matrix=sparse.csr_matrix(
            (n_observations, n_nodes),
            dtype=np.float64,
        ),
        augmented_data_s=np.zeros(n_observations, dtype=np.float64),
        n_observation_rows=n_observations,
        n_damping_rows=0,
        n_gauge_rows=0,
        n_augmented_rows=n_observations,
        n_nodes=n_nodes,
        damping_prior_s=np.zeros(n_nodes, dtype=np.float64),
        gauge_mode='none',
        component_id_by_node=np.zeros(n_nodes, dtype=np.int64),
        n_components=1,
        damping_lambda=0.0,
        gauge_weight=0.0,
        reference_node_id=None,
        min_total_observations_per_node=0,
        total_observation_count_by_node=np.ones(n_nodes, dtype=np.int64),
    )


def _sparse_result(**overrides: Any) -> TimeTermSparseSolverResult:
    used_mask = overrides.pop('used_trace_mask_sorted', USED_TRACE_MASK.copy())
    node_time_term = overrides.pop('node_time_term_s', NODE_TIME_TERM_S.copy())
    estimated_delay = overrides.pop(
        'estimated_trace_time_term_delay_s_sorted',
        ESTIMATED_DELAY_S.copy(),
    )
    row_trace_index = np.flatnonzero(used_mask).astype(np.int64, copy=False)
    n_observations = int(row_trace_index.shape[0])
    payload: dict[str, Any] = {
        'node_time_term_s': np.asarray(node_time_term, dtype=np.float64),
        'estimated_trace_time_term_delay_s_sorted': np.asarray(
            estimated_delay,
            dtype=np.float64,
        ),
        'row_estimated_time_term_delay_s': ESTIMATED_DELAY_S[row_trace_index],
        'row_residual_before_s': np.zeros(n_observations, dtype=np.float64),
        'row_residual_after_s': np.zeros(n_observations, dtype=np.float64),
        'row_residual_after_ms': np.zeros(n_observations, dtype=np.float64),
        'rms_residual_before_s': 0.0,
        'rms_residual_after_s': 0.0,
        'used_trace_mask_sorted': np.asarray(used_mask, dtype=bool),
        'row_trace_index_sorted': row_trace_index,
        'solver_name': 'lsmr',
        'solver_istop': 1,
        'solver_iterations': 1,
        'solver_normr': 0.0,
        'solver_normar': 0.0,
        'solver_conda': 1.0,
        'solver_message': 'synthetic',
        'system': _system(n_observations=n_observations),
    }
    payload.update(overrides)
    return TimeTermSparseSolverResult(**payload)


def _robust_result(**overrides: Any) -> TimeTermRobustSolverResult:
    final_solver_result = overrides.pop('final_solver_result', _sparse_result())
    final_used_mask = overrides.pop(
        'final_used_trace_mask_sorted',
        np.asarray([False, True, False], dtype=bool),
    )
    rejected_mask = overrides.pop(
        'final_rejected_trace_mask_sorted',
        np.asarray([True, False, False], dtype=bool),
    )
    rejected_iteration = overrides.pop(
        'rejected_iteration_sorted',
        np.asarray([0, -1, -1], dtype=np.int64),
    )
    payload: dict[str, Any] = {
        'final_solver_result': final_solver_result,
        'iterations': (),
        'initial_used_trace_mask_sorted': USED_TRACE_MASK.copy(),
        'final_used_trace_mask_sorted': final_used_mask,
        'final_rejected_trace_mask_sorted': rejected_mask,
        'rejected_iteration_sorted': rejected_iteration,
        'initial_row_used_mask': np.ones(2, dtype=bool),
        'final_row_used_mask': np.ones(1, dtype=bool),
        'final_row_rejected_mask': np.asarray([True, False], dtype=bool),
        'row_rejected_iteration': np.asarray([0, -1], dtype=np.int64),
        'method': 'mad',
        'enabled': True,
        'stop_reason': 'converged',
        'robust_options': TimeTermRobustSolverOptions(),
        'n_initial_used_traces': int(np.count_nonzero(USED_TRACE_MASK)),
        'n_final_used_traces': int(np.count_nonzero(final_used_mask)),
        'n_rejected_traces': int(np.count_nonzero(rejected_mask)),
        'final_used_fraction': 0.5,
    }
    payload.update(overrides)
    return TimeTermRobustSolverResult(**payload)


def _build(
    *,
    inputs: TimeTermInversionInputs | None = None,
    solver_result: TimeTermSparseSolverResult | TimeTermRobustSolverResult | None = None,
    options: TimeTermAppliedShiftOptions | None = None,
) -> TimeTermAppliedShiftResult:
    return build_time_term_applied_shift_result(
        inputs=_inputs() if inputs is None else inputs,
        solver_result=_sparse_result() if solver_result is None else solver_result,
        options=options,
    )


def test_delay_to_applied_shift_negates_delay() -> None:
    delay = np.asarray([0.010, -0.002, 0.0], dtype=np.float64)

    np.testing.assert_allclose(delay_to_applied_shift_s(delay), [-0.010, 0.002, -0.0])


def test_time_term_applied_shift_from_sparse_solver_result() -> None:
    result = _build()

    assert result.n_traces == N_TRACES
    assert result.dt == DT
    np.testing.assert_array_equal(result.final_used_trace_mask_sorted, USED_TRACE_MASK)
    np.testing.assert_array_equal(
        result.rejected_trace_mask_sorted,
        np.zeros(N_TRACES, dtype=bool),
    )
    np.testing.assert_array_equal(
        result.rejected_iteration_sorted,
        np.full(N_TRACES, -1, dtype=np.int64),
    )


def test_time_term_applied_shift_from_robust_solver_result_uses_final_solver_result() -> None:
    final_solver = _sparse_result()
    robust_result = _robust_result(final_solver_result=final_solver)

    result = _build(solver_result=robust_result)

    np.testing.assert_allclose(result.node_time_term_s, NODE_TIME_TERM_S)
    np.testing.assert_array_equal(
        result.final_used_trace_mask_sorted,
        [False, True, False],
    )
    np.testing.assert_array_equal(result.rejected_trace_mask_sorted, [True, False, False])
    np.testing.assert_array_equal(result.rejected_iteration_sorted, [0, -1, -1])
    assert result.metadata['solver_result_kind'] == 'robust'


def test_time_term_applied_shift_recomputes_trace_delay_from_node_terms() -> None:
    result = _build()

    np.testing.assert_allclose(
        result.source_node_time_term_s_sorted,
        [0.010, 0.005, -0.002],
    )
    np.testing.assert_allclose(
        result.receiver_node_time_term_s_sorted,
        [0.005, -0.002, -0.002],
    )
    np.testing.assert_allclose(
        result.estimated_trace_time_term_delay_s_sorted,
        ESTIMATED_DELAY_S,
    )


def test_time_term_applied_shift_rejects_solver_trace_delay_mapping_mismatch() -> None:
    solver_result = _sparse_result(
        estimated_trace_time_term_delay_s_sorted=ESTIMATED_DELAY_S
        + np.asarray([0.001, 0.0, 0.0], dtype=np.float64)
    )

    with pytest.raises(
        ValueError,
        match='estimated_trace_time_term_delay_s_sorted does not match node_time_term_s mapping',
    ):
        _build(solver_result=solver_result)


def test_time_term_applied_shift_is_negative_estimated_delay() -> None:
    result = _build()

    np.testing.assert_allclose(
        result.applied_weathering_shift_s_sorted,
        -result.estimated_trace_time_term_delay_s_sorted,
    )
    np.testing.assert_allclose(
        result.applied_weathering_shift_s_sorted,
        APPLIED_WEATHERING_SHIFT_S,
    )


def test_time_term_applied_shift_composes_final_shift_with_datum_and_residual() -> None:
    result = _build()

    np.testing.assert_allclose(result.final_trace_shift_s_sorted, FINAL_SHIFT_S)
    np.testing.assert_allclose(
        result.final_trace_shift_s_sorted,
        DATUM_SHIFT_S + RESIDUAL_SHIFT_S + APPLIED_WEATHERING_SHIFT_S,
    )


def test_time_term_applied_shift_keeps_sorted_trace_order() -> None:
    result = _build()

    assert result.order == ORDER
    np.testing.assert_allclose(
        result.estimated_trace_time_term_delay_s_sorted,
        [0.015, 0.003, -0.004],
    )
    np.testing.assert_allclose(
        result.final_trace_shift_s_sorted,
        [-0.034, -0.021, -0.013],
    )


def test_time_term_applied_shift_uses_final_model_for_rejected_traces_by_default() -> None:
    result = _build(solver_result=_robust_result())

    assert not bool(result.final_used_trace_mask_sorted[0])
    assert bool(result.rejected_trace_mask_sorted[0])
    assert result.rejected_iteration_sorted[0] == 0
    assert result.applied_weathering_shift_s_sorted[0] == pytest.approx(-0.015)


def test_time_term_applied_shift_rejects_unsupported_rejected_trace_policy() -> None:
    with pytest.raises(ValueError, match='unsupported rejected_trace_policy'):
        _build(
            options=TimeTermAppliedShiftOptions(
                rejected_trace_policy='zero_rejected',  # type: ignore[arg-type]
            )
        )


def test_time_term_applied_shift_rejects_non_finite_node_time_terms() -> None:
    node_time_term = NODE_TIME_TERM_S.copy()
    node_time_term[0] = np.nan

    with pytest.raises(ValueError, match='node_time_term_s'):
        _build(solver_result=_sparse_result(node_time_term_s=node_time_term))


def test_time_term_applied_shift_rejects_node_id_out_of_range() -> None:
    receiver_node_id = RECEIVER_NODE_ID_SORTED.copy()
    receiver_node_id[1] = N_NODES

    with pytest.raises(ValueError, match='receiver_node_id_sorted'):
        _build(inputs=_inputs(receiver_node_id_sorted=receiver_node_id))


def test_time_term_applied_shift_rejects_datum_shift_shape_mismatch() -> None:
    with pytest.raises(ValueError, match='datum_trace_shift_s_sorted'):
        _build(inputs=_inputs(datum_trace_shift_s_sorted=np.zeros(2, dtype=np.float64)))


def test_time_term_applied_shift_rejects_residual_shift_shape_mismatch() -> None:
    with pytest.raises(ValueError, match='residual_applied_shift_s_sorted'):
        _build(
            inputs=_inputs(
                residual_applied_shift_s_sorted=np.zeros(2, dtype=np.float64)
            )
        )


def test_time_term_applied_shift_rejects_weathering_shift_above_limit() -> None:
    with pytest.raises(ValueError, match='max_abs_weathering_shift_ms exceeded'):
        _build(options=TimeTermAppliedShiftOptions(max_abs_weathering_shift_ms=10.0))


def test_time_term_applied_shift_rejects_final_shift_above_limit() -> None:
    with pytest.raises(ValueError, match='max_abs_final_shift_ms exceeded'):
        _build(
            options=TimeTermAppliedShiftOptions(
                max_abs_weathering_shift_ms=None,
                max_abs_final_shift_ms=30.0,
            )
        )


def test_time_term_applied_shift_metadata_contains_sign_convention() -> None:
    result = _build()

    assert result.sign_convention == SIGN_CONVENTION
    assert result.metadata['delay_to_shift_convention'] == DELAY_TO_SHIFT_CONVENTION
    assert result.metadata['final_shift_convention'] == FINAL_SHIFT_CONVENTION
    assert result.metadata['rejected_trace_policy'] == 'use_final_model'


def test_time_term_applied_shift_result_does_not_confuse_delay_and_shift_names() -> None:
    field_names = {field.name for field in fields(TimeTermAppliedShiftResult)}

    assert 'estimated_trace_time_term_delay_s_sorted' in field_names
    assert 'applied_weathering_shift_s_sorted' in field_names
    assert not any(
        'estimated' in field_name and 'shift' in field_name
        for field_name in field_names
    )
    assert not any(
        'applied' in field_name and 'delay' in field_name
        for field_name in field_names
    )


def test_summarize_time_term_applied_shift_result_is_json_safe() -> None:
    result = _build(solver_result=_robust_result())

    summary = summarize_time_term_applied_shift_result(result)

    json.dumps(summary, allow_nan=False)
    assert summary['n_traces'] == N_TRACES
    assert summary['dt'] == DT
    assert summary['n_valid_picks'] == 2
    assert summary['n_final_used_traces'] == 1
    assert summary['n_rejected_traces'] == 1
    assert summary['estimated_trace_time_term_delay_ms']['count'] == N_TRACES
    assert summary['applied_weathering_shift_ms']['max_abs'] == pytest.approx(15.0)
    assert summary['final_trace_shift_ms']['max_abs'] == pytest.approx(34.0)
    assert summary['max_abs_weathering_shift_ms'] == pytest.approx(15.0)
    assert summary['max_abs_final_shift_ms'] == pytest.approx(34.0)
    assert summary['rejected_trace_policy'] == 'use_final_model'
