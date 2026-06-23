from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.api.schemas import RefractionStaticModelRequest, RefractionStaticSolverRequest
from app.statics.refraction.application.design_matrix import (
    build_refraction_static_design_matrix_from_arrays,
)
from app.statics.refraction.application.solver import (
    RefractionStaticSolverError,
    solve_refraction_static_bounded_ls,
)
from app.statics.refraction.contracts.result_types import ResolvedRefractionFirstLayer

ACTIVE_NODE_ID = np.asarray([10, 20, 30, 40], dtype=np.int64)
TRUE_HALF_INTERCEPT_S = np.asarray([0.010, 0.020, 0.015, 0.012], dtype=np.float64)
TRUE_BEDROCK_VELOCITY_M_S = 2500.0
TRUE_BEDROCK_SLOWNESS_S_PER_M = 1.0 / TRUE_BEDROCK_VELOCITY_M_S
ROW_SOURCE_NODE_ID = np.asarray([10, 10, 10, 20, 20, 30, 10, 20, 30, 40])
ROW_RECEIVER_NODE_ID = np.asarray([20, 30, 40, 30, 40, 40, 10, 20, 30, 40])
ROW_DISTANCE_M = np.asarray(
    [100.0, 240.0, 350.0, 160.0, 260.0, 120.0, 50.0, 70.0, 90.0, 110.0],
    dtype=np.float64,
)


def _model(**overrides: Any) -> RefractionStaticModelRequest:
    payload = {
        'method': 'gli_variable_thickness',
        'weathering_velocity_m_s': 800.0,
        'bedrock_velocity_mode': 'solve_global',
        'bedrock_velocity_m_s': None,
        'initial_bedrock_velocity_m_s': TRUE_BEDROCK_VELOCITY_M_S,
        'min_bedrock_velocity_m_s': 1200.0,
        'max_bedrock_velocity_m_s': 6000.0,
        'max_weathering_thickness_m': None,
    }
    payload.update(overrides)
    return RefractionStaticModelRequest.model_validate(payload)


def _solver(**overrides: Any) -> RefractionStaticSolverRequest:
    payload = {
        'damping': 0.0,
        'min_picks_per_node': 1,
        'max_abs_half_intercept_time_ms': 500.0,
        'robust': {
            'enabled': False,
            'method': 'mad',
            'threshold': 3.5,
            'max_iterations': 5,
            'min_used_fraction': 0.5,
            'min_used_observations': 1,
        },
    }
    robust_overrides = overrides.pop('robust', None)
    payload.update(overrides)
    if robust_overrides is not None:
        payload['robust'].update(robust_overrides)
    return RefractionStaticSolverRequest.model_validate(payload)


def _pick_times(*, velocity_m_s: float = TRUE_BEDROCK_VELOCITY_M_S) -> np.ndarray:
    node_to_index = {int(node): idx for idx, node in enumerate(ACTIVE_NODE_ID)}
    source_time = np.asarray(
        [TRUE_HALF_INTERCEPT_S[node_to_index[int(node)]] for node in ROW_SOURCE_NODE_ID],
        dtype=np.float64,
    )
    receiver_time = np.asarray(
        [
            TRUE_HALF_INTERCEPT_S[node_to_index[int(node)]]
            for node in ROW_RECEIVER_NODE_ID
        ],
        dtype=np.float64,
    )
    return source_time + receiver_time + ROW_DISTANCE_M / float(velocity_m_s)


def _design_matrix_solve_global(*, pick_time_s: np.ndarray | None = None):
    pick_time = _pick_times() if pick_time_s is None else np.asarray(pick_time_s)
    return build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(pick_time.shape[0], dtype=bool),
        source_node_id_sorted=ROW_SOURCE_NODE_ID,
        receiver_node_id_sorted=ROW_RECEIVER_NODE_ID,
        distance_m_sorted=ROW_DISTANCE_M,
        node_id=ACTIVE_NODE_ID,
        bedrock_velocity_mode='solve_global',
        n_traces=int(pick_time.shape[0]),
    )


def _design_matrix_solve_global_sparse_traces(*, pick_time_s: np.ndarray | None = None):
    row_pick_time = _pick_times() if pick_time_s is None else np.asarray(pick_time_s)
    trace_index = np.asarray([1, 3, 4, 6, 8, 9, 11, 12, 13, 14], dtype=np.int64)
    n_traces = 16
    pick_time = np.zeros(n_traces, dtype=np.float64)
    valid = np.zeros(n_traces, dtype=bool)
    source = np.zeros(n_traces, dtype=np.int64)
    receiver = np.zeros(n_traces, dtype=np.int64)
    distance = np.zeros(n_traces, dtype=np.float64)
    pick_time[trace_index] = row_pick_time
    valid[trace_index] = True
    source[trace_index] = ROW_SOURCE_NODE_ID
    receiver[trace_index] = ROW_RECEIVER_NODE_ID
    distance[trace_index] = ROW_DISTANCE_M
    return build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid,
        source_node_id_sorted=source,
        receiver_node_id_sorted=receiver,
        distance_m_sorted=distance,
        node_id=ACTIVE_NODE_ID,
        bedrock_velocity_mode='solve_global',
        n_traces=n_traces,
    )


def _design_matrix_fixed_global():
    pick_time = _pick_times()
    return build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(ROW_DISTANCE_M.shape[0], dtype=bool),
        source_node_id_sorted=ROW_SOURCE_NODE_ID,
        receiver_node_id_sorted=ROW_RECEIVER_NODE_ID,
        distance_m_sorted=ROW_DISTANCE_M,
        node_id=ACTIVE_NODE_ID,
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=TRUE_BEDROCK_VELOCITY_M_S,
        n_traces=int(ROW_DISTANCE_M.shape[0]),
    )


def test_solver_adapter_recovers_global_solution_and_contract_masks() -> None:
    design = _design_matrix_solve_global()

    result = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_model(),
        solver=_solver(),
    )

    np.testing.assert_allclose(
        result.active_node_half_intercept_time_s,
        TRUE_HALF_INTERCEPT_S,
        atol=1.0e-9,
    )
    assert result.bedrock_slowness_s_per_m == pytest.approx(
        TRUE_BEDROCK_SLOWNESS_S_PER_M,
        abs=1.0e-11,
    )
    assert result.bedrock_velocity_m_s == pytest.approx(
        TRUE_BEDROCK_VELOCITY_M_S,
        rel=1.0e-7,
    )
    np.testing.assert_allclose(result.modeled_pick_time_s, design.observed_pick_time_s)
    np.testing.assert_allclose(result.residual_time_s, 0.0, atol=1.0e-9)
    np.testing.assert_array_equal(result.row_trace_index_sorted, np.arange(10))
    assert result.used_row_mask.all()
    assert not result.rejected_by_robust_mask.any()
    assert result.qc['solver_status'] == 'success'


def test_solver_adapter_preserves_fixed_global_contract() -> None:
    result = solve_refraction_static_bounded_ls(
        design_matrix=_design_matrix_fixed_global(),
        model=_model(
            bedrock_velocity_mode='fixed_global',
            bedrock_velocity_m_s=TRUE_BEDROCK_VELOCITY_M_S,
        ),
        solver=_solver(),
    )

    assert result.parameter_vector.shape == (4,)
    np.testing.assert_allclose(
        result.active_node_half_intercept_time_s,
        TRUE_HALF_INTERCEPT_S,
        atol=1.0e-9,
    )
    assert result.bedrock_velocity_m_s == TRUE_BEDROCK_VELOCITY_M_S
    assert result.bedrock_slowness_s_per_m == TRUE_BEDROCK_SLOWNESS_S_PER_M
    assert result.qc['fixed_bedrock_velocity_m_s'] == TRUE_BEDROCK_VELOCITY_M_S


@pytest.mark.parametrize('method', ['mad', 'sigma'])
def test_solver_adapter_preserves_robust_used_and_rejected_masks(method: str) -> None:
    pick_time = _pick_times()
    pick_time[0] += 0.100

    result = solve_refraction_static_bounded_ls(
        design_matrix=_design_matrix_solve_global(pick_time_s=pick_time),
        model=_model(),
        solver=_solver(
            robust={'enabled': True, 'method': method, 'threshold': 2.5},
        ),
    )

    assert result.rejected_by_robust_mask[0]
    assert not result.used_row_mask[0]
    assert result.robust_iteration_count >= 1
    assert result.qc['robust_method'] == method
    assert result.qc['n_rejected_by_robust'] >= 1


def test_solver_adapter_maps_trace_indexed_core_masks_to_sparse_rows() -> None:
    pick_time = _pick_times()
    pick_time[0] += 0.100

    result = solve_refraction_static_bounded_ls(
        design_matrix=_design_matrix_solve_global_sparse_traces(pick_time_s=pick_time),
        model=_model(),
        solver=_solver(
            robust={'enabled': True, 'method': 'mad', 'threshold': 2.5},
        ),
    )

    np.testing.assert_array_equal(
        result.row_trace_index_sorted,
        [1, 3, 4, 6, 8, 9, 11, 12, 13, 14],
    )
    assert result.rejected_by_robust_mask.shape == result.row_trace_index_sorted.shape
    assert result.used_row_mask.shape == result.row_trace_index_sorted.shape
    assert result.rejected_by_robust_mask[0]
    assert not result.used_row_mask[0]
    assert result.qc['n_rejected_by_robust'] >= 1


def test_solver_adapter_uses_resolved_first_layer_for_velocity_bounds() -> None:
    model = _model(
        weathering_velocity_m_s=None,
        first_layer={
            'mode': 'estimate_direct_arrival',
            'min_direct_offset_m': 20.0,
            'max_direct_offset_m': 140.0,
        },
    )
    resolved_first_layer = ResolvedRefractionFirstLayer(
        mode='estimate_direct_arrival',
        weathering_velocity_m_s=1300.0,
        status='estimated',
        qc={},
    )

    with pytest.raises(
        RefractionStaticSolverError,
        match='model.min_bedrock_velocity_m_s must be greater than '
        'model.weathering_velocity_m_s',
    ):
        solve_refraction_static_bounded_ls(
            design_matrix=_design_matrix_solve_global(),
            model=model,
            solver=_solver(),
            resolved_first_layer=resolved_first_layer,
        )


def test_viewer_solver_module_has_no_scipy_optimization_implementation() -> None:
    source = Path('app/statics/refraction/application/solver.py').read_text(
        encoding='utf-8'
    )

    assert 'scipy.optimize' not in source
    assert 'lsq_linear' not in source
    assert 'def _solve_with_optional_robust_rejection' not in source
