from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import app.services.refraction_static_solver as solver_module
from app.api.schemas import RefractionStaticModelRequest, RefractionStaticSolverRequest
from app.services.refraction_static_design_matrix import (
    build_refraction_static_design_matrix_from_arrays,
)
from app.services.refraction_static_solver import solve_refraction_static_bounded_ls
from app.services.refraction_static_types import RefractionStaticDesignMatrix

ACTIVE_NODE_ID = np.asarray([10, 20, 30, 40], dtype=np.int64)
TRUE_HALF_INTERCEPT_S = np.asarray([0.010, 0.020, 0.015, 0.012], dtype=np.float64)
TRUE_CELL_BEDROCK_VELOCITY_M_S = np.asarray([2000.0, 3000.0], dtype=np.float64)
TRUE_CELL_BEDROCK_SLOWNESS_S_PER_M = 1.0 / TRUE_CELL_BEDROCK_VELOCITY_M_S
ROW_SOURCE_NODE_ID = np.asarray(
    [10, 10, 10, 30, 30, 40, 10, 10, 20, 20, 30, 20],
    dtype=np.int64,
)
ROW_RECEIVER_NODE_ID = np.asarray(
    [20, 20, 10, 40, 40, 40, 30, 30, 40, 40, 30, 20],
    dtype=np.int64,
)
ROW_DISTANCE_M = np.asarray(
    [100.0, 220.0, 80.0, 150.0, 310.0, 180.0, 120.0, 260.0, 90.0, 190.0, 130.0, 210.0],
    dtype=np.float64,
)
ROW_MIDPOINT_CELL_ID = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)


def _cell_model(**overrides: Any) -> RefractionStaticModelRequest:
    payload: dict[str, Any] = {
        'method': 'gli_variable_thickness',
        'weathering_velocity_m_s': 800.0,
        'bedrock_velocity_mode': 'solve_cell',
        'bedrock_velocity_m_s': None,
        'initial_bedrock_velocity_m_s': 2400.0,
        'min_bedrock_velocity_m_s': 1200.0,
        'max_bedrock_velocity_m_s': 6000.0,
        'max_weathering_thickness_m': None,
        'refractor_cell': {
            'number_of_cell_x': 3,
            'size_of_cell_x_m': 10.0,
            'x_coordinate_origin_m': 0.0,
            'number_of_cell_y': 1,
            'size_of_cell_y_m': None,
            'y_coordinate_origin_m': 0.0,
            'assignment_mode': 'midpoint',
            'outside_grid_policy': 'reject',
            'min_observations_per_cell': 1,
            'velocity_smoothing_weight': 0.0,
            'smoothing_reference_distance_m': None,
        },
    }
    payload.update(overrides)
    return RefractionStaticModelRequest.model_validate(payload)


def _cell_model_with_smoothing(
    *,
    velocity_smoothing_weight: float,
    smoothing_reference_distance_m: float | None = None,
) -> RefractionStaticModelRequest:
    payload = _cell_model().model_dump()
    refractor_cell = dict(payload['refractor_cell'])
    refractor_cell['velocity_smoothing_weight'] = velocity_smoothing_weight
    refractor_cell['smoothing_reference_distance_m'] = smoothing_reference_distance_m
    payload['refractor_cell'] = refractor_cell
    return RefractionStaticModelRequest.model_validate(payload)


def _global_model(**overrides: Any) -> RefractionStaticModelRequest:
    payload: dict[str, Any] = {
        'method': 'gli_variable_thickness',
        'weathering_velocity_m_s': 800.0,
        'bedrock_velocity_mode': 'solve_global',
        'bedrock_velocity_m_s': None,
        'initial_bedrock_velocity_m_s': 2400.0,
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


def _pick_times(
    *,
    cell_velocity_m_s: np.ndarray = TRUE_CELL_BEDROCK_VELOCITY_M_S,
) -> np.ndarray:
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
    row_velocity = cell_velocity_m_s[ROW_MIDPOINT_CELL_ID]
    return source_time + receiver_time + ROW_DISTANCE_M / row_velocity


def _cell_design(
    *,
    pick_time_s: np.ndarray | None = None,
) -> RefractionStaticDesignMatrix:
    pick_time = _pick_times() if pick_time_s is None else np.asarray(pick_time_s)
    return build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(pick_time.shape[0], dtype=bool),
        source_node_id_sorted=ROW_SOURCE_NODE_ID,
        receiver_node_id_sorted=ROW_RECEIVER_NODE_ID,
        distance_m_sorted=ROW_DISTANCE_M,
        node_id=ACTIVE_NODE_ID,
        bedrock_velocity_mode='solve_cell',
        n_traces=int(pick_time.shape[0]),
        midpoint_cell_id_sorted=ROW_MIDPOINT_CELL_ID,
        n_total_cells=3,
        cell_assignment_mode='midpoint',
    )


def test_cell_solver_recovers_two_cell_velocities_without_noise() -> None:
    design = _cell_design()

    result = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_cell_model(),
        solver=_solver(),
    )

    np.testing.assert_allclose(
        result.active_node_half_intercept_time_s,
        TRUE_HALF_INTERCEPT_S,
        atol=1.0e-8,
    )
    np.testing.assert_array_equal(result.active_cell_id, [0, 1])
    np.testing.assert_array_equal(result.inactive_cell_id, [2])
    np.testing.assert_allclose(
        result.cell_bedrock_slowness_s_per_m,
        TRUE_CELL_BEDROCK_SLOWNESS_S_PER_M,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        result.cell_bedrock_velocity_m_s,
        TRUE_CELL_BEDROCK_VELOCITY_M_S,
        rtol=1.0e-7,
    )
    np.testing.assert_array_equal(result.cell_velocity_status, ['solved', 'solved'])
    np.testing.assert_allclose(result.modeled_pick_time_s, design.observed_pick_time_s)
    np.testing.assert_allclose(result.residual_time_s, 0.0, atol=1.0e-8)
    np.testing.assert_array_equal(result.row_midpoint_cell_id, ROW_MIDPOINT_CELL_ID)
    np.testing.assert_allclose(
        result.row_midpoint_bedrock_velocity_m_s,
        TRUE_CELL_BEDROCK_VELOCITY_M_S[ROW_MIDPOINT_CELL_ID],
    )


def test_cell_solver_respects_cell_slowness_bounds() -> None:
    pick_time = _pick_times(
        cell_velocity_m_s=np.asarray([900.0, 8000.0], dtype=np.float64)
    )

    result = solve_refraction_static_bounded_ls(
        design_matrix=_cell_design(pick_time_s=pick_time),
        model=_cell_model(
            weathering_velocity_m_s=500.0,
            min_bedrock_velocity_m_s=1200.0,
            max_bedrock_velocity_m_s=6000.0,
            initial_bedrock_velocity_m_s=2400.0,
        ),
        solver=_solver(),
    )

    np.testing.assert_allclose(
        result.cell_bedrock_slowness_s_per_m,
        [1.0 / 1200.0, 1.0 / 6000.0],
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        result.cell_bedrock_velocity_m_s,
        [1200.0, 6000.0],
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(
        result.cell_velocity_status,
        ['clipped_lower', 'clipped_upper'],
    )


def test_cell_solver_uses_initial_velocity_for_all_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, np.ndarray] = {}
    original = solver_module.optimize.least_squares

    def wrapped_least_squares(fun: Any, x0: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        captured['x0'] = np.ascontiguousarray(x0, dtype=np.float64)
        return original(fun, x0, *args, **kwargs)

    monkeypatch.setattr(solver_module.optimize, 'least_squares', wrapped_least_squares)
    design = _cell_design()

    solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_cell_model(initial_bedrock_velocity_m_s=1800.0),
        solver=_solver(),
    )

    assert design.bedrock_slowness_cell_col_start is not None
    cell_cols = np.arange(
        design.bedrock_slowness_cell_col_start,
        design.n_parameters,
        dtype=np.int64,
    )
    np.testing.assert_allclose(captured['x0'][cell_cols], 1.0 / 1800.0)


def test_cell_solver_robust_rejects_outlier_rows() -> None:
    pick_time = _pick_times()
    pick_time[0] += 0.200

    result = solve_refraction_static_bounded_ls(
        design_matrix=_cell_design(pick_time_s=pick_time),
        model=_cell_model(),
        solver=_solver(
            robust={
                'enabled': True,
                'method': 'mad',
                'threshold': 3.0,
            },
        ),
    )

    assert result.rejected_by_robust_mask[0]
    assert not result.used_row_mask[0]
    assert result.robust_iteration_count == 1
    np.testing.assert_allclose(
        result.cell_bedrock_velocity_m_s,
        TRUE_CELL_BEDROCK_VELOCITY_M_S,
        rtol=1.0e-7,
    )
    assert result.qc['n_rejected_by_robust'] == 1


def test_cell_smoothing_does_not_enter_robust_rejection_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = _cell_design()
    captured_row_counts: list[int] = []

    def no_outliers(
        residual_s: np.ndarray,
        *,
        method: str,
        threshold: float,
        scale_floor_s: float,
    ) -> np.ndarray:
        del method, threshold, scale_floor_s
        captured_row_counts.append(int(residual_s.shape[0]))
        return np.zeros(residual_s.shape, dtype=bool)

    monkeypatch.setattr(
        solver_module,
        '_build_robust_outlier_mask',
        no_outliers,
    )

    result = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_cell_model_with_smoothing(
            velocity_smoothing_weight=1.0,
            smoothing_reference_distance_m=100.0,
        ),
        solver=_solver(robust={'enabled': True}),
    )

    assert captured_row_counts == [design.n_observations]
    assert result.rejected_by_robust_mask.shape == (design.n_observations,)
    assert result.used_row_mask.shape == (design.n_observations,)
    assert result.qc['n_cell_smoothing_rows'] == 1
    assert result.qc['n_rejected_by_robust'] == 0


def test_cell_smoothing_reduces_velocity_contrast_in_noisy_synthetic() -> None:
    pick_time = _pick_times()
    pick_time = pick_time + np.where(ROW_MIDPOINT_CELL_ID == 0, 0.004, -0.004)
    design = _cell_design(pick_time_s=pick_time)

    unsmoothed = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_cell_model(),
        solver=_solver(),
    )
    smoothed = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_cell_model_with_smoothing(
            velocity_smoothing_weight=2.0,
            smoothing_reference_distance_m=100.0,
        ),
        solver=_solver(),
    )

    assert unsmoothed.cell_bedrock_velocity_m_s is not None
    assert smoothed.cell_bedrock_velocity_m_s is not None
    unsmoothed_contrast = float(np.ptp(unsmoothed.cell_bedrock_velocity_m_s))
    smoothed_contrast = float(np.ptp(smoothed.cell_bedrock_velocity_m_s))
    assert smoothed_contrast < unsmoothed_contrast
    assert smoothed.qc['n_cell_smoothing_rows'] == 1
    assert smoothed.qc['smoothing_row_scale'] == pytest.approx(200.0)


def test_cell_smoothing_zero_weight_reproduces_unsmoothed_solve() -> None:
    design = _cell_design()

    default_result = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_cell_model(),
        solver=_solver(),
    )
    zero_weight_result = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_cell_model_with_smoothing(
            velocity_smoothing_weight=0.0,
            smoothing_reference_distance_m=100.0,
        ),
        solver=_solver(),
    )

    np.testing.assert_allclose(
        zero_weight_result.parameter_vector,
        default_result.parameter_vector,
        atol=1.0e-10,
    )
    assert zero_weight_result.qc['n_cell_smoothing_rows'] == 0


def test_solve_cell_summary_velocity_is_reciprocal_of_summary_slowness() -> None:
    result = solve_refraction_static_bounded_ls(
        design_matrix=_cell_design(),
        model=_cell_model(),
        solver=_solver(),
    )

    expected_slowness = float(np.median(TRUE_CELL_BEDROCK_SLOWNESS_S_PER_M))

    assert result.bedrock_velocity_mode == 'solve_cell'
    assert result.bedrock_slowness_s_per_m == pytest.approx(expected_slowness)
    assert result.bedrock_velocity_m_s == pytest.approx(1.0 / expected_slowness)
    assert result.qc['bedrock_velocity_solution_kind'] == 'per_cell'
    assert result.qc['cell_bedrock_velocity_min_m_s'] == pytest.approx(2000.0)
    assert result.qc['cell_bedrock_velocity_median_m_s'] == pytest.approx(2500.0)
    assert result.qc['cell_bedrock_velocity_max_m_s'] == pytest.approx(3000.0)


def test_cell_solver_keeps_global_solver_regression_tests_passing() -> None:
    pick_time = _pick_times(cell_velocity_m_s=np.full(2, 2500.0, dtype=np.float64))
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(pick_time.shape[0], dtype=bool),
        source_node_id_sorted=ROW_SOURCE_NODE_ID,
        receiver_node_id_sorted=ROW_RECEIVER_NODE_ID,
        distance_m_sorted=ROW_DISTANCE_M,
        node_id=ACTIVE_NODE_ID,
        bedrock_velocity_mode='solve_global',
        n_traces=int(pick_time.shape[0]),
    )

    result = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_global_model(initial_bedrock_velocity_m_s=2500.0),
        solver=_solver(),
    )

    assert result.bedrock_velocity_mode == 'solve_global'
    assert result.bedrock_velocity_m_s == pytest.approx(2500.0, rel=1.0e-7)
    np.testing.assert_allclose(result.residual_time_s, 0.0, atol=1.0e-8)
