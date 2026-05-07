from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import sparse

import app.services.refraction_static_solver as solver_module
from app.api.schemas import RefractionStaticModelRequest, RefractionStaticSolverRequest
from app.services.refraction_static_design_matrix import (
    RefractionStaticDesignMatrix,
    build_refraction_static_design_matrix_from_arrays,
)
from app.services.refraction_static_solver import (
    RefractionStaticSolverError,
    solve_refraction_static_bounded_ls,
    solve_refraction_static_bounded_ls_from_matrix,
)

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


def _invalid_model(**overrides: Any) -> RefractionStaticModelRequest:
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
    return RefractionStaticModelRequest.model_construct(**payload)


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
    velocity_m_s: float = TRUE_BEDROCK_VELOCITY_M_S,
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
    return source_time + receiver_time + ROW_DISTANCE_M / float(velocity_m_s)


def _design_matrix_solve_global(
    *,
    pick_time_s: np.ndarray | None = None,
    node_id: np.ndarray | None = None,
) -> RefractionStaticDesignMatrix:
    pick_time = _pick_times() if pick_time_s is None else np.asarray(pick_time_s)
    return build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(pick_time.shape[0], dtype=bool),
        source_node_id_sorted=ROW_SOURCE_NODE_ID,
        receiver_node_id_sorted=ROW_RECEIVER_NODE_ID,
        distance_m_sorted=ROW_DISTANCE_M,
        node_id=ACTIVE_NODE_ID if node_id is None else node_id,
        bedrock_velocity_mode='solve_global',
        n_traces=int(pick_time.shape[0]),
    )


def _design_matrix_fixed_global(
    *,
    pick_time_s: np.ndarray | None = None,
    velocity_m_s: float = TRUE_BEDROCK_VELOCITY_M_S,
) -> RefractionStaticDesignMatrix:
    pick_time = _pick_times(velocity_m_s=velocity_m_s) if pick_time_s is None else pick_time_s
    return build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray(pick_time, dtype=np.float64),
        valid_observation_mask_sorted=np.ones(ROW_DISTANCE_M.shape[0], dtype=bool),
        source_node_id_sorted=ROW_SOURCE_NODE_ID,
        receiver_node_id_sorted=ROW_RECEIVER_NODE_ID,
        distance_m_sorted=ROW_DISTANCE_M,
        node_id=ACTIVE_NODE_ID,
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=velocity_m_s,
        n_traces=int(ROW_DISTANCE_M.shape[0]),
    )


def test_solve_global_recovers_parameters_residuals_and_qc() -> None:
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
    assert result.qc['bedrock_velocity_mode'] == 'solve_global'
    assert result.qc['residual_rms_ms'] == pytest.approx(0.0, abs=1.0e-6)
    assert result.qc['half_intercept_time_clipped_lower_count'] == 0
    assert result.qc['solver_status'] == 'success'


def test_fixed_global_solves_without_slowness_column_and_models_original_pick_time() -> None:
    design = _design_matrix_fixed_global()

    result = solve_refraction_static_bounded_ls(
        design_matrix=design,
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
    np.testing.assert_allclose(result.modeled_pick_time_s, design.observed_pick_time_s)
    np.testing.assert_allclose(result.residual_time_s, 0.0, atol=1.0e-9)
    assert result.qc['fixed_bedrock_velocity_m_s'] == TRUE_BEDROCK_VELOCITY_M_S


def test_half_intercept_bounds_clip_lower_and_upper() -> None:
    matrix = sparse.eye(4, dtype=np.float64, format='csr')
    rhs = np.asarray([-0.010, 0.200, 0.015, 0.012], dtype=np.float64)
    distance = np.full(4, 100.0, dtype=np.float64)
    fixed_velocity = TRUE_BEDROCK_VELOCITY_M_S
    observed = rhs + distance / fixed_velocity

    result = solve_refraction_static_bounded_ls_from_matrix(
        matrix=matrix,
        rhs_s=rhs,
        active_node_id=ACTIVE_NODE_ID,
        bedrock_slowness_col=None,
        row_distance_m=distance,
        observed_pick_time_s=observed,
        model=_model(
            bedrock_velocity_mode='fixed_global',
            bedrock_velocity_m_s=fixed_velocity,
        ),
        solver=_solver(max_abs_half_intercept_time_ms=50.0),
    )

    assert result.parameter_vector[0] == pytest.approx(0.0, abs=1.0e-9)
    assert result.parameter_vector[1] == pytest.approx(0.050, abs=1.0e-9)
    assert result.node_solution_status[0] == 'clipped_lower'
    assert result.node_solution_status[1] == 'clipped_upper'
    assert result.qc['half_intercept_time_clipped_lower_count'] == 1
    assert result.qc['half_intercept_time_clipped_upper_count'] == 1


def test_solve_global_slowness_bounds_clip_to_velocity_bound() -> None:
    pick_time = _pick_times(velocity_m_s=900.0)
    design = _design_matrix_solve_global(pick_time_s=pick_time)

    result = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_model(
            weathering_velocity_m_s=500.0,
            min_bedrock_velocity_m_s=1200.0,
            max_bedrock_velocity_m_s=6000.0,
            initial_bedrock_velocity_m_s=2500.0,
        ),
        solver=_solver(),
    )

    assert result.bedrock_slowness_s_per_m == pytest.approx(1.0 / 1200.0)
    assert result.bedrock_velocity_m_s == pytest.approx(1200.0)
    assert result.qc['bedrock_slowness_clipped'] is True
    assert result.qc['bedrock_slowness_clipped_upper'] is True


def test_robust_disabled_uses_all_rows_with_bad_pick() -> None:
    pick_time = _pick_times()
    pick_time[0] += 0.100
    design = _design_matrix_solve_global(pick_time_s=pick_time)

    result = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_model(),
        solver=_solver(robust={'enabled': False}),
    )

    assert result.used_row_mask.all()
    assert not result.rejected_by_robust_mask.any()
    assert result.robust_iteration_count == 0


def test_robust_mad_rejects_large_bad_pick_and_refits() -> None:
    pick_time = _pick_times()
    pick_time[0] += 0.100
    design = _design_matrix_solve_global(pick_time_s=pick_time)

    result = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_model(),
        solver=_solver(robust={'enabled': True, 'method': 'mad'}),
    )

    assert result.rejected_by_robust_mask[0]
    assert not result.used_row_mask[0]
    assert result.robust_iteration_count == 1
    np.testing.assert_allclose(result.residual_time_s[1:], 0.0, atol=1.0e-8)
    assert result.qc['n_rejected_by_robust'] == 1


def test_robust_sigma_rejects_large_bad_pick() -> None:
    pick_time = _pick_times()
    pick_time[0] += 0.100
    design = _design_matrix_solve_global(pick_time_s=pick_time)

    result = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_model(),
        solver=_solver(
            robust={'enabled': True, 'method': 'sigma', 'threshold': 2.5},
        ),
    )

    assert result.rejected_by_robust_mask[0]
    assert result.robust_iteration_count == 1
    assert result.qc['robust_method'] == 'sigma'


def test_robust_rejection_raises_when_minimum_fraction_would_be_violated() -> None:
    pick_time = _pick_times()
    pick_time[0] += 0.100
    design = _design_matrix_solve_global(pick_time_s=pick_time)

    with pytest.raises(RefractionStaticSolverError, match='too few refraction'):
        solve_refraction_static_bounded_ls(
            design_matrix=design,
            model=_model(),
            solver=_solver(
                robust={
                    'enabled': True,
                    'method': 'mad',
                    'min_used_fraction': 1.0,
                },
            ),
        )


def test_robust_rejection_rejects_mask_that_orphans_active_node_column() -> None:
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    rhs = np.asarray([0.500, 0.010, 0.020, 0.030, 0.010], dtype=np.float64)
    distance = np.full(rhs.shape, 100.0, dtype=np.float64)
    fixed_velocity = TRUE_BEDROCK_VELOCITY_M_S
    observed = rhs + distance / fixed_velocity

    with pytest.raises(RefractionStaticSolverError, match='active-node column'):
        solve_refraction_static_bounded_ls_from_matrix(
            matrix=matrix,
            rhs_s=rhs,
            active_node_id=np.asarray([10, 20, 30], dtype=np.int64),
            bedrock_slowness_col=None,
            row_distance_m=distance,
            observed_pick_time_s=observed,
            model=_model(
                bedrock_velocity_mode='fixed_global',
                bedrock_velocity_m_s=fixed_velocity,
            ),
            solver=_solver(
                max_abs_half_intercept_time_ms=50.0,
                robust={
                    'enabled': True,
                    'method': 'sigma',
                    'threshold': 1.5,
                },
            ),
        )


def test_damping_rows_apply_only_to_half_intercept_columns() -> None:
    design = _design_matrix_solve_global()

    no_damping = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_model(),
        solver=_solver(damping=0.0),
    )
    damped = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=_model(),
        solver=_solver(damping=1.0e-8),
    )

    assert no_damping.qc['n_damping_rows'] == 0
    assert damped.qc['n_damping_rows'] == ACTIVE_NODE_ID.shape[0]
    assert damped.qc['n_augmented_rows'] == design.n_observations + design.n_active_nodes
    assert damped.qc['damping_applied_to'] == 'half_intercept_time_columns'
    assert damped.bedrock_slowness_s_per_m == pytest.approx(
        TRUE_BEDROCK_SLOWNESS_S_PER_M,
        rel=1.0e-4,
    )


@pytest.mark.parametrize(
    ('mutator', 'match'),
    [
        (lambda design: design.matrix.tocsc(), 'CSR'),
        (lambda design: design.rhs_s[:-1], 'rhs_s shape mismatch'),
        (
            lambda design: np.concatenate((design.rhs_s[:-1], [np.inf])),
            'rhs_s must contain only finite',
        ),
    ],
)
def test_validation_rejects_matrix_and_rhs_errors(mutator: Any, match: str) -> None:
    design = _design_matrix_solve_global()
    matrix = design.matrix
    rhs = design.rhs_s
    value = mutator(design)
    if sparse.issparse(value):
        matrix = value
    else:
        rhs = value

    with pytest.raises(RefractionStaticSolverError, match=match):
        solve_refraction_static_bounded_ls_from_matrix(
            matrix=matrix,
            rhs_s=rhs,
            active_node_id=design.active_node_id,
            bedrock_slowness_col=design.bedrock_slowness_col,
            row_distance_m=design.row_distance_m,
            observed_pick_time_s=design.observed_pick_time_s,
            model=_model(),
            solver=_solver(),
        )


def test_validation_rejects_non_finite_matrix_value() -> None:
    design = _design_matrix_solve_global()
    matrix = design.matrix.copy()
    matrix.data[0] = np.inf

    with pytest.raises(RefractionStaticSolverError, match='matrix values'):
        solve_refraction_static_bounded_ls(
            design_matrix=RefractionStaticDesignMatrix(
                **{**design.__dict__, 'matrix': matrix}
            ),
            model=_model(),
            solver=_solver(),
        )


def test_validation_rejects_no_observations() -> None:
    with pytest.raises(RefractionStaticSolverError, match='at least one'):
        solve_refraction_static_bounded_ls_from_matrix(
            matrix=sparse.csr_matrix((0, 5), dtype=np.float64),
            rhs_s=np.empty(0, dtype=np.float64),
            active_node_id=ACTIVE_NODE_ID,
            bedrock_slowness_col=4,
            row_distance_m=np.empty(0, dtype=np.float64),
            observed_pick_time_s=np.empty(0, dtype=np.float64),
            model=_model(),
            solver=_solver(),
        )


def test_validation_rejects_all_zero_row_and_active_column() -> None:
    design = _design_matrix_solve_global()
    zero_row = design.matrix.copy().tolil()
    zero_row[0, :] = 0.0
    zero_row = zero_row.tocsr()
    zero_row.eliminate_zeros()

    with pytest.raises(RefractionStaticSolverError, match='all-zero row'):
        solve_refraction_static_bounded_ls_from_matrix(
            matrix=zero_row,
            rhs_s=design.rhs_s,
            active_node_id=design.active_node_id,
            bedrock_slowness_col=design.bedrock_slowness_col,
            row_distance_m=design.row_distance_m,
            observed_pick_time_s=design.observed_pick_time_s,
            model=_model(),
            solver=_solver(),
        )

    zero_col = design.matrix.copy().tolil()
    zero_col[:, 0] = 0.0
    zero_col = zero_col.tocsr()
    zero_col.eliminate_zeros()
    with pytest.raises(RefractionStaticSolverError, match='active-node column'):
        solve_refraction_static_bounded_ls_from_matrix(
            matrix=zero_col,
            rhs_s=design.rhs_s,
            active_node_id=design.active_node_id,
            bedrock_slowness_col=design.bedrock_slowness_col,
            row_distance_m=design.row_distance_m,
            observed_pick_time_s=design.observed_pick_time_s,
            model=_model(),
            solver=_solver(),
        )


@pytest.mark.parametrize(
    ('model', 'bedrock_slowness_col', 'match'),
    [
        (_model(), None, 'requires a bedrock slowness column'),
        (
            _invalid_model(method='other'),
            4,
            'model.method must be gli_variable_thickness',
        ),
        (
            _invalid_model(bedrock_velocity_mode='other'),
            4,
            'bedrock_velocity_mode',
        ),
        (
            _invalid_model(min_bedrock_velocity_m_s=6000.0),
            4,
            'min_bedrock_velocity',
        ),
    ],
)
def test_validation_rejects_model_contract_errors(
    model: RefractionStaticModelRequest,
    bedrock_slowness_col: int | None,
    match: str,
) -> None:
    design = _design_matrix_solve_global()

    with pytest.raises(RefractionStaticSolverError, match=match):
        solve_refraction_static_bounded_ls_from_matrix(
            matrix=design.matrix,
            rhs_s=design.rhs_s,
            active_node_id=design.active_node_id,
            bedrock_slowness_col=bedrock_slowness_col,
            row_distance_m=design.row_distance_m,
            observed_pick_time_s=design.observed_pick_time_s,
            model=model,
            solver=_solver(),
        )


def test_validation_rejects_fixed_global_velocity_errors_and_slowness_column() -> None:
    design = _design_matrix_fixed_global()

    with pytest.raises(RefractionStaticSolverError, match='required'):
        solve_refraction_static_bounded_ls_from_matrix(
            matrix=design.matrix,
            rhs_s=design.rhs_s,
            active_node_id=design.active_node_id,
            bedrock_slowness_col=None,
            row_distance_m=design.row_distance_m,
            observed_pick_time_s=design.observed_pick_time_s,
            model=_invalid_model(
                bedrock_velocity_mode='fixed_global',
                bedrock_velocity_m_s=None,
            ),
            solver=_solver(),
        )

    with pytest.raises(RefractionStaticSolverError, match='weathering'):
        solve_refraction_static_bounded_ls_from_matrix(
            matrix=design.matrix,
            rhs_s=design.rhs_s,
            active_node_id=design.active_node_id,
            bedrock_slowness_col=None,
            row_distance_m=design.row_distance_m,
            observed_pick_time_s=design.observed_pick_time_s,
            model=_invalid_model(
                bedrock_velocity_mode='fixed_global',
                bedrock_velocity_m_s=700.0,
            ),
            solver=_solver(),
        )

    with pytest.raises(RefractionStaticSolverError, match='must not include'):
        solve_refraction_static_bounded_ls_from_matrix(
            matrix=design.matrix,
            rhs_s=design.rhs_s,
            active_node_id=design.active_node_id,
            bedrock_slowness_col=4,
            row_distance_m=design.row_distance_m,
            observed_pick_time_s=design.observed_pick_time_s,
            model=_model(
                bedrock_velocity_mode='fixed_global',
                bedrock_velocity_m_s=TRUE_BEDROCK_VELOCITY_M_S,
            ),
            solver=_solver(),
        )


def test_solver_rejects_non_finite_solver_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RawResult:
        success = True
        x = np.full(5, np.nan, dtype=np.float64)
        status = 1
        message = 'patched'
        cost = 0.0
        optimality = 0.0
        nit = 1

    monkeypatch.setattr(
        solver_module.optimize,
        'lsq_linear',
        lambda *args, **kwargs: _RawResult(),
    )
    design = _design_matrix_solve_global()

    with pytest.raises(RefractionStaticSolverError, match='parameter_vector'):
        solve_refraction_static_bounded_ls(
            design_matrix=design,
            model=_model(),
            solver=_solver(),
        )
