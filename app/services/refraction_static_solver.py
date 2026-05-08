"""Bounded least-squares solver for GLI refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import optimize, sparse

from app.api.schemas import RefractionStaticModelRequest, RefractionStaticSolverRequest
from app.services.refraction_static_types import (
    RefractionStaticDesignMatrix,
    RefractionStaticSolverResult,
)

BedrockVelocityMode = Literal['solve_global', 'fixed_global']
RobustMethod = Literal['mad', 'sigma']

_BOUND_TOL = 1.0e-10
_ROBUST_SCALE_FLOOR_S = 1.0e-12


class RefractionStaticSolverError(ValueError):
    """Raised when the bounded refraction static solve cannot be completed."""


@dataclass(frozen=True)
class _ValidatedProblem:
    matrix: sparse.csr_matrix
    rhs_s: np.ndarray
    active_node_id: np.ndarray
    inactive_node_id: np.ndarray
    bedrock_slowness_col: int | None
    row_distance_m: np.ndarray
    observed_pick_time_s: np.ndarray
    row_trace_index_sorted: np.ndarray
    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    mode: BedrockVelocityMode
    weathering_velocity_m_s: float
    min_bedrock_velocity_m_s: float
    max_bedrock_velocity_m_s: float
    fixed_bedrock_velocity_m_s: float | None
    fixed_bedrock_slowness_s_per_m: float | None
    damping: float
    max_abs_half_intercept_time_s: float
    min_picks_per_node: int
    robust_enabled: bool
    robust_method: RobustMethod
    robust_threshold: float
    robust_max_iterations: int
    robust_min_used_fraction: float
    robust_min_used_observations: int

    @property
    def n_observations(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_parameters(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def n_active_nodes(self) -> int:
        return int(self.active_node_id.shape[0])


@dataclass(frozen=True)
class _InternalSolveResult:
    parameter_vector: np.ndarray
    raw_status: int | None
    raw_message: str
    cost: float
    optimality: float | None
    nit: int | None
    n_damping_rows: int
    n_augmented_rows: int


def solve_refraction_static_bounded_ls(
    *,
    design_matrix: RefractionStaticDesignMatrix,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
) -> RefractionStaticSolverResult:
    """Solve a refraction static design matrix with physical bounds."""
    return solve_refraction_static_bounded_ls_from_matrix(
        matrix=design_matrix.matrix,
        rhs_s=design_matrix.rhs_s,
        active_node_id=design_matrix.active_node_id,
        inactive_node_id=design_matrix.inactive_node_id,
        bedrock_slowness_col=design_matrix.bedrock_slowness_col,
        row_distance_m=design_matrix.row_distance_m,
        observed_pick_time_s=design_matrix.observed_pick_time_s,
        row_trace_index_sorted=design_matrix.row_trace_index_sorted,
        row_source_node_id=design_matrix.row_source_node_id,
        row_receiver_node_id=design_matrix.row_receiver_node_id,
        bedrock_velocity_mode=design_matrix.bedrock_velocity_mode,
        fixed_bedrock_velocity_m_s=design_matrix.fixed_bedrock_velocity_m_s,
        fixed_bedrock_slowness_s_per_m=design_matrix.fixed_bedrock_slowness_s_per_m,
        model=model,
        solver=solver,
    )


def solve_refraction_static_bounded_ls_from_matrix(
    *,
    matrix: sparse.csr_matrix,
    rhs_s: np.ndarray,
    active_node_id: np.ndarray,
    bedrock_slowness_col: int | None,
    row_distance_m: np.ndarray,
    observed_pick_time_s: np.ndarray,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    row_trace_index_sorted: np.ndarray | None = None,
    row_source_node_id: np.ndarray | None = None,
    row_receiver_node_id: np.ndarray | None = None,
    inactive_node_id: np.ndarray | None = None,
    bedrock_velocity_mode: BedrockVelocityMode | None = None,
    fixed_bedrock_velocity_m_s: float | None = None,
    fixed_bedrock_slowness_s_per_m: float | None = None,
) -> RefractionStaticSolverResult:
    """Solve a bounded GLI system from already-built sparse arrays."""
    problem = _validate_problem(
        matrix=matrix,
        rhs_s=rhs_s,
        active_node_id=active_node_id,
        inactive_node_id=inactive_node_id,
        bedrock_slowness_col=bedrock_slowness_col,
        row_distance_m=row_distance_m,
        observed_pick_time_s=observed_pick_time_s,
        row_trace_index_sorted=row_trace_index_sorted,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        bedrock_velocity_mode=bedrock_velocity_mode,
        fixed_bedrock_velocity_m_s=fixed_bedrock_velocity_m_s,
        fixed_bedrock_slowness_s_per_m=fixed_bedrock_slowness_s_per_m,
        model=model,
        solver=solver,
    )
    lower_bounds, upper_bounds = _build_bounds(problem)
    solve_result, used_mask, rejected_mask, robust_iteration_count = (
        _solve_with_optional_robust_rejection(
            problem,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
    )
    return _build_result(
        problem=problem,
        solve_result=solve_result,
        used_mask=used_mask,
        rejected_mask=rejected_mask,
        robust_iteration_count=robust_iteration_count,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )


def _validate_problem(
    *,
    matrix: sparse.csr_matrix,
    rhs_s: np.ndarray,
    active_node_id: np.ndarray,
    inactive_node_id: np.ndarray | None,
    bedrock_slowness_col: int | None,
    row_distance_m: np.ndarray,
    observed_pick_time_s: np.ndarray,
    row_trace_index_sorted: np.ndarray | None,
    row_source_node_id: np.ndarray | None,
    row_receiver_node_id: np.ndarray | None,
    bedrock_velocity_mode: BedrockVelocityMode | None,
    fixed_bedrock_velocity_m_s: float | None,
    fixed_bedrock_slowness_s_per_m: float | None,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
) -> _ValidatedProblem:
    if getattr(model, 'method', None) != 'gli_variable_thickness':
        raise RefractionStaticSolverError(
            'model.method must be gli_variable_thickness'
        )
    mode = _validate_bedrock_velocity_mode(getattr(model, 'bedrock_velocity_mode', None))
    if bedrock_velocity_mode is not None and bedrock_velocity_mode != mode:
        raise RefractionStaticSolverError(
            'design matrix bedrock_velocity_mode does not match model'
        )

    if not sparse.isspmatrix_csr(matrix):
        raise RefractionStaticSolverError('refraction design matrix must be CSR')
    if not _is_real_numeric_dtype(matrix.dtype):
        raise RefractionStaticSolverError(
            'refraction design matrix dtype must be floating'
        )
    if not np.issubdtype(matrix.dtype, np.floating):
        raise RefractionStaticSolverError(
            'refraction design matrix dtype must be floating'
        )
    if np.any(~np.isfinite(matrix.data)):
        raise RefractionStaticSolverError(
            'refraction design matrix values must be finite'
        )
    n_observations, n_parameters = matrix.shape
    if n_observations <= 0:
        raise RefractionStaticSolverError(
            'at least one refraction observation is required'
        )
    if n_parameters <= 0:
        raise RefractionStaticSolverError(
            'at least one refraction model parameter is required'
        )

    rhs = _coerce_1d_real_numeric_float64(
        rhs_s,
        name='rhs_s',
        expected_shape=(n_observations,),
    )
    _validate_all_finite(rhs, name='rhs_s')
    observed_pick_time = _coerce_1d_real_numeric_float64(
        observed_pick_time_s,
        name='observed_pick_time_s',
        expected_shape=(n_observations,),
    )
    _validate_all_finite(observed_pick_time, name='observed_pick_time_s')
    distance = _coerce_1d_real_numeric_float64(
        row_distance_m,
        name='row_distance_m',
        expected_shape=(n_observations,),
    )
    _validate_all_finite(distance, name='row_distance_m')
    if np.any(distance <= 0.0):
        raise RefractionStaticSolverError('row_distance_m values must be positive')

    active_nodes = _coerce_1d_integer_int64(active_node_id, name='active_node_id')
    if active_nodes.size <= 0:
        raise RefractionStaticSolverError('at least one active node is required')
    if np.unique(active_nodes).shape[0] != active_nodes.shape[0]:
        raise RefractionStaticSolverError('active node IDs must be unique')
    inactive_nodes = (
        np.empty(0, dtype=np.int64)
        if inactive_node_id is None
        else _coerce_1d_integer_int64(inactive_node_id, name='inactive_node_id')
    )
    if inactive_nodes.size and np.intersect1d(active_nodes, inactive_nodes).size:
        raise RefractionStaticSolverError(
            'active and inactive node IDs must be disjoint'
        )

    n_active = int(active_nodes.shape[0])
    expected_parameters = n_active + (1 if mode == 'solve_global' else 0)
    if n_parameters != expected_parameters:
        raise RefractionStaticSolverError(
            'refraction design matrix parameter count does not match active nodes'
        )

    slowness_col = _validate_slowness_column(
        mode=mode,
        bedrock_slowness_col=bedrock_slowness_col,
        n_active_nodes=n_active,
        n_parameters=n_parameters,
    )
    _validate_matrix_structure(
        matrix,
        n_active_nodes=n_active,
        bedrock_slowness_col=slowness_col,
    )

    trace_index = (
        np.arange(n_observations, dtype=np.int64)
        if row_trace_index_sorted is None
        else _coerce_1d_integer_int64(
            row_trace_index_sorted,
            name='row_trace_index_sorted',
            expected_shape=(n_observations,),
        )
    )
    source_node = (
        np.full(n_observations, -1, dtype=np.int64)
        if row_source_node_id is None
        else _coerce_1d_integer_int64(
            row_source_node_id,
            name='row_source_node_id',
            expected_shape=(n_observations,),
        )
    )
    receiver_node = (
        np.full(n_observations, -1, dtype=np.int64)
        if row_receiver_node_id is None
        else _coerce_1d_integer_int64(
            row_receiver_node_id,
            name='row_receiver_node_id',
            expected_shape=(n_observations,),
        )
    )

    weathering_velocity = _coerce_positive_finite_float(
        getattr(model, 'weathering_velocity_m_s', None),
        name='model.weathering_velocity_m_s',
    )
    min_velocity = _coerce_positive_finite_float(
        getattr(model, 'min_bedrock_velocity_m_s', None),
        name='model.min_bedrock_velocity_m_s',
    )
    max_velocity = _coerce_positive_finite_float(
        getattr(model, 'max_bedrock_velocity_m_s', None),
        name='model.max_bedrock_velocity_m_s',
    )
    if min_velocity >= max_velocity:
        raise RefractionStaticSolverError(
            'model.min_bedrock_velocity_m_s must be less than '
            'model.max_bedrock_velocity_m_s'
        )
    if min_velocity <= weathering_velocity:
        raise RefractionStaticSolverError(
            'model.min_bedrock_velocity_m_s must be greater than '
            'model.weathering_velocity_m_s'
        )
    if max_velocity <= weathering_velocity:
        raise RefractionStaticSolverError(
            'model.max_bedrock_velocity_m_s must be greater than '
            'model.weathering_velocity_m_s'
        )
    initial_velocity = getattr(model, 'initial_bedrock_velocity_m_s', None)
    if initial_velocity is not None:
        initial_velocity = _coerce_positive_finite_float(
            initial_velocity,
            name='model.initial_bedrock_velocity_m_s',
        )
        if initial_velocity <= weathering_velocity:
            raise RefractionStaticSolverError(
                'model.initial_bedrock_velocity_m_s must be greater than '
                'model.weathering_velocity_m_s'
            )
        if not min_velocity <= initial_velocity <= max_velocity:
            raise RefractionStaticSolverError(
                'model.initial_bedrock_velocity_m_s must be within '
                'bedrock velocity bounds'
            )

    fixed_velocity, fixed_slowness = _validate_fixed_velocity(
        mode=mode,
        model=model,
        weathering_velocity=weathering_velocity,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
        fixed_bedrock_velocity_m_s=fixed_bedrock_velocity_m_s,
        fixed_bedrock_slowness_s_per_m=fixed_bedrock_slowness_s_per_m,
    )

    damping = _coerce_nonnegative_finite_float(
        getattr(solver, 'damping', None),
        name='solver.damping',
    )
    max_half_intercept_time_s = (
        _coerce_positive_finite_float(
            getattr(solver, 'max_abs_half_intercept_time_ms', None),
            name='solver.max_abs_half_intercept_time_ms',
        )
        / 1000.0
    )
    min_picks_per_node = _coerce_positive_int(
        getattr(solver, 'min_picks_per_node', None),
        name='solver.min_picks_per_node',
    )
    robust = getattr(solver, 'robust', None)
    if robust is None:
        raise RefractionStaticSolverError('solver.robust is required')
    robust_enabled = _coerce_bool(
        getattr(robust, 'enabled', None),
        name='solver.robust.enabled',
    )
    robust_method = _validate_robust_method(getattr(robust, 'method', None))
    robust_threshold = _coerce_positive_finite_float(
        getattr(robust, 'threshold', None),
        name='solver.robust.threshold',
    )
    robust_max_iterations = _coerce_positive_int(
        getattr(robust, 'max_iterations', None),
        name='solver.robust.max_iterations',
    )
    robust_min_used_fraction = _coerce_fraction(
        getattr(robust, 'min_used_fraction', None),
        name='solver.robust.min_used_fraction',
    )
    robust_min_used_observations = _coerce_positive_int(
        getattr(robust, 'min_used_observations', None),
        name='solver.robust.min_used_observations',
    )

    return _ValidatedProblem(
        matrix=matrix.astype(np.float64, copy=False),
        rhs_s=rhs,
        active_node_id=active_nodes,
        inactive_node_id=inactive_nodes,
        bedrock_slowness_col=slowness_col,
        row_distance_m=distance,
        observed_pick_time_s=observed_pick_time,
        row_trace_index_sorted=trace_index,
        row_source_node_id=source_node,
        row_receiver_node_id=receiver_node,
        mode=mode,
        weathering_velocity_m_s=weathering_velocity,
        min_bedrock_velocity_m_s=min_velocity,
        max_bedrock_velocity_m_s=max_velocity,
        fixed_bedrock_velocity_m_s=fixed_velocity,
        fixed_bedrock_slowness_s_per_m=fixed_slowness,
        damping=damping,
        max_abs_half_intercept_time_s=max_half_intercept_time_s,
        min_picks_per_node=min_picks_per_node,
        robust_enabled=robust_enabled,
        robust_method=robust_method,
        robust_threshold=robust_threshold,
        robust_max_iterations=robust_max_iterations,
        robust_min_used_fraction=robust_min_used_fraction,
        robust_min_used_observations=robust_min_used_observations,
    )


def _validate_slowness_column(
    *,
    mode: BedrockVelocityMode,
    bedrock_slowness_col: int | None,
    n_active_nodes: int,
    n_parameters: int,
) -> int | None:
    if mode == 'solve_global':
        if bedrock_slowness_col is None:
            raise RefractionStaticSolverError(
                'solve_global mode requires a bedrock slowness column'
            )
        if isinstance(bedrock_slowness_col, (bool, np.bool_)):
            raise RefractionStaticSolverError('bedrock_slowness_col must be an integer')
        col = int(bedrock_slowness_col)
        if col < 0 or col >= n_parameters:
            raise RefractionStaticSolverError('bedrock_slowness_col is out of range')
        if col != n_active_nodes:
            raise RefractionStaticSolverError(
                'bedrock_slowness_col must follow active node columns'
            )
        return col
    if bedrock_slowness_col is not None:
        raise RefractionStaticSolverError(
            'fixed_global mode must not include a bedrock slowness column'
        )
    return None


def _validate_matrix_structure(
    matrix: sparse.csr_matrix,
    *,
    n_active_nodes: int,
    bedrock_slowness_col: int | None,
) -> None:
    row_abs_sum = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
    if np.any(row_abs_sum == 0.0):
        raise RefractionStaticSolverError(
            'refraction design matrix contains an all-zero row'
        )
    col_abs_sum = np.asarray(np.abs(matrix).sum(axis=0)).ravel()
    zero_active_cols = np.flatnonzero(col_abs_sum[:n_active_nodes] == 0.0)
    if zero_active_cols.size:
        raise RefractionStaticSolverError(
            'refraction design matrix contains an all-zero active-node column'
        )
    if (
        bedrock_slowness_col is not None
        and col_abs_sum[int(bedrock_slowness_col)] == 0.0
    ):
        raise RefractionStaticSolverError(
            'refraction design matrix contains an all-zero bedrock slowness column'
        )


def _validate_fixed_velocity(
    *,
    mode: BedrockVelocityMode,
    model: RefractionStaticModelRequest,
    weathering_velocity: float,
    min_velocity: float,
    max_velocity: float,
    fixed_bedrock_velocity_m_s: float | None,
    fixed_bedrock_slowness_s_per_m: float | None,
) -> tuple[float | None, float | None]:
    if mode == 'solve_global':
        if getattr(model, 'bedrock_velocity_m_s', None) is not None:
            raise RefractionStaticSolverError(
                'model.bedrock_velocity_m_s is only allowed for fixed_global mode'
            )
        if (
            fixed_bedrock_velocity_m_s is not None
            or fixed_bedrock_slowness_s_per_m is not None
        ):
            raise RefractionStaticSolverError(
                'fixed bedrock velocity metadata is only allowed for fixed_global mode'
            )
        return None, None

    model_velocity = getattr(model, 'bedrock_velocity_m_s', None)
    if model_velocity is None:
        raise RefractionStaticSolverError(
            'model.bedrock_velocity_m_s is required for fixed_global mode'
        )
    velocity = _coerce_positive_finite_float(
        model_velocity,
        name='model.bedrock_velocity_m_s',
    )
    if velocity <= weathering_velocity:
        raise RefractionStaticSolverError(
            'model.bedrock_velocity_m_s must be greater than '
            'model.weathering_velocity_m_s'
        )
    if not min_velocity <= velocity <= max_velocity:
        raise RefractionStaticSolverError(
            'model.bedrock_velocity_m_s must be within bedrock velocity bounds'
        )
    if fixed_bedrock_velocity_m_s is not None and not np.isclose(
        float(fixed_bedrock_velocity_m_s),
        velocity,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RefractionStaticSolverError(
            'design matrix fixed bedrock velocity does not match model'
        )
    slowness = 1.0 / velocity
    if fixed_bedrock_slowness_s_per_m is not None and not np.isclose(
        float(fixed_bedrock_slowness_s_per_m),
        slowness,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise RefractionStaticSolverError(
            'design matrix fixed bedrock slowness does not match model'
        )
    return velocity, slowness


def _build_bounds(problem: _ValidatedProblem) -> tuple[np.ndarray, np.ndarray]:
    lower = np.zeros(problem.n_parameters, dtype=np.float64)
    upper = np.full(
        problem.n_parameters,
        problem.max_abs_half_intercept_time_s,
        dtype=np.float64,
    )
    if problem.mode == 'solve_global':
        if problem.bedrock_slowness_col is None:
            raise RefractionStaticSolverError(
                'solve_global mode requires a bedrock slowness column'
            )
        lower_slowness = 1.0 / problem.max_bedrock_velocity_m_s
        upper_slowness = 1.0 / problem.min_bedrock_velocity_m_s
        lower[problem.bedrock_slowness_col] = lower_slowness
        upper[problem.bedrock_slowness_col] = upper_slowness
    if lower.shape != (problem.n_parameters,) or upper.shape != (problem.n_parameters,):
        raise RefractionStaticSolverError('bounds shape mismatch')
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise RefractionStaticSolverError('bounds must be finite')
    if np.any(lower >= upper):
        raise RefractionStaticSolverError(
            'lower bounds must be less than upper bounds'
        )
    return (
        np.ascontiguousarray(lower, dtype=np.float64),
        np.ascontiguousarray(upper, dtype=np.float64),
    )


def _solve_with_optional_robust_rejection(
    problem: _ValidatedProblem,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[_InternalSolveResult, np.ndarray, np.ndarray, int]:
    used_mask = np.ones(problem.n_observations, dtype=bool)
    _validate_used_observation_count(
        problem,
        used_mask=used_mask,
        require_fraction=problem.robust_enabled,
        message='Too few refraction observations for a stable GLI solve.',
    )

    if not problem.robust_enabled:
        solve_result = _solve_once(
            problem,
            used_mask=used_mask,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        rejected_mask = np.zeros(problem.n_observations, dtype=bool)
        return solve_result, used_mask, rejected_mask, 0

    rejected_mask = np.zeros(problem.n_observations, dtype=bool)
    robust_iteration_count = 0
    final_solve: _InternalSolveResult | None = None

    for _iteration_index in range(problem.robust_max_iterations):
        solve_result = _solve_once(
            problem,
            used_mask=used_mask,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        residual = _compute_original_residual(problem, solve_result.parameter_vector)
        outlier_local = _build_robust_outlier_mask(
            residual[used_mask],
            method=problem.robust_method,
            threshold=problem.robust_threshold,
        )
        if not np.any(outlier_local):
            final_solve = solve_result
            break

        used_indices = np.flatnonzero(used_mask)
        newly_rejected = used_indices[outlier_local]
        proposed_used_mask = used_mask.copy()
        proposed_used_mask[newly_rejected] = False
        _validate_used_observation_count(
            problem,
            used_mask=proposed_used_mask,
            require_fraction=True,
            message=(
                'Robust rejection would leave too few refraction observations '
                'for a stable GLI solve.'
            ),
        )

        used_mask = proposed_used_mask
        rejected_mask[newly_rejected] = True
        robust_iteration_count += 1
    else:
        final_solve = _solve_once(
            problem,
            used_mask=used_mask,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    if final_solve is None:
        raise RefractionStaticSolverError('robust solver did not produce a result')
    return (
        final_solve,
        np.ascontiguousarray(used_mask, dtype=bool),
        np.ascontiguousarray(rejected_mask, dtype=bool),
        robust_iteration_count,
    )


def _solve_once(
    problem: _ValidatedProblem,
    *,
    used_mask: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> _InternalSolveResult:
    _validate_used_observation_count(
        problem,
        used_mask=used_mask,
        require_fraction=False,
        message='Too few refraction observations for a stable GLI solve.',
    )
    matrix_used = problem.matrix[used_mask, :]
    _validate_matrix_structure(
        matrix_used,
        n_active_nodes=problem.n_active_nodes,
        bedrock_slowness_col=problem.bedrock_slowness_col,
    )
    rhs_used = problem.rhs_s[used_mask]
    matrix_aug, rhs_aug, n_damping_rows = _augment_with_damping(
        matrix_used,
        rhs_used,
        n_active_nodes=problem.n_active_nodes,
        damping=problem.damping,
    )

    # SciPy 1.14 ``lsq_linear`` does not accept ``x0``; bounds define the feasible
    # region and TRF computes its own strictly feasible starting point.
    raw = optimize.lsq_linear(
        matrix_aug,
        rhs_aug,
        bounds=(lower_bounds, upper_bounds),
        method='trf',
        lsq_solver='lsmr',
        tol=1.0e-12,
        lsmr_tol=1.0e-12,
        lsmr_maxiter=max(matrix_aug.shape) * 20,
    )
    if not bool(getattr(raw, 'success', False)):
        raise RefractionStaticSolverError(
            f"refraction static bounded solver failed: {getattr(raw, 'message', '')}"
        )
    parameter_vector = np.ascontiguousarray(raw.x, dtype=np.float64)
    if parameter_vector.shape != (problem.n_parameters,):
        raise RefractionStaticSolverError('solver parameter vector shape mismatch')
    _validate_all_finite(parameter_vector, name='solver parameter_vector')
    return _InternalSolveResult(
        parameter_vector=parameter_vector,
        raw_status=_optional_int(getattr(raw, 'status', None)),
        raw_message=str(getattr(raw, 'message', '')),
        cost=_coerce_finite_float(getattr(raw, 'cost', 0.0), name='solver_cost'),
        optimality=_optional_finite_float(getattr(raw, 'optimality', None)),
        nit=_optional_int(getattr(raw, 'nit', None)),
        n_damping_rows=n_damping_rows,
        n_augmented_rows=int(matrix_aug.shape[0]),
    )


def _augment_with_damping(
    matrix: sparse.csr_matrix,
    rhs_s: np.ndarray,
    *,
    n_active_nodes: int,
    damping: float,
) -> tuple[sparse.csr_matrix, np.ndarray, int]:
    if damping == 0.0:
        return matrix, rhs_s, 0
    weight = float(np.sqrt(damping))
    damping_matrix = sparse.eye(
        n_active_nodes,
        matrix.shape[1],
        dtype=np.float64,
        format='csr',
    )
    damping_matrix = damping_matrix * weight
    matrix_aug = sparse.vstack((matrix, damping_matrix), format='csr')
    rhs_aug = np.concatenate(
        (rhs_s, np.zeros(n_active_nodes, dtype=np.float64)),
    )
    return matrix_aug, np.ascontiguousarray(rhs_aug, dtype=np.float64), n_active_nodes


def _build_result(
    *,
    problem: _ValidatedProblem,
    solve_result: _InternalSolveResult,
    used_mask: np.ndarray,
    rejected_mask: np.ndarray,
    robust_iteration_count: int,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> RefractionStaticSolverResult:
    parameter_vector = np.ascontiguousarray(
        solve_result.parameter_vector,
        dtype=np.float64,
    )
    modeled_pick_time = _compute_original_modeled_pick_time(problem, parameter_vector)
    residual = np.ascontiguousarray(
        problem.observed_pick_time_s - modeled_pick_time,
        dtype=np.float64,
    )
    _validate_all_finite(modeled_pick_time, name='modeled_pick_time_s')
    _validate_all_finite(residual, name='residual_time_s')

    active_half_intercept = np.ascontiguousarray(
        parameter_vector[: problem.n_active_nodes],
        dtype=np.float64,
    )
    active_status = _build_active_node_status(
        active_half_intercept,
        lower_bounds=lower_bounds[: problem.n_active_nodes],
        upper_bounds=upper_bounds[: problem.n_active_nodes],
    )
    node_id = np.concatenate((problem.active_node_id, problem.inactive_node_id))
    node_half_intercept = np.concatenate(
        (
            active_half_intercept,
            np.zeros(problem.inactive_node_id.shape[0], dtype=np.float64),
        )
    )
    node_status = np.concatenate(
        (
            active_status,
            np.full(problem.inactive_node_id.shape[0], 'inactive', dtype='<U16'),
        )
    )

    bedrock_slowness, bedrock_velocity = _extract_bedrock_solution(
        problem,
        parameter_vector,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    qc = _build_qc(
        problem=problem,
        solve_result=solve_result,
        active_half_intercept=active_half_intercept,
        active_status=active_status,
        residual=residual,
        used_mask=used_mask,
        rejected_mask=rejected_mask,
        robust_iteration_count=robust_iteration_count,
        bedrock_slowness=bedrock_slowness,
        bedrock_velocity=bedrock_velocity,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    return RefractionStaticSolverResult(
        parameter_vector=parameter_vector,
        active_node_id=np.ascontiguousarray(problem.active_node_id, dtype=np.int64),
        active_node_half_intercept_time_s=active_half_intercept,
        node_id=np.ascontiguousarray(node_id, dtype=np.int64),
        node_half_intercept_time_s=np.ascontiguousarray(
            node_half_intercept,
            dtype=np.float64,
        ),
        node_solution_status=np.ascontiguousarray(node_status, dtype='<U16'),
        bedrock_velocity_mode=problem.mode,
        bedrock_slowness_s_per_m=bedrock_slowness,
        bedrock_velocity_m_s=bedrock_velocity,
        row_trace_index_sorted=np.ascontiguousarray(
            problem.row_trace_index_sorted,
            dtype=np.int64,
        ),
        row_source_node_id=np.ascontiguousarray(
            problem.row_source_node_id,
            dtype=np.int64,
        ),
        row_receiver_node_id=np.ascontiguousarray(
            problem.row_receiver_node_id,
            dtype=np.int64,
        ),
        row_distance_m=np.ascontiguousarray(problem.row_distance_m, dtype=np.float64),
        observed_pick_time_s=np.ascontiguousarray(
            problem.observed_pick_time_s,
            dtype=np.float64,
        ),
        modeled_pick_time_s=modeled_pick_time,
        residual_time_s=residual,
        used_row_mask=np.ascontiguousarray(used_mask, dtype=bool),
        rejected_by_robust_mask=np.ascontiguousarray(rejected_mask, dtype=bool),
        solver_status='success',
        solver_message=solve_result.raw_message,
        solver_cost=solve_result.cost,
        solver_optimality=solve_result.optimality,
        solver_nit=solve_result.nit,
        robust_iteration_count=int(robust_iteration_count),
        lower_bounds=np.ascontiguousarray(lower_bounds, dtype=np.float64),
        upper_bounds=np.ascontiguousarray(upper_bounds, dtype=np.float64),
        qc=qc,
    )


def _compute_original_modeled_pick_time(
    problem: _ValidatedProblem,
    parameter_vector: np.ndarray,
) -> np.ndarray:
    modeled = np.asarray(problem.matrix @ parameter_vector, dtype=np.float64)
    if problem.mode == 'fixed_global':
        if problem.fixed_bedrock_slowness_s_per_m is None:
            raise RefractionStaticSolverError(
                'fixed_global mode requires fixed bedrock slowness'
            )
        modeled = modeled + problem.row_distance_m * problem.fixed_bedrock_slowness_s_per_m
    return np.ascontiguousarray(modeled, dtype=np.float64)


def _compute_original_residual(
    problem: _ValidatedProblem,
    parameter_vector: np.ndarray,
) -> np.ndarray:
    modeled = _compute_original_modeled_pick_time(problem, parameter_vector)
    return np.ascontiguousarray(
        problem.observed_pick_time_s - modeled,
        dtype=np.float64,
    )


def _extract_bedrock_solution(
    problem: _ValidatedProblem,
    parameter_vector: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[float, float]:
    if problem.mode == 'fixed_global':
        if (
            problem.fixed_bedrock_slowness_s_per_m is None
            or problem.fixed_bedrock_velocity_m_s is None
        ):
            raise RefractionStaticSolverError(
                'fixed_global mode requires fixed bedrock velocity'
            )
        return (
            float(problem.fixed_bedrock_slowness_s_per_m),
            float(problem.fixed_bedrock_velocity_m_s),
        )
    if problem.bedrock_slowness_col is None:
        raise RefractionStaticSolverError(
            'solve_global mode requires a bedrock slowness column'
        )
    slowness = float(parameter_vector[problem.bedrock_slowness_col])
    if not np.isfinite(slowness) or slowness <= 0.0:
        raise RefractionStaticSolverError('computed bedrock slowness must be positive')
    velocity = float(1.0 / slowness)
    if not np.isfinite(velocity):
        raise RefractionStaticSolverError('computed bedrock velocity must be finite')
    if not (
        problem.min_bedrock_velocity_m_s - 1.0e-8
        <= velocity
        <= problem.max_bedrock_velocity_m_s + 1.0e-8
    ):
        raise RefractionStaticSolverError(
            'computed bedrock velocity is outside configured bounds'
        )
    if velocity <= problem.weathering_velocity_m_s:
        raise RefractionStaticSolverError(
            'computed bedrock velocity must be greater than weathering velocity'
        )
    if slowness < lower_bounds[problem.bedrock_slowness_col] - 1.0e-12:
        raise RefractionStaticSolverError(
            'computed bedrock slowness is below configured bounds'
        )
    if slowness > upper_bounds[problem.bedrock_slowness_col] + 1.0e-12:
        raise RefractionStaticSolverError(
            'computed bedrock slowness is above configured bounds'
        )
    return slowness, velocity


def _build_active_node_status(
    values: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    status = np.full(values.shape, 'solved', dtype='<U16')
    status[values <= lower_bounds + _BOUND_TOL] = 'clipped_lower'
    status[values >= upper_bounds - _BOUND_TOL] = 'clipped_upper'
    return status


def _build_qc(
    *,
    problem: _ValidatedProblem,
    solve_result: _InternalSolveResult,
    active_half_intercept: np.ndarray,
    active_status: np.ndarray,
    residual: np.ndarray,
    used_mask: np.ndarray,
    rejected_mask: np.ndarray,
    robust_iteration_count: int,
    bedrock_slowness: float,
    bedrock_velocity: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> dict[str, Any]:
    residual_stats = _residual_stats_ms(residual)
    half_ms = active_half_intercept * 1000.0
    slowness_clipped_lower = False
    slowness_clipped_upper = False
    if problem.mode == 'solve_global' and problem.bedrock_slowness_col is not None:
        slowness_col = problem.bedrock_slowness_col
        slowness_clipped_lower = (
            bedrock_slowness <= lower_bounds[slowness_col] + _BOUND_TOL
        )
        slowness_clipped_upper = (
            bedrock_slowness >= upper_bounds[slowness_col] - _BOUND_TOL
        )
    qc: dict[str, Any] = {
        'method': 'gli_variable_thickness',
        'bedrock_velocity_mode': problem.mode,
        'n_observations': int(problem.n_observations),
        'n_used_observations': int(np.count_nonzero(used_mask)),
        'n_rejected_by_robust': int(np.count_nonzero(rejected_mask)),
        'used_fraction': float(np.count_nonzero(used_mask) / problem.n_observations),
        'n_active_nodes': int(problem.n_active_nodes),
        'n_parameters': int(problem.n_parameters),
        'bedrock_velocity_m_s': float(bedrock_velocity),
        'bedrock_slowness_s_per_m': float(bedrock_slowness),
        'weathering_velocity_m_s': float(problem.weathering_velocity_m_s),
        'min_bedrock_velocity_m_s': float(problem.min_bedrock_velocity_m_s),
        'max_bedrock_velocity_m_s': float(problem.max_bedrock_velocity_m_s),
        'bedrock_slowness_clipped': bool(
            slowness_clipped_lower or slowness_clipped_upper
        ),
        'bedrock_slowness_clipped_lower': bool(slowness_clipped_lower),
        'bedrock_slowness_clipped_upper': bool(slowness_clipped_upper),
        'half_intercept_time_min_ms': float(np.min(half_ms)),
        'half_intercept_time_max_ms': float(np.max(half_ms)),
        'half_intercept_time_median_ms': float(np.median(half_ms)),
        'half_intercept_time_clipped_lower_count': int(
            np.count_nonzero(active_status == 'clipped_lower')
        ),
        'half_intercept_time_clipped_upper_count': int(
            np.count_nonzero(active_status == 'clipped_upper')
        ),
        **residual_stats,
        'solver_status': 'success',
        'solver_message': solve_result.raw_message,
        'solver_cost': float(solve_result.cost),
        'solver_optimality': _json_optional_float(solve_result.optimality),
        'solver_nit': solve_result.nit,
        'solver_raw_status': solve_result.raw_status,
        'robust_enabled': bool(problem.robust_enabled),
        'robust_method': problem.robust_method,
        'robust_threshold': float(problem.robust_threshold),
        'robust_iteration_count': int(robust_iteration_count),
        'damping': float(problem.damping),
        'min_picks_per_node': int(problem.min_picks_per_node),
        'max_abs_half_intercept_time_ms': float(
            problem.max_abs_half_intercept_time_s * 1000.0
        ),
        'damping_applied_to': (
            'half_intercept_time_columns'
            if solve_result.n_damping_rows
            else 'none'
        ),
        'n_damping_rows': int(solve_result.n_damping_rows),
        'n_augmented_rows': int(solve_result.n_augmented_rows),
    }
    if problem.mode == 'fixed_global':
        qc.update(
            {
                'fixed_bedrock_velocity_m_s': float(bedrock_velocity),
                'fixed_bedrock_slowness_s_per_m': float(bedrock_slowness),
            }
        )
    return qc


def _residual_stats_ms(residual_s: np.ndarray) -> dict[str, float]:
    residual_ms = np.ascontiguousarray(residual_s * 1000.0, dtype=np.float64)
    median = float(np.median(residual_ms))
    return {
        'residual_rms_ms': float(np.sqrt(np.mean(residual_ms * residual_ms))),
        'residual_mad_ms': float(1.4826 * np.median(np.abs(residual_ms - median))),
        'residual_mean_ms': float(np.mean(residual_ms)),
        'residual_median_ms': median,
        'residual_p95_abs_ms': float(np.percentile(np.abs(residual_ms), 95.0)),
        'residual_max_abs_ms': float(np.max(np.abs(residual_ms))),
    }


def _build_robust_outlier_mask(
    residual_s: np.ndarray,
    *,
    method: RobustMethod,
    threshold: float,
) -> np.ndarray:
    if residual_s.size == 0:
        return np.zeros(0, dtype=bool)
    if method == 'mad':
        center = float(np.median(residual_s))
        scale = float(1.4826 * np.median(np.abs(residual_s - center)))
    elif method == 'sigma':
        center = float(np.mean(residual_s))
        scale = float(np.std(residual_s, ddof=0))
    else:
        raise RefractionStaticSolverError('robust method must be mad or sigma')
    if scale <= _ROBUST_SCALE_FLOOR_S:
        return np.zeros(residual_s.shape, dtype=bool)
    outlier = np.abs(residual_s - center) > float(threshold) * scale
    return np.ascontiguousarray(outlier, dtype=bool)


def _validate_used_observation_count(
    problem: _ValidatedProblem,
    *,
    used_mask: np.ndarray,
    require_fraction: bool,
    message: str,
) -> None:
    n_used = int(np.count_nonzero(used_mask))
    if n_used < problem.robust_min_used_observations:
        raise RefractionStaticSolverError(message)
    if require_fraction:
        used_fraction = n_used / problem.n_observations
        if used_fraction < problem.robust_min_used_fraction:
            raise RefractionStaticSolverError(message)


def _validate_bedrock_velocity_mode(value: object) -> BedrockVelocityMode:
    if value == 'solve_global':
        return 'solve_global'
    if value == 'fixed_global':
        return 'fixed_global'
    raise RefractionStaticSolverError(
        'model.bedrock_velocity_mode must be solve_global or fixed_global'
    )


def _validate_robust_method(value: object) -> RobustMethod:
    if value == 'mad':
        return 'mad'
    if value == 'sigma':
        return 'sigma'
    raise RefractionStaticSolverError('solver.robust.method must be mad or sigma')


def _coerce_1d_real_numeric_float64(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionStaticSolverError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise RefractionStaticSolverError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if not _is_real_numeric_dtype(arr.dtype):
        raise RefractionStaticSolverError(f'{name} must have a numeric dtype')
    return np.ascontiguousarray(arr, dtype=np.float64)


def _coerce_1d_integer_int64(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionStaticSolverError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise RefractionStaticSolverError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if np.issubdtype(arr.dtype, np.bool_):
        raise RefractionStaticSolverError(f'{name} must contain integer values')
    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr, dtype=np.int64)
    if not _is_real_numeric_dtype(arr.dtype):
        raise RefractionStaticSolverError(f'{name} must contain integer values')
    arr_f64 = arr.astype(np.float64, copy=False)
    _validate_all_finite(arr_f64, name=name)
    if not np.all(arr_f64 == np.rint(arr_f64)):
        raise RefractionStaticSolverError(f'{name} must contain integer values')
    return np.ascontiguousarray(arr_f64, dtype=np.int64)


def _coerce_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise RefractionStaticSolverError(f'{name} must be a bool')
    return bool(value)


def _coerce_fraction(value: object, *, name: str) -> float:
    out = _coerce_positive_finite_float(value, name=name)
    if out > 1.0:
        raise RefractionStaticSolverError(f'{name} must be <= 1')
    return out


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise RefractionStaticSolverError(f'{name} must be an integer')
    out = int(value)
    if out <= 0:
        raise RefractionStaticSolverError(f'{name} must be greater than 0')
    return out


def _coerce_positive_finite_float(value: object, *, name: str) -> float:
    out = _coerce_finite_float(value, name=name)
    if out <= 0.0:
        raise RefractionStaticSolverError(f'{name} must be greater than 0')
    return out


def _coerce_nonnegative_finite_float(value: object, *, name: str) -> float:
    out = _coerce_finite_float(value, name=name)
    if out < 0.0:
        raise RefractionStaticSolverError(f'{name} must be non-negative')
    return out


def _coerce_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise RefractionStaticSolverError(f'{name} must be finite')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticSolverError(f'{name} must be finite') from exc
    if not np.isfinite(out):
        raise RefractionStaticSolverError(f'{name} must be finite')
    return out


def _optional_finite_float(value: object) -> float | None:
    if value is None:
        return None
    return _coerce_finite_float(value, name='optional float')


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise RefractionStaticSolverError('optional int must be an integer')
    return int(value)


def _json_optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _validate_all_finite(values: np.ndarray, *, name: str) -> None:
    if np.any(~np.isfinite(values)):
        raise RefractionStaticSolverError(f'{name} must contain only finite values')


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype,
        np.complexfloating,
    )


__all__ = [
    'RefractionStaticSolverError',
    'RefractionStaticSolverResult',
    'solve_refraction_static_bounded_ls',
    'solve_refraction_static_bounded_ls_from_matrix',
]
