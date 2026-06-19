"""Robust outlier rejection loop for source/receiver node time-term inversion."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
from scipy import sparse

from seis_statics._validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_integer_int64 as _common_coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_finite_float as _coerce_finite_float,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
    is_real_numeric_dtype as _is_real_numeric_dtype,
)
from seis_statics.time_term.design_matrix import TimeTermDesignMatrix
from seis_statics.time_term.sparse_solver import (
    TimeTermSparseSolverOptions,
    TimeTermSparseSolverResult,
    solve_time_term_sparse_least_squares,
)

TimeTermRobustMethod = Literal['mad', 'sigma']
TimeTermRobustStopReason = Literal[
    'disabled',
    'converged',
    'max_iterations',
    'zero_scale',
    'min_used_fraction',
    'min_used_observations',
    'no_new_rejections',
]

ROBUST_SCALE_FLOOR_S = 1.0e-12

_coerce_1d_integer_int64 = partial(
    _common_coerce_1d_integer_int64,
    allow_integer_like_float=False,
)


@dataclass(frozen=True)
class TimeTermRobustSolverOptions:
    enabled: bool = True
    method: TimeTermRobustMethod = 'mad'
    threshold: float = 3.5
    max_iterations: int = 5
    min_used_fraction: float = 0.5
    min_used_observations: int = 1
    use_abs_residual: bool = True
    protect_reciprocal_pairs: bool = False


@dataclass(frozen=True)
class TimeTermRobustIteration:
    iteration: int
    solver_result: TimeTermSparseSolverResult

    row_used_mask: np.ndarray
    row_rejected_this_iteration_mask: np.ndarray
    row_residual_s: np.ndarray
    row_score: np.ndarray

    scale_s: float
    center_s: float
    threshold_s: float
    n_used: int
    n_rejected_total: int
    n_rejected_this_iteration: int


@dataclass(frozen=True)
class TimeTermRobustSolverResult:
    final_solver_result: TimeTermSparseSolverResult
    iterations: tuple[TimeTermRobustIteration, ...]

    initial_used_trace_mask_sorted: np.ndarray
    final_used_trace_mask_sorted: np.ndarray
    final_rejected_trace_mask_sorted: np.ndarray
    rejected_iteration_sorted: np.ndarray

    initial_row_used_mask: np.ndarray
    final_row_used_mask: np.ndarray
    final_row_rejected_mask: np.ndarray
    row_rejected_iteration: np.ndarray

    method: TimeTermRobustMethod
    enabled: bool
    stop_reason: TimeTermRobustStopReason
    robust_options: TimeTermRobustSolverOptions

    n_initial_used_traces: int
    n_final_used_traces: int
    n_rejected_traces: int
    final_used_fraction: float


@dataclass(frozen=True)
class _ValidatedTimeTermDesign:
    matrix: sparse.csr_matrix
    data_s: np.ndarray

    n_traces: int
    n_observations: int
    n_nodes: int

    used_trace_mask_sorted: np.ndarray
    row_trace_index_sorted: np.ndarray
    trace_to_row_index_sorted: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray

    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    row_pick_time_after_static_s: np.ndarray
    row_moveout_time_s: np.ndarray
    row_data_s: np.ndarray

    source_observation_count_by_node: np.ndarray
    receiver_observation_count_by_node: np.ndarray
    total_observation_count_by_node: np.ndarray


def validate_time_term_robust_solver_options(
    options: TimeTermRobustSolverOptions | None,
) -> TimeTermRobustSolverOptions:
    """Validate and normalize robust time-term outlier-rejection options."""
    opts = TimeTermRobustSolverOptions() if options is None else options
    if not isinstance(opts, TimeTermRobustSolverOptions):
        raise ValueError('options must be a TimeTermRobustSolverOptions instance')
    protect_reciprocal_pairs = _coerce_bool(
        opts.protect_reciprocal_pairs,
        name='protect_reciprocal_pairs',
    )
    if protect_reciprocal_pairs:
        raise ValueError('protect_reciprocal_pairs is not implemented')
    return TimeTermRobustSolverOptions(
        enabled=_coerce_bool(opts.enabled, name='enabled'),
        method=_validate_robust_method(opts.method),
        threshold=_coerce_positive_finite_float(opts.threshold, name='threshold'),
        max_iterations=_coerce_positive_int(
            opts.max_iterations,
            name='max_iterations',
        ),
        min_used_fraction=_coerce_min_used_fraction(opts.min_used_fraction),
        min_used_observations=_coerce_positive_int(
            opts.min_used_observations,
            name='min_used_observations',
        ),
        use_abs_residual=_coerce_bool(
            opts.use_abs_residual,
            name='use_abs_residual',
        ),
        protect_reciprocal_pairs=protect_reciprocal_pairs,
    )


def compute_time_term_robust_scores(
    residual_s: np.ndarray,
    *,
    method: TimeTermRobustMethod,
    threshold: float,
    use_abs_residual: bool = True,
) -> tuple[float, float, np.ndarray, float]:
    """Return residual center, scale, score, and threshold in seconds."""
    robust_method = _validate_robust_method(method)
    score_threshold = _coerce_positive_finite_float(threshold, name='threshold')
    use_abs = _coerce_bool(use_abs_residual, name='use_abs_residual')
    residual = _coerce_1d_real_numeric_float64(residual_s, name='residual_s')
    if residual.size == 0:
        raise ValueError('residual_s must be non-empty')
    _validate_all_finite(residual, name='residual_s')

    if robust_method == 'mad':
        center_s = float(np.median(residual))
        raw_mad_s = float(np.median(np.abs(residual - center_s)))
        scale_s = 1.4826 * raw_mad_s
    else:
        center_s = float(np.mean(residual))
        scale_s = float(np.std(residual, ddof=0))
    _coerce_finite_float(center_s, name='residual center')
    _coerce_finite_float(scale_s, name='residual scale')

    threshold_s = score_threshold * scale_s
    if scale_s <= ROBUST_SCALE_FLOOR_S:
        score = np.zeros(residual.shape, dtype=np.float64)
    else:
        centered = residual - center_s
        if use_abs:
            centered = np.abs(centered)
        score = np.ascontiguousarray(centered / scale_s, dtype=np.float64)
    _validate_all_finite(score, name='row_score')
    return center_s, scale_s, score, threshold_s


def subset_time_term_design_matrix_rows(
    design: TimeTermDesignMatrix,
    row_used_mask: np.ndarray,
) -> TimeTermDesignMatrix:
    """Build a row-subset time-term design while preserving full trace indexing."""
    validated = _validate_design(design)
    used_rows = _coerce_1d_bool_like_array(
        row_used_mask,
        name='row_used_mask',
        expected_shape=(validated.n_observations,),
    )
    selected_rows = np.flatnonzero(used_rows).astype(np.int64, copy=False)
    n_observations = int(selected_rows.shape[0])
    if n_observations <= 0:
        raise ValueError('at least one time-term design row must be selected')

    matrix = validated.matrix[selected_rows, :].tocsr().astype(np.float64, copy=False)
    matrix.sort_indices()
    row_trace_index = np.ascontiguousarray(
        validated.row_trace_index_sorted[selected_rows],
        dtype=np.int64,
    )
    trace_to_row_index = np.full(validated.n_traces, -1, dtype=np.int64)
    for subset_row_index, trace_index in enumerate(row_trace_index):
        if trace_to_row_index[int(trace_index)] < 0:
            trace_to_row_index[int(trace_index)] = int(subset_row_index)
    used_trace_mask = np.zeros(validated.n_traces, dtype=bool)
    used_trace_mask[row_trace_index] = True

    row_source_node_id = np.ascontiguousarray(
        validated.row_source_node_id[selected_rows],
        dtype=np.int64,
    )
    row_receiver_node_id = np.ascontiguousarray(
        validated.row_receiver_node_id[selected_rows],
        dtype=np.int64,
    )
    source_count = np.bincount(
        row_source_node_id,
        minlength=validated.n_nodes,
    ).astype(np.int64, copy=False)
    receiver_count = np.bincount(
        row_receiver_node_id,
        minlength=validated.n_nodes,
    ).astype(np.int64, copy=False)
    total_count = np.ascontiguousarray(source_count + receiver_count, dtype=np.int64)

    return TimeTermDesignMatrix(
        matrix=matrix,
        data_s=np.ascontiguousarray(validated.data_s[selected_rows], dtype=np.float64),
        n_traces=validated.n_traces,
        n_observations=n_observations,
        n_nodes=validated.n_nodes,
        used_trace_mask_sorted=np.ascontiguousarray(used_trace_mask, dtype=bool),
        row_trace_index_sorted=row_trace_index,
        trace_to_row_index_sorted=trace_to_row_index,
        source_node_id_sorted=np.ascontiguousarray(
            validated.source_node_id_sorted,
            dtype=np.int64,
        ),
        receiver_node_id_sorted=np.ascontiguousarray(
            validated.receiver_node_id_sorted,
            dtype=np.int64,
        ),
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        row_pick_time_after_static_s=np.ascontiguousarray(
            validated.row_pick_time_after_static_s[selected_rows],
            dtype=np.float64,
        ),
        row_moveout_time_s=np.ascontiguousarray(
            validated.row_moveout_time_s[selected_rows],
            dtype=np.float64,
        ),
        row_data_s=np.ascontiguousarray(
            validated.row_data_s[selected_rows],
            dtype=np.float64,
        ),
        source_observation_count_by_node=np.ascontiguousarray(
            source_count,
            dtype=np.int64,
        ),
        receiver_observation_count_by_node=np.ascontiguousarray(
            receiver_count,
            dtype=np.int64,
        ),
        total_observation_count_by_node=total_count,
    )


def solve_time_term_robust_least_squares(
    design: TimeTermDesignMatrix,
    *,
    solver_options: TimeTermSparseSolverOptions | None = None,
    robust_options: TimeTermRobustSolverOptions | None = None,
) -> TimeTermRobustSolverResult:
    """Iteratively reject residual outlier rows and rerun the time-term solver."""
    validated_design = _validate_design(design)
    validated_robust_options = validate_time_term_robust_solver_options(
        robust_options,
    )
    initial_row_used_mask = np.ones(validated_design.n_observations, dtype=bool)
    initial_used_trace_mask = _trace_mask_from_row_mask(
        validated_design,
        initial_row_used_mask,
    )
    n_initial_observations = int(np.count_nonzero(initial_row_used_mask))
    if n_initial_observations < validated_robust_options.min_used_observations:
        raise ValueError('initial observations are below min_used_observations')

    row_rejected_iteration = np.full(
        validated_design.n_observations,
        -1,
        dtype=np.int64,
    )

    if not validated_robust_options.enabled:
        solver_result = solve_time_term_sparse_least_squares(
            design,
            options=solver_options,
        )
        iteration = _build_iteration(
            iteration_index=0,
            solver_result=solver_result,
            row_used_mask=initial_row_used_mask,
            row_rejected_this_iteration_mask=np.zeros(
                validated_design.n_observations,
                dtype=bool,
            ),
            subset_row_indices=np.arange(
                validated_design.n_observations,
                dtype=np.int64,
            ),
            row_score_subset=np.zeros(validated_design.n_observations, dtype=np.float64),
            center_s=0.0,
            scale_s=0.0,
            threshold_s=0.0,
            n_rejected_total=0,
        )
        return _build_result(
            final_solver_result=solver_result,
            iterations=(iteration,),
            validated_design=validated_design,
            robust_options=validated_robust_options,
            initial_row_used_mask=initial_row_used_mask,
            final_row_used_mask=initial_row_used_mask,
            row_rejected_iteration=row_rejected_iteration,
            initial_used_trace_mask=initial_used_trace_mask,
            stop_reason='disabled',
        )

    current_row_used_mask = initial_row_used_mask.copy()
    iterations: list[TimeTermRobustIteration] = []
    final_solver_result: TimeTermSparseSolverResult | None = None
    stop_reason: TimeTermRobustStopReason | None = None

    for iteration_index in range(validated_robust_options.max_iterations):
        subset_row_indices = np.flatnonzero(current_row_used_mask).astype(
            np.int64,
            copy=False,
        )
        subset = subset_time_term_design_matrix_rows(
            design,
            current_row_used_mask,
        )
        solver_result = solve_time_term_sparse_least_squares(
            subset,
            options=solver_options,
        )
        residual = _coerce_1d_real_numeric_float64(
            solver_result.row_residual_after_s,
            name='row_residual_after_s',
            expected_shape=(subset.n_observations,),
        )
        _validate_all_finite(residual, name='row_residual_after_s')
        center_s, scale_s, row_score_subset, threshold_s = (
            compute_time_term_robust_scores(
                residual,
                method=validated_robust_options.method,
                threshold=validated_robust_options.threshold,
                use_abs_residual=validated_robust_options.use_abs_residual,
            )
        )

        if scale_s <= ROBUST_SCALE_FLOOR_S:
            rejected_this_iteration = np.zeros(
                validated_design.n_observations,
                dtype=bool,
            )
            iterations.append(
                _build_iteration(
                    iteration_index=iteration_index,
                    solver_result=solver_result,
                    row_used_mask=current_row_used_mask,
                    row_rejected_this_iteration_mask=rejected_this_iteration,
                    subset_row_indices=subset_row_indices,
                    row_score_subset=row_score_subset,
                    center_s=center_s,
                    scale_s=scale_s,
                    threshold_s=threshold_s,
                    n_rejected_total=int(np.count_nonzero(row_rejected_iteration >= 0)),
                )
            )
            final_solver_result = solver_result
            stop_reason = 'zero_scale'
            break

        rejected_subset_mask = np.ascontiguousarray(
            row_score_subset > validated_robust_options.threshold,
            dtype=bool,
        )
        rejected_this_iteration = np.zeros(
            validated_design.n_observations,
            dtype=bool,
        )
        rejected_this_iteration[subset_row_indices[rejected_subset_mask]] = True
        rejected_this_iteration &= current_row_used_mask

        if not np.any(rejected_this_iteration):
            iterations.append(
                _build_iteration(
                    iteration_index=iteration_index,
                    solver_result=solver_result,
                    row_used_mask=current_row_used_mask,
                    row_rejected_this_iteration_mask=rejected_this_iteration,
                    subset_row_indices=subset_row_indices,
                    row_score_subset=row_score_subset,
                    center_s=center_s,
                    scale_s=scale_s,
                    threshold_s=threshold_s,
                    n_rejected_total=int(np.count_nonzero(row_rejected_iteration >= 0)),
                )
            )
            final_solver_result = solver_result
            stop_reason = 'converged'
            break

        iteration_row_used_mask = current_row_used_mask.copy()
        proposed_row_used_mask = current_row_used_mask.copy()
        proposed_row_used_mask[rejected_this_iteration] = False
        _validate_min_remaining(
            n_initial_observations=n_initial_observations,
            proposed_row_used_mask=proposed_row_used_mask,
            options=validated_robust_options,
        )

        row_rejected_iteration[rejected_this_iteration] = iteration_index
        current_row_used_mask = proposed_row_used_mask
        iterations.append(
            _build_iteration(
                iteration_index=iteration_index,
                solver_result=solver_result,
                row_used_mask=iteration_row_used_mask,
                row_rejected_this_iteration_mask=rejected_this_iteration,
                subset_row_indices=subset_row_indices,
                row_score_subset=row_score_subset,
                center_s=center_s,
                scale_s=scale_s,
                threshold_s=threshold_s,
                n_rejected_total=int(np.count_nonzero(row_rejected_iteration >= 0)),
            )
        )

        if iteration_index == validated_robust_options.max_iterations - 1:
            final_subset = subset_time_term_design_matrix_rows(
                design,
                current_row_used_mask,
            )
            final_solver_result = solve_time_term_sparse_least_squares(
                final_subset,
                options=solver_options,
            )
            stop_reason = 'max_iterations'
            break

    if final_solver_result is None or stop_reason is None:
        raise RuntimeError('robust time-term solver did not produce a final result')

    return _build_result(
        final_solver_result=final_solver_result,
        iterations=tuple(iterations),
        validated_design=validated_design,
        robust_options=validated_robust_options,
        initial_row_used_mask=initial_row_used_mask,
        final_row_used_mask=current_row_used_mask,
        row_rejected_iteration=row_rejected_iteration,
        initial_used_trace_mask=initial_used_trace_mask,
        stop_reason=stop_reason,
    )


def summarize_time_term_robust_solver_result(
    result: TimeTermRobustSolverResult,
) -> dict[str, object]:
    """Return a JSON-safe summary for future artifacts and job logs."""
    robust_options = result.robust_options
    final_solver_result = result.final_solver_result
    return {
        'enabled': bool(result.enabled),
        'method': result.method,
        'threshold': _json_float(robust_options.threshold),
        'max_iterations': int(robust_options.max_iterations),
        'stop_reason': result.stop_reason,
        'n_iterations': int(len(result.iterations)),
        'n_initial_used_traces': int(result.n_initial_used_traces),
        'n_final_used_traces': int(result.n_final_used_traces),
        'n_rejected_traces': int(result.n_rejected_traces),
        'final_used_fraction': _json_float(result.final_used_fraction),
        'per_iteration': [
            {
                'iteration': int(iteration.iteration),
                'n_used': int(iteration.n_used),
                'n_rejected_this_iteration': int(
                    iteration.n_rejected_this_iteration
                ),
                'n_rejected_total': int(iteration.n_rejected_total),
                'center_ms': _json_float(iteration.center_s * 1000.0),
                'scale_ms': _json_float(iteration.scale_s * 1000.0),
                'threshold_ms': _json_float(iteration.threshold_s * 1000.0),
                'rms_residual_after_ms': _json_float(
                    iteration.solver_result.rms_residual_after_s * 1000.0
                ),
            }
            for iteration in result.iterations
        ],
        'final': {
            'node_time_term_ms': _stats_payload(
                final_solver_result.node_time_term_s * 1000.0
            ),
            'estimated_trace_time_term_delay_ms': _stats_payload(
                final_solver_result.estimated_trace_time_term_delay_s_sorted * 1000.0
            ),
            'row_residual_after_ms': _stats_payload(
                final_solver_result.row_residual_after_s * 1000.0
            ),
        },
    }


def _build_iteration(
    *,
    iteration_index: int,
    solver_result: TimeTermSparseSolverResult,
    row_used_mask: np.ndarray,
    row_rejected_this_iteration_mask: np.ndarray,
    subset_row_indices: np.ndarray,
    row_score_subset: np.ndarray,
    center_s: float,
    scale_s: float,
    threshold_s: float,
    n_rejected_total: int,
) -> TimeTermRobustIteration:
    full_shape = row_used_mask.shape
    row_residual = np.zeros(full_shape, dtype=np.float64)
    row_score = np.zeros(full_shape, dtype=np.float64)
    residual = _coerce_1d_real_numeric_float64(
        solver_result.row_residual_after_s,
        name='row_residual_after_s',
        expected_shape=(subset_row_indices.shape[0],),
    )
    score = _coerce_1d_real_numeric_float64(
        row_score_subset,
        name='row_score',
        expected_shape=(subset_row_indices.shape[0],),
    )
    _validate_all_finite(residual, name='row_residual_after_s')
    _validate_all_finite(score, name='row_score')
    row_residual[subset_row_indices] = residual
    row_score[subset_row_indices] = score
    rejected_this_iteration = np.ascontiguousarray(
        row_rejected_this_iteration_mask,
        dtype=bool,
    )
    return TimeTermRobustIteration(
        iteration=int(iteration_index),
        solver_result=solver_result,
        row_used_mask=np.ascontiguousarray(row_used_mask, dtype=bool),
        row_rejected_this_iteration_mask=rejected_this_iteration,
        row_residual_s=np.ascontiguousarray(row_residual, dtype=np.float64),
        row_score=np.ascontiguousarray(row_score, dtype=np.float64),
        scale_s=_coerce_finite_float(scale_s, name='scale_s'),
        center_s=_coerce_finite_float(center_s, name='center_s'),
        threshold_s=_coerce_finite_float(threshold_s, name='threshold_s'),
        n_used=int(np.count_nonzero(row_used_mask)),
        n_rejected_total=int(n_rejected_total),
        n_rejected_this_iteration=int(np.count_nonzero(rejected_this_iteration)),
    )


def _build_result(
    *,
    final_solver_result: TimeTermSparseSolverResult,
    iterations: tuple[TimeTermRobustIteration, ...],
    validated_design: _ValidatedTimeTermDesign,
    robust_options: TimeTermRobustSolverOptions,
    initial_row_used_mask: np.ndarray,
    final_row_used_mask: np.ndarray,
    row_rejected_iteration: np.ndarray,
    initial_used_trace_mask: np.ndarray,
    stop_reason: TimeTermRobustStopReason,
) -> TimeTermRobustSolverResult:
    final_row_rejected_mask = np.ascontiguousarray(
        initial_row_used_mask & ~final_row_used_mask,
        dtype=bool,
    )
    rejected_iteration_sorted = _trace_rejected_iteration_from_rows(
        validated_design,
        row_rejected_iteration,
    )
    final_rejected_trace_mask = np.ascontiguousarray(
        rejected_iteration_sorted >= 0,
        dtype=bool,
    )
    final_used_trace_mask = np.ascontiguousarray(
        initial_used_trace_mask & ~final_rejected_trace_mask,
        dtype=bool,
    )
    n_initial_used_traces = int(np.count_nonzero(initial_used_trace_mask))
    n_final_used_traces = int(np.count_nonzero(final_used_trace_mask))
    return TimeTermRobustSolverResult(
        final_solver_result=final_solver_result,
        iterations=iterations,
        initial_used_trace_mask_sorted=np.ascontiguousarray(
            initial_used_trace_mask,
            dtype=bool,
        ),
        final_used_trace_mask_sorted=final_used_trace_mask,
        final_rejected_trace_mask_sorted=final_rejected_trace_mask,
        rejected_iteration_sorted=rejected_iteration_sorted,
        initial_row_used_mask=np.ascontiguousarray(initial_row_used_mask, dtype=bool),
        final_row_used_mask=np.ascontiguousarray(final_row_used_mask, dtype=bool),
        final_row_rejected_mask=final_row_rejected_mask,
        row_rejected_iteration=np.ascontiguousarray(
            row_rejected_iteration,
            dtype=np.int64,
        ),
        method=robust_options.method,
        enabled=robust_options.enabled,
        stop_reason=stop_reason,
        robust_options=robust_options,
        n_initial_used_traces=n_initial_used_traces,
        n_final_used_traces=n_final_used_traces,
        n_rejected_traces=int(np.count_nonzero(final_rejected_trace_mask)),
        final_used_fraction=_fraction(n_final_used_traces, n_initial_used_traces),
    )


def _validate_min_remaining(
    *,
    n_initial_observations: int,
    proposed_row_used_mask: np.ndarray,
    options: TimeTermRobustSolverOptions,
) -> None:
    remaining_observations = int(np.count_nonzero(proposed_row_used_mask))
    if remaining_observations < options.min_used_observations:
        raise ValueError(
            'robust rejection would leave too few observations: min_used_observations'
        )
    used_fraction = _fraction(remaining_observations, n_initial_observations)
    if used_fraction < options.min_used_fraction:
        raise ValueError(
            'robust rejection would leave too few observations: min_used_fraction'
        )


def _trace_mask_from_row_mask(
    design: _ValidatedTimeTermDesign,
    row_mask: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(design.n_traces, dtype=bool)
    used_rows = _coerce_1d_bool_like_array(
        row_mask,
        name='row_mask',
        expected_shape=(design.n_observations,),
    )
    mask[design.row_trace_index_sorted[used_rows]] = True
    return np.ascontiguousarray(mask, dtype=bool)


def _trace_rejected_iteration_from_rows(
    design: _ValidatedTimeTermDesign,
    row_rejected_iteration: np.ndarray,
) -> np.ndarray:
    row_iteration = _coerce_1d_integer_int64(
        row_rejected_iteration,
        name='row_rejected_iteration',
        expected_shape=(design.n_observations,),
    )
    rejected_iteration = np.full(design.n_traces, -1, dtype=np.int64)
    rejected_rows = np.flatnonzero(row_iteration >= 0).astype(np.int64, copy=False)
    for row_index in rejected_rows:
        trace_index = int(design.row_trace_index_sorted[int(row_index)])
        iteration = int(row_iteration[int(row_index)])
        current = int(rejected_iteration[trace_index])
        if current < 0 or iteration < current:
            rejected_iteration[trace_index] = iteration
    return np.ascontiguousarray(rejected_iteration, dtype=np.int64)


def _validate_design(design: TimeTermDesignMatrix) -> _ValidatedTimeTermDesign:
    if not isinstance(design, TimeTermDesignMatrix):
        raise ValueError('design must be a TimeTermDesignMatrix instance')
    n_traces = _coerce_positive_int(design.n_traces, name='design.n_traces')
    n_observations = _coerce_positive_int(
        design.n_observations,
        name='design.n_observations',
    )
    n_nodes = _coerce_positive_int(design.n_nodes, name='design.n_nodes')

    matrix = _coerce_sparse_matrix_float64_csr(design.matrix)
    if matrix.shape != (n_observations, n_nodes):
        raise ValueError('design.matrix shape does not match design dimensions')
    data_s = _coerce_1d_real_numeric_float64(
        design.data_s,
        name='design.data_s',
        expected_shape=(n_observations,),
    )
    _validate_all_finite(data_s, name='design.data_s')

    used_trace_mask = _coerce_1d_bool_array(
        design.used_trace_mask_sorted,
        name='design.used_trace_mask_sorted',
        expected_shape=(n_traces,),
    )
    row_trace_index = _coerce_1d_integer_int64(
        design.row_trace_index_sorted,
        name='design.row_trace_index_sorted',
        expected_shape=(n_observations,),
    )
    _validate_index_range(
        row_trace_index,
        n_unique=n_traces,
        name='design.row_trace_index_sorted',
    )
    trace_to_row_index = _coerce_trace_to_row_index(
        design.trace_to_row_index_sorted,
        n_traces=n_traces,
        n_observations=n_observations,
    )
    _validate_trace_row_index_mapping(
        used_trace_mask=used_trace_mask,
        row_trace_index=row_trace_index,
        trace_to_row_index=trace_to_row_index,
        n_traces=n_traces,
        n_observations=n_observations,
    )
    source_node_id = _coerce_1d_integer_int64(
        design.source_node_id_sorted,
        name='design.source_node_id_sorted',
        expected_shape=(n_traces,),
    )
    receiver_node_id = _coerce_1d_integer_int64(
        design.receiver_node_id_sorted,
        name='design.receiver_node_id_sorted',
        expected_shape=(n_traces,),
    )
    row_source_node_id = _coerce_1d_integer_int64(
        design.row_source_node_id,
        name='design.row_source_node_id',
        expected_shape=(n_observations,),
    )
    row_receiver_node_id = _coerce_1d_integer_int64(
        design.row_receiver_node_id,
        name='design.row_receiver_node_id',
        expected_shape=(n_observations,),
    )
    for values, name in (
        (source_node_id, 'design.source_node_id_sorted'),
        (receiver_node_id, 'design.receiver_node_id_sorted'),
        (row_source_node_id, 'design.row_source_node_id'),
        (row_receiver_node_id, 'design.row_receiver_node_id'),
    ):
        _validate_index_range(values, n_unique=n_nodes, name=name)

    row_pick_time = _coerce_1d_real_numeric_float64(
        design.row_pick_time_after_static_s,
        name='design.row_pick_time_after_static_s',
        expected_shape=(n_observations,),
    )
    row_moveout_time = _coerce_1d_real_numeric_float64(
        design.row_moveout_time_s,
        name='design.row_moveout_time_s',
        expected_shape=(n_observations,),
    )
    row_data_s = _coerce_1d_real_numeric_float64(
        design.row_data_s,
        name='design.row_data_s',
        expected_shape=(n_observations,),
    )
    _validate_all_finite(row_pick_time, name='design.row_pick_time_after_static_s')
    _validate_all_finite(row_moveout_time, name='design.row_moveout_time_s')
    _validate_all_finite(row_data_s, name='design.row_data_s')

    source_count = _coerce_1d_integer_int64(
        design.source_observation_count_by_node,
        name='design.source_observation_count_by_node',
        expected_shape=(n_nodes,),
    )
    receiver_count = _coerce_1d_integer_int64(
        design.receiver_observation_count_by_node,
        name='design.receiver_observation_count_by_node',
        expected_shape=(n_nodes,),
    )
    total_count = _coerce_1d_integer_int64(
        design.total_observation_count_by_node,
        name='design.total_observation_count_by_node',
        expected_shape=(n_nodes,),
    )
    computed_source_count = np.bincount(
        row_source_node_id,
        minlength=n_nodes,
    ).astype(np.int64, copy=False)
    computed_receiver_count = np.bincount(
        row_receiver_node_id,
        minlength=n_nodes,
    ).astype(np.int64, copy=False)
    computed_total_count = np.ascontiguousarray(
        computed_source_count + computed_receiver_count,
        dtype=np.int64,
    )
    if np.any(source_count != computed_source_count):
        raise ValueError('design.source_observation_count_by_node does not match rows')
    if np.any(receiver_count != computed_receiver_count):
        raise ValueError('design.receiver_observation_count_by_node does not match rows')
    if np.any(total_count != computed_total_count):
        raise ValueError('design.total_observation_count_by_node does not match rows')

    return _ValidatedTimeTermDesign(
        matrix=matrix,
        data_s=data_s,
        n_traces=n_traces,
        n_observations=n_observations,
        n_nodes=n_nodes,
        used_trace_mask_sorted=used_trace_mask,
        row_trace_index_sorted=row_trace_index,
        trace_to_row_index_sorted=trace_to_row_index,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        row_pick_time_after_static_s=row_pick_time,
        row_moveout_time_s=row_moveout_time,
        row_data_s=row_data_s,
        source_observation_count_by_node=np.ascontiguousarray(
            source_count,
            dtype=np.int64,
        ),
        receiver_observation_count_by_node=np.ascontiguousarray(
            receiver_count,
            dtype=np.int64,
        ),
        total_observation_count_by_node=np.ascontiguousarray(
            total_count,
            dtype=np.int64,
        ),
    )


def _coerce_sparse_matrix_float64_csr(matrix: object) -> sparse.csr_matrix:
    if not sparse.issparse(matrix):
        raise ValueError('matrix must be a SciPy sparse matrix')
    if len(matrix.shape) != 2:
        raise ValueError('matrix must be 2D')
    n_rows = _coerce_positive_int(matrix.shape[0], name='matrix.shape[0]')
    n_cols = _coerce_positive_int(matrix.shape[1], name='matrix.shape[1]')
    dtype = np.dtype(matrix.dtype)
    if not _is_real_numeric_dtype(dtype):
        raise ValueError('matrix dtype must be real numeric')
    matrix_csr = matrix.tocsr().astype(np.float64, copy=False)
    if matrix_csr.shape != (n_rows, n_cols):
        raise ValueError('matrix shape changed during CSR conversion')
    _validate_all_finite(matrix_csr.data, name='matrix.data')
    return matrix_csr


def _coerce_1d_bool_like_array(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if arr.shape != expected_shape:
        raise ValueError(f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}')
    if np.issubdtype(arr.dtype, np.bool_):
        return np.ascontiguousarray(arr, dtype=bool)
    if not _is_real_numeric_dtype(arr.dtype):
        raise ValueError(f'{name} must be bool-like')
    arr_f64 = np.ascontiguousarray(arr, dtype=np.float64)
    _validate_all_finite(arr_f64, name=name)
    if np.any((arr_f64 != 0.0) & (arr_f64 != 1.0)):
        raise ValueError(f'{name} must be bool-like')
    return np.ascontiguousarray(arr_f64.astype(bool), dtype=bool)


def _coerce_trace_to_row_index(
    values: object,
    *,
    n_traces: int,
    n_observations: int,
) -> np.ndarray:
    arr = _coerce_1d_integer_int64(
        values,
        name='design.trace_to_row_index_sorted',
        expected_shape=(n_traces,),
    )
    if np.any(arr < -1):
        raise ValueError('design.trace_to_row_index_sorted must be greater than or equal to -1')
    if np.any(arr >= n_observations):
        raise ValueError('design.trace_to_row_index_sorted contains row indices outside range')
    return arr


def _validate_trace_row_index_mapping(
    *,
    used_trace_mask: np.ndarray,
    row_trace_index: np.ndarray,
    trace_to_row_index: np.ndarray,
    n_traces: int,
    n_observations: int,
) -> None:
    if np.unique(row_trace_index).shape[0] != n_observations:
        raise ValueError('design.row_trace_index_sorted must contain unique traces')

    expected_used_trace_mask = np.zeros(n_traces, dtype=bool)
    expected_used_trace_mask[row_trace_index] = True
    if np.any(used_trace_mask != expected_used_trace_mask):
        raise ValueError(
            'design.used_trace_mask_sorted does not match design.row_trace_index_sorted'
        )

    expected_trace_to_row_index = np.full(n_traces, -1, dtype=np.int64)
    expected_trace_to_row_index[row_trace_index] = np.arange(
        n_observations,
        dtype=np.int64,
    )
    if np.any(trace_to_row_index != expected_trace_to_row_index):
        raise ValueError(
            'design.trace_to_row_index_sorted does not match '
            'design.row_trace_index_sorted'
        )


def _coerce_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be bool')
    return bool(value)


def _coerce_min_used_fraction(value: object) -> float:
    out = _coerce_positive_finite_float(value, name='min_used_fraction')
    if out > 1.0:
        raise ValueError('min_used_fraction must be less than or equal to 1')
    return out


def _validate_index_range(
    indices: np.ndarray,
    *,
    n_unique: int,
    name: str,
) -> None:
    if indices.size == 0:
        return
    if np.any(indices < 0):
        raise ValueError(f'{name} must be greater than or equal to 0')
    if np.any(indices >= n_unique):
        raise ValueError(f'{name} contains values outside 0..{n_unique - 1}')


def _validate_robust_method(value: object) -> TimeTermRobustMethod:
    if value == 'mad':
        return 'mad'
    if value == 'sigma':
        return 'sigma'
    raise ValueError('method must be mad or sigma')


def _validate_all_finite(values: np.ndarray, *, name: str) -> None:
    if np.any(~np.isfinite(values)):
        raise ValueError(f'{name} must contain only finite values')


def _json_float(value: float) -> float:
    return float(_coerce_finite_float(value, name='summary value'))


def _stats_payload(values: np.ndarray) -> dict[str, float | int | None]:
    arr = _coerce_1d_real_numeric_float64(values, name='summary values')
    finite = np.ascontiguousarray(arr[np.isfinite(arr)], dtype=np.float64)
    count = int(finite.shape[0])
    if count == 0:
        return {
            'count': 0,
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None,
            'max_abs': None,
        }
    return {
        'count': count,
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'mean': float(np.mean(finite)),
        'median': float(np.median(finite)),
        'std': float(np.std(finite, ddof=0)),
        'max_abs': float(np.max(np.abs(finite))),
    }


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


__all__ = [
    'ROBUST_SCALE_FLOOR_S',
    'TimeTermRobustIteration',
    'TimeTermRobustMethod',
    'TimeTermRobustSolverOptions',
    'TimeTermRobustSolverResult',
    'TimeTermRobustStopReason',
    'compute_time_term_robust_scores',
    'solve_time_term_robust_least_squares',
    'subset_time_term_design_matrix_rows',
    'summarize_time_term_robust_solver_result',
    'validate_time_term_robust_solver_options',
]
