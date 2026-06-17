"""Convert time-term estimated delays into applied event-time shifts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np

from seis_statics._validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_integer_int64 as _common_coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_finite_float as _coerce_finite_float,
    coerce_nonnegative_finite_float as _coerce_nonnegative_finite_float,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
)
from seis_statics.time_term.robust_solver import (
    TimeTermRobustSolverResult,
)
from seis_statics.time_term.sparse_solver import (
    TimeTermSparseSolverResult,
)
from seis_statics.time_term.types import ORDER, TimeTermInversionInputs

TimeTermRejectedTracePolicy = Literal['use_final_model']

_coerce_1d_integer_int64 = partial(
    _common_coerce_1d_integer_int64,
    allow_integer_like_float=False,
)

SIGN_CONVENTION = (
    'estimated_trace_time_term_delay_s = source_node_time_term_s + '
    'receiver_node_time_term_s; applied_weathering_shift_s = '
    '-estimated_trace_time_term_delay_s; corrected(t)=raw(t-shift_s)'
)
DELAY_TO_SHIFT_CONVENTION = (
    'applied_weathering_shift_s_sorted = '
    '-estimated_trace_time_term_delay_s_sorted'
)
FINAL_SHIFT_CONVENTION = (
    'final_trace_shift_s_sorted = datum_trace_shift_s_sorted '
    '+ residual_applied_shift_s_sorted '
    '+ applied_weathering_shift_s_sorted'
)


@dataclass(frozen=True)
class TimeTermAppliedShiftOptions:
    max_abs_weathering_shift_ms: float | None = 500.0
    max_abs_final_shift_ms: float | None = 750.0
    include_rejected_traces: bool = True
    rejected_trace_policy: TimeTermRejectedTracePolicy = 'use_final_model'


@dataclass(frozen=True)
class TimeTermAppliedShiftResult:
    n_traces: int
    dt: float

    node_time_term_s: np.ndarray

    source_node_time_term_s_sorted: np.ndarray
    receiver_node_time_term_s_sorted: np.ndarray

    estimated_trace_time_term_delay_s_sorted: np.ndarray
    applied_weathering_shift_s_sorted: np.ndarray

    datum_trace_shift_s_sorted: np.ndarray
    residual_applied_shift_s_sorted: np.ndarray
    final_trace_shift_s_sorted: np.ndarray

    valid_pick_mask_sorted: np.ndarray
    final_used_trace_mask_sorted: np.ndarray
    rejected_trace_mask_sorted: np.ndarray
    rejected_iteration_sorted: np.ndarray

    sign_convention: str
    order: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class _ValidatedInputs:
    n_traces: int
    dt: float
    n_nodes: int
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    datum_trace_shift_s_sorted: np.ndarray
    residual_applied_shift_s_sorted: np.ndarray
    valid_pick_mask_sorted: np.ndarray


@dataclass(frozen=True)
class _ExtractedSolverResult:
    sparse_result: TimeTermSparseSolverResult
    final_used_trace_mask_sorted: np.ndarray
    rejected_trace_mask_sorted: np.ndarray
    rejected_iteration_sorted: np.ndarray
    solver_result_kind: str


def delay_to_applied_shift_s(delay_s: np.ndarray) -> np.ndarray:
    """Convert an estimated delay into an applied event-time shift."""
    return np.ascontiguousarray(-np.asarray(delay_s, dtype=np.float64), dtype=np.float64)


def compute_estimated_trace_time_term_delay_s(
    *,
    node_time_term_s: np.ndarray,
    source_node_id_sorted: np.ndarray,
    receiver_node_id_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute trace-level estimated delay from source/receiver node terms."""
    node_time_term = _coerce_1d_real_numeric_float64(
        node_time_term_s,
        name='node_time_term_s',
    )
    _validate_all_finite(node_time_term, name='node_time_term_s')
    n_nodes = int(node_time_term.shape[0])
    if n_nodes <= 0:
        raise ValueError('node_time_term_s must be non-empty')

    source_node_id = _coerce_1d_integer_int64(
        source_node_id_sorted,
        name='source_node_id_sorted',
    )
    receiver_node_id = _coerce_1d_integer_int64(
        receiver_node_id_sorted,
        name='receiver_node_id_sorted',
    )
    if source_node_id.shape != receiver_node_id.shape:
        raise ValueError('source_node_id_sorted and receiver_node_id_sorted must match')
    _validate_index_range(
        source_node_id,
        n_unique=n_nodes,
        name='source_node_id_sorted',
    )
    _validate_index_range(
        receiver_node_id,
        n_unique=n_nodes,
        name='receiver_node_id_sorted',
    )

    source = np.ascontiguousarray(node_time_term[source_node_id], dtype=np.float64)
    receiver = np.ascontiguousarray(node_time_term[receiver_node_id], dtype=np.float64)
    estimated = np.ascontiguousarray(source + receiver, dtype=np.float64)
    _validate_all_finite(estimated, name='estimated_trace_time_term_delay_s_sorted')
    return estimated, source, receiver


def build_time_term_applied_shift_result(
    *,
    inputs: TimeTermInversionInputs,
    solver_result: TimeTermSparseSolverResult | TimeTermRobustSolverResult,
    options: TimeTermAppliedShiftOptions | None = None,
) -> TimeTermAppliedShiftResult:
    """Build sorted trace-level weathering and final applied shifts."""
    validated_options = _validate_options(options)
    validated_inputs = _validate_inputs(inputs)
    extracted = _extract_final_sparse_result(
        solver_result,
        n_traces=validated_inputs.n_traces,
    )
    sparse_result = extracted.sparse_result

    node_time_term = _coerce_1d_real_numeric_float64(
        sparse_result.node_time_term_s,
        name='node_time_term_s',
        expected_shape=(validated_inputs.n_nodes,),
    )
    _validate_all_finite(node_time_term, name='node_time_term_s')
    (
        estimated_delay,
        source_node_time_term,
        receiver_node_time_term,
    ) = compute_estimated_trace_time_term_delay_s(
        node_time_term_s=node_time_term,
        source_node_id_sorted=validated_inputs.source_node_id_sorted,
        receiver_node_id_sorted=validated_inputs.receiver_node_id_sorted,
    )

    solver_estimated_delay = _coerce_1d_real_numeric_float64(
        sparse_result.estimated_trace_time_term_delay_s_sorted,
        name='estimated_trace_time_term_delay_s_sorted',
        expected_shape=(validated_inputs.n_traces,),
    )
    _validate_all_finite(
        solver_estimated_delay,
        name='estimated_trace_time_term_delay_s_sorted',
    )
    if not np.allclose(solver_estimated_delay, estimated_delay):
        raise ValueError(
            'estimated_trace_time_term_delay_s_sorted does not match '
            'node_time_term_s mapping'
        )

    applied_weathering_shift = delay_to_applied_shift_s(estimated_delay)
    _validate_shift_limit(
        applied_weathering_shift,
        max_abs_ms=validated_options.max_abs_weathering_shift_ms,
        limit_name='max_abs_weathering_shift_ms',
    )
    final_trace_shift = np.ascontiguousarray(
        validated_inputs.datum_trace_shift_s_sorted
        + validated_inputs.residual_applied_shift_s_sorted
        + applied_weathering_shift,
        dtype=np.float64,
    )
    _validate_all_finite(final_trace_shift, name='final_trace_shift_s_sorted')
    _validate_shift_limit(
        final_trace_shift,
        max_abs_ms=validated_options.max_abs_final_shift_ms,
        limit_name='max_abs_final_shift_ms',
    )

    metadata: dict[str, object] = {
        'kind': 'time_term_applied_shift',
        'order': ORDER,
        'delay_to_shift_convention': DELAY_TO_SHIFT_CONVENTION,
        'final_shift_convention': FINAL_SHIFT_CONVENTION,
        'rejected_trace_policy': validated_options.rejected_trace_policy,
        'include_rejected_traces': validated_options.include_rejected_traces,
        'solver_result_kind': extracted.solver_result_kind,
    }

    return TimeTermAppliedShiftResult(
        n_traces=validated_inputs.n_traces,
        dt=validated_inputs.dt,
        node_time_term_s=np.ascontiguousarray(node_time_term, dtype=np.float64),
        source_node_time_term_s_sorted=source_node_time_term,
        receiver_node_time_term_s_sorted=receiver_node_time_term,
        estimated_trace_time_term_delay_s_sorted=estimated_delay,
        applied_weathering_shift_s_sorted=applied_weathering_shift,
        datum_trace_shift_s_sorted=validated_inputs.datum_trace_shift_s_sorted,
        residual_applied_shift_s_sorted=validated_inputs.residual_applied_shift_s_sorted,
        final_trace_shift_s_sorted=final_trace_shift,
        valid_pick_mask_sorted=validated_inputs.valid_pick_mask_sorted,
        final_used_trace_mask_sorted=extracted.final_used_trace_mask_sorted,
        rejected_trace_mask_sorted=extracted.rejected_trace_mask_sorted,
        rejected_iteration_sorted=extracted.rejected_iteration_sorted,
        sign_convention=SIGN_CONVENTION,
        order=ORDER,
        metadata=metadata,
    )


def summarize_time_term_applied_shift_result(
    result: TimeTermAppliedShiftResult,
) -> dict[str, object]:
    """Return a JSON-safe summary for future artifacts and job logs."""
    if not isinstance(result, TimeTermAppliedShiftResult):
        raise ValueError('result must be a TimeTermAppliedShiftResult instance')
    n_traces = _coerce_positive_int(result.n_traces, name='n_traces')
    dt = _coerce_positive_finite_float(result.dt, name='dt')
    expected_shape = (n_traces,)

    valid_pick_mask = _coerce_1d_bool_array(
        result.valid_pick_mask_sorted,
        name='valid_pick_mask_sorted',
        expected_shape=expected_shape,
    )
    final_used_mask = _coerce_1d_bool_array(
        result.final_used_trace_mask_sorted,
        name='final_used_trace_mask_sorted',
        expected_shape=expected_shape,
    )
    rejected_mask = _coerce_1d_bool_array(
        result.rejected_trace_mask_sorted,
        name='rejected_trace_mask_sorted',
        expected_shape=expected_shape,
    )

    node_stats = _stats_payload(_coerce_summary_seconds(result.node_time_term_s) * 1000.0)
    source_stats = _stats_payload(
        _coerce_summary_seconds(
            result.source_node_time_term_s_sorted,
            expected_shape=expected_shape,
            name='source_node_time_term_s_sorted',
        )
        * 1000.0
    )
    receiver_stats = _stats_payload(
        _coerce_summary_seconds(
            result.receiver_node_time_term_s_sorted,
            expected_shape=expected_shape,
            name='receiver_node_time_term_s_sorted',
        )
        * 1000.0
    )
    estimated_stats = _stats_payload(
        _coerce_summary_seconds(
            result.estimated_trace_time_term_delay_s_sorted,
            expected_shape=expected_shape,
            name='estimated_trace_time_term_delay_s_sorted',
        )
        * 1000.0
    )
    weathering_stats = _stats_payload(
        _coerce_summary_seconds(
            result.applied_weathering_shift_s_sorted,
            expected_shape=expected_shape,
            name='applied_weathering_shift_s_sorted',
        )
        * 1000.0
    )
    datum_stats = _stats_payload(
        _coerce_summary_seconds(
            result.datum_trace_shift_s_sorted,
            expected_shape=expected_shape,
            name='datum_trace_shift_s_sorted',
        )
        * 1000.0
    )
    residual_stats = _stats_payload(
        _coerce_summary_seconds(
            result.residual_applied_shift_s_sorted,
            expected_shape=expected_shape,
            name='residual_applied_shift_s_sorted',
        )
        * 1000.0
    )
    final_stats = _stats_payload(
        _coerce_summary_seconds(
            result.final_trace_shift_s_sorted,
            expected_shape=expected_shape,
            name='final_trace_shift_s_sorted',
        )
        * 1000.0
    )

    return {
        'kind': 'time_term_applied_shift',
        'order': str(result.order),
        'sign_convention': str(result.sign_convention),
        'n_traces': int(n_traces),
        'dt': _json_float(dt),
        'n_valid_picks': int(np.count_nonzero(valid_pick_mask)),
        'n_final_used_traces': int(np.count_nonzero(final_used_mask)),
        'n_rejected_traces': int(np.count_nonzero(rejected_mask)),
        'node_time_term_ms': node_stats,
        'source_node_time_term_ms': source_stats,
        'receiver_node_time_term_ms': receiver_stats,
        'estimated_trace_time_term_delay_ms': estimated_stats,
        'applied_weathering_shift_ms': weathering_stats,
        'datum_trace_shift_ms': datum_stats,
        'residual_applied_shift_ms': residual_stats,
        'final_trace_shift_ms': final_stats,
        'max_abs_weathering_shift_ms': weathering_stats['max_abs'],
        'max_abs_final_shift_ms': final_stats['max_abs'],
        'rejected_trace_policy': str(
            result.metadata.get('rejected_trace_policy', 'use_final_model')
        ),
        'delay_to_shift_convention': str(
            result.metadata.get(
                'delay_to_shift_convention',
                DELAY_TO_SHIFT_CONVENTION,
            )
        ),
        'final_shift_convention': str(
            result.metadata.get('final_shift_convention', FINAL_SHIFT_CONVENTION)
        ),
    }


def _validate_options(
    options: TimeTermAppliedShiftOptions | None,
) -> TimeTermAppliedShiftOptions:
    opts = TimeTermAppliedShiftOptions() if options is None else options
    if not isinstance(opts, TimeTermAppliedShiftOptions):
        raise ValueError('options must be a TimeTermAppliedShiftOptions instance')
    include_rejected_traces = _coerce_bool(
        opts.include_rejected_traces,
        name='include_rejected_traces',
    )
    if not include_rejected_traces:
        raise ValueError('include_rejected_traces must be True')
    rejected_trace_policy = _validate_rejected_trace_policy(opts.rejected_trace_policy)
    return TimeTermAppliedShiftOptions(
        max_abs_weathering_shift_ms=_coerce_optional_nonnegative_finite_float(
            opts.max_abs_weathering_shift_ms,
            name='max_abs_weathering_shift_ms',
        ),
        max_abs_final_shift_ms=_coerce_optional_nonnegative_finite_float(
            opts.max_abs_final_shift_ms,
            name='max_abs_final_shift_ms',
        ),
        include_rejected_traces=include_rejected_traces,
        rejected_trace_policy=rejected_trace_policy,
    )


def _validate_inputs(inputs: TimeTermInversionInputs) -> _ValidatedInputs:
    if not isinstance(inputs, TimeTermInversionInputs):
        raise ValueError('inputs must be a TimeTermInversionInputs instance')
    n_traces = _coerce_positive_int(inputs.n_traces, name='inputs.n_traces')
    dt = _coerce_positive_finite_float(inputs.dt, name='inputs.dt')
    n_nodes = _coerce_positive_int(inputs.n_nodes, name='inputs.n_nodes')
    expected_shape = (n_traces,)

    source_node_id = _coerce_1d_integer_int64(
        inputs.source_node_id_sorted,
        name='source_node_id_sorted',
        expected_shape=expected_shape,
    )
    receiver_node_id = _coerce_1d_integer_int64(
        inputs.receiver_node_id_sorted,
        name='receiver_node_id_sorted',
        expected_shape=expected_shape,
    )
    _validate_index_range(source_node_id, n_unique=n_nodes, name='source_node_id_sorted')
    _validate_index_range(
        receiver_node_id,
        n_unique=n_nodes,
        name='receiver_node_id_sorted',
    )

    datum_shift = _coerce_1d_real_numeric_float64(
        inputs.datum_trace_shift_s_sorted,
        name='datum_trace_shift_s_sorted',
        expected_shape=expected_shape,
    )
    residual_shift = _coerce_1d_real_numeric_float64(
        inputs.residual_applied_shift_s_sorted,
        name='residual_applied_shift_s_sorted',
        expected_shape=expected_shape,
    )
    _validate_all_finite(datum_shift, name='datum_trace_shift_s_sorted')
    _validate_all_finite(residual_shift, name='residual_applied_shift_s_sorted')
    valid_pick_mask = _coerce_1d_bool_array(
        inputs.valid_pick_mask_sorted,
        name='valid_pick_mask_sorted',
        expected_shape=expected_shape,
    )

    return _ValidatedInputs(
        n_traces=n_traces,
        dt=dt,
        n_nodes=n_nodes,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        datum_trace_shift_s_sorted=datum_shift,
        residual_applied_shift_s_sorted=residual_shift,
        valid_pick_mask_sorted=valid_pick_mask,
    )


def _extract_final_sparse_result(
    solver_result: TimeTermSparseSolverResult | TimeTermRobustSolverResult,
    *,
    n_traces: int,
) -> _ExtractedSolverResult:
    expected_shape = (n_traces,)
    if isinstance(solver_result, TimeTermRobustSolverResult):
        sparse_result = solver_result.final_solver_result
        if not isinstance(sparse_result, TimeTermSparseSolverResult):
            raise ValueError(
                'final_solver_result must be a TimeTermSparseSolverResult instance'
            )
        final_used_mask = _coerce_1d_bool_array(
            solver_result.final_used_trace_mask_sorted,
            name='final_used_trace_mask_sorted',
            expected_shape=expected_shape,
        )
        rejected_mask = _coerce_1d_bool_array(
            solver_result.final_rejected_trace_mask_sorted,
            name='final_rejected_trace_mask_sorted',
            expected_shape=expected_shape,
        )
        rejected_iteration = _coerce_1d_integer_int64(
            solver_result.rejected_iteration_sorted,
            name='rejected_iteration_sorted',
            expected_shape=expected_shape,
        )
        return _ExtractedSolverResult(
            sparse_result=sparse_result,
            final_used_trace_mask_sorted=final_used_mask,
            rejected_trace_mask_sorted=rejected_mask,
            rejected_iteration_sorted=rejected_iteration,
            solver_result_kind='robust',
        )

    if isinstance(solver_result, TimeTermSparseSolverResult):
        final_used_mask = _coerce_1d_bool_array(
            solver_result.used_trace_mask_sorted,
            name='used_trace_mask_sorted',
            expected_shape=expected_shape,
        )
        return _ExtractedSolverResult(
            sparse_result=solver_result,
            final_used_trace_mask_sorted=final_used_mask,
            rejected_trace_mask_sorted=np.zeros(expected_shape, dtype=bool),
            rejected_iteration_sorted=np.full(expected_shape, -1, dtype=np.int64),
            solver_result_kind='sparse',
        )

    raise ValueError(
        'solver_result must be a TimeTermSparseSolverResult or '
        'TimeTermRobustSolverResult instance'
    )


def _validate_rejected_trace_policy(value: object) -> TimeTermRejectedTracePolicy:
    if value == 'use_final_model':
        return 'use_final_model'
    raise ValueError('unsupported rejected_trace_policy')


def _validate_shift_limit(
    values_s: np.ndarray,
    *,
    max_abs_ms: float | None,
    limit_name: str,
) -> None:
    arr = _coerce_1d_real_numeric_float64(values_s, name=limit_name)
    _validate_all_finite(arr, name=limit_name)
    if arr.size == 0:
        raise ValueError(f'{limit_name} values must be non-empty')
    if max_abs_ms is None:
        return
    max_abs_s = float(np.max(np.abs(arr)))
    if max_abs_s > max_abs_ms / 1000.0:
        raise ValueError(f'{limit_name} exceeded')


def _coerce_summary_seconds(
    values: object,
    *,
    name: str = 'summary values',
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = _coerce_1d_real_numeric_float64(
        values,
        name=name,
        expected_shape=expected_shape,
    )
    _validate_all_finite(arr, name=name)
    return arr


def _coerce_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be bool')
    return bool(value)


def _coerce_optional_nonnegative_finite_float(
    value: object,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None
    return _coerce_nonnegative_finite_float(value, name=name)


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


__all__ = [
    'DELAY_TO_SHIFT_CONVENTION',
    'FINAL_SHIFT_CONVENTION',
    'SIGN_CONVENTION',
    'TimeTermAppliedShiftOptions',
    'TimeTermAppliedShiftResult',
    'TimeTermRejectedTracePolicy',
    'build_time_term_applied_shift_result',
    'compute_estimated_trace_time_term_delay_s',
    'delay_to_applied_shift_s',
    'summarize_time_term_applied_shift_result',
]
