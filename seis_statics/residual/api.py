"""Public first-break residual statics API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from seis_statics._validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_integer_int64 as _coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
)
from seis_statics.residual.result import FirstBreakResidualStaticsResult
from seis_statics.residual.robust import (
    ResidualStaticRobustOptions,
    solve_residual_static_robust_least_squares,
    validate_residual_static_robust_options,
)
from seis_statics.residual.solver import (
    ResidualStaticLsmrOptions,
    ResidualStaticStabilizationOptions,
    validate_lsmr_options,
    validate_residual_static_stabilization_options,
)
from seis_statics.residual.types import MoveoutModel, ResidualStaticSolverInputs


def solve_first_break_residual_statics(
    *,
    pick_time_s: object | None = None,
    pick_time_after_datum_s: object | None = None,
    valid_pick_mask: object,
    source_id: object | None = None,
    receiver_id: object | None = None,
    source_index: object | None = None,
    receiver_index: object | None = None,
    source_unique_ids: object | None = None,
    receiver_unique_ids: object | None = None,
    offset: object | None = None,
    abs_offset: object | None = None,
    datum_trace_shift_s: object | None = None,
    used_pick_mask: object | None = None,
    moveout_model: MoveoutModel = 'none',
    stabilization_options: ResidualStaticStabilizationOptions
    | Mapping[str, object]
    | None = None,
    robust_options: ResidualStaticRobustOptions | Mapping[str, object] | None = None,
    lsmr_options: ResidualStaticLsmrOptions | Mapping[str, object] | None = None,
    key1: object | None = None,
    key2: object | None = None,
    source_elevation_m: object | None = None,
    receiver_elevation_m: object | None = None,
    dt: float = 1.0,
    n_samples: int = 1,
    key1_byte: int = 0,
    key2_byte: int = 0,
    source_id_byte: int = 0,
    receiver_id_byte: int = 0,
    offset_byte: int | None = None,
    input_file_id: str = '',
    datum_source_file_id: str = '',
    datum_job_id: str = '',
    pick_source_kind: str = '',
    metadata: Mapping[str, Any] | None = None,
) -> FirstBreakResidualStaticsResult:
    """Solve the package-level first-break/datum residual statics workflow."""
    if (pick_time_s is None) == (pick_time_after_datum_s is None):
        raise ValueError(
            'provide exactly one of pick_time_s or pick_time_after_datum_s'
        )
    valid_mask = _coerce_1d_bool_array(valid_pick_mask, name='valid_pick_mask')
    n_traces = int(valid_mask.shape[0])
    if n_traces == 0:
        raise ValueError('valid_pick_mask must be non-empty')

    datum_shift = _datum_trace_shift(datum_trace_shift_s, n_traces=n_traces)
    picks_time = _pick_time(
        pick_time_s,
        pick_time_after_datum_s=pick_time_after_datum_s,
        datum_trace_shift_s=datum_shift,
        valid_mask=valid_mask,
    )
    pick_after_datum = _pick_time_after_datum(
        pick_time_s,
        pick_time_after_datum_s=pick_time_after_datum_s,
        datum_trace_shift_s=datum_shift,
        valid_mask=valid_mask,
    )

    source = _endpoint_index(
        id_values=source_id,
        index_values=source_index,
        unique_id_values=source_unique_ids,
        valid_mask=valid_mask,
        n_traces=n_traces,
        role='source',
    )
    receiver = _endpoint_index(
        id_values=receiver_id,
        index_values=receiver_index,
        unique_id_values=receiver_unique_ids,
        valid_mask=valid_mask,
        n_traces=n_traces,
        role='receiver',
    )
    offset_array, abs_offset_array = _offset_arrays(
        offset=offset,
        abs_offset=abs_offset,
        moveout_model=moveout_model,
        n_traces=n_traces,
    )

    inputs = ResidualStaticSolverInputs(
        picks_time_s_sorted=picks_time,
        valid_pick_mask_sorted=valid_mask,
        pick_time_after_datum_s_sorted=pick_after_datum,
        datum_trace_shift_s_sorted=datum_shift,
        source_id_sorted=source.id_sorted,
        receiver_id_sorted=receiver.id_sorted,
        source_unique_ids=source.unique_ids,
        receiver_unique_ids=receiver.unique_ids,
        source_index_sorted=source.index_sorted,
        receiver_index_sorted=receiver.index_sorted,
        source_valid_pick_counts=source.valid_pick_counts,
        receiver_valid_pick_counts=receiver.valid_pick_counts,
        offset_sorted=offset_array,
        abs_offset_sorted=abs_offset_array,
        key1_sorted=_optional_integer_array(key1, n_traces=n_traces, name='key1'),
        key2_sorted=_optional_integer_array(key2, n_traces=n_traces, name='key2'),
        source_elevation_m_sorted=_optional_float_array(
            source_elevation_m,
            n_traces=n_traces,
            name='source_elevation_m',
        ),
        receiver_elevation_m_sorted=_optional_float_array(
            receiver_elevation_m,
            n_traces=n_traces,
            name='receiver_elevation_m',
        ),
        dt=_coerce_positive_finite_float(dt, name='dt'),
        n_traces=n_traces,
        n_samples=_coerce_positive_int(n_samples, name='n_samples'),
        key1_byte=int(key1_byte),
        key2_byte=int(key2_byte),
        source_id_byte=int(source_id_byte),
        receiver_id_byte=int(receiver_id_byte),
        offset_byte=offset_byte,
        moveout_model=moveout_model,
        input_file_id=str(input_file_id),
        datum_source_file_id=str(datum_source_file_id),
        datum_job_id=str(datum_job_id),
        pick_source_kind=str(pick_source_kind),
        metadata=dict(metadata or {}),
    )
    robust_result = solve_residual_static_robust_least_squares(
        inputs,
        used_mask_sorted=_optional_used_mask(used_pick_mask, n_traces=n_traces),
        stabilization_options=_stabilization_options(stabilization_options),
        robust_options=_robust_options(robust_options),
        lsmr_options=_lsmr_options(lsmr_options),
    )
    return _public_result(robust_result, source=source, receiver=receiver)


class _EndpointIndex:
    def __init__(
        self,
        *,
        id_sorted: np.ndarray,
        unique_ids: np.ndarray,
        index_sorted: np.ndarray,
        valid_pick_counts: np.ndarray,
    ) -> None:
        self.id_sorted = id_sorted
        self.unique_ids = unique_ids
        self.index_sorted = index_sorted
        self.valid_pick_counts = valid_pick_counts


def _endpoint_index(
    *,
    id_values: object | None,
    index_values: object | None,
    unique_id_values: object | None,
    valid_mask: np.ndarray,
    n_traces: int,
    role: str,
) -> _EndpointIndex:
    if (id_values is None) == (index_values is None):
        raise ValueError(f'provide exactly one of {role}_id or {role}_index')
    if id_values is not None:
        ids = _coerce_1d_integer_int64(
            id_values,
            name=f'{role}_id',
            expected_shape=(n_traces,),
        )
        unique_ids, inverse = np.unique(ids, return_inverse=True)
        index = np.ascontiguousarray(inverse, dtype=np.int64)
    else:
        index = _coerce_1d_integer_int64(
            index_values,
            name=f'{role}_index',
            expected_shape=(n_traces,),
        )
        if np.any(index < 0):
            raise ValueError(f'{role}_index must be greater than or equal to 0')
        n_unique = int(np.max(index)) + 1
        if unique_id_values is None:
            unique_ids = np.arange(n_unique, dtype=np.int64)
        else:
            unique_ids = _coerce_1d_integer_int64(
                unique_id_values,
                name=f'{role}_unique_ids',
                expected_shape=(n_unique,),
            )
        ids = np.ascontiguousarray(unique_ids[index], dtype=np.int64)
    counts = np.bincount(
        index[valid_mask],
        minlength=int(unique_ids.shape[0]),
    ).astype(np.int64)
    return _EndpointIndex(
        id_sorted=np.ascontiguousarray(ids, dtype=np.int64),
        unique_ids=np.ascontiguousarray(unique_ids, dtype=np.int64),
        index_sorted=np.ascontiguousarray(index, dtype=np.int64),
        valid_pick_counts=np.ascontiguousarray(counts, dtype=np.int64),
    )


def _datum_trace_shift(value: object | None, *, n_traces: int) -> np.ndarray:
    if value is None:
        return np.zeros(n_traces, dtype=np.float64)
    return _coerce_1d_real_numeric_float64(
        value,
        name='datum_trace_shift_s',
        expected_shape=(n_traces,),
        require_finite=True,
    )


def _pick_time(
    value: object | None,
    *,
    pick_time_after_datum_s: object | None,
    datum_trace_shift_s: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    if value is None:
        after_datum = _coerce_pick_array(
            pick_time_after_datum_s,
            valid_mask=valid_mask,
            name='pick_time_after_datum_s',
        )
        out = after_datum - datum_trace_shift_s
        out[~valid_mask] = np.nan
        return np.ascontiguousarray(out, dtype=np.float64)
    return _coerce_pick_array(value, valid_mask=valid_mask, name='pick_time_s')


def _pick_time_after_datum(
    value: object | None,
    *,
    pick_time_after_datum_s: object | None,
    datum_trace_shift_s: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    if pick_time_after_datum_s is not None:
        return _coerce_pick_array(
            pick_time_after_datum_s,
            valid_mask=valid_mask,
            name='pick_time_after_datum_s',
        )
    pick_time = _coerce_pick_array(value, valid_mask=valid_mask, name='pick_time_s')
    out = pick_time + datum_trace_shift_s
    out[~valid_mask] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)


def _coerce_pick_array(
    value: object,
    *,
    valid_mask: np.ndarray,
    name: str,
) -> np.ndarray:
    n_traces = int(valid_mask.shape[0])
    arr = _coerce_1d_real_numeric_float64(
        value,
        name=name,
        expected_shape=(n_traces,),
    )
    if np.any(~np.isfinite(arr[valid_mask])):
        raise ValueError(f'{name} must be finite for valid picks')
    out = np.ascontiguousarray(arr, dtype=np.float64)
    out[~valid_mask] = np.nan
    return out


def _offset_arrays(
    *,
    offset: object | None,
    abs_offset: object | None,
    moveout_model: MoveoutModel,
    n_traces: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if offset is None:
        offset_array = None
    else:
        offset_array = _coerce_1d_real_numeric_float64(
            offset,
            name='offset',
            expected_shape=(n_traces,),
            require_finite=True,
        )
    if abs_offset is None:
        abs_offset_array = (
            np.ascontiguousarray(np.abs(offset_array), dtype=np.float64)
            if offset_array is not None
            else None
        )
    else:
        abs_offset_array = _coerce_1d_real_numeric_float64(
            abs_offset,
            name='abs_offset',
            expected_shape=(n_traces,),
            require_finite=True,
        )
    if moveout_model == 'linear_abs_offset':
        if abs_offset_array is None:
            raise ValueError('offset or abs_offset is required for linear_abs_offset')
        if np.any(abs_offset_array < 0.0):
            raise ValueError('abs_offset must be non-negative')
    elif moveout_model != 'none':
        raise ValueError(f'unsupported moveout_model: {moveout_model!r}')
    return offset_array, abs_offset_array


def _optional_integer_array(
    value: object | None,
    *,
    n_traces: int,
    name: str,
) -> np.ndarray:
    if value is None:
        return np.arange(n_traces, dtype=np.int64)
    return _coerce_1d_integer_int64(value, name=name, expected_shape=(n_traces,))


def _optional_float_array(
    value: object | None,
    *,
    n_traces: int,
    name: str,
) -> np.ndarray:
    if value is None:
        return np.zeros(n_traces, dtype=np.float64)
    return _coerce_1d_real_numeric_float64(
        value,
        name=name,
        expected_shape=(n_traces,),
        require_finite=True,
    )


def _optional_used_mask(value: object | None, *, n_traces: int) -> np.ndarray | None:
    if value is None:
        return None
    return _coerce_1d_bool_array(
        value,
        name='used_pick_mask',
        expected_shape=(n_traces,),
    )


def _stabilization_options(
    options: ResidualStaticStabilizationOptions | Mapping[str, object] | None,
) -> ResidualStaticStabilizationOptions:
    if options is None:
        return ResidualStaticStabilizationOptions()
    if isinstance(options, Mapping):
        return validate_residual_static_stabilization_options(
            ResidualStaticStabilizationOptions(**options)
        )
    return validate_residual_static_stabilization_options(options)


def _robust_options(
    options: ResidualStaticRobustOptions | Mapping[str, object] | None,
) -> ResidualStaticRobustOptions:
    if options is None:
        return ResidualStaticRobustOptions()
    if isinstance(options, Mapping):
        return validate_residual_static_robust_options(
            ResidualStaticRobustOptions(**options)
        )
    return validate_residual_static_robust_options(options)


def _lsmr_options(
    options: ResidualStaticLsmrOptions | Mapping[str, object] | None,
) -> ResidualStaticLsmrOptions | None:
    if options is None:
        return None
    if isinstance(options, Mapping):
        return validate_lsmr_options(ResidualStaticLsmrOptions(**options))
    return validate_lsmr_options(options)


def _public_result(
    robust_result,
    *,
    source: _EndpointIndex,
    receiver: _EndpointIndex,
) -> FirstBreakResidualStaticsResult:
    final = robust_result.final_solver_result
    parts = final.parameter_parts
    evaluation = final.model_evaluation
    return FirstBreakResidualStaticsResult(
        moveout_model=final.layout.moveout_model,
        source_id=source.unique_ids,
        receiver_id=receiver.unique_ids,
        intercept_s=parts.intercept_s,
        slowness_s_per_offset_unit=parts.slowness_s_per_offset_unit,
        source_delay_s=parts.source_delay_s,
        receiver_delay_s=parts.receiver_delay_s,
        moveout_model_time_s=evaluation.moveout_model_time_s_sorted,
        estimated_trace_delay_s=evaluation.estimated_trace_delay_s_sorted,
        modeled_pick_time_s=evaluation.modeled_pick_time_s_sorted,
        residual_s=evaluation.residual_s_sorted,
        residual_valid_mask=evaluation.residual_valid_mask_sorted,
        initial_used_pick_mask=robust_result.initial_used_mask_sorted,
        used_pick_mask=robust_result.final_used_mask_sorted,
        rejected_pick_mask=robust_result.rejected_mask_sorted,
        rejected_iteration=robust_result.rejected_iteration_sorted,
        diagnostics=final.diagnostics,
        minimum_data=final.minimum_data,
        graph=final.minimum_data.graph,
        stabilization_options=robust_result.stabilization_options,
        robust_options=robust_result.robust_options,
        robust_iteration_summaries=robust_result.iteration_summaries,
        robust_stop_reason=robust_result.stop_reason,
        n_initial_used_picks=robust_result.n_initial_used_picks,
        n_final_used_picks=robust_result.n_final_used_picks,
        n_rejected_total=robust_result.n_rejected_total,
        n_observations=final.n_observations,
        n_model_parameters=final.n_model_parameters,
        n_gauge_rows=final.n_gauge_rows,
        n_damping_rows=final.n_damping_rows,
        max_abs_estimated_delay_s=final.max_abs_estimated_delay_s,
    )


__all__ = ['solve_first_break_residual_statics']
