"""First-break QC calculations after datum static correction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from app.services.first_break_qc_inputs import FirstBreakQcInputs


@dataclass(frozen=True)
class FiniteSeriesStats:
    name: str
    n_total: int
    n_valid: int
    n_nan: int
    min_s: float | None
    max_s: float | None
    mean_s: float | None
    median_s: float | None
    std_s: float | None
    mad_s: float | None


@dataclass(frozen=True)
class CorrelationQc:
    name: str
    x_name: str
    y_name: str
    n_used: int
    r: float | None
    status: Literal['ok', 'insufficient_data', 'constant_input']


@dataclass(frozen=True)
class LinearOffsetFit:
    n_used: int
    intercept_s: float | None
    slowness_s_per_offset_unit: float | None
    r2: float | None
    status: Literal['ok', 'insufficient_data', 'constant_abs_offset']


@dataclass(frozen=True)
class ResidualByKey1:
    key1: int
    n_used: int
    median_s: float | None
    mad_s: float | None
    mean_s: float | None
    std_s: float | None


@dataclass(frozen=True)
class FirstBreakQcMetrics:
    pick_time_after_datum_s_sorted: np.ndarray
    linear_moveout_model_s_sorted: np.ndarray
    residual_after_datum_s_sorted: np.ndarray
    residual_valid_mask_sorted: np.ndarray
    raw_pick_stats: FiniteSeriesStats
    after_datum_pick_stats: FiniteSeriesStats
    residual_stats: FiniteSeriesStats
    correlations: dict[str, CorrelationQc]
    linear_offset_fit: LinearOffsetFit
    residual_by_key1: list[ResidualByKey1]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _ValidatedInputs:
    picks_time_s_sorted: np.ndarray
    valid_pick_mask_sorted: np.ndarray
    datum_trace_shift_s_sorted: np.ndarray
    source_elevation_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray
    offset_sorted: np.ndarray
    key1_sorted: np.ndarray
    key2_sorted: np.ndarray
    dt: float
    n_traces: int
    n_samples: int
    offset_byte: int
    source_kind: str
    metadata: dict[str, Any]


def compute_first_break_qc_metrics(
    inputs: FirstBreakQcInputs,
    *,
    require_linear_offset_model: bool = False,
) -> FirstBreakQcMetrics:
    """Compute first-break QC metrics from sorted-order validated inputs."""
    validated = _validate_first_break_qc_inputs(inputs)
    picks = validated.picks_time_s_sorted
    valid_mask = validated.valid_pick_mask_sorted
    datum_shift = validated.datum_trace_shift_s_sorted
    source_elevation = validated.source_elevation_m_sorted
    receiver_elevation = validated.receiver_elevation_m_sorted
    offset = validated.offset_sorted
    key1 = validated.key1_sorted

    pick_time_after_datum = compute_pick_time_after_datum(
        picks,
        datum_shift,
        valid_mask,
    )
    raw_pick_stats = compute_finite_series_stats(
        'pick_time_raw_s',
        picks,
        valid_mask,
    )
    after_datum_pick_stats = compute_finite_series_stats(
        'pick_time_after_datum_s',
        pick_time_after_datum,
        valid_mask,
    )
    correlations = {
        'raw_pick_vs_source_elevation': compute_pearson_correlation(
            'raw_pick_vs_source_elevation',
            picks,
            source_elevation,
            valid_mask,
            x_name='pick_time_raw_s',
            y_name='source_elevation_m',
        ),
        'raw_pick_vs_receiver_elevation': compute_pearson_correlation(
            'raw_pick_vs_receiver_elevation',
            picks,
            receiver_elevation,
            valid_mask,
            x_name='pick_time_raw_s',
            y_name='receiver_elevation_m',
        ),
        'after_datum_pick_vs_source_elevation': compute_pearson_correlation(
            'after_datum_pick_vs_source_elevation',
            pick_time_after_datum,
            source_elevation,
            valid_mask,
            x_name='pick_time_after_datum_s',
            y_name='source_elevation_m',
        ),
        'after_datum_pick_vs_receiver_elevation': compute_pearson_correlation(
            'after_datum_pick_vs_receiver_elevation',
            pick_time_after_datum,
            receiver_elevation,
            valid_mask,
            x_name='pick_time_after_datum_s',
            y_name='receiver_elevation_m',
        ),
        'after_datum_pick_vs_abs_offset': compute_pearson_correlation(
            'after_datum_pick_vs_abs_offset',
            pick_time_after_datum,
            np.abs(offset),
            valid_mask,
            x_name='pick_time_after_datum_s',
            y_name='abs_offset',
        ),
    }
    linear_offset_fit, linear_moveout_model, residual_after_datum = (
        fit_linear_offset_model(
            pick_time_after_datum,
            offset,
            valid_mask,
            require=require_linear_offset_model,
        )
    )
    residual_valid_mask = valid_mask & np.isfinite(residual_after_datum)
    residual_stats = compute_finite_series_stats(
        'residual_after_datum_s',
        residual_after_datum,
        residual_valid_mask,
    )
    if linear_offset_fit.status == 'ok':
        residual_by_key1 = compute_residual_by_key1(
            residual_after_datum,
            residual_valid_mask,
            key1,
        )
    else:
        residual_by_key1 = []

    metadata: dict[str, object] = {
        'sign_convention': (
            'pick_time_after_datum_s = pick_time_raw_s + datum_trace_shift_s'
        ),
        'order': 'trace_store_sorted',
        'dt': validated.dt,
        'n_traces': validated.n_traces,
        'n_samples': validated.n_samples,
        'offset_byte': validated.offset_byte,
        'source_kind': validated.source_kind,
        'input_metadata': validated.metadata,
    }

    return FirstBreakQcMetrics(
        pick_time_after_datum_s_sorted=pick_time_after_datum,
        linear_moveout_model_s_sorted=linear_moveout_model,
        residual_after_datum_s_sorted=residual_after_datum,
        residual_valid_mask_sorted=residual_valid_mask,
        raw_pick_stats=raw_pick_stats,
        after_datum_pick_stats=after_datum_pick_stats,
        residual_stats=residual_stats,
        correlations=correlations,
        linear_offset_fit=linear_offset_fit,
        residual_by_key1=residual_by_key1,
        metadata=metadata,
    )


def compute_pick_time_after_datum(
    picks_time_s_sorted: np.ndarray,
    datum_trace_shift_s_sorted: np.ndarray,
    valid_pick_mask_sorted: np.ndarray,
) -> np.ndarray:
    """Apply the PR1/PR2 datum-shift sign convention to valid picks."""
    picks = _coerce_1d_real_numeric_float64(
        picks_time_s_sorted,
        name='picks_time_s_sorted',
    )
    datum_shift = _coerce_1d_finite_float64(
        datum_trace_shift_s_sorted,
        name='datum_trace_shift_s_sorted',
        expected_shape=picks.shape,
    )
    valid_mask = _coerce_1d_bool_array(
        valid_pick_mask_sorted,
        name='valid_pick_mask_sorted',
        expected_shape=picks.shape,
    )
    _validate_pick_nan_contract(picks, valid_mask)

    after_datum = np.full(picks.shape, np.nan, dtype=np.float64)
    after_datum[valid_mask] = picks[valid_mask] + datum_shift[valid_mask]
    if not np.all(np.isfinite(after_datum[valid_mask])):
        msg = 'pick_time_after_datum_s_sorted contains NaN or Inf for valid picks'
        raise ValueError(msg)
    return after_datum


def compute_finite_series_stats(
    name: str,
    values: np.ndarray,
    valid_mask: np.ndarray,
) -> FiniteSeriesStats:
    """Summarize finite values selected by ``valid_mask``."""
    arr = _coerce_1d_real_numeric_float64(values, name=name)
    mask = _coerce_1d_bool_array(
        valid_mask,
        name='valid_mask',
        expected_shape=arr.shape,
    )
    if np.any(np.isinf(arr)):
        msg = f'{name} contains inf'
        raise ValueError(msg)

    finite_mask = mask & np.isfinite(arr)
    used = arr[finite_mask]
    n_total = int(arr.shape[0])
    n_valid = int(used.size)
    n_nan = int(np.count_nonzero(np.isnan(arr)))
    if n_valid == 0:
        return FiniteSeriesStats(
            name=name,
            n_total=n_total,
            n_valid=0,
            n_nan=n_nan,
            min_s=None,
            max_s=None,
            mean_s=None,
            median_s=None,
            std_s=None,
            mad_s=None,
        )

    median = float(np.median(used))
    return FiniteSeriesStats(
        name=name,
        n_total=n_total,
        n_valid=n_valid,
        n_nan=n_nan,
        min_s=float(np.min(used)),
        max_s=float(np.max(used)),
        mean_s=float(np.mean(used)),
        median_s=median,
        std_s=float(np.std(used)),
        mad_s=float(np.median(np.abs(used - median))),
    )


def compute_pearson_correlation(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    *,
    x_name: str,
    y_name: str,
) -> CorrelationQc:
    """Compute a Pearson correlation for valid picks using NumPy only."""
    x_arr = _coerce_1d_real_numeric_float64(x, name=x_name)
    y_arr = _coerce_1d_real_numeric_float64(
        y,
        name=y_name,
        expected_shape=x_arr.shape,
    )
    mask = _coerce_1d_bool_array(
        valid_mask,
        name='valid_mask',
        expected_shape=x_arr.shape,
    )
    x_used = x_arr[mask]
    y_used = y_arr[mask]
    n_used = int(x_used.size)
    if n_used < 2:
        return CorrelationQc(
            name=name,
            x_name=x_name,
            y_name=y_name,
            n_used=n_used,
            r=None,
            status='insufficient_data',
        )
    if not np.all(np.isfinite(x_used)) or not np.all(np.isfinite(y_used)):
        msg = f'{name} contains NaN or Inf for valid picks'
        raise ValueError(msg)
    if np.all(x_used == x_used[0]) or np.all(y_used == y_used[0]):
        return CorrelationQc(
            name=name,
            x_name=x_name,
            y_name=y_name,
            n_used=n_used,
            r=None,
            status='constant_input',
        )

    x_centered = x_used - np.mean(x_used)
    y_centered = y_used - np.mean(y_used)
    denominator = float(
        np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered))
    )
    if denominator == 0.0:
        return CorrelationQc(
            name=name,
            x_name=x_name,
            y_name=y_name,
            n_used=n_used,
            r=None,
            status='constant_input',
        )
    r = float(np.sum(x_centered * y_centered) / denominator)
    r = max(-1.0, min(1.0, r))
    return CorrelationQc(
        name=name,
        x_name=x_name,
        y_name=y_name,
        n_used=n_used,
        r=r,
        status='ok',
    )


def fit_linear_offset_model(
    pick_time_after_datum_s_sorted: np.ndarray,
    offset_sorted: np.ndarray,
    valid_pick_mask_sorted: np.ndarray,
    *,
    require: bool = False,
) -> tuple[LinearOffsetFit, np.ndarray, np.ndarray]:
    """Fit ``pick_time_after_datum_s = intercept + slowness * abs(offset)``."""
    picks = _coerce_1d_real_numeric_float64(
        pick_time_after_datum_s_sorted,
        name='pick_time_after_datum_s_sorted',
    )
    offset = _coerce_1d_finite_float64(
        offset_sorted,
        name='offset_sorted',
        expected_shape=picks.shape,
    )
    valid_mask = _coerce_1d_bool_array(
        valid_pick_mask_sorted,
        name='valid_pick_mask_sorted',
        expected_shape=picks.shape,
    )
    x = np.abs(offset[valid_mask])
    y = picks[valid_mask]
    n_used = int(y.size)
    empty_model = np.full(picks.shape, np.nan, dtype=np.float64)
    empty_residual = np.full(picks.shape, np.nan, dtype=np.float64)

    if n_used < 2:
        fit = LinearOffsetFit(
            n_used=n_used,
            intercept_s=None,
            slowness_s_per_offset_unit=None,
            r2=None,
            status='insufficient_data',
        )
        _raise_if_required(fit, require=require)
        return fit, empty_model, empty_residual
    if not np.all(np.isfinite(y)):
        msg = 'pick_time_after_datum_s_sorted contains NaN or Inf for valid picks'
        raise ValueError(msg)
    if np.all(x == x[0]):
        fit = LinearOffsetFit(
            n_used=n_used,
            intercept_s=None,
            slowness_s_per_offset_unit=None,
            r2=None,
            status='constant_abs_offset',
        )
        _raise_if_required(fit, require=require)
        return fit, empty_model, empty_residual

    design = np.column_stack([np.ones_like(x), x])
    intercept, slowness = np.linalg.lstsq(design, y, rcond=None)[0]
    model = np.asarray(intercept + slowness * np.abs(offset), dtype=np.float64)
    residual = np.full(picks.shape, np.nan, dtype=np.float64)
    if (
        not np.isfinite(intercept)
        or not np.isfinite(slowness)
        or not np.all(np.isfinite(model[valid_mask]))
    ):
        msg = 'linear offset model produced NaN or Inf'
        raise ValueError(msg)
    model[~valid_mask] = np.nan
    residual[valid_mask] = picks[valid_mask] - model[valid_mask]
    if not np.all(np.isfinite(residual[valid_mask])):
        msg = 'linear offset model produced NaN or Inf'
        raise ValueError(msg)

    y_hat = model[valid_mask]
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)
    fit = LinearOffsetFit(
        n_used=n_used,
        intercept_s=float(intercept),
        slowness_s_per_offset_unit=float(slowness),
        r2=float(r2),
        status='ok',
    )
    return fit, model, residual


def compute_residual_by_key1(
    residual_after_datum_s_sorted: np.ndarray,
    residual_valid_mask_sorted: np.ndarray,
    key1_sorted: np.ndarray,
) -> list[ResidualByKey1]:
    """Aggregate residual statistics by deterministic ascending key1 value."""
    residual = _coerce_1d_real_numeric_float64(
        residual_after_datum_s_sorted,
        name='residual_after_datum_s_sorted',
    )
    if np.any(np.isinf(residual)):
        msg = 'residual_after_datum_s_sorted contains inf'
        raise ValueError(msg)
    valid_mask = _coerce_1d_bool_array(
        residual_valid_mask_sorted,
        name='residual_valid_mask_sorted',
        expected_shape=residual.shape,
    )
    key1 = _coerce_1d_integer_int64(
        key1_sorted,
        name='key1_sorted',
        expected_shape=residual.shape,
    )

    by_key1: list[ResidualByKey1] = []
    for key1_value in np.unique(key1):
        section_mask = (key1 == key1_value) & valid_mask & np.isfinite(residual)
        used = residual[section_mask]
        if used.size == 0:
            by_key1.append(
                ResidualByKey1(
                    key1=int(key1_value),
                    n_used=0,
                    median_s=None,
                    mad_s=None,
                    mean_s=None,
                    std_s=None,
                )
            )
            continue
        median = float(np.median(used))
        by_key1.append(
            ResidualByKey1(
                key1=int(key1_value),
                n_used=int(used.size),
                median_s=median,
                mad_s=float(np.median(np.abs(used - median))),
                mean_s=float(np.mean(used)),
                std_s=float(np.std(used)),
            )
        )
    return by_key1


def _validate_first_break_qc_inputs(inputs: FirstBreakQcInputs) -> _ValidatedInputs:
    n_traces = _coerce_positive_int(getattr(inputs, 'n_traces', None), name='n_traces')
    expected_shape = (n_traces,)
    n_samples = _coerce_positive_int(
        getattr(inputs, 'n_samples', None),
        name='n_samples',
    )
    dt = _coerce_positive_finite_float(getattr(inputs, 'dt', None), name='dt')
    offset_byte = _validate_header_byte(
        getattr(inputs, 'offset_byte', None),
        name='offset_byte',
    )
    source_kind = getattr(inputs, 'source_kind', None)
    if not isinstance(source_kind, str) or not source_kind:
        msg = 'source_kind must be a non-empty string'
        raise ValueError(msg)
    metadata = getattr(inputs, 'metadata', None)
    if not isinstance(metadata, Mapping):
        msg = 'metadata must be a mapping'
        raise ValueError(msg)

    picks = _coerce_1d_real_numeric_float64(
        getattr(inputs, 'picks_time_s_sorted', None),
        name='picks_time_s_sorted',
        expected_shape=expected_shape,
    )
    valid_mask = _coerce_1d_bool_array(
        getattr(inputs, 'valid_pick_mask_sorted', None),
        name='valid_pick_mask_sorted',
        expected_shape=expected_shape,
    )
    if not np.any(valid_mask):
        msg = 'at least one valid pick is required'
        raise ValueError(msg)
    _validate_pick_nan_contract(picks, valid_mask)

    return _ValidatedInputs(
        picks_time_s_sorted=picks,
        valid_pick_mask_sorted=valid_mask,
        datum_trace_shift_s_sorted=_coerce_1d_finite_float64(
            getattr(inputs, 'datum_trace_shift_s_sorted', None),
            name='datum_trace_shift_s_sorted',
            expected_shape=expected_shape,
        ),
        source_elevation_m_sorted=_coerce_1d_finite_float64(
            getattr(inputs, 'source_elevation_m_sorted', None),
            name='source_elevation_m_sorted',
            expected_shape=expected_shape,
        ),
        receiver_elevation_m_sorted=_coerce_1d_finite_float64(
            getattr(inputs, 'receiver_elevation_m_sorted', None),
            name='receiver_elevation_m_sorted',
            expected_shape=expected_shape,
        ),
        offset_sorted=_coerce_1d_finite_float64(
            getattr(inputs, 'offset_sorted', None),
            name='offset_sorted',
            expected_shape=expected_shape,
        ),
        key1_sorted=_coerce_1d_integer_int64(
            getattr(inputs, 'key1_sorted', None),
            name='key1_sorted',
            expected_shape=expected_shape,
        ),
        key2_sorted=_coerce_1d_integer_int64(
            getattr(inputs, 'key2_sorted', None),
            name='key2_sorted',
            expected_shape=expected_shape,
        ),
        dt=dt,
        n_traces=n_traces,
        n_samples=n_samples,
        offset_byte=offset_byte,
        source_kind=source_kind,
        metadata=dict(metadata),
    )


def _validate_pick_nan_contract(picks: np.ndarray, valid_mask: np.ndarray) -> None:
    if np.any(np.isinf(picks)):
        msg = 'picks_time_s_sorted contains inf'
        raise ValueError(msg)
    if np.any(~np.isfinite(picks[valid_mask])):
        msg = 'valid picks must be finite'
        raise ValueError(msg)
    if np.any(~np.isnan(picks[~valid_mask])):
        msg = 'invalid picks must be NaN'
        raise ValueError(msg)


def _raise_if_required(fit: LinearOffsetFit, *, require: bool) -> None:
    if not require:
        return
    msg = f'linear offset model is undefined: {fit.status}'
    raise ValueError(msg)


def _coerce_1d_finite_float64(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = _coerce_1d_real_numeric_float64(
        values,
        name=name,
        expected_shape=expected_shape,
    )
    if not np.all(np.isfinite(arr)):
        msg = f'{name} must contain only finite values'
        raise ValueError(msg)
    return arr


def _coerce_1d_real_numeric_float64(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        msg = f'{name} must be a 1D array'
        raise ValueError(msg)
    if expected_shape is not None and arr.shape != expected_shape:
        msg = f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        raise ValueError(msg)
    if not _is_real_numeric_dtype(arr.dtype):
        msg = f'{name} must have a numeric dtype'
        raise ValueError(msg)
    return np.ascontiguousarray(arr, dtype=np.float64)


def _coerce_1d_bool_array(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        msg = f'{name} must be a 1D array'
        raise ValueError(msg)
    if arr.shape != expected_shape:
        msg = f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        raise ValueError(msg)
    if not np.issubdtype(arr.dtype, np.bool_):
        msg = f'{name} must have bool dtype'
        raise ValueError(msg)
    return np.ascontiguousarray(arr, dtype=bool)


def _coerce_1d_integer_int64(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        msg = f'{name} must be a 1D array'
        raise ValueError(msg)
    if arr.shape != expected_shape:
        msg = f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        raise ValueError(msg)
    if np.issubdtype(arr.dtype, np.bool_):
        msg = f'{name} must contain integer values'
        raise ValueError(msg)
    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr, dtype=np.int64)
    if not _is_real_numeric_dtype(arr.dtype):
        msg = f'{name} must contain integer values'
        raise ValueError(msg)
    arr_f64 = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr_f64)):
        msg = f'{name} must contain only finite values'
        raise ValueError(msg)
    if not np.all(arr_f64 == np.rint(arr_f64)):
        msg = f'{name} must contain integer values'
        raise ValueError(msg)
    return np.ascontiguousarray(arr_f64, dtype=np.int64)


def _validate_header_byte(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        msg = f'{name} must be an integer SEG-Y trace header byte'
        raise ValueError(msg)
    byte = int(value)
    if byte < 1 or byte > 240:
        msg = f'{name} must be between 1 and 240'
        raise ValueError(msg)
    return byte


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        msg = f'{name} must be an integer'
        raise ValueError(msg)
    out = int(value)
    if out <= 0:
        msg = f'{name} must be greater than 0'
        raise ValueError(msg)
    return out


def _coerce_positive_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        msg = f'{name} must be finite and greater than 0'
        raise ValueError(msg)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        msg = f'{name} must be finite and greater than 0'
        raise ValueError(msg) from exc
    if not np.isfinite(out) or out <= 0.0:
        msg = f'{name} must be finite and greater than 0'
        raise ValueError(msg)
    return out


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype,
        np.complexfloating,
    )


__all__ = [
    'CorrelationQc',
    'FiniteSeriesStats',
    'FirstBreakQcMetrics',
    'LinearOffsetFit',
    'ResidualByKey1',
    'compute_finite_series_stats',
    'compute_first_break_qc_metrics',
    'compute_pearson_correlation',
    'compute_pick_time_after_datum',
    'compute_residual_by_key1',
    'fit_linear_offset_model',
]
