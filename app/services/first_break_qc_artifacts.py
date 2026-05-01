"""Artifact writer for first-break QC results after datum static correction."""

from __future__ import annotations

import csv
import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.first_break_qc_inputs import FirstBreakQcInputs
from app.services.first_break_qc_math import (
    CorrelationQc,
    FiniteSeriesStats,
    FirstBreakQcMetrics,
    LinearOffsetFit,
)

FIRST_BREAK_QC_JSON_NAME = 'first_break_qc.json'
FIRST_BREAK_QC_CSV_NAME = 'first_break_qc.csv'
RESIDUAL_BY_KEY1_CSV_NAME = 'residual_by_key1.csv'

_SIGN_CONVENTION = 'pick_time_after_datum_s = pick_time_raw_s + datum_trace_shift_s'
_ORDER = 'trace_store_sorted'

_TRACE_CSV_COLUMNS = [
    'sorted_trace_index',
    'key1',
    'key2',
    'valid_pick',
    'pick_time_raw_s',
    'datum_trace_shift_s',
    'pick_time_after_datum_s',
    'offset',
    'abs_offset',
    'source_elevation_m',
    'receiver_elevation_m',
    'linear_moveout_model_s',
    'residual_after_datum_s',
]

_RESIDUAL_BY_KEY1_CSV_COLUMNS = [
    'key1',
    'n_traces',
    'n_valid_picks',
    'n_used_residual',
    'residual_median_s',
    'residual_mad_s',
    'residual_mean_s',
    'residual_std_s',
]


@dataclass(frozen=True)
class FirstBreakQcArtifactPaths:
    qc_json: Path
    qc_csv: Path
    residual_by_key1_csv: Path


@dataclass(frozen=True)
class _ValidatedInputs:
    job_dir: Path
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


@dataclass(frozen=True)
class _ValidatedMetrics:
    pick_time_after_datum_s_sorted: np.ndarray
    linear_moveout_model_s_sorted: np.ndarray
    residual_after_datum_s_sorted: np.ndarray
    residual_valid_mask_sorted: np.ndarray
    raw_pick_stats: FiniteSeriesStats
    after_datum_pick_stats: FiniteSeriesStats
    residual_stats: FiniteSeriesStats
    correlations: dict[str, CorrelationQc]
    linear_offset_fit: LinearOffsetFit
    residual_by_key1: dict[int, tuple[int, float | None, float | None, float | None, float | None]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _ResidualKey1Row:
    key1: int
    n_traces: int
    n_valid_picks: int
    n_used_residual: int
    residual_median_s: float | None
    residual_mad_s: float | None
    residual_mean_s: float | None
    residual_std_s: float | None


def write_first_break_qc_artifacts(
    *,
    job_dir: Path,
    inputs: FirstBreakQcInputs,
    metrics: FirstBreakQcMetrics,
    solution_artifact_name: str = 'datum_static_solution.npz',
    pick_source_artifact_name: str | None = None,
) -> FirstBreakQcArtifactPaths:
    """Write first-break QC JSON and CSV artifacts atomically."""
    values = _validate_inputs(job_dir=job_dir, inputs=inputs)
    metric_values = _validate_metrics(metrics=metrics, values=values)
    residual_rows = _build_residual_by_key1_rows(values, metric_values)

    qc_payload = _build_qc_json_payload(
        values=values,
        metrics=metric_values,
        residual_rows=residual_rows,
        solution_artifact_name=solution_artifact_name,
        pick_source_artifact_name=pick_source_artifact_name,
    )
    _assert_json_payload(qc_payload)

    values.job_dir.mkdir(parents=True, exist_ok=True)
    paths = FirstBreakQcArtifactPaths(
        qc_json=values.job_dir / FIRST_BREAK_QC_JSON_NAME,
        qc_csv=values.job_dir / FIRST_BREAK_QC_CSV_NAME,
        residual_by_key1_csv=values.job_dir / RESIDUAL_BY_KEY1_CSV_NAME,
    )

    _write_json_atomic(paths.qc_json, qc_payload)
    _write_trace_csv_atomic(paths.qc_csv, values, metric_values)
    _write_residual_by_key1_csv_atomic(paths.residual_by_key1_csv, residual_rows)
    return paths


def _validate_inputs(*, job_dir: Path, inputs: FirstBreakQcInputs) -> _ValidatedInputs:
    try:
        job_dir_path = Path(job_dir)
    except TypeError as exc:
        msg = 'job_dir must be path-like'
        raise ValueError(msg) from exc

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
    _validate_pick_nan_contract(picks, valid_mask, name='picks_time_s_sorted')

    return _ValidatedInputs(
        job_dir=job_dir_path,
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


def _validate_metrics(
    *,
    metrics: FirstBreakQcMetrics,
    values: _ValidatedInputs,
) -> _ValidatedMetrics:
    expected_shape = (values.n_traces,)
    after_datum = _coerce_1d_real_numeric_float64(
        getattr(metrics, 'pick_time_after_datum_s_sorted', None),
        name='pick_time_after_datum_s_sorted',
        expected_shape=expected_shape,
    )
    _validate_pick_nan_contract(
        after_datum,
        values.valid_pick_mask_sorted,
        name='pick_time_after_datum_s_sorted',
    )

    residual_valid_mask = _coerce_1d_bool_array(
        getattr(metrics, 'residual_valid_mask_sorted', None),
        name='residual_valid_mask_sorted',
        expected_shape=expected_shape,
    )
    if np.any(residual_valid_mask & ~values.valid_pick_mask_sorted):
        msg = 'residual_valid_mask_sorted cannot be true for invalid picks'
        raise ValueError(msg)

    linear_fit = _coerce_linear_offset_fit(
        getattr(metrics, 'linear_offset_fit', None),
    )
    linear_model = _coerce_1d_real_numeric_float64(
        getattr(metrics, 'linear_moveout_model_s_sorted', None),
        name='linear_moveout_model_s_sorted',
        expected_shape=expected_shape,
    )
    residual = _coerce_1d_real_numeric_float64(
        getattr(metrics, 'residual_after_datum_s_sorted', None),
        name='residual_after_datum_s_sorted',
        expected_shape=expected_shape,
    )
    _validate_model_and_residual_arrays(
        linear_fit=linear_fit,
        linear_model=linear_model,
        residual=residual,
        residual_valid_mask=residual_valid_mask,
    )

    raw_stats = _coerce_series_stats(
        getattr(metrics, 'raw_pick_stats', None),
        name='raw_pick_stats',
        expected_n_total=values.n_traces,
    )
    after_stats = _coerce_series_stats(
        getattr(metrics, 'after_datum_pick_stats', None),
        name='after_datum_pick_stats',
        expected_n_total=values.n_traces,
    )
    residual_stats = _coerce_series_stats(
        getattr(metrics, 'residual_stats', None),
        name='residual_stats',
        expected_n_total=values.n_traces,
    )

    correlations = _coerce_correlations(getattr(metrics, 'correlations', None))
    residual_by_key1 = _coerce_residual_by_key1(
        getattr(metrics, 'residual_by_key1', None),
        values=values,
        residual_valid_mask=residual_valid_mask,
    )

    metadata = getattr(metrics, 'metadata', None)
    if not isinstance(metadata, Mapping):
        msg = 'metrics.metadata must be a mapping'
        raise ValueError(msg)

    return _ValidatedMetrics(
        pick_time_after_datum_s_sorted=after_datum,
        linear_moveout_model_s_sorted=linear_model,
        residual_after_datum_s_sorted=residual,
        residual_valid_mask_sorted=residual_valid_mask,
        raw_pick_stats=raw_stats,
        after_datum_pick_stats=after_stats,
        residual_stats=residual_stats,
        correlations=correlations,
        linear_offset_fit=linear_fit,
        residual_by_key1=residual_by_key1,
        metadata=dict(metadata),
    )


def _validate_model_and_residual_arrays(
    *,
    linear_fit: LinearOffsetFit,
    linear_model: np.ndarray,
    residual: np.ndarray,
    residual_valid_mask: np.ndarray,
) -> None:
    if np.any(np.isinf(linear_model)):
        msg = 'linear_moveout_model_s_sorted contains inf'
        raise ValueError(msg)
    if np.any(np.isinf(residual)):
        msg = 'residual_after_datum_s_sorted contains inf'
        raise ValueError(msg)

    if linear_fit.status != 'ok':
        if not np.all(np.isnan(linear_model)):
            msg = 'linear_moveout_model_s_sorted must be all NaN when model is undefined'
            raise ValueError(msg)
        if not np.all(np.isnan(residual)):
            msg = 'residual_after_datum_s_sorted must be all NaN when model is undefined'
            raise ValueError(msg)
        if np.any(residual_valid_mask):
            msg = 'residual_valid_mask_sorted must be all False when model is undefined'
            raise ValueError(msg)
        return

    if not np.all(np.isfinite(linear_model[residual_valid_mask])):
        msg = 'linear_moveout_model_s_sorted must be finite for residual-valid traces'
        raise ValueError(msg)
    if not np.all(np.isfinite(residual[residual_valid_mask])):
        msg = 'residual_after_datum_s_sorted must be finite for residual-valid traces'
        raise ValueError(msg)
    if np.any(~np.isnan(linear_model[~residual_valid_mask])):
        msg = 'linear_moveout_model_s_sorted must be NaN for undefined traces'
        raise ValueError(msg)
    if np.any(~np.isnan(residual[~residual_valid_mask])):
        msg = 'residual_after_datum_s_sorted must be NaN for undefined traces'
        raise ValueError(msg)


def _coerce_series_stats(
    value: object,
    *,
    name: str,
    expected_n_total: int,
) -> FiniteSeriesStats:
    if not isinstance(value, FiniteSeriesStats):
        msg = f'{name} must be FiniteSeriesStats'
        raise ValueError(msg)
    _coerce_nonnegative_int(value.n_total, name=f'{name}.n_total')
    if int(value.n_total) != expected_n_total:
        msg = f'{name}.n_total mismatch: expected {expected_n_total}, got {value.n_total}'
        raise ValueError(msg)
    _coerce_nonnegative_int(value.n_valid, name=f'{name}.n_valid')
    _coerce_nonnegative_int(value.n_nan, name=f'{name}.n_nan')
    for field_name in ('min_s', 'max_s', 'mean_s', 'median_s', 'std_s', 'mad_s'):
        _coerce_optional_finite_float(
            getattr(value, field_name),
            name=f'{name}.{field_name}',
        )
    return value


def _coerce_correlations(value: object) -> dict[str, CorrelationQc]:
    if not isinstance(value, Mapping):
        msg = 'correlations must be a mapping'
        raise ValueError(msg)
    correlations: dict[str, CorrelationQc] = {}
    for key, corr in value.items():
        if not isinstance(key, str) or not key:
            msg = 'correlation keys must be non-empty strings'
            raise ValueError(msg)
        if not isinstance(corr, CorrelationQc):
            msg = f'correlations[{key}] must be CorrelationQc'
            raise ValueError(msg)
        if corr.status not in {'ok', 'insufficient_data', 'constant_input'}:
            msg = f'correlations[{key}].status is invalid'
            raise ValueError(msg)
        _coerce_nonnegative_int(corr.n_used, name=f'correlations[{key}].n_used')
        if corr.status == 'ok':
            _coerce_finite_float(corr.r, name=f'correlations[{key}].r')
        elif corr.r is not None:
            msg = f'correlations[{key}].r must be None when undefined'
            raise ValueError(msg)
        correlations[key] = corr
    return correlations


def _coerce_linear_offset_fit(value: object) -> LinearOffsetFit:
    if not isinstance(value, LinearOffsetFit):
        msg = 'linear_offset_fit must be LinearOffsetFit'
        raise ValueError(msg)
    if value.status not in {'ok', 'insufficient_data', 'constant_abs_offset'}:
        msg = 'linear_offset_fit.status is invalid'
        raise ValueError(msg)
    _coerce_nonnegative_int(value.n_used, name='linear_offset_fit.n_used')
    fields = ('intercept_s', 'slowness_s_per_offset_unit', 'r2')
    if value.status == 'ok':
        for field_name in fields:
            _coerce_finite_float(
                getattr(value, field_name),
                name=f'linear_offset_fit.{field_name}',
            )
    else:
        for field_name in fields:
            if getattr(value, field_name) is not None:
                msg = f'linear_offset_fit.{field_name} must be None when undefined'
                raise ValueError(msg)
    return value


def _coerce_residual_by_key1(
    value: object,
    *,
    values: _ValidatedInputs,
    residual_valid_mask: np.ndarray,
) -> dict[int, tuple[int, float | None, float | None, float | None, float | None]]:
    if not isinstance(value, list):
        msg = 'residual_by_key1 must be a list'
        raise ValueError(msg)
    expected_counts = {
        int(key1): int(np.count_nonzero((values.key1_sorted == key1) & residual_valid_mask))
        for key1 in np.unique(values.key1_sorted)
    }
    rows: dict[int, tuple[int, float | None, float | None, float | None, float | None]] = {}
    for index, row in enumerate(value):
        key1 = _coerce_int_from_attr(row, 'key1', name=f'residual_by_key1[{index}].key1')
        if key1 in rows:
            msg = f'residual_by_key1 contains duplicate key1: {key1}'
            raise ValueError(msg)
        n_used = _coerce_int_from_attr(
            row,
            'n_used',
            name=f'residual_by_key1[{index}].n_used',
        )
        if n_used < 0:
            msg = f'residual_by_key1[{index}].n_used must be greater than or equal to 0'
            raise ValueError(msg)
        if key1 in expected_counts and n_used != expected_counts[key1]:
            msg = f'residual_by_key1[{index}].n_used mismatch for key1 {key1}'
            raise ValueError(msg)
        stats = (
            n_used,
            _coerce_optional_finite_float(
                getattr(row, 'median_s', None),
                name=f'residual_by_key1[{index}].median_s',
            ),
            _coerce_optional_finite_float(
                getattr(row, 'mad_s', None),
                name=f'residual_by_key1[{index}].mad_s',
            ),
            _coerce_optional_finite_float(
                getattr(row, 'mean_s', None),
                name=f'residual_by_key1[{index}].mean_s',
            ),
            _coerce_optional_finite_float(
                getattr(row, 'std_s', None),
                name=f'residual_by_key1[{index}].std_s',
            ),
        )
        if n_used == 0 and any(stat is not None for stat in stats[1:]):
            msg = f'residual_by_key1[{index}] stats must be None when n_used is 0'
            raise ValueError(msg)
        rows[key1] = stats
    return rows


def _build_qc_json_payload(
    *,
    values: _ValidatedInputs,
    metrics: _ValidatedMetrics,
    residual_rows: list[_ResidualKey1Row],
    solution_artifact_name: str,
    pick_source_artifact_name: str | None,
) -> dict[str, Any]:
    n_valid = int(np.count_nonzero(values.valid_pick_mask_sorted))
    n_nan_picks = int(np.count_nonzero(np.isnan(values.picks_time_s_sorted)))
    n_residual_valid = int(np.count_nonzero(metrics.residual_valid_mask_sorted))
    n_residual_nan = int(
        np.count_nonzero(np.isnan(metrics.residual_after_datum_s_sorted))
    )

    return {
        'schema_version': 1,
        'artifact_type': 'first_break_qc',
        'order': _ORDER,
        'sign_convention': _SIGN_CONVENTION,
        'n_traces': values.n_traces,
        'n_samples': values.n_samples,
        'dt': values.dt,
        'pick_source': _pick_source_payload(
            values=values,
            n_valid=n_valid,
            n_nan=n_nan_picks,
            artifact_name=pick_source_artifact_name,
        ),
        'datum_solution': _datum_solution_payload(
            values=values,
            artifact_name=solution_artifact_name,
        ),
        'offset': {
            'offset_byte': values.offset_byte,
            'unit': 'header_value',
            'uses_abs_offset_for_model': True,
        },
        'counts': {
            'n_traces': values.n_traces,
            'n_valid_picks': n_valid,
            'n_nan_picks': n_nan_picks,
            'n_residual_valid': n_residual_valid,
            'n_residual_nan': n_residual_nan,
        },
        'stats': {
            'raw_pick_s': _stats_payload(metrics.raw_pick_stats),
            'after_datum_pick_s': _stats_payload(metrics.after_datum_pick_stats),
            'residual_after_datum_s': _stats_payload(metrics.residual_stats),
        },
        'correlations': {
            key: _correlation_payload(corr)
            for key, corr in sorted(metrics.correlations.items())
        },
        'linear_offset_model': _linear_offset_model_payload(
            metrics.linear_offset_fit
        ),
        'residual_by_key1': {
            'artifact': RESIDUAL_BY_KEY1_CSV_NAME,
            'n_sections': len(residual_rows),
            'n_sections_with_residual': sum(
                1 for row in residual_rows if row.n_used_residual > 0
            ),
        },
        'artifacts': {
            'trace_csv': FIRST_BREAK_QC_CSV_NAME,
            'residual_by_key1_csv': RESIDUAL_BY_KEY1_CSV_NAME,
        },
    }


def _pick_source_payload(
    *,
    values: _ValidatedInputs,
    n_valid: int,
    n_nan: int,
    artifact_name: str | None,
) -> dict[str, Any]:
    metadata = values.metadata.get('pick_source_metadata', {})
    if not isinstance(metadata, Mapping):
        msg = 'pick_source_metadata must be a mapping'
        raise ValueError(msg)
    payload: dict[str, Any] = {
        'kind': values.source_kind,
        'n_valid': n_valid,
        'n_nan': n_nan,
        'metadata': _json_safe_value(metadata, name='pick_source.metadata'),
    }
    if artifact_name is not None:
        payload['artifact'] = str(artifact_name)
    return payload


def _datum_solution_payload(
    *,
    values: _ValidatedInputs,
    artifact_name: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {'artifact': str(artifact_name)}
    for key in (
        'datum_elevation_m',
        'replacement_velocity_m_s',
        'key1_byte',
        'key2_byte',
    ):
        if key not in values.metadata:
            continue
        if key in {'key1_byte', 'key2_byte'}:
            payload[key] = _validate_header_byte(values.metadata[key], name=key)
        else:
            payload[key] = _coerce_finite_float(values.metadata[key], name=key)
    return payload


def _stats_payload(stats: FiniteSeriesStats) -> dict[str, int | float | None]:
    return {
        'n_total': int(stats.n_total),
        'n_valid': int(stats.n_valid),
        'n_nan': int(stats.n_nan),
        'min': _coerce_optional_finite_float(stats.min_s, name=f'{stats.name}.min_s'),
        'max': _coerce_optional_finite_float(stats.max_s, name=f'{stats.name}.max_s'),
        'mean': _coerce_optional_finite_float(
            stats.mean_s,
            name=f'{stats.name}.mean_s',
        ),
        'median': _coerce_optional_finite_float(
            stats.median_s,
            name=f'{stats.name}.median_s',
        ),
        'std': _coerce_optional_finite_float(stats.std_s, name=f'{stats.name}.std_s'),
        'mad': _coerce_optional_finite_float(stats.mad_s, name=f'{stats.name}.mad_s'),
    }


def _correlation_payload(corr: CorrelationQc) -> dict[str, int | float | str | None]:
    return {
        'status': corr.status,
        'n_used': int(corr.n_used),
        'r': _coerce_optional_finite_float(corr.r, name=f'{corr.name}.r'),
        'x_name': corr.x_name,
        'y_name': corr.y_name,
    }


def _linear_offset_model_payload(
    fit: LinearOffsetFit,
) -> dict[str, int | float | str | None]:
    return {
        'status': fit.status,
        'n_used': int(fit.n_used),
        'intercept_s': _coerce_optional_finite_float(
            fit.intercept_s,
            name='linear_offset_fit.intercept_s',
        ),
        'slowness_s_per_offset_unit': _coerce_optional_finite_float(
            fit.slowness_s_per_offset_unit,
            name='linear_offset_fit.slowness_s_per_offset_unit',
        ),
        'r2': _coerce_optional_finite_float(
            fit.r2,
            name='linear_offset_fit.r2',
        ),
    }


def _build_residual_by_key1_rows(
    values: _ValidatedInputs,
    metrics: _ValidatedMetrics,
) -> list[_ResidualKey1Row]:
    rows: list[_ResidualKey1Row] = []
    for key1_value in np.unique(values.key1_sorted):
        key1 = int(key1_value)
        section_mask = values.key1_sorted == key1_value
        n_traces = int(np.count_nonzero(section_mask))
        n_valid_picks = int(
            np.count_nonzero(section_mask & values.valid_pick_mask_sorted)
        )
        n_used_residual = int(
            np.count_nonzero(section_mask & metrics.residual_valid_mask_sorted)
        )
        residual_stats = metrics.residual_by_key1.get(key1)
        if residual_stats is None:
            residual_stats = _compute_residual_stats_for_section(
                metrics.residual_after_datum_s_sorted,
                section_mask & metrics.residual_valid_mask_sorted,
            )
        rows.append(
            _ResidualKey1Row(
                key1=key1,
                n_traces=n_traces,
                n_valid_picks=n_valid_picks,
                n_used_residual=n_used_residual,
                residual_median_s=residual_stats[1],
                residual_mad_s=residual_stats[2],
                residual_mean_s=residual_stats[3],
                residual_std_s=residual_stats[4],
            )
        )
    return rows


def _compute_residual_stats_for_section(
    residual: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, float | None, float | None, float | None, float | None]:
    used = residual[mask & np.isfinite(residual)]
    if used.size == 0:
        return (0, None, None, None, None)
    median = float(np.median(used))
    return (
        int(used.size),
        median,
        float(np.median(np.abs(used - median))),
        float(np.mean(used)),
        float(np.std(used)),
    )


def _write_json_atomic(out_path: Path, payload: dict[str, Any]) -> None:
    def write(tmp_path: Path) -> None:
        with tmp_path.open('w', encoding='utf-8') as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write('\n')

    _atomic_write(out_path, write)


def _write_trace_csv_atomic(
    out_path: Path,
    values: _ValidatedInputs,
    metrics: _ValidatedMetrics,
) -> None:
    def write(tmp_path: Path) -> None:
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(_TRACE_CSV_COLUMNS)
            for index in range(values.n_traces):
                writer.writerow(
                    [
                        str(index),
                        str(int(values.key1_sorted[index])),
                        str(int(values.key2_sorted[index])),
                        'true'
                        if bool(values.valid_pick_mask_sorted[index])
                        else 'false',
                        _format_optional_float(values.picks_time_s_sorted[index]),
                        _format_optional_float(
                            values.datum_trace_shift_s_sorted[index]
                        ),
                        _format_optional_float(
                            metrics.pick_time_after_datum_s_sorted[index]
                        ),
                        _format_optional_float(values.offset_sorted[index]),
                        _format_optional_float(abs(values.offset_sorted[index])),
                        _format_optional_float(values.source_elevation_m_sorted[index]),
                        _format_optional_float(
                            values.receiver_elevation_m_sorted[index]
                        ),
                        _format_optional_float(
                            _defined_trace_float(
                                metrics.linear_moveout_model_s_sorted[index],
                                defined=metrics.residual_valid_mask_sorted[index],
                            )
                        ),
                        _format_optional_float(
                            _defined_trace_float(
                                metrics.residual_after_datum_s_sorted[index],
                                defined=metrics.residual_valid_mask_sorted[index],
                            )
                        ),
                    ]
                )

    _atomic_write(out_path, write)


def _write_residual_by_key1_csv_atomic(
    out_path: Path,
    rows: list[_ResidualKey1Row],
) -> None:
    def write(tmp_path: Path) -> None:
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(_RESIDUAL_BY_KEY1_CSV_COLUMNS)
            for row in rows:
                writer.writerow(
                    [
                        str(row.key1),
                        str(row.n_traces),
                        str(row.n_valid_picks),
                        str(row.n_used_residual),
                        _format_optional_float(row.residual_median_s),
                        _format_optional_float(row.residual_mad_s),
                        _format_optional_float(row.residual_mean_s),
                        _format_optional_float(row.residual_std_s),
                    ]
                )

    _atomic_write(out_path, write)


def _atomic_write(out_path: Path, write: Callable[[Path], None]) -> None:
    tmp_path = out_path.with_name(f'{out_path.name}.tmp-{uuid.uuid4().hex}')
    try:
        write(tmp_path)
        tmp_path.replace(out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _assert_json_payload(payload: dict[str, Any]) -> None:
    json.dumps(payload, allow_nan=False)


def _format_optional_float(value: object) -> str:
    if value is None:
        return ''
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        msg = 'CSV float value must be numeric'
        raise ValueError(msg) from exc
    if not np.isfinite(out):
        return ''
    if out == 0.0:
        out = 0.0
    return np.format_float_positional(
        out,
        precision=12,
        unique=False,
        fractional=False,
        trim='-',
    )


def _defined_trace_float(value: object, *, defined: object) -> object:
    return value if bool(defined) else None


def _json_safe_value(value: object, *, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _coerce_finite_float(value, name=name)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _coerce_finite_float(value, name=name)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            msg = f'{name} must not contain object arrays'
            raise ValueError(msg)
        return [
            _json_safe_value(item, name=f'{name}[]')
            for item in value.reshape(-1).tolist()
        ]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                msg = f'{name} keys must be strings'
                raise ValueError(msg)
            out[key] = _json_safe_value(nested, name=f'{name}.{key}')
        return out
    if isinstance(value, (list, tuple)):
        return [
            _json_safe_value(nested, name=f'{name}[]')
            for nested in value
        ]
    msg = f'{name} contains a non-JSON-serializable value'
    raise ValueError(msg)


def _validate_pick_nan_contract(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    name: str,
) -> None:
    if np.any(np.isinf(values)):
        msg = f'{name} contains inf'
        raise ValueError(msg)
    if np.any(~np.isfinite(values[valid_mask])):
        msg = f'{name} valid picks must be finite'
        raise ValueError(msg)
    if np.any(~np.isnan(values[~valid_mask])):
        msg = f'{name} invalid picks must be NaN'
        raise ValueError(msg)


def _coerce_1d_finite_float64(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
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
    values: object,
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
    if not _is_real_numeric_dtype(arr.dtype):
        msg = f'{name} must have a numeric dtype'
        raise ValueError(msg)
    return np.ascontiguousarray(arr, dtype=np.float64)


def _coerce_1d_bool_array(
    values: object,
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
    values: object,
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
    out = _coerce_nonnegative_int(value, name=name)
    if out <= 0:
        msg = f'{name} must be greater than 0'
        raise ValueError(msg)
    return out


def _coerce_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        msg = f'{name} must be an integer'
        raise ValueError(msg)
    out = int(value)
    if out < 0:
        msg = f'{name} must be greater than or equal to 0'
        raise ValueError(msg)
    return out


def _coerce_int_from_attr(value: object, attr: str, *, name: str) -> int:
    out = getattr(value, attr, None)
    if isinstance(out, (bool, np.bool_)) or not isinstance(out, (int, np.integer)):
        msg = f'{name} must be an integer'
        raise ValueError(msg)
    return int(out)


def _coerce_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        msg = f'{name} must be finite'
        raise ValueError(msg)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        msg = f'{name} must be finite'
        raise ValueError(msg) from exc
    if not np.isfinite(out):
        msg = f'{name} must be finite'
        raise ValueError(msg)
    return out


def _coerce_optional_finite_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _coerce_finite_float(value, name=name)


def _coerce_positive_finite_float(value: object, *, name: str) -> float:
    out = _coerce_finite_float(value, name=name)
    if out <= 0.0:
        msg = f'{name} must be finite and greater than 0'
        raise ValueError(msg)
    return out


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype,
        np.complexfloating,
    )


__all__ = [
    'FIRST_BREAK_QC_CSV_NAME',
    'FIRST_BREAK_QC_JSON_NAME',
    'RESIDUAL_BY_KEY1_CSV_NAME',
    'FirstBreakQcArtifactPaths',
    'write_first_break_qc_artifacts',
]
