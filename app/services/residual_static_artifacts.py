"""Artifact writer for residual static solve results."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app.services.common.artifact_io import (
    assert_strict_json,
    write_csv_atomic as _common_write_csv_atomic,
    write_json_atomic as _common_write_json_atomic,
    write_npz_atomic as _common_write_npz_atomic,
)
from app.services.residual_static_robust_solver import (
    ResidualStaticRobustSolveResult,
)
from app.services.residual_static_types import ResidualStaticSolverInputs

SOLUTION_NPZ_NAME = 'residual_static_solution.npz'
QC_JSON_NAME = 'residual_static_qc.json'
STATICS_CSV_NAME = 'residual_statics.csv'

SCHEMA_VERSION = 1
SOLUTION_ARTIFACT_KIND = 'residual_static_solution'
QC_ARTIFACT_KIND = 'residual_static_qc'
ORDER = 'trace_store_sorted'
SIGN_CONVENTION = (
    'estimated_trace_delay_s=source_delay_s+receiver_delay_s; '
    'applied_residual_shift_s=-estimated_trace_delay_s; '
    'corrected(t)=raw(t-shift_s)'
)

_CSV_COLUMNS = [
    'sorted_trace_index',
    'key1',
    'key2',
    'source_id',
    'receiver_id',
    'source_index',
    'receiver_index',
    'source_delay_ms',
    'receiver_delay_ms',
    'estimated_trace_delay_ms',
    'applied_residual_shift_ms',
    'pick_time_after_datum_s',
    'moveout_model_time_s',
    'modeled_pick_time_s',
    'residual_before_ms',
    'residual_after_ms',
    'offset',
    'abs_offset',
    'valid_pick',
    'initial_used',
    'final_used',
    'rejected',
    'rejected_iteration',
]


@dataclass(frozen=True)
class ResidualStaticArtifactPaths:
    solution_npz_path: Path
    qc_json_path: Path
    statics_csv_path: Path


@dataclass(frozen=True)
class ResidualStaticFiniteStats:
    count: int
    min: float | None
    max: float | None
    mean: float | None
    median: float | None
    std: float | None
    mad: float | None


@dataclass(frozen=True)
class ResidualStaticArtifactMetadata:
    job_id: str | None = None
    input_file_id: str | None = None
    datum_source_file_id: str | None = None
    datum_job_id: str | None = None
    datum_solution_artifact: str | None = None
    pick_source_kind: str | None = None
    pick_source_artifact: str | None = None
    residual_static_kind: Literal[
        'first_break_residual_statics'
    ] = 'first_break_residual_statics'


@dataclass(frozen=True)
class _ArtifactContext:
    picks_time_s_sorted: np.ndarray
    valid_mask_sorted: np.ndarray
    pick_time_after_datum_s_sorted: np.ndarray
    datum_trace_shift_s_sorted: np.ndarray

    source_id_sorted: np.ndarray
    receiver_id_sorted: np.ndarray
    source_ids: np.ndarray
    receiver_ids: np.ndarray
    source_index_sorted: np.ndarray
    receiver_index_sorted: np.ndarray
    source_valid_pick_counts: np.ndarray
    receiver_valid_pick_counts: np.ndarray
    source_initial_used_pick_counts: np.ndarray
    receiver_initial_used_pick_counts: np.ndarray
    source_final_used_pick_counts: np.ndarray
    receiver_final_used_pick_counts: np.ndarray

    offset_sorted: np.ndarray
    abs_offset_sorted: np.ndarray

    key1_sorted: np.ndarray
    key2_sorted: np.ndarray

    source_delay_s: np.ndarray
    receiver_delay_s: np.ndarray
    source_delay_s_sorted: np.ndarray
    receiver_delay_s_sorted: np.ndarray

    initial_intercept_s: float
    initial_slowness_s_per_offset_unit: float | None
    intercept_s: float
    slowness_s_per_offset_unit: float | None

    moveout_model_time_s_sorted: np.ndarray
    modeled_pick_time_s_sorted: np.ndarray
    estimated_trace_delay_s_sorted: np.ndarray
    applied_residual_shift_s_sorted: np.ndarray
    residual_before_s: np.ndarray
    residual_after_s: np.ndarray

    initial_used_mask_sorted: np.ndarray
    final_used_mask_sorted: np.ndarray
    rejected_mask_sorted: np.ndarray
    rejected_iteration_sorted: np.ndarray

    n_traces: int
    n_sources: int
    n_receivers: int


def summarize_finite_values(values: np.ndarray) -> ResidualStaticFiniteStats:
    """Summarize finite numeric values, returning null-like fields when empty."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError('values must be a 1D array')
    try:
        arr_f64 = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('values must be numeric') from exc

    finite = np.ascontiguousarray(arr_f64[np.isfinite(arr_f64)], dtype=np.float64)
    count = int(finite.shape[0])
    if count == 0:
        return ResidualStaticFiniteStats(
            count=0,
            min=None,
            max=None,
            mean=None,
            median=None,
            std=None,
            mad=None,
        )

    median = float(np.median(finite))
    return ResidualStaticFiniteStats(
        count=count,
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        mean=float(np.mean(finite)),
        median=median,
        std=float(np.std(finite, ddof=0)),
        mad=float(np.median(np.abs(finite - median))),
    )


def build_residual_static_solution_arrays(
    inputs: ResidualStaticSolverInputs,
    robust_result: ResidualStaticRobustSolveResult,
    *,
    metadata: ResidualStaticArtifactMetadata | None = None,
) -> dict[str, np.ndarray]:
    """Build the pickle-free NPZ payload for a residual static solve."""
    context = _build_artifact_context(inputs, robust_result)
    lineage = _metadata_from_inputs(inputs, metadata)
    offset_byte = -1 if inputs.offset_byte is None else inputs.offset_byte
    initial_slowness = _nan_if_none(context.initial_slowness_s_per_offset_unit)
    final_slowness = _nan_if_none(context.slowness_s_per_offset_unit)

    arrays: dict[str, np.ndarray] = {
        'schema_version': _scalar_int(SCHEMA_VERSION),
        'artifact_kind': _scalar_str(SOLUTION_ARTIFACT_KIND),
        'residual_static_kind': _scalar_str(lineage.residual_static_kind),
        'order': _scalar_str(ORDER),
        'sign_convention': _scalar_str(SIGN_CONVENTION),
        'n_traces': _scalar_int(inputs.n_traces),
        'n_samples': _scalar_int(inputs.n_samples),
        'dt': _scalar_float(inputs.dt),
        'key1_byte': _scalar_int(inputs.key1_byte),
        'key2_byte': _scalar_int(inputs.key2_byte),
        'source_id_byte': _scalar_int(inputs.source_id_byte),
        'receiver_id_byte': _scalar_int(inputs.receiver_id_byte),
        'offset_byte': _scalar_int(offset_byte),
        'moveout_model': _scalar_str(inputs.moveout_model),
        'key1_sorted': context.key1_sorted,
        'key2_sorted': context.key2_sorted,
        'picks_time_s_sorted': context.picks_time_s_sorted,
        'valid_mask_sorted': context.valid_mask_sorted,
        'pick_time_after_datum_s_sorted': context.pick_time_after_datum_s_sorted,
        'datum_trace_shift_s_sorted': context.datum_trace_shift_s_sorted,
        'source_id_sorted': context.source_id_sorted,
        'receiver_id_sorted': context.receiver_id_sorted,
        'source_index_sorted': context.source_index_sorted,
        'receiver_index_sorted': context.receiver_index_sorted,
        'offset_sorted': context.offset_sorted,
        'abs_offset_sorted': context.abs_offset_sorted,
        'moveout_model_time_s_sorted': context.moveout_model_time_s_sorted,
        'modeled_pick_time_s_sorted': context.modeled_pick_time_s_sorted,
        'estimated_trace_delay_s_sorted': context.estimated_trace_delay_s_sorted,
        'applied_residual_shift_s_sorted': context.applied_residual_shift_s_sorted,
        'source_delay_s_sorted': context.source_delay_s_sorted,
        'receiver_delay_s_sorted': context.receiver_delay_s_sorted,
        'residual_before_s': context.residual_before_s,
        'residual_after_s': context.residual_after_s,
        'initial_used_mask_sorted': context.initial_used_mask_sorted,
        'used_mask_sorted': context.final_used_mask_sorted,
        'final_used_mask_sorted': context.final_used_mask_sorted,
        'rejected_mask_sorted': context.rejected_mask_sorted,
        'rejected_iteration_sorted': context.rejected_iteration_sorted,
        'source_ids': context.source_ids,
        'source_delay_s': context.source_delay_s,
        'source_valid_pick_counts': context.source_valid_pick_counts,
        'source_initial_used_pick_counts': context.source_initial_used_pick_counts,
        'source_final_used_pick_counts': context.source_final_used_pick_counts,
        'receiver_ids': context.receiver_ids,
        'receiver_delay_s': context.receiver_delay_s,
        'receiver_valid_pick_counts': context.receiver_valid_pick_counts,
        'receiver_initial_used_pick_counts': context.receiver_initial_used_pick_counts,
        'receiver_final_used_pick_counts': context.receiver_final_used_pick_counts,
        'intercept_s': _scalar_float(context.intercept_s),
        'slowness_s_per_offset_unit': _scalar_float(final_slowness),
        'initial_intercept_s': _scalar_float(context.initial_intercept_s),
        'initial_slowness_s_per_offset_unit': _scalar_float(initial_slowness),
        **_lsmr_arrays(
            'initial',
            robust_result.initial_solver_result.diagnostics,
        ),
        **_lsmr_arrays(
            'final',
            robust_result.final_solver_result.diagnostics,
        ),
        **_robust_arrays(robust_result),
    }
    _validate_solution_arrays(context, arrays)
    return arrays


def build_residual_static_qc_payload(
    inputs: ResidualStaticSolverInputs,
    robust_result: ResidualStaticRobustSolveResult,
    *,
    metadata: ResidualStaticArtifactMetadata | None = None,
) -> dict[str, Any]:
    """Build the strict-JSON QC payload for a residual static solve."""
    context = _build_artifact_context(inputs, robust_result)
    lineage = _metadata_from_inputs(inputs, metadata)
    n_valid = int(np.count_nonzero(context.valid_mask_sorted))
    n_initial = int(np.count_nonzero(context.initial_used_mask_sorted))
    n_final = int(np.count_nonzero(context.final_used_mask_sorted))
    n_rejected = int(np.count_nonzero(context.rejected_mask_sorted))
    payload = {
        'schema_version': SCHEMA_VERSION,
        'artifact_kind': QC_ARTIFACT_KIND,
        'order': ORDER,
        'sign_convention': SIGN_CONVENTION,
        'lineage': {
            'job_id': lineage.job_id,
            'input_file_id': lineage.input_file_id,
            'datum_source_file_id': lineage.datum_source_file_id,
            'datum_job_id': lineage.datum_job_id,
            'datum_solution_artifact': lineage.datum_solution_artifact,
            'pick_source_kind': lineage.pick_source_kind,
            'pick_source_artifact': lineage.pick_source_artifact,
        },
        'input': {
            'n_traces': int(inputs.n_traces),
            'n_samples': int(inputs.n_samples),
            'dt': float(inputs.dt),
            'key1_byte': int(inputs.key1_byte),
            'key2_byte': int(inputs.key2_byte),
            'source_id_byte': int(inputs.source_id_byte),
            'receiver_id_byte': int(inputs.receiver_id_byte),
            'offset_byte': None if inputs.offset_byte is None else int(inputs.offset_byte),
            'moveout_model': str(inputs.moveout_model),
        },
        'counts': {
            'n_traces': context.n_traces,
            'n_valid_picks': n_valid,
            'n_invalid_picks': context.n_traces - n_valid,
            'n_initial_used_picks': n_initial,
            'n_final_used_picks': n_final,
            'n_rejected_total': n_rejected,
            'rejected_fraction_of_initial_used': (
                0.0 if n_initial == 0 else float(n_rejected / n_initial)
            ),
            'n_sources': context.n_sources,
            'n_receivers': context.n_receivers,
            'source_valid_pick_counts': _count_stats_payload(
                context.source_valid_pick_counts
            ),
            'receiver_valid_pick_counts': _count_stats_payload(
                context.receiver_valid_pick_counts
            ),
        },
        'moveout': {
            'model': str(inputs.moveout_model),
            'initial': {
                'intercept_s': float(context.initial_intercept_s),
                'slowness_s_per_offset_unit': _float_or_none(
                    context.initial_slowness_s_per_offset_unit
                ),
            },
            'final': {
                'intercept_s': float(context.intercept_s),
                'slowness_s_per_offset_unit': _float_or_none(
                    context.slowness_s_per_offset_unit
                ),
            },
        },
        'solver': {
            **_stabilization_options_payload(
                robust_result.stabilization_options
            ),
            'initial_lsmr': _lsmr_payload(
                robust_result.initial_solver_result.diagnostics
            ),
            'final_lsmr': _lsmr_payload(
                robust_result.final_solver_result.diagnostics
            ),
        },
        'robust': {
            'enabled': bool(robust_result.robust_options.enabled),
            'method': str(robust_result.robust_options.method),
            'max_iterations': int(robust_result.robust_options.max_iterations),
            'threshold': float(robust_result.robust_options.threshold),
            'min_used_fraction': float(robust_result.robust_options.min_used_fraction),
            'stop_reason': str(robust_result.stop_reason),
            'iterations': [
                {
                    'iteration_index': int(summary.iteration_index),
                    'method': str(summary.method),
                    'n_used_before': int(summary.n_used_before),
                    'n_rejected_this_iteration': int(
                        summary.n_rejected_this_iteration
                    ),
                    'n_used_after': int(summary.n_used_after),
                    'residual_center_s': float(summary.residual_center_s),
                    'residual_scale_s': float(summary.residual_scale_s),
                    'residual_cutoff_s': float(summary.residual_cutoff_s),
                    'max_abs_centered_residual_s': float(
                        summary.max_abs_centered_residual_s
                    ),
                    'converged': bool(summary.converged),
                    'stop_reason': summary.stop_reason,
                }
                for summary in robust_result.iteration_summaries
            ],
        },
        'stats': {
            'residual_before_ms_initial_used': _stats_payload(
                summarize_finite_values(
                    context.residual_before_s[context.initial_used_mask_sorted]
                    * 1000.0
                )
            ),
            'residual_after_ms_final_used': _stats_payload(
                summarize_finite_values(
                    context.residual_after_s[context.final_used_mask_sorted] * 1000.0
                )
            ),
            'estimated_trace_delay_ms_all_traces': _stats_payload(
                summarize_finite_values(context.estimated_trace_delay_s_sorted * 1000.0)
            ),
            'applied_residual_shift_ms_all_traces': _stats_payload(
                summarize_finite_values(
                    context.applied_residual_shift_s_sorted * 1000.0
                )
            ),
            'source_delay_ms': _stats_payload(
                summarize_finite_values(context.source_delay_s * 1000.0)
            ),
            'receiver_delay_ms': _stats_payload(
                summarize_finite_values(context.receiver_delay_s * 1000.0)
            ),
            'moveout_model_time_ms_valid_picks': _stats_payload(
                summarize_finite_values(
                    context.moveout_model_time_s_sorted[context.valid_mask_sorted]
                    * 1000.0
                )
            ),
            'pick_time_after_datum_s_valid_picks': _stats_payload(
                summarize_finite_values(
                    context.pick_time_after_datum_s_sorted[
                        context.valid_mask_sorted
                    ]
                )
            ),
        },
        'validation': _validation_payload(context),
    }
    _assert_strict_json_payload(payload)
    return payload


def build_residual_statics_csv_rows(
    inputs: ResidualStaticSolverInputs,
    robust_result: ResidualStaticRobustSolveResult,
) -> list[dict[str, object]]:
    """Build one CSV row per trace in TraceStore sorted order."""
    context = _build_artifact_context(inputs, robust_result)
    rows: list[dict[str, object]] = []
    has_offsets = inputs.moveout_model != 'none'
    for trace_index in range(context.n_traces):
        valid_pick = bool(context.valid_mask_sorted[trace_index])
        rejected = bool(context.rejected_mask_sorted[trace_index])
        rows.append(
            {
                'sorted_trace_index': trace_index,
                'key1': int(context.key1_sorted[trace_index]),
                'key2': int(context.key2_sorted[trace_index]),
                'source_id': int(context.source_id_sorted[trace_index]),
                'receiver_id': int(context.receiver_id_sorted[trace_index]),
                'source_index': int(context.source_index_sorted[trace_index]),
                'receiver_index': int(context.receiver_index_sorted[trace_index]),
                'source_delay_ms': _csv_float(
                    context.source_delay_s_sorted[trace_index] * 1000.0
                ),
                'receiver_delay_ms': _csv_float(
                    context.receiver_delay_s_sorted[trace_index] * 1000.0
                ),
                'estimated_trace_delay_ms': _csv_float(
                    context.estimated_trace_delay_s_sorted[trace_index] * 1000.0
                ),
                'applied_residual_shift_ms': _csv_float(
                    context.applied_residual_shift_s_sorted[trace_index] * 1000.0
                ),
                'pick_time_after_datum_s': _csv_float(
                    context.pick_time_after_datum_s_sorted[trace_index]
                )
                if valid_pick
                else '',
                'moveout_model_time_s': _csv_float(
                    context.moveout_model_time_s_sorted[trace_index]
                ),
                'modeled_pick_time_s': _csv_float(
                    context.modeled_pick_time_s_sorted[trace_index]
                )
                if valid_pick
                else '',
                'residual_before_ms': _csv_float(
                    context.residual_before_s[trace_index] * 1000.0
                )
                if valid_pick
                else '',
                'residual_after_ms': _csv_float(
                    context.residual_after_s[trace_index] * 1000.0
                )
                if valid_pick
                else '',
                'offset': _csv_float(context.offset_sorted[trace_index])
                if has_offsets
                else '',
                'abs_offset': _csv_float(context.abs_offset_sorted[trace_index])
                if has_offsets
                else '',
                'valid_pick': _csv_bool(valid_pick),
                'initial_used': _csv_bool(
                    bool(context.initial_used_mask_sorted[trace_index])
                ),
                'final_used': _csv_bool(
                    bool(context.final_used_mask_sorted[trace_index])
                ),
                'rejected': _csv_bool(rejected),
                'rejected_iteration': int(
                    context.rejected_iteration_sorted[trace_index]
                )
                if rejected
                else '',
            }
        )
    return rows


def write_residual_static_artifacts(
    job_dir: Path,
    inputs: ResidualStaticSolverInputs,
    robust_result: ResidualStaticRobustSolveResult,
    *,
    metadata: ResidualStaticArtifactMetadata | None = None,
) -> ResidualStaticArtifactPaths:
    """Write residual static solution, QC, and CSV artifacts atomically."""
    try:
        job_dir_path = Path(job_dir)
    except TypeError as exc:
        raise ValueError('job_dir must be path-like') from exc

    solution_arrays = build_residual_static_solution_arrays(
        inputs,
        robust_result,
        metadata=metadata,
    )
    qc_payload = build_residual_static_qc_payload(
        inputs,
        robust_result,
        metadata=metadata,
    )
    csv_rows = build_residual_statics_csv_rows(inputs, robust_result)

    job_dir_path.mkdir(parents=True, exist_ok=True)
    paths = ResidualStaticArtifactPaths(
        solution_npz_path=job_dir_path / SOLUTION_NPZ_NAME,
        qc_json_path=job_dir_path / QC_JSON_NAME,
        statics_csv_path=job_dir_path / STATICS_CSV_NAME,
    )
    _write_npz_atomic(paths.solution_npz_path, solution_arrays)
    _write_json_atomic(paths.qc_json_path, qc_payload)
    _write_csv_atomic(paths.statics_csv_path, csv_rows)
    return paths


def _build_artifact_context(
    inputs: ResidualStaticSolverInputs,
    robust_result: ResidualStaticRobustSolveResult,
) -> _ArtifactContext:
    n_traces = _coerce_positive_int(inputs.n_traces, name='n_traces')
    n_samples = _coerce_positive_int(inputs.n_samples, name='n_samples')
    expected_trace_shape = (n_traces,)
    _coerce_positive_finite_float(inputs.dt, name='dt')
    _validate_header_byte(inputs.key1_byte, name='key1_byte')
    _validate_header_byte(inputs.key2_byte, name='key2_byte')
    _validate_header_byte(inputs.source_id_byte, name='source_id_byte')
    _validate_header_byte(inputs.receiver_id_byte, name='receiver_id_byte')
    _validate_optional_header_byte(inputs.offset_byte, name='offset_byte')
    moveout_model = _validate_moveout_model(inputs.moveout_model)
    if n_samples <= 0:
        raise ValueError('n_samples must be greater than 0')

    picks_time = _coerce_1d_float64_allow_nan(
        inputs.picks_time_s_sorted,
        name='picks_time_s_sorted',
        expected_shape=expected_trace_shape,
    )
    valid_mask = _require_1d_bool_array(
        inputs.valid_pick_mask_sorted,
        name='valid_pick_mask_sorted',
        expected_shape=expected_trace_shape,
    )
    pick_time_after_datum = _coerce_1d_float64_allow_nan(
        inputs.pick_time_after_datum_s_sorted,
        name='pick_time_after_datum_s_sorted',
        expected_shape=expected_trace_shape,
    )
    _validate_no_inf(
        pick_time_after_datum,
        name='pick_time_after_datum_s_sorted',
    )
    _validate_finite_at_mask(
        pick_time_after_datum,
        mask=valid_mask,
        name='pick_time_after_datum_s_sorted',
        mask_name='valid picks',
    )
    datum_trace_shift = _coerce_1d_finite_float64(
        inputs.datum_trace_shift_s_sorted,
        name='datum_trace_shift_s_sorted',
        expected_shape=expected_trace_shape,
    )

    source_id_sorted = _coerce_1d_integer_int64(
        inputs.source_id_sorted,
        name='source_id_sorted',
        expected_shape=expected_trace_shape,
    )
    receiver_id_sorted = _coerce_1d_integer_int64(
        inputs.receiver_id_sorted,
        name='receiver_id_sorted',
        expected_shape=expected_trace_shape,
    )
    source_ids = _coerce_1d_integer_int64(
        inputs.source_unique_ids,
        name='source_unique_ids',
    )
    receiver_ids = _coerce_1d_integer_int64(
        inputs.receiver_unique_ids,
        name='receiver_unique_ids',
    )
    n_sources = _non_empty_size(source_ids, name='source_unique_ids')
    n_receivers = _non_empty_size(receiver_ids, name='receiver_unique_ids')
    source_index = _coerce_1d_integer_int64(
        inputs.source_index_sorted,
        name='source_index_sorted',
        expected_shape=expected_trace_shape,
    )
    receiver_index = _coerce_1d_integer_int64(
        inputs.receiver_index_sorted,
        name='receiver_index_sorted',
        expected_shape=expected_trace_shape,
    )
    _validate_index_range(
        source_index,
        n_unique=n_sources,
        name='source_index_sorted',
    )
    _validate_index_range(
        receiver_index,
        n_unique=n_receivers,
        name='receiver_index_sorted',
    )
    source_valid_counts = _coerce_1d_integer_int64(
        inputs.source_valid_pick_counts,
        name='source_valid_pick_counts',
        expected_shape=(n_sources,),
    )
    receiver_valid_counts = _coerce_1d_integer_int64(
        inputs.receiver_valid_pick_counts,
        name='receiver_valid_pick_counts',
        expected_shape=(n_receivers,),
    )
    _validate_nonnegative_array(source_valid_counts, name='source_valid_pick_counts')
    _validate_nonnegative_array(receiver_valid_counts, name='receiver_valid_pick_counts')

    key1 = _coerce_1d_integer_int64(
        inputs.key1_sorted,
        name='key1_sorted',
        expected_shape=expected_trace_shape,
    )
    key2 = _coerce_1d_integer_int64(
        inputs.key2_sorted,
        name='key2_sorted',
        expected_shape=expected_trace_shape,
    )

    if moveout_model == 'linear_abs_offset':
        if inputs.offset_sorted is None or inputs.abs_offset_sorted is None:
            raise ValueError(
                'offset_sorted and abs_offset_sorted are required for linear_abs_offset'
            )
        offset = _coerce_1d_finite_float64(
            inputs.offset_sorted,
            name='offset_sorted',
            expected_shape=expected_trace_shape,
        )
        abs_offset = _coerce_1d_finite_float64(
            inputs.abs_offset_sorted,
            name='abs_offset_sorted',
            expected_shape=expected_trace_shape,
        )
        if np.any(abs_offset < 0.0):
            raise ValueError('abs_offset_sorted must be nonnegative')
    else:
        offset = np.full(n_traces, np.nan, dtype=np.float64)
        abs_offset = np.full(n_traces, np.nan, dtype=np.float64)

    initial_used_mask = _require_1d_bool_array(
        robust_result.initial_used_mask_sorted,
        name='initial_used_mask_sorted',
        expected_shape=expected_trace_shape,
    )
    final_used_mask = _require_1d_bool_array(
        robust_result.final_used_mask_sorted,
        name='final_used_mask_sorted',
        expected_shape=expected_trace_shape,
    )
    rejected_mask = _require_1d_bool_array(
        robust_result.rejected_mask_sorted,
        name='rejected_mask_sorted',
        expected_shape=expected_trace_shape,
    )
    rejected_iteration = _coerce_1d_integer_int64(
        robust_result.rejected_iteration_sorted,
        name='rejected_iteration_sorted',
        expected_shape=expected_trace_shape,
    )

    initial_solver_used = _require_1d_bool_array(
        robust_result.initial_solver_result.used_mask_sorted,
        name='initial_solver_result.used_mask_sorted',
        expected_shape=expected_trace_shape,
    )
    final_solver_used = _require_1d_bool_array(
        robust_result.final_solver_result.used_mask_sorted,
        name='final_solver_result.used_mask_sorted',
        expected_shape=expected_trace_shape,
    )
    _validate_mask_relationships(
        valid_mask=valid_mask,
        initial_used_mask=initial_used_mask,
        final_used_mask=final_used_mask,
        rejected_mask=rejected_mask,
        rejected_iteration=rejected_iteration,
        initial_solver_used=initial_solver_used,
        final_solver_used=final_solver_used,
    )

    initial_parts = robust_result.initial_solver_result.parameter_parts
    final_parts = robust_result.final_solver_result.parameter_parts
    source_delay = _coerce_1d_finite_float64(
        final_parts.source_delay_s,
        name='source_delay_s',
        expected_shape=(n_sources,),
    )
    receiver_delay = _coerce_1d_finite_float64(
        final_parts.receiver_delay_s,
        name='receiver_delay_s',
        expected_shape=(n_receivers,),
    )
    source_delay_sorted = np.ascontiguousarray(
        source_delay[source_index],
        dtype=np.float64,
    )
    receiver_delay_sorted = np.ascontiguousarray(
        receiver_delay[receiver_index],
        dtype=np.float64,
    )

    final_evaluation = robust_result.final_solver_result.model_evaluation
    initial_evaluation = robust_result.initial_solver_result.model_evaluation
    moveout_time = _coerce_1d_finite_float64(
        final_evaluation.moveout_model_time_s_sorted,
        name='moveout_model_time_s_sorted',
        expected_shape=expected_trace_shape,
    )
    estimated_delay = _coerce_1d_finite_float64(
        final_evaluation.estimated_trace_delay_s_sorted,
        name='estimated_trace_delay_s_sorted',
        expected_shape=expected_trace_shape,
    )
    modeled_pick = _coerce_1d_float64_allow_nan(
        final_evaluation.modeled_pick_time_s_sorted,
        name='modeled_pick_time_s_sorted',
        expected_shape=expected_trace_shape,
    )
    _validate_finite_at_mask(
        modeled_pick,
        mask=valid_mask,
        name='modeled_pick_time_s_sorted',
        mask_name='valid picks',
    )
    residual_before = _coerce_1d_float64_allow_nan(
        initial_evaluation.residual_s_sorted,
        name='residual_before_s',
        expected_shape=expected_trace_shape,
    ).copy()
    residual_after = _coerce_1d_float64_allow_nan(
        final_evaluation.residual_s_sorted,
        name='residual_after_s',
        expected_shape=expected_trace_shape,
    ).copy()
    residual_before[~valid_mask] = np.nan
    residual_after[~valid_mask] = np.nan
    _validate_finite_at_mask(
        residual_before,
        mask=initial_used_mask,
        name='residual_before_s',
        mask_name='initial used picks',
    )
    _validate_finite_at_mask(
        residual_after,
        mask=final_used_mask,
        name='residual_after_s',
        mask_name='final used picks',
    )

    applied_shift = np.ascontiguousarray(-estimated_delay, dtype=np.float64)
    if not np.allclose(
        applied_shift,
        -estimated_delay,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            'applied_residual_shift_s_sorted must be the negative estimated delay'
        )

    return _ArtifactContext(
        picks_time_s_sorted=picks_time,
        valid_mask_sorted=valid_mask,
        pick_time_after_datum_s_sorted=pick_time_after_datum,
        datum_trace_shift_s_sorted=datum_trace_shift,
        source_id_sorted=source_id_sorted,
        receiver_id_sorted=receiver_id_sorted,
        source_ids=source_ids,
        receiver_ids=receiver_ids,
        source_index_sorted=source_index,
        receiver_index_sorted=receiver_index,
        source_valid_pick_counts=source_valid_counts,
        receiver_valid_pick_counts=receiver_valid_counts,
        source_initial_used_pick_counts=_index_counts(
            source_index,
            initial_used_mask,
            n_unique=n_sources,
        ),
        receiver_initial_used_pick_counts=_index_counts(
            receiver_index,
            initial_used_mask,
            n_unique=n_receivers,
        ),
        source_final_used_pick_counts=_index_counts(
            source_index,
            final_used_mask,
            n_unique=n_sources,
        ),
        receiver_final_used_pick_counts=_index_counts(
            receiver_index,
            final_used_mask,
            n_unique=n_receivers,
        ),
        offset_sorted=offset,
        abs_offset_sorted=abs_offset,
        key1_sorted=key1,
        key2_sorted=key2,
        source_delay_s=source_delay,
        receiver_delay_s=receiver_delay,
        source_delay_s_sorted=source_delay_sorted,
        receiver_delay_s_sorted=receiver_delay_sorted,
        initial_intercept_s=_coerce_finite_float(
            initial_parts.intercept_s,
            name='initial_intercept_s',
        ),
        initial_slowness_s_per_offset_unit=_coerce_optional_finite_float(
            initial_parts.slowness_s_per_offset_unit,
            name='initial_slowness_s_per_offset_unit',
        ),
        intercept_s=_coerce_finite_float(final_parts.intercept_s, name='intercept_s'),
        slowness_s_per_offset_unit=_coerce_optional_finite_float(
            final_parts.slowness_s_per_offset_unit,
            name='slowness_s_per_offset_unit',
        ),
        moveout_model_time_s_sorted=moveout_time,
        modeled_pick_time_s_sorted=modeled_pick,
        estimated_trace_delay_s_sorted=estimated_delay,
        applied_residual_shift_s_sorted=applied_shift,
        residual_before_s=np.ascontiguousarray(residual_before, dtype=np.float64),
        residual_after_s=np.ascontiguousarray(residual_after, dtype=np.float64),
        initial_used_mask_sorted=initial_used_mask,
        final_used_mask_sorted=final_used_mask,
        rejected_mask_sorted=rejected_mask,
        rejected_iteration_sorted=rejected_iteration,
        n_traces=n_traces,
        n_sources=n_sources,
        n_receivers=n_receivers,
    )


def _metadata_from_inputs(
    inputs: ResidualStaticSolverInputs,
    metadata: ResidualStaticArtifactMetadata | None,
) -> ResidualStaticArtifactMetadata:
    raw_metadata = inputs.metadata if isinstance(inputs.metadata, dict) else {}
    default_metadata = ResidualStaticArtifactMetadata(
        job_id=_optional_str(raw_metadata.get('job_id')),
        input_file_id=_optional_str(inputs.input_file_id),
        datum_source_file_id=_optional_str(inputs.datum_source_file_id),
        datum_job_id=_optional_str(inputs.datum_job_id),
        datum_solution_artifact=_optional_str(
            raw_metadata.get('datum_solution_artifact')
        ),
        pick_source_kind=_optional_str(inputs.pick_source_kind),
        pick_source_artifact=_optional_str(raw_metadata.get('pick_source_artifact')),
    )
    if metadata is None:
        return default_metadata
    return ResidualStaticArtifactMetadata(
        job_id=metadata.job_id if metadata.job_id is not None else default_metadata.job_id,
        input_file_id=metadata.input_file_id
        if metadata.input_file_id is not None
        else default_metadata.input_file_id,
        datum_source_file_id=metadata.datum_source_file_id
        if metadata.datum_source_file_id is not None
        else default_metadata.datum_source_file_id,
        datum_job_id=metadata.datum_job_id
        if metadata.datum_job_id is not None
        else default_metadata.datum_job_id,
        datum_solution_artifact=metadata.datum_solution_artifact
        if metadata.datum_solution_artifact is not None
        else default_metadata.datum_solution_artifact,
        pick_source_kind=metadata.pick_source_kind
        if metadata.pick_source_kind is not None
        else default_metadata.pick_source_kind,
        pick_source_artifact=metadata.pick_source_artifact
        if metadata.pick_source_artifact is not None
        else default_metadata.pick_source_artifact,
        residual_static_kind=metadata.residual_static_kind,
    )


def _validate_mask_relationships(
    *,
    valid_mask: np.ndarray,
    initial_used_mask: np.ndarray,
    final_used_mask: np.ndarray,
    rejected_mask: np.ndarray,
    rejected_iteration: np.ndarray,
    initial_solver_used: np.ndarray,
    final_solver_used: np.ndarray,
) -> None:
    if np.any(initial_used_mask & ~valid_mask):
        raise ValueError('initial_used_mask_sorted must be a subset of valid_mask_sorted')
    if np.any(final_used_mask & ~initial_used_mask):
        raise ValueError(
            'final_used_mask_sorted must be a subset of initial_used_mask_sorted'
        )
    expected_rejected_mask = initial_used_mask & ~final_used_mask
    if not np.array_equal(rejected_mask, expected_rejected_mask):
        raise ValueError(
            'rejected_mask_sorted must equal initial_used_mask_sorted & '
            '~final_used_mask_sorted'
        )
    if np.any(rejected_iteration[~rejected_mask] != -1):
        raise ValueError('rejected_iteration_sorted must be -1 for unrejected traces')
    if np.any(rejected_iteration[rejected_mask] < 0):
        raise ValueError('rejected_iteration_sorted must be nonnegative for rejections')
    if not np.array_equal(initial_solver_used, initial_used_mask):
        raise ValueError(
            'initial_solver_result.used_mask_sorted must match initial_used_mask_sorted'
        )
    if not np.array_equal(final_solver_used, final_used_mask):
        raise ValueError(
            'final_solver_result.used_mask_sorted must match final_used_mask_sorted'
        )


def _validate_solution_arrays(
    context: _ArtifactContext,
    arrays: dict[str, np.ndarray],
) -> None:
    for name, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise ValueError(f'NPZ field {name} has object dtype')
    applied_shift = _coerce_1d_finite_float64(
        arrays['applied_residual_shift_s_sorted'],
        name='applied_residual_shift_s_sorted',
        expected_shape=(context.n_traces,),
    )
    estimated_delay = _coerce_1d_finite_float64(
        arrays['estimated_trace_delay_s_sorted'],
        name='estimated_trace_delay_s_sorted',
        expected_shape=(context.n_traces,),
    )
    if not np.allclose(applied_shift, -estimated_delay, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            'applied_residual_shift_s_sorted must be the negative estimated delay'
        )
    _require_1d_bool_array(
        arrays['used_mask_sorted'],
        name='used_mask_sorted',
        expected_shape=(context.n_traces,),
    )
    _require_1d_bool_array(
        arrays['final_used_mask_sorted'],
        name='final_used_mask_sorted',
        expected_shape=(context.n_traces,),
    )
    _require_1d_bool_array(
        arrays['rejected_mask_sorted'],
        name='rejected_mask_sorted',
        expected_shape=(context.n_traces,),
    )


def _validation_payload(context: _ArtifactContext) -> dict[str, bool]:
    return {
        'valid_mask_shape_ok': context.valid_mask_sorted.shape == (context.n_traces,),
        'initial_used_subset_of_valid': bool(
            not np.any(context.initial_used_mask_sorted & ~context.valid_mask_sorted)
        ),
        'final_used_subset_of_initial_used': bool(
            not np.any(
                context.final_used_mask_sorted & ~context.initial_used_mask_sorted
            )
        ),
        'rejected_mask_consistent': bool(
            np.array_equal(
                context.rejected_mask_sorted,
                context.initial_used_mask_sorted & ~context.final_used_mask_sorted,
            )
        ),
        'applied_shift_is_negative_estimated_delay': bool(
            np.allclose(
                context.applied_residual_shift_s_sorted,
                -context.estimated_trace_delay_s_sorted,
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        'no_object_dtype_in_solution_npz': True,
    }


def _robust_arrays(
    robust_result: ResidualStaticRobustSolveResult,
) -> dict[str, np.ndarray]:
    summaries = robust_result.iteration_summaries
    return {
        'robust_enabled': _scalar_bool(robust_result.robust_options.enabled),
        'robust_method': _scalar_str(robust_result.robust_options.method),
        'robust_stop_reason': _scalar_str(robust_result.stop_reason),
        'robust_max_iterations': _scalar_int(
            robust_result.robust_options.max_iterations
        ),
        'robust_threshold': _scalar_float(robust_result.robust_options.threshold),
        'robust_min_used_fraction': _scalar_float(
            robust_result.robust_options.min_used_fraction
        ),
        'robust_iteration_index': np.asarray(
            [summary.iteration_index for summary in summaries],
            dtype=np.int64,
        ),
        'robust_iteration_n_used_before': np.asarray(
            [summary.n_used_before for summary in summaries],
            dtype=np.int64,
        ),
        'robust_iteration_n_rejected_this_iteration': np.asarray(
            [summary.n_rejected_this_iteration for summary in summaries],
            dtype=np.int64,
        ),
        'robust_iteration_n_used_after': np.asarray(
            [summary.n_used_after for summary in summaries],
            dtype=np.int64,
        ),
        'robust_iteration_residual_center_s': np.asarray(
            [summary.residual_center_s for summary in summaries],
            dtype=np.float64,
        ),
        'robust_iteration_residual_scale_s': np.asarray(
            [summary.residual_scale_s for summary in summaries],
            dtype=np.float64,
        ),
        'robust_iteration_residual_cutoff_s': np.asarray(
            [summary.residual_cutoff_s for summary in summaries],
            dtype=np.float64,
        ),
        'robust_iteration_max_abs_centered_residual_s': np.asarray(
            [summary.max_abs_centered_residual_s for summary in summaries],
            dtype=np.float64,
        ),
        'robust_iteration_converged': np.asarray(
            [summary.converged for summary in summaries],
            dtype=np.bool_,
        ),
        'robust_iteration_stop_reason': np.asarray(
            [
                '' if summary.stop_reason is None else summary.stop_reason
                for summary in summaries
            ],
            dtype=np.str_,
        ),
    }


def _lsmr_arrays(prefix: str, diagnostics: object) -> dict[str, np.ndarray]:
    return {
        f'{prefix}_lsmr_istop': _scalar_int(getattr(diagnostics, 'istop')),
        f'{prefix}_lsmr_itn': _scalar_int(getattr(diagnostics, 'itn')),
        f'{prefix}_lsmr_normr': _scalar_float(getattr(diagnostics, 'normr')),
        f'{prefix}_lsmr_normar': _scalar_float(getattr(diagnostics, 'normar')),
        f'{prefix}_lsmr_norma': _scalar_float(getattr(diagnostics, 'norma')),
        f'{prefix}_lsmr_conda': _scalar_float(getattr(diagnostics, 'conda')),
        f'{prefix}_lsmr_normx': _scalar_float(getattr(diagnostics, 'normx')),
    }


def _lsmr_payload(diagnostics: object) -> dict[str, int | float]:
    return {
        'istop': int(getattr(diagnostics, 'istop')),
        'itn': int(getattr(diagnostics, 'itn')),
        'normr': float(getattr(diagnostics, 'normr')),
        'normar': float(getattr(diagnostics, 'normar')),
        'norma': float(getattr(diagnostics, 'norma')),
        'conda': float(getattr(diagnostics, 'conda')),
        'normx': float(getattr(diagnostics, 'normx')),
    }


def _stabilization_options_payload(options: object) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields(options):
        value = getattr(options, field.name)
        if isinstance(value, (bool, np.bool_)):
            payload[field.name] = bool(value)
        elif isinstance(value, (int, np.integer)):
            payload[field.name] = int(value)
        elif isinstance(value, (float, np.floating)):
            payload[field.name] = float(value)
        else:
            payload[field.name] = str(value)
    return payload


def _stats_payload(stats: ResidualStaticFiniteStats) -> dict[str, Any]:
    return {
        'count': int(stats.count),
        'min': stats.min,
        'max': stats.max,
        'mean': stats.mean,
        'median': stats.median,
        'std': stats.std,
        'mad': stats.mad,
    }


def _count_stats_payload(values: np.ndarray) -> dict[str, int | float]:
    arr = _coerce_1d_integer_int64(values, name='count_stats')
    if arr.size == 0:
        return {'min': 0, 'max': 0, 'mean': 0.0}
    return {
        'min': int(np.min(arr)),
        'max': int(np.max(arr)),
        'mean': float(np.mean(arr)),
    }


def _index_counts(
    indices: np.ndarray,
    mask: np.ndarray,
    *,
    n_unique: int,
) -> np.ndarray:
    return np.ascontiguousarray(
        np.bincount(indices[mask], minlength=n_unique).astype(np.int64),
        dtype=np.int64,
    )


def _coerce_1d_finite_float64(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = _coerce_1d_float64_allow_nan(
        values,
        name=name,
        expected_shape=expected_shape,
    )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must contain only finite values')
    return arr


def _coerce_1d_float64_allow_nan(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{name} must be numeric')
    try:
        arr_f64 = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    return np.ascontiguousarray(arr_f64, dtype=np.float64)


def _require_1d_bool_array(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if not np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{name} must have bool dtype')
    return np.ascontiguousarray(arr, dtype=bool)


def _coerce_1d_integer_int64(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{name} must contain integer values')
    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr, dtype=np.int64)
    try:
        arr_f64 = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain integer values') from exc
    if not np.all(np.isfinite(arr_f64)):
        raise ValueError(f'{name} must contain only finite values')
    if not np.all(arr_f64 == np.rint(arr_f64)):
        raise ValueError(f'{name} must contain integer values')
    return np.ascontiguousarray(arr_f64, dtype=np.int64)


def _coerce_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be finite')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not np.isfinite(out):
        raise ValueError(f'{name} must be finite')
    return out


def _coerce_optional_finite_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _coerce_finite_float(value, name=name)


def _coerce_positive_finite_float(value: object, *, name: str) -> float:
    out = _coerce_finite_float(value, name=name)
    if out <= 0.0:
        raise ValueError(f'{name} must be greater than 0')
    return out


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    out = int(value)
    if out <= 0:
        raise ValueError(f'{name} must be greater than 0')
    return out


def _validate_header_byte(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer SEG-Y trace header byte')
    out = int(value)
    if out < 1 or out > 240:
        raise ValueError(f'{name} must be between 1 and 240')
    return out


def _validate_optional_header_byte(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _validate_header_byte(value, name=name)


def _validate_moveout_model(value: object) -> Literal['linear_abs_offset', 'none']:
    if value == 'linear_abs_offset':
        return 'linear_abs_offset'
    if value == 'none':
        return 'none'
    raise ValueError(f'unsupported moveout_model: {value!r}')


def _validate_index_range(
    indices: np.ndarray,
    *,
    n_unique: int,
    name: str,
) -> None:
    if np.any(indices < 0):
        raise ValueError(f'{name} must be greater than or equal to 0')
    if np.any(indices >= n_unique):
        raise ValueError(f'{name} contains values outside 0..{n_unique - 1}')


def _validate_nonnegative_array(values: np.ndarray, *, name: str) -> None:
    if np.any(values < 0):
        raise ValueError(f'{name} must be nonnegative')


def _validate_no_inf(values: np.ndarray, *, name: str) -> None:
    if np.any(np.isinf(values)):
        raise ValueError(f'{name} contains inf')


def _validate_finite_at_mask(
    values: np.ndarray,
    *,
    mask: np.ndarray,
    name: str,
    mask_name: str,
) -> None:
    if np.any(~np.isfinite(values[mask])):
        raise ValueError(f'{name} must be finite for {mask_name}')


def _non_empty_size(values: np.ndarray, *, name: str) -> int:
    size = int(values.shape[0])
    if size <= 0:
        raise ValueError(f'{name} must be non-empty')
    return size


def _scalar_int(value: object) -> np.ndarray:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('integer scalar expected')
    return np.asarray(int(value), dtype=np.int64)


def _scalar_float(value: object) -> np.ndarray:
    return np.asarray(_coerce_finite_or_nan_float(value), dtype=np.float64)


def _scalar_bool(value: object) -> np.ndarray:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('bool scalar expected')
    return np.asarray(bool(value), dtype=np.bool_)


def _scalar_str(value: object) -> np.ndarray:
    if value is None:
        value = ''
    return np.asarray(str(value), dtype=np.str_)


def _coerce_finite_or_nan_float(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('float scalar expected')
    out = float(value)
    if not np.isfinite(out) and not np.isnan(out):
        raise ValueError('float scalar must be finite or NaN')
    return out


def _nan_if_none(value: float | None) -> float:
    return np.nan if value is None else float(value)


def _float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _csv_bool(value: bool) -> str:
    return 'true' if value else 'false'


def _csv_float(value: object) -> float | str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ''
    if not np.isfinite(out):
        return ''
    return out


def _assert_strict_json_payload(payload: dict[str, Any]) -> None:
    assert_strict_json(payload)


def _write_npz_atomic(out_path: Path, payload: dict[str, np.ndarray]) -> None:
    _common_write_npz_atomic(
        out_path,
        payload,
        compressed=False,
        reject_object_arrays=False,
    )


def _write_json_atomic(out_path: Path, payload: dict[str, Any]) -> None:
    _common_write_json_atomic(
        out_path,
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        trailing_newline=True,
    )


def _write_csv_atomic(out_path: Path, rows: list[dict[str, object]]) -> None:
    _common_write_csv_atomic(
        out_path,
        columns=_CSV_COLUMNS,
        rows=rows,
        extrasaction='raise',
        lineterminator='\r\n',
    )


__all__ = [
    'ResidualStaticArtifactMetadata',
    'ResidualStaticArtifactPaths',
    'ResidualStaticFiniteStats',
    'build_residual_static_qc_payload',
    'build_residual_static_solution_arrays',
    'build_residual_statics_csv_rows',
    'summarize_finite_values',
    'write_residual_static_artifacts',
]
