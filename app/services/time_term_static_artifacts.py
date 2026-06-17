"""Artifact writer for time-term static inversion results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from app.services.common.artifact_io import (
    assert_strict_json,
    write_csv_atomic as _common_write_csv_atomic,
    write_json_atomic as _common_write_json_atomic,
    write_npz_atomic as _common_write_npz_atomic,
)
from app.services.common.array_validation import (
    coerce_1d_bool_array as _require_1d_bool_array,
    coerce_1d_finite_float64 as _coerce_1d_finite_float64,
    coerce_1d_integer_int64 as _common_coerce_1d_integer_int64,
    coerce_finite_float as _coerce_finite_float,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
    is_real_numeric_dtype as _is_real_numeric_dtype,
)
from seis_statics.time_term.apply_shift import (
    DELAY_TO_SHIFT_CONVENTION,
    FINAL_SHIFT_CONVENTION,
    TimeTermAppliedShiftResult,
)
from seis_statics.time_term import (
    TimeTermDesignMatrix,
    TimeTermInversionInputs,
    TimeTermMoveoutResult,
    TimeTermRobustIteration,
    TimeTermRobustSolverResult,
    TimeTermSparseSolverResult,
    TimeTermSolverSystem,
)

TIME_TERM_STATIC_SOLUTION_NPZ_NAME = 'time_term_static_solution.npz'
TIME_TERM_STATIC_QC_JSON_NAME = 'time_term_static_qc.json'
TIME_TERM_STATICS_CSV_NAME = 'time_term_statics.csv'

SCHEMA_VERSION = 1
SOLUTION_ARTIFACT_KIND = 'time_term_static_solution'
QC_ARTIFACT_KIND = 'time_term_static_qc'
ORDER = 'trace_store_sorted'

_coerce_1d_integer_int64 = partial(
    _common_coerce_1d_integer_int64,
    allow_integer_like_float=False,
)

ESTIMATED_DELAY_SIGN_CONVENTION = (
    'positive delay means observed first-break is late'
)
REJECTED_TRACE_POLICY = 'use_final_model'

_CSV_COLUMNS = [
    'sorted_trace_index',
    'source_id',
    'receiver_id',
    'source_node_id',
    'receiver_node_id',
    'offset_m',
    'source_x_m',
    'source_y_m',
    'receiver_x_m',
    'receiver_y_m',
    'source_elevation_m',
    'receiver_elevation_m',
    'source_depth_m',
    'pick_time_raw_s',
    'valid_pick',
    'pick_time_after_static_s',
    'moveout_time_s',
    'moveout_distance_m',
    'source_node_time_term_ms',
    'receiver_node_time_term_ms',
    'estimated_trace_time_term_delay_ms',
    'applied_weathering_shift_ms',
    'datum_trace_shift_ms',
    'residual_applied_shift_ms',
    'final_trace_shift_ms',
    'final_used',
    'rejected',
    'rejected_iteration',
    'row_index',
    'row_residual_before_ms',
    'row_residual_after_ms',
]


@dataclass(frozen=True)
class TimeTermStaticArtifactPaths:
    solution_npz_path: Path
    qc_json_path: Path
    statics_csv_path: Path


@dataclass(frozen=True)
class TimeTermStaticArtifactMetadata:
    job_id: str | None = None
    input_file_id: str | None = None
    key1_byte: int | None = None
    key2_byte: int | None = None

    pick_source_description: str | None = None
    datum_solution_path: str | None = None
    residual_solution_path: str | None = None
    linkage_artifact_path: str | None = None

    request: dict[str, Any] | None = None
    header_source_segy_path: str | None = None


@dataclass(frozen=True)
class TimeTermFiniteStats:
    count: int
    min: float | None
    max: float | None
    mean: float | None
    median: float | None
    std: float | None
    mad: float | None


@dataclass(frozen=True)
class _SolverContext:
    sparse_result: TimeTermSparseSolverResult
    robust_result: TimeTermRobustSolverResult | None
    solver_result_kind: str
    initial_row_used_mask: np.ndarray
    final_row_used_mask: np.ndarray
    final_row_rejected_mask: np.ndarray
    row_rejected_iteration: np.ndarray
    robust_enabled: bool
    robust_method: str | None
    robust_threshold: float | None
    robust_stop_reason: str
    robust_iterations: tuple[TimeTermRobustIteration, ...]


@dataclass(frozen=True)
class _ArtifactContext:
    metadata: TimeTermStaticArtifactMetadata
    solver: _SolverContext

    n_traces: int
    n_samples: int
    dt: float
    n_nodes: int
    n_observations: int

    sorted_trace_index: np.ndarray
    pick_time_raw_s_sorted: np.ndarray
    valid_pick_mask_sorted: np.ndarray
    pick_time_after_static_s_sorted: np.ndarray

    moveout_time_s_sorted: np.ndarray
    moveout_distance_m_sorted: np.ndarray
    valid_moveout_mask_sorted: np.ndarray

    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    source_node_time_term_s_sorted: np.ndarray
    receiver_node_time_term_s_sorted: np.ndarray

    estimated_trace_time_term_delay_s_sorted: np.ndarray
    applied_weathering_shift_s_sorted: np.ndarray
    datum_trace_shift_s_sorted: np.ndarray
    residual_applied_shift_s_sorted: np.ndarray
    final_trace_shift_s_sorted: np.ndarray

    final_used_trace_mask_sorted: np.ndarray
    rejected_trace_mask_sorted: np.ndarray
    rejected_iteration_sorted: np.ndarray

    source_id_sorted: np.ndarray
    receiver_id_sorted: np.ndarray
    offset_sorted: np.ndarray
    source_x_m_sorted: np.ndarray
    source_y_m_sorted: np.ndarray
    receiver_x_m_sorted: np.ndarray
    receiver_y_m_sorted: np.ndarray
    source_elevation_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray
    source_depth_m_sorted: np.ndarray

    node_id: np.ndarray
    node_time_term_s: np.ndarray
    component_id_by_node: np.ndarray
    source_observation_count_by_node: np.ndarray
    receiver_observation_count_by_node: np.ndarray
    total_observation_count_by_node: np.ndarray

    row_trace_index_sorted: np.ndarray
    trace_to_row_index_sorted: np.ndarray
    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    row_pick_time_after_static_s: np.ndarray
    row_moveout_time_s: np.ndarray
    row_data_s: np.ndarray
    row_estimated_time_term_delay_s: np.ndarray
    row_residual_before_s: np.ndarray
    row_residual_after_s: np.ndarray


def summarize_finite_values(values: np.ndarray) -> TimeTermFiniteStats:
    """Summarize finite numeric values for strict JSON payloads."""
    arr = _coerce_1d_float64_allow_nan(values, name='values')
    finite = np.ascontiguousarray(arr[np.isfinite(arr)], dtype=np.float64)
    count = int(finite.shape[0])
    if count == 0:
        return TimeTermFiniteStats(
            count=0,
            min=None,
            max=None,
            mean=None,
            median=None,
            std=None,
            mad=None,
        )

    median = float(np.median(finite))
    return TimeTermFiniteStats(
        count=count,
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        mean=float(np.mean(finite)),
        median=median,
        std=float(np.std(finite, ddof=0)),
        mad=float(np.median(np.abs(finite - median))),
    )


def build_time_term_solution_arrays(
    *,
    inputs: TimeTermInversionInputs,
    moveout: TimeTermMoveoutResult,
    design: TimeTermDesignMatrix,
    solver_result: TimeTermSparseSolverResult | TimeTermRobustSolverResult,
    applied_shift: TimeTermAppliedShiftResult,
    metadata: TimeTermStaticArtifactMetadata | None = None,
) -> dict[str, np.ndarray]:
    """Build the pickle-free NPZ payload for a time-term static solve."""
    context = _build_artifact_context(
        inputs=inputs,
        moveout=moveout,
        design=design,
        solver_result=solver_result,
        applied_shift=applied_shift,
        metadata=metadata,
    )
    solver = context.solver.sparse_result
    system = solver.system
    lineage = context.metadata

    arrays: dict[str, np.ndarray] = {
        'schema_version': _scalar_int(SCHEMA_VERSION),
        'artifact_kind': _scalar_str(SOLUTION_ARTIFACT_KIND),
        'order': _scalar_str(ORDER),
        'job_id': _scalar_str(lineage.job_id),
        'input_file_id': _scalar_str(lineage.input_file_id),
        'key1_byte': _scalar_int(_require_metadata_int(lineage.key1_byte, 'key1_byte')),
        'key2_byte': _scalar_int(_require_metadata_int(lineage.key2_byte, 'key2_byte')),
        'n_traces': _scalar_int(context.n_traces),
        'n_samples': _scalar_int(context.n_samples),
        'dt': _scalar_float(context.dt),
        'n_nodes': _scalar_int(context.n_nodes),
        'n_observations': _scalar_int(context.n_observations),
        'n_final_used_traces': _scalar_int(
            int(np.count_nonzero(context.final_used_trace_mask_sorted))
        ),
        'n_rejected_traces': _scalar_int(
            int(np.count_nonzero(context.rejected_trace_mask_sorted))
        ),
        'pick_source_description': _scalar_str(lineage.pick_source_description),
        'datum_solution_path': _scalar_str(lineage.datum_solution_path),
        'residual_solution_path': _scalar_str(lineage.residual_solution_path),
        'linkage_artifact_path': _scalar_str(lineage.linkage_artifact_path),
        'header_source_segy_path': _scalar_str(lineage.header_source_segy_path),
        'moveout_model': _scalar_str(moveout.model),
        'refractor_velocity_m_s': _scalar_float(moveout.refractor_velocity_m_s),
        'moveout_distance_source': _scalar_str(moveout.distance_source),
        'solver_name': _scalar_str(solver.solver_name),
        'solver_istop': _scalar_int(solver.solver_istop),
        'solver_iterations': _scalar_int(solver.solver_iterations),
        'solver_stop_message': _scalar_str(solver.solver_message),
        'gauge_mode': _scalar_str(system.gauge_mode),
        'damping_lambda': _scalar_float(system.damping_lambda),
        'robust_enabled': _scalar_bool(context.solver.robust_enabled),
        'robust_method': _scalar_str(context.solver.robust_method),
        'robust_threshold': _scalar_float(_nan_if_none(context.solver.robust_threshold)),
        'robust_stop_reason': _scalar_str(context.solver.robust_stop_reason),
        'robust_n_iterations': _scalar_int(
            _robust_iteration_count(context.solver)
        ),
        'sign_convention': _scalar_str(applied_shift.sign_convention),
        'delay_to_shift_convention': _scalar_str(
            _metadata_str(
                applied_shift.metadata,
                'delay_to_shift_convention',
                DELAY_TO_SHIFT_CONVENTION,
            )
        ),
        'final_shift_convention': _scalar_str(
            _metadata_str(
                applied_shift.metadata,
                'final_shift_convention',
                FINAL_SHIFT_CONVENTION,
            )
        ),
        'rejected_trace_policy': _scalar_str(
            _metadata_str(
                applied_shift.metadata,
                'rejected_trace_policy',
                REJECTED_TRACE_POLICY,
            )
        ),
        'solver_result_kind': _scalar_str(context.solver.solver_result_kind),
        'sorted_trace_index': context.sorted_trace_index,
        'pick_time_raw_s_sorted': context.pick_time_raw_s_sorted,
        'valid_pick_mask_sorted': context.valid_pick_mask_sorted,
        'pick_time_after_static_s_sorted': context.pick_time_after_static_s_sorted,
        'moveout_time_s_sorted': context.moveout_time_s_sorted,
        'moveout_distance_m_sorted': context.moveout_distance_m_sorted,
        'valid_moveout_mask_sorted': context.valid_moveout_mask_sorted,
        'source_node_id_sorted': context.source_node_id_sorted,
        'receiver_node_id_sorted': context.receiver_node_id_sorted,
        'source_node_time_term_s_sorted': context.source_node_time_term_s_sorted,
        'receiver_node_time_term_s_sorted': context.receiver_node_time_term_s_sorted,
        'estimated_trace_time_term_delay_s_sorted': (
            context.estimated_trace_time_term_delay_s_sorted
        ),
        'applied_weathering_shift_s_sorted': context.applied_weathering_shift_s_sorted,
        'datum_trace_shift_s_sorted': context.datum_trace_shift_s_sorted,
        'residual_applied_shift_s_sorted': context.residual_applied_shift_s_sorted,
        'final_trace_shift_s_sorted': context.final_trace_shift_s_sorted,
        'final_used_trace_mask_sorted': context.final_used_trace_mask_sorted,
        'rejected_trace_mask_sorted': context.rejected_trace_mask_sorted,
        'rejected_iteration_sorted': context.rejected_iteration_sorted,
        'source_id_sorted': context.source_id_sorted,
        'receiver_id_sorted': context.receiver_id_sorted,
        'offset_sorted': context.offset_sorted,
        'source_x_m_sorted': context.source_x_m_sorted,
        'source_y_m_sorted': context.source_y_m_sorted,
        'receiver_x_m_sorted': context.receiver_x_m_sorted,
        'receiver_y_m_sorted': context.receiver_y_m_sorted,
        'source_elevation_m_sorted': context.source_elevation_m_sorted,
        'receiver_elevation_m_sorted': context.receiver_elevation_m_sorted,
        'source_depth_m_sorted': context.source_depth_m_sorted,
        'node_id': context.node_id,
        'node_time_term_s': context.node_time_term_s,
        'node_time_term_ms': np.ascontiguousarray(
            context.node_time_term_s * 1000.0,
            dtype=np.float64,
        ),
        'source_observation_count_by_node': (
            context.source_observation_count_by_node
        ),
        'receiver_observation_count_by_node': (
            context.receiver_observation_count_by_node
        ),
        'total_observation_count_by_node': context.total_observation_count_by_node,
        'component_id_by_node': context.component_id_by_node,
        'row_trace_index_sorted': context.row_trace_index_sorted,
        'row_source_node_id': context.row_source_node_id,
        'row_receiver_node_id': context.row_receiver_node_id,
        'row_pick_time_after_static_s': context.row_pick_time_after_static_s,
        'row_moveout_time_s': context.row_moveout_time_s,
        'row_data_s': context.row_data_s,
        'row_estimated_time_term_delay_s': (
            context.row_estimated_time_term_delay_s
        ),
        'row_residual_before_s': context.row_residual_before_s,
        'row_residual_after_s': context.row_residual_after_s,
        'row_residual_after_ms': np.ascontiguousarray(
            context.row_residual_after_s * 1000.0,
            dtype=np.float64,
        ),
        'initial_row_used_mask': context.solver.initial_row_used_mask,
        'final_row_used_mask': context.solver.final_row_used_mask,
        'final_row_rejected_mask': context.solver.final_row_rejected_mask,
        'row_rejected_iteration': context.solver.row_rejected_iteration,
        **_robust_iteration_arrays(context),
    }
    _validate_solution_arrays(context, arrays)
    return arrays


def build_time_term_qc_payload(
    *,
    inputs: TimeTermInversionInputs,
    moveout: TimeTermMoveoutResult,
    design: TimeTermDesignMatrix,
    solver_result: TimeTermSparseSolverResult | TimeTermRobustSolverResult,
    applied_shift: TimeTermAppliedShiftResult,
    metadata: TimeTermStaticArtifactMetadata | None = None,
) -> dict[str, Any]:
    """Build a strict-JSON QC payload for a time-term static solve."""
    context = _build_artifact_context(
        inputs=inputs,
        moveout=moveout,
        design=design,
        solver_result=solver_result,
        applied_shift=applied_shift,
        metadata=metadata,
    )
    solver = context.solver.sparse_result
    system = solver.system
    lineage = context.metadata
    n_valid_picks = int(np.count_nonzero(context.valid_pick_mask_sorted))
    n_final_used = int(np.count_nonzero(context.final_used_trace_mask_sorted))
    n_rejected = int(np.count_nonzero(context.rejected_trace_mask_sorted))

    payload: dict[str, Any] = {
        'schema_version': SCHEMA_VERSION,
        'artifact_kind': QC_ARTIFACT_KIND,
        'order': ORDER,
        'job': {
            'job_id': lineage.job_id,
            'input_file_id': lineage.input_file_id,
        },
        'inputs': {
            'n_traces': context.n_traces,
            'n_samples': context.n_samples,
            'dt': context.dt,
            'key1_byte': lineage.key1_byte,
            'key2_byte': lineage.key2_byte,
            'pick_source_description': lineage.pick_source_description,
            'datum_solution_path': lineage.datum_solution_path,
            'residual_solution_path': lineage.residual_solution_path,
            'linkage_artifact_path': lineage.linkage_artifact_path,
            'header_source_segy_path': lineage.header_source_segy_path,
        },
        'counts': {
            'n_nodes': context.n_nodes,
            'n_observations': context.n_observations,
            'n_valid_picks': n_valid_picks,
            'n_final_used_traces': n_final_used,
            'n_rejected_traces': n_rejected,
            'final_used_fraction': _fraction(n_final_used, context.n_traces),
        },
        'moveout': {
            'model': str(moveout.model),
            'refractor_velocity_m_s': float(moveout.refractor_velocity_m_s),
            'distance_source': str(moveout.distance_source),
            'distance_m': _stats_payload(
                summarize_finite_values(context.moveout_distance_m_sorted)
            ),
            'moveout_time_ms': _stats_payload(
                summarize_finite_values(context.moveout_time_s_sorted * 1000.0)
            ),
        },
        'solver': {
            'solver_name': str(solver.solver_name),
            'solver_istop': int(solver.solver_istop),
            'solver_iterations': int(solver.solver_iterations),
            'solver_stop_message': str(solver.solver_message),
            'gauge_mode': str(system.gauge_mode),
            'damping_lambda': _json_float(system.damping_lambda),
            'rms_residual_before_ms': _json_float(
                _rms(context.row_residual_before_s) * 1000.0
            ),
            'rms_residual_after_ms': _json_float(
                _rms(context.row_residual_after_s) * 1000.0
            ),
            'n_components': int(system.n_components),
        },
        'robust': {
            'enabled': bool(context.solver.robust_enabled),
            'method': context.solver.robust_method,
            'threshold': _optional_json_float(context.solver.robust_threshold),
            'stop_reason': context.solver.robust_stop_reason,
            'n_iterations': _robust_iteration_count(context.solver),
            'iterations': _robust_iteration_payload(context),
        },
        'time_terms': {
            'node_time_term_ms': _stats_payload(
                summarize_finite_values(context.node_time_term_s * 1000.0)
            ),
            'source_node_time_term_ms': _stats_payload(
                summarize_finite_values(
                    context.source_node_time_term_s_sorted * 1000.0
                )
            ),
            'receiver_node_time_term_ms': _stats_payload(
                summarize_finite_values(
                    context.receiver_node_time_term_s_sorted * 1000.0
                )
            ),
            'estimated_trace_time_term_delay_ms': _stats_payload(
                summarize_finite_values(
                    context.estimated_trace_time_term_delay_s_sorted * 1000.0
                )
            ),
            'applied_weathering_shift_ms': _stats_payload(
                summarize_finite_values(
                    context.applied_weathering_shift_s_sorted * 1000.0
                )
            ),
        },
        'components': {
            'datum_trace_shift_ms': _stats_payload(
                summarize_finite_values(context.datum_trace_shift_s_sorted * 1000.0)
            ),
            'residual_applied_shift_ms': _stats_payload(
                summarize_finite_values(
                    context.residual_applied_shift_s_sorted * 1000.0
                )
            ),
            'final_trace_shift_ms': _stats_payload(
                summarize_finite_values(context.final_trace_shift_s_sorted * 1000.0)
            ),
        },
        'sign_convention': {
            'estimated_delay': ESTIMATED_DELAY_SIGN_CONVENTION,
            'applied_shift': _metadata_str(
                applied_shift.metadata,
                'delay_to_shift_convention',
                DELAY_TO_SHIFT_CONVENTION,
            ),
            'trace_shift': _metadata_str(
                applied_shift.metadata,
                'final_shift_convention',
                FINAL_SHIFT_CONVENTION,
            ),
        },
        'request': _json_safe(lineage.request),
    }
    _assert_strict_json_payload(payload)
    return payload


def build_time_term_statics_csv_rows(
    *,
    inputs: TimeTermInversionInputs,
    moveout: TimeTermMoveoutResult,
    design: TimeTermDesignMatrix,
    solver_result: TimeTermSparseSolverResult | TimeTermRobustSolverResult,
    applied_shift: TimeTermAppliedShiftResult,
    metadata: TimeTermStaticArtifactMetadata | None = None,
) -> list[dict[str, object]]:
    """Build one CSV row per trace in TraceStore sorted order."""
    context = _build_artifact_context(
        inputs=inputs,
        moveout=moveout,
        design=design,
        solver_result=solver_result,
        applied_shift=applied_shift,
        metadata=metadata,
    )
    rows: list[dict[str, object]] = []
    for trace_index in range(context.n_traces):
        row_index = int(context.trace_to_row_index_sorted[trace_index])
        has_row = row_index >= 0
        rejected = bool(context.rejected_trace_mask_sorted[trace_index])
        rows.append(
            {
                'sorted_trace_index': trace_index,
                'source_id': int(context.source_id_sorted[trace_index]),
                'receiver_id': int(context.receiver_id_sorted[trace_index]),
                'source_node_id': int(context.source_node_id_sorted[trace_index]),
                'receiver_node_id': int(context.receiver_node_id_sorted[trace_index]),
                'offset_m': _csv_float(context.offset_sorted[trace_index]),
                'source_x_m': _csv_float(context.source_x_m_sorted[trace_index]),
                'source_y_m': _csv_float(context.source_y_m_sorted[trace_index]),
                'receiver_x_m': _csv_float(context.receiver_x_m_sorted[trace_index]),
                'receiver_y_m': _csv_float(context.receiver_y_m_sorted[trace_index]),
                'source_elevation_m': _csv_float(
                    context.source_elevation_m_sorted[trace_index]
                ),
                'receiver_elevation_m': _csv_float(
                    context.receiver_elevation_m_sorted[trace_index]
                ),
                'source_depth_m': _csv_float(
                    context.source_depth_m_sorted[trace_index]
                ),
                'pick_time_raw_s': _csv_float(
                    context.pick_time_raw_s_sorted[trace_index]
                ),
                'valid_pick': _csv_bool(
                    bool(context.valid_pick_mask_sorted[trace_index])
                ),
                'pick_time_after_static_s': _csv_float(
                    context.pick_time_after_static_s_sorted[trace_index]
                ),
                'moveout_time_s': _csv_float(
                    context.moveout_time_s_sorted[trace_index]
                ),
                'moveout_distance_m': _csv_float(
                    context.moveout_distance_m_sorted[trace_index]
                ),
                'source_node_time_term_ms': _csv_float(
                    context.source_node_time_term_s_sorted[trace_index] * 1000.0
                ),
                'receiver_node_time_term_ms': _csv_float(
                    context.receiver_node_time_term_s_sorted[trace_index] * 1000.0
                ),
                'estimated_trace_time_term_delay_ms': _csv_float(
                    context.estimated_trace_time_term_delay_s_sorted[trace_index]
                    * 1000.0
                ),
                'applied_weathering_shift_ms': _csv_float(
                    context.applied_weathering_shift_s_sorted[trace_index] * 1000.0
                ),
                'datum_trace_shift_ms': _csv_float(
                    context.datum_trace_shift_s_sorted[trace_index] * 1000.0
                ),
                'residual_applied_shift_ms': _csv_float(
                    context.residual_applied_shift_s_sorted[trace_index] * 1000.0
                ),
                'final_trace_shift_ms': _csv_float(
                    context.final_trace_shift_s_sorted[trace_index] * 1000.0
                ),
                'final_used': _csv_bool(
                    bool(context.final_used_trace_mask_sorted[trace_index])
                ),
                'rejected': _csv_bool(rejected),
                'rejected_iteration': int(
                    context.rejected_iteration_sorted[trace_index]
                )
                if rejected
                else '',
                'row_index': row_index if has_row else '',
                'row_residual_before_ms': _csv_float(
                    context.row_residual_before_s[row_index] * 1000.0
                )
                if has_row
                else '',
                'row_residual_after_ms': _csv_float(
                    context.row_residual_after_s[row_index] * 1000.0
                )
                if has_row
                else '',
            }
        )
    return rows


def write_time_term_static_artifacts(
    *,
    job_dir: Path,
    inputs: TimeTermInversionInputs,
    moveout: TimeTermMoveoutResult,
    design: TimeTermDesignMatrix,
    solver_result: TimeTermSparseSolverResult | TimeTermRobustSolverResult,
    applied_shift: TimeTermAppliedShiftResult,
    metadata: TimeTermStaticArtifactMetadata | None = None,
) -> TimeTermStaticArtifactPaths:
    """Write time-term solution, QC, and CSV artifacts atomically."""
    try:
        job_dir_path = Path(job_dir)
    except TypeError as exc:
        raise ValueError('job_dir must be path-like') from exc

    solution_arrays = build_time_term_solution_arrays(
        inputs=inputs,
        moveout=moveout,
        design=design,
        solver_result=solver_result,
        applied_shift=applied_shift,
        metadata=metadata,
    )
    qc_payload = build_time_term_qc_payload(
        inputs=inputs,
        moveout=moveout,
        design=design,
        solver_result=solver_result,
        applied_shift=applied_shift,
        metadata=metadata,
    )
    csv_rows = build_time_term_statics_csv_rows(
        inputs=inputs,
        moveout=moveout,
        design=design,
        solver_result=solver_result,
        applied_shift=applied_shift,
        metadata=metadata,
    )

    job_dir_path.mkdir(parents=True, exist_ok=True)
    paths = TimeTermStaticArtifactPaths(
        solution_npz_path=job_dir_path / TIME_TERM_STATIC_SOLUTION_NPZ_NAME,
        qc_json_path=job_dir_path / TIME_TERM_STATIC_QC_JSON_NAME,
        statics_csv_path=job_dir_path / TIME_TERM_STATICS_CSV_NAME,
    )
    _write_time_term_solution_npz(paths.solution_npz_path, solution_arrays)
    _write_time_term_qc_json(paths.qc_json_path, qc_payload)
    _write_time_term_statics_csv(paths.statics_csv_path, csv_rows)
    return paths


def _build_artifact_context(
    *,
    inputs: TimeTermInversionInputs,
    moveout: TimeTermMoveoutResult,
    design: TimeTermDesignMatrix,
    solver_result: TimeTermSparseSolverResult | TimeTermRobustSolverResult,
    applied_shift: TimeTermAppliedShiftResult,
    metadata: TimeTermStaticArtifactMetadata | None,
) -> _ArtifactContext:
    if not isinstance(inputs, TimeTermInversionInputs):
        raise ValueError('inputs must be a TimeTermInversionInputs instance')
    if not isinstance(moveout, TimeTermMoveoutResult):
        raise ValueError('moveout must be a TimeTermMoveoutResult instance')
    if not isinstance(design, TimeTermDesignMatrix):
        raise ValueError('design must be a TimeTermDesignMatrix instance')
    if not isinstance(applied_shift, TimeTermAppliedShiftResult):
        raise ValueError('applied_shift must be a TimeTermAppliedShiftResult instance')

    n_traces = _coerce_positive_int(inputs.n_traces, name='inputs.n_traces')
    n_samples = _coerce_positive_int(inputs.n_samples, name='inputs.n_samples')
    dt = _coerce_positive_finite_float(inputs.dt, name='inputs.dt')
    n_nodes = _coerce_positive_int(inputs.n_nodes, name='inputs.n_nodes')
    expected_trace_shape = (n_traces,)
    expected_node_shape = (n_nodes,)

    if int(design.n_traces) != n_traces:
        raise ValueError('design.n_traces must match inputs.n_traces')
    if int(design.n_nodes) != n_nodes:
        raise ValueError('design.n_nodes must match inputs.n_nodes')
    n_observations = _coerce_positive_int(
        design.n_observations,
        name='design.n_observations',
    )
    expected_row_shape = (n_observations,)
    if design.matrix.shape != (n_observations, n_nodes):
        raise ValueError('design.matrix shape does not match design dimensions')

    valid_pick_mask = _require_1d_bool_array(
        inputs.valid_pick_mask_sorted,
        name='valid_pick_mask_sorted',
        expected_shape=expected_trace_shape,
    )
    pick_time_raw = _coerce_1d_float64_allow_nan(
        inputs.pick_time_raw_s_sorted,
        name='pick_time_raw_s_sorted',
        expected_shape=expected_trace_shape,
    )
    _validate_no_inf(pick_time_raw, name='pick_time_raw_s_sorted')
    _validate_finite_at_mask(
        pick_time_raw,
        mask=valid_pick_mask,
        name='pick_time_raw_s_sorted',
        mask_name='valid picks',
    )
    pick_time_after_static = _coerce_1d_float64_allow_nan(
        inputs.pick_time_after_static_s_sorted,
        name='pick_time_after_static_s_sorted',
        expected_shape=expected_trace_shape,
    )
    _validate_no_inf(
        pick_time_after_static,
        name='pick_time_after_static_s_sorted',
    )
    _validate_finite_at_mask(
        pick_time_after_static,
        mask=valid_pick_mask,
        name='pick_time_after_static_s_sorted',
        mask_name='valid picks',
    )
    datum_shift = _coerce_1d_finite_float64(
        inputs.datum_trace_shift_s_sorted,
        name='datum_trace_shift_s_sorted',
        expected_shape=expected_trace_shape,
    )
    residual_shift = _coerce_1d_finite_float64(
        inputs.residual_applied_shift_s_sorted,
        name='residual_applied_shift_s_sorted',
        expected_shape=expected_trace_shape,
    )

    source_node_id = _coerce_1d_integer_int64(
        inputs.source_node_id_sorted,
        name='source_node_id_sorted',
        expected_shape=expected_trace_shape,
    )
    receiver_node_id = _coerce_1d_integer_int64(
        inputs.receiver_node_id_sorted,
        name='receiver_node_id_sorted',
        expected_shape=expected_trace_shape,
    )
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

    source_id = _coerce_1d_integer_int64(
        inputs.source_id_sorted,
        name='source_id_sorted',
        expected_shape=expected_trace_shape,
    )
    receiver_id = _coerce_1d_integer_int64(
        inputs.receiver_id_sorted,
        name='receiver_id_sorted',
        expected_shape=expected_trace_shape,
    )
    offset = _optional_offset_array(inputs.offset_sorted, n_traces=n_traces)
    source_x = _coerce_1d_float64_allow_nan_no_inf(
        inputs.source_x_m_sorted,
        name='source_x_m_sorted',
        expected_shape=expected_trace_shape,
    )
    source_y = _coerce_1d_float64_allow_nan_no_inf(
        inputs.source_y_m_sorted,
        name='source_y_m_sorted',
        expected_shape=expected_trace_shape,
    )
    receiver_x = _coerce_1d_float64_allow_nan_no_inf(
        inputs.receiver_x_m_sorted,
        name='receiver_x_m_sorted',
        expected_shape=expected_trace_shape,
    )
    receiver_y = _coerce_1d_float64_allow_nan_no_inf(
        inputs.receiver_y_m_sorted,
        name='receiver_y_m_sorted',
        expected_shape=expected_trace_shape,
    )
    source_elevation = _coerce_1d_float64_allow_nan_no_inf(
        inputs.source_elevation_m_sorted,
        name='source_elevation_m_sorted',
        expected_shape=expected_trace_shape,
    )
    receiver_elevation = _coerce_1d_float64_allow_nan_no_inf(
        inputs.receiver_elevation_m_sorted,
        name='receiver_elevation_m_sorted',
        expected_shape=expected_trace_shape,
    )
    source_depth = _coerce_1d_float64_allow_nan_no_inf(
        inputs.source_depth_m_sorted,
        name='source_depth_m_sorted',
        expected_shape=expected_trace_shape,
    )

    moveout_time = _coerce_1d_float64_allow_nan_no_inf(
        moveout.moveout_time_s_sorted,
        name='moveout_time_s_sorted',
        expected_shape=expected_trace_shape,
    )
    moveout_distance = _coerce_1d_float64_allow_nan_no_inf(
        moveout.distance_m_sorted,
        name='moveout_distance_m_sorted',
        expected_shape=expected_trace_shape,
    )
    valid_moveout_mask = _require_1d_bool_array(
        moveout.valid_moveout_mask_sorted,
        name='valid_moveout_mask_sorted',
        expected_shape=expected_trace_shape,
    )
    _validate_finite_at_mask(
        moveout_time,
        mask=valid_moveout_mask,
        name='moveout_time_s_sorted',
        mask_name='valid moveout traces',
    )
    _validate_finite_at_mask(
        moveout_distance,
        mask=valid_moveout_mask,
        name='moveout_distance_m_sorted',
        mask_name='valid moveout traces',
    )

    solver = _extract_solver_context(
        solver_result,
        n_traces=n_traces,
        n_observations=n_observations,
    )
    node_time_term = _coerce_1d_finite_float64(
        applied_shift.node_time_term_s,
        name='node_time_term_s',
        expected_shape=expected_node_shape,
    )
    sparse_node_time_term = _coerce_1d_finite_float64(
        solver.sparse_result.node_time_term_s,
        name='solver.node_time_term_s',
        expected_shape=expected_node_shape,
    )
    if not np.allclose(node_time_term, sparse_node_time_term, rtol=0.0, atol=1.0e-12):
        raise ValueError('node_time_term_s must match final solver result')

    source_node_time_term = _coerce_1d_finite_float64(
        applied_shift.source_node_time_term_s_sorted,
        name='source_node_time_term_s_sorted',
        expected_shape=expected_trace_shape,
    )
    receiver_node_time_term = _coerce_1d_finite_float64(
        applied_shift.receiver_node_time_term_s_sorted,
        name='receiver_node_time_term_s_sorted',
        expected_shape=expected_trace_shape,
    )
    expected_source_terms = np.ascontiguousarray(
        node_time_term[source_node_id],
        dtype=np.float64,
    )
    expected_receiver_terms = np.ascontiguousarray(
        node_time_term[receiver_node_id],
        dtype=np.float64,
    )
    if not np.allclose(
        source_node_time_term,
        expected_source_terms,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError('source_node_time_term_s_sorted does not match node ids')
    if not np.allclose(
        receiver_node_time_term,
        expected_receiver_terms,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError('receiver_node_time_term_s_sorted does not match node ids')

    estimated_delay = _coerce_1d_finite_float64(
        applied_shift.estimated_trace_time_term_delay_s_sorted,
        name='estimated_trace_time_term_delay_s_sorted',
        expected_shape=expected_trace_shape,
    )
    expected_delay = np.ascontiguousarray(
        source_node_time_term + receiver_node_time_term,
        dtype=np.float64,
    )
    if not np.allclose(estimated_delay, expected_delay, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            'estimated_trace_time_term_delay_s_sorted does not match node terms'
        )
    applied_weathering_shift = _coerce_1d_finite_float64(
        applied_shift.applied_weathering_shift_s_sorted,
        name='applied_weathering_shift_s_sorted',
        expected_shape=expected_trace_shape,
    )
    if not np.allclose(
        applied_weathering_shift,
        -estimated_delay,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            'applied_weathering_shift_s_sorted must equal '
            '-estimated_trace_time_term_delay_s_sorted'
        )

    applied_datum_shift = _coerce_1d_finite_float64(
        applied_shift.datum_trace_shift_s_sorted,
        name='applied_shift.datum_trace_shift_s_sorted',
        expected_shape=expected_trace_shape,
    )
    applied_residual_shift = _coerce_1d_finite_float64(
        applied_shift.residual_applied_shift_s_sorted,
        name='applied_shift.residual_applied_shift_s_sorted',
        expected_shape=expected_trace_shape,
    )
    if not np.allclose(applied_datum_shift, datum_shift, rtol=0.0, atol=1.0e-12):
        raise ValueError('datum_trace_shift_s_sorted must match applied_shift')
    if not np.allclose(
        applied_residual_shift,
        residual_shift,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError('residual_applied_shift_s_sorted must match applied_shift')
    final_shift = _coerce_1d_finite_float64(
        applied_shift.final_trace_shift_s_sorted,
        name='final_trace_shift_s_sorted',
        expected_shape=expected_trace_shape,
    )
    expected_final_shift = np.ascontiguousarray(
        datum_shift + residual_shift + applied_weathering_shift,
        dtype=np.float64,
    )
    if not np.allclose(final_shift, expected_final_shift, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            'final_trace_shift_s_sorted must equal datum + residual + '
            'applied_weathering'
        )

    final_used_trace_mask = _require_1d_bool_array(
        applied_shift.final_used_trace_mask_sorted,
        name='final_used_trace_mask_sorted',
        expected_shape=expected_trace_shape,
    )
    rejected_trace_mask = _require_1d_bool_array(
        applied_shift.rejected_trace_mask_sorted,
        name='rejected_trace_mask_sorted',
        expected_shape=expected_trace_shape,
    )
    rejected_iteration = _coerce_1d_integer_int64(
        applied_shift.rejected_iteration_sorted,
        name='rejected_iteration_sorted',
        expected_shape=expected_trace_shape,
    )
    _validate_rejected_iteration(
        rejected_iteration,
        mask=rejected_trace_mask,
        name='rejected_iteration_sorted',
    )
    if np.any(final_used_trace_mask & rejected_trace_mask):
        raise ValueError('final_used_trace_mask_sorted overlaps rejected traces')
    if solver.robust_result is not None:
        _validate_trace_masks_match_robust_result(
            solver.robust_result,
            final_used_trace_mask=final_used_trace_mask,
            rejected_trace_mask=rejected_trace_mask,
            rejected_iteration=rejected_iteration,
        )

    row_trace_index = _coerce_1d_integer_int64(
        design.row_trace_index_sorted,
        name='row_trace_index_sorted',
        expected_shape=expected_row_shape,
    )
    _validate_index_range(
        row_trace_index,
        n_unique=n_traces,
        name='row_trace_index_sorted',
    )
    trace_to_row_index = _coerce_trace_to_row_index(
        design.trace_to_row_index_sorted,
        n_traces=n_traces,
        n_observations=n_observations,
    )
    row_source_node_id = _coerce_1d_integer_int64(
        design.row_source_node_id,
        name='row_source_node_id',
        expected_shape=expected_row_shape,
    )
    row_receiver_node_id = _coerce_1d_integer_int64(
        design.row_receiver_node_id,
        name='row_receiver_node_id',
        expected_shape=expected_row_shape,
    )
    _validate_index_range(
        row_source_node_id,
        n_unique=n_nodes,
        name='row_source_node_id',
    )
    _validate_index_range(
        row_receiver_node_id,
        n_unique=n_nodes,
        name='row_receiver_node_id',
    )
    row_pick_time = _coerce_1d_finite_float64(
        design.row_pick_time_after_static_s,
        name='row_pick_time_after_static_s',
        expected_shape=expected_row_shape,
    )
    row_moveout_time = _coerce_1d_finite_float64(
        design.row_moveout_time_s,
        name='row_moveout_time_s',
        expected_shape=expected_row_shape,
    )
    row_data = _coerce_1d_finite_float64(
        design.row_data_s,
        name='row_data_s',
        expected_shape=expected_row_shape,
    )
    design_data = _coerce_1d_finite_float64(
        design.data_s,
        name='design.data_s',
        expected_shape=expected_row_shape,
    )
    if not np.allclose(row_data, design_data, rtol=0.0, atol=1.0e-12):
        raise ValueError('row_data_s must match design.data_s')

    source_count = _coerce_1d_integer_int64(
        design.source_observation_count_by_node,
        name='source_observation_count_by_node',
        expected_shape=expected_node_shape,
    )
    receiver_count = _coerce_1d_integer_int64(
        design.receiver_observation_count_by_node,
        name='receiver_observation_count_by_node',
        expected_shape=expected_node_shape,
    )
    total_count = _coerce_1d_integer_int64(
        design.total_observation_count_by_node,
        name='total_observation_count_by_node',
        expected_shape=expected_node_shape,
    )
    _validate_nonnegative_array(
        source_count,
        name='source_observation_count_by_node',
    )
    _validate_nonnegative_array(
        receiver_count,
        name='receiver_observation_count_by_node',
    )
    if np.any(source_count + receiver_count != total_count):
        raise ValueError(
            'total_observation_count_by_node must equal source + receiver counts'
        )

    row_estimated_delay = np.ascontiguousarray(
        design.matrix @ node_time_term,
        dtype=np.float64,
    )
    if row_estimated_delay.shape != expected_row_shape:
        raise ValueError('row_estimated_time_term_delay_s shape mismatch')
    _validate_all_finite(
        row_estimated_delay,
        name='row_estimated_time_term_delay_s',
    )
    row_residual_before = np.ascontiguousarray(row_data, dtype=np.float64)
    row_residual_after = np.ascontiguousarray(
        row_data - row_estimated_delay,
        dtype=np.float64,
    )
    _validate_all_finite(row_residual_after, name='row_residual_after_s')
    _validate_solver_row_values_if_full_shape(
        solver.sparse_result,
        row_estimated_delay=row_estimated_delay,
        row_residual_before=row_residual_before,
        row_residual_after=row_residual_after,
    )

    component_id = _component_id_by_node(solver.sparse_result.system, n_nodes=n_nodes)
    lineage = _metadata_from_inputs(inputs, metadata)

    return _ArtifactContext(
        metadata=lineage,
        solver=solver,
        n_traces=n_traces,
        n_samples=n_samples,
        dt=dt,
        n_nodes=n_nodes,
        n_observations=n_observations,
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        pick_time_raw_s_sorted=pick_time_raw,
        valid_pick_mask_sorted=valid_pick_mask,
        pick_time_after_static_s_sorted=pick_time_after_static,
        moveout_time_s_sorted=moveout_time,
        moveout_distance_m_sorted=moveout_distance,
        valid_moveout_mask_sorted=valid_moveout_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        source_node_time_term_s_sorted=source_node_time_term,
        receiver_node_time_term_s_sorted=receiver_node_time_term,
        estimated_trace_time_term_delay_s_sorted=estimated_delay,
        applied_weathering_shift_s_sorted=applied_weathering_shift,
        datum_trace_shift_s_sorted=datum_shift,
        residual_applied_shift_s_sorted=residual_shift,
        final_trace_shift_s_sorted=final_shift,
        final_used_trace_mask_sorted=final_used_trace_mask,
        rejected_trace_mask_sorted=rejected_trace_mask,
        rejected_iteration_sorted=rejected_iteration,
        source_id_sorted=source_id,
        receiver_id_sorted=receiver_id,
        offset_sorted=offset,
        source_x_m_sorted=source_x,
        source_y_m_sorted=source_y,
        receiver_x_m_sorted=receiver_x,
        receiver_y_m_sorted=receiver_y,
        source_elevation_m_sorted=source_elevation,
        receiver_elevation_m_sorted=receiver_elevation,
        source_depth_m_sorted=source_depth,
        node_id=np.arange(n_nodes, dtype=np.int64),
        node_time_term_s=node_time_term,
        component_id_by_node=component_id,
        source_observation_count_by_node=source_count,
        receiver_observation_count_by_node=receiver_count,
        total_observation_count_by_node=total_count,
        row_trace_index_sorted=row_trace_index,
        trace_to_row_index_sorted=trace_to_row_index,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        row_pick_time_after_static_s=row_pick_time,
        row_moveout_time_s=row_moveout_time,
        row_data_s=row_data,
        row_estimated_time_term_delay_s=row_estimated_delay,
        row_residual_before_s=row_residual_before,
        row_residual_after_s=row_residual_after,
    )


def _extract_solver_context(
    solver_result: TimeTermSparseSolverResult | TimeTermRobustSolverResult,
    *,
    n_traces: int,
    n_observations: int,
) -> _SolverContext:
    expected_trace_shape = (n_traces,)
    expected_row_shape = (n_observations,)
    if isinstance(solver_result, TimeTermRobustSolverResult):
        sparse_result = solver_result.final_solver_result
        if not isinstance(sparse_result, TimeTermSparseSolverResult):
            raise ValueError(
                'final_solver_result must be a TimeTermSparseSolverResult instance'
            )
        initial_row_used = _require_1d_bool_array(
            solver_result.initial_row_used_mask,
            name='initial_row_used_mask',
            expected_shape=expected_row_shape,
        )
        final_row_used = _require_1d_bool_array(
            solver_result.final_row_used_mask,
            name='final_row_used_mask',
            expected_shape=expected_row_shape,
        )
        final_row_rejected = _require_1d_bool_array(
            solver_result.final_row_rejected_mask,
            name='final_row_rejected_mask',
            expected_shape=expected_row_shape,
        )
        row_rejected_iteration = _coerce_1d_integer_int64(
            solver_result.row_rejected_iteration,
            name='row_rejected_iteration',
            expected_shape=expected_row_shape,
        )
        _validate_row_mask_relationships(
            initial_row_used=initial_row_used,
            final_row_used=final_row_used,
            final_row_rejected=final_row_rejected,
            row_rejected_iteration=row_rejected_iteration,
        )
        _coerce_1d_bool_array_for_solver_trace_mask(
            solver_result.initial_used_trace_mask_sorted,
            'initial_used_trace_mask_sorted',
            expected_trace_shape,
        )
        return _SolverContext(
            sparse_result=sparse_result,
            robust_result=solver_result,
            solver_result_kind='robust',
            initial_row_used_mask=initial_row_used,
            final_row_used_mask=final_row_used,
            final_row_rejected_mask=final_row_rejected,
            row_rejected_iteration=row_rejected_iteration,
            robust_enabled=bool(solver_result.enabled),
            robust_method=str(solver_result.method),
            robust_threshold=float(solver_result.robust_options.threshold),
            robust_stop_reason=str(solver_result.stop_reason),
            robust_iterations=tuple(solver_result.iterations),
        )

    if isinstance(solver_result, TimeTermSparseSolverResult):
        _coerce_1d_bool_array_for_solver_trace_mask(
            solver_result.used_trace_mask_sorted,
            'used_trace_mask_sorted',
            expected_trace_shape,
        )
        return _SolverContext(
            sparse_result=solver_result,
            robust_result=None,
            solver_result_kind='sparse',
            initial_row_used_mask=np.ones(expected_row_shape, dtype=bool),
            final_row_used_mask=np.ones(expected_row_shape, dtype=bool),
            final_row_rejected_mask=np.zeros(expected_row_shape, dtype=bool),
            row_rejected_iteration=np.full(expected_row_shape, -1, dtype=np.int64),
            robust_enabled=False,
            robust_method=None,
            robust_threshold=None,
            robust_stop_reason='disabled',
            robust_iterations=(),
        )

    raise ValueError(
        'solver_result must be a TimeTermSparseSolverResult or '
        'TimeTermRobustSolverResult instance'
    )


def _metadata_from_inputs(
    inputs: TimeTermInversionInputs,
    metadata: TimeTermStaticArtifactMetadata | None,
) -> TimeTermStaticArtifactMetadata:
    raw_metadata = inputs.metadata if isinstance(inputs.metadata, Mapping) else {}
    default_metadata = TimeTermStaticArtifactMetadata(
        job_id=_optional_str(raw_metadata.get('job_id')),
        input_file_id=_optional_str(inputs.input_file_id),
        key1_byte=int(inputs.key1_byte),
        key2_byte=int(inputs.key2_byte),
        pick_source_description=_optional_str(inputs.pick_source_description),
        datum_solution_path=_optional_path_str(inputs.datum_solution_path),
        residual_solution_path=_optional_path_str(inputs.residual_solution_path),
        linkage_artifact_path=_optional_path_str(inputs.linkage_artifact_path),
        request=_optional_request(raw_metadata.get('request')),
        header_source_segy_path=_optional_str(
            raw_metadata.get('header_source_segy_path')
        ),
    )
    if metadata is None:
        return default_metadata
    return TimeTermStaticArtifactMetadata(
        job_id=metadata.job_id
        if metadata.job_id is not None
        else default_metadata.job_id,
        input_file_id=metadata.input_file_id
        if metadata.input_file_id is not None
        else default_metadata.input_file_id,
        key1_byte=_optional_int(metadata.key1_byte, name='key1_byte')
        if metadata.key1_byte is not None
        else default_metadata.key1_byte,
        key2_byte=_optional_int(metadata.key2_byte, name='key2_byte')
        if metadata.key2_byte is not None
        else default_metadata.key2_byte,
        pick_source_description=metadata.pick_source_description
        if metadata.pick_source_description is not None
        else default_metadata.pick_source_description,
        datum_solution_path=metadata.datum_solution_path
        if metadata.datum_solution_path is not None
        else default_metadata.datum_solution_path,
        residual_solution_path=metadata.residual_solution_path
        if metadata.residual_solution_path is not None
        else default_metadata.residual_solution_path,
        linkage_artifact_path=metadata.linkage_artifact_path
        if metadata.linkage_artifact_path is not None
        else default_metadata.linkage_artifact_path,
        request=metadata.request
        if metadata.request is not None
        else default_metadata.request,
        header_source_segy_path=metadata.header_source_segy_path
        if metadata.header_source_segy_path is not None
        else default_metadata.header_source_segy_path,
    )


def _robust_iteration_arrays(context: _ArtifactContext) -> dict[str, np.ndarray]:
    records = _robust_iteration_records(context)
    return {
        'robust_iteration_index': np.asarray(
            [record['iteration_index'] for record in records],
            dtype=np.int64,
        ),
        'robust_iteration_n_used': np.asarray(
            [record['n_used'] for record in records],
            dtype=np.int64,
        ),
        'robust_iteration_n_rejected_total': np.asarray(
            [record['n_rejected_total'] for record in records],
            dtype=np.int64,
        ),
        'robust_iteration_n_rejected_this_iteration': np.asarray(
            [record['n_rejected_this_iteration'] for record in records],
            dtype=np.int64,
        ),
        'robust_iteration_center_s': np.asarray(
            [record['center_s'] for record in records],
            dtype=np.float64,
        ),
        'robust_iteration_scale_s': np.asarray(
            [record['scale_s'] for record in records],
            dtype=np.float64,
        ),
        'robust_iteration_threshold_s': np.asarray(
            [record['threshold_s'] for record in records],
            dtype=np.float64,
        ),
        'robust_iteration_rms_residual_after_s': np.asarray(
            [record['rms_residual_after_s'] for record in records],
            dtype=np.float64,
        ),
    }


def _robust_iteration_payload(context: _ArtifactContext) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for record in _robust_iteration_records(context):
        payload.append(
            {
                'iteration_index': int(record['iteration_index']),
                'n_used': int(record['n_used']),
                'n_rejected_total': int(record['n_rejected_total']),
                'n_rejected_this_iteration': int(
                    record['n_rejected_this_iteration']
                ),
                'center_s': _json_float(record['center_s']),
                'scale_s': _json_float(record['scale_s']),
                'threshold_s': _json_float(record['threshold_s']),
                'rms_residual_after_s': _json_float(
                    record['rms_residual_after_s']
                ),
            }
        )
    return payload


def _robust_iteration_records(context: _ArtifactContext) -> list[dict[str, float | int]]:
    if context.solver.robust_iterations:
        return [
            {
                'iteration_index': int(iteration.iteration),
                'n_used': int(iteration.n_used),
                'n_rejected_total': int(iteration.n_rejected_total),
                'n_rejected_this_iteration': int(
                    iteration.n_rejected_this_iteration
                ),
                'center_s': _coerce_finite_float(
                    iteration.center_s,
                    name='robust iteration center_s',
                ),
                'scale_s': _coerce_finite_float(
                    iteration.scale_s,
                    name='robust iteration scale_s',
                ),
                'threshold_s': _coerce_finite_float(
                    iteration.threshold_s,
                    name='robust iteration threshold_s',
                ),
                'rms_residual_after_s': _coerce_finite_float(
                    iteration.solver_result.rms_residual_after_s,
                    name='robust iteration rms_residual_after_s',
                ),
            }
            for iteration in context.solver.robust_iterations
        ]

    solver = context.solver.sparse_result
    return [
        {
            'iteration_index': 0,
            'n_used': int(context.n_observations),
            'n_rejected_total': 0,
            'n_rejected_this_iteration': 0,
            'center_s': 0.0,
            'scale_s': 0.0,
            'threshold_s': 0.0,
            'rms_residual_after_s': _coerce_finite_float(
                solver.rms_residual_after_s,
                name='rms_residual_after_s',
            ),
        }
    ]


def _robust_iteration_count(solver: _SolverContext) -> int:
    if solver.robust_iterations:
        return int(len(solver.robust_iterations))
    return 1


def _validate_solution_arrays(
    context: _ArtifactContext,
    arrays: dict[str, np.ndarray],
) -> None:
    for name, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise ValueError(
                'time-term solution arrays must not contain object dtype: '
                f'{name}'
            )

    _validate_named_shapes(
        arrays,
        names=(
            'sorted_trace_index',
            'pick_time_raw_s_sorted',
            'valid_pick_mask_sorted',
            'pick_time_after_static_s_sorted',
            'moveout_time_s_sorted',
            'moveout_distance_m_sorted',
            'valid_moveout_mask_sorted',
            'source_node_id_sorted',
            'receiver_node_id_sorted',
            'source_node_time_term_s_sorted',
            'receiver_node_time_term_s_sorted',
            'estimated_trace_time_term_delay_s_sorted',
            'applied_weathering_shift_s_sorted',
            'datum_trace_shift_s_sorted',
            'residual_applied_shift_s_sorted',
            'final_trace_shift_s_sorted',
            'final_used_trace_mask_sorted',
            'rejected_trace_mask_sorted',
            'rejected_iteration_sorted',
            'source_id_sorted',
            'receiver_id_sorted',
            'offset_sorted',
            'source_x_m_sorted',
            'source_y_m_sorted',
            'receiver_x_m_sorted',
            'receiver_y_m_sorted',
            'source_elevation_m_sorted',
            'receiver_elevation_m_sorted',
            'source_depth_m_sorted',
        ),
        expected_shape=(context.n_traces,),
    )
    _validate_named_shapes(
        arrays,
        names=(
            'node_id',
            'node_time_term_s',
            'node_time_term_ms',
            'source_observation_count_by_node',
            'receiver_observation_count_by_node',
            'total_observation_count_by_node',
            'component_id_by_node',
        ),
        expected_shape=(context.n_nodes,),
    )
    _validate_named_shapes(
        arrays,
        names=(
            'row_trace_index_sorted',
            'row_source_node_id',
            'row_receiver_node_id',
            'row_pick_time_after_static_s',
            'row_moveout_time_s',
            'row_data_s',
            'row_estimated_time_term_delay_s',
            'row_residual_before_s',
            'row_residual_after_s',
            'row_residual_after_ms',
            'initial_row_used_mask',
            'final_row_used_mask',
            'final_row_rejected_mask',
            'row_rejected_iteration',
        ),
        expected_shape=(context.n_observations,),
    )

    for name in (
        'valid_pick_mask_sorted',
        'valid_moveout_mask_sorted',
        'final_used_trace_mask_sorted',
        'rejected_trace_mask_sorted',
        'initial_row_used_mask',
        'final_row_used_mask',
        'final_row_rejected_mask',
    ):
        if not np.issubdtype(np.asarray(arrays[name]).dtype, np.bool_):
            raise ValueError(f'{name} must have bool dtype')

    for name in (
        'sorted_trace_index',
        'source_node_id_sorted',
        'receiver_node_id_sorted',
        'rejected_iteration_sorted',
        'source_id_sorted',
        'receiver_id_sorted',
        'node_id',
        'source_observation_count_by_node',
        'receiver_observation_count_by_node',
        'total_observation_count_by_node',
        'component_id_by_node',
        'row_trace_index_sorted',
        'row_source_node_id',
        'row_receiver_node_id',
        'row_rejected_iteration',
        'robust_iteration_index',
        'robust_iteration_n_used',
        'robust_iteration_n_rejected_total',
        'robust_iteration_n_rejected_this_iteration',
    ):
        if not np.issubdtype(np.asarray(arrays[name]).dtype, np.integer):
            raise ValueError(f'{name} must have integer dtype')

    _validate_no_inf_for_names(
        arrays,
        names=(
            'pick_time_raw_s_sorted',
            'pick_time_after_static_s_sorted',
            'moveout_time_s_sorted',
            'moveout_distance_m_sorted',
            'source_node_time_term_s_sorted',
            'receiver_node_time_term_s_sorted',
            'estimated_trace_time_term_delay_s_sorted',
            'applied_weathering_shift_s_sorted',
            'datum_trace_shift_s_sorted',
            'residual_applied_shift_s_sorted',
            'final_trace_shift_s_sorted',
            'offset_sorted',
            'source_x_m_sorted',
            'source_y_m_sorted',
            'receiver_x_m_sorted',
            'receiver_y_m_sorted',
            'source_elevation_m_sorted',
            'receiver_elevation_m_sorted',
            'source_depth_m_sorted',
            'node_time_term_s',
            'node_time_term_ms',
            'row_pick_time_after_static_s',
            'row_moveout_time_s',
            'row_data_s',
            'row_estimated_time_term_delay_s',
            'row_residual_before_s',
            'row_residual_after_s',
            'row_residual_after_ms',
            'robust_iteration_center_s',
            'robust_iteration_scale_s',
            'robust_iteration_threshold_s',
            'robust_iteration_rms_residual_after_s',
        ),
    )

    for name in (
        'source_node_time_term_s_sorted',
        'receiver_node_time_term_s_sorted',
        'estimated_trace_time_term_delay_s_sorted',
        'applied_weathering_shift_s_sorted',
        'datum_trace_shift_s_sorted',
        'residual_applied_shift_s_sorted',
        'final_trace_shift_s_sorted',
        'node_time_term_s',
        'row_estimated_time_term_delay_s',
        'row_residual_before_s',
        'row_residual_after_s',
    ):
        _validate_all_finite(np.asarray(arrays[name]), name=name)

    _validate_index_range(
        np.asarray(arrays['source_node_id_sorted']),
        n_unique=context.n_nodes,
        name='source_node_id_sorted',
    )
    _validate_index_range(
        np.asarray(arrays['receiver_node_id_sorted']),
        n_unique=context.n_nodes,
        name='receiver_node_id_sorted',
    )
    _validate_index_range(
        np.asarray(arrays['row_source_node_id']),
        n_unique=context.n_nodes,
        name='row_source_node_id',
    )
    _validate_index_range(
        np.asarray(arrays['row_receiver_node_id']),
        n_unique=context.n_nodes,
        name='row_receiver_node_id',
    )

    _validate_rejected_iteration(
        np.asarray(arrays['rejected_iteration_sorted']),
        mask=np.asarray(arrays['rejected_trace_mask_sorted']),
        name='rejected_iteration_sorted',
    )
    _validate_rejected_iteration(
        np.asarray(arrays['row_rejected_iteration']),
        mask=np.asarray(arrays['final_row_rejected_mask']),
        name='row_rejected_iteration',
    )

    applied_shift = np.asarray(arrays['applied_weathering_shift_s_sorted'])
    estimated_delay = np.asarray(arrays['estimated_trace_time_term_delay_s_sorted'])
    if not np.allclose(applied_shift, -estimated_delay, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            'applied_weathering_shift_s_sorted must equal '
            '-estimated_trace_time_term_delay_s_sorted'
        )
    final_shift = np.asarray(arrays['final_trace_shift_s_sorted'])
    expected_final_shift = (
        np.asarray(arrays['datum_trace_shift_s_sorted'])
        + np.asarray(arrays['residual_applied_shift_s_sorted'])
        + applied_shift
    )
    if not np.allclose(final_shift, expected_final_shift, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            'final_trace_shift_s_sorted must equal datum + residual + '
            'applied_weathering'
        )


def _validate_named_shapes(
    arrays: dict[str, np.ndarray],
    *,
    names: Sequence[str],
    expected_shape: tuple[int, ...],
) -> None:
    for name in names:
        arr = np.asarray(arrays[name])
        if arr.shape != expected_shape:
            raise ValueError(
                f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
            )


def _validate_no_inf_for_names(
    arrays: dict[str, np.ndarray],
    *,
    names: Sequence[str],
) -> None:
    for name in names:
        _validate_no_inf(np.asarray(arrays[name]), name=name)


def _validate_trace_masks_match_robust_result(
    robust_result: TimeTermRobustSolverResult,
    *,
    final_used_trace_mask: np.ndarray,
    rejected_trace_mask: np.ndarray,
    rejected_iteration: np.ndarray,
) -> None:
    expected_shape = final_used_trace_mask.shape
    robust_final_used = _require_1d_bool_array(
        robust_result.final_used_trace_mask_sorted,
        name='robust_result.final_used_trace_mask_sorted',
        expected_shape=expected_shape,
    )
    robust_rejected = _require_1d_bool_array(
        robust_result.final_rejected_trace_mask_sorted,
        name='robust_result.final_rejected_trace_mask_sorted',
        expected_shape=expected_shape,
    )
    robust_rejected_iteration = _coerce_1d_integer_int64(
        robust_result.rejected_iteration_sorted,
        name='robust_result.rejected_iteration_sorted',
        expected_shape=expected_shape,
    )
    if not np.array_equal(final_used_trace_mask, robust_final_used):
        raise ValueError('final_used_trace_mask_sorted must match robust result')
    if not np.array_equal(rejected_trace_mask, robust_rejected):
        raise ValueError('rejected_trace_mask_sorted must match robust result')
    if not np.array_equal(rejected_iteration, robust_rejected_iteration):
        raise ValueError('rejected_iteration_sorted must match robust result')


def _validate_row_mask_relationships(
    *,
    initial_row_used: np.ndarray,
    final_row_used: np.ndarray,
    final_row_rejected: np.ndarray,
    row_rejected_iteration: np.ndarray,
) -> None:
    if np.any(final_row_used & ~initial_row_used):
        raise ValueError('final_row_used_mask must be a subset of initial_row_used_mask')
    expected_rejected = initial_row_used & ~final_row_used
    if not np.array_equal(final_row_rejected, expected_rejected):
        raise ValueError(
            'final_row_rejected_mask must equal initial_row_used_mask & '
            '~final_row_used_mask'
        )
    _validate_rejected_iteration(
        row_rejected_iteration,
        mask=final_row_rejected,
        name='row_rejected_iteration',
    )


def _validate_rejected_iteration(
    values: np.ndarray,
    *,
    mask: np.ndarray,
    name: str,
) -> None:
    if np.any(values < -1):
        raise ValueError(f'{name} must contain -1 or nonnegative values')
    if np.any(values[~mask] != -1):
        raise ValueError(f'{name} must be -1 for unrejected entries')
    if np.any(values[mask] < 0):
        raise ValueError(f'{name} must be nonnegative for rejected entries')


def _validate_solver_row_values_if_full_shape(
    solver: TimeTermSparseSolverResult,
    *,
    row_estimated_delay: np.ndarray,
    row_residual_before: np.ndarray,
    row_residual_after: np.ndarray,
) -> None:
    expected_shape = row_estimated_delay.shape
    solver_row_estimated = np.asarray(solver.row_estimated_time_term_delay_s)
    if solver_row_estimated.shape == expected_shape and not np.allclose(
        solver_row_estimated,
        row_estimated_delay,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError('row_estimated_time_term_delay_s must match final model')
    solver_residual_before = np.asarray(solver.row_residual_before_s)
    if solver_residual_before.shape == expected_shape and not np.allclose(
        solver_residual_before,
        row_residual_before,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError('row_residual_before_s must match design data')
    solver_residual_after = np.asarray(solver.row_residual_after_s)
    if solver_residual_after.shape == expected_shape and not np.allclose(
        solver_residual_after,
        row_residual_after,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError('row_residual_after_s must match final model residual')


def _component_id_by_node(system: TimeTermSolverSystem, *, n_nodes: int) -> np.ndarray:
    component = getattr(system, 'component_id_by_node', None)
    if component is None:
        return np.full(n_nodes, -1, dtype=np.int64)
    out = _coerce_1d_integer_int64(
        component,
        name='component_id_by_node',
        expected_shape=(n_nodes,),
    )
    if np.any(out < -1):
        raise ValueError('component_id_by_node must contain -1 or nonnegative values')
    return out


def _coerce_1d_bool_array_for_solver_trace_mask(
    values: object,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    return _require_1d_bool_array(values, name=name, expected_shape=expected_shape)


def _optional_offset_array(values: np.ndarray | None, *, n_traces: int) -> np.ndarray:
    if values is None:
        return np.full(n_traces, np.nan, dtype=np.float64)
    return _coerce_1d_float64_allow_nan_no_inf(
        values,
        name='offset_sorted',
        expected_shape=(n_traces,),
    )


def _coerce_trace_to_row_index(
    values: object,
    *,
    n_traces: int,
    n_observations: int,
) -> np.ndarray:
    arr = _coerce_1d_integer_int64(
        values,
        name='trace_to_row_index_sorted',
        expected_shape=(n_traces,),
    )
    if np.any(arr < -1):
        raise ValueError('trace_to_row_index_sorted must be greater than or equal to -1')
    if np.any(arr >= n_observations):
        raise ValueError('trace_to_row_index_sorted contains row index outside range')
    return arr


def _coerce_1d_float64_allow_nan_no_inf(
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
    _validate_no_inf(arr, name=name)
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
    if not _is_real_numeric_dtype(arr.dtype):
        raise ValueError(f'{name} must have a real numeric dtype')
    return np.ascontiguousarray(arr, dtype=np.float64)


def _coerce_finite_or_nan_float(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('float scalar expected')
    out = float(value)
    if not np.isfinite(out) and not np.isnan(out):
        raise ValueError('float scalar must be finite or NaN')
    return out


def _validate_no_inf(values: np.ndarray, *, name: str) -> None:
    arr = np.asarray(values, dtype=np.float64)
    if np.any(np.isinf(arr)):
        raise ValueError(f'{name} must not contain Inf')


def _validate_all_finite(values: np.ndarray, *, name: str) -> None:
    arr = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f'{name} must contain only finite values')


def _validate_finite_at_mask(
    values: np.ndarray,
    *,
    mask: np.ndarray,
    name: str,
    mask_name: str,
) -> None:
    if np.any(~np.isfinite(values[mask])):
        raise ValueError(f'{name} must be finite for {mask_name}')


def _validate_index_range(
    values: np.ndarray,
    *,
    n_unique: int,
    name: str,
) -> None:
    if np.any(values < 0):
        raise ValueError(f'{name} must be greater than or equal to 0')
    if np.any(values >= n_unique):
        raise ValueError(f'{name} contains values outside 0..{n_unique - 1}')


def _validate_nonnegative_array(values: np.ndarray, *, name: str) -> None:
    if np.any(values < 0):
        raise ValueError(f'{name} must contain nonnegative values')


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


def _nan_if_none(value: float | None) -> float:
    return np.nan if value is None else float(value)


def _optional_json_float(value: float | None) -> float | None:
    if value is None:
        return None
    return _json_float(value)


def _json_float(value: object) -> float:
    return float(_coerce_finite_float(value, name='JSON float'))


def _fraction(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _rms(values: np.ndarray) -> float:
    arr = _coerce_1d_finite_float64(values, name='rms values')
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def _stats_payload(stats: TimeTermFiniteStats) -> dict[str, Any]:
    return {
        'count': int(stats.count),
        'min': stats.min,
        'max': stats.max,
        'mean': stats.mean,
        'median': stats.median,
        'std': stats.std,
        'mad': stats.mad,
    }


def _metadata_str(metadata: Mapping[str, object], key: str, default: str) -> str:
    value = metadata.get(key, default) if isinstance(metadata, Mapping) else default
    if value is None:
        return default
    return str(value)


def _require_metadata_int(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f'{name} is required')
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_path_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def _optional_request(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


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


def _json_safe(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return str(value)


def _assert_strict_json_payload(payload: dict[str, Any]) -> None:
    assert_strict_json(payload)


def _write_time_term_solution_npz(
    out_path: Path,
    payload: dict[str, np.ndarray],
) -> None:
    _common_write_npz_atomic(
        out_path,
        payload,
        compressed=False,
        reject_object_arrays=False,
    )


def _write_time_term_qc_json(out_path: Path, payload: dict[str, Any]) -> None:
    _common_write_json_atomic(
        out_path,
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        trailing_newline=True,
    )


def _write_time_term_statics_csv(
    out_path: Path,
    rows: list[dict[str, object]],
) -> None:
    _common_write_csv_atomic(
        out_path,
        columns=_CSV_COLUMNS,
        rows=rows,
        extrasaction='raise',
        lineterminator='\r\n',
    )


__all__ = [
    'ORDER',
    'QC_ARTIFACT_KIND',
    'SCHEMA_VERSION',
    'SOLUTION_ARTIFACT_KIND',
    'TIME_TERM_STATICS_CSV_NAME',
    'TIME_TERM_STATIC_QC_JSON_NAME',
    'TIME_TERM_STATIC_SOLUTION_NPZ_NAME',
    'TimeTermFiniteStats',
    'TimeTermStaticArtifactMetadata',
    'TimeTermStaticArtifactPaths',
    'build_time_term_qc_payload',
    'build_time_term_solution_arrays',
    'build_time_term_statics_csv_rows',
    'summarize_finite_values',
    'write_time_term_static_artifacts',
]
