"""Global bedrock slowness estimation for GLI refraction statics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.api.schemas import (
    RefractionStaticApplyRequest,
    RefractionStaticModelRequest,
    RefractionStaticSolverRequest,
)
from app.core.state import AppState
from app.services.refraction_static_design_matrix import (
    build_refraction_static_design_matrix,
)
from app.services.refraction_static_solver import (
    RefractionStaticSolverError,
    solve_refraction_static_bounded_ls,
)
from app.services.refraction_static_types import (
    RefractionBedrockSlownessResult,
    RefractionStaticDesignMatrix,
    RefractionStaticInputModel,
    RefractionStaticSolverResult,
)

REFRACTION_BEDROCK_QC_JSON_NAME = 'refraction_bedrock_velocity_qc.json'
REFRACTION_BEDROCK_RESIDUALS_CSV_NAME = 'refraction_bedrock_velocity_residuals.csv'

_RESIDUAL_COLUMNS = (
    'row_index',
    'sorted_trace_index',
    'source_node_id',
    'receiver_node_id',
    'distance_m',
    'observed_pick_time_s',
    'modeled_pick_time_s',
    'residual_time_s',
    'residual_ms',
    'used',
    'rejected_by_robust',
)


class RefractionBedrockSlownessError(ValueError):
    """Raised when global bedrock slowness cannot be estimated."""


def estimate_global_bedrock_slowness_from_first_breaks(
    *,
    req: RefractionStaticApplyRequest,
    state: AppState,
    job_dir: Path | None = None,
) -> RefractionBedrockSlownessResult:
    """Build refraction inputs from a request and estimate global bedrock slowness."""
    _require_solve_global(req.model)
    from app.services.refraction_static_inputs import build_refraction_static_input_model

    try:
        input_model = build_refraction_static_input_model(
            req=req,
            state=state,
            job_dir=job_dir,
        )
    except ValueError as exc:
        raise RefractionBedrockSlownessError(str(exc)) from exc
    return estimate_global_bedrock_slowness_from_input_model(
        input_model=input_model,
        model=req.model,
        solver=req.solver,
        job_dir=job_dir,
    )


def estimate_global_bedrock_slowness_from_input_model(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    job_dir: Path | None = None,
    include_debug_objects: bool = False,
) -> RefractionBedrockSlownessResult:
    """Estimate global bedrock slowness from an already-built input model."""
    _require_solve_global(model)
    input_qc = _validate_input_quality(input_model=input_model, model=model, solver=solver)
    try:
        design_matrix = build_refraction_static_design_matrix(
            input_model=input_model,
            model=model,
        )
    except ValueError as exc:
        raise RefractionBedrockSlownessError(str(exc)) from exc
    _validate_design_matrix(design_matrix)

    try:
        solver_result = solve_refraction_static_bounded_ls(
            design_matrix=design_matrix,
            model=model,
            solver=solver,
        )
    except RefractionStaticSolverError as exc:
        raise RefractionBedrockSlownessError(str(exc)) from exc
    result = _build_bedrock_result(
        input_model=input_model,
        design_matrix=design_matrix,
        solver_result=solver_result,
        model=model,
        solver=solver,
        input_qc=input_qc,
        include_debug_objects=include_debug_objects,
    )
    if job_dir is not None:
        write_refraction_bedrock_debug_artifacts(Path(job_dir), result)
    return result


def write_refraction_bedrock_debug_artifacts(
    job_dir: Path,
    result: RefractionBedrockSlownessResult,
) -> dict[str, Path]:
    """Write lightweight QC and residual artifacts for the bedrock solve."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / REFRACTION_BEDROCK_QC_JSON_NAME
    residuals_path = root / REFRACTION_BEDROCK_RESIDUALS_CSV_NAME
    _write_json_atomic(qc_path, result.qc)
    _write_csv_atomic(residuals_path, _residual_rows(result))
    return {
        'qc_json': qc_path,
        'residuals_csv': residuals_path,
    }


def _require_solve_global(model: RefractionStaticModelRequest) -> None:
    if getattr(model, 'bedrock_velocity_mode', None) != 'solve_global':
        raise RefractionBedrockSlownessError(
            "Global bedrock slowness estimation requires "
            "bedrock_velocity_mode='solve_global'."
        )


def _validate_input_quality(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
) -> dict[str, Any]:
    mask = np.asarray(input_model.valid_observation_mask_sorted, dtype=bool)
    picks = np.asarray(input_model.pick_time_s_sorted, dtype=np.float64)
    distance = np.asarray(input_model.distance_m_sorted, dtype=np.float64)
    source_node = np.asarray(input_model.source_node_id_sorted, dtype=np.int64)
    receiver_node = np.asarray(input_model.receiver_node_id_sorted, dtype=np.int64)
    expected_shape = (int(input_model.n_traces),)
    for name, values in (
        ('valid_observation_mask_sorted', mask),
        ('pick_time_s_sorted', picks),
        ('distance_m_sorted', distance),
        ('source_node_id_sorted', source_node),
        ('receiver_node_id_sorted', receiver_node),
    ):
        if values.shape != expected_shape:
            raise RefractionBedrockSlownessError(
                f'input_model.{name} shape mismatch: '
                f'expected {expected_shape}, got {values.shape}'
            )

    n_valid = int(np.count_nonzero(mask))
    if n_valid <= 0:
        raise RefractionBedrockSlownessError(
            'No valid refraction observations remain for global bedrock slowness '
            'estimation.'
        )
    min_used = int(solver.robust.min_used_observations)
    if n_valid < min_used:
        raise RefractionBedrockSlownessError(
            'Too few valid refraction observations for global bedrock slowness '
            'estimation.'
        )

    selected_picks = np.ascontiguousarray(picks[mask], dtype=np.float64)
    selected_distance = np.ascontiguousarray(distance[mask], dtype=np.float64)
    if np.any(~np.isfinite(selected_picks)):
        raise RefractionBedrockSlownessError(
            'selected first-break pick times must be finite'
        )
    if np.any(~np.isfinite(selected_distance)):
        raise RefractionBedrockSlownessError('selected distances must be finite')

    active_nodes = np.unique(
        np.concatenate((source_node[mask], receiver_node[mask])),
    )
    n_active_nodes = int(active_nodes.shape[0])
    if n_active_nodes < 2:
        raise RefractionBedrockSlownessError(
            'At least two active refraction nodes are required for global '
            'bedrock slowness estimation.'
        )

    distance_min = float(np.min(selected_distance))
    distance_max = float(np.max(selected_distance))
    distance_aperture = float(distance_max - distance_min)
    if distance_aperture <= 0.0:
        raise RefractionBedrockSlownessError(
            'distance aperture must be greater than 0 for global bedrock '
            'slowness estimation'
        )
    pick_min = float(np.min(selected_picks))
    pick_max = float(np.max(selected_picks))
    pick_aperture = float(pick_max - pick_min)
    if pick_aperture <= 0.0:
        raise RefractionBedrockSlownessError(
            'pick time aperture must be greater than 0 for global bedrock '
            'slowness estimation'
        )

    distance_median = float(np.median(selected_distance))
    pick_median = float(np.median(selected_picks))
    max_velocity = float(model.max_bedrock_velocity_m_s)
    if pick_median < distance_median / max_velocity:
        raise RefractionBedrockSlownessError(
            'median pick time is too small for the configured maximum bedrock '
            'velocity and median distance'
        )

    return {
        'n_valid_observations': n_valid,
        'n_active_nodes': n_active_nodes,
        'distance_aperture_m': distance_aperture,
        'distance_m_min': distance_min,
        'distance_m_max': distance_max,
        'distance_m_median': distance_median,
        'pick_time_s_min': pick_min,
        'pick_time_s_max': pick_max,
        'pick_time_s_median': pick_median,
        'pick_time_aperture_s': pick_aperture,
    }


def _validate_design_matrix(design_matrix: RefractionStaticDesignMatrix) -> None:
    if design_matrix.bedrock_velocity_mode != 'solve_global':
        raise RefractionBedrockSlownessError(
            "Global bedrock slowness estimation requires "
            "bedrock_velocity_mode='solve_global'."
        )
    if design_matrix.bedrock_slowness_col is None:
        raise RefractionBedrockSlownessError(
            'solve_global design matrix is missing a bedrock slowness column'
        )


def _build_bedrock_result(
    *,
    input_model: RefractionStaticInputModel,
    design_matrix: RefractionStaticDesignMatrix,
    solver_result: RefractionStaticSolverResult,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    input_qc: dict[str, Any],
    include_debug_objects: bool,
) -> RefractionBedrockSlownessResult:
    _validate_solver_success(solver_result)
    _validate_row_lengths(solver_result)

    min_velocity = float(model.min_bedrock_velocity_m_s)
    max_velocity = float(model.max_bedrock_velocity_m_s)
    lower_slowness = float(1.0 / max_velocity)
    upper_slowness = float(1.0 / min_velocity)
    slowness = _positive_finite(
        solver_result.bedrock_slowness_s_per_m,
        name='solved bedrock slowness',
    )
    velocity = _positive_finite(
        1.0 / slowness,
        name='derived bedrock velocity',
    )
    solver_velocity = _positive_finite(
        solver_result.bedrock_velocity_m_s,
        name='solver bedrock velocity',
    )
    velocity_tol = max(1.0e-6, 1.0e-6 * velocity)
    if abs(solver_velocity - velocity) > velocity_tol:
        raise RefractionBedrockSlownessError(
            'solver bedrock velocity does not match solved bedrock slowness'
        )
    weathering_velocity = float(model.weathering_velocity_m_s)
    if velocity <= weathering_velocity:
        raise RefractionBedrockSlownessError(
            'solved bedrock velocity must be greater than weathering velocity'
        )
    slowness_tol = max(1.0e-12, 1.0e-6 * slowness)
    if slowness < lower_slowness - slowness_tol:
        raise RefractionBedrockSlownessError(
            'solved bedrock slowness is below configured bounds'
        )
    if slowness > upper_slowness + slowness_tol:
        raise RefractionBedrockSlownessError(
            'solved bedrock slowness is above configured bounds'
        )

    if velocity < min_velocity - velocity_tol or velocity > max_velocity + velocity_tol:
        raise RefractionBedrockSlownessError(
            'solved bedrock velocity is outside configured bounds'
        )

    at_min_velocity = bool(velocity <= min_velocity + velocity_tol)
    at_max_velocity = bool(velocity >= max_velocity - velocity_tol)
    if at_min_velocity:
        velocity_status = 'clipped_min_velocity'
    elif at_max_velocity:
        velocity_status = 'clipped_max_velocity'
    else:
        velocity_status = 'solved'

    qc = _build_result_qc(
        input_model=input_model,
        design_matrix=design_matrix,
        solver_result=solver_result,
        model=model,
        solver=solver,
        input_qc=input_qc,
        bedrock_velocity_status=velocity_status,
        lower_slowness=lower_slowness,
        upper_slowness=upper_slowness,
        bedrock_slowness=slowness,
        bedrock_velocity=velocity,
        bedrock_slowness_at_lower_bound=at_max_velocity,
        bedrock_slowness_at_upper_bound=at_min_velocity,
    )
    result = RefractionBedrockSlownessResult(
        bedrock_velocity_mode='solve_global',
        weathering_velocity_m_s=weathering_velocity,
        bedrock_slowness_s_per_m=slowness,
        bedrock_velocity_m_s=velocity,
        bedrock_velocity_status=velocity_status,
        min_bedrock_velocity_m_s=min_velocity,
        max_bedrock_velocity_m_s=max_velocity,
        lower_bedrock_slowness_s_per_m=lower_slowness,
        upper_bedrock_slowness_s_per_m=upper_slowness,
        active_node_id=np.ascontiguousarray(
            solver_result.active_node_id,
            dtype=np.int64,
        ),
        active_node_half_intercept_time_s=np.ascontiguousarray(
            solver_result.active_node_half_intercept_time_s,
            dtype=np.float64,
        ),
        row_trace_index_sorted=np.ascontiguousarray(
            solver_result.row_trace_index_sorted,
            dtype=np.int64,
        ),
        row_source_node_id=np.ascontiguousarray(
            solver_result.row_source_node_id,
            dtype=np.int64,
        ),
        row_receiver_node_id=np.ascontiguousarray(
            solver_result.row_receiver_node_id,
            dtype=np.int64,
        ),
        row_distance_m=np.ascontiguousarray(
            solver_result.row_distance_m,
            dtype=np.float64,
        ),
        observed_pick_time_s=np.ascontiguousarray(
            solver_result.observed_pick_time_s,
            dtype=np.float64,
        ),
        modeled_pick_time_s=np.ascontiguousarray(
            solver_result.modeled_pick_time_s,
            dtype=np.float64,
        ),
        residual_time_s=np.ascontiguousarray(
            solver_result.residual_time_s,
            dtype=np.float64,
        ),
        used_row_mask=np.ascontiguousarray(solver_result.used_row_mask, dtype=bool),
        rejected_by_robust_mask=np.ascontiguousarray(
            solver_result.rejected_by_robust_mask,
            dtype=bool,
        ),
        input_model=input_model if include_debug_objects else None,
        design_matrix=design_matrix if include_debug_objects else None,
        solver_result=solver_result,
        qc=qc,
    )
    _validate_row_lengths(result)
    return result


def _validate_solver_success(solver_result: RefractionStaticSolverResult) -> None:
    if solver_result.solver_status != 'success':
        raise RefractionBedrockSlownessError(
            f'global bedrock slowness solver failed: {solver_result.solver_message}'
        )


def _validate_row_lengths(
    result: RefractionStaticSolverResult | RefractionBedrockSlownessResult,
) -> None:
    n_rows = int(np.asarray(result.observed_pick_time_s).shape[0])
    fields = (
        'row_trace_index_sorted',
        'row_source_node_id',
        'row_receiver_node_id',
        'row_distance_m',
        'modeled_pick_time_s',
        'residual_time_s',
        'used_row_mask',
        'rejected_by_robust_mask',
    )
    for field in fields:
        values = np.asarray(getattr(result, field))
        if values.shape != (n_rows,):
            raise RefractionBedrockSlownessError(
                'result residual arrays length mismatch'
            )


def _build_result_qc(
    *,
    input_model: RefractionStaticInputModel,
    design_matrix: RefractionStaticDesignMatrix,
    solver_result: RefractionStaticSolverResult,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    input_qc: dict[str, Any],
    bedrock_velocity_status: str,
    lower_slowness: float,
    upper_slowness: float,
    bedrock_slowness: float,
    bedrock_velocity: float,
    bedrock_slowness_at_lower_bound: bool,
    bedrock_slowness_at_upper_bound: bool,
) -> dict[str, Any]:
    n_observations = int(solver_result.observed_pick_time_s.shape[0])
    n_used = int(np.count_nonzero(solver_result.used_row_mask))
    n_rejected = int(np.count_nonzero(solver_result.rejected_by_robust_mask))
    qc: dict[str, Any] = {
        'method': 'gli_variable_thickness',
        'bedrock_velocity_mode': 'solve_global',
        'weathering_velocity_m_s': float(model.weathering_velocity_m_s),
        'bedrock_slowness_s_per_m': float(bedrock_slowness),
        'bedrock_velocity_m_s': float(bedrock_velocity),
        'bedrock_velocity_status': bedrock_velocity_status,
        'min_bedrock_velocity_m_s': float(model.min_bedrock_velocity_m_s),
        'max_bedrock_velocity_m_s': float(model.max_bedrock_velocity_m_s),
        'initial_bedrock_velocity_m_s': _json_optional_float(
            model.initial_bedrock_velocity_m_s
        ),
        'lower_bedrock_slowness_s_per_m': float(lower_slowness),
        'upper_bedrock_slowness_s_per_m': float(upper_slowness),
        'bedrock_slowness_at_lower_bound': bool(bedrock_slowness_at_lower_bound),
        'bedrock_slowness_at_upper_bound': bool(bedrock_slowness_at_upper_bound),
        'n_traces': int(input_model.n_traces),
        'n_valid_observations': int(n_observations),
        'n_used_observations': n_used,
        'n_rejected_by_robust': n_rejected,
        'used_fraction': float(n_used / n_observations) if n_observations else 0.0,
        'n_active_nodes': int(design_matrix.n_active_nodes),
        'n_parameters': int(design_matrix.n_parameters),
        'robust_enabled': bool(solver.robust.enabled),
        'robust_method': str(solver.robust.method),
        'robust_iteration_count': int(solver_result.robust_iteration_count),
        'solver_status': solver_result.solver_status,
        'solver_message': str(solver_result.solver_message),
        'solver_cost': float(solver_result.solver_cost),
        'solver_optimality': _json_optional_float(solver_result.solver_optimality),
        'solver_nit': solver_result.solver_nit,
        'design_matrix_shape': [
            int(design_matrix.matrix.shape[0]),
            int(design_matrix.matrix.shape[1]),
        ],
        'design_matrix_nnz': int(design_matrix.matrix.nnz),
    }
    qc.update(input_qc)
    qc.update(_residual_stats_ms(solver_result.residual_time_s))
    _assert_json_safe(qc)
    return qc


def _residual_stats_ms(residual_time_s: np.ndarray) -> dict[str, float]:
    residual = np.asarray(residual_time_s, dtype=np.float64)
    if residual.ndim != 1 or residual.size == 0:
        raise RefractionBedrockSlownessError(
            'result residual arrays length mismatch'
        )
    if np.any(~np.isfinite(residual)):
        raise RefractionBedrockSlownessError('residual_time_s must be finite')
    residual_ms = residual * 1000.0
    median = float(np.median(residual_ms))
    return {
        'residual_rms_ms': float(np.sqrt(np.mean(residual_ms * residual_ms))),
        'residual_mad_ms': float(1.4826 * np.median(np.abs(residual_ms - median))),
        'residual_mean_ms': float(np.mean(residual_ms)),
        'residual_median_ms': median,
        'residual_p95_abs_ms': float(np.percentile(np.abs(residual_ms), 95.0)),
        'residual_max_abs_ms': float(np.max(np.abs(residual_ms))),
    }


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise RefractionBedrockSlownessError(f'{name} must be finite and positive')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionBedrockSlownessError(
            f'{name} must be finite and positive'
        ) from exc
    if not np.isfinite(out):
        raise RefractionBedrockSlownessError(f'{name} must be finite')
    if out <= 0.0:
        raise RefractionBedrockSlownessError(f'{name} must be positive')
    return out


def _residual_rows(
    result: RefractionBedrockSlownessResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_rows = int(result.observed_pick_time_s.shape[0])
    for row_index in range(n_rows):
        residual = float(result.residual_time_s[row_index])
        rows.append(
            {
                'row_index': row_index,
                'sorted_trace_index': int(result.row_trace_index_sorted[row_index]),
                'source_node_id': int(result.row_source_node_id[row_index]),
                'receiver_node_id': int(result.row_receiver_node_id[row_index]),
                'distance_m': float(result.row_distance_m[row_index]),
                'observed_pick_time_s': float(result.observed_pick_time_s[row_index]),
                'modeled_pick_time_s': float(result.modeled_pick_time_s[row_index]),
                'residual_time_s': residual,
                'residual_ms': float(residual * 1000.0),
                'used': bool(result.used_row_mask[row_index]),
                'rejected_by_robust': bool(
                    result.rejected_by_robust_mask[row_index]
                ),
            }
        )
    return rows


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        tmp_path.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
            ),
            encoding='utf-8',
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_RESIDUAL_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _assert_json_safe(payload: dict[str, Any]) -> None:
    json.dumps(payload, allow_nan=False)


def _json_optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


__all__ = [
    'REFRACTION_BEDROCK_QC_JSON_NAME',
    'REFRACTION_BEDROCK_RESIDUALS_CSV_NAME',
    'RefractionBedrockSlownessError',
    'RefractionBedrockSlownessResult',
    'estimate_global_bedrock_slowness_from_first_breaks',
    'estimate_global_bedrock_slowness_from_input_model',
    'write_refraction_bedrock_debug_artifacts',
]
