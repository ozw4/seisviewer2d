"""Cell velocity artifact builders and writers for refraction statics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from app.statics.refraction.contracts.apply import RefractionStaticApplyRequest
from app.statics.refraction.artifacts.arrays import (
    _qc_cell_count_array,
    _qc_cell_id_array,
    _qc_int,
    _qc_optional_float,
    _required_cell_float_array,
    _required_cell_int_array,
    _required_cell_status_array,
    _required_layer_cell_id_array,
    _string_array,
    _validate_refractor_velocity_cell_ids,
    _validate_status_array,
)
from app.statics.refraction.artifacts.validation import _validate_result
from app.statics.refraction.artifacts.contract import (
    _CELL_SOLVER_HISTORY_COLUMNS,
    _REFRACTOR_VELOCITY_CELL_COLUMNS,
    ARTIFACT_VERSION,
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    RefractionCellSolverHistoryRow,
    RefractionStaticArtifactError,
)
from app.statics.refraction.artifacts.formatters import (
    _csv_bool,
    _csv_float,
    _csv_grid_float,
    _json_float,
    _required_finite_float,
)
from app.statics.refraction.artifacts.io import (
    _assert_strict_json,
    _validate_no_object_arrays,
    _write_csv_atomic,
    _write_json_atomic,
    _write_npz_atomic,
)
from app.statics.refraction.artifacts.registry import (
    _CELL_VELOCITY_COMPONENT_BY_LAYER,
    _cell_velocity_artifact_names,
    _cell_velocity_layer_kind,
    _request_cell_velocity_layer_kinds,
)
from app.statics.refraction.artifacts.stats import _residual_stat, _stat
from app.statics.refraction.domain.cell_coordinates import (
    effective_refraction_cell_grid_config,
    project_refraction_cell_points,
    refraction_cell_coordinate_metadata_from_config,
)
from app.statics.refraction.domain.cell_grid import assign_observation_midpoint_cells
from app.statics.refraction.domain.cell_grid import build_refraction_cell_grid
from app.statics.refraction.application.design_matrix import (
    LOW_FOLD_CELL_REJECTION_REASON,
    LOW_FOLD_CELL_VELOCITY_STATUS,
)
from app.statics.refraction.domain.layer_config import (
    normalize_refraction_static_layers,
)
from app.statics.refraction.domain.layer_observations import (
    build_refraction_layer_observation_masks_from_arrays,
)
from app.statics.refraction.domain.types import (
    RefractionDatumStaticsResult,
    RefractionLayerKind,
    RefractionLayerSolveResult,
)

def write_refraction_refractor_velocity_cells_csv(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
    layer_kind: RefractionLayerKind | None = None,
) -> None:
    arrays = build_refraction_refractor_velocity_grid_arrays(
        result=result,
        req=req,
        layer_kind=layer_kind,
    )
    rows = _refractor_velocity_cell_rows(arrays)
    _write_csv_atomic(Path(path), _REFRACTOR_VELOCITY_CELL_COLUMNS, rows)

def write_refraction_refractor_velocity_grid_npz(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
    layer_kind: RefractionLayerKind | None = None,
) -> None:
    arrays = build_refraction_refractor_velocity_grid_arrays(
        result=result,
        req=req,
        layer_kind=layer_kind,
    )
    _validate_no_object_arrays(
        arrays,
        artifact_name=REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    )
    _write_npz_atomic(Path(path), arrays)

def write_refraction_refractor_velocity_qc_json(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
    layer_kind: RefractionLayerKind | None = None,
) -> dict[str, Any]:
    payload = build_refraction_refractor_velocity_qc_payload(
        result=result,
        req=req,
        layer_kind=layer_kind,
    )
    _write_json_atomic(Path(path), payload)
    return payload

def write_refraction_cell_solver_history_csv(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
    layer_kind: RefractionLayerKind | None = None,
) -> None:
    rows = build_refraction_cell_solver_history_rows(
        result=result,
        req=req,
        layer_kind=layer_kind,
    )
    csv_rows = [_cell_solver_history_csv_row(row) for row in rows]
    _write_csv_atomic(Path(path), _CELL_SOLVER_HISTORY_COLUMNS, csv_rows)

def build_refraction_refractor_velocity_grid_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    layer_kind: RefractionLayerKind | None = None,
) -> dict[str, np.ndarray]:
    request = RefractionStaticApplyRequest.model_validate(req)
    resolved_layer_kind = _cell_velocity_layer_kind(request, layer_kind=layer_kind)
    result = _cell_velocity_artifact_result_for_layer(
        result=result,
        req=request,
        layer_kind=resolved_layer_kind,
    )
    values = _validate_result(result)
    if not _request_has_cell_velocity_layer(request):
        raise RefractionStaticArtifactError(
            'refractor cell velocity artifacts require solve_cell request mode'
        )
    if not _result_has_cell_velocity_arrays(values.result):
        raise RefractionStaticArtifactError(
            'refractor cell velocity artifacts require cell velocity result arrays'
        )
    refractor_cell = request.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for cell velocity artifacts'
        )
    component = _cell_velocity_component(resolved_layer_kind)

    grid_config = effective_refraction_cell_grid_config(refractor_cell)
    grid = build_refraction_cell_grid(grid_config)
    n_total_cells = int(grid.cell_id.shape[0])
    active_cell_id = _required_cell_int_array(
        values.result.active_cell_id,
        name='active_cell_id',
    )
    inactive_cell_id = _required_cell_int_array(
        values.result.inactive_cell_id,
        name='inactive_cell_id',
    )
    cell_slowness = _required_cell_float_array(
        values.result.cell_bedrock_slowness_s_per_m,
        name='cell_bedrock_slowness_s_per_m',
    )
    cell_velocity = _required_cell_float_array(
        values.result.cell_bedrock_velocity_m_s,
        name='cell_bedrock_velocity_m_s',
    )
    cell_status = _required_cell_status_array(
        values.result.cell_velocity_status,
        name='cell_velocity_status',
    )
    row_midpoint_cell_id = _required_cell_int_array(
        values.result.row_midpoint_cell_id,
        name='row_midpoint_cell_id',
    )
    if row_midpoint_cell_id.shape != (values.n_rows,):
        raise RefractionStaticArtifactError(
            'row_midpoint_cell_id length must match residual rows'
        )
    cell_slowness = _layer_cell_values_for_active_cells(
        cell_slowness,
        active_cell_id=active_cell_id,
        n_total_cells=n_total_cells,
        name='cell_bedrock_slowness_s_per_m',
    )
    cell_velocity = _layer_cell_values_for_active_cells(
        cell_velocity,
        active_cell_id=active_cell_id,
        n_total_cells=n_total_cells,
        name='cell_bedrock_velocity_m_s',
    )
    cell_status = _layer_cell_status_for_active_cells(
        cell_status,
        active_cell_id=active_cell_id,
        n_total_cells=n_total_cells,
        name='cell_velocity_status',
    )
    active_shape = active_cell_id.shape
    for name, array in (
        ('cell_bedrock_slowness_s_per_m', cell_slowness),
        ('cell_bedrock_velocity_m_s', cell_velocity),
        ('cell_velocity_status', cell_status),
    ):
        if array.shape != active_shape:
            raise RefractionStaticArtifactError(
                f'{name} length must match active_cell_id'
            )
    _validate_refractor_velocity_cell_ids(
        grid_cell_id=grid.cell_id,
        active_cell_id=active_cell_id,
        inactive_cell_id=inactive_cell_id,
    )

    active_cell_mask = np.zeros(n_total_cells, dtype=bool)
    v2_m_s = np.full(n_total_cells, np.nan, dtype=np.float64)
    slowness_s_per_m = np.full(n_total_cells, np.nan, dtype=np.float64)
    velocity_status = np.full(n_total_cells, 'inactive', dtype='<U32')
    smoothing_neighbor_count = _active_neighbor_count_by_cell(
        active_cell_id=active_cell_id,
        n_total_cells=n_total_cells,
        number_of_cell_x=grid.number_of_cell_x,
        number_of_cell_y=grid.number_of_cell_y,
    )
    for index, raw_cell_id in enumerate(active_cell_id.tolist()):
        cell_id = int(raw_cell_id)
        active_cell_mask[cell_id] = True
        v2_m_s[cell_id] = float(cell_velocity[index])
        slowness_s_per_m[cell_id] = float(cell_slowness[index])
        velocity_status[cell_id] = str(cell_status[index])
    low_fold_cell_id = _qc_cell_id_array(
        values.result.qc,
        'low_fold_cell_id',
        n_total_cells=n_total_cells,
    )
    if low_fold_cell_id.size:
        velocity_status[low_fold_cell_id] = LOW_FOLD_CELL_VELOCITY_STATUS

    coordinate_mode = str(refractor_cell.coordinate_mode)
    center_aliases = _refractor_cell_center_alias_arrays(
        x_center_m=grid.x_center_m,
        y_center_m=grid.y_center_m,
        coordinate_mode=coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    initial_v2 = np.full(
        n_total_cells,
        _initial_cell_v2_m_s(request, layer_kind=resolved_layer_kind),
        dtype=np.float64,
    )
    v2_update = np.full(n_total_cells, np.nan, dtype=np.float64)
    finite_update = np.isfinite(v2_m_s) & np.isfinite(initial_v2)
    v2_update[finite_update] = v2_m_s[finite_update] - initial_v2[finite_update]
    smoothing_weight = _history_smoothing_weight(
        request,
        layer_kind=resolved_layer_kind,
    )
    smoothing_enabled = bool(smoothing_weight > 0.0)

    n_observations = np.zeros(n_total_cells, dtype=np.int64)
    n_used_observations = np.zeros(n_total_cells, dtype=np.int64)
    cell_candidate_row = _cell_velocity_candidate_row_mask(
        values.result,
        request,
        layer_kind=resolved_layer_kind,
    )
    cell_row_midpoint_cell_id = np.where(
        cell_candidate_row,
        row_midpoint_cell_id,
        -1,
    )
    valid_row_cell = (
        (cell_row_midpoint_cell_id >= 0)
        & (cell_row_midpoint_cell_id < n_total_cells)
    )
    np.add.at(n_observations, cell_row_midpoint_cell_id[valid_row_cell], 1)
    qc_observation_count = _qc_cell_count_array(
        values.result.qc,
        'cell_observation_count',
        n_total_cells=n_total_cells,
    )
    if qc_observation_count is not None:
        n_observations = qc_observation_count
    used_row = valid_row_cell & np.asarray(values.result.used_row_mask, dtype=bool)
    np.add.at(n_used_observations, cell_row_midpoint_cell_id[used_row], 1)
    if np.any(n_used_observations > n_observations):
        raise RefractionStaticArtifactError(
            'used observations per cell cannot exceed total observations per cell'
        )
    n_sources = _unique_observation_endpoint_count_by_cell(
        row_midpoint_cell_id=cell_row_midpoint_cell_id,
        endpoint_id=values.result.row_source_node_id,
        n_total_cells=n_total_cells,
    )
    n_receivers = _unique_observation_endpoint_count_by_cell(
        row_midpoint_cell_id=cell_row_midpoint_cell_id,
        endpoint_id=values.result.row_receiver_node_id,
        n_total_cells=n_total_cells,
    )
    residual_stats = _per_cell_residual_stats_ms(
        row_midpoint_cell_id=cell_row_midpoint_cell_id,
        residual_time_s=values.result.residual_time_s,
        used_row_mask=used_row,
        n_total_cells=n_total_cells,
    )
    status_reason = _refractor_cell_status_reasons(
        velocity_status=velocity_status,
        n_observations=n_observations,
    )

    arrays = {
        'cell_id': np.ascontiguousarray(grid.cell_id, dtype=np.int64),
        'ix': np.ascontiguousarray(grid.ix, dtype=np.int64),
        'iy': np.ascontiguousarray(grid.iy, dtype=np.int64),
        'cell_ix': np.ascontiguousarray(grid.ix, dtype=np.int64),
        'cell_iy': np.ascontiguousarray(grid.iy, dtype=np.int64),
        'coordinate_mode': _string_array(
            np.full(n_total_cells, coordinate_mode, dtype=f'<U{len(coordinate_mode)}')
        ),
        'x_min_m': np.ascontiguousarray(grid.x_min_m, dtype=np.float64),
        'x_max_m': np.ascontiguousarray(grid.x_max_m, dtype=np.float64),
        'y_min_m': np.ascontiguousarray(grid.y_min_m, dtype=np.float64),
        'y_max_m': np.ascontiguousarray(grid.y_max_m, dtype=np.float64),
        'x_center_m': np.ascontiguousarray(grid.x_center_m, dtype=np.float64),
        'y_center_m': np.ascontiguousarray(grid.y_center_m, dtype=np.float64),
        'cell_center_x_m': center_aliases['x_m'],
        'cell_center_y_m': center_aliases['y_m'],
        'cell_center_inline_m': center_aliases['inline_m'],
        'cell_center_crossline_m': center_aliases['crossline_m'],
        'active_cell_mask': np.ascontiguousarray(active_cell_mask, dtype=bool),
        'n_observations_per_cell': np.ascontiguousarray(
            n_observations,
            dtype=np.int64,
        ),
        'n_used_observations_per_cell': np.ascontiguousarray(
            n_used_observations,
            dtype=np.int64,
        ),
        'n_rejected_observations_per_cell': np.ascontiguousarray(
            n_observations - n_used_observations,
            dtype=np.int64,
        ),
        'n_sources_per_cell': np.ascontiguousarray(n_sources, dtype=np.int64),
        'n_receivers_per_cell': np.ascontiguousarray(n_receivers, dtype=np.int64),
        'cell_velocity_layer_kind': _string_array(
            np.full(
                n_total_cells,
                resolved_layer_kind,
                dtype=f'<U{len(resolved_layer_kind)}',
            )
        ),
        'cell_velocity_component': _string_array(
            np.full(n_total_cells, component, dtype=f'<U{len(component)}')
        ),
        'velocity_m_s': np.ascontiguousarray(v2_m_s, dtype=np.float64),
        'v2_m_s': np.ascontiguousarray(v2_m_s, dtype=np.float64),
        'slowness_s_per_m': np.ascontiguousarray(slowness_s_per_m, dtype=np.float64),
        'initial_velocity_m_s': np.ascontiguousarray(initial_v2, dtype=np.float64),
        'initial_v2_m_s': np.ascontiguousarray(initial_v2, dtype=np.float64),
        'velocity_update_from_initial_m_s': np.ascontiguousarray(
            v2_update,
            dtype=np.float64,
        ),
        'v2_update_from_initial_m_s': np.ascontiguousarray(
            v2_update,
            dtype=np.float64,
        ),
        'velocity_status': _string_array(velocity_status),
        'status_reason': _string_array(status_reason),
        'residual_rms_ms': residual_stats['rms'],
        'residual_mad_ms': residual_stats['mad'],
        'residual_mean_ms': residual_stats['mean'],
        'residual_p95_abs_ms': residual_stats['p95_abs'],
        'smoothing_enabled': np.full(n_total_cells, smoothing_enabled, dtype=bool),
        'smoothing_weight': np.full(
            n_total_cells,
            smoothing_weight,
            dtype=np.float64,
        ),
        'smoothing_neighbor_count': np.ascontiguousarray(
            smoothing_neighbor_count,
            dtype=np.int64,
        ),
    }
    _validate_no_object_arrays(
        arrays,
        artifact_name=REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    )
    return arrays

def build_refraction_refractor_velocity_qc_payload(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    layer_kind: RefractionLayerKind | None = None,
) -> dict[str, Any]:
    request = RefractionStaticApplyRequest.model_validate(req)
    resolved_layer_kind = _cell_velocity_layer_kind(request, layer_kind=layer_kind)
    result = _cell_velocity_artifact_result_for_layer(
        result=result,
        req=request,
        layer_kind=resolved_layer_kind,
    )
    arrays = build_refraction_refractor_velocity_grid_arrays(
        result=result,
        req=request,
        layer_kind=resolved_layer_kind,
    )
    refractor_cell = request.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for cell velocity QC'
        )
    grid_config = effective_refraction_cell_grid_config(refractor_cell)
    active_mask = np.asarray(arrays['active_cell_mask'], dtype=bool)
    velocity = np.asarray(arrays['velocity_m_s'], dtype=np.float64)
    active_velocity = velocity[active_mask & np.isfinite(velocity)]
    n_total = int(arrays['cell_id'].shape[0])
    n_active = int(np.count_nonzero(active_mask))
    n_observations_in_grid = int(np.sum(arrays['n_observations_per_cell']))
    if request.model.method == 'multilayer_time_term':
        n_valid_observations = int(
            np.count_nonzero(
                _cell_velocity_candidate_row_mask(
                    result,
                    request,
                    layer_kind=resolved_layer_kind,
                )
            )
        )
    else:
        n_valid_observations = int(
            np.count_nonzero(result.valid_observation_mask_sorted)
        )
    default_outside_observations = max(0, n_valid_observations - n_observations_in_grid)
    n_used = int(np.sum(arrays['n_used_observations_per_cell']))
    n_low_fold = int(
        np.count_nonzero(
            np.asarray(arrays['velocity_status']).astype(str, copy=False)
            == LOW_FOLD_CELL_VELOCITY_STATUS
        )
    )
    n_low_fold_rejected = int(
        np.sum(
            np.asarray(arrays['n_rejected_observations_per_cell'], dtype=np.int64)[
                np.asarray(arrays['velocity_status']).astype(str, copy=False)
                == LOW_FOLD_CELL_VELOCITY_STATUS
            ]
        )
    )
    n_smoothing_rows = _qc_int(
        result.qc,
        'n_cell_smoothing_rows',
        default=_estimated_cell_smoothing_rows(
            active_cell_mask=active_mask,
            number_of_cell_x=int(grid_config.number_of_cell_x),
            number_of_cell_y=int(grid_config.number_of_cell_y),
            velocity_smoothing_weight=float(
                _history_smoothing_weight(
                    request,
                    layer_kind=resolved_layer_kind,
                )
            ),
        ),
    )
    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'bedrock_velocity_mode': 'solve_cell',
        'cell_velocity_layer_kind': resolved_layer_kind,
        'cell_velocity_component': _cell_velocity_component(resolved_layer_kind),
        'cell_assignment_mode': refractor_cell.assignment_mode,
        **refraction_cell_coordinate_metadata_from_config(refractor_cell),
        'outside_grid_policy': refractor_cell.outside_grid_policy,
        'number_of_cell_x': int(grid_config.number_of_cell_x),
        'number_of_cell_y': int(grid_config.number_of_cell_y),
        'size_of_cell_x_m': float(grid_config.size_of_cell_x_m),
        'size_of_cell_y_m': _json_float(grid_config.size_of_cell_y_m),
        'n_total_cells': n_total,
        'n_active_cells': n_active,
        'n_inactive_cells': int(n_total - n_active),
        'min_observations_per_cell': _qc_int(
            result.qc,
            'min_observations_per_cell',
            default=_cell_velocity_min_observations_per_cell(
                request,
                layer_kind=resolved_layer_kind,
            ),
        ),
        'n_low_fold_cells': _qc_int(
            result.qc,
            'n_low_fold_cells',
            default=n_low_fold,
        ),
        'n_observations_outside_grid': _qc_int(
            result.qc,
            'n_observations_outside_grid',
            default=default_outside_observations,
        ),
        'n_observations_rejected_by_low_fold_cell': _qc_int(
            result.qc,
            'n_observations_rejected_by_low_fold_cell',
            default=n_low_fold_rejected,
        ),
        'low_fold_cell_rejection_reason': str(
            result.qc.get(
                'low_fold_cell_rejection_reason',
                LOW_FOLD_CELL_REJECTION_REASON,
            )
        ),
        'n_used_observations': n_used,
        'velocity_min_m_s': _stat(active_velocity, 'min'),
        'velocity_median_m_s': _stat(active_velocity, 'median'),
        'velocity_max_m_s': _stat(active_velocity, 'max'),
        'velocity_smoothing_weight': _history_smoothing_weight(
            request,
            layer_kind=resolved_layer_kind,
        ),
        'smoothing_reference_distance_m': _qc_optional_float(
            result.qc,
            'smoothing_reference_distance_m',
            default=refractor_cell.smoothing_reference_distance_m,
        ),
        'n_cell_smoothing_rows': n_smoothing_rows,
    }
    _assert_strict_json(
        payload,
        artifact_name=_cell_velocity_artifact_names(resolved_layer_kind).qc_json,
    )
    return payload

def build_refraction_cell_solver_history_rows(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    layer_kind: RefractionLayerKind | None = None,
) -> list[RefractionCellSolverHistoryRow]:
    request = RefractionStaticApplyRequest.model_validate(req)
    resolved_layer_kind = _cell_velocity_layer_kind(request, layer_kind=layer_kind)
    result = _cell_velocity_artifact_result_for_layer(
        result=result,
        req=request,
        layer_kind=resolved_layer_kind,
    )
    values = _validate_result(result)
    if not _request_has_cell_velocity_layer(request):
        raise RefractionStaticArtifactError(
            'cell solver history artifact requires solve_cell request mode'
        )
    if not _result_has_cell_velocity_arrays(values.result):
        raise RefractionStaticArtifactError(
            'cell solver history artifact requires cell velocity result arrays'
        )

    arrays = build_refraction_refractor_velocity_grid_arrays(
        result=values.result,
        req=request,
        layer_kind=resolved_layer_kind,
    )
    cell_counts = _cell_solver_history_cell_counts(arrays)
    initial_v2 = _initial_cell_v2_m_s(request, layer_kind=resolved_layer_kind)
    final_velocity = np.asarray(arrays['velocity_m_s'], dtype=np.float64)
    active_mask = np.asarray(arrays['active_cell_mask'], dtype=bool)
    active_final_v2 = final_velocity[active_mask & np.isfinite(final_velocity)]
    update = np.asarray(
        arrays['velocity_update_from_initial_m_s'],
        dtype=np.float64,
    )
    active_update = update[active_mask & np.isfinite(update)]

    cell_candidate_row = _cell_velocity_candidate_row_mask(
        values.result,
        request,
        layer_kind=resolved_layer_kind,
    )
    cell_used_row = cell_candidate_row & np.asarray(
        values.result.used_row_mask,
        dtype=bool,
    )
    n_candidate = int(np.count_nonzero(cell_candidate_row))
    n_used = int(np.count_nonzero(cell_used_row))
    n_robust_rejected = int(
        np.count_nonzero(
            cell_candidate_row
            & np.asarray(values.result.rejected_by_robust_mask, dtype=bool)
        )
    )
    smoothing_weight = _history_smoothing_weight(
        request,
        layer_kind=resolved_layer_kind,
    )
    damping_weight = float(request.solver.damping)
    robust_threshold = float(request.solver.robust.threshold)
    robust_iteration_count = _history_robust_iteration_count(
        values.result,
        request,
    )
    residual_stats = _history_residual_stats_ms(
        values.result,
        used_row_mask=cell_used_row,
    )

    return [
        RefractionCellSolverHistoryRow(
            iteration=0,
            stage='initial',
            n_candidate_observations=n_candidate,
            n_used_observations=n_candidate,
            n_rejected_observations=0,
            n_active_cells=cell_counts['active'],
            n_low_fold_cells=cell_counts['low_fold'],
            n_empty_cells=cell_counts['empty'],
            residual_rms_ms=None,
            residual_mad_ms=None,
            max_abs_residual_ms=None,
            median_velocity_m_s=initial_v2,
            median_v2_m_s=initial_v2,
            min_velocity_m_s=initial_v2,
            min_v2_m_s=initial_v2,
            max_velocity_m_s=initial_v2,
            max_v2_m_s=initial_v2,
            max_abs_velocity_update_m_s=0.0,
            max_abs_v2_update_m_s=0.0,
            smoothing_weight=smoothing_weight,
            damping_weight=damping_weight,
            robust_threshold=robust_threshold,
            converged=False,
            convergence_reason='initial_state',
        ),
        RefractionCellSolverHistoryRow(
            iteration=max(1, robust_iteration_count),
            stage='final',
            n_candidate_observations=n_candidate,
            n_used_observations=n_used,
            n_rejected_observations=n_robust_rejected,
            n_active_cells=cell_counts['active'],
            n_low_fold_cells=cell_counts['low_fold'],
            n_empty_cells=cell_counts['empty'],
            residual_rms_ms=residual_stats['rms'],
            residual_mad_ms=residual_stats['mad'],
            max_abs_residual_ms=residual_stats['max_abs'],
            median_velocity_m_s=_stat(active_final_v2, 'median'),
            median_v2_m_s=_stat(active_final_v2, 'median'),
            min_velocity_m_s=_stat(active_final_v2, 'min'),
            min_v2_m_s=_stat(active_final_v2, 'min'),
            max_velocity_m_s=_stat(active_final_v2, 'max'),
            max_v2_m_s=_stat(active_final_v2, 'max'),
            max_abs_velocity_update_m_s=_history_max_abs(active_update),
            max_abs_v2_update_m_s=_history_max_abs(active_update),
            smoothing_weight=smoothing_weight,
            damping_weight=damping_weight,
            robust_threshold=robust_threshold,
            converged=True,
            convergence_reason=_history_convergence_reason(
                robust_iteration_count=robust_iteration_count,
                n_robust_rejected_observations=n_robust_rejected,
                smoothing_weight=smoothing_weight,
            ),
        ),
    ]

def _request_summary(req: RefractionStaticApplyRequest) -> dict[str, Any]:
    return {
        'file_id': req.file_id,
        'key1_byte': int(req.key1_byte),
        'key2_byte': int(req.key2_byte),
        'pick_source_kind': req.pick_source.kind,
        'model_method': req.model.method,
        'apply_mode': req.apply.mode,
        'register_corrected_file': bool(req.apply.register_corrected_file),
    }

def _layer_velocity_modes_for_request(
    req: RefractionStaticApplyRequest,
) -> dict[str, str]:
    if req.model.method != 'multilayer_time_term':
        return {}
    return {
        str(config.kind): str(config.velocity_mode)
        for config in normalize_refraction_static_layers(req.model)
    }

def _request_has_cell_velocity_layer(req: RefractionStaticApplyRequest) -> bool:
    return bool(_request_cell_velocity_layer_kinds(req))

def _request_cell_velocity_layer(
    req: RefractionStaticApplyRequest,
    *,
    layer_kind: RefractionLayerKind | None = None,
) -> Any | None:
    if req.model.method != 'multilayer_time_term':
        return None
    for layer in normalize_refraction_static_layers(req.model):
        if layer.velocity_mode != 'solve_cell':
            continue
        if layer_kind is None or layer.kind == layer_kind:
            return layer
    return None

def _result_has_cell_velocity_arrays(result: RefractionDatumStaticsResult) -> bool:
    return (
        result.active_cell_id is not None
        and result.inactive_cell_id is not None
        and result.cell_bedrock_velocity_m_s is not None
        and result.cell_bedrock_slowness_s_per_m is not None
        and result.cell_velocity_status is not None
    )

def _cell_velocity_artifact_result_for_layer(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    layer_kind: RefractionLayerKind,
) -> RefractionDatumStaticsResult:
    if req.model.method != 'multilayer_time_term':
        return result

    layer = _cell_velocity_layer_result(result, layer_kind)
    if layer is None:
        if (
            len(_request_cell_velocity_layer_kinds(req)) == 1
            and _result_has_cell_velocity_arrays(result)
        ):
            return result
        raise RefractionStaticArtifactError(
            f'{layer_kind} cell velocity artifacts require layer solve result arrays'
        )

    return _cell_velocity_artifact_result_from_layer(
        result=result,
        req=req,
        layer=layer,
    )

def _cell_velocity_layer_result(
    result: RefractionDatumStaticsResult,
    layer_kind: RefractionLayerKind,
) -> RefractionLayerSolveResult | None:
    for layer in result.layer_results or ():
        if layer.layer_kind == layer_kind:
            return layer
    return None

def _cell_velocity_artifact_result_from_layer(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    layer: RefractionLayerSolveResult,
) -> RefractionDatumStaticsResult:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    active_cell_id = _required_layer_cell_id_array(
        layer.active_cell_id,
        name=f'{layer.layer_kind}.active_cell_id',
    )
    inactive_cell_id = _required_layer_cell_id_array(
        layer.inactive_cell_id,
        name=f'{layer.layer_kind}.inactive_cell_id',
    )
    n_total_cells = _cell_velocity_n_total_cells(req)
    cell_velocity = _layer_cell_values_for_active_cells(
        layer.cell_velocity_m_s,
        active_cell_id=active_cell_id,
        n_total_cells=n_total_cells,
        name=f'{layer.layer_kind}.cell_velocity_m_s',
    )
    cell_slowness = _layer_cell_values_for_active_cells(
        layer.cell_slowness_s_per_m,
        active_cell_id=active_cell_id,
        n_total_cells=n_total_cells,
        name=f'{layer.layer_kind}.cell_slowness_s_per_m',
    )
    cell_status = _layer_cell_status_for_active_cells(
        layer.cell_velocity_status,
        active_cell_id=active_cell_id,
        n_total_cells=n_total_cells,
        name=f'{layer.layer_kind}.cell_velocity_status',
    )
    velocity_summary = _stat(cell_velocity, 'median')
    slowness_summary = _stat(cell_slowness, 'median')
    qc = {**result.qc, **layer.qc}
    if 'layers' in result.qc:
        qc['layers'] = result.qc['layers']
    return replace(
        result,
        bedrock_velocity_mode='solve_cell',
        bedrock_velocity_m_s=(
            result.bedrock_velocity_m_s
            if velocity_summary is None
            else velocity_summary
        ),
        bedrock_slowness_s_per_m=(
            result.bedrock_slowness_s_per_m
            if slowness_summary is None
            else slowness_summary
        ),
        active_cell_id=active_cell_id,
        inactive_cell_id=inactive_cell_id,
        cell_bedrock_velocity_m_s=cell_velocity,
        cell_bedrock_slowness_s_per_m=cell_slowness,
        cell_velocity_status=cell_status,
        row_midpoint_cell_id=_row_midpoint_cell_id_for_cell_velocity_layer(
            result=result,
            req=req,
            layer=layer,
            n_rows=n_rows,
        ),
        modeled_pick_time_s=_layer_trace_float_array(
            layer.trace_predicted_time_s_sorted,
            n_rows=n_rows,
            name=f'{layer.layer_kind}.trace_predicted_time_s_sorted',
        ),
        residual_time_s=_layer_trace_float_array(
            layer.trace_residual_s_sorted,
            n_rows=n_rows,
            name=f'{layer.layer_kind}.trace_residual_s_sorted',
        ),
        used_row_mask=_layer_trace_bool_array(
            layer.used_observation_mask_sorted,
            n_rows=n_rows,
            name=f'{layer.layer_kind}.used_observation_mask_sorted',
        ),
        rejected_by_robust_mask=_layer_trace_bool_array(
            (
                np.zeros(n_rows, dtype=bool)
                if layer.rejected_by_robust_mask_sorted is None
                else layer.rejected_by_robust_mask_sorted
            ),
            n_rows=n_rows,
            name=f'{layer.layer_kind}.rejected_by_robust_mask_sorted',
        ),
        qc=qc,
    )

def _cell_velocity_n_total_cells(req: RefractionStaticApplyRequest) -> int:
    refractor_cell = req.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for cell velocity artifacts'
        )
    grid = build_refraction_cell_grid(
        effective_refraction_cell_grid_config(refractor_cell)
    )
    return int(grid.cell_id.shape[0])

def _layer_cell_values_for_active_cells(
    value: object,
    *,
    active_cell_id: np.ndarray,
    n_total_cells: int,
    name: str,
) -> np.ndarray:
    if value is None:
        raise RefractionStaticArtifactError(f'{name} is required')
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise RefractionStaticArtifactError(f'{name} must be one-dimensional')
    if array.shape == active_cell_id.shape:
        return np.ascontiguousarray(array, dtype=np.float64)
    if array.shape == (int(n_total_cells),):
        return np.ascontiguousarray(array[active_cell_id], dtype=np.float64)
    raise RefractionStaticArtifactError(
        f'{name} must match active_cell_id length or total cell count'
    )

def _layer_cell_status_for_active_cells(
    value: object,
    *,
    active_cell_id: np.ndarray,
    n_total_cells: int,
    name: str,
) -> np.ndarray:
    if value is None:
        return _string_array(np.full(active_cell_id.shape, 'solved', dtype='<U6'))
    status = _string_array(value)
    if status.shape == active_cell_id.shape:
        _validate_status_array(status, name=name)
        return status
    if status.shape == (int(n_total_cells),):
        out = _string_array(status[active_cell_id])
        _validate_status_array(out, name=name)
        return out
    raise RefractionStaticArtifactError(
        f'{name} must match active_cell_id length or total cell count'
    )

def _layer_trace_float_array(value: object, *, n_rows: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (int(n_rows),):
        raise RefractionStaticArtifactError(
            f'{name} length must match residual rows'
        )
    return np.ascontiguousarray(array, dtype=np.float64)

def _layer_trace_bool_array(value: object, *, n_rows: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != (int(n_rows),):
        raise RefractionStaticArtifactError(
            f'{name} length must match residual rows'
        )
    return np.ascontiguousarray(array, dtype=bool)

def _row_midpoint_cell_id_for_cell_velocity_layer(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    layer: RefractionLayerSolveResult,
    n_rows: int,
) -> np.ndarray:
    if layer.row_midpoint_cell_id is not None:
        array = np.asarray(layer.row_midpoint_cell_id, dtype=np.int64)
        if array.shape == (int(n_rows),):
            return np.ascontiguousarray(array, dtype=np.int64)
    if result.row_midpoint_cell_id is not None:
        array = np.asarray(result.row_midpoint_cell_id, dtype=np.int64)
        if array.shape == (int(n_rows),):
            return np.ascontiguousarray(array, dtype=np.int64)
    return _compute_row_midpoint_cell_id(result=result, req=req, n_rows=n_rows)

def _compute_row_midpoint_cell_id(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    n_rows: int,
) -> np.ndarray:
    refractor_cell = req.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for cell velocity artifacts'
        )
    source_x, source_y = _row_endpoint_xy(result, endpoint='source', n_rows=n_rows)
    receiver_x, receiver_y = _row_endpoint_xy(
        result,
        endpoint='receiver',
        n_rows=n_rows,
    )
    source_projected = project_refraction_cell_points(
        x_m=source_x,
        y_m=source_y,
        mode=refractor_cell.coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    receiver_projected = project_refraction_cell_points(
        x_m=receiver_x,
        y_m=receiver_y,
        mode=refractor_cell.coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    grid = build_refraction_cell_grid(
        effective_refraction_cell_grid_config(refractor_cell)
    )
    assignment = assign_observation_midpoint_cells(
        grid,
        source_x_m=source_projected.x_m,
        source_y_m=source_projected.y_m,
        receiver_x_m=receiver_projected.x_m,
        receiver_y_m=receiver_projected.y_m,
    )
    return np.ascontiguousarray(assignment.cell_id, dtype=np.int64)

def _row_endpoint_xy(
    result: RefractionDatumStaticsResult,
    *,
    endpoint: str,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    if endpoint == 'source':
        row_key = result.row_source_endpoint_key
        endpoint_key = result.source_endpoint_key
        x_m = result.source_x_m
        y_m = result.source_y_m
    elif endpoint == 'receiver':
        row_key = result.row_receiver_endpoint_key
        endpoint_key = result.receiver_endpoint_key
        x_m = result.receiver_x_m
        y_m = result.receiver_y_m
    else:
        raise RefractionStaticArtifactError(f'unsupported endpoint: {endpoint}')
    if row_key is None:
        raise RefractionStaticArtifactError(
            f'row_{endpoint}_endpoint_key is required for layer cell artifacts'
        )
    row_key_array = np.asarray(row_key, dtype=object)
    if row_key_array.shape != (int(n_rows),):
        raise RefractionStaticArtifactError(
            f'row_{endpoint}_endpoint_key length must match residual rows'
        )
    endpoint_key_array = np.asarray(endpoint_key, dtype=object)
    x_array = np.asarray(x_m, dtype=np.float64)
    y_array = np.asarray(y_m, dtype=np.float64)
    if endpoint_key_array.shape != x_array.shape or x_array.shape != y_array.shape:
        raise RefractionStaticArtifactError(
            f'{endpoint} endpoint key and coordinate arrays must match'
        )
    index_by_key = {
        str(key): index for index, key in enumerate(endpoint_key_array.tolist())
    }
    indices: list[int] = []
    for key in row_key_array.tolist():
        try:
            indices.append(index_by_key[str(key)])
        except KeyError as exc:
            raise RefractionStaticArtifactError(
                f'row {endpoint} endpoint key {key!r} is missing from endpoint table'
            ) from exc
    index_array = np.asarray(indices, dtype=np.int64)
    return (
        np.ascontiguousarray(x_array[index_array], dtype=np.float64),
        np.ascontiguousarray(y_array[index_array], dtype=np.float64),
    )

def _cell_velocity_candidate_row_mask(
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    *,
    layer_kind: RefractionLayerKind | None = None,
) -> np.ndarray:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    layer = _request_cell_velocity_layer(req, layer_kind=layer_kind)
    if layer is None:
        return np.ones(n_rows, dtype=bool)
    distance = np.asarray(result.row_distance_m, dtype=np.float64)
    if distance.shape != (n_rows,):
        raise RefractionStaticArtifactError(
            'row_distance_m length must match residual rows'
        )
    valid_observation = np.asarray(result.valid_observation_mask_sorted, dtype=bool)
    if valid_observation.shape != (n_rows,):
        raise RefractionStaticArtifactError(
            'valid_observation_mask_sorted length must match residual rows'
        )
    try:
        masks = build_refraction_layer_observation_masks_from_arrays(
            base_valid_mask_sorted=valid_observation,
            offset_m_sorted=distance,
            rejection_reason_sorted=np.full(n_rows, 'ok', dtype='<U32'),
            model=req.model,
        )
        mask = masks.layer_used_mask_sorted[layer.kind]
    except (KeyError, ValueError) as exc:
        raise RefractionStaticArtifactError(
            f'could not build cell velocity layer observation mask for {layer.kind}'
        ) from exc
    return np.ascontiguousarray(mask, dtype=bool)

def _cell_velocity_component(layer_kind: RefractionLayerKind) -> str:
    try:
        return _CELL_VELOCITY_COMPONENT_BY_LAYER[layer_kind]
    except KeyError as exc:
        raise RefractionStaticArtifactError(
            f'unsupported cell velocity layer kind: {layer_kind}'
        ) from exc

def _cell_velocity_layer_bounds(
    req: RefractionStaticApplyRequest,
    layer: Any,
) -> tuple[float | None, float | None]:
    min_velocity = getattr(layer, 'min_velocity_m_s', None)
    max_velocity = getattr(layer, 'max_velocity_m_s', None)
    if layer.kind == 'v2_t1':
        if min_velocity is None:
            min_velocity = req.model.min_bedrock_velocity_m_s
        if max_velocity is None:
            max_velocity = req.model.max_bedrock_velocity_m_s
    return min_velocity, max_velocity

def _cell_velocity_min_observations_per_cell(
    req: RefractionStaticApplyRequest,
    *,
    layer_kind: RefractionLayerKind | None = None,
) -> int:
    layer = _request_cell_velocity_layer(req, layer_kind=layer_kind)
    if layer is not None and layer.min_observations_per_cell is not None:
        return int(layer.min_observations_per_cell)
    refractor_cell = req.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for cell velocity artifacts'
        )
    return int(refractor_cell.min_observations_per_cell)

def _refractor_velocity_cell_rows(
    arrays: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_cells = int(arrays['cell_id'].shape[0])
    for index in range(n_cells):
        rows.append(
            {
                'cell_id': int(arrays['cell_id'][index]),
                'ix': int(arrays['ix'][index]),
                'iy': int(arrays['iy'][index]),
                'cell_ix': int(arrays['cell_ix'][index]),
                'cell_iy': int(arrays['cell_iy'][index]),
                'coordinate_mode': str(arrays['coordinate_mode'][index]),
                'x_min_m': _csv_grid_float(arrays['x_min_m'][index]),
                'x_max_m': _csv_grid_float(arrays['x_max_m'][index]),
                'y_min_m': _csv_grid_float(arrays['y_min_m'][index]),
                'y_max_m': _csv_grid_float(arrays['y_max_m'][index]),
                'x_center_m': _csv_grid_float(arrays['x_center_m'][index]),
                'y_center_m': _csv_grid_float(arrays['y_center_m'][index]),
                'cell_center_x_m': _csv_grid_float(
                    arrays['cell_center_x_m'][index]
                ),
                'cell_center_y_m': _csv_grid_float(
                    arrays['cell_center_y_m'][index]
                ),
                'cell_center_inline_m': _csv_grid_float(
                    arrays['cell_center_inline_m'][index]
                ),
                'cell_center_crossline_m': _csv_grid_float(
                    arrays['cell_center_crossline_m'][index]
                ),
                'active': _csv_bool(arrays['active_cell_mask'][index]),
                'n_observations': int(
                    arrays['n_observations_per_cell'][index]
                ),
                'n_used_observations': int(
                    arrays['n_used_observations_per_cell'][index]
                ),
                'n_rejected_observations': int(
                    arrays['n_rejected_observations_per_cell'][index]
                ),
                'n_sources': int(arrays['n_sources_per_cell'][index]),
                'n_receivers': int(arrays['n_receivers_per_cell'][index]),
                'cell_velocity_layer_kind': str(
                    arrays['cell_velocity_layer_kind'][index]
                ),
                'cell_velocity_component': str(
                    arrays['cell_velocity_component'][index]
                ),
                'velocity_m_s': _csv_float(arrays['velocity_m_s'][index]),
                'v2_m_s': _csv_float(arrays['v2_m_s'][index]),
                'slowness_s_per_m': _csv_float(
                    arrays['slowness_s_per_m'][index]
                ),
                'initial_velocity_m_s': _csv_float(
                    arrays['initial_velocity_m_s'][index]
                ),
                'initial_v2_m_s': _csv_float(arrays['initial_v2_m_s'][index]),
                'velocity_update_from_initial_m_s': _csv_float(
                    arrays['velocity_update_from_initial_m_s'][index]
                ),
                'v2_update_from_initial_m_s': _csv_float(
                    arrays['v2_update_from_initial_m_s'][index]
                ),
                'velocity_status': str(arrays['velocity_status'][index]),
                'status_reason': str(arrays['status_reason'][index]),
                'residual_rms_ms': _csv_float(arrays['residual_rms_ms'][index]),
                'residual_mad_ms': _csv_float(arrays['residual_mad_ms'][index]),
                'residual_mean_ms': _csv_float(arrays['residual_mean_ms'][index]),
                'residual_p95_abs_ms': _csv_float(
                    arrays['residual_p95_abs_ms'][index]
                ),
                'smoothing_enabled': _csv_bool(arrays['smoothing_enabled'][index]),
                'smoothing_weight': _csv_float(arrays['smoothing_weight'][index]),
                'smoothing_neighbor_count': int(
                    arrays['smoothing_neighbor_count'][index]
                ),
            }
        )
    return rows

def _cell_solver_history_csv_row(
    row: RefractionCellSolverHistoryRow,
) -> dict[str, object]:
    return {
        'iteration': int(row.iteration),
        'stage': row.stage,
        'n_candidate_observations': int(row.n_candidate_observations),
        'n_used_observations': int(row.n_used_observations),
        'n_rejected_observations': int(row.n_rejected_observations),
        'n_active_cells': int(row.n_active_cells),
        'n_low_fold_cells': int(row.n_low_fold_cells),
        'n_empty_cells': int(row.n_empty_cells),
        'residual_rms_ms': _csv_float(row.residual_rms_ms),
        'residual_mad_ms': _csv_float(row.residual_mad_ms),
        'max_abs_residual_ms': _csv_float(row.max_abs_residual_ms),
        'median_velocity_m_s': _csv_float(row.median_velocity_m_s),
        'median_v2_m_s': _csv_float(row.median_v2_m_s),
        'min_velocity_m_s': _csv_float(row.min_velocity_m_s),
        'min_v2_m_s': _csv_float(row.min_v2_m_s),
        'max_velocity_m_s': _csv_float(row.max_velocity_m_s),
        'max_v2_m_s': _csv_float(row.max_v2_m_s),
        'max_abs_velocity_update_m_s': _csv_float(
            row.max_abs_velocity_update_m_s
        ),
        'max_abs_v2_update_m_s': _csv_float(row.max_abs_v2_update_m_s),
        'smoothing_weight': _csv_float(row.smoothing_weight),
        'damping_weight': _csv_float(row.damping_weight),
        'robust_threshold': _csv_float(row.robust_threshold),
        'converged': _csv_bool(row.converged),
        'convergence_reason': row.convergence_reason,
    }

def _cell_solver_history_cell_counts(
    arrays: dict[str, np.ndarray],
) -> dict[str, int]:
    active_mask = np.asarray(arrays['active_cell_mask'], dtype=bool)
    velocity_status = np.asarray(arrays['velocity_status']).astype(str, copy=False)
    n_observations = np.asarray(arrays['n_observations_per_cell'], dtype=np.int64)
    return {
        'active': int(np.count_nonzero(active_mask)),
        'low_fold': int(
            np.count_nonzero(velocity_status == LOW_FOLD_CELL_VELOCITY_STATUS)
        ),
        'empty': int(np.count_nonzero(n_observations == 0)),
    }

def _history_residual_stats_ms(
    result: RefractionDatumStaticsResult,
    *,
    used_row_mask: np.ndarray,
) -> dict[str, float | None]:
    residual_ms = (
        np.asarray(result.residual_time_s, dtype=np.float64)[
            np.asarray(used_row_mask, dtype=bool)
        ]
        * 1000.0
    )
    return {
        'rms': _residual_stat(residual_ms, 'rms'),
        'mad': _residual_stat(residual_ms, 'mad'),
        'max_abs': _residual_stat(residual_ms, 'max_abs'),
    }

def _history_max_abs(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.max(np.abs(finite)))

def _initial_cell_v2_m_s(
    req: RefractionStaticApplyRequest,
    *,
    layer_kind: RefractionLayerKind | None = None,
) -> float:
    layer = _request_cell_velocity_layer(req, layer_kind=layer_kind)
    if layer is not None:
        value = getattr(layer, 'initial_velocity_m_s', None)
        if value is not None:
            return float(value)
        if layer.kind == 'v2_t1' and req.model.initial_bedrock_velocity_m_s is not None:
            return float(req.model.initial_bedrock_velocity_m_s)
        min_velocity, max_velocity = _cell_velocity_layer_bounds(req, layer)
        if min_velocity is not None and max_velocity is not None:
            return 0.5 * (float(min_velocity) + float(max_velocity))
        raise RefractionStaticArtifactError(
            'cell velocity layer initial_velocity_m_s is required'
        )

    value = req.model.initial_bedrock_velocity_m_s
    if value is not None:
        return float(value)
    return 0.5 * (
        float(req.model.min_bedrock_velocity_m_s)
        + float(req.model.max_bedrock_velocity_m_s)
    )

def _history_smoothing_weight(
    req: RefractionStaticApplyRequest,
    *,
    layer_kind: RefractionLayerKind | None = None,
) -> float:
    layer = _request_cell_velocity_layer(req, layer_kind=layer_kind)
    if layer is not None and layer.smoothing_weight is not None:
        return float(layer.smoothing_weight)
    refractor_cell = req.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for cell solver history'
        )
    return float(refractor_cell.velocity_smoothing_weight)

def _history_robust_iteration_count(
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> int:
    raw = result.qc.get('robust_iteration_count')
    if raw is not None:
        return int(raw)
    if req.solver.robust.enabled and np.count_nonzero(result.rejected_by_robust_mask):
        return 1
    return 0

def _history_convergence_reason(
    *,
    robust_iteration_count: int,
    n_robust_rejected_observations: int,
    smoothing_weight: float,
) -> str:
    if robust_iteration_count > 0 or n_robust_rejected_observations > 0:
        return 'robust_reweight_converged'
    if smoothing_weight > 0.0:
        return 'smoothed_least_squares_converged'
    return 'least_squares_converged'

def _refractor_cell_center_alias_arrays(
    *,
    x_center_m: np.ndarray,
    y_center_m: np.ndarray,
    coordinate_mode: str,
    line_origin_x_m: float | None,
    line_origin_y_m: float | None,
    line_azimuth_deg: float | None,
) -> dict[str, np.ndarray]:
    x_center = np.asarray(x_center_m, dtype=np.float64)
    y_center = np.asarray(y_center_m, dtype=np.float64)
    if x_center.shape != y_center.shape:
        raise RefractionStaticArtifactError(
            'refractor cell center coordinate shape mismatch'
        )

    inline = np.full(x_center.shape, np.nan, dtype=np.float64)
    crossline = np.full(x_center.shape, np.nan, dtype=np.float64)
    center_x = np.ascontiguousarray(x_center, dtype=np.float64)
    center_y = np.ascontiguousarray(y_center, dtype=np.float64)
    if coordinate_mode == 'line_2d_projected':
        origin_x = _required_finite_float(
            line_origin_x_m,
            name='model.refractor_cell.line_origin_x_m',
        )
        origin_y = _required_finite_float(
            line_origin_y_m,
            name='model.refractor_cell.line_origin_y_m',
        )
        azimuth_deg = _required_finite_float(
            line_azimuth_deg,
            name='model.refractor_cell.line_azimuth_deg',
        )
        inline = np.ascontiguousarray(x_center, dtype=np.float64)
        crossline = np.ascontiguousarray(y_center, dtype=np.float64)
        azimuth_rad = np.deg2rad(azimuth_deg)
        inline_unit_x = float(np.sin(azimuth_rad))
        inline_unit_y = float(np.cos(azimuth_rad))
        center_x = np.ascontiguousarray(
            origin_x + inline * inline_unit_x + crossline * inline_unit_y,
            dtype=np.float64,
        )
        center_y = np.ascontiguousarray(
            origin_y + inline * inline_unit_y - crossline * inline_unit_x,
            dtype=np.float64,
        )

    return {
        'x_m': center_x,
        'y_m': center_y,
        'inline_m': np.ascontiguousarray(inline, dtype=np.float64),
        'crossline_m': np.ascontiguousarray(crossline, dtype=np.float64),
    }

def _unique_observation_endpoint_count_by_cell(
    *,
    row_midpoint_cell_id: np.ndarray,
    endpoint_id: np.ndarray,
    n_total_cells: int,
) -> np.ndarray:
    row_cell = np.asarray(row_midpoint_cell_id, dtype=np.int64)
    endpoint = np.asarray(endpoint_id, dtype=np.int64)
    if row_cell.shape != endpoint.shape:
        raise RefractionStaticArtifactError(
            'row endpoint arrays must match row_midpoint_cell_id shape'
        )
    counts = np.zeros(int(n_total_cells), dtype=np.int64)
    for cell_id in range(int(n_total_cells)):
        in_cell = row_cell == cell_id
        if np.any(in_cell):
            counts[cell_id] = int(np.unique(endpoint[in_cell]).shape[0])
    return np.ascontiguousarray(counts, dtype=np.int64)

def _refractor_cell_status_reasons(
    *,
    velocity_status: np.ndarray,
    n_observations: np.ndarray,
) -> np.ndarray:
    status = np.asarray(velocity_status).astype(str, copy=False)
    counts = np.asarray(n_observations, dtype=np.int64)
    if status.shape != counts.shape:
        raise RefractionStaticArtifactError(
            'velocity_status and n_observations shape mismatch'
        )
    reasons: list[str] = []
    for value, count in zip(status.tolist(), counts.tolist(), strict=True):
        if value == LOW_FOLD_CELL_VELOCITY_STATUS:
            reasons.append(LOW_FOLD_CELL_REJECTION_REASON)
        elif value == 'inactive':
            reasons.append('no_observations' if int(count) == 0 else 'inactive_cell')
        else:
            reasons.append(str(value))
    return _string_array(reasons)

def _active_neighbor_count_by_cell(
    *,
    active_cell_id: np.ndarray,
    n_total_cells: int,
    number_of_cell_x: int,
    number_of_cell_y: int,
) -> np.ndarray:
    active = {int(value) for value in np.asarray(active_cell_id).tolist()}
    counts = np.zeros(int(n_total_cells), dtype=np.int64)
    for cell_id in active:
        ix = cell_id % int(number_of_cell_x)
        iy = cell_id // int(number_of_cell_x)
        neighbors = []
        if ix > 0:
            neighbors.append(cell_id - 1)
        if ix + 1 < int(number_of_cell_x):
            neighbors.append(cell_id + 1)
        if iy > 0:
            neighbors.append(cell_id - int(number_of_cell_x))
        if iy + 1 < int(number_of_cell_y):
            neighbors.append(cell_id + int(number_of_cell_x))
        counts[cell_id] = sum(1 for neighbor in neighbors if neighbor in active)
    return np.ascontiguousarray(counts, dtype=np.int64)

def _estimated_cell_smoothing_rows(
    *,
    active_cell_mask: np.ndarray,
    number_of_cell_x: int,
    number_of_cell_y: int,
    velocity_smoothing_weight: float,
) -> int:
    if float(velocity_smoothing_weight) == 0.0:
        return 0
    active = {
        index
        for index, enabled in enumerate(np.asarray(active_cell_mask, dtype=bool))
        if bool(enabled)
    }
    n_edges = 0
    for cell_id in active:
        ix = cell_id % int(number_of_cell_x)
        iy = cell_id // int(number_of_cell_x)
        right = cell_id + 1
        down = cell_id + int(number_of_cell_x)
        if ix + 1 < int(number_of_cell_x) and right in active:
            n_edges += 1
        if iy + 1 < int(number_of_cell_y) and down in active:
            n_edges += 1
    return n_edges

def _per_cell_residual_stats_ms(
    *,
    row_midpoint_cell_id: np.ndarray,
    residual_time_s: np.ndarray,
    used_row_mask: np.ndarray,
    n_total_cells: int,
) -> dict[str, np.ndarray]:
    row_cell = np.asarray(row_midpoint_cell_id, dtype=np.int64)
    residual_ms = np.asarray(residual_time_s, dtype=np.float64) * 1000.0
    used = np.asarray(used_row_mask, dtype=bool)
    out = {
        'rms': np.full(int(n_total_cells), np.nan, dtype=np.float64),
        'mad': np.full(int(n_total_cells), np.nan, dtype=np.float64),
        'mean': np.full(int(n_total_cells), np.nan, dtype=np.float64),
        'p95_abs': np.full(int(n_total_cells), np.nan, dtype=np.float64),
    }
    for cell_id in range(int(n_total_cells)):
        values = residual_ms[(row_cell == cell_id) & used]
        for stat, key in (
            ('rms', 'rms'),
            ('mad', 'mad'),
            ('mean', 'mean'),
            ('p95_abs', 'p95_abs'),
        ):
            stat_value = _residual_stat(values, stat)
            if stat_value is not None:
                out[key][cell_id] = float(stat_value)
    return {
        key: np.ascontiguousarray(value, dtype=np.float64)
        for key, value in out.items()
    }
__all__ = ['RefractionCellSolverHistoryRow', 'build_refraction_cell_solver_history_rows', 'build_refraction_refractor_velocity_grid_arrays', 'build_refraction_refractor_velocity_qc_payload', 'write_refraction_cell_solver_history_csv', 'write_refraction_refractor_velocity_cells_csv', 'write_refraction_refractor_velocity_grid_npz', 'write_refraction_refractor_velocity_qc_json', '_active_neighbor_count_by_cell', '_cell_solver_history_cell_counts', '_cell_solver_history_csv_row', '_cell_velocity_artifact_result_for_layer', '_cell_velocity_artifact_result_from_layer', '_cell_velocity_candidate_row_mask', '_cell_velocity_component', '_cell_velocity_layer_bounds', '_cell_velocity_layer_result', '_cell_velocity_min_observations_per_cell', '_cell_velocity_n_total_cells', '_compute_row_midpoint_cell_id', '_estimated_cell_smoothing_rows', '_history_convergence_reason', '_history_max_abs', '_history_residual_stats_ms', '_history_robust_iteration_count', '_history_smoothing_weight', '_initial_cell_v2_m_s', '_layer_cell_status_for_active_cells', '_layer_cell_values_for_active_cells', '_layer_trace_bool_array', '_layer_trace_float_array', '_layer_velocity_modes_for_request', '_per_cell_residual_stats_ms', '_refractor_cell_center_alias_arrays', '_refractor_cell_status_reasons', '_refractor_velocity_cell_rows', '_request_cell_velocity_layer', '_request_has_cell_velocity_layer', '_request_summary', '_result_has_cell_velocity_arrays', '_row_endpoint_xy', '_row_midpoint_cell_id_for_cell_velocity_layer', '_unique_observation_endpoint_count_by_cell']
