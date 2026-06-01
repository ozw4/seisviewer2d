"""Sparse design matrix builder for GLI refraction statics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import sparse

from app.statics.refraction.contracts.model import RefractionStaticModelRequest
from app.services.common.artifact_io import write_csv_atomic, write_json_atomic
from app.services.common.array_validation import (
    coerce_1d_bool_array,
    coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64,
    coerce_positive_finite_float,
)
from app.statics.refraction.artifacts.design_matrix import (
    REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME,
    REFRACTION_DESIGN_MATRIX_QC_JSON_NAME,
    all_refraction_design_matrix_layer_artifact_names,
    refraction_design_matrix_layer_artifact_names,
    refraction_design_matrix_layer_node_diagnostics_csv_name,
    refraction_design_matrix_layer_qc_json_name,
)
from app.statics.refraction.domain.cell_coordinates import (
    effective_refraction_cell_grid_config,
    project_refraction_cell_coordinates,
)
from app.statics.refraction.domain.cell_grid import (
    assign_observation_midpoint_cells,
    build_refraction_cell_grid,
)
from app.statics.refraction.domain.cell_velocity_status import (
    LOW_FOLD_CELL_REJECTION_REASON,
    LOW_FOLD_CELL_VELOCITY_STATUS,
)
from app.statics.refraction.domain.first_layer import resolve_weathering_velocity_m_s
from app.statics.refraction.domain.types import (
    RefractionDesignMatrixNodeDiagnostics,
    RefractionStaticDesignMatrix,
    RefractionStaticInputModel,
    ResolvedRefractionFirstLayer,
)

BedrockVelocityMode = Literal['solve_global', 'fixed_global', 'solve_cell']
CellAssignmentMode = Literal['midpoint']
OUTSIDE_REFRACTOR_CELL_GRID_REASON = 'outside_refractor_cell_grid'
_NODE_DIAGNOSTIC_COLUMNS = (
    'node_id',
    'matrix_column',
    'endpoint_kind',
    'endpoint_key',
    'source_endpoint_key',
    'receiver_endpoint_key',
    'active',
    'n_rows_pre_filter',
    'n_rows_post_filter',
    'n_nonzero_entries',
    'status',
    'reason',
    'first_trace_indices_pre_filter',
)


class RefractionStaticDesignMatrixError(ValueError):
    """Raised when refraction static design matrix inputs are inconsistent."""


_coerce_1d_real_numeric_float64 = partial(
    coerce_1d_real_numeric_float64,
    error_type=RefractionStaticDesignMatrixError,
)
_coerce_1d_integer_int64 = partial(
    coerce_1d_integer_int64,
    error_type=RefractionStaticDesignMatrixError,
)
_coerce_1d_bool_array = partial(
    coerce_1d_bool_array,
    error_type=RefractionStaticDesignMatrixError,
)
_coerce_positive_finite_float = partial(
    coerce_positive_finite_float,
    error_type=RefractionStaticDesignMatrixError,
)


@dataclass(frozen=True)
class _DesignMatrixDiagnosticsContext:
    node_id: np.ndarray
    node_kind: np.ndarray | None
    active_node_id: np.ndarray
    node_id_to_col: dict[int, int]
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    source_endpoint_key_sorted: np.ndarray | None
    receiver_endpoint_key_sorted: np.ndarray | None
    sorted_trace_index: np.ndarray
    selected_mask: np.ndarray
    rejection_reason_sorted: np.ndarray | None


def build_refraction_static_design_matrix(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    include_diagnostics: bool = False,
) -> RefractionStaticDesignMatrix:
    """Build the physical GLI sparse system from a refraction input model."""
    method = getattr(model, 'method', None)
    if method != 'gli_variable_thickness':
        raise RefractionStaticDesignMatrixError(
            'model.method must be gli_variable_thickness'
        )
    mode = _validate_bedrock_velocity_mode(
        getattr(model, 'bedrock_velocity_mode', None)
    )
    weathering_velocity = _coerce_positive_finite_float(
        resolve_weathering_velocity_m_s(
            model=model,
            resolved_first_layer=resolved_first_layer,
            name='model.weathering_velocity_m_s',
        ),
        name='model.weathering_velocity_m_s',
    )
    fixed_velocity = getattr(model, 'bedrock_velocity_m_s', None)
    if mode == 'fixed_global':
        fixed_velocity = _coerce_positive_finite_float(
            fixed_velocity,
            name='model.bedrock_velocity_m_s',
        )
        if fixed_velocity <= weathering_velocity:
            raise RefractionStaticDesignMatrixError(
                'model.bedrock_velocity_m_s must be greater than '
                'model.weathering_velocity_m_s'
            )
    elif fixed_velocity is not None:
        raise RefractionStaticDesignMatrixError(
            'model.bedrock_velocity_m_s is only allowed when '
            'model.bedrock_velocity_mode is fixed_global'
        )
    if mode == 'solve_cell':
        return build_refraction_static_cell_design_matrix(
            input_model=input_model,
            model=model,
            resolved_first_layer=resolved_first_layer,
            include_diagnostics=include_diagnostics,
        )

    return build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=input_model.pick_time_s_sorted,
        valid_observation_mask_sorted=input_model.valid_observation_mask_sorted,
        source_node_id_sorted=input_model.source_node_id_sorted,
        receiver_node_id_sorted=input_model.receiver_node_id_sorted,
        distance_m_sorted=input_model.distance_m_sorted,
        node_id=input_model.endpoint_table.node_id,
        node_kind=input_model.endpoint_table.kind,
        sorted_trace_index=input_model.sorted_trace_index,
        source_endpoint_key_sorted=input_model.source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=input_model.receiver_endpoint_key_sorted,
        bedrock_velocity_mode=mode,
        fixed_bedrock_velocity_m_s=fixed_velocity,
        n_traces=input_model.n_traces,
        rejection_reason_sorted=input_model.rejection_reason_sorted,
        include_diagnostics=include_diagnostics,
    )


def build_refraction_static_cell_design_matrix(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    include_diagnostics: bool = False,
) -> RefractionStaticDesignMatrix:
    """Build the sparse GLI system for midpoint-cell refractor slowness."""
    method = getattr(model, 'method', None)
    if method != 'gli_variable_thickness':
        raise RefractionStaticDesignMatrixError(
            'model.method must be gli_variable_thickness'
        )
    mode = _validate_bedrock_velocity_mode(
        getattr(model, 'bedrock_velocity_mode', None)
    )
    if mode != 'solve_cell':
        raise RefractionStaticDesignMatrixError(
            'model.bedrock_velocity_mode must be solve_cell'
        )
    _coerce_positive_finite_float(
        resolve_weathering_velocity_m_s(
            model=model,
            resolved_first_layer=resolved_first_layer,
            name='model.weathering_velocity_m_s',
        ),
        name='model.weathering_velocity_m_s',
    )
    if getattr(model, 'bedrock_velocity_m_s', None) is not None:
        raise RefractionStaticDesignMatrixError(
            'model.bedrock_velocity_m_s is only allowed when '
            'model.bedrock_velocity_mode is fixed_global'
        )
    refractor_cell = getattr(model, 'refractor_cell', None)
    if refractor_cell is None:
        raise RefractionStaticDesignMatrixError(
            'model.refractor_cell is required when '
            'model.bedrock_velocity_mode is solve_cell'
        )
    grid_config = effective_refraction_cell_grid_config(refractor_cell)
    grid = build_refraction_cell_grid(grid_config)
    projected = project_refraction_cell_coordinates(
        source_x_m=input_model.source_x_m_sorted,
        source_y_m=input_model.source_y_m_sorted,
        receiver_x_m=input_model.receiver_x_m_sorted,
        receiver_y_m=input_model.receiver_y_m_sorted,
        mode=refractor_cell.coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    assignment = assign_observation_midpoint_cells(
        grid,
        source_x_m=projected.source_x_m,
        source_y_m=projected.source_y_m,
        receiver_x_m=projected.receiver_x_m,
        receiver_y_m=projected.receiver_y_m,
    )
    return build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=input_model.pick_time_s_sorted,
        valid_observation_mask_sorted=input_model.valid_observation_mask_sorted,
        source_node_id_sorted=input_model.source_node_id_sorted,
        receiver_node_id_sorted=input_model.receiver_node_id_sorted,
        distance_m_sorted=input_model.distance_m_sorted,
        node_id=input_model.endpoint_table.node_id,
        node_kind=input_model.endpoint_table.kind,
        sorted_trace_index=input_model.sorted_trace_index,
        source_endpoint_key_sorted=input_model.source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=input_model.receiver_endpoint_key_sorted,
        bedrock_velocity_mode='solve_cell',
        n_traces=input_model.n_traces,
        midpoint_cell_id_sorted=assignment.cell_id,
        n_total_cells=int(grid.cell_id.shape[0]),
        number_of_cell_x=grid.number_of_cell_x,
        number_of_cell_y=grid.number_of_cell_y,
        cell_assignment_mode=refractor_cell.assignment_mode,
        min_observations_per_cell=refractor_cell.min_observations_per_cell,
        rejection_reason_sorted=input_model.rejection_reason_sorted,
        coordinate_qc=projected.qc,
        include_diagnostics=include_diagnostics,
    )


def build_refraction_static_design_matrix_from_arrays(
    *,
    pick_time_s_sorted: np.ndarray,
    valid_observation_mask_sorted: np.ndarray,
    source_node_id_sorted: np.ndarray,
    receiver_node_id_sorted: np.ndarray,
    distance_m_sorted: np.ndarray,
    node_id: np.ndarray,
    bedrock_velocity_mode: BedrockVelocityMode,
    node_kind: np.ndarray | None = None,
    sorted_trace_index: np.ndarray | None = None,
    source_endpoint_key_sorted: np.ndarray | None = None,
    receiver_endpoint_key_sorted: np.ndarray | None = None,
    fixed_bedrock_velocity_m_s: float | None = None,
    n_traces: int | None = None,
    midpoint_cell_id_sorted: np.ndarray | None = None,
    n_total_cells: int | None = None,
    number_of_cell_x: int | None = None,
    number_of_cell_y: int | None = None,
    cell_assignment_mode: CellAssignmentMode | None = None,
    min_observations_per_cell: int | None = None,
    rejection_reason_sorted: np.ndarray | None = None,
    coordinate_qc: dict[str, Any] | None = None,
    include_diagnostics: bool = False,
) -> RefractionStaticDesignMatrix:
    """Build a refraction static design matrix from sorted observation arrays."""
    mode = _validate_bedrock_velocity_mode(bedrock_velocity_mode)
    fixed_velocity: float | None = None
    fixed_slowness: float | None = None
    if mode == 'fixed_global':
        fixed_velocity = _coerce_positive_finite_float(
            fixed_bedrock_velocity_m_s,
            name='fixed_bedrock_velocity_m_s',
        )
        fixed_slowness = float(1.0 / fixed_velocity)
    elif fixed_bedrock_velocity_m_s is not None:
        raise RefractionStaticDesignMatrixError(
            'fixed_bedrock_velocity_m_s is only allowed for fixed_global mode'
        )

    pick_time = _coerce_1d_real_numeric_float64(
        pick_time_s_sorted,
        name='pick_time_s_sorted',
    )
    trace_count = _validate_n_traces(n_traces, default=int(pick_time.shape[0]))
    expected_shape = (trace_count,)
    if pick_time.shape != expected_shape:
        raise RefractionStaticDesignMatrixError(
            'pick_time_s_sorted shape mismatch: '
            f'expected {expected_shape}, got {pick_time.shape}'
        )
    valid_mask = _coerce_1d_bool_array(
        valid_observation_mask_sorted,
        name='valid_observation_mask_sorted',
        expected_shape=expected_shape,
    )
    source_node_id = _coerce_1d_integer_int64(
        source_node_id_sorted,
        name='source_node_id_sorted',
        expected_shape=expected_shape,
    )
    receiver_node_id = _coerce_1d_integer_int64(
        receiver_node_id_sorted,
        name='receiver_node_id_sorted',
        expected_shape=expected_shape,
    )
    distance_m = _coerce_1d_real_numeric_float64(
        distance_m_sorted,
        name='distance_m_sorted',
        expected_shape=expected_shape,
    )
    total_node_id = _coerce_unique_node_id(node_id)
    total_node_kind = _coerce_optional_node_kind(
        node_kind,
        expected_shape=total_node_id.shape,
    )
    source_endpoint_key = _coerce_optional_endpoint_key(
        source_endpoint_key_sorted,
        name='source_endpoint_key_sorted',
        expected_shape=expected_shape,
    )
    receiver_endpoint_key = _coerce_optional_endpoint_key(
        receiver_endpoint_key_sorted,
        name='receiver_endpoint_key_sorted',
        expected_shape=expected_shape,
    )
    trace_index_sorted = (
        np.arange(trace_count, dtype=np.int64)
        if sorted_trace_index is None
        else _coerce_1d_integer_int64(
            sorted_trace_index,
            name='sorted_trace_index',
            expected_shape=expected_shape,
        )
    )

    selected_mask = valid_mask
    midpoint_cell_id: np.ndarray | None = None
    design_rejection_reason = _coerce_optional_rejection_reason(
        rejection_reason_sorted,
        expected_shape=expected_shape,
    )
    bedrock_slowness_cell_col_start: int | None = None
    active_cell_id: np.ndarray | None = None
    inactive_cell_id: np.ndarray | None = None
    cell_id_to_col: dict[int, int] | None = None
    row_midpoint_cell_id: np.ndarray | None = None
    row_midpoint_cell_col: np.ndarray | None = None
    total_cell_count: int | None = None
    active_cell_count: int | None = None
    inactive_cell_count: int | None = None
    n_cell_x: int | None = None
    n_cell_y: int | None = None
    assignment_mode: CellAssignmentMode | None = None
    n_observations_outside_grid = 0
    n_observations_rejected_by_low_fold_cell = 0
    min_cell_observations: int | None = None
    low_fold_cell_id: np.ndarray | None = None
    cell_observation_counts: np.ndarray | None = None
    if mode == 'solve_cell':
        midpoint_cell_id = _coerce_1d_integer_int64(
            midpoint_cell_id_sorted,
            name='midpoint_cell_id_sorted',
            expected_shape=expected_shape,
        )
        total_cell_count = _validate_n_total_cells(n_total_cells)
        n_cell_x, n_cell_y = _validate_cell_grid_shape(
            n_total_cells=total_cell_count,
            number_of_cell_x=number_of_cell_x,
            number_of_cell_y=number_of_cell_y,
        )
        assignment_mode = _validate_cell_assignment_mode(cell_assignment_mode)
        min_cell_observations = _validate_min_observations_per_cell(
            min_observations_per_cell
        )
        _validate_midpoint_cell_ids(
            midpoint_cell_id,
            n_total_cells=total_cell_count,
        )
        inside_grid_mask = midpoint_cell_id >= 0
        outside_grid_mask = valid_mask & ~inside_grid_mask
        n_observations_outside_grid = int(np.count_nonzero(outside_grid_mask))
        in_grid_selected_mask = valid_mask & inside_grid_mask
        cell_observation_counts = np.bincount(
            midpoint_cell_id[in_grid_selected_mask],
            minlength=total_cell_count,
        )
        low_fold_cell_mask = (
            (cell_observation_counts > 0)
            & (cell_observation_counts < min_cell_observations)
        )
        low_fold_cell_id = np.ascontiguousarray(
            np.flatnonzero(low_fold_cell_mask),
            dtype=np.int64,
        )
        low_fold_row_mask = np.zeros(expected_shape, dtype=bool)
        if low_fold_cell_id.size:
            low_fold_row_mask[in_grid_selected_mask] = low_fold_cell_mask[
                midpoint_cell_id[in_grid_selected_mask]
            ]
        n_observations_rejected_by_low_fold_cell = int(
            np.count_nonzero(low_fold_row_mask)
        )
        selected_mask = in_grid_selected_mask & ~low_fold_row_mask
        if design_rejection_reason is None:
            design_rejection_reason = np.where(valid_mask, 'ok', '').astype('<U32')
        design_rejection_reason = np.ascontiguousarray(
            design_rejection_reason,
            dtype='<U32',
        )
        design_rejection_reason[outside_grid_mask] = (
            OUTSIDE_REFRACTOR_CELL_GRID_REASON
        )
        design_rejection_reason[low_fold_row_mask] = (
            LOW_FOLD_CELL_REJECTION_REASON
        )
    elif (
        midpoint_cell_id_sorted is not None
        or n_total_cells is not None
        or number_of_cell_x is not None
        or number_of_cell_y is not None
        or cell_assignment_mode is not None
        or min_observations_per_cell is not None
        or coordinate_qc is not None
    ):
        raise RefractionStaticDesignMatrixError(
            'cell design matrix inputs are only allowed for solve_cell mode'
        )

    row_trace_index = np.ascontiguousarray(
        np.flatnonzero(selected_mask),
        dtype=np.int64,
    )
    n_observations = int(row_trace_index.shape[0])
    if n_observations <= 0:
        if mode == 'solve_cell' and n_observations_rejected_by_low_fold_cell > 0:
            raise RefractionStaticDesignMatrixError(
                'at least one valid refraction observation meeting '
                'min_observations_per_cell is required'
            )
        if mode == 'solve_cell' and n_observations_outside_grid > 0:
            raise RefractionStaticDesignMatrixError(
                'at least one valid refraction observation inside the '
                'refractor cell grid is required'
            )
        raise RefractionStaticDesignMatrixError(
            'at least one valid refraction observation is required'
        )

    observed_pick_time = np.ascontiguousarray(
        pick_time[row_trace_index],
        dtype=np.float64,
    )
    row_source_node_id = np.ascontiguousarray(
        source_node_id[row_trace_index],
        dtype=np.int64,
    )
    row_receiver_node_id = np.ascontiguousarray(
        receiver_node_id[row_trace_index],
        dtype=np.int64,
    )
    row_distance_m = np.ascontiguousarray(
        distance_m[row_trace_index],
        dtype=np.float64,
    )

    _validate_selected_values(
        observed_pick_time=observed_pick_time,
        row_distance_m=row_distance_m,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        node_id=total_node_id,
    )
    active_node_id, inactive_node_id = _split_active_nodes(
        node_id=total_node_id,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
    )
    node_id_to_col = {int(value): idx for idx, value in enumerate(active_node_id.tolist())}
    if len(node_id_to_col) != int(active_node_id.shape[0]):
        raise RefractionStaticDesignMatrixError('active node IDs must be unique')
    source_node_col = _map_node_ids_to_cols(
        row_source_node_id,
        node_id_to_col=node_id_to_col,
        name='row_source_node_id',
    )
    receiver_node_col = _map_node_ids_to_cols(
        row_receiver_node_id,
        node_id_to_col=node_id_to_col,
        name='row_receiver_node_id',
    )

    bedrock_slowness_col = None
    if mode == 'solve_global':
        bedrock_slowness_col = int(active_node_id.shape[0])
    if mode == 'solve_cell':
        if midpoint_cell_id is None or total_cell_count is None:
            raise RefractionStaticDesignMatrixError(
                'solve_cell mode requires midpoint cell IDs'
            )
        row_midpoint_cell_id = np.ascontiguousarray(
            midpoint_cell_id[row_trace_index],
            dtype=np.int64,
        )
        active_cell_id, inactive_cell_id = _split_active_cells(
            n_total_cells=total_cell_count,
            row_midpoint_cell_id=row_midpoint_cell_id,
        )
        bedrock_slowness_cell_col_start = int(active_node_id.shape[0])
        cell_id_to_col = {
            int(value): bedrock_slowness_cell_col_start + idx
            for idx, value in enumerate(active_cell_id.tolist())
        }
        if len(cell_id_to_col) != int(active_cell_id.shape[0]):
            raise RefractionStaticDesignMatrixError('active cell IDs must be unique')
        row_midpoint_cell_col = _map_cell_ids_to_cols(
            row_midpoint_cell_id,
            cell_id_to_col=cell_id_to_col,
        )
        active_cell_count = int(active_cell_id.shape[0])
        inactive_cell_count = int(inactive_cell_id.shape[0])
        n_parameters = int(active_node_id.shape[0]) + active_cell_count
    else:
        n_parameters = int(active_node_id.shape[0]) + (
            1 if bedrock_slowness_col is not None else 0
        )

    matrix = _build_sparse_matrix(
        source_node_col=source_node_col,
        receiver_node_col=receiver_node_col,
        row_distance_m=row_distance_m,
        bedrock_slowness_col=bedrock_slowness_col,
        bedrock_slowness_col_by_row=row_midpoint_cell_col,
        n_observations=n_observations,
        n_parameters=n_parameters,
    )
    if mode == 'fixed_global':
        if fixed_slowness is None:
            raise RefractionStaticDesignMatrixError(
                'fixed_global mode requires fixed bedrock slowness'
            )
        rhs_s = np.ascontiguousarray(
            observed_pick_time - row_distance_m * fixed_slowness,
            dtype=np.float64,
        )
    else:
        rhs_s = np.ascontiguousarray(observed_pick_time, dtype=np.float64)

    _validate_matrix_package(
        matrix=matrix,
        rhs_s=rhs_s,
        n_observations=n_observations,
        n_parameters=n_parameters,
    )
    diagnostics_context = _DesignMatrixDiagnosticsContext(
        node_id=total_node_id,
        node_kind=total_node_kind,
        active_node_id=active_node_id,
        node_id_to_col=node_id_to_col,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        source_endpoint_key_sorted=source_endpoint_key,
        receiver_endpoint_key_sorted=receiver_endpoint_key,
        sorted_trace_index=trace_index_sorted,
        selected_mask=selected_mask,
        rejection_reason_sorted=design_rejection_reason,
    )
    node_diagnostics = (
        _build_node_diagnostics(
            context=diagnostics_context,
            matrix=matrix,
        )
        if include_diagnostics
        else ()
    )
    design_matrix_qc = (
        _build_design_matrix_diagnostics_qc(
            matrix=matrix,
            n_active_nodes=int(active_node_id.shape[0]),
            node_diagnostics=node_diagnostics,
        )
        if include_diagnostics
        else _build_lightweight_design_matrix_diagnostics_qc(
            matrix=matrix,
            n_active_nodes=int(active_node_id.shape[0]),
        )
    )
    qc = _build_qc(
        method='gli_variable_thickness',
        mode=mode,
        n_traces=trace_count,
        n_observations=n_observations,
        n_total_nodes=int(total_node_id.shape[0]),
        n_active_nodes=int(active_node_id.shape[0]),
        n_parameters=n_parameters,
        matrix=matrix,
        row_distance_m=row_distance_m,
        observed_pick_time=observed_pick_time,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        active_node_id=active_node_id,
        node_id_to_col=node_id_to_col,
        fixed_bedrock_velocity_m_s=fixed_velocity,
        fixed_bedrock_slowness_s_per_m=fixed_slowness,
        slowness_column_present=(
            bedrock_slowness_col is not None or row_midpoint_cell_col is not None
        ),
    )
    qc['design_matrix_diagnostics'] = design_matrix_qc
    if mode == 'solve_cell':
        if (
            assignment_mode is None
            or total_cell_count is None
            or active_cell_id is None
            or inactive_cell_id is None
            or cell_observation_counts is None
            or n_cell_x is None
            or n_cell_y is None
            or min_cell_observations is None
            or low_fold_cell_id is None
        ):
            raise RefractionStaticDesignMatrixError(
                'solve_cell mode requires cell QC inputs'
            )
        qc.update(
            _build_cell_qc(
                cell_assignment_mode=assignment_mode,
                min_observations_per_cell=min_cell_observations,
                n_total_cells=total_cell_count,
                active_cell_id=active_cell_id,
                inactive_cell_id=inactive_cell_id,
                low_fold_cell_id=low_fold_cell_id,
                cell_observation_counts=cell_observation_counts,
                n_observations_outside_grid=n_observations_outside_grid,
                n_observations_rejected_by_low_fold_cell=(
                    n_observations_rejected_by_low_fold_cell
                ),
                n_observations_used=n_observations,
                number_of_cell_x=n_cell_x,
                number_of_cell_y=n_cell_y,
                coordinate_qc=coordinate_qc,
            )
        )

    return RefractionStaticDesignMatrix(
        matrix=matrix,
        rhs_s=rhs_s,
        observed_pick_time_s=observed_pick_time,
        row_trace_index_sorted=row_trace_index,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        row_distance_m=row_distance_m,
        active_node_id=active_node_id,
        inactive_node_id=inactive_node_id,
        node_id_to_col=node_id_to_col,
        source_node_col=source_node_col,
        receiver_node_col=receiver_node_col,
        bedrock_slowness_col=bedrock_slowness_col,
        bedrock_velocity_mode=mode,
        fixed_bedrock_velocity_m_s=fixed_velocity,
        fixed_bedrock_slowness_s_per_m=fixed_slowness,
        n_total_nodes=int(total_node_id.shape[0]),
        n_active_nodes=int(active_node_id.shape[0]),
        n_observations=n_observations,
        n_parameters=n_parameters,
        qc=qc,
        node_diagnostics=node_diagnostics,
        design_matrix_qc=design_matrix_qc,
        diagnostics_context=diagnostics_context,
        bedrock_slowness_cell_col_start=bedrock_slowness_cell_col_start,
        active_cell_id=active_cell_id,
        inactive_cell_id=inactive_cell_id,
        cell_id_to_col=cell_id_to_col,
        row_midpoint_cell_id=row_midpoint_cell_id,
        row_midpoint_cell_col=row_midpoint_cell_col,
        cell_assignment_mode=assignment_mode,
        n_total_cells=total_cell_count,
        n_active_cells=active_cell_count,
        n_inactive_cells=inactive_cell_count,
        number_of_cell_x=n_cell_x,
        number_of_cell_y=n_cell_y,
        rejection_reason_sorted=design_rejection_reason,
    )


def _build_sparse_matrix(
    *,
    source_node_col: np.ndarray,
    receiver_node_col: np.ndarray,
    row_distance_m: np.ndarray,
    bedrock_slowness_col: int | None,
    bedrock_slowness_col_by_row: np.ndarray | None,
    n_observations: int,
    n_parameters: int,
) -> sparse.csr_matrix:
    if bedrock_slowness_col is not None and bedrock_slowness_col_by_row is not None:
        raise RefractionStaticDesignMatrixError(
            'global and cell bedrock slowness columns are mutually exclusive'
        )
    if bedrock_slowness_col_by_row is not None:
        if bedrock_slowness_col_by_row.shape != (n_observations,):
            raise RefractionStaticDesignMatrixError(
                'bedrock_slowness_col_by_row shape mismatch'
            )
    entries_per_row = (
        3
        if bedrock_slowness_col is not None or bedrock_slowness_col_by_row is not None
        else 2
    )
    row_indices = np.repeat(np.arange(n_observations, dtype=np.int64), entries_per_row)
    col_indices = np.empty(n_observations * entries_per_row, dtype=np.int64)
    data = np.empty(n_observations * entries_per_row, dtype=np.float64)

    col_indices[0::entries_per_row] = source_node_col
    col_indices[1::entries_per_row] = receiver_node_col
    data[0::entries_per_row] = 1.0
    data[1::entries_per_row] = 1.0
    if bedrock_slowness_col is not None:
        col_indices[2::entries_per_row] = bedrock_slowness_col
        data[2::entries_per_row] = row_distance_m
    elif bedrock_slowness_col_by_row is not None:
        col_indices[2::entries_per_row] = bedrock_slowness_col_by_row
        data[2::entries_per_row] = row_distance_m

    matrix = sparse.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_observations, n_parameters),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _build_node_diagnostics(
    *,
    context: _DesignMatrixDiagnosticsContext,
    matrix: sparse.csr_matrix,
) -> tuple[RefractionDesignMatrixNodeDiagnostics, ...]:
    node_id = context.node_id
    node_kind = context.node_kind
    active_set = {int(value) for value in context.active_node_id.tolist()}
    col_nnz = np.asarray(matrix.getnnz(axis=0)).ravel()
    node_position_by_id = {
        int(raw_node_id): idx for idx, raw_node_id in enumerate(node_id.tolist())
    }
    source_node_pos = _map_node_ids_to_diagnostic_positions(
        context.source_node_id_sorted,
        node_position_by_id=node_position_by_id,
    )
    receiver_node_pos = _map_node_ids_to_diagnostic_positions(
        context.receiver_node_id_sorted,
        node_position_by_id=node_position_by_id,
    )
    source_first_row = _first_row_index_by_node_position(
        source_node_pos,
        n_nodes=int(node_id.shape[0]),
    )
    receiver_first_row = _first_row_index_by_node_position(
        receiver_node_pos,
        n_nodes=int(node_id.shape[0]),
    )
    row_indices_by_node, pre_counts, post_counts, filtered_reasons = (
        _aggregate_node_diagnostic_rows(
            source_node_pos=source_node_pos,
            receiver_node_pos=receiver_node_pos,
            sorted_trace_index=context.sorted_trace_index,
            selected_mask=context.selected_mask,
            rejection_reason_sorted=context.rejection_reason_sorted,
            n_nodes=int(node_id.shape[0]),
        )
    )
    diagnostics: list[RefractionDesignMatrixNodeDiagnostics] = []
    for node_position, raw_node_id in enumerate(node_id.tolist()):
        node = int(raw_node_id)
        active = node in active_set
        matrix_column = int(context.node_id_to_col[node]) if active else -1
        n_rows_pre_filter = int(pre_counts[node_position])
        n_rows_post_filter = int(post_counts[node_position])
        endpoint_kind = _diagnostic_endpoint_kind(
            node_kind=None if node_kind is None else str(node_kind[node_position]),
            source_seen=bool(source_first_row[node_position] >= 0),
            receiver_seen=bool(receiver_first_row[node_position] >= 0),
        )
        source_key = _first_endpoint_key_by_row_index(
            context.source_endpoint_key_sorted,
            int(source_first_row[node_position]),
            prefix='source',
            node_id=node,
        )
        receiver_key = _first_endpoint_key_by_row_index(
            context.receiver_endpoint_key_sorted,
            int(receiver_first_row[node_position]),
            prefix='receiver',
            node_id=node,
        )
        endpoint_key = _diagnostic_endpoint_key_value(
            endpoint_kind=endpoint_kind,
            source_endpoint_key=source_key,
            receiver_endpoint_key=receiver_key,
            node_id=node,
        )
        n_nonzero = int(col_nnz[matrix_column]) if active and matrix_column >= 0 else 0
        status = _node_diagnostic_status(
            active=active,
            n_rows_pre_filter=n_rows_pre_filter,
            n_rows_post_filter=n_rows_post_filter,
            n_nonzero_entries=n_nonzero,
        )
        reason = _node_diagnostic_reason(
            status=status,
            n_rows_pre_filter=n_rows_pre_filter,
            n_rows_post_filter=n_rows_post_filter,
            filtered_reasons=filtered_reasons[node_position],
        )
        first_indices = tuple(row_indices_by_node[node_position])
        diagnostics.append(
            RefractionDesignMatrixNodeDiagnostics(
                node_id=node,
                matrix_column=matrix_column,
                endpoint_kind=endpoint_kind,
                endpoint_key=endpoint_key,
                source_endpoint_key=source_key,
                receiver_endpoint_key=receiver_key,
                n_rows_pre_filter=n_rows_pre_filter,
                n_rows_post_filter=n_rows_post_filter,
                n_nonzero_entries=n_nonzero,
                active=active,
                status=status,
                reason=reason,
                first_trace_indices_pre_filter=first_indices,
            )
        )
    return tuple(diagnostics)


def _map_node_ids_to_diagnostic_positions(
    values: np.ndarray,
    *,
    node_position_by_id: dict[int, int],
) -> np.ndarray:
    positions = np.full(values.shape, -1, dtype=np.int64)
    for index, raw_value in enumerate(values.tolist()):
        positions[index] = node_position_by_id.get(int(raw_value), -1)
    return positions


def _first_row_index_by_node_position(
    node_position_by_row: np.ndarray,
    *,
    n_nodes: int,
) -> np.ndarray:
    first_row = np.full(n_nodes, -1, dtype=np.int64)
    for row_index, raw_position in enumerate(node_position_by_row.tolist()):
        position = int(raw_position)
        if position >= 0 and first_row[position] < 0:
            first_row[position] = int(row_index)
    return first_row


def _aggregate_node_diagnostic_rows(
    *,
    source_node_pos: np.ndarray,
    receiver_node_pos: np.ndarray,
    sorted_trace_index: np.ndarray,
    selected_mask: np.ndarray,
    rejection_reason_sorted: np.ndarray | None,
    n_nodes: int,
) -> tuple[list[list[int]], np.ndarray, np.ndarray, list[set[str] | None]]:
    trace_count = int(selected_mask.shape[0])
    row_index = np.arange(trace_count, dtype=np.int64)
    valid_source = source_node_pos >= 0
    valid_receiver = receiver_node_pos >= 0
    pair_node_pos = np.concatenate(
        (source_node_pos[valid_source], receiver_node_pos[valid_receiver])
    )
    pair_row_index = np.concatenate((row_index[valid_source], row_index[valid_receiver]))
    first_indices_by_node: list[list[int]] = [[] for _ in range(n_nodes)]
    filtered_reasons: list[set[str] | None] = [
        set() if rejection_reason_sorted is not None else None for _ in range(n_nodes)
    ]
    if pair_node_pos.size == 0:
        return (
            first_indices_by_node,
            np.zeros(n_nodes, dtype=np.int64),
            np.zeros(n_nodes, dtype=np.int64),
            filtered_reasons,
        )

    unique_pair_key = np.unique(pair_node_pos * trace_count + pair_row_index)
    unique_node_pos = unique_pair_key // trace_count
    unique_row_index = unique_pair_key % trace_count
    pre_counts = np.bincount(unique_node_pos, minlength=n_nodes).astype(np.int64)
    post_counts = np.bincount(
        unique_node_pos[selected_mask[unique_row_index]],
        minlength=n_nodes,
    ).astype(np.int64)

    trace_order = np.argsort(unique_row_index, kind='stable')
    for raw_node_pos, raw_row_index in zip(
        unique_node_pos[trace_order].tolist(),
        unique_row_index[trace_order].tolist(),
        strict=True,
    ):
        values = first_indices_by_node[int(raw_node_pos)]
        if len(values) < 5:
            values.append(int(sorted_trace_index[int(raw_row_index)]))

    if rejection_reason_sorted is not None:
        filtered_mask = ~selected_mask[unique_row_index]
        for raw_node_pos, raw_reason in zip(
            unique_node_pos[filtered_mask].tolist(),
            rejection_reason_sorted[unique_row_index[filtered_mask]].tolist(),
            strict=True,
        ):
            reasons = filtered_reasons[int(raw_node_pos)]
            if reasons is not None:
                reasons.add(str(raw_reason))

    return first_indices_by_node, pre_counts, post_counts, filtered_reasons


def _build_design_matrix_diagnostics_qc(
    *,
    matrix: sparse.csr_matrix,
    n_active_nodes: int,
    node_diagnostics: tuple[RefractionDesignMatrixNodeDiagnostics, ...],
) -> dict[str, Any]:
    all_zero = [
        item
        for item in node_diagnostics
        if item.active and item.n_nonzero_entries == 0
    ]
    status_counts: dict[str, int] = {}
    for item in node_diagnostics:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1
    return {
        'n_rows': int(matrix.shape[0]),
        'n_columns': int(matrix.shape[1]),
        'n_active_nodes': int(n_active_nodes),
        'n_all_zero_active_node_columns': int(len(all_zero)),
        'first_all_zero_active_node': (
            _node_diagnostic_json(all_zero[0]) if all_zero else None
        ),
        'node_status_counts': status_counts,
    }


def _build_lightweight_design_matrix_diagnostics_qc(
    *,
    matrix: sparse.csr_matrix,
    n_active_nodes: int,
) -> dict[str, Any]:
    col_nnz = np.asarray(matrix.getnnz(axis=0)).ravel()
    n_all_zero = int(np.count_nonzero(col_nnz[:n_active_nodes] == 0))
    return {
        'n_rows': int(matrix.shape[0]),
        'n_columns': int(matrix.shape[1]),
        'n_active_nodes': int(n_active_nodes),
        'n_all_zero_active_node_columns': n_all_zero,
        'first_all_zero_active_node': None,
        'node_status_counts': {},
    }


def write_refraction_design_matrix_diagnostics_artifacts(
    job_dir: Path,
    design_matrix: RefractionStaticDesignMatrix,
) -> dict[str, Path]:
    """Write design-matrix QC JSON and node diagnostics CSV artifacts."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME
    csv_path = root / REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME
    node_diagnostics = design_matrix.node_diagnostics
    if not node_diagnostics:
        context = getattr(design_matrix, 'diagnostics_context', None)
        if not isinstance(context, _DesignMatrixDiagnosticsContext):
            raise RefractionStaticDesignMatrixError(
                'design matrix diagnostics context is required to write artifacts'
            )
        node_diagnostics = _build_node_diagnostics(
            context=context,
            matrix=design_matrix.matrix,
        )
    qc = _build_design_matrix_diagnostics_qc(
        matrix=design_matrix.matrix,
        n_active_nodes=int(design_matrix.active_node_id.shape[0]),
        node_diagnostics=node_diagnostics,
    )
    write_json_atomic(
        qc_path,
        qc,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        trailing_newline=True,
    )
    write_csv_atomic(
        csv_path,
        columns=_NODE_DIAGNOSTIC_COLUMNS,
        rows=[_node_diagnostic_csv_row(item) for item in node_diagnostics],
        extrasaction='raise',
        lineterminator='\r\n',
    )
    return {'qc_json': qc_path, 'node_diagnostics_csv': csv_path}


def _node_diagnostic_status(
    *,
    active: bool,
    n_rows_pre_filter: int,
    n_rows_post_filter: int,
    n_nonzero_entries: int,
) -> str:
    if not active:
        return 'inactive'
    if n_nonzero_entries == 0:
        return 'all_zero_active_column'
    if n_rows_post_filter == 0:
        return 'active_without_post_filter_rows'
    if n_rows_pre_filter == 0:
        return 'active_without_endpoint_rows'
    return 'ok'


def _node_diagnostic_reason(
    *,
    status: str,
    n_rows_pre_filter: int,
    n_rows_post_filter: int,
    filtered_reasons: set[str] | None,
) -> str:
    if n_rows_pre_filter == 0:
        return 'no_observations_for_node'
    if status == 'inactive' and n_rows_post_filter > 0:
        return 'inactive'
    if n_rows_post_filter > 0 and status != 'all_zero_active_column':
        return 'ok'
    if filtered_reasons is None:
        return 'unknown'
    if not filtered_reasons:
        return 'active_node_without_endpoint_rows'
    if filtered_reasons <= {'offset_gate', 'outside_layer_offset_gate'}:
        return 'all_observations_filtered_by_offset_gate'
    if filtered_reasons <= {'invalid_pick', 'outside_trace_time_range'}:
        return 'all_observations_filtered_by_invalid_pick'
    geometry_reasons = {
        'invalid_source_geometry',
        'invalid_receiver_geometry',
        'invalid_distance',
        'offset_mismatch',
        'missing_linkage',
        OUTSIDE_REFRACTOR_CELL_GRID_REASON,
        LOW_FOLD_CELL_REJECTION_REASON,
    }
    if filtered_reasons <= geometry_reasons:
        return 'all_observations_filtered_by_geometry'
    return 'unknown'


def _diagnostic_endpoint_kind(
    *,
    node_kind: str | None,
    source_seen: bool,
    receiver_seen: bool,
) -> Literal['source', 'receiver', 'linked', 'unknown']:
    if node_kind in {'source', 'receiver', 'linked'}:
        return node_kind  # type: ignore[return-value]
    if source_seen and receiver_seen:
        return 'linked'
    if source_seen:
        return 'source'
    if receiver_seen:
        return 'receiver'
    return 'unknown'


def _first_endpoint_key_by_row_index(
    values: np.ndarray | None,
    row_index: int,
    *,
    prefix: str,
    node_id: int,
) -> str | None:
    if row_index < 0:
        return None
    if values is not None:
        return str(values[row_index])
    return f'{prefix}:{node_id}'


def _diagnostic_endpoint_key_value(
    *,
    endpoint_kind: Literal['source', 'receiver', 'linked', 'unknown'],
    source_endpoint_key: str | None,
    receiver_endpoint_key: str | None,
    node_id: int,
) -> str:
    if endpoint_kind == 'source' and source_endpoint_key:
        return source_endpoint_key
    if endpoint_kind == 'receiver' and receiver_endpoint_key:
        return receiver_endpoint_key
    if endpoint_kind == 'linked':
        if source_endpoint_key and receiver_endpoint_key:
            return f'{source_endpoint_key}|{receiver_endpoint_key}'
        if source_endpoint_key:
            return source_endpoint_key
        if receiver_endpoint_key:
            return receiver_endpoint_key
    return source_endpoint_key or receiver_endpoint_key or f'node:{node_id}'


def _node_diagnostic_json(
    item: RefractionDesignMatrixNodeDiagnostics,
) -> dict[str, Any]:
    return asdict(item)


def _node_diagnostic_csv_row(
    item: RefractionDesignMatrixNodeDiagnostics,
) -> dict[str, object]:
    payload = asdict(item)
    payload['active'] = 'true' if item.active else 'false'
    payload['source_endpoint_key'] = item.source_endpoint_key or ''
    payload['receiver_endpoint_key'] = item.receiver_endpoint_key or ''
    payload['first_trace_indices_pre_filter'] = json.dumps(
        list(item.first_trace_indices_pre_filter),
        separators=(',', ':'),
    )
    return payload


def _build_qc(
    *,
    method: str,
    mode: BedrockVelocityMode,
    n_traces: int,
    n_observations: int,
    n_total_nodes: int,
    n_active_nodes: int,
    n_parameters: int,
    matrix: sparse.csr_matrix,
    row_distance_m: np.ndarray,
    observed_pick_time: np.ndarray,
    row_source_node_id: np.ndarray,
    row_receiver_node_id: np.ndarray,
    active_node_id: np.ndarray,
    node_id_to_col: dict[int, int],
    fixed_bedrock_velocity_m_s: float | None,
    fixed_bedrock_slowness_s_per_m: float | None,
    slowness_column_present: bool,
) -> dict[str, Any]:
    nnz_per_row = np.diff(matrix.indptr)
    matrix_size = int(matrix.shape[0]) * int(matrix.shape[1])
    connectivity = _connectivity_qc(
        active_node_id=active_node_id,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        node_id_to_col=node_id_to_col,
    )
    qc: dict[str, Any] = {
        'method': method,
        'bedrock_velocity_mode': mode,
        'n_traces': int(n_traces),
        'n_observations': int(n_observations),
        'n_total_nodes': int(n_total_nodes),
        'n_active_nodes': int(n_active_nodes),
        'n_inactive_nodes': int(n_total_nodes - n_active_nodes),
        'n_parameters': int(n_parameters),
        'matrix_shape': [int(matrix.shape[0]), int(matrix.shape[1])],
        'matrix_nnz': int(matrix.nnz),
        'matrix_density': float(matrix.nnz / matrix_size) if matrix_size else 0.0,
        'nnz_per_row_min': int(np.min(nnz_per_row)),
        'nnz_per_row_max': int(np.max(nnz_per_row)),
        'nnz_per_row_median': float(np.median(nnz_per_row)),
        'distance_m_min': float(np.min(row_distance_m)),
        'distance_m_max': float(np.max(row_distance_m)),
        'distance_m_median': float(np.median(row_distance_m)),
        'pick_time_s_min': float(np.min(observed_pick_time)),
        'pick_time_s_max': float(np.max(observed_pick_time)),
        'pick_time_s_median': float(np.median(observed_pick_time)),
        'source_receiver_same_node_count': int(
            np.count_nonzero(row_source_node_id == row_receiver_node_id)
        ),
        'inactive_node_count': int(n_total_nodes - n_active_nodes),
        'slowness_column_present': bool(slowness_column_present),
        **connectivity,
    }
    if mode == 'fixed_global':
        qc.update(
            {
                'fixed_bedrock_velocity_m_s': float(fixed_bedrock_velocity_m_s),
                'fixed_bedrock_slowness_s_per_m': float(
                    fixed_bedrock_slowness_s_per_m
                ),
            }
        )
    return qc


def _build_cell_qc(
    *,
    cell_assignment_mode: CellAssignmentMode,
    min_observations_per_cell: int,
    n_total_cells: int,
    active_cell_id: np.ndarray,
    inactive_cell_id: np.ndarray,
    low_fold_cell_id: np.ndarray,
    cell_observation_counts: np.ndarray,
    n_observations_outside_grid: int,
    n_observations_rejected_by_low_fold_cell: int,
    n_observations_used: int,
    number_of_cell_x: int,
    number_of_cell_y: int,
    coordinate_qc: dict[str, Any] | None,
) -> dict[str, Any]:
    active_counts = cell_observation_counts[active_cell_id]
    if active_counts.size:
        min_observations: int | None = int(np.min(active_counts))
        median_observations: float | None = float(np.median(active_counts))
        max_observations: int | None = int(np.max(active_counts))
    else:
        min_observations = None
        median_observations = None
        max_observations = None
    payload: dict[str, Any] = {
        'cell_assignment_mode': cell_assignment_mode,
        'n_total_cells': int(n_total_cells),
        'number_of_cell_x': int(number_of_cell_x),
        'number_of_cell_y': int(number_of_cell_y),
        'n_active_cells': int(active_cell_id.shape[0]),
        'n_inactive_cells': int(inactive_cell_id.shape[0]),
        'min_observations_per_cell': int(min_observations_per_cell),
        'n_low_fold_cells': int(low_fold_cell_id.shape[0]),
        'n_observations_outside_grid': int(n_observations_outside_grid),
        'n_observations_rejected_by_low_fold_cell': int(
            n_observations_rejected_by_low_fold_cell
        ),
        'n_observations_used': int(n_observations_used),
        'outside_grid_rejection_reason': OUTSIDE_REFRACTOR_CELL_GRID_REASON,
        'low_fold_cell_rejection_reason': LOW_FOLD_CELL_REJECTION_REASON,
        'low_fold_cell_id': [int(value) for value in low_fold_cell_id.tolist()],
        'cell_observation_count': [
            int(value) for value in cell_observation_counts.tolist()
        ],
        'min_observations_per_active_cell': min_observations,
        'median_observations_per_active_cell': median_observations,
        'max_observations_per_active_cell': max_observations,
    }
    if coordinate_qc is not None:
        payload.update(coordinate_qc)
    else:
        payload['coordinate_mode'] = 'grid_3d'
        payload['line_origin_x_m'] = None
        payload['line_origin_y_m'] = None
        payload['line_azimuth_deg'] = None
    return payload


def _connectivity_qc(
    *,
    active_node_id: np.ndarray,
    row_source_node_id: np.ndarray,
    row_receiver_node_id: np.ndarray,
    node_id_to_col: dict[int, int],
) -> dict[str, int]:
    n_active = int(active_node_id.shape[0])
    parent = np.arange(n_active, dtype=np.int64)
    source_seen = np.zeros(n_active, dtype=bool)
    receiver_seen = np.zeros(n_active, dtype=bool)

    def find(col: int) -> int:
        while int(parent[col]) != col:
            parent[col] = parent[int(parent[col])]
            col = int(parent[col])
        return col

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for source_id, receiver_id in zip(
        row_source_node_id.tolist(),
        row_receiver_node_id.tolist(),
        strict=True,
    ):
        source_col = node_id_to_col[int(source_id)]
        receiver_col = node_id_to_col[int(receiver_id)]
        source_seen[source_col] = True
        receiver_seen[receiver_col] = True
        if source_col != receiver_col:
            union(source_col, receiver_col)

    component_roots = {find(col) for col in range(n_active)}
    return {
        'n_source_only_nodes': int(np.count_nonzero(source_seen & ~receiver_seen)),
        'n_receiver_only_nodes': int(np.count_nonzero(receiver_seen & ~source_seen)),
        'n_source_and_receiver_nodes': int(
            np.count_nonzero(source_seen & receiver_seen)
        ),
        'n_connected_components': int(len(component_roots)),
    }


def _validate_selected_values(
    *,
    observed_pick_time: np.ndarray,
    row_distance_m: np.ndarray,
    row_source_node_id: np.ndarray,
    row_receiver_node_id: np.ndarray,
    node_id: np.ndarray,
) -> None:
    if np.any(~np.isfinite(observed_pick_time)):
        raise RefractionStaticDesignMatrixError(
            'pick_time_s_sorted must be finite for selected observations'
        )
    if np.any(observed_pick_time < 0.0):
        raise RefractionStaticDesignMatrixError(
            'pick_time_s_sorted must be non-negative for selected observations'
        )
    if np.any(~np.isfinite(row_distance_m)):
        raise RefractionStaticDesignMatrixError(
            'distance_m_sorted must be finite for selected observations'
        )
    if np.any(row_distance_m <= 0.0):
        raise RefractionStaticDesignMatrixError(
            'distance_m_sorted must be greater than 0 for selected observations'
        )

    source_missing = ~np.isin(row_source_node_id, node_id)
    if np.any(source_missing):
        missing = int(row_source_node_id[np.flatnonzero(source_missing)[0]])
        raise RefractionStaticDesignMatrixError(
            f'source_node_id_sorted contains unknown node ID {missing}'
        )
    receiver_missing = ~np.isin(row_receiver_node_id, node_id)
    if np.any(receiver_missing):
        missing = int(row_receiver_node_id[np.flatnonzero(receiver_missing)[0]])
        raise RefractionStaticDesignMatrixError(
            f'receiver_node_id_sorted contains unknown node ID {missing}'
        )


def _split_active_nodes(
    *,
    node_id: np.ndarray,
    row_source_node_id: np.ndarray,
    row_receiver_node_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    used_node_id = np.unique(np.concatenate((row_source_node_id, row_receiver_node_id)))
    active_mask = np.isin(node_id, used_node_id)
    active_node_id = np.ascontiguousarray(node_id[active_mask], dtype=np.int64)
    inactive_node_id = np.ascontiguousarray(node_id[~active_mask], dtype=np.int64)
    if active_node_id.shape[0] != used_node_id.shape[0]:
        raise RefractionStaticDesignMatrixError('active node IDs must be unique')
    return active_node_id, inactive_node_id


def _split_active_cells(
    *,
    n_total_cells: int,
    row_midpoint_cell_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    active_mask = np.zeros(n_total_cells, dtype=bool)
    active_mask[row_midpoint_cell_id] = True
    active_cell_id = np.ascontiguousarray(np.flatnonzero(active_mask), dtype=np.int64)
    inactive_cell_id = np.ascontiguousarray(
        np.flatnonzero(~active_mask),
        dtype=np.int64,
    )
    if active_cell_id.shape[0] == 0:
        raise RefractionStaticDesignMatrixError(
            'at least one active refractor cell is required'
        )
    return active_cell_id, inactive_cell_id


def _map_node_ids_to_cols(
    node_ids: np.ndarray,
    *,
    node_id_to_col: dict[int, int],
    name: str,
) -> np.ndarray:
    try:
        cols = [node_id_to_col[int(node_id)] for node_id in node_ids.tolist()]
    except KeyError as exc:
        raise RefractionStaticDesignMatrixError(
            f'{name} contains a node ID without an active column'
        ) from exc
    return np.ascontiguousarray(cols, dtype=np.int64)


def _map_cell_ids_to_cols(
    cell_ids: np.ndarray,
    *,
    cell_id_to_col: dict[int, int],
) -> np.ndarray:
    try:
        cols = [cell_id_to_col[int(cell_id)] for cell_id in cell_ids.tolist()]
    except KeyError as exc:
        raise RefractionStaticDesignMatrixError(
            'row_midpoint_cell_id contains a cell ID without an active column'
        ) from exc
    return np.ascontiguousarray(cols, dtype=np.int64)


def _validate_matrix_package(
    *,
    matrix: sparse.csr_matrix,
    rhs_s: np.ndarray,
    n_observations: int,
    n_parameters: int,
) -> None:
    if matrix.shape != (n_observations, n_parameters):
        raise RefractionStaticDesignMatrixError('refraction design matrix shape mismatch')
    if matrix.format != 'csr':
        raise RefractionStaticDesignMatrixError('refraction design matrix must be CSR')
    if matrix.dtype != np.float64:
        raise RefractionStaticDesignMatrixError(
            'refraction design matrix dtype must be float64'
        )
    if rhs_s.shape != (n_observations,):
        raise RefractionStaticDesignMatrixError('rhs_s shape mismatch')
    if rhs_s.dtype != np.float64:
        raise RefractionStaticDesignMatrixError('rhs_s dtype must be float64')


def _validate_bedrock_velocity_mode(value: object) -> BedrockVelocityMode:
    if value == 'solve_global':
        return 'solve_global'
    if value == 'fixed_global':
        return 'fixed_global'
    if value == 'solve_cell':
        return 'solve_cell'
    raise RefractionStaticDesignMatrixError(
        'model.bedrock_velocity_mode must be solve_global, fixed_global, or solve_cell'
    )


def _validate_cell_assignment_mode(value: object) -> CellAssignmentMode:
    if value == 'midpoint':
        return 'midpoint'
    raise RefractionStaticDesignMatrixError(
        'model.refractor_cell.assignment_mode must be midpoint'
    )


def _validate_min_observations_per_cell(value: int | None) -> int:
    if value is None:
        return 1
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise RefractionStaticDesignMatrixError(
            'min_observations_per_cell must be a positive integer'
        )
    out = int(value)
    if out <= 0:
        raise RefractionStaticDesignMatrixError(
            'min_observations_per_cell must be a positive integer'
        )
    return out


def _validate_n_total_cells(value: int | None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise RefractionStaticDesignMatrixError(
            'n_total_cells must be a positive integer for solve_cell mode'
        )
    out = int(value)
    if out <= 0:
        raise RefractionStaticDesignMatrixError(
            'n_total_cells must be a positive integer for solve_cell mode'
        )
    return out


def _validate_cell_grid_shape(
    *,
    n_total_cells: int,
    number_of_cell_x: int | None,
    number_of_cell_y: int | None,
) -> tuple[int, int]:
    if number_of_cell_x is None:
        n_x = int(n_total_cells)
    else:
        n_x = _validate_cell_grid_axis_count(
            number_of_cell_x,
            name='number_of_cell_x',
        )
    if number_of_cell_y is None:
        n_y = 1
    else:
        n_y = _validate_cell_grid_axis_count(
            number_of_cell_y,
            name='number_of_cell_y',
        )
    if n_x * n_y != int(n_total_cells):
        raise RefractionStaticDesignMatrixError(
            'number_of_cell_x * number_of_cell_y must equal n_total_cells'
        )
    return n_x, n_y


def _validate_cell_grid_axis_count(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise RefractionStaticDesignMatrixError(f'{name} must be a positive integer')
    out = int(value)
    if out <= 0:
        raise RefractionStaticDesignMatrixError(f'{name} must be a positive integer')
    return out


def _validate_midpoint_cell_ids(
    values: np.ndarray,
    *,
    n_total_cells: int,
) -> None:
    invalid_mask = (values < -1) | (values >= n_total_cells)
    if np.any(invalid_mask):
        invalid = int(values[np.flatnonzero(invalid_mask)[0]])
        raise RefractionStaticDesignMatrixError(
            'midpoint_cell_id_sorted contains invalid cell ID '
            f'{invalid}; expected -1 or a cell ID in [0, {n_total_cells})'
        )


def _validate_n_traces(value: int | None, *, default: int) -> int:
    if value is None:
        out = default
    else:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise RefractionStaticDesignMatrixError('input_model.n_traces must be an integer')
        out = int(value)
    if out <= 0:
        raise RefractionStaticDesignMatrixError(
            'input_model.n_traces must be greater than 0'
        )
    return out


def _coerce_unique_node_id(values: np.ndarray) -> np.ndarray:
    node_id = _coerce_1d_integer_int64(values, name='input_model.endpoint_table.node_id')
    if node_id.shape[0] == 0:
        raise RefractionStaticDesignMatrixError(
            'input_model.endpoint_table.node_id must contain at least one node'
        )
    if np.unique(node_id).shape[0] != node_id.shape[0]:
        raise RefractionStaticDesignMatrixError(
            'input_model.endpoint_table.node_id values must be unique'
        )
    return node_id


def _coerce_optional_node_kind(
    values: np.ndarray | None,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionStaticDesignMatrixError(
            'input_model.endpoint_table.kind must be a 1D array'
        )
    if arr.shape != expected_shape:
        raise RefractionStaticDesignMatrixError(
            'input_model.endpoint_table.kind shape mismatch: '
            f'expected {expected_shape}, got {arr.shape}'
        )
    return np.ascontiguousarray(arr.astype('<U16'), dtype='<U16')


def _coerce_optional_endpoint_key(
    values: np.ndarray | None,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionStaticDesignMatrixError(f'{name} must be a 1D array')
    if arr.shape != expected_shape:
        raise RefractionStaticDesignMatrixError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    return np.ascontiguousarray(arr.astype('<U128'), dtype='<U128')


def _coerce_optional_rejection_reason(
    values: np.ndarray | None,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise RefractionStaticDesignMatrixError(
            'rejection_reason_sorted must be a 1D array'
        )
    if arr.shape != expected_shape:
        raise RefractionStaticDesignMatrixError(
            'rejection_reason_sorted shape mismatch: '
            f'expected {expected_shape}, got {arr.shape}'
        )
    return np.ascontiguousarray(arr.astype('<U64'), dtype='<U64')


__all__ = [
    'LOW_FOLD_CELL_REJECTION_REASON',
    'LOW_FOLD_CELL_VELOCITY_STATUS',
    'RefractionStaticDesignMatrix',
    'RefractionStaticDesignMatrixError',
    'REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME',
    'REFRACTION_DESIGN_MATRIX_QC_JSON_NAME',
    'all_refraction_design_matrix_layer_artifact_names',
    'build_refraction_static_cell_design_matrix',
    'build_refraction_static_design_matrix',
    'build_refraction_static_design_matrix_from_arrays',
    'refraction_design_matrix_layer_artifact_names',
    'refraction_design_matrix_layer_node_diagnostics_csv_name',
    'refraction_design_matrix_layer_qc_json_name',
    'write_refraction_design_matrix_diagnostics_artifacts',
]
