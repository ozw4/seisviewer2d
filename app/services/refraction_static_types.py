"""Dependency-light result types for GLI refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


BedrockVelocityMode = Literal['solve_global', 'fixed_global', 'solve_cell']
RefractionFirstLayerMode = Literal['constant', 'estimate_direct_arrival']


@dataclass(frozen=True)
class ResolvedRefractionFirstLayer:
    """Resolved V1/first-layer velocity used by downstream refraction statics."""

    mode: RefractionFirstLayerMode
    weathering_velocity_m_s: float
    status: str
    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionEndpointTable:
    node_id: np.ndarray
    endpoint_id: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    elevation_m: np.ndarray
    kind: np.ndarray
    pick_count: np.ndarray


@dataclass(frozen=True)
class RefractionStaticInputModel:
    file_id: str
    n_traces: int

    sorted_trace_index: np.ndarray
    pick_time_s_sorted: np.ndarray
    valid_pick_mask_sorted: np.ndarray
    valid_observation_mask_sorted: np.ndarray

    source_id_sorted: np.ndarray
    receiver_id_sorted: np.ndarray

    source_x_m_sorted: np.ndarray
    source_y_m_sorted: np.ndarray
    receiver_x_m_sorted: np.ndarray
    receiver_y_m_sorted: np.ndarray

    source_elevation_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray
    source_depth_m_sorted: np.ndarray | None

    geometry_distance_m_sorted: np.ndarray
    offset_m_sorted: np.ndarray | None
    distance_m_sorted: np.ndarray

    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray

    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_elevation_m: np.ndarray
    node_kind: np.ndarray

    rejection_reason_sorted: np.ndarray
    qc: dict[str, Any]
    endpoint_table: RefractionEndpointTable
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RefractionStaticDesignMatrix:
    matrix: Any
    rhs_s: np.ndarray

    observed_pick_time_s: np.ndarray
    row_trace_index_sorted: np.ndarray
    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    row_distance_m: np.ndarray

    active_node_id: np.ndarray
    inactive_node_id: np.ndarray
    node_id_to_col: dict[int, int]
    source_node_col: np.ndarray
    receiver_node_col: np.ndarray

    bedrock_slowness_col: int | None
    bedrock_velocity_mode: BedrockVelocityMode
    fixed_bedrock_velocity_m_s: float | None
    fixed_bedrock_slowness_s_per_m: float | None

    n_total_nodes: int
    n_active_nodes: int
    n_observations: int
    n_parameters: int

    qc: dict[str, Any]

    bedrock_slowness_cell_col_start: int | None = None
    active_cell_id: np.ndarray | None = None
    inactive_cell_id: np.ndarray | None = None
    cell_id_to_col: dict[int, int] | None = None
    row_midpoint_cell_id: np.ndarray | None = None
    row_midpoint_cell_col: np.ndarray | None = None
    cell_assignment_mode: str | None = None
    n_total_cells: int | None = None
    n_active_cells: int | None = None
    n_inactive_cells: int | None = None
    rejection_reason_sorted: np.ndarray | None = None


@dataclass(frozen=True)
class RefractionStaticSolverResult:
    """Output from the bounded GLI least-squares solver."""

    parameter_vector: np.ndarray

    active_node_id: np.ndarray
    active_node_half_intercept_time_s: np.ndarray

    node_id: np.ndarray
    node_half_intercept_time_s: np.ndarray
    node_solution_status: np.ndarray

    bedrock_velocity_mode: BedrockVelocityMode
    bedrock_slowness_s_per_m: float
    bedrock_velocity_m_s: float

    row_trace_index_sorted: np.ndarray
    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    row_distance_m: np.ndarray
    observed_pick_time_s: np.ndarray
    modeled_pick_time_s: np.ndarray
    residual_time_s: np.ndarray
    used_row_mask: np.ndarray
    rejected_by_robust_mask: np.ndarray

    solver_status: str
    solver_message: str
    solver_cost: float
    solver_optimality: float | None
    solver_nit: int | None
    robust_iteration_count: int

    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    qc: dict[str, Any]

    active_cell_id: np.ndarray | None = None
    inactive_cell_id: np.ndarray | None = None
    cell_bedrock_slowness_s_per_m: np.ndarray | None = None
    cell_bedrock_velocity_m_s: np.ndarray | None = None
    cell_velocity_status: np.ndarray | None = None
    row_midpoint_cell_id: np.ndarray | None = None
    row_midpoint_bedrock_velocity_m_s: np.ndarray | None = None


@dataclass(frozen=True)
class RefractionBedrockSlownessResult:
    """Integration result for the solve-global bedrock slowness workflow."""

    bedrock_velocity_mode: Literal['solve_global']
    weathering_velocity_m_s: float

    bedrock_slowness_s_per_m: float
    bedrock_velocity_m_s: float
    bedrock_velocity_status: str

    min_bedrock_velocity_m_s: float
    max_bedrock_velocity_m_s: float
    lower_bedrock_slowness_s_per_m: float
    upper_bedrock_slowness_s_per_m: float

    active_node_id: np.ndarray
    active_node_half_intercept_time_s: np.ndarray

    row_trace_index_sorted: np.ndarray
    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    row_distance_m: np.ndarray
    observed_pick_time_s: np.ndarray
    modeled_pick_time_s: np.ndarray
    residual_time_s: np.ndarray
    used_row_mask: np.ndarray
    rejected_by_robust_mask: np.ndarray

    input_model: RefractionStaticInputModel | None
    design_matrix: RefractionStaticDesignMatrix | None
    solver_result: RefractionStaticSolverResult

    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionHalfInterceptTimeResult:
    """Full node, endpoint, trace-order, and QC output for GLI half-intercepts."""

    bedrock_velocity_mode: BedrockVelocityMode
    bedrock_slowness_s_per_m: float
    bedrock_velocity_m_s: float
    weathering_velocity_m_s: float

    node_id: np.ndarray
    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_elevation_m: np.ndarray
    node_kind: np.ndarray

    node_half_intercept_time_s: np.ndarray
    node_half_intercept_time_ms: np.ndarray
    node_solution_status: np.ndarray

    node_pick_count: np.ndarray
    node_used_pick_count: np.ndarray
    node_rejected_pick_count: np.ndarray
    node_residual_mean_s: np.ndarray
    node_residual_median_s: np.ndarray
    node_residual_rms_s: np.ndarray
    node_residual_mad_s: np.ndarray
    node_residual_max_abs_s: np.ndarray

    source_endpoint_key: np.ndarray
    source_id: np.ndarray
    source_node_id: np.ndarray
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    source_elevation_m: np.ndarray
    source_half_intercept_time_s: np.ndarray
    source_solution_status: np.ndarray
    source_pick_count: np.ndarray
    source_residual_rms_s: np.ndarray

    receiver_endpoint_key: np.ndarray
    receiver_id: np.ndarray
    receiver_node_id: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    receiver_elevation_m: np.ndarray
    receiver_half_intercept_time_s: np.ndarray
    receiver_solution_status: np.ndarray
    receiver_pick_count: np.ndarray
    receiver_residual_rms_s: np.ndarray

    sorted_trace_index: np.ndarray
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_elevation_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    source_half_intercept_time_s_sorted: np.ndarray
    receiver_half_intercept_time_s_sorted: np.ndarray
    estimated_intercept_time_sum_s_sorted: np.ndarray
    estimated_bedrock_moveout_time_s_sorted: np.ndarray
    estimated_first_break_time_s_sorted: np.ndarray
    first_break_residual_s_sorted: np.ndarray
    valid_observation_mask_sorted: np.ndarray
    used_observation_mask_sorted: np.ndarray

    row_trace_index_sorted: np.ndarray
    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    row_distance_m: np.ndarray
    observed_pick_time_s: np.ndarray
    modeled_pick_time_s: np.ndarray
    residual_time_s: np.ndarray
    used_row_mask: np.ndarray
    rejected_by_robust_mask: np.ndarray

    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionWeatheringThicknessResult:
    """Node, endpoint, trace-order, and QC output for GLI weathering thickness."""

    bedrock_velocity_mode: BedrockVelocityMode
    bedrock_slowness_s_per_m: float
    bedrock_velocity_m_s: float
    weathering_velocity_m_s: float

    node_id: np.ndarray
    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_surface_elevation_m: np.ndarray
    node_kind: np.ndarray

    node_half_intercept_time_s: np.ndarray
    node_half_intercept_time_ms: np.ndarray
    node_weathering_thickness_m: np.ndarray
    node_refractor_elevation_m: np.ndarray
    node_solution_status: np.ndarray
    node_weathering_status: np.ndarray

    node_pick_count: np.ndarray
    node_used_pick_count: np.ndarray
    node_rejected_pick_count: np.ndarray
    node_residual_rms_s: np.ndarray
    node_residual_mad_s: np.ndarray

    source_endpoint_key: np.ndarray
    source_id: np.ndarray
    source_node_id: np.ndarray
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    source_surface_elevation_m: np.ndarray
    source_half_intercept_time_s: np.ndarray
    source_weathering_thickness_m: np.ndarray
    source_refractor_elevation_m: np.ndarray
    source_weathering_status: np.ndarray

    receiver_endpoint_key: np.ndarray
    receiver_id: np.ndarray
    receiver_node_id: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    receiver_surface_elevation_m: np.ndarray
    receiver_half_intercept_time_s: np.ndarray
    receiver_weathering_thickness_m: np.ndarray
    receiver_refractor_elevation_m: np.ndarray
    receiver_weathering_status: np.ndarray

    sorted_trace_index: np.ndarray
    valid_observation_mask_sorted: np.ndarray
    used_observation_mask_sorted: np.ndarray

    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray

    source_half_intercept_time_s_sorted: np.ndarray
    receiver_half_intercept_time_s_sorted: np.ndarray

    source_weathering_thickness_m_sorted: np.ndarray
    receiver_weathering_thickness_m_sorted: np.ndarray
    source_refractor_elevation_m_sorted: np.ndarray
    receiver_refractor_elevation_m_sorted: np.ndarray
    source_weathering_status_sorted: np.ndarray
    receiver_weathering_status_sorted: np.ndarray

    estimated_first_break_time_s_sorted: np.ndarray
    first_break_residual_s_sorted: np.ndarray

    row_trace_index_sorted: np.ndarray
    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    row_distance_m: np.ndarray
    observed_pick_time_s: np.ndarray
    modeled_pick_time_s: np.ndarray
    residual_time_s: np.ndarray
    used_row_mask: np.ndarray
    rejected_by_robust_mask: np.ndarray

    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionWeatheringReplacementStaticsResult:
    """Weathering-replacement static component from a GLI weathering model."""

    bedrock_velocity_mode: BedrockVelocityMode
    bedrock_slowness_s_per_m: float
    bedrock_velocity_m_s: float
    weathering_velocity_m_s: float
    replacement_slowness_delta_s_per_m: float

    node_id: np.ndarray
    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_surface_elevation_m: np.ndarray
    node_kind: np.ndarray
    node_weathering_thickness_m: np.ndarray
    node_refractor_elevation_m: np.ndarray
    node_half_intercept_time_s: np.ndarray
    node_solution_status: np.ndarray
    node_weathering_status: np.ndarray
    node_weathering_replacement_shift_s: np.ndarray
    node_weathering_replacement_shift_ms: np.ndarray
    node_static_status: np.ndarray
    node_pick_count: np.ndarray
    node_used_pick_count: np.ndarray
    node_rejected_pick_count: np.ndarray
    node_residual_rms_s: np.ndarray
    node_residual_mad_s: np.ndarray

    source_endpoint_key: np.ndarray
    source_id: np.ndarray
    source_node_id: np.ndarray
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    source_surface_elevation_m: np.ndarray
    source_half_intercept_time_s: np.ndarray
    source_weathering_thickness_m: np.ndarray
    source_refractor_elevation_m: np.ndarray
    source_weathering_replacement_shift_s: np.ndarray
    source_static_status: np.ndarray

    receiver_endpoint_key: np.ndarray
    receiver_id: np.ndarray
    receiver_node_id: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    receiver_surface_elevation_m: np.ndarray
    receiver_half_intercept_time_s: np.ndarray
    receiver_weathering_thickness_m: np.ndarray
    receiver_refractor_elevation_m: np.ndarray
    receiver_weathering_replacement_shift_s: np.ndarray
    receiver_static_status: np.ndarray

    sorted_trace_index: np.ndarray
    valid_observation_mask_sorted: np.ndarray
    used_observation_mask_sorted: np.ndarray
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    source_half_intercept_time_s_sorted: np.ndarray
    receiver_half_intercept_time_s_sorted: np.ndarray
    source_weathering_thickness_m_sorted: np.ndarray
    receiver_weathering_thickness_m_sorted: np.ndarray
    source_refractor_elevation_m_sorted: np.ndarray
    receiver_refractor_elevation_m_sorted: np.ndarray
    source_weathering_replacement_shift_s_sorted: np.ndarray
    receiver_weathering_replacement_shift_s_sorted: np.ndarray
    weathering_replacement_trace_shift_s_sorted: np.ndarray
    source_static_status_sorted: np.ndarray
    receiver_static_status_sorted: np.ndarray
    trace_static_status_sorted: np.ndarray
    trace_static_valid_mask_sorted: np.ndarray
    estimated_first_break_time_s_sorted: np.ndarray
    first_break_residual_s_sorted: np.ndarray

    row_trace_index_sorted: np.ndarray
    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    row_distance_m: np.ndarray
    observed_pick_time_s: np.ndarray
    modeled_pick_time_s: np.ndarray
    residual_time_s: np.ndarray
    used_row_mask: np.ndarray
    rejected_by_robust_mask: np.ndarray

    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionDatumStaticsResult:
    """Composed refraction static components in TraceStore sorted trace order."""

    bedrock_velocity_mode: BedrockVelocityMode
    bedrock_slowness_s_per_m: float
    bedrock_velocity_m_s: float
    weathering_velocity_m_s: float
    replacement_slowness_delta_s_per_m: float

    datum_mode: str
    floating_datum_mode: str
    flat_datum_elevation_m: float | None

    node_id: np.ndarray
    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_surface_elevation_m: np.ndarray
    node_kind: np.ndarray
    node_weathering_thickness_m: np.ndarray
    node_refractor_elevation_m: np.ndarray
    node_half_intercept_time_s: np.ndarray
    node_weathering_replacement_shift_s: np.ndarray
    node_floating_datum_elevation_m: np.ndarray
    node_solution_status: np.ndarray
    node_datum_status: np.ndarray
    node_weathering_status: np.ndarray
    node_pick_count: np.ndarray
    node_used_pick_count: np.ndarray
    node_rejected_pick_count: np.ndarray
    node_residual_rms_s: np.ndarray
    node_residual_mad_s: np.ndarray

    source_endpoint_key: np.ndarray
    source_id: np.ndarray
    source_node_id: np.ndarray
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    source_surface_elevation_m: np.ndarray
    source_half_intercept_time_s: np.ndarray
    source_weathering_thickness_m: np.ndarray
    source_refractor_elevation_m: np.ndarray
    source_floating_datum_elevation_m: np.ndarray
    source_weathering_replacement_shift_s: np.ndarray
    source_floating_datum_elevation_shift_s: np.ndarray
    source_flat_datum_shift_s: np.ndarray
    source_refraction_shift_s: np.ndarray
    source_datum_status: np.ndarray

    receiver_endpoint_key: np.ndarray
    receiver_id: np.ndarray
    receiver_node_id: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    receiver_surface_elevation_m: np.ndarray
    receiver_half_intercept_time_s: np.ndarray
    receiver_weathering_thickness_m: np.ndarray
    receiver_refractor_elevation_m: np.ndarray
    receiver_floating_datum_elevation_m: np.ndarray
    receiver_weathering_replacement_shift_s: np.ndarray
    receiver_floating_datum_elevation_shift_s: np.ndarray
    receiver_flat_datum_shift_s: np.ndarray
    receiver_refraction_shift_s: np.ndarray
    receiver_datum_status: np.ndarray

    sorted_trace_index: np.ndarray
    valid_observation_mask_sorted: np.ndarray
    used_observation_mask_sorted: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    source_surface_elevation_m_sorted: np.ndarray
    receiver_surface_elevation_m_sorted: np.ndarray
    source_floating_datum_elevation_m_sorted: np.ndarray
    receiver_floating_datum_elevation_m_sorted: np.ndarray
    source_weathering_thickness_m_sorted: np.ndarray
    receiver_weathering_thickness_m_sorted: np.ndarray
    source_refractor_elevation_m_sorted: np.ndarray
    receiver_refractor_elevation_m_sorted: np.ndarray
    source_half_intercept_time_s_sorted: np.ndarray
    receiver_half_intercept_time_s_sorted: np.ndarray
    source_weathering_replacement_shift_s_sorted: np.ndarray
    receiver_weathering_replacement_shift_s_sorted: np.ndarray
    source_floating_datum_elevation_shift_s_sorted: np.ndarray
    receiver_floating_datum_elevation_shift_s_sorted: np.ndarray
    source_flat_datum_shift_s_sorted: np.ndarray
    receiver_flat_datum_shift_s_sorted: np.ndarray
    source_refraction_shift_s_sorted: np.ndarray
    receiver_refraction_shift_s_sorted: np.ndarray
    weathering_replacement_trace_shift_s_sorted: np.ndarray
    floating_datum_elevation_shift_s_sorted: np.ndarray
    flat_datum_shift_s_sorted: np.ndarray
    refraction_trace_shift_s_sorted: np.ndarray
    trace_static_status_sorted: np.ndarray
    trace_static_valid_mask_sorted: np.ndarray
    estimated_first_break_time_s_sorted: np.ndarray
    first_break_residual_s_sorted: np.ndarray

    row_trace_index_sorted: np.ndarray
    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    row_distance_m: np.ndarray
    observed_pick_time_s: np.ndarray
    modeled_pick_time_s: np.ndarray
    residual_time_s: np.ndarray
    used_row_mask: np.ndarray
    rejected_by_robust_mask: np.ndarray

    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionStaticArtifactSet:
    job_dir: Path
    solution_npz: Path
    qc_json: Path
    refraction_statics_csv: Path
    near_surface_model_csv: Path
    first_break_residuals_csv: Path
    refraction_static_components_csv: Path
    source_static_table_csv: Path
    receiver_static_table_csv: Path
    source_receiver_static_table_npz: Path
    manifest_json: Path | None
    artifact_names: tuple[str, ...]
    qc: dict[str, Any]
    refraction_t1lsst_1layer_components_csv: Path | None = None


@dataclass(frozen=True)
class RefractionTraceShiftValidationResult:
    trace_shift_s_sorted: np.ndarray
    trace_static_valid_mask_sorted: np.ndarray
    trace_static_status_sorted: np.ndarray
    trace_static_status_counts: dict[str, int]
    max_abs_shift_ms: float
    max_abs_applied_shift_ms: float
    exceeds_max_abs_shift_count: int
    n_valid_trace_shifts: int
    n_invalid_trace_shifts: int
    n_zero_trace_shifts: int
    n_positive_trace_shifts: int
    n_negative_trace_shifts: int


@dataclass(frozen=True)
class RefractionStaticApplyTraceStoreResult:
    source_file_id: str
    corrected_file_id: str | None

    source_trace_store_path: Path
    corrected_trace_store_path: Path | None

    n_traces: int
    n_samples: int
    sample_interval_s: float

    interpolation: str
    fill_value: float
    output_dtype: str

    applied_shift_s_sorted: np.ndarray
    applied_shift_ms_sorted: np.ndarray
    trace_static_valid_mask_sorted: np.ndarray
    trace_static_status_sorted: np.ndarray

    max_abs_applied_shift_ms: float
    n_valid_trace_shifts: int
    n_invalid_trace_shifts: int
    n_zero_trace_shifts: int
    n_positive_trace_shifts: int
    n_negative_trace_shifts: int

    corrected_file_json: Path | None
    qc_json: Path | None
    qc: dict[str, Any]


__all__ = [
    'BedrockVelocityMode',
    'RefractionFirstLayerMode',
    'RefractionBedrockSlownessResult',
    'RefractionDatumStaticsResult',
    'RefractionEndpointTable',
    'RefractionHalfInterceptTimeResult',
    'RefractionStaticApplyTraceStoreResult',
    'RefractionStaticArtifactSet',
    'RefractionStaticDesignMatrix',
    'RefractionStaticInputModel',
    'RefractionStaticSolverResult',
    'RefractionTraceShiftValidationResult',
    'RefractionWeatheringReplacementStaticsResult',
    'RefractionWeatheringThicknessResult',
    'ResolvedRefractionFirstLayer',
]
