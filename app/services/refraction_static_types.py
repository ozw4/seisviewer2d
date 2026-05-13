"""Dependency-light result types for GLI refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np


BedrockVelocityMode = Literal['solve_global', 'fixed_global', 'solve_cell']
RefractionFirstLayerMode = Literal['constant', 'estimate_direct_arrival']
RefractionLayerKind = Literal['v2_t1', 'v3_t2', 'vsub_t3']
RefractionLayerVelocityMode = BedrockVelocityMode
RefractionSourceDepthMode = Literal['none', 'weathering_velocity_time']
RefractionSourceDepthStatus = Literal[
    'ok',
    'missing_source_depth',
    'invalid_source_depth',
    'inconsistent_source_depth',
    'exceeds_max_abs_source_depth',
    'inactive_source_endpoint',
]
RefractionUpholeStatus = Literal[
    'ok',
    'missing_uphole_time',
    'invalid_uphole_time',
    'inconsistent_uphole_time',
    'exceeds_max_abs_uphole_time',
    'inactive_source_endpoint',
]
RefractionFieldCorrectionComponentName = Literal[
    'source_depth_shift_s',
    'uphole_shift_s',
    'manual_static_shift_s',
]
REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES: Final[
    tuple[RefractionFieldCorrectionComponentName, ...]
] = (
    'source_depth_shift_s',
    'uphole_shift_s',
    'manual_static_shift_s',
)


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
class RefractionLayerObservationMasks:
    """Per-layer sorted-observation masks for multi-layer refraction branches."""

    layer_kind: np.ndarray
    layer_enabled: np.ndarray
    layer_min_offset_m: np.ndarray
    layer_max_offset_m: np.ndarray
    layer_used_mask_sorted: dict[str, np.ndarray]
    layer_rejection_reason_sorted: dict[str, np.ndarray]
    layer_candidate_count: dict[str, int]
    layer_observation_count: dict[str, int]


@dataclass(frozen=True)
class RefractionSourceDepthResult:
    """Resolved source-depth values aggregated to source endpoints."""

    source_endpoint_key: np.ndarray
    source_endpoint_id: np.ndarray
    source_node_id: np.ndarray
    source_depth_m: np.ndarray
    source_depth_status: np.ndarray
    source_depth_pick_count: np.ndarray
    source_depth_trace_count: np.ndarray
    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionUpholeResult:
    """Resolved uphole-time values aggregated to source endpoints."""

    source_endpoint_key: np.ndarray
    source_endpoint_id: np.ndarray
    source_node_id: np.ndarray
    uphole_time_s: np.ndarray
    uphole_status: np.ndarray
    uphole_pick_count: np.ndarray
    uphole_trace_count: np.ndarray
    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionManualStaticResult:
    """Resolved manual static values matched to source and receiver endpoints."""

    source_endpoint_key: np.ndarray
    source_endpoint_id: np.ndarray | None
    source_node_id: np.ndarray
    source_manual_static_shift_s: np.ndarray
    source_manual_static_status: np.ndarray

    receiver_endpoint_key: np.ndarray
    receiver_endpoint_id: np.ndarray | None
    receiver_node_id: np.ndarray
    receiver_manual_static_shift_s: np.ndarray
    receiver_manual_static_status: np.ndarray

    qc: dict[str, Any]


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
    layer_observation_masks: RefractionLayerObservationMasks | None = None
    source_depth_result: RefractionSourceDepthResult | None = None
    uphole_result: RefractionUpholeResult | None = None
    source_endpoint_id_sorted: np.ndarray | None = None
    receiver_endpoint_id_sorted: np.ndarray | None = None


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
    number_of_cell_x: int | None = None
    number_of_cell_y: int | None = None
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
class RefractionLayerSolveResult:
    """Dependency-light result container for one refraction layer solve."""

    layer_kind: RefractionLayerKind
    layer_index: int
    velocity_mode: RefractionLayerVelocityMode

    source_time_term_s: np.ndarray
    receiver_time_term_s: np.ndarray
    node_time_term_s: np.ndarray | None

    global_velocity_m_s: float | None
    global_slowness_s_per_m: float | None
    cell_velocity_m_s: np.ndarray | None
    cell_slowness_s_per_m: np.ndarray | None

    trace_predicted_time_s_sorted: np.ndarray
    trace_residual_s_sorted: np.ndarray
    used_observation_mask_sorted: np.ndarray

    layer_status: str
    qc: dict[str, Any]

    active_cell_id: np.ndarray | None = None
    inactive_cell_id: np.ndarray | None = None
    cell_velocity_status: np.ndarray | None = None
    row_midpoint_cell_id: np.ndarray | None = None
    row_midpoint_velocity_m_s: np.ndarray | None = None
    rejected_by_robust_mask_sorted: np.ndarray | None = None
    candidate_observation_mask_sorted: np.ndarray | None = None
    rejection_reason_sorted: np.ndarray | None = None


@dataclass(frozen=True)
class RefractionMultiLayerSolveResult:
    """Dependency-light container for enabled multi-layer time-term solves."""

    enabled_layer_kinds: tuple[RefractionLayerKind, ...]
    layer_results: tuple[RefractionLayerSolveResult, ...]

    source_endpoint_key: np.ndarray
    receiver_endpoint_key: np.ndarray
    source_node_id: np.ndarray
    receiver_node_id: np.ndarray

    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionMultiLayerStaticComponents:
    """Dependency-light multi-layer time-term and static components."""

    source_t1_s: np.ndarray
    source_t2_s: np.ndarray | None
    source_t3_s: np.ndarray | None
    receiver_t1_s: np.ndarray
    receiver_t2_s: np.ndarray | None
    receiver_t3_s: np.ndarray | None

    source_sh1_m: np.ndarray
    source_sh2_m: np.ndarray | None
    source_sh3_m: np.ndarray | None
    receiver_sh1_m: np.ndarray
    receiver_sh2_m: np.ndarray | None
    receiver_sh3_m: np.ndarray | None

    source_weathering_correction_s: np.ndarray
    receiver_weathering_correction_s: np.ndarray

    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionEndpointFieldCorrectionResult:
    """Dependency-light endpoint-level source-depth, uphole, and manual statics."""

    endpoint_kind: np.ndarray
    endpoint_key: np.ndarray
    endpoint_id: np.ndarray
    node_id: np.ndarray

    component_shift_s: dict[RefractionFieldCorrectionComponentName, np.ndarray]
    component_status: dict[RefractionFieldCorrectionComponentName, np.ndarray]

    total_field_shift_s: np.ndarray
    field_static_status: np.ndarray
    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionTraceFieldCorrectionResult:
    """Dependency-light trace-order field static composition result."""

    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray

    source_field_shift_s_sorted: np.ndarray
    receiver_field_shift_s_sorted: np.ndarray
    trace_field_shift_s_sorted: np.ndarray
    trace_field_static_status_sorted: np.ndarray

    qc: dict[str, Any]


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

    active_cell_id: np.ndarray | None = None
    inactive_cell_id: np.ndarray | None = None
    cell_bedrock_slowness_s_per_m: np.ndarray | None = None
    cell_bedrock_velocity_m_s: np.ndarray | None = None
    cell_velocity_status: np.ndarray | None = None
    row_midpoint_cell_id: np.ndarray | None = None
    row_midpoint_bedrock_velocity_m_s: np.ndarray | None = None


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

    active_cell_id: np.ndarray | None = None
    inactive_cell_id: np.ndarray | None = None
    cell_bedrock_slowness_s_per_m: np.ndarray | None = None
    cell_bedrock_velocity_m_s: np.ndarray | None = None
    cell_velocity_status: np.ndarray | None = None
    row_midpoint_cell_id: np.ndarray | None = None
    node_v2_cell_id: np.ndarray | None = None
    node_v2_m_s: np.ndarray | None = None
    node_v2_status: np.ndarray | None = None
    source_v2_cell_id: np.ndarray | None = None
    source_v2_m_s: np.ndarray | None = None
    source_v2_status: np.ndarray | None = None
    receiver_v2_cell_id: np.ndarray | None = None
    receiver_v2_m_s: np.ndarray | None = None
    receiver_v2_status: np.ndarray | None = None
    source_v2_cell_id_sorted: np.ndarray | None = None
    source_v2_m_s_sorted: np.ndarray | None = None
    source_v2_status_sorted: np.ndarray | None = None
    receiver_v2_cell_id_sorted: np.ndarray | None = None
    receiver_v2_m_s_sorted: np.ndarray | None = None
    receiver_v2_status_sorted: np.ndarray | None = None


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

    active_cell_id: np.ndarray | None = None
    inactive_cell_id: np.ndarray | None = None
    cell_bedrock_slowness_s_per_m: np.ndarray | None = None
    cell_bedrock_velocity_m_s: np.ndarray | None = None
    cell_velocity_status: np.ndarray | None = None
    row_midpoint_cell_id: np.ndarray | None = None
    node_v2_cell_id: np.ndarray | None = None
    node_v2_m_s: np.ndarray | None = None
    node_v2_status: np.ndarray | None = None
    source_v2_cell_id: np.ndarray | None = None
    source_v2_m_s: np.ndarray | None = None
    source_v2_status: np.ndarray | None = None
    receiver_v2_cell_id: np.ndarray | None = None
    receiver_v2_m_s: np.ndarray | None = None
    receiver_v2_status: np.ndarray | None = None
    source_v2_cell_id_sorted: np.ndarray | None = None
    source_v2_m_s_sorted: np.ndarray | None = None
    source_v2_status_sorted: np.ndarray | None = None
    receiver_v2_cell_id_sorted: np.ndarray | None = None
    receiver_v2_m_s_sorted: np.ndarray | None = None
    receiver_v2_status_sorted: np.ndarray | None = None
    node_sh1_weathering_thickness_m: np.ndarray | None = None
    node_sh2_weathering_thickness_m: np.ndarray | None = None
    node_sh3_weathering_thickness_m: np.ndarray | None = None
    source_t2_time_s: np.ndarray | None = None
    source_t3_time_s: np.ndarray | None = None
    source_v3_m_s: np.ndarray | None = None
    source_vsub_m_s: np.ndarray | None = None
    source_sh1_weathering_thickness_m: np.ndarray | None = None
    source_sh2_weathering_thickness_m: np.ndarray | None = None
    source_sh3_weathering_thickness_m: np.ndarray | None = None
    receiver_t2_time_s: np.ndarray | None = None
    receiver_t3_time_s: np.ndarray | None = None
    receiver_v3_m_s: np.ndarray | None = None
    receiver_vsub_m_s: np.ndarray | None = None
    receiver_sh1_weathering_thickness_m: np.ndarray | None = None
    receiver_sh2_weathering_thickness_m: np.ndarray | None = None
    receiver_sh3_weathering_thickness_m: np.ndarray | None = None
    row_layer_kind: np.ndarray | None = None
    row_layer_index: np.ndarray | None = None
    row_source_endpoint_key: np.ndarray | None = None
    row_receiver_endpoint_key: np.ndarray | None = None
    row_rejection_reason: np.ndarray | None = None
    row_velocity_m_s: np.ndarray | None = None


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

    active_cell_id: np.ndarray | None = None
    inactive_cell_id: np.ndarray | None = None
    cell_bedrock_slowness_s_per_m: np.ndarray | None = None
    cell_bedrock_velocity_m_s: np.ndarray | None = None
    cell_velocity_status: np.ndarray | None = None
    row_midpoint_cell_id: np.ndarray | None = None
    node_v2_cell_id: np.ndarray | None = None
    node_v2_m_s: np.ndarray | None = None
    node_v2_status: np.ndarray | None = None
    source_v2_cell_id: np.ndarray | None = None
    source_v2_m_s: np.ndarray | None = None
    source_v2_status: np.ndarray | None = None
    receiver_v2_cell_id: np.ndarray | None = None
    receiver_v2_m_s: np.ndarray | None = None
    receiver_v2_status: np.ndarray | None = None
    source_v2_cell_id_sorted: np.ndarray | None = None
    source_v2_m_s_sorted: np.ndarray | None = None
    source_v2_status_sorted: np.ndarray | None = None
    receiver_v2_cell_id_sorted: np.ndarray | None = None
    receiver_v2_m_s_sorted: np.ndarray | None = None
    receiver_v2_status_sorted: np.ndarray | None = None
    node_sh1_weathering_thickness_m: np.ndarray | None = None
    node_sh2_weathering_thickness_m: np.ndarray | None = None
    node_sh3_weathering_thickness_m: np.ndarray | None = None
    source_t2_time_s: np.ndarray | None = None
    source_t3_time_s: np.ndarray | None = None
    source_v3_m_s: np.ndarray | None = None
    source_vsub_m_s: np.ndarray | None = None
    source_sh1_weathering_thickness_m: np.ndarray | None = None
    source_sh2_weathering_thickness_m: np.ndarray | None = None
    source_sh3_weathering_thickness_m: np.ndarray | None = None
    receiver_t2_time_s: np.ndarray | None = None
    receiver_t3_time_s: np.ndarray | None = None
    receiver_v3_m_s: np.ndarray | None = None
    receiver_vsub_m_s: np.ndarray | None = None
    receiver_sh1_weathering_thickness_m: np.ndarray | None = None
    receiver_sh2_weathering_thickness_m: np.ndarray | None = None
    receiver_sh3_weathering_thickness_m: np.ndarray | None = None
    row_layer_kind: np.ndarray | None = None
    row_layer_index: np.ndarray | None = None
    row_source_endpoint_key: np.ndarray | None = None
    row_receiver_endpoint_key: np.ndarray | None = None
    row_rejection_reason: np.ndarray | None = None
    row_velocity_m_s: np.ndarray | None = None
    layer_results: tuple[RefractionLayerSolveResult, ...] | None = None
    source_depth_m: np.ndarray | None = None
    source_depth_shift_s: np.ndarray | None = None
    source_depth_status: np.ndarray | None = None
    source_depth_field_correction_qc: dict[str, Any] | None = None
    source_uphole_time_s: np.ndarray | None = None
    source_uphole_shift_s: np.ndarray | None = None
    source_uphole_status: np.ndarray | None = None
    source_uphole_field_correction_qc: dict[str, Any] | None = None
    source_manual_static_shift_s: np.ndarray | None = None
    source_manual_static_status: np.ndarray | None = None
    receiver_manual_static_shift_s: np.ndarray | None = None
    receiver_manual_static_status: np.ndarray | None = None
    manual_static_field_correction_qc: dict[str, Any] | None = None
    source_field_shift_s: np.ndarray | None = None
    source_field_static_status: np.ndarray | None = None
    receiver_field_shift_s: np.ndarray | None = None
    receiver_field_static_status: np.ndarray | None = None
    source_field_shift_s_sorted: np.ndarray | None = None
    receiver_field_shift_s_sorted: np.ndarray | None = None
    trace_field_shift_s_sorted: np.ndarray | None = None
    trace_field_static_status_sorted: np.ndarray | None = None
    trace_field_static_valid_mask_sorted: np.ndarray | None = None
    base_refraction_trace_shift_s_sorted: np.ndarray | None = None
    final_trace_shift_s_sorted: np.ndarray | None = None
    final_trace_static_status_sorted: np.ndarray | None = None
    final_trace_static_valid_mask_sorted: np.ndarray | None = None
    applied_field_shift_s_sorted: np.ndarray | None = None
    field_composition_qc: dict[str, Any] | None = None


@dataclass(frozen=True)
class RefractionStaticArtifactSet:
    job_dir: Path
    solution_npz: Path
    qc_json: Path
    refraction_statics_csv: Path
    near_surface_model_csv: Path
    first_break_residuals_csv: Path
    refraction_first_break_time_export_csv: Path
    refraction_first_break_fit_qc_csv: Path
    refraction_first_break_fit_qc_npz: Path
    refraction_first_break_fit_qc_json: Path
    refraction_static_components_csv: Path
    source_static_table_csv: Path
    receiver_static_table_csv: Path
    source_receiver_static_table_npz: Path
    refraction_time_term_spreadsheet_csv: Path
    static_history_json: Path
    manifest_json: Path | None
    artifact_names: tuple[str, ...]
    qc: dict[str, Any]
    refraction_t1lsst_1layer_components_csv: Path | None = None
    refraction_refractor_velocity_cells_csv: Path | None = None
    refraction_refractor_velocity_grid_npz: Path | None = None
    refraction_refractor_velocity_qc_json: Path | None = None
    refraction_cell_solver_history_csv: Path | None = None


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
    'REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES',
    'RefractionFirstLayerMode',
    'RefractionBedrockSlownessResult',
    'RefractionDatumStaticsResult',
    'RefractionEndpointTable',
    'RefractionEndpointFieldCorrectionResult',
    'RefractionFieldCorrectionComponentName',
    'RefractionHalfInterceptTimeResult',
    'RefractionLayerKind',
    'RefractionLayerObservationMasks',
    'RefractionLayerSolveResult',
    'RefractionLayerVelocityMode',
    'RefractionManualStaticResult',
    'RefractionMultiLayerSolveResult',
    'RefractionMultiLayerStaticComponents',
    'RefractionSourceDepthMode',
    'RefractionSourceDepthResult',
    'RefractionSourceDepthStatus',
    'RefractionUpholeResult',
    'RefractionUpholeStatus',
    'RefractionStaticApplyTraceStoreResult',
    'RefractionStaticArtifactSet',
    'RefractionStaticDesignMatrix',
    'RefractionStaticInputModel',
    'RefractionStaticSolverResult',
    'RefractionTraceShiftValidationResult',
    'RefractionTraceFieldCorrectionResult',
    'RefractionWeatheringReplacementStaticsResult',
    'RefractionWeatheringThicknessResult',
    'ResolvedRefractionFirstLayer',
]
