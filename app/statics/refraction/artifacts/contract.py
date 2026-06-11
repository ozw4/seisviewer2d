"""Artifact contract for refraction static output packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.statics.refraction.domain.export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)
from app.statics.refraction.domain.types import (
    RefractionDatumStaticsResult,
    RefractionLayerKind,
)

REFRACTION_STATIC_SOLUTION_NPZ_NAME = 'refraction_static_solution.npz'
REFRACTION_STATIC_QC_JSON_NAME = 'refraction_static_qc.json'
REFRACTION_STATICS_CSV_NAME = 'refraction_statics.csv'
NEAR_SURFACE_MODEL_CSV_NAME = 'near_surface_model.csv'
FIRST_BREAK_RESIDUALS_CSV_NAME = 'first_break_residuals.csv'
REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME = 'refraction_first_break_time_export.csv'
REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME = 'refraction_first_break_fit_qc.csv'
REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME = 'refraction_first_break_fit_qc.npz'
REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME = 'refraction_first_break_fit_qc.json'
REFRACTION_REDUCED_TIME_QC_CSV_NAME = 'refraction_reduced_time_qc.csv'
REFRACTION_REDUCED_TIME_QC_NPZ_NAME = 'refraction_reduced_time_qc.npz'
REFRACTION_REDUCED_TIME_QC_JSON_NAME = 'refraction_reduced_time_qc.json'
REFRACTION_STATIC_COMPONENTS_CSV_NAME = 'refraction_static_components.csv'
REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME = (
    'refraction_static_component_qc_trace.csv'
)
REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME = (
    'refraction_static_component_qc_endpoint.csv'
)
REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME = 'refraction_static_component_qc.npz'
REFRACTION_STATIC_COMPONENT_QC_JSON_NAME = 'refraction_static_component_qc.json'
SOURCE_STATIC_TABLE_CSV_NAME = 'source_static_table.csv'
RECEIVER_STATIC_TABLE_CSV_NAME = 'receiver_static_table.csv'
SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME = 'source_receiver_static_table.npz'
REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME = (
    'refraction_line_profile_qc_source.csv'
)
REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME = (
    'refraction_line_profile_qc_receiver.csv'
)
REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME = (
    'refraction_line_profile_qc_combined.csv'
)
REFRACTION_LINE_PROFILE_QC_NPZ_NAME = 'refraction_line_profile_qc.npz'
REFRACTION_LINE_PROFILE_QC_JSON_NAME = 'refraction_line_profile_qc.json'
REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME = 'refraction_time_term_spreadsheet.csv'
REFRACTION_STATIC_HISTORY_JSON_NAME = 'refraction_static_history.json'
REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME = (
    'refraction_refractor_velocity_cells.csv'
)
REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME = (
    'refraction_refractor_velocity_grid.npz'
)
REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME = 'refraction_refractor_velocity_qc.json'
REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME = 'refraction_cell_solver_history.csv'
REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME = (
    'refraction_v3_refractor_velocity_cells.csv'
)
REFRACTION_V3_REFRACTOR_VELOCITY_GRID_NPZ_NAME = (
    'refraction_v3_refractor_velocity_grid.npz'
)
REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME = (
    'refraction_v3_refractor_velocity_qc.json'
)
REFRACTION_V3_CELL_SOLVER_HISTORY_CSV_NAME = 'refraction_v3_cell_solver_history.csv'
REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME = (
    'refraction_vsub_refractor_velocity_cells.csv'
)
REFRACTION_VSUB_REFRACTOR_VELOCITY_GRID_NPZ_NAME = (
    'refraction_vsub_refractor_velocity_grid.npz'
)
REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME = (
    'refraction_vsub_refractor_velocity_qc.json'
)
REFRACTION_VSUB_CELL_SOLVER_HISTORY_CSV_NAME = (
    'refraction_vsub_cell_solver_history.csv'
)
REFRACTION_GRID_MAP_QC_CSV_NAME = 'refraction_grid_map_qc.csv'
REFRACTION_GRID_MAP_QC_NPZ_NAME = 'refraction_grid_map_qc.npz'
REFRACTION_GRID_MAP_QC_JSON_NAME = 'refraction_grid_map_qc.json'
REFRACTION_STATIC_ARTIFACTS_JSON_NAME = 'refraction_static_artifacts.json'
REFRACTION_STATIC_REQUEST_JSON_NAME = 'refraction_static_request.json'
REFRACTION_STATIC_FAILURE_DIAGNOSTICS_JSON_NAME = 'failure_diagnostics.json'
REFRACTION_PICK_MAP_QC_COMPLETED_CACHE_DIR_NAME = 'pick_map_cache'
UPLOADED_REFRACTION_PICKS_NPZ_NAME = 'uploaded_picks_time_s.npz'


@dataclass(frozen=True)
class _CellVelocityArtifactNames:
    cells_csv: str
    grid_npz: str
    qc_json: str
    solver_history_csv: str


@dataclass(frozen=True)
class _CellVelocityArtifactPaths:
    layer_kind: RefractionLayerKind
    cells_csv: Path
    grid_npz: Path
    qc_json: Path
    solver_history_csv: Path


ARTIFACT_VERSION = '1.0'
METHOD = 'gli_variable_thickness'
WORKFLOW = 'refraction_statics'
STATIC_COMPONENT = 'final_refraction'
SIGN_CONVENTION = REFRACTION_STATIC_REPO_SIGN_CONVENTION
POSITIVE_SHIFT_DESCRIPTION = 'event appears later in corrected data'
NEGATIVE_SHIFT_DESCRIPTION = 'event appears earlier in corrected data'
TIME_TERM_SPREADSHEET_FORMAT_NAME = 'time_term_spreadsheet'
TIME_TERM_SPREADSHEET_FORMAT_VERSION = 1
TIME_TERM_SPREADSHEET_SCHEMA_VERSION = 1
LINE_PROFILE_QC_SCHEMA_VERSION = 1
FIRST_BREAK_TIME_EXPORT_FORMAT_NAME = 'first_break_time'
FIRST_BREAK_TIME_EXPORT_FORMAT_VERSION = 1
FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION = (
    'residual_ms = observed_pick_time_ms - modeled_pick_time_ms'
)

_TRACE_STATICS_COLUMNS = (
    'sorted_trace_index',
    'valid_observation',
    'used_observation',
    'trace_static_valid',
    'trace_static_status',
    'source_node_id',
    'receiver_node_id',
    'source_surface_elevation_m',
    'receiver_surface_elevation_m',
    'source_floating_datum_elevation_m',
    'receiver_floating_datum_elevation_m',
    'source_weathering_thickness_m',
    'receiver_weathering_thickness_m',
    'source_refractor_elevation_m',
    'receiver_refractor_elevation_m',
    'source_half_intercept_time_ms',
    'receiver_half_intercept_time_ms',
    'weathering_replacement_trace_shift_ms',
    'floating_datum_elevation_shift_ms',
    'flat_datum_shift_ms',
    'refraction_trace_shift_ms',
    'estimated_first_break_time_ms',
    'first_break_residual_ms',
    'source_weathering_replacement_shift_ms',
    'receiver_weathering_replacement_shift_ms',
    'source_floating_datum_elevation_shift_ms',
    'receiver_floating_datum_elevation_shift_ms',
    'source_flat_datum_shift_ms',
    'receiver_flat_datum_shift_ms',
    'source_refraction_shift_ms',
    'receiver_refraction_shift_ms',
)

_NEAR_SURFACE_COLUMNS = (
    'node_id',
    'node_kind',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'refractor_elevation_m',
    'weathering_thickness_m',
    'half_intercept_time_ms',
    'weathering_replacement_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'pick_count',
    'used_pick_count',
    'rejected_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
)

_NEAR_SURFACE_2LAYER_COLUMNS = (
    'node_id',
    'node_kind',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'refractor_elevation_m',
    'weathering_thickness_m',
    'sh1_weathering_thickness_m',
    'sh2_weathering_thickness_m',
    'layer1_base_elevation_m',
    'final_refractor_elevation_m',
    'half_intercept_time_ms',
    'weathering_replacement_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'pick_count',
    'used_pick_count',
    'rejected_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
)

_NEAR_SURFACE_3LAYER_COLUMNS = (
    'node_id',
    'node_kind',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'refractor_elevation_m',
    'weathering_thickness_m',
    'sh1_weathering_thickness_m',
    'sh2_weathering_thickness_m',
    'sh3_weathering_thickness_m',
    'layer1_base_elevation_m',
    'layer2_base_elevation_m',
    'final_refractor_elevation_m',
    'half_intercept_time_ms',
    'weathering_replacement_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'pick_count',
    'used_pick_count',
    'rejected_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
)

# Keep the legacy millisecond residual columns and append explicit seconds/cell
# aliases so residual rows can be joined to refractor-cell QC artifacts.
_RESIDUAL_COLUMNS = (
    'row_index',
    'observation_index',
    'sorted_trace_index',
    'source_node_id',
    'receiver_node_id',
    'distance_m',
    'observed_pick_time_ms',
    'observed_pick_time_s',
    'modeled_pick_time_ms',
    'modeled_pick_time_s',
    'residual_ms',
    'residual_s',
    'used',
    'used_in_solve',
    'rejected_by_robust',
    'rejection_reason',
    'cell_id',
    'cell_ix',
    'cell_iy',
    'trace_index_sorted',
    'layer_kind',
    'layer_index',
    'source_endpoint_key',
    'receiver_endpoint_key',
    'offset_m',
    'residual_time_s',
    'midpoint_cell_id',
    'row_velocity_m_s',
)

_FIRST_BREAK_TIME_EXPORT_COLUMNS = (
    'format_name',
    'format_version',
    'source_job_id',
    'observation_index',
    'sorted_trace_index',
    'source_endpoint_key',
    'receiver_endpoint_key',
    'source_id',
    'receiver_id',
    'offset_m',
    'layer_kind',
    'observed_pick_time_ms',
    'modeled_pick_time_ms',
    'residual_ms',
    'used_in_solve',
    'reject_reason',
    'sign_convention',
)

FIRST_BREAK_FIT_QC_RESIDUAL_SIGN = 'observed - modeled'
REDUCED_TIME_QC_FORMULA = (
    'reduced_time_s = observed_first_break_time_s - '
    'offset_m / reduction_velocity_m_s'
)

_FIRST_BREAK_FIT_QC_COLUMNS = (
    'observation_index',
    'sorted_trace_index',
    'trace_index_sorted',
    'source_endpoint_key',
    'receiver_endpoint_key',
    'source_id',
    'receiver_id',
    'source_node_id',
    'receiver_node_id',
    'source_x_m',
    'source_y_m',
    'receiver_x_m',
    'receiver_y_m',
    'midpoint_x_m',
    'midpoint_y_m',
    'inline_m',
    'crossline_m',
    'offset_m',
    'observed_first_break_time_s',
    'modeled_first_break_time_s',
    'residual_time_s',
    'residual_s',
    'residual_time_ms',
    'layer_kind',
    'cell_id',
    'cell_ix',
    'cell_iy',
    'used_for_inversion',
    'used_in_solve',
    'rejection_reason',
    'reject_reason',
    'status',
    'sign_convention',
)

_REDUCED_TIME_QC_COLUMNS = (
    'trace_index_sorted',
    'source_endpoint_key',
    'receiver_endpoint_key',
    'offset_m',
    'inline_m',
    'crossline_m',
    'observed_first_break_time_s',
    'reduction_velocity_m_s',
    'reduced_time_s',
    'reduced_time_ms',
    'layer_gate_kind',
    'within_v1_gate',
    'within_v2_t1_gate',
    'within_v3_t2_gate',
    'within_vsub_t3_gate',
    'used_for_inversion',
    'status',
)

_STATIC_COMPONENT_QC_TRACE_COLUMNS = (
    'trace_index_sorted',
    'source_endpoint_key',
    'receiver_endpoint_key',
    'refraction_trace_shift_ms',
    'refraction_shift_ms',
    'weathering_shift_ms',
    'datum_shift_ms',
    'trace_field_shift_ms',
    'field_shift_ms',
    'computed_field_shift_ms',
    'applied_field_shift_ms',
    'trace_field_static_status',
    'manual_static_shift_ms',
    'source_depth_shift_ms',
    'uphole_shift_ms',
    'final_trace_shift_ms',
    'applied_trace_shift_ms',
    'apply_to_trace_shift',
    'static_status',
    'sign_convention',
)

_STATIC_COMPONENT_QC_ENDPOINT_COLUMNS = (
    'endpoint_kind',
    'endpoint_key',
    'weathering_correction_ms',
    'elevation_correction_ms',
    'source_depth_correction_ms',
    'source_depth_status',
    'uphole_correction_ms',
    'uphole_status',
    'manual_static_shift_ms',
    'manual_static_ms',
    'manual_static_status',
    'source_field_shift_ms',
    'source_field_static_status',
    'receiver_field_shift_ms',
    'receiver_field_static_status',
    'field_correction_ms',
    'computed_field_correction_ms',
    'applied_field_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'source_total_with_field_shift_ms',
    'receiver_total_with_field_shift_ms',
    'total_with_field_shift_ms',
    'apply_to_trace_shift',
    'static_status',
    'sign_convention',
)

_COMPONENT_COLUMNS = (
    'kind',
    'endpoint_key',
    'station_id',
    'node_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'refractor_elevation_m',
    'weathering_thickness_m',
    'half_intercept_time_ms',
    'weathering_replacement_shift_ms',
    'floating_datum_elevation_shift_ms',
    'flat_datum_shift_ms',
    'refraction_shift_ms',
    'datum_status',
    'pick_count',
    'residual_rms_ms',
)

_SOURCE_STATIC_TABLE_COLUMNS = (
    'endpoint_kind',
    'source_endpoint_key',
    'source_id',
    'source_node_id',
    'source_v2_cell_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    'v1_m_s',
    'v2_m_s',
    'v2_status',
    'sh1_weathering_thickness_m',
    'total_weathering_thickness_m',
    'refractor_elevation_m',
    'weathering_correction_ms',
    'floating_datum_correction_ms',
    'flat_datum_correction_ms',
    'elevation_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'static_status',
    'sign_convention',
    'pick_count',
    'used_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
)

_SOURCE_STATIC_TABLE_2LAYER_COLUMNS = (
    'endpoint_kind',
    'source_endpoint_key',
    'source_id',
    'source_node_id',
    'source_v2_cell_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    't2_ms',
    'v1_m_s',
    'v2_m_s',
    'v2_status',
    'v3_m_s',
    'sh1_weathering_thickness_m',
    'sh2_weathering_thickness_m',
    'total_weathering_thickness_m',
    'layer1_base_elevation_m',
    'final_refractor_elevation_m',
    'refractor_elevation_m',
    'weathering_correction_ms',
    'floating_datum_correction_ms',
    'flat_datum_correction_ms',
    'elevation_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'static_status',
    'sign_convention',
    'pick_count',
    'used_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
    'pick_count_by_layer',
    'used_pick_count_by_layer',
    'residual_rms_by_layer_ms',
    'residual_mad_by_layer_ms',
)

_SOURCE_STATIC_TABLE_3LAYER_COLUMNS = (
    'endpoint_kind',
    'source_endpoint_key',
    'source_id',
    'source_node_id',
    'source_v2_cell_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    't2_ms',
    't3_ms',
    'v1_m_s',
    'v2_m_s',
    'v2_status',
    'v3_m_s',
    'vsub_m_s',
    'sh1_weathering_thickness_m',
    'sh2_weathering_thickness_m',
    'sh3_weathering_thickness_m',
    'total_weathering_thickness_m',
    'layer1_base_elevation_m',
    'layer2_base_elevation_m',
    'final_refractor_elevation_m',
    'refractor_elevation_m',
    'weathering_correction_ms',
    'floating_datum_correction_ms',
    'flat_datum_correction_ms',
    'elevation_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'static_status',
    'sign_convention',
    'pick_count',
    'used_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
    'pick_count_by_layer',
    'used_pick_count_by_layer',
    'residual_rms_by_layer_ms',
    'residual_mad_by_layer_ms',
)

_RECEIVER_STATIC_TABLE_COLUMNS = (
    'endpoint_kind',
    'receiver_endpoint_key',
    'receiver_id',
    'receiver_node_id',
    'receiver_v2_cell_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    'v1_m_s',
    'v2_m_s',
    'v2_status',
    'sh1_weathering_thickness_m',
    'total_weathering_thickness_m',
    'refractor_elevation_m',
    'weathering_correction_ms',
    'floating_datum_correction_ms',
    'flat_datum_correction_ms',
    'elevation_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'static_status',
    'sign_convention',
    'pick_count',
    'used_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
)

_RECEIVER_STATIC_TABLE_2LAYER_COLUMNS = (
    'endpoint_kind',
    'receiver_endpoint_key',
    'receiver_id',
    'receiver_node_id',
    'receiver_v2_cell_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    't2_ms',
    'v1_m_s',
    'v2_m_s',
    'v2_status',
    'v3_m_s',
    'sh1_weathering_thickness_m',
    'sh2_weathering_thickness_m',
    'total_weathering_thickness_m',
    'layer1_base_elevation_m',
    'final_refractor_elevation_m',
    'refractor_elevation_m',
    'weathering_correction_ms',
    'floating_datum_correction_ms',
    'flat_datum_correction_ms',
    'elevation_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'static_status',
    'sign_convention',
    'pick_count',
    'used_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
    'pick_count_by_layer',
    'used_pick_count_by_layer',
    'residual_rms_by_layer_ms',
    'residual_mad_by_layer_ms',
)

_RECEIVER_STATIC_TABLE_3LAYER_COLUMNS = (
    'endpoint_kind',
    'receiver_endpoint_key',
    'receiver_id',
    'receiver_node_id',
    'receiver_v2_cell_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    't2_ms',
    't3_ms',
    'v1_m_s',
    'v2_m_s',
    'v2_status',
    'v3_m_s',
    'vsub_m_s',
    'sh1_weathering_thickness_m',
    'sh2_weathering_thickness_m',
    'sh3_weathering_thickness_m',
    'total_weathering_thickness_m',
    'layer1_base_elevation_m',
    'layer2_base_elevation_m',
    'final_refractor_elevation_m',
    'refractor_elevation_m',
    'weathering_correction_ms',
    'floating_datum_correction_ms',
    'flat_datum_correction_ms',
    'elevation_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'static_status',
    'sign_convention',
    'pick_count',
    'used_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
    'pick_count_by_layer',
    'used_pick_count_by_layer',
    'residual_rms_by_layer_ms',
    'residual_mad_by_layer_ms',
)

_TIME_TERM_SPREADSHEET_COLUMNS = (
    'schema_version',
    'format_name',
    'format_version',
    'source_job_id',
    'endpoint_kind',
    'endpoint_key',
    'endpoint_id',
    'station_id',
    'node_id',
    'x_m',
    'y_m',
    'elevation_m',
    'surface_elevation_m',
    't1_ms',
    't2_ms',
    't3_ms',
    'v1_m_s',
    'v2_m_s',
    'v3_m_s',
    'vsub_m_s',
    'sh1_m',
    'sh2_m',
    'sh3_m',
    'layer1_base_elevation_m',
    'layer2_base_elevation_m',
    'final_refractor_elevation_m',
    'weathering_correction_ms',
    'elevation_correction_ms',
    'source_depth_correction_ms',
    'uphole_correction_ms',
    'manual_static_ms',
    'field_correction_ms',
    'total_applied_shift_ms',
    'pick_count',
    'used_pick_count',
    'pick_count_by_layer',
    'used_pick_count_by_layer',
    'residual_rms_ms',
    'residual_mad_ms',
    'residual_rms_by_layer_ms',
    'residual_mad_by_layer_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'source_depth_status',
    'uphole_status',
    'manual_static_status',
    'field_static_status',
    'static_status',
    'sign_convention',
)

_LINE_PROFILE_QC_COLUMNS = (
    'endpoint_kind',
    'endpoint_key',
    'node_id',
    'inline_m',
    'crossline_m',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'pick_count',
    'used_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
    'v1_m_s',
    'v2_m_s',
    'v3_m_s',
    'vsub_m_s',
    't1_ms',
    't2_ms',
    't3_ms',
    'sh1_m',
    'sh2_m',
    'sh3_m',
    'layer1_base_elevation_m',
    'layer2_base_elevation_m',
    'final_refractor_elevation_m',
    'weathering_correction_ms',
    'elevation_correction_ms',
    'source_field_shift_ms',
    'receiver_field_shift_ms',
    'field_correction_ms',
    'manual_static_shift_ms',
    'manual_static_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'source_total_with_field_shift_ms',
    'receiver_total_with_field_shift_ms',
    'static_status',
    'solution_status',
)

# Keep the original Phase 2 cell columns and add self-describing aliases used
# by downstream QC checks; existing artifact names and column meanings remain.
_REFRACTOR_VELOCITY_CELL_COLUMNS = (
    'cell_id',
    'ix',
    'iy',
    'cell_ix',
    'cell_iy',
    'coordinate_mode',
    'x_min_m',
    'x_max_m',
    'y_min_m',
    'y_max_m',
    'x_center_m',
    'y_center_m',
    'cell_center_x_m',
    'cell_center_y_m',
    'cell_center_inline_m',
    'cell_center_crossline_m',
    'active',
    'n_observations',
    'n_used_observations',
    'n_rejected_observations',
    'n_sources',
    'n_receivers',
    'cell_velocity_layer_kind',
    'cell_velocity_component',
    'velocity_m_s',
    'v2_m_s',
    'slowness_s_per_m',
    'initial_velocity_m_s',
    'initial_v2_m_s',
    'velocity_update_from_initial_m_s',
    'v2_update_from_initial_m_s',
    'velocity_status',
    'status_reason',
    'residual_rms_ms',
    'residual_mad_ms',
    'residual_mean_ms',
    'residual_p95_abs_ms',
    'smoothing_enabled',
    'smoothing_weight',
    'smoothing_neighbor_count',
)

_GRID_MAP_QC_COLUMNS = (
    'layer_kind',
    'cell_ix',
    'cell_iy',
    'cell_center_x_m',
    'cell_center_y_m',
    'cell_center_inline_m',
    'cell_center_crossline_m',
    'velocity_m_s',
    'initial_velocity_m_s',
    'velocity_update_from_initial_m_s',
    'slowness_s_per_m',
    'n_observations',
    'n_sources',
    'n_receivers',
    'residual_rms_ms',
    'residual_mad_ms',
    'status',
    'status_reason',
)

_CELL_SOLVER_HISTORY_COLUMNS = (
    'iteration',
    'stage',
    'n_candidate_observations',
    'n_used_observations',
    'n_rejected_observations',
    'n_active_cells',
    'n_low_fold_cells',
    'n_empty_cells',
    'residual_rms_ms',
    'residual_mad_ms',
    'max_abs_residual_ms',
    'median_velocity_m_s',
    'median_v2_m_s',
    'min_velocity_m_s',
    'min_v2_m_s',
    'max_velocity_m_s',
    'max_v2_m_s',
    'max_abs_velocity_update_m_s',
    'max_abs_v2_update_m_s',
    'smoothing_weight',
    'damping_weight',
    'robust_threshold',
    'converged',
    'convergence_reason',
)


class RefractionStaticArtifactError(ValueError):
    """Raised when final refraction static artifacts cannot be written."""


@dataclass(frozen=True)
class _ValidatedResult:
    result: RefractionDatumStaticsResult
    n_traces: int
    n_nodes: int
    n_source_endpoints: int
    n_receiver_endpoints: int
    n_rows: int


@dataclass(frozen=True)
class RefractionCellSolverHistoryRow:
    iteration: int
    stage: str
    n_candidate_observations: int
    n_used_observations: int
    n_rejected_observations: int
    n_active_cells: int
    n_low_fold_cells: int
    n_empty_cells: int
    residual_rms_ms: float | None
    residual_mad_ms: float | None
    max_abs_residual_ms: float | None
    median_velocity_m_s: float | None
    median_v2_m_s: float | None
    min_velocity_m_s: float | None
    min_v2_m_s: float | None
    max_velocity_m_s: float | None
    max_v2_m_s: float | None
    max_abs_velocity_update_m_s: float | None
    max_abs_v2_update_m_s: float | None
    smoothing_weight: float
    damping_weight: float
    robust_threshold: float
    converged: bool
    convergence_reason: str


__all__ = [
    'ARTIFACT_VERSION',
    'FIRST_BREAK_FIT_QC_RESIDUAL_SIGN',
    'FIRST_BREAK_RESIDUALS_CSV_NAME',
    'FIRST_BREAK_TIME_EXPORT_FORMAT_NAME',
    'FIRST_BREAK_TIME_EXPORT_FORMAT_VERSION',
    'FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION',
    'LINE_PROFILE_QC_SCHEMA_VERSION',
    'METHOD',
    'NEAR_SURFACE_MODEL_CSV_NAME',
    'NEGATIVE_SHIFT_DESCRIPTION',
    'POSITIVE_SHIFT_DESCRIPTION',
    'RECEIVER_STATIC_TABLE_CSV_NAME',
    'REDUCED_TIME_QC_FORMULA',
    'REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME',
    'REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME',
    'REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME',
    'REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME',
    'REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME',
    'REFRACTION_GRID_MAP_QC_CSV_NAME',
    'REFRACTION_GRID_MAP_QC_JSON_NAME',
    'REFRACTION_GRID_MAP_QC_NPZ_NAME',
    'REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME',
    'REFRACTION_LINE_PROFILE_QC_JSON_NAME',
    'REFRACTION_LINE_PROFILE_QC_NPZ_NAME',
    'REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME',
    'REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME',
    'REFRACTION_PICK_MAP_QC_COMPLETED_CACHE_DIR_NAME',
    'REFRACTION_REDUCED_TIME_QC_CSV_NAME',
    'REFRACTION_REDUCED_TIME_QC_JSON_NAME',
    'REFRACTION_REDUCED_TIME_QC_NPZ_NAME',
    'REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME',
    'REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME',
    'REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME',
    'REFRACTION_STATIC_ARTIFACTS_JSON_NAME',
    'REFRACTION_STATIC_COMPONENTS_CSV_NAME',
    'REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME',
    'REFRACTION_STATIC_COMPONENT_QC_JSON_NAME',
    'REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME',
    'REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME',
    'REFRACTION_STATIC_FAILURE_DIAGNOSTICS_JSON_NAME',
    'REFRACTION_STATIC_HISTORY_JSON_NAME',
    'REFRACTION_STATIC_QC_JSON_NAME',
    'REFRACTION_STATIC_REQUEST_JSON_NAME',
    'REFRACTION_STATIC_SOLUTION_NPZ_NAME',
    'REFRACTION_STATICS_CSV_NAME',
    'REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME',
    'REFRACTION_V3_CELL_SOLVER_HISTORY_CSV_NAME',
    'REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME',
    'REFRACTION_V3_REFRACTOR_VELOCITY_GRID_NPZ_NAME',
    'REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME',
    'REFRACTION_VSUB_CELL_SOLVER_HISTORY_CSV_NAME',
    'REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME',
    'REFRACTION_VSUB_REFRACTOR_VELOCITY_GRID_NPZ_NAME',
    'REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME',
    'SIGN_CONVENTION',
    'SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME',
    'SOURCE_STATIC_TABLE_CSV_NAME',
    'STATIC_COMPONENT',
    'TIME_TERM_SPREADSHEET_FORMAT_NAME',
    'TIME_TERM_SPREADSHEET_FORMAT_VERSION',
    'TIME_TERM_SPREADSHEET_SCHEMA_VERSION',
    'UPLOADED_REFRACTION_PICKS_NPZ_NAME',
    'WORKFLOW',
    'RefractionCellSolverHistoryRow',
    'RefractionStaticArtifactError',
]
