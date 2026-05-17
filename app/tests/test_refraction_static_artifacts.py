from __future__ import annotations

import csv
from dataclasses import replace
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import RefractionStaticApplyRequest
import app.services.refraction_static_artifacts as artifact_module
from app.services.refraction_static_artifacts import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
    REFRACTION_GRID_MAP_QC_CSV_NAME,
    REFRACTION_GRID_MAP_QC_JSON_NAME,
    REFRACTION_GRID_MAP_QC_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_JSON_NAME,
    REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
    REFRACTION_REDUCED_TIME_QC_CSV_NAME,
    REFRACTION_REDUCED_TIME_QC_JSON_NAME,
    REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
    FIRST_BREAK_FIT_QC_RESIDUAL_SIGN,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_STATICS_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_JSON_NAME,
    REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME,
    REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_HISTORY_JSON_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    REFRACTION_V3_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_VSUB_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_V1_ESTIMATES_CSV_NAME,
    REFRACTION_V1_QC_JSON_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    RefractionStaticArtifactError,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    TIME_TERM_SPREADSHEET_FORMAT_NAME,
    TIME_TERM_SPREADSHEET_FORMAT_VERSION,
    TIME_TERM_SPREADSHEET_SCHEMA_VERSION,
    build_refraction_reduced_time_qc_arrays,
    write_refraction_static_solution_npz,
    write_refraction_static_artifacts,
)
from app.services.refraction_static_source_depth import (
    REFRACTION_SOURCE_DEPTH_QC_JSON_NAME,
    REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
)
from app.services.refraction_static_types import RefractionLayerSolveResult
from app.services.refraction_static_uphole import (
    REFRACTION_UPHOLE_QC_JSON_NAME,
    REFRACTION_UPHOLE_SOURCES_CSV_NAME,
)
from app.tests._refraction_static_artifact_helpers import (
    _estimated_v1_request,
    _request,
    _resolved_estimated_v1,
    _result,
    _result_with_weathering_velocity,
)


REQUIRED_TRACE_ARRAYS = {
    'sorted_trace_index',
    'valid_observation_mask_sorted',
    'used_observation_mask_sorted',
    'trace_static_valid_mask_sorted',
    'source_node_id_sorted',
    'receiver_node_id_sorted',
    'source_surface_elevation_m_sorted',
    'receiver_surface_elevation_m_sorted',
    'source_floating_datum_elevation_m_sorted',
    'receiver_floating_datum_elevation_m_sorted',
    'source_weathering_thickness_m_sorted',
    'receiver_weathering_thickness_m_sorted',
    'source_refractor_elevation_m_sorted',
    'receiver_refractor_elevation_m_sorted',
    'source_half_intercept_time_s_sorted',
    'receiver_half_intercept_time_s_sorted',
    'weathering_replacement_trace_shift_s_sorted',
    'floating_datum_elevation_shift_s_sorted',
    'flat_datum_shift_s_sorted',
    'refraction_trace_shift_s_sorted',
    'estimated_first_break_time_s_sorted',
    'first_break_residual_s_sorted',
    'trace_static_status_sorted',
}

REQUIRED_NODE_ARRAYS = {
    'node_id',
    'node_x_m',
    'node_y_m',
    'node_surface_elevation_m',
    'node_floating_datum_elevation_m',
    'node_refractor_elevation_m',
    'node_weathering_thickness_m',
    'node_half_intercept_time_s',
    'node_weathering_replacement_shift_s',
    'node_t1_time_s',
    'node_sh1_weathering_thickness_m',
    'node_weathering_correction_s',
    'node_solution_status',
    'node_weathering_status',
    'node_datum_status',
    'node_pick_count',
    'node_used_pick_count',
    'node_rejected_pick_count',
    'node_residual_rms_s',
    'node_residual_mad_s',
}

REQUIRED_ROW_ARRAYS = {
    'row_trace_index_sorted',
    'row_source_node_id',
    'row_receiver_node_id',
    'row_distance_m',
    'observed_pick_time_s',
    'modeled_pick_time_s',
    'residual_time_s',
    'used_row_mask',
    'rejected_by_robust_mask',
}

EXPECTED_FILENAMES = {
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATICS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
    REFRACTION_REDUCED_TIME_QC_CSV_NAME,
    REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
    REFRACTION_REDUCED_TIME_QC_JSON_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME,
    REFRACTION_STATIC_COMPONENT_QC_JSON_NAME,
    REFRACTION_STATIC_HISTORY_JSON_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_JSON_NAME,
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
}

CELL_VELOCITY_FILENAMES = {
    REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
}

V3_CELL_VELOCITY_FILENAMES = {
    REFRACTION_V3_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME,
}

VSUB_CELL_VELOCITY_FILENAMES = {
    REFRACTION_VSUB_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME,
}

GRID_MAP_QC_FILENAMES = {
    REFRACTION_GRID_MAP_QC_CSV_NAME,
    REFRACTION_GRID_MAP_QC_NPZ_NAME,
    REFRACTION_GRID_MAP_QC_JSON_NAME,
}

GRID_MAP_QC_REQUIRED_COLUMNS = {
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
}

M6_QC_ARTIFACT_DESCRIPTIONS = {
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME: (
        'Viewer-ready observed-modeled first-break fit QC table'
    ),
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME: (
        'Machine-readable observed-modeled first-break fit QC arrays'
    ),
    REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME: (
        'Observed-modeled first-break fit QC schema and summary'
    ),
    REFRACTION_REDUCED_TIME_QC_CSV_NAME: (
        'Reduced-time first-break QC table for LMO displays'
    ),
    REFRACTION_REDUCED_TIME_QC_NPZ_NAME: (
        'Machine-readable reduced-time first-break QC arrays'
    ),
    REFRACTION_REDUCED_TIME_QC_JSON_NAME: (
        'Reduced-time first-break QC schema and summary'
    ),
    REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME: (
        'Source endpoint line-profile QC rows sorted by inline distance'
    ),
    REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME: (
        'Receiver endpoint line-profile QC rows sorted by inline distance'
    ),
    REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME: (
        'Combined source/receiver line-profile QC rows'
    ),
    REFRACTION_LINE_PROFILE_QC_NPZ_NAME: (
        'Machine-readable source/receiver line-profile QC arrays'
    ),
    REFRACTION_LINE_PROFILE_QC_JSON_NAME: (
        'Line-profile QC schema, availability, and summary'
    ),
    REFRACTION_GRID_MAP_QC_CSV_NAME: (
        'Viewer-ready refraction cell velocity grid map QC rows'
    ),
    REFRACTION_GRID_MAP_QC_NPZ_NAME: (
        'Machine-readable refraction cell velocity grid map QC arrays'
    ),
    REFRACTION_GRID_MAP_QC_JSON_NAME: 'Refraction cell velocity grid map QC summary',
    REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME: (
        'Trace-level static component waterfall QC table'
    ),
    REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME: (
        'Endpoint-level static component waterfall QC table'
    ),
    REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME: (
        'Machine-readable static component waterfall QC arrays'
    ),
    REFRACTION_STATIC_COMPONENT_QC_JSON_NAME: (
        'Static component waterfall QC schema and summary'
    ),
}

UPSTREAM_V1_ARTIFACT_NAMES = (
    REFRACTION_V1_QC_JSON_NAME,
    REFRACTION_V1_ESTIMATES_CSV_NAME,
)


def test_refraction_static_artifacts_public_all_snapshot() -> None:
    assert tuple(artifact_module.__all__) == (
        'FIRST_BREAK_RESIDUALS_CSV_NAME',
        'FIRST_BREAK_FIT_QC_RESIDUAL_SIGN',
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
        'REFRACTION_REDUCED_TIME_QC_CSV_NAME',
        'REFRACTION_REDUCED_TIME_QC_JSON_NAME',
        'REFRACTION_REDUCED_TIME_QC_NPZ_NAME',
        'NEAR_SURFACE_MODEL_CSV_NAME',
        'REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME',
        'REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME',
        'REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME',
        'REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME',
        'REFRACTION_V3_CELL_SOLVER_HISTORY_CSV_NAME',
        'REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME',
        'REFRACTION_V3_REFRACTOR_VELOCITY_GRID_NPZ_NAME',
        'REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME',
        'REFRACTION_VSUB_CELL_SOLVER_HISTORY_CSV_NAME',
        'REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME',
        'REFRACTION_VSUB_REFRACTOR_VELOCITY_GRID_NPZ_NAME',
        'REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME',
        'REFRACTION_STATICS_CSV_NAME',
        'REFRACTION_STATIC_ARTIFACTS_JSON_NAME',
        'REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME',
        'REFRACTION_STATIC_COMPONENT_QC_JSON_NAME',
        'REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME',
        'REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME',
        'REFRACTION_STATIC_COMPONENTS_CSV_NAME',
        'REFRACTION_STATIC_FAILURE_DIAGNOSTICS_JSON_NAME',
        'REFRACTION_STATIC_HISTORY_JSON_NAME',
        'REFRACTION_STATIC_REQUEST_JSON_NAME',
        'REFRACTION_STATIC_REGISTERED_ARTIFACT_NAMES',
        'REFRACTION_STATIC_QC_JSON_NAME',
        'REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME',
        'REFRACTION_STATIC_SOLUTION_NPZ_NAME',
        'REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME',
        'REFRACTION_V1_ESTIMATES_CSV_NAME',
        'REFRACTION_V1_QC_JSON_NAME',
        'RECEIVER_STATIC_TABLE_CSV_NAME',
        'SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME',
        'SOURCE_STATIC_TABLE_CSV_NAME',
        'UPLOADED_REFRACTION_PICKS_NPZ_NAME',
        'RefractionCellSolverHistoryRow',
        'RefractionStaticArtifactError',
        'RefractionStaticArtifactSet',
        'build_refraction_cell_solver_history_rows',
        'build_refraction_first_break_fit_qc_arrays',
        'build_refraction_first_break_fit_qc_payload',
        'build_refraction_grid_map_qc_arrays',
        'build_refraction_grid_map_qc_payload',
        'build_refraction_line_profile_qc_arrays',
        'build_refraction_line_profile_qc_payload',
        'build_refraction_reduced_time_qc_arrays',
        'build_refraction_reduced_time_qc_payload',
        'build_refraction_static_component_qc_arrays',
        'build_refraction_static_component_qc_payload',
        'build_refraction_refractor_velocity_grid_arrays',
        'build_refraction_refractor_velocity_qc_payload',
        'build_refraction_static_history_payload',
        'build_refraction_static_qc_payload',
        'build_refraction_static_solution_arrays',
        'build_source_receiver_static_table_arrays',
        'refraction_static_double_application_qc',
        'refraction_static_trace_shift_component_names',
        'static_history_double_application_qc',
        'write_first_break_residuals_csv',
        'write_near_surface_model_csv',
        'write_refraction_first_break_fit_qc_csv',
        'write_refraction_first_break_fit_qc_json',
        'write_refraction_first_break_fit_qc_npz',
        'write_refraction_first_break_time_export_csv',
        'write_refraction_grid_map_qc_csv',
        'write_refraction_grid_map_qc_json',
        'write_refraction_grid_map_qc_npz',
        'write_refraction_line_profile_qc_artifacts',
        'write_refraction_reduced_time_qc_csv',
        'write_refraction_reduced_time_qc_json',
        'write_refraction_reduced_time_qc_npz',
        'write_refraction_cell_solver_history_csv',
        'write_refraction_refractor_velocity_cells_csv',
        'write_refraction_refractor_velocity_grid_npz',
        'write_refraction_refractor_velocity_qc_json',
        'write_refraction_static_artifacts',
        'write_refraction_static_component_qc_artifacts',
        'write_refraction_static_components_csv',
        'write_refraction_static_history_json',
        'write_refraction_static_qc_json',
        'write_refraction_static_solution_npz',
        'write_refraction_statics_csv',
        'write_refraction_time_term_spreadsheet_csv',
        'write_refraction_time_term_spreadsheet_csv_from_static_tables',
        'write_receiver_static_table_csv',
        'write_source_receiver_static_table_npz',
        'write_source_static_table_csv',
    )
    assert all(hasattr(artifact_module, name) for name in artifact_module.__all__)


@pytest.mark.parametrize(
    'module_name',
    [
        'app.services.job_artifact_refs',
        'app.services.refraction_static_apply_trace_store',
        'app.services.refraction_static_export_service',
        'app.services.refraction_static_gather_preview',
        'app.services.refraction_static_multilayer_service',
        'app.services.refraction_static_qc_bundle',
        'app.services.refraction_static_qc_drilldown',
        'app.services.refraction_static_service',
        'app.services.refraction_static_table_apply_service',
        'app.api.routers.statics',
    ],
)
def test_refraction_static_artifact_consumer_import_smoke(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_refraction_static_artifact_name_snapshot() -> None:
    names = {
        'solution_npz': artifact_module.REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        'qc_json': artifact_module.REFRACTION_STATIC_QC_JSON_NAME,
        'manifest_json': artifact_module.REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
        'request_json': artifact_module.REFRACTION_STATIC_REQUEST_JSON_NAME,
        'failure_diagnostics_json': (
            artifact_module.REFRACTION_STATIC_FAILURE_DIAGNOSTICS_JSON_NAME
        ),
        'trace_statics_csv': artifact_module.REFRACTION_STATICS_CSV_NAME,
        'near_surface_model_csv': artifact_module.NEAR_SURFACE_MODEL_CSV_NAME,
        'first_break_residuals_csv': artifact_module.FIRST_BREAK_RESIDUALS_CSV_NAME,
        'first_break_time_export_csv': (
            artifact_module.REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME
        ),
        'source_static_table_csv': artifact_module.SOURCE_STATIC_TABLE_CSV_NAME,
        'receiver_static_table_csv': artifact_module.RECEIVER_STATIC_TABLE_CSV_NAME,
        'source_receiver_static_table_npz': (
            artifact_module.SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME
        ),
        'first_break_fit_qc_csv': (
            artifact_module.REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME
        ),
        'first_break_fit_qc_npz': (
            artifact_module.REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME
        ),
        'first_break_fit_qc_json': (
            artifact_module.REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME
        ),
        'reduced_time_qc_csv': artifact_module.REFRACTION_REDUCED_TIME_QC_CSV_NAME,
        'reduced_time_qc_npz': artifact_module.REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
        'reduced_time_qc_json': artifact_module.REFRACTION_REDUCED_TIME_QC_JSON_NAME,
        'line_profile_qc_source_csv': (
            artifact_module.REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME
        ),
        'line_profile_qc_receiver_csv': (
            artifact_module.REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME
        ),
        'line_profile_qc_combined_csv': (
            artifact_module.REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME
        ),
        'line_profile_qc_npz': artifact_module.REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
        'line_profile_qc_json': artifact_module.REFRACTION_LINE_PROFILE_QC_JSON_NAME,
        'grid_map_qc_csv': artifact_module.REFRACTION_GRID_MAP_QC_CSV_NAME,
        'grid_map_qc_npz': artifact_module.REFRACTION_GRID_MAP_QC_NPZ_NAME,
        'grid_map_qc_json': artifact_module.REFRACTION_GRID_MAP_QC_JSON_NAME,
        'v2_cells_csv': artifact_module.REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
        'v2_grid_npz': artifact_module.REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
        'v2_qc_json': artifact_module.REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
        'v2_solver_history_csv': artifact_module.REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
        'v3_cells_csv': artifact_module.REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
        'v3_grid_npz': artifact_module.REFRACTION_V3_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
        'v3_qc_json': artifact_module.REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME,
        'v3_solver_history_csv': artifact_module.REFRACTION_V3_CELL_SOLVER_HISTORY_CSV_NAME,
        'vsub_cells_csv': (
            artifact_module.REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME
        ),
        'vsub_grid_npz': artifact_module.REFRACTION_VSUB_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
        'vsub_qc_json': artifact_module.REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME,
        'vsub_solver_history_csv': (
            artifact_module.REFRACTION_VSUB_CELL_SOLVER_HISTORY_CSV_NAME
        ),
    }

    assert names == {
        'solution_npz': 'refraction_static_solution.npz',
        'qc_json': 'refraction_static_qc.json',
        'manifest_json': 'refraction_static_artifacts.json',
        'request_json': 'refraction_static_request.json',
        'failure_diagnostics_json': 'failure_diagnostics.json',
        'trace_statics_csv': 'refraction_statics.csv',
        'near_surface_model_csv': 'near_surface_model.csv',
        'first_break_residuals_csv': 'first_break_residuals.csv',
        'first_break_time_export_csv': 'refraction_first_break_time_export.csv',
        'source_static_table_csv': 'source_static_table.csv',
        'receiver_static_table_csv': 'receiver_static_table.csv',
        'source_receiver_static_table_npz': 'source_receiver_static_table.npz',
        'first_break_fit_qc_csv': 'refraction_first_break_fit_qc.csv',
        'first_break_fit_qc_npz': 'refraction_first_break_fit_qc.npz',
        'first_break_fit_qc_json': 'refraction_first_break_fit_qc.json',
        'reduced_time_qc_csv': 'refraction_reduced_time_qc.csv',
        'reduced_time_qc_npz': 'refraction_reduced_time_qc.npz',
        'reduced_time_qc_json': 'refraction_reduced_time_qc.json',
        'line_profile_qc_source_csv': 'refraction_line_profile_qc_source.csv',
        'line_profile_qc_receiver_csv': 'refraction_line_profile_qc_receiver.csv',
        'line_profile_qc_combined_csv': 'refraction_line_profile_qc_combined.csv',
        'line_profile_qc_npz': 'refraction_line_profile_qc.npz',
        'line_profile_qc_json': 'refraction_line_profile_qc.json',
        'grid_map_qc_csv': 'refraction_grid_map_qc.csv',
        'grid_map_qc_npz': 'refraction_grid_map_qc.npz',
        'grid_map_qc_json': 'refraction_grid_map_qc.json',
        'v2_cells_csv': 'refraction_refractor_velocity_cells.csv',
        'v2_grid_npz': 'refraction_refractor_velocity_grid.npz',
        'v2_qc_json': 'refraction_refractor_velocity_qc.json',
        'v2_solver_history_csv': 'refraction_cell_solver_history.csv',
        'v3_cells_csv': 'refraction_v3_refractor_velocity_cells.csv',
        'v3_grid_npz': 'refraction_v3_refractor_velocity_grid.npz',
        'v3_qc_json': 'refraction_v3_refractor_velocity_qc.json',
        'v3_solver_history_csv': 'refraction_v3_cell_solver_history.csv',
        'vsub_cells_csv': 'refraction_vsub_refractor_velocity_cells.csv',
        'vsub_grid_npz': 'refraction_vsub_refractor_velocity_grid.npz',
        'vsub_qc_json': 'refraction_vsub_refractor_velocity_qc.json',
        'vsub_solver_history_csv': 'refraction_vsub_cell_solver_history.csv',
    }


def test_write_refraction_static_artifacts_npz_schema(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    assert {path.name for path in tmp_path.iterdir()} == EXPECTED_FILENAMES
    assert paths.artifact_names == (
        REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        REFRACTION_STATIC_QC_JSON_NAME,
        REFRACTION_STATIC_HISTORY_JSON_NAME,
        REFRACTION_STATICS_CSV_NAME,
        NEAR_SURFACE_MODEL_CSV_NAME,
        FIRST_BREAK_RESIDUALS_CSV_NAME,
        REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
        REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
        REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
        REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
        REFRACTION_REDUCED_TIME_QC_CSV_NAME,
        REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
        REFRACTION_REDUCED_TIME_QC_JSON_NAME,
        REFRACTION_STATIC_COMPONENTS_CSV_NAME,
        REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
        REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
        REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME,
        REFRACTION_STATIC_COMPONENT_QC_JSON_NAME,
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
        REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
        REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
        REFRACTION_LINE_PROFILE_QC_JSON_NAME,
        REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    )
    with np.load(paths.solution_npz, allow_pickle=False) as data:
        assert data['artifact_version'].item() == '1.0'
        assert data['method'].item() == 'gli_variable_thickness'
        assert data['sign_convention'].item() == 'corrected(t) = raw(t - shift_s)'
        assert data['v1_mode'].item() == 'constant'
        assert data['v1_weathering_velocity_m_s'].item() == pytest.approx(800.0)
        assert data['weathering_velocity_m_s'].item() == pytest.approx(800.0)
        assert data['resolved_weathering_velocity_m_s'].item() == pytest.approx(800.0)
        assert data['bedrock_velocity_m_s'].item() == pytest.approx(2500.0)
        assert data['v2_refractor_velocity_m_s'].item() == pytest.approx(2500.0)
        assert REQUIRED_TRACE_ARRAYS.issubset(data.files)
        assert REQUIRED_NODE_ARRAYS.issubset(data.files)
        assert REQUIRED_ROW_ARRAYS.issubset(data.files)
        assert data['source_endpoint_key'].shape == (2,)
        assert data['receiver_endpoint_key'].shape == (2,)
        assert data['sorted_trace_index'].shape == (4,)
        assert data['node_id'].shape == (3,)
        assert data['row_trace_index_sorted'].shape == (3,)
        assert data['source_node_id_sorted'].dtype == np.int64
        assert data['valid_observation_mask_sorted'].dtype == bool
        assert data['source_surface_elevation_m_sorted'].dtype == np.float64
        assert data['trace_static_status_sorted'].dtype.kind == 'U'
        assert data['trace_static_status_sorted'].tolist() == [
            'ok',
            'ok',
            'not_observed',
            'ok',
        ]
        assert data['node_solution_status'].tolist() == [
            'solved',
            'solved',
            'inactive',
        ]
        assert data['node_weathering_status'].tolist() == [
            'ok',
            'zero_thickness',
            'inactive',
        ]
        assert data['node_datum_status'].tolist() == ['ok', 'ok', 'inactive']
        for key in data.files:
            assert data[key].dtype != object


def test_refraction_static_base_manifest_contract(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    manifest = json.loads(paths.manifest_json.read_text(encoding='utf-8'))
    assert tuple(
        (
            item['name'],
            item['kind'],
            item['required'],
            item['description'],
        )
        for item in manifest['artifacts']
    ) == (
        (
            'refraction_static_solution.npz',
            'npz',
            True,
            'Machine-readable final refraction statics solution',
        ),
        (
            'refraction_static_qc.json',
            'json',
            True,
            'Human-readable final refraction statics QC summary',
        ),
        (
            'refraction_static_history.json',
            'json',
            True,
            'Static-component lineage and double-application audit history',
        ),
        ('refraction_statics.csv', 'csv', True, 'Trace-level final refraction statics table'),
        ('near_surface_model.csv', 'csv', True, 'Node-level near-surface model table'),
        ('first_break_residuals.csv', 'csv', True, 'GLI first-break residual table'),
        (
            'refraction_first_break_time_export.csv',
            'csv',
            True,
            'Observation-level first-break time QC export',
        ),
        (
            'refraction_first_break_fit_qc.csv',
            'csv',
            True,
            'Viewer-ready observed-modeled first-break fit QC table',
        ),
        (
            'refraction_first_break_fit_qc.npz',
            'npz',
            True,
            'Machine-readable observed-modeled first-break fit QC arrays',
        ),
        (
            'refraction_first_break_fit_qc.json',
            'json',
            True,
            'Observed-modeled first-break fit QC schema and summary',
        ),
        (
            'refraction_reduced_time_qc.csv',
            'csv',
            True,
            'Reduced-time first-break QC table for LMO displays',
        ),
        (
            'refraction_reduced_time_qc.npz',
            'npz',
            True,
            'Machine-readable reduced-time first-break QC arrays',
        ),
        (
            'refraction_reduced_time_qc.json',
            'json',
            True,
            'Reduced-time first-break QC schema and summary',
        ),
        (
            'refraction_static_components.csv',
            'csv',
            True,
            'Source/receiver endpoint static component table',
        ),
        (
            'refraction_static_component_qc_trace.csv',
            'csv',
            True,
            'Trace-level static component waterfall QC table',
        ),
        (
            'refraction_static_component_qc_endpoint.csv',
            'csv',
            True,
            'Endpoint-level static component waterfall QC table',
        ),
        (
            'refraction_static_component_qc.npz',
            'npz',
            True,
            'Machine-readable static component waterfall QC arrays',
        ),
        (
            'refraction_static_component_qc.json',
            'json',
            True,
            'Static component waterfall QC schema and summary',
        ),
        ('source_static_table.csv', 'csv', True, 'IRAS-style source endpoint final static table'),
        (
            'receiver_static_table.csv',
            'csv',
            True,
            'IRAS-style receiver endpoint final static table',
        ),
        (
            'source_receiver_static_table.npz',
            'npz',
            True,
            'Machine-readable source/receiver endpoint static tables',
        ),
        (
            'refraction_line_profile_qc_source.csv',
            'csv',
            True,
            'Source endpoint line-profile QC rows sorted by inline distance',
        ),
        (
            'refraction_line_profile_qc_receiver.csv',
            'csv',
            True,
            'Receiver endpoint line-profile QC rows sorted by inline distance',
        ),
        (
            'refraction_line_profile_qc_combined.csv',
            'csv',
            True,
            'Combined source/receiver line-profile QC rows',
        ),
        (
            'refraction_line_profile_qc.npz',
            'npz',
            True,
            'Machine-readable source/receiver line-profile QC arrays',
        ),
        (
            'refraction_line_profile_qc.json',
            'json',
            True,
            'Line-profile QC schema, availability, and summary',
        ),
        (
            'refraction_time_term_spreadsheet.csv',
            'csv',
            True,
            'Spreadsheet endpoint time terms, layers, statics, and statuses',
        ),
    )


def test_refraction_static_json_outputs_are_strict_and_deterministic(
    tmp_path: Path,
) -> None:
    write_refraction_static_artifacts(
        result=_solve_cell_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    for path in sorted(tmp_path.glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        json.dumps(payload, allow_nan=False)
        assert path.read_text(encoding='utf-8') == (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + '\n'
        )


def test_refraction_static_representative_csv_header_contract(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )
    first_break_fit_header = (
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
    line_profile_header = (
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
    static_table_tail = (
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

    expected_headers = {
        paths.refraction_first_break_fit_qc_csv.name: first_break_fit_header,
        paths.refraction_reduced_time_qc_csv.name: (
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
        ),
        paths.refraction_line_profile_qc_combined_csv.name: line_profile_header,
        paths.source_static_table_csv.name: (
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
            'source_depth_m',
            'source_depth_shift_ms',
            'source_depth_status',
            'uphole_time_ms',
            'uphole_shift_ms',
            'uphole_status',
            'manual_static_shift_ms',
            'manual_static_status',
            'source_field_shift_ms',
            'source_field_status',
            'source_field_static_status',
            'source_total_with_field_shift_ms',
        )
        + static_table_tail,
        paths.receiver_static_table_csv.name: (
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
            'manual_static_shift_ms',
            'manual_static_status',
            'receiver_field_shift_ms',
            'receiver_field_status',
            'receiver_field_static_status',
            'receiver_total_with_field_shift_ms',
        )
        + static_table_tail,
        REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME: (
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
        ),
        REFRACTION_GRID_MAP_QC_CSV_NAME: (
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
        ),
    }
    for artifact_name, expected_header in expected_headers.items():
        _rows, fieldnames = _read_csv_with_fieldnames(tmp_path / artifact_name)
        assert tuple(fieldnames) == expected_header


def test_refraction_static_representative_npz_key_contract(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )
    first_break_fit_keys = (
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
    line_profile_keys = (
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
    expected_npz_keys = {
        paths.refraction_first_break_fit_qc_npz.name: first_break_fit_keys,
        paths.refraction_reduced_time_qc_npz.name: (
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
            'reduction_velocity_mode',
        ),
        paths.refraction_line_profile_qc_npz.name: line_profile_keys,
        paths.source_receiver_static_table_npz.name: (
            'sign_convention',
            'source_endpoint_key',
            'source_id',
            'source_node_id',
            'source_v2_cell_id',
            'source_v2_status',
            'source_x_m',
            'source_y_m',
            'source_surface_elevation_m',
            'source_t1_s',
            'source_v1_m_s',
            'source_v2_m_s',
            'source_sh1_m',
            'source_total_weathering_thickness_m',
            'source_weathering_correction_s',
            'source_elevation_correction_s',
            'source_total_static_s',
            'source_total_applied_shift_s',
            'source_static_status',
            'receiver_endpoint_key',
            'receiver_id',
            'receiver_node_id',
            'receiver_v2_cell_id',
            'receiver_v2_status',
            'receiver_x_m',
            'receiver_y_m',
            'receiver_surface_elevation_m',
            'receiver_t1_s',
            'receiver_v1_m_s',
            'receiver_v2_m_s',
            'receiver_sh1_m',
            'receiver_total_weathering_thickness_m',
            'receiver_weathering_correction_s',
            'receiver_elevation_correction_s',
            'receiver_total_static_s',
            'receiver_total_applied_shift_s',
            'receiver_static_status',
            'source_depth_m',
            'source_depth_shift_s',
            'source_depth_status',
            'source_uphole_time_s',
            'source_uphole_shift_s',
            'source_uphole_status',
            'source_manual_static_shift_s',
            'source_manual_static_status',
            'receiver_manual_static_shift_s',
            'receiver_manual_static_status',
            'source_field_shift_s',
            'source_field_static_status',
            'source_total_with_field_shift_s',
            'receiver_field_shift_s',
            'receiver_field_static_status',
            'receiver_total_with_field_shift_s',
        ),
        REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME: (
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
            'active_cell_mask',
            'n_observations_per_cell',
            'n_used_observations_per_cell',
            'n_rejected_observations_per_cell',
            'n_sources_per_cell',
            'n_receivers_per_cell',
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
        ),
        REFRACTION_GRID_MAP_QC_NPZ_NAME: (
            'layer_kind',
            'cell_id',
            'cell_ix',
            'cell_iy',
            'cell_center_x_m',
            'cell_center_y_m',
            'cell_center_inline_m',
            'cell_center_crossline_m',
            'x_min_m',
            'x_max_m',
            'y_min_m',
            'y_max_m',
            'velocity_m_s',
            'initial_velocity_m_s',
            'velocity_update_from_initial_m_s',
            'slowness_s_per_m',
            'n_observations',
            'n_used_observations',
            'n_rejected_observations',
            'n_sources',
            'n_receivers',
            'residual_rms_ms',
            'residual_mad_ms',
            'status',
            'status_reason',
            'active_cell_mask',
            'coordinate_mode',
            'cell_velocity_component',
            'artifact_version',
            'artifact_kind',
            'global_velocity_layer_behavior',
            'number_of_cell_x',
            'number_of_cell_y',
            'size_of_cell_x_m',
            'size_of_cell_y_m',
            'x_coordinate_origin_m',
            'y_coordinate_origin_m',
        ),
    }
    for artifact_name, expected_keys in expected_npz_keys.items():
        with np.load(tmp_path / artifact_name, allow_pickle=False) as data:
            assert tuple(data.files) == expected_keys
            assert all(data[key].dtype != object for key in data.files)

    with np.load(paths.solution_npz, allow_pickle=False) as data:
        assert all(data[key].dtype != object for key in data.files)
        assert data.files[:24] == [
            'artifact_version',
            'method',
            'bedrock_velocity_mode',
            'datum_mode',
            'floating_datum_mode',
            'sign_convention',
            'n_traces',
            'n_nodes',
            'n_source_endpoints',
            'n_receiver_endpoints',
            'n_valid_observations',
            'n_used_observations',
            'n_rejected_by_robust',
            'v1_mode',
            'v1_weathering_velocity_m_s',
            'weathering_velocity_m_s',
            'resolved_weathering_velocity_m_s',
            'bedrock_velocity_m_s',
            'v2_refractor_velocity_m_s',
            'bedrock_slowness_s_per_m',
            'replacement_slowness_delta_s_per_m',
            'flat_datum_elevation_m',
            'max_abs_shift_ms',
            'sorted_trace_index',
        ]


def test_time_term_spreadsheet_columns_are_stable(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
        source_job_id='refraction-job-505',
    )

    with paths.refraction_time_term_spreadsheet_csv.open(
        encoding='utf-8',
        newline='',
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert paths.refraction_time_term_spreadsheet_csv.name == (
        REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME
    )
    assert tuple(reader.fieldnames or ()) == (
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
    assert rows[0]['schema_version'] == str(TIME_TERM_SPREADSHEET_SCHEMA_VERSION)
    assert rows[0]['format_name'] == TIME_TERM_SPREADSHEET_FORMAT_NAME
    assert rows[0]['format_version'] == str(TIME_TERM_SPREADSHEET_FORMAT_VERSION)
    assert rows[0]['source_job_id'] == 'refraction-job-505'


def test_time_term_spreadsheet_contains_one_row_per_endpoint(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_time_term_spreadsheet_csv)

    assert [row['endpoint_kind'] for row in rows] == [
        'source',
        'source',
        'receiver',
        'receiver',
    ]
    assert [row['endpoint_key'] for row in rows] == ['s0', 's1', 'r0', 'r1']
    assert rows[0]['station_id'] == '100'
    assert rows[2]['station_id'] == '200'
    assert rows[0]['elevation_m'] == '100.000'
    assert rows[0]['t1_ms'] == '10.000000'
    assert rows[0]['t2_ms'] == ''
    assert rows[0]['t3_ms'] == ''
    assert rows[0]['sh1_m'] == '10.000'
    assert rows[0]['sh2_m'] == ''
    assert rows[0]['sh3_m'] == ''
    assert rows[0]['sign_convention'] == 'corrected(t) = raw(t - shift_s)'


def test_time_term_spreadsheet_units_are_explicit(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    _rows, fieldnames = _read_csv_with_fieldnames(
        paths.refraction_time_term_spreadsheet_csv
    )

    unit_columns = {
        name
        for name in fieldnames
        if name
        not in {
            'schema_version',
            'format_name',
            'format_version',
            'source_job_id',
            'endpoint_kind',
            'endpoint_key',
            'endpoint_id',
            'station_id',
            'node_id',
            'pick_count',
            'used_pick_count',
            'pick_count_by_layer',
            'used_pick_count_by_layer',
            'solution_status',
            'weathering_status',
            'datum_status',
            'source_depth_status',
            'uphole_status',
            'manual_static_status',
            'field_static_status',
            'static_status',
            'sign_convention',
        }
    }
    assert unit_columns
    assert all(
        name.endswith(('_ms', '_m', '_m_s')) or name in {'x_m', 'y_m'}
        for name in unit_columns
    )


def test_refraction_static_solution_npz_contains_v1_aliases(tmp_path: Path) -> None:
    path = tmp_path / REFRACTION_STATIC_SOLUTION_NPZ_NAME
    write_refraction_static_solution_npz(
        result=_result_with_weathering_velocity(812.5),
        req=_estimated_v1_request(),
        path=path,
        resolved_first_layer=_resolved_estimated_v1(),
    )

    with np.load(path, allow_pickle=False) as data:
        assert data['v1_mode'].item() == 'estimate_direct_arrival'
        assert data['v1_weathering_velocity_m_s'].item() == pytest.approx(812.5)
        assert data['resolved_weathering_velocity_m_s'].item() == pytest.approx(812.5)


def test_write_refraction_static_artifacts_qc_json(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    payload = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    assert {
        'artifact_version',
        'method',
        'workflow',
        'static_component',
        'sign_convention',
        'request',
        'velocity',
        'datum',
        'observations',
        'nodes',
        'endpoints',
        'first_break_fit',
        'statics',
        'status_counts',
        'artifacts',
        'warnings',
    }.issubset(payload)
    assert payload['request'] == {
        'file_id': 'raw-file-id',
        'key1_byte': 189,
        'key2_byte': 193,
        'pick_source_kind': 'batch_predicted_npz',
        'model_method': 'gli_variable_thickness',
        'apply_mode': 'refraction_from_raw',
        'register_corrected_file': False,
    }
    assert payload['velocity']['bedrock_velocity_status'] == 'solved'
    assert payload['velocity']['v1_mode'] == 'constant'
    assert payload['velocity']['resolved_weathering_velocity_m_s'] == pytest.approx(
        800.0
    )
    assert payload['observations']['n_valid_observations'] == 3
    assert payload['observations']['n_used_observations'] == 2
    assert payload['status_counts']['node_solution_status']['solved'] == 2
    assert payload['status_counts']['node_weathering_status']['zero_thickness'] == 1
    assert payload['status_counts']['trace_static_status']['ok'] == 3
    assert payload['status_counts']['node_datum_status']['ok'] == 2
    assert payload['first_break_fit']['residual_rms_ms'] == pytest.approx(1.0)
    assert len(payload['artifacts']) == len(
        EXPECTED_FILENAMES - {REFRACTION_STATIC_ARTIFACTS_JSON_NAME}
    )
    artifact_names = {item['name'] for item in payload['artifacts']}
    assert REFRACTION_V1_QC_JSON_NAME not in artifact_names
    assert REFRACTION_V1_ESTIMATES_CSV_NAME not in artifact_names
    json.dumps(payload, allow_nan=False)
    assert not _contains_absolute_path(payload)


def test_refraction_static_qc_contains_v1_mode() -> None:
    payload = artifact_module.build_refraction_static_qc_payload(
        result=_result_with_weathering_velocity(812.5),
        req=_estimated_v1_request(),
        resolved_first_layer=_resolved_estimated_v1(),
    )

    assert payload['velocity']['v1_mode'] == 'estimate_direct_arrival'
    assert payload['velocity']['v1_status'] == 'estimated'
    assert payload['velocity']['resolved_weathering_velocity_m_s'] == pytest.approx(
        812.5
    )


def test_source_depth_double_count_guard_qc_warning(tmp_path: Path) -> None:
    request_payload = _request().model_dump(mode='json')
    request_payload['geometry']['source_depth_byte'] = 115
    request_payload['field_corrections'] = {
        'source_depth': {'mode': 'weathering_velocity_time'}
    }
    req = RefractionStaticApplyRequest.model_validate(request_payload)
    result = replace(
        _result(),
        source_depth_m=np.asarray([4.0, 8.0], dtype=np.float64),
        source_depth_shift_s=np.asarray([0.005, 0.010], dtype=np.float64),
        source_depth_status=np.asarray(['ok', 'ok'], dtype='<U48'),
        source_depth_field_correction_qc={
            'source_depth_mode': 'weathering_velocity_time',
            'component_name': 'source_depth_shift_s',
            'source_depth_shift_formula': (
                'source_depth_shift_s = +source_depth_m / V1_m_s'
            ),
            'sign_convention': 'corrected(t) = raw(t - shift_s)',
            'v1_m_s': 800.0,
            'source_depth_double_count_guard': (
                'warning_existing_datum_uses_source_depth'
            ),
            'warnings': ['source depth double-count warning'],
        },
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    assert qc['source_depth_double_count_guard'] == (
        'warning_existing_datum_uses_source_depth'
    )
    assert qc['field_corrections']['source_depth']['component_name'] == (
        'source_depth_shift_s'
    )
    assert qc['warnings'] == ['source depth double-count warning']
    source_rows = _read_csv(paths.source_static_table_csv)
    assert source_rows[0]['source_depth_m'] == '4.0'
    assert float(source_rows[1]['source_depth_shift_ms']) == pytest.approx(10.0)
    component_rows = _read_csv(paths.refraction_static_components_csv)
    assert component_rows[0]['source_depth_shift_ms'] == '5.0'
    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        np.testing.assert_allclose(data['source_depth_shift_s'], [0.005, 0.010])
        assert data['source_depth_status'].tolist() == ['ok', 'ok']
    with np.load(paths.solution_npz, allow_pickle=False) as data:
        np.testing.assert_allclose(data['source_depth_m'], [4.0, 8.0])


def test_uphole_field_correction_qc_and_static_tables(tmp_path: Path) -> None:
    request_payload = _request().model_dump(mode='json')
    request_payload['field_corrections'] = {
        'uphole': {
            'mode': 'header_time',
            'uphole_time_byte': 95,
            'uphole_time_unit': 's',
        }
    }
    req = RefractionStaticApplyRequest.model_validate(request_payload)
    result = replace(
        _result(),
        source_uphole_time_s=np.asarray([0.010, 0.020], dtype=np.float64),
        source_uphole_shift_s=np.asarray([-0.010, -0.020], dtype=np.float64),
        source_uphole_status=np.asarray(['ok', 'ok'], dtype='<U48'),
        source_uphole_field_correction_qc={
            'uphole_mode': 'header_time',
            'component_name': 'uphole_shift_s',
            'uphole_shift_formula': 'uphole_shift_s = -uphole_time_s',
            'sign_convention': 'corrected(t) = raw(t - shift_s)',
            'positive_time_means_delay': True,
            'uphole_time_byte': 95,
            'uphole_time_unit': 's',
        },
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    assert qc['field_corrections']['uphole']['component_name'] == 'uphole_shift_s'
    assert qc['field_corrections']['uphole']['sign_convention'] == (
        'corrected(t) = raw(t - shift_s)'
    )
    source_rows = _read_csv(paths.source_static_table_csv)
    assert float(source_rows[0]['uphole_time_ms']) == pytest.approx(10.0)
    assert float(source_rows[1]['uphole_shift_ms']) == pytest.approx(-20.0)
    component_rows = _read_csv(paths.refraction_static_components_csv)
    assert component_rows[0]['uphole_shift_ms'] == '-10.0'
    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        np.testing.assert_allclose(data['source_uphole_time_s'], [0.010, 0.020])
        np.testing.assert_allclose(data['source_uphole_shift_s'], [-0.010, -0.020])
        assert data['source_uphole_status'].tolist() == ['ok', 'ok']
    with np.load(paths.solution_npz, allow_pickle=False) as data:
        np.testing.assert_allclose(data['source_uphole_shift_s'], [-0.010, -0.020])


def test_refraction_manifest_registers_field_correction_artifacts(
    tmp_path: Path,
) -> None:
    request_payload = _request().model_dump(mode='json')
    request_payload['field_corrections'] = {
        'source_depth': {
            'mode': 'weathering_velocity_time',
            'source_depth_byte': 115,
        },
        'uphole': {
            'mode': 'header_time',
            'uphole_time_byte': 95,
            'uphole_time_unit': 's',
        },
    }
    req = RefractionStaticApplyRequest.model_validate(request_payload)
    result = replace(
        _result(),
        source_depth_m=np.asarray([4.0, 8.0], dtype=np.float64),
        source_depth_shift_s=np.asarray([0.005, 0.010], dtype=np.float64),
        source_depth_status=np.asarray(['ok', 'ok'], dtype='<U48'),
        source_depth_field_correction_qc={
            'source_depth_mode': 'weathering_velocity_time',
            'component_name': 'source_depth_shift_s',
            'sign_convention': 'corrected(t) = raw(t - shift_s)',
            'source_depth_double_count_guard': 'checked',
        },
        source_uphole_time_s=np.asarray([0.010, 0.020], dtype=np.float64),
        source_uphole_shift_s=np.asarray([-0.010, -0.020], dtype=np.float64),
        source_uphole_status=np.asarray(['ok', 'ok'], dtype='<U48'),
        source_uphole_field_correction_qc={
            'uphole_mode': 'header_time',
            'component_name': 'uphole_shift_s',
            'sign_convention': 'corrected(t) = raw(t - shift_s)',
        },
    )
    upstream_names = (
        REFRACTION_SOURCE_DEPTH_QC_JSON_NAME,
        REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
        REFRACTION_UPHOLE_QC_JSON_NAME,
        REFRACTION_UPHOLE_SOURCES_CSV_NAME,
    )
    for name in upstream_names:
        (tmp_path / name).write_text('{}' if name.endswith('.json') else '', encoding='utf-8')

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
        upstream_artifact_names=upstream_names,
    )

    manifest = json.loads(paths.manifest_json.read_text(encoding='utf-8'))
    artifacts = {item['name']: item for item in manifest['artifacts']}
    for name in upstream_names:
        assert artifacts[name]['origin'] == 'upstream'
    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    qc_artifacts = {item['name'] for item in qc['artifacts']}
    assert set(upstream_names).issubset(qc_artifacts)
    assert qc['field_corrections']['source_depth']['sign_convention'] == (
        'corrected(t) = raw(t - shift_s)'
    )
    assert qc['field_corrections']['uphole']['sign_convention'] == (
        'corrected(t) = raw(t - shift_s)'
    )
    json.dumps(qc, allow_nan=False)


def test_write_refraction_static_artifacts_csvs(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    trace_rows = _read_csv(paths.refraction_statics_csv)
    assert len(trace_rows) == 4
    assert [int(row['sorted_trace_index']) for row in trace_rows] == [0, 1, 2, 3]
    assert trace_rows[0]['trace_static_status'] == 'ok'
    assert 'source_half_intercept_time_ms' in trace_rows[0]
    assert float(trace_rows[0]['source_half_intercept_time_ms']) == pytest.approx(10.0)
    assert float(trace_rows[0]['estimated_first_break_time_ms']) == pytest.approx(50.0)
    assert trace_rows[2]['refraction_trace_shift_ms'] == ''

    model_rows = _read_csv(paths.near_surface_model_csv)
    assert len(model_rows) == 3
    assert model_rows[0]['weathering_thickness_m'] == '10.0'
    assert model_rows[0]['solution_status'] == 'solved'
    assert model_rows[1]['weathering_status'] == 'zero_thickness'
    assert float(model_rows[0]['half_intercept_time_ms']) == pytest.approx(10.0)

    residual_rows = _read_csv(paths.first_break_residuals_csv)
    assert len(residual_rows) == 3
    assert float(residual_rows[0]['observed_pick_time_ms']) == pytest.approx(50.0)
    assert float(residual_rows[1]['residual_ms']) == pytest.approx(-2.0)

    first_break_rows = _read_csv(paths.refraction_first_break_time_export_csv)
    assert len(first_break_rows) == 3
    assert first_break_rows[0]['source_endpoint_key'] == 's0'
    assert first_break_rows[0]['receiver_endpoint_key'] == 'r0'
    assert float(first_break_rows[0]['observed_pick_time_ms']) == pytest.approx(
        50.0
    )
    assert float(first_break_rows[0]['modeled_pick_time_ms']) == pytest.approx(
        49.0
    )
    assert float(first_break_rows[0]['residual_ms']) == pytest.approx(1.0)

    component_rows = _read_csv(paths.refraction_static_components_csv)
    assert len(component_rows) == 4
    assert {row['kind'] for row in component_rows} == {'source', 'receiver'}
    assert float(component_rows[0]['half_intercept_time_ms']) == pytest.approx(10.0)
    assert 'refraction_shift_ms' in component_rows[0]


def test_first_break_time_export_contains_observed_modeled_residual(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    rows, fieldnames = _read_csv_with_fieldnames(
        paths.refraction_first_break_time_export_csv
    )

    assert tuple(fieldnames) == (
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
    assert rows[0]['format_name'] == 'first_break_time'
    assert rows[0]['format_version'] == '1'
    assert rows[0]['source_job_id'] == ''
    assert rows[0]['observation_index'] == '0'
    assert rows[0]['sorted_trace_index'] == '0'
    assert rows[0]['source_id'] == '100'
    assert rows[0]['receiver_id'] == '200'
    assert rows[0]['layer_kind'] == 'v2_t1'
    assert rows[0]['used_in_solve'] == 'true'
    assert float(rows[0]['observed_pick_time_ms']) == pytest.approx(50.0)
    assert float(rows[0]['modeled_pick_time_ms']) == pytest.approx(49.0)
    assert float(rows[0]['residual_ms']) == pytest.approx(1.0)
    assert rows[0]['sign_convention'] == (
        artifact_module.FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION
    )


def test_first_break_time_export_marks_rejected_observations(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_first_break_time_export_csv)

    assert rows[1]['used_in_solve'] == 'false'
    assert rows[1]['reject_reason'] == 'robust_outlier'


def test_first_break_time_export_preserves_unassigned_layer_context(
    tmp_path: Path,
) -> None:
    result = replace(
        _result(),
        row_layer_kind=np.asarray(['v2_t1', '', 'v2_t1'], dtype='<U16'),
        row_layer_index=np.asarray([1, 0, 1], dtype=np.int64),
        rejected_by_robust_mask=np.asarray([False, False, False], dtype=bool),
        row_rejection_reason=np.asarray(
            ['ok', 'outside_layer_gate', 'ok'],
            dtype='<U32',
        ),
        row_velocity_m_s=np.asarray([2500.0, np.nan, 2500.0], dtype=np.float64),
    )
    paths = write_refraction_static_artifacts(
        result=result,
        req=_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_first_break_time_export_csv)

    assert rows[1]['layer_kind'] == ''
    assert rows[1]['used_in_solve'] == 'false'
    assert rows[1]['reject_reason'] == 'outside_layer_gate'


def test_first_break_time_export_residual_matches_solution_npz(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_first_break_time_export_csv)
    with np.load(paths.solution_npz, allow_pickle=False) as data:
        residual_ms = data['residual_time_s'] * 1000.0

    np.testing.assert_allclose(
        np.asarray([float(row['residual_ms']) for row in rows]),
        residual_ms,
    )


def test_first_break_fit_qc_csv_npz_json_are_consistent(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_first_break_fit_qc_csv)
    payload = json.loads(
        paths.refraction_first_break_fit_qc_json.read_text(encoding='utf-8')
    )
    with np.load(paths.refraction_first_break_fit_qc_npz, allow_pickle=False) as data:
        observed = data['observed_first_break_time_s']
        modeled = data['modeled_first_break_time_s']
        residual = data['residual_time_s']
        used = data['used_for_inversion']

    assert payload['residual_sign'] == FIRST_BREAK_FIT_QC_RESIDUAL_SIGN
    assert payload['residual_definition'] == (
        'residual_time_s = observed_first_break_time_s - '
        'modeled_first_break_time_s'
    )
    assert payload['row_count'] == len(rows) == 3
    assert payload['used_count'] == 2
    assert payload['rejected_count'] == 1
    assert rows[0]['trace_index_sorted'] == '0'
    assert rows[0]['source_endpoint_key'] == 's0'
    assert rows[0]['receiver_endpoint_key'] == 'r0'
    assert rows[0]['source_node_id'] == '0'
    assert rows[0]['receiver_node_id'] == '1'
    assert rows[0]['status'] == 'ok'
    assert rows[1]['status'] == 'rejected'
    assert rows[1]['rejection_reason'] == 'robust_outlier'
    assert rows[1]['used_for_inversion'] == 'false'
    assert rows[0]['sign_convention'] == 'corrected(t) = raw(t - shift_s)'

    np.testing.assert_allclose(observed - modeled, residual)
    np.testing.assert_allclose(
        np.asarray([float(row['observed_first_break_time_s']) for row in rows]),
        observed,
    )
    np.testing.assert_allclose(
        np.asarray([float(row['modeled_first_break_time_s']) for row in rows]),
        modeled,
    )
    np.testing.assert_allclose(
        np.asarray([float(row['residual_time_s']) for row in rows]),
        residual,
    )
    assert payload['residual_summary']['used_rms_s'] == pytest.approx(
        float(np.sqrt(np.mean(residual[used] * residual[used])))
    )


def test_reduced_time_qc_fixed_velocity_formula() -> None:
    payload = _request().model_dump(mode='json')
    payload['reduced_time_qc'] = {
        'reduction_velocity_mode': 'fixed',
        'fixed_velocity_m_s': 2000.0,
    }
    arrays = build_refraction_reduced_time_qc_arrays(
        result=_result(),
        req=RefractionStaticApplyRequest.model_validate(payload),
    )

    expected = np.asarray([0.050, 0.060, 0.070]) - (
        np.asarray([100.0, 200.0, 300.0]) / 2000.0
    )
    np.testing.assert_allclose(arrays['reduced_time_s'], expected)
    np.testing.assert_allclose(arrays['reduced_time_ms'], expected * 1000.0)
    np.testing.assert_allclose(arrays['reduction_velocity_m_s'], 2000.0)
    np.testing.assert_allclose(
        arrays['observed_first_break_time_s'],
        [0.050, 0.060, 0.070],
    )
    assert arrays['trace_index_sorted'].tolist() == [0, 1, 2]
    assert arrays['used_for_inversion'].tolist() == [True, False, True]
    assert arrays['status'].tolist() == ['ok', 'ok', 'ok']


def test_reduced_time_qc_csv_preserves_legacy_row_contract(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_reduced_time_qc_csv)
    payload = json.loads(
        paths.refraction_reduced_time_qc_json.read_text(encoding='utf-8')
    )

    assert rows[0]['trace_index_sorted'] == '0'
    assert rows[0]['layer_gate_kind'] == 'v2_t1'
    assert rows[0]['observed_first_break_time_s'] == '0.05'
    assert rows[0]['reduced_time_s'] != ''
    assert rows[0]['reduced_time_ms'] != ''
    assert rows[1]['used_for_inversion'] == 'false'
    assert 'layer_kind' not in rows[0]
    assert 'observed_time_s' not in rows[0]
    assert 'modeled_reduced_time_s' not in rows[0]
    assert 'used_in_solve' not in rows[0]
    assert payload['columns'] == list(rows[0].keys())
    assert payload['layer_gate_kind_counts'] == {'v2_t1': 3}


def test_reduced_time_qc_layer_velocity_formula_for_two_layer() -> None:
    payload = _request().model_dump(mode='json')
    payload['model'] = {
        'method': 'multilayer_time_term',
        'first_layer': {
            'mode': 'constant',
            'weathering_velocity_m_s': 800.0,
        },
        'layers': [
            {
                'kind': 'v2_t1',
                'min_offset_m': 0.0,
                'max_offset_m': 150.0,
                'velocity_mode': 'solve_global',
                'initial_velocity_m_s': 2400.0,
                'min_velocity_m_s': 1200.0,
                'max_velocity_m_s': 3500.0,
            },
            {
                'kind': 'v3_t2',
                'min_offset_m': 150.0,
                'max_offset_m': None,
                'velocity_mode': 'solve_global',
                'initial_velocity_m_s': 3600.0,
                'min_velocity_m_s': 2600.0,
                'max_velocity_m_s': 6500.0,
            },
        ],
    }
    payload['conversion'] = {
        'mode': 't1lsst_multilayer',
        'layer_count': 2,
    }
    result = replace(
        _result(),
        row_layer_kind=np.asarray(['v2_t1', 'v3_t2', 'v3_t2'], dtype='<U16'),
        row_velocity_m_s=np.asarray([2400.0, 3600.0, 3600.0], dtype=np.float64),
    )

    arrays = build_refraction_reduced_time_qc_arrays(
        result=result,
        req=RefractionStaticApplyRequest.model_validate(payload),
    )

    expected = np.asarray(
        [
            0.050 - 100.0 / 2400.0,
            0.060 - 200.0 / 3600.0,
            0.070 - 300.0 / 3600.0,
        ]
    )
    np.testing.assert_allclose(arrays['reduced_time_s'], expected)
    assert arrays['layer_gate_kind'].tolist() == ['v2_t1', 'v3_t2', 'v3_t2']


def test_reduced_time_qc_marks_missing_velocity_status() -> None:
    result = replace(
        _result(),
        row_velocity_m_s=np.asarray([2500.0, np.nan, 2500.0], dtype=np.float64),
    )

    arrays = build_refraction_reduced_time_qc_arrays(
        result=result,
        req=_request(),
    )

    assert arrays['status'].tolist() == [
        'ok',
        'missing_reduction_velocity',
        'ok',
    ]
    assert np.isnan(arrays['reduced_time_s'][1])


def test_reduced_time_qc_includes_layer_gate_flags() -> None:
    arrays = build_refraction_reduced_time_qc_arrays(
        result=_result_with_weathering_velocity(812.5),
        req=_estimated_v1_request(),
    )

    assert arrays['within_v1_gate'].tolist() == [True, False, False]
    assert arrays['within_v2_t1_gate'].tolist() == [True, True, True]
    assert arrays['within_v3_t2_gate'].tolist() == [False, False, False]
    assert arrays['within_vsub_t3_gate'].tolist() == [False, False, False]


def test_first_break_fit_qc_reports_cell_and_multilayer_context(
    tmp_path: Path,
) -> None:
    result = replace(
        _solve_cell_result(),
        row_layer_kind=np.asarray(['v2_t1', 'v3_t2', 'vsub_t3'], dtype='<U16'),
        row_rejection_reason=np.asarray(
            ['ok', 'outside_layer_offset_gate', 'ok'],
            dtype='<U32',
        ),
        used_row_mask=np.asarray([True, False, True], dtype=bool),
        rejected_by_robust_mask=np.asarray([False, False, False], dtype=bool),
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_first_break_fit_qc_csv)
    with np.load(paths.refraction_first_break_fit_qc_npz, allow_pickle=False) as data:
        assert data['layer_kind'].tolist() == ['v2_t1', 'v3_t2', 'vsub_t3']
        np.testing.assert_allclose(data['cell_id'], np.asarray([0.0, np.nan, np.nan]))
        np.testing.assert_allclose(data['cell_ix'], np.asarray([0.0, np.nan, np.nan]))
        np.testing.assert_allclose(data['cell_iy'], np.asarray([0.0, np.nan, np.nan]))

    assert [row['layer_kind'] for row in rows] == ['v2_t1', 'v3_t2', 'vsub_t3']
    assert [row['cell_id'] for row in rows] == ['0', '', '']
    assert [row['cell_ix'] for row in rows] == ['0', '', '']
    assert [row['cell_iy'] for row in rows] == ['0', '', '']
    assert rows[1]['status'] == 'rejected'
    assert rows[1]['rejection_reason'] == 'outside_layer_offset_gate'


def test_refraction_static_artifacts_manifest_and_download_visibility(
    tmp_path: Path,
) -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )
    manifest = json.loads(
        (tmp_path / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    artifacts = {item['name']: item for item in manifest['artifacts']}
    assert set(artifacts) == EXPECTED_FILENAMES - {REFRACTION_STATIC_ARTIFACTS_JSON_NAME}
    assert {item['origin'] for item in manifest['artifacts']} == {'final'}
    assert GRID_MAP_QC_FILENAMES.isdisjoint(artifacts)
    for artifact_name, description in M6_QC_ARTIFACT_DESCRIPTIONS.items():
        if artifact_name in GRID_MAP_QC_FILENAMES:
            continue
        assert artifacts[artifact_name]['description'] == description
        assert artifacts[artifact_name]['content_type'] == _content_type_for_name(
            artifact_name
        )

    state = app.state.sv
    with state.lock:
        state.jobs.clear()
        state.jobs.create_static_job(
            'refraction-artifacts-job',
            file_id='raw-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction',
            artifacts_dir=str(tmp_path),
        )
    try:
        with TestClient(app) as client:
            files = client.get('/statics/job/refraction-artifacts-job/files')
            assert files.status_code == 200
            assert {item['name'] for item in files.json()['files']} == EXPECTED_FILENAMES

            download = client.get(
                '/statics/job/refraction-artifacts-job/download',
                params={'name': REFRACTION_STATIC_QC_JSON_NAME},
            )
            assert download.status_code == 200
            assert download.json()['artifact_version'] == '1.0'

            for artifact_name in M6_QC_ARTIFACT_DESCRIPTIONS:
                if artifact_name in GRID_MAP_QC_FILENAMES:
                    continue
                response = client.get(
                    '/statics/job/refraction-artifacts-job/download',
                    params={'name': artifact_name},
                )
                assert response.status_code == 200, artifact_name
                assert response.content, artifact_name
    finally:
        with state.lock:
            state.jobs.clear()


def test_solve_cell_writes_refractor_velocity_cells_csv(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    assert paths.refraction_refractor_velocity_cells_csv is not None
    rows = _read_csv(paths.refraction_refractor_velocity_cells_csv)

    assert [int(row['cell_id']) for row in rows] == [0, 1, 2]
    assert rows[0]['active'] == 'true'
    assert rows[0]['velocity_status'] == 'solved'
    assert float(rows[0]['velocity_m_s']) == pytest.approx(2400.0)
    assert float(rows[0]['v2_m_s']) == pytest.approx(2400.0)
    assert float(rows[0]['initial_velocity_m_s']) == pytest.approx(2500.0)
    assert float(rows[0]['initial_v2_m_s']) == pytest.approx(2500.0)
    assert float(rows[0]['velocity_update_from_initial_m_s']) == pytest.approx(
        -100.0
    )
    assert float(rows[0]['v2_update_from_initial_m_s']) == pytest.approx(-100.0)
    assert int(rows[0]['n_used_observations']) == 2
    assert rows[2]['active'] == 'false'
    assert rows[2]['velocity_status'] == 'inactive'
    assert rows[2]['velocity_m_s'] == ''
    assert rows[2]['v2_m_s'] == ''
    assert rows[2]['x_min_m'] == '200.0'


def test_solve_cell_writes_refractor_velocity_grid_npz(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    assert paths.refraction_refractor_velocity_grid_npz is not None
    with np.load(paths.refraction_refractor_velocity_grid_npz, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['cell_id'], [0, 1, 2])
        np.testing.assert_array_equal(data['active_cell_mask'], [True, True, False])
        np.testing.assert_allclose(data['velocity_m_s'][:2], [2400.0, 2600.0])
        np.testing.assert_allclose(data['v2_m_s'][:2], [2400.0, 2600.0])
        np.testing.assert_allclose(
            data['velocity_m_s'],
            data['v2_m_s'],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            data['initial_velocity_m_s'],
            data['initial_v2_m_s'],
        )
        np.testing.assert_allclose(
            data['velocity_update_from_initial_m_s'],
            data['v2_update_from_initial_m_s'],
            equal_nan=True,
        )
        assert np.isnan(data['velocity_m_s'][2])
        assert np.isnan(data['v2_m_s'][2])
        assert data['velocity_status'].tolist() == ['solved', 'solved', 'inactive']
        np.testing.assert_array_equal(
            data['n_observations_per_cell'],
            [2, 1, 0],
        )
        np.testing.assert_array_equal(
            data['n_used_observations_per_cell'],
            [2, 0, 0],
        )


def test_solve_cell_writes_low_fold_refractor_velocity_artifacts(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_low_fold_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    assert paths.refraction_refractor_velocity_cells_csv is not None
    rows = _read_csv(paths.refraction_refractor_velocity_cells_csv)
    assert rows[1]['active'] == 'false'
    assert rows[1]['velocity_status'] == 'low_fold'
    assert rows[1]['v2_m_s'] == ''
    assert int(rows[1]['n_observations']) == 1
    assert int(rows[1]['n_used_observations']) == 0
    assert int(rows[1]['n_rejected_observations']) == 1

    assert paths.refraction_refractor_velocity_qc_json is not None
    cell_qc = json.loads(
        paths.refraction_refractor_velocity_qc_json.read_text(encoding='utf-8')
    )
    assert cell_qc['min_observations_per_cell'] == 2
    assert cell_qc['n_low_fold_cells'] == 1
    assert cell_qc['n_observations_rejected_by_low_fold_cell'] == 1
    assert (
        cell_qc['low_fold_cell_rejection_reason']
        == 'below_min_observations_per_cell'
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    assert source_rows[1]['v2_status'] == 'low_fold_v2_cell'
    assert source_rows[1]['static_status'] == 'low_fold_v2_cell'
    assert receiver_rows[0]['v2_status'] == 'low_fold_v2_cell'
    assert receiver_rows[0]['static_status'] == 'low_fold_v2_cell'


def test_grid_map_qc_generated_for_v2_solve_cell(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    assert paths.refraction_grid_map_qc_csv is not None
    assert paths.refraction_grid_map_qc_npz is not None
    assert paths.refraction_grid_map_qc_json is not None
    rows = _read_csv(paths.refraction_grid_map_qc_csv)
    assert len(rows) == 3
    assert GRID_MAP_QC_REQUIRED_COLUMNS <= set(rows[0])
    assert {row['layer_kind'] for row in rows} == {'v2_t1'}
    assert rows[0]['status'] == 'solved'
    assert float(rows[0]['velocity_m_s']) == pytest.approx(2400.0)
    assert float(rows[0]['initial_velocity_m_s']) == pytest.approx(2500.0)
    assert float(rows[0]['velocity_update_from_initial_m_s']) == pytest.approx(
        -100.0
    )
    assert int(rows[0]['n_sources']) == 1
    assert int(rows[0]['n_receivers']) == 2

    with np.load(paths.refraction_grid_map_qc_npz, allow_pickle=False) as data:
        assert data['artifact_kind'].item() == 'refraction_grid_map_qc'
        assert data['global_velocity_layer_behavior'].item() == (
            'omitted_from_grid_map_qc_rows'
        )
        assert data['layer_kind'].tolist() == ['v2_t1', 'v2_t1', 'v2_t1']
        np.testing.assert_array_equal(data['cell_ix'], [0, 1, 2])
        np.testing.assert_allclose(
            data['velocity_m_s'],
            np.asarray([2400.0, 2600.0, np.nan], dtype=np.float64),
            equal_nan=True,
        )
        assert data['status'].tolist() == ['solved', 'solved', 'inactive']

    summary = json.loads(
        paths.refraction_grid_map_qc_json.read_text(encoding='utf-8')
    )
    assert summary['grid']['coordinate_mode'] == 'grid_3d'
    assert summary['grid']['number_of_cell_x'] == 3
    assert summary['grid']['y_axis_unbounded'] is True
    assert summary['global_velocity_layer_behavior'] == (
        'omitted_from_grid_map_qc_rows'
    )
    assert summary['layers']['v2_t1']['active_cell_count'] == 2


def test_grid_map_qc_layer_neutral_columns_for_v3_and_vsub(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_multilayer_result_with_v3_and_vsub_cell_layers(),
        req=_v3_vsub_solve_cell_request(),
        job_dir=tmp_path,
    )

    assert paths.refraction_grid_map_qc_csv is not None
    rows = _read_csv(paths.refraction_grid_map_qc_csv)
    assert len(rows) == 6
    assert GRID_MAP_QC_REQUIRED_COLUMNS <= set(rows[0])
    assert 'v2_m_s' not in rows[0]
    assert 'cell_velocity_component' not in rows[0]
    assert {row['layer_kind'] for row in rows} == {'v3_t2', 'vsub_t3'}
    v3_rows = [row for row in rows if row['layer_kind'] == 'v3_t2']
    vsub_rows = [row for row in rows if row['layer_kind'] == 'vsub_t3']
    assert [row['velocity_m_s'] for row in v3_rows] == ['3300.0', '3700.0', '']
    assert [row['velocity_m_s'] for row in vsub_rows] == ['', '4800.0', '5200.0']

    summary = json.loads(
        paths.refraction_grid_map_qc_json.read_text(encoding='utf-8')
    )
    assert summary['cell_velocity_layer_kinds'] == ['v3_t2', 'vsub_t3']
    assert summary['omitted_global_velocity_layers'] == [
        {
            'layer_kind': 'v2_t1',
            'velocity_mode': 'fixed_global',
            'row_behavior': 'omitted',
        }
    ]


def test_grid_map_qc_includes_empty_and_low_fold_statuses(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_low_fold_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    assert paths.refraction_grid_map_qc_csv is not None
    rows = {
        int(row['cell_ix']): row
        for row in _read_csv(paths.refraction_grid_map_qc_csv)
    }
    assert rows[0]['status'] == 'solved'
    assert rows[1]['status'] == 'low_fold'
    assert rows[1]['status_reason'] == 'below_min_observations_per_cell'
    assert rows[1]['velocity_m_s'] == ''
    assert rows[2]['status'] == 'inactive'
    assert rows[2]['status_reason'] == 'no_observations'

    summary = json.loads(
        paths.refraction_grid_map_qc_json.read_text(encoding='utf-8')
    )
    assert summary['layers']['v2_t1']['empty_cell_count'] == 1
    assert summary['layers']['v2_t1']['low_fold_cell_count'] == 1


def test_grid_map_qc_json_summary_matches_csv_counts(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_low_fold_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    assert paths.refraction_grid_map_qc_csv is not None
    assert paths.refraction_grid_map_qc_json is not None
    rows = _read_csv(paths.refraction_grid_map_qc_csv)
    summary = json.loads(
        paths.refraction_grid_map_qc_json.read_text(encoding='utf-8')
    )
    v2 = summary['layers']['v2_t1']
    solved_velocity = [
        float(row['velocity_m_s'])
        for row in rows
        if row['layer_kind'] == 'v2_t1' and row['status'] == 'solved'
    ]
    assert v2['active_cell_count'] == sum(
        1 for row in rows if row['status'] == 'solved'
    )
    assert v2['empty_cell_count'] == sum(
        1 for row in rows if row['status_reason'] == 'no_observations'
    )
    assert v2['low_fold_cell_count'] == sum(
        1 for row in rows if row['status'] == 'low_fold'
    )
    assert v2['velocity_min_m_s'] == pytest.approx(min(solved_velocity))
    assert v2['velocity_median_m_s'] == pytest.approx(
        float(np.median(solved_velocity))
    )
    assert v2['velocity_max_m_s'] == pytest.approx(max(solved_velocity))


def test_solve_cell_manifest_registers_cell_velocity_artifacts(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    assert {path.name for path in tmp_path.iterdir()} == (
        EXPECTED_FILENAMES | CELL_VELOCITY_FILENAMES | GRID_MAP_QC_FILENAMES
    )
    manifest = json.loads(paths.manifest_json.read_text(encoding='utf-8'))
    artifacts = {item['name']: item for item in manifest['artifacts']}
    artifact_names = set(artifacts)
    assert CELL_VELOCITY_FILENAMES.issubset(artifact_names)
    assert GRID_MAP_QC_FILENAMES.issubset(artifact_names)
    for artifact_name in GRID_MAP_QC_FILENAMES:
        assert artifacts[artifact_name]['description'] == (
            M6_QC_ARTIFACT_DESCRIPTIONS[artifact_name]
        )
        assert artifacts[artifact_name]['content_type'] == _content_type_for_name(
            artifact_name
        )

    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    assert qc['velocity']['cell_velocity_qc_artifact'] == (
        REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME
    )
    assert qc['refractor_velocity_cells']['grid_npz_artifact'] == (
        REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME
    )
    assert qc['refractor_velocity_cells']['solver_history_csv_artifact'] == (
        REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME
    )
    assert qc['refractor_grid_map_qc']['csv_artifact'] == (
        REFRACTION_GRID_MAP_QC_CSV_NAME
    )


def test_v3_cell_artifacts_use_layer_specific_names(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_multilayer_shaped_solve_cell_result(),
        req=_v3_solve_cell_request(),
        job_dir=tmp_path,
    )

    assert {path.name for path in tmp_path.iterdir()} == (
        EXPECTED_FILENAMES | V3_CELL_VELOCITY_FILENAMES | GRID_MAP_QC_FILENAMES
    )
    assert paths.refraction_refractor_velocity_cells_csv is not None
    assert (
        paths.refraction_refractor_velocity_cells_csv.name
        == REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME
    )
    assert not CELL_VELOCITY_FILENAMES.intersection(
        {path.name for path in tmp_path.iterdir()}
    )

    manifest = json.loads(paths.manifest_json.read_text(encoding='utf-8'))
    artifact_names = {item['name'] for item in manifest['artifacts']}
    assert V3_CELL_VELOCITY_FILENAMES <= artifact_names
    assert GRID_MAP_QC_FILENAMES <= artifact_names
    assert not CELL_VELOCITY_FILENAMES.intersection(artifact_names)

    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    assert qc['velocity']['cell_velocity_layer_kind'] == 'v3_t2'
    assert qc['velocity']['cell_velocity_component'] == 'v3'
    assert (
        qc['velocity']['cell_velocity_qc_artifact']
        == REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME
    )
    assert qc['refractor_velocity_cells']['cells_csv_artifact'] == (
        REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME
    )

    rows = _read_csv(paths.refraction_refractor_velocity_cells_csv)
    assert rows
    assert {row['cell_velocity_layer_kind'] for row in rows} == {'v3_t2'}
    assert {row['cell_velocity_component'] for row in rows} == {'v3'}


def test_multiple_cell_velocity_layers_write_all_layer_artifacts(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_multilayer_result_with_v3_and_vsub_cell_layers(),
        req=_v3_vsub_solve_cell_request(),
        job_dir=tmp_path,
    )

    filenames = {path.name for path in tmp_path.iterdir()}
    assert filenames == (
        EXPECTED_FILENAMES
        | V3_CELL_VELOCITY_FILENAMES
        | VSUB_CELL_VELOCITY_FILENAMES
        | GRID_MAP_QC_FILENAMES
    )
    assert paths.refraction_refractor_velocity_cells_csv is not None
    assert (
        paths.refraction_refractor_velocity_cells_csv.name
        == REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME
    )

    manifest = json.loads(paths.manifest_json.read_text(encoding='utf-8'))
    artifact_names = {item['name'] for item in manifest['artifacts']}
    assert V3_CELL_VELOCITY_FILENAMES <= artifact_names
    assert VSUB_CELL_VELOCITY_FILENAMES <= artifact_names
    assert GRID_MAP_QC_FILENAMES <= artifact_names

    v3_rows = _read_csv(tmp_path / REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME)
    vsub_rows = _read_csv(
        tmp_path / REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME
    )
    assert {row['cell_velocity_layer_kind'] for row in v3_rows} == {'v3_t2'}
    assert {row['cell_velocity_component'] for row in v3_rows} == {'v3'}
    assert {row['cell_velocity_layer_kind'] for row in vsub_rows} == {'vsub_t3'}
    assert {row['cell_velocity_component'] for row in vsub_rows} == {'vsub'}

    with np.load(
        tmp_path / REFRACTION_V3_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
        allow_pickle=False,
    ) as v3_grid:
        np.testing.assert_allclose(
            v3_grid['velocity_m_s'],
            np.asarray([3300.0, 3700.0, np.nan], dtype=np.float64),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            v3_grid['velocity_m_s'],
            v3_grid['v2_m_s'],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            v3_grid['initial_velocity_m_s'],
            v3_grid['initial_v2_m_s'],
        )
        np.testing.assert_allclose(
            v3_grid['velocity_update_from_initial_m_s'],
            v3_grid['v2_update_from_initial_m_s'],
            equal_nan=True,
        )
        assert set(v3_grid['cell_velocity_layer_kind'].astype(str).tolist()) == {
            'v3_t2'
        }
        assert set(v3_grid['cell_velocity_component'].astype(str).tolist()) == {'v3'}

    with np.load(
        tmp_path / REFRACTION_VSUB_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
        allow_pickle=False,
    ) as vsub_grid:
        np.testing.assert_allclose(
            vsub_grid['velocity_m_s'],
            np.asarray([np.nan, 4800.0, 5200.0], dtype=np.float64),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            vsub_grid['velocity_m_s'],
            vsub_grid['v2_m_s'],
            equal_nan=True,
        )
        assert set(vsub_grid['cell_velocity_layer_kind'].astype(str).tolist()) == {
            'vsub_t3'
        }
        assert set(vsub_grid['cell_velocity_component'].astype(str).tolist()) == {
            'vsub'
        }

    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    assert qc['velocity']['cell_velocity_layer_kinds'] == ['v3_t2', 'vsub_t3']
    assert qc['velocity']['cell_velocity_qc_artifacts_by_layer'] == {
        'v3_t2': REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME,
        'vsub_t3': REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME,
    }
    assert set(qc['refractor_velocity_cells_by_layer']) == {'v3_t2', 'vsub_t3'}


def test_solve_global_does_not_write_cell_velocity_artifacts(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    assert paths.refraction_refractor_velocity_cells_csv is None
    assert paths.refraction_grid_map_qc_csv is None
    assert not CELL_VELOCITY_FILENAMES.intersection(
        {path.name for path in tmp_path.iterdir()}
    )
    assert not GRID_MAP_QC_FILENAMES.intersection(
        {path.name for path in tmp_path.iterdir()}
    )
    manifest = json.loads(paths.manifest_json.read_text(encoding='utf-8'))
    artifact_names = {item['name'] for item in manifest['artifacts']}
    assert not CELL_VELOCITY_FILENAMES.intersection(artifact_names)
    assert not GRID_MAP_QC_FILENAMES.intersection(artifact_names)


def test_source_receiver_tables_include_v2_status_in_cell_mode(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)

    assert source_rows[0]['source_v2_cell_id'] == '0'
    assert source_rows[0]['v2_status'] == 'ok'
    assert receiver_rows[1]['receiver_v2_cell_id'] == '2'
    assert receiver_rows[1]['v2_status'] == 'inactive_v2_cell'


def test_download_cell_velocity_artifacts(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    write_refraction_static_artifacts(
        result=_solve_cell_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    state = app.state.sv
    with state.lock:
        state.jobs.clear()
        state.jobs.create_static_job(
            'refraction-cell-artifacts-job',
            file_id='raw-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction',
            artifacts_dir=str(tmp_path),
        )
    try:
        with TestClient(app) as client:
            files = client.get('/statics/job/refraction-cell-artifacts-job/files')
            assert files.status_code == 200
            assert CELL_VELOCITY_FILENAMES.issubset(
                {item['name'] for item in files.json()['files']}
            )

            download = client.get(
                '/statics/job/refraction-cell-artifacts-job/download',
                params={'name': REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME},
            )
            assert download.status_code == 200
            assert download.json()['bedrock_velocity_mode'] == 'solve_cell'
    finally:
        with state.lock:
            state.jobs.clear()


def test_refraction_static_manifest_includes_v1_artifacts_after_v1_estimation(
    tmp_path: Path,
) -> None:
    _write_upstream_v1_artifacts(tmp_path)

    write_refraction_static_artifacts(
        result=_result_with_weathering_velocity(812.5),
        req=_estimated_v1_request(),
        job_dir=tmp_path,
        resolved_first_layer=_resolved_estimated_v1(),
        upstream_artifact_names=UPSTREAM_V1_ARTIFACT_NAMES,
    )

    manifest = json.loads(
        (tmp_path / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    artifacts = {item['name']: item for item in manifest['artifacts']}
    assert artifacts[REFRACTION_V1_QC_JSON_NAME]['required'] is True
    assert artifacts[REFRACTION_V1_QC_JSON_NAME]['origin'] == 'upstream'
    assert artifacts[REFRACTION_V1_ESTIMATES_CSV_NAME]['required'] is True
    assert artifacts[REFRACTION_V1_ESTIMATES_CSV_NAME]['origin'] == 'upstream'
    assert artifacts[REFRACTION_STATIC_SOLUTION_NPZ_NAME]['origin'] == 'final'

    qc = json.loads((tmp_path / REFRACTION_STATIC_QC_JSON_NAME).read_text())
    qc_artifacts = {item['name']: item for item in qc['artifacts']}
    assert REFRACTION_V1_QC_JSON_NAME in qc_artifacts
    assert REFRACTION_V1_ESTIMATES_CSV_NAME in qc_artifacts


def test_refraction_static_artifact_writer_missing_upstream_v1_artifacts_error_is_clear(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RefractionStaticArtifactError,
        match='declared upstream artifact missing: refraction_v1_qc.json',
    ):
        write_refraction_static_artifacts(
            result=_result_with_weathering_velocity(812.5),
            req=_estimated_v1_request(),
            job_dir=tmp_path,
            resolved_first_layer=_resolved_estimated_v1(),
            upstream_artifact_names=UPSTREAM_V1_ARTIFACT_NAMES,
        )


def test_refraction_static_artifact_writer_estimated_v1_omits_unprovided_upstream_artifacts(
    tmp_path: Path,
) -> None:
    write_refraction_static_artifacts(
        result=_result_with_weathering_velocity(812.5),
        req=_estimated_v1_request(),
        job_dir=tmp_path,
        resolved_first_layer=_resolved_estimated_v1(),
    )

    manifest = json.loads(
        (tmp_path / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    artifacts = {item['name']: item for item in manifest['artifacts']}
    assert REFRACTION_V1_QC_JSON_NAME not in artifacts
    assert REFRACTION_V1_ESTIMATES_CSV_NAME not in artifacts
    assert not (tmp_path / REFRACTION_V1_QC_JSON_NAME).exists()
    assert not (tmp_path / REFRACTION_V1_ESTIMATES_CSV_NAME).exists()


def test_refraction_static_artifact_writer_constant_v1_does_not_require_v1_files(
    tmp_path: Path,
) -> None:
    write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    manifest = json.loads(
        (tmp_path / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    artifacts = {item['name']: item for item in manifest['artifacts']}
    assert REFRACTION_V1_QC_JSON_NAME not in artifacts
    assert REFRACTION_V1_ESTIMATES_CSV_NAME not in artifacts
    assert not (tmp_path / REFRACTION_V1_QC_JSON_NAME).exists()
    assert not (tmp_path / REFRACTION_V1_ESTIMATES_CSV_NAME).exists()

    qc = json.loads((tmp_path / REFRACTION_STATIC_QC_JSON_NAME).read_text())
    qc_artifacts = {item['name']: item for item in qc['artifacts']}
    assert REFRACTION_V1_QC_JSON_NAME not in qc_artifacts
    assert REFRACTION_V1_ESTIMATES_CSV_NAME not in qc_artifacts
    assert qc['velocity']['v1_mode'] == 'constant'


def test_refraction_static_qc_includes_layer_observation_counts_when_present(
    tmp_path: Path,
) -> None:
    layer_qc = {
        'v2_t1': {
            'enabled': True,
            'n_candidate_observations': 3,
            'n_used_observations': 2,
            'min_offset_m': 0.0,
            'max_offset_m': 1000.0,
            'rejection_counts': {
                'ok': 2,
                'outside_layer_offset_gate': 1,
            },
        }
    }
    base = _result()
    result = replace(base, qc={**base.qc, 'layers': layer_qc})

    write_refraction_static_artifacts(
        result=result,
        req=_request(),
        job_dir=tmp_path,
    )

    qc = json.loads((tmp_path / REFRACTION_STATIC_QC_JSON_NAME).read_text())
    assert qc['layers'] == layer_qc


def test_refraction_static_manifest_strict_json(tmp_path: Path) -> None:
    write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    payload = json.loads(
        (tmp_path / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    json.dumps(payload, allow_nan=False)
    assert payload['job_kind'] == 'statics'
    assert payload['statics_kind'] == 'refraction'
    assert {
        'name',
        'kind',
        'content_type',
        'required',
        'origin',
        'description',
    }.issubset(payload['artifacts'][0])


def test_write_refraction_static_artifacts_rejects_missing_job_dir(
    tmp_path: Path,
) -> None:
    with pytest.raises(RefractionStaticArtifactError, match='missing job directory'):
        write_refraction_static_artifacts(
            result=_result(),
            req=_request(),
            job_dir=tmp_path / 'missing',
        )


def test_write_refraction_static_artifacts_rejects_non_writable_job_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifact_module.os, 'access', lambda _path, _mode: False)

    with pytest.raises(RefractionStaticArtifactError, match='not writable'):
        write_refraction_static_artifacts(
            result=_result(),
            req=_request(),
            job_dir=tmp_path,
        )


def test_write_refraction_static_artifacts_validation_failures(
    tmp_path: Path,
) -> None:
    cases = [
        (
            replace(_result(), refraction_trace_shift_s_sorted=np.zeros(3)),
            'trace-order array length mismatch',
        ),
        (
            replace(_result(), node_x_m=np.zeros(2)),
            'node array length mismatch',
        ),
        (
            replace(_result(), source_x_m=np.zeros(1)),
            'source endpoint array length mismatch',
        ),
        (
            replace(_result(), receiver_x_m=np.zeros(1)),
            'receiver endpoint array length mismatch',
        ),
        (
            replace(_result(), residual_time_s=np.zeros(2)),
            'residual array length mismatch',
        ),
        (
            replace(_result(), bedrock_velocity_m_s=float('nan')),
            'non-finite required scalar bedrock_velocity_m_s',
        ),
    ]

    for index, (result, message) in enumerate(cases):
        job_dir = tmp_path / f'case-{index}'
        job_dir.mkdir()
        with pytest.raises(RefractionStaticArtifactError, match=message):
            write_refraction_static_artifacts(
                result=result,
                req=_request(),
                job_dir=job_dir,
            )


def test_solve_cell_artifacts_require_local_v2_arrays(tmp_path: Path) -> None:
    with pytest.raises(
        RefractionStaticArtifactError,
        match='solve_cell result requires node_v2_cell_id',
    ):
        write_refraction_static_solution_npz(
            result=replace(_result(), bedrock_velocity_mode='solve_cell'),
            req=_request(),
            path=tmp_path / REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        )


def test_write_refraction_static_artifacts_rejects_unknown_status(
    tmp_path: Path,
) -> None:
    result = replace(
        _result(),
        node_solution_status=np.asarray(
            ['solved', 'definitely_unknown_status', 'inactive'],
            dtype='<U32',
        ),
    )

    with pytest.raises(
        RefractionStaticArtifactError,
        match='unknown status array values in node_solution_status.*definitely_unknown_status',
    ):
        write_refraction_static_artifacts(
            result=result,
            req=_request(),
            job_dir=tmp_path,
        )


def test_write_refraction_static_artifacts_detects_missing_artifact_after_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifact_module,
        'write_refraction_static_components_csv',
        lambda **_kwargs: None,
    )

    with pytest.raises(
        RefractionStaticArtifactError,
        match='artifact file missing after write: refraction_static_components.csv',
    ):
        write_refraction_static_artifacts(
            result=_result(),
            req=_request(),
            job_dir=tmp_path,
        )


def test_write_refraction_static_solution_rejects_object_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifact_module,
        'build_refraction_static_solution_arrays',
        lambda **_kwargs: {'bad_object': np.asarray([object()], dtype=object)},
    )

    with pytest.raises(
        RefractionStaticArtifactError,
        match='object array is not allowed for bad_object',
    ):
        write_refraction_static_solution_npz(
            result=_result(),
            req=_request(),
            path=tmp_path / REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        )


def _solve_cell_request():
    payload = _request().model_dump(mode='json')
    payload['model'].update(
        {
            'bedrock_velocity_mode': 'solve_cell',
            'initial_bedrock_velocity_m_s': 2500.0,
            'min_bedrock_velocity_m_s': 1200.0,
            'max_bedrock_velocity_m_s': 6000.0,
            'refractor_cell': {
                'number_of_cell_x': 3,
                'size_of_cell_x_m': 100.0,
                'x_coordinate_origin_m': 0.0,
                'number_of_cell_y': 1,
                'size_of_cell_y_m': None,
                'assignment_mode': 'midpoint',
                'outside_grid_policy': 'reject',
                'min_observations_per_cell': 1,
                'velocity_smoothing_weight': 1.5,
                'smoothing_reference_distance_m': 200.0,
            },
        }
    )
    return RefractionStaticApplyRequest.model_validate(payload)


def _v3_solve_cell_request():
    payload = _request().model_dump(mode='json')
    payload['model'] = {
        'method': 'multilayer_time_term',
        'first_layer': {
            'mode': 'constant',
            'weathering_velocity_m_s': 800.0,
        },
        'refractor_cell': {
            'number_of_cell_x': 3,
            'size_of_cell_x_m': 100.0,
            'x_coordinate_origin_m': 0.0,
            'number_of_cell_y': 1,
            'size_of_cell_y_m': None,
            'assignment_mode': 'midpoint',
            'outside_grid_policy': 'reject',
            'min_observations_per_cell': 5,
            'velocity_smoothing_weight': 0.0,
            'smoothing_reference_distance_m': None,
        },
        'layers': [
            {
                'kind': 'v2_t1',
                'enabled': True,
                'min_offset_m': 0.0,
                'max_offset_m': 150.0,
                'velocity_mode': 'fixed_global',
                'fixed_velocity_m_s': 2400.0,
                'min_velocity_m_s': 1600.0,
                'max_velocity_m_s': 3200.0,
            },
            {
                'kind': 'v3_t2',
                'enabled': True,
                'min_offset_m': 150.0,
                'max_offset_m': None,
                'velocity_mode': 'solve_cell',
                'initial_velocity_m_s': 3600.0,
                'min_velocity_m_s': 3000.0,
                'max_velocity_m_s': 4500.0,
                'min_observations_per_cell': 1,
                'smoothing_weight': 1.25,
            },
        ],
    }
    return RefractionStaticApplyRequest.model_validate(payload)


def _v3_vsub_solve_cell_request():
    payload = _v3_solve_cell_request().model_dump(mode='json')
    payload['model']['layers'] = [
        {
            'kind': 'v2_t1',
            'enabled': True,
            'min_offset_m': 0.0,
            'max_offset_m': 150.0,
            'velocity_mode': 'fixed_global',
            'fixed_velocity_m_s': 2400.0,
            'min_velocity_m_s': 1600.0,
            'max_velocity_m_s': 3200.0,
        },
        {
            'kind': 'v3_t2',
            'enabled': True,
            'min_offset_m': 150.0,
            'max_offset_m': 300.0,
            'velocity_mode': 'solve_cell',
            'initial_velocity_m_s': 3600.0,
            'min_velocity_m_s': 3000.0,
            'max_velocity_m_s': 4500.0,
            'min_observations_per_cell': 1,
            'smoothing_weight': 1.25,
        },
        {
            'kind': 'vsub_t3',
            'enabled': True,
            'min_offset_m': 300.0,
            'max_offset_m': None,
            'velocity_mode': 'solve_cell',
            'initial_velocity_m_s': 5000.0,
            'min_velocity_m_s': 4200.0,
            'max_velocity_m_s': 6200.0,
            'min_observations_per_cell': 1,
            'smoothing_weight': 0.75,
        },
    ]
    return RefractionStaticApplyRequest.model_validate(payload)


def _solve_cell_result():
    return replace(
        _result(),
        bedrock_velocity_mode='solve_cell',
        bedrock_velocity_m_s=2500.0,
        bedrock_slowness_s_per_m=1.0 / 2500.0,
        active_cell_id=np.asarray([0, 1], dtype=np.int64),
        inactive_cell_id=np.asarray([2], dtype=np.int64),
        cell_bedrock_slowness_s_per_m=np.asarray(
            [1.0 / 2400.0, 1.0 / 2600.0],
            dtype=np.float64,
        ),
        cell_bedrock_velocity_m_s=np.asarray([2400.0, 2600.0], dtype=np.float64),
        cell_velocity_status=np.asarray(['solved', 'solved'], dtype='<U16'),
        row_midpoint_cell_id=np.asarray([0, 1, 0], dtype=np.int64),
        node_v2_cell_id=np.asarray([0, 1, 2], dtype=np.int64),
        node_v2_m_s=np.asarray([2400.0, 2600.0, np.nan], dtype=np.float64),
        node_v2_status=np.asarray(
            ['ok', 'ok', 'inactive_v2_cell'],
            dtype='<U32',
        ),
        source_v2_cell_id=np.asarray([0, 1], dtype=np.int64),
        source_v2_m_s=np.asarray([2400.0, 2600.0], dtype=np.float64),
        source_v2_status=np.asarray(['ok', 'ok'], dtype='<U2'),
        receiver_v2_cell_id=np.asarray([1, 2], dtype=np.int64),
        receiver_v2_m_s=np.asarray([2600.0, np.nan], dtype=np.float64),
        receiver_v2_status=np.asarray(['ok', 'inactive_v2_cell'], dtype='<U32'),
        source_v2_cell_id_sorted=np.asarray([0, 1, 0, 1], dtype=np.int64),
        source_v2_m_s_sorted=np.asarray(
            [2400.0, 2600.0, 2400.0, 2600.0],
            dtype=np.float64,
        ),
        source_v2_status_sorted=np.asarray(['ok', 'ok', 'ok', 'ok'], dtype='<U2'),
        receiver_v2_cell_id_sorted=np.asarray([1, 2, 2, 1], dtype=np.int64),
        receiver_v2_m_s_sorted=np.asarray(
            [2600.0, np.nan, np.nan, 2600.0],
            dtype=np.float64,
        ),
        receiver_v2_status_sorted=np.asarray(
            ['ok', 'inactive_v2_cell', 'inactive_v2_cell', 'ok'],
            dtype='<U32',
        ),
    )


def _multilayer_shaped_solve_cell_result():
    base = _solve_cell_result()
    return replace(
        base,
        row_trace_index_sorted=np.arange(4, dtype=np.int64),
        row_source_node_id=np.asarray([0, 1, 0, 1], dtype=np.int64),
        row_receiver_node_id=np.asarray([1, 2, 2, 1], dtype=np.int64),
        row_distance_m=np.asarray([100.0, 200.0, 300.0, 400.0]),
        observed_pick_time_s=np.asarray([0.050, 0.060, 0.070, 0.080]),
        modeled_pick_time_s=np.asarray([0.049, 0.062, 0.071, 0.079]),
        residual_time_s=np.asarray([0.001, -0.002, -0.001, 0.001]),
        used_row_mask=np.asarray([True, False, True, False], dtype=bool),
        rejected_by_robust_mask=np.asarray([False, True, False, False], dtype=bool),
        row_midpoint_cell_id=np.asarray([0, 1, 0, 1], dtype=np.int64),
        row_layer_kind=np.asarray(['v2_t1', 'v2_t1', 'v2_t1', 'v2_t1'], dtype='<U16'),
        row_layer_index=np.ones(4, dtype=np.int64),
        row_source_endpoint_key=np.asarray(['s0', 's1', 's0', 's1'], dtype='<U16'),
        row_receiver_endpoint_key=np.asarray(['r0', 'r1', 'r1', 'r0'], dtype='<U16'),
        row_rejection_reason=np.asarray(['ok', '', 'ok', 'not_used'], dtype='<U32'),
        row_velocity_m_s=np.full(4, 2500.0, dtype=np.float64),
    )


def _multilayer_result_with_v3_and_vsub_cell_layers():
    base = _multilayer_shaped_solve_cell_result()
    return replace(
        base,
        bedrock_velocity_mode='fixed_global',
        layer_results=(
            _cell_layer_result(
                layer_kind='v3_t2',
                layer_index=2,
                velocity_m_s=np.asarray([3300.0, 3700.0, np.nan], dtype=np.float64),
                used_mask=np.asarray([False, True, True, False], dtype=bool),
                residual_s=np.asarray([np.nan, 0.001, -0.002, np.nan]),
                row_midpoint_cell_id=np.asarray([0, 1, 0, 1], dtype=np.int64),
                smoothing_weight=1.25,
            ),
            _cell_layer_result(
                layer_kind='vsub_t3',
                layer_index=3,
                velocity_m_s=np.asarray([np.nan, 4800.0, 5200.0], dtype=np.float64),
                used_mask=np.asarray([False, False, False, True], dtype=bool),
                residual_s=np.asarray([np.nan, np.nan, np.nan, 0.003]),
                row_midpoint_cell_id=np.asarray([1, 2, 1, 2], dtype=np.int64),
                smoothing_weight=0.75,
            ),
        ),
    )


def _cell_layer_result(
    *,
    layer_kind: str,
    layer_index: int,
    velocity_m_s: np.ndarray,
    used_mask: np.ndarray,
    residual_s: np.ndarray,
    row_midpoint_cell_id: np.ndarray,
    smoothing_weight: float,
) -> RefractionLayerSolveResult:
    active_cell_id = np.flatnonzero(np.isfinite(velocity_m_s)).astype(np.int64)
    inactive_cell_id = np.flatnonzero(~np.isfinite(velocity_m_s)).astype(np.int64)
    return RefractionLayerSolveResult(
        layer_kind=layer_kind,
        layer_index=layer_index,
        velocity_mode='solve_cell',
        source_time_term_s=np.zeros(2, dtype=np.float64),
        receiver_time_term_s=np.zeros(2, dtype=np.float64),
        node_time_term_s=np.zeros(3, dtype=np.float64),
        global_velocity_m_s=None,
        global_slowness_s_per_m=None,
        cell_velocity_m_s=velocity_m_s,
        cell_slowness_s_per_m=1.0 / velocity_m_s,
        trace_predicted_time_s_sorted=np.asarray(
            [0.050, 0.060, 0.070, 0.080],
            dtype=np.float64,
        ),
        trace_residual_s_sorted=residual_s,
        used_observation_mask_sorted=used_mask,
        layer_status='solved',
        qc={
            'layer_kind': layer_kind,
            'layer_index': layer_index,
            'velocity_mode': 'solve_cell',
            'n_total_cells': 3,
            'min_observations_per_cell': 1,
            'n_low_fold_cells': 0,
            'n_observations_rejected_by_low_fold_cell': 0,
            'n_observations_outside_grid': 0,
            'low_fold_cell_id': [],
            'cell_observation_count': [1, 2, 1],
            'n_cell_smoothing_rows': 1,
            'velocity_smoothing_weight': smoothing_weight,
        },
        active_cell_id=active_cell_id,
        inactive_cell_id=inactive_cell_id,
        cell_velocity_status=np.asarray(['solved'] * active_cell_id.size, dtype='<U16'),
        row_midpoint_cell_id=row_midpoint_cell_id,
        row_midpoint_velocity_m_s=velocity_m_s[row_midpoint_cell_id],
        rejected_by_robust_mask_sorted=np.zeros(4, dtype=bool),
    )


def _solve_cell_low_fold_result():
    base = _solve_cell_result()
    qc = dict(base.qc)
    qc.update(
        {
            'min_observations_per_cell': 2,
            'n_low_fold_cells': 1,
            'n_observations_rejected_by_low_fold_cell': 1,
            'low_fold_cell_rejection_reason': 'below_min_observations_per_cell',
            'low_fold_cell_id': [1],
            'cell_observation_count': [2, 1, 0],
        }
    )
    return replace(
        base,
        active_cell_id=np.asarray([0], dtype=np.int64),
        inactive_cell_id=np.asarray([1, 2], dtype=np.int64),
        cell_bedrock_slowness_s_per_m=np.asarray([1.0 / 2400.0], dtype=np.float64),
        cell_bedrock_velocity_m_s=np.asarray([2400.0], dtype=np.float64),
        cell_velocity_status=np.asarray(['solved'], dtype='<U16'),
        row_midpoint_cell_id=np.asarray([0, 0, 0], dtype=np.int64),
        node_v2_cell_id=np.asarray([0, 1, 2], dtype=np.int64),
        node_v2_m_s=np.asarray([2400.0, np.nan, np.nan], dtype=np.float64),
        node_v2_status=np.asarray(
            ['ok', 'low_fold_v2_cell', 'inactive_v2_cell'],
            dtype='<U32',
        ),
        node_datum_status=np.asarray(
            ['ok', 'low_fold_v2_cell', 'inactive_v2_cell'],
            dtype='<U32',
        ),
        source_v2_cell_id=np.asarray([0, 1], dtype=np.int64),
        source_v2_m_s=np.asarray([2400.0, np.nan], dtype=np.float64),
        source_v2_status=np.asarray(['ok', 'low_fold_v2_cell'], dtype='<U32'),
        source_datum_status=np.asarray(['ok', 'low_fold_v2_cell'], dtype='<U32'),
        source_refraction_shift_s=np.asarray([0.0025, np.nan], dtype=np.float64),
        receiver_v2_cell_id=np.asarray([1, 2], dtype=np.int64),
        receiver_v2_m_s=np.asarray([np.nan, np.nan], dtype=np.float64),
        receiver_v2_status=np.asarray(
            ['low_fold_v2_cell', 'inactive_v2_cell'],
            dtype='<U32',
        ),
        receiver_datum_status=np.asarray(
            ['low_fold_v2_cell', 'inactive_v2_cell'],
            dtype='<U32',
        ),
        receiver_refraction_shift_s=np.asarray([np.nan, np.nan], dtype=np.float64),
        source_v2_cell_id_sorted=np.asarray([0, 1, 0, 1], dtype=np.int64),
        source_v2_m_s_sorted=np.asarray(
            [2400.0, np.nan, 2400.0, np.nan],
            dtype=np.float64,
        ),
        source_v2_status_sorted=np.asarray(
            ['ok', 'low_fold_v2_cell', 'ok', 'low_fold_v2_cell'],
            dtype='<U32',
        ),
        receiver_v2_cell_id_sorted=np.asarray([1, 2, 2, 1], dtype=np.int64),
        receiver_v2_m_s_sorted=np.asarray(
            [np.nan, np.nan, np.nan, np.nan],
            dtype=np.float64,
        ),
        receiver_v2_status_sorted=np.asarray(
            [
                'low_fold_v2_cell',
                'inactive_v2_cell',
                'inactive_v2_cell',
                'low_fold_v2_cell',
            ],
            dtype='<U32',
        ),
        receiver_refraction_shift_s_sorted=np.asarray(
            [np.nan, np.nan, np.nan, np.nan],
            dtype=np.float64,
        ),
        refraction_trace_shift_s_sorted=np.asarray(
            [np.nan, np.nan, np.nan, np.nan],
            dtype=np.float64,
        ),
        trace_static_status_sorted=np.asarray(
            [
                'low_fold_v2_cell',
                'inactive_v2_cell',
                'inactive_v2_cell',
                'low_fold_v2_cell',
            ],
            dtype='<U32',
        ),
        trace_static_valid_mask_sorted=np.asarray(
            [False, False, False, False],
            dtype=bool,
        ),
        qc=qc,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _read_csv_with_fieldnames(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, str):
        return value.startswith('/')
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    return False


def _content_type_for_name(name: str) -> str:
    if name.endswith('.csv'):
        return 'text/csv'
    if name.endswith('.json'):
        return 'application/json'
    if name.endswith('.npz'):
        return 'application/octet-stream'
    raise AssertionError(f'unhandled artifact extension: {name}')


def _write_upstream_v1_artifacts(root: Path) -> None:
    (root / REFRACTION_V1_QC_JSON_NAME).write_text(
        '{"v1_status":"estimated"}',
        encoding='utf-8',
    )
    (root / REFRACTION_V1_ESTIMATES_CSV_NAME).write_text(
        'group_kind,group_key,status\nsource_endpoint,source:1,ok\n',
        encoding='utf-8',
    )
