"""Public facade for refraction static artifact helpers."""

from __future__ import annotations

from app.statics.refraction.artifacts import contract
from app.statics.refraction.artifacts import writer
from app.statics.refraction.artifacts.writer import *  # noqa: F403
from app.statics.refraction.artifacts.components import (
    build_refraction_static_component_qc_arrays as build_refraction_static_component_qc_arrays,
    build_refraction_static_component_qc_payload as build_refraction_static_component_qc_payload,
    write_refraction_static_component_qc_artifacts as write_refraction_static_component_qc_artifacts,
    write_refraction_static_components_csv as write_refraction_static_components_csv,
)
from app.statics.refraction.artifacts.cell_velocity import (
    build_refraction_cell_solver_history_rows as build_refraction_cell_solver_history_rows,
    build_refraction_refractor_velocity_grid_arrays as build_refraction_refractor_velocity_grid_arrays,
    build_refraction_refractor_velocity_qc_payload as build_refraction_refractor_velocity_qc_payload,
    write_refraction_cell_solver_history_csv as write_refraction_cell_solver_history_csv,
    write_refraction_refractor_velocity_cells_csv as write_refraction_refractor_velocity_cells_csv,
    write_refraction_refractor_velocity_grid_npz as write_refraction_refractor_velocity_grid_npz,
    write_refraction_refractor_velocity_qc_json as write_refraction_refractor_velocity_qc_json,
)
from app.statics.refraction.artifacts.grid_map import (
    build_refraction_grid_map_qc_arrays as build_refraction_grid_map_qc_arrays,
    build_refraction_grid_map_qc_payload as build_refraction_grid_map_qc_payload,
    write_refraction_grid_map_qc_csv as write_refraction_grid_map_qc_csv,
    write_refraction_grid_map_qc_json as write_refraction_grid_map_qc_json,
    write_refraction_grid_map_qc_npz as write_refraction_grid_map_qc_npz,
)
from app.statics.refraction.artifacts.line_profile import (
    build_refraction_line_profile_qc_arrays as build_refraction_line_profile_qc_arrays,
    build_refraction_line_profile_qc_payload as build_refraction_line_profile_qc_payload,
    write_refraction_line_profile_qc_artifacts as write_refraction_line_profile_qc_artifacts,
)

for _name in contract.__all__:
    globals()[_name] = getattr(contract, _name)

FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION = (
    contract.FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION
)
SIGN_CONVENTION = contract.SIGN_CONVENTION
TIME_TERM_SPREADSHEET_FORMAT_NAME = contract.TIME_TERM_SPREADSHEET_FORMAT_NAME
TIME_TERM_SPREADSHEET_FORMAT_VERSION = contract.TIME_TERM_SPREADSHEET_FORMAT_VERSION
TIME_TERM_SPREADSHEET_SCHEMA_VERSION = contract.TIME_TERM_SPREADSHEET_SCHEMA_VERSION

__all__ = list(writer.__all__)
