"""Public facade for refraction static artifact helpers."""

from __future__ import annotations

from app.services.refraction_static_artifacts import contract
from app.services.refraction_static_artifacts import _legacy
from app.services.refraction_static_artifacts._legacy import *  # noqa: F403
from app.services.refraction_static_artifacts.cell_velocity import (
    build_refraction_cell_solver_history_rows as build_refraction_cell_solver_history_rows,
    build_refraction_refractor_velocity_grid_arrays as build_refraction_refractor_velocity_grid_arrays,
    build_refraction_refractor_velocity_qc_payload as build_refraction_refractor_velocity_qc_payload,
    write_refraction_cell_solver_history_csv as write_refraction_cell_solver_history_csv,
    write_refraction_refractor_velocity_cells_csv as write_refraction_refractor_velocity_cells_csv,
    write_refraction_refractor_velocity_grid_npz as write_refraction_refractor_velocity_grid_npz,
    write_refraction_refractor_velocity_qc_json as write_refraction_refractor_velocity_qc_json,
)
from app.services.refraction_static_artifacts.grid_map import (
    build_refraction_grid_map_qc_arrays as build_refraction_grid_map_qc_arrays,
    build_refraction_grid_map_qc_payload as build_refraction_grid_map_qc_payload,
    write_refraction_grid_map_qc_csv as write_refraction_grid_map_qc_csv,
    write_refraction_grid_map_qc_json as write_refraction_grid_map_qc_json,
    write_refraction_grid_map_qc_npz as write_refraction_grid_map_qc_npz,
)
from app.services.refraction_static_artifacts.line_profile import (
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

__all__ = list(_legacy.__all__)
