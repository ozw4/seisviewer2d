"""T1LSST-compatible refraction static artifacts."""

from __future__ import annotations

from pathlib import Path

from app.services.common.artifact_io import write_csv_atomic
from app.statics.refraction.domain.t1lsst import (
    compose_t1lsst_1layer_static_table_components,
)
from app.statics.refraction.domain.types import RefractionDatumStaticsResult

REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME = (
    'refraction_t1lsst_1layer_components.csv'
)

_T1LSST_COMPONENT_COLUMNS = (
    'endpoint_kind',
    'endpoint_key',
    'node_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    'v1_m_s',
    'v2_m_s',
    'sh1_weathering_thickness_m',
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
)


def write_refraction_t1lsst_1layer_components_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    """Write the T1LSST-compatible one-layer component CSV artifact."""
    rows = compose_t1lsst_1layer_static_table_components(result)
    write_csv_atomic(
        Path(path),
        columns=_T1LSST_COMPONENT_COLUMNS,
        rows=rows,
        lineterminator='\r\n',
    )


__all__ = [
    'REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME',
    'write_refraction_t1lsst_1layer_components_csv',
]
