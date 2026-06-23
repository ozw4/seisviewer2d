"""Dependency-light data types for M5 refraction static exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from app.statics.refraction.artifacts.export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)

RefractionStaticEndpointKind = Literal['source', 'receiver']
RefractionStaticExportFormatName = Literal[
    'canonical_static_table',
    'lsst',
    'lsst_plus',
    'time_term_spreadsheet',
    'first_break_time',
]
RefractionStaticSignConvention = Literal['corrected(t) = raw(t - shift_s)']

REFRACTION_STATIC_EXPORT_SIGN_CONVENTION: Final[RefractionStaticSignConvention] = (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION
)
REFRACTION_STATIC_EXPORT_UNITS: Final[str] = (
    'seconds_internal; milliseconds_csv; meters; meters_per_second'
)


@dataclass(frozen=True)
class RefractionStaticEndpointExportRow:
    """Endpoint row normalized for dependency-light export formatters."""

    endpoint_kind: RefractionStaticEndpointKind
    endpoint_key: str
    endpoint_id: str | int | None = None
    station_id: str | int | None = None
    node_id: int | None = None
    x_m: float | None = None
    y_m: float | None = None
    elevation_m: float | None = None
    t1_s: float | None = None
    t2_s: float | None = None
    t3_s: float | None = None
    v1_m_s: float | None = None
    v2_m_s: float | None = None
    v3_m_s: float | None = None
    vsub_m_s: float | None = None
    sh1_m: float | None = None
    sh2_m: float | None = None
    sh3_m: float | None = None
    total_weathering_thickness_m: float | None = None
    weathering_correction_s: float | None = None
    elevation_correction_s: float | None = None
    field_correction_s: float | None = None
    source_depth_m: float | None = None
    source_depth_shift_s: float | None = None
    source_depth_status: str | None = None
    uphole_time_s: float | None = None
    uphole_shift_s: float | None = None
    uphole_status: str | None = None
    manual_static_shift_s: float | None = None
    manual_static_status: str | None = None
    field_static_status: str | None = None
    total_with_field_shift_s: float | None = None
    total_applied_shift_s: float | None = None
    static_status: str = 'ok'


@dataclass(frozen=True)
class RefractionStaticExportBundle:
    """Source and receiver rows plus export-wide metadata."""

    source_rows: tuple[RefractionStaticEndpointExportRow, ...] = ()
    receiver_rows: tuple[RefractionStaticEndpointExportRow, ...] = ()
    source_job_id: str | None = None
    sign_convention: str = REFRACTION_STATIC_EXPORT_SIGN_CONVENTION
    units: str = REFRACTION_STATIC_EXPORT_UNITS


@dataclass(frozen=True)
class RefractionStaticCanonicalTableRow:
    """M5 canonical static-table row for validated import/apply workflows."""

    endpoint_kind: RefractionStaticEndpointKind
    endpoint_key: str
    endpoint_id: str | int | None
    applied_shift_ms: float
    static_status: str
    format_name: Literal['canonical_static_table'] = 'canonical_static_table'
    format_version: int = 1
    source_job_id: str | None = None
    sign_convention: str = REFRACTION_STATIC_EXPORT_SIGN_CONVENTION
    x_m: float | None = None
    y_m: float | None = None
    source_id: str | int | None = None
    receiver_id: str | int | None = None
    node_id: int | None = None
    total_static_ms: float | None = None
    total_applied_shift_ms: float | None = None
    source_field_shift_ms: float | None = None
    receiver_field_shift_ms: float | None = None
    source_total_with_field_shift_ms: float | None = None
    receiver_total_with_field_shift_ms: float | None = None
    manual_static_shift_ms: float | None = None
    source_depth_shift_ms: float | None = None
    uphole_shift_ms: float | None = None
    t1_ms: float | None = None
    t2_ms: float | None = None
    t3_ms: float | None = None
    v1_m_s: float | None = None
    v2_m_s: float | None = None
    v3_m_s: float | None = None
    vsub_m_s: float | None = None
    sh1_weathering_thickness_m: float | None = None
    sh2_weathering_thickness_m: float | None = None
    sh3_weathering_thickness_m: float | None = None
    weathering_correction_ms: float | None = None
    elevation_correction_ms: float | None = None
    comment: str | None = None


__all__ = [
    'REFRACTION_STATIC_EXPORT_SIGN_CONVENTION',
    'REFRACTION_STATIC_EXPORT_UNITS',
    'RefractionStaticCanonicalTableRow',
    'RefractionStaticEndpointExportRow',
    'RefractionStaticEndpointKind',
    'RefractionStaticExportBundle',
    'RefractionStaticExportFormatName',
    'RefractionStaticSignConvention',
]
