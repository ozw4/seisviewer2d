"""Dependency-light LSST formatters for M5 refraction static exports."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from app.services.refraction_static_export_types import (
    RefractionStaticEndpointExportRow,
    RefractionStaticExportBundle,
)
from app.services.refraction_static_export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
    seconds_to_export_units,
)

REFRACTION_LSST_CSV_NAME: Final = 'refraction_lsst.csv'
REFRACTION_LSST_FORMAT_NAME: Final = 'lsst'
REFRACTION_LSST_FORMAT_VERSION: Final = 1
REFRACTION_LSST_PLUS_CSV_NAME: Final = 'refraction_lsst_plus.csv'
REFRACTION_LSST_PLUS_FORMAT_NAME: Final = 'lsst_plus'
REFRACTION_LSST_PLUS_FORMAT_VERSION: Final = 1

REFRACTION_LSST_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    'format_name',
    'format_version',
    'source_job_id',
    'endpoint_kind',
    'endpoint_key',
    'endpoint_id',
    'node_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    't1_ms',
    'v1_m_s',
    'v2_m_s',
    'sh1_weathering_thickness_m',
    'weathering_correction_ms',
    'elevation_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'static_status',
    'sign_convention',
)
REFRACTION_LSST_OPTIONAL_MULTILAYER_COLUMNS: Final[tuple[str, ...]] = (
    't2_ms',
    't3_ms',
    'v3_m_s',
    'vsub_m_s',
    'sh2_weathering_thickness_m',
    'sh3_weathering_thickness_m',
    'total_weathering_thickness_m',
)
REFRACTION_LSST_PLUS_FIELD_COLUMNS: Final[tuple[str, ...]] = (
    'source_depth_m',
    'source_depth_shift_ms',
    'source_depth_status',
    'uphole_time_ms',
    'uphole_shift_ms',
    'uphole_status',
    'manual_static_shift_ms',
    'manual_static_status',
    'source_field_shift_ms',
    'source_field_static_status',
    'source_total_with_field_shift_ms',
    'receiver_field_shift_ms',
    'receiver_field_static_status',
    'receiver_total_with_field_shift_ms',
)
REFRACTION_LSST_PLUS_COLUMNS: Final[tuple[str, ...]] = (
    REFRACTION_LSST_REQUIRED_COLUMNS
    + REFRACTION_LSST_OPTIONAL_MULTILAYER_COLUMNS
    + REFRACTION_LSST_PLUS_FIELD_COLUMNS
)


class RefractionStaticLsstExportError(ValueError):
    """Raised when LSST output cannot be produced from endpoint rows."""


def format_refraction_lsst_csv(
    bundle: RefractionStaticExportBundle,
    *,
    fail_on_invalid_static_status: bool = True,
    include_inactive_endpoints: bool = False,
) -> str:
    """Format source and receiver endpoint rows as the documented M5 LSST CSV."""
    source_job_id = _validate_bundle(bundle)
    rows = _included_rows(
        bundle,
        fail_on_invalid_static_status=fail_on_invalid_static_status,
        include_inactive_endpoints=include_inactive_endpoints,
    )
    columns = _columns_for_rows(rows)
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(columns), lineterminator='\n')
    writer.writeheader()
    writer.writerows(
        _csv_row(row, source_job_id=source_job_id, columns=columns)
        for row in rows
    )
    return output.getvalue()


def write_refraction_lsst_csv(
    bundle: RefractionStaticExportBundle,
    path: Path,
    *,
    fail_on_invalid_static_status: bool = True,
    include_inactive_endpoints: bool = False,
) -> None:
    """Write documented M5 LSST CSV text to ``path``."""
    text = format_refraction_lsst_csv(
        bundle,
        fail_on_invalid_static_status=fail_on_invalid_static_status,
        include_inactive_endpoints=include_inactive_endpoints,
    )
    Path(path).write_text(text, encoding='utf-8')


def format_refraction_lsst_plus_csv(
    bundle: RefractionStaticExportBundle,
    *,
    fail_on_invalid_static_status: bool = False,
    include_inactive_endpoints: bool = True,
) -> str:
    """Format endpoint rows as the documented M5 LSST+ spreadsheet CSV."""
    source_job_id = _validate_bundle(bundle)
    rows = _included_rows(
        bundle,
        fail_on_invalid_static_status=fail_on_invalid_static_status,
        include_inactive_endpoints=include_inactive_endpoints,
    )
    output = io.StringIO(newline='')
    writer = csv.DictWriter(
        output,
        fieldnames=list(REFRACTION_LSST_PLUS_COLUMNS),
        lineterminator='\n',
    )
    writer.writeheader()
    writer.writerows(
        _lsst_plus_csv_row(row, source_job_id=source_job_id) for row in rows
    )
    return output.getvalue()


def write_refraction_lsst_plus_csv(
    bundle: RefractionStaticExportBundle,
    path: Path,
    *,
    fail_on_invalid_static_status: bool = False,
    include_inactive_endpoints: bool = True,
) -> None:
    """Write documented M5 LSST+ CSV text to ``path``."""
    text = format_refraction_lsst_plus_csv(
        bundle,
        fail_on_invalid_static_status=fail_on_invalid_static_status,
        include_inactive_endpoints=include_inactive_endpoints,
    )
    Path(path).write_text(text, encoding='utf-8')


def _validate_bundle(bundle: RefractionStaticExportBundle) -> str:
    if bundle.sign_convention != REFRACTION_STATIC_REPO_SIGN_CONVENTION:
        raise RefractionStaticLsstExportError(
            'LSST export sign_convention must be '
            f'{REFRACTION_STATIC_REPO_SIGN_CONVENTION!r}'
        )
    source_job_id = _optional_text(bundle.source_job_id)
    if source_job_id is None:
        raise RefractionStaticLsstExportError('LSST export requires source_job_id')
    return source_job_id


def _included_rows(
    bundle: RefractionStaticExportBundle,
    *,
    fail_on_invalid_static_status: bool,
    include_inactive_endpoints: bool,
) -> tuple[RefractionStaticEndpointExportRow, ...]:
    rows: list[RefractionStaticEndpointExportRow] = []
    for row in bundle.source_rows:
        rows.extend(
            _include_row(
                row,
                expected_kind='source',
                fail_on_invalid_static_status=fail_on_invalid_static_status,
                include_inactive_endpoints=include_inactive_endpoints,
            )
        )
    for row in bundle.receiver_rows:
        rows.extend(
            _include_row(
                row,
                expected_kind='receiver',
                fail_on_invalid_static_status=fail_on_invalid_static_status,
                include_inactive_endpoints=include_inactive_endpoints,
            )
        )
    return tuple(rows)


def _include_row(
    row: RefractionStaticEndpointExportRow,
    *,
    expected_kind: str,
    fail_on_invalid_static_status: bool,
    include_inactive_endpoints: bool,
) -> tuple[RefractionStaticEndpointExportRow, ...]:
    if row.endpoint_kind != expected_kind:
        raise RefractionStaticLsstExportError(
            f'{expected_kind} LSST export received {row.endpoint_kind!r} row'
        )
    status = _required_static_status(row)
    if status == 'ok':
        return (row,)
    if fail_on_invalid_static_status:
        raise RefractionStaticLsstExportError(
            f'{row.endpoint_kind} endpoint {row.endpoint_key!r} has '
            f'invalid static_status {status!r}'
        )
    if include_inactive_endpoints:
        return (row,)
    return ()


def _columns_for_rows(
    rows: Sequence[RefractionStaticEndpointExportRow],
) -> tuple[str, ...]:
    optional_columns = tuple(
        column
        for column in REFRACTION_LSST_OPTIONAL_MULTILAYER_COLUMNS
        if any(_row_has_value(row, column) for row in rows)
    )
    return REFRACTION_LSST_REQUIRED_COLUMNS + optional_columns


def _csv_row(
    row: RefractionStaticEndpointExportRow,
    *,
    source_job_id: str,
    columns: Sequence[str],
    format_name: str = REFRACTION_LSST_FORMAT_NAME,
    format_version: int = REFRACTION_LSST_FORMAT_VERSION,
) -> dict[str, str]:
    status = _required_static_status(row)
    valid = status == 'ok'
    out = {
        'format_name': format_name,
        'format_version': str(format_version),
        'source_job_id': source_job_id,
        'endpoint_kind': row.endpoint_kind,
        'endpoint_key': _format_text(
            row.endpoint_key,
            required=True,
            row=row,
            field_name='endpoint_key',
        ),
        'endpoint_id': _format_text(
            row.endpoint_id,
            required=valid,
            row=row,
            field_name='endpoint_id',
        ),
        'node_id': _format_int(
            row.node_id,
            required=valid,
            row=row,
            field_name='node_id',
        ),
        'x_m': _format_m(row.x_m, required=valid, row=row, field_name='x_m'),
        'y_m': _format_m(row.y_m, required=valid, row=row, field_name='y_m'),
        'surface_elevation_m': _format_m(
            row.elevation_m,
            required=valid,
            row=row,
            field_name='surface_elevation_m',
        ),
        't1_ms': _format_time_ms(
            row.t1_s,
            required=valid,
            row=row,
            field_name='t1_ms',
        ),
        'v1_m_s': _format_velocity(
            row.v1_m_s,
            required=valid,
            row=row,
            field_name='v1_m_s',
        ),
        'v2_m_s': _format_velocity(
            row.v2_m_s,
            required=valid,
            row=row,
            field_name='v2_m_s',
        ),
        'sh1_weathering_thickness_m': _format_m(
            row.sh1_m,
            required=valid,
            row=row,
            field_name='sh1_weathering_thickness_m',
        ),
        'weathering_correction_ms': _format_time_ms(
            row.weathering_correction_s,
            required=valid,
            row=row,
            field_name='weathering_correction_ms',
        ),
        'elevation_correction_ms': _format_time_ms(
            row.elevation_correction_s,
            required=valid,
            row=row,
            field_name='elevation_correction_ms',
        ),
        'total_static_ms': _format_time_ms(
            row.total_applied_shift_s,
            required=valid,
            row=row,
            field_name='total_static_ms',
        ),
        'total_applied_shift_ms': _format_time_ms(
            row.total_applied_shift_s,
            required=valid,
            row=row,
            field_name='total_applied_shift_ms',
        ),
        'static_status': status,
        'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
    }
    optional_values = {
        't2_ms': _format_time_ms(
            row.t2_s,
            required=False,
            row=row,
            field_name='t2_ms',
        ),
        't3_ms': _format_time_ms(
            row.t3_s,
            required=False,
            row=row,
            field_name='t3_ms',
        ),
        'v3_m_s': _format_velocity(
            row.v3_m_s,
            required=False,
            row=row,
            field_name='v3_m_s',
        ),
        'vsub_m_s': _format_velocity(
            row.vsub_m_s,
            required=False,
            row=row,
            field_name='vsub_m_s',
        ),
        'sh2_weathering_thickness_m': _format_m(
            row.sh2_m,
            required=False,
            row=row,
            field_name='sh2_weathering_thickness_m',
        ),
        'sh3_weathering_thickness_m': _format_m(
            row.sh3_m,
            required=False,
            row=row,
            field_name='sh3_weathering_thickness_m',
        ),
        'total_weathering_thickness_m': _format_m(
            row.total_weathering_thickness_m,
            required=False,
            row=row,
            field_name='total_weathering_thickness_m',
        ),
    }
    out.update(
        {column: optional_values[column] for column in columns if column in optional_values}
    )
    return out


def _lsst_plus_csv_row(
    row: RefractionStaticEndpointExportRow,
    *,
    source_job_id: str,
) -> dict[str, str]:
    out = _csv_row(
        row,
        source_job_id=source_job_id,
        columns=REFRACTION_LSST_PLUS_COLUMNS,
        format_name=REFRACTION_LSST_PLUS_FORMAT_NAME,
        format_version=REFRACTION_LSST_PLUS_FORMAT_VERSION,
    )
    out.update(
        {
            'source_depth_m': _format_m(
                row.source_depth_m,
                required=False,
                row=row,
                field_name='source_depth_m',
            ),
            'source_depth_shift_ms': _format_time_ms(
                row.source_depth_shift_s,
                required=False,
                row=row,
                field_name='source_depth_shift_ms',
            ),
            'source_depth_status': _format_text(
                row.source_depth_status,
                required=False,
                row=row,
                field_name='source_depth_status',
            ),
            'uphole_time_ms': _format_time_ms(
                row.uphole_time_s,
                required=False,
                row=row,
                field_name='uphole_time_ms',
            ),
            'uphole_shift_ms': _format_time_ms(
                row.uphole_shift_s,
                required=False,
                row=row,
                field_name='uphole_shift_ms',
            ),
            'uphole_status': _format_text(
                row.uphole_status,
                required=False,
                row=row,
                field_name='uphole_status',
            ),
            'manual_static_shift_ms': _format_time_ms(
                row.manual_static_shift_s,
                required=False,
                row=row,
                field_name='manual_static_shift_ms',
            ),
            'manual_static_status': _format_text(
                row.manual_static_status,
                required=False,
                row=row,
                field_name='manual_static_status',
            ),
            'source_field_shift_ms': '',
            'source_field_static_status': '',
            'source_total_with_field_shift_ms': '',
            'receiver_field_shift_ms': '',
            'receiver_field_static_status': '',
            'receiver_total_with_field_shift_ms': '',
        }
    )
    if row.endpoint_kind == 'source':
        out.update(
            {
                'source_field_shift_ms': _format_time_ms(
                    row.field_correction_s,
                    required=False,
                    row=row,
                    field_name='source_field_shift_ms',
                ),
                'source_field_static_status': _format_text(
                    row.field_static_status,
                    required=False,
                    row=row,
                    field_name='source_field_static_status',
                ),
                'source_total_with_field_shift_ms': _format_time_ms(
                    row.total_with_field_shift_s,
                    required=False,
                    row=row,
                    field_name='source_total_with_field_shift_ms',
                ),
            }
        )
    else:
        out.update(
            {
                'receiver_field_shift_ms': _format_time_ms(
                    row.field_correction_s,
                    required=False,
                    row=row,
                    field_name='receiver_field_shift_ms',
                ),
                'receiver_field_static_status': _format_text(
                    row.field_static_status,
                    required=False,
                    row=row,
                    field_name='receiver_field_static_status',
                ),
                'receiver_total_with_field_shift_ms': _format_time_ms(
                    row.total_with_field_shift_s,
                    required=False,
                    row=row,
                    field_name='receiver_total_with_field_shift_ms',
                ),
            }
        )
    return {column: out.get(column, '') for column in REFRACTION_LSST_PLUS_COLUMNS}


def _row_has_value(row: RefractionStaticEndpointExportRow, column: str) -> bool:
    value = {
        't2_ms': row.t2_s,
        't3_ms': row.t3_s,
        'v3_m_s': row.v3_m_s,
        'vsub_m_s': row.vsub_m_s,
        'sh2_weathering_thickness_m': row.sh2_m,
        'sh3_weathering_thickness_m': row.sh3_m,
        'total_weathering_thickness_m': row.total_weathering_thickness_m,
    }[column]
    numeric = _optional_float(value, row=row, field_name=column)
    return numeric is not None


def _format_text(
    value: object,
    *,
    required: bool,
    row: RefractionStaticEndpointExportRow,
    field_name: str,
) -> str:
    text = _optional_text(value)
    if text is not None:
        return text
    if required:
        _raise_missing(row, field_name)
    return ''


def _required_static_status(row: RefractionStaticEndpointExportRow) -> str:
    status = _optional_text(row.static_status)
    if status is None:
        raise RefractionStaticLsstExportError(
            f'{row.endpoint_kind} endpoint {row.endpoint_key!r} is missing '
            'static_status'
        )
    return status


def _format_int(
    value: object,
    *,
    required: bool,
    row: RefractionStaticEndpointExportRow,
    field_name: str,
) -> str:
    if value is None:
        if required:
            _raise_missing(row, field_name)
        return ''
    if isinstance(value, bool):
        raise RefractionStaticLsstExportError(
            f'{_row_label(row)} {field_name} must be an integer'
        )
    try:
        return str(int(value))
    except (TypeError, ValueError) as exc:
        raise RefractionStaticLsstExportError(
            f'{_row_label(row)} {field_name} must be an integer'
        ) from exc


def _format_time_ms(
    value_s: object,
    *,
    required: bool,
    row: RefractionStaticEndpointExportRow,
    field_name: str,
) -> str:
    value = _optional_float(value_s, row=row, field_name=field_name)
    if value is None:
        if required:
            _raise_missing(row, field_name)
        return ''
    return _format_fixed(
        seconds_to_export_units(value, 'milliseconds'),
        decimals=6,
    )


def _format_m(
    value: object,
    *,
    required: bool,
    row: RefractionStaticEndpointExportRow,
    field_name: str,
) -> str:
    return _format_numeric(
        value,
        decimals=3,
        required=required,
        row=row,
        field_name=field_name,
    )


def _format_velocity(
    value: object,
    *,
    required: bool,
    row: RefractionStaticEndpointExportRow,
    field_name: str,
) -> str:
    return _format_numeric(
        value,
        decimals=3,
        required=required,
        row=row,
        field_name=field_name,
    )


def _format_numeric(
    value: object,
    *,
    decimals: int,
    required: bool,
    row: RefractionStaticEndpointExportRow,
    field_name: str,
) -> str:
    numeric = _optional_float(value, row=row, field_name=field_name)
    if numeric is None:
        if required:
            _raise_missing(row, field_name)
        return ''
    return _format_fixed(numeric, decimals=decimals)


def _format_fixed(value: float, *, decimals: int) -> str:
    text = f'{value:.{decimals}f}'
    if text.startswith('-') and float(text) == 0.0:
        return text[1:]
    return text


def _optional_float(
    value: object,
    *,
    row: RefractionStaticEndpointExportRow,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticLsstExportError(
            f'{_row_label(row)} {field_name} must be numeric'
        ) from exc
    if not math.isfinite(numeric):
        return None
    return numeric


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _raise_missing(row: RefractionStaticEndpointExportRow, field_name: str) -> None:
    raise RefractionStaticLsstExportError(
        f'{_row_label(row)} is missing required LSST value {field_name}'
    )


def _row_label(row: RefractionStaticEndpointExportRow) -> str:
    return f'{row.endpoint_kind} endpoint {row.endpoint_key!r}'


__all__ = [
    'REFRACTION_LSST_CSV_NAME',
    'REFRACTION_LSST_FORMAT_NAME',
    'REFRACTION_LSST_FORMAT_VERSION',
    'REFRACTION_LSST_PLUS_COLUMNS',
    'REFRACTION_LSST_PLUS_CSV_NAME',
    'REFRACTION_LSST_PLUS_FIELD_COLUMNS',
    'REFRACTION_LSST_PLUS_FORMAT_NAME',
    'REFRACTION_LSST_PLUS_FORMAT_VERSION',
    'REFRACTION_LSST_OPTIONAL_MULTILAYER_COLUMNS',
    'REFRACTION_LSST_REQUIRED_COLUMNS',
    'RefractionStaticLsstExportError',
    'format_refraction_lsst_csv',
    'format_refraction_lsst_plus_csv',
    'write_refraction_lsst_csv',
    'write_refraction_lsst_plus_csv',
]
