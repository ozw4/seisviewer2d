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
)

REFRACTION_LSST_CSV_NAME: Final = 'refraction_lsst.csv'
REFRACTION_LSST_FORMAT_NAME: Final = 'lsst'
REFRACTION_LSST_FORMAT_VERSION: Final = 1
REFRACTION_LSST_PLUS_CARDS_NAME: Final = 'refraction_lsst_plus_cards.txt'
REFRACTION_LSST_PLUS_FORMAT_NAME: Final = 'lsst_plus'
REFRACTION_LSST_PLUS_SCHEMA_VERSION: Final = 1

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


def format_refraction_lsst_plus_cards(
    bundle: RefractionStaticExportBundle,
    *,
    fail_on_invalid_static_status: bool = False,
    include_inactive_endpoints: bool = True,
) -> str:
    """Format endpoint rows as documented, deterministic M5 LSST+ cards.

    LSST+ cards use pipe-delimited records. Missing or non-finite numeric values
    are emitted as ``nan`` so invalid endpoint rows remain auditable.
    """
    source_job_id = _validate_bundle(bundle)
    rows = _included_rows(
        bundle,
        fail_on_invalid_static_status=fail_on_invalid_static_status,
        include_inactive_endpoints=include_inactive_endpoints,
    )
    lines = [
        f'# format={REFRACTION_LSST_PLUS_FORMAT_NAME}',
        f'# schema_version={REFRACTION_LSST_PLUS_SCHEMA_VERSION}',
        f'# source_job_id={_format_card_text(source_job_id)}',
        f'# sign_convention={REFRACTION_STATIC_REPO_SIGN_CONVENTION}',
        '# units=ms',
        '# distance_units=m',
        '# velocity_units=m_s',
        '# delimiter=|',
        '# missing_numeric=nan',
        '# missing_text=nan',
        '# row_order=source_rows_then_receiver_rows',
    ]
    for row in rows:
        lines.append(_lsst_plus_endpoint_card(row))
        lines.append(_lsst_plus_static_card(row))
        lines.append(_lsst_plus_layer_card(row))
    return '\n'.join(lines) + '\n'


def write_refraction_lsst_plus_cards(
    bundle: RefractionStaticExportBundle,
    path: Path,
    *,
    fail_on_invalid_static_status: bool = False,
    include_inactive_endpoints: bool = True,
) -> None:
    """Write documented M5 LSST+ card text to ``path``."""
    text = format_refraction_lsst_plus_cards(
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
) -> dict[str, str]:
    status = _required_static_status(row)
    valid = status == 'ok'
    out = {
        'format_name': REFRACTION_LSST_FORMAT_NAME,
        'format_version': str(REFRACTION_LSST_FORMAT_VERSION),
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


def _lsst_plus_endpoint_card(row: RefractionStaticEndpointExportRow) -> str:
    record = 'SRC' if row.endpoint_kind == 'source' else 'REC'
    return '|'.join(
        (
            record,
            _required_card_text(row.endpoint_key, row=row, field_name='endpoint_key'),
            f'endpoint_id={_format_card_text(row.endpoint_id)}',
            f'node_id={_format_card_int(row.node_id, row=row, field_name="node_id")}',
            f'station={_format_card_text(row.station_id)}',
            f'x={_format_card_m(row.x_m, row=row, field_name="x_m")}',
            f'y={_format_card_m(row.y_m, row=row, field_name="y_m")}',
            f'elev={_format_card_m(row.elevation_m, row=row, field_name="elev")}',
            f'status={_required_static_status(row)}',
        )
    )


def _lsst_plus_static_card(row: RefractionStaticEndpointExportRow) -> str:
    endpoint_key = _required_card_text(
        row.endpoint_key,
        row=row,
        field_name='endpoint_key',
    )
    field_prefix = row.endpoint_kind
    field_shift_name = f'{field_prefix}_field_shift_ms'
    field_status_name = f'{field_prefix}_field_static_status'
    total_with_field_name = f'{field_prefix}_total_with_field_shift_ms'
    return '|'.join(
        (
            'STC',
            row.endpoint_kind,
            endpoint_key,
            f'total={_format_card_time_ms(row.total_applied_shift_s, row=row, field_name="total")}',
            f'weathering={_format_card_time_ms(row.weathering_correction_s, row=row, field_name="weathering")}',
            f'elevation={_format_card_time_ms(row.elevation_correction_s, row=row, field_name="elevation")}',
            f'{field_shift_name}={_format_card_time_ms(row.field_correction_s, row=row, field_name=field_shift_name)}',
            f'manual={_format_card_time_ms(row.manual_static_shift_s, row=row, field_name="manual")}',
            f'{total_with_field_name}={_format_card_time_ms(row.total_with_field_shift_s, row=row, field_name=total_with_field_name)}',
            f'source_depth_m={_format_card_m(row.source_depth_m, row=row, field_name="source_depth_m")}',
            f'source_depth={_format_card_time_ms(row.source_depth_shift_s, row=row, field_name="source_depth")}',
            f'uphole_time={_format_card_time_ms(row.uphole_time_s, row=row, field_name="uphole_time")}',
            f'uphole={_format_card_time_ms(row.uphole_shift_s, row=row, field_name="uphole")}',
            f'manual_status={_format_card_text(row.manual_static_status)}',
            f'{field_status_name}={_format_card_text(row.field_static_status)}',
            f'source_depth_status={_format_card_text(row.source_depth_status)}',
            f'uphole_status={_format_card_text(row.uphole_status)}',
        )
    )


def _lsst_plus_layer_card(row: RefractionStaticEndpointExportRow) -> str:
    endpoint_key = _required_card_text(
        row.endpoint_key,
        row=row,
        field_name='endpoint_key',
    )
    return '|'.join(
        (
            'LYR',
            row.endpoint_kind,
            endpoint_key,
            f't1={_format_card_time_ms(row.t1_s, row=row, field_name="t1")}',
            f't2={_format_card_time_ms(row.t2_s, row=row, field_name="t2")}',
            f't3={_format_card_time_ms(row.t3_s, row=row, field_name="t3")}',
            f'sh1={_format_card_m(row.sh1_m, row=row, field_name="sh1")}',
            f'sh2={_format_card_m(row.sh2_m, row=row, field_name="sh2")}',
            f'sh3={_format_card_m(row.sh3_m, row=row, field_name="sh3")}',
            f'v1={_format_card_velocity(row.v1_m_s, row=row, field_name="v1")}',
            f'v2={_format_card_velocity(row.v2_m_s, row=row, field_name="v2")}',
            f'v3={_format_card_velocity(row.v3_m_s, row=row, field_name="v3")}',
            f'vsub={_format_card_velocity(row.vsub_m_s, row=row, field_name="vsub")}',
        )
    )


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
    return _format_fixed(value * 1000.0, decimals=6)


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


def _format_card_time_ms(
    value_s: object,
    *,
    row: RefractionStaticEndpointExportRow,
    field_name: str,
) -> str:
    value = _optional_float(value_s, row=row, field_name=field_name)
    if value is None:
        return 'nan'
    return _format_fixed(value * 1000.0, decimals=6)


def _format_card_m(
    value: object,
    *,
    row: RefractionStaticEndpointExportRow,
    field_name: str,
) -> str:
    return _format_card_numeric(value, decimals=3, row=row, field_name=field_name)


def _format_card_velocity(
    value: object,
    *,
    row: RefractionStaticEndpointExportRow,
    field_name: str,
) -> str:
    return _format_card_numeric(value, decimals=3, row=row, field_name=field_name)


def _format_card_int(
    value: object,
    *,
    row: RefractionStaticEndpointExportRow,
    field_name: str,
) -> str:
    if value is None:
        return 'nan'
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


def _format_card_numeric(
    value: object,
    *,
    decimals: int,
    row: RefractionStaticEndpointExportRow,
    field_name: str,
) -> str:
    numeric = _optional_float(value, row=row, field_name=field_name)
    if numeric is None:
        return 'nan'
    return _format_fixed(numeric, decimals=decimals)


def _required_card_text(
    value: object,
    *,
    row: RefractionStaticEndpointExportRow,
    field_name: str,
) -> str:
    text = _optional_text(value)
    if text is None:
        _raise_missing(row, field_name)
    return _format_card_text(text)


def _format_card_text(value: object) -> str:
    text = _optional_text(value)
    if text is None:
        return 'nan'
    if '|' in text or '\n' in text or '\r' in text:
        raise RefractionStaticLsstExportError(
            f'LSST+ card text contains an unsupported delimiter: {text!r}'
        )
    return text


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
    'REFRACTION_LSST_PLUS_CARDS_NAME',
    'REFRACTION_LSST_PLUS_FORMAT_NAME',
    'REFRACTION_LSST_PLUS_SCHEMA_VERSION',
    'REFRACTION_LSST_OPTIONAL_MULTILAYER_COLUMNS',
    'REFRACTION_LSST_REQUIRED_COLUMNS',
    'RefractionStaticLsstExportError',
    'format_refraction_lsst_csv',
    'format_refraction_lsst_plus_cards',
    'write_refraction_lsst_csv',
    'write_refraction_lsst_plus_cards',
]
