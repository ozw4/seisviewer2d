"""Validator for M5 canonical refraction source/receiver static tables."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from app.services.refraction_static_export_types import (
    REFRACTION_STATIC_EXPORT_SIGN_CONVENTION,
    RefractionStaticEndpointKind,
)
from app.services.refraction_static_status import REFRACTION_STATIC_STATUSES

CANONICAL_STATIC_TABLE_FORMAT_NAME = 'canonical_static_table'
CANONICAL_STATIC_TABLE_FORMAT_VERSION = 1
CANONICAL_STATIC_TABLE_REQUIRED_COLUMNS: tuple[str, ...] = (
    'format_name',
    'format_version',
    'source_job_id',
    'endpoint_kind',
    'endpoint_key',
    'endpoint_id',
    'applied_shift_ms',
    'static_status',
    'sign_convention',
)
CANONICAL_STATIC_TABLE_OPTIONAL_COLUMNS: tuple[str, ...] = (
    'x_m',
    'y_m',
    'source_id',
    'receiver_id',
    'node_id',
    'total_static_ms',
    'total_applied_shift_ms',
    'source_field_shift_ms',
    'receiver_field_shift_ms',
    'source_total_with_field_shift_ms',
    'receiver_total_with_field_shift_ms',
    'manual_static_shift_ms',
    'source_depth_shift_ms',
    'uphole_shift_ms',
    't1_ms',
    't2_ms',
    't3_ms',
    'v1_m_s',
    'v2_m_s',
    'v3_m_s',
    'vsub_m_s',
    'sh1_weathering_thickness_m',
    'sh2_weathering_thickness_m',
    'sh3_weathering_thickness_m',
    'weathering_correction_ms',
    'elevation_correction_ms',
    'comment',
)

_VALID_ENDPOINT_KINDS = {'source', 'receiver'}
_OPTIONAL_TEXT_COLUMNS = {'source_id', 'receiver_id', 'comment'}
_OPTIONAL_NUMERIC_COLUMNS = frozenset(
    set(CANONICAL_STATIC_TABLE_OPTIONAL_COLUMNS) - _OPTIONAL_TEXT_COLUMNS
)
_KNOWN_COLUMNS = frozenset(
    set(CANONICAL_STATIC_TABLE_REQUIRED_COLUMNS)
    | set(CANONICAL_STATIC_TABLE_OPTIONAL_COLUMNS)
)


@dataclass(frozen=True)
class RefractionStaticTableNormalizedRow:
    """Apply-ready canonical endpoint row using seconds internally."""

    row_number: int
    format_name: Literal['canonical_static_table']
    format_version: int
    source_job_id: str
    endpoint_kind: RefractionStaticEndpointKind
    endpoint_key: str
    endpoint_id: str | None
    applied_shift_s: float | None
    static_status: str
    sign_convention: str


@dataclass(frozen=True)
class RefractionStaticTableValidationResult:
    """Dependency-light canonical static-table validation output."""

    is_valid: bool
    n_rows: int
    n_source_rows: int
    n_receiver_rows: int
    n_invalid_rows: int
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    normalized_rows: tuple[RefractionStaticTableNormalizedRow, ...]


def load_canonical_static_table_csv(
    path: Path,
) -> tuple[dict[str, str | None], ...]:
    """Load a canonical static-table CSV without validating row semantics."""
    table_path = Path(path)
    with table_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        return tuple(dict(row) for row in reader)


def validate_canonical_static_table_csv(
    path: Path,
    *,
    allowed_passthrough_statuses: Iterable[str] = (),
    supported_format_versions: Iterable[int] = (CANONICAL_STATIC_TABLE_FORMAT_VERSION,),
) -> RefractionStaticTableValidationResult:
    """Load and validate an M5 canonical static-table CSV."""
    table_path = Path(path)
    with table_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = tuple(dict(row) for row in reader)
        columns = tuple(reader.fieldnames or ())
    return validate_canonical_static_table_rows(
        rows,
        columns=columns,
        allowed_passthrough_statuses=allowed_passthrough_statuses,
        supported_format_versions=supported_format_versions,
    )


def validate_canonical_static_table_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    allowed_passthrough_statuses: Iterable[str] = (),
    supported_format_versions: Iterable[int] = (CANONICAL_STATIC_TABLE_FORMAT_VERSION,),
) -> RefractionStaticTableValidationResult:
    """Validate M5 canonical static-table rows and normalize shifts to seconds."""
    row_tuple = tuple(rows)
    supplied_columns = tuple(columns) if columns is not None else _columns_from_rows(row_tuple)
    missing_columns = [
        column
        for column in CANONICAL_STATIC_TABLE_REQUIRED_COLUMNS
        if column not in supplied_columns
    ]
    warnings: list[str] = []
    errors: list[str] = []
    invalid_rows: set[int] = set()
    normalized_rows: list[RefractionStaticTableNormalizedRow] = []

    if missing_columns:
        errors.append(f'missing required columns: {", ".join(missing_columns)}')
        invalid_rows.update(range(1, len(row_tuple) + 1))

    unknown_columns = sorted(set(supplied_columns) - _KNOWN_COLUMNS)
    if unknown_columns:
        warnings.append(f'unrecognized columns ignored: {", ".join(unknown_columns)}')

    supported_versions = {int(version) for version in supported_format_versions}
    passthrough_statuses = {str(status) for status in allowed_passthrough_statuses}
    seen_endpoint_keys: dict[tuple[str, str], int] = {}
    seen_endpoint_ids: dict[tuple[str, str], int] = {}
    n_source_rows = 0
    n_receiver_rows = 0

    for row_number, row in enumerate(row_tuple, start=1):
        before_error_count = len(errors)
        format_name = _required_text(
            row,
            'format_name',
            row_number=row_number,
            errors=errors,
        )
        if format_name is not None and format_name != CANONICAL_STATIC_TABLE_FORMAT_NAME:
            _row_error(
                errors,
                row_number,
                f'format_name must be {CANONICAL_STATIC_TABLE_FORMAT_NAME!r}',
            )

        format_version = _required_int(
            row,
            'format_version',
            row_number=row_number,
            errors=errors,
        )
        if format_version is not None and format_version not in supported_versions:
            _row_error(
                errors,
                row_number,
                f'unsupported format_version: {format_version}',
            )

        source_job_id = _required_text(
            row,
            'source_job_id',
            row_number=row_number,
            errors=errors,
        )
        endpoint_kind = _endpoint_kind(row, row_number=row_number, errors=errors)
        if endpoint_kind == 'source':
            n_source_rows += 1
        elif endpoint_kind == 'receiver':
            n_receiver_rows += 1

        endpoint_key = _required_text(
            row,
            'endpoint_key',
            row_number=row_number,
            errors=errors,
        )
        endpoint_id = _required_column_optional_text(
            row,
            'endpoint_id',
            row_number=row_number,
            errors=errors,
        )
        static_status = _required_text(
            row,
            'static_status',
            row_number=row_number,
            errors=errors,
        )
        sign_convention = _required_text(
            row,
            'sign_convention',
            row_number=row_number,
            errors=errors,
        )
        if (
            sign_convention is not None
            and sign_convention != REFRACTION_STATIC_EXPORT_SIGN_CONVENTION
        ):
            _row_error(
                errors,
                row_number,
                'sign_convention must be '
                f'{REFRACTION_STATIC_EXPORT_SIGN_CONVENTION!r}',
            )

        applied_shift_s = _applied_shift_s(
            row,
            row_number=row_number,
            static_status=static_status,
            allowed_passthrough_statuses=passthrough_statuses,
            errors=errors,
        )
        _validate_static_status(
            static_status,
            row_number=row_number,
            allowed_passthrough_statuses=passthrough_statuses,
            errors=errors,
            warnings=warnings,
        )
        _validate_optional_numeric_columns(row, row_number=row_number, errors=errors)

        if endpoint_kind is not None and endpoint_key is not None:
            _validate_unique_endpoint_identity(
                identity=endpoint_key,
                identity_name='endpoint_key',
                endpoint_kind=endpoint_kind,
                row_number=row_number,
                seen=seen_endpoint_keys,
                errors=errors,
                invalid_rows=invalid_rows,
            )
        if endpoint_kind is not None and endpoint_id is not None:
            _validate_unique_endpoint_identity(
                identity=endpoint_id,
                identity_name='endpoint_id',
                endpoint_kind=endpoint_kind,
                row_number=row_number,
                seen=seen_endpoint_ids,
                errors=errors,
                invalid_rows=invalid_rows,
            )

        if len(errors) > before_error_count:
            invalid_rows.add(row_number)

        if (
            len(errors) == before_error_count
            and format_name == CANONICAL_STATIC_TABLE_FORMAT_NAME
            and format_version is not None
            and source_job_id is not None
            and endpoint_kind is not None
            and endpoint_key is not None
            and static_status is not None
            and sign_convention == REFRACTION_STATIC_EXPORT_SIGN_CONVENTION
        ):
            normalized_rows.append(
                RefractionStaticTableNormalizedRow(
                    row_number=row_number,
                    format_name='canonical_static_table',
                    format_version=format_version,
                    source_job_id=source_job_id,
                    endpoint_kind=endpoint_kind,
                    endpoint_key=endpoint_key,
                    endpoint_id=endpoint_id,
                    applied_shift_s=applied_shift_s,
                    static_status=static_status,
                    sign_convention=sign_convention,
                )
            )

    is_valid = not errors
    return RefractionStaticTableValidationResult(
        is_valid=is_valid,
        n_rows=len(row_tuple),
        n_source_rows=n_source_rows,
        n_receiver_rows=n_receiver_rows,
        n_invalid_rows=len(invalid_rows),
        warnings=tuple(warnings),
        errors=tuple(errors),
        normalized_rows=tuple(normalized_rows) if is_valid else (),
    )


def _columns_from_rows(rows: tuple[Mapping[str, Any], ...]) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if isinstance(key, str) and key not in seen:
                columns.append(key)
                seen.add(key)
    return tuple(columns)


def _required_text(
    row: Mapping[str, Any],
    column: str,
    *,
    row_number: int,
    errors: list[str],
) -> str | None:
    if column not in row:
        _row_error(errors, row_number, f'missing required value for {column}')
        return None
    text = _optional_text(row[column])
    if text is None:
        _row_error(errors, row_number, f'missing required value for {column}')
        return None
    return text


def _required_column_optional_text(
    row: Mapping[str, Any],
    column: str,
    *,
    row_number: int,
    errors: list[str],
) -> str | None:
    if column not in row:
        _row_error(errors, row_number, f'missing required value for {column}')
        return None
    return _optional_text(row[column])


def _required_int(
    row: Mapping[str, Any],
    column: str,
    *,
    row_number: int,
    errors: list[str],
) -> int | None:
    text = _required_text(row, column, row_number=row_number, errors=errors)
    if text is None:
        return None
    try:
        return _parse_int(text)
    except ValueError:
        _row_error(errors, row_number, f'{column} must be an integer')
        return None


def _endpoint_kind(
    row: Mapping[str, Any],
    *,
    row_number: int,
    errors: list[str],
) -> RefractionStaticEndpointKind | None:
    text = _required_text(row, 'endpoint_kind', row_number=row_number, errors=errors)
    if text is None:
        return None
    normalized = text.lower()
    if normalized not in _VALID_ENDPOINT_KINDS:
        _row_error(errors, row_number, 'endpoint_kind must be source or receiver')
        return None
    return cast(RefractionStaticEndpointKind, normalized)


def _applied_shift_s(
    row: Mapping[str, Any],
    *,
    row_number: int,
    static_status: str | None,
    allowed_passthrough_statuses: set[str],
    errors: list[str],
) -> float | None:
    raw_value = row.get('applied_shift_ms')
    if static_status == 'ok':
        value = _required_float(
            raw_value,
            column='applied_shift_ms',
            row_number=row_number,
            errors=errors,
        )
        return None if value is None else value / 1000.0
    if static_status in allowed_passthrough_statuses:
        value = _optional_float(raw_value)
        if value is None:
            return None
        if math.isfinite(value):
            return value / 1000.0
        _row_error(errors, row_number, 'applied_shift_ms must be finite when present')
        return None
    if _optional_text(raw_value) is None:
        return None
    value = _required_float(
        raw_value,
        column='applied_shift_ms',
        row_number=row_number,
        errors=errors,
    )
    return None if value is None else value / 1000.0


def _validate_static_status(
    static_status: str | None,
    *,
    row_number: int,
    allowed_passthrough_statuses: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    if static_status is None:
        return
    if static_status == 'ok':
        return
    if static_status in allowed_passthrough_statuses:
        warnings.append(
            f'row {row_number}: static_status {static_status!r} allowed as passthrough'
        )
        return
    if static_status not in REFRACTION_STATIC_STATUSES:
        _row_error(errors, row_number, f'unknown static_status: {static_status!r}')
        return
    _row_error(
        errors,
        row_number,
        f'static_status must be ok for apply: {static_status!r}',
    )


def _validate_optional_numeric_columns(
    row: Mapping[str, Any],
    *,
    row_number: int,
    errors: list[str],
) -> None:
    for column in sorted(_OPTIONAL_NUMERIC_COLUMNS):
        if column not in row:
            continue
        text = _optional_text(row[column])
        if text is None:
            continue
        value = _optional_float(text)
        if value is None or math.isfinite(value):
            continue
        _row_error(errors, row_number, f'{column} must be finite when present')


def _validate_unique_endpoint_identity(
    *,
    identity: str,
    identity_name: str,
    endpoint_kind: str,
    row_number: int,
    seen: dict[tuple[str, str], int],
    errors: list[str],
    invalid_rows: set[int],
) -> None:
    key = (endpoint_kind, identity)
    first_row = seen.get(key)
    if first_row is None:
        seen[key] = row_number
        return
    _row_error(
        errors,
        row_number,
        f'duplicate {identity_name} for {endpoint_kind}: {identity!r} '
        f'(first row {first_row})',
    )
    invalid_rows.add(first_row)
    invalid_rows.add(row_number)


def _required_float(
    value: object,
    *,
    column: str,
    row_number: int,
    errors: list[str],
) -> float | None:
    text = _optional_text(value)
    if text is None:
        _row_error(errors, row_number, f'missing required value for {column}')
        return None
    parsed = _optional_float(text)
    if parsed is None or not math.isfinite(parsed):
        _row_error(errors, row_number, f'{column} must be finite')
        return None
    return parsed


def _optional_float(value: object) -> float | None:
    text = _optional_text(value)
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return float('nan')


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _parse_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError('boolean is not an integer version')
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError('empty integer')
    return int(text)


def _row_error(errors: list[str], row_number: int, message: str) -> None:
    errors.append(f'row {row_number}: {message}')


__all__ = [
    'CANONICAL_STATIC_TABLE_FORMAT_NAME',
    'CANONICAL_STATIC_TABLE_FORMAT_VERSION',
    'CANONICAL_STATIC_TABLE_OPTIONAL_COLUMNS',
    'CANONICAL_STATIC_TABLE_REQUIRED_COLUMNS',
    'RefractionStaticTableNormalizedRow',
    'RefractionStaticTableValidationResult',
    'load_canonical_static_table_csv',
    'validate_canonical_static_table_csv',
    'validate_canonical_static_table_rows',
]
