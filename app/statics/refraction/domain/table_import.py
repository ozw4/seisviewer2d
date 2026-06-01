"""Import service for M5 canonical refraction static tables."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.statics.refraction.domain.export_types import RefractionStaticEndpointKind
from app.statics.refraction.domain.export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
    RefractionStaticExportUnits,
)
from app.statics.refraction.domain.table_validator import (
    CANONICAL_STATIC_TABLE_FORMAT_VERSION,
    CANONICAL_STATIC_TABLE_OPTIONAL_COLUMNS,
    RefractionStaticTableNormalizedRow,
    validate_canonical_static_table_rows,
)


@dataclass(frozen=True)
class RefractionImportedEndpointStatic:
    """Validated endpoint static row normalized for later apply workflows."""

    endpoint_kind: RefractionStaticEndpointKind
    endpoint_key: str
    endpoint_id: str | None
    applied_shift_s: float
    static_status: str
    source_job_id: str
    row_number: int
    source_name: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class RefractionStaticTableImportResult:
    """Validated static table import result split by endpoint kind."""

    source_static_by_endpoint_key: Mapping[str, RefractionImportedEndpointStatic]
    receiver_static_by_endpoint_key: Mapping[str, RefractionImportedEndpointStatic]
    n_source_rows: int
    n_receiver_rows: int
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    schema_version: int
    sign_convention: str

    @property
    def is_valid(self) -> bool:
        """Whether all supplied tables were valid and importable."""
        return not self.errors


@dataclass(frozen=True)
class _LoadedStaticTable:
    rows: tuple[dict[str, str | None], ...]
    columns: tuple[str, ...]
    source_name: str
    expected_endpoint_kind: RefractionStaticEndpointKind | None = None
    default_source_job_id: str | None = None


def import_refraction_static_table_csv(
    path: Path,
    *,
    sign_convention_override: str | None = None,
    shift_units: RefractionStaticExportUnits | str | None = None,
) -> RefractionStaticTableImportResult:
    """Import one combined canonical static CSV table."""
    return import_refraction_static_tables(
        combined_table_path=path,
        sign_convention_override=sign_convention_override,
        shift_units=shift_units,
    )


def import_refraction_static_source_receiver_csvs(
    *,
    source_table_path: Path,
    receiver_table_path: Path,
    source_job_id: str | None = None,
    receiver_source_job_id: str | None = None,
    sign_convention_override: str | None = None,
    shift_units: RefractionStaticExportUnits | str | None = None,
) -> RefractionStaticTableImportResult:
    """Import separate canonical source and receiver static CSV tables."""
    return import_refraction_static_tables(
        source_table_path=source_table_path,
        receiver_table_path=receiver_table_path,
        source_job_id=source_job_id,
        receiver_source_job_id=receiver_source_job_id,
        sign_convention_override=sign_convention_override,
        shift_units=shift_units,
    )


def import_refraction_static_tables(
    *,
    combined_table_path: Path | None = None,
    source_table_path: Path | None = None,
    receiver_table_path: Path | None = None,
    source_job_id: str | None = None,
    receiver_source_job_id: str | None = None,
    sign_convention_override: str | None = None,
    shift_units: RefractionStaticExportUnits | str | None = None,
) -> RefractionStaticTableImportResult:
    """Import canonical static table CSVs and normalize shifts to seconds."""
    input_error = _validate_table_path_combination(
        combined_table_path=combined_table_path,
        source_table_path=source_table_path,
        receiver_table_path=receiver_table_path,
    )
    if input_error is not None:
        return _empty_result(errors=(input_error,))

    tables = _load_input_tables(
        combined_table_path=combined_table_path,
        source_table_path=source_table_path,
        receiver_table_path=receiver_table_path,
        source_job_id=source_job_id,
        receiver_source_job_id=receiver_source_job_id,
    )
    return import_refraction_static_table_rows(
        tables,
        sign_convention_override=sign_convention_override,
        shift_units=shift_units,
    )


def import_refraction_static_table_rows(
    tables: Sequence[_LoadedStaticTable],
    *,
    sign_convention_override: str | None = None,
    shift_units: RefractionStaticExportUnits | str | None = None,
) -> RefractionStaticTableImportResult:
    """Import already loaded canonical static table rows."""
    if not tables:
        return _empty_result(errors=('at least one static table is required',))

    valid_tables: list[
        tuple[_LoadedStaticTable, tuple[RefractionStaticTableNormalizedRow, ...]]
    ] = []
    warnings: list[str] = []
    errors: list[str] = []
    n_source_rows = 0
    n_receiver_rows = 0

    for table in tables:
        canonical_rows, canonical_columns = _canonicalized_table_rows(table)
        validation = validate_canonical_static_table_rows(
            canonical_rows,
            columns=canonical_columns,
            sign_convention_override=sign_convention_override,
            shift_units=shift_units,
        )
        warnings.extend(validation.warnings)
        errors.extend(validation.errors)
        n_source_rows += validation.n_source_rows
        n_receiver_rows += validation.n_receiver_rows
        if not validation.is_valid:
            continue
        table_kind_errors = _separate_table_kind_errors(
            validation.normalized_rows,
            expected_endpoint_kind=table.expected_endpoint_kind,
            source_name=table.source_name,
        )
        errors.extend(table_kind_errors)
        valid_tables.append((table, validation.normalized_rows))

    if errors:
        return _empty_result(
            n_source_rows=n_source_rows,
            n_receiver_rows=n_receiver_rows,
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

    source_map: dict[str, RefractionImportedEndpointStatic] = {}
    receiver_map: dict[str, RefractionImportedEndpointStatic] = {}
    for table, normalized_rows in valid_tables:
        canonical_rows, _canonical_columns = _canonicalized_table_rows(table)
        for normalized in normalized_rows:
            raw = canonical_rows[normalized.row_number - 1]
            endpoint_static = _imported_endpoint_static(
                normalized,
                raw=raw,
                source_name=table.source_name,
            )
            if endpoint_static.endpoint_kind == 'source':
                source_map[endpoint_static.endpoint_key] = endpoint_static
            else:
                receiver_map[endpoint_static.endpoint_key] = endpoint_static

    return RefractionStaticTableImportResult(
        source_static_by_endpoint_key=source_map,
        receiver_static_by_endpoint_key=receiver_map,
        n_source_rows=n_source_rows,
        n_receiver_rows=n_receiver_rows,
        warnings=tuple(warnings),
        errors=(),
        schema_version=CANONICAL_STATIC_TABLE_FORMAT_VERSION,
        sign_convention=REFRACTION_STATIC_REPO_SIGN_CONVENTION,
    )


def _load_input_tables(
    *,
    combined_table_path: Path | None,
    source_table_path: Path | None,
    receiver_table_path: Path | None,
    source_job_id: str | None,
    receiver_source_job_id: str | None,
) -> tuple[_LoadedStaticTable, ...]:
    if combined_table_path is not None:
        return (_load_static_table(combined_table_path),)
    assert source_table_path is not None
    assert receiver_table_path is not None
    return (
        _load_static_table(
            source_table_path,
            expected_endpoint_kind='source',
            default_source_job_id=source_job_id,
        ),
        _load_static_table(
            receiver_table_path,
            expected_endpoint_kind='receiver',
            default_source_job_id=receiver_source_job_id or source_job_id,
        ),
    )


def _load_static_table(
    path: Path,
    *,
    expected_endpoint_kind: RefractionStaticEndpointKind | None = None,
    default_source_job_id: str | None = None,
) -> _LoadedStaticTable:
    table_path = Path(path)
    with table_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        return _LoadedStaticTable(
            rows=tuple(dict(row) for row in reader),
            columns=tuple(reader.fieldnames or ()),
            source_name=table_path.name,
            expected_endpoint_kind=expected_endpoint_kind,
            default_source_job_id=default_source_job_id,
        )


def _canonicalized_table_rows(
    table: _LoadedStaticTable,
) -> tuple[tuple[dict[str, str | None], ...], tuple[str, ...]]:
    if not _is_documented_endpoint_static_table(table):
        return table.rows, table.columns
    endpoint_kind = _documented_endpoint_kind(table)
    rows = tuple(
        _canonicalized_endpoint_static_row(
            row,
            endpoint_kind=endpoint_kind,
            default_source_job_id=table.default_source_job_id,
        )
        for row in table.rows
    )
    columns = _columns_from_rows(rows)
    return rows, columns


def _is_documented_endpoint_static_table(table: _LoadedStaticTable) -> bool:
    column_set = set(table.columns)
    if 'endpoint_key' in column_set or 'applied_shift_ms' in column_set:
        return False
    return bool(
        {'source_endpoint_key', 'total_applied_shift_ms'} <= column_set
        or {'receiver_endpoint_key', 'total_applied_shift_ms'} <= column_set
    )


def _documented_endpoint_kind(
    table: _LoadedStaticTable,
) -> RefractionStaticEndpointKind:
    column_set = set(table.columns)
    if 'source_endpoint_key' in column_set:
        return 'source'
    if 'receiver_endpoint_key' in column_set:
        return 'receiver'
    if table.expected_endpoint_kind is None:
        raise ValueError(
            f'{table.source_name}: documented static table must include '
            'source_endpoint_key or receiver_endpoint_key'
        )
    return table.expected_endpoint_kind


def _canonicalized_endpoint_static_row(
    row: Mapping[str, Any],
    *,
    endpoint_kind: RefractionStaticEndpointKind,
    default_source_job_id: str | None,
) -> dict[str, str | None]:
    prefix = 'source' if endpoint_kind == 'source' else 'receiver'
    out: dict[str, str | None] = {
        'format_name': _row_text(row, 'format_name') or 'canonical_static_table',
        'format_version': _row_text(row, 'format_version') or '1',
        'source_job_id': _row_text(row, 'source_job_id')
        or _optional_text(default_source_job_id),
        'endpoint_kind': _row_text(row, 'endpoint_kind') or endpoint_kind,
        'endpoint_key': _row_text(row, 'endpoint_key')
        or _row_text(row, f'{prefix}_endpoint_key'),
        'endpoint_id': _row_text(row, 'endpoint_id') or _row_text(row, f'{prefix}_id'),
        'applied_shift_ms': _row_text(row, 'applied_shift_ms')
        or _row_text(row, 'total_applied_shift_ms'),
        'static_status': _row_text(row, 'static_status'),
        'sign_convention': _row_text(row, 'sign_convention'),
    }
    if f'{prefix}_node_id' in row:
        out['node_id'] = _row_text(row, f'{prefix}_node_id')
    for column in CANONICAL_STATIC_TABLE_OPTIONAL_COLUMNS:
        if column in out:
            continue
        if column in row:
            out[column] = _row_text(row, column)
    return out


def _columns_from_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    return tuple(columns)


def _row_text(row: Mapping[str, Any], column: str) -> str | None:
    return _optional_text(row.get(column))


def _validate_table_path_combination(
    *,
    combined_table_path: Path | None,
    source_table_path: Path | None,
    receiver_table_path: Path | None,
) -> str | None:
    has_combined = combined_table_path is not None
    has_separate = source_table_path is not None or receiver_table_path is not None
    if has_combined and has_separate:
        return 'provide either combined_table_path or separate source/receiver paths'
    if has_combined:
        return None
    if source_table_path is None or receiver_table_path is None:
        return 'separate static table import requires source_table_path and receiver_table_path'
    return None


def _separate_table_kind_errors(
    normalized_rows: Sequence[RefractionStaticTableNormalizedRow],
    *,
    expected_endpoint_kind: RefractionStaticEndpointKind | None,
    source_name: str,
) -> tuple[str, ...]:
    if expected_endpoint_kind is None:
        return ()
    errors: list[str] = []
    for row in normalized_rows:
        if row.endpoint_kind == expected_endpoint_kind:
            continue
        errors.append(
            f'{source_name} row {row.row_number}: endpoint_kind must be '
            f'{expected_endpoint_kind!r} for this separate table'
        )
    return tuple(errors)


def _imported_endpoint_static(
    row: RefractionStaticTableNormalizedRow,
    *,
    raw: Mapping[str, Any],
    source_name: str,
) -> RefractionImportedEndpointStatic:
    if row.applied_shift_s is None:
        raise ValueError('validated import row is missing applied_shift_s')
    return RefractionImportedEndpointStatic(
        endpoint_kind=row.endpoint_kind,
        endpoint_key=row.endpoint_key,
        endpoint_id=row.endpoint_id,
        applied_shift_s=float(row.applied_shift_s),
        static_status=row.static_status,
        source_job_id=row.source_job_id,
        row_number=row.row_number,
        source_name=source_name,
        metadata=_metadata_from_row(raw),
    )


def _metadata_from_row(row: Mapping[str, Any]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for column in CANONICAL_STATIC_TABLE_OPTIONAL_COLUMNS:
        value = _optional_text(row.get(column))
        if value is not None:
            metadata[column] = value
    return metadata


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _empty_result(
    *,
    n_source_rows: int = 0,
    n_receiver_rows: int = 0,
    warnings: tuple[str, ...] = (),
    errors: tuple[str, ...],
) -> RefractionStaticTableImportResult:
    return RefractionStaticTableImportResult(
        source_static_by_endpoint_key={},
        receiver_static_by_endpoint_key={},
        n_source_rows=n_source_rows,
        n_receiver_rows=n_receiver_rows,
        warnings=warnings,
        errors=errors,
        schema_version=CANONICAL_STATIC_TABLE_FORMAT_VERSION,
        sign_convention=REFRACTION_STATIC_REPO_SIGN_CONVENTION,
    )


__all__ = [
    'RefractionImportedEndpointStatic',
    'RefractionStaticTableImportResult',
    'import_refraction_static_source_receiver_csvs',
    'import_refraction_static_table_csv',
    'import_refraction_static_tables',
]
