"""App-owned manual-static table adapters for refraction field corrections."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal

import numpy as np
from seis_statics.refraction.manual_static import (
    RefractionManualStaticTableRow,
    manual_static_inline_rows as _core_manual_static_inline_rows,
)

ManualStaticSignConvention = Literal['applied_shift_s', 'delay_positive_ms']

_VALID_ENDPOINT_KINDS = {'source', 'receiver'}


def load_refraction_manual_static_table_rows(
    path: Path,
    *,
    default_endpoint_kind: str | None = None,
) -> tuple[RefractionManualStaticTableRow, ...]:
    """Load manual static CSV rows without matching them to endpoints."""
    table_path = Path(path)
    default_kind = _coerce_optional_endpoint_kind(default_endpoint_kind)
    with table_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = [
            _manual_static_row_from_csv(
                raw=row,
                source_name=table_path.name,
                row_number=int(row_number),
                default_endpoint_kind=default_kind,
            )
            for row_number, row in enumerate(reader, start=2)
        ]
    return tuple(rows)


def manual_static_inline_rows(
    *,
    endpoint_kind: Literal['source', 'receiver'],
    entries: list[Any] | tuple[Any, ...] | None,
    sign_convention: ManualStaticSignConvention | str = 'applied_shift_s',
) -> tuple[RefractionManualStaticTableRow, ...]:
    """Convert schema-validated inline entries to external manual-static rows."""
    plain_entries = [
        {'endpoint_id': int(entry.endpoint_id), 'value': float(entry.value)}
        for entry in entries or ()
    ]
    return _core_manual_static_inline_rows(
        endpoint_kind=endpoint_kind,
        entries=plain_entries,
        sign_convention=sign_convention,
    )


def _manual_static_row_from_csv(
    *,
    raw: dict[str, str | None],
    source_name: str,
    row_number: int,
    default_endpoint_kind: str | None,
) -> RefractionManualStaticTableRow:
    endpoint_kind = _coerce_endpoint_kind(
        _first_nonblank(raw.get('endpoint_kind'), default_endpoint_kind)
    )
    value_s, value_status = _manual_static_input_seconds(raw)
    return RefractionManualStaticTableRow(
        endpoint_kind=endpoint_kind,
        endpoint_key=_optional_str(raw.get('endpoint_key')),
        endpoint_id=_optional_int(raw.get('endpoint_id'), name='endpoint_id'),
        station_id=_optional_int(raw.get('station_id'), name='station_id'),
        node_id=_optional_int(raw.get('node_id'), name='node_id'),
        x_m=_optional_float(raw.get('x_m'), name='x_m'),
        y_m=_optional_float(raw.get('y_m'), name='y_m'),
        manual_static_input_s=value_s,
        status=value_status,
        comment=_optional_str(raw.get('comment')),
        source_name=str(source_name),
        row_number=int(row_number),
    )


def _manual_static_input_seconds(raw: dict[str, str | None]) -> tuple[float, str]:
    seconds_text = _optional_str(raw.get('manual_static_s'))
    millis_text = _optional_str(raw.get('manual_static_ms'))
    try:
        if seconds_text is not None:
            value = float(seconds_text)
        elif millis_text is not None:
            value = float(millis_text) / 1000.0
        else:
            return np.nan, 'invalid_manual_static_value'
    except ValueError:
        return np.nan, 'invalid_manual_static_value'
    if not np.isfinite(value):
        return np.nan, 'invalid_manual_static_value'
    return float(value), 'ok'


def _coerce_endpoint_kind(value: object) -> Literal['source', 'receiver']:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _VALID_ENDPOINT_KINDS:
            return normalized  # type: ignore[return-value]
    raise ValueError('endpoint_kind must be source or receiver')


def _coerce_optional_endpoint_kind(value: object) -> str | None:
    if value is None:
        return None
    return _coerce_endpoint_kind(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _first_nonblank(*values: object) -> object | None:
    for value in values:
        if _optional_str(value) is not None:
            return value
    return None


def _optional_int(value: object, *, name: str) -> int | None:
    text = _optional_str(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f'{name} must be an integer') from exc


def _optional_float(value: object, *, name: str) -> float | None:
    text = _optional_str(value)
    if text is None:
        return None
    try:
        out = float(text)
    except ValueError as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not np.isfinite(out):
        raise ValueError(f'{name} must be finite')
    return out


__all__ = [
    'ManualStaticSignConvention',
    'RefractionManualStaticTableRow',
    'load_refraction_manual_static_table_rows',
    'manual_static_inline_rows',
]
