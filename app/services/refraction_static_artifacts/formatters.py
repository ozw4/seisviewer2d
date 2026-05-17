"""Formatting helpers shared by refraction static artifact writers."""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

from app.services.refraction_static_artifacts.contract import (
    RefractionStaticArtifactError,
)
from app.services.refraction_static_export_units import seconds_to_export_units


def _nan_if_none(value: float | None) -> float:
    return float('nan') if value is None else float(value)


def _float_or_nan(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return out if np.isfinite(out) else float('nan')


def _required_finite_float(value: object, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticArtifactError(f'{name} must be finite') from exc
    if not np.isfinite(out):
        raise RefractionStaticArtifactError(f'{name} must be finite')
    return out


def _json_float(value: object) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _csv_float(value: object) -> str | float:
    if value is None:
        return ''
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ''
    return out if np.isfinite(out) else ''


def _csv_meters(value: object) -> str:
    out = _csv_float(value)
    return '' if out == '' else f'{float(out):.3f}'


def _csv_grid_float(value: object) -> str | float:
    if value is None:
        return ''
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ''
    if np.isnan(out):
        return ''
    if np.isposinf(out):
        return 'inf'
    if np.isneginf(out):
        return '-inf'
    return out


def _csv_ms(value_s: object) -> str | float:
    out = _csv_float(value_s)
    return '' if out == '' else seconds_to_export_units(out, 'milliseconds')


def _csv_bool(value: object) -> str:
    return 'true' if bool(value) else 'false'


def _csv_int(value: object) -> str | int:
    if value is None:
        return ''
    try:
        return int(value)
    except (TypeError, ValueError):
        return ''


def _csv_identifier(value: object) -> str | int:
    if value is None:
        return ''
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        out = float(value)
        if not np.isfinite(out):
            return ''
        return int(out) if out.is_integer() else str(out)
    return str(value)


def _csv_cell_id(value: object) -> str | int:
    out = _csv_int(value)
    if out == '' or int(out) < 0:
        return ''
    return out


def _csv_layer_index(value: object) -> str | int:
    out = _csv_int(value)
    if out == '' or int(out) <= 0:
        return ''
    return out


def _spreadsheet_text(value: object) -> str:
    if value is None:
        return ''
    return str(value)


def _spreadsheet_int(value: object) -> str:
    out = _csv_int(value)
    return '' if out == '' else str(out)


def _spreadsheet_ms(value: object) -> str:
    return _spreadsheet_fixed(value, decimals=6)


def _spreadsheet_m(value: object) -> str:
    return _spreadsheet_fixed(value, decimals=3)


def _spreadsheet_velocity(value: object) -> str:
    return _spreadsheet_fixed(value, decimals=3)


def _spreadsheet_fixed(value: object, *, decimals: int) -> str:
    if value is None or value == '':
        return ''
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ''
    if not np.isfinite(numeric):
        return ''
    return f'{numeric:.{decimals}f}'


def _csv_json_object(value: Mapping[str, object] | None) -> str:
    payload = {} if value is None else dict(value)
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


__all__ = [
    '_csv_bool',
    '_csv_cell_id',
    '_csv_float',
    '_csv_grid_float',
    '_csv_identifier',
    '_csv_int',
    '_csv_json_object',
    '_csv_layer_index',
    '_csv_meters',
    '_csv_ms',
    '_float_or_nan',
    '_json_float',
    '_nan_if_none',
    '_required_finite_float',
    '_spreadsheet_fixed',
    '_spreadsheet_int',
    '_spreadsheet_m',
    '_spreadsheet_ms',
    '_spreadsheet_text',
    '_spreadsheet_velocity',
]
