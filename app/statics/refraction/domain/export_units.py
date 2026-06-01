"""M5 refraction static unit conversion and sign-convention helpers."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, ROUND_HALF_UP
import math
from typing import Final, Literal, cast

RefractionStaticExportUnits = Literal['seconds', 'milliseconds']

REFRACTION_STATIC_REPO_SIGN_CONVENTION: Final = 'corrected(t) = raw(t - shift_s)'
REFRACTION_STATIC_SIGN_CONVENTION_HEADER: Final = (
    f'sign_convention={REFRACTION_STATIC_REPO_SIGN_CONVENTION}'
)
REFRACTION_STATIC_EXPORT_UNIT_VALUES: Final[tuple[RefractionStaticExportUnits, ...]] = (
    'seconds',
    'milliseconds',
)


class RefractionStaticExportUnitError(ValueError):
    """Raised when refraction static import/export units are invalid."""


class RefractionStaticSignConventionError(ValueError):
    """Raised when a refraction static sign convention is missing or invalid."""


def normalize_export_units(units: str) -> RefractionStaticExportUnits:
    """Normalize supported refraction static export/import time units."""
    if units not in REFRACTION_STATIC_EXPORT_UNIT_VALUES:
        raise RefractionStaticExportUnitError(
            'units must be "seconds" or "milliseconds"'
        )
    return cast(RefractionStaticExportUnits, units)


def seconds_to_export_units(
    value_s: object,
    units: RefractionStaticExportUnits | str,
    rounding_ms: float | None = None,
) -> float:
    """Convert an internal seconds value to the requested export units."""
    normalized_units = normalize_export_units(str(units))
    value = _coerce_float(value_s, name='value_s')
    if rounding_ms is not None:
        value = _round_seconds_for_display(value, rounding_ms=rounding_ms)
    if normalized_units == 'seconds':
        return value
    return value * 1000.0


def export_units_to_seconds(
    value: object,
    units: RefractionStaticExportUnits | str,
) -> float:
    """Convert an imported refraction static value to internal seconds."""
    normalized_units = normalize_export_units(str(units))
    numeric_value = _coerce_float(value, name='value')
    if normalized_units == 'seconds':
        return numeric_value
    return numeric_value / 1000.0


def format_shift_ms(value_s: object, rounding_ms: float | None) -> str:
    """Format a seconds shift as display milliseconds.

    This helper is for text/card display output. Do not use it for
    machine-readable CSV/NPZ artifacts that should preserve precision.
    """
    value_ms = seconds_to_export_units(
        value_s,
        'milliseconds',
        rounding_ms=rounding_ms,
    )
    if not math.isfinite(value_ms):
        return ''
    if rounding_ms is None or float(rounding_ms) == 0.0:
        return format(value_ms, '.12g')
    decimals = _decimal_places(rounding_ms)
    return f'{value_ms:.{decimals}f}'


def validate_import_sign_convention(
    sign_convention: object,
    *,
    override: object | None = None,
) -> str:
    """Validate imported table sign convention, optionally using an override."""
    if override is not None:
        override_text = _optional_text(override)
        if override_text != REFRACTION_STATIC_REPO_SIGN_CONVENTION:
            raise RefractionStaticSignConventionError(
                'sign_convention override must be '
                f'{REFRACTION_STATIC_REPO_SIGN_CONVENTION!r}'
            )
        return REFRACTION_STATIC_REPO_SIGN_CONVENTION

    text = _optional_text(sign_convention)
    if text is None:
        raise RefractionStaticSignConventionError('missing sign_convention')
    if text != REFRACTION_STATIC_REPO_SIGN_CONVENTION:
        raise RefractionStaticSignConventionError(
            'sign_convention must be '
            f'{REFRACTION_STATIC_REPO_SIGN_CONVENTION!r}'
        )
    return text


def import_shift_seconds_from_row(
    row: Mapping[str, object],
    *,
    base_name: str,
    metadata_units: RefractionStaticExportUnits | str | None = None,
    required: bool = True,
) -> tuple[float | None, str | None]:
    """Read ``<base_name>_s`` or ``<base_name>_ms`` from a row as seconds.

    An unqualified ``<base_name>`` column is accepted only when explicit
    metadata units are provided by the caller.
    """
    seconds_column = f'{base_name}_s'
    milliseconds_column = f'{base_name}_ms'
    candidates = [
        column
        for column in (seconds_column, milliseconds_column, base_name)
        if _optional_text(row.get(column)) is not None
    ]
    if len(candidates) > 1:
        raise RefractionStaticExportUnitError(
            f'{base_name} must use exactly one units column; found '
            f'{", ".join(candidates)}'
        )
    if not candidates:
        if required:
            raise RefractionStaticExportUnitError(
                f'missing required value for {milliseconds_column}'
            )
        return None, None

    column = candidates[0]
    if column == seconds_column:
        return export_units_to_seconds(row[column], 'seconds'), column
    if column == milliseconds_column:
        return export_units_to_seconds(row[column], 'milliseconds'), column

    if metadata_units is None:
        raise RefractionStaticExportUnitError(
            f'{base_name} has ambiguous units; use {seconds_column} or '
            f'{milliseconds_column}, or provide explicit metadata units'
        )
    return export_units_to_seconds(row[column], metadata_units), column


def _round_seconds_for_display(value_s: float, *, rounding_ms: float) -> float:
    rounding = _validate_rounding_ms(rounding_ms)
    if rounding == 0.0 or not math.isfinite(value_s):
        return value_s
    value_ms = Decimal(str(value_s)) * Decimal('1000')
    increment = Decimal(str(rounding))
    rounded_ms = (value_ms / increment).quantize(
        Decimal('1'),
        rounding=ROUND_HALF_UP,
    ) * increment
    return float(rounded_ms) / 1000.0


def _validate_rounding_ms(value: object) -> float:
    rounding = _coerce_float(value, name='rounding_ms')
    if not math.isfinite(rounding) or rounding < 0.0:
        raise RefractionStaticExportUnitError('rounding_ms must be finite and >= 0')
    return rounding


def _decimal_places(value: object) -> int:
    decimal_value = Decimal(str(_validate_rounding_ms(value))).normalize()
    exponent = decimal_value.as_tuple().exponent
    return max(0, -int(exponent))


def _coerce_float(value: object, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticExportUnitError(f'{name} must be numeric') from exc


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


__all__ = [
    'REFRACTION_STATIC_EXPORT_UNIT_VALUES',
    'REFRACTION_STATIC_REPO_SIGN_CONVENTION',
    'REFRACTION_STATIC_SIGN_CONVENTION_HEADER',
    'RefractionStaticExportUnitError',
    'RefractionStaticExportUnits',
    'RefractionStaticSignConventionError',
    'export_units_to_seconds',
    'format_shift_ms',
    'import_shift_seconds_from_row',
    'normalize_export_units',
    'seconds_to_export_units',
    'validate_import_sign_convention',
]
