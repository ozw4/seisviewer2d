from __future__ import annotations

import pytest

from app.statics.refraction.domain.export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
    REFRACTION_STATIC_SIGN_CONVENTION_HEADER,
    RefractionStaticExportUnitError,
    RefractionStaticSignConventionError,
    export_units_to_seconds,
    format_shift_ms,
    import_shift_seconds_from_row,
    seconds_to_export_units,
    validate_import_sign_convention,
)


def test_seconds_to_milliseconds_conversion() -> None:
    assert seconds_to_export_units(0.0125, 'milliseconds') == pytest.approx(12.5)
    assert seconds_to_export_units(-0.00325, 'milliseconds') == pytest.approx(-3.25)


def test_milliseconds_to_seconds_conversion() -> None:
    assert export_units_to_seconds(12.5, 'milliseconds') == pytest.approx(0.0125)
    assert export_units_to_seconds(-3.25, 'milliseconds') == pytest.approx(-0.00325)


def test_rounding_ms_does_not_change_internal_value() -> None:
    value_s = 0.00123456

    assert seconds_to_export_units(value_s, 'milliseconds') == pytest.approx(1.23456)
    assert format_shift_ms(value_s, rounding_ms=0.01) == '1.23'
    assert value_s == pytest.approx(0.00123456)


def test_export_metadata_includes_repo_sign_convention() -> None:
    assert REFRACTION_STATIC_SIGN_CONVENTION_HEADER == (
        f'sign_convention={REFRACTION_STATIC_REPO_SIGN_CONVENTION}'
    )
    assert validate_import_sign_convention(REFRACTION_STATIC_REPO_SIGN_CONVENTION) == (
        REFRACTION_STATIC_REPO_SIGN_CONVENTION
    )


def test_import_rejects_missing_sign_convention() -> None:
    with pytest.raises(RefractionStaticSignConventionError, match='missing'):
        validate_import_sign_convention(None)


def test_import_rejects_ambiguous_unqualified_shift_without_units() -> None:
    with pytest.raises(RefractionStaticExportUnitError, match='ambiguous units'):
        import_shift_seconds_from_row(
            {'applied_shift': '12.5'},
            base_name='applied_shift',
        )


def test_positive_and_negative_shift_round_trip() -> None:
    for value_s in (0.0125, -0.00325):
        exported_ms = seconds_to_export_units(value_s, 'milliseconds')
        assert export_units_to_seconds(exported_ms, 'milliseconds') == pytest.approx(
            value_s
        )
        exported_s = seconds_to_export_units(value_s, 'seconds')
        assert export_units_to_seconds(exported_s, 'seconds') == pytest.approx(value_s)
