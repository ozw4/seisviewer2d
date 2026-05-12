from __future__ import annotations

import pytest

from app.services.refraction_static_export_types import (
    REFRACTION_STATIC_EXPORT_SIGN_CONVENTION,
)
from app.services.refraction_static_table_validator import (
    validate_canonical_static_table_rows,
)


def _canonical_row(
    *,
    endpoint_kind: str = 'source',
    endpoint_key: str = 'source:1001',
    endpoint_id: str = '1001',
    applied_shift_ms: str = '12.5',
    static_status: str = 'ok',
    sign_convention: str = REFRACTION_STATIC_EXPORT_SIGN_CONVENTION,
) -> dict[str, str]:
    return {
        'format_name': 'canonical_static_table',
        'format_version': '1',
        'source_job_id': 'refraction-job',
        'endpoint_kind': endpoint_kind,
        'endpoint_key': endpoint_key,
        'endpoint_id': endpoint_id,
        'applied_shift_ms': applied_shift_ms,
        'static_status': static_status,
        'sign_convention': sign_convention,
    }


def test_validate_canonical_static_table_success() -> None:
    result = validate_canonical_static_table_rows(
        (
            _canonical_row(),
            _canonical_row(
                endpoint_kind='receiver',
                endpoint_key='receiver:2001',
                endpoint_id='2001',
                applied_shift_ms='-3.25',
            ),
        )
    )

    assert result.is_valid is True
    assert result.n_rows == 2
    assert result.n_source_rows == 1
    assert result.n_receiver_rows == 1
    assert result.n_invalid_rows == 0
    assert result.errors == ()
    assert len(result.normalized_rows) == 2
    assert result.normalized_rows[0].endpoint_kind == 'source'
    assert result.normalized_rows[1].endpoint_kind == 'receiver'


def test_validate_static_table_rejects_missing_required_columns() -> None:
    row = _canonical_row()
    row.pop('applied_shift_ms')

    result = validate_canonical_static_table_rows((row,))

    assert result.is_valid is False
    assert result.n_invalid_rows == 1
    assert result.normalized_rows == ()
    assert any('missing required columns: applied_shift_ms' in error for error in result.errors)


def test_validate_static_table_rejects_duplicate_endpoint_keys() -> None:
    result = validate_canonical_static_table_rows(
        (
            _canonical_row(endpoint_key='source:duplicate', endpoint_id='1001'),
            _canonical_row(endpoint_key='source:duplicate', endpoint_id='1002'),
        )
    )

    assert result.is_valid is False
    assert result.n_invalid_rows == 2
    assert result.normalized_rows == ()
    assert any('duplicate endpoint_key for source' in error for error in result.errors)


def test_validate_static_table_rejects_unknown_sign_convention() -> None:
    result = validate_canonical_static_table_rows(
        (_canonical_row(sign_convention='delay_positive_ms'),)
    )

    assert result.is_valid is False
    assert result.n_invalid_rows == 1
    assert result.normalized_rows == ()
    assert any('sign_convention must be' in error for error in result.errors)


def test_validate_static_table_rejects_missing_sign_convention() -> None:
    row = _canonical_row()
    row.pop('sign_convention')

    result = validate_canonical_static_table_rows((row,))

    assert result.is_valid is False
    assert result.n_invalid_rows == 1
    assert result.normalized_rows == ()
    assert any('missing sign_convention' in error for error in result.errors)


def test_validate_static_table_accepts_sign_convention_override() -> None:
    row = _canonical_row()
    row.pop('sign_convention')

    result = validate_canonical_static_table_rows(
        (row,),
        sign_convention_override=REFRACTION_STATIC_EXPORT_SIGN_CONVENTION,
    )

    assert result.is_valid is True
    assert result.normalized_rows[0].sign_convention == (
        REFRACTION_STATIC_EXPORT_SIGN_CONVENTION
    )


def test_validate_static_table_normalizes_ms_to_seconds() -> None:
    result = validate_canonical_static_table_rows(
        (_canonical_row(applied_shift_ms='25.5'),)
    )

    assert result.is_valid is True
    assert result.normalized_rows[0].applied_shift_s == pytest.approx(0.0255)


def test_validate_static_table_normalizes_seconds_column() -> None:
    row = _canonical_row()
    row.pop('applied_shift_ms')
    row['applied_shift_s'] = '-0.00325'

    result = validate_canonical_static_table_rows((row,))

    assert result.is_valid is True
    assert result.normalized_rows[0].applied_shift_s == pytest.approx(-0.00325)


def test_validate_static_table_rejects_multiple_shift_columns() -> None:
    row_ms = _canonical_row(
        endpoint_key='source:1001',
        endpoint_id='1001',
        applied_shift_ms='12.5',
    )
    row_ms['applied_shift_s'] = ''
    row_s = _canonical_row(
        endpoint_kind='receiver',
        endpoint_key='receiver:2001',
        endpoint_id='2001',
        applied_shift_ms='',
    )
    row_s['applied_shift_s'] = '-0.00325'

    result = validate_canonical_static_table_rows((row_ms, row_s))

    assert result.is_valid is False
    assert result.normalized_rows == ()
    assert result.n_invalid_rows == 2
    assert any('exactly one applied-shift column' in error for error in result.errors)


def test_validate_static_table_rejects_ambiguous_shift_units() -> None:
    row = _canonical_row()
    row.pop('applied_shift_ms')
    row['applied_shift'] = '12.5'

    result = validate_canonical_static_table_rows((row,))

    assert result.is_valid is False
    assert result.normalized_rows == ()
    assert any('ambiguous units' in error for error in result.errors)


def test_validate_static_table_accepts_unqualified_shift_with_units_metadata() -> None:
    row = _canonical_row()
    row.pop('applied_shift_ms')
    row['applied_shift'] = '12.5'

    result = validate_canonical_static_table_rows((row,), shift_units='milliseconds')

    assert result.is_valid is True
    assert result.normalized_rows[0].applied_shift_s == pytest.approx(0.0125)
