from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.services.refraction_static_export_types import (
    REFRACTION_STATIC_EXPORT_SIGN_CONVENTION,
)
from app.services.refraction_static_table_import import (
    import_refraction_static_source_receiver_csvs,
    import_refraction_static_table_csv,
)


def _canonical_row(
    *,
    endpoint_kind: str = 'source',
    endpoint_key: str = 'source:1001',
    endpoint_id: str = '1001',
    applied_shift_ms: str | None = '12.5',
    applied_shift_s: str | None = None,
    static_status: str = 'ok',
    sign_convention: str = REFRACTION_STATIC_EXPORT_SIGN_CONVENTION,
    comment: str | None = None,
) -> dict[str, str]:
    row = {
        'format_name': 'canonical_static_table',
        'format_version': '1',
        'source_job_id': 'refraction-job',
        'endpoint_kind': endpoint_kind,
        'endpoint_key': endpoint_key,
        'endpoint_id': endpoint_id,
        'static_status': static_status,
        'sign_convention': sign_convention,
    }
    if applied_shift_ms is not None:
        row['applied_shift_ms'] = applied_shift_ms
    if applied_shift_s is not None:
        row['applied_shift_s'] = applied_shift_s
    if comment is not None:
        row['comment'] = comment
    return row


def _documented_static_row(
    *,
    endpoint_kind: str,
    endpoint_key: str,
    endpoint_id: str,
    total_applied_shift_ms: str,
) -> dict[str, str]:
    prefix = endpoint_kind
    return {
        'endpoint_kind': endpoint_kind,
        f'{prefix}_endpoint_key': endpoint_key,
        f'{prefix}_id': endpoint_id,
        f'{prefix}_node_id': endpoint_id,
        'x_m': '1000.000',
        'y_m': '2000.000',
        'surface_elevation_m': '10.000',
        't1_ms': '50.000000',
        'v1_m_s': '800.000',
        'v2_m_s': '2400.000',
        'total_static_ms': total_applied_shift_ms,
        'total_applied_shift_ms': total_applied_shift_ms,
        'static_status': 'ok',
        'sign_convention': REFRACTION_STATIC_EXPORT_SIGN_CONVENTION,
    }


def _write_csv(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def test_import_combined_static_table_success(tmp_path: Path) -> None:
    table_path = tmp_path / 'combined.csv'
    _write_csv(
        table_path,
        (
            _canonical_row(comment='reviewed source static'),
            _canonical_row(
                endpoint_kind='receiver',
                endpoint_key='receiver:2001',
                endpoint_id='2001',
                applied_shift_ms='-3.25',
            ),
        ),
    )

    result = import_refraction_static_table_csv(table_path)

    assert result.is_valid is True
    assert result.n_source_rows == 1
    assert result.n_receiver_rows == 1
    assert result.errors == ()
    assert result.sign_convention == REFRACTION_STATIC_EXPORT_SIGN_CONVENTION
    assert result.source_static_by_endpoint_key['source:1001'].applied_shift_s == (
        pytest.approx(0.0125)
    )
    assert result.source_static_by_endpoint_key['source:1001'].metadata['comment'] == (
        'reviewed source static'
    )
    assert result.receiver_static_by_endpoint_key[
        'receiver:2001'
    ].applied_shift_s == pytest.approx(-0.00325)


def test_import_separate_source_receiver_static_tables_success(tmp_path: Path) -> None:
    source_path = tmp_path / 'source.csv'
    receiver_path = tmp_path / 'receiver.csv'
    _write_csv(source_path, (_canonical_row(applied_shift_ms='10.0'),))
    _write_csv(
        receiver_path,
        (
            _canonical_row(
                endpoint_kind='receiver',
                endpoint_key='receiver:2001',
                endpoint_id='2001',
                applied_shift_ms=None,
                applied_shift_s='-0.0025',
            ),
        ),
    )

    result = import_refraction_static_source_receiver_csvs(
        source_table_path=source_path,
        receiver_table_path=receiver_path,
    )

    assert result.is_valid is True
    assert tuple(result.source_static_by_endpoint_key) == ('source:1001',)
    assert tuple(result.receiver_static_by_endpoint_key) == ('receiver:2001',)
    assert result.source_static_by_endpoint_key['source:1001'].source_name == (
        'source.csv'
    )
    assert result.receiver_static_by_endpoint_key['receiver:2001'].source_name == (
        'receiver.csv'
    )
    assert result.receiver_static_by_endpoint_key[
        'receiver:2001'
    ].applied_shift_s == pytest.approx(-0.0025)


def test_import_documented_source_receiver_static_tables_success(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / 'source_static_table.csv'
    receiver_path = tmp_path / 'receiver_static_table.csv'
    _write_csv(
        source_path,
        (
            _documented_static_row(
                endpoint_kind='source',
                endpoint_key='source:1001',
                endpoint_id='1001',
                total_applied_shift_ms='12.5',
            ),
        ),
    )
    _write_csv(
        receiver_path,
        (
            _documented_static_row(
                endpoint_kind='receiver',
                endpoint_key='receiver:2001',
                endpoint_id='2001',
                total_applied_shift_ms='-3.25',
            ),
        ),
    )

    result = import_refraction_static_source_receiver_csvs(
        source_table_path=source_path,
        receiver_table_path=receiver_path,
        source_job_id='refraction-source-job',
    )

    assert result.is_valid is True
    assert result.source_static_by_endpoint_key['source:1001'].applied_shift_s == (
        pytest.approx(0.0125)
    )
    assert result.receiver_static_by_endpoint_key[
        'receiver:2001'
    ].applied_shift_s == pytest.approx(-0.00325)
    assert result.source_static_by_endpoint_key['source:1001'].source_job_id == (
        'refraction-source-job'
    )


def test_import_static_table_normalizes_to_seconds(tmp_path: Path) -> None:
    table_path = tmp_path / 'seconds.csv'
    _write_csv(
        table_path,
        (
            _canonical_row(
                applied_shift_ms=None,
                applied_shift_s='0.125',
            ),
        ),
    )

    result = import_refraction_static_table_csv(table_path)

    assert result.is_valid is True
    assert result.source_static_by_endpoint_key['source:1001'].applied_shift_s == (
        pytest.approx(0.125)
    )


def test_import_static_table_rejects_duplicate_endpoint_keys(tmp_path: Path) -> None:
    table_path = tmp_path / 'duplicates.csv'
    _write_csv(
        table_path,
        (
            _canonical_row(endpoint_key='source:duplicate', endpoint_id='1001'),
            _canonical_row(endpoint_key='source:duplicate', endpoint_id='1002'),
        ),
    )

    result = import_refraction_static_table_csv(table_path)

    assert result.is_valid is False
    assert result.source_static_by_endpoint_key == {}
    assert any('duplicate endpoint_key for source' in error for error in result.errors)


def test_import_static_table_rejects_bad_sign_convention(tmp_path: Path) -> None:
    table_path = tmp_path / 'bad-sign.csv'
    _write_csv(
        table_path,
        (
            _canonical_row(sign_convention='delay_positive_ms'),
        ),
    )

    result = import_refraction_static_table_csv(table_path)

    assert result.is_valid is False
    assert result.receiver_static_by_endpoint_key == {}
    assert any('sign_convention must be' in error for error in result.errors)
