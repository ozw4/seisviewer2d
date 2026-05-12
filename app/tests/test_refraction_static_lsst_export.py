from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.services.refraction_static_export_types import (
    REFRACTION_STATIC_EXPORT_SIGN_CONVENTION,
    RefractionStaticEndpointExportRow,
    RefractionStaticExportBundle,
)
from app.services.refraction_static_lsst_export import (
    REFRACTION_LSST_CSV_NAME,
    REFRACTION_LSST_FORMAT_NAME,
    REFRACTION_LSST_FORMAT_VERSION,
    REFRACTION_LSST_PLUS_CARDS_NAME,
    REFRACTION_LSST_PLUS_FORMAT_NAME,
    REFRACTION_LSST_PLUS_SCHEMA_VERSION,
    REFRACTION_LSST_REQUIRED_COLUMNS,
    RefractionStaticLsstExportError,
    format_refraction_lsst_csv,
    format_refraction_lsst_plus_cards,
    write_refraction_lsst_csv,
    write_refraction_lsst_plus_cards,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_IMPORTS = {
    'app.api.routers',
    'app.api.schemas',
    'app.main',
    'app.services.refraction_static_service',
    'app.trace_store.reader',
    'numpy',
    'segyio',
}


def test_lsst_export_imports_dependency_light() -> None:
    code = f"""
from __future__ import annotations

import importlib
import json
import sys

for name in {sorted(_FORBIDDEN_IMPORTS)!r}:
    sys.modules.pop(name, None)

module = importlib.import_module('app.services.refraction_static_lsst_export')
assert module.format_refraction_lsst_csv is not None
assert module.format_refraction_lsst_plus_cards is not None

forbidden = set({sorted(_FORBIDDEN_IMPORTS)!r})
print(json.dumps(sorted(name for name in sys.modules if name in forbidden)))
"""
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert json.loads(result.stdout) == []


def test_lsst_export_header_contains_required_schema_columns() -> None:
    text = format_refraction_lsst_csv(_basic_bundle())
    rows, fieldnames = _read_csv_text(text)

    assert tuple(fieldnames) == REFRACTION_LSST_REQUIRED_COLUMNS
    assert rows[0]['format_name'] == REFRACTION_LSST_FORMAT_NAME
    assert rows[0]['format_version'] == str(REFRACTION_LSST_FORMAT_VERSION)
    assert rows[0]['source_job_id'] == 'refraction-job-503'
    assert rows[0]['sign_convention'] == REFRACTION_STATIC_EXPORT_SIGN_CONVENTION


def test_lsst_export_emits_source_and_receiver_t1_rows() -> None:
    text = format_refraction_lsst_csv(_basic_bundle())
    rows, _fieldnames = _read_csv_text(text)

    assert [row['endpoint_kind'] for row in rows] == ['source', 'receiver']
    assert rows[0]['endpoint_key'] == 's100'
    assert rows[0]['endpoint_id'] == '100'
    assert rows[0]['t1_ms'] == '12.345600'
    assert rows[0]['total_static_ms'] == '-3.250000'
    assert rows[0]['total_applied_shift_ms'] == '-3.250000'
    assert rows[0]['v1_m_s'] == '800.000'
    assert rows[1]['endpoint_key'] == 'r200'
    assert rows[1]['endpoint_id'] == '200'
    assert rows[1]['t1_ms'] == '23.000000'
    assert rows[1]['total_applied_shift_ms'] == '4.500000'
    assert rows[1]['v2_m_s'] == '2300.000'


def test_lsst_export_emits_multilayer_columns_when_available() -> None:
    bundle = RefractionStaticExportBundle(
        source_job_id='refraction-job-503',
        source_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='s100',
                endpoint_id=100,
                node_id=10,
                x_m=1000.0,
                y_m=2000.0,
                elevation_m=25.0,
                t1_s=0.012,
                t2_s=0.020,
                t3_s=0.030,
                v1_m_s=800.0,
                v2_m_s=2400.0,
                v3_m_s=3600.0,
                vsub_m_s=5000.0,
                sh1_m=8.0,
                sh2_m=12.0,
                sh3_m=16.0,
                total_weathering_thickness_m=36.0,
                weathering_correction_s=-0.003,
                elevation_correction_s=0.001,
                total_applied_shift_s=-0.004,
            ),
        ),
        receiver_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='receiver',
                endpoint_key='r200',
                endpoint_id=200,
                node_id=20,
                x_m=1010.0,
                y_m=2010.0,
                elevation_m=30.0,
                t1_s=0.011,
                t2_s=0.021,
                t3_s=0.031,
                v1_m_s=800.0,
                v2_m_s=2350.0,
                v3_m_s=3650.0,
                vsub_m_s=5100.0,
                sh1_m=7.0,
                sh2_m=11.0,
                sh3_m=15.0,
                total_weathering_thickness_m=33.0,
                weathering_correction_s=-0.002,
                elevation_correction_s=0.0015,
                total_applied_shift_s=0.005,
            ),
        ),
    )

    rows, fieldnames = _read_csv_text(format_refraction_lsst_csv(bundle))

    assert {
        't2_ms',
        't3_ms',
        'v3_m_s',
        'vsub_m_s',
        'sh2_weathering_thickness_m',
        'sh3_weathering_thickness_m',
        'total_weathering_thickness_m',
    } <= set(fieldnames)
    assert rows[0]['t2_ms'] == '20.000000'
    assert rows[0]['t3_ms'] == '30.000000'
    assert rows[0]['v3_m_s'] == '3600.000'
    assert rows[0]['vsub_m_s'] == '5000.000'
    assert rows[0]['total_weathering_thickness_m'] == '36.000'
    assert rows[1]['t2_ms'] == '21.000000'
    assert rows[1]['t3_ms'] == '31.000000'
    assert rows[1]['v3_m_s'] == '3650.000'
    assert rows[1]['vsub_m_s'] == '5100.000'
    assert rows[1]['total_weathering_thickness_m'] == '33.000'


def test_lsst_export_rejects_invalid_rows_when_configured() -> None:
    bundle = RefractionStaticExportBundle(
        source_job_id='refraction-job-503',
        source_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='s-bad',
                static_status='inactive_endpoint',
            ),
        ),
    )

    with pytest.raises(RefractionStaticLsstExportError, match='inactive_endpoint'):
        format_refraction_lsst_csv(
            bundle,
            fail_on_invalid_static_status=True,
        )


def test_lsst_export_marks_invalid_rows_when_included() -> None:
    bundle = RefractionStaticExportBundle(
        source_job_id='refraction-job-503',
        source_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='s-bad',
                endpoint_id=999,
                static_status='inactive_endpoint',
            ),
        ),
    )

    rows, _fieldnames = _read_csv_text(
        format_refraction_lsst_csv(
            bundle,
            fail_on_invalid_static_status=False,
            include_inactive_endpoints=True,
        )
    )

    assert rows == [
        {
            'format_name': 'lsst',
            'format_version': '1',
            'source_job_id': 'refraction-job-503',
            'endpoint_kind': 'source',
            'endpoint_key': 's-bad',
            'endpoint_id': '999',
            'node_id': '',
            'x_m': '',
            'y_m': '',
            'surface_elevation_m': '',
            't1_ms': '',
            'v1_m_s': '',
            'v2_m_s': '',
            'sh1_weathering_thickness_m': '',
            'weathering_correction_ms': '',
            'elevation_correction_ms': '',
            'total_static_ms': '',
            'total_applied_shift_ms': '',
            'static_status': 'inactive_endpoint',
            'sign_convention': REFRACTION_STATIC_EXPORT_SIGN_CONVENTION,
        }
    ]


def test_lsst_export_skips_invalid_rows_when_not_included() -> None:
    bundle = RefractionStaticExportBundle(
        source_job_id='refraction-job-503',
        source_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='s-bad',
                static_status='inactive_endpoint',
            ),
        ),
    )

    text = format_refraction_lsst_csv(
        bundle,
        fail_on_invalid_static_status=False,
        include_inactive_endpoints=False,
    )

    rows, fieldnames = _read_csv_text(text)
    assert fieldnames == list(REFRACTION_LSST_REQUIRED_COLUMNS)
    assert rows == []


def test_lsst_export_requires_source_job_id() -> None:
    with pytest.raises(RefractionStaticLsstExportError, match='source_job_id'):
        format_refraction_lsst_csv(_basic_bundle(source_job_id=None))


def test_lsst_export_requires_documented_values_for_ok_rows() -> None:
    bundle = RefractionStaticExportBundle(
        source_job_id='refraction-job-503',
        source_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='s-incomplete',
                endpoint_id=100,
                static_status='ok',
            ),
        ),
    )

    with pytest.raises(RefractionStaticLsstExportError, match='node_id'):
        format_refraction_lsst_csv(bundle)


def test_lsst_export_is_deterministic() -> None:
    bundle = _basic_bundle()

    assert format_refraction_lsst_csv(bundle) == format_refraction_lsst_csv(bundle)


def test_lsst_export_writes_expected_file_name(tmp_path: Path) -> None:
    path = tmp_path / REFRACTION_LSST_CSV_NAME

    write_refraction_lsst_csv(_basic_bundle(), path)

    assert path.read_text(encoding='utf-8') == format_refraction_lsst_csv(
        _basic_bundle()
    )


def test_lsst_plus_export_contains_endpoint_metadata() -> None:
    text = format_refraction_lsst_plus_cards(_lsst_plus_bundle())
    lines = text.splitlines()

    assert f'# format={REFRACTION_LSST_PLUS_FORMAT_NAME}' in lines
    assert f'# schema_version={REFRACTION_LSST_PLUS_SCHEMA_VERSION}' in lines
    assert f'# sign_convention={REFRACTION_STATIC_EXPORT_SIGN_CONVENTION}' in lines
    assert '# units=ms' in lines
    assert '# delimiter=|' in lines
    assert _card_fields(_card(lines, 'SRC|s100|')) == {
        'endpoint_id': '100',
        'node_id': '10',
        'station': '1000',
        'x': '1000.000',
        'y': '2000.000',
        'elev': '25.000',
        'status': 'ok',
    }
    assert _card_fields(_card(lines, 'REC|r200|')) == {
        'endpoint_id': '200',
        'node_id': '20',
        'station': '2000',
        'x': '1010.000',
        'y': '2010.000',
        'elev': '30.000',
        'status': 'ok',
    }


def test_lsst_plus_export_contains_component_static_values() -> None:
    fields = _card_fields(
        _card(format_refraction_lsst_plus_cards(_lsst_plus_bundle()).splitlines(), 'STC|source|s100|')
    )
    receiver_fields = _card_fields(
        _card(format_refraction_lsst_plus_cards(_lsst_plus_bundle()).splitlines(), 'STC|receiver|r200|')
    )

    assert fields['total'] == '-3.250000'
    assert fields['weathering'] == '-4.000000'
    assert fields['elevation'] == '0.750000'
    assert fields['source_field_shift_ms'] == '1.250000'
    assert fields['manual'] == '0.500000'
    assert fields['source_total_with_field_shift_ms'] == '-2.000000'
    assert fields['source_depth_m'] == '10.000'
    assert fields['source_depth'] == '12.500000'
    assert fields['uphole_time'] == '4.000000'
    assert fields['uphole'] == '-4.000000'
    assert fields['manual_status'] == 'ok'
    assert fields['source_field_static_status'] == 'ok'
    assert fields['source_depth_status'] == 'ok'
    assert fields['uphole_status'] == 'ok'
    assert 'field' not in fields
    assert 'field_status' not in fields
    assert 'total_with_field' not in fields
    assert receiver_fields['receiver_field_shift_ms'] == '-0.250000'
    assert receiver_fields['receiver_total_with_field_shift_ms'] == '4.250000'
    assert receiver_fields['receiver_field_static_status'] == 'ok'
    assert 'field' not in receiver_fields
    assert 'field_status' not in receiver_fields
    assert 'total_with_field' not in receiver_fields


def test_lsst_plus_export_contains_three_layer_values() -> None:
    fields = _card_fields(
        _card(format_refraction_lsst_plus_cards(_lsst_plus_bundle()).splitlines(), 'LYR|source|s100|')
    )

    assert fields['t1'] == '12.345600'
    assert fields['t2'] == '20.000000'
    assert fields['t3'] == '30.000000'
    assert fields['sh1'] == '8.250'
    assert fields['sh2'] == '12.000'
    assert fields['sh3'] == '16.000'
    assert fields['v1'] == '800.000'
    assert fields['v2'] == '2400.000'
    assert fields['v3'] == '3600.000'
    assert fields['vsub'] == '5000.000'


def test_lsst_plus_export_includes_status_for_invalid_rows() -> None:
    bundle = RefractionStaticExportBundle(
        source_job_id='refraction-job-504',
        source_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='s-bad',
                station_id='1000',
                x_m=float('nan'),
                total_applied_shift_s=float('inf'),
                static_status='invalid_velocity_order',
            ),
        ),
    )

    lines = format_refraction_lsst_plus_cards(bundle).splitlines()

    assert _card_fields(_card(lines, 'SRC|s-bad|'))['status'] == (
        'invalid_velocity_order'
    )
    assert _card_fields(_card(lines, 'STC|source|s-bad|'))['total'] == 'nan'
    assert _card_fields(_card(lines, 'LYR|source|s-bad|'))['t1'] == 'nan'


def test_lsst_plus_export_is_deterministic() -> None:
    bundle = _lsst_plus_source_only_bundle()
    expected = (
        '# format=lsst_plus\n'
        '# schema_version=1\n'
        '# source_job_id=refraction-job-504\n'
        '# sign_convention=corrected(t) = raw(t - shift_s)\n'
        '# units=ms\n'
        '# distance_units=m\n'
        '# velocity_units=m_s\n'
        '# delimiter=|\n'
        '# missing_numeric=nan\n'
        '# missing_text=nan\n'
        '# row_order=source_rows_then_receiver_rows\n'
        'SRC|s100|endpoint_id=100|node_id=10|station=1000|'
        'x=1000.000|y=2000.000|elev=25.000|status=ok\n'
        'STC|source|s100|total=-3.250000|weathering=-4.000000|'
        'elevation=0.750000|source_field_shift_ms=1.250000|'
        'manual=0.500000|source_total_with_field_shift_ms=-2.000000|'
        'source_depth_m=10.000|'
        'source_depth=12.500000|uphole_time=4.000000|uphole=-4.000000|'
        'manual_status=ok|source_field_static_status=ok|'
        'source_depth_status=ok|uphole_status=ok\n'
        'LYR|source|s100|t1=12.345600|t2=20.000000|t3=30.000000|'
        'sh1=8.250|sh2=12.000|sh3=16.000|v1=800.000|'
        'v2=2400.000|v3=3600.000|vsub=5000.000\n'
    )

    assert format_refraction_lsst_plus_cards(bundle) == expected
    assert format_refraction_lsst_plus_cards(bundle) == (
        format_refraction_lsst_plus_cards(bundle)
    )


def test_lsst_plus_export_writes_expected_file_name(tmp_path: Path) -> None:
    path = tmp_path / REFRACTION_LSST_PLUS_CARDS_NAME

    write_refraction_lsst_plus_cards(_lsst_plus_source_only_bundle(), path)

    assert path.read_text(encoding='utf-8') == format_refraction_lsst_plus_cards(
        _lsst_plus_source_only_bundle()
    )


def _read_csv_text(text: str) -> tuple[list[dict[str, str]], list[str]]:
    reader = csv.DictReader(io.StringIO(text))
    return list(reader), list(reader.fieldnames or ())


def _card(lines: list[str], prefix: str) -> str:
    matches = [line for line in lines if line.startswith(prefix)]
    assert len(matches) == 1
    return matches[0]


def _card_fields(line: str) -> dict[str, str]:
    return dict(
        part.split('=', maxsplit=1) for part in line.split('|') if '=' in part
    )


def _basic_bundle(
    *,
    source_job_id: str | None = 'refraction-job-503',
) -> RefractionStaticExportBundle:
    return RefractionStaticExportBundle(
        source_job_id=source_job_id,
        source_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='s100',
                endpoint_id=100,
                station_id=1000,
                node_id=10,
                x_m=1000.0,
                y_m=2000.0,
                elevation_m=25.0,
                t1_s=0.0123456,
                v1_m_s=800.0,
                v2_m_s=2400.0,
                sh1_m=8.25,
                weathering_correction_s=-0.004,
                elevation_correction_s=0.00075,
                total_applied_shift_s=-0.00325,
            ),
        ),
        receiver_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='receiver',
                endpoint_key='r200',
                endpoint_id=200,
                station_id=2000,
                node_id=20,
                x_m=1010.0,
                y_m=2010.0,
                elevation_m=30.0,
                t1_s=0.023,
                v1_m_s=800.0,
                v2_m_s=2300.0,
                sh1_m=9.5,
                weathering_correction_s=-0.0025,
                elevation_correction_s=0.001,
                total_applied_shift_s=0.0045,
            ),
        ),
    )


def _lsst_plus_source_row() -> RefractionStaticEndpointExportRow:
    return RefractionStaticEndpointExportRow(
        endpoint_kind='source',
        endpoint_key='s100',
        endpoint_id=100,
        station_id=1000,
        node_id=10,
        x_m=1000.0,
        y_m=2000.0,
        elevation_m=25.0,
        t1_s=0.0123456,
        t2_s=0.020,
        t3_s=0.030,
        v1_m_s=800.0,
        v2_m_s=2400.0,
        v3_m_s=3600.0,
        vsub_m_s=5000.0,
        sh1_m=8.25,
        sh2_m=12.0,
        sh3_m=16.0,
        weathering_correction_s=-0.004,
        elevation_correction_s=0.00075,
        field_correction_s=0.00125,
        source_depth_m=10.0,
        source_depth_shift_s=0.0125,
        source_depth_status='ok',
        uphole_time_s=0.004,
        uphole_shift_s=-0.004,
        uphole_status='ok',
        manual_static_shift_s=0.0005,
        manual_static_status='ok',
        field_static_status='ok',
        total_with_field_shift_s=-0.002,
        total_applied_shift_s=-0.00325,
    )


def _lsst_plus_bundle() -> RefractionStaticExportBundle:
    return RefractionStaticExportBundle(
        source_job_id='refraction-job-504',
        source_rows=(_lsst_plus_source_row(),),
        receiver_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='receiver',
                endpoint_key='r200',
                endpoint_id=200,
                station_id=2000,
                node_id=20,
                x_m=1010.0,
                y_m=2010.0,
                elevation_m=30.0,
                t1_s=0.023,
                v1_m_s=800.0,
                v2_m_s=2300.0,
                sh1_m=9.5,
                weathering_correction_s=-0.0025,
                elevation_correction_s=0.001,
                field_correction_s=-0.00025,
                manual_static_shift_s=-0.00025,
                manual_static_status='ok',
                field_static_status='ok',
                total_with_field_shift_s=0.00425,
                total_applied_shift_s=0.0045,
            ),
        ),
    )


def _lsst_plus_source_only_bundle() -> RefractionStaticExportBundle:
    return RefractionStaticExportBundle(
        source_job_id='refraction-job-504',
        source_rows=(_lsst_plus_source_row(),),
    )
