from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.statics.refraction.artifacts.export_types import (
    REFRACTION_STATIC_EXPORT_SIGN_CONVENTION,
    RefractionStaticEndpointExportRow,
    RefractionStaticExportBundle,
)
from app.statics.refraction.application.lsst_export import (
    REFRACTION_LSST_CARDS_TXT_NAME,
    REFRACTION_LSST_CSV_NAME,
    REFRACTION_LSST_FORMAT_NAME,
    REFRACTION_LSST_FORMAT_VERSION,
    REFRACTION_LSST_PLUS_COLUMNS,
    REFRACTION_LSST_PLUS_CARDS_TXT_NAME,
    REFRACTION_LSST_PLUS_CSV_NAME,
    REFRACTION_LSST_PLUS_FORMAT_NAME,
    REFRACTION_LSST_PLUS_FORMAT_VERSION,
    REFRACTION_LSST_REQUIRED_COLUMNS,
    RefractionStaticLsstExportError,
    format_refraction_lsst_cards_txt,
    format_refraction_lsst_csv,
    format_refraction_lsst_plus_cards_txt,
    format_refraction_lsst_plus_csv,
    write_refraction_lsst_cards_txt,
    write_refraction_lsst_csv,
    write_refraction_lsst_plus_cards_txt,
    write_refraction_lsst_plus_csv,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_IMPORTS = {
    'app.api.routers',
    'app.api.schemas',
    'app.main',
    'app.statics.refraction.application.workflow',
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

module = importlib.import_module('app.statics.refraction.application.lsst_export')
assert module.format_refraction_lsst_csv is not None
assert module.format_refraction_lsst_cards_txt is not None
assert module.format_refraction_lsst_plus_csv is not None
assert module.format_refraction_lsst_plus_cards_txt is not None

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
    bundle = _multilayer_bundle()

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


def test_lsst_cards_txt_has_header_and_card_rows() -> None:
    text = format_refraction_lsst_cards_txt(_basic_bundle())
    lines = text.splitlines()

    assert lines[:5] == [
        '# format=lsst',
        '# schema_version=1',
        '# sign_convention=corrected(t) = raw(t - shift_s)',
        '# units=ms',
        '# missing_values=omit',
    ]
    assert 'SDT1 s100 12.345600' in lines
    assert 'SVV2 s100 2400.000' in lines


def test_lsst_cards_txt_is_not_csv_mirror() -> None:
    bundle = _basic_bundle()
    text = format_refraction_lsst_cards_txt(bundle)

    assert text != format_refraction_lsst_csv(bundle)
    assert not text.startswith('format_name,')
    assert 'format_name,format_version' not in text


def test_lsst_cards_txt_emits_source_and_receiver_stat_cards() -> None:
    lines = format_refraction_lsst_cards_txt(_basic_bundle()).splitlines()

    assert 'SSTAT s100 -3.250000' in lines
    assert 'RSTAT r200 4.500000' in lines
    assert 'SDT1 s100 12.345600' in lines
    assert 'RDT1 r200 23.000000' in lines


def test_lsst_cards_txt_emits_multilayer_cards_when_available() -> None:
    lines = format_refraction_lsst_cards_txt(_multilayer_bundle()).splitlines()

    assert 'SDT2 s100 20.000000' in lines
    assert 'SDT3 s100 30.000000' in lines
    assert 'SVV3 s100 3600.000' in lines
    assert 'SVSB s100 5000.000' in lines
    assert 'RDT2 r200 21.000000' in lines
    assert 'RDT3 r200 31.000000' in lines
    assert 'RVV3 r200 3650.000' in lines
    assert 'RVSB r200 5100.000' in lines


def test_lsst_cards_txt_writes_expected_file_name(tmp_path: Path) -> None:
    path = tmp_path / REFRACTION_LSST_CARDS_TXT_NAME

    write_refraction_lsst_cards_txt(_basic_bundle(), path)

    assert path.read_text(encoding='utf-8') == format_refraction_lsst_cards_txt(
        _basic_bundle()
    )


def test_lsst_plus_export_contains_endpoint_metadata() -> None:
    rows, fieldnames = _read_csv_text(format_refraction_lsst_plus_csv(_lsst_plus_bundle()))

    assert tuple(fieldnames) == REFRACTION_LSST_PLUS_COLUMNS
    assert rows[0]['format_name'] == REFRACTION_LSST_PLUS_FORMAT_NAME
    assert rows[0]['format_version'] == str(REFRACTION_LSST_PLUS_FORMAT_VERSION)
    assert rows[0]['source_job_id'] == 'refraction-job-504'
    assert rows[0]['sign_convention'] == REFRACTION_STATIC_EXPORT_SIGN_CONVENTION
    assert rows[0]['endpoint_kind'] == 'source'
    assert rows[0]['endpoint_key'] == 's100'
    assert rows[0]['endpoint_id'] == '100'
    assert rows[0]['node_id'] == '10'
    assert rows[0]['x_m'] == '1000.000'
    assert rows[0]['y_m'] == '2000.000'
    assert rows[0]['surface_elevation_m'] == '25.000'
    assert rows[0]['static_status'] == 'ok'
    assert rows[1]['endpoint_kind'] == 'receiver'
    assert rows[1]['endpoint_key'] == 'r200'
    assert rows[1]['endpoint_id'] == '200'
    assert rows[1]['node_id'] == '20'
    assert rows[1]['x_m'] == '1010.000'
    assert rows[1]['y_m'] == '2010.000'
    assert rows[1]['surface_elevation_m'] == '30.000'
    assert rows[1]['static_status'] == 'ok'


def test_lsst_plus_export_contains_component_static_values() -> None:
    rows, _fieldnames = _read_csv_text(format_refraction_lsst_plus_csv(_lsst_plus_bundle()))
    source_row = rows[0]
    receiver_row = rows[1]

    assert source_row['total_applied_shift_ms'] == '-3.250000'
    assert source_row['weathering_correction_ms'] == '-4.000000'
    assert source_row['elevation_correction_ms'] == '0.750000'
    assert source_row['source_field_shift_ms'] == '1.250000'
    assert source_row['manual_static_shift_ms'] == '0.500000'
    assert source_row['source_total_with_field_shift_ms'] == '-2.000000'
    assert source_row['source_depth_m'] == '10.000'
    assert source_row['source_depth_shift_ms'] == '12.500000'
    assert source_row['uphole_time_ms'] == '4.000000'
    assert source_row['uphole_shift_ms'] == '-4.000000'
    assert source_row['manual_static_status'] == 'ok'
    assert source_row['source_field_static_status'] == 'ok'
    assert source_row['source_depth_status'] == 'ok'
    assert source_row['uphole_status'] == 'ok'
    assert source_row['receiver_field_shift_ms'] == ''
    assert receiver_row['receiver_field_shift_ms'] == '-0.250000'
    assert receiver_row['receiver_total_with_field_shift_ms'] == '4.250000'
    assert receiver_row['receiver_field_static_status'] == 'ok'
    assert receiver_row['source_field_shift_ms'] == ''
    assert receiver_row['source_depth_m'] == ''
    assert receiver_row['uphole_time_ms'] == ''


def test_lsst_plus_export_contains_three_layer_values() -> None:
    rows, _fieldnames = _read_csv_text(format_refraction_lsst_plus_csv(_lsst_plus_bundle()))
    source_row = rows[0]

    assert source_row['t1_ms'] == '12.345600'
    assert source_row['t2_ms'] == '20.000000'
    assert source_row['t3_ms'] == '30.000000'
    assert source_row['sh1_weathering_thickness_m'] == '8.250'
    assert source_row['sh2_weathering_thickness_m'] == '12.000'
    assert source_row['sh3_weathering_thickness_m'] == '16.000'
    assert source_row['v1_m_s'] == '800.000'
    assert source_row['v2_m_s'] == '2400.000'
    assert source_row['v3_m_s'] == '3600.000'
    assert source_row['vsub_m_s'] == '5000.000'


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

    rows, _fieldnames = _read_csv_text(format_refraction_lsst_plus_csv(bundle))

    assert rows[0]['static_status'] == 'invalid_velocity_order'
    assert rows[0]['x_m'] == ''
    assert rows[0]['total_applied_shift_ms'] == ''
    assert rows[0]['t1_ms'] == ''


def test_lsst_plus_export_is_deterministic() -> None:
    bundle = _lsst_plus_source_only_bundle()
    expected = (
        'format_name,format_version,source_job_id,endpoint_kind,endpoint_key,'
        'endpoint_id,node_id,x_m,y_m,surface_elevation_m,t1_ms,v1_m_s,'
        'v2_m_s,sh1_weathering_thickness_m,weathering_correction_ms,'
        'elevation_correction_ms,total_static_ms,total_applied_shift_ms,'
        'static_status,sign_convention,t2_ms,t3_ms,v3_m_s,vsub_m_s,'
        'sh2_weathering_thickness_m,sh3_weathering_thickness_m,'
        'total_weathering_thickness_m,source_depth_m,source_depth_shift_ms,'
        'source_depth_status,uphole_time_ms,uphole_shift_ms,uphole_status,'
        'manual_static_shift_ms,manual_static_status,source_field_shift_ms,'
        'source_field_static_status,source_total_with_field_shift_ms,'
        'receiver_field_shift_ms,receiver_field_static_status,'
        'receiver_total_with_field_shift_ms\n'
        'lsst_plus,1,refraction-job-504,source,s100,100,10,1000.000,'
        '2000.000,25.000,12.345600,800.000,2400.000,8.250,-4.000000,'
        '0.750000,-3.250000,-3.250000,ok,'
        'corrected(t) = raw(t - shift_s),20.000000,30.000000,3600.000,'
        '5000.000,12.000,16.000,,10.000,12.500000,ok,4.000000,'
        '-4.000000,ok,0.500000,ok,1.250000,ok,-2.000000,,,\n'
    )

    assert format_refraction_lsst_plus_csv(bundle) == expected
    assert format_refraction_lsst_plus_csv(bundle) == format_refraction_lsst_plus_csv(
        bundle
    )


def test_lsst_plus_export_writes_expected_file_name(tmp_path: Path) -> None:
    path = tmp_path / REFRACTION_LSST_PLUS_CSV_NAME

    write_refraction_lsst_plus_csv(_lsst_plus_source_only_bundle(), path)

    assert path.read_text(encoding='utf-8') == format_refraction_lsst_plus_csv(
        _lsst_plus_source_only_bundle()
    )


def test_lsst_plus_cards_txt_has_endpoint_component_and_layer_rows() -> None:
    text = format_refraction_lsst_plus_cards_txt(_lsst_plus_bundle())
    lines = text.splitlines()

    assert lines[:5] == [
        '# format=lsst_plus',
        '# schema_version=1',
        '# sign_convention=corrected(t) = raw(t - shift_s)',
        '# units=ms',
        '# missing_values=omit',
    ]
    assert (
        'SRC s100 station=1000 x=1000.000 y=2000.000 elev=25.000 status=ok'
        in lines
    )
    assert (
        'REC r200 station=2000 x=1010.000 y=2010.000 elev=30.000 status=ok'
        in lines
    )
    assert (
        'STC source s100 total=-3.250000 weathering=-4.000000 '
        'elevation=0.750000 field=1.250000 manual=0.500000'
    ) in lines
    assert (
        'STC receiver r200 total=4.500000 weathering=-2.500000 '
        'elevation=1.000000 field=-0.250000 manual=-0.250000'
    ) in lines
    assert (
        'LYR source s100 t1=12.345600 t2=20.000000 t3=30.000000 '
        'sh1=8.250 sh2=12.000 sh3=16.000 v1=800.000 v2=2400.000 '
        'v3=3600.000 vsub=5000.000'
    ) in lines
    assert 'LYR receiver r200 t1=23.000000 sh1=9.500 v1=800.000 v2=2300.000' in lines


def test_lsst_plus_cards_txt_is_not_csv_mirror() -> None:
    bundle = _lsst_plus_bundle()
    text = format_refraction_lsst_plus_cards_txt(bundle)

    assert text != format_refraction_lsst_plus_csv(bundle)
    assert not text.startswith('format_name,')
    assert 'format_name,format_version' not in text


def test_lsst_plus_cards_txt_is_deterministic() -> None:
    bundle = _lsst_plus_source_only_bundle()
    expected = (
        '# format=lsst_plus\n'
        '# schema_version=1\n'
        '# sign_convention=corrected(t) = raw(t - shift_s)\n'
        '# units=ms\n'
        '# missing_values=omit\n'
        'SRC s100 station=1000 x=1000.000 y=2000.000 elev=25.000 status=ok\n'
        'STC source s100 total=-3.250000 weathering=-4.000000 '
        'elevation=0.750000 field=1.250000 manual=0.500000\n'
        'LYR source s100 t1=12.345600 t2=20.000000 t3=30.000000 '
        'sh1=8.250 sh2=12.000 sh3=16.000 v1=800.000 v2=2400.000 '
        'v3=3600.000 vsub=5000.000\n'
    )

    assert format_refraction_lsst_plus_cards_txt(bundle) == expected
    assert format_refraction_lsst_plus_cards_txt(bundle) == (
        format_refraction_lsst_plus_cards_txt(bundle)
    )


def test_lsst_plus_cards_txt_writes_expected_file_name(tmp_path: Path) -> None:
    path = tmp_path / REFRACTION_LSST_PLUS_CARDS_TXT_NAME

    write_refraction_lsst_plus_cards_txt(_lsst_plus_source_only_bundle(), path)

    assert path.read_text(encoding='utf-8') == format_refraction_lsst_plus_cards_txt(
        _lsst_plus_source_only_bundle()
    )


def _read_csv_text(text: str) -> tuple[list[dict[str, str]], list[str]]:
    reader = csv.DictReader(io.StringIO(text))
    return list(reader), list(reader.fieldnames or ())


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


def _multilayer_bundle() -> RefractionStaticExportBundle:
    return RefractionStaticExportBundle(
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
