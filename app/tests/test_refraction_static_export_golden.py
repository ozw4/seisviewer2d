from __future__ import annotations

import csv
import io
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.statics.refraction.artifacts import (
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    write_refraction_first_break_time_export_csv,
    write_refraction_time_term_spreadsheet_csv_from_static_tables,
)
from app.statics.refraction.application.export_service import (
    _canonical_static_table_rows,
    _write_canonical_static_table_csv,
)
from app.statics.refraction.application.lsst_export import (
    format_refraction_lsst_csv,
    format_refraction_lsst_plus_csv,
)
from app.statics.refraction.artifacts.table_validator import (
    CANONICAL_STATIC_TABLE_OPTIONAL_COLUMNS,
    CANONICAL_STATIC_TABLE_REQUIRED_COLUMNS,
    validate_canonical_static_table_rows,
)
from app.tests._refraction_static_artifact_helpers import _request, _result
from app.tests._refraction_static_export_fixtures import (
    GOLDEN_SOURCE_JOB_ID,
    canonical_static_table_rows,
    field_component_export_bundle,
    field_component_static_table_rows,
    first_break_time_rows,
    invalid_status_export_bundle,
    one_layer_export_bundle,
    three_layer_export_bundle,
    two_layer_export_bundle,
)


def test_lsst_golden_one_layer() -> None:
    expected = (
        'format_name,format_version,source_job_id,endpoint_kind,endpoint_key,'
        'endpoint_id,node_id,x_m,y_m,surface_elevation_m,t1_ms,v1_m_s,'
        'v2_m_s,sh1_weathering_thickness_m,weathering_correction_ms,'
        'elevation_correction_ms,total_static_ms,total_applied_shift_ms,'
        'static_status,sign_convention\n'
        'lsst,1,refraction-golden-job-522,source,source:1001,1001,10,'
        '1000.000,2000.000,25.000,12.500000,800.000,2400.000,10.607,'
        '-8.838835,21.338835,12.500000,12.500000,ok,'
        'corrected(t) = raw(t - shift_s)\n'
        'lsst,1,refraction-golden-job-522,receiver,receiver:2001,2001,'
        '20,1010.000,2010.000,30.000,8.500000,800.000,2300.000,'
        '7.253,-5.912671,2.662671,-3.250000,-3.250000,ok,'
        'corrected(t) = raw(t - shift_s)\n'
    )

    assert format_refraction_lsst_csv(one_layer_export_bundle()) == expected


def test_lsst_golden_two_layer() -> None:
    expected = (
        'format_name,format_version,source_job_id,endpoint_kind,endpoint_key,'
        'endpoint_id,node_id,x_m,y_m,surface_elevation_m,t1_ms,v1_m_s,'
        'v2_m_s,sh1_weathering_thickness_m,weathering_correction_ms,'
        'elevation_correction_ms,total_static_ms,total_applied_shift_ms,'
        'static_status,sign_convention,t2_ms,v3_m_s,'
        'sh2_weathering_thickness_m,total_weathering_thickness_m\n'
        'lsst,1,refraction-golden-job-522,source,source:1101,1101,11,'
        '1100.000,2100.000,40.000,13.000000,850.000,2450.000,11.782,'
        '-13.871600,26.371600,12.500000,12.500000,ok,'
        'corrected(t) = raw(t - shift_s),21.000000,3600.000,25.182,'
        '36.964\n'
        'lsst,1,refraction-golden-job-522,receiver,receiver:2101,2101,'
        '21,1110.000,2110.000,42.000,10.000000,850.000,2400.000,'
        '9.089,-11.701697,6.701697,-5.000000,-5.000000,ok,'
        'corrected(t) = raw(t - shift_s),18.500000,3550.000,26.441,'
        '35.530\n'
    )

    assert format_refraction_lsst_csv(two_layer_export_bundle()) == expected


def test_lsst_golden_three_layer() -> None:
    expected = (
        'format_name,format_version,source_job_id,endpoint_kind,endpoint_key,'
        'endpoint_id,node_id,x_m,y_m,surface_elevation_m,t1_ms,v1_m_s,'
        'v2_m_s,sh1_weathering_thickness_m,weathering_correction_ms,'
        'elevation_correction_ms,total_static_ms,total_applied_shift_ms,'
        'static_status,sign_convention,t2_ms,t3_ms,v3_m_s,vsub_m_s,'
        'sh2_weathering_thickness_m,sh3_weathering_thickness_m,'
        'total_weathering_thickness_m\n'
        'lsst,1,refraction-golden-job-522,source,source:1201,1201,12,'
        '1200.000,2200.000,50.000,14.000000,900.000,2500.000,13.506,'
        '-20.676157,34.676157,14.000000,14.000000,ok,'
        'corrected(t) = raw(t - shift_s),22.000000,31.000000,3700.000,'
        '5200.000,25.246,38.785,77.537\n'
        'lsst,1,refraction-golden-job-522,receiver,receiver:2201,2201,'
        '22,1210.000,2210.000,52.000,11.000000,900.000,2480.000,'
        '10.624,-18.017836,12.017836,-6.000000,-6.000000,ok,'
        'corrected(t) = raw(t - shift_s),19.000000,28.000000,3650.000,'
        '5100.000,25.552,38.555,74.731\n'
    )

    assert format_refraction_lsst_csv(three_layer_export_bundle()) == expected


def test_lsst_plus_golden_with_field_components() -> None:
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
        'lsst_plus,1,refraction-golden-job-522,source,source:1001,1001,10,'
        '1000.000,2000.000,25.000,12.500000,800.000,2400.000,10.607,'
        '-8.838835,21.338835,12.500000,12.500000,ok,'
        'corrected(t) = raw(t - shift_s),,,,,,,,6.400,8.000000,ok,'
        '3.000000,-3.000000,ok,1.250000,ok,6.250000,ok,18.750000,,,\n'
        'lsst_plus,1,refraction-golden-job-522,receiver,receiver:2001,2001,'
        '20,1010.000,2010.000,30.000,8.500000,800.000,2300.000,'
        '7.253,-5.912671,2.662671,-3.250000,-3.250000,ok,'
        'corrected(t) = raw(t - shift_s),,,,,,,,,,,,,,-2.000000,ok,,,,'
        '-2.000000,ok,-5.250000\n'
    )

    assert format_refraction_lsst_plus_csv(field_component_export_bundle()) == expected


def test_time_term_spreadsheet_golden_with_field_components(tmp_path: Path) -> None:
    source_rows, receiver_rows = field_component_static_table_rows()
    path = tmp_path / REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME
    expected = (
        'schema_version,format_name,format_version,source_job_id,endpoint_kind,'
        'endpoint_key,endpoint_id,station_id,node_id,x_m,y_m,elevation_m,'
        'surface_elevation_m,t1_ms,t2_ms,t3_ms,v1_m_s,v2_m_s,v3_m_s,'
        'vsub_m_s,sh1_m,sh2_m,sh3_m,layer1_base_elevation_m,'
        'layer2_base_elevation_m,final_refractor_elevation_m,'
        'weathering_correction_ms,elevation_correction_ms,'
        'source_depth_correction_ms,uphole_correction_ms,manual_static_ms,'
        'field_correction_ms,total_applied_shift_ms,pick_count,'
        'used_pick_count,pick_count_by_layer,used_pick_count_by_layer,'
        'residual_rms_ms,residual_mad_ms,residual_rms_by_layer_ms,'
        'residual_mad_by_layer_ms,solution_status,weathering_status,'
        'datum_status,source_depth_status,uphole_status,manual_static_status,'
        'field_static_status,static_status,sign_convention\n'
        '1,time_term_spreadsheet,1,refraction-golden-job-522,source,'
        'source:1001,1001,1001,10,1000.000,2000.000,25.000,25.000,'
        '12.500000,,,800.000,2400.000,,,10.607,,,14.393,,14.393,'
        '-8.838835,21.338835,8.000000,-3.000000,1.250000,6.250000,'
        '12.500000,5,4,v2_t1:5,v2_t1:4,1.250000,0.750000,'
        'v2_t1:1.25,v2_t1:0.75,solved,ok,ok,ok,ok,ok,ok,ok,'
        'corrected(t) = raw(t - shift_s)\n'
        '1,time_term_spreadsheet,1,refraction-golden-job-522,receiver,'
        'receiver:2001,2001,2001,20,1010.000,2010.000,30.000,30.000,'
        '8.500000,,,800.000,2300.000,,,7.253,,,22.747,,22.747,'
        '-5.912671,2.662671,,,-2.000000,-2.000000,-3.250000,6,6,'
        'v2_t1:6,v2_t1:6,1.000000,0.500000,v2_t1:1.0,v2_t1:0.5,'
        'solved,ok,ok,not_applicable,not_applicable,ok,ok,ok,'
        'corrected(t) = raw(t - shift_s)\n'
    )

    write_refraction_time_term_spreadsheet_csv_from_static_tables(
        source_rows=source_rows,
        receiver_rows=receiver_rows,
        path=path,
        source_job_id=GOLDEN_SOURCE_JOB_ID,
        include_inactive_endpoints=True,
    )

    assert path.read_text(encoding='utf-8') == expected


def test_first_break_time_export_golden_with_residuals(tmp_path: Path) -> None:
    path = tmp_path / 'refraction_first_break_time_export.csv'

    write_refraction_first_break_time_export_csv(
        result=_result(),
        path=path,
        req=_request(),
        source_job_id=GOLDEN_SOURCE_JOB_ID,
    )

    rows, fieldnames = _read_csv_with_fieldnames(path.read_text(encoding='utf-8'))
    assert tuple(fieldnames) == tuple(first_break_time_rows()[0])
    assert rows == list(first_break_time_rows())
    assert path.read_text(encoding='utf-8') == _csv_text(fieldnames, rows)


def test_first_break_time_export_preserves_text_endpoint_ids(tmp_path: Path) -> None:
    path = tmp_path / 'refraction_first_break_time_export.csv'
    result = replace(
        _result(),
        source_id=np.asarray(['SRC-A', 'SRC-B'], dtype='<U8'),
        receiver_id=np.asarray(['REC-A', 'REC-B'], dtype='<U8'),
    )

    write_refraction_first_break_time_export_csv(
        result=result,
        path=path,
        req=_request(),
        source_job_id=GOLDEN_SOURCE_JOB_ID,
    )

    rows, _fieldnames = _read_csv_with_fieldnames(path.read_text(encoding='utf-8'))
    assert rows[0]['source_id'] == 'SRC-A'
    assert rows[0]['receiver_id'] == 'REC-A'
    assert rows[1]['source_id'] == 'SRC-B'
    assert rows[1]['receiver_id'] == 'REC-B'


def test_canonical_static_table_golden_schema_version(tmp_path: Path) -> None:
    path = tmp_path / 'canonical_source_receiver_static_table.csv'
    source_rows, receiver_rows = field_component_static_table_rows()
    canonical_rows = (
        _canonical_static_table_rows(
            source_rows,
            endpoint_kind='source',
            source_job_id=GOLDEN_SOURCE_JOB_ID,
            fail_on_invalid_static_status=True,
            include_inactive_endpoints=False,
        )
        + _canonical_static_table_rows(
            receiver_rows,
            endpoint_kind='receiver',
            source_job_id=GOLDEN_SOURCE_JOB_ID,
            fail_on_invalid_static_status=True,
            include_inactive_endpoints=False,
        )
    )

    _write_canonical_static_table_csv(path, canonical_rows)

    assert canonical_rows == canonical_static_table_rows()
    assert path.read_text(encoding='utf-8') == _csv_text(
        CANONICAL_STATIC_TABLE_REQUIRED_COLUMNS
        + CANONICAL_STATIC_TABLE_OPTIONAL_COLUMNS,
        canonical_static_table_rows(),
    )
    validation = validate_canonical_static_table_rows(
        canonical_rows,
        columns=CANONICAL_STATIC_TABLE_REQUIRED_COLUMNS
        + CANONICAL_STATIC_TABLE_OPTIONAL_COLUMNS,
    )
    assert validation.is_valid is True
    assert {row['format_version'] for row in canonical_rows} == {'1'}


def test_canonical_static_table_golden_positive_and_negative_shifts() -> None:
    rows = canonical_static_table_rows()

    assert [row['endpoint_kind'] for row in rows] == ['source', 'receiver']
    assert [row['applied_shift_ms'] for row in rows] == [
        '12.500000',
        '-3.250000',
    ]
    assert rows[0]['total_applied_shift_ms'] == rows[0]['applied_shift_ms']
    assert rows[1]['total_applied_shift_ms'] == rows[1]['applied_shift_ms']


def test_export_golden_invalid_status_rows_included_for_audit_mode() -> None:
    expected = (
        'format_name,format_version,source_job_id,endpoint_kind,endpoint_key,'
        'endpoint_id,node_id,x_m,y_m,surface_elevation_m,t1_ms,v1_m_s,'
        'v2_m_s,sh1_weathering_thickness_m,weathering_correction_ms,'
        'elevation_correction_ms,total_static_ms,total_applied_shift_ms,'
        'static_status,sign_convention\n'
        'lsst,1,refraction-golden-job-522,source,source:1001,1001,10,'
        '1000.000,2000.000,25.000,12.500000,800.000,2400.000,10.607,'
        '-8.838835,21.338835,12.500000,12.500000,ok,'
        'corrected(t) = raw(t - shift_s)\n'
        'lsst,1,refraction-golden-job-522,source,source:inactive,1999,'
        ',,,,,,,,,,,,inactive_endpoint,corrected(t) = raw(t - shift_s)\n'
        'lsst,1,refraction-golden-job-522,receiver,receiver:2001,2001,20,'
        '1010.000,2010.000,30.000,8.500000,800.000,2300.000,7.253,'
        '-5.912671,2.662671,-3.250000,-3.250000,ok,'
        'corrected(t) = raw(t - shift_s)\n'
    )

    assert (
        format_refraction_lsst_csv(
            invalid_status_export_bundle(),
            fail_on_invalid_static_status=False,
            include_inactive_endpoints=True,
        )
        == expected
    )


def _read_csv_with_fieldnames(text: str) -> tuple[list[dict[str, str]], list[str]]:
    reader = csv.DictReader(io.StringIO(text))
    return list(reader), list(reader.fieldnames or ())


def _csv_text(fieldnames: list[str] | tuple[str, ...], rows: list[dict[str, str]] | tuple[dict[str, str], ...]) -> str:
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(fieldnames), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
