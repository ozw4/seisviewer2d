from __future__ import annotations

from app.api.schemas import RefractionStaticExportRequest
from app.services.refraction_static_artifacts import (
    CANONICAL_SOURCE_RECEIVER_STATIC_TABLE_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    write_refraction_first_break_time_export_csv,
    write_refraction_time_term_spreadsheet_csv_from_static_tables,
)
from app.services.refraction_static_export_service import (
    write_refraction_static_requested_export_artifacts,
)
from app.services.refraction_static_lsst_export import (
    format_refraction_lsst_csv,
    format_refraction_lsst_plus_csv,
)
from app.tests._refraction_static_export_fixtures import (
    SOURCE_JOB_ID,
    first_break_export_result,
    one_layer_lsst_bundle,
    three_layer_lsst_plus_bundle,
    two_layer_receiver_static_rows,
    two_layer_source_static_rows,
    write_static_table_csv,
)


def test_lsst_golden_one_layer() -> None:
    expected = (
        'format_name,format_version,source_job_id,endpoint_kind,endpoint_key,'
        'endpoint_id,node_id,x_m,y_m,surface_elevation_m,t1_ms,v1_m_s,'
        'v2_m_s,sh1_weathering_thickness_m,weathering_correction_ms,'
        'elevation_correction_ms,total_static_ms,total_applied_shift_ms,'
        'static_status,sign_convention\n'
        'lsst,1,refraction-golden-job,source,source:1001,1001,10,1000.000,'
        '2000.000,25.000,12.500000,800.000,2400.000,8.000,'
        '-10.500000,-2.000000,-12.500000,-12.500000,ok,'
        'corrected(t) = raw(t - shift_s)\n'
        'lsst,1,refraction-golden-job,receiver,receiver:2001,2001,20,'
        '1010.000,2010.000,30.000,8.000000,800.000,2300.000,7.000,'
        '-6.250000,11.000000,4.750000,4.750000,ok,'
        'corrected(t) = raw(t - shift_s)\n'
    )
    bundle = one_layer_lsst_bundle()

    assert format_refraction_lsst_csv(bundle) == expected
    assert format_refraction_lsst_csv(bundle) == format_refraction_lsst_csv(bundle)


def test_lsst_plus_golden_three_layer() -> None:
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
        'lsst_plus,1,refraction-golden-job,source,source:3001,3001,30,'
        '1500.000,2500.000,40.000,11.000000,850.000,2450.000,8.400,'
        '-14.000000,6.000000,-8.000000,-8.000000,ok,'
        'corrected(t) = raw(t - shift_s),21.000000,33.000000,3650.000,'
        '5200.000,12.600,18.200,39.200,11.050,13.000000,ok,'
        '4.000000,-4.000000,ok,0.500000,ok,9.000000,ok,1.000000,,,\n'
        'lsst_plus,1,refraction-golden-job,source,source:invalid,3999,,,,,,,'
        ',,,,,,invalid_velocity_order,corrected(t) = raw(t - shift_s),'
        ',,,,,,,,,invalid_source_depth,,,,,missing_manual_static,,'
        'invalid_component,,,,\n'
        'lsst_plus,1,refraction-golden-job,receiver,receiver:4001,4001,40,'
        '1512.000,2512.000,41.000,9.500000,850.000,2400.000,7.900,'
        '-12.000000,16.000000,4.000000,4.000000,ok,'
        'corrected(t) = raw(t - shift_s),19.000000,30.500000,3600.000,'
        '5150.000,11.800,17.600,37.300,,,,,,,-1.500000,ok,,,'
        ',-1.500000,ok,2.500000\n'
    )

    assert format_refraction_lsst_plus_csv(three_layer_lsst_plus_bundle()) == expected


def test_time_term_spreadsheet_golden_with_field_components(tmp_path) -> None:
    path = tmp_path / REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME
    expected = (
        'schema_version,format_name,format_version,source_job_id,endpoint_kind,'
        'endpoint_key,endpoint_id,station_id,node_id,x_m,y_m,elevation_m,'
        'surface_elevation_m,t1_ms,t2_ms,t3_ms,v1_m_s,v2_m_s,v3_m_s,'
        'vsub_m_s,sh1_m,sh2_m,sh3_m,layer1_base_elevation_m,'
        'layer2_base_elevation_m,final_refractor_elevation_m,'
        'weathering_correction_ms,elevation_correction_ms,'
        'source_depth_correction_ms,uphole_correction_ms,manual_static_ms,'
        'field_correction_ms,total_applied_shift_ms,pick_count,used_pick_count,'
        'pick_count_by_layer,used_pick_count_by_layer,residual_rms_ms,'
        'residual_mad_ms,residual_rms_by_layer_ms,residual_mad_by_layer_ms,'
        'solution_status,weathering_status,datum_status,source_depth_status,'
        'uphole_status,manual_static_status,field_static_status,static_status,'
        'sign_convention\n'
        '1,time_term_spreadsheet,1,refraction-golden-job,source,source:1001,'
        '1001,1001,10,1000.000,2000.000,25.000,25.000,12.500000,'
        '20.250000,,800.000,2400.000,3600.000,,8.000,12.500,,17.000,'
        ',4.500,-10.500000,-2.000000,12.500000,-4.000000,0.500000,'
        '9.000000,-3.500000,6,5,"{""v2_t1"":3,""v3_t2"":3}",'
        '"{""v2_t1"":3,""v3_t2"":2}",1.250000,0.750000,'
        '"{""v2_t1"":1.0,""v3_t2"":1.5}",'
        '"{""v2_t1"":0.5,""v3_t2"":1.0}",solved,ok,ok,ok,ok,ok,ok,'
        'ok,corrected(t) = raw(t - shift_s)\n'
        '1,time_term_spreadsheet,1,refraction-golden-job,receiver,'
        'receiver:2001,2001,2001,20,1010.000,2010.000,30.000,30.000,'
        '8.000000,18.750000,,800.000,2300.000,3550.000,,7.000,11.000,'
        ',23.000,,12.000,-6.250000,11.000000,,,-0.750000,-0.750000,'
        '4.750000,4,4,"{""v2_t1"":2,""v3_t2"":2}",'
        '"{""v2_t1"":2,""v3_t2"":2}",0.900000,0.400000,'
        '"{""v2_t1"":0.7,""v3_t2"":1.1}",'
        '"{""v2_t1"":0.3,""v3_t2"":0.6}",solved,ok,ok,not_applicable,'
        'not_applicable,ok,ok,ok,corrected(t) = raw(t - shift_s)\n'
    )

    write_refraction_time_term_spreadsheet_csv_from_static_tables(
        source_rows=two_layer_source_static_rows(),
        receiver_rows=two_layer_receiver_static_rows(),
        path=path,
        source_job_id=SOURCE_JOB_ID,
        include_inactive_endpoints=True,
    )

    assert path.read_text(encoding='utf-8') == expected


def test_first_break_time_export_golden_with_residuals(tmp_path) -> None:
    path = tmp_path / REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME
    expected = (
        'schema_version,trace_index_sorted,source_endpoint_key,receiver_endpoint_key,'
        'source_node_id,receiver_node_id,offset_m,midpoint_x_m,midpoint_y_m,'
        'cell_ix,cell_iy,layer_kind,used_for_layer,observed_pick_time_ms,'
        'modeled_pick_time_ms,residual_ms,moveout_time_ms,source_time_term_ms,'
        'receiver_time_term_ms,velocity_m_s,rejection_reason,observation_status,'
        'sign_convention\n'
        '1,0,s0,r0,0,1,100.0,27.5,2.5,,,v2_t1,true,50.0,49.0,1.0,'
        '27.0,10.0,12.0,2500.0,ok,used,'
        'residual_ms = observed_pick_time_ms - modeled_pick_time_ms\n'
        '1,1,s1,r1,1,2,200.0,77.5,7.5,,,v2_t1,false,60.0,62.0,-2.0,'
        '36.00000000000001,12.0,14.0,2500.0,robust_outlier,rejected,'
        'residual_ms = observed_pick_time_ms - modeled_pick_time_ms\n'
        '1,2,s0,r1,0,2,300.0,52.5,5.0,,,v2_t1,true,70.0,71.0,-1.0,'
        '46.99999999999999,10.0,14.0,2500.0,ok,used,'
        'residual_ms = observed_pick_time_ms - modeled_pick_time_ms\n'
    )

    write_refraction_first_break_time_export_csv(
        result=first_break_export_result(),
        path=path,
        source_job_id=SOURCE_JOB_ID,
    )

    assert path.read_text(encoding='utf-8') == expected


def test_canonical_static_table_golden_schema_version(tmp_path) -> None:
    source_dir = tmp_path / 'source-artifacts'
    job_dir = tmp_path / 'export-job'
    write_static_table_csv(
        source_dir / SOURCE_STATIC_TABLE_CSV_NAME,
        two_layer_source_static_rows(),
    )
    write_static_table_csv(
        source_dir / RECEIVER_STATIC_TABLE_CSV_NAME,
        two_layer_receiver_static_rows(),
    )
    expected = (
        'format_name,format_version,source_job_id,endpoint_kind,endpoint_key,'
        'endpoint_id,applied_shift_ms,static_status,sign_convention,x_m,y_m,'
        'source_id,receiver_id,node_id,total_static_ms,total_applied_shift_ms,'
        'source_field_shift_ms,receiver_field_shift_ms,'
        'source_total_with_field_shift_ms,receiver_total_with_field_shift_ms,'
        'manual_static_shift_ms,source_depth_shift_ms,uphole_shift_ms,t1_ms,'
        't2_ms,t3_ms,v1_m_s,v2_m_s,v3_m_s,vsub_m_s,'
        'sh1_weathering_thickness_m,sh2_weathering_thickness_m,'
        'sh3_weathering_thickness_m,weathering_correction_ms,'
        'elevation_correction_ms,comment\n'
        'canonical_static_table,1,refraction-golden-job,source,source:1001,'
        '1001,-3.5,ok,corrected(t) = raw(t - shift_s),1000.0,2000.0,'
        '1001,,10,-3.5,-3.5,9.0,,5.5,,0.5,12.5,-4.0,12.5,20.25,'
        ',800.0,2400.0,3600.0,,8.0,12.5,,-10.5,-2.0,\n'
        'canonical_static_table,1,refraction-golden-job,receiver,receiver:2001,'
        '2001,4.75,ok,corrected(t) = raw(t - shift_s),1010.0,2010.0,,'
        '2001,20,4.75,4.75,,-0.75,,4.0,-0.75,,,8.0,18.75,,'
        '800.0,2300.0,3550.0,,7.0,11.0,,-6.25,11.0,\n'
    )

    generated = write_refraction_static_requested_export_artifacts(
        job_dir=job_dir,
        source_artifacts_dir=source_dir,
        source_job_id=SOURCE_JOB_ID,
        source_file_id='raw-file-id',
        key1_byte=189,
        key2_byte=193,
        requested_formats=('canonical_static_table',),
        export=RefractionStaticExportRequest(
            enabled=True,
            formats=['canonical_static_table'],
        ),
    )

    assert generated == (
        'canonical_source_static_table.csv',
        'canonical_receiver_static_table.csv',
        'canonical_source_receiver_static_table.csv',
    )
    assert (
        job_dir / CANONICAL_SOURCE_RECEIVER_STATIC_TABLE_CSV_NAME
    ).read_text(encoding='utf-8') == expected
