from __future__ import annotations

import csv
from pathlib import Path

from app.services.refraction_static_export_types import (
    RefractionStaticEndpointExportRow,
    RefractionStaticExportBundle,
)
from app.services.refraction_static_export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)

SOURCE_JOB_ID = 'refraction-golden-job'


def one_layer_lsst_bundle() -> RefractionStaticExportBundle:
    return RefractionStaticExportBundle(
        source_job_id=SOURCE_JOB_ID,
        source_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='source:1001',
                endpoint_id=1001,
                node_id=10,
                x_m=1000.0,
                y_m=2000.0,
                elevation_m=25.0,
                t1_s=0.0125,
                v1_m_s=800.0,
                v2_m_s=2400.0,
                sh1_m=8.0,
                weathering_correction_s=-0.0105,
                elevation_correction_s=-0.002,
                total_applied_shift_s=-0.0125,
            ),
        ),
        receiver_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='receiver',
                endpoint_key='receiver:2001',
                endpoint_id=2001,
                node_id=20,
                x_m=1010.0,
                y_m=2010.0,
                elevation_m=30.0,
                t1_s=0.008,
                v1_m_s=800.0,
                v2_m_s=2300.0,
                sh1_m=7.0,
                weathering_correction_s=-0.00625,
                elevation_correction_s=0.011,
                total_applied_shift_s=0.00475,
            ),
        ),
    )


def three_layer_lsst_plus_bundle() -> RefractionStaticExportBundle:
    return RefractionStaticExportBundle(
        source_job_id=SOURCE_JOB_ID,
        source_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='source:3001',
                endpoint_id=3001,
                node_id=30,
                x_m=1500.0,
                y_m=2500.0,
                elevation_m=40.0,
                t1_s=0.011,
                t2_s=0.021,
                t3_s=0.033,
                v1_m_s=850.0,
                v2_m_s=2450.0,
                v3_m_s=3650.0,
                vsub_m_s=5200.0,
                sh1_m=8.4,
                sh2_m=12.6,
                sh3_m=18.2,
                total_weathering_thickness_m=39.2,
                weathering_correction_s=-0.014,
                elevation_correction_s=0.006,
                field_correction_s=0.009,
                source_depth_m=11.05,
                source_depth_shift_s=0.013,
                source_depth_status='ok',
                uphole_time_s=0.004,
                uphole_shift_s=-0.004,
                uphole_status='ok',
                manual_static_shift_s=0.0005,
                manual_static_status='ok',
                field_static_status='ok',
                total_with_field_shift_s=0.001,
                total_applied_shift_s=-0.008,
            ),
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='source:invalid',
                endpoint_id=3999,
                x_m=float('nan'),
                t1_s=float('nan'),
                source_depth_status='invalid_source_depth',
                manual_static_status='missing_manual_static',
                field_static_status='invalid_component',
                total_applied_shift_s=float('inf'),
                static_status='invalid_velocity_order',
            ),
        ),
        receiver_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='receiver',
                endpoint_key='receiver:4001',
                endpoint_id=4001,
                node_id=40,
                x_m=1512.0,
                y_m=2512.0,
                elevation_m=41.0,
                t1_s=0.0095,
                t2_s=0.019,
                t3_s=0.0305,
                v1_m_s=850.0,
                v2_m_s=2400.0,
                v3_m_s=3600.0,
                vsub_m_s=5150.0,
                sh1_m=7.9,
                sh2_m=11.8,
                sh3_m=17.6,
                total_weathering_thickness_m=37.3,
                weathering_correction_s=-0.012,
                elevation_correction_s=0.016,
                field_correction_s=-0.0015,
                manual_static_shift_s=-0.0015,
                manual_static_status='ok',
                field_static_status='ok',
                total_with_field_shift_s=0.0025,
                total_applied_shift_s=0.004,
            ),
        ),
    )


def two_layer_source_static_rows() -> tuple[dict[str, str], ...]:
    return (
        {
            'endpoint_kind': 'source',
            'source_endpoint_key': 'source:1001',
            'source_id': '1001',
            'source_node_id': '10',
            'x_m': '1000.0',
            'y_m': '2000.0',
            'surface_elevation_m': '25.0',
            't1_ms': '12.5',
            't2_ms': '20.25',
            'v1_m_s': '800.0',
            'v2_m_s': '2400.0',
            'v3_m_s': '3600.0',
            'sh1_weathering_thickness_m': '8.0',
            'sh2_weathering_thickness_m': '12.5',
            'total_weathering_thickness_m': '20.5',
            'layer1_base_elevation_m': '17.0',
            'final_refractor_elevation_m': '4.5',
            'weathering_correction_ms': '-10.5',
            'elevation_correction_ms': '-2.0',
            'total_static_ms': '-3.5',
            'total_applied_shift_ms': '-3.5',
            'source_depth_shift_ms': '12.5',
            'source_depth_status': 'ok',
            'uphole_shift_ms': '-4.0',
            'uphole_status': 'ok',
            'manual_static_shift_ms': '0.5',
            'manual_static_status': 'ok',
            'source_field_shift_ms': '9.0',
            'source_field_static_status': 'ok',
            'source_total_with_field_shift_ms': '5.5',
            'pick_count': '6',
            'used_pick_count': '5',
            'pick_count_by_layer': '{"v2_t1":3,"v3_t2":3}',
            'used_pick_count_by_layer': '{"v2_t1":3,"v3_t2":2}',
            'residual_rms_ms': '1.25',
            'residual_mad_ms': '0.75',
            'residual_rms_by_layer_ms': '{"v2_t1":1.0,"v3_t2":1.5}',
            'residual_mad_by_layer_ms': '{"v2_t1":0.5,"v3_t2":1.0}',
            'solution_status': 'solved',
            'weathering_status': 'ok',
            'datum_status': 'ok',
            'static_status': 'ok',
            'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
        },
    )


def two_layer_receiver_static_rows() -> tuple[dict[str, str], ...]:
    return (
        {
            'endpoint_kind': 'receiver',
            'receiver_endpoint_key': 'receiver:2001',
            'receiver_id': '2001',
            'receiver_node_id': '20',
            'x_m': '1010.0',
            'y_m': '2010.0',
            'surface_elevation_m': '30.0',
            't1_ms': '8.0',
            't2_ms': '18.75',
            'v1_m_s': '800.0',
            'v2_m_s': '2300.0',
            'v3_m_s': '3550.0',
            'sh1_weathering_thickness_m': '7.0',
            'sh2_weathering_thickness_m': '11.0',
            'total_weathering_thickness_m': '18.0',
            'layer1_base_elevation_m': '23.0',
            'final_refractor_elevation_m': '12.0',
            'weathering_correction_ms': '-6.25',
            'elevation_correction_ms': '11.0',
            'total_static_ms': '4.75',
            'total_applied_shift_ms': '4.75',
            'manual_static_shift_ms': '-0.75',
            'manual_static_status': 'ok',
            'receiver_field_shift_ms': '-0.75',
            'receiver_field_static_status': 'ok',
            'receiver_total_with_field_shift_ms': '4.0',
            'pick_count': '4',
            'used_pick_count': '4',
            'pick_count_by_layer': '{"v2_t1":2,"v3_t2":2}',
            'used_pick_count_by_layer': '{"v2_t1":2,"v3_t2":2}',
            'residual_rms_ms': '0.9',
            'residual_mad_ms': '0.4',
            'residual_rms_by_layer_ms': '{"v2_t1":0.7,"v3_t2":1.1}',
            'residual_mad_by_layer_ms': '{"v2_t1":0.3,"v3_t2":0.6}',
            'solution_status': 'solved',
            'weathering_status': 'ok',
            'datum_status': 'ok',
            'static_status': 'ok',
            'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
        },
    )


def write_static_table_csv(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_break_export_result():
    from app.tests._refraction_static_artifact_helpers import _result

    return _result()
