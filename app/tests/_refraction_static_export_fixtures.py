from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.refraction_static_artifacts import (
    FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION,
)
from app.services.refraction_static_export_types import (
    RefractionStaticEndpointExportRow,
    RefractionStaticExportBundle,
)
from app.services.refraction_static_export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)
from app.services.refraction_static_t1lsst import (
    compute_t1lsst_1layer_thickness,
    compute_t1lsst_1layer_weathering_correction,
    compute_t1lsst_2layer_thicknesses_with_status,
    compute_t1lsst_3layer_thicknesses_with_status,
)

GOLDEN_SOURCE_JOB_ID = 'refraction-golden-job-522'


@dataclass(frozen=True)
class _T1LSSTTerms:
    sh1_m: float
    weathering_correction_s: float
    sh2_m: float | None = None
    sh3_m: float | None = None

    @property
    def total_weathering_thickness_m(self) -> float:
        return self.sh1_m + (self.sh2_m or 0.0) + (self.sh3_m or 0.0)


def _one_layer_terms(*, t1_s: float, v1_m_s: float, v2_m_s: float) -> _T1LSSTTerms:
    sh1 = _scalar(
        compute_t1lsst_1layer_thickness(
            np.asarray([t1_s], dtype=np.float64),
            v1_m_s,
            v2_m_s,
        )
    )
    wcor = _scalar(
        compute_t1lsst_1layer_weathering_correction(
            np.asarray([sh1], dtype=np.float64),
            v1_m_s,
            v2_m_s,
        )
    )
    return _T1LSSTTerms(sh1_m=sh1, weathering_correction_s=wcor)


def _two_layer_terms(
    *,
    t1_s: float,
    t2_s: float,
    v1_m_s: float,
    v2_m_s: float,
    v3_m_s: float,
) -> _T1LSSTTerms:
    result = compute_t1lsst_2layer_thicknesses_with_status(
        t1_s=np.asarray([t1_s], dtype=np.float64),
        t2_s=np.asarray([t2_s], dtype=np.float64),
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
        strict_velocity_order=True,
    )
    if str(result.status[0]) != 'ok':
        raise AssertionError(f'unexpected two-layer fixture status: {result.status[0]}')
    return _T1LSSTTerms(
        sh1_m=_scalar(result.sh1_m),
        sh2_m=_scalar(result.sh2_m),
        weathering_correction_s=_scalar(result.weathering_correction_s),
    )


def _three_layer_terms(
    *,
    t1_s: float,
    t2_s: float,
    t3_s: float,
    v1_m_s: float,
    v2_m_s: float,
    v3_m_s: float,
    vsub_m_s: float,
) -> _T1LSSTTerms:
    result = compute_t1lsst_3layer_thicknesses_with_status(
        t1_s=np.asarray([t1_s], dtype=np.float64),
        t2_s=np.asarray([t2_s], dtype=np.float64),
        t3_s=np.asarray([t3_s], dtype=np.float64),
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
        vsub_m_s=vsub_m_s,
        strict_velocity_order=True,
    )
    if str(result.status[0]) != 'ok':
        raise AssertionError(f'unexpected three-layer fixture status: {result.status[0]}')
    return _T1LSSTTerms(
        sh1_m=_scalar(result.sh1_m),
        sh2_m=_scalar(result.sh2_m),
        sh3_m=_scalar(result.sh3_m),
        weathering_correction_s=_scalar(result.weathering_correction_s),
    )


def _scalar(array: np.ndarray | None) -> float:
    if array is None:
        raise AssertionError('fixture calculation returned no value')
    return float(np.asarray(array, dtype=np.float64).reshape(-1)[0])


def _m(value: float) -> str:
    return f'{value:.3f}'


def _ms(value_s: float) -> str:
    return f'{value_s * 1000.0:.6f}'


def one_layer_export_bundle() -> RefractionStaticExportBundle:
    source_terms = _one_layer_terms(t1_s=0.0125, v1_m_s=800.0, v2_m_s=2400.0)
    receiver_terms = _one_layer_terms(t1_s=0.0085, v1_m_s=800.0, v2_m_s=2300.0)
    return RefractionStaticExportBundle(
        source_job_id=GOLDEN_SOURCE_JOB_ID,
        source_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='source:1001',
                endpoint_id=1001,
                station_id=1001,
                node_id=10,
                x_m=1000.0,
                y_m=2000.0,
                elevation_m=25.0,
                t1_s=0.0125,
                v1_m_s=800.0,
                v2_m_s=2400.0,
                sh1_m=source_terms.sh1_m,
                weathering_correction_s=source_terms.weathering_correction_s,
                elevation_correction_s=0.0125
                - source_terms.weathering_correction_s,
                total_applied_shift_s=0.0125,
            ),
        ),
        receiver_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='receiver',
                endpoint_key='receiver:2001',
                endpoint_id=2001,
                station_id=2001,
                node_id=20,
                x_m=1010.0,
                y_m=2010.0,
                elevation_m=30.0,
                t1_s=0.0085,
                v1_m_s=800.0,
                v2_m_s=2300.0,
                sh1_m=receiver_terms.sh1_m,
                weathering_correction_s=receiver_terms.weathering_correction_s,
                elevation_correction_s=-0.00325
                - receiver_terms.weathering_correction_s,
                total_applied_shift_s=-0.00325,
            ),
        ),
    )


def two_layer_export_bundle() -> RefractionStaticExportBundle:
    source_terms = _two_layer_terms(
        t1_s=0.013,
        t2_s=0.021,
        v1_m_s=850.0,
        v2_m_s=2450.0,
        v3_m_s=3600.0,
    )
    receiver_terms = _two_layer_terms(
        t1_s=0.010,
        t2_s=0.0185,
        v1_m_s=850.0,
        v2_m_s=2400.0,
        v3_m_s=3550.0,
    )
    return RefractionStaticExportBundle(
        source_job_id=GOLDEN_SOURCE_JOB_ID,
        source_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='source:1101',
                endpoint_id=1101,
                station_id=1101,
                node_id=11,
                x_m=1100.0,
                y_m=2100.0,
                elevation_m=40.0,
                t1_s=0.013,
                t2_s=0.021,
                v1_m_s=850.0,
                v2_m_s=2450.0,
                v3_m_s=3600.0,
                sh1_m=source_terms.sh1_m,
                sh2_m=source_terms.sh2_m,
                total_weathering_thickness_m=source_terms.total_weathering_thickness_m,
                weathering_correction_s=source_terms.weathering_correction_s,
                elevation_correction_s=0.0125
                - source_terms.weathering_correction_s,
                total_applied_shift_s=0.0125,
            ),
        ),
        receiver_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='receiver',
                endpoint_key='receiver:2101',
                endpoint_id=2101,
                station_id=2101,
                node_id=21,
                x_m=1110.0,
                y_m=2110.0,
                elevation_m=42.0,
                t1_s=0.010,
                t2_s=0.0185,
                v1_m_s=850.0,
                v2_m_s=2400.0,
                v3_m_s=3550.0,
                sh1_m=receiver_terms.sh1_m,
                sh2_m=receiver_terms.sh2_m,
                total_weathering_thickness_m=(
                    receiver_terms.total_weathering_thickness_m
                ),
                weathering_correction_s=receiver_terms.weathering_correction_s,
                elevation_correction_s=-0.005
                - receiver_terms.weathering_correction_s,
                total_applied_shift_s=-0.005,
            ),
        ),
    )


def three_layer_export_bundle() -> RefractionStaticExportBundle:
    source_terms = _three_layer_terms(
        t1_s=0.014,
        t2_s=0.022,
        t3_s=0.031,
        v1_m_s=900.0,
        v2_m_s=2500.0,
        v3_m_s=3700.0,
        vsub_m_s=5200.0,
    )
    receiver_terms = _three_layer_terms(
        t1_s=0.011,
        t2_s=0.019,
        t3_s=0.028,
        v1_m_s=900.0,
        v2_m_s=2480.0,
        v3_m_s=3650.0,
        vsub_m_s=5100.0,
    )
    return RefractionStaticExportBundle(
        source_job_id=GOLDEN_SOURCE_JOB_ID,
        source_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='source:1201',
                endpoint_id=1201,
                station_id=1201,
                node_id=12,
                x_m=1200.0,
                y_m=2200.0,
                elevation_m=50.0,
                t1_s=0.014,
                t2_s=0.022,
                t3_s=0.031,
                v1_m_s=900.0,
                v2_m_s=2500.0,
                v3_m_s=3700.0,
                vsub_m_s=5200.0,
                sh1_m=source_terms.sh1_m,
                sh2_m=source_terms.sh2_m,
                sh3_m=source_terms.sh3_m,
                total_weathering_thickness_m=source_terms.total_weathering_thickness_m,
                weathering_correction_s=source_terms.weathering_correction_s,
                elevation_correction_s=0.014
                - source_terms.weathering_correction_s,
                total_applied_shift_s=0.014,
            ),
        ),
        receiver_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='receiver',
                endpoint_key='receiver:2201',
                endpoint_id=2201,
                station_id=2201,
                node_id=22,
                x_m=1210.0,
                y_m=2210.0,
                elevation_m=52.0,
                t1_s=0.011,
                t2_s=0.019,
                t3_s=0.028,
                v1_m_s=900.0,
                v2_m_s=2480.0,
                v3_m_s=3650.0,
                vsub_m_s=5100.0,
                sh1_m=receiver_terms.sh1_m,
                sh2_m=receiver_terms.sh2_m,
                sh3_m=receiver_terms.sh3_m,
                total_weathering_thickness_m=(
                    receiver_terms.total_weathering_thickness_m
                ),
                weathering_correction_s=receiver_terms.weathering_correction_s,
                elevation_correction_s=-0.006
                - receiver_terms.weathering_correction_s,
                total_applied_shift_s=-0.006,
            ),
        ),
    )


def field_component_export_bundle() -> RefractionStaticExportBundle:
    source_terms = _one_layer_terms(t1_s=0.0125, v1_m_s=800.0, v2_m_s=2400.0)
    receiver_terms = _one_layer_terms(t1_s=0.0085, v1_m_s=800.0, v2_m_s=2300.0)
    return RefractionStaticExportBundle(
        source_job_id=GOLDEN_SOURCE_JOB_ID,
        source_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='source:1001',
                endpoint_id=1001,
                station_id=1001,
                node_id=10,
                x_m=1000.0,
                y_m=2000.0,
                elevation_m=25.0,
                t1_s=0.0125,
                v1_m_s=800.0,
                v2_m_s=2400.0,
                sh1_m=source_terms.sh1_m,
                weathering_correction_s=source_terms.weathering_correction_s,
                elevation_correction_s=0.0125
                - source_terms.weathering_correction_s,
                field_correction_s=0.00625,
                source_depth_m=6.4,
                source_depth_shift_s=0.008,
                source_depth_status='ok',
                uphole_time_s=0.003,
                uphole_shift_s=-0.003,
                uphole_status='ok',
                manual_static_shift_s=0.00125,
                manual_static_status='ok',
                field_static_status='ok',
                total_with_field_shift_s=0.01875,
                total_applied_shift_s=0.0125,
            ),
        ),
        receiver_rows=(
            RefractionStaticEndpointExportRow(
                endpoint_kind='receiver',
                endpoint_key='receiver:2001',
                endpoint_id=2001,
                station_id=2001,
                node_id=20,
                x_m=1010.0,
                y_m=2010.0,
                elevation_m=30.0,
                t1_s=0.0085,
                v1_m_s=800.0,
                v2_m_s=2300.0,
                sh1_m=receiver_terms.sh1_m,
                weathering_correction_s=receiver_terms.weathering_correction_s,
                elevation_correction_s=-0.00325
                - receiver_terms.weathering_correction_s,
                field_correction_s=-0.002,
                manual_static_shift_s=-0.002,
                manual_static_status='ok',
                field_static_status='ok',
                total_with_field_shift_s=-0.00525,
                total_applied_shift_s=-0.00325,
            ),
        ),
    )


def invalid_status_export_bundle() -> RefractionStaticExportBundle:
    return RefractionStaticExportBundle(
        source_job_id=GOLDEN_SOURCE_JOB_ID,
        source_rows=(
            one_layer_export_bundle().source_rows[0],
            RefractionStaticEndpointExportRow(
                endpoint_kind='source',
                endpoint_key='source:inactive',
                endpoint_id=1999,
                static_status='inactive_endpoint',
            ),
        ),
        receiver_rows=one_layer_export_bundle().receiver_rows,
    )


def field_component_static_table_rows() -> tuple[
    tuple[dict[str, str], ...],
    tuple[dict[str, str], ...],
]:
    source_terms = _one_layer_terms(t1_s=0.0125, v1_m_s=800.0, v2_m_s=2400.0)
    receiver_terms = _one_layer_terms(t1_s=0.0085, v1_m_s=800.0, v2_m_s=2300.0)
    source_elevation_correction_s = 0.0125 - source_terms.weathering_correction_s
    receiver_elevation_correction_s = (
        -0.00325 - receiver_terms.weathering_correction_s
    )
    source_rows = (
        {
            'endpoint_kind': 'source',
            'source_endpoint_key': 'source:1001',
            'source_id': '1001',
            'source_node_id': '10',
            'x_m': '1000.000',
            'y_m': '2000.000',
            'surface_elevation_m': '25.000',
            't1_ms': '12.500000',
            'v1_m_s': '800.000',
            'v2_m_s': '2400.000',
            'sh1_weathering_thickness_m': _m(source_terms.sh1_m),
            'refractor_elevation_m': _m(25.0 - source_terms.sh1_m),
            'weathering_correction_ms': _ms(source_terms.weathering_correction_s),
            'elevation_correction_ms': _ms(source_elevation_correction_s),
            'total_static_ms': '12.500000',
            'total_applied_shift_ms': '12.500000',
            'solution_status': 'solved',
            'weathering_status': 'ok',
            'datum_status': 'ok',
            'static_status': 'ok',
            'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
            'source_depth_m': '6.400',
            'source_depth_shift_ms': '8.000000',
            'source_depth_status': 'ok',
            'uphole_time_ms': '3.000000',
            'uphole_shift_ms': '-3.000000',
            'uphole_status': 'ok',
            'manual_static_shift_ms': '1.250000',
            'manual_static_status': 'ok',
            'source_field_shift_ms': '6.250000',
            'source_field_static_status': 'ok',
            'source_total_with_field_shift_ms': '18.750000',
            'pick_count': '5',
            'used_pick_count': '4',
            'pick_count_by_layer': 'v2_t1:5',
            'used_pick_count_by_layer': 'v2_t1:4',
            'residual_rms_ms': '1.250000',
            'residual_mad_ms': '0.750000',
            'residual_rms_by_layer_ms': 'v2_t1:1.25',
            'residual_mad_by_layer_ms': 'v2_t1:0.75',
            'comment': 'positive source shift',
        },
    )
    receiver_rows = (
        {
            'endpoint_kind': 'receiver',
            'receiver_endpoint_key': 'receiver:2001',
            'receiver_id': '2001',
            'receiver_node_id': '20',
            'x_m': '1010.000',
            'y_m': '2010.000',
            'surface_elevation_m': '30.000',
            't1_ms': '8.500000',
            'v1_m_s': '800.000',
            'v2_m_s': '2300.000',
            'sh1_weathering_thickness_m': _m(receiver_terms.sh1_m),
            'refractor_elevation_m': _m(30.0 - receiver_terms.sh1_m),
            'weathering_correction_ms': _ms(receiver_terms.weathering_correction_s),
            'elevation_correction_ms': _ms(receiver_elevation_correction_s),
            'total_static_ms': '-3.250000',
            'total_applied_shift_ms': '-3.250000',
            'solution_status': 'solved',
            'weathering_status': 'ok',
            'datum_status': 'ok',
            'static_status': 'ok',
            'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
            'manual_static_shift_ms': '-2.000000',
            'manual_static_status': 'ok',
            'receiver_field_shift_ms': '-2.000000',
            'receiver_field_static_status': 'ok',
            'receiver_total_with_field_shift_ms': '-5.250000',
            'pick_count': '6',
            'used_pick_count': '6',
            'pick_count_by_layer': 'v2_t1:6',
            'used_pick_count_by_layer': 'v2_t1:6',
            'residual_rms_ms': '1.000000',
            'residual_mad_ms': '0.500000',
            'residual_rms_by_layer_ms': 'v2_t1:1.0',
            'residual_mad_by_layer_ms': 'v2_t1:0.5',
            'comment': 'negative receiver shift',
        },
    )
    return source_rows, receiver_rows


def first_break_time_rows() -> tuple[dict[str, str], ...]:
    return (
        {
            'format_name': 'first_break_time',
            'format_version': '1',
            'source_job_id': GOLDEN_SOURCE_JOB_ID,
            'observation_index': '0',
            'sorted_trace_index': '0',
            'source_endpoint_key': 's0',
            'receiver_endpoint_key': 'r0',
            'source_id': '100',
            'receiver_id': '200',
            'offset_m': '100.0',
            'layer_kind': 'v2_t1',
            'observed_first_break_time_ms': '50.0',
            'modeled_first_break_time_ms': '49.0',
            'residual_ms': '1.0',
            'used_in_solve': 'true',
            'reject_reason': 'ok',
            'sign_convention': FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION,
        },
        {
            'format_name': 'first_break_time',
            'format_version': '1',
            'source_job_id': GOLDEN_SOURCE_JOB_ID,
            'observation_index': '1',
            'sorted_trace_index': '1',
            'source_endpoint_key': 's1',
            'receiver_endpoint_key': 'r1',
            'source_id': '101',
            'receiver_id': '201',
            'offset_m': '200.0',
            'layer_kind': 'v2_t1',
            'observed_first_break_time_ms': '60.0',
            'modeled_first_break_time_ms': '62.0',
            'residual_ms': '-2.0',
            'used_in_solve': 'false',
            'reject_reason': 'robust_outlier',
            'sign_convention': FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION,
        },
        {
            'format_name': 'first_break_time',
            'format_version': '1',
            'source_job_id': GOLDEN_SOURCE_JOB_ID,
            'observation_index': '2',
            'sorted_trace_index': '2',
            'source_endpoint_key': 's0',
            'receiver_endpoint_key': 'r1',
            'source_id': '100',
            'receiver_id': '201',
            'offset_m': '300.0',
            'layer_kind': 'v2_t1',
            'observed_first_break_time_ms': '70.0',
            'modeled_first_break_time_ms': '71.0',
            'residual_ms': '-1.0',
            'used_in_solve': 'true',
            'reject_reason': 'ok',
            'sign_convention': FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION,
        },
    )


def canonical_static_table_rows() -> tuple[dict[str, str], ...]:
    source_terms = _one_layer_terms(t1_s=0.0125, v1_m_s=800.0, v2_m_s=2400.0)
    receiver_terms = _one_layer_terms(t1_s=0.0085, v1_m_s=800.0, v2_m_s=2300.0)
    source_elevation_correction_s = 0.0125 - source_terms.weathering_correction_s
    receiver_elevation_correction_s = (
        -0.00325 - receiver_terms.weathering_correction_s
    )
    return (
        {
            'format_name': 'canonical_static_table',
            'format_version': '1',
            'source_job_id': GOLDEN_SOURCE_JOB_ID,
            'endpoint_kind': 'source',
            'endpoint_key': 'source:1001',
            'endpoint_id': '1001',
            'applied_shift_ms': '12.500000',
            'static_status': 'ok',
            'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
            'x_m': '1000.000',
            'y_m': '2000.000',
            'source_id': '1001',
            'receiver_id': '',
            'node_id': '10',
            'total_static_ms': '12.500000',
            'total_applied_shift_ms': '12.500000',
            'source_field_shift_ms': '6.250000',
            'receiver_field_shift_ms': '',
            'source_total_with_field_shift_ms': '18.750000',
            'receiver_total_with_field_shift_ms': '',
            'manual_static_shift_ms': '1.250000',
            'source_depth_shift_ms': '8.000000',
            'uphole_shift_ms': '-3.000000',
            't1_ms': '12.500000',
            't2_ms': '',
            't3_ms': '',
            'v1_m_s': '800.000',
            'v2_m_s': '2400.000',
            'v3_m_s': '',
            'vsub_m_s': '',
            'sh1_weathering_thickness_m': _m(source_terms.sh1_m),
            'sh2_weathering_thickness_m': '',
            'sh3_weathering_thickness_m': '',
            'weathering_correction_ms': _ms(source_terms.weathering_correction_s),
            'elevation_correction_ms': _ms(source_elevation_correction_s),
            'comment': 'positive source shift',
        },
        {
            'format_name': 'canonical_static_table',
            'format_version': '1',
            'source_job_id': GOLDEN_SOURCE_JOB_ID,
            'endpoint_kind': 'receiver',
            'endpoint_key': 'receiver:2001',
            'endpoint_id': '2001',
            'applied_shift_ms': '-3.250000',
            'static_status': 'ok',
            'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
            'x_m': '1010.000',
            'y_m': '2010.000',
            'source_id': '',
            'receiver_id': '2001',
            'node_id': '20',
            'total_static_ms': '-3.250000',
            'total_applied_shift_ms': '-3.250000',
            'source_field_shift_ms': '',
            'receiver_field_shift_ms': '-2.000000',
            'source_total_with_field_shift_ms': '',
            'receiver_total_with_field_shift_ms': '-5.250000',
            'manual_static_shift_ms': '-2.000000',
            'source_depth_shift_ms': '',
            'uphole_shift_ms': '',
            't1_ms': '8.500000',
            't2_ms': '',
            't3_ms': '',
            'v1_m_s': '800.000',
            'v2_m_s': '2300.000',
            'v3_m_s': '',
            'vsub_m_s': '',
            'sh1_weathering_thickness_m': _m(receiver_terms.sh1_m),
            'sh2_weathering_thickness_m': '',
            'sh3_weathering_thickness_m': '',
            'weathering_correction_ms': _ms(receiver_terms.weathering_correction_s),
            'elevation_correction_ms': _ms(receiver_elevation_correction_s),
            'comment': 'negative receiver shift',
        },
    )
