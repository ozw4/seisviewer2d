from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
    ResolvedRefractionFirstLayer,
)


def _request() -> RefractionStaticApplyRequest:
    return RefractionStaticApplyRequest.model_validate(
        {
            'file_id': 'raw-file-id',
            'key1_byte': 189,
            'key2_byte': 193,
            'pick_source': {
                'kind': 'batch_predicted_npz',
                'job_id': 'pick-job',
                'artifact_name': 'predicted_picks_time_s.npz',
            },
            'linkage': {'mode': 'none'},
            'model': {
                'method': 'gli_variable_thickness',
                'weathering_velocity_m_s': 800.0,
                'bedrock_velocity_mode': 'solve_global',
            },
            'datum': {
                'mode': 'floating_and_flat',
                'floating_datum_mode': 'constant',
                'floating_datum_elevation_m': 120.0,
                'flat_datum_elevation_m': 300.0,
            },
            'apply': {
                'mode': 'refraction_from_raw',
                'interpolation': 'linear',
                'fill_value': 0.0,
                'max_abs_shift_ms': 250.0,
                'output_dtype': 'float32',
                'register_corrected_file': False,
            },
        }
    )


def _estimated_v1_request() -> RefractionStaticApplyRequest:
    payload = _request().model_dump(mode='json')
    payload['model']['weathering_velocity_m_s'] = None
    payload['model']['first_layer'] = {
        'mode': 'estimate_direct_arrival',
        'min_direct_offset_m': 20.0,
        'max_direct_offset_m': 140.0,
    }
    return RefractionStaticApplyRequest.model_validate(payload)


def _resolved_estimated_v1() -> ResolvedRefractionFirstLayer:
    return ResolvedRefractionFirstLayer(
        mode='estimate_direct_arrival',
        weathering_velocity_m_s=812.5,
        status='estimated',
        qc={
            'v1_mode': 'estimate_direct_arrival',
            'weathering_velocity_m_s': 812.5,
            'resolved_weathering_velocity_m_s': 812.5,
            'v1_status': 'estimated',
        },
    )


def _result() -> RefractionDatumStaticsResult:
    n_traces = 4
    sorted_trace_index = np.arange(n_traces, dtype=np.int64)
    valid_observation = np.asarray([True, True, True, False], dtype=bool)
    used_observation = np.asarray([True, False, True, False], dtype=bool)
    trace_valid = np.asarray([True, True, False, True], dtype=bool)
    trace_status = np.asarray(
        ['ok', 'ok', 'not_observed', 'ok'],
        dtype='<U32',
    )

    node_id = np.asarray([0, 1, 2], dtype=np.int64)
    node_surface = np.asarray([100.0, 110.0, 120.0], dtype=np.float64)
    node_thickness = np.asarray([10.0, 12.0, 14.0], dtype=np.float64)
    node_half = np.asarray([0.010, 0.012, 0.014], dtype=np.float64)
    node_weathering_shift = np.asarray([-0.0085, -0.0102, -0.0119])

    source_node_sorted = np.asarray([0, 1, 0, 1], dtype=np.int64)
    receiver_node_sorted = np.asarray([1, 2, 2, 1], dtype=np.int64)
    source_half_sorted = np.asarray([0.010, 0.012, 0.010, 0.012])
    receiver_half_sorted = np.asarray([0.012, 0.014, 0.014, 0.012])
    source_thickness_sorted = np.asarray([10.0, 12.0, 10.0, 12.0])
    receiver_thickness_sorted = np.asarray([12.0, 14.0, 14.0, 12.0])
    source_weathering_sorted = np.asarray([-0.0085, -0.0102, -0.0085, -0.0102])
    receiver_weathering_sorted = np.asarray([-0.0102, -0.0119, -0.0119, -0.0102])
    weathering_trace = source_weathering_sorted + receiver_weathering_sorted
    source_floating_sorted = np.asarray([0.0010, 0.0015, 0.0010, 0.0015])
    receiver_floating_sorted = np.asarray([0.0015, 0.0020, 0.0020, 0.0015])
    source_flat_sorted = np.asarray([0.010, 0.011, 0.010, 0.011])
    receiver_flat_sorted = np.asarray([0.011, 0.012, 0.012, 0.011])
    source_refraction_sorted = (
        source_weathering_sorted + source_floating_sorted + source_flat_sorted
    )
    receiver_refraction_sorted = (
        receiver_weathering_sorted + receiver_floating_sorted + receiver_flat_sorted
    )
    refraction_trace = source_refraction_sorted + receiver_refraction_sorted
    refraction_trace[2] = np.nan

    return RefractionDatumStaticsResult(
        bedrock_velocity_mode='solve_global',
        bedrock_slowness_s_per_m=1.0 / 2500.0,
        bedrock_velocity_m_s=2500.0,
        weathering_velocity_m_s=800.0,
        replacement_slowness_delta_s_per_m=(1.0 / 2500.0) - (1.0 / 800.0),
        datum_mode='floating_and_flat',
        floating_datum_mode='constant',
        flat_datum_elevation_m=300.0,
        node_id=node_id,
        node_x_m=np.asarray([0.0, 50.0, 100.0]),
        node_y_m=np.asarray([0.0, 5.0, 10.0]),
        node_surface_elevation_m=node_surface,
        node_kind=np.asarray(['source', 'both', 'receiver'], dtype='<U16'),
        node_weathering_thickness_m=node_thickness,
        node_refractor_elevation_m=node_surface - node_thickness,
        node_half_intercept_time_s=node_half,
        node_weathering_replacement_shift_s=node_weathering_shift,
        node_floating_datum_elevation_m=np.asarray([120.0, 120.0, 120.0]),
        node_solution_status=np.asarray(['solved', 'solved', 'inactive'], dtype='<U16'),
        node_datum_status=np.asarray(['ok', 'ok', 'inactive'], dtype='<U16'),
        node_weathering_status=np.asarray(
            ['ok', 'zero_thickness', 'inactive'],
            dtype='<U16',
        ),
        node_pick_count=np.asarray([4, 3, 2], dtype=np.int64),
        node_used_pick_count=np.asarray([4, 2, 1], dtype=np.int64),
        node_rejected_pick_count=np.asarray([0, 1, 1], dtype=np.int64),
        node_residual_rms_s=np.asarray([0.001, 0.002, 0.003]),
        node_residual_mad_s=np.asarray([0.0005, 0.0010, 0.0015]),
        source_endpoint_key=np.asarray(['s0', 's1'], dtype='<U2'),
        source_id=np.asarray([100, 101], dtype=np.int64),
        source_node_id=np.asarray([0, 1], dtype=np.int64),
        source_x_m=np.asarray([0.0, 50.0]),
        source_y_m=np.asarray([0.0, 5.0]),
        source_surface_elevation_m=np.asarray([100.0, 110.0]),
        source_half_intercept_time_s=np.asarray([0.010, 0.012]),
        source_weathering_thickness_m=np.asarray([10.0, 12.0]),
        source_refractor_elevation_m=np.asarray([90.0, 98.0]),
        source_floating_datum_elevation_m=np.asarray([120.0, 120.0]),
        source_weathering_replacement_shift_s=np.asarray([-0.0085, -0.0102]),
        source_floating_datum_elevation_shift_s=np.asarray([0.0010, 0.0015]),
        source_flat_datum_shift_s=np.asarray([0.010, 0.011]),
        source_refraction_shift_s=np.asarray([0.0025, 0.0023]),
        source_datum_status=np.asarray(['ok', 'ok'], dtype='<U16'),
        receiver_endpoint_key=np.asarray(['r0', 'r1'], dtype='<U2'),
        receiver_id=np.asarray([200, 201], dtype=np.int64),
        receiver_node_id=np.asarray([1, 2], dtype=np.int64),
        receiver_x_m=np.asarray([55.0, 105.0]),
        receiver_y_m=np.asarray([5.0, 10.0]),
        receiver_surface_elevation_m=np.asarray([110.0, 120.0]),
        receiver_half_intercept_time_s=np.asarray([0.012, 0.014]),
        receiver_weathering_thickness_m=np.asarray([12.0, 14.0]),
        receiver_refractor_elevation_m=np.asarray([98.0, 106.0]),
        receiver_floating_datum_elevation_m=np.asarray([120.0, 120.0]),
        receiver_weathering_replacement_shift_s=np.asarray([-0.0102, -0.0119]),
        receiver_floating_datum_elevation_shift_s=np.asarray([0.0015, 0.0020]),
        receiver_flat_datum_shift_s=np.asarray([0.011, 0.012]),
        receiver_refraction_shift_s=np.asarray([0.0023, 0.0021]),
        receiver_datum_status=np.asarray(['ok', 'ok'], dtype='<U16'),
        sorted_trace_index=sorted_trace_index,
        valid_observation_mask_sorted=valid_observation,
        used_observation_mask_sorted=used_observation,
        source_node_id_sorted=source_node_sorted,
        receiver_node_id_sorted=receiver_node_sorted,
        source_surface_elevation_m_sorted=np.asarray([100.0, 110.0, 100.0, 110.0]),
        receiver_surface_elevation_m_sorted=np.asarray([110.0, 120.0, 120.0, 110.0]),
        source_floating_datum_elevation_m_sorted=np.full(n_traces, 120.0),
        receiver_floating_datum_elevation_m_sorted=np.full(n_traces, 120.0),
        source_weathering_thickness_m_sorted=source_thickness_sorted,
        receiver_weathering_thickness_m_sorted=receiver_thickness_sorted,
        source_refractor_elevation_m_sorted=np.asarray([90.0, 98.0, 90.0, 98.0]),
        receiver_refractor_elevation_m_sorted=np.asarray([98.0, 106.0, 106.0, 98.0]),
        source_half_intercept_time_s_sorted=source_half_sorted,
        receiver_half_intercept_time_s_sorted=receiver_half_sorted,
        source_weathering_replacement_shift_s_sorted=source_weathering_sorted,
        receiver_weathering_replacement_shift_s_sorted=receiver_weathering_sorted,
        source_floating_datum_elevation_shift_s_sorted=source_floating_sorted,
        receiver_floating_datum_elevation_shift_s_sorted=receiver_floating_sorted,
        source_flat_datum_shift_s_sorted=source_flat_sorted,
        receiver_flat_datum_shift_s_sorted=receiver_flat_sorted,
        source_refraction_shift_s_sorted=source_refraction_sorted,
        receiver_refraction_shift_s_sorted=receiver_refraction_sorted,
        weathering_replacement_trace_shift_s_sorted=weathering_trace,
        floating_datum_elevation_shift_s_sorted=(
            source_floating_sorted + receiver_floating_sorted
        ),
        flat_datum_shift_s_sorted=source_flat_sorted + receiver_flat_sorted,
        refraction_trace_shift_s_sorted=refraction_trace,
        trace_static_status_sorted=trace_status,
        trace_static_valid_mask_sorted=trace_valid,
        estimated_first_break_time_s_sorted=np.asarray([0.05, 0.06, 0.07, np.nan]),
        first_break_residual_s_sorted=np.asarray([0.001, -0.002, -0.001, np.nan]),
        row_trace_index_sorted=np.asarray([0, 1, 2], dtype=np.int64),
        row_source_node_id=np.asarray([0, 1, 0], dtype=np.int64),
        row_receiver_node_id=np.asarray([1, 2, 2], dtype=np.int64),
        row_distance_m=np.asarray([100.0, 200.0, 300.0]),
        observed_pick_time_s=np.asarray([0.050, 0.060, 0.070]),
        modeled_pick_time_s=np.asarray([0.049, 0.062, 0.071]),
        residual_time_s=np.asarray([0.001, -0.002, -0.001]),
        used_row_mask=np.asarray([True, False, True], dtype=bool),
        rejected_by_robust_mask=np.asarray([False, True, False], dtype=bool),
        qc={
            'robust_iteration_count': 1,
            'floating_datum_below_refractor_count': 0,
            'flat_datum_below_refractor_count': 0,
        },
    )


def _result_with_weathering_velocity(
    weathering_velocity_m_s: float,
) -> RefractionDatumStaticsResult:
    return replace(
        _result(),
        weathering_velocity_m_s=weathering_velocity_m_s,
        replacement_slowness_delta_s_per_m=(
            (1.0 / 2500.0) - (1.0 / weathering_velocity_m_s)
        ),
    )
