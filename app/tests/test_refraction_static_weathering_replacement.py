from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from app.api.schemas import RefractionStaticApplyOptions
from app.services.refraction_static_types import (
    RefractionWeatheringThicknessResult,
)
import app.services.refraction_static_weathering_replacement as replacement_module
from app.services.refraction_static_weathering_replacement import (
    REFRACTION_WEATHERING_REPLACEMENT_NODES_CSV_NAME,
    REFRACTION_WEATHERING_REPLACEMENT_QC_JSON_NAME,
    REFRACTION_WEATHERING_REPLACEMENT_RECEIVERS_CSV_NAME,
    REFRACTION_WEATHERING_REPLACEMENT_SOURCES_CSV_NAME,
    REFRACTION_WEATHERING_REPLACEMENT_TRACE_PREVIEW_CSV_NAME,
    RefractionWeatheringReplacementStaticsError,
    build_refraction_weathering_replacement_statics,
    compute_weathering_replacement_shift_s,
    compute_weathering_replacement_shift_scalar_s,
    compute_weathering_replacement_statics_from_first_breaks,
)

WEATHERING_VELOCITY_M_S = 800.0
BEDROCK_VELOCITY_M_S = 2500.0
BEDROCK_SLOWNESS_S_PER_M = 1.0 / BEDROCK_VELOCITY_M_S
SLOWNESS_DELTA_S_PER_M = 1.0 / BEDROCK_VELOCITY_M_S - 1.0 / WEATHERING_VELOCITY_M_S


def _apply_options(max_abs_shift_ms: float = 250.0) -> RefractionStaticApplyOptions:
    return RefractionStaticApplyOptions.model_validate(
        {
            'mode': 'refraction_from_raw',
            'interpolation': 'linear',
            'fill_value': 0.0,
            'max_abs_shift_ms': max_abs_shift_ms,
            'output_dtype': 'float32',
            'register_corrected_file': False,
        }
    )


def _weathering_result(
    *,
    node_thickness: np.ndarray | None = None,
    node_status: np.ndarray | None = None,
    source_node_id_sorted: np.ndarray | None = None,
    receiver_node_id_sorted: np.ndarray | None = None,
) -> RefractionWeatheringThicknessResult:
    node_id = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    thickness = (
        np.asarray([10.0, 12.0, 15.0, 18.0, 20.0, np.nan], dtype=np.float64)
        if node_thickness is None
        else np.asarray(node_thickness, dtype=np.float64)
    )
    status = (
        np.asarray(['ok', 'ok', 'ok', 'ok', 'ok', 'inactive'], dtype='<U32')
        if node_status is None
        else np.asarray(node_status, dtype='<U32')
    )
    source_sorted = (
        np.asarray([0, 1, 2, 3, 4, 0, 5], dtype=np.int64)
        if source_node_id_sorted is None
        else np.asarray(source_node_id_sorted, dtype=np.int64)
    )
    receiver_sorted = (
        np.asarray([1, 2, 3, 4, 0, 2, 5], dtype=np.int64)
        if receiver_node_id_sorted is None
        else np.asarray(receiver_node_id_sorted, dtype=np.int64)
    )
    sorted_trace_index = np.asarray([4, 2, 0, 1, 3, 5, 6], dtype=np.int64)
    valid_observation = np.asarray([True, True, False, True, True, True, False])
    used_observation = np.asarray([True, False, False, True, True, False, False])
    node_pos = {int(node): index for index, node in enumerate(node_id.tolist())}

    def _map_float(values: np.ndarray) -> np.ndarray:
        out = np.full(values.shape, np.nan, dtype=np.float64)
        for index, raw_node in enumerate(values.tolist()):
            node_index = node_pos.get(int(raw_node))
            if node_index is not None:
                out[index] = thickness[node_index]
        return out

    def _map_status(values: np.ndarray) -> np.ndarray:
        out = np.full(values.shape, 'missing_node', dtype='<U32')
        for index, raw_node in enumerate(values.tolist()):
            node_index = node_pos.get(int(raw_node))
            if node_index is not None:
                out[index] = status[node_index]
        return out

    n_nodes = int(node_id.shape[0])
    node_x = node_id.astype(np.float64) * 100.0
    zeros_nodes = np.zeros(n_nodes, dtype=np.float64)
    endpoint_keys = np.asarray([f'endpoint:{int(node)}' for node in node_id])
    source_key_sorted = np.asarray(
        [f'endpoint:{int(node)}' for node in source_sorted],
        dtype=object,
    )
    receiver_key_sorted = np.asarray(
        [f'endpoint:{int(node)}' for node in receiver_sorted],
        dtype=object,
    )
    return RefractionWeatheringThicknessResult(
        bedrock_velocity_mode='solve_global',
        bedrock_slowness_s_per_m=BEDROCK_SLOWNESS_S_PER_M,
        bedrock_velocity_m_s=BEDROCK_VELOCITY_M_S,
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        node_id=node_id,
        node_x_m=node_x,
        node_y_m=zeros_nodes.copy(),
        node_surface_elevation_m=np.full(n_nodes, 100.0, dtype=np.float64),
        node_kind=np.full(n_nodes, 'linked', dtype='<U32'),
        node_half_intercept_time_s=np.zeros(n_nodes, dtype=np.float64),
        node_half_intercept_time_ms=np.zeros(n_nodes, dtype=np.float64),
        node_weathering_thickness_m=np.ascontiguousarray(thickness, dtype=np.float64),
        node_refractor_elevation_m=100.0 - thickness,
        node_solution_status=status.copy(),
        node_weathering_status=status.copy(),
        node_pick_count=np.asarray([4, 4, 3, 3, 2, 0], dtype=np.int64),
        node_used_pick_count=np.asarray([4, 3, 3, 2, 2, 0], dtype=np.int64),
        node_rejected_pick_count=np.asarray([0, 1, 0, 1, 0, 0], dtype=np.int64),
        node_residual_rms_s=np.asarray(
            [0.001, 0.002, 0.003, 0.004, 0.005, np.nan],
            dtype=np.float64,
        ),
        node_residual_mad_s=np.asarray(
            [0.0005, 0.001, 0.0015, 0.002, 0.0025, np.nan],
            dtype=np.float64,
        ),
        source_endpoint_key=endpoint_keys.copy(),
        source_id=node_id.copy(),
        source_node_id=node_id.copy(),
        source_x_m=node_x.copy(),
        source_y_m=zeros_nodes.copy(),
        source_surface_elevation_m=np.full(n_nodes, 100.0, dtype=np.float64),
        source_half_intercept_time_s=np.zeros(n_nodes, dtype=np.float64),
        source_weathering_thickness_m=thickness.copy(),
        source_refractor_elevation_m=100.0 - thickness,
        source_weathering_status=status.copy(),
        receiver_endpoint_key=endpoint_keys.copy(),
        receiver_id=node_id.copy(),
        receiver_node_id=node_id.copy(),
        receiver_x_m=node_x.copy(),
        receiver_y_m=zeros_nodes.copy(),
        receiver_surface_elevation_m=np.full(n_nodes, 100.0, dtype=np.float64),
        receiver_half_intercept_time_s=np.zeros(n_nodes, dtype=np.float64),
        receiver_weathering_thickness_m=thickness.copy(),
        receiver_refractor_elevation_m=100.0 - thickness,
        receiver_weathering_status=status.copy(),
        sorted_trace_index=sorted_trace_index,
        valid_observation_mask_sorted=valid_observation,
        used_observation_mask_sorted=used_observation,
        source_endpoint_key_sorted=source_key_sorted,
        receiver_endpoint_key_sorted=receiver_key_sorted,
        source_node_id_sorted=source_sorted,
        receiver_node_id_sorted=receiver_sorted,
        source_half_intercept_time_s_sorted=np.zeros(
            sorted_trace_index.shape,
            dtype=np.float64,
        ),
        receiver_half_intercept_time_s_sorted=np.zeros(
            sorted_trace_index.shape,
            dtype=np.float64,
        ),
        source_weathering_thickness_m_sorted=_map_float(source_sorted),
        receiver_weathering_thickness_m_sorted=_map_float(receiver_sorted),
        source_refractor_elevation_m_sorted=(
            100.0 - _map_float(source_sorted)
        ),
        receiver_refractor_elevation_m_sorted=(
            100.0 - _map_float(receiver_sorted)
        ),
        source_weathering_status_sorted=_map_status(source_sorted),
        receiver_weathering_status_sorted=_map_status(receiver_sorted),
        estimated_first_break_time_s_sorted=np.linspace(
            0.05,
            0.11,
            int(sorted_trace_index.shape[0]),
            dtype=np.float64,
        ),
        first_break_residual_s_sorted=np.linspace(
            -0.003,
            0.003,
            int(sorted_trace_index.shape[0]),
            dtype=np.float64,
        ),
        row_trace_index_sorted=np.flatnonzero(valid_observation).astype(np.int64),
        row_source_node_id=source_sorted[valid_observation].astype(np.int64),
        row_receiver_node_id=receiver_sorted[valid_observation].astype(np.int64),
        row_distance_m=np.linspace(
            100.0,
            300.0,
            int(np.count_nonzero(valid_observation)),
            dtype=np.float64,
        ),
        observed_pick_time_s=np.linspace(
            0.05,
            0.07,
            int(np.count_nonzero(valid_observation)),
            dtype=np.float64,
        ),
        modeled_pick_time_s=np.linspace(
            0.049,
            0.071,
            int(np.count_nonzero(valid_observation)),
            dtype=np.float64,
        ),
        residual_time_s=np.linspace(
            0.001,
            -0.001,
            int(np.count_nonzero(valid_observation)),
            dtype=np.float64,
        ),
        used_row_mask=np.ones(int(np.count_nonzero(valid_observation)), dtype=bool),
        rejected_by_robust_mask=np.zeros(
            int(np.count_nonzero(valid_observation)),
            dtype=bool,
        ),
        qc={'method': 'gli_variable_thickness'},
    )


def test_public_apis_are_importable() -> None:
    assert callable(compute_weathering_replacement_shift_s)
    assert callable(compute_weathering_replacement_shift_scalar_s)
    assert callable(build_refraction_weathering_replacement_statics)
    assert callable(compute_weathering_replacement_statics_from_first_breaks)


def test_math_helper_computes_vector_scalar_and_preserves_nan() -> None:
    thickness = np.asarray([10.0, np.nan, 0.0], dtype=np.float64)

    shift = compute_weathering_replacement_shift_s(
        weathering_thickness_m=thickness,
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        bedrock_velocity_m_s=BEDROCK_VELOCITY_M_S,
    )

    assert shift[0] == pytest.approx(10.0 * SLOWNESS_DELTA_S_PER_M)
    assert np.isnan(shift[1])
    assert shift[2] == pytest.approx(0.0)
    assert shift[0] < 0.0
    assert compute_weathering_replacement_shift_scalar_s(
        weathering_thickness_m=12.0,
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        bedrock_velocity_m_s=BEDROCK_VELOCITY_M_S,
    ) == pytest.approx(12.0 * SLOWNESS_DELTA_S_PER_M)


@pytest.mark.parametrize(
    ('weathering_velocity', 'bedrock_velocity', 'match'),
    [
        (np.nan, BEDROCK_VELOCITY_M_S, 'weathering_velocity_m_s'),
        (0.0, BEDROCK_VELOCITY_M_S, 'weathering_velocity_m_s'),
        (WEATHERING_VELOCITY_M_S, np.inf, 'bedrock_velocity_m_s'),
        (WEATHERING_VELOCITY_M_S, -1.0, 'bedrock_velocity_m_s'),
        (WEATHERING_VELOCITY_M_S, WEATHERING_VELOCITY_M_S, 'greater'),
    ],
)
def test_math_helper_rejects_invalid_velocity(
    weathering_velocity: float,
    bedrock_velocity: float,
    match: str,
) -> None:
    with pytest.raises(RefractionWeatheringReplacementStaticsError, match=match):
        compute_weathering_replacement_shift_s(
            weathering_thickness_m=np.asarray([10.0], dtype=np.float64),
            weathering_velocity_m_s=weathering_velocity,
            bedrock_velocity_m_s=bedrock_velocity,
        )


def test_build_computes_node_endpoint_and_trace_shifts_in_sorted_order() -> None:
    weathering = _weathering_result()

    result = build_refraction_weathering_replacement_statics(
        weathering_result=weathering,
        apply_options=_apply_options(),
    )

    expected_node_shift = (
        weathering.node_weathering_thickness_m * SLOWNESS_DELTA_S_PER_M
    )
    np.testing.assert_allclose(
        result.node_weathering_replacement_shift_s[:5],
        expected_node_shift[:5],
        rtol=1.0e-12,
    )
    assert np.isnan(result.node_weathering_replacement_shift_s[5])
    assert result.node_weathering_replacement_shift_ms[0] == pytest.approx(
        expected_node_shift[0] * 1000.0
    )

    source_zero = int(np.flatnonzero(result.source_node_id == 0)[0])
    receiver_zero = int(np.flatnonzero(result.receiver_node_id == 0)[0])
    assert result.source_weathering_replacement_shift_s[source_zero] == pytest.approx(
        expected_node_shift[0]
    )
    assert result.receiver_weathering_replacement_shift_s[receiver_zero] == pytest.approx(
        expected_node_shift[0]
    )
    assert (
        result.source_weathering_replacement_shift_s[source_zero]
        == result.receiver_weathering_replacement_shift_s[receiver_zero]
    )

    expected_source = expected_node_shift[result.source_node_id_sorted[:6]]
    expected_receiver = expected_node_shift[result.receiver_node_id_sorted[:6]]
    np.testing.assert_allclose(
        result.source_weathering_replacement_shift_s_sorted[:6],
        expected_source,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.receiver_weathering_replacement_shift_s_sorted[:6],
        expected_receiver,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.weathering_replacement_trace_shift_s_sorted[:6],
        expected_source + expected_receiver,
        rtol=1.0e-12,
    )
    np.testing.assert_array_equal(
        result.sorted_trace_index,
        np.asarray([4, 2, 0, 1, 3, 5, 6], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        result.valid_observation_mask_sorted,
        weathering.valid_observation_mask_sorted,
    )
    assert result.used_observation_mask_sorted[1] == np.False_
    assert np.isfinite(result.weathering_replacement_trace_shift_s_sorted[1])
    assert result.valid_observation_mask_sorted[2] == np.False_
    assert np.isfinite(result.weathering_replacement_trace_shift_s_sorted[2])
    assert result.trace_static_valid_mask_sorted[:6].tolist() == [True] * 6
    assert result.trace_static_valid_mask_sorted[6] == np.False_


def test_statuses_preserve_inactive_invalid_and_inherited_conditions() -> None:
    thickness = np.asarray([10.0, np.nan, -1.0, 18.0, 20.0, np.nan])
    status = np.asarray(
        [
            'ok',
            'ok',
            'ok',
            'exceeds_max_thickness',
            'clipped_half_intercept_upper',
            'inactive',
        ],
        dtype='<U32',
    )

    result = build_refraction_weathering_replacement_statics(
        weathering_result=_weathering_result(
            node_thickness=thickness,
            node_status=status,
        ),
        apply_options=_apply_options(),
    )

    assert result.node_static_status.tolist() == [
        'ok',
        'invalid_weathering_thickness',
        'negative_weathering_thickness',
        'exceeds_max_thickness',
        'clipped_half_intercept_upper',
        'inactive',
    ]
    assert np.isnan(result.node_weathering_replacement_shift_s[1])
    assert np.isnan(result.node_weathering_replacement_shift_s[2])
    assert np.isfinite(result.node_weathering_replacement_shift_s[3])
    assert result.qc['invalid_weathering_thickness_count'] == 1
    assert result.qc['exceeds_max_thickness_count'] == 1


def test_elevation_only_weathering_statuses_do_not_block_replacement_shift() -> None:
    status = np.asarray(
        [
            'invalid_surface_elevation',
            'invalid_refractor_elevation',
            'ok',
            'ok',
            'ok',
            'inactive',
        ],
        dtype='<U32',
    )

    result = build_refraction_weathering_replacement_statics(
        weathering_result=_weathering_result(node_status=status),
        apply_options=_apply_options(),
    )

    expected_node_shift = np.asarray([10.0, 12.0], dtype=np.float64) * (
        SLOWNESS_DELTA_S_PER_M
    )
    np.testing.assert_allclose(
        result.node_weathering_replacement_shift_s[:2],
        expected_node_shift,
        rtol=1.0e-12,
    )
    assert result.node_static_status[:2].tolist() == ['ok', 'ok']
    assert result.source_static_status[0] == 'ok'
    assert result.receiver_static_status[1] == 'ok'
    assert result.trace_static_status_sorted[0] == 'ok'
    assert np.isfinite(result.weathering_replacement_trace_shift_s_sorted[0])
    assert result.qc['invalid_weathering_thickness_count'] == 0


def test_status_priority_is_deterministic_for_components_and_traces() -> None:
    thickness = np.asarray([np.nan, 12.0, 15.0, 18.0, 20.0, np.nan])
    status = np.asarray(
        [
            'low_fold',
            'clipped_half_intercept_lower',
            'clipped_half_intercept_upper',
            'low_fold',
            'ok',
            'inactive',
        ],
        dtype='<U32',
    )

    result = build_refraction_weathering_replacement_statics(
        weathering_result=_weathering_result(
            node_thickness=thickness,
            node_status=status,
        ),
        apply_options=_apply_options(max_abs_shift_ms=20.0),
    )

    assert result.node_static_status[0] == 'invalid_weathering_thickness'
    assert result.node_static_status[1] == 'clipped_half_intercept_lower'
    assert result.node_static_status[2] == 'clipped_half_intercept_upper'
    assert result.trace_static_status_sorted[3] == 'exceeds_max_abs_shift'
    assert result.trace_static_status_sorted[6] == 'inactive'


def test_unknown_endpoint_nodes_raise() -> None:
    weathering = _weathering_result()
    bad_source = weathering.source_node_id.copy()
    bad_source[0] = 999
    with pytest.raises(
        RefractionWeatheringReplacementStaticsError,
        match='source_node_id',
    ):
        build_refraction_weathering_replacement_statics(
            weathering_result=replace(weathering, source_node_id=bad_source),
        )

    bad_receiver = weathering.receiver_node_id.copy()
    bad_receiver[0] = 999
    with pytest.raises(
        RefractionWeatheringReplacementStaticsError,
        match='receiver_node_id',
    ):
        build_refraction_weathering_replacement_statics(
            weathering_result=replace(weathering, receiver_node_id=bad_receiver),
        )


def test_missing_trace_nodes_produce_nan_component_and_invalid_mask() -> None:
    weathering = _weathering_result(
        source_node_id_sorted=np.asarray([999, 1, 2, 3, 4, 0, 5], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([1, 2, 999, 4, 0, 2, 5], dtype=np.int64),
    )

    result = build_refraction_weathering_replacement_statics(
        weathering_result=weathering,
        apply_options=_apply_options(),
    )

    assert np.isnan(result.source_weathering_replacement_shift_s_sorted[0])
    assert result.source_static_status_sorted[0] == 'missing_node'
    assert result.trace_static_valid_mask_sorted[0] == np.False_
    assert np.isnan(result.receiver_weathering_replacement_shift_s_sorted[2])
    assert result.receiver_static_status_sorted[2] == 'missing_node'
    assert result.trace_static_valid_mask_sorted[2] == np.False_


def test_max_abs_shift_marks_status_without_clipping() -> None:
    result = build_refraction_weathering_replacement_statics(
        weathering_result=_weathering_result(),
        apply_options=_apply_options(max_abs_shift_ms=18.0),
    )
    expected_first = (10.0 + 12.0) * SLOWNESS_DELTA_S_PER_M

    assert result.weathering_replacement_trace_shift_s_sorted[0] == pytest.approx(
        expected_first
    )
    assert abs(expected_first) * 1000.0 > 18.0
    assert result.trace_static_status_sorted[0] == 'exceeds_max_abs_shift'
    assert result.trace_static_valid_mask_sorted[0] == np.False_
    assert result.qc['exceeds_max_abs_shift_count'] == 6
    assert result.qc['invalid_trace_shift_count'] == 1


def test_max_abs_shift_marks_node_and_endpoint_status_without_clipping() -> None:
    result = build_refraction_weathering_replacement_statics(
        weathering_result=_weathering_result(),
        apply_options=_apply_options(max_abs_shift_ms=12.0),
    )
    expected_node_shift = np.asarray(
        [10.0, 12.0, 15.0, 18.0, 20.0],
        dtype=np.float64,
    ) * SLOWNESS_DELTA_S_PER_M

    np.testing.assert_allclose(
        result.node_weathering_replacement_shift_s[:5],
        expected_node_shift,
        rtol=1.0e-12,
    )
    assert result.node_static_status.tolist() == [
        'ok',
        'ok',
        'exceeds_max_abs_shift',
        'exceeds_max_abs_shift',
        'exceeds_max_abs_shift',
        'inactive',
    ]
    assert result.source_static_status[2] == 'exceeds_max_abs_shift'
    assert result.receiver_static_status[2] == 'exceeds_max_abs_shift'
    assert result.source_weathering_replacement_shift_s[2] == pytest.approx(
        expected_node_shift[2]
    )
    assert result.receiver_weathering_replacement_shift_s[2] == pytest.approx(
        expected_node_shift[2]
    )
    assert result.source_static_status_sorted[2] == 'exceeds_max_abs_shift'
    assert result.receiver_static_status_sorted[1] == 'exceeds_max_abs_shift'
    assert (
        result.qc['node_static_status_counts']['exceeds_max_abs_shift'] == 3
    )
    assert (
        result.qc['source_static_status_counts']['exceeds_max_abs_shift'] == 3
    )
    assert (
        result.qc['receiver_static_status_counts']['exceeds_max_abs_shift'] == 3
    )


def test_sign_convention_negative_shift_moves_impulse_earlier() -> None:
    shift_s = compute_weathering_replacement_shift_scalar_s(
        weathering_thickness_m=10.0,
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        bedrock_velocity_m_s=BEDROCK_VELOCITY_M_S,
    )

    assert shift_s < 0.0
    dt_s = 0.001
    raw_impulse_sample = 10
    # The later apply stage follows corrected(t) = raw(t - shift_s), so a
    # negative replacement shift advances the sampled event time.
    corrected_impulse_sample = raw_impulse_sample + int(round(shift_s / dt_s))
    assert corrected_impulse_sample < raw_impulse_sample


def test_qc_and_artifacts_include_velocity_status_and_shift_columns(
    tmp_path: Path,
) -> None:
    result = build_refraction_weathering_replacement_statics(
        weathering_result=_weathering_result(),
        apply_options=_apply_options(),
        job_dir=tmp_path,
    )

    for name in (
        REFRACTION_WEATHERING_REPLACEMENT_QC_JSON_NAME,
        REFRACTION_WEATHERING_REPLACEMENT_NODES_CSV_NAME,
        REFRACTION_WEATHERING_REPLACEMENT_SOURCES_CSV_NAME,
        REFRACTION_WEATHERING_REPLACEMENT_RECEIVERS_CSV_NAME,
        REFRACTION_WEATHERING_REPLACEMENT_TRACE_PREVIEW_CSV_NAME,
    ):
        assert (tmp_path / name).is_file()

    qc = json.loads(
        (tmp_path / REFRACTION_WEATHERING_REPLACEMENT_QC_JSON_NAME).read_text(
            encoding='utf-8'
        )
    )
    assert qc['static_component'] == 'weathering_replacement'
    assert qc['bedrock_velocity_m_s'] == pytest.approx(BEDROCK_VELOCITY_M_S)
    assert qc['replacement_slowness_delta_s_per_m'] == pytest.approx(
        SLOWNESS_DELTA_S_PER_M
    )
    assert qc['node_shift_min_ms'] == pytest.approx(
        np.min(result.node_weathering_replacement_shift_s[:5] * 1000.0)
    )
    assert qc['trace_shift_p95_abs_ms'] is not None
    assert qc['inactive_node_count'] == 1
    assert qc['sign_convention'] == 'corrected(t) = raw(t - shift_s)'
    assert qc['formula'] == 'shift = z * (1/vb - 1/vw)'

    with (tmp_path / REFRACTION_WEATHERING_REPLACEMENT_NODES_CSV_NAME).open(
        encoding='utf-8',
        newline='',
    ) as handle:
        node_rows = list(csv.DictReader(handle))
    assert 'weathering_replacement_shift_ms' in node_rows[0]
    assert node_rows[-1]['static_status'] == 'inactive'

    with (tmp_path / REFRACTION_WEATHERING_REPLACEMENT_TRACE_PREVIEW_CSV_NAME).open(
        encoding='utf-8',
        newline='',
    ) as handle:
        trace_rows = list(csv.DictReader(handle))
    assert 'weathering_replacement_trace_shift_ms' in trace_rows[0]
    assert trace_rows[-1]['trace_static_valid'] == 'False'


def test_velocity_context_validation_rejects_bad_slowness() -> None:
    weathering = replace(
        _weathering_result(),
        bedrock_slowness_s_per_m=BEDROCK_SLOWNESS_S_PER_M * 1.01,
    )

    with pytest.raises(RefractionWeatheringReplacementStaticsError, match='slowness'):
        build_refraction_weathering_replacement_statics(
            weathering_result=weathering,
            apply_options=_apply_options(),
        )


def test_high_level_pipeline_calls_weathering_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    weathering = _weathering_result()
    calls: list[dict[str, Any]] = []

    def _capture_estimate(**kwargs: Any) -> RefractionWeatheringThicknessResult:
        calls.append(kwargs)
        return weathering

    monkeypatch.setattr(
        replacement_module,
        'estimate_weathering_thickness_from_first_breaks',
        _capture_estimate,
    )
    req = SimpleNamespace(apply=_apply_options())
    state = object()

    result = compute_weathering_replacement_statics_from_first_breaks(
        req=req,  # type: ignore[arg-type]
        state=state,  # type: ignore[arg-type]
        job_dir=tmp_path,
    )

    assert calls == [{'req': req, 'state': state, 'job_dir': tmp_path}]
    assert result.qc['static_component'] == 'weathering_replacement'
    assert (tmp_path / REFRACTION_WEATHERING_REPLACEMENT_QC_JSON_NAME).is_file()
