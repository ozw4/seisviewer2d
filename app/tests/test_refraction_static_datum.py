from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import app.statics.refraction.application.datum as datum_module
from app.api.schemas import RefractionStaticApplyOptions, RefractionStaticDatumRequest
from app.core.state import AppState
from app.statics.refraction.adapters.seisviewer2d.runtime import (
    SeisViewer2DRefractionRuntime,
)
from app.statics.refraction.application.datum import (
    REFRACTION_DATUM_NODES_CSV_NAME,
    REFRACTION_DATUM_RECEIVERS_CSV_NAME,
    REFRACTION_DATUM_SOURCES_CSV_NAME,
    REFRACTION_DATUM_STATICS_QC_JSON_NAME,
    REFRACTION_DATUM_TRACE_PREVIEW_CSV_NAME,
    RefractionDatumStaticsError,
    build_refraction_datum_statics,
    compose_refraction_trace_shift_s,
    compute_datum_refraction_statics_from_first_breaks,
    compute_flat_datum_shift_s,
    compute_floating_datum_elevation_shift_s,
)
from app.statics.refraction.contracts.result_types import (
    RefractionWeatheringReplacementStaticsResult,
)

WEATHERING_VELOCITY_M_S = 800.0
BEDROCK_VELOCITY_M_S = 2500.0
BEDROCK_SLOWNESS_S_PER_M = 1.0 / BEDROCK_VELOCITY_M_S
SLOWNESS_DELTA_S_PER_M = 1.0 / BEDROCK_VELOCITY_M_S - (
    1.0 / WEATHERING_VELOCITY_M_S
)


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


def _datum(**overrides: Any) -> RefractionStaticDatumRequest:
    payload: dict[str, Any] = {
        'mode': 'floating_and_flat',
        'floating_datum_mode': 'constant',
        'floating_datum_elevation_m': 180.0,
        'flat_datum_elevation_m': 200.0,
    }
    payload.update(overrides)
    return RefractionStaticDatumRequest.model_validate(payload)


def _replacement_result(
    *,
    source_surface: np.ndarray | None = None,
    receiver_surface: np.ndarray | None = None,
    source_endpoint_key: np.ndarray | None = None,
    receiver_endpoint_key: np.ndarray | None = None,
    source_node_id: np.ndarray | None = None,
    receiver_node_id: np.ndarray | None = None,
    source_endpoint_key_sorted: np.ndarray | None = None,
    receiver_endpoint_key_sorted: np.ndarray | None = None,
    source_node_id_sorted: np.ndarray | None = None,
    receiver_node_id_sorted: np.ndarray | None = None,
    valid_observation: np.ndarray | None = None,
) -> RefractionWeatheringReplacementStaticsResult:
    node_id = np.asarray([0, 1, 2], dtype=np.int64)
    source_nodes = (
        np.asarray([0, 1, 2], dtype=np.int64)
        if source_node_id is None
        else np.asarray(source_node_id, dtype=np.int64)
    )
    receiver_nodes = (
        np.asarray([0, 1, 2], dtype=np.int64)
        if receiver_node_id is None
        else np.asarray(receiver_node_id, dtype=np.int64)
    )
    source_key = (
        np.asarray(['s0', 's1', 's2'], dtype=object)
        if source_endpoint_key is None
        else np.asarray(source_endpoint_key, dtype=object)
    )
    receiver_key = (
        np.asarray(['r0', 'r1', 'r2'], dtype=object)
        if receiver_endpoint_key is None
        else np.asarray(receiver_endpoint_key, dtype=object)
    )
    source_sorted = (
        np.asarray([0, 1, 2], dtype=np.int64)
        if source_node_id_sorted is None
        else np.asarray(source_node_id_sorted, dtype=np.int64)
    )
    receiver_sorted = (
        np.asarray([0, 1, 2], dtype=np.int64)
        if receiver_node_id_sorted is None
        else np.asarray(receiver_node_id_sorted, dtype=np.int64)
    )
    source_key_sorted = (
        source_key.copy()
        if source_endpoint_key_sorted is None
        else np.asarray(source_endpoint_key_sorted, dtype=object)
    )
    receiver_key_sorted = (
        receiver_key.copy()
        if receiver_endpoint_key_sorted is None
        else np.asarray(receiver_endpoint_key_sorted, dtype=object)
    )
    source_surface_arr = (
        np.asarray([190.0, 200.0, 210.0], dtype=np.float64)
        if source_surface is None
        else np.asarray(source_surface, dtype=np.float64)
    )
    receiver_surface_arr = (
        np.asarray([195.0, 205.0, 215.0], dtype=np.float64)
        if receiver_surface is None
        else np.asarray(receiver_surface, dtype=np.float64)
    )
    source_thickness = np.asarray([50.0, 52.0, 55.0], dtype=np.float64)
    receiver_thickness = np.asarray([51.0, 53.0, 56.0], dtype=np.float64)
    source_shift = source_thickness * SLOWNESS_DELTA_S_PER_M
    receiver_shift = receiver_thickness * SLOWNESS_DELTA_S_PER_M

    def _map(
        values: np.ndarray,
        keys: np.ndarray,
        sorted_keys: np.ndarray,
    ) -> np.ndarray:
        pos = {str(key): index for index, key in enumerate(keys.tolist())}
        out = np.full(sorted_keys.shape, np.nan, dtype=np.float64)
        for index, raw_key in enumerate(sorted_keys.tolist()):
            mapped = pos.get(str(raw_key))
            if mapped is not None:
                out[index] = values[mapped]
        return out

    source_shift_sorted = _map(source_shift, source_key, source_key_sorted)
    receiver_shift_sorted = _map(receiver_shift, receiver_key, receiver_key_sorted)
    trace_shift = source_shift_sorted + receiver_shift_sorted
    sorted_trace_index = np.asarray([2, 0, 1], dtype=np.int64)
    valid = (
        np.asarray([True, True, False], dtype=bool)
        if valid_observation is None
        else np.asarray(valid_observation, dtype=bool)
    )
    used = np.asarray([True, False, False], dtype=bool)
    status = np.full(sorted_trace_index.shape, 'ok', dtype='<U32')
    node_surface = np.asarray([192.5, 202.5, 212.5], dtype=np.float64)
    node_thickness = np.asarray([50.5, 52.5, 55.5], dtype=np.float64)
    return RefractionWeatheringReplacementStaticsResult(
        bedrock_velocity_mode='solve_global',
        bedrock_slowness_s_per_m=BEDROCK_SLOWNESS_S_PER_M,
        bedrock_velocity_m_s=BEDROCK_VELOCITY_M_S,
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        replacement_slowness_delta_s_per_m=SLOWNESS_DELTA_S_PER_M,
        node_id=node_id,
        node_x_m=np.asarray([0.0, 100.0, 200.0], dtype=np.float64),
        node_y_m=np.zeros(3, dtype=np.float64),
        node_surface_elevation_m=node_surface,
        node_kind=np.full(3, 'linked', dtype='<U32'),
        node_weathering_thickness_m=node_thickness,
        node_refractor_elevation_m=node_surface - node_thickness,
        node_half_intercept_time_s=np.asarray([0.010, 0.012, 0.015], dtype=np.float64),
        node_solution_status=np.full(3, 'ok', dtype='<U32'),
        node_weathering_status=np.full(3, 'ok', dtype='<U32'),
        node_weathering_replacement_shift_s=node_thickness
        * SLOWNESS_DELTA_S_PER_M,
        node_weathering_replacement_shift_ms=node_thickness
        * SLOWNESS_DELTA_S_PER_M
        * 1000.0,
        node_static_status=np.full(3, 'ok', dtype='<U32'),
        node_pick_count=np.asarray([4, 4, 4], dtype=np.int64),
        node_used_pick_count=np.asarray([4, 3, 3], dtype=np.int64),
        node_rejected_pick_count=np.asarray([0, 1, 1], dtype=np.int64),
        node_residual_rms_s=np.asarray([0.001, 0.002, 0.003], dtype=np.float64),
        node_residual_mad_s=np.asarray([0.001, 0.001, 0.002], dtype=np.float64),
        source_endpoint_key=source_key,
        source_id=np.asarray([10, 11, 12], dtype=np.int64),
        source_node_id=source_nodes,
        source_x_m=np.asarray([0.0, 100.0, 200.0], dtype=np.float64),
        source_y_m=np.zeros(3, dtype=np.float64),
        source_surface_elevation_m=source_surface_arr,
        source_half_intercept_time_s=np.asarray([0.010, 0.012, 0.015], dtype=np.float64),
        source_weathering_thickness_m=source_thickness,
        source_refractor_elevation_m=source_surface_arr - source_thickness,
        source_weathering_replacement_shift_s=source_shift,
        source_static_status=np.full(3, 'ok', dtype='<U32'),
        receiver_endpoint_key=receiver_key,
        receiver_id=np.asarray([20, 21, 22], dtype=np.int64),
        receiver_node_id=receiver_nodes,
        receiver_x_m=np.asarray([0.0, 100.0, 200.0], dtype=np.float64),
        receiver_y_m=np.zeros(3, dtype=np.float64),
        receiver_surface_elevation_m=receiver_surface_arr,
        receiver_half_intercept_time_s=np.asarray(
            [0.011, 0.013, 0.016],
            dtype=np.float64,
        ),
        receiver_weathering_thickness_m=receiver_thickness,
        receiver_refractor_elevation_m=receiver_surface_arr - receiver_thickness,
        receiver_weathering_replacement_shift_s=receiver_shift,
        receiver_static_status=np.full(3, 'ok', dtype='<U32'),
        sorted_trace_index=sorted_trace_index,
        valid_observation_mask_sorted=valid,
        used_observation_mask_sorted=used,
        source_endpoint_key_sorted=source_key_sorted,
        receiver_endpoint_key_sorted=receiver_key_sorted,
        source_node_id_sorted=source_sorted,
        receiver_node_id_sorted=receiver_sorted,
        source_half_intercept_time_s_sorted=_map(
            np.asarray([0.010, 0.012, 0.015], dtype=np.float64),
            source_key,
            source_key_sorted,
        ),
        receiver_half_intercept_time_s_sorted=_map(
            np.asarray([0.011, 0.013, 0.016], dtype=np.float64),
            receiver_key,
            receiver_key_sorted,
        ),
        source_weathering_thickness_m_sorted=_map(
            source_thickness,
            source_key,
            source_key_sorted,
        ),
        receiver_weathering_thickness_m_sorted=_map(
            receiver_thickness,
            receiver_key,
            receiver_key_sorted,
        ),
        source_refractor_elevation_m_sorted=_map(
            source_surface_arr - source_thickness,
            source_key,
            source_key_sorted,
        ),
        receiver_refractor_elevation_m_sorted=_map(
            receiver_surface_arr - receiver_thickness,
            receiver_key,
            receiver_key_sorted,
        ),
        source_weathering_replacement_shift_s_sorted=source_shift_sorted,
        receiver_weathering_replacement_shift_s_sorted=receiver_shift_sorted,
        weathering_replacement_trace_shift_s_sorted=trace_shift,
        source_static_status_sorted=np.full(3, 'ok', dtype='<U32'),
        receiver_static_status_sorted=np.full(3, 'ok', dtype='<U32'),
        trace_static_status_sorted=status,
        trace_static_valid_mask_sorted=np.isfinite(trace_shift),
        estimated_first_break_time_s_sorted=np.asarray(
            [0.05, 0.06, 0.07],
            dtype=np.float64,
        ),
        first_break_residual_s_sorted=np.asarray(
            [0.001, -0.001, 0.002],
            dtype=np.float64,
        ),
        row_trace_index_sorted=np.asarray([0, 1], dtype=np.int64),
        row_source_node_id=np.asarray([0, 1], dtype=np.int64),
        row_receiver_node_id=np.asarray([0, 1], dtype=np.int64),
        row_distance_m=np.asarray([100.0, 200.0], dtype=np.float64),
        observed_pick_time_s=np.asarray([0.051, 0.059], dtype=np.float64),
        modeled_pick_time_s=np.asarray([0.050, 0.060], dtype=np.float64),
        residual_time_s=np.asarray([0.001, -0.001], dtype=np.float64),
        used_row_mask=np.asarray([True, False], dtype=bool),
        rejected_by_robust_mask=np.asarray([False, True], dtype=bool),
        qc={'static_component': 'weathering_replacement'},
    )


def test_public_apis_are_importable() -> None:
    assert callable(compute_floating_datum_elevation_shift_s)
    assert callable(compute_flat_datum_shift_s)
    assert callable(compose_refraction_trace_shift_s)
    assert callable(build_refraction_datum_statics)
    assert callable(compute_datum_refraction_statics_from_first_breaks)


def test_math_helpers_compute_formulas_and_preserve_nan() -> None:
    source = np.asarray([100.0, 110.0, np.nan], dtype=np.float64)
    receiver = np.asarray([105.0, 115.0, 125.0], dtype=np.float64)
    floating_source = np.asarray([100.0, 100.0, 100.0], dtype=np.float64)
    floating_receiver = np.asarray([100.0, 100.0, 100.0], dtype=np.float64)

    floating = compute_floating_datum_elevation_shift_s(
        true_source_elevation_m=source,
        true_receiver_elevation_m=receiver,
        floating_source_elevation_m=floating_source,
        floating_receiver_elevation_m=floating_receiver,
        bedrock_velocity_m_s=BEDROCK_VELOCITY_M_S,
    )
    flat = compute_flat_datum_shift_s(
        flat_datum_elevation_m=200.0,
        floating_source_elevation_m=floating_source,
        floating_receiver_elevation_m=floating_receiver,
        bedrock_velocity_m_s=BEDROCK_VELOCITY_M_S,
    )
    composed = compose_refraction_trace_shift_s(
        weathering_replacement_trace_shift_s=np.asarray(
            [-0.01, -0.02, -0.03],
            dtype=np.float64,
        ),
        floating_datum_elevation_shift_s=floating,
        flat_datum_shift_s=flat,
    )

    assert floating[0] == pytest.approx(-0.002)
    assert floating[1] == pytest.approx(-0.010)
    assert np.isnan(floating[2])
    assert flat.tolist() == pytest.approx([0.08, 0.08, 0.08])
    assert composed[0] == pytest.approx(-0.01 - 0.002 + 0.08)
    assert np.isnan(composed[2])


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'bedrock_velocity_m_s': np.nan}, 'bedrock_velocity_m_s'),
        ({'bedrock_velocity_m_s': 0.0}, 'bedrock_velocity_m_s'),
    ],
)
def test_math_helpers_reject_invalid_velocity(kwargs: dict[str, Any], match: str) -> None:
    params = {
        'true_source_elevation_m': np.asarray([100.0], dtype=np.float64),
        'true_receiver_elevation_m': np.asarray([105.0], dtype=np.float64),
        'floating_source_elevation_m': np.asarray([100.0], dtype=np.float64),
        'floating_receiver_elevation_m': np.asarray([100.0], dtype=np.float64),
        'bedrock_velocity_m_s': BEDROCK_VELOCITY_M_S,
    }
    params.update(kwargs)
    with pytest.raises(RefractionDatumStaticsError, match=match):
        compute_floating_datum_elevation_shift_s(**params)


def test_math_helpers_reject_mismatched_lengths() -> None:
    with pytest.raises(RefractionDatumStaticsError, match='shape mismatch'):
        compute_flat_datum_shift_s(
            flat_datum_elevation_m=200.0,
            floating_source_elevation_m=np.asarray([100.0, 101.0]),
            floating_receiver_elevation_m=np.asarray([100.0]),
            bedrock_velocity_m_s=BEDROCK_VELOCITY_M_S,
        )


def test_build_constant_floating_and_flat_components_in_sorted_order(
    tmp_path: Path,
) -> None:
    replacement = _replacement_result()
    result = build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=_datum(),
        apply_options=_apply_options(),
        job_dir=tmp_path,
    )

    expected_floating = -(
        (
            result.source_surface_elevation_m_sorted
            - result.source_floating_datum_elevation_m_sorted
        )
        + (
            result.receiver_surface_elevation_m_sorted
            - result.receiver_floating_datum_elevation_m_sorted
        )
    ) / BEDROCK_VELOCITY_M_S
    expected_flat = (
        2.0 * 200.0
        - (
            result.source_floating_datum_elevation_m_sorted
            + result.receiver_floating_datum_elevation_m_sorted
        )
    ) / BEDROCK_VELOCITY_M_S

    np.testing.assert_allclose(
        result.weathering_replacement_trace_shift_s_sorted,
        replacement.weathering_replacement_trace_shift_s_sorted,
    )
    np.testing.assert_allclose(
        result.floating_datum_elevation_shift_s_sorted,
        expected_floating,
    )
    np.testing.assert_allclose(result.flat_datum_shift_s_sorted, expected_flat)
    expected_refraction = (
        replacement.weathering_replacement_trace_shift_s_sorted
        + expected_floating
        + expected_flat
    )
    expected_refraction[~replacement.valid_observation_mask_sorted] = np.nan
    np.testing.assert_allclose(
        result.refraction_trace_shift_s_sorted,
        expected_refraction,
    )
    np.testing.assert_array_equal(result.sorted_trace_index, [2, 0, 1])
    assert result.trace_static_valid_mask_sorted.tolist() == [True, True, False]
    assert result.valid_observation_mask_sorted.tolist() == [True, True, False]
    assert result.trace_static_status_sorted[2] == 'not_observed'
    assert result.qc['sign_convention'] == 'corrected(t) = raw(t - shift_s)'
    assert (tmp_path / REFRACTION_DATUM_STATICS_QC_JSON_NAME).is_file()
    assert (tmp_path / REFRACTION_DATUM_NODES_CSV_NAME).is_file()
    assert (tmp_path / REFRACTION_DATUM_SOURCES_CSV_NAME).is_file()
    assert (tmp_path / REFRACTION_DATUM_RECEIVERS_CSV_NAME).is_file()
    assert (tmp_path / REFRACTION_DATUM_TRACE_PREVIEW_CSV_NAME).is_file()

    qc = json.loads(
        (tmp_path / REFRACTION_DATUM_STATICS_QC_JSON_NAME).read_text(
            encoding='utf-8'
        )
    )
    assert qc['static_component'] == 'datum_composition'
    assert qc['datum_mode'] == 'floating_and_flat'
    assert qc['floating_datum_mode'] == 'constant'
    assert qc['flat_datum_elevation_m'] == 200.0

    with (tmp_path / REFRACTION_DATUM_TRACE_PREVIEW_CSV_NAME).open(
        encoding='utf-8',
        newline='',
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert 'floating_datum_elevation_shift_ms' in rows[0]
    assert 'flat_datum_shift_ms' in rows[0]
    assert 'refraction_trace_shift_ms' in rows[0]


def test_row_endpoint_key_fallback_maps_original_trace_ids_to_sorted_positions() -> None:
    replacement = _replacement_result(
        source_endpoint_key_sorted=np.asarray(['s2', 's0', 's1'], dtype=object),
        receiver_endpoint_key_sorted=np.asarray(['r2', 'r0', 'r1'], dtype=object),
        source_node_id_sorted=np.asarray([2, 0, 1], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([2, 0, 1], dtype=np.int64),
    )

    result = build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=_datum(),
        apply_options=_apply_options(),
    )

    np.testing.assert_array_equal(result.sorted_trace_index, [2, 0, 1])
    np.testing.assert_array_equal(result.row_trace_index_sorted, [0, 1])
    np.testing.assert_array_equal(result.row_source_endpoint_key, ['s0', 's1'])
    np.testing.assert_array_equal(result.row_receiver_endpoint_key, ['r0', 'r1'])


def test_build_preserves_external_upstream_replacement_statuses() -> None:
    replacement = replace(
        _replacement_result(),
        source_static_status=np.asarray(
            ['exceeds_max_abs_shift', 'ok', 'ok'],
            dtype='<U32',
        ),
        receiver_static_status=np.asarray(
            ['ok', 'invalid_shift', 'ok'],
            dtype='<U32',
        ),
        trace_static_status_sorted=np.asarray(
            ['exceeds_max_abs_shift', 'ok', 'ok'],
            dtype='<U32',
        ),
        source_static_status_sorted=np.asarray(
            ['ok', 'invalid_shift', 'ok'],
            dtype='<U32',
        ),
        receiver_static_status_sorted=np.asarray(['ok', 'ok', 'ok'], dtype='<U32'),
        trace_static_valid_mask_sorted=np.asarray([False, True, True], dtype=bool),
    )

    result = build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=_datum(),
        apply_options=_apply_options(),
    )

    assert result.source_datum_status[0] == 'exceeds_max_abs_shift'
    assert result.receiver_datum_status[1] == 'invalid_weathering_replacement'
    assert np.isnan(result.source_refraction_shift_s[0])
    assert np.isnan(result.receiver_refraction_shift_s[1])
    assert result.trace_static_status_sorted[0] == 'exceeds_max_abs_shift'
    assert result.trace_static_valid_mask_sorted[0] == np.False_
    assert np.isnan(result.refraction_trace_shift_s_sorted[0])
    assert result.trace_static_status_sorted[1] == 'invalid_weathering_replacement'
    assert result.trace_static_valid_mask_sorted[1] == np.False_
    assert np.isnan(result.refraction_trace_shift_s_sorted[1])

    trace_field = SimpleNamespace(
        source_field_shift_s_sorted=np.asarray([0.001, 0.002, 0.003]),
        receiver_field_shift_s_sorted=np.asarray([0.004, 0.005, 0.006]),
        trace_field_shift_s_sorted=np.asarray([0.005, 0.007, 0.009]),
        trace_field_static_status_sorted=np.asarray(['ok', 'ok', 'ok']),
        qc={},
    )
    with_field = build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=_datum(),
        apply_options=_apply_options(),
        field_corrections=datum_module.RefractionDatumFieldCorrectionInputs(
            trace_field_correction=trace_field,  # type: ignore[arg-type]
            apply_to_trace_shift=True,
        ),
    )

    assert with_field.final_trace_static_status_sorted[1] == (
        'invalid_weathering_replacement'
    )
    assert with_field.final_trace_static_valid_mask_sorted[1] == np.False_
    assert np.isnan(with_field.final_trace_shift_s_sorted[1])
    assert with_field.applied_field_shift_s_sorted[1] == 0.0


def test_build_preserves_upstream_trace_status_without_endpoint_error() -> None:
    replacement = replace(
        _replacement_result(valid_observation=[True, True, True]),
        trace_static_status_sorted=np.asarray(['ok', 'invalid_shift', 'ok'], dtype='<U32'),
        trace_static_valid_mask_sorted=np.asarray([True, False, True], dtype=bool),
    )

    result = build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=_datum(),
        apply_options=_apply_options(),
    )

    np.testing.assert_array_equal(
        result.source_datum_status,
        ['ok', 'ok', 'ok'],
    )
    np.testing.assert_array_equal(
        result.receiver_datum_status,
        ['ok', 'ok', 'ok'],
    )
    assert result.trace_static_status_sorted[1] == 'invalid_weathering_replacement'
    assert result.trace_static_valid_mask_sorted[1] == np.False_
    assert np.isnan(result.refraction_trace_shift_s_sorted[1])


@pytest.mark.parametrize(
    ('mode', 'expected_floating_active', 'expected_flat_active'),
    [
        ('none', False, False),
        ('floating_only', True, False),
        ('flat_only', False, True),
        ('floating_and_flat', True, True),
    ],
)
def test_datum_modes_control_component_composition(
    mode: str,
    expected_floating_active: bool,
    expected_flat_active: bool,
) -> None:
    replacement = _replacement_result()
    result = build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=_datum(mode=mode),
        apply_options=_apply_options(),
    )

    expected = replacement.weathering_replacement_trace_shift_s_sorted.copy()
    if expected_floating_active:
        expected += result.floating_datum_elevation_shift_s_sorted
    else:
        np.testing.assert_allclose(result.floating_datum_elevation_shift_s_sorted, 0.0)
    if expected_flat_active:
        expected += result.flat_datum_shift_s_sorted
    else:
        np.testing.assert_allclose(result.flat_datum_shift_s_sorted, 0.0)
    expected[~replacement.valid_observation_mask_sorted] = np.nan
    np.testing.assert_allclose(result.refraction_trace_shift_s_sorted, expected)


def test_surface_floating_datum_mode_produces_zero_floating_shift() -> None:
    result = build_refraction_datum_statics(
        weathering_replacement_result=_replacement_result(),
        datum=_datum(mode='floating_only', floating_datum_mode='surface'),
        apply_options=_apply_options(),
    )

    np.testing.assert_allclose(result.floating_datum_elevation_shift_s_sorted, 0.0)
    expected = result.weathering_replacement_trace_shift_s_sorted.copy()
    expected[~result.valid_observation_mask_sorted] = np.nan
    np.testing.assert_allclose(
        result.refraction_trace_shift_s_sorted,
        expected,
    )


def test_smoothed_topography_is_deterministic_and_node_based() -> None:
    replacement = replace(
        _replacement_result(),
        node_surface_elevation_m=np.asarray([100.0, 130.0, 160.0]),
    )

    result = build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=_datum(
            mode='floating_only',
            floating_datum_mode='smoothed_topography',
            smoothing_window_nodes=3,
            smoothing_method='moving_average',
        ),
        apply_options=_apply_options(),
    )

    np.testing.assert_allclose(
        result.node_floating_datum_elevation_m,
        np.asarray([115.0, 130.0, 145.0]),
    )
    np.testing.assert_allclose(
        result.source_floating_datum_elevation_m,
        result.receiver_floating_datum_elevation_m,
    )


def test_from_artifact_mode_loads_node_npz_from_static_job(tmp_path: Path) -> None:
    state = AppState()
    job_dir = tmp_path / 'floating-datum-job'
    job_dir.mkdir()
    artifact_name = 'floating_datum_elevation_m_by_node.npz'
    artifact_path = job_dir / artifact_name
    np.savez(
        artifact_path,
        node_id=np.asarray([0, 1, 2], dtype=np.int64),
        floating_datum_elevation_m=np.asarray([181.0, 191.0, 201.0]),
    )
    state.jobs.create_static_job(
        'floating-datum-job',
        file_id='line-a',
        key1_byte=189,
        key2_byte=193,
        statics_kind='refraction',
        artifacts_dir=str(job_dir),
    )

    result = build_refraction_datum_statics(
        weathering_replacement_result=_replacement_result(),
        datum=_datum(
            mode='floating_only',
            floating_datum_mode='from_artifact',
            floating_datum_job_id='floating-datum-job',
            floating_datum_artifact_name=artifact_name,
        ),
        apply_options=_apply_options(),
        runtime=SeisViewer2DRefractionRuntime(state),
        file_id='line-a',
        key1_byte=189,
        key2_byte=193,
    )

    np.testing.assert_allclose(
        result.node_floating_datum_elevation_m,
        [181.0, 191.0, 201.0],
    )
    np.testing.assert_allclose(
        result.source_floating_datum_elevation_m_sorted,
        [181.0, 191.0, 201.0],
    )
    expected_floating = -(
        (
            result.source_surface_elevation_m_sorted
            - result.source_floating_datum_elevation_m_sorted
        )
        + (
            result.receiver_surface_elevation_m_sorted
            - result.receiver_floating_datum_elevation_m_sorted
        )
    ) / BEDROCK_VELOCITY_M_S
    np.testing.assert_allclose(
        result.floating_datum_elevation_shift_s_sorted,
        expected_floating,
    )
    expected_refraction = (
        result.weathering_replacement_trace_shift_s_sorted + expected_floating
    )
    expected_refraction[~result.valid_observation_mask_sorted] = np.nan
    np.testing.assert_allclose(
        result.refraction_trace_shift_s_sorted,
        expected_refraction,
    )


def test_from_artifact_mode_loads_endpoint_npz_from_explicit_path(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / 'floating_datum_elevation_m_by_endpoint.npz'
    np.savez(
        artifact_path,
        source_endpoint_key=np.asarray(['s0', 's1', 's2']),
        source_floating_datum_elevation_m=np.asarray([181.0, 191.0, 201.0]),
        receiver_endpoint_key=np.asarray(['r0', 'r1', 'r2']),
        receiver_floating_datum_elevation_m=np.asarray([182.0, 192.0, 202.0]),
    )

    result = build_refraction_datum_statics(
        weathering_replacement_result=_replacement_result(),
        datum=_datum(
            mode='floating_only',
            floating_datum_mode='from_artifact',
            floating_datum_job_id='floating-datum-job',
            floating_datum_artifact_name=artifact_path.name,
        ),
        apply_options=_apply_options(),
        floating_datum_artifact_path=artifact_path,
    )

    np.testing.assert_allclose(
        result.source_floating_datum_elevation_m,
        [181.0, 191.0, 201.0],
    )
    np.testing.assert_allclose(
        result.receiver_floating_datum_elevation_m,
        [182.0, 192.0, 202.0],
    )
    np.testing.assert_allclose(
        result.node_floating_datum_elevation_m,
        [181.5, 191.5, 201.5],
    )


def test_from_artifact_mode_preserves_endpoint_values_for_shared_nodes(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / 'floating_datum_elevation_m_by_endpoint.npz'
    np.savez(
        artifact_path,
        source_endpoint_key=np.asarray(['s0', 's1', 's2']),
        source_floating_datum_elevation_m=np.asarray([135.0, 145.0, 140.0]),
        receiver_endpoint_key=np.asarray(['r0', 'r1', 'r2']),
        receiver_floating_datum_elevation_m=np.asarray([136.0, 146.0, 141.0]),
    )

    result = build_refraction_datum_statics(
        weathering_replacement_result=_replacement_result(
            source_surface=np.asarray([140.0, 150.0, 145.0]),
            receiver_surface=np.asarray([145.0, 155.0, 150.0]),
            source_node_id=np.asarray([0, 0, 1]),
            receiver_node_id=np.asarray([0, 0, 1]),
            source_node_id_sorted=np.asarray([0, 0, 1]),
            receiver_node_id_sorted=np.asarray([0, 0, 1]),
            source_endpoint_key_sorted=np.asarray(['s1', 's0', 's2']),
            receiver_endpoint_key_sorted=np.asarray(['r1', 'r0', 'r2']),
        ),
        datum=_datum(
            mode='floating_only',
            floating_datum_mode='from_artifact',
            floating_datum_job_id='floating-datum-job',
            floating_datum_artifact_name=artifact_path.name,
        ),
        apply_options=_apply_options(),
        floating_datum_artifact_path=artifact_path,
    )

    np.testing.assert_allclose(
        result.source_surface_elevation_m_sorted,
        [150.0, 140.0, 145.0],
    )
    np.testing.assert_allclose(
        result.receiver_surface_elevation_m_sorted,
        [155.0, 145.0, 150.0],
    )
    np.testing.assert_allclose(
        result.source_floating_datum_elevation_m_sorted,
        [145.0, 135.0, 140.0],
    )
    np.testing.assert_allclose(
        result.receiver_floating_datum_elevation_m_sorted,
        [146.0, 136.0, 141.0],
    )
    expected_floating = -(
        (
            result.source_surface_elevation_m_sorted
            - result.source_floating_datum_elevation_m_sorted
        )
        + (
            result.receiver_surface_elevation_m_sorted
            - result.receiver_floating_datum_elevation_m_sorted
        )
    ) / BEDROCK_VELOCITY_M_S
    np.testing.assert_allclose(
        result.floating_datum_elevation_shift_s_sorted,
        expected_floating,
    )


def test_from_artifact_mode_requires_resolvable_artifact() -> None:
    with pytest.raises(RefractionDatumStaticsError, match='floating datum artifact'):
        build_refraction_datum_statics(
            weathering_replacement_result=_replacement_result(),
            datum=_datum(
                mode='floating_only',
                floating_datum_mode='from_artifact',
                floating_datum_job_id='datum-job-id',
                floating_datum_artifact_name='floating_datum.csv',
            ),
            apply_options=_apply_options(),
        )


def test_physical_checks_mark_invalid_without_clipping() -> None:
    result = build_refraction_datum_statics(
        weathering_replacement_result=_replacement_result(),
        datum=_datum(
            mode='floating_only',
            floating_datum_mode='constant',
            floating_datum_elevation_m=80.0,
        ),
        apply_options=_apply_options(),
    )

    assert result.trace_static_status_sorted[0] == 'floating_datum_below_refractor'
    assert result.trace_static_valid_mask_sorted[0] == np.False_
    assert np.isnan(result.refraction_trace_shift_s_sorted[0])
    assert result.qc['floating_datum_below_refractor_count'] == 3


def test_flat_datum_below_refractor_policy_is_explicit() -> None:
    rejected = build_refraction_datum_statics(
        weathering_replacement_result=_replacement_result(),
        datum=_datum(
            mode='flat_only',
            floating_datum_mode='constant',
            floating_datum_elevation_m=100.0,
            flat_datum_elevation_m=80.0,
            allow_flat_datum_below_refractor=False,
        ),
        apply_options=_apply_options(),
    )
    allowed = build_refraction_datum_statics(
        weathering_replacement_result=_replacement_result(),
        datum=_datum(
            mode='flat_only',
            floating_datum_mode='constant',
            floating_datum_elevation_m=100.0,
            flat_datum_elevation_m=80.0,
            allow_flat_datum_below_refractor=True,
        ),
        apply_options=_apply_options(),
    )

    assert rejected.trace_static_status_sorted[0] == 'flat_datum_below_refractor'
    assert rejected.trace_static_valid_mask_sorted[0] == np.False_
    assert allowed.trace_static_valid_mask_sorted[0] == np.True_


def test_flat_datum_above_topography_policy_is_explicit() -> None:
    rejected = build_refraction_datum_statics(
        weathering_replacement_result=_replacement_result(),
        datum=_datum(
            mode='flat_only',
            floating_datum_mode='constant',
            floating_datum_elevation_m=100.0,
            flat_datum_elevation_m=220.0,
            allow_flat_datum_above_topography=False,
        ),
        apply_options=_apply_options(),
    )
    allowed = build_refraction_datum_statics(
        weathering_replacement_result=_replacement_result(),
        datum=_datum(
            mode='flat_only',
            floating_datum_mode='constant',
            floating_datum_elevation_m=100.0,
            flat_datum_elevation_m=220.0,
            allow_flat_datum_above_topography=True,
        ),
        apply_options=_apply_options(),
    )

    assert rejected.trace_static_status_sorted[0] == 'invalid_flat_datum_elevation'
    assert rejected.trace_static_valid_mask_sorted[0] == np.False_
    assert np.isnan(rejected.refraction_trace_shift_s_sorted[0])
    assert allowed.trace_static_valid_mask_sorted[0] == np.True_


def test_max_shift_status_and_shift_come_from_external_core() -> None:
    result = build_refraction_datum_statics(
        weathering_replacement_result=_replacement_result(),
        datum=_datum(flat_datum_elevation_m=1000.0),
        apply_options=_apply_options(max_abs_shift_ms=10.0),
    )

    assert np.isnan(result.refraction_trace_shift_s_sorted).all()
    assert np.isnan(result.source_refraction_shift_s).all()
    assert np.isnan(result.receiver_refraction_shift_s).all()
    np.testing.assert_array_equal(
        result.trace_static_status_sorted,
        ['invalid_datum_shift', 'invalid_datum_shift', 'invalid_datum_shift'],
    )
    np.testing.assert_array_equal(
        result.source_datum_status,
        ['invalid_datum_shift', 'invalid_datum_shift', 'invalid_datum_shift'],
    )
    np.testing.assert_array_equal(
        result.receiver_datum_status,
        ['invalid_datum_shift', 'invalid_datum_shift', 'invalid_datum_shift'],
    )
    np.testing.assert_array_equal(
        result.node_datum_status,
        ['invalid_datum_shift', 'invalid_datum_shift', 'invalid_datum_shift'],
    )
    np.testing.assert_array_equal(
        result.trace_static_valid_mask_sorted,
        [False, False, False],
    )
    assert result.qc['exceeds_max_abs_shift_count'] == 0
    assert result.qc['trace_static_status_counts'] == {'invalid_datum_shift': 3}


def test_invalid_surface_and_unknown_endpoint_nodes_are_visible() -> None:
    invalid_surface = build_refraction_datum_statics(
        weathering_replacement_result=_replacement_result(
            source_surface=np.asarray([np.nan, 110.0, 120.0]),
        ),
        datum=_datum(),
        apply_options=_apply_options(),
    )
    assert invalid_surface.trace_static_status_sorted[0] == 'invalid_surface_elevation'
    assert np.isnan(invalid_surface.refraction_trace_shift_s_sorted[0])

    with pytest.raises(RefractionDatumStaticsError, match='source_node_id'):
        build_refraction_datum_statics(
            weathering_replacement_result=_replacement_result(
                source_node_id=np.asarray([999, 1, 2]),
            ),
            datum=_datum(),
            apply_options=_apply_options(),
        )


def test_datum_statics_rejects_non_real_numeric_node_ids() -> None:
    replacement = replace(
        _replacement_result(),
        source_node_id=np.asarray(['0', '1', '2'], dtype='<U1'),
    )

    with pytest.raises(
        RefractionDatumStaticsError,
        match='source_node_id.*real numeric dtype',
    ):
        build_refraction_datum_statics(
            weathering_replacement_result=replacement,
            datum=_datum(),
            apply_options=_apply_options(),
        )


def test_build_maps_external_core_result_without_recomputing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement = _replacement_result(valid_observation=[True, True, True])
    captured_core_kwargs: dict[str, Any] = {}
    captured_node_kwargs: dict[str, Any] = {}
    source_core = SimpleNamespace(
        floating_datum_shift_s=np.asarray([0.101, 0.102, 0.103]),
        flat_datum_shift_s=np.asarray([0.201, 0.202, 0.203]),
        datum_static_status=np.asarray(
            ['exceeds_max_abs_shift', 'ok', 'ok'],
            dtype='<U32',
        ),
        total_refraction_shift_s=np.asarray([1.001, 1.002, 1.003]),
    )
    receiver_core = SimpleNamespace(
        floating_datum_shift_s=np.asarray([0.301, 0.302, 0.303]),
        flat_datum_shift_s=np.asarray([0.401, 0.402, 0.403]),
        datum_static_status=np.asarray(
            ['ok', 'exceeds_max_abs_shift', 'ok'],
            dtype='<U32',
        ),
        total_refraction_shift_s=np.asarray([2.001, 2.002, 2.003]),
    )
    core_result = SimpleNamespace(
        source_endpoint_datum=source_core,
        receiver_endpoint_datum=receiver_core,
        source_floating_datum_shift_s_sorted=np.asarray([10.0, 11.0, 12.0]),
        receiver_floating_datum_shift_s_sorted=np.asarray([20.0, 21.0, 22.0]),
        source_flat_datum_shift_s_sorted=np.asarray([30.0, 31.0, 32.0]),
        receiver_flat_datum_shift_s_sorted=np.asarray([40.0, 41.0, 42.0]),
        source_refraction_shift_s_sorted=np.asarray([50.0, 51.0, 52.0]),
        receiver_refraction_shift_s_sorted=np.asarray([60.0, 61.0, 62.0]),
        refraction_trace_shift_s_sorted=np.asarray([70.0, 71.0, 72.0]),
        trace_static_status_sorted=np.asarray(['ok', 'external_status', 'ok']),
        trace_static_valid_mask_sorted=np.asarray([True, False, True]),
        applied_field_shift_s_sorted=np.asarray([0.5, 0.0, 0.7]),
        final_trace_shift_s_sorted=np.asarray([70.5, 71.0, 72.7]),
        final_trace_static_status_sorted=np.asarray(['ok', 'external_status', 'ok']),
        final_trace_static_valid_mask_sorted=np.asarray([True, False, True]),
        qc={'core_final': True},
    )
    trace_field = SimpleNamespace(
        source_field_shift_s_sorted=np.asarray([0.1, 0.2, 0.3]),
        receiver_field_shift_s_sorted=np.asarray([0.4, 0.5, 0.6]),
        trace_field_shift_s_sorted=np.asarray([0.5, 0.7, 0.9]),
        trace_field_static_status_sorted=np.asarray(['ok', 'ok', 'ok']),
        qc={'trace_field': True},
    )

    monkeypatch.setattr(
        datum_module,
        'core_build_refraction_datum_statics',
        lambda **kwargs: captured_core_kwargs.setdefault('kwargs', kwargs)
        and core_result,
    )
    monkeypatch.setattr(
        datum_module,
        'core_build_refraction_endpoint_datum_statics',
        lambda **kwargs: captured_node_kwargs.setdefault('kwargs', kwargs)
        and SimpleNamespace(
            datum_static_status=np.asarray(
                ['node_external', 'node_external', 'node_external'],
                dtype='<U32',
            ),
            total_refraction_shift_s=np.asarray([3.001, 3.002, 3.003]),
        ),
    )

    result = build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=_datum(),
        apply_options=_apply_options(max_abs_shift_ms=1.0),
        field_corrections=datum_module.RefractionDatumFieldCorrectionInputs(
            trace_field_correction=trace_field,  # type: ignore[arg-type]
            apply_to_trace_shift=True,
            invalid_component_policy='skip_invalid_traces',
            field_composition_qc={'trace_field': trace_field.qc},
        ),
    )

    np.testing.assert_allclose(result.source_refraction_shift_s, [1.001, 1.002, 1.003])
    np.testing.assert_allclose(
        result.receiver_refraction_shift_s,
        [2.001, 2.002, 2.003],
    )
    np.testing.assert_allclose(result.source_refraction_shift_s_sorted, [50.0, 51.0, 52.0])
    np.testing.assert_allclose(
        result.receiver_refraction_shift_s_sorted,
        [60.0, 61.0, 62.0],
    )
    np.testing.assert_allclose(result.refraction_trace_shift_s_sorted, [70.0, 71.0, 72.0])
    np.testing.assert_array_equal(
        result.trace_static_status_sorted,
        ['ok', 'external_status', 'ok'],
    )
    np.testing.assert_array_equal(
        result.source_datum_status,
        ['exceeds_max_abs_shift', 'ok', 'ok'],
    )
    np.testing.assert_array_equal(
        result.receiver_datum_status,
        ['ok', 'exceeds_max_abs_shift', 'ok'],
    )
    np.testing.assert_array_equal(
        result.trace_static_valid_mask_sorted,
        [True, False, True],
    )
    np.testing.assert_array_equal(
        result.node_datum_status,
        ['node_external', 'node_external', 'node_external'],
    )
    np.testing.assert_array_equal(
        captured_core_kwargs['kwargs']['source_weathering_replacement_status'],
        replacement.source_static_status,
    )
    assert captured_core_kwargs['kwargs']['max_abs_datum_shift_ms'] == 1.0
    assert captured_core_kwargs['kwargs']['trace_field_correction'] is trace_field
    assert captured_core_kwargs['kwargs']['apply_field_correction_to_trace_shift'] is True
    assert (
        captured_core_kwargs['kwargs']['invalid_field_component_policy']
        == 'skip_invalid_traces'
    )
    assert captured_node_kwargs['kwargs']['max_abs_datum_shift_ms'] == 1.0


def test_high_level_pipeline_calls_weathering_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    replacement = _replacement_result()
    calls: list[dict[str, Any]] = []

    def _capture_replacement(**kwargs: Any) -> RefractionWeatheringReplacementStaticsResult:
        calls.append(kwargs)
        return replacement

    monkeypatch.setattr(
        datum_module,
        'compute_weathering_replacement_statics_from_first_breaks',
        _capture_replacement,
    )
    req = SimpleNamespace(
        file_id='line-a',
        key1_byte=189,
        key2_byte=193,
        datum=_datum(),
        apply=_apply_options(),
    )
    state = object()

    result = compute_datum_refraction_statics_from_first_breaks(
        req=req,  # type: ignore[arg-type]
        state=state,  # type: ignore[arg-type]
        job_dir=tmp_path,
    )

    assert calls == [{'req': req, 'state': state, 'job_dir': tmp_path}]
    assert result.qc['static_component'] == 'datum_composition'
    assert (tmp_path / REFRACTION_DATUM_STATICS_QC_JSON_NAME).is_file()
