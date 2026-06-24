from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.statics.refraction.application.cell_v2_metadata import (
    _CELL_ID_PROBE_V2_BASE_M_S,
    cell_v2_metadata_from_core_weathering,
)
import app.statics.refraction.application.weathering as weathering_module
from app.statics.refraction.application.weathering import (
    REFRACTION_WEATHERING_NODES_CSV_NAME,
    REFRACTION_WEATHERING_QC_JSON_NAME,
    REFRACTION_WEATHERING_RECEIVERS_CSV_NAME,
    REFRACTION_WEATHERING_SOURCES_CSV_NAME,
    REFRACTION_WEATHERING_TRACE_PREVIEW_CSV_NAME,
    RefractionWeatheringThicknessError,
    build_refraction_weathering_core_context,
    build_refraction_weathering_thickness_model,
    compute_weathering_thickness_from_half_intercept_time,
    compute_weathering_thickness_scalar,
    estimate_weathering_thickness_from_first_breaks,
)
from app.tests.test_refraction_static_half_intercept import (
    NODE_ID,
    TRUE_BEDROCK_SLOWNESS_S_PER_M,
    TRUE_BEDROCK_VELOCITY_M_S,
    TRUE_HALF_INTERCEPT_S,
    WEATHERING_VELOCITY_M_S,
    _build_result,
    _input_model,
    _model,
)


def _conversion_factor() -> float:
    vb = TRUE_BEDROCK_VELOCITY_M_S
    vw = WEATHERING_VELOCITY_M_S
    return vb * vw / np.sqrt(vb * vb - vw * vw)


def _with_sorted_trace_permutation(half, sorted_to_original: np.ndarray):
    def _sorted(values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(values)[sorted_to_original])

    return replace(
        half,
        sorted_trace_index=np.ascontiguousarray(sorted_to_original, dtype=np.int64),
        source_endpoint_key_sorted=_sorted(half.source_endpoint_key_sorted),
        receiver_endpoint_key_sorted=_sorted(half.receiver_endpoint_key_sorted),
        source_elevation_m_sorted=_sorted(half.source_elevation_m_sorted),
        receiver_elevation_m_sorted=_sorted(half.receiver_elevation_m_sorted),
        source_node_id_sorted=_sorted(half.source_node_id_sorted),
        receiver_node_id_sorted=_sorted(half.receiver_node_id_sorted),
        source_half_intercept_time_s_sorted=_sorted(
            half.source_half_intercept_time_s_sorted
        ),
        receiver_half_intercept_time_s_sorted=_sorted(
            half.receiver_half_intercept_time_s_sorted
        ),
        estimated_intercept_time_sum_s_sorted=_sorted(
            half.estimated_intercept_time_sum_s_sorted
        ),
        estimated_bedrock_moveout_time_s_sorted=_sorted(
            half.estimated_bedrock_moveout_time_s_sorted
        ),
        estimated_first_break_time_s_sorted=_sorted(
            half.estimated_first_break_time_s_sorted
        ),
        first_break_residual_s_sorted=_sorted(half.first_break_residual_s_sorted),
        valid_observation_mask_sorted=_sorted(half.valid_observation_mask_sorted),
        used_observation_mask_sorted=_sorted(half.used_observation_mask_sorted),
    )


def test_public_apis_are_importable() -> None:
    assert callable(estimate_weathering_thickness_from_first_breaks)
    assert callable(build_refraction_weathering_thickness_model)
    assert callable(compute_weathering_thickness_from_half_intercept_time)
    assert callable(compute_weathering_thickness_scalar)


def test_high_level_pipeline_calls_half_intercept_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs, _design, _solved, half = _build_result()
    calls: list[dict[str, object]] = []
    core_calls: list[dict[str, object]] = []
    model = _model()
    core_context = SimpleNamespace(
        app_result=half,
        core_input_model=weathering_module.core_input_model_from_app(inputs),
        core_result=weathering_module._core_half_intercept_result_from_app_result(
            half_intercept_result=half,
            model=model,
        ),
    )
    core_build = weathering_module.core_build_refraction_weathering_model_from_half_intercept_result

    def _capture_estimate(**kwargs: object):
        calls.append(kwargs)
        return core_context

    def _capture_core_build(**kwargs: object):
        core_calls.append(kwargs)
        return core_build(**kwargs)

    monkeypatch.setattr(
        weathering_module,
        'estimate_refraction_half_intercept_core_context_from_first_breaks',
        _capture_estimate,
    )
    monkeypatch.setattr(
        weathering_module,
        'core_build_refraction_weathering_model_from_half_intercept_result',
        _capture_core_build,
    )
    req = SimpleNamespace(model=model)
    state = object()

    result = estimate_weathering_thickness_from_first_breaks(
        req=req,  # type: ignore[arg-type]
        state=state,  # type: ignore[arg-type]
        job_dir=tmp_path,
    )

    assert calls == [
        {
            'req': req,
            'runtime': None,
            'state': state,
            'job_dir': tmp_path,
            'input_model': None,
            'resolved_first_layer': None,
        }
    ]
    assert core_calls
    assert core_calls[0]['half_intercept_result'] is core_context.core_result
    expected_node_thickness = TRUE_HALF_INTERCEPT_S * _conversion_factor()
    np.testing.assert_allclose(
        result.node_weathering_thickness_m[:5],
        expected_node_thickness,
        rtol=1.0e-9,
    )
    assert (tmp_path / REFRACTION_WEATHERING_QC_JSON_NAME).is_file()


def test_cell_v2_metadata_uses_endpoint_local_projection() -> None:
    model = SimpleNamespace(bedrock_velocity_mode='solve_cell', refractor_cell=object())
    core_weathering = SimpleNamespace(
        node_id=np.asarray([10, 20], dtype=np.int64),
        node_v2_m_s=np.asarray([2200.0, 2600.0], dtype=np.float64),
        node_local_v2_status=np.asarray(['ok', 'ok'], dtype='<U32'),
        cell_id=np.asarray([0, 1, 2], dtype=np.int64),
        cell_v2_m_s=np.asarray([2200.0, 2600.0, 3000.0], dtype=np.float64),
        cell_velocity_status=np.asarray(['solved', 'solved', 'inactive'], dtype='<U32'),
        source_endpoint=SimpleNamespace(
            endpoint_key=np.asarray(['s0', 's1'], dtype=object),
            node_id=np.asarray([10, 10], dtype=np.int64),
            v2_m_s=np.asarray([3000.0, 2600.0], dtype=np.float64),
            local_v2_status=np.asarray(['ok', 'ok'], dtype='<U32'),
        ),
        receiver_endpoint=SimpleNamespace(
            endpoint_key=np.asarray(['r0', 'r1'], dtype=object),
            node_id=np.asarray([20, 20], dtype=np.int64),
            v2_m_s=np.asarray([2600.0, np.nan], dtype=np.float64),
            local_v2_status=np.asarray(['ok', 'inactive_v2_cell'], dtype='<U32'),
        ),
        source_endpoint_key_sorted=np.asarray(['s1', 's0', 's0'], dtype=object),
        receiver_endpoint_key_sorted=np.asarray(['r0', 'r1', 'missing'], dtype=object),
    )
    probe = SimpleNamespace(
        node_v2_m_s=_CELL_ID_PROBE_V2_BASE_M_S
        + np.asarray([0.0, 1.0], dtype=np.float64),
        source_endpoint=SimpleNamespace(
            v2_m_s=_CELL_ID_PROBE_V2_BASE_M_S
            + np.asarray([2.0, 1.0], dtype=np.float64),
        ),
        receiver_endpoint=SimpleNamespace(
            v2_m_s=_CELL_ID_PROBE_V2_BASE_M_S
            + np.asarray([1.0, 2.0], dtype=np.float64),
        ),
    )

    metadata = cell_v2_metadata_from_core_weathering(
        core_weathering=core_weathering,
        model=model,  # type: ignore[arg-type]
        cell_id_probe=probe,
    )

    np.testing.assert_array_equal(metadata.node_cell_id, [0, 1])
    np.testing.assert_array_equal(metadata.source_cell_id, [2, 1])
    np.testing.assert_array_equal(metadata.receiver_cell_id, [1, 2])
    np.testing.assert_array_equal(metadata.source_cell_id_sorted, [1, 2, 2])
    np.testing.assert_allclose(metadata.source_v2_m_s_sorted, [2600.0, 3000.0, 3000.0])
    np.testing.assert_array_equal(
        metadata.receiver_cell_id_sorted,
        [1, 2, -1],
    )
    np.testing.assert_array_equal(
        metadata.receiver_v2_status_sorted,
        ['ok', 'inactive_v2_cell', 'missing_endpoint'],
    )


def test_build_weathering_rejects_model_velocity_mismatch() -> None:
    _inputs, _design, _solved, half = _build_result()
    model = _model(weathering_velocity_m_s=WEATHERING_VELOCITY_M_S + 100.0)

    with pytest.raises(
        RefractionWeatheringThicknessError,
        match=(
            'model.weathering_velocity_m_s does not match '
            'half_intercept_result.weathering_velocity_m_s'
        ),
    ):
        build_refraction_weathering_thickness_model(
            half_intercept_result=half,
            model=model,
        )


def test_math_helper_converts_half_intercepts_and_preserves_nan() -> None:
    half = np.asarray([0.010, np.nan, 0.020], dtype=np.float64)

    thickness = compute_weathering_thickness_from_half_intercept_time(
        half_intercept_time_s=half,
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        bedrock_velocity_m_s=TRUE_BEDROCK_VELOCITY_M_S,
    )

    np.testing.assert_allclose(
        thickness[[0, 2]],
        half[[0, 2]] * _conversion_factor(),
        rtol=1.0e-12,
    )
    assert np.isnan(thickness[1])
    assert compute_weathering_thickness_scalar(
        half_intercept_time_s=0.010,
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        bedrock_velocity_m_s=TRUE_BEDROCK_VELOCITY_M_S,
    ) == pytest.approx(0.010 * _conversion_factor())


def test_math_helper_accepts_vector_bedrock_velocity() -> None:
    half = np.asarray([0.010, 0.012, 0.014], dtype=np.float64)
    bedrock_velocity = np.asarray([2200.0, 2500.0, 3000.0], dtype=np.float64)

    thickness = compute_weathering_thickness_from_half_intercept_time(
        half_intercept_time_s=half,
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        bedrock_velocity_m_s=bedrock_velocity,
    )

    expected = (
        half
        * bedrock_velocity
        * WEATHERING_VELOCITY_M_S
        / np.sqrt(bedrock_velocity * bedrock_velocity - WEATHERING_VELOCITY_M_S**2)
    )
    np.testing.assert_allclose(thickness, expected, rtol=1.0e-12)


def test_build_weathering_model_maps_nodes_endpoints_and_trace_order(
    tmp_path: Path,
) -> None:
    inputs, _design, _solved, half = _build_result()
    model = _model()

    result = build_refraction_weathering_thickness_model(
        half_intercept_result=half,
        model=model,
        job_dir=tmp_path,
    )

    expected_node_thickness = TRUE_HALF_INTERCEPT_S * _conversion_factor()
    np.testing.assert_allclose(
        result.node_weathering_thickness_m[:5],
        expected_node_thickness,
        rtol=1.0e-9,
    )
    assert np.isnan(result.node_weathering_thickness_m[5])
    np.testing.assert_allclose(
        result.node_refractor_elevation_m[:5],
        -expected_node_thickness,
        rtol=1.0e-9,
    )
    assert result.node_weathering_status[:5].tolist() == ['ok'] * 5
    assert result.node_weathering_status[5] == 'inactive'

    source_zero = int(np.flatnonzero(result.source_node_id == 0)[0])
    receiver_zero = int(np.flatnonzero(result.receiver_node_id == 0)[0])
    assert result.source_weathering_thickness_m[source_zero] == pytest.approx(
        expected_node_thickness[0]
    )
    assert result.receiver_weathering_thickness_m[receiver_zero] == pytest.approx(
        expected_node_thickness[0]
    )

    valid = result.valid_observation_mask_sorted
    np.testing.assert_allclose(
        result.source_weathering_thickness_m_sorted[valid],
        expected_node_thickness[result.source_node_id_sorted[valid]],
        rtol=1.0e-9,
    )
    np.testing.assert_allclose(
        result.receiver_weathering_thickness_m_sorted[valid],
        expected_node_thickness[result.receiver_node_id_sorted[valid]],
        rtol=1.0e-9,
    )
    assert result.source_weathering_status_sorted[-1] == 'inactive'
    assert result.receiver_weathering_status_sorted[-1] == 'inactive'
    assert result.qc['node_weathering_status_counts']['ok'] == int(
        NODE_ID.shape[0] - 1
    )
    assert result.qc['weathering_thickness_median_m'] == pytest.approx(
        np.median(expected_node_thickness)
    )

    artifact_names = (
        REFRACTION_WEATHERING_QC_JSON_NAME,
        REFRACTION_WEATHERING_NODES_CSV_NAME,
        REFRACTION_WEATHERING_SOURCES_CSV_NAME,
        REFRACTION_WEATHERING_RECEIVERS_CSV_NAME,
        REFRACTION_WEATHERING_TRACE_PREVIEW_CSV_NAME,
    )
    for name in artifact_names:
        assert (tmp_path / name).is_file()

    qc = json.loads(
        (tmp_path / REFRACTION_WEATHERING_QC_JSON_NAME).read_text(encoding='utf-8')
    )
    assert qc['n_nodes'] == int(NODE_ID.shape[0])
    assert qc['inactive_node_count'] == 1

    with (tmp_path / REFRACTION_WEATHERING_NODES_CSV_NAME).open(
        encoding='utf-8',
        newline='',
    ) as handle:
        node_rows = list(csv.DictReader(handle))
    assert 'weathering_thickness_m' in node_rows[0]
    assert node_rows[-1]['weathering_status'] == 'inactive'

    with (tmp_path / REFRACTION_WEATHERING_TRACE_PREVIEW_CSV_NAME).open(
        encoding='utf-8',
        newline='',
    ) as handle:
        trace_rows = list(csv.DictReader(handle))
    assert 'source_refractor_elevation_m' in trace_rows[0]
    assert trace_rows[-1]['valid_observation'] == 'False'


def test_shared_source_receiver_node_keeps_endpoint_surface_elevation() -> None:
    _inputs, _design, _solved, half = _build_result()
    source_elevation = half.source_elevation_m.copy()
    receiver_elevation = half.receiver_elevation_m.copy()
    source_elevation_sorted = half.source_elevation_m_sorted.copy()
    receiver_elevation_sorted = half.receiver_elevation_m_sorted.copy()
    source_zero = int(np.flatnonzero(half.source_node_id == 0)[0])
    receiver_zero = int(np.flatnonzero(half.receiver_node_id == 0)[0])
    source_elevation[source_zero] = 120.0
    receiver_elevation[receiver_zero] = 80.0
    source_elevation_sorted[half.source_node_id_sorted == 0] = 120.0
    receiver_elevation_sorted[half.receiver_node_id_sorted == 0] = 80.0

    result = build_refraction_weathering_thickness_model(
        half_intercept_result=replace(
            half,
            source_elevation_m=source_elevation,
            receiver_elevation_m=receiver_elevation,
            source_elevation_m_sorted=source_elevation_sorted,
            receiver_elevation_m_sorted=receiver_elevation_sorted,
        ),
        model=_model(),
    )

    expected_thickness = TRUE_HALF_INTERCEPT_S[0] * _conversion_factor()
    assert result.source_surface_elevation_m[source_zero] == pytest.approx(120.0)
    assert result.receiver_surface_elevation_m[receiver_zero] == pytest.approx(80.0)
    assert result.source_refractor_elevation_m[source_zero] == pytest.approx(
        120.0 - expected_thickness
    )
    assert result.receiver_refractor_elevation_m[receiver_zero] == pytest.approx(
        80.0 - expected_thickness
    )
    np.testing.assert_allclose(
        result.source_refractor_elevation_m_sorted[
            result.source_node_id_sorted == 0
        ],
        120.0 - expected_thickness,
    )
    np.testing.assert_allclose(
        result.receiver_refractor_elevation_m_sorted[
            result.receiver_node_id_sorted == 0
        ],
        80.0 - expected_thickness,
    )


def test_production_core_context_keeps_shared_node_endpoint_geometry() -> None:
    inputs, _design, _solved, half = _build_result()
    source_elevation = half.source_elevation_m.copy()
    receiver_elevation = half.receiver_elevation_m.copy()
    source_elevation_sorted = half.source_elevation_m_sorted.copy()
    receiver_elevation_sorted = half.receiver_elevation_m_sorted.copy()
    source_zero = int(np.flatnonzero(half.source_node_id == 0)[0])
    receiver_zero = int(np.flatnonzero(half.receiver_node_id == 0)[0])
    source_elevation[source_zero] = 100.0
    receiver_elevation[receiver_zero] = 120.0
    source_elevation_sorted[half.source_node_id_sorted == 0] = 100.0
    receiver_elevation_sorted[half.receiver_node_id_sorted == 0] = 120.0
    half = replace(
        half,
        source_elevation_m=source_elevation,
        receiver_elevation_m=receiver_elevation,
        source_elevation_m_sorted=source_elevation_sorted,
        receiver_elevation_m_sorted=receiver_elevation_sorted,
    )
    model = _model()
    core_result = weathering_module._core_half_intercept_result_from_app_result(
        half_intercept_result=half,
        model=model,
    )
    context = weathering_module._HalfInterceptCoreContext(
        app_input_model=inputs,
        core_input_model=weathering_module.core_input_model_from_app(inputs),
        core_result=core_result,
        app_result=half,
    )

    weathering_context = build_refraction_weathering_core_context(
        half_intercept_context=context,
        model=model,
    )

    result = weathering_context.app_weathering_result
    core = weathering_context.core_weathering_model
    expected_thickness = TRUE_HALF_INTERCEPT_S[0] * _conversion_factor()
    assert result.source_surface_elevation_m[source_zero] == pytest.approx(100.0)
    assert result.receiver_surface_elevation_m[receiver_zero] == pytest.approx(120.0)
    assert result.source_refractor_elevation_m[source_zero] == pytest.approx(
        100.0 - expected_thickness
    )
    assert result.receiver_refractor_elevation_m[receiver_zero] == pytest.approx(
        120.0 - expected_thickness
    )
    assert core.source_endpoint.surface_elevation_m[source_zero] == pytest.approx(100.0)
    assert core.receiver_endpoint.surface_elevation_m[receiver_zero] == pytest.approx(
        120.0
    )
    assert core.source_endpoint.refractor_elevation_m[source_zero] == pytest.approx(
        100.0 - expected_thickness
    )
    assert core.receiver_endpoint.refractor_elevation_m[
        receiver_zero
    ] == pytest.approx(120.0 - expected_thickness)
    assert (
        core.receiver_endpoint.refractor_elevation_m[receiver_zero]
        - core.source_endpoint.refractor_elevation_m[source_zero]
        == pytest.approx(20.0)
    )
    np.testing.assert_allclose(
        core.trace_weathering_thickness_m_sorted,
        core.source_weathering_thickness_m_sorted
        + core.receiver_weathering_thickness_m_sorted,
        equal_nan=True,
    )
    assert core.qc['trace_weathering_status_counts'] == {
        key: int(value)
        for key, value in zip(
            *np.unique(core.trace_weathering_status_sorted, return_counts=True),
            strict=True,
        )
    }


def test_solve_cell_probe_uses_side_specific_endpoint_geometry() -> None:
    inputs, _design, _solved, half = _build_result()
    source_x = half.source_x_m.copy()
    receiver_x = half.receiver_x_m.copy()
    source_x_sorted = inputs.source_x_m_sorted.copy()
    receiver_x_sorted = inputs.receiver_x_m_sorted.copy()
    source_zero = int(np.flatnonzero(half.source_node_id == 0)[0])
    receiver_zero = int(np.flatnonzero(half.receiver_node_id == 0)[0])
    source_x[source_zero] = 50.0
    receiver_x[receiver_zero] = 150.0
    source_x_sorted[half.source_node_id_sorted == 0] = 50.0
    receiver_x_sorted[half.receiver_node_id_sorted == 0] = 150.0
    cell_velocity = np.asarray([2200.0, 2600.0, 3000.0], dtype=np.float64)
    half = replace(
        half,
        bedrock_velocity_mode='solve_cell',
        source_x_m=source_x,
        receiver_x_m=receiver_x,
        active_cell_id=np.asarray([0, 1, 2], dtype=np.int64),
        cell_bedrock_velocity_m_s=cell_velocity,
        cell_bedrock_slowness_s_per_m=1.0 / cell_velocity,
        cell_velocity_status=np.full(3, 'solved', dtype='<U32'),
        row_midpoint_cell_id=np.zeros(half.row_trace_index_sorted.shape, dtype=np.int64),
        row_midpoint_bedrock_velocity_m_s=np.full(
            half.row_trace_index_sorted.shape,
            cell_velocity[0],
            dtype=np.float64,
        ),
        qc={
            **half.qc,
            'cell_observation_count': [10, 10, 10],
        },
    )
    inputs = replace(
        inputs,
        source_x_m_sorted=source_x_sorted,
        receiver_x_m_sorted=receiver_x_sorted,
    )
    model = _model(
        bedrock_velocity_mode='solve_cell',
        refractor_cell={
            'number_of_cell_x': 3,
            'size_of_cell_x_m': 100.0,
            'x_coordinate_origin_m': 0.0,
            'number_of_cell_y': 1,
            'size_of_cell_y_m': None,
            'y_coordinate_origin_m': 0.0,
            'min_observations_per_cell': 1,
        },
    )
    core_result = weathering_module._core_half_intercept_result_from_app_result(
        half_intercept_result=half,
        model=model,
    )
    context = weathering_module._HalfInterceptCoreContext(
        app_input_model=inputs,
        core_input_model=weathering_module.core_input_model_from_app(inputs),
        core_result=core_result,
        app_result=half,
    )

    result = build_refraction_weathering_core_context(
        half_intercept_context=context,
        model=model,
    ).app_weathering_result

    assert result.node_v2_cell_id is not None
    assert result.source_v2_cell_id is not None
    assert result.receiver_v2_cell_id is not None
    assert result.source_v2_cell_id_sorted is not None
    assert result.receiver_v2_cell_id_sorted is not None
    assert result.node_v2_cell_id[0] == 0
    assert result.source_v2_cell_id[source_zero] == 0
    assert result.receiver_v2_cell_id[receiver_zero] == 1
    assert result.source_v2_cell_id[source_zero] != result.receiver_v2_cell_id[
        receiver_zero
    ]
    np.testing.assert_array_equal(
        result.source_v2_cell_id_sorted[result.source_node_id_sorted == 0],
        0,
    )
    np.testing.assert_array_equal(
        result.receiver_v2_cell_id_sorted[result.receiver_node_id_sorted == 0],
        1,
    )


def test_public_builder_reconstructs_row_values_in_sorted_position_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _inputs, _design, _solved, half = _build_result()
    n_traces = int(half.sorted_trace_index.shape[0])
    sorted_to_original = np.arange(n_traces, dtype=np.int64)
    sorted_to_original[:3] = np.asarray([2, 0, 1], dtype=np.int64)
    half = _with_sorted_trace_permutation(half, sorted_to_original)
    captured_input_models = []
    original_core_build = (
        weathering_module.core_build_refraction_weathering_model_from_half_intercept_result
    )

    def _capture_core_build(**kwargs: object):
        captured_input_models.append(kwargs['input_model'])
        return original_core_build(**kwargs)

    monkeypatch.setattr(
        weathering_module,
        'core_build_refraction_weathering_model_from_half_intercept_result',
        _capture_core_build,
    )

    result = build_refraction_weathering_thickness_model(
        half_intercept_result=half,
        model=_model(),
    )

    inverse = np.empty(n_traces, dtype=np.int64)
    inverse[sorted_to_original] = np.arange(n_traces, dtype=np.int64)
    expected_pick = np.full(n_traces, np.nan, dtype=np.float64)
    expected_distance = np.full(n_traces, np.nan, dtype=np.float64)
    row_position = inverse[half.row_trace_index_sorted]
    expected_pick[row_position] = half.observed_pick_time_s
    expected_distance[row_position] = half.row_distance_m
    core_input = captured_input_models[0]
    np.testing.assert_allclose(
        core_input.pick_time_s_sorted,
        expected_pick,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        core_input.distance_m_sorted,
        expected_distance,
        equal_nan=True,
    )
    np.testing.assert_array_equal(result.sorted_trace_index, sorted_to_original)


def test_core_weathering_values_are_mapped_without_recalculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _inputs, _design, _solved, half = _build_result()
    model = _model()
    original_core_build = (
        weathering_module.core_build_refraction_weathering_model_from_half_intercept_result
    )

    def _patched_core_build(**kwargs: object):
        core = original_core_build(**kwargs)
        node_status = np.full(core.node_weathering_status.shape, 'external_node')
        source_status = np.full(
            core.source_endpoint.weathering_status.shape,
            'external_source',
        )
        receiver_status = np.full(
            core.receiver_endpoint.weathering_status.shape,
            'external_receiver',
        )
        source_sorted_status = np.full(
            core.source_weathering_status_sorted.shape,
            'external_source_sorted',
        )
        receiver_sorted_status = np.full(
            core.receiver_weathering_status_sorted.shape,
            'external_receiver_sorted',
        )
        return replace(
            core,
            node_weathering_thickness_m=np.full(
                core.node_weathering_thickness_m.shape,
                101.0,
            ),
            node_refractor_elevation_m=np.full(
                core.node_refractor_elevation_m.shape,
                202.0,
            ),
            node_weathering_status=node_status,
            source_endpoint=replace(
                core.source_endpoint,
                weathering_thickness_m=np.full(
                    core.source_endpoint.weathering_thickness_m.shape,
                    303.0,
                ),
                refractor_elevation_m=np.full(
                    core.source_endpoint.refractor_elevation_m.shape,
                    404.0,
                ),
                weathering_status=source_status,
            ),
            receiver_endpoint=replace(
                core.receiver_endpoint,
                weathering_thickness_m=np.full(
                    core.receiver_endpoint.weathering_thickness_m.shape,
                    505.0,
                ),
                refractor_elevation_m=np.full(
                    core.receiver_endpoint.refractor_elevation_m.shape,
                    606.0,
                ),
                weathering_status=receiver_status,
            ),
            source_weathering_thickness_m_sorted=np.full(
                core.source_weathering_thickness_m_sorted.shape,
                707.0,
            ),
            receiver_weathering_thickness_m_sorted=np.full(
                core.receiver_weathering_thickness_m_sorted.shape,
                808.0,
            ),
            source_refractor_elevation_m_sorted=np.full(
                core.source_refractor_elevation_m_sorted.shape,
                909.0,
            ),
            receiver_refractor_elevation_m_sorted=np.full(
                core.receiver_refractor_elevation_m_sorted.shape,
                1001.0,
            ),
            source_weathering_status_sorted=source_sorted_status,
            receiver_weathering_status_sorted=receiver_sorted_status,
            qc={
                'qc_source': 'external_core',
                'node_weathering_status_counts': {'external_node': 6},
                'source_weathering_status_counts': {'external_source': 6},
                'receiver_weathering_status_counts': {'external_receiver': 6},
            },
        )

    monkeypatch.setattr(
        weathering_module,
        'core_build_refraction_weathering_model_from_half_intercept_result',
        _patched_core_build,
    )

    result = build_refraction_weathering_thickness_model(
        half_intercept_result=half,
        model=model,
    )

    np.testing.assert_allclose(result.node_weathering_thickness_m, 101.0)
    np.testing.assert_allclose(result.node_refractor_elevation_m, 202.0)
    assert result.node_weathering_status.tolist() == ['external_node'] * 6
    np.testing.assert_allclose(result.source_weathering_thickness_m, 303.0)
    np.testing.assert_allclose(result.source_refractor_elevation_m, 404.0)
    assert result.source_weathering_status.tolist() == ['external_source'] * 6
    np.testing.assert_allclose(result.receiver_weathering_thickness_m, 505.0)
    np.testing.assert_allclose(result.receiver_refractor_elevation_m, 606.0)
    assert result.receiver_weathering_status.tolist() == ['external_receiver'] * 6
    np.testing.assert_allclose(result.source_weathering_thickness_m_sorted, 707.0)
    np.testing.assert_allclose(result.receiver_weathering_thickness_m_sorted, 808.0)
    np.testing.assert_allclose(result.source_refractor_elevation_m_sorted, 909.0)
    np.testing.assert_allclose(result.receiver_refractor_elevation_m_sorted, 1001.0)
    assert result.source_weathering_status_sorted.tolist() == [
        'external_source_sorted'
    ] * 16
    assert result.receiver_weathering_status_sorted.tolist() == [
        'external_receiver_sorted'
    ] * 16
    assert result.qc['qc_source'] == 'external_core'
    assert result.qc['weathering_status_counts'] == {'external_node': 6}


def test_max_thickness_marks_without_clipping() -> None:
    _inputs, _design, _solved, half = _build_result()
    model = _model(max_weathering_thickness_m=10.0)

    result = build_refraction_weathering_thickness_model(
        half_intercept_result=half,
        model=model,
    )

    assert result.node_weathering_status[0] == 'ok'
    assert result.node_weathering_status[1:5].tolist() == [
        'exceeds_max_thickness',
        'exceeds_max_thickness',
        'exceeds_max_thickness',
        'exceeds_max_thickness',
    ]
    assert np.isnan(result.node_weathering_thickness_m[4])
    assert result.qc['exceeds_max_thickness_count'] == 4


def test_max_thickness_maps_core_values_without_disabled_max_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _inputs, _design, _solved, half = _build_result()
    original_core_build = (
        weathering_module.core_build_refraction_weathering_model_from_half_intercept_result
    )
    max_values: list[float | None] = []

    def _patched_core_build(**kwargs: object):
        core = original_core_build(**kwargs)
        model = kwargs['model']
        max_values.append(getattr(model, 'max_weathering_thickness_m', None))
        status = np.full(core.node_weathering_status.shape, 'exceeds_max_thickness')
        source_status = np.full(
            core.source_endpoint.weathering_status.shape,
            'exceeds_max_thickness',
        )
        receiver_status = np.full(
            core.receiver_endpoint.weathering_status.shape,
            'exceeds_max_thickness',
        )
        source_sorted_status = np.full(
            core.source_weathering_status_sorted.shape,
            'exceeds_max_thickness',
        )
        receiver_sorted_status = np.full(
            core.receiver_weathering_status_sorted.shape,
            'exceeds_max_thickness',
        )
        return replace(
            core,
            node_weathering_thickness_m=np.full(
                core.node_weathering_thickness_m.shape,
                111.0,
            ),
            node_refractor_elevation_m=np.full(
                core.node_refractor_elevation_m.shape,
                222.0,
            ),
            node_weathering_status=status,
            source_endpoint=replace(
                core.source_endpoint,
                weathering_thickness_m=np.full(
                    core.source_endpoint.weathering_thickness_m.shape,
                    333.0,
                ),
                refractor_elevation_m=np.full(
                    core.source_endpoint.refractor_elevation_m.shape,
                    444.0,
                ),
                weathering_status=source_status,
            ),
            receiver_endpoint=replace(
                core.receiver_endpoint,
                weathering_thickness_m=np.full(
                    core.receiver_endpoint.weathering_thickness_m.shape,
                    555.0,
                ),
                refractor_elevation_m=np.full(
                    core.receiver_endpoint.refractor_elevation_m.shape,
                    666.0,
                ),
                weathering_status=receiver_status,
            ),
            source_weathering_thickness_m_sorted=np.full(
                core.source_weathering_thickness_m_sorted.shape,
                777.0,
            ),
            receiver_weathering_thickness_m_sorted=np.full(
                core.receiver_weathering_thickness_m_sorted.shape,
                888.0,
            ),
            source_refractor_elevation_m_sorted=np.full(
                core.source_refractor_elevation_m_sorted.shape,
                999.0,
            ),
            receiver_refractor_elevation_m_sorted=np.full(
                core.receiver_refractor_elevation_m_sorted.shape,
                1000.0,
            ),
            source_weathering_status_sorted=source_sorted_status,
            receiver_weathering_status_sorted=receiver_sorted_status,
        )

    monkeypatch.setattr(
        weathering_module,
        'core_build_refraction_weathering_model_from_half_intercept_result',
        _patched_core_build,
    )

    result = build_refraction_weathering_thickness_model(
        half_intercept_result=half,
        model=_model(max_weathering_thickness_m=10.0),
    )

    assert max_values == [10.0]
    np.testing.assert_allclose(result.node_weathering_thickness_m, 111.0)
    np.testing.assert_allclose(result.node_refractor_elevation_m, 222.0)
    np.testing.assert_allclose(result.source_weathering_thickness_m, 333.0)
    np.testing.assert_allclose(result.source_refractor_elevation_m, 444.0)
    np.testing.assert_allclose(result.receiver_weathering_thickness_m, 555.0)
    np.testing.assert_allclose(result.receiver_refractor_elevation_m, 666.0)
    np.testing.assert_allclose(result.source_weathering_thickness_m_sorted, 777.0)
    np.testing.assert_allclose(result.receiver_weathering_thickness_m_sorted, 888.0)
    np.testing.assert_allclose(result.source_refractor_elevation_m_sorted, 999.0)
    np.testing.assert_allclose(result.receiver_refractor_elevation_m_sorted, 1000.0)


def test_weathering_status_comes_from_core_result() -> None:
    _inputs, _design, _solved, half = _build_result()
    node_half = half.node_half_intercept_time_s.copy()
    node_elevation = half.node_elevation_m.copy()
    node_half[0] = np.nan
    node_half[2] = 0.005
    node_elevation[2] = np.inf
    node_half[3] = np.finfo(np.float64).max
    node_half[4] = 0.0
    status = half.node_solution_status.copy()
    status[0] = 'low_fold'
    status[1] = 'clipped_upper'
    status[2] = 'low_fold'
    status[3] = 'clipped_upper'
    status[4] = 'clipped_lower'

    with np.errstate(over='ignore'):
        result = build_refraction_weathering_thickness_model(
            half_intercept_result=replace(
                half,
                node_half_intercept_time_s=node_half,
                node_elevation_m=node_elevation,
                node_solution_status=status,
            ),
            model=_model(max_weathering_thickness_m=10.0),
        )

    assert result.node_weathering_status.tolist() == [
        'low_fold',
        'clipped_half_intercept_upper',
        'low_fold',
        'clipped_half_intercept_upper',
        'clipped_half_intercept_lower',
        'inactive',
    ]
    assert result.qc['low_fold_node_count'] == 2
    assert result.qc['clipped_half_intercept_node_count'] == 3
    assert result.qc['invalid_half_intercept_node_count'] == 0
    assert result.qc['exceeds_max_thickness_count'] == 0
    assert result.qc['invalid_surface_elevation_count'] == 0
    assert result.qc['invalid_refractor_elevation_count'] == 0
    assert result.qc['zero_thickness_node_count'] == 0


def test_unknown_endpoint_nodes_raise() -> None:
    _inputs, _design, _solved, half = _build_result()
    source_node_id = half.source_node_id.copy()
    source_node_id[0] = 999

    with pytest.raises(RefractionWeatheringThicknessError, match='source_node_id'):
        build_refraction_weathering_thickness_model(
            half_intercept_result=replace(half, source_node_id=source_node_id),
            model=_model(),
        )

    receiver_node_id = half.receiver_node_id.copy()
    receiver_node_id[0] = 999

    with pytest.raises(RefractionWeatheringThicknessError, match='receiver_node_id'):
        build_refraction_weathering_thickness_model(
            half_intercept_result=replace(half, receiver_node_id=receiver_node_id),
            model=_model(),
        )


def test_unknown_valid_trace_nodes_raise() -> None:
    _inputs, _design, _solved, half = _build_result()
    source_node_id_sorted = half.source_node_id_sorted.copy()
    source_node_id_sorted[0] = 999

    with pytest.raises(
        RefractionWeatheringThicknessError,
        match='source_node_id_sorted',
    ):
        build_refraction_weathering_thickness_model(
            half_intercept_result=replace(
                half,
                source_node_id_sorted=source_node_id_sorted,
            ),
            model=_model(),
        )

    receiver_node_id_sorted = half.receiver_node_id_sorted.copy()
    receiver_node_id_sorted[0] = 999

    with pytest.raises(
        RefractionWeatheringThicknessError,
        match='receiver_node_id_sorted',
    ):
        build_refraction_weathering_thickness_model(
            half_intercept_result=replace(
                half,
                receiver_node_id_sorted=receiver_node_id_sorted,
            ),
            model=_model(),
        )


def test_unknown_invalid_trace_nodes_are_marked() -> None:
    _inputs, _design, _solved, half = _build_result()
    source_node_id_sorted = half.source_node_id_sorted.copy()
    source_node_id_sorted[-1] = 999

    result = build_refraction_weathering_thickness_model(
        half_intercept_result=replace(
            half,
            source_node_id_sorted=source_node_id_sorted,
        ),
        model=_model(),
    )

    assert result.source_weathering_status_sorted[-1] == 'inactive'
    assert np.isnan(result.source_weathering_thickness_m_sorted[-1])


def test_invalid_trace_unknown_nodes_from_half_intercept_raise_weathering_error() -> None:
    inputs = _input_model()
    source_node_id_sorted = inputs.source_node_id_sorted.copy()
    source_endpoint_key_sorted = inputs.source_endpoint_key_sorted.astype(object).copy()
    source_node_id_sorted[-1] = 999
    source_endpoint_key_sorted[-1] = 'source:999'

    modified_input = replace(
        inputs,
        source_node_id_sorted=source_node_id_sorted,
        source_endpoint_key_sorted=source_endpoint_key_sorted,
    )
    _inputs, _design, _solved, half = _build_result(input_model=modified_input)

    assert 999 in half.source_node_id.tolist()
    assert 'source:999' in half.source_endpoint_key.tolist()
    source_999 = int(np.flatnonzero(half.source_node_id == 999)[0])
    assert half.source_solution_status[source_999] == 'missing_node'

    with pytest.raises(
        RefractionWeatheringThicknessError,
        match='source_node_id references unknown node_id 999',
    ):
        build_refraction_weathering_thickness_model(
            half_intercept_result=half,
            model=_model(),
        )


def test_invalid_surface_elevation_yields_nan_refractor() -> None:
    _inputs, _design, _solved, half = _build_result()
    node_elevation = half.node_elevation_m.copy()
    node_elevation[0] = np.inf

    result = build_refraction_weathering_thickness_model(
        half_intercept_result=replace(half, node_elevation_m=node_elevation),
        model=_model(),
    )

    assert result.node_weathering_status[0] == 'invalid_surface_elevation'
    assert np.isnan(result.node_refractor_elevation_m[0])
    assert result.source_weathering_status_sorted[0] == 'invalid_surface_elevation'
    assert np.isnan(result.source_refractor_elevation_m_sorted[0])


def test_invalid_endpoint_surface_elevation_yields_nan_refractor() -> None:
    _inputs, _design, _solved, half = _build_result()
    source_elevation = half.source_elevation_m.copy()
    source_elevation_sorted = half.source_elevation_m_sorted.copy()
    source_elevation[0] = np.inf
    source_elevation_sorted[half.source_node_id_sorted == 0] = np.inf

    result = build_refraction_weathering_thickness_model(
        half_intercept_result=replace(
            half,
            source_elevation_m=source_elevation,
            source_elevation_m_sorted=source_elevation_sorted,
        ),
        model=_model(),
    )

    assert result.source_weathering_status[0] == 'invalid_surface_elevation'
    assert np.isnan(result.source_refractor_elevation_m[0])
    assert result.source_weathering_status_sorted[0] == 'invalid_surface_elevation'
    assert np.isnan(result.source_refractor_elevation_m_sorted[0])


def test_velocity_validation_rejects_unphysical_context() -> None:
    with pytest.raises(RefractionWeatheringThicknessError, match='greater'):
        compute_weathering_thickness_from_half_intercept_time(
            half_intercept_time_s=np.asarray([0.01], dtype=np.float64),
            weathering_velocity_m_s=1200.0,
            bedrock_velocity_m_s=1200.0,
        )

    _inputs, _design, _solved, half = _build_result()
    with pytest.raises(RefractionWeatheringThicknessError, match='slowness'):
        build_refraction_weathering_thickness_model(
            half_intercept_result=replace(
                half,
                bedrock_slowness_s_per_m=TRUE_BEDROCK_SLOWNESS_S_PER_M * 1.01,
            ),
            model=_model(),
        )
