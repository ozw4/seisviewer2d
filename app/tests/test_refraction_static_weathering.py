from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import app.services.refraction_static_weathering as weathering_module
from app.services.refraction_static_weathering import (
    REFRACTION_WEATHERING_NODES_CSV_NAME,
    REFRACTION_WEATHERING_QC_JSON_NAME,
    REFRACTION_WEATHERING_RECEIVERS_CSV_NAME,
    REFRACTION_WEATHERING_SOURCES_CSV_NAME,
    REFRACTION_WEATHERING_TRACE_PREVIEW_CSV_NAME,
    RefractionWeatheringThicknessError,
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


def test_public_apis_are_importable() -> None:
    assert callable(estimate_weathering_thickness_from_first_breaks)
    assert callable(build_refraction_weathering_thickness_model)
    assert callable(compute_weathering_thickness_from_half_intercept_time)
    assert callable(compute_weathering_thickness_scalar)


def test_high_level_pipeline_calls_half_intercept_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _inputs, _design, _solved, half = _build_result()
    calls: list[dict[str, object]] = []

    def _capture_estimate(**kwargs: object):
        calls.append(kwargs)
        return half

    monkeypatch.setattr(
        weathering_module,
        'estimate_refraction_half_intercept_times_from_first_breaks',
        _capture_estimate,
    )
    req = SimpleNamespace(model=_model())
    state = object()

    result = estimate_weathering_thickness_from_first_breaks(
        req=req,  # type: ignore[arg-type]
        state=state,  # type: ignore[arg-type]
        job_dir=tmp_path,
    )

    assert calls == [{'req': req, 'state': state, 'job_dir': tmp_path}]
    expected_node_thickness = TRUE_HALF_INTERCEPT_S * _conversion_factor()
    np.testing.assert_allclose(
        result.node_weathering_thickness_m[:5],
        expected_node_thickness,
        rtol=1.0e-9,
    )
    assert (tmp_path / REFRACTION_WEATHERING_QC_JSON_NAME).is_file()


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


def test_build_weathering_model_maps_nodes_endpoints_and_trace_order(
    tmp_path: Path,
) -> None:
    _inputs, _design, _solved, half = _build_result()
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
    assert result.qc['weathering_thickness_median_m'] == pytest.approx(
        float(np.median(expected_node_thickness))
    )
    assert result.qc['thickness_formula'] == (
        'z = T * vb * vw / sqrt(vb^2 - vw^2)'
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


def test_trace_refractor_elevation_uses_endpoint_surface_elevation() -> None:
    _inputs, _design, _solved, half = _build_result()
    source_elevation = half.source_elevation_m.copy()
    receiver_elevation = half.receiver_elevation_m.copy()
    source_elevation_sorted = half.source_elevation_m_sorted.copy()
    receiver_elevation_sorted = half.receiver_elevation_m_sorted.copy()

    source_node_zero = int(np.flatnonzero(half.source_node_id == 0)[0])
    receiver_node_zero = int(np.flatnonzero(half.receiver_node_id == 0)[0])
    source_elevation[source_node_zero] = 110.0
    receiver_elevation[receiver_node_zero] = 210.0
    source_elevation_sorted[half.source_node_id_sorted == 0] = 110.0
    receiver_elevation_sorted[half.receiver_node_id_sorted == 0] = 210.0

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
    source_trace = int(np.flatnonzero(result.source_node_id_sorted == 0)[0])
    receiver_trace = int(np.flatnonzero(result.receiver_node_id_sorted == 0)[0])
    assert result.source_refractor_elevation_m[source_node_zero] == pytest.approx(
        110.0 - expected_thickness
    )
    assert result.receiver_refractor_elevation_m[receiver_node_zero] == pytest.approx(
        210.0 - expected_thickness
    )
    assert result.source_refractor_elevation_m_sorted[source_trace] == pytest.approx(
        110.0 - expected_thickness
    )
    assert result.receiver_refractor_elevation_m_sorted[receiver_trace] == pytest.approx(
        210.0 - expected_thickness
    )


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
    assert result.node_weathering_thickness_m[4] > 10.0
    assert result.qc['exceeds_max_thickness_count'] == 4


def test_weathering_status_priority_preserves_higher_priority_statuses() -> None:
    _inputs, _design, _solved, half = _build_result()
    node_half = half.node_half_intercept_time_s.copy()
    node_elevation = half.node_elevation_m.copy()
    source_elevation = half.source_elevation_m.copy()
    source_elevation_sorted = half.source_elevation_m_sorted.copy()
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
    source_node_two = int(np.flatnonzero(half.source_node_id == 2)[0])
    source_elevation[source_node_two] = np.inf
    source_elevation_sorted[half.source_node_id_sorted == 2] = np.inf

    with np.errstate(over='ignore'):
        result = build_refraction_weathering_thickness_model(
            half_intercept_result=replace(
                half,
                node_half_intercept_time_s=node_half,
                node_elevation_m=node_elevation,
                node_solution_status=status,
                source_elevation_m=source_elevation,
                source_elevation_m_sorted=source_elevation_sorted,
            ),
            model=_model(max_weathering_thickness_m=10.0),
        )

    assert result.node_weathering_status.tolist() == [
        'invalid_half_intercept',
        'exceeds_max_thickness',
        'invalid_surface_elevation',
        'invalid_refractor_elevation',
        'clipped_half_intercept_lower',
        'inactive',
    ]
    source_node_zero = int(np.flatnonzero(result.source_node_id == 0)[0])
    receiver_node_one = int(np.flatnonzero(result.receiver_node_id == 1)[0])
    source_node_three = int(np.flatnonzero(result.source_node_id == 3)[0])
    source_node_four = int(np.flatnonzero(result.source_node_id == 4)[0])
    source_trace_zero = int(np.flatnonzero(result.source_node_id_sorted == 0)[0])
    receiver_trace_one = int(np.flatnonzero(result.receiver_node_id_sorted == 1)[0])
    source_trace_two = int(np.flatnonzero(result.source_node_id_sorted == 2)[0])
    source_trace_three = int(np.flatnonzero(result.source_node_id_sorted == 3)[0])
    source_trace_four = int(np.flatnonzero(result.source_node_id_sorted == 4)[0])
    assert result.source_weathering_status[source_node_zero] == (
        'invalid_half_intercept'
    )
    assert (
        result.receiver_weathering_status[receiver_node_one]
        == 'exceeds_max_thickness'
    )
    assert result.source_weathering_status[source_node_two] == (
        'invalid_surface_elevation'
    )
    assert result.source_weathering_status[source_node_three] == (
        'invalid_refractor_elevation'
    )
    assert result.source_weathering_status[source_node_four] == (
        'clipped_half_intercept_lower'
    )
    assert result.source_weathering_status_sorted[source_trace_zero] == (
        'invalid_half_intercept'
    )
    assert (
        result.receiver_weathering_status_sorted[receiver_trace_one]
        == 'exceeds_max_thickness'
    )
    assert result.source_weathering_status_sorted[source_trace_two] == (
        'invalid_surface_elevation'
    )
    assert result.source_weathering_status_sorted[source_trace_three] == (
        'invalid_refractor_elevation'
    )
    assert result.source_weathering_status_sorted[source_trace_four] == (
        'clipped_half_intercept_lower'
    )
    assert result.qc['low_fold_node_count'] == 0
    assert result.qc['clipped_half_intercept_node_count'] == 1
    assert result.qc['invalid_half_intercept_node_count'] == 1
    assert result.qc['exceeds_max_thickness_count'] == 1
    assert result.qc['invalid_surface_elevation_count'] == 1
    assert result.qc['invalid_refractor_elevation_count'] == 1
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

    assert result.source_weathering_status_sorted[-1] == 'missing_node'
    assert np.isnan(result.source_weathering_thickness_m_sorted[-1])


def test_invalid_trace_unknown_nodes_from_half_intercept_do_not_emit_endpoints() -> None:
    inputs = _input_model()
    source_node_id_sorted = inputs.source_node_id_sorted.copy()
    source_endpoint_key_sorted = inputs.source_endpoint_key_sorted.astype(object).copy()
    source_node_id_sorted[-1] = 999
    source_endpoint_key_sorted[-1] = 'source:999'

    _inputs, _design, _solved, half = _build_result(
        input_model=replace(
            inputs,
            source_node_id_sorted=source_node_id_sorted,
            source_endpoint_key_sorted=source_endpoint_key_sorted,
        ),
    )

    assert 999 not in half.source_node_id.tolist()
    assert 'source:999' not in half.source_endpoint_key.tolist()

    result = build_refraction_weathering_thickness_model(
        half_intercept_result=half,
        model=_model(),
    )

    assert result.source_weathering_status_sorted[-1] == 'missing_node'
    assert np.isnan(result.source_weathering_thickness_m_sorted[-1])


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
    assert result.source_weathering_status_sorted[0] == 'ok'
    assert np.isfinite(result.source_refractor_elevation_m_sorted[0])


def test_invalid_endpoint_surface_elevation_yields_nan_trace_refractor() -> None:
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
