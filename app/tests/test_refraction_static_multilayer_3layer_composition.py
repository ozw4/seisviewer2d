from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import RefractionStaticApplyOptions, RefractionStaticDatumRequest
from app.statics.refraction.application.datum import build_refraction_datum_statics
from app.statics.refraction.application.multilayer_service import (
    build_refraction_multilayer_weathering_replacement_statics,
)
from app.statics.refraction.domain.t1lsst import (
    RefractionT1LSSTError,
    compute_t1lsst_3layer_weathering_correction,
)
from app.tests._refraction_multilayer_3layer_helpers import (
    STATIC_ATOL_S,
    THICKNESS_ATOL_M,
    compute_three_layer_workflow,
    layer,
    resolved_first_layer,
)


def test_three_layer_static_composition_uses_sh1_plus_sh2_plus_sh3_for_final_refractor() -> None:
    dataset, _input_model, _model, workflow = compute_three_layer_workflow()
    replacement = workflow.weathering_replacement_result
    datum_result = workflow.datum_result
    components = workflow.components

    assert components.source_t3_s is not None
    assert components.receiver_t3_s is not None
    assert components.source_sh2_m is not None
    assert components.source_sh3_m is not None
    assert components.receiver_sh2_m is not None
    assert components.receiver_sh3_m is not None

    expected_source_total = (
        dataset.true_source_endpoint_sh1_m
        + dataset.true_source_endpoint_sh2_m
        + dataset.true_source_endpoint_sh3_m
    )
    expected_receiver_total = (
        dataset.true_receiver_endpoint_sh1_m
        + dataset.true_receiver_endpoint_sh2_m
        + dataset.true_receiver_endpoint_sh3_m
    )
    np.testing.assert_allclose(
        replacement.source_weathering_thickness_m,
        expected_source_total,
        atol=THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        replacement.receiver_weathering_thickness_m,
        expected_receiver_total,
        atol=THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        datum_result.source_refractor_elevation_m,
        dataset.source_endpoint_elevation_m - expected_source_total,
        atol=THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        datum_result.receiver_refractor_elevation_m,
        dataset.receiver_endpoint_elevation_m - expected_receiver_total,
        atol=THICKNESS_ATOL_M,
    )

    node_source_count = int(dataset.source_endpoint_node_id.size)
    expected_node_total = np.concatenate(
        (expected_source_total, expected_receiver_total)
    )
    np.testing.assert_allclose(
        datum_result.node_weathering_thickness_m,
        expected_node_total,
        atol=THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        datum_result.node_refractor_elevation_m[:node_source_count],
        dataset.source_endpoint_elevation_m - expected_source_total,
        atol=THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        datum_result.node_refractor_elevation_m[node_source_count:],
        dataset.receiver_endpoint_elevation_m - expected_receiver_total,
        atol=THICKNESS_ATOL_M,
    )


def test_three_layer_weathering_correction_matches_t1lsst_formula() -> None:
    dataset, _input_model, _model, workflow = compute_three_layer_workflow()
    result = workflow.weathering_replacement_result

    expected_source = compute_t1lsst_3layer_weathering_correction(
        sh1_m=dataset.true_source_endpoint_sh1_m,
        sh2_m=dataset.true_source_endpoint_sh2_m,
        sh3_m=dataset.true_source_endpoint_sh3_m,
        v1_m_s=dataset.true_v1_m_s,
        v2_m_s=dataset.true_v2_m_s,
        v3_m_s=dataset.true_v3_m_s,
        vsub_m_s=dataset.true_vsub_m_s,
    )
    expected_receiver = compute_t1lsst_3layer_weathering_correction(
        sh1_m=dataset.true_receiver_endpoint_sh1_m,
        sh2_m=dataset.true_receiver_endpoint_sh2_m,
        sh3_m=dataset.true_receiver_endpoint_sh3_m,
        v1_m_s=dataset.true_v1_m_s,
        v2_m_s=dataset.true_v2_m_s,
        v3_m_s=dataset.true_v3_m_s,
        vsub_m_s=dataset.true_vsub_m_s,
    )

    np.testing.assert_allclose(
        result.source_weathering_replacement_shift_s,
        expected_source,
        atol=STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        result.receiver_weathering_replacement_shift_s,
        expected_receiver,
        atol=STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        workflow.components.source_weathering_correction_s,
        dataset.true_source_endpoint_wcor_s,
        atol=STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        workflow.components.receiver_weathering_correction_s,
        dataset.true_receiver_endpoint_wcor_s,
        atol=STATIC_ATOL_S,
    )


def test_three_layer_invalid_endpoint_status_propagates_to_trace_status() -> None:
    _dataset, input_model, model, workflow = compute_three_layer_workflow()
    bad_source_index = 0
    vsub_layer = layer(workflow.solve_result, 'vsub_t3')
    bad_source_t3 = np.asarray(vsub_layer.source_time_term_s, dtype=np.float64).copy()
    bad_source_t3[bad_source_index] = 0.0
    patched_vsub_layer = replace(vsub_layer, source_time_term_s=bad_source_t3)
    patched_solve = replace(
        workflow.solve_result,
        layer_results=(
            layer(workflow.solve_result, 'v2_t1'),
            layer(workflow.solve_result, 'v3_t2'),
            patched_vsub_layer,
        ),
    )

    replacement = build_refraction_multilayer_weathering_replacement_statics(
        input_model=input_model,
        model=model,
        solve_result=patched_solve,
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=500.0),
        resolved_first_layer=resolved_first_layer(),
    )
    datum_result = build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=500.0),
        resolved_first_layer=resolved_first_layer(),
    )

    assert replacement.source_static_status[bad_source_index] != 'ok'
    assert datum_result.source_datum_status[bad_source_index] != 'ok'
    assert np.isnan(replacement.source_weathering_replacement_shift_s[bad_source_index])
    assert np.isnan(datum_result.source_refraction_shift_s[bad_source_index])

    bad_source_key = str(replacement.source_endpoint_key[bad_source_index])
    bad_trace_mask = np.asarray(
        [str(key) == bad_source_key for key in input_model.source_endpoint_key_sorted],
        dtype=bool,
    )
    assert np.any(bad_trace_mask)
    assert np.all(datum_result.trace_static_status_sorted[bad_trace_mask] != 'ok')
    assert not np.any(datum_result.trace_static_valid_mask_sorted[bad_trace_mask])
    assert np.all(np.isnan(datum_result.refraction_trace_shift_s_sorted[bad_trace_mask]))

    ok_trace_mask = ~bad_trace_mask
    assert np.any(ok_trace_mask)
    assert np.all(datum_result.trace_static_status_sorted[ok_trace_mask] == 'ok')
    assert np.all(datum_result.trace_static_valid_mask_sorted[ok_trace_mask])


def test_three_layer_global_velocity_order_is_rejected() -> None:
    dataset, input_model, model, workflow = compute_three_layer_workflow()
    vsub_layer = layer(workflow.solve_result, 'vsub_t3')
    invalid_vsub_m_s = dataset.true_v3_m_s
    patched_vsub_layer = replace(
        vsub_layer,
        global_velocity_m_s=invalid_vsub_m_s,
        global_slowness_s_per_m=1.0 / invalid_vsub_m_s,
    )
    patched_solve = replace(
        workflow.solve_result,
        layer_results=(
            layer(workflow.solve_result, 'v2_t1'),
            layer(workflow.solve_result, 'v3_t2'),
            patched_vsub_layer,
        ),
    )

    with pytest.raises(RefractionT1LSSTError, match='vsub_m_s must be greater'):
        build_refraction_multilayer_weathering_replacement_statics(
            input_model=input_model,
            model=model,
            solve_result=patched_solve,
            apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=500.0),
            resolved_first_layer=resolved_first_layer(),
        )


def test_two_layer_static_composition_regression_still_passes(tmp_path: Path) -> None:
    from app.tests.test_refraction_static_multilayer_2layer_e2e import (
        _STATIC_ATOL_S,
        _THICKNESS_ATOL_M,
        _compute_static_outputs,
        _make_two_layer_fixture,
    )

    fixture = _make_two_layer_fixture(
        coordinate_mode='grid_3d',
        v2_velocity_mode='solve_global',
    )
    outputs = _compute_static_outputs(fixture=fixture, tmp_path=tmp_path)

    np.testing.assert_allclose(
        outputs.result.source_weathering_thickness_m,
        fixture.source_sh1_m + fixture.source_sh2_m,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        outputs.result.receiver_weathering_thickness_m,
        fixture.receiver_sh1_m + fixture.receiver_sh2_m,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        outputs.trace_shift_s_sorted,
        fixture.trace_shift_s_sorted,
        atol=_STATIC_ATOL_S,
    )
