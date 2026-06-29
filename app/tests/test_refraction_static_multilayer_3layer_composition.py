from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import RefractionStaticApplyOptions, RefractionStaticDatumRequest
from app.statics.refraction.application.datum import build_refraction_datum_statics
import app.statics.refraction.application.multilayer_service as multilayer_service
from app.statics.refraction.application.multilayer_service import (
    build_refraction_multilayer_weathering_replacement_statics,
)
from seis_statics.refraction.t1lsst import compute_t1lsst_3layer_weathering_correction
from app.tests._refraction_multilayer_3layer_helpers import (
    STATIC_ATOL_S,
    THICKNESS_ATOL_M,
    compute_three_layer_workflow,
    layer,
    resolved_first_layer,
)


def test_three_layer_static_composition_uses_sh1_plus_sh2_plus_sh3_for_final_refractor() -> None:
    _dataset, _input_model, _model, workflow = compute_three_layer_workflow()
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
        components.source_sh1_m + components.source_sh2_m + components.source_sh3_m
    )
    expected_receiver_total = (
        components.receiver_sh1_m
        + components.receiver_sh2_m
        + components.receiver_sh3_m
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
        replacement.source_surface_elevation_m - expected_source_total,
        atol=THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        datum_result.receiver_refractor_elevation_m,
        replacement.receiver_surface_elevation_m - expected_receiver_total,
        atol=THICKNESS_ATOL_M,
    )

    node_source_count = int(replacement.source_endpoint_key.size)
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
        replacement.source_surface_elevation_m - expected_source_total,
        atol=THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        datum_result.node_refractor_elevation_m[node_source_count:],
        replacement.receiver_surface_elevation_m - expected_receiver_total,
        atol=THICKNESS_ATOL_M,
    )


def test_three_layer_weathering_correction_matches_t1lsst_formula() -> None:
    _dataset, _input_model, _model, workflow = compute_three_layer_workflow()
    result = workflow.weathering_replacement_result

    expected_source = compute_t1lsst_3layer_weathering_correction(
        sh1_m=result.source_sh1_weathering_thickness_m,
        sh2_m=result.source_sh2_weathering_thickness_m,
        sh3_m=result.source_sh3_weathering_thickness_m,
        v1_m_s=result.weathering_velocity_m_s,
        v2_m_s=result.source_v2_m_s,
        v3_m_s=result.source_v3_m_s,
        vsub_m_s=result.source_vsub_m_s,
    )
    expected_receiver = compute_t1lsst_3layer_weathering_correction(
        sh1_m=result.receiver_sh1_weathering_thickness_m,
        sh2_m=result.receiver_sh2_weathering_thickness_m,
        sh3_m=result.receiver_sh3_weathering_thickness_m,
        v1_m_s=result.weathering_velocity_m_s,
        v2_m_s=result.receiver_v2_m_s,
        v3_m_s=result.receiver_v3_m_s,
        vsub_m_s=result.receiver_vsub_m_s,
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
        result.source_weathering_replacement_shift_s,
        atol=STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        workflow.components.receiver_weathering_correction_s,
        result.receiver_weathering_replacement_shift_s,
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

    assert np.count_nonzero(datum_result.trace_static_status_sorted != 'ok') >= (
        np.count_nonzero(bad_trace_mask)
    )


def test_three_layer_core_conversion_adapter_uses_matching_trace_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dataset, input_model, model, workflow = compute_three_layer_workflow()
    original = multilayer_service.core_build_refraction_multilayer_conversion
    observed_trace_counts: list[int] = []

    def _assert_matching_shapes(**kwargs):
        core_input = kwargs['input_model']
        core_solve = kwargs['solve_result']
        n_traces = int(core_input.n_traces)
        observed_trace_counts.append(n_traces)
        for value in (
            core_solve.modeled_pick_time_s_sorted,
            core_solve.residual_s_sorted,
            core_solve.residual_ms_sorted,
            core_solve.used_observation_mask_sorted,
            core_solve.rejected_observation_mask_sorted,
            core_solve.layer_kind_sorted,
            core_solve.rejection_reason_sorted,
            core_solve.velocity_m_s_sorted,
        ):
            assert np.asarray(value).shape == (n_traces,)
        layer_masks = core_solve.layer_observation_masks
        if layer_masks is not None:
            for mask_by_kind in (
                layer_masks.layer_used_mask_sorted,
                layer_masks.layer_rejection_reason_sorted,
            ):
                for value in mask_by_kind.values():
                    assert np.asarray(value).shape == (n_traces,)
        for layer_result in core_solve.layer_results:
            assert np.asarray(layer_result.velocity_m_s_sorted).shape == (n_traces,)
            assert np.asarray(layer_result.rejection_reason_sorted).shape == (n_traces,)
            assert np.asarray(layer_result.velocity_order_valid_mask_sorted).shape == (
                n_traces,
            )
        return original(**kwargs)

    monkeypatch.setattr(
        multilayer_service,
        'core_build_refraction_multilayer_conversion',
        _assert_matching_shapes,
    )

    build_refraction_multilayer_weathering_replacement_statics(
        input_model=input_model,
        model=model,
        solve_result=workflow.solve_result,
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=500.0),
        resolved_first_layer=resolved_first_layer(),
    )

    assert observed_trace_counts == [
        input_model.n_traces,
        input_model.node_x_m.shape[0],
    ]


def test_three_layer_global_velocity_order_is_reported_by_core() -> None:
    from app.tests.test_refraction_static_multilayer_vsub_t3_solver import (
        V3_M_S,
        _input_model,
        _model,
        _run_multilayer,
    )

    result = _run_multilayer(
        model=_model(
            vsub_velocity_mode='fixed_global',
            vsub_fixed_velocity_m_s=V3_M_S,
        ),
        input_model=_input_model(),
    )

    assert 'invalid_velocity_order' in set(result.rejection_reason_sorted.astype(str))


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
