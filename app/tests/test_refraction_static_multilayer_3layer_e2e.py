from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import (
    RefractionStaticApplyOptions,
    RefractionStaticDatumRequest,
    RefractionStaticSolverRequest,
)
from app.services.refraction_static_artifacts import (
    NEAR_SURFACE_MODEL_CSV_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_STATICS_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)
from app.services.refraction_static_multilayer_service import (
    compute_refraction_multilayer_datum_statics_from_input_model,
)
from app.tests._refraction_multilayer_3layer_helpers import (
    STATIC_ATOL_S,
    THICKNESS_ATOL_M,
    compute_three_layer_workflow,
    make_three_layer_dataset,
    make_three_layer_input_model,
    make_three_layer_model,
    resolved_first_layer,
)


def test_three_layer_trace_shift_is_source_plus_receiver_plus_datum() -> None:
    datum = RefractionStaticDatumRequest(
        mode='floating_and_flat',
        floating_datum_mode='constant',
        floating_datum_elevation_m=125.0,
        flat_datum_elevation_m=175.0,
    )
    _dataset, _input_model, _model, workflow = compute_three_layer_workflow(
        datum=datum,
    )
    result = workflow.datum_result

    expected_trace = (
        result.source_refraction_shift_s_sorted
        + result.receiver_refraction_shift_s_sorted
    )
    np.testing.assert_allclose(
        result.refraction_trace_shift_s_sorted,
        expected_trace,
        atol=STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        result.refraction_trace_shift_s_sorted,
        result.weathering_replacement_trace_shift_s_sorted
        + result.floating_datum_elevation_shift_s_sorted
        + result.flat_datum_shift_s_sorted,
        atol=STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        result.source_refraction_shift_s,
        result.source_weathering_replacement_shift_s
        + result.source_floating_datum_elevation_shift_s
        + result.source_flat_datum_shift_s,
        atol=STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        result.receiver_refraction_shift_s,
        result.receiver_weathering_replacement_shift_s
        + result.receiver_floating_datum_elevation_shift_s
        + result.receiver_flat_datum_shift_s,
        atol=STATIC_ATOL_S,
    )
    assert np.all(result.trace_static_valid_mask_sorted)
    assert np.all(result.trace_static_status_sorted == 'ok')


def test_three_layer_job_dir_writes_sh3_vsub_and_layer2_base_artifacts(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'job'
    dataset, _input_model, _model, workflow = compute_three_layer_workflow(
        job_dir=job_dir,
    )
    result = workflow.datum_result

    for name in (
        REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        REFRACTION_STATIC_QC_JSON_NAME,
        REFRACTION_STATICS_CSV_NAME,
        NEAR_SURFACE_MODEL_CSV_NAME,
        SOURCE_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    ):
        assert (job_dir / name).is_file()

    qc = json.loads((job_dir / REFRACTION_STATIC_QC_JSON_NAME).read_text())
    assert qc['method'] == 'multilayer_time_term'
    assert qc['conversion_mode'] == 't1lsst_multilayer'
    assert qc['layer_count'] == 3
    assert qc['enabled_layer_kinds'] == ['v2_t1', 'v3_t2', 'vsub_t3']

    near_surface_rows = _read_csv(job_dir / NEAR_SURFACE_MODEL_CSV_NAME)
    assert {
        'sh3_weathering_thickness_m',
        'layer2_base_elevation_m',
        'final_refractor_elevation_m',
    } <= set(near_surface_rows[0])
    first_node_row = near_surface_rows[0]
    expected_node_sh1 = dataset.true_source_endpoint_sh1_m[0]
    expected_node_sh2 = dataset.true_source_endpoint_sh2_m[0]
    expected_node_sh3 = dataset.true_source_endpoint_sh3_m[0]
    expected_layer2 = (
        dataset.source_endpoint_elevation_m[0] - expected_node_sh1 - expected_node_sh2
    )
    expected_final = expected_layer2 - expected_node_sh3
    assert float(first_node_row['layer2_base_elevation_m']) == pytest.approx(
        expected_layer2,
        abs=THICKNESS_ATOL_M,
    )
    assert float(first_node_row['final_refractor_elevation_m']) == pytest.approx(
        expected_final,
        abs=THICKNESS_ATOL_M,
    )
    assert float(first_node_row['refractor_elevation_m']) == pytest.approx(
        expected_final,
        abs=THICKNESS_ATOL_M,
    )

    source_rows = _read_csv(job_dir / SOURCE_STATIC_TABLE_CSV_NAME)
    assert {
        't3_ms',
        'vsub_m_s',
        'sh3_weathering_thickness_m',
        'layer2_base_elevation_m',
    } <= set(source_rows[0])
    assert float(source_rows[0]['sh3_weathering_thickness_m']) == pytest.approx(
        dataset.true_source_endpoint_sh3_m[0],
        abs=THICKNESS_ATOL_M,
    )

    with np.load(job_dir / SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME) as data:
        assert {
            'source_t3_s',
            'source_vsub_m_s',
            'source_sh3_m',
            'source_layer2_base_elevation_m',
            'receiver_t3_s',
            'receiver_vsub_m_s',
            'receiver_sh3_m',
            'receiver_layer2_base_elevation_m',
        } <= set(data.files)
        np.testing.assert_allclose(
            data['source_sh3_m'],
            dataset.true_source_endpoint_sh3_m,
            atol=THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['receiver_sh3_m'],
            dataset.true_receiver_endpoint_sh3_m,
            atol=THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['source_layer2_base_elevation_m'],
            result.source_surface_elevation_m
            - result.source_sh1_weathering_thickness_m
            - result.source_sh2_weathering_thickness_m,
            atol=THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['receiver_layer2_base_elevation_m'],
            result.receiver_surface_elevation_m
            - result.receiver_sh1_weathering_thickness_m
            - result.receiver_sh2_weathering_thickness_m,
            atol=THICKNESS_ATOL_M,
        )

    with np.load(job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME) as data:
        assert {
            'node_sh3_weathering_thickness_m',
            'node_layer2_base_elevation_m',
            'source_t3_time_s',
            'source_vsub_m_s',
            'source_sh3_weathering_thickness_m',
            'receiver_t3_time_s',
            'receiver_vsub_m_s',
            'receiver_sh3_weathering_thickness_m',
        } <= set(data.files)
        np.testing.assert_allclose(
            data['source_sh3_weathering_thickness_m'],
            dataset.true_source_endpoint_sh3_m,
            atol=THICKNESS_ATOL_M,
        )


def test_three_layer_solver_missing_layer_terms_are_statused_not_raised() -> None:
    dataset = make_three_layer_dataset()
    input_model = make_three_layer_input_model(dataset)
    workflow = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=input_model,
        model=make_three_layer_model(),
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            min_picks_per_node=1,
            max_abs_half_intercept_time_ms=500.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=500.0),
        resolved_first_layer=resolved_first_layer(),
    )
    result = workflow.datum_result

    assert result.qc['layer_count'] == 3
    assert result.qc['enabled_layer_kinds'] == ['v2_t1', 'v3_t2', 'vsub_t3']
    assert np.all(result.source_datum_status == 'ok')
    assert np.all(result.receiver_datum_status == 'invalid_nonfinite_input')
    assert np.all(result.trace_static_status_sorted == 'invalid_nonfinite_input')
    assert not np.any(result.trace_static_valid_mask_sorted)
    assert np.all(np.isnan(result.refraction_trace_shift_s_sorted))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))
