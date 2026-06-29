from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import (
    RefractionStaticApplyOptions,
    RefractionStaticDatumRequest,
    RefractionStaticModelRequest,
    RefractionStaticSolverRequest,
)
from app.statics.refraction.artifacts import (
    NEAR_SURFACE_MODEL_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_STATICS_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)
from app.statics.refraction.application.core_options import (
    layer_observation_masks_from_input_model as build_refraction_layer_observation_masks,
)
from app.statics.refraction.application.multilayer_service import (
    RefractionMultiLayerStaticsWorkflowResult,
    compute_refraction_multilayer_datum_statics_from_input_model,
)
from app.statics.refraction.contracts.result_types import (
    RefractionEndpointTable,
    RefractionLayerSolveResult,
    RefractionStaticInputModel,
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
from app.tests._refraction_multilayer_synthetic import (
    SYNTHETIC_MULTILAYER_V1_M_S,
    SYNTHETIC_MULTILAYER_V2_M_S,
    SYNTHETIC_MULTILAYER_V3_M_S,
    SYNTHETIC_MULTILAYER_VSUB_M_S,
)


_V2_MIN_OFFSET_M = 250.0
_V2_MAX_OFFSET_M = 1000.0
_V3_MIN_OFFSET_M = 1000.0
_V3_MAX_OFFSET_M = 2000.0
_VSUB_MIN_OFFSET_M = 2000.0
_INITIAL_V2_M_S = 2100.0
_INITIAL_V3_M_S = 3300.0
_INITIAL_VSUB_M_S = 5400.0
_VELOCITY_RTOL = 1.0e-8
_TRACE_SHIFT_ATOL_S = 1.0e-8
_SIGN_CONVENTION = 'corrected(t) = raw(t - shift_s)'


@dataclass(frozen=True)
class _ThreeLayerSolverFixture:
    input_model: RefractionStaticInputModel
    model: RefractionStaticModelRequest
    expected_layer_kind_sorted: np.ndarray
    source_node_id: np.ndarray
    receiver_node_id: np.ndarray
    source_t1_s: np.ndarray
    source_t2_s: np.ndarray
    source_t3_s: np.ndarray
    receiver_t1_s: np.ndarray
    receiver_t2_s: np.ndarray
    receiver_t3_s: np.ndarray
    source_sh1_m: np.ndarray
    source_sh2_m: np.ndarray
    source_sh3_m: np.ndarray
    receiver_sh1_m: np.ndarray
    receiver_sh2_m: np.ndarray
    receiver_sh3_m: np.ndarray
    source_wcor_s: np.ndarray
    receiver_wcor_s: np.ndarray
    trace_shift_s_sorted: np.ndarray


def test_three_layer_global_vsub_e2e_recovers_known_truth(tmp_path: Path) -> None:
    fixture = _make_three_layer_solver_fixture()
    job_dir = tmp_path / 'job'

    workflow = _compute_three_layer_solver_workflow(fixture, job_dir=job_dir)

    _assert_layer_masks_select_expected_three_layer_branches(fixture)
    _assert_global_layer_result(
        _layer(workflow, 'v2_t1'),
        expected_velocity_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
    )
    _assert_global_layer_result(
        _layer(workflow, 'v3_t2'),
        expected_velocity_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
    )
    _assert_global_layer_result(
        _layer(workflow, 'vsub_t3'),
        expected_velocity_m_s=SYNTHETIC_MULTILAYER_VSUB_M_S,
    )
    _assert_three_layer_time_terms_and_thicknesses_match_truth(fixture, workflow)

    qc = json.loads(
        (job_dir / REFRACTION_STATIC_QC_JSON_NAME).read_text(encoding='utf-8')
    )
    assert qc['method'] == 'multilayer_time_term'
    assert qc['conversion_mode'] == 't1lsst_multilayer'
    assert qc['layer_count'] == 3
    assert qc['enabled_layer_kinds'] == ['v2_t1', 'v3_t2', 'vsub_t3']
    assert qc['velocity']['layer_velocity_modes'] == {
        'v2_t1': 'solve_global',
        'v3_t2': 'solve_global',
        'vsub_t3': 'solve_global',
    }
    assert qc['sign_convention'] == _SIGN_CONVENTION


def test_three_layer_static_table_matches_known_sh1_sh2_sh3_wcor(
    tmp_path: Path,
) -> None:
    fixture = _make_three_layer_solver_fixture()
    job_dir = tmp_path / 'job'

    workflow = _compute_three_layer_solver_workflow(fixture, job_dir=job_dir)
    source_rows = _read_csv(job_dir / SOURCE_STATIC_TABLE_CSV_NAME)
    receiver_rows = _read_csv(job_dir / RECEIVER_STATIC_TABLE_CSV_NAME)

    _assert_static_rows_match_three_layer_truth(
        source_rows,
        expected_t1_s=fixture.source_t1_s,
        expected_t2_s=fixture.source_t2_s,
        expected_t3_s=fixture.source_t3_s,
        expected_sh1_m=fixture.source_sh1_m,
        expected_sh2_m=fixture.source_sh2_m,
        expected_sh3_m=fixture.source_sh3_m,
        expected_wcor_s=fixture.source_wcor_s,
    )
    _assert_static_rows_match_three_layer_truth(
        receiver_rows,
        expected_t1_s=fixture.receiver_t1_s,
        expected_t2_s=fixture.receiver_t2_s,
        expected_t3_s=fixture.receiver_t3_s,
        expected_sh1_m=fixture.receiver_sh1_m,
        expected_sh2_m=fixture.receiver_sh2_m,
        expected_sh3_m=fixture.receiver_sh3_m,
        expected_wcor_s=fixture.receiver_wcor_s,
    )

    with np.load(
        job_dir / SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        allow_pickle=False,
    ) as data:
        expected_keys = {
            'source_t1_s',
            'source_t2_s',
            'source_t3_s',
            'source_sh1_m',
            'source_sh2_m',
            'source_sh3_m',
            'source_weathering_correction_s',
            'source_total_static_s',
            'receiver_t1_s',
            'receiver_t2_s',
            'receiver_t3_s',
            'receiver_sh1_m',
            'receiver_sh2_m',
            'receiver_sh3_m',
            'receiver_weathering_correction_s',
            'receiver_total_static_s',
        }
        assert expected_keys <= set(data.files)
        np.testing.assert_allclose(data['source_t1_s'], fixture.source_t1_s)
        np.testing.assert_allclose(data['source_t2_s'], fixture.source_t2_s)
        np.testing.assert_allclose(data['source_t3_s'], fixture.source_t3_s)
        np.testing.assert_allclose(
            data['source_sh1_m'],
            fixture.source_sh1_m,
            atol=THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['source_sh2_m'],
            fixture.source_sh2_m,
            atol=THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['source_sh3_m'],
            fixture.source_sh3_m,
            atol=THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['source_weathering_correction_s'],
            fixture.source_wcor_s,
            atol=STATIC_ATOL_S,
        )
        np.testing.assert_allclose(
            data['source_total_static_s'],
            fixture.source_wcor_s,
            atol=STATIC_ATOL_S,
        )
        np.testing.assert_allclose(data['receiver_t1_s'], fixture.receiver_t1_s)
        np.testing.assert_allclose(data['receiver_t2_s'], fixture.receiver_t2_s)
        np.testing.assert_allclose(data['receiver_t3_s'], fixture.receiver_t3_s)
        np.testing.assert_allclose(
            data['receiver_sh1_m'],
            fixture.receiver_sh1_m,
            atol=THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['receiver_sh2_m'],
            fixture.receiver_sh2_m,
            atol=THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['receiver_sh3_m'],
            fixture.receiver_sh3_m,
            atol=THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['receiver_weathering_correction_s'],
            fixture.receiver_wcor_s,
            atol=STATIC_ATOL_S,
        )
        np.testing.assert_allclose(
            data['receiver_total_static_s'],
            fixture.receiver_wcor_s,
            atol=STATIC_ATOL_S,
        )
        assert str(data['sign_convention']) == _SIGN_CONVENTION

    _assert_three_layer_time_terms_and_thicknesses_match_truth(fixture, workflow)


def test_three_layer_trace_shift_matches_source_plus_receiver_components(
    tmp_path: Path,
) -> None:
    fixture = _make_three_layer_solver_fixture()
    job_dir = tmp_path / 'job'

    workflow = _compute_three_layer_solver_workflow(fixture, job_dir=job_dir)
    result = workflow.datum_result

    np.testing.assert_allclose(
        result.source_refraction_shift_s_sorted,
        _map_endpoint_values_to_trace_nodes(
            endpoint_node_id=fixture.source_node_id,
            endpoint_values=fixture.source_wcor_s,
            trace_node_id=fixture.input_model.source_node_id_sorted,
        ),
        atol=STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        result.receiver_refraction_shift_s_sorted,
        _map_endpoint_values_to_trace_nodes(
            endpoint_node_id=fixture.receiver_node_id,
            endpoint_values=fixture.receiver_wcor_s,
            trace_node_id=fixture.input_model.receiver_node_id_sorted,
        ),
        atol=STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        result.refraction_trace_shift_s_sorted,
        fixture.trace_shift_s_sorted,
        atol=_TRACE_SHIFT_ATOL_S,
    )
    np.testing.assert_allclose(
        result.refraction_trace_shift_s_sorted,
        result.source_refraction_shift_s_sorted
        + result.receiver_refraction_shift_s_sorted,
        atol=STATIC_ATOL_S,
    )
    assert np.all(result.trace_static_valid_mask_sorted)
    assert np.all(result.trace_static_status_sorted == 'ok')

    rows = _read_csv(job_dir / REFRACTION_STATICS_CSV_NAME)
    for row, expected_shift_s in zip(
        rows,
        fixture.trace_shift_s_sorted.tolist(),
        strict=True,
    ):
        trace_shift_ms = float(row['refraction_trace_shift_ms'])
        assert trace_shift_ms / 1000.0 == pytest.approx(
            expected_shift_s,
            abs=STATIC_ATOL_S,
        )
        assert trace_shift_ms == pytest.approx(
            float(row['source_refraction_shift_ms'])
            + float(row['receiver_refraction_shift_ms'])
        )

    raw_event_time_s = 1.0
    corrected_event_time_s = raw_event_time_s + float(
        result.refraction_trace_shift_s_sorted[0]
    )
    assert corrected_event_time_s < raw_event_time_s


def test_three_layer_trace_shift_is_source_plus_receiver_plus_datum() -> None:
    datum = RefractionStaticDatumRequest(
        mode='floating_and_flat',
        floating_datum_mode='constant',
        floating_datum_elevation_m=75.0,
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


def _make_three_layer_solver_fixture() -> _ThreeLayerSolverFixture:
    n_nodes = 20
    inline_m = np.arange(n_nodes, dtype=np.float64) * 250.0
    node_id = np.arange(n_nodes, dtype=np.int64)
    station_index = np.arange(n_nodes, dtype=np.float64)
    sh1_m = 8.0 + 0.4 * (station_index % 5.0) + 0.02 * station_index
    sh2_m = 14.0 + 0.5 * (station_index % 4.0) + 0.03 * station_index
    sh3_m = 19.0 + 0.45 * (station_index % 6.0) + 0.025 * station_index
    t1_s, t2_s, t3_s = _three_layer_time_terms_s(
        sh1_m=sh1_m,
        sh2_m=sh2_m,
        sh3_m=sh3_m,
    )
    wcor_s = _three_layer_wcor_s(sh1_m=sh1_m, sh2_m=sh2_m, sh3_m=sh3_m)
    rows = _three_layer_observation_rows(
        inline_m=inline_m,
        t1_s=t1_s,
        t2_s=t2_s,
        t3_s=t3_s,
    )
    source_index = np.asarray([row[0] for row in rows], dtype=np.int64)
    receiver_index = np.asarray([row[1] for row in rows], dtype=np.int64)
    offset_m = np.asarray([row[2] for row in rows], dtype=np.float64)
    pick_time_s = np.asarray([row[3] for row in rows], dtype=np.float64)
    layer_kind = np.asarray([row[4] for row in rows], dtype='<U16')
    n_traces = int(pick_time_s.shape[0])
    source_endpoint_key = np.asarray(
        [f's:{index}' for index in source_index.tolist()],
        dtype=object,
    )
    receiver_endpoint_key = np.asarray(
        [f'r:{index}' for index in receiver_index.tolist()],
        dtype=object,
    )
    source_positions = _first_occurrence_positions(source_endpoint_key)
    receiver_positions = _first_occurrence_positions(receiver_endpoint_key)
    source_node_id = source_index[source_positions]
    receiver_node_id = receiver_index[receiver_positions]
    node_elevation_m = 90.0 + 0.002 * inline_m
    endpoint_table = _endpoint_table(
        node_id=node_id,
        inline_m=inline_m,
        elevation_m=node_elevation_m,
        source_index=source_index,
        receiver_index=receiver_index,
    )
    input_model = RefractionStaticInputModel(
        file_id='three-layer-global-solve-e2e',
        n_traces=n_traces,
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        pick_time_s_sorted=np.ascontiguousarray(pick_time_s, dtype=np.float64),
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=np.ones(n_traces, dtype=bool),
        source_id_sorted=1000 + source_index,
        receiver_id_sorted=2000 + receiver_index,
        source_x_m_sorted=np.ascontiguousarray(inline_m[source_index], dtype=np.float64),
        source_y_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_x_m_sorted=np.ascontiguousarray(
            inline_m[receiver_index],
            dtype=np.float64,
        ),
        receiver_y_m_sorted=np.zeros(n_traces, dtype=np.float64),
        source_elevation_m_sorted=np.ascontiguousarray(
            node_elevation_m[source_index],
            dtype=np.float64,
        ),
        receiver_elevation_m_sorted=np.ascontiguousarray(
            node_elevation_m[receiver_index],
            dtype=np.float64,
        ),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=np.ascontiguousarray(offset_m, dtype=np.float64),
        offset_m_sorted=np.ascontiguousarray(offset_m, dtype=np.float64),
        distance_m_sorted=np.ascontiguousarray(offset_m, dtype=np.float64),
        source_endpoint_key_sorted=source_endpoint_key,
        receiver_endpoint_key_sorted=receiver_endpoint_key,
        source_node_id_sorted=source_index,
        receiver_node_id_sorted=receiver_index,
        node_x_m=np.ascontiguousarray(inline_m, dtype=np.float64),
        node_y_m=np.zeros(n_nodes, dtype=np.float64),
        node_elevation_m=np.ascontiguousarray(node_elevation_m, dtype=np.float64),
        node_kind=np.full(n_nodes, 'both', dtype='<U8'),
        rejection_reason_sorted=np.full(n_traces, 'ok', dtype='<U32'),
        qc={'fixture': 'three_layer_global_solve_e2e'},
        endpoint_table=endpoint_table,
        metadata={'coordinate_mode': 'grid_3d'},
    )
    return _ThreeLayerSolverFixture(
        input_model=input_model,
        model=_three_layer_solve_global_model(),
        expected_layer_kind_sorted=layer_kind,
        source_node_id=np.ascontiguousarray(source_node_id, dtype=np.int64),
        receiver_node_id=np.ascontiguousarray(receiver_node_id, dtype=np.int64),
        source_t1_s=np.ascontiguousarray(t1_s[source_node_id], dtype=np.float64),
        source_t2_s=np.ascontiguousarray(t2_s[source_node_id], dtype=np.float64),
        source_t3_s=np.ascontiguousarray(t3_s[source_node_id], dtype=np.float64),
        receiver_t1_s=np.ascontiguousarray(t1_s[receiver_node_id], dtype=np.float64),
        receiver_t2_s=np.ascontiguousarray(t2_s[receiver_node_id], dtype=np.float64),
        receiver_t3_s=np.ascontiguousarray(t3_s[receiver_node_id], dtype=np.float64),
        source_sh1_m=np.ascontiguousarray(sh1_m[source_node_id], dtype=np.float64),
        source_sh2_m=np.ascontiguousarray(sh2_m[source_node_id], dtype=np.float64),
        source_sh3_m=np.ascontiguousarray(sh3_m[source_node_id], dtype=np.float64),
        receiver_sh1_m=np.ascontiguousarray(sh1_m[receiver_node_id], dtype=np.float64),
        receiver_sh2_m=np.ascontiguousarray(sh2_m[receiver_node_id], dtype=np.float64),
        receiver_sh3_m=np.ascontiguousarray(sh3_m[receiver_node_id], dtype=np.float64),
        source_wcor_s=np.ascontiguousarray(wcor_s[source_node_id], dtype=np.float64),
        receiver_wcor_s=np.ascontiguousarray(wcor_s[receiver_node_id], dtype=np.float64),
        trace_shift_s_sorted=np.ascontiguousarray(
            wcor_s[source_index] + wcor_s[receiver_index],
            dtype=np.float64,
        ),
    )


def _three_layer_solve_global_model() -> RefractionStaticModelRequest:
    return RefractionStaticModelRequest.model_validate(
        {
            'method': 'multilayer_time_term',
            'first_layer': {
                'mode': 'constant',
                'weathering_velocity_m_s': SYNTHETIC_MULTILAYER_V1_M_S,
            },
            'layers': [
                {
                    'kind': 'v2_t1',
                    'enabled': True,
                    'min_offset_m': _V2_MIN_OFFSET_M,
                    'max_offset_m': _V2_MAX_OFFSET_M,
                    'velocity_mode': 'solve_global',
                    'initial_velocity_m_s': _INITIAL_V2_M_S,
                    'min_velocity_m_s': 1600.0,
                    'max_velocity_m_s': 3200.0,
                },
                {
                    'kind': 'v3_t2',
                    'enabled': True,
                    'min_offset_m': _V3_MIN_OFFSET_M,
                    'max_offset_m': _V3_MAX_OFFSET_M,
                    'velocity_mode': 'solve_global',
                    'initial_velocity_m_s': _INITIAL_V3_M_S,
                    'min_velocity_m_s': 2600.0,
                    'max_velocity_m_s': 4800.0,
                },
                {
                    'kind': 'vsub_t3',
                    'enabled': True,
                    'min_offset_m': _VSUB_MIN_OFFSET_M,
                    'max_offset_m': None,
                    'velocity_mode': 'solve_global',
                    'initial_velocity_m_s': _INITIAL_VSUB_M_S,
                    'min_velocity_m_s': 3600.0,
                    'max_velocity_m_s': 6200.0,
                },
            ],
        }
    )


def _compute_three_layer_solver_workflow(
    fixture: _ThreeLayerSolverFixture,
    *,
    job_dir: Path | None = None,
) -> RefractionMultiLayerStaticsWorkflowResult:
    return compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=fixture.input_model,
        model=fixture.model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            min_picks_per_node=1,
            max_abs_half_intercept_time_ms=500.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=500.0),
        resolved_first_layer=resolved_first_layer(),
        job_dir=job_dir,
    )


def _three_layer_time_terms_s(
    *,
    sh1_m: np.ndarray,
    sh2_m: np.ndarray,
    sh3_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v1 = SYNTHETIC_MULTILAYER_V1_M_S
    v2 = SYNTHETIC_MULTILAYER_V2_M_S
    v3 = SYNTHETIC_MULTILAYER_V3_M_S
    vsub = SYNTHETIC_MULTILAYER_VSUB_M_S
    t1_s = sh1_m * np.sqrt(1.0 - (v1 / v2) ** 2) / v1
    t2_s = (
        sh1_m * np.sqrt(1.0 - (v1 / v3) ** 2) / v1
        + sh2_m * np.sqrt(1.0 - (v2 / v3) ** 2) / v2
    )
    t3_s = (
        sh1_m * np.sqrt(1.0 - (v1 / vsub) ** 2) / v1
        + sh2_m * np.sqrt(1.0 - (v2 / vsub) ** 2) / v2
        + sh3_m * np.sqrt(1.0 - (v3 / vsub) ** 2) / v3
    )
    return (
        np.ascontiguousarray(t1_s, dtype=np.float64),
        np.ascontiguousarray(t2_s, dtype=np.float64),
        np.ascontiguousarray(t3_s, dtype=np.float64),
    )


def _three_layer_wcor_s(
    *,
    sh1_m: np.ndarray,
    sh2_m: np.ndarray,
    sh3_m: np.ndarray,
) -> np.ndarray:
    v1 = SYNTHETIC_MULTILAYER_V1_M_S
    v2 = SYNTHETIC_MULTILAYER_V2_M_S
    v3 = SYNTHETIC_MULTILAYER_V3_M_S
    vsub = SYNTHETIC_MULTILAYER_VSUB_M_S
    return np.ascontiguousarray(
        sh1_m * (1.0 / vsub - 1.0 / v1)
        + sh2_m * (1.0 / vsub - 1.0 / v2)
        + sh3_m * (1.0 / vsub - 1.0 / v3),
        dtype=np.float64,
    )


def _three_layer_observation_rows(
    *,
    inline_m: np.ndarray,
    t1_s: np.ndarray,
    t2_s: np.ndarray,
    t3_s: np.ndarray,
) -> list[tuple[int, int, float, float, str]]:
    rows: list[tuple[int, int, float, float, str]] = []
    for source_index, source_inline_m in enumerate(inline_m.tolist()):
        for receiver_index, receiver_inline_m in enumerate(inline_m.tolist()):
            if source_index == receiver_index:
                continue
            offset_m = abs(float(receiver_inline_m) - float(source_inline_m))
            if _V2_MIN_OFFSET_M <= offset_m < _V2_MAX_OFFSET_M:
                rows.append(
                    (
                        source_index,
                        receiver_index,
                        offset_m,
                        float(
                            t1_s[source_index]
                            + t1_s[receiver_index]
                            + offset_m / SYNTHETIC_MULTILAYER_V2_M_S
                        ),
                        'v2_t1',
                    )
                )
            elif _V3_MIN_OFFSET_M <= offset_m < _V3_MAX_OFFSET_M:
                rows.append(
                    (
                        source_index,
                        receiver_index,
                        offset_m,
                        float(
                            t2_s[source_index]
                            + t2_s[receiver_index]
                            + offset_m / SYNTHETIC_MULTILAYER_V3_M_S
                        ),
                        'v3_t2',
                    )
                )
            elif offset_m >= _VSUB_MIN_OFFSET_M:
                rows.append(
                    (
                        source_index,
                        receiver_index,
                        offset_m,
                        float(
                            t3_s[source_index]
                            + t3_s[receiver_index]
                            + offset_m / SYNTHETIC_MULTILAYER_VSUB_M_S
                        ),
                        'vsub_t3',
                    )
                )
    return rows


def _endpoint_table(
    *,
    node_id: np.ndarray,
    inline_m: np.ndarray,
    elevation_m: np.ndarray,
    source_index: np.ndarray,
    receiver_index: np.ndarray,
) -> RefractionEndpointTable:
    pick_count = np.zeros(node_id.shape, dtype=np.int64)
    for source_node, receiver_node in zip(
        source_index.tolist(),
        receiver_index.tolist(),
        strict=True,
    ):
        pick_count[int(source_node)] += 1
        pick_count[int(receiver_node)] += 1
    return RefractionEndpointTable(
        node_id=node_id,
        endpoint_id=node_id.copy(),
        x_m=np.ascontiguousarray(inline_m, dtype=np.float64),
        y_m=np.zeros(node_id.shape, dtype=np.float64),
        elevation_m=np.ascontiguousarray(elevation_m, dtype=np.float64),
        kind=np.full(node_id.shape, 'both', dtype='<U8'),
        pick_count=pick_count,
    )


def _assert_layer_masks_select_expected_three_layer_branches(
    fixture: _ThreeLayerSolverFixture,
) -> None:
    masks = build_refraction_layer_observation_masks(
        input_model=fixture.input_model,
        model=fixture.model,
    )
    for layer_kind in ('v2_t1', 'v3_t2', 'vsub_t3'):
        expected = fixture.expected_layer_kind_sorted == layer_kind
        np.testing.assert_array_equal(
            masks.layer_used_mask_sorted[layer_kind],
            expected,
        )
        assert masks.layer_observation_count[layer_kind] == int(
            np.count_nonzero(expected)
        )


def _assert_global_layer_result(
    layer: RefractionLayerSolveResult,
    *,
    expected_velocity_m_s: float,
) -> None:
    assert layer.velocity_mode == 'solve_global'
    assert layer.global_velocity_m_s == pytest.approx(
        expected_velocity_m_s,
        rel=_VELOCITY_RTOL,
    )
    assert layer.global_slowness_s_per_m == pytest.approx(
        1.0 / expected_velocity_m_s,
        rel=_VELOCITY_RTOL,
    )
    np.testing.assert_allclose(
        layer.trace_residual_s_sorted[layer.used_observation_mask_sorted],
        0.0,
        atol=STATIC_ATOL_S,
    )


def _assert_three_layer_time_terms_and_thicknesses_match_truth(
    fixture: _ThreeLayerSolverFixture,
    workflow: RefractionMultiLayerStaticsWorkflowResult,
) -> None:
    components = workflow.components
    result = workflow.datum_result
    np.testing.assert_allclose(components.source_t1_s, fixture.source_t1_s)
    np.testing.assert_allclose(components.source_t2_s, fixture.source_t2_s)
    np.testing.assert_allclose(components.source_t3_s, fixture.source_t3_s)
    np.testing.assert_allclose(components.receiver_t1_s, fixture.receiver_t1_s)
    np.testing.assert_allclose(components.receiver_t2_s, fixture.receiver_t2_s)
    np.testing.assert_allclose(components.receiver_t3_s, fixture.receiver_t3_s)
    np.testing.assert_allclose(
        components.source_sh1_m,
        fixture.source_sh1_m,
        atol=THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        components.source_sh2_m,
        fixture.source_sh2_m,
        atol=THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        components.source_sh3_m,
        fixture.source_sh3_m,
        atol=THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        components.receiver_sh1_m,
        fixture.receiver_sh1_m,
        atol=THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        components.receiver_sh2_m,
        fixture.receiver_sh2_m,
        atol=THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        components.receiver_sh3_m,
        fixture.receiver_sh3_m,
        atol=THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        components.source_weathering_correction_s,
        fixture.source_wcor_s,
        atol=STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        components.receiver_weathering_correction_s,
        fixture.receiver_wcor_s,
        atol=STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        result.source_weathering_thickness_m,
        fixture.source_sh1_m + fixture.source_sh2_m + fixture.source_sh3_m,
        atol=THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        result.receiver_weathering_thickness_m,
        fixture.receiver_sh1_m + fixture.receiver_sh2_m + fixture.receiver_sh3_m,
        atol=THICKNESS_ATOL_M,
    )


def _assert_static_rows_match_three_layer_truth(
    rows: list[dict[str, str]],
    *,
    expected_t1_s: np.ndarray,
    expected_t2_s: np.ndarray,
    expected_t3_s: np.ndarray,
    expected_sh1_m: np.ndarray,
    expected_sh2_m: np.ndarray,
    expected_sh3_m: np.ndarray,
    expected_wcor_s: np.ndarray,
) -> None:
    required_columns = {
        't1_ms',
        't2_ms',
        't3_ms',
        'v1_m_s',
        'v2_m_s',
        'v3_m_s',
        'vsub_m_s',
        'sh1_weathering_thickness_m',
        'sh2_weathering_thickness_m',
        'sh3_weathering_thickness_m',
        'total_weathering_thickness_m',
        'weathering_correction_ms',
        'total_static_ms',
        'total_applied_shift_ms',
        'static_status',
        'sign_convention',
    }
    assert required_columns <= set(rows[0])
    for index, row in enumerate(rows):
        assert row['static_status'] == 'ok'
        assert row['sign_convention'] == _SIGN_CONVENTION
        assert float(row['t1_ms']) / 1000.0 == pytest.approx(
            expected_t1_s[index],
            abs=STATIC_ATOL_S,
        )
        assert float(row['t2_ms']) / 1000.0 == pytest.approx(
            expected_t2_s[index],
            abs=STATIC_ATOL_S,
        )
        assert float(row['t3_ms']) / 1000.0 == pytest.approx(
            expected_t3_s[index],
            abs=STATIC_ATOL_S,
        )
        assert float(row['v1_m_s']) == pytest.approx(SYNTHETIC_MULTILAYER_V1_M_S)
        assert float(row['v2_m_s']) == pytest.approx(
            SYNTHETIC_MULTILAYER_V2_M_S,
            rel=_VELOCITY_RTOL,
        )
        assert float(row['v3_m_s']) == pytest.approx(
            SYNTHETIC_MULTILAYER_V3_M_S,
            rel=_VELOCITY_RTOL,
        )
        assert float(row['vsub_m_s']) == pytest.approx(
            SYNTHETIC_MULTILAYER_VSUB_M_S,
            rel=_VELOCITY_RTOL,
        )
        assert float(row['sh1_weathering_thickness_m']) == pytest.approx(
            expected_sh1_m[index],
            abs=THICKNESS_ATOL_M,
        )
        assert float(row['sh2_weathering_thickness_m']) == pytest.approx(
            expected_sh2_m[index],
            abs=THICKNESS_ATOL_M,
        )
        assert float(row['sh3_weathering_thickness_m']) == pytest.approx(
            expected_sh3_m[index],
            abs=THICKNESS_ATOL_M,
        )
        expected_total_m = (
            expected_sh1_m[index] + expected_sh2_m[index] + expected_sh3_m[index]
        )
        assert float(row['total_weathering_thickness_m']) == pytest.approx(
            expected_total_m,
            abs=THICKNESS_ATOL_M,
        )
        assert float(row['weathering_correction_ms']) / 1000.0 == pytest.approx(
            expected_wcor_s[index],
            abs=STATIC_ATOL_S,
        )
        assert float(row['total_static_ms']) / 1000.0 == pytest.approx(
            expected_wcor_s[index],
            abs=STATIC_ATOL_S,
        )
        assert float(row['total_applied_shift_ms']) / 1000.0 == pytest.approx(
            expected_wcor_s[index],
            abs=STATIC_ATOL_S,
        )


def _layer(
    workflow: RefractionMultiLayerStaticsWorkflowResult,
    layer_kind: str,
) -> RefractionLayerSolveResult:
    for layer in workflow.solve_result.layer_results:
        if layer.layer_kind == layer_kind:
            return layer
    raise AssertionError(f'{layer_kind} layer result was not returned')


def _map_endpoint_values_to_trace_nodes(
    *,
    endpoint_node_id: np.ndarray,
    endpoint_values: np.ndarray,
    trace_node_id: np.ndarray,
) -> np.ndarray:
    values_by_node = {
        int(node_id): float(value)
        for node_id, value in zip(
            endpoint_node_id.tolist(),
            endpoint_values.tolist(),
            strict=True,
        )
    }
    return np.asarray(
        [values_by_node[int(node_id)] for node_id in trace_node_id.tolist()],
        dtype=np.float64,
    )


def _first_occurrence_positions(values: np.ndarray) -> np.ndarray:
    seen: set[str] = set()
    positions: list[int] = []
    for index, value in enumerate(values.tolist()):
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        positions.append(index)
    return np.asarray(positions, dtype=np.int64)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))
