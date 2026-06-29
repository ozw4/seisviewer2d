from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from app.api.schemas import (
    RefractionStaticApplyOptions,
    RefractionStaticApplyRequest,
    RefractionStaticConversionRequest,
    RefractionStaticDatumRequest,
    RefractionStaticLinkageRequest,
    RefractionStaticModelRequest,
    RefractionStaticPickSourceRequest,
    RefractionStaticSolverRequest,
)
from app.statics.refraction.artifacts import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    build_refraction_cell_solver_history_rows,
    build_refraction_refractor_velocity_grid_arrays,
    REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_STATICS_CSV_NAME,
    SIGN_CONVENTION,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    write_near_surface_model_csv,
    write_refraction_statics_csv,
    write_receiver_static_table_csv,
    write_source_receiver_static_table_npz,
    write_source_static_table_csv,
)
from app.statics.refraction.application.datum import build_refraction_datum_statics
from app.statics.refraction.application import multilayer_service
from app.statics.refraction.application.core_options import (
    layer_observation_masks_from_input_model as build_refraction_layer_observation_masks,
)
from app.statics.refraction.application.multilayer_service import (
    _components_from_replacement,
    build_refraction_multilayer_weathering_replacement_statics,
    compute_refraction_multilayer_datum_statics_from_input_model,
)
from seis_statics.refraction.t1lsst import (
    compute_t1lsst_2layer_thicknesses,
    compute_t1lsst_2layer_weathering_correction,
)
from app.statics.refraction.contracts.result_types import (
    RefractionEndpointTable,
    RefractionDatumStaticsResult,
    RefractionLayerSolveResult,
    RefractionMultiLayerSolveResult,
    RefractionMultiLayerStaticComponents,
    RefractionStaticInputModel,
    ResolvedRefractionFirstLayer,
)
from app.tests._refraction_multilayer_synthetic import (
    SYNTHETIC_MULTILAYER_V1_M_S,
    SYNTHETIC_MULTILAYER_V2_M_S,
    SYNTHETIC_MULTILAYER_V3_M_S,
)

CoordinateMode = Literal['grid_3d', 'line_2d_projected']
VelocityMode = Literal['solve_global', 'solve_cell']

_V2_MIN_OFFSET_M = 200.0
_V2_MAX_OFFSET_M = 1000.0
_V3_MIN_OFFSET_M = 1000.0
_V3_MAX_OFFSET_M = 1800.0
_TIME_ATOL_S = 1.0e-8
_THICKNESS_ATOL_M = 1.0e-5
_STATIC_ATOL_S = 1.0e-8
_CELL_VELOCITY_ARTIFACT_NAMES = {
    REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
}


@dataclass(frozen=True)
class _TwoLayerE2EFixture:
    input_model: RefractionStaticInputModel
    model: RefractionStaticModelRequest
    expected_layer_kind_sorted: np.ndarray
    source_endpoint_key: np.ndarray
    receiver_endpoint_key: np.ndarray
    source_node_id: np.ndarray
    receiver_node_id: np.ndarray
    source_t1_s: np.ndarray
    source_t2_s: np.ndarray
    receiver_t1_s: np.ndarray
    receiver_t2_s: np.ndarray
    source_sh1_m: np.ndarray
    source_sh2_m: np.ndarray
    receiver_sh1_m: np.ndarray
    receiver_sh2_m: np.ndarray
    source_weathering_correction_s: np.ndarray
    receiver_weathering_correction_s: np.ndarray
    trace_shift_s_sorted: np.ndarray


@dataclass(frozen=True)
class _StaticOutputs:
    solve_result: RefractionMultiLayerSolveResult
    components: RefractionMultiLayerStaticComponents
    result: RefractionDatumStaticsResult
    source_rows: list[dict[str, str]]
    receiver_rows: list[dict[str, str]]
    source_receiver_arrays: dict[str, np.ndarray]
    trace_rows: list[dict[str, str]]
    trace_shift_s_sorted: np.ndarray


def test_two_layer_global_e2e_recovers_solver_conversion_tables_and_trace_shift(
    tmp_path: Path,
) -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='grid_3d',
        v2_velocity_mode='solve_global',
    )

    outputs = _compute_static_outputs(fixture=fixture, tmp_path=tmp_path)
    result = outputs.solve_result

    _assert_layer_masks_select_expected_branches(fixture)
    _assert_global_layer_result(
        _layer(result, 'v2_t1'),
        expected_velocity_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
    )
    _assert_global_layer_result(
        _layer(result, 'v3_t2'),
        expected_velocity_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
    )
    _assert_two_layer_outputs_match_truth(fixture, outputs)


def test_two_layer_conversion_uses_source_and_receiver_endpoint_terms() -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='grid_3d',
        v2_velocity_mode='solve_global',
    )
    workflow = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=fixture.input_model,
        model=fixture.model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
    )
    solve = workflow.solve_result
    source_t1 = fixture.source_t1_s + np.linspace(
        0.0002,
        0.0008,
        fixture.source_t1_s.size,
    )
    receiver_t1 = fixture.receiver_t1_s + np.linspace(
        0.0009,
        0.0003,
        fixture.receiver_t1_s.size,
    )
    source_t2 = fixture.source_t2_s + np.linspace(
        0.0015,
        0.0021,
        fixture.source_t2_s.size,
    )
    receiver_t2 = fixture.receiver_t2_s + np.linspace(
        0.0022,
        0.0016,
        fixture.receiver_t2_s.size,
    )
    patched_layers = tuple(
        replace(layer, source_time_term_s=source_t1, receiver_time_term_s=receiver_t1)
        if layer.layer_kind == 'v2_t1'
        else replace(
            layer,
            source_time_term_s=source_t2,
            receiver_time_term_s=receiver_t2,
        )
        for layer in solve.layer_results
    )
    patched_solve = replace(solve, layer_results=patched_layers)

    replacement = build_refraction_multilayer_weathering_replacement_statics(
        input_model=fixture.input_model,
        model=fixture.model,
        solve_result=patched_solve,
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
    )

    source_sh1, source_sh2 = compute_t1lsst_2layer_thicknesses(
        t1_s=source_t1,
        t2_s=source_t2,
        v1_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        v2_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
        v3_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
    )
    receiver_sh1, receiver_sh2 = compute_t1lsst_2layer_thicknesses(
        t1_s=receiver_t1,
        t2_s=receiver_t2,
        v1_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        v2_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
        v3_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
    )
    expected_source_shift = compute_t1lsst_2layer_weathering_correction(
        sh1_m=source_sh1,
        sh2_m=source_sh2,
        v1_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        v2_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
        v3_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
    )
    expected_receiver_shift = compute_t1lsst_2layer_weathering_correction(
        sh1_m=receiver_sh1,
        sh2_m=receiver_sh2,
        v1_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        v2_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
        v3_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
    )
    expected_source_sorted = _values_by_sorted_key(
        fixture.input_model.source_endpoint_key_sorted,
        fixture.source_endpoint_key,
        expected_source_shift,
    )
    expected_receiver_sorted = _values_by_sorted_key(
        fixture.input_model.receiver_endpoint_key_sorted,
        fixture.receiver_endpoint_key,
        expected_receiver_shift,
    )

    np.testing.assert_allclose(
        replacement.source_weathering_replacement_shift_s,
        expected_source_shift,
        atol=_STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        replacement.receiver_weathering_replacement_shift_s,
        expected_receiver_shift,
        atol=_STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        replacement.source_weathering_replacement_shift_s_sorted,
        expected_source_sorted,
        atol=_STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        replacement.receiver_weathering_replacement_shift_s_sorted,
        expected_receiver_sorted,
        atol=_STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        replacement.weathering_replacement_trace_shift_s_sorted,
        expected_source_sorted + expected_receiver_sorted,
        atol=_STATIC_ATOL_S,
    )


def test_two_layer_line_projected_local_v2_global_v3_e2e_recovers_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='line_2d_projected',
        v2_velocity_mode='solve_cell',
    )
    original_conversion = multilayer_service.core_build_refraction_multilayer_conversion
    node_v2_velocity: list[np.ndarray] = []

    def _capture_node_v2_velocity(**kwargs):
        core_input = kwargs['input_model']
        core_solve = kwargs['solve_result']
        if int(core_input.n_traces) == int(fixture.input_model.node_x_m.shape[0]):
            v2_layer = core_solve.layer_result_by_kind['v2_t1']
            node_v2_velocity.append(
                np.asarray(v2_layer.velocity_m_s_sorted, dtype=np.float64).copy()
            )
        return original_conversion(**kwargs)

    monkeypatch.setattr(
        multilayer_service,
        'core_build_refraction_multilayer_conversion',
        _capture_node_v2_velocity,
    )

    outputs = _compute_static_outputs(fixture=fixture, tmp_path=tmp_path)
    result = outputs.solve_result
    v2_layer = _layer(result, 'v2_t1')

    assert fixture.model.refractor_cell is not None
    assert fixture.model.refractor_cell.coordinate_mode == 'line_2d_projected'
    assert v2_layer.velocity_mode == 'solve_cell'
    assert v2_layer.cell_velocity_m_s is not None
    assert node_v2_velocity
    node_v2 = node_v2_velocity[-1]
    assert np.any(np.isfinite(node_v2))
    np.testing.assert_allclose(
        node_v2[np.isfinite(node_v2)],
        SYNTHETIC_MULTILAYER_V2_M_S,
        rtol=1.0e-8,
        atol=1.0e-4,
    )
    np.testing.assert_allclose(
        v2_layer.cell_velocity_m_s,
        SYNTHETIC_MULTILAYER_V2_M_S,
        rtol=1.0e-8,
        atol=1.0e-4,
    )
    _assert_global_layer_result(
        _layer(result, 'v3_t2'),
        expected_velocity_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
    )
    _assert_layer_masks_select_expected_branches(fixture)
    _assert_two_layer_outputs_match_truth(fixture, outputs)


def test_two_layer_job_dir_writes_core_artifacts_and_solution_npz_contract(
    tmp_path: Path,
) -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='grid_3d',
        v2_velocity_mode='solve_global',
    )
    job_dir = tmp_path / 'job'

    compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=fixture.input_model,
        model=fixture.model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
        job_dir=job_dir,
    )

    for artifact_name in (
        REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        REFRACTION_STATIC_QC_JSON_NAME,
        REFRACTION_STATICS_CSV_NAME,
        NEAR_SURFACE_MODEL_CSV_NAME,
        FIRST_BREAK_RESIDUALS_CSV_NAME,
        REFRACTION_STATIC_COMPONENTS_CSV_NAME,
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    ):
        assert (job_dir / artifact_name).is_file()

    source_rows = _read_csv(job_dir / SOURCE_STATIC_TABLE_CSV_NAME)
    receiver_rows = _read_csv(job_dir / RECEIVER_STATIC_TABLE_CSV_NAME)
    required_columns = {
        't2_ms',
        'v3_m_s',
        'sh2_weathering_thickness_m',
        'layer1_base_elevation_m',
        'final_refractor_elevation_m',
    }
    assert required_columns <= set(source_rows[0])
    assert required_columns <= set(receiver_rows[0])

    near_surface_rows = _read_csv(job_dir / NEAR_SURFACE_MODEL_CSV_NAME)
    near_surface_columns = set(near_surface_rows[0])
    assert {
        'sh1_weathering_thickness_m',
        'sh2_weathering_thickness_m',
        'layer1_base_elevation_m',
        'final_refractor_elevation_m',
    } <= near_surface_columns
    for row, sh1_m, sh2_m in zip(
        near_surface_rows,
        _fixture_node_sh1_m(fixture).tolist(),
        _fixture_node_sh2_m(fixture).tolist(),
        strict=True,
    ):
        expected_total = sh1_m + sh2_m
        assert float(row['weathering_thickness_m']) == pytest.approx(
            expected_total,
            abs=_THICKNESS_ATOL_M,
        )
        assert float(row['layer1_base_elevation_m']) == pytest.approx(
            -sh1_m,
            abs=_THICKNESS_ATOL_M,
        )
        assert float(row['final_refractor_elevation_m']) == pytest.approx(
            -expected_total,
            abs=_THICKNESS_ATOL_M,
        )
        assert float(row['refractor_elevation_m']) == pytest.approx(
            -expected_total,
            abs=_THICKNESS_ATOL_M,
        )

    with np.load(job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        expected_arrays = {
            'node_sh2_weathering_thickness_m',
            'node_layer1_base_elevation_m',
            'node_final_refractor_elevation_m',
            'source_t2_time_s',
            'source_v3_m_s',
            'source_sh1_weathering_thickness_m',
            'source_sh2_weathering_thickness_m',
            'source_layer1_base_elevation_m',
            'source_final_refractor_elevation_m',
            'receiver_t2_time_s',
            'receiver_v3_m_s',
            'receiver_sh1_weathering_thickness_m',
            'receiver_sh2_weathering_thickness_m',
            'receiver_layer1_base_elevation_m',
            'receiver_final_refractor_elevation_m',
        }
        assert expected_arrays <= set(data.files)
        np.testing.assert_allclose(
            data['node_sh2_weathering_thickness_m'],
            _fixture_node_sh2_m(fixture),
            atol=_THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['node_layer1_base_elevation_m'],
            -_fixture_node_sh1_m(fixture),
            atol=_THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['node_final_refractor_elevation_m'],
            -(_fixture_node_sh1_m(fixture) + _fixture_node_sh2_m(fixture)),
            atol=_THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(data['source_t2_time_s'], fixture.source_t2_s)
        np.testing.assert_allclose(
            data['source_v3_m_s'],
            SYNTHETIC_MULTILAYER_V3_M_S,
            rtol=1.0e-9,
        )
        np.testing.assert_allclose(
            data['source_sh1_weathering_thickness_m'],
            fixture.source_sh1_m,
            atol=_THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['source_sh2_weathering_thickness_m'],
            fixture.source_sh2_m,
            atol=_THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['source_layer1_base_elevation_m'],
            -fixture.source_sh1_m,
            atol=_THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['source_final_refractor_elevation_m'],
            -(fixture.source_sh1_m + fixture.source_sh2_m),
            atol=_THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(data['receiver_t2_time_s'], fixture.receiver_t2_s)
        np.testing.assert_allclose(
            data['receiver_v3_m_s'],
            SYNTHETIC_MULTILAYER_V3_M_S,
            rtol=1.0e-9,
        )
        np.testing.assert_allclose(
            data['receiver_sh1_weathering_thickness_m'],
            fixture.receiver_sh1_m,
            atol=_THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['receiver_sh2_weathering_thickness_m'],
            fixture.receiver_sh2_m,
            atol=_THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['receiver_layer1_base_elevation_m'],
            -fixture.receiver_sh1_m,
            atol=_THICKNESS_ATOL_M,
        )
        np.testing.assert_allclose(
            data['receiver_final_refractor_elevation_m'],
            -(fixture.receiver_sh1_m + fixture.receiver_sh2_m),
            atol=_THICKNESS_ATOL_M,
        )

    for artifact_name in _CELL_VELOCITY_ARTIFACT_NAMES:
        assert not (job_dir / artifact_name).exists()
    manifest = json.loads(
        (job_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(
            encoding='utf-8'
        )
    )
    artifact_names = {item['name'] for item in manifest['artifacts']}
    assert not _CELL_VELOCITY_ARTIFACT_NAMES.intersection(artifact_names)


def test_two_layer_local_v2_job_dir_writes_cell_velocity_artifacts(
    tmp_path: Path,
) -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='line_2d_projected',
        v2_velocity_mode='solve_cell',
    )
    job_dir = tmp_path / 'job'

    workflow = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=fixture.input_model,
        model=fixture.model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
        job_dir=job_dir,
    )

    assert workflow.datum_result.bedrock_velocity_mode == 'solve_global'
    for artifact_name in _CELL_VELOCITY_ARTIFACT_NAMES:
        assert (job_dir / artifact_name).is_file()

    manifest = json.loads(
        (job_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(
            encoding='utf-8'
        )
    )
    artifact_names = {item['name'] for item in manifest['artifacts']}
    assert _CELL_VELOCITY_ARTIFACT_NAMES <= artifact_names

    qc = json.loads(
        (job_dir / REFRACTION_STATIC_QC_JSON_NAME).read_text(encoding='utf-8')
    )
    assert qc['velocity']['cell_velocity_layer_kind'] == 'v2_t1'
    assert qc['velocity']['cell_velocity_component'] == 'v2'
    assert qc['refractor_velocity_cells']['cell_velocity_layer_kind'] == 'v2_t1'
    assert qc['refractor_velocity_cells']['cell_velocity_component'] == 'v2'

    cell_qc = json.loads(
        (job_dir / REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME).read_text(
            encoding='utf-8'
        )
    )
    assert cell_qc['cell_velocity_layer_kind'] == 'v2_t1'
    assert cell_qc['cell_velocity_component'] == 'v2'

    cell_rows = _read_csv(job_dir / REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME)
    assert cell_rows
    assert {row['cell_velocity_layer_kind'] for row in cell_rows} == {'v2_t1'}
    assert {row['cell_velocity_component'] for row in cell_rows} == {'v2'}

    expected_v2_row_count = int(
        np.count_nonzero(fixture.expected_layer_kind_sorted == 'v2_t1')
    )
    boundary_row = fixture.input_model.distance_m_sorted == _V2_MAX_OFFSET_M
    assert np.any(boundary_row)
    assert np.all(fixture.expected_layer_kind_sorted[boundary_row] == 'v3_t2')

    with np.load(job_dir / REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME) as data:
        assert set(data['cell_velocity_layer_kind'].astype(str).tolist()) == {
            'v2_t1'
        }
        assert set(data['cell_velocity_component'].astype(str).tolist()) == {'v2'}
        np.testing.assert_allclose(
            data['initial_v2_m_s'],
            SYNTHETIC_MULTILAYER_V2_M_S,
        )
        assert int(np.sum(data['n_observations_per_cell'])) == expected_v2_row_count
        assert (
            int(np.sum(data['n_used_observations_per_cell']))
            == expected_v2_row_count
        )

    history_rows = _read_csv(job_dir / REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME)
    assert int(history_rows[-1]['n_candidate_observations']) == expected_v2_row_count
    assert int(history_rows[-1]['n_used_observations']) == expected_v2_row_count

    residual_rows = _read_csv(job_dir / FIRST_BREAK_RESIDUALS_CSV_NAME)
    v2_rows = [
        row
        for row, expected_layer in zip(
            residual_rows,
            fixture.expected_layer_kind_sorted.tolist(),
            strict=True,
        )
        if expected_layer == 'v2_t1'
    ]
    assert v2_rows
    assert all(row['cell_id'] != '' for row in v2_rows)
    assert all(row['cell_ix'] != '' for row in v2_rows)
    assert all(row['cell_iy'] != '' for row in v2_rows)


def test_two_layer_local_v2_cell_artifacts_use_row_aligned_validity() -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='line_2d_projected',
        v2_velocity_mode='solve_cell',
    )
    n_traces = int(fixture.input_model.n_traces)
    v2_indices = np.flatnonzero(fixture.expected_layer_kind_sorted == 'v2_t1')
    v3_indices = np.flatnonzero(fixture.expected_layer_kind_sorted == 'v3_t2')
    invalid_v3_index = int(v3_indices[0])
    swapped_v2_index = int(v2_indices[0])
    sorted_trace_index = np.arange(n_traces, dtype=np.int64)
    sorted_trace_index[[swapped_v2_index, invalid_v3_index]] = sorted_trace_index[
        [invalid_v3_index, swapped_v2_index]
    ]
    valid_observation = np.ones(n_traces, dtype=bool)
    valid_observation[invalid_v3_index] = False
    workflow = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=fixture.input_model,
        model=fixture.model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
    )
    result = replace(
        workflow.datum_result,
        sorted_trace_index=sorted_trace_index,
        valid_observation_mask_sorted=valid_observation,
        used_observation_mask_sorted=(
            workflow.datum_result.used_observation_mask_sorted
            & valid_observation
        ),
        row_trace_index_sorted=sorted_trace_index,
        used_row_mask=workflow.datum_result.used_row_mask & valid_observation,
    )
    req = RefractionStaticApplyRequest(
        file_id=fixture.input_model.file_id,
        pick_source=RefractionStaticPickSourceRequest(kind='manual_memmap'),
        linkage=RefractionStaticLinkageRequest(mode='none'),
        model=fixture.model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        conversion=RefractionStaticConversionRequest(
            mode='t1lsst_multilayer',
            layer_count=2,
        ),
        apply=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
    )

    expected_v2_row_count = int(
        np.count_nonzero(
            valid_observation
            & (fixture.expected_layer_kind_sorted == 'v2_t1')
        )
    )
    arrays = build_refraction_refractor_velocity_grid_arrays(
        result=result,
        req=req,
    )
    assert int(np.sum(arrays['n_observations_per_cell'])) == expected_v2_row_count

    history_rows = build_refraction_cell_solver_history_rows(
        result=result,
        req=req,
    )
    assert history_rows[-1].n_candidate_observations == expected_v2_row_count


def test_two_layer_local_v2_low_fold_endpoint_statuses_do_not_abort() -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='line_2d_projected',
        v2_velocity_mode='solve_cell',
    )
    workflow = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=fixture.input_model,
        model=fixture.model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
    )
    v2_layer = _layer(workflow.solve_result, 'v2_t1')
    assert v2_layer.active_cell_id is not None
    assert v2_layer.cell_velocity_m_s is not None
    assert v2_layer.cell_slowness_s_per_m is not None
    assert v2_layer.cell_velocity_status is not None
    cell_velocity_m_s = np.asarray(
        [2100.0, np.nan, 2600.0, np.nan, 3000.0],
        dtype=np.float64,
    )
    cell_slowness_s_per_m = np.full(cell_velocity_m_s.shape, np.nan, dtype=np.float64)
    valid_velocity = np.isfinite(cell_velocity_m_s)
    cell_slowness_s_per_m[valid_velocity] = 1.0 / cell_velocity_m_s[valid_velocity]
    cell_velocity_status = np.asarray(
        ['solved', 'inactive', 'solved', 'low_fold', 'solved'],
        dtype='<U32',
    )
    patched_v2_layer = replace(
        v2_layer,
        active_cell_id=np.asarray([0, 2, 4], dtype=np.int64),
        inactive_cell_id=np.asarray([1, 3], dtype=np.int64),
        cell_velocity_m_s=cell_velocity_m_s,
        cell_slowness_s_per_m=cell_slowness_s_per_m,
        cell_velocity_status=cell_velocity_status,
        qc={
            **v2_layer.qc,
            'low_fold_cell_id': [3],
            'n_low_fold_cells': 1,
        },
    )
    patched_solve = replace(
        workflow.solve_result,
        layer_results=(patched_v2_layer, _layer(workflow.solve_result, 'v3_t2')),
    )

    replacement = build_refraction_multilayer_weathering_replacement_statics(
        input_model=fixture.input_model,
        model=fixture.model,
        solve_result=patched_solve,
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
    )

    source_low_fold = replacement.source_v2_status == 'low_fold_v2_cell'
    receiver_low_fold = replacement.receiver_v2_status == 'low_fold_v2_cell'
    source_ok = replacement.source_v2_status == 'ok'
    receiver_ok = replacement.receiver_v2_status == 'ok'
    np.testing.assert_allclose(
        replacement.source_v2_m_s[source_ok],
        cell_velocity_m_s[replacement.source_v2_cell_id[source_ok]],
    )
    np.testing.assert_allclose(
        replacement.receiver_v2_m_s[receiver_ok],
        cell_velocity_m_s[replacement.receiver_v2_cell_id[receiver_ok]],
    )
    assert np.any(source_low_fold)
    assert np.any(receiver_low_fold)
    assert np.all(np.isnan(replacement.source_weathering_thickness_m[source_low_fold]))
    assert np.all(
        np.isnan(replacement.receiver_weathering_thickness_m[receiver_low_fold])
    )
    assert np.all(replacement.source_static_status[source_low_fold] == 'low_fold_v2_cell')
    assert np.all(
        replacement.receiver_static_status[receiver_low_fold] == 'low_fold_v2_cell'
    )


def test_two_layer_local_v2_trace_conversion_uses_endpoint_velocity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='line_2d_projected',
        v2_velocity_mode='solve_cell',
    )
    workflow = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=fixture.input_model,
        model=fixture.model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
    )
    v2_layer = _layer(workflow.solve_result, 'v2_t1')
    cell_velocity_m_s = np.asarray(
        [2100.0, 2300.0, 2500.0, 2700.0, 2900.0],
        dtype=np.float64,
    )
    row_midpoint_v2 = _cell_velocity_for_node_midpoints(
        fixture.input_model.source_node_id_sorted,
        fixture.input_model.receiver_node_id_sorted,
        cell_velocity_m_s=cell_velocity_m_s,
    )
    patched_v2_layer = replace(
        v2_layer,
        cell_velocity_m_s=cell_velocity_m_s,
        cell_slowness_s_per_m=1.0 / cell_velocity_m_s,
        row_midpoint_velocity_m_s=row_midpoint_v2,
    )
    patched_solve = replace(
        workflow.solve_result,
        layer_results=(patched_v2_layer, _layer(workflow.solve_result, 'v3_t2')),
    )
    expected_source_v2_sorted = _cell_velocity_for_node_ids(
        fixture.input_model.source_node_id_sorted,
        cell_velocity_m_s=cell_velocity_m_s,
    )
    expected_receiver_v2_sorted = _cell_velocity_for_node_ids(
        fixture.input_model.receiver_node_id_sorted,
        cell_velocity_m_s=cell_velocity_m_s,
    )
    assert np.any(expected_source_v2_sorted != row_midpoint_v2)
    assert np.any(expected_receiver_v2_sorted != row_midpoint_v2)

    original_conversion = multilayer_service.core_build_refraction_multilayer_conversion
    trace_v2_velocity: list[np.ndarray] = []

    def _capture_trace_v2_velocity(**kwargs):
        core_input = kwargs['input_model']
        core_solve = kwargs['solve_result']
        if int(core_input.n_traces) == int(fixture.input_model.n_traces):
            trace_v2_velocity.append(
                np.asarray(
                    core_solve.layer_result_by_kind['v2_t1'].velocity_m_s_sorted,
                    dtype=np.float64,
                ).copy()
            )
        return original_conversion(**kwargs)

    monkeypatch.setattr(
        multilayer_service,
        'core_build_refraction_multilayer_conversion',
        _capture_trace_v2_velocity,
    )

    replacement = build_refraction_multilayer_weathering_replacement_statics(
        input_model=fixture.input_model,
        model=fixture.model,
        solve_result=patched_solve,
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
    )

    assert len(trace_v2_velocity) == 2
    np.testing.assert_allclose(trace_v2_velocity[0], expected_source_v2_sorted)
    np.testing.assert_allclose(trace_v2_velocity[1], expected_receiver_v2_sorted)

    expected_source_v2 = _cell_velocity_for_node_ids(
        fixture.source_node_id,
        cell_velocity_m_s=cell_velocity_m_s,
    )
    expected_receiver_v2 = _cell_velocity_for_node_ids(
        fixture.receiver_node_id,
        cell_velocity_m_s=cell_velocity_m_s,
    )
    expected_source_sh1, expected_source_sh2 = compute_t1lsst_2layer_thicknesses(
        t1_s=fixture.source_t1_s,
        t2_s=fixture.source_t2_s,
        v1_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        v2_m_s=expected_source_v2,
        v3_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
    )
    expected_receiver_sh1, expected_receiver_sh2 = compute_t1lsst_2layer_thicknesses(
        t1_s=fixture.receiver_t1_s,
        t2_s=fixture.receiver_t2_s,
        v1_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        v2_m_s=expected_receiver_v2,
        v3_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
    )
    expected_source_wcor = compute_t1lsst_2layer_weathering_correction(
        sh1_m=expected_source_sh1,
        sh2_m=expected_source_sh2,
        v1_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        v2_m_s=expected_source_v2,
        v3_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
    )
    expected_receiver_wcor = compute_t1lsst_2layer_weathering_correction(
        sh1_m=expected_receiver_sh1,
        sh2_m=expected_receiver_sh2,
        v1_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        v2_m_s=expected_receiver_v2,
        v3_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
    )

    np.testing.assert_allclose(
        replacement.source_v2_m_s,
        expected_source_v2,
        rtol=1.0e-9,
    )
    np.testing.assert_allclose(
        replacement.receiver_v2_m_s,
        expected_receiver_v2,
        rtol=1.0e-9,
    )
    np.testing.assert_allclose(
        replacement.source_sh1_weathering_thickness_m,
        expected_source_sh1,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        replacement.source_sh2_weathering_thickness_m,
        expected_source_sh2,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        replacement.receiver_sh1_weathering_thickness_m,
        expected_receiver_sh1,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        replacement.receiver_sh2_weathering_thickness_m,
        expected_receiver_sh2,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        replacement.source_weathering_replacement_shift_s,
        expected_source_wcor,
        atol=_STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        replacement.receiver_weathering_replacement_shift_s,
        expected_receiver_wcor,
        atol=_STATIC_ATOL_S,
    )


def test_two_layer_local_v2_invalid_velocity_order_writes_nan_artifacts(
    tmp_path: Path,
) -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='line_2d_projected',
        v2_velocity_mode='solve_cell',
    )
    workflow = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=fixture.input_model,
        model=fixture.model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
    )
    v2_layer = _layer(workflow.solve_result, 'v2_t1')
    assert v2_layer.cell_velocity_m_s is not None
    assert v2_layer.cell_slowness_s_per_m is not None
    cell_velocity_m_s = np.asarray(v2_layer.cell_velocity_m_s, dtype=np.float64).copy()
    bad_cell_id = 1
    cell_velocity_m_s[bad_cell_id] = SYNTHETIC_MULTILAYER_V3_M_S + 200.0
    cell_slowness_s_per_m = 1.0 / cell_velocity_m_s
    patched_v2_layer = replace(
        v2_layer,
        cell_velocity_m_s=cell_velocity_m_s,
        cell_slowness_s_per_m=cell_slowness_s_per_m,
    )
    patched_solve = replace(
        workflow.solve_result,
        layer_results=(patched_v2_layer, _layer(workflow.solve_result, 'v3_t2')),
    )

    replacement = build_refraction_multilayer_weathering_replacement_statics(
        input_model=fixture.input_model,
        model=fixture.model,
        solve_result=patched_solve,
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
    )

    source_bad = replacement.source_v2_cell_id == bad_cell_id
    receiver_bad = replacement.receiver_v2_cell_id == bad_cell_id
    assert np.any(source_bad)
    assert np.any(receiver_bad)
    assert np.any(replacement.source_static_status == 'ok')
    assert np.all(
        replacement.source_static_status[source_bad] == 'invalid_velocity_order'
    )
    assert np.all(
        replacement.receiver_static_status[receiver_bad] == 'invalid_velocity_order'
    )
    assert np.all(np.isnan(replacement.source_sh1_weathering_thickness_m[source_bad]))
    assert np.all(np.isnan(replacement.source_sh2_weathering_thickness_m[source_bad]))
    assert np.all(np.isnan(replacement.source_weathering_replacement_shift_s[source_bad]))

    datum_result = build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
    )
    source_path = tmp_path / 'source_static_table.csv'
    table_path = tmp_path / 'source_receiver_static_table.npz'
    write_source_static_table_csv(result=datum_result, path=source_path)
    write_source_receiver_static_table_npz(result=datum_result, path=table_path)

    bad_source_index = int(np.flatnonzero(source_bad)[0])
    source_rows = _read_csv(source_path)
    assert source_rows[bad_source_index]['static_status'] == 'invalid_velocity_order'
    assert source_rows[bad_source_index]['sh1_weathering_thickness_m'] == ''
    assert source_rows[bad_source_index]['sh2_weathering_thickness_m'] == ''
    assert source_rows[bad_source_index]['weathering_correction_ms'] == ''
    assert source_rows[bad_source_index]['total_static_ms'] == ''
    with np.load(table_path, allow_pickle=False) as data:
        assert data['source_static_status'][bad_source_index] == 'invalid_velocity_order'
        assert np.isnan(data['source_sh1_m'][bad_source_index])
        assert np.isnan(data['source_sh2_m'][bad_source_index])
        assert np.isnan(data['source_weathering_correction_s'][bad_source_index])
        assert np.isnan(data['source_total_static_s'][bad_source_index])


def test_two_layer_invalid_sh2_preserves_explicit_sh1_in_artifacts(
    tmp_path: Path,
) -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='grid_3d',
        v2_velocity_mode='solve_global',
    )
    workflow = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=fixture.input_model,
        model=fixture.model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
    )
    v2_layer = _layer(workflow.solve_result, 'v2_t1')
    v3_layer = _layer(workflow.solve_result, 'v3_t2')
    assert v3_layer.node_time_term_s is not None

    bad_source_index = 0
    bad_node_index = 0
    source_t2 = np.array(v3_layer.source_time_term_s, dtype=np.float64, copy=True)
    node_t2 = np.array(v3_layer.node_time_term_s, dtype=np.float64, copy=True)
    source_t2[bad_source_index] = _invalid_negative_sh2_t2_s(
        float(fixture.source_sh1_m[bad_source_index])
    )
    node_t2[bad_node_index] = _invalid_negative_sh2_t2_s(
        float(_fixture_node_sh1_m(fixture)[bad_node_index])
    )
    patched_v3_layer = replace(
        v3_layer,
        source_time_term_s=source_t2,
        node_time_term_s=node_t2,
    )
    patched_solve = replace(
        workflow.solve_result,
        layer_results=(v2_layer, patched_v3_layer),
    )

    replacement = build_refraction_multilayer_weathering_replacement_statics(
        input_model=fixture.input_model,
        model=fixture.model,
        solve_result=patched_solve,
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
    )
    components = _components_from_replacement(replacement)

    assert replacement.source_sh1_weathering_thickness_m is not None
    assert replacement.source_sh2_weathering_thickness_m is not None
    assert replacement.receiver_sh1_weathering_thickness_m is not None
    assert replacement.receiver_sh2_weathering_thickness_m is not None
    np.testing.assert_allclose(
        replacement.source_sh1_weathering_thickness_m[bad_source_index],
        fixture.source_sh1_m[bad_source_index],
        atol=_THICKNESS_ATOL_M,
    )
    assert np.isnan(
        replacement.source_sh2_weathering_thickness_m[bad_source_index]
    )
    assert np.isnan(replacement.source_weathering_thickness_m[bad_source_index])
    np.testing.assert_allclose(
        components.source_sh1_m[bad_source_index],
        fixture.source_sh1_m[bad_source_index],
        atol=_THICKNESS_ATOL_M,
    )
    assert components.source_sh2_m is not None
    assert np.isnan(components.source_sh2_m[bad_source_index])
    assert components.receiver_sh2_m is not None

    datum_result = build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
    )
    near_surface_path = tmp_path / 'near_surface_model.csv'
    source_path = tmp_path / 'source_static_table.csv'
    receiver_path = tmp_path / 'receiver_static_table.csv'
    table_path = tmp_path / 'source_receiver_static_table.npz'
    write_near_surface_model_csv(result=datum_result, path=near_surface_path)
    write_source_static_table_csv(result=datum_result, path=source_path)
    write_receiver_static_table_csv(result=datum_result, path=receiver_path)
    write_source_receiver_static_table_npz(result=datum_result, path=table_path)

    near_surface_rows = _read_csv(near_surface_path)
    node_sh1_m = _fixture_node_sh1_m(fixture)[bad_node_index]
    assert float(
        near_surface_rows[bad_node_index]['sh1_weathering_thickness_m']
    ) == pytest.approx(node_sh1_m, abs=_THICKNESS_ATOL_M)
    assert near_surface_rows[bad_node_index]['sh2_weathering_thickness_m'] == ''
    assert near_surface_rows[bad_node_index]['weathering_thickness_m'] == ''
    assert float(
        near_surface_rows[bad_node_index]['layer1_base_elevation_m']
    ) == pytest.approx(
        datum_result.node_surface_elevation_m[bad_node_index] - node_sh1_m,
        abs=_THICKNESS_ATOL_M,
    )
    assert near_surface_rows[bad_node_index]['final_refractor_elevation_m'] == ''

    source_rows = _read_csv(source_path)
    assert float(
        source_rows[bad_source_index]['sh1_weathering_thickness_m']
    ) == pytest.approx(
        fixture.source_sh1_m[bad_source_index],
        abs=_THICKNESS_ATOL_M,
    )
    assert source_rows[bad_source_index]['sh2_weathering_thickness_m'] == ''
    assert float(
        source_rows[bad_source_index]['layer1_base_elevation_m']
    ) == pytest.approx(
        datum_result.source_surface_elevation_m[bad_source_index]
        - fixture.source_sh1_m[bad_source_index],
        abs=_THICKNESS_ATOL_M,
    )
    assert source_rows[bad_source_index]['final_refractor_elevation_m'] == ''
    assert source_rows[bad_source_index]['refractor_elevation_m'] == ''

    with np.load(table_path, allow_pickle=False) as data:
        np.testing.assert_allclose(
            data['source_sh1_m'][bad_source_index],
            fixture.source_sh1_m[bad_source_index],
            atol=_THICKNESS_ATOL_M,
        )
        assert np.isnan(data['source_sh2_m'][bad_source_index])
        np.testing.assert_allclose(
            data['source_layer1_base_elevation_m'][bad_source_index],
            datum_result.source_surface_elevation_m[bad_source_index]
            - fixture.source_sh1_m[bad_source_index],
            atol=_THICKNESS_ATOL_M,
        )
        assert np.isnan(
            data['source_final_refractor_elevation_m'][bad_source_index]
        )


def _make_two_layer_fixture(
    *,
    coordinate_mode: CoordinateMode,
    v2_velocity_mode: VelocityMode,
) -> _TwoLayerE2EFixture:
    inline_m = np.arange(10, dtype=np.float64) * 250.0
    x_m, y_m = _coordinates(
        inline_m=inline_m,
        coordinate_mode=coordinate_mode,
    )
    node_id = np.arange(inline_m.shape[0], dtype=np.int64)
    station_index = np.arange(inline_m.shape[0], dtype=np.float64)
    sh1_m = 8.0 + 0.4 * (station_index % 5.0) + 0.02 * station_index
    sh2_m = 14.0 + 0.5 * (station_index % 4.0) + 0.03 * station_index
    t1_s, t2_s = _forward_t1_t2_s(sh1_m=sh1_m, sh2_m=sh2_m)
    wcor_s = compute_t1lsst_2layer_weathering_correction(
        sh1_m=sh1_m,
        sh2_m=sh2_m,
        v1_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        v2_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
        v3_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
    )

    rows = _observation_rows(
        inline_m=inline_m,
        t1_s=t1_s,
        t2_s=t2_s,
    )
    source_index = np.asarray([row[0] for row in rows], dtype=np.int64)
    receiver_index = np.asarray([row[1] for row in rows], dtype=np.int64)
    offset_m = np.asarray([row[2] for row in rows], dtype=np.float64)
    pick_time_s = np.asarray([row[3] for row in rows], dtype=np.float64)
    layer_kind = np.asarray([row[4] for row in rows], dtype='<U16')
    n_traces = int(pick_time_s.shape[0])
    source_key_sorted = np.asarray(
        [f's:{index}' for index in source_index.tolist()],
        dtype=object,
    )
    receiver_key_sorted = np.asarray(
        [f'r:{index}' for index in receiver_index.tolist()],
        dtype=object,
    )
    source_endpoint_positions = _first_occurrence_positions(source_key_sorted)
    receiver_endpoint_positions = _first_occurrence_positions(receiver_key_sorted)
    endpoint_table = _endpoint_table(
        node_id=node_id,
        x_m=x_m,
        y_m=y_m,
        source_index=source_index,
        receiver_index=receiver_index,
    )
    input_model = RefractionStaticInputModel(
        file_id=f'two-layer-{coordinate_mode}-{v2_velocity_mode}',
        n_traces=n_traces,
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        pick_time_s_sorted=np.ascontiguousarray(pick_time_s, dtype=np.float64),
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=np.ones(n_traces, dtype=bool),
        source_id_sorted=1000 + source_index,
        receiver_id_sorted=2000 + receiver_index,
        source_x_m_sorted=x_m[source_index],
        source_y_m_sorted=y_m[source_index],
        receiver_x_m_sorted=x_m[receiver_index],
        receiver_y_m_sorted=y_m[receiver_index],
        source_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=np.ascontiguousarray(offset_m, dtype=np.float64),
        offset_m_sorted=np.ascontiguousarray(offset_m, dtype=np.float64),
        distance_m_sorted=np.ascontiguousarray(offset_m, dtype=np.float64),
        source_endpoint_key_sorted=source_key_sorted,
        receiver_endpoint_key_sorted=receiver_key_sorted,
        source_node_id_sorted=source_index,
        receiver_node_id_sorted=receiver_index,
        node_x_m=x_m,
        node_y_m=y_m,
        node_elevation_m=np.zeros(node_id.shape, dtype=np.float64),
        node_kind=np.full(node_id.shape, 'both', dtype='<U8'),
        rejection_reason_sorted=np.full(n_traces, 'ok', dtype='<U32'),
        qc={'fixture': 'two_layer_e2e'},
        endpoint_table=endpoint_table,
        metadata={'coordinate_mode': coordinate_mode},
    )

    source_node = source_index[source_endpoint_positions]
    receiver_node = receiver_index[receiver_endpoint_positions]
    return _TwoLayerE2EFixture(
        input_model=input_model,
        model=_model(
            coordinate_mode=coordinate_mode,
            v2_velocity_mode=v2_velocity_mode,
        ),
        expected_layer_kind_sorted=layer_kind,
        source_endpoint_key=source_key_sorted[source_endpoint_positions],
        receiver_endpoint_key=receiver_key_sorted[receiver_endpoint_positions],
        source_node_id=np.ascontiguousarray(source_node, dtype=np.int64),
        receiver_node_id=np.ascontiguousarray(receiver_node, dtype=np.int64),
        source_t1_s=np.ascontiguousarray(t1_s[source_node], dtype=np.float64),
        source_t2_s=np.ascontiguousarray(t2_s[source_node], dtype=np.float64),
        receiver_t1_s=np.ascontiguousarray(t1_s[receiver_node], dtype=np.float64),
        receiver_t2_s=np.ascontiguousarray(t2_s[receiver_node], dtype=np.float64),
        source_sh1_m=np.ascontiguousarray(sh1_m[source_node], dtype=np.float64),
        source_sh2_m=np.ascontiguousarray(sh2_m[source_node], dtype=np.float64),
        receiver_sh1_m=np.ascontiguousarray(sh1_m[receiver_node], dtype=np.float64),
        receiver_sh2_m=np.ascontiguousarray(sh2_m[receiver_node], dtype=np.float64),
        source_weathering_correction_s=np.ascontiguousarray(
            wcor_s[source_node],
            dtype=np.float64,
        ),
        receiver_weathering_correction_s=np.ascontiguousarray(
            wcor_s[receiver_node],
            dtype=np.float64,
        ),
        trace_shift_s_sorted=np.ascontiguousarray(
            wcor_s[source_index] + wcor_s[receiver_index],
            dtype=np.float64,
        ),
    )


def _coordinates(
    *,
    inline_m: np.ndarray,
    coordinate_mode: CoordinateMode,
) -> tuple[np.ndarray, np.ndarray]:
    if coordinate_mode == 'grid_3d':
        return (
            np.ascontiguousarray(inline_m, dtype=np.float64),
            np.zeros(inline_m.shape, dtype=np.float64),
        )
    line_origin_x_m = 1000.0
    line_origin_y_m = 2000.0
    line_azimuth_deg = 37.0
    azimuth_rad = np.deg2rad(line_azimuth_deg)
    return (
        np.ascontiguousarray(
            line_origin_x_m + inline_m * np.sin(azimuth_rad),
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            line_origin_y_m + inline_m * np.cos(azimuth_rad),
            dtype=np.float64,
        ),
    )


def _observation_rows(
    *,
    inline_m: np.ndarray,
    t1_s: np.ndarray,
    t2_s: np.ndarray,
) -> list[tuple[int, int, float, float, str]]:
    rows: list[tuple[int, int, float, float, str]] = []
    for source_index, source_inline in enumerate(inline_m.tolist()):
        for receiver_index, receiver_inline in enumerate(inline_m.tolist()):
            if source_index == receiver_index:
                continue
            offset_m = abs(float(receiver_inline) - float(source_inline))
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
    return rows


def _model(
    *,
    coordinate_mode: CoordinateMode,
    v2_velocity_mode: VelocityMode,
) -> RefractionStaticModelRequest:
    payload: dict[str, object] = {
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
                'velocity_mode': v2_velocity_mode,
                'initial_velocity_m_s': SYNTHETIC_MULTILAYER_V2_M_S,
                'min_velocity_m_s': 1600.0,
                'max_velocity_m_s': 3200.0,
                'min_observations_per_cell': 1,
                'smoothing_weight': 0.0,
            },
            {
                'kind': 'v3_t2',
                'enabled': True,
                'min_offset_m': _V3_MIN_OFFSET_M,
                'max_offset_m': _V3_MAX_OFFSET_M,
                'velocity_mode': 'solve_global',
                'initial_velocity_m_s': SYNTHETIC_MULTILAYER_V3_M_S,
                'min_velocity_m_s': 2600.0,
                'max_velocity_m_s': 4800.0,
            },
        ],
    }
    if v2_velocity_mode == 'solve_cell':
        payload['refractor_cell'] = {
            'number_of_cell_x': 5,
            'size_of_cell_x_m': 500.0,
            'x_coordinate_origin_m': 0.0,
            'number_of_cell_y': 1,
            'size_of_cell_y_m': None,
            'y_coordinate_origin_m': 0.0,
            'assignment_mode': 'midpoint',
            'outside_grid_policy': 'reject',
            'coordinate_mode': coordinate_mode,
            'line_origin_x_m': 1000.0 if coordinate_mode == 'line_2d_projected' else None,
            'line_origin_y_m': 2000.0 if coordinate_mode == 'line_2d_projected' else None,
            'line_azimuth_deg': 37.0 if coordinate_mode == 'line_2d_projected' else None,
            'min_observations_per_cell': 1,
            'velocity_smoothing_weight': 0.0,
            'smoothing_reference_distance_m': None,
        }
    return RefractionStaticModelRequest.model_validate(payload)


def _endpoint_table(
    *,
    node_id: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
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
        x_m=np.ascontiguousarray(x_m, dtype=np.float64),
        y_m=np.ascontiguousarray(y_m, dtype=np.float64),
        elevation_m=np.zeros(node_id.shape, dtype=np.float64),
        kind=np.full(node_id.shape, 'both', dtype='<U8'),
        pick_count=pick_count,
    )


def _assert_layer_masks_select_expected_branches(
    fixture: _TwoLayerE2EFixture,
) -> None:
    masks = build_refraction_layer_observation_masks(
        input_model=fixture.input_model,
        model=fixture.model,
    )

    np.testing.assert_array_equal(
        masks.layer_used_mask_sorted['v2_t1'],
        fixture.expected_layer_kind_sorted == 'v2_t1',
    )
    np.testing.assert_array_equal(
        masks.layer_used_mask_sorted['v3_t2'],
        fixture.expected_layer_kind_sorted == 'v3_t2',
    )
    assert masks.layer_observation_count['v2_t1'] == 48
    assert masks.layer_observation_count['v3_t2'] == 36


def _assert_global_layer_result(
    layer: RefractionLayerSolveResult,
    *,
    expected_velocity_m_s: float,
) -> None:
    assert layer.velocity_mode == 'solve_global'
    assert layer.global_velocity_m_s == pytest.approx(expected_velocity_m_s, rel=1.0e-9)
    assert layer.global_slowness_s_per_m == pytest.approx(
        1.0 / expected_velocity_m_s,
        rel=1.0e-9,
    )
    np.testing.assert_allclose(
        layer.trace_residual_s_sorted[layer.used_observation_mask_sorted],
        0.0,
        atol=1.0e-8,
    )


def _assert_two_layer_outputs_match_truth(
    fixture: _TwoLayerE2EFixture,
    outputs: _StaticOutputs,
) -> None:
    result = outputs.solve_result
    v2_layer = _layer(result, 'v2_t1')
    v3_layer = _layer(result, 'v3_t2')

    np.testing.assert_allclose(v2_layer.source_time_term_s, fixture.source_t1_s)
    np.testing.assert_allclose(v2_layer.receiver_time_term_s, fixture.receiver_t1_s)
    np.testing.assert_allclose(v3_layer.source_time_term_s, fixture.source_t2_s)
    np.testing.assert_allclose(v3_layer.receiver_time_term_s, fixture.receiver_t2_s)

    np.testing.assert_allclose(
        outputs.components.source_sh1_m,
        fixture.source_sh1_m,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        outputs.components.source_sh2_m,
        fixture.source_sh2_m,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        outputs.components.receiver_sh1_m,
        fixture.receiver_sh1_m,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        outputs.components.receiver_sh2_m,
        fixture.receiver_sh2_m,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        outputs.components.source_weathering_correction_s,
        fixture.source_weathering_correction_s,
        atol=_STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        outputs.components.receiver_weathering_correction_s,
        fixture.receiver_weathering_correction_s,
        atol=_STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        outputs.result.source_weathering_thickness_m,
        fixture.source_sh1_m + fixture.source_sh2_m,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        outputs.result.source_refractor_elevation_m,
        -(fixture.source_sh1_m + fixture.source_sh2_m),
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        outputs.result.receiver_weathering_thickness_m,
        fixture.receiver_sh1_m + fixture.receiver_sh2_m,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        outputs.result.receiver_refractor_elevation_m,
        -(fixture.receiver_sh1_m + fixture.receiver_sh2_m),
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        outputs.trace_shift_s_sorted,
        fixture.trace_shift_s_sorted,
        atol=_STATIC_ATOL_S,
    )

    _assert_static_tables_match_truth(fixture, outputs)
    _assert_trace_shift_sign_convention(outputs.trace_shift_s_sorted)


def _compute_static_outputs(
    *,
    fixture: _TwoLayerE2EFixture,
    tmp_path: Path,
) -> _StaticOutputs:
    workflow = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=fixture.input_model,
        model=fixture.model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
    )
    return _write_static_outputs(
        tmp_path=tmp_path,
        solve_result=workflow.solve_result,
        components=workflow.components,
        result=workflow.datum_result,
    )


def _resolved_first_layer() -> ResolvedRefractionFirstLayer:
    return ResolvedRefractionFirstLayer(
        mode='constant',
        weathering_velocity_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        status='constant',
        qc={'weathering_velocity_m_s': SYNTHETIC_MULTILAYER_V1_M_S},
    )


def _write_static_outputs(
    *,
    tmp_path: Path,
    solve_result: RefractionMultiLayerSolveResult,
    components: RefractionMultiLayerStaticComponents,
    result: RefractionDatumStaticsResult,
) -> _StaticOutputs:
    source_path = tmp_path / 'source_static_table.csv'
    receiver_path = tmp_path / 'receiver_static_table.csv'
    table_path = tmp_path / 'source_receiver_static_table.npz'
    trace_path = tmp_path / 'refraction_statics.csv'
    write_source_static_table_csv(result=result, path=source_path)
    write_receiver_static_table_csv(result=result, path=receiver_path)
    write_source_receiver_static_table_npz(result=result, path=table_path)
    write_refraction_statics_csv(result=result, path=trace_path)
    with np.load(table_path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    return _StaticOutputs(
        solve_result=solve_result,
        components=components,
        result=result,
        source_rows=_read_csv(source_path),
        receiver_rows=_read_csv(receiver_path),
        source_receiver_arrays=arrays,
        trace_rows=_read_csv(trace_path),
        trace_shift_s_sorted=np.ascontiguousarray(
            result.refraction_trace_shift_s_sorted,
            dtype=np.float64,
        ),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _values_by_sorted_key(
    sorted_key: np.ndarray,
    endpoint_key: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    lookup = {
        str(key): float(value)
        for key, value in zip(endpoint_key.tolist(), values.tolist(), strict=True)
    }
    return np.ascontiguousarray(
        [lookup[str(key)] for key in sorted_key.tolist()],
        dtype=np.float64,
    )


def _cell_velocity_for_node_ids(
    node_id: np.ndarray,
    *,
    cell_velocity_m_s: np.ndarray,
) -> np.ndarray:
    cell_id = np.asarray(node_id, dtype=np.int64) // 2
    return np.ascontiguousarray(cell_velocity_m_s[cell_id], dtype=np.float64)


def _cell_velocity_for_node_midpoints(
    source_node_id: np.ndarray,
    receiver_node_id: np.ndarray,
    *,
    cell_velocity_m_s: np.ndarray,
) -> np.ndarray:
    midpoint_inline_m = (
        np.asarray(source_node_id, dtype=np.float64)
        + np.asarray(receiver_node_id, dtype=np.float64)
    ) * 125.0
    cell_id = np.floor(midpoint_inline_m / 500.0).astype(np.int64)
    return np.ascontiguousarray(cell_velocity_m_s[cell_id], dtype=np.float64)


def _invalid_negative_sh2_t2_s(sh1_m: float) -> float:
    _, t2_s = _forward_t1_t2_s(
        sh1_m=np.asarray([sh1_m], dtype=np.float64),
        sh2_m=np.asarray([0.0], dtype=np.float64),
    )
    return float(t2_s[0] - 1.0e-3)


def _fixture_node_sh1_m(fixture: _TwoLayerE2EFixture) -> np.ndarray:
    station_index = np.arange(
        fixture.input_model.node_elevation_m.shape[0],
        dtype=np.float64,
    )
    return np.ascontiguousarray(
        8.0 + 0.4 * (station_index % 5.0) + 0.02 * station_index,
        dtype=np.float64,
    )


def _fixture_node_sh2_m(fixture: _TwoLayerE2EFixture) -> np.ndarray:
    station_index = np.arange(
        fixture.input_model.node_elevation_m.shape[0],
        dtype=np.float64,
    )
    return np.ascontiguousarray(
        14.0 + 0.5 * (station_index % 4.0) + 0.03 * station_index,
        dtype=np.float64,
    )


def _assert_static_tables_match_truth(
    fixture: _TwoLayerE2EFixture,
    outputs: _StaticOutputs,
) -> None:
    required_columns = {
        't1_ms',
        't2_ms',
        'v2_m_s',
        'v3_m_s',
        'sh1_weathering_thickness_m',
        'sh2_weathering_thickness_m',
        'layer1_base_elevation_m',
        'final_refractor_elevation_m',
        'refractor_elevation_m',
        'weathering_correction_ms',
        'total_static_ms',
        'total_applied_shift_ms',
    }
    assert required_columns <= set(outputs.source_rows[0])
    assert required_columns <= set(outputs.receiver_rows[0])
    _assert_rows_match_endpoint_truth(
        outputs.source_rows,
        expected_t1_s=fixture.source_t1_s,
        expected_t2_s=fixture.source_t2_s,
        expected_sh1_m=fixture.source_sh1_m,
        expected_sh2_m=fixture.source_sh2_m,
        expected_wcor_s=fixture.source_weathering_correction_s,
    )
    _assert_rows_match_endpoint_truth(
        outputs.receiver_rows,
        expected_t1_s=fixture.receiver_t1_s,
        expected_t2_s=fixture.receiver_t2_s,
        expected_sh1_m=fixture.receiver_sh1_m,
        expected_sh2_m=fixture.receiver_sh2_m,
        expected_wcor_s=fixture.receiver_weathering_correction_s,
    )
    _assert_npz_tables_match_truth(fixture, outputs.source_receiver_arrays)
    for row, expected_shift_s in zip(
        outputs.trace_rows,
        fixture.trace_shift_s_sorted.tolist(),
        strict=True,
    ):
        trace_shift_ms = float(row['refraction_trace_shift_ms'])
        assert trace_shift_ms / 1000.0 == pytest.approx(
            expected_shift_s,
            abs=_STATIC_ATOL_S,
        )
        source_shift_ms = float(row['source_refraction_shift_ms'])
        receiver_shift_ms = float(row['receiver_refraction_shift_ms'])
        assert trace_shift_ms == pytest.approx(source_shift_ms + receiver_shift_ms)


def _assert_rows_match_endpoint_truth(
    rows: list[dict[str, str]],
    *,
    expected_t1_s: np.ndarray,
    expected_t2_s: np.ndarray,
    expected_sh1_m: np.ndarray,
    expected_sh2_m: np.ndarray,
    expected_wcor_s: np.ndarray,
) -> None:
    for index, row in enumerate(rows):
        assert row['static_status'] == 'ok'
        assert float(row['t1_ms']) / 1000.0 == pytest.approx(
            expected_t1_s[index],
            abs=_TIME_ATOL_S,
        )
        assert float(row['t2_ms']) / 1000.0 == pytest.approx(
            expected_t2_s[index],
            abs=_TIME_ATOL_S,
        )
        assert float(row['v2_m_s']) == pytest.approx(
            SYNTHETIC_MULTILAYER_V2_M_S,
            rel=1.0e-8,
        )
        assert float(row['v3_m_s']) == pytest.approx(
            SYNTHETIC_MULTILAYER_V3_M_S,
            rel=1.0e-9,
        )
        assert float(row['sh1_weathering_thickness_m']) == pytest.approx(
            expected_sh1_m[index],
            abs=_THICKNESS_ATOL_M,
        )
        assert float(row['sh2_weathering_thickness_m']) == pytest.approx(
            expected_sh2_m[index],
            abs=_THICKNESS_ATOL_M,
        )
        layer1_base = -expected_sh1_m[index]
        final_refractor = -(expected_sh1_m[index] + expected_sh2_m[index])
        assert float(row['layer1_base_elevation_m']) == pytest.approx(
            layer1_base,
            abs=_THICKNESS_ATOL_M,
        )
        assert float(row['final_refractor_elevation_m']) == pytest.approx(
            final_refractor,
            abs=_THICKNESS_ATOL_M,
        )
        assert float(row['refractor_elevation_m']) == pytest.approx(
            final_refractor,
            abs=_THICKNESS_ATOL_M,
        )
        assert float(row['weathering_correction_ms']) / 1000.0 == pytest.approx(
            expected_wcor_s[index],
            abs=_STATIC_ATOL_S,
        )
        assert float(row['total_static_ms']) / 1000.0 == pytest.approx(
            expected_wcor_s[index],
            abs=_STATIC_ATOL_S,
        )
        assert float(row['total_applied_shift_ms']) / 1000.0 == pytest.approx(
            expected_wcor_s[index],
            abs=_STATIC_ATOL_S,
        )


def _assert_npz_tables_match_truth(
    fixture: _TwoLayerE2EFixture,
    arrays: dict[str, np.ndarray],
) -> None:
    expected_keys = {
        'source_t1_s',
        'source_t2_s',
        'source_v2_m_s',
        'source_v3_m_s',
        'source_sh1_m',
        'source_sh2_m',
        'source_layer1_base_elevation_m',
        'source_final_refractor_elevation_m',
        'source_weathering_correction_s',
        'source_total_static_s',
        'receiver_t1_s',
        'receiver_t2_s',
        'receiver_v2_m_s',
        'receiver_v3_m_s',
        'receiver_sh1_m',
        'receiver_sh2_m',
        'receiver_layer1_base_elevation_m',
        'receiver_final_refractor_elevation_m',
        'receiver_weathering_correction_s',
        'receiver_total_static_s',
    }
    assert expected_keys <= set(arrays)
    np.testing.assert_allclose(arrays['source_t1_s'], fixture.source_t1_s)
    np.testing.assert_allclose(arrays['source_t2_s'], fixture.source_t2_s)
    np.testing.assert_allclose(
        arrays['source_v2_m_s'],
        SYNTHETIC_MULTILAYER_V2_M_S,
        rtol=1.0e-8,
    )
    np.testing.assert_allclose(
        arrays['source_v3_m_s'],
        SYNTHETIC_MULTILAYER_V3_M_S,
        rtol=1.0e-9,
    )
    np.testing.assert_allclose(
        arrays['source_sh1_m'],
        fixture.source_sh1_m,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        arrays['source_sh2_m'],
        fixture.source_sh2_m,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        arrays['source_layer1_base_elevation_m'],
        -fixture.source_sh1_m,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        arrays['source_final_refractor_elevation_m'],
        -(fixture.source_sh1_m + fixture.source_sh2_m),
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        arrays['source_weathering_correction_s'],
        fixture.source_weathering_correction_s,
        atol=_STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        arrays['source_total_static_s'],
        fixture.source_weathering_correction_s,
        atol=_STATIC_ATOL_S,
    )
    np.testing.assert_allclose(arrays['receiver_t1_s'], fixture.receiver_t1_s)
    np.testing.assert_allclose(arrays['receiver_t2_s'], fixture.receiver_t2_s)
    np.testing.assert_allclose(
        arrays['receiver_v2_m_s'],
        SYNTHETIC_MULTILAYER_V2_M_S,
        rtol=1.0e-8,
    )
    np.testing.assert_allclose(
        arrays['receiver_v3_m_s'],
        SYNTHETIC_MULTILAYER_V3_M_S,
        rtol=1.0e-9,
    )
    np.testing.assert_allclose(
        arrays['receiver_sh1_m'],
        fixture.receiver_sh1_m,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        arrays['receiver_sh2_m'],
        fixture.receiver_sh2_m,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        arrays['receiver_layer1_base_elevation_m'],
        -fixture.receiver_sh1_m,
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        arrays['receiver_final_refractor_elevation_m'],
        -(fixture.receiver_sh1_m + fixture.receiver_sh2_m),
        atol=_THICKNESS_ATOL_M,
    )
    np.testing.assert_allclose(
        arrays['receiver_weathering_correction_s'],
        fixture.receiver_weathering_correction_s,
        atol=_STATIC_ATOL_S,
    )
    np.testing.assert_allclose(
        arrays['receiver_total_static_s'],
        fixture.receiver_weathering_correction_s,
        atol=_STATIC_ATOL_S,
    )


def _assert_trace_shift_sign_convention(trace_shift_s_sorted: np.ndarray) -> None:
    assert SIGN_CONVENTION == 'corrected(t) = raw(t - shift_s)'
    assert np.all(trace_shift_s_sorted < 0.0)
    raw_event_time_s = 1.0
    corrected_event_time_s = raw_event_time_s + float(trace_shift_s_sorted[0])
    assert corrected_event_time_s < raw_event_time_s


def _layer(
    result: RefractionMultiLayerSolveResult,
    layer_kind: str,
) -> RefractionLayerSolveResult:
    for layer in result.layer_results:
        if layer.layer_kind == layer_kind:
            return layer
    raise AssertionError(f'{layer_kind} layer result was not returned')


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


def _forward_t1_t2_s(
    *,
    sh1_m: np.ndarray,
    sh2_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    v1 = SYNTHETIC_MULTILAYER_V1_M_S
    v2 = SYNTHETIC_MULTILAYER_V2_M_S
    v3 = SYNTHETIC_MULTILAYER_V3_M_S
    t1_s = sh1_m * np.sqrt(v2 * v2 - v1 * v1) / (v1 * v2)
    t2_s = (
        sh1_m * np.sqrt(v3 * v3 - v1 * v1) / (v1 * v3)
        + sh2_m * np.sqrt(v3 * v3 - v2 * v2) / (v2 * v3)
    )
    return (
        np.ascontiguousarray(t1_s, dtype=np.float64),
        np.ascontiguousarray(t2_s, dtype=np.float64),
    )
