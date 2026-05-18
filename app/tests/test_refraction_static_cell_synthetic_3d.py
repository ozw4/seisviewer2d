from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_artifacts import write_refraction_static_artifacts
from app.services.refraction_static_datum import build_refraction_datum_statics
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
    RefractionEndpointTable,
    RefractionStaticInputModel,
)
from app.services.refraction_static_weathering_replacement import (
    compute_weathering_replacement_statics_from_first_breaks,
)
from app.tests.fixtures.refraction_synthetic import (
    SyntheticRefractionCellDataset,
    make_clean_3d_cell_refraction_dataset,
)

TRUE_CELL_V2_M_S = np.asarray(
    [[2200.0, 2400.0], [2600.0, 2800.0]],
    dtype=np.float64,
)
MIN_OBSERVATIONS_PER_CELL = 5
SLOWNESS_ATOL_S_PER_M = 1.0e-12
VELOCITY_ATOL_M_S = 1.0e-5
TIME_ATOL_S = 1.0e-8


def test_cell_v2_t1_clean_3d_recovers_known_velocity_grid() -> None:
    dataset, result, _ = _solve_clean_3d_cell_problem()

    expected_cell_id = np.arange(TRUE_CELL_V2_M_S.size, dtype=np.int64)
    np.testing.assert_array_equal(result.active_cell_id, expected_cell_id)
    np.testing.assert_array_equal(result.inactive_cell_id, [])
    np.testing.assert_array_equal(result.cell_velocity_status, ['solved'] * 4)
    assert np.all(dataset.cell_observation_count >= MIN_OBSERVATIONS_PER_CELL)

    np.testing.assert_allclose(
        result.cell_bedrock_slowness_s_per_m,
        1.0 / TRUE_CELL_V2_M_S.ravel(),
        atol=SLOWNESS_ATOL_S_PER_M,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.cell_bedrock_velocity_m_s,
        TRUE_CELL_V2_M_S.ravel(),
        atol=VELOCITY_ATOL_M_S,
        rtol=0.0,
    )
    assert result.bedrock_velocity_mode == 'solve_cell'
    assert result.qc['bedrock_velocity_mode'] == 'solve_cell'


def test_cell_v2_t1_clean_3d_midpoint_assignment_uses_x_and_y() -> None:
    dataset, result, _ = _solve_clean_3d_cell_problem()

    row_index = result.row_trace_index_sorted
    row_cell_id = _required_array(result.row_midpoint_cell_id)
    row_ix = row_cell_id % TRUE_CELL_V2_M_S.shape[1]
    row_iy = row_cell_id // TRUE_CELL_V2_M_S.shape[1]

    np.testing.assert_array_equal(row_cell_id, dataset.true_cell_id_for_pick[row_index])
    np.testing.assert_array_equal(row_ix, dataset.true_cell_ix_for_pick[row_index])
    np.testing.assert_array_equal(row_iy, dataset.true_cell_iy_for_pick[row_index])
    assert set(row_ix.tolist()) == {0, 1}
    assert set(row_iy.tolist()) == {0, 1}


def test_cell_v2_t1_clean_3d_predicts_picks_with_near_zero_residual() -> None:
    _, result, _ = _solve_clean_3d_cell_problem()

    np.testing.assert_allclose(
        result.modeled_pick_time_s,
        result.observed_pick_time_s,
        atol=TIME_ATOL_S,
        rtol=0.0,
    )
    assert _rms(result.residual_time_s) <= TIME_ATOL_S
    np.testing.assert_allclose(result.residual_time_s, 0.0, atol=TIME_ATOL_S)


def test_cell_v2_t1_clean_3d_velocity_artifact_matches_solution(
    tmp_path: Path,
) -> None:
    dataset, result, req = _solve_clean_3d_cell_problem()

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )
    assert paths.refraction_refractor_velocity_cells_csv is not None
    assert paths.refraction_refractor_velocity_grid_npz is not None
    assert paths.refraction_refractor_velocity_qc_json is not None

    rows = _read_csv(paths.refraction_refractor_velocity_cells_csv)
    assert len(rows) == TRUE_CELL_V2_M_S.size
    for row in rows:
        cell_id = int(row['cell_id'])
        ix = cell_id % TRUE_CELL_V2_M_S.shape[1]
        iy = cell_id // TRUE_CELL_V2_M_S.shape[1]
        true_v2 = float(TRUE_CELL_V2_M_S[iy, ix])

        assert int(row['ix']) == ix
        assert int(row['iy']) == iy
        assert row['active'] == 'true'
        assert row['velocity_status'] == 'solved'
        assert int(row['n_observations']) == int(
            dataset.cell_observation_count[cell_id]
        )
        assert int(row['n_used_observations']) == int(
            dataset.cell_observation_count[cell_id]
        )
        assert int(row['n_rejected_observations']) == 0
        assert float(row['v2_m_s']) == pytest.approx(
            true_v2,
            abs=VELOCITY_ATOL_M_S,
        )
        assert float(row['slowness_s_per_m']) == pytest.approx(
            1.0 / true_v2,
            abs=SLOWNESS_ATOL_S_PER_M,
        )

    with np.load(paths.refraction_refractor_velocity_grid_npz, allow_pickle=False) as grid:
        np.testing.assert_array_equal(grid['cell_id'], [0, 1, 2, 3])
        np.testing.assert_array_equal(grid['ix'], [0, 1, 0, 1])
        np.testing.assert_array_equal(grid['iy'], [0, 0, 1, 1])
        np.testing.assert_array_equal(grid['active_cell_mask'], [True] * 4)
        np.testing.assert_array_equal(grid['n_observations_per_cell'], dataset.cell_observation_count)
        np.testing.assert_allclose(
            grid['v2_m_s'],
            TRUE_CELL_V2_M_S.ravel(),
            atol=VELOCITY_ATOL_M_S,
            rtol=0.0,
        )
        np.testing.assert_array_equal(grid['velocity_status'], ['solved'] * 4)

    cell_qc = json.loads(
        paths.refraction_refractor_velocity_qc_json.read_text(encoding='utf-8')
    )
    assert cell_qc['coordinate_mode'] == 'grid_3d'
    assert cell_qc['number_of_cell_x'] == 2
    assert cell_qc['number_of_cell_y'] == 2
    assert cell_qc['n_active_cells'] == 4
    assert cell_qc['n_inactive_cells'] == 0
    assert cell_qc['n_low_fold_cells'] == 0


def _solve_clean_3d_cell_problem() -> tuple[
    SyntheticRefractionCellDataset,
    RefractionDatumStaticsResult,
    RefractionStaticApplyRequest,
]:
    dataset = make_clean_3d_cell_refraction_dataset(
        seed=433,
        cell_v2_m_s=TRUE_CELL_V2_M_S,
        n_sources=6,
        n_receivers=6,
        cell_size_x_m=500.0,
        cell_size_y_m=500.0,
        noise_std_s=0.0,
        outlier_fraction=0.0,
    )
    req = _cell_apply_request(dataset)
    input_model = _input_model_from_dataset(dataset)
    replacement = compute_weathering_replacement_statics_from_first_breaks(
        req=req,
        state=None,
        input_model=input_model,
    )
    result = build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=req.datum,
        apply_options=req.apply,
        state=None,
        file_id=req.file_id,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
    )
    return dataset, result, req


def _cell_apply_request(
    dataset: SyntheticRefractionCellDataset,
) -> RefractionStaticApplyRequest:
    return RefractionStaticApplyRequest.model_validate(
        {
            'file_id': 'clean-3d-cell-synthetic',
            'key1_byte': 189,
            'key2_byte': 193,
            'pick_source': {
                'kind': 'batch_predicted_npz',
                'job_id': 'clean-3d-cell-synthetic-first-breaks',
                'artifact_name': 'predicted_picks_time_s.npz',
            },
            'linkage': {'mode': 'none'},
            'model': {
                'method': 'gli_variable_thickness',
                'weathering_velocity_m_s': dataset.true_v1_m_s,
                'bedrock_velocity_mode': 'solve_cell',
                'bedrock_velocity_m_s': None,
                'initial_bedrock_velocity_m_s': 2500.0,
                'min_bedrock_velocity_m_s': 1200.0,
                'max_bedrock_velocity_m_s': 6000.0,
                'max_weathering_thickness_m': None,
                'refractor_cell': {
                    'number_of_cell_x': int(dataset.true_cell_v2_m_s.shape[1]),
                    'size_of_cell_x_m': dataset.cell_size_x_m,
                    'x_coordinate_origin_m': dataset.x_coordinate_origin_m,
                    'number_of_cell_y': int(dataset.true_cell_v2_m_s.shape[0]),
                    'size_of_cell_y_m': dataset.cell_size_y_m,
                    'y_coordinate_origin_m': dataset.y_coordinate_origin_m,
                    'assignment_mode': 'midpoint',
                    'outside_grid_policy': 'reject',
                    'coordinate_mode': 'grid_3d',
                    'min_observations_per_cell': MIN_OBSERVATIONS_PER_CELL,
                    'velocity_smoothing_weight': 0.0,
                    'smoothing_reference_distance_m': None,
                },
            },
            'moveout': {
                'model': 'head_wave_linear_offset',
                'distance_source': 'geometry',
                'offset_byte': None,
                'min_offset_m': None,
                'max_offset_m': None,
                'allow_missing_offset': False,
                'max_geometry_offset_mismatch_m': None,
            },
            'solver': {
                'damping': 0.0,
                'min_picks_per_node': 1,
                'max_abs_half_intercept_time_ms': 500.0,
                'robust': {
                    'enabled': False,
                    'method': 'mad',
                    'threshold': 3.5,
                    'max_iterations': 5,
                    'min_used_fraction': 0.5,
                    'min_used_observations': 1,
                },
            },
            'datum': {'mode': 'none'},
            'conversion': {'mode': 't1lsst_1layer'},
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


def _input_model_from_dataset(
    dataset: SyntheticRefractionCellDataset,
) -> RefractionStaticInputModel:
    n_traces = int(dataset.pick_time_s.shape[0])
    endpoint_table = _endpoint_table_from_dataset(dataset)
    node_x_m = np.concatenate(
        (dataset.source_endpoint_x_m, dataset.receiver_endpoint_x_m)
    )
    node_y_m = np.concatenate(
        (dataset.source_endpoint_y_m, dataset.receiver_endpoint_y_m)
    )
    node_elevation_m = np.zeros(endpoint_table.node_id.shape, dtype=np.float64)
    node_kind = np.concatenate(
        (
            np.full(dataset.source_endpoint_id.shape, 'source', dtype='<U16'),
            np.full(dataset.receiver_endpoint_id.shape, 'receiver', dtype='<U16'),
        )
    )
    return RefractionStaticInputModel(
        file_id='clean-3d-cell-synthetic',
        n_traces=n_traces,
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        pick_time_s_sorted=dataset.pick_time_s,
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=dataset.valid_mask,
        source_id_sorted=dataset.source_id,
        receiver_id_sorted=dataset.receiver_id,
        source_x_m_sorted=dataset.source_x_m,
        source_y_m_sorted=dataset.source_y_m,
        receiver_x_m_sorted=dataset.receiver_x_m,
        receiver_y_m_sorted=dataset.receiver_y_m,
        source_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=dataset.offset_m,
        offset_m_sorted=None,
        distance_m_sorted=dataset.offset_m,
        source_endpoint_key_sorted=np.asarray(
            [f'source:{int(value)}' for value in dataset.source_id],
            dtype='<U32',
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'receiver:{int(value)}' for value in dataset.receiver_id],
            dtype='<U32',
        ),
        source_node_id_sorted=dataset.source_node_id,
        receiver_node_id_sorted=dataset.receiver_node_id,
        node_x_m=np.ascontiguousarray(node_x_m, dtype=np.float64),
        node_y_m=np.ascontiguousarray(node_y_m, dtype=np.float64),
        node_elevation_m=node_elevation_m,
        node_kind=node_kind,
        rejection_reason_sorted=np.full(n_traces, 'ok', dtype='<U32'),
        qc={},
        endpoint_table=endpoint_table,
        metadata={'synthetic_model': 'clean_3d_cell_refraction'},
    )


def _endpoint_table_from_dataset(
    dataset: SyntheticRefractionCellDataset,
) -> RefractionEndpointTable:
    node_id = np.concatenate(
        (dataset.source_endpoint_node_id, dataset.receiver_endpoint_node_id)
    )
    endpoint_id = np.concatenate(
        (dataset.source_endpoint_id, dataset.receiver_endpoint_id)
    )
    endpoint_x_m = np.concatenate(
        (dataset.source_endpoint_x_m, dataset.receiver_endpoint_x_m)
    )
    endpoint_y_m = np.concatenate(
        (dataset.source_endpoint_y_m, dataset.receiver_endpoint_y_m)
    )
    kind = np.concatenate(
        (
            np.full(dataset.source_endpoint_id.shape, 'source', dtype='<U16'),
            np.full(dataset.receiver_endpoint_id.shape, 'receiver', dtype='<U16'),
        )
    )
    return RefractionEndpointTable(
        node_id=np.ascontiguousarray(node_id, dtype=np.int64),
        endpoint_id=np.ascontiguousarray(endpoint_id, dtype=np.int64),
        x_m=np.ascontiguousarray(endpoint_x_m, dtype=np.float64),
        y_m=np.ascontiguousarray(endpoint_y_m, dtype=np.float64),
        elevation_m=np.zeros(node_id.shape, dtype=np.float64),
        kind=kind,
        pick_count=np.zeros(node_id.shape, dtype=np.int64),
    )


def _required_array(values: np.ndarray | None) -> np.ndarray:
    assert values is not None
    return values


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))
