from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import (
    RefractionStaticApplyRequest,
    RefractionStaticModelRequest,
    RefractionStaticSolverRequest,
)
from app.statics.refraction.artifacts import (
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
    write_refraction_static_artifacts,
)
from seis_statics.refraction.cell_regularization import (
    build_cell_slowness_smoothing_rows,
)
from app.statics.refraction.application.datum import build_refraction_datum_statics
from app.statics.refraction.application.design_matrix import (
    build_refraction_static_design_matrix,
)
from app.statics.refraction.domain.solver import solve_refraction_static_bounded_ls
from app.statics.refraction.domain.types import (
    RefractionDatumStaticsResult,
    RefractionEndpointTable,
    RefractionStaticInputModel,
    RefractionStaticSolverResult,
)
from app.statics.refraction.application.weathering_replacement import (
    compute_weathering_replacement_statics_from_first_breaks,
)
from app.tests.fixtures.refraction_synthetic import (
    SyntheticRefractionCellDataset,
    make_clean_2d_cell_refraction_dataset,
    make_clean_3d_cell_refraction_dataset,
    make_rotated_2d_line_refraction_dataset,
)

SMOOTH_2D_V2_M_S = np.asarray(
    [2200.0, 2300.0, 2400.0, 2500.0, 2600.0],
    dtype=np.float64,
)
SMOOTH_3D_V2_M_S = np.asarray(
    [
        [2200.0, 2300.0, 2400.0],
        [2250.0, 2350.0, 2450.0],
        [2300.0, 2400.0, 2500.0],
    ],
    dtype=np.float64,
)
MIN_OBSERVATIONS_PER_CELL = 5


def test_cell_v2_smoothing_reduces_synthetic_velocity_spike() -> None:
    dataset = make_clean_2d_cell_refraction_dataset(
        seed=436,
        cell_v2_m_s=SMOOTH_2D_V2_M_S,
        n_sources=20,
        n_receivers=20,
        noise_std_s=0.0,
        outlier_fraction=0.0,
    )
    center_cell_id = 2
    perturbation_s = np.where(
        dataset.true_cell_id_for_pick == center_cell_id,
        -0.012,
        0.0,
    )
    spiky_dataset = replace(
        dataset,
        pick_time_s=np.ascontiguousarray(
            dataset.pick_time_s + perturbation_s,
            dtype=np.float64,
        ),
    )

    unsmoothed = _solve_cell_dataset(spiky_dataset, velocity_smoothing_weight=0.0)
    smoothed = _solve_cell_dataset(spiky_dataset, velocity_smoothing_weight=4.0)

    unsmoothed_v2 = _required_array(unsmoothed.cell_bedrock_velocity_m_s)
    smoothed_v2 = _required_array(smoothed.cell_bedrock_velocity_m_s)
    np.testing.assert_array_equal(unsmoothed.active_cell_id, np.arange(5))
    np.testing.assert_array_equal(smoothed.active_cell_id, np.arange(5))

    unsmoothed_spike = _center_spike_amplitude(unsmoothed_v2, center_cell_id)
    smoothed_spike = _center_spike_amplitude(smoothed_v2, center_cell_id)
    assert smoothed_spike < unsmoothed_spike
    assert abs(smoothed_v2[center_cell_id] - SMOOTH_2D_V2_M_S[center_cell_id]) < abs(
        unsmoothed_v2[center_cell_id] - SMOOTH_2D_V2_M_S[center_cell_id]
    )

    neighbor_cell_id = np.asarray([0, 1, 3, 4], dtype=np.int64)
    neighbor_deviation_m_s = np.abs(
        smoothed_v2[neighbor_cell_id] - SMOOTH_2D_V2_M_S[neighbor_cell_id]
    )
    assert float(np.max(neighbor_deviation_m_s)) < 150.0
    assert smoothed.qc['velocity_smoothing_weight'] == pytest.approx(4.0)
    assert smoothed.qc['n_cell_smoothing_rows'] == 4


def test_cell_v2_smoothing_preserves_smooth_velocity_trend() -> None:
    dataset = make_clean_2d_cell_refraction_dataset(
        seed=437,
        cell_v2_m_s=SMOOTH_2D_V2_M_S,
        n_sources=20,
        n_receivers=20,
        noise_std_s=0.0,
        outlier_fraction=0.0,
    )

    unsmoothed = _solve_cell_dataset(dataset, velocity_smoothing_weight=0.0)
    smoothed = _solve_cell_dataset(dataset, velocity_smoothing_weight=2.0)

    np.testing.assert_allclose(
        _required_array(unsmoothed.cell_bedrock_velocity_m_s),
        SMOOTH_2D_V2_M_S,
        atol=1.0e-5,
        rtol=0.0,
    )
    smoothed_v2 = _required_array(smoothed.cell_bedrock_velocity_m_s)
    assert bool(np.all(np.diff(smoothed_v2) > 0.0))
    assert float(np.max(np.abs(smoothed_v2 - SMOOTH_2D_V2_M_S))) < 60.0
    assert smoothed.qc['n_cell_smoothing_rows'] == 4


def test_cell_v2_smoothing_line_2d_uses_inline_neighbors_only() -> None:
    dataset = make_rotated_2d_line_refraction_dataset(
        seed=438,
        cell_v2_m_s=SMOOTH_2D_V2_M_S,
        n_sources=20,
        n_receivers=20,
        noise_std_s=0.0,
        outlier_fraction=0.0,
        line_origin_x_m=1000.0,
        line_origin_y_m=2000.0,
        line_azimuth_deg=37.0,
    )
    dataset = _with_valid_observations_by_cell(
        dataset,
        {
            1: MIN_OBSERVATIONS_PER_CELL - 1,
            4: 0,
        },
    )

    result = _solve_cell_dataset(dataset, velocity_smoothing_weight=1.0)
    active_cell_id = _required_array(result.active_cell_id)
    rows = build_cell_slowness_smoothing_rows(
        active_cell_id=active_cell_id,
        velocity_smoothing_weight=1.0,
        smoothing_reference_distance_m=dataset.cell_size_x_m,
        n_total_cells=int(dataset.true_cell_v2_m_s.size),
        number_of_cell_x=int(dataset.true_cell_v2_m_s.shape[1]),
        number_of_cell_y=1,
        n_parameters=int(active_cell_id.shape[0]),
    )

    np.testing.assert_array_equal(active_cell_id, [0, 2, 3])
    np.testing.assert_array_equal(result.inactive_cell_id, [1, 4])
    np.testing.assert_array_equal(
        rows.edge_cell_id,
        [[2, 3]],
    )
    np.testing.assert_array_equal(rows.active_cell_neighbor_count, [0, 1, 1])
    assert result.qc['low_fold_cell_id'] == [1]
    assert result.qc['n_low_fold_cells'] == 1
    assert result.qc['n_cell_smoothing_rows'] == 1
    assert result.qc['active_cell_neighbor_count_min'] == 0
    assert result.qc['active_cell_neighbor_count_median'] == pytest.approx(1.0)
    assert result.qc['active_cell_neighbor_count_max'] == 1


def test_cell_v2_smoothing_grid_3d_uses_expected_cardinal_neighbors() -> None:
    dataset = make_clean_3d_cell_refraction_dataset(
        seed=439,
        cell_v2_m_s=SMOOTH_3D_V2_M_S,
        n_sources=24,
        n_receivers=24,
        cell_size_x_m=100.0,
        cell_size_y_m=120.0,
        noise_std_s=0.0,
        outlier_fraction=0.0,
    )
    dataset = _with_valid_observations_by_cell(
        dataset,
        {
            2: MIN_OBSERVATIONS_PER_CELL - 1,
            6: 0,
        },
    )

    result = _solve_cell_dataset(dataset, velocity_smoothing_weight=1.0)
    active_cell_id = _required_array(result.active_cell_id)
    rows = build_cell_slowness_smoothing_rows(
        active_cell_id=active_cell_id,
        velocity_smoothing_weight=1.0,
        smoothing_reference_distance_m=dataset.cell_size_x_m,
        n_total_cells=int(dataset.true_cell_v2_m_s.size),
        number_of_cell_x=int(dataset.true_cell_v2_m_s.shape[1]),
        number_of_cell_y=int(dataset.true_cell_v2_m_s.shape[0]),
        n_parameters=int(active_cell_id.shape[0]),
    )

    np.testing.assert_array_equal(active_cell_id, [0, 1, 3, 4, 5, 7, 8])
    np.testing.assert_array_equal(result.inactive_cell_id, [2, 6])
    np.testing.assert_array_equal(
        rows.edge_cell_id,
        [
            [0, 1],
            [0, 3],
            [1, 4],
            [3, 4],
            [4, 5],
            [4, 7],
            [5, 8],
            [7, 8],
        ],
    )
    np.testing.assert_array_equal(
        rows.active_cell_neighbor_count,
        [2, 2, 2, 4, 2, 2, 2],
    )
    assert result.qc['low_fold_cell_id'] == [2]
    assert result.qc['n_low_fold_cells'] == 1
    assert result.qc['n_cell_smoothing_rows'] == 8
    assert result.qc['active_cell_neighbor_count_min'] == 2
    assert result.qc['active_cell_neighbor_count_median'] == pytest.approx(2.0)
    assert result.qc['active_cell_neighbor_count_max'] == 4


def test_cell_v2_smoothing_qc_records_weight_and_row_count(
    tmp_path: Path,
) -> None:
    dataset = make_clean_2d_cell_refraction_dataset(
        seed=440,
        cell_v2_m_s=SMOOTH_2D_V2_M_S,
        n_sources=20,
        n_receivers=20,
        noise_std_s=0.0,
        outlier_fraction=0.0,
    )
    req = _apply_request_from_dataset(dataset, velocity_smoothing_weight=3.0)
    result = _run_cell_refraction_statics(dataset, req=req)

    write_refraction_static_artifacts(result=result, req=req, job_dir=tmp_path)
    cell_qc = json.loads(
        (tmp_path / REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME).read_text(
            encoding='utf-8'
        )
    )
    assert cell_qc['velocity_smoothing_weight'] == pytest.approx(3.0)
    assert cell_qc['smoothing_reference_distance_m'] == pytest.approx(
        dataset.cell_size_x_m
    )
    assert cell_qc['n_cell_smoothing_rows'] == 4


def _solve_cell_dataset(
    dataset: SyntheticRefractionCellDataset,
    *,
    velocity_smoothing_weight: float,
) -> RefractionStaticSolverResult:
    model = _model_from_dataset(
        dataset,
        velocity_smoothing_weight=velocity_smoothing_weight,
    )
    design = build_refraction_static_design_matrix(
        input_model=_input_model_from_dataset(dataset),
        model=model,
    )
    return solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=model,
        solver=_solver(),
    )


def _run_cell_refraction_statics(
    dataset: SyntheticRefractionCellDataset,
    *,
    req: RefractionStaticApplyRequest,
) -> RefractionDatumStaticsResult:
    replacement = compute_weathering_replacement_statics_from_first_breaks(
        req=req,
        state=None,
        input_model=_input_model_from_dataset(dataset),
    )
    return build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=req.datum,
        apply_options=req.apply,
        state=None,
        file_id=req.file_id,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
    )


def _apply_request_from_dataset(
    dataset: SyntheticRefractionCellDataset,
    *,
    velocity_smoothing_weight: float,
) -> RefractionStaticApplyRequest:
    return RefractionStaticApplyRequest.model_validate(
        {
            'file_id': 'synthetic-cell-smoothing',
            'key1_byte': 189,
            'key2_byte': 193,
            'pick_source': {
                'kind': 'batch_predicted_npz',
                'job_id': 'synthetic-cell-smoothing-first-breaks',
                'artifact_name': 'predicted_picks_time_s.npz',
            },
            'linkage': {'mode': 'none'},
            'model': _model_from_dataset(
                dataset,
                velocity_smoothing_weight=velocity_smoothing_weight,
            ).model_dump(mode='json'),
            'moveout': {
                'model': 'head_wave_linear_offset',
                'distance_source': 'geometry',
                'offset_byte': None,
                'min_offset_m': None,
                'max_offset_m': None,
                'allow_missing_offset': False,
                'max_geometry_offset_mismatch_m': None,
            },
            'solver': _solver().model_dump(mode='json'),
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


def _model_from_dataset(
    dataset: SyntheticRefractionCellDataset,
    *,
    velocity_smoothing_weight: float,
) -> RefractionStaticModelRequest:
    return RefractionStaticModelRequest.model_validate(
        {
            'method': 'gli_variable_thickness',
            'weathering_velocity_m_s': dataset.true_v1_m_s,
            'bedrock_velocity_mode': 'solve_cell',
            'bedrock_velocity_m_s': None,
            'initial_bedrock_velocity_m_s': 2400.0,
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
                'coordinate_mode': dataset.coordinate_mode,
                'line_origin_x_m': dataset.line_origin_x_m,
                'line_origin_y_m': dataset.line_origin_y_m,
                'line_azimuth_deg': dataset.line_azimuth_deg,
                'min_observations_per_cell': MIN_OBSERVATIONS_PER_CELL,
                'velocity_smoothing_weight': velocity_smoothing_weight,
                'smoothing_reference_distance_m': (
                    dataset.cell_size_x_m
                    if velocity_smoothing_weight > 0.0
                    else None
                ),
            },
        }
    )


def _solver() -> RefractionStaticSolverRequest:
    return RefractionStaticSolverRequest.model_validate(
        {
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
    node_kind = np.concatenate(
        (
            np.full(dataset.source_endpoint_id.shape, 'source', dtype='<U16'),
            np.full(dataset.receiver_endpoint_id.shape, 'receiver', dtype='<U16'),
        )
    )
    return RefractionStaticInputModel(
        file_id='synthetic-cell-smoothing',
        n_traces=n_traces,
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        pick_time_s_sorted=np.ascontiguousarray(dataset.pick_time_s, dtype=np.float64),
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=np.ascontiguousarray(
            dataset.valid_mask,
            dtype=bool,
        ),
        source_id_sorted=np.ascontiguousarray(dataset.source_id, dtype=np.int64),
        receiver_id_sorted=np.ascontiguousarray(dataset.receiver_id, dtype=np.int64),
        source_x_m_sorted=np.ascontiguousarray(dataset.source_x_m, dtype=np.float64),
        source_y_m_sorted=np.ascontiguousarray(dataset.source_y_m, dtype=np.float64),
        receiver_x_m_sorted=np.ascontiguousarray(
            dataset.receiver_x_m,
            dtype=np.float64,
        ),
        receiver_y_m_sorted=np.ascontiguousarray(
            dataset.receiver_y_m,
            dtype=np.float64,
        ),
        source_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=np.ascontiguousarray(
            dataset.offset_m,
            dtype=np.float64,
        ),
        offset_m_sorted=None,
        distance_m_sorted=np.ascontiguousarray(dataset.offset_m, dtype=np.float64),
        source_endpoint_key_sorted=np.asarray(
            [f'source:{int(value)}' for value in dataset.source_id],
            dtype='<U32',
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'receiver:{int(value)}' for value in dataset.receiver_id],
            dtype='<U32',
        ),
        source_node_id_sorted=np.ascontiguousarray(
            dataset.source_node_id,
            dtype=np.int64,
        ),
        receiver_node_id_sorted=np.ascontiguousarray(
            dataset.receiver_node_id,
            dtype=np.int64,
        ),
        node_x_m=np.ascontiguousarray(node_x_m, dtype=np.float64),
        node_y_m=np.ascontiguousarray(node_y_m, dtype=np.float64),
        node_elevation_m=np.zeros(endpoint_table.node_id.shape, dtype=np.float64),
        node_kind=node_kind,
        rejection_reason_sorted=np.full(n_traces, 'ok', dtype='<U32'),
        qc={},
        endpoint_table=endpoint_table,
        metadata={'synthetic_model': 'cell_v2_smoothing'},
    )


def _with_valid_observations_by_cell(
    dataset: SyntheticRefractionCellDataset,
    valid_observations_by_cell: dict[int, int],
) -> SyntheticRefractionCellDataset:
    valid_mask = np.ones(dataset.valid_mask.shape, dtype=bool)
    for raw_cell_id, raw_count in valid_observations_by_cell.items():
        cell_id = int(raw_cell_id)
        count = int(raw_count)
        row_index = np.flatnonzero(dataset.true_cell_id_for_pick == cell_id)
        if count < 0 or count > int(row_index.shape[0]):
            raise ValueError('valid observation count must fit available cell rows')
        valid_mask[row_index] = False
        valid_mask[row_index[:count]] = True
    return replace(
        dataset,
        valid_mask=np.ascontiguousarray(valid_mask, dtype=bool),
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


def _center_spike_amplitude(velocity_m_s: np.ndarray, center_cell_id: int) -> float:
    neighbor_average = 0.5 * (
        float(velocity_m_s[center_cell_id - 1])
        + float(velocity_m_s[center_cell_id + 1])
    )
    return abs(float(velocity_m_s[center_cell_id]) - neighbor_average)


def _required_array(value: np.ndarray | None) -> np.ndarray:
    assert value is not None
    return value
