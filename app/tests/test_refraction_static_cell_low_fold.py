from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.statics.refraction.artifacts import (
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
    build_refraction_refractor_velocity_grid_arrays,
    write_refraction_static_artifacts,
)
from app.statics.refraction.application.design_matrix import (
    build_refraction_static_design_matrix,
)
from app.statics.refraction.application.datum import build_refraction_datum_statics
from app.statics.refraction.domain.types import (
    RefractionDatumStaticsResult,
    RefractionEndpointTable,
    RefractionStaticInputModel,
)
from app.statics.refraction.application.weathering_replacement import (
    compute_weathering_replacement_statics_from_first_breaks,
)
from app.tests.fixtures.refraction_synthetic import (
    SyntheticRefractionCellDataset,
    make_low_fold_empty_cell_refraction_dataset,
)

MIN_OBSERVATIONS_PER_CELL = 20
ACTIVE_CELL_ID = np.asarray([0, 3], dtype=np.int64)
LOW_FOLD_CELL_ID = 1
EMPTY_CELL_ID = 2


def test_min_observations_per_cell_excludes_low_fold_cells() -> None:
    dataset, req, input_model = _low_fold_case()

    design = build_refraction_static_design_matrix(
        input_model=input_model,
        model=req.model,
    )

    np.testing.assert_array_equal(
        dataset.cell_observation_count,
        [22, 19, 0, 22],
    )
    np.testing.assert_array_equal(design.active_cell_id, ACTIVE_CELL_ID)
    np.testing.assert_array_equal(design.inactive_cell_id, [1, 2])
    assert design.cell_id_to_col == {
        0: design.n_active_nodes,
        3: design.n_active_nodes + 1,
    }
    assert LOW_FOLD_CELL_ID not in design.cell_id_to_col
    assert EMPTY_CELL_ID not in design.cell_id_to_col
    assert design.rejection_reason_sorted is not None
    low_fold_rows = dataset.true_cell_id_for_pick == LOW_FOLD_CELL_ID
    assert np.all(
        design.rejection_reason_sorted[low_fold_rows]
        == 'below_min_observations_per_cell'
    )
    assert not np.any(
        np.isin(design.row_trace_index_sorted, np.flatnonzero(low_fold_rows))
    )
    assert design.qc['min_observations_per_cell'] == MIN_OBSERVATIONS_PER_CELL
    assert design.qc['n_low_fold_cells'] == 1
    assert design.qc['low_fold_cell_id'] == [LOW_FOLD_CELL_ID]
    assert design.qc['n_observations_rejected_by_low_fold_cell'] == 19
    assert design.qc['low_fold_cell_rejection_reason'] == (
        'below_min_observations_per_cell'
    )


def test_low_fold_and_empty_cells_are_reported_in_cell_artifacts(
    tmp_path: Path,
) -> None:
    dataset, req, input_model = _low_fold_case()
    result = _run_refraction_statics(req=req, input_model=input_model)

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    assert paths.refraction_refractor_velocity_cells_csv is not None
    rows = {
        int(row['cell_id']): row
        for row in _read_csv(paths.refraction_refractor_velocity_cells_csv)
    }
    assert [rows[index]['velocity_status'] for index in range(4)] == [
        'solved',
        'low_fold',
        'inactive',
        'solved',
    ]
    assert rows[LOW_FOLD_CELL_ID]['active'] == 'false'
    assert rows[LOW_FOLD_CELL_ID]['n_observations'] == '19'
    assert rows[LOW_FOLD_CELL_ID]['n_used_observations'] == '0'
    assert rows[LOW_FOLD_CELL_ID]['n_rejected_observations'] == '19'
    assert rows[LOW_FOLD_CELL_ID]['v2_m_s'] == ''
    assert rows[EMPTY_CELL_ID]['active'] == 'false'
    assert rows[EMPTY_CELL_ID]['n_observations'] == '0'
    assert rows[EMPTY_CELL_ID]['n_used_observations'] == '0'
    assert rows[EMPTY_CELL_ID]['v2_m_s'] == ''

    assert paths.refraction_refractor_velocity_grid_npz is not None
    assert paths.refraction_refractor_velocity_grid_npz.name == (
        REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME
    )
    with np.load(
        paths.refraction_refractor_velocity_grid_npz,
        allow_pickle=False,
    ) as grid:
        np.testing.assert_array_equal(
            grid['n_observations_per_cell'],
            dataset.cell_observation_count,
        )
        np.testing.assert_array_equal(
            grid['velocity_status'].astype(str),
            ['solved', 'low_fold', 'inactive', 'solved'],
        )
        assert np.isnan(float(grid['v2_m_s'][LOW_FOLD_CELL_ID]))
        assert np.isnan(float(grid['v2_m_s'][EMPTY_CELL_ID]))

    assert paths.refraction_refractor_velocity_qc_json is not None
    assert paths.refraction_refractor_velocity_qc_json.name == (
        REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME
    )
    cell_qc = json.loads(
        paths.refraction_refractor_velocity_qc_json.read_text(encoding='utf-8')
    )
    assert cell_qc['min_observations_per_cell'] == MIN_OBSERVATIONS_PER_CELL
    assert cell_qc['n_low_fold_cells'] == 1
    assert cell_qc['n_observations_rejected_by_low_fold_cell'] == 19
    assert cell_qc['low_fold_cell_rejection_reason'] == (
        'below_min_observations_per_cell'
    )


def test_outside_grid_observations_have_distinct_rejection_reason() -> None:
    _, req, input_model = _low_fold_case()
    input_with_outside = _append_outside_grid_observation(input_model)

    design = build_refraction_static_design_matrix(
        input_model=input_with_outside,
        model=req.model,
    )

    outside_trace_index = input_with_outside.n_traces - 1
    assert outside_trace_index not in design.row_trace_index_sorted.tolist()
    assert design.rejection_reason_sorted is not None
    assert (
        design.rejection_reason_sorted[outside_trace_index]
        == 'outside_refractor_cell_grid'
    )
    assert design.qc['n_observations_outside_grid'] == 1
    assert design.qc['n_observations_rejected_by_low_fold_cell'] == 19
    assert design.qc['low_fold_cell_rejection_reason'] == (
        'below_min_observations_per_cell'
    )


def test_low_fold_and_empty_cells_do_not_emit_solved_velocity_without_fallback() -> None:
    _, req, input_model = _low_fold_case()
    result = _run_refraction_statics(req=req, input_model=input_model)

    np.testing.assert_array_equal(result.active_cell_id, ACTIVE_CELL_ID)
    np.testing.assert_array_equal(result.inactive_cell_id, [1, 2])
    assert result.cell_bedrock_velocity_m_s is not None
    assert result.cell_bedrock_velocity_m_s.shape == ACTIVE_CELL_ID.shape
    np.testing.assert_array_equal(result.cell_velocity_status, ['solved', 'solved'])

    arrays = build_refraction_refractor_velocity_grid_arrays(
        result=result,
        req=req,
    )
    velocity_status = arrays['velocity_status'].astype(str)
    assert velocity_status[LOW_FOLD_CELL_ID] == 'low_fold'
    assert velocity_status[EMPTY_CELL_ID] == 'inactive'
    assert np.isnan(float(arrays['v2_m_s'][LOW_FOLD_CELL_ID]))
    assert np.isnan(float(arrays['v2_m_s'][EMPTY_CELL_ID]))
    assert result.qc['low_fold_cell_id'] == [LOW_FOLD_CELL_ID]
    assert result.qc['n_low_fold_cells'] == 1


def _low_fold_case() -> tuple[
    SyntheticRefractionCellDataset,
    RefractionStaticApplyRequest,
    RefractionStaticInputModel,
]:
    dataset = make_low_fold_empty_cell_refraction_dataset(
        seed=434,
        n_sources=16,
        n_receivers=16,
        min_observations_per_cell=MIN_OBSERVATIONS_PER_CELL,
    )
    req = _cell_apply_request(dataset)
    return dataset, req, _input_model_from_dataset(dataset)


def _cell_apply_request(
    dataset: SyntheticRefractionCellDataset,
) -> RefractionStaticApplyRequest:
    return RefractionStaticApplyRequest.model_validate(
        {
            'file_id': 'low-fold-empty-cell-synthetic',
            'key1_byte': 189,
            'key2_byte': 193,
            'pick_source': {
                'kind': 'batch_predicted_npz',
                'job_id': 'low-fold-empty-cell-first-breaks',
                'artifact_name': 'predicted_picks_time_s.npz',
            },
            'linkage': {'mode': 'none'},
            'model': {
                'method': 'gli_variable_thickness',
                'weathering_velocity_m_s': dataset.true_v1_m_s,
                'bedrock_velocity_mode': 'solve_cell',
                'bedrock_velocity_m_s': None,
                'initial_bedrock_velocity_m_s': 2600.0,
                'min_bedrock_velocity_m_s': 1200.0,
                'max_bedrock_velocity_m_s': 6000.0,
                'max_weathering_thickness_m': None,
                'refractor_cell': {
                    'number_of_cell_x': int(dataset.true_cell_v2_m_s.shape[1]),
                    'size_of_cell_x_m': dataset.cell_size_x_m,
                    'x_coordinate_origin_m': dataset.x_coordinate_origin_m,
                    'number_of_cell_y': 1,
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
        file_id='low-fold-empty-cell-synthetic',
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
        metadata={'synthetic_model': 'low_fold_empty_cell_refraction'},
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


def _append_outside_grid_observation(
    input_model: RefractionStaticInputModel,
) -> RefractionStaticInputModel:
    source_node_id = int(input_model.source_node_id_sorted[0])
    receiver_node_id = int(input_model.receiver_node_id_sorted[0])
    source_x_m = -200.0
    receiver_x_m = -100.0
    distance_m = abs(receiver_x_m - source_x_m)
    pick_time_s = float(input_model.pick_time_s_sorted[0])
    n_traces = int(input_model.n_traces + 1)
    return replace(
        input_model,
        n_traces=n_traces,
        sorted_trace_index=_append(input_model.sorted_trace_index, n_traces - 1),
        pick_time_s_sorted=_append(input_model.pick_time_s_sorted, pick_time_s),
        valid_pick_mask_sorted=_append(input_model.valid_pick_mask_sorted, True),
        valid_observation_mask_sorted=_append(
            input_model.valid_observation_mask_sorted,
            True,
        ),
        source_id_sorted=_append(
            input_model.source_id_sorted,
            input_model.source_id_sorted[0],
        ),
        receiver_id_sorted=_append(
            input_model.receiver_id_sorted,
            input_model.receiver_id_sorted[0],
        ),
        source_x_m_sorted=_append(input_model.source_x_m_sorted, source_x_m),
        source_y_m_sorted=_append(input_model.source_y_m_sorted, 0.0),
        receiver_x_m_sorted=_append(input_model.receiver_x_m_sorted, receiver_x_m),
        receiver_y_m_sorted=_append(input_model.receiver_y_m_sorted, 0.0),
        source_elevation_m_sorted=_append(
            input_model.source_elevation_m_sorted,
            0.0,
        ),
        receiver_elevation_m_sorted=_append(
            input_model.receiver_elevation_m_sorted,
            0.0,
        ),
        geometry_distance_m_sorted=_append(
            input_model.geometry_distance_m_sorted,
            distance_m,
        ),
        distance_m_sorted=_append(input_model.distance_m_sorted, distance_m),
        source_endpoint_key_sorted=_append(
            input_model.source_endpoint_key_sorted,
            f'source:{source_node_id}',
        ),
        receiver_endpoint_key_sorted=_append(
            input_model.receiver_endpoint_key_sorted,
            f'receiver:{receiver_node_id}',
        ),
        source_node_id_sorted=_append(
            input_model.source_node_id_sorted,
            source_node_id,
        ),
        receiver_node_id_sorted=_append(
            input_model.receiver_node_id_sorted,
            receiver_node_id,
        ),
        rejection_reason_sorted=_append(input_model.rejection_reason_sorted, 'ok'),
    )


def _append(values: np.ndarray, value: object) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate((values, np.asarray([value], dtype=values.dtype))),
        dtype=values.dtype,
    )


def _run_refraction_statics(
    *,
    req: RefractionStaticApplyRequest,
    input_model: RefractionStaticInputModel,
) -> RefractionDatumStaticsResult:
    replacement = compute_weathering_replacement_statics_from_first_breaks(
        req=req,
        state=None,
        input_model=input_model,
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))
