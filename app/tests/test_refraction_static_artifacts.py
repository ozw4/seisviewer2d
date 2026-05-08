from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.services.refraction_static_artifacts as artifact_module
from app.api.schemas import RefractionStaticApplyRequest
from app.main import app
from app.services.refraction_static_artifacts import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    REFRACTION_STATICS_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    RefractionStaticArtifactError,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    write_refraction_static_solution_npz,
    write_refraction_static_artifacts,
)
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
    ResolvedRefractionFirstLayer,
)


REQUIRED_TRACE_ARRAYS = {
    'sorted_trace_index',
    'valid_observation_mask_sorted',
    'used_observation_mask_sorted',
    'trace_static_valid_mask_sorted',
    'source_node_id_sorted',
    'receiver_node_id_sorted',
    'source_surface_elevation_m_sorted',
    'receiver_surface_elevation_m_sorted',
    'source_floating_datum_elevation_m_sorted',
    'receiver_floating_datum_elevation_m_sorted',
    'source_weathering_thickness_m_sorted',
    'receiver_weathering_thickness_m_sorted',
    'source_refractor_elevation_m_sorted',
    'receiver_refractor_elevation_m_sorted',
    'source_half_intercept_time_s_sorted',
    'receiver_half_intercept_time_s_sorted',
    'weathering_replacement_trace_shift_s_sorted',
    'floating_datum_elevation_shift_s_sorted',
    'flat_datum_shift_s_sorted',
    'refraction_trace_shift_s_sorted',
    'estimated_first_break_time_s_sorted',
    'first_break_residual_s_sorted',
    'trace_static_status_sorted',
}

REQUIRED_NODE_ARRAYS = {
    'node_id',
    'node_x_m',
    'node_y_m',
    'node_surface_elevation_m',
    'node_floating_datum_elevation_m',
    'node_refractor_elevation_m',
    'node_weathering_thickness_m',
    'node_half_intercept_time_s',
    'node_weathering_replacement_shift_s',
    'node_t1_time_s',
    'node_sh1_weathering_thickness_m',
    'node_weathering_correction_s',
    'node_solution_status',
    'node_weathering_status',
    'node_datum_status',
    'node_pick_count',
    'node_used_pick_count',
    'node_rejected_pick_count',
    'node_residual_rms_s',
    'node_residual_mad_s',
}

REQUIRED_ROW_ARRAYS = {
    'row_trace_index_sorted',
    'row_source_node_id',
    'row_receiver_node_id',
    'row_distance_m',
    'observed_pick_time_s',
    'modeled_pick_time_s',
    'residual_time_s',
    'used_row_mask',
    'rejected_by_robust_mask',
}

EXPECTED_FILENAMES = {
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
}


def _request() -> RefractionStaticApplyRequest:
    return RefractionStaticApplyRequest.model_validate(
        {
            'file_id': 'raw-file-id',
            'key1_byte': 189,
            'key2_byte': 193,
            'pick_source': {
                'kind': 'batch_predicted_npz',
                'job_id': 'pick-job',
                'artifact_name': 'predicted_picks_time_s.npz',
            },
            'linkage': {'mode': 'none'},
            'model': {
                'method': 'gli_variable_thickness',
                'weathering_velocity_m_s': 800.0,
                'bedrock_velocity_mode': 'solve_global',
            },
            'datum': {
                'mode': 'floating_and_flat',
                'floating_datum_mode': 'constant',
                'floating_datum_elevation_m': 120.0,
                'flat_datum_elevation_m': 300.0,
            },
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


def _estimated_v1_request() -> RefractionStaticApplyRequest:
    payload = _request().model_dump(mode='json')
    payload['model']['weathering_velocity_m_s'] = None
    payload['model']['first_layer'] = {
        'mode': 'estimate_direct_arrival',
        'weathering_velocity_m_s': 812.5,
        'min_direct_offset_m': 20.0,
        'max_direct_offset_m': 140.0,
    }
    return RefractionStaticApplyRequest.model_validate(payload)


def _resolved_estimated_v1() -> ResolvedRefractionFirstLayer:
    return ResolvedRefractionFirstLayer(
        mode='estimate_direct_arrival',
        weathering_velocity_m_s=812.5,
        status='estimated',
        qc={
            'v1_mode': 'estimate_direct_arrival',
            'weathering_velocity_m_s': 812.5,
            'resolved_weathering_velocity_m_s': 812.5,
            'v1_status': 'estimated',
        },
    )


def _result() -> RefractionDatumStaticsResult:
    n_traces = 4
    sorted_trace_index = np.arange(n_traces, dtype=np.int64)
    valid_observation = np.asarray([True, True, True, False], dtype=bool)
    used_observation = np.asarray([True, False, True, False], dtype=bool)
    trace_valid = np.asarray([True, True, False, True], dtype=bool)
    trace_status = np.asarray(
        ['ok', 'ok', 'not_observed', 'ok'],
        dtype='<U32',
    )

    node_id = np.asarray([0, 1, 2], dtype=np.int64)
    node_surface = np.asarray([100.0, 110.0, 120.0], dtype=np.float64)
    node_thickness = np.asarray([10.0, 12.0, 14.0], dtype=np.float64)
    node_half = np.asarray([0.010, 0.012, 0.014], dtype=np.float64)
    node_weathering_shift = np.asarray([-0.0085, -0.0102, -0.0119])

    source_node_sorted = np.asarray([0, 1, 0, 1], dtype=np.int64)
    receiver_node_sorted = np.asarray([1, 2, 2, 1], dtype=np.int64)
    source_half_sorted = np.asarray([0.010, 0.012, 0.010, 0.012])
    receiver_half_sorted = np.asarray([0.012, 0.014, 0.014, 0.012])
    source_thickness_sorted = np.asarray([10.0, 12.0, 10.0, 12.0])
    receiver_thickness_sorted = np.asarray([12.0, 14.0, 14.0, 12.0])
    source_weathering_sorted = np.asarray([-0.0085, -0.0102, -0.0085, -0.0102])
    receiver_weathering_sorted = np.asarray([-0.0102, -0.0119, -0.0119, -0.0102])
    weathering_trace = source_weathering_sorted + receiver_weathering_sorted
    source_floating_sorted = np.asarray([0.0010, 0.0015, 0.0010, 0.0015])
    receiver_floating_sorted = np.asarray([0.0015, 0.0020, 0.0020, 0.0015])
    source_flat_sorted = np.asarray([0.010, 0.011, 0.010, 0.011])
    receiver_flat_sorted = np.asarray([0.011, 0.012, 0.012, 0.011])
    source_refraction_sorted = (
        source_weathering_sorted + source_floating_sorted + source_flat_sorted
    )
    receiver_refraction_sorted = (
        receiver_weathering_sorted + receiver_floating_sorted + receiver_flat_sorted
    )
    refraction_trace = source_refraction_sorted + receiver_refraction_sorted
    refraction_trace[2] = np.nan

    return RefractionDatumStaticsResult(
        bedrock_velocity_mode='solve_global',
        bedrock_slowness_s_per_m=1.0 / 2500.0,
        bedrock_velocity_m_s=2500.0,
        weathering_velocity_m_s=800.0,
        replacement_slowness_delta_s_per_m=(1.0 / 2500.0) - (1.0 / 800.0),
        datum_mode='floating_and_flat',
        floating_datum_mode='constant',
        flat_datum_elevation_m=300.0,
        node_id=node_id,
        node_x_m=np.asarray([0.0, 50.0, 100.0]),
        node_y_m=np.asarray([0.0, 5.0, 10.0]),
        node_surface_elevation_m=node_surface,
        node_kind=np.asarray(['source', 'both', 'receiver'], dtype='<U16'),
        node_weathering_thickness_m=node_thickness,
        node_refractor_elevation_m=node_surface - node_thickness,
        node_half_intercept_time_s=node_half,
        node_weathering_replacement_shift_s=node_weathering_shift,
        node_floating_datum_elevation_m=np.asarray([120.0, 120.0, 120.0]),
        node_solution_status=np.asarray(['solved', 'solved', 'inactive'], dtype='<U16'),
        node_datum_status=np.asarray(['ok', 'ok', 'inactive'], dtype='<U16'),
        node_weathering_status=np.asarray(
            ['ok', 'zero_thickness', 'inactive'],
            dtype='<U16',
        ),
        node_pick_count=np.asarray([4, 3, 2], dtype=np.int64),
        node_used_pick_count=np.asarray([4, 2, 1], dtype=np.int64),
        node_rejected_pick_count=np.asarray([0, 1, 1], dtype=np.int64),
        node_residual_rms_s=np.asarray([0.001, 0.002, 0.003]),
        node_residual_mad_s=np.asarray([0.0005, 0.0010, 0.0015]),
        source_endpoint_key=np.asarray(['s0', 's1'], dtype='<U2'),
        source_id=np.asarray([100, 101], dtype=np.int64),
        source_node_id=np.asarray([0, 1], dtype=np.int64),
        source_x_m=np.asarray([0.0, 50.0]),
        source_y_m=np.asarray([0.0, 5.0]),
        source_surface_elevation_m=np.asarray([100.0, 110.0]),
        source_half_intercept_time_s=np.asarray([0.010, 0.012]),
        source_weathering_thickness_m=np.asarray([10.0, 12.0]),
        source_refractor_elevation_m=np.asarray([90.0, 98.0]),
        source_floating_datum_elevation_m=np.asarray([120.0, 120.0]),
        source_weathering_replacement_shift_s=np.asarray([-0.0085, -0.0102]),
        source_floating_datum_elevation_shift_s=np.asarray([0.0010, 0.0015]),
        source_flat_datum_shift_s=np.asarray([0.010, 0.011]),
        source_refraction_shift_s=np.asarray([0.0025, 0.0023]),
        source_datum_status=np.asarray(['ok', 'ok'], dtype='<U16'),
        receiver_endpoint_key=np.asarray(['r0', 'r1'], dtype='<U2'),
        receiver_id=np.asarray([200, 201], dtype=np.int64),
        receiver_node_id=np.asarray([1, 2], dtype=np.int64),
        receiver_x_m=np.asarray([55.0, 105.0]),
        receiver_y_m=np.asarray([5.0, 10.0]),
        receiver_surface_elevation_m=np.asarray([110.0, 120.0]),
        receiver_half_intercept_time_s=np.asarray([0.012, 0.014]),
        receiver_weathering_thickness_m=np.asarray([12.0, 14.0]),
        receiver_refractor_elevation_m=np.asarray([98.0, 106.0]),
        receiver_floating_datum_elevation_m=np.asarray([120.0, 120.0]),
        receiver_weathering_replacement_shift_s=np.asarray([-0.0102, -0.0119]),
        receiver_floating_datum_elevation_shift_s=np.asarray([0.0015, 0.0020]),
        receiver_flat_datum_shift_s=np.asarray([0.011, 0.012]),
        receiver_refraction_shift_s=np.asarray([0.0023, 0.0021]),
        receiver_datum_status=np.asarray(['ok', 'ok'], dtype='<U16'),
        sorted_trace_index=sorted_trace_index,
        valid_observation_mask_sorted=valid_observation,
        used_observation_mask_sorted=used_observation,
        source_node_id_sorted=source_node_sorted,
        receiver_node_id_sorted=receiver_node_sorted,
        source_surface_elevation_m_sorted=np.asarray([100.0, 110.0, 100.0, 110.0]),
        receiver_surface_elevation_m_sorted=np.asarray([110.0, 120.0, 120.0, 110.0]),
        source_floating_datum_elevation_m_sorted=np.full(n_traces, 120.0),
        receiver_floating_datum_elevation_m_sorted=np.full(n_traces, 120.0),
        source_weathering_thickness_m_sorted=source_thickness_sorted,
        receiver_weathering_thickness_m_sorted=receiver_thickness_sorted,
        source_refractor_elevation_m_sorted=np.asarray([90.0, 98.0, 90.0, 98.0]),
        receiver_refractor_elevation_m_sorted=np.asarray([98.0, 106.0, 106.0, 98.0]),
        source_half_intercept_time_s_sorted=source_half_sorted,
        receiver_half_intercept_time_s_sorted=receiver_half_sorted,
        source_weathering_replacement_shift_s_sorted=source_weathering_sorted,
        receiver_weathering_replacement_shift_s_sorted=receiver_weathering_sorted,
        source_floating_datum_elevation_shift_s_sorted=source_floating_sorted,
        receiver_floating_datum_elevation_shift_s_sorted=receiver_floating_sorted,
        source_flat_datum_shift_s_sorted=source_flat_sorted,
        receiver_flat_datum_shift_s_sorted=receiver_flat_sorted,
        source_refraction_shift_s_sorted=source_refraction_sorted,
        receiver_refraction_shift_s_sorted=receiver_refraction_sorted,
        weathering_replacement_trace_shift_s_sorted=weathering_trace,
        floating_datum_elevation_shift_s_sorted=(
            source_floating_sorted + receiver_floating_sorted
        ),
        flat_datum_shift_s_sorted=source_flat_sorted + receiver_flat_sorted,
        refraction_trace_shift_s_sorted=refraction_trace,
        trace_static_status_sorted=trace_status,
        trace_static_valid_mask_sorted=trace_valid,
        estimated_first_break_time_s_sorted=np.asarray([0.05, 0.06, 0.07, np.nan]),
        first_break_residual_s_sorted=np.asarray([0.001, -0.002, -0.001, np.nan]),
        row_trace_index_sorted=np.asarray([0, 1, 2], dtype=np.int64),
        row_source_node_id=np.asarray([0, 1, 0], dtype=np.int64),
        row_receiver_node_id=np.asarray([1, 2, 2], dtype=np.int64),
        row_distance_m=np.asarray([100.0, 200.0, 300.0]),
        observed_pick_time_s=np.asarray([0.050, 0.060, 0.070]),
        modeled_pick_time_s=np.asarray([0.049, 0.062, 0.071]),
        residual_time_s=np.asarray([0.001, -0.002, -0.001]),
        used_row_mask=np.asarray([True, False, True], dtype=bool),
        rejected_by_robust_mask=np.asarray([False, True, False], dtype=bool),
        qc={
            'robust_iteration_count': 1,
            'floating_datum_below_refractor_count': 0,
            'flat_datum_below_refractor_count': 0,
        },
    )


def _result_with_weathering_velocity(
    weathering_velocity_m_s: float,
) -> RefractionDatumStaticsResult:
    return replace(
        _result(),
        weathering_velocity_m_s=weathering_velocity_m_s,
        replacement_slowness_delta_s_per_m=(
            (1.0 / 2500.0) - (1.0 / weathering_velocity_m_s)
        ),
    )


def test_write_refraction_static_artifacts_npz_schema(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    assert {path.name for path in tmp_path.iterdir()} == EXPECTED_FILENAMES
    assert paths.artifact_names == (
        REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        REFRACTION_STATIC_QC_JSON_NAME,
        REFRACTION_STATICS_CSV_NAME,
        NEAR_SURFACE_MODEL_CSV_NAME,
        FIRST_BREAK_RESIDUALS_CSV_NAME,
        REFRACTION_STATIC_COMPONENTS_CSV_NAME,
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    )
    with np.load(paths.solution_npz, allow_pickle=False) as data:
        assert data['artifact_version'].item() == '1.0'
        assert data['method'].item() == 'gli_variable_thickness'
        assert data['sign_convention'].item() == 'corrected(t) = raw(t - shift_s)'
        assert data['v1_mode'].item() == 'constant'
        assert data['v1_weathering_velocity_m_s'].item() == pytest.approx(800.0)
        assert data['weathering_velocity_m_s'].item() == pytest.approx(800.0)
        assert data['resolved_weathering_velocity_m_s'].item() == pytest.approx(800.0)
        assert data['bedrock_velocity_m_s'].item() == pytest.approx(2500.0)
        assert data['v2_refractor_velocity_m_s'].item() == pytest.approx(2500.0)
        assert REQUIRED_TRACE_ARRAYS.issubset(data.files)
        assert REQUIRED_NODE_ARRAYS.issubset(data.files)
        assert REQUIRED_ROW_ARRAYS.issubset(data.files)
        assert data['source_endpoint_key'].shape == (2,)
        assert data['receiver_endpoint_key'].shape == (2,)
        assert data['sorted_trace_index'].shape == (4,)
        assert data['node_id'].shape == (3,)
        assert data['row_trace_index_sorted'].shape == (3,)
        assert data['source_node_id_sorted'].dtype == np.int64
        assert data['valid_observation_mask_sorted'].dtype == bool
        assert data['source_surface_elevation_m_sorted'].dtype == np.float64
        assert data['trace_static_status_sorted'].dtype.kind == 'U'
        assert data['trace_static_status_sorted'].tolist() == [
            'ok',
            'ok',
            'not_observed',
            'ok',
        ]
        assert data['node_solution_status'].tolist() == [
            'solved',
            'solved',
            'inactive',
        ]
        assert data['node_weathering_status'].tolist() == [
            'ok',
            'zero_thickness',
            'inactive',
        ]
        assert data['node_datum_status'].tolist() == ['ok', 'ok', 'inactive']
        for key in data.files:
            assert data[key].dtype != object


def test_refraction_static_solution_npz_contains_v1_aliases(tmp_path: Path) -> None:
    path = tmp_path / REFRACTION_STATIC_SOLUTION_NPZ_NAME
    write_refraction_static_solution_npz(
        result=_result_with_weathering_velocity(812.5),
        req=_estimated_v1_request(),
        path=path,
        resolved_first_layer=_resolved_estimated_v1(),
    )

    with np.load(path, allow_pickle=False) as data:
        assert data['v1_mode'].item() == 'estimate_direct_arrival'
        assert data['v1_weathering_velocity_m_s'].item() == pytest.approx(812.5)
        assert data['resolved_weathering_velocity_m_s'].item() == pytest.approx(812.5)


def test_write_refraction_static_artifacts_qc_json(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    payload = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    assert {
        'artifact_version',
        'method',
        'workflow',
        'static_component',
        'sign_convention',
        'request',
        'velocity',
        'datum',
        'observations',
        'nodes',
        'endpoints',
        'first_break_fit',
        'statics',
        'status_counts',
        'artifacts',
        'warnings',
    }.issubset(payload)
    assert payload['request'] == {
        'file_id': 'raw-file-id',
        'key1_byte': 189,
        'key2_byte': 193,
        'pick_source_kind': 'batch_predicted_npz',
        'model_method': 'gli_variable_thickness',
        'apply_mode': 'refraction_from_raw',
        'register_corrected_file': False,
    }
    assert payload['velocity']['bedrock_velocity_status'] == 'solved'
    assert payload['velocity']['v1_mode'] == 'constant'
    assert payload['velocity']['resolved_weathering_velocity_m_s'] == pytest.approx(
        800.0
    )
    assert payload['observations']['n_valid_observations'] == 3
    assert payload['observations']['n_used_observations'] == 2
    assert payload['status_counts']['node_solution_status']['solved'] == 2
    assert payload['status_counts']['node_weathering_status']['zero_thickness'] == 1
    assert payload['status_counts']['trace_static_status']['ok'] == 3
    assert payload['status_counts']['node_datum_status']['ok'] == 2
    assert payload['first_break_fit']['residual_rms_ms'] == pytest.approx(1.0)
    assert len(payload['artifacts']) == 9
    json.dumps(payload, allow_nan=False)
    assert not _contains_absolute_path(payload)


def test_refraction_static_qc_contains_v1_mode() -> None:
    payload = artifact_module.build_refraction_static_qc_payload(
        result=_result_with_weathering_velocity(812.5),
        req=_estimated_v1_request(),
        resolved_first_layer=_resolved_estimated_v1(),
    )

    assert payload['velocity']['v1_mode'] == 'estimate_direct_arrival'
    assert payload['velocity']['v1_status'] == 'estimated'
    assert payload['velocity']['resolved_weathering_velocity_m_s'] == pytest.approx(
        812.5
    )


def test_write_refraction_static_artifacts_csvs(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    trace_rows = _read_csv(paths.refraction_statics_csv)
    assert len(trace_rows) == 4
    assert [int(row['sorted_trace_index']) for row in trace_rows] == [0, 1, 2, 3]
    assert trace_rows[0]['trace_static_status'] == 'ok'
    assert 'source_half_intercept_time_ms' in trace_rows[0]
    assert float(trace_rows[0]['source_half_intercept_time_ms']) == pytest.approx(10.0)
    assert float(trace_rows[0]['estimated_first_break_time_ms']) == pytest.approx(50.0)
    assert trace_rows[2]['refraction_trace_shift_ms'] == ''

    model_rows = _read_csv(paths.near_surface_model_csv)
    assert len(model_rows) == 3
    assert model_rows[0]['weathering_thickness_m'] == '10.0'
    assert model_rows[0]['solution_status'] == 'solved'
    assert model_rows[1]['weathering_status'] == 'zero_thickness'
    assert float(model_rows[0]['half_intercept_time_ms']) == pytest.approx(10.0)

    residual_rows = _read_csv(paths.first_break_residuals_csv)
    assert len(residual_rows) == 3
    assert float(residual_rows[0]['observed_pick_time_ms']) == pytest.approx(50.0)
    assert float(residual_rows[1]['residual_ms']) == pytest.approx(-2.0)

    component_rows = _read_csv(paths.refraction_static_components_csv)
    assert len(component_rows) == 4
    assert {row['kind'] for row in component_rows} == {'source', 'receiver'}
    assert float(component_rows[0]['half_intercept_time_ms']) == pytest.approx(10.0)
    assert 'refraction_shift_ms' in component_rows[0]


def test_refraction_static_artifacts_manifest_and_download_visibility(
    tmp_path: Path,
) -> None:
    write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )
    manifest = json.loads(
        (tmp_path / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    assert {item['name'] for item in manifest['artifacts']} == {
        REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        REFRACTION_STATIC_QC_JSON_NAME,
        REFRACTION_STATICS_CSV_NAME,
        NEAR_SURFACE_MODEL_CSV_NAME,
        FIRST_BREAK_RESIDUALS_CSV_NAME,
        REFRACTION_STATIC_COMPONENTS_CSV_NAME,
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    }

    state = app.state.sv
    with state.lock:
        state.jobs.clear()
        state.jobs.create_static_job(
            'refraction-artifacts-job',
            file_id='raw-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction',
            artifacts_dir=str(tmp_path),
        )
    try:
        with TestClient(app) as client:
            files = client.get('/statics/job/refraction-artifacts-job/files')
            assert files.status_code == 200
            assert {item['name'] for item in files.json()['files']} == EXPECTED_FILENAMES

            download = client.get(
                '/statics/job/refraction-artifacts-job/download',
                params={'name': REFRACTION_STATIC_QC_JSON_NAME},
            )
            assert download.status_code == 200
            assert download.json()['artifact_version'] == '1.0'
    finally:
        with state.lock:
            state.jobs.clear()


def test_write_refraction_static_artifacts_rejects_missing_job_dir(
    tmp_path: Path,
) -> None:
    with pytest.raises(RefractionStaticArtifactError, match='missing job directory'):
        write_refraction_static_artifacts(
            result=_result(),
            req=_request(),
            job_dir=tmp_path / 'missing',
        )


def test_write_refraction_static_artifacts_rejects_non_writable_job_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifact_module.os, 'access', lambda _path, _mode: False)

    with pytest.raises(RefractionStaticArtifactError, match='not writable'):
        write_refraction_static_artifacts(
            result=_result(),
            req=_request(),
            job_dir=tmp_path,
        )


def test_write_refraction_static_artifacts_validation_failures(
    tmp_path: Path,
) -> None:
    cases = [
        (
            replace(_result(), refraction_trace_shift_s_sorted=np.zeros(3)),
            'trace-order array length mismatch',
        ),
        (
            replace(_result(), node_x_m=np.zeros(2)),
            'node array length mismatch',
        ),
        (
            replace(_result(), source_x_m=np.zeros(1)),
            'source endpoint array length mismatch',
        ),
        (
            replace(_result(), receiver_x_m=np.zeros(1)),
            'receiver endpoint array length mismatch',
        ),
        (
            replace(_result(), residual_time_s=np.zeros(2)),
            'residual array length mismatch',
        ),
        (
            replace(_result(), bedrock_velocity_m_s=float('nan')),
            'non-finite required scalar bedrock_velocity_m_s',
        ),
    ]

    for index, (result, message) in enumerate(cases):
        job_dir = tmp_path / f'case-{index}'
        job_dir.mkdir()
        with pytest.raises(RefractionStaticArtifactError, match=message):
            write_refraction_static_artifacts(
                result=result,
                req=_request(),
                job_dir=job_dir,
            )


def test_write_refraction_static_artifacts_rejects_unknown_status(
    tmp_path: Path,
) -> None:
    result = replace(
        _result(),
        node_solution_status=np.asarray(
            ['solved', 'definitely_unknown_status', 'inactive'],
            dtype='<U32',
        ),
    )

    with pytest.raises(
        RefractionStaticArtifactError,
        match='unknown status array values in node_solution_status.*definitely_unknown_status',
    ):
        write_refraction_static_artifacts(
            result=result,
            req=_request(),
            job_dir=tmp_path,
        )


def test_write_refraction_static_artifacts_detects_missing_artifact_after_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifact_module,
        'write_refraction_static_components_csv',
        lambda **_kwargs: None,
    )

    with pytest.raises(
        RefractionStaticArtifactError,
        match='artifact file missing after write: refraction_static_components.csv',
    ):
        write_refraction_static_artifacts(
            result=_result(),
            req=_request(),
            job_dir=tmp_path,
        )


def test_write_refraction_static_solution_rejects_object_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifact_module,
        'build_refraction_static_solution_arrays',
        lambda **_kwargs: {'bad_object': np.asarray([object()], dtype=object)},
    )

    with pytest.raises(
        RefractionStaticArtifactError,
        match='object array is not allowed for bad_object',
    ):
        write_refraction_static_solution_npz(
            result=_result(),
            req=_request(),
            path=tmp_path / REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, str):
        return value.startswith('/')
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    return False
