from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import RefractionStaticApplyRequest
import app.services.refraction_static_artifacts as artifact_module
from app.services.refraction_static_artifacts import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
    FIRST_BREAK_FIT_QC_RESIDUAL_SIGN,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_STATICS_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_HISTORY_JSON_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    REFRACTION_V3_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_VSUB_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_V1_ESTIMATES_CSV_NAME,
    REFRACTION_V1_QC_JSON_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    RefractionStaticArtifactError,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    TIME_TERM_SPREADSHEET_FORMAT_NAME,
    TIME_TERM_SPREADSHEET_FORMAT_VERSION,
    TIME_TERM_SPREADSHEET_SCHEMA_VERSION,
    write_refraction_static_solution_npz,
    write_refraction_static_artifacts,
)
from app.services.refraction_static_source_depth import (
    REFRACTION_SOURCE_DEPTH_QC_JSON_NAME,
    REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
)
from app.services.refraction_static_types import RefractionLayerSolveResult
from app.services.refraction_static_uphole import (
    REFRACTION_UPHOLE_QC_JSON_NAME,
    REFRACTION_UPHOLE_SOURCES_CSV_NAME,
)
from app.tests._refraction_static_artifact_helpers import (
    _estimated_v1_request,
    _request,
    _resolved_estimated_v1,
    _result,
    _result_with_weathering_velocity,
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
    REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_HISTORY_JSON_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
}

CELL_VELOCITY_FILENAMES = {
    REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
}

V3_CELL_VELOCITY_FILENAMES = {
    REFRACTION_V3_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME,
}

VSUB_CELL_VELOCITY_FILENAMES = {
    REFRACTION_VSUB_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME,
}

UPSTREAM_V1_ARTIFACT_NAMES = (
    REFRACTION_V1_QC_JSON_NAME,
    REFRACTION_V1_ESTIMATES_CSV_NAME,
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
        REFRACTION_STATIC_HISTORY_JSON_NAME,
        REFRACTION_STATICS_CSV_NAME,
        NEAR_SURFACE_MODEL_CSV_NAME,
        FIRST_BREAK_RESIDUALS_CSV_NAME,
        REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
        REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
        REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
        REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
        REFRACTION_STATIC_COMPONENTS_CSV_NAME,
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
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


def test_time_term_spreadsheet_columns_are_stable(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
        source_job_id='refraction-job-505',
    )

    with paths.refraction_time_term_spreadsheet_csv.open(
        encoding='utf-8',
        newline='',
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert paths.refraction_time_term_spreadsheet_csv.name == (
        REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME
    )
    assert tuple(reader.fieldnames or ()) == (
        'schema_version',
        'format_name',
        'format_version',
        'source_job_id',
        'endpoint_kind',
        'endpoint_key',
        'endpoint_id',
        'station_id',
        'node_id',
        'x_m',
        'y_m',
        'elevation_m',
        'surface_elevation_m',
        't1_ms',
        't2_ms',
        't3_ms',
        'v1_m_s',
        'v2_m_s',
        'v3_m_s',
        'vsub_m_s',
        'sh1_m',
        'sh2_m',
        'sh3_m',
        'layer1_base_elevation_m',
        'layer2_base_elevation_m',
        'final_refractor_elevation_m',
        'weathering_correction_ms',
        'elevation_correction_ms',
        'source_depth_correction_ms',
        'uphole_correction_ms',
        'manual_static_ms',
        'field_correction_ms',
        'total_applied_shift_ms',
        'pick_count',
        'used_pick_count',
        'pick_count_by_layer',
        'used_pick_count_by_layer',
        'residual_rms_ms',
        'residual_mad_ms',
        'residual_rms_by_layer_ms',
        'residual_mad_by_layer_ms',
        'solution_status',
        'weathering_status',
        'datum_status',
        'source_depth_status',
        'uphole_status',
        'manual_static_status',
        'field_static_status',
        'static_status',
        'sign_convention',
    )
    assert rows[0]['schema_version'] == str(TIME_TERM_SPREADSHEET_SCHEMA_VERSION)
    assert rows[0]['format_name'] == TIME_TERM_SPREADSHEET_FORMAT_NAME
    assert rows[0]['format_version'] == str(TIME_TERM_SPREADSHEET_FORMAT_VERSION)
    assert rows[0]['source_job_id'] == 'refraction-job-505'


def test_time_term_spreadsheet_contains_one_row_per_endpoint(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_time_term_spreadsheet_csv)

    assert [row['endpoint_kind'] for row in rows] == [
        'source',
        'source',
        'receiver',
        'receiver',
    ]
    assert [row['endpoint_key'] for row in rows] == ['s0', 's1', 'r0', 'r1']
    assert rows[0]['station_id'] == '100'
    assert rows[2]['station_id'] == '200'
    assert rows[0]['elevation_m'] == '100.000'
    assert rows[0]['t1_ms'] == '10.000000'
    assert rows[0]['t2_ms'] == ''
    assert rows[0]['t3_ms'] == ''
    assert rows[0]['sh1_m'] == '10.000'
    assert rows[0]['sh2_m'] == ''
    assert rows[0]['sh3_m'] == ''
    assert rows[0]['sign_convention'] == 'corrected(t) = raw(t - shift_s)'


def test_time_term_spreadsheet_units_are_explicit(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    _rows, fieldnames = _read_csv_with_fieldnames(
        paths.refraction_time_term_spreadsheet_csv
    )

    unit_columns = {
        name
        for name in fieldnames
        if name
        not in {
            'schema_version',
            'format_name',
            'format_version',
            'source_job_id',
            'endpoint_kind',
            'endpoint_key',
            'endpoint_id',
            'station_id',
            'node_id',
            'pick_count',
            'used_pick_count',
            'pick_count_by_layer',
            'used_pick_count_by_layer',
            'solution_status',
            'weathering_status',
            'datum_status',
            'source_depth_status',
            'uphole_status',
            'manual_static_status',
            'field_static_status',
            'static_status',
            'sign_convention',
        }
    }
    assert unit_columns
    assert all(
        name.endswith(('_ms', '_m', '_m_s')) or name in {'x_m', 'y_m'}
        for name in unit_columns
    )


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
    assert len(payload['artifacts']) == len(
        EXPECTED_FILENAMES - {REFRACTION_STATIC_ARTIFACTS_JSON_NAME}
    )
    artifact_names = {item['name'] for item in payload['artifacts']}
    assert REFRACTION_V1_QC_JSON_NAME not in artifact_names
    assert REFRACTION_V1_ESTIMATES_CSV_NAME not in artifact_names
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


def test_source_depth_double_count_guard_qc_warning(tmp_path: Path) -> None:
    request_payload = _request().model_dump(mode='json')
    request_payload['geometry']['source_depth_byte'] = 115
    request_payload['field_corrections'] = {
        'source_depth': {'mode': 'weathering_velocity_time'}
    }
    req = RefractionStaticApplyRequest.model_validate(request_payload)
    result = replace(
        _result(),
        source_depth_m=np.asarray([4.0, 8.0], dtype=np.float64),
        source_depth_shift_s=np.asarray([0.005, 0.010], dtype=np.float64),
        source_depth_status=np.asarray(['ok', 'ok'], dtype='<U48'),
        source_depth_field_correction_qc={
            'source_depth_mode': 'weathering_velocity_time',
            'component_name': 'source_depth_shift_s',
            'source_depth_shift_formula': (
                'source_depth_shift_s = +source_depth_m / V1_m_s'
            ),
            'sign_convention': 'corrected(t) = raw(t - shift_s)',
            'v1_m_s': 800.0,
            'source_depth_double_count_guard': (
                'warning_existing_datum_uses_source_depth'
            ),
            'warnings': ['source depth double-count warning'],
        },
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    assert qc['source_depth_double_count_guard'] == (
        'warning_existing_datum_uses_source_depth'
    )
    assert qc['field_corrections']['source_depth']['component_name'] == (
        'source_depth_shift_s'
    )
    assert qc['warnings'] == ['source depth double-count warning']
    source_rows = _read_csv(paths.source_static_table_csv)
    assert source_rows[0]['source_depth_m'] == '4.0'
    assert float(source_rows[1]['source_depth_shift_ms']) == pytest.approx(10.0)
    component_rows = _read_csv(paths.refraction_static_components_csv)
    assert component_rows[0]['source_depth_shift_ms'] == '5.0'
    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        np.testing.assert_allclose(data['source_depth_shift_s'], [0.005, 0.010])
        assert data['source_depth_status'].tolist() == ['ok', 'ok']
    with np.load(paths.solution_npz, allow_pickle=False) as data:
        np.testing.assert_allclose(data['source_depth_m'], [4.0, 8.0])


def test_uphole_field_correction_qc_and_static_tables(tmp_path: Path) -> None:
    request_payload = _request().model_dump(mode='json')
    request_payload['field_corrections'] = {
        'uphole': {
            'mode': 'header_time',
            'uphole_time_byte': 95,
            'uphole_time_unit': 's',
        }
    }
    req = RefractionStaticApplyRequest.model_validate(request_payload)
    result = replace(
        _result(),
        source_uphole_time_s=np.asarray([0.010, 0.020], dtype=np.float64),
        source_uphole_shift_s=np.asarray([-0.010, -0.020], dtype=np.float64),
        source_uphole_status=np.asarray(['ok', 'ok'], dtype='<U48'),
        source_uphole_field_correction_qc={
            'uphole_mode': 'header_time',
            'component_name': 'uphole_shift_s',
            'uphole_shift_formula': 'uphole_shift_s = -uphole_time_s',
            'sign_convention': 'corrected(t) = raw(t - shift_s)',
            'positive_time_means_delay': True,
            'uphole_time_byte': 95,
            'uphole_time_unit': 's',
        },
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    assert qc['field_corrections']['uphole']['component_name'] == 'uphole_shift_s'
    assert qc['field_corrections']['uphole']['sign_convention'] == (
        'corrected(t) = raw(t - shift_s)'
    )
    source_rows = _read_csv(paths.source_static_table_csv)
    assert float(source_rows[0]['uphole_time_ms']) == pytest.approx(10.0)
    assert float(source_rows[1]['uphole_shift_ms']) == pytest.approx(-20.0)
    component_rows = _read_csv(paths.refraction_static_components_csv)
    assert component_rows[0]['uphole_shift_ms'] == '-10.0'
    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        np.testing.assert_allclose(data['source_uphole_time_s'], [0.010, 0.020])
        np.testing.assert_allclose(data['source_uphole_shift_s'], [-0.010, -0.020])
        assert data['source_uphole_status'].tolist() == ['ok', 'ok']
    with np.load(paths.solution_npz, allow_pickle=False) as data:
        np.testing.assert_allclose(data['source_uphole_shift_s'], [-0.010, -0.020])


def test_refraction_manifest_registers_field_correction_artifacts(
    tmp_path: Path,
) -> None:
    request_payload = _request().model_dump(mode='json')
    request_payload['field_corrections'] = {
        'source_depth': {
            'mode': 'weathering_velocity_time',
            'source_depth_byte': 115,
        },
        'uphole': {
            'mode': 'header_time',
            'uphole_time_byte': 95,
            'uphole_time_unit': 's',
        },
    }
    req = RefractionStaticApplyRequest.model_validate(request_payload)
    result = replace(
        _result(),
        source_depth_m=np.asarray([4.0, 8.0], dtype=np.float64),
        source_depth_shift_s=np.asarray([0.005, 0.010], dtype=np.float64),
        source_depth_status=np.asarray(['ok', 'ok'], dtype='<U48'),
        source_depth_field_correction_qc={
            'source_depth_mode': 'weathering_velocity_time',
            'component_name': 'source_depth_shift_s',
            'sign_convention': 'corrected(t) = raw(t - shift_s)',
            'source_depth_double_count_guard': 'checked',
        },
        source_uphole_time_s=np.asarray([0.010, 0.020], dtype=np.float64),
        source_uphole_shift_s=np.asarray([-0.010, -0.020], dtype=np.float64),
        source_uphole_status=np.asarray(['ok', 'ok'], dtype='<U48'),
        source_uphole_field_correction_qc={
            'uphole_mode': 'header_time',
            'component_name': 'uphole_shift_s',
            'sign_convention': 'corrected(t) = raw(t - shift_s)',
        },
    )
    upstream_names = (
        REFRACTION_SOURCE_DEPTH_QC_JSON_NAME,
        REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
        REFRACTION_UPHOLE_QC_JSON_NAME,
        REFRACTION_UPHOLE_SOURCES_CSV_NAME,
    )
    for name in upstream_names:
        (tmp_path / name).write_text('{}' if name.endswith('.json') else '', encoding='utf-8')

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
        upstream_artifact_names=upstream_names,
    )

    manifest = json.loads(paths.manifest_json.read_text(encoding='utf-8'))
    artifacts = {item['name']: item for item in manifest['artifacts']}
    for name in upstream_names:
        assert artifacts[name]['origin'] == 'upstream'
    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    qc_artifacts = {item['name'] for item in qc['artifacts']}
    assert set(upstream_names).issubset(qc_artifacts)
    assert qc['field_corrections']['source_depth']['sign_convention'] == (
        'corrected(t) = raw(t - shift_s)'
    )
    assert qc['field_corrections']['uphole']['sign_convention'] == (
        'corrected(t) = raw(t - shift_s)'
    )
    json.dumps(qc, allow_nan=False)


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

    first_break_rows = _read_csv(paths.refraction_first_break_time_export_csv)
    assert len(first_break_rows) == 3
    assert first_break_rows[0]['source_endpoint_key'] == 's0'
    assert first_break_rows[0]['receiver_endpoint_key'] == 'r0'
    assert float(first_break_rows[0]['observed_pick_time_ms']) == pytest.approx(
        50.0
    )
    assert float(first_break_rows[0]['modeled_pick_time_ms']) == pytest.approx(
        49.0
    )
    assert float(first_break_rows[0]['residual_ms']) == pytest.approx(1.0)

    component_rows = _read_csv(paths.refraction_static_components_csv)
    assert len(component_rows) == 4
    assert {row['kind'] for row in component_rows} == {'source', 'receiver'}
    assert float(component_rows[0]['half_intercept_time_ms']) == pytest.approx(10.0)
    assert 'refraction_shift_ms' in component_rows[0]


def test_first_break_time_export_contains_observed_modeled_residual(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    rows, fieldnames = _read_csv_with_fieldnames(
        paths.refraction_first_break_time_export_csv
    )

    assert tuple(fieldnames) == (
        'format_name',
        'format_version',
        'source_job_id',
        'observation_index',
        'sorted_trace_index',
        'source_endpoint_key',
        'receiver_endpoint_key',
        'source_id',
        'receiver_id',
        'offset_m',
        'layer_kind',
        'observed_pick_time_ms',
        'modeled_pick_time_ms',
        'residual_ms',
        'used_in_solve',
        'reject_reason',
        'sign_convention',
    )
    assert rows[0]['format_name'] == 'first_break_time'
    assert rows[0]['format_version'] == '1'
    assert rows[0]['source_job_id'] == ''
    assert rows[0]['observation_index'] == '0'
    assert rows[0]['sorted_trace_index'] == '0'
    assert rows[0]['source_id'] == '100'
    assert rows[0]['receiver_id'] == '200'
    assert rows[0]['layer_kind'] == 'v2_t1'
    assert rows[0]['used_in_solve'] == 'true'
    assert float(rows[0]['observed_pick_time_ms']) == pytest.approx(50.0)
    assert float(rows[0]['modeled_pick_time_ms']) == pytest.approx(49.0)
    assert float(rows[0]['residual_ms']) == pytest.approx(1.0)
    assert rows[0]['sign_convention'] == (
        artifact_module.FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION
    )


def test_first_break_time_export_marks_rejected_observations(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_first_break_time_export_csv)

    assert rows[1]['used_in_solve'] == 'false'
    assert rows[1]['reject_reason'] == 'robust_outlier'


def test_first_break_time_export_preserves_unassigned_layer_context(
    tmp_path: Path,
) -> None:
    result = replace(
        _result(),
        row_layer_kind=np.asarray(['v2_t1', '', 'v2_t1'], dtype='<U16'),
        row_layer_index=np.asarray([1, 0, 1], dtype=np.int64),
        rejected_by_robust_mask=np.asarray([False, False, False], dtype=bool),
        row_rejection_reason=np.asarray(
            ['ok', 'outside_layer_gate', 'ok'],
            dtype='<U32',
        ),
        row_velocity_m_s=np.asarray([2500.0, np.nan, 2500.0], dtype=np.float64),
    )
    paths = write_refraction_static_artifacts(
        result=result,
        req=_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_first_break_time_export_csv)

    assert rows[1]['layer_kind'] == ''
    assert rows[1]['used_in_solve'] == 'false'
    assert rows[1]['reject_reason'] == 'outside_layer_gate'


def test_first_break_time_export_residual_matches_solution_npz(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_first_break_time_export_csv)
    with np.load(paths.solution_npz, allow_pickle=False) as data:
        residual_ms = data['residual_time_s'] * 1000.0

    np.testing.assert_allclose(
        np.asarray([float(row['residual_ms']) for row in rows]),
        residual_ms,
    )


def test_first_break_fit_qc_csv_npz_json_are_consistent(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_first_break_fit_qc_csv)
    payload = json.loads(
        paths.refraction_first_break_fit_qc_json.read_text(encoding='utf-8')
    )
    with np.load(paths.refraction_first_break_fit_qc_npz, allow_pickle=False) as data:
        observed = data['observed_first_break_time_s']
        modeled = data['modeled_first_break_time_s']
        residual = data['residual_time_s']
        used = data['used_for_inversion']

    assert payload['residual_sign'] == FIRST_BREAK_FIT_QC_RESIDUAL_SIGN
    assert payload['residual_definition'] == (
        'residual_time_s = observed_first_break_time_s - '
        'modeled_first_break_time_s'
    )
    assert payload['row_count'] == len(rows) == 3
    assert payload['used_count'] == 2
    assert payload['rejected_count'] == 1
    assert rows[0]['trace_index_sorted'] == '0'
    assert rows[0]['source_endpoint_key'] == 's0'
    assert rows[0]['receiver_endpoint_key'] == 'r0'
    assert rows[0]['source_node_id'] == '0'
    assert rows[0]['receiver_node_id'] == '1'
    assert rows[0]['status'] == 'ok'
    assert rows[1]['status'] == 'rejected'
    assert rows[1]['rejection_reason'] == 'robust_outlier'
    assert rows[1]['used_for_inversion'] == 'false'
    assert rows[0]['sign_convention'] == 'corrected(t) = raw(t - shift_s)'

    np.testing.assert_allclose(observed - modeled, residual)
    np.testing.assert_allclose(
        np.asarray([float(row['observed_first_break_time_s']) for row in rows]),
        observed,
    )
    np.testing.assert_allclose(
        np.asarray([float(row['modeled_first_break_time_s']) for row in rows]),
        modeled,
    )
    np.testing.assert_allclose(
        np.asarray([float(row['residual_time_s']) for row in rows]),
        residual,
    )
    assert payload['residual_summary']['used_rms_s'] == pytest.approx(
        float(np.sqrt(np.mean(residual[used] * residual[used])))
    )


def test_first_break_fit_qc_reports_cell_and_multilayer_context(
    tmp_path: Path,
) -> None:
    result = replace(
        _solve_cell_result(),
        row_layer_kind=np.asarray(['v2_t1', 'v3_t2', 'vsub_t3'], dtype='<U16'),
        row_rejection_reason=np.asarray(
            ['ok', 'outside_layer_offset_gate', 'ok'],
            dtype='<U32',
        ),
        used_row_mask=np.asarray([True, False, True], dtype=bool),
        rejected_by_robust_mask=np.asarray([False, False, False], dtype=bool),
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_first_break_fit_qc_csv)
    with np.load(paths.refraction_first_break_fit_qc_npz, allow_pickle=False) as data:
        assert data['layer_kind'].tolist() == ['v2_t1', 'v3_t2', 'vsub_t3']
        np.testing.assert_allclose(data['cell_id'], np.asarray([0.0, np.nan, np.nan]))
        np.testing.assert_allclose(data['cell_ix'], np.asarray([0.0, np.nan, np.nan]))
        np.testing.assert_allclose(data['cell_iy'], np.asarray([0.0, np.nan, np.nan]))

    assert [row['layer_kind'] for row in rows] == ['v2_t1', 'v3_t2', 'vsub_t3']
    assert [row['cell_id'] for row in rows] == ['0', '', '']
    assert [row['cell_ix'] for row in rows] == ['0', '', '']
    assert [row['cell_iy'] for row in rows] == ['0', '', '']
    assert rows[1]['status'] == 'rejected'
    assert rows[1]['rejection_reason'] == 'outside_layer_offset_gate'


def test_refraction_static_artifacts_manifest_and_download_visibility(
    tmp_path: Path,
) -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )
    manifest = json.loads(
        (tmp_path / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    assert {item['name'] for item in manifest['artifacts']} == (
        EXPECTED_FILENAMES - {REFRACTION_STATIC_ARTIFACTS_JSON_NAME}
    )
    assert {item['origin'] for item in manifest['artifacts']} == {'final'}

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


def test_solve_cell_writes_refractor_velocity_cells_csv(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    assert paths.refraction_refractor_velocity_cells_csv is not None
    rows = _read_csv(paths.refraction_refractor_velocity_cells_csv)

    assert [int(row['cell_id']) for row in rows] == [0, 1, 2]
    assert rows[0]['active'] == 'true'
    assert rows[0]['velocity_status'] == 'solved'
    assert float(rows[0]['velocity_m_s']) == pytest.approx(2400.0)
    assert float(rows[0]['v2_m_s']) == pytest.approx(2400.0)
    assert float(rows[0]['initial_velocity_m_s']) == pytest.approx(2500.0)
    assert float(rows[0]['initial_v2_m_s']) == pytest.approx(2500.0)
    assert float(rows[0]['velocity_update_from_initial_m_s']) == pytest.approx(
        -100.0
    )
    assert float(rows[0]['v2_update_from_initial_m_s']) == pytest.approx(-100.0)
    assert int(rows[0]['n_used_observations']) == 2
    assert rows[2]['active'] == 'false'
    assert rows[2]['velocity_status'] == 'inactive'
    assert rows[2]['velocity_m_s'] == ''
    assert rows[2]['v2_m_s'] == ''
    assert rows[2]['x_min_m'] == '200.0'


def test_solve_cell_writes_refractor_velocity_grid_npz(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    assert paths.refraction_refractor_velocity_grid_npz is not None
    with np.load(paths.refraction_refractor_velocity_grid_npz, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['cell_id'], [0, 1, 2])
        np.testing.assert_array_equal(data['active_cell_mask'], [True, True, False])
        np.testing.assert_allclose(data['velocity_m_s'][:2], [2400.0, 2600.0])
        np.testing.assert_allclose(data['v2_m_s'][:2], [2400.0, 2600.0])
        np.testing.assert_allclose(
            data['velocity_m_s'],
            data['v2_m_s'],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            data['initial_velocity_m_s'],
            data['initial_v2_m_s'],
        )
        np.testing.assert_allclose(
            data['velocity_update_from_initial_m_s'],
            data['v2_update_from_initial_m_s'],
            equal_nan=True,
        )
        assert np.isnan(data['velocity_m_s'][2])
        assert np.isnan(data['v2_m_s'][2])
        assert data['velocity_status'].tolist() == ['solved', 'solved', 'inactive']
        np.testing.assert_array_equal(
            data['n_observations_per_cell'],
            [2, 1, 0],
        )
        np.testing.assert_array_equal(
            data['n_used_observations_per_cell'],
            [2, 0, 0],
        )


def test_solve_cell_writes_low_fold_refractor_velocity_artifacts(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_low_fold_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    assert paths.refraction_refractor_velocity_cells_csv is not None
    rows = _read_csv(paths.refraction_refractor_velocity_cells_csv)
    assert rows[1]['active'] == 'false'
    assert rows[1]['velocity_status'] == 'low_fold'
    assert rows[1]['v2_m_s'] == ''
    assert int(rows[1]['n_observations']) == 1
    assert int(rows[1]['n_used_observations']) == 0
    assert int(rows[1]['n_rejected_observations']) == 1

    assert paths.refraction_refractor_velocity_qc_json is not None
    cell_qc = json.loads(
        paths.refraction_refractor_velocity_qc_json.read_text(encoding='utf-8')
    )
    assert cell_qc['min_observations_per_cell'] == 2
    assert cell_qc['n_low_fold_cells'] == 1
    assert cell_qc['n_observations_rejected_by_low_fold_cell'] == 1
    assert (
        cell_qc['low_fold_cell_rejection_reason']
        == 'below_min_observations_per_cell'
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    assert source_rows[1]['v2_status'] == 'low_fold_v2_cell'
    assert source_rows[1]['static_status'] == 'low_fold_v2_cell'
    assert receiver_rows[0]['v2_status'] == 'low_fold_v2_cell'
    assert receiver_rows[0]['static_status'] == 'low_fold_v2_cell'


def test_solve_cell_manifest_registers_cell_velocity_artifacts(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    assert {path.name for path in tmp_path.iterdir()} == (
        EXPECTED_FILENAMES | CELL_VELOCITY_FILENAMES
    )
    manifest = json.loads(paths.manifest_json.read_text(encoding='utf-8'))
    artifact_names = {item['name'] for item in manifest['artifacts']}
    assert CELL_VELOCITY_FILENAMES.issubset(artifact_names)

    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    assert qc['velocity']['cell_velocity_qc_artifact'] == (
        REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME
    )
    assert qc['refractor_velocity_cells']['grid_npz_artifact'] == (
        REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME
    )
    assert qc['refractor_velocity_cells']['solver_history_csv_artifact'] == (
        REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME
    )


def test_v3_cell_artifacts_use_layer_specific_names(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_multilayer_shaped_solve_cell_result(),
        req=_v3_solve_cell_request(),
        job_dir=tmp_path,
    )

    assert {path.name for path in tmp_path.iterdir()} == (
        EXPECTED_FILENAMES | V3_CELL_VELOCITY_FILENAMES
    )
    assert paths.refraction_refractor_velocity_cells_csv is not None
    assert (
        paths.refraction_refractor_velocity_cells_csv.name
        == REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME
    )
    assert not CELL_VELOCITY_FILENAMES.intersection(
        {path.name for path in tmp_path.iterdir()}
    )

    manifest = json.loads(paths.manifest_json.read_text(encoding='utf-8'))
    artifact_names = {item['name'] for item in manifest['artifacts']}
    assert V3_CELL_VELOCITY_FILENAMES <= artifact_names
    assert not CELL_VELOCITY_FILENAMES.intersection(artifact_names)

    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    assert qc['velocity']['cell_velocity_layer_kind'] == 'v3_t2'
    assert qc['velocity']['cell_velocity_component'] == 'v3'
    assert (
        qc['velocity']['cell_velocity_qc_artifact']
        == REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME
    )
    assert qc['refractor_velocity_cells']['cells_csv_artifact'] == (
        REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME
    )

    rows = _read_csv(paths.refraction_refractor_velocity_cells_csv)
    assert rows
    assert {row['cell_velocity_layer_kind'] for row in rows} == {'v3_t2'}
    assert {row['cell_velocity_component'] for row in rows} == {'v3'}


def test_multiple_cell_velocity_layers_write_all_layer_artifacts(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_multilayer_result_with_v3_and_vsub_cell_layers(),
        req=_v3_vsub_solve_cell_request(),
        job_dir=tmp_path,
    )

    filenames = {path.name for path in tmp_path.iterdir()}
    assert filenames == (
        EXPECTED_FILENAMES | V3_CELL_VELOCITY_FILENAMES | VSUB_CELL_VELOCITY_FILENAMES
    )
    assert paths.refraction_refractor_velocity_cells_csv is not None
    assert (
        paths.refraction_refractor_velocity_cells_csv.name
        == REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME
    )

    manifest = json.loads(paths.manifest_json.read_text(encoding='utf-8'))
    artifact_names = {item['name'] for item in manifest['artifacts']}
    assert V3_CELL_VELOCITY_FILENAMES <= artifact_names
    assert VSUB_CELL_VELOCITY_FILENAMES <= artifact_names

    v3_rows = _read_csv(tmp_path / REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME)
    vsub_rows = _read_csv(
        tmp_path / REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME
    )
    assert {row['cell_velocity_layer_kind'] for row in v3_rows} == {'v3_t2'}
    assert {row['cell_velocity_component'] for row in v3_rows} == {'v3'}
    assert {row['cell_velocity_layer_kind'] for row in vsub_rows} == {'vsub_t3'}
    assert {row['cell_velocity_component'] for row in vsub_rows} == {'vsub'}

    with np.load(
        tmp_path / REFRACTION_V3_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
        allow_pickle=False,
    ) as v3_grid:
        np.testing.assert_allclose(
            v3_grid['velocity_m_s'],
            np.asarray([3300.0, 3700.0, np.nan], dtype=np.float64),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            v3_grid['velocity_m_s'],
            v3_grid['v2_m_s'],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            v3_grid['initial_velocity_m_s'],
            v3_grid['initial_v2_m_s'],
        )
        np.testing.assert_allclose(
            v3_grid['velocity_update_from_initial_m_s'],
            v3_grid['v2_update_from_initial_m_s'],
            equal_nan=True,
        )
        assert set(v3_grid['cell_velocity_layer_kind'].astype(str).tolist()) == {
            'v3_t2'
        }
        assert set(v3_grid['cell_velocity_component'].astype(str).tolist()) == {'v3'}

    with np.load(
        tmp_path / REFRACTION_VSUB_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
        allow_pickle=False,
    ) as vsub_grid:
        np.testing.assert_allclose(
            vsub_grid['velocity_m_s'],
            np.asarray([np.nan, 4800.0, 5200.0], dtype=np.float64),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            vsub_grid['velocity_m_s'],
            vsub_grid['v2_m_s'],
            equal_nan=True,
        )
        assert set(vsub_grid['cell_velocity_layer_kind'].astype(str).tolist()) == {
            'vsub_t3'
        }
        assert set(vsub_grid['cell_velocity_component'].astype(str).tolist()) == {
            'vsub'
        }

    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    assert qc['velocity']['cell_velocity_layer_kinds'] == ['v3_t2', 'vsub_t3']
    assert qc['velocity']['cell_velocity_qc_artifacts_by_layer'] == {
        'v3_t2': REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME,
        'vsub_t3': REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME,
    }
    assert set(qc['refractor_velocity_cells_by_layer']) == {'v3_t2', 'vsub_t3'}


def test_solve_global_does_not_write_cell_velocity_artifacts(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    assert paths.refraction_refractor_velocity_cells_csv is None
    assert not CELL_VELOCITY_FILENAMES.intersection(
        {path.name for path in tmp_path.iterdir()}
    )
    manifest = json.loads(paths.manifest_json.read_text(encoding='utf-8'))
    artifact_names = {item['name'] for item in manifest['artifacts']}
    assert not CELL_VELOCITY_FILENAMES.intersection(artifact_names)


def test_source_receiver_tables_include_v2_status_in_cell_mode(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_solve_cell_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)

    assert source_rows[0]['source_v2_cell_id'] == '0'
    assert source_rows[0]['v2_status'] == 'ok'
    assert receiver_rows[1]['receiver_v2_cell_id'] == '2'
    assert receiver_rows[1]['v2_status'] == 'inactive_v2_cell'


def test_download_cell_velocity_artifacts(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    write_refraction_static_artifacts(
        result=_solve_cell_result(),
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    state = app.state.sv
    with state.lock:
        state.jobs.clear()
        state.jobs.create_static_job(
            'refraction-cell-artifacts-job',
            file_id='raw-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction',
            artifacts_dir=str(tmp_path),
        )
    try:
        with TestClient(app) as client:
            files = client.get('/statics/job/refraction-cell-artifacts-job/files')
            assert files.status_code == 200
            assert CELL_VELOCITY_FILENAMES.issubset(
                {item['name'] for item in files.json()['files']}
            )

            download = client.get(
                '/statics/job/refraction-cell-artifacts-job/download',
                params={'name': REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME},
            )
            assert download.status_code == 200
            assert download.json()['bedrock_velocity_mode'] == 'solve_cell'
    finally:
        with state.lock:
            state.jobs.clear()


def test_refraction_static_manifest_includes_v1_artifacts_after_v1_estimation(
    tmp_path: Path,
) -> None:
    _write_upstream_v1_artifacts(tmp_path)

    write_refraction_static_artifacts(
        result=_result_with_weathering_velocity(812.5),
        req=_estimated_v1_request(),
        job_dir=tmp_path,
        resolved_first_layer=_resolved_estimated_v1(),
        upstream_artifact_names=UPSTREAM_V1_ARTIFACT_NAMES,
    )

    manifest = json.loads(
        (tmp_path / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    artifacts = {item['name']: item for item in manifest['artifacts']}
    assert artifacts[REFRACTION_V1_QC_JSON_NAME]['required'] is True
    assert artifacts[REFRACTION_V1_QC_JSON_NAME]['origin'] == 'upstream'
    assert artifacts[REFRACTION_V1_ESTIMATES_CSV_NAME]['required'] is True
    assert artifacts[REFRACTION_V1_ESTIMATES_CSV_NAME]['origin'] == 'upstream'
    assert artifacts[REFRACTION_STATIC_SOLUTION_NPZ_NAME]['origin'] == 'final'

    qc = json.loads((tmp_path / REFRACTION_STATIC_QC_JSON_NAME).read_text())
    qc_artifacts = {item['name']: item for item in qc['artifacts']}
    assert REFRACTION_V1_QC_JSON_NAME in qc_artifacts
    assert REFRACTION_V1_ESTIMATES_CSV_NAME in qc_artifacts


def test_refraction_static_artifact_writer_missing_upstream_v1_artifacts_error_is_clear(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RefractionStaticArtifactError,
        match='declared upstream artifact missing: refraction_v1_qc.json',
    ):
        write_refraction_static_artifacts(
            result=_result_with_weathering_velocity(812.5),
            req=_estimated_v1_request(),
            job_dir=tmp_path,
            resolved_first_layer=_resolved_estimated_v1(),
            upstream_artifact_names=UPSTREAM_V1_ARTIFACT_NAMES,
        )


def test_refraction_static_artifact_writer_estimated_v1_omits_unprovided_upstream_artifacts(
    tmp_path: Path,
) -> None:
    write_refraction_static_artifacts(
        result=_result_with_weathering_velocity(812.5),
        req=_estimated_v1_request(),
        job_dir=tmp_path,
        resolved_first_layer=_resolved_estimated_v1(),
    )

    manifest = json.loads(
        (tmp_path / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    artifacts = {item['name']: item for item in manifest['artifacts']}
    assert REFRACTION_V1_QC_JSON_NAME not in artifacts
    assert REFRACTION_V1_ESTIMATES_CSV_NAME not in artifacts
    assert not (tmp_path / REFRACTION_V1_QC_JSON_NAME).exists()
    assert not (tmp_path / REFRACTION_V1_ESTIMATES_CSV_NAME).exists()


def test_refraction_static_artifact_writer_constant_v1_does_not_require_v1_files(
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
    artifacts = {item['name']: item for item in manifest['artifacts']}
    assert REFRACTION_V1_QC_JSON_NAME not in artifacts
    assert REFRACTION_V1_ESTIMATES_CSV_NAME not in artifacts
    assert not (tmp_path / REFRACTION_V1_QC_JSON_NAME).exists()
    assert not (tmp_path / REFRACTION_V1_ESTIMATES_CSV_NAME).exists()

    qc = json.loads((tmp_path / REFRACTION_STATIC_QC_JSON_NAME).read_text())
    qc_artifacts = {item['name']: item for item in qc['artifacts']}
    assert REFRACTION_V1_QC_JSON_NAME not in qc_artifacts
    assert REFRACTION_V1_ESTIMATES_CSV_NAME not in qc_artifacts
    assert qc['velocity']['v1_mode'] == 'constant'


def test_refraction_static_qc_includes_layer_observation_counts_when_present(
    tmp_path: Path,
) -> None:
    layer_qc = {
        'v2_t1': {
            'enabled': True,
            'n_candidate_observations': 3,
            'n_used_observations': 2,
            'min_offset_m': 0.0,
            'max_offset_m': 1000.0,
            'rejection_counts': {
                'ok': 2,
                'outside_layer_offset_gate': 1,
            },
        }
    }
    base = _result()
    result = replace(base, qc={**base.qc, 'layers': layer_qc})

    write_refraction_static_artifacts(
        result=result,
        req=_request(),
        job_dir=tmp_path,
    )

    qc = json.loads((tmp_path / REFRACTION_STATIC_QC_JSON_NAME).read_text())
    assert qc['layers'] == layer_qc


def test_refraction_static_manifest_strict_json(tmp_path: Path) -> None:
    write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    payload = json.loads(
        (tmp_path / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    json.dumps(payload, allow_nan=False)
    assert payload['job_kind'] == 'statics'
    assert payload['statics_kind'] == 'refraction'
    assert {
        'name',
        'kind',
        'required',
        'origin',
    }.issubset(payload['artifacts'][0])


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


def test_solve_cell_artifacts_require_local_v2_arrays(tmp_path: Path) -> None:
    with pytest.raises(
        RefractionStaticArtifactError,
        match='solve_cell result requires node_v2_cell_id',
    ):
        write_refraction_static_solution_npz(
            result=replace(_result(), bedrock_velocity_mode='solve_cell'),
            req=_request(),
            path=tmp_path / REFRACTION_STATIC_SOLUTION_NPZ_NAME,
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


def _solve_cell_request():
    payload = _request().model_dump(mode='json')
    payload['model'].update(
        {
            'bedrock_velocity_mode': 'solve_cell',
            'initial_bedrock_velocity_m_s': 2500.0,
            'min_bedrock_velocity_m_s': 1200.0,
            'max_bedrock_velocity_m_s': 6000.0,
            'refractor_cell': {
                'number_of_cell_x': 3,
                'size_of_cell_x_m': 100.0,
                'x_coordinate_origin_m': 0.0,
                'number_of_cell_y': 1,
                'size_of_cell_y_m': None,
                'assignment_mode': 'midpoint',
                'outside_grid_policy': 'reject',
                'min_observations_per_cell': 1,
                'velocity_smoothing_weight': 1.5,
                'smoothing_reference_distance_m': 200.0,
            },
        }
    )
    return RefractionStaticApplyRequest.model_validate(payload)


def _v3_solve_cell_request():
    payload = _request().model_dump(mode='json')
    payload['model'] = {
        'method': 'multilayer_time_term',
        'first_layer': {
            'mode': 'constant',
            'weathering_velocity_m_s': 800.0,
        },
        'refractor_cell': {
            'number_of_cell_x': 3,
            'size_of_cell_x_m': 100.0,
            'x_coordinate_origin_m': 0.0,
            'number_of_cell_y': 1,
            'size_of_cell_y_m': None,
            'assignment_mode': 'midpoint',
            'outside_grid_policy': 'reject',
            'min_observations_per_cell': 5,
            'velocity_smoothing_weight': 0.0,
            'smoothing_reference_distance_m': None,
        },
        'layers': [
            {
                'kind': 'v2_t1',
                'enabled': True,
                'min_offset_m': 0.0,
                'max_offset_m': 150.0,
                'velocity_mode': 'fixed_global',
                'fixed_velocity_m_s': 2400.0,
                'min_velocity_m_s': 1600.0,
                'max_velocity_m_s': 3200.0,
            },
            {
                'kind': 'v3_t2',
                'enabled': True,
                'min_offset_m': 150.0,
                'max_offset_m': None,
                'velocity_mode': 'solve_cell',
                'initial_velocity_m_s': 3600.0,
                'min_velocity_m_s': 3000.0,
                'max_velocity_m_s': 4500.0,
                'min_observations_per_cell': 1,
                'smoothing_weight': 1.25,
            },
        ],
    }
    return RefractionStaticApplyRequest.model_validate(payload)


def _v3_vsub_solve_cell_request():
    payload = _v3_solve_cell_request().model_dump(mode='json')
    payload['model']['layers'] = [
        {
            'kind': 'v2_t1',
            'enabled': True,
            'min_offset_m': 0.0,
            'max_offset_m': 150.0,
            'velocity_mode': 'fixed_global',
            'fixed_velocity_m_s': 2400.0,
            'min_velocity_m_s': 1600.0,
            'max_velocity_m_s': 3200.0,
        },
        {
            'kind': 'v3_t2',
            'enabled': True,
            'min_offset_m': 150.0,
            'max_offset_m': 300.0,
            'velocity_mode': 'solve_cell',
            'initial_velocity_m_s': 3600.0,
            'min_velocity_m_s': 3000.0,
            'max_velocity_m_s': 4500.0,
            'min_observations_per_cell': 1,
            'smoothing_weight': 1.25,
        },
        {
            'kind': 'vsub_t3',
            'enabled': True,
            'min_offset_m': 300.0,
            'max_offset_m': None,
            'velocity_mode': 'solve_cell',
            'initial_velocity_m_s': 5000.0,
            'min_velocity_m_s': 4200.0,
            'max_velocity_m_s': 6200.0,
            'min_observations_per_cell': 1,
            'smoothing_weight': 0.75,
        },
    ]
    return RefractionStaticApplyRequest.model_validate(payload)


def _solve_cell_result():
    return replace(
        _result(),
        bedrock_velocity_mode='solve_cell',
        bedrock_velocity_m_s=2500.0,
        bedrock_slowness_s_per_m=1.0 / 2500.0,
        active_cell_id=np.asarray([0, 1], dtype=np.int64),
        inactive_cell_id=np.asarray([2], dtype=np.int64),
        cell_bedrock_slowness_s_per_m=np.asarray(
            [1.0 / 2400.0, 1.0 / 2600.0],
            dtype=np.float64,
        ),
        cell_bedrock_velocity_m_s=np.asarray([2400.0, 2600.0], dtype=np.float64),
        cell_velocity_status=np.asarray(['solved', 'solved'], dtype='<U16'),
        row_midpoint_cell_id=np.asarray([0, 1, 0], dtype=np.int64),
        node_v2_cell_id=np.asarray([0, 1, 2], dtype=np.int64),
        node_v2_m_s=np.asarray([2400.0, 2600.0, np.nan], dtype=np.float64),
        node_v2_status=np.asarray(
            ['ok', 'ok', 'inactive_v2_cell'],
            dtype='<U32',
        ),
        source_v2_cell_id=np.asarray([0, 1], dtype=np.int64),
        source_v2_m_s=np.asarray([2400.0, 2600.0], dtype=np.float64),
        source_v2_status=np.asarray(['ok', 'ok'], dtype='<U2'),
        receiver_v2_cell_id=np.asarray([1, 2], dtype=np.int64),
        receiver_v2_m_s=np.asarray([2600.0, np.nan], dtype=np.float64),
        receiver_v2_status=np.asarray(['ok', 'inactive_v2_cell'], dtype='<U32'),
        source_v2_cell_id_sorted=np.asarray([0, 1, 0, 1], dtype=np.int64),
        source_v2_m_s_sorted=np.asarray(
            [2400.0, 2600.0, 2400.0, 2600.0],
            dtype=np.float64,
        ),
        source_v2_status_sorted=np.asarray(['ok', 'ok', 'ok', 'ok'], dtype='<U2'),
        receiver_v2_cell_id_sorted=np.asarray([1, 2, 2, 1], dtype=np.int64),
        receiver_v2_m_s_sorted=np.asarray(
            [2600.0, np.nan, np.nan, 2600.0],
            dtype=np.float64,
        ),
        receiver_v2_status_sorted=np.asarray(
            ['ok', 'inactive_v2_cell', 'inactive_v2_cell', 'ok'],
            dtype='<U32',
        ),
    )


def _multilayer_shaped_solve_cell_result():
    base = _solve_cell_result()
    return replace(
        base,
        row_trace_index_sorted=np.arange(4, dtype=np.int64),
        row_source_node_id=np.asarray([0, 1, 0, 1], dtype=np.int64),
        row_receiver_node_id=np.asarray([1, 2, 2, 1], dtype=np.int64),
        row_distance_m=np.asarray([100.0, 200.0, 300.0, 400.0]),
        observed_pick_time_s=np.asarray([0.050, 0.060, 0.070, 0.080]),
        modeled_pick_time_s=np.asarray([0.049, 0.062, 0.071, 0.079]),
        residual_time_s=np.asarray([0.001, -0.002, -0.001, 0.001]),
        used_row_mask=np.asarray([True, False, True, False], dtype=bool),
        rejected_by_robust_mask=np.asarray([False, True, False, False], dtype=bool),
        row_midpoint_cell_id=np.asarray([0, 1, 0, 1], dtype=np.int64),
        row_layer_kind=np.asarray(['v2_t1', 'v2_t1', 'v2_t1', 'v2_t1'], dtype='<U16'),
        row_layer_index=np.ones(4, dtype=np.int64),
        row_source_endpoint_key=np.asarray(['s0', 's1', 's0', 's1'], dtype='<U16'),
        row_receiver_endpoint_key=np.asarray(['r0', 'r1', 'r1', 'r0'], dtype='<U16'),
        row_rejection_reason=np.asarray(['ok', '', 'ok', 'not_used'], dtype='<U32'),
        row_velocity_m_s=np.full(4, 2500.0, dtype=np.float64),
    )


def _multilayer_result_with_v3_and_vsub_cell_layers():
    base = _multilayer_shaped_solve_cell_result()
    return replace(
        base,
        bedrock_velocity_mode='fixed_global',
        layer_results=(
            _cell_layer_result(
                layer_kind='v3_t2',
                layer_index=2,
                velocity_m_s=np.asarray([3300.0, 3700.0, np.nan], dtype=np.float64),
                used_mask=np.asarray([False, True, True, False], dtype=bool),
                residual_s=np.asarray([np.nan, 0.001, -0.002, np.nan]),
                row_midpoint_cell_id=np.asarray([0, 1, 0, 1], dtype=np.int64),
                smoothing_weight=1.25,
            ),
            _cell_layer_result(
                layer_kind='vsub_t3',
                layer_index=3,
                velocity_m_s=np.asarray([np.nan, 4800.0, 5200.0], dtype=np.float64),
                used_mask=np.asarray([False, False, False, True], dtype=bool),
                residual_s=np.asarray([np.nan, np.nan, np.nan, 0.003]),
                row_midpoint_cell_id=np.asarray([1, 2, 1, 2], dtype=np.int64),
                smoothing_weight=0.75,
            ),
        ),
    )


def _cell_layer_result(
    *,
    layer_kind: str,
    layer_index: int,
    velocity_m_s: np.ndarray,
    used_mask: np.ndarray,
    residual_s: np.ndarray,
    row_midpoint_cell_id: np.ndarray,
    smoothing_weight: float,
) -> RefractionLayerSolveResult:
    active_cell_id = np.flatnonzero(np.isfinite(velocity_m_s)).astype(np.int64)
    inactive_cell_id = np.flatnonzero(~np.isfinite(velocity_m_s)).astype(np.int64)
    return RefractionLayerSolveResult(
        layer_kind=layer_kind,
        layer_index=layer_index,
        velocity_mode='solve_cell',
        source_time_term_s=np.zeros(2, dtype=np.float64),
        receiver_time_term_s=np.zeros(2, dtype=np.float64),
        node_time_term_s=np.zeros(3, dtype=np.float64),
        global_velocity_m_s=None,
        global_slowness_s_per_m=None,
        cell_velocity_m_s=velocity_m_s,
        cell_slowness_s_per_m=1.0 / velocity_m_s,
        trace_predicted_time_s_sorted=np.asarray(
            [0.050, 0.060, 0.070, 0.080],
            dtype=np.float64,
        ),
        trace_residual_s_sorted=residual_s,
        used_observation_mask_sorted=used_mask,
        layer_status='solved',
        qc={
            'layer_kind': layer_kind,
            'layer_index': layer_index,
            'velocity_mode': 'solve_cell',
            'n_total_cells': 3,
            'min_observations_per_cell': 1,
            'n_low_fold_cells': 0,
            'n_observations_rejected_by_low_fold_cell': 0,
            'n_observations_outside_grid': 0,
            'low_fold_cell_id': [],
            'cell_observation_count': [1, 2, 1],
            'n_cell_smoothing_rows': 1,
            'velocity_smoothing_weight': smoothing_weight,
        },
        active_cell_id=active_cell_id,
        inactive_cell_id=inactive_cell_id,
        cell_velocity_status=np.asarray(['solved'] * active_cell_id.size, dtype='<U16'),
        row_midpoint_cell_id=row_midpoint_cell_id,
        row_midpoint_velocity_m_s=velocity_m_s[row_midpoint_cell_id],
        rejected_by_robust_mask_sorted=np.zeros(4, dtype=bool),
    )


def _solve_cell_low_fold_result():
    base = _solve_cell_result()
    qc = dict(base.qc)
    qc.update(
        {
            'min_observations_per_cell': 2,
            'n_low_fold_cells': 1,
            'n_observations_rejected_by_low_fold_cell': 1,
            'low_fold_cell_rejection_reason': 'below_min_observations_per_cell',
            'low_fold_cell_id': [1],
            'cell_observation_count': [2, 1, 0],
        }
    )
    return replace(
        base,
        active_cell_id=np.asarray([0], dtype=np.int64),
        inactive_cell_id=np.asarray([1, 2], dtype=np.int64),
        cell_bedrock_slowness_s_per_m=np.asarray([1.0 / 2400.0], dtype=np.float64),
        cell_bedrock_velocity_m_s=np.asarray([2400.0], dtype=np.float64),
        cell_velocity_status=np.asarray(['solved'], dtype='<U16'),
        row_midpoint_cell_id=np.asarray([0, 0, 0], dtype=np.int64),
        node_v2_cell_id=np.asarray([0, 1, 2], dtype=np.int64),
        node_v2_m_s=np.asarray([2400.0, np.nan, np.nan], dtype=np.float64),
        node_v2_status=np.asarray(
            ['ok', 'low_fold_v2_cell', 'inactive_v2_cell'],
            dtype='<U32',
        ),
        node_datum_status=np.asarray(
            ['ok', 'low_fold_v2_cell', 'inactive_v2_cell'],
            dtype='<U32',
        ),
        source_v2_cell_id=np.asarray([0, 1], dtype=np.int64),
        source_v2_m_s=np.asarray([2400.0, np.nan], dtype=np.float64),
        source_v2_status=np.asarray(['ok', 'low_fold_v2_cell'], dtype='<U32'),
        source_datum_status=np.asarray(['ok', 'low_fold_v2_cell'], dtype='<U32'),
        source_refraction_shift_s=np.asarray([0.0025, np.nan], dtype=np.float64),
        receiver_v2_cell_id=np.asarray([1, 2], dtype=np.int64),
        receiver_v2_m_s=np.asarray([np.nan, np.nan], dtype=np.float64),
        receiver_v2_status=np.asarray(
            ['low_fold_v2_cell', 'inactive_v2_cell'],
            dtype='<U32',
        ),
        receiver_datum_status=np.asarray(
            ['low_fold_v2_cell', 'inactive_v2_cell'],
            dtype='<U32',
        ),
        receiver_refraction_shift_s=np.asarray([np.nan, np.nan], dtype=np.float64),
        source_v2_cell_id_sorted=np.asarray([0, 1, 0, 1], dtype=np.int64),
        source_v2_m_s_sorted=np.asarray(
            [2400.0, np.nan, 2400.0, np.nan],
            dtype=np.float64,
        ),
        source_v2_status_sorted=np.asarray(
            ['ok', 'low_fold_v2_cell', 'ok', 'low_fold_v2_cell'],
            dtype='<U32',
        ),
        receiver_v2_cell_id_sorted=np.asarray([1, 2, 2, 1], dtype=np.int64),
        receiver_v2_m_s_sorted=np.asarray(
            [np.nan, np.nan, np.nan, np.nan],
            dtype=np.float64,
        ),
        receiver_v2_status_sorted=np.asarray(
            [
                'low_fold_v2_cell',
                'inactive_v2_cell',
                'inactive_v2_cell',
                'low_fold_v2_cell',
            ],
            dtype='<U32',
        ),
        receiver_refraction_shift_s_sorted=np.asarray(
            [np.nan, np.nan, np.nan, np.nan],
            dtype=np.float64,
        ),
        refraction_trace_shift_s_sorted=np.asarray(
            [np.nan, np.nan, np.nan, np.nan],
            dtype=np.float64,
        ),
        trace_static_status_sorted=np.asarray(
            [
                'low_fold_v2_cell',
                'inactive_v2_cell',
                'inactive_v2_cell',
                'low_fold_v2_cell',
            ],
            dtype='<U32',
        ),
        trace_static_valid_mask_sorted=np.asarray(
            [False, False, False, False],
            dtype=bool,
        ),
        qc=qc,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _read_csv_with_fieldnames(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, str):
        return value.startswith('/')
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    return False


def _write_upstream_v1_artifacts(root: Path) -> None:
    (root / REFRACTION_V1_QC_JSON_NAME).write_text(
        '{"v1_status":"estimated"}',
        encoding='utf-8',
    )
    (root / REFRACTION_V1_ESTIMATES_CSV_NAME).write_text(
        'group_kind,group_key,status\nsource_endpoint,source:1,ok\n',
        encoding='utf-8',
    )
