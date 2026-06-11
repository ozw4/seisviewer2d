from __future__ import annotations

import csv
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
import app.statics.refraction.application.bedrock as refraction_bedrock_module
import app.statics.refraction.application.input_model as refraction_inputs_module
import app.statics.refraction.application.workflow as refraction_service_module
from app.api.schemas import RefractionStaticApplyRequest
from app.main import app
from app.statics.refraction.adapters.seisviewer2d.trace_store import (
    SeisViewer2DRefractionTraceStoreProvider,
)
from app.statics.refraction.application.apply_trace_store import (
    CORRECTED_FILE_JSON_NAME,
    REFRACTION_STATIC_APPLY_QC_JSON_NAME,
)
from app.statics.refraction.artifacts import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_JSON_NAME,
    REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
    REFRACTION_REDUCED_TIME_QC_CSV_NAME,
    REFRACTION_REDUCED_TIME_QC_JSON_NAME,
    REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_STATICS_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_JSON_NAME,
    REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME,
    REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_FAILURE_DIAGNOSTICS_JSON_NAME,
    REFRACTION_STATIC_HISTORY_JSON_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
    REFRACTION_V1_ESTIMATES_CSV_NAME,
    REFRACTION_V1_QC_JSON_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    UPLOADED_REFRACTION_PICKS_NPZ_NAME,
)
from app.statics.refraction.application.design_matrix import (
    REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME,
    REFRACTION_DESIGN_MATRIX_QC_JSON_NAME,
    build_refraction_static_design_matrix_from_arrays,
    write_refraction_design_matrix_diagnostics_artifacts,
)
from app.statics.refraction.application.export_service import (
    CANONICAL_SOURCE_RECEIVER_STATIC_TABLE_CSV_NAME,
)
from app.statics.refraction.domain.layer_observations import (
    build_refraction_layer_observation_masks,
    refraction_layer_observation_qc,
)
from app.statics.refraction.application.preflight_diagnostics import (
    REFRACTION_STATIC_PREFLIGHT_OBSERVATIONS_CSV_NAME,
    REFRACTION_STATIC_PREFLIGHT_QC_JSON_NAME,
    RefractionStaticPreflightDiagnostics,
    RefractionStaticPreflightError,
    write_refraction_static_preflight_artifacts,
)
from app.statics.refraction.adapters.seisviewer2d.workflow_runner import run_refraction_static_apply_job
from app.statics.refraction.domain.source_depth import (
    REFRACTION_SOURCE_DEPTH_QC_JSON_NAME,
    REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
    resolve_refraction_source_depth,
    write_refraction_source_depth_artifacts,
)
from app.statics.refraction.domain.v1 import RefractionV1EstimateResult
from app.services.trace_store_registration import trace_store_cache_key
from app.tests._refraction_static_synthetic import (
    SYNTHETIC_SH1_TOLERANCE_M,
    SYNTHETIC_T1_TOLERANCE_MS,
    SYNTHETIC_V1_M_S,
    SYNTHETIC_V1_TOLERANCE_M_S,
    SYNTHETIC_V2_M_S,
    SYNTHETIC_V2_TOLERANCE_M_S,
    SYNTHETIC_WCOR_TOLERANCE_MS,
    expected_sh1_m_for_node,
    expected_t1_s_for_node,
    expected_wcor_s_for_node,
    synthetic_cell_refracted_arrival_input_model,
    synthetic_cell_refraction_apply_request,
    synthetic_direct_arrival_input_model,
    synthetic_refracted_arrival_input_model,
    synthetic_refraction_apply_request,
)
from app.tests._refraction_multilayer_synthetic import (
    SYNTHETIC_MULTILAYER_V1_M_S,
    SYNTHETIC_MULTILAYER_V2_M_S,
    SYNTHETIC_MULTILAYER_V3_M_S,
    SYNTHETIC_MULTILAYER_VSUB_M_S,
)
from app.tests.test_refraction_static_multilayer_2layer_e2e import (
    _CELL_VELOCITY_ARTIFACT_NAMES,
    _make_two_layer_fixture,
)
from app.tests.test_refraction_static_apply_trace_store import (
    DT,
    KEY1,
    KEY2,
    SOURCE_FILE_ID,
    _valid_result,
    _write_source_store,
)
from app.tests._refraction_static_artifact_helpers import _result as _artifact_result


FINAL_REFRACTION_ARTIFACTS = {
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATICS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
    REFRACTION_REDUCED_TIME_QC_CSV_NAME,
    REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
    REFRACTION_REDUCED_TIME_QC_JSON_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME,
    REFRACTION_STATIC_COMPONENT_QC_JSON_NAME,
    REFRACTION_STATIC_HISTORY_JSON_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_JSON_NAME,
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
}
REQUEST_JSON_NAME = 'refraction_static_request.json'


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(tmp_path / 'jobs'))
    state = app.state.sv
    with state.lock:
        state.jobs.clear()
        state.cached_readers.clear()
        state.file_registry.clear()
    with TestClient(app) as test_client:
        yield test_client
    with state.lock:
        state.jobs.clear()
        state.cached_readers.clear()
        state.file_registry.clear()


def _payload() -> dict[str, Any]:
    return {
        'file_id': 'raw-file-id',
        'key1_byte': 189,
        'key2_byte': 193,
        'pick_source': {
            'kind': 'batch_predicted_npz',
            'job_id': 'first-break-job-id',
            'artifact_name': 'predicted_picks_time_s.npz',
        },
        'geometry': {
            'source_id_byte': 9,
            'receiver_id_byte': 13,
            'source_x_byte': 73,
            'source_y_byte': 77,
            'receiver_x_byte': 81,
            'receiver_y_byte': 85,
            'source_elevation_byte': 45,
            'receiver_elevation_byte': 41,
            'source_depth_byte': None,
            'coordinate_scalar_byte': 71,
            'elevation_scalar_byte': 69,
            'coordinate_unit': 'm',
            'elevation_unit': 'm',
        },
        'linkage': {
            'mode': 'required',
            'job_id': 'linkage-job-id',
            'artifact_name': 'geometry_linkage.npz',
        },
        'model': {
            'method': 'gli_variable_thickness',
            'weathering_velocity_m_s': 800.0,
            'bedrock_velocity_mode': 'solve_global',
            'bedrock_velocity_m_s': None,
            'initial_bedrock_velocity_m_s': 2500.0,
            'min_bedrock_velocity_m_s': 1200.0,
            'max_bedrock_velocity_m_s': 6000.0,
            'max_weathering_thickness_m': None,
        },
        'moveout': {
            'model': 'head_wave_linear_offset',
            'distance_source': 'geometry',
            'offset_byte': 37,
            'min_offset_m': None,
            'max_offset_m': None,
            'allow_missing_offset': False,
            'max_geometry_offset_mismatch_m': None,
        },
        'solver': {
            'damping': 0.01,
            'min_picks_per_node': 1,
            'max_abs_half_intercept_time_ms': 500.0,
            'robust': {
                'enabled': True,
                'method': 'mad',
                'threshold': 3.5,
                'max_iterations': 5,
                'min_used_fraction': 0.5,
                'min_used_observations': 1,
            },
        },
        'datum': {
            'mode': 'floating_and_flat',
            'floating_datum_mode': 'constant',
            'flat_datum_elevation_m': 200.0,
            'floating_datum_elevation_m': 100.0,
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


def _uploaded_pick_payload() -> dict[str, Any]:
    payload = _payload()
    payload['pick_source'] = {'kind': 'uploaded_npz'}
    return payload


def _validation_uploaded_pick_payload() -> dict[str, Any]:
    payload = _uploaded_pick_payload()
    payload['linkage'] = {'mode': 'none'}
    return payload


def _pick_npz_bytes() -> bytes:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        picks_time_s=np.asarray([0.010, 0.020], dtype=np.float32),
        n_traces=np.asarray(2, dtype=np.int64),
        n_samples=np.asarray(100, dtype=np.int64),
        dt=np.asarray(0.001, dtype=np.float64),
    )
    return buffer.getvalue()


def _write_minimal_preflight_diagnostics(job_dir: Path) -> None:
    diagnostics = RefractionStaticPreflightDiagnostics(
        status='error',
        warnings=[],
        errors=['No valid refraction observations remain after preflight filtering.'],
        summary={
            'observation_filters': {
                'n_total_traces': 2,
                'n_used_for_inversion': 0,
            }
        },
        observation_reason_counts={'offset_gate': 2},
        endpoint_summary={},
    )
    write_refraction_static_preflight_artifacts(job_dir, diagnostics)


def _write_minimal_design_matrix_diagnostics(job_dir: Path) -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.20]),
        valid_observation_mask_sorted=np.asarray([True]),
        source_node_id_sorted=np.asarray([17]),
        receiver_node_id_sorted=np.asarray([21]),
        source_endpoint_key_sorted=np.asarray(['source:1007']),
        receiver_endpoint_key_sorted=np.asarray(['receiver:2001']),
        distance_m_sorted=np.asarray([500.0]),
        node_id=np.asarray([17, 21]),
        node_kind=np.asarray(['source', 'receiver']),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
        rejection_reason_sorted=np.asarray(['ok']),
    )
    write_refraction_design_matrix_diagnostics_artifacts(job_dir, design)


def test_refraction_static_apply_endpoint_starts_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []

    def _capture_start_job_thread(**kwargs: Any) -> object:
        started.append(kwargs)
        return object()

    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        _capture_start_job_thread,
    )

    response = client.post('/statics/refraction/apply', json=_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload['state'] == 'queued'
    assert isinstance(payload['job_id'], str)
    assert len(started) == 1
    assert started[0]['target'] is statics_router_module.run_refraction_static_apply_job
    assert started[0]['args'][0] == payload['job_id']
    assert isinstance(started[0]['args'][1], RefractionStaticApplyRequest)
    assert started[0]['args'][2] is client.app.state.sv

    state = client.app.state.sv
    with state.lock:
        job = dict(state.jobs[payload['job_id']])

    assert job['status'] == 'queued'
    assert job['job_type'] == 'statics'
    assert job['statics_kind'] == 'refraction'
    assert job['file_id'] == 'raw-file-id'
    assert job['key1_byte'] == 189
    assert job['key2_byte'] == 193


def test_refraction_static_apply_endpoint_rejects_uploaded_npz_json(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )

    response = client.post(
        '/statics/refraction/apply',
        json=_uploaded_pick_payload(),
    )

    assert response.status_code == 422
    assert '/statics/refraction/apply-with-picks' in response.text
    assert started == []
    with client.app.state.sv.lock:
        assert len(client.app.state.sv.jobs) == 0


def test_refraction_apply_with_uploaded_picks_accepts_multipart_npz(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    upload_bytes = _pick_npz_bytes()

    response = client.post(
        '/statics/refraction/apply-with-picks',
        data={'request_json': json.dumps(_uploaded_pick_payload())},
        files={
            'pick_npz': (
                '../predicted_picks_time_s.npz',
                upload_bytes,
                'application/octet-stream',
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['state'] == 'queued'
    assert len(started) == 1
    assert started[0]['target'] is statics_router_module.run_refraction_static_apply_job
    assert started[0]['args'][0] == payload['job_id']
    assert isinstance(started[0]['args'][1], RefractionStaticApplyRequest)
    assert started[0]['args'][1].pick_source.kind == 'uploaded_npz'
    assert started[0]['args'][3].name == 'uploaded_picks_time_s.npz'
    assert started[0]['args'][4] == {
        'original_filename': '../predicted_picks_time_s.npz',
        'stored_name': 'uploaded_picks_time_s.npz',
    }

    state = client.app.state.sv
    with state.lock:
        job = dict(state.jobs[payload['job_id']])
    job_dir = Path(str(job['artifacts_dir']))
    assert (job_dir / 'uploaded_picks_time_s.npz').read_bytes() == upload_bytes
    assert job['pick_source'] == {
        'kind': 'uploaded_npz',
        'original_filename': '../predicted_picks_time_s.npz',
        'stored_name': 'uploaded_picks_time_s.npz',
    }


def test_refraction_apply_with_uploaded_picks_rejects_missing_file(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )

    response = client.post(
        '/statics/refraction/apply-with-picks',
        data={'request_json': json.dumps(_uploaded_pick_payload())},
    )

    assert response.status_code == 422
    assert started == []
    with client.app.state.sv.lock:
        assert len(client.app.state.sv.jobs) == 0


def _validation_pick_npz_bytes(picks: np.ndarray | None = None) -> bytes:
    buffer = io.BytesIO()
    values = (
        np.asarray([0.010, 0.020, 0.030, 0.040], dtype=np.float64)
        if picks is None
        else np.asarray(picks, dtype=np.float64)
    )
    np.savez(
        buffer,
        pick_time_s=values,
        n_traces=np.asarray(values.shape[0], dtype=np.int64),
        n_samples=np.asarray(100, dtype=np.int64),
        dt=np.asarray(0.001, dtype=np.float64),
        order=np.asarray('trace_store_sorted'),
    )
    return buffer.getvalue()


class _ValidationReader:
    key1_byte = 189
    key2_byte = 193

    def __init__(self) -> None:
        n_traces = 4
        self.traces = np.zeros((n_traces, 100), dtype=np.float32)
        self.meta = {'dt': 0.001, 'n_traces': n_traces}
        self._sorted_to_original = np.arange(n_traces, dtype=np.int64)
        zeros = np.zeros(n_traces, dtype=np.float64)
        ones = np.ones(n_traces, dtype=np.int32)
        self._headers = {
            9: np.asarray([1, 1, 2, 2], dtype=np.int32),
            13: np.asarray([10, 11, 12, 13], dtype=np.int32),
            73: zeros,
            77: zeros,
            81: np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float64),
            85: zeros,
            45: zeros,
            41: zeros,
            71: ones,
            69: ones,
        }

    def get_n_samples(self) -> int:
        return 100

    def get_sorted_to_original(self) -> np.ndarray:
        return self._sorted_to_original

    def ensure_header(self, byte: int) -> np.ndarray:
        return self._headers[byte]


def test_validate_with_picks_returns_preflight_summary(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SeisViewer2DRefractionTraceStoreProvider,
        'get_reader',
        lambda *_args, **_kwargs: _ValidationReader(),
    )
    monkeypatch.setattr(
        SeisViewer2DRefractionTraceStoreProvider,
        'get_dt',
        lambda *_args, **_kwargs: 0.001,
    )

    response = client.post(
        '/statics/refraction/validate-with-picks',
        data={'request_json': json.dumps(_validation_uploaded_pick_payload())},
        files={
            'pick_npz': (
                'uploaded-picks.npz',
                _validation_pick_npz_bytes(),
                'application/octet-stream',
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ok'
    assert payload['target'] == {
        'file_id': 'raw-file-id',
        'key1_byte': 189,
        'key2_byte': 193,
    }
    assert payload['pick_npz']['selected_key'] == 'pick_time_s'
    assert payload['pick_npz']['shape'] == [4]
    diagnostics = payload['diagnostics']
    assert diagnostics['n_total_traces'] == 4
    assert diagnostics['n_finite_picks'] == 4
    assert diagnostics['n_used_for_inversion'] == 4
    assert diagnostics['n_unique_source_endpoints'] == 2
    assert diagnostics['n_unique_receiver_endpoints'] == 4
    assert diagnostics['offset_m'] == {'min': 100.0, 'median': 250.0, 'max': 400.0}
    assert diagnostics['filter_reason_counts']['offset_gate'] == 0
    with client.app.state.sv.lock:
        assert len(client.app.state.sv.jobs) == 0


def test_validate_with_picks_rejects_missing_npz(
    client: TestClient,
) -> None:
    response = client.post(
        '/statics/refraction/validate-with-picks',
        data={'request_json': json.dumps(_validation_uploaded_pick_payload())},
    )

    assert response.status_code == 422
    with client.app.state.sv.lock:
        assert len(client.app.state.sv.jobs) == 0


def test_validate_with_picks_reports_corrupt_npz_without_job(
    client: TestClient,
) -> None:
    response = client.post(
        '/statics/refraction/validate-with-picks',
        data={'request_json': json.dumps(_validation_uploaded_pick_payload())},
        files={
            'pick_npz': (
                'corrupt-picks.npz',
                b'not an npz archive',
                'application/octet-stream',
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'error'
    assert payload['pick_npz']['selected_key'] is None
    assert payload['pick_npz']['shape'] is None
    assert payload['pick_npz']['keys'] == []
    assert any('Unable to read pick NPZ' in error for error in payload['errors'])
    with client.app.state.sv.lock:
        assert len(client.app.state.sv.jobs) == 0


def test_validate_with_picks_reports_pick_count_mismatch_without_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SeisViewer2DRefractionTraceStoreProvider,
        'get_reader',
        lambda *_args, **_kwargs: _ValidationReader(),
    )
    monkeypatch.setattr(
        SeisViewer2DRefractionTraceStoreProvider,
        'get_dt',
        lambda *_args, **_kwargs: 0.001,
    )

    response = client.post(
        '/statics/refraction/validate-with-picks',
        data={'request_json': json.dumps(_validation_uploaded_pick_payload())},
        files={
            'pick_npz': (
                'short-picks.npz',
                _validation_pick_npz_bytes(np.asarray([0.010, 0.020, 0.030])),
                'application/octet-stream',
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'error'
    assert payload['pick_npz']['selected_key'] == 'pick_time_s'
    assert payload['pick_npz']['shape'] == [3]
    assert any('n_traces mismatch' in error for error in payload['errors'])
    with client.app.state.sv.lock:
        assert len(client.app.state.sv.jobs) == 0


def test_refraction_apply_with_uploaded_picks_rejects_non_uploaded_npz_kind(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )

    response = client.post(
        '/statics/refraction/apply-with-picks',
        data={'request_json': json.dumps(_payload())},
        files={
            'pick_npz': (
                'predicted_picks_time_s.npz',
                _pick_npz_bytes(),
                'application/octet-stream',
            )
        },
    )

    assert response.status_code == 422
    assert 'uploaded_npz' in response.text
    assert started == []
    with client.app.state.sv.lock:
        assert len(client.app.state.sv.jobs) == 0


def test_refraction_apply_with_uploaded_picks_rejects_empty_upload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )

    response = client.post(
        '/statics/refraction/apply-with-picks',
        data={'request_json': json.dumps(_uploaded_pick_payload())},
        files={
            'pick_npz': (
                'predicted_picks_time_s.npz',
                b'',
                'application/octet-stream',
            )
        },
    )

    assert response.status_code == 422
    assert 'empty' in response.text
    assert started == []
    with client.app.state.sv.lock:
        assert len(client.app.state.sv.jobs) == 0


def test_refraction_apply_with_uploaded_picks_stores_input_artifact_metadata(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    response = client.post(
        '/statics/refraction/apply-with-picks',
        data={'request_json': json.dumps(_uploaded_pick_payload())},
        files={
            'pick_npz': (
                'predicted_picks_time_s.npz',
                _pick_npz_bytes(),
                'application/octet-stream',
            )
        },
    )
    assert response.status_code == 200
    job_id = response.json()['job_id']

    build_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_static_input_model',
        lambda **kwargs: build_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'compute_weathering_replacement_statics_from_first_breaks',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_datum_statics',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'write_refraction_static_artifacts',
        lambda **_kwargs: object(),
    )

    started[0]['target'](*started[0]['args'])

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    job_dir = Path(str(job['artifacts_dir']))
    request_payload = json.loads(
        (job_dir / REQUEST_JSON_NAME).read_text(encoding='utf-8')
    )
    assert request_payload['request']['pick_source'] == {
        'kind': 'uploaded_npz',
        'original_filename': 'predicted_picks_time_s.npz',
        'stored_name': 'uploaded_picks_time_s.npz',
    }
    assert len(build_calls) == 1
    assert build_calls[0]['uploaded_pick_npz_path'] == (
        job_dir / 'uploaded_picks_time_s.npz'
    )
    assert build_calls[0]['uploaded_pick_metadata'] == {
        'original_filename': 'predicted_picks_time_s.npz',
        'stored_name': 'uploaded_picks_time_s.npz',
    }


def test_refraction_apply_with_uploaded_picks_job_status_and_files(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **_kwargs: object(),
    )
    response = client.post(
        '/statics/refraction/apply-with-picks',
        data={'request_json': json.dumps(_uploaded_pick_payload())},
        files={
            'pick_npz': (
                'predicted_picks_time_s.npz',
                _pick_npz_bytes(),
                'application/octet-stream',
            )
        },
    )
    assert response.status_code == 200
    job_id = response.json()['job_id']

    status_response = client.get(f'/statics/job/{job_id}/status')
    assert status_response.status_code == 200
    assert status_response.json()['state'] == 'queued'

    files_response = client.get(f'/statics/job/{job_id}/files')
    assert files_response.status_code == 200
    file_names = {item['name'] for item in files_response.json()['files']}
    assert 'uploaded_picks_time_s.npz' in file_names


def test_uploaded_npz_preserved_for_failed_apply_with_picks_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_synthetic_failure(**_kwargs: Any) -> object:
        raise RuntimeError('synthetic failure')

    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: kwargs['target'](*kwargs['args']),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_static_input_model',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'compute_weathering_replacement_statics_from_first_breaks',
        _raise_synthetic_failure,
    )
    upload_bytes = _pick_npz_bytes()

    response = client.post(
        '/statics/refraction/apply-with-picks',
        data={'request_json': json.dumps(_uploaded_pick_payload())},
        files={
            'pick_npz': (
                'predicted_picks_time_s.npz',
                upload_bytes,
                'application/octet-stream',
            )
        },
    )

    assert response.status_code == 200
    job_id = response.json()['job_id']
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    job_dir = Path(str(job['artifacts_dir']))
    assert job['status'] == 'error'
    assert (job_dir / UPLOADED_REFRACTION_PICKS_NPZ_NAME).read_bytes() == upload_bytes
    listed = _listed_job_files(client, job_id)
    assert UPLOADED_REFRACTION_PICKS_NPZ_NAME in listed
    assert REFRACTION_STATIC_FAILURE_DIAGNOSTICS_JSON_NAME in listed


def test_apply_with_uploaded_picks_synthetic_one_layer_job_completes(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-uploaded-picks-synthetic-one-layer'
    job_dir = tmp_path / 'jobs' / job_id
    input_model = synthetic_refracted_arrival_input_model()
    uploaded_npz_path = job_dir / 'uploaded_picks_time_s.npz'
    job_dir.mkdir(parents=True)
    np.savez(
        uploaded_npz_path,
        pick_time_s=input_model.pick_time_s_sorted,
        valid_pick_mask=input_model.valid_pick_mask_sorted,
        n_traces=np.asarray(input_model.n_traces, dtype=np.int64),
        n_samples=np.asarray(4000, dtype=np.int64),
        dt=np.asarray(0.001, dtype=np.float64),
    )
    req_payload = synthetic_refraction_apply_request(
        conversion_mode='t1lsst_1layer',
    ).model_dump(mode='json')
    req_payload['pick_source'] = {'kind': 'uploaded_npz'}
    req = RefractionStaticApplyRequest.model_validate(req_payload)
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    build_calls: list[dict[str, Any]] = []

    def _build_uploaded_input_model(**kwargs: Any) -> object:
        build_calls.append(kwargs)
        return input_model

    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_static_input_model',
        _build_uploaded_input_model,
    )

    run_refraction_static_apply_job(
        job_id,
        req,
        client.app.state.sv,
        uploaded_npz_path,
        {
            'original_filename': 'field-picks.npz',
            'stored_name': 'uploaded_picks_time_s.npz',
        },
    )

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    assert len(build_calls) == 1
    assert build_calls[0]['uploaded_pick_npz_path'] == uploaded_npz_path
    assert build_calls[0]['uploaded_pick_metadata'] == {
        'original_filename': 'field-picks.npz',
        'stored_name': 'uploaded_picks_time_s.npz',
    }
    assert build_calls[0]['req'].pick_source.kind == 'uploaded_npz'
    assert build_calls[0]['req'].pick_source.job_id is None
    assert build_calls[0]['req'].pick_source.artifact_name is None
    file_names = _job_file_names(job_dir)
    assert {
        REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        REFRACTION_STATIC_QC_JSON_NAME,
        REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
        REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
        'uploaded_picks_time_s.npz',
    }.issubset(file_names)
    qc_response = client.post(
        '/statics/refraction/qc',
        json={'job_id': job_id, 'max_points': 2000},
    )
    assert qc_response.status_code == 200
    qc_payload = qc_response.json()
    assert qc_payload['job_id'] == job_id
    assert qc_payload['summary']['workflow'] == 'refraction_statics'


def test_failed_static_job_lists_preflight_diagnostics(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = RefractionStaticApplyRequest.model_validate(_payload())
    job_id = 'refraction-preflight-failure-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)

    def _raise_preflight(**kwargs: Any) -> object:
        _write_minimal_preflight_diagnostics(kwargs['job_dir'])
        raise RefractionStaticPreflightError(
            'Refraction static preflight failed: no observations'
        )

    monkeypatch.setattr(
        refraction_service_module,
        'compute_weathering_replacement_statics_from_first_breaks',
        _raise_preflight,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'error'
    listed = _listed_job_files(client, job_id)
    assert REFRACTION_STATIC_PREFLIGHT_QC_JSON_NAME in listed
    assert REFRACTION_STATIC_PREFLIGHT_OBSERVATIONS_CSV_NAME not in listed
    assert REFRACTION_STATIC_FAILURE_DIAGNOSTICS_JSON_NAME in listed

    failure = json.loads(
        (job_dir / REFRACTION_STATIC_FAILURE_DIAGNOSTICS_JSON_NAME).read_text(
            encoding='utf-8'
        )
    )
    assert failure['state'] == 'error'
    assert failure['failed_stage'] == 'preflight'
    assert failure['error_type'] == 'RefractionStaticPreflightError'
    assert REFRACTION_STATIC_PREFLIGHT_QC_JSON_NAME in (
        failure['available_diagnostic_artifacts']
    )

    download = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': REFRACTION_STATIC_PREFLIGHT_QC_JSON_NAME},
    )
    assert download.status_code == 200
    assert download.json()['status'] == 'error'


def test_failed_static_job_lists_design_matrix_diagnostics(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = synthetic_refraction_apply_request()
    input_model = synthetic_refracted_arrival_input_model()
    job_id = 'refraction-design-matrix-failure-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)

    monkeypatch.setattr(
        refraction_service_module,
        'resolve_refraction_first_layer_request',
        lambda **_kwargs: SimpleNamespace(
            req=req,
            resolved=refraction_service_module.ResolvedRefractionFirstLayer(
                mode='constant',
                weathering_velocity_m_s=SYNTHETIC_V1_M_S,
                status='configured',
                qc={'weathering_velocity_m_s': SYNTHETIC_V1_M_S},
            ),
            input_model=input_model,
            upstream_artifact_names=(),
        ),
    )
    original_build_design = (
        refraction_bedrock_module.build_refraction_static_design_matrix
    )

    def _build_design_with_zero_active_column(**kwargs: Any) -> object:
        design = original_build_design(**kwargs)
        matrix = design.matrix.tolil()
        matrix[:, 0] = 0.0
        matrix = matrix.tocsr()
        matrix.eliminate_zeros()
        diagnostics = tuple(
            replace(
                item,
                n_nonzero_entries=0,
                status='all_zero_active_column',
                reason='no_observations_for_node',
            )
            if int(item.matrix_column) == 0
            else item
            for item in design.node_diagnostics
        )
        return replace(
            design,
            matrix=matrix,
            node_diagnostics=diagnostics,
            design_matrix_qc=None,
        )

    monkeypatch.setattr(
        refraction_bedrock_module,
        'build_refraction_static_design_matrix',
        _build_design_with_zero_active_column,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'error'
    listed = _listed_job_files(client, job_id)
    assert REFRACTION_DESIGN_MATRIX_QC_JSON_NAME in listed
    assert REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME in listed
    assert REFRACTION_STATIC_FAILURE_DIAGNOSTICS_JSON_NAME in listed

    failure = json.loads(
        (job_dir / REFRACTION_STATIC_FAILURE_DIAGNOSTICS_JSON_NAME).read_text(
            encoding='utf-8'
        )
    )
    assert failure['failed_stage'] == 'design_matrix'
    assert 'all-zero active-node columns' in failure['error_message']
    assert REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME in (
        failure['available_diagnostic_artifacts']
    )
    qc = json.loads(
        (job_dir / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME).read_text(encoding='utf-8')
    )
    assert qc['n_all_zero_active_node_columns'] == 1


def test_refraction_static_apply_endpoint_accepts_omitted_linkage(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    payload = _payload()
    del payload['linkage']

    response = client.post('/statics/refraction/apply', json=payload)

    assert response.status_code == 200
    assert len(started) == 1
    req = started[0]['args'][1]
    assert isinstance(req, RefractionStaticApplyRequest)
    assert req.linkage.mode == 'none'
    assert req.linkage.job_id is None


def test_refraction_static_apply_endpoint_rejects_invalid_schema_without_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    payload = _payload()
    payload['file_id'] = ''

    response = client.post('/statics/refraction/apply', json=payload)

    assert response.status_code == 422
    assert started == []
    with client.app.state.sv.lock:
        assert len(client.app.state.sv.jobs) == 0


def test_refraction_static_apply_endpoint_starts_solve_cell_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    payload = _payload()
    payload['model'].update(
        {
            'weathering_velocity_m_s': None,
            'first_layer': {
                'mode': 'constant',
                'weathering_velocity_m_s': 800.0,
            },
            'bedrock_velocity_mode': 'solve_cell',
            'initial_bedrock_velocity_m_s': 2400.0,
            'refractor_cell': {
                'number_of_cell_x': 4,
                'size_of_cell_x_m': 500.0,
                'x_coordinate_origin_m': 0.0,
                'number_of_cell_y': 1,
                'size_of_cell_y_m': 1000.0,
                'y_coordinate_origin_m': 0.0,
                'assignment_mode': 'midpoint',
                'outside_grid_policy': 'reject',
                'min_observations_per_cell': 5,
                'velocity_smoothing_weight': 0.0,
            },
        }
    )

    response = client.post('/statics/refraction/apply', json=payload)

    assert response.status_code == 200
    assert response.json()['state'] == 'queued'
    assert len(started) == 1
    assert started[0]['target'] is statics_router_module.run_refraction_static_apply_job
    assert started[0]['args'][1].model.bedrock_velocity_mode == 'solve_cell'
    with client.app.state.sv.lock:
        assert len(client.app.state.sv.jobs) == 1


def test_refraction_static_rejects_public_estimated_v1_value(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    payload = _payload()
    payload['model']['weathering_velocity_m_s'] = None
    payload['model']['first_layer'] = {
        'mode': 'estimate_direct_arrival',
        'weathering_velocity_m_s': 812.5,
        'min_direct_offset_m': 20.0,
        'max_direct_offset_m': 140.0,
    }

    response = client.post('/statics/refraction/apply', json=payload)

    assert response.status_code == 422
    assert 'model.first_layer.weathering_velocity_m_s must be omitted' in (
        response.text
    )
    assert started == []
    with client.app.state.sv.lock:
        assert len(client.app.state.sv.jobs) == 0


def test_run_refraction_static_apply_job_rejects_multilayer_conversion_explicitly(
    client: TestClient,
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload['model'] = {
        'method': 'multilayer_time_term',
        'first_layer': {
            'mode': 'constant',
            'weathering_velocity_m_s': 800.0,
        },
        'layers': [
            {
                'kind': 'v2_t1',
                'enabled': True,
                'min_offset_m': 300.0,
                'max_offset_m': None,
                'velocity_mode': 'solve_global',
                'initial_velocity_m_s': 2400.0,
                'min_velocity_m_s': 1200.0,
                'max_velocity_m_s': 5000.0,
            },
        ],
    }
    payload['conversion'] = {'mode': 't1lsst_multilayer', 'layer_count': 1}
    req = RefractionStaticApplyRequest.model_validate(payload)
    job_id = 'refraction-multilayer-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'error'
    assert 'conversion.mode=t1lsst_multilayer' in str(job['message'])
    assert 'public multi-layer refraction apply requires' in str(job['message'])
    assert REQUEST_JSON_NAME in _job_file_names(job_dir)
    assert FINAL_REFRACTION_ARTIFACTS.isdisjoint(_job_file_names(job_dir))


def test_run_refraction_static_apply_job_completes_two_layer_global_v2_v3(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='grid_3d',
        v2_velocity_mode='solve_global',
    )
    req = _two_layer_apply_request(fixture)
    job_id = 'refraction-two-layer-global-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    build_calls: list[dict[str, Any]] = []

    def _build_input_model(**kwargs: Any) -> object:
        build_calls.append(kwargs)
        return fixture.input_model

    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_static_input_model',
        _build_input_model,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    assert job['message'] == 'refraction_static_artifacts_written_artifact_only'
    assert len(build_calls) == 1
    assert build_calls[0]['req'] is req
    file_names = _job_file_names(job_dir)
    assert REQUEST_JSON_NAME in file_names
    assert FINAL_REFRACTION_ARTIFACTS.issubset(file_names)
    assert _CELL_VELOCITY_ARTIFACT_NAMES.isdisjoint(file_names)

    request_payload = json.loads(
        (job_dir / REQUEST_JSON_NAME).read_text(encoding='utf-8')
    )
    assert request_payload['request']['model']['method'] == 'multilayer_time_term'
    assert request_payload['request']['conversion'] == {
        'mode': 't1lsst_multilayer',
        'layer_count': 2,
    }
    qc = json.loads(
        (job_dir / REFRACTION_STATIC_QC_JSON_NAME).read_text(encoding='utf-8')
    )
    assert qc['method'] == 'multilayer_time_term'
    assert qc['conversion_mode'] == 't1lsst_multilayer'
    assert qc['layer_count'] == 2
    _assert_two_layer_solution_arrays(job_dir)


def test_run_refraction_static_apply_job_completes_two_layer_cell_v2_global_v3(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='line_2d_projected',
        v2_velocity_mode='solve_cell',
    )
    req = _two_layer_apply_request(fixture)
    job_id = 'refraction-two-layer-cell-v2-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)

    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_static_input_model',
        lambda **_kwargs: fixture.input_model,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    file_names = _job_file_names(job_dir)
    assert FINAL_REFRACTION_ARTIFACTS.issubset(file_names)
    assert _CELL_VELOCITY_ARTIFACT_NAMES.issubset(file_names)
    _assert_two_layer_solution_arrays(job_dir)

    manifest = json.loads(
        (job_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    listed = {item['name'] for item in manifest['artifacts']}
    assert _CELL_VELOCITY_ARTIFACT_NAMES.issubset(listed)
    qc = json.loads(
        (job_dir / REFRACTION_STATIC_QC_JSON_NAME).read_text(encoding='utf-8')
    )
    assert qc['velocity']['cell_velocity_layer_kind'] == 'v2_t1'
    assert qc['velocity']['cell_velocity_component'] == 'v2'


def test_public_apply_accepts_three_layer_multilayer_request(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _three_layer_apply_payload()
    req = RefractionStaticApplyRequest.model_validate(payload)
    job_id = 'refraction-vsub-t3-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    input_model = object()
    build_calls: list[dict[str, Any]] = []
    workflow_calls: list[dict[str, Any]] = []

    def _build_input_model(**kwargs: Any) -> object:
        build_calls.append(kwargs)
        return input_model

    def _compute_workflow(**kwargs: Any) -> SimpleNamespace:
        workflow_calls.append(kwargs)
        return SimpleNamespace(datum_result=_three_layer_contract_datum_result())

    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_static_input_model',
        _build_input_model,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'compute_refraction_multilayer_datum_statics_from_input_model',
        _compute_workflow,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done', job.get('message')
    assert len(build_calls) == 1
    assert build_calls[0]['req'] is req
    assert len(workflow_calls) == 1
    assert workflow_calls[0]['input_model'] is input_model
    assert workflow_calls[0]['model'] is req.model
    assert REQUEST_JSON_NAME in _job_file_names(job_dir)
    assert FINAL_REFRACTION_ARTIFACTS.issubset(_job_file_names(job_dir))

    qc = json.loads(
        (job_dir / REFRACTION_STATIC_QC_JSON_NAME).read_text(encoding='utf-8')
    )
    assert qc['conversion_mode'] == 't1lsst_multilayer'
    assert qc['layer_count'] == 3
    assert qc['enabled_layer_kinds'] == ['v2_t1', 'v3_t2', 'vsub_t3']
    _assert_three_layer_solution_arrays(job_dir)


def test_three_layer_job_downloads_source_receiver_static_tables(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _three_layer_apply_payload()
    req = RefractionStaticApplyRequest.model_validate(payload)
    job_id = 'refraction-vsub-t3-download-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)

    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_static_input_model',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'compute_refraction_multilayer_datum_statics_from_input_model',
        lambda **_kwargs: SimpleNamespace(
            datum_result=_three_layer_contract_datum_result()
        ),
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done', job.get('message')

    manifest = json.loads(
        (job_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(
            encoding='utf-8'
        )
    )
    manifest_names = {item['name'] for item in manifest['artifacts']}
    assert {
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    }.issubset(manifest_names)

    for artifact_name in manifest_names:
        response = client.get(
            f'/statics/job/{job_id}/download',
            params={'name': artifact_name},
        )
        assert response.status_code == 200, artifact_name
        assert response.content, artifact_name


def test_public_apply_rejects_unsupported_vsub_cell_velocity_mode(
    client: TestClient,
    tmp_path: Path,
) -> None:
    payload = _three_layer_apply_payload()
    payload['model']['refractor_cell'] = {
        'number_of_cell_x': 5,
        'size_of_cell_x_m': 500.0,
        'x_coordinate_origin_m': 0.0,
        'number_of_cell_y': 1,
        'size_of_cell_y_m': None,
        'y_coordinate_origin_m': 0.0,
        'assignment_mode': 'midpoint',
        'outside_grid_policy': 'reject',
        'coordinate_mode': 'grid_3d',
        'min_observations_per_cell': 1,
        'velocity_smoothing_weight': 0.0,
        'smoothing_reference_distance_m': None,
    }
    vsub_layer = payload['model']['layers'][2]
    vsub_layer['velocity_mode'] = 'solve_cell'
    vsub_layer['fixed_velocity_m_s'] = None
    vsub_layer['initial_velocity_m_s'] = 5200.0
    req = RefractionStaticApplyRequest.model_validate(payload)
    job_id = 'refraction-vsub-cell-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'error'
    message = str(job['message'])
    assert 'conversion.layer_count=3' in message
    assert 'enabled layer kinds=v2_t1, v3_t2, vsub_t3' in message
    assert 'vsub_t3 velocity_mode=solve_cell is not supported' in message


def test_run_refraction_static_apply_job_rejects_cell_v3_until_publicly_wired(
    client: TestClient,
    tmp_path: Path,
) -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='grid_3d',
        v2_velocity_mode='solve_global',
    )
    payload = _two_layer_apply_payload(fixture)
    payload['model']['layers'][1]['velocity_mode'] = 'solve_cell'
    payload['model']['layers'][1]['initial_velocity_m_s'] = 3600.0
    payload['model']['refractor_cell'] = {
        'number_of_cell_x': 5,
        'size_of_cell_x_m': 500.0,
        'x_coordinate_origin_m': 0.0,
        'number_of_cell_y': 1,
        'size_of_cell_y_m': None,
        'y_coordinate_origin_m': 0.0,
        'assignment_mode': 'midpoint',
        'outside_grid_policy': 'reject',
        'coordinate_mode': 'grid_3d',
        'min_observations_per_cell': 1,
        'velocity_smoothing_weight': 0.0,
        'smoothing_reference_distance_m': None,
    }
    req = RefractionStaticApplyRequest.model_validate(payload)
    job_id = 'refraction-cell-v3-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'error'
    assert 'v3_t2 velocity_mode=solve_cell is not supported' in str(job['message'])
    assert 'cell V3/T2 is available only for internal layer solving' in str(
        job['message']
    )
    assert REQUEST_JSON_NAME in _job_file_names(job_dir)
    assert FINAL_REFRACTION_ARTIFACTS.isdisjoint(_job_file_names(job_dir))


def test_run_refraction_static_apply_job_writes_multilayer_layer_qc(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload()
    payload['model'] = {
        'method': 'multilayer_time_term',
        'first_layer': {
            'mode': 'constant',
            'weathering_velocity_m_s': 800.0,
        },
        'layers': [
            {
                'kind': 'v2_t1',
                'enabled': True,
                'min_offset_m': 0.0,
                'max_offset_m': 240.0,
                'velocity_mode': 'solve_global',
                'initial_velocity_m_s': 2500.0,
                'min_velocity_m_s': 1200.0,
                'max_velocity_m_s': 6000.0,
            },
        ],
    }
    req = RefractionStaticApplyRequest.model_validate(payload)
    job_id = 'refraction-multilayer-qc-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)

    input_model = synthetic_refracted_arrival_input_model()
    expected_layer_masks = build_refraction_layer_observation_masks(
        input_model=input_model,
        model=req.model,
    )
    expected_layer_qc = refraction_layer_observation_qc(expected_layer_masks)
    replacement_result = object()
    datum_result = replace(
        _artifact_result(),
        qc={**_artifact_result().qc, 'layers': expected_layer_qc},
    )
    build_calls: list[dict[str, Any]] = []
    replacement_calls: list[dict[str, Any]] = []

    def _build_input_model(**kwargs: Any) -> object:
        build_calls.append(kwargs)
        return input_model

    def _capture_replacement(**kwargs: Any) -> object:
        replacement_calls.append(kwargs)
        return replacement_result

    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_static_input_model',
        _build_input_model,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'compute_weathering_replacement_statics_from_first_breaks',
        _capture_replacement,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_datum_statics',
        lambda **_kwargs: datum_result,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    assert len(build_calls) == 1
    assert build_calls[0]['req'] is req
    assert len(replacement_calls) == 1
    gated_input_model = replacement_calls[0]['input_model']
    assert gated_input_model is not input_model
    np.testing.assert_array_equal(
        gated_input_model.valid_observation_mask_sorted,
        expected_layer_masks.layer_used_mask_sorted['v2_t1'],
    )
    np.testing.assert_array_equal(
        gated_input_model.rejection_reason_sorted,
        expected_layer_masks.layer_rejection_reason_sorted['v2_t1'],
    )
    assert gated_input_model.qc['active_layer_kind'] == 'v2_t1'
    assert gated_input_model.qc['layers'] == expected_layer_qc
    downstream_req = replacement_calls[0]['req']
    assert downstream_req is not req
    assert downstream_req.model.method == 'gli_variable_thickness'
    assert downstream_req.model.layers is None
    assert downstream_req.model.bedrock_velocity_mode == 'solve_global'
    assert downstream_req.model.initial_bedrock_velocity_m_s == pytest.approx(2500.0)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    assert REFRACTION_STATIC_QC_JSON_NAME in _job_file_names(job_dir)
    request_payload = json.loads(
        (job_dir / REQUEST_JSON_NAME).read_text(encoding='utf-8')
    )
    assert request_payload['request']['model']['method'] == 'multilayer_time_term'
    qc = json.loads(
        (job_dir / REFRACTION_STATIC_QC_JSON_NAME).read_text(encoding='utf-8')
    )
    assert qc['layers'] == expected_layer_qc


def test_run_refraction_static_apply_job_writes_artifacts_and_completes(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    req = RefractionStaticApplyRequest.model_validate(_payload())
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            job_id,
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            statics_kind='refraction',
            artifacts_dir=str(job_dir),
        )
    replacements: list[dict[str, Any]] = []
    datum_builds: list[dict[str, Any]] = []
    artifact_writes: list[dict[str, Any]] = []
    replacement_result = object()
    datum_result = object()

    def _capture_replacement(**kwargs: Any) -> object:
        replacements.append(kwargs)
        return replacement_result

    def _capture_datum_build(**kwargs: Any) -> object:
        datum_builds.append(kwargs)
        return datum_result

    def _capture_artifact_write(**kwargs: Any) -> object:
        artifact_writes.append(kwargs)
        return object()

    monkeypatch.setattr(
        refraction_service_module,
        'compute_weathering_replacement_statics_from_first_breaks',
        _capture_replacement,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_datum_statics',
        _capture_datum_build,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'write_refraction_static_artifacts',
        _capture_artifact_write,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    request_json = job_dir / 'refraction_static_request.json'
    assert request_json.is_file()
    request_payload = json.loads(request_json.read_text(encoding='utf-8'))
    assert request_payload['job_id'] == job_id
    assert request_payload['statics_kind'] == 'refraction'
    assert request_payload['request']['file_id'] == 'raw-file-id'
    assert len(replacements) == 1
    assert replacements[0]['req'] is req
    assert replacements[0]['job_dir'] == job_dir
    replacement_resolved = replacements[0]['resolved_first_layer']
    assert replacement_resolved.mode == 'constant'
    assert replacement_resolved.weathering_velocity_m_s == pytest.approx(800.0)
    assert len(datum_builds) == 1
    assert datum_builds[0]['weathering_replacement_result'] is replacement_result
    assert datum_builds[0]['datum'] is req.datum
    assert datum_builds[0]['apply_options'] is req.apply
    assert datum_builds[0]['job_dir'] == job_dir
    assert datum_builds[0]['state'] is client.app.state.sv
    assert datum_builds[0]['file_id'] == req.file_id
    assert datum_builds[0]['key1_byte'] == req.key1_byte
    assert datum_builds[0]['key2_byte'] == req.key2_byte
    assert datum_builds[0]['resolved_first_layer'] is replacement_resolved
    assert len(artifact_writes) == 1
    assert artifact_writes[0]['result'] is datum_result
    assert artifact_writes[0]['req'] is req
    assert artifact_writes[0]['job_dir'] == job_dir
    resolved_first_layer = artifact_writes[0]['resolved_first_layer']
    assert resolved_first_layer.mode == 'constant'
    assert resolved_first_layer.weathering_velocity_m_s == pytest.approx(800.0)
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    assert job['progress'] == pytest.approx(1.0)
    assert job['message'] == 'refraction_static_artifacts_written_artifact_only'


def test_refraction_static_job_uses_resolved_first_layer_dataclass_downstream(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-v1-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    payload = _payload()
    payload['model']['weathering_velocity_m_s'] = None
    payload['model']['first_layer'] = {
        'mode': 'estimate_direct_arrival',
        'min_weathering_velocity_m_s': 500.0,
        'max_weathering_velocity_m_s': 1200.0,
        'min_direct_offset_m': 20.0,
        'max_direct_offset_m': 140.0,
        'min_picks_per_fit': 5,
        'min_groups': 3,
        'robust_enabled': True,
        'robust_threshold': 3.5,
    }
    payload['moveout']['min_offset_m'] = 240.0
    req = RefractionStaticApplyRequest.model_validate(payload)
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            job_id,
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            statics_kind='refraction',
            artifacts_dir=str(job_dir),
        )

    input_model = object()
    weathering_input_model = object()
    replacement_result = object()
    datum_result = object()
    build_calls: list[RefractionStaticApplyRequest] = []
    replacement_calls: list[dict[str, Any]] = []
    datum_calls: list[dict[str, Any]] = []
    artifact_calls: list[dict[str, Any]] = []
    v1_write_calls: list[dict[str, Any]] = []
    estimate = RefractionV1EstimateResult(
        mode='estimate_direct_arrival',
        resolved_weathering_velocity_m_s=812.5,
        group_kind='source_endpoint',
        group_key=np.asarray(['source:1']),
        group_v1_m_s=np.asarray([812.5]),
        group_slope_s_per_m=np.asarray([1.0 / 812.5]),
        group_intercept_s=np.asarray([0.01]),
        group_n_candidates=np.asarray([6]),
        group_n_used=np.asarray([6]),
        group_offset_min_m=np.asarray([20.0]),
        group_offset_max_m=np.asarray([120.0]),
        group_residual_rms_s=np.asarray([0.0]),
        group_residual_mad_s=np.asarray([0.0]),
        group_status=np.asarray(['ok']),
        qc={
            'v1_mode': 'estimate_direct_arrival',
            'resolved_weathering_velocity_m_s': 812.5,
            'v1_status': 'estimated',
        },
    )

    def _build_input_model(**kwargs: Any) -> object:
        build_req = kwargs['req']
        build_calls.append(build_req)
        if len(build_calls) == 1:
            return input_model
        return weathering_input_model

    def _estimate_v1(**kwargs: Any) -> RefractionV1EstimateResult:
        assert kwargs['input_model'] is input_model
        return estimate

    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_static_input_model',
        _build_input_model,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'estimate_global_v1_from_direct_arrivals',
        _estimate_v1,
    )

    def _capture_v1_write(*args: Any, **kwargs: Any) -> object:
        v1_write_calls.append({'args': args, 'kwargs': kwargs})
        return object()

    def _capture_replacement(**kwargs: Any) -> object:
        replacement_calls.append(kwargs)
        return replacement_result

    def _capture_artifact_write(**kwargs: Any) -> object:
        artifact_calls.append(kwargs)
        return object()

    def _capture_datum_build(**kwargs: Any) -> object:
        datum_calls.append(kwargs)
        return datum_result

    monkeypatch.setattr(
        refraction_service_module,
        'write_refraction_v1_artifacts',
        _capture_v1_write,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'compute_weathering_replacement_statics_from_first_breaks',
        _capture_replacement,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_datum_statics',
        _capture_datum_build,
    )
    monkeypatch.setattr(
        refraction_service_module,
        'write_refraction_static_artifacts',
        _capture_artifact_write,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    assert len(v1_write_calls) == 1
    assert v1_write_calls[0]['args'] == (job_dir, estimate)
    assert len(build_calls) == 2
    assert build_calls[0].moveout.min_offset_m is None
    assert build_calls[1] is req
    assert build_calls[1].moveout.min_offset_m == pytest.approx(240.0)
    assert len(replacement_calls) == 1
    downstream_req = replacement_calls[0]['req']
    assert downstream_req is req
    assert replacement_calls[0]['input_model'] is weathering_input_model
    assert downstream_req.model.first_layer_mode == 'estimate_direct_arrival'
    assert downstream_req.model.first_layer is not None
    assert downstream_req.model.first_layer.weathering_velocity_m_s is None
    with pytest.raises(ValueError, match='requires a resolved weathering velocity'):
        _ = downstream_req.model.resolved_weathering_velocity_m_s
    replacement_resolved = replacement_calls[0]['resolved_first_layer']
    assert replacement_resolved.mode == 'estimate_direct_arrival'
    assert replacement_resolved.weathering_velocity_m_s == pytest.approx(812.5)
    assert len(datum_calls) == 1
    assert datum_calls[0]['weathering_replacement_result'] is replacement_result
    assert datum_calls[0]['resolved_first_layer'] is replacement_resolved
    assert len(artifact_calls) == 1
    assert artifact_calls[0]['req'] is req
    resolved_first_layer = artifact_calls[0]['resolved_first_layer']
    assert resolved_first_layer is replacement_resolved
    assert resolved_first_layer.mode == 'estimate_direct_arrival'
    assert resolved_first_layer.weathering_velocity_m_s == pytest.approx(812.5)
    assert artifact_calls[0]['upstream_artifact_names'] == (
        REFRACTION_V1_QC_JSON_NAME,
        REFRACTION_V1_ESTIMATES_CSV_NAME,
    )


def test_run_refraction_static_apply_job_registers_corrected_file_when_requested(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-job-id'
    job_dir = tmp_path / 'jobs' / job_id
    payload = _payload()
    payload['apply']['register_corrected_file'] = True
    req = RefractionStaticApplyRequest.model_validate(payload)
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            job_id,
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            statics_kind='refraction',
            artifacts_dir=str(job_dir),
        )
    apply_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        refraction_service_module,
        'compute_weathering_replacement_statics_from_first_breaks',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_datum_statics',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'write_refraction_static_artifacts',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'apply_refraction_statics_to_trace_store',
        lambda **kwargs: (
            apply_calls.append(kwargs)
            or SimpleNamespace(
                corrected_file_id='corrected-refraction-file-id',
                corrected_trace_store_path=job_dir / 'corrected-store',
            )
        ),
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert len(apply_calls) == 1
    assert apply_calls[0]['req'] is req
    assert apply_calls[0]['state'] is client.app.state.sv
    assert apply_calls[0]['job_id'] == job_id
    assert apply_calls[0]['job_dir'] == job_dir
    assert job['status'] == 'done'
    assert job['corrected_file_id'] == 'corrected-refraction-file-id'
    assert job['corrected_store_path'] == str(job_dir / 'corrected-store')


def test_run_refraction_static_apply_job_artifact_only_writes_real_final_artifacts(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-artifact-only-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    req = RefractionStaticApplyRequest.model_validate(_payload())
    datum_result = _artifact_result()
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    _stub_upstream_refraction_result(monkeypatch, datum_result=datum_result)

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
        registry_count = len(client.app.state.sv.file_registry.records)
    assert job['status'] == 'done'
    assert job['message'] == 'refraction_static_artifacts_written_artifact_only'
    assert 'corrected_file_id' not in job
    assert registry_count == 0
    assert FINAL_REFRACTION_ARTIFACTS.issubset(_job_file_names(job_dir))
    assert CORRECTED_FILE_JSON_NAME not in _job_file_names(job_dir)
    assert REFRACTION_STATIC_APPLY_QC_JSON_NAME not in _job_file_names(job_dir)
    assert not _segy_output_files(job_dir)

    with np.load(job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        assert data['sorted_trace_index'].shape == (4,)
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
        for key in data.files:
            assert data[key].dtype != object

    qc = json.loads((job_dir / REFRACTION_STATIC_QC_JSON_NAME).read_text(encoding='utf-8'))
    assert qc['workflow'] == 'refraction_statics'
    assert qc['method'] == 'gli_variable_thickness'
    assert qc['sign_convention'] == 'corrected(t) = raw(t - shift_s)'
    assert qc['status_counts']['node_solution_status']['solved'] == 2
    assert qc['status_counts']['node_weathering_status']['zero_thickness'] == 1
    assert qc['status_counts']['trace_static_status']['not_observed'] == 1
    expected_qc_artifacts = FINAL_REFRACTION_ARTIFACTS - {
        REFRACTION_STATIC_ARTIFACTS_JSON_NAME
    }
    assert {item['name'] for item in qc['artifacts']} == expected_qc_artifacts

    assert len(_read_csv(job_dir / REFRACTION_STATICS_CSV_NAME)) == 4
    assert len(_read_csv(job_dir / NEAR_SURFACE_MODEL_CSV_NAME)) == 3
    assert len(_read_csv(job_dir / FIRST_BREAK_RESIDUALS_CSV_NAME)) == 3
    assert len(_read_csv(job_dir / REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME)) == 3
    assert len(_read_csv(job_dir / REFRACTION_STATIC_COMPONENTS_CSV_NAME)) == 4

    files_response = client.get(f'/statics/job/{job_id}/files')
    assert files_response.status_code == 200
    listed = {item['name'] for item in files_response.json()['files']}
    assert FINAL_REFRACTION_ARTIFACTS.issubset(listed)
    assert REQUEST_JSON_NAME in listed
    assert CORRECTED_FILE_JSON_NAME not in listed

    download = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': REFRACTION_STATIC_QC_JSON_NAME},
    )
    assert download.status_code == 200
    assert download.json()['status_counts']['node_weathering_status'][
        'zero_thickness'
    ] == 1

    request_download = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': REQUEST_JSON_NAME},
    )
    assert request_download.status_code == 200
    assert request_download.json()['job_id'] == job_id


def test_run_refraction_static_apply_job_inline_canonical_export_uses_invalid_policy(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-inline-canonical-export'
    job_dir = tmp_path / 'jobs' / job_id
    payload = _payload()
    payload['export'] = {
        'enabled': True,
        'formats': ['canonical_static_table'],
        'fail_on_invalid_static_status': False,
        'include_inactive_endpoints': False,
    }
    req = RefractionStaticApplyRequest.model_validate(payload)
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    _stub_upstream_refraction_result(monkeypatch, datum_result=object())

    def _write_static_table_artifacts(**kwargs: Any) -> object:
        root = Path(kwargs['job_dir'])
        root.mkdir(parents=True, exist_ok=True)
        _write_csv_rows(
            root / SOURCE_STATIC_TABLE_CSV_NAME,
            [
                _source_static_export_row(
                    endpoint_key='source:1001',
                    source_id='1001',
                    total_applied_shift_ms='12.5',
                    static_status='ok',
                ),
                _source_static_export_row(
                    endpoint_key='source:inactive',
                    source_id='1002',
                    total_applied_shift_ms='',
                    static_status='inactive_endpoint',
                ),
            ],
        )
        _write_csv_rows(
            root / RECEIVER_STATIC_TABLE_CSV_NAME,
            [
                _receiver_static_export_row(
                    endpoint_key='receiver:2001',
                    receiver_id='2001',
                    total_applied_shift_ms='-3.25',
                    static_status='ok',
                ),
            ],
        )
        return object()

    monkeypatch.setattr(
        refraction_service_module,
        'write_refraction_static_artifacts',
        _write_static_table_artifacts,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    rows = _read_csv(job_dir / CANONICAL_SOURCE_RECEIVER_STATIC_TABLE_CSV_NAME)
    assert [row['endpoint_key'] for row in rows] == ['source:1001', 'receiver:2001']
    assert rows[0]['applied_shift_ms'] == '12.5'
    assert rows[1]['applied_shift_ms'] == '-3.25'


def test_refraction_static_job_lists_v1_artifacts_when_v1_estimated(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-v1-artifacts-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    payload = _payload()
    _enable_v1_estimation(payload)
    req = RefractionStaticApplyRequest.model_validate(payload)
    datum_result = _artifact_result()
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    _stub_upstream_refraction_result(monkeypatch, datum_result=datum_result)
    _stub_v1_estimation(monkeypatch)

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    listed = _listed_job_files(client, job_id)
    assert REFRACTION_V1_QC_JSON_NAME in listed
    assert REFRACTION_V1_ESTIMATES_CSV_NAME in listed
    manifest = json.loads(
        (job_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    manifest_artifacts = {item['name']: item for item in manifest['artifacts']}
    manifest_names = set(manifest_artifacts)
    assert REFRACTION_V1_QC_JSON_NAME in manifest_names
    assert REFRACTION_V1_ESTIMATES_CSV_NAME in manifest_names
    assert manifest_artifacts[REFRACTION_V1_QC_JSON_NAME]['origin'] == 'upstream'
    assert (
        manifest_artifacts[REFRACTION_V1_ESTIMATES_CSV_NAME]['origin'] == 'upstream'
    )
    qc = json.loads((job_dir / REFRACTION_STATIC_QC_JSON_NAME).read_text())
    assert REFRACTION_V1_QC_JSON_NAME in {item['name'] for item in qc['artifacts']}


def test_refraction_static_download_new_v1_artifacts(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-download-new-v1-artifacts-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    payload = _payload()
    _enable_v1_estimation(payload)
    req = RefractionStaticApplyRequest.model_validate(payload)
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    _stub_upstream_refraction_result(monkeypatch, datum_result=_artifact_result())
    _stub_v1_estimation(monkeypatch)

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    for artifact_name in (
        REFRACTION_V1_QC_JSON_NAME,
        REFRACTION_V1_ESTIMATES_CSV_NAME,
    ):
        response = client.get(
            f'/statics/job/{job_id}/download',
            params={'name': artifact_name},
        )
        assert response.status_code == 200
        assert response.content


def test_refraction_job_lists_field_correction_artifacts_when_enabled(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-field-correction-artifacts-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    payload = _payload()
    payload['field_corrections'] = {
        'source_depth': {
            'mode': 'weathering_velocity_time',
            'source_depth_byte': 115,
        },
        'composition': {'enabled': False},
    }
    req = RefractionStaticApplyRequest.model_validate(payload)
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    _stub_upstream_refraction_result(monkeypatch, datum_result=_artifact_result())

    def _build_input_model(**kwargs: Any) -> object:
        source_depth = resolve_refraction_source_depth(
            source_endpoint_key_sorted=np.asarray(['s0', 's1'], dtype='<U2'),
            source_endpoint_id_sorted=np.asarray([10, 11], dtype=np.int64),
            source_node_id_sorted=np.asarray([0, 1], dtype=np.int64),
            source_depth_m_sorted=np.asarray([4.0, 8.0], dtype=np.float64),
            mode='weathering_velocity_time',
            source_depth_byte=115,
        )
        write_refraction_source_depth_artifacts(kwargs['job_dir'], source_depth)
        return SimpleNamespace(source_depth_result=source_depth)

    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_static_input_model',
        _build_input_model,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done', job.get('message')
    listed = _listed_job_files(client, job_id)
    assert REFRACTION_SOURCE_DEPTH_QC_JSON_NAME in listed
    assert REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME in listed

    manifest = json.loads(
        (job_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    artifacts = {item['name']: item for item in manifest['artifacts']}
    assert artifacts[REFRACTION_SOURCE_DEPTH_QC_JSON_NAME]['origin'] == 'upstream'
    assert (
        artifacts[REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME]['origin'] == 'upstream'
    )

    download = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': REFRACTION_SOURCE_DEPTH_QC_JSON_NAME},
    )
    assert download.status_code == 200
    assert download.json()['sign_convention'] == 'corrected(t) = raw(t - shift_s)'


def test_refraction_static_job_lists_t1lsst_artifact_when_conversion_enabled(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-t1lsst-artifacts-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    payload = _payload()
    payload['conversion'] = {'mode': 't1lsst_1layer'}
    req = RefractionStaticApplyRequest.model_validate(payload)
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    _stub_upstream_refraction_result(monkeypatch, datum_result=_artifact_result())

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    listed = _listed_job_files(client, job_id)
    assert REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME in listed
    manifest = json.loads(
        (job_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    assert REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME in {
        item['name'] for item in manifest['artifacts']
    }
    download = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME},
    )
    assert download.status_code == 200
    assert 'sign_convention' in download.text


def test_refraction_static_job_lists_source_receiver_tables(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-static-table-artifacts-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    req = RefractionStaticApplyRequest.model_validate(_payload())
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    _stub_upstream_refraction_result(monkeypatch, datum_result=_artifact_result())

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    listed = _listed_job_files(client, job_id)
    assert {
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    }.issubset(listed)


def test_refraction_static_job_lists_solve_cell_artifacts(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-solve-cell-artifacts-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    req = synthetic_cell_refraction_apply_request()
    input_model = synthetic_cell_refracted_arrival_input_model()
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    monkeypatch.setattr(
        refraction_inputs_module,
        'build_refraction_static_input_model',
        lambda **_kwargs: input_model,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    listed = _listed_job_files(client, job_id)
    assert {
        REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
        REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
        REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
    }.issubset(listed)
    manifest = json.loads(
        (job_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    manifest_names = {item['name'] for item in manifest['artifacts']}
    assert {
        REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
        REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
        REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
    }.issubset(manifest_names)


def test_refraction_static_download_new_artifacts(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-download-new-artifacts-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    payload = _payload()
    payload['conversion'] = {'mode': 't1lsst_1layer'}
    _enable_v1_estimation(payload)
    req = RefractionStaticApplyRequest.model_validate(payload)
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    _stub_upstream_refraction_result(monkeypatch, datum_result=_artifact_result())
    _stub_v1_estimation(monkeypatch)

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['message'] == 'refraction_static_artifacts_written_artifact_only'

    for artifact_name in (
        REFRACTION_V1_QC_JSON_NAME,
        REFRACTION_V1_ESTIMATES_CSV_NAME,
        REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    ):
        response = client.get(
            f'/statics/job/{job_id}/download',
            params={'name': artifact_name},
        )
        assert response.status_code == 200
        assert response.content


def test_refraction_static_register_corrected_file_lists_new_artifacts(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-corrected-new-artifacts-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    payload = _payload()
    payload['conversion'] = {'mode': 't1lsst_1layer'}
    payload['apply']['register_corrected_file'] = True
    _enable_v1_estimation(payload)
    req = RefractionStaticApplyRequest.model_validate(payload)
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    _stub_upstream_refraction_result(monkeypatch, datum_result=_artifact_result())
    _stub_v1_estimation(monkeypatch)
    monkeypatch.setattr(
        refraction_service_module,
        'apply_refraction_statics_to_trace_store',
        lambda **_kwargs: SimpleNamespace(
            corrected_file_id='corrected-refraction-file-id',
            corrected_trace_store_path=job_dir / 'corrected-store',
        ),
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    assert job['corrected_file_id'] == 'corrected-refraction-file-id'
    listed = _listed_job_files(client, job_id)
    assert {
        REFRACTION_V1_QC_JSON_NAME,
        REFRACTION_V1_ESTIMATES_CSV_NAME,
        REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    }.issubset(listed)


def test_refraction_static_job_writes_v1_t1lsst_and_static_table_artifacts(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-synthetic-one-layer-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    req = synthetic_refraction_apply_request(
        first_layer_mode='estimate_direct_arrival',
        conversion_mode='t1lsst_1layer',
    )
    req = req.model_copy(
        update={
            'model': req.model.model_copy(
                update={
                    'first_layer': req.model.first_layer.model_copy(
                        update={
                            'max_direct_offset_m': 180.0,
                            'min_picks_per_fit': 3,
                            'min_groups': 5,
                        }
                    )
                }
            ),
            'moveout': req.moveout.model_copy(update={'min_offset_m': 240.0}),
        }
    )
    refracted_input = synthetic_refracted_arrival_input_model()
    direct_input = synthetic_direct_arrival_input_model()
    direct_mask = direct_input.distance_m_sorted <= 180.0
    mixed_input = replace(
        refracted_input,
        pick_time_s_sorted=np.where(
            direct_mask,
            direct_input.pick_time_s_sorted,
            refracted_input.pick_time_s_sorted,
        ),
    )
    weathering_input = replace(
        mixed_input,
        valid_observation_mask_sorted=(
            mixed_input.valid_observation_mask_sorted
            & (mixed_input.distance_m_sorted >= 240.0)
        ),
    )
    build_calls: list[RefractionStaticApplyRequest] = []
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)

    def _build_synthetic_input_model(**kwargs: Any) -> object:
        build_req = kwargs['req']
        build_calls.append(build_req)
        if build_req.moveout.min_offset_m is None:
            return mixed_input
        assert build_req.moveout.min_offset_m == pytest.approx(240.0)
        return weathering_input

    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_static_input_model',
        _build_synthetic_input_model,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    assert len(build_calls) == 2
    assert build_calls[0].moveout.min_offset_m is None
    assert build_calls[1].moveout.min_offset_m == pytest.approx(240.0)
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    names = _job_file_names(job_dir)
    assert {
        REFRACTION_V1_QC_JSON_NAME,
        REFRACTION_V1_ESTIMATES_CSV_NAME,
        REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    }.issubset(names)

    manifest = json.loads(
        (job_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(encoding='utf-8')
    )
    manifest_names = {item['name'] for item in manifest['artifacts']}
    assert {
        REFRACTION_V1_QC_JSON_NAME,
        REFRACTION_V1_ESTIMATES_CSV_NAME,
        REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    }.issubset(manifest_names)

    v1_qc = json.loads((job_dir / REFRACTION_V1_QC_JSON_NAME).read_text())
    assert v1_qc['resolved_weathering_velocity_m_s'] == pytest.approx(
        SYNTHETIC_V1_M_S,
        abs=SYNTHETIC_V1_TOLERANCE_M_S,
    )
    with np.load(job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        assert data['v1_mode'].item() == 'estimate_direct_arrival'
        assert data['resolved_weathering_velocity_m_s'].item() == pytest.approx(
            SYNTHETIC_V1_M_S,
            abs=SYNTHETIC_V1_TOLERANCE_M_S,
        )
        assert data['v2_refractor_velocity_m_s'].item() == pytest.approx(
            SYNTHETIC_V2_M_S,
            abs=SYNTHETIC_V2_TOLERANCE_M_S,
        )

    rows = _read_csv(job_dir / REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME)
    assert rows
    for row in rows:
        node_id = int(row['node_id'])
        assert float(row['t1_ms']) == pytest.approx(
            expected_t1_s_for_node(node_id) * 1000.0,
            abs=SYNTHETIC_T1_TOLERANCE_MS,
        )
        assert float(row['sh1_weathering_thickness_m']) == pytest.approx(
            expected_sh1_m_for_node(node_id),
            abs=SYNTHETIC_SH1_TOLERANCE_M,
        )
        assert float(row['weathering_correction_ms']) == pytest.approx(
            expected_wcor_s_for_node(node_id) * 1000.0,
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )
        assert float(row['total_applied_shift_ms']) == pytest.approx(
            float(row['total_static_ms']),
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )

    source_rows = _read_csv(job_dir / SOURCE_STATIC_TABLE_CSV_NAME)
    receiver_rows = _read_csv(job_dir / RECEIVER_STATIC_TABLE_CSV_NAME)
    assert len(source_rows) == 6
    assert len(receiver_rows) == 12
    listed = _listed_job_files(client, job_id)
    assert manifest_names.issubset(listed)


def test_refraction_static_legacy_request_still_works(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-legacy-synthetic-one-layer-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    req = synthetic_refraction_apply_request()
    req = req.model_copy(
        update={'moveout': req.moveout.model_copy(update={'min_offset_m': 240.0})}
    )
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    refracted_input = synthetic_refracted_arrival_input_model()
    direct_input = synthetic_direct_arrival_input_model()
    direct_mask = direct_input.distance_m_sorted <= 180.0
    mixed_input = replace(
        refracted_input,
        pick_time_s_sorted=np.where(
            direct_mask,
            direct_input.pick_time_s_sorted,
            refracted_input.pick_time_s_sorted,
        ),
    )
    weathering_input = replace(
        mixed_input,
        valid_observation_mask_sorted=(
            mixed_input.valid_observation_mask_sorted
            & (mixed_input.distance_m_sorted >= 240.0)
        ),
    )
    build_calls: list[RefractionStaticApplyRequest] = []

    def _build_synthetic_input_model(**kwargs: Any) -> object:
        build_req = kwargs['req']
        build_calls.append(build_req)
        if build_req.moveout.min_offset_m is None:
            return mixed_input
        assert build_req.moveout.min_offset_m == pytest.approx(240.0)
        return weathering_input

    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_static_input_model',
        _build_synthetic_input_model,
    )
    monkeypatch.setattr(
        refraction_inputs_module,
        'build_refraction_static_input_model',
        _build_synthetic_input_model,
    )

    run_refraction_static_apply_job(job_id, req, client.app.state.sv)

    assert len(build_calls) == 1
    assert build_calls[0].moveout.min_offset_m == pytest.approx(240.0)
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
    assert job['status'] == 'done'
    assert REFRACTION_STATIC_SOLUTION_NPZ_NAME in _job_file_names(job_dir)
    assert REFRACTION_V1_QC_JSON_NAME not in _job_file_names(job_dir)
    assert REFRACTION_V1_ESTIMATES_CSV_NAME not in _job_file_names(job_dir)

    with np.load(job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        assert data['v1_mode'].item() == 'constant'
        assert data['v1_weathering_velocity_m_s'].item() == pytest.approx(
            SYNTHETIC_V1_M_S,
            abs=SYNTHETIC_V1_TOLERANCE_M_S,
        )
        assert data['v2_refractor_velocity_m_s'].item() == pytest.approx(
            SYNTHETIC_V2_M_S,
            abs=SYNTHETIC_V2_TOLERANCE_M_S,
        )
        np.testing.assert_allclose(
            data['node_t1_time_s'] * 1000.0,
            [expected_t1_s_for_node(index) * 1000.0 for index in range(12)],
            atol=SYNTHETIC_T1_TOLERANCE_MS,
        )


def test_run_refraction_static_apply_job_writes_real_artifacts_and_corrected_store(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-corrected-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    source_store = tmp_path / 'trace_stores' / 'line001.sgy'
    source_traces = _write_source_store(source_store)
    state = client.app.state.sv
    state.file_registry.update(SOURCE_FILE_ID, store_path=str(source_store), dt=DT)
    payload = _payload()
    payload['apply']['register_corrected_file'] = True
    req = RefractionStaticApplyRequest.model_validate(payload)
    datum_result = _valid_result()
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    _stub_upstream_refraction_result(monkeypatch, datum_result=datum_result)

    run_refraction_static_apply_job(job_id, req, state)

    with state.lock:
        job = dict(state.jobs[job_id])
    corrected_file_id = job.get('corrected_file_id')
    assert job['status'] == 'done'
    assert isinstance(corrected_file_id, str)
    assert corrected_file_id != SOURCE_FILE_ID
    corrected_store = Path(str(job['corrected_store_path']))
    assert state.file_registry.get_store_path(corrected_file_id) == str(corrected_store)
    with state.lock:
        assert trace_store_cache_key(corrected_file_id, KEY1, KEY2) in state.cached_readers

    assert FINAL_REFRACTION_ARTIFACTS.issubset(_job_file_names(job_dir))
    assert CORRECTED_FILE_JSON_NAME in _job_file_names(job_dir)
    assert REFRACTION_STATIC_APPLY_QC_JSON_NAME in _job_file_names(job_dir)
    assert not _segy_output_files(job_dir)
    assert not _segy_output_files(corrected_store)

    corrected = np.load(corrected_store / 'traces.npy')
    assert [int(np.argmax(corrected[index])) for index in range(3)] == [8, 10, 6]
    np.testing.assert_array_equal(np.load(source_store / 'traces.npy'), source_traces)

    corrected_manifest = json.loads(
        (job_dir / CORRECTED_FILE_JSON_NAME).read_text(encoding='utf-8')
    )
    assert corrected_manifest['derived_from_file_id'] == SOURCE_FILE_ID
    assert corrected_manifest['derivation'] == 'refraction_static_correction'
    assert corrected_manifest['statics_kind'] == 'refraction'
    assert corrected_manifest['sign_convention'] == 'corrected(t) = raw(t - shift_s)'
    assert corrected_manifest['solution_artifact'] == REFRACTION_STATIC_SOLUTION_NPZ_NAME

    apply_qc = json.loads(
        (job_dir / REFRACTION_STATIC_APPLY_QC_JSON_NAME).read_text(encoding='utf-8')
    )
    assert apply_qc['corrected_file_id'] == corrected_file_id
    assert apply_qc['n_positive_trace_shifts'] == 1
    assert apply_qc['n_negative_trace_shifts'] == 1

    meta = json.loads((corrected_store / 'meta.json').read_text(encoding='utf-8'))
    assert meta['derived']['from_file_id'] == SOURCE_FILE_ID
    assert meta['derived']['derivation'] == 'refraction_static_correction'
    assert meta['derived']['solution_artifact'] == REFRACTION_STATIC_SOLUTION_NPZ_NAME

    files_response = client.get(f'/statics/job/{job_id}/files')
    assert files_response.status_code == 200
    listed = {item['name'] for item in files_response.json()['files']}
    assert FINAL_REFRACTION_ARTIFACTS.issubset(listed)
    assert CORRECTED_FILE_JSON_NAME in listed
    assert REFRACTION_STATIC_APPLY_QC_JSON_NAME in listed

    download = client.get(
        f'/statics/job/{job_id}/download',
        params={'name': CORRECTED_FILE_JSON_NAME},
    )
    assert download.status_code == 200
    assert download.json()['corrected_file_id'] == corrected_file_id


def test_run_refraction_static_apply_job_apply_failure_keeps_final_artifacts(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'refraction-apply-failure-e2e'
    job_dir = tmp_path / 'jobs' / job_id
    source_store = tmp_path / 'trace_stores' / 'line001.sgy'
    _write_source_store(source_store)
    state = client.app.state.sv
    state.file_registry.update(SOURCE_FILE_ID, store_path=str(source_store), dt=DT)
    payload = _payload()
    payload['apply']['register_corrected_file'] = True
    req = RefractionStaticApplyRequest.model_validate(payload)
    invalid_result = _valid_result(
        shifts=np.asarray([0.0, 0.008, np.nan, 0.0], dtype=np.float64),
        valid_mask=np.asarray([True, True, False, True], dtype=bool),
        statuses=np.asarray(['ok', 'ok', 'not_observed', 'ok'], dtype='<U16'),
    )
    _create_refraction_job(client, job_id=job_id, req=req, job_dir=job_dir)
    _stub_upstream_refraction_result(monkeypatch, datum_result=invalid_result)

    run_refraction_static_apply_job(job_id, req, state)

    with state.lock:
        job = dict(state.jobs[job_id])
        registry_ids = set(state.file_registry.records)
    assert job['status'] == 'error'
    assert 'invalid_trace_shift_count=1' in str(job['message'])
    assert 'not_observed' in str(job['message'])
    assert 'corrected_file_id' not in job
    assert registry_ids == {SOURCE_FILE_ID}
    assert FINAL_REFRACTION_ARTIFACTS.issubset(_job_file_names(job_dir))
    assert CORRECTED_FILE_JSON_NAME not in _job_file_names(job_dir)
    assert REFRACTION_STATIC_APPLY_QC_JSON_NAME not in _job_file_names(job_dir)
    assert list(source_store.parent.glob(f'line001.sgy.statics.refraction.{job_id}*')) == []

    files_response = client.get(f'/statics/job/{job_id}/files')
    assert files_response.status_code == 200
    listed = {item['name'] for item in files_response.json()['files']}
    assert FINAL_REFRACTION_ARTIFACTS.issubset(listed)
    assert CORRECTED_FILE_JSON_NAME not in listed


def _two_layer_apply_payload(fixture: Any) -> dict[str, Any]:
    payload = _payload()
    payload['file_id'] = fixture.input_model.file_id
    payload['model'] = fixture.model.model_dump(mode='json')
    payload['solver'] = {
        'damping': 0.0,
        'min_picks_per_node': 1,
        'max_abs_half_intercept_time_ms': 500.0,
        'robust': {
            'enabled': False,
        },
    }
    payload['datum'] = {'mode': 'none'}
    payload['conversion'] = {
        'mode': 't1lsst_multilayer',
        'layer_count': 2,
    }
    payload['apply']['max_abs_shift_ms'] = 250.0
    payload['apply']['register_corrected_file'] = False
    return payload


def _two_layer_apply_request(fixture: Any) -> RefractionStaticApplyRequest:
    return RefractionStaticApplyRequest.model_validate(
        _two_layer_apply_payload(fixture)
    )


def _three_layer_apply_payload() -> dict[str, Any]:
    payload = _payload()
    layer_ranges = {
        'v2_t1': (0.0, 1000.0),
        'v3_t2': (1000.0, 2000.0),
        'vsub_t3': (2000.0, None),
    }
    v2_min, v2_max = layer_ranges['v2_t1']
    v3_min, v3_max = layer_ranges['v3_t2']
    vsub_min, vsub_max = layer_ranges['vsub_t3']
    payload['model'] = {
        'method': 'multilayer_time_term',
        'first_layer': {
            'mode': 'constant',
            'weathering_velocity_m_s': SYNTHETIC_MULTILAYER_V1_M_S,
        },
        'layers': [
            {
                'kind': 'v2_t1',
                'enabled': True,
                'min_offset_m': v2_min,
                'max_offset_m': v2_max,
                'velocity_mode': 'fixed_global',
                'fixed_velocity_m_s': SYNTHETIC_MULTILAYER_V2_M_S,
                'min_velocity_m_s': 1200.0,
                'max_velocity_m_s': 3200.0,
            },
            {
                'kind': 'v3_t2',
                'enabled': True,
                'min_offset_m': v3_min,
                'max_offset_m': v3_max,
                'velocity_mode': 'fixed_global',
                'fixed_velocity_m_s': SYNTHETIC_MULTILAYER_V3_M_S,
                'min_velocity_m_s': 2600.0,
                'max_velocity_m_s': 4800.0,
            },
            {
                'kind': 'vsub_t3',
                'enabled': True,
                'min_offset_m': vsub_min,
                'max_offset_m': vsub_max,
                'velocity_mode': 'fixed_global',
                'fixed_velocity_m_s': SYNTHETIC_MULTILAYER_VSUB_M_S,
                'min_velocity_m_s': 3600.0,
                'max_velocity_m_s': 7000.0,
            },
        ],
    }
    payload['solver'] = {
        'damping': 0.0,
        'min_picks_per_node': 1,
        'max_abs_half_intercept_time_ms': 500.0,
        'robust': {
            'enabled': False,
        },
    }
    payload['datum'] = {'mode': 'none'}
    payload['conversion'] = {
        'mode': 't1lsst_multilayer',
        'layer_count': 3,
    }
    return payload


def _assert_two_layer_solution_arrays(job_dir: Path) -> None:
    expected_arrays = {
        'source_t2_time_s',
        'source_v3_m_s',
        'source_sh2_weathering_thickness_m',
        'receiver_t2_time_s',
        'receiver_v3_m_s',
        'receiver_sh2_weathering_thickness_m',
    }
    with np.load(job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        assert expected_arrays <= set(data.files)
        for key in expected_arrays:
            assert data[key].dtype != object


def _three_layer_contract_datum_result() -> Any:
    result = _artifact_result()
    node_sh1 = result.node_weathering_thickness_m
    node_sh2 = np.asarray([4.0, 5.0, 6.0], dtype=np.float64)
    node_sh3 = np.asarray([2.0, 3.0, 4.0], dtype=np.float64)
    node_total = node_sh1 + node_sh2 + node_sh3

    source_sh1 = result.source_weathering_thickness_m
    source_sh2 = np.asarray([4.0, 5.0], dtype=np.float64)
    source_sh3 = np.asarray([2.0, 3.0], dtype=np.float64)
    source_total = source_sh1 + source_sh2 + source_sh3
    source_sh2_sorted = np.asarray([4.0, 5.0, 4.0, 5.0], dtype=np.float64)
    source_sh3_sorted = np.asarray([2.0, 3.0, 2.0, 3.0], dtype=np.float64)
    source_total_sorted = (
        result.source_weathering_thickness_m_sorted
        + source_sh2_sorted
        + source_sh3_sorted
    )

    receiver_sh1 = result.receiver_weathering_thickness_m
    receiver_sh2 = np.asarray([5.0, 6.0], dtype=np.float64)
    receiver_sh3 = np.asarray([3.0, 4.0], dtype=np.float64)
    receiver_total = receiver_sh1 + receiver_sh2 + receiver_sh3
    receiver_sh2_sorted = np.asarray([5.0, 6.0, 6.0, 5.0], dtype=np.float64)
    receiver_sh3_sorted = np.asarray([3.0, 4.0, 4.0, 3.0], dtype=np.float64)
    receiver_total_sorted = (
        result.receiver_weathering_thickness_m_sorted
        + receiver_sh2_sorted
        + receiver_sh3_sorted
    )

    return replace(
        result,
        bedrock_velocity_mode='fixed_global',
        bedrock_slowness_s_per_m=1.0 / SYNTHETIC_MULTILAYER_VSUB_M_S,
        bedrock_velocity_m_s=SYNTHETIC_MULTILAYER_VSUB_M_S,
        weathering_velocity_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        replacement_slowness_delta_s_per_m=(
            1.0 / SYNTHETIC_MULTILAYER_VSUB_M_S
            - 1.0 / SYNTHETIC_MULTILAYER_V1_M_S
        ),
        qc={
            **result.qc,
            'method': 'multilayer_time_term',
            'conversion_mode': 't1lsst_multilayer',
            'layer_count': 3,
            'enabled_layer_kinds': ['v2_t1', 'v3_t2', 'vsub_t3'],
        },
        node_weathering_thickness_m=node_total,
        node_refractor_elevation_m=result.node_surface_elevation_m - node_total,
        node_sh1_weathering_thickness_m=node_sh1,
        node_sh2_weathering_thickness_m=node_sh2,
        node_sh3_weathering_thickness_m=node_sh3,
        source_weathering_thickness_m=source_total,
        source_refractor_elevation_m=(
            result.source_surface_elevation_m - source_total
        ),
        source_weathering_thickness_m_sorted=source_total_sorted,
        source_refractor_elevation_m_sorted=(
            result.source_surface_elevation_m_sorted - source_total_sorted
        ),
        source_t2_time_s=np.asarray([0.020, 0.022], dtype=np.float64),
        source_t3_time_s=np.asarray([0.030, 0.033], dtype=np.float64),
        source_v3_m_s=np.full(2, SYNTHETIC_MULTILAYER_V3_M_S, dtype=np.float64),
        source_vsub_m_s=np.full(
            2,
            SYNTHETIC_MULTILAYER_VSUB_M_S,
            dtype=np.float64,
        ),
        source_sh1_weathering_thickness_m=source_sh1,
        source_sh2_weathering_thickness_m=source_sh2,
        source_sh3_weathering_thickness_m=source_sh3,
        receiver_weathering_thickness_m=receiver_total,
        receiver_refractor_elevation_m=(
            result.receiver_surface_elevation_m - receiver_total
        ),
        receiver_weathering_thickness_m_sorted=receiver_total_sorted,
        receiver_refractor_elevation_m_sorted=(
            result.receiver_surface_elevation_m_sorted - receiver_total_sorted
        ),
        receiver_t2_time_s=np.asarray([0.023, 0.025], dtype=np.float64),
        receiver_t3_time_s=np.asarray([0.034, 0.037], dtype=np.float64),
        receiver_v3_m_s=np.full(2, SYNTHETIC_MULTILAYER_V3_M_S, dtype=np.float64),
        receiver_vsub_m_s=np.full(
            2,
            SYNTHETIC_MULTILAYER_VSUB_M_S,
            dtype=np.float64,
        ),
        receiver_sh1_weathering_thickness_m=receiver_sh1,
        receiver_sh2_weathering_thickness_m=receiver_sh2,
        receiver_sh3_weathering_thickness_m=receiver_sh3,
    )


def _assert_three_layer_solution_arrays(job_dir: Path) -> None:
    expected_arrays = {
        'node_sh3_weathering_thickness_m',
        'source_t3_time_s',
        'source_vsub_m_s',
        'source_sh3_weathering_thickness_m',
        'receiver_t3_time_s',
        'receiver_vsub_m_s',
        'receiver_sh3_weathering_thickness_m',
    }
    with np.load(job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME, allow_pickle=False) as data:
        assert expected_arrays <= set(data.files)
        for key in expected_arrays:
            assert data[key].dtype != object
        assert np.count_nonzero(np.isfinite(data['source_t3_time_s'])) > 0
        assert np.count_nonzero(np.isfinite(data['receiver_t3_time_s'])) > 0
        assert (
            np.count_nonzero(
                np.isfinite(data['source_sh3_weathering_thickness_m'])
            )
            > 0
        )
        np.testing.assert_allclose(
            data['source_vsub_m_s'],
            SYNTHETIC_MULTILAYER_VSUB_M_S,
            rtol=1.0e-9,
        )
        np.testing.assert_allclose(
            data['receiver_vsub_m_s'],
            SYNTHETIC_MULTILAYER_VSUB_M_S,
            rtol=1.0e-9,
        )


def _create_refraction_job(
    client: TestClient,
    *,
    job_id: str,
    req: RefractionStaticApplyRequest,
    job_dir: Path,
) -> None:
    with client.app.state.sv.lock:
        client.app.state.sv.jobs.create_static_job(
            job_id,
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            statics_kind='refraction',
            artifacts_dir=str(job_dir),
        )


def _stub_upstream_refraction_result(
    monkeypatch: pytest.MonkeyPatch,
    *,
    datum_result: object,
) -> None:
    monkeypatch.setattr(
        refraction_service_module,
        'compute_weathering_replacement_statics_from_first_breaks',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_datum_statics',
        lambda **_kwargs: datum_result,
    )


def _enable_v1_estimation(payload: dict[str, Any]) -> None:
    payload['model']['weathering_velocity_m_s'] = None
    payload['model']['first_layer'] = {
        'mode': 'estimate_direct_arrival',
        'min_weathering_velocity_m_s': 500.0,
        'max_weathering_velocity_m_s': 1200.0,
        'min_direct_offset_m': 20.0,
        'max_direct_offset_m': 140.0,
        'min_picks_per_fit': 5,
        'min_groups': 1,
        'robust_enabled': True,
        'robust_threshold': 3.5,
    }


def _stub_v1_estimation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    estimate: RefractionV1EstimateResult | None = None,
) -> None:
    if estimate is None:
        estimate = _v1_estimate_result()
    monkeypatch.setattr(
        refraction_service_module,
        'build_refraction_static_input_model',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        refraction_service_module,
        'estimate_global_v1_from_direct_arrivals',
        lambda **_kwargs: estimate,
    )


def _v1_estimate_result() -> RefractionV1EstimateResult:
    return RefractionV1EstimateResult(
        mode='estimate_direct_arrival',
        resolved_weathering_velocity_m_s=800.0,
        group_kind='source_endpoint',
        group_key=np.asarray(['source:1']),
        group_v1_m_s=np.asarray([800.0]),
        group_slope_s_per_m=np.asarray([1.0 / 800.0]),
        group_intercept_s=np.asarray([0.01]),
        group_n_candidates=np.asarray([6]),
        group_n_used=np.asarray([6]),
        group_offset_min_m=np.asarray([20.0]),
        group_offset_max_m=np.asarray([120.0]),
        group_residual_rms_s=np.asarray([0.0]),
        group_residual_mad_s=np.asarray([0.0]),
        group_status=np.asarray(['ok']),
        qc={
            'v1_mode': 'estimate_direct_arrival',
            'resolved_weathering_velocity_m_s': 800.0,
            'n_candidate_picks': 6,
            'n_used_groups': 1,
            'v1_status': 'estimated',
            'warnings': [],
        },
    )


def _job_file_names(job_dir: Path) -> set[str]:
    return {path.name for path in job_dir.iterdir() if path.is_file()}


def _listed_job_files(client: TestClient, job_id: str) -> set[str]:
    response = client.get(f'/statics/job/{job_id}/files')
    assert response.status_code == 200
    return {item['name'] for item in response.json()['files']}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    assert rows
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _source_static_export_row(
    *,
    endpoint_key: str,
    source_id: str,
    total_applied_shift_ms: str,
    static_status: str,
) -> dict[str, str]:
    return {
        'endpoint_kind': 'source',
        'source_endpoint_key': endpoint_key,
        'source_id': source_id,
        'source_node_id': source_id,
        'total_applied_shift_ms': total_applied_shift_ms,
        'static_status': static_status,
        'sign_convention': 'corrected(t) = raw(t - shift_s)',
    }


def _receiver_static_export_row(
    *,
    endpoint_key: str,
    receiver_id: str,
    total_applied_shift_ms: str,
    static_status: str,
) -> dict[str, str]:
    return {
        'endpoint_kind': 'receiver',
        'receiver_endpoint_key': endpoint_key,
        'receiver_id': receiver_id,
        'receiver_node_id': receiver_id,
        'total_applied_shift_ms': total_applied_shift_ms,
        'static_status': static_status,
        'sign_convention': 'corrected(t) = raw(t - shift_s)',
    }


def _segy_output_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob('*')
        if path.is_file() and path.suffix.lower() in {'.sgy', '.segy'}
    ]
