from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
from app.api.schemas import RefractionStaticApplyRequest
from app.main import app
from app.statics.refraction.adapters.seisviewer2d.runtime import (
    SeisViewer2DRefractionRuntime,
)
from app.statics.refraction.artifacts import (
    RECEIVER_STATIC_TABLE_CSV_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)
from app.statics.refraction.application.design_matrix import (
    build_refraction_static_design_matrix,
)
from app.statics.refraction.application.input_model import build_refraction_static_input_model
from app.tests.fixtures.refraction_static_ui_fixture import (
    FILE_ID,
    build_ui_fixture_trace_store,
    ui_fixture_pick_npz_bytes,
    ui_fixture_static_correction_payload,
)


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


def test_ui_fixture_direct_npz_fixed_global_static_correction_succeeds(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, fixture, store_dir = build_ui_fixture_trace_store(tmp_path / 'ui-fixture')
    pick_npz = ui_fixture_pick_npz_bytes(config, fixture)
    payload = ui_fixture_static_correction_payload()
    request_json = json.dumps(payload)
    state = client.app.state.sv
    state.file_registry.update(FILE_ID, store_path=str(store_dir), dt=config.dt_s)

    validate_response = client.post(
        '/statics/refraction/validate-with-picks',
        data={'request_json': request_json},
        files={
            'pick_npz': (
                'predicted_picks_time_s.npz',
                pick_npz,
                'application/octet-stream',
            )
        },
    )

    assert validate_response.status_code == 200, validate_response.text
    preflight = validate_response.json()
    assert preflight['status'] == 'ok'
    diagnostics = preflight['diagnostics']
    assert diagnostics['n_used_for_inversion'] > 0

    pick_npz_path = tmp_path / 'predicted_picks_time_s.npz'
    pick_npz_path.write_bytes(pick_npz)
    req = RefractionStaticApplyRequest.model_validate(payload)
    input_model = build_refraction_static_input_model(
        req=req,
        runtime=SeisViewer2DRefractionRuntime(state),
        uploaded_pick_npz_path=pick_npz_path,
        uploaded_pick_metadata={
            'original_filename': 'predicted_picks_time_s.npz',
            'stored_name': 'uploaded_picks_time_s.npz',
        },
    )
    design = build_refraction_static_design_matrix(
        input_model=input_model,
        model=req.model,
    )
    used = input_model.valid_observation_mask_sorted
    active_nodes = set(design.active_node_id.tolist())
    n_active_source_nodes = _active_endpoint_node_count(
        input_model.source_node_id_sorted[used],
        active_nodes,
    )
    n_active_receiver_nodes = _active_endpoint_node_count(
        input_model.receiver_node_id_sorted[used],
        active_nodes,
    )
    col_abs_sum = np.asarray(np.abs(design.matrix).sum(axis=0)).ravel()
    n_all_zero_active_node_columns = int(
        np.count_nonzero(col_abs_sum[: design.n_active_nodes] == 0.0)
    )

    assert n_active_source_nodes > 0
    assert n_active_receiver_nodes > 0
    assert n_all_zero_active_node_columns == 0

    def _run_synchronously(**kwargs: Any) -> None:
        kwargs['target'](
            *kwargs.get('args', ()),
            **(kwargs.get('kwargs') or {}),
        )

    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        _run_synchronously,
    )

    apply_response = client.post(
        '/statics/refraction/apply-with-picks',
        data={'request_json': request_json},
        files={
            'pick_npz': (
                'predicted_picks_time_s.npz',
                pick_npz,
                'application/octet-stream',
            )
        },
    )

    assert apply_response.status_code == 200, apply_response.text
    job_id = apply_response.json()['job_id']
    with state.lock:
        job = dict(state.jobs[job_id])
    assert job['status'] == 'done', job.get('message')
    assert 'all-zero active-node column' not in str(job.get('message', ''))

    job_dir = Path(str(job['artifacts_dir']))
    assert (job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME).is_file()
    assert (job_dir / SOURCE_STATIC_TABLE_CSV_NAME).is_file()
    assert (job_dir / RECEIVER_STATIC_TABLE_CSV_NAME).is_file()

    qc_response = client.post(
        '/statics/refraction/qc',
        json={'job_id': job_id, 'max_points': 2000},
    )
    assert qc_response.status_code == 200, qc_response.text
    qc_payload = qc_response.json()
    assert qc_payload['job_id'] == job_id
    assert qc_payload['summary']['workflow'] == 'refraction_statics'


def _active_endpoint_node_count(
    endpoint_node_ids: np.ndarray,
    active_nodes: set[int],
) -> int:
    return len({int(node_id) for node_id in endpoint_node_ids if int(node_id) in active_nodes})
