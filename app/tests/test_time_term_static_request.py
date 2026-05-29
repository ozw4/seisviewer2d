from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.api.routers.statics as statics_router_module
from app.api.schemas import TimeTermStaticApplyRequest
from app.main import app
from app.services.geometry_linkage_artifacts import (
    GEOMETRY_LINKAGE_NPZ_NAME,
    GeometryLinkageArtifactMetadata,
    build_geometry_linkage_solution_arrays,
)
from app.services.geometry_linkage_linker import (
    GeometryLinkageOptions,
    build_geometry_linkage,
)
from app.services.geometry_linkage_tables import build_endpoint_geometry_tables
from app.services.geometry_linkage_validation import GeometryLinkageHeaders
from app.services.time_term_static_service import validate_time_term_request

FILE_ID = 'current-file-id'
LINKAGE_JOB_ID = 'linkage-job-id'
KEY1_BYTE = 189
KEY2_BYTE = 193
DT = 0.004
N_TRACES = 5


class _Reader:
    key1_byte = KEY1_BYTE
    key2_byte = KEY2_BYTE

    def __init__(self, *, dt: float = DT, n_traces: int = N_TRACES) -> None:
        self.traces = np.zeros((n_traces, 16), dtype=np.float32)
        self.meta = {'dt': dt, 'n_traces': n_traces}

    def get_n_samples(self) -> int:
        return int(self.traces.shape[-1])


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
        'file_id': FILE_ID,
        'key1_byte': KEY1_BYTE,
        'key2_byte': KEY2_BYTE,
        'pick_source': {
            'kind': 'batch_predicted_npz',
            'job_id': 'batch-job-id',
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
            'job_id': LINKAGE_JOB_ID,
            'artifact_name': GEOMETRY_LINKAGE_NPZ_NAME,
        },
        'velocity': {
            'replacement_velocity_m_s': 2000.0,
            'refractor_velocity_m_s': 4500.0,
            'weathering_velocity_m_s': None,
        },
        'moveout': {
            'model': 'head_wave_linear_offset',
            'distance_source': 'geometry',
            'offset_byte': 37,
            'allow_missing_offset': False,
        },
        'solver': {
            'damping': 0.01,
            'gauge': 'mean_zero',
            'robust': {
                'enabled': True,
                'method': 'mad',
                'threshold': 3.5,
                'max_iterations': 5,
                'min_used_fraction': 0.5,
                'min_used_observations': 1,
            },
        },
        'apply': {
            'register_corrected_file': False,
            'max_abs_shift_ms': 250.0,
            'output_dtype': 'float32',
        },
    }


def _validate(payload: dict[str, Any]) -> TimeTermStaticApplyRequest:
    return TimeTermStaticApplyRequest.model_validate(deepcopy(payload))


def _install_trace_store(client: TestClient, tmp_path: Path, *, dt: float = DT) -> None:
    state = client.app.state.sv
    store = tmp_path / 'trace-store'
    store.mkdir()
    state.file_registry.set_record(
        FILE_ID,
        {
            'store_path': str(store),
            'dt': dt,
        },
    )
    with state.lock:
        state.cached_readers[f'{FILE_ID}_{KEY1_BYTE}_{KEY2_BYTE}'] = _Reader(dt=dt)


def _install_linkage_job(
    client: TestClient,
    tmp_path: Path,
    *,
    payload: dict[str, np.ndarray] | None = None,
) -> Path:
    state = client.app.state.sv
    job_dir = tmp_path / 'linkage-job'
    job_dir.mkdir()
    arrays = _linkage_payload() if payload is None else payload
    with (job_dir / GEOMETRY_LINKAGE_NPZ_NAME).open('wb') as handle:
        np.savez(handle, **arrays)
    with state.lock:
        state.jobs.create_static_job(
            LINKAGE_JOB_ID,
            file_id=FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            statics_kind='geometry_linkage',
            artifacts_dir=str(job_dir),
        )
    return job_dir / GEOMETRY_LINKAGE_NPZ_NAME


def _linkage_payload() -> dict[str, np.ndarray]:
    headers = GeometryLinkageHeaders(
        source_x=np.asarray([0.0, 10.0, 11.0, 30.0, 0.0], dtype=np.float64),
        source_y=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        receiver_x=np.asarray([0.0, 100.0, 100.0, 0.0, 100.0], dtype=np.float64),
        receiver_y=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        coordinate_scalar=np.ones(N_TRACES, dtype=np.int64),
        checked_bytes=(71, 73, 77, 81, 85),
    )
    tables = build_endpoint_geometry_tables(headers)
    linkage = build_geometry_linkage(
        tables,
        GeometryLinkageOptions(
            mode='auto_threshold',
            threshold_m=1.1,
            receiver_location_interval_m=25.0,
        ),
    )
    return build_geometry_linkage_solution_arrays(
        tables,
        linkage,
        metadata=GeometryLinkageArtifactMetadata(
            job_id=LINKAGE_JOB_ID,
            input_file_id=FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            source_x_byte=73,
            source_y_byte=77,
            receiver_x_byte=81,
            receiver_y_byte=85,
            coordinate_scalar_byte=71,
        ),
    )


def test_time_term_request_accepts_minimal_valid_schema() -> None:
    payload = {
        'file_id': FILE_ID,
        'pick_source': {
            'kind': 'batch_predicted_npz',
            'job_id': 'batch-job-id',
        },
        'linkage': {
            'mode': 'required',
            'job_id': LINKAGE_JOB_ID,
        },
        'velocity': {
            'replacement_velocity_m_s': 2000.0,
            'refractor_velocity_m_s': 4500.0,
        },
    }

    req = _validate(payload)

    assert req.pick_source.artifact_name == 'predicted_picks_time_s.npz'
    assert req.linkage.artifact_name == GEOMETRY_LINKAGE_NPZ_NAME
    assert req.geometry.coordinate_unit == 'm'
    assert req.moveout.model == 'head_wave_linear_offset'
    assert req.solver.robust.max_iterations == 5
    assert req.apply.register_corrected_file is False


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('replacement_velocity_m_s', 0.0),
        ('refractor_velocity_m_s', 0.0),
        ('weathering_velocity_m_s', 0.0),
    ],
)
def test_time_term_request_rejects_non_positive_velocities(
    field: str,
    value: float,
) -> None:
    payload = _payload()
    payload['velocity'][field] = value

    with pytest.raises(ValidationError, match=field):
        _validate(payload)


def test_time_term_request_rejects_refractor_velocity_not_greater_than_replacement_velocity() -> None:
    payload = _payload()
    payload['velocity']['refractor_velocity_m_s'] = 1500.0

    with pytest.raises(ValidationError, match='refractor_velocity_m_s'):
        _validate(payload)


def test_time_term_request_rejects_invalid_units() -> None:
    payload = _payload()
    payload['geometry']['coordinate_unit'] = 'km'

    with pytest.raises(ValidationError, match='coordinate_unit'):
        _validate(payload)


def test_time_term_request_rejects_invalid_moveout_offset_config() -> None:
    payload = _payload()
    payload['moveout']['distance_source'] = 'offset_header'
    payload['moveout']['offset_byte'] = None

    with pytest.raises(ValidationError, match='moveout.offset_byte is required'):
        _validate(payload)


def test_time_term_request_accepts_reciprocal_head_wave_model() -> None:
    payload = _payload()
    payload['moveout']['model'] = 'reciprocal_head_wave'

    req = _validate(payload)

    assert req.moveout.model == 'reciprocal_head_wave'


def test_time_term_request_accepts_distance_source_geometry_without_offset_byte() -> None:
    payload = _payload()
    payload['moveout']['distance_source'] = 'geometry'
    payload['moveout']['offset_byte'] = None

    req = _validate(payload)

    assert req.moveout.distance_source == 'geometry'
    assert req.moveout.offset_byte is None


def test_time_term_request_accepts_distance_source_offset_header_with_offset_byte() -> None:
    payload = _payload()
    payload['moveout']['distance_source'] = 'offset_header'
    payload['moveout']['offset_byte'] = 37

    req = _validate(payload)

    assert req.moveout.distance_source == 'offset_header'
    assert req.moveout.offset_byte == 37


def test_time_term_request_accepts_distance_source_auto_without_offset_byte() -> None:
    payload = _payload()
    payload['moveout']['distance_source'] = 'auto'
    payload['moveout']['offset_byte'] = None

    req = _validate(payload)

    assert req.moveout.distance_source == 'auto'
    assert req.moveout.offset_byte is None


def test_time_term_request_accepts_max_geometry_offset_mismatch() -> None:
    payload = _payload()
    payload['moveout']['max_geometry_offset_mismatch_m'] = 2.5

    req = _validate(payload)

    assert req.moveout.max_geometry_offset_mismatch_m == pytest.approx(2.5)


def test_time_term_linkage_required_rejects_missing_job() -> None:
    payload = _payload()
    del payload['linkage']['job_id']

    with pytest.raises(ValidationError, match='linkage.job_id is required'):
        _validate(payload)


@pytest.mark.parametrize(
    ('field_path', 'value', 'match'),
    [
        (('solver', 'damping'), -0.1, 'solver.damping'),
        (('solver', 'robust', 'max_iterations'), 0, 'max_iterations'),
        (('solver', 'robust', 'threshold'), 0.0, 'threshold'),
        (('solver', 'robust', 'min_used_fraction'), 1.1, 'min_used_fraction'),
        (('solver', 'robust', 'min_used_observations'), 0, 'min_used_observations'),
        (('apply', 'max_abs_shift_ms'), 0.0, 'max_abs_shift_ms'),
    ],
)
def test_time_term_request_rejects_invalid_solver_and_apply_values(
    field_path: tuple[str, ...],
    value: object,
    match: str,
) -> None:
    payload = _payload()
    target = payload
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value

    with pytest.raises(ValidationError, match=match):
        _validate(payload)


@pytest.mark.parametrize('method', ['mad', 'sigma'])
def test_time_term_request_accepts_robust_method(method: str) -> None:
    payload = _payload()
    payload['solver']['robust']['method'] = method

    req = _validate(payload)

    assert req.solver.robust.method == method
    assert req.solver.robust.threshold == pytest.approx(3.5)


def test_time_term_request_accepts_reference_node_gauge() -> None:
    payload = _payload()
    payload['solver']['gauge'] = 'reference_node'
    payload['solver']['reference_node_id'] = 0

    req = _validate(payload)

    assert req.solver.gauge == 'reference_node'
    assert req.solver.reference_node_id == 0


def test_time_term_request_rejects_reference_node_gauge_without_reference_node_id() -> None:
    payload = _payload()
    payload['solver']['gauge'] = 'reference_node'

    with pytest.raises(ValidationError, match='reference_node_id'):
        _validate(payload)


@pytest.mark.parametrize('reference_node_id', [-1, True, 1.5, '0'])
def test_time_term_request_rejects_invalid_reference_node_id(
    reference_node_id: object,
) -> None:
    payload = _payload()
    payload['solver']['gauge'] = 'reference_node'
    payload['solver']['reference_node_id'] = reference_node_id

    with pytest.raises(ValidationError, match='reference_node_id'):
        _validate(payload)


def test_time_term_request_rejects_reference_node_id_without_reference_gauge() -> None:
    payload = _payload()
    payload['solver']['reference_node_id'] = 0

    with pytest.raises(ValidationError, match='reference_node_id'):
        _validate(payload)


def test_time_term_request_rejects_unknown_robust_method() -> None:
    payload = _payload()
    payload['solver']['robust']['method'] = 'median'

    with pytest.raises(ValidationError, match='method'):
        _validate(payload)


def test_validate_time_term_request_accepts_valid_preconditions(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _install_trace_store(client, tmp_path)
    linkage_path = _install_linkage_job(client, tmp_path)

    result = validate_time_term_request(
        _validate(_payload()),
        state=client.app.state.sv,
    )

    assert result.file_id == FILE_ID
    assert result.dt == pytest.approx(DT)
    assert result.n_traces == N_TRACES
    assert result.linkage_artifact_path == linkage_path


def test_validate_time_term_request_accepts_optional_linkage_without_artifact(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _install_trace_store(client, tmp_path)
    payload = _payload()
    payload['linkage'] = {'mode': 'optional'}

    result = validate_time_term_request(
        _validate(payload),
        state=client.app.state.sv,
    )

    assert result.linkage_artifact_path is None


def test_validate_time_term_request_rejects_unknown_file_id(client: TestClient) -> None:
    with pytest.raises(ValueError, match='file_id not found'):
        validate_time_term_request(_validate(_payload()), state=client.app.state.sv)


def test_time_term_linkage_rejects_missing_artifact(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _install_trace_store(client, tmp_path)
    job_dir = tmp_path / 'linkage-job'
    job_dir.mkdir()
    state = client.app.state.sv
    with state.lock:
        state.jobs.create_static_job(
            LINKAGE_JOB_ID,
            file_id=FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            statics_kind='geometry_linkage',
            artifacts_dir=str(job_dir),
        )

    with pytest.raises(ValueError, match='job artifact not found'):
        validate_time_term_request(_validate(_payload()), state=state)


def test_time_term_linkage_rejects_shape_mismatch(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _install_trace_store(client, tmp_path)
    payload = _linkage_payload()
    payload['source_node_id_sorted'] = payload['source_node_id_sorted'][:-1]
    _install_linkage_job(client, tmp_path, payload=payload)

    with pytest.raises(ValueError, match='source_node_id_sorted shape mismatch'):
        validate_time_term_request(
            _validate(_payload()),
            state=client.app.state.sv,
        )


def test_time_term_linkage_rejects_negative_node_ids(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _install_trace_store(client, tmp_path)
    payload = _linkage_payload()
    source_nodes = payload['source_node_id_sorted'].copy()
    source_nodes[0] = -1
    payload['source_node_id_sorted'] = source_nodes
    _install_linkage_job(client, tmp_path, payload=payload)

    with pytest.raises(ValueError, match='source_node_id_sorted.*negative'):
        validate_time_term_request(
            _validate(_payload()),
            state=client.app.state.sv,
        )


def test_time_term_linkage_rejects_non_real_numeric_node_ids(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _install_trace_store(client, tmp_path)
    payload = _linkage_payload()
    payload['source_node_id_sorted'] = payload['source_node_id_sorted'].astype('<U8')
    _install_linkage_job(client, tmp_path, payload=payload)

    with pytest.raises(ValueError, match='source_node_id_sorted.*real numeric dtype'):
        validate_time_term_request(
            _validate(_payload()),
            state=client.app.state.sv,
        )


def test_time_term_apply_endpoint_creates_async_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(
        statics_router_module,
        'start_job_thread',
        lambda **kwargs: started.append(kwargs),
    )
    _install_trace_store(client, tmp_path)
    _install_linkage_job(client, tmp_path)

    response = client.post('/statics/time-term/apply', json=_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload['state'] == 'queued'
    assert len(started) == 1
    assert started[0]['target'] is statics_router_module.run_time_term_static_apply_job
    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[payload['job_id']])
    assert job['job_type'] == 'statics'
    assert job['statics_kind'] == 'time_term'


def test_time_term_apply_endpoint_rejects_invalid_schema_without_starting_job(
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

    response = client.post('/statics/time-term/apply', json=payload)

    assert response.status_code == 422
    assert started == []
