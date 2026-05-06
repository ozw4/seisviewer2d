from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.routers.statics as statics_router_module
import app.services.geometry_linkage_service as linkage_service
from app.api.schemas import StaticLinkageBuildRequest
from app.main import app
from app.services.geometry_linkage_artifacts import (
    GEOMETRY_LINKAGE_CSV_NAME,
    GEOMETRY_LINKAGE_NPZ_NAME,
    GEOMETRY_LINKAGE_QC_JSON_NAME,
)
from app.services.geometry_linkage_loader import load_geometry_linkage_from_job_dir

FILE_ID = 'source-file-id'
KEY1_BYTE = 189
KEY2_BYTE = 193
DT = 0.004


class _Reader:
    key1_byte = KEY1_BYTE
    key2_byte = KEY2_BYTE

    def __init__(self, *, missing_header: int | None = None) -> None:
        self.traces = np.zeros((4, 8), dtype=np.float32)
        self.meta = {'dt': DT, 'original_segy_path': '/data/source.sgy'}
        self._missing_header = missing_header
        self.headers = {
            71: np.ones(4, dtype=np.int16),
            73: np.asarray([0, 0, 25, 25], dtype=np.int32),
            77: np.asarray([0, 0, 0, 0], dtype=np.int32),
            81: np.asarray([0, 10, 25, 40], dtype=np.int32),
            85: np.asarray([0, 0, 0, 0], dtype=np.int32),
        }

    def ensure_header(self, byte: int) -> np.ndarray:
        if self._missing_header == int(byte):
            raise KeyError(byte)
        return self.headers[int(byte)]


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


def _payload(*, mode: str = 'auto_threshold') -> dict[str, Any]:
    linkage: dict[str, Any]
    if mode == 'none':
        linkage = {'mode': 'none', 'prefer_receiver_anchor': False}
    else:
        linkage = {
            'mode': 'auto_threshold',
            'threshold_m': 20.0,
            'receiver_location_interval_m': 25.0,
            'prefer_receiver_anchor': True,
        }
    return {
        'file_id': FILE_ID,
        'key1_byte': KEY1_BYTE,
        'key2_byte': KEY2_BYTE,
        'geometry': {
            'source_x_byte': 73,
            'source_y_byte': 77,
            'receiver_x_byte': 81,
            'receiver_y_byte': 85,
            'coordinate_scalar_byte': 71,
        },
        'linkage': linkage,
    }


def _request(*, mode: str = 'auto_threshold') -> StaticLinkageBuildRequest:
    return StaticLinkageBuildRequest.model_validate(_payload(mode=mode))


def _run_job_sync(**kwargs: Any) -> object:
    target = kwargs['target']
    args = kwargs.get('args', ())
    extra_kwargs = kwargs.get('kwargs') or {}
    target(*args, **extra_kwargs)
    return object()


def _install_reader(
    client: TestClient,
    tmp_path: Path,
    *,
    missing_header: int | None = None,
) -> None:
    store_dir = tmp_path / 'trace-store'
    store_dir.mkdir()
    state = client.app.state.sv
    state.file_registry.update(FILE_ID, store_path=store_dir, dt=DT)
    with state.lock:
        state.cached_readers[f'{FILE_ID}_{KEY1_BYTE}_{KEY2_BYTE}'] = _Reader(
            missing_header=missing_header,
        )


def test_static_linkage_build_endpoint_creates_geometry_linkage_job(
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

    response = client.post('/statics/linkage/build', json=_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload['state'] == 'queued'
    assert isinstance(payload['job_id'], str)
    assert len(started) == 1
    assert (
        started[0]['target']
        is statics_router_module.run_geometry_linkage_build_job
    )

    state = client.app.state.sv
    with state.lock:
        job = dict(state.jobs[payload['job_id']])

    assert job['status'] == 'queued'
    assert job['job_type'] == 'statics'
    assert job['statics_kind'] == 'geometry_linkage'
    assert job['file_id'] == FILE_ID
    assert job['key1_byte'] == KEY1_BYTE
    assert job['key2_byte'] == KEY2_BYTE


@pytest.mark.parametrize('mode', ['none', 'auto_threshold'])
def test_static_linkage_build_job_writes_expected_artifacts(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    _install_reader(client, tmp_path)
    monkeypatch.setattr(statics_router_module, 'start_job_thread', _run_job_sync)

    response = client.post('/statics/linkage/build', json=_payload(mode=mode))

    assert response.status_code == 200
    job_id = response.json()['job_id']
    status_response = client.get(f'/statics/job/{job_id}/status')
    files_response = client.get(f'/statics/job/{job_id}/files')

    assert status_response.json() == {
        'state': 'done',
        'progress': 1.0,
        'message': '',
    }
    assert [item['name'] for item in files_response.json()['files']] == [
        GEOMETRY_LINKAGE_CSV_NAME,
        GEOMETRY_LINKAGE_NPZ_NAME,
        GEOMETRY_LINKAGE_QC_JSON_NAME,
        'job_meta.json',
    ]

    with client.app.state.sv.lock:
        job = dict(client.app.state.sv.jobs[job_id])
        registry_keys = set(client.app.state.sv.file_registry.records)
    job_dir = Path(str(job['artifacts_dir']))
    loaded = load_geometry_linkage_from_job_dir(
        job_dir,
        expected_n_traces=4,
        expected_key1_byte=KEY1_BYTE,
        expected_key2_byte=KEY2_BYTE,
    )
    meta = json.loads((job_dir / 'job_meta.json').read_text(encoding='utf-8'))

    assert loaded.mode == mode
    assert meta['job_type'] == 'statics'
    assert meta['statics_kind'] == 'geometry_linkage'
    assert meta['source_file_id'] == FILE_ID
    assert meta['request']['linkage']['mode'] == mode
    assert meta['inputs']['geometry']['source_x_byte'] == 73
    assert meta['artifacts'] == {
        'linkage_npz': GEOMETRY_LINKAGE_NPZ_NAME,
        'linkage_csv': GEOMETRY_LINKAGE_CSV_NAME,
        'qc_json': GEOMETRY_LINKAGE_QC_JSON_NAME,
    }
    assert not (job_dir / 'corrected_file.json').exists()
    assert 'corrected_file_id' not in job
    assert registry_keys == {FILE_ID}


def test_static_linkage_build_job_downloads_artifacts(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_reader(client, tmp_path)
    monkeypatch.setattr(statics_router_module, 'start_job_thread', _run_job_sync)
    response = client.post('/statics/linkage/build', json=_payload())
    job_id = response.json()['job_id']

    for name in (
        GEOMETRY_LINKAGE_NPZ_NAME,
        GEOMETRY_LINKAGE_CSV_NAME,
        GEOMETRY_LINKAGE_QC_JSON_NAME,
        'job_meta.json',
    ):
        download_response = client.get(
            f'/statics/job/{job_id}/download',
            params={'name': name},
        )
        assert download_response.status_code == 200
        assert download_response.content


def test_static_linkage_build_job_self_validates_npz(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_reader(client, tmp_path)
    calls: list[dict[str, Any]] = []
    original_loader = linkage_service.load_geometry_linkage_from_job_dir

    def _recording_loader(*args: Any, **kwargs: Any) -> object:
        calls.append({'args': args, 'kwargs': kwargs})
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(
        linkage_service,
        'load_geometry_linkage_from_job_dir',
        _recording_loader,
    )
    monkeypatch.setattr(statics_router_module, 'start_job_thread', _run_job_sync)

    response = client.post('/statics/linkage/build', json=_payload())

    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0]['kwargs'] == {
        'expected_n_traces': 4,
        'expected_key1_byte': KEY1_BYTE,
        'expected_key2_byte': KEY2_BYTE,
    }


def test_static_linkage_build_endpoint_rejects_invalid_schema_with_422(
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
    payload['linkage']['threshold_m'] = 0.0

    response = client.post('/statics/linkage/build', json=payload)

    assert response.status_code == 422
    assert started == []


def test_static_linkage_build_job_invalid_file_id_becomes_error_status(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(statics_router_module, 'start_job_thread', _run_job_sync)

    response = client.post('/statics/linkage/build', json=_payload())
    job_id = response.json()['job_id']
    status_response = client.get(f'/statics/job/{job_id}/status')

    assert response.status_code == 200
    assert status_response.json()['state'] == 'error'
    assert status_response.json()['message'] == 'File ID not found'


def test_static_linkage_build_job_header_validation_error_becomes_error_status(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_reader(client, tmp_path, missing_header=73)
    monkeypatch.setattr(statics_router_module, 'start_job_thread', _run_job_sync)

    response = client.post('/statics/linkage/build', json=_payload())
    job_id = response.json()['job_id']
    status_response = client.get(f'/statics/job/{job_id}/status')

    assert status_response.json()['state'] == 'error'
    assert 'failed to read geometry linkage header byte 73' in (
        status_response.json()['message']
    )


def test_static_linkage_build_job_cancel_before_artifact_write(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_reader(client, tmp_path)
    job_id = 'geometry-linkage-job-id'
    job_dir = tmp_path / 'linkage-job'
    state = client.app.state.sv
    with state.lock:
        state.jobs.create_static_job(
            job_id,
            file_id=FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            statics_kind='geometry_linkage',
            artifacts_dir=str(job_dir),
        )

    original_builder = linkage_service.build_geometry_linkage

    def _build_and_cancel(*args: Any, **kwargs: Any) -> object:
        result = original_builder(*args, **kwargs)
        with state.lock:
            state.jobs.request_cancel(job_id)
        return result

    monkeypatch.setattr(linkage_service, 'build_geometry_linkage', _build_and_cancel)

    linkage_service.run_geometry_linkage_build_job(job_id, _request(), state)

    with state.lock:
        job = dict(state.jobs[job_id])
    assert job['status'] == 'cancelled'
    assert (job_dir / 'job_meta.json').is_file()
    assert not (job_dir / GEOMETRY_LINKAGE_NPZ_NAME).exists()
    assert not (job_dir / GEOMETRY_LINKAGE_CSV_NAME).exists()
    assert not (job_dir / GEOMETRY_LINKAGE_QC_JSON_NAME).exists()
