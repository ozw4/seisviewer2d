from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import app.services.datum_static_service as service
from app.api.schemas import DatumStaticApplyRequest
from app.core.state import AppState, create_app_state
from app.services.datum_static_service import run_datum_static_apply_job
from app.services.trace_store_registration import trace_store_cache_key

KEY1 = 189
KEY2 = 193
SOURCE_ELEVATION = 45
RECEIVER_ELEVATION = 41
ELEVATION_SCALAR = 69
SOURCE_STATIC = 99
RECEIVER_STATIC = 101
TOTAL_STATIC = 103
SOURCE_FILE_ID = 'source-file-id'


def _write_source_store(
    store: Path,
    *,
    traces: np.ndarray | None = None,
    source_elevation_m: float = 100.0,
    receiver_elevation_m: float = 100.0,
    existing_static_value: int = 0,
    sorted_to_original: np.ndarray | None = None,
    dt: float = 0.004,
) -> np.ndarray:
    store.mkdir(parents=True, exist_ok=True)
    if traces is None:
        traces = np.zeros((4, 96), dtype=np.float32)
        traces[:, 40] = 1.0
    traces = np.asarray(traces, dtype=np.float32)
    n_traces, n_samples = traces.shape
    np.save(store / 'traces.npy', traces)

    if sorted_to_original is None:
        sorted_to_original = np.asarray([2, 0, 3, 1], dtype=np.int64)
    np.savez(
        store / 'index.npz',
        key1_values=np.asarray([100], dtype=np.int64),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([n_traces], dtype=np.int64),
        sorted_to_original=np.asarray(sorted_to_original, dtype=np.int64),
    )

    headers = {
        KEY1: np.full(n_traces, 100, dtype=np.int32),
        KEY2: np.arange(10, 10 + n_traces, dtype=np.int32),
        SOURCE_ELEVATION: np.full(n_traces, source_elevation_m, dtype=np.int32),
        RECEIVER_ELEVATION: np.full(n_traces, receiver_elevation_m, dtype=np.int32),
        ELEVATION_SCALAR: np.ones(n_traces, dtype=np.int16),
        SOURCE_STATIC: np.full(n_traces, existing_static_value, dtype=np.int16),
        RECEIVER_STATIC: np.zeros(n_traces, dtype=np.int16),
        TOTAL_STATIC: np.zeros(n_traces, dtype=np.int16),
    }
    for byte, values in headers.items():
        np.save(store / f'headers_byte_{byte}.npy', values)

    meta = {
        'schema_version': 1,
        'dtype': 'float32',
        'n_traces': int(n_traces),
        'n_samples': int(n_samples),
        'key_bytes': {'key1': KEY1, 'key2': KEY2},
        'sorted_by': ['key1', 'key2'],
        'dt': dt,
        'original_segy_path': None,
        'source_sha256': None,
        'original_name': 'line001.sgy',
    }
    (store / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    return traces


def _state_with_source_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **source_kwargs: Any,
) -> tuple[AppState, Path, Path]:
    trace_dir = tmp_path / 'trace_stores'
    jobs_dir = tmp_path / 'jobs'
    monkeypatch.setenv('SV_TRACE_DIR', str(trace_dir))
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(jobs_dir))

    source_store = trace_dir / 'line001.sgy'
    _write_source_store(source_store, **source_kwargs)

    state = create_app_state()
    state.file_registry.update(
        SOURCE_FILE_ID,
        store_path=str(source_store),
        dt=0.004,
    )
    return state, source_store, trace_dir


def _request(**overrides: Any) -> DatumStaticApplyRequest:
    payload: dict[str, Any] = {
        'file_id': SOURCE_FILE_ID,
        'key1_byte': KEY1,
        'key2_byte': KEY2,
        'datum': {
            'mode': 'constant',
            'elevation_m': 0.0,
            'replacement_velocity_m_s': 2000.0,
        },
        'apply': {
            'interpolation': 'linear',
            'fill_value': 0.0,
            'max_abs_shift_ms': 250.0,
            'output_dtype': 'float32',
            'register_corrected_file': True,
        },
    }
    payload.update(overrides)
    return DatumStaticApplyRequest(**payload)


def _create_job(
    state: AppState,
    tmp_path: Path,
    *,
    job_id: str,
    req: DatumStaticApplyRequest,
) -> Path:
    job_dir = tmp_path / 'jobs' / job_id
    with state.lock:
        state.jobs.create_static_job(
            job_id,
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            statics_kind='datum',
            artifacts_dir=str(job_dir),
        )
    return job_dir


def _run_job(
    state: AppState,
    tmp_path: Path,
    *,
    job_id: str = '11111111-2222-3333-4444-555555555555',
    req: DatumStaticApplyRequest | None = None,
) -> tuple[DatumStaticApplyRequest, Path]:
    if req is None:
        req = _request()
    job_dir = _create_job(state, tmp_path, job_id=job_id, req=req)
    run_datum_static_apply_job(job_id, req, state)
    return req, job_dir


def test_datum_static_apply_job_success_builds_and_registers_corrected_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, source_store, _trace_dir = _state_with_source_store(tmp_path, monkeypatch)
    req, job_dir = _run_job(state, tmp_path)

    with state.lock:
        job = dict(state.jobs['11111111-2222-3333-4444-555555555555'])

    assert job['status'] == 'done'
    corrected_file_id = job['corrected_file_id']
    corrected_store_path = Path(str(job['corrected_store_path']))
    assert corrected_store_path.name == 'line001.sgy.statics.datum.11111111'
    assert corrected_store_path.is_dir()
    assert state.file_registry.get_store_path(str(corrected_file_id)) == str(
        corrected_store_path
    )
    assert state.file_registry.get_dt(str(corrected_file_id)) == pytest.approx(0.004)
    with state.lock:
        assert (
            trace_store_cache_key(str(corrected_file_id), KEY1, KEY2)
            in state.cached_readers
        )

    corrected = np.load(corrected_store_path / 'traces.npy')
    assert int(np.argmax(corrected[0])) == 15
    assert corrected[0, 15] == pytest.approx(1.0)

    with np.load(source_store / 'index.npz', allow_pickle=False) as src:
        with np.load(corrected_store_path / 'index.npz', allow_pickle=False) as dst:
            np.testing.assert_array_equal(
                dst['sorted_to_original'],
                src['sorted_to_original'],
            )

    meta = json.loads((corrected_store_path / 'meta.json').read_text(encoding='utf-8'))
    assert meta['derived']['statics_kind'] == 'datum'
    assert meta['derived']['from_file_id'] == req.file_id
    assert meta['derived']['datum_elevation_m'] == pytest.approx(0.0)
    assert meta['derived']['replacement_velocity_m_s'] == pytest.approx(2000.0)

    corrected_manifest = json.loads(
        (job_dir / 'corrected_file.json').read_text(encoding='utf-8')
    )
    assert corrected_manifest['file_id'] == corrected_file_id
    assert corrected_manifest['store_path'] == str(corrected_store_path)
    assert corrected_manifest['derived_from_file_id'] == SOURCE_FILE_ID
    assert corrected_manifest['derived_from_store_path'] == str(source_store)
    assert corrected_manifest['job_id'] == '11111111-2222-3333-4444-555555555555'
    assert corrected_manifest['key1_byte'] == KEY1
    assert corrected_manifest['key2_byte'] == KEY2
    assert corrected_manifest['dt'] == pytest.approx(0.004)

    for name in (
        'job_meta.json',
        'datum_static_solution.npz',
        'datum_static_qc.json',
        'datum_statics.csv',
        'corrected_file.json',
    ):
        assert (job_dir / name).is_file()


def test_datum_static_apply_job_writes_job_meta_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _source_store, _trace_dir = _state_with_source_store(tmp_path, monkeypatch)
    _req, job_dir = _run_job(state, tmp_path)

    meta = json.loads((job_dir / 'job_meta.json').read_text(encoding='utf-8'))

    assert meta['job_type'] == 'statics'
    assert meta['statics_kind'] == 'datum'
    assert meta['source_file_id'] == SOURCE_FILE_ID
    assert meta['key1_byte'] == KEY1
    assert meta['key2_byte'] == KEY2
    assert meta['request']['datum']['mode'] == 'constant'


def test_datum_static_apply_job_materializes_required_headers_in_corrected_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _source_store, _trace_dir = _state_with_source_store(tmp_path, monkeypatch)
    _run_job(state, tmp_path)
    with state.lock:
        corrected_store_path = Path(str(state.jobs[
            '11111111-2222-3333-4444-555555555555'
        ]['corrected_store_path']))

    for byte in (
        KEY1,
        KEY2,
        SOURCE_ELEVATION,
        RECEIVER_ELEVATION,
        ELEVATION_SCALAR,
        SOURCE_STATIC,
        RECEIVER_STATIC,
        TOTAL_STATIC,
    ):
        assert (corrected_store_path / f'headers_byte_{byte}.npy').is_file()


def test_datum_static_apply_job_errors_on_existing_static_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _source_store, trace_dir = _state_with_source_store(
        tmp_path,
        monkeypatch,
        existing_static_value=1,
    )
    req = _request()
    job_id = '22222222-nonzero'
    _run_job(state, tmp_path, job_id=job_id, req=req)

    with state.lock:
        job = dict(state.jobs[job_id])

    assert job['status'] == 'error'
    assert 'existing SEG-Y static headers contain nonzero values' in str(
        job['message']
    )
    assert 'corrected_file_id' not in job
    assert not (trace_dir / 'line001.sgy.statics.datum.22222222').exists()


def test_datum_static_apply_job_errors_on_large_shift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _source_store, trace_dir = _state_with_source_store(tmp_path, monkeypatch)
    req = _request(
        apply={
            'interpolation': 'linear',
            'fill_value': 0.0,
            'max_abs_shift_ms': 50.0,
            'output_dtype': 'float32',
            'register_corrected_file': True,
        }
    )
    job_id = '33333333-large-shift'
    _run_job(state, tmp_path, job_id=job_id, req=req)

    with state.lock:
        job = dict(state.jobs[job_id])

    assert job['status'] == 'error'
    assert 'exceeds max_abs_shift_ms' in str(job['message'])
    assert not (trace_dir / 'line001.sgy.statics.datum.33333333').exists()


def test_datum_static_apply_job_failure_after_build_removes_corrected_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _source_store, trace_dir = _state_with_source_store(tmp_path, monkeypatch)

    def _fail_register(**_kwargs: Any) -> None:
        raise RuntimeError('registration failed')

    monkeypatch.setattr(service, 'register_trace_store', _fail_register)
    job_id = '44444444-register-fails'
    _run_job(state, tmp_path, job_id=job_id)

    with state.lock:
        job = dict(state.jobs[job_id])

    assert job['status'] == 'error'
    assert job['message'] == 'registration failed'
    assert not (trace_dir / 'line001.sgy.statics.datum.44444444').exists()


def test_datum_static_apply_job_cancel_removes_partial_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _source_store, trace_dir = _state_with_source_store(tmp_path, monkeypatch)
    job_id = '55555555-cancelled'

    def _cancelled_build(**kwargs: Any) -> None:
        output_path = Path(kwargs['output_store_path'])
        tmp_store = output_path.with_name(f'{output_path.name}.tmp-test')
        tmp_store.mkdir()
        with state.lock:
            state.jobs.request_cancel(job_id)
        assert kwargs['cancel_check']() is True
        raise RuntimeError('time-shifted TraceStore build cancelled')

    monkeypatch.setattr(
        service,
        'build_time_shifted_trace_store',
        _cancelled_build,
    )
    _run_job(state, tmp_path, job_id=job_id)

    with state.lock:
        job = dict(state.jobs[job_id])

    output = trace_dir / 'line001.sgy.statics.datum.55555555'
    assert job['status'] == 'cancelled'
    assert not output.exists()
    assert not (trace_dir / f'{output.name}.tmp-test').exists()
