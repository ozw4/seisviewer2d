from __future__ import annotations

import json
import time

import numpy as np
import pytest

from app.api.schemas import BatchApplyRequest
from app.core.state import create_app_state
from app.services import batch_apply_service
from app.services.pipeline_artifacts import get_job_dir
from app.services.reader import coerce_section_f32
from app.trace_store.reader import TraceStoreSectionReader

KEY1 = 189
KEY2 = 193


def _write_min_store(
    tmp_path, key1s: np.ndarray, key2s: np.ndarray, traces: np.ndarray
):
    store = tmp_path / 'store'
    store.mkdir(parents=True, exist_ok=True)
    np.save(store / 'traces.npy', traces.astype(np.float32, copy=False))
    np.save(store / f'headers_byte_{KEY1}.npy', key1s.astype(np.int32, copy=False))
    np.save(store / f'headers_byte_{KEY2}.npy', key2s.astype(np.int32, copy=False))
    np.savez(
        store / 'index.npz',
        key1_values=np.unique(key1s),
        key1_offsets=np.array([], dtype=np.int32),
        key1_counts=np.array([], dtype=np.int32),
        sorted_to_original=np.arange(int(key1s.size), dtype=np.int64),
    )
    meta = {
        'dt': 0.002,
        'key_bytes': {'key1': KEY1, 'key2': KEY2},
        'original_segy_path': 'dummy.sgy',
        'original_mtime': 0.0,
        'original_size': 0,
    }
    (store / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    return store


def _job_request() -> BatchApplyRequest:
    return BatchApplyRequest(
        file_id='file-a',
        key1_byte=KEY1,
        key2_byte=KEY2,
        pipeline_spec={
            'steps': [
                {'kind': 'transform', 'name': 'denoise', 'params': {}},
                {'kind': 'analyzer', 'name': 'fbpick', 'params': {}},
            ]
        },
    )


def test_batch_apply_outputs_3d_saved_with_padding_and_done(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    jobs_dir = tmp_path / 'jobs'
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(jobs_dir))

    key1s = np.array([20, 10, 20, 10, 30], dtype=np.int32)
    key2s = np.array([2, 5, 1, 3, 4], dtype=np.int32)
    traces = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    store = _write_min_store(tmp_path, key1s, key2s, traces)

    def _stub_pipeline_outputs(*, section, meta, spec, denoise_taps, fbpick_label):
        del meta, spec, denoise_taps, fbpick_label
        denoise = np.asarray(section, dtype=np.float32, order='C')
        prob = np.ones(section.shape, dtype=np.float16)
        return denoise, prob

    monkeypatch.setattr(
        batch_apply_service, '_run_pipeline_outputs', _stub_pipeline_outputs
    )

    state = create_app_state()
    state.file_registry.set_record('file-a', {'store_path': str(store), 'dt': 0.002})
    job_id = 'job3d'
    state.jobs[job_id] = {
        'status': 'queued',
        'progress': 0.0,
        'message': '',
        'created_ts': time.time(),
        'file_id': 'file-a',
        'key1_byte': KEY1,
        'key2_byte': KEY2,
    }
    req = _job_request()

    batch_apply_service.run_batch_apply_job(job_id, req, state)

    assert state.jobs[job_id]['status'] == 'done'
    assert state.jobs[job_id]['progress'] == pytest.approx(1.0)

    job_dir = get_job_dir(job_id)
    key1_values = np.load(job_dir / 'key1_values.npy')
    assert key1_values.dtype == np.int32
    assert key1_values.tolist() == [10, 20, 30]

    key2_padded = np.load(job_dir / 'key2_values_padded.npy')
    assert key2_padded.dtype == np.int32
    assert key2_padded.shape == (3, 2)
    assert key2_padded.tolist() == [[3, 5], [1, 2], [4, 0]]

    denoise = np.load(job_dir / 'denoise_f32_padded.npy')
    prob = np.load(job_dir / 'fbpick_prob_f16_padded.npy')
    assert denoise.dtype == np.float32
    assert prob.dtype == np.float16
    assert denoise.shape == (3, 2, 3)
    assert prob.shape == (3, 2, 3)

    reader = TraceStoreSectionReader(store, KEY1, KEY2)
    for i, key1 in enumerate(key1_values.tolist()):
        view = reader.get_section(int(key1))
        section = coerce_section_f32(view.arr, view.scale)
        n_traces = int(section.shape[0])
        assert np.array_equal(denoise[i, :n_traces, :], section)
        assert np.array_equal(
            prob[i, :n_traces, :], np.ones(section.shape, dtype=np.float16)
        )
        if n_traces < denoise.shape[1]:
            assert np.all(denoise[i, n_traces:, :] == 0.0)
            assert np.all(prob[i, n_traces:, :] == np.float16(0.0))


def test_batch_apply_stops_on_section_failure_and_sets_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    jobs_dir = tmp_path / 'jobs'
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(jobs_dir))

    key1s = np.array([20, 10, 20, 10, 30], dtype=np.int32)
    key2s = np.array([2, 5, 1, 3, 4], dtype=np.int32)
    traces = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    store = _write_min_store(tmp_path, key1s, key2s, traces)

    def _stub_pipeline_outputs(*, section, meta, spec, denoise_taps, fbpick_label):
        del meta, spec, denoise_taps, fbpick_label
        denoise = np.asarray(section, dtype=np.float32, order='C')
        prob = np.ones(section.shape, dtype=np.float16)
        return denoise, prob

    calls: list[int] = []
    orig_load = batch_apply_service._load_section_and_key2

    def _failing_load(*, reader, key1, key2_byte):
        calls.append(int(key1))
        if int(key1) == 20:
            raise ValueError('boom section 20')
        return orig_load(reader=reader, key1=key1, key2_byte=key2_byte)

    monkeypatch.setattr(
        batch_apply_service, '_run_pipeline_outputs', _stub_pipeline_outputs
    )
    monkeypatch.setattr(batch_apply_service, '_load_section_and_key2', _failing_load)

    state = create_app_state()
    state.file_registry.set_record('file-a', {'store_path': str(store), 'dt': 0.002})
    job_id = 'joberr'
    state.jobs[job_id] = {
        'status': 'queued',
        'progress': 0.0,
        'message': '',
        'created_ts': time.time(),
        'file_id': 'file-a',
        'key1_byte': KEY1,
        'key2_byte': KEY2,
    }
    req = _job_request()

    batch_apply_service.run_batch_apply_job(job_id, req, state)

    assert state.jobs[job_id]['status'] == 'error'
    assert 'boom section 20' in str(state.jobs[job_id]['message'])
    assert float(state.jobs[job_id]['progress']) < 1.0
    assert calls == [10, 20]
