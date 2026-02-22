"""Batch apply predicted picks NPZ integration tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from app.api.schemas import (
    BatchApplyRequest,
    PickOptions,
    PipelineOp,
    PipelineSpec,
    SnapOptions,
)
from app.core.state import AppState
from app.services import batch_apply_service


MANUAL_NPZ_KEYS = {
    'picks_time_s',
    'n_traces',
    'n_samples',
    'dt',
    'format_version',
    'exported_at',
    'export_app',
    'source_hint',
}


def test_batch_apply_request_save_picks_defaults_false() -> None:
    req = BatchApplyRequest(
        file_id='dummy',
        pipeline_spec=PipelineSpec(
            steps=[PipelineOp(kind='analyzer', name='fbpick', params={})]
        ),
    )
    assert req.save_picks is False


def test_predict_section_picks_subsample_skips_endpoint_refine(monkeypatch) -> None:
    prob = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float16,
    )
    raw = np.zeros_like(prob, dtype=np.float32)

    def _stub_pick_index(prob, *, method, chunk=4096):
        del prob, chunk
        assert method == 'argmax'
        return np.asarray([0.0, 2.0, 4.0], dtype=np.float64)

    def _stub_sigma(prob, *, dt, chunk=4096):
        del prob, dt, chunk
        return np.asarray([1.0, 1.0, 1.0], dtype=np.float64)

    refine_calls: list[int] = []

    def _stub_parabolic(arr, i):
        del arr
        refine_calls.append(int(i))
        return float(i) + 0.25

    monkeypatch.setattr(batch_apply_service, 'pick_index_from_prob', _stub_pick_index)
    monkeypatch.setattr(batch_apply_service, 'sigma_ms_from_prob', _stub_sigma)
    monkeypatch.setattr(batch_apply_service, 'parabolic_refine', _stub_parabolic)

    out = batch_apply_service._predict_section_picks_time_s(
        prob=prob,
        raw_section=raw,
        dt=0.002,
        pick_options=PickOptions(method='argmax', subsample=True, sigma_ms_max=None),
    )
    assert refine_calls == [2]
    np.testing.assert_allclose(
        out,
        np.asarray([0.0, 0.0045, 0.008], dtype=np.float64),
    )


def _create_min_store(tmp_path: Path) -> Path:
    store_dir = tmp_path / 'store'
    store_dir.mkdir(parents=True, exist_ok=True)
    traces = np.asarray(
        [
            [10, 11, 12, 13, 14],  # sorted idx 0, key1=10 key2=200
            [20, 21, 22, 23, 24],  # sorted idx 1, key1=10 key2=100
            [30, 31, 32, 33, 34],  # sorted idx 2, key1=20 key2=200
            [40, 41, 42, 43, 44],  # sorted idx 3, key1=20 key2=100
        ],
        dtype=np.float32,
    )
    np.save(store_dir / 'traces.npy', traces)
    np.save(
        store_dir / 'headers_byte_189.npy',
        np.asarray([10, 10, 20, 20], dtype=np.int32),
    )
    np.save(
        store_dir / 'headers_byte_193.npy',
        np.asarray([200, 100, 200, 100], dtype=np.int32),
    )
    np.savez(
        store_dir / 'index.npz',
        sorted_to_original=np.asarray([2, 0, 3, 1], dtype=np.int64),
    )
    meta = {
        'dt': 0.002,
        'original_segy_path': str(tmp_path / 'dummy.sgy'),
    }
    (store_dir / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    return store_dir


def _fbpick_only_request(*, file_id: str, save_picks: bool = True) -> BatchApplyRequest:
    return BatchApplyRequest(
        file_id=file_id,
        key1_byte=189,
        key2_byte=193,
        pipeline_spec=PipelineSpec(
            steps=[
                PipelineOp(
                    kind='analyzer',
                    name='fbpick',
                    params={},
                    label='fbpick',
                )
            ]
        ),
        pick_options=PickOptions(
            method='argmax',
            subsample=True,
            sigma_ms_max=50.0,
            snap=SnapOptions(
                enabled=True,
                mode='peak',
                refine='parabolic',
                window_ms=20.0,
            ),
        ),
        save_picks=save_picks,
    )


def test_batch_apply_writes_predicted_picks_npz_manual_compatible(
    tmp_path: Path, monkeypatch
) -> None:
    file_id = 'pr5-picks-file'
    store_dir = _create_min_store(tmp_path)

    job_root = tmp_path / 'jobs'
    monkeypatch.setattr(
        batch_apply_service,
        'get_job_dir',
        lambda job_id: job_root / job_id,
    )

    def _stub_pipeline_outputs(*, section, meta, spec, denoise_taps, fbpick_label):
        n_traces, n_samples = section.shape
        prob = np.zeros((n_traces, n_samples), dtype=np.float16)
        prob[:, 1] = 0.3
        prob[:, 2] = 0.7
        return None, prob

    def _stub_pick_index(prob, *, method, chunk=4096):
        assert method == 'argmax'
        return np.asarray([1.0, 2.0], dtype=np.float64)

    def _stub_sigma(prob, *, dt, chunk=4096):
        return np.asarray([10.0, 999.0], dtype=np.float64)

    refine_calls: list[int] = []

    def _stub_parabolic(arr, i):
        refine_calls.append(int(i))
        return float(i) + 0.5

    snap_calls: list[float] = []

    def _stub_snap(trace, time_s, *, dt, mode, refine, window_ms):
        if not np.isfinite(time_s):
            raise AssertionError('snap_pick_time_s must not receive NaN picks')
        snap_calls.append(float(time_s))
        return float(time_s) + float(dt)

    monkeypatch.setattr(
        batch_apply_service,
        '_run_pipeline_outputs',
        _stub_pipeline_outputs,
    )
    monkeypatch.setattr(batch_apply_service, 'pick_index_from_prob', _stub_pick_index)
    monkeypatch.setattr(batch_apply_service, 'sigma_ms_from_prob', _stub_sigma)
    monkeypatch.setattr(batch_apply_service, 'parabolic_refine', _stub_parabolic)
    monkeypatch.setattr(batch_apply_service, 'snap_pick_time_s', _stub_snap)

    state = AppState()
    state.file_registry.set_record(
        file_id,
        {
            'store_path': str(store_dir),
            'path': str(tmp_path / 'dummy.sgy'),
            'dt': 0.002,
        },
    )
    job_id = 'job-pr5-picks'
    state.jobs[job_id] = {
        'status': 'queued',
        'progress': 0.0,
        'message': '',
        'created_ts': time.time(),
    }
    req = _fbpick_only_request(file_id=file_id, save_picks=True)
    batch_apply_service.run_batch_apply_job(job_id, req, state)

    assert state.jobs[job_id]['status'] == 'done'
    assert len(refine_calls) == 4
    assert len(snap_calls) == 2

    predicted_npz = job_root / job_id / 'predicted_picks_time_s.npz'
    assert predicted_npz.is_file()
    with np.load(predicted_npz, allow_pickle=False) as npz:
        assert set(npz.files) == MANUAL_NPZ_KEYS
        picks_time_s = npz['picks_time_s']
        assert picks_time_s.dtype == np.float32
        assert picks_time_s.shape == (4,)
        np.testing.assert_allclose(
            picks_time_s,
            np.asarray([0.005, 0.005, np.nan, np.nan], dtype=np.float32),
            equal_nan=True,
        )
        assert np.asarray(npz['n_traces']).dtype == np.int64
        assert int(np.asarray(npz['n_traces']).item()) == 4
        assert np.asarray(npz['n_samples']).dtype == np.int64
        assert int(np.asarray(npz['n_samples']).item()) == 5
        assert np.asarray(npz['dt']).dtype == np.float64
        assert float(np.asarray(npz['dt']).item()) == 0.002
        assert np.asarray(npz['format_version']).dtype == np.int64
        assert int(np.asarray(npz['format_version']).item()) == 1
        assert str(np.asarray(npz['export_app']).item()) == 'seisviewer2d'

    meta_path = job_root / job_id / 'job_meta.json'
    finished_meta = json.loads(meta_path.read_text(encoding='utf-8'))
    assert 'predicted_picks_time_s.npz' in finished_meta['outputs']


def test_batch_apply_skips_predicted_picks_when_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    file_id = 'pr5-picks-disabled-file'
    store_dir = _create_min_store(tmp_path)

    job_root = tmp_path / 'jobs'
    monkeypatch.setattr(
        batch_apply_service,
        'get_job_dir',
        lambda job_id: job_root / job_id,
    )

    def _stub_pipeline_outputs(*, section, meta, spec, denoise_taps, fbpick_label):
        n_traces, n_samples = section.shape
        return None, np.zeros((n_traces, n_samples), dtype=np.float16)

    monkeypatch.setattr(
        batch_apply_service,
        '_run_pipeline_outputs',
        _stub_pipeline_outputs,
    )

    state = AppState()
    state.file_registry.set_record(
        file_id,
        {
            'store_path': str(store_dir),
            'path': str(tmp_path / 'dummy.sgy'),
            'dt': 0.002,
        },
    )
    job_id = 'job-pr5-picks-disabled'
    state.jobs[job_id] = {
        'status': 'queued',
        'progress': 0.0,
        'message': '',
        'created_ts': time.time(),
    }
    req = _fbpick_only_request(file_id=file_id, save_picks=False)
    batch_apply_service.run_batch_apply_job(job_id, req, state)

    assert state.jobs[job_id]['status'] == 'done'
    predicted_npz = job_root / job_id / 'predicted_picks_time_s.npz'
    assert not predicted_npz.exists()

    meta_path = job_root / job_id / 'job_meta.json'
    finished_meta = json.loads(meta_path.read_text(encoding='utf-8'))
    assert 'predicted_picks_time_s.npz' not in finished_meta['outputs']
