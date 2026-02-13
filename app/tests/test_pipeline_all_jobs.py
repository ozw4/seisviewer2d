# app/tests/test_pipeline_all_jobs.py
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api._helpers import get_state
from app.main import app
from app.services.pipeline_artifacts import get_job_dir, safe_filename


class _CapturedThread:
    """Capture background threads created by the router without starting them."""

    created: list["_CapturedThread"] = []

    def __init__(self, *, target, args=(), daemon=None):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False
        _CapturedThread.created.append(self)

    def start(self):
        self.started = True


class _StubReader:
    """Minimal reader used by _run_pipeline_all_job."""

    def __init__(self, key1_vals: list[int]):
        self._key1_vals = np.asarray(key1_vals, dtype=np.int32)
        self.meta = {'dt': 0.002}

    def get_key1_values(self):
        return self._key1_vals

    def get_section(self, key1_val: int):
        # section[0,0] が key1_val になるように固定値で埋める
        arr = np.full((4, 8), float(key1_val), dtype=np.float32)
        return SimpleNamespace(arr=arr, scale=None)


@pytest.fixture()
def pipeline_env(tmp_path: Path, monkeypatch):
    """
    - PIPELINE_JOBS_DIR を tmp に隔離
    - pipeline router の threading.Thread を捕捉クラスに差し替え
    - get_reader/apply_pipeline をスタブ化
    """
    from app.api.routers import pipeline as pipe

    state = get_state(app)
    state.jobs.clear()
    state.pipeline_tap_cache.clear()
    state.cached_readers.clear()

    jobs_dir = tmp_path / 'pipeline_jobs'
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(jobs_dir))

    _CapturedThread.created.clear()
    monkeypatch.setattr(
        pipe, 'threading', SimpleNamespace(Thread=_CapturedThread), raising=True
    )

    reader = _StubReader([10, 20])
    monkeypatch.setattr(pipe, 'get_reader', lambda *a, **k: reader, raising=True)

    started = threading.Event()
    proceed = threading.Event()

    def _fake_apply_pipeline(section, *, spec, meta, taps):
        # running 状態になった後の最初の apply_pipeline で一旦止める（artifact not ready を検証）
        if not started.is_set():
            started.set()
            ok = proceed.wait(timeout=2.0)
            assert ok, 'test coordination failed: proceed event not set'
        labels = taps or ['final']
        val = float(section[0, 0])
        return {
            label: {'value': val, 'dt': float(meta.get('dt', 0.0))} for label in labels
        }

    monkeypatch.setattr(pipe, 'apply_pipeline', _fake_apply_pipeline, raising=True)

    client = TestClient(app)
    return client, pipe, state, started, proceed, jobs_dir


def _post_pipeline_all(client: TestClient, *, file_id: str, taps: list[str]):
    r = client.post(
        '/pipeline/all',
        params={'file_id': file_id, 'key1_byte': 189, 'key2_byte': 193},
        json={'spec': {'steps': []}, 'taps': taps},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert 'job_id' in body
    return body['job_id']


def test_pipeline_all_job_status_and_artifact_flow(pipeline_env):
    client, pipe, state, started, proceed, jobs_dir = pipeline_env
    file_id = 'f1'
    taps = ['final']

    # 404: unknown job id
    r0 = client.get('/pipeline/job/missing/status')
    assert r0.status_code == 404
    assert r0.json()['detail'] == 'Job ID not found'

    r0b = client.get(
        '/pipeline/job/missing/artifact', params={'key1_val': 10, 'tap': 'final'}
    )
    assert r0b.status_code == 404
    assert r0b.json()['detail'] == 'Job ID not found'

    # POST /pipeline/all: job created (queued)
    job_id = _post_pipeline_all(client, file_id=file_id, taps=taps)
    assert _CapturedThread.created, 'background thread was not created'
    captured = _CapturedThread.created[-1]
    assert captured.started is True  # router called .start()

    st_q = client.get(f'/pipeline/job/{job_id}/status')
    assert st_q.status_code == 200
    assert st_q.json()['state'] == 'queued'
    assert st_q.json()['progress'] == pytest.approx(0.0)

    # queued の間は artifact は当然無い
    art_q = client.get(
        f'/pipeline/job/{job_id}/artifact', params={'key1_val': 10, 'tap': 'final'}
    )
    assert art_q.status_code == 404
    assert art_q.json()['detail'] == 'Artifact not ready'

    # ---- run the captured job in a real thread (controlled) ----
    worker = threading.Thread(target=captured.target, args=captured.args, daemon=True)
    worker.start()

    ok = started.wait(timeout=2.0)
    assert ok, 'apply_pipeline was not reached; job did not enter running state'

    st_r = client.get(f'/pipeline/job/{job_id}/status')
    assert st_r.status_code == 200
    assert st_r.json()['state'] == 'running'
    assert st_r.json()['progress'] == pytest.approx(0.0)

    # running 中（最初の apply_pipeline で停止中）は artifact はまだ無い
    art_r = client.get(
        f'/pipeline/job/{job_id}/artifact', params={'key1_val': 10, 'tap': 'final'}
    )
    assert art_r.status_code == 404
    assert art_r.json()['detail'] == 'Artifact not ready'

    # job を最後まで進める
    proceed.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive(), 'job thread did not finish'

    st_d = client.get(f'/pipeline/job/{job_id}/status')
    assert st_d.status_code == 200
    assert st_d.json()['state'] == 'done'
    assert st_d.json()['progress'] == pytest.approx(1.0)
    assert st_d.json()['message'] == ''

    # ---- done 後: disk artifact が取得できる ----
    job = state.jobs[job_id]
    pipe_key = job['pipeline_key']
    offset_byte = job.get('offset_byte')

    # key1=10 の cache を消して disk read を強制（disk 経由で取れればOK）
    base_key_10 = (file_id, 10, 189, pipe_key, None, offset_byte)
    cache_key_10 = (*base_key_10, 'final')
    state.pipeline_tap_cache.pop(cache_key_10, None)

    art10 = client.get(
        f'/pipeline/job/{job_id}/artifact', params={'key1_val': 10, 'tap': 'final'}
    )
    assert art10.status_code == 200
    assert art10.json() == {'value': 10.0, 'dt': 0.002}

    # ---- disk 成果物が無い場合: in-memory LRU にフォールバック ----
    # key1=20 の disk artifact を削除して fallback を検証
    job_dir = get_job_dir(job_id)
    artifact_path_20 = job_dir / '20' / f'{safe_filename("final")}.bin'
    assert artifact_path_20.is_file()
    artifact_path_20.unlink()

    # cache は残っているはずなので取得できる
    art20 = client.get(
        f'/pipeline/job/{job_id}/artifact', params={'key1_val': 20, 'tap': 'final'}
    )
    assert art20.status_code == 200
    assert art20.json() == {'value': 20.0, 'dt': 0.002}

    # cache も消すと 404（fallback 不能）
    base_key_20 = (file_id, 20, 189, pipe_key, None, offset_byte)
    cache_key_20 = (*base_key_20, 'final')
    state.pipeline_tap_cache.pop(cache_key_20, None)

    art20_nf = client.get(
        f'/pipeline/job/{job_id}/artifact', params={'key1_val': 20, 'tap': 'final'}
    )
    assert art20_nf.status_code == 404
    assert art20_nf.json()['detail'] == 'Artifact not ready'
