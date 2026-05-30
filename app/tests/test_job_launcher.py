from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.state import create_app_state
from app.services.jobs import launch_managed_job


def test_launch_managed_job_creates_job_and_starts_thread(monkeypatch, tmp_path):
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(tmp_path / 'jobs'))
    state = create_app_state()
    calls: list[tuple[str, object, object | None]] = []
    created_threads: list[Any] = []

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.started = False
            created_threads.append(self)

        def start(self) -> None:
            self.started = True
            calls.append(('thread_start', self.args, None))

    def create_job(job_id: str, artifacts_dir: Path):
        calls.append(('create_job', job_id, artifacts_dir))
        return state.jobs.create_batch_apply_job(
            job_id,
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir=str(artifacts_dir),
        )

    def pre_create(job_id: str, artifacts_dir: Path) -> None:
        calls.append(('pre_create', job_id, artifacts_dir))

    def after_create(job_state) -> None:
        calls.append(('after_create', job_state['status'], None))
        job_state['export_formats'] = ['json']

    def target(job_id: str) -> None:
        raise AssertionError('fake thread should not run the worker')

    def target_args(job_id: str):
        calls.append(('target_args', job_id, None))
        return (job_id,)

    launched = launch_managed_job(
        state,
        create_job=create_job,
        target=target,
        target_args=target_args,
        thread_factory=FakeThread,
        job_id_factory=lambda: 'job-123',
        pre_create=pre_create,
        after_create=after_create,
    )

    expected_dir = tmp_path / 'jobs' / 'job-123'
    assert launched.job_id == 'job-123'
    assert launched.state == 'queued'
    assert launched.artifacts_dir == expected_dir
    assert calls == [
        ('pre_create', 'job-123', expected_dir),
        ('create_job', 'job-123', expected_dir),
        ('after_create', 'queued', None),
        ('target_args', 'job-123', None),
        ('thread_start', ('job-123',), None),
    ]

    with state.lock:
        job_state = dict(state.jobs['job-123'])

    assert job_state['artifacts_dir'] == str(expected_dir)
    assert job_state['export_formats'] == ['json']

    thread = created_threads[0]
    assert thread.target is target
    assert thread.args == ('job-123',)
    assert thread.daemon is True
    assert thread.started is True
    assert not expected_dir.exists()


def test_launch_managed_job_normalizes_created_status(monkeypatch, tmp_path):
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(tmp_path / 'jobs'))
    state = create_app_state()

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            pass

        def start(self) -> None:
            pass

    def create_job(job_id: str, artifacts_dir: Path):
        job = state.jobs.create_batch_apply_job(
            job_id,
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir=str(artifacts_dir),
        )
        job['status'] = 'completed'
        return job

    launched = launch_managed_job(
        state,
        create_job=create_job,
        target=lambda: None,
        target_args=lambda job_id: (),
        thread_factory=FakeThread,
        job_id_factory=lambda: 'job-done',
    )

    assert launched.state == 'done'
