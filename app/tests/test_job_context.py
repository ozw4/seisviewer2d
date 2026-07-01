from __future__ import annotations

from pathlib import Path

import pytest

from app.core.state import create_app_state
from app.services.jobs import JobContext
from app.services.job_runner import JobCancelledError


def test_job_context_resolve_uses_existing_created_ts_and_artifacts_dir() -> None:
    state = create_app_state()
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-1',
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir='/tmp/job-1-artifacts',
            created_ts=123.5,
        )

    context = JobContext.resolve(state, 'job-1')

    assert context.state is state
    assert context.job_id == 'job-1'
    assert context.created_ts == 123.5
    assert context.job_dir == Path('/tmp/job-1-artifacts')


def test_job_context_resolve_falls_back_for_missing_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(tmp_path / 'jobs'))
    monkeypatch.setattr('app.services.jobs.context.time.time', lambda: 987.0)
    state = create_app_state()

    context = JobContext.resolve(state, 'missing-job')

    assert context.created_ts == 987.0
    assert context.job_dir == tmp_path / 'jobs' / 'missing-job'


def test_job_context_resolve_falls_back_for_malformed_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(tmp_path / 'jobs'))
    monkeypatch.setattr('app.services.jobs.context.time.time', lambda: 654.0)
    state = create_app_state()
    with state.lock:
        state.jobs['job-1'] = {
            'created_ts': '123.5',
            'artifacts_dir': '',
        }

    context = JobContext.resolve(state, 'job-1')

    assert context.created_ts == 654.0
    assert context.job_dir == tmp_path / 'jobs' / 'job-1'


def test_set_progress_message_updates_progress_and_message() -> None:
    state = create_app_state()
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-1',
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir='/tmp/job-1-artifacts',
        )
    context = JobContext.resolve(state, 'job-1')

    assert context.set_progress_message(progress=0.42, message='working') is True

    with state.lock:
        job = dict(state.jobs['job-1'])
    assert job['progress'] == 0.42
    assert job['message'] == 'working'


def test_job_context_update_methods_noop_when_job_missing() -> None:
    state = create_app_state()
    context = JobContext.resolve(state, 'missing-job')

    assert context.set_progress_message(progress=0.5, message='working') is False
    assert context.set_progress(0.6) is False
    assert context.set_message('working') is False
    assert len(state.jobs) == 0


def test_job_context_update_methods_modify_existing_job() -> None:
    state = create_app_state()
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-1',
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir='/tmp/job-1-artifacts',
        )
    context = JobContext.resolve(state, 'job-1')

    assert context.set_progress(0.7) is True
    assert context.set_message('almost done') is True

    with state.lock:
        job = dict(state.jobs['job-1'])
    assert job['progress'] == 0.7
    assert job['message'] == 'almost done'


def test_cancel_requested_job_is_reported_and_raises() -> None:
    state = create_app_state()
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-1',
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir='/tmp/job-1-artifacts',
        )
        state.jobs.set_status('job-1', 'running')
        state.jobs.request_cancel('job-1')
    context = JobContext.resolve(state, 'job-1')

    assert context.is_cancel_requested() is True
    with pytest.raises(JobCancelledError) as exc_info:
        context.ensure_not_cancelled()
    assert exc_info.value.message == 'The job was cancelled by the user.'


def test_cancel_check_reads_latest_cancel_state() -> None:
    state = create_app_state()
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-1',
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir='/tmp/job-1-artifacts',
        )
        state.jobs.set_status('job-1', 'running')
    context = JobContext.resolve(state, 'job-1')
    cancel_check = context.cancel_check()

    assert cancel_check() is False
    with state.lock:
        state.jobs.request_cancel('job-1')
    assert cancel_check() is True
