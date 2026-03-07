from __future__ import annotations

from app.core.state import create_app_state
from app.services.job_runner import (
    JobCompletion,
    JobFailure,
    run_job_with_lifecycle,
    start_job_thread,
)


def _create_batch_job(job_id: str = 'job'):
    state = create_app_state()
    with state.lock:
        state.jobs.create_batch_apply_job(
            job_id,
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir='/tmp/job-artifacts',
        )
    return state


def test_run_job_with_lifecycle_transitions_running_to_done():
    state = _create_batch_job()
    with state.lock:
        state.jobs.set_message('job', 'stale')

    observed: list[tuple[object, object, object]] = []

    def worker() -> JobCompletion:
        with state.lock:
            job = state.jobs['job']
            observed.append((job['status'], job['progress'], job['message']))
        return JobCompletion(finished_ts=123.0)

    run_job_with_lifecycle(
        state=state,
        job_id='job',
        worker=worker,
        start_progress=0.25,
        clear_message_on_start=True,
    )

    with state.lock:
        job = dict(state.jobs['job'])

    assert observed == [('running', 0.25, '')]
    assert job['status'] == 'done'
    assert job['progress'] == 0.25
    assert job['message'] == ''
    assert job['finished_ts'] == 123.0


def test_run_job_with_lifecycle_marks_error_and_message():
    state = _create_batch_job()

    def worker() -> JobCompletion:
        raise RuntimeError('boom')

    run_job_with_lifecycle(
        state=state,
        job_id='job',
        worker=worker,
        on_error=lambda exc: JobFailure(finished_ts=456.0),
    )

    with state.lock:
        job = dict(state.jobs['job'])

    assert job['status'] == 'error'
    assert job['message'] == 'boom'
    assert job['finished_ts'] == 456.0


def test_run_job_with_lifecycle_noops_when_job_missing():
    state = create_app_state()
    called = False

    def worker() -> JobCompletion | None:
        nonlocal called
        called = True
        return None

    run_job_with_lifecycle(
        state=state,
        job_id='missing',
        worker=worker,
        start_progress=0.0,
        clear_message_on_start=True,
    )

    assert called is False
    assert len(state.jobs) == 0


def test_run_job_with_lifecycle_progress_1_on_done_switch():
    state = _create_batch_job('job-1')
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-2',
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir='/tmp/job-artifacts-2',
        )
        state.jobs.set_progress('job-1', 0.4)
        state.jobs.set_progress('job-2', 0.4)

    run_job_with_lifecycle(
        state=state,
        job_id='job-1',
        worker=lambda: None,
        progress_1_on_done=False,
    )
    run_job_with_lifecycle(
        state=state,
        job_id='job-2',
        worker=lambda: None,
        progress_1_on_done=True,
    )

    with state.lock:
        job_1 = dict(state.jobs['job-1'])
        job_2 = dict(state.jobs['job-2'])

    assert job_1['progress'] == 0.4
    assert job_2['progress'] == 1.0


def test_start_job_thread_uses_injected_thread_factory():
    created = []

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.started = False
            created.append(self)

        def start(self) -> None:
            self.started = True

    def target(job_id: str) -> None:
        return None

    thread = start_job_thread(
        thread_factory=FakeThread,
        target=target,
        args=('job-1',),
    )

    assert thread is created[0]
    assert thread.target is target
    assert thread.args == ('job-1',)
    assert thread.daemon is True
    assert thread.started is True
