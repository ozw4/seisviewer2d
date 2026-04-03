"""Shared helpers for background job lifecycle orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.core.state import AppState


@dataclass(frozen=True)
class JobCompletion:
    finished_ts: float | None = None


@dataclass(frozen=True)
class JobFailure:
    message: str | None = None
    finished_ts: float | None = None


@dataclass(frozen=True)
class JobCancelledError(Exception):
    message: str = 'The job was cancelled by the user.'
    finished_ts: float | None = None


def mark_job_running(
    state: AppState,
    job_id: str,
    *,
    progress: float | None = None,
    clear_message: bool = False,
) -> bool:
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            return False
        status = job.get('status')
        if isinstance(status, str) and status.lower() == 'cancelled':
            return False
        state.jobs.set_status(job_id, 'running')
        if progress is not None:
            state.jobs.set_progress(job_id, progress)
        if clear_message:
            state.jobs.set_message(job_id, '')
        return True


def set_job_progress(state: AppState, job_id: str, progress: float) -> bool:
    with state.lock:
        if state.jobs.get(job_id) is None:
            return False
        state.jobs.set_progress(job_id, progress)
        return True


def set_job_message(state: AppState, job_id: str, message: str) -> bool:
    with state.lock:
        if state.jobs.get(job_id) is None:
            return False
        state.jobs.set_message(job_id, message)
        return True


def request_job_cancel(
    state: AppState,
    job_id: str,
    *,
    message: str = 'Cancel requested. The job will stop at the next safe point.',
) -> bool:
    with state.lock:
        return state.jobs.request_cancel(job_id, message=message)


def ensure_job_not_cancelled(
    state: AppState,
    job_id: str,
    *,
    message: str = 'The job was cancelled by the user.',
) -> None:
    with state.lock:
        if not state.jobs.is_cancel_requested(job_id):
            return
    raise JobCancelledError(message=message)


def run_job_with_lifecycle(
    *,
    state: AppState,
    job_id: str,
    worker: Callable[[], JobCompletion | None],
    progress_1_on_done: bool = False,
    start_progress: float | None = None,
    clear_message_on_start: bool = False,
    on_error: Callable[[Exception], JobFailure | None] | None = None,
) -> None:
    if not mark_job_running(
        state,
        job_id,
        progress=start_progress,
        clear_message=clear_message_on_start,
    ):
        return

    try:
        completion = worker()
    except JobCancelledError as exc:
        with state.lock:
            if state.jobs.get(job_id) is None:
                return
            state.jobs.mark_cancelled(
                job_id,
                exc.message,
                finished_ts=exc.finished_ts,
            )
        return
    except Exception as exc:  # noqa: BLE001
        failure = on_error(exc) if on_error is not None else None
        message = (
            str(exc) if failure is None or failure.message is None else failure.message
        )
        finished_ts = None if failure is None else failure.finished_ts
        with state.lock:
            if state.jobs.get(job_id) is None:
                return
            state.jobs.mark_error(job_id, message, finished_ts=finished_ts)
        return

    finished_ts = None if completion is None else completion.finished_ts
    with state.lock:
        if state.jobs.get(job_id) is None:
            return
        state.jobs.mark_done(
            job_id,
            finished_ts=finished_ts,
            progress_1=progress_1_on_done,
        )


def start_job_thread(
    *,
    target: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    thread_factory: Callable[..., Any],
) -> Any:
    if kwargs is None:
        thread = thread_factory(target=target, args=args, daemon=True)
    else:
        thread = thread_factory(target=target, args=args, kwargs=kwargs, daemon=True)
    thread.start()
    return thread


__all__ = [
    'JobCompletion',
    'JobCancelledError',
    'JobFailure',
    'ensure_job_not_cancelled',
    'mark_job_running',
    'request_job_cancel',
    'run_job_with_lifecycle',
    'set_job_message',
    'set_job_progress',
    'start_job_thread',
]
