"""Launch helpers for managed background jobs."""

from __future__ import annotations

import threading
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.state import AppState
from app.services.job_manager import JobManager
from app.services.job_runner import start_job_thread
from app.services.pipeline_artifacts import get_job_dir


@dataclass(frozen=True)
class LaunchedJob:
    job_id: str
    state: str
    artifacts_dir: Path


def launch_managed_job(
    state: AppState,
    *,
    create_job: Callable[[str, Path], MutableMapping[str, object]],
    target: Callable[..., Any],
    target_args: Callable[[str], tuple[Any, ...]],
    thread_factory: Callable[..., Any] | None = None,
    start_thread: Callable[..., Any] | None = None,
    job_id_factory: Callable[[], str] | None = None,
    pre_create: Callable[[str, Path], None] | None = None,
    after_create: Callable[[MutableMapping[str, object]], None] | None = None,
) -> LaunchedJob:
    """Create a managed job record and launch its worker thread."""
    job_id = str(uuid4()) if job_id_factory is None else str(job_id_factory())
    artifacts_dir = get_job_dir(job_id)
    thread_factory = threading.Thread if thread_factory is None else thread_factory
    start_thread = start_job_thread if start_thread is None else start_thread

    if pre_create is not None:
        pre_create(job_id, artifacts_dir)

    with state.lock:
        job_state = create_job(job_id, artifacts_dir)
        if after_create is not None:
            after_create(job_state)
        status = JobManager.normalize_status_value(job_state.get('status', 'unknown'))

    args = target_args(job_id)
    start_thread(
        target=target,
        args=args,
        thread_factory=thread_factory,
    )

    return LaunchedJob(
        job_id=job_id,
        state=status,
        artifacts_dir=artifacts_dir,
    )


__all__ = [
    'LaunchedJob',
    'launch_managed_job',
]
