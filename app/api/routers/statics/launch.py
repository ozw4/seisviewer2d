"""Shared static correction job launch helpers."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

from app.core.state import AppState
from app.services.jobs import LaunchedJob, launch_managed_job
from app.services import static_job_targets


def launch_static_job(
    *,
    state: AppState,
    file_id: str,
    key1_byte: int,
    key2_byte: int,
    statics_kind: str,
    target: Callable[..., Any],
    target_args: Callable[[str], tuple[Any, ...]],
    pre_create: Callable[[str, Path], None] | None = None,
    after_create: Callable[[MutableMapping[str, object]], None] | None = None,
) -> LaunchedJob:
    return launch_managed_job(
        state,
        create_job=lambda job_id, artifacts_dir: state.jobs.create_static_job(
            job_id,
            file_id=file_id,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            statics_kind=statics_kind,
            artifacts_dir=str(artifacts_dir),
        ),
        target=target,
        target_args=target_args,
        start_thread=static_job_targets.start_static_job_thread,
        pre_create=pre_create,
        after_create=after_create,
    )
