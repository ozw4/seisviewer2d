"""Shared static correction job launch helpers."""

from __future__ import annotations

import threading
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

from app.core.state import AppState
from app.services.jobs import LaunchedJob, launch_managed_job


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
        thread_factory=threading.Thread,
        start_thread=_static_router_start_job_thread,
        pre_create=pre_create,
        after_create=after_create,
    )


def _static_router_start_job_thread(**kwargs: Any) -> object:
    from app.api.routers import statics as statics_router_module

    return statics_router_module.start_job_thread(**kwargs)
