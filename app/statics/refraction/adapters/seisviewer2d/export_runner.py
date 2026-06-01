"""SeisViewer2D lifecycle runner for refraction export jobs."""

from __future__ import annotations

import time

from app.core.state import AppState
from app.services.job_runner import JobFailure, run_job_with_lifecycle
from app.statics.refraction.adapters.seisviewer2d.runtime import (
    SeisViewer2DRefractionRuntime,
)
from app.statics.refraction.application.export_service import (
    _run_refraction_static_export_job_body,
)
from app.statics.refraction.contracts.export import RefractionStaticExportJobRequest


def run_refraction_static_export_job(
    job_id: str,
    req: RefractionStaticExportJobRequest,
    state: AppState,
) -> None:
    """Run a standalone M5 export job through SeisViewer2D services."""
    runtime = SeisViewer2DRefractionRuntime(state)
    job_dir = runtime.job_artifacts_dir(job_id)
    context = runtime.job_context(job_id=job_id, artifacts_dir=job_dir)

    def worker():
        return _run_refraction_static_export_job_body(
            job_id=job_id,
            req=req,
            runtime=runtime,
            context=context,
        )

    run_job_with_lifecycle(
        state=state,
        job_id=job_id,
        worker=worker,
        progress_1_on_done=False,
        start_progress=0.0,
        clear_message_on_start=True,
        on_error=lambda _exc: JobFailure(finished_ts=time.time()),
    )


__all__ = ['run_refraction_static_export_job']
