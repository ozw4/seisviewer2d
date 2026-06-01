"""SeisViewer2D lifecycle runner for refraction static apply jobs."""

from __future__ import annotations

from pathlib import Path
import time

from app.core.state import AppState
from app.services.job_runner import JobFailure, run_job_with_lifecycle
from app.statics.refraction.adapters.seisviewer2d.runtime import (
    SeisViewer2DRefractionRuntime,
)
from app.statics.refraction.application.workflow import (
    _run_refraction_static_apply_job_body,
    _write_refraction_static_failure_diagnostics,
)
from app.statics.refraction.contracts.apply import RefractionStaticApplyRequest


def run_refraction_static_apply_job(
    job_id: str,
    req: RefractionStaticApplyRequest,
    state: AppState,
    uploaded_pick_npz_path: Path | None = None,
    uploaded_pick_metadata: dict[str, object] | None = None,
) -> None:
    """Start the refraction statics job lifecycle through SeisViewer2D services."""
    runtime = SeisViewer2DRefractionRuntime(state)
    job_dir = runtime.job_artifacts_dir(job_id)
    context = runtime.job_context(job_id=job_id, artifacts_dir=job_dir)

    def worker():
        return _run_refraction_static_apply_job_body(
            job_id=job_id,
            req=req,
            runtime=runtime,
            context=context,
            uploaded_pick_npz_path=uploaded_pick_npz_path,
            uploaded_pick_metadata=uploaded_pick_metadata,
        )

    def on_error(exc: Exception) -> JobFailure:
        try:
            _write_refraction_static_failure_diagnostics(
                job_id=job_id,
                exc=exc,
                runtime=runtime,
            )
        except Exception:  # noqa: BLE001
            pass
        return JobFailure(finished_ts=time.time())

    run_job_with_lifecycle(
        state=state,
        job_id=job_id,
        worker=worker,
        progress_1_on_done=False,
        start_progress=0.0,
        clear_message_on_start=True,
        on_error=on_error,
    )


__all__ = ['run_refraction_static_apply_job']
