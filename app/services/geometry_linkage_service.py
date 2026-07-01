"""Static geometry linkage background job service."""

from __future__ import annotations

from pathlib import Path
import time

from app.contracts.statics.geometry_linkage import StaticLinkageBuildRequest
from app.services.common.artifact_io import write_json_atomic
from app.core.state import AppState
from app.services.geometry_linkage_artifacts import (
    GEOMETRY_LINKAGE_CSV_NAME,
    GEOMETRY_LINKAGE_NPZ_NAME,
    GEOMETRY_LINKAGE_QC_JSON_NAME,
    GeometryLinkageArtifactMetadata,
    write_geometry_linkage_artifacts,
)
from app.services.geometry_linkage_linker import (
    GeometryLinkageOptions,
    build_geometry_linkage,
)
from app.services.geometry_linkage_loader import load_geometry_linkage_from_job_dir
from app.services.geometry_linkage_tables import build_endpoint_geometry_tables
from app.services.geometry_linkage_validation import (
    GeometryLinkageHeaderConfig,
    validate_geometry_linkage_headers,
)
from app.services.job_runner import (
    JobCancelledError,
    JobCompletion,
    JobFailure,
    ensure_job_not_cancelled,
    run_job_with_lifecycle,
)
from app.services.jobs import JobContext
from app.services.reader import get_reader


def _write_job_meta(
    *,
    job_id: str,
    job_dir: Path,
    req: StaticLinkageBuildRequest,
) -> None:
    write_json_atomic(
        job_dir / 'job_meta.json',
        {
            'job_id': job_id,
            'job_type': 'statics',
            'statics_kind': 'geometry_linkage',
            'source_file_id': req.file_id,
            'key1_byte': req.key1_byte,
            'key2_byte': req.key2_byte,
            'request': req.model_dump(mode='json'),
            'inputs': {
                'geometry': req.geometry.model_dump(mode='json'),
                'linkage': req.linkage.model_dump(mode='json'),
            },
            'artifacts': {
                'linkage_npz': GEOMETRY_LINKAGE_NPZ_NAME,
                'linkage_csv': GEOMETRY_LINKAGE_CSV_NAME,
                'qc_json': GEOMETRY_LINKAGE_QC_JSON_NAME,
            },
        },
        make_parent=True,
    )


def _reader_header_source_segy_path(reader: object) -> str | None:
    meta = getattr(reader, 'meta', None)
    if not isinstance(meta, dict):
        return None
    original = meta.get('original_segy_path')
    return original if isinstance(original, str) else None


def _run_geometry_linkage_build_job_body(
    job_id: str,
    req: StaticLinkageBuildRequest,
    state: AppState,
) -> JobCompletion | None:
    ctx = JobContext.resolve(state, job_id)
    job_dir = ctx.job_dir
    job_dir.mkdir(parents=True, exist_ok=True)

    ctx.set_progress_message(
        progress=0.02,
        message='preparing_geometry_linkage_job',
    )
    _write_job_meta(job_id=job_id, job_dir=job_dir, req=req)
    ensure_job_not_cancelled(state, job_id)

    ctx.set_progress_message(
        progress=0.05,
        message='resolving_trace_store_reader',
    )
    reader = get_reader(req.file_id, req.key1_byte, req.key2_byte, state=state)
    ensure_job_not_cancelled(state, job_id)

    ctx.set_progress_message(
        progress=0.15,
        message='validating_geometry_linkage_headers',
    )
    headers = validate_geometry_linkage_headers(
        reader=reader,
        config=GeometryLinkageHeaderConfig(
            source_x_byte=req.geometry.source_x_byte,
            source_y_byte=req.geometry.source_y_byte,
            receiver_x_byte=req.geometry.receiver_x_byte,
            receiver_y_byte=req.geometry.receiver_y_byte,
            coordinate_scalar_byte=req.geometry.coordinate_scalar_byte,
        ),
    )
    ensure_job_not_cancelled(state, job_id)

    ctx.set_progress_message(
        progress=0.30,
        message='building_endpoint_geometry_tables',
    )
    tables = build_endpoint_geometry_tables(
        headers,
        coordinate_unit=req.geometry.coordinate_unit,
    )
    ensure_job_not_cancelled(state, job_id)

    ctx.set_progress_message(
        progress=0.50,
        message='building_geometry_linkage',
    )
    linkage = build_geometry_linkage(
        tables,
        options=GeometryLinkageOptions(
            mode=req.linkage.mode,
            threshold_m=req.linkage.threshold_m,
            receiver_location_interval_m=req.linkage.receiver_location_interval_m,
            prefer_receiver_anchor=req.linkage.prefer_receiver_anchor,
        ),
    )
    ensure_job_not_cancelled(state, job_id)

    ctx.set_progress_message(
        progress=0.75,
        message='writing_geometry_linkage_artifacts',
    )
    ensure_job_not_cancelled(state, job_id)
    write_geometry_linkage_artifacts(
        job_dir=job_dir,
        tables=tables,
        linkage=linkage,
        metadata=GeometryLinkageArtifactMetadata(
            job_id=job_id,
            input_file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            source_x_byte=req.geometry.source_x_byte,
            source_y_byte=req.geometry.source_y_byte,
            receiver_x_byte=req.geometry.receiver_x_byte,
            receiver_y_byte=req.geometry.receiver_y_byte,
            coordinate_scalar_byte=req.geometry.coordinate_scalar_byte,
            header_source_segy_path=_reader_header_source_segy_path(reader),
        ),
    )
    ensure_job_not_cancelled(state, job_id)

    ctx.set_progress_message(
        progress=0.90,
        message='validating_geometry_linkage_artifacts',
    )
    load_geometry_linkage_from_job_dir(
        job_dir,
        expected_n_traces=tables.n_traces,
        expected_key1_byte=req.key1_byte,
        expected_key2_byte=req.key2_byte,
    )
    ensure_job_not_cancelled(state, job_id)

    ctx.set_progress_message(progress=1.0, message='done')
    return JobCompletion(finished_ts=time.time())


def _handle_geometry_linkage_job_error() -> JobFailure:
    return JobFailure(finished_ts=time.time())


def _handle_geometry_linkage_job_cancel(
    *,
    exc: JobCancelledError,
) -> JobCompletion:
    finished_ts = float(exc.finished_ts) if exc.finished_ts is not None else time.time()
    return JobCompletion(finished_ts=finished_ts)


def run_geometry_linkage_build_job(
    job_id: str,
    req: StaticLinkageBuildRequest,
    state: AppState,
) -> None:
    """Build geometry linkage artifacts for a statics job."""
    def worker() -> JobCompletion | None:
        return _run_geometry_linkage_build_job_body(
            job_id=job_id,
            req=req,
            state=state,
        )

    run_job_with_lifecycle(
        state=state,
        job_id=job_id,
        worker=worker,
        progress_1_on_done=True,
        start_progress=0.0,
        clear_message_on_start=True,
        on_error=lambda _exc: _handle_geometry_linkage_job_error(),
        on_cancel=lambda exc: _handle_geometry_linkage_job_cancel(
            exc=exc,
        ),
    )


__all__ = ['run_geometry_linkage_build_job']
