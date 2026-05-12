"""M5 refraction export request validation and artifact-writing job service."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from app.api.schemas import (
    REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS,
    RefractionStaticExportFormat,
    RefractionStaticExportJobRequest,
    RefractionStaticExportRequest,
)
from app.core.state import AppState
from app.services.job_manager import JobManager
from app.services.job_runner import JobCompletion, JobFailure, run_job_with_lifecycle
from app.services.refraction_static_artifacts import (
    CANONICAL_RECEIVER_STATIC_TABLE_CSV_NAME,
    CANONICAL_SOURCE_RECEIVER_STATIC_TABLE_CSV_NAME,
    CANONICAL_SOURCE_STATIC_TABLE_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_REQUEST_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    write_refraction_time_term_spreadsheet_csv_from_static_tables,
)
from app.services.refraction_static_export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)
from app.services.refraction_static_export_types import (
    RefractionStaticEndpointExportRow,
    RefractionStaticEndpointKind,
    RefractionStaticExportBundle,
)
from app.services.refraction_static_lsst_export import (
    REFRACTION_LSST_CARDS_TXT_NAME,
    REFRACTION_LSST_CSV_NAME,
    REFRACTION_LSST_PLUS_CARDS_TXT_NAME,
    REFRACTION_LSST_PLUS_CSV_NAME,
    write_refraction_lsst_cards_txt,
    write_refraction_lsst_csv,
    write_refraction_lsst_plus_cards_txt,
    write_refraction_lsst_plus_csv,
)
from app.services.refraction_static_table_validator import (
    CANONICAL_STATIC_TABLE_FORMAT_NAME,
    CANONICAL_STATIC_TABLE_FORMAT_VERSION,
    CANONICAL_STATIC_TABLE_OPTIONAL_COLUMNS,
    CANONICAL_STATIC_TABLE_REQUIRED_COLUMNS,
)

REFRACTION_STATIC_EXPORT_REQUEST_JSON_NAME = 'refraction_static_export_request.json'
REFRACTION_STATIC_EXPORT_JOB_META_JSON_NAME = 'job_meta.json'
REFRACTION_STATIC_EXPORT_DONE_MESSAGE = 'refraction_static_export_artifacts_written'

REFRACTION_STATIC_EXPORT_REGISTERED_ARTIFACT_NAMES = frozenset(
    {
        REFRACTION_STATIC_EXPORT_JOB_META_JSON_NAME,
        REFRACTION_STATIC_EXPORT_REQUEST_JSON_NAME,
        CANONICAL_SOURCE_STATIC_TABLE_CSV_NAME,
        CANONICAL_RECEIVER_STATIC_TABLE_CSV_NAME,
        CANONICAL_SOURCE_RECEIVER_STATIC_TABLE_CSV_NAME,
        REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
        REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
        REFRACTION_LSST_CARDS_TXT_NAME,
        REFRACTION_LSST_PLUS_CARDS_TXT_NAME,
        REFRACTION_LSST_CSV_NAME,
        REFRACTION_LSST_PLUS_CSV_NAME,
    }
)

_BASE_SOURCE_ARTIFACTS = (
    REFRACTION_STATIC_REQUEST_JSON_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
)

_FORMAT_SOURCE_ARTIFACTS: dict[
    RefractionStaticExportFormat,
    tuple[str, ...],
] = {
    'canonical_static_table': (
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
    ),
    'lsst': (
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
    ),
    'lsst_plus': (
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
    ),
    'time_term_spreadsheet': (
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    ),
    'first_break_time': (
        REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
    ),
}


class RefractionStaticExportSourceJobNotFound(ValueError):
    """Raised when a standalone export references an unknown source job."""


class RefractionStaticExportValidationError(ValueError):
    """Raised when a standalone export source job is not export-compatible."""


@dataclass(frozen=True)
class ResolvedRefractionStaticExportSourceJob:
    source_job_id: str
    source_file_id: str
    key1_byte: int
    key2_byte: int
    source_artifacts_dir: Path
    requested_formats: tuple[RefractionStaticExportFormat, ...]
    required_source_artifacts: tuple[str, ...]


def resolve_refraction_static_export_formats(
    export: RefractionStaticExportRequest,
) -> tuple[RefractionStaticExportFormat, ...]:
    """Resolve default M5 export formats without running any formatter."""
    if not bool(export.enabled):
        return ()
    if export.formats:
        return tuple(export.formats)
    return REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS


def required_refraction_static_export_source_artifacts(
    requested_formats: tuple[RefractionStaticExportFormat, ...],
) -> tuple[str, ...]:
    """Return source refraction artifacts required for the requested formats."""
    if not requested_formats:
        return ()
    names: list[str] = list(_BASE_SOURCE_ARTIFACTS)
    for export_format in requested_formats:
        names.extend(_FORMAT_SOURCE_ARTIFACTS[export_format])
    return tuple(dict.fromkeys(names))


def validate_refraction_static_export_source_job(
    *,
    req: RefractionStaticExportJobRequest,
    state: AppState,
) -> ResolvedRefractionStaticExportSourceJob:
    """Validate the completed source refraction job used by standalone export."""
    requested_formats = resolve_refraction_static_export_formats(req.export)
    required_artifacts = required_refraction_static_export_source_artifacts(
        requested_formats,
    )
    with state.lock:
        raw_job = state.jobs.get(req.source_job_id)
        job = dict(raw_job) if isinstance(raw_job, dict) else None
    if job is None:
        raise RefractionStaticExportSourceJobNotFound(
            f'source refraction job not found: {req.source_job_id}'
        )

    if job.get('job_type') != 'statics' or job.get('statics_kind') != 'refraction':
        raise RefractionStaticExportValidationError(
            'source_job_id must reference a refraction static job'
        )
    status = JobManager.normalize_status_value(job.get('status'))
    if status != 'done':
        raise RefractionStaticExportValidationError(
            'source_job_id must reference a completed refraction static job'
        )

    source_file_id = job.get('file_id')
    if not isinstance(source_file_id, str) or not source_file_id:
        raise RefractionStaticExportValidationError(
            'source refraction job metadata is missing file_id'
        )
    key1_byte = _job_int(job, 'key1_byte')
    key2_byte = _job_int(job, 'key2_byte')

    artifacts_dir_raw = job.get('artifacts_dir')
    if not isinstance(artifacts_dir_raw, str) or not artifacts_dir_raw:
        raise RefractionStaticExportValidationError(
            'source refraction job metadata is missing artifacts_dir'
        )
    artifacts_dir = Path(artifacts_dir_raw)
    if not artifacts_dir.is_dir():
        raise RefractionStaticExportValidationError(
            'source refraction job artifacts_dir is missing'
        )

    missing = [
        name for name in required_artifacts if not (artifacts_dir / name).is_file()
    ]
    if missing:
        missing_text = ', '.join(missing)
        raise RefractionStaticExportValidationError(
            'source refraction job is missing required export artifacts: '
            f'{missing_text}'
        )

    return ResolvedRefractionStaticExportSourceJob(
        source_job_id=req.source_job_id,
        source_file_id=source_file_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        source_artifacts_dir=artifacts_dir,
        requested_formats=requested_formats,
        required_source_artifacts=required_artifacts,
    )


def run_refraction_static_export_job(
    job_id: str,
    req: RefractionStaticExportJobRequest,
    state: AppState,
) -> None:
    """Run a standalone M5 export job.

    The job validates the completed source refraction artifacts, writes the
    requested export artifacts into its own job directory, and persists request
    metadata including resolved default formats.
    """

    def worker() -> JobCompletion:
        return _run_refraction_static_export_job_body(
            job_id=job_id,
            req=req,
            state=state,
        )

    run_job_with_lifecycle(
        state=state,
        job_id=job_id,
        worker=worker,
        progress_1_on_done=False,
        start_progress=0.0,
        clear_message_on_start=True,
        on_error=_handle_refraction_static_export_job_error,
    )


def _run_refraction_static_export_job_body(
    *,
    job_id: str,
    req: RefractionStaticExportJobRequest,
    state: AppState,
) -> JobCompletion:
    _set_job_progress_message(
        state,
        job_id,
        progress=0.10,
        message='validating_source_refraction_static_job',
    )
    source = validate_refraction_static_export_source_job(req=req, state=state)
    job_dir = _resolve_job_dir(state, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = _refraction_static_export_job_payload(
        job_id=job_id,
        req=req,
        source=source,
    )
    _set_job_progress_message(
        state,
        job_id,
        progress=0.60,
        message='writing_refraction_static_export_artifacts',
    )
    _write_requested_export_artifacts(
        job_dir=job_dir,
        export=req.export,
        source=source,
    )
    _set_job_progress_message(
        state,
        job_id,
        progress=0.80,
        message='writing_refraction_static_export_request',
    )
    _write_json_atomic(job_dir / REFRACTION_STATIC_EXPORT_JOB_META_JSON_NAME, payload)
    _write_json_atomic(job_dir / REFRACTION_STATIC_EXPORT_REQUEST_JSON_NAME, payload)
    _set_job_progress_message(
        state,
        job_id,
        progress=1.0,
        message=REFRACTION_STATIC_EXPORT_DONE_MESSAGE,
    )
    return JobCompletion(
        finished_ts=time.time(),
        message=REFRACTION_STATIC_EXPORT_DONE_MESSAGE,
    )


def _refraction_static_export_job_payload(
    *,
    job_id: str,
    req: RefractionStaticExportJobRequest,
    source: ResolvedRefractionStaticExportSourceJob,
) -> dict[str, Any]:
    return {
        'job_id': job_id,
        'job_type': 'statics',
        'statics_kind': 'refraction_export',
        'source_job_id': source.source_job_id,
        'source_file_id': source.source_file_id,
        'key1_byte': source.key1_byte,
        'key2_byte': source.key2_byte,
        'request': req.model_dump(mode='json'),
        'export': {
            'enabled': bool(req.export.enabled),
            'requested_formats': list(source.requested_formats),
            'units': req.export.units,
            'rounding_ms': req.export.rounding_ms,
            'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
            'include_inactive_endpoints': bool(req.export.include_inactive_endpoints),
            'include_legacy_alias_columns': bool(
                req.export.include_legacy_alias_columns
            ),
            'fail_on_invalid_static_status': bool(
                req.export.fail_on_invalid_static_status
            ),
        },
        'required_source_artifacts': list(source.required_source_artifacts),
        'generated_artifacts': list(
            _generated_refraction_static_export_artifacts(source.requested_formats)
        ),
        'source_artifacts_dir': str(source.source_artifacts_dir),
    }


def _generated_refraction_static_export_artifacts(
    requested_formats: tuple[RefractionStaticExportFormat, ...],
) -> tuple[str, ...]:
    names: list[str] = []
    if 'canonical_static_table' in requested_formats:
        names.extend(
            (
                CANONICAL_SOURCE_STATIC_TABLE_CSV_NAME,
                CANONICAL_RECEIVER_STATIC_TABLE_CSV_NAME,
                CANONICAL_SOURCE_RECEIVER_STATIC_TABLE_CSV_NAME,
            )
        )
    if 'time_term_spreadsheet' in requested_formats:
        names.append(REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME)
    if 'lsst' in requested_formats:
        names.extend((REFRACTION_LSST_CARDS_TXT_NAME, REFRACTION_LSST_CSV_NAME))
    if 'lsst_plus' in requested_formats:
        names.extend(
            (REFRACTION_LSST_PLUS_CARDS_TXT_NAME, REFRACTION_LSST_PLUS_CSV_NAME)
        )
    if 'first_break_time' in requested_formats:
        names.append(REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME)
    return tuple(names)


def _write_requested_export_artifacts(
    *,
    job_dir: Path,
    export: RefractionStaticExportRequest,
    source: ResolvedRefractionStaticExportSourceJob,
) -> None:
    if 'canonical_static_table' in source.requested_formats:
        _write_canonical_static_table_exports(source=source, job_dir=job_dir)
    if 'time_term_spreadsheet' in source.requested_formats:
        _write_time_term_spreadsheet_export(
            source=source,
            job_dir=job_dir,
            include_inactive_endpoints=bool(export.include_inactive_endpoints),
        )
    if 'first_break_time' in source.requested_formats:
        _copy_first_break_time_export(source=source, job_dir=job_dir)
    if (
        'lsst' not in source.requested_formats
        and 'lsst_plus' not in source.requested_formats
    ):
        return
    bundle = _load_lsst_export_bundle(source)
    if 'lsst' in source.requested_formats:
        write_refraction_lsst_csv(
            bundle,
            job_dir / REFRACTION_LSST_CSV_NAME,
            fail_on_invalid_static_status=bool(export.fail_on_invalid_static_status),
            include_inactive_endpoints=bool(export.include_inactive_endpoints),
        )
        write_refraction_lsst_cards_txt(
            bundle,
            job_dir / REFRACTION_LSST_CARDS_TXT_NAME,
            fail_on_invalid_static_status=bool(export.fail_on_invalid_static_status),
            include_inactive_endpoints=bool(export.include_inactive_endpoints),
        )
    if 'lsst_plus' in source.requested_formats:
        write_refraction_lsst_plus_csv(
            bundle,
            job_dir / REFRACTION_LSST_PLUS_CSV_NAME,
            fail_on_invalid_static_status=bool(export.fail_on_invalid_static_status),
            include_inactive_endpoints=bool(export.include_inactive_endpoints),
        )
        write_refraction_lsst_plus_cards_txt(
            bundle,
            job_dir / REFRACTION_LSST_PLUS_CARDS_TXT_NAME,
            fail_on_invalid_static_status=bool(export.fail_on_invalid_static_status),
            include_inactive_endpoints=bool(export.include_inactive_endpoints),
        )


def write_refraction_static_requested_export_artifacts(
    *,
    job_dir: Path,
    source_artifacts_dir: Path,
    source_job_id: str,
    source_file_id: str,
    key1_byte: int,
    key2_byte: int,
    requested_formats: tuple[RefractionStaticExportFormat, ...],
    export: RefractionStaticExportRequest,
) -> tuple[str, ...]:
    """Write requested M5 export artifacts for a completed refraction job."""
    if not requested_formats:
        return ()
    source = ResolvedRefractionStaticExportSourceJob(
        source_job_id=source_job_id,
        source_file_id=source_file_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        source_artifacts_dir=Path(source_artifacts_dir),
        requested_formats=requested_formats,
        required_source_artifacts=required_refraction_static_export_source_artifacts(
            requested_formats
        ),
    )
    _write_requested_export_artifacts(
        job_dir=Path(job_dir),
        export=export,
        source=source,
    )
    generated = _generated_refraction_static_export_artifacts(requested_formats)
    _register_generated_refraction_export_artifacts_in_manifest(Path(job_dir), generated)
    return generated


def _write_canonical_static_table_exports(
    *,
    source: ResolvedRefractionStaticExportSourceJob,
    job_dir: Path,
) -> None:
    source_rows = _canonical_static_table_rows(
        _load_static_table_rows(source.source_artifacts_dir / SOURCE_STATIC_TABLE_CSV_NAME),
        endpoint_kind='source',
        source_job_id=source.source_job_id,
    )
    receiver_rows = _canonical_static_table_rows(
        _load_static_table_rows(
            source.source_artifacts_dir / RECEIVER_STATIC_TABLE_CSV_NAME
        ),
        endpoint_kind='receiver',
        source_job_id=source.source_job_id,
    )
    _write_canonical_static_table_csv(
        job_dir / CANONICAL_SOURCE_STATIC_TABLE_CSV_NAME,
        source_rows,
    )
    _write_canonical_static_table_csv(
        job_dir / CANONICAL_RECEIVER_STATIC_TABLE_CSV_NAME,
        receiver_rows,
    )
    _write_canonical_static_table_csv(
        job_dir / CANONICAL_SOURCE_RECEIVER_STATIC_TABLE_CSV_NAME,
        (*source_rows, *receiver_rows),
    )


_CANONICAL_STATIC_TABLE_COLUMNS = tuple(
    dict.fromkeys(
        CANONICAL_STATIC_TABLE_REQUIRED_COLUMNS
        + CANONICAL_STATIC_TABLE_OPTIONAL_COLUMNS
    )
)


def _canonical_static_table_rows(
    rows: tuple[dict[str, str | None], ...],
    *,
    endpoint_kind: RefractionStaticEndpointKind,
    source_job_id: str,
) -> tuple[dict[str, str], ...]:
    return tuple(
        _canonical_static_table_row(
            row,
            endpoint_kind=endpoint_kind,
            source_job_id=source_job_id,
        )
        for row in rows
    )


def _canonical_static_table_row(
    row: dict[str, str | None],
    *,
    endpoint_kind: RefractionStaticEndpointKind,
    source_job_id: str,
) -> dict[str, str]:
    prefix = 'source' if endpoint_kind == 'source' else 'receiver'
    out = {column: '' for column in _CANONICAL_STATIC_TABLE_COLUMNS}
    out.update(
        {
            'format_name': CANONICAL_STATIC_TABLE_FORMAT_NAME,
            'format_version': str(CANONICAL_STATIC_TABLE_FORMAT_VERSION),
            'source_job_id': _optional_row_text(row.get('source_job_id'))
            or source_job_id,
            'endpoint_kind': endpoint_kind,
            'endpoint_key': _required_row_text(row, f'{prefix}_endpoint_key'),
            'endpoint_id': _optional_row_text(row.get(f'{prefix}_id')) or '',
            'applied_shift_ms': _required_row_text(row, 'total_applied_shift_ms'),
            'static_status': _required_row_text(row, 'static_status'),
            'sign_convention': _required_row_text(row, 'sign_convention'),
        }
    )
    direct_columns = set(CANONICAL_STATIC_TABLE_OPTIONAL_COLUMNS)
    for column in direct_columns:
        value = _optional_row_text(row.get(column))
        if value is not None:
            out[column] = value
    out['node_id'] = _optional_row_text(row.get(f'{prefix}_node_id')) or out['node_id']
    if endpoint_kind == 'source':
        out['source_id'] = _optional_row_text(row.get('source_id')) or out['source_id']
    else:
        out['receiver_id'] = (
            _optional_row_text(row.get('receiver_id')) or out['receiver_id']
        )
    return out


def _write_canonical_static_table_csv(
    path: Path,
    rows: tuple[dict[str, str], ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(_CANONICAL_STATIC_TABLE_COLUMNS),
                lineterminator='\n',
            )
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _register_generated_refraction_export_artifacts_in_manifest(
    job_dir: Path,
    generated_artifacts: tuple[str, ...],
) -> None:
    if not generated_artifacts:
        return
    manifest_path = job_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME
    if not manifest_path.is_file():
        return
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    artifacts = payload.get('artifacts')
    if not isinstance(artifacts, list):
        return
    existing = {
        item.get('name')
        for item in artifacts
        if isinstance(item, dict) and isinstance(item.get('name'), str)
    }
    for name in generated_artifacts:
        if name in existing:
            continue
        artifacts.append(
            {
                'name': name,
                'kind': _artifact_kind_from_name(name),
                'required': True,
                'origin': 'export',
            }
        )
        existing.add(name)
    _write_json_atomic(manifest_path, payload)


def _artifact_kind_from_name(name: str) -> str:
    suffix = Path(name).suffix.lower().lstrip('.')
    if suffix:
        return suffix
    return 'file'


def _write_time_term_spreadsheet_export(
    *,
    source: ResolvedRefractionStaticExportSourceJob,
    job_dir: Path,
    include_inactive_endpoints: bool,
) -> None:
    write_refraction_time_term_spreadsheet_csv_from_static_tables(
        source_rows=_load_static_table_rows(
            source.source_artifacts_dir / SOURCE_STATIC_TABLE_CSV_NAME,
        ),
        receiver_rows=_load_static_table_rows(
            source.source_artifacts_dir / RECEIVER_STATIC_TABLE_CSV_NAME,
        ),
        path=job_dir / REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
        source_job_id=source.source_job_id,
        include_inactive_endpoints=include_inactive_endpoints,
    )


def _copy_first_break_time_export(
    *,
    source: ResolvedRefractionStaticExportSourceJob,
    job_dir: Path,
) -> None:
    source_path = (
        source.source_artifacts_dir / REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME
    )
    dest_path = job_dir / REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(f'{dest_path.name}.{uuid4().hex}.tmp')
    try:
        with source_path.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _load_lsst_export_bundle(
    source: ResolvedRefractionStaticExportSourceJob,
) -> RefractionStaticExportBundle:
    return RefractionStaticExportBundle(
        source_job_id=source.source_job_id,
        source_rows=_load_lsst_endpoint_rows(
            source.source_artifacts_dir / SOURCE_STATIC_TABLE_CSV_NAME,
            endpoint_kind='source',
        ),
        receiver_rows=_load_lsst_endpoint_rows(
            source.source_artifacts_dir / RECEIVER_STATIC_TABLE_CSV_NAME,
            endpoint_kind='receiver',
        ),
    )


def _load_lsst_endpoint_rows(
    path: Path,
    *,
    endpoint_kind: RefractionStaticEndpointKind,
) -> tuple[RefractionStaticEndpointExportRow, ...]:
    rows = _load_static_table_rows(path)
    return tuple(_lsst_endpoint_row(row, endpoint_kind=endpoint_kind) for row in rows)


def _load_static_table_rows(path: Path) -> tuple[dict[str, str | None], ...]:
    with Path(path).open('r', encoding='utf-8-sig', newline='') as handle:
        return tuple(csv.DictReader(handle))


def _lsst_endpoint_row(
    row: dict[str, str | None],
    *,
    endpoint_kind: RefractionStaticEndpointKind,
) -> RefractionStaticEndpointExportRow:
    if row.get('sign_convention') != REFRACTION_STATIC_REPO_SIGN_CONVENTION:
        raise RefractionStaticExportValidationError(
            f'{endpoint_kind} static table sign_convention must be '
            f'{REFRACTION_STATIC_REPO_SIGN_CONVENTION!r}'
        )
    if row.get('endpoint_kind') != endpoint_kind:
        raise RefractionStaticExportValidationError(
            f'{endpoint_kind} static table contains endpoint_kind '
            f'{row.get("endpoint_kind")!r}'
        )
    prefix = 'source' if endpoint_kind == 'source' else 'receiver'
    return RefractionStaticEndpointExportRow(
        endpoint_kind=endpoint_kind,
        endpoint_key=_required_row_text(row, f'{prefix}_endpoint_key'),
        endpoint_id=_optional_row_text(row.get(f'{prefix}_id')),
        station_id=_optional_row_text(row.get(f'{prefix}_id')),
        node_id=_optional_int(row.get(f'{prefix}_node_id')),
        x_m=_optional_float(row.get('x_m')),
        y_m=_optional_float(row.get('y_m')),
        elevation_m=_optional_float(row.get('surface_elevation_m')),
        t1_s=_optional_ms_to_s(row.get('t1_ms')),
        t2_s=_optional_ms_to_s(row.get('t2_ms')),
        t3_s=_optional_ms_to_s(row.get('t3_ms')),
        v1_m_s=_optional_float(row.get('v1_m_s')),
        v2_m_s=_optional_float(row.get('v2_m_s')),
        v3_m_s=_optional_float(row.get('v3_m_s')),
        vsub_m_s=_optional_float(row.get('vsub_m_s')),
        sh1_m=_optional_float(row.get('sh1_weathering_thickness_m')),
        sh2_m=_optional_float(row.get('sh2_weathering_thickness_m')),
        sh3_m=_optional_float(row.get('sh3_weathering_thickness_m')),
        total_weathering_thickness_m=_optional_float(
            row.get('total_weathering_thickness_m')
        ),
        weathering_correction_s=_optional_ms_to_s(
            row.get('weathering_correction_ms')
        ),
        elevation_correction_s=_optional_ms_to_s(row.get('elevation_correction_ms')),
        field_correction_s=_optional_ms_to_s(row.get(f'{prefix}_field_shift_ms')),
        source_depth_m=_optional_float(row.get('source_depth_m')),
        source_depth_shift_s=_optional_ms_to_s(row.get('source_depth_shift_ms')),
        source_depth_status=_optional_row_text(row.get('source_depth_status')),
        uphole_time_s=_optional_ms_to_s(row.get('uphole_time_ms')),
        uphole_shift_s=_optional_ms_to_s(row.get('uphole_shift_ms')),
        uphole_status=_optional_row_text(row.get('uphole_status')),
        manual_static_shift_s=_optional_ms_to_s(row.get('manual_static_shift_ms')),
        manual_static_status=_optional_row_text(row.get('manual_static_status')),
        field_static_status=_optional_row_text(row.get(f'{prefix}_field_static_status')),
        total_with_field_shift_s=_optional_ms_to_s(
            row.get(f'{prefix}_total_with_field_shift_ms')
        ),
        total_applied_shift_s=_optional_ms_to_s(row.get('total_applied_shift_ms')),
        static_status=_required_row_text(row, 'static_status'),
    )


def _required_row_text(row: dict[str, str | None], column: str) -> str:
    text = _optional_row_text(row.get(column))
    if text is None:
        raise RefractionStaticExportValidationError(
            f'static artifact is missing required value {column}'
        )
    return text


def _optional_row_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_float(value: object) -> float | None:
    text = _optional_row_text(value)
    if text is None:
        return None
    return float(text)


def _optional_int(value: object) -> int | None:
    text = _optional_row_text(value)
    if text is None:
        return None
    return int(text)


def _optional_ms_to_s(value: object) -> float | None:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    return parsed / 1000.0


def _job_int(job: dict[str, object], field: str) -> int:
    value = job.get(field)
    if not isinstance(value, int):
        raise RefractionStaticExportValidationError(
            f'source refraction job metadata is missing {field}'
        )
    return int(value)


def _resolve_job_dir(state: AppState, job_id: str) -> Path:
    with state.lock:
        job = state.jobs.get(job_id)
        artifacts_dir = job.get('artifacts_dir') if isinstance(job, dict) else None
    if not isinstance(artifacts_dir, str) or not artifacts_dir:
        raise ValueError('job artifacts_dir is not available')
    return Path(artifacts_dir)


def _set_job_progress_message(
    state: AppState,
    job_id: str,
    *,
    progress: float,
    message: str,
) -> None:
    with state.lock:
        if state.jobs.get(job_id) is None:
            return
        state.jobs.set_progress(job_id, progress)
        state.jobs.set_message(job_id, message)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding='utf-8',
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _handle_refraction_static_export_job_error(_exc: Exception) -> JobFailure:
    return JobFailure(finished_ts=time.time())


__all__ = [
    'REFRACTION_STATIC_EXPORT_DONE_MESSAGE',
    'REFRACTION_STATIC_EXPORT_JOB_META_JSON_NAME',
    'REFRACTION_STATIC_EXPORT_REGISTERED_ARTIFACT_NAMES',
    'REFRACTION_STATIC_EXPORT_REQUEST_JSON_NAME',
    'RefractionStaticExportSourceJobNotFound',
    'RefractionStaticExportValidationError',
    'ResolvedRefractionStaticExportSourceJob',
    'required_refraction_static_export_source_artifacts',
    'resolve_refraction_static_export_formats',
    'run_refraction_static_export_job',
    'validate_refraction_static_export_source_job',
    'write_refraction_static_requested_export_artifacts',
]
