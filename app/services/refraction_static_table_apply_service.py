"""Standalone M5 static-table apply job service."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from app.api.schemas import (
    RefractionStaticGeometryRequest,
    RefractionStaticTableApplyRequest,
)
from app.core.state import AppState
from app.services.corrected_trace_store import (
    TimeShiftedTraceStoreResult,
    build_time_shifted_trace_store,
)
from app.services.job_artifact_refs import resolve_job_artifact_path
from app.services.job_runner import JobCompletion, JobFailure, run_job_with_lifecycle
from app.services.reader import get_reader
from app.services.refraction_static_artifacts import (
    REFRACTION_STATIC_HISTORY_JSON_NAME,
    REFRACTION_STATIC_REQUEST_JSON_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    static_history_double_application_qc,
)
from app.services.refraction_static_apply_trace_store import (
    CORRECTED_FILE_JSON_NAME,
    RefractionStaticTraceStoreApplyError,
    validate_refraction_trace_shifts_for_application,
)
from app.services.refraction_static_export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)
from app.services.refraction_static_table_import import (
    RefractionImportedEndpointStatic,
    RefractionStaticTableImportResult,
    import_refraction_static_table_csv,
    import_refraction_static_tables,
)
from app.services.trace_store_index_validation import validate_sorted_to_original
from app.services.trace_store_registration import register_trace_store, trace_store_cache_key
from app.utils.segy_scalars import apply_segy_scalar, normalize_elevation_unit

STATIC_TABLE_APPLY_REQUEST_JSON_NAME = 'static_table_apply_request.json'
STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME = 'static_table_apply_solution.npz'
STATIC_TABLE_APPLY_QC_JSON_NAME = 'static_table_apply_qc.json'
STATIC_TABLE_APPLY_TRACE_SHIFTS_CSV_NAME = 'static_table_apply_trace_shifts.csv'
STATIC_TABLE_APPLY_HISTORY_JSON_NAME = 'static_table_apply_history.json'
STATIC_TABLE_APPLY_REFRACTION_HISTORY_JSON_NAME = REFRACTION_STATIC_HISTORY_JSON_NAME

_DONE_MESSAGE = 'static_table_apply_artifacts_written'
_CORRECTED_DONE_MESSAGE = 'static_table_apply_corrected_trace_store_registered'
_ARTIFACT_ID_RE = re.compile(r'^(?P<job_id>[^:]+):(?P<name>[^:]+)$')
_SAFE_STORE_NAME_RE = re.compile(r'[^A-Za-z0-9_.-]+')


EndpointIdentityMode = Literal['endpoint_key', 'endpoint_id']


@dataclass(frozen=True)
class _ResolvedTablePaths:
    source_table_path: Path | None
    receiver_table_path: Path | None
    combined_table_path: Path | None
    artifact_ids: dict[str, str]
    source_job_id: str | None
    artifact_job_ids: tuple[str, ...]


@dataclass(frozen=True)
class _TraceEndpointKeys:
    sorted_trace_index: np.ndarray
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_endpoint_id_sorted: np.ndarray
    receiver_endpoint_id_sorted: np.ndarray
    source_identity_sorted: np.ndarray
    receiver_identity_sorted: np.ndarray


@dataclass(frozen=True)
class _StaticTableTraceShift:
    sorted_trace_index: np.ndarray
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_endpoint_id_sorted: np.ndarray
    receiver_endpoint_id_sorted: np.ndarray
    source_identity_sorted: np.ndarray
    receiver_identity_sorted: np.ndarray
    source_static_shift_s_sorted: np.ndarray
    receiver_static_shift_s_sorted: np.ndarray
    trace_shift_s_sorted: np.ndarray
    trace_static_status_sorted: np.ndarray
    trace_static_valid_mask_sorted: np.ndarray
    missing_source_identity: tuple[str, ...]
    missing_receiver_identity: tuple[str, ...]
    source_identity_mode: EndpointIdentityMode
    receiver_identity_mode: EndpointIdentityMode


@dataclass(frozen=True)
class _StaticTableApplyLineage:
    source_table_digest: str | None
    receiver_table_digest: str | None
    combined_table_digest: str
    trace_shift_s_sorted_digest: str
    applied_component_name: str
    created_from_refraction_job_id: str | None
    created_from_export_job_id: str | None


def run_refraction_static_table_apply_job(
    job_id: str,
    req: RefractionStaticTableApplyRequest,
    state: AppState,
) -> None:
    """Run the standalone static-table apply job lifecycle."""

    def worker() -> JobCompletion:
        return _run_refraction_static_table_apply_job_body(
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
        on_error=_handle_static_table_apply_error,
    )


def _run_refraction_static_table_apply_job_body(
    *,
    job_id: str,
    req: RefractionStaticTableApplyRequest,
    state: AppState,
) -> JobCompletion:
    request = RefractionStaticTableApplyRequest.model_validate(req)
    job_dir = _resolve_job_dir(state, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.05,
        message='writing_static_table_apply_request',
    )
    _write_json_atomic(
        job_dir / STATIC_TABLE_APPLY_REQUEST_JSON_NAME,
        {
            'job_id': job_id,
            'job_type': 'statics',
            'statics_kind': 'refraction_static_table_apply',
            'source_file_id': request.file_id,
            'key1_byte': int(request.key1_byte),
            'key2_byte': int(request.key2_byte),
            'request': request.model_dump(mode='json'),
        },
    )

    _set_job_progress_message(
        state,
        job_id,
        progress=0.15,
        message='validating_static_table_artifacts',
    )
    table_paths = _resolve_table_paths(state=state, req=request)
    imported = _import_static_tables(table_paths)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.35,
        message='mapping_static_table_to_traces',
    )
    endpoint_keys = _load_trace_endpoint_keys(
        req=request,
        state=state,
        table_paths=table_paths,
        imported=imported,
    )
    trace_shift = _build_trace_shift(
        req=request,
        imported=imported,
        endpoint_keys=endpoint_keys,
    )
    _validate_imported_endpoint_static_shifts(
        imported,
        max_abs_shift_ms=float(request.max_abs_shift_ms),
    )
    validate_refraction_trace_shifts_for_application(
        trace_shift_s_sorted=trace_shift.trace_shift_s_sorted,
        trace_static_valid_mask_sorted=trace_shift.trace_static_valid_mask_sorted,
        trace_static_status_sorted=trace_shift.trace_static_status_sorted,
        n_traces=int(trace_shift.trace_shift_s_sorted.shape[0]),
        max_abs_shift_ms=float(request.max_abs_shift_ms),
        require_all_traces_valid=True,
    )
    table_lineage = _static_table_apply_lineage(
        req=request,
        state=state,
        imported=imported,
        table_paths=table_paths,
        trace_shift=trace_shift,
    )
    table_reapply_qc = _static_table_reapply_qc(
        req=request,
        state=state,
        table_lineage=table_lineage,
    )
    _enforce_static_table_reapply_guard(table_reapply_qc)
    double_application_qc = _double_application_qc(
        req=request,
        state=state,
    )
    _enforce_double_application_policy(double_application_qc)

    _set_job_progress_message(
        state,
        job_id,
        progress=0.65,
        message='writing_static_table_apply_artifacts',
    )
    artifacts = _write_static_table_apply_artifacts(
        job_dir=job_dir,
        job_id=job_id,
        req=request,
        imported=imported,
        table_paths=table_paths,
        trace_shift=trace_shift,
        table_lineage=table_lineage,
        table_reapply_qc=table_reapply_qc,
        double_application_qc=double_application_qc,
        validation_status='ok',
        corrected_file_id=None,
    )

    corrected_file_id: str | None = None
    corrected_store_path: Path | None = None
    if request.register_corrected_file:
        _set_job_progress_message(
            state,
            job_id,
            progress=0.78,
            message='applying_static_table_trace_shift',
        )
        corrected_file_id, corrected_store_path = _register_corrected_trace_store(
            state=state,
            job_id=job_id,
            req=request,
            imported=imported,
            table_paths=table_paths,
            job_dir=job_dir,
            trace_shift=trace_shift,
            table_lineage=table_lineage,
            table_reapply_qc=table_reapply_qc,
            double_application_qc=double_application_qc,
            artifact_names=artifacts,
        )
        _write_static_table_apply_qc(
            path=job_dir / STATIC_TABLE_APPLY_QC_JSON_NAME,
            job_id=job_id,
            req=request,
            imported=imported,
            table_paths=table_paths,
            trace_shift=trace_shift,
            table_lineage=table_lineage,
            table_reapply_qc=table_reapply_qc,
            double_application_qc=double_application_qc,
            validation_status='ok',
            corrected_file_id=corrected_file_id,
        )
        history = _write_static_table_apply_history(
            path=job_dir / STATIC_TABLE_APPLY_HISTORY_JSON_NAME,
            job_id=job_id,
            req=request,
            imported=imported,
            table_paths=table_paths,
            trace_shift=trace_shift,
            table_lineage=table_lineage,
            table_reapply_qc=table_reapply_qc,
            double_application_qc=double_application_qc,
            corrected_file_id=corrected_file_id,
        )
        _write_json_atomic(
            job_dir / STATIC_TABLE_APPLY_REFRACTION_HISTORY_JSON_NAME,
            history,
        )
        with state.lock:
            state.jobs.set_static_corrected_file(
                job_id,
                corrected_file_id=corrected_file_id,
                corrected_store_path=str(corrected_store_path),
            )
        _set_job_progress_message(
            state,
            job_id,
            progress=1.0,
            message=_CORRECTED_DONE_MESSAGE,
        )
        return JobCompletion(finished_ts=time.time(), message=_CORRECTED_DONE_MESSAGE)

    _set_job_progress_message(
        state,
        job_id,
        progress=1.0,
        message=_DONE_MESSAGE,
    )
    return JobCompletion(finished_ts=time.time(), message=_DONE_MESSAGE)


def _resolve_table_paths(
    *,
    state: AppState,
    req: RefractionStaticTableApplyRequest,
) -> _ResolvedTablePaths:
    artifact_ids: dict[str, str] = {}
    if req.combined_table_artifact_id is not None:
        artifact_ids['combined_table_artifact_id'] = req.combined_table_artifact_id
        source_job_id, _artifact_name = _parse_artifact_id(
            req.combined_table_artifact_id
        )
        combined_path = _resolve_table_artifact(
            state,
            artifact_id=req.combined_table_artifact_id,
        )
        return _ResolvedTablePaths(
            source_table_path=None,
            receiver_table_path=None,
            combined_table_path=combined_path,
            artifact_ids=artifact_ids,
            source_job_id=source_job_id,
            artifact_job_ids=(source_job_id,),
        )

    assert req.source_table_artifact_id is not None
    assert req.receiver_table_artifact_id is not None
    artifact_ids['source_table_artifact_id'] = req.source_table_artifact_id
    artifact_ids['receiver_table_artifact_id'] = req.receiver_table_artifact_id
    source_job_id, _source_name = _parse_artifact_id(req.source_table_artifact_id)
    receiver_job_id, _receiver_name = _parse_artifact_id(req.receiver_table_artifact_id)
    source_path = _resolve_table_artifact(
        state,
        artifact_id=req.source_table_artifact_id,
    )
    receiver_path = _resolve_table_artifact(
        state,
        artifact_id=req.receiver_table_artifact_id,
    )
    return _ResolvedTablePaths(
        source_table_path=source_path,
        receiver_table_path=receiver_path,
        combined_table_path=None,
        artifact_ids=artifact_ids,
        source_job_id=source_job_id if source_job_id == receiver_job_id else None,
        artifact_job_ids=tuple(dict.fromkeys((source_job_id, receiver_job_id))),
    )


def _parse_artifact_id(artifact_id: str) -> tuple[str, str]:
    match = _ARTIFACT_ID_RE.match(artifact_id)
    if match is None:
        raise ValueError(
            'static table artifact id must have the form '
            '<job_id>:<artifact_name>'
        )
    return match.group('job_id'), match.group('name')


def _resolve_table_artifact(state: AppState, *, artifact_id: str) -> Path:
    job_id, name = _parse_artifact_id(artifact_id)
    return resolve_job_artifact_path(
        state,
        job_id=job_id,
        name=name,
        allowed_job_types={'statics'},
        reference_label='static_table_apply',
    )


def _import_static_tables(
    table_paths: _ResolvedTablePaths,
) -> RefractionStaticTableImportResult:
    if table_paths.combined_table_path is not None:
        if _is_source_receiver_static_table_npz(table_paths.combined_table_path):
            imported = _import_source_receiver_static_table_npz(
                table_paths.combined_table_path,
                source_job_id=table_paths.source_job_id,
            )
        else:
            imported = import_refraction_static_table_csv(table_paths.combined_table_path)
    else:
        assert table_paths.source_table_path is not None
        assert table_paths.receiver_table_path is not None
        imported = import_refraction_static_tables(
            source_table_path=table_paths.source_table_path,
            receiver_table_path=table_paths.receiver_table_path,
        )
    if imported.is_valid:
        return imported
    joined = '; '.join(imported.errors)
    raise ValueError(f'static table import failed: {joined}')


def _is_source_receiver_static_table_npz(path: Path) -> bool:
    return (
        path.name == SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME
        or path.suffix.lower() == '.npz'
    )


def _import_source_receiver_static_table_npz(
    path: Path,
    *,
    source_job_id: str | None,
) -> RefractionStaticTableImportResult:
    source_name = Path(path).name
    try:
        with np.load(path, allow_pickle=False) as data:
            sign_convention = _npz_scalar_string(data, 'sign_convention')
            if sign_convention != REFRACTION_STATIC_REPO_SIGN_CONVENTION:
                raise ValueError(
                    f'{source_name} sign_convention must be '
                    f'{REFRACTION_STATIC_REPO_SIGN_CONVENTION!r}'
                )
            source_rows = _import_npz_endpoint_static_rows(
                data,
                endpoint_kind='source',
                source_job_id=source_job_id,
                source_name=source_name,
            )
            receiver_rows = _import_npz_endpoint_static_rows(
                data,
                endpoint_kind='receiver',
                source_job_id=source_job_id,
                source_name=source_name,
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f'failed to import static table NPZ: {source_name}') from exc

    return RefractionStaticTableImportResult(
        source_static_by_endpoint_key=source_rows,
        receiver_static_by_endpoint_key=receiver_rows,
        n_source_rows=len(source_rows),
        n_receiver_rows=len(receiver_rows),
        warnings=(),
        errors=(),
        schema_version=1,
        sign_convention=REFRACTION_STATIC_REPO_SIGN_CONVENTION,
    )


def _import_npz_endpoint_static_rows(
    data: Any,
    *,
    endpoint_kind: Literal['source', 'receiver'],
    source_job_id: str | None,
    source_name: str,
) -> dict[str, RefractionImportedEndpointStatic]:
    prefix = endpoint_kind
    endpoint_key = _npz_1d_string_array(data, f'{prefix}_endpoint_key')
    endpoint_id = _npz_1d_string_array(
        data,
        f'{prefix}_id',
        expected_shape=endpoint_key.shape,
    )
    applied_shift_s = _npz_1d_float_array(
        data,
        f'{prefix}_total_applied_shift_s',
        expected_shape=endpoint_key.shape,
    )
    static_status = _npz_1d_string_array(
        data,
        f'{prefix}_static_status',
        expected_shape=endpoint_key.shape,
    )

    rows: dict[str, RefractionImportedEndpointStatic] = {}
    for index, key_value in enumerate(endpoint_key.tolist()):
        key = str(key_value)
        if not key:
            raise ValueError(
                f'{source_name} {endpoint_kind} row {index + 1}: '
                'endpoint_key must be non-empty'
            )
        if key in rows:
            raise ValueError(
                f'{source_name}: duplicate {endpoint_kind} endpoint_key {key!r}'
            )
        rows[key] = RefractionImportedEndpointStatic(
            endpoint_kind=endpoint_kind,
            endpoint_key=key,
            endpoint_id=str(endpoint_id[index]),
            applied_shift_s=float(applied_shift_s[index]),
            static_status=str(static_status[index]),
            source_job_id=str(source_job_id or ''),
            row_number=index + 1,
            source_name=source_name,
            metadata={},
        )
    return rows


def _npz_scalar_string(data: Any, name: str) -> str:
    if name not in data.files:
        raise ValueError(f'static table NPZ is missing required array {name!r}')
    arr = np.asarray(data[name])
    if arr.shape not in {(), (1,)}:
        raise ValueError(f'static table NPZ array {name!r} must be scalar')
    return str(arr.reshape(-1)[0])


def _npz_1d_string_array(
    data: Any,
    name: str,
    *,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if name not in data.files:
        raise ValueError(f'static table NPZ is missing required array {name!r}')
    arr = np.asarray(data[name])
    if arr.ndim != 1:
        raise ValueError(f'static table NPZ array {name!r} must be 1D')
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f'static table NPZ array {name!r} shape mismatch: '
            f'expected {expected_shape}, got {arr.shape}'
        )
    return _string_array(arr)


def _npz_1d_float_array(
    data: Any,
    name: str,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if name not in data.files:
        raise ValueError(f'static table NPZ is missing required array {name!r}')
    arr = np.asarray(data[name], dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f'static table NPZ array {name!r} must be 1D')
    if arr.shape != expected_shape:
        raise ValueError(
            f'static table NPZ array {name!r} shape mismatch: '
            f'expected {expected_shape}, got {arr.shape}'
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


def _trace_endpoint_geometry(
    req: RefractionStaticTableApplyRequest,
    *,
    state: AppState,
    table_paths: _ResolvedTablePaths,
    imported: RefractionStaticTableImportResult,
) -> RefractionStaticGeometryRequest:
    if 'geometry' not in req.model_fields_set:
        producer_geometry = _producer_trace_endpoint_geometry(
            state=state,
            table_paths=table_paths,
            imported=imported,
        )
        if producer_geometry is not None:
            return producer_geometry
    return RefractionStaticGeometryRequest.model_validate(req.geometry)


def _load_trace_endpoint_keys(
    *,
    req: RefractionStaticTableApplyRequest,
    state: AppState,
    table_paths: _ResolvedTablePaths,
    imported: RefractionStaticTableImportResult,
) -> _TraceEndpointKeys:
    reader = get_reader(req.file_id, req.key1_byte, req.key2_byte, state=state)
    n_traces = _reader_n_traces(reader)
    sorted_trace_index = validate_sorted_to_original(
        reader.get_sorted_to_original(),
        expected_n_traces=n_traces,
        role='static-table target TraceStore',
    )
    geometry = _trace_endpoint_geometry(
        req,
        state=state,
        table_paths=table_paths,
        imported=imported,
    )
    headers = {
        byte: _reader_header(reader, byte, n_traces=n_traces)
        for byte in (
            geometry.source_id_byte,
            geometry.receiver_id_byte,
            geometry.source_x_byte,
            geometry.source_y_byte,
            geometry.receiver_x_byte,
            geometry.receiver_y_byte,
            geometry.source_elevation_byte,
            geometry.receiver_elevation_byte,
            geometry.coordinate_scalar_byte,
            geometry.elevation_scalar_byte,
        )
    }
    source_id = _coerce_id_header(
        headers[geometry.source_id_byte],
        name='source_id',
        n_traces=n_traces,
    )
    receiver_id = _coerce_id_header(
        headers[geometry.receiver_id_byte],
        name='receiver_id',
        n_traces=n_traces,
    )
    coordinate_scalar = _coerce_scalar_header(
        headers[geometry.coordinate_scalar_byte],
        name='coordinate_scalar',
        n_traces=n_traces,
    )
    elevation_scalar = _coerce_scalar_header(
        headers[geometry.elevation_scalar_byte],
        name='elevation_scalar',
        n_traces=n_traces,
    )
    source_x = _scaled_header_to_meters(
        headers[geometry.source_x_byte],
        scalars=coordinate_scalar,
        unit=geometry.coordinate_unit,
        name='source_x',
        n_traces=n_traces,
    )
    source_y = _scaled_header_to_meters(
        headers[geometry.source_y_byte],
        scalars=coordinate_scalar,
        unit=geometry.coordinate_unit,
        name='source_y',
        n_traces=n_traces,
    )
    receiver_x = _scaled_header_to_meters(
        headers[geometry.receiver_x_byte],
        scalars=coordinate_scalar,
        unit=geometry.coordinate_unit,
        name='receiver_x',
        n_traces=n_traces,
    )
    receiver_y = _scaled_header_to_meters(
        headers[geometry.receiver_y_byte],
        scalars=coordinate_scalar,
        unit=geometry.coordinate_unit,
        name='receiver_y',
        n_traces=n_traces,
    )
    source_elevation = _scaled_header_to_meters(
        headers[geometry.source_elevation_byte],
        scalars=elevation_scalar,
        unit=geometry.elevation_unit,
        name='source_elevation',
        n_traces=n_traces,
    )
    receiver_elevation = _scaled_header_to_meters(
        headers[geometry.receiver_elevation_byte],
        scalars=elevation_scalar,
        unit=geometry.elevation_unit,
        name='receiver_elevation',
        n_traces=n_traces,
    )
    source_key = _endpoint_key_array(
        endpoint_kind='source',
        endpoint_id=source_id,
        x_m=source_x,
        y_m=source_y,
        elevation_m=source_elevation,
    )
    receiver_key = _endpoint_key_array(
        endpoint_kind='receiver',
        endpoint_id=receiver_id,
        x_m=receiver_x,
        y_m=receiver_y,
        elevation_m=receiver_elevation,
    )
    return _TraceEndpointKeys(
        sorted_trace_index=sorted_trace_index,
        source_endpoint_key_sorted=source_key,
        receiver_endpoint_key_sorted=receiver_key,
        source_endpoint_id_sorted=_string_array(source_id),
        receiver_endpoint_id_sorted=_string_array(receiver_id),
        source_identity_sorted=source_key
        if (req.source_key_header or 'endpoint_key') == 'endpoint_key'
        else _string_array(source_id),
        receiver_identity_sorted=receiver_key
        if (req.receiver_key_header or 'endpoint_key') == 'endpoint_key'
        else _string_array(receiver_id),
    )


def _producer_trace_endpoint_geometry(
    *,
    state: AppState,
    table_paths: _ResolvedTablePaths,
    imported: RefractionStaticTableImportResult,
) -> RefractionStaticGeometryRequest | None:
    geometries = [
        _geometry_from_refraction_static_request(path)
        for path in _producer_refraction_request_paths(
            state=state,
            table_paths=table_paths,
            imported=imported,
        )
    ]
    if not geometries:
        return None
    first = geometries[0]
    first_payload = first.model_dump(mode='json')
    for geometry in geometries[1:]:
        if geometry.model_dump(mode='json') != first_payload:
            raise ValueError(
                'referenced static table artifacts have conflicting producer geometry'
            )
    return first


def _producer_refraction_request_paths(
    *,
    state: AppState,
    table_paths: _ResolvedTablePaths,
    imported: RefractionStaticTableImportResult,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for table_path in (
        table_paths.combined_table_path,
        table_paths.source_table_path,
        table_paths.receiver_table_path,
    ):
        if table_path is not None:
            candidates.append(table_path.parent / REFRACTION_STATIC_REQUEST_JSON_NAME)

    for source_job_id in _imported_source_job_ids(imported):
        job_dir = _optional_static_job_dir(state, source_job_id)
        if job_dir is not None:
            candidates.append(job_dir / REFRACTION_STATIC_REQUEST_JSON_NAME)

    existing: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.is_file():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        existing.append(path)
    return tuple(existing)


def _imported_source_job_ids(
    imported: RefractionStaticTableImportResult,
) -> tuple[str, ...]:
    ids: list[str] = []
    rows = [
        *imported.source_static_by_endpoint_key.values(),
        *imported.receiver_static_by_endpoint_key.values(),
    ]
    for row in rows:
        source_job_id = str(row.source_job_id).strip()
        if source_job_id:
            ids.append(source_job_id)
    return tuple(dict.fromkeys(ids))


def _validate_imported_endpoint_static_shifts(
    imported: RefractionStaticTableImportResult,
    *,
    max_abs_shift_ms: float,
) -> None:
    limit = float(max_abs_shift_ms)
    if not np.isfinite(limit) or limit < 0.0:
        raise RefractionStaticTraceStoreApplyError(
            'max_abs_shift_ms must be finite and non-negative'
        )

    max_abs_endpoint_shift_ms = 0.0
    source_exceeds_count = 0
    receiver_exceeds_count = 0
    for endpoint_kind, row in _iter_imported_endpoint_statics(imported):
        if row.static_status != 'ok':
            continue
        shift_ms = float(row.applied_shift_s) * 1000.0
        if not np.isfinite(shift_ms):
            raise RefractionStaticTraceStoreApplyError(
                f'{endpoint_kind} imported endpoint static contains non-finite shift'
            )
        abs_shift_ms = abs(shift_ms)
        max_abs_endpoint_shift_ms = max(max_abs_endpoint_shift_ms, abs_shift_ms)
        if abs_shift_ms > limit:
            if endpoint_kind == 'source':
                source_exceeds_count += 1
            else:
                receiver_exceeds_count += 1

    exceeds_count = source_exceeds_count + receiver_exceeds_count
    if exceeds_count:
        raise RefractionStaticTraceStoreApplyError(
            'imported endpoint static shift exceeds max_abs_shift_ms: '
            f'{max_abs_endpoint_shift_ms:.6g} > {limit:.6g}; '
            f'source_exceeds_count={source_exceeds_count}; '
            f'receiver_exceeds_count={receiver_exceeds_count}'
        )


def _iter_imported_endpoint_statics(
    imported: RefractionStaticTableImportResult,
) -> tuple[tuple[str, RefractionImportedEndpointStatic], ...]:
    return (
        *(('source', row) for row in imported.source_static_by_endpoint_key.values()),
        *(('receiver', row) for row in imported.receiver_static_by_endpoint_key.values()),
    )


def _optional_static_job_dir(state: AppState, job_id: str) -> Path | None:
    with state.lock:
        job = state.jobs.get(job_id)
        artifacts_dir = job.get('artifacts_dir') if isinstance(job, dict) else None
    if not isinstance(artifacts_dir, str) or not artifacts_dir:
        return None
    return Path(artifacts_dir)


def _geometry_from_refraction_static_request(
    path: Path,
) -> RefractionStaticGeometryRequest:
    payload = _read_json_object(path)
    request = payload.get('request')
    if not isinstance(request, dict):
        raise ValueError(f'{path.name} must contain a request object')
    geometry = request.get('geometry')
    if not isinstance(geometry, dict):
        raise ValueError(f'{path.name} request must contain geometry')
    return RefractionStaticGeometryRequest.model_validate(geometry)


def _build_trace_shift(
    *,
    req: RefractionStaticTableApplyRequest,
    imported: RefractionStaticTableImportResult,
    endpoint_keys: _TraceEndpointKeys,
) -> _StaticTableTraceShift:
    source_mode = _identity_mode(req.source_key_header)
    receiver_mode = _identity_mode(req.receiver_key_header)
    source_map = _imported_static_map(
        imported.source_static_by_endpoint_key.values(),
        identity_mode=source_mode,
        endpoint_kind='source',
    )
    receiver_map = _imported_static_map(
        imported.receiver_static_by_endpoint_key.values(),
        identity_mode=receiver_mode,
        endpoint_kind='receiver',
    )

    source_shift, source_missing_mask = _map_trace_static(
        endpoint_identity_sorted=endpoint_keys.source_identity_sorted,
        static_by_identity=source_map,
        endpoint_kind='source',
    )
    receiver_shift, receiver_missing_mask = _map_trace_static(
        endpoint_identity_sorted=endpoint_keys.receiver_identity_sorted,
        static_by_identity=receiver_map,
        endpoint_kind='receiver',
    )
    missing_source = _unique_missing(
        endpoint_keys.source_identity_sorted,
        source_missing_mask,
    )
    missing_receiver = _unique_missing(
        endpoint_keys.receiver_identity_sorted,
        receiver_missing_mask,
    )
    _apply_missing_policy(
        req=req,
        source_shift=source_shift,
        receiver_shift=receiver_shift,
        source_missing_mask=source_missing_mask,
        receiver_missing_mask=receiver_missing_mask,
        missing_source=missing_source,
        missing_receiver=missing_receiver,
    )
    trace_status = _trace_status(
        source_missing_mask=source_missing_mask,
        receiver_missing_mask=receiver_missing_mask,
    )
    valid_mask = np.ones(source_shift.shape, dtype=bool)
    trace_shift = source_shift + receiver_shift
    return _StaticTableTraceShift(
        sorted_trace_index=endpoint_keys.sorted_trace_index,
        source_endpoint_key_sorted=endpoint_keys.source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=endpoint_keys.receiver_endpoint_key_sorted,
        source_endpoint_id_sorted=endpoint_keys.source_endpoint_id_sorted,
        receiver_endpoint_id_sorted=endpoint_keys.receiver_endpoint_id_sorted,
        source_identity_sorted=endpoint_keys.source_identity_sorted,
        receiver_identity_sorted=endpoint_keys.receiver_identity_sorted,
        source_static_shift_s_sorted=np.ascontiguousarray(source_shift, dtype=np.float64),
        receiver_static_shift_s_sorted=np.ascontiguousarray(
            receiver_shift,
            dtype=np.float64,
        ),
        trace_shift_s_sorted=np.ascontiguousarray(trace_shift, dtype=np.float64),
        trace_static_status_sorted=np.ascontiguousarray(trace_status),
        trace_static_valid_mask_sorted=np.ascontiguousarray(valid_mask, dtype=bool),
        missing_source_identity=missing_source,
        missing_receiver_identity=missing_receiver,
        source_identity_mode=source_mode,
        receiver_identity_mode=receiver_mode,
    )


def _identity_mode(value: str | None) -> EndpointIdentityMode:
    return 'endpoint_key' if value is None else value


def _imported_static_map(
    rows: Any,
    *,
    identity_mode: EndpointIdentityMode,
    endpoint_kind: str,
) -> dict[str, RefractionImportedEndpointStatic]:
    out: dict[str, RefractionImportedEndpointStatic] = {}
    for row in rows:
        if identity_mode == 'endpoint_key':
            identity = row.endpoint_key
        else:
            if row.endpoint_id is None:
                raise ValueError(
                    f'{endpoint_kind} static table row {row.row_number} is '
                    'missing endpoint_id required by endpoint_id matching'
                )
            identity = row.endpoint_id
        if identity in out:
            raise ValueError(
                f'duplicate {identity_mode} for {endpoint_kind}: {identity!r}'
            )
        out[str(identity)] = row
    return out


def _map_trace_static(
    *,
    endpoint_identity_sorted: np.ndarray,
    static_by_identity: dict[str, RefractionImportedEndpointStatic],
    endpoint_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    identities = np.asarray(endpoint_identity_sorted, dtype=str)
    shifts = np.full(identities.shape, np.nan, dtype=np.float64)
    missing = np.zeros(identities.shape, dtype=bool)
    for index, identity in enumerate(identities.tolist()):
        row = static_by_identity.get(identity)
        if row is None:
            missing[index] = True
            continue
        if row.static_status != 'ok' or not np.isfinite(row.applied_shift_s):
            raise ValueError(
                f'{endpoint_kind} static row for {identity!r} is not apply-ready'
            )
        shifts[index] = float(row.applied_shift_s)
    return shifts, missing


def _apply_missing_policy(
    *,
    req: RefractionStaticTableApplyRequest,
    source_shift: np.ndarray,
    receiver_shift: np.ndarray,
    source_missing_mask: np.ndarray,
    receiver_missing_mask: np.ndarray,
    missing_source: tuple[str, ...],
    missing_receiver: tuple[str, ...],
) -> None:
    errors: list[str] = []
    source_zero_allowed = (
        req.missing_static_policy == 'zero' and req.allow_missing_source_static
    )
    receiver_zero_allowed = (
        req.missing_static_policy == 'zero' and req.allow_missing_receiver_static
    )
    if missing_source and not source_zero_allowed:
        errors.append(
            'missing_source_static: '
            f'{len(missing_source)} source endpoint statics are missing'
        )
    if missing_receiver and not receiver_zero_allowed:
        errors.append(
            'missing_receiver_static: '
            f'{len(missing_receiver)} receiver endpoint statics are missing'
        )
    if errors:
        raise ValueError('; '.join(errors))
    if missing_source:
        source_shift[source_missing_mask] = 0.0
    if missing_receiver:
        receiver_shift[receiver_missing_mask] = 0.0


def _trace_status(
    *,
    source_missing_mask: np.ndarray,
    receiver_missing_mask: np.ndarray,
) -> np.ndarray:
    status = np.full(source_missing_mask.shape, 'ok', dtype='<U48')
    source_only = source_missing_mask & ~receiver_missing_mask
    receiver_only = receiver_missing_mask & ~source_missing_mask
    both = source_missing_mask & receiver_missing_mask
    status[source_only] = 'missing_source_static_zeroed'
    status[receiver_only] = 'missing_receiver_static_zeroed'
    status[both] = 'missing_source_receiver_static_zeroed'
    return status


def _register_corrected_trace_store(
    *,
    state: AppState,
    job_id: str,
    req: RefractionStaticTableApplyRequest,
    imported: RefractionStaticTableImportResult,
    table_paths: _ResolvedTablePaths,
    job_dir: Path,
    trace_shift: _StaticTableTraceShift,
    table_lineage: _StaticTableApplyLineage,
    table_reapply_qc: dict[str, Any],
    double_application_qc: dict[str, Any],
    artifact_names: tuple[str, ...],
) -> tuple[str, Path]:
    source_store_path = Path(state.file_registry.get_store_path(req.file_id))
    source_meta = _read_json_object(source_store_path / 'meta.json')
    corrected_file_id = str(uuid4())
    output_store_path = _corrected_store_path(
        source_store_path=source_store_path,
        job_id=job_id,
        output_name=req.output_name,
    )
    history_metadata = _static_table_apply_history_payload(
        job_id=job_id,
        req=req,
        imported=imported,
        table_paths=table_paths,
        trace_shift=trace_shift,
        table_lineage=table_lineage,
        table_reapply_qc=table_reapply_qc,
        double_application_qc=double_application_qc,
        corrected_file_id=corrected_file_id,
    )
    corrected_file_json = job_dir / CORRECTED_FILE_JSON_NAME
    try:
        build_result = build_time_shifted_trace_store(
            source_store_path=source_store_path,
            output_store_path=output_store_path,
            trace_shift_s_sorted=trace_shift.trace_shift_s_sorted,
            fill_value=req.fill_value,
            output_dtype='float32',
            derived_metadata=_derived_metadata(
                job_id=job_id,
                history_metadata=history_metadata,
            ),
            from_file_id=req.file_id,
            original_segy_path=_optional_string(source_meta.get('original_segy_path')),
            header_bytes_to_materialize=(req.key1_byte, req.key2_byte),
        )
        reader = register_trace_store(
            state=state,
            file_id=corrected_file_id,
            store_dir=build_result.store_path,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            dt=build_result.dt,
            update_registry=True,
            touch_meta=True,
            preload_header_bytes=(req.key1_byte, req.key2_byte),
        )
        _verify_registered_trace_store(
            state=state,
            file_id=corrected_file_id,
            store_path=build_result.store_path,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            reader=reader,
        )
        _write_json_atomic(
            corrected_file_json,
            _corrected_file_payload(
                req=req,
                job_id=job_id,
                corrected_file_id=corrected_file_id,
                build_result=build_result,
                trace_shift=trace_shift,
                table_lineage=table_lineage,
                artifact_names=artifact_names,
            ),
        )
    except Exception:
        _cleanup_registration(
            state,
            file_id=corrected_file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
        )
        _cleanup_store(output_store_path)
        _cleanup_artifact(corrected_file_json)
        raise
    return corrected_file_id, build_result.store_path


def _write_static_table_apply_qc(
    *,
    path: Path,
    job_id: str,
    req: RefractionStaticTableApplyRequest,
    imported: RefractionStaticTableImportResult,
    table_paths: _ResolvedTablePaths,
    trace_shift: _StaticTableTraceShift,
    table_lineage: _StaticTableApplyLineage,
    table_reapply_qc: dict[str, Any],
    double_application_qc: dict[str, Any],
    validation_status: str,
    corrected_file_id: str | None,
) -> dict[str, Any]:
    qc = _qc_payload(
        job_id=job_id,
        req=req,
        imported=imported,
        table_paths=table_paths,
        trace_shift=trace_shift,
        table_lineage=table_lineage,
        table_reapply_qc=table_reapply_qc,
        double_application_qc=double_application_qc,
        validation_status=validation_status,
        corrected_file_id=corrected_file_id,
    )
    _write_json_atomic(path, qc)
    return qc


def _write_static_table_apply_artifacts(
    *,
    job_dir: Path,
    job_id: str,
    req: RefractionStaticTableApplyRequest,
    imported: RefractionStaticTableImportResult,
    table_paths: _ResolvedTablePaths,
    trace_shift: _StaticTableTraceShift,
    table_lineage: _StaticTableApplyLineage,
    table_reapply_qc: dict[str, Any],
    double_application_qc: dict[str, Any],
    validation_status: str,
    corrected_file_id: str | None,
) -> tuple[str, ...]:
    solution_path = job_dir / STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME
    qc_path = job_dir / STATIC_TABLE_APPLY_QC_JSON_NAME
    csv_path = job_dir / STATIC_TABLE_APPLY_TRACE_SHIFTS_CSV_NAME
    history_path = job_dir / STATIC_TABLE_APPLY_HISTORY_JSON_NAME
    refraction_history_path = job_dir / STATIC_TABLE_APPLY_REFRACTION_HISTORY_JSON_NAME
    _write_npz_atomic(solution_path, _solution_arrays(trace_shift, imported=imported))
    _write_static_table_apply_qc(
        path=qc_path,
        job_id=job_id,
        req=req,
        imported=imported,
        table_paths=table_paths,
        trace_shift=trace_shift,
        table_lineage=table_lineage,
        table_reapply_qc=table_reapply_qc,
        double_application_qc=double_application_qc,
        validation_status=validation_status,
        corrected_file_id=corrected_file_id,
    )
    _write_trace_shift_csv(csv_path, trace_shift)
    history = _write_static_table_apply_history(
        path=history_path,
        job_id=job_id,
        req=req,
        imported=imported,
        table_paths=table_paths,
        trace_shift=trace_shift,
        table_lineage=table_lineage,
        table_reapply_qc=table_reapply_qc,
        double_application_qc=double_application_qc,
        corrected_file_id=corrected_file_id,
    )
    _write_json_atomic(refraction_history_path, history)
    return (
        STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME,
        STATIC_TABLE_APPLY_QC_JSON_NAME,
        STATIC_TABLE_APPLY_TRACE_SHIFTS_CSV_NAME,
        STATIC_TABLE_APPLY_HISTORY_JSON_NAME,
        STATIC_TABLE_APPLY_REFRACTION_HISTORY_JSON_NAME,
    )


def _static_table_apply_lineage(
    *,
    req: RefractionStaticTableApplyRequest,
    state: AppState,
    imported: RefractionStaticTableImportResult,
    table_paths: _ResolvedTablePaths,
    trace_shift: _StaticTableTraceShift,
) -> _StaticTableApplyLineage:
    source_digest: str | None = None
    receiver_digest: str | None = None
    if table_paths.combined_table_path is not None:
        combined_digest = _file_sha256(table_paths.combined_table_path)
    else:
        assert table_paths.source_table_path is not None
        assert table_paths.receiver_table_path is not None
        source_digest = _file_sha256(table_paths.source_table_path)
        receiver_digest = _file_sha256(table_paths.receiver_table_path)
        combined_digest = _digest_json(
            {
                'source_table_digest': source_digest,
                'receiver_table_digest': receiver_digest,
            }
        )
    return _StaticTableApplyLineage(
        source_table_digest=source_digest,
        receiver_table_digest=receiver_digest,
        combined_table_digest=combined_digest,
        trace_shift_s_sorted_digest=_array_sha256(trace_shift.trace_shift_s_sorted),
        applied_component_name='refraction',
        created_from_refraction_job_id=_first_source_job_id(imported),
        created_from_export_job_id=_created_from_export_job_id(
            state=state,
            table_paths=table_paths,
        ),
    )


def _static_table_reapply_qc(
    *,
    req: RefractionStaticTableApplyRequest,
    state: AppState,
    table_lineage: _StaticTableApplyLineage,
) -> dict[str, Any]:
    source_meta = _source_trace_store_meta(req=req, state=state)
    histories = _static_table_history_records(source_meta)
    current_digests = _lineage_table_digest_set(table_lineage)
    existing_digests: set[str] = set()
    duplicate_component_digests: set[str] = set()
    existing_source_jobs: set[str] = set()
    duplicate_source_jobs: set[str] = set()
    component_name = _history_text(table_lineage.applied_component_name)

    for history in histories:
        existing_digests.update(_history_table_digests(history))
        existing_source_jobs.update(_history_source_job_ids(history))
        duplicate_component_digests.update(
            _history_duplicate_component_digests(
                history,
                component_name=component_name,
                current_digests=current_digests,
            )
        )

    duplicate_table_digests = sorted(current_digests.intersection(existing_digests))
    current_source_jobs = {
        job_id
        for job_id in (
            table_lineage.created_from_refraction_job_id,
            table_lineage.created_from_export_job_id,
        )
        if job_id
    }
    duplicate_source_jobs = current_source_jobs.intersection(existing_source_jobs)

    duplicate_reasons: list[str] = []
    if duplicate_table_digests:
        duplicate_reasons.append('same_table_digest')
    if duplicate_component_digests:
        duplicate_reasons.append('same_component_name_and_digest')
    if duplicate_source_jobs:
        duplicate_reasons.append('table_source_job_already_applied')

    allow_override = bool(req.allow_reapply_same_static_table)
    status = 'checked'
    message = ''
    warnings: list[str] = []
    if duplicate_reasons:
        detail = ', '.join(duplicate_reasons)
        message = (
            f'static table apply history check for input file_id {req.file_id!r}: '
            f'{detail}; allow_reapply_same_static_table={allow_override}'
        )
        if allow_override:
            status = 'duplicate_allowed_by_override'
            warnings.append(message)
        else:
            status = 'duplicate_rejected'

    return {
        'status': status,
        'allow_reapply_same_static_table': allow_override,
        'override_used': bool(duplicate_reasons and allow_override),
        'duplicate_reasons': duplicate_reasons,
        'duplicate_table_digests': duplicate_table_digests,
        'duplicate_component_digests': sorted(duplicate_component_digests),
        'duplicate_source_job_ids': sorted(duplicate_source_jobs),
        'existing_table_digests': sorted(existing_digests),
        'existing_source_job_ids': sorted(existing_source_jobs),
        'message': message,
        'warnings': warnings,
    }


def _double_application_qc(
    *,
    req: RefractionStaticTableApplyRequest,
    state: AppState,
) -> dict[str, Any]:
    source_meta = _source_trace_store_meta(req=req, state=state)
    return static_history_double_application_qc(
        input_file_id=req.file_id,
        policy=req.double_application_policy,
        requested_components=('refraction',),
        source_meta=source_meta,
    )


def _source_trace_store_meta(
    *,
    req: RefractionStaticTableApplyRequest,
    state: AppState,
) -> dict[str, Any]:
    source_store_path = Path(state.file_registry.get_store_path(req.file_id))
    return _read_json_object(source_store_path / 'meta.json')


def _enforce_double_application_policy(qc: dict[str, Any]) -> None:
    if qc.get('status') != 'duplicate_rejected':
        return
    message = str(
        qc.get('message')
        or 'static history double-application policy rejected the job'
    )
    raise ValueError(message)


def _enforce_static_table_reapply_guard(qc: dict[str, Any]) -> None:
    if qc.get('status') != 'duplicate_rejected':
        return
    message = str(
        qc.get('message') or 'static table history rejected duplicate application'
    )
    raise ValueError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.dtype('<f8')))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode('ascii'))
    digest.update(json.dumps(arr.shape, separators=(',', ':')).encode('ascii'))
    digest.update(arr.tobytes(order='C'))
    return digest.hexdigest()


def _digest_json(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(',', ':'),
        sort_keys=True,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _created_from_export_job_id(
    *,
    state: AppState,
    table_paths: _ResolvedTablePaths,
) -> str | None:
    export_job_ids: list[str] = []
    with state.lock:
        for job_id in table_paths.artifact_job_ids:
            job = state.jobs.get(job_id)
            if not isinstance(job, Mapping):
                continue
            if job.get('statics_kind') == 'refraction_export':
                export_job_ids.append(str(job_id))
    unique = tuple(dict.fromkeys(export_job_ids))
    return unique[0] if len(unique) == 1 else None


def _static_table_history_records(
    source_meta: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []

    def visit(value: object) -> None:
        if not isinstance(value, Mapping):
            return
        for key in ('static_table_apply_history', 'static_history'):
            history = value.get(key)
            if isinstance(history, Mapping):
                records.append(history)
        derived = value.get('derived')
        if isinstance(derived, Mapping):
            visit(derived)
        components = value.get('components')
        if isinstance(components, list):
            for component in components:
                if isinstance(component, Mapping):
                    visit(component)

    visit(source_meta)
    return tuple(records)


def _lineage_table_digest_set(lineage: _StaticTableApplyLineage) -> set[str]:
    return {
        digest
        for digest in (
            lineage.source_table_digest,
            lineage.receiver_table_digest,
            lineage.combined_table_digest,
        )
        if digest
    }


def _history_table_digests(history: Mapping[str, object]) -> set[str]:
    digests = {
        digest
        for digest in (
            _history_text(history.get('source_table_digest')),
            _history_text(history.get('receiver_table_digest')),
            _history_text(history.get('combined_table_digest')),
        )
        if digest
    }
    table_digests = history.get('table_digests')
    if isinstance(table_digests, Mapping):
        digests.update(
            digest
            for digest in (
                _history_text(table_digests.get('source')),
                _history_text(table_digests.get('receiver')),
                _history_text(table_digests.get('combined')),
            )
            if digest
        )
    return digests


def _history_source_job_ids(history: Mapping[str, object]) -> set[str]:
    return {
        job_id
        for job_id in (
            _history_text(history.get('created_from_refraction_job_id')),
            _history_text(history.get('created_from_export_job_id')),
            _history_text(history.get('source_job_id')),
        )
        if job_id
    }


def _history_duplicate_component_digests(
    history: Mapping[str, object],
    *,
    component_name: str | None,
    current_digests: set[str],
) -> set[str]:
    if not component_name:
        return set()
    components = history.get('components')
    if not isinstance(components, list):
        return set()
    duplicates: set[str] = set()
    for component in components:
        if not isinstance(component, Mapping):
            continue
        if _history_text(component.get('name')) != component_name:
            continue
        component_digests = {
            digest
            for digest in (
                _history_text(component.get('table_digest')),
                _history_text(component.get('combined_table_digest')),
                _history_text(component.get('source_table_digest')),
                _history_text(component.get('receiver_table_digest')),
            )
            if digest
        }
        duplicates.update(component_digests.intersection(current_digests))
    return duplicates


def _history_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _solution_arrays(
    trace_shift: _StaticTableTraceShift,
    *,
    imported: RefractionStaticTableImportResult,
) -> dict[str, np.ndarray]:
    return {
        'schema_version': np.asarray(1, dtype=np.int64),
        'artifact_kind': np.asarray('static_table_apply_solution'),
        'sign_convention': np.asarray(REFRACTION_STATIC_REPO_SIGN_CONVENTION),
        'import_schema_version': np.asarray(imported.schema_version, dtype=np.int64),
        'source_identity_mode': np.asarray(trace_shift.source_identity_mode),
        'receiver_identity_mode': np.asarray(trace_shift.receiver_identity_mode),
        'sorted_trace_index': np.asarray(trace_shift.sorted_trace_index, dtype=np.int64),
        'source_endpoint_key_sorted': _string_array(
            trace_shift.source_endpoint_key_sorted
        ),
        'receiver_endpoint_key_sorted': _string_array(
            trace_shift.receiver_endpoint_key_sorted
        ),
        'source_endpoint_id_sorted': _string_array(
            trace_shift.source_endpoint_id_sorted
        ),
        'receiver_endpoint_id_sorted': _string_array(
            trace_shift.receiver_endpoint_id_sorted
        ),
        'source_identity_sorted': _string_array(trace_shift.source_identity_sorted),
        'receiver_identity_sorted': _string_array(trace_shift.receiver_identity_sorted),
        'source_static_shift_s_sorted': np.asarray(
            trace_shift.source_static_shift_s_sorted,
            dtype=np.float64,
        ),
        'receiver_static_shift_s_sorted': np.asarray(
            trace_shift.receiver_static_shift_s_sorted,
            dtype=np.float64,
        ),
        'trace_shift_s_sorted': np.asarray(
            trace_shift.trace_shift_s_sorted,
            dtype=np.float64,
        ),
        'trace_static_status_sorted': _string_array(
            trace_shift.trace_static_status_sorted
        ),
        'trace_static_valid_mask_sorted': np.asarray(
            trace_shift.trace_static_valid_mask_sorted,
            dtype=bool,
        ),
    }


def _qc_payload(
    *,
    job_id: str,
    req: RefractionStaticTableApplyRequest,
    imported: RefractionStaticTableImportResult,
    table_paths: _ResolvedTablePaths,
    trace_shift: _StaticTableTraceShift,
    table_lineage: _StaticTableApplyLineage,
    table_reapply_qc: dict[str, Any],
    double_application_qc: dict[str, Any],
    validation_status: str,
    corrected_file_id: str | None,
) -> dict[str, Any]:
    shift_ms = trace_shift.trace_shift_s_sorted * 1000.0
    artifact_names = [
        STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME,
        STATIC_TABLE_APPLY_QC_JSON_NAME,
        STATIC_TABLE_APPLY_TRACE_SHIFTS_CSV_NAME,
        STATIC_TABLE_APPLY_HISTORY_JSON_NAME,
        STATIC_TABLE_APPLY_REFRACTION_HISTORY_JSON_NAME,
    ]
    if corrected_file_id is not None:
        artifact_names.append(CORRECTED_FILE_JSON_NAME)
    return {
        'schema_version': 1,
        'artifact_kind': 'static_table_apply_qc',
        'statics_kind': 'refraction_static_table_apply',
        'job_id': job_id,
        'source_file_id': req.file_id,
        'corrected_file_id': corrected_file_id,
        'validation_status': validation_status,
        'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
        'source_table_digest': table_lineage.source_table_digest,
        'receiver_table_digest': table_lineage.receiver_table_digest,
        'combined_table_digest': table_lineage.combined_table_digest,
        'trace_shift_s_sorted_digest': (
            table_lineage.trace_shift_s_sorted_digest
        ),
        'applied_component_name': table_lineage.applied_component_name,
        'created_from_refraction_job_id': (
            table_lineage.created_from_refraction_job_id
        ),
        'created_from_export_job_id': table_lineage.created_from_export_job_id,
        'missing_static_policy': req.missing_static_policy,
        'allow_missing_source_static': bool(req.allow_missing_source_static),
        'allow_missing_receiver_static': bool(req.allow_missing_receiver_static),
        'allow_reapply_same_static_table': bool(
            req.allow_reapply_same_static_table
        ),
        'source_identity_mode': trace_shift.source_identity_mode,
        'receiver_identity_mode': trace_shift.receiver_identity_mode,
        'n_traces': int(trace_shift.trace_shift_s_sorted.shape[0]),
        'n_source_rows': int(imported.n_source_rows),
        'n_receiver_rows': int(imported.n_receiver_rows),
        'n_missing_source_endpoints': len(trace_shift.missing_source_identity),
        'n_missing_receiver_endpoints': len(trace_shift.missing_receiver_identity),
        'missing_source_identity': list(trace_shift.missing_source_identity),
        'missing_receiver_identity': list(trace_shift.missing_receiver_identity),
        'trace_static_status_counts': _status_counts(
            trace_shift.trace_static_status_sorted
        ),
        'n_positive_trace_shifts': int(np.count_nonzero(shift_ms > 0.0)),
        'n_negative_trace_shifts': int(np.count_nonzero(shift_ms < 0.0)),
        'n_zero_trace_shifts': int(np.count_nonzero(shift_ms == 0.0)),
        'applied_shift_min_ms': _stat(shift_ms, 'min'),
        'applied_shift_max_ms': _stat(shift_ms, 'max'),
        'applied_shift_median_ms': _stat(shift_ms, 'median'),
        'max_abs_applied_shift_ms': _stat(np.abs(shift_ms), 'max'),
        'max_abs_shift_ms': float(req.max_abs_shift_ms),
        'register_corrected_file': bool(req.register_corrected_file),
        'table_artifact_ids': dict(table_paths.artifact_ids),
        'import_warnings': list(imported.warnings),
        'static_table_reapply_guard': dict(table_reapply_qc),
        'double_application_policy': dict(double_application_qc),
        'artifact_names': artifact_names,
    }


def _write_static_table_apply_history(
    *,
    path: Path,
    job_id: str,
    req: RefractionStaticTableApplyRequest,
    imported: RefractionStaticTableImportResult,
    table_paths: _ResolvedTablePaths,
    trace_shift: _StaticTableTraceShift,
    table_lineage: _StaticTableApplyLineage,
    table_reapply_qc: dict[str, Any],
    double_application_qc: dict[str, Any],
    corrected_file_id: str | None,
) -> dict[str, Any]:
    payload = _static_table_apply_history_payload(
        job_id=job_id,
        req=req,
        imported=imported,
        table_paths=table_paths,
        trace_shift=trace_shift,
        table_lineage=table_lineage,
        table_reapply_qc=table_reapply_qc,
        double_application_qc=double_application_qc,
        corrected_file_id=corrected_file_id,
    )
    _write_json_atomic(path, payload)
    return payload


def _static_table_apply_history_payload(
    *,
    job_id: str,
    req: RefractionStaticTableApplyRequest,
    imported: RefractionStaticTableImportResult,
    table_paths: _ResolvedTablePaths,
    trace_shift: _StaticTableTraceShift,
    table_lineage: _StaticTableApplyLineage,
    table_reapply_qc: dict[str, Any],
    double_application_qc: dict[str, Any],
    corrected_file_id: str | None,
) -> dict[str, Any]:
    warnings = [
        *list(imported.warnings),
        *[
            str(item)
            for item in table_reapply_qc.get('warnings', [])
            if str(item)
        ],
        *[
            str(item)
            for item in double_application_qc.get('warnings', [])
            if str(item)
        ],
    ]
    payload: dict[str, Any] = {
        'schema_version': 1,
        'artifact_kind': 'refraction_static_history',
        'history_kind': 'static_table_apply',
        'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
        'source_table_digest': table_lineage.source_table_digest,
        'receiver_table_digest': table_lineage.receiver_table_digest,
        'combined_table_digest': table_lineage.combined_table_digest,
        'trace_shift_s_sorted_digest': (
            table_lineage.trace_shift_s_sorted_digest
        ),
        'applied_component_name': table_lineage.applied_component_name,
        'created_from_refraction_job_id': (
            table_lineage.created_from_refraction_job_id
        ),
        'created_from_export_job_id': table_lineage.created_from_export_job_id,
        'input_file_id': req.file_id,
        'output_file_id': corrected_file_id,
        'source_file_id': req.file_id,
        'corrected_file_id': corrected_file_id,
        'job_id': job_id,
        'source_export_format': 'canonical_static_table',
        'source_job_id': table_lineage.created_from_refraction_job_id,
        'import_schema_name': 'canonical_static_table',
        'import_schema_version': int(imported.schema_version),
        'imported_table_artifacts': dict(table_paths.artifact_ids),
        'endpoint_identity_mode': {
            'source': trace_shift.source_identity_mode,
            'receiver': trace_shift.receiver_identity_mode,
        },
        'cumulative_shift_artifact': STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME,
        'cumulative_shift_field': 'trace_shift_s_sorted',
        'components': [
            {
                'name': table_lineage.applied_component_name,
                'applied_to_trace_shift': True,
                'artifact': STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME,
                'table_digest': table_lineage.combined_table_digest,
                'trace_shift_s_sorted_digest': (
                    table_lineage.trace_shift_s_sorted_digest
                ),
            },
        ],
        'validation_status': 'ok',
        'warnings': warnings,
        'allow_reapply_same_static_table': bool(
            req.allow_reapply_same_static_table
        ),
        'static_table_reapply_guard': dict(table_reapply_qc),
        'double_application_policy': dict(double_application_qc),
        'double_application': dict(double_application_qc),
    }
    return payload


def _write_trace_shift_csv(path: Path, trace_shift: _StaticTableTraceShift) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    columns = (
        'sorted_trace_index',
        'source_endpoint_key',
        'receiver_endpoint_key',
        'source_endpoint_id',
        'receiver_endpoint_id',
        'source_identity',
        'receiver_identity',
        'source_static_shift_ms',
        'receiver_static_shift_ms',
        'trace_shift_ms',
        'trace_static_status',
        'sign_convention',
    )
    try:
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            for index in range(int(trace_shift.trace_shift_s_sorted.shape[0])):
                writer.writerow(
                    {
                        'sorted_trace_index': int(trace_shift.sorted_trace_index[index]),
                        'source_endpoint_key': str(
                            trace_shift.source_endpoint_key_sorted[index]
                        ),
                        'receiver_endpoint_key': str(
                            trace_shift.receiver_endpoint_key_sorted[index]
                        ),
                        'source_endpoint_id': str(
                            trace_shift.source_endpoint_id_sorted[index]
                        ),
                        'receiver_endpoint_id': str(
                            trace_shift.receiver_endpoint_id_sorted[index]
                        ),
                        'source_identity': str(trace_shift.source_identity_sorted[index]),
                        'receiver_identity': str(
                            trace_shift.receiver_identity_sorted[index]
                        ),
                        'source_static_shift_ms': _format_ms(
                            trace_shift.source_static_shift_s_sorted[index]
                        ),
                        'receiver_static_shift_ms': _format_ms(
                            trace_shift.receiver_static_shift_s_sorted[index]
                        ),
                        'trace_shift_ms': _format_ms(
                            trace_shift.trace_shift_s_sorted[index]
                        ),
                        'trace_static_status': str(
                            trace_shift.trace_static_status_sorted[index]
                        ),
                        'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
                    }
                )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _corrected_file_payload(
    *,
    req: RefractionStaticTableApplyRequest,
    job_id: str,
    corrected_file_id: str,
    build_result: TimeShiftedTraceStoreResult,
    trace_shift: _StaticTableTraceShift,
    table_lineage: _StaticTableApplyLineage,
    artifact_names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        'schema_version': 1,
        'artifact_kind': 'corrected_file',
        'statics_kind': 'refraction_static_table_apply',
        'apply_mode': 'static_table',
        'source_file_id': req.file_id,
        'corrected_file_id': corrected_file_id,
        'file_id': corrected_file_id,
        'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
        'source_table_digest': table_lineage.source_table_digest,
        'receiver_table_digest': table_lineage.receiver_table_digest,
        'combined_table_digest': table_lineage.combined_table_digest,
        'trace_shift_s_sorted_digest': (
            table_lineage.trace_shift_s_sorted_digest
        ),
        'applied_component_name': table_lineage.applied_component_name,
        'created_from_refraction_job_id': (
            table_lineage.created_from_refraction_job_id
        ),
        'created_from_export_job_id': table_lineage.created_from_export_job_id,
        'shift_field': 'trace_shift_s_sorted',
        'static_components_applied': ['refraction'],
        'interpolation': 'linear',
        'fill_value': float(req.fill_value),
        'output_dtype': 'float32',
        'store_name': build_result.store_path.name,
        'derived_from_file_id': req.file_id,
        'derived_by': 'refraction_static_table_apply',
        'derivation': 'refraction_static_table_apply',
        'source_job_id': job_id,
        'job_id': job_id,
        'key1_byte': int(req.key1_byte),
        'key2_byte': int(req.key2_byte),
        'dt': float(build_result.dt),
        'n_traces': int(build_result.n_traces),
        'n_samples': int(build_result.n_samples),
        'sample_interval_s': float(build_result.dt),
        'max_abs_applied_shift_ms': float(
            np.max(np.abs(trace_shift.trace_shift_s_sorted * 1000.0))
        ),
        'solution_artifact': STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME,
        'apply_qc_artifact': STATIC_TABLE_APPLY_QC_JSON_NAME,
        'static_table_apply_history_artifact': STATIC_TABLE_APPLY_HISTORY_JSON_NAME,
        'refraction_static_history_artifact': (
            STATIC_TABLE_APPLY_REFRACTION_HISTORY_JSON_NAME
        ),
        'artifact_names': [*artifact_names, CORRECTED_FILE_JSON_NAME],
    }


def _derived_metadata(
    *,
    job_id: str,
    history_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    warnings = [
        str(item)
        for item in history_metadata.get('warnings', [])
        if str(item)
    ]
    metadata: dict[str, Any] = {
        'statics_kind': 'refraction_static_table_apply',
        'derived_by': 'refraction_static_table_apply',
        'derivation': 'refraction_static_table_apply',
        'source_job_id': str(job_id),
        'applied_to': 'raw_trace_store',
        'source_table_digest': history_metadata.get('source_table_digest'),
        'receiver_table_digest': history_metadata.get('receiver_table_digest'),
        'combined_table_digest': history_metadata.get('combined_table_digest'),
        'trace_shift_s_sorted_digest': history_metadata.get(
            'trace_shift_s_sorted_digest'
        ),
        'applied_component_name': history_metadata.get('applied_component_name'),
        'created_from_refraction_job_id': history_metadata.get(
            'created_from_refraction_job_id'
        ),
        'created_from_export_job_id': history_metadata.get(
            'created_from_export_job_id'
        ),
        'components': [
            {
                'name': 'static_table_apply',
                'job_id': str(job_id),
                'solution_artifact': STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME,
                'shift_field': 'trace_shift_s_sorted',
                'value_kind': 'applied_event_time_shift_s',
                'static_components_applied': ['refraction'],
                'sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
                'table_digest': history_metadata.get('combined_table_digest'),
                'trace_shift_s_sorted_digest': history_metadata.get(
                    'trace_shift_s_sorted_digest'
                ),
            }
        ],
        'solution_artifact': STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME,
        'shift_field': 'trace_shift_s_sorted',
        'value_kind': 'applied_event_time_shift_s',
        'static_components_applied': ['refraction'],
        'source_identity_mode': (
            history_metadata.get('endpoint_identity_mode', {}).get('source')
            if isinstance(history_metadata.get('endpoint_identity_mode'), Mapping)
            else None
        ),
        'receiver_identity_mode': (
            history_metadata.get('endpoint_identity_mode', {}).get('receiver')
            if isinstance(history_metadata.get('endpoint_identity_mode'), Mapping)
            else None
        ),
        'refraction_sign_convention': REFRACTION_STATIC_REPO_SIGN_CONVENTION,
        'static_table_apply_history': dict(history_metadata),
        'static_history': dict(history_metadata),
    }
    if warnings:
        metadata['warnings'] = warnings
    return metadata


def _corrected_store_path(
    *,
    source_store_path: Path,
    job_id: str,
    output_name: str | None,
) -> Path:
    store_name = (
        _safe_store_name_component(output_name)
        if output_name is not None
        else (
            f'{_safe_store_name_component(source_store_path.name)}'
            f'.statics.static-table.{_safe_store_name_component(job_id)}'
        )
    )
    output_path = source_store_path.parent / store_name
    if output_path.exists() or output_path.is_symlink():
        raise RefractionStaticTraceStoreApplyError(
            f'corrected output path already exists: {output_path}'
        )
    return output_path


def _verify_registered_trace_store(
    *,
    state: AppState,
    file_id: str,
    store_path: Path,
    key1_byte: int,
    key2_byte: int,
    reader: Any,
) -> None:
    registered_path = Path(state.file_registry.get_store_path(file_id))
    if registered_path.resolve() != store_path.resolve():
        raise RuntimeError('registered corrected TraceStore path mismatch')
    cache_key = trace_store_cache_key(file_id, key1_byte, key2_byte)
    with state.lock:
        if cache_key not in state.cached_readers:
            raise RuntimeError('registered corrected TraceStore reader is missing')
    key1_values = np.asarray(reader.get_key1_values())
    if key1_values.size == 0:
        raise RuntimeError('registered corrected TraceStore has no key1 values')


def _reader_n_traces(reader: Any) -> int:
    traces = getattr(reader, 'traces', None)
    if isinstance(traces, np.ndarray) and traces.ndim == 2:
        return int(traces.shape[0])
    raise ValueError('target TraceStore reader.traces must be a 2D array')


def _reader_header(reader: Any, byte: int, *, n_traces: int) -> np.ndarray:
    values = np.asarray(reader.ensure_header(byte))
    if values.ndim != 1 or values.shape != (n_traces,):
        raise ValueError(
            f'header byte {byte} shape mismatch: '
            f'expected {(n_traces,)}, got {values.shape}'
        )
    return values


def _coerce_id_header(values: np.ndarray, *, name: str, n_traces: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape != (n_traces,):
        raise ValueError(f'{name} shape mismatch: expected {(n_traces,)}, got {arr.shape}')
    if np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{name} must contain integer-compatible values')
    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr, dtype=np.int64)
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f'{name} must contain integer-compatible values')
    arr_f64 = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr_f64)):
        raise ValueError(f'{name} must contain finite values')
    if not np.all(arr_f64 == np.rint(arr_f64)):
        raise ValueError(f'{name} must contain integer-compatible values')
    return np.ascontiguousarray(arr_f64, dtype=np.int64)


def _coerce_scalar_header(values: np.ndarray, *, name: str, n_traces: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape != (n_traces,):
        raise ValueError(f'{name} shape mismatch: expected {(n_traces,)}, got {arr.shape}')
    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr, dtype=np.int64)
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f'{name} must contain integer-compatible values')
    arr_f64 = arr.astype(np.float64, copy=False)
    valid = np.isfinite(arr_f64) & (arr_f64 == np.rint(arr_f64))
    if not np.all(valid):
        raise ValueError(f'{name} must contain integer-compatible values')
    return np.ascontiguousarray(arr_f64.astype(np.int64), dtype=np.int64)


def _scaled_header_to_meters(
    values: np.ndarray,
    *,
    scalars: np.ndarray,
    unit: str,
    name: str,
    n_traces: int,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.shape != (n_traces,):
        raise ValueError(f'{name} shape mismatch: expected {(n_traces,)}, got {raw.shape}')
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f'{name} must have a real numeric dtype')
    raw_f64 = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(raw_f64)):
        raise ValueError(f'{name} must contain finite values')
    return np.ascontiguousarray(
        normalize_elevation_unit(apply_segy_scalar(raw_f64, scalars), unit),
        dtype=np.float64,
    )


def _endpoint_key_array(
    *,
    endpoint_kind: str,
    endpoint_id: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    elevation_m: np.ndarray,
) -> np.ndarray:
    values = [
        (
            f'{endpoint_kind}:'
            f'{int(endpoint_id[index])}:'
            f'{float(x_m[index]):.17g}:'
            f'{float(y_m[index]):.17g}:'
            f'{float(elevation_m[index]):.17g}'
        )
        for index in range(int(endpoint_id.shape[0]))
    ]
    return _string_array(values)


def _unique_missing(values: np.ndarray, missing_mask: np.ndarray) -> tuple[str, ...]:
    missing = [str(value) for value in np.asarray(values, dtype=str)[missing_mask]]
    return tuple(dict.fromkeys(missing))


def _first_source_job_id(imported: RefractionStaticTableImportResult) -> str | None:
    rows = [
        *imported.source_static_by_endpoint_key.values(),
        *imported.receiver_static_by_endpoint_key.values(),
    ]
    if not rows:
        return None
    source_job_id = str(rows[0].source_job_id).strip()
    return source_job_id or None


def _status_counts(statuses: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(statuses, dtype=str), return_counts=True)
    return {str(status): int(count) for status, count in zip(unique, counts, strict=True)}


def _stat(values: np.ndarray, name: str) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if name == 'min':
        return float(np.min(arr))
    if name == 'max':
        return float(np.max(arr))
    if name == 'median':
        return float(np.median(arr))
    raise ValueError(f'unsupported stat: {name}')


def _format_ms(value_s: object) -> str:
    value_ms = float(value_s) * 1000.0
    if not np.isfinite(value_ms):
        return ''
    return f'{value_ms:.6f}'


def _string_array(values: Any) -> np.ndarray:
    texts = [str(value) for value in np.asarray(values).reshape(-1).tolist()]
    max_len = max((len(text) for text in texts), default=1)
    return np.asarray(texts, dtype=f'<U{max_len}')


def _safe_store_name_component(value: str) -> str:
    safe = _SAFE_STORE_NAME_RE.sub('_', str(value))
    if safe in {'', '.', '..'}:
        raise RefractionStaticTraceStoreApplyError(
            'TraceStore name cannot be made filesystem-safe'
        )
    return safe


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError('source TraceStore original_segy_path must be a string or null')
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8') as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
            )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('wb') as handle:
            np.savez_compressed(handle, **payload)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid JSON file: {path}') from exc
    if not isinstance(payload, dict):
        raise ValueError(f'JSON file must contain an object: {path}')
    return payload


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


def _handle_static_table_apply_error(_exc: Exception) -> JobFailure:
    return JobFailure(finished_ts=time.time())


def _cleanup_registration(
    state: AppState,
    *,
    file_id: str,
    key1_byte: int,
    key2_byte: int,
) -> None:
    with state.lock:
        state.file_registry.pop(file_id, None)
        state.cached_readers.pop(trace_store_cache_key(file_id, key1_byte, key2_byte), None)


def _cleanup_store(output_path: Path) -> None:
    for tmp_path in output_path.parent.glob(f'{output_path.name}.tmp-*'):
        if tmp_path.is_dir():
            shutil.rmtree(tmp_path, ignore_errors=True)
    if output_path.exists():
        shutil.rmtree(output_path, ignore_errors=True)


def _cleanup_artifact(path: Path) -> None:
    path.unlink(missing_ok=True)
    for tmp_path in path.parent.glob(f'{path.name}.*.tmp'):
        tmp_path.unlink(missing_ok=True)


__all__ = [
    'STATIC_TABLE_APPLY_QC_JSON_NAME',
    'STATIC_TABLE_APPLY_HISTORY_JSON_NAME',
    'STATIC_TABLE_APPLY_REFRACTION_HISTORY_JSON_NAME',
    'STATIC_TABLE_APPLY_REQUEST_JSON_NAME',
    'STATIC_TABLE_APPLY_SOLUTION_NPZ_NAME',
    'STATIC_TABLE_APPLY_TRACE_SHIFTS_CSV_NAME',
    'run_refraction_static_table_apply_job',
]
