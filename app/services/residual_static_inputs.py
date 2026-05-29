"""Validated sorted-order inputs for residual static estimation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.contracts.statics.residual import ResidualStaticApplyRequest
from app.core.state import AppState
from app.services.common.array_validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_integer_int64 as _common_coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_nonnegative_int as _coerce_nonnegative_int,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
)
from app.services.first_break_qc_inputs import (
    load_datum_static_solution_npz,
    load_offset_header_sorted,
)
from app.services.job_artifact_refs import resolve_job_artifact_path
from app.services.pick_source_loader import (
    LoadedPickSource,
    load_manual_memmap_pick_source,
    load_npz_pick_source,
)
from app.services.residual_static_types import (
    MoveoutModel,
    ResidualStaticSolverInputs,
)
from app.trace_store.reader import TraceStoreSectionReader

_DT_TOLERANCE = 1e-9
_CORRECTED_FILE_MANIFEST = 'corrected_file.json'
_coerce_1d_integer_int64 = partial(
    _common_coerce_1d_integer_int64,
    nonfinite_message='must contain only finite values',
)


@dataclass(frozen=True)
class ResidualStaticResolvedArtifacts:
    datum_solution_path: Path
    pick_artifact_path: Path | None
    datum_job_id: str
    datum_source_file_id: str
    datum_corrected_file_id: str
    pick_source_artifact_name: str | None


def resolve_residual_static_input_artifacts(
    state: AppState,
    req: ResidualStaticApplyRequest,
) -> ResidualStaticResolvedArtifacts:
    """Resolve datum and pick artifacts for residual statics."""
    datum_job_id = _non_empty_str(
        getattr(req.datum_solution, 'job_id', None),
        name='datum_solution.job_id',
    )
    datum_artifact_name = _plain_artifact_name(
        _non_empty_str(
            getattr(req.datum_solution, 'name', None),
            name='datum_solution.name',
        )
    )
    req_file_id = _non_empty_str(getattr(req, 'file_id', None), name='file_id')
    key1_byte = _validate_header_byte(req.key1_byte, name='key1_byte')
    key2_byte = _validate_header_byte(req.key2_byte, name='key2_byte')

    datum_job = _get_job_snapshot(state, datum_job_id)
    if datum_job is None:
        raise ValueError(f'job_id not found: {datum_job_id}')
    _validate_datum_job(
        job_id=datum_job_id,
        job=datum_job,
        req_file_id=req_file_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    datum_source_file_id = _non_empty_str(
        datum_job.get('file_id'),
        name=f'datum job {datum_job_id} file_id',
    )
    datum_corrected_file_id = _non_empty_str(
        datum_job.get('corrected_file_id'),
        name=f'datum job {datum_job_id} corrected_file_id',
    )

    datum_solution_path = resolve_job_artifact_path(
        state,
        job_id=datum_job_id,
        name=datum_artifact_name,
        allowed_job_types={'statics'},
        allowed_statics_kinds={'datum'},
        expected_key1_byte=key1_byte,
        expected_key2_byte=key2_byte,
        reference_label='datum_solution',
    )
    _validate_corrected_file_manifest_if_present(
        datum_solution_path.parent / _CORRECTED_FILE_MANIFEST,
        datum_source_file_id=datum_source_file_id,
        datum_corrected_file_id=datum_corrected_file_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )

    pick_path, pick_name = _resolve_pick_source_artifact(
        state,
        req=req,
        datum_source_file_id=datum_source_file_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    return ResidualStaticResolvedArtifacts(
        datum_solution_path=datum_solution_path,
        pick_artifact_path=pick_path,
        datum_job_id=datum_job_id,
        datum_source_file_id=datum_source_file_id,
        datum_corrected_file_id=datum_corrected_file_id,
        pick_source_artifact_name=pick_name,
    )


def load_residual_static_pick_source(
    *,
    req: ResidualStaticApplyRequest,
    artifacts: ResidualStaticResolvedArtifacts,
    reader: TraceStoreSectionReader,
    expected_dt: float,
    expected_n_samples: int,
    state: AppState,
) -> LoadedPickSource:
    """Load the residual-static pick source in TraceStore sorted order."""
    pick_source = req.pick_source
    if pick_source.kind == 'manual_memmap':
        return load_manual_memmap_pick_source(
            file_id=artifacts.datum_source_file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            state=state,
        )

    if artifacts.pick_artifact_path is None:
        raise ValueError('pick source artifact path is required')
    source_kind = 'batch_npz' if pick_source.kind == 'batch_job_artifact' else 'manual_npz'
    return load_npz_pick_source(
        artifacts.pick_artifact_path,
        reader=reader,
        expected_dt=expected_dt,
        expected_n_samples=expected_n_samples,
        source_kind=source_kind,
    )


def load_source_receiver_id_headers_sorted(
    reader: TraceStoreSectionReader,
    *,
    source_id_byte: int,
    receiver_id_byte: int,
    expected_n_traces: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read source and receiver ID headers in TraceStore sorted order."""
    source_byte = _validate_header_byte(source_id_byte, name='source_id_byte')
    receiver_byte = _validate_header_byte(receiver_id_byte, name='receiver_id_byte')
    if source_byte == receiver_byte:
        raise ValueError('source_id_byte and receiver_id_byte must differ')
    n_traces = _coerce_positive_int(expected_n_traces, name='expected_n_traces')
    source_id = _coerce_1d_integer_int64(
        _read_reader_header(reader, byte=source_byte, role='source_id'),
        name=f'source_id header byte {source_byte}',
        expected_shape=(n_traces,),
    )
    receiver_id = _coerce_1d_integer_int64(
        _read_reader_header(reader, byte=receiver_byte, role='receiver_id'),
        name=f'receiver_id header byte {receiver_byte}',
        expected_shape=(n_traces,),
    )
    return source_id, receiver_id


def build_residual_static_solver_inputs(
    *,
    req: ResidualStaticApplyRequest,
    artifacts: ResidualStaticResolvedArtifacts,
    pick_source: LoadedPickSource,
    reader: TraceStoreSectionReader,
    expected_dt: float,
    expected_n_samples: int,
) -> ResidualStaticSolverInputs:
    """Build the sorted-order input object consumed by later PR4 solver stages."""
    dt = _coerce_positive_finite_float(expected_dt, name='expected_dt')
    n_samples = _coerce_positive_int(
        expected_n_samples,
        name='expected_n_samples',
    )
    key1_byte = _validate_header_byte(req.key1_byte, name='key1_byte')
    key2_byte = _validate_header_byte(req.key2_byte, name='key2_byte')
    source_id_byte = _validate_header_byte(
        req.source_id_byte,
        name='source_id_byte',
    )
    receiver_id_byte = _validate_header_byte(
        req.receiver_id_byte,
        name='receiver_id_byte',
    )
    if source_id_byte == receiver_id_byte:
        raise ValueError('source_id_byte and receiver_id_byte must differ')

    moveout_model = _moveout_model(req)
    offset_byte = _request_offset_byte(req, moveout_model=moveout_model)
    _validate_reader_key_bytes(
        reader,
        expected_key1_byte=key1_byte,
        expected_key2_byte=key2_byte,
    )
    _validate_reader_dt(reader, expected_dt=dt)

    n_traces = _reader_n_traces(reader)
    reader_n_samples = _reader_n_samples(reader)
    if reader_n_samples != n_samples:
        msg = f'n_samples mismatch: expected {n_samples}, reader has {reader_n_samples}'
        raise ValueError(msg)

    solution = load_datum_static_solution_npz(
        artifacts.datum_solution_path,
        expected_n_traces=n_traces,
        expected_dt=dt,
        expected_key1_byte=key1_byte,
        expected_key2_byte=key2_byte,
    )
    _validate_pick_source(
        pick_source,
        expected_n_traces=n_traces,
        expected_n_samples=n_samples,
        expected_dt=dt,
    )
    picks = _coerce_1d_pick_times(
        pick_source.picks_time_s_sorted,
        valid_mask=pick_source.valid_mask_sorted,
        expected_shape=(n_traces,),
    )
    valid_mask = _coerce_1d_bool_array(
        pick_source.valid_mask_sorted,
        name='valid_mask_sorted',
        expected_shape=(n_traces,),
    )
    pick_time_after_datum = np.ascontiguousarray(
        picks + solution.trace_shift_s_sorted,
        dtype=np.float64,
    )
    pick_time_after_datum[~valid_mask] = np.nan
    if np.any(~np.isfinite(pick_time_after_datum[valid_mask])):
        raise ValueError('pick_time_after_datum_s_sorted must be finite for valid picks')
    if np.any(~np.isnan(pick_time_after_datum[~valid_mask])):
        raise ValueError('pick_time_after_datum_s_sorted must be NaN for invalid picks')

    key1_header = _coerce_1d_integer_int64(
        _read_reader_header(reader, byte=key1_byte, role='key1'),
        name=f'reader header byte {key1_byte}',
        expected_shape=(n_traces,),
    )
    key2_header = _coerce_1d_integer_int64(
        _read_reader_header(reader, byte=key2_byte, role='key2'),
        name=f'reader header byte {key2_byte}',
        expected_shape=(n_traces,),
    )
    if not np.array_equal(solution.key1_sorted, key1_header):
        raise ValueError(
            f'solution key1_sorted does not match reader header byte {key1_byte}'
        )
    if not np.array_equal(solution.key2_sorted, key2_header):
        raise ValueError(
            f'solution key2_sorted does not match reader header byte {key2_byte}'
        )

    source_id, receiver_id = load_source_receiver_id_headers_sorted(
        reader,
        source_id_byte=source_id_byte,
        receiver_id_byte=receiver_id_byte,
        expected_n_traces=n_traces,
    )
    source_unique_ids, source_inverse = np.unique(source_id, return_inverse=True)
    receiver_unique_ids, receiver_inverse = np.unique(receiver_id, return_inverse=True)
    source_index = np.ascontiguousarray(source_inverse, dtype=np.int64)
    receiver_index = np.ascontiguousarray(receiver_inverse, dtype=np.int64)
    source_counts = _valid_pick_counts(
        source_index,
        valid_mask=valid_mask,
        n_unique=int(source_unique_ids.shape[0]),
        name='source_index_sorted',
    )
    receiver_counts = _valid_pick_counts(
        receiver_index,
        valid_mask=valid_mask,
        n_unique=int(receiver_unique_ids.shape[0]),
        name='receiver_index_sorted',
    )

    if moveout_model == 'linear_abs_offset':
        if offset_byte is None:
            raise ValueError('offset_byte is required for linear_abs_offset moveout')
        offset = load_offset_header_sorted(
            reader,
            offset_byte=offset_byte,
            expected_n_traces=n_traces,
        )
        abs_offset = np.ascontiguousarray(np.abs(offset), dtype=np.float64)
    else:
        offset = None
        abs_offset = None

    pick_source_kind = _pick_source_kind(pick_source)
    metadata: dict[str, Any] = {
        'order': 'trace_store_sorted',
        'sign_convention': (
            'pick_time_after_datum_s = pick_time_raw_s + datum_trace_shift_s; '
            'estimated_trace_delay_s is solved later; '
            'applied_residual_shift_s = -estimated_trace_delay_s'
        ),
        'input_file_role': 'datum_corrected_trace_store',
        'input_file_id': req.file_id,
        'datum_source_file_id': artifacts.datum_source_file_id,
        'datum_corrected_file_id': artifacts.datum_corrected_file_id,
        'datum_job_id': artifacts.datum_job_id,
        'datum_solution_artifact': artifacts.datum_solution_path.name,
        'datum_solution_npz_path': str(artifacts.datum_solution_path),
        'pick_source_kind': pick_source_kind,
        'pick_source_artifact': artifacts.pick_source_artifact_name,
        'pick_source_metadata': _pick_source_metadata(pick_source),
        'key1_byte': key1_byte,
        'key2_byte': key2_byte,
        'source_id_byte': source_id_byte,
        'receiver_id_byte': receiver_id_byte,
        'offset_byte': offset_byte,
        'moveout_model': moveout_model,
        'n_valid_picks': int(np.count_nonzero(valid_mask)),
        'n_sources': int(source_unique_ids.shape[0]),
        'n_receivers': int(receiver_unique_ids.shape[0]),
        'datum_static_solution_metadata': solution.metadata,
    }

    return ResidualStaticSolverInputs(
        picks_time_s_sorted=picks,
        valid_pick_mask_sorted=valid_mask,
        pick_time_after_datum_s_sorted=pick_time_after_datum,
        datum_trace_shift_s_sorted=solution.trace_shift_s_sorted,
        source_id_sorted=source_id,
        receiver_id_sorted=receiver_id,
        source_unique_ids=np.ascontiguousarray(source_unique_ids, dtype=np.int64),
        receiver_unique_ids=np.ascontiguousarray(receiver_unique_ids, dtype=np.int64),
        source_index_sorted=source_index,
        receiver_index_sorted=receiver_index,
        source_valid_pick_counts=source_counts,
        receiver_valid_pick_counts=receiver_counts,
        offset_sorted=offset,
        abs_offset_sorted=abs_offset,
        key1_sorted=solution.key1_sorted,
        key2_sorted=solution.key2_sorted,
        source_elevation_m_sorted=solution.source_elevation_m_sorted,
        receiver_elevation_m_sorted=solution.receiver_elevation_m_sorted,
        dt=dt,
        n_traces=n_traces,
        n_samples=n_samples,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        source_id_byte=source_id_byte,
        receiver_id_byte=receiver_id_byte,
        offset_byte=offset_byte,
        moveout_model=moveout_model,
        input_file_id=req.file_id,
        datum_source_file_id=artifacts.datum_source_file_id,
        datum_job_id=artifacts.datum_job_id,
        pick_source_kind=pick_source_kind,
        metadata=metadata,
    )


def _resolve_pick_source_artifact(
    state: AppState,
    *,
    req: ResidualStaticApplyRequest,
    datum_source_file_id: str,
    key1_byte: int,
    key2_byte: int,
) -> tuple[Path | None, str | None]:
    pick_source = req.pick_source
    if pick_source.kind == 'manual_memmap':
        return None, None
    job_id = _non_empty_str(pick_source.job_id, name='pick_source.job_id')
    name = _plain_artifact_name(
        _non_empty_str(pick_source.name, name='pick_source.name')
    )
    if pick_source.kind == 'batch_job_artifact':
        return (
            resolve_job_artifact_path(
                state,
                job_id=job_id,
                name=name,
                allowed_job_types={'batch_apply'},
                expected_file_id=datum_source_file_id,
                expected_key1_byte=key1_byte,
                expected_key2_byte=key2_byte,
                reference_label='pick_source',
            ),
            name,
        )
    if pick_source.kind != 'manual_npz_artifact':
        raise ValueError(f'unsupported pick_source.kind: {pick_source.kind}')
    if not name.endswith('.npz'):
        raise ValueError('manual_npz_artifact name must end with .npz')
    path = resolve_job_artifact_path(
        state,
        job_id=job_id,
        name=name,
        allowed_job_types={'statics', 'batch_apply', 'pipeline'},
    )
    job = _get_job_snapshot(state, job_id)
    if job is None:
        raise ValueError(f'job_id not found: {job_id}')
    _validate_optional_pick_job_metadata(
        job_id=job_id,
        job=job,
        expected_file_id=datum_source_file_id,
        expected_key1_byte=key1_byte,
        expected_key2_byte=key2_byte,
    )
    return path, name


def _validate_datum_job(
    *,
    job_id: str,
    job: Mapping[str, object],
    req_file_id: str,
    key1_byte: int,
    key2_byte: int,
) -> None:
    job_type = job.get('job_type')
    if job_type != 'statics':
        raise ValueError(f'job {job_id} has unsupported job_type: {job_type}')
    statics_kind = job.get('statics_kind')
    if statics_kind != 'datum':
        raise ValueError(f'job {job_id} has unsupported statics_kind: {statics_kind}')
    _validate_required_job_field(
        job_id=job_id,
        job=job,
        reference_label='datum_solution',
        field='key1_byte',
        expected=key1_byte,
    )
    _validate_required_job_field(
        job_id=job_id,
        job=job,
        reference_label='datum_solution',
        field='key2_byte',
        expected=key2_byte,
    )
    corrected_file_id = job.get('corrected_file_id')
    if not isinstance(corrected_file_id, str) or not corrected_file_id:
        raise ValueError(f'datum job {job_id} has no corrected_file_id')
    if corrected_file_id != req_file_id:
        msg = (
            f'datum job {job_id} corrected_file_id mismatch: '
            f'expected {req_file_id!r}, got {corrected_file_id!r}'
        )
        raise ValueError(msg)


def _validate_corrected_file_manifest_if_present(
    path: Path,
    *,
    datum_source_file_id: str,
    datum_corrected_file_id: str,
    key1_byte: int,
    key2_byte: int,
) -> None:
    if not path.exists():
        return
    if not path.is_file():
        raise ValueError(f'corrected_file.json is not a file: {path}')
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'Could not read corrected_file.json: {path}') from exc
    if not isinstance(payload, dict):
        raise ValueError('corrected_file.json must contain an object')
    _validate_manifest_field(
        payload,
        field='file_id',
        expected=datum_corrected_file_id,
        label='corrected_file.json',
    )
    _validate_manifest_field(
        payload,
        field='derived_from_file_id',
        expected=datum_source_file_id,
        label='corrected_file.json',
    )
    _validate_manifest_field(
        payload,
        field='key1_byte',
        expected=key1_byte,
        label='corrected_file.json',
    )
    _validate_manifest_field(
        payload,
        field='key2_byte',
        expected=key2_byte,
        label='corrected_file.json',
    )


def _validate_optional_pick_job_metadata(
    *,
    job_id: str,
    job: Mapping[str, object],
    expected_file_id: str,
    expected_key1_byte: int,
    expected_key2_byte: int,
) -> None:
    _validate_optional_job_field(
        job_id=job_id,
        job=job,
        reference_label='pick_source',
        field='file_id',
        expected=expected_file_id,
    )
    _validate_optional_job_field(
        job_id=job_id,
        job=job,
        reference_label='pick_source',
        field='key1_byte',
        expected=expected_key1_byte,
    )
    _validate_optional_job_field(
        job_id=job_id,
        job=job,
        reference_label='pick_source',
        field='key2_byte',
        expected=expected_key2_byte,
    )


def _validate_required_job_field(
    *,
    job_id: str,
    job: Mapping[str, object],
    reference_label: str,
    field: str,
    expected: object,
) -> None:
    if job.get(field) == expected:
        return
    msg = (
        f'{reference_label} job {job_id} metadata mismatch: '
        f'{field} expected {expected!r}, got {job.get(field)!r}'
    )
    raise ValueError(msg)


def _validate_optional_job_field(
    *,
    job_id: str,
    job: Mapping[str, object],
    reference_label: str,
    field: str,
    expected: object,
) -> None:
    if field not in job or job.get(field) is None:
        return
    _validate_required_job_field(
        job_id=job_id,
        job=job,
        reference_label=reference_label,
        field=field,
        expected=expected,
    )


def _validate_manifest_field(
    payload: Mapping[str, object],
    *,
    field: str,
    expected: object,
    label: str,
) -> None:
    actual = payload.get(field)
    if actual == expected:
        return
    msg = f'{label} mismatch: {field} expected {expected!r}, got {actual!r}'
    raise ValueError(msg)


def _get_job_snapshot(state: AppState, job_id: str) -> dict[str, object] | None:
    with state.lock:
        raw_job = state.jobs.get(job_id)
        return dict(raw_job) if isinstance(raw_job, dict) else None


def _moveout_model(req: ResidualStaticApplyRequest) -> MoveoutModel:
    model = getattr(getattr(req, 'moveout', None), 'model', None)
    if model == 'linear_abs_offset':
        return 'linear_abs_offset'
    if model == 'none':
        return 'none'
    raise ValueError(f'unsupported moveout.model: {model}')


def _request_offset_byte(
    req: ResidualStaticApplyRequest,
    *,
    moveout_model: MoveoutModel,
) -> int | None:
    raw_offset_byte = getattr(req, 'offset_byte', None)
    if raw_offset_byte is None:
        if moveout_model == 'linear_abs_offset':
            raise ValueError('offset_byte is required for linear_abs_offset moveout')
        return None
    if moveout_model == 'none':
        raise ValueError('offset_byte must be None for none moveout')
    return _validate_header_byte(raw_offset_byte, name='offset_byte')


def _validate_pick_source(
    pick_source: LoadedPickSource,
    *,
    expected_n_traces: int,
    expected_n_samples: int,
    expected_dt: float,
) -> None:
    pick_n_traces = _coerce_nonnegative_int(
        getattr(pick_source, 'n_traces', None),
        name='pick source n_traces',
    )
    if pick_n_traces != expected_n_traces:
        msg = f'pick source n_traces mismatch: expected {expected_n_traces}, got {pick_n_traces}'
        raise ValueError(msg)
    pick_n_samples = _coerce_positive_int(
        getattr(pick_source, 'n_samples', None),
        name='pick source n_samples',
    )
    if pick_n_samples != expected_n_samples:
        msg = f'pick source n_samples mismatch: expected {expected_n_samples}, got {pick_n_samples}'
        raise ValueError(msg)
    pick_dt = _coerce_positive_finite_float(
        getattr(pick_source, 'dt', None),
        name='pick source dt',
    )
    if abs(pick_dt - expected_dt) > _DT_TOLERANCE:
        msg = f'pick source dt mismatch: expected {expected_dt}, got {pick_dt}'
        raise ValueError(msg)
    _pick_source_kind(pick_source)
    _pick_source_metadata(pick_source)


def _coerce_1d_pick_times(
    values: np.ndarray,
    *,
    valid_mask: np.ndarray,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    picks = _coerce_1d_real_numeric_float64(
        values,
        name='picks_time_s_sorted',
        expected_shape=expected_shape,
    )
    mask = _coerce_1d_bool_array(
        valid_mask,
        name='valid_mask_sorted',
        expected_shape=expected_shape,
    )
    if np.any(np.isinf(picks)):
        raise ValueError('picks_time_s_sorted contains inf')
    if np.any(~np.isfinite(picks[mask])):
        raise ValueError('valid picks must be finite')
    if np.any(~np.isnan(picks[~mask])):
        raise ValueError('invalid picks must be NaN')
    return picks


def _valid_pick_counts(
    indices: np.ndarray,
    *,
    valid_mask: np.ndarray,
    n_unique: int,
    name: str,
) -> np.ndarray:
    if indices.shape != valid_mask.shape:
        raise ValueError(f'{name} shape mismatch')
    counts = np.bincount(indices[valid_mask], minlength=n_unique)
    return np.ascontiguousarray(counts, dtype=np.int64)


def _read_reader_header(
    reader: TraceStoreSectionReader,
    *,
    byte: int,
    role: str,
) -> np.ndarray:
    reader_header = getattr(reader, 'ensure_header', None)
    if not callable(reader_header):
        raise ValueError(f'reader cannot read {role} header byte {byte}')
    try:
        return reader_header(byte)
    except Exception as exc:  # noqa: BLE001
        msg = f'failed to read {role} header byte {byte}: {exc}'
        raise ValueError(msg) from exc


def _validate_reader_key_bytes(
    reader: TraceStoreSectionReader,
    *,
    expected_key1_byte: int,
    expected_key2_byte: int,
) -> None:
    if not hasattr(reader, 'key1_byte'):
        raise ValueError('reader key1_byte is required')
    if not hasattr(reader, 'key2_byte'):
        raise ValueError('reader key2_byte is required')
    reader_key1 = _validate_header_byte(reader.key1_byte, name='reader key1_byte')
    reader_key2 = _validate_header_byte(reader.key2_byte, name='reader key2_byte')
    if reader_key1 != expected_key1_byte:
        msg = (
            f'reader key1_byte mismatch: expected {expected_key1_byte}, '
            f'got {reader_key1}'
        )
        raise ValueError(msg)
    if reader_key2 != expected_key2_byte:
        msg = (
            f'reader key2_byte mismatch: expected {expected_key2_byte}, '
            f'got {reader_key2}'
        )
        raise ValueError(msg)


def _validate_reader_dt(
    reader: TraceStoreSectionReader,
    *,
    expected_dt: float,
) -> None:
    meta = getattr(reader, 'meta', None)
    if not isinstance(meta, Mapping):
        raise ValueError('reader meta must be a mapping containing dt')
    if 'dt' not in meta:
        raise ValueError('reader meta missing dt')
    reader_dt = _coerce_positive_finite_float(meta['dt'], name='reader dt')
    if abs(reader_dt - expected_dt) > _DT_TOLERANCE:
        msg = f'reader dt mismatch: expected {expected_dt}, got {reader_dt}'
        raise ValueError(msg)


def _reader_n_traces(reader: TraceStoreSectionReader) -> int:
    if hasattr(reader, 'traces'):
        shape = getattr(reader.traces, 'shape', ())
        if shape:
            return _coerce_positive_int(shape[0], name='reader n_traces')
    meta = getattr(reader, 'meta', None)
    if isinstance(meta, Mapping) and 'n_traces' in meta:
        return _coerce_positive_int(meta['n_traces'], name='reader n_traces')
    raise ValueError('reader cannot provide number of traces')


def _reader_n_samples(reader: TraceStoreSectionReader) -> int:
    getter = getattr(reader, 'get_n_samples', None)
    if callable(getter):
        return _coerce_positive_int(getter(), name='reader n_samples')
    if hasattr(reader, 'traces'):
        shape = getattr(reader.traces, 'shape', ())
        if len(shape) >= 2:
            return _coerce_positive_int(shape[-1], name='reader n_samples')
    raise ValueError('reader cannot provide number of samples')


def _validate_header_byte(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer SEG-Y trace header byte')
    byte = int(value)
    if byte < 1 or byte > 240:
        raise ValueError(f'{name} must be between 1 and 240')
    return byte


def _pick_source_kind(pick_source: LoadedPickSource) -> str:
    source_kind = getattr(pick_source, 'source_kind', None)
    if not isinstance(source_kind, str) or not source_kind:
        raise ValueError('pick source source_kind must be a non-empty string')
    return source_kind


def _pick_source_metadata(pick_source: LoadedPickSource) -> dict[str, object]:
    metadata = getattr(pick_source, 'metadata', None)
    if not isinstance(metadata, Mapping):
        raise ValueError('pick source metadata must be a mapping')
    return dict(metadata)


def _plain_artifact_name(name: str) -> str:
    if not name or name in {'.', '..'} or Path(name).name != name:
        raise ValueError('artifact name must be a plain file name')
    return name


def _non_empty_str(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{name} must be a non-empty string')
    return value


__all__ = [
    'MoveoutModel',
    'ResidualStaticResolvedArtifacts',
    'ResidualStaticSolverInputs',
    'build_residual_static_solver_inputs',
    'load_residual_static_pick_source',
    'load_source_receiver_id_headers_sorted',
    'resolve_residual_static_input_artifacts',
]
