"""Sorted-order input assembly for future time-term static inversion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from app.contracts.statics.time_term import TimeTermStaticApplyRequest
from app.core.state import AppState
from app.services.common.array_validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_finite_float64 as _coerce_1d_finite_float64,
    coerce_1d_integer_int64 as _common_coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    is_real_numeric_dtype as _is_real_numeric_dtype,
)
from app.services.errors import DomainError
from app.services.geometry_linkage_loader import (
    LoadedGeometryLinkageArtifact,
    load_geometry_linkage_artifact,
)
from app.services.job_artifact_refs import resolve_job_artifact_path
from app.services.pick_source_loader import (
    LoadedPickSource,
    load_manual_memmap_pick_source,
    load_npz_pick_source,
)
from app.services.reader import get_reader
from app.services.residual_static_inputs import load_source_receiver_id_headers_sorted
from app.services.time_term_types import (
    ORDER,
    SIGN_CONVENTION,
    TimeTermInversionInputs,
)
from app.trace_store.reader import TraceStoreSectionReader
from app.utils.segy_scalars import apply_segy_scalar

_DT_TOLERANCE = 1e-9
_coerce_1d_integer_int64 = partial(
    _common_coerce_1d_integer_int64,
    nonfinite_message='must contain only finite values',
)


@dataclass(frozen=True)
class _ShiftLoadResult:
    values_s_sorted: np.ndarray
    path: Path | None
    field_name: str | None
    metadata: dict[str, object]


def build_time_term_inversion_inputs(
    *,
    request: TimeTermStaticApplyRequest,
    state: AppState,
    datum_solution_path: Path | None = None,
    residual_solution_path: Path | None = None,
    linkage_artifact_path: Path | None = None,
    pick_artifact_path: Path | None = None,
) -> TimeTermInversionInputs:
    """Resolve artifacts from app state and build time-term inversion inputs."""
    if state.file_registry.get_record(request.file_id) is None:
        raise ValueError(f'file_id not found: {request.file_id}')
    reader = _resolve_reader(state, request)
    dt = _resolve_dt(state, request.file_id, reader=reader)
    n_samples = _reader_n_samples(reader)
    pick_source = load_time_term_static_pick_source(
        request=request,
        reader=reader,
        expected_dt=dt,
        expected_n_samples=n_samples,
        state=state,
        pick_artifact_path=pick_artifact_path,
    )
    resolved_linkage_path = _resolve_linkage_artifact_path(
        state,
        request=request,
        expected_n_traces=_reader_n_traces(reader),
        explicit_path=linkage_artifact_path,
    )
    return build_time_term_inversion_inputs_from_sources(
        request=request,
        reader=reader,
        pick_source=pick_source,
        expected_dt=dt,
        expected_n_samples=n_samples,
        datum_solution_path=datum_solution_path,
        residual_solution_path=residual_solution_path,
        linkage_artifact_path=resolved_linkage_path,
    )


def load_time_term_static_pick_source(
    *,
    request: TimeTermStaticApplyRequest,
    reader: TraceStoreSectionReader,
    expected_dt: float,
    expected_n_samples: int,
    state: AppState,
    pick_artifact_path: Path | None = None,
) -> LoadedPickSource:
    """Load the configured first-break pick source in TraceStore sorted order."""
    pick_source = request.pick_source
    if pick_source.kind == 'manual_memmap':
        return load_manual_memmap_pick_source(
            file_id=request.file_id,
            key1_byte=request.key1_byte,
            key2_byte=request.key2_byte,
            state=state,
        )

    path = (
        Path(pick_artifact_path)
        if pick_artifact_path is not None
        else _resolve_pick_artifact_path(state, request=request)
    )
    source_kind = 'batch_npz' if pick_source.kind == 'batch_predicted_npz' else 'manual_npz'
    return load_npz_pick_source(
        path,
        reader=reader,
        expected_dt=expected_dt,
        expected_n_samples=expected_n_samples,
        source_kind=source_kind,
    )


def build_time_term_inversion_inputs_from_sources(
    *,
    request: TimeTermStaticApplyRequest,
    reader: TraceStoreSectionReader,
    pick_source: LoadedPickSource,
    expected_dt: float,
    expected_n_samples: int,
    datum_solution_path: Path | None = None,
    residual_solution_path: Path | None = None,
    linkage_artifact_path: Path | None = None,
) -> TimeTermInversionInputs:
    """Build the sorted-order input object from already resolved sources."""
    dt = _coerce_positive_finite_float(expected_dt, name='expected_dt')
    n_samples = _coerce_positive_int(expected_n_samples, name='expected_n_samples')
    key1_byte = _validate_header_byte(request.key1_byte, name='key1_byte')
    key2_byte = _validate_header_byte(request.key2_byte, name='key2_byte')
    _validate_reader_key_bytes(
        reader,
        expected_key1_byte=key1_byte,
        expected_key2_byte=key2_byte,
    )
    _validate_reader_dt(reader, expected_dt=dt)

    n_traces = _reader_n_traces(reader)
    reader_n_samples = _reader_n_samples(reader)
    if reader_n_samples != n_samples:
        raise ValueError(
            f'n_samples mismatch: expected {n_samples}, reader has {reader_n_samples}'
        )

    _validate_pick_source(
        pick_source,
        expected_n_traces=n_traces,
        expected_n_samples=n_samples,
        expected_dt=dt,
    )
    expected_shape = (n_traces,)
    picks = _coerce_1d_pick_times(
        pick_source.picks_time_s_sorted,
        valid_mask=pick_source.valid_mask_sorted,
        expected_shape=expected_shape,
    )
    valid_mask = _coerce_1d_bool_array(
        pick_source.valid_mask_sorted,
        name='valid_pick_mask_sorted',
        expected_shape=expected_shape,
    )

    datum_shift = load_time_term_datum_trace_shift(
        datum_solution_path,
        expected_n_traces=n_traces,
        expected_dt=dt,
        expected_key1_byte=key1_byte,
        expected_key2_byte=key2_byte,
    )
    residual_shift = load_time_term_residual_applied_shift(
        residual_solution_path,
        expected_n_traces=n_traces,
        expected_dt=dt,
        expected_key1_byte=key1_byte,
        expected_key2_byte=key2_byte,
    )
    valid_mask = np.ascontiguousarray(
        valid_mask
        & np.isfinite(datum_shift.values_s_sorted)
        & np.isfinite(residual_shift.values_s_sorted),
        dtype=bool,
    )
    pick_time_after_static = np.ascontiguousarray(
        picks + datum_shift.values_s_sorted + residual_shift.values_s_sorted,
        dtype=np.float64,
    )
    pick_time_after_static[~valid_mask] = np.nan

    source_id, receiver_id = load_source_receiver_id_headers_sorted(
        reader,
        source_id_byte=request.geometry.source_id_byte,
        receiver_id_byte=request.geometry.receiver_id_byte,
        expected_n_traces=n_traces,
    )
    linkage = _load_or_build_trace_node_mapping(
        linkage_artifact_path,
        request=request,
        n_traces=n_traces,
        source_id_sorted=source_id,
        receiver_id_sorted=receiver_id,
    )
    coordinates = _load_coordinate_headers_sorted(
        reader,
        request=request,
        expected_n_traces=n_traces,
    )
    elevations = _load_elevation_headers_sorted(
        reader,
        request=request,
        expected_n_traces=n_traces,
    )
    offset = _load_optional_offset_header_sorted(
        reader,
        request=request,
        expected_n_traces=n_traces,
    )

    inputs = TimeTermInversionInputs(
        n_traces=n_traces,
        n_samples=n_samples,
        dt=dt,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        pick_time_raw_s_sorted=picks,
        valid_pick_mask_sorted=valid_mask,
        datum_trace_shift_s_sorted=datum_shift.values_s_sorted,
        residual_applied_shift_s_sorted=residual_shift.values_s_sorted,
        pick_time_after_static_s_sorted=pick_time_after_static,
        source_node_id_sorted=linkage.source_node_id_sorted,
        receiver_node_id_sorted=linkage.receiver_node_id_sorted,
        n_nodes=linkage.n_nodes,
        source_id_sorted=source_id,
        receiver_id_sorted=receiver_id,
        offset_sorted=offset,
        source_x_m_sorted=coordinates['source_x_m_sorted'],
        source_y_m_sorted=coordinates['source_y_m_sorted'],
        receiver_x_m_sorted=coordinates['receiver_x_m_sorted'],
        receiver_y_m_sorted=coordinates['receiver_y_m_sorted'],
        source_elevation_m_sorted=elevations['source_elevation_m_sorted'],
        receiver_elevation_m_sorted=elevations['receiver_elevation_m_sorted'],
        source_depth_m_sorted=elevations['source_depth_m_sorted'],
        input_file_id=request.file_id,
        pick_source_description=_pick_source_description(pick_source),
        datum_solution_path=datum_shift.path,
        residual_solution_path=residual_shift.path,
        linkage_artifact_path=Path(linkage_artifact_path)
        if linkage_artifact_path is not None
        else None,
        metadata={
            'order': ORDER,
            'sign_convention': SIGN_CONVENTION,
            'pick_source_kind': _pick_source_kind(pick_source),
            'pick_source_metadata': _pick_source_metadata(pick_source),
            'datum_solution_metadata': datum_shift.metadata,
            'datum_solution_shift_field': datum_shift.field_name,
            'residual_solution_metadata': residual_shift.metadata,
            'residual_solution_shift_field': residual_shift.field_name,
            'linkage_mode': linkage.mode,
            'coordinate_unit': request.geometry.coordinate_unit,
            'elevation_unit': request.geometry.elevation_unit,
            'distance_source': request.moveout.distance_source,
            'offset_byte': request.moveout.offset_byte,
            'allow_missing_offset': request.moveout.allow_missing_offset,
            'max_geometry_offset_mismatch_m': (
                request.moveout.max_geometry_offset_mismatch_m
            ),
        },
    )
    validate_time_term_inversion_inputs(inputs, offset_required=_offset_required(request))
    return inputs


def load_time_term_datum_trace_shift(
    solution_npz_path: Path | None,
    *,
    expected_n_traces: int,
    expected_dt: float,
    expected_key1_byte: int,
    expected_key2_byte: int,
) -> _ShiftLoadResult:
    """Load optional datum trace shifts, defaulting to zero when absent."""
    n_traces = _coerce_positive_int(expected_n_traces, name='expected_n_traces')
    if solution_npz_path is None:
        return _ShiftLoadResult(
            values_s_sorted=np.zeros(n_traces, dtype=np.float64),
            path=None,
            field_name=None,
            metadata={'source': 'zero_default'},
        )
    return _load_shift_npz(
        solution_npz_path,
        accepted_fields=('trace_shift_s_sorted',),
        estimated_delay_field_allowed=False,
        expected_n_traces=n_traces,
        expected_dt=expected_dt,
        expected_key1_byte=expected_key1_byte,
        expected_key2_byte=expected_key2_byte,
        label='datum static solution',
    )


def load_time_term_residual_applied_shift(
    solution_npz_path: Path | None,
    *,
    expected_n_traces: int,
    expected_dt: float,
    expected_key1_byte: int,
    expected_key2_byte: int,
) -> _ShiftLoadResult:
    """Load optional residual applied event-time shifts, defaulting to zero."""
    n_traces = _coerce_positive_int(expected_n_traces, name='expected_n_traces')
    if solution_npz_path is None:
        return _ShiftLoadResult(
            values_s_sorted=np.zeros(n_traces, dtype=np.float64),
            path=None,
            field_name=None,
            metadata={'source': 'zero_default'},
        )
    return _load_shift_npz(
        solution_npz_path,
        accepted_fields=(
            'applied_residual_shift_s_sorted',
            'residual_applied_shift_s_sorted',
            'trace_shift_s_sorted',
        ),
        estimated_delay_field_allowed=False,
        expected_n_traces=n_traces,
        expected_dt=expected_dt,
        expected_key1_byte=expected_key1_byte,
        expected_key2_byte=expected_key2_byte,
        label='residual static solution',
    )


def validate_time_term_inversion_inputs(
    inputs: TimeTermInversionInputs,
    *,
    offset_required: bool,
) -> None:
    """Validate shape, sorted-order, and trace-level integrity constraints."""
    n_traces = _coerce_positive_int(inputs.n_traces, name='n_traces')
    _coerce_positive_int(inputs.n_samples, name='n_samples')
    _coerce_positive_finite_float(inputs.dt, name='dt')
    expected_shape = (n_traces,)

    picks = _coerce_1d_real_numeric_float64(
        inputs.pick_time_raw_s_sorted,
        name='pick_time_raw_s_sorted',
        expected_shape=expected_shape,
    )
    if np.any(np.isinf(picks)):
        raise ValueError('pick_time_raw_s_sorted contains inf')
    valid_mask = _coerce_1d_bool_array(
        inputs.valid_pick_mask_sorted,
        name='valid_pick_mask_sorted',
        expected_shape=expected_shape,
    )
    datum = _coerce_1d_finite_float64(
        inputs.datum_trace_shift_s_sorted,
        name='datum_trace_shift_s_sorted',
        expected_shape=expected_shape,
    )
    residual = _coerce_1d_finite_float64(
        inputs.residual_applied_shift_s_sorted,
        name='residual_applied_shift_s_sorted',
        expected_shape=expected_shape,
    )
    expected_valid = np.isfinite(picks) & np.isfinite(datum) & np.isfinite(residual)
    if not np.array_equal(valid_mask, expected_valid):
        raise ValueError('valid_pick_mask_sorted does not match finite input traces')
    if int(np.count_nonzero(valid_mask)) <= 0:
        raise ValueError('at least one valid pick is required')

    pick_after = _coerce_1d_real_numeric_float64(
        inputs.pick_time_after_static_s_sorted,
        name='pick_time_after_static_s_sorted',
        expected_shape=expected_shape,
    )
    if np.any(~np.isfinite(pick_after[valid_mask])):
        raise ValueError('pick_time_after_static_s_sorted must be finite for valid picks')
    if np.any(~np.isnan(pick_after[~valid_mask])):
        raise ValueError('pick_time_after_static_s_sorted must be NaN for invalid picks')
    expected_after = picks + datum + residual
    if not np.allclose(pick_after[valid_mask], expected_after[valid_mask]):
        raise ValueError('pick_time_after_static_s_sorted does not match sign convention')

    _validate_node_mapping(
        inputs.source_node_id_sorted,
        inputs.receiver_node_id_sorted,
        n_nodes=inputs.n_nodes,
        expected_shape=expected_shape,
    )
    _coerce_1d_integer_int64(
        inputs.source_id_sorted,
        name='source_id_sorted',
        expected_shape=expected_shape,
    )
    _coerce_1d_integer_int64(
        inputs.receiver_id_sorted,
        name='receiver_id_sorted',
        expected_shape=expected_shape,
    )
    for name, values in (
        ('source_x_m_sorted', inputs.source_x_m_sorted),
        ('source_y_m_sorted', inputs.source_y_m_sorted),
        ('receiver_x_m_sorted', inputs.receiver_x_m_sorted),
        ('receiver_y_m_sorted', inputs.receiver_y_m_sorted),
        ('source_elevation_m_sorted', inputs.source_elevation_m_sorted),
        ('receiver_elevation_m_sorted', inputs.receiver_elevation_m_sorted),
        ('source_depth_m_sorted', inputs.source_depth_m_sorted),
    ):
        _coerce_1d_finite_float64(values, name=name, expected_shape=expected_shape)

    if inputs.offset_sorted is None:
        if offset_required:
            raise ValueError('offset_sorted is required by moveout configuration')
    else:
        _coerce_1d_finite_float64(
            inputs.offset_sorted,
            name='offset_sorted',
            expected_shape=expected_shape,
        )


def summarize_time_term_inversion_inputs(
    inputs: TimeTermInversionInputs,
) -> dict[str, object]:
    """Return a lightweight JSON-safe summary of assembled inversion inputs."""
    n_valid = int(np.count_nonzero(inputs.valid_pick_mask_sorted))
    valid_fraction = float(n_valid / inputs.n_traces) if inputs.n_traces else 0.0
    return {
        'n_traces': int(inputs.n_traces),
        'n_valid_picks': n_valid,
        'valid_pick_fraction': valid_fraction,
        'n_nodes': int(inputs.n_nodes),
        'pick_time_raw_s': _stats_payload(inputs.pick_time_raw_s_sorted),
        'pick_time_after_static_s': _stats_payload(
            inputs.pick_time_after_static_s_sorted
        ),
        'datum_shift_ms': _stats_payload(inputs.datum_trace_shift_s_sorted * 1000.0),
        'residual_shift_ms': _stats_payload(
            inputs.residual_applied_shift_s_sorted * 1000.0
        ),
        'has_offset': inputs.offset_sorted is not None,
        'source_elevation_m': _stats_payload(inputs.source_elevation_m_sorted),
        'receiver_elevation_m': _stats_payload(inputs.receiver_elevation_m_sorted),
    }


def _resolve_reader(
    state: AppState,
    request: TimeTermStaticApplyRequest,
) -> TraceStoreSectionReader:
    try:
        return get_reader(
            request.file_id,
            request.key1_byte,
            request.key2_byte,
            state=state,
        )
    except DomainError as exc:
        raise ValueError(exc.detail) from exc
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f'Could not open TraceStore for file_id {request.file_id}: {exc}'
        ) from exc


def _resolve_dt(
    state: AppState,
    file_id: str,
    *,
    reader: TraceStoreSectionReader,
) -> float:
    dt_raw: object = None
    record = state.file_registry.get_record(file_id)
    if isinstance(record, Mapping):
        dt_raw = record.get('dt')
    if not _is_positive_finite_number(dt_raw):
        meta = getattr(reader, 'meta', None)
        if isinstance(meta, Mapping):
            dt_raw = meta.get('dt')
    if not _is_positive_finite_number(dt_raw):
        try:
            dt_raw = state.file_registry.get_dt(file_id)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f'dt must be finite and greater than 0 for file_id {file_id}'
            ) from exc
    if not _is_positive_finite_number(dt_raw):
        raise ValueError(f'dt must be finite and greater than 0 for file_id {file_id}')
    return float(dt_raw)


def _resolve_pick_artifact_path(
    state: AppState,
    *,
    request: TimeTermStaticApplyRequest,
) -> Path:
    pick_source = request.pick_source
    if not pick_source.job_id or not pick_source.artifact_name:
        raise ValueError('pick_source job_id/artifact_name is required')
    if pick_source.kind == 'batch_predicted_npz':
        return resolve_job_artifact_path(
            state,
            job_id=pick_source.job_id,
            name=pick_source.artifact_name,
            allowed_job_types={'batch_apply'},
            expected_file_id=request.file_id,
            expected_key1_byte=request.key1_byte,
            expected_key2_byte=request.key2_byte,
            reference_label='pick_source',
        )
    if pick_source.kind != 'manual_npz_artifact':
        raise ValueError(f'unsupported pick_source.kind: {pick_source.kind}')
    return resolve_job_artifact_path(
        state,
        job_id=pick_source.job_id,
        name=pick_source.artifact_name,
        allowed_job_types={'statics', 'batch_apply', 'pipeline'},
        expected_file_id=request.file_id,
        expected_key1_byte=request.key1_byte,
        expected_key2_byte=request.key2_byte,
        reference_label='pick_source',
    )


def _resolve_linkage_artifact_path(
    state: AppState,
    *,
    request: TimeTermStaticApplyRequest,
    expected_n_traces: int,
    explicit_path: Path | None,
) -> Path | None:
    if explicit_path is not None:
        return Path(explicit_path)
    linkage = request.linkage
    if linkage.mode == 'none':
        return None
    if linkage.mode == 'optional' and linkage.job_id is None:
        return None
    if not linkage.job_id:
        raise ValueError('linkage.job_id is required when linkage.mode is required')
    path = resolve_job_artifact_path(
        state,
        job_id=linkage.job_id,
        name=linkage.artifact_name,
        allowed_job_types={'statics'},
        allowed_statics_kinds={'geometry_linkage'},
        expected_file_id=request.file_id,
        expected_key1_byte=request.key1_byte,
        expected_key2_byte=request.key2_byte,
        reference_label='linkage',
    )
    load_geometry_linkage_artifact(
        path,
        expected_n_traces=expected_n_traces,
        expected_key1_byte=request.key1_byte,
        expected_key2_byte=request.key2_byte,
    )
    return path


def _load_shift_npz(
    solution_npz_path: Path,
    *,
    accepted_fields: tuple[str, ...],
    estimated_delay_field_allowed: bool,
    expected_n_traces: int,
    expected_dt: float,
    expected_key1_byte: int,
    expected_key2_byte: int,
    label: str,
) -> _ShiftLoadResult:
    path = Path(solution_npz_path)
    if not path.is_file():
        raise ValueError(f'{label} npz not found: {path}')
    n_traces = _coerce_positive_int(expected_n_traces, name='expected_n_traces')
    expected_shape = (n_traces,)
    try:
        npz_file = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f'Could not read {label} npz: {path}') from exc

    with npz_file as npz:
        field_name = next((name for name in accepted_fields if name in npz.files), None)
        if field_name is None:
            if 'estimated_trace_delay_s_sorted' in npz.files and not estimated_delay_field_allowed:
                raise ValueError(
                    f'{label} contains only estimated_trace_delay_s_sorted; '
                    'explicit applied-shift conversion is required'
                )
            raise ValueError(f'{label} missing applied trace shift field')
        values = _coerce_1d_finite_float64(
            np.asarray(npz[field_name]),
            name=field_name,
            expected_shape=expected_shape,
        )
        metadata = _validate_optional_solution_metadata(
            npz,
            expected_n_traces=n_traces,
            expected_dt=expected_dt,
            expected_key1_byte=expected_key1_byte,
            expected_key2_byte=expected_key2_byte,
            label=label,
            path=path,
        )
        metadata['shift_field'] = field_name
    return _ShiftLoadResult(
        values_s_sorted=values,
        path=path,
        field_name=field_name,
        metadata=metadata,
    )


def _validate_optional_solution_metadata(
    npz: np.lib.npyio.NpzFile,
    *,
    expected_n_traces: int,
    expected_dt: float,
    expected_key1_byte: int,
    expected_key2_byte: int,
    label: str,
    path: Path,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        'npz_path': str(path),
        'npz_keys': tuple(npz.files),
    }
    if 'n_traces' in npz.files:
        value = _read_int_scalar(npz, 'n_traces')
        metadata['n_traces'] = value
        if value != expected_n_traces:
            raise ValueError(
                f'{label} n_traces mismatch: expected {expected_n_traces}, got {value}'
            )
    if 'dt' in npz.files:
        value = _read_float_scalar(npz, 'dt')
        metadata['dt'] = value
        if not np.isfinite(value) or abs(value - expected_dt) > _DT_TOLERANCE:
            raise ValueError(f'{label} dt mismatch: expected {expected_dt}, got {value}')
    if 'key1_byte' in npz.files:
        value = _read_int_scalar(npz, 'key1_byte')
        metadata['key1_byte'] = value
        if value != expected_key1_byte:
            raise ValueError(
                f'{label} key1_byte mismatch: expected {expected_key1_byte}, got {value}'
            )
    if 'key2_byte' in npz.files:
        value = _read_int_scalar(npz, 'key2_byte')
        metadata['key2_byte'] = value
        if value != expected_key2_byte:
            raise ValueError(
                f'{label} key2_byte mismatch: expected {expected_key2_byte}, got {value}'
            )
    return metadata


def _load_or_build_trace_node_mapping(
    linkage_artifact_path: Path | None,
    *,
    request: TimeTermStaticApplyRequest,
    n_traces: int,
    source_id_sorted: np.ndarray,
    receiver_id_sorted: np.ndarray,
) -> LoadedGeometryLinkageArtifact | Any:
    if linkage_artifact_path is not None:
        return load_geometry_linkage_artifact(
            Path(linkage_artifact_path),
            expected_n_traces=n_traces,
            expected_key1_byte=request.key1_byte,
            expected_key2_byte=request.key2_byte,
        )
    if request.linkage.mode == 'required':
        raise ValueError('linkage artifact path is required')
    source_nodes, receiver_nodes, n_nodes = _independent_nodes_from_endpoint_ids(
        source_id_sorted=source_id_sorted,
        receiver_id_sorted=receiver_id_sorted,
    )
    return _IndependentTraceNodeMapping(
        n_traces=n_traces,
        n_nodes=n_nodes,
        source_node_id_sorted=source_nodes,
        receiver_node_id_sorted=receiver_nodes,
        mode='independent_source_receiver_id',
    )


@dataclass(frozen=True)
class _IndependentTraceNodeMapping:
    n_traces: int
    n_nodes: int
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    mode: str


def _independent_nodes_from_endpoint_ids(
    *,
    source_id_sorted: np.ndarray,
    receiver_id_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    source_ids = _coerce_1d_integer_int64(
        source_id_sorted,
        name='source_id_sorted',
        expected_shape=source_id_sorted.shape,
    )
    receiver_ids = _coerce_1d_integer_int64(
        receiver_id_sorted,
        name='receiver_id_sorted',
        expected_shape=receiver_id_sorted.shape,
    )
    _, receiver_inverse = np.unique(receiver_ids, return_inverse=True)
    _, source_inverse = np.unique(source_ids, return_inverse=True)
    n_receivers = int(np.max(receiver_inverse)) + 1 if receiver_inverse.size else 0
    source_nodes = np.ascontiguousarray(
        np.asarray(source_inverse, dtype=np.int64) + n_receivers,
        dtype=np.int64,
    )
    receiver_nodes = np.ascontiguousarray(receiver_inverse, dtype=np.int64)
    return source_nodes, receiver_nodes, int(n_receivers + np.unique(source_ids).size)


def _load_coordinate_headers_sorted(
    reader: TraceStoreSectionReader,
    *,
    request: TimeTermStaticApplyRequest,
    expected_n_traces: int,
) -> dict[str, np.ndarray]:
    geometry = request.geometry
    n_traces = _coerce_positive_int(expected_n_traces, name='expected_n_traces')
    expected_shape = (n_traces,)
    scalars = _coerce_1d_integer_int64(
        _read_reader_header(
            reader,
            byte=geometry.coordinate_scalar_byte,
            role='coordinate_scalar',
        ),
        name=f'coordinate scalar header byte {geometry.coordinate_scalar_byte}',
        expected_shape=expected_shape,
    )

    result: dict[str, np.ndarray] = {}
    for output_name, byte, role in (
        ('source_x_m_sorted', geometry.source_x_byte, 'source_x'),
        ('source_y_m_sorted', geometry.source_y_byte, 'source_y'),
        ('receiver_x_m_sorted', geometry.receiver_x_byte, 'receiver_x'),
        ('receiver_y_m_sorted', geometry.receiver_y_byte, 'receiver_y'),
    ):
        raw = _coerce_1d_real_numeric_float64(
            _read_reader_header(reader, byte=byte, role=role),
            name=f'{role} header byte {byte}',
            expected_shape=expected_shape,
        )
        scaled = apply_segy_scalar(raw, scalars)
        result[output_name] = _normalize_linear_unit(
            scaled,
            unit=geometry.coordinate_unit,
            name=output_name,
        )
    return result


def _load_elevation_headers_sorted(
    reader: TraceStoreSectionReader,
    *,
    request: TimeTermStaticApplyRequest,
    expected_n_traces: int,
) -> dict[str, np.ndarray]:
    geometry = request.geometry
    n_traces = _coerce_positive_int(expected_n_traces, name='expected_n_traces')
    expected_shape = (n_traces,)
    scalars = _coerce_1d_integer_int64(
        _read_reader_header(
            reader,
            byte=geometry.elevation_scalar_byte,
            role='elevation_scalar',
        ),
        name=f'elevation scalar header byte {geometry.elevation_scalar_byte}',
        expected_shape=expected_shape,
    )
    source_elevation = _scaled_elevation_header(
        reader,
        byte=geometry.source_elevation_byte,
        role='source_elevation',
        scalars=scalars,
        unit=geometry.elevation_unit,
        expected_shape=expected_shape,
    )
    receiver_elevation = _scaled_elevation_header(
        reader,
        byte=geometry.receiver_elevation_byte,
        role='receiver_elevation',
        scalars=scalars,
        unit=geometry.elevation_unit,
        expected_shape=expected_shape,
    )
    if geometry.source_depth_byte is None:
        source_depth = np.zeros(n_traces, dtype=np.float64)
    else:
        source_depth = _scaled_elevation_header(
            reader,
            byte=geometry.source_depth_byte,
            role='source_depth',
            scalars=scalars,
            unit=geometry.elevation_unit,
            expected_shape=expected_shape,
        )
    return {
        'source_elevation_m_sorted': source_elevation,
        'receiver_elevation_m_sorted': receiver_elevation,
        'source_depth_m_sorted': np.ascontiguousarray(source_depth, dtype=np.float64),
    }


def _scaled_elevation_header(
    reader: TraceStoreSectionReader,
    *,
    byte: int,
    role: str,
    scalars: np.ndarray,
    unit: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    raw = _coerce_1d_real_numeric_float64(
        _read_reader_header(reader, byte=byte, role=role),
        name=f'{role} header byte {byte}',
        expected_shape=expected_shape,
    )
    scaled = apply_segy_scalar(raw, scalars)
    return _normalize_linear_unit(scaled, unit=unit, name=f'{role}_m_sorted')


def _load_optional_offset_header_sorted(
    reader: TraceStoreSectionReader,
    *,
    request: TimeTermStaticApplyRequest,
    expected_n_traces: int,
) -> np.ndarray | None:
    moveout = request.moveout
    if moveout.model == 'none':
        return None
    if (
        moveout.distance_source == 'geometry'
        and moveout.max_geometry_offset_mismatch_m is None
    ):
        return None
    if moveout.offset_byte is None:
        if moveout.distance_source == 'offset_header':
            raise ValueError('offset_byte is required by moveout configuration')
        if moveout.max_geometry_offset_mismatch_m is not None:
            raise ValueError('offset_byte is required for geometry offset QC')
        return None
    n_traces = _coerce_positive_int(expected_n_traces, name='expected_n_traces')
    try:
        values = _read_reader_header(reader, byte=moveout.offset_byte, role='offset')
    except ValueError:
        if (
            moveout.distance_source == 'auto'
            and moveout.max_geometry_offset_mismatch_m is None
        ):
            return None
        if moveout.allow_missing_offset and moveout.distance_source != 'offset_header':
            return None
        raise
    offset = _coerce_1d_finite_float64(
        values,
        name=f'offset header byte {moveout.offset_byte}',
        expected_shape=(n_traces,),
    )
    return _normalize_linear_unit(
        offset,
        unit=request.geometry.coordinate_unit,
        name='offset_sorted',
    )


def _normalize_linear_unit(values: np.ndarray, *, unit: str, name: str) -> np.ndarray:
    arr = _coerce_1d_real_numeric_float64(
        values,
        name=name,
        expected_shape=np.asarray(values).shape,
    )
    if unit == 'm':
        result = arr
    elif unit == 'ft':
        result = arr * 0.3048
    else:
        raise ValueError(f'{name} unit must be "m" or "ft"')
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must contain only finite values')
    return np.ascontiguousarray(result, dtype=np.float64)


def _offset_required(request: TimeTermStaticApplyRequest) -> bool:
    return (
        request.moveout.model != 'none'
        and request.moveout.distance_source == 'offset_header'
    )


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
        raise ValueError(
            f'pick source n_traces mismatch: expected {expected_n_traces}, got {pick_n_traces}'
        )
    pick_n_samples = _coerce_positive_int(
        getattr(pick_source, 'n_samples', None),
        name='pick source n_samples',
    )
    if pick_n_samples != expected_n_samples:
        raise ValueError(
            f'pick source n_samples mismatch: expected {expected_n_samples}, got {pick_n_samples}'
        )
    pick_dt = _coerce_positive_finite_float(
        getattr(pick_source, 'dt', None),
        name='pick source dt',
    )
    if abs(pick_dt - expected_dt) > _DT_TOLERANCE:
        raise ValueError(f'pick source dt mismatch: expected {expected_dt}, got {pick_dt}')
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
        name='pick_time_raw_s_sorted',
        expected_shape=expected_shape,
    )
    mask = _coerce_1d_bool_array(
        valid_mask,
        name='valid_pick_mask_sorted',
        expected_shape=expected_shape,
    )
    if np.any(np.isinf(picks)):
        raise ValueError('pick_time_raw_s_sorted contains inf')
    if np.any(~np.isfinite(picks[mask])):
        raise ValueError('valid picks must be finite')
    if np.any(~np.isnan(picks[~mask])):
        raise ValueError('invalid picks must be NaN')
    return picks


def _validate_node_mapping(
    source_node_id_sorted: np.ndarray,
    receiver_node_id_sorted: np.ndarray,
    *,
    n_nodes: int,
    expected_shape: tuple[int, ...],
) -> None:
    node_count = _coerce_positive_int(n_nodes, name='n_nodes')
    source = _coerce_1d_integer_int64(
        source_node_id_sorted,
        name='source_node_id_sorted',
        expected_shape=expected_shape,
    )
    receiver = _coerce_1d_integer_int64(
        receiver_node_id_sorted,
        name='receiver_node_id_sorted',
        expected_shape=expected_shape,
    )
    if np.any(source < 0) or np.any(receiver < 0):
        raise ValueError('node ids must be non-negative')
    if np.any(source >= node_count) or np.any(receiver >= node_count):
        raise ValueError('node ids must be less than n_nodes')
    used = np.unique(np.concatenate((source, receiver)))
    if not np.array_equal(used, np.arange(node_count, dtype=np.int64)):
        raise ValueError('node ids must be 0-based contiguous')


def _read_reader_header(
    reader: TraceStoreSectionReader,
    *,
    byte: int,
    role: str,
) -> np.ndarray:
    reader_header = getattr(reader, 'get_header', None)
    if not callable(reader_header):
        reader_header = getattr(reader, 'ensure_header', None)
    if not callable(reader_header):
        raise ValueError(f'reader cannot read {role} header byte {byte}')
    try:
        return reader_header(byte)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f'failed to read {role} header byte {byte}: {exc}') from exc


def _validate_reader_key_bytes(
    reader: TraceStoreSectionReader,
    *,
    expected_key1_byte: int,
    expected_key2_byte: int,
) -> None:
    if not hasattr(reader, 'key1_byte') or not hasattr(reader, 'key2_byte'):
        raise ValueError('reader key1_byte/key2_byte are required')
    reader_key1 = _validate_header_byte(reader.key1_byte, name='reader key1_byte')
    reader_key2 = _validate_header_byte(reader.key2_byte, name='reader key2_byte')
    if reader_key1 != expected_key1_byte:
        raise ValueError(
            f'reader key1_byte mismatch: expected {expected_key1_byte}, got {reader_key1}'
        )
    if reader_key2 != expected_key2_byte:
        raise ValueError(
            f'reader key2_byte mismatch: expected {expected_key2_byte}, got {reader_key2}'
        )


def _validate_reader_dt(
    reader: TraceStoreSectionReader,
    *,
    expected_dt: float,
) -> None:
    meta = getattr(reader, 'meta', None)
    if not isinstance(meta, Mapping) or 'dt' not in meta:
        raise ValueError('reader meta must contain dt')
    reader_dt = _coerce_positive_finite_float(meta['dt'], name='reader dt')
    if abs(reader_dt - expected_dt) > _DT_TOLERANCE:
        raise ValueError(f'reader dt mismatch: expected {expected_dt}, got {reader_dt}')


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


def _coerce_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    out = int(value)
    if out < 0:
        raise ValueError(f'{name} must be greater than or equal to 0')
    return out


def _coerce_positive_int(value: object, *, name: str) -> int:
    out = _coerce_nonnegative_int(value, name=name)
    if out <= 0:
        raise ValueError(f'{name} must be greater than 0')
    return out


def _read_int_scalar(npz: np.lib.npyio.NpzFile, key: str) -> int:
    arr = np.asarray(npz[key])
    if arr.size != 1:
        raise ValueError(f'{key} must be a scalar')
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f'{key} must have an integer dtype')
    return int(arr.reshape(-1)[0])


def _read_float_scalar(npz: np.lib.npyio.NpzFile, key: str) -> float:
    arr = np.asarray(npz[key])
    if arr.size != 1:
        raise ValueError(f'{key} must be a scalar')
    if not _is_real_numeric_dtype(arr.dtype):
        raise ValueError(f'{key} must be numeric')
    return float(arr.reshape(-1)[0])


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


def _pick_source_description(pick_source: LoadedPickSource) -> str:
    kind = _pick_source_kind(pick_source)
    metadata = _pick_source_metadata(pick_source)
    for key in ('npz_path', 'memmap_path', 'file_name'):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return f'{kind}:{Path(value).name}'
    return kind


def _stats_payload(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError('summary values must be a 1D array')
    try:
        arr_f64 = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('summary values must be numeric') from exc
    finite = np.ascontiguousarray(arr_f64[np.isfinite(arr_f64)], dtype=np.float64)
    count = int(finite.shape[0])
    if count == 0:
        return {
            'count': 0,
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None,
            'max_abs': None,
        }
    return {
        'count': count,
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'mean': float(np.mean(finite)),
        'median': float(np.median(finite)),
        'std': float(np.std(finite, ddof=0)),
        'max_abs': float(np.max(np.abs(finite))),
    }


def _is_positive_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) > 0.0
    )


__all__ = [
    'ORDER',
    'SIGN_CONVENTION',
    'TimeTermInversionInputs',
    'build_time_term_inversion_inputs',
    'build_time_term_inversion_inputs_from_sources',
    'load_time_term_datum_trace_shift',
    'load_time_term_residual_applied_shift',
    'load_time_term_static_pick_source',
    'summarize_time_term_inversion_inputs',
    'validate_time_term_inversion_inputs',
]
