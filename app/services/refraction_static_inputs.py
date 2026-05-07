"""Refraction statics input assembly from picks and TraceStore geometry."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.api.schemas import (
    RefractionStaticApplyRequest,
    RefractionStaticGeometryRequest,
    RefractionStaticLinkageRequest,
    RefractionStaticMoveoutRequest,
)
from app.core.state import AppState
from app.services.geometry_linkage_loader import (
    LoadedGeometryLinkageArtifact,
    load_geometry_linkage_artifact,
)
from app.services.job_artifact_refs import resolve_job_artifact_path
from app.services.reader import get_reader
from app.services.trace_store_index_validation import validate_sorted_to_original
from app.trace_store.reader import TraceStoreSectionReader
from app.utils.pick_cache_file1d_mem import path_for_file
from app.utils.segy_scalars import apply_segy_scalar, normalize_elevation_unit

PICK_TIME_KEYS: tuple[str, ...] = (
    'pick_time_s',
    'picks_time_s',
    'predicted_picks_time_s',
    'first_break_time_s',
)

REFRACTION_INPUT_QC_JSON_NAME = 'refraction_input_model_qc.json'
REFRACTION_INPUT_PREVIEW_CSV_NAME = 'refraction_input_model_preview.csv'

_ORDER_TRACE_STORE_SORTED = 'trace_store_sorted'
_ORDER_ORIGINAL_TRACE = 'original_trace_order'
_FEET_TO_METERS = 0.3048
_COORD_ATOL_M = 1.0e-6
_ENDPOINT_KEY_DTYPE = '<U192'

_REJECTION_REASONS = (
    'ok',
    'missing_pick',
    'nonfinite_pick',
    'negative_pick',
    'outside_trace_time_range',
    'invalid_source_geometry',
    'invalid_receiver_geometry',
    'invalid_distance',
    'offset_gate',
    'offset_mismatch',
    'missing_linkage',
    'low_fold_node',
)

_PREVIEW_COLUMNS = (
    'sorted_trace_index',
    'pick_time_s',
    'valid_observation',
    'rejection_reason',
    'source_id',
    'receiver_id',
    'source_x_m',
    'source_y_m',
    'receiver_x_m',
    'receiver_y_m',
    'source_elevation_m',
    'receiver_elevation_m',
    'geometry_distance_m',
    'offset_m',
    'distance_m',
    'source_node_id',
    'receiver_node_id',
)


@dataclass(frozen=True)
class RefractionEndpointTable:
    node_id: np.ndarray
    endpoint_id: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    elevation_m: np.ndarray
    kind: np.ndarray
    pick_count: np.ndarray


@dataclass(frozen=True)
class RefractionStaticInputModel:
    file_id: str
    n_traces: int

    sorted_trace_index: np.ndarray
    pick_time_s_sorted: np.ndarray
    valid_pick_mask_sorted: np.ndarray
    valid_observation_mask_sorted: np.ndarray

    source_id_sorted: np.ndarray
    receiver_id_sorted: np.ndarray

    source_x_m_sorted: np.ndarray
    source_y_m_sorted: np.ndarray
    receiver_x_m_sorted: np.ndarray
    receiver_y_m_sorted: np.ndarray

    source_elevation_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray
    source_depth_m_sorted: np.ndarray | None

    geometry_distance_m_sorted: np.ndarray
    offset_m_sorted: np.ndarray | None
    distance_m_sorted: np.ndarray

    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray

    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_elevation_m: np.ndarray
    node_kind: np.ndarray

    rejection_reason_sorted: np.ndarray
    qc: dict[str, Any]
    endpoint_table: RefractionEndpointTable
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _LoadedRefractionPickSource:
    picks_time_s_sorted: np.ndarray
    sorted_trace_index: np.ndarray
    source_kind: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _GeometryArrays:
    source_id: np.ndarray
    receiver_id: np.ndarray
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    source_elevation_m: np.ndarray
    receiver_elevation_m: np.ndarray
    source_depth_m: np.ndarray | None
    valid_source: np.ndarray
    valid_receiver: np.ndarray


@dataclass(frozen=True)
class _EndpointMapping:
    source_endpoint_id_sorted: np.ndarray
    receiver_endpoint_id_sorted: np.ndarray
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_endpoint_x_m: np.ndarray
    source_endpoint_y_m: np.ndarray
    source_endpoint_elevation_m: np.ndarray
    receiver_endpoint_x_m: np.ndarray
    receiver_endpoint_y_m: np.ndarray
    receiver_endpoint_elevation_m: np.ndarray


@dataclass(frozen=True)
class _NodeMapping:
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_elevation_m: np.ndarray
    node_kind: np.ndarray
    linkage_used: bool
    missing_linkage_mask: np.ndarray
    endpoint_table: RefractionEndpointTable


def build_refraction_static_input_model(
    *,
    req: RefractionStaticApplyRequest,
    state: AppState,
    job_dir: Path | None = None,
) -> RefractionStaticInputModel:
    """Build a sorted-order refraction statics input model for a request."""
    file_id = _non_empty_str(req.file_id, name='file_id')
    key1_byte = _validate_header_byte(req.key1_byte, name='key1_byte')
    key2_byte = _validate_header_byte(req.key2_byte, name='key2_byte')

    reader = get_reader(file_id, key1_byte, key2_byte, state=state)
    n_traces = _reader_n_traces(reader)
    n_samples = _reader_n_samples(reader)
    dt = _reader_dt(reader, state=state, file_id=file_id)
    sorted_trace_index = _reader_sorted_to_original(reader, n_traces=n_traces)

    pick_source = _load_refraction_pick_source(
        req=req,
        state=state,
        reader=reader,
        n_traces=n_traces,
        n_samples=n_samples,
        dt=dt,
        sorted_trace_index=sorted_trace_index,
    )
    headers = _load_refraction_trace_headers(reader=reader, req=req, n_traces=n_traces)
    linkage_artifact = _resolve_linkage_artifact(
        req=req,
        state=state,
        n_traces=n_traces,
    )

    return build_refraction_static_input_model_from_arrays(
        file_id=file_id,
        pick_time_s_sorted=pick_source.picks_time_s_sorted,
        trace_headers_sorted=headers,
        geometry=req.geometry,
        linkage=req.linkage,
        moveout=req.moveout,
        sorted_trace_index=pick_source.sorted_trace_index,
        n_samples=n_samples,
        dt=dt,
        linkage_artifact=linkage_artifact,
        job_dir=job_dir,
        metadata={
            'pick_source_kind': pick_source.source_kind,
            'pick_source_metadata': pick_source.metadata,
            'key1_byte': key1_byte,
            'key2_byte': key2_byte,
        },
    )


def build_refraction_static_input_model_from_arrays(
    *,
    pick_time_s_sorted: np.ndarray,
    trace_headers_sorted: Mapping[object, np.ndarray],
    geometry: RefractionStaticGeometryRequest,
    linkage: RefractionStaticLinkageRequest | None,
    moveout: RefractionStaticMoveoutRequest,
    file_id: str = '',
    sorted_trace_index: np.ndarray | None = None,
    n_samples: int | None = None,
    dt: float | None = None,
    linkage_artifact: LoadedGeometryLinkageArtifact | Mapping[str, object] | None = None,
    job_dir: Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RefractionStaticInputModel:
    """Build a refraction input bundle from already sorted arrays."""
    picks = _coerce_pick_array(pick_time_s_sorted)
    n_traces = int(picks.shape[0])
    if n_traces <= 0:
        raise ValueError('pick_time_s_sorted must contain at least one trace')
    sorted_index = _coerce_sorted_trace_index(sorted_trace_index, n_traces=n_traces)
    sample_count = _optional_positive_int(n_samples, name='n_samples')
    sample_interval = _optional_positive_float(dt, name='dt')

    pick_valid, pick_reason = _validate_pick_mask(
        picks,
        n_samples=sample_count,
        dt=sample_interval,
    )
    geometry_arrays = _build_geometry_arrays(
        trace_headers_sorted,
        geometry=geometry,
        n_traces=n_traces,
    )
    endpoint_mapping = _build_endpoint_mapping(geometry_arrays)

    geometry_distance = np.hypot(
        geometry_arrays.source_x_m - geometry_arrays.receiver_x_m,
        geometry_arrays.source_y_m - geometry_arrays.receiver_y_m,
    )
    geometry_distance = np.ascontiguousarray(geometry_distance, dtype=np.float64)
    geometry_distance_valid = np.isfinite(geometry_distance) & (geometry_distance > 0.0)

    offset_m = _load_offset_if_requested(
        trace_headers_sorted,
        geometry=geometry,
        moveout=moveout,
        n_traces=n_traces,
    )
    distance, distance_valid, distance_source_used = _select_distance(
        geometry_distance=geometry_distance,
        geometry_distance_valid=geometry_distance_valid,
        offset_m=offset_m,
        moveout=moveout,
    )
    offset_mismatch_mask = _build_offset_mismatch_mask(
        geometry_distance=geometry_distance,
        geometry_distance_valid=geometry_distance_valid,
        offset_m=offset_m,
        moveout=moveout,
    )
    offset_gate_mask = _build_offset_gate_mask(distance, moveout=moveout)

    node_mapping = _build_node_mapping(
        endpoint_mapping=endpoint_mapping,
        geometry_arrays=geometry_arrays,
        linkage=linkage,
        linkage_artifact=linkage_artifact,
        n_traces=n_traces,
    )

    valid_observation = (
        pick_valid
        & geometry_arrays.valid_source
        & geometry_arrays.valid_receiver
        & distance_valid
        & offset_gate_mask
        & ~offset_mismatch_mask
        & ~node_mapping.missing_linkage_mask
    )

    rejection_reason = _build_rejection_reasons(
        pick_reason=pick_reason,
        valid_source=geometry_arrays.valid_source,
        valid_receiver=geometry_arrays.valid_receiver,
        distance_valid=distance_valid,
        offset_gate_mask=offset_gate_mask,
        offset_mismatch_mask=offset_mismatch_mask,
        missing_linkage_mask=node_mapping.missing_linkage_mask,
    )
    qc = _build_qc(
        n_traces=n_traces,
        valid_pick_mask=pick_valid,
        valid_observation_mask=valid_observation,
        source_id=geometry_arrays.source_id,
        receiver_id=geometry_arrays.receiver_id,
        node_x_m=node_mapping.node_x_m,
        distance=distance,
        moveout=moveout,
        distance_source_used=distance_source_used,
        offset_gate_mask=offset_gate_mask,
        offset_mismatch_mask=offset_mismatch_mask,
        linkage=linkage,
        linkage_used=node_mapping.linkage_used,
        missing_linkage_mask=node_mapping.missing_linkage_mask,
        rejection_reason=rejection_reason,
    )

    if int(qc['n_valid_observations']) <= 0:
        raise ValueError(
            'No valid refraction observations remain after pick, geometry, '
            'offset, and linkage filtering.'
        )

    endpoint_table = _endpoint_table_with_pick_counts(
        node_mapping.endpoint_table,
        source_node_id=node_mapping.source_node_id_sorted,
        receiver_node_id=node_mapping.receiver_node_id_sorted,
        valid_observation_mask=valid_observation,
    )

    input_model = RefractionStaticInputModel(
        file_id=str(file_id),
        n_traces=n_traces,
        sorted_trace_index=sorted_index,
        pick_time_s_sorted=np.ascontiguousarray(picks, dtype=np.float64),
        valid_pick_mask_sorted=np.ascontiguousarray(pick_valid, dtype=bool),
        valid_observation_mask_sorted=np.ascontiguousarray(
            valid_observation,
            dtype=bool,
        ),
        source_id_sorted=geometry_arrays.source_id,
        receiver_id_sorted=geometry_arrays.receiver_id,
        source_x_m_sorted=geometry_arrays.source_x_m,
        source_y_m_sorted=geometry_arrays.source_y_m,
        receiver_x_m_sorted=geometry_arrays.receiver_x_m,
        receiver_y_m_sorted=geometry_arrays.receiver_y_m,
        source_elevation_m_sorted=geometry_arrays.source_elevation_m,
        receiver_elevation_m_sorted=geometry_arrays.receiver_elevation_m,
        source_depth_m_sorted=geometry_arrays.source_depth_m,
        geometry_distance_m_sorted=geometry_distance,
        offset_m_sorted=offset_m,
        distance_m_sorted=distance,
        source_endpoint_key_sorted=endpoint_mapping.source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=endpoint_mapping.receiver_endpoint_key_sorted,
        source_node_id_sorted=node_mapping.source_node_id_sorted,
        receiver_node_id_sorted=node_mapping.receiver_node_id_sorted,
        node_x_m=node_mapping.node_x_m,
        node_y_m=node_mapping.node_y_m,
        node_elevation_m=node_mapping.node_elevation_m,
        node_kind=node_mapping.node_kind,
        rejection_reason_sorted=np.ascontiguousarray(rejection_reason, dtype='<U32'),
        qc=qc,
        endpoint_table=endpoint_table,
        metadata=dict(metadata or {}),
    )
    if job_dir is not None:
        write_refraction_static_input_artifacts(Path(job_dir), input_model)
    return input_model


def write_refraction_static_input_artifacts(
    job_dir: Path,
    input_model: RefractionStaticInputModel,
) -> dict[str, Path]:
    """Write strict-JSON QC and a one-row-per-trace preview CSV."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / REFRACTION_INPUT_QC_JSON_NAME
    preview_path = root / REFRACTION_INPUT_PREVIEW_CSV_NAME
    _write_json_atomic(qc_path, input_model.qc)
    _write_csv_atomic(preview_path, _preview_rows(input_model))
    return {
        'qc_json': qc_path,
        'preview_csv': preview_path,
    }


def _load_refraction_pick_source(
    *,
    req: RefractionStaticApplyRequest,
    state: AppState,
    reader: TraceStoreSectionReader,
    n_traces: int,
    n_samples: int,
    dt: float,
    sorted_trace_index: np.ndarray,
) -> _LoadedRefractionPickSource:
    pick_source = req.pick_source
    if pick_source.kind == 'manual_memmap':
        return _load_manual_memmap_pick_source(
            file_id=req.file_id,
            state=state,
            n_traces=n_traces,
            sorted_trace_index=sorted_trace_index,
        )

    job_id = _non_empty_str(pick_source.job_id, name='pick_source.job_id')
    artifact_name = _non_empty_str(
        pick_source.artifact_name,
        name='pick_source.artifact_name',
    )
    if pick_source.kind == 'batch_predicted_npz':
        path = resolve_job_artifact_path(
            state,
            job_id=job_id,
            name=artifact_name,
            allowed_job_types={'batch_apply'},
            expected_file_id=req.file_id,
            expected_key1_byte=req.key1_byte,
            expected_key2_byte=req.key2_byte,
            reference_label='pick_source',
        )
        source_kind = 'batch_predicted_npz'
    elif pick_source.kind == 'manual_npz_artifact':
        path = resolve_job_artifact_path(
            state,
            job_id=job_id,
            name=artifact_name,
            allowed_job_types={'statics', 'batch_apply', 'pipeline'},
            reference_label='pick_source',
        )
        source_kind = 'manual_npz_artifact'
    else:
        raise ValueError(f'unsupported pick_source.kind: {pick_source.kind}')

    loaded = _load_npz_refraction_pick_source(
        path,
        reader=reader,
        n_traces=n_traces,
        n_samples=n_samples,
        dt=dt,
        sorted_trace_index=sorted_trace_index,
    )
    return _LoadedRefractionPickSource(
        picks_time_s_sorted=loaded.picks_time_s_sorted,
        sorted_trace_index=loaded.sorted_trace_index,
        source_kind=source_kind,
        metadata=loaded.metadata,
    )


def _load_manual_memmap_pick_source(
    *,
    file_id: str,
    state: AppState,
    n_traces: int,
    sorted_trace_index: np.ndarray,
) -> _LoadedRefractionPickSource:
    file_name = state.file_registry.filename(file_id)
    if not file_name:
        raise ValueError(
            'manual_memmap pick source is not available for refraction statics '
            'input building'
        )
    memmap_path = path_for_file(file_name)
    if not memmap_path.is_file():
        raise ValueError(
            'manual_memmap pick source is not available for refraction statics '
            'input building'
        )
    try:
        memmap = np.load(memmap_path, mmap_mode='r', allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        msg = f'Could not read manual pick memmap: {memmap_path}'
        raise ValueError(msg) from exc
    try:
        picks = _coerce_pick_array(np.asarray(memmap))
        if picks.shape != (n_traces,):
            msg = (
                'manual pick memmap length mismatch: '
                f'expected {(n_traces,)}, got {picks.shape}'
            )
            raise ValueError(msg)
        return _LoadedRefractionPickSource(
            picks_time_s_sorted=picks,
            sorted_trace_index=sorted_trace_index,
            source_kind='manual_memmap',
            metadata={
                'file_id': file_id,
                'file_name': file_name,
                'memmap_path': str(memmap_path),
                'order': _ORDER_TRACE_STORE_SORTED,
            },
        )
    finally:
        del memmap


def _load_npz_refraction_pick_source(
    npz_path: Path,
    *,
    reader: TraceStoreSectionReader,
    n_traces: int,
    n_samples: int,
    dt: float,
    sorted_trace_index: np.ndarray,
) -> _LoadedRefractionPickSource:
    path = Path(npz_path)
    try:
        npz_file = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        msg = f'Could not read npz pick source: {path}'
        raise ValueError(msg) from exc

    with npz_file as npz:
        key = _select_pick_key(npz.files)
        metadata = _validate_pick_npz_metadata(
            npz,
            n_traces=n_traces,
            n_samples=n_samples,
            dt=dt,
            path=path,
            key=key,
        )
        picks = _coerce_pick_array(np.asarray(npz[key]))
        if picks.shape != (n_traces,):
            msg = (
                'pick array length mismatch: '
                f'expected {(n_traces,)}, got {picks.shape}'
            )
            raise ValueError(msg)

        artifact_order = _read_npz_order(npz)
        reader_sorted_to_original = _reader_sorted_to_original(
            reader,
            n_traces=n_traces,
        )
        if 'sorted_to_original' in npz.files:
            npz_sorted_to_original = validate_sorted_to_original(
                np.asarray(npz['sorted_to_original']),
                expected_n_traces=n_traces,
                role='npz',
            )
            if not np.array_equal(npz_sorted_to_original, reader_sorted_to_original):
                raise ValueError('sorted_to_original mismatch')
            metadata['has_sorted_to_original'] = True
        else:
            metadata['has_sorted_to_original'] = False

    if artifact_order == _ORDER_TRACE_STORE_SORTED:
        picks_sorted = picks
    elif artifact_order == _ORDER_ORIGINAL_TRACE:
        picks_sorted = np.ascontiguousarray(picks[reader_sorted_to_original])
    else:
        raise ValueError(f'unsupported pick artifact order: {artifact_order}')

    return _LoadedRefractionPickSource(
        picks_time_s_sorted=picks_sorted,
        sorted_trace_index=sorted_trace_index,
        source_kind='npz',
        metadata=metadata,
    )


def _resolve_linkage_artifact(
    *,
    req: RefractionStaticApplyRequest,
    state: AppState,
    n_traces: int,
) -> LoadedGeometryLinkageArtifact | None:
    linkage = req.linkage
    if linkage.mode == 'none':
        return None
    if linkage.mode == 'optional' and linkage.job_id is None:
        return None

    job_id = _non_empty_str(linkage.job_id, name='linkage.job_id')
    artifact_name = _non_empty_str(
        linkage.artifact_name,
        name='linkage.artifact_name',
    )
    try:
        path = resolve_job_artifact_path(
            state,
            job_id=job_id,
            name=artifact_name,
            allowed_job_types={'statics'},
            allowed_statics_kinds={'geometry_linkage'},
            expected_file_id=req.file_id,
            expected_key1_byte=req.key1_byte,
            expected_key2_byte=req.key2_byte,
            reference_label='linkage',
        )
    except ValueError:
        if linkage.mode == 'optional':
            return None
        raise

    return load_geometry_linkage_artifact(
        path,
        expected_n_traces=n_traces,
        expected_key1_byte=req.key1_byte,
        expected_key2_byte=req.key2_byte,
    )


def _load_refraction_trace_headers(
    *,
    reader: TraceStoreSectionReader,
    req: RefractionStaticApplyRequest,
    n_traces: int,
) -> dict[int, np.ndarray]:
    geometry = req.geometry
    header_bytes = {
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
    }
    if geometry.source_depth_byte is not None:
        header_bytes.add(geometry.source_depth_byte)
    if _offset_header_needed(req.moveout):
        if req.moveout.offset_byte is None:
            raise ValueError('moveout.offset_byte is required')
        header_bytes.add(req.moveout.offset_byte)

    headers: dict[int, np.ndarray] = {}
    for byte in sorted(header_bytes):
        values = _read_reader_header(reader, byte=byte, role=f'byte_{byte}')
        arr = np.asarray(values)
        if arr.ndim != 1 or arr.shape != (n_traces,):
            msg = (
                f'header byte {byte} shape mismatch: '
                f'expected {(n_traces,)}, got {arr.shape}'
            )
            raise ValueError(msg)
        headers[int(byte)] = arr
    return headers


def _build_geometry_arrays(
    trace_headers_sorted: Mapping[object, np.ndarray],
    *,
    geometry: RefractionStaticGeometryRequest,
    n_traces: int,
) -> _GeometryArrays:
    source_id = _coerce_id_header(
        _header(trace_headers_sorted, geometry.source_id_byte, 'source_id'),
        name='source_id',
        n_traces=n_traces,
    )
    receiver_id = _coerce_id_header(
        _header(trace_headers_sorted, geometry.receiver_id_byte, 'receiver_id'),
        name='receiver_id',
        n_traces=n_traces,
    )

    coordinate_scalar, coordinate_scalar_valid = _coerce_scalar_header(
        _header(
            trace_headers_sorted,
            geometry.coordinate_scalar_byte,
            'coordinate_scalar',
        ),
        name='coordinate_scalar',
        n_traces=n_traces,
    )
    elevation_scalar, elevation_scalar_valid = _coerce_scalar_header(
        _header(
            trace_headers_sorted,
            geometry.elevation_scalar_byte,
            'elevation_scalar',
        ),
        name='elevation_scalar',
        n_traces=n_traces,
    )

    source_x, source_x_valid = _scaled_header_to_meters(
        _header(trace_headers_sorted, geometry.source_x_byte, 'source_x'),
        scalars=coordinate_scalar,
        scalar_valid=coordinate_scalar_valid,
        unit=geometry.coordinate_unit,
        name='source_x',
        n_traces=n_traces,
    )
    source_y, source_y_valid = _scaled_header_to_meters(
        _header(trace_headers_sorted, geometry.source_y_byte, 'source_y'),
        scalars=coordinate_scalar,
        scalar_valid=coordinate_scalar_valid,
        unit=geometry.coordinate_unit,
        name='source_y',
        n_traces=n_traces,
    )
    receiver_x, receiver_x_valid = _scaled_header_to_meters(
        _header(trace_headers_sorted, geometry.receiver_x_byte, 'receiver_x'),
        scalars=coordinate_scalar,
        scalar_valid=coordinate_scalar_valid,
        unit=geometry.coordinate_unit,
        name='receiver_x',
        n_traces=n_traces,
    )
    receiver_y, receiver_y_valid = _scaled_header_to_meters(
        _header(trace_headers_sorted, geometry.receiver_y_byte, 'receiver_y'),
        scalars=coordinate_scalar,
        scalar_valid=coordinate_scalar_valid,
        unit=geometry.coordinate_unit,
        name='receiver_y',
        n_traces=n_traces,
    )
    source_elevation, source_elevation_valid = _scaled_header_to_meters(
        _header(
            trace_headers_sorted,
            geometry.source_elevation_byte,
            'source_elevation',
        ),
        scalars=elevation_scalar,
        scalar_valid=elevation_scalar_valid,
        unit=geometry.elevation_unit,
        name='source_elevation',
        n_traces=n_traces,
    )
    receiver_elevation, receiver_elevation_valid = _scaled_header_to_meters(
        _header(
            trace_headers_sorted,
            geometry.receiver_elevation_byte,
            'receiver_elevation',
        ),
        scalars=elevation_scalar,
        scalar_valid=elevation_scalar_valid,
        unit=geometry.elevation_unit,
        name='receiver_elevation',
        n_traces=n_traces,
    )

    source_depth: np.ndarray | None
    source_depth_valid = np.ones(n_traces, dtype=bool)
    if geometry.source_depth_byte is None:
        source_depth = None
    else:
        source_depth, source_depth_valid = _scaled_header_to_meters(
            _header(trace_headers_sorted, geometry.source_depth_byte, 'source_depth'),
            scalars=elevation_scalar,
            scalar_valid=elevation_scalar_valid,
            unit=geometry.elevation_unit,
            name='source_depth',
            n_traces=n_traces,
        )

    valid_source = (
        source_x_valid
        & source_y_valid
        & source_elevation_valid
        & source_depth_valid
    )
    valid_receiver = receiver_x_valid & receiver_y_valid & receiver_elevation_valid
    return _GeometryArrays(
        source_id=source_id,
        receiver_id=receiver_id,
        source_x_m=source_x,
        source_y_m=source_y,
        receiver_x_m=receiver_x,
        receiver_y_m=receiver_y,
        source_elevation_m=source_elevation,
        receiver_elevation_m=receiver_elevation,
        source_depth_m=source_depth,
        valid_source=np.ascontiguousarray(valid_source, dtype=bool),
        valid_receiver=np.ascontiguousarray(valid_receiver, dtype=bool),
    )


def _build_endpoint_mapping(geometry: _GeometryArrays) -> _EndpointMapping:
    (
        source_endpoint,
        source_key,
        source_x,
        source_y,
        source_z,
    ) = _unique_endpoints(
        'source',
        geometry.source_id,
        geometry.source_x_m,
        geometry.source_y_m,
        geometry.source_elevation_m,
        valid_mask=geometry.valid_source,
    )
    (
        receiver_endpoint,
        receiver_key,
        receiver_x,
        receiver_y,
        receiver_z,
    ) = _unique_endpoints(
        'receiver',
        geometry.receiver_id,
        geometry.receiver_x_m,
        geometry.receiver_y_m,
        geometry.receiver_elevation_m,
        valid_mask=geometry.valid_receiver,
    )
    return _EndpointMapping(
        source_endpoint_id_sorted=source_endpoint,
        receiver_endpoint_id_sorted=receiver_endpoint,
        source_endpoint_key_sorted=source_key,
        receiver_endpoint_key_sorted=receiver_key,
        source_endpoint_x_m=source_x,
        source_endpoint_y_m=source_y,
        source_endpoint_elevation_m=source_z,
        receiver_endpoint_x_m=receiver_x,
        receiver_endpoint_y_m=receiver_y,
        receiver_endpoint_elevation_m=receiver_z,
    )


def _build_node_mapping(
    *,
    endpoint_mapping: _EndpointMapping,
    geometry_arrays: _GeometryArrays,
    linkage: RefractionStaticLinkageRequest | None,
    linkage_artifact: LoadedGeometryLinkageArtifact | Mapping[str, object] | None,
    n_traces: int,
) -> _NodeMapping:
    linkage_mode = 'none' if linkage is None else linkage.mode
    if linkage_mode == 'required' and linkage_artifact is None:
        raise ValueError('missing linkage artifact when linkage.mode is required')
    if linkage_mode == 'none' or linkage_artifact is None:
        return _build_unlinked_node_mapping(
            endpoint_mapping=endpoint_mapping,
            geometry_arrays=geometry_arrays,
            n_traces=n_traces,
            linkage_used=False,
        )

    source_node_id, receiver_node_id, n_nodes = _trace_node_ids_from_linkage(
        linkage_artifact,
        n_traces=n_traces,
    )
    _validate_linkage_coordinates(
        linkage_artifact,
        geometry_arrays=geometry_arrays,
        n_traces=n_traces,
    )
    missing = (source_node_id < 0) | (receiver_node_id < 0)
    node_x, node_y, node_z, node_kind = _node_geometry_from_trace_mapping(
        source_node_id=source_node_id,
        receiver_node_id=receiver_node_id,
        n_nodes=n_nodes,
        geometry_arrays=geometry_arrays,
    )
    endpoint_table = _endpoint_table_from_nodes(
        source_node_id=source_node_id,
        receiver_node_id=receiver_node_id,
        node_x=node_x,
        node_y=node_y,
        node_z=node_z,
        node_kind=node_kind,
    )
    return _NodeMapping(
        source_node_id_sorted=np.ascontiguousarray(source_node_id, dtype=np.int64),
        receiver_node_id_sorted=np.ascontiguousarray(receiver_node_id, dtype=np.int64),
        node_x_m=node_x,
        node_y_m=node_y,
        node_elevation_m=node_z,
        node_kind=node_kind,
        linkage_used=True,
        missing_linkage_mask=np.ascontiguousarray(missing, dtype=bool),
        endpoint_table=endpoint_table,
    )


def _build_unlinked_node_mapping(
    *,
    endpoint_mapping: _EndpointMapping,
    geometry_arrays: _GeometryArrays,
    n_traces: int,
    linkage_used: bool,
) -> _NodeMapping:
    n_source = int(endpoint_mapping.source_endpoint_x_m.shape[0])
    n_receiver = int(endpoint_mapping.receiver_endpoint_x_m.shape[0])
    source_node_id = endpoint_mapping.source_endpoint_id_sorted.copy()
    receiver_node_id = endpoint_mapping.receiver_endpoint_id_sorted.copy()
    receiver_valid = receiver_node_id >= 0
    receiver_node_id[receiver_valid] += n_source

    node_x = np.concatenate(
        (
            endpoint_mapping.source_endpoint_x_m,
            endpoint_mapping.receiver_endpoint_x_m,
        )
    ).astype(np.float64, copy=False)
    node_y = np.concatenate(
        (
            endpoint_mapping.source_endpoint_y_m,
            endpoint_mapping.receiver_endpoint_y_m,
        )
    ).astype(np.float64, copy=False)
    node_z = np.concatenate(
        (
            endpoint_mapping.source_endpoint_elevation_m,
            endpoint_mapping.receiver_endpoint_elevation_m,
        )
    ).astype(np.float64, copy=False)
    node_kind = np.asarray(['source'] * n_source + ['receiver'] * n_receiver)
    endpoint_id = np.arange(n_source + n_receiver, dtype=np.int64)
    endpoint_table = RefractionEndpointTable(
        node_id=endpoint_id,
        endpoint_id=endpoint_id,
        x_m=np.ascontiguousarray(node_x, dtype=np.float64),
        y_m=np.ascontiguousarray(node_y, dtype=np.float64),
        elevation_m=np.ascontiguousarray(node_z, dtype=np.float64),
        kind=np.ascontiguousarray(node_kind, dtype='<U16'),
        pick_count=np.zeros(n_source + n_receiver, dtype=np.int64),
    )
    return _NodeMapping(
        source_node_id_sorted=np.ascontiguousarray(source_node_id, dtype=np.int64),
        receiver_node_id_sorted=np.ascontiguousarray(receiver_node_id, dtype=np.int64),
        node_x_m=np.ascontiguousarray(node_x, dtype=np.float64),
        node_y_m=np.ascontiguousarray(node_y, dtype=np.float64),
        node_elevation_m=np.ascontiguousarray(node_z, dtype=np.float64),
        node_kind=np.ascontiguousarray(node_kind, dtype='<U16'),
        linkage_used=linkage_used,
        missing_linkage_mask=np.zeros(n_traces, dtype=bool),
        endpoint_table=endpoint_table,
    )


def _trace_node_ids_from_linkage(
    linkage_artifact: LoadedGeometryLinkageArtifact | Mapping[str, object],
    *,
    n_traces: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if isinstance(linkage_artifact, Mapping):
        source_raw = linkage_artifact.get('source_node_id_sorted')
        receiver_raw = linkage_artifact.get('receiver_node_id_sorted')
        n_nodes_raw = linkage_artifact.get('n_nodes')
    else:
        source_raw = linkage_artifact.source_node_id_sorted
        receiver_raw = linkage_artifact.receiver_node_id_sorted
        n_nodes_raw = linkage_artifact.n_nodes

    source_node_id = _coerce_node_id_array(
        source_raw,
        name='source_node_id_sorted',
        n_traces=n_traces,
    )
    receiver_node_id = _coerce_node_id_array(
        receiver_raw,
        name='receiver_node_id_sorted',
        n_traces=n_traces,
    )
    non_missing = np.concatenate(
        (source_node_id[source_node_id >= 0], receiver_node_id[receiver_node_id >= 0])
    )
    if non_missing.size == 0:
        n_nodes = 0
    elif n_nodes_raw is None:
        n_nodes = int(non_missing.max()) + 1
    else:
        n_nodes = _coerce_nonnegative_int(n_nodes_raw, name='n_nodes')
    if non_missing.size and int(non_missing.max()) >= n_nodes:
        raise ValueError('linkage node ids exceed n_nodes')
    return source_node_id, receiver_node_id, n_nodes


def _validate_linkage_coordinates(
    linkage_artifact: LoadedGeometryLinkageArtifact | Mapping[str, object],
    *,
    geometry_arrays: _GeometryArrays,
    n_traces: int,
) -> None:
    if isinstance(linkage_artifact, Mapping):
        source_x_raw = linkage_artifact.get('source_x_m_sorted')
        source_y_raw = linkage_artifact.get('source_y_m_sorted')
        receiver_x_raw = linkage_artifact.get('receiver_x_m_sorted')
        receiver_y_raw = linkage_artifact.get('receiver_y_m_sorted')
    else:
        source_x_raw = linkage_artifact.source_x_m_sorted
        source_y_raw = linkage_artifact.source_y_m_sorted
        receiver_x_raw = linkage_artifact.receiver_x_m_sorted
        receiver_y_raw = linkage_artifact.receiver_y_m_sorted

    if source_x_raw is None:
        return
    source_x = _coerce_float_header(source_x_raw, name='linkage.source_x_m', n_traces=n_traces)
    source_y = _coerce_float_header(source_y_raw, name='linkage.source_y_m', n_traces=n_traces)
    receiver_x = _coerce_float_header(
        receiver_x_raw,
        name='linkage.receiver_x_m',
        n_traces=n_traces,
    )
    receiver_y = _coerce_float_header(
        receiver_y_raw,
        name='linkage.receiver_y_m',
        n_traces=n_traces,
    )
    source_mask = geometry_arrays.valid_source
    receiver_mask = geometry_arrays.valid_receiver
    if source_mask.any() and not (
        np.allclose(
            source_x[source_mask],
            geometry_arrays.source_x_m[source_mask],
            atol=_COORD_ATOL_M,
            rtol=0.0,
        )
        and np.allclose(
            source_y[source_mask],
            geometry_arrays.source_y_m[source_mask],
            atol=_COORD_ATOL_M,
            rtol=0.0,
        )
    ):
        raise ValueError('linkage artifact incompatible with request source geometry')
    if receiver_mask.any() and not (
        np.allclose(
            receiver_x[receiver_mask],
            geometry_arrays.receiver_x_m[receiver_mask],
            atol=_COORD_ATOL_M,
            rtol=0.0,
        )
        and np.allclose(
            receiver_y[receiver_mask],
            geometry_arrays.receiver_y_m[receiver_mask],
            atol=_COORD_ATOL_M,
            rtol=0.0,
        )
    ):
        raise ValueError('linkage artifact incompatible with request receiver geometry')


def _node_geometry_from_trace_mapping(
    *,
    source_node_id: np.ndarray,
    receiver_node_id: np.ndarray,
    n_nodes: int,
    geometry_arrays: _GeometryArrays,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_x = np.full(n_nodes, np.nan, dtype=np.float64)
    node_y = np.full(n_nodes, np.nan, dtype=np.float64)
    node_z = np.full(n_nodes, np.nan, dtype=np.float64)
    source_seen = np.zeros(n_nodes, dtype=bool)
    receiver_seen = np.zeros(n_nodes, dtype=bool)

    for trace_idx, node_id in enumerate(source_node_id.tolist()):
        if node_id < 0 or not geometry_arrays.valid_source[trace_idx]:
            continue
        _accumulate_node_value(
            node_id,
            x=geometry_arrays.source_x_m[trace_idx],
            y=geometry_arrays.source_y_m[trace_idx],
            z=geometry_arrays.source_elevation_m[trace_idx],
            node_x=node_x,
            node_y=node_y,
            node_z=node_z,
        )
        source_seen[node_id] = True
    for trace_idx, node_id in enumerate(receiver_node_id.tolist()):
        if node_id < 0 or not geometry_arrays.valid_receiver[trace_idx]:
            continue
        _accumulate_node_value(
            node_id,
            x=geometry_arrays.receiver_x_m[trace_idx],
            y=geometry_arrays.receiver_y_m[trace_idx],
            z=geometry_arrays.receiver_elevation_m[trace_idx],
            node_x=node_x,
            node_y=node_y,
            node_z=node_z,
        )
        receiver_seen[node_id] = True

    missing_nodes = np.isnan(node_x) | np.isnan(node_y) | np.isnan(node_z)
    if np.any(missing_nodes):
        missing_id = int(np.flatnonzero(missing_nodes)[0])
        raise ValueError(f'linkage node {missing_id} has no finite geometry')
    node_kind = np.full(n_nodes, 'linked', dtype='<U16')
    node_kind[source_seen & ~receiver_seen] = 'source'
    node_kind[receiver_seen & ~source_seen] = 'receiver'
    return (
        np.ascontiguousarray(node_x, dtype=np.float64),
        np.ascontiguousarray(node_y, dtype=np.float64),
        np.ascontiguousarray(node_z, dtype=np.float64),
        np.ascontiguousarray(node_kind, dtype='<U16'),
    )


def _accumulate_node_value(
    node_id: int,
    *,
    x: float,
    y: float,
    z: float,
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
) -> None:
    if np.isnan(node_x[node_id]):
        node_x[node_id] = float(x)
        node_y[node_id] = float(y)
        node_z[node_id] = float(z)
        return
    node_x[node_id] = (node_x[node_id] + float(x)) / 2.0
    node_y[node_id] = (node_y[node_id] + float(y)) / 2.0
    node_z[node_id] = (node_z[node_id] + float(z)) / 2.0


def _endpoint_table_from_nodes(
    *,
    source_node_id: np.ndarray,
    receiver_node_id: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
    node_kind: np.ndarray,
) -> RefractionEndpointTable:
    n_nodes = int(node_x.shape[0])
    node_id = np.arange(n_nodes, dtype=np.int64)
    pick_count = np.bincount(
        np.concatenate(
            (source_node_id[source_node_id >= 0], receiver_node_id[receiver_node_id >= 0])
        ),
        minlength=n_nodes,
    ).astype(np.int64)
    return RefractionEndpointTable(
        node_id=node_id,
        endpoint_id=node_id,
        x_m=np.ascontiguousarray(node_x, dtype=np.float64),
        y_m=np.ascontiguousarray(node_y, dtype=np.float64),
        elevation_m=np.ascontiguousarray(node_z, dtype=np.float64),
        kind=np.ascontiguousarray(node_kind, dtype='<U16'),
        pick_count=np.ascontiguousarray(pick_count, dtype=np.int64),
    )


def _endpoint_table_with_pick_counts(
    endpoint_table: RefractionEndpointTable,
    *,
    source_node_id: np.ndarray,
    receiver_node_id: np.ndarray,
    valid_observation_mask: np.ndarray,
) -> RefractionEndpointTable:
    n_nodes = int(endpoint_table.node_id.shape[0])
    valid = np.asarray(valid_observation_mask, dtype=bool)
    source = np.asarray(source_node_id, dtype=np.int64)
    receiver = np.asarray(receiver_node_id, dtype=np.int64)
    if source.shape != valid.shape or receiver.shape != valid.shape:
        raise ValueError('node id arrays must match valid observation mask shape')

    referenced = np.concatenate(
        (
            source[valid & (source >= 0)],
            receiver[valid & (receiver >= 0)],
        )
    )
    if referenced.size:
        pick_count = np.bincount(referenced, minlength=n_nodes)
        if pick_count.shape[0] != n_nodes:
            raise ValueError('node id outside endpoint table')
    else:
        pick_count = np.zeros(n_nodes, dtype=np.int64)
    return replace(
        endpoint_table,
        pick_count=np.ascontiguousarray(pick_count, dtype=np.int64),
    )


def _load_offset_if_requested(
    trace_headers_sorted: Mapping[object, np.ndarray],
    *,
    geometry: RefractionStaticGeometryRequest,
    moveout: RefractionStaticMoveoutRequest,
    n_traces: int,
) -> np.ndarray | None:
    if not _offset_header_needed(moveout):
        return None
    if moveout.offset_byte is None:
        raise ValueError('moveout.offset_byte is required')
    raw = _coerce_float_header(
        _header(trace_headers_sorted, moveout.offset_byte, 'offset'),
        name='offset',
        n_traces=n_traces,
    )
    coordinate_scalar, coordinate_scalar_valid = _coerce_scalar_header(
        _header(
            trace_headers_sorted,
            geometry.coordinate_scalar_byte,
            'coordinate_scalar',
        ),
        name='coordinate_scalar',
        n_traces=n_traces,
    )
    finite_raw = np.isfinite(raw)
    safe_raw = raw.copy()
    safe_raw[~finite_raw] = 0.0
    try:
        scaled = apply_segy_scalar(safe_raw, coordinate_scalar)
    except ValueError as exc:
        raise ValueError('offset scalar application failed') from exc
    scaled[~finite_raw | ~coordinate_scalar_valid] = np.nan
    offset = np.abs(scaled)
    coordinate_unit = geometry.coordinate_unit
    if coordinate_unit == 'ft':
        offset = offset * _FEET_TO_METERS
    elif coordinate_unit != 'm':
        raise ValueError('coordinate_unit must be "m" or "ft"')
    return np.ascontiguousarray(offset, dtype=np.float64)


def _select_distance(
    *,
    geometry_distance: np.ndarray,
    geometry_distance_valid: np.ndarray,
    offset_m: np.ndarray | None,
    moveout: RefractionStaticMoveoutRequest,
) -> tuple[np.ndarray, np.ndarray, str]:
    source = moveout.distance_source
    if source == 'geometry':
        distance = geometry_distance.copy()
        valid = geometry_distance_valid.copy()
    elif source == 'offset_header':
        if offset_m is None:
            raise ValueError('offset_byte is required for offset_header distance_source')
        distance = offset_m.copy()
        valid = np.isfinite(distance) & (distance > 0.0)
    elif source == 'auto':
        distance = geometry_distance.copy()
        valid = geometry_distance_valid.copy()
        if offset_m is not None:
            offset_valid = np.isfinite(offset_m) & (offset_m > 0.0)
            fallback = ~valid & offset_valid
            distance[fallback] = offset_m[fallback]
            valid = valid | fallback
    else:
        raise ValueError(f'unsupported moveout.distance_source: {source}')
    return (
        np.ascontiguousarray(distance, dtype=np.float64),
        np.ascontiguousarray(valid, dtype=bool),
        str(source),
    )


def _build_offset_mismatch_mask(
    *,
    geometry_distance: np.ndarray,
    geometry_distance_valid: np.ndarray,
    offset_m: np.ndarray | None,
    moveout: RefractionStaticMoveoutRequest,
) -> np.ndarray:
    threshold = moveout.max_geometry_offset_mismatch_m
    if threshold is None:
        return np.zeros(geometry_distance.shape, dtype=bool)
    if offset_m is None:
        raise ValueError('moveout.offset_byte is required for geometry offset mismatch')
    offset_valid = np.isfinite(offset_m) & (offset_m > 0.0)
    comparable = geometry_distance_valid & offset_valid
    mismatch = np.zeros(geometry_distance.shape, dtype=bool)
    mismatch[comparable] = (
        np.abs(offset_m[comparable] - geometry_distance[comparable])
        > float(threshold)
    )
    if not moveout.allow_missing_offset:
        mismatch[geometry_distance_valid & ~offset_valid] = True
    return mismatch


def _build_offset_gate_mask(
    distance: np.ndarray,
    *,
    moveout: RefractionStaticMoveoutRequest,
) -> np.ndarray:
    mask = np.ones(distance.shape, dtype=bool)
    finite_distance = np.isfinite(distance)
    if moveout.min_offset_m is not None:
        mask &= ~finite_distance | (distance >= float(moveout.min_offset_m))
    if moveout.max_offset_m is not None:
        mask &= ~finite_distance | (distance <= float(moveout.max_offset_m))
    return np.ascontiguousarray(mask, dtype=bool)


def _build_rejection_reasons(
    *,
    pick_reason: np.ndarray,
    valid_source: np.ndarray,
    valid_receiver: np.ndarray,
    distance_valid: np.ndarray,
    offset_gate_mask: np.ndarray,
    offset_mismatch_mask: np.ndarray,
    missing_linkage_mask: np.ndarray,
) -> np.ndarray:
    reasons = np.asarray(pick_reason, dtype='<U32').copy()
    _set_reason_if_ok(reasons, ~valid_source, 'invalid_source_geometry')
    _set_reason_if_ok(reasons, ~valid_receiver, 'invalid_receiver_geometry')
    _set_reason_if_ok(reasons, ~distance_valid, 'invalid_distance')
    _set_reason_if_ok(reasons, offset_mismatch_mask, 'offset_mismatch')
    _set_reason_if_ok(reasons, ~offset_gate_mask, 'offset_gate')
    _set_reason_if_ok(reasons, missing_linkage_mask, 'missing_linkage')
    return np.ascontiguousarray(reasons, dtype='<U32')


def _set_reason_if_ok(reasons: np.ndarray, mask: np.ndarray, reason: str) -> None:
    reasons[(reasons == 'ok') & mask] = reason


def _build_qc(
    *,
    n_traces: int,
    valid_pick_mask: np.ndarray,
    valid_observation_mask: np.ndarray,
    source_id: np.ndarray,
    receiver_id: np.ndarray,
    node_x_m: np.ndarray,
    distance: np.ndarray,
    moveout: RefractionStaticMoveoutRequest,
    distance_source_used: str,
    offset_gate_mask: np.ndarray,
    offset_mismatch_mask: np.ndarray,
    linkage: RefractionStaticLinkageRequest | None,
    linkage_used: bool,
    missing_linkage_mask: np.ndarray,
    rejection_reason: np.ndarray,
) -> dict[str, Any]:
    valid_distance = distance[valid_observation_mask & np.isfinite(distance)]
    n_valid_observations = int(np.count_nonzero(valid_observation_mask))
    n_traces_int = int(n_traces)
    rejection_counts = {
        reason: int(np.count_nonzero(rejection_reason == reason))
        for reason in _REJECTION_REASONS
        if reason != 'ok'
    }
    payload: dict[str, Any] = {
        'n_traces': n_traces_int,
        'n_valid_picks': int(np.count_nonzero(valid_pick_mask)),
        'n_valid_observations': n_valid_observations,
        'valid_observation_fraction': (
            float(n_valid_observations) / float(n_traces_int)
            if n_traces_int > 0
            else 0.0
        ),
        'n_unique_sources': int(np.unique(source_id).shape[0]),
        'n_unique_receivers': int(np.unique(receiver_id).shape[0]),
        'n_nodes': int(node_x_m.shape[0]),
        'distance_source_requested': str(moveout.distance_source),
        'distance_source_used': distance_source_used,
        'min_distance_m': _finite_stat(valid_distance, 'min'),
        'max_distance_m': _finite_stat(valid_distance, 'max'),
        'median_distance_m': _finite_stat(valid_distance, 'median'),
        'min_offset_gate_m': _json_optional_float(moveout.min_offset_m),
        'max_offset_gate_m': _json_optional_float(moveout.max_offset_m),
        'n_rejected_by_offset_gate': int(np.count_nonzero(~offset_gate_mask)),
        'n_rejected_by_offset_mismatch': int(np.count_nonzero(offset_mismatch_mask)),
        'linkage_requested': 'none' if linkage is None else str(linkage.mode),
        'linkage_used': bool(linkage_used),
        'n_rejected_by_missing_linkage': int(np.count_nonzero(missing_linkage_mask)),
        'rejection_counts': rejection_counts,
    }
    _assert_json_safe(payload)
    return payload


def _validate_pick_mask(
    picks: np.ndarray,
    *,
    n_samples: int | None,
    dt: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    reasons = np.full(picks.shape, 'ok', dtype='<U32')
    finite = np.isfinite(picks)
    reasons[np.isnan(picks)] = 'missing_pick'
    reasons[np.isinf(picks)] = 'nonfinite_pick'
    negative = finite & (picks < 0.0)
    reasons[negative] = 'negative_pick'
    if n_samples is not None and dt is not None:
        max_time = float(n_samples - 1) * float(dt)
        outside = finite & (picks > max_time + 1.0e-9)
        reasons[outside] = 'outside_trace_time_range'
    return reasons == 'ok', reasons


def _select_pick_key(keys: list[str]) -> str:
    for key in PICK_TIME_KEYS:
        if key in keys:
            return key
    accepted = ', '.join(PICK_TIME_KEYS)
    raise ValueError(f'unsupported pick artifact key; accepted keys: {accepted}')


def _validate_pick_npz_metadata(
    npz: np.lib.npyio.NpzFile,
    *,
    n_traces: int,
    n_samples: int,
    dt: float,
    path: Path,
    key: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        'npz_path': str(path),
        'npz_keys': tuple(npz.files),
        'accepted_pick_key': key,
        'accepted_key_priority': PICK_TIME_KEYS,
    }
    if 'n_traces' in npz.files:
        value = _read_int_scalar_from_npz(npz, 'n_traces')
        metadata['n_traces'] = value
        if value != n_traces:
            raise ValueError(f'n_traces mismatch: expected {n_traces}, got {value}')
    if 'n_samples' in npz.files:
        value = _read_int_scalar_from_npz(npz, 'n_samples')
        metadata['n_samples'] = value
        if value != n_samples:
            raise ValueError(f'n_samples mismatch: expected {n_samples}, got {value}')
    if 'dt' in npz.files:
        value = _read_float_scalar_from_npz(npz, 'dt')
        metadata['dt'] = value
        if not np.isfinite(value) or abs(value - dt) > 1.0e-9:
            raise ValueError(f'dt mismatch: expected {dt}, got {value}')
    return metadata


def _read_npz_order(npz: np.lib.npyio.NpzFile) -> str:
    for key in ('order', 'trace_order'):
        if key not in npz.files:
            continue
        value = _read_string_scalar_from_npz(npz, key)
        if value in {_ORDER_TRACE_STORE_SORTED, 'sorted'}:
            return _ORDER_TRACE_STORE_SORTED
        if value in {_ORDER_ORIGINAL_TRACE, 'original'}:
            return _ORDER_ORIGINAL_TRACE
        raise ValueError(f'unsupported pick artifact {key}: {value}')
    return _ORDER_ORIGINAL_TRACE


def _unique_endpoints(
    endpoint_kind: str,
    endpoint_id_values: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    elevation_m: np.ndarray,
    *,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    endpoint_id_sorted = np.full(x_m.shape, -1, dtype=np.int64)
    endpoint_key_sorted = np.full(x_m.shape, '', dtype=_ENDPOINT_KEY_DTYPE)
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return (
            endpoint_id_sorted,
            endpoint_key_sorted,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    keys = np.empty(
        valid_indices.size,
        dtype=[
            ('header_id', '<i8'),
            ('x_m', '<f8'),
            ('y_m', '<f8'),
            ('elevation_m', '<f8'),
        ],
    )
    keys['header_id'] = endpoint_id_values[valid_indices].astype(
        np.int64,
        copy=False,
    )
    keys['x_m'] = x_m[valid_indices].astype(np.float64, copy=False)
    keys['y_m'] = y_m[valid_indices].astype(np.float64, copy=False)
    keys['elevation_m'] = elevation_m[valid_indices].astype(np.float64, copy=False)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    endpoint_id_sorted[valid_indices] = inverse.astype(np.int64, copy=False)
    for trace_index in valid_indices.tolist():
        endpoint_key_sorted[trace_index] = _format_endpoint_key(
            endpoint_kind=endpoint_kind,
            header_id=int(endpoint_id_values[trace_index]),
            x_m=float(x_m[trace_index]),
            y_m=float(y_m[trace_index]),
            elevation_m=float(elevation_m[trace_index]),
        )
    return (
        np.ascontiguousarray(endpoint_id_sorted, dtype=np.int64),
        np.ascontiguousarray(endpoint_key_sorted, dtype=_ENDPOINT_KEY_DTYPE),
        np.ascontiguousarray(unique_keys['x_m'], dtype=np.float64),
        np.ascontiguousarray(unique_keys['y_m'], dtype=np.float64),
        np.ascontiguousarray(unique_keys['elevation_m'], dtype=np.float64),
    )


def _format_endpoint_key(
    *,
    endpoint_kind: str,
    header_id: int,
    x_m: float,
    y_m: float,
    elevation_m: float,
) -> str:
    return (
        f'{endpoint_kind}:'
        f'{header_id}:'
        f'{x_m:.17g}:'
        f'{y_m:.17g}:'
        f'{elevation_m:.17g}'
    )


def _scaled_header_to_meters(
    values: np.ndarray,
    *,
    scalars: np.ndarray,
    scalar_valid: np.ndarray,
    unit: str,
    name: str,
    n_traces: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = _coerce_float_header(values, name=name, n_traces=n_traces)
    finite_raw = np.isfinite(raw)
    safe_raw = raw.copy()
    safe_raw[~finite_raw] = 0.0
    try:
        scaled = apply_segy_scalar(safe_raw, scalars)
    except ValueError as exc:
        raise ValueError(f'{name} scalar application failed') from exc
    scaled[~finite_raw] = np.nan
    scaled = _normalize_unit_allow_nan(scaled, unit=unit)
    valid = np.isfinite(scaled) & scalar_valid
    return (
        np.ascontiguousarray(scaled, dtype=np.float64),
        np.ascontiguousarray(valid, dtype=bool),
    )


def _normalize_unit_allow_nan(values: np.ndarray, *, unit: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    safe = arr.copy()
    safe[~finite] = 0.0
    normalized = normalize_elevation_unit(safe, unit)
    normalized[~finite] = np.nan
    return np.asarray(normalized, dtype=np.float64)


def _coerce_pick_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError('pick_time_s_sorted must be 1D')
    if np.issubdtype(arr.dtype, np.bool_) or not _is_real_numeric_dtype(arr.dtype):
        raise ValueError('pick_time_s_sorted must have a real numeric dtype')
    return np.ascontiguousarray(arr, dtype=np.float64)


def _coerce_id_header(values: np.ndarray, *, name: str, n_traces: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape != (n_traces,):
        raise ValueError(f'{name} shape mismatch: expected {(n_traces,)}, got {arr.shape}')
    if np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{name} must contain integer-compatible values')
    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr, dtype=np.int64)
    if not _is_real_numeric_dtype(arr.dtype):
        raise ValueError(f'{name} must contain integer-compatible values')
    arr_f64 = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr_f64)):
        raise ValueError(f'{name} must contain finite values')
    if not np.all(arr_f64 == np.rint(arr_f64)):
        raise ValueError(f'{name} must contain integer-compatible values')
    return np.ascontiguousarray(arr_f64, dtype=np.int64)


def _coerce_scalar_header(
    values: np.ndarray,
    *,
    name: str,
    n_traces: int,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape != (n_traces,):
        raise ValueError(f'{name} shape mismatch: expected {(n_traces,)}, got {arr.shape}')
    if np.issubdtype(arr.dtype, np.integer):
        scalars = np.ascontiguousarray(arr, dtype=np.int64)
        return scalars, np.ones(n_traces, dtype=bool)
    if np.issubdtype(arr.dtype, np.bool_) or not _is_real_numeric_dtype(arr.dtype):
        raise ValueError(f'{name} must contain integer-compatible values')
    arr_f64 = arr.astype(np.float64, copy=False)
    valid = np.isfinite(arr_f64) & (arr_f64 == np.rint(arr_f64))
    scalars = np.zeros(n_traces, dtype=np.int64)
    if np.any(valid):
        scalars[valid] = arr_f64[valid].astype(np.int64)
    return np.ascontiguousarray(scalars, dtype=np.int64), np.ascontiguousarray(valid)


def _coerce_float_header(values: object, *, name: str, n_traces: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape != (n_traces,):
        raise ValueError(f'{name} shape mismatch: expected {(n_traces,)}, got {arr.shape}')
    if np.issubdtype(arr.dtype, np.bool_) or not _is_real_numeric_dtype(arr.dtype):
        raise ValueError(f'{name} must have a real numeric dtype')
    return np.ascontiguousarray(arr, dtype=np.float64)


def _coerce_node_id_array(
    values: object,
    *,
    name: str,
    n_traces: int,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape != (n_traces,):
        raise ValueError(f'{name} shape mismatch: expected {(n_traces,)}, got {arr.shape}')
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f'{name} must have an integer dtype')
    out = np.ascontiguousarray(arr, dtype=np.int64)
    if out.size and int(out.min()) < -1:
        raise ValueError(f'{name} values must be >= -1')
    return out


def _coerce_sorted_trace_index(
    values: np.ndarray | None,
    *,
    n_traces: int,
) -> np.ndarray:
    if values is None:
        return np.arange(n_traces, dtype=np.int64)
    return validate_sorted_to_original(
        np.asarray(values),
        expected_n_traces=n_traces,
        role='sorted_trace_index',
    )


def _header(
    headers: Mapping[object, np.ndarray],
    byte: int | None,
    role: str,
) -> np.ndarray:
    if byte is None:
        raise ValueError(f'{role} header byte is required')
    keys = (int(byte), str(int(byte)), role, f'{role}_sorted')
    for key in keys:
        if key in headers:
            return np.asarray(headers[key])
    raise ValueError(f'missing required header byte {byte} for {role}')


def _offset_header_needed(moveout: RefractionStaticMoveoutRequest) -> bool:
    return (
        moveout.distance_source == 'offset_header'
        or (moveout.distance_source == 'auto' and moveout.offset_byte is not None)
        or moveout.max_geometry_offset_mismatch_m is not None
    )


def _reader_sorted_to_original(
    reader: TraceStoreSectionReader,
    *,
    n_traces: int,
) -> np.ndarray:
    values = reader.get_sorted_to_original()
    return validate_sorted_to_original(
        np.asarray(values),
        expected_n_traces=n_traces,
        role='reader',
    )


def _read_reader_header(
    reader: TraceStoreSectionReader,
    *,
    byte: int,
    role: str,
) -> np.ndarray:
    reader_header = getattr(reader, 'ensure_header', None)
    if not callable(reader_header):
        reader_header = getattr(reader, 'get_header', None)
    if not callable(reader_header):
        raise ValueError(f'reader cannot read {role} header byte {byte}')
    try:
        return reader_header(byte)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f'failed to read {role} header byte {byte}: {exc}') from exc


def _reader_n_traces(reader: TraceStoreSectionReader) -> int:
    if hasattr(reader, 'traces'):
        shape = getattr(reader.traces, 'shape', ())
        if shape:
            return _coerce_positive_int(shape[0], name='reader n_traces')
    meta = getattr(reader, 'meta', None)
    if isinstance(meta, Mapping) and 'n_traces' in meta:
        return _coerce_positive_int(meta['n_traces'], name='reader n_traces')
    raise ValueError('TraceStore metadata unavailable: n_traces')


def _reader_n_samples(reader: TraceStoreSectionReader) -> int:
    getter = getattr(reader, 'get_n_samples', None)
    if callable(getter):
        return _coerce_positive_int(getter(), name='reader n_samples')
    if hasattr(reader, 'traces'):
        shape = getattr(reader.traces, 'shape', ())
        if len(shape) >= 2:
            return _coerce_positive_int(shape[-1], name='reader n_samples')
    raise ValueError('TraceStore metadata unavailable: n_samples')


def _reader_dt(reader: TraceStoreSectionReader, *, state: AppState, file_id: str) -> float:
    meta = getattr(reader, 'meta', None)
    if isinstance(meta, Mapping):
        raw_dt = meta.get('dt')
        if isinstance(raw_dt, (int, float)) and raw_dt > 0:
            return float(raw_dt)
    return _coerce_positive_float(state.file_registry.get_dt(file_id), name='dt')


def _read_int_scalar_from_npz(npz: np.lib.npyio.NpzFile, key: str) -> int:
    arr = np.asarray(npz[key])
    if arr.size != 1 or np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{key} must be an integer scalar')
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f'{key} must be an integer scalar')
    return int(arr.reshape(-1)[0])


def _read_float_scalar_from_npz(npz: np.lib.npyio.NpzFile, key: str) -> float:
    arr = np.asarray(npz[key])
    if arr.size != 1:
        raise ValueError(f'{key} must be a numeric scalar')
    if np.issubdtype(arr.dtype, np.bool_) or not _is_real_numeric_dtype(arr.dtype):
        raise ValueError(f'{key} must be a numeric scalar')
    return float(arr.reshape(-1)[0])


def _read_string_scalar_from_npz(npz: np.lib.npyio.NpzFile, key: str) -> str:
    arr = np.asarray(npz[key])
    if arr.size != 1:
        raise ValueError(f'{key} must be a string scalar')
    value = arr.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def _preview_rows(input_model: RefractionStaticInputModel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(input_model.n_traces):
        rows.append(
            {
                'sorted_trace_index': int(input_model.sorted_trace_index[i]),
                'pick_time_s': _csv_float(input_model.pick_time_s_sorted[i]),
                'valid_observation': bool(input_model.valid_observation_mask_sorted[i]),
                'rejection_reason': str(input_model.rejection_reason_sorted[i]),
                'source_id': int(input_model.source_id_sorted[i]),
                'receiver_id': int(input_model.receiver_id_sorted[i]),
                'source_x_m': _csv_float(input_model.source_x_m_sorted[i]),
                'source_y_m': _csv_float(input_model.source_y_m_sorted[i]),
                'receiver_x_m': _csv_float(input_model.receiver_x_m_sorted[i]),
                'receiver_y_m': _csv_float(input_model.receiver_y_m_sorted[i]),
                'source_elevation_m': _csv_float(
                    input_model.source_elevation_m_sorted[i]
                ),
                'receiver_elevation_m': _csv_float(
                    input_model.receiver_elevation_m_sorted[i]
                ),
                'geometry_distance_m': _csv_float(
                    input_model.geometry_distance_m_sorted[i]
                ),
                'offset_m': _csv_float(
                    None
                    if input_model.offset_m_sorted is None
                    else input_model.offset_m_sorted[i]
                ),
                'distance_m': _csv_float(input_model.distance_m_sorted[i]),
                'source_node_id': int(input_model.source_node_id_sorted[i]),
                'receiver_node_id': int(input_model.receiver_node_id_sorted[i]),
            }
        )
    return rows


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        tmp_path.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
            ),
            encoding='utf-8',
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_PREVIEW_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _assert_json_safe(payload: dict[str, Any]) -> None:
    json.dumps(payload, allow_nan=False)


def _finite_stat(values: np.ndarray, stat: str) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    if stat == 'min':
        return float(np.min(finite))
    if stat == 'max':
        return float(np.max(finite))
    if stat == 'median':
        return float(np.median(finite))
    raise ValueError(f'unsupported stat: {stat}')


def _json_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out):
        raise ValueError('optional float must be finite')
    return out


def _csv_float(value: object) -> str:
    if value is None:
        return ''
    try:
        return f'{float(value):.12g}'
    except (TypeError, ValueError):
        return str(value)


def _optional_positive_int(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    return _coerce_positive_int(value, name=name)


def _optional_positive_float(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    return _coerce_positive_float(value, name=name)


def _validate_header_byte(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer SEG-Y trace header byte')
    out = int(value)
    if out < 1 or out > 240:
        raise ValueError(f'{name} must be between 1 and 240')
    return out


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


def _coerce_positive_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a positive finite float')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a positive finite float') from exc
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f'{name} must be a positive finite float')
    return out


def _non_empty_str(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{name} must be a non-empty string')
    return value


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype,
        np.complexfloating,
    )


__all__ = [
    'PICK_TIME_KEYS',
    'REFRACTION_INPUT_PREVIEW_CSV_NAME',
    'REFRACTION_INPUT_QC_JSON_NAME',
    'RefractionEndpointTable',
    'RefractionStaticInputModel',
    'build_refraction_static_input_model',
    'build_refraction_static_input_model_from_arrays',
    'write_refraction_static_input_artifacts',
]
