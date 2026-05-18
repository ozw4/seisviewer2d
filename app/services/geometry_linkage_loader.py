"""Strict loader for static geometry linkage NPZ artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from app.services.geometry_linkage_artifacts import (
    GEOMETRY_LINKAGE_NPZ_NAME,
    ORDER,
    SCHEMA_VERSION,
    SOLUTION_ARTIFACT_KIND,
)

LinkageMode = Literal['none', 'auto_threshold']

_KNOWN_MODES = {'none', 'auto_threshold'}
_KNOWN_RECORD_LINKED_TO_KINDS = {'', 'source', 'receiver'}
_KNOWN_RECORD_METHODS = {
    'none_mode_receiver_independent',
    'none_mode_source_independent',
    'receiver_seed',
    'receiver_anchor',
    'source_fallback',
    'source_independent',
}
_DISTANCE_RTOL = 1.0e-9
_DISTANCE_ATOL = 1.0e-9

_SCALAR_FIELDS = (
    'schema_version',
    'artifact_kind',
    'order',
    'mode',
    'threshold_m',
    'receiver_location_interval_m',
    'prefer_receiver_anchor',
    'n_traces',
    'n_source_endpoints',
    'n_receiver_endpoints',
    'n_nodes',
    'n_receiver_anchor_links',
    'n_source_fallback_links',
    'n_independent_source_nodes',
    'coordinate_scalar_zero_count',
    'job_id',
    'input_file_id',
    'key1_byte',
    'key2_byte',
    'source_x_byte',
    'source_y_byte',
    'receiver_x_byte',
    'receiver_y_byte',
    'coordinate_scalar_byte',
    'header_source_segy_path',
)
_TRACE_LEVEL_FIELDS = (
    'source_x_m_sorted',
    'source_y_m_sorted',
    'receiver_x_m_sorted',
    'receiver_y_m_sorted',
    'coordinate_scalar_sorted',
    'source_endpoint_id_sorted',
    'receiver_endpoint_id_sorted',
    'source_node_id_sorted',
    'receiver_node_id_sorted',
)
_SOURCE_ENDPOINT_FIELDS = (
    'source_endpoint_id',
    'source_endpoint_x_m',
    'source_endpoint_y_m',
    'source_endpoint_first_sorted_trace_index',
    'source_endpoint_trace_count',
    'source_node_id_by_endpoint',
)
_RECEIVER_ENDPOINT_FIELDS = (
    'receiver_endpoint_id',
    'receiver_endpoint_x_m',
    'receiver_endpoint_y_m',
    'receiver_endpoint_first_sorted_trace_index',
    'receiver_endpoint_trace_count',
    'receiver_node_id_by_endpoint',
)
_RECORD_FIELDS = (
    'record_endpoint_kind',
    'record_endpoint_id',
    'record_x_m',
    'record_y_m',
    'record_node_id',
    'record_linked_to_kind',
    'record_linked_to_id',
    'record_distance_m',
    'record_method',
)
_REQUIRED_FIELDS = (
    *_SCALAR_FIELDS,
    *_TRACE_LEVEL_FIELDS,
    *_SOURCE_ENDPOINT_FIELDS,
    *_RECEIVER_ENDPOINT_FIELDS,
    *_RECORD_FIELDS,
)


@dataclass(frozen=True)
class GeometryLinkageLoadedMetadata:
    job_id: str | None
    input_file_id: str | None
    key1_byte: int | None
    key2_byte: int | None
    source_x_byte: int | None
    source_y_byte: int | None
    receiver_x_byte: int | None
    receiver_y_byte: int | None
    coordinate_scalar_byte: int | None
    header_source_segy_path: str | None


@dataclass(frozen=True)
class GeometryLinkageTraceNodeMapping:
    n_traces: int
    n_nodes: int
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray


@dataclass(frozen=True)
class LoadedGeometryLinkageArtifact:
    path: Path
    schema_version: int
    artifact_kind: str
    order: str
    mode: LinkageMode
    threshold_m: float | None
    receiver_location_interval_m: float | None
    prefer_receiver_anchor: bool
    n_traces: int
    n_source_endpoints: int
    n_receiver_endpoints: int
    n_nodes: int
    n_receiver_anchor_links: int
    n_source_fallback_links: int
    n_independent_source_nodes: int
    coordinate_scalar_zero_count: int
    metadata: GeometryLinkageLoadedMetadata

    source_x_m_sorted: np.ndarray
    source_y_m_sorted: np.ndarray
    receiver_x_m_sorted: np.ndarray
    receiver_y_m_sorted: np.ndarray
    coordinate_scalar_sorted: np.ndarray
    source_endpoint_id_sorted: np.ndarray
    receiver_endpoint_id_sorted: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray

    source_endpoint_id: np.ndarray
    source_endpoint_x_m: np.ndarray
    source_endpoint_y_m: np.ndarray
    source_endpoint_first_sorted_trace_index: np.ndarray
    source_endpoint_trace_count: np.ndarray
    source_node_id_by_endpoint: np.ndarray

    receiver_endpoint_id: np.ndarray
    receiver_endpoint_x_m: np.ndarray
    receiver_endpoint_y_m: np.ndarray
    receiver_endpoint_first_sorted_trace_index: np.ndarray
    receiver_endpoint_trace_count: np.ndarray
    receiver_node_id_by_endpoint: np.ndarray

    record_endpoint_kind: np.ndarray
    record_endpoint_id: np.ndarray
    record_x_m: np.ndarray
    record_y_m: np.ndarray
    record_node_id: np.ndarray
    record_linked_to_kind: np.ndarray
    record_linked_to_id: np.ndarray
    record_distance_m: np.ndarray
    record_method: np.ndarray

    def trace_node_mapping(self) -> GeometryLinkageTraceNodeMapping:
        return GeometryLinkageTraceNodeMapping(
            n_traces=self.n_traces,
            n_nodes=self.n_nodes,
            source_node_id_sorted=self.source_node_id_sorted,
            receiver_node_id_sorted=self.receiver_node_id_sorted,
        )


def load_geometry_linkage_artifact(
    npz_path: Path,
    *,
    expected_n_traces: int | None = None,
    expected_key1_byte: int | None = None,
    expected_key2_byte: int | None = None,
) -> LoadedGeometryLinkageArtifact:
    """Load and validate a geometry linkage solution artifact."""
    path = Path(npz_path)
    arrays = _read_required_arrays(path)

    schema_version = _read_integer_scalar(
        arrays['schema_version'],
        name='schema_version',
    )
    artifact_kind = _read_string_scalar(
        arrays['artifact_kind'],
        name='artifact_kind',
    )
    order = _read_string_scalar(arrays['order'], name='order')
    mode_value = _read_string_scalar(arrays['mode'], name='mode')
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f'schema_version must be {SCHEMA_VERSION}')
    if artifact_kind != SOLUTION_ARTIFACT_KIND:
        raise ValueError(
            f'geometry_linkage artifact kind must be {SOLUTION_ARTIFACT_KIND}'
        )
    if order != ORDER:
        raise ValueError(f'geometry_linkage artifact order must be {ORDER}')
    if mode_value not in _KNOWN_MODES:
        raise ValueError(f'mode contains unsupported value: {mode_value!r}')
    mode: LinkageMode = 'none' if mode_value == 'none' else 'auto_threshold'

    threshold_m = _read_optional_float_scalar(
        arrays['threshold_m'],
        name='threshold_m',
    )
    receiver_location_interval_m = _read_optional_float_scalar(
        arrays['receiver_location_interval_m'],
        name='receiver_location_interval_m',
    )
    prefer_receiver_anchor = _read_bool_scalar(
        arrays['prefer_receiver_anchor'],
        name='prefer_receiver_anchor',
    )
    n_traces = _read_positive_integer_scalar(arrays['n_traces'], name='n_traces')
    n_source_endpoints = _read_positive_integer_scalar(
        arrays['n_source_endpoints'],
        name='n_source_endpoints',
    )
    n_receiver_endpoints = _read_positive_integer_scalar(
        arrays['n_receiver_endpoints'],
        name='n_receiver_endpoints',
    )
    n_nodes = _read_positive_integer_scalar(arrays['n_nodes'], name='n_nodes')
    n_receiver_anchor_links = _read_nonnegative_integer_scalar(
        arrays['n_receiver_anchor_links'],
        name='n_receiver_anchor_links',
    )
    n_source_fallback_links = _read_nonnegative_integer_scalar(
        arrays['n_source_fallback_links'],
        name='n_source_fallback_links',
    )
    n_independent_source_nodes = _read_nonnegative_integer_scalar(
        arrays['n_independent_source_nodes'],
        name='n_independent_source_nodes',
    )
    coordinate_scalar_zero_count = _read_nonnegative_integer_scalar(
        arrays['coordinate_scalar_zero_count'],
        name='coordinate_scalar_zero_count',
    )
    _validate_expected_n_traces(n_traces, expected_n_traces)

    metadata = GeometryLinkageLoadedMetadata(
        job_id=_read_optional_string_scalar(arrays['job_id'], name='job_id'),
        input_file_id=_read_optional_string_scalar(
            arrays['input_file_id'],
            name='input_file_id',
        ),
        key1_byte=_read_optional_integer_scalar(
            arrays['key1_byte'],
            name='key1_byte',
        ),
        key2_byte=_read_optional_integer_scalar(
            arrays['key2_byte'],
            name='key2_byte',
        ),
        source_x_byte=_read_optional_integer_scalar(
            arrays['source_x_byte'],
            name='source_x_byte',
        ),
        source_y_byte=_read_optional_integer_scalar(
            arrays['source_y_byte'],
            name='source_y_byte',
        ),
        receiver_x_byte=_read_optional_integer_scalar(
            arrays['receiver_x_byte'],
            name='receiver_x_byte',
        ),
        receiver_y_byte=_read_optional_integer_scalar(
            arrays['receiver_y_byte'],
            name='receiver_y_byte',
        ),
        coordinate_scalar_byte=_read_optional_integer_scalar(
            arrays['coordinate_scalar_byte'],
            name='coordinate_scalar_byte',
        ),
        header_source_segy_path=_read_optional_string_scalar(
            arrays['header_source_segy_path'],
            name='header_source_segy_path',
        ),
    )
    _validate_expected_key_byte(
        metadata.key1_byte,
        expected_key1_byte,
        name='key1_byte',
        expected_name='expected_key1_byte',
    )
    _validate_expected_key_byte(
        metadata.key2_byte,
        expected_key2_byte,
        name='key2_byte',
        expected_name='expected_key2_byte',
    )

    if receiver_location_interval_m is not None and receiver_location_interval_m <= 0.0:
        raise ValueError('receiver_location_interval_m must be finite and greater than 0')

    source_x_m_sorted = _read_finite_float_1d(
        arrays['source_x_m_sorted'],
        name='source_x_m_sorted',
        expected_shape=(n_traces,),
    )
    source_y_m_sorted = _read_finite_float_1d(
        arrays['source_y_m_sorted'],
        name='source_y_m_sorted',
        expected_shape=(n_traces,),
    )
    receiver_x_m_sorted = _read_finite_float_1d(
        arrays['receiver_x_m_sorted'],
        name='receiver_x_m_sorted',
        expected_shape=(n_traces,),
    )
    receiver_y_m_sorted = _read_finite_float_1d(
        arrays['receiver_y_m_sorted'],
        name='receiver_y_m_sorted',
        expected_shape=(n_traces,),
    )
    coordinate_scalar_sorted = _read_integer_1d(
        arrays['coordinate_scalar_sorted'],
        name='coordinate_scalar_sorted',
        expected_shape=(n_traces,),
    )
    source_endpoint_id_sorted = _read_integer_1d(
        arrays['source_endpoint_id_sorted'],
        name='source_endpoint_id_sorted',
        expected_shape=(n_traces,),
    )
    receiver_endpoint_id_sorted = _read_integer_1d(
        arrays['receiver_endpoint_id_sorted'],
        name='receiver_endpoint_id_sorted',
        expected_shape=(n_traces,),
    )
    source_node_id_sorted = _read_integer_1d(
        arrays['source_node_id_sorted'],
        name='source_node_id_sorted',
        expected_shape=(n_traces,),
    )
    receiver_node_id_sorted = _read_integer_1d(
        arrays['receiver_node_id_sorted'],
        name='receiver_node_id_sorted',
        expected_shape=(n_traces,),
    )

    source_endpoint_id = _read_integer_1d(
        arrays['source_endpoint_id'],
        name='source_endpoint_id',
        expected_shape=(n_source_endpoints,),
    )
    source_endpoint_x_m = _read_finite_float_1d(
        arrays['source_endpoint_x_m'],
        name='source_endpoint_x_m',
        expected_shape=(n_source_endpoints,),
    )
    source_endpoint_y_m = _read_finite_float_1d(
        arrays['source_endpoint_y_m'],
        name='source_endpoint_y_m',
        expected_shape=(n_source_endpoints,),
    )
    source_endpoint_first_sorted_trace_index = _read_integer_1d(
        arrays['source_endpoint_first_sorted_trace_index'],
        name='source_endpoint_first_sorted_trace_index',
        expected_shape=(n_source_endpoints,),
    )
    source_endpoint_trace_count = _read_integer_1d(
        arrays['source_endpoint_trace_count'],
        name='source_endpoint_trace_count',
        expected_shape=(n_source_endpoints,),
    )
    source_node_id_by_endpoint = _read_integer_1d(
        arrays['source_node_id_by_endpoint'],
        name='source_node_id_by_endpoint',
        expected_shape=(n_source_endpoints,),
    )

    receiver_endpoint_id = _read_integer_1d(
        arrays['receiver_endpoint_id'],
        name='receiver_endpoint_id',
        expected_shape=(n_receiver_endpoints,),
    )
    receiver_endpoint_x_m = _read_finite_float_1d(
        arrays['receiver_endpoint_x_m'],
        name='receiver_endpoint_x_m',
        expected_shape=(n_receiver_endpoints,),
    )
    receiver_endpoint_y_m = _read_finite_float_1d(
        arrays['receiver_endpoint_y_m'],
        name='receiver_endpoint_y_m',
        expected_shape=(n_receiver_endpoints,),
    )
    receiver_endpoint_first_sorted_trace_index = _read_integer_1d(
        arrays['receiver_endpoint_first_sorted_trace_index'],
        name='receiver_endpoint_first_sorted_trace_index',
        expected_shape=(n_receiver_endpoints,),
    )
    receiver_endpoint_trace_count = _read_integer_1d(
        arrays['receiver_endpoint_trace_count'],
        name='receiver_endpoint_trace_count',
        expected_shape=(n_receiver_endpoints,),
    )
    receiver_node_id_by_endpoint = _read_integer_1d(
        arrays['receiver_node_id_by_endpoint'],
        name='receiver_node_id_by_endpoint',
        expected_shape=(n_receiver_endpoints,),
    )

    n_records = n_receiver_endpoints + n_source_endpoints
    record_endpoint_kind = _read_string_1d(
        arrays['record_endpoint_kind'],
        name='record_endpoint_kind',
        expected_shape=(n_records,),
    )
    record_endpoint_id = _read_integer_1d(
        arrays['record_endpoint_id'],
        name='record_endpoint_id',
        expected_shape=(n_records,),
    )
    record_x_m = _read_finite_float_1d(
        arrays['record_x_m'],
        name='record_x_m',
        expected_shape=(n_records,),
    )
    record_y_m = _read_finite_float_1d(
        arrays['record_y_m'],
        name='record_y_m',
        expected_shape=(n_records,),
    )
    record_node_id = _read_integer_1d(
        arrays['record_node_id'],
        name='record_node_id',
        expected_shape=(n_records,),
    )
    record_linked_to_kind = _read_string_1d(
        arrays['record_linked_to_kind'],
        name='record_linked_to_kind',
        expected_shape=(n_records,),
    )
    record_linked_to_id = _read_integer_1d(
        arrays['record_linked_to_id'],
        name='record_linked_to_id',
        expected_shape=(n_records,),
    )
    record_distance_m = _read_record_distance_1d(
        arrays['record_distance_m'],
        name='record_distance_m',
        expected_shape=(n_records,),
    )
    record_method = _read_string_1d(
        arrays['record_method'],
        name='record_method',
        expected_shape=(n_records,),
    )

    _validate_mode_scalars(
        mode=mode,
        threshold_m=threshold_m,
        prefer_receiver_anchor=prefer_receiver_anchor,
        n_receiver_anchor_links=n_receiver_anchor_links,
        n_source_fallback_links=n_source_fallback_links,
        n_independent_source_nodes=n_independent_source_nodes,
        n_source_endpoints=n_source_endpoints,
        n_receiver_endpoints=n_receiver_endpoints,
        n_nodes=n_nodes,
    )
    _validate_endpoint_ids(
        source_endpoint_id=source_endpoint_id,
        receiver_endpoint_id=receiver_endpoint_id,
    )
    _validate_sorted_endpoint_ids(
        source_endpoint_id_sorted,
        name='source_endpoint_id_sorted',
        n_endpoints=n_source_endpoints,
    )
    _validate_sorted_endpoint_ids(
        receiver_endpoint_id_sorted,
        name='receiver_endpoint_id_sorted',
        n_endpoints=n_receiver_endpoints,
    )
    _validate_endpoint_trace_metadata(
        endpoint_id_sorted=source_endpoint_id_sorted,
        first_sorted_trace_index=source_endpoint_first_sorted_trace_index,
        trace_count=source_endpoint_trace_count,
        n_endpoints=n_source_endpoints,
        kind='source',
    )
    _validate_endpoint_trace_metadata(
        endpoint_id_sorted=receiver_endpoint_id_sorted,
        first_sorted_trace_index=receiver_endpoint_first_sorted_trace_index,
        trace_count=receiver_endpoint_trace_count,
        n_endpoints=n_receiver_endpoints,
        kind='receiver',
    )
    _validate_node_ids(
        source_node_id_by_endpoint=source_node_id_by_endpoint,
        receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
        n_nodes=n_nodes,
    )
    _validate_trace_node_mapping(
        source_endpoint_id_sorted=source_endpoint_id_sorted,
        receiver_endpoint_id_sorted=receiver_endpoint_id_sorted,
        source_node_id_by_endpoint=source_node_id_by_endpoint,
        receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
        source_node_id_sorted=source_node_id_sorted,
        receiver_node_id_sorted=receiver_node_id_sorted,
    )
    _validate_trace_coordinate_mapping(
        source_endpoint_id_sorted=source_endpoint_id_sorted,
        receiver_endpoint_id_sorted=receiver_endpoint_id_sorted,
        source_x_m_sorted=source_x_m_sorted,
        source_y_m_sorted=source_y_m_sorted,
        receiver_x_m_sorted=receiver_x_m_sorted,
        receiver_y_m_sorted=receiver_y_m_sorted,
        source_endpoint_x_m=source_endpoint_x_m,
        source_endpoint_y_m=source_endpoint_y_m,
        receiver_endpoint_x_m=receiver_endpoint_x_m,
        receiver_endpoint_y_m=receiver_endpoint_y_m,
    )
    _validate_records(
        mode=mode,
        n_source_endpoints=n_source_endpoints,
        n_receiver_endpoints=n_receiver_endpoints,
        n_receiver_anchor_links=n_receiver_anchor_links,
        n_source_fallback_links=n_source_fallback_links,
        n_independent_source_nodes=n_independent_source_nodes,
        source_endpoint_x_m=source_endpoint_x_m,
        source_endpoint_y_m=source_endpoint_y_m,
        source_node_id_by_endpoint=source_node_id_by_endpoint,
        receiver_endpoint_x_m=receiver_endpoint_x_m,
        receiver_endpoint_y_m=receiver_endpoint_y_m,
        receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
        record_endpoint_kind=record_endpoint_kind,
        record_endpoint_id=record_endpoint_id,
        record_x_m=record_x_m,
        record_y_m=record_y_m,
        record_node_id=record_node_id,
        record_linked_to_kind=record_linked_to_kind,
        record_linked_to_id=record_linked_to_id,
        record_distance_m=record_distance_m,
        record_method=record_method,
    )

    return LoadedGeometryLinkageArtifact(
        path=path,
        schema_version=schema_version,
        artifact_kind=artifact_kind,
        order=order,
        mode=mode,
        threshold_m=threshold_m,
        receiver_location_interval_m=receiver_location_interval_m,
        prefer_receiver_anchor=prefer_receiver_anchor,
        n_traces=n_traces,
        n_source_endpoints=n_source_endpoints,
        n_receiver_endpoints=n_receiver_endpoints,
        n_nodes=n_nodes,
        n_receiver_anchor_links=n_receiver_anchor_links,
        n_source_fallback_links=n_source_fallback_links,
        n_independent_source_nodes=n_independent_source_nodes,
        coordinate_scalar_zero_count=coordinate_scalar_zero_count,
        metadata=metadata,
        source_x_m_sorted=source_x_m_sorted,
        source_y_m_sorted=source_y_m_sorted,
        receiver_x_m_sorted=receiver_x_m_sorted,
        receiver_y_m_sorted=receiver_y_m_sorted,
        coordinate_scalar_sorted=coordinate_scalar_sorted,
        source_endpoint_id_sorted=source_endpoint_id_sorted,
        receiver_endpoint_id_sorted=receiver_endpoint_id_sorted,
        source_node_id_sorted=source_node_id_sorted,
        receiver_node_id_sorted=receiver_node_id_sorted,
        source_endpoint_id=source_endpoint_id,
        source_endpoint_x_m=source_endpoint_x_m,
        source_endpoint_y_m=source_endpoint_y_m,
        source_endpoint_first_sorted_trace_index=source_endpoint_first_sorted_trace_index,
        source_endpoint_trace_count=source_endpoint_trace_count,
        source_node_id_by_endpoint=source_node_id_by_endpoint,
        receiver_endpoint_id=receiver_endpoint_id,
        receiver_endpoint_x_m=receiver_endpoint_x_m,
        receiver_endpoint_y_m=receiver_endpoint_y_m,
        receiver_endpoint_first_sorted_trace_index=receiver_endpoint_first_sorted_trace_index,
        receiver_endpoint_trace_count=receiver_endpoint_trace_count,
        receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
        record_endpoint_kind=record_endpoint_kind,
        record_endpoint_id=record_endpoint_id,
        record_x_m=record_x_m,
        record_y_m=record_y_m,
        record_node_id=record_node_id,
        record_linked_to_kind=record_linked_to_kind,
        record_linked_to_id=record_linked_to_id,
        record_distance_m=record_distance_m,
        record_method=record_method,
    )


def load_geometry_linkage_from_job_dir(
    job_dir: Path,
    *,
    expected_n_traces: int | None = None,
    expected_key1_byte: int | None = None,
    expected_key2_byte: int | None = None,
) -> LoadedGeometryLinkageArtifact:
    """Load the standard geometry linkage NPZ artifact from a job directory."""
    return load_geometry_linkage_artifact(
        Path(job_dir) / GEOMETRY_LINKAGE_NPZ_NAME,
        expected_n_traces=expected_n_traces,
        expected_key1_byte=expected_key1_byte,
        expected_key2_byte=expected_key2_byte,
    )


def load_geometry_linkage_trace_node_mapping(
    npz_path: Path,
    *,
    expected_n_traces: int | None = None,
) -> GeometryLinkageTraceNodeMapping:
    """Load only the sorted trace node mapping after full artifact validation."""
    loaded = load_geometry_linkage_artifact(
        npz_path,
        expected_n_traces=expected_n_traces,
    )
    return loaded.trace_node_mapping()


def _read_required_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as npz:
        available_fields = set(npz.files)
        for field in _REQUIRED_FIELDS:
            if field not in available_fields:
                raise ValueError(f'missing required field: {field}')
        arrays: dict[str, np.ndarray] = {}
        for field in npz.files:
            arrays[field] = _copy_npz_array(npz, field)
        return {field: arrays[field] for field in _REQUIRED_FIELDS}


def _copy_npz_array(npz: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    try:
        arr = np.array(npz[name], copy=True)
    except ValueError as exc:
        if 'Object arrays cannot be loaded' in str(exc):
            raise ValueError(f'{name} has object dtype') from exc
        raise
    _reject_object_dtype(arr, name=name)
    return arr


def _reject_object_dtype(arr: np.ndarray, *, name: str) -> None:
    if arr.dtype == object:
        raise ValueError(f'{name} has object dtype')


def _require_scalar(arr: np.ndarray, *, name: str) -> None:
    _reject_object_dtype(arr, name=name)
    if arr.ndim != 0:
        raise ValueError(f'{name} must be a scalar')


def _read_integer_scalar(arr: np.ndarray, *, name: str) -> int:
    _require_scalar(arr, name=name)
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(
        arr.dtype,
        np.integer,
    ):
        raise ValueError(f'{name} must be an integer scalar')
    return int(arr.item())


def _read_positive_integer_scalar(arr: np.ndarray, *, name: str) -> int:
    value = _read_integer_scalar(arr, name=name)
    if value <= 0:
        raise ValueError(f'{name} must be greater than 0')
    return value


def _read_nonnegative_integer_scalar(arr: np.ndarray, *, name: str) -> int:
    value = _read_integer_scalar(arr, name=name)
    if value < 0:
        raise ValueError(f'{name} must be nonnegative')
    return value


def _read_optional_integer_scalar(arr: np.ndarray, *, name: str) -> int | None:
    _require_scalar(arr, name=name)
    if np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{name} must be an integer scalar or NaN')
    if np.issubdtype(arr.dtype, np.integer):
        return int(arr.item())
    if not _is_real_numeric_dtype(arr.dtype):
        raise ValueError(f'{name} must be an integer scalar or NaN')
    value = float(arr.item())
    if np.isnan(value):
        return None
    if not np.isfinite(value) or value != round(value):
        raise ValueError(f'{name} must be an integer scalar or NaN')
    return int(value)


def _read_optional_float_scalar(arr: np.ndarray, *, name: str) -> float | None:
    _require_scalar(arr, name=name)
    if np.issubdtype(arr.dtype, np.bool_) or not _is_real_numeric_dtype(arr.dtype):
        raise ValueError(f'{name} must be a numeric scalar')
    value = float(arr.item())
    if np.isnan(value):
        return None
    if not np.isfinite(value):
        raise ValueError(f'{name} must be finite or NaN')
    return value


def _read_bool_scalar(arr: np.ndarray, *, name: str) -> bool:
    _require_scalar(arr, name=name)
    if np.issubdtype(arr.dtype, np.bool_):
        return bool(arr.item())
    if np.issubdtype(arr.dtype, np.integer):
        value = int(arr.item())
        if value in {0, 1}:
            return bool(value)
    raise ValueError(f'{name} must be a bool scalar')


def _read_string_scalar(arr: np.ndarray, *, name: str) -> str:
    _require_scalar(arr, name=name)
    if not _is_string_dtype(arr.dtype):
        raise ValueError(f'{name} must be a string scalar')
    return _scalar_string_value(arr.item())


def _read_optional_string_scalar(arr: np.ndarray, *, name: str) -> str | None:
    value = _read_string_scalar(arr, name=name)
    return None if value == '' else value


def _scalar_string_value(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def _read_integer_1d(
    arr: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    _validate_1d_shape(arr, name=name, expected_shape=expected_shape)
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(
        arr.dtype,
        np.integer,
    ):
        raise ValueError(f'{name} must have an integer dtype')
    return _readonly(np.array(arr, dtype=np.int64, copy=True, order='C'))


def _read_finite_float_1d(
    arr: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    _validate_1d_shape(arr, name=name, expected_shape=expected_shape)
    if np.issubdtype(arr.dtype, np.bool_) or not _is_real_numeric_dtype(arr.dtype):
        raise ValueError(f'{name} must have a real numeric dtype')
    out = np.array(arr, dtype=np.float64, copy=True, order='C')
    if not np.all(np.isfinite(out)):
        raise ValueError(f'{name} must contain only finite values')
    return _readonly(out)


def _read_record_distance_1d(
    arr: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    _validate_1d_shape(arr, name=name, expected_shape=expected_shape)
    if np.issubdtype(arr.dtype, np.bool_) or not _is_real_numeric_dtype(arr.dtype):
        raise ValueError(f'{name} must have a real numeric dtype')
    out = np.array(arr, dtype=np.float64, copy=True, order='C')
    if np.any(np.isinf(out)):
        raise ValueError(f'{name} must not contain Inf')
    finite = out[np.isfinite(out)]
    if np.any(finite < 0.0):
        raise ValueError(f'{name} finite values must be greater than or equal to 0')
    return _readonly(out)


def _read_string_1d(
    arr: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    _validate_1d_shape(arr, name=name, expected_shape=expected_shape)
    if not _is_string_dtype(arr.dtype):
        raise ValueError(f'{name} must have a string dtype')
    return _readonly(np.array(arr, dtype=np.str_, copy=True, order='C'))


def _validate_1d_shape(
    arr: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> None:
    _reject_object_dtype(arr, name=name)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if arr.shape != expected_shape:
        raise ValueError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )


def _readonly(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(
        dtype,
        np.complexfloating,
    )


def _is_string_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.bytes_)


def _validate_expected_n_traces(
    n_traces: int,
    expected_n_traces: int | None,
) -> None:
    if expected_n_traces is None:
        return
    if isinstance(expected_n_traces, bool) or not isinstance(
        expected_n_traces,
        int,
    ):
        raise ValueError('expected_n_traces must be an integer')
    if n_traces != expected_n_traces:
        raise ValueError(
            f'n_traces mismatch: expected {expected_n_traces}, got {n_traces}'
        )


def _validate_expected_key_byte(
    actual: int | None,
    expected: int | None,
    *,
    name: str,
    expected_name: str,
) -> None:
    if expected is None:
        return
    if isinstance(expected, bool) or not isinstance(expected, int):
        raise ValueError(f'{expected_name} must be an integer')
    if actual != expected:
        raise ValueError(f'{name} mismatch: expected {expected}, got {actual}')


def _validate_mode_scalars(
    *,
    mode: LinkageMode,
    threshold_m: float | None,
    prefer_receiver_anchor: bool,
    n_receiver_anchor_links: int,
    n_source_fallback_links: int,
    n_independent_source_nodes: int,
    n_source_endpoints: int,
    n_receiver_endpoints: int,
    n_nodes: int,
) -> None:
    if mode == 'none':
        if threshold_m is not None:
            raise ValueError('threshold_m must be None when mode is none')
        if n_receiver_anchor_links != 0:
            raise ValueError('n_receiver_anchor_links must be 0 when mode is none')
        if n_source_fallback_links != 0:
            raise ValueError('n_source_fallback_links must be 0 when mode is none')
        if n_independent_source_nodes != n_source_endpoints:
            raise ValueError(
                'n_independent_source_nodes must equal n_source_endpoints when mode is none'
            )
        if n_nodes != n_receiver_endpoints + n_source_endpoints:
            raise ValueError(
                'n_nodes must equal n_receiver_endpoints + n_source_endpoints when mode is none'
            )
        return

    if threshold_m is None or threshold_m <= 0.0:
        raise ValueError('threshold_m must be finite and greater than 0')
    if not prefer_receiver_anchor:
        raise ValueError('prefer_receiver_anchor must be True when mode is auto_threshold')
    if n_receiver_anchor_links > n_source_endpoints:
        raise ValueError('n_receiver_anchor_links must be <= n_source_endpoints')
    if n_source_fallback_links > n_source_endpoints:
        raise ValueError('n_source_fallback_links must be <= n_source_endpoints')
    if n_independent_source_nodes > n_source_endpoints:
        raise ValueError('n_independent_source_nodes must be <= n_source_endpoints')


def _validate_endpoint_ids(
    *,
    source_endpoint_id: np.ndarray,
    receiver_endpoint_id: np.ndarray,
) -> None:
    expected_source = np.arange(source_endpoint_id.shape[0], dtype=np.int64)
    if not np.array_equal(source_endpoint_id, expected_source):
        raise ValueError('source_endpoint_id must be 0-based contiguous')
    expected_receiver = np.arange(receiver_endpoint_id.shape[0], dtype=np.int64)
    if not np.array_equal(receiver_endpoint_id, expected_receiver):
        raise ValueError('receiver_endpoint_id must be 0-based contiguous')


def _validate_sorted_endpoint_ids(
    values: np.ndarray,
    *,
    name: str,
    n_endpoints: int,
) -> None:
    if values.size and (int(values.min()) < 0 or int(values.max()) >= n_endpoints):
        raise ValueError(f'{name} values must be in range 0..{n_endpoints - 1}')


def _validate_endpoint_trace_metadata(
    *,
    endpoint_id_sorted: np.ndarray,
    first_sorted_trace_index: np.ndarray,
    trace_count: np.ndarray,
    n_endpoints: int,
    kind: str,
) -> None:
    if np.any(trace_count <= 0):
        raise ValueError(f'{kind}_endpoint_trace_count values must be greater than 0')
    expected_trace_count = np.bincount(
        endpoint_id_sorted,
        minlength=n_endpoints,
    ).astype(np.int64)
    if not np.array_equal(trace_count, expected_trace_count):
        raise ValueError(
            f'{kind}_endpoint_trace_count does not match {kind}_endpoint_id_sorted'
        )

    expected_first = np.full(n_endpoints, -1, dtype=np.int64)
    for trace_index, endpoint_id in enumerate(endpoint_id_sorted):
        endpoint_index = int(endpoint_id)
        if expected_first[endpoint_index] < 0:
            expected_first[endpoint_index] = trace_index
    if not np.array_equal(first_sorted_trace_index, expected_first):
        raise ValueError(
            f'{kind}_endpoint_first_sorted_trace_index does not match {kind}_endpoint_id_sorted'
        )


def _validate_node_ids(
    *,
    source_node_id_by_endpoint: np.ndarray,
    receiver_node_id_by_endpoint: np.ndarray,
    n_nodes: int,
) -> None:
    all_endpoint_nodes = np.concatenate(
        (source_node_id_by_endpoint, receiver_node_id_by_endpoint)
    )
    if all_endpoint_nodes.size == 0:
        raise ValueError('node ids must be non-empty')
    if int(all_endpoint_nodes.min()) != 0:
        raise ValueError('node id values must start at 0')
    if int(all_endpoint_nodes.max()) != n_nodes - 1:
        raise ValueError('node id values must end at n_nodes - 1')
    if not np.array_equal(np.unique(all_endpoint_nodes), np.arange(n_nodes)):
        raise ValueError('node id values must be 0-based contiguous')


def _validate_trace_node_mapping(
    *,
    source_endpoint_id_sorted: np.ndarray,
    receiver_endpoint_id_sorted: np.ndarray,
    source_node_id_by_endpoint: np.ndarray,
    receiver_node_id_by_endpoint: np.ndarray,
    source_node_id_sorted: np.ndarray,
    receiver_node_id_sorted: np.ndarray,
) -> None:
    expected_source = source_node_id_by_endpoint[source_endpoint_id_sorted]
    if not np.array_equal(source_node_id_sorted, expected_source):
        raise ValueError(
            'source_node_id_sorted does not match source_node_id_by_endpoint[source_endpoint_id_sorted]'
        )
    expected_receiver = receiver_node_id_by_endpoint[receiver_endpoint_id_sorted]
    if not np.array_equal(receiver_node_id_sorted, expected_receiver):
        raise ValueError(
            'receiver_node_id_sorted does not match receiver_node_id_by_endpoint[receiver_endpoint_id_sorted]'
        )


def _validate_trace_coordinate_mapping(
    *,
    source_endpoint_id_sorted: np.ndarray,
    receiver_endpoint_id_sorted: np.ndarray,
    source_x_m_sorted: np.ndarray,
    source_y_m_sorted: np.ndarray,
    receiver_x_m_sorted: np.ndarray,
    receiver_y_m_sorted: np.ndarray,
    source_endpoint_x_m: np.ndarray,
    source_endpoint_y_m: np.ndarray,
    receiver_endpoint_x_m: np.ndarray,
    receiver_endpoint_y_m: np.ndarray,
) -> None:
    if not np.allclose(
        source_x_m_sorted,
        source_endpoint_x_m[source_endpoint_id_sorted],
    ):
        raise ValueError(
            'source_x_m_sorted does not match source_endpoint_x_m[source_endpoint_id_sorted]'
        )
    if not np.allclose(
        source_y_m_sorted,
        source_endpoint_y_m[source_endpoint_id_sorted],
    ):
        raise ValueError(
            'source_y_m_sorted does not match source_endpoint_y_m[source_endpoint_id_sorted]'
        )
    if not np.allclose(
        receiver_x_m_sorted,
        receiver_endpoint_x_m[receiver_endpoint_id_sorted],
    ):
        raise ValueError(
            'receiver_x_m_sorted does not match receiver_endpoint_x_m[receiver_endpoint_id_sorted]'
        )
    if not np.allclose(
        receiver_y_m_sorted,
        receiver_endpoint_y_m[receiver_endpoint_id_sorted],
    ):
        raise ValueError(
            'receiver_y_m_sorted does not match receiver_endpoint_y_m[receiver_endpoint_id_sorted]'
        )


def _validate_records(
    *,
    mode: LinkageMode,
    n_source_endpoints: int,
    n_receiver_endpoints: int,
    n_receiver_anchor_links: int,
    n_source_fallback_links: int,
    n_independent_source_nodes: int,
    source_endpoint_x_m: np.ndarray,
    source_endpoint_y_m: np.ndarray,
    source_node_id_by_endpoint: np.ndarray,
    receiver_endpoint_x_m: np.ndarray,
    receiver_endpoint_y_m: np.ndarray,
    receiver_node_id_by_endpoint: np.ndarray,
    record_endpoint_kind: np.ndarray,
    record_endpoint_id: np.ndarray,
    record_x_m: np.ndarray,
    record_y_m: np.ndarray,
    record_node_id: np.ndarray,
    record_linked_to_kind: np.ndarray,
    record_linked_to_id: np.ndarray,
    record_distance_m: np.ndarray,
    record_method: np.ndarray,
) -> None:
    expected_kind = np.asarray(
        ['receiver'] * n_receiver_endpoints + ['source'] * n_source_endpoints,
        dtype=np.str_,
    )
    expected_id = np.concatenate(
        (
            np.arange(n_receiver_endpoints, dtype=np.int64),
            np.arange(n_source_endpoints, dtype=np.int64),
        )
    )
    if not np.array_equal(record_endpoint_kind, expected_kind) or not np.array_equal(
        record_endpoint_id,
        expected_id,
    ):
        raise ValueError(
            'record arrays must be ordered by receiver endpoint id, then source endpoint id'
        )
    if not np.allclose(record_x_m[:n_receiver_endpoints], receiver_endpoint_x_m):
        raise ValueError('record_x_m receiver rows must match receiver_endpoint_x_m')
    if not np.allclose(record_y_m[:n_receiver_endpoints], receiver_endpoint_y_m):
        raise ValueError('record_y_m receiver rows must match receiver_endpoint_y_m')
    if not np.array_equal(
        record_node_id[:n_receiver_endpoints],
        receiver_node_id_by_endpoint,
    ):
        raise ValueError(
            'record_node_id receiver rows must match receiver_node_id_by_endpoint'
        )

    source_slice = slice(n_receiver_endpoints, None)
    if not np.allclose(record_x_m[source_slice], source_endpoint_x_m):
        raise ValueError('record_x_m source rows must match source_endpoint_x_m')
    if not np.allclose(record_y_m[source_slice], source_endpoint_y_m):
        raise ValueError('record_y_m source rows must match source_endpoint_y_m')
    if not np.array_equal(record_node_id[source_slice], source_node_id_by_endpoint):
        raise ValueError('record_node_id source rows must match source_node_id_by_endpoint')

    _validate_record_values(
        n_source_endpoints=n_source_endpoints,
        n_receiver_endpoints=n_receiver_endpoints,
        source_endpoint_x_m=source_endpoint_x_m,
        source_endpoint_y_m=source_endpoint_y_m,
        source_node_id_by_endpoint=source_node_id_by_endpoint,
        receiver_endpoint_x_m=receiver_endpoint_x_m,
        receiver_endpoint_y_m=receiver_endpoint_y_m,
        receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
        record_endpoint_kind=record_endpoint_kind,
        record_endpoint_id=record_endpoint_id,
        record_node_id=record_node_id,
        record_linked_to_kind=record_linked_to_kind,
        record_linked_to_id=record_linked_to_id,
        record_distance_m=record_distance_m,
        record_method=record_method,
    )
    _validate_record_methods_for_mode(
        mode=mode,
        n_receiver_endpoints=n_receiver_endpoints,
        record_method=record_method,
    )
    _validate_summary_counts(
        mode=mode,
        n_receiver_endpoints=n_receiver_endpoints,
        n_receiver_anchor_links=n_receiver_anchor_links,
        n_source_fallback_links=n_source_fallback_links,
        n_independent_source_nodes=n_independent_source_nodes,
        record_method=record_method,
    )


def _validate_record_values(
    *,
    n_source_endpoints: int,
    n_receiver_endpoints: int,
    source_endpoint_x_m: np.ndarray,
    source_endpoint_y_m: np.ndarray,
    source_node_id_by_endpoint: np.ndarray,
    receiver_endpoint_x_m: np.ndarray,
    receiver_endpoint_y_m: np.ndarray,
    receiver_node_id_by_endpoint: np.ndarray,
    record_endpoint_kind: np.ndarray,
    record_endpoint_id: np.ndarray,
    record_node_id: np.ndarray,
    record_linked_to_kind: np.ndarray,
    record_linked_to_id: np.ndarray,
    record_distance_m: np.ndarray,
    record_method: np.ndarray,
) -> None:
    unsupported_link_kinds = sorted(
        set(record_linked_to_kind.tolist()) - _KNOWN_RECORD_LINKED_TO_KINDS
    )
    if unsupported_link_kinds:
        raise ValueError(
            f'record_linked_to_kind contains unsupported value: {unsupported_link_kinds[0]!r}'
        )
    unsupported_methods = sorted(set(record_method.tolist()) - _KNOWN_RECORD_METHODS)
    if unsupported_methods:
        raise ValueError(
            f'record_method contains unsupported value: {unsupported_methods[0]!r}'
        )

    for index, linked_to_kind in enumerate(record_linked_to_kind):
        linked_to_id = int(record_linked_to_id[index])
        distance_m = float(record_distance_m[index])
        endpoint_kind = str(record_endpoint_kind[index])
        endpoint_id = int(record_endpoint_id[index])
        method = str(record_method[index])
        if linked_to_kind == '':
            if linked_to_id != -1:
                raise ValueError(
                    f'record_linked_to_id[{index}] must be -1 when record_linked_to_kind is empty'
                )
            if not np.isnan(distance_m):
                raise ValueError(
                    f'record_distance_m[{index}] must be NaN when record_linked_to_kind is empty'
                )
        else:
            endpoint_count = (
                n_source_endpoints
                if linked_to_kind == 'source'
                else n_receiver_endpoints
            )
            if linked_to_id < 0 or linked_to_id >= endpoint_count:
                raise ValueError(
                    f'record_linked_to_id[{index}] is out of range for {linked_to_kind}'
                )
            if not np.isfinite(distance_m) or distance_m < 0.0:
                raise ValueError(
                    f'record_distance_m[{index}] must be finite and greater than or equal to 0'
                )
        _validate_record_method_link_contract(
            index=index,
            endpoint_kind=endpoint_kind,
            endpoint_id=endpoint_id,
            node_id=int(record_node_id[index]),
            linked_to_kind=str(linked_to_kind),
            linked_to_id=linked_to_id,
            distance_m=distance_m,
            method=method,
            source_endpoint_x_m=source_endpoint_x_m,
            source_endpoint_y_m=source_endpoint_y_m,
            source_node_id_by_endpoint=source_node_id_by_endpoint,
            receiver_endpoint_x_m=receiver_endpoint_x_m,
            receiver_endpoint_y_m=receiver_endpoint_y_m,
            receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
        )


def _validate_record_method_link_contract(
    *,
    index: int,
    endpoint_kind: str,
    endpoint_id: int,
    node_id: int,
    linked_to_kind: str,
    linked_to_id: int,
    distance_m: float,
    method: str,
    source_endpoint_x_m: np.ndarray,
    source_endpoint_y_m: np.ndarray,
    source_node_id_by_endpoint: np.ndarray,
    receiver_endpoint_x_m: np.ndarray,
    receiver_endpoint_y_m: np.ndarray,
    receiver_node_id_by_endpoint: np.ndarray,
) -> None:
    no_link_endpoint_kind_by_method = {
        'receiver_seed': 'receiver',
        'none_mode_receiver_independent': 'receiver',
        'none_mode_source_independent': 'source',
        'source_independent': 'source',
    }
    no_link_endpoint_kind = no_link_endpoint_kind_by_method.get(method)
    if no_link_endpoint_kind is not None:
        if endpoint_kind != no_link_endpoint_kind:
            raise ValueError(
                f'record_method[{index}] {method!r} requires {no_link_endpoint_kind} endpoint'
            )
        if linked_to_kind != '' or linked_to_id != -1 or not np.isnan(distance_m):
            raise ValueError(
                f'record_method[{index}] {method!r} requires empty link and NaN distance_m'
            )
        return

    if method == 'receiver_anchor':
        if endpoint_kind != 'source':
            raise ValueError(
                f'record_method[{index}] receiver_anchor requires source endpoint'
            )
        if linked_to_kind != 'receiver':
            raise ValueError(
                f'record_method[{index}] receiver_anchor requires linked_to_kind receiver'
            )
        expected_node_id = int(receiver_node_id_by_endpoint[linked_to_id])
        if node_id != expected_node_id:
            raise ValueError(
                f'record_node_id[{index}] does not match linked receiver node'
            )
        expected_distance_m = float(
            np.hypot(
                source_endpoint_x_m[endpoint_id]
                - receiver_endpoint_x_m[linked_to_id],
                source_endpoint_y_m[endpoint_id]
                - receiver_endpoint_y_m[linked_to_id],
            )
        )
        _validate_record_distance_matches(
            index=index,
            distance_m=distance_m,
            expected_distance_m=expected_distance_m,
        )
        return

    if method == 'source_fallback':
        if endpoint_kind != 'source':
            raise ValueError(
                f'record_method[{index}] source_fallback requires source endpoint'
            )
        if linked_to_kind != 'source':
            raise ValueError(
                f'record_method[{index}] source_fallback requires linked_to_kind source'
            )
        if linked_to_id == endpoint_id:
            raise ValueError(
                f'record_linked_to_id[{index}] source_fallback must not self-link'
            )
        expected_node_id = int(source_node_id_by_endpoint[linked_to_id])
        if node_id != expected_node_id:
            raise ValueError(
                f'record_node_id[{index}] does not match linked source node'
            )
        expected_distance_m = float(
            np.hypot(
                source_endpoint_x_m[endpoint_id]
                - source_endpoint_x_m[linked_to_id],
                source_endpoint_y_m[endpoint_id]
                - source_endpoint_y_m[linked_to_id],
            )
        )
        _validate_record_distance_matches(
            index=index,
            distance_m=distance_m,
            expected_distance_m=expected_distance_m,
        )


def _validate_record_distance_matches(
    *,
    index: int,
    distance_m: float,
    expected_distance_m: float,
) -> None:
    if not np.isclose(
        distance_m,
        expected_distance_m,
        rtol=_DISTANCE_RTOL,
        atol=_DISTANCE_ATOL,
    ):
        raise ValueError(
            f'record_distance_m[{index}] does not match endpoint geometry distance'
        )


def _validate_record_methods_for_mode(
    *,
    mode: LinkageMode,
    n_receiver_endpoints: int,
    record_method: np.ndarray,
) -> None:
    receiver_methods = record_method[:n_receiver_endpoints]
    source_methods = record_method[n_receiver_endpoints:]
    if mode == 'none':
        if not np.all(receiver_methods == 'none_mode_receiver_independent'):
            raise ValueError(
                'record_method receiver rows must be none_mode_receiver_independent when mode is none'
            )
        if not np.all(source_methods == 'none_mode_source_independent'):
            raise ValueError(
                'record_method source rows must be none_mode_source_independent when mode is none'
            )
        return

    if not np.all(receiver_methods == 'receiver_seed'):
        raise ValueError(
            'record_method receiver rows must be receiver_seed when mode is auto_threshold'
        )
    allowed_source_methods = {
        'receiver_anchor',
        'source_fallback',
        'source_independent',
    }
    bad_source_methods = sorted(set(source_methods.tolist()) - allowed_source_methods)
    if bad_source_methods:
        raise ValueError(
            f'record_method source rows contain unsupported auto_threshold value: {bad_source_methods[0]!r}'
        )


def _validate_summary_counts(
    *,
    mode: LinkageMode,
    n_receiver_endpoints: int,
    n_receiver_anchor_links: int,
    n_source_fallback_links: int,
    n_independent_source_nodes: int,
    record_method: np.ndarray,
) -> None:
    source_methods = record_method[n_receiver_endpoints:]
    receiver_anchor_count = int(np.count_nonzero(source_methods == 'receiver_anchor'))
    source_fallback_count = int(np.count_nonzero(source_methods == 'source_fallback'))
    independent_method = (
        'none_mode_source_independent' if mode == 'none' else 'source_independent'
    )
    independent_count = int(np.count_nonzero(source_methods == independent_method))
    if n_receiver_anchor_links != receiver_anchor_count:
        raise ValueError(
            'n_receiver_anchor_links does not match record_method receiver_anchor count'
        )
    if n_source_fallback_links != source_fallback_count:
        raise ValueError(
            'n_source_fallback_links does not match record_method source_fallback count'
        )
    if n_independent_source_nodes != independent_count:
        raise ValueError(
            'n_independent_source_nodes does not match record_method independent source count'
        )


__all__ = [
    'GeometryLinkageLoadedMetadata',
    'GeometryLinkageTraceNodeMapping',
    'LinkageMode',
    'LoadedGeometryLinkageArtifact',
    'load_geometry_linkage_artifact',
    'load_geometry_linkage_from_job_dir',
    'load_geometry_linkage_trace_node_mapping',
]
