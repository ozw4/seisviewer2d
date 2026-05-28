"""Artifact writer for static geometry linkage results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.common.artifact_io import (
    assert_strict_json,
    validate_npz_no_object_arrays,
    write_csv_atomic as _common_write_csv_atomic,
    write_json_atomic as _common_write_json_atomic,
    write_npz_atomic as _common_write_npz_atomic,
)
from app.services.geometry_linkage_linker import (
    EndpointLinkageRecord,
    GeometryLinkageResult,
)
from app.services.geometry_linkage_tables import (
    EndpointGeometryTable,
    EndpointGeometryTables,
)

GEOMETRY_LINKAGE_NPZ_NAME = 'geometry_linkage.npz'
GEOMETRY_LINKAGE_CSV_NAME = 'geometry_linkage.csv'
GEOMETRY_LINKAGE_QC_JSON_NAME = 'geometry_linkage_qc.json'

SCHEMA_VERSION = 1
SOLUTION_ARTIFACT_KIND = 'geometry_linkage'
QC_ARTIFACT_KIND = 'geometry_linkage_qc'
ORDER = 'trace_store_sorted'

_KNOWN_MODES = {'none', 'auto_threshold'}
_KNOWN_ENDPOINT_KINDS = {'source', 'receiver'}
_KNOWN_METHODS = {
    'none_mode_receiver_independent',
    'none_mode_source_independent',
    'receiver_seed',
    'receiver_anchor',
    'source_fallback',
    'source_independent',
}
_DISTANCE_RTOL = 1.0e-9
_DISTANCE_ATOL = 1.0e-9

_CSV_COLUMNS = [
    'endpoint_kind',
    'endpoint_id',
    'x_m',
    'y_m',
    'node_id',
    'linked_to_kind',
    'linked_to_id',
    'distance_m',
    'method',
]


@dataclass(frozen=True)
class GeometryLinkageArtifactPaths:
    linkage_npz_path: Path
    linkage_csv_path: Path
    qc_json_path: Path


@dataclass(frozen=True)
class GeometryLinkageArtifactMetadata:
    job_id: str | None = None
    input_file_id: str | None = None
    key1_byte: int | None = None
    key2_byte: int | None = None
    source_x_byte: int | None = None
    source_y_byte: int | None = None
    receiver_x_byte: int | None = None
    receiver_y_byte: int | None = None
    coordinate_scalar_byte: int | None = None
    header_source_segy_path: str | None = None


@dataclass(frozen=True)
class _ValidatedLinkageContext:
    n_traces: int
    n_source_endpoints: int
    n_receiver_endpoints: int
    n_nodes: int
    source_x_m_sorted: np.ndarray
    source_y_m_sorted: np.ndarray
    receiver_x_m_sorted: np.ndarray
    receiver_y_m_sorted: np.ndarray
    coordinate_scalar_sorted: np.ndarray
    source_endpoint_id_sorted: np.ndarray
    receiver_endpoint_id_sorted: np.ndarray
    source_endpoint_id: np.ndarray
    source_endpoint_x_m: np.ndarray
    source_endpoint_y_m: np.ndarray
    source_endpoint_first_sorted_trace_index: np.ndarray
    source_endpoint_trace_count: np.ndarray
    receiver_endpoint_id: np.ndarray
    receiver_endpoint_x_m: np.ndarray
    receiver_endpoint_y_m: np.ndarray
    receiver_endpoint_first_sorted_trace_index: np.ndarray
    receiver_endpoint_trace_count: np.ndarray
    source_node_id_by_endpoint: np.ndarray
    receiver_node_id_by_endpoint: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    records: tuple[EndpointLinkageRecord, ...]


def build_geometry_linkage_solution_arrays(
    tables: EndpointGeometryTables,
    linkage: GeometryLinkageResult,
    *,
    metadata: GeometryLinkageArtifactMetadata | None = None,
) -> dict[str, np.ndarray]:
    """Build the pickle-free NPZ payload for geometry linkage artifacts."""
    context = _validate_artifact_inputs(tables, linkage)
    metadata = metadata or GeometryLinkageArtifactMetadata()

    arrays: dict[str, np.ndarray] = {
        'schema_version': _scalar_int(SCHEMA_VERSION),
        'artifact_kind': _scalar_str(SOLUTION_ARTIFACT_KIND),
        'order': _scalar_str(ORDER),
        'mode': _scalar_str(linkage.mode),
        'threshold_m': _scalar_optional_float(linkage.threshold_m),
        'receiver_location_interval_m': _scalar_optional_float(
            linkage.receiver_location_interval_m
        ),
        'prefer_receiver_anchor': _scalar_bool(linkage.prefer_receiver_anchor),
        'n_traces': _scalar_int(context.n_traces),
        'n_source_endpoints': _scalar_int(context.n_source_endpoints),
        'n_receiver_endpoints': _scalar_int(context.n_receiver_endpoints),
        'n_nodes': _scalar_int(context.n_nodes),
        'n_receiver_anchor_links': _scalar_int(linkage.n_receiver_anchor_links),
        'n_source_fallback_links': _scalar_int(linkage.n_source_fallback_links),
        'n_independent_source_nodes': _scalar_int(
            linkage.n_independent_source_nodes
        ),
        'coordinate_scalar_zero_count': _scalar_int(tables.scalar_zero_count),
        'job_id': _scalar_str(metadata.job_id),
        'input_file_id': _scalar_str(metadata.input_file_id),
        'key1_byte': _scalar_optional_float(metadata.key1_byte),
        'key2_byte': _scalar_optional_float(metadata.key2_byte),
        'source_x_byte': _scalar_optional_float(metadata.source_x_byte),
        'source_y_byte': _scalar_optional_float(metadata.source_y_byte),
        'receiver_x_byte': _scalar_optional_float(metadata.receiver_x_byte),
        'receiver_y_byte': _scalar_optional_float(metadata.receiver_y_byte),
        'coordinate_scalar_byte': _scalar_optional_float(
            metadata.coordinate_scalar_byte
        ),
        'header_source_segy_path': _scalar_str(metadata.header_source_segy_path),
        'source_x_m_sorted': context.source_x_m_sorted,
        'source_y_m_sorted': context.source_y_m_sorted,
        'receiver_x_m_sorted': context.receiver_x_m_sorted,
        'receiver_y_m_sorted': context.receiver_y_m_sorted,
        'coordinate_scalar_sorted': context.coordinate_scalar_sorted,
        'source_endpoint_id_sorted': context.source_endpoint_id_sorted,
        'receiver_endpoint_id_sorted': context.receiver_endpoint_id_sorted,
        'source_node_id_sorted': context.source_node_id_sorted,
        'receiver_node_id_sorted': context.receiver_node_id_sorted,
        'source_endpoint_id': context.source_endpoint_id,
        'source_endpoint_x_m': context.source_endpoint_x_m,
        'source_endpoint_y_m': context.source_endpoint_y_m,
        'source_endpoint_first_sorted_trace_index': (
            context.source_endpoint_first_sorted_trace_index
        ),
        'source_endpoint_trace_count': context.source_endpoint_trace_count,
        'source_node_id_by_endpoint': context.source_node_id_by_endpoint,
        'receiver_endpoint_id': context.receiver_endpoint_id,
        'receiver_endpoint_x_m': context.receiver_endpoint_x_m,
        'receiver_endpoint_y_m': context.receiver_endpoint_y_m,
        'receiver_endpoint_first_sorted_trace_index': (
            context.receiver_endpoint_first_sorted_trace_index
        ),
        'receiver_endpoint_trace_count': context.receiver_endpoint_trace_count,
        'receiver_node_id_by_endpoint': context.receiver_node_id_by_endpoint,
        **_record_arrays(context.records),
    }
    _validate_npz_payload_is_pickle_free(arrays)
    return arrays


def build_geometry_linkage_qc_payload(
    tables: EndpointGeometryTables,
    linkage: GeometryLinkageResult,
    *,
    metadata: GeometryLinkageArtifactMetadata | None = None,
) -> dict[str, Any]:
    """Build the strict-JSON QC summary for geometry linkage artifacts."""
    context = _validate_artifact_inputs(tables, linkage)
    metadata = metadata or GeometryLinkageArtifactMetadata()

    source_nodes_used = set(context.source_node_id_sorted.tolist())
    receiver_nodes_used = set(context.receiver_node_id_sorted.tolist())
    both = source_nodes_used & receiver_nodes_used
    source_only = source_nodes_used - receiver_nodes_used
    receiver_only = receiver_nodes_used - source_nodes_used

    receiver_anchor_distance_m = [
        record.distance_m
        for record in context.records
        if record.method == 'receiver_anchor'
    ]
    source_fallback_distance_m = [
        record.distance_m
        for record in context.records
        if record.method == 'source_fallback'
    ]
    node_trace_count = np.bincount(
        np.concatenate(
            (context.source_node_id_sorted, context.receiver_node_id_sorted)
        ),
        minlength=context.n_nodes,
    ).astype(np.int64)

    payload = {
        'schema_version': SCHEMA_VERSION,
        'artifact_kind': QC_ARTIFACT_KIND,
        'order': ORDER,
        'mode': str(linkage.mode),
        'threshold_m': _json_float_or_none(linkage.threshold_m),
        'receiver_location_interval_m': _json_float_or_none(
            linkage.receiver_location_interval_m
        ),
        'prefer_receiver_anchor': bool(linkage.prefer_receiver_anchor),
        'job': {
            'job_id': _json_str_or_none(metadata.job_id),
            'input_file_id': _json_str_or_none(metadata.input_file_id),
        },
        'headers': {
            'key1_byte': _json_int_or_none(metadata.key1_byte),
            'key2_byte': _json_int_or_none(metadata.key2_byte),
            'source_x_byte': _json_int_or_none(metadata.source_x_byte),
            'source_y_byte': _json_int_or_none(metadata.source_y_byte),
            'receiver_x_byte': _json_int_or_none(metadata.receiver_x_byte),
            'receiver_y_byte': _json_int_or_none(metadata.receiver_y_byte),
            'coordinate_scalar_byte': _json_int_or_none(
                metadata.coordinate_scalar_byte
            ),
            'header_source_segy_path': _json_str_or_none(
                metadata.header_source_segy_path
            ),
        },
        'counts': {
            'n_traces': context.n_traces,
            'n_source_endpoints': context.n_source_endpoints,
            'n_receiver_endpoints': context.n_receiver_endpoints,
            'n_nodes': context.n_nodes,
            'n_receiver_anchor_links': int(linkage.n_receiver_anchor_links),
            'n_source_fallback_links': int(linkage.n_source_fallback_links),
            'n_independent_source_nodes': int(linkage.n_independent_source_nodes),
            'n_source_only_nodes': max(
                0,
                context.n_nodes - context.n_receiver_endpoints,
            ),
            'n_nodes_used_by_both_source_and_receiver': len(both),
            'n_nodes_used_only_by_source': len(source_only),
            'n_nodes_used_only_by_receiver': len(receiver_only),
        },
        'coordinate_scalar': {
            'zero_count': int(tables.scalar_zero_count),
        },
        'source_endpoint_trace_count': summarize_finite_values(
            context.source_endpoint_trace_count
        ),
        'receiver_endpoint_trace_count': summarize_finite_values(
            context.receiver_endpoint_trace_count
        ),
        'receiver_anchor_distance_m': summarize_finite_values(
            np.asarray(receiver_anchor_distance_m, dtype=np.float64)
        ),
        'source_fallback_distance_m': summarize_finite_values(
            np.asarray(source_fallback_distance_m, dtype=np.float64)
        ),
        'node_trace_count': summarize_finite_values(node_trace_count),
    }
    _assert_strict_json_payload(payload)
    return payload


def write_geometry_linkage_artifacts(
    *,
    job_dir: Path,
    tables: EndpointGeometryTables,
    linkage: GeometryLinkageResult,
    metadata: GeometryLinkageArtifactMetadata | None = None,
) -> GeometryLinkageArtifactPaths:
    """Write geometry linkage NPZ, CSV, and QC artifacts atomically."""
    solution_payload = build_geometry_linkage_solution_arrays(
        tables,
        linkage,
        metadata=metadata,
    )
    qc_payload = build_geometry_linkage_qc_payload(
        tables,
        linkage,
        metadata=metadata,
    )
    csv_rows = _build_csv_rows(_validate_artifact_inputs(tables, linkage).records)

    job_dir_path = Path(job_dir)
    job_dir_path.mkdir(parents=True, exist_ok=True)
    paths = GeometryLinkageArtifactPaths(
        linkage_npz_path=job_dir_path / GEOMETRY_LINKAGE_NPZ_NAME,
        linkage_csv_path=job_dir_path / GEOMETRY_LINKAGE_CSV_NAME,
        qc_json_path=job_dir_path / GEOMETRY_LINKAGE_QC_JSON_NAME,
    )

    _write_npz_atomic(paths.linkage_npz_path, solution_payload)
    _write_csv_atomic(paths.linkage_csv_path, csv_rows)
    _write_json_atomic(paths.qc_json_path, qc_payload)
    return paths


def summarize_finite_values(values: np.ndarray) -> dict[str, int | float | None]:
    """Summarize finite numeric values for strict-JSON QC payloads."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError('values must be a 1D array')
    if np.issubdtype(arr.dtype, np.bool_):
        raise ValueError('values must be numeric')
    try:
        arr_f64 = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('values must be numeric') from exc

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
            'mad': None,
        }

    median = float(np.median(finite))
    return {
        'count': count,
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'mean': float(np.mean(finite)),
        'median': median,
        'std': float(np.std(finite, ddof=0)),
        'mad': float(np.median(np.abs(finite - median))),
    }


def _validate_artifact_inputs(
    tables: EndpointGeometryTables,
    linkage: GeometryLinkageResult,
) -> _ValidatedLinkageContext:
    if linkage.mode not in _KNOWN_MODES:
        raise ValueError("method mode must be 'none' or 'auto_threshold'")

    n_traces = _coerce_positive_int(tables.n_traces, name='tables.n_traces')
    if int(linkage.n_traces) != n_traces:
        raise ValueError('linkage.n_traces must match tables.n_traces')

    expected_trace_shape = (n_traces,)
    source_x_m_sorted = _coerce_1d_finite_float64(
        tables.source_x_m_sorted,
        name='source_x_m_sorted',
        expected_shape=expected_trace_shape,
    )
    source_y_m_sorted = _coerce_1d_finite_float64(
        tables.source_y_m_sorted,
        name='source_y_m_sorted',
        expected_shape=expected_trace_shape,
    )
    receiver_x_m_sorted = _coerce_1d_finite_float64(
        tables.receiver_x_m_sorted,
        name='receiver_x_m_sorted',
        expected_shape=expected_trace_shape,
    )
    receiver_y_m_sorted = _coerce_1d_finite_float64(
        tables.receiver_y_m_sorted,
        name='receiver_y_m_sorted',
        expected_shape=expected_trace_shape,
    )
    coordinate_scalar_sorted = _coerce_1d_integer_int64(
        tables.coordinate_scalar_sorted,
        name='coordinate_scalar_sorted',
        expected_shape=expected_trace_shape,
    )

    (
        source_endpoint_id,
        source_endpoint_x_m,
        source_endpoint_y_m,
        source_endpoint_first_sorted_trace_index,
        source_endpoint_trace_count,
    ) = _validate_endpoint_table(
        tables.source_endpoints,
        expected_kind='source',
        name='source_endpoints',
    )
    (
        receiver_endpoint_id,
        receiver_endpoint_x_m,
        receiver_endpoint_y_m,
        receiver_endpoint_first_sorted_trace_index,
        receiver_endpoint_trace_count,
    ) = _validate_endpoint_table(
        tables.receiver_endpoints,
        expected_kind='receiver',
        name='receiver_endpoints',
    )
    n_source_endpoints = int(source_endpoint_id.shape[0])
    n_receiver_endpoints = int(receiver_endpoint_id.shape[0])
    if int(linkage.n_source_endpoints) != n_source_endpoints:
        raise ValueError(
            'linkage.n_source_endpoints must match source endpoint count'
        )
    if int(linkage.n_receiver_endpoints) != n_receiver_endpoints:
        raise ValueError(
            'linkage.n_receiver_endpoints must match receiver endpoint count'
        )

    source_endpoint_id_sorted = _validate_sorted_endpoint_ids(
        tables.source_endpoint_id_sorted,
        name='source_endpoint_id_sorted',
        n_traces=n_traces,
        n_endpoints=n_source_endpoints,
    )
    receiver_endpoint_id_sorted = _validate_sorted_endpoint_ids(
        tables.receiver_endpoint_id_sorted,
        name='receiver_endpoint_id_sorted',
        n_traces=n_traces,
        n_endpoints=n_receiver_endpoints,
    )

    source_node_id_by_endpoint = _coerce_1d_integer_int64(
        linkage.source_node_id_by_endpoint,
        name='source_node_id_by_endpoint',
        expected_shape=(n_source_endpoints,),
    )
    receiver_node_id_by_endpoint = _coerce_1d_integer_int64(
        linkage.receiver_node_id_by_endpoint,
        name='receiver_node_id_by_endpoint',
        expected_shape=(n_receiver_endpoints,),
    )
    source_node_id_sorted = _coerce_1d_integer_int64(
        linkage.source_node_id_sorted,
        name='source_node_id_sorted',
        expected_shape=expected_trace_shape,
    )
    receiver_node_id_sorted = _coerce_1d_integer_int64(
        linkage.receiver_node_id_sorted,
        name='receiver_node_id_sorted',
        expected_shape=expected_trace_shape,
    )
    if not np.array_equal(
        source_node_id_sorted,
        source_node_id_by_endpoint[source_endpoint_id_sorted],
    ):
        raise ValueError(
            'source_node_id_sorted must match source_node_id_by_endpoint'
        )
    if not np.array_equal(
        receiver_node_id_sorted,
        receiver_node_id_by_endpoint[receiver_endpoint_id_sorted],
    ):
        raise ValueError(
            'receiver_node_id_sorted must match receiver_node_id_by_endpoint'
        )

    n_nodes = _validate_contiguous_node_ids(
        source_node_id_by_endpoint,
        receiver_node_id_by_endpoint,
        linkage_n_nodes=int(linkage.n_nodes),
    )
    records = _validate_records(
        linkage.records,
        source_endpoint_x_m=source_endpoint_x_m,
        source_endpoint_y_m=source_endpoint_y_m,
        receiver_endpoint_x_m=receiver_endpoint_x_m,
        receiver_endpoint_y_m=receiver_endpoint_y_m,
        source_node_id_by_endpoint=source_node_id_by_endpoint,
        receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
    )
    _validate_nonnegative_count(
        linkage.n_receiver_anchor_links,
        name='n_receiver_anchor_links',
    )
    _validate_nonnegative_count(
        linkage.n_source_fallback_links,
        name='n_source_fallback_links',
    )
    _validate_nonnegative_count(
        linkage.n_independent_source_nodes,
        name='n_independent_source_nodes',
    )
    _validate_optional_finite_nonnegative_float(
        linkage.threshold_m,
        name='threshold_m',
    )
    _validate_optional_finite_nonnegative_float(
        linkage.receiver_location_interval_m,
        name='receiver_location_interval_m',
    )
    if not isinstance(linkage.prefer_receiver_anchor, (bool, np.bool_)):
        raise ValueError('prefer_receiver_anchor must be bool')

    return _ValidatedLinkageContext(
        n_traces=n_traces,
        n_source_endpoints=n_source_endpoints,
        n_receiver_endpoints=n_receiver_endpoints,
        n_nodes=n_nodes,
        source_x_m_sorted=source_x_m_sorted,
        source_y_m_sorted=source_y_m_sorted,
        receiver_x_m_sorted=receiver_x_m_sorted,
        receiver_y_m_sorted=receiver_y_m_sorted,
        coordinate_scalar_sorted=coordinate_scalar_sorted,
        source_endpoint_id_sorted=source_endpoint_id_sorted,
        receiver_endpoint_id_sorted=receiver_endpoint_id_sorted,
        source_endpoint_id=source_endpoint_id,
        source_endpoint_x_m=source_endpoint_x_m,
        source_endpoint_y_m=source_endpoint_y_m,
        source_endpoint_first_sorted_trace_index=(
            source_endpoint_first_sorted_trace_index
        ),
        source_endpoint_trace_count=source_endpoint_trace_count,
        receiver_endpoint_id=receiver_endpoint_id,
        receiver_endpoint_x_m=receiver_endpoint_x_m,
        receiver_endpoint_y_m=receiver_endpoint_y_m,
        receiver_endpoint_first_sorted_trace_index=(
            receiver_endpoint_first_sorted_trace_index
        ),
        receiver_endpoint_trace_count=receiver_endpoint_trace_count,
        source_node_id_by_endpoint=source_node_id_by_endpoint,
        receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
        source_node_id_sorted=source_node_id_sorted,
        receiver_node_id_sorted=receiver_node_id_sorted,
        records=records,
    )


def _validate_endpoint_table(
    table: EndpointGeometryTable,
    *,
    expected_kind: str,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if getattr(table, 'endpoint_kind', None) != expected_kind:
        raise ValueError(f'{name}.endpoint_kind must be {expected_kind!r}')

    endpoint_id = _coerce_1d_integer_int64(
        getattr(table, 'endpoint_id'),
        name=f'{name}.endpoint_id',
    )
    expected_endpoint_id = np.arange(endpoint_id.shape[0], dtype=np.int64)
    if not np.array_equal(endpoint_id, expected_endpoint_id):
        raise ValueError(f'{name}.endpoint_id must be 0-based contiguous')
    expected_shape = endpoint_id.shape

    x_m = _coerce_1d_finite_float64(
        getattr(table, 'x_m'),
        name=f'{name}.x_m',
        expected_shape=expected_shape,
    )
    y_m = _coerce_1d_finite_float64(
        getattr(table, 'y_m'),
        name=f'{name}.y_m',
        expected_shape=expected_shape,
    )
    first_sorted_trace_index = _coerce_1d_integer_int64(
        getattr(table, 'first_sorted_trace_index'),
        name=f'{name}.first_sorted_trace_index',
        expected_shape=expected_shape,
    )
    trace_count = _coerce_1d_integer_int64(
        getattr(table, 'trace_count'),
        name=f'{name}.trace_count',
        expected_shape=expected_shape,
    )
    if np.any(first_sorted_trace_index < 0):
        raise ValueError(f'{name}.first_sorted_trace_index must be nonnegative')
    if np.any(trace_count <= 0):
        raise ValueError(f'{name}.trace_count must be greater than 0')
    return endpoint_id, x_m, y_m, first_sorted_trace_index, trace_count


def _validate_sorted_endpoint_ids(
    values: np.ndarray,
    *,
    name: str,
    n_traces: int,
    n_endpoints: int,
) -> np.ndarray:
    arr = _coerce_1d_integer_int64(values, name=name, expected_shape=(n_traces,))
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= n_endpoints):
        raise ValueError(f'{name} values must be in range 0..{n_endpoints - 1}')
    return arr


def _validate_contiguous_node_ids(
    source_node_id_by_endpoint: np.ndarray,
    receiver_node_id_by_endpoint: np.ndarray,
    *,
    linkage_n_nodes: int,
) -> int:
    node_ids = np.concatenate((source_node_id_by_endpoint, receiver_node_id_by_endpoint))
    if node_ids.size == 0:
        raise ValueError('node ids must be non-empty')
    if np.any(node_ids < 0):
        raise ValueError('node ids must be greater than or equal to 0')
    n_nodes = int(node_ids.max()) + 1
    expected_node_ids = np.arange(n_nodes, dtype=np.int64)
    if not np.array_equal(np.unique(node_ids), expected_node_ids):
        raise ValueError('node id values must be 0-based contiguous')
    if linkage_n_nodes != n_nodes:
        raise ValueError('linkage.n_nodes must equal max(node_id) + 1')
    return n_nodes


def _validate_records(
    records: tuple[EndpointLinkageRecord, ...],
    *,
    source_endpoint_x_m: np.ndarray,
    source_endpoint_y_m: np.ndarray,
    receiver_endpoint_x_m: np.ndarray,
    receiver_endpoint_y_m: np.ndarray,
    source_node_id_by_endpoint: np.ndarray,
    receiver_node_id_by_endpoint: np.ndarray,
) -> tuple[EndpointLinkageRecord, ...]:
    expected: list[tuple[str, int]] = [
        ('receiver', endpoint_id)
        for endpoint_id in range(int(receiver_endpoint_x_m.shape[0]))
    ]
    expected.extend(
        ('source', endpoint_id)
        for endpoint_id in range(int(source_endpoint_x_m.shape[0]))
    )
    if len(records) != len(expected):
        raise ValueError(
            'records length must equal n_receiver_endpoints + n_source_endpoints'
        )

    for index, (record, expected_key) in enumerate(zip(records, expected)):
        if (record.endpoint_kind, int(record.endpoint_id)) != expected_key:
            raise ValueError(
                'records must be ordered by receiver endpoint id, then source endpoint id'
            )
        if record.endpoint_kind not in _KNOWN_ENDPOINT_KINDS:
            raise ValueError(f'records[{index}].endpoint_kind is unsupported')
        if record.method not in _KNOWN_METHODS:
            raise ValueError(f'records[{index}].method is unsupported')

        expected_node_id = _record_expected_node_id(
            record,
            source_node_id_by_endpoint=source_node_id_by_endpoint,
            receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
        )
        if int(record.node_id) != expected_node_id:
            raise ValueError(
                f'records[{index}].node_id must match endpoint-level mapping'
            )
        expected_x_m, expected_y_m = _record_expected_coordinates(
            record,
            source_endpoint_x_m=source_endpoint_x_m,
            source_endpoint_y_m=source_endpoint_y_m,
            receiver_endpoint_x_m=receiver_endpoint_x_m,
            receiver_endpoint_y_m=receiver_endpoint_y_m,
        )
        if not np.isfinite(float(record.x_m)) or not np.isfinite(float(record.y_m)):
            raise ValueError(f'records[{index}] coordinate must be finite')
        if not np.isclose(float(record.x_m), expected_x_m) or not np.isclose(
            float(record.y_m),
            expected_y_m,
        ):
            raise ValueError(f'records[{index}] coordinate must match endpoint table')
        _validate_record_link(record, index=index)
        _validate_optional_finite_nonnegative_float(
            record.distance_m,
            name=f'records[{index}].distance_m',
        )
        _validate_record_method_link_contract(
            record,
            index=index,
            source_endpoint_x_m=source_endpoint_x_m,
            source_endpoint_y_m=source_endpoint_y_m,
            receiver_endpoint_x_m=receiver_endpoint_x_m,
            receiver_endpoint_y_m=receiver_endpoint_y_m,
            source_node_id_by_endpoint=source_node_id_by_endpoint,
            receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
        )
    return tuple(records)


def _record_expected_node_id(
    record: EndpointLinkageRecord,
    *,
    source_node_id_by_endpoint: np.ndarray,
    receiver_node_id_by_endpoint: np.ndarray,
) -> int:
    if record.endpoint_kind == 'source':
        return int(source_node_id_by_endpoint[int(record.endpoint_id)])
    return int(receiver_node_id_by_endpoint[int(record.endpoint_id)])


def _record_expected_coordinates(
    record: EndpointLinkageRecord,
    *,
    source_endpoint_x_m: np.ndarray,
    source_endpoint_y_m: np.ndarray,
    receiver_endpoint_x_m: np.ndarray,
    receiver_endpoint_y_m: np.ndarray,
) -> tuple[float, float]:
    endpoint_id = int(record.endpoint_id)
    if record.endpoint_kind == 'source':
        return (
            float(source_endpoint_x_m[endpoint_id]),
            float(source_endpoint_y_m[endpoint_id]),
        )
    return (
        float(receiver_endpoint_x_m[endpoint_id]),
        float(receiver_endpoint_y_m[endpoint_id]),
    )


def _validate_record_link(record: EndpointLinkageRecord, *, index: int) -> None:
    if record.linked_to_kind is None:
        if record.linked_to_id is not None:
            raise ValueError(
                f'records[{index}].linked_to_id must be None when linked_to_kind is None'
            )
        return
    if record.linked_to_kind not in _KNOWN_ENDPOINT_KINDS:
        raise ValueError(f'records[{index}].linked_to_kind is unsupported')
    if record.linked_to_id is None:
        raise ValueError(
            f'records[{index}].linked_to_id is required when linked_to_kind is set'
        )
    linked_to_id = int(record.linked_to_id)
    if linked_to_id < 0:
        raise ValueError(f'records[{index}].linked_to_id must be nonnegative')


def _validate_record_method_link_contract(
    record: EndpointLinkageRecord,
    *,
    index: int,
    source_endpoint_x_m: np.ndarray,
    source_endpoint_y_m: np.ndarray,
    receiver_endpoint_x_m: np.ndarray,
    receiver_endpoint_y_m: np.ndarray,
    source_node_id_by_endpoint: np.ndarray,
    receiver_node_id_by_endpoint: np.ndarray,
) -> None:
    no_link_endpoint_kind_by_method = {
        'receiver_seed': 'receiver',
        'none_mode_receiver_independent': 'receiver',
        'none_mode_source_independent': 'source',
        'source_independent': 'source',
    }
    no_link_endpoint_kind = no_link_endpoint_kind_by_method.get(record.method)
    if no_link_endpoint_kind is not None:
        if record.endpoint_kind != no_link_endpoint_kind:
            raise ValueError(
                f'records[{index}].method {record.method!r} requires '
                f'{no_link_endpoint_kind} endpoint'
            )
        if (
            record.linked_to_kind is not None
            or record.linked_to_id is not None
            or record.distance_m is not None
        ):
            raise ValueError(
                f'records[{index}].method {record.method!r} requires empty link '
                'and no distance_m'
            )
        return

    if record.method == 'receiver_anchor':
        if record.endpoint_kind != 'source':
            raise ValueError(
                f'records[{index}].receiver_anchor requires source endpoint'
            )
        if record.linked_to_kind != 'receiver':
            raise ValueError(
                f'records[{index}].receiver_anchor requires linked_to_kind receiver'
            )
        linked_receiver_id = _record_linked_to_id_in_range(
            record,
            index=index,
            endpoint_count=int(receiver_endpoint_x_m.shape[0]),
            linked_to_kind='receiver',
        )
        expected_node_id = int(receiver_node_id_by_endpoint[linked_receiver_id])
        if int(record.node_id) != expected_node_id:
            raise ValueError(
                f'records[{index}].node_id must match linked receiver node'
            )
        endpoint_id = int(record.endpoint_id)
        expected_distance_m = float(
            np.hypot(
                source_endpoint_x_m[endpoint_id]
                - receiver_endpoint_x_m[linked_receiver_id],
                source_endpoint_y_m[endpoint_id]
                - receiver_endpoint_y_m[linked_receiver_id],
            )
        )
        _validate_record_distance_matches(
            record,
            index=index,
            expected_distance_m=expected_distance_m,
        )
        return

    if record.method == 'source_fallback':
        if record.endpoint_kind != 'source':
            raise ValueError(
                f'records[{index}].source_fallback requires source endpoint'
            )
        if record.linked_to_kind != 'source':
            raise ValueError(
                f'records[{index}].source_fallback requires linked_to_kind source'
            )
        linked_source_id = _record_linked_to_id_in_range(
            record,
            index=index,
            endpoint_count=int(source_endpoint_x_m.shape[0]),
            linked_to_kind='source',
        )
        endpoint_id = int(record.endpoint_id)
        if linked_source_id == endpoint_id:
            raise ValueError(f'records[{index}].source_fallback must not self-link')
        expected_node_id = int(source_node_id_by_endpoint[linked_source_id])
        if int(record.node_id) != expected_node_id:
            raise ValueError(
                f'records[{index}].node_id must match linked source node'
            )
        expected_distance_m = float(
            np.hypot(
                source_endpoint_x_m[endpoint_id] - source_endpoint_x_m[linked_source_id],
                source_endpoint_y_m[endpoint_id] - source_endpoint_y_m[linked_source_id],
            )
        )
        _validate_record_distance_matches(
            record,
            index=index,
            expected_distance_m=expected_distance_m,
        )


def _record_linked_to_id_in_range(
    record: EndpointLinkageRecord,
    *,
    index: int,
    endpoint_count: int,
    linked_to_kind: str,
) -> int:
    if record.linked_to_id is None:
        raise ValueError(
            f'records[{index}].linked_to_id is required for {record.method}'
        )
    linked_to_id = int(record.linked_to_id)
    if linked_to_id < 0 or linked_to_id >= endpoint_count:
        raise ValueError(
            f'records[{index}].linked_to_id is out of range for {linked_to_kind}'
        )
    return linked_to_id


def _validate_record_distance_matches(
    record: EndpointLinkageRecord,
    *,
    index: int,
    expected_distance_m: float,
) -> None:
    if record.distance_m is None:
        raise ValueError(f'records[{index}].distance_m is required for {record.method}')
    distance_m = float(record.distance_m)
    if not np.isclose(
        distance_m,
        expected_distance_m,
        rtol=_DISTANCE_RTOL,
        atol=_DISTANCE_ATOL,
    ):
        raise ValueError(
            f'records[{index}].distance_m must match endpoint geometry distance'
        )


def _record_arrays(
    records: tuple[EndpointLinkageRecord, ...],
) -> dict[str, np.ndarray]:
    return {
        'record_endpoint_kind': _str_array(
            [record.endpoint_kind for record in records]
        ),
        'record_endpoint_id': _int_array(
            [int(record.endpoint_id) for record in records]
        ),
        'record_x_m': _float_array([float(record.x_m) for record in records]),
        'record_y_m': _float_array([float(record.y_m) for record in records]),
        'record_node_id': _int_array([int(record.node_id) for record in records]),
        'record_linked_to_kind': _str_array(
            [
                '' if record.linked_to_kind is None else record.linked_to_kind
                for record in records
            ]
        ),
        'record_linked_to_id': _int_array(
            [
                -1 if record.linked_to_id is None else int(record.linked_to_id)
                for record in records
            ]
        ),
        'record_distance_m': _float_array(
            [
                np.nan if record.distance_m is None else float(record.distance_m)
                for record in records
            ]
        ),
        'record_method': _str_array([record.method for record in records]),
    }


def _build_csv_rows(
    records: tuple[EndpointLinkageRecord, ...],
) -> list[dict[str, object]]:
    return [
        {
            'endpoint_kind': record.endpoint_kind,
            'endpoint_id': int(record.endpoint_id),
            'x_m': float(record.x_m),
            'y_m': float(record.y_m),
            'node_id': int(record.node_id),
            'linked_to_kind': '' if record.linked_to_kind is None else record.linked_to_kind,
            'linked_to_id': '' if record.linked_to_id is None else int(record.linked_to_id),
            'distance_m': _csv_optional_float(record.distance_m),
            'method': record.method,
        }
        for record in records
    ]


def _validate_npz_payload_is_pickle_free(arrays: dict[str, np.ndarray]) -> None:
    validate_npz_no_object_arrays(arrays)


def _write_npz_atomic(out_path: Path, payload: dict[str, np.ndarray]) -> None:
    _common_write_npz_atomic(
        out_path,
        payload,
        compressed=False,
        reject_object_arrays=True,
    )


def _write_json_atomic(out_path: Path, payload: dict[str, Any]) -> None:
    _common_write_json_atomic(
        out_path,
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        trailing_newline=True,
    )


def _write_csv_atomic(out_path: Path, rows: list[dict[str, object]]) -> None:
    _common_write_csv_atomic(
        out_path,
        columns=_CSV_COLUMNS,
        rows=rows,
        extrasaction='raise',
        lineterminator='\r\n',
    )


def _coerce_1d_finite_float64(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{name} must be numeric')
    try:
        arr_f64 = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not np.all(np.isfinite(arr_f64)):
        raise ValueError(f'{name} must contain only finite values')
    return np.ascontiguousarray(arr_f64, dtype=np.float64)


def _coerce_1d_integer_int64(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{name} must contain integer values')
    if np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr, dtype=np.int64)
    try:
        arr_f64 = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain integer values') from exc
    if not np.all(np.isfinite(arr_f64)):
        raise ValueError(f'{name} must contain only finite values')
    if not np.all(arr_f64 == np.rint(arr_f64)):
        raise ValueError(f'{name} must contain integer values')
    return np.ascontiguousarray(arr_f64, dtype=np.int64)


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    out = int(value)
    if out <= 0:
        raise ValueError(f'{name} must be greater than 0')
    return out


def _validate_nonnegative_count(value: object, *, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if int(value) < 0:
        raise ValueError(f'{name} must be nonnegative')


def _validate_optional_finite_nonnegative_float(
    value: object,
    *,
    name: str,
) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be finite and nonnegative')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite and nonnegative') from exc
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f'{name} must be finite and nonnegative')


def _scalar_int(value: object) -> np.ndarray:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('integer scalar expected')
    return np.asarray(int(value), dtype=np.int64)


def _scalar_optional_float(value: object) -> np.ndarray:
    if value is None:
        return np.asarray(np.nan, dtype=np.float64)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('float scalar expected')
    out = float(value)
    if not np.isfinite(out):
        raise ValueError('float scalar must be finite')
    return np.asarray(out, dtype=np.float64)


def _scalar_bool(value: object) -> np.ndarray:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('bool scalar expected')
    return np.asarray(bool(value), dtype=np.bool_)


def _scalar_str(value: object) -> np.ndarray:
    if value is None:
        value = ''
    return np.asarray(str(value), dtype=np.str_)


def _str_array(values: list[object]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=np.str_)


def _int_array(values: list[int]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.int64))


def _float_array(values: list[float]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def _csv_optional_float(value: object) -> float | str:
    if value is None:
        return ''
    out = float(value)
    if not np.isfinite(out):
        return ''
    return out


def _json_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('integer metadata expected')
    return int(value)


def _json_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('float value expected')
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


def _json_str_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _assert_strict_json_payload(payload: dict[str, Any]) -> None:
    assert_strict_json(payload)


__all__ = [
    'GEOMETRY_LINKAGE_CSV_NAME',
    'GEOMETRY_LINKAGE_NPZ_NAME',
    'GEOMETRY_LINKAGE_QC_JSON_NAME',
    'GeometryLinkageArtifactMetadata',
    'GeometryLinkageArtifactPaths',
    'build_geometry_linkage_qc_payload',
    'build_geometry_linkage_solution_arrays',
    'summarize_finite_values',
    'write_geometry_linkage_artifacts',
]
