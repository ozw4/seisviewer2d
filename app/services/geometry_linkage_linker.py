"""Deterministic endpoint linker for static linkage geometry tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.services.geometry_linkage_tables import EndpointGeometryTables

EndpointKind = Literal['source', 'receiver']
LinkageMode = Literal['none', 'auto_threshold']
LinkageMethod = Literal[
    'none_mode_receiver_independent',
    'none_mode_source_independent',
    'receiver_seed',
    'receiver_anchor',
    'source_fallback',
    'source_independent',
]


@dataclass(frozen=True)
class GeometryLinkageOptions:
    mode: LinkageMode
    threshold_m: float | None = None
    receiver_location_interval_m: float | None = None
    prefer_receiver_anchor: bool = True


@dataclass(frozen=True)
class EndpointLinkageRecord:
    endpoint_kind: EndpointKind
    endpoint_id: int
    x_m: float
    y_m: float
    node_id: int
    linked_to_kind: EndpointKind | None
    linked_to_id: int | None
    distance_m: float | None
    method: LinkageMethod


@dataclass(frozen=True)
class GeometryLinkageResult:
    mode: LinkageMode
    threshold_m: float | None
    receiver_location_interval_m: float | None
    prefer_receiver_anchor: bool
    n_traces: int
    n_source_endpoints: int
    n_receiver_endpoints: int
    n_nodes: int
    source_node_id_by_endpoint: np.ndarray
    receiver_node_id_by_endpoint: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    records: tuple[EndpointLinkageRecord, ...]
    n_receiver_anchor_links: int
    n_source_fallback_links: int
    n_independent_source_nodes: int


@dataclass(frozen=True)
class _ValidatedTables:
    n_traces: int
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    source_endpoint_id_sorted: np.ndarray
    receiver_endpoint_id_sorted: np.ndarray


@dataclass(frozen=True)
class _ValidatedOptions:
    mode: LinkageMode
    threshold_m: float | None
    receiver_location_interval_m: float | None
    prefer_receiver_anchor: bool


def build_geometry_linkage(
    tables: EndpointGeometryTables,
    options: GeometryLinkageOptions,
) -> GeometryLinkageResult:
    """Assign endpoint node ids from source/receiver endpoint geometry."""
    validated_tables = _validate_tables(tables)
    validated_options = _validate_options(options)

    if validated_options.mode == 'none':
        return _build_none_linkage(validated_tables, validated_options)
    return _build_auto_threshold_linkage(validated_tables, validated_options)


def _build_none_linkage(
    tables: _ValidatedTables,
    options: _ValidatedOptions,
) -> GeometryLinkageResult:
    n_source = int(tables.source_x_m.shape[0])
    n_receiver = int(tables.receiver_x_m.shape[0])
    receiver_node_id_by_endpoint = np.arange(n_receiver, dtype=np.int64)
    source_node_id_by_endpoint = n_receiver + np.arange(n_source, dtype=np.int64)

    records = _build_records(
        mode=options.mode,
        source_x_m=tables.source_x_m,
        source_y_m=tables.source_y_m,
        receiver_x_m=tables.receiver_x_m,
        receiver_y_m=tables.receiver_y_m,
        source_node_id_by_endpoint=source_node_id_by_endpoint,
        receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
        source_methods=[
            'none_mode_source_independent' for _ in range(n_source)
        ],
        source_linked_to_kind=[None for _ in range(n_source)],
        source_linked_to_id=[None for _ in range(n_source)],
        source_distance_m=[None for _ in range(n_source)],
    )

    return _result_from_endpoint_mappings(
        tables=tables,
        options=options,
        source_node_id_by_endpoint=source_node_id_by_endpoint,
        receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
        records=records,
        n_receiver_anchor_links=0,
        n_source_fallback_links=0,
        n_independent_source_nodes=n_source,
    )


def _build_auto_threshold_linkage(
    tables: _ValidatedTables,
    options: _ValidatedOptions,
) -> GeometryLinkageResult:
    if options.threshold_m is None:
        msg = 'threshold_m is required when mode is auto_threshold'
        raise ValueError(msg)

    n_source = int(tables.source_x_m.shape[0])
    n_receiver = int(tables.receiver_x_m.shape[0])
    receiver_node_id_by_endpoint = np.arange(n_receiver, dtype=np.int64)
    source_node_id_by_endpoint = np.full(n_source, -1, dtype=np.int64)

    source_methods: list[LinkageMethod] = [
        'source_independent' for _ in range(n_source)
    ]
    source_linked_to_kind: list[EndpointKind | None] = [
        None for _ in range(n_source)
    ]
    source_linked_to_id: list[int | None] = [None for _ in range(n_source)]
    source_distance_m: list[float | None] = [None for _ in range(n_source)]

    nearest_receiver_id, nearest_receiver_distance = _nearest_receiver_for_sources(
        source_x_m=tables.source_x_m,
        source_y_m=tables.source_y_m,
        receiver_x_m=tables.receiver_x_m,
        receiver_y_m=tables.receiver_y_m,
        threshold_m=options.threshold_m,
    )
    for source_id, receiver_id in enumerate(nearest_receiver_id):
        if receiver_id < 0:
            continue
        source_node_id_by_endpoint[source_id] = int(receiver_id)
        source_methods[source_id] = 'receiver_anchor'
        source_linked_to_kind[source_id] = 'receiver'
        source_linked_to_id[source_id] = int(receiver_id)
        source_distance_m[source_id] = float(nearest_receiver_distance[source_id])

    fallback_source_ids = np.flatnonzero(source_node_id_by_endpoint < 0).astype(
        np.int64,
        copy=False,
    )
    nearest_source_id, nearest_source_distance, fallback_edges = (
        _nearest_source_for_fallback_sources(
            source_x_m=tables.source_x_m,
            source_y_m=tables.source_y_m,
            fallback_source_ids=fallback_source_ids,
            threshold_m=options.threshold_m,
        )
    )
    for local_index, source_id_value in enumerate(fallback_source_ids):
        source_id = int(source_id_value)
        linked_source_id = int(nearest_source_id[local_index])
        if linked_source_id < 0:
            continue
        source_methods[source_id] = 'source_fallback'
        source_linked_to_kind[source_id] = 'source'
        source_linked_to_id[source_id] = linked_source_id
        source_distance_m[source_id] = float(nearest_source_distance[local_index])

    components = _connected_components_from_edges(
        node_ids=fallback_source_ids,
        edges=fallback_edges,
    )
    for component_index, component in enumerate(components):
        node_id = n_receiver + component_index
        for source_id in component:
            source_node_id_by_endpoint[source_id] = node_id

    records = _build_records(
        mode=options.mode,
        source_x_m=tables.source_x_m,
        source_y_m=tables.source_y_m,
        receiver_x_m=tables.receiver_x_m,
        receiver_y_m=tables.receiver_y_m,
        source_node_id_by_endpoint=source_node_id_by_endpoint,
        receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
        source_methods=source_methods,
        source_linked_to_kind=source_linked_to_kind,
        source_linked_to_id=source_linked_to_id,
        source_distance_m=source_distance_m,
    )

    return _result_from_endpoint_mappings(
        tables=tables,
        options=options,
        source_node_id_by_endpoint=source_node_id_by_endpoint,
        receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
        records=records,
        n_receiver_anchor_links=sum(
            method == 'receiver_anchor' for method in source_methods
        ),
        n_source_fallback_links=sum(
            method == 'source_fallback' for method in source_methods
        ),
        n_independent_source_nodes=sum(
            method == 'source_independent' for method in source_methods
        ),
    )


def _result_from_endpoint_mappings(
    *,
    tables: _ValidatedTables,
    options: _ValidatedOptions,
    source_node_id_by_endpoint: np.ndarray,
    receiver_node_id_by_endpoint: np.ndarray,
    records: tuple[EndpointLinkageRecord, ...],
    n_receiver_anchor_links: int,
    n_source_fallback_links: int,
    n_independent_source_nodes: int,
) -> GeometryLinkageResult:
    source_node_id_by_endpoint = np.ascontiguousarray(
        source_node_id_by_endpoint,
        dtype=np.int64,
    )
    receiver_node_id_by_endpoint = np.ascontiguousarray(
        receiver_node_id_by_endpoint,
        dtype=np.int64,
    )
    source_node_id_sorted = np.ascontiguousarray(
        source_node_id_by_endpoint[tables.source_endpoint_id_sorted],
        dtype=np.int64,
    )
    receiver_node_id_sorted = np.ascontiguousarray(
        receiver_node_id_by_endpoint[tables.receiver_endpoint_id_sorted],
        dtype=np.int64,
    )

    n_nodes = max(
        _max_or_minus_one(source_node_id_by_endpoint),
        _max_or_minus_one(receiver_node_id_by_endpoint),
    ) + 1

    return GeometryLinkageResult(
        mode=options.mode,
        threshold_m=options.threshold_m,
        receiver_location_interval_m=options.receiver_location_interval_m,
        prefer_receiver_anchor=options.prefer_receiver_anchor,
        n_traces=tables.n_traces,
        n_source_endpoints=int(source_node_id_by_endpoint.shape[0]),
        n_receiver_endpoints=int(receiver_node_id_by_endpoint.shape[0]),
        n_nodes=n_nodes,
        source_node_id_by_endpoint=source_node_id_by_endpoint,
        receiver_node_id_by_endpoint=receiver_node_id_by_endpoint,
        source_node_id_sorted=source_node_id_sorted,
        receiver_node_id_sorted=receiver_node_id_sorted,
        records=records,
        n_receiver_anchor_links=int(n_receiver_anchor_links),
        n_source_fallback_links=int(n_source_fallback_links),
        n_independent_source_nodes=int(n_independent_source_nodes),
    )


def _build_records(
    *,
    mode: LinkageMode,
    source_x_m: np.ndarray,
    source_y_m: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
    source_node_id_by_endpoint: np.ndarray,
    receiver_node_id_by_endpoint: np.ndarray,
    source_methods: list[LinkageMethod],
    source_linked_to_kind: list[EndpointKind | None],
    source_linked_to_id: list[int | None],
    source_distance_m: list[float | None],
) -> tuple[EndpointLinkageRecord, ...]:
    records: list[EndpointLinkageRecord] = []
    receiver_method: LinkageMethod = (
        'none_mode_receiver_independent' if mode == 'none' else 'receiver_seed'
    )
    for receiver_id in range(int(receiver_x_m.shape[0])):
        records.append(
            EndpointLinkageRecord(
                endpoint_kind='receiver',
                endpoint_id=receiver_id,
                x_m=float(receiver_x_m[receiver_id]),
                y_m=float(receiver_y_m[receiver_id]),
                node_id=int(receiver_node_id_by_endpoint[receiver_id]),
                linked_to_kind=None,
                linked_to_id=None,
                distance_m=None,
                method=receiver_method,
            )
        )

    for source_id in range(int(source_x_m.shape[0])):
        records.append(
            EndpointLinkageRecord(
                endpoint_kind='source',
                endpoint_id=source_id,
                x_m=float(source_x_m[source_id]),
                y_m=float(source_y_m[source_id]),
                node_id=int(source_node_id_by_endpoint[source_id]),
                linked_to_kind=source_linked_to_kind[source_id],
                linked_to_id=source_linked_to_id[source_id],
                distance_m=source_distance_m[source_id],
                method=source_methods[source_id],
            )
        )
    return tuple(records)


def _nearest_receiver_for_sources(
    *,
    source_x_m: np.ndarray,
    source_y_m: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
    threshold_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    nearest_receiver_id = np.full(source_x_m.shape, -1, dtype=np.int64)
    nearest_receiver_distance = np.full(source_x_m.shape, np.nan, dtype=np.float64)
    threshold_squared = threshold_m * threshold_m

    for source_id, (source_x, source_y) in enumerate(zip(source_x_m, source_y_m)):
        dx = receiver_x_m - source_x
        dy = receiver_y_m - source_y
        distance_squared = (dx * dx) + (dy * dy)
        candidate_indices = np.flatnonzero(distance_squared <= threshold_squared)
        if candidate_indices.size == 0:
            continue
        best_index = int(
            candidate_indices[np.argmin(distance_squared[candidate_indices])]
        )
        nearest_receiver_id[source_id] = best_index
        nearest_receiver_distance[source_id] = float(
            np.hypot(dx[best_index], dy[best_index])
        )

    return nearest_receiver_id, nearest_receiver_distance


def _nearest_source_for_fallback_sources(
    *,
    source_x_m: np.ndarray,
    source_y_m: np.ndarray,
    fallback_source_ids: np.ndarray,
    threshold_m: float,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    nearest_source_id = np.full(fallback_source_ids.shape, -1, dtype=np.int64)
    nearest_source_distance = np.full(
        fallback_source_ids.shape,
        np.nan,
        dtype=np.float64,
    )
    edges: list[tuple[int, int]] = []
    threshold_squared = threshold_m * threshold_m

    fallback_x_m = source_x_m[fallback_source_ids]
    fallback_y_m = source_y_m[fallback_source_ids]
    for local_index, source_id_value in enumerate(fallback_source_ids):
        source_id = int(source_id_value)
        dx = fallback_x_m - source_x_m[source_id]
        dy = fallback_y_m - source_y_m[source_id]
        distance_squared = (dx * dx) + (dy * dy)
        distance_squared[local_index] = np.inf
        candidate_indices = np.flatnonzero(distance_squared <= threshold_squared)
        if candidate_indices.size == 0:
            continue
        best_local_index = int(
            candidate_indices[np.argmin(distance_squared[candidate_indices])]
        )
        linked_source_id = int(fallback_source_ids[best_local_index])
        nearest_source_id[local_index] = linked_source_id
        nearest_source_distance[local_index] = float(
            np.hypot(dx[best_local_index], dy[best_local_index])
        )
        edges.append((source_id, linked_source_id))

    return nearest_source_id, nearest_source_distance, tuple(edges)


def _connected_components_from_edges(
    *,
    node_ids: np.ndarray,
    edges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], ...]:
    parent = {int(node_id): int(node_id) for node_id in node_ids}

    def find(node_id: int) -> int:
        root = parent[node_id]
        while root != parent[root]:
            root = parent[root]
        while node_id != root:
            next_node_id = parent[node_id]
            parent[node_id] = root
            node_id = next_node_id
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for left, right in edges:
        union(left, right)

    groups: dict[int, list[int]] = {}
    for node_id_value in node_ids:
        node_id = int(node_id_value)
        groups.setdefault(find(node_id), []).append(node_id)

    components = [tuple(sorted(group)) for group in groups.values()]
    return tuple(sorted(components, key=lambda component: component[0]))


def _validate_tables(tables: EndpointGeometryTables) -> _ValidatedTables:
    n_traces = int(tables.n_traces)
    if n_traces <= 0:
        msg = 'endpoint geometry tables must contain at least one trace'
        raise ValueError(msg)

    source_x_m, source_y_m, n_source_endpoints = _validate_endpoint_table(
        tables.source_endpoints,
        expected_kind='source',
        name='source_endpoints',
    )
    receiver_x_m, receiver_y_m, n_receiver_endpoints = _validate_endpoint_table(
        tables.receiver_endpoints,
        expected_kind='receiver',
        name='receiver_endpoints',
    )
    source_endpoint_id_sorted = _validate_sorted_endpoint_mapping(
        tables.source_endpoint_id_sorted,
        name='source_endpoint_id_sorted',
        n_traces=n_traces,
        n_endpoints=n_source_endpoints,
    )
    receiver_endpoint_id_sorted = _validate_sorted_endpoint_mapping(
        tables.receiver_endpoint_id_sorted,
        name='receiver_endpoint_id_sorted',
        n_traces=n_traces,
        n_endpoints=n_receiver_endpoints,
    )

    return _ValidatedTables(
        n_traces=n_traces,
        source_x_m=source_x_m,
        source_y_m=source_y_m,
        receiver_x_m=receiver_x_m,
        receiver_y_m=receiver_y_m,
        source_endpoint_id_sorted=source_endpoint_id_sorted,
        receiver_endpoint_id_sorted=receiver_endpoint_id_sorted,
    )


def _validate_endpoint_table(
    table: object,
    *,
    expected_kind: EndpointKind,
    name: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    if getattr(table, 'endpoint_kind', None) != expected_kind:
        msg = f'{name}.endpoint_kind must be {expected_kind!r}'
        raise ValueError(msg)

    endpoint_id = _coerce_1d_integer_array(
        getattr(table, 'endpoint_id'),
        name=f'{name}.endpoint_id',
    )
    expected_endpoint_id = np.arange(endpoint_id.shape[0], dtype=np.int64)
    if not np.array_equal(endpoint_id, expected_endpoint_id):
        msg = f'{name}.endpoint_id must be 0-based contiguous'
        raise ValueError(msg)

    x_m = _coerce_1d_finite_float_array(getattr(table, 'x_m'), name=f'{name}.x_m')
    y_m = _coerce_1d_finite_float_array(getattr(table, 'y_m'), name=f'{name}.y_m')
    if x_m.shape != endpoint_id.shape:
        msg = f'{name}.x_m shape mismatch: expected {endpoint_id.shape}, got {x_m.shape}'
        raise ValueError(msg)
    if y_m.shape != endpoint_id.shape:
        msg = f'{name}.y_m shape mismatch: expected {endpoint_id.shape}, got {y_m.shape}'
        raise ValueError(msg)

    return x_m, y_m, int(endpoint_id.shape[0])


def _validate_sorted_endpoint_mapping(
    values: np.ndarray,
    *,
    name: str,
    n_traces: int,
    n_endpoints: int,
) -> np.ndarray:
    arr = _coerce_1d_integer_array(values, name=name)
    expected_shape = (n_traces,)
    if arr.shape != expected_shape:
        msg = f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        raise ValueError(msg)
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= n_endpoints):
        msg = f'{name} values must be in range 0..{n_endpoints - 1}'
        raise ValueError(msg)
    return arr


def _validate_options(options: GeometryLinkageOptions) -> _ValidatedOptions:
    if options.mode not in ('none', 'auto_threshold'):
        msg = "mode must be 'none' or 'auto_threshold'"
        raise ValueError(msg)

    receiver_location_interval_m = _validate_optional_positive_float(
        options.receiver_location_interval_m,
        name='receiver_location_interval_m',
    )
    threshold_m = _validate_threshold(options)
    if options.mode == 'auto_threshold' and not options.prefer_receiver_anchor:
        msg = (
            'prefer_receiver_anchor=False is not supported for auto_threshold '
            'linkage'
        )
        raise ValueError(msg)

    return _ValidatedOptions(
        mode=options.mode,
        threshold_m=threshold_m,
        receiver_location_interval_m=receiver_location_interval_m,
        prefer_receiver_anchor=options.prefer_receiver_anchor,
    )


def _validate_threshold(options: GeometryLinkageOptions) -> float | None:
    if options.mode == 'none':
        if options.threshold_m is not None:
            msg = 'threshold_m must be None when mode is none'
            raise ValueError(msg)
        return None

    if options.threshold_m is None:
        msg = 'threshold_m is required when mode is auto_threshold'
        raise ValueError(msg)
    return _validate_positive_float(options.threshold_m, name='threshold_m')


def _validate_optional_positive_float(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    return _validate_positive_float(value, name=name)


def _validate_positive_float(value: float, *, name: str) -> float:
    try:
        value_float = float(value)
    except (TypeError, ValueError) as exc:
        msg = f'{name} must be finite and > 0'
        raise ValueError(msg) from exc
    if not np.isfinite(value_float) or value_float <= 0:
        msg = f'{name} must be finite and > 0'
        raise ValueError(msg)
    return value_float


def _coerce_1d_integer_array(values: object, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        msg = f'{name} must be a 1D array'
        raise ValueError(msg)
    if not np.issubdtype(arr.dtype, np.integer):
        msg = f'{name} must have an integer dtype'
        raise ValueError(msg)
    return np.asarray(arr, dtype=np.int64)


def _coerce_1d_finite_float_array(values: object, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        msg = f'{name} must be a 1D array'
        raise ValueError(msg)
    if not np.issubdtype(arr.dtype, np.floating):
        msg = f'{name} must have a floating dtype'
        raise ValueError(msg)
    arr_f64 = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr_f64)):
        msg = f'{name} must contain only finite values'
        raise ValueError(msg)
    return arr_f64


def _max_or_minus_one(values: np.ndarray) -> int:
    if values.size == 0:
        return -1
    return int(values.max())


__all__ = [
    'EndpointKind',
    'EndpointLinkageRecord',
    'GeometryLinkageOptions',
    'GeometryLinkageResult',
    'LinkageMethod',
    'LinkageMode',
    'build_geometry_linkage',
]
