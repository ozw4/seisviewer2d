from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from app.services.geometry_linkage_linker import (
    GeometryLinkageOptions,
    build_geometry_linkage,
)
from app.services.geometry_linkage_tables import (
    EndpointGeometryTable,
    EndpointGeometryTables,
)


def _endpoint_table(
    endpoint_kind: str,
    xy: list[tuple[float, float]],
) -> EndpointGeometryTable:
    coordinates = np.asarray(xy, dtype=np.float64)
    endpoint_count = int(coordinates.shape[0])
    return EndpointGeometryTable(
        endpoint_kind=endpoint_kind,  # type: ignore[arg-type]
        endpoint_id=np.arange(endpoint_count, dtype=np.int64),
        x_m=np.ascontiguousarray(coordinates[:, 0], dtype=np.float64),
        y_m=np.ascontiguousarray(coordinates[:, 1], dtype=np.float64),
        first_sorted_trace_index=np.arange(endpoint_count, dtype=np.int64),
        trace_count=np.ones(endpoint_count, dtype=np.int64),
    )


def _tables(
    *,
    source_xy: list[tuple[float, float]],
    receiver_xy: list[tuple[float, float]],
    source_endpoint_id_sorted: list[int] | None = None,
    receiver_endpoint_id_sorted: list[int] | None = None,
) -> EndpointGeometryTables:
    source_endpoints = _endpoint_table('source', source_xy)
    receiver_endpoints = _endpoint_table('receiver', receiver_xy)
    if source_endpoint_id_sorted is not None:
        n_traces = len(source_endpoint_id_sorted)
    elif receiver_endpoint_id_sorted is not None:
        n_traces = len(receiver_endpoint_id_sorted)
    else:
        n_traces = len(source_xy)
    if source_endpoint_id_sorted is None:
        source_endpoint_id_sorted = [
            trace_index % len(source_xy) for trace_index in range(n_traces)
        ]
    if receiver_endpoint_id_sorted is None:
        receiver_endpoint_id_sorted = [
            trace_index % len(receiver_xy) for trace_index in range(n_traces)
        ]

    source_sorted = np.asarray(source_endpoint_id_sorted, dtype=np.int64)
    receiver_sorted = np.asarray(receiver_endpoint_id_sorted, dtype=np.int64)
    return EndpointGeometryTables(
        n_traces=n_traces,
        source_x_m_sorted=np.zeros(n_traces, dtype=np.float64),
        source_y_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_x_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_y_m_sorted=np.zeros(n_traces, dtype=np.float64),
        coordinate_scalar_sorted=np.ones(n_traces, dtype=np.int64),
        source_endpoint_id_sorted=source_sorted,
        receiver_endpoint_id_sorted=receiver_sorted,
        source_endpoints=source_endpoints,
        receiver_endpoints=receiver_endpoints,
        scalar_zero_count=0,
    )


def _auto(threshold_m: float) -> GeometryLinkageOptions:
    return GeometryLinkageOptions(mode='auto_threshold', threshold_m=threshold_m)


def test_build_geometry_linkage_none_mode_assigns_disjoint_nodes() -> None:
    tables = _tables(
        source_xy=[(0.0, 0.0), (1.0, 0.0)],
        receiver_xy=[(10.0, 0.0), (11.0, 0.0)],
        source_endpoint_id_sorted=[1, 0, 1],
        receiver_endpoint_id_sorted=[0, 1, 0],
    )

    result = build_geometry_linkage(tables, GeometryLinkageOptions(mode='none'))

    np.testing.assert_array_equal(result.receiver_node_id_by_endpoint, [0, 1])
    np.testing.assert_array_equal(result.source_node_id_by_endpoint, [2, 3])
    np.testing.assert_array_equal(result.source_node_id_sorted, [3, 2, 3])
    np.testing.assert_array_equal(result.receiver_node_id_sorted, [0, 1, 0])
    assert result.n_nodes == 4
    assert result.n_receiver_anchor_links == 0
    assert result.n_source_fallback_links == 0
    assert result.n_independent_source_nodes == 2
    assert [record.method for record in result.records] == [
        'none_mode_receiver_independent',
        'none_mode_receiver_independent',
        'none_mode_source_independent',
        'none_mode_source_independent',
    ]
    assert all(record.linked_to_kind is None for record in result.records)
    assert all(record.linked_to_id is None for record in result.records)
    assert all(record.distance_m is None for record in result.records)


def test_build_geometry_linkage_none_mode_rejects_threshold() -> None:
    with pytest.raises(ValueError, match='threshold_m.*none'):
        build_geometry_linkage(
            _tables(source_xy=[(0.0, 0.0)], receiver_xy=[(1.0, 0.0)]),
            GeometryLinkageOptions(mode='none', threshold_m=1.0),
        )


def test_build_geometry_linkage_auto_threshold_links_source_to_nearest_receiver() -> None:
    result = build_geometry_linkage(
        _tables(
            source_xy=[(9.0, 0.0), (2.0, 0.0)],
            receiver_xy=[(0.0, 0.0), (10.0, 0.0)],
        ),
        _auto(3.0),
    )

    np.testing.assert_array_equal(result.source_node_id_by_endpoint, [1, 0])
    source_records = result.records[2:]
    assert [(record.linked_to_id, record.distance_m) for record in source_records] == [
        (1, 1.0),
        (0, 2.0),
    ]


def test_build_geometry_linkage_auto_threshold_uses_inclusive_threshold() -> None:
    result = build_geometry_linkage(
        _tables(source_xy=[(3.0, 4.0)], receiver_xy=[(0.0, 0.0)]),
        _auto(5.0),
    )

    assert result.source_node_id_by_endpoint[0] == 0
    assert result.records[1].method == 'receiver_anchor'
    assert result.records[1].distance_m == 5.0


def test_build_geometry_linkage_auto_threshold_receiver_tie_breaks_by_endpoint_id() -> None:
    result = build_geometry_linkage(
        _tables(
            source_xy=[(0.0, 0.0)],
            receiver_xy=[(0.0, -1.0), (0.0, 1.0)],
        ),
        _auto(1.0),
    )

    source_record = result.records[2]
    assert source_record.linked_to_kind == 'receiver'
    assert source_record.linked_to_id == 0
    assert source_record.node_id == 0


def test_build_geometry_linkage_auto_threshold_prefers_receiver_anchor_over_nearer_source() -> None:
    result = build_geometry_linkage(
        _tables(
            source_xy=[(9.0, 0.0), (9.1, 0.0)],
            receiver_xy=[(10.0, 0.0)],
        ),
        _auto(2.0),
    )

    np.testing.assert_array_equal(result.source_node_id_by_endpoint, [0, 0])
    assert [record.method for record in result.records[1:]] == [
        'receiver_anchor',
        'receiver_anchor',
    ]
    assert result.n_receiver_anchor_links == 2
    assert result.n_source_fallback_links == 0


def test_build_geometry_linkage_auto_threshold_excludes_receiver_anchored_sources_from_fallback() -> None:
    result = build_geometry_linkage(
        _tables(
            source_xy=[(1.0, 0.0), (1.5, 0.0)],
            receiver_xy=[(0.0, 0.0)],
        ),
        _auto(1.0),
    )

    np.testing.assert_array_equal(result.source_node_id_by_endpoint, [0, 1])
    assert result.records[1].method == 'receiver_anchor'
    assert result.records[2].method == 'source_independent'
    assert result.records[2].linked_to_id is None
    assert result.n_source_fallback_links == 0


def test_build_geometry_linkage_auto_threshold_source_fallback_links_nearest_source() -> None:
    result = build_geometry_linkage(
        _tables(
            source_xy=[(0.0, 0.0), (2.0, 0.0), (10.0, 0.0)],
            receiver_xy=[(100.0, 0.0)],
        ),
        _auto(3.0),
    )

    np.testing.assert_array_equal(result.source_node_id_by_endpoint, [1, 1, 2])
    assert result.records[1].linked_to_id == 1
    assert result.records[2].linked_to_id == 0
    assert result.records[1].distance_m == 2.0
    assert result.records[3].method == 'source_independent'


def test_build_geometry_linkage_auto_threshold_source_fallback_tie_breaks_by_endpoint_id() -> None:
    result = build_geometry_linkage(
        _tables(
            source_xy=[(0.0, -1.0), (0.0, 0.0), (0.0, 1.0)],
            receiver_xy=[(100.0, 0.0)],
        ),
        _auto(2.0),
    )

    tied_source_record = result.records[2]
    assert tied_source_record.method == 'source_fallback'
    assert tied_source_record.linked_to_kind == 'source'
    assert tied_source_record.linked_to_id == 0


def test_build_geometry_linkage_auto_threshold_source_fallback_components_share_node() -> None:
    result = build_geometry_linkage(
        _tables(
            source_xy=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            receiver_xy=[(100.0, 0.0)],
        ),
        _auto(1.1),
    )

    np.testing.assert_array_equal(result.source_node_id_by_endpoint, [1, 1, 1])
    assert result.n_nodes == 2


def test_build_geometry_linkage_auto_threshold_source_independent_when_no_candidates() -> None:
    result = build_geometry_linkage(
        _tables(
            source_xy=[(0.0, 0.0), (10.0, 0.0)],
            receiver_xy=[(100.0, 0.0)],
        ),
        _auto(1.0),
    )

    np.testing.assert_array_equal(result.source_node_id_by_endpoint, [1, 2])
    assert [record.method for record in result.records[1:]] == [
        'source_independent',
        'source_independent',
    ]
    assert result.n_independent_source_nodes == 2


def test_build_geometry_linkage_auto_threshold_source_only_node_ids_follow_min_source_endpoint_id() -> None:
    result = build_geometry_linkage(
        _tables(
            source_xy=[
                (20.0, 0.0),
                (21.0, 0.0),
                (10.0, 0.0),
                (11.0, 0.0),
            ],
            receiver_xy=[(100.0, 0.0)],
        ),
        _auto(1.1),
    )

    np.testing.assert_array_equal(result.source_node_id_by_endpoint, [1, 1, 2, 2])


def test_build_geometry_linkage_returns_sorted_trace_node_mappings() -> None:
    result = build_geometry_linkage(
        _tables(
            source_xy=[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)],
            receiver_xy=[(100.0, 0.0), (200.0, 0.0)],
            source_endpoint_id_sorted=[2, 0, 1],
            receiver_endpoint_id_sorted=[1, 0, 1],
        ),
        GeometryLinkageOptions(mode='none'),
    )

    np.testing.assert_array_equal(result.source_node_id_by_endpoint, [2, 3, 4])
    np.testing.assert_array_equal(result.receiver_node_id_by_endpoint, [0, 1])
    np.testing.assert_array_equal(result.source_node_id_sorted, [4, 2, 3])
    np.testing.assert_array_equal(result.receiver_node_id_sorted, [1, 0, 1])


def test_build_geometry_linkage_records_are_deterministically_ordered() -> None:
    result = build_geometry_linkage(
        _tables(
            source_xy=[(10.0, 0.0), (0.0, 0.0)],
            receiver_xy=[(30.0, 0.0), (20.0, 0.0)],
        ),
        GeometryLinkageOptions(mode='none'),
    )

    assert [(record.endpoint_kind, record.endpoint_id) for record in result.records] == [
        ('receiver', 0),
        ('receiver', 1),
        ('source', 0),
        ('source', 1),
    ]


def test_build_geometry_linkage_auto_threshold_receiver_records_use_receiver_seed_method() -> None:
    result = build_geometry_linkage(
        _tables(
            source_xy=[(10.0, 0.0)],
            receiver_xy=[(0.0, 0.0), (20.0, 0.0)],
        ),
        _auto(1.0),
    )

    assert [record.method for record in result.records[:2]] == [
        'receiver_seed',
        'receiver_seed',
    ]


def test_build_geometry_linkage_counts_summary_values() -> None:
    result = build_geometry_linkage(
        _tables(
            source_xy=[
                (0.5, 0.0),
                (10.0, 0.0),
                (11.0, 0.0),
                (20.0, 0.0),
            ],
            receiver_xy=[(0.0, 0.0)],
        ),
        _auto(1.1),
    )

    assert result.n_receiver_anchor_links == 1
    assert result.n_source_fallback_links == 2
    assert result.n_independent_source_nodes == 1
    assert result.n_nodes == 3


def test_build_geometry_linkage_rejects_missing_threshold_for_auto_threshold() -> None:
    with pytest.raises(ValueError, match='threshold_m.*required'):
        build_geometry_linkage(
            _tables(source_xy=[(0.0, 0.0)], receiver_xy=[(1.0, 0.0)]),
            GeometryLinkageOptions(mode='auto_threshold'),
        )


@pytest.mark.parametrize('threshold_m', [0.0, -1.0])
def test_build_geometry_linkage_rejects_non_positive_threshold(
    threshold_m: float,
) -> None:
    with pytest.raises(ValueError, match='threshold_m.*> 0'):
        build_geometry_linkage(
            _tables(source_xy=[(0.0, 0.0)], receiver_xy=[(1.0, 0.0)]),
            _auto(threshold_m),
        )


@pytest.mark.parametrize('threshold_m', [np.nan, np.inf, -np.inf])
def test_build_geometry_linkage_rejects_non_finite_threshold(
    threshold_m: float,
) -> None:
    with pytest.raises(ValueError, match='threshold_m.*finite'):
        build_geometry_linkage(
            _tables(source_xy=[(0.0, 0.0)], receiver_xy=[(1.0, 0.0)]),
            _auto(threshold_m),
        )


def test_build_geometry_linkage_rejects_unsupported_prefer_receiver_anchor_false() -> None:
    with pytest.raises(ValueError, match='prefer_receiver_anchor=False'):
        build_geometry_linkage(
            _tables(source_xy=[(0.0, 0.0)], receiver_xy=[(1.0, 0.0)]),
            GeometryLinkageOptions(
                mode='auto_threshold',
                threshold_m=1.0,
                prefer_receiver_anchor=False,
            ),
        )


def test_build_geometry_linkage_rejects_invalid_endpoint_ids() -> None:
    tables = _tables(
        source_xy=[(0.0, 0.0), (1.0, 0.0)],
        receiver_xy=[(10.0, 0.0)],
    )
    bad_source_endpoints = replace(
        tables.source_endpoints,
        endpoint_id=np.array([0, 2], dtype=np.int64),
    )

    with pytest.raises(ValueError, match='source_endpoints.endpoint_id'):
        build_geometry_linkage(
            replace(tables, source_endpoints=bad_source_endpoints),
            GeometryLinkageOptions(mode='none'),
        )


def test_build_geometry_linkage_rejects_sorted_mapping_out_of_range() -> None:
    with pytest.raises(ValueError, match='source_endpoint_id_sorted.*range'):
        build_geometry_linkage(
            _tables(
                source_xy=[(0.0, 0.0), (1.0, 0.0)],
                receiver_xy=[(10.0, 0.0), (11.0, 0.0)],
                source_endpoint_id_sorted=[0, 2],
                receiver_endpoint_id_sorted=[0, 1],
            ),
            GeometryLinkageOptions(mode='none'),
        )


def test_build_geometry_linkage_rejects_non_finite_endpoint_coordinate() -> None:
    tables = _tables(source_xy=[(0.0, 0.0)], receiver_xy=[(1.0, 0.0)])
    bad_receiver_endpoints = replace(
        tables.receiver_endpoints,
        x_m=np.array([np.nan], dtype=np.float64),
    )

    with pytest.raises(ValueError, match='receiver_endpoints.x_m.*finite'):
        build_geometry_linkage(
            replace(tables, receiver_endpoints=bad_receiver_endpoints),
            GeometryLinkageOptions(mode='none'),
        )
