from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import app.services.geometry_linkage_artifacts as artifacts
from app.services.geometry_linkage_artifacts import (
    GeometryLinkageArtifactMetadata,
    build_geometry_linkage_qc_payload,
    build_geometry_linkage_solution_arrays,
    write_geometry_linkage_artifacts,
)
from app.services.geometry_linkage_linker import (
    GeometryLinkageOptions,
    GeometryLinkageResult,
    build_geometry_linkage,
)
from app.services.geometry_linkage_tables import (
    EndpointGeometryTable,
    EndpointGeometryTables,
)

_REQUIRED_SCALAR_FIELDS = {
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
}

_TRACE_LEVEL_FIELDS = {
    'source_x_m_sorted',
    'source_y_m_sorted',
    'receiver_x_m_sorted',
    'receiver_y_m_sorted',
    'coordinate_scalar_sorted',
    'source_endpoint_id_sorted',
    'receiver_endpoint_id_sorted',
    'source_node_id_sorted',
    'receiver_node_id_sorted',
}

_SOURCE_ENDPOINT_FIELDS = {
    'source_endpoint_id',
    'source_endpoint_x_m',
    'source_endpoint_y_m',
    'source_endpoint_first_sorted_trace_index',
    'source_endpoint_trace_count',
    'source_node_id_by_endpoint',
}

_RECEIVER_ENDPOINT_FIELDS = {
    'receiver_endpoint_id',
    'receiver_endpoint_x_m',
    'receiver_endpoint_y_m',
    'receiver_endpoint_first_sorted_trace_index',
    'receiver_endpoint_trace_count',
    'receiver_node_id_by_endpoint',
}

_RECORD_FIELDS = {
    'record_endpoint_kind',
    'record_endpoint_id',
    'record_x_m',
    'record_y_m',
    'record_node_id',
    'record_linked_to_kind',
    'record_linked_to_id',
    'record_distance_m',
    'record_method',
}

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


def _endpoint_table(
    endpoint_kind: str,
    xy: list[tuple[float, float]],
    endpoint_id: np.ndarray | None = None,
) -> EndpointGeometryTable:
    coordinates = np.asarray(xy, dtype=np.float64)
    n_endpoints = int(coordinates.shape[0])
    if endpoint_id is None:
        endpoint_id = np.arange(n_endpoints, dtype=np.int64)
    return EndpointGeometryTable(
        endpoint_kind=endpoint_kind,  # type: ignore[arg-type]
        endpoint_id=np.ascontiguousarray(endpoint_id, dtype=np.int64),
        x_m=np.ascontiguousarray(coordinates[:, 0], dtype=np.float64),
        y_m=np.ascontiguousarray(coordinates[:, 1], dtype=np.float64),
        first_sorted_trace_index=np.arange(n_endpoints, dtype=np.int64),
        trace_count=np.ones(n_endpoints, dtype=np.int64),
    )


def _tables(
    *,
    source_xy: list[tuple[float, float]],
    receiver_xy: list[tuple[float, float]],
    source_endpoint_id_sorted: list[int] | None = None,
    receiver_endpoint_id_sorted: list[int] | None = None,
    coordinate_scalar_sorted: list[int] | None = None,
) -> EndpointGeometryTables:
    source_endpoints = _endpoint_table('source', source_xy)
    receiver_endpoints = _endpoint_table('receiver', receiver_xy)
    if source_endpoint_id_sorted is None:
        source_endpoint_id_sorted = list(range(len(source_xy)))
    if receiver_endpoint_id_sorted is None:
        receiver_endpoint_id_sorted = [
            trace_index % len(receiver_xy)
            for trace_index in range(len(source_endpoint_id_sorted))
        ]
    source_sorted = np.asarray(source_endpoint_id_sorted, dtype=np.int64)
    receiver_sorted = np.asarray(receiver_endpoint_id_sorted, dtype=np.int64)
    n_traces = int(source_sorted.shape[0])
    if coordinate_scalar_sorted is None:
        coordinate_scalar_sorted = [1 for _ in range(n_traces)]
    return EndpointGeometryTables(
        n_traces=n_traces,
        source_x_m_sorted=np.ascontiguousarray(
            source_endpoints.x_m[source_sorted],
            dtype=np.float64,
        ),
        source_y_m_sorted=np.ascontiguousarray(
            source_endpoints.y_m[source_sorted],
            dtype=np.float64,
        ),
        receiver_x_m_sorted=np.ascontiguousarray(
            receiver_endpoints.x_m[receiver_sorted],
            dtype=np.float64,
        ),
        receiver_y_m_sorted=np.ascontiguousarray(
            receiver_endpoints.y_m[receiver_sorted],
            dtype=np.float64,
        ),
        coordinate_scalar_sorted=np.asarray(coordinate_scalar_sorted, dtype=np.int64),
        source_endpoint_id_sorted=source_sorted,
        receiver_endpoint_id_sorted=receiver_sorted,
        source_endpoints=source_endpoints,
        receiver_endpoints=receiver_endpoints,
        scalar_zero_count=sum(value == 0 for value in coordinate_scalar_sorted),
    )


def _none_case() -> tuple[EndpointGeometryTables, GeometryLinkageResult]:
    tables = _tables(
        source_xy=[(0.0, 0.0), (10.0, 0.0)],
        receiver_xy=[(100.0, 0.0), (200.0, 0.0)],
        source_endpoint_id_sorted=[0, 1, 0],
        receiver_endpoint_id_sorted=[0, 1, 1],
        coordinate_scalar_sorted=[1, 0, -2],
    )
    return tables, build_geometry_linkage(tables, GeometryLinkageOptions(mode='none'))


def _auto_case() -> tuple[EndpointGeometryTables, GeometryLinkageResult]:
    tables = _tables(
        source_xy=[(0.0, 0.0), (10.0, 0.0), (50.0, 0.0), (52.0, 0.0)],
        receiver_xy=[(0.0, 0.0), (100.0, 0.0)],
        source_endpoint_id_sorted=[0, 1, 2, 3, 2],
        receiver_endpoint_id_sorted=[0, 0, 1, 1, 1],
    )
    return tables, build_geometry_linkage(
        tables,
        GeometryLinkageOptions(
            mode='auto_threshold',
            threshold_m=5.0,
            receiver_location_interval_m=25.0,
        ),
    )


def _metadata() -> GeometryLinkageArtifactMetadata:
    return GeometryLinkageArtifactMetadata(
        job_id='linkage-job',
        input_file_id='file-id',
        key1_byte=189,
        key2_byte=193,
        source_x_byte=73,
        source_y_byte=77,
        receiver_x_byte=81,
        receiver_y_byte=85,
        coordinate_scalar_byte=71,
        header_source_segy_path='/data/input.sgy',
    )


def test_build_geometry_linkage_solution_arrays_contains_required_scalar_fields() -> None:
    tables, linkage = _none_case()

    arrays = build_geometry_linkage_solution_arrays(
        tables,
        linkage,
        metadata=_metadata(),
    )

    assert _REQUIRED_SCALAR_FIELDS.issubset(arrays)
    assert arrays['schema_version'].item() == 1
    assert arrays['artifact_kind'].item() == 'geometry_linkage'
    assert arrays['order'].item() == 'trace_store_sorted'
    assert arrays['mode'].item() == 'none'
    assert np.isnan(arrays['threshold_m'].item())
    assert arrays['job_id'].item() == 'linkage-job'
    assert arrays['key1_byte'].item() == pytest.approx(189.0)


def test_build_geometry_linkage_solution_arrays_contains_trace_level_mapping() -> None:
    tables, linkage = _none_case()

    arrays = build_geometry_linkage_solution_arrays(tables, linkage)

    assert _TRACE_LEVEL_FIELDS.issubset(arrays)
    np.testing.assert_array_equal(arrays['source_endpoint_id_sorted'], [0, 1, 0])
    np.testing.assert_array_equal(arrays['receiver_endpoint_id_sorted'], [0, 1, 1])
    np.testing.assert_array_equal(arrays['source_node_id_sorted'], [2, 3, 2])
    np.testing.assert_array_equal(arrays['receiver_node_id_sorted'], [0, 1, 1])
    np.testing.assert_array_equal(arrays['coordinate_scalar_sorted'], [1, 0, -2])


def test_build_geometry_linkage_solution_arrays_contains_endpoint_level_mapping() -> None:
    tables, linkage = _none_case()

    arrays = build_geometry_linkage_solution_arrays(tables, linkage)

    assert _SOURCE_ENDPOINT_FIELDS.issubset(arrays)
    assert _RECEIVER_ENDPOINT_FIELDS.issubset(arrays)
    np.testing.assert_array_equal(arrays['source_endpoint_id'], [0, 1])
    np.testing.assert_allclose(arrays['source_endpoint_x_m'], [0.0, 10.0])
    np.testing.assert_array_equal(arrays['source_node_id_by_endpoint'], [2, 3])
    np.testing.assert_array_equal(arrays['receiver_endpoint_id'], [0, 1])
    np.testing.assert_array_equal(arrays['receiver_node_id_by_endpoint'], [0, 1])


def test_build_geometry_linkage_solution_arrays_contains_record_arrays() -> None:
    tables, linkage = _auto_case()

    arrays = build_geometry_linkage_solution_arrays(tables, linkage)

    assert _RECORD_FIELDS.issubset(arrays)
    np.testing.assert_array_equal(
        arrays['record_endpoint_kind'],
        ['receiver', 'receiver', 'source', 'source', 'source', 'source'],
    )
    np.testing.assert_array_equal(arrays['record_endpoint_id'], [0, 1, 0, 1, 2, 3])
    np.testing.assert_array_equal(arrays['record_linked_to_kind'][:2], ['', ''])
    np.testing.assert_array_equal(arrays['record_linked_to_id'][:2], [-1, -1])
    assert np.isnan(arrays['record_distance_m'][0])
    assert arrays['record_method'][2] == 'receiver_anchor'


def test_geometry_linkage_npz_payload_has_no_object_dtype() -> None:
    tables, linkage = _auto_case()

    arrays = build_geometry_linkage_solution_arrays(tables, linkage)

    assert all(np.asarray(value).dtype != object for value in arrays.values())


def test_write_geometry_linkage_artifacts_writes_npz_csv_qc_json(
    tmp_path: Path,
) -> None:
    tables, linkage = _auto_case()

    paths = write_geometry_linkage_artifacts(
        job_dir=tmp_path,
        tables=tables,
        linkage=linkage,
        metadata=_metadata(),
    )

    assert paths.linkage_npz_path == tmp_path / 'geometry_linkage.npz'
    assert paths.linkage_csv_path == tmp_path / 'geometry_linkage.csv'
    assert paths.qc_json_path == tmp_path / 'geometry_linkage_qc.json'
    assert paths.linkage_npz_path.is_file()
    assert paths.linkage_csv_path.is_file()
    assert paths.qc_json_path.is_file()


def test_write_geometry_linkage_artifacts_npz_loads_with_allow_pickle_false(
    tmp_path: Path,
) -> None:
    tables, linkage = _auto_case()

    paths = write_geometry_linkage_artifacts(
        job_dir=tmp_path,
        tables=tables,
        linkage=linkage,
    )

    with np.load(paths.linkage_npz_path, allow_pickle=False) as npz:
        assert 'record_method' in npz.files
        assert all(npz[key].dtype != object for key in npz.files)


def test_write_geometry_linkage_artifacts_csv_columns_match_contract(
    tmp_path: Path,
) -> None:
    tables, linkage = _none_case()

    paths = write_geometry_linkage_artifacts(
        job_dir=tmp_path,
        tables=tables,
        linkage=linkage,
    )

    assert paths.linkage_csv_path.read_text(encoding='utf-8').splitlines()[0] == (
        ','.join(_CSV_COLUMNS)
    )
    with paths.linkage_csv_path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == _CSV_COLUMNS


def test_write_geometry_linkage_artifacts_csv_orders_receiver_then_source(
    tmp_path: Path,
) -> None:
    tables, linkage = _auto_case()

    paths = write_geometry_linkage_artifacts(
        job_dir=tmp_path,
        tables=tables,
        linkage=linkage,
    )

    with paths.linkage_csv_path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [(row['endpoint_kind'], row['endpoint_id']) for row in rows] == [
        ('receiver', '0'),
        ('receiver', '1'),
        ('source', '0'),
        ('source', '1'),
        ('source', '2'),
        ('source', '3'),
    ]


def test_write_geometry_linkage_artifacts_csv_serializes_none_as_blank(
    tmp_path: Path,
) -> None:
    tables, linkage = _none_case()

    paths = write_geometry_linkage_artifacts(
        job_dir=tmp_path,
        tables=tables,
        linkage=linkage,
    )

    with paths.linkage_csv_path.open(encoding='utf-8', newline='') as handle:
        first_row = next(csv.DictReader(handle))
    assert first_row['linked_to_kind'] == ''
    assert first_row['linked_to_id'] == ''
    assert first_row['distance_m'] == ''


def test_build_geometry_linkage_qc_payload_contains_counts_for_none_mode() -> None:
    tables, linkage = _none_case()

    payload = build_geometry_linkage_qc_payload(tables, linkage)

    assert payload['mode'] == 'none'
    assert payload['threshold_m'] is None
    assert payload['counts']['n_traces'] == 3
    assert payload['counts']['n_source_endpoints'] == 2
    assert payload['counts']['n_receiver_endpoints'] == 2
    assert payload['counts']['n_nodes'] == 4
    assert payload['counts']['n_source_only_nodes'] == 2
    assert payload['counts']['n_nodes_used_by_both_source_and_receiver'] == 0
    assert payload['counts']['n_nodes_used_only_by_source'] == 2
    assert payload['counts']['n_nodes_used_only_by_receiver'] == 2


def test_build_geometry_linkage_qc_payload_contains_counts_for_auto_threshold_mode() -> None:
    tables, linkage = _auto_case()

    payload = build_geometry_linkage_qc_payload(
        tables,
        linkage,
        metadata=_metadata(),
    )

    assert payload['mode'] == 'auto_threshold'
    assert payload['threshold_m'] == pytest.approx(5.0)
    assert payload['receiver_location_interval_m'] == pytest.approx(25.0)
    assert payload['counts']['n_receiver_anchor_links'] == 1
    assert payload['counts']['n_source_fallback_links'] == 2
    assert payload['counts']['n_independent_source_nodes'] == 1
    assert payload['job']['job_id'] == 'linkage-job'
    assert payload['headers']['key1_byte'] == 189


def test_build_geometry_linkage_qc_payload_contains_distance_stats() -> None:
    tables, linkage = _auto_case()

    payload = build_geometry_linkage_qc_payload(tables, linkage)

    assert payload['receiver_anchor_distance_m']['count'] == 1
    assert payload['receiver_anchor_distance_m']['min'] == pytest.approx(0.0)
    assert payload['source_fallback_distance_m']['count'] == 2
    assert payload['source_fallback_distance_m']['mean'] == pytest.approx(2.0)


def test_build_geometry_linkage_qc_payload_contains_endpoint_trace_count_stats() -> None:
    tables, linkage = _none_case()

    payload = build_geometry_linkage_qc_payload(tables, linkage)

    assert payload['source_endpoint_trace_count']['count'] == 2
    assert payload['source_endpoint_trace_count']['min'] == pytest.approx(1.0)
    assert payload['source_endpoint_trace_count']['max'] == pytest.approx(1.0)
    assert payload['receiver_endpoint_trace_count']['median'] == pytest.approx(1.0)


def test_build_geometry_linkage_qc_payload_contains_node_trace_count_stats() -> None:
    tables, linkage = _none_case()

    payload = build_geometry_linkage_qc_payload(tables, linkage)

    assert payload['node_trace_count']['count'] == 4
    assert payload['node_trace_count']['min'] == pytest.approx(1.0)
    assert payload['node_trace_count']['max'] == pytest.approx(2.0)
    assert payload['coordinate_scalar'] == {'zero_count': 1}


def test_build_geometry_linkage_qc_payload_has_no_nan_or_inf() -> None:
    tables, linkage = _none_case()

    payload = build_geometry_linkage_qc_payload(tables, linkage)

    json.dumps(payload, allow_nan=False)
    _assert_json_contains_no_nan_or_inf(payload)


def test_write_geometry_linkage_artifacts_rejects_mismatched_n_traces(
    tmp_path: Path,
) -> None:
    tables, linkage = _none_case()
    bad_linkage = replace(linkage, n_traces=tables.n_traces + 1)

    with pytest.raises(ValueError, match='n_traces'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=tables,
            linkage=bad_linkage,
        )


def test_write_geometry_linkage_artifacts_rejects_invalid_sorted_node_mapping(
    tmp_path: Path,
) -> None:
    tables, linkage = _none_case()
    bad_linkage = replace(
        linkage,
        source_node_id_sorted=np.asarray([2, 2, 2], dtype=np.int64),
    )

    with pytest.raises(ValueError, match='source_node_id_sorted'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=tables,
            linkage=bad_linkage,
        )


def test_write_geometry_linkage_artifacts_rejects_non_contiguous_node_ids(
    tmp_path: Path,
) -> None:
    tables, linkage = _none_case()
    bad_source_nodes = np.asarray([2, 4], dtype=np.int64)
    bad_linkage = replace(
        linkage,
        n_nodes=5,
        source_node_id_by_endpoint=bad_source_nodes,
        source_node_id_sorted=bad_source_nodes[tables.source_endpoint_id_sorted],
    )

    with pytest.raises(ValueError, match='0-based contiguous'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=tables,
            linkage=bad_linkage,
        )


def test_write_geometry_linkage_artifacts_rejects_invalid_record_count(
    tmp_path: Path,
) -> None:
    tables, linkage = _none_case()
    bad_linkage = replace(linkage, records=linkage.records[:-1])

    with pytest.raises(ValueError, match='records length'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=tables,
            linkage=bad_linkage,
        )


def test_write_geometry_linkage_artifacts_rejects_invalid_record_order(
    tmp_path: Path,
) -> None:
    tables, linkage = _none_case()
    bad_records = (linkage.records[1], linkage.records[0], *linkage.records[2:])
    bad_linkage = replace(linkage, records=bad_records)

    with pytest.raises(ValueError, match='ordered'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=tables,
            linkage=bad_linkage,
        )


def test_write_geometry_linkage_artifacts_rejects_non_finite_coordinate(
    tmp_path: Path,
) -> None:
    tables, linkage = _none_case()
    bad_source_endpoints = replace(
        tables.source_endpoints,
        x_m=np.asarray([0.0, np.nan], dtype=np.float64),
    )
    bad_tables = replace(tables, source_endpoints=bad_source_endpoints)

    with pytest.raises(ValueError, match='finite'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=bad_tables,
            linkage=linkage,
        )


def test_write_geometry_linkage_artifacts_rejects_negative_distance(
    tmp_path: Path,
) -> None:
    tables, linkage = _auto_case()
    source_record = linkage.records[2]
    bad_records = (
        *linkage.records[:2],
        replace(source_record, distance_m=-1.0),
        *linkage.records[3:],
    )
    bad_linkage = replace(linkage, records=bad_records)

    with pytest.raises(ValueError, match='distance_m'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=tables,
            linkage=bad_linkage,
        )


def test_write_geometry_linkage_artifacts_rejects_receiver_anchor_linked_to_source(
    tmp_path: Path,
) -> None:
    tables, linkage = _auto_case()
    index = _first_record_index(linkage, 'receiver_anchor')
    bad_records = _replace_record(
        linkage.records,
        index,
        linked_to_kind='source',
        linked_to_id=0,
    )
    bad_linkage = replace(linkage, records=bad_records)

    with pytest.raises(ValueError, match='receiver_anchor.*linked_to_kind receiver'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=tables,
            linkage=bad_linkage,
        )


def test_write_geometry_linkage_artifacts_rejects_source_fallback_linked_to_receiver(
    tmp_path: Path,
) -> None:
    tables, linkage = _auto_case()
    index = _first_record_index(linkage, 'source_fallback')
    bad_records = _replace_record(
        linkage.records,
        index,
        linked_to_kind='receiver',
        linked_to_id=0,
    )
    bad_linkage = replace(linkage, records=bad_records)

    with pytest.raises(ValueError, match='source_fallback.*linked_to_kind source'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=tables,
            linkage=bad_linkage,
        )


@pytest.mark.parametrize('method', ['receiver_seed', 'source_independent'])
def test_write_geometry_linkage_artifacts_rejects_unlinked_record_with_link(
    tmp_path: Path,
    method: str,
) -> None:
    tables, linkage = _auto_case()
    index = _first_record_index(linkage, method)
    bad_records = _replace_record(
        linkage.records,
        index,
        linked_to_kind='receiver',
        linked_to_id=0,
        distance_m=0.0,
    )
    bad_linkage = replace(linkage, records=bad_records)

    with pytest.raises(ValueError, match=f'{method}.*empty link'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=tables,
            linkage=bad_linkage,
        )


def test_write_geometry_linkage_artifacts_rejects_source_fallback_self_link(
    tmp_path: Path,
) -> None:
    tables, linkage = _auto_case()
    index = _first_record_index(linkage, 'source_fallback')
    endpoint_id = int(linkage.records[index].endpoint_id)
    bad_records = _replace_record(
        linkage.records,
        index,
        linked_to_id=endpoint_id,
        distance_m=0.0,
    )
    bad_linkage = replace(linkage, records=bad_records)

    with pytest.raises(ValueError, match='source_fallback must not self-link'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=tables,
            linkage=bad_linkage,
        )


def test_write_geometry_linkage_artifacts_rejects_linked_endpoint_out_of_range(
    tmp_path: Path,
) -> None:
    tables, linkage = _auto_case()
    index = _first_record_index(linkage, 'receiver_anchor')
    bad_records = _replace_record(
        linkage.records,
        index,
        linked_to_id=linkage.n_receiver_endpoints,
    )
    bad_linkage = replace(linkage, records=bad_records)

    with pytest.raises(ValueError, match='out of range for receiver'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=tables,
            linkage=bad_linkage,
        )


def test_write_geometry_linkage_artifacts_rejects_linked_node_mismatch(
    tmp_path: Path,
) -> None:
    tables, linkage = _auto_case()
    index = _first_record_index(linkage, 'receiver_anchor')
    source_id = int(linkage.records[index].endpoint_id)
    bad_records = _replace_record(
        linkage.records,
        index,
        linked_to_id=1,
        distance_m=float(
            np.hypot(
                tables.source_endpoints.x_m[source_id] - tables.receiver_endpoints.x_m[1],
                tables.source_endpoints.y_m[source_id] - tables.receiver_endpoints.y_m[1],
            )
        ),
    )
    bad_linkage = replace(linkage, records=bad_records)

    with pytest.raises(ValueError, match='linked receiver node'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=tables,
            linkage=bad_linkage,
        )


def test_write_geometry_linkage_artifacts_rejects_record_distance_mismatch(
    tmp_path: Path,
) -> None:
    tables, linkage = _auto_case()
    index = _first_record_index(linkage, 'receiver_anchor')
    bad_records = _replace_record(
        linkage.records,
        index,
        distance_m=float(linkage.records[index].distance_m or 0.0) + 1.0,
    )
    bad_linkage = replace(linkage, records=bad_records)

    with pytest.raises(ValueError, match='distance_m.*endpoint geometry distance'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=tables,
            linkage=bad_linkage,
        )


def test_write_geometry_linkage_artifacts_cleans_up_tmp_file_on_npz_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables, linkage = _none_case()

    def fail_savez(handle: Any, **payload: np.ndarray) -> None:
        handle.write(b'partial')
        raise RuntimeError('npz write failed')

    monkeypatch.setattr(artifacts.np, 'savez', fail_savez)

    with pytest.raises(RuntimeError, match='npz write failed'):
        write_geometry_linkage_artifacts(
            job_dir=tmp_path,
            tables=tables,
            linkage=linkage,
        )

    assert not (tmp_path / 'geometry_linkage.npz').exists()
    assert not list(tmp_path.glob('geometry_linkage.npz.tmp-*'))


def _assert_json_contains_no_nan_or_inf(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_json_contains_no_nan_or_inf(child)
        return
    if isinstance(value, list):
        for child in value:
            _assert_json_contains_no_nan_or_inf(child)
        return
    if isinstance(value, float):
        assert np.isfinite(value)


def _first_record_index(linkage: GeometryLinkageResult, method: str) -> int:
    for index, record in enumerate(linkage.records):
        if record.method == method:
            return index
    raise AssertionError(f'missing record method: {method}')


def _replace_record(
    records: tuple[Any, ...],
    index: int,
    **changes: object,
) -> tuple[Any, ...]:
    return (
        *records[:index],
        replace(records[index], **changes),
        *records[index + 1 :],
    )
