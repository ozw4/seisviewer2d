from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.services.refraction_static_artifacts import (
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    write_refraction_static_artifacts,
)
from app.tests._refraction_static_synthetic import (
    SYNTHETIC_RECEIVER_NODE_ID,
    SYNTHETIC_SOURCE_NODE_ID,
    SYNTHETIC_WCOR_TOLERANCE_MS,
    expected_wcor_s_for_node,
    run_synthetic_refraction_statics,
    synthetic_refraction_apply_request,
)
from app.tests._refraction_static_artifact_helpers import _request, _result


SOURCE_COLUMNS = [
    'endpoint_kind',
    'source_endpoint_key',
    'source_id',
    'source_node_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    'v1_m_s',
    'v2_m_s',
    'sh1_weathering_thickness_m',
    'refractor_elevation_m',
    'weathering_correction_ms',
    'floating_datum_correction_ms',
    'flat_datum_correction_ms',
    'elevation_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'static_status',
    'pick_count',
    'used_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
]

RECEIVER_COLUMNS = [
    'endpoint_kind',
    'receiver_endpoint_key',
    'receiver_id',
    'receiver_node_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    'v1_m_s',
    'v2_m_s',
    'sh1_weathering_thickness_m',
    'refractor_elevation_m',
    'weathering_correction_ms',
    'floating_datum_correction_ms',
    'flat_datum_correction_ms',
    'elevation_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'static_status',
    'pick_count',
    'used_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
]


def test_source_static_table_has_one_row_per_source_endpoint(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.source_static_table_csv)
    assert paths.source_static_table_csv.name == SOURCE_STATIC_TABLE_CSV_NAME
    assert len(rows) == 2
    assert list(rows[0]) == SOURCE_COLUMNS
    assert [row['endpoint_kind'] for row in rows] == ['source', 'source']
    assert [row['source_endpoint_key'] for row in rows] == ['s0', 's1']
    assert [int(row['source_node_id']) for row in rows] == [0, 1]


def test_receiver_static_table_has_one_row_per_receiver_endpoint(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.receiver_static_table_csv)
    assert paths.receiver_static_table_csv.name == RECEIVER_STATIC_TABLE_CSV_NAME
    assert len(rows) == 2
    assert list(rows[0]) == RECEIVER_COLUMNS
    assert [row['endpoint_kind'] for row in rows] == ['receiver', 'receiver']
    assert [row['receiver_endpoint_key'] for row in rows] == ['r0', 'r1']
    assert [int(row['receiver_node_id']) for row in rows] == [1, 2]


def test_source_receiver_static_tables_have_one_row_per_endpoint(
    tmp_path: Path,
) -> None:
    req = synthetic_refraction_apply_request()
    result = run_synthetic_refraction_statics(req=req)

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    trace_rows = _read_csv(paths.refraction_statics_csv)
    assert len(source_rows) == int(SYNTHETIC_SOURCE_NODE_ID.shape[0])
    assert len(receiver_rows) == int(SYNTHETIC_RECEIVER_NODE_ID.shape[0])
    assert {int(row['source_node_id']) for row in source_rows} == set(
        SYNTHETIC_SOURCE_NODE_ID.tolist()
    )
    assert {int(row['receiver_node_id']) for row in receiver_rows} == set(
        SYNTHETIC_RECEIVER_NODE_ID.tolist()
    )

    for row in source_rows:
        node_id = int(row['source_node_id'])
        expected_wcor_ms = expected_wcor_s_for_node(node_id) * 1000.0
        assert float(row['weathering_correction_ms']) == pytest.approx(
            expected_wcor_ms,
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )
        assert float(row['total_static_ms']) == pytest.approx(
            expected_wcor_ms,
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )
        assert float(row['total_applied_shift_ms']) == pytest.approx(
            expected_wcor_ms,
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )

    for row in receiver_rows:
        node_id = int(row['receiver_node_id'])
        expected_wcor_ms = expected_wcor_s_for_node(node_id) * 1000.0
        assert float(row['weathering_correction_ms']) == pytest.approx(
            expected_wcor_ms,
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )
        assert float(row['total_static_ms']) == pytest.approx(
            expected_wcor_ms,
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )
        assert float(row['total_applied_shift_ms']) == pytest.approx(
            expected_wcor_ms,
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )

    for row in trace_rows:
        expected_trace_shift_ms = (
            float(row['source_refraction_shift_ms'])
            + float(row['receiver_refraction_shift_ms'])
        )
        assert float(row['refraction_trace_shift_ms']) == pytest.approx(
            expected_trace_shift_ms,
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )
        assert float(row['weathering_replacement_trace_shift_ms']) == pytest.approx(
            expected_trace_shift_ms,
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )


def test_source_receiver_static_tables_match_npz(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    assert paths.source_receiver_static_table_npz.name == (
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME
    )

    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        _assert_source_row_matches_npz(source_rows[0], data, 0)
        _assert_receiver_row_matches_npz(receiver_rows[0], data, 0)


def test_linked_source_receiver_share_same_node_t1_and_sh1(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    source = next(row for row in source_rows if row['source_node_id'] == '1')
    receiver = next(row for row in receiver_rows if row['receiver_node_id'] == '1')

    assert float(source['t1_ms']) == pytest.approx(float(receiver['t1_ms']))
    assert float(source['sh1_weathering_thickness_m']) == pytest.approx(
        float(receiver['sh1_weathering_thickness_m'])
    )


def test_static_tables_include_inactive_endpoint_status(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    inactive = next(row for row in receiver_rows if row['receiver_node_id'] == '2')
    assert inactive['solution_status'] == 'inactive'
    assert inactive['weathering_status'] == 'inactive'
    assert inactive['static_status'] == 'inactive_endpoint'

    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        assert data['receiver_static_status'].tolist() == ['ok', 'inactive_endpoint']


def test_static_tables_include_missing_linkage_status(tmp_path: Path) -> None:
    result = replace(
        _result(),
        receiver_node_id=np.asarray([1, 99], dtype=np.int64),
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=_request(),
        job_dir=tmp_path,
    )

    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    missing = next(row for row in receiver_rows if row['receiver_endpoint_key'] == 'r1')
    assert missing['solution_status'] == 'missing_solution'
    assert missing['weathering_status'] == 'missing_node'
    assert missing['static_status'] == 'missing_linkage'

    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        assert data['receiver_static_status'].tolist() == ['ok', 'missing_linkage']


def test_static_tables_include_missing_geometry_status(tmp_path: Path) -> None:
    source_x = np.asarray([np.nan, 50.0], dtype=np.float64)
    result = replace(_result(), source_x_m=source_x)

    paths = write_refraction_static_artifacts(
        result=result,
        req=_request(),
        job_dir=tmp_path,
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    missing = next(row for row in source_rows if row['source_endpoint_key'] == 's0')
    assert missing['solution_status'] == 'solved'
    assert missing['weathering_status'] == 'ok'
    assert missing['static_status'] == 'missing_geometry'

    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        assert data['source_static_status'].tolist() == ['missing_geometry', 'ok']


def test_static_tables_include_insufficient_pick_fold_status(tmp_path: Path) -> None:
    result = replace(
        _result(),
        node_solution_status=np.asarray(['solved', 'low_fold', 'inactive'], dtype='<U16'),
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=_request(),
        job_dir=tmp_path,
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    source = next(row for row in source_rows if row['source_endpoint_key'] == 's1')
    receiver = next(row for row in receiver_rows if row['receiver_endpoint_key'] == 'r0')
    assert source['solution_status'] == 'low_fold'
    assert receiver['solution_status'] == 'low_fold'
    assert source['static_status'] == 'insufficient_pick_fold'
    assert receiver['static_status'] == 'insufficient_pick_fold'

    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        assert data['source_static_status'].tolist() == [
            'ok',
            'insufficient_pick_fold',
        ]
        assert data['receiver_static_status'].tolist() == [
            'insufficient_pick_fold',
            'inactive_endpoint',
        ]


def test_static_tables_are_pickle_free_npz(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        assert data.files
        for key in data.files:
            assert data[key].dtype != object


def _assert_source_row_matches_npz(
    row: dict[str, str],
    data: np.lib.npyio.NpzFile,
    index: int,
) -> None:
    assert row['source_endpoint_key'] == str(data['source_endpoint_key'][index])
    assert int(row['source_id']) == int(data['source_id'][index])
    assert int(row['source_node_id']) == int(data['source_node_id'][index])
    assert float(row['x_m']) == pytest.approx(float(data['source_x_m'][index]))
    assert float(row['y_m']) == pytest.approx(float(data['source_y_m'][index]))
    assert float(row['surface_elevation_m']) == pytest.approx(
        float(data['source_surface_elevation_m'][index])
    )
    assert float(row['t1_ms']) == pytest.approx(float(data['source_t1_s'][index]) * 1000.0)
    assert float(row['v1_m_s']) == pytest.approx(float(data['source_v1_m_s'][index]))
    assert float(row['v2_m_s']) == pytest.approx(float(data['source_v2_m_s'][index]))
    assert float(row['sh1_weathering_thickness_m']) == pytest.approx(
        float(data['source_sh1_m'][index])
    )
    assert float(row['weathering_correction_ms']) == pytest.approx(
        float(data['source_weathering_correction_s'][index]) * 1000.0
    )
    assert float(row['elevation_correction_ms']) == pytest.approx(
        float(data['source_elevation_correction_s'][index]) * 1000.0
    )
    assert float(row['total_static_ms']) == pytest.approx(
        float(data['source_total_static_s'][index]) * 1000.0
    )
    assert float(row['total_applied_shift_ms']) == pytest.approx(
        float(data['source_total_applied_shift_s'][index]) * 1000.0
    )
    assert row['static_status'] == str(data['source_static_status'][index])


def _assert_receiver_row_matches_npz(
    row: dict[str, str],
    data: np.lib.npyio.NpzFile,
    index: int,
) -> None:
    assert row['receiver_endpoint_key'] == str(data['receiver_endpoint_key'][index])
    assert int(row['receiver_id']) == int(data['receiver_id'][index])
    assert int(row['receiver_node_id']) == int(data['receiver_node_id'][index])
    assert float(row['x_m']) == pytest.approx(float(data['receiver_x_m'][index]))
    assert float(row['y_m']) == pytest.approx(float(data['receiver_y_m'][index]))
    assert float(row['surface_elevation_m']) == pytest.approx(
        float(data['receiver_surface_elevation_m'][index])
    )
    assert float(row['t1_ms']) == pytest.approx(
        float(data['receiver_t1_s'][index]) * 1000.0
    )
    assert float(row['v1_m_s']) == pytest.approx(float(data['receiver_v1_m_s'][index]))
    assert float(row['v2_m_s']) == pytest.approx(float(data['receiver_v2_m_s'][index]))
    assert float(row['sh1_weathering_thickness_m']) == pytest.approx(
        float(data['receiver_sh1_m'][index])
    )
    assert float(row['weathering_correction_ms']) == pytest.approx(
        float(data['receiver_weathering_correction_s'][index]) * 1000.0
    )
    assert float(row['elevation_correction_ms']) == pytest.approx(
        float(data['receiver_elevation_correction_s'][index]) * 1000.0
    )
    assert float(row['total_static_ms']) == pytest.approx(
        float(data['receiver_total_static_s'][index]) * 1000.0
    )
    assert float(row['total_applied_shift_ms']) == pytest.approx(
        float(data['receiver_total_applied_shift_s'][index]) * 1000.0
    )
    assert row['static_status'] == str(data['receiver_static_status'][index])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))
