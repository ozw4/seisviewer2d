from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_artifacts import (
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    write_refraction_static_artifacts,
)
from app.services.refraction_static_artifacts.static_tables import (
    _receiver_static_table_columns,
    _source_static_table_columns,
)
from app.services.refraction_static_t1lsst import (
    compute_t1lsst_1layer_thickness,
    compute_t1lsst_1layer_weathering_correction,
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
    'source_v2_cell_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    'v1_m_s',
    'v2_m_s',
    'v2_status',
    'sh1_weathering_thickness_m',
    'total_weathering_thickness_m',
    'refractor_elevation_m',
    'weathering_correction_ms',
    'source_depth_m',
    'source_depth_shift_ms',
    'source_depth_status',
    'uphole_time_ms',
    'uphole_shift_ms',
    'uphole_status',
    'manual_static_shift_ms',
    'manual_static_status',
    'source_field_shift_ms',
    'source_field_status',
    'source_field_static_status',
    'source_total_with_field_shift_ms',
    'floating_datum_correction_ms',
    'flat_datum_correction_ms',
    'elevation_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'static_status',
    'sign_convention',
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
    'receiver_v2_cell_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    'v1_m_s',
    'v2_m_s',
    'v2_status',
    'sh1_weathering_thickness_m',
    'total_weathering_thickness_m',
    'refractor_elevation_m',
    'weathering_correction_ms',
    'manual_static_shift_ms',
    'manual_static_status',
    'receiver_field_shift_ms',
    'receiver_field_status',
    'receiver_field_static_status',
    'receiver_total_with_field_shift_ms',
    'floating_datum_correction_ms',
    'flat_datum_correction_ms',
    'elevation_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'static_status',
    'sign_convention',
    'pick_count',
    'used_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
]


def test_static_table_optional_column_order_contract() -> None:
    result = _result()

    source_columns = _source_static_table_columns(result)
    receiver_columns = _receiver_static_table_columns(result)

    assert list(source_columns) == SOURCE_COLUMNS
    assert list(receiver_columns) == RECEIVER_COLUMNS
    assert source_columns.index('source_depth_m') == (
        source_columns.index('weathering_correction_ms') + 1
    )
    assert source_columns.index('uphole_time_ms') == (
        source_columns.index('source_depth_status') + 1
    )
    assert source_columns.index('manual_static_shift_ms') == (
        source_columns.index('uphole_status') + 1
    )
    assert source_columns.index('source_field_shift_ms') == (
        source_columns.index('manual_static_status') + 1
    )
    assert receiver_columns.index('manual_static_shift_ms') == (
        receiver_columns.index('weathering_correction_ms') + 1
    )
    assert receiver_columns.index('receiver_field_shift_ms') == (
        receiver_columns.index('manual_static_status') + 1
    )


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


def test_manual_static_components_are_written_to_endpoint_artifacts(
    tmp_path: Path,
) -> None:
    base = _result()
    result = replace(
        base,
        source_manual_static_shift_s=np.asarray([0.003, np.nan], dtype=np.float64),
        source_manual_static_status=np.asarray(
            ['ok', 'missing_manual_static'],
            dtype='<U48',
        ),
        receiver_manual_static_shift_s=np.asarray(
            [-0.004, np.nan],
            dtype=np.float64,
        ),
        receiver_manual_static_status=np.asarray(
            ['ok', 'missing_manual_static'],
            dtype='<U48',
        ),
        manual_static_field_correction_qc={
            'manual_static_mode': 'artifact_table',
            'manual_static_sign_convention': 'applied_shift_s',
            'component_name': 'manual_static_shift_s',
        },
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=_request(),
        job_dir=tmp_path,
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    assert float(source_rows[0]['manual_static_shift_ms']) == pytest.approx(3.0)
    assert source_rows[0]['manual_static_status'] == 'ok'
    assert float(receiver_rows[0]['manual_static_shift_ms']) == pytest.approx(-4.0)
    assert receiver_rows[0]['manual_static_status'] == 'ok'
    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        np.testing.assert_allclose(data['source_manual_static_shift_s'][0], 0.003)
        np.testing.assert_allclose(data['receiver_manual_static_shift_s'][0], -0.004)
        assert data['source_manual_static_status'][0] == 'ok'
        assert data['receiver_manual_static_status'][0] == 'ok'


def test_solve_cell_static_tables_use_endpoint_local_v2(tmp_path: Path) -> None:
    base = _result()
    source_v2 = np.asarray([2200.0, 3000.0], dtype=np.float64)
    receiver_v2 = np.asarray([2600.0, 3200.0], dtype=np.float64)
    source_sh1 = compute_t1lsst_1layer_thickness(
        base.source_half_intercept_time_s,
        base.weathering_velocity_m_s,
        source_v2,
    )
    receiver_sh1 = compute_t1lsst_1layer_thickness(
        base.receiver_half_intercept_time_s,
        base.weathering_velocity_m_s,
        receiver_v2,
    )
    source_wcor = compute_t1lsst_1layer_weathering_correction(
        source_sh1,
        base.weathering_velocity_m_s,
        source_v2,
    )
    receiver_wcor = compute_t1lsst_1layer_weathering_correction(
        receiver_sh1,
        base.weathering_velocity_m_s,
        receiver_v2,
    )
    source_index_sorted = np.asarray([0, 1, 0, 1], dtype=np.int64)
    receiver_index_sorted = np.asarray([0, 1, 1, 0], dtype=np.int64)
    source_refraction = (
        source_wcor
        + base.source_floating_datum_elevation_shift_s
        + base.source_flat_datum_shift_s
    )
    receiver_refraction = (
        receiver_wcor
        + base.receiver_floating_datum_elevation_shift_s
        + base.receiver_flat_datum_shift_s
    )
    result = _solve_cell_result(
        base,
        bedrock_velocity_m_s=float(np.median(np.concatenate([source_v2, receiver_v2]))),
        source_v2_cell_id=np.asarray([0, 1], dtype=np.int64),
        source_v2_m_s=source_v2,
        source_v2_status=np.asarray(['ok', 'ok'], dtype='<U2'),
        receiver_v2_cell_id=np.asarray([1, 2], dtype=np.int64),
        receiver_v2_m_s=receiver_v2,
        receiver_v2_status=np.asarray(['ok', 'ok'], dtype='<U2'),
        source_weathering_thickness_m=source_sh1,
        receiver_weathering_thickness_m=receiver_sh1,
        source_refractor_elevation_m=base.source_surface_elevation_m - source_sh1,
        receiver_refractor_elevation_m=(
            base.receiver_surface_elevation_m - receiver_sh1
        ),
        source_weathering_replacement_shift_s=source_wcor,
        receiver_weathering_replacement_shift_s=receiver_wcor,
        source_refraction_shift_s=source_refraction,
        receiver_refraction_shift_s=receiver_refraction,
        source_v2_cell_id_sorted=np.asarray([0, 1, 0, 1], dtype=np.int64),
        source_v2_m_s_sorted=source_v2[source_index_sorted],
        source_v2_status_sorted=np.asarray(['ok', 'ok', 'ok', 'ok'], dtype='<U2'),
        receiver_v2_cell_id_sorted=np.asarray([1, 2, 2, 1], dtype=np.int64),
        receiver_v2_m_s_sorted=receiver_v2[receiver_index_sorted],
        receiver_v2_status_sorted=np.asarray(['ok', 'ok', 'ok', 'ok'], dtype='<U2'),
        source_weathering_thickness_m_sorted=source_sh1[source_index_sorted],
        receiver_weathering_thickness_m_sorted=receiver_sh1[receiver_index_sorted],
        source_weathering_replacement_shift_s_sorted=source_wcor[
            source_index_sorted
        ],
        receiver_weathering_replacement_shift_s_sorted=receiver_wcor[
            receiver_index_sorted
        ],
        source_refraction_shift_s_sorted=source_refraction[source_index_sorted],
        receiver_refraction_shift_s_sorted=receiver_refraction[receiver_index_sorted],
        weathering_replacement_trace_shift_s_sorted=(
            source_wcor[source_index_sorted] + receiver_wcor[receiver_index_sorted]
        ),
        refraction_trace_shift_s_sorted=(
            source_refraction[source_index_sorted]
            + receiver_refraction[receiver_index_sorted]
        ),
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    assert source_rows[0]['source_v2_cell_id'] == '0'
    assert receiver_rows[1]['receiver_v2_cell_id'] == '2'
    assert float(source_rows[0]['v2_m_s']) == pytest.approx(source_v2[0])
    assert float(receiver_rows[1]['v2_m_s']) == pytest.approx(receiver_v2[1])
    assert float(source_rows[1]['weathering_correction_ms']) == pytest.approx(
        source_wcor[1] * 1000.0
    )
    assert float(receiver_rows[0]['weathering_correction_ms']) == pytest.approx(
        receiver_wcor[0] * 1000.0
    )

    trace_rows = _read_csv(paths.refraction_statics_csv)
    assert float(trace_rows[0]['refraction_trace_shift_ms']) == pytest.approx(
        (
            float(trace_rows[0]['source_refraction_shift_ms'])
            + float(trace_rows[0]['receiver_refraction_shift_ms'])
        )
    )
    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        assert data['source_v2_cell_id'].tolist() == [0, 1]
        assert data['receiver_v2_m_s'].tolist() == pytest.approx(receiver_v2.tolist())


def test_solve_cell_endpoint_outside_grid_statused_explicitly(
    tmp_path: Path,
) -> None:
    source_status_sorted = np.asarray(
        ['outside_refractor_cell_grid', 'ok', 'outside_refractor_cell_grid', 'ok'],
        dtype='<U32',
    )
    result = _solve_cell_result(
        _result(),
        source_v2_cell_id=np.asarray([-1, 1], dtype=np.int64),
        source_v2_m_s=np.asarray([np.nan, 2500.0], dtype=np.float64),
        source_v2_status=np.asarray(
            ['outside_refractor_cell_grid', 'ok'],
            dtype='<U32',
        ),
        source_weathering_thickness_m=np.asarray([np.nan, 12.0], dtype=np.float64),
        source_weathering_replacement_shift_s=np.asarray(
            [np.nan, -0.0102],
            dtype=np.float64,
        ),
        source_refraction_shift_s=np.asarray([np.nan, 0.0023], dtype=np.float64),
        source_datum_status=np.asarray(
            ['outside_refractor_cell_grid', 'ok'],
            dtype='<U32',
        ),
        source_v2_cell_id_sorted=np.asarray([-1, 1, -1, 1], dtype=np.int64),
        source_v2_m_s_sorted=np.asarray(
            [np.nan, 2500.0, np.nan, 2500.0],
            dtype=np.float64,
        ),
        source_v2_status_sorted=source_status_sorted,
        source_weathering_thickness_m_sorted=np.asarray(
            [np.nan, 12.0, np.nan, 12.0],
            dtype=np.float64,
        ),
        source_weathering_replacement_shift_s_sorted=np.asarray(
            [np.nan, -0.0102, np.nan, -0.0102],
            dtype=np.float64,
        ),
        source_refraction_shift_s_sorted=np.asarray(
            [np.nan, 0.0023, np.nan, 0.0023],
            dtype=np.float64,
        ),
        weathering_replacement_trace_shift_s_sorted=np.asarray(
            [np.nan, -0.0221, np.nan, -0.0204],
            dtype=np.float64,
        ),
        refraction_trace_shift_s_sorted=np.asarray(
            [np.nan, 0.0044, np.nan, 0.0046],
            dtype=np.float64,
        ),
        trace_static_valid_mask_sorted=np.asarray(
            [False, True, False, True],
            dtype=bool,
        ),
        trace_static_status_sorted=source_status_sorted,
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=_solve_cell_request(),
        job_dir=tmp_path,
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    assert source_rows[0]['source_v2_cell_id'] == ''
    assert source_rows[0]['v2_m_s'] == ''
    assert source_rows[0]['static_status'] == 'outside_refractor_cell_grid'
    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        assert data['source_static_status'][0] == 'outside_refractor_cell_grid'


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
    assert float(row['total_weathering_thickness_m']) == pytest.approx(
        float(data['source_total_weathering_thickness_m'][index])
    )
    assert float(row['weathering_correction_ms']) == pytest.approx(
        float(data['source_weathering_correction_s'][index]) * 1000.0
    )
    if 'source_depth_shift_ms' in row:
        assert float(row['source_depth_shift_ms']) == pytest.approx(
            float(data['source_depth_shift_s'][index]) * 1000.0
        )
        assert row['source_depth_status'] == str(data['source_depth_status'][index])
    if 'uphole_shift_ms' in row:
        assert float(row['uphole_shift_ms']) == pytest.approx(
            float(data['source_uphole_shift_s'][index]) * 1000.0
        )
        assert row['uphole_status'] == str(data['source_uphole_status'][index])
    if 'manual_static_shift_ms' in row:
        assert float(row['manual_static_shift_ms']) == pytest.approx(
            float(data['source_manual_static_shift_s'][index]) * 1000.0
        )
        assert row['manual_static_status'] == str(
            data['source_manual_static_status'][index]
        )
    if 'source_field_shift_ms' in row:
        assert float(row['source_field_shift_ms']) == pytest.approx(
            float(data['source_field_shift_s'][index]) * 1000.0
        )
        assert row['source_field_static_status'] == str(
            data['source_field_static_status'][index]
        )
        assert float(row['source_total_with_field_shift_ms']) == pytest.approx(
            float(data['source_total_with_field_shift_s'][index]) * 1000.0
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
    assert row['sign_convention'] == str(data['sign_convention'])


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
    assert float(row['total_weathering_thickness_m']) == pytest.approx(
        float(data['receiver_total_weathering_thickness_m'][index])
    )
    assert float(row['weathering_correction_ms']) == pytest.approx(
        float(data['receiver_weathering_correction_s'][index]) * 1000.0
    )
    if 'manual_static_shift_ms' in row:
        assert float(row['manual_static_shift_ms']) == pytest.approx(
            float(data['receiver_manual_static_shift_s'][index]) * 1000.0
        )
        assert row['manual_static_status'] == str(
            data['receiver_manual_static_status'][index]
        )
    if 'receiver_field_shift_ms' in row:
        assert float(row['receiver_field_shift_ms']) == pytest.approx(
            float(data['receiver_field_shift_s'][index]) * 1000.0
        )
        assert row['receiver_field_static_status'] == str(
            data['receiver_field_static_status'][index]
        )
        assert float(row['receiver_total_with_field_shift_ms']) == pytest.approx(
            float(data['receiver_total_with_field_shift_s'][index]) * 1000.0
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
    assert row['sign_convention'] == str(data['sign_convention'])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _solve_cell_request():
    payload = _request().model_dump(mode='json')
    payload['model']['bedrock_velocity_mode'] = 'solve_cell'
    payload['model']['refractor_cell'] = {
        'number_of_cell_x': 3,
        'size_of_cell_x_m': 100.0,
        'x_coordinate_origin_m': 0.0,
        'number_of_cell_y': 1,
        'size_of_cell_y_m': None,
        'y_coordinate_origin_m': 0.0,
        'assignment_mode': 'midpoint',
        'outside_grid_policy': 'reject',
        'min_observations_per_cell': 1,
        'velocity_smoothing_weight': 0.0,
    }
    return RefractionStaticApplyRequest.model_validate(payload)


def _solve_cell_result(result, **overrides):
    n_nodes = int(result.node_id.shape[0])
    n_sources = int(result.source_endpoint_key.shape[0])
    n_receivers = int(result.receiver_endpoint_key.shape[0])
    n_traces = int(result.sorted_trace_index.shape[0])
    n_rows = int(result.row_trace_index_sorted.shape[0])
    n_cells = 3
    bedrock_velocity = float(
        overrides.get('bedrock_velocity_m_s', result.bedrock_velocity_m_s)
    )
    defaults = {
        'bedrock_velocity_mode': 'solve_cell',
        'active_cell_id': np.arange(n_cells, dtype=np.int64),
        'inactive_cell_id': np.empty(0, dtype=np.int64),
        'cell_bedrock_slowness_s_per_m': np.full(
            n_cells,
            1.0 / bedrock_velocity,
            dtype=np.float64,
        ),
        'cell_bedrock_velocity_m_s': np.full(
            n_cells,
            bedrock_velocity,
            dtype=np.float64,
        ),
        'cell_velocity_status': np.full(n_cells, 'solved', dtype='<U16'),
        'row_midpoint_cell_id': np.arange(n_rows, dtype=np.int64) % n_cells,
        'node_v2_cell_id': np.arange(n_nodes, dtype=np.int64),
        'node_v2_m_s': np.full(n_nodes, bedrock_velocity, dtype=np.float64),
        'node_v2_status': np.full(n_nodes, 'ok', dtype='<U32'),
        'source_v2_cell_id': np.arange(n_sources, dtype=np.int64),
        'source_v2_m_s': np.full(
            n_sources,
            bedrock_velocity,
            dtype=np.float64,
        ),
        'source_v2_status': np.full(n_sources, 'ok', dtype='<U32'),
        'receiver_v2_cell_id': np.arange(n_receivers, dtype=np.int64),
        'receiver_v2_m_s': np.full(
            n_receivers,
            bedrock_velocity,
            dtype=np.float64,
        ),
        'receiver_v2_status': np.full(n_receivers, 'ok', dtype='<U32'),
        'source_v2_cell_id_sorted': np.zeros(n_traces, dtype=np.int64),
        'source_v2_m_s_sorted': np.full(
            n_traces,
            bedrock_velocity,
            dtype=np.float64,
        ),
        'source_v2_status_sorted': np.full(n_traces, 'ok', dtype='<U32'),
        'receiver_v2_cell_id_sorted': np.zeros(n_traces, dtype=np.int64),
        'receiver_v2_m_s_sorted': np.full(
            n_traces,
            bedrock_velocity,
            dtype=np.float64,
        ),
        'receiver_v2_status_sorted': np.full(n_traces, 'ok', dtype='<U32'),
    }
    defaults.update(overrides)
    return replace(result, **defaults)
