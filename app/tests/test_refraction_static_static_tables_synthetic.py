from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.services.refraction_static_artifacts import write_refraction_static_artifacts
from app.tests._refraction_static_synthetic import (
    SYNTHETIC_CELL_NODE_CELL_ID,
    SYNTHETIC_CELL_V2_M_S,
    SYNTHETIC_CELL_V2_TOLERANCE_M_S,
    SYNTHETIC_SH1_TOLERANCE_M,
    SYNTHETIC_T1_TOLERANCE_MS,
    SYNTHETIC_V1_M_S,
    SYNTHETIC_WCOR_TOLERANCE_MS,
    expected_cell_sh1_m_for_node,
    expected_cell_t1_s_for_node,
    expected_cell_wcor_s_for_node,
    run_synthetic_cell_refraction_statics,
    synthetic_cell_refraction_apply_request,
)


@pytest.fixture(scope='module')
def synthetic_static_table_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    req = synthetic_cell_refraction_apply_request(conversion_mode='t1lsst_1layer')
    result = run_synthetic_cell_refraction_statics(req=req)
    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path_factory.mktemp('refraction-static-tables-synthetic'),
    )

    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        table_arrays = {name: data[name].copy() for name in data.files}

    return {
        'result': result,
        'source_rows': _read_csv(paths.source_static_table_csv),
        'receiver_rows': _read_csv(paths.receiver_static_table_csv),
        'trace_rows': _read_csv(paths.refraction_statics_csv),
        'table_arrays': table_arrays,
    }


def test_source_static_table_uses_local_cell_v2_for_t1lsst(
    synthetic_static_table_artifacts: dict[str, Any],
) -> None:
    result = synthetic_static_table_artifacts['result']
    source_rows = synthetic_static_table_artifacts['source_rows']

    for node_id in (0, 6):
        row = _row_by_node(source_rows, node_column='source_node_id', node_id=node_id)
        _assert_endpoint_uses_direct_cell_v2(row, node_id, result.bedrock_velocity_m_s)
        _assert_endpoint_row_matches_known_truth(row, node_id)


def test_receiver_static_table_uses_local_cell_v2_for_t1lsst(
    synthetic_static_table_artifacts: dict[str, Any],
) -> None:
    result = synthetic_static_table_artifacts['result']
    receiver_rows = synthetic_static_table_artifacts['receiver_rows']

    for node_id in (0, 6):
        row = _row_by_node(
            receiver_rows,
            node_column='receiver_node_id',
            node_id=node_id,
        )
        _assert_endpoint_uses_direct_cell_v2(row, node_id, result.bedrock_velocity_m_s)
        _assert_endpoint_row_matches_known_truth(row, node_id)


def test_static_table_sh1_matches_known_truth_with_local_v2(
    synthetic_static_table_artifacts: dict[str, Any],
) -> None:
    for row, node_id in _all_endpoint_rows(synthetic_static_table_artifacts):
        assert float(row['t1_ms']) == pytest.approx(
            expected_cell_t1_s_for_node(node_id) * 1000.0,
            abs=SYNTHETIC_T1_TOLERANCE_MS,
        )
        assert float(row['sh1_weathering_thickness_m']) == pytest.approx(
            expected_cell_sh1_m_for_node(node_id),
            abs=SYNTHETIC_SH1_TOLERANCE_M,
        )


def test_static_table_wcor_matches_known_truth_and_sign_convention(
    synthetic_static_table_artifacts: dict[str, Any],
) -> None:
    for row, node_id in _all_endpoint_rows(synthetic_static_table_artifacts):
        expected_wcor_ms = expected_cell_wcor_s_for_node(node_id) * 1000.0

        assert float(row['weathering_correction_ms']) == pytest.approx(
            expected_wcor_ms,
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )
        assert float(row['weathering_correction_ms']) < 0.0
        assert float(row['total_static_ms']) == pytest.approx(
            expected_wcor_ms,
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )
        assert float(row['total_applied_shift_ms']) == pytest.approx(
            float(row['total_static_ms'])
        )
        assert row['static_status'] == 'ok'


def test_source_receiver_static_table_csv_and_npz_are_consistent(
    synthetic_static_table_artifacts: dict[str, Any],
) -> None:
    source_rows = synthetic_static_table_artifacts['source_rows']
    receiver_rows = synthetic_static_table_artifacts['receiver_rows']
    table = synthetic_static_table_artifacts['table_arrays']

    for index, row in enumerate(source_rows):
        _assert_npz_endpoint_matches_csv(table, 'source', row, index)

    for index, row in enumerate(receiver_rows):
        _assert_npz_endpoint_matches_csv(table, 'receiver', row, index)


def test_trace_shift_equals_source_plus_receiver_components_for_synthetic_case(
    synthetic_static_table_artifacts: dict[str, Any],
) -> None:
    for row in synthetic_static_table_artifacts['trace_rows']:
        source_node_id = int(row['source_node_id'])
        receiver_node_id = int(row['receiver_node_id'])
        expected_source_ms = expected_cell_wcor_s_for_node(source_node_id) * 1000.0
        expected_receiver_ms = expected_cell_wcor_s_for_node(receiver_node_id) * 1000.0
        expected_trace_ms = expected_source_ms + expected_receiver_ms

        assert float(row['source_refraction_shift_ms']) == pytest.approx(
            expected_source_ms,
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )
        assert float(row['receiver_refraction_shift_ms']) == pytest.approx(
            expected_receiver_ms,
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )
        assert float(row['refraction_trace_shift_ms']) == pytest.approx(
            expected_trace_ms,
            abs=2.0 * SYNTHETIC_WCOR_TOLERANCE_MS,
        )
        assert float(row['refraction_trace_shift_ms']) == pytest.approx(
            float(row['source_refraction_shift_ms'])
            + float(row['receiver_refraction_shift_ms'])
        )
        assert float(row['weathering_replacement_trace_shift_ms']) == pytest.approx(
            expected_trace_ms,
            abs=2.0 * SYNTHETIC_WCOR_TOLERANCE_MS,
        )


def _assert_endpoint_uses_direct_cell_v2(
    row: dict[str, str],
    node_id: int,
    summary_v2_m_s: float,
) -> None:
    # Phase 2 projects endpoint-local V2 by direct endpoint-coordinate cell assignment.
    expected_cell_id = _expected_endpoint_cell_id(node_id)
    expected_v2_m_s = _expected_endpoint_v2_m_s(node_id)

    assert int(row[_cell_id_column(row)]) == expected_cell_id
    assert float(row['v2_m_s']) == pytest.approx(
        expected_v2_m_s,
        abs=SYNTHETIC_CELL_V2_TOLERANCE_M_S,
    )
    assert float(row['v2_m_s']) != pytest.approx(
        summary_v2_m_s,
        abs=SYNTHETIC_CELL_V2_TOLERANCE_M_S,
    )


def _assert_endpoint_row_matches_known_truth(
    row: dict[str, str],
    node_id: int,
) -> None:
    expected_wcor_ms = expected_cell_wcor_s_for_node(node_id) * 1000.0

    assert float(row['t1_ms']) == pytest.approx(
        expected_cell_t1_s_for_node(node_id) * 1000.0,
        abs=SYNTHETIC_T1_TOLERANCE_MS,
    )
    assert float(row['v1_m_s']) == pytest.approx(SYNTHETIC_V1_M_S)
    assert float(row['sh1_weathering_thickness_m']) == pytest.approx(
        expected_cell_sh1_m_for_node(node_id),
        abs=SYNTHETIC_SH1_TOLERANCE_M,
    )
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
    assert row['static_status'] == 'ok'


def _assert_npz_endpoint_matches_csv(
    table: dict[str, np.ndarray],
    prefix: str,
    row: dict[str, str],
    index: int,
) -> None:
    node_column = f'{prefix}_node_id'
    cell_column = f'{prefix}_v2_cell_id'

    assert str(table[f'{prefix}_endpoint_key'][index]) == row[f'{prefix}_endpoint_key']
    assert int(table[f'{prefix}_id'][index]) == int(row[f'{prefix}_id'])
    assert int(table[node_column][index]) == int(row[node_column])
    assert int(table[cell_column][index]) == int(row[cell_column])
    assert str(table[f'{prefix}_v2_status'][index]) == row['v2_status']
    assert float(table[f'{prefix}_t1_s'][index]) * 1000.0 == pytest.approx(
        float(row['t1_ms'])
    )
    assert float(table[f'{prefix}_v1_m_s'][index]) == pytest.approx(
        float(row['v1_m_s'])
    )
    assert float(table[f'{prefix}_v2_m_s'][index]) == pytest.approx(
        float(row['v2_m_s'])
    )
    assert float(table[f'{prefix}_sh1_m'][index]) == pytest.approx(
        float(row['sh1_weathering_thickness_m'])
    )
    assert (
        float(table[f'{prefix}_weathering_correction_s'][index]) * 1000.0
    ) == pytest.approx(float(row['weathering_correction_ms']))
    assert float(table[f'{prefix}_total_static_s'][index]) * 1000.0 == pytest.approx(
        float(row['total_static_ms'])
    )
    assert (
        float(table[f'{prefix}_total_applied_shift_s'][index]) * 1000.0
    ) == pytest.approx(float(row['total_applied_shift_ms']))
    assert str(table[f'{prefix}_static_status'][index]) == row['static_status']


def _all_endpoint_rows(
    artifacts: dict[str, Any],
) -> list[tuple[dict[str, str], int]]:
    source = [
        (row, int(row['source_node_id']))
        for row in artifacts['source_rows']
    ]
    receiver = [
        (row, int(row['receiver_node_id']))
        for row in artifacts['receiver_rows']
    ]
    return source + receiver


def _row_by_node(
    rows: list[dict[str, str]],
    *,
    node_column: str,
    node_id: int,
) -> dict[str, str]:
    matches = [row for row in rows if int(row[node_column]) == node_id]
    assert len(matches) == 1
    return matches[0]


def _cell_id_column(row: dict[str, str]) -> str:
    if 'source_v2_cell_id' in row:
        return 'source_v2_cell_id'
    return 'receiver_v2_cell_id'


def _expected_endpoint_cell_id(node_id: int) -> int:
    return int(SYNTHETIC_CELL_NODE_CELL_ID[int(node_id)])


def _expected_endpoint_v2_m_s(node_id: int) -> float:
    return float(SYNTHETIC_CELL_V2_M_S[_expected_endpoint_cell_id(node_id)])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))
