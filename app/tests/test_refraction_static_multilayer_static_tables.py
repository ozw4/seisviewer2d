from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.statics.refraction.artifacts import (
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)
from app.tests._refraction_multilayer_3layer_helpers import (
    STATIC_ATOL_S,
    THICKNESS_ATOL_M,
    compute_three_layer_workflow,
)


SIGN_CONVENTION = 'corrected(t) = raw(t - shift_s)'


@pytest.fixture
def three_layer_static_table_artifacts(tmp_path: Path) -> dict[str, Any]:
    job_dir = tmp_path / 'job'
    dataset, _input_model, _model, workflow = compute_three_layer_workflow(
        job_dir=job_dir,
    )
    with np.load(
        job_dir / SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        allow_pickle=False,
    ) as data:
        table_arrays = {name: data[name].copy() for name in data.files}
    return {
        'dataset': dataset,
        'result': workflow.datum_result,
        'source_rows': _read_csv(job_dir / SOURCE_STATIC_TABLE_CSV_NAME),
        'receiver_rows': _read_csv(job_dir / RECEIVER_STATIC_TABLE_CSV_NAME),
        'table_arrays': table_arrays,
    }


def test_three_layer_source_table_has_t3_vsub_sh3_columns(
    three_layer_static_table_artifacts: dict[str, Any],
) -> None:
    dataset = three_layer_static_table_artifacts['dataset']
    rows = three_layer_static_table_artifacts['source_rows']

    assert len(rows) == int(dataset.source_endpoint_id.shape[0])
    assert {
        't1_ms',
        't2_ms',
        't3_ms',
        'v1_m_s',
        'v2_m_s',
        'v3_m_s',
        'vsub_m_s',
        'sh1_weathering_thickness_m',
        'sh2_weathering_thickness_m',
        'sh3_weathering_thickness_m',
        'total_weathering_thickness_m',
        'layer1_base_elevation_m',
        'layer2_base_elevation_m',
        'final_refractor_elevation_m',
        'weathering_correction_ms',
        'elevation_correction_ms',
        'total_static_ms',
        'total_applied_shift_ms',
        'solution_status',
        'weathering_status',
        'datum_status',
        'static_status',
        'sign_convention',
    } <= set(rows[0])
    assert float(rows[0]['t3_ms']) / 1000.0 == pytest.approx(
        dataset.true_source_endpoint_t3_s[0],
        abs=STATIC_ATOL_S,
    )
    assert float(rows[0]['vsub_m_s']) == pytest.approx(dataset.true_vsub_m_s)
    assert float(rows[0]['sh3_weathering_thickness_m']) == pytest.approx(
        dataset.true_source_endpoint_sh3_m[0],
        abs=THICKNESS_ATOL_M,
    )
    assert rows[0]['sign_convention'] == SIGN_CONVENTION


def test_three_layer_receiver_table_has_t3_vsub_sh3_columns(
    three_layer_static_table_artifacts: dict[str, Any],
) -> None:
    dataset = three_layer_static_table_artifacts['dataset']
    rows = three_layer_static_table_artifacts['receiver_rows']

    assert len(rows) == int(dataset.receiver_endpoint_id.shape[0])
    assert {'t3_ms', 'vsub_m_s', 'sh3_weathering_thickness_m'} <= set(rows[0])
    assert float(rows[0]['t3_ms']) / 1000.0 == pytest.approx(
        dataset.true_receiver_endpoint_t3_s[0],
        abs=STATIC_ATOL_S,
    )
    assert float(rows[0]['vsub_m_s']) == pytest.approx(dataset.true_vsub_m_s)
    assert float(rows[0]['sh3_weathering_thickness_m']) == pytest.approx(
        dataset.true_receiver_endpoint_sh3_m[0],
        abs=THICKNESS_ATOL_M,
    )
    assert rows[0]['sign_convention'] == SIGN_CONVENTION


def test_three_layer_static_table_total_thickness_and_final_refractor_are_consistent(
    three_layer_static_table_artifacts: dict[str, Any],
) -> None:
    dataset = three_layer_static_table_artifacts['dataset']
    table = three_layer_static_table_artifacts['table_arrays']

    _assert_endpoint_totals(
        three_layer_static_table_artifacts['source_rows'],
        surface_elevation_m=dataset.source_endpoint_elevation_m,
        sh1_m=dataset.true_source_endpoint_sh1_m,
        sh2_m=dataset.true_source_endpoint_sh2_m,
        sh3_m=dataset.true_source_endpoint_sh3_m,
        npz_total_m=table['source_total_weathering_thickness_m'],
    )
    _assert_endpoint_totals(
        three_layer_static_table_artifacts['receiver_rows'],
        surface_elevation_m=dataset.receiver_endpoint_elevation_m,
        sh1_m=dataset.true_receiver_endpoint_sh1_m,
        sh2_m=dataset.true_receiver_endpoint_sh2_m,
        sh3_m=dataset.true_receiver_endpoint_sh3_m,
        npz_total_m=table['receiver_total_weathering_thickness_m'],
    )
    assert str(table['sign_convention']) == SIGN_CONVENTION


def _assert_endpoint_totals(
    rows: list[dict[str, str]],
    *,
    surface_elevation_m: np.ndarray,
    sh1_m: np.ndarray,
    sh2_m: np.ndarray,
    sh3_m: np.ndarray,
    npz_total_m: np.ndarray,
) -> None:
    expected_total = sh1_m + sh2_m + sh3_m
    expected_layer1_base = surface_elevation_m - sh1_m
    expected_layer2_base = expected_layer1_base - sh2_m
    expected_final = surface_elevation_m - expected_total

    np.testing.assert_allclose(npz_total_m, expected_total, atol=THICKNESS_ATOL_M)
    for index, row in enumerate(rows):
        assert float(row['total_weathering_thickness_m']) == pytest.approx(
            expected_total[index],
            abs=THICKNESS_ATOL_M,
        )
        assert float(row['layer1_base_elevation_m']) == pytest.approx(
            expected_layer1_base[index],
            abs=THICKNESS_ATOL_M,
        )
        assert float(row['layer2_base_elevation_m']) == pytest.approx(
            expected_layer2_base[index],
            abs=THICKNESS_ATOL_M,
        )
        assert float(row['final_refractor_elevation_m']) == pytest.approx(
            expected_final[index],
            abs=THICKNESS_ATOL_M,
        )
        assert float(row['refractor_elevation_m']) == pytest.approx(
            expected_final[index],
            abs=THICKNESS_ATOL_M,
        )
        assert float(row['total_applied_shift_ms']) == pytest.approx(
            float(row['total_static_ms'])
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))
