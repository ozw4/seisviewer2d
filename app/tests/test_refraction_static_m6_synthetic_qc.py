from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from app.services.refraction_static_artifacts import (
    REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
)
from app.tests._refraction_static_synthetic import (
    SYNTHETIC_CELL_NODE_T1_S,
    SYNTHETIC_CELL_V2_M_S,
)
from app.tests.fixtures.refraction_m6_qc import (
    LINE_AZIMUTH_DEG,
    LINE_ORIGIN_X_M,
    LINE_ORIGIN_Y_M,
    SIGN_CONVENTION,
    TRUE_3D_CELL_V2_M_S,
    projected_inline_crossline,
    run_field_component_case,
    write_grid_3d_cell_case,
    write_line_projected_one_layer_case,
    write_line_projected_three_layer_case,
)

TIME_ATOL_S = 1.0e-8
VELOCITY_ATOL_M_S = 1.0e-5
PROJECTION_ATOL_M = 1.0e-9


def test_synthetic_first_break_fit_qc_known_truth(tmp_path: Path) -> None:
    case = write_line_projected_one_layer_case(tmp_path / 'line-one-layer')

    with np.load(case.paths.refraction_first_break_fit_qc_npz, allow_pickle=False) as qc:
        trace_index = qc['sorted_trace_index']
        expected_inline, expected_crossline = projected_inline_crossline(
            x_m=0.5
            * (
                case.input_model.source_x_m_sorted[trace_index]
                + case.input_model.receiver_x_m_sorted[trace_index]
            ),
            y_m=0.5
            * (
                case.input_model.source_y_m_sorted[trace_index]
                + case.input_model.receiver_y_m_sorted[trace_index]
            ),
            line_origin_x_m=LINE_ORIGIN_X_M,
            line_origin_y_m=LINE_ORIGIN_Y_M,
            line_azimuth_deg=LINE_AZIMUTH_DEG,
        )

        np.testing.assert_allclose(
            qc['observed_first_break_time_s'],
            case.input_model.pick_time_s_sorted[trace_index],
        )
        np.testing.assert_allclose(
            qc['modeled_first_break_time_s'],
            _expected_one_layer_modeled_first_break_time_s(
                case=case,
                trace_index=trace_index,
                inline_m=expected_inline,
            ),
            atol=TIME_ATOL_S,
        )
        np.testing.assert_allclose(
            qc['residual_time_s'],
            qc['observed_first_break_time_s'] - qc['modeled_first_break_time_s'],
            atol=TIME_ATOL_S,
        )
        np.testing.assert_allclose(qc['inline_m'], expected_inline, atol=PROJECTION_ATOL_M)
        np.testing.assert_allclose(
            qc['crossline_m'],
            expected_crossline,
            atol=PROJECTION_ATOL_M,
        )
        assert set(qc['status'].astype(str).tolist()) == {'ok'}
        assert set(qc['rejection_reason'].astype(str).tolist()) == {'ok'}
        assert set(qc['sign_convention'].astype(str).tolist()) == {SIGN_CONVENTION}

    rejected = write_line_projected_one_layer_case(
        tmp_path / 'line-one-layer-rejected',
        robust_outliers=True,
    )
    with np.load(
        rejected.paths.refraction_first_break_fit_qc_npz,
        allow_pickle=False,
    ) as qc:
        rows_by_trace = {
            int(trace_index): index
            for index, trace_index in enumerate(qc['sorted_trace_index'].tolist())
        }
        for trace_index in rejected.outlier_trace_indices.tolist():
            row_index = rows_by_trace[trace_index]
            assert qc['used_for_inversion'][row_index].item() is False
            assert str(qc['status'][row_index]) == 'rejected'
            assert str(qc['rejection_reason'][row_index]) == 'robust_outlier'

    rejected_payload = json.loads(
        rejected.paths.refraction_first_break_fit_qc_json.read_text(encoding='utf-8')
    )
    assert rejected_payload['rejected_count'] >= int(
        rejected.outlier_trace_indices.shape[0]
    )
    assert (
        rejected_payload['rejection_reason_counts']['robust_outlier']
        >= int(rejected.outlier_trace_indices.shape[0])
    )


def test_synthetic_reduced_time_qc_known_truth(tmp_path: Path) -> None:
    case = write_line_projected_one_layer_case(tmp_path / 'line-reduced-time')

    with np.load(case.paths.refraction_reduced_time_qc_npz, allow_pickle=False) as qc:
        expected = (
            qc['observed_first_break_time_s']
            - qc['offset_m'] / qc['reduction_velocity_m_s']
        )
        np.testing.assert_allclose(qc['reduced_time_s'], expected, atol=TIME_ATOL_S)
        np.testing.assert_allclose(qc['reduced_time_ms'], expected * 1000.0)
        assert set(qc['status'].astype(str).tolist()) == {'ok'}
        assert set(qc['layer_gate_kind'].astype(str).tolist()) == {'v2_t1'}
        assert str(qc['reduction_velocity_mode']) == 'layer_velocity'

    payload = json.loads(
        case.paths.refraction_reduced_time_qc_json.read_text(encoding='utf-8')
    )
    assert payload['formula'] == (
        'reduced_time_s = observed_first_break_time_s - '
        'offset_m / reduction_velocity_m_s'
    )
    assert payload['status_counts'] == {'ok': int(case.result.row_distance_m.shape[0])}


def test_synthetic_line_profile_qc_known_truth_three_layer(tmp_path: Path) -> None:
    case = write_line_projected_three_layer_case(tmp_path / 'line-three-layer')
    dataset = case.dataset

    payload = json.loads(case.line_profile_qc_json.read_text(encoding='utf-8'))
    assert payload['status'] == 'available'
    assert payload['coordinate_mode'] == 'line_2d_projected'

    rows = _read_csv(case.line_profile_qc_combined_csv)
    assert rows
    rows_by_key = {
        (row['endpoint_kind'], row['endpoint_key']): row
        for row in rows
    }
    for index, endpoint_id in enumerate(dataset.source_endpoint_id.tolist()):
        row = rows_by_key[('source', str(int(endpoint_id)))]
        assert float(row['inline_m']) == pytest.approx(
            float(dataset.source_endpoint_inline_m[index]),
            abs=PROJECTION_ATOL_M,
        )
        assert float(row['crossline_m']) == pytest.approx(0.0, abs=PROJECTION_ATOL_M)
        assert float(row['t1_ms']) == pytest.approx(
            float(dataset.true_source_endpoint_t1_s[index]) * 1000.0,
            abs=1.0e-5,
        )
        assert float(row['t2_ms']) == pytest.approx(
            float(dataset.true_source_endpoint_t2_s[index]) * 1000.0,
            abs=1.0e-5,
        )
        assert float(row['t3_ms']) == pytest.approx(
            float(dataset.true_source_endpoint_t3_s[index]) * 1000.0,
            abs=1.0e-5,
        )
        assert float(row['sh1_m']) == pytest.approx(
            float(dataset.true_source_endpoint_sh1_m[index]),
            abs=1.0e-5,
        )
        assert float(row['sh2_m']) == pytest.approx(
            float(dataset.true_source_endpoint_sh2_m[index]),
            abs=1.0e-5,
        )
        assert float(row['sh3_m']) == pytest.approx(
            float(dataset.true_source_endpoint_sh3_m[index]),
            abs=1.0e-5,
        )
        assert float(row['weathering_correction_ms']) == pytest.approx(
            float(dataset.true_source_endpoint_wcor_s[index]) * 1000.0,
            abs=1.0e-5,
        )
        assert row['static_status'] == 'ok'

    for index, endpoint_id in enumerate(dataset.receiver_endpoint_id.tolist()):
        row = rows_by_key[('receiver', str(int(endpoint_id)))]
        assert float(row['inline_m']) == pytest.approx(
            float(dataset.receiver_endpoint_inline_m[index]),
            abs=PROJECTION_ATOL_M,
        )
        assert float(row['weathering_correction_ms']) == pytest.approx(
            float(dataset.true_receiver_endpoint_wcor_s[index]) * 1000.0,
            abs=1.0e-5,
        )
        assert row['static_status'] == 'ok'


def test_synthetic_grid_map_qc_known_truth_cell_v2(tmp_path: Path) -> None:
    case = write_grid_3d_cell_case(tmp_path / 'grid-3d-cell')
    dataset = case.dataset

    with np.load(case.paths.refraction_grid_map_qc_npz, allow_pickle=False) as qc:
        for cell_id, true_v2 in enumerate(TRUE_3D_CELL_V2_M_S.ravel().tolist()):
            row_index = int(np.flatnonzero(qc['cell_id'] == cell_id)[0])
            expected_ix = cell_id % TRUE_3D_CELL_V2_M_S.shape[1]
            expected_iy = cell_id // TRUE_3D_CELL_V2_M_S.shape[1]
            assert int(qc['cell_ix'][row_index]) == expected_ix
            assert int(qc['cell_iy'][row_index]) == expected_iy
            assert float(qc['cell_center_x_m'][row_index]) == pytest.approx(
                (expected_ix + 0.5) * dataset.cell_size_x_m
            )
            assert float(qc['cell_center_y_m'][row_index]) == pytest.approx(
                (expected_iy + 0.5) * dataset.cell_size_y_m
            )
            assert float(qc['velocity_m_s'][row_index]) == pytest.approx(
                true_v2,
                abs=VELOCITY_ATOL_M_S,
            )
            assert float(qc['slowness_s_per_m'][row_index]) == pytest.approx(
                1.0 / true_v2,
                abs=1.0e-12,
            )
            assert int(qc['n_observations'][row_index]) == int(
                dataset.cell_observation_count[cell_id]
            )
            assert str(qc['status'][row_index]) == 'solved'
            assert str(qc['status_reason'][row_index]) == 'solved'

    with np.load(case.paths.refraction_first_break_fit_qc_npz, allow_pickle=False) as fb:
        trace_index = fb['sorted_trace_index']
        np.testing.assert_array_equal(
            fb['cell_ix'].astype(np.int64),
            dataset.true_cell_ix_for_pick[trace_index],
        )
        np.testing.assert_array_equal(
            fb['cell_iy'].astype(np.int64),
            dataset.true_cell_iy_for_pick[trace_index],
        )

    payload = json.loads(case.paths.refraction_grid_map_qc_json.read_text())
    assert payload['grid']['coordinate_mode'] == 'grid_3d'
    assert payload['grid']['number_of_cell_y'] == TRUE_3D_CELL_V2_M_S.shape[0]


def test_synthetic_static_component_qc_known_truth_field_manual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = run_field_component_case(tmp_path, monkeypatch)
    with case.state.lock:
        job = dict(case.state.jobs['m6-field-component-qc'])
    assert job['status'] == 'done', job.get('message')

    dataset = case.fixture.dataset
    with np.load(
        case.job_dir / REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME,
        allow_pickle=False,
    ) as qc:
        np.testing.assert_allclose(
            qc['source_depth_shift_s'],
            dataset.expected_source_depth_shift_s[dataset.source_endpoint_index],
        )
        np.testing.assert_allclose(
            qc['uphole_shift_s'],
            dataset.expected_uphole_shift_s[dataset.source_endpoint_index],
        )
        np.testing.assert_allclose(
            qc['manual_static_shift_s'],
            (
                dataset.expected_source_manual_static_shift_s[
                    dataset.source_endpoint_index
                ]
                + dataset.expected_receiver_manual_static_shift_s[
                    dataset.receiver_endpoint_index
                ]
            ),
        )
        np.testing.assert_allclose(
            qc['computed_field_shift_s'],
            dataset.expected_trace_field_shift_s,
        )
        np.testing.assert_allclose(
            qc['applied_field_shift_s'],
            dataset.expected_trace_field_shift_s,
        )
        np.testing.assert_allclose(
            qc['final_trace_shift_s'],
            dataset.expected_final_trace_shift_s,
        )
        np.testing.assert_allclose(
            qc['endpoint_source_field_shift_s'][: dataset.source_depth_m.shape[0]],
            dataset.expected_source_field_shift_s,
        )
        np.testing.assert_allclose(
            qc['endpoint_receiver_field_shift_s'][dataset.source_depth_m.shape[0] :],
            dataset.expected_receiver_field_shift_s,
        )

    trace_rows = _read_csv(case.job_dir / 'refraction_static_component_qc_trace.csv')
    assert trace_rows[0]['apply_to_trace_shift'] == 'true'
    assert float(trace_rows[0]['final_trace_shift_ms']) == pytest.approx(
        dataset.expected_final_trace_shift_s[0] * 1000.0
    )

    endpoint_rows = _read_csv(
        case.job_dir / 'refraction_static_component_qc_endpoint.csv'
    )
    first_source = endpoint_rows[0]
    assert first_source['endpoint_kind'] == 'source'
    assert float(first_source['source_depth_correction_ms']) == pytest.approx(
        dataset.expected_source_depth_shift_s[0] * 1000.0
    )
    assert float(first_source['uphole_correction_ms']) == pytest.approx(
        dataset.expected_uphole_shift_s[0] * 1000.0
    )
    assert float(first_source['manual_static_shift_ms']) == pytest.approx(
        dataset.expected_source_manual_static_shift_s[0] * 1000.0
    )
    assert float(first_source['field_correction_ms']) == pytest.approx(
        dataset.expected_source_field_shift_s[0] * 1000.0
    )

    with np.load(case.job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME, allow_pickle=False) as solution:
        np.testing.assert_allclose(
            solution['trace_field_shift_s_sorted'],
            dataset.expected_trace_field_shift_s,
        )
        np.testing.assert_allclose(
            solution['final_trace_shift_s_sorted'],
            dataset.expected_final_trace_shift_s,
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _expected_one_layer_modeled_first_break_time_s(
    *,
    case,
    trace_index: np.ndarray,
    inline_m: np.ndarray,
) -> np.ndarray:
    refractor_cell = case.req.model.refractor_cell
    assert refractor_cell is not None
    source_node_id = case.input_model.source_node_id_sorted[trace_index].astype(
        np.int64,
    )
    receiver_node_id = case.input_model.receiver_node_id_sorted[trace_index].astype(
        np.int64,
    )
    cell_ix = np.floor(
        (inline_m - refractor_cell.x_coordinate_origin_m)
        / refractor_cell.size_of_cell_x_m
    ).astype(np.int64)
    assert np.all(cell_ix >= 0)
    assert np.all(cell_ix < SYNTHETIC_CELL_V2_M_S.shape[0])
    return (
        SYNTHETIC_CELL_NODE_T1_S[source_node_id]
        + SYNTHETIC_CELL_NODE_T1_S[receiver_node_id]
        + case.input_model.distance_m_sorted[trace_index] / SYNTHETIC_CELL_V2_M_S[cell_ix]
    )
