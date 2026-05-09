from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from app.services.refraction_static_cell_regularization import (
    augment_design_matrix_with_cell_smoothing,
    build_cell_slowness_smoothing_rows,
)


def test_cell_smoothing_builds_1d_neighbor_edges() -> None:
    rows = build_cell_slowness_smoothing_rows(
        active_cell_id=np.asarray([0, 1, 2], dtype=np.int64),
        velocity_smoothing_weight=0.5,
        smoothing_reference_distance_m=100.0,
        n_total_cells=3,
        number_of_cell_x=3,
        number_of_cell_y=1,
        n_parameters=3,
    )

    np.testing.assert_array_equal(rows.edge_cell_id, [[0, 1], [1, 2]])
    np.testing.assert_allclose(
        rows.matrix.toarray(),
        [
            [50.0, -50.0, 0.0],
            [0.0, 50.0, -50.0],
        ],
    )
    np.testing.assert_array_equal(rows.active_cell_neighbor_count, [1, 2, 1])
    assert rows.qc['n_cell_smoothing_edges'] == 2
    assert rows.qc['n_cell_smoothing_rows'] == 2
    assert rows.qc['smoothing_row_scale'] == pytest.approx(50.0)


def test_cell_smoothing_builds_2d_four_neighbor_edges() -> None:
    rows = build_cell_slowness_smoothing_rows(
        active_cell_id=np.asarray([0, 1, 2, 3], dtype=np.int64),
        velocity_smoothing_weight=1.0,
        smoothing_reference_distance_m=10.0,
        n_total_cells=4,
        number_of_cell_x=2,
        number_of_cell_y=2,
        n_parameters=4,
    )

    np.testing.assert_array_equal(
        rows.edge_cell_id,
        [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
        ],
    )
    np.testing.assert_array_equal(rows.active_cell_neighbor_count, [2, 2, 2, 2])
    assert rows.qc['active_cell_neighbor_count_min'] == 2
    assert rows.qc['active_cell_neighbor_count_median'] == pytest.approx(2.0)
    assert rows.qc['active_cell_neighbor_count_max'] == 2


def test_cell_smoothing_excludes_inactive_cells() -> None:
    rows = build_cell_slowness_smoothing_rows(
        active_cell_id=np.asarray([0, 2], dtype=np.int64),
        velocity_smoothing_weight=1.0,
        smoothing_reference_distance_m=100.0,
        n_total_cells=3,
        number_of_cell_x=3,
        number_of_cell_y=1,
        n_parameters=3,
    )

    assert rows.n_rows == 0
    assert rows.qc['n_cell_smoothing_edges'] == 0
    np.testing.assert_array_equal(rows.active_cell_neighbor_count, [0, 0])


def test_cell_smoothing_zero_weight_adds_no_rows() -> None:
    rows = build_cell_slowness_smoothing_rows(
        active_cell_id=np.asarray([0, 1], dtype=np.int64),
        velocity_smoothing_weight=0.0,
        smoothing_reference_distance_m=100.0,
        n_total_cells=2,
        number_of_cell_x=2,
        number_of_cell_y=1,
        n_parameters=2,
    )

    assert rows.n_rows == 0
    assert rows.matrix.shape == (0, 2)
    assert rows.qc['n_cell_smoothing_edges'] == 0
    assert rows.qc['smoothing_row_scale'] == pytest.approx(0.0)


def test_cell_smoothing_uses_median_row_distance_when_reference_omitted() -> None:
    rows = build_cell_slowness_smoothing_rows(
        active_cell_id=np.asarray([0, 1], dtype=np.int64),
        velocity_smoothing_weight=0.25,
        row_distance_m=np.asarray([100.0, 300.0, 500.0], dtype=np.float64),
        n_total_cells=2,
        number_of_cell_x=2,
        number_of_cell_y=1,
        n_parameters=2,
    )

    assert rows.reference_distance_m == pytest.approx(300.0)
    assert rows.qc['smoothing_reference_distance_m'] == pytest.approx(300.0)
    assert rows.qc['smoothing_row_scale'] == pytest.approx(75.0)


def test_cell_smoothing_augment_appends_rows() -> None:
    data_matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    rows = build_cell_slowness_smoothing_rows(
        active_cell_id=np.asarray([0, 1], dtype=np.int64),
        velocity_smoothing_weight=2.0,
        smoothing_reference_distance_m=5.0,
        n_total_cells=2,
        number_of_cell_x=2,
        number_of_cell_y=1,
        n_parameters=2,
    )

    matrix_aug, rhs_aug, n_rows = augment_design_matrix_with_cell_smoothing(
        sparse.csr_matrix(data_matrix),
        np.asarray([1.0, 2.0], dtype=np.float64),
        rows,
    )

    assert n_rows == 1
    np.testing.assert_allclose(
        matrix_aug.toarray(),
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, -10.0],
        ],
    )
    np.testing.assert_allclose(rhs_aug, [1.0, 2.0, 0.0])
