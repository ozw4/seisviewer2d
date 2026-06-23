from __future__ import annotations

import numpy as np
import pytest

from app.api.schemas import RefractionStaticRefractorCellRequest
from app.statics.refraction.application.core_options import (
    refractor_cell_options_from_request,
)
from seis_statics.refraction.cell_grid import (
    assign_observation_midpoint_cells,
    assign_points_to_refraction_cells,
    build_refraction_cell_grid,
    compute_source_receiver_midpoints,
)


def _cell_request(**overrides: object) -> RefractionStaticRefractorCellRequest:
    payload: dict[str, object] = {
        'number_of_cell_x': 3,
        'size_of_cell_x_m': 10.0,
        'x_coordinate_origin_m': 0.0,
        'number_of_cell_y': 1,
        'size_of_cell_y_m': None,
        'y_coordinate_origin_m': 0.0,
        'assignment_mode': 'midpoint',
        'outside_grid_policy': 'reject',
        'min_observations_per_cell': 5,
        'velocity_smoothing_weight': 0.0,
        'smoothing_reference_distance_m': None,
    }
    payload.update(overrides)
    return RefractionStaticRefractorCellRequest.model_validate(payload)


def _cell_options(**overrides: object):
    return refractor_cell_options_from_request(_cell_request(**overrides))


def test_cell_grid_builds_1d_line_cells() -> None:
    grid = build_refraction_cell_grid(
        _cell_options(
            number_of_cell_x=4,
            size_of_cell_x_m=25.0,
            x_coordinate_origin_m=100.0,
        )
    )

    np.testing.assert_array_equal(grid.cell_id, np.asarray([0, 1, 2, 3]))
    np.testing.assert_array_equal(grid.ix, np.asarray([0, 1, 2, 3]))
    np.testing.assert_array_equal(grid.iy, np.asarray([0, 0, 0, 0]))
    np.testing.assert_allclose(grid.x_min_m, [100.0, 125.0, 150.0, 175.0])
    np.testing.assert_allclose(grid.x_max_m, [125.0, 150.0, 175.0, 200.0])
    np.testing.assert_allclose(grid.x_center_m, [112.5, 137.5, 162.5, 187.5])
    assert np.isneginf(grid.y_min_m).all()
    assert np.isposinf(grid.y_max_m).all()
    np.testing.assert_allclose(grid.y_center_m, [0.0, 0.0, 0.0, 0.0])
    assert grid.number_of_cell_y == 1
    assert grid.size_of_cell_y_m == float('inf')


def test_line_2d_projected_requires_origin_and_azimuth() -> None:
    payload = _cell_request().model_dump(mode='json')
    payload['coordinate_mode'] = 'line_2d_projected'

    with pytest.raises(ValueError, match='line_origin_x_m'):
        RefractionStaticRefractorCellRequest.model_validate(payload)


def test_line_2d_projected_rejects_number_of_cell_y_greater_than_one() -> None:
    payload = _cell_request().model_dump(mode='json')
    payload.update(
        {
            'coordinate_mode': 'line_2d_projected',
            'line_origin_x_m': 1000.0,
            'line_origin_y_m': 2000.0,
            'line_azimuth_deg': 45.0,
            'number_of_cell_y': 2,
            'size_of_cell_y_m': 25.0,
        }
    )

    with pytest.raises(ValueError, match='number_of_cell_y must be 1'):
        RefractionStaticRefractorCellRequest.model_validate(payload)


def test_cell_grid_builds_2d_cells_with_row_major_ids() -> None:
    grid = build_refraction_cell_grid(
        _cell_options(
            number_of_cell_x=3,
            size_of_cell_x_m=10.0,
            x_coordinate_origin_m=100.0,
            number_of_cell_y=2,
            size_of_cell_y_m=5.0,
            y_coordinate_origin_m=200.0,
        )
    )

    np.testing.assert_array_equal(grid.cell_id, np.asarray([0, 1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(grid.ix, np.asarray([0, 1, 2, 0, 1, 2]))
    np.testing.assert_array_equal(grid.iy, np.asarray([0, 0, 0, 1, 1, 1]))
    np.testing.assert_allclose(
        grid.x_min_m,
        [100.0, 110.0, 120.0, 100.0, 110.0, 120.0],
    )
    np.testing.assert_allclose(
        grid.y_min_m,
        [200.0, 200.0, 200.0, 205.0, 205.0, 205.0],
    )
    np.testing.assert_allclose(
        grid.y_center_m,
        [202.5, 202.5, 202.5, 207.5, 207.5, 207.5],
    )


def test_assign_points_to_cells_uses_half_open_intervals() -> None:
    grid = build_refraction_cell_grid(
        _cell_options(number_of_cell_y=2, size_of_cell_y_m=10.0)
    )

    assignment = assign_points_to_refraction_cells(
        grid,
        x_m=np.asarray([0.0, 9.999, 10.0, 10.0, 29.999]),
        y_m=np.asarray([0.0, 9.999, 0.0, 10.0, 19.999]),
    )

    assert assignment.inside_grid_mask.tolist() == [True] * 5
    np.testing.assert_array_equal(
        assignment.cell_id,
        np.asarray([0, 0, 1, 4, 5]),
    )
    np.testing.assert_array_equal(assignment.ix, np.asarray([0, 0, 1, 1, 2]))
    np.testing.assert_array_equal(assignment.iy, np.asarray([0, 0, 0, 1, 1]))


def test_assign_points_includes_final_boundary_with_tolerance() -> None:
    grid = build_refraction_cell_grid(
        _cell_options(number_of_cell_y=2, size_of_cell_y_m=10.0)
    )

    assignment = assign_points_to_refraction_cells(
        grid,
        x_m=np.asarray([30.0, np.nextafter(30.0, np.inf)]),
        y_m=np.asarray([20.0, np.nextafter(20.0, np.inf)]),
    )

    assert assignment.inside_grid_mask.tolist() == [True, True]
    np.testing.assert_array_equal(assignment.cell_id, np.asarray([5, 5]))
    np.testing.assert_array_equal(assignment.ix, np.asarray([2, 2]))
    np.testing.assert_array_equal(assignment.iy, np.asarray([1, 1]))


def test_assign_points_marks_outside_grid() -> None:
    grid = build_refraction_cell_grid(
        _cell_options(number_of_cell_y=2, size_of_cell_y_m=10.0)
    )

    assignment = assign_points_to_refraction_cells(
        grid,
        x_m=np.asarray([-1.0, 0.0, 30.001, 0.0, np.nan, 0.0, 5.0]),
        y_m=np.asarray([0.0, -1.0, 0.0, 20.001, 0.0, np.nan, 5.0]),
    )

    assert assignment.inside_grid_mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    np.testing.assert_array_equal(
        assignment.cell_id,
        np.asarray([-1, -1, -1, -1, -1, -1, 0]),
    )
    np.testing.assert_array_equal(
        assignment.ix,
        np.asarray([-1, 0, -1, 0, -1, 0, 0]),
    )
    np.testing.assert_array_equal(
        assignment.iy,
        np.asarray([0, -1, 0, -1, 0, -1, 0]),
    )
    assert assignment.qc['n_points'] == 7
    assert assignment.qc['n_inside_grid'] == 1
    assert assignment.qc['n_outside_grid'] == 6
    assert assignment.qc['inside_grid_fraction'] == pytest.approx(1.0 / 7.0)


def test_assign_observation_midpoint_cells() -> None:
    grid = build_refraction_cell_grid(
        _cell_options(number_of_cell_x=4, size_of_cell_x_m=10.0)
    )

    source_x_m = np.asarray([0.0, 10.0, 25.0])
    source_y_m = np.asarray([100.0, 100.0, 100.0])
    receiver_x_m = np.asarray([10.0, 30.0, 35.0])
    receiver_y_m = np.asarray([300.0, 300.0, 300.0])

    midpoint_x_m, midpoint_y_m = compute_source_receiver_midpoints(
        source_x_m=source_x_m,
        source_y_m=source_y_m,
        receiver_x_m=receiver_x_m,
        receiver_y_m=receiver_y_m,
    )
    midpoint_assignment = assign_observation_midpoint_cells(
        grid,
        source_x_m=source_x_m,
        source_y_m=source_y_m,
        receiver_x_m=receiver_x_m,
        receiver_y_m=receiver_y_m,
    )
    point_assignment = assign_points_to_refraction_cells(
        grid,
        x_m=midpoint_x_m,
        y_m=midpoint_y_m,
    )

    np.testing.assert_allclose(midpoint_x_m, [5.0, 20.0, 30.0])
    np.testing.assert_allclose(midpoint_y_m, [200.0, 200.0, 200.0])
    np.testing.assert_array_equal(midpoint_assignment.cell_id, [0, 2, 3])
    np.testing.assert_array_equal(
        midpoint_assignment.cell_id,
        point_assignment.cell_id,
    )
    np.testing.assert_array_equal(
        midpoint_assignment.inside_grid_mask,
        point_assignment.inside_grid_mask,
    )


def test_cell_assignment_qc_counts_active_and_inactive_cells() -> None:
    grid = build_refraction_cell_grid(
        _cell_options(number_of_cell_x=4, size_of_cell_x_m=10.0)
    )

    assignment = assign_points_to_refraction_cells(
        grid,
        x_m=np.asarray([1.0, 2.0, 11.0, 31.0]),
        y_m=np.asarray([0.0, 1.0, 2.0, 3.0]),
    )

    assert assignment.qc['n_points'] == 4
    assert assignment.qc['n_inside_grid'] == 4
    assert assignment.qc['n_outside_grid'] == 0
    assert assignment.qc['inside_grid_fraction'] == pytest.approx(1.0)
    assert assignment.qc['x_min_m'] == pytest.approx(1.0)
    assert assignment.qc['x_max_m'] == pytest.approx(31.0)
    assert assignment.qc['y_min_m'] == pytest.approx(0.0)
    assert assignment.qc['y_max_m'] == pytest.approx(3.0)
    assert assignment.qc['active_cell_count'] == 3
    assert assignment.qc['inactive_cell_count'] == 1
    assert assignment.qc['min_points_per_active_cell'] == 1
    assert assignment.qc['median_points_per_active_cell'] == pytest.approx(1.0)
    assert assignment.qc['max_points_per_active_cell'] == 2
