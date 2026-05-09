from __future__ import annotations

import numpy as np
import pytest

from app.api.schemas import RefractionStaticRefractorCellRequest
from app.services.refraction_static_cell_coordinates import (
    effective_refraction_cell_grid_config,
    project_refraction_cell_coordinates,
    project_refraction_cell_points,
)
from app.services.refraction_static_cell_grid import (
    assign_observation_midpoint_cells,
    build_refraction_cell_grid,
)
from app.tests.fixtures.refraction_synthetic import (
    SyntheticRefractionCellDataset,
    make_rotated_2d_line_refraction_dataset,
)


def test_line_2d_projected_recovers_inline_coordinates_for_rotated_line() -> None:
    dataset = make_rotated_2d_line_refraction_dataset(
        seed=432,
        line_origin_x_m=1000.0,
        line_origin_y_m=2000.0,
        line_azimuth_deg=37.0,
    )
    assert dataset.source_endpoint_inline_m is not None
    assert dataset.receiver_endpoint_inline_m is not None

    source = project_refraction_cell_points(
        x_m=dataset.source_endpoint_x_m,
        y_m=dataset.source_endpoint_y_m,
        mode='line_2d_projected',
        line_origin_x_m=dataset.line_origin_x_m,
        line_origin_y_m=dataset.line_origin_y_m,
        line_azimuth_deg=dataset.line_azimuth_deg,
    )
    receiver = project_refraction_cell_points(
        x_m=dataset.receiver_endpoint_x_m,
        y_m=dataset.receiver_endpoint_y_m,
        mode='line_2d_projected',
        line_origin_x_m=dataset.line_origin_x_m,
        line_origin_y_m=dataset.line_origin_y_m,
        line_azimuth_deg=dataset.line_azimuth_deg,
    )

    np.testing.assert_allclose(source.x_m, dataset.source_endpoint_inline_m)
    np.testing.assert_allclose(receiver.x_m, dataset.receiver_endpoint_inline_m)
    np.testing.assert_allclose(source.y_m, 0.0, atol=0.0)
    np.testing.assert_allclose(receiver.y_m, 0.0, atol=0.0)
    np.testing.assert_allclose(source.projected_crossline_m, 0.0, atol=1.0e-9)
    np.testing.assert_allclose(receiver.projected_crossline_m, 0.0, atol=1.0e-9)
    assert source.qc['coordinate_mode'] == 'line_2d_projected'
    assert source.qc['line_azimuth_deg'] == pytest.approx(37.0)


def test_line_2d_projected_midpoint_cell_assignment_uses_inline_midpoint() -> None:
    dataset = make_rotated_2d_line_refraction_dataset(seed=432)
    assert dataset.source_inline_m is not None
    assert dataset.receiver_inline_m is not None
    request = _line_cell_request(dataset)
    grid = build_refraction_cell_grid(effective_refraction_cell_grid_config(request))
    projected = project_refraction_cell_coordinates(
        source_x_m=dataset.source_x_m,
        source_y_m=dataset.source_y_m,
        receiver_x_m=dataset.receiver_x_m,
        receiver_y_m=dataset.receiver_y_m,
        mode='line_2d_projected',
        line_origin_x_m=dataset.line_origin_x_m,
        line_origin_y_m=dataset.line_origin_y_m,
        line_azimuth_deg=dataset.line_azimuth_deg,
    )

    assignment = assign_observation_midpoint_cells(
        grid,
        source_x_m=projected.source_x_m,
        source_y_m=projected.source_y_m,
        receiver_x_m=projected.receiver_x_m,
        receiver_y_m=projected.receiver_y_m,
    )
    inline_midpoint_m = 0.5 * (dataset.source_inline_m + dataset.receiver_inline_m)
    expected_cell_id = np.floor(inline_midpoint_m / dataset.cell_size_x_m).astype(
        np.int64
    )

    np.testing.assert_allclose(assignment.x_m, inline_midpoint_m)
    np.testing.assert_allclose(assignment.y_m, 0.0, atol=0.0)
    np.testing.assert_array_equal(assignment.cell_id, expected_cell_id)
    np.testing.assert_array_equal(assignment.cell_id, dataset.true_cell_id_for_pick)
    assert bool(np.all(assignment.inside_grid_mask))


def test_line_2d_projected_requires_single_y_cell() -> None:
    dataset = make_rotated_2d_line_refraction_dataset(seed=432)
    payload = _line_cell_request(dataset).model_dump(mode='json')
    payload.update({'number_of_cell_y': 2, 'size_of_cell_y_m': 100.0})

    with pytest.raises(ValueError, match='number_of_cell_y must be 1'):
        RefractionStaticRefractorCellRequest.model_validate(payload)


@pytest.mark.parametrize(
    'missing_field',
    ['line_origin_x_m', 'line_origin_y_m', 'line_azimuth_deg'],
)
def test_line_2d_projected_requires_line_origin_and_azimuth(
    missing_field: str,
) -> None:
    dataset = make_rotated_2d_line_refraction_dataset(seed=432)
    payload = _line_cell_request(dataset).model_dump(mode='json')
    payload[missing_field] = None

    with pytest.raises(ValueError, match='line_origin_x_m'):
        RefractionStaticRefractorCellRequest.model_validate(payload)


def test_rotated_line_grid_3d_and_line_2d_projected_assignments_can_differ() -> None:
    dataset = make_rotated_2d_line_refraction_dataset(
        seed=432,
        line_origin_x_m=1000.0,
        line_origin_y_m=2000.0,
        line_azimuth_deg=37.0,
    )
    line_request = _line_cell_request(dataset)
    line_grid = build_refraction_cell_grid(
        effective_refraction_cell_grid_config(line_request)
    )
    line_projected = project_refraction_cell_coordinates(
        source_x_m=dataset.source_x_m,
        source_y_m=dataset.source_y_m,
        receiver_x_m=dataset.receiver_x_m,
        receiver_y_m=dataset.receiver_y_m,
        mode='line_2d_projected',
        line_origin_x_m=dataset.line_origin_x_m,
        line_origin_y_m=dataset.line_origin_y_m,
        line_azimuth_deg=dataset.line_azimuth_deg,
    )
    line_assignment = assign_observation_midpoint_cells(
        line_grid,
        source_x_m=line_projected.source_x_m,
        source_y_m=line_projected.source_y_m,
        receiver_x_m=line_projected.receiver_x_m,
        receiver_y_m=line_projected.receiver_y_m,
    )
    grid_3d_grid = build_refraction_cell_grid(
        _grid_3d_request_covering_rotated_line(dataset)
    )
    grid_3d_assignment = assign_observation_midpoint_cells(
        grid_3d_grid,
        source_x_m=dataset.source_x_m,
        source_y_m=dataset.source_y_m,
        receiver_x_m=dataset.receiver_x_m,
        receiver_y_m=dataset.receiver_y_m,
    )

    np.testing.assert_array_equal(line_assignment.cell_id, dataset.true_cell_id_for_pick)
    assert bool(np.all(grid_3d_assignment.inside_grid_mask))
    assert not np.array_equal(grid_3d_assignment.cell_id, line_assignment.cell_id)
    assert int(np.max(line_assignment.iy)) == 0
    assert int(np.max(grid_3d_assignment.iy)) > 0


def _line_cell_request(
    dataset: SyntheticRefractionCellDataset,
) -> RefractionStaticRefractorCellRequest:
    return RefractionStaticRefractorCellRequest.model_validate(
        {
            'number_of_cell_x': int(dataset.true_cell_v2_m_s.shape[1]),
            'size_of_cell_x_m': dataset.cell_size_x_m,
            'x_coordinate_origin_m': dataset.x_coordinate_origin_m,
            'number_of_cell_y': 1,
            'size_of_cell_y_m': None,
            'y_coordinate_origin_m': dataset.y_coordinate_origin_m,
            'assignment_mode': 'midpoint',
            'outside_grid_policy': 'reject',
            'coordinate_mode': 'line_2d_projected',
            'line_origin_x_m': dataset.line_origin_x_m,
            'line_origin_y_m': dataset.line_origin_y_m,
            'line_azimuth_deg': dataset.line_azimuth_deg,
            'min_observations_per_cell': 1,
            'velocity_smoothing_weight': 0.0,
            'smoothing_reference_distance_m': None,
        }
    )


def _grid_3d_request_covering_rotated_line(
    dataset: SyntheticRefractionCellDataset,
) -> RefractionStaticRefractorCellRequest:
    return RefractionStaticRefractorCellRequest.model_validate(
        {
            'number_of_cell_x': 3,
            'size_of_cell_x_m': dataset.cell_size_x_m,
            'x_coordinate_origin_m': dataset.line_origin_x_m,
            'number_of_cell_y': 3,
            'size_of_cell_y_m': dataset.cell_size_x_m,
            'y_coordinate_origin_m': dataset.line_origin_y_m,
            'assignment_mode': 'midpoint',
            'outside_grid_policy': 'reject',
            'coordinate_mode': 'grid_3d',
            'min_observations_per_cell': 1,
            'velocity_smoothing_weight': 0.0,
            'smoothing_reference_distance_m': None,
        }
    )
