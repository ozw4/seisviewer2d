from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields

import numpy as np
import pytest

from app.tests._refraction_multilayer_synthetic import (
    LAYER_KINDS,
    SYNTHETIC_MULTILAYER_V1_M_S,
    SYNTHETIC_MULTILAYER_V2_M_S,
    SYNTHETIC_MULTILAYER_V3_M_S,
    SYNTHETIC_MULTILAYER_VSUB_M_S,
    SyntheticMultiLayerRefractionDataset,
    V2_T1,
    V3_T2,
    VSUB_T3,
    make_2d_rotated_two_layer_refraction_dataset,
    make_2d_straight_three_layer_refraction_dataset,
    make_2d_straight_two_layer_refraction_dataset,
    make_3d_small_grid_three_layer_refraction_dataset,
    make_3d_small_grid_two_layer_refraction_dataset,
)

Builder = Callable[..., SyntheticMultiLayerRefractionDataset]

_BUILDERS: tuple[Builder, ...] = (
    make_2d_straight_two_layer_refraction_dataset,
    make_2d_rotated_two_layer_refraction_dataset,
    make_2d_straight_three_layer_refraction_dataset,
    make_3d_small_grid_two_layer_refraction_dataset,
    make_3d_small_grid_three_layer_refraction_dataset,
)


@pytest.mark.parametrize('builder', _BUILDERS)
def test_multilayer_synthetic_builders_are_deterministic(builder: Builder) -> None:
    first = builder(seed=17, noise_std_s=0.0005, outlier_fraction=0.08)
    second = builder(seed=17, noise_std_s=0.0005, outlier_fraction=0.08)

    _assert_dataset_equal(first, second)


def test_multilayer_synthetic_builders_cover_required_scenarios() -> None:
    datasets = [builder() for builder in _BUILDERS]

    assert [dataset.name for dataset in datasets] == [
        '2d_straight_two_layer',
        '2d_rotated_two_layer',
        '2d_straight_three_layer',
        '3d_small_grid_two_layer',
        '3d_small_grid_three_layer',
    ]
    assert [dataset.layer_count for dataset in datasets] == [2, 2, 3, 2, 3]
    assert datasets[1].coordinate_mode == 'line_2d_projected'
    assert datasets[3].coordinate_mode == 'grid_3d'
    assert datasets[4].coordinate_mode == 'grid_3d'
    assert np.ptp(datasets[3].source_endpoint_y_m) > 0.0
    assert np.ptp(datasets[4].receiver_endpoint_y_m) > 0.0

    for dataset in datasets:
        assert dataset.sorted_trace_index.size <= 64
        assert set(dataset.expected_layer_mask_by_kind) == set(LAYER_KINDS)
        for layer in dataset.enabled_layer_kinds:
            assert int(np.count_nonzero(dataset.expected_layer_mask_by_kind[layer])) > 0
        if dataset.layer_count == 2:
            assert not bool(np.any(dataset.expected_layer_mask_by_kind[VSUB_T3]))
        np.testing.assert_allclose(dataset.true_noise_s, 0.0, atol=0.0)
        assert not bool(np.any(dataset.outlier_mask))
        np.testing.assert_allclose(
            dataset.first_break_time_s,
            dataset.noiseless_first_break_time_s,
            atol=0.0,
        )


@pytest.mark.parametrize('builder', _BUILDERS)
def test_multilayer_synthetic_picks_match_forward_equations(
    builder: Builder,
) -> None:
    dataset = builder()
    expected = np.empty(dataset.first_break_time_s.shape, dtype=np.float64)

    v2_mask = dataset.expected_layer_mask_by_kind[V2_T1]
    expected[v2_mask] = (
        dataset.true_source_t1_s[v2_mask]
        + dataset.true_receiver_t1_s[v2_mask]
        + dataset.offset_m[v2_mask] / dataset.true_v2_m_s
    )
    v3_mask = dataset.expected_layer_mask_by_kind[V3_T2]
    expected[v3_mask] = (
        dataset.true_source_t2_s[v3_mask]
        + dataset.true_receiver_t2_s[v3_mask]
        + dataset.offset_m[v3_mask] / dataset.true_v3_m_s
    )
    vsub_mask = dataset.expected_layer_mask_by_kind[VSUB_T3]
    if bool(np.any(vsub_mask)):
        assert dataset.true_source_t3_s is not None
        assert dataset.true_receiver_t3_s is not None
        expected[vsub_mask] = (
            dataset.true_source_t3_s[vsub_mask]
            + dataset.true_receiver_t3_s[vsub_mask]
            + dataset.offset_m[vsub_mask] / dataset.true_vsub_m_s
        )

    np.testing.assert_allclose(dataset.noiseless_first_break_time_s, expected)
    np.testing.assert_allclose(dataset.first_break_time_s, expected)
    np.testing.assert_array_equal(dataset.pick_time_s, dataset.first_break_time_s)


@pytest.mark.parametrize('builder', _BUILDERS)
def test_multilayer_synthetic_truth_arrays_match_t1lsst_forward_terms(
    builder: Builder,
) -> None:
    dataset = builder()
    _assert_endpoint_truth_matches_t1lsst(dataset, endpoint='source')
    _assert_endpoint_truth_matches_t1lsst(dataset, endpoint='receiver')

    np.testing.assert_allclose(
        dataset.true_source_total_static_s,
        dataset.true_source_wcor_s,
    )
    np.testing.assert_allclose(
        dataset.true_receiver_total_static_s,
        dataset.true_receiver_wcor_s,
    )
    np.testing.assert_allclose(
        dataset.true_source_endpoint_total_static_s,
        dataset.true_source_endpoint_wcor_s,
    )
    np.testing.assert_allclose(
        dataset.true_receiver_endpoint_total_static_s,
        dataset.true_receiver_endpoint_wcor_s,
    )
    assert np.all(dataset.true_source_endpoint_wcor_s < 0.0)
    assert np.all(dataset.true_receiver_endpoint_wcor_s < 0.0)

    if dataset.layer_count == 2:
        assert dataset.true_source_endpoint_t3_s is None
        assert dataset.true_receiver_endpoint_sh3_m is None
    else:
        assert dataset.true_source_endpoint_t3_s is not None
        assert dataset.true_receiver_endpoint_sh3_m is not None


def test_multilayer_rotated_2d_builder_returns_line_projection_truth() -> None:
    dataset = make_2d_rotated_two_layer_refraction_dataset(
        line_origin_x_m=1200.0,
        line_origin_y_m=800.0,
        line_azimuth_deg=42.0,
    )

    assert dataset.coordinate_mode == 'line_2d_projected'
    assert dataset.source_endpoint_inline_m is not None
    assert dataset.receiver_endpoint_inline_m is not None
    assert dataset.source_inline_m is not None
    assert dataset.receiver_inline_m is not None

    source_inline = _project_inline(
        x_m=dataset.source_endpoint_x_m,
        y_m=dataset.source_endpoint_y_m,
        line_origin_x_m=dataset.line_origin_x_m,
        line_origin_y_m=dataset.line_origin_y_m,
        line_azimuth_deg=dataset.line_azimuth_deg,
    )
    receiver_inline = _project_inline(
        x_m=dataset.receiver_endpoint_x_m,
        y_m=dataset.receiver_endpoint_y_m,
        line_origin_x_m=dataset.line_origin_x_m,
        line_origin_y_m=dataset.line_origin_y_m,
        line_azimuth_deg=dataset.line_azimuth_deg,
    )

    np.testing.assert_allclose(source_inline, dataset.source_endpoint_inline_m)
    np.testing.assert_allclose(receiver_inline, dataset.receiver_endpoint_inline_m)
    np.testing.assert_allclose(
        dataset.source_inline_m,
        dataset.source_endpoint_inline_m[dataset.source_endpoint_index],
    )
    np.testing.assert_allclose(
        dataset.receiver_inline_m,
        dataset.receiver_endpoint_inline_m[dataset.receiver_endpoint_index],
    )


def test_multilayer_3d_builders_use_non_collinear_geometry() -> None:
    for dataset in (
        make_3d_small_grid_two_layer_refraction_dataset(),
        make_3d_small_grid_three_layer_refraction_dataset(),
    ):
        assert np.ptp(dataset.source_endpoint_y_m) > 0.0
        assert np.ptp(dataset.receiver_endpoint_y_m) > 0.0
        assert not np.allclose(
            dataset.offset_m,
            np.abs(dataset.receiver_x_m - dataset.source_x_m),
        )


def test_multilayer_optional_noise_and_outliers_are_deterministic() -> None:
    clean = make_2d_straight_three_layer_refraction_dataset()
    noisy = make_2d_straight_three_layer_refraction_dataset(
        seed=23,
        noise_std_s=0.0005,
        outlier_fraction=0.1,
    )
    repeated = make_2d_straight_three_layer_refraction_dataset(
        seed=23,
        noise_std_s=0.0005,
        outlier_fraction=0.1,
    )

    _assert_dataset_equal(noisy, repeated)
    np.testing.assert_allclose(
        noisy.noiseless_first_break_time_s,
        clean.noiseless_first_break_time_s,
    )
    np.testing.assert_allclose(
        noisy.first_break_time_s,
        noisy.noiseless_first_break_time_s + noisy.true_noise_s,
    )
    assert int(np.count_nonzero(noisy.outlier_mask)) == 5
    assert np.max(np.abs(noisy.true_noise_s[noisy.outlier_mask])) >= 0.050


@pytest.mark.parametrize('builder', _BUILDERS)
def test_multilayer_synthetic_returns_input_model_like_arrays(
    builder: Builder,
) -> None:
    dataset = builder()
    arrays = dataset.as_input_model_arrays()
    expected_keys = {
        'sorted_trace_index',
        'pick_time_s_sorted',
        'valid_pick_mask_sorted',
        'valid_observation_mask_sorted',
        'source_id_sorted',
        'receiver_id_sorted',
        'source_x_m_sorted',
        'source_y_m_sorted',
        'receiver_x_m_sorted',
        'receiver_y_m_sorted',
        'source_elevation_m_sorted',
        'receiver_elevation_m_sorted',
        'source_depth_m_sorted',
        'geometry_distance_m_sorted',
        'offset_m_sorted',
        'distance_m_sorted',
        'source_endpoint_key_sorted',
        'receiver_endpoint_key_sorted',
        'source_node_id_sorted',
        'receiver_node_id_sorted',
        'rejection_reason_sorted',
        'layer_kind_sorted',
    }

    assert set(arrays) == expected_keys
    assert arrays['source_depth_m_sorted'] is None
    n_traces = dataset.sorted_trace_index.shape
    for key, value in arrays.items():
        if value is None:
            continue
        assert value.shape == n_traces, key
    np.testing.assert_array_equal(
        arrays['pick_time_s_sorted'],
        dataset.first_break_time_s,
    )
    np.testing.assert_array_equal(arrays['layer_kind_sorted'], dataset.layer_kind)


def _assert_endpoint_truth_matches_t1lsst(
    dataset: SyntheticMultiLayerRefractionDataset,
    *,
    endpoint: str,
) -> None:
    sh1 = getattr(dataset, f'true_{endpoint}_endpoint_sh1_m')
    sh2 = getattr(dataset, f'true_{endpoint}_endpoint_sh2_m')
    sh3 = getattr(dataset, f'true_{endpoint}_endpoint_sh3_m')
    t1 = getattr(dataset, f'true_{endpoint}_endpoint_t1_s')
    t2 = getattr(dataset, f'true_{endpoint}_endpoint_t2_s')
    t3 = getattr(dataset, f'true_{endpoint}_endpoint_t3_s')
    wcor = getattr(dataset, f'true_{endpoint}_endpoint_wcor_s')

    np.testing.assert_allclose(
        t1,
        _forward_time_term(
            thicknesses=(sh1,),
            velocities=(
                SYNTHETIC_MULTILAYER_V1_M_S,
            ),
            refractor_velocity=SYNTHETIC_MULTILAYER_V2_M_S,
        ),
    )
    np.testing.assert_allclose(
        t2,
        _forward_time_term(
            thicknesses=(sh1, sh2),
            velocities=(
                SYNTHETIC_MULTILAYER_V1_M_S,
                SYNTHETIC_MULTILAYER_V2_M_S,
            ),
            refractor_velocity=SYNTHETIC_MULTILAYER_V3_M_S,
        ),
    )
    if dataset.layer_count == 3:
        assert sh3 is not None
        assert t3 is not None
        np.testing.assert_allclose(
            t3,
            _forward_time_term(
                thicknesses=(sh1, sh2, sh3),
                velocities=(
                    SYNTHETIC_MULTILAYER_V1_M_S,
                    SYNTHETIC_MULTILAYER_V2_M_S,
                    SYNTHETIC_MULTILAYER_V3_M_S,
                ),
                refractor_velocity=SYNTHETIC_MULTILAYER_VSUB_M_S,
            ),
        )
        expected_wcor = _layered_wcor(
            thicknesses=(sh1, sh2, sh3),
            velocities=(
                SYNTHETIC_MULTILAYER_V1_M_S,
                SYNTHETIC_MULTILAYER_V2_M_S,
                SYNTHETIC_MULTILAYER_V3_M_S,
            ),
            replacement_velocity=SYNTHETIC_MULTILAYER_VSUB_M_S,
        )
    else:
        expected_wcor = _layered_wcor(
            thicknesses=(sh1, sh2),
            velocities=(
                SYNTHETIC_MULTILAYER_V1_M_S,
                SYNTHETIC_MULTILAYER_V2_M_S,
            ),
            replacement_velocity=SYNTHETIC_MULTILAYER_V3_M_S,
        )
    np.testing.assert_allclose(wcor, expected_wcor)


def _forward_time_term(
    *,
    thicknesses: tuple[np.ndarray, ...],
    velocities: tuple[float, ...],
    refractor_velocity: float,
) -> np.ndarray:
    result = np.zeros(thicknesses[0].shape, dtype=np.float64)
    for thickness, velocity in zip(thicknesses, velocities, strict=True):
        result += (
            thickness
            * np.sqrt(refractor_velocity**2 - velocity**2)
            / (velocity * refractor_velocity)
        )
    return result


def _layered_wcor(
    *,
    thicknesses: tuple[np.ndarray, ...],
    velocities: tuple[float, ...],
    replacement_velocity: float,
) -> np.ndarray:
    result = np.zeros(thicknesses[0].shape, dtype=np.float64)
    for thickness, velocity in zip(thicknesses, velocities, strict=True):
        result += thickness * ((1.0 / replacement_velocity) - (1.0 / velocity))
    return result


def _project_inline(
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    line_origin_x_m: float | None,
    line_origin_y_m: float | None,
    line_azimuth_deg: float | None,
) -> np.ndarray:
    assert line_origin_x_m is not None
    assert line_origin_y_m is not None
    assert line_azimuth_deg is not None
    azimuth_rad = np.deg2rad(float(line_azimuth_deg))
    inline_unit_x = float(np.sin(azimuth_rad))
    inline_unit_y = float(np.cos(azimuth_rad))
    dx = np.asarray(x_m, dtype=np.float64) - float(line_origin_x_m)
    dy = np.asarray(y_m, dtype=np.float64) - float(line_origin_y_m)
    return dx * inline_unit_x + dy * inline_unit_y


def _assert_dataset_equal(
    first: SyntheticMultiLayerRefractionDataset,
    second: SyntheticMultiLayerRefractionDataset,
) -> None:
    for field in fields(first):
        _assert_value_equal(
            getattr(first, field.name),
            getattr(second, field.name),
            path=field.name,
        )


def _assert_value_equal(first: object, second: object, *, path: str) -> None:
    if isinstance(first, np.ndarray):
        assert isinstance(second, np.ndarray), path
        np.testing.assert_array_equal(first, second)
        return
    if isinstance(first, dict):
        assert isinstance(second, dict), path
        assert set(first) == set(second), path
        for key, value in first.items():
            _assert_value_equal(value, second[key], path=f'{path}.{key}')
        return
    if first is None:
        assert second is None, path
        return
    assert first == second, path
