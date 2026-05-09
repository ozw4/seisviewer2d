from __future__ import annotations

from dataclasses import fields
import subprocess
import sys

import numpy as np

from app.tests.fixtures.refraction_synthetic import (
    SyntheticRefractionCellDataset,
    make_clean_2d_cell_refraction_dataset,
    make_clean_3d_cell_refraction_dataset,
    make_low_fold_empty_cell_refraction_dataset,
    make_outlier_refraction_dataset,
    make_rotated_2d_line_refraction_dataset,
    make_v2_spike_smoothing_dataset,
)


def test_synthetic_clean_2d_builder_is_deterministic() -> None:
    first = make_clean_2d_cell_refraction_dataset(seed=42, noise_std_s=0.001)
    second = make_clean_2d_cell_refraction_dataset(seed=42, noise_std_s=0.001)

    _assert_dataset_equal(first, second)


def test_synthetic_clean_3d_builder_is_deterministic() -> None:
    first = make_clean_3d_cell_refraction_dataset(seed=42, noise_std_s=0.001)
    second = make_clean_3d_cell_refraction_dataset(seed=42, noise_std_s=0.001)

    _assert_dataset_equal(first, second)


def test_synthetic_fixture_pick_times_match_known_truth_without_noise() -> None:
    dataset = make_clean_2d_cell_refraction_dataset(
        seed=7,
        noise_std_s=0.0,
        outlier_fraction=0.0,
    )

    expected_pick_time_s = (
        dataset.true_source_t1_s
        + dataset.true_receiver_t1_s
        + dataset.offset_m / dataset.true_midpoint_v2_m_s
    )

    np.testing.assert_allclose(dataset.pick_time_s, expected_pick_time_s, atol=1.0e-12)
    np.testing.assert_allclose(dataset.true_noise_s, 0.0, atol=0.0)
    assert not bool(np.any(dataset.outlier_mask))


def test_synthetic_fixture_known_sh1_t1_wcor_are_consistent() -> None:
    dataset = make_clean_3d_cell_refraction_dataset(seed=9)

    source_sh1_m = _sh1_from_t1(
        t1_s=dataset.true_source_endpoint_t1_s,
        v1_m_s=dataset.true_v1_m_s,
        v2_m_s=dataset.true_source_endpoint_v2_m_s,
    )
    receiver_sh1_m = _sh1_from_t1(
        t1_s=dataset.true_receiver_endpoint_t1_s,
        v1_m_s=dataset.true_v1_m_s,
        v2_m_s=dataset.true_receiver_endpoint_v2_m_s,
    )
    source_wcor_s = _wcor_from_sh1(
        sh1_m=dataset.true_source_endpoint_sh1_m,
        v1_m_s=dataset.true_v1_m_s,
        v2_m_s=dataset.true_source_endpoint_v2_m_s,
    )
    receiver_wcor_s = _wcor_from_sh1(
        sh1_m=dataset.true_receiver_endpoint_sh1_m,
        v1_m_s=dataset.true_v1_m_s,
        v2_m_s=dataset.true_receiver_endpoint_v2_m_s,
    )

    np.testing.assert_allclose(source_sh1_m, dataset.true_source_endpoint_sh1_m)
    np.testing.assert_allclose(receiver_sh1_m, dataset.true_receiver_endpoint_sh1_m)
    np.testing.assert_allclose(source_wcor_s, dataset.true_source_endpoint_wcor_s)
    np.testing.assert_allclose(receiver_wcor_s, dataset.true_receiver_endpoint_wcor_s)
    np.testing.assert_allclose(
        dataset.true_source_endpoint_static_s,
        dataset.true_source_endpoint_wcor_s,
    )
    np.testing.assert_allclose(
        dataset.true_receiver_endpoint_static_s,
        dataset.true_receiver_endpoint_wcor_s,
    )


def test_synthetic_rotated_2d_builder_generates_projected_line_mode() -> None:
    dataset = make_rotated_2d_line_refraction_dataset(
        seed=11,
        line_origin_x_m=1200.0,
        line_origin_y_m=800.0,
        line_azimuth_deg=42.0,
    )

    assert dataset.coordinate_mode == 'line_2d_projected'
    assert dataset.source_inline_m is not None
    assert dataset.receiver_inline_m is not None
    projected_source_inline = _project_inline(
        x_m=dataset.source_x_m,
        y_m=dataset.source_y_m,
        line_origin_x_m=dataset.line_origin_x_m,
        line_origin_y_m=dataset.line_origin_y_m,
        line_azimuth_deg=dataset.line_azimuth_deg,
    )
    projected_receiver_inline = _project_inline(
        x_m=dataset.receiver_x_m,
        y_m=dataset.receiver_y_m,
        line_origin_x_m=dataset.line_origin_x_m,
        line_origin_y_m=dataset.line_origin_y_m,
        line_azimuth_deg=dataset.line_azimuth_deg,
    )
    expected_ix = np.floor(
        0.5 * (dataset.source_inline_m + dataset.receiver_inline_m)
        / dataset.cell_size_x_m
    ).astype(np.int64)

    np.testing.assert_allclose(projected_source_inline, dataset.source_inline_m)
    np.testing.assert_allclose(projected_receiver_inline, dataset.receiver_inline_m)
    np.testing.assert_array_equal(dataset.true_cell_ix_for_pick, expected_ix)
    np.testing.assert_array_equal(dataset.true_cell_iy_for_pick, 0)


def test_synthetic_required_special_builders_mark_expected_rows() -> None:
    low_fold = make_low_fold_empty_cell_refraction_dataset(seed=3)
    outlier = make_outlier_refraction_dataset(seed=3)
    spike = make_v2_spike_smoothing_dataset(seed=3)

    assert low_fold.cell_observation_count[1] == 2
    assert low_fold.cell_observation_count[2] == 0
    assert int(np.count_nonzero(outlier.outlier_mask)) > 0
    assert spike.true_cell_v2_m_s[0, 2] > spike.true_cell_v2_m_s[0, 1]
    assert spike.true_cell_v2_m_s[0, 2] > spike.true_cell_v2_m_s[0, 3]


def test_synthetic_fixture_import_is_dependency_light() -> None:
    script = """
import sys
import app.tests.fixtures.refraction_synthetic as fixture
fixture.make_clean_2d_cell_refraction_dataset()
forbidden_exact = {'app.main', 'segyio', 'torch'}
forbidden_prefix = (
    'app.api',
    'app.services',
    'app.trace_store',
    'app.utils',
    'app.web',
)
loaded = [
    name for name in sys.modules
    if name in forbidden_exact or name.startswith(forbidden_prefix)
]
if loaded:
    raise SystemExit(','.join(sorted(loaded)))
"""

    result = subprocess.run(
        [sys.executable, '-c', script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def _assert_dataset_equal(
    first: SyntheticRefractionCellDataset,
    second: SyntheticRefractionCellDataset,
) -> None:
    for field in fields(first):
        first_value = getattr(first, field.name)
        second_value = getattr(second, field.name)
        if isinstance(first_value, np.ndarray):
            np.testing.assert_array_equal(first_value, second_value)
        elif first_value is None:
            assert second_value is None
        else:
            assert first_value == second_value


def _sh1_from_t1(
    *,
    t1_s: np.ndarray,
    v1_m_s: float,
    v2_m_s: np.ndarray,
) -> np.ndarray:
    return t1_s * v1_m_s * v2_m_s / np.sqrt(v2_m_s**2 - v1_m_s**2)


def _wcor_from_sh1(
    *,
    sh1_m: np.ndarray,
    v1_m_s: float,
    v2_m_s: np.ndarray,
) -> np.ndarray:
    return sh1_m * ((1.0 / v2_m_s) - (1.0 / v1_m_s))


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
    dx = x_m - float(line_origin_x_m)
    dy = y_m - float(line_origin_y_m)
    return np.round(dx * np.sin(azimuth_rad) + dy * np.cos(azimuth_rad), decimals=9)
