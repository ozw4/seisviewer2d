from __future__ import annotations

import numpy as np
import pytest

from app.services.refraction_static_t1lsst import (
    RefractionT1LSSTError,
    compute_t1lsst_2layer_thicknesses,
    compute_t1lsst_2layer_thicknesses_with_status,
    compute_t1lsst_2layer_weathering_correction,
)
from app.tests._refraction_multilayer_synthetic import (
    SYNTHETIC_MULTILAYER_V1_M_S,
    SYNTHETIC_MULTILAYER_V2_M_S,
    SYNTHETIC_MULTILAYER_V3_M_S,
    make_2d_straight_two_layer_refraction_dataset,
)


def test_t1lsst_2layer_scalar_formula_matches_known_truth() -> None:
    v1_m_s = 800.0
    v2_m_s = 2400.0
    v3_m_s = 3600.0
    expected_sh1_m = np.asarray([10.0], dtype=np.float64)
    expected_sh2_m = np.asarray([20.0], dtype=np.float64)
    t1_s, t2_s = _forward_t1_t2_s(
        sh1_m=expected_sh1_m,
        sh2_m=expected_sh2_m,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
    )

    sh1_m, sh2_m = compute_t1lsst_2layer_thicknesses(
        t1_s=t1_s,
        t2_s=t2_s,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
    )

    np.testing.assert_allclose(sh1_m, expected_sh1_m)
    np.testing.assert_allclose(sh2_m, expected_sh2_m)


def test_t1lsst_2layer_wcor_is_negative_under_repo_shift_convention() -> None:
    sh1_m = np.asarray([10.0, 12.0], dtype=np.float64)
    sh2_m = np.asarray([20.0, 18.0], dtype=np.float64)

    wcor_s = compute_t1lsst_2layer_weathering_correction(
        sh1_m=sh1_m,
        sh2_m=sh2_m,
        v1_m_s=800.0,
        v2_m_s=2400.0,
        v3_m_s=3600.0,
    )

    expected = sh1_m * (1.0 / 3600.0 - 1.0 / 800.0) + sh2_m * (
        1.0 / 3600.0 - 1.0 / 2400.0
    )
    np.testing.assert_allclose(wcor_s, expected)
    assert np.all(wcor_s < 0.0)


def test_t1lsst_2layer_matches_synthetic_endpoint_truth() -> None:
    dataset = make_2d_straight_two_layer_refraction_dataset()

    for endpoint in ('source', 'receiver'):
        t1_s = getattr(dataset, f'true_{endpoint}_endpoint_t1_s')
        t2_s = getattr(dataset, f'true_{endpoint}_endpoint_t2_s')
        expected_sh1_m = getattr(dataset, f'true_{endpoint}_endpoint_sh1_m')
        expected_sh2_m = getattr(dataset, f'true_{endpoint}_endpoint_sh2_m')
        expected_wcor_s = getattr(dataset, f'true_{endpoint}_endpoint_wcor_s')

        result = compute_t1lsst_2layer_thicknesses_with_status(
            t1_s=t1_s,
            t2_s=t2_s,
            v1_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
            v2_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
            v3_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
        )
        wcor_s = compute_t1lsst_2layer_weathering_correction(
            sh1_m=result.sh1_m,
            sh2_m=result.sh2_m,
            v1_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
            v2_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
            v3_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
        )

        assert result.status.tolist() == ['ok'] * expected_sh1_m.size
        np.testing.assert_allclose(result.sh1_m, expected_sh1_m)
        np.testing.assert_allclose(result.sh2_m, expected_sh2_m)
        np.testing.assert_allclose(wcor_s, expected_wcor_s)
        assert np.all(wcor_s < 0.0)


def test_t1lsst_2layer_supports_vectorized_endpoint_local_velocities() -> None:
    v1_m_s = np.asarray([700.0, 800.0, 900.0], dtype=np.float64)
    v2_m_s = np.asarray([2200.0, 2500.0, 2800.0], dtype=np.float64)
    v3_m_s = np.asarray([3300.0, 3600.0, 4100.0], dtype=np.float64)
    expected_sh1_m = np.asarray([8.0, 10.0, 12.0], dtype=np.float64)
    expected_sh2_m = np.asarray([14.0, 18.0, 22.0], dtype=np.float64)
    t1_s, t2_s = _forward_t1_t2_s(
        sh1_m=expected_sh1_m,
        sh2_m=expected_sh2_m,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
    )

    result = compute_t1lsst_2layer_thicknesses_with_status(
        t1_s=t1_s,
        t2_s=t2_s,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
    )
    wcor_s = compute_t1lsst_2layer_weathering_correction(
        sh1_m=result.sh1_m,
        sh2_m=result.sh2_m,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
    )

    assert result.status.tolist() == ['ok', 'ok', 'ok']
    assert result.sh1_m.shape == expected_sh1_m.shape
    np.testing.assert_allclose(result.sh1_m, expected_sh1_m)
    np.testing.assert_allclose(result.sh2_m, expected_sh2_m)
    np.testing.assert_allclose(
        wcor_s,
        expected_sh1_m * (1.0 / v3_m_s - 1.0 / v1_m_s)
        + expected_sh2_m * (1.0 / v3_m_s - 1.0 / v2_m_s),
    )


def test_t1lsst_2layer_local_v2_greater_than_v3_statuses_endpoint() -> None:
    result = compute_t1lsst_2layer_thicknesses_with_status(
        t1_s=np.asarray([0.01, 0.01], dtype=np.float64),
        t2_s=np.asarray([0.02, 0.02], dtype=np.float64),
        v1_m_s=800.0,
        v2_m_s=np.asarray([2400.0, 3800.0], dtype=np.float64),
        v3_m_s=3600.0,
    )

    assert result.status.tolist() == ['ok', 'invalid_velocity_order']
    assert np.isfinite(result.sh1_m[0])
    assert np.isfinite(result.sh2_m[0])
    assert result.weathering_correction_s is not None
    assert np.isfinite(result.weathering_correction_s[0])
    assert np.isnan(result.sh1_m[1])
    assert np.isnan(result.sh2_m[1])
    assert np.isnan(result.weathering_correction_s[1])


def test_t1lsst_2layer_rejects_invalid_velocity_order() -> None:
    with pytest.raises(RefractionT1LSSTError, match='v2_m_s must be greater'):
        compute_t1lsst_2layer_thicknesses(
            t1_s=np.asarray([0.01]),
            t2_s=np.asarray([0.02]),
            v1_m_s=800.0,
            v2_m_s=800.0,
            v3_m_s=3600.0,
        )

    with pytest.raises(RefractionT1LSSTError, match='v3_m_s must be greater'):
        compute_t1lsst_2layer_weathering_correction(
            sh1_m=np.asarray([10.0]),
            sh2_m=np.asarray([20.0]),
            v1_m_s=800.0,
            v2_m_s=2400.0,
            v3_m_s=2400.0,
        )


def test_t1lsst_2layer_rejects_nonfinite_inputs() -> None:
    with pytest.raises(RefractionT1LSSTError, match='t1_s must contain finite'):
        compute_t1lsst_2layer_thicknesses(
            t1_s=np.asarray([np.nan]),
            t2_s=np.asarray([0.02]),
            v1_m_s=800.0,
            v2_m_s=2400.0,
            v3_m_s=3600.0,
        )

    with pytest.raises(RefractionT1LSSTError, match='v3_m_s must contain finite'):
        compute_t1lsst_2layer_thicknesses(
            t1_s=np.asarray([0.01]),
            t2_s=np.asarray([0.02]),
            v1_m_s=800.0,
            v2_m_s=2400.0,
            v3_m_s=np.inf,
        )


def test_t1lsst_2layer_negative_sh2_is_statused_and_nan_not_clipped() -> None:
    v1_m_s = 800.0
    v2_m_s = 2400.0
    v3_m_s = 3600.0
    sh1_m = np.asarray([10.0], dtype=np.float64)
    t1_s, t2_s = _forward_t1_t2_s(
        sh1_m=sh1_m,
        sh2_m=np.asarray([0.0], dtype=np.float64),
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
    )

    result = compute_t1lsst_2layer_thicknesses_with_status(
        t1_s=t1_s,
        t2_s=t2_s - 0.001,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
    )

    assert result.status.tolist() == ['invalid_negative_thickness']
    np.testing.assert_allclose(result.sh1_m, sh1_m)
    assert np.isnan(result.sh2_m[0])
    assert result.weathering_correction_s is not None
    assert np.isnan(result.weathering_correction_s[0])


def test_t1lsst_2layer_negative_sh1_blanks_dependent_outputs() -> None:
    result = compute_t1lsst_2layer_thicknesses_with_status(
        t1_s=np.asarray([-0.001], dtype=np.float64),
        t2_s=np.asarray([0.1], dtype=np.float64),
        v1_m_s=800.0,
        v2_m_s=2000.0,
        v3_m_s=3000.0,
    )

    assert result.status.tolist() == ['invalid_negative_thickness']
    assert np.isnan(result.sh1_m[0])
    assert np.isnan(result.sh2_m[0])
    assert result.weathering_correction_s is not None
    assert np.isnan(result.weathering_correction_s[0])


def _forward_t1_t2_s(
    *,
    sh1_m: np.ndarray,
    sh2_m: np.ndarray,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    t1_s = sh1_m * np.sqrt(v2_m_s**2 - v1_m_s**2) / (v1_m_s * v2_m_s)
    t2_s = (
        sh1_m * np.sqrt(v3_m_s**2 - v1_m_s**2) / (v1_m_s * v3_m_s)
        + sh2_m * np.sqrt(v3_m_s**2 - v2_m_s**2) / (v2_m_s * v3_m_s)
    )
    return np.asarray(t1_s, dtype=np.float64), np.asarray(t2_s, dtype=np.float64)
