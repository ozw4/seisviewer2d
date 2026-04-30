from __future__ import annotations

import numpy as np
import pytest

from app.services.datum_static_geometry import (
    DatumStaticGeometryConfig,
    load_datum_static_geometry,
)
from app.services.datum_static_math import compute_datum_static_shifts


class FakeTraceStoreReader:
    def __init__(self, headers: dict[int, np.ndarray], n_traces: int) -> None:
        self.headers = headers
        self.traces = np.zeros((n_traces, 1), dtype=np.float32)
        self.ensure_calls: list[int] = []

    def ensure_header(self, byte: int) -> np.ndarray:
        self.ensure_calls.append(byte)
        if byte not in self.headers:
            raise ValueError(f'missing header byte {byte}')
        return self.headers[byte]


def _reader(headers: dict[int, np.ndarray] | None = None) -> FakeTraceStoreReader:
    base_headers = {
        45: np.array([100.0, 200.0, 300.0], dtype=np.float64),
        41: np.array([10.0, 20.0, 30.0], dtype=np.float64),
        69: np.array([1, 1, 1], dtype=np.int16),
    }
    if headers is not None:
        base_headers.update(headers)
    return FakeTraceStoreReader(base_headers, n_traces=3)


def test_load_datum_static_geometry_defaults_without_source_depth() -> None:
    reader = _reader()

    geometry = load_datum_static_geometry(
        reader=reader,
        config=DatumStaticGeometryConfig(),
    )

    np.testing.assert_allclose(
        geometry.source_surface_elevation_m_sorted,
        np.array([100.0, 200.0, 300.0]),
    )
    np.testing.assert_allclose(
        geometry.receiver_elevation_m_sorted,
        np.array([10.0, 20.0, 30.0]),
    )
    np.testing.assert_allclose(geometry.source_depth_m_sorted, np.zeros(3))
    np.testing.assert_array_equal(
        geometry.source_depth_used_sorted,
        np.array([False, False, False]),
    )
    np.testing.assert_array_equal(geometry.elevation_scalar_sorted, np.array([1, 1, 1]))
    assert geometry.elevation_scalar_zero_count == 0
    assert geometry.source_elevation_byte == 45
    assert geometry.receiver_elevation_byte == 41
    assert geometry.elevation_scalar_byte == 69
    assert geometry.source_depth_byte is None
    assert geometry.elevation_unit == 'm'
    assert geometry.n_traces == 3
    assert reader.ensure_calls == [45, 41, 69]
    for arr in (
        geometry.source_surface_elevation_m_sorted,
        geometry.receiver_elevation_m_sorted,
        geometry.source_depth_m_sorted,
        geometry.source_depth_used_sorted,
        geometry.elevation_scalar_sorted,
    ):
        assert arr.ndim == 1
        assert arr.shape == (3,)


def test_load_datum_static_geometry_with_source_depth() -> None:
    reader = _reader(
        {
            49: np.array([10.0, 20.0, 30.0], dtype=np.float64),
            69: np.array([2, -4, 0], dtype=np.int16),
        }
    )

    geometry = load_datum_static_geometry(
        reader=reader,
        config=DatumStaticGeometryConfig(source_depth_byte=49),
    )

    np.testing.assert_allclose(
        geometry.source_depth_m_sorted,
        np.array([20.0, 5.0, 30.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        geometry.source_depth_used_sorted,
        np.array([True, True, True]),
    )
    assert geometry.source_depth_byte == 49
    assert reader.ensure_calls == [45, 41, 69, 49]


def test_load_datum_static_geometry_applies_positive_negative_zero_scalar() -> None:
    reader = _reader(
        {
            45: np.array([10.0, 10.0, 10.0], dtype=np.float64),
            41: np.array([20.0, 20.0, 20.0], dtype=np.float64),
            69: np.array([2, -4, 0], dtype=np.int16),
        }
    )

    geometry = load_datum_static_geometry(
        reader=reader,
        config=DatumStaticGeometryConfig(),
    )

    np.testing.assert_allclose(
        geometry.source_surface_elevation_m_sorted,
        np.array([20.0, 2.5, 10.0]),
    )
    np.testing.assert_allclose(
        geometry.receiver_elevation_m_sorted,
        np.array([40.0, 5.0, 20.0]),
    )


def test_load_datum_static_geometry_counts_zero_scalar() -> None:
    reader = _reader({69: np.array([0, 1, 0], dtype=np.int16)})

    geometry = load_datum_static_geometry(
        reader=reader,
        config=DatumStaticGeometryConfig(),
    )

    assert geometry.elevation_scalar_zero_count == 2


def test_load_datum_static_geometry_accepts_integer_valued_float_scalar() -> None:
    reader = _reader({69: np.array([2.0, -4.0, 0.0], dtype=np.float64)})

    geometry = load_datum_static_geometry(
        reader=reader,
        config=DatumStaticGeometryConfig(),
    )

    np.testing.assert_array_equal(geometry.elevation_scalar_sorted, np.array([2, -4, 0]))
    np.testing.assert_allclose(
        geometry.source_surface_elevation_m_sorted,
        np.array([200.0, 50.0, 300.0]),
    )


def test_load_datum_static_geometry_converts_ft_to_m() -> None:
    reader = _reader(
        {
            45: np.array([10.0, 20.0, 30.0], dtype=np.float64),
            41: np.array([5.0, 15.0, 25.0], dtype=np.float64),
            49: np.array([1.0, 2.0, 3.0], dtype=np.float64),
        }
    )

    geometry = load_datum_static_geometry(
        reader=reader,
        config=DatumStaticGeometryConfig(source_depth_byte=49, elevation_unit='ft'),
    )

    np.testing.assert_allclose(
        geometry.source_surface_elevation_m_sorted,
        np.array([3.048, 6.096, 9.144]),
    )
    np.testing.assert_allclose(
        geometry.receiver_elevation_m_sorted,
        np.array([1.524, 4.572, 7.62]),
    )
    np.testing.assert_allclose(
        geometry.source_depth_m_sorted,
        np.array([0.3048, 0.6096, 0.9144]),
    )


def test_load_datum_static_geometry_rejects_unknown_unit() -> None:
    with pytest.raises(ValueError):
        load_datum_static_geometry(
            reader=_reader(),
            config=DatumStaticGeometryConfig(elevation_unit='km'),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    'field',
    [
        'source_elevation_byte',
        'receiver_elevation_byte',
        'elevation_scalar_byte',
        'source_depth_byte',
    ],
)
def test_load_datum_static_geometry_rejects_bool_header_byte(field: str) -> None:
    config = DatumStaticGeometryConfig()
    kwargs = {
        'source_elevation_byte': config.source_elevation_byte,
        'receiver_elevation_byte': config.receiver_elevation_byte,
        'elevation_scalar_byte': config.elevation_scalar_byte,
        'source_depth_byte': config.source_depth_byte,
    }
    kwargs[field] = True

    with pytest.raises(ValueError):
        load_datum_static_geometry(
            reader=_reader(),
            config=DatumStaticGeometryConfig(**kwargs),
        )


@pytest.mark.parametrize('bad_byte', [0, 241])
def test_load_datum_static_geometry_rejects_out_of_range_header_byte(
    bad_byte: int,
) -> None:
    with pytest.raises(ValueError):
        load_datum_static_geometry(
            reader=_reader(),
            config=DatumStaticGeometryConfig(source_elevation_byte=bad_byte),
        )


@pytest.mark.parametrize(
    'config',
    [
        DatumStaticGeometryConfig(source_elevation_byte=41),
        DatumStaticGeometryConfig(source_depth_byte=69),
    ],
)
def test_load_datum_static_geometry_rejects_duplicate_header_bytes(
    config: DatumStaticGeometryConfig,
) -> None:
    with pytest.raises(ValueError):
        load_datum_static_geometry(reader=_reader(), config=config)


def test_load_datum_static_geometry_rejects_missing_required_header() -> None:
    reader = FakeTraceStoreReader(
        {
            41: np.array([10.0, 20.0, 30.0], dtype=np.float64),
            69: np.array([1, 1, 1], dtype=np.int16),
        },
        n_traces=3,
    )

    with pytest.raises(ValueError):
        load_datum_static_geometry(
            reader=reader,
            config=DatumStaticGeometryConfig(),
        )


def test_load_datum_static_geometry_rejects_header_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        load_datum_static_geometry(
            reader=_reader({45: np.array([1.0, 2.0], dtype=np.float64)}),
            config=DatumStaticGeometryConfig(),
        )


def test_load_datum_static_geometry_rejects_non_numeric_header() -> None:
    with pytest.raises(ValueError):
        load_datum_static_geometry(
            reader=_reader({45: np.array(['high', 'mid', 'low'])}),
            config=DatumStaticGeometryConfig(),
        )


def test_load_datum_static_geometry_rejects_non_finite_header() -> None:
    with pytest.raises(ValueError):
        load_datum_static_geometry(
            reader=_reader({45: np.array([1.0, np.nan, 3.0], dtype=np.float64)}),
            config=DatumStaticGeometryConfig(),
        )


def test_load_datum_static_geometry_rejects_non_integer_scalar() -> None:
    with pytest.raises(ValueError):
        load_datum_static_geometry(
            reader=_reader({69: np.array([1.0, 1.5, 2.0], dtype=np.float64)}),
            config=DatumStaticGeometryConfig(),
        )


def test_load_datum_static_geometry_preserves_sorted_order() -> None:
    reader = _reader(
        {
            45: np.array([300.0, 100.0, 200.0], dtype=np.float64),
            41: np.array([30.0, 10.0, 20.0], dtype=np.float64),
        }
    )

    geometry = load_datum_static_geometry(
        reader=reader,
        config=DatumStaticGeometryConfig(),
    )

    np.testing.assert_allclose(
        geometry.source_surface_elevation_m_sorted,
        np.array([300.0, 100.0, 200.0]),
    )
    np.testing.assert_allclose(
        geometry.receiver_elevation_m_sorted,
        np.array([30.0, 10.0, 20.0]),
    )


def test_load_datum_static_geometry_connects_to_datum_static_math() -> None:
    geometry = load_datum_static_geometry(
        reader=_reader(),
        config=DatumStaticGeometryConfig(),
    )

    from_geometry = compute_datum_static_shifts(
        source_surface_elevation_m_sorted=geometry.source_surface_elevation_m_sorted,
        receiver_elevation_m_sorted=geometry.receiver_elevation_m_sorted,
        datum_elevation_m=0.0,
        replacement_velocity_m_s=2000.0,
        source_depth_m_sorted=geometry.source_depth_m_sorted,
    )
    without_depth_arg = compute_datum_static_shifts(
        source_surface_elevation_m_sorted=geometry.source_surface_elevation_m_sorted,
        receiver_elevation_m_sorted=geometry.receiver_elevation_m_sorted,
        datum_elevation_m=0.0,
        replacement_velocity_m_s=2000.0,
    )

    np.testing.assert_allclose(
        from_geometry.trace_shift_s_sorted,
        without_depth_arg.trace_shift_s_sorted,
    )
