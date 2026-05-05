from __future__ import annotations

import numpy as np
import pytest

from app.services.geometry_linkage_validation import (
    GeometryLinkageHeaderConfig,
    validate_geometry_linkage_headers,
)


class FakeReader:
    def __init__(self, headers: dict[int, np.ndarray], n_traces: int = 3) -> None:
        self.headers = headers
        self.traces = np.zeros((n_traces, 4), dtype=np.float32)
        self.ensure_calls: list[int] = []

    def ensure_header(self, byte: int) -> np.ndarray:
        self.ensure_calls.append(byte)
        if byte not in self.headers:
            raise KeyError(byte)
        return self.headers[byte]


def _headers() -> dict[int, np.ndarray]:
    return {
        71: np.array([1, -10, 0], dtype=np.int16),
        73: np.array([100.0, 110.0, 120.0], dtype=np.float32),
        77: np.array([200.0, 210.0, 220.0], dtype=np.float32),
        81: np.array([130.0, 140.0, 150.0], dtype=np.float32),
        85: np.array([230.0, 240.0, 250.0], dtype=np.float32),
    }


def test_validate_geometry_linkage_headers_reads_required_headers() -> None:
    reader = FakeReader(_headers())

    result = validate_geometry_linkage_headers(
        reader=reader,
        config=GeometryLinkageHeaderConfig(),
    )

    assert reader.ensure_calls == [71, 73, 77, 81, 85]
    assert result.checked_bytes == (71, 73, 77, 81, 85)
    np.testing.assert_array_equal(result.coordinate_scalar, _headers()[71])
    np.testing.assert_array_equal(result.source_x, _headers()[73])
    np.testing.assert_array_equal(result.source_y, _headers()[77])
    np.testing.assert_array_equal(result.receiver_x, _headers()[81])
    np.testing.assert_array_equal(result.receiver_y, _headers()[85])


def test_validate_geometry_linkage_headers_rejects_missing_header() -> None:
    headers = _headers()
    del headers[81]

    with pytest.raises(ValueError, match='header byte 81'):
        validate_geometry_linkage_headers(
            reader=FakeReader(headers),
            config=GeometryLinkageHeaderConfig(),
        )


def test_validate_geometry_linkage_headers_rejects_shape_mismatch() -> None:
    headers = _headers()
    headers[73] = np.array([100.0, 110.0], dtype=np.float32)

    with pytest.raises(ValueError, match='shape mismatch'):
        validate_geometry_linkage_headers(
            reader=FakeReader(headers),
            config=GeometryLinkageHeaderConfig(),
        )


def test_validate_geometry_linkage_headers_rejects_non_numeric_coordinate() -> None:
    headers = _headers()
    headers[73] = np.array(['100', '110', '120'])

    with pytest.raises(ValueError, match='real numeric dtype'):
        validate_geometry_linkage_headers(
            reader=FakeReader(headers),
            config=GeometryLinkageHeaderConfig(),
        )


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
def test_validate_geometry_linkage_headers_rejects_non_finite_coordinate(
    bad_value: float,
) -> None:
    headers = _headers()
    headers[77] = np.array([200.0, bad_value, 220.0], dtype=np.float64)

    with pytest.raises(ValueError, match='finite values'):
        validate_geometry_linkage_headers(
            reader=FakeReader(headers),
            config=GeometryLinkageHeaderConfig(),
        )


def test_validate_geometry_linkage_headers_rejects_non_integer_scalar() -> None:
    headers = _headers()
    headers[71] = np.array([1.0, 2.5, 3.0], dtype=np.float64)

    with pytest.raises(ValueError, match='integer values'):
        validate_geometry_linkage_headers(
            reader=FakeReader(headers),
            config=GeometryLinkageHeaderConfig(),
        )


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
def test_validate_geometry_linkage_headers_rejects_non_finite_scalar(
    bad_value: float,
) -> None:
    headers = _headers()
    headers[71] = np.array([1.0, bad_value, 3.0], dtype=np.float64)

    with pytest.raises(ValueError, match='finite values'):
        validate_geometry_linkage_headers(
            reader=FakeReader(headers),
            config=GeometryLinkageHeaderConfig(),
        )


def test_validate_geometry_linkage_headers_accepts_integer_valued_float_scalar() -> None:
    headers = _headers()
    headers[71] = np.array([1.0, -10.0, 0.0], dtype=np.float64)
    reader = FakeReader(headers)

    result = validate_geometry_linkage_headers(
        reader=reader,
        config=GeometryLinkageHeaderConfig(),
    )

    np.testing.assert_array_equal(result.coordinate_scalar, headers[71])
