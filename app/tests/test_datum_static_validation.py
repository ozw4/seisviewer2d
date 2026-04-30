from __future__ import annotations

import numpy as np
import pytest

from app.services.datum_static_math import compute_datum_static_shifts
from app.services.datum_static_validation import (
    ExistingStaticHeaderConfig,
    validate_existing_static_headers,
    validate_trace_shift_limits,
)


class FakeReader:
    def __init__(self, headers: dict[int, np.ndarray], n_traces: int) -> None:
        self.headers = headers
        self.traces = np.zeros((n_traces, 4), dtype=np.float32)
        self.ensure_calls: list[int] = []
        self.get_calls: list[int] = []

    def ensure_header(self, byte: int) -> np.ndarray:
        self.ensure_calls.append(byte)
        if byte not in self.headers:
            raise ValueError(f'missing header byte {byte}')
        return self.headers[byte]

    def get_header(self, byte: int) -> np.ndarray:
        self.get_calls.append(byte)
        return self.headers[byte]


def _reader(headers: dict[int, np.ndarray] | None = None) -> FakeReader:
    base_headers = {
        99: np.zeros(3, dtype=np.int16),
        101: np.zeros(3, dtype=np.int16),
        103: np.zeros(3, dtype=np.int16),
    }
    if headers is not None:
        base_headers.update(headers)
    return FakeReader(base_headers, n_traces=3)


def test_existing_static_config_defaults() -> None:
    config = ExistingStaticHeaderConfig()

    assert config.policy == 'fail_if_nonzero'
    assert config.source_static_byte == 99
    assert config.receiver_static_byte == 101
    assert config.total_static_byte == 103


def test_existing_static_config_rejects_unsupported_policy() -> None:
    with pytest.raises(ValueError, match='fail_if_nonzero'):
        ExistingStaticHeaderConfig(policy='ignore')  # type: ignore[arg-type]


@pytest.mark.parametrize(
    'field',
    ['source_static_byte', 'receiver_static_byte', 'total_static_byte'],
)
def test_existing_static_config_rejects_bool_byte(field: str) -> None:
    kwargs = {
        'source_static_byte': 99,
        'receiver_static_byte': 101,
        'total_static_byte': 103,
    }
    kwargs[field] = True

    with pytest.raises(ValueError):
        ExistingStaticHeaderConfig(**kwargs)


@pytest.mark.parametrize('bad_byte', [0, 241])
def test_existing_static_config_rejects_out_of_range_byte(bad_byte: int) -> None:
    with pytest.raises(ValueError):
        ExistingStaticHeaderConfig(source_static_byte=bad_byte)


def test_existing_static_config_rejects_duplicate_bytes() -> None:
    with pytest.raises(ValueError, match='unique'):
        ExistingStaticHeaderConfig(source_static_byte=99, receiver_static_byte=99)


def test_existing_static_config_rejects_all_bytes_none() -> None:
    with pytest.raises(ValueError, match='at least one'):
        ExistingStaticHeaderConfig(
            source_static_byte=None,
            receiver_static_byte=None,
            total_static_byte=None,
        )


def test_validate_existing_static_headers_all_zero() -> None:
    reader = _reader()

    result = validate_existing_static_headers(
        reader=reader,
        config=ExistingStaticHeaderConfig(),
    )

    assert result.checked is True
    assert result.policy == 'fail_if_nonzero'
    assert result.checked_bytes == (99, 101, 103)
    assert result.nonzero_source_static_count == 0
    assert result.nonzero_receiver_static_count == 0
    assert result.nonzero_total_static_count == 0
    assert result.nonzero_any_count == 0
    assert reader.ensure_calls == []
    assert reader.get_calls == [99, 101, 103]


def test_validate_existing_static_headers_nonzero_source_fails() -> None:
    with pytest.raises(ValueError, match=r'source_static_byte=99 count=1'):
        validate_existing_static_headers(
            reader=_reader({99: np.array([0, 7, 0], dtype=np.int16)}),
            config=ExistingStaticHeaderConfig(),
        )


def test_validate_existing_static_headers_nonzero_receiver_fails() -> None:
    with pytest.raises(ValueError, match=r'receiver_static_byte=101 count=1'):
        validate_existing_static_headers(
            reader=_reader({101: np.array([0, -4, 0], dtype=np.int16)}),
            config=ExistingStaticHeaderConfig(),
        )


def test_validate_existing_static_headers_nonzero_total_fails() -> None:
    with pytest.raises(ValueError, match=r'total_static_byte=103 count=1'):
        validate_existing_static_headers(
            reader=_reader({103: np.array([0, 0, 3], dtype=np.int16)}),
            config=ExistingStaticHeaderConfig(),
        )


def test_validate_existing_static_headers_reports_multiple_nonzero_counts() -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_existing_static_headers(
            reader=_reader(
                {
                    99: np.array([1, 0, 1], dtype=np.int16),
                    101: np.array([0, 2, 0], dtype=np.int16),
                    103: np.array([3, 0, 4], dtype=np.int16),
                }
            ),
            config=ExistingStaticHeaderConfig(),
        )

    message = str(exc_info.value)
    assert 'source_static_byte=99 count=2' in message
    assert 'receiver_static_byte=101 count=1' in message
    assert 'total_static_byte=103 count=2' in message


def test_validate_existing_static_headers_skips_none_byte() -> None:
    reader = _reader()

    result = validate_existing_static_headers(
        reader=reader,
        config=ExistingStaticHeaderConfig(receiver_static_byte=None),
    )

    assert result.checked_bytes == (99, 103)
    assert result.receiver_static_byte is None
    assert result.nonzero_receiver_static_count == 0
    assert reader.ensure_calls == []
    assert reader.get_calls == [99, 103]


def test_validate_existing_static_headers_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match='shape mismatch'):
        validate_existing_static_headers(
            reader=_reader({99: np.array([0, 0], dtype=np.int16)}),
            config=ExistingStaticHeaderConfig(),
        )


def test_validate_existing_static_headers_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match='numeric dtype'):
        validate_existing_static_headers(
            reader=_reader({99: np.array(['0', '0', '0'])}),
            config=ExistingStaticHeaderConfig(),
        )


def test_validate_existing_static_headers_rejects_non_integer_values() -> None:
    with pytest.raises(ValueError, match='integer values'):
        validate_existing_static_headers(
            reader=_reader({99: np.array([0.0, 1.5, 0.0], dtype=np.float64)}),
            config=ExistingStaticHeaderConfig(),
        )


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
def test_validate_existing_static_headers_rejects_nan_inf(bad_value: float) -> None:
    with pytest.raises(ValueError, match='finite'):
        validate_existing_static_headers(
            reader=_reader({99: np.array([0.0, bad_value, 0.0], dtype=np.float64)}),
            config=ExistingStaticHeaderConfig(),
        )


def test_validate_trace_shift_limits_accepts_within_limit() -> None:
    summary = validate_trace_shift_limits(
        trace_shift_s_sorted=np.array([-0.1, 0.05, 0.0], dtype=np.float64),
        max_abs_shift_ms=100.1,
        expected_n_traces=3,
    )

    assert summary.n_traces == 3
    assert summary.max_abs_shift_ms == pytest.approx(100.1)
    assert summary.max_abs_observed_shift_ms == pytest.approx(100.0)


def test_validate_trace_shift_limits_accepts_equal_to_limit() -> None:
    summary = validate_trace_shift_limits(
        trace_shift_s_sorted=np.array([-0.25, 0.25], dtype=np.float64),
        max_abs_shift_ms=250.0,
        expected_n_traces=2,
    )

    assert summary.max_abs_observed_shift_ms == pytest.approx(250.0)


def test_validate_trace_shift_limits_rejects_exceeding_limit() -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_trace_shift_limits(
            trace_shift_s_sorted=np.array([-0.1, 0.3012, -0.26], dtype=np.float64),
            max_abs_shift_ms=250.0,
            expected_n_traces=3,
        )

    message = str(exc_info.value)
    assert 'limit=250.0 ms' in message
    assert 'observed=301.2' in message
    assert 'count=2' in message


def test_validate_trace_shift_limits_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match='shape mismatch'):
        validate_trace_shift_limits(
            trace_shift_s_sorted=np.array([0.0, 0.1], dtype=np.float64),
            max_abs_shift_ms=250.0,
            expected_n_traces=3,
        )


def test_validate_trace_shift_limits_rejects_non_1d_shift() -> None:
    with pytest.raises(ValueError, match='1D array'):
        validate_trace_shift_limits(
            trace_shift_s_sorted=np.array([[0.0, 0.1]], dtype=np.float64),
            max_abs_shift_ms=250.0,
        )


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
def test_validate_trace_shift_limits_rejects_non_finite_shift(
    bad_value: float,
) -> None:
    with pytest.raises(ValueError, match='finite values'):
        validate_trace_shift_limits(
            trace_shift_s_sorted=np.array([0.0, bad_value], dtype=np.float64),
            max_abs_shift_ms=250.0,
        )


def test_validate_trace_shift_limits_rejects_non_numeric_shift() -> None:
    with pytest.raises(ValueError, match='numeric dtype'):
        validate_trace_shift_limits(
            trace_shift_s_sorted=np.array(['0.0', '0.1']),
            max_abs_shift_ms=250.0,
        )


@pytest.mark.parametrize('bad_limit', [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_validate_trace_shift_limits_rejects_non_positive_limit(
    bad_limit: float,
) -> None:
    with pytest.raises(ValueError, match='greater than 0'):
        validate_trace_shift_limits(
            trace_shift_s_sorted=np.array([0.0], dtype=np.float64),
            max_abs_shift_ms=bad_limit,
        )


def test_validate_trace_shift_limits_summarizes_min_max_mean() -> None:
    summary = validate_trace_shift_limits(
        trace_shift_s_sorted=np.array([-0.1, 0.025, 0.05], dtype=np.float64),
        max_abs_shift_ms=250.0,
    )

    assert summary.n_traces == 3
    assert summary.min_shift_ms == pytest.approx(-100.0)
    assert summary.max_shift_ms == pytest.approx(50.0)
    assert summary.mean_shift_ms == pytest.approx(-25.0 / 3.0)
    assert summary.max_abs_observed_shift_ms == pytest.approx(100.0)


def test_validate_trace_shift_limits_accepts_datum_static_math_output() -> None:
    result = compute_datum_static_shifts(
        source_surface_elevation_m_sorted=np.array([100.0, 110.0]),
        receiver_elevation_m_sorted=np.array([100.0, 120.0]),
        datum_elevation_m=0.0,
        replacement_velocity_m_s=2000.0,
    )

    summary = validate_trace_shift_limits(
        trace_shift_s_sorted=result.trace_shift_s_sorted,
        max_abs_shift_ms=250.0,
        expected_n_traces=2,
    )

    assert summary.max_abs_observed_shift_ms <= 250.0
