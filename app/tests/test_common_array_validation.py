from __future__ import annotations

import numpy as np
import pytest

from app.services.common.array_validation import (
    coerce_1d_bool_array,
    coerce_1d_finite_float64,
    coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64,
    coerce_1d_string_array,
    coerce_finite_float,
    coerce_header_byte,
    coerce_nonnegative_finite_float,
    coerce_nonnegative_int,
    coerce_optional_finite_float,
    coerce_positive_finite_float,
    coerce_positive_int,
    is_real_numeric_dtype,
)


class CustomValidationError(Exception):
    pass


@pytest.mark.parametrize('value', [True, np.bool_(True)])
def test_integer_scalar_coercion_rejects_bool(value: object) -> None:
    with pytest.raises(ValueError, match='count must be an integer'):
        coerce_positive_int(value, name='count')


@pytest.mark.parametrize(
    'dtype',
    [np.float32, np.float64, np.int16, np.int64, np.uint8],
)
def test_is_real_numeric_dtype_accepts_real_numbers(dtype: np.dtype) -> None:
    assert is_real_numeric_dtype(np.dtype(dtype)) is True


@pytest.mark.parametrize(
    'dtype',
    [np.bool_, np.complex64, np.complex128, object, np.dtype('<U3')],
)
def test_is_real_numeric_dtype_rejects_bool_complex_object_and_string(
    dtype: np.dtype,
) -> None:
    assert is_real_numeric_dtype(np.dtype(dtype)) is False


def test_1d_float_coercion_checks_expected_shape() -> None:
    out = coerce_1d_real_numeric_float64(
        [1, 2, 3],
        name='values',
        expected_shape=(3,),
    )

    assert out.shape == (3,)

    with pytest.raises(ValueError, match=r'expected \(2,\), got \(3,\)'):
        coerce_1d_real_numeric_float64(
            [1, 2, 3],
            name='values',
            expected_shape=(2,),
        )


def test_1d_float_coercion_returns_contiguous_float64() -> None:
    source = np.array([1, 2, 3, 4], dtype=np.int16)[::2]

    out = coerce_1d_real_numeric_float64(source, name='values')

    assert out.dtype == np.float64
    assert out.flags.c_contiguous is True
    np.testing.assert_array_equal(out, np.array([1.0, 3.0], dtype=np.float64))


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
def test_1d_float_coercion_rejects_nonfinite_when_required(
    bad_value: float,
) -> None:
    with pytest.raises(ValueError, match='finite'):
        coerce_1d_real_numeric_float64(
            [1.0, bad_value],
            name='values',
            require_finite=True,
        )


def test_1d_finite_float64_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match='finite'):
        coerce_1d_finite_float64([1.0, np.nan], name='values')


def test_1d_float_coercion_rejects_bool_complex_and_object_dtype() -> None:
    for values in (
        np.array([True, False], dtype=bool),
        np.array([1.0 + 2.0j], dtype=np.complex128),
        np.array([1.0], dtype=object),
    ):
        with pytest.raises(ValueError, match='real numeric dtype'):
            coerce_1d_real_numeric_float64(values, name='values')


def test_integer_array_coercion_accepts_integer_like_float() -> None:
    out = coerce_1d_integer_int64([1.0, 2.0, -3.0], name='indices')

    assert out.dtype == np.int64
    assert out.flags.c_contiguous is True
    np.testing.assert_array_equal(out, np.array([1, 2, -3], dtype=np.int64))


def test_integer_array_coercion_accepts_min_int64_float() -> None:
    out = coerce_1d_integer_int64(
        np.array([np.iinfo(np.int64).min], dtype=np.float64),
        name='indices',
    )

    np.testing.assert_array_equal(
        out,
        np.array([np.iinfo(np.int64).min], dtype=np.int64),
    )


@pytest.mark.parametrize(
    'values',
    [
        np.array([1.2], dtype=np.float64),
        np.array([np.nan], dtype=np.float64),
        np.array([True, False], dtype=bool),
        np.array([2**63], dtype=np.float64),
    ],
)
def test_integer_array_coercion_rejects_non_integer_values(
    values: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match='integer values'):
        coerce_1d_integer_int64(values, name='indices')


def test_bool_array_coercion_accepts_only_bool_dtype() -> None:
    out = coerce_1d_bool_array([True, False], name='mask')

    assert out.dtype == bool
    assert out.flags.c_contiguous is True
    np.testing.assert_array_equal(out, np.array([True, False], dtype=bool))

    with pytest.raises(ValueError, match='bool dtype'):
        coerce_1d_bool_array([1, 0], name='mask')


def test_string_array_coercion_returns_unicode_string_dtype() -> None:
    out = coerce_1d_string_array(['source', 'receiver'], name='kind')

    assert out.dtype.kind == 'U'
    assert out.flags.c_contiguous is True
    np.testing.assert_array_equal(out, np.array(['source', 'receiver']))


def test_string_array_coercion_rejects_object_dtype_by_default() -> None:
    with pytest.raises(ValueError, match='object dtype'):
        coerce_1d_string_array(
            np.array(['source'], dtype=object),
            name='kind',
        )


def test_string_array_coercion_can_allow_object_dtype() -> None:
    out = coerce_1d_string_array(
        np.array(['source', 'receiver'], dtype=object),
        name='kind',
        reject_object_dtype=False,
    )

    assert out.dtype.kind == 'U'
    np.testing.assert_array_equal(out, np.array(['source', 'receiver']))


def test_custom_error_type_is_used() -> None:
    with pytest.raises(CustomValidationError, match='values must be a 1D array'):
        coerce_1d_real_numeric_float64(
            np.array([[1.0]]),
            name='values',
            error_type=CustomValidationError,
        )


def test_scalar_float_and_int_coercers() -> None:
    assert coerce_nonnegative_int(0, name='count') == 0
    assert coerce_positive_int(np.int64(1), name='count') == 1
    assert coerce_finite_float('1.5', name='value') == pytest.approx(1.5)
    assert coerce_optional_finite_float(None, name='value') is None
    assert coerce_optional_finite_float(2, name='value') == pytest.approx(2.0)
    assert coerce_positive_finite_float(0.25, name='value') == pytest.approx(0.25)
    assert coerce_nonnegative_finite_float(0.0, name='value') == pytest.approx(0.0)
    assert coerce_header_byte(240, name='byte') == 240


@pytest.mark.parametrize(
    ('coercer', 'value', 'message'),
    [
        (coerce_nonnegative_int, -1, 'non-negative'),
        (coerce_positive_int, 0, 'greater than 0'),
        (coerce_finite_float, np.inf, 'finite'),
        (coerce_positive_finite_float, 0.0, 'greater than 0'),
        (coerce_nonnegative_finite_float, -0.1, 'non-negative'),
        (coerce_header_byte, 241, 'between 1 and 240'),
    ],
)
def test_scalar_coercers_reject_invalid_values(
    coercer: object,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        coercer(value, name='value')
