from __future__ import annotations

import numpy as np
import pytest

from app.statics.refraction.application.trace_order import (
    original_to_sorted_position,
    sorted_positions_for_original_trace_ids,
)


def test_original_to_sorted_position_inverts_non_identity_permutation() -> None:
    sorted_to_original = np.asarray([2, 0, 1], dtype=np.int64)

    inverse = original_to_sorted_position(sorted_to_original)

    np.testing.assert_array_equal(inverse, [1, 2, 0])
    assert inverse.dtype == np.int64
    assert inverse.flags.c_contiguous


def test_sorted_positions_for_original_trace_ids_maps_rows() -> None:
    sorted_to_original = np.asarray([2, 0, 1], dtype=np.int64)
    original_trace_id = np.asarray([0, 1, 2, 0], dtype=np.int64)

    position = sorted_positions_for_original_trace_ids(
        sorted_to_original=sorted_to_original,
        original_trace_id=original_trace_id,
    )

    np.testing.assert_array_equal(position, [1, 2, 0, 1])
    assert position.dtype == np.int64
    assert position.flags.c_contiguous


@pytest.mark.parametrize(
    'sorted_to_original',
    [
        np.asarray([0, 0, 1], dtype=np.int64),
        np.asarray([0, -1, 2], dtype=np.int64),
        np.asarray([0, 1, 3], dtype=np.int64),
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        np.asarray([[0, 1, 2]], dtype=np.int64),
    ],
)
def test_original_to_sorted_position_rejects_invalid_permutations(
    sorted_to_original: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        original_to_sorted_position(sorted_to_original)


@pytest.mark.parametrize(
    'original_trace_id',
    [
        np.asarray([-1], dtype=np.int64),
        np.asarray([3], dtype=np.int64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([[0]], dtype=np.int64),
    ],
)
def test_sorted_positions_for_original_trace_ids_rejects_invalid_trace_ids(
    original_trace_id: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        sorted_positions_for_original_trace_ids(
            sorted_to_original=np.asarray([2, 0, 1], dtype=np.int64),
            original_trace_id=original_trace_id,
        )
