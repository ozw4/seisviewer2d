from __future__ import annotations

from collections import Counter

import numpy as np

from app.tests._refraction_static_field_synthetic import (
    make_clean_2d_field_corrections,
    make_messy_2d_field_corrections,
)


def test_field_synthetic_clean_fixture_expected_trace_field_shift() -> None:
    dataset = make_clean_2d_field_corrections()

    np.testing.assert_allclose(
        dataset.expected_source_depth_shift_s,
        dataset.source_depth_m / dataset.true_v1_m_s,
    )
    np.testing.assert_allclose(
        dataset.expected_uphole_shift_s,
        -dataset.uphole_time_s,
    )
    np.testing.assert_allclose(
        dataset.expected_source_field_shift_s,
        (
            dataset.expected_source_depth_shift_s
            + dataset.expected_uphole_shift_s
            + dataset.expected_source_manual_static_shift_s
        ),
    )
    np.testing.assert_allclose(
        dataset.expected_receiver_field_shift_s,
        dataset.expected_receiver_manual_static_shift_s,
    )
    np.testing.assert_allclose(
        dataset.expected_trace_field_shift_s,
        (
            dataset.expected_source_field_shift_s_sorted
            + dataset.expected_receiver_field_shift_s_sorted
        ),
    )
    np.testing.assert_array_equal(
        dataset.trace_field_static_status,
        np.full(dataset.sorted_trace_index.shape, 'ok', dtype='<U64'),
    )


def test_field_synthetic_clean_fixture_expected_final_shift() -> None:
    dataset = make_clean_2d_field_corrections()

    expected_refraction = (
        dataset.source_endpoint_table.refraction_shift_s[dataset.source_endpoint_index]
        + dataset.receiver_endpoint_table.refraction_shift_s[
            dataset.receiver_endpoint_index
        ]
    )
    np.testing.assert_allclose(dataset.expected_refraction_trace_shift_s, expected_refraction)
    np.testing.assert_allclose(
        dataset.expected_final_trace_shift_s,
        dataset.expected_refraction_trace_shift_s + dataset.expected_trace_field_shift_s,
    )
    np.testing.assert_allclose(dataset.pick_time_s, dataset.base_dataset.first_break_time_s)

    arrays = dataset.as_sorted_trace_arrays()
    assert arrays['source_depth_m_sorted'].shape == dataset.sorted_trace_index.shape
    assert arrays['uphole_time_s_sorted'].shape == dataset.sorted_trace_index.shape
    np.testing.assert_allclose(
        arrays['final_trace_shift_s_sorted'],
        dataset.expected_final_trace_shift_s,
    )


def test_field_synthetic_messy_fixture_contains_missing_and_invalid_cases() -> None:
    dataset = make_messy_2d_field_corrections()

    missing_depth_index = dataset.missing_source_depth_endpoint_index
    missing_uphole_index = dataset.missing_uphole_endpoint_index
    invalid_source_index = dataset.invalid_source_endpoint_index
    assert missing_depth_index is not None
    assert missing_uphole_index is not None
    assert invalid_source_index is not None

    assert np.isnan(dataset.source_depth_m[missing_depth_index])
    assert dataset.source_depth_status[missing_depth_index] == 'missing_source_depth'
    assert np.isnan(dataset.uphole_time_s[missing_uphole_index])
    assert dataset.uphole_status[missing_uphole_index] == 'missing_uphole_time'
    assert dataset.source_endpoint_table.node_id[invalid_source_index] == -1
    assert not bool(dataset.source_endpoint_table.valid_endpoint_mask[invalid_source_index])
    assert dataset.source_field_static_status[invalid_source_index] == (
        'inactive_source_endpoint'
    )

    duplicate_key = dataset.duplicate_manual_static_endpoint_keys[0]
    row_keys = Counter(
        (row.endpoint_kind, row.endpoint_key)
        for row in dataset.manual_static_rows
        if row.endpoint_key is not None
    )
    assert row_keys[('source', duplicate_key)] == 2
    assert len(dataset.manual_static_rows) == (
        len(dataset.manual_static_rows_without_duplicates) + 1
    )

    for status in (
        'missing_source_depth',
        'missing_uphole_time',
        'inactive_source_endpoint',
    ):
        assert status in set(dataset.trace_field_static_status.tolist())

    invalid_trace_mask = dataset.trace_field_static_status != 'ok'
    assert bool(np.any(invalid_trace_mask))
    assert bool(np.all(np.isnan(dataset.expected_trace_field_shift_s[invalid_trace_mask])))
    assert bool(np.all(np.isnan(dataset.expected_final_trace_shift_s[invalid_trace_mask])))


def test_field_synthetic_manual_static_sign_conversion_expected_values() -> None:
    delay_positive = make_clean_2d_field_corrections(
        manual_static_sign_convention='delay_positive_ms',
    )
    applied_shift = make_clean_2d_field_corrections(
        manual_static_sign_convention='applied_shift_s',
    )

    np.testing.assert_allclose(
        delay_positive.expected_source_manual_static_shift_s,
        -delay_positive.source_manual_static_input_s,
    )
    np.testing.assert_allclose(
        delay_positive.expected_receiver_manual_static_shift_s,
        -delay_positive.receiver_manual_static_input_s,
    )
    np.testing.assert_allclose(
        applied_shift.expected_source_manual_static_shift_s,
        applied_shift.source_manual_static_input_s,
    )
    np.testing.assert_allclose(
        applied_shift.expected_receiver_manual_static_shift_s,
        applied_shift.receiver_manual_static_input_s,
    )
    assert delay_positive.sign_convention == 'corrected(t) = raw(t - shift_s)'
    assert delay_positive.manual_static_sign_convention == 'delay_positive_ms'
    assert applied_shift.manual_static_sign_convention == 'applied_shift_s'
