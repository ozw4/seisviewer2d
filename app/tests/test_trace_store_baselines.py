from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from app.services.trace_store_baselines import (
    compute_raw_baseline_stats,
    write_trace_store_raw_baseline_artifacts,
)
from app.utils.baseline_artifacts import (
    BASELINE_ARRAY_KEYS,
    BASELINE_ARTIFACT_ID_FIELD,
    BASELINE_STAGE_RAW,
    build_baseline_manifest_path,
    build_baseline_npz_path,
)


def _trace_stats(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        traces.sum(axis=1, dtype=np.float64),
        np.einsum('ij,ij->i', traces, traces, dtype=np.float64),
    )


def _valid_compute_args() -> dict[str, Any]:
    traces = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    trace_sum, trace_sumsq = _trace_stats(traces)
    return {
        'key1_values': np.asarray([10, 20], dtype=np.int64),
        'key1_offsets': np.asarray([0, 2], dtype=np.int64),
        'key1_counts': np.asarray([2, 1], dtype=np.int64),
        'trace_sum': trace_sum,
        'trace_sumsq': trace_sumsq,
        'n_samples': traces.shape[1],
    }


def _compute_with(**overrides: Any):
    args = _valid_compute_args()
    args.update(overrides)
    return compute_raw_baseline_stats(**args)


def test_compute_raw_baseline_stats_matches_existing_formula() -> None:
    traces = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [-1.0, 0.0, 1.0, 2.0],
            [4.0, 4.0, 4.0, 4.0],
            [10.0, 12.0, 14.0, 16.0],
        ],
        dtype=np.float32,
    )
    trace_sum, trace_sumsq = _trace_stats(traces)
    key1_values = np.asarray([10, 20], dtype=np.int64)
    key1_offsets = np.asarray([0, 3], dtype=np.int64)
    key1_counts = np.asarray([3, 2], dtype=np.int64)

    stats = compute_raw_baseline_stats(
        key1_values=key1_values,
        key1_offsets=key1_offsets,
        key1_counts=key1_counts,
        trace_sum=trace_sum,
        trace_sumsq=trace_sumsq,
        n_samples=traces.shape[1],
    )

    n_samples_f64 = float(traces.shape[1])
    expected_mu_traces = trace_sum / n_samples_f64
    expected_trace_var = np.maximum(
        (trace_sumsq / n_samples_f64) - np.square(expected_mu_traces),
        0.0,
    )
    expected_sigma_traces = np.sqrt(expected_trace_var)
    expected_zero_var_mask = expected_sigma_traces <= 1e-12
    expected_sigma_traces[expected_zero_var_mask] = 1.0

    section_sum = np.add.reduceat(trace_sum, key1_offsets)
    section_sumsq = np.add.reduceat(trace_sumsq, key1_offsets)
    total_samples = key1_counts.astype(np.float64) * n_samples_f64
    expected_mu_sections = section_sum / total_samples
    expected_section_var = np.maximum(
        (section_sumsq / total_samples) - np.square(expected_mu_sections),
        0.0,
    )
    expected_sigma_sections = np.sqrt(expected_section_var)

    assert stats.mu_traces.dtype == np.float32
    assert stats.sigma_traces.dtype == np.float32
    assert stats.zero_var_mask.dtype == bool
    assert stats.mu_sections.dtype == np.float32
    assert stats.sigma_sections.dtype == np.float32
    np.testing.assert_allclose(stats.mu_traces, expected_mu_traces.astype(np.float32))
    np.testing.assert_allclose(
        stats.sigma_traces,
        expected_sigma_traces.astype(np.float32),
    )
    np.testing.assert_array_equal(stats.zero_var_mask, expected_zero_var_mask)
    np.testing.assert_allclose(
        stats.mu_sections,
        expected_mu_sections.astype(np.float32),
    )
    np.testing.assert_allclose(
        stats.sigma_sections,
        expected_sigma_sections.astype(np.float32),
    )


def test_compute_raw_baseline_stats_sets_zero_variance_trace_sigma_to_one() -> None:
    traces = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 4.0, 4.0],
            [10.0, 12.0, 14.0],
        ],
        dtype=np.float32,
    )
    trace_sum, trace_sumsq = _trace_stats(traces)

    stats = compute_raw_baseline_stats(
        key1_values=np.asarray([10, 20], dtype=np.int64),
        key1_offsets=np.asarray([0, 2], dtype=np.int64),
        key1_counts=np.asarray([2, 1], dtype=np.int64),
        trace_sum=trace_sum,
        trace_sumsq=trace_sumsq,
        n_samples=3,
    )

    np.testing.assert_array_equal(
        stats.zero_var_mask,
        np.asarray([False, True, False], dtype=bool),
    )
    np.testing.assert_allclose(stats.sigma_traces[1], np.float32(1.0))


def test_compute_raw_baseline_stats_does_not_replace_zero_variance_section_sigma() -> None:
    traces = np.full((2, 3), 5.0, dtype=np.float32)
    trace_sum, trace_sumsq = _trace_stats(traces)

    stats = compute_raw_baseline_stats(
        key1_values=np.asarray([10], dtype=np.int64),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([2], dtype=np.int64),
        trace_sum=trace_sum,
        trace_sumsq=trace_sumsq,
        n_samples=3,
    )

    np.testing.assert_allclose(stats.sigma_traces, np.asarray([1.0, 1.0]))
    np.testing.assert_allclose(stats.sigma_sections, np.asarray([0.0]))


def test_compute_raw_baseline_stats_builds_trace_spans_by_key1() -> None:
    stats = _compute_with(
        key1_values=np.asarray([10, 20], dtype=np.int64),
        key1_offsets=np.asarray([0, 2], dtype=np.int64),
        key1_counts=np.asarray([2, 1], dtype=np.int64),
    )

    assert stats.trace_spans_by_key1 == {'10': [[0, 2]], '20': [[2, 3]]}


def test_write_trace_store_raw_baseline_artifacts_writes_split_manifest_and_npz(
    tmp_path,
) -> None:
    args = _valid_compute_args()
    payload = write_trace_store_raw_baseline_artifacts(
        store_path=tmp_path,
        key1_byte=189,
        key2_byte=193,
        dtype_base='float32',
        dt=0.004,
        source_sha256='abc123',
        **args,
    )

    manifest_path = build_baseline_manifest_path(
        tmp_path,
        stage=BASELINE_STAGE_RAW,
        key1_byte=189,
        key2_byte=193,
    )
    npz_path = build_baseline_npz_path(
        tmp_path,
        stage=BASELINE_STAGE_RAW,
        key1_byte=189,
        key2_byte=193,
    )

    assert manifest_path.is_file()
    assert npz_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['stage'] == BASELINE_STAGE_RAW
    assert manifest['key1_byte'] == 189
    assert manifest['key2_byte'] == 193
    assert manifest['dtype_base'] == 'float32'
    assert manifest['dt'] == 0.004
    assert manifest['source_sha256'] == 'abc123'
    assert manifest['key1_values'] == [10, 20]
    assert manifest['trace_spans_by_key1'] == {'10': [[0, 2]], '20': [[2, 3]]}
    for key in BASELINE_ARRAY_KEYS:
        assert key not in manifest

    with np.load(npz_path, allow_pickle=False) as arrays:
        assert set(BASELINE_ARRAY_KEYS).issubset(arrays.files)
        assert BASELINE_ARTIFACT_ID_FIELD in arrays.files
        np.testing.assert_allclose(arrays['mu_traces'], payload['mu_traces'])
        np.testing.assert_allclose(arrays['sigma_traces'], payload['sigma_traces'])
        np.testing.assert_array_equal(arrays['zero_var_mask'], payload['zero_var_mask'])
        assert str(arrays[BASELINE_ARTIFACT_ID_FIELD]) == manifest[BASELINE_ARTIFACT_ID_FIELD]


def test_write_trace_store_raw_baseline_artifacts_returns_numpy_array_payload(
    tmp_path,
) -> None:
    payload = write_trace_store_raw_baseline_artifacts(
        store_path=tmp_path,
        key1_byte=189,
        key2_byte=193,
        dtype_base='float32',
        dt=None,
        source_sha256=None,
        **_valid_compute_args(),
    )

    assert isinstance(payload['mu_traces'], np.ndarray)
    assert payload['mu_traces'].dtype == np.float32
    assert isinstance(payload['sigma_traces'], np.ndarray)
    assert payload['sigma_traces'].dtype == np.float32
    assert isinstance(payload['zero_var_mask'], np.ndarray)
    assert payload['zero_var_mask'].dtype == bool


@pytest.mark.parametrize('n_samples', [0, -1])
def test_compute_raw_baseline_stats_rejects_non_positive_n_samples(
    n_samples: int,
) -> None:
    with pytest.raises(ValueError):
        _compute_with(n_samples=n_samples)


@pytest.mark.parametrize(
    'overrides',
    [
        {'trace_sum': np.ones((3, 1), dtype=np.float64)},
        {'trace_sumsq': np.ones((3, 1), dtype=np.float64)},
        {'trace_sumsq': np.ones(2, dtype=np.float64)},
    ],
)
def test_compute_raw_baseline_stats_rejects_trace_sum_shape_mismatch(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _compute_with(**overrides)


def test_compute_raw_baseline_stats_rejects_empty_trace_stats() -> None:
    with pytest.raises(ValueError):
        _compute_with(
            trace_sum=np.asarray([], dtype=np.float64),
            trace_sumsq=np.asarray([], dtype=np.float64),
        )


@pytest.mark.parametrize(
    'overrides',
    [
        {'trace_sum': np.asarray([1.0, np.nan, 3.0])},
        {'trace_sumsq': np.asarray([1.0, np.inf, 3.0])},
    ],
)
def test_compute_raw_baseline_stats_rejects_non_finite_trace_stats(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _compute_with(**overrides)


def test_compute_raw_baseline_stats_rejects_float_key1_values() -> None:
    with pytest.raises(ValueError, match='integer dtype'):
        _compute_with(key1_values=np.asarray([10.0, 20.0], dtype=np.float64))


def test_compute_raw_baseline_stats_rejects_non_finite_key1_values() -> None:
    with pytest.raises(ValueError, match='finite'):
        _compute_with(key1_values=np.asarray([10.0, np.inf], dtype=np.float64))


@pytest.mark.parametrize(
    'overrides',
    [
        {'key1_values': np.asarray([[10, 20]], dtype=np.int64)},
        {'key1_offsets': np.asarray([[0, 2]], dtype=np.int64)},
        {'key1_counts': np.asarray([[2, 1]], dtype=np.int64)},
        {'key1_values': np.asarray([10], dtype=np.int64)},
        {'key1_offsets': np.asarray([0], dtype=np.int64)},
        {'key1_counts': np.asarray([3], dtype=np.int64)},
    ],
)
def test_compute_raw_baseline_stats_rejects_key1_shape_mismatch(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _compute_with(**overrides)


def test_compute_raw_baseline_stats_rejects_empty_key1_sections() -> None:
    with pytest.raises(ValueError):
        _compute_with(
            key1_values=np.asarray([], dtype=np.int64),
            key1_offsets=np.asarray([], dtype=np.int64),
            key1_counts=np.asarray([], dtype=np.int64),
        )


def test_compute_raw_baseline_stats_rejects_negative_offsets() -> None:
    with pytest.raises(ValueError):
        _compute_with(
            key1_offsets=np.asarray([0, -2], dtype=np.int64),
            key1_counts=np.asarray([1, 2], dtype=np.int64),
        )


@pytest.mark.parametrize('counts', [[2, 0], [2, -1]])
def test_compute_raw_baseline_stats_rejects_non_positive_counts(
    counts: list[int],
) -> None:
    with pytest.raises(ValueError):
        _compute_with(key1_counts=np.asarray(counts, dtype=np.int64))


def test_compute_raw_baseline_stats_rejects_non_contiguous_spans() -> None:
    with pytest.raises(ValueError):
        _compute_with(
            key1_offsets=np.asarray([0, 3], dtype=np.int64),
            key1_counts=np.asarray([2, 1], dtype=np.int64),
        )


def test_compute_raw_baseline_stats_rejects_spans_not_covering_all_traces() -> None:
    traces = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )
    trace_sum, trace_sumsq = _trace_stats(traces)

    with pytest.raises(ValueError):
        compute_raw_baseline_stats(
            key1_values=np.asarray([10, 20], dtype=np.int64),
            key1_offsets=np.asarray([0, 2], dtype=np.int64),
            key1_counts=np.asarray([2, 1], dtype=np.int64),
            trace_sum=trace_sum,
            trace_sumsq=trace_sumsq,
            n_samples=3,
        )
