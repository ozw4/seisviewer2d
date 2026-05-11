from __future__ import annotations

import json

import numpy as np

from app.services.refraction_static_source_depth import (
    REFRACTION_SOURCE_DEPTH_QC_JSON_NAME,
    REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
    resolve_refraction_source_depth,
    write_refraction_source_depth_artifacts,
)


def _keys(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=object)


def _ids(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _depths(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def test_source_depth_resolves_median_per_source_endpoint() -> None:
    result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=_keys(['source:a', 'source:a', 'source:b']),
        source_endpoint_id_sorted=_ids([101, 101, 102]),
        source_node_id_sorted=_ids([0, 0, 1]),
        source_depth_m_sorted=_depths([1.0, 3.0, 4.0]),
        mode='weathering_velocity_time',
        source_depth_byte=115,
        inconsistency_tolerance_m=5.0,
    )

    np.testing.assert_array_equal(result.source_endpoint_key, ['source:a', 'source:b'])
    np.testing.assert_array_equal(result.source_endpoint_id, [101, 102])
    np.testing.assert_array_equal(result.source_node_id, [0, 1])
    np.testing.assert_allclose(result.source_depth_m, [2.0, 4.0])
    np.testing.assert_array_equal(result.source_depth_status, ['ok', 'ok'])
    np.testing.assert_array_equal(result.source_depth_pick_count, [2, 1])
    np.testing.assert_array_equal(result.source_depth_trace_count, [2, 1])


def test_source_depth_marks_missing_when_required() -> None:
    result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=_keys(['source:a']),
        source_endpoint_id_sorted=_ids([101]),
        source_node_id_sorted=_ids([0]),
        source_depth_m_sorted=None,
        mode='weathering_velocity_time',
        source_depth_byte=115,
    )

    assert result.source_depth_status.tolist() == ['missing_source_depth']
    assert np.isnan(result.source_depth_m[0])
    assert result.source_depth_pick_count.tolist() == [0]


def test_source_depth_marks_missing_trace_with_resolved_endpoint_depth() -> None:
    result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=_keys(['source:a', 'source:a']),
        source_endpoint_id_sorted=_ids([101, 101]),
        source_node_id_sorted=_ids([0, 0]),
        source_depth_m_sorted=_depths([np.nan, 4.0]),
        mode='weathering_velocity_time',
        source_depth_byte=115,
    )

    assert result.source_depth_status.tolist() == ['missing_source_depth']
    np.testing.assert_allclose(result.source_depth_m, [4.0])
    assert result.source_depth_pick_count.tolist() == [1]
    assert result.source_depth_trace_count.tolist() == [2]


def test_source_depth_marks_inconsistent_values() -> None:
    result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=_keys(['source:a', 'source:a']),
        source_endpoint_id_sorted=_ids([101, 101]),
        source_node_id_sorted=_ids([0, 0]),
        source_depth_m_sorted=_depths([2.0, 2.5]),
        mode='weathering_velocity_time',
        source_depth_byte=115,
        inconsistency_tolerance_m=0.1,
    )

    assert result.source_depth_status.tolist() == ['inconsistent_source_depth']
    np.testing.assert_allclose(result.source_depth_m, [2.25])


def test_source_depth_marks_negative_depth_invalid_when_positive_down() -> None:
    result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=_keys(['source:a', 'source:a']),
        source_endpoint_id_sorted=_ids([101, 101]),
        source_node_id_sorted=_ids([0, 0]),
        source_depth_m_sorted=_depths([-2.0, -4.0]),
        mode='weathering_velocity_time',
        source_depth_byte=115,
        positive_down=True,
    )

    assert result.source_depth_status.tolist() == ['invalid_source_depth']
    assert np.isnan(result.source_depth_m[0])
    assert result.source_depth_pick_count.tolist() == [0]


def test_source_depth_qc_counts_statuses() -> None:
    result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=_keys(
            [
                'source:ok',
                'source:ok',
                'source:missing',
                'source:invalid',
                'source:wide',
                'source:wide',
                'source:large',
                'source:inactive',
            ]
        ),
        source_endpoint_id_sorted=_ids([10, 10, 11, 12, 13, 13, 14, 15]),
        source_node_id_sorted=_ids([0, 0, 1, 2, 3, 3, 4, -1]),
        source_depth_m_sorted=_depths([2.0, 2.005, np.nan, -1.0, 4.0, 5.0, 101.0, 7.0]),
        mode='weathering_velocity_time',
        source_depth_byte=115,
        max_abs_source_depth_m=100.0,
        inconsistency_tolerance_m=0.1,
    )

    assert result.source_depth_status.tolist() == [
        'ok',
        'missing_source_depth',
        'invalid_source_depth',
        'inconsistent_source_depth',
        'exceeds_max_abs_source_depth',
        'inactive_source_endpoint',
    ]
    qc = result.qc
    assert qc['source_depth_mode'] == 'weathering_velocity_time'
    assert qc['source_depth_byte'] == 115
    assert qc['n_source_endpoints'] == 6
    assert qc['n_sources_with_depth'] == 4
    assert qc['n_missing_source_depth'] == 1
    assert qc['n_invalid_source_depth'] == 1
    assert qc['n_inconsistent_source_depth'] == 1
    assert qc['n_exceeds_max_abs_source_depth'] == 1
    assert qc['n_inactive_source_endpoints'] == 1
    assert qc['status_counts']['ok'] == 1
    assert qc['min_source_depth_m'] == 2.0025
    assert qc['max_source_depth_m'] == 101.0


def test_source_depth_artifacts_write_qc_and_source_rows(tmp_path) -> None:
    result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=_keys(['source:a']),
        source_endpoint_id_sorted=_ids([101]),
        source_node_id_sorted=_ids([0]),
        source_depth_m_sorted=_depths([3.5]),
        mode='weathering_velocity_time',
        source_depth_byte=115,
    )

    paths = write_refraction_source_depth_artifacts(tmp_path, result)

    assert paths['qc_json'] == tmp_path / REFRACTION_SOURCE_DEPTH_QC_JSON_NAME
    assert paths['sources_csv'] == tmp_path / REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME
    qc = json.loads(paths['qc_json'].read_text(encoding='utf-8'))
    assert qc['n_sources_with_depth'] == 1
    text = paths['sources_csv'].read_text(encoding='utf-8')
    assert 'source_endpoint_key,source_endpoint_id,source_node_id' in text
    assert 'source:a,101,0,3.5,ok,1,1' in text
