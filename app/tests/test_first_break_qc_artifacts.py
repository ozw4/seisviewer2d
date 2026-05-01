from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.services.first_break_qc_artifacts import (
    FIRST_BREAK_QC_CSV_NAME,
    FIRST_BREAK_QC_JSON_NAME,
    RESIDUAL_BY_KEY1_CSV_NAME,
    FirstBreakQcArtifactPaths,
    write_first_break_qc_artifacts,
)
from app.services.first_break_qc_inputs import FirstBreakQcInputs
from app.services.first_break_qc_math import (
    CorrelationQc,
    FirstBreakQcMetrics,
    LinearOffsetFit,
    compute_finite_series_stats,
    compute_first_break_qc_metrics,
    compute_residual_by_key1,
)

DT = 0.004
N_SAMPLES = 512
OFFSET_BYTE = 37

_TRACE_CSV_COLUMNS = [
    'sorted_trace_index',
    'key1',
    'key2',
    'valid_pick',
    'pick_time_raw_s',
    'datum_trace_shift_s',
    'pick_time_after_datum_s',
    'offset',
    'abs_offset',
    'source_elevation_m',
    'receiver_elevation_m',
    'linear_moveout_model_s',
    'residual_after_datum_s',
]

_RESIDUAL_BY_KEY1_CSV_COLUMNS = [
    'key1',
    'n_traces',
    'n_valid_picks',
    'n_used_residual',
    'residual_median_s',
    'residual_mad_s',
    'residual_mean_s',
    'residual_std_s',
]


def _base_inputs() -> FirstBreakQcInputs:
    offset = np.asarray([-1000.0, -500.0, 0.0, 500.0, 1000.0, -1500.0])
    datum_shift = np.asarray([0.010, -0.005, 0.0, 0.015, -0.010, 0.005])
    after_datum = 0.100 + 0.0001 * np.abs(offset)
    valid_mask = np.asarray([True, True, False, True, True, True], dtype=bool)
    picks = after_datum - datum_shift
    picks[~valid_mask] = np.nan
    return FirstBreakQcInputs(
        picks_time_s_sorted=np.asarray(picks, dtype=np.float64),
        valid_pick_mask_sorted=valid_mask,
        datum_trace_shift_s_sorted=np.asarray(datum_shift, dtype=np.float64),
        source_elevation_m_sorted=np.asarray(
            [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            dtype=np.float64,
        ),
        receiver_elevation_m_sorted=np.asarray(
            [150.0, 140.0, 130.0, 120.0, 110.0, 100.0],
            dtype=np.float64,
        ),
        offset_sorted=np.asarray(offset, dtype=np.float64),
        key1_sorted=np.asarray([20, 10, 20, 10, 20, 10], dtype=np.int64),
        key2_sorted=np.asarray([4, 1, 5, 2, 6, 3], dtype=np.int64),
        dt=DT,
        n_traces=6,
        n_samples=N_SAMPLES,
        offset_byte=OFFSET_BYTE,
        source_kind='batch_npz',
        metadata={
            'datum_elevation_m': 500.0,
            'replacement_velocity_m_s': 2000.0,
            'key1_byte': 189,
            'key2_byte': 193,
            'pick_source_metadata': {
                'source': 'test',
                'weights': np.asarray([1, 2], dtype=np.int64),
            },
        },
    )


def _metrics(inputs: FirstBreakQcInputs | None = None) -> FirstBreakQcMetrics:
    actual_inputs = _base_inputs() if inputs is None else inputs
    metrics = compute_first_break_qc_metrics(actual_inputs)
    linear_model = metrics.linear_moveout_model_s_sorted.copy()
    linear_model[~metrics.residual_valid_mask_sorted] = np.nan
    return replace(metrics, linear_moveout_model_s_sorted=linear_model)


def _write(
    job_dir: Path,
    *,
    inputs: FirstBreakQcInputs | None = None,
    metrics: FirstBreakQcMetrics | None = None,
    **kwargs: Any,
) -> FirstBreakQcArtifactPaths:
    actual_inputs = _base_inputs() if inputs is None else inputs
    actual_metrics = _metrics(actual_inputs) if metrics is None else metrics
    return write_first_break_qc_artifacts(
        job_dir=job_dir,
        inputs=actual_inputs,
        metrics=actual_metrics,
        **kwargs,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def test_write_first_break_qc_artifacts_writes_json(tmp_path: Path) -> None:
    paths = _write(tmp_path, pick_source_artifact_name='manual_picks.npz')

    assert paths.qc_json == tmp_path / FIRST_BREAK_QC_JSON_NAME
    assert paths.qc_csv == tmp_path / FIRST_BREAK_QC_CSV_NAME
    assert paths.residual_by_key1_csv == tmp_path / RESIDUAL_BY_KEY1_CSV_NAME

    payload = json.loads(paths.qc_json.read_text(encoding='utf-8'))

    assert payload['schema_version'] == 1
    assert payload['artifact_type'] == 'first_break_qc'
    assert payload['order'] == 'trace_store_sorted'
    assert payload['sign_convention'] == (
        'pick_time_after_datum_s = pick_time_raw_s + datum_trace_shift_s'
    )
    assert payload['n_traces'] == 6
    assert payload['n_samples'] == N_SAMPLES
    assert payload['dt'] == pytest.approx(DT)
    assert payload['pick_source'] == {
        'artifact': 'manual_picks.npz',
        'kind': 'batch_npz',
        'metadata': {'source': 'test', 'weights': [1, 2]},
        'n_nan': 1,
        'n_valid': 5,
    }
    assert payload['datum_solution'] == {
        'artifact': 'datum_static_solution.npz',
        'datum_elevation_m': 500.0,
        'replacement_velocity_m_s': 2000.0,
        'key1_byte': 189,
        'key2_byte': 193,
    }
    assert payload['offset'] == {
        'offset_byte': OFFSET_BYTE,
        'unit': 'header_value',
        'uses_abs_offset_for_model': True,
    }
    assert payload['artifacts'] == {
        'trace_csv': FIRST_BREAK_QC_CSV_NAME,
        'residual_by_key1_csv': RESIDUAL_BY_KEY1_CSV_NAME,
    }


def test_write_first_break_qc_artifacts_json_contains_no_nan_or_inf(
    tmp_path: Path,
) -> None:
    payload = json.loads(_write(tmp_path).qc_json.read_text(encoding='utf-8'))

    json.dumps(payload, allow_nan=False)
    _assert_json_contains_no_nan_or_inf(payload)


def test_write_first_break_qc_artifacts_writes_trace_csv(tmp_path: Path) -> None:
    paths = _write(tmp_path)
    rows = _read_csv(paths.qc_csv)

    assert paths.qc_csv.read_text(encoding='utf-8').splitlines()[0] == ','.join(
        _TRACE_CSV_COLUMNS
    )
    assert len(rows) == 6
    assert rows[0]['sorted_trace_index'] == '0'
    assert rows[0]['key1'] == '20'
    assert rows[0]['key2'] == '4'
    assert rows[0]['valid_pick'] == 'true'
    assert float(rows[0]['pick_time_raw_s']) == pytest.approx(0.190)
    assert float(rows[0]['datum_trace_shift_s']) == pytest.approx(0.010)
    assert float(rows[0]['pick_time_after_datum_s']) == pytest.approx(0.200)
    assert float(rows[0]['offset']) == pytest.approx(-1000.0)
    assert float(rows[0]['abs_offset']) == pytest.approx(1000.0)
    assert float(rows[0]['source_elevation_m']) == pytest.approx(100.0)
    assert float(rows[0]['receiver_elevation_m']) == pytest.approx(150.0)


def test_write_first_break_qc_artifacts_writes_residual_by_key1_csv(
    tmp_path: Path,
) -> None:
    inputs = _base_inputs()
    metrics = _metrics_with_custom_residual(inputs)

    rows = _read_csv(_write(tmp_path, inputs=inputs, metrics=metrics).residual_by_key1_csv)

    assert rows
    assert ','.join(_RESIDUAL_BY_KEY1_CSV_COLUMNS) == (
        tmp_path / RESIDUAL_BY_KEY1_CSV_NAME
    ).read_text(encoding='utf-8').splitlines()[0]
    assert [row['key1'] for row in rows] == ['10', '20']
    key10 = rows[0]
    assert key10['n_traces'] == '3'
    assert key10['n_valid_picks'] == '3'
    assert key10['n_used_residual'] == '3'
    assert float(key10['residual_median_s']) == pytest.approx(0.02)
    assert float(key10['residual_mad_s']) == pytest.approx(0.02)
    assert float(key10['residual_mean_s']) == pytest.approx(0.01)


def test_write_first_break_qc_artifacts_preserves_sorted_trace_order(
    tmp_path: Path,
) -> None:
    rows = _read_csv(_write(tmp_path).qc_csv)

    assert [row['sorted_trace_index'] for row in rows] == [
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
    ]
    assert [row['key2'] for row in rows] == ['4', '1', '5', '2', '6', '3']


def test_write_first_break_qc_artifacts_writes_blank_for_invalid_pick_nan(
    tmp_path: Path,
) -> None:
    rows = _read_csv(_write(tmp_path).qc_csv)

    invalid = rows[2]
    assert invalid['valid_pick'] == 'false'
    assert invalid['pick_time_raw_s'] == ''
    assert invalid['pick_time_after_datum_s'] == ''
    assert invalid['linear_moveout_model_s'] == ''
    assert invalid['residual_after_datum_s'] == ''


def test_write_first_break_qc_artifacts_writes_blank_for_undefined_residual(
    tmp_path: Path,
) -> None:
    inputs = _base_inputs()
    paths = _write(tmp_path, inputs=inputs, metrics=_undefined_linear_metrics(inputs))

    rows = _read_csv(paths.qc_csv)

    assert all(row['linear_moveout_model_s'] == '' for row in rows)
    assert all(row['residual_after_datum_s'] == '' for row in rows)


def test_write_first_break_qc_artifacts_writes_null_for_undefined_correlation(
    tmp_path: Path,
) -> None:
    metrics = _metrics()
    correlations = dict(metrics.correlations)
    correlations['raw_pick_vs_source_elevation'] = CorrelationQc(
        name='raw_pick_vs_source_elevation',
        x_name='pick_time_raw_s',
        y_name='source_elevation_m',
        n_used=5,
        r=None,
        status='constant_input',
    )

    payload = json.loads(
        _write(tmp_path, metrics=replace(metrics, correlations=correlations)).qc_json.read_text(
            encoding='utf-8'
        )
    )

    assert payload['correlations']['raw_pick_vs_source_elevation']['r'] is None


def test_write_first_break_qc_artifacts_writes_null_for_undefined_linear_offset_model(
    tmp_path: Path,
) -> None:
    inputs = _base_inputs()
    payload = json.loads(
        _write(tmp_path, inputs=inputs, metrics=_undefined_linear_metrics(inputs)).qc_json.read_text(
            encoding='utf-8'
        )
    )

    linear = payload['linear_offset_model']
    assert linear['status'] == 'constant_abs_offset'
    assert linear['intercept_s'] is None
    assert linear['slowness_s_per_offset_unit'] is None
    assert linear['r2'] is None


def test_write_first_break_qc_artifacts_includes_signed_offset_and_abs_offset(
    tmp_path: Path,
) -> None:
    rows = _read_csv(_write(tmp_path).qc_csv)

    assert float(rows[0]['offset']) == pytest.approx(-1000.0)
    assert float(rows[0]['abs_offset']) == pytest.approx(1000.0)
    assert float(rows[5]['offset']) == pytest.approx(-1500.0)
    assert float(rows[5]['abs_offset']) == pytest.approx(1500.0)


def test_write_first_break_qc_artifacts_rejects_shape_mismatch(
    tmp_path: Path,
) -> None:
    inputs = replace(
        _base_inputs(),
        source_elevation_m_sorted=np.asarray([1.0, 2.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match='shape mismatch'):
        _write(tmp_path, inputs=inputs, metrics=_metrics())


def test_write_first_break_qc_artifacts_rejects_non_bool_valid_mask(
    tmp_path: Path,
) -> None:
    inputs = replace(
        _base_inputs(),
        valid_pick_mask_sorted=np.asarray([1, 1, 0, 1, 1, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match='valid_pick_mask_sorted'):
        _write(tmp_path, inputs=inputs, metrics=_metrics())


def test_write_first_break_qc_artifacts_rejects_non_finite_valid_pick(
    tmp_path: Path,
) -> None:
    picks = _base_inputs().picks_time_s_sorted.copy()
    picks[0] = np.inf
    inputs = replace(_base_inputs(), picks_time_s_sorted=picks)

    with pytest.raises(ValueError, match='picks_time_s_sorted'):
        _write(tmp_path, inputs=inputs, metrics=_metrics())


def test_write_first_break_qc_artifacts_rejects_invalid_pick_not_nan(
    tmp_path: Path,
) -> None:
    picks = _base_inputs().picks_time_s_sorted.copy()
    picks[2] = 0.100
    inputs = replace(_base_inputs(), picks_time_s_sorted=picks)

    with pytest.raises(ValueError, match='invalid picks must be NaN'):
        _write(tmp_path, inputs=inputs, metrics=_metrics())


def test_write_first_break_qc_artifacts_rejects_non_finite_datum_shift(
    tmp_path: Path,
) -> None:
    datum_shift = _base_inputs().datum_trace_shift_s_sorted.copy()
    datum_shift[0] = np.nan
    inputs = replace(_base_inputs(), datum_trace_shift_s_sorted=datum_shift)

    with pytest.raises(ValueError, match='datum_trace_shift_s_sorted'):
        _write(tmp_path, inputs=inputs, metrics=_metrics())


@pytest.mark.parametrize(
    'field',
    ['source_elevation_m_sorted', 'receiver_elevation_m_sorted'],
)
def test_write_first_break_qc_artifacts_rejects_non_finite_elevation(
    tmp_path: Path,
    field: str,
) -> None:
    values = getattr(_base_inputs(), field).copy()
    values[0] = np.nan
    inputs = replace(_base_inputs(), **{field: values})

    with pytest.raises(ValueError, match=field):
        _write(tmp_path, inputs=inputs, metrics=_metrics())


def test_write_first_break_qc_artifacts_rejects_non_finite_offset(
    tmp_path: Path,
) -> None:
    offset = _base_inputs().offset_sorted.copy()
    offset[0] = np.nan
    inputs = replace(_base_inputs(), offset_sorted=offset)

    with pytest.raises(ValueError, match='offset_sorted'):
        _write(tmp_path, inputs=inputs, metrics=_metrics())


@pytest.mark.parametrize('field', ['key1_sorted', 'key2_sorted'])
def test_write_first_break_qc_artifacts_rejects_non_integer_key1_key2(
    tmp_path: Path,
    field: str,
) -> None:
    values = getattr(_base_inputs(), field).astype(np.float64)
    values[0] += 0.5
    inputs = replace(_base_inputs(), **{field: values})

    with pytest.raises(ValueError, match='integer values'):
        _write(tmp_path, inputs=inputs, metrics=_metrics())


def test_write_first_break_qc_artifacts_rejects_non_finite_linear_model(
    tmp_path: Path,
) -> None:
    inputs = _base_inputs()
    metrics = _metrics(inputs)
    linear_model = metrics.linear_moveout_model_s_sorted.copy()
    linear_model[0] = np.inf

    with pytest.raises(ValueError, match='linear_moveout_model_s_sorted'):
        _write(
            tmp_path,
            inputs=inputs,
            metrics=replace(metrics, linear_moveout_model_s_sorted=linear_model),
        )


def test_write_first_break_qc_artifacts_rejects_defined_linear_model_on_undefined_trace(
    tmp_path: Path,
) -> None:
    inputs = _base_inputs()
    metrics = _metrics(inputs)
    linear_model = metrics.linear_moveout_model_s_sorted.copy()
    undefined_indices = np.flatnonzero(~metrics.residual_valid_mask_sorted)
    linear_model[undefined_indices[0]] = 0.123

    with pytest.raises(ValueError, match='linear_moveout_model_s_sorted'):
        _write(
            tmp_path,
            inputs=inputs,
            metrics=replace(metrics, linear_moveout_model_s_sorted=linear_model),
        )


def test_write_first_break_qc_artifacts_replaces_existing_files_atomically(
    tmp_path: Path,
) -> None:
    paths = _write(tmp_path)
    paths.qc_json.write_text('stale', encoding='utf-8')
    inputs = replace(_base_inputs(), source_kind='manual')

    _write(tmp_path, inputs=inputs)
    payload = json.loads(paths.qc_json.read_text(encoding='utf-8'))

    assert payload['pick_source']['kind'] == 'manual'


def test_write_first_break_qc_artifacts_cleans_tmp_files_on_failure(
    tmp_path: Path,
) -> None:
    (tmp_path / FIRST_BREAK_QC_CSV_NAME).mkdir()

    with pytest.raises(OSError):
        _write(tmp_path)

    assert list(tmp_path.glob('*.tmp-*')) == []


def _metrics_with_custom_residual(
    inputs: FirstBreakQcInputs,
) -> FirstBreakQcMetrics:
    metrics = _metrics(inputs)
    residual = np.asarray(
        [0.01, -0.03, np.nan, 0.02, -0.02, 0.04],
        dtype=np.float64,
    )
    residual_valid_mask = inputs.valid_pick_mask_sorted.copy()
    linear_model = metrics.linear_moveout_model_s_sorted.copy()
    linear_model[residual_valid_mask] = (
        metrics.pick_time_after_datum_s_sorted[residual_valid_mask]
        - residual[residual_valid_mask]
    )
    return replace(
        metrics,
        linear_moveout_model_s_sorted=linear_model,
        residual_after_datum_s_sorted=residual,
        residual_valid_mask_sorted=residual_valid_mask,
        residual_stats=compute_finite_series_stats(
            'residual_after_datum_s',
            residual,
            residual_valid_mask,
        ),
        residual_by_key1=compute_residual_by_key1(
            residual,
            residual_valid_mask,
            inputs.key1_sorted,
        ),
    )


def _undefined_linear_metrics(inputs: FirstBreakQcInputs) -> FirstBreakQcMetrics:
    metrics = _metrics(inputs)
    residual_valid_mask = np.zeros(inputs.n_traces, dtype=bool)
    residual = np.full(inputs.n_traces, np.nan, dtype=np.float64)
    return replace(
        metrics,
        linear_offset_fit=LinearOffsetFit(
            n_used=int(np.count_nonzero(inputs.valid_pick_mask_sorted)),
            intercept_s=None,
            slowness_s_per_offset_unit=None,
            r2=None,
            status='constant_abs_offset',
        ),
        linear_moveout_model_s_sorted=np.full(inputs.n_traces, np.nan, dtype=np.float64),
        residual_after_datum_s_sorted=residual,
        residual_valid_mask_sorted=residual_valid_mask,
        residual_stats=compute_finite_series_stats(
            'residual_after_datum_s',
            residual,
            residual_valid_mask,
        ),
        residual_by_key1=[],
    )


def _assert_json_contains_no_nan_or_inf(value: Any) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _assert_json_contains_no_nan_or_inf(nested)
        return
    if isinstance(value, list):
        for nested in value:
            _assert_json_contains_no_nan_or_inf(nested)
        return
    if isinstance(value, float):
        assert np.isfinite(value)
