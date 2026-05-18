from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.services.datum_static_validation import (
    ExistingStaticHeaderCheck,
    TraceShiftValidationResult,
)
from app.services.static_artifacts import (
    DatumStaticArtifactPaths,
    write_datum_static_artifacts,
)

_REQUIRED_NPZ_KEYS = {
    'trace_shift_s_sorted',
    'source_shift_s_sorted',
    'receiver_shift_s_sorted',
    'source_surface_elevation_m_sorted',
    'source_depth_m_sorted',
    'source_depth_used_sorted',
    'source_elevation_m_sorted',
    'receiver_elevation_m_sorted',
    'key1_sorted',
    'key2_sorted',
    'datum_elevation_m',
    'replacement_velocity_m_s',
    'dt',
    'n_traces',
    'key1_byte',
    'key2_byte',
    'source_elevation_byte',
    'receiver_elevation_byte',
    'elevation_scalar_byte',
    'source_depth_byte',
    'source_depth_enabled',
    'elevation_unit',
    'elevation_scalar_zero_count',
    'header_source_segy_path',
}

_CSV_COLUMNS = [
    'sorted_trace_index',
    'key1',
    'key2',
    'source_surface_elevation_m',
    'source_depth_m',
    'source_depth_used',
    'source_elevation_m',
    'receiver_elevation_m',
    'source_shift_ms',
    'receiver_shift_ms',
    'trace_shift_ms',
]


def _existing_static_check() -> ExistingStaticHeaderCheck:
    return ExistingStaticHeaderCheck(
        policy='fail_if_nonzero',
        checked=True,
        source_static_byte=99,
        receiver_static_byte=101,
        total_static_byte=103,
        nonzero_source_static_count=0,
        nonzero_receiver_static_count=0,
        nonzero_total_static_count=0,
        nonzero_any_count=0,
        checked_bytes=(99, 101, 103),
    )


def _trace_shift_validation() -> TraceShiftValidationResult:
    return TraceShiftValidationResult(
        n_traces=2,
        max_abs_shift_ms=250.0,
        min_shift_ms=-100.0,
        max_shift_ms=-100.0,
        mean_shift_ms=-100.0,
        max_abs_observed_shift_ms=100.0,
    )


def _base_kwargs(job_dir: Path) -> dict[str, Any]:
    return {
        'job_dir': job_dir,
        'trace_shift_s_sorted': np.array([-0.10, -0.10], dtype=np.float64),
        'source_shift_s_sorted': np.array([-0.05, -0.06], dtype=np.float64),
        'receiver_shift_s_sorted': np.array([-0.05, -0.04], dtype=np.float64),
        'source_surface_elevation_m_sorted': np.array(
            [100.0, 120.0],
            dtype=np.float64,
        ),
        'source_depth_m_sorted': np.array([0.0, 10.0], dtype=np.float64),
        'source_depth_used_sorted': np.array([False, True], dtype=bool),
        'source_elevation_m_sorted': np.array([100.0, 110.0], dtype=np.float64),
        'receiver_elevation_m_sorted': np.array([100.0, 80.0], dtype=np.float64),
        'key1_sorted': np.array([1, 1], dtype=np.int64),
        'key2_sorted': np.array([10, 11], dtype=np.int64),
        'datum_elevation_m': 500.0,
        'replacement_velocity_m_s': 2000.0,
        'dt': 0.004,
        'key1_byte': 189,
        'key2_byte': 193,
        'source_elevation_byte': 41,
        'receiver_elevation_byte': 45,
        'elevation_scalar_byte': 69,
        'source_depth_byte': None,
        'source_depth_enabled': True,
        'elevation_unit': 'm',
        'elevation_scalar_zero_count': 1,
        'existing_static_check': _existing_static_check(),
        'trace_shift_validation': _trace_shift_validation(),
        'header_source_segy_path': None,
    }


def _write(job_dir: Path, **overrides: Any) -> DatumStaticArtifactPaths:
    kwargs = _base_kwargs(job_dir)
    kwargs.update(overrides)
    return write_datum_static_artifacts(**kwargs)


def test_write_datum_static_artifacts_creates_job_dir(tmp_path: Path) -> None:
    job_dir = tmp_path / 'nested' / 'job'

    paths = _write(job_dir)

    assert job_dir.is_dir()
    assert paths.solution_npz == job_dir / 'datum_static_solution.npz'
    assert paths.qc_json == job_dir / 'datum_static_qc.json'
    assert paths.statics_csv == job_dir / 'datum_statics.csv'


def test_write_datum_static_artifacts_writes_solution_npz(tmp_path: Path) -> None:
    paths = _write(tmp_path)

    with np.load(paths.solution_npz, allow_pickle=False) as data:
        assert _REQUIRED_NPZ_KEYS.issubset(data.files)
        for key in data.files:
            assert data[key].dtype != object
        np.testing.assert_allclose(
            data['trace_shift_s_sorted'],
            np.array([-0.10, -0.10], dtype=np.float64),
        )
        np.testing.assert_array_equal(data['source_depth_used_sorted'], [False, True])
        np.testing.assert_array_equal(data['key1_sorted'], [1, 1])
        np.testing.assert_array_equal(data['key2_sorted'], [10, 11])
        assert data['source_depth_byte'].item() == -1
        assert data['header_source_segy_path'].item() == ''
        assert data['n_traces'].item() == 2
        assert data['elevation_unit'].item() == 'm'


def test_write_datum_static_artifacts_writes_qc_json(tmp_path: Path) -> None:
    paths = _write(tmp_path)

    payload = json.loads(paths.qc_json.read_text(encoding='utf-8'))

    assert payload['n_traces'] == 2
    assert payload['dt'] == pytest.approx(0.004)
    assert payload['datum_elevation_m'] == pytest.approx(500.0)
    assert payload['replacement_velocity_m_s'] == pytest.approx(2000.0)
    assert payload['source_depth_enabled'] is True
    assert payload['elevation_unit'] == 'm'
    assert payload['source_elevation_m'] == {
        'min': 100.0,
        'max': 110.0,
        'mean': 105.0,
    }
    assert payload['receiver_elevation_m'] == {
        'min': 80.0,
        'max': 100.0,
        'mean': 90.0,
    }
    assert payload['source_shift_ms']['min'] == pytest.approx(-60.0)
    assert payload['source_shift_ms']['max_abs'] == pytest.approx(60.0)
    assert payload['receiver_shift_ms']['mean'] == pytest.approx(-45.0)
    assert payload['trace_shift_ms']['max_abs'] == pytest.approx(100.0)
    assert payload['scalar'] == {'zero_count': 1, 'zero_fraction': 0.5}
    assert payload['existing_statics'] == {
        'checked': True,
        'nonzero_any_count': 0,
        'nonzero_receiver_static_count': 0,
        'nonzero_source_static_count': 0,
        'nonzero_total_static_count': 0,
        'policy': 'fail_if_nonzero',
    }
    assert payload['validation'] == {
        'max_abs_observed_shift_ms': 100.0,
        'max_abs_shift_ms': 250.0,
    }
    _assert_json_contains_no_nan_or_inf(payload)


def test_qc_json_marks_optional_checks_unchecked(tmp_path: Path) -> None:
    paths = _write(
        tmp_path,
        existing_static_check=None,
        trace_shift_validation=None,
    )

    payload = json.loads(paths.qc_json.read_text(encoding='utf-8'))

    assert payload['existing_statics'] == {'checked': False}
    assert payload['validation'] == {'max_abs_shift_checked': False}


def test_write_datum_static_artifacts_writes_csv(tmp_path: Path) -> None:
    paths = _write(tmp_path)

    with paths.statics_csv.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert rows[0].keys() == set(_CSV_COLUMNS)
    assert Path(paths.statics_csv).read_text(encoding='utf-8').splitlines()[
        0
    ] == ','.join(_CSV_COLUMNS)
    assert len(rows) == 2
    assert rows[0]['sorted_trace_index'] == '0'
    assert rows[0]['key1'] == '1'
    assert rows[0]['key2'] == '10'
    assert rows[0]['source_depth_used'] == 'false'
    assert rows[1]['source_depth_used'] == 'true'
    assert float(rows[0]['source_shift_ms']) == pytest.approx(-50.0)
    assert float(rows[0]['receiver_shift_ms']) == pytest.approx(-50.0)
    assert float(rows[0]['trace_shift_ms']) == pytest.approx(-100.0)


def test_writer_rejects_shape_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='shape mismatch'):
        _write(
            tmp_path,
            source_shift_s_sorted=np.array([-0.05], dtype=np.float64),
        )


def test_writer_rejects_non_finite_arrays(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='finite values'):
        _write(
            tmp_path,
            receiver_elevation_m_sorted=np.array([100.0, np.inf], dtype=np.float64),
        )


@pytest.mark.parametrize('dt', [0.0, -0.001, np.nan, np.inf])
def test_writer_rejects_invalid_dt(tmp_path: Path, dt: float) -> None:
    with pytest.raises(ValueError, match='dt'):
        _write(tmp_path, dt=dt)


@pytest.mark.parametrize('replacement_velocity_m_s', [0.0, -1.0, np.nan, np.inf])
def test_writer_rejects_invalid_replacement_velocity(
    tmp_path: Path,
    replacement_velocity_m_s: float,
) -> None:
    with pytest.raises(ValueError, match='replacement_velocity_m_s'):
        _write(tmp_path, replacement_velocity_m_s=replacement_velocity_m_s)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('key1_byte', 0),
        ('key2_byte', 241),
        ('source_elevation_byte', True),
        ('receiver_elevation_byte', 0),
        ('elevation_scalar_byte', 241),
        ('source_depth_byte', 0),
    ],
)
def test_writer_rejects_invalid_header_byte(
    tmp_path: Path,
    field: str,
    value: int | bool,
) -> None:
    with pytest.raises(ValueError, match=field):
        _write(tmp_path, **{field: value})


def test_writer_rejects_unknown_elevation_unit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='elevation_unit'):
        _write(tmp_path, elevation_unit='yards')


def test_writer_rejects_inconsistent_trace_shift_sum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='source_shift_s_sorted'):
        _write(
            tmp_path,
            trace_shift_s_sorted=np.array([-0.10, -0.12], dtype=np.float64),
        )


def test_writer_rejects_source_depth_when_disabled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='source_depth_m_sorted'):
        _write(
            tmp_path,
            source_depth_enabled=False,
            source_depth_m_sorted=np.array([0.0, 10.0], dtype=np.float64),
            source_depth_used_sorted=np.array([False, False], dtype=bool),
        )


def test_writer_replaces_existing_artifacts(tmp_path: Path) -> None:
    paths = _write(tmp_path)
    paths.qc_json.write_text('stale', encoding='utf-8')

    _write(tmp_path, datum_elevation_m=600.0)
    payload = json.loads(paths.qc_json.read_text(encoding='utf-8'))

    assert payload['datum_elevation_m'] == pytest.approx(600.0)


def test_writer_cleans_tmp_files_on_failure(tmp_path: Path) -> None:
    job_dir = tmp_path / 'job'
    job_dir.mkdir()
    (job_dir / 'datum_statics.csv').mkdir()

    with pytest.raises(OSError):
        _write(job_dir)

    assert list(job_dir.glob('*.tmp-*')) == []


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
