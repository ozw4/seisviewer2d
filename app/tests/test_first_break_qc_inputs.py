from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.services.first_break_qc_inputs import (
    build_first_break_qc_inputs,
    load_datum_static_solution_npz,
    load_offset_header_sorted,
)
from app.services.pick_source_loader import LoadedPickSource

KEY1_BYTE = 189
KEY2_BYTE = 193
OFFSET_BYTE = 37
SOURCE_ELEVATION_BYTE = 45
RECEIVER_ELEVATION_BYTE = 41
DT = 0.004
N_SAMPLES = 8
KEY1_SORTED = np.asarray([10, 10, 20, 20], dtype=np.int64)
KEY2_SORTED = np.asarray([1, 2, 1, 2], dtype=np.int64)
SOURCE_SHIFT = np.asarray([0.100, 0.110, 0.120, 0.130], dtype=np.float64)
RECEIVER_SHIFT = np.asarray([0.010, 0.020, 0.030, 0.040], dtype=np.float64)
TRACE_SHIFT = SOURCE_SHIFT + RECEIVER_SHIFT
SOURCE_ELEVATION = np.asarray([100.0, 105.0, 110.0, 115.0], dtype=np.float64)
RECEIVER_ELEVATION = np.asarray([90.0, 95.0, 100.0, 105.0], dtype=np.float64)
OFFSET_SORTED = np.asarray([-1200.0, -400.0, 350.0, 1250.0], dtype=np.float64)
PICKS_SORTED = np.asarray([0.004, np.nan, 0.012, 0.020], dtype=np.float64)
VALID_PICK_MASK = np.asarray([True, False, True, True], dtype=bool)
_ABSENT = object()


class _Reader:
    def __init__(
        self,
        *,
        key1_sorted: np.ndarray = KEY1_SORTED,
        key2_sorted: np.ndarray = KEY2_SORTED,
        offset_sorted: np.ndarray = OFFSET_SORTED,
        n_samples: int = N_SAMPLES,
        dt: float | object = DT,
        include_key_bytes: bool = True,
    ) -> None:
        if include_key_bytes:
            self.key1_byte = KEY1_BYTE
            self.key2_byte = KEY2_BYTE
        self.traces = np.zeros((int(np.asarray(key1_sorted).shape[0]), n_samples))
        if dt is _ABSENT:
            self.meta = {}
        else:
            self.meta = {'dt': dt}
        self.headers = {
            KEY1_BYTE: np.asarray(key1_sorted),
            KEY2_BYTE: np.asarray(key2_sorted),
            OFFSET_BYTE: np.asarray(offset_sorted),
        }

    def ensure_header(self, byte: int) -> np.ndarray:
        return self.headers[int(byte)]

    def get_n_samples(self) -> int:
        return int(self.traces.shape[-1])


def _base_solution_payload() -> dict[str, Any]:
    return {
        'trace_shift_s_sorted': TRACE_SHIFT,
        'source_shift_s_sorted': SOURCE_SHIFT,
        'receiver_shift_s_sorted': RECEIVER_SHIFT,
        'source_elevation_m_sorted': SOURCE_ELEVATION,
        'receiver_elevation_m_sorted': RECEIVER_ELEVATION,
        'key1_sorted': KEY1_SORTED,
        'key2_sorted': KEY2_SORTED,
        'datum_elevation_m': np.float64(500.0),
        'replacement_velocity_m_s': np.float64(2000.0),
        'dt': np.float64(DT),
        'n_traces': np.int64(4),
        'key1_byte': np.int64(KEY1_BYTE),
        'key2_byte': np.int64(KEY2_BYTE),
        'source_elevation_byte': np.int64(SOURCE_ELEVATION_BYTE),
        'receiver_elevation_byte': np.int64(RECEIVER_ELEVATION_BYTE),
        'source_surface_elevation_m_sorted': SOURCE_ELEVATION,
        'source_depth_m_sorted': np.zeros(4, dtype=np.float64),
        'source_depth_used_sorted': np.zeros(4, dtype=bool),
        'source_depth_byte': np.int64(-1),
        'elevation_unit': np.asarray('m'),
    }


def _write_solution(path: Path, **overrides: Any) -> Path:
    payload = _base_solution_payload()
    for key, value in overrides.items():
        if value is _ABSENT:
            payload.pop(key, None)
        else:
            payload[key] = value
    np.savez(path, **payload)
    return path


def _load_solution(path: Path):
    return load_datum_static_solution_npz(
        path,
        expected_n_traces=4,
        expected_dt=DT,
        expected_key1_byte=KEY1_BYTE,
        expected_key2_byte=KEY2_BYTE,
    )


def _pick_source(**overrides: Any) -> LoadedPickSource:
    source = LoadedPickSource(
        picks_time_s_sorted=PICKS_SORTED,
        valid_mask_sorted=VALID_PICK_MASK,
        source_kind='batch_npz',
        n_traces=4,
        n_samples=N_SAMPLES,
        dt=DT,
        n_valid=3,
        n_nan=1,
        metadata={'source': 'test'},
    )
    if not overrides:
        return source
    return replace(source, **overrides)


def test_load_datum_static_solution_npz_success(tmp_path: Path) -> None:
    path = _write_solution(tmp_path / 'datum_static_solution.npz')

    solution = _load_solution(path)

    assert solution.n_traces == 4
    assert solution.dt == pytest.approx(DT)
    assert solution.key1_byte == KEY1_BYTE
    assert solution.key2_byte == KEY2_BYTE
    assert solution.source_elevation_byte == SOURCE_ELEVATION_BYTE
    assert solution.receiver_elevation_byte == RECEIVER_ELEVATION_BYTE
    np.testing.assert_allclose(solution.trace_shift_s_sorted, TRACE_SHIFT)
    np.testing.assert_array_equal(solution.key1_sorted, KEY1_SORTED)
    np.testing.assert_array_equal(solution.key2_sorted, KEY2_SORTED)
    assert solution.metadata['source_depth_byte'] == -1
    assert solution.metadata['elevation_unit'] == 'm'


def test_load_datum_static_solution_npz_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='not found'):
        _load_solution(tmp_path / 'missing.npz')


def test_load_datum_static_solution_npz_rejects_missing_required_key(
    tmp_path: Path,
) -> None:
    path = _write_solution(
        tmp_path / 'datum_static_solution.npz',
        trace_shift_s_sorted=_ABSENT,
    )

    with pytest.raises(ValueError, match='trace_shift_s_sorted'):
        _load_solution(path)


def test_load_datum_static_solution_npz_rejects_shape_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_solution(
        tmp_path / 'datum_static_solution.npz',
        source_shift_s_sorted=SOURCE_SHIFT[:3],
    )

    with pytest.raises(ValueError, match='shape mismatch'):
        _load_solution(path)


def test_load_datum_static_solution_npz_rejects_dt_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_solution(tmp_path / 'datum_static_solution.npz', dt=np.float64(0.002))

    with pytest.raises(ValueError, match='dt mismatch'):
        _load_solution(path)


def test_load_datum_static_solution_npz_rejects_n_traces_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_solution(
        tmp_path / 'datum_static_solution.npz',
        n_traces=np.int64(5),
    )

    with pytest.raises(ValueError, match='n_traces mismatch'):
        _load_solution(path)


def test_load_datum_static_solution_npz_rejects_key_byte_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_solution(
        tmp_path / 'datum_static_solution.npz',
        key2_byte=np.int64(195),
    )

    with pytest.raises(ValueError, match='key2_byte mismatch'):
        _load_solution(path)


def test_load_datum_static_solution_npz_rejects_non_finite_shift(
    tmp_path: Path,
) -> None:
    shifts = SOURCE_SHIFT.copy()
    shifts[1] = np.inf
    path = _write_solution(
        tmp_path / 'datum_static_solution.npz',
        source_shift_s_sorted=shifts,
    )

    with pytest.raises(ValueError, match='source_shift_s_sorted'):
        _load_solution(path)


def test_load_datum_static_solution_npz_rejects_non_finite_elevation(
    tmp_path: Path,
) -> None:
    elevations = RECEIVER_ELEVATION.copy()
    elevations[2] = np.nan
    path = _write_solution(
        tmp_path / 'datum_static_solution.npz',
        receiver_elevation_m_sorted=elevations,
    )

    with pytest.raises(ValueError, match='receiver_elevation_m_sorted'):
        _load_solution(path)


def test_load_datum_static_solution_npz_rejects_shift_component_mismatch(
    tmp_path: Path,
) -> None:
    trace_shift = TRACE_SHIFT.copy()
    trace_shift[0] += 0.01
    path = _write_solution(
        tmp_path / 'datum_static_solution.npz',
        trace_shift_s_sorted=trace_shift,
    )

    with pytest.raises(ValueError, match='source_shift_s_sorted'):
        _load_solution(path)


def test_load_offset_header_sorted_success() -> None:
    offset = load_offset_header_sorted(
        _Reader(),
        offset_byte=OFFSET_BYTE,
        expected_n_traces=4,
    )

    np.testing.assert_allclose(offset, OFFSET_SORTED)


def test_load_offset_header_sorted_rejects_shape_mismatch() -> None:
    reader = _Reader(offset_sorted=OFFSET_SORTED[:3])

    with pytest.raises(ValueError, match='shape mismatch'):
        load_offset_header_sorted(reader, offset_byte=OFFSET_BYTE, expected_n_traces=4)


def test_load_offset_header_sorted_rejects_non_finite() -> None:
    offset = OFFSET_SORTED.copy()
    offset[0] = np.inf
    reader = _Reader(offset_sorted=offset)

    with pytest.raises(ValueError, match='finite values'):
        load_offset_header_sorted(reader, offset_byte=OFFSET_BYTE, expected_n_traces=4)


def test_build_first_break_qc_inputs_success(tmp_path: Path) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    inputs = build_first_break_qc_inputs(
        pick_source=_pick_source(),
        solution_npz_path=solution_path,
        reader=_Reader(),
        offset_byte=OFFSET_BYTE,
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
        expected_key1_byte=KEY1_BYTE,
        expected_key2_byte=KEY2_BYTE,
    )

    np.testing.assert_allclose(inputs.picks_time_s_sorted, PICKS_SORTED, equal_nan=True)
    np.testing.assert_array_equal(inputs.valid_pick_mask_sorted, VALID_PICK_MASK)
    np.testing.assert_allclose(inputs.datum_trace_shift_s_sorted, TRACE_SHIFT)
    np.testing.assert_allclose(inputs.source_elevation_m_sorted, SOURCE_ELEVATION)
    np.testing.assert_allclose(inputs.receiver_elevation_m_sorted, RECEIVER_ELEVATION)
    np.testing.assert_allclose(inputs.offset_sorted, OFFSET_SORTED)
    np.testing.assert_array_equal(inputs.key1_sorted, KEY1_SORTED)
    np.testing.assert_array_equal(inputs.key2_sorted, KEY2_SORTED)
    assert inputs.source_kind == 'batch_npz'
    assert inputs.metadata['solution_artifact'] == 'datum_static_solution.npz'
    assert inputs.metadata['offset_byte'] == OFFSET_BYTE
    assert inputs.metadata['order'] == 'trace_store_sorted'


def test_build_first_break_qc_inputs_rejects_pick_dt_mismatch(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    with pytest.raises(ValueError, match='pick source dt mismatch'):
        build_first_break_qc_inputs(
            pick_source=_pick_source(dt=0.002),
            solution_npz_path=solution_path,
            reader=_Reader(),
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
            expected_key1_byte=KEY1_BYTE,
            expected_key2_byte=KEY2_BYTE,
        )


def test_build_first_break_qc_inputs_rejects_pick_n_traces_mismatch(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    with pytest.raises(ValueError, match='pick source n_traces mismatch'):
        build_first_break_qc_inputs(
            pick_source=_pick_source(n_traces=5),
            solution_npz_path=solution_path,
            reader=_Reader(),
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
            expected_key1_byte=KEY1_BYTE,
            expected_key2_byte=KEY2_BYTE,
        )


def test_build_first_break_qc_inputs_rejects_missing_reader_key_bytes(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    with pytest.raises(ValueError, match='reader key1_byte is required'):
        build_first_break_qc_inputs(
            pick_source=_pick_source(),
            solution_npz_path=solution_path,
            reader=_Reader(include_key_bytes=False),
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
            expected_key1_byte=KEY1_BYTE,
            expected_key2_byte=KEY2_BYTE,
        )


def test_build_first_break_qc_inputs_rejects_missing_reader_dt(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    with pytest.raises(ValueError, match='reader meta missing dt'):
        build_first_break_qc_inputs(
            pick_source=_pick_source(),
            solution_npz_path=solution_path,
            reader=_Reader(dt=_ABSENT),
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
            expected_key1_byte=KEY1_BYTE,
            expected_key2_byte=KEY2_BYTE,
        )


def test_build_first_break_qc_inputs_rejects_reader_dt_mismatch(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    with pytest.raises(ValueError, match='reader dt mismatch'):
        build_first_break_qc_inputs(
            pick_source=_pick_source(),
            solution_npz_path=solution_path,
            reader=_Reader(dt=0.002),
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
            expected_key1_byte=KEY1_BYTE,
            expected_key2_byte=KEY2_BYTE,
        )


def test_build_first_break_qc_inputs_rejects_key1_key2_header_mismatch(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')
    key1 = KEY1_SORTED.copy()
    key1[0] = 99

    with pytest.raises(ValueError, match='key1_sorted does not match'):
        build_first_break_qc_inputs(
            pick_source=_pick_source(),
            solution_npz_path=solution_path,
            reader=_Reader(key1_sorted=key1),
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
            expected_key1_byte=KEY1_BYTE,
            expected_key2_byte=KEY2_BYTE,
        )


def test_build_first_break_qc_inputs_preserves_signed_offset(tmp_path: Path) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    inputs = build_first_break_qc_inputs(
        pick_source=_pick_source(),
        solution_npz_path=solution_path,
        reader=_Reader(),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
        expected_key1_byte=KEY1_BYTE,
        expected_key2_byte=KEY2_BYTE,
    )

    assert inputs.offset_sorted[0] < 0
    assert inputs.offset_sorted[-1] > 0
    np.testing.assert_allclose(inputs.offset_sorted, OFFSET_SORTED)
