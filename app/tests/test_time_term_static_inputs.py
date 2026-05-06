from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.api.schemas import TimeTermStaticApplyRequest
from app.core.state import create_app_state
from app.services.geometry_linkage_artifacts import (
    GEOMETRY_LINKAGE_NPZ_NAME,
    GeometryLinkageArtifactMetadata,
    build_geometry_linkage_solution_arrays,
)
from app.services.geometry_linkage_linker import (
    GeometryLinkageOptions,
    build_geometry_linkage,
)
from app.services.geometry_linkage_tables import build_endpoint_geometry_tables
from app.services.geometry_linkage_validation import GeometryLinkageHeaders
from app.services.pick_source_loader import LoadedPickSource
from app.services.time_term_static_inputs import (
    build_time_term_inversion_inputs,
    build_time_term_inversion_inputs_from_sources,
    load_time_term_residual_applied_shift,
    summarize_time_term_inversion_inputs,
)

FILE_ID = 'file-id'
KEY1_BYTE = 189
KEY2_BYTE = 193
SOURCE_ID_BYTE = 9
RECEIVER_ID_BYTE = 13
OFFSET_BYTE = 37
SOURCE_ELEVATION_BYTE = 45
RECEIVER_ELEVATION_BYTE = 41
SOURCE_DEPTH_BYTE = 49
ELEVATION_SCALAR_BYTE = 69
COORDINATE_SCALAR_BYTE = 71
SOURCE_X_BYTE = 73
SOURCE_Y_BYTE = 77
RECEIVER_X_BYTE = 81
RECEIVER_Y_BYTE = 85
DT = 0.004
N_SAMPLES = 80
N_TRACES = 4
SORTED_TO_ORIGINAL = np.asarray([2, 0, 3, 1], dtype=np.int64)
PICKS_ORIGINAL = np.asarray([0.020, np.nan, 0.010, 0.030], dtype=np.float32)
PICKS_SORTED = np.asarray([0.010, 0.020, 0.030, np.nan], dtype=np.float64)
VALID_PICK_MASK = np.asarray([True, True, True, False], dtype=bool)
DATUM_SHIFT = np.asarray([0.001, 0.002, 0.003, 0.004], dtype=np.float64)
RESIDUAL_SHIFT = np.asarray([-0.0005, 0.001, -0.001, 0.002], dtype=np.float64)


class _Reader:
    def __init__(
        self,
        *,
        headers: dict[int, np.ndarray] | None = None,
        missing_headers: set[int] | None = None,
        n_samples: int = N_SAMPLES,
        dt: float = DT,
        sorted_to_original: np.ndarray = SORTED_TO_ORIGINAL,
    ) -> None:
        self.key1_byte = KEY1_BYTE
        self.key2_byte = KEY2_BYTE
        self.traces = np.zeros((N_TRACES, n_samples), dtype=np.float32)
        self.meta = {'dt': dt, 'n_traces': N_TRACES}
        self._sorted_to_original = np.asarray(sorted_to_original, dtype=np.int64)
        self.headers = _base_headers()
        if headers is not None:
            self.headers.update(headers)
        if missing_headers is not None:
            for byte in missing_headers:
                self.headers.pop(byte, None)

    def ensure_header(self, byte: int) -> np.ndarray:
        if int(byte) not in self.headers:
            raise ValueError(f'missing header byte {byte}')
        return self.headers[int(byte)]

    def get_sorted_to_original(self) -> np.ndarray:
        return self._sorted_to_original

    def get_n_samples(self) -> int:
        return int(self.traces.shape[-1])


def _base_headers() -> dict[int, np.ndarray]:
    return {
        KEY1_BYTE: np.asarray([10, 10, 20, 20], dtype=np.int64),
        KEY2_BYTE: np.asarray([1, 2, 1, 2], dtype=np.int64),
        SOURCE_ID_BYTE: np.asarray([100, 100, 200, 300], dtype=np.int64),
        RECEIVER_ID_BYTE: np.asarray([1, 2, 1, 2], dtype=np.int64),
        OFFSET_BYTE: np.asarray([-100.0, -50.0, 50.0, 100.0], dtype=np.float64),
        COORDINATE_SCALAR_BYTE: np.asarray([1, -10, 2, 0], dtype=np.int16),
        SOURCE_X_BYTE: np.asarray([100.0, 2000.0, 10.0, 5.0], dtype=np.float64),
        SOURCE_Y_BYTE: np.asarray([0.0, 100.0, 10.0, 20.0], dtype=np.float64),
        RECEIVER_X_BYTE: np.asarray([300.0, 4000.0, 20.0, 15.0], dtype=np.float64),
        RECEIVER_Y_BYTE: np.asarray([0.0, 200.0, 20.0, 40.0], dtype=np.float64),
        ELEVATION_SCALAR_BYTE: np.asarray([1, -10, 2, 0], dtype=np.int16),
        SOURCE_ELEVATION_BYTE: np.asarray(
            [1000.0, 2000.0, 300.0, 400.0],
            dtype=np.float64,
        ),
        RECEIVER_ELEVATION_BYTE: np.asarray(
            [900.0, 1000.0, 150.0, 200.0],
            dtype=np.float64,
        ),
        SOURCE_DEPTH_BYTE: np.asarray([10.0, 100.0, 5.0, 0.0], dtype=np.float64),
    }


def _request(**overrides: Any) -> TimeTermStaticApplyRequest:
    payload: dict[str, Any] = {
        'file_id': FILE_ID,
        'key1_byte': KEY1_BYTE,
        'key2_byte': KEY2_BYTE,
        'pick_source': {
            'kind': 'batch_predicted_npz',
            'job_id': 'pick-job',
            'artifact_name': 'predicted_picks_time_s.npz',
        },
        'geometry': {
            'source_id_byte': SOURCE_ID_BYTE,
            'receiver_id_byte': RECEIVER_ID_BYTE,
            'source_x_byte': SOURCE_X_BYTE,
            'source_y_byte': SOURCE_Y_BYTE,
            'receiver_x_byte': RECEIVER_X_BYTE,
            'receiver_y_byte': RECEIVER_Y_BYTE,
            'source_elevation_byte': SOURCE_ELEVATION_BYTE,
            'receiver_elevation_byte': RECEIVER_ELEVATION_BYTE,
            'source_depth_byte': SOURCE_DEPTH_BYTE,
            'coordinate_scalar_byte': COORDINATE_SCALAR_BYTE,
            'elevation_scalar_byte': ELEVATION_SCALAR_BYTE,
            'coordinate_unit': 'ft',
            'elevation_unit': 'ft',
        },
        'linkage': {
            'mode': 'required',
            'job_id': 'linkage-job',
            'artifact_name': GEOMETRY_LINKAGE_NPZ_NAME,
        },
        'velocity': {
            'replacement_velocity_m_s': 2000.0,
            'refractor_velocity_m_s': 4500.0,
        },
        'moveout': {
            'model': 'head_wave_linear_offset',
            'offset_byte': OFFSET_BYTE,
            'allow_missing_offset': False,
        },
    }
    payload.update(overrides)
    return TimeTermStaticApplyRequest.model_validate(payload)


def _pick_source(**overrides: Any) -> LoadedPickSource:
    source = LoadedPickSource(
        picks_time_s_sorted=PICKS_SORTED,
        valid_mask_sorted=VALID_PICK_MASK,
        source_kind='batch_npz',
        n_traces=N_TRACES,
        n_samples=N_SAMPLES,
        dt=DT,
        n_valid=3,
        n_nan=1,
        metadata={'source': 'test'},
    )
    return source if not overrides else replace(source, **overrides)


def _write_npz_picks(path: Path, picks_time_s: np.ndarray = PICKS_ORIGINAL) -> Path:
    np.savez(
        path,
        picks_time_s=picks_time_s,
        n_traces=np.int64(N_TRACES),
        n_samples=np.int64(N_SAMPLES),
        dt=np.float64(DT),
    )
    return path


def _write_datum(path: Path, **overrides: Any) -> Path:
    payload: dict[str, object] = {
        'trace_shift_s_sorted': DATUM_SHIFT,
        'n_traces': np.int64(N_TRACES),
        'dt': np.float64(DT),
        'key1_byte': np.int64(KEY1_BYTE),
        'key2_byte': np.int64(KEY2_BYTE),
    }
    payload.update(overrides)
    np.savez(path, **payload)
    return path


def _write_residual(path: Path, **overrides: Any) -> Path:
    payload: dict[str, object] = {
        'applied_residual_shift_s_sorted': RESIDUAL_SHIFT,
        'n_traces': np.int64(N_TRACES),
        'dt': np.float64(DT),
        'key1_byte': np.int64(KEY1_BYTE),
        'key2_byte': np.int64(KEY2_BYTE),
    }
    payload.update(overrides)
    np.savez(path, **payload)
    return path


def _write_linkage(path: Path) -> Path:
    headers = GeometryLinkageHeaders(
        source_x=np.asarray([0.0, 100.0, 200.0, 300.0], dtype=np.float64),
        source_y=np.zeros(N_TRACES, dtype=np.float64),
        receiver_x=np.asarray([10.0, 20.0, 10.0, 20.0], dtype=np.float64),
        receiver_y=np.zeros(N_TRACES, dtype=np.float64),
        coordinate_scalar=np.ones(N_TRACES, dtype=np.int64),
        checked_bytes=(COORDINATE_SCALAR_BYTE, SOURCE_X_BYTE, SOURCE_Y_BYTE),
    )
    tables = build_endpoint_geometry_tables(headers)
    linkage = build_geometry_linkage(tables, GeometryLinkageOptions(mode='none'))
    arrays = build_geometry_linkage_solution_arrays(
        tables,
        linkage,
        metadata=GeometryLinkageArtifactMetadata(
            job_id='linkage-job',
            input_file_id=FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            source_x_byte=SOURCE_X_BYTE,
            source_y_byte=SOURCE_Y_BYTE,
            receiver_x_byte=RECEIVER_X_BYTE,
            receiver_y_byte=RECEIVER_Y_BYTE,
            coordinate_scalar_byte=COORDINATE_SCALAR_BYTE,
        ),
    )
    np.savez(path, **arrays)
    return path


def _state_with_artifacts(tmp_path: Path, reader: _Reader):
    state = create_app_state()
    state.file_registry.set_record(
        FILE_ID,
        {
            'path': str(tmp_path / 'line.sgy'),
            'store_path': str(tmp_path / 'store'),
            'dt': DT,
        },
    )
    with state.lock:
        state.cached_readers[f'{FILE_ID}_{KEY1_BYTE}_{KEY2_BYTE}'] = reader

    pick_dir = tmp_path / 'pick-job'
    pick_dir.mkdir()
    pick_path = _write_npz_picks(pick_dir / 'predicted_picks_time_s.npz')
    with state.lock:
        state.jobs.create_batch_apply_job(
            'pick-job',
            file_id=FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            artifacts_dir=str(pick_dir),
        )

    linkage_dir = tmp_path / 'linkage-job'
    linkage_dir.mkdir()
    linkage_path = _write_linkage(linkage_dir / GEOMETRY_LINKAGE_NPZ_NAME)
    with state.lock:
        state.jobs.create_static_job(
            'linkage-job',
            file_id=FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            statics_kind='geometry_linkage',
            artifacts_dir=str(linkage_dir),
        )
    return state, pick_path, linkage_path


def test_time_term_inputs_loads_predicted_picks_original_to_sorted_order(
    tmp_path: Path,
) -> None:
    state, _pick_path, linkage_path = _state_with_artifacts(tmp_path, _Reader())

    inputs = build_time_term_inversion_inputs(
        request=_request(),
        state=state,
        linkage_artifact_path=linkage_path,
    )

    np.testing.assert_allclose(
        inputs.pick_time_raw_s_sorted,
        PICKS_SORTED,
        equal_nan=True,
    )
    np.testing.assert_array_equal(inputs.valid_pick_mask_sorted, VALID_PICK_MASK)
    np.testing.assert_allclose(inputs.datum_trace_shift_s_sorted, np.zeros(N_TRACES))
    np.testing.assert_allclose(
        inputs.residual_applied_shift_s_sorted,
        np.zeros(N_TRACES),
    )


def test_time_term_inputs_builds_full_object_with_shifts_geometry_and_linkage(
    tmp_path: Path,
) -> None:
    datum_path = _write_datum(tmp_path / 'datum_static_solution.npz')
    residual_path = _write_residual(tmp_path / 'residual_static_solution.npz')
    linkage_path = _write_linkage(tmp_path / GEOMETRY_LINKAGE_NPZ_NAME)

    inputs = build_time_term_inversion_inputs_from_sources(
        request=_request(),
        reader=_Reader(),
        pick_source=_pick_source(),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
        datum_solution_path=datum_path,
        residual_solution_path=residual_path,
        linkage_artifact_path=linkage_path,
    )

    expected_after = PICKS_SORTED + DATUM_SHIFT + RESIDUAL_SHIFT
    expected_after[-1] = np.nan
    np.testing.assert_allclose(
        inputs.pick_time_after_static_s_sorted,
        expected_after,
        equal_nan=True,
    )
    np.testing.assert_allclose(inputs.datum_trace_shift_s_sorted, DATUM_SHIFT)
    np.testing.assert_allclose(
        inputs.residual_applied_shift_s_sorted,
        RESIDUAL_SHIFT,
    )
    assert inputs.n_nodes > 0
    np.testing.assert_array_equal(inputs.source_id_sorted, [100, 100, 200, 300])
    np.testing.assert_array_equal(inputs.receiver_id_sorted, [1, 2, 1, 2])
    np.testing.assert_allclose(
        inputs.offset_sorted,
        np.asarray([-100.0, -50.0, 50.0, 100.0]) * 0.3048,
    )

    np.testing.assert_allclose(
        inputs.source_x_m_sorted,
        np.asarray([100.0, 200.0, 20.0, 5.0]) * 0.3048,
    )
    np.testing.assert_allclose(
        inputs.receiver_x_m_sorted,
        np.asarray([300.0, 400.0, 40.0, 15.0]) * 0.3048,
    )
    np.testing.assert_allclose(
        inputs.source_depth_m_sorted,
        np.asarray([10.0, 10.0, 10.0, 0.0]) * 0.3048,
    )
    np.testing.assert_allclose(
        inputs.source_elevation_m_sorted,
        np.asarray([1000.0, 200.0, 600.0, 400.0]) * 0.3048,
    )
    np.testing.assert_allclose(
        inputs.receiver_elevation_m_sorted,
        np.asarray([900.0, 100.0, 300.0, 200.0]) * 0.3048,
    )
    assert inputs.sign_convention.startswith('pick_time_after_static_s')


def test_time_term_inputs_loads_manual_pick_export_original_to_sorted_order(
    tmp_path: Path,
) -> None:
    state, _pick_path, linkage_path = _state_with_artifacts(tmp_path, _Reader())
    manual_dir = tmp_path / 'manual-job'
    manual_dir.mkdir()
    manual_path = _write_npz_picks(manual_dir / 'manual_picks.npz')
    with state.lock:
        state.jobs.create_static_job(
            'manual-job',
            file_id=FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            statics_kind='first_break_qc',
            artifacts_dir=str(manual_dir),
        )

    req = _request(
        pick_source={
            'kind': 'manual_npz_artifact',
            'job_id': 'manual-job',
            'artifact_name': manual_path.name,
        },
    )

    inputs = build_time_term_inversion_inputs(
        request=req,
        state=state,
        linkage_artifact_path=linkage_path,
    )

    np.testing.assert_allclose(
        inputs.pick_time_raw_s_sorted,
        PICKS_SORTED,
        equal_nan=True,
    )
    assert inputs.pick_source_description.startswith('manual_npz:')


def test_time_term_inputs_rejects_infinite_picks(tmp_path: Path) -> None:
    state, pick_path, linkage_path = _state_with_artifacts(tmp_path, _Reader())
    bad = PICKS_ORIGINAL.copy()
    bad[0] = np.inf
    _write_npz_picks(pick_path, picks_time_s=bad)

    with pytest.raises(ValueError, match='contains inf'):
        build_time_term_inversion_inputs(
            request=_request(),
            state=state,
            linkage_artifact_path=linkage_path,
        )


def test_time_term_inputs_allows_nan_picks_as_invalid(tmp_path: Path) -> None:
    linkage_path = _write_linkage(tmp_path / GEOMETRY_LINKAGE_NPZ_NAME)

    inputs = build_time_term_inversion_inputs_from_sources(
        request=_request(),
        reader=_Reader(),
        pick_source=_pick_source(),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
        linkage_artifact_path=linkage_path,
    )

    assert np.isnan(inputs.pick_time_raw_s_sorted[-1])
    assert np.isnan(inputs.pick_time_after_static_s_sorted[-1])
    assert inputs.valid_pick_mask_sorted.tolist() == [True, True, True, False]


def test_time_term_inputs_loads_datum_trace_shift_sorted(tmp_path: Path) -> None:
    datum_path = _write_datum(tmp_path / 'datum_static_solution.npz')
    linkage_path = _write_linkage(tmp_path / GEOMETRY_LINKAGE_NPZ_NAME)

    inputs = build_time_term_inversion_inputs_from_sources(
        request=_request(),
        reader=_Reader(),
        pick_source=_pick_source(),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
        datum_solution_path=datum_path,
        linkage_artifact_path=linkage_path,
    )

    np.testing.assert_allclose(inputs.datum_trace_shift_s_sorted, DATUM_SHIFT)


def test_time_term_inputs_rejects_datum_shift_shape_mismatch(tmp_path: Path) -> None:
    datum_path = _write_datum(
        tmp_path / 'datum_static_solution.npz',
        trace_shift_s_sorted=DATUM_SHIFT[:-1],
    )
    linkage_path = _write_linkage(tmp_path / GEOMETRY_LINKAGE_NPZ_NAME)

    with pytest.raises(ValueError, match='shape mismatch'):
        build_time_term_inversion_inputs_from_sources(
            request=_request(),
            reader=_Reader(),
            pick_source=_pick_source(),
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
            datum_solution_path=datum_path,
            linkage_artifact_path=linkage_path,
        )


@pytest.mark.parametrize(
    'field_name',
    [
        'applied_residual_shift_s_sorted',
        'residual_applied_shift_s_sorted',
        'trace_shift_s_sorted',
    ],
)
def test_time_term_inputs_loads_applied_residual_shift_aliases(
    tmp_path: Path,
    field_name: str,
) -> None:
    path = tmp_path / 'residual_static_solution.npz'
    np.savez(
        path,
        **{
            field_name: RESIDUAL_SHIFT,
            'n_traces': np.int64(N_TRACES),
            'dt': np.float64(DT),
            'key1_byte': np.int64(KEY1_BYTE),
            'key2_byte': np.int64(KEY2_BYTE),
        },
    )

    loaded = load_time_term_residual_applied_shift(
        path,
        expected_n_traces=N_TRACES,
        expected_dt=DT,
        expected_key1_byte=KEY1_BYTE,
        expected_key2_byte=KEY2_BYTE,
    )

    np.testing.assert_allclose(loaded.values_s_sorted, RESIDUAL_SHIFT)
    assert loaded.field_name == field_name


def test_time_term_inputs_rejects_estimated_delay_only_residual_solution(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'residual_static_solution.npz'
    np.savez(path, estimated_trace_delay_s_sorted=RESIDUAL_SHIFT)

    with pytest.raises(ValueError, match='explicit applied-shift conversion'):
        load_time_term_residual_applied_shift(
            path,
            expected_n_traces=N_TRACES,
            expected_dt=DT,
            expected_key1_byte=KEY1_BYTE,
            expected_key2_byte=KEY2_BYTE,
        )


def test_time_term_inputs_rejects_linkage_n_traces_mismatch(tmp_path: Path) -> None:
    linkage_path = _write_linkage(tmp_path / GEOMETRY_LINKAGE_NPZ_NAME)
    with np.load(linkage_path, allow_pickle=False) as npz:
        payload = {key: np.asarray(npz[key]) for key in npz.files}
    payload['n_traces'] = np.asarray(N_TRACES + 1, dtype=np.int64)
    np.savez(linkage_path, **payload)

    with pytest.raises(ValueError, match='n_traces mismatch'):
        build_time_term_inversion_inputs_from_sources(
            request=_request(),
            reader=_Reader(),
            pick_source=_pick_source(),
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
            linkage_artifact_path=linkage_path,
        )


def test_time_term_inputs_uses_zero_source_depth_when_missing(tmp_path: Path) -> None:
    linkage_path = _write_linkage(tmp_path / GEOMETRY_LINKAGE_NPZ_NAME)
    req = _request(
        geometry={
            **_request().geometry.model_dump(),
            'source_depth_byte': None,
        }
    )

    inputs = build_time_term_inversion_inputs_from_sources(
        request=req,
        reader=_Reader(),
        pick_source=_pick_source(),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
        linkage_artifact_path=linkage_path,
    )

    np.testing.assert_allclose(inputs.source_depth_m_sorted, np.zeros(N_TRACES))


def test_time_term_inputs_rejects_non_finite_geometry(tmp_path: Path) -> None:
    linkage_path = _write_linkage(tmp_path / GEOMETRY_LINKAGE_NPZ_NAME)
    headers = {SOURCE_X_BYTE: _base_headers()[SOURCE_X_BYTE].copy()}
    headers[SOURCE_X_BYTE][0] = np.inf

    with pytest.raises(ValueError, match='scaled values'):
        build_time_term_inversion_inputs_from_sources(
            request=_request(),
            reader=_Reader(headers=headers),
            pick_source=_pick_source(),
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
            linkage_artifact_path=linkage_path,
        )


def test_time_term_inputs_requires_offset_when_moveout_requires_offset(
    tmp_path: Path,
    ) -> None:
    linkage_path = _write_linkage(tmp_path / GEOMETRY_LINKAGE_NPZ_NAME)

    with pytest.raises(ValueError, match='failed to read offset header'):
        build_time_term_inversion_inputs_from_sources(
            request=_request(),
            reader=_Reader(missing_headers={OFFSET_BYTE}),
            pick_source=_pick_source(),
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
            linkage_artifact_path=linkage_path,
        )


def test_time_term_inputs_allows_missing_offset_when_configured(tmp_path: Path) -> None:
    linkage_path = _write_linkage(tmp_path / GEOMETRY_LINKAGE_NPZ_NAME)
    req = _request(
        moveout={
            'model': 'head_wave_linear_offset',
            'offset_byte': OFFSET_BYTE,
            'allow_missing_offset': True,
        },
    )

    inputs = build_time_term_inversion_inputs_from_sources(
        request=req,
        reader=_Reader(missing_headers={OFFSET_BYTE}),
        pick_source=_pick_source(),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
        linkage_artifact_path=linkage_path,
    )

    assert inputs.offset_sorted is None


def test_time_term_inputs_rejects_no_valid_picks(tmp_path: Path) -> None:
    linkage_path = _write_linkage(tmp_path / GEOMETRY_LINKAGE_NPZ_NAME)
    no_valid = _pick_source(
        picks_time_s_sorted=np.full(N_TRACES, np.nan, dtype=np.float64),
        valid_mask_sorted=np.zeros(N_TRACES, dtype=bool),
        n_valid=0,
        n_nan=N_TRACES,
    )

    with pytest.raises(ValueError, match='at least one valid pick'):
        build_time_term_inversion_inputs_from_sources(
            request=_request(),
            reader=_Reader(),
            pick_source=no_valid,
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
            linkage_artifact_path=linkage_path,
        )


def test_summarize_time_term_inversion_inputs_is_json_safe(tmp_path: Path) -> None:
    linkage_path = _write_linkage(tmp_path / GEOMETRY_LINKAGE_NPZ_NAME)
    inputs = build_time_term_inversion_inputs_from_sources(
        request=_request(),
        reader=_Reader(),
        pick_source=_pick_source(),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
        linkage_artifact_path=linkage_path,
    )

    payload = summarize_time_term_inversion_inputs(inputs)

    assert payload['n_traces'] == N_TRACES
    assert payload['n_valid_picks'] == 3
    assert payload['has_offset'] is True
    json.dumps(payload, allow_nan=False)
