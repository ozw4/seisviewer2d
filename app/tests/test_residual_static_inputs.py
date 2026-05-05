from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.api.schemas import ResidualStaticApplyRequest
from app.core.state import create_app_state
from app.services.pick_source_loader import LoadedPickSource
from app.services.residual_static_inputs import (
    ResidualStaticResolvedArtifacts,
    build_residual_static_solver_inputs,
    load_residual_static_pick_source,
    load_source_receiver_id_headers_sorted,
    resolve_residual_static_input_artifacts,
)

KEY1_BYTE = 189
KEY2_BYTE = 193
SOURCE_ID_BYTE = 17
RECEIVER_ID_BYTE = 13
OFFSET_BYTE = 37
SOURCE_ELEVATION_BYTE = 45
RECEIVER_ELEVATION_BYTE = 41
DT = 0.004
N_SAMPLES = 64
SOURCE_FILE_ID = 'source-file-id'
CORRECTED_FILE_ID = 'corrected-file-id'
DATUM_JOB_ID = 'datum-job'
KEY1_SORTED = np.asarray([100, 100, 100, 200, 200, 200], dtype=np.int64)
KEY2_SORTED = np.asarray([1, 2, 3, 1, 2, 3], dtype=np.int64)
SOURCE_ID_SORTED = np.asarray([10, 10, 20, 20, 30, 30], dtype=np.int64)
RECEIVER_ID_SORTED = np.asarray([1, 2, 1, 2, 1, 2], dtype=np.int64)
SOURCE_SHIFT = np.asarray([-0.004, -0.004, -0.004, -0.008, -0.008, -0.008])
RECEIVER_SHIFT = np.asarray([-0.006, -0.006, -0.006, -0.012, -0.012, -0.012])
TRACE_SHIFT = SOURCE_SHIFT + RECEIVER_SHIFT
SOURCE_ELEVATION = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
RECEIVER_ELEVATION = np.asarray([90.0, 91.0, 92.0, 93.0, 94.0, 95.0])
OFFSET_SORTED = np.asarray([-1200.0, -800.0, -400.0, 400.0, 800.0, 1200.0])
PICKS_SORTED = np.asarray([0.100, 0.110, np.nan, 0.130, 0.140, 0.150])
VALID_PICK_MASK = np.asarray([True, True, False, True, True, True])


class _Reader:
    def __init__(
        self,
        *,
        key1_sorted: np.ndarray = KEY1_SORTED,
        key2_sorted: np.ndarray = KEY2_SORTED,
        source_id_sorted: np.ndarray = SOURCE_ID_SORTED,
        receiver_id_sorted: np.ndarray = RECEIVER_ID_SORTED,
        offset_sorted: np.ndarray = OFFSET_SORTED,
        n_samples: int = N_SAMPLES,
        dt: float = DT,
        fail_offset_read: bool = False,
    ) -> None:
        self.key1_byte = KEY1_BYTE
        self.key2_byte = KEY2_BYTE
        self.traces = np.zeros((int(np.asarray(key1_sorted).shape[0]), n_samples))
        self.meta = {'dt': dt}
        self.fail_offset_read = fail_offset_read
        self.headers = {
            KEY1_BYTE: np.asarray(key1_sorted),
            KEY2_BYTE: np.asarray(key2_sorted),
            SOURCE_ID_BYTE: np.asarray(source_id_sorted),
            RECEIVER_ID_BYTE: np.asarray(receiver_id_sorted),
            OFFSET_BYTE: np.asarray(offset_sorted),
        }

    def ensure_header(self, byte: int) -> np.ndarray:
        if self.fail_offset_read and int(byte) == OFFSET_BYTE:
            raise AssertionError('offset header should not be read')
        return self.headers[int(byte)]

    def get_n_samples(self) -> int:
        return int(self.traces.shape[-1])


def _request(**overrides: Any) -> ResidualStaticApplyRequest:
    payload: dict[str, Any] = {
        'file_id': CORRECTED_FILE_ID,
        'key1_byte': KEY1_BYTE,
        'key2_byte': KEY2_BYTE,
        'datum_solution': {
            'job_id': DATUM_JOB_ID,
            'name': 'datum_static_solution.npz',
        },
        'pick_source': {
            'kind': 'batch_job_artifact',
            'job_id': 'pick-job',
            'name': 'predicted_picks_time_s.npz',
        },
        'source_id_byte': SOURCE_ID_BYTE,
        'receiver_id_byte': RECEIVER_ID_BYTE,
        'offset_byte': OFFSET_BYTE,
        'moveout': {'model': 'linear_abs_offset'},
    }
    payload.update(overrides)
    return ResidualStaticApplyRequest.model_validate(payload)


def _state_with_datum_job(tmp_path: Path):
    state = create_app_state()
    job_dir = tmp_path / 'datum-job'
    job_dir.mkdir()
    (job_dir / 'datum_static_solution.npz').write_bytes(b'data')
    corrected_store = tmp_path / 'corrected-store'
    with state.lock:
        state.jobs.create_static_job(
            DATUM_JOB_ID,
            file_id=SOURCE_FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            statics_kind='datum',
            artifacts_dir=str(job_dir),
        )
        state.jobs.set_static_corrected_file(
            DATUM_JOB_ID,
            corrected_file_id=CORRECTED_FILE_ID,
            corrected_store_path=str(corrected_store),
        )
    return state, job_dir, corrected_store


def _write_corrected_manifest(
    job_dir: Path,
    *,
    corrected_store: Path,
    overrides: dict[str, Any] | None = None,
) -> None:
    payload = {
        'file_id': CORRECTED_FILE_ID,
        'store_path': str(corrected_store),
        'derived_from_file_id': SOURCE_FILE_ID,
        'job_id': DATUM_JOB_ID,
        'key1_byte': KEY1_BYTE,
        'key2_byte': KEY2_BYTE,
    }
    if overrides:
        payload.update(overrides)
    (job_dir / 'corrected_file.json').write_text(
        json.dumps(payload),
        encoding='utf-8',
    )


def _add_batch_pick_job(
    state,
    tmp_path: Path,
    *,
    job_id: str = 'pick-job',
    file_id: str = SOURCE_FILE_ID,
    key1_byte: int = KEY1_BYTE,
    key2_byte: int = KEY2_BYTE,
) -> Path:
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    artifact = job_dir / 'predicted_picks_time_s.npz'
    artifact.write_bytes(b'data')
    with state.lock:
        state.jobs.create_batch_apply_job(
            job_id,
            file_id=file_id,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            artifacts_dir=str(job_dir),
        )
    return artifact


def _add_manual_npz_job(
    state,
    tmp_path: Path,
    *,
    job_id: str = 'manual-job',
    file_id: str = SOURCE_FILE_ID,
) -> Path:
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    artifact = job_dir / 'manual_picks.npz'
    artifact.write_bytes(b'data')
    with state.lock:
        state.jobs.create_static_job(
            job_id,
            file_id=file_id,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            statics_kind='first_break_qc',
            artifacts_dir=str(job_dir),
        )
    return artifact


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
        'n_traces': np.int64(KEY1_SORTED.shape[0]),
        'key1_byte': np.int64(KEY1_BYTE),
        'key2_byte': np.int64(KEY2_BYTE),
        'source_elevation_byte': np.int64(SOURCE_ELEVATION_BYTE),
        'receiver_elevation_byte': np.int64(RECEIVER_ELEVATION_BYTE),
    }


def _write_solution(path: Path, **overrides: Any) -> Path:
    payload = _base_solution_payload()
    payload.update(overrides)
    np.savez(path, **payload)
    return path


def _artifacts(solution_path: Path) -> ResidualStaticResolvedArtifacts:
    return ResidualStaticResolvedArtifacts(
        datum_solution_path=solution_path,
        pick_artifact_path=None,
        datum_job_id=DATUM_JOB_ID,
        datum_source_file_id=SOURCE_FILE_ID,
        datum_corrected_file_id=CORRECTED_FILE_ID,
        pick_source_artifact_name=None,
    )


def _pick_source(**overrides: Any) -> LoadedPickSource:
    source = LoadedPickSource(
        picks_time_s_sorted=PICKS_SORTED,
        valid_mask_sorted=VALID_PICK_MASK,
        source_kind='batch_npz',
        n_traces=int(KEY1_SORTED.shape[0]),
        n_samples=N_SAMPLES,
        dt=DT,
        n_valid=5,
        n_nan=1,
        metadata={'source': 'test'},
    )
    return source if not overrides else replace(source, **overrides)


def test_resolve_residual_static_artifacts_accepts_datum_job_corrected_file_id(
    tmp_path: Path,
) -> None:
    state, job_dir, corrected_store = _state_with_datum_job(tmp_path)
    _write_corrected_manifest(job_dir, corrected_store=corrected_store)
    pick_artifact = _add_batch_pick_job(state, tmp_path)

    artifacts = resolve_residual_static_input_artifacts(state, _request())

    assert artifacts.datum_solution_path == job_dir / 'datum_static_solution.npz'
    assert artifacts.pick_artifact_path == pick_artifact
    assert artifacts.datum_source_file_id == SOURCE_FILE_ID
    assert artifacts.datum_corrected_file_id == CORRECTED_FILE_ID
    assert artifacts.pick_source_artifact_name == 'predicted_picks_time_s.npz'


def test_resolve_residual_static_artifacts_rejects_missing_datum_job() -> None:
    state = create_app_state()

    with pytest.raises(ValueError, match='job_id not found'):
        resolve_residual_static_input_artifacts(state, _request())


def test_resolve_residual_static_artifacts_rejects_non_datum_job(
    tmp_path: Path,
) -> None:
    state = create_app_state()
    job_dir = tmp_path / 'qc-job'
    job_dir.mkdir()
    (job_dir / 'datum_static_solution.npz').write_bytes(b'data')
    with state.lock:
        state.jobs.create_static_job(
            DATUM_JOB_ID,
            file_id=SOURCE_FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            statics_kind='first_break_qc',
            artifacts_dir=str(job_dir),
        )

    with pytest.raises(ValueError, match='unsupported statics_kind'):
        resolve_residual_static_input_artifacts(state, _request())


def test_resolve_residual_static_artifacts_rejects_missing_corrected_file_id(
    tmp_path: Path,
) -> None:
    state = create_app_state()
    job_dir = tmp_path / 'datum-job'
    job_dir.mkdir()
    (job_dir / 'datum_static_solution.npz').write_bytes(b'data')
    with state.lock:
        state.jobs.create_static_job(
            DATUM_JOB_ID,
            file_id=SOURCE_FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            statics_kind='datum',
            artifacts_dir=str(job_dir),
        )

    with pytest.raises(ValueError, match='corrected_file_id'):
        resolve_residual_static_input_artifacts(state, _request())


def test_resolve_residual_static_artifacts_rejects_corrected_file_id_mismatch(
    tmp_path: Path,
) -> None:
    state, _job_dir, _corrected_store = _state_with_datum_job(tmp_path)
    _add_batch_pick_job(state, tmp_path)

    with pytest.raises(ValueError, match='corrected_file_id mismatch'):
        resolve_residual_static_input_artifacts(
            state,
            _request(file_id='other-corrected-file-id'),
        )


def test_resolve_residual_static_artifacts_validates_corrected_file_manifest_when_present(
    tmp_path: Path,
) -> None:
    state, job_dir, corrected_store = _state_with_datum_job(tmp_path)
    _add_batch_pick_job(state, tmp_path)
    _write_corrected_manifest(
        job_dir,
        corrected_store=corrected_store,
        overrides={'derived_from_file_id': 'wrong-source'},
    )

    with pytest.raises(ValueError, match='derived_from_file_id'):
        resolve_residual_static_input_artifacts(state, _request())


def test_resolve_residual_static_artifacts_accepts_batch_pick_job_on_datum_source_file_id(
    tmp_path: Path,
) -> None:
    state, _job_dir, _corrected_store = _state_with_datum_job(tmp_path)
    pick_artifact = _add_batch_pick_job(state, tmp_path)

    artifacts = resolve_residual_static_input_artifacts(state, _request())

    assert artifacts.pick_artifact_path == pick_artifact


def test_resolve_residual_static_artifacts_rejects_batch_pick_job_on_wrong_file_id(
    tmp_path: Path,
) -> None:
    state, _job_dir, _corrected_store = _state_with_datum_job(tmp_path)
    _add_batch_pick_job(state, tmp_path, file_id='wrong-source')

    with pytest.raises(ValueError, match='file_id expected'):
        resolve_residual_static_input_artifacts(state, _request())


def test_resolve_residual_static_artifacts_accepts_manual_npz_artifact(
    tmp_path: Path,
) -> None:
    state, _job_dir, _corrected_store = _state_with_datum_job(tmp_path)
    manual_artifact = _add_manual_npz_job(state, tmp_path)
    req = _request(
        pick_source={
            'kind': 'manual_npz_artifact',
            'job_id': 'manual-job',
            'name': 'manual_picks.npz',
        },
    )

    artifacts = resolve_residual_static_input_artifacts(state, req)

    assert artifacts.pick_artifact_path == manual_artifact
    assert artifacts.pick_source_artifact_name == 'manual_picks.npz'


def test_resolve_residual_static_artifacts_rejects_manual_npz_without_npz_suffix(
    tmp_path: Path,
) -> None:
    state, _job_dir, _corrected_store = _state_with_datum_job(tmp_path)
    job_dir = tmp_path / 'manual-job'
    job_dir.mkdir()
    (job_dir / 'manual_picks.txt').write_bytes(b'data')
    with state.lock:
        state.jobs.create_static_job(
            'manual-job',
            file_id=SOURCE_FILE_ID,
            key1_byte=KEY1_BYTE,
            key2_byte=KEY2_BYTE,
            statics_kind='first_break_qc',
            artifacts_dir=str(job_dir),
        )
    req = _request(
        pick_source={
            'kind': 'manual_npz_artifact',
            'job_id': 'manual-job',
            'name': 'manual_picks.txt',
        },
    )

    with pytest.raises(ValueError, match='must end with .npz'):
        resolve_residual_static_input_artifacts(state, req)


def test_resolve_residual_static_artifacts_returns_none_for_manual_memmap_artifact_path(
    tmp_path: Path,
) -> None:
    state, _job_dir, _corrected_store = _state_with_datum_job(tmp_path)
    req = _request(pick_source={'kind': 'manual_memmap'})

    artifacts = resolve_residual_static_input_artifacts(state, req)

    assert artifacts.pick_artifact_path is None
    assert artifacts.pick_source_artifact_name is None


def test_load_source_receiver_id_headers_sorted_success() -> None:
    source_id, receiver_id = load_source_receiver_id_headers_sorted(
        _Reader(),
        source_id_byte=SOURCE_ID_BYTE,
        receiver_id_byte=RECEIVER_ID_BYTE,
        expected_n_traces=6,
    )

    np.testing.assert_array_equal(source_id, SOURCE_ID_SORTED)
    np.testing.assert_array_equal(receiver_id, RECEIVER_ID_SORTED)


def test_load_source_receiver_id_headers_sorted_accepts_zero_and_negative_ids() -> None:
    source_ids = np.asarray([0, -1, -1, 2, 2, 0], dtype=np.int64)
    receiver_ids = np.asarray([-10, -10, 0, 0, 5, 5], dtype=np.float64)
    reader = _Reader(source_id_sorted=source_ids, receiver_id_sorted=receiver_ids)

    source_id, receiver_id = load_source_receiver_id_headers_sorted(
        reader,
        source_id_byte=SOURCE_ID_BYTE,
        receiver_id_byte=RECEIVER_ID_BYTE,
        expected_n_traces=6,
    )

    np.testing.assert_array_equal(source_id, source_ids)
    np.testing.assert_array_equal(receiver_id, receiver_ids.astype(np.int64))


def test_load_source_receiver_id_headers_sorted_rejects_shape_mismatch() -> None:
    reader = _Reader(source_id_sorted=SOURCE_ID_SORTED[:5])

    with pytest.raises(ValueError, match='shape mismatch'):
        load_source_receiver_id_headers_sorted(
            reader,
            source_id_byte=SOURCE_ID_BYTE,
            receiver_id_byte=RECEIVER_ID_BYTE,
            expected_n_traces=6,
        )


def test_load_source_receiver_id_headers_sorted_rejects_non_integer_values() -> None:
    source_ids = SOURCE_ID_SORTED.astype(np.float64)
    source_ids[0] = 10.5
    reader = _Reader(source_id_sorted=source_ids)

    with pytest.raises(ValueError, match='integer values'):
        load_source_receiver_id_headers_sorted(
            reader,
            source_id_byte=SOURCE_ID_BYTE,
            receiver_id_byte=RECEIVER_ID_BYTE,
            expected_n_traces=6,
        )


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
def test_load_source_receiver_id_headers_sorted_rejects_nan_inf(
    bad_value: float,
) -> None:
    source_ids = SOURCE_ID_SORTED.astype(np.float64)
    source_ids[0] = bad_value
    reader = _Reader(source_id_sorted=source_ids)

    with pytest.raises(ValueError, match='finite values'):
        load_source_receiver_id_headers_sorted(
            reader,
            source_id_byte=SOURCE_ID_BYTE,
            receiver_id_byte=RECEIVER_ID_BYTE,
            expected_n_traces=6,
        )


def test_build_residual_static_solver_inputs_success_linear_abs_offset(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    inputs = build_residual_static_solver_inputs(
        req=_request(),
        artifacts=_artifacts(solution_path),
        pick_source=_pick_source(),
        reader=_Reader(),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
    )

    np.testing.assert_allclose(inputs.picks_time_s_sorted, PICKS_SORTED, equal_nan=True)
    np.testing.assert_array_equal(inputs.valid_pick_mask_sorted, VALID_PICK_MASK)
    np.testing.assert_allclose(inputs.datum_trace_shift_s_sorted, TRACE_SHIFT)
    np.testing.assert_allclose(inputs.offset_sorted, OFFSET_SORTED)
    np.testing.assert_allclose(inputs.abs_offset_sorted, np.abs(OFFSET_SORTED))
    assert inputs.moveout_model == 'linear_abs_offset'
    assert inputs.input_file_id == CORRECTED_FILE_ID
    assert inputs.datum_source_file_id == SOURCE_FILE_ID
    assert inputs.metadata['input_file_role'] == 'datum_corrected_trace_store'


def test_build_residual_static_solver_inputs_success_moveout_none(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    inputs = build_residual_static_solver_inputs(
        req=_request(moveout={'model': 'none'}, offset_byte=None),
        artifacts=_artifacts(solution_path),
        pick_source=_pick_source(),
        reader=_Reader(fail_offset_read=True),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
    )

    assert inputs.offset_sorted is None
    assert inputs.abs_offset_sorted is None
    assert inputs.offset_byte is None
    assert inputs.moveout_model == 'none'


def test_build_residual_static_solver_inputs_computes_pick_time_after_datum(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    inputs = build_residual_static_solver_inputs(
        req=_request(),
        artifacts=_artifacts(solution_path),
        pick_source=_pick_source(),
        reader=_Reader(),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
    )

    expected = np.asarray([0.090, 0.100, np.nan, 0.110, 0.120, 0.130])
    np.testing.assert_allclose(
        inputs.pick_time_after_datum_s_sorted,
        expected,
        equal_nan=True,
    )


def test_build_residual_static_solver_inputs_keeps_nan_for_invalid_picks(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    inputs = build_residual_static_solver_inputs(
        req=_request(),
        artifacts=_artifacts(solution_path),
        pick_source=_pick_source(),
        reader=_Reader(),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
    )

    assert np.isnan(inputs.pick_time_after_datum_s_sorted[2])


def test_build_residual_static_solver_inputs_builds_unique_ids_and_inverse_indices(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    inputs = build_residual_static_solver_inputs(
        req=_request(),
        artifacts=_artifacts(solution_path),
        pick_source=_pick_source(),
        reader=_Reader(),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
    )

    np.testing.assert_array_equal(inputs.source_unique_ids, [10, 20, 30])
    np.testing.assert_array_equal(inputs.source_index_sorted, [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(inputs.receiver_unique_ids, [1, 2])
    np.testing.assert_array_equal(inputs.receiver_index_sorted, [0, 1, 0, 1, 0, 1])


def test_build_residual_static_solver_inputs_computes_valid_pick_counts(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    inputs = build_residual_static_solver_inputs(
        req=_request(),
        artifacts=_artifacts(solution_path),
        pick_source=_pick_source(),
        reader=_Reader(),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
    )

    np.testing.assert_array_equal(inputs.source_valid_pick_counts, [2, 1, 2])
    np.testing.assert_array_equal(inputs.receiver_valid_pick_counts, [2, 3])
    assert inputs.metadata['n_valid_picks'] == 5
    assert inputs.metadata['n_sources'] == 3
    assert inputs.metadata['n_receivers'] == 2


def test_build_residual_static_solver_inputs_rejects_key1_key2_mismatch(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')
    key1 = KEY1_SORTED.copy()
    key1[0] = 999

    with pytest.raises(ValueError, match='key1_sorted does not match'):
        build_residual_static_solver_inputs(
            req=_request(),
            artifacts=_artifacts(solution_path),
            pick_source=_pick_source(),
            reader=_Reader(key1_sorted=key1),
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
        )


def test_build_residual_static_solver_inputs_rejects_pick_source_dt_mismatch(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    with pytest.raises(ValueError, match='pick source dt mismatch'):
        build_residual_static_solver_inputs(
            req=_request(),
            artifacts=_artifacts(solution_path),
            pick_source=_pick_source(dt=0.002),
            reader=_Reader(),
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
        )


def test_build_residual_static_solver_inputs_rejects_pick_source_n_samples_mismatch(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    with pytest.raises(ValueError, match='pick source n_samples mismatch'):
        build_residual_static_solver_inputs(
            req=_request(),
            artifacts=_artifacts(solution_path),
            pick_source=_pick_source(n_samples=N_SAMPLES + 1),
            reader=_Reader(),
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
        )


def test_build_residual_static_solver_inputs_rejects_nonfinite_offset_for_linear_abs_offset(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')
    offset = OFFSET_SORTED.copy()
    offset[0] = np.inf

    with pytest.raises(ValueError, match='offset header'):
        build_residual_static_solver_inputs(
            req=_request(),
            artifacts=_artifacts(solution_path),
            pick_source=_pick_source(),
            reader=_Reader(offset_sorted=offset),
            expected_dt=DT,
            expected_n_samples=N_SAMPLES,
        )


def test_build_residual_static_solver_inputs_does_not_read_offset_for_moveout_none(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(tmp_path / 'datum_static_solution.npz')

    inputs = build_residual_static_solver_inputs(
        req=_request(moveout={'model': 'none'}, offset_byte=None),
        artifacts=_artifacts(solution_path),
        pick_source=_pick_source(),
        reader=_Reader(fail_offset_read=True),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
    )

    assert inputs.offset_sorted is None


def test_load_residual_static_pick_source_manual_memmap_uses_datum_source_file_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = create_app_state()
    req = _request(pick_source={'kind': 'manual_memmap'})
    artifacts = _artifacts(tmp_path / 'datum_static_solution.npz')
    calls: list[dict[str, Any]] = []
    loaded = _pick_source(source_kind='manual_memmap')

    def _fake_loader(**kwargs: Any) -> LoadedPickSource:
        calls.append(kwargs)
        return loaded

    monkeypatch.setattr(
        'app.services.residual_static_inputs.load_manual_memmap_pick_source',
        _fake_loader,
    )

    result = load_residual_static_pick_source(
        req=req,
        artifacts=artifacts,
        reader=_Reader(),
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
        state=state,
    )

    assert result is loaded
    assert calls == [
        {
            'file_id': SOURCE_FILE_ID,
            'key1_byte': KEY1_BYTE,
            'key2_byte': KEY2_BYTE,
            'state': state,
        }
    ]
