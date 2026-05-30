from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import app.services.residual_static_corrected_store as svc
from app.core.state import AppState, create_app_state
from app.services.residual_static_corrected_store import (
    ResidualStaticTraceStoreApplyOptions,
    apply_residual_static_correction_to_trace_store,
    load_residual_static_solution_for_apply,
    validate_residual_static_shift_for_apply,
)
from app.services.trace_store_registration import trace_store_cache_key

KEY1 = 189
KEY2 = 193
SOURCE_FILE_ID = 'datum-corrected-file-id'
CORRECTED_FILE_ID = 'residual-corrected-file-id'
JOB_ID = 'bbbb2222-residual-job'
DT = 0.004
SIGN_CONVENTION = (
    'estimated_trace_delay_s=source_delay_s+receiver_delay_s; '
    'applied_residual_shift_s=-estimated_trace_delay_s; '
    'corrected(t)=raw(t-shift_s)'
)


def _datum_component() -> dict[str, object]:
    return {
        'name': 'datum_static_correction',
        'job_id': 'aaaa1111-datum-job',
        'solution_artifact': 'datum_static_solution.npz',
        'shift_field': 'trace_shift_s_sorted',
        'value_kind': 'applied_event_time_shift_s',
    }


def _write_datum_store(
    store: Path,
    *,
    traces: np.ndarray | None = None,
    dt: float = DT,
    derived: dict[str, object] | None = None,
    original_segy_path: str = '/data/line001.sgy',
    source_sha256: str | None = 'datum-source-sha',
) -> np.ndarray:
    store.mkdir(parents=True, exist_ok=True)
    if traces is None:
        traces = np.zeros((3, 120), dtype=np.float32)
        traces[:, 100] = 1.0
    traces = np.asarray(traces, dtype=np.float32)
    n_traces, n_samples = traces.shape
    np.save(store / 'traces.npy', traces)
    np.savez(
        store / 'index.npz',
        key1_values=np.asarray([100], dtype=np.int64),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([n_traces], dtype=np.int64),
        sorted_to_original=np.arange(n_traces, dtype=np.int64),
    )
    np.save(store / f'headers_byte_{KEY1}.npy', np.full(n_traces, 100, dtype=np.int32))
    np.save(store / f'headers_byte_{KEY2}.npy', np.arange(n_traces, dtype=np.int32))

    if derived is None:
        derived = {
            'kind': 'time_shifted_trace_store',
            'from_file_id': 'raw-file-id',
            'components': [_datum_component()],
        }
    meta = {
        'schema_version': 1,
        'dtype': 'float32',
        'n_traces': int(n_traces),
        'n_samples': int(n_samples),
        'key_bytes': {'key1': KEY1, 'key2': KEY2},
        'sorted_by': ['key1', 'key2'],
        'dt': float(dt),
        'original_segy_path': original_segy_path,
        'source_sha256': source_sha256,
        'derived': derived,
    }
    (store / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    return traces


def _write_solution(
    path: Path,
    *,
    n_traces: int,
    applied: np.ndarray | None = None,
    estimated: np.ndarray | None = None,
    dt: float = DT,
    key1_byte: int = KEY1,
    key2_byte: int = KEY2,
    include_shift: bool = True,
) -> Path:
    if estimated is None:
        estimated = np.zeros(n_traces, dtype=np.float64)
    if applied is None:
        applied = -np.asarray(estimated, dtype=np.float64)
    payload: dict[str, Any] = {
        'estimated_trace_delay_s_sorted': np.asarray(estimated, dtype=np.float64),
        'dt': np.asarray(dt, dtype=np.float64),
        'n_traces': np.asarray(n_traces, dtype=np.int64),
        'key1_byte': np.asarray(key1_byte, dtype=np.int64),
        'key2_byte': np.asarray(key2_byte, dtype=np.int64),
        'sign_convention': np.asarray(SIGN_CONVENTION, dtype=np.str_),
    }
    if include_shift:
        payload['applied_residual_shift_s_sorted'] = np.asarray(
            applied,
            dtype=np.float64,
        )
    np.savez(path, **payload)
    return path


def _state_with_store(tmp_path: Path, **store_kwargs: Any) -> tuple[AppState, Path]:
    state = create_app_state()
    store = tmp_path / 'line001.sgy.statics.datum.aaaa1111'
    _write_datum_store(store, **store_kwargs)
    state.file_registry.update(SOURCE_FILE_ID, store_path=str(store), dt=DT)
    return state, store


def _apply(
    tmp_path: Path,
    state: AppState,
    store: Path,
    solution_path: Path,
    *,
    options: ResidualStaticTraceStoreApplyOptions | None = None,
) -> svc.ResidualStaticCorrectedStoreResult:
    return apply_residual_static_correction_to_trace_store(
        source_file_id=SOURCE_FILE_ID,
        source_store_path=store,
        key1_byte=KEY1,
        key2_byte=KEY2,
        residual_solution_npz_path=solution_path,
        artifacts_dir=tmp_path / 'job',
        job_id=JOB_ID,
        state=state,
        options=options or ResidualStaticTraceStoreApplyOptions(),
        corrected_file_id=CORRECTED_FILE_ID,
    )


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_require_1d_float64_converts_float_array_to_contiguous_float64(
    dtype: np.dtype,
) -> None:
    values = np.asarray([0.0, 1.5, 2.5, 3.5], dtype=dtype)[::2]

    out = svc._require_1d_float64(values, name='field', expected_shape=(2,))

    assert out.dtype == np.float64
    assert out.flags.c_contiguous
    np.testing.assert_allclose(out, np.asarray([0.0, 2.5], dtype=np.float64))


@pytest.mark.parametrize(
    'values',
    [
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([True, False], dtype=bool),
        np.asarray([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128),
        np.asarray([1.0, 2.0], dtype=object),
    ],
)
def test_require_1d_float64_rejects_non_float_dtype(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match='field must be a float array'):
        svc._require_1d_float64(values, name='field', expected_shape=(2,))


@pytest.mark.parametrize('bad_value', [np.nan, np.inf])
def test_require_1d_float64_rejects_nonfinite_values(bad_value: float) -> None:
    values = np.asarray([0.0, bad_value], dtype=np.float64)

    with pytest.raises(ValueError, match='field must contain only finite values'):
        svc._require_1d_float64(values, name='field', expected_shape=(2,))


def test_require_1d_float64_rejects_shape_mismatch() -> None:
    values = np.asarray([0.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match='field shape mismatch'):
        svc._require_1d_float64(values, name='field', expected_shape=(3,))


def test_load_residual_solution_for_apply_validates_required_fields(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(
        tmp_path / 'residual_static_solution.npz',
        n_traces=2,
        estimated=np.asarray([0.001, -0.002], dtype=np.float64),
    )

    solution = load_residual_static_solution_for_apply(
        solution_path,
        expected_n_traces=2,
        expected_dt=DT,
        expected_key1_byte=KEY1,
        expected_key2_byte=KEY2,
    )

    np.testing.assert_allclose(
        solution.applied_residual_shift_s_sorted,
        np.asarray([-0.001, 0.002], dtype=np.float64),
    )
    np.testing.assert_allclose(
        solution.estimated_trace_delay_s_sorted,
        np.asarray([0.001, -0.002], dtype=np.float64),
    )
    assert solution.sign_convention == SIGN_CONVENTION


def test_load_residual_solution_for_apply_rejects_missing_shift(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(
        tmp_path / 'residual_static_solution.npz',
        n_traces=2,
        include_shift=False,
    )

    with pytest.raises(ValueError, match='applied_residual_shift_s_sorted'):
        load_residual_static_solution_for_apply(
            solution_path,
            expected_n_traces=2,
            expected_dt=DT,
            expected_key1_byte=KEY1,
            expected_key2_byte=KEY2,
        )


def test_load_residual_solution_for_apply_rejects_shape_mismatch(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(
        tmp_path / 'residual_static_solution.npz',
        n_traces=3,
        estimated=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match='shape mismatch'):
        load_residual_static_solution_for_apply(
            solution_path,
            expected_n_traces=2,
            expected_dt=DT,
            expected_key1_byte=KEY1,
            expected_key2_byte=KEY2,
        )


@pytest.mark.parametrize('bad_value', [np.nan, np.inf])
def test_load_residual_solution_for_apply_rejects_nan_or_inf_shift(
    tmp_path: Path,
    bad_value: float,
) -> None:
    solution_path = _write_solution(
        tmp_path / 'residual_static_solution.npz',
        n_traces=2,
        applied=np.asarray([0.0, bad_value], dtype=np.float64),
        estimated=np.asarray([0.0, 0.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match='finite'):
        load_residual_static_solution_for_apply(
            solution_path,
            expected_n_traces=2,
            expected_dt=DT,
            expected_key1_byte=KEY1,
            expected_key2_byte=KEY2,
        )


def test_load_residual_solution_for_apply_rejects_sign_mismatch(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(
        tmp_path / 'residual_static_solution.npz',
        n_traces=2,
        applied=np.asarray([0.001, 0.0], dtype=np.float64),
        estimated=np.asarray([0.001, 0.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match='must equal'):
        load_residual_static_solution_for_apply(
            solution_path,
            expected_n_traces=2,
            expected_dt=DT,
            expected_key1_byte=KEY1,
            expected_key2_byte=KEY2,
        )


def test_residual_shift_rejects_large_shift_without_clip(tmp_path: Path) -> None:
    solution = load_residual_static_solution_for_apply(
        _write_solution(
            tmp_path / 'residual_static_solution.npz',
            n_traces=2,
            estimated=np.asarray([0.0, 0.251], dtype=np.float64),
        ),
        expected_n_traces=2,
        expected_dt=DT,
        expected_key1_byte=KEY1,
        expected_key2_byte=KEY2,
    )

    with pytest.raises(ValueError, match='max_abs_shift_ms'):
        validate_residual_static_shift_for_apply(solution, max_abs_shift_ms=250.0)

    np.testing.assert_allclose(
        solution.applied_residual_shift_s_sorted,
        np.asarray([-0.0, -0.251], dtype=np.float64),
    )


def test_residual_corrected_store_applies_opposite_sign_shift_to_impulse(
    tmp_path: Path,
) -> None:
    traces = np.zeros((1, 128), dtype=np.float32)
    traces[0, 100] = 1.0
    state, store = _state_with_store(tmp_path, traces=traces)
    solution_path = _write_solution(
        tmp_path / 'residual_static_solution.npz',
        n_traces=1,
        estimated=np.asarray([0.008], dtype=np.float64),
    )

    result = _apply(tmp_path, state, store, solution_path)

    corrected = np.load(result.store_path / 'traces.npy')
    assert int(np.argmax(corrected[0])) == 98
    assert corrected[0, 98] == pytest.approx(1.0)
    assert corrected[0, 100] == pytest.approx(0.0)


def test_residual_corrected_store_registers_new_file_id(tmp_path: Path) -> None:
    state, store = _state_with_store(tmp_path)
    solution_path = _write_solution(
        tmp_path / 'residual_static_solution.npz',
        n_traces=3,
    )

    result = _apply(tmp_path, state, store, solution_path)

    assert result.file_id == CORRECTED_FILE_ID
    assert result.store_name == (
        'line001.sgy.statics.datum.aaaa1111.statics.residual.bbbb2222'
    )
    assert state.file_registry.get_store_path(CORRECTED_FILE_ID) == str(
        result.store_path
    )
    assert state.file_registry.get_dt(CORRECTED_FILE_ID) == pytest.approx(DT)
    with state.lock:
        cache_key = trace_store_cache_key(CORRECTED_FILE_ID, KEY1, KEY2)
        assert cache_key in state.cached_readers
        reader = state.cached_readers[cache_key]
    assert reader.get_section(100).arr.shape == (3, 120)


def test_residual_corrected_store_writes_corrected_file_json(
    tmp_path: Path,
) -> None:
    state, store = _state_with_store(tmp_path)
    solution_path = _write_solution(
        tmp_path / 'residual_static_solution.npz',
        n_traces=3,
    )

    result = _apply(tmp_path, state, store, solution_path)
    payload = json.loads(result.corrected_file_json_path.read_text(encoding='utf-8'))

    json.dumps(payload, allow_nan=False)
    assert payload['schema_version'] == 1
    assert payload['artifact_kind'] == 'corrected_file'
    assert payload['file_id'] == CORRECTED_FILE_ID
    assert payload['store_path'] == str(result.store_path)
    assert payload['derived_from_file_id'] == SOURCE_FILE_ID
    assert payload['derived_by'] == 'residual_static_correction'
    assert payload['applied_to'] == 'datum_corrected_trace_store'
    assert payload['shift_field'] == 'applied_residual_shift_s_sorted'
    assert payload['estimated_delay_field'] == 'estimated_trace_delay_s_sorted'
    assert not (tmp_path / 'job' / 'corrected_file.json.tmp').exists()


def test_residual_corrected_store_meta_appends_residual_component(
    tmp_path: Path,
) -> None:
    state, store = _state_with_store(tmp_path, original_segy_path='/keep/source.sgy')
    solution_path = _write_solution(
        tmp_path / 'residual_static_solution.npz',
        n_traces=3,
    )

    result = _apply(tmp_path, state, store, solution_path)
    meta = json.loads((result.store_path / 'meta.json').read_text(encoding='utf-8'))

    assert meta['original_segy_path'] == '/keep/source.sgy'
    assert meta['source_sha256'] is None
    components = meta['derived']['components']
    assert [component['name'] for component in components] == [
        'datum_static_correction',
        'residual_static_correction',
    ]
    assert components[-1]['job_id'] == JOB_ID
    assert components[-1]['shift_field'] == 'applied_residual_shift_s_sorted'
    assert meta['derived']['applied_to'] == 'datum_corrected_trace_store'


def test_residual_corrected_store_rejects_non_datum_source_store(
    tmp_path: Path,
) -> None:
    state, store = _state_with_store(tmp_path, derived={'kind': 'raw_trace_store'})
    solution_path = _write_solution(
        tmp_path / 'residual_static_solution.npz',
        n_traces=3,
    )

    with pytest.raises(ValueError, match='datum static lineage'):
        _apply(tmp_path, state, store, solution_path)


def test_residual_corrected_store_rejects_double_residual_application(
    tmp_path: Path,
) -> None:
    derived = {
        'kind': 'time_shifted_trace_store',
        'components': [
            _datum_component(),
            {'name': 'residual_static_correction', 'job_id': 'old-job'},
        ],
    }
    state, store = _state_with_store(tmp_path, derived=derived)
    solution_path = _write_solution(
        tmp_path / 'residual_static_solution.npz',
        n_traces=3,
    )

    with pytest.raises(ValueError, match='already has residual_static_correction'):
        _apply(tmp_path, state, store, solution_path)


def test_residual_corrected_store_rejects_output_dtype_other_than_float32(
    tmp_path: Path,
) -> None:
    state, store = _state_with_store(tmp_path)
    solution_path = _write_solution(
        tmp_path / 'residual_static_solution.npz',
        n_traces=3,
    )

    with pytest.raises(ValueError, match='output_dtype'):
        _apply(
            tmp_path,
            state,
            store,
            solution_path,
            options=ResidualStaticTraceStoreApplyOptions(output_dtype='float64'),
        )


def test_residual_corrected_store_rejects_register_false(tmp_path: Path) -> None:
    state, store = _state_with_store(tmp_path)
    solution_path = _write_solution(
        tmp_path / 'residual_static_solution.npz',
        n_traces=3,
    )

    with pytest.raises(ValueError, match='register_corrected_file'):
        _apply(
            tmp_path,
            state,
            store,
            solution_path,
            options=ResidualStaticTraceStoreApplyOptions(register_corrected_file=False),
        )


def test_residual_corrected_store_does_not_leave_partial_store_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, store = _state_with_store(tmp_path)
    solution_path = _write_solution(
        tmp_path / 'residual_static_solution.npz',
        n_traces=3,
    )

    def _fail_register(**_kwargs: Any) -> None:
        raise RuntimeError('registration failed')

    monkeypatch.setattr(svc, 'register_trace_store', _fail_register)

    with pytest.raises(RuntimeError, match='registration failed'):
        _apply(tmp_path, state, store, solution_path)

    output = tmp_path / (
        'line001.sgy.statics.datum.aaaa1111.statics.residual.bbbb2222'
    )
    assert not output.exists()
    assert list(tmp_path.glob(f'{output.name}.tmp-*')) == []
    assert state.file_registry.get_record(CORRECTED_FILE_ID) is None
    with state.lock:
        assert trace_store_cache_key(CORRECTED_FILE_ID, KEY1, KEY2) not in (
            state.cached_readers
        )
