from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import app.services.time_term_static_apply_trace_store as svc
from app.core.state import AppState, create_app_state
from app.services.time_term_static_artifacts import (
    DELAY_TO_SHIFT_CONVENTION,
    FINAL_SHIFT_CONVENTION,
)
from app.services.time_term_static_apply_trace_store import (
    TimeTermTraceStoreApplyOptions,
    apply_time_term_static_correction_to_trace_store,
    load_time_term_static_solution,
    select_time_term_shift_for_apply_mode,
)
from app.services.trace_store_baselines import write_trace_store_raw_baseline_artifacts
from app.services.trace_store_registration import trace_store_cache_key

KEY1 = 189
KEY2 = 193
DT = 0.004
SOURCE_FILE_ID = 'datum-residual-corrected-file-id'
CORRECTED_FILE_ID = 'time-term-corrected-file-id'
JOB_ID = 'cccc3333-time-term-job'
SIGN_CONVENTION = (
    'estimated_trace_time_term_delay_s = source_node_time_term_s + '
    'receiver_node_time_term_s; applied_weathering_shift_s = '
    '-estimated_trace_time_term_delay_s; corrected(t)=raw(t-shift_s)'
)


def _datum_component() -> dict[str, object]:
    return {
        'name': 'datum_static_correction',
        'job_id': 'aaaa1111-datum-job',
        'solution_artifact': 'datum_static_solution.npz',
        'shift_field': 'trace_shift_s_sorted',
        'value_kind': 'applied_event_time_shift_s',
    }


def _residual_component() -> dict[str, object]:
    return {
        'name': 'residual_static_correction',
        'job_id': 'bbbb2222-residual-job',
        'solution_artifact': 'residual_static_solution.npz',
        'shift_field': 'applied_residual_shift_s_sorted',
        'value_kind': 'applied_event_time_shift_s',
    }


def _default_derived() -> dict[str, object]:
    return {
        'kind': 'time_shifted_trace_store',
        'from_file_id': 'raw-file-id',
        'components': [_datum_component(), _residual_component()],
    }


def _write_source_store(
    store: Path,
    *,
    traces: np.ndarray | None = None,
    dt: float = DT,
    derived: dict[str, object] | None = None,
    include_derived: bool = True,
    original_segy_path: str = '/data/line001.sgy',
    source_sha256: str | None = None,
    write_baseline: bool = True,
) -> np.ndarray:
    store.mkdir(parents=True, exist_ok=True)
    if traces is None:
        traces = np.zeros((3, 16), dtype=np.float32)
        traces[:, 8] = 1.0
    traces = np.asarray(traces, dtype=np.float32)
    n_traces, n_samples = traces.shape
    np.save(store / 'traces.npy', traces)
    key1_values = np.asarray([100], dtype=np.int64)
    key1_offsets = np.asarray([0], dtype=np.int64)
    key1_counts = np.asarray([n_traces], dtype=np.int64)
    np.savez(
        store / 'index.npz',
        key1_values=key1_values,
        key1_offsets=key1_offsets,
        key1_counts=key1_counts,
        sorted_to_original=np.arange(n_traces, dtype=np.int64),
    )
    np.save(store / f'headers_byte_{KEY1}.npy', np.full(n_traces, 100, dtype=np.int32))
    np.save(store / f'headers_byte_{KEY2}.npy', np.arange(n_traces, dtype=np.int32))

    meta: dict[str, object] = {
        'schema_version': 1,
        'dtype': 'float32',
        'n_traces': int(n_traces),
        'n_samples': int(n_samples),
        'key_bytes': {'key1': KEY1, 'key2': KEY2},
        'sorted_by': ['key1', 'key2'],
        'dt': float(dt),
        'original_segy_path': original_segy_path,
        'source_sha256': source_sha256,
    }
    if include_derived:
        meta['derived'] = derived if derived is not None else _default_derived()
    (store / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')

    if write_baseline:
        trace_sum = traces.sum(axis=1, dtype=np.float64)
        trace_sumsq = np.einsum('ij,ij->i', traces, traces, dtype=np.float64)
        write_trace_store_raw_baseline_artifacts(
            store_path=store,
            key1_byte=KEY1,
            key2_byte=KEY2,
            dtype_base='float32',
            dt=dt,
            key1_values=key1_values,
            key1_offsets=key1_offsets,
            key1_counts=key1_counts,
            trace_sum=trace_sum,
            trace_sumsq=trace_sumsq,
            n_samples=n_samples,
            source_sha256=source_sha256,
        )
    return traces


def _write_solution(
    path: Path,
    *,
    n_traces: int = 3,
    n_samples: int = 16,
    dt: float = DT,
    key1_byte: int = KEY1,
    key2_byte: int = KEY2,
    applied_weathering: np.ndarray | None = None,
    estimated_delay: np.ndarray | None = None,
    datum_shift: np.ndarray | None = None,
    residual_shift: np.ndarray | None = None,
    final_shift: np.ndarray | None = None,
    artifact_kind: str = 'time_term_static_solution',
    order: str = 'trace_store_sorted',
    overrides: dict[str, Any] | None = None,
    drop_fields: tuple[str, ...] = (),
) -> Path:
    if applied_weathering is None:
        applied_weathering = np.zeros(n_traces, dtype=np.float64)
    applied_weathering = np.asarray(applied_weathering, dtype=np.float64)
    if estimated_delay is None:
        estimated_delay = -applied_weathering
    if datum_shift is None:
        datum_shift = np.zeros(n_traces, dtype=np.float64)
    if residual_shift is None:
        residual_shift = np.zeros(n_traces, dtype=np.float64)
    if final_shift is None:
        final_shift = (
            np.asarray(datum_shift, dtype=np.float64)
            + np.asarray(residual_shift, dtype=np.float64)
            + applied_weathering
        )
    payload: dict[str, Any] = {
        'schema_version': np.asarray(1, dtype=np.int64),
        'artifact_kind': np.asarray(artifact_kind, dtype=np.str_),
        'order': np.asarray(order, dtype=np.str_),
        'job_id': np.asarray(JOB_ID, dtype=np.str_),
        'input_file_id': np.asarray(SOURCE_FILE_ID, dtype=np.str_),
        'n_traces': np.asarray(n_traces, dtype=np.int64),
        'n_samples': np.asarray(n_samples, dtype=np.int64),
        'dt': np.asarray(dt, dtype=np.float64),
        'key1_byte': np.asarray(key1_byte, dtype=np.int64),
        'key2_byte': np.asarray(key2_byte, dtype=np.int64),
        'applied_weathering_shift_s_sorted': applied_weathering,
        'final_trace_shift_s_sorted': np.asarray(final_shift, dtype=np.float64),
        'datum_trace_shift_s_sorted': np.asarray(datum_shift, dtype=np.float64),
        'residual_applied_shift_s_sorted': np.asarray(
            residual_shift,
            dtype=np.float64,
        ),
        'final_used_trace_mask_sorted': np.ones(n_traces, dtype=bool),
        'rejected_trace_mask_sorted': np.zeros(n_traces, dtype=bool),
        'rejected_iteration_sorted': np.full(n_traces, -1, dtype=np.int64),
        'estimated_trace_time_term_delay_s_sorted': np.asarray(
            estimated_delay,
            dtype=np.float64,
        ),
        'node_time_term_s': np.zeros(2, dtype=np.float64),
        'sign_convention': np.asarray(SIGN_CONVENTION, dtype=np.str_),
        'delay_to_shift_convention': np.asarray(
            DELAY_TO_SHIFT_CONVENTION,
            dtype=np.str_,
        ),
        'final_shift_convention': np.asarray(FINAL_SHIFT_CONVENTION, dtype=np.str_),
    }
    if overrides:
        payload.update(overrides)
    for field in drop_fields:
        payload.pop(field, None)
    np.savez(path, **payload)
    return path


def _state_with_store(
    tmp_path: Path,
    **store_kwargs: Any,
) -> tuple[AppState, Path]:
    state = create_app_state()
    store = tmp_path / 'line001.sgy.statics.residual.bbbb2222'
    _write_source_store(store, **store_kwargs)
    state.file_registry.update(SOURCE_FILE_ID, store_path=str(store), dt=DT)
    return state, store


def _apply(
    tmp_path: Path,
    state: AppState,
    solution_path: Path,
    *,
    options: TimeTermTraceStoreApplyOptions | None = None,
) -> svc.TimeTermTraceStoreApplyResult:
    if options is None:
        options = TimeTermTraceStoreApplyOptions(
            corrected_file_id=CORRECTED_FILE_ID,
        )
    return apply_time_term_static_correction_to_trace_store(
        source_file_id=SOURCE_FILE_ID,
        key1_byte=KEY1,
        key2_byte=KEY2,
        solution_npz_path=solution_path,
        artifacts_dir=tmp_path / 'job',
        state=state,
        options=options,
    )


def test_load_time_term_static_solution_reads_required_fields(tmp_path: Path) -> None:
    solution_path = _write_solution(
        tmp_path / 'time_term_static_solution.npz',
        applied_weathering=np.asarray([-0.004, 0.0, 0.004], dtype=np.float64),
    )

    solution = load_time_term_static_solution(
        solution_path,
        expected_n_traces=3,
        expected_dt=DT,
        expected_key1_byte=KEY1,
        expected_key2_byte=KEY2,
    )

    assert solution.artifact_kind == 'time_term_static_solution'
    assert solution.order == 'trace_store_sorted'
    assert solution.job_id == JOB_ID
    np.testing.assert_allclose(
        solution.applied_weathering_shift_s_sorted,
        np.asarray([-0.004, 0.0, 0.004], dtype=np.float64),
    )


def test_load_time_term_static_solution_rejects_wrong_artifact_kind(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(
        tmp_path / 'time_term_static_solution.npz',
        artifact_kind='other',
    )

    with pytest.raises(ValueError, match='artifact_kind'):
        load_time_term_static_solution(solution_path)


def test_load_time_term_static_solution_rejects_order_not_sorted(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(
        tmp_path / 'time_term_static_solution.npz',
        order='input_order',
    )

    with pytest.raises(ValueError, match='order'):
        load_time_term_static_solution(solution_path)


def test_load_time_term_static_solution_rejects_shift_shape_mismatch(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(
        tmp_path / 'time_term_static_solution.npz',
        n_traces=3,
        overrides={
            'applied_weathering_shift_s_sorted': np.zeros(2, dtype=np.float64),
        },
    )

    with pytest.raises(ValueError, match='shape mismatch'):
        load_time_term_static_solution(solution_path)


def test_load_time_term_static_solution_rejects_delay_to_shift_sign_mismatch(
    tmp_path: Path,
) -> None:
    solution_path = _write_solution(
        tmp_path / 'time_term_static_solution.npz',
        applied_weathering=np.asarray([0.001, 0.0, 0.0], dtype=np.float64),
        estimated_delay=np.asarray([0.001, 0.0, 0.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match='must equal'):
        load_time_term_static_solution(solution_path)


def test_load_time_term_static_solution_rejects_object_dtype(tmp_path: Path) -> None:
    solution_path = _write_solution(
        tmp_path / 'time_term_static_solution.npz',
        overrides={'artifact_kind': np.asarray({'bad': 'object'}, dtype=object)},
    )

    with pytest.raises(ValueError, match='object dtype'):
        load_time_term_static_solution(solution_path)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_require_1d_float64_coerces_floating_array(dtype: np.dtype) -> None:
    out = svc._require_1d_float64(
        np.asarray([1.0, 2.0], dtype=dtype),
        name='values',
        expected_shape=(2,),
    )

    assert out.dtype == np.float64
    assert out.flags.c_contiguous
    np.testing.assert_allclose(out, [1.0, 2.0])


@pytest.mark.parametrize(
    'values',
    [
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([True, False], dtype=bool),
        np.asarray([1.0 + 0.0j], dtype=np.complex128),
        np.asarray([1.0], dtype=object),
    ],
)
def test_require_1d_float64_rejects_non_floating_dtype(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        svc._require_1d_float64(values, name='values')


@pytest.mark.parametrize(
    'values',
    [
        np.asarray([np.nan], dtype=np.float64),
        np.asarray([np.inf], dtype=np.float64),
    ],
)
def test_require_1d_float64_rejects_non_finite(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match='finite'):
        svc._require_1d_float64(values, name='values')


def test_require_1d_bool_accepts_bool_only() -> None:
    out = svc._require_1d_bool(
        np.asarray([True, False], dtype=bool),
        name='mask',
        expected_shape=(2,),
    )

    assert out.dtype == bool
    assert out.flags.c_contiguous
    with pytest.raises(ValueError):
        svc._require_1d_bool(
            np.asarray([1, 0], dtype=np.int64),
            name='mask',
            expected_shape=(2,),
        )


def test_require_1d_int64_accepts_integer_only() -> None:
    out = svc._require_1d_int64(
        np.asarray([1, 2], dtype=np.int32),
        name='indices',
        expected_shape=(2,),
    )

    assert out.dtype == np.int64
    assert out.flags.c_contiguous
    np.testing.assert_array_equal(out, [1, 2])
    with pytest.raises(ValueError):
        svc._require_1d_int64(
            np.asarray([True, False], dtype=bool),
            name='indices',
            expected_shape=(2,),
        )


def test_require_1d_int64_rejects_integer_like_float() -> None:
    with pytest.raises(ValueError):
        svc._require_1d_int64(
            np.asarray([1.0, 2.0], dtype=np.float64),
            name='indices',
            expected_shape=(2,),
        )


@pytest.mark.parametrize(
    'validator,values,kwargs',
    [
        (svc._require_1d_float64, np.asarray([1.0, 2.0]), {'expected_shape': (3,)}),
        (svc._require_1d_bool, np.asarray([True, False]), {'expected_shape': (3,)}),
        (svc._require_1d_int64, np.asarray([1, 2]), {'expected_shape': (3,)}),
    ],
)
def test_require_1d_validators_reject_shape_mismatch(
    validator: Any,
    values: np.ndarray,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match='shape mismatch'):
        validator(values, name='values', **kwargs)


def test_time_term_apply_selects_weathering_shift_for_weathering_only_mode(
    tmp_path: Path,
) -> None:
    solution = load_time_term_static_solution(
        _write_solution(
            tmp_path / 'time_term_static_solution.npz',
            applied_weathering=np.asarray([-0.004, 0.0, 0.004], dtype=np.float64),
            residual_shift=np.asarray([0.104, 0.2, 0.296], dtype=np.float64),
            final_shift=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        )
    )

    field, shift = select_time_term_shift_for_apply_mode(
        solution,
        mode='weathering_only',
    )

    assert field == 'applied_weathering_shift_s_sorted'
    np.testing.assert_allclose(shift, [-0.004, 0.0, 0.004])


def test_time_term_apply_builds_and_registers_corrected_trace_store(
    tmp_path: Path,
) -> None:
    state, store = _state_with_store(tmp_path)
    solution_path = _write_solution(
        tmp_path / 'time_term_static_solution.npz',
        applied_weathering=np.asarray([-0.004, 0.0, 0.004], dtype=np.float64),
    )

    result = _apply(tmp_path, state, solution_path)

    corrected = np.load(result.store_path / 'traces.npy')
    assert [int(np.argmax(row)) for row in corrected] == [7, 8, 9]
    assert result.file_id == CORRECTED_FILE_ID
    assert result.store_name == 'line001.sgy.statics.residual.bbbb2222.statics.time_term.cccc3333'
    assert result.source_store_path == store
    assert result.applied_shift_field == 'applied_weathering_shift_s_sorted'
    assert result.shift_min_ms == pytest.approx(-4.0)
    assert result.shift_max_ms == pytest.approx(4.0)
    assert state.file_registry.get_store_path(CORRECTED_FILE_ID) == str(
        result.store_path
    )
    with state.lock:
        cache_key = trace_store_cache_key(CORRECTED_FILE_ID, KEY1, KEY2)
        assert cache_key in state.cached_readers
        reader = state.cached_readers[cache_key]
    assert reader.get_section(100).arr.shape == (3, 16)


def test_time_term_apply_writes_manifest_and_appends_metadata(
    tmp_path: Path,
) -> None:
    state, _store = _state_with_store(tmp_path, original_segy_path='/keep/source.sgy')
    solution_path = _write_solution(
        tmp_path / 'time_term_static_solution.npz',
        applied_weathering=np.asarray([-0.004, 0.0, 0.004], dtype=np.float64),
    )

    result = _apply(tmp_path, state, solution_path)

    assert result.corrected_file_json_path is not None
    payload = json.loads(result.corrected_file_json_path.read_text(encoding='utf-8'))
    json.dumps(payload, allow_nan=False)
    assert payload['schema_version'] == 1
    assert payload['artifact_kind'] == 'corrected_file'
    assert payload['file_id'] == CORRECTED_FILE_ID
    assert payload['derived_by'] == 'time_term_static_correction'
    assert payload['apply_mode'] == 'weathering_only'
    assert payload['applied_shift_field'] == 'applied_weathering_shift_s_sorted'
    assert payload['shift_ms']['max_abs'] == pytest.approx(4.0)
    assert not (tmp_path / 'job' / 'corrected_file.json.tmp').exists()

    meta = json.loads((result.store_path / 'meta.json').read_text(encoding='utf-8'))
    assert meta['original_segy_path'] == '/keep/source.sgy'
    assert meta['source_sha256'] is None
    components = meta['derived']['components']
    assert [component['name'] for component in components] == [
        'datum_static_correction',
        'residual_static_correction',
        'time_term_static_correction',
    ]
    assert components[-1]['job_id'] == JOB_ID
    assert components[-1]['shift_field'] == 'applied_weathering_shift_s_sorted'
    assert components[-1]['apply_mode'] == 'weathering_only'


def test_time_term_apply_rejects_final_from_raw_if_not_implemented(
    tmp_path: Path,
) -> None:
    state, _store = _state_with_store(tmp_path)
    solution_path = _write_solution(tmp_path / 'time_term_static_solution.npz')

    with pytest.raises(ValueError, match='final_from_raw mode is not implemented'):
        _apply(
            tmp_path,
            state,
            solution_path,
            options=TimeTermTraceStoreApplyOptions(
                mode='final_from_raw',
                corrected_file_id=CORRECTED_FILE_ID,
            ),
        )


def test_time_term_apply_rejects_output_dtype_other_than_float32(
    tmp_path: Path,
) -> None:
    state, _store = _state_with_store(tmp_path)
    solution_path = _write_solution(tmp_path / 'time_term_static_solution.npz')

    with pytest.raises(ValueError, match='output_dtype'):
        _apply(
            tmp_path,
            state,
            solution_path,
            options=TimeTermTraceStoreApplyOptions(
                output_dtype='float64',
                corrected_file_id=CORRECTED_FILE_ID,
            ),
        )


def test_time_term_apply_rejects_interpolation_other_than_linear(
    tmp_path: Path,
) -> None:
    state, _store = _state_with_store(tmp_path)
    solution_path = _write_solution(tmp_path / 'time_term_static_solution.npz')

    with pytest.raises(ValueError, match='interpolation'):
        _apply(
            tmp_path,
            state,
            solution_path,
            options=TimeTermTraceStoreApplyOptions(
                interpolation='nearest',
                corrected_file_id=CORRECTED_FILE_ID,
            ),
        )


def test_time_term_apply_rejects_non_finite_shift(tmp_path: Path) -> None:
    state, _store = _state_with_store(tmp_path)
    solution_path = _write_solution(
        tmp_path / 'time_term_static_solution.npz',
        applied_weathering=np.asarray([np.nan, 0.0, 0.0], dtype=np.float64),
        estimated_delay=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match='finite'):
        _apply(tmp_path, state, solution_path)


def test_time_term_apply_rejects_shift_above_max_abs_shift_ms(
    tmp_path: Path,
) -> None:
    state, _store = _state_with_store(tmp_path)
    solution_path = _write_solution(
        tmp_path / 'time_term_static_solution.npz',
        applied_weathering=np.asarray([0.501, 0.0, 0.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match='max_abs_shift_ms'):
        _apply(tmp_path, state, solution_path)


def test_time_term_apply_rejects_n_traces_mismatch(tmp_path: Path) -> None:
    state, _store = _state_with_store(tmp_path)
    solution_path = _write_solution(
        tmp_path / 'time_term_static_solution.npz',
        n_traces=2,
        n_samples=16,
    )

    with pytest.raises(ValueError, match='n_traces mismatch'):
        _apply(tmp_path, state, solution_path)


def test_time_term_apply_rejects_dt_mismatch(tmp_path: Path) -> None:
    state, _store = _state_with_store(tmp_path)
    solution_path = _write_solution(
        tmp_path / 'time_term_static_solution.npz',
        dt=0.006,
    )

    with pytest.raises(ValueError, match='dt mismatch'):
        _apply(tmp_path, state, solution_path)


def test_time_term_apply_rejects_key_byte_mismatch(tmp_path: Path) -> None:
    state, _store = _state_with_store(tmp_path)
    solution_path = _write_solution(
        tmp_path / 'time_term_static_solution.npz',
        key1_byte=181,
    )

    with pytest.raises(ValueError, match='key1_byte mismatch'):
        _apply(tmp_path, state, solution_path)


def test_time_term_apply_rejects_raw_store_in_weathering_only_mode(
    tmp_path: Path,
) -> None:
    state, _store = _state_with_store(tmp_path, include_derived=False)
    solution_path = _write_solution(tmp_path / 'time_term_static_solution.npz')

    with pytest.raises(ValueError, match='weathering_only mode requires'):
        _apply(tmp_path, state, solution_path)


def test_time_term_apply_rejects_source_sha_raw_store_in_weathering_only_mode(
    tmp_path: Path,
) -> None:
    state, _store = _state_with_store(tmp_path, source_sha256='raw-source-sha')
    solution_path = _write_solution(tmp_path / 'time_term_static_solution.npz')

    with pytest.raises(ValueError, match='weathering_only mode requires'):
        _apply(tmp_path, state, solution_path)


def test_time_term_apply_rejects_double_time_term_application(
    tmp_path: Path,
) -> None:
    derived = _default_derived()
    derived['components'] = [
        *_default_derived()['components'],
        {'name': 'time_term_static_correction', 'job_id': 'old-job'},
    ]
    state, _store = _state_with_store(tmp_path, derived=derived)
    solution_path = _write_solution(tmp_path / 'time_term_static_solution.npz')

    with pytest.raises(ValueError, match='already has time_term_static_correction'):
        _apply(tmp_path, state, solution_path)


def test_time_term_apply_registration_failure_cleans_registry_cache_and_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _store = _state_with_store(tmp_path)
    solution_path = _write_solution(tmp_path / 'time_term_static_solution.npz')

    def _fail_register(**kwargs: Any) -> object:
        state.file_registry.update(
            str(kwargs['file_id']),
            store_path=str(kwargs['store_dir']),
            dt=DT,
        )
        with state.lock:
            state.cached_readers[
                trace_store_cache_key(str(kwargs['file_id']), KEY1, KEY2)
            ] = object()
        raise RuntimeError('registration failed')

    monkeypatch.setattr(svc, 'register_trace_store', _fail_register)

    with pytest.raises(RuntimeError, match='registration failed'):
        _apply(tmp_path, state, solution_path)

    output = tmp_path / (
        'line001.sgy.statics.residual.bbbb2222.statics.time_term.cccc3333'
    )
    assert not output.exists()
    assert state.file_registry.get_record(CORRECTED_FILE_ID) is None
    assert not (tmp_path / 'job' / 'corrected_file.json').exists()
    with state.lock:
        assert trace_store_cache_key(CORRECTED_FILE_ID, KEY1, KEY2) not in (
            state.cached_readers
        )


def test_time_term_apply_build_failure_cleans_partial_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _store = _state_with_store(tmp_path)
    solution_path = _write_solution(tmp_path / 'time_term_static_solution.npz')

    def _fail_build(**kwargs: Any) -> object:
        output_path = Path(kwargs['output_store_path'])
        output_path.mkdir()
        output_path.with_name(f'{output_path.name}.tmp-test').mkdir()
        raise RuntimeError('build failed')

    monkeypatch.setattr(svc, 'build_time_shifted_trace_store', _fail_build)

    with pytest.raises(RuntimeError, match='build failed'):
        _apply(tmp_path, state, solution_path)

    output = tmp_path / (
        'line001.sgy.statics.residual.bbbb2222.statics.time_term.cccc3333'
    )
    assert not output.exists()
    assert list(tmp_path.glob(f'{output.name}.tmp-*')) == []
    assert state.file_registry.get_record(CORRECTED_FILE_ID) is None
