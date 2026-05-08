from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import app.services.refraction_static_apply_trace_store as svc
from app.api.schemas import RefractionStaticApplyRequest
from app.core.state import AppState, create_app_state
from app.services.refraction_static_apply_trace_store import (
    CORRECTED_FILE_JSON_NAME,
    REFRACTION_STATIC_APPLY_QC_JSON_NAME,
    RefractionStaticTraceStoreApplyError,
    apply_refraction_statics_from_solution_artifact,
    apply_refraction_statics_to_trace_store,
    apply_trace_shifts_to_array,
    validate_refraction_trace_shifts_for_application,
)
from app.services.refraction_static_artifacts import (
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    write_refraction_static_solution_npz,
)
from app.services.trace_store_registration import trace_store_cache_key
from app.tests._refraction_static_artifact_helpers import _result as _artifact_result

KEY1 = 189
KEY2 = 193
DT = 0.004
SOURCE_FILE_ID = 'raw-file-id'
JOB_ID = 'ref-job-386'
SORTED_TO_ORIGINAL = np.asarray([2, 0, 3, 1], dtype=np.int64)


def _request(*, register_corrected_file: bool = True) -> RefractionStaticApplyRequest:
    return RefractionStaticApplyRequest.model_validate(
        {
            'file_id': SOURCE_FILE_ID,
            'key1_byte': KEY1,
            'key2_byte': KEY2,
            'pick_source': {
                'kind': 'batch_predicted_npz',
                'job_id': 'pick-job',
                'artifact_name': 'predicted_picks_time_s.npz',
            },
            'linkage': {'mode': 'none'},
            'model': {
                'method': 'gli_variable_thickness',
                'weathering_velocity_m_s': 800.0,
                'bedrock_velocity_mode': 'solve_global',
            },
            'datum': {
                'mode': 'floating_and_flat',
                'floating_datum_mode': 'constant',
                'floating_datum_elevation_m': 120.0,
                'flat_datum_elevation_m': 300.0,
            },
            'apply': {
                'mode': 'refraction_from_raw',
                'interpolation': 'linear',
                'fill_value': 0.0,
                'max_abs_shift_ms': 250.0,
                'output_dtype': 'float32',
                'register_corrected_file': register_corrected_file,
            },
        }
    )


def _valid_result(
    *,
    shifts: np.ndarray | None = None,
    sorted_trace_index: np.ndarray = SORTED_TO_ORIGINAL,
    valid_mask: np.ndarray | None = None,
    statuses: np.ndarray | None = None,
):
    if shifts is None:
        shifts = np.asarray([0.0, 0.008, -0.008, 0.0], dtype=np.float64)
    if valid_mask is None:
        valid_mask = np.ones(4, dtype=bool)
    if statuses is None:
        statuses = np.asarray(['ok', 'ok', 'ok', 'ok'], dtype='<U16')
    return replace(
        _artifact_result(),
        sorted_trace_index=np.asarray(sorted_trace_index, dtype=np.int64),
        refraction_trace_shift_s_sorted=np.asarray(shifts, dtype=np.float64),
        trace_static_valid_mask_sorted=np.asarray(valid_mask, dtype=bool),
        trace_static_status_sorted=np.asarray(statuses),
    )


def _write_source_store(
    store: Path,
    *,
    traces: np.ndarray | None = None,
    sorted_to_original: np.ndarray = SORTED_TO_ORIGINAL,
    dt: float = DT,
) -> np.ndarray:
    store.mkdir(parents=True, exist_ok=True)
    if traces is None:
        traces = np.zeros((4, 16), dtype=np.float32)
        traces[:, 8] = 1.0
        traces[3] = np.arange(16, dtype=np.float32)
    traces = np.asarray(traces, dtype=np.float32)
    n_traces, n_samples = traces.shape
    np.save(store / 'traces.npy', traces)
    np.savez(
        store / 'index.npz',
        key1_values=np.asarray([100], dtype=np.int64),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([n_traces], dtype=np.int64),
        sorted_to_original=np.asarray(sorted_to_original, dtype=np.int64),
    )
    np.save(store / f'headers_byte_{KEY1}.npy', np.full(n_traces, 100, dtype=np.int32))
    np.save(store / f'headers_byte_{KEY2}.npy', np.arange(n_traces, dtype=np.int32))
    np.save(store / 'headers_byte_45.npy', np.arange(10, 10 + n_traces, dtype=np.int32))
    meta: dict[str, Any] = {
        'schema_version': 1,
        'dtype': 'float32',
        'n_traces': int(n_traces),
        'n_samples': int(n_samples),
        'key_bytes': {'key1': KEY1, 'key2': KEY2},
        'sorted_by': ['key1', 'key2'],
        'dt': float(dt),
        'original_segy_path': '/data/line001.sgy',
        'source_sha256': None,
        'original_name': 'line001.sgy',
    }
    (store / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    return traces


def _state_with_source_store(tmp_path: Path) -> tuple[AppState, Path, np.ndarray]:
    store = tmp_path / 'trace_stores' / 'line001.sgy'
    traces = _write_source_store(store)
    state = create_app_state()
    state.file_registry.update(SOURCE_FILE_ID, store_path=str(store), dt=DT)
    return state, store, traces


def test_apply_trace_shifts_to_array_sign_fill_fraction_and_dtype() -> None:
    traces = np.zeros((4, 16), dtype=np.float32)
    traces[:3, 8] = 1.0
    traces[3] = np.arange(16, dtype=np.float32)
    original = traces.copy()

    corrected = apply_trace_shifts_to_array(
        traces=traces,
        sample_interval_s=DT,
        trace_shift_s_sorted=np.asarray([0.0, 0.008, -0.008, 0.002]),
        fill_value=-9.0,
        output_dtype=np.dtype('float32'),
    )

    assert corrected.dtype == np.float32
    np.testing.assert_array_equal(traces, original)
    assert int(np.argmax(corrected[0])) == 8
    assert int(np.argmax(corrected[1])) == 10
    assert int(np.argmax(corrected[2])) == 6
    assert corrected[3, 0] == pytest.approx(-9.0)
    assert corrected[3, 1] == pytest.approx(0.5)


def test_refraction_shift_validation_rejects_bad_inputs() -> None:
    with pytest.raises(RefractionStaticTraceStoreApplyError, match='shape mismatch'):
        validate_refraction_trace_shifts_for_application(
            trace_shift_s_sorted=np.zeros(3, dtype=np.float64),
            trace_static_valid_mask_sorted=np.ones(4, dtype=bool),
            trace_static_status_sorted=np.asarray(['ok'] * 4),
            n_traces=4,
            max_abs_shift_ms=250.0,
        )

    with pytest.raises(RefractionStaticTraceStoreApplyError, match='finite'):
        validate_refraction_trace_shifts_for_application(
            trace_shift_s_sorted=np.asarray([0.0, np.nan, 0.0, 0.0]),
            trace_static_valid_mask_sorted=np.ones(4, dtype=bool),
            trace_static_status_sorted=np.asarray(['ok'] * 4),
            n_traces=4,
            max_abs_shift_ms=250.0,
        )

    with pytest.raises(RefractionStaticTraceStoreApplyError, match='invalid'):
        validate_refraction_trace_shifts_for_application(
            trace_shift_s_sorted=np.asarray([0.0, np.nan, 0.0, 0.0]),
            trace_static_valid_mask_sorted=np.asarray([True, False, True, True]),
            trace_static_status_sorted=np.asarray(['ok', 'missing_node', 'ok', 'ok']),
            n_traces=4,
            max_abs_shift_ms=250.0,
        )

    with pytest.raises(RefractionStaticTraceStoreApplyError, match='exceeds'):
        validate_refraction_trace_shifts_for_application(
            trace_shift_s_sorted=np.asarray([0.0, 0.006, 0.0, 0.0]),
            trace_static_valid_mask_sorted=np.ones(4, dtype=bool),
            trace_static_status_sorted=np.asarray(['ok'] * 4),
            n_traces=4,
            max_abs_shift_ms=5.0,
        )


def test_apply_refraction_statics_builds_and_registers_corrected_trace_store(
    tmp_path: Path,
) -> None:
    state, source_store, source_traces = _state_with_source_store(tmp_path)
    req = _request(register_corrected_file=True)
    job_dir = tmp_path / 'jobs' / JOB_ID

    result = apply_refraction_statics_to_trace_store(
        req=req,
        result=_valid_result(),
        state=state,
        job_id=JOB_ID,
        job_dir=job_dir,
    )

    assert result.corrected_file_id is not None
    assert result.corrected_file_id != SOURCE_FILE_ID
    corrected_store = result.corrected_trace_store_path
    assert corrected_store is not None
    assert corrected_store.name == 'line001.sgy.statics.refraction.ref-job-386'
    assert state.file_registry.get_store_path(result.corrected_file_id) == str(
        corrected_store
    )
    with state.lock:
        assert trace_store_cache_key(result.corrected_file_id, KEY1, KEY2) in (
            state.cached_readers
        )

    corrected = np.load(corrected_store / 'traces.npy')
    assert [int(np.argmax(corrected[i])) for i in range(3)] == [8, 10, 6]
    np.testing.assert_allclose(corrected[3], source_traces[3])
    np.testing.assert_array_equal(np.load(source_store / 'traces.npy'), source_traces)

    with np.load(source_store / 'index.npz', allow_pickle=False) as src:
        with np.load(corrected_store / 'index.npz', allow_pickle=False) as dst:
            np.testing.assert_array_equal(
                dst['sorted_to_original'],
                src['sorted_to_original'],
            )
    np.testing.assert_array_equal(
        np.load(corrected_store / 'headers_byte_45.npy'),
        np.load(source_store / 'headers_byte_45.npy'),
    )

    meta = json.loads((corrected_store / 'meta.json').read_text(encoding='utf-8'))
    assert meta['n_traces'] == 4
    assert meta['n_samples'] == 16
    assert meta['dt'] == pytest.approx(DT)
    assert meta['derived']['statics_kind'] == 'refraction'
    assert meta['derived']['from_file_id'] == SOURCE_FILE_ID
    assert meta['derived']['components'][-1]['name'] == 'refraction_static_correction'

    corrected_manifest = json.loads(
        (job_dir / CORRECTED_FILE_JSON_NAME).read_text(encoding='utf-8')
    )
    assert corrected_manifest['corrected_file_id'] == result.corrected_file_id
    assert corrected_manifest['source_file_id'] == SOURCE_FILE_ID
    assert corrected_manifest['statics_kind'] == 'refraction'
    assert corrected_manifest['sign_convention'] == 'corrected(t) = raw(t - shift_s)'
    assert corrected_manifest['n_traces'] == 4
    assert 'store_path' not in corrected_manifest
    assert 'derived_from_store_path' not in corrected_manifest
    assert REFRACTION_STATIC_APPLY_QC_JSON_NAME in corrected_manifest['artifact_names']

    qc = json.loads(
        (job_dir / REFRACTION_STATIC_APPLY_QC_JSON_NAME).read_text(encoding='utf-8')
    )
    assert qc['corrected_file_id'] == result.corrected_file_id
    assert qc['n_valid_trace_shifts'] == 4
    assert qc['n_invalid_trace_shifts'] == 0
    assert qc['max_abs_applied_shift_ms'] == pytest.approx(8.0)
    assert 'source_trace_store_path' not in qc
    assert 'corrected_trace_store_path' not in qc


def test_apply_refraction_statics_from_solution_artifact(
    tmp_path: Path,
) -> None:
    state, _source_store, _source_traces = _state_with_source_store(tmp_path)
    req = _request(register_corrected_file=True)
    job_dir = tmp_path / 'jobs' / JOB_ID
    solution_path = job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME
    job_dir.mkdir(parents=True)
    write_refraction_static_solution_npz(
        result=_valid_result(),
        req=req,
        path=solution_path,
    )

    result = apply_refraction_statics_from_solution_artifact(
        req=req,
        solution_npz_path=solution_path,
        state=state,
        job_id=JOB_ID,
        job_dir=job_dir,
    )

    assert result.corrected_file_id is not None
    assert result.corrected_trace_store_path is not None
    corrected = np.load(result.corrected_trace_store_path / 'traces.npy')
    assert [int(np.argmax(corrected[i])) for i in range(3)] == [8, 10, 6]


def test_apply_refraction_statics_rejects_sorted_order_mismatch_without_output(
    tmp_path: Path,
) -> None:
    state, source_store, _source_traces = _state_with_source_store(tmp_path)
    req = _request(register_corrected_file=True)
    job_dir = tmp_path / 'jobs' / JOB_ID

    with pytest.raises(RefractionStaticTraceStoreApplyError, match='sorted_trace_index'):
        apply_refraction_statics_to_trace_store(
            req=req,
            result=_valid_result(sorted_trace_index=np.arange(4, dtype=np.int64)),
            state=state,
            job_id=JOB_ID,
            job_dir=job_dir,
        )

    assert len(state.file_registry.records) == 1
    assert not (
        source_store.parent / 'line001.sgy.statics.refraction.ref-job-386'
    ).exists()
    assert not (job_dir / CORRECTED_FILE_JSON_NAME).exists()


def test_apply_refraction_statics_cleans_built_store_on_registration_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, source_store, _source_traces = _state_with_source_store(tmp_path)
    req = _request(register_corrected_file=True)
    job_dir = tmp_path / 'jobs' / JOB_ID

    def _fail_register(**_kwargs: Any) -> object:
        raise RuntimeError('registration failed')

    monkeypatch.setattr(svc, 'register_trace_store', _fail_register)

    with pytest.raises(RuntimeError, match='registration failed'):
        apply_refraction_statics_to_trace_store(
            req=req,
            result=_valid_result(),
            state=state,
            job_id=JOB_ID,
            job_dir=job_dir,
        )

    assert len(state.file_registry.records) == 1
    assert not (
        source_store.parent / 'line001.sgy.statics.refraction.ref-job-386'
    ).exists()
    assert not (job_dir / CORRECTED_FILE_JSON_NAME).exists()
    assert not (job_dir / REFRACTION_STATIC_APPLY_QC_JSON_NAME).exists()
