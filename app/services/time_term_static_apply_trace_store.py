"""Apply time-term static solution artifacts to TraceStores."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from app.core.state import AppState
from app.services.common.artifact_io import write_json_atomic
from app.services.corrected_trace_store import (
    TimeShiftedTraceStoreResult,
    build_time_shifted_trace_store,
)
from app.services.reader import get_reader
from app.services.trace_store_registration import (
    register_trace_store,
    trace_store_cache_key,
)
from app.utils.baseline_artifacts import has_split_baseline_artifacts

TimeTermTraceStoreApplyMode = Literal['weathering_only', 'final_from_raw']

_CORRECTED_FILE_NAME = 'corrected_file.json'
_SAFE_STORE_NAME_RE = re.compile(r'[^A-Za-z0-9_.-]+')
_SCHEMA_VERSION = 1
_ARTIFACT_KIND = 'time_term_static_solution'
_ORDER = 'trace_store_sorted'
_DT_ATOL = 1.0e-9
_DT_RTOL = 0.0
_SHIFT_RTOL = 1.0e-7
_SHIFT_ATOL = 1.0e-9
_BUILDER_SIGN_CONVENTION = (
    'corrected(t)=raw(t-shift_s); positive_shift_delays_events'
)
_REQUIRED_SOLUTION_FIELDS = {
    'schema_version',
    'artifact_kind',
    'order',
    'job_id',
    'input_file_id',
    'n_traces',
    'n_samples',
    'dt',
    'key1_byte',
    'key2_byte',
    'applied_weathering_shift_s_sorted',
    'final_trace_shift_s_sorted',
    'datum_trace_shift_s_sorted',
    'residual_applied_shift_s_sorted',
    'final_used_trace_mask_sorted',
    'rejected_trace_mask_sorted',
    'rejected_iteration_sorted',
    'estimated_trace_time_term_delay_s_sorted',
    'node_time_term_s',
    'sign_convention',
    'delay_to_shift_convention',
    'final_shift_convention',
}


@dataclass(frozen=True)
class TimeTermTraceStoreApplyOptions:
    mode: TimeTermTraceStoreApplyMode = 'weathering_only'
    interpolation: str = 'linear'
    fill_value: float = 0.0
    output_dtype: str = 'float32'
    max_abs_shift_ms: float = 500.0
    register_corrected_file: bool = True
    corrected_file_id: str | None = None
    corrected_store_name: str | None = None


@dataclass(frozen=True)
class TimeTermTraceStoreApplyResult:
    file_id: str
    store_path: Path
    store_name: str

    source_file_id: str
    source_store_path: Path

    solution_npz_path: Path
    job_id: str | None

    mode: TimeTermTraceStoreApplyMode
    applied_shift_field: str

    key1_byte: int
    key2_byte: int
    dt: float
    n_traces: int
    n_samples: int

    shift_min_ms: float
    shift_max_ms: float
    shift_mean_ms: float
    shift_max_abs_ms: float

    corrected_file_json_path: Path | None


@dataclass(frozen=True)
class LoadedTimeTermStaticSolution:
    path: Path
    schema_version: int
    artifact_kind: str
    order: str

    n_traces: int
    n_samples: int | None
    dt: float
    key1_byte: int | None
    key2_byte: int | None

    job_id: str | None
    input_file_id: str | None

    applied_weathering_shift_s_sorted: np.ndarray
    final_trace_shift_s_sorted: np.ndarray
    datum_trace_shift_s_sorted: np.ndarray
    residual_applied_shift_s_sorted: np.ndarray

    final_used_trace_mask_sorted: np.ndarray
    rejected_trace_mask_sorted: np.ndarray
    rejected_iteration_sorted: np.ndarray

    estimated_trace_time_term_delay_s_sorted: np.ndarray
    node_time_term_s: np.ndarray

    sign_convention: str
    delay_to_shift_convention: str
    final_shift_convention: str


@dataclass(frozen=True)
class _SourceTraceStoreMeta:
    raw: dict[str, object]
    n_traces: int
    n_samples: int
    dt: float
    key1_byte: int
    key2_byte: int
    original_segy_path: str
    source_sha256: str | None


def load_time_term_static_solution(
    npz_path: Path,
    *,
    expected_n_traces: int | None = None,
    expected_dt: float | None = None,
    expected_key1_byte: int | None = None,
    expected_key2_byte: int | None = None,
) -> LoadedTimeTermStaticSolution:
    """Load and validate a PR6-8 time-term static solution artifact."""
    path = Path(npz_path)
    if not path.exists():
        raise ValueError(f'time_term_static_solution.npz does not exist: {path}')
    if not path.is_file():
        raise ValueError(f'time_term_static_solution.npz is not a file: {path}')

    try:
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(_REQUIRED_SOLUTION_FIELDS.difference(data.files))
            if missing:
                raise ValueError(
                    'time_term_static_solution.npz is missing required fields: '
                    + ', '.join(missing)
                )

            schema_version = _require_scalar_int(
                data['schema_version'],
                name='schema_version',
            )
            artifact_kind = _require_scalar_str(
                data['artifact_kind'],
                name='artifact_kind',
            )
            order = _require_scalar_str(data['order'], name='order')
            job_id = _require_optional_scalar_str(data['job_id'], name='job_id')
            input_file_id = _require_optional_scalar_str(
                data['input_file_id'],
                name='input_file_id',
            )
            n_traces = _require_scalar_int(data['n_traces'], name='n_traces')
            n_samples = _require_scalar_int(data['n_samples'], name='n_samples')
            dt = _require_scalar_float(data['dt'], name='dt')
            key1_byte = _require_scalar_int(data['key1_byte'], name='key1_byte')
            key2_byte = _require_scalar_int(data['key2_byte'], name='key2_byte')

            expected_trace_shape = (n_traces,)
            applied_weathering = _require_1d_float64(
                data['applied_weathering_shift_s_sorted'],
                name='applied_weathering_shift_s_sorted',
                expected_shape=expected_trace_shape,
            )
            final_shift = _require_1d_float64(
                data['final_trace_shift_s_sorted'],
                name='final_trace_shift_s_sorted',
                expected_shape=expected_trace_shape,
            )
            datum_shift = _require_1d_float64(
                data['datum_trace_shift_s_sorted'],
                name='datum_trace_shift_s_sorted',
                expected_shape=expected_trace_shape,
            )
            residual_shift = _require_1d_float64(
                data['residual_applied_shift_s_sorted'],
                name='residual_applied_shift_s_sorted',
                expected_shape=expected_trace_shape,
            )
            final_used = _require_1d_bool(
                data['final_used_trace_mask_sorted'],
                name='final_used_trace_mask_sorted',
                expected_shape=expected_trace_shape,
            )
            rejected = _require_1d_bool(
                data['rejected_trace_mask_sorted'],
                name='rejected_trace_mask_sorted',
                expected_shape=expected_trace_shape,
            )
            rejected_iteration = _require_1d_int64(
                data['rejected_iteration_sorted'],
                name='rejected_iteration_sorted',
                expected_shape=expected_trace_shape,
            )
            estimated_delay = _require_1d_float64(
                data['estimated_trace_time_term_delay_s_sorted'],
                name='estimated_trace_time_term_delay_s_sorted',
                expected_shape=expected_trace_shape,
            )
            node_time_term = _require_1d_float64(
                data['node_time_term_s'],
                name='node_time_term_s',
            )
            sign_convention = _require_scalar_str(
                data['sign_convention'],
                name='sign_convention',
            )
            delay_to_shift_convention = _require_scalar_str(
                data['delay_to_shift_convention'],
                name='delay_to_shift_convention',
            )
            final_shift_convention = _require_scalar_str(
                data['final_shift_convention'],
                name='final_shift_convention',
            )
    except ValueError as exc:
        if 'Object arrays cannot be loaded' in str(exc):
            raise ValueError(
                'time_term_static_solution.npz contains object dtype arrays'
            ) from exc
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f'time_term_static_solution.npz could not be loaded: {path}'
        ) from exc

    if schema_version != _SCHEMA_VERSION:
        raise ValueError(
            'time_term_static_solution.npz schema_version mismatch: '
            f'expected {_SCHEMA_VERSION}, got {schema_version}'
        )
    if artifact_kind != _ARTIFACT_KIND:
        raise ValueError(
            'time_term_static_solution.npz artifact_kind mismatch: '
            f'expected {_ARTIFACT_KIND!r}, got {artifact_kind!r}'
        )
    if order != _ORDER:
        raise ValueError(
            'time_term_static_solution.npz order mismatch: '
            f'expected {_ORDER!r}, got {order!r}'
        )
    if expected_n_traces is not None and n_traces != int(expected_n_traces):
        raise ValueError(
            'time_term_static_solution.npz n_traces mismatch: '
            f'expected {int(expected_n_traces)}, got {n_traces}'
        )
    if expected_dt is not None and not np.isclose(
        dt,
        float(expected_dt),
        rtol=_DT_RTOL,
        atol=_DT_ATOL,
    ):
        raise ValueError(
            'time_term_static_solution.npz dt mismatch: '
            f'expected {float(expected_dt)}, got {dt}'
        )
    if expected_key1_byte is not None and key1_byte != int(expected_key1_byte):
        raise ValueError(
            'time_term_static_solution.npz key1_byte mismatch: '
            f'expected {int(expected_key1_byte)}, got {key1_byte}'
        )
    if expected_key2_byte is not None and key2_byte != int(expected_key2_byte):
        raise ValueError(
            'time_term_static_solution.npz key2_byte mismatch: '
            f'expected {int(expected_key2_byte)}, got {key2_byte}'
        )
    if not np.allclose(
        applied_weathering,
        -estimated_delay,
        rtol=_SHIFT_RTOL,
        atol=_SHIFT_ATOL,
    ):
        raise ValueError(
            'applied_weathering_shift_s_sorted must equal '
            '-estimated_trace_time_term_delay_s_sorted'
        )
    expected_final = datum_shift + residual_shift + applied_weathering
    if not np.allclose(
        final_shift,
        expected_final,
        rtol=_SHIFT_RTOL,
        atol=_SHIFT_ATOL,
    ):
        raise ValueError(
            'final_trace_shift_s_sorted must equal datum + residual + '
            'applied_weathering'
        )

    return LoadedTimeTermStaticSolution(
        path=path,
        schema_version=schema_version,
        artifact_kind=artifact_kind,
        order=order,
        n_traces=n_traces,
        n_samples=n_samples,
        dt=dt,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        job_id=job_id,
        input_file_id=input_file_id,
        applied_weathering_shift_s_sorted=applied_weathering,
        final_trace_shift_s_sorted=final_shift,
        datum_trace_shift_s_sorted=datum_shift,
        residual_applied_shift_s_sorted=residual_shift,
        final_used_trace_mask_sorted=final_used,
        rejected_trace_mask_sorted=rejected,
        rejected_iteration_sorted=rejected_iteration,
        estimated_trace_time_term_delay_s_sorted=estimated_delay,
        node_time_term_s=node_time_term,
        sign_convention=sign_convention,
        delay_to_shift_convention=delay_to_shift_convention,
        final_shift_convention=final_shift_convention,
    )


def select_time_term_shift_for_apply_mode(
    solution: LoadedTimeTermStaticSolution,
    *,
    mode: TimeTermTraceStoreApplyMode,
) -> tuple[str, np.ndarray]:
    if mode == 'weathering_only':
        return (
            'applied_weathering_shift_s_sorted',
            solution.applied_weathering_shift_s_sorted,
        )
    if mode == 'final_from_raw':
        return 'final_trace_shift_s_sorted', solution.final_trace_shift_s_sorted
    raise ValueError(f'unsupported time-term apply mode: {mode}')


def apply_time_term_static_correction_to_trace_store(
    *,
    source_file_id: str,
    key1_byte: int,
    key2_byte: int,
    solution_npz_path: Path,
    artifacts_dir: Path | None,
    state: AppState,
    options: TimeTermTraceStoreApplyOptions | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> TimeTermTraceStoreApplyResult:
    """Create and register a time-term-corrected TraceStore."""
    resolved_options = options or TimeTermTraceStoreApplyOptions()
    _validate_apply_options(resolved_options)
    _raise_if_cancelled(cancel_check)

    source_store_path = _resolve_source_store_path(
        state=state,
        source_file_id=source_file_id,
    )
    source_reader = get_reader(source_file_id, key1_byte, key2_byte, state=state)
    source_meta = _load_and_validate_source_meta(
        source_store_path,
        reader=source_reader,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    if not has_split_baseline_artifacts(
        source_store_path,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        source_sha256=source_meta.source_sha256,
    ):
        raise ValueError(
            'source TraceStore is incomplete: missing split raw-baseline artifacts'
        )
    lineage_warnings = _validate_source_lineage_for_weathering_only(source_meta.raw)

    solution = load_time_term_static_solution(
        Path(solution_npz_path),
        expected_n_traces=source_meta.n_traces,
        expected_dt=source_meta.dt,
        expected_key1_byte=key1_byte,
        expected_key2_byte=key2_byte,
    )
    if solution.n_samples is not None and solution.n_samples != source_meta.n_samples:
        raise ValueError(
            'time_term_static_solution.npz n_samples mismatch: '
            f'expected {source_meta.n_samples}, got {solution.n_samples}'
        )

    applied_shift_field, trace_shift_s_sorted = select_time_term_shift_for_apply_mode(
        solution,
        mode=resolved_options.mode,
    )
    trace_shift_s_sorted = _validate_trace_shift_for_apply(
        trace_shift_s_sorted,
        field_name=applied_shift_field,
        expected_n_traces=source_meta.n_traces,
        max_abs_shift_ms=resolved_options.max_abs_shift_ms,
    )
    shift_stats_ms = _shift_stats_ms(trace_shift_s_sorted)
    _raise_if_cancelled(cancel_check)

    corrected_file_id = _resolve_corrected_file_id(resolved_options.corrected_file_id)
    corrected_store_path = _corrected_store_path(
        source_store_path=source_store_path,
        corrected_file_id=corrected_file_id,
        corrected_store_name=resolved_options.corrected_store_name,
        job_id=solution.job_id,
    )
    corrected_file_json_path = (
        Path(artifacts_dir) / _CORRECTED_FILE_NAME
        if artifacts_dir is not None
        else None
    )
    derived_metadata = _build_derived_metadata(
        source_meta=source_meta.raw,
        solution=solution,
        solution_npz_path=Path(solution_npz_path),
        applied_shift_field=applied_shift_field,
        mode=resolved_options.mode,
        lineage_warnings=lineage_warnings,
    )

    build_result: TimeShiftedTraceStoreResult | None = None
    try:
        _raise_if_cancelled(cancel_check)
        build_result = build_time_shifted_trace_store(
            source_store_path=source_store_path,
            output_store_path=corrected_store_path,
            trace_shift_s_sorted=trace_shift_s_sorted,
            fill_value=resolved_options.fill_value,
            output_dtype=resolved_options.output_dtype,
            derived_metadata=derived_metadata,
            from_file_id=source_file_id,
            original_segy_path=source_meta.original_segy_path,
            header_bytes_to_materialize=(key1_byte, key2_byte),
            cancel_check=cancel_check,
        )
        _raise_if_cancelled(cancel_check)
        registered_reader = register_trace_store(
            state=state,
            file_id=corrected_file_id,
            store_dir=build_result.store_path,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            dt=build_result.dt,
            update_registry=True,
            touch_meta=True,
        )
        _verify_registered_trace_store(
            state=state,
            file_id=corrected_file_id,
            store_path=build_result.store_path,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            reader=registered_reader,
        )
        _raise_if_cancelled(cancel_check)
        if corrected_file_json_path is not None:
            write_json_atomic(
                corrected_file_json_path,
                _build_corrected_file_payload(
                    corrected_file_id=corrected_file_id,
                    build_result=build_result,
                    source_file_id=source_file_id,
                    source_store_path=source_store_path,
                    solution=solution,
                    solution_npz_path=Path(solution_npz_path),
                    mode=resolved_options.mode,
                    applied_shift_field=applied_shift_field,
                    shift_stats_ms=shift_stats_ms,
                ),
                make_parent=True,
            )
    except Exception:
        _cleanup_registration(
            state,
            file_id=corrected_file_id,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
        )
        _cleanup_store(corrected_store_path)
        if corrected_file_json_path is not None:
            _cleanup_corrected_file_json(corrected_file_json_path)
        raise

    return TimeTermTraceStoreApplyResult(
        file_id=corrected_file_id,
        store_path=build_result.store_path,
        store_name=build_result.store_path.name,
        source_file_id=source_file_id,
        source_store_path=source_store_path,
        solution_npz_path=Path(solution_npz_path),
        job_id=solution.job_id,
        mode=resolved_options.mode,
        applied_shift_field=applied_shift_field,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        dt=build_result.dt,
        n_traces=build_result.n_traces,
        n_samples=build_result.n_samples,
        shift_min_ms=shift_stats_ms['min'],
        shift_max_ms=shift_stats_ms['max'],
        shift_mean_ms=shift_stats_ms['mean'],
        shift_max_abs_ms=shift_stats_ms['max_abs'],
        corrected_file_json_path=corrected_file_json_path,
    )


def _validate_apply_options(options: TimeTermTraceStoreApplyOptions) -> None:
    if options.mode == 'final_from_raw':
        raise ValueError('final_from_raw mode is not implemented yet')
    if options.mode != 'weathering_only':
        raise ValueError(f'unsupported time-term apply mode: {options.mode}')
    if options.interpolation != 'linear':
        raise ValueError('interpolation must be "linear"')
    if options.output_dtype != 'float32':
        raise ValueError('output_dtype must be "float32"')
    if options.register_corrected_file is not True:
        raise ValueError('register_corrected_file must be true')
    _coerce_nonnegative_finite_float(
        options.max_abs_shift_ms,
        name='max_abs_shift_ms',
    )


def _raise_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise RuntimeError('time-term static TraceStore apply cancelled')


def _resolve_source_store_path(*, state: AppState, source_file_id: str) -> Path:
    if not isinstance(source_file_id, str) or not source_file_id:
        raise ValueError('source_file_id must be a non-empty string')
    try:
        source_store_path = Path(state.file_registry.get_store_path(source_file_id))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f'source_file_id is not registered with a TraceStore: {source_file_id}'
        ) from exc
    if not source_store_path.exists():
        raise ValueError(f'source TraceStore path does not exist: {source_store_path}')
    if not source_store_path.is_dir():
        raise ValueError(f'source TraceStore path is not a directory: {source_store_path}')
    return source_store_path


def _load_and_validate_source_meta(
    source_store_path: Path,
    *,
    reader: object,
    key1_byte: int,
    key2_byte: int,
) -> _SourceTraceStoreMeta:
    meta_path = source_store_path / 'meta.json'
    if not meta_path.exists():
        raise ValueError(f'source TraceStore meta.json is missing: {meta_path}')
    meta = _read_json_object(meta_path, label='source TraceStore meta.json')

    n_traces = _coerce_positive_int(meta.get('n_traces'), name='meta.n_traces')
    n_samples = _coerce_positive_int(meta.get('n_samples'), name='meta.n_samples')
    dt = _coerce_positive_finite_float(meta.get('dt'), name='meta.dt')

    key_bytes = meta.get('key_bytes')
    if not isinstance(key_bytes, dict):
        raise ValueError('source TraceStore meta.key_bytes must be an object')
    meta_key1 = _coerce_header_byte(key_bytes.get('key1'), name='meta.key_bytes.key1')
    meta_key2 = _coerce_header_byte(key_bytes.get('key2'), name='meta.key_bytes.key2')
    if meta_key1 != int(key1_byte):
        raise ValueError(
            f'source TraceStore key1_byte mismatch: expected {key1_byte}, got {meta_key1}'
        )
    if meta_key2 != int(key2_byte):
        raise ValueError(
            f'source TraceStore key2_byte mismatch: expected {key2_byte}, got {meta_key2}'
        )

    traces = getattr(reader, 'traces', None)
    if not isinstance(traces, np.ndarray) or traces.ndim != 2:
        raise ValueError('source TraceStore reader.traces must be a 2D array')
    reader_n_traces = int(traces.shape[0])
    reader_n_samples = int(traces.shape[1])
    if reader_n_traces != n_traces:
        raise ValueError(
            'source TraceStore n_traces mismatch: '
            f'meta={n_traces}, traces={reader_n_traces}'
        )
    if reader_n_samples != n_samples:
        raise ValueError(
            'source TraceStore n_samples mismatch: '
            f'meta={n_samples}, traces={reader_n_samples}'
        )

    original_segy_path = meta.get('original_segy_path')
    if not isinstance(original_segy_path, str) or not original_segy_path:
        raise ValueError('source TraceStore original_segy_path must be a non-empty string')
    source_sha256 = meta.get('source_sha256')
    if source_sha256 is not None and not isinstance(source_sha256, str):
        raise ValueError('source TraceStore source_sha256 must be a string or null')

    return _SourceTraceStoreMeta(
        raw=meta,
        n_traces=n_traces,
        n_samples=n_samples,
        dt=dt,
        key1_byte=meta_key1,
        key2_byte=meta_key2,
        original_segy_path=original_segy_path,
        source_sha256=source_sha256,
    )


def _validate_source_lineage_for_weathering_only(
    source_meta: dict[str, object],
) -> list[str]:
    derived = source_meta.get('derived')
    if not isinstance(derived, dict):
        raise ValueError(
            'weathering_only mode requires a datum/residual corrected TraceStore; '
            'use final_from_raw for raw TraceStore'
        )
    components = _component_dicts(derived)
    if _components_include(components, 'time_term_static_correction'):
        raise ValueError('source TraceStore already has time_term_static_correction')
    if _components_include(components, 'weathering_static_correction'):
        raise ValueError('source TraceStore already has weathering_static_correction')
    if source_meta.get('source_sha256') is not None:
        raise ValueError(
            'weathering_only mode requires a datum/residual corrected TraceStore; '
            'use final_from_raw for raw TraceStore'
        )
    if _components_include(components, 'residual_static_correction'):
        return []
    if _components_include(components, 'datum_static_correction'):
        return [
            'weathering_only source has datum_static_correction but no '
            'residual_static_correction'
        ]
    raise ValueError(
        'weathering_only mode requires a datum/residual corrected TraceStore; '
        'use final_from_raw for raw TraceStore'
    )


def _build_derived_metadata(
    *,
    source_meta: dict[str, object],
    solution: LoadedTimeTermStaticSolution,
    solution_npz_path: Path,
    applied_shift_field: str,
    mode: TimeTermTraceStoreApplyMode,
    lineage_warnings: list[str],
) -> dict[str, object]:
    derived = source_meta.get('derived')
    components = _component_dicts(derived if isinstance(derived, dict) else {})
    components.append(
        {
            'name': 'time_term_static_correction',
            'job_id': solution.job_id,
            'solution_artifact': solution_npz_path.name,
            'shift_field': applied_shift_field,
            'value_kind': 'applied_event_time_shift_s',
            'apply_mode': mode,
            'sign_convention': solution.sign_convention,
            'delay_to_shift_convention': solution.delay_to_shift_convention,
            'final_shift_convention': solution.final_shift_convention,
        }
    )
    metadata: dict[str, object] = {
        'applied_to': (
            'datum_residual_corrected_trace_store'
            if any(
                component.get('name') == 'residual_static_correction'
                for component in components
            )
            else 'datum_corrected_trace_store'
        ),
        'components': components,
    }
    if lineage_warnings:
        metadata['warnings'] = list(lineage_warnings)
    return metadata


def _validate_trace_shift_for_apply(
    trace_shift_s_sorted: np.ndarray,
    *,
    field_name: str,
    expected_n_traces: int,
    max_abs_shift_ms: float,
) -> np.ndarray:
    shifts = _require_1d_float64(
        trace_shift_s_sorted,
        name=field_name,
        expected_shape=(int(expected_n_traces),),
    )
    max_abs_ms = _coerce_nonnegative_finite_float(
        max_abs_shift_ms,
        name='max_abs_shift_ms',
    )
    actual_max_abs_ms = float(np.max(np.abs(shifts)) * 1000.0)
    if actual_max_abs_ms > max_abs_ms:
        raise ValueError(
            f'{field_name} exceeds max_abs_shift_ms: '
            f'{actual_max_abs_ms:.6g} > {max_abs_ms:.6g}'
        )
    return shifts


def _resolve_corrected_file_id(corrected_file_id: str | None) -> str:
    if corrected_file_id is None:
        return str(uuid4())
    if not isinstance(corrected_file_id, str) or not corrected_file_id:
        raise ValueError('corrected_file_id must be a non-empty string')
    return corrected_file_id


def _corrected_store_path(
    *,
    source_store_path: Path,
    corrected_file_id: str,
    corrected_store_name: str | None,
    job_id: str | None,
) -> Path:
    if corrected_store_name is not None:
        store_name = _validate_corrected_store_name(corrected_store_name)
    else:
        source_name = _safe_store_name_component(source_store_path.name)
        suffix_source = job_id if job_id else corrected_file_id
        suffix = _safe_store_name_component(str(suffix_source)[:8])
        store_name = f'{source_name}.statics.time_term.{suffix}'
    output_path = source_store_path.parent / store_name
    if output_path.exists() or output_path.is_symlink():
        raise ValueError(f'corrected output path already exists: {output_path}')
    return output_path


def _validate_corrected_store_name(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError('corrected_store_name must be a non-empty string')
    if Path(value).name != value:
        raise ValueError('corrected_store_name must not contain path separators')
    safe = _safe_store_name_component(value)
    if safe != value:
        raise ValueError('corrected_store_name contains unsupported characters')
    return safe


def _safe_store_name_component(value: str) -> str:
    safe = _SAFE_STORE_NAME_RE.sub('_', str(value))
    if safe in {'', '.', '..'}:
        raise ValueError('TraceStore name cannot be made filesystem-safe')
    return safe


def _verify_registered_trace_store(
    *,
    state: AppState,
    file_id: str,
    store_path: Path,
    key1_byte: int,
    key2_byte: int,
    reader: object,
) -> None:
    registered_path = Path(state.file_registry.get_store_path(file_id))
    if registered_path.resolve() != store_path.resolve():
        raise RuntimeError('registered corrected TraceStore path mismatch')
    cache_key = trace_store_cache_key(file_id, key1_byte, key2_byte)
    with state.lock:
        if cache_key not in state.cached_readers:
            raise RuntimeError('registered corrected TraceStore reader is missing')
    key1_values = np.asarray(reader.get_key1_values())
    if key1_values.size == 0:
        raise RuntimeError('registered corrected TraceStore has no key1 values')
    reader.get_section(int(key1_values[0]))


def _build_corrected_file_payload(
    *,
    corrected_file_id: str,
    build_result: TimeShiftedTraceStoreResult,
    source_file_id: str,
    source_store_path: Path,
    solution: LoadedTimeTermStaticSolution,
    solution_npz_path: Path,
    mode: TimeTermTraceStoreApplyMode,
    applied_shift_field: str,
    shift_stats_ms: dict[str, float],
) -> dict[str, object]:
    return {
        'schema_version': 1,
        'artifact_kind': 'corrected_file',
        'file_id': corrected_file_id,
        'store_path': str(build_result.store_path),
        'store_name': build_result.store_path.name,
        'derived_from_file_id': source_file_id,
        'derived_from_store_path': str(source_store_path),
        'derived_by': 'time_term_static_correction',
        'job_id': solution.job_id,
        'solution_artifact': solution_npz_path.name,
        'apply_mode': mode,
        'applied_shift_field': applied_shift_field,
        'key1_byte': int(build_result.key1_byte),
        'key2_byte': int(build_result.key2_byte),
        'dt': float(build_result.dt),
        'n_traces': int(build_result.n_traces),
        'n_samples': int(build_result.n_samples),
        'shift_ms': {
            'min': shift_stats_ms['min'],
            'max': shift_stats_ms['max'],
            'mean': shift_stats_ms['mean'],
            'max_abs': shift_stats_ms['max_abs'],
        },
        'sign_convention': _BUILDER_SIGN_CONVENTION,
        'delay_to_shift_convention': solution.delay_to_shift_convention,
        'final_shift_convention': solution.final_shift_convention,
    }


def _shift_stats_ms(shift_s: np.ndarray) -> dict[str, float]:
    shift_ms = np.asarray(shift_s, dtype=np.float64) * 1000.0
    return {
        'min': float(np.min(shift_ms)),
        'max': float(np.max(shift_ms)),
        'mean': float(np.mean(shift_ms)),
        'max_abs': float(np.max(np.abs(shift_ms))),
    }


def _cleanup_registration(
    state: AppState,
    *,
    file_id: str,
    key1_byte: int,
    key2_byte: int,
) -> None:
    lock = getattr(state, 'lock', None)
    context = lock if lock is not None else nullcontext()
    with context:
        state.file_registry.pop(file_id, None)
        state.cached_readers.pop(trace_store_cache_key(file_id, key1_byte, key2_byte), None)


def _cleanup_store(output_path: Path) -> None:
    for tmp_path in output_path.parent.glob(f'{output_path.name}.tmp-*'):
        if tmp_path.is_dir():
            shutil.rmtree(tmp_path, ignore_errors=True)
    if output_path.exists():
        shutil.rmtree(output_path, ignore_errors=True)


def _cleanup_corrected_file_json(path: Path) -> None:
    path.with_name(f'{path.name}.tmp').unlink(missing_ok=True)
    path.unlink(missing_ok=True)


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'{label} is invalid: {path}') from exc
    if not isinstance(payload, dict):
        raise ValueError(f'{label} must be an object')
    return payload


def _component_dicts(derived: dict[str, object]) -> list[dict[str, object]]:
    components = derived.get('components')
    if components is None:
        return []
    if not isinstance(components, list):
        raise ValueError('source TraceStore derived.components must be a list')
    output: list[dict[str, object]] = []
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise ValueError(
                f'source TraceStore derived.components[{index}] must be an object'
            )
        output.append(dict(component))
    return output


def _components_include(components: list[dict[str, object]], name: str) -> bool:
    return any(component.get('name') == name for component in components)


def _require_1d_float64(
    value: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(value)
    _reject_object_dtype(arr, name=name)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be 1-dimensional')
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f'{name} must be a float array')
    out = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f'{name} must contain only finite values')
    return np.ascontiguousarray(out, dtype=np.float64)


def _require_1d_bool(
    value: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(value)
    _reject_object_dtype(arr, name=name)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be 1-dimensional')
    if arr.shape != expected_shape:
        raise ValueError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if not np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{name} must have bool dtype')
    return np.ascontiguousarray(arr, dtype=bool)


def _require_1d_int64(
    value: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(value)
    _reject_object_dtype(arr, name=name)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be 1-dimensional')
    if arr.shape != expected_shape:
        raise ValueError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(
        arr.dtype,
        np.integer,
    ):
        raise ValueError(f'{name} must have integer dtype')
    return np.ascontiguousarray(arr, dtype=np.int64)


def _require_scalar_int(value: np.ndarray, *, name: str) -> int:
    arr = np.asarray(value)
    _reject_object_dtype(arr, name=name)
    if arr.shape != ():
        raise ValueError(f'{name} must be a scalar')
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(
        arr.dtype,
        np.integer,
    ):
        raise ValueError(f'{name} must be an integer scalar')
    return _coerce_positive_int(arr.item(), name=name)


def _require_scalar_float(value: np.ndarray, *, name: str) -> float:
    arr = np.asarray(value)
    _reject_object_dtype(arr, name=name)
    if arr.shape != ():
        raise ValueError(f'{name} must be a scalar')
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(
        arr.dtype,
        np.number,
    ):
        raise ValueError(f'{name} must be a numeric scalar')
    return _coerce_positive_finite_float(arr.item(), name=name)


def _require_scalar_str(value: np.ndarray, *, name: str) -> str:
    text = _require_optional_scalar_str(value, name=name)
    if text is None:
        raise ValueError(f'{name} must be a non-empty string scalar')
    return text


def _require_optional_scalar_str(value: np.ndarray, *, name: str) -> str | None:
    arr = np.asarray(value)
    _reject_object_dtype(arr, name=name)
    if arr.shape != ():
        raise ValueError(f'{name} must be a scalar')
    if arr.dtype.kind not in {'U', 'S'}:
        raise ValueError(f'{name} must be a string scalar')
    item = arr.item()
    text = item.decode('utf-8') if isinstance(item, bytes) else str(item)
    return text if text else None


def _reject_object_dtype(arr: np.ndarray, *, name: str) -> None:
    if arr.dtype == object:
        raise ValueError(f'{name} must not have object dtype')


def _coerce_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a positive integer')
    if not isinstance(value, int | np.integer):
        raise ValueError(f'{name} must be a positive integer')
    out = int(value)
    if out <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return out


def _coerce_header_byte(value: Any, *, name: str) -> int:
    out = _coerce_positive_int(value, name=name)
    if out > 240:
        raise ValueError(f'{name} must be between 1 and 240')
    return out


def _coerce_positive_finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be finite and greater than 0')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite and greater than 0') from exc
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f'{name} must be finite and greater than 0')
    return out


def _coerce_nonnegative_finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be finite and non-negative')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite and non-negative') from exc
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f'{name} must be finite and non-negative')
    return out


__all__ = [
    'LoadedTimeTermStaticSolution',
    'TimeTermTraceStoreApplyMode',
    'TimeTermTraceStoreApplyOptions',
    'TimeTermTraceStoreApplyResult',
    'apply_time_term_static_correction_to_trace_store',
    'load_time_term_static_solution',
    'select_time_term_shift_for_apply_mode',
]
