"""Apply residual static solution artifacts to datum-corrected TraceStores."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Literal
from uuid import uuid4

import numpy as np

from app.services.common.artifact_io import write_json_atomic
from app.services.common.array_validation import (
    coerce_1d_real_numeric_float64,
    coerce_header_byte as _coerce_header_byte,
    coerce_nonnegative_finite_float as _coerce_nonnegative_finite_float,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
)
from app.services.corrected_trace_store import build_time_shifted_trace_store
from app.services.trace_store_registration import (
    register_trace_store,
    trace_store_cache_key,
)

SOLUTION_NPZ_NAME = 'residual_static_solution.npz'
_CORRECTED_FILE_NAME = 'corrected_file.json'
_SAFE_STORE_NAME_RE = re.compile(r'[^A-Za-z0-9_.-]+')
_SIGN_CONVENTION = (
    'estimated_trace_delay_s=source_delay_s+receiver_delay_s; '
    'applied_residual_shift_s=-estimated_trace_delay_s; '
    'corrected(t)=raw(t-shift_s)'
)
_BUILDER_SIGN_CONVENTION = (
    'corrected(t)=raw(t-shift_s); positive_shift_delays_events'
)
_REQUIRED_SOLUTION_FIELDS = {
    'applied_residual_shift_s_sorted',
    'estimated_trace_delay_s_sorted',
    'dt',
    'n_traces',
    'key1_byte',
    'key2_byte',
    'sign_convention',
}


@dataclass(frozen=True)
class ResidualStaticTraceStoreApplyOptions:
    interpolation: Literal['linear'] = 'linear'
    fill_value: float = 0.0
    max_abs_shift_ms: float = 250.0
    output_dtype: Literal['float32'] = 'float32'
    register_corrected_file: bool = True


@dataclass(frozen=True)
class ResidualStaticSolutionForApply:
    applied_residual_shift_s_sorted: np.ndarray
    estimated_trace_delay_s_sorted: np.ndarray
    dt: float
    n_traces: int
    key1_byte: int
    key2_byte: int
    sign_convention: str


@dataclass(frozen=True)
class ResidualStaticCorrectedStoreResult:
    file_id: str
    store_path: Path
    store_name: str
    corrected_file_json_path: Path
    derived_from_file_id: str
    derived_from_store_path: Path
    job_id: str
    key1_byte: int
    key2_byte: int
    dt: float


@dataclass(frozen=True)
class _SourceTraceStoreMeta:
    raw: dict[str, object]
    n_traces: int
    n_samples: int
    dt: float
    key1_byte: int
    key2_byte: int
    original_segy_path: str


def load_residual_static_solution_for_apply(
    solution_npz_path: Path,
    *,
    expected_n_traces: int,
    expected_dt: float,
    expected_key1_byte: int,
    expected_key2_byte: int,
) -> ResidualStaticSolutionForApply:
    """Load the residual static solution fields needed for TraceStore apply."""
    path = Path(solution_npz_path)
    if not path.exists():
        raise ValueError(f'residual_static_solution.npz does not exist: {path}')
    if not path.is_file():
        raise ValueError(f'residual_static_solution.npz is not a file: {path}')
    expected_n = _coerce_positive_int(
        expected_n_traces,
        name='expected_n_traces',
    )
    expected_dt_value = _coerce_positive_finite_float(
        expected_dt,
        name='expected_dt',
    )

    try:
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(_REQUIRED_SOLUTION_FIELDS.difference(data.files))
            if missing:
                raise ValueError(
                    'residual_static_solution.npz is missing required fields: '
                    + ', '.join(missing)
                )
            applied = _require_1d_float64(
                data['applied_residual_shift_s_sorted'],
                name='applied_residual_shift_s_sorted',
                expected_shape=(expected_n,),
            )
            estimated = _require_1d_float64(
                data['estimated_trace_delay_s_sorted'],
                name='estimated_trace_delay_s_sorted',
                expected_shape=(expected_n,),
            )
            dt = _require_scalar_float(data['dt'], name='dt')
            n_traces = _require_scalar_int(data['n_traces'], name='n_traces')
            key1_byte = _require_scalar_int(data['key1_byte'], name='key1_byte')
            key2_byte = _require_scalar_int(data['key2_byte'], name='key2_byte')
            sign_convention = _require_scalar_str(
                data['sign_convention'],
                name='sign_convention',
            )
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f'residual_static_solution.npz could not be loaded: {path}'
        ) from exc

    if n_traces != expected_n:
        raise ValueError(
            'residual_static_solution.npz n_traces mismatch: '
            f'expected {expected_n}, got {n_traces}'
        )
    if not np.isclose(dt, expected_dt_value, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError(
            'residual_static_solution.npz dt mismatch: '
            f'expected {expected_dt_value}, got {dt}'
        )
    if key1_byte != int(expected_key1_byte):
        raise ValueError(
            'residual_static_solution.npz key1_byte mismatch: '
            f'expected {int(expected_key1_byte)}, got {key1_byte}'
        )
    if key2_byte != int(expected_key2_byte):
        raise ValueError(
            'residual_static_solution.npz key2_byte mismatch: '
            f'expected {int(expected_key2_byte)}, got {key2_byte}'
        )
    if not sign_convention:
        raise ValueError('residual_static_solution.npz sign_convention is empty')
    _validate_applied_shift_sign(applied, estimated)

    return ResidualStaticSolutionForApply(
        applied_residual_shift_s_sorted=applied,
        estimated_trace_delay_s_sorted=estimated,
        dt=dt,
        n_traces=n_traces,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        sign_convention=sign_convention,
    )


def validate_residual_static_shift_for_apply(
    solution: ResidualStaticSolutionForApply,
    *,
    max_abs_shift_ms: float,
) -> None:
    """Validate service-level residual shift limits without clipping."""
    max_abs_ms = _coerce_nonnegative_finite_float(
        max_abs_shift_ms,
        name='max_abs_shift_ms',
    )
    applied = _require_1d_float64(
        solution.applied_residual_shift_s_sorted,
        name='applied_residual_shift_s_sorted',
        expected_shape=(solution.n_traces,),
    )
    estimated = _require_1d_float64(
        solution.estimated_trace_delay_s_sorted,
        name='estimated_trace_delay_s_sorted',
        expected_shape=(solution.n_traces,),
    )
    _validate_applied_shift_sign(applied, estimated)
    max_abs_shift_ms_actual = float(np.max(np.abs(applied)) * 1000.0)
    if max_abs_shift_ms_actual > max_abs_ms:
        raise ValueError(
            'applied_residual_shift_s_sorted exceeds max_abs_shift_ms: '
            f'{max_abs_shift_ms_actual:.6g} > {max_abs_ms:.6g}'
        )


def build_residual_static_derived_metadata(
    *,
    source_meta: dict[str, object],
    source_file_id: str,
    source_store_path: Path,
    job_id: str,
    residual_solution_artifact_name: str = SOLUTION_NPZ_NAME,
) -> dict[str, object]:
    """Build derived metadata extras for the residual corrected TraceStore."""
    derived = _require_derived_dict(source_meta)
    components = _component_dicts(derived)
    if not _components_include(components, 'datum_static_correction'):
        if not _has_legacy_datum_lineage(derived):
            raise ValueError('source TraceStore is missing datum static lineage')
        components.append(_datum_component_from_legacy_derived(derived))

    components.append(
        {
            'name': 'residual_static_correction',
            'job_id': str(job_id),
            'solution_artifact': str(residual_solution_artifact_name),
            'estimated_delay_field': 'estimated_trace_delay_s_sorted',
            'shift_field': 'applied_residual_shift_s_sorted',
            'value_kind': 'applied_event_time_shift_s',
            'sign_convention': (
                'applied_residual_shift_s=-estimated_trace_delay_s'
            ),
        }
    )
    return {
        'applied_to': 'datum_corrected_trace_store',
        'components': components,
    }


def apply_residual_static_correction_to_trace_store(
    *,
    source_file_id: str,
    source_store_path: Path,
    key1_byte: int,
    key2_byte: int,
    residual_solution_npz_path: Path,
    artifacts_dir: Path,
    job_id: str,
    state: object,
    options: ResidualStaticTraceStoreApplyOptions,
    corrected_file_id: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> ResidualStaticCorrectedStoreResult:
    """Create, register, and describe a residual-corrected TraceStore."""
    _validate_apply_options(options)
    _notify_progress(progress_callback, 0.0, 'validating_residual_static_trace_shift')
    _raise_if_cancelled(cancel_check)
    source_path = Path(source_store_path)
    _validate_registry_source_path(
        state=state,
        source_file_id=source_file_id,
        source_store_path=source_path,
    )
    source_meta = _load_and_validate_source_meta(
        source_path,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    solution = load_residual_static_solution_for_apply(
        Path(residual_solution_npz_path),
        expected_n_traces=source_meta.n_traces,
        expected_dt=source_meta.dt,
        expected_key1_byte=key1_byte,
        expected_key2_byte=key2_byte,
    )
    validate_residual_static_shift_for_apply(
        solution,
        max_abs_shift_ms=options.max_abs_shift_ms,
    )
    _raise_if_cancelled(cancel_check)

    resolved_file_id = str(corrected_file_id or uuid4())
    output_store_path = _corrected_store_path(
        source_store_path=source_path,
        job_id=job_id,
    )
    derived_metadata = build_residual_static_derived_metadata(
        source_meta=source_meta.raw,
        source_file_id=source_file_id,
        source_store_path=source_path,
        job_id=job_id,
        residual_solution_artifact_name=Path(residual_solution_npz_path).name,
    )
    corrected_file_json_path = Path(artifacts_dir) / _CORRECTED_FILE_NAME

    def builder_progress_callback(progress: float, message: str) -> None:
        mapped = 0.10 + (0.75 * max(0.0, min(1.0, float(progress))))
        _notify_progress(progress_callback, mapped, message)

    build_result = None
    try:
        _notify_progress(
            progress_callback,
            0.10,
            'building_residual_corrected_trace_store',
        )
        _raise_if_cancelled(cancel_check)
        build_result = build_time_shifted_trace_store(
            source_store_path=source_path,
            output_store_path=output_store_path,
            trace_shift_s_sorted=solution.applied_residual_shift_s_sorted,
            fill_value=options.fill_value,
            output_dtype=options.output_dtype,
            derived_metadata=derived_metadata,
            from_file_id=source_file_id,
            original_segy_path=source_meta.original_segy_path,
            progress_callback=(
                builder_progress_callback
                if progress_callback is not None
                else None
            ),
            cancel_check=cancel_check,
        )
        _raise_if_cancelled(cancel_check)
        _notify_progress(
            progress_callback,
            0.90,
            'registering_residual_corrected_trace_store',
        )
        _raise_if_cancelled(cancel_check)
        register_trace_store(
            state=state,
            file_id=resolved_file_id,
            store_dir=build_result.store_path,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            dt=build_result.dt,
            update_registry=True,
            touch_meta=True,
        )
        _raise_if_cancelled(cancel_check)
        _notify_progress(progress_callback, 0.97, 'writing_corrected_file_manifest')
        write_json_atomic(
            corrected_file_json_path,
            _build_corrected_file_payload(
                corrected_file_id=resolved_file_id,
                store_path=build_result.store_path,
                source_file_id=source_file_id,
                source_store_path=source_path,
                job_id=job_id,
                key1_byte=key1_byte,
                key2_byte=key2_byte,
                dt=build_result.dt,
            ),
            make_parent=True,
        )
    except Exception:
        _cleanup_registration(
            state,
            file_id=resolved_file_id,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
        )
        _cleanup_store(output_store_path)
        raise

    return ResidualStaticCorrectedStoreResult(
        file_id=resolved_file_id,
        store_path=build_result.store_path,
        store_name=build_result.store_path.name,
        corrected_file_json_path=corrected_file_json_path,
        derived_from_file_id=source_file_id,
        derived_from_store_path=source_path,
        job_id=job_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        dt=build_result.dt,
    )


def _notify_progress(
    progress_callback: Callable[[float, str], None] | None,
    progress: float,
    message: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback(float(progress), message)


def _raise_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise RuntimeError('residual static TraceStore apply cancelled')


def _validate_apply_options(options: ResidualStaticTraceStoreApplyOptions) -> None:
    if options.interpolation != 'linear':
        raise ValueError('interpolation must be "linear"')
    if options.output_dtype != 'float32':
        raise ValueError('output_dtype must be "float32"')
    if options.register_corrected_file is not True:
        raise ValueError('register_corrected_file must be true')


def _validate_registry_source_path(
    *,
    state: object,
    source_file_id: str,
    source_store_path: Path,
) -> None:
    if not source_store_path.exists():
        raise ValueError(f'source_store_path does not exist: {source_store_path}')
    if not source_store_path.is_dir():
        raise ValueError(f'source_store_path must be a directory: {source_store_path}')
    try:
        registry_store_path = Path(state.file_registry.get_store_path(source_file_id))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f'source_file_id is not registered with a TraceStore: {source_file_id}'
        ) from exc
    if registry_store_path.resolve() != source_store_path.resolve():
        raise ValueError(
            'source_store_path does not match file registry for source_file_id'
        )


def _load_and_validate_source_meta(
    source_store_path: Path,
    *,
    key1_byte: int,
    key2_byte: int,
) -> _SourceTraceStoreMeta:
    meta_path = source_store_path / 'meta.json'
    if not meta_path.exists():
        raise ValueError(f'source TraceStore meta.json is missing: {meta_path}')
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'source TraceStore meta.json is invalid: {meta_path}') from exc
    if not isinstance(meta, dict):
        raise ValueError('source TraceStore meta.json must be an object')

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

    original_segy_path = meta.get('original_segy_path')
    if not isinstance(original_segy_path, str) or not original_segy_path:
        raise ValueError('source TraceStore original_segy_path must be a non-empty string')
    _validate_source_lineage(meta)

    return _SourceTraceStoreMeta(
        raw=meta,
        n_traces=n_traces,
        n_samples=n_samples,
        dt=dt,
        key1_byte=meta_key1,
        key2_byte=meta_key2,
        original_segy_path=original_segy_path,
    )


def _validate_source_lineage(source_meta: dict[str, object]) -> None:
    derived = _require_derived_dict(source_meta)
    components = _component_dicts(derived)
    if _components_include(components, 'residual_static_correction'):
        raise ValueError('source TraceStore already has residual_static_correction')
    if _components_include(components, 'datum_static_correction'):
        return
    if _has_legacy_datum_lineage(derived):
        return
    raise ValueError('source TraceStore is missing datum static lineage')


def _require_derived_dict(source_meta: dict[str, object]) -> dict[str, object]:
    derived = source_meta.get('derived')
    if not isinstance(derived, dict):
        raise ValueError('source TraceStore is missing derived metadata')
    return dict(derived)


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


def _has_legacy_datum_lineage(derived: dict[str, object]) -> bool:
    return (
        derived.get('statics_kind') == 'datum'
        or derived.get('derived_by') == 'datum_static_correction'
    )


def _datum_component_from_legacy_derived(
    derived: dict[str, object],
) -> dict[str, object]:
    component: dict[str, object] = {
        'name': 'datum_static_correction',
        'solution_artifact': str(
            derived.get('solution_artifact') or 'datum_static_solution.npz'
        ),
        'shift_field': str(derived.get('shift_field') or 'trace_shift_s_sorted'),
        'value_kind': str(
            derived.get('value_kind') or 'applied_event_time_shift_s'
        ),
    }
    job_id = derived.get('job_id')
    if job_id is not None:
        component['job_id'] = str(job_id)
    return component


def _corrected_store_path(*, source_store_path: Path, job_id: str) -> Path:
    source_name = _safe_store_name_component(source_store_path.name)
    job_prefix = _safe_store_name_component(str(job_id)[:8])
    store_name = f'{source_name}.statics.residual.{job_prefix}'
    output_path = source_store_path.parent / store_name
    if output_path.exists() or output_path.is_symlink():
        raise ValueError(f'corrected output path already exists: {output_path}')
    return output_path


def _safe_store_name_component(value: str) -> str:
    safe = _SAFE_STORE_NAME_RE.sub('_', str(value))
    if safe in {'', '.', '..'}:
        raise ValueError('TraceStore name cannot be made filesystem-safe')
    return safe


def _build_corrected_file_payload(
    *,
    corrected_file_id: str,
    store_path: Path,
    source_file_id: str,
    source_store_path: Path,
    job_id: str,
    key1_byte: int,
    key2_byte: int,
    dt: float,
) -> dict[str, object]:
    return {
        'schema_version': 1,
        'artifact_kind': 'corrected_file',
        'file_id': corrected_file_id,
        'store_path': str(store_path),
        'store_name': store_path.name,
        'derived_from_file_id': source_file_id,
        'derived_from_store_path': str(source_store_path),
        'derived_by': 'residual_static_correction',
        'applied_to': 'datum_corrected_trace_store',
        'job_id': job_id,
        'key1_byte': int(key1_byte),
        'key2_byte': int(key2_byte),
        'dt': float(dt),
        'solution_artifact': SOLUTION_NPZ_NAME,
        'shift_field': 'applied_residual_shift_s_sorted',
        'estimated_delay_field': 'estimated_trace_delay_s_sorted',
        'sign_convention': _SIGN_CONVENTION,
    }


def _cleanup_registration(
    state: object,
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


def _validate_applied_shift_sign(applied: np.ndarray, estimated: np.ndarray) -> None:
    if not np.allclose(applied, -estimated, rtol=1.0e-7, atol=1.0e-9):
        raise ValueError(
            'applied_residual_shift_s_sorted must equal '
            '-estimated_trace_delay_s_sorted'
        )


def _require_1d_float64(
    value: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f'{name} must be a float array')
    return coerce_1d_real_numeric_float64(
        arr,
        name=name,
        expected_shape=expected_shape,
        require_finite=True,
    )


def _require_scalar_int(value: np.ndarray, *, name: str) -> int:
    arr = np.asarray(value)
    if arr.shape != ():
        raise ValueError(f'{name} must be a scalar')
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f'{name} must be an integer scalar')
    return _coerce_positive_int(arr.item(), name=name)


def _require_scalar_float(value: np.ndarray, *, name: str) -> float:
    arr = np.asarray(value)
    if arr.shape != ():
        raise ValueError(f'{name} must be a scalar')
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f'{name} must be a numeric scalar')
    return _coerce_positive_finite_float(arr.item(), name=name)


def _require_scalar_str(value: np.ndarray, *, name: str) -> str:
    arr = np.asarray(value)
    if arr.shape != ():
        raise ValueError(f'{name} must be a scalar')
    if arr.dtype.kind not in {'U', 'S'}:
        raise ValueError(f'{name} must be a string scalar')
    item = arr.item()
    if isinstance(item, bytes):
        return item.decode('utf-8')
    return str(item)


__all__ = [
    'ResidualStaticCorrectedStoreResult',
    'ResidualStaticSolutionForApply',
    'ResidualStaticTraceStoreApplyOptions',
    'apply_residual_static_correction_to_trace_store',
    'build_residual_static_derived_metadata',
    'load_residual_static_solution_for_apply',
    'validate_residual_static_shift_for_apply',
]
