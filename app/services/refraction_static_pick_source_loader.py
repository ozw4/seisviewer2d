"""Dependency-light refraction first-break pick NPZ loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.contracts.statics.refraction.apply import RefractionStaticApplyRequest
from app.services.common.array_validation import (
    is_real_numeric_dtype as _is_real_numeric_dtype,
)
from app.services.trace_store_index_validation import validate_sorted_to_original

PICK_TIME_KEYS: tuple[str, ...] = (
    'pick_time_s',
    'picks_time_s',
    'predicted_picks_time_s',
    'first_break_time_s',
)

REFRACTION_PICK_ORDER_TRACE_STORE_SORTED = 'trace_store_sorted'
REFRACTION_PICK_ORDER_ORIGINAL_TRACE = 'original_trace_order'


@dataclass(frozen=True)
class LoadedRefractionPickSource:
    picks_time_s_sorted: np.ndarray
    sorted_trace_index: np.ndarray
    source_kind: str
    metadata: dict[str, Any]


def load_refraction_pick_source_from_npz_path(
    *,
    npz_path: Path,
    request: RefractionStaticApplyRequest,
    n_traces: int,
    n_samples: int,
    dt_s: float,
    sorted_trace_index: np.ndarray,
) -> LoadedRefractionPickSource:
    """Load uploaded first-break picks from a concrete NPZ path."""
    if request.pick_source.kind != 'uploaded_npz':
        raise ValueError('pick_source.kind must be uploaded_npz for direct NPZ loading')
    loaded = load_npz_refraction_pick_source_from_path(
        npz_path,
        n_traces=n_traces,
        n_samples=n_samples,
        dt_s=dt_s,
        sorted_trace_index=sorted_trace_index,
        source_kind='uploaded_npz',
    )
    metadata = dict(loaded.metadata)
    metadata['loaded_from'] = 'uploaded_npz'
    return LoadedRefractionPickSource(
        picks_time_s_sorted=loaded.picks_time_s_sorted,
        sorted_trace_index=loaded.sorted_trace_index,
        source_kind='uploaded_npz',
        metadata=metadata,
    )


def load_npz_refraction_pick_source_from_path(
    npz_path: Path,
    *,
    n_traces: int,
    n_samples: int,
    dt_s: float,
    sorted_trace_index: np.ndarray,
    source_kind: str,
    allow_invalid_pick_values: bool = False,
) -> LoadedRefractionPickSource:
    """Load a refraction pick NPZ and normalize pick times to sorted order."""
    path = Path(npz_path)
    sorted_to_original = validate_sorted_to_original(
        np.asarray(sorted_trace_index),
        expected_n_traces=n_traces,
        role='sorted_trace_index',
    )
    try:
        npz_file = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        msg = f'Could not read npz pick source: {path}'
        raise ValueError(msg) from exc

    try:
        with npz_file as npz:
            key = _select_pick_key(npz.files)
            metadata = _validate_pick_npz_metadata(
                npz,
                n_traces=n_traces,
                n_samples=n_samples,
                dt_s=dt_s,
                path=path,
                key=key,
            )
            picks = _coerce_pick_array(
                np.asarray(npz[key]),
                allow_invalid_pick_values=allow_invalid_pick_values,
            )
            metadata['pick_shape'] = tuple(int(dim) for dim in picks.shape)
            if picks.shape != (n_traces,):
                msg = (
                    'pick array length mismatch: '
                    f'expected {(n_traces,)}, got {picks.shape}'
                )
                raise ValueError(msg)

            artifact_order = _read_npz_order(npz)
            if 'sorted_to_original' in npz.files:
                npz_sorted_to_original = validate_sorted_to_original(
                    np.asarray(npz['sorted_to_original']),
                    expected_n_traces=n_traces,
                    role='npz',
                )
                if not np.array_equal(npz_sorted_to_original, sorted_to_original):
                    raise ValueError('sorted_to_original mismatch')
                metadata['has_sorted_to_original'] = True
            else:
                metadata['has_sorted_to_original'] = False
            metadata.update(_optional_pick_npz_array_metadata(npz, n_traces=n_traces))
    except ValueError as exc:
        msg = f'Invalid npz pick source {path}: {exc}'
        raise ValueError(msg) from exc

    if artifact_order == REFRACTION_PICK_ORDER_TRACE_STORE_SORTED:
        picks_sorted = picks
    elif artifact_order == REFRACTION_PICK_ORDER_ORIGINAL_TRACE:
        picks_sorted = np.ascontiguousarray(picks[sorted_to_original])
    else:
        raise ValueError(f'unsupported pick artifact order: {artifact_order}')

    return LoadedRefractionPickSource(
        picks_time_s_sorted=picks_sorted,
        sorted_trace_index=sorted_to_original,
        source_kind=source_kind,
        metadata=metadata,
    )


def _select_pick_key(keys: list[str]) -> str:
    for key in PICK_TIME_KEYS:
        if key in keys:
            return key
    accepted = ', '.join(PICK_TIME_KEYS)
    raise ValueError(f'unsupported pick artifact key; accepted keys: {accepted}')


def _validate_pick_npz_metadata(
    npz: np.lib.npyio.NpzFile,
    *,
    n_traces: int,
    n_samples: int,
    dt_s: float,
    path: Path,
    key: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        'npz_path': str(path),
        'npz_keys': tuple(npz.files),
        'accepted_pick_key': key,
        'accepted_key_priority': PICK_TIME_KEYS,
    }
    if 'n_traces' in npz.files:
        value = _read_int_scalar_from_npz(npz, 'n_traces')
        metadata['n_traces'] = value
        if value != n_traces:
            raise ValueError(f'n_traces mismatch: expected {n_traces}, got {value}')
    if 'n_samples' in npz.files:
        value = _read_int_scalar_from_npz(npz, 'n_samples')
        metadata['n_samples'] = value
        if value != n_samples:
            raise ValueError(f'n_samples mismatch: expected {n_samples}, got {value}')
    if 'dt' in npz.files:
        value = _read_float_scalar_from_npz(npz, 'dt')
        metadata['dt'] = value
        if not np.isfinite(value) or abs(value - dt_s) > 1.0e-9:
            raise ValueError(f'dt mismatch: expected {dt_s}, got {value}')
    return metadata


def _optional_pick_npz_array_metadata(
    npz: np.lib.npyio.NpzFile,
    *,
    n_traces: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if 'valid_pick_mask' in npz.files:
        mask = np.asarray(npz['valid_pick_mask'])
        if mask.shape != (n_traces,):
            raise ValueError(
                'valid_pick_mask shape mismatch: '
                f'expected {(n_traces,)}, got {mask.shape}'
            )
        if not np.issubdtype(mask.dtype, np.bool_):
            raise ValueError('valid_pick_mask must have a bool dtype')
        metadata['has_valid_pick_mask'] = True
        metadata['n_valid_pick_mask'] = int(np.count_nonzero(mask))
    else:
        metadata['has_valid_pick_mask'] = False

    if 'confidence' in npz.files:
        confidence = np.asarray(npz['confidence'])
        if confidence.ndim != 0 and confidence.shape != (n_traces,):
            raise ValueError(
                'confidence shape mismatch: '
                f'expected scalar or {(n_traces,)}, got {confidence.shape}'
            )
        if np.issubdtype(confidence.dtype, np.bool_) or not _is_real_numeric_dtype(
            confidence.dtype
        ):
            raise ValueError('confidence must have a real numeric dtype')
        metadata['has_confidence'] = True
        metadata['confidence_shape'] = tuple(int(dim) for dim in confidence.shape)
    else:
        metadata['has_confidence'] = False
    return metadata


def _read_npz_order(npz: np.lib.npyio.NpzFile) -> str:
    for key in ('order', 'trace_order'):
        if key not in npz.files:
            continue
        value = _read_string_scalar_from_npz(npz, key)
        if value in {REFRACTION_PICK_ORDER_TRACE_STORE_SORTED, 'sorted'}:
            return REFRACTION_PICK_ORDER_TRACE_STORE_SORTED
        if value in {REFRACTION_PICK_ORDER_ORIGINAL_TRACE, 'original'}:
            return REFRACTION_PICK_ORDER_ORIGINAL_TRACE
        raise ValueError(f'unsupported pick artifact {key}: {value}')
    return REFRACTION_PICK_ORDER_ORIGINAL_TRACE


def _coerce_pick_array(
    values: np.ndarray,
    *,
    allow_invalid_pick_values: bool = False,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError('pick_time_s_sorted must be 1D')
    if np.issubdtype(arr.dtype, np.bool_) or not _is_real_numeric_dtype(arr.dtype):
        raise ValueError('pick_time_s_sorted must have a real numeric dtype')
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if not allow_invalid_pick_values and not np.all(np.isfinite(out)):
        raise ValueError('pick_time_s_sorted must contain only finite values')
    return out


def _read_int_scalar_from_npz(npz: np.lib.npyio.NpzFile, key: str) -> int:
    arr = np.asarray(npz[key])
    if arr.size != 1 or np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{key} must be an integer scalar')
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f'{key} must be an integer scalar')
    return int(arr.reshape(-1)[0])


def _read_float_scalar_from_npz(npz: np.lib.npyio.NpzFile, key: str) -> float:
    arr = np.asarray(npz[key])
    if arr.size != 1:
        raise ValueError(f'{key} must be a numeric scalar')
    if np.issubdtype(arr.dtype, np.bool_) or not _is_real_numeric_dtype(arr.dtype):
        raise ValueError(f'{key} must be a numeric scalar')
    return float(arr.reshape(-1)[0])


def _read_string_scalar_from_npz(npz: np.lib.npyio.NpzFile, key: str) -> str:
    arr = np.asarray(npz[key])
    if arr.size != 1:
        raise ValueError(f'{key} must be a string scalar')
    value = arr.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


__all__ = [
    'LoadedRefractionPickSource',
    'PICK_TIME_KEYS',
    'REFRACTION_PICK_ORDER_ORIGINAL_TRACE',
    'REFRACTION_PICK_ORDER_TRACE_STORE_SORTED',
    'load_npz_refraction_pick_source_from_path',
    'load_refraction_pick_source_from_npz_path',
]
