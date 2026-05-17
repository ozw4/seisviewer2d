"""Preflight diagnostics for refraction static input assembly."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_pick_source_loader import PICK_TIME_KEYS
from app.services.trace_store_index_validation import validate_sorted_to_original

REFRACTION_STATIC_PREFLIGHT_QC_JSON_NAME = 'refraction_static_preflight_qc.json'
REFRACTION_STATIC_PREFLIGHT_OBSERVATIONS_CSV_NAME = (
    'refraction_static_preflight_observations.csv'
)

_OBSERVATION_COLUMNS = (
    'trace_sorted_index',
    'trace_original_index',
    'source_id',
    'receiver_id',
    'offset_m',
    'pick_time_s',
    'finite_pick',
    'inside_offset_gate',
    'source_endpoint_key',
    'receiver_endpoint_key',
    'used_for_inversion',
    'rejection_reason',
)


class RefractionStaticPreflightError(ValueError):
    """Raised when preflight diagnostics identify invalid static inputs."""


@dataclass(frozen=True)
class RefractionStaticPreflightDiagnostics:
    status: Literal['ok', 'warning', 'error']
    warnings: list[str]
    errors: list[str]
    summary: dict[str, Any]
    observation_reason_counts: dict[str, int]
    endpoint_summary: dict[str, Any]


def scan_refraction_static_pick_npz(
    *,
    npz_path: Path,
    n_traces: int,
    n_samples: int,
    dt_s: float,
    sorted_trace_index: np.ndarray,
    uploaded_pick_metadata: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Read dependency-light NPZ metadata without normalizing pick order."""
    path = Path(npz_path)
    errors: list[str] = []
    summary: dict[str, Any] = {
        'stored_npz_path': str(path),
        'uploaded_npz_original_filename': None,
        'npz_keys': [],
        'selected_pick_key': None,
        'pick_array_shape': None,
        'n_pick_values': None,
        'n_finite_pick_values': None,
        'n_nan_pick_values': None,
        'has_sorted_to_original': False,
        'sorted_to_original_shape': None,
        'sorted_to_original_is_permutation': None,
    }
    if uploaded_pick_metadata:
        summary['uploaded_npz_original_filename'] = uploaded_pick_metadata.get(
            'original_filename'
        )

    try:
        npz_file = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        summary['errors'] = [f'Could not read npz pick source: {path}: {exc}']
        return summary

    with npz_file as npz:
        keys = list(npz.files)
        summary['npz_keys'] = keys
        selected_key = next((key for key in PICK_TIME_KEYS if key in keys), None)
        summary['selected_pick_key'] = selected_key
        if selected_key is None:
            errors.append(
                'unsupported pick artifact key; selected pick key is missing; '
                'accepted keys: '
                + ', '.join(PICK_TIME_KEYS)
            )
        else:
            picks = np.asarray(npz[selected_key])
            summary['pick_array_shape'] = tuple(int(dim) for dim in picks.shape)
            summary['n_pick_values'] = int(picks.size)
            if np.issubdtype(picks.dtype, np.number) and not np.issubdtype(
                picks.dtype,
                np.complexfloating,
            ):
                picks_f64 = picks.astype(np.float64, copy=False)
                n_finite = int(np.count_nonzero(np.isfinite(picks_f64)))
                summary['n_finite_pick_values'] = n_finite
                summary['n_nan_pick_values'] = int(np.count_nonzero(np.isnan(picks_f64)))
                if n_finite != int(picks.size):
                    errors.append(
                        'pick array contains non-finite values: '
                        f'n_finite={n_finite}, n_values={int(picks.size)}'
                    )
            else:
                summary['n_finite_pick_values'] = 0
                summary['n_nan_pick_values'] = 0
                errors.append('pick array must have a real numeric dtype')
            if picks.ndim != 1 or picks.shape != (n_traces,):
                errors.append(
                    'pick array length mismatch: '
                    f'expected {(n_traces,)}, got {picks.shape}'
                )

        if 'n_traces' in keys:
            _validate_npz_int_scalar(
                npz,
                key='n_traces',
                expected=int(n_traces),
                errors=errors,
                summary=summary,
            )
        if 'n_samples' in keys:
            _validate_npz_int_scalar(
                npz,
                key='n_samples',
                expected=int(n_samples),
                errors=errors,
                summary=summary,
            )
        if 'dt' in keys:
            _validate_npz_float_scalar(
                npz,
                key='dt',
                expected=float(dt_s),
                errors=errors,
                summary=summary,
            )

        if 'sorted_to_original' in keys:
            raw_order = np.asarray(npz['sorted_to_original'])
            summary['has_sorted_to_original'] = True
            summary['sorted_to_original_shape'] = tuple(
                int(dim) for dim in raw_order.shape
            )
            try:
                npz_order = validate_sorted_to_original(
                    raw_order,
                    expected_n_traces=n_traces,
                    role='npz',
                )
                reader_order = validate_sorted_to_original(
                    np.asarray(sorted_trace_index),
                    expected_n_traces=n_traces,
                    role='reader',
                )
                summary['sorted_to_original_is_permutation'] = True
                if not np.array_equal(npz_order, reader_order):
                    errors.append('sorted_to_original mismatch')
            except ValueError as exc:
                summary['sorted_to_original_is_permutation'] = False
                errors.append(str(exc))
        else:
            summary['sorted_to_original_is_permutation'] = None

    summary['errors'] = errors
    return _json_safe(summary)


def build_preflight_diagnostics_from_input_model(
    *,
    req: RefractionStaticApplyRequest,
    input_model: Any,
    n_samples: int | None,
    dt_s: float | None,
    pick_npz_summary: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> RefractionStaticPreflightDiagnostics:
    """Build full preflight diagnostics from the assembled sorted input model."""
    error_values = list(errors or [])
    warning_values = list(warnings or [])
    used_mask = _used_for_inversion_mask(input_model)
    distance = np.asarray(input_model.distance_m_sorted, dtype=np.float64)
    finite_distance = np.isfinite(distance)
    positive_offset = finite_distance & (distance > 0.0)
    inside_gate = _inside_offset_gate(distance, req=req)
    source_geometry_valid = _finite_all(
        input_model.source_x_m_sorted,
        input_model.source_y_m_sorted,
        input_model.source_elevation_m_sorted,
    )
    receiver_geometry_valid = _finite_all(
        input_model.receiver_x_m_sorted,
        input_model.receiver_y_m_sorted,
        input_model.receiver_elevation_m_sorted,
    )
    valid_geometry = source_geometry_valid & receiver_geometry_valid
    finite_pick = np.isfinite(np.asarray(input_model.pick_time_s_sorted, dtype=np.float64))

    filter_counts = {
        'n_total_traces': int(input_model.n_traces),
        'n_finite_picks': int(np.count_nonzero(finite_pick)),
        'n_valid_geometry': int(np.count_nonzero(valid_geometry)),
        'n_positive_offset': int(np.count_nonzero(positive_offset)),
        'n_inside_offset_gate': int(np.count_nonzero(positive_offset & inside_gate)),
        'n_after_static_input_mask': int(
            np.count_nonzero(input_model.valid_observation_mask_sorted)
        ),
        'n_used_for_inversion': int(np.count_nonzero(used_mask)),
    }
    pick_summary = dict(pick_npz_summary or {})
    if not pick_summary:
        pick_metadata = input_model.metadata.get('pick_source_metadata', {})
        if isinstance(pick_metadata, dict):
            pick_summary = _pick_summary_from_metadata(pick_metadata)
    summary: dict[str, Any] = {
        'file_id': req.file_id,
        'key1_byte': int(req.key1_byte),
        'key2_byte': int(req.key2_byte),
        'n_traces': int(input_model.n_traces),
        'n_samples': None if n_samples is None else int(n_samples),
        'dt_s': None if dt_s is None else float(dt_s),
        'trace_order_summary': _trace_order_summary(input_model.sorted_trace_index),
        'pick_source_kind': input_model.metadata.get('pick_source_kind'),
        'pick_npz': pick_summary,
        'geometry': _geometry_summary(
            req=req,
            input_model=input_model,
            source_geometry_valid=source_geometry_valid,
            receiver_geometry_valid=receiver_geometry_valid,
            distance=distance,
        ),
        'observation_filters': filter_counts,
        'input_rejection_counts': _counts(input_model.rejection_reason_sorted),
        'linkage_mode': req.linkage.mode,
    }
    observation_reason_counts = _observation_reason_counts(
        input_model=input_model,
        finite_pick=finite_pick,
        source_geometry_valid=source_geometry_valid,
        receiver_geometry_valid=receiver_geometry_valid,
        positive_offset=positive_offset,
        inside_gate=inside_gate,
        used_mask=used_mask,
    )
    endpoint_summary = _endpoint_summary(input_model=input_model, used_mask=used_mask)
    status: Literal['ok', 'warning', 'error'] = (
        'error' if error_values else 'warning' if warning_values else 'ok'
    )
    return RefractionStaticPreflightDiagnostics(
        status=status,
        warnings=warning_values,
        errors=error_values,
        summary=_json_safe(summary),
        observation_reason_counts=observation_reason_counts,
        endpoint_summary=_json_safe(endpoint_summary),
    )


def build_preflight_diagnostics_for_npz_error(
    *,
    req: RefractionStaticApplyRequest,
    n_traces: int,
    n_samples: int,
    dt_s: float,
    sorted_trace_index: np.ndarray,
    pick_npz_summary: dict[str, Any],
    errors: list[str],
) -> RefractionStaticPreflightDiagnostics:
    """Build a partial diagnostics payload when NPZ validation fails early."""
    summary = {
        'file_id': req.file_id,
        'key1_byte': int(req.key1_byte),
        'key2_byte': int(req.key2_byte),
        'n_traces': int(n_traces),
        'n_samples': int(n_samples),
        'dt_s': float(dt_s),
        'trace_order_summary': _trace_order_summary(sorted_trace_index),
        'pick_source_kind': req.pick_source.kind,
        'pick_npz': pick_npz_summary,
        'geometry': _requested_geometry_summary(req),
        'observation_filters': {
            'n_total_traces': int(n_traces),
            'n_finite_picks': int(
                pick_npz_summary.get('n_finite_pick_values') or 0
            ),
            'n_valid_geometry': None,
            'n_positive_offset': None,
            'n_inside_offset_gate': None,
            'n_after_static_input_mask': None,
            'n_used_for_inversion': None,
        },
        'linkage_mode': req.linkage.mode,
    }
    return RefractionStaticPreflightDiagnostics(
        status='error',
        warnings=[],
        errors=list(errors),
        summary=_json_safe(summary),
        observation_reason_counts={},
        endpoint_summary={},
    )


def write_refraction_static_preflight_artifacts(
    job_dir: Path,
    diagnostics: RefractionStaticPreflightDiagnostics,
    *,
    input_model: Any | None = None,
    req: RefractionStaticApplyRequest | None = None,
) -> dict[str, Path]:
    """Write preflight QC JSON and compact observation CSV artifacts."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / REFRACTION_STATIC_PREFLIGHT_QC_JSON_NAME
    observations_path = root / REFRACTION_STATIC_PREFLIGHT_OBSERVATIONS_CSV_NAME
    _write_json_atomic(qc_path, asdict(diagnostics))
    rows: list[dict[str, Any]]
    if input_model is None or req is None:
        rows = []
    else:
        rows = _observation_rows(input_model=input_model, req=req)
    _write_csv_atomic(observations_path, rows)
    return {'qc_json': qc_path, 'observations_csv': observations_path}


def preflight_error_message(
    diagnostics: RefractionStaticPreflightDiagnostics,
    *,
    fallback: str = 'Refraction static preflight failed',
) -> str:
    count_parts = _preflight_error_count_parts(diagnostics)
    suffix = '' if not count_parts else ' (' + '; '.join(count_parts) + ')'
    if diagnostics.errors:
        return (
            'Refraction static preflight failed: '
            + '; '.join(diagnostics.errors)
            + suffix
        )
    return fallback + suffix


def _preflight_error_count_parts(
    diagnostics: RefractionStaticPreflightDiagnostics,
) -> list[str]:
    parts: list[str] = []
    observation_counts = _format_nonzero_counts(
        diagnostics.observation_reason_counts,
        skip={'used_for_inversion'},
    )
    if observation_counts:
        parts.append(f'observation reason counts: {observation_counts}')
    summary = diagnostics.summary
    if isinstance(summary, dict):
        rejection_counts = summary.get('input_rejection_counts')
        if isinstance(rejection_counts, dict):
            formatted = _format_nonzero_counts(rejection_counts, skip={'ok'})
            if formatted:
                parts.append(f'input rejection counts: {formatted}')
    return parts


def _format_nonzero_counts(
    counts: dict[str, Any],
    *,
    skip: set[str],
) -> str:
    items: list[str] = []
    for key in sorted(counts):
        if key in skip:
            continue
        value = counts[key]
        if not isinstance(value, int):
            continue
        if value <= 0:
            continue
        items.append(f'{key}={value}')
    return ', '.join(items)


def no_observations_preflight_error(summary: dict[str, Any]) -> str:
    filters = summary.get('observation_filters')
    if not isinstance(filters, dict):
        return 'No valid refraction observations remain after preflight filtering.'
    count_text = ', '.join(f'{key}={value}' for key, value in filters.items())
    return (
        'No valid refraction observations remain after preflight filtering '
        f'({count_text}).'
    )


def _validate_npz_int_scalar(
    npz: np.lib.npyio.NpzFile,
    *,
    key: str,
    expected: int,
    errors: list[str],
    summary: dict[str, Any],
) -> None:
    arr = np.asarray(npz[key])
    if arr.size != 1 or np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(
        arr.dtype,
        np.integer,
    ):
        errors.append(f'{key} must be an integer scalar')
        return
    value = int(arr.reshape(-1)[0])
    summary[key] = value
    if value != expected:
        errors.append(f'{key} mismatch: expected {expected}, got {value}')


def _validate_npz_float_scalar(
    npz: np.lib.npyio.NpzFile,
    *,
    key: str,
    expected: float,
    errors: list[str],
    summary: dict[str, Any],
) -> None:
    arr = np.asarray(npz[key])
    if arr.size != 1 or np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(
        arr.dtype,
        np.number,
    ):
        errors.append(f'{key} must be a numeric scalar')
        return
    value = float(arr.reshape(-1)[0])
    summary[key] = value
    if not np.isfinite(value) or abs(value - expected) > 1.0e-9:
        errors.append(f'{key} mismatch: expected {expected}, got {value}')


def _pick_summary_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        'uploaded_npz_original_filename': metadata.get('original_filename'),
        'stored_npz_path': metadata.get('npz_path') or metadata.get('stored_npz_path'),
        'npz_keys': metadata.get('npz_keys', []),
        'selected_pick_key': metadata.get('accepted_pick_key'),
        'pick_array_shape': metadata.get('pick_array_shape'),
        'n_pick_values': metadata.get('n_pick_values'),
        'n_finite_pick_values': metadata.get('n_finite_pick_values'),
        'n_nan_pick_values': metadata.get('n_nan_pick_values'),
        'has_sorted_to_original': metadata.get('has_sorted_to_original'),
        'sorted_to_original_shape': metadata.get('sorted_to_original_shape'),
        'sorted_to_original_is_permutation': metadata.get(
            'sorted_to_original_is_permutation'
        ),
    }


def _trace_order_summary(sorted_trace_index: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(sorted_trace_index)
    summary: dict[str, Any] = {
        'shape': tuple(int(dim) for dim in arr.shape),
        'is_permutation': False,
        'min_original_index': None,
        'max_original_index': None,
        'first_original_indices': [],
        'last_original_indices': [],
    }
    try:
        valid = validate_sorted_to_original(
            arr,
            expected_n_traces=int(arr.shape[0]) if arr.ndim == 1 else 0,
            role='trace_order',
        )
    except ValueError:
        return summary
    summary['is_permutation'] = True
    if valid.size:
        summary['min_original_index'] = int(valid.min())
        summary['max_original_index'] = int(valid.max())
        summary['first_original_indices'] = [int(item) for item in valid[:5].tolist()]
        summary['last_original_indices'] = [int(item) for item in valid[-5:].tolist()]
    return summary


def _requested_geometry_summary(req: RefractionStaticApplyRequest) -> dict[str, Any]:
    geometry = req.geometry
    return {
        'source_id_byte': int(geometry.source_id_byte),
        'receiver_id_byte': int(geometry.receiver_id_byte),
        'source_x_byte': int(geometry.source_x_byte),
        'source_y_byte': int(geometry.source_y_byte),
        'receiver_x_byte': int(geometry.receiver_x_byte),
        'receiver_y_byte': int(geometry.receiver_y_byte),
        'source_elevation_byte': int(geometry.source_elevation_byte),
        'receiver_elevation_byte': int(geometry.receiver_elevation_byte),
        'offset_byte': (
            None if req.moveout.offset_byte is None else int(req.moveout.offset_byte)
        ),
    }


def _geometry_summary(
    *,
    req: RefractionStaticApplyRequest,
    input_model: Any,
    source_geometry_valid: np.ndarray,
    receiver_geometry_valid: np.ndarray,
    distance: np.ndarray,
) -> dict[str, Any]:
    summary = _requested_geometry_summary(req)
    source_endpoint_keys = np.asarray(input_model.source_endpoint_key_sorted)
    receiver_endpoint_keys = np.asarray(input_model.receiver_endpoint_key_sorted)
    summary.update(
        {
            'n_unique_source_ids': int(np.unique(input_model.source_id_sorted).shape[0]),
            'n_unique_receiver_ids': int(
                np.unique(input_model.receiver_id_sorted).shape[0]
            ),
            'n_unique_source_endpoints': _n_nonempty_unique(source_endpoint_keys),
            'n_unique_receiver_endpoints': _n_nonempty_unique(receiver_endpoint_keys),
            'n_missing_source_geometry': int(
                np.count_nonzero(~source_geometry_valid)
            ),
            'n_missing_receiver_geometry': int(
                np.count_nonzero(~receiver_geometry_valid)
            ),
            'source_x_min': _finite_stat(input_model.source_x_m_sorted, 'min'),
            'source_x_max': _finite_stat(input_model.source_x_m_sorted, 'max'),
            'receiver_x_min': _finite_stat(input_model.receiver_x_m_sorted, 'min'),
            'receiver_x_max': _finite_stat(input_model.receiver_x_m_sorted, 'max'),
            'offset_min': _finite_stat(distance, 'min'),
            'offset_median': _finite_stat(distance, 'median'),
            'offset_max': _finite_stat(distance, 'max'),
        }
    )
    return summary


def _endpoint_summary(*, input_model: Any, used_mask: np.ndarray) -> dict[str, Any]:
    source_nodes = np.asarray(input_model.source_node_id_sorted, dtype=np.int64)
    receiver_nodes = np.asarray(input_model.receiver_node_id_sorted, dtype=np.int64)
    return {
        'n_source_endpoint_rows': _n_nonempty_unique(
            input_model.source_endpoint_key_sorted
        ),
        'n_receiver_endpoint_rows': _n_nonempty_unique(
            input_model.receiver_endpoint_key_sorted
        ),
        'n_source_nodes_used': int(np.unique(source_nodes[used_mask & (source_nodes >= 0)]).shape[0]),
        'n_receiver_nodes_used': int(
            np.unique(receiver_nodes[used_mask & (receiver_nodes >= 0)]).shape[0]
        ),
        'n_invalid_source_endpoints': int(np.count_nonzero(source_nodes < 0)),
        'n_invalid_receiver_endpoints': int(np.count_nonzero(receiver_nodes < 0)),
    }


def _observation_reason_counts(
    *,
    input_model: Any,
    finite_pick: np.ndarray,
    source_geometry_valid: np.ndarray,
    receiver_geometry_valid: np.ndarray,
    positive_offset: np.ndarray,
    inside_gate: np.ndarray,
    used_mask: np.ndarray,
) -> dict[str, int]:
    source_nodes = np.asarray(input_model.source_node_id_sorted, dtype=np.int64)
    receiver_nodes = np.asarray(input_model.receiver_node_id_sorted, dtype=np.int64)
    rejection = np.asarray(input_model.rejection_reason_sorted).astype(str)
    return {
        'non_finite_pick': int(np.count_nonzero(~finite_pick)),
        'missing_geometry': int(
            np.count_nonzero(~source_geometry_valid | ~receiver_geometry_valid)
        ),
        'non_positive_offset': int(np.count_nonzero(~positive_offset)),
        'outside_offset_gate': int(np.count_nonzero(positive_offset & ~inside_gate)),
        'invalid_endpoint': int(
            np.count_nonzero((source_nodes < 0) | (receiver_nodes < 0))
        ),
        'missing_linkage': int(np.count_nonzero(rejection == 'missing_linkage')),
        'used_for_inversion': int(np.count_nonzero(used_mask)),
    }


def _observation_rows(
    *,
    input_model: Any,
    req: RefractionStaticApplyRequest,
) -> list[dict[str, Any]]:
    distance = np.asarray(input_model.distance_m_sorted, dtype=np.float64)
    inside_gate = _inside_offset_gate(distance, req=req)
    used_mask = _used_for_inversion_mask(input_model)
    sorted_index = np.asarray(input_model.sorted_trace_index, dtype=np.int64)
    picks = np.asarray(input_model.pick_time_s_sorted, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index in range(int(input_model.n_traces)):
        rows.append(
            {
                'trace_sorted_index': index,
                'trace_original_index': int(sorted_index[index]),
                'source_id': _json_scalar(input_model.source_id_sorted[index]),
                'receiver_id': _json_scalar(input_model.receiver_id_sorted[index]),
                'offset_m': _json_scalar(distance[index]),
                'pick_time_s': _json_scalar(picks[index]),
                'finite_pick': bool(np.isfinite(picks[index])),
                'inside_offset_gate': bool(inside_gate[index]),
                'source_endpoint_key': str(input_model.source_endpoint_key_sorted[index]),
                'receiver_endpoint_key': str(
                    input_model.receiver_endpoint_key_sorted[index]
                ),
                'used_for_inversion': bool(used_mask[index]),
                'rejection_reason': str(input_model.rejection_reason_sorted[index]),
            }
        )
    return rows


def _used_for_inversion_mask(input_model: Any) -> np.ndarray:
    base = np.asarray(input_model.valid_observation_mask_sorted, dtype=bool)
    layer_masks = getattr(input_model, 'layer_observation_masks', None)
    if layer_masks is None:
        return base
    masks = [
        np.asarray(mask, dtype=bool)
        for mask in layer_masks.layer_used_mask_sorted.values()
        if np.asarray(mask).shape == base.shape
    ]
    if not masks:
        return base
    out = np.zeros(base.shape, dtype=bool)
    for mask in masks:
        out |= mask
    return np.ascontiguousarray(out, dtype=bool)


def _inside_offset_gate(distance: np.ndarray, *, req: RefractionStaticApplyRequest) -> np.ndarray:
    finite = np.isfinite(distance)
    mask = np.ones(distance.shape, dtype=bool)
    if req.moveout.min_offset_m is not None:
        mask &= ~finite | (distance >= float(req.moveout.min_offset_m))
    if req.moveout.max_offset_m is not None:
        mask &= ~finite | (distance <= float(req.moveout.max_offset_m))
    return np.ascontiguousarray(mask, dtype=bool)


def _finite_all(*values: np.ndarray) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=bool)
    mask = np.ones(np.asarray(values[0]).shape, dtype=bool)
    for value in values:
        mask &= np.isfinite(np.asarray(value, dtype=np.float64))
    return mask


def _counts(values: np.ndarray) -> dict[str, int]:
    arr = np.asarray(values).astype(str)
    return {str(value): int(np.count_nonzero(arr == value)) for value in np.unique(arr)}


def _finite_stat(values: np.ndarray, stat: str) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    if stat == 'min':
        return float(np.min(finite))
    if stat == 'max':
        return float(np.max(finite))
    if stat == 'median':
        return float(np.median(finite))
    raise ValueError(f'unsupported statistic: {stat}')


def _n_nonempty_unique(values: np.ndarray) -> int:
    arr = np.asarray(values).astype(str)
    arr = arr[arr != '']
    return int(np.unique(arr).shape[0])


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        tmp_path.write_text(
            json.dumps(_json_safe(payload), ensure_ascii=True, sort_keys=True),
            encoding='utf-8',
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=_OBSERVATION_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in _OBSERVATION_COLUMNS})
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _csv_value(value: Any) -> Any:
    value = _json_scalar(value)
    if value is None:
        return ''
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _json_scalar(value)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return value


__all__ = [
    'REFRACTION_STATIC_PREFLIGHT_OBSERVATIONS_CSV_NAME',
    'REFRACTION_STATIC_PREFLIGHT_QC_JSON_NAME',
    'RefractionStaticPreflightDiagnostics',
    'RefractionStaticPreflightError',
    'build_preflight_diagnostics_for_npz_error',
    'build_preflight_diagnostics_from_input_model',
    'no_observations_preflight_error',
    'preflight_error_message',
    'scan_refraction_static_pick_npz',
    'write_refraction_static_preflight_artifacts',
]
