"""Validation-only diagnostics for direct refraction static pick uploads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from app.statics.refraction.contracts.apply import RefractionStaticApplyRequest
from app.core.state import AppState
from app.statics.refraction.application.input_model import build_refraction_static_input_model
from app.statics.refraction.application.pick_source_loader import PICK_TIME_KEYS
from app.statics.refraction.domain.types import RefractionStaticInputModel


def validate_refraction_static_inputs_with_picks(
    *,
    req: RefractionStaticApplyRequest,
    state: AppState,
    pick_npz_path: Path,
    uploaded_pick_metadata: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build preflight diagnostics without launching correction work."""
    req = RefractionStaticApplyRequest.model_validate(req)
    target = {
        'file_id': req.file_id,
        'key1_byte': int(req.key1_byte),
        'key2_byte': int(req.key2_byte),
    }
    errors: list[str] = []
    warnings: list[str] = []
    pick_summary = _empty_pick_npz_summary(pick_npz_path)

    try:
        pick_summary = _inspect_pick_npz(pick_npz_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f'Unable to read pick NPZ: {exc}')

    input_model: RefractionStaticInputModel | None = None
    if not errors:
        try:
            input_model = build_refraction_static_input_model(
                req=req,
                state=state,
                job_dir=None,
                uploaded_pick_npz_path=pick_npz_path,
                uploaded_pick_metadata=uploaded_pick_metadata,
                require_valid_observations=False,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    diagnostics = (
        _diagnostics_from_input_model(input_model)
        if input_model is not None
        else _empty_diagnostics()
    )
    if input_model is not None and diagnostics['n_used_for_inversion'] <= 0:
        errors.append(
            'No valid refraction observations remain after pick, geometry, '
            'offset, and linkage filtering.'
        )

    return {
        'status': 'error' if errors else 'ok',
        'target': target,
        'pick_npz': pick_summary,
        'diagnostics': diagnostics,
        'warnings': warnings,
        'errors': errors,
    }


def _empty_pick_npz_summary(path: Path) -> dict[str, Any]:
    return {
        'filename': Path(path).name,
        'selected_key': None,
        'shape': None,
        'keys': [],
        'order': None,
    }


def _inspect_pick_npz(path: Path) -> dict[str, Any]:
    summary = _empty_pick_npz_summary(path)
    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.files)
        selected = next((key for key in PICK_TIME_KEYS if key in keys), None)
        summary['keys'] = keys
        summary['selected_key'] = selected
        if selected is not None:
            summary['shape'] = [int(dim) for dim in np.asarray(npz[selected]).shape]
        for order_key in ('order', 'trace_order'):
            if order_key in keys:
                summary['order'] = str(np.asarray(npz[order_key]).reshape(-1)[0])
                break
        for scalar_key in ('n_traces', 'n_samples', 'dt'):
            if scalar_key in keys:
                value = np.asarray(npz[scalar_key]).reshape(-1)[0]
                if isinstance(value, np.generic):
                    value = value.item()
                summary[scalar_key] = value
    return summary


def _diagnostics_from_input_model(
    input_model: RefractionStaticInputModel,
) -> dict[str, Any]:
    qc = dict(input_model.qc)
    finite_distance = input_model.distance_m_sorted[
        input_model.valid_observation_mask_sorted
        & np.isfinite(input_model.distance_m_sorted)
    ]
    return {
        'n_total_traces': int(input_model.n_traces),
        'n_finite_picks': int(np.count_nonzero(np.isfinite(input_model.pick_time_s_sorted))),
        'n_valid_picks': int(qc.get('n_valid_picks', 0)),
        'n_used_for_inversion': int(qc.get('n_valid_observations', 0)),
        'n_unique_source_endpoints': _unique_endpoint_count(
            input_model.source_endpoint_key_sorted,
            input_model.source_endpoint_id_sorted,
        ),
        'n_unique_receiver_endpoints': _unique_endpoint_count(
            input_model.receiver_endpoint_key_sorted,
            input_model.receiver_endpoint_id_sorted,
        ),
        'offset_m': {
            'min': _finite_stat(finite_distance, 'min'),
            'median': _finite_stat(finite_distance, 'median'),
            'max': _finite_stat(finite_distance, 'max'),
        },
        'filter_reason_counts': dict(qc.get('rejection_counts', {})),
        'input_qc': qc,
    }


def _unique_endpoint_count(keys: np.ndarray, ids: np.ndarray | None) -> int:
    arr = np.asarray(keys)
    if ids is None:
        mask = arr != ''
    else:
        mask = np.asarray(ids) >= 0
    return int(np.unique(arr[mask]).shape[0])


def _finite_stat(values: np.ndarray, method: str) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if method == 'min':
        return float(np.min(arr))
    if method == 'max':
        return float(np.max(arr))
    if method == 'median':
        return float(np.median(arr))
    raise ValueError(f'unknown finite stat method: {method}')


def _empty_diagnostics() -> dict[str, Any]:
    return {
        'n_total_traces': 0,
        'n_finite_picks': 0,
        'n_valid_picks': 0,
        'n_used_for_inversion': 0,
        'n_unique_source_endpoints': 0,
        'n_unique_receiver_endpoints': 0,
        'offset_m': {'min': None, 'median': None, 'max': None},
        'filter_reason_counts': {},
        'input_qc': {},
    }


__all__ = ['validate_refraction_static_inputs_with_picks']
