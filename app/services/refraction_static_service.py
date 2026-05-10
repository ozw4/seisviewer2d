"""Refraction static correction background job service."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import numpy as np

from app.api.schemas import (
    RefractionStaticApplyRequest,
    RefractionStaticModelRequest,
)
from app.core.state import AppState
from app.services.job_runner import JobCompletion, JobFailure, run_job_with_lifecycle
from app.services.refraction_static_apply_trace_store import (
    apply_refraction_statics_to_trace_store,
)
from app.services.refraction_static_artifacts import write_refraction_static_artifacts
from app.services.refraction_static_datum import (
    build_refraction_datum_statics,
    write_refraction_datum_statics_artifacts,
)
from app.services.refraction_static_first_layer import (
    normalize_refraction_first_layer_request,
)
from app.services.refraction_static_inputs import build_refraction_static_input_model
from app.services.refraction_static_layer_config import (
    RefractionStaticLayerConfig,
    normalize_refraction_static_layers,
)
from app.services.refraction_static_layer_observations import (
    build_refraction_layer_observation_masks,
    refraction_layer_observation_qc,
)
from app.services.refraction_static_multilayer_service import (
    compute_refraction_multilayer_datum_statics_from_input_model,
)
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
    RefractionStaticInputModel,
    ResolvedRefractionFirstLayer,
)
from app.services.refraction_static_v1 import (
    REFRACTION_V1_ESTIMATES_CSV_NAME,
    REFRACTION_V1_QC_JSON_NAME,
    estimate_global_v1_from_direct_arrivals,
    write_refraction_v1_artifacts,
)
from app.services.refraction_static_weathering_replacement import (
    compute_weathering_replacement_statics_from_first_breaks,
)

_REQUEST_JSON_NAME = 'refraction_static_request.json'
_ARTIFACT_ONLY_DONE_MESSAGE = 'refraction_static_artifacts_written_artifact_only'
_PUBLIC_TWO_LAYER_APPLY_CONTRACT = (
    'public two-layer refraction apply requires '
    'model.method=multilayer_time_term, '
    'conversion.mode=t1lsst_multilayer, conversion.layer_count=2, '
    'and exactly enabled layers v2_t1 and v3_t2'
)


@dataclass(frozen=True)
class _ResolvedFirstLayerRequest:
    req: RefractionStaticApplyRequest
    resolved: ResolvedRefractionFirstLayer
    input_model: RefractionStaticInputModel | None
    upstream_artifact_names: tuple[str, ...] = ()


class RefractionFirstLayerNotImplemented(NotImplementedError):
    """Raised when a requested first-layer mode is accepted but not implemented."""


class RefractionMultiLayerApplyNotImplemented(NotImplementedError):
    """Raised when accepted multi-layer statics fields reach the apply service."""


def resolve_refraction_first_layer_request(
    *,
    req: RefractionStaticApplyRequest,
    state: AppState,
    job_dir: Path,
) -> _ResolvedFirstLayerRequest:
    """Resolve V1 for a full refraction statics request."""
    mode = req.model.first_layer_mode
    if mode == 'constant':
        return _ResolvedFirstLayerRequest(
            req=req,
            resolved=normalize_refraction_first_layer_request(req.model),
            input_model=None,
        )

    first_layer = req.model.first_layer
    if first_layer is None:
        raise ValueError('model.first_layer is required for V1 estimation')
    if req.model.weathering_velocity_m_s is not None:
        raise ValueError(
            'model.weathering_velocity_m_s must be omitted when '
            'model.first_layer.mode is estimate_direct_arrival'
        )
    if first_layer.weathering_velocity_m_s is not None:
        raise ValueError(
            'model.first_layer.weathering_velocity_m_s must be omitted when '
            'model.first_layer.mode is estimate_direct_arrival'
        )

    v1_req = _request_without_moveout_offset_gates(req)
    input_model = build_refraction_static_input_model(
        req=v1_req,
        state=state,
        job_dir=job_dir,
    )
    estimate = estimate_global_v1_from_direct_arrivals(
        input_model=input_model,
        first_layer=first_layer,
    )
    write_refraction_v1_artifacts(job_dir, estimate)
    return _ResolvedFirstLayerRequest(
        req=req,
        resolved=ResolvedRefractionFirstLayer(
            mode='estimate_direct_arrival',
            weathering_velocity_m_s=estimate.resolved_weathering_velocity_m_s,
            status='estimated',
            qc={
                **estimate.qc,
                'weathering_velocity_m_s': (
                    estimate.resolved_weathering_velocity_m_s
                ),
            },
        ),
        input_model=input_model,
        upstream_artifact_names=(
            REFRACTION_V1_QC_JSON_NAME,
            REFRACTION_V1_ESTIMATES_CSV_NAME,
        ),
    )


def _request_without_moveout_offset_gates(
    req: RefractionStaticApplyRequest,
) -> RefractionStaticApplyRequest:
    moveout = req.moveout.model_copy(
        update={
            'min_offset_m': None,
            'max_offset_m': None,
        }
    )
    return req.model_copy(update={'moveout': moveout})


def _reject_unsupported_multilayer_apply(req: RefractionStaticApplyRequest) -> None:
    unsupported: list[str] = []
    if req.conversion.mode == 't1lsst_multilayer':
        unsupported.append('conversion.mode=t1lsst_multilayer')
    if req.model.method == 'multilayer_time_term':
        unsupported_layers = [
            config.kind
            for config in normalize_refraction_static_layers(req.model)
            if config.kind != 'v2_t1'
        ]
        if unsupported_layers:
            unsupported.append(
                'enabled multi-layer refraction layers='
                f'{", ".join(unsupported_layers)}'
            )
    if not unsupported:
        return
    raise RefractionMultiLayerApplyNotImplemented(
        'refraction static apply supports accepted multi-layer request fields '
        'only for the M3 public two-layer contract. '
        f'{_PUBLIC_TWO_LAYER_APPLY_CONTRACT}. Unsupported request fields: '
        f'{", ".join(unsupported)}.'
    )


def _is_public_two_layer_multilayer_apply(
    req: RefractionStaticApplyRequest,
) -> bool:
    if (
        req.model.method != 'multilayer_time_term'
        or req.conversion.mode != 't1lsst_multilayer'
    ):
        return False
    _require_public_two_layer_multilayer_apply(req)
    return True


def _require_public_two_layer_multilayer_apply(
    req: RefractionStaticApplyRequest,
) -> None:
    normalized_layers = normalize_refraction_static_layers(req.model)
    enabled_kinds = tuple(config.kind for config in normalized_layers)
    if req.conversion.layer_count != 2 or enabled_kinds != ('v2_t1', 'v3_t2'):
        enabled_text = ', '.join(enabled_kinds) if enabled_kinds else 'none'
        raise RefractionMultiLayerApplyNotImplemented(
            f'{_PUBLIC_TWO_LAYER_APPLY_CONTRACT}; got '
            f'conversion.layer_count={req.conversion.layer_count!r}, '
            f'enabled layers={enabled_text}. Public apply does not implement '
            'vsub_t3 or three-layer T1LSST conversion.'
        )

    v3_config = normalized_layers[1]
    if v3_config.velocity_mode not in ('fixed_global', 'solve_global'):
        raise RefractionMultiLayerApplyNotImplemented(
            f'{_PUBLIC_TWO_LAYER_APPLY_CONTRACT}; v3_t2 velocity_mode='
            f'{v3_config.velocity_mode} is not supported. Public apply '
            'currently requires global V3/T2 velocity; cell V3 is not '
            'implemented.'
        )


def _solver_request_for_refraction_static_apply(
    req: RefractionStaticApplyRequest,
) -> RefractionStaticApplyRequest:
    if req.model.method != 'multilayer_time_term':
        return req
    normalized_layers = normalize_refraction_static_layers(req.model)
    if len(normalized_layers) != 1 or normalized_layers[0].kind != 'v2_t1':
        raise RefractionMultiLayerApplyNotImplemented(
            'refraction static apply can currently execute only a single '
            'enabled v2_t1 layer'
        )
    model = _legacy_model_for_v2_layer(
        model=req.model,
        config=normalized_layers[0],
    )
    return req.model_copy(update={'model': model})


def _legacy_model_for_v2_layer(
    *,
    model: RefractionStaticModelRequest,
    config: RefractionStaticLayerConfig,
) -> RefractionStaticModelRequest:
    payload = model.model_dump(mode='python')
    payload.update(
        {
            'method': 'gli_variable_thickness',
            'bedrock_velocity_mode': config.velocity_mode,
            'bedrock_velocity_m_s': (
                config.fixed_velocity_m_s
                if config.velocity_mode == 'fixed_global'
                else None
            ),
            'initial_bedrock_velocity_m_s': (
                config.initial_velocity_m_s
                if config.velocity_mode in ('solve_global', 'solve_cell')
                else None
            ),
            'min_bedrock_velocity_m_s': config.min_velocity_m_s,
            'max_bedrock_velocity_m_s': config.max_velocity_m_s,
            'refractor_cell': _refractor_cell_payload_for_v2_layer(
                payload=payload,
                config=config,
            ),
            'layers': None,
            'allow_overlapping_layer_gates': False,
        }
    )
    return RefractionStaticModelRequest.model_validate(payload)


def _solver_input_model_for_refraction_static_apply(
    *,
    req: RefractionStaticApplyRequest,
    input_model: RefractionStaticInputModel | None,
) -> RefractionStaticInputModel | None:
    if input_model is None or req.model.method != 'multilayer_time_term':
        return input_model
    normalized_layers = normalize_refraction_static_layers(req.model)
    if len(normalized_layers) != 1 or normalized_layers[0].kind != 'v2_t1':
        raise RefractionMultiLayerApplyNotImplemented(
            'refraction static apply can currently execute only a single '
            'enabled v2_t1 layer'
        )
    layer_masks = input_model.layer_observation_masks
    if layer_masks is None:
        layer_masks = build_refraction_layer_observation_masks(
            input_model=input_model,
            model=req.model,
        )
    try:
        used_mask = layer_masks.layer_used_mask_sorted['v2_t1']
        rejection_reason = layer_masks.layer_rejection_reason_sorted['v2_t1']
    except KeyError as exc:
        raise RefractionMultiLayerApplyNotImplemented(
            'enabled v2_t1 layer is missing observation masks'
        ) from exc
    used = np.ascontiguousarray(used_mask, dtype=bool)
    reason = np.asarray(rejection_reason).astype('<U32', copy=False)
    expected_shape = (int(input_model.n_traces),)
    if used.shape != expected_shape:
        raise RefractionMultiLayerApplyNotImplemented(
            'enabled v2_t1 layer observation mask shape mismatch'
        )
    if reason.shape != expected_shape:
        raise RefractionMultiLayerApplyNotImplemented(
            'enabled v2_t1 layer rejection-reason shape mismatch'
        )
    return replace(
        input_model,
        valid_observation_mask_sorted=used,
        rejection_reason_sorted=np.ascontiguousarray(reason, dtype='<U32'),
        qc={
            **input_model.qc,
            'active_layer_kind': 'v2_t1',
            'layers': refraction_layer_observation_qc(layer_masks),
        },
        layer_observation_masks=layer_masks,
    )


def _refractor_cell_payload_for_v2_layer(
    *,
    payload: dict[str, Any],
    config: RefractionStaticLayerConfig,
) -> dict[str, Any] | None:
    if config.velocity_mode != 'solve_cell':
        return None
    raw_cell = payload.get('refractor_cell')
    if raw_cell is None:
        return None
    cell = dict(raw_cell)
    if config.min_observations_per_cell is not None:
        cell['min_observations_per_cell'] = config.min_observations_per_cell
    if config.smoothing_weight is not None:
        cell['velocity_smoothing_weight'] = config.smoothing_weight
    return cell


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding='utf-8',
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _set_job_progress_message(
    state: AppState,
    job_id: str,
    *,
    progress: float,
    message: str,
) -> None:
    with state.lock:
        if state.jobs.get(job_id) is None:
            return
        state.jobs.set_progress(job_id, progress)
        state.jobs.set_message(job_id, message)


def _resolve_job_dir(state: AppState, job_id: str) -> Path:
    with state.lock:
        job = state.jobs.get(job_id)
        artifacts_dir = job.get('artifacts_dir') if isinstance(job, dict) else None
    if not isinstance(artifacts_dir, str) or not artifacts_dir:
        raise ValueError('job artifacts_dir is not available')
    return Path(artifacts_dir)


def _finish_refraction_static_apply_job(
    *,
    job_id: str,
    req: RefractionStaticApplyRequest,
    state: AppState,
    job_dir: Path,
    result: RefractionDatumStaticsResult,
) -> JobCompletion:
    if req.apply.register_corrected_file:
        _set_job_progress_message(
            state,
            job_id,
            progress=0.92,
            message='applying_refraction_static_trace_shift',
        )
        corrected_result = apply_refraction_statics_to_trace_store(
            req=req,
            result=result,
            state=state,
            job_id=job_id,
            job_dir=job_dir,
        )
        if corrected_result.corrected_file_id is not None:
            with state.lock:
                state.jobs.set_static_corrected_file(
                    job_id,
                    corrected_file_id=corrected_result.corrected_file_id,
                    corrected_store_path=str(
                        corrected_result.corrected_trace_store_path
                    ),
                )
        _set_job_progress_message(
            state,
            job_id,
            progress=1.0,
            message='refraction_corrected_trace_store_registered',
        )
        return JobCompletion(finished_ts=time.time())

    _set_job_progress_message(
        state,
        job_id,
        progress=1.0,
        message=_ARTIFACT_ONLY_DONE_MESSAGE,
    )
    return JobCompletion(
        finished_ts=time.time(),
        message=_ARTIFACT_ONLY_DONE_MESSAGE,
    )


def _run_public_two_layer_refraction_static_apply_job(
    *,
    job_id: str,
    req: RefractionStaticApplyRequest,
    state: AppState,
    job_dir: Path,
    first_layer: _ResolvedFirstLayerRequest,
) -> JobCompletion:
    _set_job_progress_message(
        state,
        job_id,
        progress=0.20,
        message='building_refraction_multilayer_input_model',
    )
    input_model = build_refraction_static_input_model(
        req=req,
        state=state,
        job_dir=job_dir,
    )
    _set_job_progress_message(
        state,
        job_id,
        progress=0.55,
        message='computing_refraction_multilayer_datum_statics',
    )
    workflow = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=input_model,
        model=req.model,
        solver=req.solver,
        datum=req.datum,
        apply_options=req.apply,
        resolved_first_layer=first_layer.resolved,
        state=state,
        file_id=req.file_id,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
    )
    _set_job_progress_message(
        state,
        job_id,
        progress=0.90,
        message='writing_refraction_static_artifacts',
    )
    write_refraction_datum_statics_artifacts(job_dir, workflow.datum_result)
    write_refraction_static_artifacts(
        result=workflow.datum_result,
        req=req,
        job_dir=job_dir,
        resolved_first_layer=first_layer.resolved,
        upstream_artifact_names=first_layer.upstream_artifact_names,
    )
    return _finish_refraction_static_apply_job(
        job_id=job_id,
        req=req,
        state=state,
        job_dir=job_dir,
        result=workflow.datum_result,
    )


def _run_refraction_static_apply_job_body(
    *,
    job_id: str,
    req: RefractionStaticApplyRequest,
    state: AppState,
) -> JobCompletion | None:
    req = RefractionStaticApplyRequest.model_validate(req)
    _set_job_progress_message(
        state,
        job_id,
        progress=0.05,
        message='writing_refraction_static_request',
    )
    job_dir = _resolve_job_dir(state, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        job_dir / _REQUEST_JSON_NAME,
        {
            'job_id': job_id,
            'job_type': 'statics',
            'statics_kind': 'refraction',
            'source_file_id': req.file_id,
            'key1_byte': req.key1_byte,
            'key2_byte': req.key2_byte,
            'request': req.model_dump(mode='json'),
        },
    )
    public_two_layer_apply = _is_public_two_layer_multilayer_apply(req)
    if not public_two_layer_apply:
        _reject_unsupported_multilayer_apply(req)
    _set_job_progress_message(
        state,
        job_id,
        progress=0.12,
        message='resolving_refraction_first_layer',
    )
    first_layer = resolve_refraction_first_layer_request(
        req=req,
        state=state,
        job_dir=job_dir,
    )
    input_req = first_layer.req
    if public_two_layer_apply:
        return _run_public_two_layer_refraction_static_apply_job(
            job_id=job_id,
            req=input_req,
            state=state,
            job_dir=job_dir,
            first_layer=first_layer,
        )

    active_req = _solver_request_for_refraction_static_apply(input_req)
    _set_job_progress_message(
        state,
        job_id,
        progress=0.20,
        message='computing_refraction_weathering_replacement_statics',
    )
    weathering_input_model = first_layer.input_model
    if (
        first_layer.resolved.mode == 'estimate_direct_arrival'
        or input_req.model.method == 'multilayer_time_term'
    ):
        weathering_input_model = build_refraction_static_input_model(
            req=input_req,
            state=state,
            job_dir=job_dir,
        )
    weathering_input_model = _solver_input_model_for_refraction_static_apply(
        req=input_req,
        input_model=weathering_input_model,
    )
    replacement_result = compute_weathering_replacement_statics_from_first_breaks(
        req=active_req,
        state=state,
        job_dir=job_dir,
        input_model=weathering_input_model,
        resolved_first_layer=first_layer.resolved,
    )
    _set_job_progress_message(
        state,
        job_id,
        progress=0.70,
        message='refraction_weathering_replacement_statics_computed',
    )
    _set_job_progress_message(
        state,
        job_id,
        progress=0.75,
        message='computing_refraction_datum_statics',
    )
    datum_result = build_refraction_datum_statics(
        weathering_replacement_result=replacement_result,
        datum=active_req.datum,
        apply_options=active_req.apply,
        job_dir=job_dir,
        state=state,
        file_id=active_req.file_id,
        key1_byte=active_req.key1_byte,
        key2_byte=active_req.key2_byte,
        resolved_first_layer=first_layer.resolved,
    )
    _set_job_progress_message(
        state,
        job_id,
        progress=0.85,
        message='refraction_datum_statics_computed',
    )
    _set_job_progress_message(
        state,
        job_id,
        progress=0.90,
        message='writing_refraction_static_artifacts',
    )
    write_refraction_static_artifacts(
        result=datum_result,
        req=active_req,
        job_dir=job_dir,
        resolved_first_layer=first_layer.resolved,
        upstream_artifact_names=first_layer.upstream_artifact_names,
    )
    return _finish_refraction_static_apply_job(
        job_id=job_id,
        req=active_req,
        state=state,
        job_dir=job_dir,
        result=datum_result,
    )


def _handle_refraction_static_job_error(_exc: Exception) -> JobFailure:
    return JobFailure(finished_ts=time.time())


def run_refraction_static_apply_job(
    job_id: str,
    req: RefractionStaticApplyRequest,
    state: AppState,
) -> None:
    """Start the refraction statics job lifecycle."""

    def worker() -> JobCompletion | None:
        return _run_refraction_static_apply_job_body(
            job_id=job_id,
            req=req,
            state=state,
        )

    run_job_with_lifecycle(
        state=state,
        job_id=job_id,
        worker=worker,
        progress_1_on_done=False,
        start_progress=0.0,
        clear_message_on_start=True,
        on_error=_handle_refraction_static_job_error,
    )


__all__ = [
    'RefractionFirstLayerNotImplemented',
    'RefractionMultiLayerApplyNotImplemented',
    'ResolvedRefractionFirstLayer',
    'normalize_refraction_first_layer_request',
    'resolve_refraction_first_layer_request',
    'run_refraction_static_apply_job',
]
