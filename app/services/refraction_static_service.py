"""Refraction static correction background job service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

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
from app.services.refraction_static_datum import build_refraction_datum_statics
from app.services.refraction_static_first_layer import (
    normalize_refraction_first_layer_request,
)
from app.services.refraction_static_inputs import build_refraction_static_input_model
from app.services.refraction_static_layer_config import (
    RefractionStaticLayerConfig,
    normalize_refraction_static_layers,
)
from app.services.refraction_static_types import (
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
        'refraction static apply does not yet implement accepted multi-layer '
        f'request fields: {", ".join(unsupported)}. Multi-layer apply '
        'artifacts and conversion must be wired before this request can run.'
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
    if active_req.apply.register_corrected_file:
        _set_job_progress_message(
            state,
            job_id,
            progress=0.92,
            message='applying_refraction_static_trace_shift',
        )
        corrected_result = apply_refraction_statics_to_trace_store(
            req=active_req,
            result=datum_result,
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
