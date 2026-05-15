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
from app.services.job_artifact_refs import resolve_job_artifact_path
from app.services.refraction_static_apply_trace_store import (
    apply_refraction_statics_to_trace_store,
)
from app.services.refraction_static_artifacts import (
    REFRACTION_STATIC_HISTORY_JSON_NAME,
    refraction_static_double_application_qc,
    write_refraction_static_artifacts,
    write_refraction_static_history_json,
)
from app.services.refraction_static_datum import (
    build_refraction_datum_statics,
    write_refraction_datum_statics_artifacts,
)
from app.services.refraction_static_export_service import (
    resolve_refraction_static_export_formats,
    write_inline_refraction_static_export_artifacts,
)
from app.services.refraction_static_field_composition import (
    compose_refraction_endpoint_field_corrections,
    compose_refraction_final_trace_shift,
    compose_refraction_trace_field_corrections,
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
from app.services.refraction_static_manual_static import (
    RefractionManualStaticTableRow,
    load_refraction_manual_static_table_rows,
    manual_static_inline_rows,
    resolve_refraction_manual_static,
)
from app.services.refraction_static_multilayer_service import (
    compute_refraction_multilayer_datum_statics_from_input_model,
)
from app.services.refraction_static_source_depth import (
    REFRACTION_SOURCE_DEPTH_QC_JSON_NAME,
    REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
    compute_source_depth_weathering_time_correction_from_result,
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
from app.services.refraction_static_uphole import (
    REFRACTION_UPHOLE_QC_JSON_NAME,
    REFRACTION_UPHOLE_SOURCES_CSV_NAME,
    compute_uphole_time_correction_from_result,
)
from app.services.refraction_static_weathering_replacement import (
    compute_weathering_replacement_statics_from_first_breaks,
)

_REQUEST_JSON_NAME = 'refraction_static_request.json'
_ARTIFACT_ONLY_DONE_MESSAGE = 'refraction_static_artifacts_written_artifact_only'
_PUBLIC_MULTILAYER_APPLY_CONTRACT = (
    'public multi-layer refraction apply requires '
    'model.method=multilayer_time_term, '
    'conversion.mode=t1lsst_multilayer, and exactly enabled layers '
    'v2_t1 and v3_t2 for conversion.layer_count=2 or v2_t1, v3_t2, '
    'and vsub_t3 for conversion.layer_count=3'
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


class RefractionFieldCorrectionNotImplemented(NotImplementedError):
    """Raised when M4 field-correction modes reach the apply service."""


def reject_unsupported_refraction_field_corrections(
    req: RefractionStaticApplyRequest,
) -> None:
    """Reject unsupported M4 field-correction modes before expensive I/O."""
    field_corrections = req.field_corrections
    unsupported: list[str] = []
    if field_corrections.uphole.mode not in {'none', 'header_time'}:
        unsupported.append(
            f'field_corrections.uphole.mode={field_corrections.uphole.mode}'
        )
    if not unsupported:
        return
    raise RefractionFieldCorrectionNotImplemented(
        'M4 field-correction follow-up implementation is required for '
        f'{", ".join(unsupported)}. Configure each field-correction mode to '
        'none for the current public refraction apply workflow.'
    )


def resolve_refraction_first_layer_request(
    *,
    req: RefractionStaticApplyRequest,
    state: AppState,
    job_dir: Path,
    uploaded_pick_npz_path: Path | None = None,
    uploaded_pick_metadata: dict[str, object] | None = None,
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
        uploaded_pick_npz_path=uploaded_pick_npz_path,
        uploaded_pick_metadata=uploaded_pick_metadata,
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
        'only for the M3 public multi-layer contract. '
        f'{_PUBLIC_MULTILAYER_APPLY_CONTRACT}. Unsupported request fields: '
        f'{", ".join(unsupported)}.'
    )


def _is_public_multilayer_apply(
    req: RefractionStaticApplyRequest,
) -> bool:
    if (
        req.model.method != 'multilayer_time_term'
        or req.conversion.mode != 't1lsst_multilayer'
    ):
        return False
    _require_public_multilayer_apply(req)
    return True


def _require_public_multilayer_apply(
    req: RefractionStaticApplyRequest,
) -> None:
    normalized_layers = normalize_refraction_static_layers(req.model)
    enabled_kinds = tuple(config.kind for config in normalized_layers)
    enabled_text = ', '.join(enabled_kinds) if enabled_kinds else 'none'
    expected_by_count = {
        2: ('v2_t1', 'v3_t2'),
        3: ('v2_t1', 'v3_t2', 'vsub_t3'),
    }
    expected = expected_by_count.get(req.conversion.layer_count)
    if expected is None or enabled_kinds != expected:
        raise RefractionMultiLayerApplyNotImplemented(
            f'{_PUBLIC_MULTILAYER_APPLY_CONTRACT}; got '
            f'conversion.layer_count={req.conversion.layer_count!r}, '
            f'enabled layer kinds={enabled_text}.'
        )

    v3_config = normalized_layers[1]
    if v3_config.velocity_mode not in ('fixed_global', 'solve_global'):
        raise RefractionMultiLayerApplyNotImplemented(
            f'{_PUBLIC_MULTILAYER_APPLY_CONTRACT}; '
            f'conversion.layer_count={req.conversion.layer_count!r}, '
            f'enabled layer kinds={enabled_text}; v3_t2 velocity_mode='
            f'{v3_config.velocity_mode} is not supported. Public apply '
            'currently requires global V3/T2 velocity; cell V3/T2 is '
            'available only for internal layer solving.'
        )
    if req.conversion.layer_count == 3:
        vsub_config = normalized_layers[2]
        if vsub_config.velocity_mode not in ('fixed_global', 'solve_global'):
            raise RefractionMultiLayerApplyNotImplemented(
                f'{_PUBLIC_MULTILAYER_APPLY_CONTRACT}; '
                f'conversion.layer_count={req.conversion.layer_count!r}, '
                f'enabled layer kinds={enabled_text}; vsub_t3 velocity_mode='
                f'{vsub_config.velocity_mode} is not supported. Public apply '
                'currently requires global Vsub/T3 velocity; cell Vsub/T3 is '
                'available only for internal layer solving.'
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


def _refraction_static_request_payload(
    req: RefractionStaticApplyRequest,
    *,
    uploaded_pick_metadata: dict[str, object] | None = None,
) -> dict[str, Any]:
    payload = req.model_dump(mode='json')
    if uploaded_pick_metadata is not None:
        pick_source = payload.get('pick_source')
        if isinstance(pick_source, dict):
            pick_source.update(uploaded_pick_metadata)
            pick_source.pop('job_id', None)
            pick_source.pop('artifact_name', None)
    if 'field_corrections' not in req.model_fields_set:
        payload.pop('field_corrections', None)
    if 'export' not in req.model_fields_set:
        payload.pop('export', None)
    return payload


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
    if req.export.enabled:
        _set_job_progress_message(
            state,
            job_id,
            progress=0.91,
            message='writing_refraction_static_export_artifacts',
        )
        write_inline_refraction_static_export_artifacts(
            job_id=job_id,
            req=req,
            job_dir=job_dir,
        )
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
            if isinstance(result, RefractionDatumStaticsResult):
                write_refraction_static_history_json(
                    result=result,
                    req=req,
                    path=job_dir / REFRACTION_STATIC_HISTORY_JSON_NAME,
                    output_file_id=corrected_result.corrected_file_id,
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


def _upstream_artifact_names_for_final_refraction_job(
    *,
    first_layer: _ResolvedFirstLayerRequest,
    req: RefractionStaticApplyRequest,
) -> tuple[str, ...]:
    return (
        first_layer.upstream_artifact_names
        + _field_correction_upstream_artifact_names(req)
    )


def _field_correction_upstream_artifact_names(
    req: RefractionStaticApplyRequest,
) -> tuple[str, ...]:
    names: list[str] = []
    if req.field_corrections.source_depth.mode == 'weathering_velocity_time':
        names.extend(
            (
                REFRACTION_SOURCE_DEPTH_QC_JSON_NAME,
                REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
            )
        )
    if req.field_corrections.uphole.mode == 'header_time':
        names.extend(
            (
                REFRACTION_UPHOLE_QC_JSON_NAME,
                REFRACTION_UPHOLE_SOURCES_CSV_NAME,
            )
        )
    return tuple(names)


def _with_source_depth_field_correction(
    *,
    result: RefractionDatumStaticsResult,
    input_model: RefractionStaticInputModel | None,
    req: RefractionStaticApplyRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer,
) -> RefractionDatumStaticsResult:
    correction = req.field_corrections.source_depth
    if correction.mode == 'none':
        return result
    if input_model is None or input_model.source_depth_result is None:
        raise ValueError(
            'source-depth field correction requires resolved source-depth input rows'
        )
    v1_m_s = float(resolved_first_layer.weathering_velocity_m_s)
    max_shift_s = (
        float(correction.max_abs_source_depth_m) / v1_m_s
        if np.isfinite(v1_m_s) and v1_m_s > 0.0
        else None
    )
    field_result = compute_source_depth_weathering_time_correction_from_result(
        input_model.source_depth_result,
        v1_m_s,
        max_abs_shift_s=max_shift_s,
    )
    depth, shift, status = _map_source_depth_field_correction_to_sources(
        source_endpoint_key=result.source_endpoint_key,
        source_depth_result=input_model.source_depth_result,
        field_result=field_result,
    )
    guard, warnings = _source_depth_double_count_guard_qc(req)
    source_depth_qc = {
        **field_result.qc,
        'source_depth_double_count_guard': guard,
        'warnings': warnings,
    }
    return replace(
        result,
        source_depth_m=depth,
        source_depth_shift_s=shift,
        source_depth_status=status,
        source_depth_field_correction_qc=source_depth_qc,
        qc={
            **result.qc,
            'field_corrections': {
                **(
                    result.qc.get('field_corrections')
                    if isinstance(result.qc.get('field_corrections'), dict)
                    else {}
                ),
                'source_depth': source_depth_qc,
            },
        },
    )


def _with_uphole_field_correction(
    *,
    result: RefractionDatumStaticsResult,
    input_model: RefractionStaticInputModel | None,
    req: RefractionStaticApplyRequest,
) -> RefractionDatumStaticsResult:
    correction = req.field_corrections.uphole
    if correction.mode == 'none':
        return result
    if correction.mode != 'header_time':
        raise RefractionFieldCorrectionNotImplemented(
            'M4 field-correction follow-up implementation is required for '
            f'field_corrections.uphole.mode={correction.mode}.'
        )
    if input_model is None or input_model.uphole_result is None:
        raise ValueError('uphole field correction requires resolved uphole input rows')
    field_result = compute_uphole_time_correction_from_result(
        input_model.uphole_result,
        positive_time_means_delay=bool(correction.positive_time_means_delay),
        max_abs_uphole_time_s=float(correction.max_abs_uphole_time_s),
    )
    uphole_time, shift, status = _map_uphole_field_correction_to_sources(
        source_endpoint_key=result.source_endpoint_key,
        uphole_result=input_model.uphole_result,
        field_result=field_result,
    )
    uphole_qc = {
        **input_model.uphole_result.qc,
        **field_result.qc,
        'uphole_time_byte': correction.uphole_time_byte,
        'uphole_time_unit': correction.uphole_time_unit,
    }
    return replace(
        result,
        source_uphole_time_s=uphole_time,
        source_uphole_shift_s=shift,
        source_uphole_status=status,
        source_uphole_field_correction_qc=uphole_qc,
        qc={
            **result.qc,
            'field_corrections': {
                **(
                    result.qc.get('field_corrections')
                    if isinstance(result.qc.get('field_corrections'), dict)
                    else {}
                ),
                'uphole': uphole_qc,
            },
        },
    )


def _with_manual_static_field_correction(
    *,
    result: RefractionDatumStaticsResult,
    input_model: RefractionStaticInputModel | None,
    req: RefractionStaticApplyRequest,
    state: AppState | None,
) -> RefractionDatumStaticsResult:
    correction = req.field_corrections.manual_static
    if correction.mode == 'none':
        return result
    rows = _manual_static_rows_from_request(req=req, state=state)
    source_endpoint_id = _endpoint_ids_for_result(
        result_endpoint_key=result.source_endpoint_key,
        input_endpoint_key=(
            None if input_model is None else input_model.source_endpoint_key_sorted
        ),
        input_endpoint_id=(
            None if input_model is None else input_model.source_endpoint_id_sorted
        ),
    )
    receiver_endpoint_id = _endpoint_ids_for_result(
        result_endpoint_key=result.receiver_endpoint_key,
        input_endpoint_key=(
            None if input_model is None else input_model.receiver_endpoint_key_sorted
        ),
        input_endpoint_id=(
            None if input_model is None else input_model.receiver_endpoint_id_sorted
        ),
    )
    manual_result = resolve_refraction_manual_static(
        source_endpoint_key=result.source_endpoint_key,
        source_endpoint_id=source_endpoint_id,
        source_node_id=result.source_node_id,
        receiver_endpoint_key=result.receiver_endpoint_key,
        receiver_endpoint_id=receiver_endpoint_id,
        receiver_node_id=result.receiver_node_id,
        rows=rows,
        mode=correction.mode,
        sign_convention=correction.sign_convention,
        allow_missing_endpoints=bool(correction.allow_missing_endpoints),
    )
    return replace(
        result,
        source_manual_static_shift_s=manual_result.source_manual_static_shift_s,
        source_manual_static_status=manual_result.source_manual_static_status,
        receiver_manual_static_shift_s=manual_result.receiver_manual_static_shift_s,
        receiver_manual_static_status=manual_result.receiver_manual_static_status,
        manual_static_field_correction_qc=manual_result.qc,
        qc={
            **result.qc,
            'field_corrections': {
                **(
                    result.qc.get('field_corrections')
                    if isinstance(result.qc.get('field_corrections'), dict)
                    else {}
                ),
                'manual_static': manual_result.qc,
            },
        },
    )


def _with_field_correction_composition(
    *,
    result: RefractionDatumStaticsResult,
    input_model: RefractionStaticInputModel | None,
    req: RefractionStaticApplyRequest,
) -> RefractionDatumStaticsResult:
    composition = req.field_corrections.composition
    if (
        not bool(composition.enabled)
        or not _field_correction_components_requested(req)
    ):
        return result
    if not _has_field_correction_components(result):
        return result
    if input_model is None:
        raise ValueError('field-correction composition requires input trace endpoints')

    source_endpoint_id = _endpoint_ids_for_result(
        result_endpoint_key=result.source_endpoint_key,
        input_endpoint_key=input_model.source_endpoint_key_sorted,
        input_endpoint_id=input_model.source_endpoint_id_sorted,
    )
    receiver_endpoint_id = _endpoint_ids_for_result(
        result_endpoint_key=result.receiver_endpoint_key,
        input_endpoint_key=input_model.receiver_endpoint_key_sorted,
        input_endpoint_id=input_model.receiver_endpoint_id_sorted,
    )
    source_field = compose_refraction_endpoint_field_corrections(
        endpoint_kind='source',
        endpoint_key=result.source_endpoint_key,
        endpoint_id=source_endpoint_id,
        node_id=result.source_node_id,
        source_depth_shift_s=result.source_depth_shift_s,
        source_depth_status=result.source_depth_status,
        uphole_shift_s=result.source_uphole_shift_s,
        uphole_status=result.source_uphole_status,
        manual_static_shift_s=result.source_manual_static_shift_s,
        manual_static_status=result.source_manual_static_status,
    )
    receiver_field = compose_refraction_endpoint_field_corrections(
        endpoint_kind='receiver',
        endpoint_key=result.receiver_endpoint_key,
        endpoint_id=receiver_endpoint_id,
        node_id=result.receiver_node_id,
        manual_static_shift_s=result.receiver_manual_static_shift_s,
        manual_static_status=result.receiver_manual_static_status,
    )
    trace_field = compose_refraction_trace_field_corrections(
        source_endpoint_field=source_field,
        receiver_endpoint_field=receiver_field,
        source_endpoint_key_sorted=input_model.source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=input_model.receiver_endpoint_key_sorted,
    )
    final_shift = compose_refraction_final_trace_shift(
        refraction_trace_shift_s_sorted=result.refraction_trace_shift_s_sorted,
        trace_static_status_sorted=result.trace_static_status_sorted,
        trace_static_valid_mask_sorted=result.trace_static_valid_mask_sorted,
        trace_field_correction=trace_field,
        apply_to_trace_shift=bool(composition.apply_to_trace_shift),
        invalid_component_policy=composition.invalid_component_policy,
    )
    composition_qc = {
        **final_shift.qc,
        'source_endpoint_field': source_field.qc,
        'receiver_endpoint_field': receiver_field.qc,
        'trace_field': trace_field.qc,
    }
    updated = replace(
        result,
        source_field_shift_s=source_field.total_field_shift_s,
        source_field_static_status=source_field.field_static_status,
        receiver_field_shift_s=receiver_field.total_field_shift_s,
        receiver_field_static_status=receiver_field.field_static_status,
        source_field_shift_s_sorted=trace_field.source_field_shift_s_sorted,
        receiver_field_shift_s_sorted=trace_field.receiver_field_shift_s_sorted,
        trace_field_shift_s_sorted=trace_field.trace_field_shift_s_sorted,
        trace_field_static_status_sorted=(
            trace_field.trace_field_static_status_sorted
        ),
        trace_field_static_valid_mask_sorted=(
            trace_field.trace_field_static_status_sorted == 'ok'
        ),
        base_refraction_trace_shift_s_sorted=(
            final_shift.base_refraction_trace_shift_s_sorted
        ),
        final_trace_shift_s_sorted=final_shift.final_trace_shift_s_sorted,
        final_trace_static_status_sorted=final_shift.final_trace_static_status_sorted,
        final_trace_static_valid_mask_sorted=(
            final_shift.final_trace_static_valid_mask_sorted
        ),
        applied_field_shift_s_sorted=final_shift.applied_field_shift_s_sorted,
        field_composition_qc=composition_qc,
        qc={
            **result.qc,
            'field_corrections': {
                **(
                    result.qc.get('field_corrections')
                    if isinstance(result.qc.get('field_corrections'), dict)
                    else {}
                ),
                'composition': composition_qc,
            },
        },
    )
    return updated


def _has_field_correction_components(result: RefractionDatumStaticsResult) -> bool:
    return any(
        value is not None
        for value in (
            result.source_depth_shift_s,
            result.source_uphole_shift_s,
            result.source_manual_static_shift_s,
            result.receiver_manual_static_shift_s,
        )
    )


def _field_correction_components_requested(req: RefractionStaticApplyRequest) -> bool:
    return (
        req.field_corrections.source_depth.mode != 'none'
        or req.field_corrections.uphole.mode != 'none'
        or req.field_corrections.manual_static.mode != 'none'
    )


def _manual_static_rows_from_request(
    *,
    req: RefractionStaticApplyRequest,
    state: AppState | None,
) -> tuple[RefractionManualStaticTableRow, ...]:
    correction = req.field_corrections.manual_static
    if correction.mode == 'inline_table':
        return (
            manual_static_inline_rows(
                endpoint_kind='source',
                entries=correction.source_inline_table,
                sign_convention=correction.sign_convention or 'applied_shift_s',
            )
            + manual_static_inline_rows(
                endpoint_kind='receiver',
                entries=correction.receiver_inline_table,
                sign_convention=correction.sign_convention or 'applied_shift_s',
            )
        )
    if correction.mode != 'artifact_table':
        raise RefractionFieldCorrectionNotImplemented(
            'M4 field-correction follow-up implementation is required for '
            f'field_corrections.manual_static.mode={correction.mode}.'
        )
    if state is None:
        raise ValueError('manual static artifact table mode requires app state')

    artifact_specs: dict[tuple[str, str], set[str]] = {}
    for default_kind, artifact in (
        ('source', correction.source_table_artifact),
        ('receiver', correction.receiver_table_artifact),
    ):
        if artifact is None:
            continue
        key = (artifact.job_id, artifact.artifact_name)
        artifact_specs.setdefault(key, set()).add(default_kind)

    rows: list[RefractionManualStaticTableRow] = []
    for (job_id, artifact_name), default_kinds in artifact_specs.items():
        path = resolve_job_artifact_path(
            state,
            job_id=job_id,
            name=artifact_name,
            expected_file_id=req.file_id,
            expected_key1_byte=req.key1_byte,
            expected_key2_byte=req.key2_byte,
            reference_label='manual static',
        )
        default_kind = next(iter(default_kinds)) if len(default_kinds) == 1 else None
        rows.extend(
            load_refraction_manual_static_table_rows(
                path,
                default_endpoint_kind=default_kind,
            )
        )
    return tuple(rows)


def _endpoint_ids_for_result(
    *,
    result_endpoint_key: np.ndarray,
    input_endpoint_key: np.ndarray | None,
    input_endpoint_id: np.ndarray | None,
) -> np.ndarray | None:
    result_keys = np.asarray(result_endpoint_key)
    if input_endpoint_key is None or input_endpoint_id is None:
        return None
    key_values = np.asarray(input_endpoint_key)
    id_values = np.asarray(input_endpoint_id)
    if key_values.shape != id_values.shape:
        return None
    id_by_key: dict[str, int] = {}
    for raw_key, raw_id in zip(key_values.tolist(), id_values.tolist(), strict=True):
        key = str(raw_key)
        if key and key not in id_by_key:
            id_by_key[key] = int(raw_id)
    out = np.empty(int(result_keys.shape[0]), dtype=np.int64)
    for index, raw_key in enumerate(result_keys.tolist()):
        key = str(raw_key)
        if key not in id_by_key:
            return None
        out[index] = int(id_by_key[key])
    return np.ascontiguousarray(out, dtype=np.int64)


def _map_uphole_field_correction_to_sources(
    *,
    source_endpoint_key: np.ndarray,
    uphole_result: Any,
    field_result: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    endpoint_count = int(np.asarray(source_endpoint_key).shape[0])
    uphole_out = np.full(endpoint_count, np.nan, dtype=np.float64)
    shift_out = np.full(endpoint_count, np.nan, dtype=np.float64)
    status_out = np.full(endpoint_count, 'missing_uphole_time', dtype='<U48')
    time_by_key = {
        str(raw_key): float(uphole_result.uphole_time_s[index])
        for index, raw_key in enumerate(uphole_result.source_endpoint_key.tolist())
    }
    shift = field_result.component_shift_s['uphole_shift_s']
    component_status = field_result.component_status['uphole_shift_s']
    field_by_key = {
        str(raw_key): (float(shift[index]), str(component_status[index]))
        for index, raw_key in enumerate(field_result.endpoint_key.tolist())
    }
    for index, raw_key in enumerate(np.asarray(source_endpoint_key).tolist()):
        key = str(raw_key)
        if key in time_by_key:
            uphole_out[index] = time_by_key[key]
        field = field_by_key.get(key)
        if field is None:
            continue
        shift_value, status_value = field
        shift_out[index] = shift_value
        status_out[index] = status_value
    return (
        np.ascontiguousarray(uphole_out, dtype=np.float64),
        np.ascontiguousarray(shift_out, dtype=np.float64),
        np.ascontiguousarray(status_out, dtype='<U48'),
    )


def _map_source_depth_field_correction_to_sources(
    *,
    source_endpoint_key: np.ndarray,
    source_depth_result: Any,
    field_result: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    endpoint_count = int(np.asarray(source_endpoint_key).shape[0])
    depth_out = np.full(endpoint_count, np.nan, dtype=np.float64)
    shift_out = np.full(endpoint_count, np.nan, dtype=np.float64)
    status_out = np.full(endpoint_count, 'missing_source_depth', dtype='<U48')
    depth_by_key = {
        str(raw_key): float(source_depth_result.source_depth_m[index])
        for index, raw_key in enumerate(source_depth_result.source_endpoint_key.tolist())
    }
    shift = field_result.component_shift_s['source_depth_shift_s']
    component_status = field_result.component_status['source_depth_shift_s']
    field_by_key = {
        str(raw_key): (float(shift[index]), str(component_status[index]))
        for index, raw_key in enumerate(field_result.endpoint_key.tolist())
    }
    for index, raw_key in enumerate(np.asarray(source_endpoint_key).tolist()):
        key = str(raw_key)
        if key in depth_by_key:
            depth_out[index] = depth_by_key[key]
        field = field_by_key.get(key)
        if field is None:
            continue
        shift_value, status_value = field
        shift_out[index] = shift_value
        status_out[index] = status_value
    return (
        np.ascontiguousarray(depth_out, dtype=np.float64),
        np.ascontiguousarray(shift_out, dtype=np.float64),
        np.ascontiguousarray(status_out, dtype='<U48'),
    )


def _source_depth_double_count_guard_qc(
    req: RefractionStaticApplyRequest,
) -> tuple[str, list[str]]:
    if req.field_corrections.source_depth.mode == 'none':
        return 'not_applicable', []
    source_depth_byte_configured = (
        req.geometry.source_depth_byte is not None
        or req.field_corrections.source_depth.source_depth_byte is not None
    )
    if req.datum.mode != 'none' and source_depth_byte_configured:
        return (
            'warning_existing_datum_uses_source_depth',
            [
                'source depth is configured while '
                'field_corrections.source_depth.mode=weathering_velocity_time '
                'and datum corrections are enabled; verify source depth is not '
                'already included in datum source elevation handling.'
            ],
        )
    return 'checked', []


def _with_static_history_double_application_qc(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    state: AppState,
) -> RefractionDatumStaticsResult:
    source_meta = _source_trace_store_meta_for_static_history(req=req, state=state)
    if source_meta is None:
        return result
    qc = refraction_static_double_application_qc(req=req, source_meta=source_meta)
    if qc.get('status') == 'duplicate_rejected':
        message = qc.get('message')
        raise ValueError(
            str(message)
            if message
            else 'static history double-application policy rejected the job'
        )
    if qc.get('status') in {'duplicate_warned', 'duplicate_allowed'}:
        return replace(
            result,
            qc={
                **result.qc,
                'static_history': qc,
            },
        )
    return result


def _source_trace_store_meta_for_static_history(
    *,
    req: RefractionStaticApplyRequest,
    state: AppState,
) -> dict[str, object] | None:
    try:
        store_path = Path(state.file_registry.get_store_path(req.file_id))
    except Exception:  # noqa: BLE001
        return None
    meta_path = store_path / 'meta.json'
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def _run_public_multilayer_refraction_static_apply_job(
    *,
    job_id: str,
    req: RefractionStaticApplyRequest,
    state: AppState,
    job_dir: Path,
    first_layer: _ResolvedFirstLayerRequest,
    uploaded_pick_npz_path: Path | None = None,
    uploaded_pick_metadata: dict[str, object] | None = None,
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
        uploaded_pick_npz_path=uploaded_pick_npz_path,
        uploaded_pick_metadata=uploaded_pick_metadata,
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
    datum_result = _with_source_depth_field_correction(
        result=workflow.datum_result,
        input_model=input_model,
        req=req,
        resolved_first_layer=first_layer.resolved,
    )
    datum_result = _with_uphole_field_correction(
        result=datum_result,
        input_model=input_model,
        req=req,
    )
    datum_result = _with_manual_static_field_correction(
        result=datum_result,
        input_model=input_model,
        req=req,
        state=state,
    )
    datum_result = _with_field_correction_composition(
        result=datum_result,
        input_model=input_model,
        req=req,
    )
    datum_result = _with_static_history_double_application_qc(
        result=datum_result,
        req=req,
        state=state,
    )
    _set_job_progress_message(
        state,
        job_id,
        progress=0.90,
        message='writing_refraction_static_artifacts',
    )
    write_refraction_datum_statics_artifacts(job_dir, datum_result)
    write_refraction_static_artifacts(
        result=datum_result,
        req=req,
        job_dir=job_dir,
        resolved_first_layer=first_layer.resolved,
        upstream_artifact_names=_upstream_artifact_names_for_final_refraction_job(
            first_layer=first_layer,
            req=req,
        ),
        source_job_id=job_id,
    )
    return _finish_refraction_static_apply_job(
        job_id=job_id,
        req=req,
        state=state,
        job_dir=job_dir,
        result=datum_result,
    )


def _run_refraction_static_apply_job_body(
    *,
    job_id: str,
    req: RefractionStaticApplyRequest,
    state: AppState,
    uploaded_pick_npz_path: Path | None = None,
    uploaded_pick_metadata: dict[str, object] | None = None,
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
    requested_export_formats = resolve_refraction_static_export_formats(req.export)
    request_record = {
        'job_id': job_id,
        'job_type': 'statics',
        'statics_kind': 'refraction',
        'source_file_id': req.file_id,
        'key1_byte': req.key1_byte,
        'key2_byte': req.key2_byte,
        'request': _refraction_static_request_payload(
            req,
            uploaded_pick_metadata=uploaded_pick_metadata,
        ),
    }
    if req.export.enabled:
        request_record['export'] = {
            'enabled': True,
            'requested_formats': list(requested_export_formats),
        }
    _write_json_atomic(
        job_dir / _REQUEST_JSON_NAME,
        request_record,
    )
    reject_unsupported_refraction_field_corrections(req)
    public_multilayer_apply = _is_public_multilayer_apply(req)
    if not public_multilayer_apply:
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
        uploaded_pick_npz_path=uploaded_pick_npz_path,
        uploaded_pick_metadata=uploaded_pick_metadata,
    )
    input_req = first_layer.req
    if public_multilayer_apply:
        return _run_public_multilayer_refraction_static_apply_job(
            job_id=job_id,
            req=input_req,
            state=state,
            job_dir=job_dir,
            first_layer=first_layer,
            uploaded_pick_npz_path=uploaded_pick_npz_path,
            uploaded_pick_metadata=uploaded_pick_metadata,
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
        or input_req.pick_source.kind == 'uploaded_npz'
        or input_req.model.method == 'multilayer_time_term'
        or input_req.field_corrections.source_depth.mode != 'none'
        or input_req.field_corrections.uphole.mode != 'none'
        or input_req.field_corrections.manual_static.mode != 'none'
    ):
        weathering_input_model = build_refraction_static_input_model(
            req=input_req,
            state=state,
            job_dir=job_dir,
            uploaded_pick_npz_path=uploaded_pick_npz_path,
            uploaded_pick_metadata=uploaded_pick_metadata,
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
    datum_result = _with_source_depth_field_correction(
        result=datum_result,
        input_model=weathering_input_model,
        req=active_req,
        resolved_first_layer=first_layer.resolved,
    )
    datum_result = _with_uphole_field_correction(
        result=datum_result,
        input_model=weathering_input_model,
        req=active_req,
    )
    datum_result = _with_manual_static_field_correction(
        result=datum_result,
        input_model=weathering_input_model,
        req=active_req,
        state=state,
    )
    datum_result = _with_field_correction_composition(
        result=datum_result,
        input_model=weathering_input_model,
        req=active_req,
    )
    datum_result = _with_static_history_double_application_qc(
        result=datum_result,
        req=active_req,
        state=state,
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
    if (
        isinstance(datum_result, RefractionDatumStaticsResult)
        and datum_result.field_composition_qc is not None
    ):
        write_refraction_datum_statics_artifacts(job_dir, datum_result)
    write_refraction_static_artifacts(
        result=datum_result,
        req=active_req,
        job_dir=job_dir,
        resolved_first_layer=first_layer.resolved,
        upstream_artifact_names=_upstream_artifact_names_for_final_refraction_job(
            first_layer=first_layer,
            req=active_req,
        ),
        source_job_id=job_id,
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
    uploaded_pick_npz_path: Path | None = None,
    uploaded_pick_metadata: dict[str, object] | None = None,
) -> None:
    """Start the refraction statics job lifecycle."""

    def worker() -> JobCompletion | None:
        return _run_refraction_static_apply_job_body(
            job_id=job_id,
            req=req,
            state=state,
            uploaded_pick_npz_path=uploaded_pick_npz_path,
            uploaded_pick_metadata=uploaded_pick_metadata,
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
    'RefractionFieldCorrectionNotImplemented',
    'RefractionFirstLayerNotImplemented',
    'RefractionMultiLayerApplyNotImplemented',
    'ResolvedRefractionFirstLayer',
    'normalize_refraction_first_layer_request',
    'reject_unsupported_refraction_field_corrections',
    'resolve_refraction_first_layer_request',
    'run_refraction_static_apply_job',
    '_with_field_correction_composition',
]
