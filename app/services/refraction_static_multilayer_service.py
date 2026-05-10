"""Sequential multi-layer time-term orchestration for refraction statics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np

from app.api.schemas import RefractionStaticModelRequest, RefractionStaticSolverRequest
from app.services.refraction_static_cell_grid import RefractionCellGrid
from app.services.refraction_static_half_intercept import (
    estimate_refraction_half_intercept_times_from_first_breaks,
)
from app.services.refraction_static_layer_config import RefractionStaticLayerConfig
from app.services.refraction_static_layer_observations import (
    refraction_layer_observation_qc,
)
from app.services.refraction_static_types import (
    RefractionHalfInterceptTimeResult,
    RefractionLayerKind,
    RefractionLayerObservationMasks,
    RefractionLayerSolveResult,
    RefractionLayerVelocityMode,
    RefractionMultiLayerSolveResult,
    RefractionStaticInputModel,
    ResolvedRefractionFirstLayer,
)

_LAYER_INDEX_BY_KIND: dict[RefractionLayerKind, int] = {
    'v2_t1': 1,
    'v3_t2': 2,
    'vsub_t3': 3,
}


class RefractionMultiLayerSolveError(ValueError):
    """Raised when multi-layer refraction orchestration cannot continue."""


@dataclass(frozen=True)
class RefractionLayerSolverContext:
    """Inputs for one layer solver dispatched by the multi-layer orchestrator."""

    base_input_model: RefractionStaticInputModel
    input_model: RefractionStaticInputModel
    resolved_first_layer: ResolvedRefractionFirstLayer
    layer_config: RefractionStaticLayerConfig
    layer_index: int
    layer_masks: RefractionLayerObservationMasks
    model: RefractionStaticModelRequest
    solver: RefractionStaticSolverRequest
    grid: RefractionCellGrid | None = None


RefractionLayerSolver = Callable[
    [RefractionLayerSolverContext],
    RefractionLayerSolveResult,
]


def solve_refraction_multilayer_time_terms(
    *,
    input_model: RefractionStaticInputModel,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    normalized_layers: tuple[RefractionStaticLayerConfig, ...],
    layer_masks: RefractionLayerObservationMasks,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    grid: RefractionCellGrid | None = None,
    solver_dispatch: Mapping[
        tuple[RefractionLayerKind, RefractionLayerVelocityMode],
        RefractionLayerSolver,
    ]
    | None = None,
) -> RefractionMultiLayerSolveResult:
    """Run enabled refraction layer solves in configured order.

    The built-in dispatcher intentionally supports only the existing one-layer
    V2/T1 solver path. Deeper-layer entries can be supplied through
    ``solver_dispatch`` once their numerical solvers exist; otherwise they fail
    with an explicit unsupported-combination error.
    """
    if not normalized_layers:
        raise RefractionMultiLayerSolveError(
            'at least one enabled refraction layer is required'
        )

    dispatch = _effective_solver_dispatch(solver_dispatch)
    layer_results: list[RefractionLayerSolveResult] = []
    for config in normalized_layers:
        layer_solver = _solver_for_layer(config, dispatch)
        _require_layer_observations(config, layer_masks)
        layer_input = _input_model_for_layer(
            input_model=input_model,
            layer_masks=layer_masks,
            layer_kind=config.kind,
        )
        layer_model = _model_for_layer(model=model, config=config)
        context = RefractionLayerSolverContext(
            base_input_model=input_model,
            input_model=layer_input,
            resolved_first_layer=resolved_first_layer,
            layer_config=config,
            layer_index=_layer_index(config.kind),
            layer_masks=layer_masks,
            model=layer_model,
            solver=solver,
            grid=grid,
        )
        result = layer_solver(context)
        _validate_layer_result(result=result, config=config)
        layer_results.append(result)

    source_endpoint_key, source_node_id = _unique_endpoint_key_nodes(
        input_model.source_endpoint_key_sorted,
        input_model.source_node_id_sorted,
    )
    receiver_endpoint_key, receiver_node_id = _unique_endpoint_key_nodes(
        input_model.receiver_endpoint_key_sorted,
        input_model.receiver_node_id_sorted,
    )
    enabled_kinds = tuple(config.kind for config in normalized_layers)
    qc = {
        'enabled_layer_count': len(enabled_kinds),
        'enabled_layer_kinds': list(enabled_kinds),
        'observation_gates': refraction_layer_observation_qc(layer_masks),
        'layers': {result.layer_kind: result.qc for result in layer_results},
    }
    return RefractionMultiLayerSolveResult(
        enabled_layer_kinds=enabled_kinds,
        layer_results=tuple(layer_results),
        source_endpoint_key=source_endpoint_key,
        receiver_endpoint_key=receiver_endpoint_key,
        source_node_id=source_node_id,
        receiver_node_id=receiver_node_id,
        qc=qc,
    )


def _effective_solver_dispatch(
    overrides: Mapping[
        tuple[RefractionLayerKind, RefractionLayerVelocityMode],
        RefractionLayerSolver,
    ]
    | None,
) -> dict[tuple[RefractionLayerKind, RefractionLayerVelocityMode], RefractionLayerSolver]:
    dispatch: dict[
        tuple[RefractionLayerKind, RefractionLayerVelocityMode],
        RefractionLayerSolver,
    ] = {
        ('v2_t1', 'fixed_global'): _solve_existing_time_term_layer,
        ('v2_t1', 'solve_global'): _solve_existing_time_term_layer,
        ('v2_t1', 'solve_cell'): _solve_existing_time_term_layer,
    }
    if overrides is not None:
        dispatch.update(dict(overrides))
    return dispatch


def _solver_for_layer(
    config: RefractionStaticLayerConfig,
    dispatch: Mapping[
        tuple[RefractionLayerKind, RefractionLayerVelocityMode],
        RefractionLayerSolver,
    ],
) -> RefractionLayerSolver:
    key = (config.kind, config.velocity_mode)
    layer_solver = dispatch.get(key)
    if layer_solver is None:
        raise RefractionMultiLayerSolveError(
            f'refraction layer {config.kind} with velocity_mode='
            f'{config.velocity_mode} is not implemented'
        )
    return layer_solver


def _require_layer_observations(
    config: RefractionStaticLayerConfig,
    layer_masks: RefractionLayerObservationMasks,
) -> None:
    count = int(layer_masks.layer_observation_count.get(config.kind, 0))
    if count <= 0:
        raise RefractionMultiLayerSolveError(
            f'refraction layer {config.kind} has no valid observations'
        )


def _input_model_for_layer(
    *,
    input_model: RefractionStaticInputModel,
    layer_masks: RefractionLayerObservationMasks,
    layer_kind: RefractionLayerKind,
) -> RefractionStaticInputModel:
    try:
        used_mask = layer_masks.layer_used_mask_sorted[layer_kind]
        rejection_reason = layer_masks.layer_rejection_reason_sorted[layer_kind]
    except KeyError as exc:
        raise RefractionMultiLayerSolveError(
            f'refraction layer {layer_kind} does not have observation masks'
        ) from exc
    used = np.ascontiguousarray(used_mask, dtype=bool)
    reason = np.asarray(rejection_reason).astype('<U32', copy=False)
    if used.shape != (int(input_model.n_traces),):
        raise RefractionMultiLayerSolveError(
            f'refraction layer {layer_kind} mask shape mismatch'
        )
    if reason.shape != used.shape:
        raise RefractionMultiLayerSolveError(
            f'refraction layer {layer_kind} rejection-reason shape mismatch'
        )
    return replace(
        input_model,
        valid_observation_mask_sorted=used,
        rejection_reason_sorted=np.ascontiguousarray(reason, dtype='<U32'),
        qc={
            **input_model.qc,
            'active_layer_kind': layer_kind,
            'layers': refraction_layer_observation_qc(layer_masks),
        },
        layer_observation_masks=layer_masks,
    )


def _model_for_layer(
    *,
    model: RefractionStaticModelRequest,
    config: RefractionStaticLayerConfig,
) -> RefractionStaticModelRequest:
    min_velocity = (
        config.min_velocity_m_s
        if config.min_velocity_m_s is not None
        else model.min_bedrock_velocity_m_s
    )
    max_velocity = (
        config.max_velocity_m_s
        if config.max_velocity_m_s is not None
        else model.max_bedrock_velocity_m_s
    )
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
            'min_bedrock_velocity_m_s': min_velocity,
            'max_bedrock_velocity_m_s': max_velocity,
            'refractor_cell': (
                payload.get('refractor_cell')
                if config.velocity_mode == 'solve_cell'
                else None
            ),
            'layers': None,
            'allow_overlapping_layer_gates': False,
        }
    )
    return RefractionStaticModelRequest.model_validate(payload)


def _solve_existing_time_term_layer(
    context: RefractionLayerSolverContext,
) -> RefractionLayerSolveResult:
    result = estimate_refraction_half_intercept_times_from_first_breaks(
        req=_LayerSolveRequest(context.model, context.solver),
        state=_LayerSolveState(),
        input_model=context.input_model,
        resolved_first_layer=context.resolved_first_layer,
    )
    return _layer_result_from_half_intercept(
        result=result,
        layer_kind=context.layer_config.kind,
        layer_index=context.layer_index,
    )


def _layer_result_from_half_intercept(
    *,
    result: RefractionHalfInterceptTimeResult,
    layer_kind: RefractionLayerKind,
    layer_index: int,
) -> RefractionLayerSolveResult:
    velocity_mode = result.bedrock_velocity_mode
    is_cell = velocity_mode == 'solve_cell'
    qc = {
        **result.qc,
        'layer_kind': layer_kind,
        'layer_index': layer_index,
        'velocity_mode': velocity_mode,
    }
    return RefractionLayerSolveResult(
        layer_kind=layer_kind,
        layer_index=layer_index,
        velocity_mode=velocity_mode,
        source_time_term_s=np.ascontiguousarray(
            result.source_half_intercept_time_s,
            dtype=np.float64,
        ),
        receiver_time_term_s=np.ascontiguousarray(
            result.receiver_half_intercept_time_s,
            dtype=np.float64,
        ),
        node_time_term_s=np.ascontiguousarray(
            result.node_half_intercept_time_s,
            dtype=np.float64,
        ),
        global_velocity_m_s=None if is_cell else float(result.bedrock_velocity_m_s),
        global_slowness_s_per_m=(
            None if is_cell else float(result.bedrock_slowness_s_per_m)
        ),
        cell_velocity_m_s=(
            None
            if result.cell_bedrock_velocity_m_s is None
            else np.ascontiguousarray(
                result.cell_bedrock_velocity_m_s,
                dtype=np.float64,
            )
        ),
        cell_slowness_s_per_m=(
            None
            if result.cell_bedrock_slowness_s_per_m is None
            else np.ascontiguousarray(
                result.cell_bedrock_slowness_s_per_m,
                dtype=np.float64,
            )
        ),
        trace_predicted_time_s_sorted=np.ascontiguousarray(
            result.estimated_first_break_time_s_sorted,
            dtype=np.float64,
        ),
        trace_residual_s_sorted=np.ascontiguousarray(
            result.first_break_residual_s_sorted,
            dtype=np.float64,
        ),
        used_observation_mask_sorted=np.ascontiguousarray(
            result.used_observation_mask_sorted,
            dtype=bool,
        ),
        layer_status='solved',
        qc=qc,
    )


def _validate_layer_result(
    *,
    result: RefractionLayerSolveResult,
    config: RefractionStaticLayerConfig,
) -> None:
    if result.layer_kind != config.kind:
        raise RefractionMultiLayerSolveError(
            f'layer solver returned {result.layer_kind} for requested {config.kind}'
        )
    if result.velocity_mode != config.velocity_mode:
        raise RefractionMultiLayerSolveError(
            f'layer solver returned velocity_mode={result.velocity_mode} for '
            f'requested {config.velocity_mode}'
        )
    if result.layer_status != 'solved':
        raise RefractionMultiLayerSolveError(
            f'refraction layer {config.kind} failed with status '
            f'{result.layer_status}'
        )


def _unique_endpoint_key_nodes(
    endpoint_key_sorted: np.ndarray,
    node_id_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(endpoint_key_sorted, dtype=object)
    nodes = np.asarray(node_id_sorted, dtype=np.int64)
    if keys.ndim != 1 or nodes.ndim != 1 or keys.shape != nodes.shape:
        raise RefractionMultiLayerSolveError('endpoint key/node arrays are invalid')
    seen: set[str] = set()
    positions: list[int] = []
    for index, key in enumerate(keys.tolist()):
        text = str(key)
        if text in seen:
            continue
        seen.add(text)
        positions.append(index)
    pos = np.asarray(positions, dtype=np.int64)
    return (
        np.ascontiguousarray(keys[pos], dtype=object),
        np.ascontiguousarray(nodes[pos], dtype=np.int64),
    )


def _layer_index(kind: RefractionLayerKind) -> int:
    try:
        return _LAYER_INDEX_BY_KIND[kind]
    except KeyError as exc:
        raise RefractionMultiLayerSolveError(
            f'unsupported refraction layer kind: {kind}'
        ) from exc


class _LayerSolveRequest:
    """Minimal request shim for the existing input-model solve path."""

    def __init__(
        self,
        model: RefractionStaticModelRequest,
        solver: RefractionStaticSolverRequest,
    ) -> None:
        self.model = model
        self.solver = solver


class _LayerSolveState:
    """Placeholder state for layer solves that already have an input model."""


__all__ = [
    'RefractionLayerSolver',
    'RefractionLayerSolverContext',
    'RefractionMultiLayerSolveError',
    'solve_refraction_multilayer_time_terms',
]
