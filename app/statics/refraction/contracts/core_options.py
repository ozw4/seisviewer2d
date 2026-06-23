"""Adapters from app-owned refraction requests to external core options."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from seis_statics.refraction import (
    RefractionStaticConversionOptions,
    RefractionStaticDatumOptions,
    RefractionStaticFirstLayerOptions,
    RefractionStaticLayerOptions,
    RefractionStaticModelOptions,
    RefractionStaticMoveoutOptions,
    RefractionStaticReducedTimeQcOptions,
    RefractionStaticRefractorCellOptions,
    RefractionStaticRobustOptions,
    RefractionStaticSolverOptions,
)
from seis_statics.refraction.layer_config import (
    RefractionLayerConfig,
    RefractionLayerConfigLayer,
    normalize_refraction_layer_config,
)
from seis_statics.refraction.first_layer import (
    normalize_refraction_first_layer_request,
    resolve_weathering_velocity_m_s,
)
from seis_statics.refraction.layer_observations import (
    build_refraction_layer_observation_masks,
)
from seis_statics.refraction.types import (
    RefractionStaticInputModel as CoreRefractionStaticInputModel,
    RefractionLayerObservationMasks,
    ResolvedRefractionFirstLayer,
)

from app.statics.refraction.contracts.model import (
    RefractionStaticFirstLayerRequest,
    RefractionStaticLayerKind,
    RefractionStaticLayerRequest,
    RefractionStaticModelRequest,
    RefractionStaticRefractorCellRequest,
)
from app.statics.refraction.contracts.options import (
    RefractionStaticConversionRequest,
    RefractionStaticDatumRequest,
    RefractionStaticMoveoutRequest,
    RefractionStaticReducedTimeQcRequest,
    RefractionStaticRobustRequest,
    RefractionStaticSolverRequest,
)


def first_layer_options_from_request(
    request: RefractionStaticFirstLayerRequest,
) -> RefractionStaticFirstLayerOptions:
    """Convert a first-layer Pydantic request to external core options."""
    return RefractionStaticFirstLayerOptions(
        mode=request.mode,
        weathering_velocity_m_s=request.weathering_velocity_m_s,
        min_weathering_velocity_m_s=request.min_weathering_velocity_m_s,
        max_weathering_velocity_m_s=request.max_weathering_velocity_m_s,
        min_direct_offset_m=request.min_direct_offset_m,
        max_direct_offset_m=request.max_direct_offset_m,
        min_picks_per_fit=request.min_picks_per_fit,
        min_groups=request.min_groups,
        robust_enabled=request.robust_enabled,
        robust_threshold=request.robust_threshold,
    )


def refractor_cell_options_from_request(
    request: RefractionStaticRefractorCellRequest,
) -> RefractionStaticRefractorCellOptions:
    """Convert a refractor-cell Pydantic request to external core options."""
    return RefractionStaticRefractorCellOptions(
        number_of_cell_x=request.number_of_cell_x,
        size_of_cell_x_m=request.size_of_cell_x_m,
        x_coordinate_origin_m=request.x_coordinate_origin_m,
        number_of_cell_y=request.number_of_cell_y,
        size_of_cell_y_m=request.size_of_cell_y_m,
        y_coordinate_origin_m=request.y_coordinate_origin_m,
        assignment_mode=request.assignment_mode,
        outside_grid_policy=request.outside_grid_policy,
        coordinate_mode=request.coordinate_mode,
        line_origin_x_m=request.line_origin_x_m,
        line_origin_y_m=request.line_origin_y_m,
        line_azimuth_deg=request.line_azimuth_deg,
        min_observations_per_cell=request.min_observations_per_cell,
        velocity_smoothing_weight=request.velocity_smoothing_weight,
        smoothing_reference_distance_m=request.smoothing_reference_distance_m,
    )


def layer_options_from_request(
    request: RefractionStaticLayerRequest,
) -> RefractionStaticLayerOptions:
    """Convert one layer Pydantic request to external core options."""
    return RefractionStaticLayerOptions(
        kind=request.kind,
        enabled=request.enabled,
        min_offset_m=request.min_offset_m,
        max_offset_m=request.max_offset_m,
        velocity_mode=request.velocity_mode,
        initial_velocity_m_s=request.initial_velocity_m_s,
        fixed_velocity_m_s=request.fixed_velocity_m_s,
        min_velocity_m_s=request.min_velocity_m_s,
        max_velocity_m_s=request.max_velocity_m_s,
        min_observations_per_cell=request.min_observations_per_cell,
        smoothing_weight=request.smoothing_weight,
    )


def model_options_from_request(
    request: RefractionStaticModelRequest,
) -> RefractionStaticModelOptions:
    """Convert model Pydantic request to external core options."""
    return RefractionStaticModelOptions(
        method=request.method,
        weathering_velocity_m_s=request.weathering_velocity_m_s,
        first_layer=(
            first_layer_options_from_request(request.first_layer)
            if request.first_layer is not None
            else None
        ),
        bedrock_velocity_mode=request.bedrock_velocity_mode,
        bedrock_velocity_m_s=request.bedrock_velocity_m_s,
        initial_bedrock_velocity_m_s=request.initial_bedrock_velocity_m_s,
        min_bedrock_velocity_m_s=request.min_bedrock_velocity_m_s,
        max_bedrock_velocity_m_s=request.max_bedrock_velocity_m_s,
        max_weathering_thickness_m=request.max_weathering_thickness_m,
        refractor_cell=(
            refractor_cell_options_from_request(request.refractor_cell)
            if request.refractor_cell is not None
            else None
        ),
        layers=(
            tuple(layer_options_from_request(layer) for layer in request.layers)
            if request.layers is not None
            else None
        ),
        layer_assignment_policy=(
            'independent' if request.allow_overlapping_layer_gates else 'reject_overlap'
        ),
    )


def layer_config_from_model_request(
    request: RefractionStaticModelRequest,
) -> RefractionLayerConfig:
    """Convert a model request to the external normalized layer config."""
    if request.method != 'multilayer_time_term':
        return RefractionLayerConfig(
            layers=(_legacy_v2_layer_from_model_request(request),),
            assignment_policy='reject_overlap',
        )
    return normalize_refraction_layer_config(model_options_from_request(request))


def normalized_layers_from_model_request(
    request: RefractionStaticModelRequest,
) -> tuple[RefractionLayerConfigLayer, ...]:
    """Return enabled external layer configs for an app model request."""
    return layer_config_from_model_request(request).layers


def _legacy_v2_layer_from_model_request(
    request: RefractionStaticModelRequest,
) -> RefractionLayerConfigLayer:
    refractor_cell = request.refractor_cell
    return RefractionLayerConfigLayer(
        kind='v2_t1',
        min_offset_m=None,
        max_offset_m=None,
        velocity_mode=request.bedrock_velocity_mode,
        initial_velocity_m_s=request.initial_bedrock_velocity_m_s,
        fixed_velocity_m_s=(
            request.bedrock_velocity_m_s
            if request.bedrock_velocity_mode == 'fixed_global'
            else None
        ),
        min_velocity_m_s=request.min_bedrock_velocity_m_s,
        max_velocity_m_s=request.max_bedrock_velocity_m_s,
        min_observations_per_cell=(
            refractor_cell.min_observations_per_cell
            if request.bedrock_velocity_mode == 'solve_cell'
            and refractor_cell is not None
            else None
        ),
        smoothing_weight=(
            refractor_cell.velocity_smoothing_weight
            if request.bedrock_velocity_mode == 'solve_cell'
            and refractor_cell is not None
            else None
        ),
    )


def layer_observation_masks_from_input_model(
    *,
    input_model: object,
    model: RefractionStaticModelRequest,
) -> RefractionLayerObservationMasks:
    """Build external layer masks from an app input model."""
    offset = getattr(input_model, 'distance_m_sorted', None)
    if offset is None:
        offset = getattr(input_model, 'offset_m_sorted', None)
    return layer_observation_masks_from_arrays(
        base_valid_mask_sorted=getattr(input_model, 'valid_observation_mask_sorted'),
        offset_m_sorted=offset,
        rejection_reason_sorted=getattr(input_model, 'rejection_reason_sorted'),
        model=model,
    )


def layer_observation_masks_from_arrays(
    *,
    base_valid_mask_sorted: object,
    offset_m_sorted: object,
    rejection_reason_sorted: object,
    model: RefractionStaticModelRequest,
) -> RefractionLayerObservationMasks:
    """Build external layer masks from primitive observation arrays."""
    masks = build_refraction_layer_observation_masks(
        layer_config=layer_config_from_model_request(model),
        offset_m_sorted=offset_m_sorted,
        valid_observation_mask_sorted=base_valid_mask_sorted,
        rejection_reason_sorted=rejection_reason_sorted,
    )
    return _layer_masks_with_viewer_ok_reason(masks)


def _layer_masks_with_viewer_ok_reason(
    masks: RefractionLayerObservationMasks,
) -> RefractionLayerObservationMasks:
    reasons: dict[str, np.ndarray] = {}
    for kind, raw_reason in masks.layer_rejection_reason_sorted.items():
        reason = np.asarray(raw_reason).astype('<U64', copy=True)
        reason[reason == ''] = 'ok'
        reasons[kind] = np.ascontiguousarray(reason)
    return replace(masks, layer_rejection_reason_sorted=reasons)


_MULTILAYER_LAYER_KINDS: tuple[RefractionStaticLayerKind, ...] = (
    'v2_t1',
    'v3_t2',
    'vsub_t3',
)


def layer_observation_qc_for_viewer(
    masks: RefractionLayerObservationMasks,
    *,
    model: RefractionStaticModelRequest | None = None,
) -> dict[str, dict[str, object]]:
    """Format external layer masks in the viewer's existing per-layer QC shape."""
    kinds = [str(value) for value in masks.layer_kind.tolist()]
    enabled = list(np.asarray(masks.layer_enabled, dtype=bool).tolist())
    min_offset = np.asarray(masks.layer_min_offset_m, dtype=np.float64)
    max_offset = np.asarray(masks.layer_max_offset_m, dtype=np.float64)
    payload: dict[str, dict[str, object]] = {}
    for index, kind in enumerate(kinds):
        reasons = np.asarray(masks.layer_rejection_reason_sorted[kind]).astype(
            str,
            copy=False,
        )
        payload[kind] = {
            'enabled': bool(enabled[index]),
            'n_candidate_observations': int(masks.layer_candidate_count[kind]),
            'n_used_observations': int(masks.layer_observation_count[kind]),
            'min_offset_m': _json_optional_gate_value(min_offset[index]),
            'max_offset_m': _json_optional_gate_value(max_offset[index]),
            'rejection_counts': _reason_counts(reasons),
        }
    if model is None or model.method != 'multilayer_time_term':
        return payload
    return _with_multilayer_viewer_qc_slots(payload, model=model, masks=masks)


def _with_multilayer_viewer_qc_slots(
    payload: dict[str, dict[str, object]],
    *,
    model: RefractionStaticModelRequest,
    masks: RefractionLayerObservationMasks,
) -> dict[str, dict[str, object]]:
    layer_by_kind = {layer.kind: layer for layer in model.layers or ()}
    row_count = _layer_mask_row_count(masks)
    ordered: dict[str, dict[str, object]] = {}
    for kind in _MULTILAYER_LAYER_KINDS:
        layer = layer_by_kind.get(kind)
        if kind in payload and layer is not None:
            entry = dict(payload[kind])
            entry['enabled'] = bool(layer.enabled)
            entry['min_offset_m'] = _json_request_gate_value(layer.min_offset_m)
            entry['max_offset_m'] = _json_request_gate_value(layer.max_offset_m)
            ordered[kind] = entry
            continue
        if layer is None:
            ordered[kind] = _disabled_layer_viewer_qc(
                row_count=row_count,
                reason='layer_not_configured',
                min_offset_m=None,
                max_offset_m=None,
            )
            continue
        ordered[kind] = _disabled_layer_viewer_qc(
            row_count=row_count,
            reason='layer_disabled',
            min_offset_m=layer.min_offset_m,
            max_offset_m=layer.max_offset_m,
        )
    return ordered


def _disabled_layer_viewer_qc(
    *,
    row_count: int,
    reason: str,
    min_offset_m: float | None,
    max_offset_m: float | None,
) -> dict[str, object]:
    rejection_counts: dict[str, int] = {}
    if row_count > 0:
        rejection_counts[reason] = row_count
    return {
        'enabled': False,
        'n_candidate_observations': 0,
        'n_used_observations': 0,
        'min_offset_m': _json_request_gate_value(min_offset_m),
        'max_offset_m': _json_request_gate_value(max_offset_m),
        'rejection_counts': rejection_counts,
    }


def _layer_mask_row_count(masks: RefractionLayerObservationMasks) -> int:
    for values in masks.layer_rejection_reason_sorted.values():
        return int(np.asarray(values).shape[0])
    for values in masks.layer_used_mask_sorted.values():
        return int(np.asarray(values).shape[0])
    return 0


def _json_optional_gate_value(value: float) -> float | None:
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _json_request_gate_value(value: float | None) -> float | None:
    if value is None:
        return None
    return _json_optional_gate_value(float(value))


def _reason_counts(values: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in values.astype(str, copy=False).tolist():
        reason = str(raw)
        out[reason] = out.get(reason, 0) + 1
    return dict(sorted(out.items()))


def normalize_first_layer_from_model_request(
    request: RefractionStaticModelRequest,
) -> ResolvedRefractionFirstLayer:
    """Resolve constant first-layer request fields through the external core."""
    return normalize_refraction_first_layer_request(
        _first_layer_model_context(request)
    )


def resolve_weathering_velocity_from_model_request(
    *,
    model: RefractionStaticModelRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    name: str = 'model.resolved_weathering_velocity_m_s',
) -> float:
    """Resolve V1 through the external core using external model options."""
    return resolve_weathering_velocity_m_s(
        model=_first_layer_model_context(model),
        resolved_first_layer=resolved_first_layer,
        name=name,
    )


def _first_layer_model_context(request: RefractionStaticModelRequest) -> object:
    try:
        resolved_velocity = request.resolved_weathering_velocity_m_s
    except ValueError as exc:
        return _FirstLayerEstimateContext(request.first_layer_mode, exc)
    return SimpleNamespace(
        first_layer_mode=request.first_layer_mode,
        resolved_weathering_velocity_m_s=resolved_velocity,
    )


class _FirstLayerEstimateContext:
    def __init__(self, first_layer_mode: str, error: ValueError) -> None:
        self.first_layer_mode = first_layer_mode
        self._error = error

    @property
    def resolved_weathering_velocity_m_s(self) -> Any:
        raise self._error


def moveout_options_from_request(
    request: RefractionStaticMoveoutRequest,
) -> RefractionStaticMoveoutOptions:
    """Convert moveout Pydantic request to external core options."""
    return RefractionStaticMoveoutOptions(
        model=request.model,
        distance_source=request.distance_source,
        offset_byte=request.offset_byte,
        min_offset_m=request.min_offset_m,
        max_offset_m=request.max_offset_m,
        allow_missing_offset=request.allow_missing_offset,
        max_geometry_offset_mismatch_m=request.max_geometry_offset_mismatch_m,
    )


def robust_options_from_request(
    request: RefractionStaticRobustRequest,
) -> RefractionStaticRobustOptions:
    """Convert robust-solver Pydantic request to external core options."""
    return RefractionStaticRobustOptions(
        enabled=request.enabled,
        method=request.method,
        threshold=request.threshold,
        scale_floor_ms=request.scale_floor_ms,
        max_iterations=request.max_iterations,
        min_used_fraction=request.min_used_fraction,
        min_used_observations=request.min_used_observations,
    )


def solver_options_from_request(
    request: RefractionStaticSolverRequest,
) -> RefractionStaticSolverOptions:
    """Convert solver Pydantic request to external core options."""
    return RefractionStaticSolverOptions(
        half_intercept_damping_lambda=request.damping,
        min_picks_per_node=request.min_picks_per_node,
        max_abs_half_intercept_time_ms=request.max_abs_half_intercept_time_ms,
        robust=robust_options_from_request(request.robust),
    )


def datum_options_from_request(
    request: RefractionStaticDatumRequest,
) -> RefractionStaticDatumOptions:
    """Convert datum Pydantic request to external core options."""
    floating_datum_mode = request.floating_datum_mode
    if floating_datum_mode == 'from_artifact':
        floating_datum_mode = 'provided'
    return RefractionStaticDatumOptions(
        mode=request.mode,
        floating_datum_mode=floating_datum_mode,
        flat_datum_elevation_m=request.flat_datum_elevation_m,
        floating_datum_elevation_m=request.floating_datum_elevation_m,
        smoothing_radius_m=request.smoothing_radius_m,
        smoothing_window_nodes=request.smoothing_window_nodes,
        smoothing_method=request.smoothing_method,
        allow_flat_datum_above_topography=request.allow_flat_datum_above_topography,
        allow_flat_datum_below_refractor=request.allow_flat_datum_below_refractor,
    )


def conversion_options_from_request(
    request: RefractionStaticConversionRequest,
) -> RefractionStaticConversionOptions:
    """Convert conversion Pydantic request to external core options."""
    return RefractionStaticConversionOptions(
        mode=request.mode,
        layer_count=request.layer_count,
    )


def reduced_time_qc_options_from_request(
    request: RefractionStaticReducedTimeQcRequest,
) -> RefractionStaticReducedTimeQcOptions:
    """Convert reduced-time QC Pydantic request to external core options."""
    return RefractionStaticReducedTimeQcOptions(
        reduction_velocity_mode=request.reduction_velocity_mode,
        fixed_velocity_m_s=request.fixed_velocity_m_s,
    )


def core_input_model_from_app(input_model: object) -> CoreRefractionStaticInputModel:
    """Convert the app input model container to the external core dataclass."""
    return CoreRefractionStaticInputModel(
        file_id=getattr(input_model, 'file_id'),
        n_traces=getattr(input_model, 'n_traces'),
        sorted_trace_index=getattr(input_model, 'sorted_trace_index'),
        pick_time_s_sorted=getattr(input_model, 'pick_time_s_sorted'),
        valid_pick_mask_sorted=getattr(input_model, 'valid_pick_mask_sorted'),
        valid_observation_mask_sorted=getattr(
            input_model,
            'valid_observation_mask_sorted',
        ),
        source_id_sorted=getattr(input_model, 'source_id_sorted'),
        receiver_id_sorted=getattr(input_model, 'receiver_id_sorted'),
        source_x_m_sorted=getattr(input_model, 'source_x_m_sorted'),
        source_y_m_sorted=getattr(input_model, 'source_y_m_sorted'),
        receiver_x_m_sorted=getattr(input_model, 'receiver_x_m_sorted'),
        receiver_y_m_sorted=getattr(input_model, 'receiver_y_m_sorted'),
        source_elevation_m_sorted=getattr(input_model, 'source_elevation_m_sorted'),
        receiver_elevation_m_sorted=getattr(
            input_model,
            'receiver_elevation_m_sorted',
        ),
        source_depth_m_sorted=getattr(input_model, 'source_depth_m_sorted'),
        geometry_distance_m_sorted=getattr(input_model, 'geometry_distance_m_sorted'),
        offset_m_sorted=getattr(input_model, 'offset_m_sorted'),
        distance_m_sorted=getattr(input_model, 'distance_m_sorted'),
        source_endpoint_key_sorted=getattr(
            input_model,
            'source_endpoint_key_sorted',
        ),
        receiver_endpoint_key_sorted=getattr(
            input_model,
            'receiver_endpoint_key_sorted',
        ),
        source_node_id_sorted=getattr(input_model, 'source_node_id_sorted'),
        receiver_node_id_sorted=getattr(input_model, 'receiver_node_id_sorted'),
        node_x_m=getattr(input_model, 'node_x_m'),
        node_y_m=getattr(input_model, 'node_y_m'),
        node_elevation_m=getattr(input_model, 'node_elevation_m'),
        node_kind=getattr(input_model, 'node_kind'),
        rejection_reason_sorted=getattr(input_model, 'rejection_reason_sorted'),
        qc=getattr(input_model, 'qc'),
        endpoint_table=getattr(input_model, 'endpoint_table'),
        metadata=getattr(input_model, 'metadata'),
        layer_observation_masks=getattr(input_model, 'layer_observation_masks', None),
        source_endpoint_id_sorted=getattr(
            input_model,
            'source_endpoint_id_sorted',
            None,
        ),
        receiver_endpoint_id_sorted=getattr(
            input_model,
            'receiver_endpoint_id_sorted',
            None,
        ),
    )


__all__ = [
    'conversion_options_from_request',
    'core_input_model_from_app',
    'datum_options_from_request',
    'first_layer_options_from_request',
    'layer_config_from_model_request',
    'layer_observation_masks_from_arrays',
    'layer_observation_masks_from_input_model',
    'layer_observation_qc_for_viewer',
    'layer_options_from_request',
    'model_options_from_request',
    'moveout_options_from_request',
    'normalize_first_layer_from_model_request',
    'normalized_layers_from_model_request',
    'reduced_time_qc_options_from_request',
    'refractor_cell_options_from_request',
    'resolve_weathering_velocity_from_model_request',
    'robust_options_from_request',
    'solver_options_from_request',
]
