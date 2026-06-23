"""Adapters from app-owned refraction requests to external core options."""

from __future__ import annotations

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

from app.statics.refraction.contracts.model import (
    RefractionStaticFirstLayerRequest,
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


__all__ = [
    'conversion_options_from_request',
    'datum_options_from_request',
    'first_layer_options_from_request',
    'layer_options_from_request',
    'model_options_from_request',
    'moveout_options_from_request',
    'reduced_time_qc_options_from_request',
    'refractor_cell_options_from_request',
    'robust_options_from_request',
    'solver_options_from_request',
]
