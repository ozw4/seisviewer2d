"""Sequential multi-layer time-term orchestration for refraction statics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from app.api.schemas import (
    RefractionStaticApplyOptions,
    RefractionStaticDatumRequest,
    RefractionStaticModelRequest,
    RefractionStaticSolverRequest,
)
from app.core.state import AppState
from app.services.refraction_static_cell_coordinates import (
    effective_refraction_cell_grid_config,
    project_refraction_cell_points,
)
from app.services.refraction_static_cell_grid import (
    RefractionCellGrid,
    assign_observation_midpoint_cells,
    assign_points_to_refraction_cells,
    build_refraction_cell_grid,
)
from app.services.refraction_static_datum import (
    build_refraction_datum_statics,
    write_refraction_datum_statics_artifacts,
)
from app.services.refraction_static_half_intercept import (
    estimate_refraction_half_intercept_times_from_first_breaks,
)
from app.services.refraction_static_layer_config import (
    RefractionStaticLayerConfig,
    normalize_refraction_static_layers,
)
from app.services.refraction_static_layer_observations import (
    build_refraction_layer_observation_masks,
    refraction_layer_observation_qc,
)
from app.services.refraction_static_t1lsst import (
    compute_t1lsst_2layer_thicknesses_with_status,
    compute_t1lsst_2layer_weathering_correction,
)
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
    RefractionHalfInterceptTimeResult,
    RefractionLayerKind,
    RefractionLayerObservationMasks,
    RefractionLayerSolveResult,
    RefractionLayerVelocityMode,
    RefractionMultiLayerStaticComponents,
    RefractionMultiLayerSolveResult,
    RefractionStaticInputModel,
    RefractionWeatheringReplacementStaticsResult,
    ResolvedRefractionFirstLayer,
)

_LAYER_INDEX_BY_KIND: dict[RefractionLayerKind, int] = {
    'v2_t1': 1,
    'v3_t2': 2,
    'vsub_t3': 3,
}
_STATUS_DTYPE = '<U32'
_SIGN_CONVENTION_TEXT = 'corrected(t) = raw(t - shift_s)'


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


@dataclass(frozen=True)
class RefractionMultiLayerStaticsWorkflowResult:
    """Production outputs for a multi-layer statics workflow."""

    solve_result: RefractionMultiLayerSolveResult
    components: RefractionMultiLayerStaticComponents
    weathering_replacement_result: RefractionWeatheringReplacementStaticsResult
    datum_result: RefractionDatumStaticsResult


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
    """Run enabled refraction layer solves in configured order."""
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
        result = _validate_layer_velocity_sequence(
            result=result,
            previous_results=tuple(layer_results),
            normalized_layers=normalized_layers,
        )
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


def compute_refraction_multilayer_datum_statics_from_input_model(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    datum: RefractionStaticDatumRequest,
    apply_options: RefractionStaticApplyOptions | None,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    job_dir: Path | None = None,
    state: AppState | None = None,
    file_id: str | None = None,
    key1_byte: int | None = None,
    key2_byte: int | None = None,
    floating_datum_artifact_path: Path | None = None,
) -> RefractionMultiLayerStaticsWorkflowResult:
    """Run the implemented two-layer time-term, T1LSST, and datum workflow."""
    normalized_layers = normalize_refraction_static_layers(model)
    _require_two_layer_t1lsst_layers(normalized_layers)
    layer_masks = build_refraction_layer_observation_masks(
        input_model=input_model,
        model=model,
    )
    solve_result = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        resolved_first_layer=resolved_first_layer,
        normalized_layers=normalized_layers,
        layer_masks=layer_masks,
        model=model,
        solver=solver,
    )
    weathering_replacement = build_refraction_multilayer_weathering_replacement_statics(
        input_model=input_model,
        model=model,
        solve_result=solve_result,
        apply_options=apply_options,
        resolved_first_layer=resolved_first_layer,
    )
    datum_result = build_refraction_datum_statics(
        weathering_replacement_result=weathering_replacement,
        datum=datum,
        apply_options=apply_options,
        job_dir=None,
        state=state,
        file_id=file_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        floating_datum_artifact_path=floating_datum_artifact_path,
        resolved_first_layer=resolved_first_layer,
    )
    components = _components_from_replacement(weathering_replacement)
    datum_result = replace(
        datum_result,
        qc={
            **datum_result.qc,
            'method': 'multilayer_time_term',
            'conversion_mode': 't1lsst_multilayer',
            'layer_count': 2,
            'layers': solve_result.qc,
        },
        source_t2_time_s=components.source_t2_s,
        source_v3_m_s=_required_result_array(
            weathering_replacement.source_v3_m_s,
            name='source_v3_m_s',
        ),
        source_sh2_weathering_thickness_m=components.source_sh2_m,
        receiver_t2_time_s=components.receiver_t2_s,
        receiver_v3_m_s=_required_result_array(
            weathering_replacement.receiver_v3_m_s,
            name='receiver_v3_m_s',
        ),
        receiver_sh2_weathering_thickness_m=components.receiver_sh2_m,
    )
    if job_dir is not None:
        write_refraction_datum_statics_artifacts(Path(job_dir), datum_result)
    return RefractionMultiLayerStaticsWorkflowResult(
        solve_result=solve_result,
        components=components,
        weathering_replacement_result=weathering_replacement,
        datum_result=datum_result,
    )


def build_refraction_multilayer_weathering_replacement_statics(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    solve_result: RefractionMultiLayerSolveResult,
    apply_options: RefractionStaticApplyOptions | None,
    resolved_first_layer: ResolvedRefractionFirstLayer,
) -> RefractionWeatheringReplacementStaticsResult:
    """Build two-layer T1LSST replacement statics from production layer solves."""
    _require_two_layer_t1lsst_layers(normalize_refraction_static_layers(model))
    v2_layer = _required_layer_result(solve_result, 'v2_t1')
    v3_layer = _required_layer_result(solve_result, 'v3_t2')
    v1_m_s = _positive_float(
        resolved_first_layer.weathering_velocity_m_s,
        name='resolved_first_layer.weathering_velocity_m_s',
    )
    v3_m_s = _required_global_velocity(v3_layer)
    node_id = _input_node_id(input_model)
    source = _endpoint_metadata(input_model, endpoint='source')
    receiver = _endpoint_metadata(input_model, endpoint='receiver')
    _validate_endpoint_order(
        actual=source.endpoint_key,
        expected=solve_result.source_endpoint_key,
        name='source_endpoint_key',
    )
    _validate_endpoint_order(
        actual=receiver.endpoint_key,
        expected=solve_result.receiver_endpoint_key,
        name='receiver_endpoint_key',
    )

    v2 = _build_v2_static_model(
        input_model=input_model,
        model=model,
        layer=v2_layer,
        node_id=node_id,
        source=source,
        receiver=receiver,
        v1_m_s=v1_m_s,
    )
    node_t1 = _layer_node_terms(v2_layer, shape=node_id.shape, name='v2_t1')
    node_t2 = _layer_node_terms(v3_layer, shape=node_id.shape, name='v3_t2')
    source_t1 = _endpoint_terms(v2_layer.source_time_term_s, source, name='source_t1')
    source_t2 = _endpoint_terms(v3_layer.source_time_term_s, source, name='source_t2')
    receiver_t1 = _endpoint_terms(
        v2_layer.receiver_time_term_s,
        receiver,
        name='receiver_t1',
    )
    receiver_t2 = _endpoint_terms(
        v3_layer.receiver_time_term_s,
        receiver,
        name='receiver_t2',
    )

    node_thickness = compute_t1lsst_2layer_thicknesses_with_status(
        t1_s=node_t1,
        t2_s=node_t2,
        v1_m_s=v1_m_s,
        v2_m_s=v2.node_v2_m_s,
        v3_m_s=v3_m_s,
    )
    source_thickness = compute_t1lsst_2layer_thicknesses_with_status(
        t1_s=source_t1,
        t2_s=source_t2,
        v1_m_s=v1_m_s,
        v2_m_s=v2.source_v2_m_s,
        v3_m_s=v3_m_s,
    )
    receiver_thickness = compute_t1lsst_2layer_thicknesses_with_status(
        t1_s=receiver_t1,
        t2_s=receiver_t2,
        v1_m_s=v1_m_s,
        v2_m_s=v2.receiver_v2_m_s,
        v3_m_s=v3_m_s,
    )
    node_wcor = compute_t1lsst_2layer_weathering_correction(
        sh1_m=node_thickness.sh1_m,
        sh2_m=node_thickness.sh2_m,
        v1_m_s=v1_m_s,
        v2_m_s=v2.node_v2_m_s,
        v3_m_s=v3_m_s,
    )
    source_wcor = compute_t1lsst_2layer_weathering_correction(
        sh1_m=source_thickness.sh1_m,
        sh2_m=source_thickness.sh2_m,
        v1_m_s=v1_m_s,
        v2_m_s=v2.source_v2_m_s,
        v3_m_s=v3_m_s,
    )
    receiver_wcor = compute_t1lsst_2layer_weathering_correction(
        sh1_m=receiver_thickness.sh1_m,
        sh2_m=receiver_thickness.sh2_m,
        v1_m_s=v1_m_s,
        v2_m_s=v2.receiver_v2_m_s,
        v3_m_s=v3_m_s,
    )

    max_abs_shift_ms = (
        None if apply_options is None else float(apply_options.max_abs_shift_ms)
    )
    node_static_status = _status_from_conversion(
        node_thickness.status,
        node_wcor,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    source_static_status = _status_from_conversion(
        source_thickness.status,
        source_wcor,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    receiver_static_status = _status_from_conversion(
        receiver_thickness.status,
        receiver_wcor,
        max_abs_shift_ms=max_abs_shift_ms,
    )

    source_t1_sorted = _map_endpoint_values_to_trace_order(
        endpoint_key_sorted=input_model.source_endpoint_key_sorted,
        endpoint_key=source.endpoint_key,
        endpoint_values=source_t1,
        name='source_t1_s_sorted',
    )
    receiver_t1_sorted = _map_endpoint_values_to_trace_order(
        endpoint_key_sorted=input_model.receiver_endpoint_key_sorted,
        endpoint_key=receiver.endpoint_key,
        endpoint_values=receiver_t1,
        name='receiver_t1_s_sorted',
    )
    source_sh1_sorted = _map_endpoint_values_to_trace_order(
        endpoint_key_sorted=input_model.source_endpoint_key_sorted,
        endpoint_key=source.endpoint_key,
        endpoint_values=source_thickness.sh1_m,
        name='source_sh1_m_sorted',
    )
    receiver_sh1_sorted = _map_endpoint_values_to_trace_order(
        endpoint_key_sorted=input_model.receiver_endpoint_key_sorted,
        endpoint_key=receiver.endpoint_key,
        endpoint_values=receiver_thickness.sh1_m,
        name='receiver_sh1_m_sorted',
    )
    source_wcor_sorted = _map_endpoint_values_to_trace_order(
        endpoint_key_sorted=input_model.source_endpoint_key_sorted,
        endpoint_key=source.endpoint_key,
        endpoint_values=source_wcor,
        name='source_wcor_s_sorted',
    )
    receiver_wcor_sorted = _map_endpoint_values_to_trace_order(
        endpoint_key_sorted=input_model.receiver_endpoint_key_sorted,
        endpoint_key=receiver.endpoint_key,
        endpoint_values=receiver_wcor,
        name='receiver_wcor_s_sorted',
    )
    source_status_sorted = _map_endpoint_strings_to_trace_order(
        endpoint_key_sorted=input_model.source_endpoint_key_sorted,
        endpoint_key=source.endpoint_key,
        endpoint_values=source_static_status,
        name='source_static_status_sorted',
    )
    receiver_status_sorted = _map_endpoint_strings_to_trace_order(
        endpoint_key_sorted=input_model.receiver_endpoint_key_sorted,
        endpoint_key=receiver.endpoint_key,
        endpoint_values=receiver_static_status,
        name='receiver_static_status_sorted',
    )
    trace_shift = _combine_trace_shifts(source_wcor_sorted, receiver_wcor_sorted)
    trace_status = _trace_static_status(
        source_status=source_status_sorted,
        receiver_status=receiver_status_sorted,
        trace_shift_s=trace_shift,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    trace_valid = (trace_status == 'ok') & np.isfinite(trace_shift)
    modeled = _combined_modeled_pick_time(solve_result, input_model.n_traces)
    residual = np.ascontiguousarray(
        input_model.pick_time_s_sorted - modeled,
        dtype=np.float64,
    )
    used = _combined_used_mask(solve_result, input_model.n_traces)
    node_residual_rms, node_residual_mad = _node_residual_stats(
        input_model=input_model,
        node_id=node_id,
        residual_s=residual,
        used_mask=used,
    )
    node_pick_count, node_used_pick_count = _node_pick_counts(
        input_model=input_model,
        node_id=node_id,
        used_mask=used,
    )
    qc = {
        'method': 'multilayer_time_term',
        'static_component': 't1lsst_multilayer_weathering_replacement',
        'conversion_mode': 't1lsst_multilayer',
        'layer_count': 2,
        'sign_convention': _SIGN_CONVENTION_TEXT,
        'bedrock_velocity_mode': v3_layer.velocity_mode,
        'weathering_velocity_m_s': float(v1_m_s),
        'v2_velocity_mode': v2_layer.velocity_mode,
        'v3_velocity_mode': v3_layer.velocity_mode,
        'v3_m_s': float(v3_m_s),
        'layers': solve_result.qc,
    }
    source_v3_m_s = np.full(source.endpoint_key.shape, v3_m_s, dtype=np.float64)
    receiver_v3_m_s = np.full(receiver.endpoint_key.shape, v3_m_s, dtype=np.float64)
    return RefractionWeatheringReplacementStaticsResult(
        bedrock_velocity_mode=v3_layer.velocity_mode,
        bedrock_slowness_s_per_m=1.0 / v3_m_s,
        bedrock_velocity_m_s=v3_m_s,
        weathering_velocity_m_s=v1_m_s,
        replacement_slowness_delta_s_per_m=1.0 / v3_m_s - 1.0 / v1_m_s,
        node_id=node_id,
        node_x_m=np.ascontiguousarray(input_model.node_x_m, dtype=np.float64),
        node_y_m=np.ascontiguousarray(input_model.node_y_m, dtype=np.float64),
        node_surface_elevation_m=np.ascontiguousarray(
            input_model.node_elevation_m,
            dtype=np.float64,
        ),
        node_kind=np.asarray(input_model.node_kind).astype('<U16', copy=True),
        node_weathering_thickness_m=node_thickness.sh1_m,
        node_refractor_elevation_m=(
            np.ascontiguousarray(input_model.node_elevation_m, dtype=np.float64)
            - node_thickness.sh1_m
        ),
        node_half_intercept_time_s=node_t1,
        node_solution_status=np.full(node_id.shape, 'solved', dtype=_STATUS_DTYPE),
        node_weathering_status=node_thickness.status,
        node_weathering_replacement_shift_s=node_wcor,
        node_weathering_replacement_shift_ms=node_wcor * 1000.0,
        node_static_status=node_static_status,
        node_pick_count=node_pick_count,
        node_used_pick_count=node_used_pick_count,
        node_rejected_pick_count=node_pick_count - node_used_pick_count,
        node_residual_rms_s=node_residual_rms,
        node_residual_mad_s=node_residual_mad,
        source_endpoint_key=source.endpoint_key,
        source_id=source.endpoint_id,
        source_node_id=source.node_id,
        source_x_m=source.x_m,
        source_y_m=source.y_m,
        source_surface_elevation_m=source.elevation_m,
        source_half_intercept_time_s=source_t1,
        source_weathering_thickness_m=source_thickness.sh1_m,
        source_refractor_elevation_m=source.elevation_m - source_thickness.sh1_m,
        source_weathering_replacement_shift_s=source_wcor,
        source_static_status=source_static_status,
        receiver_endpoint_key=receiver.endpoint_key,
        receiver_id=receiver.endpoint_id,
        receiver_node_id=receiver.node_id,
        receiver_x_m=receiver.x_m,
        receiver_y_m=receiver.y_m,
        receiver_surface_elevation_m=receiver.elevation_m,
        receiver_half_intercept_time_s=receiver_t1,
        receiver_weathering_thickness_m=receiver_thickness.sh1_m,
        receiver_refractor_elevation_m=(
            receiver.elevation_m - receiver_thickness.sh1_m
        ),
        receiver_weathering_replacement_shift_s=receiver_wcor,
        receiver_static_status=receiver_static_status,
        sorted_trace_index=np.ascontiguousarray(
            input_model.sorted_trace_index,
            dtype=np.int64,
        ),
        valid_observation_mask_sorted=np.ascontiguousarray(
            input_model.valid_observation_mask_sorted,
            dtype=bool,
        ),
        used_observation_mask_sorted=used,
        source_endpoint_key_sorted=np.asarray(
            input_model.source_endpoint_key_sorted,
            dtype=object,
        ),
        receiver_endpoint_key_sorted=np.asarray(
            input_model.receiver_endpoint_key_sorted,
            dtype=object,
        ),
        source_node_id_sorted=np.ascontiguousarray(
            input_model.source_node_id_sorted,
            dtype=np.int64,
        ),
        receiver_node_id_sorted=np.ascontiguousarray(
            input_model.receiver_node_id_sorted,
            dtype=np.int64,
        ),
        source_half_intercept_time_s_sorted=source_t1_sorted,
        receiver_half_intercept_time_s_sorted=receiver_t1_sorted,
        source_weathering_thickness_m_sorted=source_sh1_sorted,
        receiver_weathering_thickness_m_sorted=receiver_sh1_sorted,
        source_refractor_elevation_m_sorted=(
            np.ascontiguousarray(input_model.source_elevation_m_sorted, dtype=np.float64)
            - source_sh1_sorted
        ),
        receiver_refractor_elevation_m_sorted=(
            np.ascontiguousarray(
                input_model.receiver_elevation_m_sorted,
                dtype=np.float64,
            )
            - receiver_sh1_sorted
        ),
        source_weathering_replacement_shift_s_sorted=source_wcor_sorted,
        receiver_weathering_replacement_shift_s_sorted=receiver_wcor_sorted,
        weathering_replacement_trace_shift_s_sorted=trace_shift,
        source_static_status_sorted=source_status_sorted,
        receiver_static_status_sorted=receiver_status_sorted,
        trace_static_status_sorted=trace_status,
        trace_static_valid_mask_sorted=trace_valid,
        estimated_first_break_time_s_sorted=modeled,
        first_break_residual_s_sorted=residual,
        row_trace_index_sorted=np.ascontiguousarray(
            input_model.sorted_trace_index,
            dtype=np.int64,
        ),
        row_source_node_id=np.ascontiguousarray(
            input_model.source_node_id_sorted,
            dtype=np.int64,
        ),
        row_receiver_node_id=np.ascontiguousarray(
            input_model.receiver_node_id_sorted,
            dtype=np.int64,
        ),
        row_distance_m=np.ascontiguousarray(input_model.distance_m_sorted, dtype=np.float64),
        observed_pick_time_s=np.ascontiguousarray(
            input_model.pick_time_s_sorted,
            dtype=np.float64,
        ),
        modeled_pick_time_s=modeled,
        residual_time_s=residual,
        used_row_mask=used,
        rejected_by_robust_mask=np.zeros(input_model.n_traces, dtype=bool),
        qc=qc,
        active_cell_id=v2.active_cell_id,
        inactive_cell_id=v2.inactive_cell_id,
        cell_bedrock_slowness_s_per_m=v2.cell_bedrock_slowness_s_per_m,
        cell_bedrock_velocity_m_s=v2.cell_bedrock_velocity_m_s,
        cell_velocity_status=v2.cell_velocity_status,
        row_midpoint_cell_id=v2.row_midpoint_cell_id,
        node_v2_cell_id=v2.node_v2_cell_id,
        node_v2_m_s=v2.node_v2_m_s,
        node_v2_status=v2.node_v2_status,
        source_v2_cell_id=v2.source_v2_cell_id,
        source_v2_m_s=v2.source_v2_m_s,
        source_v2_status=v2.source_v2_status,
        receiver_v2_cell_id=v2.receiver_v2_cell_id,
        receiver_v2_m_s=v2.receiver_v2_m_s,
        receiver_v2_status=v2.receiver_v2_status,
        source_v2_cell_id_sorted=v2.source_v2_cell_id_sorted,
        source_v2_m_s_sorted=v2.source_v2_m_s_sorted,
        source_v2_status_sorted=v2.source_v2_status_sorted,
        receiver_v2_cell_id_sorted=v2.receiver_v2_cell_id_sorted,
        receiver_v2_m_s_sorted=v2.receiver_v2_m_s_sorted,
        receiver_v2_status_sorted=v2.receiver_v2_status_sorted,
        source_t2_time_s=source_t2,
        source_v3_m_s=source_v3_m_s,
        source_sh2_weathering_thickness_m=source_thickness.sh2_m,
        receiver_t2_time_s=receiver_t2,
        receiver_v3_m_s=receiver_v3_m_s,
        receiver_sh2_weathering_thickness_m=receiver_thickness.sh2_m,
    )


@dataclass(frozen=True)
class _EndpointMetadata:
    endpoint_key: np.ndarray
    endpoint_id: np.ndarray
    node_id: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    elevation_m: np.ndarray


@dataclass(frozen=True)
class _V2StaticModel:
    node_v2_m_s: np.ndarray
    node_v2_status: np.ndarray | None
    source_v2_m_s: np.ndarray
    source_v2_status: np.ndarray | None
    receiver_v2_m_s: np.ndarray
    receiver_v2_status: np.ndarray | None
    source_v2_m_s_sorted: np.ndarray
    source_v2_status_sorted: np.ndarray | None
    receiver_v2_m_s_sorted: np.ndarray
    receiver_v2_status_sorted: np.ndarray | None
    active_cell_id: np.ndarray | None = None
    inactive_cell_id: np.ndarray | None = None
    cell_bedrock_slowness_s_per_m: np.ndarray | None = None
    cell_bedrock_velocity_m_s: np.ndarray | None = None
    cell_velocity_status: np.ndarray | None = None
    row_midpoint_cell_id: np.ndarray | None = None
    node_v2_cell_id: np.ndarray | None = None
    source_v2_cell_id: np.ndarray | None = None
    receiver_v2_cell_id: np.ndarray | None = None
    source_v2_cell_id_sorted: np.ndarray | None = None
    receiver_v2_cell_id_sorted: np.ndarray | None = None


def _require_two_layer_t1lsst_layers(
    normalized_layers: tuple[RefractionStaticLayerConfig, ...],
) -> None:
    kinds = tuple(config.kind for config in normalized_layers)
    if kinds != ('v2_t1', 'v3_t2'):
        raise RefractionMultiLayerSolveError(
            'two-layer T1LSST statics requires enabled layers v2_t1 and v3_t2'
        )
    v3_mode = normalized_layers[1].velocity_mode
    if v3_mode != 'solve_global':
        raise RefractionMultiLayerSolveError(
            'two-layer T1LSST statics currently requires global V3/T2 velocity'
        )


def _components_from_replacement(
    result: RefractionWeatheringReplacementStaticsResult,
) -> RefractionMultiLayerStaticComponents:
    source_t2 = _required_result_array(
        result.source_t2_time_s,
        name='source_t2_s',
    )
    receiver_t2 = _required_result_array(
        result.receiver_t2_time_s,
        name='receiver_t2_s',
    )
    source_sh2 = _required_result_array(
        result.source_sh2_weathering_thickness_m,
        name='source_sh2_m',
    )
    receiver_sh2 = _required_result_array(
        result.receiver_sh2_weathering_thickness_m,
        name='receiver_sh2_m',
    )
    return RefractionMultiLayerStaticComponents(
        source_t1_s=np.ascontiguousarray(result.source_half_intercept_time_s),
        source_t2_s=source_t2,
        source_t3_s=None,
        receiver_t1_s=np.ascontiguousarray(result.receiver_half_intercept_time_s),
        receiver_t2_s=receiver_t2,
        receiver_t3_s=None,
        source_sh1_m=np.ascontiguousarray(result.source_weathering_thickness_m),
        source_sh2_m=source_sh2,
        source_sh3_m=None,
        receiver_sh1_m=np.ascontiguousarray(result.receiver_weathering_thickness_m),
        receiver_sh2_m=receiver_sh2,
        receiver_sh3_m=None,
        source_weathering_correction_s=np.ascontiguousarray(
            result.source_weathering_replacement_shift_s
        ),
        receiver_weathering_correction_s=np.ascontiguousarray(
            result.receiver_weathering_replacement_shift_s
        ),
        qc={
            'conversion_mode': 't1lsst_multilayer',
            'layer_count': 2,
            'sign_convention': _SIGN_CONVENTION_TEXT,
        },
    )


def _required_result_array(value: object, *, name: str) -> np.ndarray:
    if value is None:
        raise RefractionMultiLayerSolveError(f'{name} is required')
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise RefractionMultiLayerSolveError(f'{name} must be one-dimensional')
    return np.ascontiguousarray(arr, dtype=np.float64)


def _required_layer_result(
    result: RefractionMultiLayerSolveResult,
    layer_kind: RefractionLayerKind,
) -> RefractionLayerSolveResult:
    for layer in result.layer_results:
        if layer.layer_kind == layer_kind:
            return layer
    raise RefractionMultiLayerSolveError(f'{layer_kind} layer result is required')


def _required_global_velocity(layer: RefractionLayerSolveResult) -> float:
    if layer.global_velocity_m_s is None:
        raise RefractionMultiLayerSolveError(
            f'{layer.layer_kind} must return a global velocity'
        )
    return _positive_float(
        layer.global_velocity_m_s,
        name=f'{layer.layer_kind}.global_velocity_m_s',
    )


def _positive_float(value: object, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionMultiLayerSolveError(f'{name} must be numeric') from exc
    if not np.isfinite(out) or out <= 0.0:
        raise RefractionMultiLayerSolveError(f'{name} must be positive and finite')
    return out


def _input_node_id(input_model: RefractionStaticInputModel) -> np.ndarray:
    node_id = np.asarray(input_model.endpoint_table.node_id, dtype=np.int64)
    if node_id.shape != np.asarray(input_model.node_x_m).shape:
        raise RefractionMultiLayerSolveError(
            'input_model.endpoint_table.node_id must match node arrays'
        )
    if np.unique(node_id).shape[0] != int(node_id.shape[0]):
        raise RefractionMultiLayerSolveError('input_model node IDs must be unique')
    return np.ascontiguousarray(node_id, dtype=np.int64)


def _endpoint_metadata(
    input_model: RefractionStaticInputModel,
    *,
    endpoint: str,
) -> _EndpointMetadata:
    if endpoint == 'source':
        key = input_model.source_endpoint_key_sorted
        endpoint_id = input_model.source_id_sorted
        node_id = input_model.source_node_id_sorted
        x_m = input_model.source_x_m_sorted
        y_m = input_model.source_y_m_sorted
        elevation_m = input_model.source_elevation_m_sorted
    elif endpoint == 'receiver':
        key = input_model.receiver_endpoint_key_sorted
        endpoint_id = input_model.receiver_id_sorted
        node_id = input_model.receiver_node_id_sorted
        x_m = input_model.receiver_x_m_sorted
        y_m = input_model.receiver_y_m_sorted
        elevation_m = input_model.receiver_elevation_m_sorted
    else:
        raise RefractionMultiLayerSolveError(f'unsupported endpoint: {endpoint}')
    positions = _first_endpoint_positions(key)
    return _EndpointMetadata(
        endpoint_key=np.asarray(key, dtype=object)[positions],
        endpoint_id=np.ascontiguousarray(endpoint_id, dtype=np.int64)[positions],
        node_id=np.ascontiguousarray(node_id, dtype=np.int64)[positions],
        x_m=np.ascontiguousarray(x_m, dtype=np.float64)[positions],
        y_m=np.ascontiguousarray(y_m, dtype=np.float64)[positions],
        elevation_m=np.ascontiguousarray(elevation_m, dtype=np.float64)[positions],
    )


def _first_endpoint_positions(values: np.ndarray) -> np.ndarray:
    seen: set[str] = set()
    positions: list[int] = []
    for index, value in enumerate(np.asarray(values, dtype=object).tolist()):
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        positions.append(index)
    return np.asarray(positions, dtype=np.int64)


def _validate_endpoint_order(
    *,
    actual: np.ndarray,
    expected: np.ndarray,
    name: str,
) -> None:
    if actual.shape != expected.shape or not np.array_equal(
        actual.astype(str),
        np.asarray(expected).astype(str),
    ):
        raise RefractionMultiLayerSolveError(f'{name} order mismatch')


def _layer_node_terms(
    layer: RefractionLayerSolveResult,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if layer.node_time_term_s is None:
        raise RefractionMultiLayerSolveError(f'{name} node time terms are required')
    arr = np.asarray(layer.node_time_term_s, dtype=np.float64)
    if arr.shape != shape:
        raise RefractionMultiLayerSolveError(f'{name} node time-term shape mismatch')
    return np.ascontiguousarray(arr, dtype=np.float64)


def _endpoint_terms(
    values: np.ndarray,
    endpoint: _EndpointMetadata,
    *,
    name: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != endpoint.endpoint_key.shape:
        raise RefractionMultiLayerSolveError(f'{name} shape mismatch')
    return np.ascontiguousarray(arr, dtype=np.float64)


def _build_v2_static_model(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    layer: RefractionLayerSolveResult,
    node_id: np.ndarray,
    source: _EndpointMetadata,
    receiver: _EndpointMetadata,
    v1_m_s: float,
) -> _V2StaticModel:
    if layer.velocity_mode != 'solve_cell':
        v2_m_s = _required_global_velocity(layer)
        return _global_v2_static_model(
            input_model=input_model,
            node_id=node_id,
            source=source,
            receiver=receiver,
            v2_m_s=v2_m_s,
        )
    return _cell_v2_static_model(
        input_model=input_model,
        model=model,
        layer=layer,
        node_id=node_id,
        source=source,
        receiver=receiver,
        v1_m_s=v1_m_s,
    )


def _global_v2_static_model(
    *,
    input_model: RefractionStaticInputModel,
    node_id: np.ndarray,
    source: _EndpointMetadata,
    receiver: _EndpointMetadata,
    v2_m_s: float,
) -> _V2StaticModel:
    return _V2StaticModel(
        node_v2_m_s=np.full(node_id.shape, v2_m_s, dtype=np.float64),
        node_v2_status=None,
        source_v2_m_s=np.full(source.endpoint_key.shape, v2_m_s, dtype=np.float64),
        source_v2_status=None,
        receiver_v2_m_s=np.full(receiver.endpoint_key.shape, v2_m_s, dtype=np.float64),
        receiver_v2_status=None,
        source_v2_m_s_sorted=np.full(input_model.n_traces, v2_m_s, dtype=np.float64),
        source_v2_status_sorted=None,
        receiver_v2_m_s_sorted=np.full(input_model.n_traces, v2_m_s, dtype=np.float64),
        receiver_v2_status_sorted=None,
    )


def _cell_v2_static_model(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    layer: RefractionLayerSolveResult,
    node_id: np.ndarray,
    source: _EndpointMetadata,
    receiver: _EndpointMetadata,
    v1_m_s: float,
) -> _V2StaticModel:
    if model.refractor_cell is None:
        raise RefractionMultiLayerSolveError(
            'model.refractor_cell is required for solve_cell V2'
        )
    if (
        layer.active_cell_id is None
        or layer.inactive_cell_id is None
        or layer.cell_velocity_m_s is None
        or layer.cell_slowness_s_per_m is None
    ):
        raise RefractionMultiLayerSolveError(
            'solve_cell V2 layer must return active cell velocities'
        )
    cell_status = (
        np.full(layer.cell_velocity_m_s.shape, 'solved', dtype=_STATUS_DTYPE)
        if layer.cell_velocity_status is None
        else np.asarray(layer.cell_velocity_status).astype(_STATUS_DTYPE, copy=True)
    )
    grid = build_refraction_cell_grid(
        effective_refraction_cell_grid_config(model.refractor_cell)
    )
    node_cell, node_v2, node_status = _project_v2_to_points(
        grid=grid,
        refractor_cell=model.refractor_cell,
        x_m=input_model.node_x_m,
        y_m=input_model.node_y_m,
        active_cell_id=layer.active_cell_id,
        cell_velocity_m_s=layer.cell_velocity_m_s,
        v1_m_s=v1_m_s,
    )
    source_cell, source_v2, source_status = _project_v2_to_points(
        grid=grid,
        refractor_cell=model.refractor_cell,
        x_m=source.x_m,
        y_m=source.y_m,
        active_cell_id=layer.active_cell_id,
        cell_velocity_m_s=layer.cell_velocity_m_s,
        v1_m_s=v1_m_s,
    )
    receiver_cell, receiver_v2, receiver_status = _project_v2_to_points(
        grid=grid,
        refractor_cell=model.refractor_cell,
        x_m=receiver.x_m,
        y_m=receiver.y_m,
        active_cell_id=layer.active_cell_id,
        cell_velocity_m_s=layer.cell_velocity_m_s,
        v1_m_s=v1_m_s,
    )
    source_sorted = _map_endpoint_v2_to_trace_order(
        endpoint_key_sorted=input_model.source_endpoint_key_sorted,
        endpoint_key=source.endpoint_key,
        endpoint_cell_id=source_cell,
        endpoint_v2_m_s=source_v2,
        endpoint_v2_status=source_status,
    )
    receiver_sorted = _map_endpoint_v2_to_trace_order(
        endpoint_key_sorted=input_model.receiver_endpoint_key_sorted,
        endpoint_key=receiver.endpoint_key,
        endpoint_cell_id=receiver_cell,
        endpoint_v2_m_s=receiver_v2,
        endpoint_v2_status=receiver_status,
    )
    source_projected = project_refraction_cell_points(
        x_m=input_model.source_x_m_sorted,
        y_m=input_model.source_y_m_sorted,
        mode=model.refractor_cell.coordinate_mode,
        line_origin_x_m=model.refractor_cell.line_origin_x_m,
        line_origin_y_m=model.refractor_cell.line_origin_y_m,
        line_azimuth_deg=model.refractor_cell.line_azimuth_deg,
    )
    receiver_projected = project_refraction_cell_points(
        x_m=input_model.receiver_x_m_sorted,
        y_m=input_model.receiver_y_m_sorted,
        mode=model.refractor_cell.coordinate_mode,
        line_origin_x_m=model.refractor_cell.line_origin_x_m,
        line_origin_y_m=model.refractor_cell.line_origin_y_m,
        line_azimuth_deg=model.refractor_cell.line_azimuth_deg,
    )
    row_assignment = assign_observation_midpoint_cells(
        grid,
        source_x_m=source_projected.x_m,
        source_y_m=source_projected.y_m,
        receiver_x_m=receiver_projected.x_m,
        receiver_y_m=receiver_projected.y_m,
    )
    return _V2StaticModel(
        node_v2_m_s=node_v2,
        node_v2_status=node_status,
        source_v2_m_s=source_v2,
        source_v2_status=source_status,
        receiver_v2_m_s=receiver_v2,
        receiver_v2_status=receiver_status,
        source_v2_m_s_sorted=source_sorted[1],
        source_v2_status_sorted=source_sorted[2],
        receiver_v2_m_s_sorted=receiver_sorted[1],
        receiver_v2_status_sorted=receiver_sorted[2],
        active_cell_id=np.ascontiguousarray(layer.active_cell_id, dtype=np.int64),
        inactive_cell_id=np.ascontiguousarray(layer.inactive_cell_id, dtype=np.int64),
        cell_bedrock_slowness_s_per_m=np.ascontiguousarray(
            layer.cell_slowness_s_per_m,
            dtype=np.float64,
        ),
        cell_bedrock_velocity_m_s=np.ascontiguousarray(
            layer.cell_velocity_m_s,
            dtype=np.float64,
        ),
        cell_velocity_status=cell_status,
        row_midpoint_cell_id=np.ascontiguousarray(
            row_assignment.cell_id,
            dtype=np.int64,
        ),
        node_v2_cell_id=node_cell,
        source_v2_cell_id=source_cell,
        receiver_v2_cell_id=receiver_cell,
        source_v2_cell_id_sorted=source_sorted[0],
        receiver_v2_cell_id_sorted=receiver_sorted[0],
    )


def _project_v2_to_points(
    *,
    grid: RefractionCellGrid,
    refractor_cell: Any,
    x_m: np.ndarray,
    y_m: np.ndarray,
    active_cell_id: np.ndarray,
    cell_velocity_m_s: np.ndarray,
    v1_m_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = project_refraction_cell_points(
        x_m=x_m,
        y_m=y_m,
        mode=refractor_cell.coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    assignment = assign_points_to_refraction_cells(
        grid,
        x_m=projected.x_m,
        y_m=projected.y_m,
    )
    cell_id = np.ascontiguousarray(assignment.cell_id, dtype=np.int64)
    velocity_by_cell = {
        int(cell): float(cell_velocity_m_s[index])
        for index, cell in enumerate(np.asarray(active_cell_id, dtype=np.int64).tolist())
    }
    v2 = np.full(cell_id.shape, np.nan, dtype=np.float64)
    status = np.full(cell_id.shape, 'solved', dtype=_STATUS_DTYPE)
    for index, raw_cell in enumerate(cell_id.tolist()):
        cell = int(raw_cell)
        if cell < 0:
            status[index] = 'outside_refractor_cell_grid'
            continue
        velocity = velocity_by_cell.get(cell)
        if velocity is None:
            status[index] = 'inactive_v2_cell'
            continue
        v2[index] = velocity
        if not np.isfinite(velocity) or velocity <= 0.0:
            status[index] = 'invalid_local_v2'
        elif velocity <= v1_m_s:
            status[index] = 'v2_not_greater_than_v1'
    return (
        np.ascontiguousarray(cell_id, dtype=np.int64),
        np.ascontiguousarray(v2, dtype=np.float64),
        status,
    )


def _map_endpoint_v2_to_trace_order(
    *,
    endpoint_key_sorted: np.ndarray,
    endpoint_key: np.ndarray,
    endpoint_cell_id: np.ndarray,
    endpoint_v2_m_s: np.ndarray,
    endpoint_v2_status: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_to_index = _endpoint_key_to_index(endpoint_key)
    n_traces = int(endpoint_key_sorted.shape[0])
    cell_id = np.full(n_traces, -1, dtype=np.int64)
    v2 = np.full(n_traces, np.nan, dtype=np.float64)
    status = np.full(n_traces, 'missing_endpoint', dtype=_STATUS_DTYPE)
    for index, raw_key in enumerate(np.asarray(endpoint_key_sorted, dtype=object).tolist()):
        endpoint_index = key_to_index.get(str(raw_key))
        if endpoint_index is None:
            continue
        cell_id[index] = int(endpoint_cell_id[endpoint_index])
        v2[index] = float(endpoint_v2_m_s[endpoint_index])
        status[index] = str(endpoint_v2_status[endpoint_index])
    return (
        np.ascontiguousarray(cell_id, dtype=np.int64),
        np.ascontiguousarray(v2, dtype=np.float64),
        status,
    )


def _map_endpoint_values_to_trace_order(
    *,
    endpoint_key_sorted: np.ndarray,
    endpoint_key: np.ndarray,
    endpoint_values: np.ndarray,
    name: str,
) -> np.ndarray:
    key_to_index = _endpoint_key_to_index(endpoint_key)
    values = np.asarray(endpoint_values, dtype=np.float64)
    out = np.full(endpoint_key_sorted.shape, np.nan, dtype=np.float64)
    for index, raw_key in enumerate(np.asarray(endpoint_key_sorted, dtype=object).tolist()):
        endpoint_index = key_to_index.get(str(raw_key))
        if endpoint_index is None:
            raise RefractionMultiLayerSolveError(f'{name} missing endpoint {raw_key!s}')
        out[index] = float(values[endpoint_index])
    return np.ascontiguousarray(out, dtype=np.float64)


def _map_endpoint_strings_to_trace_order(
    *,
    endpoint_key_sorted: np.ndarray,
    endpoint_key: np.ndarray,
    endpoint_values: np.ndarray,
    name: str,
) -> np.ndarray:
    key_to_index = _endpoint_key_to_index(endpoint_key)
    values = np.asarray(endpoint_values).astype(str, copy=False)
    out = np.full(endpoint_key_sorted.shape, 'missing_endpoint', dtype=_STATUS_DTYPE)
    for index, raw_key in enumerate(np.asarray(endpoint_key_sorted, dtype=object).tolist()):
        endpoint_index = key_to_index.get(str(raw_key))
        if endpoint_index is None:
            raise RefractionMultiLayerSolveError(f'{name} missing endpoint {raw_key!s}')
        out[index] = str(values[endpoint_index])
    return out


def _endpoint_key_to_index(endpoint_key: np.ndarray) -> dict[str, int]:
    return {
        str(key): index
        for index, key in enumerate(np.asarray(endpoint_key, dtype=object).tolist())
    }


def _status_from_conversion(
    conversion_status: np.ndarray,
    shift_s: np.ndarray,
    *,
    max_abs_shift_ms: float | None,
) -> np.ndarray:
    status = np.asarray(conversion_status).astype(_STATUS_DTYPE, copy=True)
    shift = np.asarray(shift_s, dtype=np.float64)
    status[(status == 'ok') & (~np.isfinite(shift))] = 'invalid_shift'
    if max_abs_shift_ms is not None:
        too_large = np.isfinite(shift) & (np.abs(shift) * 1000.0 > max_abs_shift_ms)
        status[too_large] = 'exceeds_max_abs_shift'
    return status


def _trace_static_status(
    *,
    source_status: np.ndarray,
    receiver_status: np.ndarray,
    trace_shift_s: np.ndarray,
    max_abs_shift_ms: float | None,
) -> np.ndarray:
    status = np.full(trace_shift_s.shape, 'ok', dtype=_STATUS_DTYPE)
    source_not_ok = np.asarray(source_status).astype(str, copy=False) != 'ok'
    receiver_not_ok = np.asarray(receiver_status).astype(str, copy=False) != 'ok'
    status[source_not_ok] = np.asarray(source_status).astype(str, copy=False)[
        source_not_ok
    ]
    status[receiver_not_ok] = np.asarray(receiver_status).astype(str, copy=False)[
        receiver_not_ok
    ]
    status[(status == 'ok') & (~np.isfinite(trace_shift_s))] = 'invalid_shift'
    if max_abs_shift_ms is not None:
        too_large = (
            (status == 'ok')
            & np.isfinite(trace_shift_s)
            & (np.abs(trace_shift_s) * 1000.0 > max_abs_shift_ms)
        )
        status[too_large] = 'exceeds_max_abs_shift'
    return status


def _combine_trace_shifts(source_shift_s: np.ndarray, receiver_shift_s: np.ndarray) -> np.ndarray:
    source = np.asarray(source_shift_s, dtype=np.float64)
    receiver = np.asarray(receiver_shift_s, dtype=np.float64)
    if source.shape != receiver.shape:
        raise RefractionMultiLayerSolveError('trace shift endpoint shape mismatch')
    out = np.full(source.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(source) & np.isfinite(receiver)
    out[finite] = source[finite] + receiver[finite]
    return np.ascontiguousarray(out, dtype=np.float64)


def _combined_modeled_pick_time(
    solve_result: RefractionMultiLayerSolveResult,
    n_traces: int,
) -> np.ndarray:
    modeled = np.full(int(n_traces), np.nan, dtype=np.float64)
    for layer in solve_result.layer_results:
        mask = np.asarray(layer.used_observation_mask_sorted, dtype=bool)
        if mask.shape != modeled.shape:
            raise RefractionMultiLayerSolveError(
                f'{layer.layer_kind} used mask shape mismatch'
            )
        modeled[mask] = np.asarray(
            layer.trace_predicted_time_s_sorted,
            dtype=np.float64,
        )[mask]
    return np.ascontiguousarray(modeled, dtype=np.float64)


def _combined_used_mask(
    solve_result: RefractionMultiLayerSolveResult,
    n_traces: int,
) -> np.ndarray:
    used = np.zeros(int(n_traces), dtype=bool)
    for layer in solve_result.layer_results:
        mask = np.asarray(layer.used_observation_mask_sorted, dtype=bool)
        if mask.shape != used.shape:
            raise RefractionMultiLayerSolveError(
                f'{layer.layer_kind} used mask shape mismatch'
            )
        used |= mask
    return np.ascontiguousarray(used, dtype=bool)


def _node_pick_counts(
    *,
    input_model: RefractionStaticInputModel,
    node_id: np.ndarray,
    used_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(node_id.shape, dtype=np.int64)
    used = np.zeros(node_id.shape, dtype=np.int64)
    pos = {int(node): index for index, node in enumerate(node_id.tolist())}
    for index, (source_node, receiver_node) in enumerate(
        zip(
            input_model.source_node_id_sorted.tolist(),
            input_model.receiver_node_id_sorted.tolist(),
            strict=True,
        )
    ):
        for node in (source_node, receiver_node):
            node_index = pos.get(int(node))
            if node_index is None:
                continue
            total[node_index] += 1
            if bool(used_mask[index]):
                used[node_index] += 1
    return total, used


def _node_residual_stats(
    *,
    input_model: RefractionStaticInputModel,
    node_id: np.ndarray,
    residual_s: np.ndarray,
    used_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values: list[list[float]] = [[] for _ in range(int(node_id.shape[0]))]
    pos = {int(node): index for index, node in enumerate(node_id.tolist())}
    for index, (source_node, receiver_node) in enumerate(
        zip(
            input_model.source_node_id_sorted.tolist(),
            input_model.receiver_node_id_sorted.tolist(),
            strict=True,
        )
    ):
        if not bool(used_mask[index]) or not np.isfinite(residual_s[index]):
            continue
        for node in (source_node, receiver_node):
            node_index = pos.get(int(node))
            if node_index is not None:
                values[node_index].append(float(residual_s[index]))
    rms = np.full(node_id.shape, np.nan, dtype=np.float64)
    mad = np.full(node_id.shape, np.nan, dtype=np.float64)
    for index, node_values in enumerate(values):
        if not node_values:
            continue
        arr = np.asarray(node_values, dtype=np.float64)
        rms[index] = float(np.sqrt(np.mean(arr * arr)))
        mad[index] = float(np.median(np.abs(arr - np.median(arr))))
    return np.ascontiguousarray(rms), np.ascontiguousarray(mad)


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
        ('v3_t2', 'fixed_global'): _solve_existing_time_term_layer,
        ('v3_t2', 'solve_global'): _solve_existing_time_term_layer,
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
    refractor_cell = _refractor_cell_payload_for_layer(
        payload=payload,
        config=config,
    )
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
            'refractor_cell': refractor_cell,
            'layers': None,
            'allow_overlapping_layer_gates': False,
        }
    )
    return RefractionStaticModelRequest.model_validate(payload)


def _refractor_cell_payload_for_layer(
    *,
    payload: Mapping[str, object],
    config: RefractionStaticLayerConfig,
) -> dict[str, object] | None:
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


def _solve_existing_time_term_layer(
    context: RefractionLayerSolverContext,
) -> RefractionLayerSolveResult:
    try:
        result = estimate_refraction_half_intercept_times_from_first_breaks(
            req=_LayerSolveRequest(context.model, context.solver),
            state=_LayerSolveState(),
            input_model=context.input_model,
            resolved_first_layer=context.resolved_first_layer,
        )
    except ValueError as exc:
        raise RefractionMultiLayerSolveError(
            f'refraction layer {context.layer_config.kind} solve failed: {exc}'
        ) from exc
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
    global_velocity = None if is_cell else float(result.bedrock_velocity_m_s)
    global_slowness = None if is_cell else float(result.bedrock_slowness_s_per_m)
    qc = {
        **result.qc,
        'layer_kind': layer_kind,
        'layer_index': layer_index,
        'velocity_mode': velocity_mode,
        'n_observations': int(result.row_trace_index_sorted.shape[0]),
        'n_sources': int(np.unique(result.row_source_node_id).shape[0]),
        'n_receivers': int(np.unique(result.row_receiver_node_id).shape[0]),
        'robust_iterations': int(result.qc.get('robust_iteration_count', 0)),
        'n_rejected_by_robust': int(
            np.count_nonzero(result.rejected_by_robust_mask)
        ),
    }
    qc.update(
        _layer_velocity_qc_aliases(
            layer_kind=layer_kind,
            global_velocity_m_s=global_velocity,
            global_slowness_s_per_m=global_slowness,
        )
    )
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
        global_velocity_m_s=global_velocity,
        global_slowness_s_per_m=global_slowness,
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
        active_cell_id=(
            None
            if result.active_cell_id is None
            else np.ascontiguousarray(result.active_cell_id, dtype=np.int64)
        ),
        inactive_cell_id=(
            None
            if result.inactive_cell_id is None
            else np.ascontiguousarray(result.inactive_cell_id, dtype=np.int64)
        ),
        cell_velocity_status=(
            None
            if result.cell_velocity_status is None
            else np.asarray(result.cell_velocity_status).astype(_STATUS_DTYPE, copy=True)
        ),
        row_midpoint_cell_id=(
            None
            if result.row_midpoint_cell_id is None
            else np.ascontiguousarray(result.row_midpoint_cell_id, dtype=np.int64)
        ),
        row_midpoint_velocity_m_s=(
            None
            if result.row_midpoint_bedrock_velocity_m_s is None
            else np.ascontiguousarray(
                result.row_midpoint_bedrock_velocity_m_s,
                dtype=np.float64,
            )
        ),
    )


def _layer_velocity_qc_aliases(
    *,
    layer_kind: RefractionLayerKind,
    global_velocity_m_s: float | None,
    global_slowness_s_per_m: float | None,
) -> dict[str, float]:
    if global_velocity_m_s is None or global_slowness_s_per_m is None:
        return {}
    if layer_kind == 'v2_t1':
        return {
            'v2_m_s': float(global_velocity_m_s),
            'slowness2_s_per_m': float(global_slowness_s_per_m),
        }
    if layer_kind == 'v3_t2':
        return {
            'v3_m_s': float(global_velocity_m_s),
            'slowness3_s_per_m': float(global_slowness_s_per_m),
        }
    if layer_kind == 'vsub_t3':
        return {
            'vsub_m_s': float(global_velocity_m_s),
            'slowness_sub_s_per_m': float(global_slowness_s_per_m),
        }
    return {}


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


def _validate_layer_velocity_sequence(
    *,
    result: RefractionLayerSolveResult,
    previous_results: tuple[RefractionLayerSolveResult, ...],
    normalized_layers: tuple[RefractionStaticLayerConfig, ...],
) -> RefractionLayerSolveResult:
    if result.layer_kind != 'v3_t2':
        return result
    current_velocity = _summary_velocity_m_s(result)
    if current_velocity is None:
        return result
    reference = _prior_layer_velocity_reference(
        layer_kind='v2_t1',
        previous_results=previous_results,
        normalized_layers=normalized_layers,
    )
    if reference is None:
        return result
    reference_kind, reference_velocity, reference_source = reference
    if current_velocity <= reference_velocity:
        raise RefractionMultiLayerSolveError(
            f'refraction layer {result.layer_kind} velocity must be greater '
            f'than {reference_kind} velocity ({current_velocity:.6g} <= '
            f'{reference_velocity:.6g} m/s)'
        )
    return replace(
        result,
        qc={
            **result.qc,
            'velocity_sequence_reference_layer_kind': reference_kind,
            'velocity_sequence_reference_m_s': float(reference_velocity),
            'velocity_sequence_reference_source': reference_source,
        },
    )


def _prior_layer_velocity_reference(
    *,
    layer_kind: RefractionLayerKind,
    previous_results: tuple[RefractionLayerSolveResult, ...],
    normalized_layers: tuple[RefractionStaticLayerConfig, ...],
) -> tuple[RefractionLayerKind, float, str] | None:
    for previous in reversed(previous_results):
        if previous.layer_kind != layer_kind:
            continue
        velocity = _summary_velocity_m_s(previous)
        if velocity is not None:
            return layer_kind, velocity, 'solved_summary'
    for config in normalized_layers:
        if config.kind == layer_kind and config.min_velocity_m_s is not None:
            return layer_kind, float(config.min_velocity_m_s), 'configured_min'
    return None


def _summary_velocity_m_s(result: RefractionLayerSolveResult) -> float | None:
    if result.global_velocity_m_s is not None:
        velocity = float(result.global_velocity_m_s)
        if np.isfinite(velocity) and velocity > 0.0:
            return velocity
        return None
    if result.cell_velocity_m_s is None:
        return None
    values = np.asarray(result.cell_velocity_m_s, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None
    return float(np.median(values))


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
    'RefractionMultiLayerStaticsWorkflowResult',
    'build_refraction_multilayer_weathering_replacement_statics',
    'compute_refraction_multilayer_datum_statics_from_input_model',
    'solve_refraction_multilayer_time_terms',
]
