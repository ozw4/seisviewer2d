"""Sequential multi-layer time-term orchestration for refraction statics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from seis_statics.refraction.multilayer_conversion import (
    RefractionMultilayerConversionError as CoreRefractionMultilayerConversionError,
    build_refraction_multilayer_conversion as core_build_refraction_multilayer_conversion,
)
from seis_statics.refraction.multilayer_solver import (
    RefractionMultilayerTimeTermLayerResult as CoreRefractionMultilayerTimeTermLayerResult,
    RefractionMultilayerTimeTermSolveResult as CoreRefractionMultilayerTimeTermSolveResult,
)
from seis_statics.refraction.multilayer_solver import (
    RefractionMultilayerTimeTermSolverError as CoreRefractionMultilayerTimeTermSolverError,
)
from seis_statics.refraction.multilayer_solver import (
    solve_refraction_multilayer_time_terms as core_solve_refraction_multilayer_time_terms,
)
from app.statics.refraction.contracts.options import (
    RefractionStaticApplyOptions,
    RefractionStaticConversionRequest,
    RefractionStaticDatumRequest,
    RefractionStaticSolverRequest,
)
from app.statics.refraction.contracts.apply import RefractionStaticApplyRequest
from app.statics.refraction.contracts.inputs import (
    RefractionStaticLinkageRequest,
    RefractionStaticPickSourceRequest,
)
from app.statics.refraction.contracts.model import RefractionStaticModelRequest
from app.statics.refraction.artifacts import write_refraction_static_artifacts
from seis_statics.refraction.cell_coordinates import (
    effective_refraction_cell_grid_config,
    project_refraction_cell_points,
)
from seis_statics.refraction.cell_grid import (
    RefractionCellGrid,
    assign_observation_midpoint_cells,
    assign_points_to_refraction_cells,
    build_refraction_cell_grid,
)
from seis_statics.refraction.layer_config import (
    RefractionLayerConfigLayer as RefractionStaticLayerConfig,
)
from seis_statics.refraction.types import RefractionLayerObservationMasks
from app.statics.refraction.application.core_options import (
    core_input_model_from_app,
    layer_observation_masks_from_input_model,
    layer_observation_qc_for_viewer as refraction_layer_observation_qc,
    model_options_from_request,
    normalized_layers_from_model_request,
    refractor_cell_options_from_request,
    solver_options_from_request,
)
from app.statics.refraction.application.datum import (
    build_refraction_datum_statics,
    write_refraction_datum_statics_artifacts,
)
from app.statics.refraction.ports.runtime import RefractionRuntime
from app.statics.refraction.application.design_matrix import (
    LOW_FOLD_CELL_REJECTION_REASON,
    REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME,
    REFRACTION_DESIGN_MATRIX_QC_JSON_NAME,
    build_refraction_static_design_matrix,
    refraction_design_matrix_layer_node_diagnostics_csv_name,
    refraction_design_matrix_layer_qc_json_name,
    write_refraction_design_matrix_diagnostics_artifacts,
)
from app.statics.refraction.contracts.result_types import (
    RefractionDatumStaticsResult,
    RefractionLayerKind,
    RefractionLayerSolveResult,
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
_TRACE_OK_STATUSES = {'ok', 'solved', 'zero_thickness'}
_SIGN_CONVENTION_TEXT = 'corrected(t) = raw(t - shift_s)'
_CELL_THRESHOLD_QC_KEYS = (
    'cell_observation_count',
    'low_fold_cell_id',
    'low_fold_cell_rejection_reason',
    'min_observations_per_cell',
    'n_low_fold_cells',
    'n_observations_outside_grid',
    'n_observations_rejected_by_low_fold_cell',
)
_DESIGN_MATRIX_ARTIFACT_DIR_PREFIX = 'refraction_design_matrix'


class RefractionMultiLayerSolveError(ValueError):
    """Raised when multi-layer refraction orchestration cannot continue."""


@dataclass(frozen=True)
class RefractionMultiLayerStaticsWorkflowResult:
    """Production outputs for a multi-layer statics workflow."""

    solve_result: RefractionMultiLayerSolveResult
    components: RefractionMultiLayerStaticComponents
    weathering_replacement_result: RefractionWeatheringReplacementStaticsResult
    datum_result: RefractionDatumStaticsResult


def solve_refraction_multilayer_time_terms(
    *,
    input_model: RefractionStaticInputModel,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    normalized_layers: tuple[RefractionStaticLayerConfig, ...],
    layer_masks: RefractionLayerObservationMasks,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    job_dir: Path | None = None,
) -> RefractionMultiLayerSolveResult:
    """Run enabled refraction layer solves in configured order."""
    if not normalized_layers:
        raise RefractionMultiLayerSolveError(
            'at least one enabled refraction layer is required'
        )
    for config in normalized_layers:
        _require_layer_observations(config, layer_masks)
        min_used = int(solver.robust.min_used_observations)
        count = int(layer_masks.layer_observation_count.get(config.kind, 0))
        if count < min_used:
            raise RefractionMultiLayerSolveError(
                f'refraction layer {config.kind} solve failed: '
                'Too few valid refraction observations'
            )
    return _solve_refraction_multilayer_time_terms_with_core(
        input_model=input_model,
        resolved_first_layer=resolved_first_layer,
        normalized_layers=normalized_layers,
        layer_masks=layer_masks,
        model=model,
        solver=solver,
        job_dir=job_dir,
    )


def _solve_refraction_multilayer_time_terms_with_core(
    *,
    input_model: RefractionStaticInputModel,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    normalized_layers: tuple[RefractionStaticLayerConfig, ...],
    layer_masks: RefractionLayerObservationMasks,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    job_dir: Path | None,
) -> RefractionMultiLayerSolveResult:
    core_input = core_input_model_from_app(
        replace(input_model, layer_observation_masks=layer_masks)
    )
    try:
        core_result = core_solve_refraction_multilayer_time_terms(
            input_model=core_input,
            model=model_options_from_request(model),
            solver_options=solver_options_from_request(solver),
            resolved_first_layer=resolved_first_layer,
            include_diagnostics=job_dir is not None,
        )
    except CoreRefractionMultilayerTimeTermSolverError as exc:
        _write_failed_core_layer_design_matrix_diagnostics(
            input_model=input_model,
            resolved_first_layer=resolved_first_layer,
            normalized_layers=normalized_layers,
            layer_masks=layer_masks,
            model=model,
            solver=solver,
            job_dir=job_dir,
        )
        raise RefractionMultiLayerSolveError(str(exc)) from exc
    return _app_solve_result_from_core(
        input_model=input_model,
        normalized_layers=normalized_layers,
        layer_masks=layer_masks,
        model=model,
        core_result=core_result,
        job_dir=job_dir,
    )


def _app_solve_result_from_core(
    *,
    input_model: RefractionStaticInputModel,
    normalized_layers: tuple[RefractionStaticLayerConfig, ...],
    layer_masks: RefractionLayerObservationMasks,
    model: RefractionStaticModelRequest,
    core_result: CoreRefractionMultilayerTimeTermSolveResult,
    job_dir: Path | None,
) -> RefractionMultiLayerSolveResult:
    source_endpoint_key, source_node_id = _unique_endpoint_key_nodes(
        input_model.source_endpoint_key_sorted,
        input_model.source_node_id_sorted,
    )
    receiver_endpoint_key, receiver_node_id = _unique_endpoint_key_nodes(
        input_model.receiver_endpoint_key_sorted,
        input_model.receiver_node_id_sorted,
    )
    layer_results: list[RefractionLayerSolveResult] = []
    for core_layer in core_result.layer_results:
        layer_dir = _layer_design_matrix_artifact_dir(job_dir, core_layer.layer)
        if layer_dir is not None and core_layer.solve_result.design is not None:
            write_refraction_design_matrix_diagnostics_artifacts(
                layer_dir,
                core_layer.solve_result.design,
            )
        _copy_layer_design_matrix_diagnostics_to_root_artifacts(
            root=job_dir,
            layer_dir=layer_dir,
            layer_kind=core_layer.layer_kind,
            layer_index=int(core_layer.layer_index),
        )
        layer_result = _layer_result_from_core_layer(
            input_model=input_model,
            source_endpoint_key=source_endpoint_key,
            source_node_id=source_node_id,
            receiver_endpoint_key=receiver_endpoint_key,
            receiver_node_id=receiver_node_id,
            core_layer=core_layer,
        )
        layer_results.append(layer_result)

    enabled_kinds = tuple(config.kind for config in normalized_layers)
    observation_gates = refraction_layer_observation_qc(layer_masks, model=model)
    layer_qc = _multilayer_layer_qc(
        layer_results=tuple(layer_results),
        observation_gates=observation_gates,
    )
    return RefractionMultiLayerSolveResult(
        enabled_layer_kinds=enabled_kinds,
        layer_results=tuple(layer_results),
        source_endpoint_key=source_endpoint_key,
        receiver_endpoint_key=receiver_endpoint_key,
        source_node_id=source_node_id,
        receiver_node_id=receiver_node_id,
        qc={
            'enabled_layer_count': len(enabled_kinds),
            'enabled_layer_kinds': list(enabled_kinds),
            'observation_gates': observation_gates,
            'layers': layer_qc,
        },
        modeled_pick_time_s_sorted=_core_sorted_float_array(
            core_result.modeled_pick_time_s_sorted,
            n_traces=input_model.n_traces,
            name='modeled_pick_time_s_sorted',
        ),
        residual_s_sorted=_core_sorted_float_array(
            core_result.residual_s_sorted,
            n_traces=input_model.n_traces,
            name='residual_s_sorted',
        ),
        used_observation_mask_sorted=_core_sorted_bool_array(
            core_result.used_observation_mask_sorted,
            n_traces=input_model.n_traces,
            name='used_observation_mask_sorted',
        ),
        rejected_observation_mask_sorted=_core_sorted_bool_array(
            core_result.rejected_observation_mask_sorted,
            n_traces=input_model.n_traces,
            name='rejected_observation_mask_sorted',
        ),
        layer_kind_sorted=_core_sorted_string_array(
            core_result.layer_kind_sorted,
            n_traces=input_model.n_traces,
            name='layer_kind_sorted',
        ),
        rejection_reason_sorted=_core_sorted_string_array(
            core_result.rejection_reason_sorted,
            n_traces=input_model.n_traces,
            name='rejection_reason_sorted',
        ),
        velocity_m_s_sorted=_core_sorted_float_array(
            core_result.velocity_m_s_sorted,
            n_traces=input_model.n_traces,
            name='velocity_m_s_sorted',
        ),
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
    design_matrix_job_dir: Path | None = None,
    runtime: RefractionRuntime | None = None,
    state: object | None = None,
    file_id: str | None = None,
    key1_byte: int | None = None,
    key2_byte: int | None = None,
    floating_datum_artifact_path: Path | None = None,
) -> RefractionMultiLayerStaticsWorkflowResult:
    """Run the implemented multi-layer time-term, T1LSST, and datum workflow."""
    if runtime is None and state is not None:
        raise TypeError('runtime is required; AppState adaptation belongs in adapters')
    normalized_layers = normalized_layers_from_model_request(model)
    _require_multilayer_t1lsst_layers(normalized_layers)
    layer_count = len(normalized_layers)
    layer_masks = layer_observation_masks_from_input_model(
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
        job_dir=design_matrix_job_dir if design_matrix_job_dir is not None else job_dir,
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
        runtime=runtime,
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
            'layer_count': layer_count,
            'enabled_layer_kinds': list(solve_result.enabled_layer_kinds),
            'layers': solve_result.qc,
        },
        node_sh1_weathering_thickness_m=_required_result_array(
            weathering_replacement.node_sh1_weathering_thickness_m,
            name='node_sh1_weathering_thickness_m',
        ),
        node_sh2_weathering_thickness_m=_required_result_array(
            weathering_replacement.node_sh2_weathering_thickness_m,
            name='node_sh2_weathering_thickness_m',
        ),
        node_sh3_weathering_thickness_m=(
            weathering_replacement.node_sh3_weathering_thickness_m
        ),
        source_t2_time_s=components.source_t2_s,
        source_t3_time_s=components.source_t3_s,
        source_v3_m_s=_required_result_array(
            weathering_replacement.source_v3_m_s,
            name='source_v3_m_s',
        ),
        source_vsub_m_s=weathering_replacement.source_vsub_m_s,
        source_sh1_weathering_thickness_m=components.source_sh1_m,
        source_sh2_weathering_thickness_m=components.source_sh2_m,
        source_sh3_weathering_thickness_m=components.source_sh3_m,
        receiver_t2_time_s=components.receiver_t2_s,
        receiver_t3_time_s=components.receiver_t3_s,
        receiver_v3_m_s=_required_result_array(
            weathering_replacement.receiver_v3_m_s,
            name='receiver_v3_m_s',
        ),
        receiver_vsub_m_s=weathering_replacement.receiver_vsub_m_s,
        receiver_sh1_weathering_thickness_m=components.receiver_sh1_m,
        receiver_sh2_weathering_thickness_m=components.receiver_sh2_m,
        receiver_sh3_weathering_thickness_m=components.receiver_sh3_m,
        layer_results=solve_result.layer_results,
    )
    if job_dir is not None:
        root = Path(job_dir)
        write_refraction_datum_statics_artifacts(root, datum_result)
        write_refraction_static_artifacts(
            result=datum_result,
            req=_artifact_request_for_multilayer_workflow(
                input_model=input_model,
                model=model,
                solver=solver,
                datum=datum,
                apply_options=apply_options,
                file_id=file_id,
                key1_byte=key1_byte,
                key2_byte=key2_byte,
            ),
            job_dir=root,
            resolved_first_layer=resolved_first_layer,
        )
    return RefractionMultiLayerStaticsWorkflowResult(
        solve_result=solve_result,
        components=components,
        weathering_replacement_result=weathering_replacement,
        datum_result=datum_result,
    )


def _artifact_request_for_multilayer_workflow(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    datum: RefractionStaticDatumRequest,
    apply_options: RefractionStaticApplyOptions | None,
    file_id: str | None,
    key1_byte: int | None,
    key2_byte: int | None,
) -> RefractionStaticApplyRequest:
    request = RefractionStaticApplyRequest(
        file_id=file_id if file_id is not None else input_model.file_id,
        pick_source=RefractionStaticPickSourceRequest(kind='manual_memmap'),
        linkage=RefractionStaticLinkageRequest(mode='none'),
        model=model,
        solver=solver,
        datum=datum,
        conversion=RefractionStaticConversionRequest(
            mode='t1lsst_multilayer',
            layer_count=len(normalized_layers_from_model_request(model)),
        ),
        apply=apply_options or RefractionStaticApplyOptions(),
    )
    if key1_byte is None and key2_byte is None:
        return request
    return request.model_copy(
        update={
            'key1_byte': request.key1_byte if key1_byte is None else key1_byte,
            'key2_byte': request.key2_byte if key2_byte is None else key2_byte,
        }
    )


def build_refraction_multilayer_weathering_replacement_statics(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    solve_result: RefractionMultiLayerSolveResult,
    apply_options: RefractionStaticApplyOptions | None,
    resolved_first_layer: ResolvedRefractionFirstLayer,
) -> RefractionWeatheringReplacementStaticsResult:
    """Build T1LSST replacement statics from production layer solves."""
    normalized_layers = normalized_layers_from_model_request(model)
    _require_multilayer_t1lsst_layers(normalized_layers)
    layer_count = len(normalized_layers)
    v2_layer = _required_layer_result(solve_result, 'v2_t1')
    v3_layer = _required_layer_result(solve_result, 'v3_t2')
    vsub_layer = (
        _required_layer_result(solve_result, 'vsub_t3')
        if layer_count == 3
        else None
    )
    v1_m_s = _positive_float(
        resolved_first_layer.weathering_velocity_m_s,
        name='resolved_first_layer.weathering_velocity_m_s',
    )
    v3_m_s = _required_global_velocity(v3_layer)
    vsub_m_s = None if vsub_layer is None else _required_global_velocity(vsub_layer)
    replacement_velocity_m_s = v3_m_s if vsub_m_s is None else vsub_m_s
    replacement_velocity_mode = (
        v3_layer.velocity_mode if vsub_layer is None else vsub_layer.velocity_mode
    )
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
    if vsub_layer is None:
        source_t3 = receiver_t3 = None
    else:
        source_t3 = _endpoint_terms(
            vsub_layer.source_time_term_s,
            source,
            name='source_t3',
        )
        receiver_t3 = _endpoint_terms(
            vsub_layer.receiver_time_term_s,
            receiver,
            name='receiver_t3',
        )
    trace_conversion = _core_multilayer_trace_conversion_from_app_result(
        input_model=input_model,
        model=model,
        solve_result=solve_result,
        resolved_first_layer=resolved_first_layer,
        layer_count=layer_count,
        source=source,
        receiver=receiver,
    )
    source_conversion = trace_conversion.source_endpoint
    receiver_conversion = trace_conversion.receiver_endpoint
    node_conversion = _core_multilayer_node_conversion_from_app_result(
        input_model=input_model,
        model=model,
        solve_result=solve_result,
        resolved_first_layer=resolved_first_layer,
        layer_count=layer_count,
    )
    node_sh3_m = None if vsub_layer is None else node_conversion.sh3_m
    source_sh3_m = None if vsub_layer is None else source_conversion.sh3_m
    receiver_sh3_m = None if vsub_layer is None else receiver_conversion.sh3_m

    max_abs_shift_ms = (
        None if apply_options is None else float(apply_options.max_abs_shift_ms)
    )
    node_static_status = _status_from_conversion(
        node_conversion.static_status,
        node_conversion.weathering_replacement_shift_s,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    source_static_status = _status_from_conversion(
        source_conversion.static_status,
        source_conversion.weathering_replacement_shift_s,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    receiver_static_status = _status_from_conversion(
        receiver_conversion.static_status,
        receiver_conversion.weathering_replacement_shift_s,
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
        endpoint_values=source_conversion.sh1_m,
        name='source_sh1_m_sorted',
    )
    receiver_sh1_sorted = _map_endpoint_values_to_trace_order(
        endpoint_key_sorted=input_model.receiver_endpoint_key_sorted,
        endpoint_key=receiver.endpoint_key,
        endpoint_values=receiver_conversion.sh1_m,
        name='receiver_sh1_m_sorted',
    )
    source_sh2_sorted = _map_endpoint_values_to_trace_order(
        endpoint_key_sorted=input_model.source_endpoint_key_sorted,
        endpoint_key=source.endpoint_key,
        endpoint_values=source_conversion.sh2_m,
        name='source_sh2_m_sorted',
    )
    receiver_sh2_sorted = _map_endpoint_values_to_trace_order(
        endpoint_key_sorted=input_model.receiver_endpoint_key_sorted,
        endpoint_key=receiver.endpoint_key,
        endpoint_values=receiver_conversion.sh2_m,
        name='receiver_sh2_m_sorted',
    )
    source_sh3_sorted = (
        None
        if vsub_layer is None
        else _map_endpoint_values_to_trace_order(
            endpoint_key_sorted=input_model.source_endpoint_key_sorted,
            endpoint_key=source.endpoint_key,
            endpoint_values=source_sh3_m,
            name='source_sh3_m_sorted',
        )
    )
    receiver_sh3_sorted = (
        None
        if vsub_layer is None
        else _map_endpoint_values_to_trace_order(
            endpoint_key_sorted=input_model.receiver_endpoint_key_sorted,
            endpoint_key=receiver.endpoint_key,
            endpoint_values=receiver_sh3_m,
            name='receiver_sh3_m_sorted',
        )
    )
    node_surface = np.ascontiguousarray(input_model.node_elevation_m, dtype=np.float64)
    node_total_thickness = _total_weathering_thickness(
        node_conversion.sh1_m,
        node_conversion.sh2_m,
        node_sh3_m,
    )
    source_total_thickness = _total_weathering_thickness(
        source_conversion.sh1_m,
        source_conversion.sh2_m,
        source_sh3_m,
    )
    receiver_total_thickness = _total_weathering_thickness(
        receiver_conversion.sh1_m,
        receiver_conversion.sh2_m,
        receiver_sh3_m,
    )
    source_total_thickness_sorted = _total_weathering_thickness(
        source_sh1_sorted,
        source_sh2_sorted,
        source_sh3_sorted,
    )
    receiver_total_thickness_sorted = _total_weathering_thickness(
        receiver_sh1_sorted,
        receiver_sh2_sorted,
        receiver_sh3_sorted,
    )
    source_wcor_sorted = _core_sorted_float_array(
        trace_conversion.source_weathering_replacement_shift_s_sorted,
        n_traces=input_model.n_traces,
        name='source_weathering_replacement_shift_s_sorted',
    )
    receiver_wcor_sorted = _core_sorted_float_array(
        trace_conversion.receiver_weathering_replacement_shift_s_sorted,
        n_traces=input_model.n_traces,
        name='receiver_weathering_replacement_shift_s_sorted',
    )
    source_status_sorted = _core_sorted_string_array(
        trace_conversion.source_static_status_sorted,
        n_traces=input_model.n_traces,
        name='source_static_status_sorted',
    )
    source_status_sorted = _status_from_conversion(
        source_status_sorted,
        source_wcor_sorted,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    receiver_status_sorted = _core_sorted_string_array(
        trace_conversion.receiver_static_status_sorted,
        n_traces=input_model.n_traces,
        name='receiver_static_status_sorted',
    )
    receiver_status_sorted = _status_from_conversion(
        receiver_status_sorted,
        receiver_wcor_sorted,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    trace_shift = _core_sorted_float_array(
        trace_conversion.weathering_replacement_trace_shift_s_sorted,
        n_traces=input_model.n_traces,
        name='weathering_replacement_trace_shift_s_sorted',
    )
    trace_status = _core_sorted_string_array(
        trace_conversion.trace_static_status_sorted,
        n_traces=input_model.n_traces,
        name='trace_static_status_sorted',
    )
    trace_status = _status_from_conversion(
        trace_status,
        trace_shift,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    trace_valid = _core_sorted_bool_array(
        trace_conversion.trace_static_valid_mask_sorted,
        n_traces=input_model.n_traces,
        name='trace_static_valid_mask_sorted',
    )
    modeled = _required_multilayer_sorted_float_array(
        solve_result,
        'modeled_pick_time_s_sorted',
    )
    residual = _required_multilayer_sorted_float_array(
        solve_result,
        'residual_s_sorted',
    )
    used = _required_multilayer_sorted_bool_array(
        solve_result,
        'used_observation_mask_sorted',
    )
    row_layer_kind = _required_multilayer_sorted_string_array(
        solve_result,
        'layer_kind_sorted',
    )
    row_layer_index = _layer_index_sorted_from_kind(row_layer_kind)
    row_rejection_reason = _artifact_rejection_reason_from_core(
        _required_multilayer_sorted_string_array(
            solve_result,
            'rejection_reason_sorted',
        )
    )
    row_velocity_m_s = _required_multilayer_sorted_float_array(
        solve_result,
        'velocity_m_s_sorted',
    )
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
        'layer_count': layer_count,
        'enabled_layer_kinds': list(solve_result.enabled_layer_kinds),
        'sign_convention': _SIGN_CONVENTION_TEXT,
        'bedrock_velocity_mode': replacement_velocity_mode,
        'weathering_velocity_m_s': float(v1_m_s),
        'v2_velocity_mode': v2_layer.velocity_mode,
        'v3_velocity_mode': v3_layer.velocity_mode,
        'v3_m_s': float(v3_m_s),
        'observation_gates': solve_result.qc.get('observation_gates', {}),
        'layers': solve_result.qc.get('layers', {}),
        **_cell_threshold_qc_from_layer(v2_layer),
    }
    if vsub_layer is not None:
        qc.update(
            {
                'vsub_velocity_mode': vsub_layer.velocity_mode,
                'vsub_m_s': float(replacement_velocity_m_s),
            }
        )
    source_v3_m_s = np.full(source.endpoint_key.shape, v3_m_s, dtype=np.float64)
    receiver_v3_m_s = np.full(receiver.endpoint_key.shape, v3_m_s, dtype=np.float64)
    source_vsub_m_s = (
        None
        if vsub_layer is None
        else np.full(source.endpoint_key.shape, replacement_velocity_m_s, dtype=np.float64)
    )
    receiver_vsub_m_s = (
        None
        if vsub_layer is None
        else np.full(receiver.endpoint_key.shape, replacement_velocity_m_s, dtype=np.float64)
    )
    rejected_by_robust = _robust_rejection_mask_from_reason(
        row_rejection_reason,
    )
    return RefractionWeatheringReplacementStaticsResult(
        bedrock_velocity_mode=replacement_velocity_mode,
        bedrock_slowness_s_per_m=1.0 / replacement_velocity_m_s,
        bedrock_velocity_m_s=replacement_velocity_m_s,
        weathering_velocity_m_s=v1_m_s,
        replacement_slowness_delta_s_per_m=(
            1.0 / replacement_velocity_m_s - 1.0 / v1_m_s
        ),
        node_id=node_id,
        node_x_m=np.ascontiguousarray(input_model.node_x_m, dtype=np.float64),
        node_y_m=np.ascontiguousarray(input_model.node_y_m, dtype=np.float64),
        node_surface_elevation_m=node_surface,
        node_kind=np.asarray(input_model.node_kind).astype('<U16', copy=True),
        node_weathering_thickness_m=node_total_thickness,
        node_refractor_elevation_m=node_surface - node_total_thickness,
        node_half_intercept_time_s=node_t1,
        node_solution_status=np.full(node_id.shape, 'solved', dtype=_STATUS_DTYPE),
        node_weathering_status=node_conversion.static_status,
        node_weathering_replacement_shift_s=(
            node_conversion.weathering_replacement_shift_s
        ),
        node_weathering_replacement_shift_ms=(
            node_conversion.weathering_replacement_shift_s * 1000.0
        ),
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
        source_weathering_thickness_m=source_total_thickness,
        source_refractor_elevation_m=source.elevation_m - source_total_thickness,
        source_weathering_replacement_shift_s=(
            source_conversion.weathering_replacement_shift_s
        ),
        source_static_status=source_static_status,
        receiver_endpoint_key=receiver.endpoint_key,
        receiver_id=receiver.endpoint_id,
        receiver_node_id=receiver.node_id,
        receiver_x_m=receiver.x_m,
        receiver_y_m=receiver.y_m,
        receiver_surface_elevation_m=receiver.elevation_m,
        receiver_half_intercept_time_s=receiver_t1,
        receiver_weathering_thickness_m=receiver_total_thickness,
        receiver_refractor_elevation_m=(
            receiver.elevation_m - receiver_total_thickness
        ),
        receiver_weathering_replacement_shift_s=(
            receiver_conversion.weathering_replacement_shift_s
        ),
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
        source_weathering_thickness_m_sorted=source_total_thickness_sorted,
        receiver_weathering_thickness_m_sorted=receiver_total_thickness_sorted,
        source_refractor_elevation_m_sorted=(
            np.ascontiguousarray(input_model.source_elevation_m_sorted, dtype=np.float64)
            - source_total_thickness_sorted
        ),
        receiver_refractor_elevation_m_sorted=(
            np.ascontiguousarray(
                input_model.receiver_elevation_m_sorted,
                dtype=np.float64,
            )
            - receiver_total_thickness_sorted
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
        rejected_by_robust_mask=rejected_by_robust,
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
        node_sh1_weathering_thickness_m=node_conversion.sh1_m,
        node_sh2_weathering_thickness_m=node_conversion.sh2_m,
        node_sh3_weathering_thickness_m=node_sh3_m,
        source_t2_time_s=source_t2,
        source_t3_time_s=source_t3,
        source_v3_m_s=source_v3_m_s,
        source_vsub_m_s=source_vsub_m_s,
        source_sh1_weathering_thickness_m=source_conversion.sh1_m,
        source_sh2_weathering_thickness_m=source_conversion.sh2_m,
        source_sh3_weathering_thickness_m=source_sh3_m,
        receiver_t2_time_s=receiver_t2,
        receiver_t3_time_s=receiver_t3,
        receiver_v3_m_s=receiver_v3_m_s,
        receiver_vsub_m_s=receiver_vsub_m_s,
        receiver_sh1_weathering_thickness_m=receiver_conversion.sh1_m,
        receiver_sh2_weathering_thickness_m=receiver_conversion.sh2_m,
        receiver_sh3_weathering_thickness_m=receiver_sh3_m,
        row_layer_kind=row_layer_kind,
        row_layer_index=row_layer_index,
        row_source_endpoint_key=np.asarray(
            input_model.source_endpoint_key_sorted,
            dtype=object,
        ),
        row_receiver_endpoint_key=np.asarray(
            input_model.receiver_endpoint_key_sorted,
            dtype=object,
        ),
        row_rejection_reason=row_rejection_reason,
        row_velocity_m_s=row_velocity_m_s,
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
    row_velocity_m_s: np.ndarray | None = None


@dataclass(frozen=True)
class _CoreEndpointConversionDomain:
    core_input_model: Any
    node_id: np.ndarray


def _core_multilayer_conversion_from_app_result(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    solve_result: RefractionMultiLayerSolveResult,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    layer_count: int,
    core_input_model: Any,
    trace_domain: bool,
) -> Any:
    try:
        return core_build_refraction_multilayer_conversion(
            input_model=core_input_model,
            model=model_options_from_request(model),
            solve_result=_core_multilayer_solve_result_from_app(
                input_model=input_model,
                model=model,
                solve_result=solve_result,
                resolved_first_layer=resolved_first_layer,
                endpoint_terms='node',
                trace_domain=trace_domain,
            ),
            resolved_first_layer=resolved_first_layer,
            layer_count=layer_count,
        )
    except CoreRefractionMultilayerConversionError as exc:
        raise RefractionMultiLayerSolveError(str(exc)) from exc


def _core_multilayer_trace_conversion_from_app_result(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    solve_result: RefractionMultiLayerSolveResult,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    layer_count: int,
    source: _EndpointMetadata,
    receiver: _EndpointMetadata,
) -> Any:
    domain = _core_endpoint_conversion_domain_from_app(
        input_model=input_model,
        source=source,
        receiver=receiver,
    )
    source_conversion = _core_multilayer_endpoint_trace_conversion_from_app_result(
        input_model=input_model,
        model=model,
        solve_result=solve_result,
        resolved_first_layer=resolved_first_layer,
        layer_count=layer_count,
        source=source,
        receiver=receiver,
        domain=domain,
        endpoint_terms='source',
    )
    receiver_conversion = _core_multilayer_endpoint_trace_conversion_from_app_result(
        input_model=input_model,
        model=model,
        solve_result=solve_result,
        resolved_first_layer=resolved_first_layer,
        layer_count=layer_count,
        source=source,
        receiver=receiver,
        domain=domain,
        endpoint_terms='receiver',
    )
    return _combine_endpoint_trace_conversions(
        source_conversion=source_conversion,
        receiver_conversion=receiver_conversion,
        n_traces=input_model.n_traces,
    )


def _core_multilayer_endpoint_trace_conversion_from_app_result(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    solve_result: RefractionMultiLayerSolveResult,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    layer_count: int,
    source: _EndpointMetadata,
    receiver: _EndpointMetadata,
    domain: _CoreEndpointConversionDomain,
    endpoint_terms: str,
) -> Any:
    try:
        return core_build_refraction_multilayer_conversion(
            input_model=domain.core_input_model,
            model=model_options_from_request(model),
            solve_result=_core_endpoint_multilayer_solve_result_from_app(
                input_model=input_model,
                model=model,
                solve_result=solve_result,
                resolved_first_layer=resolved_first_layer,
                source=source,
                receiver=receiver,
                domain=domain,
                endpoint_terms=endpoint_terms,
            ),
            resolved_first_layer=resolved_first_layer,
            layer_count=layer_count,
        )
    except CoreRefractionMultilayerConversionError as exc:
        raise RefractionMultiLayerSolveError(str(exc)) from exc


def _combine_endpoint_trace_conversions(
    *,
    source_conversion: Any,
    receiver_conversion: Any,
    n_traces: int,
) -> Any:
    source_shift = _core_sorted_float_array(
        source_conversion.source_weathering_replacement_shift_s_sorted,
        n_traces=n_traces,
        name='source_weathering_replacement_shift_s_sorted',
    )
    receiver_shift = _core_sorted_float_array(
        receiver_conversion.receiver_weathering_replacement_shift_s_sorted,
        n_traces=n_traces,
        name='receiver_weathering_replacement_shift_s_sorted',
    )
    trace_shift = _combine_source_receiver_trace_shift(
        source_shift=source_shift,
        receiver_shift=receiver_shift,
    )
    source_status = _core_sorted_string_array(
        source_conversion.source_static_status_sorted,
        n_traces=n_traces,
        name='source_static_status_sorted',
    )
    receiver_status = _core_sorted_string_array(
        receiver_conversion.receiver_static_status_sorted,
        n_traces=n_traces,
        name='receiver_static_status_sorted',
    )
    trace_status = _combine_source_receiver_trace_status(
        source_status=source_status,
        receiver_status=receiver_status,
        trace_shift=trace_shift,
    )
    return replace(
        source_conversion,
        receiver_endpoint=receiver_conversion.receiver_endpoint,
        receiver_weathering_replacement_shift_s_sorted=receiver_shift,
        receiver_static_status_sorted=receiver_status,
        weathering_replacement_trace_shift_s_sorted=trace_shift,
        trace_static_status_sorted=trace_status,
        trace_static_valid_mask_sorted=np.ascontiguousarray(
            (trace_status == 'ok') & np.isfinite(trace_shift),
            dtype=bool,
        ),
    )


def _core_multilayer_node_conversion_from_app_result(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    solve_result: RefractionMultiLayerSolveResult,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    layer_count: int,
) -> Any:
    conversion = _core_multilayer_conversion_from_app_result(
        input_model=input_model,
        model=model,
        solve_result=solve_result,
        resolved_first_layer=resolved_first_layer,
        layer_count=layer_count,
        core_input_model=_core_node_input_model_from_app(input_model),
        trace_domain=False,
    )
    return conversion.source_endpoint


def _core_endpoint_conversion_domain_from_app(
    *,
    input_model: RefractionStaticInputModel,
    source: _EndpointMetadata,
    receiver: _EndpointMetadata,
) -> _CoreEndpointConversionDomain:
    source_count = int(source.endpoint_key.shape[0])
    receiver_count = int(receiver.endpoint_key.shape[0])
    source_node_id = np.arange(source_count, dtype=np.int64)
    receiver_node_id = np.arange(
        source_count,
        source_count + receiver_count,
        dtype=np.int64,
    )
    core = core_input_model_from_app(input_model)
    core_input = replace(
        core,
        source_node_id_sorted=_map_endpoint_int_to_trace_order(
            endpoint_key_sorted=input_model.source_endpoint_key_sorted,
            endpoint_key=source.endpoint_key,
            endpoint_values=source_node_id,
            name='source_endpoint_conversion_node_id_sorted',
        ),
        receiver_node_id_sorted=_map_endpoint_int_to_trace_order(
            endpoint_key_sorted=input_model.receiver_endpoint_key_sorted,
            endpoint_key=receiver.endpoint_key,
            endpoint_values=receiver_node_id,
            name='receiver_endpoint_conversion_node_id_sorted',
        ),
        source_endpoint_id_sorted=_map_endpoint_int_to_trace_order(
            endpoint_key_sorted=input_model.source_endpoint_key_sorted,
            endpoint_key=source.endpoint_key,
            endpoint_values=source.endpoint_id,
            name='source_endpoint_id_sorted',
        ),
        receiver_endpoint_id_sorted=_map_endpoint_int_to_trace_order(
            endpoint_key_sorted=input_model.receiver_endpoint_key_sorted,
            endpoint_key=receiver.endpoint_key,
            endpoint_values=receiver.endpoint_id,
            name='receiver_endpoint_id_sorted',
        ),
        node_x_m=np.ascontiguousarray(
            np.concatenate((source.x_m, receiver.x_m)),
            dtype=np.float64,
        ),
        node_y_m=np.ascontiguousarray(
            np.concatenate((source.y_m, receiver.y_m)),
            dtype=np.float64,
        ),
        node_elevation_m=np.ascontiguousarray(
            np.concatenate((source.elevation_m, receiver.elevation_m)),
            dtype=np.float64,
        ),
        node_kind=np.ascontiguousarray(
            np.concatenate(
                (
                    np.full(source_count, 'source', dtype='<U16'),
                    np.full(receiver_count, 'receiver', dtype='<U16'),
                )
            ),
            dtype='<U16',
        ),
    )
    return _CoreEndpointConversionDomain(
        core_input_model=core_input,
        node_id=np.ascontiguousarray(
            np.concatenate((source_node_id, receiver_node_id)),
            dtype=np.int64,
        ),
    )


def _core_endpoint_multilayer_solve_result_from_app(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    solve_result: RefractionMultiLayerSolveResult,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    source: _EndpointMetadata,
    receiver: _EndpointMetadata,
    domain: _CoreEndpointConversionDomain,
    endpoint_terms: str,
) -> CoreRefractionMultilayerTimeTermSolveResult:
    base = _core_multilayer_solve_result_from_app(
        input_model=input_model,
        model=model,
        solve_result=solve_result,
        resolved_first_layer=resolved_first_layer,
        endpoint_terms=endpoint_terms,
        trace_domain=True,
    )
    layers = tuple(
        _core_endpoint_layer_result_from_app(
            input_model=input_model,
            model=model,
            layer=layer,
            resolved_first_layer=resolved_first_layer,
            source=source,
            receiver=receiver,
            domain=domain,
            endpoint_terms=endpoint_terms,
        )
        for layer in solve_result.layer_results
    )
    return replace(
        base,
        layer_results=layers,
        layer_result_by_kind={layer.layer_kind: layer for layer in layers},
    )


def _core_endpoint_layer_result_from_app(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    layer: RefractionLayerSolveResult,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    source: _EndpointMetadata,
    receiver: _EndpointMetadata,
    domain: _CoreEndpointConversionDomain,
    endpoint_terms: str,
) -> CoreRefractionMultilayerTimeTermLayerResult:
    base = _core_layer_result_from_app(
        input_model=input_model,
        model=model,
        layer=layer,
        resolved_first_layer=resolved_first_layer,
        endpoint_terms=endpoint_terms,
        trace_count=int(input_model.n_traces),
        trace_domain=True,
    )
    solve = base.solve_result
    return replace(
        base,
        solve_result=SimpleNamespace(
            node_id=domain.node_id,
            node_half_intercept_time_s=_endpoint_layer_terms_for_core_conversion(
                layer=layer,
                source=source,
                receiver=receiver,
            ),
            node_solution_status=np.full(
                domain.node_id.shape,
                'solved',
                dtype=_STATUS_DTYPE,
            ),
            bedrock_velocity_mode=solve.bedrock_velocity_mode,
            bedrock_velocity_m_s=solve.bedrock_velocity_m_s,
            bedrock_slowness_s_per_m=solve.bedrock_slowness_s_per_m,
            cell_id=solve.cell_id,
            cell_bedrock_velocity_m_s=solve.cell_bedrock_velocity_m_s,
            cell_bedrock_slowness_s_per_m=solve.cell_bedrock_slowness_s_per_m,
            cell_velocity_status=solve.cell_velocity_status,
        ),
    )


def _endpoint_layer_terms_for_core_conversion(
    *,
    layer: RefractionLayerSolveResult,
    source: _EndpointMetadata,
    receiver: _EndpointMetadata,
) -> np.ndarray:
    source_terms = _endpoint_terms(
        layer.source_time_term_s,
        source,
        name=f'{layer.layer_kind} source endpoint time terms',
    )
    receiver_terms = _endpoint_terms(
        layer.receiver_time_term_s,
        receiver,
        name=f'{layer.layer_kind} receiver endpoint time terms',
    )
    return np.ascontiguousarray(
        np.concatenate((source_terms, receiver_terms)),
        dtype=np.float64,
    )


def _core_node_input_model_from_app(input_model: RefractionStaticInputModel) -> Any:
    core = core_input_model_from_app(input_model)
    node_id = _input_node_id(input_model)
    node_count = int(node_id.shape[0])
    row_index = np.arange(node_count, dtype=np.int64)
    true_mask = np.ones(node_count, dtype=bool)
    zeros = np.zeros(node_count, dtype=np.float64)
    node_key = np.asarray(node_id, dtype=object)
    node_id_i64 = np.ascontiguousarray(node_id, dtype=np.int64)
    node_x = np.ascontiguousarray(input_model.node_x_m, dtype=np.float64)
    node_y = np.ascontiguousarray(input_model.node_y_m, dtype=np.float64)
    node_elevation = np.ascontiguousarray(
        input_model.node_elevation_m,
        dtype=np.float64,
    )
    return replace(
        core,
        n_traces=node_count,
        sorted_trace_index=row_index,
        pick_time_s_sorted=zeros,
        valid_pick_mask_sorted=true_mask,
        valid_observation_mask_sorted=true_mask,
        source_id_sorted=node_id_i64,
        receiver_id_sorted=node_id_i64,
        source_x_m_sorted=node_x,
        source_y_m_sorted=node_y,
        receiver_x_m_sorted=node_x,
        receiver_y_m_sorted=node_y,
        source_elevation_m_sorted=node_elevation,
        receiver_elevation_m_sorted=node_elevation,
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=zeros,
        offset_m_sorted=zeros,
        distance_m_sorted=zeros,
        source_endpoint_key_sorted=node_key,
        receiver_endpoint_key_sorted=node_key,
        source_node_id_sorted=node_id_i64,
        receiver_node_id_sorted=node_id_i64,
        source_endpoint_id_sorted=node_id_i64,
        receiver_endpoint_id_sorted=node_id_i64,
    )


def _core_multilayer_solve_result_from_app(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    solve_result: RefractionMultiLayerSolveResult,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    endpoint_terms: str,
    trace_domain: bool = False,
) -> CoreRefractionMultilayerTimeTermSolveResult:
    node_domain = endpoint_terms == 'node' and not trace_domain
    trace_count = (
        int(_input_node_id(input_model).shape[0])
        if node_domain
        else int(input_model.n_traces)
    )
    core_layers = tuple(
        _core_layer_result_from_app(
            input_model=input_model,
            model=model,
            layer=layer,
            resolved_first_layer=resolved_first_layer,
            endpoint_terms=endpoint_terms,
            trace_count=trace_count,
            trace_domain=trace_domain,
        )
        for layer in solve_result.layer_results
    )
    if node_domain:
        modeled_pick_time_s = np.zeros(trace_count, dtype=np.float64)
        residual_s = np.zeros(trace_count, dtype=np.float64)
        residual_ms = np.zeros(trace_count, dtype=np.float64)
        used_mask = np.ones(trace_count, dtype=bool)
        rejected_mask = np.zeros(trace_count, dtype=bool)
        layer_kind = np.full(trace_count, '', dtype=_STATUS_DTYPE)
        rejection_reason = np.full(trace_count, '', dtype=_STATUS_DTYPE)
        velocity_m_s = np.full(trace_count, np.nan, dtype=np.float64)
        layer_masks = _core_node_layer_observation_masks_from_app(
            input_model=input_model,
            trace_count=trace_count,
        )
    else:
        modeled_pick_time_s = _required_multilayer_sorted_float_array(
            solve_result,
            'modeled_pick_time_s_sorted',
        )
        residual_s = _required_multilayer_sorted_float_array(
            solve_result,
            'residual_s_sorted',
        )
        residual_ms = np.ascontiguousarray(residual_s * 1000.0, dtype=np.float64)
        used_mask = _required_multilayer_sorted_bool_array(
            solve_result,
            'used_observation_mask_sorted',
        )
        rejected_mask = _required_multilayer_sorted_bool_array(
            solve_result,
            'rejected_observation_mask_sorted',
        )
        layer_kind = _required_multilayer_sorted_string_array(
            solve_result,
            'layer_kind_sorted',
        )
        rejection_reason = _required_multilayer_sorted_string_array(
            solve_result,
            'rejection_reason_sorted',
        )
        velocity_m_s = _required_multilayer_sorted_float_array(
            solve_result,
            'velocity_m_s_sorted',
        )
        layer_masks = input_model.layer_observation_masks
    return CoreRefractionMultilayerTimeTermSolveResult(
        layer_results=core_layers,
        layer_result_by_kind={layer.layer_kind: layer for layer in core_layers},
        layer_observation_masks=layer_masks,
        modeled_pick_time_s_sorted=np.ascontiguousarray(
            modeled_pick_time_s,
            dtype=np.float64,
        ),
        residual_s_sorted=np.ascontiguousarray(residual_s, dtype=np.float64),
        residual_ms_sorted=np.ascontiguousarray(residual_ms, dtype=np.float64),
        used_observation_mask_sorted=np.ascontiguousarray(used_mask, dtype=bool),
        rejected_observation_mask_sorted=np.ascontiguousarray(
            rejected_mask,
            dtype=bool,
        ),
        layer_kind_sorted=np.asarray(layer_kind).astype(_STATUS_DTYPE, copy=True),
        rejection_reason_sorted=np.asarray(rejection_reason).astype(
            _STATUS_DTYPE,
            copy=True,
        ),
        velocity_m_s_sorted=np.ascontiguousarray(velocity_m_s, dtype=np.float64),
        qc=dict(solve_result.qc),
    )


def _core_node_layer_observation_masks_from_app(
    *,
    input_model: RefractionStaticInputModel,
    trace_count: int,
) -> RefractionLayerObservationMasks | None:
    masks = input_model.layer_observation_masks
    if masks is None:
        return masks
    enabled_kind = [str(kind) for kind in np.asarray(masks.layer_kind).tolist()]
    true_mask = np.ones(trace_count, dtype=bool)
    empty_reason = np.full(trace_count, '', dtype=_STATUS_DTYPE)
    return replace(
        masks,
        layer_used_mask_sorted={
            kind: np.array(true_mask, dtype=bool, copy=True)
            for kind in enabled_kind
        },
        layer_rejection_reason_sorted={
            kind: np.array(empty_reason, dtype=_STATUS_DTYPE, copy=True)
            for kind in enabled_kind
        },
        layer_candidate_count={kind: trace_count for kind in enabled_kind},
        layer_observation_count={kind: trace_count for kind in enabled_kind},
        overlapping_valid_observation_count=0,
        unassigned_valid_observation_count=0,
        unique_used_trace_count=trace_count,
        layer_membership_total_count=trace_count * len(enabled_kind),
    )


def _core_layer_result_from_app(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    layer: RefractionLayerSolveResult,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    endpoint_terms: str,
    trace_count: int,
    trace_domain: bool,
) -> CoreRefractionMultilayerTimeTermLayerResult:
    node_id = _input_node_id(input_model)
    cell_id = (
        np.asarray([], dtype=np.int64)
        if layer.cell_velocity_m_s is None
        else np.arange(np.asarray(layer.cell_velocity_m_s).shape[0], dtype=np.int64)
    )
    cell_velocity = (
        np.asarray([], dtype=np.float64)
        if layer.cell_velocity_m_s is None
        else np.ascontiguousarray(layer.cell_velocity_m_s, dtype=np.float64)
    )
    cell_slowness = (
        np.asarray([], dtype=np.float64)
        if layer.cell_slowness_s_per_m is None
        else np.ascontiguousarray(layer.cell_slowness_s_per_m, dtype=np.float64)
    )
    cell_status = (
        np.asarray([], dtype=_STATUS_DTYPE)
        if layer.cell_velocity_status is None
        else np.asarray(layer.cell_velocity_status).astype(_STATUS_DTYPE, copy=True)
    )
    core_solve = SimpleNamespace(
        node_id=node_id,
        node_half_intercept_time_s=_layer_terms_for_core_conversion(
            input_model=input_model,
            layer=layer,
            endpoint_terms=endpoint_terms,
        ),
        node_solution_status=np.full(node_id.shape, 'solved', dtype=_STATUS_DTYPE),
        bedrock_velocity_mode=layer.velocity_mode,
        bedrock_velocity_m_s=(
            float('nan')
            if layer.global_velocity_m_s is None
            else float(layer.global_velocity_m_s)
        ),
        bedrock_slowness_s_per_m=(
            float('nan')
            if layer.global_slowness_s_per_m is None
            else float(layer.global_slowness_s_per_m)
        ),
        cell_id=cell_id,
        cell_bedrock_velocity_m_s=cell_velocity,
        cell_bedrock_slowness_s_per_m=cell_slowness,
        cell_velocity_status=cell_status,
    )
    velocity_m_s_sorted = _core_layer_velocity_for_conversion(
        input_model=input_model,
        model=model,
        layer=layer,
        resolved_first_layer=resolved_first_layer,
        trace_count=trace_count,
        endpoint_terms=endpoint_terms,
        trace_domain=trace_domain,
    )
    return CoreRefractionMultilayerTimeTermLayerResult(
        layer_kind=layer.layer_kind,
        layer_index=layer.layer_index,
        layer=RefractionStaticLayerConfig(
            kind=layer.layer_kind,
            min_offset_m=None,
            max_offset_m=None,
            velocity_mode=layer.velocity_mode,
        ),
        solve_result=core_solve,
        velocity_m_s_sorted=velocity_m_s_sorted,
        rejection_reason_sorted=(
            np.full(trace_count, '', dtype=_STATUS_DTYPE)
            if layer.rejection_reason_sorted is None
            else _core_layer_rejection_reason_from_app(
                layer=layer,
                trace_count=trace_count,
                endpoint_terms=endpoint_terms,
            )
        ),
        velocity_order_valid_mask_sorted=np.ones(trace_count, dtype=bool),
    )


def _core_layer_velocity_for_conversion(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    layer: RefractionLayerSolveResult,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    trace_count: int,
    endpoint_terms: str,
    trace_domain: bool,
) -> np.ndarray:
    if layer.global_velocity_m_s is not None:
        return np.full(trace_count, layer.global_velocity_m_s, dtype=np.float64)
    if (
        trace_domain
        and layer.layer_kind == 'v2_t1'
        and layer.velocity_mode == 'solve_cell'
    ):
        return _core_layer_row_velocity_from_app(
            input_model=input_model,
            model=model,
            layer=layer,
            resolved_first_layer=resolved_first_layer,
            trace_count=trace_count,
            endpoint_terms=endpoint_terms,
        )
    if trace_domain:
        return _core_velocity_array_for_trace_count(
            layer.row_midpoint_velocity_m_s,
            trace_count=trace_count,
            name=f'{layer.layer_kind}.row_midpoint_velocity_m_s',
        )
    return _core_layer_row_velocity_from_app(
        input_model=input_model,
        model=model,
        layer=layer,
        resolved_first_layer=resolved_first_layer,
        trace_count=trace_count,
        endpoint_terms=endpoint_terms,
    )


def _core_layer_row_velocity_from_app(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    layer: RefractionLayerSolveResult,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    trace_count: int,
    endpoint_terms: str,
) -> np.ndarray:
    if layer.layer_kind == 'v2_t1' and layer.velocity_mode == 'solve_cell':
        v2 = _build_v2_static_model(
            input_model=input_model,
            model=model,
            layer=layer,
            node_id=_input_node_id(input_model),
            source=_endpoint_metadata(input_model, endpoint='source'),
            receiver=_endpoint_metadata(input_model, endpoint='receiver'),
            v1_m_s=_positive_float(
                resolved_first_layer.weathering_velocity_m_s,
                name='resolved_first_layer.weathering_velocity_m_s',
            ),
        )
        if endpoint_terms == 'node':
            return _core_velocity_array_for_trace_count(
                v2.node_v2_m_s,
                trace_count=trace_count,
                name='node_v2_m_s',
            )
        if endpoint_terms == 'source':
            return _core_velocity_array_for_trace_count(
                v2.source_v2_m_s_sorted,
                trace_count=trace_count,
                name='source_v2_m_s_sorted',
            )
        if endpoint_terms == 'receiver':
            return _core_velocity_array_for_trace_count(
                v2.receiver_v2_m_s_sorted,
                trace_count=trace_count,
                name='receiver_v2_m_s_sorted',
            )
    if endpoint_terms == 'node':
        return np.full(trace_count, np.nan, dtype=np.float64)
    return _core_velocity_array_for_trace_count(
        layer.row_midpoint_velocity_m_s,
        trace_count=trace_count,
        name=f'{layer.layer_kind}.row_midpoint_velocity_m_s',
    )


def _core_velocity_array_for_trace_count(
    value: np.ndarray | None,
    *,
    trace_count: int,
    name: str,
) -> np.ndarray:
    if value is None:
        raise RefractionMultiLayerSolveError(f'{name} is required')
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.shape != (trace_count,):
        raise RefractionMultiLayerSolveError(
            f'{name} shape must match conversion trace count'
        )
    return array


def _core_layer_rejection_reason_from_app(
    *,
    layer: RefractionLayerSolveResult,
    trace_count: int,
    endpoint_terms: str,
) -> np.ndarray:
    if endpoint_terms == 'node':
        return np.full(trace_count, '', dtype=_STATUS_DTYPE)
    return np.asarray(layer.rejection_reason_sorted).astype(_STATUS_DTYPE, copy=True)


def _layer_terms_for_core_conversion(
    input_model: RefractionStaticInputModel,
    layer: RefractionLayerSolveResult,
    endpoint_terms: str,
) -> np.ndarray:
    node_id = _input_node_id(input_model)
    values = _layer_node_terms(layer, shape=node_id.shape, name=layer.layer_kind)
    if endpoint_terms == 'node':
        return values
    source_key, source_node = _unique_endpoint_key_nodes(
        input_model.source_endpoint_key_sorted,
        input_model.source_node_id_sorted,
    )
    receiver_key, receiver_node = _unique_endpoint_key_nodes(
        input_model.receiver_endpoint_key_sorted,
        input_model.receiver_node_id_sorted,
    )
    output = np.array(values, dtype=np.float64, copy=True, order='C')
    node_pos = {int(node): index for index, node in enumerate(node_id.tolist())}
    if layer.source_time_term_s.shape != source_key.shape:
        raise RefractionMultiLayerSolveError(
            f'{layer.layer_kind} source endpoint time-term shape mismatch'
        )
    if layer.receiver_time_term_s.shape != receiver_key.shape:
        raise RefractionMultiLayerSolveError(
            f'{layer.layer_kind} receiver endpoint time-term shape mismatch'
        )
    if endpoint_terms == 'source':
        term_node = source_node
        terms = layer.source_time_term_s
    elif endpoint_terms == 'receiver':
        term_node = receiver_node
        terms = layer.receiver_time_term_s
    else:
        raise RefractionMultiLayerSolveError(
            f'unsupported endpoint terms: {endpoint_terms}'
        )
    for endpoint_node, term in zip(
        term_node.tolist(),
        np.asarray(terms, dtype=np.float64).tolist(),
        strict=True,
    ):
        pos = node_pos.get(int(endpoint_node))
        if pos is not None:
            output[pos] = float(term)
    return np.ascontiguousarray(output, dtype=np.float64)


def _require_multilayer_t1lsst_layers(
    normalized_layers: tuple[RefractionStaticLayerConfig, ...],
) -> None:
    kinds = tuple(config.kind for config in normalized_layers)
    if kinds not in (('v2_t1', 'v3_t2'), ('v2_t1', 'v3_t2', 'vsub_t3')):
        enabled_text = ', '.join(kinds) if kinds else 'none'
        raise RefractionMultiLayerSolveError(
            'multi-layer T1LSST statics requires enabled layers v2_t1 and '
            'v3_t2 for layer_count=2 or v2_t1, v3_t2, and vsub_t3 for '
            f'layer_count=3; enabled layer kinds={enabled_text}'
        )
    v3_mode = normalized_layers[1].velocity_mode
    if v3_mode not in ('solve_global', 'fixed_global'):
        raise RefractionMultiLayerSolveError(
            'multi-layer T1LSST statics currently requires global V3/T2 velocity; '
            'solve_cell V3/T2 is available only for internal layer solving'
        )
    if len(normalized_layers) == 3:
        vsub_mode = normalized_layers[2].velocity_mode
        if vsub_mode not in ('solve_global', 'fixed_global'):
            raise RefractionMultiLayerSolveError(
                'multi-layer T1LSST statics currently requires global '
                'Vsub/T3 velocity; solve_cell Vsub/T3 is available only for '
                'internal layer solving'
            )


def _total_weathering_thickness(
    *thicknesses_m: np.ndarray | None,
) -> np.ndarray:
    arrays = [
        np.asarray(thickness, dtype=np.float64)
        for thickness in thicknesses_m
        if thickness is not None
    ]
    if not arrays:
        raise RefractionMultiLayerSolveError(
            'at least one weathering thickness array is required'
        )
    total = np.array(arrays[0], dtype=np.float64, copy=True, order='C')
    for array in arrays[1:]:
        total = total + array
    return np.ascontiguousarray(total, dtype=np.float64)


def _cell_threshold_qc_from_layer(
    layer: RefractionLayerSolveResult,
) -> dict[str, Any]:
    if layer.velocity_mode != 'solve_cell':
        return {}
    return {key: layer.qc[key] for key in _CELL_THRESHOLD_QC_KEYS if key in layer.qc}


def _low_fold_cell_id_from_qc(qc: Mapping[str, Any]) -> np.ndarray:
    raw = qc.get('low_fold_cell_id', ())
    try:
        arr = np.asarray(raw, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise RefractionMultiLayerSolveError('low_fold_cell_id must be integer') from exc
    if arr.ndim != 1:
        raise RefractionMultiLayerSolveError('low_fold_cell_id must be one-dimensional')
    return np.ascontiguousarray(arr, dtype=np.int64)


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
    source_sh1 = _required_result_array(
        result.source_sh1_weathering_thickness_m,
        name='source_sh1_m',
    )
    receiver_sh2 = _required_result_array(
        result.receiver_sh2_weathering_thickness_m,
        name='receiver_sh2_m',
    )
    receiver_sh1 = _required_result_array(
        result.receiver_sh1_weathering_thickness_m,
        name='receiver_sh1_m',
    )
    has_3layer_components = any(
        value is not None
        for value in (
            result.source_t3_time_s,
            result.receiver_t3_time_s,
            result.source_sh3_weathering_thickness_m,
            result.receiver_sh3_weathering_thickness_m,
        )
    )
    source_t3 = receiver_t3 = source_sh3 = receiver_sh3 = None
    if has_3layer_components:
        source_t3 = _required_result_array(
            result.source_t3_time_s,
            name='source_t3_s',
        )
        receiver_t3 = _required_result_array(
            result.receiver_t3_time_s,
            name='receiver_t3_s',
        )
        source_sh3 = _required_result_array(
            result.source_sh3_weathering_thickness_m,
            name='source_sh3_m',
        )
        receiver_sh3 = _required_result_array(
            result.receiver_sh3_weathering_thickness_m,
            name='receiver_sh3_m',
        )
    layer_count = 3 if has_3layer_components else 2
    return RefractionMultiLayerStaticComponents(
        source_t1_s=np.ascontiguousarray(result.source_half_intercept_time_s),
        source_t2_s=source_t2,
        source_t3_s=source_t3,
        receiver_t1_s=np.ascontiguousarray(result.receiver_half_intercept_time_s),
        receiver_t2_s=receiver_t2,
        receiver_t3_s=receiver_t3,
        source_sh1_m=source_sh1,
        source_sh2_m=source_sh2,
        source_sh3_m=source_sh3,
        receiver_sh1_m=receiver_sh1,
        receiver_sh2_m=receiver_sh2,
        receiver_sh3_m=receiver_sh3,
        source_weathering_correction_s=np.ascontiguousarray(
            result.source_weathering_replacement_shift_s
        ),
        receiver_weathering_correction_s=np.ascontiguousarray(
            result.receiver_weathering_replacement_shift_s
        ),
        qc={
            'conversion_mode': 't1lsst_multilayer',
            'layer_count': layer_count,
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


def _core_sorted_float_array(
    value: object,
    *,
    n_traces: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (int(n_traces),):
        raise RefractionMultiLayerSolveError(f'core {name} shape mismatch')
    return np.ascontiguousarray(arr, dtype=np.float64)


def _core_sorted_bool_array(
    value: object,
    *,
    n_traces: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(value, dtype=bool)
    if arr.shape != (int(n_traces),):
        raise RefractionMultiLayerSolveError(f'core {name} shape mismatch')
    return np.ascontiguousarray(arr, dtype=bool)


def _core_sorted_string_array(
    value: object,
    *,
    n_traces: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(value).astype(_STATUS_DTYPE, copy=True)
    if arr.shape != (int(n_traces),):
        raise RefractionMultiLayerSolveError(f'core {name} shape mismatch')
    return np.ascontiguousarray(arr, dtype=_STATUS_DTYPE)


def _required_multilayer_sorted_float_array(
    solve_result: RefractionMultiLayerSolveResult,
    field_name: str,
) -> np.ndarray:
    value = getattr(solve_result, field_name)
    if value is None:
        raise RefractionMultiLayerSolveError(f'{field_name} is required')
    return _core_sorted_float_array(
        value,
        n_traces=_solve_result_trace_count(solve_result),
        name=field_name,
    )


def _required_multilayer_sorted_bool_array(
    solve_result: RefractionMultiLayerSolveResult,
    field_name: str,
) -> np.ndarray:
    value = getattr(solve_result, field_name)
    if value is None:
        raise RefractionMultiLayerSolveError(f'{field_name} is required')
    return _core_sorted_bool_array(
        value,
        n_traces=_solve_result_trace_count(solve_result),
        name=field_name,
    )


def _required_multilayer_sorted_string_array(
    solve_result: RefractionMultiLayerSolveResult,
    field_name: str,
) -> np.ndarray:
    value = getattr(solve_result, field_name)
    if value is None:
        raise RefractionMultiLayerSolveError(f'{field_name} is required')
    return _core_sorted_string_array(
        value,
        n_traces=_solve_result_trace_count(solve_result),
        name=field_name,
    )


def _solve_result_trace_count(solve_result: RefractionMultiLayerSolveResult) -> int:
    for layer in solve_result.layer_results:
        return int(np.asarray(layer.used_observation_mask_sorted).shape[0])
    raise RefractionMultiLayerSolveError('at least one layer result is required')


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
        row_velocity_m_s=np.full(input_model.n_traces, v2_m_s, dtype=np.float64),
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
    cell_options = refractor_cell_options_from_request(model.refractor_cell)
    grid = build_refraction_cell_grid(
        effective_refraction_cell_grid_config(cell_options)
    )
    n_total_cells = int(grid.cell_id.shape[0])
    cell_velocity_m_s = _cell_indexed_float_array(
        layer.cell_velocity_m_s,
        n_total_cells=n_total_cells,
        name='cell_velocity_m_s',
    )
    cell_slowness_s_per_m = _cell_indexed_float_array(
        layer.cell_slowness_s_per_m,
        n_total_cells=n_total_cells,
        name='cell_slowness_s_per_m',
    )
    cell_status = _cell_indexed_status_array(
        layer.cell_velocity_status,
        active_cell_id=layer.active_cell_id,
        n_total_cells=n_total_cells,
        name='cell_velocity_status',
    )
    node_cell, node_v2, node_status = _project_v2_to_points(
        grid=grid,
        refractor_cell=cell_options,
        x_m=input_model.node_x_m,
        y_m=input_model.node_y_m,
        active_cell_id=layer.active_cell_id,
        cell_velocity_m_s=cell_velocity_m_s,
        v1_m_s=v1_m_s,
        low_fold_cell_id=_low_fold_cell_id_from_qc(layer.qc),
    )
    source_cell, source_v2, source_status = _project_v2_to_points(
        grid=grid,
        refractor_cell=cell_options,
        x_m=source.x_m,
        y_m=source.y_m,
        active_cell_id=layer.active_cell_id,
        cell_velocity_m_s=cell_velocity_m_s,
        v1_m_s=v1_m_s,
        low_fold_cell_id=_low_fold_cell_id_from_qc(layer.qc),
    )
    receiver_cell, receiver_v2, receiver_status = _project_v2_to_points(
        grid=grid,
        refractor_cell=cell_options,
        x_m=receiver.x_m,
        y_m=receiver.y_m,
        active_cell_id=layer.active_cell_id,
        cell_velocity_m_s=cell_velocity_m_s,
        v1_m_s=v1_m_s,
        low_fold_cell_id=_low_fold_cell_id_from_qc(layer.qc),
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
        mode=cell_options.coordinate_mode,
        line_origin_x_m=cell_options.line_origin_x_m,
        line_origin_y_m=cell_options.line_origin_y_m,
        line_azimuth_deg=cell_options.line_azimuth_deg,
    )
    receiver_projected = project_refraction_cell_points(
        x_m=input_model.receiver_x_m_sorted,
        y_m=input_model.receiver_y_m_sorted,
        mode=cell_options.coordinate_mode,
        line_origin_x_m=cell_options.line_origin_x_m,
        line_origin_y_m=cell_options.line_origin_y_m,
        line_azimuth_deg=cell_options.line_azimuth_deg,
    )
    row_assignment = assign_observation_midpoint_cells(
        grid,
        source_x_m=source_projected.x_m,
        source_y_m=source_projected.y_m,
        receiver_x_m=receiver_projected.x_m,
        receiver_y_m=receiver_projected.y_m,
    )
    row_velocity_m_s = _row_velocity_from_cell_assignment(
        row_midpoint_cell_id=row_assignment.cell_id,
        cell_velocity_m_s=cell_velocity_m_s,
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
            cell_slowness_s_per_m,
            dtype=np.float64,
        ),
        cell_bedrock_velocity_m_s=np.ascontiguousarray(
            cell_velocity_m_s,
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
        row_velocity_m_s=row_velocity_m_s,
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
    low_fold_cell_id: np.ndarray,
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
    velocity_by_cell = _active_velocity_by_cell_id(
        active_cell_id=active_cell_id,
        cell_velocity_m_s=cell_velocity_m_s,
        n_total_cells=int(grid.cell_id.shape[0]),
    )
    v2 = np.full(cell_id.shape, np.nan, dtype=np.float64)
    low_fold_cells = {int(cell) for cell in np.asarray(low_fold_cell_id).tolist()}
    status = np.full(cell_id.shape, 'ok', dtype=_STATUS_DTYPE)
    for index, raw_cell in enumerate(cell_id.tolist()):
        cell = int(raw_cell)
        if cell < 0:
            status[index] = 'outside_refractor_cell_grid'
            continue
        velocity = velocity_by_cell.get(cell)
        if velocity is None:
            status[index] = (
                'low_fold_v2_cell' if cell in low_fold_cells else 'inactive_v2_cell'
            )
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


def _row_velocity_from_cell_assignment(
    *,
    row_midpoint_cell_id: np.ndarray,
    cell_velocity_m_s: np.ndarray,
) -> np.ndarray:
    cell_id = np.asarray(row_midpoint_cell_id, dtype=np.int64)
    velocity = np.asarray(cell_velocity_m_s, dtype=np.float64)
    out = np.full(cell_id.shape, np.nan, dtype=np.float64)
    valid = (cell_id >= 0) & (cell_id < int(velocity.shape[0]))
    out[valid] = velocity[cell_id[valid]]
    return np.ascontiguousarray(out, dtype=np.float64)


def _cell_indexed_float_array(
    value: np.ndarray,
    *,
    n_total_cells: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise RefractionMultiLayerSolveError(f'{name} must be one-dimensional')
    if array.shape != (n_total_cells,):
        raise RefractionMultiLayerSolveError(
            f'{name} must be indexed by cell_id and have length {n_total_cells}'
        )
    return np.ascontiguousarray(array, dtype=np.float64)


def _cell_indexed_status_array(
    value: np.ndarray | None,
    *,
    active_cell_id: np.ndarray,
    n_total_cells: int,
    name: str,
) -> np.ndarray:
    if value is None:
        status = np.full(n_total_cells, 'inactive', dtype=_STATUS_DTYPE)
        active = _cell_id_array_in_range(
            active_cell_id,
            n_total_cells=n_total_cells,
            name='active_cell_id',
        )
        status[active] = 'solved'
        return status
    status = np.asarray(value).astype(_STATUS_DTYPE, copy=True)
    if status.ndim != 1:
        raise RefractionMultiLayerSolveError(f'{name} must be one-dimensional')
    if status.shape != (n_total_cells,):
        raise RefractionMultiLayerSolveError(
            f'{name} must be indexed by cell_id and have length {n_total_cells}'
        )
    return status


def _active_velocity_by_cell_id(
    *,
    active_cell_id: np.ndarray,
    cell_velocity_m_s: np.ndarray,
    n_total_cells: int,
) -> dict[int, float]:
    active = _cell_id_array_in_range(
        active_cell_id,
        n_total_cells=n_total_cells,
        name='active_cell_id',
    )
    velocity = _cell_indexed_float_array(
        cell_velocity_m_s,
        n_total_cells=n_total_cells,
        name='cell_velocity_m_s',
    )
    return {int(cell): float(velocity[int(cell)]) for cell in active.tolist()}


def _cell_id_array_in_range(
    value: np.ndarray,
    *,
    n_total_cells: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if array.ndim != 1:
        raise RefractionMultiLayerSolveError(f'{name} must be one-dimensional')
    if array.size and (np.any(array < 0) or np.any(array >= n_total_cells)):
        raise RefractionMultiLayerSolveError(
            f'{name} contains a cell ID outside the refractor grid'
        )
    return np.ascontiguousarray(array, dtype=np.int64)


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


def _map_endpoint_int_to_trace_order(
    *,
    endpoint_key_sorted: np.ndarray,
    endpoint_key: np.ndarray,
    endpoint_values: np.ndarray,
    name: str,
) -> np.ndarray:
    key_to_index = _endpoint_key_to_index(endpoint_key)
    values = np.asarray(endpoint_values, dtype=np.int64)
    out = np.full(endpoint_key_sorted.shape, -1, dtype=np.int64)
    for index, raw_key in enumerate(np.asarray(endpoint_key_sorted, dtype=object).tolist()):
        endpoint_index = key_to_index.get(str(raw_key))
        if endpoint_index is None:
            raise RefractionMultiLayerSolveError(f'{name} missing endpoint {raw_key!s}')
        out[index] = int(values[endpoint_index])
    return np.ascontiguousarray(out, dtype=np.int64)


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


def _combine_source_receiver_trace_shift(
    *,
    source_shift: np.ndarray,
    receiver_shift: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_shift, dtype=np.float64)
    receiver = np.asarray(receiver_shift, dtype=np.float64)
    if source.shape != receiver.shape:
        raise RefractionMultiLayerSolveError(
            'source/receiver trace shift shape mismatch'
        )
    out = np.full(source.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(source) & np.isfinite(receiver)
    out[valid] = source[valid] + receiver[valid]
    return np.ascontiguousarray(out, dtype=np.float64)


def _combine_source_receiver_trace_status(
    *,
    source_status: np.ndarray,
    receiver_status: np.ndarray,
    trace_shift: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_status).astype(str, copy=False)
    receiver = np.asarray(receiver_status).astype(str, copy=False)
    shift = np.asarray(trace_shift, dtype=np.float64)
    if source.shape != receiver.shape or source.shape != shift.shape:
        raise RefractionMultiLayerSolveError(
            'source/receiver trace status shape mismatch'
        )
    out = np.asarray(source).astype(_STATUS_DTYPE, copy=True)
    source_ok = np.isin(source, list(_TRACE_OK_STATUSES))
    receiver_ok = np.isin(receiver, list(_TRACE_OK_STATUSES))
    out[source_ok & ~receiver_ok] = receiver[source_ok & ~receiver_ok]
    out[np.isin(out.astype(str, copy=False), list(_TRACE_OK_STATUSES))] = 'ok'
    out[(out == 'ok') & ~np.isfinite(shift)] = 'invalid_shift'
    return np.ascontiguousarray(out, dtype=_STATUS_DTYPE)


def _layer_index_sorted_from_kind(layer_kind_sorted: np.ndarray) -> np.ndarray:
    kinds = np.asarray(layer_kind_sorted).astype(str, copy=False)
    out = np.zeros(kinds.shape, dtype=np.int64)
    for kind, layer_index in _LAYER_INDEX_BY_KIND.items():
        out[kinds == kind] = int(layer_index)
    return np.ascontiguousarray(out, dtype=np.int64)


def _artifact_rejection_reason_from_core(rejection_reason_sorted: np.ndarray) -> np.ndarray:
    reason = np.asarray(rejection_reason_sorted).astype(_STATUS_DTYPE, copy=True)
    reason[reason.astype(str, copy=False) == 'robust_rejected'] = 'robust_outlier'
    return np.ascontiguousarray(reason, dtype=_STATUS_DTYPE)


def _robust_rejection_mask_from_reason(rejection_reason_sorted: np.ndarray) -> np.ndarray:
    reason = np.asarray(rejection_reason_sorted).astype(str, copy=False)
    return np.ascontiguousarray(
        (reason == 'robust_rejected') | (reason == 'robust_outlier'),
        dtype=bool,
    )


def _multilayer_layer_qc(
    *,
    layer_results: tuple[RefractionLayerSolveResult, ...],
    observation_gates: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for layer in layer_results:
        section = dict(layer.qc)
        gate = observation_gates.get(layer.layer_kind, {})
        if 'n_candidate_observations' in gate:
            section.setdefault(
                'n_candidate_observations',
                int(gate['n_candidate_observations']),
            )
        gate_used = int(gate.get('n_used_observations', 0))
        section['n_observation_gate_used_observations'] = gate_used
        section['n_used_observations'] = int(
            np.count_nonzero(np.asarray(layer.used_observation_mask_sorted, dtype=bool))
        )
        section['n_rejected_by_observation_gate'] = max(
            int(section.get('n_candidate_observations', gate_used)) - gate_used,
            0,
        )
        rejection_counts = gate.get('rejection_counts')
        if isinstance(rejection_counts, Mapping):
            section['observation_gate_rejection_counts'] = dict(rejection_counts)
        payload[layer.layer_kind] = section
    return payload


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


def _require_layer_observations(
    config: RefractionStaticLayerConfig,
    layer_masks: RefractionLayerObservationMasks,
) -> None:
    count = int(layer_masks.layer_observation_count.get(config.kind, 0))
    if count <= 0:
        raise RefractionMultiLayerSolveError(
            f'refraction layer {config.kind} has no valid observations'
        )


def _layer_design_matrix_artifact_dir(
    root: Path | None,
    config: RefractionStaticLayerConfig,
) -> Path | None:
    if root is None:
        return None
    return Path(root) / f'{_DESIGN_MATRIX_ARTIFACT_DIR_PREFIX}_{config.kind}'


def _write_failed_core_layer_design_matrix_diagnostics(
    *,
    input_model: RefractionStaticInputModel,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    normalized_layers: tuple[RefractionStaticLayerConfig, ...],
    layer_masks: RefractionLayerObservationMasks,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    job_dir: Path | None,
) -> None:
    if job_dir is None:
        return
    for config in normalized_layers:
        layer_dir = _layer_design_matrix_artifact_dir(job_dir, config)
        if layer_dir is None:
            continue
        design = build_refraction_static_design_matrix(
            input_model=_layer_input_model_for_design_diagnostics(
                input_model=input_model,
                layer_masks=layer_masks,
                config=config,
            ),
            model=_layer_model_request_for_design_diagnostics(
                model=model,
                resolved_first_layer=resolved_first_layer,
                config=config,
            ),
            resolved_first_layer=None,
            include_diagnostics=True,
            min_observations_per_node=solver.min_picks_per_node,
        )
        write_refraction_design_matrix_diagnostics_artifacts(layer_dir, design)
        _copy_layer_design_matrix_diagnostics_to_root_artifacts(
            root=job_dir,
            layer_dir=layer_dir,
            layer_kind=config.kind,
            layer_index=_layer_index(config.kind),
        )


def _layer_input_model_for_design_diagnostics(
    *,
    input_model: RefractionStaticInputModel,
    layer_masks: RefractionLayerObservationMasks,
    config: RefractionStaticLayerConfig,
) -> RefractionStaticInputModel:
    return replace(
        input_model,
        valid_observation_mask_sorted=layer_masks.layer_used_mask_sorted[config.kind],
        rejection_reason_sorted=layer_masks.layer_rejection_reason_sorted[config.kind],
        layer_observation_masks=None,
    )


def _layer_model_request_for_design_diagnostics(
    *,
    model: RefractionStaticModelRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer,
    config: RefractionStaticLayerConfig,
) -> RefractionStaticModelRequest:
    weathering_velocity = float(resolved_first_layer.weathering_velocity_m_s)
    refractor_cell: dict[str, Any] | None = None
    if config.velocity_mode == 'solve_cell':
        if model.refractor_cell is None:
            raise RefractionMultiLayerSolveError(
                'model.refractor_cell is required for solve_cell layers'
            )
        refractor_cell = model.refractor_cell.model_dump()
        if config.min_observations_per_cell is not None:
            refractor_cell['min_observations_per_cell'] = int(
                config.min_observations_per_cell
            )
        if config.smoothing_weight is not None:
            refractor_cell['velocity_smoothing_weight'] = float(
                config.smoothing_weight
            )
    payload: dict[str, Any] = {
        'method': 'gli_variable_thickness',
        'first_layer': {
            'mode': 'constant',
            'weathering_velocity_m_s': weathering_velocity,
        },
        'bedrock_velocity_mode': config.velocity_mode,
        'bedrock_velocity_m_s': config.fixed_velocity_m_s,
        'initial_bedrock_velocity_m_s': config.initial_velocity_m_s,
        'min_bedrock_velocity_m_s': (
            float(np.nextafter(weathering_velocity, np.inf))
            if config.min_velocity_m_s is None
            else float(config.min_velocity_m_s)
        ),
        'max_bedrock_velocity_m_s': (
            float(np.finfo(np.float64).max)
            if config.max_velocity_m_s is None
            else float(config.max_velocity_m_s)
        ),
        'refractor_cell': refractor_cell,
    }
    return RefractionStaticModelRequest.model_validate(payload)


def _copy_layer_design_matrix_diagnostics_to_root_artifacts(
    *,
    root: Path | None,
    layer_dir: Path | None,
    layer_kind: RefractionLayerKind,
    layer_index: int,
) -> None:
    """Expose layer diagnostics through disambiguated root-level artifacts."""
    if root is None or layer_dir is None:
        return
    root_path = Path(root)
    layer_path = Path(layer_dir)
    qc_source = layer_path / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME
    if qc_source.is_file():
        root_path.mkdir(parents=True, exist_ok=True)
        qc_payload = json.loads(qc_source.read_text(encoding='utf-8'))
        if not isinstance(qc_payload, dict):
            raise RefractionMultiLayerSolveError(
                'design matrix diagnostics QC artifact must contain a JSON object'
            )
        qc_payload = {
            **qc_payload,
            'layer_kind': layer_kind,
            'layer_index': int(layer_index),
            'source_artifact_dir': layer_path.name,
        }
        (
            root_path / refraction_design_matrix_layer_qc_json_name(layer_kind)
        ).write_text(
            json.dumps(qc_payload, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )

    csv_source = layer_path / REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME
    if csv_source.is_file():
        root_path.mkdir(parents=True, exist_ok=True)
        (
            root_path
            / refraction_design_matrix_layer_node_diagnostics_csv_name(layer_kind)
        ).write_bytes(csv_source.read_bytes())


def _layer_result_from_core_layer(
    *,
    input_model: RefractionStaticInputModel,
    source_endpoint_key: np.ndarray,
    source_node_id: np.ndarray,
    receiver_endpoint_key: np.ndarray,
    receiver_node_id: np.ndarray,
    core_layer: CoreRefractionMultilayerTimeTermLayerResult,
) -> RefractionLayerSolveResult:
    result = core_layer.solve_result
    velocity_mode = result.bedrock_velocity_mode
    is_cell = velocity_mode == 'solve_cell'
    global_velocity = None if is_cell else float(result.bedrock_velocity_m_s)
    global_slowness = None if is_cell else float(result.bedrock_slowness_s_per_m)
    active_cell_id, inactive_cell_id, cell_velocity_m_s, cell_slowness_s_per_m, (
        cell_velocity_status
    ) = _core_cell_arrays(result)
    source_time_term = _node_values_by_id(
        endpoint_node_id=source_node_id,
        layer_node_id=result.node_id,
        values=result.node_half_intercept_time_s,
        name=f'{core_layer.layer_kind}.source_time_term_s',
    )
    receiver_time_term = _node_values_by_id(
        endpoint_node_id=receiver_node_id,
        layer_node_id=result.node_id,
        values=result.node_half_intercept_time_s,
        name=f'{core_layer.layer_kind}.receiver_time_term_s',
    )
    rejected_by_robust = np.asarray(core_layer.rejection_reason_sorted).astype(
        str,
        copy=False,
    ) == 'robust_rejected'
    used_mask = np.asarray(result.used_observation_mask_sorted, dtype=bool)
    qc = {
        **result.qc,
        'layer_kind': core_layer.layer_kind,
        'layer_index': int(core_layer.layer_index),
        'velocity_mode': velocity_mode,
        'n_observations': int(np.count_nonzero(used_mask)),
        'n_sources': int(np.unique(input_model.source_node_id_sorted).shape[0]),
        'n_receivers': int(np.unique(input_model.receiver_node_id_sorted).shape[0]),
        'robust_iterations': int(result.qc.get('robust_iteration_count', 0)),
        'n_rejected_by_robust': int(np.count_nonzero(rejected_by_robust)),
        'residual_rms_ms': float(result.rms_residual_ms),
        'residual_mad_ms': _used_residual_mad_ms(
            result.residual_s_sorted,
            used_mask,
        ),
    }
    if core_layer.layer.min_observations_per_cell is not None:
        qc['min_observations_per_cell'] = int(
            core_layer.layer.min_observations_per_cell
        )
    if cell_velocity_m_s is not None:
        finite_velocity = np.asarray(cell_velocity_m_s, dtype=np.float64)
        finite_velocity = finite_velocity[np.isfinite(finite_velocity)]
        if finite_velocity.size:
            qc['cell_bedrock_velocity_median_m_s'] = float(np.median(finite_velocity))
    if cell_velocity_status is not None:
        low_fold = np.flatnonzero(
            np.asarray(cell_velocity_status).astype(str, copy=False) == 'low_fold'
        )
        qc['low_fold_cell_id'] = low_fold.astype(int).tolist()
        qc['n_low_fold_cells'] = int(low_fold.size)
        qc['n_observations_rejected_by_low_fold_cell'] = int(
            np.count_nonzero(
                np.asarray(core_layer.rejection_reason_sorted).astype(str, copy=False)
                == LOW_FOLD_CELL_REJECTION_REASON
            )
        )
    qc.update(
        _layer_velocity_qc_aliases(
            layer_kind=core_layer.layer_kind,
            global_velocity_m_s=global_velocity,
            global_slowness_s_per_m=global_slowness,
        )
    )
    return RefractionLayerSolveResult(
        layer_kind=core_layer.layer_kind,
        layer_index=int(core_layer.layer_index),
        velocity_mode=velocity_mode,
        source_time_term_s=source_time_term,
        receiver_time_term_s=receiver_time_term,
        node_time_term_s=np.ascontiguousarray(
            result.node_half_intercept_time_s,
            dtype=np.float64,
        ),
        global_velocity_m_s=global_velocity,
        global_slowness_s_per_m=global_slowness,
        cell_velocity_m_s=cell_velocity_m_s,
        cell_slowness_s_per_m=cell_slowness_s_per_m,
        trace_predicted_time_s_sorted=np.ascontiguousarray(
            result.modeled_pick_time_s_sorted,
            dtype=np.float64,
        ),
        trace_residual_s_sorted=np.ascontiguousarray(
            result.residual_s_sorted,
            dtype=np.float64,
        ),
        used_observation_mask_sorted=np.ascontiguousarray(
            result.used_observation_mask_sorted,
            dtype=bool,
        ),
        layer_status='solved',
        qc=qc,
        active_cell_id=active_cell_id,
        inactive_cell_id=inactive_cell_id,
        cell_velocity_status=cell_velocity_status,
        row_midpoint_cell_id=(
            None
            if result.row_midpoint_cell_id is None
            else np.ascontiguousarray(result.row_midpoint_cell_id, dtype=np.int64)
        ),
        row_midpoint_velocity_m_s=(
            None
            if core_layer.velocity_m_s_sorted is None
            else np.ascontiguousarray(
                core_layer.velocity_m_s_sorted,
                dtype=np.float64,
            )
        ),
        rejected_by_robust_mask_sorted=np.ascontiguousarray(rejected_by_robust),
        candidate_observation_mask_sorted=(
            np.asarray(core_layer.rejection_reason_sorted).astype(str, copy=False)
            != 'outside_layer_offset_gate'
        ),
        rejection_reason_sorted=np.asarray(
            core_layer.rejection_reason_sorted,
        ).astype(_STATUS_DTYPE, copy=True),
    )


def _used_residual_mad_ms(residual_s: np.ndarray, used_mask: np.ndarray) -> float:
    residual = np.asarray(residual_s, dtype=np.float64)
    used = np.asarray(used_mask, dtype=bool)
    values = residual[used & np.isfinite(residual)]
    if values.size == 0:
        return float('nan')
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)) * 1000.0)


def _node_values_by_id(
    *,
    endpoint_node_id: np.ndarray,
    layer_node_id: np.ndarray,
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    layer_nodes = np.asarray(layer_node_id, dtype=np.int64)
    layer_values = np.asarray(values, dtype=np.float64)
    if layer_nodes.shape != layer_values.shape:
        raise RefractionMultiLayerSolveError(f'{name} node/value shape mismatch')
    lookup = {
        int(node): float(value)
        for node, value in zip(layer_nodes.tolist(), layer_values.tolist(), strict=True)
    }
    out = np.full(np.asarray(endpoint_node_id).shape, np.nan, dtype=np.float64)
    for index, raw_node in enumerate(np.asarray(endpoint_node_id, dtype=np.int64).tolist()):
        try:
            out[index] = lookup[int(raw_node)]
        except KeyError as exc:
            raise RefractionMultiLayerSolveError(
                f'{name} missing node_id {int(raw_node)}'
            ) from exc
    return np.ascontiguousarray(out, dtype=np.float64)


def _core_cell_arrays(
    result: object,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if getattr(result, 'bedrock_velocity_mode') != 'solve_cell':
        return None, None, None, None, None
    raw_cell_id = np.asarray(getattr(result, 'cell_id'), dtype=np.int64)
    if raw_cell_id.ndim != 1:
        raise RefractionMultiLayerSolveError('core cell_id must be one-dimensional')
    n_total = _core_cell_count(result, raw_cell_id)
    velocity = np.full(n_total, np.nan, dtype=np.float64)
    slowness = np.full(n_total, np.nan, dtype=np.float64)
    status = np.full(n_total, 'inactive', dtype=_STATUS_DTYPE)
    raw_velocity = np.asarray(
        getattr(result, 'cell_bedrock_velocity_m_s'),
        dtype=np.float64,
    )
    raw_slowness = np.asarray(
        getattr(result, 'cell_bedrock_slowness_s_per_m'),
        dtype=np.float64,
    )
    raw_status = np.asarray(getattr(result, 'cell_velocity_status')).astype(
        _STATUS_DTYPE,
        copy=False,
    )
    if not (
        raw_velocity.shape == raw_cell_id.shape
        and raw_slowness.shape == raw_cell_id.shape
        and raw_status.shape == raw_cell_id.shape
    ):
        raise RefractionMultiLayerSolveError('core cell arrays must match cell_id')
    velocity[raw_cell_id] = raw_velocity
    slowness[raw_cell_id] = raw_slowness
    status[raw_cell_id] = raw_status
    active = raw_cell_id[
        np.isfinite(raw_velocity)
        & (raw_velocity > 0.0)
        & np.isin(raw_status.astype(str), ['solved', 'ok', 'clipped_lower', 'clipped_upper'])
    ]
    all_ids = np.arange(n_total, dtype=np.int64)
    inactive = np.setdiff1d(all_ids, active, assume_unique=False)
    return (
        np.ascontiguousarray(active, dtype=np.int64),
        np.ascontiguousarray(inactive, dtype=np.int64),
        np.ascontiguousarray(velocity, dtype=np.float64),
        np.ascontiguousarray(slowness, dtype=np.float64),
        np.ascontiguousarray(status, dtype=_STATUS_DTYPE),
    )


def _core_cell_count(result: object, cell_id: np.ndarray) -> int:
    qc = getattr(result, 'qc', {})
    if isinstance(qc, Mapping) and qc.get('n_total_cells') is not None:
        return int(qc['n_total_cells'])
    return int(np.max(cell_id)) + 1 if cell_id.size else 0


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
            'vsub_velocity_m_s': float(global_velocity_m_s),
            'slowness_sub_s_per_m': float(global_slowness_s_per_m),
            'vsub_slowness_s_per_m': float(global_slowness_s_per_m),
        }
    return {}


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


__all__ = [
    'RefractionMultiLayerSolveError',
    'RefractionMultiLayerStaticsWorkflowResult',
    'build_refraction_multilayer_weathering_replacement_statics',
    'compute_refraction_multilayer_datum_statics_from_input_model',
    'solve_refraction_multilayer_time_terms',
]
