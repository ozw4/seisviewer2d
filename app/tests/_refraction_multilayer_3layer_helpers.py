from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from app.api.schemas import (
    RefractionStaticApplyOptions,
    RefractionStaticApplyRequest,
    RefractionStaticConversionRequest,
    RefractionStaticDatumRequest,
    RefractionStaticLinkageRequest,
    RefractionStaticModelRequest,
    RefractionStaticPickSourceRequest,
    RefractionStaticSolverRequest,
)
from app.statics.refraction.artifacts import write_refraction_static_artifacts
from app.statics.refraction.application.datum import (
    build_refraction_datum_statics,
    write_refraction_datum_statics_artifacts,
)
from app.statics.refraction.application.design_matrix import (
    REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME,
    REFRACTION_DESIGN_MATRIX_QC_JSON_NAME,
    refraction_design_matrix_layer_node_diagnostics_csv_name,
    refraction_design_matrix_layer_qc_json_name,
)
from app.statics.refraction.application.multilayer_service import (
    RefractionMultiLayerStaticsWorkflowResult,
    _components_from_replacement,
    build_refraction_multilayer_weathering_replacement_statics,
)
from app.statics.refraction.domain.types import (
    RefractionEndpointTable,
    RefractionLayerKind,
    RefractionLayerSolveResult,
    RefractionMultiLayerSolveResult,
    RefractionStaticInputModel,
    ResolvedRefractionFirstLayer,
)
from app.tests._refraction_multilayer_synthetic import (
    SYNTHETIC_MULTILAYER_V1_M_S,
    SYNTHETIC_MULTILAYER_V2_M_S,
    SYNTHETIC_MULTILAYER_V3_M_S,
    SYNTHETIC_MULTILAYER_VSUB_M_S,
    SyntheticMultiLayerRefractionDataset,
    make_2d_straight_three_layer_refraction_dataset,
)

TIME_ATOL_S = 1.0e-8
THICKNESS_ATOL_M = 1.0e-5
STATIC_ATOL_S = 1.0e-8


def make_three_layer_dataset() -> SyntheticMultiLayerRefractionDataset:
    return make_2d_straight_three_layer_refraction_dataset()


def make_three_layer_input_model(
    dataset: SyntheticMultiLayerRefractionDataset,
) -> RefractionStaticInputModel:
    source_node_id = np.asarray(dataset.source_endpoint_node_id, dtype=np.int64)
    receiver_node_id = np.asarray(dataset.receiver_endpoint_node_id, dtype=np.int64)
    node_id = np.concatenate((source_node_id, receiver_node_id)).astype(np.int64)
    endpoint_table = RefractionEndpointTable(
        node_id=np.ascontiguousarray(node_id, dtype=np.int64),
        endpoint_id=np.ascontiguousarray(
            np.concatenate((dataset.source_endpoint_id, dataset.receiver_endpoint_id)),
            dtype=np.int64,
        ),
        x_m=np.ascontiguousarray(
            np.concatenate((dataset.source_endpoint_x_m, dataset.receiver_endpoint_x_m)),
            dtype=np.float64,
        ),
        y_m=np.ascontiguousarray(
            np.concatenate((dataset.source_endpoint_y_m, dataset.receiver_endpoint_y_m)),
            dtype=np.float64,
        ),
        elevation_m=np.ascontiguousarray(
            np.concatenate(
                (
                    dataset.source_endpoint_elevation_m,
                    dataset.receiver_endpoint_elevation_m,
                )
            ),
            dtype=np.float64,
        ),
        kind=np.asarray(
            ['source'] * int(source_node_id.size)
            + ['receiver'] * int(receiver_node_id.size),
            dtype='<U8',
        ),
        pick_count=_pick_count_by_node(
            node_id=node_id,
            source_node_id_sorted=dataset.source_node_id,
            receiver_node_id_sorted=dataset.receiver_node_id,
        ),
    )
    n_traces = int(dataset.sorted_trace_index.shape[0])
    return RefractionStaticInputModel(
        file_id=dataset.name,
        n_traces=n_traces,
        sorted_trace_index=np.ascontiguousarray(
            dataset.sorted_trace_index,
            dtype=np.int64,
        ),
        pick_time_s_sorted=np.ascontiguousarray(
            dataset.first_break_time_s,
            dtype=np.float64,
        ),
        valid_pick_mask_sorted=np.ascontiguousarray(dataset.valid_mask, dtype=bool),
        valid_observation_mask_sorted=np.ascontiguousarray(
            dataset.valid_mask,
            dtype=bool,
        ),
        source_id_sorted=np.ascontiguousarray(dataset.source_id, dtype=np.int64),
        receiver_id_sorted=np.ascontiguousarray(dataset.receiver_id, dtype=np.int64),
        source_x_m_sorted=np.ascontiguousarray(dataset.source_x_m, dtype=np.float64),
        source_y_m_sorted=np.ascontiguousarray(dataset.source_y_m, dtype=np.float64),
        receiver_x_m_sorted=np.ascontiguousarray(
            dataset.receiver_x_m,
            dtype=np.float64,
        ),
        receiver_y_m_sorted=np.ascontiguousarray(
            dataset.receiver_y_m,
            dtype=np.float64,
        ),
        source_elevation_m_sorted=np.ascontiguousarray(
            dataset.source_elevation_m,
            dtype=np.float64,
        ),
        receiver_elevation_m_sorted=np.ascontiguousarray(
            dataset.receiver_elevation_m,
            dtype=np.float64,
        ),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=np.ascontiguousarray(
            dataset.offset_m,
            dtype=np.float64,
        ),
        offset_m_sorted=np.ascontiguousarray(dataset.offset_m, dtype=np.float64),
        distance_m_sorted=np.ascontiguousarray(dataset.offset_m, dtype=np.float64),
        source_endpoint_key_sorted=np.asarray(
            dataset.source_endpoint_key,
            dtype=object,
        ),
        receiver_endpoint_key_sorted=np.asarray(
            dataset.receiver_endpoint_key,
            dtype=object,
        ),
        source_node_id_sorted=np.ascontiguousarray(
            dataset.source_node_id,
            dtype=np.int64,
        ),
        receiver_node_id_sorted=np.ascontiguousarray(
            dataset.receiver_node_id,
            dtype=np.int64,
        ),
        node_x_m=np.ascontiguousarray(endpoint_table.x_m, dtype=np.float64),
        node_y_m=np.ascontiguousarray(endpoint_table.y_m, dtype=np.float64),
        node_elevation_m=np.ascontiguousarray(
            endpoint_table.elevation_m,
            dtype=np.float64,
        ),
        node_kind=np.asarray(endpoint_table.kind, dtype='<U8'),
        rejection_reason_sorted=np.asarray(dataset.rejection_reason, dtype='<U32'),
        qc={'fixture': dataset.name},
        endpoint_table=endpoint_table,
        metadata={'coordinate_mode': dataset.coordinate_mode},
    )


def make_three_layer_model() -> RefractionStaticModelRequest:
    return RefractionStaticModelRequest.model_validate(
        {
            'method': 'multilayer_time_term',
            'first_layer': {
                'mode': 'constant',
                'weathering_velocity_m_s': SYNTHETIC_MULTILAYER_V1_M_S,
            },
            'layers': [
                {
                    'kind': 'v2_t1',
                    'enabled': True,
                    'min_offset_m': 250.0,
                    'max_offset_m': 900.0,
                    'velocity_mode': 'fixed_global',
                    'fixed_velocity_m_s': SYNTHETIC_MULTILAYER_V2_M_S,
                    'min_velocity_m_s': 1600.0,
                    'max_velocity_m_s': 3200.0,
                },
                {
                    'kind': 'v3_t2',
                    'enabled': True,
                    'min_offset_m': 1000.0,
                    'max_offset_m': 1900.0,
                    'velocity_mode': 'fixed_global',
                    'fixed_velocity_m_s': SYNTHETIC_MULTILAYER_V3_M_S,
                    'min_velocity_m_s': 2600.0,
                    'max_velocity_m_s': 4800.0,
                },
                {
                    'kind': 'vsub_t3',
                    'enabled': True,
                    'min_offset_m': 2200.0,
                    'max_offset_m': None,
                    'velocity_mode': 'fixed_global',
                    'fixed_velocity_m_s': SYNTHETIC_MULTILAYER_VSUB_M_S,
                    'min_velocity_m_s': 3600.0,
                    'max_velocity_m_s': 6200.0,
                },
            ],
        }
    )


def resolved_first_layer() -> ResolvedRefractionFirstLayer:
    return ResolvedRefractionFirstLayer(
        mode='constant',
        weathering_velocity_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        status='constant',
        qc={'weathering_velocity_m_s': SYNTHETIC_MULTILAYER_V1_M_S},
    )


def compute_three_layer_workflow(
    *,
    datum: RefractionStaticDatumRequest | None = None,
    job_dir: Path | None = None,
) -> tuple[
    SyntheticMultiLayerRefractionDataset,
    RefractionStaticInputModel,
    RefractionStaticModelRequest,
    RefractionMultiLayerStaticsWorkflowResult,
]:
    dataset = make_three_layer_dataset()
    input_model = make_three_layer_input_model(dataset)
    model = make_three_layer_model()
    solver = RefractionStaticSolverRequest(
        damping=0.0,
        robust={'enabled': False},
    )
    apply_options = RefractionStaticApplyOptions(max_abs_shift_ms=500.0)
    active_datum = datum if datum is not None else RefractionStaticDatumRequest(mode='none')
    solve_result = make_three_layer_solve_result(dataset)
    weathering_replacement = build_refraction_multilayer_weathering_replacement_statics(
        input_model=input_model,
        model=model,
        solve_result=solve_result,
        apply_options=apply_options,
        resolved_first_layer=resolved_first_layer(),
    )
    datum_result = build_refraction_datum_statics(
        weathering_replacement_result=weathering_replacement,
        datum=active_datum,
        apply_options=apply_options,
        resolved_first_layer=resolved_first_layer(),
    )
    datum_result = replace(
        datum_result,
        qc={
            **datum_result.qc,
            'method': 'multilayer_time_term',
            'conversion_mode': 't1lsst_multilayer',
            'layer_count': 3,
            'enabled_layer_kinds': ['v2_t1', 'v3_t2', 'vsub_t3'],
            'layers': solve_result.qc,
        },
    )
    components = _components_from_replacement(weathering_replacement)
    workflow = RefractionMultiLayerStaticsWorkflowResult(
        solve_result=solve_result,
        components=components,
        weathering_replacement_result=weathering_replacement,
        datum_result=datum_result,
    )
    if job_dir is not None:
        root = Path(job_dir)
        write_refraction_datum_statics_artifacts(root, datum_result)
        _write_fixture_design_matrix_diagnostics(root, solve_result)
        write_refraction_static_artifacts(
            result=datum_result,
            req=_artifact_request(
                input_model=input_model,
                model=model,
                solver=solver,
                datum=active_datum,
                apply_options=apply_options,
            ),
            job_dir=root,
            resolved_first_layer=resolved_first_layer(),
        )
    return dataset, input_model, model, workflow


def _write_fixture_design_matrix_diagnostics(
    root: Path,
    solve_result: RefractionMultiLayerSolveResult,
) -> None:
    for layer in solve_result.layer_results:
        layer_dir = root / f'refraction_design_matrix_{layer.layer_kind}'
        layer_dir.mkdir(parents=True, exist_ok=True)
        layer_qc = {
            'n_active_nodes': int(layer.node_time_term_s.shape[0]),
            'n_rows': int(np.count_nonzero(layer.used_observation_mask_sorted)),
        }
        (layer_dir / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME).write_text(
            json.dumps(layer_qc, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        (
            layer_dir / REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME
        ).write_text('node_id,status\n1,ok\n', encoding='utf-8')
        root_qc = {
            **layer_qc,
            'layer_kind': layer.layer_kind,
            'layer_index': int(layer.layer_index),
            'source_artifact_dir': layer_dir.name,
        }
        root_qc_path = root / refraction_design_matrix_layer_qc_json_name(
            layer.layer_kind
        )
        root_qc_path.write_text(
            json.dumps(root_qc, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        (
            root
            / refraction_design_matrix_layer_node_diagnostics_csv_name(
                layer.layer_kind
            )
        ).write_text('node_id,status\n1,ok\n', encoding='utf-8')


def make_three_layer_solve_result(
    dataset: SyntheticMultiLayerRefractionDataset,
) -> RefractionMultiLayerSolveResult:
    source_endpoint_key = np.asarray(dataset.source_endpoint_id, dtype=object)
    receiver_endpoint_key = np.asarray(dataset.receiver_endpoint_id, dtype=object)
    source_node_id = np.ascontiguousarray(
        dataset.source_endpoint_node_id,
        dtype=np.int64,
    )
    receiver_node_id = np.ascontiguousarray(
        dataset.receiver_endpoint_node_id,
        dtype=np.int64,
    )
    layers = (
        _layer_result(
            dataset=dataset,
            layer_kind='v2_t1',
            layer_index=1,
            velocity_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
            source_time_term_s=dataset.true_source_endpoint_t1_s,
            receiver_time_term_s=dataset.true_receiver_endpoint_t1_s,
            used_mask=dataset.expected_layer_mask_by_kind['v2_t1'],
        ),
        _layer_result(
            dataset=dataset,
            layer_kind='v3_t2',
            layer_index=2,
            velocity_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
            source_time_term_s=dataset.true_source_endpoint_t2_s,
            receiver_time_term_s=dataset.true_receiver_endpoint_t2_s,
            used_mask=dataset.expected_layer_mask_by_kind['v3_t2'],
        ),
        _layer_result(
            dataset=dataset,
            layer_kind='vsub_t3',
            layer_index=3,
            velocity_m_s=SYNTHETIC_MULTILAYER_VSUB_M_S,
            source_time_term_s=dataset.true_source_endpoint_t3_s,
            receiver_time_term_s=dataset.true_receiver_endpoint_t3_s,
            used_mask=dataset.expected_layer_mask_by_kind['vsub_t3'],
        ),
    )
    return RefractionMultiLayerSolveResult(
        enabled_layer_kinds=('v2_t1', 'v3_t2', 'vsub_t3'),
        layer_results=layers,
        source_endpoint_key=source_endpoint_key,
        receiver_endpoint_key=receiver_endpoint_key,
        source_node_id=source_node_id,
        receiver_node_id=receiver_node_id,
        qc={
            'enabled_layer_count': 3,
            'enabled_layer_kinds': ['v2_t1', 'v3_t2', 'vsub_t3'],
            'layers': {item.layer_kind: item.qc for item in layers},
        },
    )


def layer(
    result: RefractionMultiLayerSolveResult,
    kind: RefractionLayerKind,
) -> RefractionLayerSolveResult:
    for layer_result in result.layer_results:
        if layer_result.layer_kind == kind:
            return layer_result
    raise AssertionError(f'{kind} layer result was not returned')


def _layer_result(
    *,
    dataset: SyntheticMultiLayerRefractionDataset,
    layer_kind: RefractionLayerKind,
    layer_index: int,
    velocity_m_s: float,
    source_time_term_s: np.ndarray,
    receiver_time_term_s: np.ndarray,
    used_mask: np.ndarray,
) -> RefractionLayerSolveResult:
    node_time_term_s = np.ascontiguousarray(
        np.concatenate((source_time_term_s, receiver_time_term_s)),
        dtype=np.float64,
    )
    predicted = np.full(dataset.sorted_trace_index.shape, np.nan, dtype=np.float64)
    predicted[used_mask] = dataset.noiseless_first_break_time_s[used_mask]
    residual = np.full(dataset.sorted_trace_index.shape, np.nan, dtype=np.float64)
    residual[used_mask] = dataset.first_break_time_s[used_mask] - predicted[used_mask]
    return RefractionLayerSolveResult(
        layer_kind=layer_kind,
        layer_index=layer_index,
        velocity_mode='fixed_global',
        source_time_term_s=np.ascontiguousarray(source_time_term_s, dtype=np.float64),
        receiver_time_term_s=np.ascontiguousarray(
            receiver_time_term_s,
            dtype=np.float64,
        ),
        node_time_term_s=node_time_term_s,
        global_velocity_m_s=float(velocity_m_s),
        global_slowness_s_per_m=1.0 / float(velocity_m_s),
        cell_velocity_m_s=None,
        cell_slowness_s_per_m=None,
        trace_predicted_time_s_sorted=np.ascontiguousarray(
            predicted,
            dtype=np.float64,
        ),
        trace_residual_s_sorted=np.ascontiguousarray(residual, dtype=np.float64),
        used_observation_mask_sorted=np.ascontiguousarray(used_mask, dtype=bool),
        layer_status='solved',
        qc={
            'layer_kind': layer_kind,
            'layer_index': layer_index,
            'velocity_mode': 'fixed_global',
            'global_velocity_m_s': float(velocity_m_s),
            'n_used_observations': int(np.count_nonzero(used_mask)),
        },
    )


def _artifact_request(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    solver: RefractionStaticSolverRequest,
    datum: RefractionStaticDatumRequest,
    apply_options: RefractionStaticApplyOptions,
) -> RefractionStaticApplyRequest:
    return RefractionStaticApplyRequest(
        file_id=input_model.file_id,
        pick_source=RefractionStaticPickSourceRequest(kind='manual_memmap'),
        linkage=RefractionStaticLinkageRequest(mode='none'),
        model=model,
        solver=solver,
        datum=datum,
        conversion=RefractionStaticConversionRequest(
            mode='t1lsst_multilayer',
            layer_count=3,
        ),
        apply=apply_options,
    )


def _pick_count_by_node(
    *,
    node_id: np.ndarray,
    source_node_id_sorted: np.ndarray,
    receiver_node_id_sorted: np.ndarray,
) -> np.ndarray:
    counts = np.zeros(node_id.shape, dtype=np.int64)
    node_to_index = {int(node): index for index, node in enumerate(node_id.tolist())}
    for source_node, receiver_node in zip(
        source_node_id_sorted.tolist(),
        receiver_node_id_sorted.tolist(),
        strict=True,
    ):
        counts[node_to_index[int(source_node)]] += 1
        counts[node_to_index[int(receiver_node)]] += 1
    return np.ascontiguousarray(counts, dtype=np.int64)
