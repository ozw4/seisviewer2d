from __future__ import annotations

from typing import Any, Literal

import numpy as np

from app.api.schemas import (
    RefractionStaticApplyRequest,
    RefractionStaticFirstLayerRequest,
)
from app.statics.refraction.application.datum import build_refraction_datum_statics
from app.statics.refraction.domain.types import (
    RefractionDatumStaticsResult,
    RefractionEndpointTable,
    RefractionStaticInputModel,
)
from app.statics.refraction.application.weathering_replacement import (
    compute_weathering_replacement_statics_from_first_breaks,
)

SYNTHETIC_V1_M_S = 800.0
SYNTHETIC_V2_M_S = 2400.0
SYNTHETIC_V1_TOLERANCE_M_S = 20.0
SYNTHETIC_V2_TOLERANCE_M_S = 50.0
SYNTHETIC_T1_TOLERANCE_MS = 2.0
SYNTHETIC_SH1_TOLERANCE_M = 2.0
SYNTHETIC_WCOR_TOLERANCE_MS = 2.0

SYNTHETIC_NODE_ID = np.arange(12, dtype=np.int64)
SYNTHETIC_SOURCE_NODE_ID = np.arange(6, dtype=np.int64)
SYNTHETIC_RECEIVER_NODE_ID = np.arange(12, dtype=np.int64)
SYNTHETIC_NODE_X_M = SYNTHETIC_NODE_ID.astype(np.float64) * 60.0
SYNTHETIC_NODE_SH1_M = np.asarray(
    [8.0, 11.0, 14.0, 10.0, 16.0, 13.0, 18.0, 21.0, 15.0, 19.0, 12.0, 17.0],
    dtype=np.float64,
)
SYNTHETIC_NODE_SURFACE_ELEVATION_M = 80.0 + SYNTHETIC_NODE_SH1_M
SYNTHETIC_NODE_T1_S = (
    SYNTHETIC_NODE_SH1_M
    * np.sqrt(SYNTHETIC_V2_M_S**2 - SYNTHETIC_V1_M_S**2)
    / (SYNTHETIC_V1_M_S * SYNTHETIC_V2_M_S)
)
SYNTHETIC_NODE_WCOR_S = SYNTHETIC_NODE_SH1_M * (
    (1.0 / SYNTHETIC_V2_M_S) - (1.0 / SYNTHETIC_V1_M_S)
)

SYNTHETIC_CELL_SIZE_X_M = 100.0
SYNTHETIC_CELL_V2_M_S = np.asarray([2200.0, 2600.0, 3000.0], dtype=np.float64)
SYNTHETIC_CELL_V2_TOLERANCE_M_S = 1.0e-3
SYNTHETIC_CELL_NODE_ID = np.arange(9, dtype=np.int64)
SYNTHETIC_CELL_NODE_X_M = np.asarray(
    [25.0, 50.0, 75.0, 125.0, 150.0, 175.0, 225.0, 250.0, 275.0],
    dtype=np.float64,
)
SYNTHETIC_CELL_NODE_CELL_ID = (
    SYNTHETIC_CELL_NODE_X_M // SYNTHETIC_CELL_SIZE_X_M
).astype(np.int64)
SYNTHETIC_CELL_NODE_SH1_M = np.asarray(
    [8.0, 11.0, 14.0, 10.0, 16.0, 13.0, 18.0, 21.0, 15.0],
    dtype=np.float64,
)
SYNTHETIC_CELL_NODE_SURFACE_ELEVATION_M = 80.0 + SYNTHETIC_CELL_NODE_SH1_M
SYNTHETIC_CELL_NODE_T1_S = (
    SYNTHETIC_CELL_NODE_SH1_M
    * np.sqrt(
        SYNTHETIC_CELL_V2_M_S[SYNTHETIC_CELL_NODE_CELL_ID] ** 2
        - SYNTHETIC_V1_M_S**2
    )
    / (
        SYNTHETIC_CELL_V2_M_S[SYNTHETIC_CELL_NODE_CELL_ID]
        * SYNTHETIC_V1_M_S
    )
)
SYNTHETIC_CELL_NODE_WCOR_S = SYNTHETIC_CELL_NODE_SH1_M * (
    (1.0 / SYNTHETIC_CELL_V2_M_S[SYNTHETIC_CELL_NODE_CELL_ID])
    - (1.0 / SYNTHETIC_V1_M_S)
)


def synthetic_first_layer_request() -> RefractionStaticFirstLayerRequest:
    return RefractionStaticFirstLayerRequest.model_validate(
        {
            'mode': 'estimate_direct_arrival',
            'min_weathering_velocity_m_s': 500.0,
            'max_weathering_velocity_m_s': 1200.0,
            'min_direct_offset_m': 30.0,
            'max_direct_offset_m': 360.0,
            'min_picks_per_fit': 5,
            'min_groups': int(SYNTHETIC_SOURCE_NODE_ID.shape[0]),
            'robust_enabled': True,
            'robust_threshold': 3.5,
        }
    )


def synthetic_refraction_apply_request(
    *,
    first_layer_mode: Literal['constant', 'estimate_direct_arrival'] = 'constant',
    conversion_mode: Literal['existing', 't1lsst_1layer'] = 'existing',
) -> RefractionStaticApplyRequest:
    model: dict[str, Any] = {
        'method': 'gli_variable_thickness',
        'bedrock_velocity_mode': 'solve_global',
        'bedrock_velocity_m_s': None,
        'initial_bedrock_velocity_m_s': SYNTHETIC_V2_M_S,
        'min_bedrock_velocity_m_s': 1200.0,
        'max_bedrock_velocity_m_s': 6000.0,
        'max_weathering_thickness_m': None,
    }
    if first_layer_mode == 'constant':
        model['weathering_velocity_m_s'] = SYNTHETIC_V1_M_S
    else:
        model['weathering_velocity_m_s'] = None
        model['first_layer'] = synthetic_first_layer_request().model_dump(
            mode='json',
            exclude={'weathering_velocity_m_s'},
        )

    return RefractionStaticApplyRequest.model_validate(
        {
            'file_id': 'synthetic-refraction-line',
            'key1_byte': 189,
            'key2_byte': 193,
            'pick_source': {
                'kind': 'batch_predicted_npz',
                'job_id': 'synthetic-first-break-job',
                'artifact_name': 'predicted_picks_time_s.npz',
            },
            'linkage': {'mode': 'none'},
            'model': model,
            'moveout': {
                'model': 'head_wave_linear_offset',
                'distance_source': 'geometry',
                'offset_byte': None,
                'min_offset_m': None,
                'max_offset_m': None,
                'allow_missing_offset': False,
                'max_geometry_offset_mismatch_m': None,
            },
            'solver': {
                'damping': 0.0,
                'min_picks_per_node': 1,
                'max_abs_half_intercept_time_ms': 200.0,
                'robust': {
                    'enabled': False,
                    'method': 'mad',
                    'threshold': 3.5,
                    'max_iterations': 5,
                    'min_used_fraction': 0.5,
                    'min_used_observations': 1,
                },
            },
            'datum': {'mode': 'none'},
            'conversion': {'mode': conversion_mode},
            'apply': {
                'mode': 'refraction_from_raw',
                'interpolation': 'linear',
                'fill_value': 0.0,
                'max_abs_shift_ms': 250.0,
                'output_dtype': 'float32',
                'register_corrected_file': False,
            },
        }
    )


def synthetic_refracted_arrival_input_model() -> RefractionStaticInputModel:
    source, receiver, distance = _synthetic_trace_geometry()
    picks = (
        SYNTHETIC_NODE_T1_S[source]
        + SYNTHETIC_NODE_T1_S[receiver]
        + distance / SYNTHETIC_V2_M_S
    )
    return _synthetic_input_model(source=source, receiver=receiver, distance=distance, picks=picks)


def synthetic_direct_arrival_input_model() -> RefractionStaticInputModel:
    source, receiver, distance = _synthetic_trace_geometry()
    source_intercept_s = 0.008 + SYNTHETIC_SOURCE_NODE_ID.astype(np.float64) * 0.0007
    noise_s = 0.00015 * np.sin(np.arange(distance.shape[0], dtype=np.float64) * 1.7)
    picks = source_intercept_s[source] + distance / SYNTHETIC_V1_M_S + noise_s
    outlier_index = int(np.flatnonzero((source == 2) & (distance == 120.0))[0])
    picks[outlier_index] += 0.018
    return _synthetic_input_model(source=source, receiver=receiver, distance=distance, picks=picks)


def run_synthetic_refraction_statics(
    *,
    state: Any | None = None,
    req: RefractionStaticApplyRequest | None = None,
    input_model: RefractionStaticInputModel | None = None,
) -> RefractionDatumStaticsResult:
    active_req = req or synthetic_refraction_apply_request()
    active_input = input_model or synthetic_refracted_arrival_input_model()
    replacement = compute_weathering_replacement_statics_from_first_breaks(
        req=active_req,
        state=state,
        input_model=active_input,
    )
    return build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=active_req.datum,
        apply_options=active_req.apply,
        state=state,
        file_id=active_req.file_id,
        key1_byte=active_req.key1_byte,
        key2_byte=active_req.key2_byte,
    )


def synthetic_cell_refraction_apply_request(
    *,
    robust_enabled: bool = False,
    velocity_smoothing_weight: float = 0.0,
    conversion_mode: Literal['existing', 't1lsst_1layer'] = 't1lsst_1layer',
) -> RefractionStaticApplyRequest:
    smoothing_reference_distance_m = (
        SYNTHETIC_CELL_SIZE_X_M if velocity_smoothing_weight > 0.0 else None
    )
    return RefractionStaticApplyRequest.model_validate(
        {
            'file_id': 'synthetic-cell-refraction-line',
            'key1_byte': 189,
            'key2_byte': 193,
            'pick_source': {
                'kind': 'batch_predicted_npz',
                'job_id': 'synthetic-cell-first-break-job',
                'artifact_name': 'predicted_picks_time_s.npz',
            },
            'linkage': {'mode': 'none'},
            'model': {
                'method': 'gli_variable_thickness',
                'weathering_velocity_m_s': SYNTHETIC_V1_M_S,
                'bedrock_velocity_mode': 'solve_cell',
                'bedrock_velocity_m_s': None,
                'initial_bedrock_velocity_m_s': 2600.0,
                'min_bedrock_velocity_m_s': 1200.0,
                'max_bedrock_velocity_m_s': 6000.0,
                'max_weathering_thickness_m': None,
                'refractor_cell': {
                    'number_of_cell_x': int(SYNTHETIC_CELL_V2_M_S.shape[0]),
                    'size_of_cell_x_m': SYNTHETIC_CELL_SIZE_X_M,
                    'x_coordinate_origin_m': 0.0,
                    'number_of_cell_y': 1,
                    'size_of_cell_y_m': None,
                    'y_coordinate_origin_m': 0.0,
                    'assignment_mode': 'midpoint',
                    'outside_grid_policy': 'reject',
                    'min_observations_per_cell': 1,
                    'velocity_smoothing_weight': velocity_smoothing_weight,
                    'smoothing_reference_distance_m': smoothing_reference_distance_m,
                },
            },
            'moveout': {
                'model': 'head_wave_linear_offset',
                'distance_source': 'geometry',
                'offset_byte': None,
                'min_offset_m': None,
                'max_offset_m': None,
                'allow_missing_offset': False,
                'max_geometry_offset_mismatch_m': None,
            },
            'solver': {
                'damping': 0.0,
                'min_picks_per_node': 1,
                'max_abs_half_intercept_time_ms': 200.0,
                'robust': {
                    'enabled': robust_enabled,
                    'method': 'mad',
                    'threshold': 3.5,
                    'max_iterations': 5,
                    'min_used_fraction': 0.5,
                    'min_used_observations': 1,
                },
            },
            'datum': {'mode': 'none'},
            'conversion': {'mode': conversion_mode},
            'apply': {
                'mode': 'refraction_from_raw',
                'interpolation': 'linear',
                'fill_value': 0.0,
                'max_abs_shift_ms': 250.0,
                'output_dtype': 'float32',
                'register_corrected_file': False,
            },
        }
    )


def synthetic_cell_refracted_arrival_input_model(
    *,
    allowed_midpoint_cell_ids: tuple[int, ...] | None = None,
    include_outside_grid_observation: bool = False,
) -> RefractionStaticInputModel:
    source, receiver, distance, midpoint_cell, picks = _synthetic_cell_trace_geometry(
        allowed_midpoint_cell_ids=allowed_midpoint_cell_ids,
    )
    node_id = SYNTHETIC_CELL_NODE_ID
    node_x = SYNTHETIC_CELL_NODE_X_M
    node_sh1 = SYNTHETIC_CELL_NODE_SH1_M
    node_t1 = SYNTHETIC_CELL_NODE_T1_S

    if include_outside_grid_observation:
        node_id = np.arange(node_id.shape[0] + 2, dtype=np.int64)
        outside_x = np.asarray([-50.0, -25.0], dtype=np.float64)
        outside_sh1 = np.asarray([9.0, 10.0], dtype=np.float64)
        outside_t1 = (
            outside_sh1
            * np.sqrt(SYNTHETIC_CELL_V2_M_S[0] ** 2 - SYNTHETIC_V1_M_S**2)
            / (SYNTHETIC_CELL_V2_M_S[0] * SYNTHETIC_V1_M_S)
        )
        node_x = np.concatenate((node_x, outside_x))
        node_sh1 = np.concatenate((node_sh1, outside_sh1))
        node_t1 = np.concatenate((node_t1, outside_t1))
        outside_source = int(node_id[-2])
        outside_receiver = int(node_id[-1])
        outside_distance = abs(float(node_x[-1]) - float(node_x[-2]))
        outside_pick = (
            float(node_t1[-2])
            + float(node_t1[-1])
            + outside_distance / float(SYNTHETIC_CELL_V2_M_S[0])
        )
        source = np.concatenate((source, np.asarray([outside_source], dtype=np.int64)))
        receiver = np.concatenate(
            (receiver, np.asarray([outside_receiver], dtype=np.int64))
        )
        distance = np.concatenate(
            (distance, np.asarray([outside_distance], dtype=np.float64))
        )
        midpoint_cell = np.concatenate(
            (midpoint_cell, np.asarray([-1], dtype=np.int64))
        )
        picks = np.concatenate((picks, np.asarray([outside_pick], dtype=np.float64)))

    return _synthetic_cell_input_model(
        node_id=node_id,
        node_x_m=node_x,
        node_sh1_m=node_sh1,
        source=source,
        receiver=receiver,
        distance=distance,
        picks=picks,
        midpoint_cell_id=midpoint_cell,
    )


def synthetic_cell_midpoint_cell_id_sorted(
    input_model: RefractionStaticInputModel,
) -> np.ndarray:
    midpoint_x = 0.5 * (
        np.asarray(input_model.source_x_m_sorted, dtype=np.float64)
        + np.asarray(input_model.receiver_x_m_sorted, dtype=np.float64)
    )
    return np.floor(midpoint_x / SYNTHETIC_CELL_SIZE_X_M).astype(np.int64)


def run_synthetic_cell_refraction_statics(
    *,
    state: Any | None = None,
    req: RefractionStaticApplyRequest | None = None,
    input_model: RefractionStaticInputModel | None = None,
) -> RefractionDatumStaticsResult:
    active_req = req or synthetic_cell_refraction_apply_request()
    active_input = input_model or synthetic_cell_refracted_arrival_input_model()
    replacement = compute_weathering_replacement_statics_from_first_breaks(
        req=active_req,
        state=state,
        input_model=active_input,
    )
    return build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=active_req.datum,
        apply_options=active_req.apply,
        state=state,
        file_id=active_req.file_id,
        key1_byte=active_req.key1_byte,
        key2_byte=active_req.key2_byte,
    )


def expected_t1_s_for_node(node_id: int) -> float:
    return float(SYNTHETIC_NODE_T1_S[int(node_id)])


def expected_sh1_m_for_node(node_id: int) -> float:
    return float(SYNTHETIC_NODE_SH1_M[int(node_id)])


def expected_wcor_s_for_node(node_id: int) -> float:
    return float(SYNTHETIC_NODE_WCOR_S[int(node_id)])


def expected_cell_t1_s_for_node(node_id: int) -> float:
    return float(SYNTHETIC_CELL_NODE_T1_S[int(node_id)])


def expected_cell_sh1_m_for_node(node_id: int) -> float:
    return float(SYNTHETIC_CELL_NODE_SH1_M[int(node_id)])


def expected_cell_wcor_s_for_node(node_id: int) -> float:
    return float(SYNTHETIC_CELL_NODE_WCOR_S[int(node_id)])


def _synthetic_trace_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[int, int, float]] = []
    for raw_source in SYNTHETIC_SOURCE_NODE_ID.tolist():
        source = int(raw_source)
        for raw_receiver in SYNTHETIC_RECEIVER_NODE_ID.tolist():
            receiver = int(raw_receiver)
            if receiver == source:
                continue
            distance = abs(
                float(SYNTHETIC_NODE_X_M[receiver])
                - float(SYNTHETIC_NODE_X_M[source])
            )
            rows.append((source, receiver, distance))
    return (
        np.asarray([row[0] for row in rows], dtype=np.int64),
        np.asarray([row[1] for row in rows], dtype=np.int64),
        np.asarray([row[2] for row in rows], dtype=np.float64),
    )


def _synthetic_cell_trace_geometry(
    *,
    allowed_midpoint_cell_ids: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    allowed = (
        None
        if allowed_midpoint_cell_ids is None
        else {int(cell_id) for cell_id in allowed_midpoint_cell_ids}
    )
    rows: list[tuple[int, int, float, int, float]] = []
    for raw_source in SYNTHETIC_CELL_NODE_ID.tolist():
        source = int(raw_source)
        for raw_receiver in SYNTHETIC_CELL_NODE_ID.tolist():
            receiver = int(raw_receiver)
            if receiver == source:
                continue
            distance = abs(
                float(SYNTHETIC_CELL_NODE_X_M[receiver])
                - float(SYNTHETIC_CELL_NODE_X_M[source])
            )
            midpoint = 0.5 * (
                float(SYNTHETIC_CELL_NODE_X_M[source])
                + float(SYNTHETIC_CELL_NODE_X_M[receiver])
            )
            midpoint_cell = int(np.floor(midpoint / SYNTHETIC_CELL_SIZE_X_M))
            if allowed is not None and midpoint_cell not in allowed:
                continue
            pick = (
                float(SYNTHETIC_CELL_NODE_T1_S[source])
                + float(SYNTHETIC_CELL_NODE_T1_S[receiver])
                + distance / float(SYNTHETIC_CELL_V2_M_S[midpoint_cell])
            )
            rows.append((source, receiver, distance, midpoint_cell, pick))
    return (
        np.asarray([row[0] for row in rows], dtype=np.int64),
        np.asarray([row[1] for row in rows], dtype=np.int64),
        np.asarray([row[2] for row in rows], dtype=np.float64),
        np.asarray([row[3] for row in rows], dtype=np.int64),
        np.asarray([row[4] for row in rows], dtype=np.float64),
    )


def _synthetic_input_model(
    *,
    source: np.ndarray,
    receiver: np.ndarray,
    distance: np.ndarray,
    picks: np.ndarray,
) -> RefractionStaticInputModel:
    n_traces = int(picks.shape[0])
    zeros = np.zeros(n_traces, dtype=np.float64)
    endpoint_table = RefractionEndpointTable(
        node_id=np.ascontiguousarray(SYNTHETIC_NODE_ID, dtype=np.int64),
        endpoint_id=np.ascontiguousarray(SYNTHETIC_NODE_ID, dtype=np.int64),
        x_m=np.ascontiguousarray(SYNTHETIC_NODE_X_M, dtype=np.float64),
        y_m=np.zeros(SYNTHETIC_NODE_ID.shape[0], dtype=np.float64),
        elevation_m=np.ascontiguousarray(
            SYNTHETIC_NODE_SURFACE_ELEVATION_M,
            dtype=np.float64,
        ),
        kind=np.asarray(['both'] * 6 + ['receiver'] * 6, dtype='<U16'),
        pick_count=np.zeros(SYNTHETIC_NODE_ID.shape[0], dtype=np.int64),
    )
    return RefractionStaticInputModel(
        file_id='synthetic-refraction-line',
        n_traces=n_traces,
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        pick_time_s_sorted=np.ascontiguousarray(picks, dtype=np.float64),
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=np.ones(n_traces, dtype=bool),
        source_id_sorted=np.ascontiguousarray(100 + source, dtype=np.int64),
        receiver_id_sorted=np.ascontiguousarray(200 + receiver, dtype=np.int64),
        source_x_m_sorted=np.ascontiguousarray(SYNTHETIC_NODE_X_M[source], dtype=np.float64),
        source_y_m_sorted=zeros.copy(),
        receiver_x_m_sorted=np.ascontiguousarray(
            SYNTHETIC_NODE_X_M[receiver],
            dtype=np.float64,
        ),
        receiver_y_m_sorted=zeros.copy(),
        source_elevation_m_sorted=np.ascontiguousarray(
            SYNTHETIC_NODE_SURFACE_ELEVATION_M[source],
            dtype=np.float64,
        ),
        receiver_elevation_m_sorted=np.ascontiguousarray(
            SYNTHETIC_NODE_SURFACE_ELEVATION_M[receiver],
            dtype=np.float64,
        ),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=np.ascontiguousarray(distance, dtype=np.float64),
        offset_m_sorted=None,
        distance_m_sorted=np.ascontiguousarray(distance, dtype=np.float64),
        source_endpoint_key_sorted=np.asarray(
            [f'source:{int(value)}' for value in source],
            dtype='<U32',
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'receiver:{int(value)}' for value in receiver],
            dtype='<U32',
        ),
        source_node_id_sorted=np.ascontiguousarray(source, dtype=np.int64),
        receiver_node_id_sorted=np.ascontiguousarray(receiver, dtype=np.int64),
        node_x_m=np.ascontiguousarray(SYNTHETIC_NODE_X_M, dtype=np.float64),
        node_y_m=np.zeros(SYNTHETIC_NODE_ID.shape[0], dtype=np.float64),
        node_elevation_m=np.ascontiguousarray(
            SYNTHETIC_NODE_SURFACE_ELEVATION_M,
            dtype=np.float64,
        ),
        node_kind=np.asarray(['both'] * 6 + ['receiver'] * 6, dtype='<U16'),
        rejection_reason_sorted=np.full(n_traces, 'ok', dtype='<U32'),
        qc={'linkage_used': True},
        endpoint_table=endpoint_table,
        metadata={'synthetic_model': 'one_layer_refraction'},
    )


def _synthetic_cell_input_model(
    *,
    node_id: np.ndarray,
    node_x_m: np.ndarray,
    node_sh1_m: np.ndarray,
    source: np.ndarray,
    receiver: np.ndarray,
    distance: np.ndarray,
    picks: np.ndarray,
    midpoint_cell_id: np.ndarray,
) -> RefractionStaticInputModel:
    del midpoint_cell_id
    n_traces = int(picks.shape[0])
    zeros = np.zeros(n_traces, dtype=np.float64)
    node_elevation = 80.0 + np.asarray(node_sh1_m, dtype=np.float64)
    endpoint_table = RefractionEndpointTable(
        node_id=np.ascontiguousarray(node_id, dtype=np.int64),
        endpoint_id=np.ascontiguousarray(node_id, dtype=np.int64),
        x_m=np.ascontiguousarray(node_x_m, dtype=np.float64),
        y_m=np.zeros(node_id.shape[0], dtype=np.float64),
        elevation_m=np.ascontiguousarray(node_elevation, dtype=np.float64),
        kind=np.asarray(['both'] * int(node_id.shape[0]), dtype='<U16'),
        pick_count=np.zeros(node_id.shape[0], dtype=np.int64),
    )
    return RefractionStaticInputModel(
        file_id='synthetic-cell-refraction-line',
        n_traces=n_traces,
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        pick_time_s_sorted=np.ascontiguousarray(picks, dtype=np.float64),
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=np.ones(n_traces, dtype=bool),
        source_id_sorted=np.ascontiguousarray(100 + source, dtype=np.int64),
        receiver_id_sorted=np.ascontiguousarray(200 + receiver, dtype=np.int64),
        source_x_m_sorted=np.ascontiguousarray(node_x_m[source], dtype=np.float64),
        source_y_m_sorted=zeros.copy(),
        receiver_x_m_sorted=np.ascontiguousarray(
            node_x_m[receiver],
            dtype=np.float64,
        ),
        receiver_y_m_sorted=zeros.copy(),
        source_elevation_m_sorted=np.ascontiguousarray(
            node_elevation[source],
            dtype=np.float64,
        ),
        receiver_elevation_m_sorted=np.ascontiguousarray(
            node_elevation[receiver],
            dtype=np.float64,
        ),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=np.ascontiguousarray(distance, dtype=np.float64),
        offset_m_sorted=None,
        distance_m_sorted=np.ascontiguousarray(distance, dtype=np.float64),
        source_endpoint_key_sorted=np.asarray(
            [f'source:{int(value)}' for value in source],
            dtype='<U32',
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'receiver:{int(value)}' for value in receiver],
            dtype='<U32',
        ),
        source_node_id_sorted=np.ascontiguousarray(source, dtype=np.int64),
        receiver_node_id_sorted=np.ascontiguousarray(receiver, dtype=np.int64),
        node_x_m=np.ascontiguousarray(node_x_m, dtype=np.float64),
        node_y_m=np.zeros(node_id.shape[0], dtype=np.float64),
        node_elevation_m=np.ascontiguousarray(node_elevation, dtype=np.float64),
        node_kind=np.asarray(['both'] * int(node_id.shape[0]), dtype='<U16'),
        rejection_reason_sorted=np.full(n_traces, 'ok', dtype='<U32'),
        qc={'linkage_used': True},
        endpoint_table=endpoint_table,
        metadata={'synthetic_model': 'cell_refraction'},
    )


__all__ = [
    'SYNTHETIC_RECEIVER_NODE_ID',
    'SYNTHETIC_SH1_TOLERANCE_M',
    'SYNTHETIC_SOURCE_NODE_ID',
    'SYNTHETIC_T1_TOLERANCE_MS',
    'SYNTHETIC_CELL_NODE_ID',
    'SYNTHETIC_CELL_NODE_SH1_M',
    'SYNTHETIC_CELL_NODE_T1_S',
    'SYNTHETIC_CELL_NODE_WCOR_S',
    'SYNTHETIC_CELL_SIZE_X_M',
    'SYNTHETIC_CELL_V2_M_S',
    'SYNTHETIC_CELL_V2_TOLERANCE_M_S',
    'SYNTHETIC_V1_M_S',
    'SYNTHETIC_V1_TOLERANCE_M_S',
    'SYNTHETIC_V2_M_S',
    'SYNTHETIC_V2_TOLERANCE_M_S',
    'SYNTHETIC_WCOR_TOLERANCE_MS',
    'expected_cell_sh1_m_for_node',
    'expected_cell_t1_s_for_node',
    'expected_cell_wcor_s_for_node',
    'expected_sh1_m_for_node',
    'expected_t1_s_for_node',
    'expected_wcor_s_for_node',
    'run_synthetic_cell_refraction_statics',
    'run_synthetic_refraction_statics',
    'synthetic_cell_midpoint_cell_id_sorted',
    'synthetic_cell_refracted_arrival_input_model',
    'synthetic_cell_refraction_apply_request',
    'synthetic_direct_arrival_input_model',
    'synthetic_first_layer_request',
    'synthetic_refracted_arrival_input_model',
    'synthetic_refraction_apply_request',
]
