"""Reusable synthetic M6 refraction QC artifact fixtures."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

import app.statics.refraction.application.workflow as refraction_service_module
from app.statics.refraction.adapters.seisviewer2d import (
    workflow_runner as refraction_runner_module,
)
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
from app.core.state import AppState, create_app_state
from app.statics.refraction.artifacts import write_refraction_static_artifacts
from app.statics.refraction.artifacts import (
    write_refraction_line_profile_qc_artifacts,
)
from app.statics.refraction.application.datum import build_refraction_datum_statics
from app.statics.refraction.application.multilayer_service import (
    RefractionMultiLayerStaticsWorkflowResult,
    build_refraction_multilayer_weathering_replacement_statics,
)
from app.statics.refraction.domain.types import (
    RefractionDatumStaticsResult,
    RefractionEndpointTable,
    RefractionStaticArtifactSet,
    RefractionStaticInputModel,
    ResolvedRefractionFirstLayer,
)
from app.statics.refraction.application.weathering_replacement import (
    compute_weathering_replacement_statics_from_first_breaks,
)
from app.tests._refraction_multilayer_3layer_helpers import (
    make_three_layer_input_model,
    make_three_layer_solve_result,
)
from app.tests._refraction_multilayer_synthetic import (
    SYNTHETIC_MULTILAYER_V1_M_S,
    SYNTHETIC_MULTILAYER_V2_M_S,
    SYNTHETIC_MULTILAYER_V3_M_S,
    SYNTHETIC_MULTILAYER_VSUB_M_S,
    SyntheticMultiLayerRefractionDataset,
    make_2d_rotated_three_layer_refraction_dataset,
)
from app.tests._refraction_static_field_e2e_helpers import (
    FIELD_FILE_ID,
    FIELD_ORIGINAL_NAME,
    FIELD_SAMPLE_INTERVAL_S,
    clean_field_e2e_fixture,
    create_field_refraction_job,
    field_apply_request,
    install_field_job_stubs,
    write_field_manual_picks,
    write_field_trace_store,
)
from app.tests._refraction_static_synthetic import (
    run_synthetic_cell_refraction_statics,
    synthetic_cell_refracted_arrival_input_model,
    synthetic_cell_refraction_apply_request,
)
from app.tests.fixtures.refraction_synthetic import (
    SyntheticRefractionCellDataset,
    make_clean_3d_cell_refraction_dataset,
)

SIGN_CONVENTION = 'corrected(t) = raw(t - shift_s)'

LINE_ORIGIN_X_M = 1000.0
LINE_ORIGIN_Y_M = 2000.0
LINE_AZIMUTH_DEG = 37.0

TRUE_3D_CELL_V2_M_S = np.asarray(
    [[2200.0, 2400.0], [2600.0, 2800.0]],
    dtype=np.float64,
)
MIN_3D_CELL_OBSERVATIONS = 5

_OUTLIER_SHIFT_PATTERN_S = np.asarray([0.050, -0.060, 0.100], dtype=np.float64)
_OUTLIER_SEED = 432


@dataclass(frozen=True)
class M6WrittenRefractionCase:
    dataset: Any
    input_model: RefractionStaticInputModel
    req: RefractionStaticApplyRequest
    result: RefractionDatumStaticsResult
    paths: RefractionStaticArtifactSet
    outlier_trace_indices: np.ndarray


@dataclass(frozen=True)
class M6ThreeLayerLineCase:
    dataset: SyntheticMultiLayerRefractionDataset
    input_model: RefractionStaticInputModel
    req: RefractionStaticApplyRequest
    workflow: RefractionMultiLayerStaticsWorkflowResult | None
    line_profile_qc_source_csv: Path
    line_profile_qc_receiver_csv: Path
    line_profile_qc_combined_csv: Path
    line_profile_qc_npz: Path
    line_profile_qc_json: Path


@dataclass(frozen=True)
class M6FieldComponentCase:
    fixture: Any
    req: RefractionStaticApplyRequest
    state: AppState
    job_dir: Path


def write_line_projected_one_layer_case(
    job_dir: Path,
    *,
    robust_outliers: bool = False,
) -> M6WrittenRefractionCase:
    """Write a 2D projected one-layer cell-V2 job and all QC artifacts."""
    job_dir.mkdir(parents=True, exist_ok=True)
    base_input = synthetic_cell_refracted_arrival_input_model()
    outlier_trace_indices = np.asarray([], dtype=np.int64)
    if robust_outliers:
        base_input, outlier_trace_indices = _with_deterministic_bad_picks(base_input)

    input_model = map_inline_input_model_to_line_coordinates(
        base_input,
        line_origin_x_m=LINE_ORIGIN_X_M,
        line_origin_y_m=LINE_ORIGIN_Y_M,
        line_azimuth_deg=LINE_AZIMUTH_DEG,
    )
    req = _line_projected_one_layer_request(robust_enabled=robust_outliers)
    result = run_synthetic_cell_refraction_statics(
        req=req,
        input_model=input_model,
    )
    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=job_dir,
    )
    return M6WrittenRefractionCase(
        dataset=None,
        input_model=input_model,
        req=req,
        result=result,
        paths=paths,
        outlier_trace_indices=outlier_trace_indices,
    )


def write_line_projected_three_layer_case(
    job_dir: Path,
) -> M6ThreeLayerLineCase:
    """Write a projected 3-layer job with cell V2 and global V3/Vsub."""
    job_dir.mkdir(parents=True, exist_ok=True)
    dataset = make_2d_rotated_three_layer_refraction_dataset(
        line_origin_x_m=LINE_ORIGIN_X_M,
        line_origin_y_m=LINE_ORIGIN_Y_M,
        line_azimuth_deg=LINE_AZIMUTH_DEG,
    )
    input_model = make_three_layer_input_model(dataset)
    model = _line_projected_three_layer_model()
    solver = RefractionStaticSolverRequest(damping=0.0, robust={'enabled': False})
    datum = RefractionStaticDatumRequest(mode='none')
    apply_options = RefractionStaticApplyOptions(max_abs_shift_ms=500.0)
    resolved_first_layer = _resolved_multilayer_first_layer()
    solve_result = make_three_layer_solve_result(dataset)
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
        resolved_first_layer=resolved_first_layer,
    )
    workflow = RefractionMultiLayerStaticsWorkflowResult(
        solve_result=solve_result,
        components=None,
        weathering_replacement_result=weathering_replacement,
        datum_result=datum_result,
    )
    req = RefractionStaticApplyRequest(
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
    source_csv = job_dir / 'refraction_line_profile_qc_source.csv'
    receiver_csv = job_dir / 'refraction_line_profile_qc_receiver.csv'
    combined_csv = job_dir / 'refraction_line_profile_qc_combined.csv'
    npz_path = job_dir / 'refraction_line_profile_qc.npz'
    json_path = job_dir / 'refraction_line_profile_qc.json'
    write_refraction_line_profile_qc_artifacts(
        result=datum_result,
        req=req,
        source_csv_path=source_csv,
        receiver_csv_path=receiver_csv,
        combined_csv_path=combined_csv,
        npz_path=npz_path,
        json_path=json_path,
    )
    return M6ThreeLayerLineCase(
        dataset=dataset,
        input_model=input_model,
        req=req,
        workflow=workflow,
        line_profile_qc_source_csv=source_csv,
        line_profile_qc_receiver_csv=receiver_csv,
        line_profile_qc_combined_csv=combined_csv,
        line_profile_qc_npz=npz_path,
        line_profile_qc_json=json_path,
    )


def write_grid_3d_cell_case(job_dir: Path) -> M6WrittenRefractionCase:
    """Write a 3D grid cell-V2 one-layer job and all QC artifacts."""
    job_dir.mkdir(parents=True, exist_ok=True)
    dataset = make_clean_3d_cell_refraction_dataset(
        seed=433,
        cell_v2_m_s=TRUE_3D_CELL_V2_M_S,
        n_sources=6,
        n_receivers=6,
        cell_size_x_m=500.0,
        cell_size_y_m=500.0,
        noise_std_s=0.0,
        outlier_fraction=0.0,
    )
    req = _grid_3d_cell_request(dataset)
    input_model = input_model_from_cell_dataset(
        dataset,
        file_id=req.file_id,
        synthetic_model='m6_clean_3d_cell_refraction',
    )
    replacement = compute_weathering_replacement_statics_from_first_breaks(
        req=req,
        state=None,
        input_model=input_model,
    )
    result = build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=req.datum,
        apply_options=req.apply,
        state=None,
        file_id=req.file_id,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
    )
    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=job_dir,
    )
    return M6WrittenRefractionCase(
        dataset=dataset,
        input_model=input_model,
        req=req,
        result=result,
        paths=paths,
        outlier_trace_indices=np.asarray([], dtype=np.int64),
    )


def run_field_component_case(
    tmp_path: Path,
    monkeypatch: Any,
) -> M6FieldComponentCase:
    """Run the M4 field/manual correction path and write M6 component QC."""
    fixture = clean_field_e2e_fixture(manual_static_sign_convention='applied_shift_s')
    req = field_apply_request(fixture.dataset, apply_to_trace_shift=True)
    state = create_app_state()
    job_id = 'm6-field-component-qc'
    job_dir = tmp_path / 'jobs' / job_id
    create_field_refraction_job(state, job_id=job_id, req=req, job_dir=job_dir)
    install_field_job_stubs(monkeypatch, refraction_service_module, fixture)
    monkeypatch.setenv('SV_APP_DATA_DIR', str(tmp_path / 'app_data'))
    store = tmp_path / 'trace_stores' / f'{job_id}.sgy'
    write_field_trace_store(store, fixture.dataset)
    write_field_manual_picks(fixture.dataset)
    state.file_registry.update(
        FIELD_FILE_ID,
        path=f'/data/{FIELD_ORIGINAL_NAME}',
        store_path=str(store),
        dt=FIELD_SAMPLE_INTERVAL_S,
    )

    refraction_runner_module.run_refraction_static_apply_job(job_id, req, state)
    return M6FieldComponentCase(
        fixture=fixture,
        req=req,
        state=state,
        job_dir=job_dir,
    )


def map_inline_input_model_to_line_coordinates(
    input_model: RefractionStaticInputModel,
    *,
    line_origin_x_m: float,
    line_origin_y_m: float,
    line_azimuth_deg: float,
) -> RefractionStaticInputModel:
    """Convert an inline-only synthetic model to map coordinates on a line."""
    source_x, source_y = line_inline_to_map(
        input_model.source_x_m_sorted,
        line_origin_x_m=line_origin_x_m,
        line_origin_y_m=line_origin_y_m,
        line_azimuth_deg=line_azimuth_deg,
    )
    receiver_x, receiver_y = line_inline_to_map(
        input_model.receiver_x_m_sorted,
        line_origin_x_m=line_origin_x_m,
        line_origin_y_m=line_origin_y_m,
        line_azimuth_deg=line_azimuth_deg,
    )
    node_x, node_y = line_inline_to_map(
        input_model.node_x_m,
        line_origin_x_m=line_origin_x_m,
        line_origin_y_m=line_origin_y_m,
        line_azimuth_deg=line_azimuth_deg,
    )
    endpoint_x, endpoint_y = line_inline_to_map(
        input_model.endpoint_table.x_m,
        line_origin_x_m=line_origin_x_m,
        line_origin_y_m=line_origin_y_m,
        line_azimuth_deg=line_azimuth_deg,
    )
    endpoint_table = replace(
        input_model.endpoint_table,
        x_m=endpoint_x,
        y_m=endpoint_y,
    )
    return replace(
        input_model,
        source_x_m_sorted=source_x,
        source_y_m_sorted=source_y,
        receiver_x_m_sorted=receiver_x,
        receiver_y_m_sorted=receiver_y,
        node_x_m=node_x,
        node_y_m=node_y,
        endpoint_table=endpoint_table,
    )


def line_inline_to_map(
    inline_m: np.ndarray,
    *,
    line_origin_x_m: float,
    line_origin_y_m: float,
    line_azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    azimuth_rad = np.deg2rad(line_azimuth_deg)
    inline = np.asarray(inline_m, dtype=np.float64)
    return (
        np.ascontiguousarray(
            float(line_origin_x_m) + inline * np.sin(azimuth_rad),
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            float(line_origin_y_m) + inline * np.cos(azimuth_rad),
            dtype=np.float64,
        ),
    )


def projected_inline_crossline(
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    line_origin_x_m: float = LINE_ORIGIN_X_M,
    line_origin_y_m: float = LINE_ORIGIN_Y_M,
    line_azimuth_deg: float = LINE_AZIMUTH_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    azimuth_rad = np.deg2rad(line_azimuth_deg)
    inline_unit_x = float(np.sin(azimuth_rad))
    inline_unit_y = float(np.cos(azimuth_rad))
    dx = np.asarray(x_m, dtype=np.float64) - float(line_origin_x_m)
    dy = np.asarray(y_m, dtype=np.float64) - float(line_origin_y_m)
    inline = dx * inline_unit_x + dy * inline_unit_y
    crossline = dx * inline_unit_y - dy * inline_unit_x
    return (
        np.ascontiguousarray(inline, dtype=np.float64),
        np.ascontiguousarray(crossline, dtype=np.float64),
    )


def input_model_from_cell_dataset(
    dataset: SyntheticRefractionCellDataset,
    *,
    file_id: str,
    synthetic_model: str,
) -> RefractionStaticInputModel:
    n_traces = int(dataset.pick_time_s.shape[0])
    endpoint_table = _endpoint_table_from_cell_dataset(dataset)
    node_x_m = np.concatenate(
        (dataset.source_endpoint_x_m, dataset.receiver_endpoint_x_m)
    )
    node_y_m = np.concatenate(
        (dataset.source_endpoint_y_m, dataset.receiver_endpoint_y_m)
    )
    node_elevation_m = np.zeros(endpoint_table.node_id.shape, dtype=np.float64)
    node_kind = np.concatenate(
        (
            np.full(dataset.source_endpoint_id.shape, 'source', dtype='<U16'),
            np.full(dataset.receiver_endpoint_id.shape, 'receiver', dtype='<U16'),
        )
    )
    return RefractionStaticInputModel(
        file_id=file_id,
        n_traces=n_traces,
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        pick_time_s_sorted=dataset.pick_time_s,
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=dataset.valid_mask,
        source_id_sorted=dataset.source_id,
        receiver_id_sorted=dataset.receiver_id,
        source_x_m_sorted=dataset.source_x_m,
        source_y_m_sorted=dataset.source_y_m,
        receiver_x_m_sorted=dataset.receiver_x_m,
        receiver_y_m_sorted=dataset.receiver_y_m,
        source_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=dataset.offset_m,
        offset_m_sorted=None,
        distance_m_sorted=dataset.offset_m,
        source_endpoint_key_sorted=np.asarray(
            [f'source:{int(value)}' for value in dataset.source_id],
            dtype='<U32',
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'receiver:{int(value)}' for value in dataset.receiver_id],
            dtype='<U32',
        ),
        source_node_id_sorted=dataset.source_node_id,
        receiver_node_id_sorted=dataset.receiver_node_id,
        node_x_m=np.ascontiguousarray(node_x_m, dtype=np.float64),
        node_y_m=np.ascontiguousarray(node_y_m, dtype=np.float64),
        node_elevation_m=node_elevation_m,
        node_kind=node_kind,
        rejection_reason_sorted=np.full(n_traces, 'ok', dtype='<U32'),
        qc={},
        endpoint_table=endpoint_table,
        metadata={'synthetic_model': synthetic_model},
    )


def _line_projected_one_layer_request(
    *,
    robust_enabled: bool,
) -> RefractionStaticApplyRequest:
    req = synthetic_cell_refraction_apply_request(robust_enabled=robust_enabled)
    payload = req.model_dump(mode='json')
    payload['file_id'] = 'm6-line-projected-one-layer'
    payload['model']['refractor_cell'].update(
        {
            'coordinate_mode': 'line_2d_projected',
            'line_origin_x_m': LINE_ORIGIN_X_M,
            'line_origin_y_m': LINE_ORIGIN_Y_M,
            'line_azimuth_deg': LINE_AZIMUTH_DEG,
        }
    )
    payload['reduced_time_qc'] = {'reduction_velocity_mode': 'layer_velocity'}
    return RefractionStaticApplyRequest.model_validate(payload)


def _grid_3d_cell_request(
    dataset: SyntheticRefractionCellDataset,
) -> RefractionStaticApplyRequest:
    return RefractionStaticApplyRequest.model_validate(
        {
            'file_id': 'm6-clean-3d-cell-synthetic',
            'key1_byte': 189,
            'key2_byte': 193,
            'pick_source': {
                'kind': 'batch_predicted_npz',
                'job_id': 'm6-clean-3d-cell-first-breaks',
                'artifact_name': 'predicted_picks_time_s.npz',
            },
            'linkage': {'mode': 'none'},
            'model': {
                'method': 'gli_variable_thickness',
                'weathering_velocity_m_s': dataset.true_v1_m_s,
                'bedrock_velocity_mode': 'solve_cell',
                'bedrock_velocity_m_s': None,
                'initial_bedrock_velocity_m_s': 2500.0,
                'min_bedrock_velocity_m_s': 1200.0,
                'max_bedrock_velocity_m_s': 6000.0,
                'max_weathering_thickness_m': None,
                'refractor_cell': {
                    'number_of_cell_x': int(dataset.true_cell_v2_m_s.shape[1]),
                    'size_of_cell_x_m': dataset.cell_size_x_m,
                    'x_coordinate_origin_m': dataset.x_coordinate_origin_m,
                    'number_of_cell_y': int(dataset.true_cell_v2_m_s.shape[0]),
                    'size_of_cell_y_m': dataset.cell_size_y_m,
                    'y_coordinate_origin_m': dataset.y_coordinate_origin_m,
                    'assignment_mode': 'midpoint',
                    'outside_grid_policy': 'reject',
                    'coordinate_mode': 'grid_3d',
                    'min_observations_per_cell': MIN_3D_CELL_OBSERVATIONS,
                    'velocity_smoothing_weight': 0.0,
                    'smoothing_reference_distance_m': None,
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
                'max_abs_half_intercept_time_ms': 500.0,
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
            'conversion': {'mode': 't1lsst_1layer'},
            'reduced_time_qc': {'reduction_velocity_mode': 'layer_velocity'},
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


def _line_projected_three_layer_model() -> RefractionStaticModelRequest:
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
                    'velocity_mode': 'solve_cell',
                    'initial_velocity_m_s': SYNTHETIC_MULTILAYER_V2_M_S,
                    'min_velocity_m_s': 1600.0,
                    'max_velocity_m_s': 3200.0,
                    'min_observations_per_cell': 1,
                    'smoothing_weight': 0.0,
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
            'refractor_cell': {
                'number_of_cell_x': 3,
                'size_of_cell_x_m': 500.0,
                'x_coordinate_origin_m': 0.0,
                'number_of_cell_y': 1,
                'size_of_cell_y_m': None,
                'y_coordinate_origin_m': 0.0,
                'assignment_mode': 'midpoint',
                'outside_grid_policy': 'reject',
                'coordinate_mode': 'line_2d_projected',
                'line_origin_x_m': LINE_ORIGIN_X_M,
                'line_origin_y_m': LINE_ORIGIN_Y_M,
                'line_azimuth_deg': LINE_AZIMUTH_DEG,
                'min_observations_per_cell': 1,
                'velocity_smoothing_weight': 0.0,
                'smoothing_reference_distance_m': None,
            },
        }
    )


def _resolved_multilayer_first_layer() -> ResolvedRefractionFirstLayer:
    return ResolvedRefractionFirstLayer(
        mode='constant',
        weathering_velocity_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        status='constant',
        qc={'weathering_velocity_m_s': SYNTHETIC_MULTILAYER_V1_M_S},
    )


def _with_deterministic_bad_picks(
    input_model: RefractionStaticInputModel,
) -> tuple[RefractionStaticInputModel, np.ndarray]:
    eligible = np.flatnonzero(input_model.pick_time_s_sorted > 0.061)
    rng = np.random.default_rng(_OUTLIER_SEED)
    outlier_indices = np.sort(rng.choice(eligible, size=6, replace=False))
    pick_time_s = input_model.pick_time_s_sorted.copy()
    shifts = np.resize(_OUTLIER_SHIFT_PATTERN_S, int(outlier_indices.shape[0]))
    pick_time_s[outlier_indices] += shifts
    return (
        replace(
            input_model,
            pick_time_s_sorted=np.ascontiguousarray(pick_time_s, dtype=np.float64),
        ),
        np.ascontiguousarray(outlier_indices, dtype=np.int64),
    )


def _endpoint_table_from_cell_dataset(
    dataset: SyntheticRefractionCellDataset,
) -> RefractionEndpointTable:
    node_id = np.concatenate(
        (dataset.source_endpoint_node_id, dataset.receiver_endpoint_node_id)
    )
    endpoint_id = np.concatenate(
        (dataset.source_endpoint_id, dataset.receiver_endpoint_id)
    )
    endpoint_x_m = np.concatenate(
        (dataset.source_endpoint_x_m, dataset.receiver_endpoint_x_m)
    )
    endpoint_y_m = np.concatenate(
        (dataset.source_endpoint_y_m, dataset.receiver_endpoint_y_m)
    )
    kind = np.concatenate(
        (
            np.full(dataset.source_endpoint_id.shape, 'source', dtype='<U16'),
            np.full(dataset.receiver_endpoint_id.shape, 'receiver', dtype='<U16'),
        )
    )
    return RefractionEndpointTable(
        node_id=np.ascontiguousarray(node_id, dtype=np.int64),
        endpoint_id=np.ascontiguousarray(endpoint_id, dtype=np.int64),
        x_m=np.ascontiguousarray(endpoint_x_m, dtype=np.float64),
        y_m=np.ascontiguousarray(endpoint_y_m, dtype=np.float64),
        elevation_m=np.zeros(node_id.shape, dtype=np.float64),
        kind=kind,
        pick_count=np.zeros(node_id.shape, dtype=np.int64),
    )


__all__ = [
    'LINE_AZIMUTH_DEG',
    'LINE_ORIGIN_X_M',
    'LINE_ORIGIN_Y_M',
    'MIN_3D_CELL_OBSERVATIONS',
    'M6FieldComponentCase',
    'M6ThreeLayerLineCase',
    'M6WrittenRefractionCase',
    'SIGN_CONVENTION',
    'TRUE_3D_CELL_V2_M_S',
    'input_model_from_cell_dataset',
    'line_inline_to_map',
    'map_inline_input_model_to_line_coordinates',
    'projected_inline_crossline',
    'run_field_component_case',
    'write_grid_3d_cell_case',
    'write_line_projected_one_layer_case',
    'write_line_projected_three_layer_case',
]
