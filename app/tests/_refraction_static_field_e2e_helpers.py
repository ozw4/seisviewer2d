from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.core.state import AppState
from app.services.refraction_static_source_depth import resolve_refraction_source_depth
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
    RefractionEndpointTable,
    RefractionStaticInputModel,
)
from app.services.refraction_static_uphole import resolve_refraction_uphole
from app.tests._refraction_static_field_synthetic import (
    SyntheticRefractionFieldCorrectionDataset,
    make_clean_2d_field_corrections,
    make_messy_2d_field_corrections,
)
from app.utils.pick_cache_file1d_mem import ensure_root_dir, path_for_file

FIELD_FILE_ID = 'field-correction-synthetic-file'
FIELD_KEY1_BYTE = 189
FIELD_KEY2_BYTE = 193
FIELD_SAMPLE_INTERVAL_S = 0.004
FIELD_ORIGINAL_NAME = 'field-correction.sgy'
FIELD_SOURCE_ID_BYTE = 9
FIELD_RECEIVER_ID_BYTE = 13
FIELD_SOURCE_X_BYTE = 73
FIELD_SOURCE_Y_BYTE = 77
FIELD_RECEIVER_X_BYTE = 81
FIELD_RECEIVER_Y_BYTE = 85
FIELD_SOURCE_ELEVATION_BYTE = 45
FIELD_RECEIVER_ELEVATION_BYTE = 41
FIELD_COORDINATE_SCALAR_BYTE = 71
FIELD_ELEVATION_SCALAR_BYTE = 69
FIELD_SOURCE_DEPTH_BYTE = 115
FIELD_UPHOLE_TIME_BYTE = 95


@dataclass(frozen=True)
class RefractionFieldE2EFixture:
    dataset: SyntheticRefractionFieldCorrectionDataset
    input_model: RefractionStaticInputModel
    base_result: RefractionDatumStaticsResult


def clean_field_e2e_fixture(
    *,
    manual_static_sign_convention: str = 'delay_positive_ms',
) -> RefractionFieldE2EFixture:
    dataset = make_clean_2d_field_corrections(
        manual_static_sign_convention=manual_static_sign_convention,
    )
    return _field_e2e_fixture(dataset)


def messy_field_e2e_fixture(
    *,
    manual_static_sign_convention: str = 'delay_positive_ms',
) -> RefractionFieldE2EFixture:
    dataset = make_messy_2d_field_corrections(
        manual_static_sign_convention=manual_static_sign_convention,
    )
    return _field_e2e_fixture(dataset)


def field_apply_request(
    dataset: SyntheticRefractionFieldCorrectionDataset,
    *,
    apply_to_trace_shift: bool,
    register_corrected_file: bool = False,
    invalid_component_policy: str = 'fail',
    manual_static_endpoint_kind: str = 'both',
) -> RefractionStaticApplyRequest:
    manual_static = {
        'mode': 'inline_table',
        'sign_convention': dataset.manual_static_sign_convention,
        'allow_missing_endpoints': True,
    }
    if manual_static_endpoint_kind in {'both', 'source'}:
        manual_static['source_inline_table'] = [
            {
                'endpoint_id': int(endpoint_id),
                'value': _manual_static_request_value(
                    float(value_s),
                    sign_convention=dataset.manual_static_sign_convention,
                ),
            }
            for endpoint_id, value_s in zip(
                range(int(dataset.source_endpoint_table.endpoint_id.shape[0])),
                dataset.source_manual_static_input_s.tolist(),
                strict=True,
            )
        ]
    if manual_static_endpoint_kind in {'both', 'receiver'}:
        manual_static['receiver_inline_table'] = [
            {
                'endpoint_id': int(endpoint_id),
                'value': _manual_static_request_value(
                    float(value_s),
                    sign_convention=dataset.manual_static_sign_convention,
                ),
            }
            for endpoint_id, value_s in zip(
                range(int(dataset.receiver_endpoint_table.endpoint_id.shape[0])),
                dataset.receiver_manual_static_input_s.tolist(),
                strict=True,
            )
        ]
    return RefractionStaticApplyRequest.model_validate(
        {
            'file_id': FIELD_FILE_ID,
            'key1_byte': FIELD_KEY1_BYTE,
            'key2_byte': FIELD_KEY2_BYTE,
            'pick_source': {'kind': 'manual_memmap'},
            'geometry': {
                'source_id_byte': FIELD_SOURCE_ID_BYTE,
                'receiver_id_byte': FIELD_RECEIVER_ID_BYTE,
            },
            'linkage': {'mode': 'none'},
            'model': {
                'method': 'multilayer_time_term',
                'first_layer': {
                    'mode': 'constant',
                    'weathering_velocity_m_s': dataset.true_v1_m_s,
                },
                'layers': [
                    {
                        'kind': 'v2_t1',
                        'enabled': True,
                        'min_offset_m': 300.0,
                        'max_offset_m': 1000.0,
                        'velocity_mode': 'solve_global',
                        'initial_velocity_m_s': dataset.true_v2_m_s,
                        'min_velocity_m_s': 1200.0,
                        'max_velocity_m_s': 5000.0,
                    },
                    {
                        'kind': 'v3_t2',
                        'enabled': True,
                        'min_offset_m': 1000.0,
                        'max_offset_m': 2000.0,
                        'velocity_mode': 'solve_global',
                        'initial_velocity_m_s': dataset.true_v3_m_s,
                        'min_velocity_m_s': 2500.0,
                        'max_velocity_m_s': 6000.0,
                    },
                ],
            },
            'solver': {
                'damping': 0.0,
                'min_picks_per_node': 1,
                'robust': {'enabled': False},
            },
            'datum': {'mode': 'none'},
            'conversion': {
                'mode': 't1lsst_multilayer',
                'layer_count': 2,
            },
            'field_corrections': {
                'source_depth': {
                    'mode': 'weathering_velocity_time',
                    'source_depth_byte': FIELD_SOURCE_DEPTH_BYTE,
                    'source_depth_unit': 'm',
                },
                'uphole': {
                    'mode': 'header_time',
                    'uphole_time_byte': FIELD_UPHOLE_TIME_BYTE,
                    'uphole_time_unit': 'ms',
                    'positive_time_means_delay': True,
                },
                'manual_static': manual_static,
                'composition': {
                    'enabled': True,
                    'apply_to_trace_shift': apply_to_trace_shift,
                    'invalid_component_policy': invalid_component_policy,
                },
            },
            'apply': {
                'mode': 'refraction_from_raw',
                'interpolation': 'linear',
                'fill_value': 0.0,
                'max_abs_shift_ms': 250.0,
                'output_dtype': 'float32',
                'register_corrected_file': register_corrected_file,
            },
        }
    )


def install_field_job_stubs(
    monkeypatch: Any,
    service_module: Any,
    fixture: RefractionFieldE2EFixture,
) -> None:
    def _compute_from_input_model(**kwargs: Any) -> SimpleNamespace:
        input_model = kwargs['input_model']
        return SimpleNamespace(
            datum_result=_base_result_with_input_endpoint_keys(
                fixture.base_result,
                input_model,
            )
        )

    monkeypatch.setattr(
        service_module,
        'compute_refraction_multilayer_datum_statics_from_input_model',
        _compute_from_input_model,
    )


def create_field_refraction_job(
    state: AppState,
    *,
    job_id: str,
    req: RefractionStaticApplyRequest,
    job_dir: Path,
) -> None:
    with state.lock:
        state.jobs.create_static_job(
            job_id,
            file_id=req.file_id,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            statics_kind='refraction',
            artifacts_dir=str(job_dir),
        )


def write_field_trace_store(
    root: Path,
    dataset: SyntheticRefractionFieldCorrectionDataset,
    *,
    sample_interval_s: float = FIELD_SAMPLE_INTERVAL_S,
) -> np.ndarray:
    root.mkdir(parents=True, exist_ok=True)
    n_traces = int(dataset.sorted_trace_index.shape[0])
    n_samples = 192
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    traces[:, n_samples // 2] = 1.0
    np.save(root / 'traces.npy', traces)
    np.savez(
        root / 'index.npz',
        key1_values=np.asarray([100], dtype=np.int64),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([n_traces], dtype=np.int64),
        sorted_to_original=dataset.sorted_trace_index.astype(np.int64, copy=False),
    )
    np.save(
        root / f'headers_byte_{FIELD_KEY1_BYTE}.npy',
        np.full(n_traces, 100, dtype=np.int32),
    )
    np.save(
        root / f'headers_byte_{FIELD_KEY2_BYTE}.npy',
        np.arange(n_traces, dtype=np.int32),
    )
    (root / 'meta.json').write_text(
        json.dumps(
            {
                'schema_version': 1,
                'dtype': 'float32',
                'n_traces': n_traces,
                'n_samples': n_samples,
                'key_bytes': {'key1': FIELD_KEY1_BYTE, 'key2': FIELD_KEY2_BYTE},
                'sorted_by': ['key1', 'key2'],
                'dt': sample_interval_s,
                'original_segy_path': '/data/field-correction.sgy',
                'source_sha256': None,
                'original_name': 'field-correction.sgy',
            }
        ),
        encoding='utf-8',
    )
    _write_field_trace_store_headers(root, dataset)
    return traces


def write_field_manual_picks(
    dataset: SyntheticRefractionFieldCorrectionDataset,
) -> Path:
    ensure_root_dir()
    path = path_for_file(FIELD_ORIGINAL_NAME)
    np.save(path, dataset.first_break_time_s.astype(np.float32, copy=False))
    return path


def _write_field_trace_store_headers(
    root: Path,
    dataset: SyntheticRefractionFieldCorrectionDataset,
) -> None:
    base = dataset.base_dataset
    source_index = dataset.source_endpoint_index
    _write_header(root, FIELD_KEY1_BYTE, np.full(dataset.sorted_trace_index.shape, 100))
    _write_header(root, FIELD_KEY2_BYTE, np.arange(dataset.sorted_trace_index.shape[0]))
    _write_header(root, FIELD_SOURCE_ID_BYTE, dataset.source_endpoint_id_sorted)
    _write_header(root, FIELD_RECEIVER_ID_BYTE, dataset.receiver_endpoint_id_sorted)
    _write_header(root, FIELD_SOURCE_X_BYTE, np.rint(base.source_x_m).astype(np.int32))
    _write_header(root, FIELD_SOURCE_Y_BYTE, np.rint(base.source_y_m).astype(np.int32))
    _write_header(
        root,
        FIELD_RECEIVER_X_BYTE,
        np.rint(base.receiver_x_m).astype(np.int32),
    )
    _write_header(
        root,
        FIELD_RECEIVER_Y_BYTE,
        np.rint(base.receiver_y_m).astype(np.int32),
    )
    _write_header(root, FIELD_SOURCE_ELEVATION_BYTE, _centimeters(base.source_elevation_m))
    _write_header(
        root,
        FIELD_RECEIVER_ELEVATION_BYTE,
        _centimeters(base.receiver_elevation_m),
    )
    _write_header(
        root,
        FIELD_COORDINATE_SCALAR_BYTE,
        np.zeros(dataset.sorted_trace_index.shape, dtype=np.int32),
    )
    _write_header(
        root,
        FIELD_ELEVATION_SCALAR_BYTE,
        np.full(dataset.sorted_trace_index.shape, -100, dtype=np.int32),
    )
    _write_header(root, FIELD_SOURCE_DEPTH_BYTE, _centimeters(dataset.source_depth_m[source_index]))
    _write_header(
        root,
        FIELD_UPHOLE_TIME_BYTE,
        _integer_milliseconds(dataset.uphole_time_s[source_index]),
    )


def _write_header(root: Path, byte: int, values: np.ndarray) -> None:
    np.save(root / f'headers_byte_{byte}.npy', np.asarray(values))


def _centimeters(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(arr)):
        return arr * 100.0
    return np.rint(arr * 100.0).astype(np.int32)


def _integer_milliseconds(values_s: np.ndarray) -> np.ndarray:
    values_ms = np.asarray(values_s, dtype=np.float64) * 1000.0
    if np.any(~np.isfinite(values_ms)):
        return values_ms
    return np.rint(values_ms).astype(np.int32)


def _field_e2e_fixture(
    dataset: SyntheticRefractionFieldCorrectionDataset,
) -> RefractionFieldE2EFixture:
    input_model = _field_input_model(dataset)
    return RefractionFieldE2EFixture(
        dataset=dataset,
        input_model=input_model,
        base_result=_base_refraction_result(dataset),
    )


def _field_input_model(
    dataset: SyntheticRefractionFieldCorrectionDataset,
) -> RefractionStaticInputModel:
    base = dataset.base_dataset
    node = _combined_node_arrays(dataset)
    endpoint_table = RefractionEndpointTable(
        node_id=node['node_id'],
        endpoint_id=node['node_id'].copy(),
        x_m=node['x_m'],
        y_m=node['y_m'],
        elevation_m=node['elevation_m'],
        kind=node['kind'],
        pick_count=node['pick_count'],
    )
    source_depth_sorted = dataset.source_depth_m[dataset.source_endpoint_index]
    uphole_sorted = dataset.uphole_time_s[dataset.source_endpoint_index]
    source_depth = resolve_refraction_source_depth(
        source_endpoint_key_sorted=dataset.source_endpoint_key_sorted,
        source_endpoint_id_sorted=dataset.source_endpoint_id_sorted,
        source_node_id_sorted=dataset.source_node_id_sorted,
        source_depth_m_sorted=source_depth_sorted,
        mode='weathering_velocity_time',
        source_depth_byte=115,
    )
    uphole = resolve_refraction_uphole(
        source_endpoint_key_sorted=dataset.source_endpoint_key_sorted,
        source_endpoint_id_sorted=dataset.source_endpoint_id_sorted,
        source_node_id_sorted=dataset.source_node_id_sorted,
        uphole_time_sorted=uphole_sorted,
        mode='header_time',
        uphole_time_byte=95,
        uphole_time_unit='s',
        positive_time_means_delay=True,
    )
    return RefractionStaticInputModel(
        file_id=FIELD_FILE_ID,
        n_traces=int(dataset.sorted_trace_index.shape[0]),
        sorted_trace_index=dataset.sorted_trace_index.astype(np.int64, copy=False),
        pick_time_s_sorted=dataset.first_break_time_s,
        valid_pick_mask_sorted=dataset.valid_pick_mask,
        valid_observation_mask_sorted=dataset.valid_pick_mask,
        source_id_sorted=dataset.source_endpoint_id_sorted,
        receiver_id_sorted=dataset.receiver_endpoint_id_sorted,
        source_x_m_sorted=base.source_x_m,
        source_y_m_sorted=base.source_y_m,
        receiver_x_m_sorted=base.receiver_x_m,
        receiver_y_m_sorted=base.receiver_y_m,
        source_elevation_m_sorted=base.source_elevation_m,
        receiver_elevation_m_sorted=base.receiver_elevation_m,
        source_depth_m_sorted=source_depth_sorted,
        geometry_distance_m_sorted=dataset.offset_m,
        offset_m_sorted=dataset.offset_m,
        distance_m_sorted=dataset.offset_m,
        source_endpoint_key_sorted=dataset.source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=dataset.receiver_endpoint_key_sorted,
        source_node_id_sorted=dataset.source_node_id_sorted,
        receiver_node_id_sorted=dataset.receiver_node_id_sorted,
        node_x_m=node['x_m'],
        node_y_m=node['y_m'],
        node_elevation_m=node['elevation_m'],
        node_kind=node['kind'],
        rejection_reason_sorted=np.full(
            dataset.sorted_trace_index.shape,
            'ok',
            dtype='<U32',
        ),
        qc={'fixture': dataset.name},
        endpoint_table=endpoint_table,
        metadata={'fixture': dataset.name},
        source_depth_result=source_depth,
        uphole_result=uphole,
        source_endpoint_id_sorted=dataset.source_endpoint_id_sorted,
        receiver_endpoint_id_sorted=dataset.receiver_endpoint_id_sorted,
    )


def _base_result_with_input_endpoint_keys(
    result: RefractionDatumStaticsResult,
    input_model: RefractionStaticInputModel,
) -> RefractionDatumStaticsResult:
    source_keys = _unique_nonempty_in_order(input_model.source_endpoint_key_sorted)
    receiver_keys = _unique_nonempty_in_order(input_model.receiver_endpoint_key_sorted)
    if source_keys.shape != result.source_endpoint_key.shape:
        raise AssertionError('source endpoint count changed in field E2E fixture')
    if receiver_keys.shape != result.receiver_endpoint_key.shape:
        raise AssertionError('receiver endpoint count changed in field E2E fixture')
    return replace(
        result,
        source_endpoint_key=source_keys,
        receiver_endpoint_key=receiver_keys,
        source_endpoint_key_sorted=np.asarray(
            input_model.source_endpoint_key_sorted,
            dtype=object,
        ),
        receiver_endpoint_key_sorted=np.asarray(
            input_model.receiver_endpoint_key_sorted,
            dtype=object,
        ),
        row_source_endpoint_key=np.asarray(
            input_model.source_endpoint_key_sorted,
            dtype=object,
        ),
        row_receiver_endpoint_key=np.asarray(
            input_model.receiver_endpoint_key_sorted,
            dtype=object,
        ),
    )


def _unique_nonempty_in_order(values: np.ndarray) -> np.ndarray:
    seen: set[str] = set()
    out: list[str] = []
    for raw_value in np.asarray(values).tolist():
        value = str(raw_value)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return np.asarray(out, dtype=object)


def _base_refraction_result(
    dataset: SyntheticRefractionFieldCorrectionDataset,
) -> RefractionDatumStaticsResult:
    base = dataset.base_dataset
    source = dataset.source_endpoint_table
    receiver = dataset.receiver_endpoint_table
    node = _combined_node_arrays(dataset)
    source_index = dataset.source_endpoint_index
    receiver_index = dataset.receiver_endpoint_index
    n_traces = int(dataset.sorted_trace_index.shape[0])
    ok_trace_status = np.full(n_traces, 'ok', dtype='<U48')
    zero_trace = np.zeros(n_traces, dtype=np.float64)
    source_shift = source.refraction_shift_s
    receiver_shift = receiver.refraction_shift_s
    source_shift_sorted = source_shift[source_index]
    receiver_shift_sorted = receiver_shift[receiver_index]
    return RefractionDatumStaticsResult(
        bedrock_velocity_mode='solve_global',
        bedrock_slowness_s_per_m=1.0 / dataset.true_v2_m_s,
        bedrock_velocity_m_s=dataset.true_v2_m_s,
        weathering_velocity_m_s=dataset.true_v1_m_s,
        replacement_slowness_delta_s_per_m=(1.0 / dataset.true_v3_m_s)
        - (1.0 / dataset.true_v1_m_s),
        datum_mode='none',
        floating_datum_mode='none',
        flat_datum_elevation_m=None,
        node_id=node['node_id'],
        node_x_m=node['x_m'],
        node_y_m=node['y_m'],
        node_surface_elevation_m=node['elevation_m'],
        node_kind=node['kind'],
        node_weathering_thickness_m=node['total_thickness_m'],
        node_refractor_elevation_m=node['refractor_elevation_m'],
        node_half_intercept_time_s=node['t1_s'],
        node_weathering_replacement_shift_s=node['wcor_s'],
        node_floating_datum_elevation_m=node['elevation_m'],
        node_solution_status=node['status'],
        node_datum_status=node['status'],
        node_weathering_status=node['status'],
        node_pick_count=node['pick_count'],
        node_used_pick_count=node['pick_count'],
        node_rejected_pick_count=np.zeros(node['node_id'].shape, dtype=np.int64),
        node_residual_rms_s=np.zeros(node['node_id'].shape, dtype=np.float64),
        node_residual_mad_s=np.zeros(node['node_id'].shape, dtype=np.float64),
        source_endpoint_key=source.endpoint_key,
        source_id=source.endpoint_id,
        source_node_id=source.node_id,
        source_x_m=source.x_m,
        source_y_m=source.y_m,
        source_surface_elevation_m=source.elevation_m,
        source_half_intercept_time_s=base.true_source_endpoint_t1_s,
        source_weathering_thickness_m=(
            base.true_source_endpoint_sh1_m + base.true_source_endpoint_sh2_m
        ),
        source_refractor_elevation_m=(
            source.elevation_m
            - base.true_source_endpoint_sh1_m
            - base.true_source_endpoint_sh2_m
        ),
        source_floating_datum_elevation_m=source.elevation_m,
        source_weathering_replacement_shift_s=source_shift,
        source_floating_datum_elevation_shift_s=np.zeros(
            source.endpoint_key.shape,
            dtype=np.float64,
        ),
        source_flat_datum_shift_s=np.zeros(source.endpoint_key.shape, dtype=np.float64),
        source_refraction_shift_s=source_shift,
        source_datum_status=np.full(source.endpoint_key.shape, 'ok', dtype='<U48'),
        receiver_endpoint_key=receiver.endpoint_key,
        receiver_id=receiver.endpoint_id,
        receiver_node_id=receiver.node_id,
        receiver_x_m=receiver.x_m,
        receiver_y_m=receiver.y_m,
        receiver_surface_elevation_m=receiver.elevation_m,
        receiver_half_intercept_time_s=base.true_receiver_endpoint_t1_s,
        receiver_weathering_thickness_m=(
            base.true_receiver_endpoint_sh1_m + base.true_receiver_endpoint_sh2_m
        ),
        receiver_refractor_elevation_m=(
            receiver.elevation_m
            - base.true_receiver_endpoint_sh1_m
            - base.true_receiver_endpoint_sh2_m
        ),
        receiver_floating_datum_elevation_m=receiver.elevation_m,
        receiver_weathering_replacement_shift_s=receiver_shift,
        receiver_floating_datum_elevation_shift_s=np.zeros(
            receiver.endpoint_key.shape,
            dtype=np.float64,
        ),
        receiver_flat_datum_shift_s=np.zeros(
            receiver.endpoint_key.shape,
            dtype=np.float64,
        ),
        receiver_refraction_shift_s=receiver_shift,
        receiver_datum_status=np.full(receiver.endpoint_key.shape, 'ok', dtype='<U48'),
        sorted_trace_index=dataset.sorted_trace_index,
        valid_observation_mask_sorted=dataset.valid_pick_mask,
        used_observation_mask_sorted=dataset.valid_pick_mask,
        source_node_id_sorted=dataset.source_node_id_sorted,
        receiver_node_id_sorted=dataset.receiver_node_id_sorted,
        source_surface_elevation_m_sorted=base.source_elevation_m,
        receiver_surface_elevation_m_sorted=base.receiver_elevation_m,
        source_floating_datum_elevation_m_sorted=base.source_elevation_m,
        receiver_floating_datum_elevation_m_sorted=base.receiver_elevation_m,
        source_weathering_thickness_m_sorted=(
            base.true_source_sh1_m + base.true_source_sh2_m
        ),
        receiver_weathering_thickness_m_sorted=(
            base.true_receiver_sh1_m + base.true_receiver_sh2_m
        ),
        source_refractor_elevation_m_sorted=(
            base.source_elevation_m - base.true_source_sh1_m - base.true_source_sh2_m
        ),
        receiver_refractor_elevation_m_sorted=(
            base.receiver_elevation_m
            - base.true_receiver_sh1_m
            - base.true_receiver_sh2_m
        ),
        source_half_intercept_time_s_sorted=base.true_source_t1_s,
        receiver_half_intercept_time_s_sorted=base.true_receiver_t1_s,
        source_weathering_replacement_shift_s_sorted=source_shift_sorted,
        receiver_weathering_replacement_shift_s_sorted=receiver_shift_sorted,
        source_floating_datum_elevation_shift_s_sorted=zero_trace.copy(),
        receiver_floating_datum_elevation_shift_s_sorted=zero_trace.copy(),
        source_flat_datum_shift_s_sorted=zero_trace.copy(),
        receiver_flat_datum_shift_s_sorted=zero_trace.copy(),
        source_refraction_shift_s_sorted=source_shift_sorted,
        receiver_refraction_shift_s_sorted=receiver_shift_sorted,
        source_endpoint_key_sorted=dataset.source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=dataset.receiver_endpoint_key_sorted,
        weathering_replacement_trace_shift_s_sorted=dataset.expected_refraction_trace_shift_s,
        floating_datum_elevation_shift_s_sorted=zero_trace.copy(),
        flat_datum_shift_s_sorted=zero_trace.copy(),
        refraction_trace_shift_s_sorted=dataset.expected_refraction_trace_shift_s,
        trace_static_status_sorted=ok_trace_status,
        trace_static_valid_mask_sorted=np.ones(n_traces, dtype=bool),
        estimated_first_break_time_s_sorted=dataset.noiseless_first_break_time_s,
        first_break_residual_s_sorted=(
            dataset.first_break_time_s - dataset.noiseless_first_break_time_s
        ),
        row_trace_index_sorted=dataset.sorted_trace_index,
        row_source_node_id=dataset.source_node_id_sorted,
        row_receiver_node_id=dataset.receiver_node_id_sorted,
        row_distance_m=dataset.offset_m,
        observed_pick_time_s=dataset.first_break_time_s,
        modeled_pick_time_s=dataset.noiseless_first_break_time_s,
        residual_time_s=dataset.first_break_time_s - dataset.noiseless_first_break_time_s,
        used_row_mask=dataset.valid_pick_mask,
        rejected_by_robust_mask=np.zeros(n_traces, dtype=bool),
        qc={
            'workflow': 'refraction_statics',
            'method': 'multilayer_time_term',
            'conversion_mode': 't1lsst_multilayer',
            'layer_count': 2,
            'fixture': dataset.name,
            'sign_convention': dataset.sign_convention,
        },
        node_sh1_weathering_thickness_m=node['sh1_m'],
        node_sh2_weathering_thickness_m=node['sh2_m'],
        source_t2_time_s=base.true_source_endpoint_t2_s,
        source_v3_m_s=np.full(source.endpoint_key.shape, dataset.true_v3_m_s),
        source_sh1_weathering_thickness_m=base.true_source_endpoint_sh1_m,
        source_sh2_weathering_thickness_m=base.true_source_endpoint_sh2_m,
        receiver_t2_time_s=base.true_receiver_endpoint_t2_s,
        receiver_v3_m_s=np.full(receiver.endpoint_key.shape, dataset.true_v3_m_s),
        receiver_sh1_weathering_thickness_m=base.true_receiver_endpoint_sh1_m,
        receiver_sh2_weathering_thickness_m=base.true_receiver_endpoint_sh2_m,
        row_layer_kind=dataset.layer_kind,
        row_layer_index=np.where(dataset.layer_kind == 'v2_t1', 1, 2).astype(np.int64),
        row_source_endpoint_key=dataset.source_endpoint_key_sorted,
        row_receiver_endpoint_key=dataset.receiver_endpoint_key_sorted,
        row_rejection_reason=np.full(n_traces, 'ok', dtype='<U32'),
        row_velocity_m_s=np.where(
            dataset.layer_kind == 'v2_t1',
            dataset.true_v2_m_s,
            dataset.true_v3_m_s,
        ).astype(np.float64),
    )


def _combined_node_arrays(
    dataset: SyntheticRefractionFieldCorrectionDataset,
) -> dict[str, np.ndarray]:
    base = dataset.base_dataset
    source = dataset.source_endpoint_table
    receiver = dataset.receiver_endpoint_table
    node_id = np.concatenate([source.node_id, receiver.node_id]).astype(np.int64)
    source_count = int(source.node_id.shape[0])
    kind = np.concatenate(
        [
            np.full(source_count, 'source', dtype='<U16'),
            np.full(receiver.node_id.shape, 'receiver', dtype='<U16'),
        ]
    )
    x_m = np.concatenate([source.x_m, receiver.x_m])
    y_m = np.concatenate([source.y_m, receiver.y_m])
    elevation_m = np.concatenate([source.elevation_m, receiver.elevation_m])
    sh1_m = np.concatenate(
        [base.true_source_endpoint_sh1_m, base.true_receiver_endpoint_sh1_m]
    )
    sh2_m = np.concatenate(
        [base.true_source_endpoint_sh2_m, base.true_receiver_endpoint_sh2_m]
    )
    t1_s = np.concatenate(
        [base.true_source_endpoint_t1_s, base.true_receiver_endpoint_t1_s]
    )
    wcor_s = np.concatenate([source.refraction_shift_s, receiver.refraction_shift_s])
    pick_count = np.concatenate([source.pick_count, receiver.pick_count]).astype(np.int64)
    status = np.where(node_id >= 0, 'ok', 'inactive').astype('<U48')
    keep = node_id >= 0
    order = np.argsort(node_id[keep])
    return {
        'node_id': np.ascontiguousarray(node_id[keep][order], dtype=np.int64),
        'x_m': np.ascontiguousarray(x_m[keep][order], dtype=np.float64),
        'y_m': np.ascontiguousarray(y_m[keep][order], dtype=np.float64),
        'elevation_m': np.ascontiguousarray(elevation_m[keep][order], dtype=np.float64),
        'kind': np.ascontiguousarray(kind[keep][order], dtype='<U16'),
        'sh1_m': np.ascontiguousarray(sh1_m[keep][order], dtype=np.float64),
        'sh2_m': np.ascontiguousarray(sh2_m[keep][order], dtype=np.float64),
        'total_thickness_m': np.ascontiguousarray(
            (sh1_m + sh2_m)[keep][order],
            dtype=np.float64,
        ),
        'refractor_elevation_m': np.ascontiguousarray(
            (elevation_m - sh1_m - sh2_m)[keep][order],
            dtype=np.float64,
        ),
        't1_s': np.ascontiguousarray(t1_s[keep][order], dtype=np.float64),
        'wcor_s': np.ascontiguousarray(wcor_s[keep][order], dtype=np.float64),
        'pick_count': np.ascontiguousarray(pick_count[keep][order], dtype=np.int64),
        'status': np.ascontiguousarray(status[keep][order], dtype='<U48'),
    }


def _manual_static_request_value(value_s: float, *, sign_convention: str) -> float:
    if sign_convention == 'delay_positive_ms':
        return value_s * 1000.0
    return value_s
