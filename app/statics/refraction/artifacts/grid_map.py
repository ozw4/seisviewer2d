"""Grid-map QC artifact builders and writers for refraction statics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.contracts.statics.refraction.apply import RefractionStaticApplyRequest
from app.services.refraction_static_cell_coordinates import (
    effective_refraction_cell_grid_config,
    refraction_cell_coordinate_metadata_from_config,
)
from app.services.refraction_static_design_matrix import (
    LOW_FOLD_CELL_VELOCITY_STATUS,
)
from app.services.refraction_static_layer_config import (
    normalize_refraction_static_layers,
)
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
)
from app.statics.refraction.artifacts.arrays import (
    _scalar_float,
    _scalar_int,
    _scalar_str,
    _string_array,
)
from app.statics.refraction.artifacts.cell_velocity import (
    build_refraction_refractor_velocity_grid_arrays,
)
from app.statics.refraction.artifacts.contract import (
    _GRID_MAP_QC_COLUMNS,
    ARTIFACT_VERSION,
    REFRACTION_GRID_MAP_QC_CSV_NAME,
    REFRACTION_GRID_MAP_QC_JSON_NAME,
    REFRACTION_GRID_MAP_QC_NPZ_NAME,
    RefractionStaticArtifactError,
)
from app.statics.refraction.artifacts.formatters import (
    _csv_float,
    _csv_grid_float,
    _json_float,
    _nan_if_none,
)
from app.statics.refraction.artifacts.io import (
    _assert_strict_json,
    _validate_no_object_arrays,
    _write_csv_atomic,
    _write_json_atomic,
    _write_npz_atomic,
)
from app.statics.refraction.artifacts.registry import (
    _request_cell_velocity_layer_kinds,
)
from app.statics.refraction.artifacts.stats import (
    _stat,
    _status_counts,
)


def write_refraction_grid_map_qc_csv(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> None:
    arrays = build_refraction_grid_map_qc_arrays(result=result, req=req)
    rows = _grid_map_qc_rows(arrays)
    _write_csv_atomic(Path(path), _GRID_MAP_QC_COLUMNS, rows)


def write_refraction_grid_map_qc_npz(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> None:
    arrays = build_refraction_grid_map_qc_arrays(result=result, req=req)
    _validate_no_object_arrays(
        arrays,
        artifact_name=REFRACTION_GRID_MAP_QC_NPZ_NAME,
    )
    _write_npz_atomic(Path(path), arrays)


def write_refraction_grid_map_qc_json(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> dict[str, Any]:
    payload = build_refraction_grid_map_qc_payload(result=result, req=req)
    _write_json_atomic(Path(path), payload)
    return payload


def build_refraction_grid_map_qc_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, np.ndarray]:
    request = RefractionStaticApplyRequest.model_validate(req)
    layer_kinds = _request_cell_velocity_layer_kinds(request)
    if not layer_kinds:
        raise RefractionStaticArtifactError(
            'grid map QC artifacts require a solve_cell velocity layer'
        )

    pieces: dict[str, list[np.ndarray]] = {
        'layer_kind': [],
        'cell_id': [],
        'cell_ix': [],
        'cell_iy': [],
        'cell_center_x_m': [],
        'cell_center_y_m': [],
        'cell_center_inline_m': [],
        'cell_center_crossline_m': [],
        'x_min_m': [],
        'x_max_m': [],
        'y_min_m': [],
        'y_max_m': [],
        'velocity_m_s': [],
        'initial_velocity_m_s': [],
        'velocity_update_from_initial_m_s': [],
        'slowness_s_per_m': [],
        'n_observations': [],
        'n_used_observations': [],
        'n_rejected_observations': [],
        'n_sources': [],
        'n_receivers': [],
        'residual_rms_ms': [],
        'residual_mad_ms': [],
        'status': [],
        'status_reason': [],
        'active_cell_mask': [],
        'coordinate_mode': [],
        'cell_velocity_component': [],
    }

    for layer_kind in layer_kinds:
        layer_arrays = build_refraction_refractor_velocity_grid_arrays(
            result=result,
            req=request,
            layer_kind=layer_kind,
        )
        mapped = {
            'layer_kind': layer_arrays['cell_velocity_layer_kind'],
            'cell_id': layer_arrays['cell_id'],
            'cell_ix': layer_arrays['cell_ix'],
            'cell_iy': layer_arrays['cell_iy'],
            'cell_center_x_m': layer_arrays['cell_center_x_m'],
            'cell_center_y_m': layer_arrays['cell_center_y_m'],
            'cell_center_inline_m': layer_arrays['cell_center_inline_m'],
            'cell_center_crossline_m': layer_arrays['cell_center_crossline_m'],
            'x_min_m': layer_arrays['x_min_m'],
            'x_max_m': layer_arrays['x_max_m'],
            'y_min_m': layer_arrays['y_min_m'],
            'y_max_m': layer_arrays['y_max_m'],
            'velocity_m_s': layer_arrays['velocity_m_s'],
            'initial_velocity_m_s': layer_arrays['initial_velocity_m_s'],
            'velocity_update_from_initial_m_s': (
                layer_arrays['velocity_update_from_initial_m_s']
            ),
            'slowness_s_per_m': layer_arrays['slowness_s_per_m'],
            'n_observations': layer_arrays['n_observations_per_cell'],
            'n_used_observations': layer_arrays['n_used_observations_per_cell'],
            'n_rejected_observations': (
                layer_arrays['n_rejected_observations_per_cell']
            ),
            'n_sources': layer_arrays['n_sources_per_cell'],
            'n_receivers': layer_arrays['n_receivers_per_cell'],
            'residual_rms_ms': layer_arrays['residual_rms_ms'],
            'residual_mad_ms': layer_arrays['residual_mad_ms'],
            'status': layer_arrays['velocity_status'],
            'status_reason': layer_arrays['status_reason'],
            'active_cell_mask': layer_arrays['active_cell_mask'],
            'coordinate_mode': layer_arrays['coordinate_mode'],
            'cell_velocity_component': layer_arrays['cell_velocity_component'],
        }
        for key, value in mapped.items():
            pieces[key].append(np.asarray(value))

    arrays: dict[str, np.ndarray] = {}
    for key, values in pieces.items():
        merged = np.concatenate(values)
        if merged.dtype.kind in {'U', 'S'}:
            arrays[key] = _string_array(merged)
        elif merged.dtype == bool:
            arrays[key] = np.ascontiguousarray(merged, dtype=bool)
        elif np.issubdtype(merged.dtype, np.integer):
            arrays[key] = np.ascontiguousarray(merged, dtype=np.int64)
        else:
            arrays[key] = np.ascontiguousarray(merged, dtype=np.float64)

    refractor_cell = request.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for grid map QC artifacts'
        )
    grid_config = effective_refraction_cell_grid_config(refractor_cell)
    arrays.update(
        {
            'artifact_version': _scalar_str(ARTIFACT_VERSION),
            'artifact_kind': _scalar_str('refraction_grid_map_qc'),
            'global_velocity_layer_behavior': _scalar_str(
                'omitted_from_grid_map_qc_rows'
            ),
            'coordinate_mode': _string_array(arrays['coordinate_mode']),
            'number_of_cell_x': _scalar_int(grid_config.number_of_cell_x),
            'number_of_cell_y': _scalar_int(grid_config.number_of_cell_y),
            'size_of_cell_x_m': _scalar_float(grid_config.size_of_cell_x_m),
            'size_of_cell_y_m': _scalar_float(
                _nan_if_none(grid_config.size_of_cell_y_m)
            ),
            'x_coordinate_origin_m': _scalar_float(
                grid_config.x_coordinate_origin_m
            ),
            'y_coordinate_origin_m': _scalar_float(
                grid_config.y_coordinate_origin_m
            ),
        }
    )
    _validate_no_object_arrays(
        arrays,
        artifact_name=REFRACTION_GRID_MAP_QC_NPZ_NAME,
    )
    return arrays


def build_refraction_grid_map_qc_payload(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, Any]:
    request = RefractionStaticApplyRequest.model_validate(req)
    layer_kinds = _request_cell_velocity_layer_kinds(request)
    if not layer_kinds:
        raise RefractionStaticArtifactError(
            'grid map QC artifacts require a solve_cell velocity layer'
        )
    arrays = build_refraction_grid_map_qc_arrays(result=result, req=request)
    refractor_cell = request.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for grid map QC artifacts'
        )
    grid_config = effective_refraction_cell_grid_config(refractor_cell)

    layers = {}
    raw_layer_kind = np.asarray(arrays['layer_kind']).astype(str, copy=False)
    for layer_kind in layer_kinds:
        mask = raw_layer_kind == layer_kind
        layers[layer_kind] = _grid_map_qc_layer_summary(arrays, mask=mask)

    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'artifact_kind': 'refraction_grid_map_qc',
        'row_count': int(raw_layer_kind.size),
        'cell_layer_count': len(layer_kinds),
        'cell_velocity_layer_kinds': list(layer_kinds),
        'global_velocity_layer_behavior': 'omitted_from_grid_map_qc_rows',
        'omitted_global_velocity_layers': _grid_map_qc_global_velocity_layers(
            request
        ),
        'artifacts': {
            'csv': REFRACTION_GRID_MAP_QC_CSV_NAME,
            'npz': REFRACTION_GRID_MAP_QC_NPZ_NAME,
            'json': REFRACTION_GRID_MAP_QC_JSON_NAME,
        },
        'grid': {
            'cell_assignment_mode': refractor_cell.assignment_mode,
            **refraction_cell_coordinate_metadata_from_config(refractor_cell),
            'outside_grid_policy': refractor_cell.outside_grid_policy,
            'number_of_cell_x': int(grid_config.number_of_cell_x),
            'number_of_cell_y': int(grid_config.number_of_cell_y),
            'size_of_cell_x_m': float(grid_config.size_of_cell_x_m),
            'size_of_cell_y_m': _json_float(grid_config.size_of_cell_y_m),
            'x_coordinate_origin_m': float(grid_config.x_coordinate_origin_m),
            'y_coordinate_origin_m': float(grid_config.y_coordinate_origin_m),
            'y_axis_unbounded': grid_config.size_of_cell_y_m is None,
        },
        'layers': layers,
    }
    _assert_strict_json(payload, artifact_name=REFRACTION_GRID_MAP_QC_JSON_NAME)
    return payload


def _grid_map_qc_rows(arrays: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_rows = int(arrays['layer_kind'].shape[0])
    for index in range(n_rows):
        rows.append(
            {
                'layer_kind': str(arrays['layer_kind'][index]),
                'cell_ix': int(arrays['cell_ix'][index]),
                'cell_iy': int(arrays['cell_iy'][index]),
                'cell_center_x_m': _csv_grid_float(
                    arrays['cell_center_x_m'][index]
                ),
                'cell_center_y_m': _csv_grid_float(
                    arrays['cell_center_y_m'][index]
                ),
                'cell_center_inline_m': _csv_grid_float(
                    arrays['cell_center_inline_m'][index]
                ),
                'cell_center_crossline_m': _csv_grid_float(
                    arrays['cell_center_crossline_m'][index]
                ),
                'velocity_m_s': _csv_float(arrays['velocity_m_s'][index]),
                'initial_velocity_m_s': _csv_float(
                    arrays['initial_velocity_m_s'][index]
                ),
                'velocity_update_from_initial_m_s': _csv_float(
                    arrays['velocity_update_from_initial_m_s'][index]
                ),
                'slowness_s_per_m': _csv_float(arrays['slowness_s_per_m'][index]),
                'n_observations': int(arrays['n_observations'][index]),
                'n_sources': int(arrays['n_sources'][index]),
                'n_receivers': int(arrays['n_receivers'][index]),
                'residual_rms_ms': _csv_float(arrays['residual_rms_ms'][index]),
                'residual_mad_ms': _csv_float(arrays['residual_mad_ms'][index]),
                'status': str(arrays['status'][index]),
                'status_reason': str(arrays['status_reason'][index]),
            }
        )
    return rows


def _grid_map_qc_layer_summary(
    arrays: dict[str, np.ndarray],
    *,
    mask: np.ndarray,
) -> dict[str, Any]:
    status = np.asarray(arrays['status']).astype(str, copy=False)[mask]
    status_reason = np.asarray(arrays['status_reason']).astype(str, copy=False)[mask]
    active_mask = np.asarray(arrays['active_cell_mask'], dtype=bool)[mask]
    velocity = np.asarray(arrays['velocity_m_s'], dtype=np.float64)[mask]
    active_velocity = velocity[active_mask & np.isfinite(velocity)]
    return {
        'active_cell_count': int(np.count_nonzero(active_mask)),
        'empty_cell_count': int(np.count_nonzero(status_reason == 'no_observations')),
        'low_fold_cell_count': int(
            np.count_nonzero(status == LOW_FOLD_CELL_VELOCITY_STATUS)
        ),
        'velocity_min_m_s': _stat(active_velocity, 'min'),
        'velocity_median_m_s': _stat(active_velocity, 'median'),
        'velocity_max_m_s': _stat(active_velocity, 'max'),
        'status_counts': _status_counts(status),
    }


def _grid_map_qc_global_velocity_layers(
    req: RefractionStaticApplyRequest,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for layer in normalize_refraction_static_layers(req.model):
        if layer.velocity_mode == 'solve_cell':
            continue
        rows.append(
            {
                'layer_kind': layer.kind,
                'velocity_mode': layer.velocity_mode,
                'row_behavior': 'omitted',
            }
        )
    return rows


__all__ = [
    'build_refraction_grid_map_qc_arrays',
    'build_refraction_grid_map_qc_payload',
    'write_refraction_grid_map_qc_csv',
    'write_refraction_grid_map_qc_json',
    'write_refraction_grid_map_qc_npz',
]
