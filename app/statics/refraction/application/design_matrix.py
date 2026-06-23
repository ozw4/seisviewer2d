"""Application adapter for external refraction design-matrix construction."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from seis_statics.refraction import design_matrix as core_design_matrix

from app.services.common.artifact_io import write_csv_atomic, write_json_atomic
from app.statics.refraction.artifacts.design_matrix import (
    REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME,
    REFRACTION_DESIGN_MATRIX_QC_JSON_NAME,
    all_refraction_design_matrix_layer_artifact_names,
    refraction_design_matrix_layer_artifact_names,
    refraction_design_matrix_layer_node_diagnostics_csv_name,
    refraction_design_matrix_layer_qc_json_name,
)
from app.statics.refraction.contracts.model import RefractionStaticModelRequest
from app.statics.refraction.core_options import (
    core_input_model_from_app,
    model_options_from_request,
)
from app.statics.refraction.domain.types import (
    RefractionDesignMatrixNodeDiagnostics,
    RefractionStaticDesignMatrix,
    RefractionStaticInputModel,
    ResolvedRefractionFirstLayer,
)

RefractionStaticDesignMatrixError = (
    core_design_matrix.RefractionStaticDesignMatrixError
)
OUTSIDE_REFRACTOR_CELL_GRID_REASON = (
    core_design_matrix.OUTSIDE_REFRACTOR_CELL_GRID_REASON
)
LOW_FOLD_CELL_REJECTION_REASON = core_design_matrix.LOW_FOLD_CELL_REJECTION_REASON
LOW_FOLD_CELL_VELOCITY_STATUS = core_design_matrix.LOW_FOLD_CELL_VELOCITY_STATUS

_NODE_DIAGNOSTIC_COLUMNS = (
    'node_id',
    'matrix_column',
    'endpoint_kind',
    'endpoint_key',
    'source_endpoint_key',
    'receiver_endpoint_key',
    'active',
    'n_rows_pre_filter',
    'n_rows_post_filter',
    'n_nonzero_entries',
    'status',
    'reason',
    'first_trace_indices_pre_filter',
)


def build_refraction_static_design_matrix(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    include_diagnostics: bool = False,
    min_observations_per_node: int | None = None,
) -> RefractionStaticDesignMatrix:
    """Build the physical GLI sparse system through the external core."""
    _validate_input_model_trace_shapes(input_model)
    _validate_fixed_bedrock_velocity_compat(model)
    core_design = core_design_matrix.build_refraction_static_design_matrix(
        input_model=core_input_model_from_app(input_model),
        model=_core_model_from_app(model),
        resolved_first_layer=resolved_first_layer,
        min_observations_per_node=min_observations_per_node,
        include_diagnostics=include_diagnostics,
    )
    return _app_design_matrix_from_core(core_design)


def build_refraction_static_cell_design_matrix(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    include_diagnostics: bool = False,
    min_observations_per_node: int | None = None,
) -> RefractionStaticDesignMatrix:
    """Build a cell-based GLI sparse system through the external core."""
    _validate_input_model_trace_shapes(input_model)
    core_design = core_design_matrix.build_refraction_static_cell_design_matrix(
        input_model=core_input_model_from_app(input_model),
        model=model_options_from_request(model),
        resolved_first_layer=resolved_first_layer,
        min_observations_per_node=min_observations_per_node,
        include_diagnostics=include_diagnostics,
    )
    return _app_design_matrix_from_core(core_design)


def build_refraction_static_design_matrix_from_arrays(
    **kwargs: Any,
) -> RefractionStaticDesignMatrix:
    """Build a refraction static design matrix from sorted observation arrays."""
    core_design = core_design_matrix.build_refraction_static_design_matrix_from_arrays(
        **kwargs
    )
    return _app_design_matrix_from_core(core_design)


def _core_model_from_app(model: RefractionStaticModelRequest) -> object:
    if getattr(model, 'bedrock_velocity_mode', None) == 'solve_cell':
        return model_options_from_request(model)
    return model


def write_refraction_design_matrix_diagnostics_artifacts(
    job_dir: Path,
    design_matrix: RefractionStaticDesignMatrix,
) -> dict[str, Path]:
    """Write design-matrix QC JSON and node diagnostics CSV artifacts."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME
    csv_path = root / REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME

    node_diagnostics = design_matrix.node_diagnostics
    if not node_diagnostics:
        node_diagnostics = _build_node_diagnostics(
            design=design_matrix,
        )
    qc = core_design_matrix.summarize_refraction_static_design_matrix(
        _core_design_matrix_view(
            design_matrix,
            node_diagnostics=node_diagnostics,
        )
    )
    write_json_atomic(
        qc_path,
        qc,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        trailing_newline=True,
    )
    write_csv_atomic(
        csv_path,
        columns=_NODE_DIAGNOSTIC_COLUMNS,
        rows=[_node_diagnostic_csv_row(item) for item in node_diagnostics],
        extrasaction='raise',
        lineterminator='\r\n',
    )
    return {'qc_json': qc_path, 'node_diagnostics_csv': csv_path}


def _build_node_diagnostics(
    *,
    design: RefractionStaticDesignMatrix,
) -> tuple[RefractionDesignMatrixNodeDiagnostics, ...]:
    return _app_node_diagnostics_from_core(
        core_design_matrix.build_refraction_design_matrix_node_diagnostics(
            _core_design_matrix_view(design)
        )
    )


def _app_design_matrix_from_core(
    core_design: core_design_matrix.RefractionStaticDesignMatrix,
) -> RefractionStaticDesignMatrix:
    payload = {
        field.name: getattr(core_design, field.name)
        for field in fields(RefractionStaticDesignMatrix)
    }
    payload['node_diagnostics'] = _app_node_diagnostics_from_core(
        core_design.node_diagnostics
    )
    return RefractionStaticDesignMatrix(**payload)


def _core_design_matrix_view(
    design: RefractionStaticDesignMatrix,
    *,
    node_diagnostics: tuple[RefractionDesignMatrixNodeDiagnostics, ...] | None = None,
) -> object:
    payload: dict[str, Any] = {}
    for field in fields(RefractionStaticDesignMatrix):
        payload[field.name] = getattr(design, field.name)
    payload['node_diagnostics'] = (
        design.node_diagnostics if node_diagnostics is None else node_diagnostics
    )
    return SimpleNamespace(**payload)


def _validate_input_model_trace_shapes(input_model: RefractionStaticInputModel) -> None:
    n_traces = int(getattr(input_model, 'n_traces'))
    pick_time = np.asarray(getattr(input_model, 'pick_time_s_sorted'))
    if pick_time.shape != (n_traces,):
        raise RefractionStaticDesignMatrixError(
            'pick_time_s_sorted shape mismatch: '
            f'expected {(n_traces,)}, got {pick_time.shape}'
        )


def _validate_fixed_bedrock_velocity_compat(
    model: RefractionStaticModelRequest,
) -> None:
    if getattr(model, 'bedrock_velocity_mode', None) != 'fixed_global':
        return
    fixed_velocity = getattr(model, 'bedrock_velocity_m_s', None)
    weathering_velocity = getattr(model, 'weathering_velocity_m_s', None)
    first_layer = getattr(model, 'first_layer', None)
    if weathering_velocity is None and first_layer is not None:
        weathering_velocity = getattr(first_layer, 'weathering_velocity_m_s', None)
    if (
        fixed_velocity is not None
        and weathering_velocity is not None
        and float(fixed_velocity) <= float(weathering_velocity)
    ):
        raise RefractionStaticDesignMatrixError(
            'model.bedrock_velocity_m_s must be greater than '
            'model.weathering_velocity_m_s'
        )


def _app_node_diagnostics_from_core(
    diagnostics: tuple[core_design_matrix.RefractionDesignMatrixNodeDiagnostics, ...],
) -> tuple[RefractionDesignMatrixNodeDiagnostics, ...]:
    return tuple(
        RefractionDesignMatrixNodeDiagnostics(**asdict(item))
        for item in diagnostics
    )


def _node_diagnostic_csv_row(
    item: RefractionDesignMatrixNodeDiagnostics,
) -> dict[str, object]:
    payload = asdict(item)
    payload['active'] = 'true' if item.active else 'false'
    payload['source_endpoint_key'] = item.source_endpoint_key or ''
    payload['receiver_endpoint_key'] = item.receiver_endpoint_key or ''
    payload['first_trace_indices_pre_filter'] = json.dumps(
        list(item.first_trace_indices_pre_filter),
        separators=(',', ':'),
    )
    return payload


__all__ = [
    'LOW_FOLD_CELL_REJECTION_REASON',
    'LOW_FOLD_CELL_VELOCITY_STATUS',
    'OUTSIDE_REFRACTOR_CELL_GRID_REASON',
    'REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME',
    'REFRACTION_DESIGN_MATRIX_QC_JSON_NAME',
    'RefractionStaticDesignMatrix',
    'RefractionStaticDesignMatrixError',
    'all_refraction_design_matrix_layer_artifact_names',
    'build_refraction_static_cell_design_matrix',
    'build_refraction_static_design_matrix',
    'build_refraction_static_design_matrix_from_arrays',
    'refraction_design_matrix_layer_artifact_names',
    'refraction_design_matrix_layer_node_diagnostics_csv_name',
    'refraction_design_matrix_layer_qc_json_name',
    'write_refraction_design_matrix_diagnostics_artifacts',
]
