"""Static component CSV and component-QC artifact helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_types import RefractionDatumStaticsResult


def write_refraction_static_components_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    from app.services.refraction_static_artifacts import _legacy

    values = _legacy._validate_result(result)
    rows = _legacy._component_rows(values.result)
    _legacy._write_csv_atomic(
        Path(path),
        _legacy._component_columns(values.result),
        rows,
    )


def write_refraction_static_component_qc_artifacts(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    trace_csv_path: Path,
    endpoint_csv_path: Path,
    npz_path: Path,
    json_path: Path,
) -> dict[str, Any]:
    from app.services.refraction_static_artifacts import _legacy

    values = _legacy._validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    arrays = _legacy.build_refraction_static_component_qc_arrays(
        result=values.result,
        req=request,
    )
    _legacy._write_csv_atomic(
        Path(trace_csv_path),
        _legacy._STATIC_COMPONENT_QC_TRACE_COLUMNS,
        _legacy._static_component_qc_trace_rows(arrays),
    )
    _legacy._write_csv_atomic(
        Path(endpoint_csv_path),
        _legacy._STATIC_COMPONENT_QC_ENDPOINT_COLUMNS,
        _legacy._static_component_qc_endpoint_rows(arrays),
    )
    _legacy._write_npz_atomic(Path(npz_path), arrays)
    payload = _legacy.build_refraction_static_component_qc_payload(
        arrays=arrays,
    )
    _legacy._write_json_atomic(Path(json_path), payload)
    return payload


def build_refraction_static_component_qc_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, np.ndarray]:
    from app.services.refraction_static_artifacts import _legacy

    return _legacy.build_refraction_static_component_qc_arrays(result=result, req=req)


def build_refraction_static_component_qc_payload(
    *,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    from app.services.refraction_static_artifacts import _legacy

    return _legacy.build_refraction_static_component_qc_payload(arrays=arrays)


def _static_component_qc_trace_rows(
    arrays: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    from app.services.refraction_static_artifacts import _legacy

    return _legacy._static_component_qc_trace_rows(arrays)


def _static_component_qc_endpoint_rows(
    arrays: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    from app.services.refraction_static_artifacts import _legacy

    return _legacy._static_component_qc_endpoint_rows(arrays)


def _component_rows(result: RefractionDatumStaticsResult) -> list[dict[str, object]]:
    from app.services.refraction_static_artifacts import _legacy

    return _legacy._component_rows(result)


def _component_columns(result: RefractionDatumStaticsResult) -> tuple[str, ...]:
    from app.services.refraction_static_artifacts import _legacy

    return _legacy._component_columns(result)


__all__ = [
    '_component_columns',
    '_component_rows',
    '_static_component_qc_endpoint_rows',
    '_static_component_qc_trace_rows',
    'build_refraction_static_component_qc_arrays',
    'build_refraction_static_component_qc_payload',
    'write_refraction_static_component_qc_artifacts',
    'write_refraction_static_components_csv',
]
