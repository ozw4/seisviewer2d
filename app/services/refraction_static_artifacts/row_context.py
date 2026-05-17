"""Shared row-context helpers for refraction static QC artifacts."""

from __future__ import annotations

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_cell_coordinates import (
    effective_refraction_cell_grid_config,
    project_refraction_cell_points,
)
from app.services.refraction_static_types import RefractionDatumStaticsResult
from app.services.refraction_static_artifacts.contract import (
    RefractionStaticArtifactError,
)
from app.services.refraction_static_artifacts.formatters import _float_or_nan
from app.services.refraction_static_artifacts.registry import (
    _request_cell_velocity_layer_kinds,
)
from app.services.refraction_static_artifacts.validation import (
    _required_positive_qc_int,
)


def _scalar_str(value: object) -> np.ndarray:
    text = '' if value is None else str(value)
    return np.asarray(text, dtype=f'<U{max(1, len(text))}')


def _int_array(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.int64)


def _float_array(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64)


def _cell_id_float_array(value: object) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).copy()
    out[out < 0] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)


def _bool_array(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=bool)


def _string_array(value: object) -> np.ndarray:
    raw = [str(item) for item in np.asarray(value).tolist()]
    max_len = max([1, *(len(item) for item in raw)])
    return np.ascontiguousarray(raw, dtype=f'<U{max_len}')


def _residual_row_layer_context(
    result: RefractionDatumStaticsResult,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    kind = np.full(n_rows, '', dtype='<U16')
    index = np.zeros(n_rows, dtype=np.int64)
    raw_kind = getattr(result, 'row_layer_kind', None)
    raw_index = getattr(result, 'row_layer_index', None)
    if raw_kind is not None:
        kind = np.asarray(raw_kind).astype('<U16', copy=False)
        if kind.shape != (n_rows,):
            raise RefractionStaticArtifactError(
                'row_layer_kind length must match residual rows'
            )
    if raw_index is not None:
        index = np.asarray(raw_index, dtype=np.int64)
        if index.shape != (n_rows,):
            raise RefractionStaticArtifactError(
                'row_layer_index length must match residual rows'
            )
    return np.ascontiguousarray(kind), np.ascontiguousarray(index)


def _residual_row_string_context(
    result: RefractionDatumStaticsResult,
    field: str,
) -> np.ndarray:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    raw = getattr(result, field, None)
    if raw is None:
        return np.full(n_rows, '', dtype=object)
    out = np.asarray(raw, dtype=object)
    if out.shape != (n_rows,):
        raise RefractionStaticArtifactError(
            f'{field} length must match residual rows'
        )
    return np.ascontiguousarray(out, dtype=object)


def _residual_row_velocity_context(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    raw = getattr(result, 'row_velocity_m_s', None)
    if raw is not None:
        out = np.asarray(raw, dtype=np.float64)
        if out.shape != (n_rows,):
            raise RefractionStaticArtifactError(
                'row_velocity_m_s length must match residual rows'
            )
        return np.ascontiguousarray(out, dtype=np.float64)
    return _row_velocity_from_cell_or_scalar(result, n_rows)


def _row_velocity_from_cell_or_scalar(
    result: RefractionDatumStaticsResult,
    n_rows: int,
) -> np.ndarray:
    out = np.full(n_rows, _float_or_nan(result.bedrock_velocity_m_s), dtype=np.float64)
    if result.row_midpoint_cell_id is None or result.cell_bedrock_velocity_m_s is None:
        return np.ascontiguousarray(out, dtype=np.float64)
    cell_id = np.asarray(result.row_midpoint_cell_id, dtype=np.int64)
    if cell_id.shape != (n_rows,):
        return np.ascontiguousarray(out, dtype=np.float64)
    velocity = np.asarray(result.cell_bedrock_velocity_m_s, dtype=np.float64)
    active_cell_id = result.active_cell_id
    if active_cell_id is not None:
        active = np.asarray(active_cell_id, dtype=np.int64)
        if active.shape == velocity.shape:
            out.fill(np.nan)
            for raw_cell, raw_velocity in zip(active.tolist(), velocity.tolist(), strict=True):
                rows = cell_id == int(raw_cell)
                out[rows] = float(raw_velocity)
            return np.ascontiguousarray(out, dtype=np.float64)
    if velocity.ndim == 1 and velocity.size > int(np.max(cell_id, initial=-1)):
        valid = (cell_id >= 0) & (cell_id < int(velocity.size))
        out[valid] = velocity[cell_id[valid]]
    return np.ascontiguousarray(out, dtype=np.float64)


def _residual_row_cell_context(
    result: RefractionDatumStaticsResult,
    *,
    req: RefractionStaticApplyRequest | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    empty = np.full(n_rows, -1, dtype=np.int64)
    request = RefractionStaticApplyRequest.model_validate(req) if req is not None else None
    request_cell_layer_kinds = (
        _request_cell_velocity_layer_kinds(request) if request is not None else ()
    )
    request_has_cell_velocity = bool(request_cell_layer_kinds)
    if (
        not request_has_cell_velocity
        and result.bedrock_velocity_mode != 'solve_cell'
    ):
        return empty, empty.copy(), empty.copy()

    if result.row_midpoint_cell_id is None:
        raise RefractionStaticArtifactError(
            'cell velocity residual rows require row_midpoint_cell_id'
        )
    cell_id = np.ascontiguousarray(result.row_midpoint_cell_id, dtype=np.int64)
    if cell_id.shape != (n_rows,):
        raise RefractionStaticArtifactError(
            'row_midpoint_cell_id length must match residual rows'
        )
    if request is not None and request_has_cell_velocity:
        layer_kind_by_row, _layer_index_by_row = _residual_row_layer_context(result)
        if np.any(layer_kind_by_row.astype(str, copy=False) != ''):
            supported_layer = np.isin(
                layer_kind_by_row.astype(str, copy=False),
                np.asarray(request_cell_layer_kinds, dtype=str),
            )
            cell_id = np.where(supported_layer, cell_id, -1)
    number_of_cell_x = _residual_cell_x_count(result=result, req=req)
    cell_ix = np.full(n_rows, -1, dtype=np.int64)
    cell_iy = np.full(n_rows, -1, dtype=np.int64)
    if number_of_cell_x is not None:
        valid = cell_id >= 0
        cell_ix[valid] = cell_id[valid] % number_of_cell_x
        cell_iy[valid] = cell_id[valid] // number_of_cell_x
    return (
        np.ascontiguousarray(cell_id, dtype=np.int64),
        np.ascontiguousarray(cell_ix, dtype=np.int64),
        np.ascontiguousarray(cell_iy, dtype=np.int64),
    )


def _residual_cell_x_count(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest | None,
) -> int | None:
    if req is not None:
        request = RefractionStaticApplyRequest.model_validate(req)
        refractor_cell = request.model.refractor_cell
        if _request_cell_velocity_layer_kinds(request) and refractor_cell is not None:
            return int(
                effective_refraction_cell_grid_config(
                    refractor_cell
                ).number_of_cell_x
            )
    raw = result.qc.get('number_of_cell_x')
    if raw is None:
        return None
    return _required_positive_qc_int(result.qc, 'number_of_cell_x')


def _residual_rejection_reason(
    *,
    used: bool,
    rejected_by_robust: bool,
    explicit_reason: object = '',
) -> str:
    if rejected_by_robust:
        return 'robust_outlier'
    reason = str(explicit_reason)
    if reason:
        return reason
    if not used:
        return 'not_used'
    return 'ok'


def _source_job_id(
    source_job_id: str | None,
    req: RefractionStaticApplyRequest | None,
) -> str:
    raw = source_job_id
    if raw is None and req is not None:
        raw = getattr(req, 'source_job_id', None)
    return '' if raw is None else str(raw)


def _first_break_export_layer_context(
    result: RefractionDatumStaticsResult,
) -> tuple[np.ndarray, np.ndarray]:
    return _residual_row_layer_context(result)


def _row_endpoint_id_context(
    result: RefractionDatumStaticsResult,
    *,
    endpoint: str,
    row_endpoint_key: np.ndarray,
    value_field: str,
) -> np.ndarray:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    keys = (
        result.source_endpoint_key
        if endpoint == 'source'
        else result.receiver_endpoint_key
    )
    values = getattr(result, value_field)
    lookup = {
        str(key): value
        for key, value in zip(
            np.asarray(keys).tolist(),
            np.asarray(values).tolist(),
            strict=True,
        )
    }
    out = np.full(n_rows, '', dtype=object)
    for index, raw_key in enumerate(row_endpoint_key.tolist()):
        value = lookup.get(str(raw_key))
        if value is not None:
            out[index] = value
    return np.ascontiguousarray(out, dtype=object)


def _row_endpoint_float_context(
    result: RefractionDatumStaticsResult,
    *,
    endpoint: str,
    row_endpoint_key: np.ndarray,
    value_field: str,
) -> np.ndarray:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    keys = (
        result.source_endpoint_key
        if endpoint == 'source'
        else result.receiver_endpoint_key
    )
    values = getattr(result, value_field)
    lookup = {
        str(key): float(value)
        for key, value in zip(
            np.asarray(keys).tolist(),
            np.asarray(values).tolist(),
            strict=True,
        )
    }
    out = np.full(n_rows, np.nan, dtype=np.float64)
    for index, raw_key in enumerate(row_endpoint_key.tolist()):
        value = lookup.get(str(raw_key))
        if value is not None:
            out[index] = value
    return np.ascontiguousarray(out, dtype=np.float64)


def _first_break_row_time_terms(
    result: RefractionDatumStaticsResult,
    *,
    layer_kind_by_row: np.ndarray,
    source_key_by_row: np.ndarray,
    receiver_key_by_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    source = np.full(n_rows, np.nan, dtype=np.float64)
    receiver = np.full(n_rows, np.nan, dtype=np.float64)
    source_t1_sorted = np.asarray(
        result.source_half_intercept_time_s_sorted,
        dtype=np.float64,
    )
    receiver_t1_sorted = np.asarray(
        result.receiver_half_intercept_time_s_sorted,
        dtype=np.float64,
    )
    source_t2 = _endpoint_time_lookup(
        result.source_endpoint_key,
        result.source_t2_time_s,
    )
    receiver_t2 = _endpoint_time_lookup(
        result.receiver_endpoint_key,
        result.receiver_t2_time_s,
    )
    source_t3 = _endpoint_time_lookup(
        result.source_endpoint_key,
        result.source_t3_time_s,
    )
    receiver_t3 = _endpoint_time_lookup(
        result.receiver_endpoint_key,
        result.receiver_t3_time_s,
    )
    for row_index, trace_index in enumerate(result.row_trace_index_sorted.tolist()):
        kind = str(layer_kind_by_row[row_index])
        if kind == 'v3_t2':
            source[row_index] = source_t2.get(
                str(source_key_by_row[row_index]),
                np.nan,
            )
            receiver[row_index] = receiver_t2.get(
                str(receiver_key_by_row[row_index]),
                np.nan,
            )
        elif kind == 'vsub_t3':
            source[row_index] = source_t3.get(
                str(source_key_by_row[row_index]),
                np.nan,
            )
            receiver[row_index] = receiver_t3.get(
                str(receiver_key_by_row[row_index]),
                np.nan,
            )
        elif kind == 'v2_t1':
            source[row_index] = source_t1_sorted[int(trace_index)]
            receiver[row_index] = receiver_t1_sorted[int(trace_index)]
    return (
        np.ascontiguousarray(source, dtype=np.float64),
        np.ascontiguousarray(receiver, dtype=np.float64),
    )


def _endpoint_time_lookup(
    endpoint_key: np.ndarray,
    values: np.ndarray | None,
) -> dict[str, float]:
    if values is None:
        return {}
    return {
        str(key): float(value)
        for key, value in zip(
            np.asarray(endpoint_key).tolist(),
            np.asarray(values, dtype=np.float64).tolist(),
            strict=True,
        )
    }


def _first_break_moveout_time_s(
    *,
    modeled_pick_time_s: np.ndarray,
    source_time_term_s: np.ndarray,
    receiver_time_term_s: np.ndarray,
) -> np.ndarray:
    modeled = np.asarray(modeled_pick_time_s, dtype=np.float64)
    source = np.asarray(source_time_term_s, dtype=np.float64)
    receiver = np.asarray(receiver_time_term_s, dtype=np.float64)
    out = modeled - source - receiver
    invalid = ~np.isfinite(modeled) | ~np.isfinite(source) | ~np.isfinite(receiver)
    out[invalid] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)


def _midpoint_coordinate(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    out = 0.5 * (left_arr + right_arr)
    out[~np.isfinite(left_arr) | ~np.isfinite(right_arr)] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)


def _first_break_fit_inline_crossline(
    *,
    midpoint_x_m: np.ndarray,
    midpoint_y_m: np.ndarray,
    req: RefractionStaticApplyRequest | None,
) -> tuple[np.ndarray, np.ndarray]:
    midpoint_x = np.asarray(midpoint_x_m, dtype=np.float64)
    midpoint_y = np.asarray(midpoint_y_m, dtype=np.float64)
    inline = np.full(midpoint_x.shape, np.nan, dtype=np.float64)
    crossline = np.full(midpoint_x.shape, np.nan, dtype=np.float64)
    if req is None or req.model.refractor_cell is None:
        return inline, crossline
    refractor_cell = req.model.refractor_cell
    if refractor_cell.coordinate_mode != 'line_2d_projected':
        return inline, crossline
    projected = project_refraction_cell_points(
        x_m=midpoint_x,
        y_m=midpoint_y,
        mode=refractor_cell.coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    if projected.projected_inline_m is not None:
        inline = projected.projected_inline_m
    if projected.projected_crossline_m is not None:
        crossline = projected.projected_crossline_m
    return (
        np.ascontiguousarray(inline, dtype=np.float64),
        np.ascontiguousarray(crossline, dtype=np.float64),
    )


__all__ = [
    '_bool_array',
    '_cell_id_float_array',
    '_endpoint_time_lookup',
    '_first_break_export_layer_context',
    '_first_break_fit_inline_crossline',
    '_first_break_moveout_time_s',
    '_first_break_row_time_terms',
    '_float_array',
    '_int_array',
    '_midpoint_coordinate',
    '_residual_rejection_reason',
    '_residual_row_cell_context',
    '_residual_row_layer_context',
    '_residual_row_string_context',
    '_residual_row_velocity_context',
    '_scalar_str',
    '_source_job_id',
    '_string_array',
    '_row_endpoint_float_context',
    '_row_endpoint_id_context',
]
