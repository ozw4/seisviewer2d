"""Cell V2 metadata mapping for app artifact DTOs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.statics.refraction.contracts.model import RefractionStaticModelRequest

_STATUS_DTYPE = '<U32'
_CELL_ID_PROBE_V2_BASE_M_S = 10_000_000.0


@dataclass(frozen=True)
class CellV2Metadata:
    node_cell_id: np.ndarray | None
    source_cell_id: np.ndarray | None
    receiver_cell_id: np.ndarray | None
    source_cell_id_sorted: np.ndarray | None
    source_v2_m_s_sorted: np.ndarray | None
    source_v2_status_sorted: np.ndarray | None
    receiver_cell_id_sorted: np.ndarray | None
    receiver_v2_m_s_sorted: np.ndarray | None
    receiver_v2_status_sorted: np.ndarray | None


def cell_v2_metadata_from_core_weathering(
    *,
    core_weathering: object,
    model: RefractionStaticModelRequest,
    cell_id_probe: object | None = None,
) -> CellV2Metadata:
    """Copy solve-cell V2 metadata exposed by the external weathering result."""
    if model.bedrock_velocity_mode != 'solve_cell' or model.refractor_cell is None:
        return CellV2Metadata(
            node_cell_id=None,
            source_cell_id=None,
            receiver_cell_id=None,
            source_cell_id_sorted=None,
            source_v2_m_s_sorted=None,
            source_v2_status_sorted=None,
            receiver_cell_id_sorted=None,
            receiver_v2_m_s_sorted=None,
            receiver_v2_status_sorted=None,
        )

    source = core_weathering.source_endpoint
    receiver = core_weathering.receiver_endpoint
    probe_source = cell_id_probe.source_endpoint if cell_id_probe is not None else None
    probe_receiver = (
        cell_id_probe.receiver_endpoint if cell_id_probe is not None else None
    )
    node_cell_id = (
        _cell_id_from_core_probe_v2(
            local_v2_m_s=cell_id_probe.node_v2_m_s,
            cell_id=core_weathering.cell_id,
        )
        if cell_id_probe is not None
        else _cell_id_from_core_local_v2(
            local_v2_m_s=core_weathering.node_v2_m_s,
            local_v2_status=core_weathering.node_local_v2_status,
            cell_id=core_weathering.cell_id,
            cell_v2_m_s=core_weathering.cell_v2_m_s,
            cell_velocity_status=core_weathering.cell_velocity_status,
        )
    )
    source_cell_id = _cell_id_from_core_endpoint_v2(
        endpoint=source,
        cell_id=core_weathering.cell_id,
        cell_v2_m_s=core_weathering.cell_v2_m_s,
        cell_velocity_status=core_weathering.cell_velocity_status,
        cell_id_probe_endpoint=probe_source,
    )
    receiver_cell_id = _cell_id_from_core_endpoint_v2(
        endpoint=receiver,
        cell_id=core_weathering.cell_id,
        cell_v2_m_s=core_weathering.cell_v2_m_s,
        cell_velocity_status=core_weathering.cell_velocity_status,
        cell_id_probe_endpoint=probe_receiver,
    )
    (
        source_cell_id_sorted,
        source_v2_m_s_sorted,
        source_v2_status_sorted,
    ) = _endpoint_v2_by_key_sorted(
        endpoint_key_sorted=core_weathering.source_endpoint_key_sorted,
        endpoint_key=source.endpoint_key,
        endpoint_cell_id=source_cell_id,
        endpoint_v2_m_s=source.v2_m_s,
        endpoint_v2_status=source.local_v2_status,
    )
    (
        receiver_cell_id_sorted,
        receiver_v2_m_s_sorted,
        receiver_v2_status_sorted,
    ) = _endpoint_v2_by_key_sorted(
        endpoint_key_sorted=core_weathering.receiver_endpoint_key_sorted,
        endpoint_key=receiver.endpoint_key,
        endpoint_cell_id=receiver_cell_id,
        endpoint_v2_m_s=receiver.v2_m_s,
        endpoint_v2_status=receiver.local_v2_status,
    )

    return CellV2Metadata(
        node_cell_id=node_cell_id,
        source_cell_id=source_cell_id,
        receiver_cell_id=receiver_cell_id,
        source_cell_id_sorted=source_cell_id_sorted,
        source_v2_m_s_sorted=source_v2_m_s_sorted,
        source_v2_status_sorted=source_v2_status_sorted,
        receiver_cell_id_sorted=receiver_cell_id_sorted,
        receiver_v2_m_s_sorted=receiver_v2_m_s_sorted,
        receiver_v2_status_sorted=receiver_v2_status_sorted,
    )


def _cell_id_from_core_local_v2(
    *,
    local_v2_m_s: np.ndarray,
    local_v2_status: np.ndarray,
    cell_id: np.ndarray,
    cell_v2_m_s: np.ndarray,
    cell_velocity_status: np.ndarray,
) -> np.ndarray:
    local_v2 = np.asarray(local_v2_m_s, dtype=np.float64)
    local_status = np.asarray(local_v2_status, dtype=_STATUS_DTYPE)
    ids = np.asarray(cell_id, dtype=np.int64)
    cell_v2 = np.asarray(cell_v2_m_s, dtype=np.float64)
    cell_status = np.asarray(cell_velocity_status, dtype=_STATUS_DTYPE)
    out = np.full(local_v2.shape, -1, dtype=np.int64)
    for index, (value, status) in enumerate(
        zip(local_v2.tolist(), local_status.tolist(), strict=True)
    ):
        matches = _matching_cell_ids(
            value=float(value),
            local_status=str(status),
            cell_id=ids,
            cell_v2_m_s=cell_v2,
            cell_velocity_status=cell_status,
        )
        if matches.shape[0] == 1:
            out[index] = int(matches[0])
    return np.ascontiguousarray(out, dtype=np.int64)


def _cell_id_from_core_probe_v2(
    *,
    local_v2_m_s: np.ndarray,
    cell_id: np.ndarray,
) -> np.ndarray:
    local_v2 = np.asarray(local_v2_m_s, dtype=np.float64)
    valid_cell_ids = {int(value) for value in np.asarray(cell_id, dtype=np.int64)}
    out = np.full(local_v2.shape, -1, dtype=np.int64)
    valid = np.isfinite(local_v2)
    candidate = np.full(local_v2.shape, -1, dtype=np.int64)
    candidate[valid] = np.rint(
        local_v2[valid] - _CELL_ID_PROBE_V2_BASE_M_S
    ).astype(np.int64)
    for index, raw_cell in enumerate(candidate.tolist()):
        cell = int(raw_cell)
        if bool(valid[index]) and cell in valid_cell_ids:
            out[index] = cell
    return np.ascontiguousarray(out, dtype=np.int64)


def _cell_id_from_core_endpoint_v2(
    *,
    endpoint: object,
    cell_id: np.ndarray,
    cell_v2_m_s: np.ndarray,
    cell_velocity_status: np.ndarray,
    cell_id_probe_endpoint: object | None,
) -> np.ndarray:
    if cell_id_probe_endpoint is not None:
        return _cell_id_from_core_probe_v2(
            local_v2_m_s=cell_id_probe_endpoint.v2_m_s,
            cell_id=cell_id,
        )
    return _cell_id_from_core_local_v2(
        local_v2_m_s=endpoint.v2_m_s,
        local_v2_status=endpoint.local_v2_status,
        cell_id=cell_id,
        cell_v2_m_s=cell_v2_m_s,
        cell_velocity_status=cell_velocity_status,
    )


def _matching_cell_ids(
    *,
    value: float,
    local_status: str,
    cell_id: np.ndarray,
    cell_v2_m_s: np.ndarray,
    cell_velocity_status: np.ndarray,
) -> np.ndarray:
    if local_status == 'ok':
        match = np.isfinite(cell_v2_m_s) & np.isclose(
            cell_v2_m_s,
            value,
            rtol=1.0e-9,
            atol=1.0e-6,
        )
    elif local_status == 'inactive_v2_cell':
        match = cell_velocity_status == 'inactive'
    elif local_status == 'low_fold_v2_cell':
        match = cell_velocity_status == 'low_fold'
    else:
        return np.asarray([], dtype=np.int64)
    return np.ascontiguousarray(cell_id[match], dtype=np.int64)


def _endpoint_v2_by_key_sorted(
    *,
    endpoint_key_sorted: np.ndarray,
    endpoint_key: np.ndarray,
    endpoint_cell_id: np.ndarray,
    endpoint_v2_m_s: np.ndarray,
    endpoint_v2_status: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_to_index = {
        str(key): index
        for index, key in enumerate(np.asarray(endpoint_key, dtype=object).tolist())
    }
    n_traces = int(np.asarray(endpoint_key_sorted).shape[0])
    cell_id = np.full(n_traces, -1, dtype=np.int64)
    v2 = np.full(n_traces, np.nan, dtype=np.float64)
    status = np.full(n_traces, 'missing_endpoint', dtype=_STATUS_DTYPE)
    endpoint_cells = np.asarray(endpoint_cell_id, dtype=np.int64)
    endpoint_v2 = np.asarray(endpoint_v2_m_s, dtype=np.float64)
    endpoint_status = np.asarray(endpoint_v2_status, dtype=_STATUS_DTYPE)
    for index, raw_key in enumerate(
        np.asarray(endpoint_key_sorted, dtype=object).tolist()
    ):
        endpoint_index = key_to_index.get(str(raw_key))
        if endpoint_index is None:
            continue
        cell_id[index] = int(endpoint_cells[endpoint_index])
        v2[index] = float(endpoint_v2[endpoint_index])
        status[index] = str(endpoint_status[endpoint_index])
    return (
        np.ascontiguousarray(cell_id, dtype=np.int64),
        np.ascontiguousarray(v2, dtype=np.float64),
        np.ascontiguousarray(status, dtype=_STATUS_DTYPE),
    )
