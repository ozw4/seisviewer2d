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
    source_cell_id = _cell_id_by_node(
        endpoint_node_id=source.node_id,
        node_id=core_weathering.node_id,
        node_cell_id=node_cell_id,
    )
    receiver_cell_id = _cell_id_by_node(
        endpoint_node_id=receiver.node_id,
        node_id=core_weathering.node_id,
        node_cell_id=node_cell_id,
    )
    source_cell_id_sorted = _values_by_node(
        node_id_sorted=core_weathering.source_node_id_sorted,
        node_id=core_weathering.node_id,
        values=node_cell_id,
        dtype=np.int64,
        fill_value=-1,
    )
    receiver_cell_id_sorted = _values_by_node(
        node_id_sorted=core_weathering.receiver_node_id_sorted,
        node_id=core_weathering.node_id,
        values=node_cell_id,
        dtype=np.int64,
        fill_value=-1,
    )
    source_v2_m_s_sorted = _values_by_node(
        node_id_sorted=core_weathering.source_node_id_sorted,
        node_id=core_weathering.node_id,
        values=core_weathering.node_v2_m_s,
        dtype=np.float64,
        fill_value=np.nan,
    )
    receiver_v2_m_s_sorted = _values_by_node(
        node_id_sorted=core_weathering.receiver_node_id_sorted,
        node_id=core_weathering.node_id,
        values=core_weathering.node_v2_m_s,
        dtype=np.float64,
        fill_value=np.nan,
    )
    source_v2_status_sorted = _values_by_node(
        node_id_sorted=core_weathering.source_node_id_sorted,
        node_id=core_weathering.node_id,
        values=core_weathering.node_local_v2_status,
        dtype=_STATUS_DTYPE,
        fill_value='missing_node',
    )
    receiver_v2_status_sorted = _values_by_node(
        node_id_sorted=core_weathering.receiver_node_id_sorted,
        node_id=core_weathering.node_id,
        values=core_weathering.node_local_v2_status,
        dtype=_STATUS_DTYPE,
        fill_value='missing_node',
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


def _cell_id_by_node(
    *,
    endpoint_node_id: np.ndarray,
    node_id: np.ndarray,
    node_cell_id: np.ndarray,
) -> np.ndarray:
    lookup = {
        int(node): int(cell)
        for node, cell in zip(
            np.asarray(node_id, dtype=np.int64).tolist(),
            np.asarray(node_cell_id, dtype=np.int64).tolist(),
            strict=True,
        )
    }
    return np.ascontiguousarray(
        np.asarray(
            [
                lookup.get(int(node), -1)
                for node in np.asarray(endpoint_node_id, dtype=np.int64).tolist()
            ],
            dtype=np.int64,
        )
    )


def _values_by_node(
    *,
    node_id_sorted: np.ndarray,
    node_id: np.ndarray,
    values: np.ndarray,
    dtype: object,
    fill_value: object,
) -> np.ndarray:
    lookup = {
        int(node): value
        for node, value in zip(
            np.asarray(node_id, dtype=np.int64).tolist(),
            np.asarray(values, dtype=dtype).tolist(),
            strict=True,
        )
    }
    return np.ascontiguousarray(
        np.asarray(
            [
                lookup.get(int(node), fill_value)
                for node in np.asarray(node_id_sorted, dtype=np.int64).tolist()
            ],
            dtype=dtype,
        )
    )
