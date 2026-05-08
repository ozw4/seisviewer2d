from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.api.schemas import (
    RefractionStaticFirstLayerRequest,
    RefractionStaticGeometryRequest,
    RefractionStaticLinkageRequest,
    RefractionStaticMoveoutRequest,
)
from app.services.refraction_static_inputs import (
    build_refraction_static_input_model_from_arrays,
)
from app.services.refraction_static_v1 import (
    REFRACTION_V1_ESTIMATES_CSV_NAME,
    REFRACTION_V1_QC_JSON_NAME,
    RefractionV1EstimationError,
    estimate_global_v1_from_direct_arrivals,
    write_refraction_v1_artifacts,
)
from app.tests._refraction_static_synthetic import (
    SYNTHETIC_V1_M_S,
    SYNTHETIC_V1_TOLERANCE_M_S,
    synthetic_direct_arrival_input_model,
    synthetic_first_layer_request,
)


def _geometry() -> RefractionStaticGeometryRequest:
    return RefractionStaticGeometryRequest.model_validate(
        {
            'source_id_byte': 1,
            'receiver_id_byte': 2,
            'source_x_byte': 3,
            'source_y_byte': 4,
            'receiver_x_byte': 5,
            'receiver_y_byte': 6,
            'source_elevation_byte': 7,
            'receiver_elevation_byte': 8,
            'source_depth_byte': None,
            'coordinate_scalar_byte': 9,
            'elevation_scalar_byte': 10,
            'coordinate_unit': 'm',
            'elevation_unit': 'm',
        }
    )


def _moveout() -> RefractionStaticMoveoutRequest:
    return RefractionStaticMoveoutRequest.model_validate(
        {
            'distance_source': 'geometry',
            'offset_byte': None,
        }
    )


def _first_layer(**overrides: Any) -> RefractionStaticFirstLayerRequest:
    payload = {
        'mode': 'estimate_direct_arrival',
        'min_weathering_velocity_m_s': 500.0,
        'max_weathering_velocity_m_s': 1200.0,
        'min_direct_offset_m': 20.0,
        'max_direct_offset_m': 140.0,
        'min_picks_per_fit': 5,
        'min_groups': 3,
        'robust_enabled': True,
        'robust_threshold': 3.5,
    }
    payload.update(overrides)
    return RefractionStaticFirstLayerRequest.model_validate(payload)


def _input_model(
    *,
    v1_m_s: float = 800.0,
    intercept_by_source: tuple[float, ...] = (0.010, 0.012, 0.014),
    offsets_m: tuple[float, ...] = (20.0, 40.0, 60.0, 80.0, 100.0, 120.0),
    pick_overrides: dict[tuple[int, int], float] | None = None,
):
    geom = _geometry()
    n_sources = len(intercept_by_source)
    n_offsets = len(offsets_m)
    n_traces = n_sources * n_offsets
    source_id = np.repeat(np.arange(100, 100 + n_sources), n_offsets)
    receiver_id = np.arange(1000, 1000 + n_traces)
    source_x = np.repeat(np.arange(n_sources, dtype=np.float64) * 1000.0, n_offsets)
    source_y = np.zeros(n_traces, dtype=np.float64)
    offsets = np.tile(np.asarray(offsets_m, dtype=np.float64), n_sources)
    receiver_x = source_x + offsets
    receiver_y = np.zeros(n_traces, dtype=np.float64)
    source_elevation = np.full(n_traces, 100.0, dtype=np.float64)
    receiver_elevation = np.full(n_traces, 95.0, dtype=np.float64)
    pick_time = np.empty(n_traces, dtype=np.float64)
    for source_index, intercept in enumerate(intercept_by_source):
        start = source_index * n_offsets
        stop = start + n_offsets
        pick_time[start:stop] = intercept + offsets[:n_offsets] / v1_m_s
    for (source_index, offset_index), value in (pick_overrides or {}).items():
        pick_time[source_index * n_offsets + offset_index] = float(value)

    headers = {
        geom.source_id_byte: source_id,
        geom.receiver_id_byte: receiver_id,
        geom.source_x_byte: source_x,
        geom.source_y_byte: source_y,
        geom.receiver_x_byte: receiver_x,
        geom.receiver_y_byte: receiver_y,
        geom.source_elevation_byte: source_elevation,
        geom.receiver_elevation_byte: receiver_elevation,
        geom.coordinate_scalar_byte: np.ones(n_traces, dtype=np.int64),
        geom.elevation_scalar_byte: np.ones(n_traces, dtype=np.int64),
    }
    return build_refraction_static_input_model_from_arrays(
        file_id='line-a',
        pick_time_s_sorted=pick_time,
        trace_headers_sorted=headers,
        geometry=geom,
        linkage=RefractionStaticLinkageRequest.model_validate({'mode': 'none'}),
        moveout=_moveout(),
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        n_samples=2000,
        dt=0.001,
    )


def test_v1_estimate_global_from_direct_arrivals(tmp_path: Path) -> None:
    result = estimate_global_v1_from_direct_arrivals(
        input_model=synthetic_direct_arrival_input_model(),
        first_layer=synthetic_first_layer_request(),
    )

    assert result.resolved_weathering_velocity_m_s == pytest.approx(
        SYNTHETIC_V1_M_S,
        abs=SYNTHETIC_V1_TOLERANCE_M_S,
    )
    assert result.qc['v1_status'] == 'estimated'
    assert result.qc['n_used_groups'] == 6
    assert set(result.group_status.tolist()) == {'ok'}
    assert np.any(result.group_n_used < result.group_n_candidates)

    paths = write_refraction_v1_artifacts(tmp_path, result)
    assert paths['qc_json'].name == REFRACTION_V1_QC_JSON_NAME
    assert paths['estimates_csv'].name == REFRACTION_V1_ESTIMATES_CSV_NAME

    qc = json.loads((tmp_path / REFRACTION_V1_QC_JSON_NAME).read_text())
    assert qc['resolved_weathering_velocity_m_s'] == pytest.approx(
        SYNTHETIC_V1_M_S,
        abs=SYNTHETIC_V1_TOLERANCE_M_S,
    )
    with (tmp_path / REFRACTION_V1_ESTIMATES_CSV_NAME).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert rows[0]['group_kind'] == 'source_endpoint'
    assert float(rows[0]['v1_m_s']) == pytest.approx(
        SYNTHETIC_V1_M_S,
        abs=SYNTHETIC_V1_TOLERANCE_M_S,
    )


def test_v1_estimate_robust_to_outlier_picks() -> None:
    model = _input_model(pick_overrides={(1, 2): 0.300})

    result = estimate_global_v1_from_direct_arrivals(
        input_model=model,
        first_layer=_first_layer(robust_threshold=2.5),
    )

    assert result.resolved_weathering_velocity_m_s == pytest.approx(800.0)
    assert np.min(result.group_n_used) < np.max(result.group_n_candidates)


def test_v1_estimate_respects_direct_offset_gate() -> None:
    offsets = (20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 300.0, 400.0)
    overrides = {
        (source_index, offset_index): 0.020 + offsets[offset_index] / 2200.0
        for source_index in range(3)
        for offset_index in (6, 7)
    }
    model = _input_model(offsets_m=offsets, pick_overrides=overrides)

    result = estimate_global_v1_from_direct_arrivals(
        input_model=model,
        first_layer=_first_layer(max_direct_offset_m=120.0),
    )

    assert result.resolved_weathering_velocity_m_s == pytest.approx(800.0)
    assert result.group_n_candidates.tolist() == [6, 6, 6]
    assert result.group_offset_max_m.tolist() == [120.0, 120.0, 120.0]


def test_v1_estimate_fails_with_insufficient_picks() -> None:
    model = _input_model(offsets_m=(20.0, 40.0, 60.0, 80.0))

    with pytest.raises(RefractionV1EstimationError, match='Insufficient'):
        estimate_global_v1_from_direct_arrivals(
            input_model=model,
            first_layer=_first_layer(min_picks_per_fit=5),
        )


def test_v1_estimate_rejects_velocity_outside_bounds() -> None:
    model = _input_model(v1_m_s=2200.0)

    with pytest.raises(RefractionV1EstimationError, match='velocity bounds'):
        estimate_global_v1_from_direct_arrivals(
            input_model=model,
            first_layer=_first_layer(
                min_weathering_velocity_m_s=500.0,
                max_weathering_velocity_m_s=1200.0,
            ),
        )
