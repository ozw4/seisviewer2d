from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import app.services.refraction_static_inputs as inputs_module
from app.api.schemas import (
    RefractionStaticApplyRequest,
    RefractionStaticGeometryRequest,
    RefractionStaticLinkageRequest,
    RefractionStaticModelRequest,
    RefractionStaticMoveoutRequest,
)
from app.core.state import AppState
from app.services.refraction_static_inputs import (
    REFRACTION_INPUT_PREVIEW_CSV_NAME,
    REFRACTION_INPUT_QC_JSON_NAME,
    build_refraction_static_input_model,
    build_refraction_static_input_model_from_arrays,
)


def _geometry(**overrides: Any) -> RefractionStaticGeometryRequest:
    payload = {
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
    payload.update(overrides)
    return RefractionStaticGeometryRequest.model_validate(payload)


def _moveout(**overrides: Any) -> RefractionStaticMoveoutRequest:
    payload = {
        'distance_source': 'geometry',
        'offset_byte': 11,
    }
    payload.update(overrides)
    return RefractionStaticMoveoutRequest.model_validate(payload)


def _linkage(mode: str = 'none') -> RefractionStaticLinkageRequest:
    payload: dict[str, Any] = {'mode': mode}
    if mode != 'none':
        payload['job_id'] = 'linkage-job'
    return RefractionStaticLinkageRequest.model_validate(payload)


def _headers(
    n_traces: int = 4,
    *,
    geometry: RefractionStaticGeometryRequest | None = None,
    source_x: np.ndarray | None = None,
    source_y: np.ndarray | None = None,
    receiver_x: np.ndarray | None = None,
    receiver_y: np.ndarray | None = None,
    source_elevation: np.ndarray | None = None,
    receiver_elevation: np.ndarray | None = None,
    coordinate_scalar: np.ndarray | None = None,
    elevation_scalar: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    source_depth: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    geom = geometry or _geometry()
    out = {
        geom.source_id_byte: np.arange(100, 100 + n_traces, dtype=np.int64),
        geom.receiver_id_byte: np.arange(200, 200 + n_traces, dtype=np.int64),
        geom.source_x_byte: (
            np.zeros(n_traces, dtype=np.float64)
            if source_x is None
            else np.asarray(source_x, dtype=np.float64)
        ),
        geom.source_y_byte: (
            np.zeros(n_traces, dtype=np.float64)
            if source_y is None
            else np.asarray(source_y, dtype=np.float64)
        ),
        geom.receiver_x_byte: (
            np.arange(1, n_traces + 1, dtype=np.float64) * 10.0
            if receiver_x is None
            else np.asarray(receiver_x, dtype=np.float64)
        ),
        geom.receiver_y_byte: (
            np.zeros(n_traces, dtype=np.float64)
            if receiver_y is None
            else np.asarray(receiver_y, dtype=np.float64)
        ),
        geom.source_elevation_byte: (
            np.full(n_traces, 100.0, dtype=np.float64)
            if source_elevation is None
            else np.asarray(source_elevation, dtype=np.float64)
        ),
        geom.receiver_elevation_byte: (
            np.full(n_traces, 90.0, dtype=np.float64)
            if receiver_elevation is None
            else np.asarray(receiver_elevation, dtype=np.float64)
        ),
        geom.coordinate_scalar_byte: (
            np.ones(n_traces, dtype=np.int64)
            if coordinate_scalar is None
            else np.asarray(coordinate_scalar)
        ),
        geom.elevation_scalar_byte: (
            np.ones(n_traces, dtype=np.int64)
            if elevation_scalar is None
            else np.asarray(elevation_scalar)
        ),
        11: (
            np.arange(1, n_traces + 1, dtype=np.float64) * 10.0
            if offset is None
            else np.asarray(offset, dtype=np.float64)
        ),
    }
    if geom.source_depth_byte is not None:
        out[geom.source_depth_byte] = (
            np.zeros(n_traces, dtype=np.float64)
            if source_depth is None
            else np.asarray(source_depth, dtype=np.float64)
        )
    return out


def _build(
    picks: np.ndarray | None = None,
    *,
    headers: dict[int, np.ndarray] | None = None,
    geometry: RefractionStaticGeometryRequest | None = None,
    moveout: RefractionStaticMoveoutRequest | None = None,
    linkage: RefractionStaticLinkageRequest | None = None,
    linkage_artifact: dict[str, object] | None = None,
    sorted_trace_index: np.ndarray | None = None,
    job_dir: Path | None = None,
    source_depth_mode: str = 'none',
    source_depth_invalidates_source_geometry: bool = True,
):
    geom = geometry or _geometry()
    data = headers if headers is not None else _headers(geometry=geom)
    return build_refraction_static_input_model_from_arrays(
        file_id='line-a',
        pick_time_s_sorted=(
            np.asarray([0.010, 0.020, np.nan, 0.040], dtype=np.float64)
            if picks is None
            else np.asarray(picks, dtype=np.float64)
        ),
        trace_headers_sorted=data,
        geometry=geom,
        linkage=_linkage('none') if linkage is None else linkage,
        moveout=_moveout() if moveout is None else moveout,
        sorted_trace_index=sorted_trace_index,
        n_samples=100,
        dt=0.001,
        linkage_artifact=linkage_artifact,
        job_dir=job_dir,
        source_depth_mode=source_depth_mode,
        source_depth_byte=geom.source_depth_byte,
        source_depth_invalidates_source_geometry=(
            source_depth_invalidates_source_geometry
        ),
    )


def test_build_refraction_static_input_model_from_arrays_basic_sorted_contract() -> None:
    sorted_trace_index = np.asarray([2, 0, 3, 1], dtype=np.int64)

    model = _build(sorted_trace_index=sorted_trace_index)

    np.testing.assert_array_equal(model.sorted_trace_index, sorted_trace_index)
    np.testing.assert_allclose(
        model.pick_time_s_sorted,
        np.asarray([0.010, 0.020, np.nan, 0.040]),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        model.valid_pick_mask_sorted,
        np.asarray([True, True, False, True]),
    )
    assert model.valid_observation_mask_sorted.tolist() == [True, True, False, True]
    np.testing.assert_allclose(model.geometry_distance_m_sorted, [10.0, 20.0, 30.0, 40.0])
    assert model.source_depth_m_sorted is None
    assert model.qc['n_traces'] == 4
    assert model.qc['n_valid_picks'] == 3
    assert model.qc['n_valid_observations'] == 3
    assert model.qc['rejection_counts']['missing_pick'] == 1


def test_pick_validation_masks_nan_inf_negative_and_out_of_range() -> None:
    picks = np.asarray([np.nan, np.inf, -1.0, 0.200, 0.005], dtype=np.float64)

    model = _build(
        picks,
        headers=_headers(5),
    )

    assert model.valid_pick_mask_sorted.tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert model.rejection_reason_sorted.tolist() == [
        'missing_pick',
        'nonfinite_pick',
        'negative_pick',
        'outside_trace_time_range',
        'ok',
    ]
    assert model.qc['n_valid_observations'] == 1


def test_pick_array_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match='shape mismatch'):
        _build(np.asarray([0.010, 0.020]), headers=_headers(4))


@pytest.mark.parametrize(
    ('scalar', 'expected_distance'),
    [
        (np.asarray([10], dtype=np.int64), 20.0),
        (np.asarray([-10], dtype=np.int64), 0.2),
        (np.asarray([0], dtype=np.int64), 2.0),
    ],
)
def test_coordinate_scalar_convention(
    scalar: np.ndarray,
    expected_distance: float,
) -> None:
    headers = _headers(
        1,
        source_x=np.asarray([1.0]),
        receiver_x=np.asarray([3.0]),
        coordinate_scalar=scalar,
    )

    model = _build(np.asarray([0.005]), headers=headers)

    assert model.geometry_distance_m_sorted[0] == pytest.approx(expected_distance)


@pytest.mark.parametrize(
    ('scalar', 'expected_elevation'),
    [
        (np.asarray([10], dtype=np.int64), 1000.0),
        (np.asarray([-10], dtype=np.int64), 10.0),
        (np.asarray([0], dtype=np.int64), 100.0),
    ],
)
def test_elevation_scalar_convention(
    scalar: np.ndarray,
    expected_elevation: float,
) -> None:
    headers = _headers(1, elevation_scalar=scalar)

    model = _build(np.asarray([0.005]), headers=headers)

    assert model.source_elevation_m_sorted[0] == pytest.approx(expected_elevation)


def test_coordinate_and_elevation_units_convert_feet_to_meters() -> None:
    geom = _geometry(coordinate_unit='ft', elevation_unit='ft')
    headers = _headers(
        1,
        geometry=geom,
        source_x=np.asarray([0.0]),
        receiver_x=np.asarray([10.0]),
        source_elevation=np.asarray([100.0]),
    )

    model = _build(np.asarray([0.005]), headers=headers, geometry=geom)

    assert model.geometry_distance_m_sorted[0] == pytest.approx(3.048)
    assert model.source_elevation_m_sorted[0] == pytest.approx(30.48)


def test_source_depth_is_loaded_when_configured() -> None:
    geom = _geometry(source_depth_byte=12)
    headers = _headers(1, geometry=geom, source_depth=np.asarray([5.0]))

    model = _build(np.asarray([0.005]), headers=headers, geometry=geom)

    assert model.source_depth_m_sorted is not None
    assert model.source_depth_m_sorted[0] == pytest.approx(5.0)


def test_source_depth_resolution_does_not_filter_missing_depth_when_required() -> None:
    geom = _geometry(source_depth_byte=12)
    headers = _headers(
        2,
        geometry=geom,
        source_depth=np.asarray([np.nan, 4.0], dtype=np.float64),
    )

    model = _build(
        np.asarray([0.005, 0.006]),
        headers=headers,
        geometry=geom,
        source_depth_mode='weathering_velocity_time',
        source_depth_invalidates_source_geometry=False,
    )

    assert model.valid_observation_mask_sorted.tolist() == [True, True]
    assert model.source_depth_result is not None
    assert model.source_depth_result.source_depth_status.tolist() == [
        'missing_source_depth',
        'ok',
    ]
    assert model.qc['source_depth']['n_missing_source_depth'] == 1


def test_source_depth_uses_resolved_endpoint_ids_not_source_header_ids() -> None:
    geom = _geometry(source_depth_byte=12)
    headers = _headers(
        2,
        geometry=geom,
        source_x=np.asarray([0.0, 100.0], dtype=np.float64),
        receiver_x=np.asarray([10.0, 110.0], dtype=np.float64),
        source_depth=np.asarray([2.0, 4.0], dtype=np.float64),
    )
    headers[geom.source_id_byte] = np.asarray([42, 42], dtype=np.int64)

    model = _build(
        np.asarray([0.005, 0.006], dtype=np.float64),
        headers=headers,
        geometry=geom,
        source_depth_mode='weathering_velocity_time',
    )

    assert model.source_depth_result is not None
    np.testing.assert_array_equal(model.source_endpoint_id_sorted, [0, 1])
    np.testing.assert_array_equal(
        model.source_depth_result.source_endpoint_id,
        [0, 1],
    )
    np.testing.assert_array_equal(model.source_id_sorted, [42, 42])


def test_nonfinite_source_and_receiver_geometry_are_rejected_by_mask() -> None:
    headers = _headers(
        3,
        source_x=np.asarray([0.0, np.nan, 0.0]),
        receiver_x=np.asarray([10.0, 20.0, np.inf]),
    )

    model = _build(np.asarray([0.005, 0.006, 0.007]), headers=headers)

    assert model.valid_observation_mask_sorted.tolist() == [True, False, False]
    assert model.rejection_reason_sorted.tolist() == [
        'ok',
        'invalid_source_geometry',
        'invalid_receiver_geometry',
    ]


def test_offset_header_distance_uses_absolute_header_value() -> None:
    moveout = _moveout(distance_source='offset_header', offset_byte=11)
    headers = _headers(1, offset=np.asarray([-35.0]))

    model = _build(np.asarray([0.005]), headers=headers, moveout=moveout)

    assert model.offset_m_sorted is not None
    assert model.offset_m_sorted[0] == pytest.approx(35.0)
    assert model.distance_m_sorted[0] == pytest.approx(35.0)


@pytest.mark.parametrize(
    ('coordinate_scalar', 'expected_offset_m'),
    [
        (np.asarray([10], dtype=np.int64), 350.0),
        (np.asarray([-10], dtype=np.int64), 3.5),
        (np.asarray([0], dtype=np.int64), 35.0),
    ],
)
def test_offset_header_distance_applies_coordinate_scalar(
    coordinate_scalar: np.ndarray,
    expected_offset_m: float,
) -> None:
    moveout = _moveout(distance_source='offset_header', offset_byte=11)
    headers = _headers(
        1,
        offset=np.asarray([-35.0]),
        coordinate_scalar=coordinate_scalar,
    )

    model = _build(np.asarray([0.005]), headers=headers, moveout=moveout)

    assert model.offset_m_sorted is not None
    assert model.offset_m_sorted[0] == pytest.approx(expected_offset_m)
    assert model.distance_m_sorted[0] == pytest.approx(expected_offset_m)


def test_offset_header_distance_allows_missing_offsets_when_configured() -> None:
    moveout = _moveout(
        distance_source='offset_header',
        offset_byte=11,
        allow_missing_offset=True,
    )
    headers = _headers(3, offset=np.asarray([np.nan, np.inf, 30.0]))

    model = _build(
        np.asarray([0.005, 0.006, 0.007]),
        headers=headers,
        moveout=moveout,
    )

    assert model.offset_m_sorted is not None
    np.testing.assert_allclose(
        model.offset_m_sorted,
        np.asarray([np.nan, np.nan, 30.0]),
        equal_nan=True,
    )
    np.testing.assert_allclose(model.distance_m_sorted, [10.0, 20.0, 30.0])
    assert model.valid_observation_mask_sorted.tolist() == [True, True, True]
    assert model.rejection_reason_sorted.tolist() == ['ok', 'ok', 'ok']


def test_auto_distance_falls_back_to_offset_when_geometry_distance_is_invalid() -> None:
    moveout = _moveout(distance_source='auto', offset_byte=11)
    headers = _headers(
        1,
        source_x=np.asarray([10.0]),
        receiver_x=np.asarray([10.0]),
        offset=np.asarray([50.0]),
    )

    model = _build(np.asarray([0.005]), headers=headers, moveout=moveout)

    assert model.geometry_distance_m_sorted[0] == pytest.approx(0.0)
    assert model.distance_m_sorted[0] == pytest.approx(50.0)
    assert model.valid_observation_mask_sorted[0]


def test_auto_distance_with_no_offset_header_uses_geometry_only() -> None:
    moveout = _moveout(distance_source='auto', offset_byte=None)

    model = _build(np.asarray([0.005]), headers=_headers(1), moveout=moveout)

    assert model.offset_m_sorted is None
    assert model.distance_m_sorted[0] == pytest.approx(10.0)


def test_offset_mismatch_and_gates_reject_through_masks() -> None:
    moveout = _moveout(
        distance_source='geometry',
        offset_byte=11,
        min_offset_m=15.0,
        max_offset_m=35.0,
        max_geometry_offset_mismatch_m=2.0,
    )
    headers = _headers(
        5,
        receiver_x=np.asarray([10.0, 20.0, 30.0, 40.0, 25.0]),
        offset=np.asarray([10.0, 25.0, 60.0, 40.0, 25.0]),
    )

    model = _build(
        np.asarray([0.005, 0.006, 0.007, 0.008, 0.009]),
        headers=headers,
        moveout=moveout,
    )

    assert model.valid_observation_mask_sorted.tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert model.rejection_reason_sorted.tolist() == [
        'offset_gate',
        'offset_mismatch',
        'offset_mismatch',
        'offset_gate',
        'ok',
    ]
    assert model.qc['n_rejected_by_offset_gate'] == 2
    assert model.qc['n_rejected_by_offset_mismatch'] == 2


def test_linkage_none_creates_separate_source_and_receiver_nodes() -> None:
    headers = _headers(2, source_x=np.asarray([0.0, 100.0]))

    model = _build(np.asarray([0.005, 0.006]), headers=headers)

    assert model.qc['linkage_used'] is False
    assert model.qc['n_nodes'] == 4
    assert set(model.node_kind.tolist()) == {'source', 'receiver'}


def test_unlinked_endpoint_table_counts_valid_observations_only() -> None:
    model = _build(np.asarray([0.005, np.nan, 0.007]), headers=_headers(3))

    np.testing.assert_array_equal(model.endpoint_table.pick_count, [1, 0, 1, 1, 0, 1])


def test_endpoint_keys_include_header_ids_when_coordinates_match() -> None:
    headers = _headers(
        2,
        source_x=np.asarray([0.0, 0.0]),
        source_y=np.asarray([0.0, 0.0]),
        receiver_x=np.asarray([10.0, 10.0]),
        receiver_y=np.asarray([0.0, 0.0]),
    )

    model = _build(np.asarray([0.005, 0.006]), headers=headers)

    assert model.source_endpoint_key_sorted[0].startswith('source:100:')
    assert model.source_endpoint_key_sorted[1].startswith('source:101:')
    assert model.receiver_endpoint_key_sorted[0].startswith('receiver:200:')
    assert model.receiver_endpoint_key_sorted[1].startswith('receiver:201:')
    assert model.source_endpoint_key_sorted[0] != model.source_endpoint_key_sorted[1]
    assert model.receiver_endpoint_key_sorted[0] != model.receiver_endpoint_key_sorted[1]
    assert model.source_node_id_sorted[0] != model.source_node_id_sorted[1]
    assert model.receiver_node_id_sorted[0] != model.receiver_node_id_sorted[1]
    assert model.qc['n_nodes'] == 4


def test_required_linkage_uses_trace_node_ids() -> None:
    artifact = {
        'source_node_id_sorted': np.asarray([0, 0, 2], dtype=np.int64),
        'receiver_node_id_sorted': np.asarray([1, 1, 3], dtype=np.int64),
        'n_nodes': 4,
    }

    model = _build(
        np.asarray([0.005, 0.006, 0.007]),
        headers=_headers(3),
        linkage=_linkage('required'),
        linkage_artifact=artifact,
    )

    assert model.qc['linkage_requested'] == 'required'
    assert model.qc['linkage_used'] is True
    np.testing.assert_array_equal(model.source_node_id_sorted, [0, 0, 2])
    np.testing.assert_array_equal(model.receiver_node_id_sorted, [1, 1, 3])


def test_required_linkage_without_artifact_raises() -> None:
    with pytest.raises(ValueError, match='missing linkage artifact'):
        _build(
            np.asarray([0.005, 0.006]),
            headers=_headers(2),
            linkage=_linkage('required'),
        )


def test_required_linkage_rejects_rows_with_missing_node_sentinel() -> None:
    artifact = {
        'source_node_id_sorted': np.asarray([0, -1, 2], dtype=np.int64),
        'receiver_node_id_sorted': np.asarray([1, 1, 3], dtype=np.int64),
        'n_nodes': 4,
    }

    model = _build(
        np.asarray([0.005, 0.006, 0.007]),
        headers=_headers(3),
        linkage=_linkage('required'),
        linkage_artifact=artifact,
    )

    assert model.valid_observation_mask_sorted.tolist() == [True, False, True]
    assert model.rejection_reason_sorted.tolist() == ['ok', 'missing_linkage', 'ok']
    assert model.qc['n_rejected_by_missing_linkage'] == 1


def test_optional_linkage_without_artifact_falls_back_to_unlinked_nodes() -> None:
    linkage = RefractionStaticLinkageRequest.model_validate({'mode': 'optional'})

    model = _build(
        np.asarray([0.005, 0.006]),
        headers=_headers(2),
        linkage=linkage,
    )

    assert model.qc['linkage_requested'] == 'optional'
    assert model.qc['linkage_used'] is False
    assert model.qc['n_valid_observations'] == 2


def test_incompatible_linkage_coordinates_raise() -> None:
    artifact = {
        'source_node_id_sorted': np.asarray([0], dtype=np.int64),
        'receiver_node_id_sorted': np.asarray([1], dtype=np.int64),
        'n_nodes': 2,
        'source_x_m_sorted': np.asarray([999.0]),
        'source_y_m_sorted': np.asarray([0.0]),
        'receiver_x_m_sorted': np.asarray([10.0]),
        'receiver_y_m_sorted': np.asarray([0.0]),
    }

    with pytest.raises(ValueError, match='incompatible.*source geometry'):
        _build(
            np.asarray([0.005]),
            headers=_headers(1),
            linkage=_linkage('required'),
            linkage_artifact=artifact,
        )


def test_artifacts_are_written_when_job_dir_is_provided(tmp_path: Path) -> None:
    model = _build(job_dir=tmp_path)

    qc_path = tmp_path / REFRACTION_INPUT_QC_JSON_NAME
    preview_path = tmp_path / REFRACTION_INPUT_PREVIEW_CSV_NAME
    assert qc_path.is_file()
    assert preview_path.is_file()
    payload = json.loads(qc_path.read_text(encoding='utf-8'))
    assert payload['n_valid_observations'] == model.qc['n_valid_observations']
    assert 'sorted_trace_index,pick_time_s,valid_observation' in preview_path.read_text(
        encoding='utf-8'
    )


def test_no_valid_observations_raises() -> None:
    with pytest.raises(ValueError, match='No valid refraction observations remain'):
        _build(np.full(4, np.nan, dtype=np.float64))


def test_build_refraction_static_input_model_loads_npz_original_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sorted_to_original = np.asarray([2, 0, 3, 1], dtype=np.int64)
    headers = _headers(4)
    reader = _FakeReader(headers=headers, sorted_to_original=sorted_to_original)
    monkeypatch.setattr(inputs_module, 'get_reader', lambda *args, **kwargs: reader)

    state = AppState()
    job_dir = tmp_path / 'pick-job'
    job_dir.mkdir()
    state.jobs.create_batch_apply_job(
        'pick-job',
        file_id='line-a',
        key1_byte=189,
        key2_byte=193,
        artifacts_dir=str(job_dir),
    )
    np.savez(
        job_dir / 'predicted_picks_time_s.npz',
        picks_time_s=np.asarray([0.020, 0.040, 0.010, 0.030], dtype=np.float32),
        sorted_to_original=sorted_to_original,
        n_traces=np.asarray(4, dtype=np.int64),
        n_samples=np.asarray(100, dtype=np.int64),
        dt=np.asarray(0.001, dtype=np.float64),
    )
    req = RefractionStaticApplyRequest(
        file_id='line-a',
        pick_source={
            'kind': 'batch_predicted_npz',
            'job_id': 'pick-job',
        },
        geometry=_geometry(),
        linkage={'mode': 'none'},
        model=RefractionStaticModelRequest(weathering_velocity_m_s=800.0),
        datum={'mode': 'none'},
    )

    model = build_refraction_static_input_model(req=req, state=state)

    np.testing.assert_allclose(model.pick_time_s_sorted, [0.010, 0.020, 0.030, 0.040])
    np.testing.assert_array_equal(model.sorted_trace_index, sorted_to_original)
    assert model.metadata['pick_source_metadata']['accepted_pick_key'] == 'picks_time_s'


def test_build_refraction_static_input_model_rejects_missing_pick_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sorted_to_original = np.asarray([0, 1, 2, 3], dtype=np.int64)
    reader = _FakeReader(headers=_headers(4), sorted_to_original=sorted_to_original)
    monkeypatch.setattr(inputs_module, 'get_reader', lambda *args, **kwargs: reader)

    state = AppState()
    job_dir = tmp_path / 'pick-job'
    job_dir.mkdir()
    state.jobs.create_batch_apply_job(
        'pick-job',
        file_id='line-a',
        key1_byte=189,
        key2_byte=193,
        artifacts_dir=str(job_dir),
    )
    req = RefractionStaticApplyRequest(
        file_id='line-a',
        pick_source={
            'kind': 'batch_predicted_npz',
            'job_id': 'pick-job',
        },
        geometry=_geometry(),
        linkage={'mode': 'none'},
        model=RefractionStaticModelRequest(weathering_velocity_m_s=800.0),
        datum={'mode': 'none'},
    )

    with pytest.raises(ValueError, match='job artifact not found'):
        build_refraction_static_input_model(req=req, state=state)


def test_build_refraction_static_input_model_rejects_unsupported_pick_artifact_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sorted_to_original = np.asarray([0, 1, 2, 3], dtype=np.int64)
    reader = _FakeReader(headers=_headers(4), sorted_to_original=sorted_to_original)
    monkeypatch.setattr(inputs_module, 'get_reader', lambda *args, **kwargs: reader)

    state = AppState()
    job_dir = tmp_path / 'pick-job'
    job_dir.mkdir()
    state.jobs.create_batch_apply_job(
        'pick-job',
        file_id='line-a',
        key1_byte=189,
        key2_byte=193,
        artifacts_dir=str(job_dir),
    )
    np.savez(
        job_dir / 'predicted_picks_time_s.npz',
        travel_time_s=np.asarray([0.010, 0.020, 0.030, 0.040], dtype=np.float32),
    )
    req = RefractionStaticApplyRequest(
        file_id='line-a',
        pick_source={
            'kind': 'batch_predicted_npz',
            'job_id': 'pick-job',
        },
        geometry=_geometry(),
        linkage={'mode': 'none'},
        model=RefractionStaticModelRequest(weathering_velocity_m_s=800.0),
        datum={'mode': 'none'},
    )

    with pytest.raises(ValueError, match='unsupported pick artifact key'):
        build_refraction_static_input_model(req=req, state=state)


def test_build_refraction_static_input_model_rejects_cross_file_linkage_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sorted_to_original = np.asarray([0, 1, 2, 3], dtype=np.int64)
    reader = _FakeReader(headers=_headers(4), sorted_to_original=sorted_to_original)
    monkeypatch.setattr(inputs_module, 'get_reader', lambda *args, **kwargs: reader)

    state = AppState()
    pick_dir = tmp_path / 'pick-job'
    pick_dir.mkdir()
    state.jobs.create_batch_apply_job(
        'pick-job',
        file_id='line-a',
        key1_byte=189,
        key2_byte=193,
        artifacts_dir=str(pick_dir),
    )
    np.savez(
        pick_dir / 'predicted_picks_time_s.npz',
        pick_time_s=np.asarray([0.010, 0.020, 0.030, 0.040], dtype=np.float32),
    )
    linkage_dir = tmp_path / 'linkage-job'
    linkage_dir.mkdir()
    state.jobs.create_static_job(
        'linkage-job',
        file_id='line-b',
        key1_byte=189,
        key2_byte=193,
        statics_kind='geometry_linkage',
        artifacts_dir=str(linkage_dir),
    )
    req = RefractionStaticApplyRequest(
        file_id='line-a',
        pick_source={
            'kind': 'batch_predicted_npz',
            'job_id': 'pick-job',
        },
        geometry=_geometry(),
        linkage={'job_id': 'linkage-job'},
        model=RefractionStaticModelRequest(weathering_velocity_m_s=800.0),
        datum={'mode': 'none'},
    )

    with pytest.raises(ValueError, match='metadata mismatch: file_id'):
        build_refraction_static_input_model(req=req, state=state)


class _FakeReader:
    key1_byte = 189
    key2_byte = 193

    def __init__(self, *, headers: dict[int, np.ndarray], sorted_to_original: np.ndarray):
        self._headers = headers
        self._sorted_to_original = sorted_to_original
        self.traces = np.zeros((4, 100), dtype=np.float32)
        self.meta = {'dt': 0.001, 'n_traces': 4}

    def get_n_samples(self) -> int:
        return 100

    def get_sorted_to_original(self) -> np.ndarray:
        return self._sorted_to_original

    def ensure_header(self, byte: int) -> np.ndarray:
        return self._headers[byte]
