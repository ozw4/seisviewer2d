from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import app.statics.refraction.application.bedrock as bedrock_module
from app.api.schemas import RefractionStaticModelRequest, RefractionStaticSolverRequest
from app.statics.refraction.application.bedrock import (
    REFRACTION_BEDROCK_QC_JSON_NAME,
    REFRACTION_BEDROCK_RESIDUALS_CSV_NAME,
    RefractionBedrockSlownessError,
    estimate_global_bedrock_slowness_from_first_breaks,
    estimate_global_bedrock_slowness_from_input_model,
)
from app.statics.refraction.domain.types import (
    RefractionEndpointTable,
    RefractionStaticInputModel,
)
from app.statics.refraction.domain.solver import RefractionStaticSolverError

NODE_ID = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
TRUE_HALF_INTERCEPT_S = np.asarray([0.010, 0.012, 0.015, 0.018, 0.020])
TRUE_BEDROCK_VELOCITY_M_S = 2500.0
TRUE_BEDROCK_SLOWNESS_S_PER_M = 1.0 / TRUE_BEDROCK_VELOCITY_M_S
WEATHERING_VELOCITY_M_S = 800.0
ROW_SOURCE_NODE_ID = np.asarray(
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 0, 1, 2, 3, 4],
    dtype=np.int64,
)
ROW_RECEIVER_NODE_ID = np.asarray(
    [1, 2, 3, 4, 2, 3, 4, 3, 4, 4, 0, 1, 2, 3, 4],
    dtype=np.int64,
)
ROW_DISTANCE_M = np.asarray(
    [
        120.0,
        230.0,
        360.0,
        520.0,
        140.0,
        270.0,
        410.0,
        160.0,
        310.0,
        190.0,
        60.0,
        75.0,
        90.0,
        105.0,
        130.0,
    ],
    dtype=np.float64,
)


def _model(**overrides: Any) -> RefractionStaticModelRequest:
    payload = {
        'method': 'gli_variable_thickness',
        'weathering_velocity_m_s': WEATHERING_VELOCITY_M_S,
        'bedrock_velocity_mode': 'solve_global',
        'bedrock_velocity_m_s': None,
        'initial_bedrock_velocity_m_s': TRUE_BEDROCK_VELOCITY_M_S,
        'min_bedrock_velocity_m_s': 1200.0,
        'max_bedrock_velocity_m_s': 6000.0,
        'max_weathering_thickness_m': None,
    }
    payload.update(overrides)
    return RefractionStaticModelRequest.model_validate(payload)


def _invalid_model(**overrides: Any) -> RefractionStaticModelRequest:
    payload = {
        'method': 'gli_variable_thickness',
        'weathering_velocity_m_s': WEATHERING_VELOCITY_M_S,
        'bedrock_velocity_mode': 'solve_global',
        'bedrock_velocity_m_s': None,
        'initial_bedrock_velocity_m_s': TRUE_BEDROCK_VELOCITY_M_S,
        'min_bedrock_velocity_m_s': 1200.0,
        'max_bedrock_velocity_m_s': 6000.0,
        'max_weathering_thickness_m': None,
    }
    payload.update(overrides)
    return RefractionStaticModelRequest.model_construct(**payload)


def _solver(**overrides: Any) -> RefractionStaticSolverRequest:
    payload = {
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
    }
    robust_overrides = overrides.pop('robust', None)
    payload.update(overrides)
    if robust_overrides is not None:
        payload['robust'].update(robust_overrides)
    return RefractionStaticSolverRequest.model_validate(payload)


def _pick_times(
    *,
    velocity_m_s: float = TRUE_BEDROCK_VELOCITY_M_S,
    noise_s: np.ndarray | None = None,
) -> np.ndarray:
    picks = (
        TRUE_HALF_INTERCEPT_S[ROW_SOURCE_NODE_ID]
        + TRUE_HALF_INTERCEPT_S[ROW_RECEIVER_NODE_ID]
        + ROW_DISTANCE_M / float(velocity_m_s)
    )
    if noise_s is not None:
        picks = picks + np.asarray(noise_s, dtype=np.float64)
    return np.ascontiguousarray(picks, dtype=np.float64)


def _endpoint_table(node_id: np.ndarray) -> RefractionEndpointTable:
    n_nodes = int(node_id.shape[0])
    return RefractionEndpointTable(
        node_id=np.ascontiguousarray(node_id, dtype=np.int64),
        endpoint_id=np.ascontiguousarray(node_id, dtype=np.int64),
        x_m=np.arange(n_nodes, dtype=np.float64),
        y_m=np.zeros(n_nodes, dtype=np.float64),
        elevation_m=np.zeros(n_nodes, dtype=np.float64),
        kind=np.full(n_nodes, 'linked', dtype='<U16'),
        pick_count=np.zeros(n_nodes, dtype=np.int64),
    )


def _input_model(
    *,
    pick_time_s_sorted: np.ndarray | None = None,
    valid_observation_mask_sorted: np.ndarray | None = None,
    source_node_id_sorted: np.ndarray | None = None,
    receiver_node_id_sorted: np.ndarray | None = None,
    distance_m_sorted: np.ndarray | None = None,
    node_id: np.ndarray = NODE_ID,
) -> RefractionStaticInputModel:
    picks = _pick_times() if pick_time_s_sorted is None else pick_time_s_sorted
    source = (
        ROW_SOURCE_NODE_ID
        if source_node_id_sorted is None
        else np.asarray(source_node_id_sorted, dtype=np.int64)
    )
    receiver = (
        ROW_RECEIVER_NODE_ID
        if receiver_node_id_sorted is None
        else np.asarray(receiver_node_id_sorted, dtype=np.int64)
    )
    distance = (
        ROW_DISTANCE_M
        if distance_m_sorted is None
        else np.asarray(distance_m_sorted, dtype=np.float64)
    )
    n_traces = int(np.asarray(picks).shape[0])
    valid = (
        np.ones(n_traces, dtype=bool)
        if valid_observation_mask_sorted is None
        else np.asarray(valid_observation_mask_sorted, dtype=bool)
    )
    trace_index = np.arange(n_traces, dtype=np.int64)
    zeros = np.zeros(n_traces, dtype=np.float64)
    return RefractionStaticInputModel(
        file_id='synthetic-file-id',
        n_traces=n_traces,
        sorted_trace_index=trace_index,
        pick_time_s_sorted=np.ascontiguousarray(picks, dtype=np.float64),
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=np.ascontiguousarray(valid, dtype=bool),
        source_id_sorted=source.copy(),
        receiver_id_sorted=receiver.copy(),
        source_x_m_sorted=zeros.copy(),
        source_y_m_sorted=zeros.copy(),
        receiver_x_m_sorted=distance.copy(),
        receiver_y_m_sorted=zeros.copy(),
        source_elevation_m_sorted=zeros.copy(),
        receiver_elevation_m_sorted=zeros.copy(),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=distance.copy(),
        offset_m_sorted=None,
        distance_m_sorted=np.ascontiguousarray(distance, dtype=np.float64),
        source_endpoint_key_sorted=np.asarray(
            [f's:{value}' for value in source],
            dtype='<U32',
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'r:{value}' for value in receiver],
            dtype='<U32',
        ),
        source_node_id_sorted=np.ascontiguousarray(source, dtype=np.int64),
        receiver_node_id_sorted=np.ascontiguousarray(receiver, dtype=np.int64),
        node_x_m=np.asarray(node_id, dtype=np.float64),
        node_y_m=np.zeros(int(node_id.shape[0]), dtype=np.float64),
        node_elevation_m=np.zeros(int(node_id.shape[0]), dtype=np.float64),
        node_kind=np.full(int(node_id.shape[0]), 'linked', dtype='<U16'),
        rejection_reason_sorted=np.full(n_traces, 'ok', dtype='<U32'),
        qc={},
        endpoint_table=_endpoint_table(np.asarray(node_id, dtype=np.int64)),
        metadata={},
    )


def test_public_apis_are_importable() -> None:
    assert callable(estimate_global_bedrock_slowness_from_first_breaks)
    assert callable(estimate_global_bedrock_slowness_from_input_model)


def test_estimate_global_bedrock_slowness_recovers_noiseless_solution() -> None:
    result = estimate_global_bedrock_slowness_from_input_model(
        input_model=_input_model(),
        model=_model(),
        solver=_solver(),
    )

    assert result.bedrock_velocity_mode == 'solve_global'
    assert result.bedrock_slowness_s_per_m == pytest.approx(
        TRUE_BEDROCK_SLOWNESS_S_PER_M,
        abs=1.0e-11,
    )
    assert result.bedrock_velocity_m_s == pytest.approx(
        TRUE_BEDROCK_VELOCITY_M_S,
        rel=1.0e-7,
    )
    assert result.bedrock_velocity_status == 'solved'
    np.testing.assert_array_equal(result.active_node_id, NODE_ID)
    np.testing.assert_allclose(
        result.active_node_half_intercept_time_s,
        TRUE_HALF_INTERCEPT_S,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(result.modeled_pick_time_s, result.observed_pick_time_s)
    np.testing.assert_allclose(result.residual_time_s, 0.0, atol=1.0e-9)
    np.testing.assert_array_equal(result.row_trace_index_sorted, np.arange(15))
    assert result.used_row_mask.all()
    assert not result.rejected_by_robust_mask.any()
    assert result.input_model is None
    assert result.design_matrix is None
    assert result.qc['bedrock_velocity_status'] == 'solved'
    assert result.qc['n_valid_observations'] == 15
    assert result.qc['n_active_nodes'] == 5
    assert result.qc['residual_rms_ms'] == pytest.approx(0.0, abs=1.0e-6)
    assert result.qc['robust_enabled'] is False
    json.dumps(result.qc, allow_nan=False)


def test_estimate_global_bedrock_slowness_recovers_noisy_solution() -> None:
    noise = np.linspace(-0.00035, 0.00035, ROW_DISTANCE_M.shape[0])

    result = estimate_global_bedrock_slowness_from_input_model(
        input_model=_input_model(pick_time_s_sorted=_pick_times(noise_s=noise)),
        model=_model(),
        solver=_solver(),
    )

    assert result.bedrock_velocity_m_s == pytest.approx(
        TRUE_BEDROCK_VELOCITY_M_S,
        rel=0.03,
    )
    assert result.qc['residual_rms_ms'] > 0.0


def test_estimate_global_bedrock_slowness_classifies_min_velocity_clip() -> None:
    result = estimate_global_bedrock_slowness_from_input_model(
        input_model=_input_model(
            pick_time_s_sorted=_pick_times(velocity_m_s=900.0),
        ),
        model=_model(
            weathering_velocity_m_s=500.0,
            min_bedrock_velocity_m_s=1200.0,
            max_bedrock_velocity_m_s=6000.0,
        ),
        solver=_solver(),
    )

    assert result.bedrock_velocity_m_s == pytest.approx(1200.0)
    assert result.bedrock_velocity_status == 'clipped_min_velocity'
    assert result.qc['bedrock_slowness_at_upper_bound'] is True


def test_estimate_global_bedrock_slowness_classifies_max_velocity_clip() -> None:
    result = estimate_global_bedrock_slowness_from_input_model(
        input_model=_input_model(
            pick_time_s_sorted=_pick_times(velocity_m_s=9000.0),
        ),
        model=_model(
            min_bedrock_velocity_m_s=1200.0,
            max_bedrock_velocity_m_s=6000.0,
        ),
        solver=_solver(),
    )

    assert result.bedrock_velocity_m_s == pytest.approx(6000.0)
    assert result.bedrock_velocity_status == 'clipped_max_velocity'
    assert result.qc['bedrock_slowness_at_lower_bound'] is True


def test_robust_disabled_uses_all_rows_with_bad_pick() -> None:
    picks = _pick_times()
    picks[0] += 0.150

    result = estimate_global_bedrock_slowness_from_input_model(
        input_model=_input_model(pick_time_s_sorted=picks),
        model=_model(),
        solver=_solver(robust={'enabled': False}),
    )

    assert result.used_row_mask.all()
    assert not result.rejected_by_robust_mask.any()
    assert result.qc['n_rejected_by_robust'] == 0


@pytest.mark.parametrize('method', ['mad', 'sigma'])
def test_robust_rejects_large_bad_pick(method: str) -> None:
    picks = _pick_times()
    picks[0] += 0.150

    result = estimate_global_bedrock_slowness_from_input_model(
        input_model=_input_model(pick_time_s_sorted=picks),
        model=_model(),
        solver=_solver(
            robust={
                'enabled': True,
                'method': method,
                'threshold': 2.5,
            },
        ),
    )

    assert result.rejected_by_robust_mask[0]
    assert not result.used_row_mask[0]
    assert result.qc['n_rejected_by_robust'] >= 1
    assert result.qc['robust_method'] == method


def test_debug_artifacts_write_qc_and_residuals(tmp_path: Path) -> None:
    picks = _pick_times()
    picks[0] += 0.150

    result = estimate_global_bedrock_slowness_from_input_model(
        input_model=_input_model(pick_time_s_sorted=picks),
        model=_model(),
        solver=_solver(
            robust={
                'enabled': True,
                'method': 'mad',
                'threshold': 2.5,
            },
        ),
        job_dir=tmp_path,
    )

    qc_path = tmp_path / REFRACTION_BEDROCK_QC_JSON_NAME
    csv_path = tmp_path / REFRACTION_BEDROCK_RESIDUALS_CSV_NAME
    assert qc_path.is_file()
    assert csv_path.is_file()
    qc = json.loads(qc_path.read_text(encoding='utf-8'))
    assert qc['bedrock_velocity_m_s'] == pytest.approx(result.bedrock_velocity_m_s)
    with csv_path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result.observed_pick_time_s.shape[0]
    assert rows[0]['rejected_by_robust'] == 'True'


def test_estimate_rejects_fixed_global_mode() -> None:
    with pytest.raises(
        RefractionBedrockSlownessError,
        match="bedrock_velocity_mode='solve_global'",
    ):
        estimate_global_bedrock_slowness_from_input_model(
            input_model=_input_model(),
            model=_model(
                bedrock_velocity_mode='fixed_global',
                bedrock_velocity_m_s=TRUE_BEDROCK_VELOCITY_M_S,
            ),
            solver=_solver(),
        )


def test_estimate_rejects_no_valid_observations() -> None:
    with pytest.raises(RefractionBedrockSlownessError, match='No valid'):
        estimate_global_bedrock_slowness_from_input_model(
            input_model=_input_model(
                valid_observation_mask_sorted=np.zeros(
                    ROW_DISTANCE_M.shape[0],
                    dtype=bool,
                )
            ),
            model=_model(),
            solver=_solver(),
        )


def test_estimate_rejects_insufficient_active_nodes() -> None:
    n_rows = ROW_DISTANCE_M.shape[0]
    with pytest.raises(RefractionBedrockSlownessError, match='two active'):
        estimate_global_bedrock_slowness_from_input_model(
            input_model=_input_model(
                pick_time_s_sorted=np.linspace(0.05, 0.10, n_rows),
                source_node_id_sorted=np.zeros(n_rows, dtype=np.int64),
                receiver_node_id_sorted=np.zeros(n_rows, dtype=np.int64),
            ),
            model=_model(),
            solver=_solver(),
        )


def test_estimate_rejects_zero_distance_aperture() -> None:
    with pytest.raises(RefractionBedrockSlownessError, match='distance aperture'):
        estimate_global_bedrock_slowness_from_input_model(
            input_model=_input_model(
                pick_time_s_sorted=np.linspace(0.05, 0.10, ROW_DISTANCE_M.shape[0]),
                distance_m_sorted=np.full(ROW_DISTANCE_M.shape[0], 100.0),
            ),
            model=_model(),
            solver=_solver(),
        )


def test_estimate_rejects_zero_pick_time_aperture() -> None:
    with pytest.raises(RefractionBedrockSlownessError, match='pick time aperture'):
        estimate_global_bedrock_slowness_from_input_model(
            input_model=_input_model(
                pick_time_s_sorted=np.full(ROW_DISTANCE_M.shape[0], 0.10),
            ),
            model=_model(),
            solver=_solver(),
        )


def test_estimate_rejects_obvious_pick_distance_mismatch() -> None:
    with pytest.raises(RefractionBedrockSlownessError, match='median pick time'):
        estimate_global_bedrock_slowness_from_input_model(
            input_model=_input_model(
                pick_time_s_sorted=np.linspace(0.001, 0.002, ROW_DISTANCE_M.shape[0]),
            ),
            model=_model(),
            solver=_solver(),
        )


def test_lower_level_can_retain_debug_objects() -> None:
    input_model = _input_model()

    result = estimate_global_bedrock_slowness_from_input_model(
        input_model=input_model,
        model=_model(),
        solver=_solver(),
        include_debug_objects=True,
    )

    assert result.input_model is input_model
    assert result.design_matrix is not None


def test_estimate_wraps_high_level_input_builder_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.statics.refraction.application.input_model as inputs_module

    monkeypatch.setattr(
        inputs_module,
        'build_refraction_static_input_model',
        lambda **_kwargs: (_ for _ in ()).throw(ValueError('input builder failed')),
    )
    req = type(
        'Req',
        (),
        {
            'model': _model(),
            'solver': _solver(),
        },
    )()

    with pytest.raises(RefractionBedrockSlownessError, match='input builder failed'):
        estimate_global_bedrock_slowness_from_first_breaks(
            req=req,  # type: ignore[arg-type]
            state=object(),  # type: ignore[arg-type]
        )


def test_estimate_wraps_solver_result_conversion_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_conversion_error(**_kwargs: Any) -> None:
        raise RefractionStaticSolverError('adapter conversion failed')

    monkeypatch.setattr(
        bedrock_module,
        '_app_solver_result_from_core',
        raise_conversion_error,
    )

    with pytest.raises(
        RefractionBedrockSlownessError,
        match='adapter conversion failed',
    ) as exc_info:
        estimate_global_bedrock_slowness_from_input_model(
            input_model=_input_model(),
            model=_model(),
            solver=_solver(),
        )

    assert isinstance(exc_info.value.__cause__, RefractionStaticSolverError)


def test_estimate_rejects_invalid_model_contract() -> None:
    with pytest.raises(RefractionBedrockSlownessError, match='min_bedrock_velocity'):
        estimate_global_bedrock_slowness_from_input_model(
            input_model=_input_model(),
            model=_invalid_model(min_bedrock_velocity_m_s=6000.0),
            solver=_solver(),
        )
