from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import app.statics.refraction.application.input_model as inputs_module
from app.api.schemas import (
    RefractionStaticApplyRequest,
    RefractionStaticGeometryRequest,
    RefractionStaticModelRequest,
    RefractionStaticMoveoutRequest,
)
from app.core.state import AppState
from app.statics.refraction.application.input_model import build_refraction_static_input_model
from app.statics.refraction.application.preflight_diagnostics import (
    REFRACTION_STATIC_PREFLIGHT_OBSERVATIONS_CSV_NAME,
    REFRACTION_STATIC_PREFLIGHT_QC_JSON_NAME,
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


def _request(
    *,
    pick_source: dict[str, Any] | None = None,
    moveout: RefractionStaticMoveoutRequest | None = None,
) -> RefractionStaticApplyRequest:
    return RefractionStaticApplyRequest(
        file_id='line-a',
        key1_byte=189,
        key2_byte=193,
        pick_source=pick_source or {'kind': 'uploaded_npz'},
        geometry=_geometry(),
        linkage={'mode': 'none'},
        moveout=moveout or _moveout(),
        model=RefractionStaticModelRequest(weathering_velocity_m_s=800.0),
        datum={'mode': 'none'},
    )


def _headers(n_traces: int = 4) -> dict[int, np.ndarray]:
    geometry = _geometry()
    return {
        geometry.source_id_byte: np.arange(100, 100 + n_traces, dtype=np.int64),
        geometry.receiver_id_byte: np.arange(200, 200 + n_traces, dtype=np.int64),
        geometry.source_x_byte: np.zeros(n_traces, dtype=np.float64),
        geometry.source_y_byte: np.zeros(n_traces, dtype=np.float64),
        geometry.receiver_x_byte: (
            np.arange(1, n_traces + 1, dtype=np.float64) * 10.0
        ),
        geometry.receiver_y_byte: np.zeros(n_traces, dtype=np.float64),
        geometry.source_elevation_byte: np.full(n_traces, 100.0, dtype=np.float64),
        geometry.receiver_elevation_byte: np.full(n_traces, 90.0, dtype=np.float64),
        geometry.coordinate_scalar_byte: np.ones(n_traces, dtype=np.int64),
        geometry.elevation_scalar_byte: np.ones(n_traces, dtype=np.int64),
        11: np.arange(1, n_traces + 1, dtype=np.float64) * 10.0,
    }


class _FakeReader:
    def __init__(
        self,
        *,
        headers: dict[int, np.ndarray],
        sorted_to_original: np.ndarray,
    ):
        self._headers = headers
        self._sorted_to_original = sorted_to_original
        self.traces = np.zeros((int(sorted_to_original.shape[0]), 100), dtype=np.float32)
        self.meta = {'dt': 0.001, 'n_traces': int(sorted_to_original.shape[0])}

    def get_n_samples(self) -> int:
        return 100

    def get_sorted_to_original(self) -> np.ndarray:
        return self._sorted_to_original

    def ensure_header(self, byte: int) -> np.ndarray:
        return self._headers[byte]


def _install_reader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sorted_to_original: np.ndarray | None = None,
    headers: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    order = (
        np.asarray([2, 0, 3, 1], dtype=np.int64)
        if sorted_to_original is None
        else np.asarray(sorted_to_original, dtype=np.int64)
    )
    reader = _FakeReader(
        headers=headers or _headers(int(order.shape[0])),
        sorted_to_original=order,
    )
    monkeypatch.setattr(inputs_module, 'get_reader', lambda *args, **kwargs: reader)
    return order


def _read_qc(job_dir: Path) -> dict[str, Any]:
    return json.loads(
        (job_dir / REFRACTION_STATIC_PREFLIGHT_QC_JSON_NAME).read_text(
            encoding='utf-8'
        )
    )


def test_refraction_static_preflight_uploaded_npz_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sorted_to_original = _install_reader(monkeypatch)
    npz_path = tmp_path / 'uploaded_picks_time_s.npz'
    np.savez(
        npz_path,
        pick_time_s=np.asarray([0.020, 0.040, 0.010, 0.030], dtype=np.float32),
        sorted_to_original=sorted_to_original,
        n_traces=np.asarray(4, dtype=np.int64),
        n_samples=np.asarray(100, dtype=np.int64),
        dt=np.asarray(0.001, dtype=np.float64),
    )

    build_refraction_static_input_model(
        req=_request(),
        state=AppState(),
        job_dir=tmp_path / 'job',
        uploaded_pick_npz_path=npz_path,
        uploaded_pick_metadata={'original_filename': 'picks.npz'},
    )

    qc = _read_qc(tmp_path / 'job')
    assert qc['status'] == 'ok'
    assert qc['summary']['file_id'] == 'line-a'
    assert qc['summary']['key1_byte'] == 189
    assert qc['summary']['key2_byte'] == 193
    assert qc['summary']['pick_npz']['npz_keys'] == [
        'pick_time_s',
        'sorted_to_original',
        'n_traces',
        'n_samples',
        'dt',
    ]
    assert qc['summary']['pick_npz']['selected_pick_key'] == 'pick_time_s'
    assert qc['summary']['pick_npz']['pick_array_shape'] == [4]
    assert qc['summary']['observation_filters']['n_total_traces'] == 4
    assert qc['summary']['observation_filters']['n_used_for_inversion'] == 4


def test_refraction_static_preflight_rejects_pick_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_reader(monkeypatch)
    npz_path = tmp_path / 'uploaded_picks_time_s.npz'
    np.savez(npz_path, pick_time_s=np.asarray([0.010, 0.020, 0.030]))

    with pytest.raises(ValueError, match='preflight.*pick array length mismatch'):
        build_refraction_static_input_model(
            req=_request(),
            state=AppState(),
            job_dir=tmp_path / 'job',
            uploaded_pick_npz_path=npz_path,
        )

    qc = _read_qc(tmp_path / 'job')
    assert qc['status'] == 'error'
    assert 'pick array length mismatch' in qc['errors'][0]
    assert qc['summary']['pick_npz']['pick_array_shape'] == [3]


def test_refraction_static_preflight_rejects_bad_sorted_to_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_reader(monkeypatch)
    npz_path = tmp_path / 'uploaded_picks_time_s.npz'
    np.savez(
        npz_path,
        pick_time_s=np.asarray([0.010, 0.020, 0.030, 0.040]),
        sorted_to_original=np.asarray([0, 0, 2, 3], dtype=np.int64),
    )

    with pytest.raises(ValueError, match='preflight.*sorted_to_original'):
        build_refraction_static_input_model(
            req=_request(),
            state=AppState(),
            job_dir=tmp_path / 'job',
            uploaded_pick_npz_path=npz_path,
        )

    qc = _read_qc(tmp_path / 'job')
    assert qc['summary']['pick_npz']['has_sorted_to_original'] is True
    assert qc['summary']['pick_npz']['sorted_to_original_is_permutation'] is False


def test_refraction_static_preflight_counts_offset_gate_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sorted_to_original = _install_reader(monkeypatch)
    npz_path = tmp_path / 'uploaded_picks_time_s.npz'
    np.savez(
        npz_path,
        pick_time_s=np.asarray([0.020, 0.040, 0.010, 0.030], dtype=np.float64),
        sorted_to_original=sorted_to_original,
    )
    req = _request(moveout=_moveout(min_offset_m=15.0, max_offset_m=35.0))

    build_refraction_static_input_model(
        req=req,
        state=AppState(),
        job_dir=tmp_path / 'job',
        uploaded_pick_npz_path=npz_path,
    )

    qc = _read_qc(tmp_path / 'job')
    assert qc['summary']['observation_filters']['n_inside_offset_gate'] == 2
    assert qc['observation_reason_counts']['outside_offset_gate'] == 2
    assert qc['summary']['input_rejection_counts']['offset_gate'] == 2


def test_refraction_static_preflight_rejects_nonfinite_uploaded_npz_picks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sorted_to_original = _install_reader(monkeypatch)
    npz_path = tmp_path / 'uploaded_picks_time_s.npz'
    np.savez(
        npz_path,
        pick_time_s=np.asarray([0.020, np.nan, 0.010, np.inf], dtype=np.float64),
        sorted_to_original=sorted_to_original,
    )

    with pytest.raises(ValueError, match='preflight.*non-finite values'):
        build_refraction_static_input_model(
            req=_request(),
            state=AppState(),
            job_dir=tmp_path / 'job',
            uploaded_pick_npz_path=npz_path,
        )

    qc = _read_qc(tmp_path / 'job')
    assert qc['status'] == 'error'
    assert qc['summary']['pick_npz']['n_pick_values'] == 4
    assert qc['summary']['pick_npz']['n_finite_pick_values'] == 2
    assert qc['summary']['pick_npz']['n_nan_pick_values'] == 1
    assert qc['summary']['observation_filters']['n_finite_picks'] == 2
    assert 'non-finite values' in qc['errors'][0]


def test_refraction_static_preflight_skips_observation_csv_for_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sorted_to_original = _install_reader(monkeypatch)
    npz_path = tmp_path / 'uploaded_picks_time_s.npz'
    np.savez(
        npz_path,
        pick_time_s=np.asarray([0.020, 0.040, 0.010, 0.030], dtype=np.float64),
        sorted_to_original=sorted_to_original,
    )

    build_refraction_static_input_model(
        req=_request(),
        state=AppState(),
        job_dir=tmp_path / 'job',
        uploaded_pick_npz_path=npz_path,
    )

    qc = _read_qc(tmp_path / 'job')
    observations_path = (
        tmp_path / 'job' / REFRACTION_STATIC_PREFLIGHT_OBSERVATIONS_CSV_NAME
    )
    assert not observations_path.exists()
    assert qc['observations_csv_written'] is False
    assert qc['observations_csv_total_rows'] == 4
    assert qc['observations_csv_written_rows'] == 0
    assert qc['observations_csv_omitted_rows'] == 4
    assert qc['observations_csv_policy']['mode'] == 'success_summary_only'


def test_refraction_static_preflight_writes_bounded_rejected_rows_for_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        'SV_REFRACTION_STATIC_PREFLIGHT_OBSERVATIONS_CSV_MAX_ROWS',
        '2',
    )
    headers = _headers()
    headers[_geometry().source_x_byte] = np.asarray([np.nan, 0.0, np.nan, 0.0])
    sorted_to_original = _install_reader(monkeypatch, headers=headers)
    npz_path = tmp_path / 'uploaded_picks_time_s.npz'
    np.savez(
        npz_path,
        pick_time_s=np.asarray([0.020, 0.040, 0.010, 0.030], dtype=np.float64),
        sorted_to_original=sorted_to_original,
    )

    with pytest.raises(ValueError, match='No valid refraction observations remain'):
        build_refraction_static_input_model(
            req=_request(moveout=_moveout(max_offset_m=15.0)),
            state=AppState(),
            job_dir=tmp_path / 'job',
            uploaded_pick_npz_path=npz_path,
        )

    with (tmp_path / 'job' / REFRACTION_STATIC_PREFLIGHT_OBSERVATIONS_CSV_NAME).open(
        encoding='utf-8',
        newline='',
    ) as handle:
        rows = list(csv.DictReader(handle))
    qc = _read_qc(tmp_path / 'job')
    assert len(rows) == 2
    assert {row['rejection_reason'] for row in rows} == {
        'invalid_source_geometry',
        'offset_gate',
    }
    assert qc['observations_csv_written'] is True
    assert qc['observations_csv_total_rows'] == 4
    assert qc['observations_csv_written_rows'] == 2
    assert qc['observations_csv_omitted_rows'] == 2
    assert qc['observations_csv_policy']['max_rows'] == 2


def test_refraction_static_preflight_does_not_materialize_large_success_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 10_050
    sorted_to_original = _install_reader(
        monkeypatch,
        sorted_to_original=np.arange(n_traces, dtype=np.int64),
    )
    npz_path = tmp_path / 'uploaded_picks_time_s.npz'
    np.savez(
        npz_path,
        pick_time_s=np.linspace(0.020, 0.040, n_traces, dtype=np.float64),
        sorted_to_original=sorted_to_original,
    )

    build_refraction_static_input_model(
        req=_request(),
        state=AppState(),
        job_dir=tmp_path / 'job',
        uploaded_pick_npz_path=npz_path,
    )

    qc = _read_qc(tmp_path / 'job')
    observations_path = (
        tmp_path / 'job' / REFRACTION_STATIC_PREFLIGHT_OBSERVATIONS_CSV_NAME
    )
    assert not observations_path.exists()
    assert qc['observations_csv_written'] is False
    assert qc['observations_csv_total_rows'] == n_traces
    assert qc['observations_csv_written_rows'] == 0
    assert qc['observations_csv_omitted_rows'] == n_traces
