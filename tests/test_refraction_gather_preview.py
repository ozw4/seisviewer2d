from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.refraction_static_artifacts import (
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
)
from app.services.refraction_static_export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)

KEY1 = 189
KEY2 = 193
SECTION_KEY1 = 100
RAW_FILE_ID = 'raw-refraction-preview-file'
JOB_ID = 'refraction-gather-preview-job'
DT = 1.0
SORTED_TO_ORIGINAL = np.asarray([2, 0, 3, 1], dtype=np.int64)


@pytest.fixture()
def gather_preview_client(tmp_path: Path):
    state = app.state.sv
    with state.lock:
        state.jobs.clear()
        state.file_registry.clear()
        state.cached_readers.clear()

    traces = np.zeros((4, 8), dtype=np.float32)
    traces[:, 3] = 1.0
    raw_store = tmp_path / 'stores' / 'raw'
    _write_trace_store(raw_store, traces, sorted_to_original=SORTED_TO_ORIGINAL)
    artifacts_dir = tmp_path / 'refraction-job'
    artifacts_dir.mkdir()
    shifts = np.asarray([1.0, -1.0, 0.0, 2.0], dtype=np.float64)
    _write_refraction_artifacts(artifacts_dir, shifts=shifts)

    with state.lock:
        state.file_registry.update(RAW_FILE_ID, store_path=str(raw_store), dt=DT)
        state.jobs.create_static_job(
            JOB_ID,
            file_id=RAW_FILE_ID,
            key1_byte=KEY1,
            key2_byte=KEY2,
            statics_kind='refraction',
            artifacts_dir=str(artifacts_dir),
        )
        state.jobs.mark_done(JOB_ID, progress_1=True)

    with TestClient(app) as client:
        yield client, state, tmp_path, traces, shifts

    with state.lock:
        state.jobs.clear()
        state.file_registry.clear()
        state.cached_readers.clear()


def test_refraction_gather_preview_source_gather_raw_only(
    gather_preview_client,
):
    client, *_ = gather_preview_client

    response = client.post(
        '/statics/refraction/qc/gather-preview',
        json=_preview_request(gather_axis='source', endpoint_key='source:1001'),
    )

    assert response.status_code == 200
    body = response.json()
    assert body['sign_convention'] == REFRACTION_STATIC_REPO_SIGN_CONVENTION
    assert body['raw_window_ref']['status'] == 'ok'
    assert body['raw_window_ref']['endpoint'] == '/get_section_window_bin'
    assert body['raw_window_ref']['query']['file_id'] == RAW_FILE_ID
    assert body['raw_window_ref']['query']['key1'] == SECTION_KEY1
    assert body['corrected_window_ref']['status'] == 'not_registered'
    assert body['shape'] == [8, 4]
    assert len(body['raw_samples']) == 8
    assert len(body['raw_samples'][0]) == 4
    assert body['raw_samples'][3] == [1.0, 1.0, 1.0, 1.0]
    assert body['corrected_samples_source'] == 'raw_tracestore_shifted_on_the_fly'
    assert body['trace_indices'] == [2, 0, 3, 1]
    assert body['x_indices'] == [0, 1, 2, 3]


def test_refraction_gather_preview_raw_and_corrected(
    gather_preview_client,
):
    client, *_ = gather_preview_client

    response = client.post(
        '/statics/refraction/qc/gather-preview',
        json=_preview_request(gather_axis='source', endpoint_key='source:1001'),
    )

    assert response.status_code == 200
    body = response.json()
    corrected_ref = body['corrected_window_ref']
    assert corrected_ref['status'] == 'not_registered'
    assert corrected_ref['source'] == 'corrected_tracestore'
    assert body['corrected_samples_source'] == 'raw_tracestore_shifted_on_the_fly'
    corrected = body['corrected_samples']
    assert corrected[4][0] == pytest.approx(1.0)
    assert corrected[2][1] == pytest.approx(1.0)
    assert corrected[3][2] == pytest.approx(1.0)
    assert corrected[5][3] == pytest.approx(1.0)
    assert body['final_trace_shift_s'] == [1.0, -1.0, 0.0, 2.0]
    assert body['corrected_observed_pick_time_s'] == pytest.approx(
        [1.2, -0.7, 0.4, 2.5]
    )
    assert body['corrected_modeled_pick_time_s'] == pytest.approx(
        [1.19, -0.71, 0.39, 2.49]
    )


def test_refraction_gather_preview_uses_registered_corrected_store(
    gather_preview_client,
):
    client, state, tmp_path, traces, _shifts = gather_preview_client
    corrected_store = tmp_path / 'stores' / 'corrected'
    corrected_traces = traces * 2.0
    _write_trace_store(corrected_store, corrected_traces)
    with state.lock:
        state.file_registry.update(
            'corrected-refraction-preview-file',
            store_path=str(corrected_store),
            dt=DT,
        )
        state.jobs.set_static_corrected_file(
            JOB_ID,
            corrected_file_id='corrected-refraction-preview-file',
            corrected_store_path=str(corrected_store),
        )
        state.cached_readers.clear()

    response = client.post(
        '/statics/refraction/qc/gather-preview',
        json=_preview_request(gather_axis='source', endpoint_key='source:1001'),
    )

    assert response.status_code == 200
    body = response.json()
    corrected_ref = body['corrected_window_ref']
    assert corrected_ref['status'] == 'ok'
    assert corrected_ref['source'] == 'corrected_tracestore'
    assert corrected_ref['endpoint'] == '/get_section_window_bin'
    assert corrected_ref['query']['file_id'] == 'corrected-refraction-preview-file'
    assert body['corrected_samples_source'] == 'corrected_tracestore'
    assert body['corrected_samples'][3] == [2.0, 2.0, 2.0, 2.0]
    assert body['corrected_observed_pick_time_s'][3] == 2.5


def test_refraction_gather_preview_falls_back_for_unreadable_corrected_store(
    gather_preview_client,
):
    client, state, tmp_path, *_ = gather_preview_client
    corrected_store = tmp_path / 'stores' / 'unreadable-corrected'
    corrected_store.mkdir()
    with state.lock:
        state.file_registry.update(
            'corrected-refraction-preview-file',
            store_path=str(corrected_store),
            dt=DT,
        )
        state.jobs.set_static_corrected_file(
            JOB_ID,
            corrected_file_id='corrected-refraction-preview-file',
            corrected_store_path=str(corrected_store),
        )
        state.cached_readers.clear()

    response = client.post(
        '/statics/refraction/qc/gather-preview',
        json=_preview_request(gather_axis='source', endpoint_key='source:1001'),
    )

    assert response.status_code == 200
    body = response.json()
    corrected_ref = body['corrected_window_ref']
    assert corrected_ref['status'] == 'unavailable'
    assert 'Registered corrected TraceStore' in corrected_ref['message']
    assert 'on-the-fly shifted preview samples' in corrected_ref['message']
    assert body['corrected_samples_source'] == 'raw_tracestore_shifted_on_the_fly'
    assert body['corrected_samples'][4][0] == pytest.approx(1.0)


def test_refraction_gather_preview_caps_trace_count(gather_preview_client):
    client, *_ = gather_preview_client

    request = _preview_request(gather_axis='section')
    request['max_traces'] = 2
    response = client.post('/statics/refraction/qc/gather-preview', json=request)

    assert response.status_code == 200
    body = response.json()
    assert body['window']['requested_trace_count'] == 4
    assert body['window']['returned_trace_count'] == 2
    assert body['window']['trace_capped'] is True
    assert body['raw_window_ref']['query']['step_x'] == 2
    assert body['trace_indices'] == [2, 3]
    assert body['x_indices'] == [0, 2]
    assert len(body['raw_samples'][0]) == 2


def test_refraction_gather_preview_selects_receiver_endpoint_without_x_window(
    gather_preview_client,
):
    client, *_ = gather_preview_client

    request = _preview_request(gather_axis='receiver', endpoint_key='receiver:2002')
    del request['key1']
    del request['x0']
    del request['x1']
    response = client.post('/statics/refraction/qc/gather-preview', json=request)

    assert response.status_code == 200
    body = response.json()
    assert body['window']['key1'] == SECTION_KEY1
    assert body['trace_indices'] == [0]
    assert body['x_indices'] == [1]
    assert body['source_endpoint_key'] == ['source:1001']
    assert body['receiver_endpoint_key'] == ['receiver:2002']
    assert len(body['raw_samples']) == 8
    assert len(body['raw_samples'][0]) == 1


def test_refraction_gather_preview_accepts_time_range(gather_preview_client):
    client, *_ = gather_preview_client

    request = _preview_request(gather_axis='section')
    del request['y0']
    del request['y1']
    request['time_start_s'] = 2.0
    request['time_end_s'] = 4.0
    response = client.post('/statics/refraction/qc/gather-preview', json=request)

    assert response.status_code == 200
    body = response.json()
    assert body['window']['y0'] == 2
    assert body['window']['y1'] == 4
    assert body['shape'] == [3, 4]
    assert body['raw_samples'][1] == [1.0, 1.0, 1.0, 1.0]


def test_refraction_gather_preview_rejects_invalid_sample_range(
    gather_preview_client,
):
    client, *_ = gather_preview_client

    request = _preview_request(gather_axis='section')
    request['y0'] = 4
    request['y1'] = 3
    response = client.post('/statics/refraction/qc/gather-preview', json=request)

    assert response.status_code == 422


def test_refraction_gather_preview_overlay_contains_observed_and_modeled_first_breaks(
    gather_preview_client,
):
    client, *_ = gather_preview_client

    request = _preview_request(gather_axis='source', endpoint_key='source:1001')
    request['reduction_velocity_m_s'] = 1000.0
    response = client.post('/statics/refraction/qc/gather-preview', json=request)

    assert response.status_code == 200
    body = response.json()
    assert body['overlay_status']['first_break_fit'] == 'available'
    assert body['observed_pick_time_s'] == [0.2, 0.3, 0.4, 0.5]
    assert body['modeled_pick_time_s'] == [0.19, 0.29, 0.39, 0.49]
    assert body['residual_s'] == pytest.approx([0.01, 0.01, 0.01, 0.01])
    assert body['offset_m'] == [100.0, 200.0, 300.0, 400.0]
    assert body['reduced_observed_time_s'] == pytest.approx(
        [0.1, 0.1, 0.1, 0.1]
    )
    assert body['reduced_modeled_time_s'] == pytest.approx(
        [0.09, 0.09, 0.09, 0.09]
    )


def test_refraction_gather_preview_rejects_missing_gather_target(
    gather_preview_client,
):
    client, *_ = gather_preview_client

    response = client.post(
        '/statics/refraction/qc/gather-preview',
        json=_preview_request(gather_axis='receiver', endpoint_key='receiver:2999'),
    )

    assert response.status_code == 404
    assert 'receiver:2999' in response.json()['detail']


def _preview_request(
    *,
    gather_axis: str,
    endpoint_key: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        'job_id': JOB_ID,
        'file_id': RAW_FILE_ID,
        'key1': SECTION_KEY1,
        'key1_byte': KEY1,
        'key2_byte': KEY2,
        'gather_axis': gather_axis,
        'x0': 0,
        'x1': 3,
        'y0': 0,
        'y1': 7,
        'step_x': 1,
        'step_y': 1,
        'scaling': 'amax',
    }
    if endpoint_key is not None:
        payload['endpoint_key'] = endpoint_key
    return payload


def _write_trace_store(
    store: Path,
    traces: np.ndarray,
    *,
    sorted_to_original: np.ndarray = SORTED_TO_ORIGINAL,
) -> None:
    store.mkdir(parents=True, exist_ok=True)
    traces = np.asarray(traces, dtype=np.float32)
    n_traces, n_samples = traces.shape
    np.save(store / 'traces.npy', traces)
    np.savez(
        store / 'index.npz',
        key1_values=np.asarray([SECTION_KEY1], dtype=np.int64),
        key1_offsets=np.asarray([0], dtype=np.int64),
        key1_counts=np.asarray([n_traces], dtype=np.int64),
        sorted_to_original=np.asarray(sorted_to_original, dtype=np.int64),
    )
    np.save(
        store / f'headers_byte_{KEY1}.npy',
        np.full(n_traces, SECTION_KEY1, dtype=np.int32),
    )
    np.save(store / f'headers_byte_{KEY2}.npy', np.arange(n_traces, dtype=np.int32))
    meta = {
        'schema_version': 1,
        'dtype': 'float32',
        'n_traces': int(n_traces),
        'n_samples': int(n_samples),
        'key_bytes': {'key1': KEY1, 'key2': KEY2},
        'sorted_by': ['key1', 'key2'],
        'dt': DT,
        'original_segy_path': '/data/refraction-preview.sgy',
        'source_sha256': 'source-sha',
        'original_name': 'refraction-preview.sgy',
    }
    (store / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')


def _write_refraction_artifacts(artifacts_dir: Path, *, shifts: np.ndarray) -> None:
    (artifacts_dir / REFRACTION_STATIC_QC_JSON_NAME).write_text(
        json.dumps(
            {'sign_convention': {'trace_shift_s': REFRACTION_STATIC_REPO_SIGN_CONVENTION}}
        ),
        encoding='utf-8',
    )
    trace_index = np.arange(4, dtype=np.int64)
    source_keys = np.asarray(['source:1001'] * 4)
    receiver_keys = np.asarray([f'receiver:{2001 + idx}' for idx in range(4)])
    observed = np.asarray([0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    modeled = np.asarray([0.19, 0.29, 0.39, 0.49], dtype=np.float64)
    np.savez(
        artifacts_dir / REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
        observation_index=trace_index,
        sorted_trace_index=trace_index,
        trace_index_sorted=trace_index,
        source_endpoint_key=source_keys,
        receiver_endpoint_key=receiver_keys,
        offset_m=np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float64),
        inline_m=np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        crossline_m=np.zeros(4, dtype=np.float64),
        midpoint_x_m=np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        midpoint_y_m=np.zeros(4, dtype=np.float64),
        observed_first_break_time_s=observed,
        modeled_first_break_time_s=modeled,
        residual_time_s=observed - modeled,
        layer_kind=np.asarray(['v2_t1'] * 4),
        status=np.asarray(['ok'] * 4),
        rejection_reason=np.asarray(['none'] * 4),
    )
    np.savez(
        artifacts_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        final_trace_shift_s_sorted=np.asarray(shifts, dtype=np.float64),
        refraction_trace_shift_s_sorted=np.asarray(shifts, dtype=np.float64),
        trace_static_valid_mask_sorted=np.ones(4, dtype=bool),
        trace_static_status_sorted=np.asarray(['ok'] * 4),
        sorted_trace_index=SORTED_TO_ORIGINAL,
    )
