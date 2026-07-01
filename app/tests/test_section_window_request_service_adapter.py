from __future__ import annotations

import gzip
from collections.abc import Callable
from pathlib import Path

import msgpack
import numpy as np
import pytest

from app.services import section_service as svc
from app.services.section_window_request import SectionWindowRequest
from app.tests._stubs import make_stub_reader, write_baseline_raw


def _request(**overrides: object) -> SectionWindowRequest:
    values: dict[str, object] = {
        'file_id': 'f',
        'key1': 7,
        'key1_byte': 189,
        'key2_byte': 193,
        'normalization_file_id': None,
        'offset_byte': None,
        'x0': 0,
        'x1': 1,
        'y0': 0,
        'y1': 1,
        'step_x': 1,
        'step_y': 1,
        'transpose': False,
        'pipeline_key': None,
        'tap_label': None,
        'reference_pipeline_key': None,
        'reference_tap_label': None,
        'scaling': 'amax',
        'lmo_enabled': False,
        'lmo_velocity_mps': None,
        'lmo_offset_byte': 37,
        'lmo_offset_scale': 1.0,
        'lmo_offset_mode': 'absolute',
        'lmo_ref_mode': 'min',
        'lmo_ref_trace': None,
        'lmo_polarity': 1,
    }
    values.update(overrides)
    return SectionWindowRequest(**values)  # type: ignore[arg-type]


def _payload_obj(payload: bytes) -> dict[str, object]:
    return msgpack.unpackb(gzip.decompress(payload), raw=False)


def _assert_same_payload(left: bytes, right: bytes) -> None:
    assert _payload_obj(left) == _payload_obj(right)


def _payload_array(payload: bytes) -> np.ndarray:
    obj = _payload_obj(payload)
    return np.frombuffer(obj['data'], dtype=np.int8).reshape(obj['shape'])


def _build_direct_and_adapter(
    window_request: SectionWindowRequest,
    *,
    trace_stats_cache: object,
    reader_getter: Callable[[str, int, int], object],
    pipeline_section_getter: Callable[..., np.ndarray],
    store_dir_resolver: Callable[[str], str],
    dt_resolver: Callable[[str], float],
) -> tuple[bytes, bytes]:
    direct = svc.build_section_window_payload(
        **window_request.payload_kwargs(),
        trace_stats_cache=trace_stats_cache,
        reader_getter=reader_getter,
        pipeline_section_getter=pipeline_section_getter,
        store_dir_resolver=store_dir_resolver,
        dt_resolver=dt_resolver,
    )
    adapter = svc.build_section_window_payload_for_request(
        window_request,
        trace_stats_cache=trace_stats_cache,
        reader_getter=reader_getter,
        pipeline_section_getter=pipeline_section_getter,
        store_dir_resolver=store_dir_resolver,
        dt_resolver=dt_resolver,
    )
    return direct, adapter


@pytest.fixture(autouse=True)
def _fixed_quantization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('FIXED_INT8_SCALE', '1')


def test_adapter_returns_same_payload_as_existing_builder_for_raw_source(
    tmp_path: Path,
) -> None:
    write_baseline_raw(tmp_path, key1=7, n_traces=2)
    reader = make_stub_reader(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    request = _request(file_id='f')

    direct, adapter = _build_direct_and_adapter(
        request,
        trace_stats_cache={},
        reader_getter=lambda _fid, _kb1, _kb2: reader,
        pipeline_section_getter=lambda **_kwargs: None,
        store_dir_resolver=lambda _fid: str(tmp_path),
        dt_resolver=lambda _fid: 0.004,
    )

    _assert_same_payload(direct, adapter)


def test_adapter_preserves_raw_normalization_file_id_baseline(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / 'data'
    stats_dir = tmp_path / 'stats-a'
    write_baseline_raw(
        stats_dir,
        key1=7,
        section_mean=8.0,
        section_std=2.0,
        trace_means=[8.0, 12.0],
        trace_stds=[1.0, 1.0],
    )
    readers = {
        'B': make_stub_reader(
            np.array([[10.0, 12.0], [14.0, 16.0]], dtype=np.float32)
        ),
        'A': make_stub_reader(
            np.array([[8.0, 8.0], [12.0, 12.0]], dtype=np.float32)
        ),
    }
    request = _request(file_id='B', normalization_file_id='A')

    direct, adapter = _build_direct_and_adapter(
        request,
        trace_stats_cache={},
        reader_getter=lambda fid, _kb1, _kb2: readers[fid],
        pipeline_section_getter=lambda **_kwargs: None,
        store_dir_resolver=lambda fid: str(stats_dir if fid == 'A' else data_dir),
        dt_resolver=lambda _fid: 0.004,
    )

    _assert_same_payload(direct, adapter)
    np.testing.assert_array_equal(
        _payload_array(adapter),
        np.array([[1, 2], [3, 4]], dtype=np.int8),
    )


def test_adapter_returns_same_payload_as_existing_builder_with_lmo(
    tmp_path: Path,
) -> None:
    write_baseline_raw(tmp_path, key1=7, n_traces=2)
    reader = make_stub_reader(
        np.array(
            [
                [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
                [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            ],
            dtype=np.float32,
        ),
        offsets=[0.0, 2.0],
    )
    request = _request(
        y0=1,
        y1=3,
        lmo_enabled=True,
        lmo_velocity_mps=4.0,
        lmo_offset_mode='signed',
        lmo_ref_mode='zero',
    )

    direct, adapter = _build_direct_and_adapter(
        request,
        trace_stats_cache={},
        reader_getter=lambda _fid, _kb1, _kb2: reader,
        pipeline_section_getter=lambda **_kwargs: None,
        store_dir_resolver=lambda _fid: str(tmp_path),
        dt_resolver=lambda _fid: 1.0,
    )

    _assert_same_payload(direct, adapter)
    np.testing.assert_array_equal(
        _payload_array(adapter),
        np.array([[10, 20, 30], [15, 25, 35]], dtype=np.int8),
    )


def test_adapter_returns_same_payload_for_pipeline_reference_source() -> None:
    arr_a = np.array([[0.0, 2.0], [10.0, 12.0]], dtype=np.float32)
    arr_b = np.array([[2.0, 4.0], [14.0, 16.0]], dtype=np.float32)
    taps = {'tapA': arr_a, 'tapB': arr_b}
    request = _request(
        pipeline_key='pk1',
        tap_label='tapB',
        reference_pipeline_key='pk1',
        reference_tap_label='tapA',
        scaling='tracewise',
    )

    direct, adapter = _build_direct_and_adapter(
        request,
        trace_stats_cache={},
        reader_getter=lambda _fid, _kb1, _kb2: (_ for _ in ()).throw(
            AssertionError('raw reader should not be used')
        ),
        pipeline_section_getter=lambda **kwargs: np.array(
            taps[kwargs['tap_label']], copy=True
        ),
        store_dir_resolver=lambda _fid: '',
        dt_resolver=lambda _fid: 0.004,
    )

    _assert_same_payload(direct, adapter)
    np.testing.assert_array_equal(
        _payload_array(adapter),
        np.array([[1, 3], [3, 5]], dtype=np.int8),
    )
