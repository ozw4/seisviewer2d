from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.api.schemas import PipelineSpec
from app.services.errors import ConflictError
from app.services.fbpick_support import OFFSET_BYTE_FIXED
from app.services.pipeline_execution import (
    SectionSourceSpec,
    build_fbpick_spec,
    prepare_pipeline_execution,
    resolve_execution_dt,
    resolve_section_source,
)


class _DummyState:
    def __init__(self, dt: float) -> None:
        self.file_registry = SimpleNamespace(get_dt=lambda _file_id: dt)


class _DummyReader:
    def __init__(
        self,
        *,
        arr: np.ndarray,
        scale: float | None = None,
        dt: float = 0.004,
        offsets: np.ndarray | None = None,
    ) -> None:
        self._arr = arr
        self._scale = scale
        self.meta = {'dt': dt}
        self._offsets = offsets

    def get_section(self, _key1: int) -> SimpleNamespace:
        return SimpleNamespace(arr=self._arr, scale=self._scale)

    def get_offsets_for_section(self, _key1: int, _offset_byte: int) -> np.ndarray:
        if self._offsets is None:
            raise AssertionError('offsets were not configured')
        return self._offsets


def test_resolve_section_source_raw_window_returns_contiguous_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _DummyState(0.004)
    reader = _DummyReader(
        arr=np.arange(24, dtype=np.int16).reshape(4, 6),
        scale=0.5,
    )
    monkeypatch.setattr(
        'app.services.pipeline_execution.get_reader',
        lambda *_args, **_kwargs: reader,
    )

    resolved = resolve_section_source(
        SectionSourceSpec(
            file_id='file-a',
            key1=10,
            key1_byte=189,
            key2_byte=193,
            window={'tr_min': 1, 'tr_max': 3, 't_min': 2, 't_max': 5},
        ),
        state=state,
    )

    assert resolved.source_kind == 'raw'
    assert resolved.reader is reader
    assert resolved.section.dtype == np.float32
    assert resolved.section.flags['C_CONTIGUOUS']
    assert resolved.section.shape == (2, 3)
    np.testing.assert_allclose(
        resolved.section,
        np.arange(24, dtype=np.float32).reshape(4, 6)[1:3, 2:5] * 0.5,
    )
    assert resolved.trace_slice == slice(1, 3)
    assert resolved.window_bounds == {
        'tr_min': 1,
        'tr_max': 3,
        't_min': 2,
        't_max': 5,
    }


def test_resolve_section_source_empty_window_matches_no_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _DummyState(0.004)
    reader = _DummyReader(arr=np.arange(12, dtype=np.float32).reshape(3, 4))
    monkeypatch.setattr(
        'app.services.pipeline_execution.get_reader',
        lambda *_args, **_kwargs: reader,
    )

    resolved = resolve_section_source(
        SectionSourceSpec(
            file_id='file-a',
            key1=10,
            key1_byte=189,
            key2_byte=193,
            window={},
        ),
        state=state,
    )

    assert resolved.trace_slice is None
    assert resolved.window_bounds is None
    assert resolved.section.shape == (3, 4)


def test_resolve_section_source_pipeline_tap_returns_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _DummyState(0.004)
    tap_meta = {'dt': 0.004, 'label': 'denoise'}
    monkeypatch.setattr(
        'app.services.pipeline_execution.get_section_and_meta_from_pipeline_tap',
        lambda **_kwargs: (np.ones((3, 4), dtype=np.float32), tap_meta),
    )

    resolved = resolve_section_source(
        SectionSourceSpec(
            file_id='file-a',
            key1=11,
            key1_byte=189,
            key2_byte=193,
            pipeline_key='pipe-key',
            tap_label='denoise',
            offset_byte=37,
        ),
        state=state,
    )

    assert resolved.source_kind == 'pipeline_tap'
    assert resolved.reader is None
    assert resolved.source_meta == tap_meta
    assert resolved.trace_slice is None
    assert resolved.window_bounds is None
    assert resolved.section.dtype == np.float32
    assert resolved.section.flags['C_CONTIGUOUS']


def test_resolve_execution_dt_detects_tap_mismatch() -> None:
    state = _DummyState(0.004)

    with pytest.raises(ConflictError, match='dt mismatch between tap and source'):
        resolve_execution_dt(
            'file-a',
            {'dt': 0.002},
            state=state,
            validate_source_dt=True,
        )


def test_prepare_pipeline_execution_forces_offset_byte_and_slices_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _DummyState(0.004)
    reader = _DummyReader(
        arr=np.arange(24, dtype=np.float32).reshape(4, 6),
        offsets=np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        'app.services.pipeline_execution.get_reader',
        lambda *_args, **_kwargs: reader,
    )

    spec = build_fbpick_spec(
        model_id='fbpick_offset_small.pt',
        channel='P',
        tile=(128, 6016),
        overlap=32,
        amp=True,
    )
    context = prepare_pipeline_execution(
        spec=spec,
        source=SectionSourceSpec(
            file_id='file-a',
            key1=12,
            key1_byte=189,
            key2_byte=193,
            window={'tr_min': 1, 'tr_max': 3, 't_min': 0, 't_max': 6},
        ),
        state=state,
    )

    assert context.effective_offset_byte == OFFSET_BYTE_FIXED
    assert context.dt == 0.004


def test_prepare_pipeline_execution_raw_ignores_reader_dt_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _DummyState(0.004)
    reader = _DummyReader(arr=np.ones((2, 3), dtype=np.float32), dt=0.004)
    reader.meta = {'dt': 'broken'}
    monkeypatch.setattr(
        'app.services.pipeline_execution.get_reader',
        lambda *_args, **_kwargs: reader,
    )

    spec = PipelineSpec(
        steps=[
            {
                'kind': 'transform',
                'name': 'bandpass',
                'params': {'low_hz': 1.0, 'high_hz': 10.0},
            }
        ]
    )
    context = prepare_pipeline_execution(
        spec=spec,
        source=SectionSourceSpec(
            file_id='file-a',
            key1=14,
            key1_byte=189,
            key2_byte=193,
        ),
        state=state,
    )

    assert context.dt == 0.004
    assert context.meta['dt'] == 0.004
    assert context.section.shape == (2, 3)


def test_prepare_pipeline_execution_pipeline_tap_uses_tap_dt_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _DummyState(0.004)
    monkeypatch.setattr(
        'app.services.pipeline_execution.get_section_and_meta_from_pipeline_tap',
        lambda **_kwargs: (np.ones((2, 3), dtype=np.float32), {'dt': 0.002}),
    )

    spec = PipelineSpec(
        steps=[
            {
                'kind': 'transform',
                'name': 'bandpass',
                'params': {'low_hz': 1.0, 'high_hz': 10.0},
            }
        ]
    )

    with pytest.raises(ConflictError, match='dt mismatch between tap and source'):
        prepare_pipeline_execution(
            spec=spec,
            source=SectionSourceSpec(
                file_id='file-a',
                key1=13,
                key1_byte=189,
                key2_byte=193,
                pipeline_key='pipe-key',
                tap_label='tap-a',
            ),
            state=state,
            validate_source_dt=True,
        )
