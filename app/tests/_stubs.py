from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from app.trace_store.types import SectionView
from app.utils.baseline_artifacts import build_raw_baseline_payload, write_raw_baseline_artifacts


def make_stub_reader(
    section_arr: np.ndarray,
    *,
    offsets: Sequence[float] | np.ndarray | None = None,
    key1_values: Sequence[int] = (7,),
    key_bytes: tuple[int, int] = (189, 193),
):
    section = np.asarray(section_arr, dtype=np.float32)
    offsets_arr = (
        None
        if offsets is None
        else np.asarray(offsets, dtype=np.float32)
    )
    key1_values_arr = np.asarray(key1_values, dtype=np.int32)
    key1_byte = int(key_bytes[0])
    key2_byte = int(key_bytes[1])

    class _StubReader:
        def get_key1_values(self):
            return key1_values_arr

        def get_section(self, _key1_val: int):
            arr = np.array(section, dtype=np.float32, copy=True)
            return SectionView(arr=arr, dtype=arr.dtype, scale=None)

        def get_trace_seq_for_value(self, _key1_val: int, align_to: str = 'display'):
            if align_to not in {'display', 'original'}:
                raise ValueError("align_to must be 'display' or 'original'")
            return np.arange(section.shape[0], dtype=np.int64)

        def get_offsets_for_section(self, _key1_val: int, _offset_byte: int):
            if offsets_arr is None:
                raise ValueError('offsets unavailable')
            return np.array(offsets_arr, dtype=np.float32, copy=True)

    _StubReader.key1_byte = key1_byte
    _StubReader.key2_byte = key2_byte

    return _StubReader()


def write_baseline_raw(
    store_dir: Path,
    *,
    key1: int,
    key1_byte: int = 189,
    key2_byte: int = 193,
    source_sha256: str | None = None,
    n_traces: int | None = None,
    section_mean: float = 0.0,
    section_std: float = 1.0,
    trace_means: Sequence[float] | None = None,
    trace_stds: Sequence[float] | None = None,
    legacy_only: bool = False,
) -> None:
    if trace_means is None and trace_stds is None:
        if n_traces is None:
            raise ValueError('n_traces is required when trace stats are omitted')
        trace_means_values = [0.0] * int(n_traces)
        trace_stds_values = [1.0] * int(n_traces)
    elif trace_means is not None and trace_stds is not None:
        if n_traces is not None:
            raise ValueError('n_traces must be omitted when trace stats are provided')
        trace_means_values = [float(v) for v in trace_means]
        trace_stds_values = [float(v) for v in trace_stds]
        if len(trace_means_values) != len(trace_stds_values):
            raise ValueError('trace_means and trace_stds must have equal length')
    else:
        raise ValueError('trace_means and trace_stds must be provided together')

    payload = build_raw_baseline_payload(
        dtype_base='float32',
        dt=0.004,
        key1_values=np.asarray([int(key1)], dtype=np.int64),
        mu_sections=np.asarray([float(section_mean)], dtype=np.float32),
        sigma_sections=np.asarray([float(section_std)], dtype=np.float32),
        mu_traces=np.asarray(trace_means_values, dtype=np.float32),
        sigma_traces=np.asarray(trace_stds_values, dtype=np.float32),
        zero_var_mask=np.asarray(
            [float(v) <= 0.0 for v in trace_stds_values],
            dtype=bool,
        ),
        trace_spans_by_key1={str(int(key1)): [[0, int(len(trace_means_values))]]},
        source_sha256=source_sha256,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    if legacy_only:
        (store_dir / 'baseline_raw.json').write_text(
            json.dumps(payload),
            encoding='utf-8',
        )
        return
    write_raw_baseline_artifacts(
        store_dir,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        payload=payload,
    )


def make_pipeline_outputs_stub():
    def _stub_pipeline_outputs(*, section, meta, spec, denoise_taps, fbpick_label):
        del meta, spec, denoise_taps, fbpick_label
        denoise = np.asarray(section, dtype=np.float32, order='C')
        prob = np.ones(denoise.shape, dtype=np.float16)
        return denoise, prob

    return _stub_pipeline_outputs


__all__ = [
    'make_pipeline_outputs_stub',
    'make_stub_reader',
    'write_baseline_raw',
]
