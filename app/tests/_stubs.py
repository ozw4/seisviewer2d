from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from app.trace_store.types import SectionView


def make_stub_reader(
    section_arr: np.ndarray,
    *,
    key1_values: Sequence[int] = (7,),
    key_bytes: tuple[int, int] = (189, 193),
):
    section = np.asarray(section_arr, dtype=np.float32)
    key1_values_arr = np.asarray(key1_values, dtype=np.int32)
    key1_byte = int(key_bytes[0])
    key2_byte = int(key_bytes[1])

    class _StubReader:
        def get_key1_values(self):
            return key1_values_arr

        def get_section(self, _key1_val: int):
            arr = np.array(section, dtype=np.float32, copy=True)
            return SectionView(arr=arr, dtype=arr.dtype, scale=None)

    _StubReader.key1_byte = key1_byte
    _StubReader.key2_byte = key2_byte

    return _StubReader()


def write_baseline_raw(
    store_dir: Path,
    *,
    key1: int,
    n_traces: int | None = None,
    section_mean: float = 0.0,
    section_std: float = 1.0,
    trace_means: Sequence[float] | None = None,
    trace_stds: Sequence[float] | None = None,
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

    baseline = {
        'key1_values': [int(key1)],
        'mu_section_by_key1': [float(section_mean)],
        'sigma_section_by_key1': [float(section_std)],
        'mu_traces': trace_means_values,
        'sigma_traces': trace_stds_values,
        'trace_spans_by_key1': {str(int(key1)): [[0, int(len(trace_means_values))]]},
    }
    (store_dir / 'baseline_raw.json').write_text(json.dumps(baseline), encoding='utf-8')


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
