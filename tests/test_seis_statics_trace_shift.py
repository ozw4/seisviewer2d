from __future__ import annotations

import numpy as np

from seis_statics.trace_shift import apply_trace_shifts_to_array


def test_apply_trace_shifts_to_array_uses_corrected_t_equals_raw_t_minus_shift() -> None:
    traces = np.zeros((1, 5), dtype=np.float32)
    traces[0, 1] = 1.0

    corrected = apply_trace_shifts_to_array(
        traces=traces,
        sample_interval_s=1.0,
        trace_shift_s_sorted=np.asarray([1.0], dtype=np.float64),
        fill_value=0.0,
    )

    np.testing.assert_allclose(
        corrected,
        np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    )
