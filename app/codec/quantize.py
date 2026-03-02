"""Array quantization helpers."""

from __future__ import annotations

import os

import numpy as np


def quantize_float32(
    arr: np.ndarray, *, bits: int = 8, fixed_scale: float | None = None
) -> tuple[float, np.ndarray]:
    """Quantize ``arr`` into int8 with an optional externally provided scale."""
    qmax = (1 << (bits - 1)) - 1
    default_scale = float(os.getenv('FIXED_INT8_SCALE', '42.333333'))
    scale = float(fixed_scale) if fixed_scale is not None else default_scale
    q = np.clip(np.round(arr * scale), -qmax, qmax).astype(np.int8)
    return scale, q


__all__ = ['quantize_float32']
