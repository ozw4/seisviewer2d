"""Binary payload codecs used by API endpoints."""

from __future__ import annotations

import gzip

import msgpack
import numpy as np

from app.utils.utils import quantize_float32


def pack_msgpack_gzip(obj: dict[str, object]) -> bytes:
    """Pack ``obj`` as msgpack and gzip-compress it."""
    payload = msgpack.packb(obj, use_bin_type=True)
    return gzip.compress(payload)


def pack_quantized_array_gzip(
    arr_f32: np.ndarray,
    *,
    scale: float | None,
    dt: float | None,
    extra: dict[str, object] | None = None,
) -> bytes:
    """Quantize a float array and return a msgpack+gzip payload."""
    scale_val, q = quantize_float32(arr_f32, fixed_scale=scale)
    obj: dict[str, object] = {
        'scale': scale_val,
        'shape': arr_f32.shape,
        'data': q.tobytes(),
    }
    if dt is not None:
        obj['dt'] = float(dt)
    if extra:
        obj.update(extra)
    return pack_msgpack_gzip(obj)


__all__ = ['pack_msgpack_gzip', 'pack_quantized_array_gzip']
