"""Operator interfaces and registry for pipeline steps."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch

from .bandpass import bandpass_np
from .denoise import denoise_tensor
from .fbpick import infer_prob_map


class Transform(Protocol):
    """Callable signature for transform operations."""

    def __call__(
        self,
        x: np.ndarray,
        *,
        params: dict[str, Any],
        meta: dict[str, Any],
    ) -> np.ndarray:
        """Run the transform."""
        ...


class Analyzer(Protocol):
    """Callable signature for analyzer operations."""

    def __call__(
        self,
        x: np.ndarray,
        *,
        params: dict[str, Any],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the analyzer."""
        ...


def op_bandpass(
    x: np.ndarray,
    *,
    params: dict[str, Any],
    meta: dict[str, Any],
) -> np.ndarray:
    """Apply bandpass transform."""
    _ = meta
    return bandpass_np(x, **params)


def op_denoise(
    x: np.ndarray,
    *,
    params: dict[str, Any],
    meta: dict[str, Any],
) -> np.ndarray:
    """Apply denoise transform."""
    _ = meta
    x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    y_t = denoise_tensor(x_t, **params)
    return y_t.squeeze(0).squeeze(0).numpy()


def op_fbpick(x: np.ndarray, *, params: dict, meta: dict) -> dict:
    """Run fbpick analyzer with safe param mapping."""
    amp = params.get("amp", params.get("use_amp", True))
    overlap = int(params.get("overlap", 32))
    tau = float(params.get("tau", 1.0))

    tile = params.get("tile")
    if tile is None:
        chunk_h = params.get("chunk_h")
        tile_w = params.get("tile_w")
        if chunk_h is not None and tile_w is not None:
            tile = (int(chunk_h), int(tile_w))

    arr = np.ascontiguousarray(x, dtype=np.float32)

    kwargs: dict[str, Any] = {"amp": bool(amp), "overlap": overlap, "tau": tau}
    if tile is not None:
        kwargs["tile"] = tuple(tile)

    offsets = None
    if isinstance(params, dict):
        offsets = params.get("offsets")
    if offsets is None and isinstance(meta, dict):
        offsets = meta.get("offsets")
    if offsets is not None:
        kwargs["offsets"] = offsets

    prob = infer_prob_map(arr, **kwargs)
    return {"prob": prob}


TRANSFORMS: dict[str, Transform] = {"bandpass": op_bandpass, "denoise": op_denoise}
ANALYZERS: dict[str, Analyzer] = {"fbpick": op_fbpick}
