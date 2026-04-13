"""Operator interfaces and registry for pipeline steps."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch

from .bandpass import bandpass_np
from .denoise import denoise_tensor
from .fbpick import infer_prob_map
from .fbpick_models import resolve_model_path


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
    dt = None
    if isinstance(meta, dict):
        dt = meta.get("dt")
    if not isinstance(dt, (int, float)) or dt <= 0:
        msg = "Bandpass transform requires a positive dt value in metadata"
        raise ValueError(msg)

    kwargs = dict(params or {})
    kwargs.pop("dt", None)

    high_hz = kwargs.get("high_hz")
    if isinstance(high_hz, (int, float)):
        nyquist = 0.5 / float(dt)
        if high_hz > nyquist:
            msg = f"high_hz must be <= Nyquist (0.5/dt={nyquist:g})"
            raise ValueError(msg)

    return bandpass_np(x, dt=float(dt), **kwargs)


def op_denoise(
    x: np.ndarray,
    *,
    params: dict[str, Any],
    meta: dict[str, Any],
) -> np.ndarray:
    """Apply denoise transform."""
    _ = meta
    arr = np.ascontiguousarray(x, dtype=np.float32)
    x_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    y_t = denoise_tensor(x_t, **params)
    y = np.ascontiguousarray(y_t.squeeze(0).squeeze(0).numpy(), dtype=np.float32)
    if y.shape != arr.shape:
        raise ValueError(
            f'Denoise output shape mismatch: got {y.shape}, expected {arr.shape}'
        )
    return y


def op_fbpick(x: np.ndarray, *, params: dict, meta: dict) -> dict:
    """Run fbpick analyzer with safe param mapping."""
    amp = params.get("amp", params.get("use_amp", True))
    overlap = params.get("overlap", 32)
    tau = float(params.get("tau", 1.0))
    channel = params.get("channel")
    tiles_per_batch = int(params.get("tiles_per_batch", 4))

    tile = params.get("tile")

    arr = np.ascontiguousarray(x, dtype=np.float32)

    kwargs: dict[str, Any] = {
        "amp": bool(amp),
        "overlap": overlap,
        "tau": tau,
        "channel": channel,
        "tiles_per_batch": tiles_per_batch,
    }
    if tile is not None:
        kwargs["tile"] = tuple(tile)

    offsets = None
    if isinstance(params, dict):
        offsets = params.get("offsets")
    if offsets is None and isinstance(meta, dict):
        offsets = meta.get("offsets")
    if offsets is not None:
        kwargs["offsets"] = offsets

    model_id = params.get("model_id") if isinstance(params, dict) else None
    if model_id is not None:
        model_path = resolve_model_path(model_id, require_exists=True)
        kwargs["model_path"] = model_path

    prob = infer_prob_map(arr, **kwargs)
    return {"prob": prob}


TRANSFORMS: dict[str, Transform] = {"bandpass": op_bandpass, "denoise": op_denoise}
ANALYZERS: dict[str, Analyzer] = {"fbpick": op_fbpick}
