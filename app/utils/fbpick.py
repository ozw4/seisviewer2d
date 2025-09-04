from __future__ import annotations

"""First-break probability model wrapper."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .model import NetAE

__all__ = ["_MODEL_PATH", "infer_prob_map"]

_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "model" / "fbpick_edgenext_small.pth"
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _load_model() -> tuple[torch.nn.Module, torch.device]:
    device = _device()
    model = NetAE(
        backbone="edgenext_small.usi_in1k",
        pretrained=False,
        stage_strides=[(2, 4), (2, 2), (2, 4), (2, 2)],
        pre_stages=2,
        pre_stage_strides=((1, 1), (1, 2)),
    )
    state = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_ema" in state:
        state = state["model_ema"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, device


@torch.no_grad()
def _run_tiled(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    tile: tuple[int, int] = (256, 256),
    overlap: int = 32,
    amp: bool = True,
) -> torch.Tensor:
    """Run ``model`` on ``x`` using sliding-window tiling."""
    b, c, h, w = x.shape
    tile_h, tile_w = tile
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    out = torch.zeros((b, 1, h, w), device=x.device, dtype=torch.float32)
    weight = torch.zeros_like(out)
    for top in range(0, h, stride_h):
        for left in range(0, w, stride_w):
            bottom = min(top + tile_h, h)
            right = min(left + tile_w, w)
            h0 = max(0, bottom - tile_h)
            w0 = max(0, right - tile_w)
            patch = x[:, :, h0:bottom, w0:right]
            with torch.cuda.amp.autocast(enabled=amp and torch.cuda.is_available()):
                yp = model(patch)
            out[:, :, h0:bottom, w0:right] += yp
            weight[:, :, h0:bottom, w0:right] += 1
    out /= weight.clamp_min(1.0)
    return out


@torch.no_grad()
def infer_prob_map(
    section: np.ndarray,
    *,
    amp: bool = True,
    tile: tuple[int, int] = (256, 256),
    overlap: int = 32,
) -> np.ndarray:
    """Infer first-break probability for ``section``."""
    model, device = _load_model()
    x = torch.from_numpy(section).float().unsqueeze(0).unsqueeze(0).to(device)
    y = _run_tiled(model, x, tile=tile, overlap=overlap, amp=amp)
    y = torch.sigmoid(y)
    return y.squeeze(0).squeeze(0).clamp_(0, 1).cpu().numpy().astype(np.float32)
