"""Model loader for first-arrival probability (fbpick)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from .model import NetAE

_MODEL_PATH = Path(__file__).resolve().parents[2] / 'model' / 'fbpick_edgenext_small.pth'


@lru_cache(maxsize=2)
def _load(device_str: str) -> torch.nn.Module:
        """Load the FBPICK model on the requested device."""
        device = torch.device(device_str)
        if not _MODEL_PATH.exists():
                msg = f'fbpick weights not found at {_MODEL_PATH}'
                raise ValueError(msg)
        model = NetAE(
                backbone='edgenext_small.usi_in1k',
                pretrained=False,
                stage_strides=[(2, 4), (2, 2), (2, 4), (2, 2)],
                pre_stages=2,
                pre_stage_strides=((1, 1), (1, 2)),
                out_chans=1,
        )
        checkpoint = torch.load(_MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                state = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'model_ema' in checkpoint:
                state = checkpoint['model_ema']
        else:
                state = checkpoint
        try:
                model.load_state_dict(state, strict=True)
        except Exception as exc:  # pragma: no cover - rare
                msg = f'failed to load weights from {_MODEL_PATH}: {exc}'
                raise ValueError(msg) from exc
        model.to(device)
        model.eval()
        for p in model.parameters():
                p.requires_grad_(False)
        return model


def get_fbpick_model(device: torch.device) -> torch.nn.Module:
        """Return cached FBPICK model on ``device``."""
        return _load(str(device))
