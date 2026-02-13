"""Noise suppression model wrapper."""

import logging
import threading
from pathlib import Path
from typing import Literal

import torch

from .model import NetAE
from .predict import cover_all_traces_predict_chunked
from .signal_utils import denorm_per_trace_tensor_b1hw, zscore_per_trace_tensor_b1hw

__all__ = ['denoise_tensor', 'get_denoise_model', 'get_model']

logger = logging.getLogger(__name__)

_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / 'model'
    / 'recon_replace_eq_edgenext_small.pth'
)
_MODEL: torch.nn.Module | None = None
_DEVICE: torch.device | None = None
_MODEL_LOCK = threading.Lock()


def _resolve_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_model(device: torch.device) -> torch.nn.Module:
    model = NetAE(
        backbone='edgenext_small.usi_in1k',
        pretrained=False,
        stage_strides=[(2, 4), (2, 2), (2, 4), (2, 2)],
        pre_stages=2,
        pre_stage_strides=((1, 1), (1, 2)),
    )
    checkpoint = torch.load(_MODEL_PATH, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_ema'], strict=True)
    model.to(device)
    model.eval()
    return model


def get_denoise_model() -> torch.nn.Module:
    """Return the singleton denoise model (lazy-loaded, thread-safe)."""
    global _MODEL, _DEVICE
    if _MODEL is not None:
        if _DEVICE is None:
            _DEVICE = next(_MODEL.parameters()).device
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            device = _resolve_device()
            logger.info(
                'Loading denoise model from %s (device=%s)', _MODEL_PATH, device
            )
            _MODEL = _load_model(device)
            _DEVICE = device
        elif _DEVICE is None:
            _DEVICE = next(_MODEL.parameters()).device
    return _MODEL


def get_model() -> torch.nn.Module:
    """Backward-compatible alias for the denoise model accessor."""
    return get_denoise_model()


def _get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        model = get_denoise_model()
        _DEVICE = next(model.parameters()).device
    return _DEVICE


@torch.no_grad()
def denoise_tensor(
    x: torch.Tensor,
    *,
    chunk_h: int = 128,
    overlap: int = 32,
    mask_ratio: float = 0.5,
    noise_std: float = 1.0,
    mask_noise_mode: Literal['replace', 'add'] = 'replace',
    use_amp: bool = True,
    seed: int = 12345,
    passes_batch: int = 4,
) -> torch.Tensor:
    """Denoise input of shape ``(B,1,H,W)`` and return CPU tensor."""
    expected_dim = 4
    if x.dim() != expected_dim or x.size(1) != 1:
        msg = 'x must be (B,1,H,W)'
        raise ValueError(msg)
    xn, mean, inv_std = zscore_per_trace_tensor_b1hw(x, eps=1e-6)
    model = get_denoise_model()
    device = _get_device()
    xt = xn.to(device)
    yt = cover_all_traces_predict_chunked(
        model,
        xt,
        chunk_h=chunk_h,
        overlap=overlap,
        mask_ratio=mask_ratio,
        noise_std=noise_std,
        mask_noise_mode=mask_noise_mode,
        use_amp=use_amp,
        device=device,
        seed=seed,
        passes_batch=passes_batch,
    )
    y_restored = denorm_per_trace_tensor_b1hw(
        yt, mean=mean.to(yt.device), inv_std=inv_std.to(yt.device)
    )
    return y_restored.cpu()
