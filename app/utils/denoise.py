"""Noise suppression inference wrapper backed by ``seisai_engine.viewer``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch

from .signal_utils import denorm_per_trace_tensor_b1hw, zscore_per_trace_tensor_b1hw
from .validation import require_non_negative_int, require_positive_int

__all__ = [
    '_MODEL_PATH',
    'denoise_tensor',
    'get_denoise_ckpt_path',
    'get_denoise_model',
    'get_model',
]

_MODEL_PATH = Path(__file__).resolve().parents[2] / 'model' / 'denoise_default.pt'
_NUMBA_CACHE_DIR = '/tmp/numba'
_DEFAULT_CHUNK_H = 128
_DEFAULT_OVERLAP_H = 32
_DEFAULT_TILE_W = 6016
_DEFAULT_OVERLAP_W = 1024

_InferDenoiseHwFn = Callable[..., np.ndarray]


def _viewer_api() -> _InferDenoiseHwFn:
    os.environ.setdefault('NUMBA_CACHE_DIR', _NUMBA_CACHE_DIR)
    try:
        from seisai_engine.viewer import infer_denoise_hw
    except ImportError as exc:
        msg = (
            'seisai_engine.viewer is required for denoise inference. '
            'Install seisai-engine in the runtime environment.'
        )
        raise RuntimeError(msg) from exc
    return infer_denoise_hw


def _normalize_tile(
    tile: int | tuple[int, int] | list[int],
) -> tuple[int, int]:
    if isinstance(tile, int | np.integer) and not isinstance(tile, bool):
        tile_value = require_positive_int(tile, 'tile')
        return (tile_value, tile_value)
    if isinstance(tile, tuple | list) and len(tile) == 2:
        return (
            require_positive_int(tile[0], 'tile[0]'),
            require_positive_int(tile[1], 'tile[1]'),
        )
    raise ValueError(f'tile must be int or pair of ints, got {tile!r}')


def _normalize_overlap(
    overlap: int | tuple[int, int] | list[int], *, tile: tuple[int, int]
) -> tuple[int, int]:
    if isinstance(overlap, int | np.integer) and not isinstance(overlap, bool):
        ov_h = require_non_negative_int(overlap, 'overlap')
        ov_w = 0
    elif isinstance(overlap, tuple | list) and len(overlap) == 2:
        ov_h = require_non_negative_int(overlap[0], 'overlap[0]')
        ov_w = require_non_negative_int(overlap[1], 'overlap[1]')
    else:
        raise ValueError(f'overlap must be int or pair of ints, got {overlap!r}')
    if ov_h >= tile[0] or ov_w >= tile[1]:
        raise ValueError(
            f'overlap must satisfy overlap < tile, got overlap={(ov_h, ov_w)}, tile={tile}'
        )
    return (ov_h, ov_w)


def _resolve_ckpt_path(ckpt_path: Path | str | None) -> Path:
    resolved = _MODEL_PATH if ckpt_path is None else Path(ckpt_path)
    if not resolved.is_file():
        raise ValueError(f'denoise checkpoint not found: {resolved}')
    return resolved


def get_denoise_ckpt_path(*, ckpt_path: Path | str | None = None) -> Path:
    """Return the resolved denoise checkpoint path."""
    return _resolve_ckpt_path(ckpt_path)


def get_denoise_model(*, ckpt_path: Path | str | None = None) -> Path:
    """Deprecated alias that returns checkpoint path without loading any model."""
    return get_denoise_ckpt_path(ckpt_path=ckpt_path)


def get_model(*, ckpt_path: Path | str | None = None) -> Path:
    """Deprecated alias that returns checkpoint path without loading any model."""
    return get_denoise_model(ckpt_path=ckpt_path)


@torch.no_grad()
def denoise_tensor(
    x: torch.Tensor,
    *,
    chunk_h: int = _DEFAULT_CHUNK_H,
    overlap: int | tuple[int, int] = _DEFAULT_OVERLAP_H,
    mask_ratio: float = 0.5,
    noise_std: float = 1.0,
    mask_noise_mode: Literal['replace', 'add'] = 'replace',
    use_amp: bool = True,
    seed: int = 12345,
    passes_batch: int = 4,
    ckpt_path: Path | str | None = None,
    tile: int | tuple[int, int] | list[int] | None = None,
    amp: bool | None = None,
    tiles_per_batch: int | None = None,
    use_ema: bool | None = None,
    device: str | torch.device = 'auto',
) -> torch.Tensor:
    """Denoise input of shape ``(B,1,H,W)`` and return CPU tensor."""
    if x.dim() != 4 or x.size(1) != 1:
        raise ValueError('x must be (B,1,H,W)')
    batch_size = int(x.size(0))
    h = int(x.size(2))
    w = int(x.size(3))
    if batch_size <= 0 or h <= 0 or w <= 0:
        raise ValueError(f'x must have positive shape, got {tuple(x.shape)}')

    chunk_h_value = require_positive_int(chunk_h, 'chunk_h')

    if isinstance(mask_ratio, bool):
        raise ValueError(f'mask_ratio must be a float in [0, 1], got {mask_ratio!r}')
    try:
        mask_ratio_value = float(mask_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'mask_ratio must be a float in [0, 1], got {mask_ratio!r}'
        ) from exc
    if mask_ratio_value < 0.0 or mask_ratio_value > 1.0:
        raise ValueError(f'mask_ratio must be in [0, 1], got {mask_ratio_value}')

    if isinstance(noise_std, bool):
        raise ValueError(f'noise_std must be a float >= 0, got {noise_std!r}')
    try:
        noise_std_value = float(noise_std)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'noise_std must be a float >= 0, got {noise_std!r}') from exc
    if noise_std_value < 0.0:
        raise ValueError(f'noise_std must be >= 0, got {noise_std_value}')

    if mask_noise_mode not in ('replace', 'add'):
        raise ValueError(
            "mask_noise_mode must be either 'replace' or 'add', "
            f'got {mask_noise_mode!r}'
        )

    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise ValueError(f'seed must be an int, got {seed!r}')
    seed_value = int(seed)

    passes_batch_value = require_positive_int(passes_batch, 'passes_batch')

    amp_flag = bool(use_amp) if amp is None else bool(amp)

    if tile is None:
        overlap_h = require_non_negative_int(overlap, 'overlap')
        if overlap_h >= chunk_h_value:
            raise ValueError(
                f'overlap must satisfy overlap < chunk_h, got overlap={overlap_h}, chunk_h={chunk_h_value}'
            )
        tile_hw = (chunk_h_value, _DEFAULT_TILE_W)
        overlap_hw = (overlap_h, _DEFAULT_OVERLAP_W)
    else:
        tile_hw = _normalize_tile(tile)
        overlap_hw = _normalize_overlap(overlap, tile=tile_hw)

    infer_kwargs: dict[str, object] = {
        'ckpt_path': _resolve_ckpt_path(ckpt_path),
        'device': device,
        'amp': amp_flag,
        'mask_ratio': mask_ratio_value,
        'noise_std': noise_std_value,
        'mask_noise_mode': mask_noise_mode,
        'seed': seed_value,
        'passes_batch': passes_batch_value,
        'tile': tile_hw,
        'overlap': overlap_hw,
    }
    if tiles_per_batch is not None:
        infer_kwargs['tiles_per_batch'] = require_positive_int(
            tiles_per_batch, 'tiles_per_batch'
        )
    if use_ema is not None and not isinstance(use_ema, bool):
        raise ValueError(f'use_ema must be bool or None, got {type(use_ema)}')
    if use_ema is not None:
        infer_kwargs['use_ema'] = use_ema

    infer_denoise_hw = _viewer_api()

    x_f32 = x.to(dtype=torch.float32)
    xn, mean, inv_std = zscore_per_trace_tensor_b1hw(x_f32, eps=1e-6)
    xn_cpu = xn.detach().cpu()
    mean_cpu = mean.detach().cpu()
    inv_std_cpu = inv_std.detach().cpu()

    denoised_norm_batches: list[torch.Tensor] = []
    for batch_idx in range(batch_size):
        section_hw = np.ascontiguousarray(
            xn_cpu[batch_idx, 0].numpy(), dtype=np.float32
        )
        denoised_norm_hw = infer_denoise_hw(
            section_hw,
            **infer_kwargs,
        )
        denoised_norm_hw_f32 = np.ascontiguousarray(
            np.asarray(denoised_norm_hw, dtype=np.float32)
        )
        if denoised_norm_hw_f32.shape != (h, w):
            raise ValueError(
                f'infer_denoise_hw returned unexpected shape {denoised_norm_hw_f32.shape}; expected {(h, w)}'
            )
        denoised_norm_batches.append(torch.from_numpy(denoised_norm_hw_f32))

    y_norm = torch.stack(denoised_norm_batches, dim=0).unsqueeze(1)
    y_restored = denorm_per_trace_tensor_b1hw(
        y_norm, mean=mean_cpu, inv_std=inv_std_cpu
    )
    return y_restored.to(dtype=torch.float32).contiguous()
