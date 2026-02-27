"""Noise suppression inference wrapper backed by ``seisai_engine.viewer``."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch

from .signal_utils import denorm_per_trace_tensor_b1hw, zscore_per_trace_tensor_b1hw

__all__ = [
    '_MODEL_PATH',
    'denoise_tensor',
    'get_denoise_ckpt_path',
    'get_denoise_model',
    'get_model',
]

_MODEL_PATH = Path(__file__).resolve().parents[2] / 'model' / 'denoise_default.pt'
_NUMBA_CACHE_DIR = '/tmp/numba'
_LEGACY_DEFAULT_CHUNK_H = 128
_LEGACY_DEFAULT_OVERLAP = 32
_LEGACY_DEFAULT_MASK_RATIO = 0.5
_LEGACY_DEFAULT_NOISE_STD = 1.0
_LEGACY_DEFAULT_MASK_NOISE_MODE = 'replace'
_LEGACY_DEFAULT_SEED = 12345
_LEGACY_DEFAULT_PASSES_BATCH = 4

_InferDenoiseHwFn = Callable[..., np.ndarray]
logger = logging.getLogger(__name__)


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


def _to_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f'{name} must be an int, got {value!r}')
    int_value = int(value)
    if int_value <= 0:
        raise ValueError(f'{name} must be positive, got {int_value}')
    return int_value


def _to_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f'{name} must be an int, got {value!r}')
    int_value = int(value)
    if int_value < 0:
        raise ValueError(f'{name} must be non-negative, got {int_value}')
    return int_value


def _normalize_tile(
    tile: int | tuple[int, int] | list[int],
) -> tuple[int, int]:
    if isinstance(tile, int | np.integer) and not isinstance(tile, bool):
        tile_value = _to_positive_int(tile, name='tile')
        return (tile_value, tile_value)
    if isinstance(tile, tuple | list) and len(tile) == 2:
        return (
            _to_positive_int(tile[0], name='tile[0]'),
            _to_positive_int(tile[1], name='tile[1]'),
        )
    raise ValueError(f'tile must be int or pair of ints, got {tile!r}')


def _normalize_overlap(
    overlap: int | tuple[int, int] | list[int], *, tile: tuple[int, int]
) -> tuple[int, int]:
    if isinstance(overlap, int | np.integer) and not isinstance(overlap, bool):
        ov_h = _to_non_negative_int(overlap, name='overlap')
        ov_w = 0
    elif isinstance(overlap, tuple | list) and len(overlap) == 2:
        ov_h = _to_non_negative_int(overlap[0], name='overlap[0]')
        ov_w = _to_non_negative_int(overlap[1], name='overlap[1]')
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


def _warn_ignored_legacy_param(
    *, name: str, value: object, legacy_default: object
) -> None:
    logger.warning(
        '[DENOISE][IGNORED] %s=%r is ignored by seisai_engine-backed denoise (legacy default=%r)',
        name,
        value,
        legacy_default,
    )


def _warn_ignored_legacy_args(
    *,
    mask_ratio: float,
    noise_std: float,
    mask_noise_mode: Literal['replace', 'add'],
    seed: int,
) -> None:
    if float(mask_ratio) != _LEGACY_DEFAULT_MASK_RATIO:
        _warn_ignored_legacy_param(
            name='mask_ratio',
            value=mask_ratio,
            legacy_default=_LEGACY_DEFAULT_MASK_RATIO,
        )
    if float(noise_std) != _LEGACY_DEFAULT_NOISE_STD:
        _warn_ignored_legacy_param(
            name='noise_std', value=noise_std, legacy_default=_LEGACY_DEFAULT_NOISE_STD
        )
    if mask_noise_mode != _LEGACY_DEFAULT_MASK_NOISE_MODE:
        _warn_ignored_legacy_param(
            name='mask_noise_mode',
            value=mask_noise_mode,
            legacy_default=_LEGACY_DEFAULT_MASK_NOISE_MODE,
        )
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int | np.integer)
        or int(seed) != _LEGACY_DEFAULT_SEED
    ):
        _warn_ignored_legacy_param(
            name='seed',
            value=seed,
            legacy_default=_LEGACY_DEFAULT_SEED,
        )


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
    chunk_h: int = 128,
    overlap: int | tuple[int, int] = 32,
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

    _warn_ignored_legacy_args(
        mask_ratio=mask_ratio,
        noise_std=noise_std,
        mask_noise_mode=mask_noise_mode,
        seed=seed,
    )
    if chunk_h != _LEGACY_DEFAULT_CHUNK_H:
        _warn_ignored_legacy_param(
            name='chunk_h', value=chunk_h, legacy_default=_LEGACY_DEFAULT_CHUNK_H
        )
    if passes_batch != _LEGACY_DEFAULT_PASSES_BATCH:
        _warn_ignored_legacy_param(
            name='passes_batch',
            value=passes_batch,
            legacy_default=_LEGACY_DEFAULT_PASSES_BATCH,
        )

    amp_flag = bool(use_amp) if amp is None else bool(amp)

    infer_kwargs: dict[str, object] = {
        'ckpt_path': _resolve_ckpt_path(ckpt_path),
        'device': device,
        'amp': amp_flag,
    }
    if tile is not None:
        tile_hw = _normalize_tile(tile)
        overlap_hw = _normalize_overlap(overlap, tile=tile_hw)
        infer_kwargs['tile'] = tile_hw
        infer_kwargs['overlap'] = overlap_hw
    elif overlap != _LEGACY_DEFAULT_OVERLAP:
        _warn_ignored_legacy_param(
            name='overlap', value=overlap, legacy_default=_LEGACY_DEFAULT_OVERLAP
        )
    if tiles_per_batch is not None:
        infer_kwargs['tiles_per_batch'] = _to_positive_int(
            tiles_per_batch, name='tiles_per_batch'
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
