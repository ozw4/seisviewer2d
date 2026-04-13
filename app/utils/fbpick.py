"""First-break probability inference wrapper backed by ``seisai_engine``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.utils.fbpick_models import resolve_model_path

__all__ = ['_MODEL_PATH', 'clear_model_cache', 'infer_prob_map']

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).resolve().parents[2] / 'model' / 'fbpick_edgenext_small.pt'

_OFFSET_VECTOR_NDIM = 1

_InferProbHwFn = Callable[..., np.ndarray]
_ClearModelCacheFn = Callable[..., Any]


def _viewer_api() -> tuple[_InferProbHwFn, _ClearModelCacheFn]:
    try:
        from seisai_engine.viewer import clear_model_cache, infer_prob_hw
    except ImportError as exc:
        msg = (
            'seisai_engine.viewer is required for fbpick inference. '
            'Install seisai-engine in the runtime environment.'
        )
        raise RuntimeError(msg) from exc
    return infer_prob_hw, clear_model_cache


def _normalize_pair(
    value: int | tuple[int, int] | list[int], *, name: str
) -> tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f'{name} must be positive, got {value}')
        return (value, value)
    if isinstance(value, list | tuple) and len(value) == 2:
        h = int(value[0])
        w = int(value[1])
        if h <= 0 or w <= 0:
            raise ValueError(f'{name} must be positive, got {(h, w)}')
        return (h, w)
    raise ValueError(f'{name} must be int or a pair of ints, got {value!r}')


def _coerce_offsets_h(offsets: np.ndarray, *, h: int, w: int) -> np.ndarray:
    arr = np.asarray(offsets, dtype=np.float32)
    if arr.ndim == _OFFSET_VECTOR_NDIM:
        n = int(arr.shape[0])
        if n != h:
            raise ValueError(
                f'Offsets length must match H={h} for seisai offsets_h input, got {n} (W={w})'
            )
        return np.ascontiguousarray(arr, dtype=np.float32)
    raise ValueError(
        f'Offsets must be a 1D array of length H={h} for seisai offsets_h input, got shape={arr.shape} (W={w})'
    )


def clear_model_cache(*args: Any, **kwargs: Any) -> Any:
    _, clear_cache = _viewer_api()
    return clear_cache(*args, **kwargs)


def infer_prob_map(
    section: np.ndarray,
    *,
    amp: bool = True,
    tile: tuple[int, int] = (128, 6016),
    overlap: int | tuple[int, int] = 32,
    tau: float = 1.0,
    offsets: np.ndarray | None = None,
    model_path: Path | None = None,
    uses_offset: bool | None = None,
    model_id: str | None = None,
    channel: str | int | None = None,
    device: str = 'auto',
    tiles_per_batch: int = 4,
) -> np.ndarray:
    """Infer first-break probability for ``section`` via ``seisai_engine.viewer``."""
    arr = np.ascontiguousarray(section, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f'Section must be 2D, got shape={arr.shape}')
    h, w = int(arr.shape[0]), int(arr.shape[1])

    if uses_offset is not None:
        logger.warning(
            'uses_offset argument is ignored; offset usage is inferred by checkpoint'
        )

    resolved_model_path: Path
    if model_id is not None:
        resolved_model_path = resolve_model_path(model_id, require_exists=True)
    elif model_path is not None:
        resolved_model_path = Path(model_path)
    else:
        resolved_model_path = resolve_model_path(None, require_exists=True)

    offsets_h = None if offsets is None else _coerce_offsets_h(offsets, h=h, w=w)
    tile_hw = _normalize_pair(tile, name='tile')
    overlap_hw = _normalize_pair(overlap, name='overlap')

    infer_prob_hw, _ = _viewer_api()
    prob = infer_prob_hw(
        arr,
        ckpt_path=resolved_model_path,
        offsets_h=offsets_h,
        channel=channel,
        device=device,
        tile=tile_hw,
        overlap=overlap_hw,
        amp=bool(amp),
        tiles_per_batch=int(tiles_per_batch),
        tau=float(tau),
    )
    prob_hw = np.ascontiguousarray(np.asarray(prob, dtype=np.float32))
    if prob_hw.shape != (h, w):
        raise ValueError(
            f'infer_prob_hw returned unexpected shape {prob_hw.shape}; expected {(h, w)}'
        )
    return prob_hw
