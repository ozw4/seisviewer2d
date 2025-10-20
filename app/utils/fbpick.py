"""First-break probability model wrapper."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_nn_func

from app.utils.signal_utils import zscore_per_trace_np

from .model import NetAE
from .model_utils import inflate_input_convs_to_2ch

__all__ = ['_MODEL_PATH', 'infer_prob_map', 'make_offset_channel']

_MODEL_PATH = (
	Path(__file__).resolve().parents[2]
	/ 'model'
	/ 'fbpick_edgenext_small_mobara_tr1.pth'
)


_OFFSET_VECTOR_NDIM = 1
_OFFSET_IMAGE_NDIM = 2


def _device() -> torch.device:
	return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@lru_cache(maxsize=1)
def _load_model() -> tuple[torch.nn.Module, torch.device]:
	device = _device()

	use_offset = 'offset' in _MODEL_PATH.name
	model = NetAE(
		backbone='edgenext_small.usi_in1k',
		pretrained=False,
		stage_strides=[(2, 4), (2, 2), (2, 4), (2, 2)],
		pre_stages=2,
		pre_stage_strides=((1, 1), (1, 2)),
	)
	if use_offset:
		inflate_input_convs_to_2ch(model, verbose=True, init_mode='zero')

	state = torch.load(_MODEL_PATH, map_location='cpu', weights_only=False)
	if isinstance(state, dict) and 'model_ema' in state:
		state = state['model_ema']
	model.load_state_dict(state, strict=True)
	model.eval().to(device)
	return model, device


@torch.no_grad()
def _run_tiled(
	model: torch.nn.Module,
	x: torch.Tensor,
	*,
	tile: tuple[int, int] = (128, 6016),
	overlap: int = 32,
	amp: bool = True,
	offset_channel: int | None = None,
) -> torch.Tensor:
	"""Run ``model`` on ``x`` using sliding-window tiling."""
	b, _, h, w = x.shape
	tile_h, tile_w = tile
	stride_h = tile_h - overlap
	stride_w = tile_w - overlap
	out = torch.zeros((b, 1, h, w), device=x.device, dtype=torch.float32)
	weight = torch.zeros_like(out)
	autocast_available = torch.cuda.is_available()
	for top in range(0, h, stride_h):
		for left in range(0, w, stride_w):
			bottom = min(top + tile_h, h)
			right = min(left + tile_w, w)
			h0 = max(0, bottom - tile_h)
			w0 = max(0, right - tile_w)
			patch = x[:, :, h0:bottom, w0:right]
			if offset_channel is not None:
				patch = patch.clone()
				off = patch[:, offset_channel : offset_channel + 1, :, :]
				mean = off.mean(dim=(2, 3), keepdim=True)
				var = off.var(dim=(2, 3), keepdim=True, unbiased=False)
				std = torch.sqrt(var).clamp_min(1e-6)
				patch[:, offset_channel : offset_channel + 1, :, :] = (off - mean) / std
			ph, pw = patch.shape[-2], patch.shape[-1]
			pad_h = max(0, tile_h - ph)
			pad_w = max(0, tile_w - pw)

			if pad_h or pad_w:
				patch = torch_nn_func.pad(
					patch,
					(0, pad_w, 0, pad_h),
					mode='constant',
					value=0.0,
				)
			autocast_enabled = amp and autocast_available
			with torch.cuda.amp.autocast(enabled=autocast_enabled):
				model.print_shapes = False
				yp = model(patch)  # (B,1,tile_h,tile_w) expected
			yp = yp[..., :ph, :pw]
			out[:, :, h0:bottom, w0:right] += yp
			weight[:, :, h0:bottom, w0:right] += 1
	out /= weight.clamp_min(1.0)
	return out


def make_offset_channel(offsets: np.ndarray, h: int, w: int) -> np.ndarray:
	"""Return a (h, w) float32 offset channel from ``offsets``.

	Accepts:
		- 1D vector of length **W** (per-trace along width)  -> broadcast to (H, W)
		- 1D vector of length **H** (per-trace along height) -> broadcast to (H, W)
		- 2D array of shape (H, W)                           -> used as-is
		(and also tolerates (W, H) by transposing)
	"""
	arr = np.asarray(offsets, dtype=np.float32)

	if arr.ndim == _OFFSET_VECTOR_NDIM:
		n = arr.shape[0]
		if n == w:
			# vector aligned to width (samples)
			arr = np.broadcast_to(arr.reshape(1, w), (h, w)).copy()
		elif n == h:
			# vector aligned to height (traces)
			arr = np.broadcast_to(arr.reshape(h, 1), (h, w)).copy()
		else:
			raise ValueError(
				f'Offset vector length {n} does not match either height {h} or width {w}'
			)
	elif arr.ndim == _OFFSET_IMAGE_NDIM:
		if arr.shape == (h, w):
			pass
		elif arr.shape == (w, h):
			arr = arr.T
		else:
			raise ValueError(
				f'Offset array shape {arr.shape} does not match (H,W)=({h},{w}) nor (W,H)=({w},{h})'
			)
		if arr.dtype != np.float32:
			arr = arr.astype(np.float32, copy=False)
	else:
		raise ValueError(
			'Offsets must be a 1D vector of length H or W, or a 2D array of shape (H,W)/(W,H)'
		)

	return np.ascontiguousarray(arr, dtype=np.float32)


@torch.no_grad()
def infer_prob_map(
	section: np.ndarray,
	*,
	amp: bool = True,
	tile: tuple[int, int] = (128, 6016),
	overlap: int = 32,
	tau: float = 1.0,
	offsets: np.ndarray | None = None,
) -> np.ndarray:
	"""Infer first-break probability for ``section``.

	When ``offsets`` are provided, they are used as a second channel
	"""
	arr = np.ascontiguousarray(section, dtype=np.float32)
	arr = zscore_per_trace_np(arr, axis=1, eps=1e-6)
	h, w = arr.shape
	model, device = _load_model()

	if offsets is None:
		x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
		logits = _run_tiled(
			model,
			x,
			tile=tile,
			overlap=overlap,
			amp=amp,
		)  # (1,1,H,W)
	else:
		offset_ch = make_offset_channel(offsets, h, w)
		stacked = np.stack((arr, offset_ch), axis=0)
		x = torch.from_numpy(stacked).unsqueeze(0).to(device)
		logits = _run_tiled(
			model,
			x,
			tile=tile,
			overlap=overlap,
			amp=amp,
			offset_channel=1,
		)
	# 学習と同じ: 時間軸に沿ってsoftmax
	prob = torch.softmax(logits.squeeze(1) / tau, dim=-1)  # (1,H,W)

	return prob.squeeze(0).detach().cpu().numpy().astype(np.float32)
