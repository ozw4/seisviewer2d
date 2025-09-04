"""First-break picking utilities.

This module assumes seismic sections are shaped ``[Trace, Sample]``. The
EdgeNeXt model consumes inputs shaped ``[1, 1, Trace, Sample]`` and outputs
probability maps in the same ``[Trace, Sample]`` format.
"""

from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp.autocast_mode import autocast

try:
	from scipy.ndimage import gaussian_filter1d, median_filter
	from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - optional dependency  # noqa: BLE001
	gaussian_filter1d = None  # type: ignore[assignment]
	median_filter = None  # type: ignore[assignment]
	savgol_filter = None  # type: ignore[assignment]

from .model import NetAE

_CACHE_DIR = Path('.cache/fbpick')


def build_fb_model(
	weights_path: Path = Path('./model/fbpick_edgenext_small.pth'),
	device: torch.device | None = None,
) -> nn.Module:
	"""Construct the EdgeNeXt-based first-break model.

	The model expects input tensors shaped ``[1, 1, Trace, Sample]`` and
	produces a probability map of the same shape after a sigmoid
	activation.
	"""
	device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	core = NetAE(
		backbone='edgenext_small.usi_in1k',
		pretrained=False,
		in_chans=1,
		out_chans=1,
	)
	ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
	state: dict[str, Any] = (
		ckpt.get('state_dict') or ckpt.get('model') or ckpt.get('model_ema') or ckpt
	)
	state = {k.removeprefix('module.'): v for k, v in state.items()}
	core.load_state_dict(state, strict=False)
	model = nn.Sequential(core, nn.Sigmoid())
	model.to(device)
	model.eval()
	return model


@torch.no_grad()
def infer_prob_map(
	section: np.ndarray,
	*,
	weights_path: Path = Path('./model/fbpick_edgenext_small.pth'),
	device: torch.device | None = None,
	batch_traces: int = 256,
	dtype: np.dtype = np.float16,
) -> tuple[np.ndarray, dict[str, int]]:
	"""Run inference on ``section`` and return probability map and metadata.

	Parameters
	----------
	section:
		Input section shaped ``[Trace, Sample]``.
	weights_path:
		Location of model weights.
	device:
		Torch device for inference.
	batch_traces:
		Number of traces per inference batch.
	dtype:
		Output dtype for the probability map.

	Returns
	-------
	probs, meta:
		Probability map shaped ``[Trace, Sample]`` and metadata with the
		original sizes.

	"""
	nt, ns = section.shape
	x = np.asarray(section, dtype=np.float32, order='C')
	mean = x.mean(axis=1, keepdims=True)
	std = x.std(axis=1, keepdims=True)
	std[std == 0] = 1.0
	x = (x - mean) / std
	device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = build_fb_model(weights_path=weights_path, device=device)
	probs = torch.empty((nt, ns), dtype=torch.float32)
	x_tensor = torch.from_numpy(x)
	for s in range(0, nt, batch_traces):
		e = min(s + batch_traces, nt)
		xb = x_tensor[s:e, :].unsqueeze(0).unsqueeze(0)
		with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
			yb = model(xb.to(device))
		probs[s:e, :] = yb.squeeze(0).squeeze(0).cpu()
	probs_np = probs.numpy().astype(dtype, copy=False)
	meta = {'ns': int(ns), 'nt': int(nt)}
	return probs_np, meta


def picks_from_prob(  # noqa: PLR0913, PLR0912, C901
	probs: np.ndarray,
	*,
	dt_us: int = 1000,
	t0_us: int = 0,
	method: str = 'argmax',
	median_kernel: int = 5,
	gaussian_sigma: float | None = None,
	sg_window: int | None = None,
	sg_poly: int = 2,
	conf_threshold: float | None = None,
	max_jump: int | None = None,
) -> tuple[list[dict[str, int]], dict[str, float]]:
	"""Extract pick times from a probability map.

	The input ``probs`` is expected to be shaped ``[Trace, Sample]``. Optional
	smoothing operations act across traces (axis 0) before picking the
	argmax along the sample axis (axis 1).
	"""
	p = np.asarray(probs, dtype=np.float32)
	if median_kernel > 1 and median_filter is not None:
		p = median_filter(p, size=(median_kernel, 1))
	if gaussian_sigma is not None and gaussian_filter1d is not None:
		p = gaussian_filter1d(p, gaussian_sigma, axis=0)
	if sg_window is not None and sg_window > 1 and savgol_filter is not None:
		with contextlib.suppress(ValueError):
			p = savgol_filter(p, sg_window, sg_poly, axis=0)
	if method != 'argmax':
		raise NotImplementedError(method)
	idx = p.argmax(axis=1)
	conf = p[np.arange(p.shape[0]), idx]
	valid = np.ones(p.shape[0], dtype=bool)
	if max_jump is not None and p.shape[0] > 0:
		prev = idx[0]
		for i in range(1, p.shape[0]):
			if abs(int(idx[i]) - int(prev)) > max_jump:
				valid[i] = False
			else:
				prev = idx[i]
	if conf_threshold is not None:
		valid &= conf >= conf_threshold
	picks: list[dict[str, int]] = []
	for tr in range(p.shape[0]):
		if valid[tr]:
			sample = int(idx[tr])
			t_us = int(t0_us + sample * dt_us)
		else:
			sample = -1
			t_us = -1
		picks.append({'trace': tr, 'sample': sample, 't_us': t_us})
	conf_valid = conf[valid]
	if conf_valid.size:
		aux = {
			'conf_mean': float(conf_valid.mean()),
			'conf_min': float(conf_valid.min()),
			'conf_max': float(conf_valid.max()),
		}
	else:
		aux = {'conf_mean': 0.0, 'conf_min': 0.0, 'conf_max': 0.0}
	return picks, aux


def _cache_key(
	path: str | Path,
	axis: int,
	index: int,
	weights_path: str | Path,
	shape: tuple[int, int],
) -> str:
	"""Return cache identifier based on input parameters."""
	m = hashlib.md5()  # noqa: S324
	m.update(str(path).encode('utf-8'))
	m.update(str(axis).encode('utf-8'))
	m.update(str(index).encode('utf-8'))
	m.update(str(weights_path).encode('utf-8'))
	m.update(str(shape).encode('utf-8'))
	return m.hexdigest()


def save_cached_prob(cache_id: str, probs: np.ndarray) -> None:
	"""Persist probability map to the cache."""
	_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	np.save(_CACHE_DIR / f'{cache_id}.npy', probs)


def load_cached_prob(cache_id: str) -> np.ndarray | None:
	"""Load probability map from cache if available."""
	path = _CACHE_DIR / f'{cache_id}.npy'
	if path.exists():
		return np.load(path)
	return None
