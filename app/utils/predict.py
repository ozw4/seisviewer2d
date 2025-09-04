import math
from typing import Literal

import numpy as np
import torch
from torch.amp.autocast_mode import autocast

__all__ = [
	'cover_all_traces_predict',
	'cover_all_traces_predict_chunked',
	'predict_section_fbpick',
]


@torch.no_grad()
def cover_all_traces_predict(
	model: torch.nn.Module,
	x: torch.Tensor,
	*,
	mask_ratio: float = 0.5,
	noise_std: float = 1.0,
	mask_noise_mode: Literal['replace', 'add'] = 'replace',
	use_amp: bool = True,
	device=None,
	seed: int | None = 12345,
	passes_batch: int = 4,
) -> torch.Tensor:
	"""Predict each trace by covering all traces once.

	Args:
		mask_noise_mode: replace to overwrite, add to perturb traces.

	"""
	assert x.dim() == 4 and x.size(1) == 1, 'x must be (B,1,H,W)'
	device = device or x.device
	B, _, H, W = x.shape
	m = max(1, min(int(round(mask_ratio * H)), H - 1))
	K = math.ceil(H / m)
	y_full = torch.empty_like(x)
	for b in range(B):
		if seed is not None:
			g = torch.Generator(device='cpu').manual_seed(seed + b)
			perm = torch.randperm(H, generator=g)
		else:
			perm = torch.randperm(H)
		chunks = [perm[i : i + m] for i in range(0, H, m)]
		for s in range(0, K, passes_batch):
			batch_chunks = chunks[s : s + passes_batch]
			xmb = []
			for idxs in batch_chunks:
				xm = x[b : b + 1].clone()
				if seed is not None:
					gk = torch.Generator(device='cpu').manual_seed(
						(seed + b) * 100003 + s * 1009 + int(idxs[0])
					)
					n = (
						torch.randn((1, 1, len(idxs), W), generator=gk, device='cpu')
						* noise_std
					)
				else:
					n = torch.randn((1, 1, len(idxs), W), device='cpu') * noise_std
				n = n.to(device=device, non_blocking=True)
				idxs_dev = idxs.to(device)
				if mask_noise_mode == 'replace':
					xm[:, :, idxs_dev, :] = n
				elif mask_noise_mode == 'add':
					xm[:, :, idxs_dev, :] += n
				else:
					raise ValueError(f'Invalid mask_noise_mode: {mask_noise_mode}')
				xmb.append(xm)
			xmb = torch.cat(xmb, dim=0)
			dev_type = 'cuda' if xmb.is_cuda else 'cpu'
			with autocast(device_type=dev_type, enabled=use_amp):
				yb = model(xmb)
			for k, idxs in enumerate(batch_chunks):
				y_full[b, :, idxs.to(device), :] = yb[k, :, idxs.to(device), :]
	return y_full


@torch.no_grad()
def cover_all_traces_predict_chunked(
	model: torch.nn.Module,
	x: torch.Tensor,
	*,
	chunk_h: int = 128,
	overlap: int = 32,
	mask_ratio: float = 0.5,
	noise_std: float = 1.0,
	mask_noise_mode: Literal['replace', 'add'] = 'replace',
	use_amp: bool = True,
	device=None,
	seed: int = 12345,
	passes_batch: int = 4,
) -> torch.Tensor:
	"""Apply cover_all_traces_predict on tiled H-axis chunks.

	Args:
		mask_noise_mode: replace to overwrite, add to perturb traces.

	"""
	assert overlap < chunk_h, 'overlap は chunk_h より小さくしてください'
	device = device or x.device
	B, _, H, W = x.shape
	y_acc = torch.zeros_like(x)
	w_acc = torch.zeros((B, 1, H, 1), dtype=x.dtype, device=device)
	step = chunk_h - overlap
	s = 0
	while s < H:
		e = min(s + chunk_h, H)
		xt = x[:, :, s:e, :]
		yt = cover_all_traces_predict(
			model,
			xt,
			mask_ratio=mask_ratio,
			noise_std=noise_std,
			mask_noise_mode=mask_noise_mode,
			use_amp=use_amp,
			device=device,
			seed=seed + s,
			passes_batch=passes_batch,
		)
		h_t = e - s
		w = torch.ones((1, 1, h_t, 1), dtype=x.dtype, device=device)
		left_ov = min(overlap, s)
		right_ov = min(overlap, H - e)
		if left_ov > 0:
			ramp = torch.linspace(
				0, 1, steps=left_ov, device=device, dtype=x.dtype
			).view(1, 1, -1, 1)
			w[:, :, :left_ov, :] = ramp
		if right_ov > 0:
			ramp = torch.linspace(
				1, 0, steps=right_ov, device=device, dtype=x.dtype
			).view(1, 1, -1, 1)
			w[:, :, -right_ov:, :] = ramp
		y_acc[:, :, s:e, :] += yt * w
		w_acc[:, :, s:e, :] += w
		if e == H:
			break
		s += step
	y_full = y_acc / (w_acc + 1e-8)
	return y_full


@torch.no_grad()
def predict_section_fbpick(
	section_f32: np.ndarray,
	model: torch.nn.Module,
	device: torch.device,
	chunk_h: int,
	overlap: int,
	*,
	amp: bool = True,
	smooth_window: int = 0,
) -> np.ndarray:
	"""Predict first-arrival probabilities for a section.

	The section is tiled along the time axis (H) with overlapping windows
	and blended using linear ramps, mirroring the denoise pipeline.
	"""
	if overlap >= chunk_h:
		msg = 'overlap must be smaller than chunk_h'
		raise ValueError(msg)

	xt = torch.from_numpy(section_f32).unsqueeze(0).unsqueeze(0).to(device)
	B, _, H, W = xt.shape
	y_acc = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
	w_acc = torch.zeros((B, 1, H, 1), device=device, dtype=torch.float32)
	step = chunk_h - overlap
	s = 0
	while s < H:
		e = min(s + chunk_h, H)
		xt_chunk = xt[:, :, s:e, :]
		dev_type = 'cuda' if device.type == 'cuda' else 'cpu'
		with autocast(device_type=dev_type, enabled=amp):
			logits = model(xt_chunk)
			probs = torch.sigmoid(logits)
		h_t = e - s
		w = torch.ones((1, 1, h_t, 1), device=device, dtype=torch.float32)
		left_ov = min(overlap, s)
		right_ov = min(overlap, H - e)
		if left_ov > 0:
			ramp = torch.linspace(
				0, 1, steps=left_ov, device=device, dtype=torch.float32
			).view(1, 1, -1, 1)
			w[:, :, :left_ov, :] = ramp
		if right_ov > 0:
			ramp = torch.linspace(
				1, 0, steps=right_ov, device=device, dtype=torch.float32
			).view(1, 1, -1, 1)
			w[:, :, -right_ov:, :] = ramp
		y_acc[:, :, s:e, :] += probs * w
		w_acc[:, :, s:e, :] += w
		if e == H:
			break
		s += step
	y_full = y_acc / (w_acc + 1e-8)
	prob = y_full.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
	if smooth_window > 1:
		kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
		prob = np.apply_along_axis(
			lambda m: np.convolve(m, kernel, mode='same'), 0, prob
		)
	prob = np.clip(prob, 1e-4, 1.0 - 1e-4)
	H_expected, W_expected = section_f32.shape
	if prob.shape == (W_expected, H_expected):
		prob = prob.T
	assert prob.shape == (H_expected, W_expected)
	return prob
