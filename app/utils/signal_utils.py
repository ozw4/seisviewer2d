import os

import numpy as np
import torch

_NORM_EPS = float(os.getenv('NORM_EPS', '1e-6'))


def zscore_per_trace_np(
	arr: np.ndarray, axis: int = 1, eps: float = 1e-6
) -> np.ndarray:
	"""Per-trace z-score (axis=1: samples次元) を想定。
	arr は (traces, samples) float32。
	"""
	if arr.dtype != np.float32:
		arr = arr.astype(np.float32, copy=True)
	elif not arr.flags.writeable:
		arr = arr.copy()
	mu = arr.mean(axis=axis, keepdims=True, dtype=np.float32)
	sd = arr.std(axis=axis, keepdims=True, dtype=np.float32)
	sd[sd < eps] = 1.0
	arr -= mu
	arr /= sd
	return arr


def zscore_per_trace_tensor_b1hw(
	x: torch.Tensor,
	*,
	inplace: bool = False,
	eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Per-trace z-score along W (samples) for input x of shape (B,1,H,W).
	- H: number of traces
	- W: number of samples per trace
	Returns:
	  xn:      normalized tensor, shape (B,1,H,W)
	  mean:    per-trace mean,   shape (B,1,H,1)
	  inv_std: per-trace 1/std,  shape (B,1,H,1)
	"""
	if x.ndim != 4 or x.size(1) != 1:
		raise ValueError('expected x shape (B,1,H,W)')

	eps_val = float(_NORM_EPS if eps is None else eps)
	device, dtype = x.device, x.dtype

	# 統計は W 次元（dim=3）で計算（各トレースごと）
	# mean:(B,1,H,1), std:(B,1,H,1)
	mean = x.mean(dim=3, keepdim=True)
	# var = E[(x-mean)^2]（不偏なし）→ sqrt
	std = (x - mean).pow(2).mean(dim=3, keepdim=True)
	std.sqrt_()  # in-place sqrt
	std.clamp_min_(eps_val)  # 0 除算回避（ε 導入）

	inv_std = std.reciprocal()  # 逆数を前計算（以後は乗算のみ）

	# 正規化（in-place / 非 in-place 選択）
	xn = x if inplace else x.clone()
	xn.add_(-mean).mul_(inv_std)
	return xn, mean, inv_std


@torch.no_grad()
def denorm_per_trace_tensor_b1hw(
	y: torch.Tensor,
	mean: torch.Tensor,
	inv_std: torch.Tensor,
	*,
	inplace: bool = False,
) -> torch.Tensor:
	"""
	Inverse of per-trace z-score: y*std + mean, where std = 1/inv_std.
	All tensors are (B,1,H,*) with mean/inv_std shaped (B,1,H,1).
	"""
	if y.ndim != 4 or mean.ndim != 4 or inv_std.ndim != 4:
		raise ValueError('expected (B,1,H,*) tensors for y, mean, inv_std')

	std = inv_std.reciprocal()
	out = y if inplace else y.clone()
	out.mul_(std).add_(mean)
	return out
