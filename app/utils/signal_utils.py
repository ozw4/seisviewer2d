import numpy as np
import torch


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
	x: torch.Tensor, eps: float = 1e-6, inplace: bool = False
) -> torch.Tensor:
	"""Z-score normalize per trace for x shaped (B, 1, H, W),
	where H = #traces, W = #samples per trace.
	Normalization is done along dim=-1 (W) for each (B, 1, H) slice.

	Args:
		x: Tensor of shape (B, 1, H, W). Prefer float32.
		eps: Small floor for std to avoid division by zero.
		inplace: If True, modify x in-place (may conflict with autograd).

	Returns:
		Tensor of same shape, z-scored per trace.

	"""
	if x.dtype != torch.float32:
		x = x.to(torch.float32)  # keep it simple & stable

	mean = x.mean(dim=-1, keepdim=True)
	std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)

	if inplace:
		x.sub_(mean).div_(std)
		return x
	return (x - mean) / std
