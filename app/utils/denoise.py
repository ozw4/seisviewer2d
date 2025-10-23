"""Noise suppression model wrapper."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch

from .model import NetAE
from .predict import cover_all_traces_predict_chunked
from .signal_utils import denorm_per_trace_tensor_b1hw, zscore_per_trace_tensor_b1hw

__all__ = ['denoise_tensor', 'get_model']

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_MODEL_PATH = (
	Path(__file__).resolve().parents[2]
	/ 'model'
	/ 'recon_replace_eq_edgenext_small.pth'
)
print(f'Loading model from {_MODEL_PATH}, device: {_DEVICE}')


@lru_cache(maxsize=1)
def _load_model() -> torch.nn.Module:
	print('Model loading...')
	model = NetAE(
		backbone='edgenext_small.usi_in1k',
		pretrained=False,
		stage_strides=[(2, 4), (2, 2), (2, 4), (2, 2)],
		pre_stages=2,
		pre_stage_strides=((1, 1), (1, 2)),
	)
	checkpoint = torch.load(_MODEL_PATH, map_location=_DEVICE, weights_only=False)

	model.load_state_dict(checkpoint['model_ema'], strict=True)
	model.to(_DEVICE)
	model.eval()
	print('Model loaded.')
	return model


def get_model() -> torch.nn.Module:
	"""Return the singleton model instance."""
	return _load_model()


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
	model = get_model()
	xt = xn.to(_DEVICE)
	yt = cover_all_traces_predict_chunked(
		model,
		xt,
		chunk_h=chunk_h,
		overlap=overlap,
		mask_ratio=mask_ratio,
		noise_std=noise_std,
		mask_noise_mode=mask_noise_mode,
		use_amp=use_amp,
		device=_DEVICE,
		seed=seed,
		passes_batch=passes_batch,
	)
	y_restored = denorm_per_trace_tensor_b1hw(
		yt, mean=mean.to(yt.device), inv_std=inv_std.to(yt.device)
	)
	return y_restored.cpu()
