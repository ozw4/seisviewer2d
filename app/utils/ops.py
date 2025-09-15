"""Operator interfaces and registry for pipeline steps."""

from collections.abc import Callable  # noqa: F401
from typing import Any, Protocol

import numpy as np
import torch

from .bandpass import bandpass_np
from .denoise import denoise_tensor
from .fbpick import infer_prob_map


class Transform(Protocol):
	"""Callable signature for transform operations."""

	def __call__(
		self,
		x: np.ndarray,
		*,
		params: dict[str, Any],
		meta: dict[str, Any],
	) -> np.ndarray:
		"""Run the transform."""
		...


class Analyzer(Protocol):
	"""Callable signature for analyzer operations."""

	def __call__(
		self,
		x: np.ndarray,
		*,
		params: dict[str, Any],
		meta: dict[str, Any],
	) -> dict[str, Any]:
		"""Run the analyzer."""
		...


def op_bandpass(
	x: np.ndarray,
	*,
	params: dict[str, Any],
	meta: dict[str, Any],
) -> np.ndarray:
	"""Apply bandpass transform."""
	_ = meta
	return bandpass_np(x, **params)


def op_denoise(
	x: np.ndarray,
	*,
	params: dict[str, Any],
	meta: dict[str, Any],
) -> np.ndarray:
	"""Apply denoise transform."""
	_ = meta
	x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).unsqueeze(0)
	y_t = denoise_tensor(x_t, **params)
	return y_t.squeeze(0).squeeze(0).numpy()


def op_fbpick(x: np.ndarray, *, params: dict, meta: dict) -> dict:
	"""Run fbpick analyzer with safe param mapping."""
	# map/clean params
	amp = params.get('amp', params.get('use_amp', True))
	overlap = int(params.get('overlap', 32))
	tau = float(params.get('tau', 1.0))

	# tile: prefer explicit 'tile'; else compose from 'chunk_h'/'tile_w' if both given
	tile = params.get('tile')
	if tile is None:
		ch = params.get('chunk_h')
		tw = params.get('tile_w')
		if ch is not None and tw is not None:
			tile = (int(ch), int(tw))
		# どちらか無ければ渡さない（infer_prob_map の既定に任せる）

	# ensure contiguous float32 [H, W]
	arr = np.ascontiguousarray(x, dtype=np.float32)

	kwargs = {'amp': bool(amp), 'overlap': overlap, 'tau': tau}
	if tile is not None:
		kwargs['tile'] = tuple(tile)

	prob = infer_prob_map(arr, **kwargs)
	return {'prob': prob}


TRANSFORMS: dict[str, Transform] = {'bandpass': op_bandpass, 'denoise': op_denoise}
ANALYZERS: dict[str, Analyzer] = {'fbpick': op_fbpick}
