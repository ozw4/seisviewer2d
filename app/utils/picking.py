"""Picking utilities."""

import numpy as np


def sta_lta(trace: np.ndarray, win_sta: int, win_lta: int) -> np.ndarray:
	"""Compute classic STA/LTA ratio for a trace."""
	trace = np.asarray(trace, dtype=np.float32)
	if win_sta <= 0 or win_lta <= 0 or win_sta >= win_lta:
		msg = "invalid window lengths"
		raise ValueError(msg)
	pow_trace = trace ** 2
	sta = np.convolve(pow_trace, np.ones(win_sta) / win_sta, "valid")
	lta = np.convolve(pow_trace, np.ones(win_lta) / win_lta, "valid")
	return sta[win_lta - win_sta:] / (lta + 1e-12)


def pick_first_arrival(
	trace: np.ndarray,
	win_sta: int = 20,
	win_lta: int = 100,
	threshold: float = 3.0,
) -> int | None:
	"""Return the index of the first arrival based on STA/LTA."""
	ratio = sta_lta(trace, win_sta, win_lta)
	indices = np.where(ratio > threshold)[0]
	if len(indices) == 0:
		return None
	return int(indices[0] + win_lta)
