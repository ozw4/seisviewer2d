"""Frequency-domain band-pass filter implemented with NumPy FFT."""

from __future__ import annotations

import numpy as np

__all__ = ['bandpass_np']


def bandpass_np(  # noqa: D417
	section: np.ndarray,
	*,
	low_hz: float,
	high_hz: float,
	dt: float,
	taper: float = 0.0,
) -> np.ndarray:
	"""Apply a zero-phase rectangular band-pass filter to ``section``.

	Each trace (rows) is filtered independently along the time axis
	(columns). ``section`` is expected to be ``(H, W)`` with dtype
	``float32``.

	Parameters
	----------
	section:
	Input array of shape ``(H, W)``.
	low_hz, high_hz:
	Band limits in Hz. ``low_hz <= 0`` disables the lower bound,
	``high_hz >= Nyquist`` disables the upper bound. ``low_hz`` must be
	strictly less than ``high_hz``.
	dt:
	Sampling interval in seconds.
	taper:
	Optional cosine taper width in Hz applied to the band edges. ``0``
	means a sharp rectangular filter.

	Returns
	-------
	numpy.ndarray
	Filtered array with the same shape and ``float32`` dtype as the
	input.

	Raises
	------
	ValueError
	If ``low_hz`` is not less than ``high_hz``.

	"""
	if low_hz >= high_hz:
		msg = 'low_hz must be less than high_hz'
		raise ValueError(msg)

	fs = 1.0 / dt
	nyq = fs / 2.0
	low_hz = max(0.0, low_hz)
	high_hz = min(nyq, high_hz)

	n = section.shape[1]
	freqs = np.fft.rfftfreq(n, dt)
	spec = np.fft.rfft(section, axis=1)
	mask = np.ones_like(freqs)

	# Apply lower bound
	if low_hz > 0.0:
		mask[freqs < low_hz] = 0.0
		if taper > 0.0:
			lo_taper = (low_hz, min(low_hz + taper, high_hz))
			idx = (freqs >= lo_taper[0]) & (freqs < lo_taper[1])
			sub = (freqs[idx] - lo_taper[0]) / taper
			mask[idx] = 0.5 * (1 - np.cos(np.pi * sub))

	# Apply upper bound
	if high_hz < nyq:
		mask[freqs > high_hz] = 0.0
		if taper > 0.0:
			hi_taper = (max(high_hz - taper, low_hz), high_hz)
			idx = (freqs > hi_taper[0]) & (freqs <= hi_taper[1])
			sub = (hi_taper[1] - freqs[idx]) / taper
			mask[idx] = 0.5 * (1 - np.cos(np.pi * sub))

	spec *= mask
	return np.fft.irfft(spec, n=n, axis=1).astype(np.float32)
