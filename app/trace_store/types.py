"""Trace-store public types."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class SectionView(NamedTuple):
    """Lightweight wrapper describing a section view."""

    arr: np.ndarray
    dtype: np.dtype
    scale: float | None = None


__all__ = ['SectionView']
