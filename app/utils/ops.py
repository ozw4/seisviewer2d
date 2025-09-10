"""Operator interfaces and registry for pipeline steps."""

from collections.abc import Callable  # noqa: F401
from typing import Any, Protocol

import numpy as np


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
    raise NotImplementedError


def op_denoise(
    x: np.ndarray,
    *,
    params: dict[str, Any],
    meta: dict[str, Any],

) -> np.ndarray:
    """Apply denoise transform."""
    raise NotImplementedError


def op_fbpick(
    x: np.ndarray,
    *,
    params: dict[str, Any],
    meta: dict[str, Any],

) -> dict[str, Any]:
    """Run fbpick analyzer."""
    raise NotImplementedError


TRANSFORMS: dict[str, Transform] = {"bandpass": op_bandpass, "denoise": op_denoise}
ANALYZERS: dict[str, Analyzer] = {"fbpick": op_fbpick}
