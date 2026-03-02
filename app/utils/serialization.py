"""Serialization helpers."""

from __future__ import annotations

import numpy as np


def to_builtin(obj: object) -> object:
    """Recursively convert numpy containers to built-in Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {key: to_builtin(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(val) for val in obj]
    return obj


__all__ = ['to_builtin']
