"""Shared input validation helpers."""

from __future__ import annotations

from numbers import Integral


def require_positive_int(value: object, name: str) -> int:
    """Return ``value`` as ``int`` if it is a positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an int, got {value!r}')
    int_value = int(value)
    if int_value <= 0:
        raise ValueError(f'{name} must be positive, got {int_value}')
    return int_value


def require_non_negative_int(value: object, name: str) -> int:
    """Return ``value`` as ``int`` if it is a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an int, got {value!r}')
    int_value = int(value)
    if int_value < 0:
        raise ValueError(f'{name} must be non-negative, got {int_value}')
    return int_value


__all__ = ['require_non_negative_int', 'require_positive_int']
