"""Shared validation helpers for API contract models."""

import math
from pathlib import Path


def _validate_artifact_basename(name: str, field_name: str) -> str:
    if not name:
        raise ValueError(f'{field_name} must be a non-empty file name')
    if name in {'.', '..'}:
        raise ValueError(f'{field_name} must be a plain file name')
    if Path(name).name != name:
        raise ValueError(f'{field_name} must be a plain file name')
    return name


def require_trace_header_byte(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be an integer SEG-Y trace header byte')
    if not isinstance(value, int):
        raise ValueError(f'{name} must be an integer SEG-Y trace header byte')
    if value < 1 or value > 240:
        raise ValueError(f'{name} must be in the range 1..240')
    return value


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a positive integer')
    if not isinstance(value, int):
        raise ValueError(f'{name} must be a positive integer')
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f'{name} must be a bool')
    return value


def _require_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be finite')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not math.isfinite(out):
        raise ValueError(f'{name} must be finite')
    return out


def _require_nonnegative_finite_float(value: object, name: str) -> float:
    out = _require_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f'{name} must be finite and >= 0')
    return out


def _require_positive_finite_float(value: object, name: str) -> float:
    out = _require_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f'{name} must be finite and > 0')
    return out


def _velocity_values_match(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-9, abs_tol=1.0e-9)
