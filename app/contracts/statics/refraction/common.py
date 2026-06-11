"""Backward-compatible import shim for refraction statics contracts."""

from app.statics.refraction.contracts.common import *  # noqa: F403
from app.statics.refraction.contracts.common import (
    _REFRACTION_STATIC_LAYER_ORDER as _REFRACTION_STATIC_LAYER_ORDER,
)
