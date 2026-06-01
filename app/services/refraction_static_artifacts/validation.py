"""Backward-compatible import shim for refraction static artifacts."""

import sys

from app.statics.refraction.artifacts import validation as _module

sys.modules[__name__] = _module
