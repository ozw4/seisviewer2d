"""Backward-compatible import shim for refraction static half-intercept helpers."""

from importlib import import_module
import sys

_module = import_module('app.statics.refraction.application.half_intercept')
sys.modules[__name__] = _module
