"""Backward-compatible import shim for refraction static preflight diagnostics."""

from importlib import import_module
import sys

_module = import_module('app.statics.refraction.application.preflight_diagnostics')
sys.modules[__name__] = _module
