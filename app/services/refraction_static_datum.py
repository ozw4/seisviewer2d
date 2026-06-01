"""Backward-compatible import shim for refraction datum statics helpers."""

from importlib import import_module
import sys

_module = import_module('app.statics.refraction.application.datum')
sys.modules[__name__] = _module
