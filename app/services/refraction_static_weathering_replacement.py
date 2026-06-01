"""Backward-compatible import shim for refraction static replacement helpers."""

from importlib import import_module
import sys

_module = import_module('app.statics.refraction.application.weathering_replacement')
sys.modules[__name__] = _module
