"""Backward-compatible import shim for refraction multi-layer apply helpers."""

from importlib import import_module
import sys

_module = import_module('app.statics.refraction.application.multilayer_service')
sys.modules[__name__] = _module
