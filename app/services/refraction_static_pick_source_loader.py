"""Backward-compatible import shim for refraction static pick-source helpers."""

from importlib import import_module
import sys

_module = import_module('app.statics.refraction.application.pick_source_loader')
sys.modules[__name__] = _module
