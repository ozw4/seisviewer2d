"""Backward-compatible import shim for refraction static domain helpers."""

from importlib import import_module
import sys

_module = import_module('app.statics.refraction.domain.solver')
sys.modules[__name__] = _module
