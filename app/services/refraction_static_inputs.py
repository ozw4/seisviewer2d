"""Backward-compatible import shim for refraction static input model helpers."""

from importlib import import_module
import sys

_module = import_module('app.statics.refraction.application.input_model')
sys.modules[__name__] = _module
