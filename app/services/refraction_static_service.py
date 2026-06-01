"""Backward-compatible import shim for refraction static apply workflow."""

from importlib import import_module
import sys

_module = import_module('app.statics.refraction.application.workflow')
sys.modules[__name__] = _module
