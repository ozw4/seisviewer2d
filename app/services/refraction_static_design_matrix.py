"""Backward-compatible import shim for refraction static design matrix helpers."""

from importlib import import_module
import sys

_module = import_module('app.statics.refraction.application.design_matrix')
sys.modules[__name__] = _module
