"""Backward-compatible import shim for refraction LSST export formatters."""

from importlib import import_module
import sys

_module = import_module('app.statics.refraction.application.lsst_export')
sys.modules[__name__] = _module
