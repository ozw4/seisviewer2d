"""Backward-compatible import shim for refraction export services."""

from importlib import import_module
import sys

_module = import_module('app.statics.refraction.application.export_service')
sys.modules[__name__] = _module
