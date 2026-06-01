"""Backward-compatible import shim for static-table apply services."""

from importlib import import_module
import sys

_module = import_module('app.statics.refraction.application.table_apply_service')
sys.modules[__name__] = _module
