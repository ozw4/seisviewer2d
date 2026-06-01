"""Backward-compatible import shim for refraction static bedrock helpers."""

from importlib import import_module
import sys

_module = import_module('app.statics.refraction.application.bedrock')
sys.modules[__name__] = _module
