"""Backward-compatible import shim for refraction static artifacts."""

from app.statics.refraction import artifacts as _artifacts

for _name in dir(_artifacts):
    if not _name.startswith('_'):
        globals()[_name] = getattr(_artifacts, _name)

__all__ = _artifacts.__all__
