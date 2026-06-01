from __future__ import annotations

import importlib


REFRACTION_PACKAGE_MODULES = [
    'app.statics',
    'app.statics.common',
    'app.statics.refraction',
    'app.statics.refraction.contracts',
    'app.statics.refraction.domain',
    'app.statics.refraction.artifacts',
    'app.statics.refraction.application',
    'app.statics.refraction.ports',
    'app.statics.refraction.adapters',
    'app.statics.refraction.adapters.seisviewer2d',
    'app.statics.refraction.api',
]


def test_refraction_package_skeleton_imports() -> None:
    for module_name in REFRACTION_PACKAGE_MODULES:
        importlib.import_module(module_name)
