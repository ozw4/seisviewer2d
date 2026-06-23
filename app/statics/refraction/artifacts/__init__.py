"""Public facade for refraction static artifact helpers."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path
from typing import Any


_WRITER_MODULE = 'app.statics.refraction.artifacts.writer'
_CONTRACT_MODULE = 'app.statics.refraction.artifacts.contract'
_LAZY_MODULES = {
    'contract': _CONTRACT_MODULE,
    'writer': _WRITER_MODULE,
    'registry': 'app.statics.refraction.artifacts.registry',
}
_CONTRACT_EXPORTS = {
    'FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION',
    'SIGN_CONVENTION',
    'TIME_TERM_SPREADSHEET_FORMAT_NAME',
    'TIME_TERM_SPREADSHEET_FORMAT_VERSION',
    'TIME_TERM_SPREADSHEET_SCHEMA_VERSION',
}
_FALLBACK_EXPORT_MODULES = (
    'app.statics.refraction.artifacts.components',
    'app.statics.refraction.artifacts.cell_velocity',
    'app.statics.refraction.artifacts.grid_map',
    'app.statics.refraction.artifacts.line_profile',
)


def _load_writer_all() -> list[str]:
    writer_source = Path(__file__).with_name('writer.py').read_text(encoding='utf-8')
    writer_tree = ast.parse(writer_source)
    for node in writer_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == '__all__'
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        return [str(item) for item in value]
    return []


__all__ = _load_writer_all()


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        value = import_module(_LAZY_MODULES[name])
        globals()[name] = value
        return value
    if name in __all__:
        value = getattr(import_module(_WRITER_MODULE), name)
        globals()[name] = value
        return value
    contract_module = import_module(_CONTRACT_MODULE)
    if name in _CONTRACT_EXPORTS or hasattr(contract_module, name):
        value = getattr(contract_module, name)
        globals()[name] = value
        return value
    for module_name in _FALLBACK_EXPORT_MODULES:
        module = import_module(module_name)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_CONTRACT_EXPORTS))
