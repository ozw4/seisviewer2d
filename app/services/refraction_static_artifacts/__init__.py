"""Public facade for refraction static artifact helpers."""

from __future__ import annotations

from app.services.refraction_static_artifacts import contract
from app.services.refraction_static_artifacts import _legacy
from app.services.refraction_static_artifacts import first_break_qc
from app.services.refraction_static_artifacts import reduced_time_qc
from app.services.refraction_static_artifacts import solution
from app.services.refraction_static_artifacts import writer
from app.services.refraction_static_artifacts._legacy import *  # noqa: F403

for _name in contract.__all__:
    globals()[_name] = getattr(contract, _name)

FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION = (
    contract.FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION
)
SIGN_CONVENTION = contract.SIGN_CONVENTION
TIME_TERM_SPREADSHEET_FORMAT_NAME = contract.TIME_TERM_SPREADSHEET_FORMAT_NAME
TIME_TERM_SPREADSHEET_FORMAT_VERSION = contract.TIME_TERM_SPREADSHEET_FORMAT_VERSION
TIME_TERM_SPREADSHEET_SCHEMA_VERSION = contract.TIME_TERM_SPREADSHEET_SCHEMA_VERSION

__all__ = list(_legacy.__all__)
for _module in (first_break_qc, reduced_time_qc):
    for _name in _module.__all__:
        globals()[_name] = getattr(_module, _name)
        if _name not in __all__:
            __all__.append(_name)
for _name in solution.__all__:
    globals()[_name] = getattr(solution, _name)
    if _name not in __all__:
        __all__.append(_name)
write_refraction_static_artifacts = writer.write_refraction_static_artifacts
if 'write_refraction_static_artifacts' not in __all__:
    _anchor = 'write_refraction_static_component_qc_artifacts'
    if _anchor in __all__:
        __all__.insert(
            __all__.index(_anchor),
            'write_refraction_static_artifacts',
        )
    else:
        __all__.append('write_refraction_static_artifacts')
