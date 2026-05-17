"""Public facade for refraction static artifact helpers."""

from __future__ import annotations

from app.services.refraction_static_artifacts import contract
from app.services.refraction_static_artifacts import _legacy
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
