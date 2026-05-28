"""Atomic file writers for refraction static artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.services.common.artifact_io import (
    assert_strict_json,
    validate_npz_no_object_arrays,
    write_csv_atomic,
    write_json_atomic,
    write_npz_atomic,
)
from app.services.refraction_static_artifacts.contract import (
    RefractionStaticArtifactError,
)


def _artifact_error(message: str, _exc: BaseException) -> RefractionStaticArtifactError:
    return RefractionStaticArtifactError(message)


def _validate_no_object_arrays(
    arrays: dict[str, np.ndarray],
    *,
    artifact_name: str,
) -> None:
    try:
        validate_npz_no_object_arrays(arrays, artifact_name=artifact_name)
    except ValueError as exc:
        raise RefractionStaticArtifactError(str(exc)) from exc


def _write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    _validate_no_object_arrays(payload, artifact_name=path.name)
    write_npz_atomic(
        path,
        payload,
        compressed=True,
        reject_object_arrays=False,
        error_factory=_artifact_error,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _assert_strict_json(payload, artifact_name=path.name)
    write_json_atomic(
        path,
        payload,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        trailing_newline=True,
        error_factory=_artifact_error,
    )


def _write_csv_atomic(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    write_csv_atomic(
        path,
        columns=columns,
        rows=rows,
        extrasaction='raise',
        lineterminator='\r\n',
        error_factory=_artifact_error,
    )


def _assert_strict_json(payload: dict[str, Any], *, artifact_name: str) -> None:
    try:
        assert_strict_json(payload, artifact_name=artifact_name)
    except ValueError as exc:
        raise RefractionStaticArtifactError(str(exc)) from exc


__all__ = [
    '_assert_strict_json',
    '_validate_no_object_arrays',
    '_write_csv_atomic',
    '_write_json_atomic',
    '_write_npz_atomic',
]
