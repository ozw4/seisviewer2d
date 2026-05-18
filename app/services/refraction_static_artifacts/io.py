"""Atomic file writers for refraction static artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.services.refraction_static_artifacts.contract import (
    RefractionStaticArtifactError,
)


def _validate_no_object_arrays(
    arrays: dict[str, np.ndarray],
    *,
    artifact_name: str,
) -> None:
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise RefractionStaticArtifactError(
                f'{artifact_name}: object array is not allowed for {key}'
            )


def _write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    _validate_no_object_arrays(payload, artifact_name=path.name)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('wb') as handle:
            np.savez_compressed(handle, **payload)
        tmp_path.replace(path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RefractionStaticArtifactError(
            f'{path.name}: failed to write NPZ artifact'
        ) from exc


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _assert_strict_json(payload, artifact_name=path.name)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8') as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            handle.write('\n')
        tmp_path.replace(path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RefractionStaticArtifactError(
            f'{path.name}: failed to write JSON artifact'
        ) from exc


def _write_csv_atomic(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='raise')
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RefractionStaticArtifactError(
            f'{path.name}: failed to write CSV artifact'
        ) from exc


def _assert_strict_json(payload: dict[str, Any], *, artifact_name: str) -> None:
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticArtifactError(
            f'{artifact_name}: payload is not strict JSON serializable'
        ) from exc


__all__ = [
    '_assert_strict_json',
    '_validate_no_object_arrays',
    '_write_csv_atomic',
    '_write_json_atomic',
    '_write_npz_atomic',
]
