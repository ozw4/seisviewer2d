"""Reusable atomic artifact writers for service modules."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from uuid import uuid4

import numpy as np


ErrorFactory = Callable[[str, BaseException], Exception]


def atomic_write(
    path: Path,
    writer: Callable[[Path], None],
    *,
    make_parent: bool = True,
    tmp_suffix: str = '.tmp',
    error_factory: ErrorFactory | None = None,
) -> None:
    """Write a sibling temp file and replace the target only after success."""
    if make_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}{tmp_suffix}')
    try:
        writer(tmp_path)
        tmp_path.replace(path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        if error_factory is not None:
            message = f'{path.name}: failed to write artifact'
            raise error_factory(message, exc) from exc
        raise


def assert_strict_json(
    payload: object,
    *,
    artifact_name: str | None = None,
) -> None:
    """Raise when a payload cannot be serialized as strict JSON."""
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        prefix = f'{artifact_name}: ' if artifact_name else ''
        raise ValueError(
            f'{prefix}payload is not strict JSON serializable'
        ) from exc


def validate_npz_no_object_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    artifact_name: str | None = None,
) -> None:
    """Raise when an NPZ payload contains arrays requiring pickle support."""
    for key, value in arrays.items():
        if np.asarray(value).dtype.hasobject:
            prefix = f'{artifact_name}: ' if artifact_name else ''
            raise ValueError(f'{prefix}object array is not allowed for {key}')


def write_json_atomic(
    path: Path,
    payload: object,
    *,
    allow_nan: bool = False,
    ensure_ascii: bool = True,
    indent: int | None = None,
    sort_keys: bool = True,
    trailing_newline: bool = False,
    make_parent: bool = True,
    error_factory: ErrorFactory | None = None,
) -> None:
    """Atomically write a JSON artifact."""

    def write(tmp_path: Path) -> None:
        with tmp_path.open('w', encoding='utf-8') as handle:
            json.dump(
                payload,
                handle,
                allow_nan=allow_nan,
                ensure_ascii=ensure_ascii,
                indent=indent,
                sort_keys=sort_keys,
            )
            if trailing_newline:
                handle.write('\n')

    atomic_write(
        path,
        write,
        make_parent=make_parent,
        error_factory=_typed_error_factory('JSON', path, error_factory),
    )


def write_csv_atomic(
    path: Path,
    *,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    extrasaction: str = 'raise',
    make_parent: bool = True,
    error_factory: ErrorFactory | None = None,
) -> None:
    """Atomically write a CSV artifact with explicit column order."""

    def write(tmp_path: Path) -> None:
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
                extrasaction=extrasaction,
                lineterminator='\n',
            )
            writer.writeheader()
            writer.writerows(rows)

    atomic_write(
        path,
        write,
        make_parent=make_parent,
        error_factory=_typed_error_factory('CSV', path, error_factory),
    )


def write_npz_atomic(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    compressed: bool = False,
    reject_object_arrays: bool = True,
    make_parent: bool = True,
    error_factory: ErrorFactory | None = None,
) -> None:
    """Atomically write an NPZ artifact."""
    if reject_object_arrays:
        try:
            validate_npz_no_object_arrays(arrays, artifact_name=path.name)
        except Exception as exc:
            if error_factory is not None:
                message = f'{path.name}: failed to write NPZ artifact'
                raise error_factory(message, exc) from exc
            raise

    def write(tmp_path: Path) -> None:
        with tmp_path.open('wb') as handle:
            if compressed:
                np.savez_compressed(handle, **arrays)
            else:
                np.savez(handle, **arrays)

    atomic_write(
        path,
        write,
        make_parent=make_parent,
        error_factory=_typed_error_factory('NPZ', path, error_factory),
    )


def _typed_error_factory(
    artifact_type: str,
    path: Path,
    error_factory: ErrorFactory | None,
) -> ErrorFactory | None:
    if error_factory is None:
        return None

    def make_error(_: str, exc: BaseException) -> Exception:
        message = f'{path.name}: failed to write {artifact_type} artifact'
        return error_factory(message, exc)

    return make_error
