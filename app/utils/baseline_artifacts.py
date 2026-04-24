"""Helpers for split baseline manifest and array artifacts."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

BASELINE_STAGE_RAW = 'raw'
BASELINE_ARTIFACT_VERSION = 2
LEGACY_BASELINE_FILENAME_RAW = 'baseline_raw.json'
BASELINE_MANIFEST_KEYS = (
    'key1_values',
    'mu_section_by_key1',
    'sigma_section_by_key1',
    'trace_spans_by_key1',
)
BASELINE_ARRAY_KEYS = ('mu_traces', 'sigma_traces', 'zero_var_mask')
BASELINE_ARTIFACT_ID_FIELD = 'artifact_id'
SPLIT_READ_RETRY_COUNT = 5
SPLIT_READ_RETRY_SECONDS = 0.01


class SplitBaselineArtifactsError(RuntimeError):
    """Raised when split baseline artifacts cannot be read consistently."""


def build_baseline_stem(*, stage: str, key1_byte: int, key2_byte: int) -> str:
    return f'baseline_{stage}.k1_{int(key1_byte)}.k2_{int(key2_byte)}'


def build_baseline_manifest_path(
    store_path: str | Path,
    *,
    stage: str,
    key1_byte: int,
    key2_byte: int,
) -> Path:
    return Path(store_path) / f'{build_baseline_stem(stage=stage, key1_byte=key1_byte, key2_byte=key2_byte)}.json'


def build_baseline_npz_path(
    store_path: str | Path,
    *,
    stage: str,
    key1_byte: int,
    key2_byte: int,
) -> Path:
    return Path(store_path) / f'{build_baseline_stem(stage=stage, key1_byte=key1_byte, key2_byte=key2_byte)}.npz'


def build_legacy_baseline_path(store_path: str | Path) -> Path:
    return Path(store_path) / LEGACY_BASELINE_FILENAME_RAW


def split_baseline_payload(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = dict(payload)
    arrays: dict[str, np.ndarray] = {
        'mu_traces': np.asarray(payload.get('mu_traces'), dtype=np.float32),
        'sigma_traces': np.asarray(payload.get('sigma_traces'), dtype=np.float32),
        'zero_var_mask': np.asarray(payload.get('zero_var_mask'), dtype=bool),
    }
    for key in BASELINE_ARRAY_KEYS:
        manifest.pop(key, None)
    return manifest, arrays


def _normalize_artifact_id(value: Any) -> str | None:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    text = str(value).strip()
    return text or None


def _load_json_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise SplitBaselineArtifactsError(
            f'Corrupted baseline manifest payload: {path}'
        ) from exc
    if not isinstance(payload, dict):
        raise SplitBaselineArtifactsError(
            f'Baseline manifest payload must be an object: {path}'
        )
    return payload


def _payload_matches_stage_and_key_bytes(
    payload: dict[str, Any],
    *,
    stage: str,
    key1_byte: int,
    key2_byte: int,
) -> bool:
    if payload.get('stage') != stage:
        return False
    try:
        stored_key1 = int(payload['key1_byte'])
        stored_key2 = int(payload['key2_byte'])
    except (KeyError, TypeError, ValueError):
        return False
    return stored_key1 == int(key1_byte) and stored_key2 == int(key2_byte)


def _payload_matches_source_sha(
    payload: dict[str, Any], *, source_sha256: str | None
) -> bool:
    stored = payload.get('source_sha256')
    if source_sha256 is None:
        return stored is None
    return stored == source_sha256


def _build_split_cache_key(
    manifest_path: Path,
    npz_path: Path,
    *,
    artifact_id: str | None,
) -> str:
    if artifact_id is not None:
        return f'split|{manifest_path.resolve()}|{artifact_id}'
    return (
        f'split|{manifest_path.resolve()}|{manifest_path.stat().st_mtime_ns}|'
        f'{npz_path.resolve()}|{npz_path.stat().st_mtime_ns}'
    )


def read_split_baseline_payload(
    store_path: str | Path,
    *,
    stage: str,
    key1_byte: int,
    key2_byte: int,
    include_arrays: bool,
) -> tuple[dict[str, Any], str] | None:
    manifest_path = build_baseline_manifest_path(
        store_path,
        stage=stage,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    npz_path = build_baseline_npz_path(
        store_path,
        stage=stage,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    if not manifest_path.is_file() or not npz_path.is_file():
        return None
    for attempt in range(SPLIT_READ_RETRY_COUNT):
        manifest = _load_json_payload(manifest_path)
        if not _payload_matches_stage_and_key_bytes(
            manifest,
            stage=stage,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
        ):
            return None
        missing_manifest_keys = [
            key for key in BASELINE_MANIFEST_KEYS if key not in manifest
        ]
        if missing_manifest_keys:
            raise SplitBaselineArtifactsError(
                'Baseline manifest missing keys: '
                + ', '.join(sorted(missing_manifest_keys))
            )
        try:
            with np.load(npz_path, allow_pickle=False) as arrays:
                missing_array_keys = [
                    key for key in BASELINE_ARRAY_KEYS if key not in arrays.files
                ]
                if missing_array_keys:
                    raise SplitBaselineArtifactsError(
                        'Baseline array payload missing keys: '
                        + ', '.join(sorted(missing_array_keys))
                    )
                manifest_artifact_id = _normalize_artifact_id(
                    manifest.get(BASELINE_ARTIFACT_ID_FIELD)
                )
                npz_artifact_id = _normalize_artifact_id(
                    arrays[BASELINE_ARTIFACT_ID_FIELD]
                ) if BASELINE_ARTIFACT_ID_FIELD in arrays.files else None
                if (
                    manifest_artifact_id is not None
                    or npz_artifact_id is not None
                ) and manifest_artifact_id != npz_artifact_id:
                    if attempt + 1 >= SPLIT_READ_RETRY_COUNT:
                        raise SplitBaselineArtifactsError(
                            'Split baseline artifacts are mid-update'
                        )
                    time.sleep(SPLIT_READ_RETRY_SECONDS)
                    continue
                payload = dict(manifest)
                if include_arrays:
                    payload['mu_traces'] = np.asarray(
                        arrays['mu_traces'], dtype=np.float32
                    )
                    payload['sigma_traces'] = np.asarray(
                        arrays['sigma_traces'], dtype=np.float32
                    )
                    payload['zero_var_mask'] = np.asarray(
                        arrays['zero_var_mask'], dtype=bool
                    )
        except SplitBaselineArtifactsError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SplitBaselineArtifactsError(
                f'Corrupted baseline array payload: {npz_path}'
            ) from exc
        return payload, _build_split_cache_key(
            manifest_path,
            npz_path,
            artifact_id=manifest_artifact_id,
        )
    return None


def merge_baseline_payload(
    manifest: dict[str, Any],
    *,
    mu_traces: np.ndarray,
    sigma_traces: np.ndarray,
    zero_var_mask: np.ndarray,
) -> dict[str, Any]:
    payload = dict(manifest)
    payload['mu_traces'] = np.asarray(mu_traces, dtype=np.float32).tolist()
    payload['sigma_traces'] = np.asarray(sigma_traces, dtype=np.float32).tolist()
    payload['zero_var_mask'] = np.asarray(zero_var_mask, dtype=bool).tolist()
    return payload


def build_trace_spans_by_key1(
    key1_values: np.ndarray,
    key1_offsets: np.ndarray,
    key1_counts: np.ndarray,
) -> dict[str, list[list[int]]]:
    return {
        str(int(value)): [[int(offset), int(offset + count)]]
        for value, offset, count in zip(
            key1_values,
            key1_offsets,
            key1_counts,
            strict=True,
        )
    }


def build_raw_baseline_payload(
    *,
    dtype_base: str,
    dt: float | None,
    key1_values: np.ndarray,
    mu_sections: np.ndarray,
    sigma_sections: np.ndarray,
    mu_traces: np.ndarray,
    sigma_traces: np.ndarray,
    zero_var_mask: np.ndarray,
    trace_spans_by_key1: dict[str, list[list[int]]],
    source_sha256: str | None,
    key1_byte: int,
    key2_byte: int,
    computed_at: str | None = None,
    serialize_arrays: bool = True,
) -> dict[str, Any]:
    manifest = {
        'stage': BASELINE_STAGE_RAW,
        'artifact_version': BASELINE_ARTIFACT_VERSION,
        'ddof': 0,
        'method': 'mean_std',
        'dtype_base': str(dtype_base),
        'dt': float(dt) if isinstance(dt, (int, float)) else None,
        'key1_values': np.asarray(key1_values, dtype=np.int64).tolist(),
        'mu_section_by_key1': np.asarray(
            mu_sections, dtype=np.float32
        ).tolist(),
        'sigma_section_by_key1': np.asarray(
            sigma_sections, dtype=np.float32
        ).tolist(),
        'trace_spans_by_key1': trace_spans_by_key1,
        'source_sha256': source_sha256,
        'computed_at': computed_at
        or datetime.now(timezone.utc).isoformat(),
        'key1_byte': int(key1_byte),
        'key2_byte': int(key2_byte),
    }
    if serialize_arrays:
        return merge_baseline_payload(
            manifest,
            mu_traces=mu_traces,
            sigma_traces=sigma_traces,
            zero_var_mask=zero_var_mask,
        )
    payload = dict(manifest)
    payload['mu_traces'] = np.ascontiguousarray(
        np.asarray(mu_traces, dtype=np.float32)
    )
    payload['sigma_traces'] = np.ascontiguousarray(
        np.asarray(sigma_traces, dtype=np.float32)
    )
    payload['zero_var_mask'] = np.ascontiguousarray(
        np.asarray(zero_var_mask, dtype=bool)
    )
    return payload


def write_raw_baseline_artifacts(
    store_path: str | Path,
    *,
    key1_byte: int,
    key2_byte: int,
    payload: dict[str, Any],
) -> None:
    base_path = Path(store_path)
    manifest_path = build_baseline_manifest_path(
        base_path,
        stage=BASELINE_STAGE_RAW,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    npz_path = build_baseline_npz_path(
        base_path,
        stage=BASELINE_STAGE_RAW,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    manifest, arrays = split_baseline_payload(payload)
    artifact_id = _normalize_artifact_id(manifest.get(BASELINE_ARTIFACT_ID_FIELD))
    if artifact_id is None:
        artifact_id = uuid.uuid4().hex
    manifest[BASELINE_ARTIFACT_ID_FIELD] = artifact_id
    manifest_tmp = manifest_path.with_name(f'{manifest_path.name}.tmp')
    npz_tmp = npz_path.with_name(f'{npz_path.name}.tmp')
    try:
        with npz_tmp.open('wb') as fh:
            np.savez(
                fh,
                mu_traces=np.asarray(arrays['mu_traces'], dtype=np.float32),
                sigma_traces=np.asarray(arrays['sigma_traces'], dtype=np.float32),
                zero_var_mask=np.asarray(arrays['zero_var_mask'], dtype=bool),
                artifact_id=np.asarray(artifact_id),
            )
        manifest_tmp.write_text(json.dumps(manifest), encoding='utf-8')
        os.replace(npz_tmp, npz_path)
        os.replace(manifest_tmp, manifest_path)
    finally:
        npz_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def has_split_baseline_artifacts(
    store_path: str | Path,
    *,
    key1_byte: int,
    key2_byte: int,
    source_sha256: str | None = None,
) -> bool:
    try:
        resolved = read_split_baseline_payload(
            store_path,
            stage=BASELINE_STAGE_RAW,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            include_arrays=False,
        )
    except SplitBaselineArtifactsError:
        return False
    if resolved is None:
        return False
    payload, _cache_key = resolved
    return _payload_matches_source_sha(payload, source_sha256=source_sha256)


def has_legacy_baseline_artifact(
    store_path: str | Path,
    *,
    key1_byte: int,
    key2_byte: int,
    source_sha256: str | None = None,
) -> bool:
    legacy_path = build_legacy_baseline_path(store_path)
    if not legacy_path.is_file():
        return False
    try:
        payload = _load_json_payload(legacy_path)
    except SplitBaselineArtifactsError:
        return False
    if not _payload_matches_stage_and_key_bytes(
        payload,
        stage=BASELINE_STAGE_RAW,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    ):
        return False
    if not _payload_matches_source_sha(payload, source_sha256=source_sha256):
        return False
    required_keys = (*BASELINE_MANIFEST_KEYS, *BASELINE_ARRAY_KEYS)
    return all(key in payload for key in required_keys)


def has_compatible_baseline_artifact(
    store_path: str | Path,
    *,
    key1_byte: int,
    key2_byte: int,
    source_sha256: str | None = None,
) -> bool:
    return has_split_baseline_artifacts(
        store_path,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        source_sha256=source_sha256,
    ) or has_legacy_baseline_artifact(
        store_path,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        source_sha256=source_sha256,
    )


__all__ = [
    'BASELINE_ARRAY_KEYS',
    'BASELINE_ARTIFACT_ID_FIELD',
    'BASELINE_ARTIFACT_VERSION',
    'BASELINE_STAGE_RAW',
    'LEGACY_BASELINE_FILENAME_RAW',
    'SplitBaselineArtifactsError',
    'build_baseline_manifest_path',
    'build_baseline_npz_path',
    'build_baseline_stem',
    'build_legacy_baseline_path',
    'build_raw_baseline_payload',
    'build_trace_spans_by_key1',
    'has_compatible_baseline_artifact',
    'has_legacy_baseline_artifact',
    'has_split_baseline_artifacts',
    'merge_baseline_payload',
    'read_split_baseline_payload',
    'split_baseline_payload',
    'write_raw_baseline_artifacts',
]
