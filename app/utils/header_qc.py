"""Lightweight SEG-Y trace-header QC and key-byte recommendations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import segyio

HEADER_CANDIDATES: list[tuple[int, str]] = [
    (1, "TRACE_SEQUENCE_LINE"),
    (5, "TRACE_SEQUENCE_FILE"),
    (9, "FIELD_RECORD"),
    (13, "TRACE_NUMBER"),
    (21, "CDP"),
    (25, "CDP_TRACE"),
    (37, "OFFSET"),
    (115, "NS"),
    (117, "DT"),
    (189, "INLINE_3D"),
    (193, "CROSSLINE_3D"),
    (197, "SHOT_POINT"),
]

_CANDIDATE_NAMES = dict(HEADER_CANDIDATES)
_MAX_PAIR_GROUPS = 64

_KEY1_PRIOR = {
    1: 0.20,
    5: 0.20,
    9: 0.75,
    13: 0.30,
    21: 0.70,
    25: 0.45,
    37: 0.20,
    115: 0.0,
    117: 0.0,
    189: 1.0,
    193: 0.65,
    197: 0.65,
}

_KEY2_PRIOR = {
    1: 0.35,
    5: 0.35,
    9: 0.35,
    13: 0.85,
    21: 0.45,
    25: 0.80,
    37: 0.75,
    115: 0.0,
    117: 0.0,
    189: 0.45,
    193: 1.0,
    197: 0.55,
}


def inspect_segy_header_qc(path: str | Path) -> dict:
    """Inspect SEG-Y trace headers and recommend key-byte pairs."""
    segy_path = Path(path)
    header_values: dict[int, np.ndarray] = {}
    unavailable: dict[int, str] = {}

    with segyio.open(str(segy_path), 'r', ignore_geometry=True) as segy_file:
        n_traces = int(getattr(segy_file, 'tracecount', 0))
        n_samples = len(getattr(segy_file, 'samples', []))
        raw_dt = _read_binary_dt(segy_file)

        for byte, _name in HEADER_CANDIDATES:
            try:
                values = np.asarray(segy_file.attributes(byte)[:])
            except Exception as exc:  # noqa: BLE001
                unavailable[byte] = f'Header could not be read: {exc}'
                continue

            if values.ndim != 1 or int(values.shape[0]) != n_traces:
                unavailable[byte] = 'Header length does not match trace count'
                continue

            try:
                header_values[byte] = values.astype(np.int64, copy=False)
            except (TypeError, ValueError) as exc:
                unavailable[byte] = f'Header values are not integer-like: {exc}'

    dt = _dt_seconds(raw_dt)
    if dt is None and 117 in header_values:
        dt = _dt_from_header_values(header_values[117])

    return _build_qc_result(
        n_traces=n_traces,
        n_samples=n_samples,
        dt=dt,
        header_values=header_values,
        unavailable=unavailable,
    )


def _read_binary_dt(segy_file: Any) -> Any:
    try:
        return segy_file.bin[segyio.BinField.Interval]
    except Exception:  # noqa: BLE001
        return None


def _dt_seconds(raw_dt: Any) -> float | None:
    if isinstance(raw_dt, (int, float, np.integer, np.floating)) and raw_dt > 0:
        return float(raw_dt) / 1_000_000.0
    return None


def _dt_from_header_values(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    positive = values[values > 0]
    if positive.size == 0:
        return None
    return float(np.median(positive)) / 1_000_000.0


def _build_qc_result(
    *,
    n_traces: int,
    n_samples: int,
    dt: float | None,
    header_values: dict[int, np.ndarray],
    unavailable: dict[int, str],
) -> dict:
    headers: list[dict] = []
    headers_by_byte: dict[int, dict] = {}

    for byte, name in HEADER_CANDIDATES:
        values = header_values.get(byte)
        if values is None:
            warning = unavailable.get(byte, 'Header is unavailable')
            item = _unavailable_header(byte, name, warning)
        else:
            item = _score_key1_header(byte, name, values, n_traces)
        headers.append(item)
        headers_by_byte[byte] = item

    recommended_pairs = _rank_pairs(header_values, headers_by_byte)
    warnings: list[str] = []
    if n_traces <= 0:
        warnings.append('SEG-Y contains no traces')
    if not header_values:
        warnings.append('No candidate trace headers could be read')
    elif not recommended_pairs or recommended_pairs[0]['score'] < 0.55:
        warnings.append('No confident key byte pair found')

    return {
        'segy': {
            'n_traces': int(n_traces),
            'n_samples': int(n_samples),
            'dt': dt,
        },
        'recommended_pairs': recommended_pairs,
        'headers': headers,
        'warnings': warnings,
    }


def _unavailable_header(byte: int, name: str, warning: str) -> dict:
    return {
        'byte': int(byte),
        'name': name,
        'available': False,
        'min': None,
        'max': None,
        'unique_count': 0,
        'unique_ratio': 0.0,
        'key1_score': 0.0,
        'group_size': None,
        'warnings': [warning],
    }


def _score_key1_header(
    byte: int,
    name: str,
    values: np.ndarray,
    n_traces: int,
) -> dict:
    warnings: list[str] = []
    if n_traces <= 0 or values.size == 0:
        return {
            'byte': int(byte),
            'name': name,
            'available': True,
            'min': None,
            'max': None,
            'unique_count': 0,
            'unique_ratio': 0.0,
            'key1_score': 0.0,
            'group_size': None,
            'warnings': ['Header cannot be scored without traces'],
        }

    unique_values, counts = np.unique(values, return_counts=True)
    unique_count = int(unique_values.size)
    unique_ratio = float(unique_count / max(int(n_traces), 1))
    group_size = _group_size_stats(counts)
    median_group_size = float(group_size['p50'])

    if unique_count <= 1:
        warnings.append('Header is constant')
    if unique_count == n_traces:
        warnings.append('Header is unique for every trace')
    if median_group_size <= 1:
        warnings.append('Median group size is too small')
    if _is_extremely_imbalanced(group_size):
        warnings.append('Group sizes are extremely imbalanced')

    if unique_count <= 1:
        score = 0.05 + (0.05 * _KEY1_PRIOR.get(byte, 0.0))
    elif unique_count == n_traces:
        score = 0.06 + (0.04 * _KEY1_PRIOR.get(byte, 0.0))
    else:
        group_score = min(1.0, max(0.0, (median_group_size - 1.0) / 9.0))
        ratio_score = 1.0 if unique_ratio <= 0.5 else max(0.0, (1.0 - unique_ratio) / 0.5)
        contiguity_score = _contiguity_score(values, unique_count)
        balance_score = _balance_score(group_size)
        semantic_score = _KEY1_PRIOR.get(byte, 0.0)
        score = (
            0.28 * group_score
            + 0.22 * ratio_score
            + 0.22 * contiguity_score
            + 0.18 * balance_score
            + 0.10 * semantic_score
        )

    return {
        'byte': int(byte),
        'name': name,
        'available': True,
        'min': int(np.min(values)),
        'max': int(np.max(values)),
        'unique_count': unique_count,
        'unique_ratio': _round_score(unique_ratio),
        'key1_score': _round_score(score),
        'group_size': group_size,
        'warnings': warnings,
    }


def _group_size_stats(counts: np.ndarray) -> dict[str, int | float]:
    if counts.size == 0:
        return {'min': 0, 'p05': 0, 'p50': 0, 'p95': 0, 'max': 0}
    percentiles = np.percentile(counts.astype(np.float64), [5, 50, 95])
    return {
        'min': int(np.min(counts)),
        'p05': _number(percentiles[0]),
        'p50': _number(percentiles[1]),
        'p95': _number(percentiles[2]),
        'max': int(np.max(counts)),
    }


def _number(value: float) -> int | float:
    value = float(value)
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return int(rounded)
    return round(value, 6)


def _is_extremely_imbalanced(group_size: dict[str, int | float]) -> bool:
    p05 = float(group_size['p05'])
    p95 = float(group_size['p95'])
    median = float(group_size['p50'])
    maximum = float(group_size['max'])
    if median <= 0:
        return False
    if p05 > 0 and p95 / p05 >= 10.0:
        return True
    return maximum / median >= 20.0


def _balance_score(group_size: dict[str, int | float]) -> float:
    p05 = float(group_size['p05'])
    p95 = float(group_size['p95'])
    if p05 <= 0 or p95 <= 0:
        return 0.0
    return min(1.0, p05 / p95)


def _contiguity_score(values: np.ndarray, unique_count: int) -> float:
    if values.size <= 1 or unique_count <= 1:
        return 1.0
    runs = int(np.count_nonzero(values[1:] != values[:-1])) + 1
    if runs <= 0:
        return 0.0
    return min(1.0, float(unique_count) / float(runs))


def _rank_pairs(
    header_values: dict[int, np.ndarray],
    headers_by_byte: dict[int, dict],
) -> list[dict]:
    pairs: list[dict] = []
    for key1_byte, key1_name in HEADER_CANDIDATES:
        key1_values = header_values.get(key1_byte)
        key1_stats = headers_by_byte[key1_byte]
        if key1_values is None:
            continue
        for key2_byte, key2_name in HEADER_CANDIDATES:
            if key1_byte == key2_byte:
                continue
            key2_values = header_values.get(key2_byte)
            if key2_values is None:
                continue
            pairs.append(
                _score_pair(
                    key1_byte=key1_byte,
                    key1_name=key1_name,
                    key1_values=key1_values,
                    key1_stats=key1_stats,
                    key2_byte=key2_byte,
                    key2_name=key2_name,
                    key2_values=key2_values,
                )
            )
    pairs.sort(key=lambda item: item['score'], reverse=True)
    return pairs


def _score_pair(
    *,
    key1_byte: int,
    key1_name: str,
    key1_values: np.ndarray,
    key1_stats: dict,
    key2_byte: int,
    key2_name: str,
    key2_values: np.ndarray,
) -> dict:
    pair_metrics = _key2_within_group_metrics(key1_values, key2_values)
    key1_score = float(key1_stats.get('key1_score') or 0.0)
    key2_quality = (
        0.55 * pair_metrics['median_unique_ratio']
        + 0.25 * (1.0 - pair_metrics['median_duplicate_rate'])
        + 0.20 * pair_metrics['monotonic_fraction']
    )
    semantic_multiplier = 0.95 + (0.05 * _KEY2_PRIOR.get(key2_byte, 0.0))
    score = ((0.65 * key1_score) + (0.35 * key2_quality)) * semantic_multiplier
    if pair_metrics['constant_section_fraction'] >= 0.30:
        score *= 0.65
    if pair_metrics['median_duplicate_rate'] > 0.25:
        score *= 0.75
    score = max(0.0, min(1.0, score))

    group_size = key1_stats.get('group_size') or {}
    sections = int(key1_stats.get('unique_count') or 0)
    median_group_size = group_size.get('p50', 0)
    reasons = [
        f'key1 has {sections} sections with median {median_group_size} traces/section',
        f"key2 median unique ratio within sections is {pair_metrics['median_unique_ratio']:.2f}",
    ]
    if pair_metrics['monotonic_fraction'] >= 0.80:
        reasons.append('key2 is mostly monotonic within key1 sections')
    elif pair_metrics['monotonic_fraction'] <= 0.20:
        reasons.append('key2 is rarely monotonic within key1 sections')

    warnings = _pair_warnings(key1_stats, pair_metrics)

    rounded_score = _round_score(score)
    return {
        'key1_byte': int(key1_byte),
        'key1_name': key1_name,
        'key2_byte': int(key2_byte),
        'key2_name': key2_name,
        'score': rounded_score,
        'confidence': _confidence(rounded_score),
        'reasons': reasons,
        'warnings': warnings,
    }


def _key2_within_group_metrics(
    key1_values: np.ndarray,
    key2_values: np.ndarray,
) -> dict[str, float]:
    if key1_values.size == 0 or key2_values.size == 0:
        return {
            'median_unique_ratio': 0.0,
            'median_duplicate_rate': 1.0,
            'monotonic_fraction': 0.0,
            'constant_section_fraction': 1.0,
            'evaluated_groups': 0.0,
        }

    order = np.argsort(key1_values, kind='stable')
    sorted_key1 = key1_values[order]
    boundaries = np.flatnonzero(sorted_key1[1:] != sorted_key1[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [sorted_key1.size]))
    group_count = int(starts.size)
    if group_count > _MAX_PAIR_GROUPS:
        sampled = np.unique(
            np.linspace(0, group_count - 1, _MAX_PAIR_GROUPS).astype(np.int64)
        )
        starts = starts[sampled]
        ends = ends[sampled]

    unique_ratios: list[float] = []
    duplicate_rates: list[float] = []
    monotonic: list[float] = []
    constant: list[float] = []

    for start, end in zip(starts, ends):
        group_order = order[int(start) : int(end)]
        if group_order.size <= 1:
            continue
        values = key2_values[group_order]
        unique_count = int(np.unique(values).size)
        unique_ratio = float(unique_count / int(values.size))
        unique_ratios.append(unique_ratio)
        duplicate_rates.append(1.0 - unique_ratio)
        constant.append(1.0 if unique_count <= 1 else 0.0)

        diffs = np.diff(values)
        is_monotonic = bool(
            np.any(diffs != 0)
            and (np.all(diffs >= 0) or np.all(diffs <= 0))
        )
        monotonic.append(1.0 if is_monotonic else 0.0)

    if not unique_ratios:
        return {
            'median_unique_ratio': 0.0,
            'median_duplicate_rate': 1.0,
            'monotonic_fraction': 0.0,
            'constant_section_fraction': 1.0,
            'evaluated_groups': 0.0,
        }

    return {
        'median_unique_ratio': float(np.median(unique_ratios)),
        'median_duplicate_rate': float(np.median(duplicate_rates)),
        'monotonic_fraction': float(np.mean(monotonic)),
        'constant_section_fraction': float(np.mean(constant)),
        'evaluated_groups': float(len(unique_ratios)),
    }


def _pair_warnings(key1_stats: dict, pair_metrics: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    for warning in key1_stats.get('warnings') or []:
        warnings.append(f'key1 {str(warning)[0].lower()}{str(warning)[1:]}')
    if pair_metrics['evaluated_groups'] <= 0:
        warnings.append('key2 could not be evaluated inside key1 sections')
    if pair_metrics['constant_section_fraction'] >= 0.30:
        warnings.append('key2 is constant in many key1 sections')
    if pair_metrics['median_duplicate_rate'] > 0.25:
        warnings.append('key2 has many duplicates inside key1 sections')
    return warnings


def _confidence(score: float) -> str:
    if score >= 0.80:
        return 'high'
    if score >= 0.55:
        return 'medium'
    return 'low'


def _round_score(value: float) -> float:
    return round(float(value), 6)


__all__ = ['HEADER_CANDIDATES', 'inspect_segy_header_qc']
