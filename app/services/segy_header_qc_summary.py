"""Helpers for summarizing SEG-Y header QC selections."""

from __future__ import annotations


def selected_header_qc_summary(
    header_qc: object,
    *,
    key1_byte: int,
    key2_byte: int,
) -> dict | None:
    if not isinstance(header_qc, dict):
        return None

    pairs = header_qc.get('recommended_pairs')
    if isinstance(pairs, list):
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            if (
                pair.get('key1_byte') == key1_byte
                and pair.get('key2_byte') == key2_byte
            ):
                warnings = pair.get('warnings')
                return {
                    'selected_pair_score': pair.get('score'),
                    'confidence': pair.get('confidence', 'unknown'),
                    'warnings': warnings if isinstance(warnings, list) else [],
                }

    warnings = header_qc.get('warnings')
    return {
        'selected_pair_score': None,
        'confidence': 'unknown',
        'warnings': warnings if isinstance(warnings, list) else [],
    }
