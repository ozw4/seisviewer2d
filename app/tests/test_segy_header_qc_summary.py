from __future__ import annotations

from app.services.segy_header_qc_summary import selected_header_qc_summary


def test_selected_header_qc_summary_returns_matching_pair_details():
    summary = selected_header_qc_summary(
        {
            'recommended_pairs': [
                {
                    'key1_byte': 189,
                    'key2_byte': 193,
                    'score': 0.94,
                    'confidence': 'high',
                    'warnings': ['check fold'],
                }
            ],
            'warnings': ['global warning'],
        },
        key1_byte=189,
        key2_byte=193,
    )

    assert summary == {
        'selected_pair_score': 0.94,
        'confidence': 'high',
        'warnings': ['check fold'],
    }


def test_selected_header_qc_summary_falls_back_to_global_warnings():
    summary = selected_header_qc_summary(
        {
            'recommended_pairs': [
                {'key1_byte': 9, 'key2_byte': 13, 'score': 0.5}
            ],
            'warnings': ['selected pair was not recommended'],
        },
        key1_byte=189,
        key2_byte=193,
    )

    assert summary == {
        'selected_pair_score': None,
        'confidence': 'unknown',
        'warnings': ['selected pair was not recommended'],
    }


def test_selected_header_qc_summary_rejects_non_mapping_payload():
    assert (
        selected_header_qc_summary(
            None,
            key1_byte=189,
            key2_byte=193,
        )
        is None
    )
