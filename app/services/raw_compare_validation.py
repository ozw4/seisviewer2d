"""Raw compare grid validation service."""

from __future__ import annotations

import numpy as np

from app.core.state import AppState
from app.services.reader import get_reader

COMPARE_DT_TOLERANCE = 1.0e-9


def _mismatch_payload(
    *,
    reason: str,
    message: str,
    checked_key1_count: int,
    mismatch: dict[str, object],
) -> dict[str, object]:
    return {
        'ok': False,
        'reason': reason,
        'message': message,
        'checked_key1_count': int(checked_key1_count),
        'mismatch': mismatch,
        'files': [],
    }


def _first_array_mismatch(
    a_values: np.ndarray,
    b_values: np.ndarray,
) -> dict[str, object]:
    mismatch: dict[str, object] = {
        'type': 'key1_values',
        'a_count': int(a_values.size),
        'b_count': int(b_values.size),
    }
    shared_count = min(int(a_values.size), int(b_values.size))
    if shared_count == 0 or a_values.size != b_values.size:
        return mismatch
    diff_positions = np.flatnonzero(a_values != b_values)
    if diff_positions.size:
        index = int(diff_positions[0])
        mismatch.update(
            {
                'index': index,
                'a_value': int(a_values[index]),
                'b_value': int(b_values[index]),
            }
        )
    return mismatch


def _file_summary(
    *,
    role: str,
    file_id: str,
    file_name: str,
    key1_count: int,
    n_samples: int,
    dt: float,
) -> dict[str, object]:
    return {
        'role': role,
        'file_id': file_id,
        'file_name': file_name,
        'key1_count': int(key1_count),
        'n_samples': int(n_samples),
        'dt': float(dt),
    }


def validate_raw_compare_grid(
    *,
    file_id_a: str,
    file_id_b: str,
    key1_byte: int,
    key2_byte: int,
    state: AppState,
) -> dict[str, object]:
    reader_a = get_reader(file_id_a, key1_byte, key2_byte, state=state)
    reader_b = get_reader(file_id_b, key1_byte, key2_byte, state=state)

    key1_values_a = np.asarray(reader_a.get_key1_values(), dtype=np.int64)
    key1_values_b = np.asarray(reader_b.get_key1_values(), dtype=np.int64)
    if key1_values_a.ndim != 1 or key1_values_b.ndim != 1:
        raise ValueError('key1_values must be 1D')
    if not np.array_equal(key1_values_a, key1_values_b):
        return _mismatch_payload(
            reason='key1_values',
            message='A-B unavailable: key1 values differ.',
            checked_key1_count=0,
            mismatch=_first_array_mismatch(key1_values_a, key1_values_b),
        )

    n_samples_a = int(reader_a.get_n_samples())
    n_samples_b = int(reader_b.get_n_samples())
    if n_samples_a != n_samples_b:
        return _mismatch_payload(
            reason='n_samples',
            message='A-B unavailable: sample counts differ.',
            checked_key1_count=0,
            mismatch={
                'type': 'n_samples',
                'a_n_samples': n_samples_a,
                'b_n_samples': n_samples_b,
            },
        )

    dt_a = float(state.file_registry.get_dt(file_id_a))
    dt_b = float(state.file_registry.get_dt(file_id_b))
    if (
        not np.isfinite(dt_a)
        or not np.isfinite(dt_b)
        or abs(dt_a - dt_b) > COMPARE_DT_TOLERANCE
    ):
        return _mismatch_payload(
            reason='dt',
            message='A-B unavailable: dt differs.',
            checked_key1_count=0,
            mismatch={
                'type': 'dt',
                'a_dt': dt_a,
                'b_dt': dt_b,
                'tolerance': COMPARE_DT_TOLERANCE,
            },
        )

    key2_header_a = np.asarray(reader_a.get_header(key2_byte))
    key2_header_b = np.asarray(reader_b.get_header(key2_byte))
    for index, key1 in enumerate(key1_values_a):
        key1_int = int(key1)
        seq_a = np.asarray(
            reader_a.get_trace_seq_for_value(key1_int, align_to='display'),
            dtype=np.int64,
        )
        seq_b = np.asarray(
            reader_b.get_trace_seq_for_value(key1_int, align_to='display'),
            dtype=np.int64,
        )
        key2_values_a = key2_header_a[seq_a]
        key2_values_b = key2_header_b[seq_b]
        if not np.array_equal(key2_values_a, key2_values_b):
            return _mismatch_payload(
                reason='key2_sequence',
                message=f'A-B unavailable: key2 sequence differs for key1 {key1_int}.',
                checked_key1_count=index + 1,
                mismatch={
                    'type': 'key2_sequence',
                    'key1': key1_int,
                    'a_count': int(key2_values_a.size),
                    'b_count': int(key2_values_b.size),
                },
            )

    file_name_a = state.file_registry.filename(file_id_a) or file_id_a
    file_name_b = state.file_registry.filename(file_id_b) or file_id_b
    return {
        'ok': True,
        'reason': '',
        'message': '',
        'checked_key1_count': int(key1_values_a.size),
        'files': [
            _file_summary(
                role='A',
                file_id=file_id_a,
                file_name=file_name_a,
                key1_count=int(key1_values_a.size),
                n_samples=n_samples_a,
                dt=dt_a,
            ),
            _file_summary(
                role='B',
                file_id=file_id_b,
                file_name=file_name_b,
                key1_count=int(key1_values_b.size),
                n_samples=n_samples_b,
                dt=dt_b,
            ),
        ],
    }


__all__ = ['COMPARE_DT_TOLERANCE', 'validate_raw_compare_grid']
