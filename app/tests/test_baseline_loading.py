from __future__ import annotations

import numpy as np
import pytest

from app.services.scaling import apply_scaling_from_baseline
from app.tests._stubs import write_baseline_raw
from app.utils.baseline_artifacts import (
    BASELINE_STAGE_RAW,
    build_baseline_npz_path,
    build_raw_baseline_payload,
)
from app.utils.segy_meta import _BASELINE_CACHE, load_baseline


@pytest.fixture(autouse=True)
def _clear_baseline_cache():
    _BASELINE_CACHE.clear()
    yield
    _BASELINE_CACHE.clear()


@pytest.mark.parametrize('mode', ['amax', 'tracewise'])
def test_apply_scaling_matches_split_and_legacy_baselines(
    tmp_path, mode: str
):
    split_dir = tmp_path / 'split'
    legacy_dir = tmp_path / 'legacy'
    split_dir.mkdir()
    legacy_dir.mkdir()
    kwargs = {
        'key1': 7,
        'section_mean': 10.0,
        'section_std': 2.0,
        'trace_means': [1.0, 10.0, 20.0],
        'trace_stds': [2.0, 4.0, 5.0],
        'key1_byte': 189,
        'key2_byte': 193,
    }
    write_baseline_raw(split_dir, **kwargs)
    write_baseline_raw(legacy_dir, legacy_only=True, **kwargs)

    arr = np.asarray(
        [
            [2.0, 6.0, 10.0],
            [14.0, 18.0, 22.0],
            [25.0, 30.0, 35.0],
        ],
        dtype=np.float32,
    )

    scaled_split = apply_scaling_from_baseline(
        arr.copy(),
        scaling=mode,
        file_id='file-a',
        key1=7,
        store_dir=split_dir,
        key1_byte=189,
        key2_byte=193,
        trace_stats_cache={},
        x0=0,
        x1=2,
        step_x=1,
    )
    _BASELINE_CACHE.clear()
    scaled_legacy = apply_scaling_from_baseline(
        arr.copy(),
        scaling=mode,
        file_id='file-a',
        key1=7,
        store_dir=legacy_dir,
        key1_byte=189,
        key2_byte=193,
        trace_stats_cache={},
        x0=0,
        x1=2,
        step_x=1,
    )

    np.testing.assert_allclose(scaled_split, scaled_legacy, rtol=0, atol=0)


def test_load_baseline_rejects_mismatched_key_bytes(tmp_path):
    store_dir = tmp_path / 'store'
    store_dir.mkdir()
    write_baseline_raw(store_dir, key1=7, n_traces=3, key1_byte=189, key2_byte=193)

    entry = load_baseline(store_dir, key1_byte=189, key2_byte=193)
    assert entry['store_key'].endswith('::189::193')

    with pytest.raises(FileNotFoundError):
        load_baseline(store_dir, key1_byte=200, key2_byte=193)


def test_build_raw_baseline_payload_can_preserve_numpy_arrays_for_split_write():
    payload = build_raw_baseline_payload(
        dtype_base='float32',
        dt=0.004,
        key1_values=np.asarray([7], dtype=np.int64),
        mu_sections=np.asarray([1.5], dtype=np.float32),
        sigma_sections=np.asarray([2.5], dtype=np.float32),
        mu_traces=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        sigma_traces=np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
        zero_var_mask=np.asarray([False, True, False], dtype=bool),
        trace_spans_by_key1={'7': [[0, 3]]},
        source_sha256='sha',
        key1_byte=189,
        key2_byte=193,
        serialize_arrays=False,
    )

    assert isinstance(payload['mu_traces'], np.ndarray)
    assert payload['mu_traces'].dtype == np.float32
    assert isinstance(payload['sigma_traces'], np.ndarray)
    assert payload['sigma_traces'].dtype == np.float32
    assert isinstance(payload['zero_var_mask'], np.ndarray)
    assert payload['zero_var_mask'].dtype == bool


def test_load_baseline_falls_back_to_legacy_when_split_artifacts_are_torn(tmp_path):
    store_dir = tmp_path / 'store'
    other_dir = tmp_path / 'other'
    store_dir.mkdir()
    other_dir.mkdir()
    write_baseline_raw(
        store_dir,
        key1=7,
        trace_means=[1.0, 2.0, 3.0],
        trace_stds=[1.0, 1.0, 1.0],
        key1_byte=189,
        key2_byte=193,
    )
    write_baseline_raw(
        store_dir,
        key1=7,
        trace_means=[1.0, 2.0, 3.0],
        trace_stds=[1.0, 1.0, 1.0],
        key1_byte=189,
        key2_byte=193,
        legacy_only=True,
    )
    write_baseline_raw(
        other_dir,
        key1=7,
        trace_means=[10.0, 20.0, 30.0],
        trace_stds=[2.0, 2.0, 2.0],
        key1_byte=189,
        key2_byte=193,
    )

    store_npz = build_baseline_npz_path(
        store_dir,
        stage=BASELINE_STAGE_RAW,
        key1_byte=189,
        key2_byte=193,
    )
    other_npz = build_baseline_npz_path(
        other_dir,
        stage=BASELINE_STAGE_RAW,
        key1_byte=189,
        key2_byte=193,
    )
    store_npz.write_bytes(other_npz.read_bytes())

    entry = load_baseline(store_dir, key1_byte=189, key2_byte=193)
    np.testing.assert_allclose(entry['trace_mean'], [1.0, 2.0, 3.0])
