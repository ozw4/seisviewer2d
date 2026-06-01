from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from app.core.state import create_app_state
from app.services.trace_store_registration import trace_store_cache_key
from app.statics.common import corrected_store, trace_shift
from app.statics.common.corrected_store import (
    cleanup_artifact,
    cleanup_registration,
    cleanup_store,
    corrected_store_path,
)
from app.statics.common.trace_shift import (
    apply_trace_shifts_to_array,
    require_1d_bool,
    require_1d_float64,
    require_1d_string,
    validate_trace_shifts_for_application,
)


def test_validate_trace_shifts_accepts_generic_fields() -> None:
    result = validate_trace_shifts_for_application(
        trace_shift_s_sorted=np.asarray([0.0, 0.004, -0.008], dtype=np.float64),
        trace_static_valid_mask_sorted=np.ones(3, dtype=bool),
        trace_static_status_sorted=np.asarray(['ok', 'ok', 'ok'], dtype='<U8'),
        n_traces=3,
        max_abs_shift_ms=10.0,
        shift_field_name='datum_trace_shift_s_sorted',
    )

    assert result.max_abs_applied_shift_ms == pytest.approx(8.0)
    assert result.n_positive_trace_shifts == 1
    assert result.n_negative_trace_shifts == 1
    assert result.trace_static_status_counts == {'ok': 3}


def test_validate_trace_shifts_rejects_invalid_and_excessive_shifts() -> None:
    with pytest.raises(ValueError, match='invalid_trace_shift_count=1'):
        validate_trace_shifts_for_application(
            trace_shift_s_sorted=np.asarray([0.0, np.nan], dtype=np.float64),
            trace_static_valid_mask_sorted=np.asarray([True, False]),
            trace_static_status_sorted=np.asarray(['ok', 'missing']),
            n_traces=2,
            max_abs_shift_ms=10.0,
            shift_field_name='generic_shift_s_sorted',
        )

    with pytest.raises(ValueError, match='generic_shift_s_sorted exceeds'):
        validate_trace_shifts_for_application(
            trace_shift_s_sorted=np.asarray([0.0, 0.011], dtype=np.float64),
            trace_static_valid_mask_sorted=np.ones(2, dtype=bool),
            trace_static_status_sorted=np.asarray(['ok', 'ok']),
            n_traces=2,
            max_abs_shift_ms=10.0,
            shift_field_name='generic_shift_s_sorted',
        )


def test_1d_validation_helpers_are_not_refraction_specific() -> None:
    np.testing.assert_allclose(
        require_1d_float64(
            np.asarray([1.0, 2.0], dtype=np.float32),
            name='custom_shift',
            expected_shape=(2,),
        ),
        [1.0, 2.0],
    )
    np.testing.assert_array_equal(
        require_1d_bool(
            np.asarray([True, False]),
            name='custom_mask',
            expected_shape=(2,),
        ),
        [True, False],
    )
    np.testing.assert_array_equal(
        require_1d_string(
            np.asarray(['ok', 'bad'], dtype='|S4'),
            name='custom_status',
            expected_shape=(2,),
        ),
        ['ok', 'bad'],
    )


def test_apply_trace_shifts_to_array_uses_repo_sign_convention() -> None:
    traces = np.zeros((2, 8), dtype=np.float32)
    traces[:, 3] = 1.0

    corrected = apply_trace_shifts_to_array(
        traces=traces,
        sample_interval_s=0.004,
        trace_shift_s_sorted=np.asarray([0.004, -0.004]),
        fill_value=0.0,
    )

    assert [int(np.argmax(corrected[index])) for index in range(2)] == [4, 2]


def test_corrected_store_path_uses_supplied_statics_kind(tmp_path: Path) -> None:
    source = tmp_path / 'line 001.sgy'
    source.mkdir()

    path = corrected_store_path(
        source_store_path=source,
        statics_kind='datum',
        suffix='job 1',
    )

    assert path.name == 'line_001.sgy.statics.datum.job_1'
    assert 'refraction' not in path.name


def test_cleanup_helpers_remove_registry_cache_store_and_artifacts(tmp_path: Path) -> None:
    state = create_app_state()
    store = tmp_path / 'corrected.store'
    store.mkdir()
    (tmp_path / 'corrected.store.tmp-build').mkdir()
    file_id = 'corrected-file-id'
    key1_byte = 189
    key2_byte = 193
    state.file_registry.update(file_id, store_path=str(store), dt=0.004)
    with state.lock:
        state.cached_readers[trace_store_cache_key(file_id, key1_byte, key2_byte)] = object()

    cleanup_registration(
        state,
        file_id=file_id,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
    )
    cleanup_store(store)

    artifact = tmp_path / 'corrected_file.json'
    artifact.write_text('{}', encoding='utf-8')
    tmp_artifact = tmp_path / 'corrected_file.json.abc.tmp'
    tmp_artifact.write_text('{}', encoding='utf-8')
    cleanup_artifact(artifact)

    assert file_id not in state.file_registry.records
    with state.lock:
        assert trace_store_cache_key(file_id, key1_byte, key2_byte) not in state.cached_readers
    assert not store.exists()
    assert not (tmp_path / 'corrected.store.tmp-build').exists()
    assert not artifact.exists()
    assert not tmp_artifact.exists()


def test_common_modules_do_not_import_refraction() -> None:
    for module in (corrected_store, trace_shift):
        tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        assert all('refraction' not in name for name in imported_modules)
