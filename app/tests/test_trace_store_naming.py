from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.trace_store.naming import (
    CONTENT_ADDRESSED_STORE_NAME_MAX_CHARS,
    DIRECT_IMPORT_RAW_NAME_MAX_CHARS,
    bounded_direct_import_raw_name,
    content_addressed_compare_store_name,
    safe_store_name,
    safe_upload_name,
)


def test_safe_upload_name_preserves_safe_name_and_replaces_unsafe_chars():
    assert safe_upload_name('line-001_foo.sgy') == 'line-001_foo.sgy'
    assert safe_upload_name('line 001/@foo.sgy') == 'line_001__foo.sgy'


@pytest.mark.parametrize('store_name', ['', '.', '..', '../line.sgy', '/line.sgy'])
def test_safe_store_name_rejects_path_traversal_absolute_path_and_slash(store_name):
    with pytest.raises(HTTPException) as exc_info:
        safe_store_name(store_name)

    assert exc_info.value.status_code == 400


@pytest.mark.parametrize('store_name', ['dir/line.sgy', r'dir\line.sgy'])
def test_safe_store_name_rejects_slashes(store_name):
    with pytest.raises(HTTPException) as exc_info:
        safe_store_name(store_name)

    assert exc_info.value.status_code == 400


def test_content_addressed_compare_store_name_rejects_invalid_sha256():
    with pytest.raises(HTTPException) as exc_info:
        content_addressed_compare_store_name(
            safe_name='line.sgy',
            source_sha256='not-a-sha',
            key1_byte=189,
            key2_byte=193,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == 'Source sha256 is invalid'


def test_content_addressed_compare_store_name_bounds_long_stem_and_keeps_hash_suffix():
    source_sha256 = 'abcdef1234567890' * 4
    suffix = f'__k189_193__sha256_{source_sha256}'

    store_name = content_addressed_compare_store_name(
        safe_name=f'{"line_" * 70}.sgy',
        source_sha256=source_sha256,
        key1_byte=189,
        key2_byte=193,
    )

    assert len(store_name) <= CONTENT_ADDRESSED_STORE_NAME_MAX_CHARS
    assert store_name.endswith(suffix)


def test_bounded_direct_import_raw_name_bounds_long_raw_name():
    raw_name = bounded_direct_import_raw_name(f'{"line_" * 70}.sgy')

    assert len(raw_name) <= DIRECT_IMPORT_RAW_NAME_MAX_CHARS
    assert raw_name.endswith('.sgy')
