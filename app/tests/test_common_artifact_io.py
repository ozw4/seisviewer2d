from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.services.common.artifact_io import (
    atomic_write,
    write_csv_atomic,
    write_json_atomic,
    write_npz_atomic,
)


def test_atomic_write_creates_final_path(tmp_path: Path) -> None:
    path = tmp_path / 'nested' / 'artifact.txt'

    atomic_write(path, lambda tmp_path: tmp_path.write_text('ok', encoding='utf-8'))

    assert path.read_text(encoding='utf-8') == 'ok'


def test_atomic_write_failure_keeps_existing_file_and_cleans_temp(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'artifact.txt'
    path.write_text('original', encoding='utf-8')

    def fail(tmp_path: Path) -> None:
        tmp_path.write_text('partial', encoding='utf-8')
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        atomic_write(path, fail)

    assert path.read_text(encoding='utf-8') == 'original'
    assert not list(tmp_path.glob('artifact.txt.*.tmp'))


def test_write_json_atomic_honors_format_options(tmp_path: Path) -> None:
    path = tmp_path / 'contract.json'

    write_json_atomic(
        path,
        {'z': '雪', 'a': 1},
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        trailing_newline=True,
    )

    assert path.read_text(encoding='utf-8') == (
        '{\n'
        '  "a": 1,\n'
        '  "z": "\\u96ea"\n'
        '}\n'
    )


def test_write_json_atomic_can_disable_ascii_sort_indent_and_newline(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'contract.json'

    write_json_atomic(
        path,
        {'z': '雪', 'a': 1},
        ensure_ascii=False,
        indent=None,
        sort_keys=False,
        trailing_newline=False,
    )

    assert path.read_text(encoding='utf-8') == '{"z": "雪", "a": 1}'


def test_write_csv_atomic_column_order_and_extra_field_rejection(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'contract.csv'

    write_csv_atomic(
        path,
        columns=('b', 'a'),
        rows=({'a': 1, 'b': 2},),
    )

    assert path.read_text(encoding='utf-8') == 'b,a\n2,1\n'

    with pytest.raises(ValueError, match='dict contains fields not in fieldnames'):
        write_csv_atomic(
            path,
            columns=('a',),
            rows=({'a': 1, 'extra': 2},),
            extrasaction='raise',
        )

    assert not list(tmp_path.glob('contract.csv.*.tmp'))


def test_write_npz_atomic_uncompressed_and_compressed(tmp_path: Path) -> None:
    uncompressed_path = tmp_path / 'uncompressed.npz'
    compressed_path = tmp_path / 'compressed.npz'
    arrays = {'values': np.asarray([1.0, 2.0], dtype=np.float64)}

    write_npz_atomic(uncompressed_path, arrays, compressed=False)
    write_npz_atomic(compressed_path, arrays, compressed=True)

    with np.load(uncompressed_path) as data:
        np.testing.assert_array_equal(data['values'], arrays['values'])
    with np.load(compressed_path) as data:
        np.testing.assert_array_equal(data['values'], arrays['values'])
    with zipfile.ZipFile(uncompressed_path) as archive:
        assert archive.getinfo('values.npy').compress_type == zipfile.ZIP_STORED
    with zipfile.ZipFile(compressed_path) as archive:
        assert archive.getinfo('values.npy').compress_type == zipfile.ZIP_DEFLATED


def test_write_npz_atomic_rejects_object_arrays(tmp_path: Path) -> None:
    path = tmp_path / 'contract.npz'

    with pytest.raises(ValueError, match='object array is not allowed for bad'):
        write_npz_atomic(path, {'bad': np.asarray([object()], dtype=object)})

    assert not path.exists()


def test_error_factory_converts_write_failures(tmp_path: Path) -> None:
    path = tmp_path / 'nested' / 'contract.json'

    class ArtifactWriteError(Exception):
        pass

    def make_error(message: str, exc: BaseException) -> Exception:
        return ArtifactWriteError(f'{message}: {type(exc).__name__}')

    with pytest.raises(
        ArtifactWriteError,
        match=r'contract\.json: failed to write JSON artifact: TypeError',
    ):
        write_json_atomic(
            path,
            {'bad': object()},
            error_factory=make_error,
        )

    assert not path.exists()
    assert not list(path.parent.glob('contract.json.*.tmp'))


def test_write_json_atomic_allows_nan_when_requested(tmp_path: Path) -> None:
    path = tmp_path / 'nan.json'

    write_json_atomic(path, {'value': float('nan')}, allow_nan=True)

    assert path.read_text(encoding='utf-8') == '{"value": NaN}'
