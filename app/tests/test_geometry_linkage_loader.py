from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import app.services.geometry_linkage_loader as loader
from app.services.geometry_linkage_artifacts import (
    GEOMETRY_LINKAGE_NPZ_NAME,
    GeometryLinkageArtifactMetadata,
    build_geometry_linkage_solution_arrays,
)
from app.services.geometry_linkage_linker import (
    GeometryLinkageOptions,
    build_geometry_linkage,
)
from app.services.geometry_linkage_tables import build_endpoint_geometry_tables
from app.services.geometry_linkage_validation import GeometryLinkageHeaders
from app.services.geometry_linkage_loader import (
    load_geometry_linkage_artifact,
    load_geometry_linkage_from_job_dir,
    load_geometry_linkage_trace_node_mapping,
)


def test_load_geometry_linkage_artifact_reads_npz_with_allow_pickle_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_payload(tmp_path, _payload())
    original_load = loader.np.load
    allow_pickle_values: list[object] = []

    def spy_load(*args: object, **kwargs: object) -> object:
        allow_pickle_values.append(kwargs.get('allow_pickle'))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(loader.np, 'load', spy_load)

    load_geometry_linkage_artifact(path)

    assert allow_pickle_values == [False]


def test_load_geometry_linkage_artifact_returns_copied_arrays_after_npz_close(
    tmp_path: Path,
) -> None:
    path = _write_payload(tmp_path, _payload())

    loaded = load_geometry_linkage_artifact(path)

    np.testing.assert_array_equal(
        loaded.source_node_id_sorted,
        loaded.source_node_id_by_endpoint[loaded.source_endpoint_id_sorted],
    )
    assert loaded.source_node_id_sorted.flags.writeable is False
    assert loaded.record_method.flags.writeable is False


def test_load_geometry_linkage_artifact_returns_metadata_and_counts(
    tmp_path: Path,
) -> None:
    path = _write_payload(tmp_path, _payload(metadata=_metadata()))

    loaded = load_geometry_linkage_artifact(
        path,
        expected_n_traces=5,
        expected_key1_byte=189,
        expected_key2_byte=193,
    )

    assert loaded.schema_version == 1
    assert loaded.artifact_kind == 'geometry_linkage'
    assert loaded.order == 'trace_store_sorted'
    assert loaded.mode == 'auto_threshold'
    assert loaded.threshold_m == pytest.approx(1.1)
    assert loaded.receiver_location_interval_m == pytest.approx(25.0)
    assert loaded.metadata.job_id == 'linkage-job'
    assert loaded.metadata.key1_byte == 189
    assert loaded.n_traces == 5
    assert loaded.n_source_endpoints == 4
    assert loaded.n_receiver_endpoints == 2
    assert loaded.n_receiver_anchor_links == 1
    assert loaded.n_source_fallback_links == 2
    assert loaded.n_independent_source_nodes == 1


def test_load_geometry_linkage_artifact_parses_none_scalars(
    tmp_path: Path,
) -> None:
    path = _write_payload(tmp_path, _payload(mode='none'))

    loaded = load_geometry_linkage_artifact(path)

    assert loaded.threshold_m is None
    assert loaded.receiver_location_interval_m is None
    assert loaded.metadata.job_id is None
    assert loaded.metadata.input_file_id is None
    assert loaded.metadata.key1_byte is None
    assert loaded.metadata.header_source_segy_path is None


def test_load_geometry_linkage_artifact_rejects_missing_required_field(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload.pop('source_node_id_sorted')
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match='missing required field: source_node_id_sorted'):
        load_geometry_linkage_artifact(path)


def test_load_geometry_linkage_artifact_rejects_object_dtype(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload['record_method'] = np.asarray(payload['record_method'], dtype=object)
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match='record_method.*object dtype'):
        load_geometry_linkage_artifact(path)


@pytest.mark.parametrize(
    ('field', 'value', 'match'),
    [
        ('schema_version', np.asarray(2, dtype=np.int64), 'schema_version'),
        ('artifact_kind', np.asarray('wrong'), 'artifact kind'),
        ('order', np.asarray('original_trace_order'), 'order'),
        ('mode', np.asarray('manual'), 'mode'),
    ],
)
def test_load_geometry_linkage_artifact_rejects_invalid_scalar_contract(
    tmp_path: Path,
    field: str,
    value: np.ndarray,
    match: str,
) -> None:
    payload = _payload()
    payload[field] = value
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match=match):
        load_geometry_linkage_artifact(path)


def test_load_geometry_linkage_artifact_rejects_trace_shape_mismatch(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload['source_node_id_sorted'] = payload['source_node_id_sorted'][:-1]
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match='source_node_id_sorted shape mismatch'):
        load_geometry_linkage_artifact(path)


def test_load_geometry_linkage_artifact_rejects_expected_n_traces_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_payload(tmp_path, _payload())

    with pytest.raises(ValueError, match='n_traces mismatch'):
        load_geometry_linkage_artifact(path, expected_n_traces=6)


def test_load_geometry_linkage_artifact_rejects_expected_key_bytes_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_payload(tmp_path, _payload(metadata=_metadata()))

    with pytest.raises(ValueError, match='key1_byte mismatch'):
        load_geometry_linkage_artifact(path, expected_key1_byte=188)


def test_load_geometry_linkage_artifact_rejects_float_id_dtype(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload['source_endpoint_id_sorted'] = payload[
        'source_endpoint_id_sorted'
    ].astype(np.float64)
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match='source_endpoint_id_sorted.*integer dtype'):
        load_geometry_linkage_artifact(path)


@pytest.mark.parametrize(
    ('field', 'match'),
    [
        ('source_endpoint_id', 'source_endpoint_id must be 0-based contiguous'),
        ('receiver_endpoint_id', 'receiver_endpoint_id must be 0-based contiguous'),
    ],
)
def test_load_geometry_linkage_artifact_rejects_non_contiguous_endpoint_ids(
    tmp_path: Path,
    field: str,
    match: str,
) -> None:
    payload = _payload()
    payload[field] = payload[field].copy()
    payload[field][-1] += 1
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match=match):
        load_geometry_linkage_artifact(path)


def test_load_geometry_linkage_artifact_rejects_endpoint_id_sorted_out_of_range(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload['receiver_endpoint_id_sorted'] = payload[
        'receiver_endpoint_id_sorted'
    ].copy()
    payload['receiver_endpoint_id_sorted'][0] = payload['n_receiver_endpoints'].item()
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match='receiver_endpoint_id_sorted values'):
        load_geometry_linkage_artifact(path)


@pytest.mark.parametrize(
    ('field', 'match'),
    [
        ('source_endpoint_trace_count', 'source_endpoint_trace_count'),
        (
            'receiver_endpoint_first_sorted_trace_index',
            'receiver_endpoint_first_sorted_trace_index',
        ),
    ],
)
def test_load_geometry_linkage_artifact_validates_endpoint_trace_metadata(
    tmp_path: Path,
    field: str,
    match: str,
) -> None:
    payload = _payload()
    payload[field] = payload[field].copy()
    payload[field][0] += 1
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match=match):
        load_geometry_linkage_artifact(path)


def test_load_geometry_linkage_artifact_rejects_non_contiguous_node_ids(
    tmp_path: Path,
) -> None:
    payload = _payload()
    source_nodes = payload['source_node_id_by_endpoint'].copy()
    source_nodes[-1] = 5
    payload['source_node_id_by_endpoint'] = source_nodes
    payload['source_node_id_sorted'] = source_nodes[payload['source_endpoint_id_sorted']]
    payload['n_nodes'] = np.asarray(6, dtype=np.int64)
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match='0-based contiguous|end at n_nodes'):
        load_geometry_linkage_artifact(path)


@pytest.mark.parametrize(
    ('field', 'match'),
    [
        (
            'source_node_id_sorted',
            'source_node_id_sorted does not match source_node_id_by_endpoint',
        ),
        (
            'receiver_node_id_sorted',
            'receiver_node_id_sorted does not match receiver_node_id_by_endpoint',
        ),
    ],
)
def test_load_geometry_linkage_artifact_rejects_sorted_node_mapping_mismatch(
    tmp_path: Path,
    field: str,
    match: str,
) -> None:
    payload = _payload()
    payload[field] = payload[field].copy()
    payload[field][0] += 1
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match=match):
        load_geometry_linkage_artifact(path)


def test_load_geometry_linkage_artifact_rejects_trace_coordinate_mapping_mismatch(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload['source_x_m_sorted'] = payload['source_x_m_sorted'].copy()
    payload['source_x_m_sorted'][0] += 10.0
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match='source_x_m_sorted does not match'):
        load_geometry_linkage_artifact(path)


def test_load_geometry_linkage_artifact_rejects_non_finite_coordinate(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload['receiver_endpoint_y_m'] = payload['receiver_endpoint_y_m'].copy()
    payload['receiver_endpoint_y_m'][0] = np.nan
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match='receiver_endpoint_y_m.*finite'):
        load_geometry_linkage_artifact(path)


@pytest.mark.parametrize(
    ('value', 'match'),
    [(np.inf, 'Inf'), (-1.0, 'greater than or equal to 0')],
)
def test_load_geometry_linkage_artifact_rejects_invalid_record_distance(
    tmp_path: Path,
    value: float,
    match: str,
) -> None:
    payload = _payload()
    linked = int(np.flatnonzero(~np.isnan(payload['record_distance_m']))[0])
    payload['record_distance_m'] = payload['record_distance_m'].copy()
    payload['record_distance_m'][linked] = value
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match=match):
        load_geometry_linkage_artifact(path)


def test_load_geometry_linkage_artifact_validates_record_order(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload['record_endpoint_id'] = payload['record_endpoint_id'].copy()
    payload['record_endpoint_id'][0], payload['record_endpoint_id'][1] = 1, 0
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match='ordered by receiver endpoint id'):
        load_geometry_linkage_artifact(path)


def test_load_geometry_linkage_artifact_validates_record_node_ids(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload['record_node_id'] = payload['record_node_id'].copy()
    payload['record_node_id'][0] += 1
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match='record_node_id receiver rows'):
        load_geometry_linkage_artifact(path)


def test_load_geometry_linkage_artifact_rejects_unknown_record_method(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload['record_method'] = payload['record_method'].copy()
    payload['record_method'][0] = 'unknown'
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match="record_method contains unsupported value: 'unknown'"):
        load_geometry_linkage_artifact(path)


def test_load_geometry_linkage_artifact_validates_none_mode_methods(
    tmp_path: Path,
) -> None:
    payload = _payload(mode='none')
    payload['record_method'] = payload['record_method'].copy()
    payload['record_method'][0] = 'receiver_seed'
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match='none_mode_receiver_independent'):
        load_geometry_linkage_artifact(path)


def test_load_geometry_linkage_artifact_validates_auto_threshold_methods(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload['record_method'] = payload['record_method'].astype('<U64')
    payload['record_method'][0] = 'none_mode_receiver_independent'
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match='receiver_seed'):
        load_geometry_linkage_artifact(path)


def test_load_geometry_linkage_artifact_validates_summary_counts_against_records(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload['n_receiver_anchor_links'] = np.asarray(0, dtype=np.int64)
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match='n_receiver_anchor_links does not match'):
        load_geometry_linkage_artifact(path)


def test_load_geometry_linkage_from_job_dir_uses_standard_filename(
    tmp_path: Path,
) -> None:
    _write_payload(tmp_path, _payload())

    loaded = load_geometry_linkage_from_job_dir(tmp_path)

    assert loaded.path == tmp_path / GEOMETRY_LINKAGE_NPZ_NAME


def test_load_geometry_linkage_trace_node_mapping_returns_minimal_mapping(
    tmp_path: Path,
) -> None:
    path = _write_payload(tmp_path, _payload())

    mapping = load_geometry_linkage_trace_node_mapping(path, expected_n_traces=5)

    assert mapping.n_traces == 5
    assert mapping.n_nodes == 4
    np.testing.assert_array_equal(mapping.source_node_id_sorted, [0, 2, 2, 3, 0])
    np.testing.assert_array_equal(mapping.receiver_node_id_sorted, [0, 1, 1, 0, 1])


def _payload(
    *,
    mode: str = 'auto_threshold',
    metadata: GeometryLinkageArtifactMetadata | None = None,
) -> dict[str, np.ndarray]:
    tables = _tables()
    options = (
        GeometryLinkageOptions(
            mode='auto_threshold',
            threshold_m=1.1,
            receiver_location_interval_m=25.0,
        )
        if mode == 'auto_threshold'
        else GeometryLinkageOptions(mode='none')
    )
    linkage = build_geometry_linkage(tables, options)
    return build_geometry_linkage_solution_arrays(
        tables,
        linkage,
        metadata=metadata,
    )


def _tables():
    headers = GeometryLinkageHeaders(
        source_x=np.asarray([0.0, 10.0, 11.0, 30.0, 0.0], dtype=np.float64),
        source_y=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        receiver_x=np.asarray([0.0, 100.0, 100.0, 0.0, 100.0], dtype=np.float64),
        receiver_y=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        coordinate_scalar=np.ones(5, dtype=np.int64),
        checked_bytes=(71, 73, 77, 81, 85),
    )
    return build_endpoint_geometry_tables(headers)


def _metadata() -> GeometryLinkageArtifactMetadata:
    return GeometryLinkageArtifactMetadata(
        job_id='linkage-job',
        input_file_id='file-id',
        key1_byte=189,
        key2_byte=193,
        source_x_byte=73,
        source_y_byte=77,
        receiver_x_byte=81,
        receiver_y_byte=85,
        coordinate_scalar_byte=71,
        header_source_segy_path='/data/input.sgy',
    )


def _write_payload(tmp_path: Path, payload: dict[str, np.ndarray]) -> Path:
    path = tmp_path / GEOMETRY_LINKAGE_NPZ_NAME
    with path.open('wb') as handle:
        np.savez(handle, **payload)
    return path
