from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import RefractionStaticApplyRequest
from app.statics.refraction.artifacts import (
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    write_refraction_static_artifacts,
)
from app.statics.refraction.artifacts.t1lsst import (
    REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
)
from app.statics.refraction.domain.t1lsst import (
    T1LSST_SIGN_CONVENTION,
    RefractionT1LSSTError,
    compute_t1lsst_1layer_thickness,
    compute_t1lsst_1layer_weathering_correction,
)
from app.statics.refraction.application.weathering import (
    compute_weathering_thickness_from_half_intercept_time,
)
from app.statics.refraction.application.weathering_replacement import (
    compute_weathering_replacement_shift_s,
)
from app.tests._refraction_static_synthetic import (
    SYNTHETIC_SH1_TOLERANCE_M,
    SYNTHETIC_T1_TOLERANCE_MS,
    SYNTHETIC_V2_M_S,
    SYNTHETIC_V2_TOLERANCE_M_S,
    SYNTHETIC_WCOR_TOLERANCE_MS,
    expected_sh1_m_for_node,
    expected_t1_s_for_node,
    expected_wcor_s_for_node,
    run_synthetic_refraction_statics,
    synthetic_refraction_apply_request,
)
from app.tests._refraction_static_artifact_helpers import _request, _result


def _t1lsst_request() -> RefractionStaticApplyRequest:
    payload = _request().model_dump(mode='json')
    payload['conversion'] = {'mode': 't1lsst_1layer'}
    return RefractionStaticApplyRequest.model_validate(payload)


def test_refraction_static_conversion_request_accepts_t1lsst_1layer() -> None:
    req = _t1lsst_request()

    assert req.conversion.mode == 't1lsst_1layer'


def test_t1lsst_1layer_scalar_formula() -> None:
    t1_s = 0.010
    v1_m_s = 800.0
    v2_m_s = 2400.0

    sh1 = compute_t1lsst_1layer_thickness(
        np.asarray([t1_s]),
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
    )[0]
    wcor = compute_t1lsst_1layer_weathering_correction(
        np.asarray([sh1]),
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
    )[0]

    expected_sh1 = t1_s * v2_m_s * v1_m_s / np.sqrt(v2_m_s**2 - v1_m_s**2)
    assert sh1 == pytest.approx(expected_sh1)
    assert wcor == pytest.approx(expected_sh1 * (1.0 / v2_m_s - 1.0 / v1_m_s))
    assert wcor < 0.0


def test_t1lsst_1layer_vector_formula() -> None:
    t1_s = np.asarray([0.010, 0.012, 0.014], dtype=np.float64)
    v1_m_s = 800.0
    v2_m_s = np.asarray([2200.0, 2500.0, 3000.0], dtype=np.float64)

    sh1 = compute_t1lsst_1layer_thickness(
        t1_s,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
    )
    wcor = compute_t1lsst_1layer_weathering_correction(
        sh1,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
    )

    np.testing.assert_allclose(
        sh1,
        t1_s * v2_m_s * v1_m_s / np.sqrt(v2_m_s**2 - v1_m_s**2),
    )
    np.testing.assert_allclose(wcor, sh1 * (1.0 / v2_m_s - 1.0 / v1_m_s))


def test_t1lsst_1layer_matches_existing_weathering_thickness() -> None:
    t1_s = np.asarray([0.010, 0.012, 0.014], dtype=np.float64)

    t1lsst = compute_t1lsst_1layer_thickness(
        t1_s,
        v1_m_s=800.0,
        v2_m_s=2500.0,
    )
    existing = compute_weathering_thickness_from_half_intercept_time(
        half_intercept_time_s=t1_s,
        weathering_velocity_m_s=800.0,
        bedrock_velocity_m_s=2500.0,
    )

    np.testing.assert_allclose(t1lsst, existing)


def test_t1lsst_1layer_matches_existing_weathering_replacement_shift() -> None:
    sh1_m = np.asarray([10.0, 12.0, 14.0], dtype=np.float64)

    t1lsst = compute_t1lsst_1layer_weathering_correction(
        sh1_m,
        v1_m_s=800.0,
        v2_m_s=2500.0,
    )
    existing = compute_weathering_replacement_shift_s(
        weathering_thickness_m=sh1_m,
        weathering_velocity_m_s=800.0,
        bedrock_velocity_m_s=2500.0,
    )

    np.testing.assert_allclose(t1lsst, existing)


def test_t1lsst_1layer_matches_existing_weathering_replacement(
    tmp_path: Path,
) -> None:
    req = synthetic_refraction_apply_request(conversion_mode='t1lsst_1layer')
    result = run_synthetic_refraction_statics(req=req)

    paths = write_refraction_static_artifacts(
        result=result,
        req=req,
        job_dir=tmp_path,
    )

    assert result.bedrock_velocity_m_s == pytest.approx(
        SYNTHETIC_V2_M_S,
        abs=SYNTHETIC_V2_TOLERANCE_M_S,
    )
    assert paths.refraction_t1lsst_1layer_components_csv is not None
    rows = _read_csv(paths.refraction_t1lsst_1layer_components_csv)
    assert rows

    for row in rows:
        node_id = int(row['node_id'])
        expected_t1_ms = expected_t1_s_for_node(node_id) * 1000.0
        expected_wcor_ms = expected_wcor_s_for_node(node_id) * 1000.0
        assert float(row['t1_ms']) == pytest.approx(
            expected_t1_ms,
            abs=SYNTHETIC_T1_TOLERANCE_MS,
        )
        assert float(row['sh1_weathering_thickness_m']) == pytest.approx(
            expected_sh1_m_for_node(node_id),
            abs=SYNTHETIC_SH1_TOLERANCE_M,
        )
        assert float(row['weathering_correction_ms']) == pytest.approx(
            expected_wcor_ms,
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )
        assert float(row['weathering_correction_ms']) < 0.0
        assert float(row['total_applied_shift_ms']) == pytest.approx(
            float(row['total_static_ms']),
            abs=SYNTHETIC_WCOR_TOLERANCE_MS,
        )


def test_t1lsst_1layer_rejects_v2_less_than_or_equal_v1() -> None:
    with pytest.raises(RefractionT1LSSTError, match='v2_m_s must be greater'):
        compute_t1lsst_1layer_thickness(
            np.asarray([0.010]),
            v1_m_s=800.0,
            v2_m_s=800.0,
        )

    with pytest.raises(RefractionT1LSSTError, match='v2_m_s must be greater'):
        compute_t1lsst_1layer_weathering_correction(
            np.asarray([10.0]),
            v1_m_s=800.0,
            v2_m_s=799.0,
        )


def test_t1lsst_1layer_artifact_contains_sign_convention(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_t1lsst_request(),
        job_dir=tmp_path,
    )

    t1lsst_path = paths.refraction_t1lsst_1layer_components_csv
    assert t1lsst_path is not None
    assert t1lsst_path.name == REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME
    assert t1lsst_path.is_file()
    assert REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME in paths.artifact_names

    rows, fieldnames = _read_csv_with_fieldnames(t1lsst_path)
    assert tuple(fieldnames) == (
        'endpoint_kind',
        'endpoint_key',
        'node_id',
        'x_m',
        'y_m',
        'surface_elevation_m',
        'floating_datum_elevation_m',
        'flat_datum_elevation_m',
        't1_ms',
        'v1_m_s',
        'v2_m_s',
        'sh1_weathering_thickness_m',
        'refractor_elevation_m',
        'weathering_correction_ms',
        'floating_datum_correction_ms',
        'flat_datum_correction_ms',
        'elevation_correction_ms',
        'total_static_ms',
        'total_applied_shift_ms',
        'solution_status',
        'weathering_status',
        'datum_status',
        'static_status',
        'sign_convention',
    )
    assert len(rows) == 4
    assert {row['endpoint_kind'] for row in rows} == {'source', 'receiver'}
    assert rows[0]['sign_convention'] == T1LSST_SIGN_CONVENTION
    assert rows[0]['endpoint_kind'] == 'source'
    assert rows[0]['endpoint_key'] == 's0'
    assert rows[0]['node_id'] == '0'
    assert float(rows[0]['t1_ms']) == pytest.approx(10.0)
    assert float(rows[0]['sh1_weathering_thickness_m']) == pytest.approx(10.0)
    assert float(rows[0]['weathering_correction_ms']) == pytest.approx(-8.5)
    assert float(rows[0]['total_applied_shift_ms']) == pytest.approx(
        float(rows[0]['total_static_ms'])
    )

    qc = json.loads(paths.qc_json.read_text(encoding='utf-8'))
    assert qc['sign_convention'] == {
        'trace_shift_s': T1LSST_SIGN_CONVENTION,
        'positive_shift': 'event appears later in corrected data',
        'negative_shift': 'event appears earlier in corrected data',
    }

    manifest = json.loads(
        (tmp_path / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(
            encoding='utf-8'
        )
    )
    assert REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME in {
        item['name'] for item in manifest['artifacts']
    }


def test_t1lsst_1layer_static_status_follows_invalid_weathering(
    tmp_path: Path,
) -> None:
    result = replace(
        _result(),
        node_weathering_status=np.asarray(
            ['invalid_weathering_thickness', 'zero_thickness', 'inactive'],
            dtype='<U32',
        ),
        source_weathering_thickness_m=np.asarray([np.nan, 12.0]),
        source_weathering_replacement_shift_s=np.asarray([np.nan, -0.0102]),
        source_refraction_shift_s=np.asarray([np.nan, 0.0023]),
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=_t1lsst_request(),
        job_dir=tmp_path,
    )

    assert paths.refraction_t1lsst_1layer_components_csv is not None
    rows = _read_csv(paths.refraction_t1lsst_1layer_components_csv)
    source = next(
        row
        for row in rows
        if row['endpoint_kind'] == 'source' and row['endpoint_key'] == 's0'
    )
    assert source['datum_status'] == 'ok'
    assert source['weathering_status'] == 'invalid_weathering_thickness'
    assert source['static_status'] == 'invalid_weathering_thickness'


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _read_csv_with_fieldnames(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])
