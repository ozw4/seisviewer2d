from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from app.statics.refraction.artifacts.v1 import (
    REFRACTION_V1_ESTIMATES_CSV_NAME,
    REFRACTION_V1_QC_JSON_NAME,
    write_refraction_v1_artifacts,
)
from app.statics.refraction.application.core_options import (
    core_input_model_from_app,
    first_layer_options_from_request,
)
from seis_statics.refraction.v1 import estimate_global_v1_from_direct_arrivals
from app.tests._refraction_static_synthetic import (
    SYNTHETIC_V1_M_S,
    SYNTHETIC_V1_TOLERANCE_M_S,
    synthetic_direct_arrival_input_model,
    synthetic_first_layer_request,
)


def test_v1_estimate_global_from_direct_arrivals(tmp_path: Path) -> None:
    result = estimate_global_v1_from_direct_arrivals(
        input_model=core_input_model_from_app(synthetic_direct_arrival_input_model()),
        first_layer=first_layer_options_from_request(synthetic_first_layer_request()),
    )

    assert result.resolved_weathering_velocity_m_s == pytest.approx(
        SYNTHETIC_V1_M_S,
        abs=SYNTHETIC_V1_TOLERANCE_M_S,
    )
    assert result.qc['v1_status'] == 'estimated'
    assert result.qc['n_used_groups'] == 6
    assert result.qc['group_status_counts'] == {'ok': 6}
    assert set(result.group_status.tolist()) == {'ok'}
    assert np.any(result.group_n_used < result.group_n_candidates)

    paths = write_refraction_v1_artifacts(tmp_path, result)
    assert paths['qc_json'].name == REFRACTION_V1_QC_JSON_NAME
    assert paths['estimates_csv'].name == REFRACTION_V1_ESTIMATES_CSV_NAME

    qc = json.loads((tmp_path / REFRACTION_V1_QC_JSON_NAME).read_text())
    assert qc['resolved_weathering_velocity_m_s'] == pytest.approx(
        SYNTHETIC_V1_M_S,
        abs=SYNTHETIC_V1_TOLERANCE_M_S,
    )
    with (tmp_path / REFRACTION_V1_ESTIMATES_CSV_NAME).open(newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == (
        'group_kind',
        'group_key',
        'n_candidates',
        'n_used',
        'offset_min_m',
        'offset_max_m',
        'slope_s_per_m',
        'v1_m_s',
        'intercept_s',
        'residual_rms_ms',
        'residual_mad_ms',
        'status',
    )
    assert len(rows) == 6
    assert rows[0]['group_kind'] == 'source_endpoint'
    assert rows[0]['n_candidates'] == '6'
    assert rows[0]['n_used'] == '6'
    assert rows[0]['status'] == 'ok'
    assert float(rows[0]['v1_m_s']) == pytest.approx(
        SYNTHETIC_V1_M_S,
        abs=SYNTHETIC_V1_TOLERANCE_M_S,
    )
