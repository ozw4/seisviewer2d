from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import get_args, get_type_hints

import pytest

from app.services.refraction_static_export_types import (
    REFRACTION_STATIC_EXPORT_SIGN_CONVENTION,
    REFRACTION_STATIC_EXPORT_UNITS,
    RefractionStaticCanonicalTableRow,
    RefractionStaticEndpointExportRow,
    RefractionStaticExportBundle,
    RefractionStaticExportFormatName,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_IMPORTS = {
    'app.api.routers',
    'app.api.schemas',
    'app.main',
    'app.services.refraction_static_service',
    'app.trace_store.reader',
    'segyio',
}
_FORBIDDEN_SOURCE_STRINGS = _FORBIDDEN_IMPORTS | {
    'BaseModel',
    'TraceStoreSectionReader',
    'numpy',
}


def test_refraction_static_export_types_import_dependency_light() -> None:
    code = f"""
from __future__ import annotations

import importlib
import json
import sys

for name in {sorted(_FORBIDDEN_IMPORTS)!r}:
    sys.modules.pop(name, None)

module = importlib.import_module('app.services.refraction_static_export_types')
assert module.RefractionStaticEndpointExportRow is not None
assert module.RefractionStaticExportBundle is not None
assert module.RefractionStaticCanonicalTableRow is not None

forbidden = set({sorted(_FORBIDDEN_IMPORTS)!r})
print(json.dumps(sorted(name for name in sys.modules if name in forbidden)))
"""
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert json.loads(result.stdout) == []

    module = importlib.import_module('app.services.refraction_static_export_types')
    source = Path(module.__file__ or '').read_text(encoding='utf-8')
    for forbidden in _FORBIDDEN_SOURCE_STRINGS:
        assert forbidden not in source


def test_export_row_allows_multilayer_fields() -> None:
    row = RefractionStaticEndpointExportRow(
        endpoint_kind='source',
        endpoint_key='source:1001',
        station_id=1001,
        node_id=7,
        x_m=1234.5,
        y_m=6789.0,
        elevation_m=42.0,
        t1_s=0.011,
        t2_s=0.022,
        t3_s=0.033,
        v1_m_s=800.0,
        v2_m_s=2400.0,
        v3_m_s=3600.0,
        vsub_m_s=5000.0,
        sh1_m=8.0,
        sh2_m=12.0,
        sh3_m=16.0,
        weathering_correction_s=-0.004,
        elevation_correction_s=0.001,
        field_correction_s=0.002,
        total_applied_shift_s=-0.001,
        static_status='ok',
    )

    assert row.endpoint_kind == 'source'
    assert row.t2_s == pytest.approx(0.022)
    assert row.t3_s == pytest.approx(0.033)
    assert row.vsub_m_s == pytest.approx(5000.0)
    assert row.sh3_m == pytest.approx(16.0)

    row_fields = {field.name for field in fields(RefractionStaticEndpointExportRow)}
    assert {
        't1_s',
        't2_s',
        't3_s',
        'v1_m_s',
        'v2_m_s',
        'v3_m_s',
        'vsub_m_s',
        'sh1_m',
        'sh2_m',
        'sh3_m',
        'field_correction_s',
        'total_applied_shift_s',
    } <= row_fields

    hints = get_type_hints(RefractionStaticEndpointExportRow)
    assert hints['endpoint_kind'].__args__ == ('source', 'receiver')
    assert hints['endpoint_key'] is str

    with pytest.raises(FrozenInstanceError):
        row.static_status = 'changed'


def test_export_bundle_carries_sign_convention() -> None:
    source = RefractionStaticEndpointExportRow(
        endpoint_kind='source',
        endpoint_key='source:1001',
        total_applied_shift_s=-0.001,
    )
    receiver = RefractionStaticEndpointExportRow(
        endpoint_kind='receiver',
        endpoint_key='receiver:2001',
        total_applied_shift_s=0.002,
    )
    bundle = RefractionStaticExportBundle(
        source_rows=(source,),
        receiver_rows=(receiver,),
    )

    assert bundle.source_rows == (source,)
    assert bundle.receiver_rows == (receiver,)
    assert bundle.sign_convention == REFRACTION_STATIC_EXPORT_SIGN_CONVENTION
    assert bundle.sign_convention == 'corrected(t) = raw(t - shift_s)'
    assert bundle.units == REFRACTION_STATIC_EXPORT_UNITS

    hints = get_type_hints(RefractionStaticExportBundle)
    assert hints['source_rows'] == tuple[RefractionStaticEndpointExportRow, ...]
    assert hints['receiver_rows'] == tuple[RefractionStaticEndpointExportRow, ...]


def test_canonical_static_table_row_uses_m5_required_fields() -> None:
    row = RefractionStaticCanonicalTableRow(
        endpoint_kind='receiver',
        endpoint_key='receiver:2001',
        endpoint_id=2001,
        applied_shift_ms=-1.5,
        static_status='ok',
        source_job_id='refraction-job',
        sign_convention=REFRACTION_STATIC_EXPORT_SIGN_CONVENTION,
        t1_ms=11.0,
        t2_ms=22.0,
        t3_ms=33.0,
        vsub_m_s=5000.0,
        sh3_weathering_thickness_m=16.0,
        total_applied_shift_ms=-1.5,
    )

    assert row.format_name == 'canonical_static_table'
    assert row.format_version == 1
    assert row.endpoint_kind == 'receiver'
    assert row.applied_shift_ms == pytest.approx(-1.5)
    assert row.sign_convention == REFRACTION_STATIC_EXPORT_SIGN_CONVENTION
    assert row.t3_ms == pytest.approx(33.0)
    assert row.vsub_m_s == pytest.approx(5000.0)

    hints = get_type_hints(RefractionStaticCanonicalTableRow)
    assert hints['format_name'].__args__ == ('canonical_static_table',)
    assert hints['endpoint_kind'].__args__ == ('source', 'receiver')


def test_export_format_name_matches_m5_formats() -> None:
    assert get_args(RefractionStaticExportFormatName) == (
        'canonical_static_table',
        'lsst',
        'lsst_plus',
        'time_term_spreadsheet',
        'first_break_time',
    )
