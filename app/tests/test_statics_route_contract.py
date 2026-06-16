from __future__ import annotations

import importlib

from app.main import app
from app.tests.route_helpers import iter_app_routes

EXPECTED_STATICS_ROUTE_CONTRACT = {
    ('POST', '/statics/datum/apply'): 'datum_static_apply',
    ('POST', '/statics/first-break/qc'): 'first_break_qc',
    ('POST', '/statics/linkage/build'): 'static_linkage_build',
    ('POST', '/statics/residual/apply'): 'residual_static_apply',
    ('POST', '/statics/time-term/apply'): 'time_term_static_apply',
    ('POST', '/statics/refraction/apply'): 'refraction_static_apply',
    (
        'POST',
        '/statics/refraction/apply-with-picks',
    ): 'refraction_static_apply_with_picks',
    (
        'POST',
        '/statics/refraction/validate-with-picks',
    ): 'refraction_static_validate_with_picks',
    ('POST', '/statics/refraction/qc'): 'refraction_static_qc_bundle',
    (
        'POST',
        '/statics/refraction/qc/endpoints',
    ): 'refraction_static_qc_endpoint_search',
    ('POST', '/statics/refraction/qc/pick-map'): 'refraction_static_qc_pick_map',
    ('POST', '/statics/refraction/qc/drilldown'): 'refraction_static_qc_drilldown',
    (
        'POST',
        '/statics/refraction/qc/station-structure',
    ): 'refraction_static_qc_station_structure',
    (
        'POST',
        '/statics/refraction/qc/gather-preview',
    ): 'refraction_static_gather_preview',
    (
        'POST',
        '/statics/refraction/static-table/apply',
    ): 'refraction_static_table_apply',
    ('POST', '/statics/refraction/export'): 'refraction_static_export',
    ('GET', '/statics/job/{job_id}/status'): 'job_status',
    ('POST', '/statics/job/{job_id}/cancel'): 'job_cancel',
    ('GET', '/statics/job/{job_id}/files'): 'job_files',
    ('GET', '/statics/job/{job_id}/download'): 'job_download',
}


def test_statics_route_contract_method_path_and_endpoint_name() -> None:
    actual = {}
    for route in iter_app_routes(app.routes):
        path = getattr(route, 'path', None)
        methods = getattr(route, 'methods', None)
        if not path or not path.startswith('/statics') or not methods:
            continue
        for method in sorted(methods - {'HEAD', 'OPTIONS'}):
            actual[(method, path)] = route.name

    assert actual == EXPECTED_STATICS_ROUTE_CONTRACT


def test_statics_router_package_exports_only_router() -> None:
    statics = importlib.import_module('app.api.routers.statics')

    assert statics.__all__ == ['router']
    assert not hasattr(statics, 'start_job_thread')
    assert not hasattr(statics, 'run_datum_static_apply_job')
