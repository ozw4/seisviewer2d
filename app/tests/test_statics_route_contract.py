from __future__ import annotations

from app.main import app

EXPECTED_STATICS_ROUTES = [
    ('POST', '/statics/datum/apply'),
    ('POST', '/statics/first-break/qc'),
    ('POST', '/statics/linkage/build'),
    ('POST', '/statics/residual/apply'),
    ('POST', '/statics/time-term/apply'),
    ('POST', '/statics/refraction/apply'),
    ('POST', '/statics/refraction/apply-with-picks'),
    ('POST', '/statics/refraction/validate-with-picks'),
    ('POST', '/statics/refraction/qc'),
    ('POST', '/statics/refraction/qc/endpoints'),
    ('POST', '/statics/refraction/qc/pick-map'),
    ('POST', '/statics/refraction/qc/drilldown'),
    ('POST', '/statics/refraction/qc/station-structure'),
    ('POST', '/statics/refraction/qc/gather-preview'),
    ('POST', '/statics/refraction/static-table/apply'),
    ('POST', '/statics/refraction/export'),
    ('GET', '/statics/job/{job_id}/status'),
    ('POST', '/statics/job/{job_id}/cancel'),
    ('GET', '/statics/job/{job_id}/files'),
    ('GET', '/statics/job/{job_id}/download'),
]


def test_statics_route_contract_method_and_path() -> None:
    actual = []
    for route in app.routes:
        path = getattr(route, 'path', None)
        methods = getattr(route, 'methods', None)
        if not path or not path.startswith('/statics') or not methods:
            continue
        for method in sorted(methods - {'HEAD', 'OPTIONS'}):
            actual.append((method, path))

    assert sorted(actual) == sorted(EXPECTED_STATICS_ROUTES)


def test_statics_route_contract_names_are_stable() -> None:
    actual = {
        (method, route.path): route.name
        for route in app.routes
        if getattr(route, 'path', '').startswith('/statics')
        for method in sorted(getattr(route, 'methods', set()) - {'HEAD', 'OPTIONS'})
    }

    assert actual[('POST', '/statics/datum/apply')] == 'datum_static_apply'
    assert actual[('POST', '/statics/first-break/qc')] == 'first_break_qc'
    assert actual[('POST', '/statics/linkage/build')] == 'static_linkage_build'
    assert actual[('POST', '/statics/residual/apply')] == 'residual_static_apply'
    assert actual[('POST', '/statics/time-term/apply')] == 'time_term_static_apply'
    assert actual[('POST', '/statics/refraction/apply')] == 'refraction_static_apply'
    assert (
        actual[('POST', '/statics/refraction/apply-with-picks')]
        == 'refraction_static_apply_with_picks'
    )
    assert (
        actual[('POST', '/statics/refraction/validate-with-picks')]
        == 'refraction_static_validate_with_picks'
    )
    assert actual[('POST', '/statics/refraction/qc')] == 'refraction_static_qc_bundle'
    assert (
        actual[('POST', '/statics/refraction/qc/endpoints')]
        == 'refraction_static_qc_endpoint_search'
    )
    assert (
        actual[('POST', '/statics/refraction/qc/pick-map')]
        == 'refraction_static_qc_pick_map'
    )
    assert (
        actual[('POST', '/statics/refraction/qc/drilldown')]
        == 'refraction_static_qc_drilldown'
    )
    assert (
        actual[('POST', '/statics/refraction/qc/station-structure')]
        == 'refraction_static_qc_station_structure'
    )
    assert (
        actual[('POST', '/statics/refraction/qc/gather-preview')]
        == 'refraction_static_gather_preview'
    )
    assert (
        actual[('POST', '/statics/refraction/static-table/apply')]
        == 'refraction_static_table_apply'
    )
    assert actual[('POST', '/statics/refraction/export')] == 'refraction_static_export'
    assert actual[('GET', '/statics/job/{job_id}/status')] == 'static_job_status'
    assert actual[('POST', '/statics/job/{job_id}/cancel')] == 'static_job_cancel'
    assert actual[('GET', '/statics/job/{job_id}/files')] == 'static_job_files'
    assert actual[('GET', '/statics/job/{job_id}/download')] == 'static_job_download'
