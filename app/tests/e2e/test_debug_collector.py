from app.tests.e2e.conftest import E2EDebug


def test_unexpected_request_failed_allows_only_aborted_get_section_window_fetches(
    tmp_path,
):
    dbg = E2EDebug(
        artifact_dir=tmp_path,
        request_failed=[
            "REQ_FAILED GET http://127.0.0.1:8000/get_section_window_bin?file_id=x (net::ERR_ABORTED)",
            "REQ_FAILED GET http://127.0.0.1:8000/get_section_window_bin?file_id=x (net::ERR_FAILED)",
            "REQ_FAILED POST http://127.0.0.1:8000/get_section_window_bin?file_id=x (net::ERR_ABORTED)",
            "REQ_FAILED POST http://127.0.0.1:8000/open_segy (net::ERR_ABORTED)",
            "REQ_FAILED GET http://127.0.0.1:8000/favicon.ico (net::ERR_ABORTED)",
        ],
    )

    assert dbg.unexpected_request_failed() == [
        "REQ_FAILED GET http://127.0.0.1:8000/get_section_window_bin?file_id=x (net::ERR_FAILED)",
        "REQ_FAILED POST http://127.0.0.1:8000/get_section_window_bin?file_id=x (net::ERR_ABORTED)",
    ]
    assert dbg.unexpected_request_failed(allow_window_fetch_aborted=False) == [
        "REQ_FAILED GET http://127.0.0.1:8000/get_section_window_bin?file_id=x (net::ERR_ABORTED)",
        "REQ_FAILED GET http://127.0.0.1:8000/get_section_window_bin?file_id=x (net::ERR_FAILED)",
        "REQ_FAILED POST http://127.0.0.1:8000/get_section_window_bin?file_id=x (net::ERR_ABORTED)",
    ]
