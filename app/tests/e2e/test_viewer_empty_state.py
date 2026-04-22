import pytest


@pytest.mark.e2e
def test_viewer_without_file_id_shows_empty_state_and_link(
    page, base_url, e2e_debug
):
    page.set_default_timeout(60_000)

    section_requests: list[str] = []

    def on_request(req):
        if any(
            path in req.url
            for path in ("/get_key1_values", "/get_section_window_bin")
        ):
            section_requests.append(f"{req.method} {req.url}")

    page.on("request", on_request)

    page.goto(f"{base_url}/", wait_until="domcontentloaded")

    page.locator("#viewerEmptyState:not([hidden])").wait_for(timeout=60_000)
    assert page.locator("#viewerEmptyStateTitle").text_content() == "No dataset open"
    assert page.locator("#viewerEmptyStateAction").text_content() == "Open SEG-Y"
    assert page.locator("#viewerEmptyStateAction").get_attribute("href") == "/upload"

    page.wait_for_timeout(750)
    assert section_requests == []

    page.locator("#viewerEmptyStateAction").click()
    page.wait_for_url(f"{base_url}/upload", timeout=60_000)

    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_viewer_with_unavailable_file_id_shows_recovery_state(
    page, base_url, e2e_debug
):
    page.set_default_timeout(60_000)

    section_requests: list[str] = []

    def on_request(req):
        if any(
            path in req.url
            for path in ("/get_key1_values", "/get_section_window_bin")
        ):
            section_requests.append(f"{req.method} {req.url}")

    page.on("request", on_request)

    page.goto(f"{base_url}/?file_id=missing-file-id", wait_until="domcontentloaded")

    page.locator("#viewerEmptyState:not([hidden])").wait_for(timeout=60_000)
    assert page.locator("#viewerEmptyStateTitle").text_content() == "Dataset unavailable"
    assert (
        page.locator("#viewerEmptyStateDescription").text_content()
        == "The previously selected dataset could not be opened. Open or upload a SEG-Y file to continue."
    )
    assert page.locator("#viewerEmptyStateAction").get_attribute("href") == "/upload"

    page.wait_for_timeout(750)
    assert section_requests == []
    assert e2e_debug.unexpected_404(allow_404=("favicon.ico", ".map", "/file_info")) == []
    assert e2e_debug.page_error == []
    assert e2e_debug.unexpected_request_failed() == []
