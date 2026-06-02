import pytest


@pytest.mark.e2e
def test_playwright_viewer_perf_overlay_is_absent_by_default(
    page, base_url, e2e_debug
):
    page.goto(f"{base_url}/", wait_until="domcontentloaded")

    assert page.locator(".sv-viewer-perf-overlay").count() == 0
    assert e2e_debug.console_error == []
    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_playwright_viewer_perf_overlay_is_visible_and_non_interactive(
    page, base_url, e2e_debug
):
    page.goto(f"{base_url}/?perf=1", wait_until="domcontentloaded")

    overlay = page.locator(".sv-viewer-perf-overlay")
    overlay.wait_for(state="visible")

    assert overlay.evaluate("node => getComputedStyle(node).pointerEvents") == "none"
    text = overlay.text_content()
    assert "fetch" in text
    assert "decode" in text
    assert "render" in text
    assert "overlay" in text
    assert "request started/completed/aborted/stale" in text
    assert "key1" in text
    assert "layer" in text
    assert "mode" in text
    assert e2e_debug.console_error == []
    e2e_debug.assert_clean()
