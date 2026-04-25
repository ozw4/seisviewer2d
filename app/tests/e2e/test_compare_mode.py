import pytest


@pytest.mark.e2e
def test_playwright_compare_mode_toggle_and_diff_panel(
    page, base_url, tiny_segy_path, e2e_debug
):
    page.set_default_timeout(60_000)

    page.goto(f"{base_url}/upload", wait_until="domcontentloaded")
    page.select_option("#key1_byte", "189")
    page.select_option("#key2_byte", "193")
    page.set_input_files("#upload_segy", str(tiny_segy_path))
    page.click("#upload_btn")
    page.wait_for_url("**/?file_id=*", timeout=60_000)
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_function(
        "() => document.getElementById('viewerEmptyState')?.hidden === true"
    )

    with page.expect_response(
        lambda r: "/get_section_window_bin" in r.url and r.status == 200,
        timeout=60_000,
    ):
        page.check("#compareModeToggle")

    page.wait_for_function(
        "() => document.getElementById('plot')?.__svComparePanelCount === 3"
    )
    assert page.locator("#compareSourceA").input_value() == "raw"
    assert page.locator("#compareSourceB").input_value() == "raw"

    page.uncheck("#compareShowDiff")
    page.wait_for_function(
        "() => document.getElementById('plot')?.__svComparePanelCount === 2"
    )
    assert page.locator("#compareSourceA").input_value() == "raw"
    assert page.locator("#compareSourceB").input_value() == "raw"

    page.check("#compareShowDiff")
    page.wait_for_function(
        "() => document.getElementById('plot')?.__svComparePanelCount === 3"
    )

    e2e_debug.assert_clean()
