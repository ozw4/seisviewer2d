import pytest


def _collect_section_requests(page) -> list[str]:
    requests: list[str] = []

    def on_request(req):
        if any(
            path in req.url
            for path in ("/get_key1_values", "/get_section", "/get_section_window_bin")
        ):
            requests.append(f"{req.method} {req.url}")

    page.on("request", on_request)
    return requests


def _upload_tiny_dataset_and_render_plot(page, base_url, tiny_segy_path) -> None:
    page.goto(f"{base_url}/upload", wait_until="domcontentloaded")
    page.select_option("#key1_byte", "189")
    page.select_option("#key2_byte", "193")
    page.set_input_files("#upload_segy", str(tiny_segy_path))
    page.click("#analyzeHeadersBtn")
    page.wait_for_function("() => !document.getElementById('upload_btn')?.disabled")
    page.click("#upload_btn")

    page.wait_for_url("**/?file_id=*", timeout=120_000)
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_function(
        "() => document.getElementById('viewerEmptyState')?.hidden === true"
    )

    plot_btn = page.locator("button:has-text('Plot')")
    with page.expect_response(
        lambda r: "/get_section_window_bin" in r.url and r.status == 200,
        timeout=120_000,
    ):
        plot_btn.click()

    page.wait_for_selector("#plot.js-plotly-plot", timeout=120_000)
    page.wait_for_selector("#plot .plot-container", timeout=120_000)


@pytest.mark.e2e
def test_shortcuts_dialog_is_available_without_dataset_and_stays_local(
    page, base_url, e2e_debug
):
    page.set_default_timeout(60_000)
    section_requests = _collect_section_requests(page)

    page.goto(f"{base_url}/", wait_until="domcontentloaded")

    button = page.get_by_test_id("viewer-shortcuts-button")
    dialog = page.get_by_role("dialog", name="Keyboard shortcuts")

    button.wait_for(timeout=120_000)
    assert button.text_content() == "Shortcuts"

    button.click()
    dialog.wait_for(timeout=120_000)
    assert dialog.get_by_text("Navigation").is_visible()
    assert dialog.get_by_text("A / D").is_visible()
    assert dialog.get_by_text("Ctrl+Y / Ctrl+Shift+Z").is_visible()

    page.get_by_test_id("viewer-shortcuts-close").click()
    page.get_by_test_id("viewer-shortcuts-dialog").wait_for(state="hidden")

    page.keyboard.press("?")
    dialog.wait_for(timeout=120_000)

    page.keyboard.press("Escape")
    page.get_by_test_id("viewer-shortcuts-dialog").wait_for(state="hidden")

    page.wait_for_timeout(400)
    assert section_requests == []

    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_shortcuts_dialog_question_mark_respects_editable_focus(
    page, base_url, e2e_debug
):
    page.set_default_timeout(60_000)

    page.goto(f"{base_url}/", wait_until="domcontentloaded")

    dialog = page.get_by_test_id("viewer-shortcuts-dialog")
    page.keyboard.press("?")
    dialog.wait_for(timeout=120_000)
    page.keyboard.press("Escape")
    dialog.wait_for(state="hidden")

    page.locator("#key1_val_display").focus()
    page.keyboard.press("?")
    page.wait_for_timeout(200)
    assert dialog.is_hidden()

    page.locator("#scalingMode").focus()
    page.keyboard.press("?")
    page.wait_for_timeout(200)
    assert dialog.is_hidden()

    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_shortcuts_dialog_does_not_change_alt_pan_viewer_state(
    page, base_url, tiny_segy_path, e2e_debug
):
    page.set_default_timeout(60_000)

    _upload_tiny_dataset_and_render_plot(page, base_url, tiny_segy_path)

    button = page.get_by_test_id("viewer-shortcuts-button")
    dialog = page.get_by_test_id("viewer-shortcuts-dialog")

    page.keyboard.down("Alt")
    page.wait_for_function(
        """() => {
          const gd = document.getElementById('plot');
          return gd?._fullLayout?.dragmode === 'pan'
            && gd?._fullLayout?.xaxis?.fixedrange === false
            && gd?._fullLayout?.yaxis?.fixedrange === false;
        }"""
    )

    button.click()
    dialog.wait_for(timeout=120_000)
    page.wait_for_function(
        """() => {
          const gd = document.getElementById('plot');
          return gd?._fullLayout?.dragmode === 'pan'
            && gd?._fullLayout?.xaxis?.fixedrange === false
            && gd?._fullLayout?.yaxis?.fixedrange === false;
        }"""
    )

    page.keyboard.press("Escape")
    dialog.wait_for(state="hidden")
    page.wait_for_function(
        """() => {
          const gd = document.getElementById('plot');
          return gd?._fullLayout?.dragmode === 'pan'
            && gd?._fullLayout?.xaxis?.fixedrange === false
            && gd?._fullLayout?.yaxis?.fixedrange === false;
        }"""
    )

    page.keyboard.up("Alt")
    page.wait_for_function(
        """() => {
          const gd = document.getElementById('plot');
          return gd?._fullLayout?.dragmode === 'zoom'
            && gd?._fullLayout?.xaxis?.fixedrange === false
            && gd?._fullLayout?.yaxis?.fixedrange === false;
        }"""
    )

    e2e_debug.assert_clean()
