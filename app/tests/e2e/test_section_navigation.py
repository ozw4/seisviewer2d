import pytest


def _collect_section_window_requests(page) -> list[str]:
    requests: list[str] = []

    def on_request(req):
        if "/get_section_window_bin" in req.url:
            requests.append(f"{req.method} {req.url}")

    page.on("request", on_request)
    return requests


def _upload_tiny_dataset_and_wait(page, base_url, tiny_segy_path) -> None:
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
    page.wait_for_function(
        "() => document.querySelector('[data-testid=\"section-nav-position\"]')?.textContent?.trim() === 'Section 1 / 3'"
    )
    page.wait_for_function(
        "() => document.querySelector('[data-testid=\"section-nav-key1-current\"]')?.textContent?.trim() === 'key1: 100'"
    )
    page.wait_for_selector("#plot .plot-container", timeout=60_000)
    page.wait_for_timeout(1000)


@pytest.mark.e2e
def test_section_navigation_prev_next_plot_adjacent_sections(
    page, base_url, tiny_segy_path, e2e_debug
):
    requests = _collect_section_window_requests(page)
    _upload_tiny_dataset_and_wait(page, base_url, tiny_segy_path)
    requests.clear()

    position = page.get_by_test_id("section-nav-position")
    prev_btn = page.get_by_test_id("section-nav-prev")
    next_btn = page.get_by_test_id("section-nav-next")
    key1_current = page.get_by_test_id("section-nav-key1-current")

    assert position.text_content() == "Section 1 / 3"
    assert prev_btn.is_disabled()
    assert next_btn.is_enabled()

    before_next = len(requests)
    next_btn.click()

    page.wait_for_function(
        "() => document.querySelector('[data-testid=\"section-nav-key1-current\"]')?.textContent?.trim() === 'key1: 101'"
    )
    page.wait_for_timeout(800)
    assert position.text_content() == "Section 2 / 3"
    assert key1_current.text_content() == "key1: 101"
    assert len(requests) > before_next

    before_prev = len(requests)
    prev_btn.click()

    page.wait_for_function(
        "() => document.querySelector('[data-testid=\"section-nav-key1-current\"]')?.textContent?.trim() === 'key1: 100'"
    )
    page.wait_for_timeout(800)
    assert position.text_content() == "Section 1 / 3"
    assert key1_current.text_content() == "key1: 100"
    assert len(requests) >= before_prev
    assert page.locator("#plot .plot-container").is_visible()

    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_section_navigation_boundary_buttons_disable_without_plot(
    page, base_url, tiny_segy_path, e2e_debug
):
    requests = _collect_section_window_requests(page)
    _upload_tiny_dataset_and_wait(page, base_url, tiny_segy_path)
    requests.clear()

    prev_btn = page.get_by_test_id("section-nav-prev")
    next_btn = page.get_by_test_id("section-nav-next")

    assert prev_btn.is_disabled()

    page.evaluate("() => document.getElementById('sectionNavPrev').click()")
    page.wait_for_timeout(400)
    assert requests == []

    for _ in range(2):
        next_btn.click()
        page.wait_for_timeout(800)

    page.wait_for_function(
        "() => document.querySelector('[data-testid=\"section-nav-position\"]')?.textContent?.trim() === 'Section 3 / 3'"
    )
    assert next_btn.is_disabled()

    page.wait_for_timeout(800)
    request_count = len(requests)
    page.evaluate("() => document.getElementById('sectionNavNext').click()")
    page.wait_for_timeout(400)
    assert len(requests) == request_count

    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_section_navigation_direct_key1_jump_accepts_valid_value(
    page, base_url, tiny_segy_path, e2e_debug
):
    requests = _collect_section_window_requests(page)
    _upload_tiny_dataset_and_wait(page, base_url, tiny_segy_path)
    requests.clear()

    jump_input = page.get_by_test_id("section-nav-key1-input")
    go_btn = page.get_by_test_id("section-nav-key1-go")

    jump_input.fill("102")
    before_go = len(requests)
    go_btn.click()

    page.wait_for_function(
        "() => document.querySelector('[data-testid=\"section-nav-key1-current\"]')?.textContent?.trim() === 'key1: 102'"
    )
    page.wait_for_timeout(800)
    assert page.get_by_test_id("section-nav-position").text_content() == "Section 3 / 3"
    assert page.locator("#key1_val_display").input_value() == "102"
    assert page.locator("#key1_slider").input_value() == "2"
    assert len(requests) > before_go

    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_section_navigation_direct_key1_jump_rejects_invalid_value(
    page, base_url, tiny_segy_path, e2e_debug
):
    requests = _collect_section_window_requests(page)
    _upload_tiny_dataset_and_wait(page, base_url, tiny_segy_path)
    requests.clear()

    jump_input = page.get_by_test_id("section-nav-key1-input")
    go_btn = page.get_by_test_id("section-nav-key1-go")
    key1_current = page.get_by_test_id("section-nav-key1-current")

    assert key1_current.text_content() == "key1: 100"

    jump_input.fill("999")
    go_btn.click()

    page.get_by_test_id("section-nav-validation").wait_for(timeout=60_000)
    assert (
        page.get_by_test_id("section-nav-validation").text_content()
        == "key1 999 is not available in this dataset."
    )
    assert key1_current.text_content() == "key1: 100"
    assert page.get_by_test_id("section-nav-position").text_content() == "Section 1 / 3"

    page.wait_for_timeout(800)
    assert requests == []

    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_section_navigation_slider_auto_plots_with_debounce(
    page, base_url, tiny_segy_path, e2e_debug
):
    requests = _collect_section_window_requests(page)
    _upload_tiny_dataset_and_wait(page, base_url, tiny_segy_path)
    requests.clear()

    page.evaluate(
        """
        () => {
          const slider = document.getElementById('key1_slider');
          slider.value = '1';
          slider.dispatchEvent(new Event('input', { bubbles: true }));
          slider.value = '2';
          slider.dispatchEvent(new Event('input', { bubbles: true }));
        }
        """
    )

    page.wait_for_function(
        "() => document.querySelector('[data-testid=\"section-nav-key1-current\"]')?.textContent?.trim() === 'key1: 102'"
    )
    page.wait_for_timeout(1200)
    assert page.get_by_test_id("section-nav-position").text_content() == "Section 3 / 3"
    assert 1 <= len(requests) <= 3
    assert e2e_debug.unexpected_404() == []
    assert e2e_debug.page_error == []


@pytest.mark.e2e
def test_section_navigation_without_dataset_is_safe(page, base_url, e2e_debug):
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

    assert page.get_by_test_id("section-nav-prev").is_disabled()
    assert page.get_by_test_id("section-nav-next").is_disabled()
    assert page.get_by_test_id("section-nav-key1-input").is_disabled()
    assert page.get_by_test_id("section-nav-key1-go").is_disabled()
    assert page.get_by_test_id("section-nav-position").text_content() == "Section - / -"
    assert page.get_by_test_id("section-nav-key1-current").text_content() == "key1: -"

    page.wait_for_timeout(750)
    assert section_requests == []

    e2e_debug.assert_clean()
