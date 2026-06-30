import shutil
from urllib.parse import parse_qs, urlparse

import pytest


def _upload_compare_dataset(page, base_url, segy_path):
    page.goto(f"{base_url}/upload", wait_until="domcontentloaded")
    page.select_option("#key1_byte", "189")
    page.select_option("#key2_byte", "193")
    page.set_input_files("#upload_segy", str(segy_path))
    page.click("#analyzeHeadersBtn")
    page.wait_for_function("() => !document.getElementById('upload_btn')?.disabled")
    page.click("#upload_btn")
    page.wait_for_url("**/?file_id=*", timeout=60_000)
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_function(
        "() => document.getElementById('viewerEmptyState')?.hidden === true"
    )
    page.wait_for_load_state("networkidle")
    params = parse_qs(urlparse(page.url).query)
    return params["file_id"][0]


def _raw_compare_options(page, select_id):
    return page.evaluate(
        """(selectId) => Array.from(
            document.querySelectorAll(`#${selectId} option`)
        )
            .filter((option) => option.textContent.trim().endsWith(' / raw'))
            .map((option) => ({
                value: option.value,
                text: option.textContent.trim(),
            }))""",
        arg=select_id,
    )


def _file_id_from_compare_source_value(value):
    parts = str(value).split(":")
    assert len(parts) >= 3
    assert parts[0] == "file"
    assert parts[-1] == "raw"
    return parts[1]


@pytest.mark.e2e
def test_playwright_compare_mode_toggle_and_diff_panel(
    page, base_url, tiny_segy_path, e2e_debug
):
    page.set_default_timeout(60_000)

    page.goto(f"{base_url}/upload", wait_until="domcontentloaded")
    page.select_option("#key1_byte", "189")
    page.select_option("#key2_byte", "193")
    page.set_input_files("#upload_segy", str(tiny_segy_path))
    page.click("#analyzeHeadersBtn")
    page.wait_for_function("() => !document.getElementById('upload_btn')?.disabled")
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
    assert page.locator("#compareSourceA option:checked").text_content().endswith(
        " / raw"
    )
    assert page.locator("#compareSourceB option:checked").text_content().endswith(
        " / raw"
    )

    page.uncheck("#compareShowDiff")
    page.wait_for_function(
        "() => document.getElementById('plot')?.__svComparePanelCount === 2"
    )
    assert page.locator("#compareSourceA option:checked").text_content().endswith(
        " / raw"
    )
    assert page.locator("#compareSourceB option:checked").text_content().endswith(
        " / raw"
    )

    page.check("#compareShowDiff")
    page.wait_for_function(
        "() => document.getElementById('plot')?.__svComparePanelCount === 3"
    )

    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_playwright_compare_two_raw_sgy_files_with_a_based_normalization(
    page, base_url, tiny_segy_path, tmp_path, e2e_debug
):
    page.set_default_timeout(60_000)
    dataset_a = tmp_path / "compare_a.sgy"
    dataset_b = tmp_path / "compare_b.sgy"
    shutil.copyfile(tiny_segy_path, dataset_a)
    shutil.copyfile(tiny_segy_path, dataset_b)

    uploaded_a_file_id = _upload_compare_dataset(page, base_url, dataset_a)
    uploaded_b_file_id = _upload_compare_dataset(page, base_url, dataset_b)
    assert uploaded_a_file_id != uploaded_b_file_id

    page.wait_for_function(
        """(name) => Array.from(
            document.querySelectorAll('#compareDatasetPicker option')
        ).some((option) => (
            option.textContent.trim() === name && !option.disabled
        ))""",
        arg=dataset_a.name,
    )
    page.select_option("#compareDatasetPicker", label=dataset_a.name)
    page.click("#compareAddDataset")
    page.wait_for_function("() => window.compareFileTargets?.length === 2")
    page.wait_for_function(
        """() => Array.from(
            document.querySelectorAll('#compareSourceA option')
        ).filter((option) => (
            option.textContent.trim().endsWith(' / raw')
        )).length >= 2"""
    )

    raw_options = _raw_compare_options(page, "compareSourceA")
    assert len(raw_options) >= 2
    assert _raw_compare_options(page, "compareSourceB") == raw_options

    page.wait_for_load_state("networkidle")
    page.evaluate(
        """([sourceA, sourceB]) => {
            document.getElementById('compareSourceA').value = sourceA;
            document.getElementById('compareSourceB').value = sourceB;
        }""",
        arg=[raw_options[0]["value"], raw_options[1]["value"]],
    )

    selected = page.evaluate(
        """() => ({
            a: document.getElementById('compareSourceA').value,
            b: document.getElementById('compareSourceB').value,
            aText: document.querySelector('#compareSourceA option:checked')
                ?.textContent.trim(),
            bText: document.querySelector('#compareSourceB option:checked')
                ?.textContent.trim(),
        })"""
    )
    assert selected["a"] != selected["b"]
    assert selected["aText"].endswith(" / raw")
    assert selected["bText"].endswith(" / raw")
    source_a_file_id = _file_id_from_compare_source_value(selected["a"])
    source_b_file_id = _file_id_from_compare_source_value(selected["b"])
    assert uploaded_b_file_id in {source_a_file_id, source_b_file_id}

    events = []

    def on_request(req):
        if "/compare/raw/validate" in req.url:
            events.append(("validate_request", req.url))
        elif "/get_section_window_bin" in req.url:
            events.append(("window_request", req.url))

    def on_response(resp):
        if "/compare/raw/validate" in resp.url:
            events.append((f"validate_response_{resp.status}", resp.url))

    page.on("request", on_request)
    page.on("response", on_response)

    with page.expect_response(
        lambda r: "/compare/raw/validate" in r.url and r.status == 200,
        timeout=60_000,
    ):
        page.check("#compareModeToggle")

    page.wait_for_function(
        "() => document.getElementById('plot')?.__svComparePanelCount === 3"
    )
    page.wait_for_function(
        """() => {
            const annotations = document.getElementById('plot')
                ?._fullLayout?.annotations || [];
            const labels = annotations.map((annotation) => annotation.text || '');
            return labels.some((text) => text.startsWith('A:'))
                && labels.some((text) => text.startsWith('B:'))
                && labels.some((text) => text.startsWith('A-B:'));
        }"""
    )

    status = page.evaluate(
        """() => {
            const el = document.getElementById('compareStatus');
            return { hidden: el.hidden, text: el.textContent.trim() };
        }"""
    )
    assert status["hidden"] or status["text"] == ""

    validate_response_index = next(
        i for i, (kind, _) in enumerate(events) if kind == "validate_response_200"
    )
    compare_window_urls = [
        url
        for kind, url in events[validate_response_index + 1 :]
        if kind == "window_request"
        and parse_qs(urlparse(url).query).get("normalization_file_id") == [
            source_a_file_id
        ]
    ]
    assert compare_window_urls
    params_by_file_id = {
        parse_qs(urlparse(url).query)["file_id"][0]: parse_qs(urlparse(url).query)
        for url in compare_window_urls
    }
    assert {source_a_file_id, source_b_file_id}.issubset(params_by_file_id)
    assert params_by_file_id[source_a_file_id]["normalization_file_id"] == [
        source_a_file_id
    ]
    assert params_by_file_id[source_b_file_id]["normalization_file_id"] == [
        source_a_file_id
    ]

    e2e_debug.assert_clean()
