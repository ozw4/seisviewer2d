from urllib.parse import parse_qs, urlparse


def _query(url: str) -> dict[str, list[str]]:
    return parse_qs(urlparse(url).query)


def _assert_no_target_blank(page, selector: str) -> None:
    link = page.locator(selector)
    assert link.get_attribute("target") is None
    assert link.get_attribute("rel") is None


def test_playwright_tool_navigation_links_do_not_force_new_tabs(page, base_url):
    page.goto(
        f"{base_url}/static-correction?file_id=line-a&key1_byte=9&key2_byte=13",
        wait_until="domcontentloaded",
    )
    _assert_no_target_blank(page, "header a:has-text('Viewer')")
    _assert_no_target_blank(page, "header a:has-text('Refraction QC')")
    _assert_no_target_blank(page, "[data-testid='static-correction-open-qc-link']")

    page.goto(
        f"{base_url}/refraction-qc?file_id=line-a&key1_byte=9&key2_byte=13",
        wait_until="domcontentloaded",
    )
    _assert_no_target_blank(page, "header a:has-text('Viewer')")
    _assert_no_target_blank(page, "header a:has-text('Static Correction')")

    page.goto(f"{base_url}/?file_id=line-a&key1_byte=9&key2_byte=13", wait_until="domcontentloaded")
    _assert_no_target_blank(page, "#staticCorrectionLink")
    _assert_no_target_blank(page, "#refractionQcLink")
    _assert_no_target_blank(page, "#batchApplyLink")


def test_playwright_static_correction_to_refraction_qc_uses_same_tab(page, base_url):
    page.goto(
        f"{base_url}/static-correction?file_id=line-a&key1_byte=9&key2_byte=13",
        wait_until="domcontentloaded",
    )
    before_pages = len(page.context.pages)

    page.get_by_role("link", name="Refraction QC").click()
    page.wait_for_url("**/refraction-qc?**")

    assert len(page.context.pages) == before_pages
    query = _query(page.url)
    assert query["file_id"] == ["line-a"]
    assert query["key1_byte"] == ["9"]
    assert query["key2_byte"] == ["13"]


def test_playwright_refraction_qc_to_static_correction_uses_same_tab(page, base_url):
    page.goto(
        f"{base_url}/refraction-qc?file_id=line-b&key1_byte=17&key2_byte=21",
        wait_until="domcontentloaded",
    )
    before_pages = len(page.context.pages)

    page.get_by_role("link", name="Static Correction").click()
    page.wait_for_url("**/static-correction?**")

    assert len(page.context.pages) == before_pages
    query = _query(page.url)
    assert query["file_id"] == ["line-b"]
    assert query["key1_byte"] == ["17"]
    assert query["key2_byte"] == ["21"]


def test_playwright_viewer_batch_apply_viewer_roundtrip_uses_same_tab(page, base_url):
    page.goto(f"{base_url}/?file_id=line-c&key1_byte=189&key2_byte=193", wait_until="domcontentloaded")
    before_pages = len(page.context.pages)

    page.locator("#batchApplyLink").click()
    page.wait_for_url("**/batch?**")

    assert len(page.context.pages) == before_pages
    batch_query = _query(page.url)
    assert batch_query["file_id"] == ["line-c"]
    assert batch_query["key1_byte"] == ["189"]
    assert batch_query["key2_byte"] == ["193"]

    page.locator("#fileIdInput").fill("line-c-edited")
    page.locator("#key1ByteInput").fill("101")
    page.locator("#key2ByteInput").fill("202")
    page.get_by_role("link", name="Viewer").click()
    page.wait_for_url("**/?**")

    assert len(page.context.pages) == before_pages
    viewer_query = _query(page.url)
    assert viewer_query["file_id"] == ["line-c-edited"]
    assert viewer_query["key1_byte"] == ["101"]
    assert viewer_query["key2_byte"] == ["202"]
