import pytest


@pytest.mark.e2e
def test_upload_redirect_and_plot(page, base_url, tiny_segy_path, e2e_debug):
    page.set_default_timeout(60_000)

    page.goto(f"{base_url}/upload", wait_until="domcontentloaded")

    page.select_option("#key1_byte", "189")
    page.select_option("#key2_byte", "193")
    page.set_input_files("#upload_segy", str(tiny_segy_path))

    page.click("#upload_btn")
    page.wait_for_url("**/?file_id=*", timeout=60_000)
    page.wait_for_load_state("domcontentloaded")

    plot_btn = page.locator("button:has-text('Plot')")
    with page.expect_response(
        lambda r: "/get_section_window_bin" in r.url and r.status == 200,
        timeout=60_000,
    ):
        plot_btn.click()

    page.wait_for_selector("#plot.js-plotly-plot", timeout=60_000)
    page.wait_for_selector("#plot .plot-container", timeout=60_000)

    # (1) 404 の URL を特定して表示（想定内は除外）
    allow_404 = (
        "favicon.ico",
        ".map",
    )

    unexpected_404 = []
    for x in e2e_debug.not_found:
        if any(a in x for a in allow_404):
            continue
        # 初回は open_segy が 404 で正しい（キャッシュ無し）
        if x.startswith("404 POST") and "/open_segy" in x:
            continue
        unexpected_404.append(x)

    assert not unexpected_404, "Unexpected 404 responses:\n" + "\n".join(unexpected_404)

    # JS 例外は落とす
    assert not e2e_debug.page_error, "Page errors:\n" + "\n".join(e2e_debug.page_error)

    # request_failed のうち想定内を除外して落とす
    unexpected_req_failed = []
    for x in e2e_debug.request_failed:
        # upload フローで遷移が起きると /open_segy が abort 扱いになることがある
        if "/open_segy" in x and "net::ERR_ABORTED" in x:
            continue
        # 環境によっては favicon が abort になることがある（必要なら）
        if "favicon.ico" in x and "ERR_ABORTED" in x:
            continue
        unexpected_req_failed.append(x)

    assert not unexpected_req_failed, "Request failed:\n" + "\n".join(
        unexpected_req_failed
    )
