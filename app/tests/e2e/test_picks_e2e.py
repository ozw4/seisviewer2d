import json
import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest


def _get_query_param(url: str, key: str) -> str:
    qs = parse_qs(urlparse(url).query)
    v = qs.get(key)
    assert v and len(v) == 1
    return v[0]


def _click_plot_center(page, *, modifiers: list[str] | None = None) -> None:
    x, y = page.evaluate(
        """() => {
          const env = getPlotEnv();
          if (!env) return null;
          const x = env.rect.left + env.m.l + env.m.w * 0.5;
          const y = env.rect.top  + env.m.t + env.m.h * 0.5;
          return [x, y];
        }"""
    )
    assert x is not None and y is not None
    pressed = modifiers or []
    for key in pressed:
        page.keyboard.down(key)
    try:
        page.mouse.click(x, y)
    finally:
        for key in reversed(pressed):
            page.keyboard.up(key)


@pytest.mark.e2e
def test_add_single_pick_then_get(page, base_url, tiny_segy_path, e2e_debug):
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

    page.select_option("#snap_mode", "none")
    page.select_option("#snap_refine", "none")

    pick_btn = page.locator("#pickModeBtn")
    pick_btn.click()
    page.wait_for_function(
        "() => document.getElementById('pickModeBtn')?.classList.contains('active') === true"
    )

    pick_posts: list[dict] = []

    def on_request(req):
        if req.method == "POST" and "/picks" in req.url:
            body = req.post_data
            if body:
                pick_posts.append(json.loads(body))

    page.on("request", on_request)

    with page.expect_response(
        lambda r: "/picks" in r.url and r.request.method == "POST" and r.status == 200,
        timeout=60_000,
    ):
        _click_plot_center(page)

    assert pick_posts, "POST /picks was not observed"
    posted = pick_posts[-1]

    file_id = _get_query_param(page.url, "file_id")
    assert posted["file_id"] == file_id

    params = {
        "file_id": posted["file_id"],
        "key1": posted["key1"],
        "key1_byte": posted["key1_byte"],
        "key2_byte": posted["key2_byte"],
    }

    deadline = time.time() + 5.0
    matched = False
    last = None
    while time.time() < deadline:
        r = httpx.get(f"{base_url}/picks", params=params, timeout=2.0, trust_env=False)
        assert r.status_code == 200
        j = r.json()
        last = j
        picks = j.get("picks", [])
        for p in picks:
            if (
                int(p["trace"]) == int(posted["trace"])
                and abs(float(p["time"]) - float(posted["time"])) < 1e-3
            ):
                matched = True
                break
        if matched:
            break
        time.sleep(0.1)

    assert matched, f"Pick not found via GET /picks. last={last}"

    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_pending_pick_anchors_are_visible_without_write(
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

    plot_btn = page.locator("button:has-text('Plot')")
    with page.expect_response(
        lambda r: "/get_section_window_bin" in r.url and r.status == 200,
        timeout=60_000,
    ):
        plot_btn.click()

    page.wait_for_selector("#plot.js-plotly-plot", timeout=60_000)
    page.wait_for_selector("#plot .plot-container", timeout=60_000)

    page.select_option("#snap_mode", "none")
    page.select_option("#snap_refine", "none")

    page.locator("#pickModeBtn").click()
    page.wait_for_function(
        "() => document.getElementById('pickModeBtn')?.classList.contains('active') === true"
    )

    pick_posts: list[dict] = []

    def on_request(req):
        if req.method == "POST" and "/picks" in req.url:
            body = req.post_data
            if body:
                pick_posts.append(json.loads(body))

    page.on("request", on_request)

    _click_plot_center(page, modifiers=["Shift"])
    page.wait_for_function(
        """() => {
          const text = document.getElementById('pendingPickStatus')?.textContent || '';
          return text.includes('Line pick anchor:');
        }"""
    )
    page.wait_for_function(
        """() => {
          const gd = document.getElementById('plot');
          const tr = Array.isArray(gd?.data)
            ? gd.data.find((item) => item?.meta?.svRole === 'pick' && item?.meta?.svKind === 'pending')
            : null;
          return !!(tr && tr.visible && Array.isArray(tr.x) ? tr.x.length === 1 : tr?.x?.length === 1);
        }"""
    )
    assert pick_posts == []

    page.keyboard.up("Shift")
    page.wait_for_function(
        "() => document.getElementById('pendingPickStatus')?.hidden === true"
    )

    _click_plot_center(page, modifiers=["Control"])
    page.wait_for_function(
        """() => {
          const text = document.getElementById('pendingPickStatus')?.textContent || '';
          return text.includes('Delete range anchor:');
        }"""
    )
    page.wait_for_function(
        """() => {
          const gd = document.getElementById('plot');
          const tr = Array.isArray(gd?.data)
            ? gd.data.find((item) => item?.meta?.svRole === 'pick' && item?.meta?.svKind === 'pending')
            : null;
          return !!(tr && tr.visible && tr.mode === 'lines' && tr.x?.length === 2 && tr.y?.length === 2);
        }"""
    )
    assert pick_posts == []

    e2e_debug.assert_clean()
