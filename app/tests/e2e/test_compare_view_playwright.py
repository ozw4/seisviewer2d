import time
from urllib.parse import parse_qs, urlparse

import pytest


def _wait_for(condition, message: str, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {message}.")


def _install_compare_window_stubs(page, *, tap_dt: float = 0.002) -> None:
    page.evaluate(
        """
        ({ tapDt }) => {
          const slider = document.getElementById('key1_slider');
          if (slider) {
            slider.value = '0';
            slider.max = '0';
          }

          const attachPlotHooks = (div) => {
            if (!div) return;
            div.on = () => div;
            div.data = [];
            div._fullLayout = {
              xaxis: { range: [0, 1] },
              yaxis: { range: [1, 0] },
            };
          };

          ['comparePlotA', 'comparePlotB', 'comparePlotDiff', 'compareRmsPlot'].forEach((id) => {
            attachPlotHooks(document.getElementById(id));
          });

          window.currentFileId = 'compare-test-file';
          window.sectionShape = [4, 4];
          window.key1Values = [100];
          window.currentKey1Byte = 189;
          window.currentKey2Byte = 193;
          window.currentScaling = 'amax';
          window.defaultDt = 0.002;
          window.savedXRange = null;
          window.savedYRange = null;

          window.currentVisibleWindow = () => ({
            x0: 0,
            x1: 1,
            y0: 0,
            y1: 1,
            nTraces: 2,
            nSamples: 2,
          });
          window.computeStepsForWindow = () => ({ step_x: 1, step_y: 1 });
          window.readWindowDecodeUseWorker = () => false;
          window.__compareApplyDt = null;
          window.decodeWindowPayload = (_bin, payloadMeta, _perfMeta, _onInvalidShape, options = {}) => {
            window.__compareApplyDt = options?.applyDt ?? null;
            return {
              ...payloadMeta,
              shape: [2, 2],
              dt: payloadMeta.effectiveLayer && payloadMeta.effectiveLayer !== 'raw' ? tapDt : 0.002,
              zBacking: new Float32Array([1, 2, 3, 4]),
            };
          };
          window.windowCacheGet = () => null;
          window.windowCacheSet = (_key, payload) => payload;
          window.showLoading = () => {};
          window.hideLoading = () => {};
          window.Plotly = {
            react: async (div, data) => {
              attachPlotHooks(div);
              div.data = Array.isArray(data) ? data : [];
              return div;
            },
            purge: (div) => {
              if (!div) return;
              attachPlotHooks(div);
              div.data = [];
            },
            relayout: async (div) => div,
            Plots: {
              resize: () => {},
            },
          };
        }
        """,
        {"tapDt": tap_dt},
    )


def _install_compare_worker_window_stubs(page, *, tap_dt: float = 0.002) -> None:
    _install_compare_window_stubs(page, tap_dt=tap_dt)
    page.evaluate(
        """
        ({ tapDt }) => {
          window.readWindowDecodeUseWorker = () => true;
          window.__compareWorkerJobs = 0;
          window.enqueueDecodeJob = () => {
            window.__compareWorkerJobs += 1;
            return {
              jobId: window.__compareWorkerJobs,
              promise: Promise.resolve({
                ok: true,
                rows: 2,
                cols: 2,
                dt: tapDt,
                zBuf: new Float32Array([1, 2, 3, 4]).buffer,
              }),
            };
          };
          window.cancelDecodeJob = () => {};
          window.buildWindowPayloadFromWorkerDecoded = (decoded, payloadMeta) => ({
            ...payloadMeta,
            shape: [2, 2],
            dt: payloadMeta.effectiveLayer && payloadMeta.effectiveLayer !== 'raw' ? tapDt : 0.002,
            zBacking: new Float32Array([1, 2, 3, 4]),
            zRows: [
              new Float32Array([1, 2]),
              new Float32Array([3, 4]),
            ],
          });
        }
        """,
        {"tapDt": tap_dt},
    )


def _append_pipeline_layer_option(page, label: str = "denoise") -> None:
    page.evaluate(
        """
        ({ label }) => {
          window.latestPipelineKey = 'pipe-1';
          const select = document.getElementById('layerSelect');
          if (!select) return;
          const hasOption = Array.from(select.options).some((option) => option.value === label);
          if (!hasOption) {
            select.appendChild(new Option(label, label));
          }
        }
        """,
        {"label": label},
    )


@pytest.mark.e2e
def test_compare_view_playwright_refetches_when_auto_source_b_changes(
    page, base_url, e2e_debug
):
    page.set_default_timeout(60_000)
    page.goto(f"{base_url}/", wait_until="domcontentloaded")

    requests: list[str] = []

    def handle_window(route, request):
        requests.append(request.url)
        route.fulfill(
            status=200,
            headers={"content-type": "application/octet-stream"},
            body=b"x",
        )

    page.route("**/get_section_window_bin?*", handle_window)
    _install_compare_window_stubs(page)

    page.select_option("#compareModeSelect", "side_by_side")
    _wait_for(lambda: len(requests) >= 1, "the initial compare fetch")
    initial_request_count = len(requests)

    _append_pipeline_layer_option(page)
    page.wait_for_function(
        "() => document.getElementById('compareSourceBSelect')?.value === 'pipeline_tap:pipe-1:denoise'"
    )
    _wait_for(
        lambda: len(requests) >= initial_request_count + 2,
        "the compare refetch after source B auto-switches",
    )

    queries = [parse_qs(urlparse(url).query) for url in requests]
    assert any(
        query.get("pipeline_key") == ["pipe-1"]
        and query.get("tap_label") == ["denoise"]
        for query in queries
    )
    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_compare_view_playwright_preserves_server_dt_in_non_worker_decode(
    page, base_url, e2e_debug
):
    page.set_default_timeout(60_000)
    page.goto(f"{base_url}/", wait_until="domcontentloaded")

    def handle_window(route, _request):
        route.fulfill(
            status=200,
            headers={"content-type": "application/octet-stream"},
            body=b"x",
        )

    page.route("**/get_section_window_bin?*", handle_window)
    _install_compare_window_stubs(page, tap_dt=0.004)
    _append_pipeline_layer_option(page)
    page.evaluate("() => { window.savedYRange = [5, 1]; }")

    page.wait_for_function(
        "() => document.getElementById('compareSourceBSelect')?.value === 'pipeline_tap:pipe-1:denoise'"
    )
    page.select_option("#compareModeSelect", "difference")
    page.wait_for_function(
        "() => (document.getElementById('compareStatus')?.textContent || '').includes('sample intervals differ')"
    )

    assert (
        page.locator("#compareStatus").text_content()
        == "Cannot compare sources: decoded window sample intervals differ."
    )
    assert page.evaluate("() => window.__compareApplyDt") is False
    assert page.evaluate("() => window.defaultDt") == pytest.approx(0.002)
    assert page.evaluate("() => window.savedYRange") == [5, 1]
    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_compare_view_playwright_preserves_server_dt_in_worker_decode(
    page, base_url, e2e_debug
):
    page.set_default_timeout(60_000)
    page.goto(f"{base_url}/", wait_until="domcontentloaded")

    def handle_window(route, _request):
        route.fulfill(
            status=200,
            headers={"content-type": "application/octet-stream"},
            body=b"x",
        )

    page.route("**/get_section_window_bin?*", handle_window)
    _install_compare_worker_window_stubs(page, tap_dt=0.004)
    _append_pipeline_layer_option(page)
    page.evaluate("() => { window.savedYRange = [7, 2]; }")

    page.wait_for_function(
        "() => document.getElementById('compareSourceBSelect')?.value === 'pipeline_tap:pipe-1:denoise'"
    )
    page.select_option("#compareModeSelect", "difference")
    page.wait_for_function(
        "() => (document.getElementById('compareStatus')?.textContent || '').includes('sample intervals differ')"
    )

    assert (
        page.locator("#compareStatus").text_content()
        == "Cannot compare sources: decoded window sample intervals differ."
    )
    assert page.evaluate("() => window.defaultDt") == pytest.approx(0.002)
    assert page.evaluate("() => window.savedYRange") == [7, 2]
    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_compare_view_playwright_uses_main_heatmap_decode_values(
    page, base_url, e2e_debug
):
    page.set_default_timeout(60_000)
    page.goto(f"{base_url}/", wait_until="domcontentloaded")

    def handle_window(route, _request):
        route.fulfill(
            status=200,
            headers={"content-type": "application/octet-stream"},
            body=b"x",
        )

    page.route("**/get_section_window_bin?*", handle_window)
    _install_compare_window_stubs(page)
    page.evaluate(
        """
        () => {
          window.SeisHeatmap = {
            getQuantLUT: () => {
              const lut = new Float32Array(256);
              lut[128] = 10;
              lut[129] = 20;
              lut[130] = 30;
              lut[131] = 40;
              return lut;
            },
          };
          window.decodeWindowPayload = (_bin, payloadMeta, _perfMeta, _onInvalidShape, options = {}) => {
            window.__compareApplyDt = options?.applyDt ?? null;
            return {
              ...payloadMeta,
              shape: [2, 2],
              dt: 0.002,
              scale: 2,
              quant: { mode: 'linear', lo: -1, hi: 1, mu: 255 },
              valuesI8: new Int8Array([0, 1, 2, 3]),
            };
          };
        }
        """
    )

    page.select_option("#compareModeSelect", "side_by_side")
    page.wait_for_function(
        "() => Array.isArray(document.getElementById('comparePlotA')?.data?.[0]?.z)"
    )

    z_values = page.evaluate(
        """
        () => Array.from(
          document.getElementById('comparePlotA').data[0].z,
          (row) => Array.from(row),
        )
        """
    )
    assert z_values == [[10.0, 20.0], [30.0, 40.0]]
    assert page.evaluate("() => window.__compareApplyDt") is False
    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_compare_view_playwright_scales_probability_layers_to_display_range(
    page, base_url, e2e_debug
):
    page.set_default_timeout(60_000)
    page.goto(f"{base_url}/", wait_until="domcontentloaded")

    def handle_window(route, _request):
        route.fulfill(
            status=200,
            headers={"content-type": "application/octet-stream"},
            body=b"x",
        )

    page.route("**/get_section_window_bin?*", handle_window)
    _install_compare_window_stubs(page)
    _append_pipeline_layer_option(page, label="fbprob")
    page.wait_for_function(
        "() => document.getElementById('compareSourceBSelect')?.value === 'pipeline_tap:pipe-1:fbprob'"
    )
    page.evaluate(
        """
        () => {
          window.SeisHeatmap = {};
          window.decodeWindowPayload = (_bin, payloadMeta, _perfMeta, _onInvalidShape, options = {}) => {
            window.__compareApplyDt = options?.applyDt ?? null;
            return {
              ...payloadMeta,
              shape: [2, 2],
              dt: 0.002,
              scale: 4,
              quant: { scale: 4 },
              valuesI8: new Int8Array([0, 1, 4, 2]),
            };
          };
        }
        """
    )

    page.select_option("#compareModeSelect", "side_by_side")
    page.select_option("#compareSourceASelect", "pipeline_tap:pipe-1:fbprob")
    page.wait_for_function(
        "() => Array.isArray(document.getElementById('comparePlotB')?.data?.[0]?.z)"
    )

    rendered = page.evaluate(
        """
        () => {
          const trace = document.getElementById('comparePlotB').data[0];
          return {
            z: Array.from(trace.z, (row) => Array.from(row)),
            zmin: trace.zmin,
            zmax: trace.zmax,
          };
        }
        """
    )
    assert rendered["z"][0] == pytest.approx([0.0, 63.75])
    assert rendered["z"][1] == pytest.approx([255.0, 127.5])
    assert rendered["zmin"] == pytest.approx(0)
    assert rendered["zmax"] == pytest.approx(255)
    e2e_debug.assert_clean()
