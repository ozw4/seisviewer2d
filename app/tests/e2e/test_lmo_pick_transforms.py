import pytest


@pytest.mark.e2e
def test_playwright_lmo_pick_helpers_apply_display_raw_transforms(
    page, base_url, e2e_debug
):
    page.goto(f"{base_url}/", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => typeof window.rawTimeToDisplayTime === 'function' "
        "&& typeof window.displayTimeToRawTime === 'function' "
        "&& typeof window.buildPickMarkerTraces === 'function' "
        "&& typeof window.msgpack?.encode === 'function'"
    )

    result = page.evaluate(
        """async () => {
            const originalFetch = window.fetch.bind(window);
            let offsetFetches = 0;

            window.fetch = (input, init) => {
                const url = String(input instanceof Request ? input.url : input);
                if (url.includes('/get_section_offsets_bin')) {
                    offsetFetches += 1;
                    const offsets = new Float32Array([10, 20, 40]);
                    const payload = window.msgpack.encode({
                        file_id: 'pick-lmo-file',
                        key1: 7,
                        key1_byte: 189,
                        key2_byte: 193,
                        offset_byte: 41,
                        dtype: 'float32',
                        shape: [offsets.length],
                        offsets: new Uint8Array(offsets.buffer),
                    });
                    return Promise.resolve(new Response(payload, { status: 200 }));
                }
                return originalFetch(input, init);
            };

            try {
                window.setCurrentLinearMoveout({
                    enabled: true,
                    velocityMps: 10,
                    offsetByte: 41,
                    offsetScale: 1,
                    offsetMode: 'signed',
                    refMode: 'zero',
                    refTrace: 0,
                    polarity: 1,
                });
                window.currentFileId = 'pick-lmo-file';
                window.currentKey1Byte = 189;
                window.currentKey2Byte = 193;
                window.key1Values = [7];
                document.getElementById('key1_slider').value = '0';

                const beforeReady = window.rawTimeToDisplayTime(1, 7);
                let display = NaN;
                for (let i = 0; i < 50; i += 1) {
                    await new Promise((resolve) => setTimeout(resolve, 10));
                    display = window.rawTimeToDisplayTime(1, 7);
                    if (Number.isFinite(display)) break;
                }

                const traces = window.buildPickMarkerTraces({
                    manualPicks: [{ trace: 1, time: 7 }],
                    predicted: [{ trace: 2, time: 9 }],
                    xMin: 0,
                    xMax: 2,
                    showPredicted: true,
                    timeTransform: window.pickRawTimeToDisplayTime,
                });
                const rawFromDisplay = window.displayTimeToRawTime(2, 5);

                window.currentFileId = '';
                window.key1Values = [];
                window.setCurrentLinearMoveout({ enabled: false });
                const offDisplay = window.rawTimeToDisplayTime(2, 9);
                const offRaw = window.displayTimeToRawTime(2, 5);

                return {
                    beforeReadyIsNaN: Number.isNaN(beforeReady),
                    display,
                    rawFromDisplay,
                    manualY: Array.from(traces[0].y),
                    predictedY: Array.from(traces[1].y),
                    offsetFetches,
                    offDisplay,
                    offRaw,
                };
            } finally {
                window.fetch = originalFetch;
            }
        }"""
    )

    assert result == {
        "beforeReadyIsNaN": True,
        "display": 5,
        "rawFromDisplay": 9,
        "manualY": [5],
        "predictedY": [5],
        "offsetFetches": 1,
        "offDisplay": 9,
        "offRaw": 5,
    }
    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_playwright_lmo_failed_single_pick_conversion_keeps_existing_raw_pick(
    page, base_url, e2e_debug
):
    page.goto(f"{base_url}/", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => typeof handlePickNormalized === 'function' "
        "&& typeof window.debugDump === 'function' "
        "&& typeof window.setCurrentLinearMoveout === 'function'"
    )

    result = page.evaluate(
        """async () => {
            const originalFetch = window.fetch.bind(window);
            const calls = [];

            window.fetch = (input, init = {}) => {
                const url = String(input instanceof Request ? input.url : input);
                const method = String(init.method || (input instanceof Request ? input.method : 'GET')).toUpperCase();
                calls.push({ url, method });
                if (url.includes('/get_section_offsets_bin')) {
                    return Promise.resolve(new Response('{}', { status: 409 }));
                }
                if (url.includes('/get_section_meta')) {
                    return Promise.resolve(new Response(JSON.stringify({
                        shape: [3, 3],
                        dt: 1,
                    }), {
                        status: 200,
                        headers: { 'Content-Type': 'application/json' },
                    }));
                }
                if (url.includes('/get_section_window_bin')) {
                    return Promise.resolve(new Response('{}', { status: 409 }));
                }
                if (url.includes('/picks')) {
                    return Promise.resolve(new Response('{}', {
                        status: 200,
                        headers: { 'Content-Type': 'application/json' },
                    }));
                }
                return originalFetch(input, init);
            };

            try {
                currentFileId = 'pick-lmo-missing-offsets-file';
                currentKey1Byte = 189;
                currentKey2Byte = 193;
                key1Values = [7];
                document.getElementById('key1_slider').value = '0';
                document.getElementById('snap_mode').value = 'none';
                document.getElementById('snap_refine').value = 'none';
                picks = [{ trace: 1, time: 11 }];
                isPickMode = true;

                window.setCurrentLinearMoveout({
                    enabled: true,
                    velocityMps: 10,
                    offsetByte: 41,
                    offsetScale: 1,
                    offsetMode: 'signed',
                    refMode: 'zero',
                    refTrace: 0,
                    polarity: 1,
                });

                await handlePickNormalized({
                    trace: 1,
                    time: 5,
                    shiftKey: false,
                    ctrlKey: false,
                    altKey: false,
                });
                await new Promise((resolve) => setTimeout(resolve, 180));

                return {
                    picks: window.debugDump().picks,
                    pickCalls: calls.filter((call) => call.url.includes('/picks')),
                    offsetFetches: calls.filter((call) => call.url.includes('/get_section_offsets_bin')).length,
                };
            } finally {
                isPickMode = false;
                window.setCurrentLinearMoveout({ enabled: false });
                window.fetch = originalFetch;
            }
        }"""
    )

    assert result["picks"] == [{"tr": 1, "t": 11}]
    assert result["pickCalls"] == []
    assert result["offsetFetches"] >= 1
    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_playwright_lmo_failed_delete_hit_test_keeps_existing_raw_pick(
    page, base_url, e2e_debug
):
    page.goto(f"{base_url}/", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => typeof handlePickNormalized === 'function' "
        "&& typeof window.debugDump === 'function' "
        "&& typeof window.setCurrentLinearMoveout === 'function'"
    )

    result = page.evaluate(
        """async () => {
            const originalFetch = window.fetch.bind(window);
            const calls = [];

            window.fetch = (input, init = {}) => {
                const url = String(input instanceof Request ? input.url : input);
                const method = String(init.method || (input instanceof Request ? input.method : 'GET')).toUpperCase();
                calls.push({ url, method });
                if (url.includes('/get_section_offsets_bin')) {
                    return Promise.resolve(new Response('{}', { status: 409 }));
                }
                if (url.includes('/get_section_meta')) {
                    return Promise.resolve(new Response(JSON.stringify({
                        shape: [3, 3],
                        dt: 1,
                    }), {
                        status: 200,
                        headers: { 'Content-Type': 'application/json' },
                    }));
                }
                if (url.includes('/get_section_window_bin')) {
                    return Promise.resolve(new Response('{}', { status: 409 }));
                }
                if (url.includes('/picks')) {
                    return Promise.resolve(new Response('{}', {
                        status: 200,
                        headers: { 'Content-Type': 'application/json' },
                    }));
                }
                return originalFetch(input, init);
            };

            try {
                currentFileId = 'pick-lmo-missing-delete-offsets-file';
                currentKey1Byte = 189;
                currentKey2Byte = 193;
                key1Values = [7];
                document.getElementById('key1_slider').value = '0';
                document.getElementById('snap_mode').value = 'none';
                document.getElementById('snap_refine').value = 'none';
                picks = [{ trace: 1, time: 11 }];
                isPickMode = true;

                window.setCurrentLinearMoveout({
                    enabled: true,
                    velocityMps: 10,
                    offsetByte: 41,
                    offsetScale: 1,
                    offsetMode: 'signed',
                    refMode: 'zero',
                    refTrace: 0,
                    polarity: 1,
                });

                await handlePickNormalized({
                    trace: 1,
                    time: 5,
                    shiftKey: false,
                    ctrlKey: true,
                    altKey: false,
                });
                await handlePickNormalized({
                    trace: 1,
                    time: 5,
                    shiftKey: false,
                    ctrlKey: true,
                    altKey: false,
                });
                await new Promise((resolve) => setTimeout(resolve, 180));

                return {
                    picks: window.debugDump().picks,
                    pickCalls: calls.filter((call) => call.url.includes('/picks')),
                    offsetFetches: calls.filter((call) => call.url.includes('/get_section_offsets_bin')).length,
                };
            } finally {
                isPickMode = false;
                window.setCurrentLinearMoveout({ enabled: false });
                window.fetch = originalFetch;
            }
        }"""
    )

    assert result["picks"] == [{"tr": 1, "t": 11}]
    assert result["pickCalls"] == []
    assert result["offsetFetches"] >= 1
    e2e_debug.assert_clean()
