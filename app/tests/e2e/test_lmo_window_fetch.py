import pytest


@pytest.mark.e2e
def test_playwright_lmo_window_request_artifacts_are_canonical(
    page, base_url, e2e_debug
):
    page.goto(f"{base_url}/", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => typeof window.buildWindowRequestArtifacts === 'function' "
        "&& typeof window.setCurrentLinearMoveout === 'function'"
    )

    result = page.evaluate(
        """() => {
            const context = {
                fileId: 'demo-file',
                key1Val: 100,
                key1Byte: 189,
                key2Byte: 193,
                windowInfo: { x0: 0, x1: 3, y0: 0, y1: 9 },
                stepX: 1,
                stepY: 2,
                requestedLayer: 'raw',
                effectiveLayer: 'raw',
                pipelineKey: null,
                tapLabel: null,
                scaling: 'amax',
                transpose: '1',
                mode: 'heatmap',
            };

            function snapshot() {
                const artifacts = window.buildWindowRequestArtifacts(context);
                return {
                    params: Object.fromEntries(artifacts.params.entries()),
                    cacheKey: artifacts.cacheKey,
                    payloadLmoKey: artifacts.payloadMeta.lmoKey,
                    requestLmoKey: artifacts.requestContext.lmoKey,
                };
            }

            window.setCurrentLinearMoveout({
                enabled: false,
                velocityMps: 2500,
                offsetByte: 41,
                offsetScale: 2,
                offsetMode: 'signed',
                refMode: 'trace',
                refTrace: 12,
                polarity: -1,
            });
            const disabledA = snapshot();

            window.setCurrentLinearMoveout({
                enabled: false,
                velocityMps: 3500,
                offsetByte: 45,
                offsetScale: 3,
                offsetMode: 'absolute',
                refMode: 'zero',
                refTrace: 2,
                polarity: 1,
            });
            const disabledB = snapshot();

            window.setCurrentLinearMoveout({
                enabled: true,
                velocityMps: 2000,
                offsetByte: 41,
                offsetScale: 2.5,
                offsetMode: 'signed',
                refMode: 'trace',
                refTrace: 12,
                polarity: -1,
            });
            const enabled = snapshot();

            return { disabledA, disabledB, enabled };
        }"""
    )

    lmo_params = [
        "lmo_velocity_mps",
        "lmo_offset_byte",
        "lmo_offset_scale",
        "lmo_offset_mode",
        "lmo_ref_mode",
        "lmo_ref_trace",
        "lmo_polarity",
    ]

    assert result["disabledA"]["params"]["lmo_enabled"] == "false"
    assert result["disabledA"]["payloadLmoKey"] == "lmo:off"
    assert result["disabledA"]["requestLmoKey"] == "lmo:off"
    assert "lmo=lmo:off" in result["disabledA"]["cacheKey"]
    for param in lmo_params:
        assert param not in result["disabledA"]["params"]

    assert result["disabledB"]["params"]["lmo_enabled"] == "false"
    assert result["disabledB"]["cacheKey"] == result["disabledA"]["cacheKey"]

    enabled_params = result["enabled"]["params"]
    for param, value in {
        "lmo_enabled": "true",
        "lmo_velocity_mps": "2000",
        "lmo_offset_byte": "41",
        "lmo_offset_scale": "2.5",
        "lmo_offset_mode": "signed",
        "lmo_ref_mode": "trace",
        "lmo_ref_trace": "12",
        "lmo_polarity": "-1",
    }.items():
        assert enabled_params[param] == value
    assert (
        "lmo=lmo:on|v=2000|ob=41|os=2.5|om=signed|rm=trace|rt=12|p=-1"
        in result["enabled"]["cacheKey"]
    )
    assert result["enabled"]["payloadLmoKey"] == (
        "lmo:on|v=2000|ob=41|os=2.5|om=signed|rm=trace|rt=12|p=-1"
    )
    assert result["enabled"]["requestLmoKey"] == result["enabled"]["payloadLmoKey"]

    e2e_debug.assert_clean()
