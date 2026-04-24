import pytest


def _compare_metric_error(page, expression: str) -> str | None:
    return page.evaluate(
        """
        ({ expression }) => {
          try {
            Function(expression)();
            return null;
          } catch (error) {
            if (error instanceof Error) return error.message;
            return String(error);
          }
        }
        """,
        {"expression": expression},
    )


@pytest.mark.e2e
def test_compare_metrics_playwright_computes_exact_diff_and_rms(
    page, base_url, e2e_debug
):
    page.goto(f"{base_url}/", wait_until="domcontentloaded")

    result = page.evaluate(
        """
        () => {
          const metrics = window.compareMetrics;
          const a = new Float32Array([1, 2, 3, 4]);
          const b = new Float32Array([2, 1, 5, 0]);
          return {
            diffBA: Array.from(metrics.computeDiff(a, b, 'b_minus_a')),
            diffAB: Array.from(metrics.computeDiff(a, b, 'a_minus_b')),
            rmsByTrace: Array.from(metrics.computeRmsByTrace(a, b, 2, 2)),
          };
        }
        """
    )

    assert result["diffBA"] == [1, -1, 2, -4]
    assert result["diffAB"] == [-1, 1, -2, 4]
    assert result["rmsByTrace"][0] == pytest.approx(((1**2 + 2**2) / 2) ** 0.5)
    assert result["rmsByTrace"][1] == pytest.approx((((-1) ** 2 + (-4) ** 2) / 2) ** 0.5)
    e2e_debug.assert_clean()


@pytest.mark.e2e
def test_compare_metrics_playwright_rejects_invalid_inputs(page, base_url, e2e_debug):
    page.goto(f"{base_url}/", wait_until="domcontentloaded")

    assert _compare_metric_error(
        page,
        "window.compareMetrics.computeDiff(new Float32Array([1]), new Float32Array([1, 2]), 'b_minus_a')",
    ) == "Compare inputs must have the same length."
    assert _compare_metric_error(
        page,
        "window.compareMetrics.computeDiff([1, 2], new Float32Array([1, 2]), 'b_minus_a')",
    ) == "a must be Float32Array."
    assert _compare_metric_error(
        page,
        "window.compareMetrics.computeDiff(new Float32Array([1, NaN]), new Float32Array([1, 2]), 'b_minus_a')",
    ) == "a contains non-finite value at index 1."
    assert _compare_metric_error(
        page,
        "window.compareMetrics.computeSummaryStats(new Float32Array([1, Infinity]))",
    ) == "diff contains non-finite value at index 1."
    assert _compare_metric_error(
        page,
        "window.compareMetrics.computeRmsByTrace(new Float32Array([1, 2, 3, 4]), new Float32Array([2, 3, 4, 5]), 0, 2)",
    ) == "RMS shape must be positive integers."
    assert _compare_metric_error(
        page,
        "window.compareMetrics.computeRmsByTrace(new Float32Array([1, 2, 3, 4]), new Float32Array([2, 3, 4, 5]), 1.5, 2)",
    ) == "RMS shape must be positive integers."
    assert _compare_metric_error(
        page,
        "window.compareMetrics.computeRmsByTrace(new Float32Array([1, 2, 3, 4]), new Float32Array([2, 3, 4, 5]), 3, 2)",
    ) == "RMS shape does not match input length."
    assert _compare_metric_error(
        page,
        "window.compareMetrics.percentileAbs(new Float32Array([1, NaN]), 99)",
    ) == "values contains non-finite value at index 1."
    e2e_debug.assert_clean()
