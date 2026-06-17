let plotlyOverride = null;

export function getPlotly(globalObject = globalThis) {
  return plotlyOverride || globalObject?.Plotly || null;
}

export function setPlotlyForTests(plotly) {
  plotlyOverride = plotly || null;
}

export function clearPlotlyForTests() {
  plotlyOverride = null;
}
