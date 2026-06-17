let injectedPlotly = null;

export function setPlotlyForTests(plotly) {
  injectedPlotly = plotly || null;
}

export function clearPlotlyForTests() {
  injectedPlotly = null;
}

export function getPlotly(globalObject = globalThis) {
  return injectedPlotly || globalObject?.Plotly || null;
}

export function plotlyUnavailableMessage() {
  return 'Plot library is unavailable.';
}

export function newPlot(plot, traces, layout, config, options = {}) {
  const plotly = getPlotly(options.globalObject);
  if (!plotly || typeof plotly.newPlot !== 'function') {
    throw new Error(plotlyUnavailableMessage());
  }
  return plotly.newPlot(plot, traces, layout, config);
}

export function resizePlot(plot, options = {}) {
  const plotly = getPlotly(options.globalObject);
  if (plotly?.Plots?.resize) plotly.Plots.resize(plot);
}
