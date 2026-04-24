import { expect, test, vi } from 'vitest';

function flushPromises() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function installPlotDiv(id) {
  const plotDiv = document.getElementById(id);
  plotDiv.__plotlyHandlers = {};
  plotDiv.on = (eventName, handler) => {
    const handlers = plotDiv.__plotlyHandlers[eventName] || [];
    handlers.push(handler);
    plotDiv.__plotlyHandlers[eventName] = handlers;
    return plotDiv;
  };
  Object.defineProperty(plotDiv, 'clientWidth', { value: 640, configurable: true });
  Object.defineProperty(plotDiv, 'clientHeight', { value: 320, configurable: true });
  return plotDiv;
}

function getRelayoutHandlers(plotDiv) {
  return plotDiv.__plotlyHandlers?.plotly_relayout || [];
}

function setRange(plotDiv, xRange, yRange) {
  plotDiv._fullLayout = {
    xaxis: { range: xRange },
    yaxis: { range: yRange },
  };
}

test('compare purge clears role markers and side-by-side relayout handlers rebind after reset', async () => {
  document.body.innerHTML = `
    <div id="plot"></div>
    <select id="compareModeSelect">
      <option value="single">single</option>
      <option value="side_by_side">side_by_side</option>
      <option value="difference">difference</option>
    </select>
    <select id="compareSourceASelect"></select>
    <select id="compareSourceBSelect"></select>
    <select id="compareDiffModeSelect">
      <option value="b_minus_a">b_minus_a</option>
      <option value="a_minus_b">a_minus_b</option>
    </select>
    <input id="compareShowRms" type="checkbox" checked />
    <div id="compareStatus"></div>
    <div id="compareStatusViewport"></div>
    <div id="compareSurface"></div>
    <div id="compareSideBySide"></div>
    <div id="compareDiffView"></div>
    <div id="compareSourceALabel"></div>
    <div id="compareSourceBLabel"></div>
    <div id="compareDiffExpression"></div>
    <div id="compareNotice"></div>
    <div id="compareStatMean"></div>
    <div id="compareStatStd"></div>
    <div id="compareStatRms"></div>
    <div id="compareStatMaxAbs"></div>
    <div id="compareRmsSection"></div>
    <div id="comparePlotA"></div>
    <div id="comparePlotB"></div>
    <div id="comparePlotDiff"></div>
    <div id="compareRmsPlot"></div>
    <select id="layerSelect">
      <option value="raw">raw</option>
    </select>
    <select id="colormap">
      <option value="Greys" selected>Greys</option>
    </select>
    <input id="cmReverse" type="checkbox" />
    <input id="gain" value="1" />
    <input id="key1_slider" value="0" />
  `;

  const compareSurface = document.getElementById('compareSurface');
  Object.defineProperty(compareSurface, 'clientWidth', { value: 640, configurable: true });
  Object.defineProperty(compareSurface, 'clientHeight', { value: 320, configurable: true });

  const plotA = installPlotDiv('comparePlotA');
  const plotB = installPlotDiv('comparePlotB');
  const plotDiff = installPlotDiv('comparePlotDiff');
  const plotRms = installPlotDiv('compareRmsPlot');

  global.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };

  const payload = {
    shape: [2, 2],
    x0: 0,
    x1: 1,
    y0: 0,
    y1: 1,
    stepX: 1,
    stepY: 1,
    dt: 0.002,
    effectiveLayer: 'raw',
    zBacking: new Float32Array([1, 2, 3, 4]),
  };
  const windowCache = new Map();

  window.currentFileId = 'file-1';
  window.currentKey1Byte = 189;
  window.currentKey2Byte = 193;
  window.currentScaling = 'amax';
  window.key1Values = [1001];
  window.sectionShape = [2, 2];
  window.savedXRange = null;
  window.savedYRange = null;
  window.computeStepsForWindow = vi.fn(() => ({ step_x: 1, step_y: 1 }));
  window.currentVisibleWindow = vi.fn(() => ({ nTraces: 2, nSamples: 2 }));
  window.readWindowDecodeUseWorker = vi.fn(() => false);
  window.decodeWindowPayload = vi.fn(() => ({ ...payload, zBacking: new Float32Array(payload.zBacking) }));
  window.buildWindowRequestArtifacts = vi.fn((requestContext) => ({
    cacheKey: `${requestContext.effectiveLayer}:${requestContext.key1Val}`,
    params: new URLSearchParams({ layer: requestContext.effectiveLayer, key1: String(requestContext.key1Val) }),
    payloadMeta: { effectiveLayer: requestContext.effectiveLayer },
  }));
  window.windowCacheGet = vi.fn((key) => windowCache.get(key) || null);
  window.windowCacheSet = vi.fn((key, value) => {
    windowCache.set(key, value);
  });
  window.buildLayout = vi.fn((opts) => ({
    xaxis: { range: opts.savedXRange || [opts.x0, opts.x1] },
    yaxis: { range: opts.savedYRange || [opts.y1 * opts.dt, opts.y0 * opts.dt] },
  }));
  window.scheduleWindowFetch = vi.fn();

  global.fetch = vi.fn(async () => ({
    ok: true,
    arrayBuffer: async () => new ArrayBuffer(0),
  }));

  window.Plotly = {
    react: vi.fn(async (plotDiv, data, layout) => {
      plotDiv.data = data;
      plotDiv._fullLayout = {
        xaxis: { range: layout?.xaxis?.range || [0, 1] },
        yaxis: { range: layout?.yaxis?.range || [1, 0] },
      };
    }),
    purge: vi.fn((plotDiv) => {
      plotDiv.data = [];
      plotDiv._fullLayout = {};
      plotDiv.__plotlyHandlers = {};
    }),
    relayout: vi.fn(async (plotDiv, updates) => {
      plotDiv._fullLayout = plotDiv._fullLayout || {};
      plotDiv._fullLayout.xaxis = plotDiv._fullLayout.xaxis || {};
      plotDiv._fullLayout.yaxis = plotDiv._fullLayout.yaxis || {};
      if (updates['xaxis.range']) plotDiv._fullLayout.xaxis.range = updates['xaxis.range'];
      if (updates['yaxis.range']) plotDiv._fullLayout.yaxis.range = updates['yaxis.range'];
    }),
    Plots: {
      resize: vi.fn(),
    },
  };

  await import('../../static/viewer/compare_view.js');
  document.dispatchEvent(new Event('DOMContentLoaded'));

  const compareModeSelect = document.getElementById('compareModeSelect');

  compareModeSelect.value = 'side_by_side';
  compareModeSelect.dispatchEvent(new Event('change'));
  await flushPromises();
  await flushPromises();

  expect(plotA.__svCompareRole).toBe('a');
  expect(plotB.__svCompareRole).toBe('b');
  expect(getRelayoutHandlers(plotA)).toHaveLength(1);
  expect(getRelayoutHandlers(plotB)).toHaveLength(1);

  setRange(plotA, [10, 20], [0.4, 0.1]);
  getRelayoutHandlers(plotA)[0]({});
  await flushPromises();

  expect(window.Plotly.relayout).toHaveBeenCalledTimes(1);
  expect(window.Plotly.relayout).toHaveBeenLastCalledWith(plotB, {
    'xaxis.range': [10, 20],
    'yaxis.range': [0.4, 0.1],
  });

  await window.compareView.renderFromCache();
  expect(getRelayoutHandlers(plotA)).toHaveLength(1);
  expect(getRelayoutHandlers(plotB)).toHaveLength(1);

  plotDiff.__svCompareRole = 'diff';
  plotRms.__svCompareRole = 'rms';
  compareModeSelect.value = 'single';
  compareModeSelect.dispatchEvent(new Event('change'));
  await flushPromises();

  expect(plotA.__svCompareRole).toBeUndefined();
  expect(plotB.__svCompareRole).toBeUndefined();
  expect(plotDiff.__svCompareRole).toBeUndefined();
  expect(plotRms.__svCompareRole).toBeUndefined();

  compareModeSelect.value = 'side_by_side';
  compareModeSelect.dispatchEvent(new Event('change'));
  await flushPromises();
  await flushPromises();

  expect(getRelayoutHandlers(plotA)).toHaveLength(1);
  expect(getRelayoutHandlers(plotB)).toHaveLength(1);

  setRange(plotA, [30, 40], [0.8, 0.2]);
  getRelayoutHandlers(plotA)[0]({});
  await flushPromises();

  expect(window.Plotly.relayout).toHaveBeenCalledTimes(2);
  expect(window.Plotly.relayout).toHaveBeenLastCalledWith(plotB, {
    'xaxis.range': [30, 40],
    'yaxis.range': [0.8, 0.2],
  });
});
