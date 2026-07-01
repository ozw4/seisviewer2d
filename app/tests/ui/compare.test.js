import { afterEach, beforeAll, beforeEach, expect, test, vi } from 'vitest';

beforeAll(async () => {
  await import('../../static/viewer/compare/models.js');
  await import('../../static/viewer/compare/sources.js');
  await import('../../static/viewer/compare/data.js');
  await import('../../static/viewer/compare/render.js');
  await import('../../static/viewer/compare.js');
});

beforeEach(() => {
  document.body.innerHTML = `
    <input id="compareShowDiff" type="checkbox" checked>
    <input id="gain" value="2">
    <select id="colormap"><option value="Greys" selected>Greys</option></select>
    <input id="cmReverse" type="checkbox">
  `;
});

afterEach(() => {
  window.__svCompare.setLatestCompareRenderForTest(null);
  window.__svCompare.setCompareFileTargetsForTest([]);
  window.__svCompare.setCompareRecentDatasetsForTest([]);
  window.__svCompare.clearRawCompareValidationCache();
  delete window.buildWindowRequestArtifacts;
  vi.unstubAllGlobals();
});

function okJson(payload = {}) {
  return {
    ok: true,
    headers: { get: () => 'application/json' },
    json: async () => payload,
  };
}

function errorText(status, text) {
  return {
    ok: false,
    status,
    headers: { get: () => 'text/plain' },
    text: async () => text,
  };
}

function okBinary() {
  return {
    ok: true,
    headers: { get: () => 'application/octet-stream' },
    arrayBuffer: async () => new ArrayBuffer(0),
  };
}

async function flushPromises() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

function setupCompareFetchHarness() {
  document.body.innerHTML = `
    <input id="compareModeToggle" type="checkbox" checked>
    <select id="compareSourceA"></select>
    <select id="compareSourceB"></select>
    <input id="compareShowDiff" type="checkbox">
    <input id="gain" value="2">
    <select id="colormap"><option value="Greys" selected>Greys</option></select>
    <input id="cmReverse" type="checkbox">
    <select id="layerSelect"><option value="raw" selected>raw</option></select>
    <input id="key1_slider" value="0">
    <div id="compareStatus" hidden></div>
    <div id="plot"></div>
  `;
  const plot = document.getElementById('plot');
  Object.defineProperty(plot, 'clientWidth', { configurable: true, value: 800 });
  Object.defineProperty(plot, 'clientHeight', { configurable: true, value: 400 });

  vi.stubGlobal('currentFileId', 'active-file');
  vi.stubGlobal('currentFileName', 'active.sgy');
  vi.stubGlobal('currentKey1Byte', 189);
  vi.stubGlobal('currentKey2Byte', 193);
  vi.stubGlobal('key1Values', [101]);
  vi.stubGlobal('sectionShape', [1, 1]);
  vi.stubGlobal('currentScaling', 'amax');
  vi.stubGlobal('currentVisibleWindow', vi.fn(() => ({
    x0: 0,
    x1: 1,
    y0: 0,
    y1: 1,
    nTraces: 1,
    nSamples: 1,
  })));
  vi.stubGlobal('wantWiggleForWindow', vi.fn(() => false));
  vi.stubGlobal('computeStepsForWindow', vi.fn(() => ({ step_x: 1, step_y: 1 })));
  vi.stubGlobal('buildWindowLoadingMessage', vi.fn(() => 'loading'));
  vi.stubGlobal('showLoading', vi.fn());
  vi.stubGlobal('hideLoading', vi.fn());
  vi.stubGlobal('cancelActiveMainDecodeJob', vi.fn());
  vi.stubGlobal('windowFetchCtrl', null);
  vi.stubGlobal('currentLmoKey', vi.fn(() => 'lmo:off'));
  vi.stubGlobal('windowCacheGet', vi.fn(() => null));
  vi.stubGlobal('windowCacheSet', vi.fn());
  vi.stubGlobal('decodeWindowPayload', vi.fn(() => null));
  vi.stubGlobal('beginRenderRequest', vi.fn(() => ({
    requestId: 7,
    signal: new AbortController().signal,
  })));
  vi.stubGlobal('isCurrentRenderRequest', vi.fn(() => true));
  vi.stubGlobal('markRenderRequestCompleted', vi.fn());
  vi.stubGlobal('markRenderRequestFailed', vi.fn());
  vi.stubGlobal('markStaleRenderDropped', vi.fn());
  vi.stubGlobal('abortRenderRequest', vi.fn());
  vi.stubGlobal('buildWindowRequestArtifacts', vi.fn((context) => {
    const params = new URLSearchParams({
      file_id: context.fileId,
      normalization_file_id: context.normalizationFileId,
    });
    return {
      params,
      cacheKey: `cache:${context.fileId}:${context.normalizationFileId}`,
      payloadMeta: { lmoKey: 'lmo:off', fileId: context.fileId },
    };
  }));

  window.__svCompare.setCompareFileTargetsForTest([
    {
      fileId: 'active-file',
      displayName: 'active.sgy',
      key1Byte: 189,
      key2Byte: 193,
      isActive: true,
    },
    {
      fileId: 'added-file',
      displayName: 'added.sgy',
      key1Byte: 189,
      key2Byte: 193,
      isActive: false,
    },
  ]);
  window.updateCompareSourceOptions();
  document.getElementById('compareSourceA').value = 'file:added-file:raw';
  document.getElementById('compareSourceB').value = 'file:active-file:raw';
}

function setupCompareImportDom() {
  document.body.innerHTML = `
    <select id="compareDatasetPicker"></select>
    <button type="button" id="compareAddDataset">Add dataset</button>
    <button type="button" id="compareImportBSource">Import B source…</button>
    <input id="compareBSourceFile" type="file" accept=".sgy,.segy" hidden>
    <select id="compareSourceA"></select>
    <select id="compareSourceB"></select>
    <input id="compareShowDiff" type="checkbox" checked>
    <select id="layerSelect"><option value="raw" selected>raw</option></select>
    <div id="compareStatus" hidden></div>
  `;
  vi.stubGlobal('currentFileId', 'active-file');
  vi.stubGlobal('currentFileName', 'active.sgy');
  vi.stubGlobal('currentKey1Byte', 189);
  vi.stubGlobal('currentKey2Byte', 193);
}

function scale(panel) {
  return window.__svCompare.compareHeatmapScale(panel, 2);
}

function expectF32Values(actual, expected) {
  expect(actual).toBeInstanceOf(Float32Array);
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, index) => {
    expect(actual[index]).toBeCloseTo(value, 6);
  });
}

test('compare heatmap scales mixed-domain source panels independently', () => {
  expect(scale({ kind: 'source', domain: 'amplitude' })).toMatchObject({
    zmin: -1.5,
    zmax: 1.5,
    signed: true,
  });
  expect(scale({ kind: 'source', domain: 'probability' })).toMatchObject({
    zmin: 0,
    zmax: 0.5,
    signed: false,
  });
});

test('compare heatmap allows signed probability diff values', () => {
  const render = {
    sources: {
      a: { domain: 'probability', label: 'prob-a' },
      b: { domain: 'probability', label: 'prob-b' },
    },
    a: { values: new Float32Array([0.1]) },
    b: { values: new Float32Array([0.8]) },
    diffAvailable: true,
    diffValues: new Float32Array([-0.7]),
  };

  const panels = window.__svCompare.buildComparePanels(render);
  const diffPanel = panels.find((panel) => panel.kind === 'diff');

  expect(diffPanel).toMatchObject({ role: 'A-B', domain: 'probability' });
  expect(scale(diffPanel)).toMatchObject({
    zmin: -0.5,
    zmax: 0.5,
    signed: true,
  });
});

test('compare heatmap uses signed amplitude scale for amplitude diff values', () => {
  expect(scale({ kind: 'diff', domain: 'amplitude' })).toMatchObject({
    zmin: -1.5,
    zmax: 1.5,
    signed: true,
  });
});

test('compare payload decode prefers quantized compute values over display backing', () => {
  const payload = {
    shape: [1, 2],
    valuesI8: new Int8Array([32, 64]),
    scale: 128,
    zBacking: new Float32Array([64, 128]),
  };

  const values = window.__svCompare.payloadToF32(payload, { domain: 'probability' });

  expectF32Values(values, [0.25, 0.5]);
});

test('compare payload decode rejects display backing for probability sources', () => {
  expect(window.__svCompare.payloadToF32({
    shape: [1, 2],
    zBacking: new Float32Array([64, 128]),
  }, { domain: 'probability' })).toBeNull();

  expect(window.__svCompare.payloadToF32({
    shape: [1, 2],
    zRows: [new Float32Array([64, 128])],
  }, { domain: 'probability' })).toBeNull();
});

test('compare payload decode keeps amplitude display backing fallback', () => {
  const values = window.__svCompare.payloadToF32({
    shape: [1, 2],
    zBacking: new Float32Array([1.5, -2.25]),
  }, { domain: 'amplitude' });

  expectF32Values(values, [1.5, -2.25]);
});

test('probability diff uses decoded probability values, not display-scaled backing', () => {
  const sourceA = window.__svCompare.payloadToF32({
    shape: [1, 1],
    valuesI8: new Int8Array([80]),
    scale: 100,
    zBacking: new Float32Array([204]),
  }, { domain: 'probability' });
  const sourceB = window.__svCompare.payloadToF32({
    shape: [1, 1],
    valuesI8: new Int8Array([20]),
    scale: 100,
    zBacking: new Float32Array([51]),
  }, { domain: 'probability' });

  const diff = window.__svCompare.subtractF32(sourceA, sourceB);

  expectF32Values(sourceA, [0.8]);
  expectF32Values(sourceB, [0.2]);
  expectF32Values(diff, [0.6]);
});

test('sourcePairKey distinguishes raw sources by fileId', () => {
  const firstPair = {
    a: { fileId: 'line-a.sgy', layerId: 'raw', pipelineKey: null, tapLabel: null },
    b: { fileId: 'line-b.sgy', layerId: 'raw', pipelineKey: null, tapLabel: null },
  };
  const secondPair = {
    a: { fileId: 'line-a.sgy', layerId: 'raw', pipelineKey: null, tapLabel: null },
    b: { fileId: 'line-c.sgy', layerId: 'raw', pipelineKey: null, tapLabel: null },
  };

  expect(window.__svCompare.sourcePairKey(firstPair)).not.toEqual(
    window.__svCompare.sourcePairKey(secondPair),
  );
});

test('buildCompareRequest sends raw A/B windows with source file ids and A normalization file id', () => {
  vi.stubGlobal('currentScaling', 'amax');
  const buildArtifacts = vi.fn((context) => {
    const params = new URLSearchParams();
    if (context.normalizationFileId) {
      params.set('normalization_file_id', context.normalizationFileId);
    }
    return { params, cacheKey: `cache:${context.fileId}:${context.normalizationFileId || ''}`, payloadMeta: {} };
  });
  window.buildWindowRequestArtifacts = buildArtifacts;
  vi.stubGlobal('currentFileId', 'active-file');
  const sourceA = {
    id: 'raw',
    layerId: 'raw',
    fileId: 'file-a',
    key1Byte: 189,
    key2Byte: 193,
    pipelineKey: null,
    tapLabel: null,
  };
  const sourceB = {
    id: 'raw',
    layerId: 'raw',
    fileId: 'file-b',
    key1Byte: 189,
    key2Byte: 193,
    pipelineKey: null,
    tapLabel: null,
  };

  const requestA = window.__svCompare.buildCompareRequest(
    sourceA,
    sourceA,
    101,
    { x0: 0, x1: 10, y0: 0, y1: 20 },
    { stepX: 1, stepY: 1, mode: 'heatmap' },
  );
  const requestB = window.__svCompare.buildCompareRequest(
    sourceB,
    sourceA,
    101,
    { x0: 0, x1: 10, y0: 0, y1: 20 },
    { stepX: 1, stepY: 1, mode: 'heatmap' },
  );

  expect(buildArtifacts).toHaveBeenNthCalledWith(1, expect.objectContaining({
    fileId: 'file-a',
    normalizationFileId: 'file-a',
  }));
  expect(buildArtifacts).toHaveBeenNthCalledWith(2, expect.objectContaining({
    fileId: 'file-b',
    normalizationFileId: 'file-a',
  }));
  expect(requestA.params.get('normalization_file_id')).toBe('file-a');
  expect(requestB.params.get('normalization_file_id')).toBe('file-a');
  expect(requestA.cacheKey).not.toBe(requestB.cacheKey);
});

test('buildCompareRequest keeps current file fallback for single-file sources', () => {
  vi.stubGlobal('currentScaling', 'amax');
  vi.stubGlobal('currentFileId', 'active-file');
  const buildArtifacts = vi.fn(() => ({
    params: new URLSearchParams(),
    cacheKey: 'cache:active-file',
    payloadMeta: {},
  }));
  window.buildWindowRequestArtifacts = buildArtifacts;
  const source = {
    id: 'raw',
    layerId: 'raw',
    key1Byte: 189,
    key2Byte: 193,
    pipelineKey: null,
    tapLabel: null,
  };

  window.__svCompare.buildCompareRequest(
    source,
    source,
    101,
    { x0: 0, x1: 10, y0: 0, y1: 20 },
    { stepX: 1, stepY: 1, mode: 'heatmap' },
  );

  expect(buildArtifacts).toHaveBeenCalledWith(expect.objectContaining({
    fileId: 'active-file',
    normalizationFileId: 'active-file',
  }));
});

test('raw compare validation calls backend for distinct raw file sources', async () => {
  const fetchMock = vi.fn(async () => ({
    ok: true,
    headers: { get: () => 'application/json' },
    json: async () => ({ ok: true, reason: '', message: '' }),
  }));
  vi.stubGlobal('fetch', fetchMock);

  const result = await window.__svCompare.validateRawCompareSources({
    a: { id: 'raw', layerId: 'raw', fileId: 'file-a', key1Byte: 189, key2Byte: 193 },
    b: { id: 'raw', layerId: 'raw', fileId: 'file-b', key1Byte: 189, key2Byte: 193 },
  });

  expect(result).toMatchObject({ ok: true });
  expect(fetchMock).toHaveBeenCalledTimes(1);
  const url = new URL(fetchMock.mock.calls[0][0], 'http://localhost');
  expect(url.pathname).toBe('/compare/raw/validate');
  expect(url.searchParams.get('file_id_a')).toBe('file-a');
  expect(url.searchParams.get('file_id_b')).toBe('file-b');
  expect(url.searchParams.get('key1_byte')).toBe('189');
  expect(url.searchParams.get('key2_byte')).toBe('193');
});

test('raw compare validation cache key includes file ids and key bytes', () => {
  const sources = {
    a: { id: 'raw', layerId: 'raw', fileId: 'file-a' },
    b: { id: 'raw', layerId: 'raw', fileId: 'file-b' },
  };

  expect(window.__svCompare.rawCompareValidationKey(sources, 189, 193)).not.toBe(
    window.__svCompare.rawCompareValidationKey({
      a: { id: 'raw', layerId: 'raw', fileId: 'file-a' },
      b: { id: 'raw', layerId: 'raw', fileId: 'file-c' },
    }, 189, 193),
  );
  expect(window.__svCompare.rawCompareValidationKey(sources, 189, 193)).not.toBe(
    window.__svCompare.rawCompareValidationKey(sources, 17, 193),
  );
  expect(window.__svCompare.rawCompareValidationKey(sources, 189, 193)).not.toBe(
    window.__svCompare.rawCompareValidationKey(sources, 189, 197),
  );
});

test('raw compare validation caches ok=true by source pair and key bytes', async () => {
  const fetchMock = vi.fn(async () => ({
    ok: true,
    headers: { get: () => 'application/json' },
    json: async () => ({ ok: true, reason: '', message: '' }),
  }));
  vi.stubGlobal('fetch', fetchMock);
  const sources = {
    a: { id: 'raw', layerId: 'raw', fileId: 'file-a', key1Byte: 189, key2Byte: 193 },
    b: { id: 'raw', layerId: 'raw', fileId: 'file-b', key1Byte: 189, key2Byte: 193 },
  };

  await window.__svCompare.validateRawCompareSources(sources);
  await window.__svCompare.validateRawCompareSources(sources);
  await window.__svCompare.validateRawCompareSources({
    a: { ...sources.a, key1Byte: 17 },
    b: { ...sources.b, key1Byte: 17 },
  });

  expect(fetchMock).toHaveBeenCalledTimes(2);
});

test('raw compare validation ok=false returns backend message for caller to stop fetch', async () => {
  const fetchMock = vi.fn(async () => ({
    ok: true,
    headers: { get: () => 'application/json' },
    json: async () => ({
      ok: false,
      reason: 'key2_sequence',
      message: 'A-B unavailable: key2 sequence differs.',
    }),
  }));
  vi.stubGlobal('fetch', fetchMock);

  const result = await window.__svCompare.validateRawCompareSources({
    a: { id: 'raw', layerId: 'raw', fileId: 'file-a', key1Byte: 189, key2Byte: 193 },
    b: { id: 'raw', layerId: 'raw', fileId: 'file-b', key1Byte: 189, key2Byte: 193 },
  });

  expect(result).toMatchObject({
    ok: false,
    reason: 'key2_sequence',
    message: 'A-B unavailable: key2 sequence differs.',
  });
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

test('raw compare preflights A section meta after validation before A/B window fetches', async () => {
  setupCompareFetchHarness();
  const fetchMock = vi.fn(async (input) => {
    const url = new URL(input, 'http://localhost');
    if (url.pathname === '/compare/raw/validate') {
      return okJson({ ok: true, reason: '', message: '' });
    }
    if (url.pathname === '/get_section_meta') return okJson({ shape: [1, 1] });
    if (url.pathname === '/get_section_window_bin') return okBinary();
    throw new Error(`unexpected fetch: ${url.pathname}`);
  });
  vi.stubGlobal('fetch', fetchMock);

  await window.fetchCompareAndPlot();

  expect(fetchMock).toHaveBeenCalledTimes(4);
  const urls = fetchMock.mock.calls.map(([input]) => new URL(input, 'http://localhost'));
  expect(urls.map((url) => url.pathname)).toEqual([
    '/compare/raw/validate',
    '/get_section_meta',
    '/get_section_window_bin',
    '/get_section_window_bin',
  ]);
  expect(urls[0].searchParams.get('file_id_a')).toBe('added-file');
  expect(urls[0].searchParams.get('file_id_b')).toBe('active-file');
  expect(urls[1].searchParams.get('file_id')).toBe('added-file');
  expect(urls[1].searchParams.get('key1_byte')).toBe('189');
  expect(urls[1].searchParams.get('key2_byte')).toBe('193');
  expect(urls[2].searchParams.get('normalization_file_id')).toBe('added-file');
  expect(urls[3].searchParams.get('normalization_file_id')).toBe('added-file');
});

test('raw compare preflight failure stops window fetch and reports status', async () => {
  setupCompareFetchHarness();
  const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  const plot = document.getElementById('plot');
  plot.__svComparePanelCount = 3;
  plot.__svCompareMode = 'heatmap';
  window.__svCompare.setLatestCompareRenderForTest({ mode: 'heatmap', key1: 101 });
  const react = vi.fn(async () => true);
  vi.stubGlobal('Plotly', { react });
  const fetchMock = vi.fn(async (input) => {
    const url = new URL(input, 'http://localhost');
    if (url.pathname === '/compare/raw/validate') {
      return okJson({ ok: true, reason: '', message: '' });
    }
    if (url.pathname === '/get_section_meta') {
      return errorText(500, 'baseline failed');
    }
    if (url.pathname === '/get_section_window_bin') return okBinary();
    throw new Error(`unexpected fetch: ${url.pathname}`);
  });
  vi.stubGlobal('fetch', fetchMock);

  try {
    await window.fetchCompareAndPlot();

    const paths = fetchMock.mock.calls.map(([input]) => new URL(input, 'http://localhost').pathname);
    expect(paths).toEqual(['/compare/raw/validate', '/get_section_meta']);
    expect(react).toHaveBeenCalledTimes(1);
    expect(react.mock.calls[0][0]).toBe(plot);
    expect(react.mock.calls[0][1]).toEqual([]);
    expect(react.mock.calls[0][2].annotations[0].text).toBe('baseline failed');
    expect(plot.__svComparePanelCount).toBe(0);
    expect(plot.__svCompareMode).toBe('unavailable');
    expect(window.__svCompare.getLatestCompareRender()).toBeNull();
    expect(document.getElementById('compareStatus').textContent).toBe('baseline failed');
    expect(document.getElementById('compareStatus').hidden).toBe(false);
    expect(globalThis.markRenderRequestFailed).toHaveBeenCalledWith('compare-window', 7);
  } finally {
    warnSpy.mockRestore();
  }
});

test('stale raw compare preflight does not fetch windows or overwrite status', async () => {
  setupCompareFetchHarness();
  document.getElementById('compareStatus').textContent = 'current status';
  document.getElementById('compareStatus').hidden = false;
  let current = true;
  globalThis.isCurrentRenderRequest.mockImplementation(() => current);
  const fetchMock = vi.fn(async (input) => {
    const url = new URL(input, 'http://localhost');
    if (url.pathname === '/compare/raw/validate') {
      return okJson({ ok: true, reason: '', message: '' });
    }
    if (url.pathname === '/get_section_meta') {
      current = false;
      return okJson({ shape: [1, 1] });
    }
    if (url.pathname === '/get_section_window_bin') return okBinary();
    throw new Error(`unexpected fetch: ${url.pathname}`);
  });
  vi.stubGlobal('fetch', fetchMock);

  await window.fetchCompareAndPlot();

  const paths = fetchMock.mock.calls.map(([input]) => new URL(input, 'http://localhost').pathname);
  expect(paths).toEqual(['/compare/raw/validate', '/get_section_meta']);
  expect(globalThis.markStaleRenderDropped).toHaveBeenCalledWith('compare-window', 7);
  expect(document.getElementById('compareStatus').textContent).toBe('current status');
  expect(document.getElementById('compareStatus').hidden).toBe(false);
});

test('renderCompareUnavailable clears previous compare plot and status shows validation message', async () => {
  document.body.innerHTML += `
    <div id="plot"></div>
    <div id="compareStatus" hidden></div>
  `;
  const plot = document.getElementById('plot');
  const message = 'A-B unavailable: key2 sequence differs.';
  plot.__svComparePanelCount = 3;
  plot.__svCompareMode = 'heatmap';
  const react = vi.fn(async () => true);
  vi.stubGlobal('Plotly', { react });

  const rendered = await window.__svCompare.renderCompareUnavailable(message);

  expect(rendered).toBe(true);
  expect(window.__svCompare.getLatestCompareRender()).toBeNull();
  expect(react).toHaveBeenCalledTimes(1);
  expect(react.mock.calls[0][0]).toBe(plot);
  expect(react.mock.calls[0][1]).toEqual([]);
  expect(react.mock.calls[0][2].annotations[0].text).toBe(message);
  expect(plot.__svComparePanelCount).toBe(0);
  expect(plot.__svCompareMode).toBe('unavailable');
  expect(document.getElementById('compareStatus').textContent).toBe(message);
  expect(document.getElementById('compareStatus').hidden).toBe(false);
});

test('renderCompareUnavailable does not clear plot when queued request is stale', async () => {
  document.body.innerHTML += `
    <div id="plot"></div>
    <div id="compareStatus" hidden></div>
  `;
  const plot = document.getElementById('plot');
  plot.__svComparePanelCount = 3;
  plot.__svCompareMode = 'heatmap';
  const previousRender = { mode: 'heatmap', key1: 100 };
  window.__svCompare.setLatestCompareRenderForTest(previousRender);
  const react = vi.fn(async () => true);
  const queueViewerPlotlyRender = vi.fn(async (_plot, renderData) => {
    expect(renderData).toEqual({
      __requestSlot: 'compare-window',
      __requestId: 42,
    });
    return false;
  });
  vi.stubGlobal('Plotly', { react });
  vi.stubGlobal('isCurrentRenderRequest', vi.fn(() => true));
  vi.stubGlobal('queueViewerPlotlyRender', queueViewerPlotlyRender);

  const rendered = await window.__svCompare.renderCompareUnavailable(
    'A-B unavailable: stale.',
    42,
  );

  expect(rendered).toBe(false);
  expect(queueViewerPlotlyRender).toHaveBeenCalledTimes(1);
  expect(react).not.toHaveBeenCalled();
  expect(plot.__svComparePanelCount).toBe(3);
  expect(plot.__svCompareMode).toBe('heatmap');
  expect(window.__svCompare.getLatestCompareRender()).toBe(previousRender);
  expect(document.getElementById('compareStatus').hidden).toBe(true);
});

test('buildComparePanels uses file names in raw diff label', () => {
  const panels = window.__svCompare.buildComparePanels({
    sources: {
      a: { domain: 'amplitude', label: 'line_a.sgy / raw' },
      b: { domain: 'amplitude', label: 'line_b.sgy / raw' },
    },
    a: { values: new Float32Array([1]) },
    b: { values: new Float32Array([2]) },
    diffAvailable: true,
    diffValues: new Float32Array([-1]),
  });

  expect(panels.map((panel) => panel.label)).toEqual([
    'line_a.sgy / raw',
    'line_b.sgy / raw',
    'line_a.sgy / raw - line_b.sgy / raw',
  ]);
});

test('raw source unavailable message does not use tap pipeline wording', () => {
  const message = window.__svCompare.compareUnavailableMessage({
    a: { id: 'raw', layerId: 'raw', available: false, domain: 'amplitude' },
    b: { id: 'raw', layerId: 'raw', available: true, domain: 'amplitude' },
  });

  expect(message).toBe('A-B unavailable: A raw source is not available.');
  expect(message).not.toMatch(/Run pipeline first|tap/i);
});

test('compare recent dataset selection includes key-byte identity', () => {
  const datasets = [
    { original_name: 'line.sgy', store_name: 'a', key1_byte: 189, key2_byte: 193 },
    { original_name: 'line.sgy', store_name: 'b', key1_byte: 17, key2_byte: 193 },
  ];
  const selectedValue = window.__svCompare.compareRecentDatasetValue({
    originalName: 'line.sgy',
    storeName: 'b',
    key1Byte: 17,
    key2Byte: 193,
  });

  const selected = window.__svCompare.resolveCompareRecentDataset(datasets, selectedValue);

  expect(selected).toMatchObject({
    originalName: 'line.sgy',
    key1Byte: 17,
    key2Byte: 193,
  });
  expect(selectedValue).toBe('store:b|17|193');
  expect(selectedValue).not.toEqual(window.__svCompare.compareRecentDatasetValue({
    originalName: 'line.sgy',
    storeName: 'a',
    key1Byte: 189,
    key2Byte: 193,
  }));
});

test('compareRecentDatasetValue falls back to original name for old recent responses', () => {
  expect(window.__svCompare.compareRecentDatasetValue({
    originalName: 'line.sgy',
    key1Byte: 189,
    key2Byte: 193,
  })).toBe('name:line.sgy|189|193');
});

test('compare source catalog builds active raw source with file metadata', () => {
  const catalog = window.__svCompare.buildCompareSourceCatalog([
    {
      fileId: 'line/a.sgy',
      displayName: 'line_a.sgy',
      key1Byte: 189,
      key2Byte: 193,
      isActive: true,
    },
  ], { layerValues: ['raw'] });

  expect(catalog).toHaveLength(1);
  expect(catalog[0]).toMatchObject({
    sourceId: 'file:line%2Fa.sgy:raw',
    fileId: 'line/a.sgy',
    fileName: 'line_a.sgy',
    key1Byte: 189,
    key2Byte: 193,
    layerId: 'raw',
    label: 'line_a.sgy / raw',
    pipelineKey: null,
    tapLabel: null,
    domain: 'amplitude',
    available: true,
  });
});

test('compare source catalog does not expose taps for non-active targets', () => {
  const catalog = window.__svCompare.buildCompareSourceCatalog([
    {
      fileId: 'active.sgy',
      displayName: 'active.sgy',
      key1Byte: 189,
      key2Byte: 193,
      isActive: true,
    },
    {
      fileId: 'other.sgy',
      displayName: 'other.sgy',
      key1Byte: 189,
      key2Byte: 193,
      isActive: false,
    },
  ], {
    layerValues: ['raw', 'fbpick_prob'],
    latestPipelineKey: 'pipeline-1',
    latestTapData: { fbpick_prob: { prob: true } },
  });

  expect(catalog.map((source) => source.sourceId)).toEqual([
    'file:active.sgy:raw',
    'file:active.sgy:tap:fbpick_prob',
    'file:other.sgy:raw',
  ]);
  expect(catalog.find((source) => source.fileId === 'other.sgy' && source.tapLabel)).toBeUndefined();
});

test('compare dataset target list keeps active target and added raw target', () => {
  const active = {
    fileId: 'active-file-id',
    displayName: 'active.sgy',
    originalName: 'active.sgy',
    key1Byte: 189,
    key2Byte: 193,
    isActive: true,
  };

  const result = window.__svCompare.addCompareDatasetTarget([], {
    fileId: 'added-file-id',
    displayName: 'added.sgy',
    originalName: 'added.sgy',
    key1Byte: 189,
    key2Byte: 193,
  }, active);
  const catalog = window.__svCompare.buildCompareSourceCatalog(result.targets, {
    layerValues: ['raw', 'fbpick_prob'],
    latestPipelineKey: 'pipeline-1',
  });

  expect(result.added).toBe(true);
  expect(result.targets).toMatchObject([
    { fileId: 'active-file-id', displayName: 'active.sgy', isActive: true },
    { fileId: 'added-file-id', displayName: 'added.sgy', isActive: false },
  ]);
  expect(catalog.map((source) => source.label)).toContain('active.sgy / raw');
  expect(catalog.map((source) => source.label)).toContain('added.sgy / raw');
  expect(catalog.find((source) => source.fileId === 'added-file-id' && source.layerId !== 'raw')).toBeUndefined();
});

test('compare dataset manager allows same basename with different hashes and dedupes same hash', () => {
  const active = {
    fileId: 'active-file-id',
    displayName: 'active.sgy',
    originalName: 'active.sgy',
    key1Byte: 189,
    key2Byte: 193,
    isActive: true,
  };

  const first = window.__svCompare.addCompareDatasetTarget([], {
    fileId: 'added-file-a',
    displayName: 'line.sgy',
    originalName: 'line.sgy',
    sourceSha256: 'aaaaaaaa11111111',
    key1Byte: 189,
    key2Byte: 193,
  }, active);
  const second = window.__svCompare.addCompareDatasetTarget(first.targets, {
    fileId: 'added-file-b',
    displayName: 'line.sgy',
    originalName: 'line.sgy',
    sourceSha256: 'bbbbbbbb22222222',
    key1Byte: 189,
    key2Byte: 193,
  }, active);
  const duplicate = window.__svCompare.addCompareDatasetTarget(second.targets, {
    fileId: 'added-file-c',
    displayName: 'line.sgy',
    originalName: 'line.sgy',
    sourceSha256: 'bbbbbbbb22222222',
    key1Byte: 189,
    key2Byte: 193,
  }, active);
  const hashAKey = window.__svCompare.compareTargetDatasetKey(first.targets[1]);
  const hashBKey = window.__svCompare.compareTargetDatasetKey(second.targets[2]);

  expect(first.added).toBe(true);
  expect(second.added).toBe(true);
  expect(hashAKey).toBe('sha256:aaaaaaaa11111111|189|193');
  expect(hashBKey).toBe('sha256:bbbbbbbb22222222|189|193');
  expect(hashAKey).not.toBe(hashBKey);
  expect(second.targets.map((target) => target.fileId)).toEqual([
    'active-file-id',
    'added-file-a',
    'added-file-b',
  ]);
  expect(duplicate.added).toBe(false);
  expect(duplicate.reason).toMatch(/already added/i);
  expect(duplicate.targets).toHaveLength(3);
});

test('compare source catalog distinguishes same display names with hash suffix', () => {
  const catalog = window.__svCompare.buildCompareSourceCatalog([
    {
      fileId: 'active-file',
      displayName: 'line.sgy',
      originalName: 'line.sgy',
      key1Byte: 189,
      key2Byte: 193,
      isActive: true,
    },
    {
      fileId: 'added-file',
      displayName: 'line.sgy',
      originalName: 'line.sgy',
      sourceSha256: 'abcdef1234567890',
      key1Byte: 189,
      key2Byte: 193,
      isActive: false,
    },
  ], {
    layerValues: ['raw', 'fbpick_prob'],
    latestPipelineKey: 'pipeline-1',
  });

  expect(catalog.map((source) => source.label)).toEqual([
    'line.sgy / raw',
    'line.sgy / fbpick_prob',
    'line.sgy [abcdef12] / raw',
  ]);
  expect(catalog.find((source) => source.fileId === 'added-file' && source.tapLabel)).toBeUndefined();
});

test('addSelectedCompareDataset sends store_name to open_segy', async () => {
  document.body.innerHTML = `
    <select id="compareDatasetPicker"></select>
    <select id="compareSourceA"></select>
    <select id="compareSourceB"></select>
    <select id="layerSelect"><option value="raw" selected>raw</option></select>
    <div id="compareStatus" hidden></div>
  `;
  vi.stubGlobal('currentFileId', 'active-file');
  vi.stubGlobal('currentFileName', 'active.sgy');
  vi.stubGlobal('currentKey1Byte', 189);
  vi.stubGlobal('currentKey2Byte', 193);
  const fetch = vi.fn(async () => okJson({ file_id: 'opened-file' }));
  vi.stubGlobal('fetch', fetch);

  window.__svCompare.setCompareRecentDatasetsForTest([
    {
      original_name: 'line.sgy',
      store_name: 'line-b__k189_193__sha256_abcdef1234567890',
      source_sha256: 'abcdef1234567890',
      key1_byte: 189,
      key2_byte: 193,
    },
  ]);

  const added = await window.__svCompare.addSelectedCompareDataset();

  expect(added).toBe(true);
  expect(fetch).toHaveBeenCalledWith('/open_segy', expect.objectContaining({ method: 'POST' }));
  const formData = fetch.mock.calls[0][1].body;
  expect(formData.get('store_name')).toBe('line-b__k189_193__sha256_abcdef1234567890');
  expect(formData.get('original_name')).toBe('line.sgy');
  expect(window.compareFileTargets).toMatchObject([
    { fileId: 'active-file', isActive: true },
    {
      fileId: 'opened-file',
      originalName: 'line.sgy',
      storeName: 'line-b__k189_193__sha256_abcdef1234567890',
      sourceSha256: 'abcdef1234567890',
      isActive: false,
    },
  ]);
});

test('import B source button opens hidden file picker', async () => {
  setupCompareImportDom();
  const fetch = vi.fn(async () => okJson({ datasets: [] }));
  vi.stubGlobal('fetch', fetch);
  window.__svCompare.initCompareControls();
  const input = document.getElementById('compareBSourceFile');
  const click = vi.spyOn(input, 'click');

  document.getElementById('compareImportBSource').click();

  expect(click).toHaveBeenCalledTimes(1);
});

test('file selection imports B source with active key bytes and selects imported raw source', async () => {
  setupCompareImportDom();
  const file = new File(['sgy'], 'line-b.sgy', { type: 'application/octet-stream' });
  const fetch = vi.fn(async (url) => {
    if (url === '/compare/raw/import') {
      return okJson({
        file_id: 'imported-file',
        display_name: 'imported B',
        original_name: 'line-b.sgy',
        store_name: 'imports/line-b.sgy',
        source_sha256: 'abcdef1234567890',
        key1_byte: 189,
        key2_byte: 193,
      });
    }
    return okJson({ datasets: [] });
  });
  vi.stubGlobal('fetch', fetch);
  window.__svCompare.initCompareControls();
  const input = document.getElementById('compareBSourceFile');
  Object.defineProperty(input, 'files', { configurable: true, value: [file] });

  input.dispatchEvent(new Event('change'));
  await flushPromises();

  const importCall = fetch.mock.calls.find(([url]) => url === '/compare/raw/import');
  expect(importCall).toBeTruthy();
  const formData = importCall[1].body;
  expect(formData.get('file').name).toBe('line-b.sgy');
  expect(formData.get('key1_byte')).toBe('189');
  expect(formData.get('key2_byte')).toBe('193');
  expect(window.compareFileTargets).toMatchObject([
    { fileId: 'active-file', isActive: true },
    {
      fileId: 'imported-file',
      displayName: 'imported B',
      originalName: 'line-b.sgy',
      storeName: 'imports/line-b.sgy',
      sourceSha256: 'abcdef1234567890',
      isActive: false,
    },
  ]);
  expect(document.getElementById('compareSourceA').value).toBe(
    window.__svCompare.compareSourceId('active-file', 'raw'),
  );
  expect(document.getElementById('compareSourceB').value).toBe(
    window.__svCompare.compareSourceId('imported-file', 'raw'),
  );
  expect(document.getElementById('compareStatus').textContent).toBe('B source imported.');
});

test('import B source clears raw validation cache', async () => {
  setupCompareImportDom();
  const sources = {
    a: { fileId: 'active-file', layerId: 'raw', key1Byte: 189, key2Byte: 193 },
    b: { fileId: 'other-file', layerId: 'raw', key1Byte: 189, key2Byte: 193 },
  };
  const fetch = vi.fn(async (url) => {
    if (String(url).startsWith('/compare/raw/validate?')) return okJson({ ok: true });
    if (url === '/compare/raw/import') return okJson({ file_id: 'imported-file', original_name: 'line-b.sgy' });
    return okJson({ datasets: [] });
  });
  vi.stubGlobal('fetch', fetch);

  await window.__svCompare.validateRawCompareSources(sources);
  await window.__svCompare.validateRawCompareSources(sources);
  expect(fetch.mock.calls.filter(([url]) => String(url).startsWith('/compare/raw/validate?'))).toHaveLength(1);

  await window.__svCompare.importCompareBSourceFile(new File(['sgy'], 'line-b.sgy'));
  await window.__svCompare.validateRawCompareSources(sources);

  expect(fetch.mock.calls.filter(([url]) => String(url).startsWith('/compare/raw/validate?'))).toHaveLength(2);
});

test('import B source failure does not add target and shows backend detail', async () => {
  setupCompareImportDom();
  const fetch = vi.fn(async () => errorText(400, 'bad SEG-Y file'));
  vi.stubGlobal('fetch', fetch);

  const added = await window.__svCompare.importCompareBSourceFile(new File(['bad'], 'bad.sgy'));

  expect(added).toBe(false);
  expect(window.compareFileTargets).toEqual([]);
  expect(document.getElementById('compareStatus').textContent).toBe('bad SEG-Y file');
});

test('import B source failure preserves existing selections and render', async () => {
  setupCompareImportDom();
  const latestRender = { panels: [{ label: 'existing compare render' }] };
  const existingTargets = [
    {
      fileId: 'active-file',
      displayName: 'active.sgy',
      originalName: 'active.sgy',
      key1Byte: 189,
      key2Byte: 193,
      isActive: true,
    },
    {
      fileId: 'existing-b',
      displayName: 'line-b.sgy',
      originalName: 'line-b.sgy',
      sourceSha256: 'abcdef1234567890',
      key1Byte: 189,
      key2Byte: 193,
      isActive: false,
    },
  ];
  window.__svCompare.setCompareFileTargetsForTest(existingTargets);
  window.__svCompare.setLatestCompareRenderForTest(latestRender);
  window.updateCompareSourceOptions();
  const sourceA = document.getElementById('compareSourceA');
  const sourceB = document.getElementById('compareSourceB');
  sourceA.value = window.__svCompare.compareSourceId('active-file', 'raw');
  sourceB.value = window.__svCompare.compareSourceId('existing-b', 'raw');
  const fetch = vi.fn(async () => errorText(409, 'Trace store already exists for a different source or key bytes'));
  vi.stubGlobal('fetch', fetch);

  const added = await window.__svCompare.importCompareBSourceFile(new File(['bad'], 'line-b.sgy'));

  expect(added).toBe(false);
  expect(window.compareFileTargets).toHaveLength(2);
  expect(window.compareFileTargets).toMatchObject(existingTargets);
  expect(sourceA.value).toBe(window.__svCompare.compareSourceId('active-file', 'raw'));
  expect(sourceB.value).toBe(window.__svCompare.compareSourceId('existing-b', 'raw'));
  expect(window.__svCompare.getLatestCompareRender()).toBe(latestRender);
  expect(document.getElementById('compareStatus').textContent).toBe(
    'Trace store already exists for a different source or key bytes',
  );
});

test('duplicate import result preserves existing B selection and render', async () => {
  setupCompareImportDom();
  const latestRender = { panels: [{ label: 'existing compare render' }] };
  const existingTargets = [
    {
      fileId: 'active-file',
      displayName: 'active.sgy',
      originalName: 'active.sgy',
      key1Byte: 189,
      key2Byte: 193,
      isActive: true,
    },
    {
      fileId: 'existing-b',
      displayName: 'line-b.sgy',
      originalName: 'line-b.sgy',
      sourceSha256: 'abcdef1234567890',
      key1Byte: 189,
      key2Byte: 193,
      isActive: false,
    },
  ];
  window.__svCompare.setCompareFileTargetsForTest(existingTargets);
  window.__svCompare.setLatestCompareRenderForTest(latestRender);
  window.updateCompareSourceOptions();
  const sourceA = document.getElementById('compareSourceA');
  const sourceB = document.getElementById('compareSourceB');
  sourceA.value = window.__svCompare.compareSourceId('active-file', 'raw');
  sourceB.value = window.__svCompare.compareSourceId('existing-b', 'raw');
  const targetsBeforeImport = window.compareFileTargets.map((target) => ({ ...target }));
  const fetch = vi.fn(async (url) => {
    if (url === '/compare/raw/import') {
      return okJson({
        file_id: 'imported-duplicate',
        original_name: 'line-b.sgy',
        source_sha256: 'abcdef1234567890',
        key1_byte: 189,
        key2_byte: 193,
      });
    }
    return okJson({ datasets: [] });
  });
  vi.stubGlobal('fetch', fetch);

  const added = await window.__svCompare.importCompareBSourceFile(new File(['sgy'], 'line-b.sgy'));

  expect(added).toBe(false);
  expect(window.compareFileTargets).toEqual(targetsBeforeImport);
  expect(sourceA.value).toBe(window.__svCompare.compareSourceId('active-file', 'raw'));
  expect(sourceB.value).toBe(window.__svCompare.compareSourceId('existing-b', 'raw'));
  expect([...sourceB.options].map((option) => option.value)).not.toContain(
    window.__svCompare.compareSourceId('imported-duplicate', 'raw'),
  );
  expect(window.__svCompare.getLatestCompareRender()).toBe(latestRender);
  expect(document.getElementById('compareStatus').textContent).toBe('Dataset is already added.');
});

test('import B source suppresses duplicate imports while request is in flight', async () => {
  setupCompareImportDom();
  let resolveImport;
  const importResponse = new Promise((resolve) => { resolveImport = resolve; });
  const fetch = vi.fn(async (url) => {
    if (url === '/compare/raw/import') return importResponse;
    return okJson({ datasets: [] });
  });
  vi.stubGlobal('fetch', fetch);

  const first = window.__svCompare.importCompareBSourceFile(new File(['sgy'], 'line-b.sgy'));
  const second = await window.__svCompare.importCompareBSourceFile(new File(['sgy'], 'line-c.sgy'));

  expect(second).toBe(false);
  expect(fetch.mock.calls.filter(([url]) => url === '/compare/raw/import')).toHaveLength(1);
  expect(document.getElementById('compareImportBSource').disabled).toBe(true);
  expect(document.getElementById('compareBSourceFile').disabled).toBe(true);
  expect(document.getElementById('compareAddDataset').disabled).toBe(true);

  resolveImport(okJson({ file_id: 'imported-file', original_name: 'line-b.sgy' }));
  expect(await first).toBe(true);
  expect(document.getElementById('compareImportBSource').disabled).toBe(false);
  expect(document.getElementById('compareBSourceFile').disabled).toBe(false);
});

test('import B source does not add target when active A dataset changes during import', async () => {
  setupCompareImportDom();
  const fetch = vi.fn(async (url) => {
    if (url === '/compare/raw/import') {
      vi.stubGlobal('currentFileId', 'new-active-file');
      vi.stubGlobal('currentFileName', 'new-active.sgy');
      return okJson({ file_id: 'imported-file', original_name: 'line-b.sgy' });
    }
    return okJson({ datasets: [] });
  });
  vi.stubGlobal('fetch', fetch);

  const added = await window.__svCompare.importCompareBSourceFile(new File(['sgy'], 'line-b.sgy'));

  expect(added).toBe(false);
  expect(fetch.mock.calls.filter(([url]) => url === '/compare/raw/import')).toHaveLength(1);
  expect(fetch.mock.calls.filter(([url]) => url === '/recent_datasets')).toHaveLength(1);
  expect(window.compareFileTargets).toEqual([]);
  expect(document.getElementById('compareSourceB').options).toHaveLength(0);
  expect(document.getElementById('compareStatus').textContent).toBe(
    'B source was imported, but the active A dataset changed. Add it from recent datasets.',
  );
});

test('compare dataset manager rejects mismatched key bytes', () => {
  const active = {
    fileId: 'active-file-id',
    displayName: 'active.sgy',
    originalName: 'active.sgy',
    key1Byte: 189,
    key2Byte: 193,
    isActive: true,
  };

  const result = window.__svCompare.addCompareDatasetTarget([], {
    fileId: 'mismatch-file-id',
    displayName: 'mismatch.sgy',
    originalName: 'mismatch.sgy',
    key1Byte: 17,
    key2Byte: 193,
  }, active);

  expect(result.added).toBe(false);
  expect(result.reason).toMatch(/key bytes/i);
  expect(result.targets).toMatchObject([
    { fileId: 'active-file-id', displayName: 'active.sgy', isActive: true },
  ]);
});

test('clear compare datasets keeps active target only', () => {
  const active = {
    fileId: 'active-file-id',
    displayName: 'active.sgy',
    originalName: 'active.sgy',
    key1Byte: 189,
    key2Byte: 193,
    isActive: true,
  };
  const targets = [
    active,
    {
      fileId: 'added-file-id',
      displayName: 'added.sgy',
      originalName: 'added.sgy',
      key1Byte: 189,
      key2Byte: 193,
      isActive: false,
    },
  ];

  expect(window.__svCompare.clearCompareDatasetTargets(targets, active)).toMatchObject([
    {
      fileId: 'active-file-id',
      displayName: 'active.sgy',
      originalName: 'active.sgy',
      key1Byte: 189,
      key2Byte: 193,
      isActive: true,
    },
  ]);
});
