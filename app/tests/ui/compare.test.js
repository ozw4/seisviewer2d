import { afterEach, beforeAll, beforeEach, expect, test, vi } from 'vitest';

beforeAll(async () => {
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
  window.__svCompare.clearRawCompareValidationCache();
  delete window.buildWindowRequestArtifacts;
  vi.unstubAllGlobals();
});

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
    key1Byte: 17,
    key2Byte: 193,
  });

  const selected = window.__svCompare.resolveCompareRecentDataset(datasets, selectedValue);

  expect(selected).toMatchObject({
    originalName: 'line.sgy',
    key1Byte: 17,
    key2Byte: 193,
  });
  expect(selectedValue).not.toEqual(window.__svCompare.compareRecentDatasetValue({
    originalName: 'line.sgy',
    key1Byte: 189,
    key2Byte: 193,
  }));
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

  expect(window.__svCompare.clearCompareDatasetTargets(targets, active)).toEqual([
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
