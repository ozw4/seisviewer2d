import { beforeAll, expect, test } from 'vitest';

beforeAll(async () => {
  await import('../../static/viewer/window_fetch.js');
});

function baseWindowRequest(overrides = {}) {
  return {
    fileId: 'file-b',
    key1Val: 101,
    key1Byte: 189,
    key2Byte: 193,
    windowInfo: { x0: 0, x1: 10, y0: 0, y1: 20 },
    stepX: 1,
    stepY: 2,
    requestedLayer: 'raw',
    effectiveLayer: 'raw',
    pipelineKey: null,
    tapLabel: null,
    referencePipelineKey: null,
    referenceTapLabel: null,
    scaling: 'amax',
    transpose: '1',
    mode: 'heatmap',
    purpose: 'compare',
    ...overrides,
  };
}

test('buildWindowRequestArtifacts includes normalization_file_id query param and metadata', () => {
  const artifacts = window.__svWindowFetch.buildWindowRequestArtifacts(baseWindowRequest({
    normalizationFileId: 'file-a',
  }));

  expect(artifacts.params.get('file_id')).toBe('file-b');
  expect(artifacts.params.get('normalization_file_id')).toBe('file-a');
  expect(artifacts.requestContext.normalizationFileId).toBe('file-a');
  expect(artifacts.payloadMeta.normalizationFileId).toBe('file-a');
  expect(artifacts.cacheKey).toContain('norm=file-a');
});

test('buildWindowCacheKey distinguishes normalizationFileId', () => {
  const base = {
    fileId: 'file-b',
    key1: 101,
    key1Byte: 189,
    key2Byte: 193,
    x0: 0,
    x1: 10,
    y0: 0,
    y1: 20,
    stepX: 1,
    stepY: 2,
    requestedLayer: 'raw',
    effectiveLayer: 'raw',
    pipelineKey: null,
    tapLabel: null,
    referencePipelineKey: null,
    referenceTapLabel: null,
    scaling: 'amax',
    transpose: '1',
    mode: 'heatmap',
    purpose: 'compare',
    lmoKey: 'lmo:off',
  };

  const withA = window.__svWindowFetch.buildWindowCacheKey({
    ...base,
    normalizationFileId: 'file-a',
  });
  const withB = window.__svWindowFetch.buildWindowCacheKey({
    ...base,
    normalizationFileId: 'file-b',
  });

  expect(withA).not.toBe(withB);
  expect(withA).toContain('norm=file-a');
  expect(withB).toContain('norm=file-b');
});
