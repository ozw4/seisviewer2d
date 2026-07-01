import { beforeAll, expect, test } from 'vitest';

beforeAll(async () => {
  await import('../../static/viewer/compare/models.js');
  await import('../../static/viewer/compare/sources.js');
});

function sources() {
  return window.__svCompareSources;
}

test('active target resolves from first target or explicit active target', () => {
  expect(sources().activeCompareFileTarget([
    { fileId: 'first-file', displayName: 'first.sgy' },
    { fileId: 'second-file', displayName: 'second.sgy' },
  ])).toMatchObject({ fileId: 'first-file' });

  expect(sources().activeCompareFileTarget([
    { fileId: 'first-file', displayName: 'first.sgy' },
    { fileId: 'second-file', displayName: 'second.sgy', isActive: true },
  ])).toMatchObject({ fileId: 'second-file', isActive: true });
});

test('catalog exposes raw only for datasets outside the active file', () => {
  const catalog = sources().buildCompareSourceCatalog({
    targets: [
      { fileId: 'active-file', displayName: 'active.sgy', isActive: true },
      { fileId: 'other-file', displayName: 'other.sgy', isActive: false },
    ],
    layerOptions: ['raw', 'fbpick_prob'],
    activeFileId: 'active-file',
    latestPipelineKey: 'pipeline-1',
    latestTapData: { fbpick_prob: { prob: true } },
  });

  expect(catalog.map((source) => source.sourceId)).toEqual([
    'file:active-file:raw',
    'file:active-file:tap:fbpick_prob',
    'file:other-file:raw',
  ]);
  expect(catalog.find((source) => source.fileId === 'other-file' && source.tapLabel)).toBeUndefined();
});

test('catalog exposes active raw source and available tap sources', () => {
  const catalog = sources().buildCompareSourceCatalog({
    targets: [{ fileId: 'active-file', displayName: 'active.sgy', key1Byte: 189, key2Byte: 193 }],
    layerOptions: ['raw', 'fbpick_prob', 'prediction'],
    latestPipelineKey: 'pipeline-1',
    latestTapData: {
      fbpick_prob: { prob: true },
      prediction: { meta: { domain: 'amplitude' } },
    },
  });

  expect(catalog).toMatchObject([
    { fileId: 'active-file', layerId: 'raw', domain: 'amplitude', available: true },
    { fileId: 'active-file', layerId: 'fbpick_prob', domain: 'probability', available: true },
    { fileId: 'active-file', layerId: 'prediction', domain: 'amplitude', available: true },
  ]);
});

test('duplicate display labels get hash, store, or file suffixes', () => {
  const catalog = sources().buildCompareSourceCatalog({
    targets: [
      { fileId: 'active-file', displayName: 'line.sgy', isActive: true },
      { fileId: 'hash-file', displayName: 'line.sgy', sourceSha256: 'abcdef1234567890' },
      { fileId: 'store-file', displayName: 'line.sgy', storeName: 'stores/line-b.sgy' },
      { fileId: 'plain-file-id', displayName: 'line.sgy' },
    ],
    layerOptions: ['raw'],
    activeFileId: 'active-file',
  });

  expect(catalog.map((source) => source.label)).toEqual([
    'line.sgy / raw',
    'line.sgy [abcdef12] / raw',
    'line.sgy [stores/line-] / raw',
    'line.sgy [plain-file-i] / raw',
  ]);
});

test('sourcePairKey always includes file id differences', () => {
  const firstPair = {
    a: { fileId: 'file-a', layerId: 'raw' },
    b: { fileId: 'file-b', layerId: 'raw' },
  };
  const secondPair = {
    a: { fileId: 'file-a', layerId: 'raw' },
    b: { fileId: 'file-c', layerId: 'raw' },
  };

  expect(sources().sourcePairKey(firstPair)).not.toBe(sources().sourcePairKey(secondPair));
});

test('only raw/raw pairs require raw compare validation', () => {
  expect(sources().shouldValidateRawCompareSources({
    a: { fileId: 'file-a', layerId: 'raw' },
    b: { fileId: 'file-b', layerId: 'raw' },
  })).toBe(true);
  expect(sources().shouldValidateRawCompareSources({
    a: { fileId: 'file-a', layerId: 'raw' },
    b: { fileId: 'file-b', layerId: 'fbpick_prob' },
  })).toBe(false);
  expect(sources().shouldValidateRawCompareSources({
    a: { fileId: 'file-a', layerId: 'raw' },
    b: { fileId: 'file-a', layerId: 'raw' },
  })).toBe(false);
});

test('A-reference normalization resolves to A file id for raw/raw sources', () => {
  const sourceA = { fileId: 'file-a', layerId: 'raw' };
  const sourceB = { fileId: 'file-b', layerId: 'raw' };

  expect(sources().resolveCompareNormalizationFileId(sourceA, sourceA)).toBe('file-a');
  expect(sources().resolveCompareNormalizationFileId(sourceB, sourceA)).toBe('file-a');
});

test('pipeline and reference sources do not reuse raw normalization file id', () => {
  const sourceA = { fileId: 'file-a', layerId: 'raw' };
  const pipelineSource = {
    fileId: 'file-a',
    layerId: 'fbpick_prob',
    pipelineKey: 'pipeline-1',
    tapLabel: 'fbpick_prob',
  };

  expect(sources().resolveCompareNormalizationFileId(pipelineSource, sourceA, 'file-a')).toBeNull();
  expect(sources().resolveCompareNormalizationFileId(sourceA, pipelineSource, 'file-a')).toBeNull();
});
