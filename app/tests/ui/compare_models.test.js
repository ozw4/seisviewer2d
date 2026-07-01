import { beforeAll, expect, test } from 'vitest';

beforeAll(async () => {
  await import('../../static/viewer/compare/models.js');
});

function models() {
  return window.__svCompareModels;
}

test('normalizeCompareFileTarget requires a file id', () => {
  expect(models().normalizeCompareFileTarget(null)).toBeNull();
  expect(models().normalizeCompareFileTarget({ file_name: 'line.sgy' })).toBeNull();
  expect(models().normalizeCompareFileTarget({ file_id: '  ' })).toBeNull();
});

test('normalizeCompareFileTarget reads store/hash fields and normalizes key bytes', () => {
  expect(models().normalizeCompareFileTarget({
    file_id: 'file-1',
    file_name: 'line.sgy',
    key1_byte: '189',
    key2_byte: '193',
    original_name: 'line.sgy',
    store_name: 'stores/a.sgy',
    source_sha256: 'abcdef1234567890',
  })).toMatchObject({
    fileId: 'file-1',
    displayName: 'line.sgy',
    key1Byte: 189,
    key2Byte: 193,
    originalName: 'line.sgy',
    storeName: 'stores/a.sgy',
    sourceSha256: 'abcdef1234567890',
  });
});

test('compareTargetDatasetKey prefers source hash then store key then original name', () => {
  const base = {
    fileId: 'file-1',
    displayName: 'line.sgy',
    originalName: 'line.sgy',
    storeName: 'stores/a.sgy',
    key1Byte: 189,
    key2Byte: 193,
  };

  expect(models().compareTargetDatasetKey({
    ...base,
    sourceSha256: 'abcdef1234567890',
  })).toBe('sha256:abcdef1234567890|189|193');
  expect(models().compareTargetDatasetKey(base)).toBe('store:stores/a.sgy|189|193');
  expect(models().compareTargetDatasetKey({
    ...base,
    storeName: '',
  })).toBe('name:line.sgy|189|193');
});

test('compareSourceId raw value matches existing source select value format', () => {
  expect(models().compareSourceId('active-file', 'raw')).toBe('file:active-file:raw');
});

test('normalizeRecentDataset preserves store name, source hash, and display name identity', () => {
  expect(models().normalizeRecentDataset({
    original_name: 'line.sgy',
    display_name: 'Line A',
    store_name: 'stores/a.sgy',
    source_sha256: 'abcdef1234567890',
    key1_byte: '189',
    key2_byte: '193',
  })).toMatchObject({
    originalName: 'line.sgy',
    displayName: 'Line A',
    storeName: 'stores/a.sgy',
    sourceSha256: 'abcdef1234567890',
    key1Byte: 189,
    key2Byte: 193,
  });
});

test('compareRecentDatasetValue uses store identity and falls back to original name', () => {
  expect(models().compareRecentDatasetValue({
    originalName: 'line.sgy',
    storeName: 'stores/a.sgy',
    key1Byte: 189,
    key2Byte: 193,
  })).toBe('store:stores%2Fa.sgy|189|193');
  expect(models().compareRecentDatasetValue({
    originalName: 'line.sgy',
    key1Byte: 189,
    key2Byte: 193,
  })).toBe('name:line.sgy|189|193');
});
