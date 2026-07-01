(function () {
  'use strict';

  function compareSourceId(fileId, layerId, tapLabel = null) {
    const encodedFileId = encodeURIComponent(String(fileId || ''));
    if (layerId === 'raw') return `file:${encodedFileId}:raw`;
    return `file:${encodedFileId}:tap:${encodeURIComponent(String(tapLabel || layerId || ''))}`;
  }

  function normalizeCompareFileTarget(candidate) {
    if (!candidate || typeof candidate !== 'object') return null;
    const fileId = String(candidate.fileId ?? candidate.file_id ?? '').trim();
    if (!fileId) return null;
    const displayName = String(
      candidate.displayName ?? candidate.fileName ?? candidate.file_name ?? candidate.name ?? fileId,
    ).trim() || fileId;
    const key1Byte = Number(candidate.key1Byte ?? candidate.key1_byte);
    const key2Byte = Number(candidate.key2Byte ?? candidate.key2_byte);
    return {
      fileId,
      displayName,
      key1Byte: Number.isFinite(key1Byte) ? key1Byte : null,
      key2Byte: Number.isFinite(key2Byte) ? key2Byte : null,
      isActive: candidate.isActive === true,
      originalName: String(candidate.originalName ?? candidate.original_name ?? '').trim(),
      storeName: String(candidate.storeName ?? candidate.store_name ?? '').trim(),
      sourceSha256: String(candidate.sourceSha256 ?? candidate.source_sha256 ?? '').trim(),
    };
  }

  function compareTargetDatasetKey(target) {
    const normalized = normalizeCompareFileTarget(target);
    if (!normalized) return '';
    const keyBytes = `${normalized.key1Byte ?? ''}|${normalized.key2Byte ?? ''}`;
    if (normalized.sourceSha256) return `sha256:${normalized.sourceSha256}|${keyBytes}`;
    if (normalized.storeName) return `store:${normalized.storeName}|${keyBytes}`;
    const name = String(normalized.originalName || normalized.displayName || '').trim();
    return `name:${name}|${keyBytes}`;
  }

  function compareTargetIdentity(target) {
    const normalized = normalizeCompareFileTarget(target);
    if (!normalized) return '';
    return `${normalized.fileId}|${normalized.key1Byte ?? ''}|${normalized.key2Byte ?? ''}`;
  }

  function normalizeRecentDataset(candidate) {
    if (!candidate || typeof candidate !== 'object') return null;
    const originalName = String(candidate.original_name ?? candidate.originalName ?? '').trim();
    if (!originalName) return null;
    const displayName = String(candidate.display_name ?? candidate.displayName ?? originalName).trim()
      || originalName;
    const storeName = String(candidate.store_name ?? candidate.storeName ?? '').trim();
    const sourceSha256 = String(candidate.source_sha256 ?? candidate.sourceSha256 ?? '').trim();
    const key1Byte = Number(candidate.key1_byte ?? candidate.key1Byte);
    const key2Byte = Number(candidate.key2_byte ?? candidate.key2Byte);
    if (!Number.isFinite(key1Byte) || !Number.isFinite(key2Byte)) return null;
    return {
      originalName,
      displayName,
      storeName,
      sourceSha256,
      key1Byte,
      key2Byte,
    };
  }

  function compareRecentDatasetValue(candidate) {
    const dataset = normalizeRecentDataset(candidate);
    if (!dataset) return '';
    const prefix = dataset.storeName
      ? `store:${encodeURIComponent(dataset.storeName)}`
      : `name:${encodeURIComponent(dataset.originalName)}`;
    return [
      prefix,
      dataset.key1Byte,
      dataset.key2Byte,
    ].join('|');
  }

  window.__svCompareModels = Object.freeze({
    compareSourceId,
    normalizeCompareFileTarget,
    compareTargetDatasetKey,
    compareTargetIdentity,
    normalizeRecentDataset,
    compareRecentDatasetValue,
  });
})();
