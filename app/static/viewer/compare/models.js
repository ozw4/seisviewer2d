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

  function resetCompareTargetsForActive(targets, activeTarget) {
    const active = normalizeCompareFileTarget(activeTarget);
    if (!active) return [];
    const activeKey = compareTargetIdentity(active);
    const current = Array.isArray(targets) ? targets : [];
    if (current.length === 0 || compareTargetIdentity(current[0]) !== activeKey) {
      return [{ ...active, isActive: true }];
    }
    return [{ ...active, isActive: true }, ...current.slice(1).map((target) => ({
      ...target,
      isActive: false,
    }))];
  }

  function clearCompareDatasetTargets(targets, activeTarget) {
    return resetCompareTargetsForActive([], activeTarget);
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

  function addCompareDatasetTarget(targets, candidate, activeTarget) {
    const active = normalizeCompareFileTarget(activeTarget);
    const nextTarget = normalizeCompareFileTarget(candidate);
    if (!active) return { targets: [], added: false, reason: 'Open a dataset before adding compare targets.' };
    if (!nextTarget) {
      return {
        targets: resetCompareTargetsForActive(targets, active),
        added: false,
        reason: 'Compare dataset could not be opened.',
      };
    }
    if (nextTarget.key1Byte !== active.key1Byte || nextTarget.key2Byte !== active.key2Byte) {
      return {
        targets: resetCompareTargetsForActive(targets, active),
        added: false,
        reason: 'Dataset key bytes do not match the active file.',
      };
    }
    const next = resetCompareTargetsForActive(targets, active);
    const nextDatasetKey = compareTargetDatasetKey(nextTarget);
    if (nextDatasetKey && next.some((target) => compareTargetDatasetKey(target) === nextDatasetKey)) {
      return { targets: next, added: false, reason: 'Dataset is already added.' };
    }
    return { targets: [...next, { ...nextTarget, isActive: false }], added: true, reason: '' };
  }

  function resolveCompareRecentDataset(datasets, selectedValue) {
    if (!selectedValue) return null;
    return (Array.isArray(datasets) ? datasets : [])
      .map(normalizeRecentDataset)
      .find((dataset) => dataset && compareRecentDatasetValue(dataset) === selectedValue) || null;
  }

  function datasetMatchesActiveKeys(dataset, activeTarget) {
    const active = normalizeCompareFileTarget(activeTarget);
    return !!active
      && Number(dataset?.key1Byte) === active.key1Byte
      && Number(dataset?.key2Byte) === active.key2Byte;
  }

  window.__svCompareModels = Object.freeze({
    compareSourceId,
    normalizeCompareFileTarget,
    compareTargetDatasetKey,
    compareTargetIdentity,
    resetCompareTargetsForActive,
    clearCompareDatasetTargets,
    normalizeRecentDataset,
    compareRecentDatasetValue,
    addCompareDatasetTarget,
    resolveCompareRecentDataset,
    datasetMatchesActiveKeys,
  });
})();
