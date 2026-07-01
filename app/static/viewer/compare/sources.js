(function () {
  'use strict';

  const models = window.__svCompareModels;
  if (!models) throw new Error('compare/models.js must be loaded before compare/sources.js');

  const {
    compareSourceId,
    normalizeCompareFileTarget,
  } = models;

  for (const helper of [compareSourceId, normalizeCompareFileTarget]) {
    if (typeof helper !== 'function') {
      throw new Error('compare/models.js must be loaded before compare/sources.js');
    }
  }

  function activeCompareFileTarget(targets) {
    const normalizedTargets = (Array.isArray(targets) ? targets : [])
      .map(normalizeCompareFileTarget)
      .filter(Boolean);
    return normalizedTargets.find((target) => target.isActive) || normalizedTargets[0] || null;
  }

  function compareTargetLabelName(target, displayNameCounts, displayNameSeen) {
    const displayName = target.displayName;
    if ((displayNameCounts.get(displayName) || 0) <= 1) return displayName;
    const seen = displayNameSeen.get(displayName) || 0;
    displayNameSeen.set(displayName, seen + 1);
    if (seen === 0) return displayName;
    const suffix = target.sourceSha256
      ? target.sourceSha256.slice(0, 8)
      : String(target.storeName || target.fileId || '').slice(0, 12);
    return suffix ? `${displayName} [${suffix}]` : displayName;
  }

  function rawCompareSource(target, labelName = target.displayName) {
    const sourceId = compareSourceId(target.fileId, 'raw');
    return {
      role: null,
      id: 'raw',
      sourceId,
      fileId: target.fileId,
      fileName: target.displayName,
      key1Byte: target.key1Byte,
      key2Byte: target.key2Byte,
      layerId: 'raw',
      label: `${labelName} / raw`,
      pipelineKey: null,
      tapLabel: null,
      domain: 'amplitude',
      available: true,
    };
  }

  function tapCompareSource(target, tapLabel, options = {}) {
    const sourceId = compareSourceId(target.fileId, 'tap', tapLabel);
    return {
      role: null,
      id: tapLabel,
      sourceId,
      fileId: target.fileId,
      fileName: target.displayName,
      key1Byte: target.key1Byte,
      key2Byte: target.key2Byte,
      layerId: tapLabel,
      label: `${target.displayName} / ${tapLabel}`,
      pipelineKey: options.latestPipelineKey || null,
      tapLabel,
      domain: resolveSourceDomain(tapLabel, options.latestTapData),
      available: !!options.latestPipelineKey,
    };
  }

  function normalizeCatalogOptions(input, legacyOptions = {}) {
    if (Array.isArray(input)) {
      return {
        ...legacyOptions,
        targets: input,
        layerOptions: legacyOptions.layerOptions || legacyOptions.layerValues,
      };
    }
    const options = input && typeof input === 'object' ? input : {};
    return {
      ...options,
      layerOptions: options.layerOptions || options.layerValues,
    };
  }

  function buildCompareSourceCatalog(input, legacyOptions = {}) {
    const options = normalizeCatalogOptions(input, legacyOptions);
    const catalog = [];
    const seen = new Set();
    const layerValues = Array.isArray(options.layerOptions) ? options.layerOptions : ['raw'];
    const normalizedTargets = (Array.isArray(options.targets) ? options.targets : [])
      .map(normalizeCompareFileTarget)
      .filter(Boolean);
    const activeTarget = activeCompareFileTarget(normalizedTargets);
    const activeFileId = String(options.activeFileId || activeTarget?.fileId || '').trim();
    const displayNameCounts = new Map();
    for (const target of normalizedTargets) {
      displayNameCounts.set(target.displayName, (displayNameCounts.get(target.displayName) || 0) + 1);
    }
    const displayNameSeen = new Map();
    for (const target of normalizedTargets) {
      if (!target || seen.has(target.fileId)) continue;
      seen.add(target.fileId);
      const labelName = compareTargetLabelName(target, displayNameCounts, displayNameSeen);
      catalog.push(rawCompareSource(target, labelName));
      if (target.fileId !== activeFileId) continue;
      for (const layer of layerValues) {
        const tapLabel = String(layer || '').trim();
        if (!tapLabel || tapLabel === 'raw') continue;
        catalog.push(tapCompareSource(target, tapLabel, options));
      }
    }
    return catalog;
  }

  function resolveSourceDomain(sourceId, tapDataByLabel = window.latestTapData) {
    if (!sourceId || sourceId === 'raw') return 'amplitude';
    const tapData = tapDataByLabel && tapDataByLabel[sourceId];
    if (tapData && typeof tapData === 'object') {
      const meta = tapData.meta;
      if (meta && typeof meta.domain === 'string') return meta.domain;
      if (Object.prototype.hasOwnProperty.call(tapData, 'prob')) return 'probability';
    }
    const lowered = String(sourceId).toLowerCase();
    if (lowered.includes('fbpick') || lowered.includes('prob')) return 'probability';
    return 'amplitude';
  }

  function resolveCompareSource(catalog, selectedValue) {
    const sources = Array.isArray(catalog) ? catalog : [];
    const activeRaw = sources.find((source) => source.layerId === 'raw') || sources[0] || null;
    if (!activeRaw) return null;
    const value = selectedValue === 'raw' ? activeRaw.sourceId : (selectedValue || activeRaw.sourceId);
    return sources.find((entry) => entry.sourceId === value) || activeRaw;
  }

  function sourcePairKey(sources) {
    const sourceA = sources?.a || {};
    const sourceB = sources?.b || {};
    return [
      sourceA.fileId || '',
      sourceA.layerId || sourceA.id || '',
      sourceA.pipelineKey || '',
      sourceA.tapLabel || '',
      sourceB.fileId || '',
      sourceB.layerId || sourceB.id || '',
      sourceB.pipelineKey || '',
      sourceB.tapLabel || '',
    ].join('|');
  }

  function isRawCompareSource(source) {
    return (source?.layerId || source?.id || '') === 'raw';
  }

  function resolveCompareNormalizationFileId(source, referenceSource, fallbackFileId = null) {
    if (!isRawCompareSource(source) || !isRawCompareSource(referenceSource)) return null;
    const fileId = String(source?.fileId || referenceSource?.fileId || fallbackFileId || '').trim();
    return fileId || null;
  }

  function shouldValidateRawCompareSources(sources) {
    return isRawCompareSource(sources?.a)
      && isRawCompareSource(sources?.b)
      && !!sources.a.fileId
      && !!sources.b.fileId
      && sources.a.fileId !== sources.b.fileId;
  }

  function rawCompareValidationKey(sources, key1Byte, key2Byte) {
    return [
      sources?.a?.fileId,
      sources?.b?.fileId,
      key1Byte,
      key2Byte,
    ].map((value) => encodeURIComponent(String(value ?? ''))).join('|');
  }

  window.__svCompareSources = Object.freeze({
    activeCompareFileTarget,
    compareTargetLabelName,
    rawCompareSource,
    tapCompareSource,
    buildCompareSourceCatalog,
    resolveSourceDomain,
    resolveCompareSource,
    sourcePairKey,
    isRawCompareSource,
    resolveCompareNormalizationFileId,
    shouldValidateRawCompareSources,
    rawCompareValidationKey,
  });
})();
