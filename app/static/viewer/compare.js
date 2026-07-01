(function () {
  const compareModels = window.__svCompareModels || {};
  const compareSources = window.__svCompareSources || {};
  const compareData = window.__svCompareData || {};
  const compareRender = window.__svCompareRender || {};
  const {
    compareSourceId,
    normalizeCompareFileTarget,
    compareTargetDatasetKey,
    compareTargetIdentity,
    normalizeRecentDataset,
    compareRecentDatasetValue,
  } = compareModels;
  const {
    rawCompareSource,
    buildCompareSourceCatalog,
    resolveSourceDomain,
    resolveCompareSource,
    sourcePairKey,
    isRawCompareSource,
    resolveCompareNormalizationFileId,
    shouldValidateRawCompareSources,
    rawCompareValidationKey,
  } = compareSources;
  const {
    payloadShapeInfo,
    payloadInvScale,
    payloadHasComputeValues,
    canUseCachedComparePayload,
    sourceDomain,
    payloadToF32,
    sameShape,
    sameGrid,
    validateComparePair,
    subtractF32,
    compareHeatmapScale,
  } = compareData;
  const {
    axisSuffix,
    axisLayoutName,
    buildCompareLayout,
    buildCompareWiggleTraces,
    buildCompareHeatmapTrace,
    buildComparePanels: buildComparePanelModels,
    buildCompareRender: buildCompareRenderModel,
    buildCompareUnavailableFigure,
  } = compareRender;

  for (const helper of [
    compareSourceId,
    normalizeCompareFileTarget,
    compareTargetDatasetKey,
    compareTargetIdentity,
    normalizeRecentDataset,
    compareRecentDatasetValue,
  ]) {
    if (typeof helper !== 'function') {
      throw new Error('compare/models.js must be loaded before compare.js');
    }
  }
  for (const helper of [
    rawCompareSource,
    buildCompareSourceCatalog,
    resolveSourceDomain,
    resolveCompareSource,
    sourcePairKey,
    isRawCompareSource,
    resolveCompareNormalizationFileId,
    shouldValidateRawCompareSources,
    rawCompareValidationKey,
  ]) {
    if (typeof helper !== 'function') {
      throw new Error('compare/sources.js must be loaded before compare.js');
    }
  }
  for (const helper of [
    payloadShapeInfo,
    payloadInvScale,
    payloadHasComputeValues,
    canUseCachedComparePayload,
    sourceDomain,
    payloadToF32,
    sameShape,
    sameGrid,
    validateComparePair,
    subtractF32,
    compareHeatmapScale,
  ]) {
    if (typeof helper !== 'function') {
      throw new Error('compare/data.js must be loaded before compare.js');
    }
  }
  for (const helper of [
    axisSuffix,
    axisLayoutName,
    buildCompareLayout,
    buildCompareWiggleTraces,
    buildCompareHeatmapTrace,
    buildComparePanelModels,
    buildCompareRenderModel,
    buildCompareUnavailableFigure,
  ]) {
    if (typeof helper !== 'function') {
      throw new Error('compare/render.js must be loaded before compare.js');
    }
  }

  const COMPARE_CACHE_PURPOSE = 'compare';
  const COMPARE_RENDER_SLOT = 'compare-window';
  const SECTION_RENDER_SLOT = 'section-window';
  let latestCompareRender = null;
  let compareSyncing = false;
  let compareRecentDatasets = [];
  let compareFileTargets = [];
  let compareActiveTargetKey = '';
  let compareActiveSyncWrapped = false;
  let compareActiveSyncQueued = false;
  let rawCompareValidationCache = new Map();
  let compareImportInFlight = false;

  class CompareFetchError extends Error {
    constructor(source, status, detail) {
      const label = source && source.role === 'A' ? 'A' : 'B';
      const suffix = detail ? `: ${detail}` : '';
      super(`${label} source fetch failed (${status})${suffix}`);
      this.name = 'CompareFetchError';
      this.source = source;
      this.status = status;
      this.detail = detail || '';
    }
  }

  function getCompareNodes() {
    return {
      toggle: document.getElementById('compareModeToggle'),
      sourceA: document.getElementById('compareSourceA'),
      sourceB: document.getElementById('compareSourceB'),
      showDiff: document.getElementById('compareShowDiff'),
      datasetPicker: document.getElementById('compareDatasetPicker'),
      addDataset: document.getElementById('compareAddDataset'),
      importBSource: document.getElementById('compareImportBSource'),
      bSourceFile: document.getElementById('compareBSourceFile'),
      clearDatasets: document.getElementById('compareClearDatasets'),
      datasetList: document.getElementById('compareDatasetList'),
      status: document.getElementById('compareStatus'),
    };
  }

  function setCompareStatus(message) {
    const { status } = getCompareNodes();
    if (!status) return;
    const text = String(message || '').trim();
    status.textContent = text;
    status.hidden = text.length === 0;
  }

  function isCompareRequestCurrent(requestId) {
    return isCurrentRenderRequest(COMPARE_RENDER_SLOT, requestId);
  }

  function markStaleCompareRequest(requestId) {
    markStaleRenderDropped(COMPARE_RENDER_SLOT, requestId);
  }

  function setCompareStatusIfCurrent(requestId, message) {
    if (!isCompareRequestCurrent(requestId)) return false;
    setCompareStatus(message);
    return true;
  }

  function isCompareModeEnabled() {
    return !!document.getElementById('compareModeToggle')?.checked;
  }

  function currentCompareLmoKey() {
    return typeof window.currentLmoKey === 'function'
      ? window.currentLmoKey()
      : 'lmo:off';
  }

  function isCompareLmoCurrent(lmoKey) {
    return lmoKey === currentCompareLmoKey();
  }

  function compareShowDiffEnabled() {
    return !!document.getElementById('compareShowDiff')?.checked;
  }

  function optionValues(select) {
    if (!select) return ['raw'];
    const values = Array.from(select.options || [])
      .map((opt) => opt.value || opt.textContent || '')
      .filter(Boolean);
    return values.length ? values : ['raw'];
  }

  function getLayerSourceOptions() {
    const layerSelect = document.getElementById('layerSelect');
    const values = optionValues(layerSelect);
    const unique = [];
    for (const value of ['raw', ...values]) {
      if (!unique.includes(value)) unique.push(value);
    }
    return unique;
  }

  function activeCompareFileTarget() {
    return normalizeCompareFileTarget({
      fileId: window.currentFileId || '',
      displayName: window.currentFileName || window.currentFileId || '',
      key1Byte: window.currentKey1Byte,
      key2Byte: window.currentKey2Byte,
      isActive: true,
      originalName: window.currentFileName || '',
      storeName: '',
      sourceSha256: '',
    });
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

  function resetCompareTargetsForActiveFile() {
    clearRawCompareValidationCache();
    compareActiveTargetKey = '';
    syncCompareTargetsWithActive();
    updateCompareSourceOptions();
  }

  function clearRawCompareValidationCache() {
    rawCompareValidationCache.clear();
  }

  function setCompareImportInFlight(inFlight) {
    compareImportInFlight = inFlight === true;
    const { importBSource, bSourceFile } = getCompareNodes();
    if (importBSource) importBSource.disabled = compareImportInFlight;
    if (bSourceFile) bSourceFile.disabled = compareImportInFlight;
    renderCompareDatasetPicker();
  }

  function wrapActiveFileTargetSync() {
    if (compareActiveSyncWrapped) return;
    const state = window.SeisViewerState;
    if (!state || typeof state.syncActiveFileTarget !== 'function') return;
    const originalSync = state.syncActiveFileTarget;
    state.syncActiveFileTarget = function syncActiveFileTargetWithCompareReset() {
      const result = originalSync.apply(this, arguments);
      resetCompareTargetsForActiveFile();
      return result;
    };
    compareActiveSyncWrapped = true;
  }

  function ensureActiveFileTargetSyncWrapped() {
    wrapActiveFileTargetSync();
    if (compareActiveSyncWrapped) return;
    if (typeof window.setTimeout === 'function') {
      window.setTimeout(wrapActiveFileTargetSync, 0);
    }
    if (compareActiveSyncQueued || typeof window.whenViewerBootstrapReady !== 'function') return;
    compareActiveSyncQueued = true;
    window.whenViewerBootstrapReady(() => {
      wrapActiveFileTargetSync();
      if (!compareActiveSyncWrapped && typeof window.setTimeout === 'function') {
        window.setTimeout(wrapActiveFileTargetSync, 0);
      }
    });
  }

  function syncCompareTargetsWithActive() {
    const active = activeCompareFileTarget();
    if (!active) {
      compareFileTargets = [];
      compareActiveTargetKey = '';
      window.compareFileTargets = compareFileTargets;
      return compareFileTargets;
    }
    const activeKey = compareTargetIdentity(active);
    if (activeKey !== compareActiveTargetKey) {
      compareFileTargets = [{ ...active, isActive: true }];
      compareActiveTargetKey = activeKey;
    } else {
      compareFileTargets = resetCompareTargetsForActive(compareFileTargets, active);
    }
    window.compareFileTargets = compareFileTargets;
    return compareFileTargets;
  }

  function currentCompareSourceCatalog() {
    return buildCompareSourceCatalog({
      targets: syncCompareTargetsWithActive(),
      layerOptions: getLayerSourceOptions(),
      activeFileId: window.currentFileId || '',
      latestPipelineKey: window.latestPipelineKey || null,
      latestTapData: window.latestTapData || {},
    });
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

  function renderCompareDatasetPicker() {
    const { datasetPicker, addDataset } = getCompareNodes();
    if (!datasetPicker) return;
    const active = activeCompareFileTarget();
    const previous = datasetPicker.value;
    datasetPicker.innerHTML = '';
    const datasets = compareRecentDatasets
      .map(normalizeRecentDataset)
      .filter(Boolean);
    for (const dataset of datasets) {
      const value = compareRecentDatasetValue(dataset);
      const option = new Option(dataset.originalName, value);
      option.dataset.key1Byte = String(dataset.key1Byte);
      option.dataset.key2Byte = String(dataset.key2Byte);
      option.disabled = !datasetMatchesActiveKeys(dataset, active);
      datasetPicker.appendChild(option);
    }
    if (previous && Array.from(datasetPicker.options).some((option) => option.value === previous)) {
      datasetPicker.value = previous;
    }
    if (addDataset) addDataset.disabled = compareImportInFlight || !active || datasetPicker.options.length === 0;
  }

  function renderCompareDatasetList() {
    const { datasetList } = getCompareNodes();
    if (!datasetList) return;
    syncCompareTargetsWithActive();
    datasetList.innerHTML = '';
    for (const target of compareFileTargets) {
      const item = document.createElement('div');
      item.textContent = target.displayName;
      datasetList.appendChild(item);
    }
  }

  async function loadCompareRecentDatasets() {
    const { datasetPicker } = getCompareNodes();
    if (!datasetPicker || typeof fetch !== 'function') return;
    try {
      const response = await fetch('/recent_datasets');
      if (!response.ok) throw new Error(`Recent datasets request failed (${response.status})`);
      const payload = await response.json();
      compareRecentDatasets = Array.isArray(payload?.datasets) ? payload.datasets : [];
      renderCompareDatasetPicker();
    } catch (err) {
      compareRecentDatasets = [];
      renderCompareDatasetPicker();
      setCompareStatus(err instanceof Error ? err.message : String(err));
    }
  }

  function fillSourceSelect(select, sources, preferred, fallback) {
    if (!select) return;
    const previous = preferred || select.value || '';
    select.innerHTML = '';
    for (const source of sources) {
      select.appendChild(new Option(source.label, source.sourceId));
    }
    const sourceIds = sources.map((source) => source.sourceId);
    const activeRaw = sources.find((source) => source.layerId === 'raw')?.sourceId || '';
    const normalizedPrevious = previous === 'raw' ? activeRaw : previous;
    const normalizedFallback = fallback === 'raw' ? activeRaw : fallback;
    const target = sourceIds.includes(normalizedPrevious)
      ? normalizedPrevious
      : (sourceIds.includes(normalizedFallback) ? normalizedFallback : sourceIds[0]);
    select.value = target || '';
  }

  function updateCompareSourceOptions() {
    const { sourceA, sourceB } = getCompareNodes();
    const sources = currentCompareSourceCatalog();
    const rawSourceId = sources.find((source) => source.layerId === 'raw')?.sourceId || '';
    const firstTap = sources.find((source) => source.layerId !== 'raw')?.sourceId || rawSourceId;
    fillSourceSelect(sourceA, sources, sourceA?.value || rawSourceId, rawSourceId);
    fillSourceSelect(sourceB, sources, sourceB?.value || firstTap, firstTap);
    renderCompareDatasetPicker();
    renderCompareDatasetList();
  }

  function getCompareSources() {
    updateCompareSourceOptions();
    const { sourceA, sourceB } = getCompareNodes();
    const catalog = currentCompareSourceCatalog();
    const activeRaw = catalog.find((source) => source.layerId === 'raw') || rawCompareSource({
      fileId: window.currentFileId || '',
      displayName: window.currentFileName || window.currentFileId || 'raw',
      key1Byte: window.currentKey1Byte ?? null,
      key2Byte: window.currentKey2Byte ?? null,
      isActive: true,
    });
    return {
      a: { ...(resolveCompareSource(catalog, sourceA?.value) || activeRaw), role: 'A' },
      b: { ...(resolveCompareSource(catalog, sourceB?.value) || activeRaw), role: 'B' },
    };
  }

  function currentCompareKey1() {
    const slider = document.getElementById('key1_slider');
    const idx = slider ? parseInt(slider.value, 10) : 0;
    return Array.isArray(key1Values) ? key1Values[idx] : undefined;
  }

  function canAttemptDiff(sources) {
    return sources.a.domain === sources.b.domain;
  }

  async function readCompareResponseDetail(response) {
    try {
      const contentType = response.headers?.get?.('content-type') || '';
      if (contentType.includes('application/json')) {
        const json = await response.json();
        return typeof json?.detail === 'string' ? json.detail : '';
      }
      return await response.text();
    } catch (_) {
      return '';
    }
  }

  async function validateRawCompareSources(sources, signal) {
    if (!shouldValidateRawCompareSources(sources)) return { ok: true, reason: '', message: '' };

    const key1ByteA = Number(sources.a.key1Byte ?? currentKey1Byte);
    const key2ByteA = Number(sources.a.key2Byte ?? currentKey2Byte);
    const key1ByteB = Number(sources.b.key1Byte ?? currentKey1Byte);
    const key2ByteB = Number(sources.b.key2Byte ?? currentKey2Byte);
    if (![key1ByteA, key2ByteA, key1ByteB, key2ByteB].every(Number.isFinite)) {
      return { ok: false, reason: 'key_bytes', message: 'A-B unavailable: source key bytes are missing.' };
    }
    if (key1ByteA !== key1ByteB || key2ByteA !== key2ByteB) {
      return { ok: false, reason: 'key_bytes', message: 'A-B unavailable: source key bytes are different.' };
    }

    const cacheKey = rawCompareValidationKey(sources, key1ByteA, key2ByteA);
    if (rawCompareValidationCache.has(cacheKey)) {
      return rawCompareValidationCache.get(cacheKey);
    }

    const params = new URLSearchParams({
      file_id_a: String(sources.a.fileId),
      file_id_b: String(sources.b.fileId),
      key1_byte: String(key1ByteA),
      key2_byte: String(key2ByteA),
    });
    const response = await fetch(`/compare/raw/validate?${params.toString()}`, { signal });
    if (!response.ok) {
      const detail = await readCompareResponseDetail(response);
      throw new Error(detail || `A-B validation failed (${response.status})`);
    }
    const payload = await response.json();
    const result = {
      ok: payload?.ok === true,
      reason: String(payload?.reason || ''),
      message: String(payload?.message || ''),
    };
    if (!result.ok && !result.message) {
      result.message = 'A-B unavailable: raw source grids are different.';
    }
    rawCompareValidationCache.set(cacheKey, result);
    return result;
  }

  async function ensureRawCompareReferenceBaseline(sources, signal) {
    if (!isRawCompareSource(sources?.a) || !isRawCompareSource(sources?.b)) return true;
    if (!sources.a.fileId) return true;

    const key1Byte = Number(sources.a.key1Byte ?? currentKey1Byte);
    const key2Byte = Number(sources.a.key2Byte ?? currentKey2Byte);
    if (!Number.isFinite(key1Byte) || !Number.isFinite(key2Byte)) {
      throw new Error('A-B unavailable: source key bytes are missing.');
    }

    const params = new URLSearchParams({
      file_id: String(sources.a.fileId),
      key1_byte: String(key1Byte),
      key2_byte: String(key2Byte),
    });
    const response = await fetch(`/get_section_meta?${params.toString()}`, { signal });
    if (!response.ok) {
      const detail = await readCompareResponseDetail(response);
      throw new Error(detail || `A normalization baseline is unavailable (${response.status})`);
    }
    return true;
  }

  function visibleComparePanelCount(sources) {
    return compareShowDiffEnabled() && canAttemptDiff(sources) ? 3 : 2;
  }

  function decideCompareWindowMode(windowInfo, plotDiv, sources) {
    const panelCount = visibleComparePanelCount(sources);
    const fullWidth = plotDiv.clientWidth || plotDiv.offsetWidth || 1;
    const widthPx = Math.max(1, fullWidth / panelCount);
    const heightPx = plotDiv.clientHeight || plotDiv.offsetHeight || 1;
    const probabilityInvolved = sources.a.domain === 'probability' || sources.b.domain === 'probability';
    const wantWiggle = !probabilityInvolved && wantWiggleForWindow({
      tracesVisible: windowInfo.nTraces,
      samplesVisible: windowInfo.nSamples,
      widthPx,
    });
    if (wantWiggle) {
      return {
        mode: 'wiggle',
        stepX: 1,
        stepY: 1,
        panelCount,
        panelWidth: widthPx,
        probabilityInvolved,
      };
    }
    const steps = computeStepsForWindow({
      tracesVisible: windowInfo.nTraces,
      samplesVisible: windowInfo.nSamples,
      widthPx,
      heightPx,
    });
    return {
      mode: 'heatmap',
      stepX: steps.step_x,
      stepY: steps.step_y,
      panelCount,
      panelWidth: widthPx,
      probabilityInvolved,
    };
  }

  function buildCompareRequest(source, referenceSource, key1Val, windowInfo, decision) {
    const effectiveLayer = source.id === 'raw' ? 'raw' : source.id;
    const tapLabel = source.id === 'raw' ? null : source.tapLabel;
    const fileId = source.fileId || currentFileId;
    const referencePipelineKey = referenceSource?.id === 'raw'
      ? null
      : (referenceSource?.pipelineKey || null);
    const referenceTapLabel = referenceSource?.id === 'raw'
      ? null
      : (referenceSource?.tapLabel || null);
    const requestContext = {
      fileId,
      key1Val,
      key1Byte: source.key1Byte ?? currentKey1Byte,
      key2Byte: source.key2Byte ?? currentKey2Byte,
      windowInfo,
      stepX: decision.stepX,
      stepY: decision.stepY,
      requestedLayer: source.id,
      effectiveLayer,
      pipelineKey: source.pipelineKey,
      tapLabel,
      referencePipelineKey,
      referenceTapLabel,
      normalizationFileId: resolveCompareNormalizationFileId(source, referenceSource, fileId),
      scaling: currentScaling,
      transpose: '1',
      mode: decision.mode,
      purpose: COMPARE_CACHE_PURPOSE,
    };
    const artifacts = buildCompareWindowRequestArtifacts(requestContext);
    return { source, requestContext, ...artifacts };
  }

  function buildCompareWindowRequestArtifacts(requestContext) {
    const builder = typeof window.buildWindowRequestArtifacts === 'function'
      ? window.buildWindowRequestArtifacts
      : (typeof buildWindowRequestArtifacts === 'function' ? buildWindowRequestArtifacts : null);
    if (!builder) throw new Error('Window request builder is not available.');
    return builder(requestContext);
  }

  function selectedCompareRecentDataset() {
    const { datasetPicker } = getCompareNodes();
    if (!datasetPicker) return null;
    return resolveCompareRecentDataset(compareRecentDatasets, datasetPicker.value);
  }

  async function addSelectedCompareDataset() {
    const dataset = selectedCompareRecentDataset();
    const active = activeCompareFileTarget();
    if (!dataset) {
      setCompareStatus('Select a recent dataset to add.');
      return false;
    }
    if (!datasetMatchesActiveKeys(dataset, active)) {
      setCompareStatus('Dataset key bytes do not match the active file.');
      return false;
    }
    const duplicateKey = compareTargetDatasetKey({
      fileId: 'candidate',
      displayName: dataset.originalName,
      originalName: dataset.originalName,
      storeName: dataset.storeName,
      sourceSha256: dataset.sourceSha256,
      key1Byte: active.key1Byte,
      key2Byte: active.key2Byte,
    });
    if (duplicateKey && syncCompareTargetsWithActive().some((target) => compareTargetDatasetKey(target) === duplicateKey)) {
      setCompareStatus('Dataset is already added.');
      return false;
    }
    const formData = new FormData();
    if (dataset.storeName) formData.append('store_name', dataset.storeName);
    formData.append('original_name', dataset.originalName);
    formData.append('key1_byte', String(active.key1Byte));
    formData.append('key2_byte', String(active.key2Byte));
    try {
      const response = await fetch('/open_segy', { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`Open dataset failed (${response.status})`);
      const payload = await response.json();
      const result = addCompareDatasetTarget(compareFileTargets, {
        fileId: payload.file_id,
        displayName: dataset.originalName,
        key1Byte: active.key1Byte,
        key2Byte: active.key2Byte,
        originalName: dataset.originalName,
        storeName: dataset.storeName,
        sourceSha256: dataset.sourceSha256,
        isActive: false,
      }, active);
      compareFileTargets = result.targets;
      window.compareFileTargets = compareFileTargets;
      setCompareStatus(result.reason);
      updateCompareSourceOptions();
      onCompareControlChange();
      return result.added;
    } catch (err) {
      setCompareStatus(err instanceof Error ? err.message : String(err));
      return false;
    }
  }

  async function importCompareBSourceFile(file) {
    if (compareImportInFlight) return false;
    if (!file) return false;

    const active = activeCompareFileTarget();
    if (!active) {
      setCompareStatus('Open an A dataset before importing B source.');
      return false;
    }
    const startingActiveIdentity = compareTargetIdentity(active);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('key1_byte', String(active.key1Byte));
    formData.append('key2_byte', String(active.key2Byte));

    setCompareImportInFlight(true);
    try {
      const response = await fetch('/compare/raw/import', { method: 'POST', body: formData });
      if (!response.ok) {
        const detail = await readCompareResponseDetail(response);
        throw new Error(detail || `Import B source failed (${response.status})`);
      }
      const payload = await response.json();
      if (compareTargetIdentity(activeCompareFileTarget()) !== startingActiveIdentity) {
        setCompareStatus('B source was imported, but the active A dataset changed. Add it from recent datasets.');
        loadCompareRecentDatasets();
        return false;
      }

      const result = addCompareDatasetTarget(compareFileTargets, {
        fileId: payload.file_id,
        displayName: payload.display_name || payload.original_name || file.name,
        originalName: payload.original_name || file.name,
        storeName: payload.store_name || '',
        sourceSha256: payload.source_sha256 || '',
        key1Byte: payload.key1_byte ?? active.key1Byte,
        key2Byte: payload.key2_byte ?? active.key2Byte,
        isActive: false,
      }, active);
      if (!result.added) {
        setCompareStatus(result.reason);
        loadCompareRecentDatasets();
        return false;
      }
      compareFileTargets = result.targets;
      window.compareFileTargets = compareFileTargets;
      clearRawCompareValidationCache();
      updateCompareSourceOptions();
      const { sourceB } = getCompareNodes();
      const importedSourceId = compareSourceId(payload.file_id, 'raw');
      if (sourceB) sourceB.value = importedSourceId;
      loadCompareRecentDatasets();
      clearCompareDataState();
      if (isCompareModeEnabled()) {
        const renderPromise = fetchCompareAndPlot();
        if (renderPromise && typeof renderPromise.catch === 'function') {
          renderPromise.catch((err) => console.warn('Compare B source import render failed', err));
        }
      } else {
        onCompareControlChange();
      }
      setCompareStatus('B source imported.');
      return true;
    } catch (err) {
      setCompareStatus(err instanceof Error ? err.message : String(err));
      return false;
    } finally {
      setCompareImportInFlight(false);
    }
  }

  function clearCompareDatasets() {
    clearRawCompareValidationCache();
    compareFileTargets = clearCompareDatasetTargets(compareFileTargets, activeCompareFileTarget());
    window.compareFileTargets = compareFileTargets;
    setCompareStatus('');
    updateCompareSourceOptions();
    onCompareControlChange();
  }

  async function fetchComparePayload(request, signal, requestId) {
    if (!isCompareRequestCurrent(requestId)) {
      markStaleCompareRequest(requestId);
      return null;
    }
    if (!isCompareLmoCurrent(request.payloadMeta?.lmoKey)) return null;
    const cached = windowCacheGet(request.cacheKey);
    if (
      cached &&
      isCompareLmoCurrent(cached.lmoKey) &&
      canUseCachedComparePayload(cached, request.source)
    ) {
      if (!isCompareRequestCurrent(requestId)) {
        markStaleCompareRequest(requestId);
        return null;
      }
      return cached;
    }

    const res = await fetch(`/get_section_window_bin?${request.params.toString()}`, { signal });
    if (!res.ok) {
      let detail = '';
      try {
        const contentType = res.headers.get('content-type') || '';
        if (contentType.includes('application/json')) {
          const json = await res.json();
          detail = typeof json?.detail === 'string' ? json.detail : '';
        } else {
          detail = await res.text();
        }
      } catch (_) {
        detail = '';
      }
      throw new CompareFetchError(request.source, res.status, detail);
    }
    const buf = await res.arrayBuffer();
    if (!isCompareRequestCurrent(requestId)) {
      markStaleCompareRequest(requestId);
      return null;
    }
    if (!isCompareLmoCurrent(request.payloadMeta?.lmoKey)) return null;
    const payload = decodeWindowPayload(
      new Uint8Array(buf),
      request.payloadMeta,
      null,
      (shape) => console.warn('Unexpected compare window shape', shape),
    );
    if (!payload) return null;
    if (!isCompareRequestCurrent(requestId)) {
      markStaleCompareRequest(requestId);
      return null;
    }
    if (!isCompareLmoCurrent(payload.lmoKey)) return null;
    windowCacheSet(request.cacheKey, payload);
    return payload;
  }

  function currentCompareGain() {
    return parseFloat(document.getElementById('gain')?.value) || 1.0;
  }

  function buildComparePanelsForCurrentState(render) {
    return buildComparePanelModels({
      render,
      showDiff: compareShowDiffEnabled(),
    });
  }

  function buildCompareLayoutForCurrentState(render, panels, xRange, yRange) {
    return buildCompareLayout({
      render,
      panels,
      xRange,
      yRange,
      clickmode: clickModeForCurrentState(),
      dragmode: effectiveDragMode(),
      uiRevision: currentUiRevision(),
    });
  }

  function buildCompareWiggleTracesForCurrentState(panel, axisIndex, render) {
    return buildCompareWiggleTraces({
      panel,
      axisIndex,
      render,
      gain: currentCompareGain(),
    });
  }

  function buildCompareHeatmapTraceForCurrentState(panel, axisIndex, render) {
    const colormapName = document.getElementById('colormap')?.value || 'Greys';
    return buildCompareHeatmapTrace({
      panel,
      axisIndex,
      render,
      gain: currentCompareGain(),
      colormapName,
      reverse: !!document.getElementById('cmReverse')?.checked,
      colormaps: window.COLORMAPS,
    });
  }

  function renderCompareLatestView() {
    if (!isCompareModeEnabled()) return false;
    const render = latestCompareRender;
    if (!render) return false;
    if (
      Number.isInteger(render.__requestId) &&
      !isCompareRequestCurrent(render.__requestId)
    ) {
      markStaleCompareRequest(render.__requestId);
      return false;
    }
    const key1Val = currentCompareKey1();
    const sources = getCompareSources();
    if (render.key1 !== key1Val || sourcePairKey(render.sources) !== sourcePairKey(sources)) return false;
    if (render.scaling !== currentScaling) return false;
    if (!isCompareLmoCurrent(render.lmoKey)) return false;

    const plotDiv = document.getElementById('plot');
    if (!plotDiv) return false;
    const panels = buildComparePanelsForCurrentState(render);
    if (
      Number.isInteger(render.__requestId) &&
      !isCompareRequestCurrent(render.__requestId)
    ) {
      markStaleCompareRequest(render.__requestId);
      return false;
    }
    if (compareShowDiffEnabled() && !render.diffAvailable) {
      setCompareStatus(render.diffMessage || 'A-B unavailable.');
    } else {
      setCompareStatus('');
    }

    const xRange = savedXRange || null;
    const yRange = savedYRange || null;
    const traces = [];
    for (let i = 0; i < panels.length; i++) {
      if (render.mode === 'wiggle') traces.push(...buildCompareWiggleTracesForCurrentState(panels[i], i, render));
      else traces.push(buildCompareHeatmapTraceForCurrentState(panels[i], i, render));
    }
    const layout = buildCompareLayoutForCurrentState(render, panels, xRange, yRange);
    if (
      Number.isInteger(render.__requestId) &&
      !isCompareRequestCurrent(render.__requestId)
    ) {
      markStaleCompareRequest(render.__requestId);
      return false;
    }
    downsampleFactor = render.stepY || 1;
    renderedStart = render.x0;
    renderedEnd = render.x1;
    latestWindowRender = null;
    setGrid({ x0: render.x0, stepX: render.mode === 'wiggle' ? 1 : render.stepX, y0: render.y0, stepY: render.stepY });
    const startRender = () => withSuppressedRelayout(Plotly.react(plotDiv, traces, layout, {
      responsive: true,
      doubleClick: false,
      doubleClickDelay: 300,
    }));
    const promise = typeof window.queueViewerPlotlyRender === 'function'
      ? window.queueViewerPlotlyRender(plotDiv, render, startRender)
      : Promise.resolve(startRender()).then(() => true);
    return promise
      .then((rendered) => {
        if (!rendered) return false;
        if (
          Number.isInteger(render.__requestId) &&
          !isCompareRequestCurrent(render.__requestId)
        ) {
          markStaleCompareRequest(render.__requestId);
          return false;
        }
        plotDiv.__svPlotMode = `compare-${render.mode}`;
        plotDiv.__svComparePanelCount = panels.length;
        plotDiv.__svCompareMode = render.mode;
        if (typeof maybeResizePlot === 'function') maybeResizePlot(plotDiv, true);
        if (typeof window.scheduleViewerOverlaySync === 'function') {
          window.scheduleViewerOverlaySync('compare-render');
        }
        requestAnimationFrame(applyDragMode);
        if (typeof installPlotlyViewportHandlersOnce === 'function') installPlotlyViewportHandlersOnce();
        return true;
      })
      .catch((err) => {
        if (
          Number.isInteger(render.__requestId) &&
          isCompareRequestCurrent(render.__requestId)
        ) {
          console.warn('Compare Plotly render failed', err);
        }
        return false;
      });
  }

  function clearCompareDataState() {
    latestCompareRender = null;
    if (typeof latestSeismicData !== 'undefined') latestSeismicData = null;
    if (typeof latestWindowRender !== 'undefined') latestWindowRender = null;
  }

  function withCompareSuppressedRelayout(promiseLike) {
    if (typeof withSuppressedRelayout === 'function') {
      return withSuppressedRelayout(promiseLike);
    }
    return Promise.resolve(promiseLike);
  }

  async function renderCompareUnavailable(message, requestId = null) {
    const text = String(message || 'A-B unavailable.').trim() || 'A-B unavailable.';
    if (Number.isInteger(requestId) && !isCompareRequestCurrent(requestId)) {
      markStaleCompareRequest(requestId);
      return false;
    }

    const plotDiv = document.getElementById('plot');
    if (!plotDiv || !window.Plotly || typeof window.Plotly.react !== 'function') {
      clearCompareDataState();
      setCompareStatus(text);
      return false;
    }

    const figure = buildCompareUnavailableFigure(text);

    const startRender = () => {
      return withCompareSuppressedRelayout(
        window.Plotly.react(plotDiv, figure.data, figure.layout, figure.config),
      );
    };
    const renderData = Number.isInteger(requestId)
      ? { __requestSlot: COMPARE_RENDER_SLOT, __requestId: requestId }
      : null;

    try {
      const rendered = typeof window.queueViewerPlotlyRender === 'function'
        ? await window.queueViewerPlotlyRender(plotDiv, renderData, startRender)
        : await Promise.resolve(startRender()).then(() => true);
      if (!rendered) return false;
    } catch (err) {
      if (
        Number.isInteger(requestId) &&
        !isCompareRequestCurrent(requestId)
      ) {
        markStaleCompareRequest(requestId);
        return false;
      }
      console.warn('Compare unavailable Plotly render failed', err);
      clearCompareDataState();
      plotDiv.__svPlotMode = 'compare-unavailable';
      plotDiv.__svComparePanelCount = 0;
      plotDiv.__svCompareMode = 'unavailable';
      setCompareStatus(text);
      return false;
    }

    if (Number.isInteger(requestId) && !isCompareRequestCurrent(requestId)) {
      markStaleCompareRequest(requestId);
      return false;
    }
    clearCompareDataState();
    plotDiv.__svPlotMode = 'compare-unavailable';
    plotDiv.__svComparePanelCount = 0;
    plotDiv.__svCompareMode = 'unavailable';
    setCompareStatus(text);
    return true;
  }

  function buildCompareRender(aPayload, bPayload, sources, decision, validation, windowInfo) {
    return buildCompareRenderModel({
      aPayload,
      bPayload,
      sources,
      decision,
      validation,
      windowInfo,
      scaling: currentScaling,
    });
  }

  function compareUnavailableMessage(sources) {
    if (!sources.a.available) {
      return isRawCompareSource(sources.a)
        ? 'A-B unavailable: A raw source is not available.'
        : 'A-B unavailable: A tap is not available. Run pipeline first.';
    }
    if (!sources.b.available) {
      return isRawCompareSource(sources.b)
        ? 'A-B unavailable: B raw source is not available.'
        : 'A-B unavailable: B tap is not available. Run pipeline first.';
    }
    if (compareShowDiffEnabled() && sources.a.domain !== sources.b.domain) {
      return 'A-B unavailable: source domains are different.';
    }
    return '';
  }

  async function fetchCompareAndPlot() {
    if (!isCompareModeEnabled()) return false;
    if (!currentFileId) return true;
    if (!sectionShape) {
      await fetchSectionMeta();
      if (!sectionShape) return true;
    }
    const key1Val = currentCompareKey1();
    if (key1Val === undefined) return true;
    const windowInfo = currentVisibleWindow();
    if (!windowInfo) return true;
    const plotDiv = document.getElementById('plot');
    if (!plotDiv) return true;

    const sources = getCompareSources();
    if (!sources.a.available || !sources.b.available) {
      const { requestId } = beginRenderRequest(COMPARE_RENDER_SLOT);
      if (typeof cancelActiveMainDecodeJob === 'function') cancelActiveMainDecodeJob();
      const message = compareUnavailableMessage(sources);
      await renderCompareUnavailable(message, requestId);
      if (isCompareRequestCurrent(requestId)) {
        markRenderRequestFailed(COMPARE_RENDER_SLOT, requestId);
      }
      return true;
    }
    const decision = decideCompareWindowMode(windowInfo, plotDiv, sources);
    const requestA = buildCompareRequest(sources.a, sources.a, key1Val, windowInfo, decision);
    const requestB = buildCompareRequest(sources.b, sources.a, key1Val, windowInfo, decision);
    const lmoKey = requestA.payloadMeta?.lmoKey;

    const { requestId, signal } = beginRenderRequest(
      COMPARE_RENDER_SLOT,
      [SECTION_RENDER_SLOT],
    );
    if (typeof cancelActiveMainDecodeJob === 'function') cancelActiveMainDecodeJob();
    const ctrl = {
      signal,
      abort: () => abortRenderRequest(COMPARE_RENDER_SLOT),
    };
    windowFetchCtrl = ctrl;
    showLoading(buildWindowLoadingMessage({
      mode: `compare ${decision.mode}`,
      stepX: decision.stepX,
      stepY: decision.stepY,
    }), { slotName: COMPARE_RENDER_SLOT, requestId });

    try {
      const rawValidation = await validateRawCompareSources(sources, signal);
      if (!isCompareRequestCurrent(requestId)) {
        markStaleCompareRequest(requestId);
        return true;
      }
      if (!rawValidation.ok) {
        const message = rawValidation.message || 'A-B unavailable: raw source grids are different.';
        const rendered = await renderCompareUnavailable(message, requestId);
        if (isCompareRequestCurrent(requestId)) {
          if (!rendered) setCompareStatus(message);
          markRenderRequestFailed(COMPARE_RENDER_SLOT, requestId);
        }
        return true;
      }

      try {
        await ensureRawCompareReferenceBaseline(sources, signal);
      } catch (err) {
        if (err && err.name === 'AbortError') return true;
        if (!isCompareRequestCurrent(requestId)) {
          markStaleCompareRequest(requestId);
          return true;
        }
        const message = err instanceof Error ? err.message : String(err);
        const rendered = await renderCompareUnavailable(message, requestId);
        if (isCompareRequestCurrent(requestId)) {
          if (!rendered) setCompareStatus(message);
          markRenderRequestFailed(COMPARE_RENDER_SLOT, requestId);
        }
        return true;
      }
      if (!isCompareRequestCurrent(requestId)) {
        markStaleCompareRequest(requestId);
        return true;
      }

      const aPromise = fetchComparePayload(requestA, signal, requestId);
      const bPromise = requestA.cacheKey === requestB.cacheKey
        ? aPromise
        : fetchComparePayload(requestB, signal, requestId);
      const [aPayload, bPayload] = await Promise.all([aPromise, bPromise]);
      if (!isCompareRequestCurrent(requestId) || !aPayload || !bPayload) {
        if (!isCompareRequestCurrent(requestId)) {
          markStaleCompareRequest(requestId);
        }
        return true;
      }
      if (!isCompareLmoCurrent(lmoKey) || aPayload.lmoKey !== lmoKey || bPayload.lmoKey !== lmoKey) {
        return true;
      }

      const validation = validateComparePair(aPayload, bPayload, sources);
      const render = buildCompareRender(aPayload, bPayload, sources, decision, validation, windowInfo);
      if (!render) {
        if (setCompareStatusIfCurrent(requestId, 'A-B unavailable: source data could not be decoded.')) {
          markRenderRequestFailed(COMPARE_RENDER_SLOT, requestId);
        }
        return true;
      }
      if (
        !isCompareRequestCurrent(requestId) ||
        !isCompareLmoCurrent(lmoKey) ||
        aPayload.lmoKey !== lmoKey ||
        bPayload.lmoKey !== lmoKey
      ) {
        if (!isCompareRequestCurrent(requestId)) markStaleCompareRequest(requestId);
        return true;
      }
      render.__requestSlot = COMPARE_RENDER_SLOT;
      render.__requestId = requestId;
      if (!isCompareRequestCurrent(requestId)) {
        markStaleCompareRequest(requestId);
        return true;
      }
      latestCompareRender = render;
      latestSeismicData = null;
      if (await renderCompareLatestView() && isCompareRequestCurrent(requestId)) {
        markRenderRequestCompleted(COMPARE_RENDER_SLOT, requestId);
      }
      return true;
    } catch (err) {
      if (err && err.name === 'AbortError') return true;
      if (err instanceof CompareFetchError) {
        const role = err.source?.role === 'A' ? 'A' : 'B';
        if (isCompareRequestCurrent(requestId)) {
          if (err.status === 409) {
            setCompareStatus(isRawCompareSource(err.source)
              ? err.message
              : `A-B unavailable: ${role} tap is not available. Run pipeline first.`);
          } else {
            setCompareStatus(err.message);
          }
          markRenderRequestFailed(COMPARE_RENDER_SLOT, requestId);
        } else {
          markStaleCompareRequest(requestId);
        }
        return true;
      }
      if (isCompareRequestCurrent(requestId)) {
        console.warn('Compare window fetch error', err);
        setCompareStatus(err instanceof Error ? err.message : String(err));
        markRenderRequestFailed(COMPARE_RENDER_SLOT, requestId);
      } else {
        markStaleCompareRequest(requestId);
      }
      return true;
    } finally {
      if (windowFetchCtrl === ctrl) windowFetchCtrl = null;
      hideLoading({ slotName: COMPARE_RENDER_SLOT, requestId });
    }
  }

  function compareCurrentDesiredMode() {
    if (!isCompareModeEnabled()) return null;
    const win = currentVisibleWindow();
    const plotDiv = document.getElementById('plot');
    if (!win || !plotDiv) return null;
    const sources = getCompareSources();
    return decideCompareWindowMode(win, plotDiv, sources).mode;
  }

  function compareNeedsFresh(decision, win, sources) {
    if (!latestCompareRender) return true;
    if (latestCompareRender.key1 !== currentCompareKey1()) return true;
    if (sourcePairKey(latestCompareRender.sources) !== sourcePairKey(sources)) return true;
    if (latestCompareRender.scaling !== currentScaling) return true;
    if (!isCompareLmoCurrent(latestCompareRender.lmoKey)) return true;
    if (latestCompareRender.mode !== decision.mode) return true;
    if (latestCompareRender.stepX !== decision.stepX || latestCompareRender.stepY !== decision.stepY) return true;
    if (latestCompareRender.x0 > win.x0 || latestCompareRender.x1 < win.x1) return true;
    if (latestCompareRender.y0 > win.y0 || latestCompareRender.y1 < win.y1) return true;
    return false;
  }

  function requestCompareWindowFetch(immediate) {
    if (typeof requestWindowFetch === 'function') {
      requestWindowFetch({ immediate: immediate === true });
      return;
    }
    if (typeof scheduleWindowFetch === 'function') scheduleWindowFetch();
  }

  function checkCompareModeFlipAndRefetch({ immediate = false } = {}) {
    if (!isCompareModeEnabled()) return false;
    const win = currentVisibleWindow();
    const plotDiv = document.getElementById('plot');
    if (!win || !plotDiv) return false;
    const sources = getCompareSources();
    const unavailable = compareUnavailableMessage(sources);
    if (unavailable && (!sources.a.available || !sources.b.available || compareShowDiffEnabled())) {
      setCompareStatus(unavailable);
    }
    const decision = decideCompareWindowMode(win, plotDiv, sources);
    if (compareNeedsFresh(decision, win, sources)) {
      requestCompareWindowFetch(immediate);
      return true;
    }
    renderCompareLatestView();
    return false;
  }

  function readCompareAxisRange(ev, base, index) {
    if (typeof readAxisRange !== 'function') return null;
    return readAxisRange(ev, `${base}axis${axisSuffix(index)}`);
  }

  function firstEventAxisRange(ev, base) {
    for (let i = 0; i < 3; i++) {
      const range = readCompareAxisRange(ev, base, i);
      if (range) return range;
    }
    return null;
  }

  function firstFullLayoutAxisRange(plotDiv, base) {
    const layout = plotDiv?._fullLayout;
    if (!layout) return null;
    const count = Math.max(2, Number(plotDiv.__svComparePanelCount) || 2);
    for (let i = 0; i < count; i++) {
      const axis = layout[axisLayoutName(base, i)];
      const range = axis?.range;
      if (Array.isArray(range) && range.length === 2 && Number.isFinite(range[0]) && Number.isFinite(range[1])) {
        return [range[0], range[1]];
      }
    }
    return null;
  }

  function syncCompareAxes(plotDiv, xRange, yRange) {
    if (!plotDiv || compareSyncing || !window.Plotly || typeof window.Plotly.relayout !== 'function') return null;
    const count = Math.max(2, Number(plotDiv.__svComparePanelCount) || 2);
    const props = {};
    for (let i = 0; i < count; i++) {
      const xs = axisSuffix(i);
      if (xRange) props[`xaxis${xs}.range`] = [xRange[0], xRange[1]];
      if (yRange) props[`yaxis${xs}.range`] = [yRange[0], yRange[1]];
      props[`xaxis${xs}.autorange`] = false;
      props[`yaxis${xs}.autorange`] = false;
    }
    compareSyncing = true;
    const promise = withSuppressedRelayout(window.Plotly.relayout(plotDiv, props));
    if (promise && typeof promise.finally === 'function') {
      return promise.finally(() => { compareSyncing = false; });
    }
    compareSyncing = false;
    return promise;
  }

  async function handleCompareRelayout(ev) {
    if (!isCompareModeEnabled() || compareSyncing) return;
    const plotDiv = document.getElementById('plot');
    if (!plotDiv) return;
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const xRange = firstEventAxisRange(ev, 'x') || firstFullLayoutAxisRange(plotDiv, 'x');
    const yRangeRaw = firstEventAxisRange(ev, 'y') || firstFullLayoutAxisRange(plotDiv, 'y');
    if (xRange) savedXRange = [xRange[0], xRange[1]];
    let yRange = null;
    if (yRangeRaw) {
      yRange = yRangeRaw[0] > yRangeRaw[1]
        ? [yRangeRaw[0], yRangeRaw[1]]
        : [yRangeRaw[1], yRangeRaw[0]];
      savedYRange = yRange;
    }
    await syncCompareAxes(plotDiv, savedXRange, savedYRange);
    checkCompareModeFlipAndRefetch({ immediate: typeof isResetRelayout === 'function' && isResetRelayout(ev) });
  }

  function snapshotCompareAxesRangesFromDOM() {
    if (!isCompareModeEnabled()) return false;
    const plotDiv = document.getElementById('plot');
    const xRange = firstFullLayoutAxisRange(plotDiv, 'x');
    const yRangeRaw = firstFullLayoutAxisRange(plotDiv, 'y');
    if (xRange) savedXRange = [xRange[0], xRange[1]];
    if (yRangeRaw) {
      savedYRange = yRangeRaw[0] > yRangeRaw[1]
        ? [yRangeRaw[0], yRangeRaw[1]]
        : [yRangeRaw[1], yRangeRaw[0]];
    }
    return true;
  }

  function clearCompareRender() {
    latestCompareRender = null;
    setCompareStatus('');
  }

  function onCompareControlChange() {
    updateCompareSourceOptions();
    snapshotCompareAxesRangesFromDOM();
    if (!isCompareModeEnabled()) {
      clearCompareRender();
      if (typeof renderLatestView === 'function') renderLatestView();
      if (typeof scheduleWindowFetch === 'function') scheduleWindowFetch();
      return;
    }
    const requested = checkCompareModeFlipAndRefetch({ immediate: true });
    if (!requested && !latestCompareRender) requestCompareWindowFetch(true);
  }

  function initCompareControls() {
    ensureActiveFileTargetSyncWrapped();
    updateCompareSourceOptions();
    const { toggle, sourceA, sourceB, showDiff, addDataset, importBSource, bSourceFile, clearDatasets } = getCompareNodes();
    for (const node of [toggle, sourceA, sourceB, showDiff]) {
      if (!node) continue;
      node.addEventListener('change', onCompareControlChange);
    }
    if (addDataset) addDataset.addEventListener('click', () => {
      addSelectedCompareDataset();
    });
    if (importBSource && bSourceFile) {
      importBSource.addEventListener('click', () => {
        if (compareImportInFlight) return;
        bSourceFile.value = '';
        bSourceFile.click();
      });
      bSourceFile.addEventListener('change', () => {
        const file = bSourceFile.files && bSourceFile.files[0];
        if (!file) return;
        importCompareBSourceFile(file);
      });
    }
    if (clearDatasets) clearDatasets.addEventListener('click', clearCompareDatasets);
    loadCompareRecentDatasets();
  }

  window.isCompareModeEnabled = isCompareModeEnabled;
  window.compareShowDiffEnabled = compareShowDiffEnabled;
  window.updateCompareSourceOptions = updateCompareSourceOptions;
  window.fetchCompareAndPlot = fetchCompareAndPlot;
  window.renderCompareLatestView = renderCompareLatestView;
  window.compareCurrentDesiredMode = compareCurrentDesiredMode;
  window.checkCompareModeFlipAndRefetch = checkCompareModeFlipAndRefetch;
  window.handleCompareRelayout = handleCompareRelayout;
  window.snapshotCompareAxesRangesFromDOM = snapshotCompareAxesRangesFromDOM;
  window.clearCompareRender = clearCompareRender;
  window.resetCompareTargetsForActiveFile = resetCompareTargetsForActiveFile;
  window.__svCompare = {
    validateComparePair,
    subtractF32,
    payloadToF32,
    activeCompareFileTarget: compareSources.activeCompareFileTarget,
    compareTargetLabelName: compareSources.compareTargetLabelName,
    rawCompareSource,
    tapCompareSource: compareSources.tapCompareSource,
    buildCompareSourceCatalog,
    resolveSourceDomain,
    resolveCompareSource,
    sourcePairKey,
    isRawCompareSource,
    normalizeCompareFileTarget,
    compareTargetDatasetKey,
    normalizeRecentDataset,
    compareRecentDatasetValue,
    resolveCompareRecentDataset,
    resolveCompareNormalizationFileId,
    shouldValidateRawCompareSources,
    rawCompareValidationKey,
    validateRawCompareSources,
    ensureRawCompareReferenceBaseline,
    renderCompareUnavailable,
    getLatestCompareRender: () => latestCompareRender,
    setLatestCompareRenderForTest: (render) => { latestCompareRender = render; },
    setCompareFileTargetsForTest: (targets) => {
      compareFileTargets = Array.isArray(targets) ? targets : [];
      compareActiveTargetKey = compareTargetIdentity(compareFileTargets[0]);
      window.compareFileTargets = compareFileTargets;
    },
    setCompareRecentDatasetsForTest: (datasets) => {
      compareRecentDatasets = Array.isArray(datasets) ? datasets : [];
      renderCompareDatasetPicker();
    },
    clearRawCompareValidationCache,
    compareUnavailableMessage,
    buildCompareRequest,
    addSelectedCompareDataset,
    importCompareBSourceFile,
    compareSourceId,
    addCompareDatasetTarget,
    clearCompareDatasetTargets,
    resetCompareTargetsForActive,
    compareHeatmapScale,
    buildComparePanels: buildComparePanelsForCurrentState,
    buildCompareLayout: buildCompareLayoutForCurrentState,
    buildCompareWiggleTraces: buildCompareWiggleTracesForCurrentState,
    buildCompareHeatmapTrace: buildCompareHeatmapTraceForCurrentState,
    initCompareControls,
  };

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', initCompareControls, { once: true });
  } else {
    initCompareControls();
  }
})();
