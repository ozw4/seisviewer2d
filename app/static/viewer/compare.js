(function () {
  const AXIS_MARGIN = 0.035;
  const AMP_LIMIT = 3.0;
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
    };
  }

  function compareTargetDatasetKey(target) {
    const normalized = normalizeCompareFileTarget(target);
    if (!normalized) return '';
    const name = String(normalized.originalName || normalized.displayName || '').trim();
    return `${name}|${normalized.key1Byte ?? ''}|${normalized.key2Byte ?? ''}`;
  }

  function activeCompareFileTarget() {
    return normalizeCompareFileTarget({
      fileId: window.currentFileId || '',
      displayName: window.currentFileName || window.currentFileId || '',
      key1Byte: window.currentKey1Byte,
      key2Byte: window.currentKey2Byte,
      isActive: true,
      originalName: window.currentFileName || '',
    });
  }

  function compareTargetIdentity(target) {
    const normalized = normalizeCompareFileTarget(target);
    if (!normalized) return '';
    return `${normalized.fileId}|${normalized.key1Byte ?? ''}|${normalized.key2Byte ?? ''}`;
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

  function rawCompareSource(target) {
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
      label: `${target.displayName} / raw`,
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

  function buildCompareSourceCatalog(targets, options = {}) {
    const catalog = [];
    const seen = new Set();
    const layerValues = Array.isArray(options.layerValues) ? options.layerValues : ['raw'];
    for (const candidate of targets || []) {
      const target = normalizeCompareFileTarget(candidate);
      if (!target || seen.has(target.fileId)) continue;
      seen.add(target.fileId);
      catalog.push(rawCompareSource(target));
      if (!target.isActive) continue;
      for (const layer of layerValues) {
        const tapLabel = String(layer || '').trim();
        if (!tapLabel || tapLabel === 'raw') continue;
        catalog.push(tapCompareSource(target, tapLabel, options));
      }
    }
    return catalog;
  }

  function currentCompareSourceCatalog() {
    return buildCompareSourceCatalog(syncCompareTargetsWithActive(), {
      layerValues: getLayerSourceOptions(),
      latestPipelineKey: window.latestPipelineKey || null,
      latestTapData: window.latestTapData || {},
    });
  }

  function normalizeRecentDataset(candidate) {
    if (!candidate || typeof candidate !== 'object') return null;
    const originalName = String(candidate.original_name ?? candidate.originalName ?? '').trim();
    if (!originalName) return null;
    const storeName = String(candidate.store_name ?? candidate.storeName ?? originalName).trim();
    const key1Byte = Number(candidate.key1_byte ?? candidate.key1Byte);
    const key2Byte = Number(candidate.key2_byte ?? candidate.key2Byte);
    if (!Number.isFinite(key1Byte) || !Number.isFinite(key2Byte)) return null;
    return {
      originalName,
      storeName,
      key1Byte,
      key2Byte,
    };
  }

  function compareRecentDatasetValue(candidate) {
    const dataset = normalizeRecentDataset(candidate);
    if (!dataset) return '';
    return [
      encodeURIComponent(dataset.originalName),
      dataset.key1Byte,
      dataset.key2Byte,
    ].join('|');
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
    if (addDataset) addDataset.disabled = !active || datasetPicker.options.length === 0;
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

  function resolveCompareSource(select, role) {
    const catalog = currentCompareSourceCatalog();
    const activeRaw = catalog.find((source) => source.layerId === 'raw') || rawCompareSource({
      fileId: window.currentFileId || '',
      displayName: window.currentFileName || window.currentFileId || 'raw',
      key1Byte: window.currentKey1Byte ?? null,
      key2Byte: window.currentKey2Byte ?? null,
      isActive: true,
    });
    const value = select?.value || activeRaw.sourceId;
    const source = catalog.find((entry) => entry.sourceId === value) || activeRaw;
    return { ...source, role };
  }

  function getCompareSources() {
    updateCompareSourceOptions();
    const { sourceA, sourceB } = getCompareNodes();
    return {
      a: resolveCompareSource(sourceA, 'A'),
      b: resolveCompareSource(sourceB, 'B'),
    };
  }

  function currentCompareKey1() {
    const slider = document.getElementById('key1_slider');
    const idx = slider ? parseInt(slider.value, 10) : 0;
    return Array.isArray(key1Values) ? key1Values[idx] : undefined;
  }

  function sourcePairKey(sources) {
    return [
      sources.a.fileId || '',
      sources.a.layerId || sources.a.id || '',
      sources.a.pipelineKey || '',
      sources.a.tapLabel || '',
      sources.b.fileId || '',
      sources.b.layerId || sources.b.id || '',
      sources.b.pipelineKey || '',
      sources.b.tapLabel || '',
    ].join('|');
  }

  function canAttemptDiff(sources) {
    return sources.a.domain === sources.b.domain;
  }

  function isRawCompareSource(source) {
    return (source?.layerId || source?.id || '') === 'raw';
  }

  function resolveCompareNormalizationFileId(source, referenceSource, fallbackFileId = null) {
    if (!isRawCompareSource(source) || !isRawCompareSource(referenceSource)) return null;
    const fileId = String(referenceSource?.fileId || source?.fileId || fallbackFileId || '').trim();
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
      sources.a.fileId,
      sources.b.fileId,
      key1Byte,
      key2Byte,
    ].map((value) => encodeURIComponent(String(value ?? ''))).join('|');
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
      key1Byte: active.key1Byte,
      key2Byte: active.key2Byte,
    });
    if (duplicateKey && syncCompareTargetsWithActive().some((target) => compareTargetDatasetKey(target) === duplicateKey)) {
      setCompareStatus('Dataset is already added.');
      return false;
    }
    const formData = new FormData();
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

  function clearCompareDatasets() {
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

  function payloadDt(payload) {
    const dt = Number(payload?.dt);
    if (Number.isFinite(dt) && dt > 0) return dt;
    const fallback = Number(window.defaultDt ?? defaultDt);
    return Number.isFinite(fallback) && fallback > 0 ? fallback : null;
  }

  function payloadShapeInfo(payload) {
    if (!payload || !Array.isArray(payload.shape) || payload.shape.length !== 2) return null;
    const rows = Number(payload.shape[0]);
    const cols = Number(payload.shape[1]);
    if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows <= 0 || cols <= 0) return null;
    return { rows, cols, total: rows * cols };
  }

  function payloadInvScale(payload) {
    const payloadScale = Number(payload?.scale);
    const quantScale = Number(payload?.quant?.scale);
    const scale = Number.isFinite(payloadScale) && payloadScale !== 0
      ? payloadScale
      : quantScale;
    return Number.isFinite(scale) && scale !== 0 ? 1 / scale : 1;
  }

  function payloadHasComputeValues(payload) {
    const shape = payloadShapeInfo(payload);
    if (!shape) return false;
    return (
      (payload.valuesI8 instanceof Int8Array && payload.valuesI8.length >= shape.total) ||
      (payload.values instanceof Float32Array && payload.values.length >= shape.total)
    );
  }

  function canUseCachedComparePayload(payload, source) {
    if (source?.domain !== 'probability') return true;
    return payloadHasComputeValues(payload);
  }

  function sourceDomain(options) {
    if (typeof options === 'string') return options;
    return options?.domain || '';
  }

  function payloadToF32(payload, options = {}) {
    const shape = payloadShapeInfo(payload);
    if (!shape) return null;
    const { rows, cols, total } = shape;
    let out = null;
    if (payload.valuesI8 instanceof Int8Array && payload.valuesI8.length >= total) {
      const invScale = payloadInvScale(payload);
      out = new Float32Array(total);
      for (let i = 0; i < total; i++) out[i] = payload.valuesI8[i] * invScale;
    } else if (payload.values instanceof Float32Array && payload.values.length >= total) {
      out = new Float32Array(payload.values.subarray(0, total));
    } else if (sourceDomain(options) === 'probability') {
      return null;
    } else if (payload.zBacking instanceof Float32Array && payload.zBacking.length >= total) {
      out = new Float32Array(payload.zBacking.subarray(0, total));
    } else if (Array.isArray(payload.zRows) && payload.zRows.length === rows) {
      out = new Float32Array(total);
      for (let r = 0; r < rows; r++) {
        const row = payload.zRows[r];
        if (!row || row.length < cols) return null;
        out.set(row.subarray ? row.subarray(0, cols) : Array.from(row).slice(0, cols), r * cols);
      }
    }
    return out;
  }

  function sameShape(a, b) {
    return Array.isArray(a?.shape) && Array.isArray(b?.shape) &&
      a.shape.length === 2 && b.shape.length === 2 &&
      Number(a.shape[0]) === Number(b.shape[0]) &&
      Number(a.shape[1]) === Number(b.shape[1]);
  }

  function sameGrid(a, b) {
    return ['x0', 'x1', 'y0', 'y1', 'stepX', 'stepY'].every((key) => Number(a?.[key]) === Number(b?.[key]));
  }

  function validateComparePair(a, b, sources) {
    if (!sameShape(a, b)) return { ok: false, reason: 'shape', message: 'A-B unavailable: source shapes are different.' };
    const dtA = payloadDt(a);
    const dtB = payloadDt(b);
    if (!(Number.isFinite(dtA) && Number.isFinite(dtB)) || Math.abs(dtA - dtB) > 1e-9) {
      return { ok: false, reason: 'dt', message: 'A-B unavailable: source sample intervals are different.' };
    }
    if (!sameGrid(a, b)) return { ok: false, reason: 'grid', message: 'A-B unavailable: source grids are different.' };
    if (sources.a.domain !== sources.b.domain) {
      return { ok: false, reason: 'domain', message: 'A-B unavailable: source domains are different.' };
    }
    return { ok: true, reason: '', message: '' };
  }

  function subtractF32(a, b) {
    if (!(a instanceof Float32Array) || !(b instanceof Float32Array) || a.length !== b.length) return null;
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
    return out;
  }

  function rowsFromF32(values, rows, cols) {
    const out = new Array(rows);
    for (let r = 0; r < rows; r++) out[r] = values.subarray(r * cols, (r + 1) * cols);
    return out;
  }

  function axisSuffix(index) {
    return index === 0 ? '' : String(index + 1);
  }

  function axisRef(base, index) {
    return index === 0 ? base : `${base}${index + 1}`;
  }

  function axisLayoutName(base, index) {
    return `${base}axis${axisSuffix(index)}`;
  }

  function compareDomains(count) {
    const gapTotal = AXIS_MARGIN * Math.max(0, count - 1);
    const panelWidth = (1 - gapTotal) / count;
    const domains = [];
    let start = 0;
    for (let i = 0; i < count; i++) {
      const end = i === count - 1 ? 1 : start + panelWidth;
      domains.push([start, end]);
      start = end + AXIS_MARGIN;
    }
    return domains;
  }

  function panelTitle(panel) {
    if (panel.kind === 'diff') return `A-B: ${panel.left} - ${panel.right}`;
    return `${panel.role}: ${panel.label}`;
  }

  function buildCompareLayout(render, panels, xRange, yRange) {
    const domains = compareDomains(panels.length);
    const dt = Number(payloadDt(render.a.payload)) || Number(window.defaultDt ?? defaultDt) || 0;
    const yDefault = [(render.windowInfo.y1 * dt), (render.windowInfo.y0 * dt)];
    const layout = {
      clickmode: clickModeForCurrentState(),
      dragmode: effectiveDragMode(),
      uirevision: `${currentUiRevision()}:compare:${sourcePairKey(render.sources)}`,
      paper_bgcolor: '#fff',
      plot_bgcolor: '#fff',
      margin: { t: 38, r: 12, l: 58, b: 42 },
      annotations: [],
      showlegend: false,
    };
    for (let i = 0; i < panels.length; i++) {
      const xName = axisLayoutName('x', i);
      const yName = axisLayoutName('y', i);
      layout[xName] = {
        domain: domains[i],
        title: 'Trace',
        showgrid: false,
        tickfont: { color: '#000' },
        titlefont: { color: '#000' },
        autorange: false,
        range: xRange || [render.windowInfo.x0, render.windowInfo.x1],
      };
      layout[yName] = {
        domain: [0, 1],
        title: i === 0 ? 'Time (s)' : '',
        showgrid: false,
        tickfont: { color: '#000' },
        titlefont: { color: '#000' },
        autorange: false,
        range: yRange || yDefault,
      };
      layout.annotations.push({
        xref: 'paper',
        yref: 'paper',
        x: (domains[i][0] + domains[i][1]) / 2,
        y: 1.06,
        xanchor: 'center',
        yanchor: 'bottom',
        showarrow: false,
        text: panelTitle(panels[i]),
        font: { size: 13, color: '#111827' },
      });
    }
    return layout;
  }

  function buildCompareWiggleTraces(panel, axisIndex, render) {
    const { rows, cols, x0, stepX, y0, stepY } = render;
    const values = panel.values;
    const dt = Number(payloadDt(render.a.payload)) || Number(window.defaultDt ?? defaultDt) || 0;
    const gain = parseFloat(document.getElementById('gain')?.value) || 1.0;
    const lineSegLen = rows + 1;
    const fillSegLen = (2 * rows) + 2;
    const baseX = new Float32Array(cols * lineSegLen);
    const baseY = new Float32Array(cols * lineSegLen);
    const lineX = new Float32Array(cols * lineSegLen);
    const lineY = new Float32Array(cols * lineSegLen);
    const fillX = new Float32Array(cols * fillSegLen);
    const fillY = new Float32Array(cols * fillSegLen);
    for (let c = 0; c < cols; c++) {
      const traceIndex = x0 + c * stepX;
      const lineStart = c * lineSegLen;
      const fillStart = c * fillSegLen;
      for (let r = 0; r < rows; r++) {
        const t = (y0 + r * stepY) * dt;
        const idx = r * cols + c;
        let val = values[idx] * gain;
        if (val > AMP_LIMIT) val = AMP_LIMIT;
        if (val < -AMP_LIMIT) val = -AMP_LIMIT;
        const posVal = val < 0 ? 0 : val;
        const lineIdx = lineStart + r;
        const fillBaseIdx = fillStart + r;
        const fillPosIdx = fillStart + rows + (rows - 1 - r);
        baseX[lineIdx] = traceIndex;
        baseY[lineIdx] = t;
        lineX[lineIdx] = traceIndex + val;
        lineY[lineIdx] = t;
        fillX[fillBaseIdx] = traceIndex;
        fillY[fillBaseIdx] = t;
        fillX[fillPosIdx] = traceIndex + posVal;
        fillY[fillPosIdx] = t;
      }
      const lineNanIdx = lineStart + rows;
      baseX[lineNanIdx] = NaN;
      baseY[lineNanIdx] = NaN;
      lineX[lineNanIdx] = NaN;
      lineY[lineNanIdx] = NaN;
      const fillCloseIdx = fillStart + (2 * rows);
      const fillNanIdx = fillCloseIdx + 1;
      fillX[fillCloseIdx] = traceIndex;
      fillY[fillCloseIdx] = (y0 * dt);
      fillX[fillNanIdx] = NaN;
      fillY[fillNanIdx] = NaN;
    }
    const xaxis = axisRef('x', axisIndex);
    const yaxis = axisRef('y', axisIndex);
    return [
      {
        type: 'scatter',
        mode: 'lines',
        x: baseX,
        y: baseY,
        xaxis,
        yaxis,
        line: { width: 0 },
        connectgaps: false,
        hoverinfo: 'skip',
        showlegend: false,
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: fillX,
        y: fillY,
        xaxis,
        yaxis,
        fill: 'toself',
        fillcolor: 'black',
        line: { width: 0 },
        opacity: 0.6,
        connectgaps: false,
        hoverinfo: 'skip',
        showlegend: false,
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: lineX,
        y: lineY,
        xaxis,
        yaxis,
        line: { color: 'black', width: 0.5 },
        connectgaps: false,
        hoverinfo: 'x+y',
        showlegend: false,
      },
    ];
  }

  function buildCompareHeatmapTrace(panel, axisIndex, render) {
    const { rows, cols, x0, stepX, y0, stepY } = render;
    const xVals = new Float32Array(cols);
    for (let c = 0; c < cols; c++) xVals[c] = x0 + c * stepX;
    const dt = Number(payloadDt(render.a.payload)) || Number(window.defaultDt ?? defaultDt) || 0;
    const yVals = new Float32Array(rows);
    for (let r = 0; r < rows; r++) yVals[r] = (y0 + r * stepY) * dt;
    const gain = parseFloat(document.getElementById('gain')?.value) || 1.0;
    const cmName = document.getElementById('colormap')?.value || 'Greys';
    const reverse = !!document.getElementById('cmReverse')?.checked;
    const cm = (window.COLORMAPS && window.COLORMAPS[cmName]) || 'Greys';
    const scale = compareHeatmapScale(panel, gain);
    const isDiv = scale.signed && (cmName === 'RdBu' || cmName === 'BWR');
    return {
      type: 'heatmap',
      x: xVals,
      y: yVals,
      z: rowsFromF32(panel.values, rows, cols),
      xaxis: axisRef('x', axisIndex),
      yaxis: axisRef('y', axisIndex),
      colorscale: cm,
      reversescale: reverse,
      zmin: scale.zmin,
      zmax: scale.zmax,
      zmid: isDiv ? 0 : null,
      showscale: false,
      hoverinfo: 'x+y',
      hovertemplate: '',
    };
  }

  function compareHeatmapScale(panel, gain) {
    const g = Math.max(Number(gain) || 1.0, 1e-9);
    if (panel?.kind === 'source' && panel.domain === 'probability') {
      return { zmin: 0, zmax: 1 / g, signed: false };
    }
    if (panel?.kind === 'diff' && panel.domain === 'probability') {
      return { zmin: -1 / g, zmax: 1 / g, signed: true };
    }
    return { zmin: -AMP_LIMIT / g, zmax: AMP_LIMIT / g, signed: true };
  }

  function buildComparePanels(render) {
    const panels = [
      {
        kind: 'source',
        role: 'A',
        domain: render.sources.a.domain,
        label: render.sources.a.label,
        values: render.a.values,
      },
      {
        kind: 'source',
        role: 'B',
        domain: render.sources.b.domain,
        label: render.sources.b.label,
        values: render.b.values,
      },
    ];
    if (compareShowDiffEnabled() && render.diffAvailable && render.diffValues) {
      panels.push({
        kind: 'diff',
        role: 'A-B',
        domain: render.sources.a.domain,
        label: `${render.sources.a.label} - ${render.sources.b.label}`,
        left: render.sources.a.label,
        right: render.sources.b.label,
        values: render.diffValues,
      });
    }
    return panels;
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
    const panels = buildComparePanels(render);
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
      if (render.mode === 'wiggle') traces.push(...buildCompareWiggleTraces(panels[i], i, render));
      else traces.push(buildCompareHeatmapTrace(panels[i], i, render));
    }
    const layout = buildCompareLayout(render, panels, xRange, yRange);
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

  function buildCompareRender(aPayload, bPayload, sources, decision, validation, windowInfo) {
    const aValues = payloadToF32(aPayload, sources.a);
    const bValues = payloadToF32(bPayload, sources.b);
    if (!aValues || !bValues) {
      return null;
    }
    const rows = Number(aPayload.shape[0]);
    const cols = Number(aPayload.shape[1]);
    const diffValues = validation.ok ? subtractF32(aValues, bValues) : null;
    return {
      key1: aPayload.key1,
      sources,
      sourcePair: sourcePairKey(sources),
      scaling: currentScaling,
      lmoKey: aPayload.lmoKey,
      mode: decision.mode,
      panelCount: decision.panelCount,
      stepX: decision.stepX,
      stepY: decision.stepY,
      x0: aPayload.x0,
      x1: aPayload.x1,
      y0: aPayload.y0,
      y1: aPayload.y1,
      rows,
      cols,
      windowInfo,
      a: { payload: aPayload, values: aValues },
      b: { payload: bPayload, values: bValues },
      diffAvailable: validation.ok && !!diffValues,
      diffMessage: validation.message,
      diffValues,
    };
  }

  function compareUnavailableMessage(sources) {
    if (!sources.a.available) return 'A-B unavailable: A tap is not available. Run pipeline first.';
    if (!sources.b.available) return 'A-B unavailable: B tap is not available. Run pipeline first.';
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
      setCompareStatus(compareUnavailableMessage(sources));
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
        if (setCompareStatusIfCurrent(
          requestId,
          rawValidation.message || 'A-B unavailable: raw source grids are different.',
        )) {
          markRenderRequestFailed(COMPARE_RENDER_SLOT, requestId);
        }
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
            setCompareStatus(`A-B unavailable: ${role} tap is not available. Run pipeline first.`);
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
    const { toggle, sourceA, sourceB, showDiff, addDataset, clearDatasets } = getCompareNodes();
    for (const node of [toggle, sourceA, sourceB, showDiff]) {
      if (!node) continue;
      node.addEventListener('change', onCompareControlChange);
    }
    if (addDataset) addDataset.addEventListener('click', () => {
      addSelectedCompareDataset();
    });
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
    resolveSourceDomain,
    sourcePairKey,
    normalizeCompareFileTarget,
    normalizeRecentDataset,
    compareRecentDatasetValue,
    resolveCompareRecentDataset,
    resolveCompareNormalizationFileId,
    validateRawCompareSources,
    clearRawCompareValidationCache,
    buildCompareRequest,
    addCompareDatasetTarget,
    clearCompareDatasetTargets,
    resetCompareTargetsForActive,
    buildCompareSourceCatalog,
    compareHeatmapScale,
    buildComparePanels,
  };

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', initCompareControls, { once: true });
  } else {
    initCompareControls();
  }
})();
