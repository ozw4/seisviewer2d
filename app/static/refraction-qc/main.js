import {
  DEFAULT_MAX_POINTS,
  INCLUDE_ALL,
  TASK_DEFS,
  VIEW_DEFS,
} from './constants.js';
import * as qcApi from './api.js';
import { collectRefractionQcDom, isStandaloneRefractionQcPage } from './dom.js';
import { writeRecentJob } from './recent_jobs.js';
import { clearSelectedObject, resetState, state } from './state.js';
import {
  safeLocalStorageJson,
  safeLocalStorageValue,
  searchOrStorageValue,
  searchParamValue,
  writeJobIdUrlParam,
} from './url_params.js';
import {
  defaultViewForTask as taskDefaultViewForTask,
  taskForView as taskModuleTaskForView,
} from './tasks.js';
import {
  createRefractionQcRenderRuntime,
  firstDefined,
  parsePositiveInteger,
  pickMapGatherNumber,
  toFiniteNumber,
} from './runtime.js';

  const ACTIVE_VIEWER_TARGET_STORAGE_KEY = 'sv.active_viewer_target';
  const STATIC_DRAFT_STORAGE_KEY = 'sv.static_correction.form_draft.v1';
  const STATIC_PICK_DB_NAME = 'seisviewer2d-static-correction';
  const STATIC_PICK_DB_VERSION = 1;
  const STATIC_PICK_STORE = 'pick_npz_blobs';
  const QC_DRILLDOWN_MAX_OBSERVATIONS = 200;
  const PICK_MAP_DRAFT_GEOMETRY_HEADER_FIELDS = [
    'source_id_byte',
    'receiver_id_byte',
    'source_x_byte',
    'source_y_byte',
    'receiver_x_byte',
    'receiver_y_byte',
    'source_elevation_byte',
    'receiver_elevation_byte',
    'coordinate_scalar_byte',
    'elevation_scalar_byte',
  ];

  let dom = null;
  let requestSerial = 0;
  let gatherRequestSerial = 0;
  let qcDrilldownRequestSerial = 0;
  let pickMapRequestSerial = 0;
  let stationStructureRequestSerial = 0;
  let renderRuntime = null;

  function positiveIntegerText(value) {
    const text = String(value ?? '').trim();
    const parsed = Number(text);
    if (!text || !Number.isInteger(parsed) || parsed < 1) return '';
    return text;
  }

  function gatherContextFromQcBundle(bundle) {
    if (!bundle || typeof bundle !== 'object') return { fileId: '', key1Byte: '', key2Byte: '' };
    const summary = bundle.summary && typeof bundle.summary === 'object' ? bundle.summary : {};
    const request = summary.request && typeof summary.request === 'object' ? summary.request : {};
    return {
      fileId: String(firstDefined(request, ['file_id', 'input_file_id', 'source_file_id', 'raw_file_id'])
        ?? firstDefined(summary, ['file_id', 'input_file_id', 'source_file_id', 'raw_file_id'])
        ?? firstDefined(bundle, ['file_id', 'input_file_id', 'source_file_id', 'raw_file_id'])
        ?? '').trim(),
      key1Byte: positiveIntegerText(firstDefined(request, ['key1_byte'])
        ?? firstDefined(summary, ['key1_byte'])
        ?? firstDefined(bundle, ['key1_byte'])),
      key2Byte: positiveIntegerText(firstDefined(request, ['key2_byte'])
        ?? firstDefined(summary, ['key2_byte'])
        ?? firstDefined(bundle, ['key2_byte'])),
    };
  }

  function syncGatherContextFromBundle(bundle) {
    const context = gatherContextFromQcBundle(bundle);
    let changed = false;
    if (context.fileId && state.gatherFileId !== context.fileId) {
      state.gatherFileId = context.fileId;
      changed = true;
    }
    if (context.key1Byte && state.gatherKey1Byte !== context.key1Byte) {
      state.gatherKey1Byte = context.key1Byte;
      changed = true;
    }
    if (context.key2Byte && state.gatherKey2Byte !== context.key2Byte) {
      state.gatherKey2Byte = context.key2Byte;
      changed = true;
    }
    if (changed && state.gatherPreview) {
      state.gatherPreview = null;
      state.gatherError = null;
      gatherRequestSerial += 1;
    }
  }

  function normalizePickMapTargetCandidate(candidate, options = {}) {
    if (!candidate || typeof candidate !== 'object') return null;
    if (options.requireLoaded && candidate.isFileLoaded === false) return null;
    const fileId = String(candidate.fileId ?? candidate.file_id ?? '').trim();
    if (!fileId) return null;
    const key1Byte = Number(candidate.key1Byte ?? candidate.key1_byte);
    const key2Byte = Number(candidate.key2Byte ?? candidate.key2_byte);
    if (!Number.isInteger(key1Byte) || key1Byte <= 0 || !Number.isInteger(key2Byte) || key2Byte <= 0) {
      return null;
    }
    return { file_id: fileId, key1_byte: key1Byte, key2_byte: key2Byte };
  }

  function storedActivePickMapTarget() {
    return normalizePickMapTargetCandidate(
      safeLocalStorageJson(ACTIVE_VIEWER_TARGET_STORAGE_KEY),
      { requireLoaded: true }
    );
  }

  function standalonePickMapTarget() {
    const storedTarget = storedActivePickMapTarget();
    return normalizePickMapTargetCandidate({
      fileId: searchParamValue('file_id'),
      key1Byte: searchParamValue('key1_byte') || (storedTarget && storedTarget.key1_byte),
      key2Byte: searchParamValue('key2_byte') || (storedTarget && storedTarget.key2_byte),
      isFileLoaded: true,
    })
      || storedTarget
      || normalizePickMapTargetCandidate({
        fileId: searchOrStorageValue('file_id', 'file_id', ''),
        key1Byte: searchOrStorageValue('key1_byte', 'key1_byte', safeLocalStorageValue('last_key1_byte') || '189'),
        key2Byte: searchOrStorageValue('key2_byte', 'key2_byte', safeLocalStorageValue('last_key2_byte') || '193'),
        isFileLoaded: true,
      });
  }

  function activePickMapTarget() {
    const viewerState = window.SeisViewerState;
    if (viewerState && typeof viewerState.getActiveFileTarget === 'function') {
      const active = normalizePickMapTargetCandidate(
        viewerState.getActiveFileTarget(),
        { requireLoaded: true }
      );
      if (active) return active;
    }
    if (viewerState && typeof viewerState.getActiveFileTargetState === 'function') {
      const targetState = normalizePickMapTargetCandidate(viewerState.getActiveFileTargetState());
      if (targetState) return targetState;
    }
    return standalonePickMapTarget();
  }

  function samePickMapTarget(left, right) {
    if (!left || !right) return false;
    return String(left.file_id || '') === String(right.file_id || '')
      && Number(left.key1_byte) === Number(right.key1_byte)
      && Number(left.key2_byte) === Number(right.key2_byte);
  }

  function pickMapDraftHeaderByte(value, { optional = false } = {}) {
    const text = String(value ?? '').trim();
    if (optional && !text) return null;
    if (!/^\d+$/.test(text)) return undefined;
    const parsed = Number(text);
    return parsed >= 1 && parsed <= 240 ? parsed : undefined;
  }

  function staticCorrectionDraftPickMapGeometry(target) {
    const draft = safeLocalStorageJson(STATIC_DRAFT_STORAGE_KEY);
    if (!draft || draft.version !== 1 || !samePickMapTarget(draft.target, target)) return null;
    const values = draft.form?.geometry;
    if (!values || typeof values !== 'object') return null;

    const geometry = { receiver_number_mode: 'global_sequential' };
    for (const key of PICK_MAP_DRAFT_GEOMETRY_HEADER_FIELDS) {
      const parsed = pickMapDraftHeaderByte(values[key]);
      if (parsed === undefined) return null;
      geometry[key] = parsed;
    }
    const sourceDepthByte = pickMapDraftHeaderByte(values.source_depth_byte, { optional: true });
    if (sourceDepthByte === undefined) return null;
    geometry.source_depth_byte = sourceDepthByte;

    const coordinateUnit = String(values.coordinate_unit ?? '').trim();
    const elevationUnit = String(values.elevation_unit ?? '').trim();
    if (!['m', 'ft'].includes(coordinateUnit) || !['m', 'ft'].includes(elevationUnit)) {
      return null;
    }
    geometry.coordinate_unit = coordinateUnit;
    geometry.elevation_unit = elevationUnit;
    return geometry;
  }

  function pickMapGeometryRequest(target) {
    return staticCorrectionDraftPickMapGeometry(target) || { receiver_number_mode: 'global_sequential' };
  }

  function openStaticPickDb() {
    return new Promise((resolve, reject) => {
      if (!window.indexedDB) {
        reject(new Error('IndexedDB is not available in this browser.'));
        return;
      }
      const request = window.indexedDB.open(STATIC_PICK_DB_NAME, STATIC_PICK_DB_VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STATIC_PICK_STORE)) {
          db.createObjectStore(STATIC_PICK_STORE, { keyPath: 'id' });
        }
      };
      request.onerror = () => reject(request.error || new Error('Failed to open IndexedDB.'));
      request.onsuccess = () => resolve(request.result);
    });
  }

  function requestToPromise(request) {
    return new Promise((resolve, reject) => {
      request.onerror = () => reject(request.error || new Error('IndexedDB request failed.'));
      request.onsuccess = () => resolve(request.result);
    });
  }

  async function loadStaticPickRecord(recordId) {
    if (!recordId) return null;
    const db = await openStaticPickDb();
    try {
      const tx = db.transaction(STATIC_PICK_STORE, 'readonly');
      const store = tx.objectStore(STATIC_PICK_STORE);
      return await requestToPromise(store.get(recordId));
    } finally {
      db.close();
    }
  }

  async function restoreCachedPickMapSource() {
    const target = activePickMapTarget();
    const draft = safeLocalStorageJson(STATIC_DRAFT_STORAGE_KEY);
    const meta = draft?.pickNpz || null;
    state.pickMapCachedFile = null;
    state.pickMapCachedMeta = null;
    state.pickMapCacheStatus = '';
    if (!target) {
      state.pickMapCacheStatus = 'Open a viewer target before loading a pre-statics Pick Map.';
      render();
      return;
    }
    if (!draft || !meta) {
      state.pickMapCacheStatus = 'No Static Correction NPZ is cached for the active viewer target. Open Static Correction, select a pick NPZ, then return to Refraction QC.';
      render();
      return;
    }
    if (!samePickMapTarget(draft.target, target) || !samePickMapTarget({
      file_id: meta.fileId,
      key1_byte: meta.key1Byte,
      key2_byte: meta.key2Byte,
    }, target)) {
      state.pickMapCacheStatus = 'Saved Static Correction NPZ belongs to a different viewer target.';
      render();
      return;
    }
    try {
      const record = await loadStaticPickRecord(meta.indexedDbRecordId);
      if (!record || !record.blob) {
        state.pickMapCacheStatus = 'Saved Static Correction NPZ is no longer available.';
        render();
        return;
      }
      state.pickMapCachedFile = record.blob instanceof File
        ? record.blob
        : new File([record.blob], record.filename || meta.filename || 'first_breaks.npz', {
          type: record.type || meta.type || 'application/octet-stream',
          lastModified: record.lastModified || meta.lastModified || Date.now(),
        });
      state.pickMapCachedMeta = meta;
      state.pickMapCacheStatus = `Cached NPZ available: ${state.pickMapCachedFile.name}`;
    } catch (error) {
      state.pickMapCacheStatus = error instanceof Error ? error.message : String(error);
    }
    render();
  }

  function staticCorrectionLinkForTarget(target) {
    const url = new URL('/static-correction', window.location.origin);
    if (target) {
      url.searchParams.set('file_id', target.file_id);
      url.searchParams.set('key1_byte', String(target.key1_byte));
      url.searchParams.set('key2_byte', String(target.key2_byte));
    }
    return `${url.pathname}${url.search}`;
  }

  function getRenderRuntime() {
    if (!renderRuntime) {
      renderRuntime = createRefractionQcRenderRuntime({
        actions: {
          activePickMapTarget,
          invalidateGatherRequest: () => { gatherRequestSerial += 1; },
          invalidateQcDrilldownRequest: () => { qcDrilldownRequestSerial += 1; },
          isCurrentQcDrilldownRequest: (serial) => serial === qcDrilldownRequestSerial,
          loadCompletedPickMap,
          loadGatherPreview,
          loadPreStaticsPickMap,
          loadStationStructureQc,
          nextQcDrilldownRequestSerial: () => {
            qcDrilldownRequestSerial += 1;
            return qcDrilldownRequestSerial;
          },
          setSelectedView,
          staticCorrectionLinkForTarget,
        },
      });
    }
    return renderRuntime;
  }

  function taskForView(viewId) {
    return taskModuleTaskForView(viewId);
  }

  function defaultViewForTask(taskId) {
    return taskDefaultViewForTask(taskId);
  }

  function isPickMapView(viewId) {
    return getRenderRuntime().isPickMapView(viewId);
  }

  function render() {
    getRenderRuntime().render();
  }

  function renderRecentJobs() {
    getRenderRuntime().renderRecentJobs();
  }

  function resetJobScopedFilters() {
    getRenderRuntime().resetJobScopedFilters();
  }

  function buildGatherPreviewRequest() {
    return getRenderRuntime().buildGatherPreviewRequest();
  }

  function setSelectedView(viewId) {
    if (!VIEW_DEFS.some((view) => view.id === viewId)) return;
    state.selectedView = viewId;
    state.activeTask = taskForView(viewId);
    render();
    if (isPickMapView(viewId)) {
      restoreCachedPickMapSource();
      if (state.qcBundle?.job_id && !state.pickMap && !state.pickMapLoading) {
        loadCompletedPickMap(state.qcBundle.job_id);
      }
    } else if (viewId === 'station_structure') {
      if (state.qcBundle?.job_id && !state.stationStructure && !state.stationStructureLoading) {
        loadStationStructureQc(state.qcBundle.job_id);
      }
    }
  }

  function setActiveTask(taskId) {
    if (!TASK_DEFS.some((task) => task.id === taskId)) return;
    state.activeTask = taskId;
    state.selectedView = defaultViewForTask(taskId);
    render();
    if (isPickMapView(state.selectedView)) {
      restoreCachedPickMapSource();
      if (state.qcBundle?.job_id && !state.pickMap && !state.pickMapLoading) {
        loadCompletedPickMap(state.qcBundle.job_id);
      }
    } else if (state.selectedView === 'station_structure') {
      if (state.qcBundle?.job_id && !state.stationStructure && !state.stationStructureLoading) {
        loadStationStructureQc(state.qcBundle.job_id);
      }
    }
  }

  function activateSidebarTab(tabName) {
    if (!dom) return;
    const tabs = [
      { name: 'pipeline', tab: dom.pipelineTab, panel: dom.pipelinePanel },
      { name: 'static_correction', tab: dom.staticCorrectionTab, panel: dom.staticCorrectionPanel },
      { name: 'refraction_qc', tab: dom.qcTab, panel: dom.qcPanel },
    ].filter((item) => item.tab && item.panel);
    if (!tabs.length) return;
    const selectedTabName = tabs.some((tab) => tab.name === tabName) ? tabName : 'pipeline';
    for (const item of tabs) {
      const active = item.name === selectedTabName;
      item.tab.classList.toggle('is-active', active);
      item.tab.setAttribute('aria-selected', active ? 'true' : 'false');
      item.panel.hidden = !active;
    }
  }

  function buildPickMapUploadRequest() {
    const target = activePickMapTarget();
    if (!target) return { payload: null, error: 'Open a viewer target before loading a pre-statics Pick Map.' };
    return {
      payload: {
        file_id: target.file_id,
        key1_byte: target.key1_byte,
        key2_byte: target.key2_byte,
        pick_source: { kind: 'uploaded_npz' },
        geometry: pickMapGeometryRequest(target),
      },
      error: '',
    };
  }

  async function loadPreStaticsPickMap(file) {
    if (!file) {
      state.pickMapError = 'Select a first-break pick NPZ.';
      render();
      return;
    }
    const { payload, error } = buildPickMapUploadRequest();
    if (error) {
      state.pickMapError = error;
      render();
      return;
    }
    await loadPickMapRequest(() => qcApi.fetchPickMapUpload(payload, file));
  }

  async function loadCompletedPickMap(jobId) {
    const cleanJobId = String(jobId || state.qcBundle?.job_id || state.selectedJobId || '').trim();
    if (!cleanJobId) {
      state.pickMapError = 'Job ID is required for completed-job Pick Map.';
      render();
      return;
    }
    await loadPickMapRequest(() => qcApi.fetchCompletedPickMap(cleanJobId));
  }

  async function loadPickMapRequest(requestFactory) {
    const serial = ++pickMapRequestSerial;
    state.pickMapLoading = true;
    state.pickMapError = null;
    render();
    try {
      const pickMap = await requestFactory();
      if (serial !== pickMapRequestSerial) return;
      state.pickMap = pickMap;
      state.pickMapError = null;
      if (!pickMap.has_after_statics) state.pickMapDisplayMode = 'before';
    } catch (error) {
      if (serial !== pickMapRequestSerial) return;
      state.pickMap = null;
      state.pickMapError = error instanceof Error ? error.message : String(error);
    } finally {
      if (serial === pickMapRequestSerial) {
        state.pickMapLoading = false;
        render();
      }
    }
  }

  function buildStationStructureRequest(jobId) {
    return {
      job_id: jobId,
      gather_start: stationStructureOptionalNumber(state.stationStructureGatherStart),
      gather_end: stationStructureOptionalNumber(state.stationStructureGatherEnd),
      x_axis: 'auto',
      velocity_field: state.stationStructureVelocityField,
      depth_field: state.stationStructureDepthField,
    };
  }

  function stationStructureOptionalNumber(value) {
    const number = toFiniteNumber(value);
    return Number.isFinite(number) ? number : null;
  }

  async function loadStationStructureQc(jobId) {
    const cleanJobId = String(jobId || state.qcBundle?.job_id || state.selectedJobId || '').trim();
    if (!cleanJobId) {
      state.stationStructureError = 'Job ID is required for station-structure QC.';
      render();
      return;
    }
    const serial = ++stationStructureRequestSerial;
    state.stationStructureLoading = true;
    state.stationStructureError = null;
    render();
    try {
      const payload = await qcApi.fetchStationStructure(buildStationStructureRequest(cleanJobId));
      if (serial !== stationStructureRequestSerial) return;
      state.stationStructure = payload;
      state.stationStructureError = null;
    } catch (error) {
      if (serial !== stationStructureRequestSerial) return;
      state.stationStructure = null;
      state.stationStructureError = error instanceof Error ? error.message : String(error);
    } finally {
      if (serial === stationStructureRequestSerial) {
        state.stationStructureLoading = false;
        render();
      }
    }
  }

  async function loadGatherPreview() {
    const { payload, errors } = buildGatherPreviewRequest();
    if (errors.length) {
      state.gatherError = errors.join(' ');
      state.gatherPreview = null;
      render();
      return;
    }

    const serial = ++gatherRequestSerial;
    state.gatherLoading = true;
    state.gatherError = null;
    render();

    try {
      const preview = await qcApi.fetchGatherPreview(payload);
      if (serial !== gatherRequestSerial) return;
      state.gatherPreview = preview;
      state.gatherError = null;
    } catch (error) {
      if (serial !== gatherRequestSerial) return;
      state.gatherPreview = null;
      state.gatherError = error instanceof Error ? error.message : String(error);
    } finally {
      if (serial === gatherRequestSerial) {
        state.gatherLoading = false;
        render();
      }
    }
  }

  async function loadBundle() {
    if (!dom) return;
    const jobId = String(dom.jobId.value || '').trim();
    const maxPoints = parsePositiveInteger(dom.maxPoints.value, DEFAULT_MAX_POINTS);
    const previousBundleJobId = String(state.qcBundle?.job_id || '').trim();
    const jobChanged = Boolean(previousBundleJobId && jobId && previousBundleJobId !== jobId);
    state.selectedJobId = jobId;
    state.maxPoints = maxPoints;
    state.selectedFirstBreakPick = null;
    clearSelectedObject();
    state.firstBreakDrilldown = null;
    state.firstBreakDrilldownError = null;
    state.firstBreakDrilldownLoading = false;
    state.qcDrilldown = null;
    state.qcDrilldownError = null;
    state.qcDrilldownLoading = false;
    state.qcDrilldownTarget = null;
    qcDrilldownRequestSerial += 1;
    if (jobChanged) resetJobScopedFilters();
    if (!jobId) {
      state.error = 'Job ID is required.';
      state.qcBundle = null;
      render();
      return null;
    }

    const serial = ++requestSerial;
    state.loading = true;
    state.error = null;
    render();

    try {
      const bundle = await qcApi.fetchQcBundle({
        jobId,
        include: INCLUDE_ALL,
        maxPoints,
      });
      if (serial !== requestSerial) return;
      const bundleJobId = String(bundle.job_id || '').trim();
      const currentPickMapJobId = String(state.pickMap?.job_id || '').trim();
      const sameJobCompletedPickMap = state.pickMap?.mode === 'completed_job' && currentPickMapJobId === bundleJobId;
      const stalePickMap = state.pickMap && !sameJobCompletedPickMap;
      const staleInFlightPickMap = state.pickMapLoading;
      if (stalePickMap || staleInFlightPickMap) {
        pickMapRequestSerial += 1;
        if (stalePickMap || !state.pickMap) {
          state.pickMap = null;
          state.pickMapDisplayMode = 'before';
        }
        state.pickMapLoading = false;
        state.pickMapError = null;
      }
      state.qcBundle = bundle;
      syncGatherContextFromBundle(bundle);
      if (previousBundleJobId && bundleJobId && previousBundleJobId !== bundleJobId) {
        resetJobScopedFilters();
        syncGatherContextFromBundle(bundle);
      }
      if (state.gatherPreview && state.gatherPreview.job_id !== bundle.job_id) {
        state.gatherPreview = null;
      }
      if (state.stationStructure && state.stationStructure.job_id !== bundle.job_id) {
        stationStructureRequestSerial += 1;
        state.stationStructure = null;
        state.stationStructureLoading = false;
        state.stationStructureError = null;
      }
      state.gatherError = null;
      state.error = null;
      writeRecentJob(jobId);
      renderRecentJobs();
      writeJobIdUrlParam(jobId);
      if (isPickMapView(state.selectedView) && !state.pickMap) {
        loadCompletedPickMap(bundleJobId);
      }
      if (state.selectedView === 'station_structure' && !state.stationStructure) {
        loadStationStructureQc(bundleJobId);
      }
      return bundle;
    } catch (error) {
      if (serial !== requestSerial) return;
      state.qcBundle = null;
      state.error = error instanceof Error ? error.message : String(error);
      return null;
    } finally {
      if (serial === requestSerial) {
        state.loading = false;
        render();
      }
    }
  }

  export async function loadJob(jobId, options = {}) {
    if (!dom) return null;
    const safeJobId = String(jobId || '').trim();
    if (safeJobId) {
      dom.jobId.value = safeJobId;
      state.selectedJobId = safeJobId;
    }
    if (options.maxPoints !== undefined && dom.maxPoints) {
      const maxPoints = parsePositiveInteger(options.maxPoints, DEFAULT_MAX_POINTS);
      dom.maxPoints.value = String(maxPoints);
      state.maxPoints = maxPoints;
    }
    if (options.activateTab !== false) {
      activateSidebarTab('refraction_qc');
    }
    return loadBundle();
  }

  export function initRefractionQcPage() {
    resetState();
    requestSerial = 0;
    gatherRequestSerial = 0;
    qcDrilldownRequestSerial = 0;
    pickMapRequestSerial = 0;
    stationStructureRequestSerial = 0;
    getRenderRuntime().reset();
    getRenderRuntime().setDom(null);
    exposeRefractionQcPublicApi();
    const collectedDom = collectRefractionQcDom();
    const { form } = collectedDom;
    if (!form) return;

    dom = collectedDom;
    getRenderRuntime().setDom(dom);

    if (!dom.jobId || !dom.maxPoints || !dom.loadButton || !dom.status || !dom.error || !dom.sign) return;
    if (!dom.jobSummary || !dom.inspector) return;
    if (!dom.activeFilters || !dom.viewControls) return;

    if (dom.pipelineTab) dom.pipelineTab.addEventListener('click', () => activateSidebarTab('pipeline'));
    if (dom.staticCorrectionTab) dom.staticCorrectionTab.addEventListener('click', () => activateSidebarTab('static_correction'));
    if (dom.qcTab) dom.qcTab.addEventListener('click', () => activateSidebarTab('refraction_qc'));
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      loadBundle();
    });
    dom.jobId.addEventListener('input', () => {
      state.selectedJobId = dom.jobId.value.trim();
    });
    dom.maxPoints.addEventListener('input', () => {
      state.maxPoints = parsePositiveInteger(dom.maxPoints.value, DEFAULT_MAX_POINTS);
    });
    for (const button of dom.taskButtons) {
      button.addEventListener('click', () => setActiveTask(button.dataset.task));
    }
    for (const button of dom.viewButtons) {
      button.addEventListener('click', () => setSelectedView(button.dataset.view));
    }

    const params = new URLSearchParams(window.location.search || '');
    const jobId = params.get('refraction_job_id') || params.get('refraction_qc_job_id') || '';
    if (jobId) {
      state.selectedJobId = jobId;
      dom.jobId.value = jobId;
    }
    state.gatherFileId = searchOrStorageValue('file_id', 'file_id', state.gatherFileId);
    state.gatherKey1Byte = searchOrStorageValue('key1_byte', 'key1_byte', state.gatherKey1Byte);
    state.gatherKey2Byte = searchOrStorageValue('key2_byte', 'key2_byte', state.gatherKey2Byte);
    state.gatherSectionKey1 = searchParamValue('key1') || state.gatherSectionKey1;
    state.gatherX0 = searchParamValue('x0') || state.gatherX0;
    state.gatherX1 = searchParamValue('x1') || state.gatherX1;

    renderRecentJobs();
    render();
    restoreCachedPickMapSource();
    if (jobId && isStandaloneRefractionQcPage()) {
      loadBundle();
    }
  }

  function exposeRefractionQcPublicApi() {
    window.refractionQcState = state;
    window.RefractionQc = {
      loadJob,
    };
    window.refractionQcUI = {
      loadBundle,
      loadJob,
      loadGatherPreview,
      loadStationStructureQc,
      setSelectedView,
      activateSidebarTab,
      pickMapGatherNumber,
    };
  }

  exposeRefractionQcPublicApi();
