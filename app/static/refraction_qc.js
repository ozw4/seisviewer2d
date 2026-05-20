(function () {
  const RECENT_JOBS_KEY = 'sv.refraction_qc.recent_jobs';
  const ACTIVE_VIEWER_TARGET_STORAGE_KEY = 'sv.active_viewer_target';
  const STATIC_DRAFT_STORAGE_KEY = 'sv.static_correction.form_draft.v1';
  const STATIC_PICK_DB_NAME = 'seisviewer2d-static-correction';
  const STATIC_PICK_DB_VERSION = 1;
  const STATIC_PICK_STORE = 'pick_npz_blobs';
  const MAX_RECENT_JOBS = 8;
  const DEFAULT_MAX_POINTS = 20000;
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

  const VIEW_DEFS = [
    {
      id: 'summary',
      include: 'summary',
      viewKeys: [],
      unavailableKeys: ['summary'],
    },
    {
      id: 'first_break_residuals',
      include: 'first_break',
      viewKeys: ['first_break_fit', 'first_break_residual'],
      unavailableKeys: ['first_break'],
    },
    {
      id: 'reduced_time',
      include: 'reduced_time',
      viewKeys: ['reduced_time'],
      unavailableKeys: ['reduced_time'],
    },
    {
      id: 'profiles_2d',
      include: 'profiles',
      viewKeys: ['line_profiles'],
      unavailableKeys: ['profiles'],
    },
    {
      id: 'cell_maps_3d',
      include: 'cells',
      viewKeys: ['refraction_grid_map_qc'],
      unavailableKeys: ['cells'],
    },
    {
      id: 'static_components',
      include: 'static_components',
      viewKeys: ['static_components'],
      unavailableKeys: ['static_components'],
    },
    {
      id: 'pick_map',
      include: 'summary',
      viewKeys: [],
      unavailableKeys: [],
    },
    {
      id: 'offset_time',
      include: 'summary',
      viewKeys: [],
      unavailableKeys: [],
    },
    {
      id: 'station_structure',
      include: 'summary',
      viewKeys: [],
      unavailableKeys: [],
    },
    {
      id: 'gather_preview',
      include: 'gather_preview',
      viewKeys: [],
      unavailableKeys: ['gather_preview'],
    },
  ];

  const INCLUDE_ALL = Array.from(new Set(VIEW_DEFS.map((view) => view.include)));

  const state = {
    selectedJobId: '',
    qcBundle: null,
    selectedView: 'summary',
    selectedFirstBreakPick: null,
    firstBreakDrilldown: null,
    firstBreakDrilldownLoading: false,
    firstBreakDrilldownError: null,
    selectedLayerKind: 'all',
    firstBreakXAxis: 'offset',
    showRejectedFirstBreaks: true,
    selectedEndpointKind: 'source',
    selectedCell: null,
    selectedCellMapQuantity: 'velocity',
    selectedEndpoint: '',
    selectedTraceIndex: '',
    selectedProfileGroup: 'time_terms',
    selectedProfileUnits: 'auto',
    profileStatusFilter: 'all',
    maxPoints: DEFAULT_MAX_POINTS,
    gatherAxis: 'source',
    gatherDisplayMode: 'side_by_side',
    gatherEndpointKey: '',
    gatherEndpointSearch: '',
    gatherFileId: '',
    gatherKey1Byte: '',
    gatherKey2Byte: '',
    gatherSectionKey1: '',
    gatherX0: '',
    gatherX1: '',
    gatherTimeStartS: '0',
    gatherTimeEndS: '1',
    gatherMaxTraces: 120,
    gatherReductionVelocity: '1500',
    gatherPreview: null,
    gatherLoading: false,
    gatherError: null,
    pickMap: null,
    pickMapLoading: false,
    pickMapError: null,
    pickMapDisplayMode: 'before',
    pickMapGatherStart: '',
    pickMapGatherEnd: '',
    pickMapCachedFile: null,
    pickMapCachedMeta: null,
    pickMapCacheStatus: '',
    stationStructure: null,
    stationStructureLoading: false,
    stationStructureError: null,
    stationStructureGatherStart: '',
    stationStructureGatherEnd: '',
    stationStructureVelocityField: 'auto',
    stationStructureDepthField: 'auto',
    error: null,
    loading: false,
  };

  let dom = null;
  let requestSerial = 0;
  let gatherRequestSerial = 0;
  let pickMapRequestSerial = 0;
  let stationStructureRequestSerial = 0;
  let pickMapCanvasCleanup = null;
  let stationStructureCanvasCleanup = null;

  const PICK_MAP_VIEWS = {
    pick_map: {
      label: 'RecNo-time',
      xField: 'receiver_number',
      xAxisTitle: 'Global receiver number',
      testIdPrefix: 'refraction-qc-pick-map',
      emptyMessage: 'No Pick Map records match the current gather range.',
      colorByOffset: true,
    },
    offset_time: {
      label: 'Offset-time',
      xField: 'offset_m',
      xAxisTitle: 'Offset (m)',
      testIdPrefix: 'refraction-qc-offset-time',
      emptyMessage: 'No Offset-time records are available because offset_m is missing or non-finite for the selected gather range.',
      colorByOffset: false,
    },
  };

  function safeLocalStorageValue(key) {
    try {
      return localStorage.getItem(key) || '';
    } catch (_) {
      return '';
    }
  }

  function safeLocalStorageJson(key) {
    try {
      const raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : null;
    } catch (_) {
      return null;
    }
  }

  function searchParamValue(key) {
    try {
      const params = new URLSearchParams(window.location.search || '');
      return params.get(key) || '';
    } catch (_) {
      return '';
    }
  }

  function searchOrStorageValue(searchKey, storageKey, fallback = '') {
    return searchParamValue(searchKey) || safeLocalStorageValue(storageKey || searchKey) || fallback;
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

  function isStandaloneRefractionQcPage() {
    return Boolean(document.body && document.body.classList.contains('refraction-qc-page'));
  }

  function plotHeight(compactHeight, standaloneHeight) {
    return isStandaloneRefractionQcPage() ? standaloneHeight : compactHeight;
  }

  const LAYER_LABELS = {
    v1_direct_arrival: 'V1 direct',
    v2_t1: 'V2/T1',
    v3_t2: 'V3/T2',
    vsub_t3: 'Vsub/T3',
  };

  const LAYER_COLORS = {
    v1_direct_arrival: '#7c3aed',
    v2_t1: '#2563eb',
    v3_t2: '#059669',
    vsub_t3: '#c2410c',
    unknown: '#64748b',
  };

  const REDUCED_TIME_GATE_KINDS = [
    'v1_direct_arrival',
    'v2_t1',
    'v3_t2',
    'vsub_t3',
  ];

  const FIRST_BREAK_X_AXES = {
    offset: {
      label: 'Offset (m)',
      columns: ['offset_m'],
    },
    inline: {
      label: 'Inline distance (m)',
      columns: ['inline_m', 'inline_distance_m'],
    },
    trace: {
      label: 'Trace index',
      columns: ['trace_index_sorted', 'sorted_trace_index', 'observation_index'],
    },
  };

  const PROFILE_SERIES_COLORS = {
    t1: '#2563eb',
    t2: '#059669',
    t3: '#c2410c',
    v1_m_s: '#7c3aed',
    v2_m_s: '#2563eb',
    v3_m_s: '#059669',
    vsub_m_s: '#c2410c',
    sh1_m: '#0891b2',
    sh2_m: '#65a30d',
    sh3_m: '#b45309',
    layer1_base_elevation_m: '#475569',
    layer2_base_elevation_m: '#64748b',
    final_refractor_elevation_m: '#0f172a',
    weathering_correction: '#2563eb',
    elevation_correction: '#0891b2',
    field_shift: '#059669',
    manual_static_shift: '#c2410c',
    total_applied_shift: '#be123c',
    pick_fold: '#2563eb',
    used_pick_fold: '#059669',
    residual_rms: '#c2410c',
    residual_mad: '#7c3aed',
  };

  const PROFILE_GROUPS = {
    time_terms: {
      label: 'Time terms',
      axisLabel: 'Time term',
      unit: 'ms',
      series: [
        { key: 't1', columns: ['t1_ms'], label: 'T1', layers: ['v2_t1'] },
        { key: 't2', columns: ['t2_ms'], label: 'T2', layers: ['v3_t2'] },
        { key: 't3', columns: ['t3_ms'], label: 'T3', layers: ['vsub_t3'] },
      ],
    },
    velocities: {
      label: 'Velocities',
      axisLabel: 'Velocity',
      unit: 'm/s',
      series: [
        { key: 'v1_m_s', columns: ['v1_m_s'], label: 'V1', layers: ['v1_direct_arrival'] },
        { key: 'v2_m_s', columns: ['v2_m_s'], label: 'V2', layers: ['v2_t1'] },
        { key: 'v3_m_s', columns: ['v3_m_s'], label: 'V3', layers: ['v3_t2'] },
        { key: 'vsub_m_s', columns: ['vsub_m_s'], label: 'Vsub', layers: ['vsub_t3'] },
      ],
    },
    thickness_elevation: {
      label: 'Thickness / elevations',
      axisLabel: 'Thickness / elevation',
      unit: 'm',
      series: [
        { key: 'sh1_m', columns: ['sh1_weathering_thickness_m', 'sh1_m'], label: 'SH1', layers: ['v2_t1'] },
        { key: 'sh2_m', columns: ['sh2_weathering_thickness_m', 'sh2_m'], label: 'SH2', layers: ['v3_t2'] },
        { key: 'sh3_m', columns: ['sh3_weathering_thickness_m', 'sh3_m'], label: 'SH3', layers: ['vsub_t3'] },
        { key: 'layer1_base_elevation_m', columns: ['layer1_base_elevation_m'], label: 'Layer 1 base', layers: ['v2_t1'] },
        { key: 'layer2_base_elevation_m', columns: ['layer2_base_elevation_m'], label: 'Layer 2 base', layers: ['v3_t2', 'vsub_t3'] },
        { key: 'final_refractor_elevation_m', columns: ['final_refractor_elevation_m'], label: 'Final refractor', layers: ['v2_t1', 'v3_t2', 'vsub_t3'] },
      ],
    },
    statics: {
      label: 'Static components',
      axisLabel: 'Static shift',
      unit: 'ms',
      series: [
        { key: 'weathering_correction', columns: ['weathering_correction_ms'], label: 'Weathering correction' },
        { key: 'elevation_correction', columns: ['elevation_correction_ms'], label: 'Elevation / datum correction' },
        {
          key: 'field_shift',
          endpointColumns: {
            source: ['source_field_shift_ms'],
            receiver: ['receiver_field_shift_ms'],
          },
          label: 'Field correction',
        },
        { key: 'manual_static_shift', columns: ['manual_static_shift_ms'], label: 'Manual static' },
        {
          key: 'total_applied_shift',
          columns: ['total_applied_shift_ms'],
          label: 'Final applied static',
        },
      ],
    },
    qc_metrics: {
      label: 'QC metrics',
      axisLabel: 'QC metric',
      unit: 'mixed',
      series: [
        { key: 'pick_fold', columns: ['pick_count'], label: 'Pick fold', unit: 'count' },
        { key: 'used_pick_fold', columns: ['used_pick_count'], label: 'Used pick fold', unit: 'count' },
        { key: 'residual_rms', columns: ['residual_rms_ms'], label: 'Residual RMS', unit: 'ms' },
        { key: 'residual_mad', columns: ['residual_mad_ms'], label: 'Residual MAD', unit: 'ms' },
      ],
    },
  };

  const PROFILE_STATUS_OK = new Set([
    '',
    'ok',
    'valid',
    'used',
    'solved',
    'computed',
    'available',
    'success',
    'none',
  ]);

  const COMPONENT_STATUS_OK = new Set([
    ...PROFILE_STATUS_OK,
    'not_enabled',
    'not_applicable',
  ]);

  const STATIC_ENDPOINT_VIEW_KEY = 'static_component_qc_endpoint';
  const STATIC_TRACE_VIEW_KEY = 'static_component_qc_trace';

  const STATIC_COMPONENT_DEFS = [
    {
      key: 'weathering',
      label: 'Weathering correction',
      columns: ['weathering_correction_ms'],
      statusColumns: ['weathering_status', 'solution_status'],
    },
    {
      key: 'datum',
      label: 'Datum / elevation correction',
      columns: ['elevation_correction_ms'],
      statusColumns: ['datum_status'],
    },
    {
      key: 'source_depth',
      label: 'Source-depth correction',
      columns: ['source_depth_correction_ms'],
      statusColumns: ['source_depth_status'],
    },
    {
      key: 'uphole',
      label: 'Uphole correction',
      columns: ['uphole_correction_ms'],
      statusColumns: ['uphole_status'],
    },
    {
      key: 'manual',
      label: 'Manual static',
      columns: ['manual_static_shift_ms'],
      statusColumns: ['manual_static_status'],
    },
    {
      key: 'computed_field',
      label: 'Computed field correction',
      columns: ['computed_field_correction_ms'],
      statusEndpointColumns: {
        source: ['source_field_static_status', 'source_field_status'],
        receiver: ['receiver_field_static_status', 'receiver_field_status'],
      },
    },
    {
      key: 'applied_field',
      label: 'Applied field correction',
      columns: ['applied_field_correction_ms'],
      statusEndpointColumns: {
        source: ['source_field_static_status', 'source_field_status'],
        receiver: ['receiver_field_static_status', 'receiver_field_status'],
      },
    },
    {
      key: 'total',
      label: 'Final endpoint shift',
      columns: ['total_applied_shift_ms'],
      statusColumns: ['static_status'],
    },
    {
      key: 'total_with_field',
      label: 'Total with computed field shift',
      endpointColumns: {
        source: ['source_total_with_field_shift_ms'],
        receiver: ['receiver_total_with_field_shift_ms'],
      },
      statusColumns: ['static_status'],
    },
  ];

  const STATIC_TRACE_COMPONENT_DEFS = [
    {
      key: 'refraction',
      label: 'Refraction shift',
      columns: ['refraction_trace_shift_ms'],
    },
    {
      key: 'weathering',
      label: 'Weathering shift',
      columns: ['weathering_shift_ms'],
    },
    {
      key: 'datum',
      label: 'Datum shift',
      columns: ['datum_shift_ms'],
    },
    {
      key: 'computed_field',
      label: 'Computed field shift',
      columns: ['computed_field_shift_ms'],
      statusColumns: ['trace_field_static_status'],
    },
    {
      key: 'applied_field',
      label: 'Applied field shift',
      columns: ['applied_field_shift_ms'],
      statusColumns: ['trace_field_static_status'],
    },
    {
      key: 'manual',
      label: 'Manual static shift',
      columns: ['manual_static_shift_ms'],
    },
    {
      key: 'source_depth',
      label: 'Source-depth shift',
      columns: ['source_depth_shift_ms'],
    },
    {
      key: 'uphole',
      label: 'Uphole shift',
      columns: ['uphole_shift_ms'],
    },
    {
      key: 'final',
      label: 'Final applied trace shift',
      columns: ['final_trace_shift_ms'],
    },
  ];

  const STATIC_COMPONENT_COLORS = {
    positive: '#2563eb',
    negative: '#be123c',
    zero: '#64748b',
  };

  const CELL_MAP_LAYER_ORDER = ['v2_t1', 'v3_t2', 'vsub_t3'];

  const CELL_MAP_QUANTITIES = {
    velocity: {
      label: 'Velocity',
      axisLabel: 'Velocity (m/s)',
      unit: 'm/s',
      columns: ['velocity_m_s', 'v2_m_s'],
      colorscale: 'Viridis',
    },
    velocity_update: {
      label: 'Velocity update from initial',
      axisLabel: 'Velocity update (m/s)',
      unit: 'm/s',
      columns: ['velocity_update_from_initial_m_s', 'v2_update_from_initial_m_s'],
      colorscale: 'RdBu',
      reversescale: true,
    },
    fold: {
      label: 'Fold / observation count',
      axisLabel: 'Observation count',
      unit: 'count',
      columns: ['n_observations', 'fold', 'observation_count'],
      colorscale: 'YlGnBu',
    },
    residual_rms: {
      label: 'Residual RMS',
      axisLabel: 'Residual RMS (ms)',
      unit: 'ms',
      columns: ['residual_rms_ms'],
      secondColumns: ['residual_rms_s'],
      scale: 1000.0,
      colorscale: 'YlOrRd',
    },
    residual_mad: {
      label: 'Residual MAD',
      axisLabel: 'Residual MAD (ms)',
      unit: 'ms',
      columns: ['residual_mad_ms'],
      secondColumns: ['residual_mad_s'],
      scale: 1000.0,
      colorscale: 'YlOrRd',
    },
    status: {
      label: 'Status',
      axisLabel: 'Status',
      unit: '',
      status: true,
    },
  };

  const CELL_STATUS_COLORS = {
    solved: '#2563eb',
    ok: '#2563eb',
    active: '#2563eb',
    low_fold: '#f59e0b',
    inactive: '#cbd5e1',
    no_observations: '#cbd5e1',
    empty: '#cbd5e1',
    rejected: '#dc2626',
    unknown: '#64748b',
  };

  const GATHER_DISPLAY_LABELS = {
    raw: 'Raw only',
    corrected: 'Corrected only',
    side_by_side: 'Raw and corrected',
    reduced_time: 'Reduced-time / LMO',
  };

  const GATHER_AXIS_LABELS = {
    source: 'Source',
    receiver: 'Receiver',
    section: 'Midpoint/CMP window',
  };

  function readRecentJobs() {
    try {
      const raw = localStorage.getItem(RECENT_JOBS_KEY);
      const parsed = JSON.parse(raw || '[]');
      if (!Array.isArray(parsed)) return [];
      return parsed.filter((item) => typeof item === 'string' && item).slice(0, MAX_RECENT_JOBS);
    } catch (_) {
      return [];
    }
  }

  function writeRecentJob(jobId) {
    const cleanJobId = String(jobId || '').trim();
    if (!cleanJobId) return;
    const recent = readRecentJobs().filter((item) => item !== cleanJobId);
    recent.unshift(cleanJobId);
    try {
      localStorage.setItem(RECENT_JOBS_KEY, JSON.stringify(recent.slice(0, MAX_RECENT_JOBS)));
    } catch (_) {
    }
    renderRecentJobs();
  }

  function parsePositiveInteger(value, fallback) {
    const parsed = Number.parseInt(String(value), 10);
    if (!Number.isFinite(parsed) || parsed < 1) return fallback;
    return parsed;
  }

  function parseCell(value) {
    const text = String(value || '').trim();
    if (!text) return null;
    const parts = text.split(/[,\s]+/).filter(Boolean);
    if (parts.length !== 2) return { text };
    const ix = Number.parseInt(parts[0], 10);
    const iy = Number.parseInt(parts[1], 10);
    if (!Number.isInteger(ix) || !Number.isInteger(iy) || ix < 0 || iy < 0) {
      return { text };
    }
    return { cell_ix: ix, cell_iy: iy };
  }

  function selectedCellLabel(cell) {
    if (!cell) return '-';
    if (cell.cell_ix !== undefined && cell.cell_iy !== undefined) {
      const layer = cell.layer_kind ? ` ${layerLabel(cell.layer_kind)}` : '';
      return `${cell.cell_ix},${cell.cell_iy}${layer}`;
    }
    return textOrDash(cell.text);
  }

  function textOrDash(value) {
    if (value === null || value === undefined || value === '') return '-';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  }

  function firstDefined(record, columns) {
    if (!record || typeof record !== 'object') return undefined;
    for (const column of columns) {
      const value = record[column];
      if (value !== null && value !== undefined && value !== '') return value;
    }
    return undefined;
  }

  function toFiniteNumber(value) {
    if (typeof value === 'number') return Number.isFinite(value) ? value : NaN;
    if (typeof value !== 'string') return NaN;
    const parsed = Number.parseFloat(value.trim());
    return Number.isFinite(parsed) ? parsed : NaN;
  }

  function pickMapGatherNumber(value) {
    if (value === null || value === undefined) return NaN;
    const text = String(value).trim();
    const direct = Number(text);
    if (Number.isFinite(direct)) return direct;
    const tail = text
      .split(':')
      .find((part, index) => index > 0 && Number.isFinite(Number(part)));
    return tail === undefined ? NaN : Number(tail);
  }

  function normalizedText(value) {
    return String(value ?? '').trim().toLowerCase();
  }

  function formatNumber(value, digits = 3) {
    if (!Number.isFinite(value)) return '-';
    return value.toFixed(digits);
  }

  function layerLabel(layerKind) {
    return LAYER_LABELS[layerKind] || (layerKind ? String(layerKind) : 'Unlayered');
  }

  function firstBreakLayerMatches(record) {
    if (state.selectedLayerKind === 'all') return true;
    const layerKind = String(firstDefined(record, ['layer_kind']) || '').trim();
    return !layerKind || layerKind === state.selectedLayerKind;
  }

  function isRejectedFirstBreakRecord(record) {
    const usedValue = firstDefined(record, ['used_in_solve', 'used_for_inversion']);
    const usedText = normalizedText(usedValue);
    if (usedText === 'false' || usedText === '0' || usedText === 'no') return true;
    if (usedValue === false) return true;

    const statusText = normalizedText(firstDefined(record, ['status']));
    if (statusText === 'rejected' || statusText === 'unused') return true;

    const rejectText = normalizedText(firstDefined(record, ['reject_reason', 'rejection_reason']));
    return Boolean(rejectText && rejectText !== 'ok' && rejectText !== 'none' && rejectText !== 'nan');
  }

  function isUnusedReducedTimeRecord(record) {
    const usedValue = firstDefined(record, ['used_for_inversion']);
    const usedText = normalizedText(usedValue);
    if (usedText === 'false' || usedText === '0' || usedText === 'no') return true;
    if (usedValue === false) return true;

    const statusText = normalizedText(firstDefined(record, ['status']));
    if (statusText === 'rejected' || statusText === 'unused') return true;

    const rejectText = normalizedText(firstDefined(record, ['reject_reason', 'rejection_reason']));
    return Boolean(rejectText && rejectText !== 'ok' && rejectText !== 'none' && rejectText !== 'nan');
  }

  function firstBreakXAxisDefinition() {
    return FIRST_BREAK_X_AXES[state.firstBreakXAxis] || FIRST_BREAK_X_AXES.offset;
  }

  function reducedTimeLayerKind(record) {
    return String(firstDefined(record, ['layer_kind', 'layer_gate_kind']) || '').trim() || 'unknown';
  }

  function reducedTimeLayerMatches(record) {
    if (state.selectedLayerKind === 'all') return true;
    return reducedTimeLayerKind(record) === state.selectedLayerKind;
  }

  function finiteReducedTimeMs(record) {
    const reducedS = toFiniteNumber(firstDefined(record, ['reduced_time_s']));
    if (Number.isFinite(reducedS)) return reducedS * 1000.0;
    return toFiniteNumber(firstDefined(record, ['reduced_time_ms']));
  }

  function normalizeReducedTimeRecord(record) {
    const xAxis = firstBreakXAxisDefinition();
    const x = toFiniteNumber(firstDefined(record, xAxis.columns));
    if (!Number.isFinite(x)) return null;

    const reducedMs = finiteReducedTimeMs(record);
    const observedS = toFiniteNumber(firstDefined(record, [
      'observed_time_s',
      'observed_first_break_time_s',
    ]));
    const velocity = toFiniteNumber(firstDefined(record, ['reduction_velocity_m_s']));
    const statusText = normalizedText(firstDefined(record, ['status'])) || 'ok';
    const used = !isUnusedReducedTimeRecord(record);
    const layerKind = reducedTimeLayerKind(record);
    const unavailableReason = Number.isFinite(reducedMs)
      ? ''
      : (statusText && statusText !== 'ok' ? statusText : 'missing_reduced_time');

    return {
      x,
      reducedMs,
      observedMs: Number.isFinite(observedS) ? observedS * 1000.0 : NaN,
      reductionVelocity: velocity,
      layerKind,
      used,
      status: unavailableReason ? 'unavailable' : (used ? 'used' : 'unused'),
      rawStatus: statusText,
      unavailableReason,
      traceIndex: textOrDash(firstDefined(record, ['trace_index_sorted', 'sorted_trace_index'])),
      source: textOrDash(firstDefined(record, ['source_endpoint_key', 'source_id'])),
      receiver: textOrDash(firstDefined(record, ['receiver_endpoint_key', 'receiver_id'])),
    };
  }

  function filteredReducedTimePoints(view) {
    const records = Array.isArray(view.records) ? view.records : [];
    const points = [];
    for (const record of records) {
      if (!reducedTimeLayerMatches(record)) continue;
      if (!state.showRejectedFirstBreaks && isUnusedReducedTimeRecord(record)) continue;
      const point = normalizeReducedTimeRecord(record);
      if (point) points.push(point);
    }
    return points;
  }

  function normalizeFirstBreakRecord(record) {
    const xAxis = firstBreakXAxisDefinition();
    const x = toFiniteNumber(firstDefined(record, xAxis.columns));
    const observedS = toFiniteNumber(firstDefined(record, [
      'observed_first_break_time_s',
      'observed_time_s',
    ]));
    const modeledS = toFiniteNumber(firstDefined(record, [
      'modeled_first_break_time_s',
      'modeled_time_s',
    ]));
    if (!Number.isFinite(x) || !Number.isFinite(observedS) || !Number.isFinite(modeledS)) {
      return null;
    }

    const layerKind = String(firstDefined(record, ['layer_kind']) || '').trim() || 'unknown';
    const rejected = isRejectedFirstBreakRecord(record);
    const sourceEndpointKey = String(firstDefined(record, ['source_endpoint_key']) || '').trim();
    const receiverEndpointKey = String(firstDefined(record, ['receiver_endpoint_key']) || '').trim();
    const traceIndex = textOrDash(firstDefined(record, ['trace_index_sorted', 'sorted_trace_index']));
    return {
      x,
      observedMs: observedS * 1000.0,
      modeledMs: modeledS * 1000.0,
      residualMs: (observedS - modeledS) * 1000.0,
      layerKind,
      status: rejected ? 'rejected' : 'used',
      opacity: rejected ? 0.42 : 0.9,
      traceIndex,
      sourceEndpointKey,
      receiverEndpointKey,
      source: textOrDash(sourceEndpointKey || firstDefined(record, ['source_id'])),
      receiver: textOrDash(receiverEndpointKey || firstDefined(record, ['receiver_id'])),
    };
  }

  function filteredFirstBreakPoints(view) {
    const records = Array.isArray(view.records) ? view.records : [];
    const points = [];
    for (const record of records) {
      if (!firstBreakLayerMatches(record)) continue;
      if (!state.showRejectedFirstBreaks && isRejectedFirstBreakRecord(record)) continue;
      const point = normalizeFirstBreakRecord(record);
      if (point) points.push(point);
    }
    return points;
  }

  function profileGroupDefinition() {
    return PROFILE_GROUPS[state.selectedProfileGroup] || PROFILE_GROUPS.time_terms;
  }

  function profileSeriesMatchesLayer(series) {
    if (state.selectedLayerKind === 'all') return true;
    if (!Array.isArray(series.layers) || !series.layers.length) return true;
    return series.layers.includes(state.selectedLayerKind);
  }

  function profileSeriesUnit(series, group) {
    return series.unit || group.unit || '';
  }

  function profileDisplayUnit(series, group) {
    const unit = profileSeriesUnit(series, group);
    if (unit === 'ms' && state.selectedProfileUnits === 's') return 's';
    return unit;
  }

  function profileDisplayValue(value, series, group) {
    const unit = profileSeriesUnit(series, group);
    if (!Number.isFinite(value)) return NaN;
    if (unit === 'ms' && state.selectedProfileUnits === 's') return value / 1000.0;
    return value;
  }

  function profileSeriesRawValue(record, series) {
    const value = toFiniteNumber(firstDefined(record, columnsForRecord(record, series)));
    if (!Number.isFinite(value)) return NaN;
    return value * (series.scale || 1.0);
  }

  function profileAxisTitle(group, seriesList) {
    if (group.unit === 'mixed') {
      const hasTime = seriesList.some((series) => profileSeriesUnit(series, group) === 'ms');
      const timeUnit = state.selectedProfileUnits === 's' ? 's' : 'ms';
      return hasTime ? `${group.axisLabel} (count or ${timeUnit})` : `${group.axisLabel} (count)`;
    }
    if (group.unit === 'ms') {
      return `${group.axisLabel} (${state.selectedProfileUnits === 's' ? 's' : 'ms'})`;
    }
    return group.unit ? `${group.axisLabel} (${group.unit})` : group.axisLabel;
  }

  function profileStatusText(record) {
    const staticStatus = normalizedText(firstDefined(record, ['static_status']));
    const solutionStatus = normalizedText(firstDefined(record, ['solution_status']));
    return { staticStatus, solutionStatus };
  }

  function isInvalidProfileRecord(record) {
    const { staticStatus, solutionStatus } = profileStatusText(record);
    return !PROFILE_STATUS_OK.has(staticStatus) || !PROFILE_STATUS_OK.has(solutionStatus);
  }

  function profileStatusMatches(record) {
    const invalid = isInvalidProfileRecord(record);
    if (state.profileStatusFilter === 'valid') return !invalid;
    if (state.profileStatusFilter === 'invalid') return invalid;
    return true;
  }

  function profileEndpointMatches(record) {
    const kind = normalizedText(firstDefined(record, ['endpoint_kind']));
    if (state.selectedEndpointKind !== 'both' && kind !== state.selectedEndpointKind) return false;
    const endpointFilter = normalizedText(state.selectedEndpoint);
    if (!endpointFilter) return true;
    const endpointKey = normalizedText(firstDefined(record, ['endpoint_key']));
    return endpointKey.includes(endpointFilter);
  }

  function normalizeProfileRecord(record) {
    const inline = toFiniteNumber(firstDefined(record, ['inline_m']));
    if (!Number.isFinite(inline)) return null;
    return {
      raw: record,
      inline,
      endpointKind: normalizedText(firstDefined(record, ['endpoint_kind'])) || 'unknown',
      endpointKey: textOrDash(firstDefined(record, ['endpoint_key'])),
      nodeId: textOrDash(firstDefined(record, ['node_id'])),
      staticStatus: textOrDash(firstDefined(record, ['static_status'])),
      solutionStatus: textOrDash(firstDefined(record, ['solution_status'])),
      invalid: isInvalidProfileRecord(record),
    };
  }

  function filteredProfileRecords(view) {
    const records = Array.isArray(view.records) ? view.records : [];
    const points = [];
    for (const record of records) {
      if (!profileEndpointMatches(record)) continue;
      if (!profileStatusMatches(record)) continue;
      const normalized = normalizeProfileRecord(record);
      if (normalized) points.push(normalized);
    }
    return points.sort((a, b) => (
      a.endpointKind.localeCompare(b.endpointKind)
      || a.inline - b.inline
      || a.endpointKey.localeCompare(b.endpointKey)
    ));
  }

  function profileSeriesAvailable(series, records) {
    return records.some((record) => Number.isFinite(profileSeriesRawValue(record.raw, series)));
  }

  function selectedProfileSeries(records) {
    const group = profileGroupDefinition();
    const layerSeries = group.series.filter(profileSeriesMatchesLayer);
    return {
      group,
      available: layerSeries.filter((series) => profileSeriesAvailable(series, records)),
      unavailable: layerSeries.filter((series) => !profileSeriesAvailable(series, records)),
    };
  }

  function cellMapQuantityDefinition() {
    return CELL_MAP_QUANTITIES[state.selectedCellMapQuantity] || CELL_MAP_QUANTITIES.velocity;
  }

  function cellMapLayerKind(record) {
    return String(firstDefined(record, ['layer_kind', 'cell_velocity_layer_kind']) || '').trim() || 'unknown';
  }

  function finiteMetric(record, columns, secondColumns, scale = 1.0) {
    const direct = toFiniteNumber(firstDefined(record, columns || []));
    if (Number.isFinite(direct)) return direct;
    const seconds = toFiniteNumber(firstDefined(record, secondColumns || []));
    return Number.isFinite(seconds) ? seconds * scale : NaN;
  }

  function normalizeCellMapRecord(record) {
    const cellIx = Number.parseInt(String(firstDefined(record, ['cell_ix', 'ix']) ?? ''), 10);
    const cellIy = Number.parseInt(String(firstDefined(record, ['cell_iy', 'iy']) ?? ''), 10);
    if (!Number.isInteger(cellIx) || !Number.isInteger(cellIy)) return null;

    const centerX = finiteMetric(record, [
      'cell_center_x_m',
      'x_center_m',
      'center_x_m',
      'cell_center_inline_m',
    ]);
    const centerY = finiteMetric(record, [
      'cell_center_y_m',
      'y_center_m',
      'center_y_m',
      'cell_center_crossline_m',
    ]);
    const velocity = finiteMetric(record, ['velocity_m_s', 'v2_m_s']);
    const initialVelocity = finiteMetric(record, ['initial_velocity_m_s', 'initial_v2_m_s']);
    const velocityUpdate = finiteMetric(record, [
      'velocity_update_from_initial_m_s',
      'v2_update_from_initial_m_s',
    ]);
    const fold = finiteMetric(record, ['n_observations', 'fold', 'observation_count']);
    const usedFold = finiteMetric(record, ['n_used_observations', 'used_fold']);
    const rejectedFold = finiteMetric(record, ['n_rejected_observations', 'rejected_fold']);
    const residualRmsMs = finiteMetric(record, ['residual_rms_ms'], ['residual_rms_s'], 1000.0);
    const residualMadMs = finiteMetric(record, ['residual_mad_ms'], ['residual_mad_s'], 1000.0);
    const rawStatus = normalizedText(firstDefined(record, ['status', 'velocity_status'])) || (
      Number.isFinite(fold) && fold <= 0 ? 'no_observations' : 'unknown'
    );

    return {
      raw: record,
      layerKind: cellMapLayerKind(record),
      cellIx,
      cellIy,
      centerX: Number.isFinite(centerX) ? centerX : cellIx,
      centerY: Number.isFinite(centerY) ? centerY : cellIy,
      velocity,
      initialVelocity,
      velocityUpdate,
      fold,
      usedFold,
      rejectedFold,
      residualRmsMs,
      residualMadMs,
      status: rawStatus,
      statusReason: textOrDash(firstDefined(record, ['status_reason', 'velocity_status_reason'])),
      component: textOrDash(firstDefined(record, ['cell_velocity_component'])),
    };
  }

  function filteredCellMapRecords(view) {
    const records = Array.isArray(view.records) ? view.records : [];
    const normalized = [];
    for (const record of records) {
      const point = normalizeCellMapRecord(record);
      if (point) normalized.push(point);
    }
    return normalized;
  }

  function viewByKey(bundle, key) {
    const views = bundle && typeof bundle.views === 'object' && bundle.views ? bundle.views : {};
    const view = views[key];
    return view && typeof view === 'object' ? view : null;
  }

  function staticEndpointKind(record) {
    return normalizedText(firstDefined(record, ['endpoint_kind', 'kind']));
  }

  function staticEndpointKey(record) {
    return String(firstDefined(record, ['endpoint_key']) ?? '').trim();
  }

  function staticTraceIndex(record) {
    return String(firstDefined(record, ['trace_index_sorted', 'sorted_trace_index', 'trace']) ?? '').trim();
  }

  function staticEndpointRecords(view) {
    const records = Array.isArray(view?.records) ? view.records : [];
    return records.filter((record) => staticEndpointKind(record) && staticEndpointKey(record));
  }

  function gatherEndpointStationId(record, endpointKind) {
    const stationValue = firstDefined(record, [
      'station_id',
      `${endpointKind}_station_id`,
      endpointKind === 'source' ? 'shot_point' : 'receiver_point',
      endpointKind === 'source' ? 'source_point' : 'receiver_station',
      'station',
      'point_id',
    ]);
    return String(stationValue ?? '').trim();
  }

  function gatherEndpointStatus(record) {
    return textOrDash(firstDefined(record, [
      'static_status',
      'component_status',
      'solution_status',
      'datum_status',
      'weathering_status',
      'status',
    ]));
  }

  function gatherEndpointCoordinateParts(record) {
    const parts = [];
    const x = toFiniteNumber(firstDefined(record, ['x_m', 'inline_m']));
    const y = toFiniteNumber(firstDefined(record, ['y_m', 'crossline_m']));
    const z = toFiniteNumber(firstDefined(record, ['surface_elevation_m', 'elevation_m', 'z_m']));
    if (Number.isFinite(x)) parts.push(`x=${formatNumber(x, 1)}`);
    if (Number.isFinite(y)) parts.push(`y=${formatNumber(y, 1)}`);
    if (Number.isFinite(z)) parts.push(`z=${formatNumber(z, 1)}`);
    return parts;
  }

  function gatherEndpointLabel(record, endpointKind, duplicateStations) {
    const endpointKey = staticEndpointKey(record);
    const stationId = gatherEndpointStationId(record, endpointKind);
    const prefix = endpointKind === 'receiver' ? 'R' : 'S';
    const title = `${prefix} ${stationId || endpointKey}`;
    const parts = [title];
    const nodeId = String(firstDefined(record, ['node_id']) ?? '').trim();
    if (nodeId) {
      parts.push(`node ${nodeId}`);
    } else if (stationId && duplicateStations.has(stationId)) {
      parts.push(...gatherEndpointCoordinateParts(record));
    }
    const pickCount = toFiniteNumber(firstDefined(record, ['pick_count', 'used_pick_count']));
    if (Number.isFinite(pickCount)) parts.push(`picks ${formatNumber(pickCount, 0)}`);
    const residualRms = toFiniteNumber(firstDefined(record, ['residual_rms_ms']));
    if (Number.isFinite(residualRms)) parts.push(`RMS ${formatNumber(residualRms, 1)} ms`);
    parts.push(gatherEndpointStatus(record));
    return parts.filter(Boolean).join(' · ');
  }

  function buildGatherEndpointOptions(bundle, endpointKind) {
    const componentView = viewByKey(bundle, 'static_components');
    const qcView = viewByKey(bundle, STATIC_ENDPOINT_VIEW_KEY);
    const componentRecords = staticEndpointRecords(componentView)
      .filter((record) => staticEndpointKind(record) === endpointKind);
    const qcByEndpointKey = new Map();
    for (const record of staticEndpointRecords(qcView)) {
      const key = staticEndpointKey(record);
      if (key && !qcByEndpointKey.has(key)) qcByEndpointKey.set(key, record);
    }
    const mergedRecords = componentRecords.map((record) => ({
      ...qcByEndpointKey.get(staticEndpointKey(record)),
      ...record,
    }));
    const stationCounts = new Map();
    for (const record of mergedRecords) {
      const stationId = gatherEndpointStationId(record, endpointKind);
      if (stationId) stationCounts.set(stationId, (stationCounts.get(stationId) || 0) + 1);
    }
    const duplicateStations = new Set(
      Array.from(stationCounts.entries())
        .filter(([, count]) => count > 1)
        .map(([stationId]) => stationId),
    );
    return mergedRecords
      .map((record) => ({
        value: staticEndpointKey(record),
        label: gatherEndpointLabel(record, endpointKind, duplicateStations),
        stationId: gatherEndpointStationId(record, endpointKind),
      }))
      .sort((a, b) => (
        String(a.stationId || '').localeCompare(String(b.stationId || ''), undefined, { numeric: true })
        || a.value.localeCompare(b.value, undefined, { numeric: true })
      ));
  }

  function staticTraceRecords(view) {
    const records = Array.isArray(view?.records) ? view.records : [];
    return records.filter((record) => staticTraceIndex(record));
  }

  function selectedStaticEndpointRecord(records) {
    if (!records.length) return null;
    const selectedKind = state.selectedEndpointKind === 'both'
      ? ''
      : normalizedText(state.selectedEndpointKind);
    const endpointFilter = normalizedText(state.selectedEndpoint);
    const byKind = selectedKind
      ? records.filter((record) => staticEndpointKind(record) === selectedKind)
      : records.slice();
    const candidates = byKind;

    if (endpointFilter) {
      const exact = candidates.find((record) => normalizedText(staticEndpointKey(record)) === endpointFilter);
      if (exact) return exact;
      const partial = candidates.find((record) => normalizedText(staticEndpointKey(record)).includes(endpointFilter));
      if (partial) return partial;
      return null;
    }
    return candidates[0] || null;
  }

  function matchingLegacyStaticRecord(bundle, endpointRecord) {
    const view = viewByKey(bundle, 'static_components');
    const records = Array.isArray(view?.records) ? view.records : [];
    const kind = staticEndpointKind(endpointRecord);
    const key = normalizedText(staticEndpointKey(endpointRecord));
    if (!kind || !key) return null;
    return records.find((record) => (
      staticEndpointKind(record) === kind
      && normalizedText(staticEndpointKey(record)) === key
    )) || null;
  }

  function traceMatchesEndpoint(record, endpointRecord) {
    if (!endpointRecord) return true;
    const kind = staticEndpointKind(endpointRecord);
    const key = staticEndpointKey(endpointRecord);
    if (!kind || !key) return true;
    const column = kind === 'receiver' ? 'receiver_endpoint_key' : 'source_endpoint_key';
    return String(firstDefined(record, [column]) ?? '').trim() === key;
  }

  function selectedStaticTraceRecord(records, endpointRecord) {
    if (!records.length) return null;
    const traceFilter = String(state.selectedTraceIndex || '').trim();
    if (traceFilter) {
      const exact = records.find((record) => staticTraceIndex(record) === traceFilter);
      if (exact) return exact;
    }
    const endpointMatch = records.find((record) => traceMatchesEndpoint(record, endpointRecord));
    return endpointMatch || records[0] || null;
  }

  function staticApplyToTraceShift(record) {
    const raw = firstDefined(record, ['apply_to_trace_shift', 'trace_apply_to_trace_shift']);
    const text = normalizedText(raw);
    if (raw === true || text === 'true' || text === '1' || text === 'yes') return 'true';
    if (raw === false || text === 'false' || text === '0' || text === 'no') return 'false';
    return '-';
  }

  function componentStatus(record, statusRecord, columns, fallbackColumns = ['static_status']) {
    const fromPrimary = firstDefined(record, columns || []);
    if (fromPrimary !== undefined) return textOrDash(fromPrimary);
    const fromStatus = firstDefined(statusRecord, columns || []);
    if (fromStatus !== undefined) return textOrDash(fromStatus);
    const fallback = firstDefined(record, fallbackColumns);
    if (fallback !== undefined) return textOrDash(fallback);
    const statusFallback = firstDefined(statusRecord, fallbackColumns);
    if (statusFallback !== undefined) return textOrDash(statusFallback);
    return 'missing';
  }

  function columnsForRecord(record, definition, key = 'columns', endpointKey = 'endpointColumns') {
    if (!definition[endpointKey]) return definition[key] || [];
    const kind = staticEndpointKind(record);
    return definition[endpointKey][kind] || [];
  }

  function componentColumns(record, definition) {
    return columnsForRecord(record, definition);
  }

  function componentStatusColumns(record, definition) {
    return columnsForRecord(record, definition, 'statusColumns', 'statusEndpointColumns');
  }

  function componentValueMs(record, definition) {
    if (definition.applyToTraceShift && staticApplyToTraceShift(record) === 'false') return 0.0;
    const value = toFiniteNumber(firstDefined(record, componentColumns(record, definition)));
    if (!Number.isFinite(value)) return NaN;
    return value * (definition.scale || 1.0);
  }

  function shiftDirection(value) {
    if (!Number.isFinite(value)) return 'missing';
    if (value > 0) return 'delays displayed events';
    if (value < 0) return 'advances displayed events';
    return 'no shift';
  }

  function componentColor(value) {
    if (!Number.isFinite(value) || value === 0) return STATIC_COMPONENT_COLORS.zero;
    return value > 0 ? STATIC_COMPONENT_COLORS.positive : STATIC_COMPONENT_COLORS.negative;
  }

  function statusIsOk(status) {
    return COMPONENT_STATUS_OK.has(normalizedText(status));
  }

  function buildStaticComponentRows(record, statusRecord, defs, fallbackStatusColumns) {
    if (!record) return [];
    return defs.map((definition) => {
      const value = componentValueMs(record, definition);
      return {
        key: definition.key,
        label: definition.label,
        value,
        valueText: Number.isFinite(value) ? `${formatNumber(value, 3)} ms` : '-',
        direction: shiftDirection(value),
        status: componentStatus(
          record,
          statusRecord,
          componentStatusColumns(record, definition) || fallbackStatusColumns || [],
          fallbackStatusColumns || ['static_status'],
        ),
      };
    });
  }

  function createStaticComponentTable(rows, testId) {
    const wrap = document.createElement('div');
    wrap.className = 'refraction-qc-table-wrap';
    wrap.dataset.testid = testId;
    const table = document.createElement('table');
    table.className = 'refraction-qc-table refraction-qc-component-table';

    const head = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const label of ['Component', 'Shift', 'Effect', 'Status']) {
      const th = document.createElement('th');
      th.textContent = label;
      headRow.appendChild(th);
    }
    head.appendChild(headRow);
    table.appendChild(head);

    const body = document.createElement('tbody');
    for (const row of rows) {
      const tr = document.createElement('tr');
      const cells = [row.label, row.valueText, row.direction, row.status];
      for (let index = 0; index < cells.length; index += 1) {
        const td = document.createElement('td');
        td.textContent = cells[index];
        if (index === 3) {
          td.className = statusIsOk(row.status)
            ? 'refraction-qc-component-status is-ok'
            : 'refraction-qc-component-status is-alert';
        }
        tr.appendChild(td);
      }
      body.appendChild(tr);
    }
    table.appendChild(body);
    wrap.appendChild(table);
    return wrap;
  }

  function renderStaticComponentBars(content, rows, testId, title) {
    const finiteRows = rows.filter((row) => Number.isFinite(row.value));
    if (!finiteRows.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = `No finite ${title.toLowerCase()} values are available to plot.`;
      content.appendChild(missing);
      return;
    }

    const plot = createFirstBreakPlot(testId);
    plot.classList.add('refraction-qc-static-waterfall');
    plot.dataset.pointCount = String(finiteRows.length);
    content.appendChild(plot);

    if (window.Plotly) {
      window.Plotly.newPlot(plot, [
        {
          name: title,
          type: 'bar',
          orientation: 'h',
          y: finiteRows.map((row) => row.label),
          x: finiteRows.map((row) => row.value),
          text: finiteRows.map((row) => `${row.valueText}; ${row.direction}; ${row.status}`),
          hovertemplate: '%{y}<br>%{text}<extra></extra>',
          marker: {
            color: finiteRows.map((row) => componentColor(row.value)),
          },
        },
      ], {
        height: Math.max(plotHeight(260, 460), finiteRows.length * 30 + 90),
        margin: { l: 150, r: 18, t: 32, b: 44 },
        font: { size: 10, color: '#334155' },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        title: { text: title, font: { size: 12 } },
        xaxis: {
          title: { text: 'Shift (ms)' },
          zeroline: true,
          zerolinecolor: '#0f172a',
          gridcolor: '#e5e7eb',
        },
        yaxis: {
          automargin: true,
        },
        showlegend: false,
      }, { displayModeBar: false, responsive: true });
    } else {
      plot.textContent = 'Plot library is unavailable.';
    }
  }

  function availableCellMapLayers(records) {
    const layers = Array.from(new Set(records.map((record) => record.layerKind).filter(Boolean)));
    return layers.sort((a, b) => {
      const orderA = CELL_MAP_LAYER_ORDER.indexOf(a);
      const orderB = CELL_MAP_LAYER_ORDER.indexOf(b);
      const rankA = orderA === -1 ? CELL_MAP_LAYER_ORDER.length : orderA;
      const rankB = orderB === -1 ? CELL_MAP_LAYER_ORDER.length : orderB;
      return rankA - rankB || a.localeCompare(b);
    });
  }

  function selectedCellMapLayer(records) {
    const layers = availableCellMapLayers(records);
    if (!layers.length) return '';
    if (state.selectedLayerKind !== 'all' && layers.includes(state.selectedLayerKind)) {
      return state.selectedLayerKind;
    }
    return layers[0];
  }

  function cellStatusCode(status) {
    const clean = normalizedText(status);
    if (clean === 'solved' || clean === 'ok' || clean === 'active' || clean === 'valid') return 3;
    if (clean === 'low_fold') return 2;
    if (clean === 'inactive' || clean === 'no_observations' || clean === 'empty') return 1;
    return 0;
  }

  function cellStatusColor(status) {
    return CELL_STATUS_COLORS[normalizedText(status)] || CELL_STATUS_COLORS.unknown;
  }

  function cellMapQuantityValue(point, quantity) {
    if (quantity.status) return cellStatusCode(point.status);
    const raw = finiteMetric(
      point.raw,
      quantity.columns,
      quantity.secondColumns,
      quantity.scale || 1.0,
    );
    return Number.isFinite(raw) ? raw : NaN;
  }

  function cellMapHoverText(point, quantity, value) {
    const quantityValue = quantity.status
      ? point.status
      : `${formatNumber(value, quantity.unit === 'count' ? 0 : 3)}${quantity.unit ? ` ${quantity.unit}` : ''}`;
    return [
      `Cell: ix ${point.cellIx}, iy ${point.cellIy}`,
      `Layer: ${layerLabel(point.layerKind)}`,
      `Center X: ${formatNumber(point.centerX, 2)} m`,
      `Center Y: ${formatNumber(point.centerY, 2)} m`,
      `${quantity.label}: ${quantityValue}`,
      `Velocity: ${formatNumber(point.velocity, 2)} m/s`,
      `Initial velocity: ${formatNumber(point.initialVelocity, 2)} m/s`,
      `Velocity update: ${formatNumber(point.velocityUpdate, 2)} m/s`,
      `Fold: ${formatNumber(point.fold, 0)}`,
      `Used fold: ${formatNumber(point.usedFold, 0)}`,
      `Rejected fold: ${formatNumber(point.rejectedFold, 0)}`,
      `Residual RMS: ${formatNumber(point.residualRmsMs, 3)} ms`,
      `Residual MAD: ${formatNumber(point.residualMadMs, 3)} ms`,
      `Status: ${point.status}`,
      `Status reason: ${point.statusReason}`,
    ].join('<br>');
  }

  function buildCellMapMatrix(records, quantity) {
    const xKeys = Array.from(new Set(records.map((point) => point.cellIx))).sort((a, b) => a - b);
    const yKeys = Array.from(new Set(records.map((point) => point.cellIy))).sort((a, b) => a - b);
    const byCell = new Map(records.map((point) => [`${point.cellIy}|${point.cellIx}`, point]));
    const x = xKeys.map((cellIx) => {
      const point = records.find((candidate) => candidate.cellIx === cellIx);
      return point ? point.centerX : cellIx;
    });
    const y = yKeys.map((cellIy) => {
      const point = records.find((candidate) => candidate.cellIy === cellIy);
      return point ? point.centerY : cellIy;
    });
    const z = [];
    const text = [];
    const customdata = [];
    for (const cellIy of yKeys) {
      const zRow = [];
      const textRow = [];
      const customRow = [];
      for (const cellIx of xKeys) {
        const point = byCell.get(`${cellIy}|${cellIx}`);
        if (!point) {
          zRow.push(null);
          textRow.push(`Cell: ix ${cellIx}, iy ${cellIy}<br>Status: missing_from_qc_rows`);
          customRow.push(null);
          continue;
        }
        const value = cellMapQuantityValue(point, quantity);
        zRow.push(Number.isFinite(value) ? value : null);
        textRow.push(cellMapHoverText(point, quantity, value));
        customRow.push({
          cell_ix: point.cellIx,
          cell_iy: point.cellIy,
          layer_kind: point.layerKind,
        });
      }
      z.push(zRow);
      text.push(textRow);
      customdata.push(customRow);
    }
    return { x, y, z, text, customdata };
  }

  function selectedCellMapPoint(records, layerKind) {
    if (!state.selectedCell || state.selectedCell.cell_ix === undefined) return null;
    const selectedLayer = state.selectedCell.layer_kind || layerKind;
    if (selectedLayer !== layerKind) return null;
    return records.find((point) => (
      point.cellIx === state.selectedCell.cell_ix
      && point.cellIy === state.selectedCell.cell_iy
    )) || null;
  }

  function profileTraceName(series, endpointKind, group) {
    const unit = profileDisplayUnit(series, group);
    const suffix = unit ? ` (${unit})` : '';
    return `${series.label}${suffix} ${endpointKind}`;
  }

  function profileHoverText(point, series, value, group) {
    const unit = profileDisplayUnit(series, group);
    const formatted = formatNumber(value, unit === 'count' ? 0 : 3);
    return [
      `Endpoint: ${point.endpointKind} ${point.endpointKey}`,
      `Node: ${point.nodeId}`,
      `Inline: ${formatNumber(point.inline, 2)} m`,
      `${series.label}: ${formatted}${unit ? ` ${unit}` : ''}`,
      `Static status: ${point.staticStatus}`,
      `Solution status: ${point.solutionStatus}`,
    ].join('<br>');
  }

  function clearNode(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
  }

  function appendText(parent, text) {
    parent.appendChild(document.createTextNode(text));
  }

  function createKv(items) {
    const list = document.createElement('dl');
    list.className = 'refraction-qc-kv';
    for (const [label, value] of items) {
      const dt = document.createElement('dt');
      dt.textContent = label;
      const dd = document.createElement('dd');
      dd.textContent = textOrDash(value);
      list.append(dt, dd);
    }
    return list;
  }

  function createTable(view) {
    const wrap = document.createElement('div');
    wrap.className = 'refraction-qc-table-wrap';
    const table = document.createElement('table');
    table.className = 'refraction-qc-table';

    const columns = Array.isArray(view.columns) ? view.columns.slice(0, 5) : [];
    const head = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const column of columns) {
      const th = document.createElement('th');
      th.textContent = column;
      headRow.appendChild(th);
    }
    head.appendChild(headRow);
    table.appendChild(head);

    const body = document.createElement('tbody');
    const records = Array.isArray(view.records) ? view.records.slice(0, 5) : [];
    for (const record of records) {
      const row = document.createElement('tr');
      for (const column of columns) {
        const td = document.createElement('td');
        td.textContent = textOrDash(record ? record[column] : '');
        row.appendChild(td);
      }
      body.appendChild(row);
    }
    table.appendChild(body);
    wrap.appendChild(table);
    return wrap;
  }

  function findViewData(bundle, viewDef) {
    const views = bundle && typeof bundle.views === 'object' && bundle.views ? bundle.views : {};
    for (const key of viewDef.viewKeys) {
      if (views[key]) return { key, view: views[key] };
    }
    return null;
  }

  function isUnavailable(bundle, viewDef) {
    const unavailable = Array.isArray(bundle?.unavailable_views) ? bundle.unavailable_views : [];
    return viewDef.unavailableKeys.some((key) => unavailable.includes(key));
  }

  function unavailableReason(bundle, viewDef) {
    const reasons = bundle && typeof bundle.unavailable_view_reasons === 'object' && bundle.unavailable_view_reasons
      ? bundle.unavailable_view_reasons
      : {};
    for (const key of viewDef.unavailableKeys) {
      const reason = reasons[key];
      if (typeof reason === 'string' && reason.trim()) return reason.trim();
    }
    return '';
  }

  function findDownsampling(bundle, key, view) {
    const downsampling = bundle && typeof bundle.downsampling === 'object' && bundle.downsampling
      ? bundle.downsampling
      : {};
    const entry = downsampling[key];
    if (entry && typeof entry === 'object') return entry;
    if (view && typeof view === 'object') {
      return {
        total_points: view.total_points,
        returned_points: view.returned_points,
        downsampled: view.downsampled,
        method: view.downsampling_method,
      };
    }
    return null;
  }

  function renderSummary(content, bundle) {
    const summary = bundle && typeof bundle.summary === 'object' && bundle.summary ? bundle.summary : {};
    content.appendChild(createKv([
      ['Job ID', bundle.job_id],
      ['State', summary.job_state || summary.status],
      ['Workflow', summary.workflow],
      ['Method', summary.method],
      ['Conversion', summary.conversion_mode],
      ['Layer count', summary.layer_count],
      ['Coordinate mode', bundle.coordinate_mode],
      ['Available views', Array.isArray(bundle.available_views) ? bundle.available_views.join(', ') : ''],
      ['Unavailable views', Array.isArray(bundle.unavailable_views) ? bundle.unavailable_views.join(', ') : ''],
    ]));
  }

  function renderTabular(content, bundle, viewDef) {
    const found = findViewData(bundle, viewDef);
    if (!found) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      if (isUnavailable(bundle, viewDef)) {
        const reason = unavailableReason(bundle, viewDef);
        missing.textContent = reason
          ? `This view is unavailable from the loaded QC bundle artifacts: ${reason}.`
          : 'This view is unavailable from the loaded QC bundle artifacts.';
      } else {
        missing.textContent = 'No sampled records are present for this view.';
      }
      content.appendChild(missing);
      return;
    }

    const { key, view } = found;
    content.appendChild(createKv([
      ['Bundle view', key],
      ['Artifact', view.artifact],
      ['Rows', `${view.returned_points || 0} of ${view.total_points || 0}`],
      ['Downsampled', view.downsampled ? 'yes' : 'no'],
      ['Columns', Array.isArray(view.columns) ? view.columns.join(', ') : ''],
    ]));
    if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
      content.appendChild(createTable(view));
    }
  }

  function createFirstBreakPlot(testId) {
    const plot = document.createElement('div');
    plot.className = 'refraction-qc-plot';
    plot.dataset.testid = testId;
    return plot;
  }

  function plotHoverText(point) {
    return [
      `Layer: ${layerLabel(point.layerKind)}`,
      `Status: ${point.status}`,
      `Trace: ${point.traceIndex}`,
      `Source: ${point.source}`,
      `Receiver: ${point.receiver}`,
      `Observed: ${formatNumber(point.observedMs, 2)} ms`,
      `Modeled: ${formatNumber(point.modeledMs, 2)} ms`,
      `Residual: ${formatNumber(point.residualMs, 2)} ms`,
    ].join('<br>');
  }

  function firstBreakPickCustomData(point) {
    return {
      x: point.x,
      observed_ms: point.observedMs,
      modeled_ms: point.modeledMs,
      residual_ms: point.residualMs,
      source_endpoint_key: point.sourceEndpointKey,
      receiver_endpoint_key: point.receiverEndpointKey,
      trace_index: point.traceIndex,
      layer_kind: point.layerKind,
      status: point.status,
      source: point.source,
      receiver: point.receiver,
    };
  }

  function firstBreakPickFromCustomData(customdata) {
    if (!customdata || typeof customdata !== 'object') return null;
    const residualMs = toFiniteNumber(customdata.residual_ms);
    return {
      x: toFiniteNumber(customdata.x),
      observedMs: toFiniteNumber(customdata.observed_ms),
      modeledMs: toFiniteNumber(customdata.modeled_ms),
      residualMs,
      sourceEndpointKey: String(customdata.source_endpoint_key || '').trim(),
      receiverEndpointKey: String(customdata.receiver_endpoint_key || '').trim(),
      traceIndex: textOrDash(customdata.trace_index),
      layerKind: String(customdata.layer_kind || '').trim() || 'unknown',
      status: String(customdata.status || '').trim() || 'used',
      source: textOrDash(customdata.source || customdata.source_endpoint_key),
      receiver: textOrDash(customdata.receiver || customdata.receiver_endpoint_key),
    };
  }

  function firstBreakPickKey(point) {
    if (!point) return '';
    return [
      point.traceIndex,
      point.layerKind,
      point.sourceEndpointKey,
      point.receiverEndpointKey,
      Number.isFinite(point.x) ? formatNumber(point.x, 6) : '',
      Number.isFinite(point.residualMs) ? formatNumber(point.residualMs, 6) : '',
    ].join('|');
  }

  function selectedFirstBreakPick(points) {
    const selected = state.selectedFirstBreakPick;
    if (!selected) return null;
    const selectedKey = firstBreakPickKey(selected);
    return points.find((point) => firstBreakPickKey(point) === selectedKey) || null;
  }

  function attachFirstBreakPickClickActions(plot) {
    if (!plot || typeof plot.on !== 'function' || plot.dataset.firstBreakPickClickAttached === 'true') return;
    plot.dataset.firstBreakPickClickAttached = 'true';
    plot.on('plotly_click', (event) => {
      const point = event?.points?.[0];
      const pick = firstBreakPickFromCustomData(point?.customdata);
      if (!pick) return;
      state.selectedFirstBreakPick = pick;
      state.firstBreakDrilldown = null;
      state.firstBreakDrilldownError = null;
      render();
    });
  }

  function gatherPreviewInputsReady() {
    const { errors } = buildGatherPreviewRequest();
    return !errors.length;
  }

  function previewGatherForEndpoint(endpointKind, endpointKey) {
    const cleanKey = String(endpointKey || '').trim();
    if (!cleanKey) return;
    state.gatherAxis = endpointKind === 'receiver' ? 'receiver' : 'source';
    state.gatherEndpointKey = cleanKey;
    state.gatherEndpointSearch = '';
    state.gatherPreview = null;
    state.gatherError = null;
    setSelectedView('gather_preview');
    if (gatherPreviewInputsReady()) loadGatherPreview();
  }

  async function openEndpointDrilldownForPick(pick) {
    const sourceKey = String(pick?.sourceEndpointKey || '').trim();
    const receiverKey = String(pick?.receiverEndpointKey || '').trim();
    const endpointKind = sourceKey ? 'source' : (receiverKey ? 'receiver' : '');
    const endpointKey = endpointKind === 'source' ? sourceKey : receiverKey;
    if (!endpointKind || !endpointKey) return;

    state.selectedEndpointKind = endpointKind;
    state.selectedEndpoint = endpointKey;
    state.firstBreakDrilldown = null;
    state.firstBreakDrilldownError = null;
    state.firstBreakDrilldownLoading = true;
    render();
    try {
      const response = await fetch('/statics/refraction/qc/drilldown', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: String(state.selectedJobId || dom?.jobId?.value || '').trim(),
          target: { kind: 'endpoint', endpoint_kind: endpointKind, endpoint_key: endpointKey },
        }),
      });
      if (!response.ok) throw new Error(await readError(response, 'Endpoint drilldown request'));
      state.firstBreakDrilldown = await response.json();
    } catch (error) {
      state.firstBreakDrilldownError = error instanceof Error ? error.message : String(error);
    } finally {
      state.firstBreakDrilldownLoading = false;
      render();
    }
  }

  function createFirstBreakPickActionButton(label, disabledReason, onClick) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    if (disabledReason) {
      button.disabled = true;
      button.title = disabledReason;
    } else {
      button.addEventListener('click', onClick);
    }
    return button;
  }

  function createFirstBreakPickActions(pick) {
    const panel = document.createElement('section');
    panel.className = 'refraction-qc-pick-actions';
    panel.dataset.testid = 'refraction-qc-first-break-pick-actions';

    const title = document.createElement('h3');
    title.textContent = 'Selected pick';
    panel.appendChild(title);
    panel.appendChild(createKv([
      ['Trace', pick.traceIndex],
      ['Layer', layerLabel(pick.layerKind)],
      ['Residual', `${formatNumber(pick.residualMs, 1)} ms`],
      ['Source', pick.source],
      ['Receiver', pick.receiver],
    ]));

    const actions = document.createElement('div');
    actions.className = 'refraction-qc-actions';
    actions.append(
      createFirstBreakPickActionButton(
        'Preview source gather',
        pick.sourceEndpointKey ? '' : 'Source endpoint key is missing for this pick.',
        () => previewGatherForEndpoint('source', pick.sourceEndpointKey),
      ),
      createFirstBreakPickActionButton(
        'Preview receiver gather',
        pick.receiverEndpointKey ? '' : 'Receiver endpoint key is missing for this pick.',
        () => previewGatherForEndpoint('receiver', pick.receiverEndpointKey),
      ),
      createFirstBreakPickActionButton(
        state.firstBreakDrilldownLoading ? 'Opening drilldown...' : 'Open endpoint drilldown',
        pick.sourceEndpointKey || pick.receiverEndpointKey ? '' : 'Endpoint key is missing for this pick.',
        () => openEndpointDrilldownForPick(pick),
      ),
    );
    panel.appendChild(actions);

    const missingReasons = [];
    if (!pick.sourceEndpointKey) missingReasons.push('Source gather preview is disabled because source_endpoint_key is missing.');
    if (!pick.receiverEndpointKey) missingReasons.push('Receiver gather preview is disabled because receiver_endpoint_key is missing.');
    if (missingReasons.length) {
      const reason = document.createElement('p');
      reason.className = 'refraction-qc-note';
      reason.dataset.testid = 'refraction-qc-first-break-pick-action-reason';
      reason.textContent = missingReasons.join(' ');
      panel.appendChild(reason);
    }

    if (state.firstBreakDrilldownError) {
      const error = document.createElement('p');
      error.className = 'refraction-qc-error';
      error.dataset.testid = 'refraction-qc-first-break-drilldown-error';
      error.textContent = state.firstBreakDrilldownError;
      panel.appendChild(error);
    } else if (state.firstBreakDrilldown) {
      const endpoint = state.firstBreakDrilldown.endpoint || {};
      const details = createKv([
        ['Drilldown endpoint', textOrDash(endpoint.endpoint_key)],
        ['Observations', textOrDash(state.firstBreakDrilldown.observations?.total_count)],
      ]);
      details.dataset.testid = 'refraction-qc-first-break-drilldown-summary';
      panel.appendChild(details);
    }

    return panel;
  }

  function reducedTimeHoverText(point) {
    return [
      `Layer gate: ${layerLabel(point.layerKind)}`,
      `Status: ${point.status}`,
      `Trace: ${point.traceIndex}`,
      `Source: ${point.source}`,
      `Receiver: ${point.receiver}`,
      `Observed: ${formatNumber(point.observedMs, 2)} ms`,
      `Reduction velocity: ${formatNumber(point.reductionVelocity, 2)} m/s`,
      `Reduced time: ${formatNumber(point.reducedMs, 2)} ms`,
    ].join('<br>');
  }

  function reducedTimeVelocitySummary(points) {
    const velocities = points
      .map((point) => point.reductionVelocity)
      .filter((value) => Number.isFinite(value));
    if (!velocities.length) return 'unavailable';
    velocities.sort((a, b) => a - b);
    const min = velocities[0];
    const max = velocities[velocities.length - 1];
    const median = velocities[Math.floor((velocities.length - 1) / 2)];
    if (min === max) return `${formatNumber(min, 2)} m/s`;
    return `${formatNumber(min, 2)}-${formatNumber(max, 2)} m/s; median ${formatNumber(median, 2)} m/s`;
  }

  function statusCounts(points, field) {
    const counts = new Map();
    for (const point of points) {
      const key = point[field] || 'unknown';
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    return Array.from(counts.entries())
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([key, count]) => `${key}: ${count}`)
      .join(', ');
  }

  function gateBound(value) {
    const parsed = toFiniteNumber(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function gateMinOffset(gate, kind) {
    if (kind === 'v1_direct_arrival') {
      const directBound = gateBound(gate.min_direct_offset_m);
      if (directBound !== null) return directBound;
    }
    return gateBound(gate.min_offset_m);
  }

  function gateMaxOffset(gate, kind) {
    if (kind === 'v1_direct_arrival') {
      const directBound = gateBound(gate.max_direct_offset_m);
      if (directBound !== null) return directBound;
    }
    return gateBound(gate.max_offset_m);
  }

  function gateHasBounds(gate, kind) {
    if (kind === 'v1_direct_arrival') {
      return (
        gate.min_direct_offset_m !== undefined
        || gate.max_direct_offset_m !== undefined
        || gate.min_offset_m !== undefined
        || gate.max_offset_m !== undefined
      );
    }
    return gate.min_offset_m !== undefined || gate.max_offset_m !== undefined;
  }

  function gateEnabled(gate, kind) {
    if (!gate || typeof gate !== 'object') return false;
    const enabled = gate.enabled;
    if (enabled === false) return false;
    const enabledText = normalizedText(enabled);
    if (enabledText === 'false' || enabledText === '0' || enabledText === 'no') return false;
    return gateHasBounds(gate, kind);
  }

  function reducedTimeGateSourceFromLayers(layers) {
    if (!Array.isArray(layers)) return layers;
    const byKind = {};
    for (const layer of layers) {
      if (!layer || typeof layer !== 'object') continue;
      const kind = normalizedText(layer.kind || layer.layer_kind);
      if (!kind || byKind[kind]) continue;
      byKind[kind] = layer;
    }
    return byKind;
  }

  function reducedTimeGateSourceFromObservationGates(observationGates) {
    if (!Array.isArray(observationGates)) return observationGates;
    const byKind = {};
    for (const gate of observationGates) {
      if (!gate || typeof gate !== 'object') continue;
      const kind = normalizedText(gate.layer_kind);
      if (!kind || byKind[kind]) continue;
      byKind[kind] = gate;
    }
    return byKind;
  }

  function collectReducedTimeGates(bundle) {
    const summary = bundle && typeof bundle.summary === 'object' && bundle.summary ? bundle.summary : {};
    const sources = [];
    if (summary.observation_gates && typeof summary.observation_gates === 'object') {
      sources.push(reducedTimeGateSourceFromObservationGates(summary.observation_gates));
    }
    const layerSource = reducedTimeGateSourceFromLayers(summary.layers);
    if (layerSource && typeof layerSource === 'object') {
      sources.push(layerSource);
    }
    const gates = [];
    const seen = new Set();
    for (const source of sources) {
      for (const kind of REDUCED_TIME_GATE_KINDS) {
        if (seen.has(kind)) continue;
        const gate = source[kind];
        if (!gateEnabled(gate, kind)) continue;
        gates.push({
          kind,
          minOffset: gateMinOffset(gate, kind),
          maxOffset: gateMaxOffset(gate, kind),
        });
        seen.add(kind);
      }
    }
    return gates;
  }

  function gateRangeLabel(gate) {
    if (gate.minOffset !== null && gate.maxOffset !== null) {
      return `${formatNumber(gate.minOffset, 1)}-${formatNumber(gate.maxOffset, 1)} m`;
    }
    if (gate.minOffset !== null) return `>= ${formatNumber(gate.minOffset, 1)} m`;
    if (gate.maxOffset !== null) return `<= ${formatNumber(gate.maxOffset, 1)} m`;
    return 'all offsets';
  }

  function reducedTimeGateOverlays(bundle) {
    if (state.firstBreakXAxis !== 'offset') return { shapes: [], label: 'Offset gate overlays are shown when X axis is Offset.' };
    const shapes = [];
    const labels = [];
    for (const gate of collectReducedTimeGates(bundle)) {
      const color = LAYER_COLORS[gate.kind] || LAYER_COLORS.unknown;
      if (gate.minOffset !== null && gate.maxOffset !== null) {
        shapes.push({
          type: 'rect',
          xref: 'x',
          yref: 'paper',
          x0: gate.minOffset,
          x1: gate.maxOffset,
          y0: 0,
          y1: 1,
          fillcolor: color,
          opacity: 0.08,
          line: { color, width: 1, dash: 'dot' },
          layer: 'below',
        });
      } else {
        const x = gate.minOffset !== null ? gate.minOffset : gate.maxOffset;
        if (x !== null) {
          shapes.push({
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: x,
            x1: x,
            y0: 0,
            y1: 1,
            line: { color, width: 1, dash: 'dot' },
            layer: 'below',
          });
        }
      }
      labels.push(`${layerLabel(gate.kind)} ${gateRangeLabel(gate)}`);
    }
    return {
      shapes,
      label: labels.length ? labels.join('; ') : 'No offset gate metadata is available for this QC bundle.',
    };
  }

  function renderFirstBreakPlots(content, bundle, viewDef) {
    const found = findViewData(bundle, viewDef);
    if (!found) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = isUnavailable(bundle, viewDef)
        ? 'This view is unavailable from the loaded QC bundle artifacts.'
        : 'No sampled first-break fit records are present for this view.';
      content.appendChild(missing);
      return;
    }

    const { key, view } = found;
    const points = filteredFirstBreakPoints(view);
    const downsampling = findDownsampling(bundle, key, view);
    const downsamplingText = downsampling
      ? `${downsampling.returned_points || 0} of ${downsampling.total_points || 0}; ${downsampling.downsampled ? 'downsampled' : 'not downsampled'}${downsampling.method ? ` (${downsampling.method})` : ''}`
      : 'not reported';

    content.appendChild(createKv([
      ['Bundle view', key],
      ['Artifact', view.artifact],
      ['Rows', `${view.returned_points || 0} of ${view.total_points || 0}`],
      ['Plotted points', `${points.length}`],
      ['Layer filter', state.selectedLayerKind === 'all' ? 'all' : layerLabel(state.selectedLayerKind)],
      ['Rejected picks', state.showRejectedFirstBreaks ? 'shown' : 'hidden'],
    ]));

    const residualNote = document.createElement('p');
    residualNote.className = 'refraction-qc-note';
    residualNote.textContent = 'Residual = observed - modeled, shown in ms.';
    residualNote.dataset.testid = 'refraction-qc-first-break-residual-note';
    content.appendChild(residualNote);

    const downsamplingNote = document.createElement('p');
    downsamplingNote.className = 'refraction-qc-note';
    downsamplingNote.textContent = `Downsampling: ${downsamplingText}`;
    downsamplingNote.dataset.testid = 'refraction-qc-first-break-downsampling';
    content.appendChild(downsamplingNote);

    if (!points.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = 'No plottable observed/modeled first-break records match the current filters.';
      content.appendChild(missing);
      if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
        content.appendChild(createTable(view));
      }
      return;
    }

    const selectedPick = selectedFirstBreakPick(points);
    if (selectedPick) content.appendChild(createFirstBreakPickActions(selectedPick));

    const plotGrid = document.createElement('div');
    plotGrid.className = 'refraction-qc-plot-grid';
    const timePlot = createFirstBreakPlot('refraction-qc-first-break-time-plot');
    const residualPlot = createFirstBreakPlot('refraction-qc-first-break-residual-plot');
    timePlot.dataset.pointCount = String(points.length);
    residualPlot.dataset.pointCount = String(points.length);
    plotGrid.append(timePlot, residualPlot);
    content.appendChild(plotGrid);

    if (window.Plotly) {
      const xAxis = firstBreakXAxisDefinition();
      const hoverText = points.map(plotHoverText);
      const customdata = points.map(firstBreakPickCustomData);
      const commonLayout = {
        height: plotHeight(260, 440),
        margin: { l: 58, r: 14, t: 34, b: 50 },
        font: { size: 10, color: '#334155' },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        xaxis: {
          title: { text: xAxis.label },
          zeroline: false,
          gridcolor: '#e5e7eb',
        },
        legend: {
          orientation: 'h',
          x: 0,
          y: 1.16,
          xanchor: 'left',
          yanchor: 'top',
          font: { size: 10 },
        },
      };
      const config = { displayModeBar: false, responsive: true };

      Promise.resolve(window.Plotly.newPlot(timePlot, [
        {
          name: 'Observed',
          type: 'scatter',
          mode: 'markers',
          x: points.map((point) => point.x),
          y: points.map((point) => point.observedMs),
          text: hoverText,
          customdata,
          hovertemplate: '%{text}<extra>Observed</extra>',
          marker: {
            color: '#2563eb',
            size: 7,
            opacity: points.map((point) => point.opacity),
          },
        },
        {
          name: 'Modeled',
          type: 'scatter',
          mode: 'markers',
          x: points.map((point) => point.x),
          y: points.map((point) => point.modeledMs),
          text: hoverText,
          customdata,
          hovertemplate: '%{text}<extra>Modeled</extra>',
          marker: {
            color: '#f97316',
            symbol: 'x',
            size: 8,
            opacity: points.map((point) => point.opacity),
          },
        },
      ], {
        ...commonLayout,
        title: { text: 'Observed vs modeled first-break time', font: { size: 12 } },
        yaxis: {
          title: { text: 'First-break time (ms)' },
          zeroline: false,
          gridcolor: '#e5e7eb',
        },
      }, config)).then(() => attachFirstBreakPickClickActions(timePlot));

      const residualGroups = new Map();
      for (const point of points) {
        const groupKey = `${point.layerKind}|${point.status}`;
        if (!residualGroups.has(groupKey)) {
          residualGroups.set(groupKey, {
            layerKind: point.layerKind,
            status: point.status,
            x: [],
            y: [],
            text: [],
            customdata: [],
          });
        }
        const group = residualGroups.get(groupKey);
        group.x.push(point.x);
        group.y.push(point.residualMs);
        group.text.push(plotHoverText(point));
        group.customdata.push(firstBreakPickCustomData(point));
      }
      const residualTraces = Array.from(residualGroups.values()).map((group) => ({
        name: `${layerLabel(group.layerKind)} ${group.status}`,
        type: 'scatter',
        mode: 'markers',
        x: group.x,
        y: group.y,
        text: group.text,
        customdata: group.customdata,
        hovertemplate: '%{text}<extra></extra>',
        marker: {
          color: LAYER_COLORS[group.layerKind] || LAYER_COLORS.unknown,
          symbol: group.status === 'rejected' ? 'x' : 'circle',
          size: group.status === 'rejected' ? 8 : 7,
          opacity: group.status === 'rejected' ? 0.55 : 0.9,
        },
      }));
      Promise.resolve(window.Plotly.newPlot(residualPlot, residualTraces, {
        ...commonLayout,
        title: { text: 'First-break residuals', font: { size: 12 } },
        yaxis: {
          title: { text: 'Residual (ms)' },
          zeroline: true,
          zerolinecolor: '#94a3b8',
          gridcolor: '#e5e7eb',
        },
      }, config)).then(() => attachFirstBreakPickClickActions(residualPlot));
    } else {
      timePlot.textContent = 'Plot library is unavailable.';
      residualPlot.textContent = 'Plot library is unavailable.';
    }

    if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
      content.appendChild(createTable(view));
    }
  }

  function renderReducedTimePlot(content, bundle, viewDef) {
    const found = findViewData(bundle, viewDef);
    if (!found) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = isUnavailable(bundle, viewDef)
        ? 'This view is unavailable from the loaded QC bundle artifacts.'
        : 'No sampled reduced-time records are present for this view.';
      content.appendChild(missing);
      return;
    }

    const { key, view } = found;
    const points = filteredReducedTimePoints(view);
    const plottedPoints = points.filter((point) => Number.isFinite(point.reducedMs));
    const unavailablePoints = points.filter((point) => !Number.isFinite(point.reducedMs));
    const downsampling = findDownsampling(bundle, key, view);
    const downsamplingText = downsampling
      ? `${downsampling.returned_points || 0} of ${downsampling.total_points || 0}; ${downsampling.downsampled ? 'downsampled' : 'not downsampled'}${downsampling.method ? ` (${downsampling.method})` : ''}`
      : 'not reported';
    const gateOverlay = reducedTimeGateOverlays(bundle);

    content.appendChild(createKv([
      ['Bundle view', key],
      ['Artifact', view.artifact],
      ['Rows', `${view.returned_points || 0} of ${view.total_points || 0}`],
      ['Plotted points', `${plottedPoints.length}`],
      ['Layer filter', state.selectedLayerKind === 'all' ? 'all' : layerLabel(state.selectedLayerKind)],
      ['Unused picks', state.showRejectedFirstBreaks ? 'shown' : 'hidden'],
      ['Reduction velocity', reducedTimeVelocitySummary(points)],
      ['Unavailable rows', unavailablePoints.length ? `${unavailablePoints.length}; ${statusCounts(unavailablePoints, 'unavailableReason')}` : '0'],
    ]));

    const formulaNote = document.createElement('p');
    formulaNote.className = 'refraction-qc-note';
    formulaNote.textContent = 'Reduced time = observed first-break time - offset / reduction velocity, shown in ms.';
    formulaNote.dataset.testid = 'refraction-qc-reduced-time-formula-note';
    content.appendChild(formulaNote);

    const gateNote = document.createElement('p');
    gateNote.className = 'refraction-qc-note';
    gateNote.textContent = `Gate overlays: ${gateOverlay.label}`;
    gateNote.dataset.testid = 'refraction-qc-reduced-time-gates';
    content.appendChild(gateNote);

    const downsamplingNote = document.createElement('p');
    downsamplingNote.className = 'refraction-qc-note';
    downsamplingNote.textContent = `Downsampling: ${downsamplingText}`;
    downsamplingNote.dataset.testid = 'refraction-qc-reduced-time-downsampling';
    content.appendChild(downsamplingNote);

    if (!plottedPoints.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = unavailablePoints.length
        ? 'Reduced-time rows matched the filters, but none have an available reduced-time value.'
        : 'No plottable reduced-time records match the current filters.';
      content.appendChild(missing);
      if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
        content.appendChild(createTable(view));
      }
      return;
    }

    const plot = createFirstBreakPlot('refraction-qc-reduced-time-plot');
    plot.dataset.pointCount = String(plottedPoints.length);
    content.appendChild(plot);

    if (window.Plotly) {
      const xAxis = firstBreakXAxisDefinition();
      const groups = new Map();
      for (const point of plottedPoints) {
        const groupKey = `${point.layerKind}|${point.used ? 'used' : 'unused'}`;
        if (!groups.has(groupKey)) {
          groups.set(groupKey, {
            layerKind: point.layerKind,
            used: point.used,
            x: [],
            y: [],
            text: [],
          });
        }
        const group = groups.get(groupKey);
        group.x.push(point.x);
        group.y.push(point.reducedMs);
        group.text.push(reducedTimeHoverText(point));
      }
      const traces = Array.from(groups.values()).map((group) => ({
        name: `${layerLabel(group.layerKind)} ${group.used ? 'used' : 'unused'}`,
        type: 'scatter',
        mode: 'markers',
        x: group.x,
        y: group.y,
        text: group.text,
        hovertemplate: '%{text}<extra></extra>',
        marker: {
          color: LAYER_COLORS[group.layerKind] || LAYER_COLORS.unknown,
          symbol: group.used ? 'circle' : 'x',
          size: group.used ? 7 : 8,
          opacity: group.used ? 0.9 : 0.55,
        },
      }));
      window.Plotly.newPlot(plot, traces, {
        height: plotHeight(300, 480),
        margin: { l: 62, r: 14, t: 34, b: 50 },
        font: { size: 10, color: '#334155' },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        title: { text: 'Reduced-time first-break QC', font: { size: 12 } },
        xaxis: {
          title: { text: xAxis.label },
          zeroline: false,
          gridcolor: '#e5e7eb',
        },
        yaxis: {
          title: { text: 'Reduced time (ms)' },
          zeroline: true,
          zerolinecolor: '#94a3b8',
          gridcolor: '#e5e7eb',
        },
        legend: {
          orientation: 'h',
          x: 0,
          y: 1.16,
          xanchor: 'left',
          yanchor: 'top',
          font: { size: 10 },
        },
        shapes: gateOverlay.shapes,
      }, { displayModeBar: false, responsive: true });
    } else {
      plot.textContent = 'Plot library is unavailable.';
    }

    if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
      content.appendChild(createTable(view));
    }
  }

  function renderProfilePlot(content, bundle, viewDef) {
    const found = findViewData(bundle, viewDef);
    if (!found) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      if (isUnavailable(bundle, viewDef)) {
        const reason = unavailableReason(bundle, viewDef);
        missing.textContent = reason
          ? `This view is unavailable from refraction_line_profile_qc_* artifacts: ${reason}.`
          : 'This view is unavailable from refraction_line_profile_qc_* artifacts.';
      } else {
        missing.textContent = 'No sampled line-profile records are present for this view.';
      }
      content.appendChild(missing);
      return;
    }

    const { key, view } = found;
    const records = filteredProfileRecords(view);
    const { group, available, unavailable } = selectedProfileSeries(records);
    const invalidCount = records.filter((record) => record.invalid).length;
    const downsampling = findDownsampling(bundle, key, view);
    const downsamplingText = downsampling
      ? `${downsampling.returned_points || 0} of ${downsampling.total_points || 0}; ${downsampling.downsampled ? 'downsampled' : 'not downsampled'}${downsampling.method ? ` (${downsampling.method})` : ''}`
      : 'not reported';
    const missingLabels = unavailable.map((series) => series.label).join(', ');

    content.appendChild(createKv([
      ['Bundle view', key],
      ['Artifact', view.artifact],
      ['Rows', `${view.returned_points || 0} of ${view.total_points || 0}`],
      ['Plotted endpoints', `${records.length}`],
      ['Profile group', group.label],
      ['Endpoint kind', state.selectedEndpointKind],
      ['Layer filter', state.selectedLayerKind === 'all' ? 'all' : layerLabel(state.selectedLayerKind)],
      ['Status filter', state.profileStatusFilter],
      ['Invalid endpoints', `${invalidCount}`],
      ['Y units', profileAxisTitle(group, available)],
      ['Unavailable fields', missingLabels || 'none'],
    ]));

    const signNote = document.createElement('p');
    signNote.className = 'refraction-qc-note';
    signNote.textContent = 'Static shifts follow corrected(t) = raw(t - shift_s); positive shift_s delays displayed events.';
    signNote.dataset.testid = 'refraction-qc-profile-sign-note';
    content.appendChild(signNote);

    const downsamplingNote = document.createElement('p');
    downsamplingNote.className = 'refraction-qc-note';
    downsamplingNote.textContent = `Downsampling: ${downsamplingText}`;
    downsamplingNote.dataset.testid = 'refraction-qc-profile-downsampling';
    content.appendChild(downsamplingNote);

    if (!records.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = 'No line-profile endpoints match the current endpoint and status filters.';
      content.appendChild(missing);
      if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
        content.appendChild(createTable(view));
      }
      return;
    }

    if (!available.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = 'No plottable profile fields match the current group and layer selection.';
      content.appendChild(missing);
      if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
        content.appendChild(createTable(view));
      }
      return;
    }

    const plot = createFirstBreakPlot('refraction-qc-profile-plot');
    plot.dataset.pointCount = String(records.length);
    content.appendChild(plot);

    if (window.Plotly) {
      const traces = [];
      for (const series of available) {
        const grouped = new Map();
        for (const point of records) {
          const rawValue = profileSeriesRawValue(point.raw, series);
          const y = profileDisplayValue(rawValue, series, group);
          if (!Number.isFinite(y)) continue;
          if (!grouped.has(point.endpointKind)) {
            grouped.set(point.endpointKind, {
              endpointKind: point.endpointKind,
              x: [],
              y: [],
              text: [],
              invalid: [],
            });
          }
          const entry = grouped.get(point.endpointKind);
          entry.x.push(point.inline);
          entry.y.push(y);
          entry.text.push(profileHoverText(point, series, y, group));
          entry.invalid.push(point.invalid);
        }
        for (const entry of grouped.values()) {
          traces.push({
            name: profileTraceName(series, entry.endpointKind, group),
            type: 'scatter',
            mode: 'lines+markers',
            x: entry.x,
            y: entry.y,
            text: entry.text,
            hovertemplate: '%{text}<extra></extra>',
            line: {
              color: PROFILE_SERIES_COLORS[series.key] || '#64748b',
              width: 2,
              dash: entry.endpointKind === 'receiver' ? 'dot' : 'solid',
            },
            marker: {
              color: PROFILE_SERIES_COLORS[series.key] || '#64748b',
              symbol: entry.invalid.map((invalid) => {
                if (invalid) return 'x';
                return entry.endpointKind === 'receiver' ? 'diamond' : 'circle';
              }),
              size: entry.invalid.map((invalid) => invalid ? 9 : 7),
              opacity: entry.invalid.map((invalid) => invalid ? 0.62 : 0.9),
            },
          });
        }
      }

      window.Plotly.newPlot(plot, traces, {
        height: plotHeight(320, 520),
        margin: { l: 62, r: 14, t: 34, b: 50 },
        font: { size: 10, color: '#334155' },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        title: { text: `${group.label} profile`, font: { size: 12 } },
        xaxis: {
          title: { text: 'Inline distance (m)' },
          zeroline: false,
          gridcolor: '#e5e7eb',
        },
        yaxis: {
          title: { text: profileAxisTitle(group, available) },
          zeroline: group.unit === 'ms' || group.unit === 'mixed',
          zerolinecolor: '#94a3b8',
          gridcolor: '#e5e7eb',
        },
        legend: {
          orientation: 'h',
          x: 0,
          y: 1.16,
          xanchor: 'left',
          yanchor: 'top',
          font: { size: 10 },
        },
      }, { displayModeBar: false, responsive: true });
    } else {
      plot.textContent = 'Plot library is unavailable.';
    }

    if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
      content.appendChild(createTable(view));
    }
  }

  function renderCellMapPlot(content, bundle, viewDef) {
    const found = findViewData(bundle, viewDef);
    if (!found) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = isUnavailable(bundle, viewDef)
        ? '3D cell maps are unavailable for global-velocity jobs or QC bundles without refraction_grid_map_qc.csv.'
        : 'No refraction_grid_map_qc rows are present for this QC bundle.';
      content.appendChild(missing);
      return;
    }

    const { key, view } = found;
    const records = filteredCellMapRecords(view);
    const layerKind = selectedCellMapLayer(records);
    const layerRecords = records.filter((record) => record.layerKind === layerKind);
    const quantity = cellMapQuantityDefinition();
    const downsampling = findDownsampling(bundle, key, view);
    const downsamplingText = downsampling
      ? `${downsampling.returned_points || 0} of ${downsampling.total_points || 0}; ${downsampling.downsampled ? 'downsampled' : 'not downsampled'}${downsampling.method ? ` (${downsampling.method})` : ''}`
      : 'not reported';
    const statusSummary = statusCounts(layerRecords, 'status') || 'none';
    const plottedValues = layerRecords
      .map((point) => cellMapQuantityValue(point, quantity))
      .filter((value) => Number.isFinite(value)).length;

    content.appendChild(createKv([
      ['Bundle view', key],
      ['Artifact', view.artifact],
      ['Rows', `${view.returned_points || 0} of ${view.total_points || 0}`],
      ['Coordinate mode', bundle.coordinate_mode],
      ['Layer filter', state.selectedLayerKind === 'all' ? 'all' : layerLabel(state.selectedLayerKind)],
      ['Plotted layer', layerKind ? layerLabel(layerKind) : '-'],
      ['Quantity', quantity.label],
      ['Cells', `${layerRecords.length}`],
      ['Plotted values', `${plottedValues}`],
      ['Status counts', statusSummary],
      ['Selected cell', selectedCellLabel(state.selectedCell)],
    ]));

    const downsamplingNote = document.createElement('p');
    downsamplingNote.className = 'refraction-qc-note';
    downsamplingNote.textContent = `Downsampling: ${downsamplingText}`;
    downsamplingNote.dataset.testid = 'refraction-qc-cell-map-downsampling';
    content.appendChild(downsamplingNote);

    if (!records.length || !layerRecords.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = 'No grid-map cells match the current layer selection.';
      content.appendChild(missing);
      if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
        content.appendChild(createTable(view));
      }
      return;
    }

    const plot = createFirstBreakPlot('refraction-qc-cell-map-plot');
    plot.classList.add('refraction-qc-cell-map-plot');
    plot.dataset.pointCount = String(layerRecords.length);
    content.appendChild(plot);

    if (window.Plotly) {
      const matrix = buildCellMapMatrix(layerRecords, quantity);
      const traces = [
        {
          name: quantity.label,
          type: 'heatmap',
          x: matrix.x,
          y: matrix.y,
          z: matrix.z,
          text: matrix.text,
          customdata: matrix.customdata,
          hovertemplate: '%{text}<extra></extra>',
          colorscale: quantity.status
            ? [
              [0, '#dc2626'],
              [0.249, '#dc2626'],
              [0.25, '#cbd5e1'],
              [0.499, '#cbd5e1'],
              [0.5, '#f59e0b'],
              [0.749, '#f59e0b'],
              [0.75, '#2563eb'],
              [1, '#2563eb'],
            ]
            : quantity.colorscale,
          reversescale: Boolean(quantity.reversescale),
          zmin: quantity.status ? 0 : undefined,
          zmax: quantity.status ? 3 : undefined,
          colorbar: quantity.status
            ? {
              title: { text: 'Status' },
              tickmode: 'array',
              tickvals: [0, 1, 2, 3],
              ticktext: ['other', 'inactive', 'low_fold', 'solved'],
            }
            : { title: { text: quantity.axisLabel } },
        },
      ];

      const flagged = layerRecords.filter((point) => cellStatusCode(point.status) < 3);
      if (flagged.length) {
        traces.push({
          name: 'Flagged cells',
          type: 'scatter',
          mode: 'markers',
          x: flagged.map((point) => point.centerX),
          y: flagged.map((point) => point.centerY),
          text: flagged.map((point) => cellMapHoverText(
            point,
            quantity,
            cellMapQuantityValue(point, quantity),
          )),
          customdata: flagged.map((point) => ({
            cell_ix: point.cellIx,
            cell_iy: point.cellIy,
            layer_kind: point.layerKind,
          })),
          hovertemplate: '%{text}<extra>Flagged</extra>',
          marker: {
            color: flagged.map((point) => cellStatusColor(point.status)),
            symbol: flagged.map((point) => point.status === 'low_fold' ? 'diamond-open' : 'x'),
            size: flagged.map((point) => point.status === 'low_fold' ? 12 : 10),
            line: { color: '#0f172a', width: 1 },
          },
        });
      }

      const selected = selectedCellMapPoint(layerRecords, layerKind);
      if (selected) {
        traces.push({
          name: 'Selected cell',
          type: 'scatter',
          mode: 'markers',
          x: [selected.centerX],
          y: [selected.centerY],
          text: [cellMapHoverText(selected, quantity, cellMapQuantityValue(selected, quantity))],
          customdata: [{
            cell_ix: selected.cellIx,
            cell_iy: selected.cellIy,
            layer_kind: selected.layerKind,
          }],
          hovertemplate: '%{text}<extra>Selected</extra>',
          marker: {
            color: '#0f172a',
            symbol: 'square-open',
            size: 16,
            line: { color: '#0f172a', width: 2 },
          },
        });
      }

      window.Plotly.newPlot(plot, traces, {
        height: plotHeight(340, 560),
        margin: { l: 62, r: 14, t: 42, b: 56 },
        font: { size: 10, color: '#334155' },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        title: { text: `${layerLabel(layerKind)} ${quantity.label} map`, font: { size: 12 } },
        xaxis: {
          title: { text: 'Cell center X (m)' },
          zeroline: false,
          gridcolor: '#e5e7eb',
        },
        yaxis: {
          title: { text: 'Cell center Y (m)' },
          zeroline: false,
          gridcolor: '#e5e7eb',
          scaleanchor: matrix.x.length > 1 && matrix.y.length > 1 ? 'x' : undefined,
        },
        legend: {
          orientation: 'h',
          x: 0,
          y: 1.14,
          xanchor: 'left',
          yanchor: 'top',
          font: { size: 10 },
        },
      }, { displayModeBar: false, responsive: true }).then(() => {
        plot.on('plotly_click', (event) => {
          const point = event?.points?.[0];
          const cell = point?.customdata;
          if (!cell || cell.cell_ix === undefined || cell.cell_iy === undefined) return;
          state.selectedCell = {
            cell_ix: Number(cell.cell_ix),
            cell_iy: Number(cell.cell_iy),
            layer_kind: String(cell.layer_kind || layerKind),
          };
          if (dom?.cell) dom.cell.value = `${state.selectedCell.cell_ix},${state.selectedCell.cell_iy}`;
          render();
        });
      });
    } else {
      plot.textContent = 'Plot library is unavailable.';
    }

    if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
      content.appendChild(createTable(view));
    }
  }

  function renderStaticComponents(content, bundle, viewDef) {
    const endpointView = viewByKey(bundle, STATIC_ENDPOINT_VIEW_KEY);
    const traceView = viewByKey(bundle, STATIC_TRACE_VIEW_KEY);
    if (!endpointView || !traceView) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = isUnavailable(bundle, viewDef)
        ? 'Static component QC is unavailable from the loaded refraction_static_component_qc_* artifacts.'
        : 'Static component QC endpoint and trace records are required for this view.';
      content.appendChild(missing);
      return;
    }

    const endpointRecords = staticEndpointRecords(endpointView);
    const traceRecords = staticTraceRecords(traceView);
    const endpointRecord = selectedStaticEndpointRecord(endpointRecords);
    const endpointNoMatch = Boolean(normalizedText(state.selectedEndpoint) && !endpointRecord);
    const traceRecord = endpointNoMatch ? null : selectedStaticTraceRecord(traceRecords, endpointRecord);
    const legacyRecord = matchingLegacyStaticRecord(bundle, endpointRecord);
    const endpointRows = buildStaticComponentRows(
      endpointRecord,
      legacyRecord,
      STATIC_COMPONENT_DEFS,
      ['static_status'],
    );
    const traceRows = buildStaticComponentRows(
      traceRecord,
      null,
      STATIC_TRACE_COMPONENT_DEFS,
      ['static_status'],
    );
    const signConvention = bundle.sign_convention || firstDefined(endpointRecord, ['sign_convention'])
      || firstDefined(traceRecord, ['sign_convention']);

    content.appendChild(createKv([
      ['Endpoint artifact', endpointView.artifact],
      ['Trace artifact', traceView.artifact],
      ['Endpoint rows', `${endpointView.returned_points || 0} of ${endpointView.total_points || 0}`],
      ['Trace rows', `${traceView.returned_points || 0} of ${traceView.total_points || 0}`],
      ['Selected endpoint kind', endpointRecord ? staticEndpointKind(endpointRecord) : state.selectedEndpointKind],
      ['Selected endpoint', endpointRecord ? staticEndpointKey(endpointRecord) : '-'],
      ['Selected trace', traceRecord ? staticTraceIndex(traceRecord) : '-'],
      ['Apply field shift', staticApplyToTraceShift(traceRecord || endpointRecord)],
      ['Endpoint status', textOrDash(firstDefined(endpointRecord, ['static_status']))],
      ['Trace status', textOrDash(firstDefined(traceRecord, ['static_status']))],
    ]));

    const signNote = document.createElement('p');
    signNote.className = 'refraction-qc-note';
    signNote.textContent = `${signConvention || 'corrected(t) = raw(t - shift_s)'}; positive shift_s delays displayed events and negative shift_s advances displayed events.`;
    signNote.dataset.testid = 'refraction-qc-static-sign-note';
    content.appendChild(signNote);

    if (!endpointRecord) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = 'No source/receiver endpoint component rows match the current endpoint selector.';
      content.appendChild(missing);
    } else {
      renderStaticComponentBars(
        content,
        endpointRows,
        'refraction-qc-static-waterfall',
        'Endpoint static components',
      );
      content.appendChild(createStaticComponentTable(
        endpointRows,
        'refraction-qc-static-component-list',
      ));
    }

    if (!traceRecord) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = 'No trace component row matches the current trace or endpoint selector.';
      content.appendChild(missing);
    } else {
      renderStaticComponentBars(
        content,
        traceRows,
        'refraction-qc-static-trace-waterfall',
        'Trace static components',
      );
      content.appendChild(createStaticComponentTable(
        traceRows,
        'refraction-qc-static-trace-component-list',
      ));
    }
  }

  function currentGatherContext() {
    const slider = document.getElementById('key1_slider');
    const sliderIndex = Number.parseInt(slider?.value || '0', 10) || 0;
    const key1FromViewer = Array.isArray(window.key1Values)
      ? window.key1Values[sliderIndex]
      : undefined;
    const windowData = window.latestWindowRender || {};
    const xRange = Array.isArray(window.savedXRange) && window.savedXRange.length === 2
      ? window.savedXRange
      : null;
    const x0FromRange = xRange
      ? Math.max(0, Math.floor(Math.min(Number(xRange[0]), Number(xRange[1]))))
      : NaN;
    const x1FromRange = xRange
      ? Math.max(0, Math.ceil(Math.max(Number(xRange[0]), Number(xRange[1]))))
      : NaN;
    const fileIdInput = document.getElementById('file_id');

    return {
      fileId: state.gatherFileId
        || String(fileIdInput?.value || '')
        || String(window.currentFileId || '')
        || searchOrStorageValue('file_id', 'file_id')
        || '',
      key1Byte: state.gatherKey1Byte
        || String(window.currentKey1Byte || '')
        || searchOrStorageValue('key1_byte', 'key1_byte', '189'),
      key2Byte: state.gatherKey2Byte
        || String(window.currentKey2Byte || '')
        || searchOrStorageValue('key2_byte', 'key2_byte', '193'),
      key1: state.gatherSectionKey1
        || searchParamValue('key1')
        || (key1FromViewer !== undefined && key1FromViewer !== null ? String(key1FromViewer) : ''),
      x0: state.gatherX0
        || searchParamValue('x0')
        || (Number.isFinite(x0FromRange) ? String(x0FromRange) : '')
        || (Number.isFinite(Number(windowData.x0)) ? String(windowData.x0) : '0'),
      x1: state.gatherX1
        || searchParamValue('x1')
        || (Number.isFinite(x1FromRange) ? String(x1FromRange) : '')
        || (Number.isFinite(Number(windowData.x1)) ? String(windowData.x1) : ''),
    };
  }

  function parseFiniteNumberField(value, label, errors) {
    const parsed = Number.parseFloat(String(value ?? '').trim());
    if (!Number.isFinite(parsed)) {
      errors.push(`${label} must be a finite number.`);
      return NaN;
    }
    return parsed;
  }

  function parsePositiveIntegerField(value, label, errors) {
    const text = String(value ?? '').trim();
    const parsed = Number(text);
    if (!text || !Number.isInteger(parsed) || parsed < 1) {
      errors.push(`${label} must be a positive integer.`);
      return NaN;
    }
    return parsed;
  }

  function parseIntegerField(value, label, errors) {
    const text = String(value ?? '').trim();
    const parsed = Number(text);
    if (!text || !Number.isInteger(parsed)) {
      errors.push(`${label} must be an integer.`);
      return NaN;
    }
    return parsed;
  }

  function buildGatherPreviewRequest() {
    const errors = [];
    const context = currentGatherContext();
    const jobId = String(state.selectedJobId || dom?.jobId?.value || '').trim();
    const fileId = String(context.fileId || '').trim();
    if (!jobId) errors.push('Job ID is required.');
    if (!fileId) errors.push('File ID is required.');

    const key1Byte = parsePositiveIntegerField(context.key1Byte, 'key1 byte', errors);
    const key2Byte = parsePositiveIntegerField(context.key2Byte, 'key2 byte', errors);
    const timeStart = parseFiniteNumberField(state.gatherTimeStartS, 'Time start', errors);
    const timeEnd = parseFiniteNumberField(state.gatherTimeEndS, 'Time end', errors);
    const maxTraces = parsePositiveIntegerField(state.gatherMaxTraces, 'Max traces', errors);
    if (Number.isFinite(timeStart) && timeStart < 0) {
      errors.push('Time start must be greater than or equal to 0.');
    }
    if (Number.isFinite(timeEnd) && Number.isFinite(timeStart) && timeEnd <= timeStart) {
      errors.push('Time end must be greater than time start.');
    }

    const payload = {
      job_id: jobId,
      file_id: fileId,
      key1_byte: key1Byte,
      key2_byte: key2Byte,
      gather_axis: state.gatherAxis,
      time_start_s: timeStart,
      time_end_s: timeEnd,
      max_traces: maxTraces,
      scaling: 'amax',
    };

    if (state.gatherAxis === 'section') {
      const key1 = parseIntegerField(context.key1, 'Section key1', errors);
      const x0 = parseIntegerField(context.x0, 'Trace start', errors);
      const x1 = parseIntegerField(context.x1, 'Trace end', errors);
      if (Number.isInteger(x0) && Number.isInteger(x1) && x1 < x0) {
        errors.push('Trace end must be greater than or equal to trace start.');
      }
      payload.key1 = key1;
      payload.x0 = x0;
      payload.x1 = x1;
    } else {
      const endpointKey = String(state.gatherEndpointKey || '').trim();
      if (!endpointKey) {
        const label = state.gatherAxis === 'receiver' ? 'Receiver station' : 'Source station';
        errors.push(`${label}を選択してください.`);
      }
      payload.endpoint_key = endpointKey;
    }

    if (state.gatherDisplayMode === 'reduced_time') {
      const velocity = parseFiniteNumberField(
        state.gatherReductionVelocity,
        'Reduction velocity',
        errors,
      );
      if (Number.isFinite(velocity) && velocity <= 0) {
        errors.push('Reduction velocity must be greater than 0.');
      }
      payload.reduction_velocity_m_s = velocity;
    }

    return { payload, errors };
  }

  function makeGatherField(label, input) {
    const field = document.createElement('label');
    field.className = 'refraction-qc-field';
    field.append(document.createTextNode(label), input);
    return field;
  }

  function makeGatherInput(type, value, testId, onInput, options = {}) {
    const input = document.createElement('input');
    input.type = type;
    input.value = value ?? '';
    input.autocomplete = 'off';
    input.dataset.testid = testId;
    if (options.min !== undefined) input.min = String(options.min);
    if (options.step !== undefined) input.step = String(options.step);
    input.addEventListener('input', () => {
      onInput(input.value);
    });
    return input;
  }

  function makeGatherSelect(value, testId, options, onChange) {
    const select = document.createElement('select');
    select.dataset.testid = testId;
    for (const [optionValue, label] of options) {
      const option = document.createElement('option');
      option.value = optionValue;
      option.textContent = label;
      select.appendChild(option);
    }
    select.value = value;
    select.addEventListener('change', () => {
      onChange(select.value);
    });
    return select;
  }

  function createGatherEndpointControls() {
    const endpointKind = state.gatherAxis === 'receiver' ? 'receiver' : 'source';
    const label = endpointKind === 'receiver' ? 'Receiver station' : 'Source station';
    const options = buildGatherEndpointOptions(state.qcBundle, endpointKind);
    const hasSelectedOption = options.some((option) => option.value === state.gatherEndpointKey);
    if (state.gatherEndpointKey && !hasSelectedOption) {
      options.push({
        value: state.gatherEndpointKey,
        label: `${state.gatherEndpointKey} · selected from first-break pick`,
        stationId: state.gatherEndpointKey,
      });
    }
    const searchText = normalizedText(state.gatherEndpointSearch);
    const filteredOptions = searchText
      ? options.filter((option) => (
        normalizedText(option.label).includes(searchText)
        || normalizedText(option.value).includes(searchText)
      ))
      : options;

    const search = makeGatherInput(
      'search',
      state.gatherEndpointSearch,
      'refraction-qc-gather-endpoint-search',
      (value) => { state.gatherEndpointSearch = value.trim(); render(); },
    );
    search.placeholder = `Search ${label.toLowerCase()}`;

    const select = document.createElement('select');
    select.dataset.testid = 'refraction-qc-gather-endpoint';
    select.disabled = !options.length;
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = options.length ? `Select ${label.toLowerCase()}` : `No ${label.toLowerCase()} candidates`;
    select.appendChild(placeholder);
    for (const optionData of filteredOptions) {
      const option = document.createElement('option');
      option.value = optionData.value;
      option.textContent = optionData.label;
      select.appendChild(option);
    }
    select.value = state.gatherEndpointKey;
    select.addEventListener('change', () => {
      state.gatherEndpointKey = select.value;
      render();
    });

    const fragment = document.createDocumentFragment();
    fragment.append(
      makeGatherField(`${label} search`, search),
      makeGatherField(label, select),
    );
    if (!options.length || (options.length && !filteredOptions.length)) {
      const empty = document.createElement('p');
      empty.className = 'refraction-qc-placeholder refraction-qc-gather-endpoint-empty';
      empty.dataset.testid = 'refraction-qc-gather-endpoint-empty';
      empty.textContent = !options.length
        ? `No ${label.toLowerCase()} candidates are present in static components.`
        : `No ${label.toLowerCase()} candidates match the search.`;
      fragment.appendChild(empty);
    }
    return fragment;
  }

  function createGatherEndpointDetails() {
    const details = document.createElement('details');
    details.className = 'refraction-qc-gather-details';
    details.dataset.testid = 'refraction-qc-gather-endpoint-details';
    const summary = document.createElement('summary');
    summary.textContent = 'Advanced details';
    const row = document.createElement('div');
    row.className = 'refraction-qc-gather-detail-row';
    const label = document.createElement('span');
    label.textContent = 'endpoint_key';
    const value = document.createElement('code');
    value.dataset.testid = 'refraction-qc-gather-endpoint-key';
    value.textContent = state.gatherEndpointKey || 'not selected';
    const copy = document.createElement('button');
    copy.type = 'button';
    copy.textContent = 'Copy';
    copy.disabled = !state.gatherEndpointKey;
    copy.dataset.testid = 'refraction-qc-gather-endpoint-copy';
    copy.addEventListener('click', async () => {
      if (!state.gatherEndpointKey || !navigator.clipboard?.writeText) return;
      await navigator.clipboard.writeText(state.gatherEndpointKey);
    });
    row.append(label, value, copy);
    details.append(summary, row);
    return details;
  }

  function createGatherPreviewControls() {
    const context = currentGatherContext();
    const form = document.createElement('form');
    form.className = 'refraction-qc-gather-controls';
    form.dataset.testid = 'refraction-qc-gather-controls';
    form.noValidate = true;
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      loadGatherPreview();
    });

    const axis = makeGatherSelect(
      state.gatherAxis,
      'refraction-qc-gather-axis',
      Object.entries(GATHER_AXIS_LABELS),
      (value) => {
        state.gatherAxis = value;
        state.gatherEndpointKey = '';
        state.gatherEndpointSearch = '';
        render();
      },
    );
    const mode = makeGatherSelect(
      state.gatherDisplayMode,
      'refraction-qc-gather-display',
      Object.entries(GATHER_DISPLAY_LABELS),
      (value) => {
        state.gatherDisplayMode = value;
        render();
      },
    );
    form.append(
      makeGatherField('Gather', axis),
      makeGatherField('Display', mode),
      makeGatherField('File ID', makeGatherInput(
        'text',
        context.fileId,
        'refraction-qc-gather-file-id',
        (value) => { state.gatherFileId = value.trim(); },
      )),
      makeGatherField('key1 byte', makeGatherInput(
        'number',
        context.key1Byte,
        'refraction-qc-gather-key1-byte',
        (value) => { state.gatherKey1Byte = value.trim(); },
        { min: 1, step: 1 },
      )),
      makeGatherField('key2 byte', makeGatherInput(
        'number',
        context.key2Byte,
        'refraction-qc-gather-key2-byte',
        (value) => { state.gatherKey2Byte = value.trim(); },
        { min: 1, step: 1 },
      )),
    );

    if (state.gatherAxis === 'section') {
      form.append(
        makeGatherField('key1', makeGatherInput(
          'number',
          context.key1,
          'refraction-qc-gather-key1',
          (value) => { state.gatherSectionKey1 = value.trim(); },
          { step: 1 },
        )),
        makeGatherField('Trace start', makeGatherInput(
          'number',
          context.x0,
          'refraction-qc-gather-x0',
          (value) => { state.gatherX0 = value.trim(); },
          { min: 0, step: 1 },
        )),
        makeGatherField('Trace end', makeGatherInput(
          'number',
          context.x1,
          'refraction-qc-gather-x1',
          (value) => { state.gatherX1 = value.trim(); },
          { min: 0, step: 1 },
        )),
      );
    } else {
      form.appendChild(createGatherEndpointControls());
    }

    form.append(
      makeGatherField('Time start (s)', makeGatherInput(
        'number',
        state.gatherTimeStartS,
        'refraction-qc-gather-time-start',
        (value) => { state.gatherTimeStartS = value.trim(); },
        { min: 0, step: '0.001' },
      )),
      makeGatherField('Time end (s)', makeGatherInput(
        'number',
        state.gatherTimeEndS,
        'refraction-qc-gather-time-end',
        (value) => { state.gatherTimeEndS = value.trim(); },
        { min: 0, step: '0.001' },
      )),
      makeGatherField('Max traces', makeGatherInput(
        'number',
        String(state.gatherMaxTraces),
        'refraction-qc-gather-max-traces',
        (value) => { state.gatherMaxTraces = value.trim(); },
        { min: 1, step: 1 },
      )),
    );

    if (state.gatherDisplayMode === 'reduced_time') {
      form.appendChild(makeGatherField('Reduction velocity (m/s)', makeGatherInput(
        'number',
        state.gatherReductionVelocity,
        'refraction-qc-gather-reduction-velocity',
        (value) => { state.gatherReductionVelocity = value.trim(); },
        { min: 1, step: '0.1' },
      )));
    }

    const actions = document.createElement('div');
    actions.className = 'refraction-qc-actions';
    const loadButton = document.createElement('button');
    loadButton.type = 'submit';
    loadButton.disabled = state.gatherLoading;
    loadButton.textContent = state.gatherLoading ? 'Loading preview...' : 'Load Preview';
    loadButton.dataset.testid = 'refraction-qc-gather-load';
    actions.appendChild(loadButton);
    form.appendChild(actions);
    if (state.gatherAxis !== 'section') {
      form.appendChild(createGatherEndpointDetails());
    }
    return form;
  }

  function gatherAxisValues(preview) {
    const offsets = Array.isArray(preview.offset_m) ? preview.offset_m : [];
    const hasOffsets = offsets.length && offsets.every((value) => Number.isFinite(Number(value)));
    if (hasOffsets) {
      return {
        x: offsets.map((value) => Number(value)),
        label: 'Offset (m)',
      };
    }
    return {
      x: (Array.isArray(preview.x_indices) ? preview.x_indices : []).map((value) => Number(value)),
      label: 'Trace position',
    };
  }

  function gatherTimeValues(preview) {
    const rowCount = Array.isArray(preview.raw_samples) ? preview.raw_samples.length : 0;
    const dt = Number(preview.dt_s);
    const windowMeta = preview.window || {};
    const startSample = Number(windowMeta.sample_start ?? windowMeta.y0 ?? 0);
    const stepY = Number(windowMeta.effective_step_y ?? windowMeta.step_y ?? 1);
    const out = [];
    for (let index = 0; index < rowCount; index += 1) {
      out.push((startSample + index * stepY) * dt);
    }
    return out;
  }

  function finitePairPoints(xValues, yValues) {
    const x = [];
    const y = [];
    const indices = [];
    for (let index = 0; index < xValues.length && index < yValues.length; index += 1) {
      const xValue = Number(xValues[index]);
      const yValue = Number(yValues[index]);
      if (!Number.isFinite(xValue) || !Number.isFinite(yValue)) continue;
      x.push(xValue);
      y.push(yValue);
      indices.push(index);
    }
    return { x, y, indices };
  }

  function gatherOverlayTrace(preview, xValues, options) {
    const times = Array.isArray(preview[options.field]) ? preview[options.field] : [];
    const points = finitePairPoints(xValues, times);
    if (!points.x.length) return null;
    const residuals = Array.isArray(preview.residual_s) ? preview.residual_s : [];
    const residualMs = points.indices.map((index) => {
      const value = Number(residuals[index]);
      return Number.isFinite(value) ? value * 1000.0 : NaN;
    });
    const hasResidual = options.residual && residualMs.some((value) => Number.isFinite(value));
    return {
      name: options.name,
      type: 'scatter',
      mode: 'markers',
      x: points.x,
      y: points.y,
      hovertemplate: `${options.name}<br>x=%{x}<br>time=%{y:.4f} s<extra></extra>`,
      marker: {
        color: hasResidual ? residualMs : options.color,
        colorscale: hasResidual ? 'RdBu' : undefined,
        reversescale: hasResidual ? true : undefined,
        cmid: hasResidual ? 0 : undefined,
        showscale: hasResidual,
        colorbar: hasResidual ? { title: { text: 'Residual (ms)' } } : undefined,
        symbol: options.symbol,
        size: options.size || 8,
        line: { color: '#0f172a', width: 0.5 },
      },
    };
  }

  function correctedStatusMessage(preview) {
    const ref = preview?.corrected_window_ref || {};
    const status = String(ref.status || '').trim();
    if (!status) return '';
    if (status === 'ok') {
      return `Corrected data source: ${textOrDash(preview.corrected_samples_source)}.`;
    }
    const message = String(ref.message || '').trim();
    return `Missing corrected data: status ${status}. ${message || `Preview samples source: ${textOrDash(preview.corrected_samples_source)}.`}`;
  }

  function renderGatherHeatmap(content, preview, options) {
    const plot = createFirstBreakPlot(options.testId);
    plot.classList.add('refraction-qc-gather-plot');
    plot.dataset.pointCount = String(preview?.shape?.[1] || 0);
    content.appendChild(plot);

    if (!window.Plotly) {
      plot.textContent = 'Plot library is unavailable.';
      return;
    }

    const xAxis = gatherAxisValues(preview);
    const y = gatherTimeValues(preview);
    const traces = [
      {
        name: options.title,
        type: 'heatmap',
        x: xAxis.x,
        y,
        z: options.samples,
        colorscale: 'Greys',
        zsmooth: false,
        colorbar: { title: { text: 'Amplitude' } },
        hovertemplate: `${options.title}<br>${xAxis.label}: %{x}<br>Time: %{y:.4f} s<br>Amplitude: %{z:.4g}<extra></extra>`,
      },
    ];
    const observedField = options.corrected
      ? 'corrected_observed_pick_time_s'
      : 'observed_pick_time_s';
    const modeledField = options.corrected
      ? 'corrected_modeled_pick_time_s'
      : 'modeled_pick_time_s';
    const observedTrace = gatherOverlayTrace(preview, xAxis.x, {
      field: observedField,
      name: options.corrected ? 'Corrected observed first break' : 'Observed first break',
      color: '#2563eb',
      symbol: 'circle',
      residual: true,
    });
    const modeledTrace = gatherOverlayTrace(preview, xAxis.x, {
      field: modeledField,
      name: options.corrected ? 'Corrected modeled first break' : 'Modeled first break',
      color: '#f97316',
      symbol: 'x',
      size: 9,
    });
    if (observedTrace) traces.push(observedTrace);
    if (modeledTrace) traces.push(modeledTrace);

    window.Plotly.newPlot(plot, traces, {
      height: plotHeight(320, 520),
      margin: { l: 62, r: 18, t: 38, b: 52 },
      font: { size: 10, color: '#334155' },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      title: { text: options.title, font: { size: 12 } },
      xaxis: {
        title: { text: xAxis.label },
        zeroline: false,
        gridcolor: '#e5e7eb',
      },
      yaxis: {
        title: { text: 'Time (s)' },
        autorange: 'reversed',
        zeroline: false,
        gridcolor: '#e5e7eb',
      },
      legend: {
        orientation: 'h',
        x: 0,
        y: 1.16,
        xanchor: 'left',
        yanchor: 'top',
        font: { size: 10 },
      },
    }, { displayModeBar: false, responsive: true });
  }

  function renderGatherReducedTime(content, preview) {
    const plot = createFirstBreakPlot('refraction-qc-gather-reduced-plot');
    plot.classList.add('refraction-qc-gather-plot');
    content.appendChild(plot);

    if (!window.Plotly) {
      plot.textContent = 'Plot library is unavailable.';
      return;
    }

    const xAxis = gatherAxisValues(preview);
    const traces = [];
    const observed = gatherOverlayTrace(preview, xAxis.x, {
      field: 'reduced_observed_time_s',
      name: 'Reduced observed first break',
      color: '#2563eb',
      symbol: 'circle',
      residual: true,
    });
    const modeled = gatherOverlayTrace(preview, xAxis.x, {
      field: 'reduced_modeled_time_s',
      name: 'Reduced modeled first break',
      color: '#f97316',
      symbol: 'x',
      size: 9,
    });
    if (observed) traces.push(observed);
    if (modeled) traces.push(modeled);

    if (!traces.length) {
      plot.textContent = 'Reduced-time values are unavailable for this preview.';
      return;
    }

    window.Plotly.newPlot(plot, traces, {
      height: plotHeight(320, 520),
      margin: { l: 62, r: 18, t: 38, b: 52 },
      font: { size: 10, color: '#334155' },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      title: { text: 'Reduced-time first-break preview', font: { size: 12 } },
      xaxis: {
        title: { text: xAxis.label },
        zeroline: false,
        gridcolor: '#e5e7eb',
      },
      yaxis: {
        title: { text: 'Reduced time (s)' },
        zeroline: true,
        zerolinecolor: '#94a3b8',
        gridcolor: '#e5e7eb',
      },
      legend: {
        orientation: 'h',
        x: 0,
        y: 1.16,
        xanchor: 'left',
        yanchor: 'top',
        font: { size: 10 },
      },
    }, { displayModeBar: false, responsive: true });
  }

  function renderGatherPreviewPayload(content, preview) {
    const statusMessage = correctedStatusMessage(preview);
    content.appendChild(createKv([
      ['Job ID', preview.job_id],
      ['Gather', GATHER_AXIS_LABELS[preview.gather?.axis] || preview.gather?.axis],
      ['Endpoint', preview.gather?.endpoint_key],
      ['Trace count', `${preview.window?.returned_trace_count || 0} of ${preview.window?.requested_trace_count || 0}`],
      ['Samples', `${preview.window?.returned_sample_count || 0} of ${preview.window?.requested_sample_count || 0}`],
      ['dt', `${formatNumber(Number(preview.dt_s), 6)} s`],
      ['Display', GATHER_DISPLAY_LABELS[state.gatherDisplayMode] || state.gatherDisplayMode],
      ['Corrected source', preview.corrected_samples_source],
    ]));

    const signNote = document.createElement('p');
    signNote.className = 'refraction-qc-note';
    signNote.textContent = `${preview.sign_convention}; positive shift_s delays displayed events and negative shift_s advances displayed events.`;
    signNote.dataset.testid = 'refraction-qc-gather-sign-note';
    content.appendChild(signNote);

    if (statusMessage) {
      const status = document.createElement('p');
      status.className = 'refraction-qc-note';
      status.textContent = statusMessage;
      status.dataset.testid = 'refraction-qc-gather-corrected-status';
      content.appendChild(status);
    }

    const overlayNote = document.createElement('p');
    overlayNote.className = 'refraction-qc-note';
    overlayNote.textContent = `First-break overlays: ${textOrDash(preview.overlay_status?.first_break_fit)}; residual colors use observed - modeled.`;
    overlayNote.dataset.testid = 'refraction-qc-gather-overlay-status';
    content.appendChild(overlayNote);

    if (state.gatherDisplayMode === 'reduced_time') {
      renderGatherReducedTime(content, preview);
      return;
    }

    const plotGrid = document.createElement('div');
    plotGrid.className = state.gatherDisplayMode === 'side_by_side'
      ? 'refraction-qc-gather-grid'
      : 'refraction-qc-plot-grid';
    content.appendChild(plotGrid);

    if (state.gatherDisplayMode === 'raw' || state.gatherDisplayMode === 'side_by_side') {
      renderGatherHeatmap(plotGrid, preview, {
        testId: 'refraction-qc-gather-raw-plot',
        title: 'Raw gather',
        samples: preview.raw_samples,
        corrected: false,
      });
    }
    if (state.gatherDisplayMode === 'corrected' || state.gatherDisplayMode === 'side_by_side') {
      renderGatherHeatmap(plotGrid, preview, {
        testId: 'refraction-qc-gather-corrected-plot',
        title: 'Corrected gather',
        samples: preview.corrected_samples,
        corrected: true,
      });
    }
  }

  function renderGatherPreview(content, bundle) {
    content.appendChild(createGatherPreviewControls());
    if (state.gatherLoading) {
      const message = document.createElement('p');
      message.className = 'refraction-qc-placeholder';
      message.textContent = 'Loading bounded gather preview from the M6 API...';
      content.appendChild(message);
    }
    if (state.gatherError) {
      const error = document.createElement('p');
      error.className = 'refraction-qc-error';
      error.dataset.testid = 'refraction-qc-gather-error';
      error.textContent = state.gatherError;
      content.appendChild(error);
    }
    if (!state.gatherPreview && !state.gatherLoading && !state.gatherError) {
      const message = document.createElement('p');
      message.className = 'refraction-qc-placeholder';
      message.textContent = bundle
        ? 'Load a bounded gather preview from the M6 API.'
        : 'Load a QC bundle before requesting a gather preview.';
      content.appendChild(message);
    }
    if (state.gatherPreview) {
      renderGatherPreviewPayload(content, state.gatherPreview);
    }
    content.appendChild(createKv([
      ['Selected endpoint kind', state.selectedEndpointKind],
      ['Endpoint', state.selectedEndpoint],
      ['Gather endpoint', state.gatherEndpointKey],
      ['Cell', selectedCellLabel(state.selectedCell)],
      ['Layer', state.selectedLayerKind],
    ]));
  }

  function renderPickMap(content, viewConfig = PICK_MAP_VIEWS.pick_map) {
    cleanupPickMapCanvasRenderer();
    if (state.pickMap && !state.pickMap.has_after_statics && state.pickMapDisplayMode !== 'before') {
      state.pickMapDisplayMode = 'before';
    }
    const testIdPrefix = viewConfig.testIdPrefix;

    const controls = document.createElement('div');
    controls.className = 'refraction-qc-controls';

    const beforeButton = document.createElement('button');
    beforeButton.type = 'button';
    beforeButton.textContent = 'Before Statics';
    beforeButton.dataset.testid = `${testIdPrefix}-before`;
    beforeButton.className = state.pickMapDisplayMode === 'before' ? 'is-active' : '';
    beforeButton.addEventListener('click', () => {
      state.pickMapDisplayMode = 'before';
      render();
    });

    const afterButton = document.createElement('button');
    afterButton.type = 'button';
    afterButton.textContent = 'After Statics';
    afterButton.dataset.testid = `${testIdPrefix}-after`;
    afterButton.disabled = !state.pickMap?.has_after_statics;
    afterButton.className = state.pickMapDisplayMode === 'after' ? 'is-active' : '';
    afterButton.addEventListener('click', () => {
      if (!state.pickMap?.has_after_statics) return;
      state.pickMapDisplayMode = 'after';
      render();
    });

    const gatherStart = document.createElement('input');
    gatherStart.type = 'number';
    gatherStart.placeholder = 'Gather start';
    gatherStart.value = state.pickMapGatherStart;
    gatherStart.dataset.testid = `${testIdPrefix}-gather-start`;
    gatherStart.addEventListener('input', () => {
      state.pickMapGatherStart = gatherStart.value;
      render();
    });

    const gatherEnd = document.createElement('input');
    gatherEnd.type = 'number';
    gatherEnd.placeholder = 'Gather end';
    gatherEnd.value = state.pickMapGatherEnd;
    gatherEnd.dataset.testid = `${testIdPrefix}-gather-end`;
    gatherEnd.addEventListener('input', () => {
      state.pickMapGatherEnd = gatherEnd.value;
      render();
    });

    const cachedButton = document.createElement('button');
    cachedButton.type = 'button';
    cachedButton.textContent = 'Load from Static Correction NPZ';
    cachedButton.dataset.testid = `${testIdPrefix}-load-cached`;
    cachedButton.disabled = !state.pickMapCachedFile;
    cachedButton.addEventListener('click', () => {
      loadPreStaticsPickMap(state.pickMapCachedFile);
    });

    const completedButton = document.createElement('button');
    completedButton.type = 'button';
    completedButton.textContent = `Load completed-job ${viewConfig.label}`;
    completedButton.dataset.testid = `${testIdPrefix}-load-job`;
    completedButton.disabled = !(state.qcBundle?.job_id || state.selectedJobId);
    completedButton.addEventListener('click', () => {
      loadCompletedPickMap();
    });

    controls.append(beforeButton, afterButton, gatherStart, gatherEnd, cachedButton, completedButton);
    content.appendChild(controls);

    if (state.pickMapCacheStatus) {
      const cacheStatus = document.createElement('p');
      cacheStatus.className = 'refraction-qc-note';
      cacheStatus.dataset.testid = `${testIdPrefix}-cache-status`;
      cacheStatus.textContent = state.pickMapCacheStatus;
      if (!state.pickMapCachedFile && activePickMapTarget()) {
        cacheStatus.appendChild(document.createTextNode(' '));
        const link = document.createElement('a');
        link.href = staticCorrectionLinkForTarget(activePickMapTarget());
        link.textContent = 'Open Static Correction';
        cacheStatus.appendChild(link);
      }
      content.appendChild(cacheStatus);
    }
    if (state.pickMapLoading) {
      const loading = document.createElement('p');
      loading.className = 'refraction-qc-placeholder';
      loading.textContent = 'Loading Pick Map...';
      content.appendChild(loading);
      return;
    }
    if (state.pickMapError) {
      const error = document.createElement('div');
      error.className = 'refraction-qc-error';
      error.dataset.testid = `${testIdPrefix}-error`;
      error.textContent = state.pickMapError;
      content.appendChild(error);
    }
    if (!state.pickMap) {
      const empty = document.createElement('p');
      empty.className = 'refraction-qc-placeholder';
      empty.textContent = state.qcBundle
        ? 'No Pick Map loaded for this static job.'
        : 'Load a completed job or a cached Static Correction NPZ.';
      content.appendChild(empty);
      return;
    }

    const status = document.createElement('p');
    status.className = 'refraction-qc-note';
    status.dataset.testid = `${testIdPrefix}-status`;
    status.textContent = state.pickMap.status_message || '';
    content.appendChild(status);

    const pointResult = filteredPickMapPoints(state.pickMap, viewConfig);
    const points = pointResult.points;
    content.appendChild(createKv([
      ['Mode', state.pickMap.mode],
      ['Receiver numbering', state.pickMap.receiver_number_mode],
      ['Gather min', state.pickMap.gather_range?.min],
      ['Gather max', state.pickMap.gather_range?.max],
      ['Displayed points', points.length],
    ]));

    const plot = document.createElement('div');
    plot.className = 'refraction-qc-plot refraction-qc-pick-map-plot';
    plot.dataset.testid = `${testIdPrefix}-plot`;
    plot.dataset.pointCount = String(points.length);
    plot.dataset.renderer = 'canvas';
    plot.dataset.xAxisTitle = viewConfig.xAxisTitle;
    plot.dataset.yAxisTitle = 'First-break pick time (ms)';
    plot.dataset.yAxisDirection = 'down';
    plot.dataset.yAxisAutorange = 'reversed';
    content.appendChild(plot);

    if (!points.length) {
      plot.textContent = pointResult.missingXCount > 0 ? viewConfig.emptyMessage : 'No Pick Map records match the current gather range.';
      return;
    }
    renderPickMapCanvas(plot, points, state.pickMap, viewConfig);
  }

  function filteredPickMapPoints(payload, viewConfig = PICK_MAP_VIEWS.pick_map) {
    const data = payload?.pick_map || {};
    const count = Array.isArray(data.pick_before_ms) ? data.pick_before_ms.length : 0;
    const start = toFiniteNumber(state.pickMapGatherStart);
    const end = toFiniteNumber(state.pickMapGatherEnd);
    const hasStart = Number.isFinite(start);
    const hasEnd = Number.isFinite(end);
    const points = [];
    let missingXCount = 0;
    for (let index = 0; index < count; index += 1) {
      const gather = data.gather_id?.[index];
      const gatherNumber = pickMapGatherNumber(gather);
      if (hasStart && Number.isFinite(gatherNumber) && gatherNumber < start) continue;
      if (hasEnd && Number.isFinite(gatherNumber) && gatherNumber > end) continue;
      const beforeMs = toFiniteNumber(data.pick_before_ms?.[index]);
      const afterMs = toFiniteNumber(data.pick_after_ms?.[index]);
      const y = state.pickMapDisplayMode === 'after' && payload.has_after_statics ? afterMs : beforeMs;
      const x = toFiniteNumber(data[viewConfig.xField]?.[index]);
      if (!Number.isFinite(y)) continue;
      if (!Number.isFinite(x)) {
        missingXCount += 1;
        continue;
      }
      const used = data.used_in_statics?.[index] === true;
      points.push({
        x,
        y,
        gather,
        sourceId: data.source_id?.[index],
        receiverId: data.receiver_id?.[index],
        beforeMs,
        afterMs,
        offsetM: toFiniteNumber(data.offset_m?.[index]),
        offsetUsed: toFiniteNumber(data.offset_used?.[index]),
        used,
        appliedShiftMs: toFiniteNumber(data.applied_shift_ms?.[index]),
      });
    }
    return { points, missingXCount };
  }

  function cleanupPickMapCanvasRenderer() {
    if (!pickMapCanvasCleanup) return;
    pickMapCanvasCleanup();
    pickMapCanvasCleanup = null;
  }

  function renderPickMapCanvas(plot, points, payload, viewConfig = PICK_MAP_VIEWS.pick_map) {
    cleanupPickMapCanvasRenderer();
    clearNode(plot);

    const canvas = document.createElement('canvas');
    canvas.className = 'refraction-qc-pick-map-canvas';
    canvas.dataset.testid = `${viewConfig.testIdPrefix}-canvas`;
    canvas.setAttribute('role', 'img');
    canvas.setAttribute(
      'aria-label',
      `${viewConfig.label} scatter plot with ${viewConfig.xAxisTitle} on x and first-break pick time in milliseconds increasing downward.`
    );
    plot.appendChild(canvas);

    const pointStats = pickMapPointStats(points, payload, viewConfig);
    plot.dataset.usedPointCount = String(pointStats.used);
    plot.dataset.unusedPointCount = String(pointStats.unused);
    plot.dataset.offsetColorCount = String(pointStats.offsetColored);

    const draw = () => drawPickMapCanvas(plot, canvas, points, payload, viewConfig);
    draw();

    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(draw);
      observer.observe(plot);
      pickMapCanvasCleanup = () => observer.disconnect();
    } else {
      window.addEventListener('resize', draw);
      pickMapCanvasCleanup = () => window.removeEventListener('resize', draw);
    }
  }

  function pickMapPointStats(points, payload, viewConfig = PICK_MAP_VIEWS.pick_map) {
    if (payload.mode !== 'completed_job') {
      let offsetColored = 0;
      for (const point of points) {
        if (viewConfig.colorByOffset && Number.isFinite(point.offsetM)) offsetColored += 1;
      }
      return { used: points.length, unused: 0, offsetColored };
    }
    let used = 0;
    let unused = 0;
    let offsetColored = 0;
    for (const point of points) {
      if (point.used) {
        used += 1;
        if (viewConfig.colorByOffset && Number.isFinite(pickMapColorValue(point))) offsetColored += 1;
      } else {
        unused += 1;
      }
    }
    return { used, unused, offsetColored };
  }

  function drawPickMapCanvas(plot, canvas, points, payload, viewConfig = PICK_MAP_VIEWS.pick_map) {
    let context = null;
    try {
      context = canvas.getContext('2d');
    } catch (_) {
      context = null;
    }
    if (!context) {
      plot.textContent = 'Canvas rendering is unavailable.';
      return;
    }

    const rect = plot.getBoundingClientRect();
    const cssWidth = Math.max(1, Math.floor(rect.width || plot.clientWidth || plot.offsetWidth || 640));
    const cssHeight = Math.max(plotHeight(340, 560), Math.floor(rect.height || plot.clientHeight || plot.offsetHeight || 0));
    const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
    const pixelWidth = Math.max(1, Math.floor(cssWidth * pixelRatio));
    const pixelHeight = Math.max(1, Math.floor(cssHeight * pixelRatio));
    if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
    if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;

    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, cssWidth, cssHeight);
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, cssWidth, cssHeight);

    const margin = { left: 66, right: 22, top: 34, bottom: 56 };
    const plotWidth = Math.max(1, cssWidth - margin.left - margin.right);
    const plotHeightCss = Math.max(1, cssHeight - margin.top - margin.bottom);
    const xRange = paddedRange(points, (point) => point.x);
    const yRange = paddedRange(points, (point) => point.y);
    const colorRange = paddedRange(points, (point) => (
      viewConfig.colorByOffset && !(payload.mode === 'completed_job' && !point.used)
        ? pickMapColorValue(point)
        : NaN
    ));
    const xScale = (value) => margin.left + ((value - xRange.min) / (xRange.max - xRange.min)) * plotWidth;
    const yScale = (value) => margin.top + ((value - yRange.min) / (yRange.max - yRange.min)) * plotHeightCss;

    drawPickMapGrid(context, margin, plotWidth, plotHeightCss, xRange, yRange, xScale, yScale, viewConfig);

    const title = `${state.pickMapDisplayMode === 'after' ? 'After Statics' : 'Before Statics'} ${viewConfig.label}`;
    context.fillStyle = '#334155';
    context.font = '12px sans-serif';
    context.textAlign = 'left';
    context.textBaseline = 'top';
    context.fillText(title, margin.left, 10);

    if (payload.mode === 'completed_job') {
      drawPickMapPoints(context, points, xScale, yScale, colorRange, payload, viewConfig, (point) => !point.used);
      drawPickMapPoints(context, points, xScale, yScale, colorRange, payload, viewConfig, (point) => point.used);
    } else {
      drawPickMapPoints(context, points, xScale, yScale, colorRange, payload, viewConfig);
    }
  }

  function paddedRange(values, valueForItem = (value) => value) {
    let min = Infinity;
    let max = -Infinity;
    let hasFinite = false;
    for (const item of values) {
      const value = valueForItem(item);
      if (!Number.isFinite(value)) continue;
      if (value < min) min = value;
      if (value > max) max = value;
      hasFinite = true;
    }
    if (!hasFinite) return { min: 0, max: 1 };
    if (min === max) {
      const pad = Math.max(1, Math.abs(min) * 0.05);
      min -= pad;
      max += pad;
    }
    return { min, max };
  }

  function pickMapTicks(range, count = 5) {
    if (!Number.isFinite(range.min) || !Number.isFinite(range.max) || range.max <= range.min) return [];
    const ticks = [];
    const steps = Math.max(1, count - 1);
    for (let index = 0; index <= steps; index += 1) {
      ticks.push(range.min + ((range.max - range.min) * index) / steps);
    }
    return ticks;
  }

  function drawPickMapGrid(context, margin, plotWidth, plotHeightCss, xRange, yRange, xScale, yScale, viewConfig) {
    const right = margin.left + plotWidth;
    const bottom = margin.top + plotHeightCss;
    context.save();
    context.strokeStyle = '#e5e7eb';
    context.lineWidth = 1;
    context.font = '10px sans-serif';
    context.fillStyle = '#334155';
    context.textBaseline = 'middle';

    for (const tick of pickMapTicks(xRange)) {
      const x = xScale(tick);
      context.beginPath();
      context.moveTo(x, margin.top);
      context.lineTo(x, bottom);
      context.stroke();
      context.textAlign = 'center';
      context.fillText(formatNumber(tick, 0), x, bottom + 16);
    }
    for (const tick of pickMapTicks(yRange)) {
      const y = yScale(tick);
      context.beginPath();
      context.moveTo(margin.left, y);
      context.lineTo(right, y);
      context.stroke();
      context.textAlign = 'right';
      context.fillText(formatNumber(tick, 1), margin.left - 8, y);
    }

    context.strokeStyle = '#94a3b8';
    context.strokeRect(margin.left, margin.top, plotWidth, plotHeightCss);
    context.textAlign = 'center';
    context.textBaseline = 'bottom';
    context.fillText(viewConfig.xAxisTitle, margin.left + plotWidth / 2, margin.top + plotHeightCss + 44);

    context.save();
    context.translate(14, margin.top + plotHeightCss / 2);
    context.rotate(-Math.PI / 2);
    context.textBaseline = 'top';
    context.fillText('First-break pick time (ms)', 0, 0);
    context.restore();
    context.restore();
  }

  function drawPickMapPoints(context, points, xScale, yScale, colorRange, payload, viewConfig, includePoint = null) {
    for (const point of points) {
      if (includePoint && !includePoint(point)) continue;
      const x = xScale(point.x);
      const y = yScale(point.y);
      context.beginPath();
      context.arc(x, y, point.used || payload.mode !== 'completed_job' ? 3 : 2.5, 0, Math.PI * 2);
      context.fillStyle = pickMapPointColor(point, colorRange, payload, viewConfig);
      context.globalAlpha = payload.mode === 'completed_job' && !point.used ? 0.35 : 0.85;
      context.fill();
    }
    context.globalAlpha = 1;
  }

  function renderStationStructure(content) {
    cleanupStationStructureCanvasRenderer();
    content.appendChild(createStationStructureControls());

    if (state.stationStructureLoading) {
      const loading = document.createElement('p');
      loading.className = 'refraction-qc-placeholder';
      loading.textContent = 'Loading station-structure QC...';
      content.appendChild(loading);
      return;
    }
    if (state.stationStructureError) {
      const error = document.createElement('div');
      error.className = 'refraction-qc-error';
      error.dataset.testid = 'refraction-qc-station-structure-error';
      error.textContent = state.stationStructureError;
      content.appendChild(error);
    }
    if (!state.stationStructure) {
      const empty = document.createElement('p');
      empty.className = 'refraction-qc-placeholder';
      empty.textContent = state.qcBundle
        ? 'Load station-structure QC for this completed refraction job.'
        : 'Load a completed job before requesting station-structure QC.';
      content.appendChild(empty);
      return;
    }

    const payload = state.stationStructure;
    const status = document.createElement('p');
    status.className = 'refraction-qc-note';
    status.dataset.testid = 'refraction-qc-station-structure-filter-status';
    status.textContent = stationStructureFilterNote(payload);
    content.appendChild(status);

    const warnings = Array.isArray(payload.warnings) ? payload.warnings : [];
    for (const warningText of warnings) {
      const warning = document.createElement('p');
      warning.className = 'refraction-qc-note';
      warning.textContent = warningText;
      content.appendChild(warning);
    }

    content.appendChild(createKv([
      ['X axis', payload.x_axis_label || payload.x_axis],
      ['Velocity', String(payload.velocity?.field || '').toUpperCase()],
      ['Depth / structure', payload.depth?.label || payload.depth?.field],
      ['Source color', payload.colors?.source || 'cyan'],
      ['Receiver color', payload.colors?.receiver || 'red'],
    ]));

    const grid = document.createElement('div');
    grid.className = 'refraction-qc-plot-grid refraction-qc-station-structure-grid';
    content.appendChild(grid);

    const panels = [
      {
        key: 'time_term',
        title: 'Time-term distribution',
        yAxisTitle: 'Time term (ms)',
        testId: 'refraction-qc-station-structure-time-term',
      },
      {
        key: 'velocity',
        title: payload.velocity?.label || 'Velocity structure',
        yAxisTitle: 'Velocity (m/s)',
        testId: 'refraction-qc-station-structure-velocity',
      },
      {
        key: 'depth',
        title: payload.depth?.label || 'Depth / structure',
        yAxisTitle: stationStructureDepthYAxisTitle(payload.depth),
        testId: 'refraction-qc-station-structure-depth',
      },
    ];

    const cleanups = [];
    for (const panelConfig of panels) {
      const plot = document.createElement('div');
      plot.className = 'refraction-qc-plot refraction-qc-station-structure-plot';
      plot.dataset.testid = `${panelConfig.testId}-plot`;
      plot.dataset.renderer = 'canvas';
      plot.dataset.xAxisTitle = payload.x_axis_label || payload.x_axis || '';
      plot.dataset.yAxisTitle = panelConfig.yAxisTitle;
      const panel = payload[panelConfig.key] || {};
      const pointCount = stationStructurePointCount(panel);
      plot.dataset.pointCount = String(pointCount);
      grid.appendChild(plot);
      if (!pointCount) {
        plot.textContent = `No finite ${panelConfig.title.toLowerCase()} values are available for the selected range.`;
        continue;
      }
      cleanups.push(renderStationStructureCanvas(plot, payload, panel, panelConfig));
    }
    stationStructureCanvasCleanup = () => {
      for (const cleanup of cleanups) cleanup();
    };
  }

  function createStationStructureControls() {
    const controls = document.createElement('div');
    controls.className = 'refraction-qc-controls';

    const gatherStart = document.createElement('input');
    gatherStart.type = 'number';
    gatherStart.placeholder = 'Shot gather start';
    gatherStart.value = state.stationStructureGatherStart;
    gatherStart.dataset.testid = 'refraction-qc-station-structure-gather-start';
    gatherStart.addEventListener('input', () => {
      state.stationStructureGatherStart = gatherStart.value;
    });

    const gatherEnd = document.createElement('input');
    gatherEnd.type = 'number';
    gatherEnd.placeholder = 'Shot gather end';
    gatherEnd.value = state.stationStructureGatherEnd;
    gatherEnd.dataset.testid = 'refraction-qc-station-structure-gather-end';
    gatherEnd.addEventListener('input', () => {
      state.stationStructureGatherEnd = gatherEnd.value;
    });

    const velocity = document.createElement('select');
    velocity.dataset.testid = 'refraction-qc-station-structure-velocity-field';
    for (const [value, label] of [
      ['auto', 'Auto'],
      ['v1', 'V1'],
      ['v2', 'V2'],
      ['v3', 'V3'],
      ['vsub', 'Vsub'],
    ]) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      velocity.appendChild(option);
    }
    velocity.value = state.stationStructureVelocityField;
    velocity.addEventListener('change', () => {
      state.stationStructureVelocityField = velocity.value;
    });

    const depth = document.createElement('select');
    depth.dataset.testid = 'refraction-qc-station-structure-depth-field';
    for (const [value, label] of [
      ['auto', 'Auto'],
      ['sh1', 'Weathering thickness SH1'],
      ['sh2', 'Weathering thickness SH2'],
      ['sh3', 'Weathering thickness SH3'],
      ['refractor_depth', 'Refractor depth'],
      ['refractor_elevation', 'Refractor elevation'],
      ['layer1_base_elevation', 'Layer 1 base elevation'],
      ['layer2_base_elevation', 'Layer 2 base elevation'],
    ]) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      depth.appendChild(option);
    }
    depth.value = state.stationStructureDepthField;
    depth.addEventListener('change', () => {
      state.stationStructureDepthField = depth.value;
    });

    const loadButton = document.createElement('button');
    loadButton.type = 'button';
    loadButton.textContent = 'Load / Refresh';
    loadButton.dataset.testid = 'refraction-qc-station-structure-load';
    loadButton.disabled = !(state.qcBundle?.job_id || state.selectedJobId);
    loadButton.addEventListener('click', () => {
      loadStationStructureQc();
    });

    controls.append(gatherStart, gatherEnd, velocity, depth, loadButton);
    return controls;
  }

  function stationStructureFilterNote(payload) {
    const status = payload.filter_status || 'unknown';
    const start = payload.gather_range?.start ?? '';
    const end = payload.gather_range?.end ?? '';
    const rangeText = start === '' && end === '' ? 'all gathers' : `shot gathers ${start || 'min'} to ${end || 'max'}`;
    if (status === 'ok') return `Filter: ${rangeText}.`;
    if (status === 'unfiltered') return 'Filter: all available source and receiver endpoints.';
    if (status === 'receiver_participation_unavailable') {
      return `Filter: source endpoints use ${rangeText}; receiver participation is unavailable from artifacts.`;
    }
    return `Filter status: ${status}.`;
  }

  function stationStructureDepthYAxisTitle(panel) {
    const field = panel?.field || '';
    if (field === 'refractor_elevation' || field === 'layer1_base_elevation' || field === 'layer2_base_elevation') {
      return 'Elevation (m)';
    }
    return 'Depth / structure (m)';
  }

  function stationStructurePointCount(panel) {
    const source = Array.isArray(panel?.source?.x) ? panel.source.x.length : 0;
    const receiver = Array.isArray(panel?.receiver?.x) ? panel.receiver.x.length : 0;
    return source + receiver;
  }

  function cleanupStationStructureCanvasRenderer() {
    if (!stationStructureCanvasCleanup) return;
    stationStructureCanvasCleanup();
    stationStructureCanvasCleanup = null;
  }

  function renderStationStructureCanvas(plot, payload, panel, panelConfig) {
    clearNode(plot);
    const canvas = document.createElement('canvas');
    canvas.className = 'refraction-qc-station-structure-canvas';
    canvas.dataset.testid = `${panelConfig.testId}-canvas`;
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', `${panelConfig.title} station-structure scatter plot.`);
    plot.appendChild(canvas);

    const draw = () => drawStationStructureCanvas(plot, canvas, payload, panel, panelConfig);
    draw();
    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(draw);
      observer.observe(plot);
      return () => observer.disconnect();
    }
    window.addEventListener('resize', draw);
    return () => window.removeEventListener('resize', draw);
  }

  function drawStationStructureCanvas(plot, canvas, payload, panel, panelConfig) {
    let context = null;
    try {
      context = canvas.getContext('2d');
    } catch (_) {
      context = null;
    }
    if (!context) {
      plot.textContent = 'Canvas rendering is unavailable.';
      return;
    }

    const points = stationStructurePoints(panel, payload.colors || {});
    const rect = plot.getBoundingClientRect();
    const cssWidth = Math.max(1, Math.floor(rect.width || plot.clientWidth || plot.offsetWidth || 640));
    const cssHeight = Math.max(plotHeight(250, 300), Math.floor(rect.height || plot.clientHeight || plot.offsetHeight || 0));
    const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
    const pixelWidth = Math.max(1, Math.floor(cssWidth * pixelRatio));
    const pixelHeight = Math.max(1, Math.floor(cssHeight * pixelRatio));
    if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
    if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;

    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, cssWidth, cssHeight);
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, cssWidth, cssHeight);

    const margin = { left: 68, right: 22, top: 34, bottom: 52 };
    const plotWidth = Math.max(1, cssWidth - margin.left - margin.right);
    const plotHeightCss = Math.max(1, cssHeight - margin.top - margin.bottom);
    const xRange = paddedRange(points, (point) => point.x);
    const yRange = paddedRange(points, (point) => point.y);
    const xScale = (value) => margin.left + ((value - xRange.min) / (xRange.max - xRange.min)) * plotWidth;
    const yScale = (value) => margin.top + plotHeightCss - ((value - yRange.min) / (yRange.max - yRange.min)) * plotHeightCss;

    drawStationStructureGrid(context, margin, plotWidth, plotHeightCss, xRange, yRange, xScale, yScale, payload, panelConfig);
    for (const point of points) {
      context.beginPath();
      context.arc(xScale(point.x), yScale(point.y), point.status === 'ok' ? 3 : 2.5, 0, Math.PI * 2);
      context.globalAlpha = point.status === 'ok' ? 0.85 : 0.35;
      context.fillStyle = point.color;
      context.fill();
    }
    context.globalAlpha = 1;
  }

  function stationStructurePoints(panel, colors) {
    const points = [];
    for (const side of ['source', 'receiver']) {
      const series = panel?.[side] || {};
      const count = Array.isArray(series.x) ? series.x.length : 0;
      for (let index = 0; index < count; index += 1) {
        const x = toFiniteNumber(series.x[index]);
        const y = toFiniteNumber(series.y?.[index]);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        points.push({
          x,
          y,
          side,
          color: side === 'source' ? (colors.source || 'cyan') : (colors.receiver || 'red'),
          status: String(series.status?.[index] || 'ok'),
        });
      }
    }
    return points;
  }

  function drawStationStructureGrid(context, margin, plotWidth, plotHeightCss, xRange, yRange, xScale, yScale, payload, panelConfig) {
    const right = margin.left + plotWidth;
    const bottom = margin.top + plotHeightCss;
    context.save();
    context.strokeStyle = '#e5e7eb';
    context.lineWidth = 1;
    context.font = '10px sans-serif';
    context.fillStyle = '#334155';
    context.textBaseline = 'middle';
    for (const tick of pickMapTicks(xRange)) {
      const x = xScale(tick);
      context.beginPath();
      context.moveTo(x, margin.top);
      context.lineTo(x, bottom);
      context.stroke();
      context.textAlign = 'center';
      context.fillText(formatNumber(tick, 0), x, bottom + 16);
    }
    for (const tick of pickMapTicks(yRange)) {
      const y = yScale(tick);
      context.beginPath();
      context.moveTo(margin.left, y);
      context.lineTo(right, y);
      context.stroke();
      context.textAlign = 'right';
      context.fillText(formatNumber(tick, 1), margin.left - 8, y);
    }
    context.strokeStyle = '#94a3b8';
    context.strokeRect(margin.left, margin.top, plotWidth, plotHeightCss);
    context.textAlign = 'left';
    context.textBaseline = 'top';
    context.fillText(panelConfig.title, margin.left, 10);
    context.fillStyle = payload.colors?.source || 'cyan';
    context.fillText('source', right - 90, 10);
    context.fillStyle = payload.colors?.receiver || 'red';
    context.fillText('receiver', right - 48, 10);
    context.fillStyle = '#334155';
    context.textAlign = 'center';
    context.textBaseline = 'bottom';
    context.fillText(payload.x_axis_label || payload.x_axis || 'Station', margin.left + plotWidth / 2, bottom + 42);
    context.save();
    context.translate(14, margin.top + plotHeightCss / 2);
    context.rotate(-Math.PI / 2);
    context.textBaseline = 'top';
    context.fillText(panelConfig.yAxisTitle, 0, 0);
    context.restore();
    context.restore();
  }

  function pickMapColorValue(point) {
    return Number.isFinite(point.offsetUsed) ? point.offsetUsed : point.offsetM;
  }

  function pickMapPointColor(point, colorRange, payload, viewConfig = PICK_MAP_VIEWS.pick_map) {
    if (payload.mode === 'completed_job' && !point.used) return '#94a3b8';
    if (!viewConfig.colorByOffset) return '#2563eb';
    const value = pickMapColorValue(point);
    if (!Number.isFinite(value) || !Number.isFinite(colorRange.min) || colorRange.max <= colorRange.min) {
      return '#2563eb';
    }
    const ratio = Math.max(0, Math.min(1, (value - colorRange.min) / (colorRange.max - colorRange.min)));
    const stops = [
      [68, 1, 84],
      [59, 82, 139],
      [33, 145, 140],
      [94, 201, 98],
      [253, 231, 37],
    ];
    const scaled = ratio * (stops.length - 1);
    const index = Math.min(stops.length - 2, Math.floor(scaled));
    const local = scaled - index;
    const start = stops[index];
    const end = stops[index + 1];
    const channel = (offset) => Math.round(start[offset] + (end[offset] - start[offset]) * local);
    return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
  }

  function renderViewContent(viewDef) {
    if (!dom) return;
    const content = dom.viewContents.get(viewDef.id);
    if (!content) return;
    clearNode(content);
    content.className = 'refraction-qc-placeholder';

    if (state.loading) {
      appendText(content, 'Loading QC bundle...');
      return;
    }
    if (PICK_MAP_VIEWS[viewDef.id]) {
      content.className = '';
      renderPickMap(content, PICK_MAP_VIEWS[viewDef.id]);
      return;
    }
    if (!state.qcBundle) {
      appendText(content, state.error ? 'QC bundle is not loaded.' : 'No QC bundle loaded.');
      return;
    }

    content.className = '';
    if (viewDef.id === 'summary') {
      renderSummary(content, state.qcBundle);
    } else if (viewDef.id === 'first_break_residuals') {
      renderFirstBreakPlots(content, state.qcBundle, viewDef);
    } else if (viewDef.id === 'reduced_time') {
      renderReducedTimePlot(content, state.qcBundle, viewDef);
    } else if (viewDef.id === 'profiles_2d') {
      renderProfilePlot(content, state.qcBundle, viewDef);
    } else if (viewDef.id === 'cell_maps_3d') {
      renderCellMapPlot(content, state.qcBundle, viewDef);
    } else if (viewDef.id === 'static_components') {
      renderStaticComponents(content, state.qcBundle, viewDef);
    } else if (viewDef.id === 'station_structure') {
      renderStationStructure(content);
    } else if (viewDef.id === 'gather_preview') {
      renderGatherPreview(content, state.qcBundle);
    } else {
      renderTabular(content, state.qcBundle, viewDef);
    }
  }

  function renderRecentJobs() {
    if (!dom?.jobList) return;
    clearNode(dom.jobList);
    for (const jobId of readRecentJobs()) {
      const option = document.createElement('option');
      option.value = jobId;
      dom.jobList.appendChild(option);
    }
  }

  function render() {
    if (!dom) return;
    if (!PICK_MAP_VIEWS[state.selectedView]) cleanupPickMapCanvasRenderer();
    if (state.selectedView !== 'station_structure') cleanupStationStructureCanvasRenderer();

    dom.loadButton.disabled = state.loading;
    dom.maxPoints.value = String(state.maxPoints);
    dom.layerKind.value = state.selectedLayerKind;
    dom.xAxisMode.value = state.firstBreakXAxis;
    dom.profileGroup.value = state.selectedProfileGroup;
    dom.profileUnits.value = state.selectedProfileUnits;
    dom.statusFilter.value = state.profileStatusFilter;
    dom.mapQuantity.value = state.selectedCellMapQuantity;
    dom.showRejected.checked = state.showRejectedFirstBreaks;
    dom.endpointKind.value = state.selectedEndpointKind;
    dom.endpoint.value = state.selectedEndpoint;
    dom.trace.value = state.selectedTraceIndex;
    dom.cell.value = state.selectedCell?.cell_ix !== undefined
      ? `${state.selectedCell.cell_ix},${state.selectedCell.cell_iy}`
      : (state.selectedCell?.text || '');

    if (state.loading) {
      dom.status.textContent = 'Loading QC bundle...';
    } else if (state.qcBundle) {
      const available = Array.isArray(state.qcBundle.available_views)
        ? state.qcBundle.available_views.length
        : 0;
      const unavailable = Array.isArray(state.qcBundle.unavailable_views)
        ? state.qcBundle.unavailable_views.length
        : 0;
      dom.status.textContent = `Loaded ${state.qcBundle.job_id}: ${available} available, ${unavailable} unavailable.`;
    } else {
      dom.status.textContent = 'No QC bundle loaded.';
    }

    dom.error.hidden = !state.error;
    dom.error.textContent = state.error || '';

    const signConvention = state.qcBundle?.sign_convention;
    dom.sign.hidden = !signConvention;
    dom.sign.textContent = signConvention ? `Sign: ${signConvention}` : '';

    for (const button of dom.viewButtons) {
      const active = button.dataset.view === state.selectedView;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-selected', active ? 'true' : 'false');
    }
    for (const panel of dom.viewPanels) {
      panel.hidden = panel.dataset.viewPanel !== state.selectedView;
    }
    const selectedViewDef = VIEW_DEFS.find((viewDef) => viewDef.id === state.selectedView);
    if (selectedViewDef) renderViewContent(selectedViewDef);
  }

  function setSelectedView(viewId) {
    if (!VIEW_DEFS.some((view) => view.id === viewId)) return;
    state.selectedView = viewId;
    render();
    if (PICK_MAP_VIEWS[viewId]) {
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

  async function readError(response, label = 'QC bundle request') {
    try {
      const text = await response.text();
      if (!text) return `${label} failed with status ${response.status}`;
      try {
        const payload = JSON.parse(text);
        if (payload && typeof payload.detail === 'string') return payload.detail;
        if (payload && payload.detail) return JSON.stringify(payload.detail);
      } catch (_) {
      }
      return text;
    } catch (_) {
    }
    return `${label} failed with status ${response.status}`;
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
    const formData = new FormData();
    formData.append('request_json', JSON.stringify(payload));
    formData.append('pick_npz', file, file.name || 'first_breaks.npz');
    await loadPickMapRequest(() => fetch('/statics/refraction/qc/pick-map', {
      method: 'POST',
      body: formData,
    }));
  }

  async function loadCompletedPickMap(jobId) {
    const cleanJobId = String(jobId || state.qcBundle?.job_id || state.selectedJobId || '').trim();
    if (!cleanJobId) {
      state.pickMapError = 'Job ID is required for completed-job Pick Map.';
      render();
      return;
    }
    await loadPickMapRequest(() => fetch('/statics/refraction/qc/pick-map', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: cleanJobId }),
    }));
  }

  async function loadPickMapRequest(requestFactory) {
    const serial = ++pickMapRequestSerial;
    state.pickMapLoading = true;
    state.pickMapError = null;
    render();
    try {
      const response = await requestFactory();
      if (!response.ok) {
        throw new Error(await readError(response, 'Pick Map request'));
      }
      const pickMap = await response.json();
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
      const response = await fetch('/statics/refraction/qc/station-structure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildStationStructureRequest(cleanJobId)),
      });
      if (!response.ok) {
        throw new Error(await readError(response, 'Station-structure QC request'));
      }
      const payload = await response.json();
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
      const response = await fetch('/statics/refraction/qc/gather-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(await readError(response, 'Gather preview request'));
      }
      const preview = await response.json();
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

  function writeJobIdUrlParam(jobId) {
    if (!window.history || !window.location) return;
    try {
      const url = new URL(window.location.href);
      url.searchParams.set('refraction_job_id', jobId);
      url.searchParams.delete('refraction_qc_job_id');
      window.history.replaceState(window.history.state, '', url);
    } catch (_) {
    }
  }

  async function loadBundle() {
    if (!dom) return;
    const jobId = String(dom.jobId.value || '').trim();
    const maxPoints = parsePositiveInteger(dom.maxPoints.value, DEFAULT_MAX_POINTS);
    state.selectedJobId = jobId;
    state.maxPoints = maxPoints;
    state.selectedFirstBreakPick = null;
    state.firstBreakDrilldown = null;
    state.firstBreakDrilldownError = null;
    state.firstBreakDrilldownLoading = false;
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
      const response = await fetch('/statics/refraction/qc', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          include: INCLUDE_ALL,
          max_points: maxPoints,
          coordinate_mode: 'auto',
        }),
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const bundle = await response.json();
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
      writeJobIdUrlParam(jobId);
      if (PICK_MAP_VIEWS[state.selectedView] && !state.pickMap) {
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

  async function loadJob(jobId, options = {}) {
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

  function init() {
    const pipelineTab = document.getElementById('pipelineSidebarTab');
    const staticCorrectionTab = document.getElementById('staticCorrectionSidebarTab');
    const qcTab = document.getElementById('refractionQcSidebarTab');
    const pipelinePanel = document.getElementById('pipelineTabPanel');
    const staticCorrectionPanel = document.getElementById('staticCorrectionTabPanel');
    const qcPanel = document.getElementById('refractionQcTabPanel');
    const form = document.getElementById('refractionQcForm');
    if (!form) return;

    dom = {
      pipelineTab,
      staticCorrectionTab,
      qcTab,
      pipelinePanel,
      staticCorrectionPanel,
      qcPanel,
      form,
      jobId: document.getElementById('refractionQcJobId'),
      jobList: document.getElementById('refractionQcJobList'),
      maxPoints: document.getElementById('refractionQcMaxPoints'),
      loadButton: document.getElementById('refractionQcLoadButton'),
      status: document.getElementById('refractionQcStatus'),
      error: document.getElementById('refractionQcError'),
      sign: document.getElementById('refractionQcSign'),
      layerKind: document.getElementById('refractionQcLayerKind'),
      xAxisMode: document.getElementById('refractionQcXAxisMode'),
      profileGroup: document.getElementById('refractionQcProfileGroup'),
      profileUnits: document.getElementById('refractionQcProfileUnits'),
      statusFilter: document.getElementById('refractionQcStatusFilter'),
      mapQuantity: document.getElementById('refractionQcMapQuantity'),
      showRejected: document.getElementById('refractionQcShowRejected'),
      endpointKind: document.getElementById('refractionQcEndpointKind'),
      endpoint: document.getElementById('refractionQcEndpoint'),
      trace: document.getElementById('refractionQcTrace'),
      cell: document.getElementById('refractionQcCell'),
      viewButtons: Array.from(document.querySelectorAll('.refraction-qc-view-button')),
      viewPanels: Array.from(document.querySelectorAll('.refraction-qc-view')),
      viewContents: new Map(Array.from(document.querySelectorAll('[data-view-content]')).map(
        (node) => [node.dataset.viewContent, node],
      )),
    };

    if (!dom.jobId || !dom.maxPoints || !dom.loadButton || !dom.status || !dom.error || !dom.sign) return;
    if (!dom.layerKind || !dom.xAxisMode || !dom.profileGroup || !dom.profileUnits || !dom.statusFilter) return;
    if (!dom.mapQuantity) return;
    if (!dom.showRejected || !dom.endpointKind || !dom.endpoint || !dom.trace || !dom.cell) return;

    if (pipelineTab) pipelineTab.addEventListener('click', () => activateSidebarTab('pipeline'));
    if (staticCorrectionTab) staticCorrectionTab.addEventListener('click', () => activateSidebarTab('static_correction'));
    if (qcTab) qcTab.addEventListener('click', () => activateSidebarTab('refraction_qc'));
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
    dom.layerKind.addEventListener('change', () => {
      state.selectedLayerKind = dom.layerKind.value;
      render();
    });
    dom.xAxisMode.addEventListener('change', () => {
      state.firstBreakXAxis = dom.xAxisMode.value;
      render();
    });
    dom.profileGroup.addEventListener('change', () => {
      state.selectedProfileGroup = dom.profileGroup.value;
      render();
    });
    dom.profileUnits.addEventListener('change', () => {
      state.selectedProfileUnits = dom.profileUnits.value;
      render();
    });
    dom.statusFilter.addEventListener('change', () => {
      state.profileStatusFilter = dom.statusFilter.value;
      render();
    });
    dom.mapQuantity.addEventListener('change', () => {
      state.selectedCellMapQuantity = dom.mapQuantity.value;
      render();
    });
    dom.showRejected.addEventListener('change', () => {
      state.showRejectedFirstBreaks = dom.showRejected.checked;
      render();
    });
    dom.endpointKind.addEventListener('change', () => {
      state.selectedEndpointKind = dom.endpointKind.value;
      render();
    });
    dom.endpoint.addEventListener('input', () => {
      state.selectedEndpoint = dom.endpoint.value.trim();
      render();
    });
    dom.trace.addEventListener('input', () => {
      state.selectedTraceIndex = dom.trace.value.trim();
      render();
    });
    dom.cell.addEventListener('input', () => {
      state.selectedCell = parseCell(dom.cell.value);
      render();
    });
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

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
