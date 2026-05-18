(function () {
  const RECENT_JOBS_KEY = 'sv.refraction_qc.recent_jobs';
  const MAX_RECENT_JOBS = 8;
  const DEFAULT_MAX_POINTS = 20000;

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
    error: null,
    loading: false,
  };

  let dom = null;
  let requestSerial = 0;
  let gatherRequestSerial = 0;

  function safeLocalStorageValue(key) {
    try {
      return localStorage.getItem(key) || '';
    } catch (_) {
      return '';
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
    return {
      x,
      observedMs: observedS * 1000.0,
      modeledMs: modeledS * 1000.0,
      residualMs: (observedS - modeledS) * 1000.0,
      layerKind,
      status: rejected ? 'rejected' : 'used',
      opacity: rejected ? 0.42 : 0.9,
      traceIndex: textOrDash(firstDefined(record, ['trace_index_sorted', 'sorted_trace_index'])),
      source: textOrDash(firstDefined(record, ['source_endpoint_key', 'source_id'])),
      receiver: textOrDash(firstDefined(record, ['receiver_endpoint_key', 'receiver_id'])),
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

      window.Plotly.newPlot(timePlot, [
        {
          name: 'Observed',
          type: 'scatter',
          mode: 'markers',
          x: points.map((point) => point.x),
          y: points.map((point) => point.observedMs),
          text: hoverText,
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
      }, config);

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
          });
        }
        const group = residualGroups.get(groupKey);
        group.x.push(point.x);
        group.y.push(point.residualMs);
        group.text.push(plotHoverText(point));
      }
      const residualTraces = Array.from(residualGroups.values()).map((group) => ({
        name: `${layerLabel(group.layerKind)} ${group.status}`,
        type: 'scatter',
        mode: 'markers',
        x: group.x,
        y: group.y,
        text: group.text,
        hovertemplate: '%{text}<extra></extra>',
        marker: {
          color: LAYER_COLORS[group.layerKind] || LAYER_COLORS.unknown,
          symbol: group.status === 'rejected' ? 'x' : 'circle',
          size: group.status === 'rejected' ? 8 : 7,
          opacity: group.status === 'rejected' ? 0.55 : 0.9,
        },
      }));
      window.Plotly.newPlot(residualPlot, residualTraces, {
        ...commonLayout,
        title: { text: 'First-break residuals', font: { size: 12 } },
        yaxis: {
          title: { text: 'Residual (ms)' },
          zeroline: true,
          zerolinecolor: '#94a3b8',
          gridcolor: '#e5e7eb',
        },
      }, config);
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
      const endpointKey = String(state.selectedEndpoint || '').trim();
      if (!endpointKey) {
        errors.push('Endpoint is required for source and receiver gathers.');
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
      ['Cell', selectedCellLabel(state.selectedCell)],
      ['Layer', state.selectedLayerKind],
    ]));
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
      state.qcBundle = bundle;
      if (state.gatherPreview && state.gatherPreview.job_id !== bundle.job_id) {
        state.gatherPreview = null;
      }
      state.gatherError = null;
      state.error = null;
      writeRecentJob(jobId);
      writeJobIdUrlParam(jobId);
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
    setSelectedView,
    activateSidebarTab,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
