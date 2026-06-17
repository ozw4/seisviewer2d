import {
  FIRST_BREAK_SORT_CONTROL_OPTIONS,
  VIEW_DEFS,
} from './constants.js';
import * as qcApi from './api.js';
import {
  filterCellMapRecords,
  filterFirstBreakPoints,
  filterProfileRecords,
  filterReducedTimePoints,
  firstBreakResidualThresholdMs as filteredFirstBreakResidualThresholdMs,
} from './filters.js';
import { buildGatherPreviewRequest as buildGatherPreviewPayload } from './gather_api.js';
import { filterArtifactRows } from './artifact_loader.js';
import { isStandaloneRefractionQcPage } from './dom.js';
import {
  normalizeCellMapRecord as normalizeCellMapRecordData,
  normalizeFirstBreakRecord as normalizeFirstBreakRecordData,
  normalizeProfileRecord as normalizeProfileRecordData,
  normalizeReducedTimeRecord as normalizeReducedTimeRecordData,
} from './normalizers.js';
import { readRecentJobs } from './recent_jobs.js';
import { clearSelectedObject, state } from './state.js';
import { searchOrStorageValue, searchParamValue } from './url_params.js';
import {
  defaultViewForTask as taskDefaultViewForTask,
  taskForView as taskModuleTaskForView,
  viewIdsForTask as taskViewIdsForTask,
} from './tasks.js';
import { getCanvas2dContext } from './render/canvas_helpers.js';
import {
  renderActiveFilterChipsPanel,
  renderControlsPanel,
} from './render/controls.js';
import { renderInspectorPanel } from './render/inspector.js';
import { renderActiveView } from './render/index.js';
import { cleanupPickMapCanvasRenderer } from './render/pick_map.js';
import {
  getPlotly,
  newPlot as plotlyNewPlot,
  plotlyUnavailableMessage,
  resizePlot as plotlyResize,
} from './render/plotly_helpers.js';
import { cleanupStationStructureCanvasRenderer } from './render/station_structure.js';

let dom = null;
let controllerActions = null;

const QC_DRILLDOWN_MAX_OBSERVATIONS = 200;

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

const requiredAction = (name) => (...args) => {
  throw new Error(`Refraction QC render action is not configured: ${name}`);
};

const defaultActions = {
  activePickMapTarget: requiredAction('activePickMapTarget'),
  invalidateGatherRequest: requiredAction('invalidateGatherRequest'),
  invalidateQcDrilldownRequest: requiredAction('invalidateQcDrilldownRequest'),
  isCurrentQcDrilldownRequest: requiredAction('isCurrentQcDrilldownRequest'),
  loadCompletedPickMap: requiredAction('loadCompletedPickMap'),
  loadGatherPreview: requiredAction('loadGatherPreview'),
  loadPreStaticsPickMap: requiredAction('loadPreStaticsPickMap'),
  loadStationStructureQc: requiredAction('loadStationStructureQc'),
  nextQcDrilldownRequestSerial: requiredAction('nextQcDrilldownRequestSerial'),
  setSelectedView: requiredAction('setSelectedView'),
  staticCorrectionLinkForTarget: requiredAction('staticCorrectionLinkForTarget'),
};

controllerActions = defaultActions;

export function createRefractionQcRenderRuntime(options = {}) {
  controllerActions = { ...defaultActions, ...(options.actions || {}) };
  return {
    buildGatherPreviewRequest,
    render,
    renderActiveFilterChips,
    renderControlUpdate,
    renderRecentJobs,
    resetJobScopedFilters,
    reset() {
      cleanupPickMapCanvasRenderer();
      cleanupStationStructureCanvasRenderer();
    },
    setDom(nextDom) {
      dom = nextDom;
    },
    isPickMapView(viewId) {
      return Boolean(PICK_MAP_VIEWS[viewId]);
    },
  };
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
  side_by_side: 'Raw + corrected',
  reduced_time: 'Reduced-time / LMO',
};

const GATHER_AXIS_LABELS = {
  source: 'Source station',
  receiver: 'Receiver station',
  section: 'Section window',
};

export function parsePositiveInteger(value, fallback) {
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

function taskForView(viewId) {
  return taskModuleTaskForView(viewId);
}

function defaultViewForTask(taskId) {
  return taskDefaultViewForTask(taskId);
}

function viewIdsForTask(taskId) {
  return taskViewIdsForTask(taskId);
}

function createMetricCard(label, value, options = {}) {
  const card = document.createElement('div');
  card.className = options.prominent ? 'refraction-qc-metric is-prominent' : 'refraction-qc-metric';
  const key = document.createElement('span');
  key.className = 'refraction-qc-metric-label';
  key.textContent = label;
  const val = document.createElement('strong');
  val.className = 'refraction-qc-metric-value';
  val.textContent = textOrDash(value);
  card.append(key, val);
  return card;
}

function summaryValue(summary, columns) {
  return firstDefined(summary, columns);
}

function warningsCount(bundle, summary) {
  const warnings = summaryValue(summary, ['warnings_count', 'warning_count', 'n_warnings']);
  if (warnings !== undefined) return warnings;
  if (Array.isArray(summary?.warnings)) return summary.warnings.length;
  if (Array.isArray(bundle?.warnings)) return bundle.warnings.length;
  return 0;
}

function renderJobSummary() {
  if (!dom?.jobSummary) return;
  clearNode(dom.jobSummary);
  const bundle = state.qcBundle;
  const summary = bundle && typeof bundle.summary === 'object' && bundle.summary ? bundle.summary : {};
  const jobId = bundle?.job_id || state.selectedJobId || 'Not loaded';
  const fileId = summaryValue(summary, ['file_id', 'input_file_id', 'corrected_file_id']) || bundle?.file_id;
  const status = summaryValue(summary, ['job_state', 'status', 'state']);
  const method = summaryValue(summary, ['method', 'workflow']);
  const convention = bundle?.sign_convention || summaryValue(summary, ['sign_convention']);
  const totalPicks = summaryValue(summary, ['total_picks', 'total_pick_count', 'pick_count', 'observation_count']);
  const usedPicks = summaryValue(summary, ['used_picks', 'used_pick_count', 'used_observation_count']);
  const rejectedPicks = summaryValue(summary, ['rejected_picks', 'rejected_pick_count', 'rejected_observation_count']);
  const rms = summaryValue(summary, ['rms_ms', 'residual_rms_ms', 'used_rms_ms', 'all_rms_ms']);
  const mad = summaryValue(summary, ['mad_ms', 'residual_mad_ms', 'used_mad_ms']);
  const correctedStatus = summaryValue(summary, [
    'corrected_tracestore_status',
    'corrected_trace_store_status',
    'corrected_file_status',
    'apply_status',
  ]);

  dom.jobSummary.append(
    createMetricCard('Job ID', jobId, { prominent: true }),
    createMetricCard('File ID', fileId),
    createMetricCard('Method / workflow', method),
    createMetricCard('Status', status),
    createMetricCard('Sign convention', convention),
    createMetricCard('Picks total / used / rejected', [
      textOrDash(totalPicks),
      textOrDash(usedPicks),
      textOrDash(rejectedPicks),
    ].join(' / ')),
    createMetricCard('RMS / MAD', [
      textOrDash(rms),
      textOrDash(mad),
    ].join(' / ')),
    createMetricCard('Corrected TraceStore', correctedStatus),
    createMetricCard('Warnings', warningsCount(bundle, summary)),
  );
}

export function firstDefined(record, columns) {
  if (!record || typeof record !== 'object') return undefined;
  for (const column of columns) {
    const value = record[column];
    if (value !== null && value !== undefined && value !== '') return value;
  }
  return undefined;
}

export function toFiniteNumber(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : NaN;
  if (typeof value !== 'string') return NaN;
  const parsed = Number.parseFloat(value.trim());
  return Number.isFinite(parsed) ? parsed : NaN;
}

export function pickMapGatherNumber(value) {
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

function formatAsciiTick(value, digits) {
  const normalized = Math.abs(value) < Number.EPSILON ? 0 : value;
  const text = normalized.toFixed(digits).replace(/\.?0+$/, '');
  return text === '-0' ? '0' : text;
}

function residualTickDigits(maxAbs) {
  if (maxAbs >= 10) return 1;
  if (maxAbs >= 1) return 2;
  if (maxAbs >= 0.1) return 3;
  return 4;
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

function firstBreakResidualThresholdMs() {
  return filteredFirstBreakResidualThresholdMs(state);
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
  return normalizeReducedTimeRecordData(record, firstBreakXAxisDefinition());
}

function filteredReducedTimePoints(view) {
  return filterReducedTimePoints(view, state, firstBreakXAxisDefinition());
}

function normalizeFirstBreakRecord(record) {
  return normalizeFirstBreakRecordData(record, firstBreakXAxisDefinition());
}

function compareFiniteAscending(a, b) {
  const aFinite = Number.isFinite(a);
  const bFinite = Number.isFinite(b);
  if (aFinite && bFinite) return a - b;
  if (aFinite) return -1;
  if (bFinite) return 1;
  return 0;
}

function compareFirstBreakPoints(a, b) {
  const residualOrder = Math.abs(b.residualMs) - Math.abs(a.residualMs);
  const traceOrder = compareFiniteAscending(a.traceSortIndex, b.traceSortIndex);
  const offsetOrder = compareFiniteAscending(a.offsetM, b.offsetM);
  const endpointOrder = a.source.localeCompare(b.source) || a.receiver.localeCompare(b.receiver);
  if (state.firstBreakSortBy === 'trace') {
    return traceOrder || residualOrder || offsetOrder || endpointOrder;
  }
  if (state.firstBreakSortBy === 'offset') {
    return offsetOrder || residualOrder || traceOrder || endpointOrder;
  }
  return residualOrder || traceOrder || offsetOrder || endpointOrder;
}

function filteredFirstBreakPoints(view) {
  return filterFirstBreakPoints(view, state, firstBreakXAxisDefinition());
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
  return normalizeProfileRecordData(record, PROFILE_STATUS_OK);
}

function filteredProfileRecords(view) {
  return filterProfileRecords(view, state, PROFILE_STATUS_OK);
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
  return normalizeCellMapRecordData(record);
}

function filteredCellMapRecords(view) {
  return filterCellMapRecords(view, state, cellStatusCode);
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

function gatherEndpointOption(endpointKind, endpointKey) {
  const cleanKey = String(endpointKey || '').trim();
  if (!cleanKey) return null;
  return buildGatherEndpointOptions(state.qcBundle, endpointKind)
    .find((option) => option.value === cleanKey) || null;
}

function gatherEndpointSummary(endpointKind, endpointKey) {
  const cleanKey = String(endpointKey || '').trim();
  if (!cleanKey) return '';
  const option = gatherEndpointOption(endpointKind, cleanKey);
  if (option?.label) return option.label;
  const prefix = endpointKind === 'receiver' ? 'R' : 'S';
  return `${prefix} ${cleanKey}`;
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
        velocity_m_s: point.velocity,
        fold: point.fold,
        residual_rms_ms: point.residualRmsMs,
        status: point.status,
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

function profileEndpointCustomData(point) {
  return {
    endpoint_kind: point.endpointKind,
    endpoint_key: point.endpointKey,
    station_id: point.stationId,
    node_id: point.nodeId,
    inline_m: point.inline,
    static_status: point.staticStatus,
    solution_status: point.solutionStatus,
    raw: point.raw,
  };
}

function profileEndpointFromCustomData(customdata) {
  if (!customdata || typeof customdata !== 'object') return null;
  const endpointKind = normalizedText(customdata.endpoint_kind);
  const endpointKey = String(customdata.endpoint_key || '').trim();
  if (!endpointKind || !endpointKey || endpointKey === '-') return null;
  return {
    endpointKind,
    endpointKey,
    stationId: textOrDash(customdata.station_id),
    nodeId: textOrDash(customdata.node_id),
    inlineM: toFiniteNumber(customdata.inline_m),
    staticStatus: textOrDash(customdata.static_status),
    solutionStatus: textOrDash(customdata.solution_status),
    raw: customdata.raw && typeof customdata.raw === 'object' ? customdata.raw : {},
  };
}

function profileEndpointKey(endpoint) {
  return `${endpoint?.endpointKind || ''}|${endpoint?.endpointKey || ''}`;
}

function attachProfileEndpointClickActions(plot) {
  if (!plot || typeof plot.on !== 'function' || plot.dataset.profileEndpointClickAttached === 'true') return;
  plot.dataset.profileEndpointClickAttached = 'true';
  plot.on('plotly_click', (event) => {
    const endpoint = profileEndpointFromCustomData(event?.points?.[0]?.customdata);
    if (!endpoint) return;
    state.selectedProfileEndpoint = endpoint;
    state.selectedObject = {
      kind: 'endpoint',
      key: profileEndpointKey(endpoint),
      payload: endpoint,
    };
    render();
  });
}

function setEndpointFilter(endpointKind, endpointKey) {
  const cleanKey = String(endpointKey || '').trim();
  if (!endpointKind || !cleanKey) return;
  state.selectedEndpointKind = endpointKind === 'receiver' ? 'receiver' : 'source';
  state.selectedEndpoint = cleanKey;
}

function openEndpointStaticDrilldown(endpointKind, endpointKey) {
  setEndpointFilter(endpointKind, endpointKey);
  controllerActions.setSelectedView('static_components');
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

function optionLabel(options, value) {
  const match = options.find(([optionValue]) => optionValue === value);
  return match ? match[1] : textOrDash(value);
}

function createControlField(label, input) {
  const field = document.createElement('label');
  field.className = 'refraction-qc-field';
  field.append(document.createTextNode(label), input);
  return field;
}

function createSelectControl(label, testId, value, options, onChange) {
  const select = document.createElement('select');
  select.dataset.testid = testId;
  for (const [optionValue, optionLabelText] of options) {
    const option = document.createElement('option');
    option.value = optionValue;
    option.textContent = optionLabelText;
    select.appendChild(option);
  }
  select.value = value;
  select.addEventListener('change', () => {
    onChange(select.value);
    renderControlUpdate();
  });
  return createControlField(label, select);
}

function createTextControl(label, testId, value, onInput) {
  const input = document.createElement('input');
  input.type = 'text';
  input.autocomplete = 'off';
  input.value = value || '';
  input.dataset.testid = testId;
  input.addEventListener('input', () => {
    onInput(input.value.trim());
    renderControlUpdate();
  });
  return createControlField(label, input);
}

function createNumberControl(label, testId, value, onInput, options = {}) {
  const input = document.createElement('input');
  input.type = 'number';
  input.autocomplete = 'off';
  input.value = value || '';
  input.dataset.testid = testId;
  if (options.min !== undefined) input.min = String(options.min);
  if (options.step !== undefined) input.step = String(options.step);
  input.addEventListener('input', () => {
    onInput(input.value.trim());
    renderControlUpdate();
  });
  return createControlField(label, input);
}

function createCheckboxControl(label, testId, checked, onChange) {
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = checked;
  input.dataset.testid = testId;
  input.addEventListener('change', () => {
    onChange(input.checked);
    renderControlUpdate();
  });
  const field = createControlField(label, input);
  field.classList.add('refraction-qc-check-field');
  return field;
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
    offset_m: point.offsetM,
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
    offsetM: toFiniteNumber(customdata.offset_m),
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
    state.selectedObject = {
      kind: 'pick',
      key: firstBreakPickKey(pick),
      payload: pick,
    };
    state.firstBreakDrilldown = null;
    state.firstBreakDrilldownError = null;
    render();
  });
}

function gatherPreviewInputsReady() {
  const { errors } = buildGatherPreviewRequest();
  return !errors.length;
}

function previewGatherForEndpoint(endpointKind, endpointKey, options = {}) {
  const cleanKey = String(endpointKey || '').trim();
  if (!cleanKey) return;
  state.gatherAxis = endpointKind === 'receiver' ? 'receiver' : 'source';
  updateGatherEndpointSelection(cleanKey, '');
  controllerActions.setSelectedView('gather_preview');
  if (options.autoLoad !== false && gatherPreviewInputsReady()) controllerActions.loadGatherPreview();
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
    state.firstBreakDrilldown = await qcApi.fetchQcDrilldown({
      jobId: String(state.selectedJobId || dom?.jobId?.value || '').trim(),
      target: { kind: 'endpoint', endpoint_kind: endpointKind, endpoint_key: endpointKey },
      label: 'Endpoint drilldown request',
    });
  } catch (error) {
    state.firstBreakDrilldownError = error instanceof Error ? error.message : String(error);
  } finally {
    state.firstBreakDrilldownLoading = false;
    render();
  }
}

async function loadQcDrilldown(target) {
  const jobId = String(state.selectedJobId || dom?.jobId?.value || '').trim();
  const requestTarget = target && typeof target === 'object' ? { ...target } : null;
  const serial = controllerActions.nextQcDrilldownRequestSerial();
  state.qcDrilldownTarget = requestTarget;
  state.qcDrilldown = null;
  state.qcDrilldownError = null;
  state.qcDrilldownLoading = true;
  render();

  if (!jobId || !requestTarget) {
    if (!controllerActions.isCurrentQcDrilldownRequest(serial)) return;
    state.qcDrilldownLoading = false;
    state.qcDrilldownError = !jobId ? 'Job ID is required.' : 'Cell drilldown target is required.';
    render();
    return;
  }

  try {
    const payload = await qcApi.fetchQcDrilldown({
      jobId,
      target: requestTarget,
      maxObservations: QC_DRILLDOWN_MAX_OBSERVATIONS,
    });
    if (!controllerActions.isCurrentQcDrilldownRequest(serial)) return;
    state.qcDrilldown = payload;
    state.qcDrilldownError = null;
  } catch (error) {
    if (!controllerActions.isCurrentQcDrilldownRequest(serial)) return;
    state.qcDrilldown = null;
    state.qcDrilldownError = error instanceof Error ? error.message : String(error);
  } finally {
    if (controllerActions.isCurrentQcDrilldownRequest(serial)) {
      state.qcDrilldownLoading = false;
      render();
    }
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

function createEndpointActionButton(label, disabledReason, onClick, testId) {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = label;
  if (testId) button.dataset.testid = testId;
  if (disabledReason) {
    button.disabled = true;
    button.title = disabledReason;
  } else {
    button.addEventListener('click', onClick);
  }
  return button;
}

function drilldownNumber(value) {
  const number = toFiniteNumber(value);
  return Number.isFinite(number) ? number : NaN;
}

function drilldownMetric(value, digits, unit = '') {
  const number = drilldownNumber(value);
  if (!Number.isFinite(number)) return '-';
  const formatted = formatNumber(number, digits);
  return unit ? `${formatted} ${unit}` : formatted;
}

function drilldownResidualMs(record) {
  return finiteMetric(
    record,
    ['residual_time_ms', 'residual_ms'],
    ['residual_time_s', 'residual_s'],
    1000.0,
  );
}

function cellDrilldownTargetFromSelectedCell(cell) {
  if (!cell || cell.cell_ix === undefined || cell.cell_iy === undefined) return null;
  return {
    kind: 'cell',
    layer_kind: cell.layer_kind || 'v2_t1',
    cell_ix: Number(cell.cell_ix),
    cell_iy: Number(cell.cell_iy),
  };
}

function cellEndpointDisplayName(endpointKind, endpointKey) {
  const cleanKey = String(endpointKey || '').trim();
  const prefix = endpointKind === 'receiver' ? 'R' : 'S';
  const lowered = cleanKey.toLowerCase();
  if (
    cleanKey.toUpperCase().startsWith(prefix)
    || lowered.startsWith(endpointKind)
  ) {
    return cleanKey;
  }
  return `${prefix} ${cleanKey}`;
}

function endpointSummaryItems(endpoint) {
  const station = endpoint.stationId && endpoint.stationId !== '-'
    ? `${endpoint.endpointKind === 'receiver' ? 'Receiver' : 'Source'} ${endpoint.stationId}`
    : `${endpoint.endpointKind === 'receiver' ? 'Receiver' : 'Source'} ${endpoint.endpointKey}`;
  const inline = Number.isFinite(endpoint.inlineM) ? `${formatNumber(endpoint.inlineM, 2)} m` : '-';
  return [
    ['Endpoint', `${station} · key ${endpoint.endpointKey}`],
    ['Node', endpoint.nodeId],
    ['Inline', inline],
    ['Static status', endpoint.staticStatus],
    ['Solution status', endpoint.solutionStatus],
  ];
}

function createProfileEndpointActions(endpoint) {
  const panel = document.createElement('section');
  panel.className = 'refraction-qc-endpoint-actions';
  panel.dataset.testid = 'refraction-qc-profile-endpoint-actions';

  const title = document.createElement('h3');
  title.textContent = 'Selected endpoint';
  panel.appendChild(title);
  panel.appendChild(createKv(endpointSummaryItems(endpoint)));

  const actions = document.createElement('div');
  actions.className = 'refraction-qc-actions';
  const gatherKind = endpoint.endpointKind === 'receiver' ? 'receiver' : 'source';
  actions.append(
    createEndpointActionButton(
      `Preview ${gatherKind} gather`,
      '',
      () => previewGatherForEndpoint(gatherKind, endpoint.endpointKey),
      'refraction-qc-profile-preview-gather',
    ),
    createEndpointActionButton(
      'Open endpoint drilldown',
      '',
      () => openEndpointStaticDrilldown(gatherKind, endpoint.endpointKey),
      'refraction-qc-profile-open-drilldown',
    ),
    createEndpointActionButton(
      'Use as endpoint filter',
      '',
      () => {
        setEndpointFilter(gatherKind, endpoint.endpointKey);
        render();
      },
      'refraction-qc-profile-use-endpoint-filter',
    ),
  );
  panel.appendChild(actions);
  return panel;
}

function createStaticEndpointActions(endpointRecord, disabledReason) {
  const panel = document.createElement('section');
  panel.className = 'refraction-qc-endpoint-actions';
  panel.dataset.testid = 'refraction-qc-static-endpoint-actions';

  const endpointKind = endpointRecord ? staticEndpointKind(endpointRecord) : '';
  const endpointKey = endpointRecord ? staticEndpointKey(endpointRecord) : '';
  const gatherKind = endpointKind === 'receiver' ? 'receiver' : 'source';
  const actions = document.createElement('div');
  actions.className = 'refraction-qc-actions';
  actions.append(
    createEndpointActionButton(
      `Preview selected ${gatherKind} gather`,
      disabledReason,
      () => previewGatherForEndpoint(gatherKind, endpointKey),
      'refraction-qc-static-preview-gather',
    ),
    createEndpointActionButton(
      'Open endpoint drilldown',
      disabledReason,
      () => openEndpointStaticDrilldown(gatherKind, endpointKey),
      'refraction-qc-static-open-drilldown',
    ),
    createEndpointActionButton(
      'Copy endpoint key',
      disabledReason,
      async () => {
        if (!endpointKey || !navigator.clipboard?.writeText) return;
        await navigator.clipboard.writeText(endpointKey);
      },
      'refraction-qc-static-copy-endpoint',
    ),
  );
  panel.appendChild(actions);

  const value = document.createElement('p');
  value.className = 'refraction-qc-note';
  value.dataset.testid = 'refraction-qc-static-action-endpoint-key';
  value.textContent = endpointKey ? `endpoint_key: ${endpointKey}` : 'endpoint_key: not selected';
  panel.appendChild(value);

  if (disabledReason) {
    const reason = document.createElement('p');
    reason.className = 'refraction-qc-note';
    reason.dataset.testid = 'refraction-qc-static-action-reason';
    reason.textContent = disabledReason;
    panel.appendChild(reason);
  }
  return panel;
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
  const bundleContext = gatherContextFromQcBundle(state.qcBundle);
  const hasBundle = Boolean(state.qcBundle);

  return {
    fileId: hasBundle
      ? bundleContext.fileId
      : (
        state.gatherFileId
        || String(fileIdInput?.value || '')
        || String(window.currentFileId || '')
        || searchOrStorageValue('file_id', 'file_id')
        || ''
      ),
    key1Byte: hasBundle
      ? bundleContext.key1Byte
      : (
        state.gatherKey1Byte
        || String(window.currentKey1Byte || '')
        || searchOrStorageValue('key1_byte', 'key1_byte', '189')
      ),
    key2Byte: hasBundle
      ? bundleContext.key2Byte
      : (
        state.gatherKey2Byte
        || String(window.currentKey2Byte || '')
        || searchOrStorageValue('key2_byte', 'key2_byte', '193')
      ),
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
  const context = currentGatherContext();
  const jobId = String(state.selectedJobId || dom?.jobId?.value || '').trim();
  return buildGatherPreviewPayload(state, jobId, context);
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

function findGatherEndpointOption(options, value) {
  const query = normalizedText(value);
  return options.find((option) => (
    normalizedText(option.label) === query
    || normalizedText(option.value) === query
    || normalizedText(option.stationId) === query
  ));
}

function renderGatherEndpointSelectionState() {
  renderActiveFilterChips();
  const endpointKey = document.querySelector('[data-testid="refraction-qc-gather-endpoint-key"]');
  if (endpointKey) {
    endpointKey.textContent = state.gatherAxis === 'section'
      ? 'not used'
      : textOrDash(state.gatherEndpointKey);
    syncGatherDetailCopyButton(endpointKey);
  }
  if (state.selectedView === 'gather_preview') {
    const viewDef = VIEW_DEFS.find((view) => view.id === 'gather_preview');
    if (viewDef) renderViewContent(viewDef);
  }
}

function clearGatherPreviewPayload() {
  if (state.gatherPreview || state.gatherError || state.gatherLoading) {
    state.gatherPreview = null;
    state.gatherError = null;
    state.gatherLoading = false;
    controllerActions.invalidateGatherRequest();
  }
}

function updateGatherEndpointSelection(endpointKey, searchText) {
  const cleanKey = String(endpointKey || '').trim();
  const changed = state.gatherEndpointKey !== cleanKey;
  state.gatherEndpointKey = cleanKey;
  if (searchText !== undefined) state.gatherEndpointSearch = searchText;
  if (changed) clearGatherPreviewPayload();
  return changed;
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
  const selectedOption = options.find((option) => option.value === state.gatherEndpointKey);
  const inputValue = state.gatherEndpointSearch || selectedOption?.label || '';
  const listId = `refraction-qc-gather-${endpointKind}-stations`;
  const list = document.createElement('datalist');
  list.id = listId;
  for (const optionData of options) {
    const option = document.createElement('option');
    option.value = optionData.label;
    option.dataset.endpointKey = optionData.value;
    list.appendChild(option);
  }

  const input = makeGatherInput(
    'search',
    inputValue,
    'refraction-qc-gather-endpoint',
    (value) => {
      state.gatherEndpointSearch = value;
      const match = findGatherEndpointOption(options, value);
      const nextEndpointKey = match ? match.value : '';
      if (updateGatherEndpointSelection(nextEndpointKey)) {
        renderGatherEndpointSelectionState();
      }
    },
  );
  input.setAttribute('list', listId);
  input.disabled = !options.length;
  input.placeholder = options.length ? `Search ${label.toLowerCase()}...` : `No ${label.toLowerCase()} candidates`;
  input.addEventListener('change', () => {
    const match = findGatherEndpointOption(options, input.value);
    if (match) {
      updateGatherEndpointSelection(match.value, match.label);
    } else {
      updateGatherEndpointSelection('', input.value);
    }
    render();
  });

  const fragment = document.createDocumentFragment();
  fragment.append(makeGatherField(label, input), list);
  if (!options.length) {
    const empty = document.createElement('p');
    empty.className = 'refraction-qc-placeholder refraction-qc-gather-endpoint-empty';
    empty.dataset.testid = 'refraction-qc-gather-endpoint-empty';
    empty.textContent = `No ${label.toLowerCase()} candidates are present in static components.`;
    fragment.appendChild(empty);
  }
  return fragment;
}

function gatherDetailCopyText(valueElement) {
  const text = String(valueElement?.textContent || '').trim();
  return text && text !== '-' ? text : '';
}

function syncGatherDetailCopyButton(valueElement) {
  const row = valueElement?.closest?.('.refraction-qc-gather-detail-row');
  const copy = row?.querySelector?.('button');
  if (!copy) return;
  const canCopy = typeof navigator !== 'undefined' && navigator.clipboard?.writeText;
  copy.disabled = !canCopy || !gatherDetailCopyText(valueElement);
}

function createGatherDetailRow(labelText, valueText, testId) {
  const row = document.createElement('div');
  row.className = 'refraction-qc-gather-detail-row';
  const label = document.createElement('span');
  label.textContent = labelText;
  const value = document.createElement('code');
  if (testId) value.dataset.testid = testId;
  value.textContent = textOrDash(valueText);
  const copy = document.createElement('button');
  copy.type = 'button';
  copy.textContent = 'Copy';
  const canCopy = typeof navigator !== 'undefined' && navigator.clipboard?.writeText;
  copy.disabled = !canCopy || !gatherDetailCopyText(value);
  copy.addEventListener('click', async () => {
    const currentValue = gatherDetailCopyText(value);
    if (!currentValue || !canCopy) return;
    await navigator.clipboard.writeText(currentValue);
  });
  row.append(label, value, copy);
  return row;
}

function createGatherAdvancedDetails(context) {
  const details = document.createElement('details');
  details.className = 'refraction-qc-gather-details';
  details.dataset.testid = 'refraction-qc-gather-endpoint-details';
  const summary = document.createElement('summary');
  summary.textContent = 'Advanced';
  details.append(
    summary,
    createGatherDetailRow('File ID', context.fileId, 'refraction-qc-gather-file-id-value'),
    createGatherDetailRow('key1 byte', context.key1Byte, 'refraction-qc-gather-key1-byte-value'),
    createGatherDetailRow('key2 byte', context.key2Byte, 'refraction-qc-gather-key2-byte-value'),
    createGatherDetailRow('endpoint_key', state.gatherAxis === 'section' ? 'not used' : state.gatherEndpointKey, 'refraction-qc-gather-endpoint-key'),
    makeGatherField('Max traces', makeGatherInput(
      'number',
      String(state.gatherMaxTraces),
      'refraction-qc-gather-max-traces',
      (value) => { state.gatherMaxTraces = value.trim(); },
      { min: 1, step: 1 },
    )),
    makeGatherField('Reduction velocity (m/s)', makeGatherInput(
      'number',
      state.gatherReductionVelocity,
      'refraction-qc-gather-reduction-velocity',
      (value) => { state.gatherReductionVelocity = value.trim(); },
      { min: 1, step: '0.1' },
    )),
    createGatherDetailRow('Scaling', 'amax', 'refraction-qc-gather-scaling'),
  );
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
    controllerActions.loadGatherPreview();
  });

  const axis = makeGatherSelect(
    state.gatherAxis,
    'refraction-qc-gather-axis',
    Object.entries(GATHER_AXIS_LABELS),
    (value) => {
      state.gatherAxis = value;
      clearGatherPreviewFilter();
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
    makeGatherField('Gather type', axis),
    makeGatherField('Display', mode),
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
  );

  const actions = document.createElement('div');
  actions.className = 'refraction-qc-actions';
  const loadButton = document.createElement('button');
  loadButton.type = 'submit';
  loadButton.disabled = state.gatherLoading;
  loadButton.textContent = state.gatherLoading ? 'Loading preview...' : 'Preview gather';
  loadButton.dataset.testid = 'refraction-qc-gather-load';
  actions.appendChild(loadButton);
  form.appendChild(actions);
  form.appendChild(createGatherAdvancedDetails(context));
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

function gatherOverlayTimeValues(preview) {
  const fields = [
    'observed_pick_time_s',
    'modeled_pick_time_s',
    'corrected_observed_pick_time_s',
    'corrected_modeled_pick_time_s',
  ];
  const values = [];
  for (const field of fields) {
    const fieldValues = Array.isArray(preview[field]) ? preview[field] : [];
    for (const value of fieldValues) {
      const numeric = Number(value);
      if (Number.isFinite(numeric)) values.push(numeric);
    }
  }
  return values;
}

function numericRangeWithPadding(values, options = {}) {
  const finite = (Array.isArray(values) ? values : [])
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);
  if (!finite.length) return undefined;

  if (finite.length === 1) {
    const pad = Number.isFinite(Number(options.singlePad)) ? Number(options.singlePad) : 1;
    return [finite[0] - pad, finite[0] + pad];
  }

  const diffs = [];
  for (let index = 1; index < finite.length; index += 1) {
    const diff = finite[index] - finite[index - 1];
    if (diff > 0 && Number.isFinite(diff)) diffs.push(diff);
  }
  diffs.sort((a, b) => a - b);
  const step = diffs.length ? diffs[Math.floor(diffs.length / 2)] : 0;
  const spread = finite[finite.length - 1] - finite[0];
  if (spread === 0) {
    const pad = Number.isFinite(Number(options.singlePad)) ? Number(options.singlePad) : 1;
    return [finite[0] - pad, finite[0] + pad];
  }
  const pad = Number.isFinite(Number(options.pad))
    ? Number(options.pad)
    : (step > 0 ? step * 0.5 : Math.abs(spread) * 0.01);
  return [finite[0] - pad, finite[finite.length - 1] + pad];
}

function gatherPreviewAxisContext(preview) {
  const xAxis = gatherAxisValues(preview);
  const yValues = gatherTimeValues(preview);
  const yRangeValues = yValues.concat(gatherOverlayTimeValues(preview));
  return {
    xAxis,
    yValues,
    xRange: numericRangeWithPadding(xAxis.x),
    yRange: numericRangeWithPadding(yRangeValues),
  };
}

function gatherContextLines(preview = null) {
  const axis = preview?.gather?.axis || state.gatherAxis;
  const axisLabel = axis === 'receiver'
    ? 'Receiver gather'
    : (axis === 'section' ? 'Section window' : 'Source gather');
  const parts = [];
  if (axis === 'section') {
    const context = currentGatherContext();
    parts.push(`${axisLabel}: key1 ${textOrDash(preview?.gather?.key1 ?? context.key1)}`);
    parts.push(`traces ${textOrDash(preview?.window?.x0 ?? context.x0)}-${textOrDash(preview?.window?.x1 ?? context.x1)}`);
  } else {
    const endpointKind = axis === 'receiver' ? 'receiver' : 'source';
    const endpointKey = preview?.gather?.endpoint_key || state.gatherEndpointKey;
    parts.push(`${axisLabel}: ${gatherEndpointSummary(endpointKind, endpointKey) || 'station not selected'}`);
  }

  const windowMeta = preview?.window || {};
  const timeStart = Number.isFinite(Number(preview?.time_start_s))
    ? Number(preview.time_start_s)
    : Number(state.gatherTimeStartS);
  const timeEnd = Number.isFinite(Number(preview?.time_end_s))
    ? Number(preview.time_end_s)
    : Number(state.gatherTimeEndS);
  const timeLabel = Number.isFinite(timeStart) && Number.isFinite(timeEnd)
    ? `${formatNumber(timeStart, 3)}-${formatNumber(timeEnd, 3)} s`
    : `${textOrDash(state.gatherTimeStartS)}-${textOrDash(state.gatherTimeEndS)} s`;
  const traceCount = windowMeta.returned_trace_count ?? state.gatherMaxTraces;
  parts.push(`${GATHER_DISPLAY_LABELS[state.gatherDisplayMode] || state.gatherDisplayMode} · ${timeLabel} · ${textOrDash(traceCount)} traces`);
  return parts;
}

function createGatherContextBanner(preview = null) {
  const banner = document.createElement('div');
  banner.className = 'refraction-qc-gather-context';
  banner.dataset.testid = 'refraction-qc-gather-context';
  for (const lineText of gatherContextLines(preview)) {
    const line = document.createElement('div');
    line.textContent = lineText;
    banner.appendChild(line);
  }
  return banner;
}

function createGatherOverlayLegendItem(symbol, label) {
  const item = document.createElement('span');
  item.className = 'refraction-qc-gather-shared-legend-item';
  const marker = document.createElement('span');
  marker.className = `refraction-qc-gather-shared-legend-marker refraction-qc-gather-shared-legend-marker-${symbol}`;
  marker.setAttribute('aria-hidden', 'true');
  const text = document.createElement('span');
  text.textContent = label;
  item.appendChild(marker);
  item.appendChild(text);
  return item;
}

function createGatherOverlayLegend() {
  const legend = document.createElement('div');
  legend.className = 'refraction-qc-gather-shared-legend';
  legend.dataset.testid = 'refraction-qc-gather-shared-legend';
  legend.appendChild(createGatherOverlayLegendItem('observed', 'Observed first break'));
  legend.appendChild(createGatherOverlayLegendItem('modeled', 'Modeled first break'));
  return legend;
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

function finiteResidualMs(preview) {
  const residuals = Array.isArray(preview?.residual_s) ? preview.residual_s : [];
  return residuals
    .map((value) => Number(value) * 1000.0)
    .filter((value) => Number.isFinite(value));
}

function gatherHasResidualScale(preview) {
  return finiteResidualMs(preview).length > 0;
}

function residualColorScale(residualMs) {
  const finite = residualMs.filter((value) => Number.isFinite(value));
  if (!finite.length) return null;
  const maxAbs = finite.reduce((current, value) => Math.max(current, Math.abs(value)), 0);
  if (maxAbs === 0) {
    return {
      colorbar: {
        title: { text: 'Residual (ms)' },
        tickmode: 'array',
        tickvals: [0],
        ticktext: ['0'],
        x: 1.02,
        xanchor: 'left',
      },
    };
  }
  const tickvals = [-maxAbs, -maxAbs / 2, 0, maxAbs / 2, maxAbs];
  const digits = residualTickDigits(maxAbs);
  return {
    cmin: -maxAbs,
    cmax: maxAbs,
    colorbar: {
      title: { text: 'Residual (ms)' },
      tickmode: 'array',
      tickvals,
      ticktext: tickvals.map((value) => formatAsciiTick(value, digits)),
      x: 1.02,
      xanchor: 'left',
    },
  };
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
  const showResidualScale = Boolean(hasResidual && options.showResidualScale);
  const residualScale = showResidualScale ? residualColorScale(residualMs) : null;
  return {
    name: options.name,
    showlegend: options.showlegend !== false,
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
      cmin: residualScale?.cmin,
      cmax: residualScale?.cmax,
      showscale: showResidualScale,
      colorbar: residualScale?.colorbar,
      symbol: options.symbol,
      size: options.size || 8,
      line: { color: '#0f172a', width: 0.5 },
    },
  };
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








function viewState(keys) {
  const facade = {};
  for (const key of keys) {
    Object.defineProperty(facade, key, {
      enumerable: true,
      get: () => state[key],
      set: (value) => {
        state[key] = value;
      },
    });
  }
  return facade;
}

function viewStateFor(viewId) {
  if (viewId === 'artifacts') return viewState(['artifactSearch', 'artifactTypeFilter']);
  if (viewId === 'first_break_residuals') {
    return viewState(['firstBreakSortBy', 'selectedLayerKind', 'showRejectedFirstBreaks']);
  }
  if (viewId === 'reduced_time') return viewState(['selectedLayerKind', 'showRejectedFirstBreaks']);
  if (viewId === 'profiles_2d') {
    return viewState([
      'profileStatusFilter',
      'selectedEndpoint',
      'selectedEndpointKind',
      'selectedLayerKind',
      'selectedProfileEndpoint',
      'selectedProfileGroup',
      'selectedProfileUnits',
    ]);
  }
  if (viewId === 'cell_maps_3d') {
    return viewState([
      'qcDrilldown',
      'selectedCell',
      'selectedCellMapQuantity',
      'selectedLayerKind',
      'selectedObject',
    ]);
  }
  if (viewId === 'static_components') {
    return viewState(['selectedEndpoint', 'selectedEndpointKind']);
  }
  if (viewId === 'gather_preview') {
    return viewState([
      'gatherDisplayMode',
      'gatherEndpointKey',
      'gatherError',
      'gatherLoading',
      'gatherPreview',
    ]);
  }
  if (PICK_MAP_VIEWS[viewId]) {
    return viewState([
      'pickMap',
      'pickMapCacheStatus',
      'pickMapCachedFile',
      'pickMapDisplayMode',
      'pickMapError',
      'pickMapGatherEnd',
      'pickMapGatherStart',
      'pickMapLoading',
      'qcBundle',
      'selectedJobId',
    ]);
  }
  if (viewId === 'station_structure') {
    return viewState([
      'qcBundle',
      'selectedJobId',
      'stationStructure',
      'stationStructureDepthField',
      'stationStructureError',
      'stationStructureGatherEnd',
      'stationStructureGatherStart',
      'stationStructureLoading',
      'stationStructureVelocityField',
    ]);
  }
  return {};
}

function renderDepsForView(viewId) {
  if (viewId === 'summary') {
    return { createKv, summaryValue, warningsCount };
  }
  if (viewId === 'first_break_residuals') {
    return {
      FIRST_BREAK_SORT_CONTROL_OPTIONS,
      LAYER_COLORS,
      attachFirstBreakPickClickActions,
      createFirstBreakPickActions,
      createFirstBreakPlot,
      createKv,
      createTable,
      filteredFirstBreakPoints,
      findDownsampling,
      findViewData,
      firstBreakPickCustomData,
      firstBreakResidualThresholdMs,
      firstBreakXAxisDefinition,
      formatNumber,
      getPlotly,
      isUnavailable,
      layerLabel,
      optionLabel,
      plotHeight,
      plotHoverText,
      plotlyNewPlot,
      plotlyUnavailableMessage,
      selectedFirstBreakPick,
    };
  }
  if (viewId === 'reduced_time') {
    return {
      LAYER_COLORS,
      createFirstBreakPlot,
      createKv,
      createTable,
      filteredReducedTimePoints,
      findDownsampling,
      findViewData,
      firstBreakXAxisDefinition,
      getPlotly,
      isUnavailable,
      layerLabel,
      plotHeight,
      plotlyNewPlot,
      plotlyUnavailableMessage,
      reducedTimeGateOverlays,
      reducedTimeHoverText,
      reducedTimeVelocitySummary,
      statusCounts,
    };
  }
  if (viewId === 'profiles_2d') {
    return {
      PROFILE_SERIES_COLORS,
      attachProfileEndpointClickActions,
      createFirstBreakPlot,
      createKv,
      createProfileEndpointActions,
      createTable,
      filteredProfileRecords,
      findDownsampling,
      findViewData,
      getPlotly,
      isUnavailable,
      layerLabel,
      plotHeight,
      plotlyNewPlot,
      plotlyUnavailableMessage,
      profileAxisTitle,
      profileDisplayValue,
      profileEndpointCustomData,
      profileEndpointKey,
      profileHoverText,
      profileSeriesRawValue,
      profileTraceName,
      selectedProfileSeries,
      unavailableReason,
    };
  }
  if (viewId === 'cell_maps_3d') {
    return {
      buildCellMapMatrix,
      cellDrilldownTargetFromSelectedCell,
      cellEndpointDisplayName,
      cellMapHoverText,
      cellMapQuantityDefinition,
      cellMapQuantityValue,
      cellStatusCode,
      cellStatusColor,
      createEndpointActionButton,
      createFirstBreakPlot,
      createKv,
      createTable,
      domRef: () => dom,
      drilldownMetric,
      drilldownResidualMs,
      filteredCellMapRecords,
      findDownsampling,
      findViewData,
      firstDefined,
      formatNumber,
      getPlotly,
      isUnavailable,
      layerLabel,
      loadQcDrilldown,
      plotHeight,
      plotlyNewPlot,
      plotlyUnavailableMessage,
      previewGatherForEndpoint,
      render,
      selectedCellLabel,
      selectedCellMapLayer,
      selectedCellMapPoint,
      statusCounts,
      textOrDash,
      toFiniteNumber,
    };
  }
  if (viewId === 'static_components') {
    return {
      STATIC_COMPONENT_DEFS,
      STATIC_ENDPOINT_VIEW_KEY,
      STATIC_TRACE_COMPONENT_DEFS,
      STATIC_TRACE_VIEW_KEY,
      buildStaticComponentRows,
      componentColor,
      createFirstBreakPlot,
      createKv,
      createStaticComponentTable,
      createStaticEndpointActions,
      firstDefined,
      getPlotly,
      isUnavailable,
      matchingLegacyStaticRecord,
      normalizedText,
      plotHeight,
      plotlyNewPlot,
      plotlyUnavailableMessage,
      selectedStaticEndpointRecord,
      selectedStaticTraceRecord,
      staticApplyToTraceShift,
      staticEndpointKey,
      staticEndpointKind,
      staticEndpointRecords,
      staticTraceIndex,
      staticTraceRecords,
      textOrDash,
      viewByKey,
    };
  }
  if (viewId === 'gather_preview') {
    return {
      GATHER_AXIS_LABELS,
      GATHER_DISPLAY_LABELS,
      createFirstBreakPlot,
      createGatherContextBanner,
      createGatherOverlayLegend,
      createKv,
      formatNumber,
      gatherEndpointSummary,
      gatherHasResidualScale,
      gatherOverlayTrace,
      gatherPreviewAxisContext,
      getPlotly,
      plotHeight,
      plotlyNewPlot,
      plotlyResize,
      plotlyUnavailableMessage,
      textOrDash,
    };
  }
  if (viewId === 'artifacts') {
    return { appendText, createKv, createTable, filterArtifactRows };
  }
  if (PICK_MAP_VIEWS[viewId]) {
    return {
      clearNode,
      controllerActions,
      createKv,
      formatNumber,
      getCanvas2dContext,
      paddedRange,
      pickMapGatherNumber,
      pickMapTicks,
      plotHeight,
      render,
      toFiniteNumber,
    };
  }
  if (viewId === 'station_structure') {
    return {
      clearNode,
      controllerActions,
      createKv,
      formatNumber,
      getCanvas2dContext,
      paddedRange,
      pickMapTicks,
      plotHeight,
      toFiniteNumber,
    };
  }
  return {};
}

function activeFilterDeps() {
  return {
    clearGatherPreviewFilter,
    clearNode,
    clearSelectedCellFilter,
    clearSelectedObject,
    firstBreakResidualThresholdMs,
    formatNumber,
    layerLabel,
    optionLabel,
    render,
    selectedCellLabel,
  };
}

function controlsDeps() {
  return {
    clearSelectedCellFilter,
    clearNode,
    createCheckboxControl,
    createGatherPreviewControls,
    createNumberControl,
    createSelectControl,
    createTextControl,
    firstBreakResidualThresholdMs,
    formatNumber,
    layerLabel,
    optionLabel,
    parseCell,
    render,
  };
}

function inspectorDeps() {
  return {
    cellDrilldownTargetFromSelectedCell,
    cellEndpointDisplayName,
    clearNode,
    controllerActions,
    createKv,
    drilldownMetric,
    firstDefined,
    formatNumber,
    gatherEndpointStationId,
    layerLabel,
    loadQcDrilldown,
    normalizedText,
    openEndpointStaticDrilldown,
    parseCell,
    previewGatherForEndpoint,
    selectedCellLabel,
    selectedCellMapPoint,
    selectedFirstBreakPick,
    selectedProfileSeries,
    selectedStaticEndpointRecord,
    setEndpointFilter,
    STATIC_ENDPOINT_VIEW_KEY,
    staticEndpointKey,
    staticEndpointKind,
    staticEndpointRecords,
    textOrDash,
    toFiniteNumber,
    viewByKey,
  };
}

function renderViewContent(viewDef) {
  renderActiveView(state, dom, {
    appendText,
    clearNode,
    pickMapViews: PICK_MAP_VIEWS,
    renderDepsForView,
    renderTabularView: renderTabular,
    viewState: viewStateFor(viewDef.id),
    viewDef,
  });
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

function clearSelectedCellFilter() {
  state.selectedCell = null;
  if (state.selectedObject?.kind === 'cell') clearSelectedObject();
  state.qcDrilldown = null;
  state.qcDrilldownError = null;
  state.qcDrilldownLoading = false;
  state.qcDrilldownTarget = null;
  controllerActions.invalidateQcDrilldownRequest();
}

function clearGatherPreviewFilter() {
  state.gatherEndpointKey = '';
  state.gatherEndpointSearch = '';
  state.gatherPreview = null;
  state.gatherError = null;
  state.gatherLoading = false;
  controllerActions.invalidateGatherRequest();
}

function resetJobScopedFilters() {
  state.selectedEndpoint = '';
  state.selectedTraceIndex = '';
  state.selectedProfileEndpoint = null;
  clearSelectedObject();
  clearGatherPreviewFilter();
  clearSelectedCellFilter();
}

function renderActiveFilterChips() {
  renderActiveFilterChipsPanel({
    state,
    dom,
    context: activeFilterDeps(),
  });
}

function renderViewControls() {
  renderControlsPanel({
    state,
    dom,
    context: controlsDeps(),
  });
}

function renderControlUpdate() {
  if (!dom) return;
  renderActiveFilterChips();
  const selectedViewDef = VIEW_DEFS.find((viewDef) => viewDef.id === state.selectedView);
  if (selectedViewDef) renderViewContent(selectedViewDef);
  renderInspectorPanel({
    state,
    dom,
    context: inspectorDeps(),
  });
}

function render() {
  if (!dom) return;
  if (!PICK_MAP_VIEWS[state.selectedView]) cleanupPickMapCanvasRenderer();
  if (state.selectedView !== 'station_structure') cleanupStationStructureCanvasRenderer();

  renderJobSummary();

  dom.loadButton.disabled = state.loading;
  dom.maxPoints.value = String(state.maxPoints);
  renderActiveFilterChips();
  renderViewControls();

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

  for (const button of dom.taskButtons) {
    const active = button.dataset.task === state.activeTask;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  }

  const taskViewIds = new Set(viewIdsForTask(state.activeTask));
  const hasTaskViewChoices = taskViewIds.size > 1;
  if (dom.viewButtonsContainer) {
    dom.viewButtonsContainer.hidden = !hasTaskViewChoices;
  }
  for (const button of dom.viewButtons) {
    button.hidden = !hasTaskViewChoices || !taskViewIds.has(button.dataset.view);
    const active = button.dataset.view === state.selectedView;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  }
  for (const panel of dom.viewPanels) {
    panel.hidden = panel.dataset.viewPanel !== state.selectedView;
  }
  const selectedViewDef = VIEW_DEFS.find((viewDef) => viewDef.id === state.selectedView);
  if (selectedViewDef) renderViewContent(selectedViewDef);
  renderInspectorPanel({
    state,
    dom,
    context: inspectorDeps(),
  });
}
