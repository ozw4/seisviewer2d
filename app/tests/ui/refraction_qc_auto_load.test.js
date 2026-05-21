import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { afterEach, beforeEach, expect, test, vi } from 'vitest';

const SCRIPT = readFileSync(
  resolve(process.cwd(), 'static/refraction_qc.js'),
  'utf8'
);

let canvasOps = [];

function renderRefractionQcPanel() {
  document.body.innerHTML = `
    <button id="pipelineSidebarTab" class="sidebar-tab is-active" aria-selected="true"></button>
    <button id="staticCorrectionSidebarTab" class="sidebar-tab" aria-selected="false"></button>
    <button id="refractionQcSidebarTab" class="sidebar-tab" aria-selected="false"></button>
    <section id="pipelineTabPanel"></section>
    <section id="staticCorrectionTabPanel" hidden></section>
    <section id="refractionQcTabPanel" hidden>
      <form id="refractionQcForm">
        <input id="refractionQcJobId" value="" />
        <datalist id="refractionQcJobList"></datalist>
        <input id="refractionQcMaxPoints" value="20000" />
        <button id="refractionQcLoadButton" type="submit"></button>
      </form>
      <div id="refractionQcStatus"></div>
      <div id="refractionQcError" hidden></div>
      <div id="refractionQcSign" hidden></div>
      <div id="refractionQcJobSummary"></div>
      <div id="refractionQcActiveFilters" data-testid="refraction-qc-active-filters"></div>
      <div id="refractionQcViewControls" data-testid="refraction-qc-view-controls"></div>
      <button type="button" class="refraction-qc-task-button" data-task="overview"></button>
      <button type="button" class="refraction-qc-task-button" data-task="find_problems"></button>
      <button type="button" class="refraction-qc-task-button" data-task="inspect_station"></button>
      <button type="button" class="refraction-qc-task-button" data-task="inspect_cell"></button>
      <button type="button" class="refraction-qc-task-button" data-task="preview_gather"></button>
      <button type="button" class="refraction-qc-task-button" data-task="artifacts"></button>
      <div id="refractionQcViewButtons">
      <button type="button" class="refraction-qc-view-button" data-view="summary"></button>
      <button type="button" class="refraction-qc-view-button" data-view="first_break_residuals"></button>
      <button type="button" class="refraction-qc-view-button" data-view="reduced_time"></button>
      <button type="button" class="refraction-qc-view-button" data-view="profiles_2d"></button>
      <button type="button" class="refraction-qc-view-button" data-view="cell_maps_3d"></button>
      <button type="button" class="refraction-qc-view-button" data-view="static_components"></button>
      <button type="button" class="refraction-qc-view-button" data-view="pick_map"></button>
      <button type="button" class="refraction-qc-view-button" data-view="offset_time"></button>
      <button type="button" class="refraction-qc-view-button" data-view="station_structure"></button>
      <button type="button" class="refraction-qc-view-button" data-view="gather_preview"></button>
      <button type="button" class="refraction-qc-view-button" data-view="artifacts"></button>
      </div>
      <section class="refraction-qc-view" data-view-panel="summary">
        <div data-view-content="summary"></div>
      </section>
      <section class="refraction-qc-view" data-view-panel="first_break_residuals" hidden>
        <div data-view-content="first_break_residuals"></div>
      </section>
      <section class="refraction-qc-view" data-view-panel="reduced_time" hidden>
        <div data-view-content="reduced_time"></div>
      </section>
      <section class="refraction-qc-view" data-view-panel="profiles_2d" hidden>
        <div data-view-content="profiles_2d"></div>
      </section>
      <section class="refraction-qc-view" data-view-panel="cell_maps_3d" hidden>
        <div data-view-content="cell_maps_3d"></div>
      </section>
      <section class="refraction-qc-view" data-view-panel="static_components" hidden>
        <div data-view-content="static_components"></div>
      </section>
      <section class="refraction-qc-view" data-view-panel="pick_map" hidden>
        <div data-view-content="pick_map"></div>
      </section>
      <section class="refraction-qc-view" data-testid="refraction-qc-view-offset-time" data-view-panel="offset_time" hidden>
        <div data-view-content="offset_time"></div>
      </section>
      <section class="refraction-qc-view" data-testid="refraction-qc-view-station-structure" data-view-panel="station_structure" hidden>
        <div data-view-content="station_structure"></div>
      </section>
      <section class="refraction-qc-view" data-view-panel="gather_preview" hidden>
        <div data-view-content="gather_preview"></div>
      </section>
      <section class="refraction-qc-view" data-view-panel="artifacts" hidden>
        <div data-view-content="artifacts"></div>
      </section>
      <aside id="refractionQcInspector"></aside>
    </section>
  `;
}

function installCanvasContextStub() {
  canvasOps = [];
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation((type) => {
    if (type !== '2d') return null;
    return {
      setTransform: (...args) => canvasOps.push(['setTransform', ...args]),
      clearRect: (...args) => canvasOps.push(['clearRect', ...args]),
      fillRect: (...args) => canvasOps.push(['fillRect', ...args]),
      beginPath: (...args) => canvasOps.push(['beginPath', ...args]),
      moveTo: (...args) => canvasOps.push(['moveTo', ...args]),
      lineTo: (...args) => canvasOps.push(['lineTo', ...args]),
      stroke: (...args) => canvasOps.push(['stroke', ...args]),
      strokeRect: (...args) => canvasOps.push(['strokeRect', ...args]),
      fillText: (...args) => canvasOps.push(['fillText', ...args]),
      save: (...args) => canvasOps.push(['save', ...args]),
      restore: (...args) => canvasOps.push(['restore', ...args]),
      translate: (...args) => canvasOps.push(['translate', ...args]),
      rotate: (...args) => canvasOps.push(['rotate', ...args]),
      arc: (...args) => canvasOps.push(['arc', ...args]),
      fill: (...args) => canvasOps.push(['fill', ...args]),
    };
  });
  return canvasOps;
}

function installNoopCanvasContextStub() {
  const context = {
    setTransform: () => {},
    clearRect: () => {},
    fillRect: () => {},
    beginPath: () => {},
    moveTo: () => {},
    lineTo: () => {},
    stroke: () => {},
    strokeRect: () => {},
    fillText: () => {},
    save: () => {},
    restore: () => {},
    translate: () => {},
    rotate: () => {},
    arc: () => {},
    fill: () => {},
  };
  HTMLCanvasElement.prototype.getContext.mockImplementation((type) => (type === '2d' ? context : null));
}

function pointArcs() {
  return canvasOps.filter((op) => op[0] === 'arc');
}

function jsonResponse(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}

function loadRefractionQcScript() {
  window.eval(SCRIPT);
  document.dispatchEvent(new Event('DOMContentLoaded'));
  return window.RefractionQc;
}

async function flushAsyncWork(times = 2) {
  for (let index = 0; index < times; index += 1) {
    await Promise.resolve();
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

function completedPickMap(jobId, statusMessage = `Completed Pick Map for ${jobId}`) {
  return {
    mode: 'completed_job',
    job_id: jobId,
    has_after_statics: true,
    receiver_number_mode: 'global_sequential',
    gather_range: { min: 100, max: 100 },
    status_message: statusMessage,
    pick_map: {
      gather_id: [100],
      receiver_number: [2000],
      pick_before_ms: [84],
      pick_after_ms: [74],
      used_in_statics: [true],
      offset_m: [120],
      offset_used: [120],
      applied_shift_ms: [-10],
    },
  };
}

function preStaticsPickMap() {
  return {
    mode: 'pre_statics',
    job_id: null,
    has_after_statics: false,
    receiver_number_mode: 'global_sequential',
    gather_range: { min: 100, max: 100 },
    status_message: 'Pre-statics Pick Map',
    pick_map: {
      gather_id: [100],
      receiver_number: [2000],
      pick_before_ms: [84],
      pick_after_ms: [null],
      used_in_statics: [null],
      offset_m: [120],
    },
  };
}

function multiPointCompletedPickMap() {
  return {
    mode: 'completed_job',
    job_id: 'completed-job-map',
    has_after_statics: true,
    receiver_number_mode: 'global_sequential',
    gather_range: { min: 100, max: 103 },
    status_message: 'Completed Pick Map',
    pick_map: {
      gather_id: [100, 101, 102, 103],
      receiver_number: [2000, 2001, 2002, 2003],
      pick_before_ms: [80, 100, 120, 140],
      pick_after_ms: [70, 85, 115, 138],
      used_in_statics: [true, false, true, false],
      offset_m: [100, 150, 200, 250],
      offset_used: [90, null, 210, null],
      applied_shift_ms: [-10, -10, -10, -10],
    },
  };
}

function qcBundle(jobId) {
  return {
    job_id: jobId,
    summary: { status: 'ready', workflow: 'refraction' },
    available_views: ['summary'],
    unavailable_views: [],
    coordinate_mode: 'auto',
  };
}

function firstBreakQcBundle(jobId = 'job-first-break') {
  return {
    ...qcBundle(jobId),
    available_views: ['first_break_fit'],
    views: {
      first_break_fit: {
        artifact: 'first-break-fit.json',
        columns: [
          'trace_index_sorted',
          'offset_m',
          'observed_first_break_time_s',
          'modeled_first_break_time_s',
          'source_endpoint_key',
          'receiver_endpoint_key',
        ],
        records: [{
          trace_index_sorted: 12345,
          offset_m: 850,
          observed_first_break_time_s: 0.4125,
          modeled_first_break_time_s: 0.395,
          layer_kind: 'v2_t1',
          source_endpoint_key: '1001',
          receiver_endpoint_key: '2034',
        }],
      },
    },
  };
}

function profileQcBundle(jobId = 'job-profile') {
  return {
    ...qcBundle(jobId),
    available_views: ['line_profiles'],
    views: {
      line_profiles: {
        artifact: 'line-profiles.json',
        columns: ['endpoint_kind', 'endpoint_key', 'inline_m', 't1_ms'],
        records: [{
          endpoint_kind: 'source',
          endpoint_key: '1001',
          station_id: '1001',
          node_id: 45,
          inline_m: 1500,
          t1_ms: 8.3,
          pick_count: 96,
          residual_rms_ms: 12.4,
          total_static_ms: 8.3,
          static_status: 'ok',
          solution_status: 'ok',
        }],
      },
    },
  };
}

function staticEndpointQcBundle(jobId = 'job-static-endpoint') {
  return {
    ...qcBundle(jobId),
    available_views: ['static_component_qc_endpoint', 'static_component_qc_trace'],
    views: {
      static_component_qc_endpoint: {
        artifact: 'static-endpoint.csv',
        columns: ['endpoint_kind', 'endpoint_key', 'node_id', 'pick_count', 'residual_rms_ms'],
        records: [{
          endpoint_kind: 'source',
          endpoint_key: '1001',
          station_id: '1001',
          node_id: 45,
          pick_count: 96,
          residual_rms_ms: 12.4,
          total_static_ms: 8.3,
          static_status: 'ok',
          solution_status: 'ok',
        }],
      },
      static_component_qc_trace: {
        artifact: 'static-trace.csv',
        columns: ['trace_index_sorted', 'source_endpoint_key', 'receiver_endpoint_key'],
        records: [{
          trace_index_sorted: 12345,
          source_endpoint_key: '1001',
          receiver_endpoint_key: '2034',
          static_status: 'ok',
        }],
      },
    },
  };
}

function gatherPreviewQcBundle(jobId = 'job-gather') {
  const base = qcBundle(jobId);
  return {
    ...base,
    summary: {
      ...base.summary,
      request: {
        file_id: 'line-a.sgy',
        key1_byte: 189,
        key2_byte: 193,
      },
    },
    available_views: ['static_components', 'static_component_qc_endpoint'],
    views: {
      static_components: {
        artifact: 'static-components.csv',
        columns: ['endpoint_kind', 'endpoint_key', 'station_id', 'static_status'],
        records: [{
          endpoint_kind: 'source',
          endpoint_key: '1001',
          station_id: '1001',
          static_status: 'ok',
        }],
      },
      static_component_qc_endpoint: {
        artifact: 'static-endpoint.csv',
        columns: ['endpoint_kind', 'endpoint_key', 'pick_count', 'residual_rms_ms'],
        records: [{
          endpoint_kind: 'source',
          endpoint_key: '1001',
          pick_count: 96,
          residual_rms_ms: 12.4,
          static_status: 'ok',
        }],
      },
    },
  };
}

function cellQcBundle(jobId = 'job-cell') {
  return {
    ...qcBundle(jobId),
    available_views: ['refraction_grid_map_qc'],
    views: {
      refraction_grid_map_qc: {
        artifact: 'grid-map.csv',
        columns: ['cell_ix', 'cell_iy', 'cell_velocity_layer_kind', 'velocity_m_s'],
        records: [{
          cell_ix: 4,
          cell_iy: 2,
          cell_velocity_layer_kind: 'v2_t1',
          velocity_m_s: 1820,
          fold: 24,
          residual_rms_ms: 16.1,
          velocity_status: 'ok',
          cell_center_x_m: 40,
          cell_center_y_m: 20,
        }],
      },
    },
  };
}

function installPlotlyClickStub() {
  const plots = [];
  const handlers = new Map();
  const plotly = {
    newPlot: vi.fn((plot, traces, layout, config) => {
      plots.push({ plot, traces, layout, config });
      plot.on = (event, callback) => {
        handlers.set(`${plot.dataset.testid}:${event}`, callback);
      };
      return Promise.resolve();
    }),
  };
  window.Plotly = plotly;
  return { handlers, plots, plotly };
}

function stationStructurePayload(jobId = 'completed-job-structure') {
  const source = {
    x: [100, 101],
    y: [8, 9],
    endpoint_key: ['source:100', 'source:101'],
    status: ['ok', 'ok'],
  };
  const receiver = {
    x: [200, 201],
    y: [18, 19],
    endpoint_key: ['receiver:200', 'receiver:201'],
    status: ['ok', 'low_fold'],
  };
  return {
    job_id: jobId,
    statics_kind: 'refraction',
    view_kind: 'station_structure',
    x_axis: 'global_station_number',
    x_axis_label: 'Global station number',
    x_axis_status: 'ok',
    station_mapping: {
      source_method: 'coordinate_interpolation',
      receiver_method: 'coordinate_order',
      coordinate_field: 'inline_m',
      warnings: [],
    },
    filter_status: 'ok',
    gather_range: { start: 100, end: 101 },
    colors: { source: 'cyan', receiver: 'red' },
    time_term: {
      field: 't1',
      label: 'Time-term distribution',
      unit: 'ms',
      source,
      receiver,
    },
    velocity: {
      field: 'v2',
      label: 'Velocity structure: V2',
      unit: 'm/s',
      source: { ...source, y: [2400, 2450] },
      receiver: { ...receiver, y: [2600, 2650] },
    },
    depth: {
      field: 'sh1',
      label: 'Weathering thickness SH1',
      unit: 'm',
      source: { ...source, y: [12, 13] },
      receiver: { ...receiver, y: [22, 23] },
    },
    warnings: [],
  };
}

function activeViewerTarget({
  fileId = 'pick-map-file',
  key1Byte = 189,
  key2Byte = 193,
} = {}) {
  return {
    getActiveFileTarget: () => ({
      file_id: fileId,
      key1_byte: key1Byte,
      key2_byte: key2Byte,
      isFileLoaded: true,
    }),
  };
}

function seedStaticCorrectionPickDraft({
  fileId = 'pick-map-file',
  key1Byte = 189,
  key2Byte = 193,
  filename = 'first-breaks.npz',
  recordId = 'pick-map-file:189:193',
} = {}) {
  localStorage.setItem('sv.static_correction.form_draft.v1', JSON.stringify({
    version: 1,
    target: { file_id: fileId, key1_byte: key1Byte, key2_byte: key2Byte },
    pickNpz: {
      indexedDbRecordId: recordId,
      filename,
      type: 'application/octet-stream',
      lastModified: 1700000000000,
      fileId,
      key1Byte,
      key2Byte,
    },
  }));
}

function stubStaticCorrectionPickDb({
  filename = 'first-breaks.npz',
  blob = new Blob(['npz-bytes'], { type: 'application/octet-stream' }),
} = {}) {
  const open = vi.fn(() => {
    const openRequest = {};
    const db = {
      transaction: () => ({
        objectStore: () => ({
          get: () => {
            const getRequest = {};
            setTimeout(() => {
              getRequest.result = {
                filename,
                type: 'application/octet-stream',
                lastModified: 1700000000000,
                blob,
              };
              getRequest.onsuccess?.();
            }, 0);
            return getRequest;
          },
        }),
      }),
      close: vi.fn(),
    };
    setTimeout(() => {
      openRequest.result = db;
      openRequest.onsuccess?.();
    }, 0);
    return openRequest;
  });
  vi.stubGlobal('indexedDB', { open });
  return open;
}

async function openPickMapWithCachedNpz() {
  window.SeisViewerState = activeViewerTarget();
  seedStaticCorrectionPickDraft();
  stubStaticCorrectionPickDb();
  loadRefractionQcScript();
  window.refractionQcUI.setSelectedView('pick_map');
  await flushAsyncWork(4);
}

beforeEach(() => {
  localStorage.clear();
  window.history.replaceState(null, '', '/');
  delete window.RefractionQc;
  delete window.refractionQcUI;
  delete window.refractionQcState;
  delete window.SeisViewerState;
  renderRefractionQcPanel();
  installCanvasContextStub();
});

afterEach(() => {
  localStorage.clear();
  delete window.RefractionQc;
  delete window.refractionQcUI;
  delete window.refractionQcState;
  delete window.SeisViewerState;
  try {
    delete window.navigator.clipboard;
  } catch (_) {}
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

test('loadJob populates the QC job input, loads the bundle, and activates Refraction QC', async () => {
  const calls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    calls.push({ url: String(url), body: JSON.parse(options.body || '{}') });
    return jsonResponse({
      job_id: 'static-job-a',
      summary: { status: 'ready', workflow: 'refraction' },
      available_views: ['summary'],
      unavailable_views: [],
      coordinate_mode: 'auto',
    });
  }));
  const refractionQc = loadRefractionQcScript();

  await refractionQc.loadJob('static-job-a', { maxPoints: 123 });

  expect(document.getElementById('refractionQcJobId').value).toBe('static-job-a');
  expect(document.getElementById('refractionQcSidebarTab').getAttribute('aria-selected')).toBe('true');
  expect(document.getElementById('refractionQcTabPanel').hidden).toBe(false);
  expect(calls).toHaveLength(1);
  expect(calls[0]).toMatchObject({
    url: '/statics/refraction/qc',
    body: { job_id: 'static-job-a', max_points: 123 },
  });
  expect(new URL(window.location.href).searchParams.get('refraction_job_id')).toBe('static-job-a');
  expect(
    Array.from(document.getElementById('refractionQcJobList').children).map((node) => node.value)
  ).toContain('static-job-a');
});

test('manual Refraction QC load still uses the shared bundle loader', async () => {
  const calls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    calls.push({ url: String(url), body: JSON.parse(options.body || '{}') });
    return jsonResponse({
      job_id: 'manual-job-a',
      summary: { status: 'ready', workflow: 'refraction' },
      available_views: ['summary'],
      unavailable_views: [],
      coordinate_mode: 'auto',
    });
  }));
  loadRefractionQcScript();

  document.getElementById('refractionQcJobId').value = 'manual-job-a';
  document.getElementById('refractionQcForm').dispatchEvent(
    new Event('submit', { bubbles: true, cancelable: true })
  );
  await flushAsyncWork();

  expect(calls).toHaveLength(1);
  expect(calls[0]).toMatchObject({
    url: '/statics/refraction/qc',
    body: { job_id: 'manual-job-a', max_points: 20000 },
  });
  expect(document.getElementById('refractionQcStatus').textContent).toContain(
    'Loaded manual-job-a'
  );
});

test('Overview does not show global selector flood', () => {
  loadRefractionQcScript();

  const controls = document.querySelector('[data-testid="refraction-qc-view-controls"]');
  expect(controls.hidden).toBe(true);
  expect(document.querySelector('[data-testid="refraction-qc-profile-group"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-profile-units"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-map-quantity"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-endpoint"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-trace"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-cell"]')).toBeNull();
});

test('Artifacts task shows the empty bundle state before a QC bundle is loaded', () => {
  loadRefractionQcScript();

  expect(() => window.refractionQcUI.setSelectedView('artifacts')).not.toThrow();
  expect(document.querySelector('[data-view-content="artifacts"]').textContent)
    .toContain('No QC bundle loaded.');
});

test('Run summary warning count ignores unavailable view metadata', () => {
  loadRefractionQcScript();
  window.refractionQcState.qcBundle = {
    ...qcBundle('job-unavailable-views'),
    unavailable_views: ['profiles', 'cells'],
  };

  window.refractionQcUI.setSelectedView('summary');

  const warningsMetric = Array.from(
    document.querySelectorAll('#refractionQcJobSummary .refraction-qc-metric')
  ).find((card) => (
    card.querySelector('.refraction-qc-metric-label')?.textContent === 'Warnings'
  ));
  expect(warningsMetric.querySelector('.refraction-qc-metric-value').textContent).toBe('0');
});

test('Refraction QC renders controls only for the selected view', () => {
  loadRefractionQcScript();

  window.refractionQcUI.setSelectedView('cell_maps_3d');
  expect(document.querySelector('[data-testid="refraction-qc-map-quantity"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-status-filter"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-cell"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-profile-group"]')).toBeNull();

  window.refractionQcUI.setSelectedView('profiles_2d');
  expect(document.querySelector('[data-testid="refraction-qc-profile-group"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-profile-units"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-map-quantity"]')).toBeNull();

  window.refractionQcUI.setSelectedView('first_break_residuals');
  expect(document.querySelector('[data-testid="refraction-qc-x-axis"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-show-rejected"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-residual-threshold"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-residual-sort"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-endpoint"]')).toBeNull();

  window.refractionQcUI.setSelectedView('reduced_time');
  expect(document.querySelector('[data-testid="refraction-qc-x-axis"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-show-rejected"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-residual-threshold"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-residual-sort"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-map-quantity"]')).toBeNull();

  window.refractionQcUI.setSelectedView('gather_preview');
  expect(document.querySelector('[data-testid="refraction-qc-gather-axis"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-gather-display"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-gather-time-start"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-gather-max-traces"]')).not.toBeNull();
  expect(document.querySelector('[data-view-content="gather_preview"] [data-testid="refraction-qc-gather-controls"]')).toBeNull();

  window.refractionQcUI.setSelectedView('artifacts');
  expect(document.querySelector('[data-testid="refraction-qc-artifact-type"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-artifact-search"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-gather-axis"]')).toBeNull();
});

test('Gather Preview keeps internal context in the advanced drawer', async () => {
  const writeText = vi.fn(async () => {});
  Object.defineProperty(window.navigator, 'clipboard', {
    value: { writeText },
    configurable: true,
  });
  loadRefractionQcScript();
  window.currentFileId = 'stale-line.sgy';
  window.currentKey1Byte = 17;
  window.currentKey2Byte = 21;
  window.refractionQcState.qcBundle = gatherPreviewQcBundle();

  window.refractionQcUI.setSelectedView('gather_preview');

  expect(document.querySelector('[data-testid="refraction-qc-gather-file-id"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-gather-key1-byte"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-gather-key2-byte"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-gather-endpoint-search"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-gather-endpoint"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-gather-load"]').textContent)
    .toBe('Preview gather');

  const details = document.querySelector('[data-testid="refraction-qc-gather-endpoint-details"]');
  expect(details.open).toBe(false);
  expect(details.querySelector('[data-testid="refraction-qc-gather-file-id-value"]').textContent)
    .toBe('line-a.sgy');
  expect(details.querySelector('[data-testid="refraction-qc-gather-key1-byte-value"]').textContent)
    .toBe('189');
  expect(details.querySelector('[data-testid="refraction-qc-gather-key2-byte-value"]').textContent)
    .toBe('193');
  const endpointValue = details.querySelector('[data-testid="refraction-qc-gather-endpoint-key"]');
  const endpointCopy = endpointValue.closest('.refraction-qc-gather-detail-row').querySelector('button');
  expect(endpointValue.textContent)
    .toBe('-');
  expect(endpointCopy.disabled).toBe(true);

  const station = document.querySelector('[data-testid="refraction-qc-gather-endpoint"]');
  station.value = 'S 1001 · picks 96 · RMS 12.4 ms · ok';
  station.dispatchEvent(new Event('input', { bubbles: true }));
  expect(window.refractionQcState.gatherEndpointKey).toBe('1001');
  expect(endpointValue.textContent)
    .toBe('1001');
  expect(endpointCopy.disabled).toBe(false);
  endpointCopy.click();
  await flushAsyncWork();
  expect(writeText).toHaveBeenCalledWith('1001');
  expect(document.querySelector('[data-testid="refraction-qc-filter-chip"][data-filter="gather-endpoint"]').textContent)
    .toContain('Gather 1001');

  station.value = 'S 9999';
  station.dispatchEvent(new Event('input', { bubbles: true }));
  expect(window.refractionQcState.gatherEndpointKey).toBe('');
  expect(endpointValue.textContent)
    .toBe('-');
  expect(endpointCopy.disabled).toBe(true);
  expect(document.querySelector('[data-testid="refraction-qc-filter-chip"][data-filter="gather-endpoint"]')).toBeNull();

  const axis = document.querySelector('[data-testid="refraction-qc-gather-axis"]');
  axis.value = 'section';
  axis.dispatchEvent(new Event('change', { bubbles: true }));

  expect(document.querySelector('[data-testid="refraction-qc-gather-key1"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-gather-x0"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-gather-x1"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-gather-endpoint"]')).toBeNull();
});

test('Gather Preview clears stale preview output when the selected station changes', () => {
  installPlotlyClickStub();
  loadRefractionQcScript();
  window.refractionQcState.qcBundle = gatherPreviewQcBundle();
  window.refractionQcState.gatherEndpointKey = '1001';
  window.refractionQcState.gatherPreview = {
    job_id: 'job-gather',
    gather: { axis: 'source', endpoint_key: '1001' },
    window: {
      requested_trace_count: 2,
      returned_trace_count: 2,
      requested_sample_count: 2,
      returned_sample_count: 2,
    },
    dt_s: 0.001,
    shape: [2, 2],
    x_indices: [0, 1],
    offset_m: [100, 200],
    raw_samples: [[1, 2], [3, 4]],
    corrected_samples: [[1, 1], [2, 2]],
    corrected_samples_source: 'corrected-tracestore',
    corrected_window_ref: { status: 'ok' },
    sign_convention: 'positive static shifts delay traces',
    overlay_status: { first_break_fit: 'ok' },
  };

  window.refractionQcUI.setSelectedView('gather_preview');
  expect(document.querySelector('[data-testid="refraction-qc-gather-raw-plot"]')).not.toBeNull();

  const station = document.querySelector('[data-testid="refraction-qc-gather-endpoint"]');
  station.value = 'S 9999';
  station.dispatchEvent(new Event('input', { bubbles: true }));

  expect(window.refractionQcState.gatherEndpointKey).toBe('');
  expect(window.refractionQcState.gatherPreview).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-gather-raw-plot"]')).toBeNull();
  expect(document.querySelector('[data-view-content="gather_preview"]').textContent)
    .toContain('Choose a station and preview the gather.');
});

test('Gather Preview selects a source station from one searchable control and previews it', async () => {
  installPlotlyClickStub();
  const calls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    calls.push({ url: String(url), body: JSON.parse(options.body || '{}') });
    return jsonResponse({
      job_id: 'job-gather',
      gather: { axis: 'source', endpoint_key: '1001' },
      window: {
        requested_trace_count: 2,
        returned_trace_count: 2,
        requested_sample_count: 2,
        returned_sample_count: 2,
      },
      dt_s: 0.001,
      shape: [2, 2],
      x_indices: [0, 1],
      offset_m: [100, 200],
      raw_samples: [[1, 2], [3, 4]],
      corrected_samples: [[1, 1], [2, 2]],
      corrected_samples_source: 'corrected-tracestore',
      corrected_window_ref: { status: 'ok' },
      sign_convention: 'positive static shifts delay traces',
      overlay_status: { first_break_fit: 'ok' },
    });
  }));
  loadRefractionQcScript();
  window.currentFileId = 'line-a.sgy';
  window.currentKey1Byte = 189;
  window.currentKey2Byte = 193;
  document.getElementById('refractionQcJobId').value = 'job-gather';
  window.refractionQcState.qcBundle = gatherPreviewQcBundle();

  window.refractionQcUI.setSelectedView('gather_preview');
  const station = document.querySelector('[data-testid="refraction-qc-gather-endpoint"]');
  station.value = 'S 1001 · picks 96 · RMS 12.4 ms · ok';
  station.dispatchEvent(new Event('input', { bubbles: true }));

  expect(window.refractionQcState.gatherEndpointKey).toBe('1001');

  document.querySelector('[data-testid="refraction-qc-gather-load"]').click();
  await flushAsyncWork();

  expect(calls).toHaveLength(1);
  expect(calls[0]).toMatchObject({
    url: '/statics/refraction/qc/gather-preview',
    body: {
      job_id: 'job-gather',
      file_id: 'line-a.sgy',
      key1_byte: 189,
      key2_byte: 193,
      gather_axis: 'source',
      endpoint_key: '1001',
    },
  });
  expect(document.querySelector('[data-testid="refraction-qc-gather-context"]').textContent)
    .toContain('Source gather: S 1001 · picks 96 · RMS 12.4 ms · ok');
  expect(document.querySelector('[data-testid="refraction-qc-gather-raw-plot"]')).not.toBeNull();
});

test('Gather Preview side-by-side plots share explicit offset and time ranges', () => {
  const { plots } = installPlotlyClickStub();
  loadRefractionQcScript();
  window.refractionQcState.qcBundle = gatherPreviewQcBundle();
  window.refractionQcState.gatherDisplayMode = 'side_by_side';
  window.refractionQcState.gatherEndpointKey = '1001';
  window.refractionQcState.gatherPreview = {
    job_id: 'job-gather',
    gather: { axis: 'source', endpoint_key: '1001' },
    window: {
      requested_trace_count: 10,
      returned_trace_count: 10,
      requested_sample_count: 3,
      returned_sample_count: 3,
    },
    dt_s: 0.004,
    shape: [3, 10],
    x_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    offset_m: [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2400],
    raw_samples: [
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
      [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ],
    corrected_samples: [
      [1, 1, 2, 3, 5, 8, 13, 21, 34, 55],
      [2, 2, 3, 4, 6, 9, 14, 22, 35, 56],
      [3, 3, 4, 5, 7, 10, 15, 23, 36, 57],
    ],
    observed_pick_time_s: [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
    modeled_pick_time_s: [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
    corrected_observed_pick_time_s: [0.0015, 0.0025, 0.0035, 0.0045, 0.0055, 0.0065, 0.0075, 0.0085, 0.0095, 0.0105],
    corrected_modeled_pick_time_s: [0.0015, 0.0025, 0.0035, 0.0045, 0.0055, 0.0065, 0.0075, 0.0085, 0.0095, 0.0105],
    residual_s: [0, 0.001, -0.001, 0, 0.001, -0.001, 0, 0.001, -0.001, 0],
    corrected_samples_source: 'raw_tracestore_shifted_on_the_fly',
    corrected_window_ref: { status: 'ok' },
    sign_convention: 'positive static shifts delay traces',
    overlay_status: { first_break_fit: 'ok' },
  };

  window.refractionQcUI.setSelectedView('gather_preview');

  const rawPlot = plots.find((entry) => entry.plot.dataset.testid === 'refraction-qc-gather-raw-plot');
  const correctedPlot = plots.find((entry) => entry.plot.dataset.testid === 'refraction-qc-gather-corrected-plot');

  expect(rawPlot.layout.xaxis.range).toEqual(correctedPlot.layout.xaxis.range);
  expect(rawPlot.layout.yaxis.range).toEqual(correctedPlot.layout.yaxis.range);
  expect(rawPlot.layout.xaxis.title.text).toBe('Offset (m)');
  expect(correctedPlot.layout.xaxis.title.text).toBe('Offset (m)');
  expect(rawPlot.layout.xaxis.autorange).toBe(false);
  expect(correctedPlot.layout.yaxis.autorange).toBe(false);
  expect(rawPlot.layout.xaxis.range[0]).toBeLessThan(250);
  expect(rawPlot.layout.xaxis.range[1]).toBeGreaterThan(2400);
  expect(rawPlot.layout.yaxis.range[0]).toBeGreaterThan(0.0105);
});

test('Gather Preview explains missing station selection without endpoint wording', () => {
  loadRefractionQcScript();
  window.currentFileId = 'line-a.sgy';
  window.currentKey1Byte = 189;
  window.currentKey2Byte = 193;
  document.getElementById('refractionQcJobId').value = 'job-gather';
  window.refractionQcState.qcBundle = gatherPreviewQcBundle();

  window.refractionQcUI.setSelectedView('gather_preview');
  document.querySelector('[data-testid="refraction-qc-gather-load"]').click();

  const error = document.querySelector('[data-testid="refraction-qc-gather-error"]');
  expect(error.textContent).toContain('Source station を選択してください');
  expect(error.textContent).not.toContain('endpoint');
});

test('typing in view-specific text controls keeps the focused input mounted', () => {
  loadRefractionQcScript();
  window.refractionQcUI.setSelectedView('profiles_2d');

  const station = document.querySelector('[data-testid="refraction-qc-endpoint"]');
  station.focus();
  station.value = 'S1001';
  station.dispatchEvent(new Event('input', { bubbles: true }));

  expect(window.refractionQcState.selectedEndpoint).toBe('S1001');
  expect(station.isConnected).toBe(true);
  expect(document.activeElement).toBe(station);
  expect(document.querySelector('[data-testid="refraction-qc-endpoint"]')).toBe(station);
  expect(document.querySelector('[data-testid="refraction-qc-filter-chip"][data-filter="endpoint"]').textContent)
    .toContain('S1001');
});

test('active filter chips clear filters and update view controls', () => {
  loadRefractionQcScript();
  window.refractionQcUI.setSelectedView('cell_maps_3d');

  const layer = document.querySelector('[data-testid="refraction-qc-layer-kind"]');
  layer.value = 'v2_t1';
  layer.dispatchEvent(new Event('change', { bubbles: true }));

  let chip = document.querySelector('[data-testid="refraction-qc-filter-chip"][data-filter="layer"]');
  expect(chip.textContent).toContain('Layer V2/T1');
  chip.click();

  expect(window.refractionQcState.selectedLayerKind).toBe('all');
  expect(document.querySelector('[data-testid="refraction-qc-layer-kind"]').value).toBe('all');
  expect(document.querySelector('[data-testid="refraction-qc-filter-chip"][data-filter="layer"]')).toBeNull();
});

test('first-break residual threshold filters points and can be cleared from a chip', () => {
  loadRefractionQcScript();
  window.refractionQcState.qcBundle = {
    job_id: 'job-a',
    views: {
      first_break_fit: {
        artifact: 'first-break-fit.json',
        columns: [
          'trace_index_sorted',
          'offset_m',
          'observed_first_break_time_s',
          'modeled_first_break_time_s',
        ],
        records: [
          {
            trace_index_sorted: 1,
            offset_m: 100,
            observed_first_break_time_s: 0.105,
            modeled_first_break_time_s: 0.1,
            layer_kind: 'v2_t1',
          },
          {
            trace_index_sorted: 2,
            offset_m: 150,
            observed_first_break_time_s: 0.13,
            modeled_first_break_time_s: 0.1,
            layer_kind: 'v2_t1',
          },
        ],
      },
    },
  };
  window.refractionQcUI.setSelectedView('first_break_residuals');

  expect(document.querySelector('[data-testid="refraction-qc-first-break-residual-plot"]').dataset.pointCount)
    .toBe('2');

  const threshold = document.querySelector('[data-testid="refraction-qc-residual-threshold"]');
  threshold.value = '10';
  threshold.dispatchEvent(new Event('input', { bubbles: true }));

  expect(window.refractionQcState.firstBreakResidualThresholdMs).toBe('10');
  expect(document.querySelector('[data-testid="refraction-qc-first-break-residual-plot"]').dataset.pointCount)
    .toBe('1');
  const chip = document.querySelector('[data-testid="refraction-qc-filter-chip"][data-filter="residual-threshold"]');
  expect(chip.textContent).toContain('Residual >= 10');

  chip.click();

  expect(window.refractionQcState.firstBreakResidualThresholdMs).toBe('');
  expect(document.querySelector('[data-testid="refraction-qc-first-break-residual-plot"]').dataset.pointCount)
    .toBe('2');
});

test('loading a different job resets endpoint trace and cell filters', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(qcBundle('job-b'))));
  loadRefractionQcScript();
  window.refractionQcState.qcBundle = qcBundle('job-a');
  window.refractionQcState.selectedEndpoint = 'S1001';
  window.refractionQcState.selectedTraceIndex = '42';
  window.refractionQcState.selectedCell = { cell_ix: 2, cell_iy: 3, layer_kind: 'v2_t1' };
  window.refractionQcState.selectedObject = {
    kind: 'cell',
    key: '2,3 V2/T1',
    payload: { cell_ix: 2, cell_iy: 3, layer_kind: 'v2_t1' },
  };

  document.getElementById('refractionQcJobId').value = 'job-b';
  await window.refractionQcUI.loadBundle();

  expect(window.refractionQcState.selectedEndpoint).toBe('');
  expect(window.refractionQcState.selectedTraceIndex).toBe('');
  expect(window.refractionQcState.selectedCell).toBeNull();
  expect(window.refractionQcState.selectedObject.kind).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-filter-chip"][data-filter="endpoint"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-filter-chip"][data-filter="trace"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-filter-chip"][data-filter="cell"]')).toBeNull();
});

test('Inspector starts empty and updates after first-break pick click', async () => {
  const { handlers, plots } = installPlotlyClickStub();
  loadRefractionQcScript();
  expect(document.getElementById('refractionQcInspector').textContent).toContain('No selection');

  window.refractionQcState.qcBundle = firstBreakQcBundle();
  window.refractionQcUI.setSelectedView('first_break_residuals');
  await flushAsyncWork();

  const residualPlot = plots.find((entry) => entry.plot.dataset.testid === 'refraction-qc-first-break-residual-plot');
  handlers.get('refraction-qc-first-break-residual-plot:plotly_click')({
    points: [{ customdata: residualPlot.traces[0].customdata[0] }],
  });

  const inspector = document.getElementById('refractionQcInspector');
  expect(inspector.textContent).toContain('Selected pick');
  expect(inspector.textContent).toContain('V2/T1');
  expect(inspector.textContent).toContain('850.0 m');
  expect(inspector.textContent).toContain('+17.5 ms');

  inspector.querySelector('button').click();
  expect(window.refractionQcState.selectedView).toBe('gather_preview');
  expect(window.refractionQcState.gatherAxis).toBe('source');
  expect(window.refractionQcState.gatherEndpointKey).toBe('1001');
});

test('Inspector updates after profile endpoint click', async () => {
  const { handlers, plots } = installPlotlyClickStub();
  loadRefractionQcScript();
  window.refractionQcState.qcBundle = profileQcBundle();
  window.refractionQcUI.setSelectedView('profiles_2d');
  await flushAsyncWork();

  const profilePlot = plots.find((entry) => entry.plot.dataset.testid === 'refraction-qc-profile-plot');
  handlers.get('refraction-qc-profile-plot:plotly_click')({
    points: [{ customdata: profilePlot.traces[0].customdata[0] }],
  });

  const inspector = document.getElementById('refractionQcInspector');
  expect(inspector.textContent).toContain('Selected source station');
  expect(inspector.textContent).toContain('S 1001');
  expect(inspector.textContent).toContain('96');
  expect(inspector.textContent).toContain('12.4 ms');

  Array.from(inspector.querySelectorAll('button'))
    .find((button) => button.textContent === 'Preview gather')
    .click();
  expect(window.refractionQcState.selectedView).toBe('gather_preview');
  expect(window.refractionQcState.gatherAxis).toBe('source');
  expect(window.refractionQcState.gatherEndpointKey).toBe('1001');
});

test('Inspector renders endpoint filter selections from static components', () => {
  loadRefractionQcScript();
  window.refractionQcState.qcBundle = staticEndpointQcBundle();
  window.refractionQcState.selectedEndpointKind = 'source';
  window.refractionQcState.selectedEndpoint = '1001';

  window.refractionQcUI.setSelectedView('static_components');

  const inspector = document.getElementById('refractionQcInspector');
  expect(inspector.textContent).toContain('Selected source station');
  expect(inspector.textContent).toContain('S 1001');
  expect(inspector.textContent).toContain('96');
  expect(inspector.textContent).toContain('12.4 ms');

  Array.from(inspector.querySelectorAll('button'))
    .find((button) => button.textContent === 'Preview gather')
    .click();
  expect(window.refractionQcState.selectedView).toBe('gather_preview');
  expect(window.refractionQcState.gatherAxis).toBe('source');
  expect(window.refractionQcState.gatherEndpointKey).toBe('1001');
});

test('Inspector updates after cell click and opens cell drilldown', async () => {
  const { handlers, plots } = installPlotlyClickStub();
  const calls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    calls.push({ url: String(url), body: JSON.parse(options.body || '{}') });
    return jsonResponse({
      target: { kind: 'cell', layer_kind: 'v2_t1', cell_ix: 4, cell_iy: 2 },
      cell: {
        cell_ix: 4,
        cell_iy: 2,
        layer_kind: 'v2_t1',
        velocity_m_s: 1820,
        fold: 24,
        residual_summary: { cell_residual_rms_ms: 16.1 },
      },
      observations: { returned_count: 0, total_count: 0, records: [] },
    });
  }));
  loadRefractionQcScript();
  window.refractionQcState.selectedJobId = 'job-cell';
  window.refractionQcState.qcBundle = cellQcBundle();
  window.refractionQcUI.setSelectedView('cell_maps_3d');
  await flushAsyncWork();

  const cellPlot = plots.find((entry) => entry.plot.dataset.testid === 'refraction-qc-cell-map-plot');
  handlers.get('refraction-qc-cell-map-plot:plotly_click')({
    points: [{ customdata: cellPlot.traces[0].customdata[0][0] }],
  });
  await flushAsyncWork();

  const inspector = document.getElementById('refractionQcInspector');
  expect(inspector.textContent).toContain('Selected cell');
  expect(inspector.textContent).toContain('4,2 V2/T1');
  expect(inspector.textContent).toContain('1820.00 m/s');
  expect(inspector.textContent).toContain('16.1 ms');

  Array.from(inspector.querySelectorAll('button'))
    .find((button) => button.textContent === 'Open cell drilldown')
    .click();
  expect(window.refractionQcState.selectedView).toBe('cell_maps_3d');
  expect(calls.at(-1).body.target).toMatchObject({
    kind: 'cell',
    layer_kind: 'v2_t1',
    cell_ix: 4,
    cell_iy: 2,
  });
});

test('loadJob displays QC loading errors with the requested job id still visible', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => jsonResponse({ detail: 'QC bundle missing' }, 404)));
  const refractionQc = loadRefractionQcScript();

  await refractionQc.loadJob('missing-static-job');

  expect(document.getElementById('refractionQcJobId').value).toBe('missing-static-job');
  expect(document.getElementById('refractionQcError').hidden).toBe(false);
  expect(document.getElementById('refractionQcError').textContent).toContain('QC bundle missing');
});

test('Pick Map uses canvas renderer without Plotly', () => {
  const plotly = {
    newPlot: vi.fn(() => {
      throw new Error('Pick Map must not call Plotly.newPlot');
    }),
    react: vi.fn(() => {
      throw new Error('Pick Map must not call Plotly.react');
    }),
  };
  window.Plotly = plotly;
  loadRefractionQcScript();

  window.refractionQcState.pickMap = preStaticsPickMap();
  window.refractionQcUI.setSelectedView('pick_map');

  expect(document.querySelector('[data-testid="refraction-qc-pick-map-canvas"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-plot"]').dataset.renderer).toBe('canvas');
  expect(plotly.newPlot).not.toHaveBeenCalled();
  expect(plotly.react).not.toHaveBeenCalled();
});

test('Pick Map canvas handles dense point ranges without spreading values into Math min/max', () => {
  installNoopCanvasContextStub();
  const count = 150000;
  const pickMap = preStaticsPickMap();
  pickMap.pick_map.gather_id = Array.from({ length: count }, (_, index) => index);
  pickMap.pick_map.receiver_number = Array.from({ length: count }, (_, index) => 1000 + index);
  pickMap.pick_map.pick_before_ms = Array.from({ length: count }, (_, index) => 50 + (index % 500));
  pickMap.pick_map.pick_after_ms = Array.from({ length: count }, () => null);
  pickMap.pick_map.used_in_statics = Array.from({ length: count }, () => null);
  pickMap.pick_map.offset_m = Array.from({ length: count }, (_, index) => index % 2000);
  loadRefractionQcScript();

  window.refractionQcState.pickMap = pickMap;
  expect(() => window.refractionQcUI.setSelectedView('pick_map')).not.toThrow();

  expect(document.querySelector('[data-testid="refraction-qc-pick-map-canvas"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-plot"]').dataset.pointCount).toBe(String(count));
});

test('Pick Map canvas draws pick time increasing downward', () => {
  const pickMap = preStaticsPickMap();
  pickMap.pick_map.receiver_number = [2000, 2001];
  pickMap.pick_map.pick_before_ms = [80, 120];
  pickMap.pick_map.gather_id = [100, 101];
  pickMap.pick_map.offset_m = [100, 200];
  loadRefractionQcScript();

  window.refractionQcState.pickMap = pickMap;
  window.refractionQcUI.setSelectedView('pick_map');

  const arcs = pointArcs().slice(-2);
  expect(arcs).toHaveLength(2);
  expect(arcs[1][2]).toBeGreaterThan(arcs[0][2]);
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-plot"]').dataset.yAxisDirection).toBe('down');
});

test('Pick Map canvas renders before and after modes', () => {
  loadRefractionQcScript();

  window.refractionQcState.pickMap = multiPointCompletedPickMap();
  window.refractionQcUI.setSelectedView('pick_map');
  const beforeArcs = pointArcs().map((op) => op[2]);

  canvasOps = [];
  document.querySelector('[data-testid="refraction-qc-pick-map-after"]').click();

  expect(window.refractionQcState.pickMapDisplayMode).toBe('after');
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-after"]').className).toBe('is-active');
  expect(pointArcs().map((op) => op[2])).not.toEqual(beforeArcs);
});

test('completed Pick Map canvas distinguishes used, unused, and offset-colored points', () => {
  loadRefractionQcScript();

  window.refractionQcState.pickMap = multiPointCompletedPickMap();
  window.refractionQcUI.setSelectedView('pick_map');

  const plot = document.querySelector('[data-testid="refraction-qc-pick-map-plot"]');
  expect(plot.dataset.usedPointCount).toBe('2');
  expect(plot.dataset.unusedPointCount).toBe('2');
  expect(plot.dataset.offsetColorCount).toBe('2');
});

test('pre-statics Pick Map canvas disables After Statics', () => {
  loadRefractionQcScript();

  window.refractionQcState.pickMap = preStaticsPickMap();
  window.refractionQcState.pickMapDisplayMode = 'after';
  window.refractionQcUI.setSelectedView('pick_map');

  expect(document.querySelector('[data-testid="refraction-qc-pick-map-canvas"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-after"]').disabled).toBe(true);
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-before"]').className).toBe('is-active');
});

test('Pick Map canvas applies gather range filter', () => {
  loadRefractionQcScript();

  window.refractionQcState.pickMap = multiPointCompletedPickMap();
  window.refractionQcState.pickMapGatherStart = '101';
  window.refractionQcState.pickMapGatherEnd = '102';
  window.refractionQcUI.setSelectedView('pick_map');

  const plot = document.querySelector('[data-testid="refraction-qc-pick-map-plot"]');
  expect(plot.dataset.pointCount).toBe('2');
  expect(pointArcs().slice(-2)).toHaveLength(2);
});

test('Offset-time uses canvas renderer without Plotly', () => {
  const plotly = {
    newPlot: vi.fn(() => {
      throw new Error('Offset-time must not call Plotly.newPlot');
    }),
    react: vi.fn(() => {
      throw new Error('Offset-time must not call Plotly.react');
    }),
  };
  window.Plotly = plotly;
  loadRefractionQcScript();

  window.refractionQcState.pickMap = preStaticsPickMap();
  window.refractionQcUI.setSelectedView('offset_time');

  const plot = document.querySelector('[data-testid="refraction-qc-offset-time-plot"]');
  expect(document.querySelector('[data-testid="refraction-qc-offset-time-canvas"]')).not.toBeNull();
  expect(plot.dataset.renderer).toBe('canvas');
  expect(plot.dataset.xAxisTitle).toBe('Offset (m)');
  expect(plot.dataset.pointCount).toBe('1');
  expect(plotly.newPlot).not.toHaveBeenCalled();
  expect(plotly.react).not.toHaveBeenCalled();
});

test('Offset-time reuses Pick Map payload and shared gather range controls', () => {
  const fetch = vi.fn();
  vi.stubGlobal('fetch', fetch);
  loadRefractionQcScript();

  window.refractionQcState.pickMap = multiPointCompletedPickMap();
  window.refractionQcUI.setSelectedView('pick_map');
  const gatherStart = document.querySelector('[data-testid="refraction-qc-pick-map-gather-start"]');
  const gatherEnd = document.querySelector('[data-testid="refraction-qc-pick-map-gather-end"]');
  gatherStart.value = '101';
  gatherStart.dispatchEvent(new Event('input', { bubbles: true }));
  gatherEnd.value = '102';
  gatherEnd.dispatchEvent(new Event('input', { bubbles: true }));

  window.refractionQcUI.setSelectedView('offset_time');

  expect(fetch).not.toHaveBeenCalled();
  expect(document.querySelector('[data-testid="refraction-qc-offset-time-gather-start"]').value).toBe('101');
  expect(document.querySelector('[data-testid="refraction-qc-offset-time-gather-end"]').value).toBe('102');
  expect(document.querySelector('[data-testid="refraction-qc-offset-time-plot"]').dataset.pointCount).toBe('2');
});

test('Offset-time skips records with missing offsets and reports displayed count', () => {
  loadRefractionQcScript();
  const pickMap = multiPointCompletedPickMap();
  pickMap.pick_map.offset_m = [100, NaN, 'bad', 250];

  window.refractionQcState.pickMap = pickMap;
  window.refractionQcUI.setSelectedView('offset_time');

  const plot = document.querySelector('[data-testid="refraction-qc-offset-time-plot"]');
  expect(plot.dataset.pointCount).toBe('2');
  expect(document.querySelector('[data-testid="refraction-qc-offset-time-canvas"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-view-offset-time"]').textContent).toContain(
    'Displayed points'
  );
  expect(document.querySelector('[data-testid="refraction-qc-view-offset-time"]').textContent).toContain('2');
});

test('Offset-time shows a clear message when selected records have no finite offsets', () => {
  loadRefractionQcScript();
  const pickMap = multiPointCompletedPickMap();
  pickMap.pick_map.offset_m = [100, NaN, 'bad', 250];

  window.refractionQcState.pickMap = pickMap;
  window.refractionQcState.pickMapGatherStart = '101';
  window.refractionQcState.pickMapGatherEnd = '102';
  window.refractionQcUI.setSelectedView('offset_time');

  const plot = document.querySelector('[data-testid="refraction-qc-offset-time-plot"]');
  expect(plot.dataset.pointCount).toBe('0');
  expect(plot.textContent).toBe(
    'No Offset-time records are available because offset_m is missing or non-finite for the selected gather range.'
  );
});

test('Structure QC uses canvas renderer without Plotly', () => {
  const plotly = {
    newPlot: vi.fn(() => {
      throw new Error('Structure QC must not call Plotly.newPlot');
    }),
    react: vi.fn(() => {
      throw new Error('Structure QC must not call Plotly.react');
    }),
  };
  window.Plotly = plotly;
  loadRefractionQcScript();

  window.refractionQcState.qcBundle = qcBundle('completed-job-structure');
  window.refractionQcState.stationStructure = stationStructurePayload();
  window.refractionQcUI.setSelectedView('station_structure');

  const timePlot = document.querySelector('[data-testid="refraction-qc-station-structure-time-term-plot"]');
  expect(document.querySelector('[data-testid="refraction-qc-station-structure-time-term-canvas"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-station-structure-velocity-canvas"]')).not.toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-station-structure-depth-canvas"]')).not.toBeNull();
  expect(timePlot.parentElement.classList.contains('refraction-qc-station-structure-grid')).toBe(true);
  expect(timePlot.dataset.renderer).toBe('canvas');
  expect(timePlot.dataset.xAxisTitle).toBe('Global station number');
  expect(timePlot.dataset.pointCount).toBe('4');
  expect(plotly.newPlot).not.toHaveBeenCalled();
  expect(plotly.react).not.toHaveBeenCalled();
});

test('Structure QC displays fallback axis warning and backend x arrays', () => {
  loadRefractionQcScript();

  const payload = stationStructurePayload();
  payload.x_axis = 'endpoint_id_fallback';
  payload.x_axis_label = 'source/receiver endpoint id fallback';
  payload.x_axis_status = 'fallback';
  payload.station_mapping = {
    source_method: 'source_id_fallback',
    receiver_method: 'receiver_id',
    coordinate_field: null,
    warnings: ['Source x-axis fell back to source_id because receiver station reference could not be inferred.'],
  };
  payload.warnings = payload.station_mapping.warnings;
  payload.time_term.source.x = [1, 5, 9, 73];
  payload.velocity.source.x = [1, 5, 9, 73];
  payload.depth.source.x = [1, 5, 9, 73];
  window.refractionQcState.qcBundle = qcBundle('completed-job-structure');
  window.refractionQcState.stationStructure = payload;
  window.refractionQcUI.setSelectedView('station_structure');

  const timePlot = document.querySelector('[data-testid="refraction-qc-station-structure-time-term-plot"]');
  expect(timePlot.dataset.xAxisTitle).toBe('source/receiver endpoint id fallback');
  expect(document.body.textContent).toContain('fell back to source_id');
  expect(payload.time_term.source.x[3]).toBe(73);
});

test('Structure QC sends gather range and selectors to station endpoint', async () => {
  const fetch = vi.fn(async () => jsonResponse(stationStructurePayload('completed-job-structure')));
  vi.stubGlobal('fetch', fetch);
  loadRefractionQcScript();

  window.refractionQcState.qcBundle = qcBundle('completed-job-structure');
  window.refractionQcUI.setSelectedView('station_structure');
  document.querySelector('[data-testid="refraction-qc-station-structure-gather-start"]').value = '101';
  document.querySelector('[data-testid="refraction-qc-station-structure-gather-start"]')
    .dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('[data-testid="refraction-qc-station-structure-gather-end"]').value = '120';
  document.querySelector('[data-testid="refraction-qc-station-structure-gather-end"]')
    .dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('[data-testid="refraction-qc-station-structure-velocity-field"]').value = 'v2';
  document.querySelector('[data-testid="refraction-qc-station-structure-velocity-field"]')
    .dispatchEvent(new Event('change', { bubbles: true }));
  document.querySelector('[data-testid="refraction-qc-station-structure-depth-field"]').value = 'refractor_elevation';
  document.querySelector('[data-testid="refraction-qc-station-structure-depth-field"]')
    .dispatchEvent(new Event('change', { bubbles: true }));

  await window.refractionQcUI.loadStationStructureQc();

  expect(fetch).toHaveBeenCalledWith('/statics/refraction/qc/station-structure', expect.objectContaining({
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  }));
  const body = JSON.parse(fetch.mock.calls.at(-1)[1].body);
  expect(body).toMatchObject({
    job_id: 'completed-job-structure',
    gather_start: 101,
    gather_end: 120,
    x_axis: 'auto',
    velocity_field: 'v2',
    depth_field: 'refractor_elevation',
  });
  expect(document.querySelector('[data-testid="refraction-qc-station-structure-time-term-canvas"]')).not.toBeNull();
});

test('Pick Map does not render manual NPZ file input or upload button', () => {
  loadRefractionQcScript();
  window.refractionQcUI.setSelectedView('pick_map');

  expect(document.querySelector('[data-testid="refraction-qc-pick-map-npz"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-load-upload"]')).toBeNull();
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-load-cached"]')).not.toBeNull();
});

test('Pick Map loads pre-statics map from Static Correction cached NPZ', async () => {
  const pickMapCalls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    if (String(url) === '/statics/refraction/qc/pick-map') {
      pickMapCalls.push({
        request: JSON.parse(options.body.get('request_json')),
        pickName: options.body.get('pick_npz').name,
      });
      return jsonResponse(preStaticsPickMap());
    }
    throw new Error(`Unexpected fetch ${url}`);
  }));

  await openPickMapWithCachedNpz();
  document.querySelector('[data-testid="refraction-qc-pick-map-load-cached"]').click();
  await flushAsyncWork();

  expect(pickMapCalls).toEqual([{
    request: {
      file_id: 'pick-map-file',
      key1_byte: 189,
      key2_byte: 193,
      pick_source: { kind: 'uploaded_npz' },
      geometry: { receiver_number_mode: 'global_sequential' },
    },
    pickName: 'first-breaks.npz',
  }]);
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-status"]').textContent).toBe(
    'Pre-statics Pick Map'
  );
});

test('Pick Map shows Static Correction guidance without cached NPZ', async () => {
  window.SeisViewerState = activeViewerTarget();
  loadRefractionQcScript();
  window.refractionQcUI.setSelectedView('pick_map');
  await flushAsyncWork();

  expect(document.querySelector('[data-testid="refraction-qc-pick-map-cache-status"]').textContent).toContain(
    'No Static Correction NPZ is cached for the active viewer target.'
  );
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-cache-status"] a').getAttribute('href')).toBe(
    '/static-correction?file_id=pick-map-file&key1_byte=189&key2_byte=193'
  );
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-load-cached"]').disabled).toBe(true);
});

test('Pick Map rejects cached NPZ for a different viewer target', async () => {
  window.SeisViewerState = activeViewerTarget();
  seedStaticCorrectionPickDraft({ fileId: 'other-file', key1Byte: 189, key2Byte: 193 });
  loadRefractionQcScript();
  window.refractionQcUI.setSelectedView('pick_map');
  await flushAsyncWork();

  expect(document.querySelector('[data-testid="refraction-qc-pick-map-cache-status"]').textContent).toContain(
    'Saved Static Correction NPZ belongs to a different viewer target.'
  );
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-load-cached"]').disabled).toBe(true);
});

test('Pick Map refreshes cached NPZ controls when cache becomes invalid with a map open', async () => {
  window.SeisViewerState = activeViewerTarget();
  seedStaticCorrectionPickDraft();
  stubStaticCorrectionPickDb();
  loadRefractionQcScript();
  window.refractionQcState.pickMap = preStaticsPickMap();
  window.refractionQcUI.setSelectedView('pick_map');
  await flushAsyncWork(4);

  expect(document.querySelector('[data-testid="refraction-qc-pick-map-cache-status"]').textContent).toContain(
    'Cached NPZ available: first-breaks.npz'
  );
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-load-cached"]').disabled).toBe(false);

  seedStaticCorrectionPickDraft({ fileId: 'other-file', key1Byte: 189, key2Byte: 193 });
  window.refractionQcUI.setSelectedView('pick_map');
  await flushAsyncWork();

  expect(document.querySelector('[data-testid="refraction-qc-pick-map-cache-status"]').textContent).toContain(
    'Saved Static Correction NPZ belongs to a different viewer target.'
  );
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-load-cached"]').disabled).toBe(true);
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-status"]').textContent).toBe(
    'Pre-statics Pick Map'
  );
});

test('Pick Map completed job button still loads completed-job map', async () => {
  const pickMapCalls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    if (String(url) === '/statics/refraction/qc/pick-map') {
      pickMapCalls.push(JSON.parse(options.body || '{}'));
      return jsonResponse(completedPickMap('completed-job-map'));
    }
    throw new Error(`Unexpected fetch ${url}`);
  }));
  loadRefractionQcScript();
  document.getElementById('refractionQcJobId').value = 'completed-job-map';
  window.refractionQcState.selectedJobId = 'completed-job-map';
  window.refractionQcUI.setSelectedView('pick_map');

  document.querySelector('[data-testid="refraction-qc-pick-map-load-job"]').click();
  await flushAsyncWork();

  expect(pickMapCalls).toEqual([{ job_id: 'completed-job-map' }]);
  expect(window.refractionQcState.pickMap).toMatchObject({
    mode: 'completed_job',
    job_id: 'completed-job-map',
  });
});

test('completed QC bundle clears active pre-statics Pick Map and loads completed job map', async () => {
  const pickMapCalls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    if (String(url) === '/statics/refraction/qc') {
      return jsonResponse(qcBundle('completed-job-a'));
    }
    if (String(url) === '/statics/refraction/qc/pick-map') {
      pickMapCalls.push(JSON.parse(options.body || '{}'));
      return jsonResponse(completedPickMap('completed-job-a'));
    }
    throw new Error(`Unexpected fetch ${url}`);
  }));
  loadRefractionQcScript();
  window.refractionQcState.pickMap = preStaticsPickMap();
  window.refractionQcUI.setSelectedView('pick_map');

  document.getElementById('refractionQcJobId').value = 'completed-job-a';
  await window.refractionQcUI.loadBundle();
  await flushAsyncWork();

  expect(pickMapCalls).toEqual([{ job_id: 'completed-job-a' }]);
  expect(window.refractionQcState.pickMap).toMatchObject({
    mode: 'completed_job',
    job_id: 'completed-job-a',
    has_after_statics: true,
  });
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-after"]').disabled).toBe(false);
});

test('completed QC bundle defers completed Pick Map load until Pick Map view opens', async () => {
  const pickMapCalls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    if (String(url) === '/statics/refraction/qc') {
      return jsonResponse(qcBundle('completed-job-b'));
    }
    if (String(url) === '/statics/refraction/qc/pick-map') {
      pickMapCalls.push(JSON.parse(options.body || '{}'));
      return jsonResponse(completedPickMap('completed-job-b'));
    }
    throw new Error(`Unexpected fetch ${url}`);
  }));
  loadRefractionQcScript();
  window.refractionQcState.pickMap = preStaticsPickMap();

  document.getElementById('refractionQcJobId').value = 'completed-job-b';
  await window.refractionQcUI.loadBundle();
  await flushAsyncWork();

  expect(window.refractionQcState.pickMap).toBeNull();
  expect(pickMapCalls).toHaveLength(0);

  window.refractionQcUI.setSelectedView('pick_map');
  await flushAsyncWork();

  expect(pickMapCalls).toEqual([{ job_id: 'completed-job-b' }]);
  expect(window.refractionQcState.pickMap).toMatchObject({
    mode: 'completed_job',
    job_id: 'completed-job-b',
  });
});

test('completed QC bundle keeps same-job completed Pick Map without reloading', async () => {
  const existingPickMap = completedPickMap('completed-job-c', 'Existing same-job Pick Map');
  const pickMapCalls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    if (String(url) === '/statics/refraction/qc') {
      return jsonResponse(qcBundle('completed-job-c'));
    }
    if (String(url) === '/statics/refraction/qc/pick-map') {
      pickMapCalls.push(JSON.parse(options.body || '{}'));
      return jsonResponse(completedPickMap('completed-job-c', 'Reloaded Pick Map'));
    }
    throw new Error(`Unexpected fetch ${url}`);
  }));
  loadRefractionQcScript();
  window.refractionQcState.pickMap = existingPickMap;
  window.refractionQcUI.setSelectedView('pick_map');

  document.getElementById('refractionQcJobId').value = 'completed-job-c';
  await window.refractionQcUI.loadBundle();
  await flushAsyncWork();

  expect(pickMapCalls).toHaveLength(0);
  expect(window.refractionQcState.pickMap).toBe(existingPickMap);
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-status"]').textContent).toBe(
    'Existing same-job Pick Map'
  );
});

test('completed QC bundle cancels in-flight Pick Map without clearing same-job completed map', async () => {
  const existingPickMap = completedPickMap('completed-job-c', 'Existing same-job Pick Map');
  const preStaticsResponse = deferred();
  const pickMapCalls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    if (String(url) === '/statics/refraction/qc') {
      return jsonResponse(qcBundle('completed-job-c'));
    }
    if (String(url) === '/statics/refraction/qc/pick-map') {
      if (options.body instanceof FormData) {
        pickMapCalls.push({ mode: 'pre_statics' });
        return preStaticsResponse.promise;
      }
      pickMapCalls.push(JSON.parse(options.body || '{}'));
      return jsonResponse(completedPickMap('completed-job-c', 'Reloaded Pick Map'));
    }
    throw new Error(`Unexpected fetch ${url}`);
  }));
  window.SeisViewerState = activeViewerTarget();
  seedStaticCorrectionPickDraft();
  stubStaticCorrectionPickDb();
  loadRefractionQcScript();
  window.refractionQcState.pickMap = existingPickMap;
  window.refractionQcUI.setSelectedView('pick_map');
  await flushAsyncWork(4);

  document.querySelector('[data-testid="refraction-qc-pick-map-load-cached"]').click();

  expect(window.refractionQcState.pickMapLoading).toBe(true);
  expect(window.refractionQcState.pickMap).toBe(existingPickMap);
  document.getElementById('refractionQcJobId').value = 'completed-job-c';
  await window.refractionQcUI.loadBundle();
  await flushAsyncWork();

  expect(pickMapCalls).toEqual([{ mode: 'pre_statics' }]);
  expect(window.refractionQcState.pickMapLoading).toBe(false);
  expect(window.refractionQcState.pickMap).toBe(existingPickMap);

  preStaticsResponse.resolve(jsonResponse(preStaticsPickMap()));
  await flushAsyncWork();

  expect(window.refractionQcState.pickMap).toBe(existingPickMap);
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-status"]').textContent).toBe(
    'Existing same-job Pick Map'
  );
});

test('completed QC bundle reloads active completed Pick Map when job changes', async () => {
  const pickMapCalls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    if (String(url) === '/statics/refraction/qc') {
      return jsonResponse(qcBundle('completed-job-new'));
    }
    if (String(url) === '/statics/refraction/qc/pick-map') {
      pickMapCalls.push(JSON.parse(options.body || '{}'));
      return jsonResponse(completedPickMap('completed-job-new'));
    }
    throw new Error(`Unexpected fetch ${url}`);
  }));
  loadRefractionQcScript();
  window.refractionQcState.pickMap = completedPickMap('completed-job-old');
  window.refractionQcUI.setSelectedView('pick_map');

  document.getElementById('refractionQcJobId').value = 'completed-job-new';
  await window.refractionQcUI.loadBundle();
  await flushAsyncWork();

  expect(pickMapCalls).toEqual([{ job_id: 'completed-job-new' }]);
  expect(window.refractionQcState.pickMap).toMatchObject({
    mode: 'completed_job',
    job_id: 'completed-job-new',
  });
});

test('completed QC bundle invalidates in-flight pre-statics Pick Map', async () => {
  const preStaticsResponse = deferred();
  const pickMapCalls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    if (String(url) === '/statics/refraction/qc') {
      return jsonResponse(qcBundle('completed-job-d'));
    }
    if (String(url) === '/statics/refraction/qc/pick-map') {
      if (options.body instanceof FormData) {
        pickMapCalls.push({ mode: 'pre_statics' });
        return preStaticsResponse.promise;
      }
      pickMapCalls.push(JSON.parse(options.body || '{}'));
      return jsonResponse(completedPickMap('completed-job-d'));
    }
    throw new Error(`Unexpected fetch ${url}`);
  }));
  window.SeisViewerState = activeViewerTarget();
  seedStaticCorrectionPickDraft();
  stubStaticCorrectionPickDb();
  loadRefractionQcScript();
  window.refractionQcUI.setSelectedView('pick_map');
  await flushAsyncWork(4);

  document.querySelector('[data-testid="refraction-qc-pick-map-load-cached"]').click();

  expect(window.refractionQcState.pickMapLoading).toBe(true);
  document.getElementById('refractionQcJobId').value = 'completed-job-d';
  await window.refractionQcUI.loadBundle();
  await flushAsyncWork();

  expect(pickMapCalls).toEqual([
    { mode: 'pre_statics' },
    { job_id: 'completed-job-d' },
  ]);
  expect(window.refractionQcState.pickMap).toMatchObject({
    mode: 'completed_job',
    job_id: 'completed-job-d',
    has_after_statics: true,
  });

  preStaticsResponse.resolve(jsonResponse(preStaticsPickMap()));
  await flushAsyncWork();

  expect(window.refractionQcState.pickMap).toMatchObject({
    mode: 'completed_job',
    job_id: 'completed-job-d',
    has_after_statics: true,
  });
  expect(document.querySelector('[data-testid="refraction-qc-pick-map-after"]').disabled).toBe(false);
});
