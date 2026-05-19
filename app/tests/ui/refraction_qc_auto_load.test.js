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
      <select id="refractionQcLayerKind"><option value="all">All</option></select>
      <select id="refractionQcXAxisMode"><option value="offset">Offset</option></select>
      <select id="refractionQcProfileGroup"><option value="time_terms">Time terms</option></select>
      <select id="refractionQcProfileUnits"><option value="auto">Auto</option></select>
      <select id="refractionQcStatusFilter"><option value="all">All</option></select>
      <select id="refractionQcMapQuantity"><option value="velocity">Velocity</option></select>
      <input id="refractionQcShowRejected" type="checkbox" checked />
      <select id="refractionQcEndpointKind"><option value="source">source</option></select>
      <input id="refractionQcEndpoint" />
      <input id="refractionQcTrace" />
      <input id="refractionQcCell" />
      <button type="button" class="refraction-qc-view-button" data-view="summary"></button>
      <button type="button" class="refraction-qc-view-button" data-view="pick_map"></button>
      <section class="refraction-qc-view" data-view-panel="summary">
        <div data-view-content="summary"></div>
      </section>
      <section class="refraction-qc-view" data-view-panel="pick_map" hidden>
        <div data-view-content="pick_map"></div>
      </section>
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
