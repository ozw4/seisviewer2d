import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { afterEach, beforeEach, expect, test, vi } from 'vitest';

const SCRIPT = readFileSync(
  resolve(process.cwd(), 'static/refraction_qc.js'),
  'utf8'
);

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

function jsonResponse(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  };
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

function qcBundle(jobId) {
  return {
    job_id: jobId,
    summary: { status: 'ready', workflow: 'refraction' },
    available_views: ['summary'],
    unavailable_views: [],
    coordinate_mode: 'auto',
  };
}

beforeEach(() => {
  localStorage.clear();
  window.history.replaceState(null, '', '/');
  delete window.RefractionQc;
  delete window.refractionQcUI;
  delete window.refractionQcState;
  renderRefractionQcPanel();
});

afterEach(() => {
  localStorage.clear();
  delete window.RefractionQc;
  delete window.refractionQcUI;
  delete window.refractionQcState;
  vi.unstubAllGlobals();
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
