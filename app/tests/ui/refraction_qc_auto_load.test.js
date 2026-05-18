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
      <section class="refraction-qc-view" data-view-panel="summary">
        <div data-view-content="summary"></div>
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
