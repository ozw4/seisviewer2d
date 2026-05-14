import { readFileSync } from 'node:fs';

import { afterEach, beforeEach, expect, test, vi } from 'vitest';

const SCRIPT = readFileSync(
  new URL('../../static/refraction_static_run.js', import.meta.url),
  'utf8'
);

function renderStaticCorrectionForm() {
  document.body.innerHTML = `
    <form id="staticCorrectionForm">
      <input id="staticCorrectionFileId" value="file-a" />
      <input id="staticCorrectionKey1Byte" value="189" />
      <input id="staticCorrectionKey2Byte" value="193" />
      <select id="staticCorrectionPickKind">
        <option value="batch_predicted_npz" selected>batch_predicted_npz</option>
      </select>
      <input id="staticCorrectionPickJobId" value="pick-job-a" />
      <input id="staticCorrectionPickArtifactName" value="predicted_picks_time_s.npz" />
      <button id="staticCorrectionLoadPickArtifactsButton" type="button"></button>
      <div id="staticCorrectionPickArtifactList"></div>
      <select id="staticCorrectionGeometryPreset">
        <option value="segy_default" selected>SEG-Y default</option>
        <option value="custom">custom</option>
      </select>
      <input id="staticCorrectionSourceIdByte" value="9" />
      <input id="staticCorrectionReceiverIdByte" value="13" />
      <input id="staticCorrectionSourceXByte" value="73" />
      <input id="staticCorrectionSourceYByte" value="77" />
      <input id="staticCorrectionReceiverXByte" value="81" />
      <input id="staticCorrectionReceiverYByte" value="85" />
      <input id="staticCorrectionSourceElevationByte" value="45" />
      <input id="staticCorrectionReceiverElevationByte" value="41" />
      <input id="staticCorrectionCoordinateScalarByte" value="71" />
      <input id="staticCorrectionElevationScalarByte" value="69" />
      <input id="staticCorrectionSourceDepthByte" value="" />
      <select id="staticCorrectionCoordinateUnit">
        <option value="m" selected>m</option>
        <option value="ft">ft</option>
      </select>
      <select id="staticCorrectionElevationUnit">
        <option value="m" selected>m</option>
        <option value="ft">ft</option>
      </select>
      <input id="staticCorrectionOffsetByte" value="37" />
      <input id="staticCorrectionEnableLinkage" type="checkbox" />
      <div id="staticCorrectionLinkageOptions" hidden>
        <select id="staticCorrectionLinkageMode">
          <option value="auto_threshold" selected>auto_threshold</option>
        </select>
        <input id="staticCorrectionLinkageThresholdM" value="25" />
        <input id="staticCorrectionReceiverLocationIntervalM" value="" />
        <input id="staticCorrectionPreferReceiverAnchor" type="checkbox" checked />
      </div>
      <button id="staticCorrectionRunButton" type="button"></button>
    </form>
    <div id="staticCorrectionStatus"></div>
    <div id="staticCorrectionError" hidden></div>
  `;
}

function loadStaticCorrectionScript() {
  window.eval(SCRIPT);
  document.dispatchEvent(new Event('DOMContentLoaded'));
  return window.refractionStaticRunUI;
}

function setCustomPreset() {
  const preset = document.getElementById('staticCorrectionGeometryPreset');
  preset.value = 'custom';
  preset.dispatchEvent(new Event('change'));
}

beforeEach(() => {
  delete window.refractionStaticRunUI;
  delete window.refractionStaticRunState;
  renderStaticCorrectionForm();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function jsonResponse(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: {
      get: () => 'application/json',
    },
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  };
}

function enableLinkage() {
  const checkbox = document.getElementById('staticCorrectionEnableLinkage');
  checkbox.checked = true;
  checkbox.dispatchEvent(new Event('change'));
}

function createStaticCorrectionFetchMock(statuses = [{ state: 'done', message: '', progress: 1 }]) {
  const calls = [];
  const statusQueue = [...statuses];
  const fetchMock = vi.fn(async (url, options = {}) => {
    const body = options.body ? JSON.parse(options.body) : null;
    calls.push({ url: String(url), options, body });

    if (url === '/statics/linkage/build') {
      return jsonResponse({ job_id: 'linkage-job-a', state: 'queued' });
    }
    if (url === '/statics/job/linkage-job-a/status') {
      return jsonResponse(statusQueue.shift() || statuses[statuses.length - 1]);
    }
    if (url === '/statics/refraction/apply') {
      return jsonResponse({ job_id: 'static-job-a', state: 'queued' });
    }
    throw new Error(`unexpected fetch ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return { calls, fetchMock };
}

test('geometry defaults render from the SEG-Y preset', () => {
  loadStaticCorrectionScript();

  expect(document.getElementById('staticCorrectionSourceIdByte').value).toBe('9');
  expect(document.getElementById('staticCorrectionReceiverIdByte').value).toBe('13');
  expect(document.getElementById('staticCorrectionCoordinateScalarByte').value).toBe('71');
  expect(document.getElementById('staticCorrectionElevationScalarByte').value).toBe('69');
  expect(document.getElementById('staticCorrectionOffsetByte').value).toBe('37');
  expect(document.getElementById('staticCorrectionCoordinateUnit').value).toBe('m');
  expect(document.getElementById('staticCorrectionSourceIdByte').disabled).toBe(true);
});

test('custom geometry values are included in the request fragment', () => {
  const ui = loadStaticCorrectionScript();
  setCustomPreset();

  document.getElementById('staticCorrectionSourceIdByte').value = '101';
  document.getElementById('staticCorrectionReceiverIdByte').value = '102';
  document.getElementById('staticCorrectionSourceXByte').value = '103';
  document.getElementById('staticCorrectionSourceYByte').value = '104';
  document.getElementById('staticCorrectionReceiverXByte').value = '105';
  document.getElementById('staticCorrectionReceiverYByte').value = '106';
  document.getElementById('staticCorrectionSourceElevationByte').value = '107';
  document.getElementById('staticCorrectionReceiverElevationByte').value = '108';
  document.getElementById('staticCorrectionCoordinateScalarByte').value = '109';
  document.getElementById('staticCorrectionElevationScalarByte').value = '110';
  document.getElementById('staticCorrectionSourceDepthByte').value = '111';
  document.getElementById('staticCorrectionCoordinateUnit').value = 'ft';
  document.getElementById('staticCorrectionElevationUnit').value = 'ft';
  document.getElementById('staticCorrectionOffsetByte').value = '112';

  expect(ui.buildStaticCorrectionGeometryRequest()).toEqual({
    geometry: {
      source_id_byte: 101,
      receiver_id_byte: 102,
      source_x_byte: 103,
      source_y_byte: 104,
      receiver_x_byte: 105,
      receiver_y_byte: 106,
      source_elevation_byte: 107,
      receiver_elevation_byte: 108,
      coordinate_scalar_byte: 109,
      elevation_scalar_byte: 110,
      source_depth_byte: 111,
      coordinate_unit: 'ft',
      elevation_unit: 'ft',
    },
    moveout: {
      offset_byte: 112,
    },
  });
});

test('invalid byte values block submit and show an error', () => {
  loadStaticCorrectionScript();
  setCustomPreset();
  document.getElementById('staticCorrectionSourceIdByte').value = '';

  document.getElementById('staticCorrectionRunButton').click();

  expect(document.getElementById('staticCorrectionError').hidden).toBe(false);
  expect(document.getElementById('staticCorrectionError').textContent).toContain(
    'geometry.source_id_byte'
  );
  expect(window.refractionStaticRunState.lastRequest).toBe(null);
});

test('preset switching restores defaults and toggles editability', () => {
  loadStaticCorrectionScript();
  setCustomPreset();
  const sourceId = document.getElementById('staticCorrectionSourceIdByte');
  sourceId.value = '101';
  expect(sourceId.disabled).toBe(false);

  const preset = document.getElementById('staticCorrectionGeometryPreset');
  preset.value = 'segy_default';
  preset.dispatchEvent(new Event('change'));

  expect(sourceId.value).toBe('9');
  expect(sourceId.disabled).toBe(true);

  preset.value = 'custom';
  preset.dispatchEvent(new Event('change'));
  expect(sourceId.value).toBe('9');
  expect(sourceId.disabled).toBe(false);
});

test('linkage checkbox is unchecked by default and builds none mode', () => {
  const ui = loadStaticCorrectionScript();

  expect(document.getElementById('staticCorrectionEnableLinkage').checked).toBe(false);
  expect(document.getElementById('staticCorrectionLinkageOptions').hidden).toBe(true);
  expect(document.getElementById('staticCorrectionLinkageMode').disabled).toBe(true);

  const result = ui.buildStaticCorrectionRequest();

  expect(result.errors).toEqual([]);
  expect(result.payload.linkage).toEqual({ mode: 'none' });
});

test('checked linkage reveals options and includes linkage builder fields', () => {
  const ui = loadStaticCorrectionScript();
  const checkbox = document.getElementById('staticCorrectionEnableLinkage');
  checkbox.checked = true;
  checkbox.dispatchEvent(new Event('change'));

  expect(document.getElementById('staticCorrectionLinkageOptions').hidden).toBe(false);
  expect(document.getElementById('staticCorrectionLinkageMode').disabled).toBe(false);
  expect(document.getElementById('staticCorrectionLinkageThresholdM').disabled).toBe(false);

  document.getElementById('staticCorrectionLinkageThresholdM').value = '12.5';
  document.getElementById('staticCorrectionReceiverLocationIntervalM').value = '25';
  document.getElementById('staticCorrectionPreferReceiverAnchor').checked = false;

  expect(ui.buildStaticCorrectionLinkageRequest()).toEqual({
    linkage: {
      mode: 'auto_threshold',
      threshold_m: 12.5,
      receiver_location_interval_m: 25,
      prefer_receiver_anchor: false,
    },
  });
});

test('checked auto-threshold linkage validates threshold only when enabled', () => {
  const ui = loadStaticCorrectionScript();
  document.getElementById('staticCorrectionLinkageThresholdM').value = '';

  expect(ui.buildStaticCorrectionRequest().errors).toEqual([]);

  const checkbox = document.getElementById('staticCorrectionEnableLinkage');
  checkbox.checked = true;
  checkbox.dispatchEvent(new Event('change'));
  document.getElementById('staticCorrectionRunButton').click();

  expect(document.getElementById('staticCorrectionError').hidden).toBe(false);
  expect(document.getElementById('staticCorrectionError').textContent).toContain(
    'linkage.threshold_m'
  );
  expect(window.refractionStaticRunState.lastRequest).toBe(null);
});

test('unchecked linkage skips linkage build when running static correction', async () => {
  const ui = loadStaticCorrectionScript();
  const { calls } = createStaticCorrectionFetchMock();

  await ui.runStaticCorrection();

  expect(calls.map((call) => call.url)).toEqual(['/statics/refraction/apply']);
  expect(calls[0].body.linkage).toEqual({ mode: 'none' });
});

test('checked linkage posts linkage build payload before static correction apply', async () => {
  const ui = loadStaticCorrectionScript();
  enableLinkage();
  document.getElementById('staticCorrectionLinkageThresholdM').value = '12.5';
  document.getElementById('staticCorrectionReceiverLocationIntervalM').value = '25';
  document.getElementById('staticCorrectionPreferReceiverAnchor').checked = false;
  const { calls } = createStaticCorrectionFetchMock();

  await ui.runStaticCorrection();

  expect(calls.map((call) => call.url)).toEqual([
    '/statics/linkage/build',
    '/statics/job/linkage-job-a/status',
    '/statics/refraction/apply',
  ]);
  expect(calls[0].body).toEqual({
    file_id: 'file-a',
    key1_byte: 189,
    key2_byte: 193,
    geometry: {
      source_x_byte: 73,
      source_y_byte: 77,
      receiver_x_byte: 81,
      receiver_y_byte: 85,
      coordinate_scalar_byte: 71,
    },
    linkage: {
      mode: 'auto_threshold',
      threshold_m: 12.5,
      receiver_location_interval_m: 25,
      prefer_receiver_anchor: false,
    },
  });
});

test('checked linkage polls linkage status until ready', async () => {
  const ui = loadStaticCorrectionScript();
  enableLinkage();
  window.refractionStaticRunState.pollIntervalMs = 0;
  const { calls } = createStaticCorrectionFetchMock([
    { state: 'queued', message: '', progress: 0 },
    { state: 'running', message: 'building', progress: 0.5 },
    { state: 'done', message: '', progress: 1 },
  ]);

  await ui.runStaticCorrection();

  expect(calls.filter((call) => call.url === '/statics/job/linkage-job-a/status')).toHaveLength(3);
  expect(calls[calls.length - 1].url).toBe('/statics/refraction/apply');
});

test('failed linkage prevents refraction apply submit', async () => {
  const ui = loadStaticCorrectionScript();
  enableLinkage();
  const { calls } = createStaticCorrectionFetchMock([
    { state: 'error', message: 'linkage build failed', progress: 1 },
  ]);

  await ui.runStaticCorrection();

  expect(calls.map((call) => call.url)).not.toContain('/statics/refraction/apply');
  expect(document.getElementById('staticCorrectionError').hidden).toBe(false);
  expect(document.getElementById('staticCorrectionError').textContent).toContain(
    'linkage build failed'
  );
  expect(document.getElementById('staticCorrectionStatus').textContent).toContain(
    'Static correction was not submitted'
  );
});

test('successful linkage injects linkage job reference into refraction request', async () => {
  const ui = loadStaticCorrectionScript();
  enableLinkage();
  const { calls } = createStaticCorrectionFetchMock();

  await ui.runStaticCorrection();

  const applyCall = calls.find((call) => call.url === '/statics/refraction/apply');
  expect(applyCall.body.linkage).toEqual({
    mode: 'required',
    job_id: 'linkage-job-a',
    artifact_name: 'geometry_linkage.npz',
  });
  expect(window.refractionStaticRunState.lastLinkageJobId).toBe('linkage-job-a');
  expect(document.getElementById('staticCorrectionStatus').textContent).toContain('static-job-a');
});
