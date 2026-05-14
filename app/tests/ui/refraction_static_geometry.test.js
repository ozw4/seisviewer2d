import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { afterEach, beforeEach, expect, test, vi } from 'vitest';

const SCRIPT = readFileSync(
  resolve(process.cwd(), 'static/refraction_static_run.js'),
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
      <select id="staticCorrectionModelKind">
        <option value="one_layer_t1lsst" selected>one-layer T1LSST</option>
      </select>
      <input id="staticCorrectionWeatheringVelocityMS" value="800" />
      <select id="staticCorrectionBedrockVelocityMode">
        <option value="solve_global" selected>solve_global</option>
        <option value="fixed_global">fixed_global</option>
      </select>
      <input id="staticCorrectionInitialBedrockVelocityMS" value="2400" />
      <input id="staticCorrectionFixedBedrockVelocityMS" value="2400" />
      <input id="staticCorrectionMinOffsetM" value="300" />
      <input id="staticCorrectionMaxOffsetM" value="4000" />
      <select id="staticCorrectionConversionMode">
        <option value="t1lsst_1layer" selected>t1lsst_1layer</option>
      </select>
      <input id="staticCorrectionRegisterCorrectedFile" type="checkbox" />
      <input id="staticCorrectionExportEnabled" type="checkbox" checked />
      <input
        id="staticCorrectionExportCanonicalTable"
        type="checkbox"
        value="canonical_static_table"
        checked
        data-static-correction-export-format
      />
      <input
        id="staticCorrectionExportTimeTermSpreadsheet"
        type="checkbox"
        value="time_term_spreadsheet"
        checked
        data-static-correction-export-format
      />
      <button id="staticCorrectionRunButton" type="button"></button>
      <button id="staticCorrectionCancelButton" type="button" hidden></button>
      <pre id="staticCorrectionRequestPreview" hidden></pre>
    </form>
    <div id="staticCorrectionStatus"></div>
    <div id="staticCorrectionJobPanel" hidden>
      <span id="staticCorrectionJobIdValue"></span>
      <span id="staticCorrectionJobStateValue"></span>
      <span id="staticCorrectionJobMessageValue"></span>
      <progress id="staticCorrectionJobProgress" max="1" value="0"></progress>
      <span id="staticCorrectionJobProgressValue"></span>
    </div>
    <table id="staticCorrectionArtifactTable" hidden>
      <tbody id="staticCorrectionArtifactBody"></tbody>
    </table>
    <div id="staticCorrectionArtifactEmpty"></div>
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
  delete window.RefractionQc;
  renderStaticCorrectionForm();
});

afterEach(() => {
  if (window.refractionStaticRunUI && window.refractionStaticRunUI.stopStaticCorrectionPolling) {
    window.refractionStaticRunUI.stopStaticCorrectionPolling();
  }
  delete window.RefractionQc;
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

async function flushAsyncWork(times = 4) {
  for (let index = 0; index < times; index += 1) {
    await Promise.resolve();
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

function createStaticCorrectionFetchMock(
  linkageStatuses = [{ state: 'done', message: '', progress: 1 }],
  staticStatuses = [{ state: 'done', message: 'finished', progress: 1 }]
) {
  const calls = [];
  const linkageStatusQueue = [...linkageStatuses];
  const staticStatusQueue = [...staticStatuses];
  const fetchMock = vi.fn(async (url, options = {}) => {
    const body = options.body ? JSON.parse(options.body) : null;
    calls.push({ url: String(url), options, body });

    if (url === '/statics/linkage/build') {
      return jsonResponse({ job_id: 'linkage-job-a', state: 'queued' });
    }
    if (url === '/statics/job/linkage-job-a/status') {
      return jsonResponse(
        linkageStatusQueue.shift() || linkageStatuses[linkageStatuses.length - 1]
      );
    }
    if (url === '/statics/refraction/apply') {
      return jsonResponse({ job_id: 'static-job-a', state: 'queued' });
    }
    if (url === '/statics/job/static-job-a/status') {
      return jsonResponse(
        staticStatusQueue.shift() || staticStatuses[staticStatuses.length - 1]
      );
    }
    if (url === '/statics/job/static-job-a/files') {
      return jsonResponse({
        files: [
          { name: 'refraction_static_artifacts.json', size_bytes: 123 },
          { name: 'source_static_table.csv', size_bytes: 2048 },
        ],
      });
    }
    if (url === '/statics/job/static-job-a/cancel') {
      return jsonResponse({
        state: 'cancel_requested',
        progress: 0.25,
        message: 'Cancel requested. The job will stop at the next safe point.',
      });
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

test('one-layer validation ignores inactive bedrock velocity inputs', () => {
  const ui = loadStaticCorrectionScript();
  const mode = document.getElementById('staticCorrectionBedrockVelocityMode');
  const initialVelocity = document.getElementById('staticCorrectionInitialBedrockVelocityMS');
  const fixedVelocity = document.getElementById('staticCorrectionFixedBedrockVelocityMS');

  fixedVelocity.value = 'not-a-number';
  expect(ui.buildStaticCorrectionRequest().errors).toEqual([]);

  mode.value = 'fixed_global';
  mode.dispatchEvent(new Event('change'));
  initialVelocity.value = 'not-a-number';
  fixedVelocity.value = '2400';

  expect(ui.buildStaticCorrectionRequest().errors).toEqual([]);
});

test('unchecked linkage skips linkage build when running static correction', async () => {
  const ui = loadStaticCorrectionScript();
  const { calls } = createStaticCorrectionFetchMock();

  await ui.runStaticCorrection();
  await flushAsyncWork();

  expect(calls.map((call) => call.url)).toEqual([
    '/statics/refraction/apply',
    '/statics/job/static-job-a/status',
    '/statics/job/static-job-a/files',
  ]);
  expect(calls[0].body.linkage).toEqual({ mode: 'none' });
});

test('static correction submit failure returns to a retryable idle state', async () => {
  const ui = loadStaticCorrectionScript();
  const calls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    calls.push({ url: String(url), options });
    if (url === '/statics/refraction/apply') {
      return jsonResponse({ detail: 'apply failed' }, 400);
    }
    throw new Error(`unexpected fetch ${url}`);
  }));

  await ui.runStaticCorrection();
  await flushAsyncWork();

  expect(calls.map((call) => call.url)).toEqual(['/statics/refraction/apply']);
  expect(window.refractionStaticRunState.phase).toBe('idle');
  expect(document.getElementById('staticCorrectionRunButton').disabled).toBe(false);
  expect(document.getElementById('staticCorrectionError').textContent).toContain('apply failed');
  expect(document.getElementById('staticCorrectionStatus').textContent).toContain(
    'Static correction submission failed'
  );
});

test('checked linkage posts linkage build payload before static correction apply', async () => {
  const ui = loadStaticCorrectionScript();
  enableLinkage();
  document.getElementById('staticCorrectionLinkageThresholdM').value = '12.5';
  document.getElementById('staticCorrectionReceiverLocationIntervalM').value = '25';
  document.getElementById('staticCorrectionPreferReceiverAnchor').checked = false;
  const { calls } = createStaticCorrectionFetchMock();

  await ui.runStaticCorrection();
  await flushAsyncWork();

  expect(calls.map((call) => call.url).slice(0, 3)).toEqual([
    '/statics/linkage/build',
    '/statics/job/linkage-job-a/status',
    '/statics/refraction/apply',
  ]);
  expect(calls[0].body).toEqual({
    file_id: 'file-a',
    key1_byte: 189,
    key2_byte: 193,
    geometry: {
      source_id_byte: 9,
      receiver_id_byte: 13,
      source_x_byte: 73,
      source_y_byte: 77,
      receiver_x_byte: 81,
      receiver_y_byte: 85,
      source_elevation_byte: 45,
      receiver_elevation_byte: 41,
      coordinate_scalar_byte: 71,
      elevation_scalar_byte: 69,
      source_depth_byte: null,
      coordinate_unit: 'm',
      elevation_unit: 'm',
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
  await flushAsyncWork();

  expect(calls.filter((call) => call.url === '/statics/job/linkage-job-a/status')).toHaveLength(3);
  expect(calls.map((call) => call.url)).toContain('/statics/refraction/apply');
});

test('failed linkage prevents refraction apply submit', async () => {
  const ui = loadStaticCorrectionScript();
  enableLinkage();
  const { calls } = createStaticCorrectionFetchMock([
    { state: 'error', message: 'linkage build failed', progress: 1 },
  ]);

  await ui.runStaticCorrection();
  await flushAsyncWork();

  expect(calls.map((call) => call.url)).not.toContain('/statics/refraction/apply');
  expect(document.getElementById('staticCorrectionError').hidden).toBe(false);
  expect(document.getElementById('staticCorrectionError').textContent).toContain(
    'linkage build failed'
  );
  expect(document.getElementById('staticCorrectionStatus').textContent).toContain(
    'Static correction was not submitted'
  );
  expect(window.refractionStaticRunState.phase).toBe('idle');
  expect(document.getElementById('staticCorrectionRunButton').disabled).toBe(false);
});

test('successful linkage injects linkage job reference into refraction request', async () => {
  const ui = loadStaticCorrectionScript();
  enableLinkage();
  window.refractionStaticRunState.pollIntervalMs = 0;
  const { calls } = createStaticCorrectionFetchMock([
    { state: 'completed', message: '', progress: 1 },
  ]);

  await ui.runStaticCorrection();
  await flushAsyncWork();

  const applyCall = calls.find((call) => call.url === '/statics/refraction/apply');
  expect(applyCall.body.linkage).toEqual({
    mode: 'required',
    job_id: 'linkage-job-a',
    artifact_name: 'geometry_linkage.npz',
  });
  expect(window.refractionStaticRunState.lastLinkageJobId).toBe('linkage-job-a');
  expect(document.getElementById('staticCorrectionStatus').textContent).toContain('static-job-a');
});

test('static correction submit polls job status and loads ready artifacts', async () => {
  const ui = loadStaticCorrectionScript();
  window.refractionStaticRunState.pollIntervalMs = 0;
  const { calls } = createStaticCorrectionFetchMock(
    [{ state: 'done', message: '', progress: 1 }],
    [
      { state: 'queued', message: 'waiting', progress: 0 },
      { state: 'running', message: 'solving statics', progress: 0.5 },
      { state: 'ready', message: 'finished', progress: 1 },
    ]
  );

  await ui.runStaticCorrection();
  await flushAsyncWork(8);

  expect(calls.filter((call) => call.url === '/statics/job/static-job-a/status')).toHaveLength(3);
  expect(calls.map((call) => call.url)).toContain('/statics/job/static-job-a/files');
  expect(document.getElementById('staticCorrectionJobIdValue').textContent).toBe('static-job-a');
  expect(document.getElementById('staticCorrectionJobStateValue').textContent).toBe('ready');
  expect(document.getElementById('staticCorrectionJobMessageValue').textContent).toBe('finished');
  expect(document.getElementById('staticCorrectionCancelButton').hidden).toBe(true);
  expect(document.getElementById('staticCorrectionArtifactBody').textContent).toContain(
    'source_static_table.csv'
  );
  expect(document.querySelector('a[href="/statics/job/static-job-a/download?name=source_static_table.csv"]')).not.toBeNull();
});

test('ready static correction job auto-loads refraction QC with the completed job id', async () => {
  const ui = loadStaticCorrectionScript();
  window.refractionStaticRunState.pollIntervalMs = 0;
  const loadJob = vi.fn(async () => ({ job_id: 'static-job-a' }));
  window.RefractionQc = { loadJob };
  createStaticCorrectionFetchMock(
    [{ state: 'done', message: '', progress: 1 }],
    [{ state: 'ready', message: 'finished', progress: 1 }]
  );

  await ui.runStaticCorrection();
  await flushAsyncWork();

  expect(loadJob).toHaveBeenCalledWith('static-job-a', { activateTab: true });
});

test('static correction polling stops on failed state and shows an error', async () => {
  const ui = loadStaticCorrectionScript();
  window.refractionStaticRunState.pollIntervalMs = 0;
  const { calls } = createStaticCorrectionFetchMock(
    [{ state: 'done', message: '', progress: 1 }],
    [{ state: 'error', message: 'solver failed', progress: 1 }]
  );

  await ui.runStaticCorrection();
  await flushAsyncWork();

  expect(calls.filter((call) => call.url === '/statics/job/static-job-a/status')).toHaveLength(1);
  expect(calls.map((call) => call.url)).not.toContain('/statics/job/static-job-a/files');
  expect(document.getElementById('staticCorrectionError').hidden).toBe(false);
  expect(document.getElementById('staticCorrectionError').textContent).toContain('solver failed');
  expect(document.getElementById('staticCorrectionJobStateValue').textContent).toBe('error');
});

test('static correction cancel posts to the static cancel endpoint', async () => {
  const ui = loadStaticCorrectionScript();
  const { calls } = createStaticCorrectionFetchMock(
    [{ state: 'done', message: '', progress: 1 }],
    [{ state: 'running', message: 'working', progress: 0.25 }]
  );

  await ui.runStaticCorrection();
  await flushAsyncWork();
  expect(document.getElementById('staticCorrectionCancelButton').hidden).toBe(false);

  await ui.cancelStaticCorrectionJob();

  const cancelCall = calls.find((call) => call.url === '/statics/job/static-job-a/cancel');
  expect(cancelCall).toBeTruthy();
  expect(cancelCall.options.method).toBe('POST');
  expect(document.getElementById('staticCorrectionJobStateValue').textContent).toBe(
    'cancel_requested'
  );
  expect(document.getElementById('staticCorrectionCancelButton').disabled).toBe(true);
});
