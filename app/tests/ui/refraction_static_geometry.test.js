import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { afterEach, beforeEach, expect, test, vi } from 'vitest';

const SCRIPT = readFileSync(
  resolve(process.cwd(), 'static/refraction_static_run.js'),
  'utf8'
);
const INDEX_HTML = readFileSync(resolve(process.cwd(), 'static/index.html'), 'utf8');

function renderStaticCorrectionForm() {
  document.body.innerHTML = `
    <form id="staticCorrectionForm">
      <div id="staticCorrectionTargetEmpty"></div>
      <div id="staticCorrectionTargetDetails" hidden>
        <span id="staticCorrectionTargetFile"></span>
        <span id="staticCorrectionTargetKeys"></span>
        <span id="staticCorrectionTargetStatus"></span>
      </div>
      <select id="staticCorrectionPresetSelect">
        <option value="">No saved presets</option>
      </select>
      <input id="staticCorrectionPresetName" value="" />
      <button id="staticCorrectionSavePresetButton" type="button"></button>
      <button id="staticCorrectionLoadPresetButton" type="button"></button>
      <button id="staticCorrectionDeletePresetButton" type="button"></button>
      <input id="staticCorrectionPickNpz" type="file" accept=".npz" />
      <div id="staticCorrectionPickNpzSummary"></div>
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
        <option value="one_layer_global" selected>One-layer global V2/T1</option>
        <option value="two_layer_global">Two-layer global V3/T2</option>
        <option value="three_layer_global">Three-layer global Vsub/T3</option>
        <option value="cell_v2_t1_line_2d">Cell V2/T1 - 2D projected line</option>
        <option value="cell_v2_t1_grid_3d">Cell V2/T1 - 3D grid</option>
      </select>
      <input id="staticCorrectionWeatheringVelocityMS" value="800" />
      <select id="staticCorrectionBedrockVelocityMode">
        <option value="solve_global" selected>solve_global</option>
        <option value="fixed_global">fixed_global</option>
        <option value="solve_cell" disabled>solve_cell</option>
      </select>
      <input id="staticCorrectionInitialBedrockVelocityMS" value="2400" />
      <input id="staticCorrectionFixedBedrockVelocityMS" value="2400" />
      <input id="staticCorrectionMinOffsetM" value="300" />
      <input id="staticCorrectionMaxOffsetM" value="4000" />
      <select id="staticCorrectionConversionMode">
        <option value="t1lsst_1layer" selected>t1lsst_1layer</option>
        <option value="t1lsst_multilayer">t1lsst_multilayer</option>
      </select>
      <div id="staticCorrectionV3LayerFields" hidden>
        <input id="staticCorrectionV3MinOffsetM" value="4000" />
        <input id="staticCorrectionV3MaxOffsetM" value="6000" />
        <input id="staticCorrectionInitialV3VelocityMS" value="3600" />
      </div>
      <div id="staticCorrectionVsubLayerFields" hidden>
        <input id="staticCorrectionVsubMinOffsetM" value="6000" />
        <input id="staticCorrectionInitialVsubVelocityMS" value="5000" />
      </div>
      <div id="staticCorrectionCellFields" hidden>
        <input id="staticCorrectionCellXOriginM" value="0" />
        <input id="staticCorrectionCellYOriginM" value="0" />
        <input id="staticCorrectionCellCountX" value="20" />
        <input id="staticCorrectionCellCountY" value="1" />
        <input id="staticCorrectionCellSizeXM" value="500" />
        <input id="staticCorrectionCellSizeYM" value="500" />
        <input id="staticCorrectionCellMinObservations" value="5" />
        <input id="staticCorrectionCellSmoothingWeight" value="0" />
      </div>
      <div id="staticCorrectionLine2DFields" hidden>
        <input id="staticCorrectionLineOriginXM" value="0" />
        <input id="staticCorrectionLineOriginYM" value="0" />
        <input id="staticCorrectionLineAzimuthDeg" value="0" />
      </div>
      <input id="staticCorrectionFieldCorrectionsEnabled" type="checkbox" />
      <div id="staticCorrectionFieldCorrectionOptions" hidden>
        <select id="staticCorrectionFieldSourceDepthMode">
          <option value="none" selected>none</option>
          <option value="weathering_velocity_time">weathering_velocity_time</option>
        </select>
        <input id="staticCorrectionFieldSourceDepthByte" value="" />
        <select id="staticCorrectionFieldUpholeMode">
          <option value="none" selected>none</option>
          <option value="header_time">header_time</option>
        </select>
        <input id="staticCorrectionFieldUpholeTimeByte" value="" />
        <select id="staticCorrectionFieldManualStaticMode">
          <option value="none" selected>none</option>
          <option value="artifact_table">artifact_table</option>
        </select>
        <div id="staticCorrectionFieldManualArtifactFields" hidden>
          <select id="staticCorrectionFieldManualStaticSignConvention">
            <option value="applied_shift_s" selected>applied_shift_s</option>
            <option value="delay_positive_ms">delay_positive_ms</option>
          </select>
          <input id="staticCorrectionFieldManualSourceJobId" value="" />
          <input id="staticCorrectionFieldManualSourceArtifactName" value="" />
          <input id="staticCorrectionFieldManualReceiverJobId" value="" />
          <input id="staticCorrectionFieldManualReceiverArtifactName" value="" />
        </div>
        <input id="staticCorrectionFieldApplyToTraceShift" type="checkbox" checked />
      </div>
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
        id="staticCorrectionExportLsst"
        type="checkbox"
        value="lsst"
        data-static-correction-export-format
      />
      <input
        id="staticCorrectionExportLsstPlus"
        type="checkbox"
        value="lsst_plus"
        checked
        data-static-correction-export-format
      />
      <input
        id="staticCorrectionExportTimeTermSpreadsheet"
        type="checkbox"
        value="time_term_spreadsheet"
        data-static-correction-export-format
      />
      <input
        id="staticCorrectionExportFirstBreakTime"
        type="checkbox"
        value="first_break_time"
        data-static-correction-export-format
      />
      <button id="staticCorrectionRunButton" type="button"></button>
      <button id="staticCorrectionCancelButton" type="button" hidden></button>
      <div id="staticCorrectionValidationSummary" hidden></div>
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

let activeViewerTarget;

function setViewerTarget(fileId = 'file-a', key1Byte = 189, key2Byte = 193) {
  activeViewerTarget = fileId
    ? {
        fileId,
        key1Byte,
        key2Byte,
        displayName: fileId,
      }
    : null;
}

function selectedPickFile(name = 'first-break-picks.npz', bytes = 'npz-bytes') {
  const input = document.getElementById('staticCorrectionPickNpz');
  const file = new File([bytes], name, { type: 'application/octet-stream' });
  Object.defineProperty(input, 'files', {
    value: [file],
    configurable: true,
  });
  return file;
}

function clearSelectedPickFile() {
  Object.defineProperty(document.getElementById('staticCorrectionPickNpz'), 'files', {
    value: [],
    configurable: true,
  });
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
  setViewerTarget();
  window.SeisViewerState = {
    getActiveFileTarget: () => activeViewerTarget,
  };
  window.localStorage.clear();
  renderStaticCorrectionForm();
  selectedPickFile();
});

afterEach(() => {
  if (window.refractionStaticRunUI && window.refractionStaticRunUI.stopStaticCorrectionPolling) {
    window.refractionStaticRunUI.stopStaticCorrectionPolling();
  }
  delete window.RefractionQc;
  delete window.SeisViewerState;
  vi.unstubAllGlobals();
  window.localStorage.clear();
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

function enableFieldCorrections() {
  const checkbox = document.getElementById('staticCorrectionFieldCorrectionsEnabled');
  checkbox.checked = true;
  checkbox.dispatchEvent(new Event('change'));
}

function selectModelPreset(value) {
  const select = document.getElementById('staticCorrectionModelKind');
  select.value = value;
  select.dispatchEvent(new Event('change'));
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
    let body = null;
    let formData = null;
    if (options.body instanceof FormData) {
      formData = options.body;
      const requestJson = formData.get('request_json');
      body = requestJson ? JSON.parse(requestJson) : null;
    } else if (options.body) {
      body = JSON.parse(options.body);
    }
    calls.push({ url: String(url), options, body, formData });

    if (url === '/statics/linkage/build') {
      return jsonResponse({ job_id: 'linkage-job-a', state: 'queued' });
    }
    if (url === '/statics/job/linkage-job-a/status') {
      return jsonResponse(
        linkageStatusQueue.shift() || linkageStatuses[linkageStatuses.length - 1]
      );
    }
    if (url === '/statics/refraction/apply-with-picks') {
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

test('selected pick NPZ renders in the file summary', () => {
  selectedPickFile('manual_picks_time_review.npz', 'abc');
  loadStaticCorrectionScript();

  expect(document.getElementById('staticCorrectionPickNpzSummary').textContent).toContain(
    'manual_picks_time_review.npz'
  );
});

test('missing pick NPZ blocks submit and shows validation', async () => {
  const ui = loadStaticCorrectionScript();
  clearSelectedPickFile();
  document.getElementById('staticCorrectionPickNpz').dispatchEvent(new Event('change'));

  await ui.runStaticCorrection();
  await flushAsyncWork();

  expect(document.getElementById('staticCorrectionError').hidden).toBe(false);
  expect(document.getElementById('staticCorrectionError').textContent).toContain(
    'First-break pick NPZ is required'
  );
  expect(window.refractionStaticRunState.lastRequest).toBe(null);
});

test('non-NPZ pick upload blocks submit', async () => {
  selectedPickFile('picks.txt', 'abc');
  const ui = loadStaticCorrectionScript();

  await ui.runStaticCorrection();
  await flushAsyncWork();

  expect(document.getElementById('staticCorrectionError').hidden).toBe(false);
  expect(document.getElementById('staticCorrectionError').textContent).toContain(
    'First-break pick file must use the .npz extension'
  );
  expect(window.refractionStaticRunState.lastRequest).toBe(null);
});

test('static correction request always uses uploaded NPZ pick source', () => {
  const ui = loadStaticCorrectionScript();

  const result = ui.buildStaticCorrectionRequest();

  expect(result.errors).toEqual([]);
  expect(result.payload.pick_source).toEqual({ kind: 'uploaded_npz' });
});

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

test('request preview renders valid JSON and updates when fields change', () => {
  loadStaticCorrectionScript();

  let preview = JSON.parse(document.getElementById('staticCorrectionRequestPreview').textContent);
  expect(preview.file_id).toBe('file-a');
  expect(preview.model.first_layer.weathering_velocity_m_s).toBe(800);

  const velocity = document.getElementById('staticCorrectionWeatheringVelocityMS');
  velocity.value = '925';
  velocity.dispatchEvent(new Event('input', { bubbles: true }));

  preview = JSON.parse(document.getElementById('staticCorrectionRequestPreview').textContent);
  expect(preview.model.first_layer.weathering_velocity_m_s).toBe(925);
});

test('request preview supports uploaded pick source without artifact fields', () => {
  loadStaticCorrectionScript();

  const preview = JSON.parse(document.getElementById('staticCorrectionRequestPreview').textContent);
  expect(preview.pick_source).toEqual({ kind: 'uploaded_npz' });
  expect(INDEX_HTML).toContain('Pick NPZ is sent as multipart file field: pick_npz');
  expect(INDEX_HTML).not.toContain('staticCorrectionPickJobId');
  expect(INDEX_HTML).not.toContain('staticCorrectionPickArtifactName');
  expect(INDEX_HTML).not.toContain('staticCorrectionLoadPickArtifactsButton');
});

test('static correction does not render manual target id or sort key inputs', () => {
  loadStaticCorrectionScript();

  expect(document.querySelector('#staticCorrectionFileId')).toBeNull();
  expect(document.querySelector('#staticCorrectionKey1Byte')).toBeNull();
  expect(document.querySelector('#staticCorrectionKey2Byte')).toBeNull();
  expect(INDEX_HTML).not.toContain('name="file_id"');
  expect(INDEX_HTML).not.toContain('name="key1_byte"');
  expect(INDEX_HTML).not.toContain('name="key2_byte"');
  expect(document.getElementById('staticCorrectionTargetFile').textContent).toContain('file-a');
  expect(document.getElementById('staticCorrectionTargetKeys').textContent).toContain(
    'key1=189, key2=193'
  );
});

test('request preview uses current viewer target', () => {
  setViewerTarget('current-line', 9, 13);
  loadStaticCorrectionScript();

  const preview = JSON.parse(document.getElementById('staticCorrectionRequestPreview').textContent);
  expect(preview.file_id).toBe('current-line');
  expect(preview.key1_byte).toBe(9);
  expect(preview.key2_byte).toBe(13);
});

test('validation error no active viewer file', () => {
  setViewerTarget(null);
  const ui = loadStaticCorrectionScript();

  const result = ui.buildStaticCorrectionRequest();

  expect(result.payload).toBe(null);
  expect(result.errors).toContain(
    'No active viewer file. Open an SGY/TraceStore in the viewer before running Static Correction.'
  );
});

test('validation error no pick npz', () => {
  clearSelectedPickFile();
  const ui = loadStaticCorrectionScript();

  const result = ui.buildStaticCorrectionRequest();

  expect(result.payload).toBe(null);
  expect(result.errors).toContain('First-break pick NPZ is required.');
});

test('validation error missing viewer target fields', () => {
  activeViewerTarget = {
    fileId: '',
    key1Byte: '',
    key2Byte: null,
    displayName: 'Line A',
  };
  const ui = loadStaticCorrectionScript();

  const result = ui.buildStaticCorrectionRequest();

  expect(result.payload).toBe(null);
  expect(result.errors).toEqual(expect.arrayContaining([
    'Active viewer target is missing fileId.',
    'Active viewer target is missing key1Byte.',
    'Active viewer target is missing key2Byte.',
  ]));
  expect(document.getElementById('staticCorrectionTargetEmpty').textContent).toContain(
    'Active viewer target is missing fileId.'
  );
});

test('validation error missing required geometry byte', () => {
  const ui = loadStaticCorrectionScript();
  setCustomPreset();
  document.getElementById('staticCorrectionSourceIdByte').value = '';

  const result = ui.buildStaticCorrectionRequest();

  expect(result.payload).toBe(null);
  expect(result.errors).toContain(
    'geometry.source_id_byte must be an integer SEG-Y trace header byte from 1 to 240.'
  );
});

test('validation summary lists invalid fields before submit', () => {
  loadStaticCorrectionScript();
  setViewerTarget(null);
  clearSelectedPickFile();

  document.getElementById('staticCorrectionRunButton').click();

  const summary = document.getElementById('staticCorrectionValidationSummary');
  expect(summary.hidden).toBe(false);
  expect(summary.textContent).toContain('No active viewer file');
  expect(summary.textContent).toContain('First-break pick NPZ is required');
});

test('save preset writes localStorage without transient job state', () => {
  loadStaticCorrectionScript();
  document.getElementById('staticCorrectionPresetName').value = 'field layout';
  document.getElementById('staticCorrectionSavePresetButton').click();

  const presets = JSON.parse(window.localStorage.getItem('sv.static_correction.presets'));
  expect(presets).toHaveLength(1);
  expect(presets[0].name).toBe('field layout');
  expect(presets[0].values).not.toHaveProperty('file_id');
  expect(presets[0].values).not.toHaveProperty('pick_source');
  expect(presets[0].values).not.toHaveProperty('staticArtifacts');
});

test('load preset restores model and header fields while keeping current ids', () => {
  loadStaticCorrectionScript();
  setCustomPreset();
  document.getElementById('staticCorrectionSourceIdByte').value = '101';
  selectModelPreset('two_layer_global');
  document.getElementById('staticCorrectionWeatheringVelocityMS').value = '900';
  document.getElementById('staticCorrectionPresetName').value = 'two-layer custom';
  document.getElementById('staticCorrectionSavePresetButton').click();

  setViewerTarget('current-file', 17, 21);
  selectedPickFile('current-picks.npz');
  document.getElementById('staticCorrectionSourceIdByte').value = '9';
  selectModelPreset('one_layer_global');
  document.getElementById('staticCorrectionWeatheringVelocityMS').value = '700';
  document.getElementById('staticCorrectionLoadPresetButton').click();

  expect(window.SeisViewerState.getActiveFileTarget().fileId).toBe('current-file');
  expect(document.getElementById('staticCorrectionPickNpz').files[0].name).toBe('current-picks.npz');
  expect(document.getElementById('staticCorrectionGeometryPreset').value).toBe('custom');
  expect(document.getElementById('staticCorrectionSourceIdByte').value).toBe('101');
  expect(document.getElementById('staticCorrectionModelKind').value).toBe('two_layer_global');
  expect(document.getElementById('staticCorrectionWeatheringVelocityMS').value).toBe('900');
});

test('delete preset removes it from localStorage', () => {
  loadStaticCorrectionScript();
  document.getElementById('staticCorrectionPresetName').value = 'delete me';
  document.getElementById('staticCorrectionSavePresetButton').click();

  document.getElementById('staticCorrectionDeletePresetButton').click();

  expect(JSON.parse(window.localStorage.getItem('sv.static_correction.presets'))).toEqual([]);
  expect(document.getElementById('staticCorrectionPresetSelect').textContent).toContain('No saved presets');
});

test('loading preset does not submit a job', () => {
  loadStaticCorrectionScript();
  const fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
  document.getElementById('staticCorrectionPresetName').value = 'quiet load';
  document.getElementById('staticCorrectionSavePresetButton').click();

  document.getElementById('staticCorrectionLoadPresetButton').click();

  expect(fetchMock).not.toHaveBeenCalled();
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

test('field correction and export sections are collapsed details controls', () => {
  expect(INDEX_HTML).toContain('id="staticCorrectionPickNpz"');
  expect(INDEX_HTML).not.toContain('id="staticCorrectionPickKind"');
  expect(INDEX_HTML).toContain('<details id="staticCorrectionFieldCorrectionsSection"');
  expect(INDEX_HTML).toContain('<summary id="staticCorrectionFieldCorrectionsHeading">Field corrections</summary>');
  expect(INDEX_HTML).toContain('<details id="staticCorrectionExportSection"');
  expect(INDEX_HTML).toContain('<summary id="staticCorrectionExportHeading">Exports</summary>');
});

test('field corrections disabled omits optional field_corrections request block', () => {
  const ui = loadStaticCorrectionScript();

  const result = ui.buildStaticCorrectionRequest();

  expect(result.errors).toEqual([]);
  expect(result.payload).not.toHaveProperty('field_corrections');
});

test('source-depth weathering_velocity_time request fragment includes source_depth_byte', () => {
  const ui = loadStaticCorrectionScript();
  enableFieldCorrections();
  const mode = document.getElementById('staticCorrectionFieldSourceDepthMode');
  mode.value = 'weathering_velocity_time';
  mode.dispatchEvent(new Event('change'));
  document.getElementById('staticCorrectionFieldSourceDepthByte').value = '115';

  const result = ui.buildStaticCorrectionRequest();

  expect(result.errors).toEqual([]);
  expect(result.payload.field_corrections.source_depth).toEqual({
    mode: 'weathering_velocity_time',
    source_depth_byte: 115,
  });
  expect(result.payload.field_corrections.composition.apply_to_trace_shift).toBe(true);
});

test('uphole header_time request fragment includes uphole_time_byte', () => {
  const ui = loadStaticCorrectionScript();
  enableFieldCorrections();
  const mode = document.getElementById('staticCorrectionFieldUpholeMode');
  mode.value = 'header_time';
  mode.dispatchEvent(new Event('change'));
  document.getElementById('staticCorrectionFieldUpholeTimeByte').value = '95';

  const result = ui.buildStaticCorrectionRequest();

  expect(result.errors).toEqual([]);
  expect(result.payload.field_corrections.uphole).toEqual({
    mode: 'header_time',
    uphole_time_byte: 95,
  });
});

test('apply_to_trace_shift checkbox changes the field correction request', () => {
  const ui = loadStaticCorrectionScript();
  enableFieldCorrections();
  document.getElementById('staticCorrectionFieldApplyToTraceShift').checked = false;

  const result = ui.buildStaticCorrectionRequest();

  expect(result.errors).toEqual([]);
  expect(result.payload.field_corrections.composition).toEqual({
    enabled: true,
    apply_to_trace_shift: false,
  });
});

test('manual static artifact mode validates and builds artifact refs', () => {
  const ui = loadStaticCorrectionScript();
  enableFieldCorrections();
  const mode = document.getElementById('staticCorrectionFieldManualStaticMode');
  mode.value = 'artifact_table';
  mode.dispatchEvent(new Event('change'));
  document.getElementById('staticCorrectionFieldManualSourceJobId').value = 'static-job-a';
  document.getElementById('staticCorrectionFieldManualSourceArtifactName').value = 'source_static_table.csv';

  const result = ui.buildStaticCorrectionRequest();

  expect(result.errors).toEqual([]);
  expect(result.payload.field_corrections.manual_static).toEqual({
    mode: 'artifact_table',
    sign_convention: 'applied_shift_s',
    source_table_artifact: {
      job_id: 'static-job-a',
      artifact_name: 'source_static_table.csv',
    },
  });
});

test('export format checkboxes build export formats list', () => {
  const ui = loadStaticCorrectionScript();
  document.getElementById('staticCorrectionExportLsst').checked = true;
  document.getElementById('staticCorrectionExportFirstBreakTime').checked = true;

  const result = ui.buildStaticCorrectionRequest();

  expect(result.errors).toEqual([]);
  expect(result.payload.export).toEqual({
    enabled: true,
    formats: [
      'canonical_static_table',
      'lsst',
      'lsst_plus',
      'first_break_time',
    ],
  });
});

test('unsupported seconds export option is not present in the UI', () => {
  loadStaticCorrectionScript();

  expect(document.querySelector('[name="export.units"]')).toBeNull();
  expect(INDEX_HTML).not.toContain('value="seconds"');
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

test('one-layer model preset is the default request shape', () => {
  const ui = loadStaticCorrectionScript();

  const result = ui.buildStaticCorrectionRequest();

  expect(document.getElementById('staticCorrectionModelKind').value).toBe('one_layer_global');
  expect(result.errors).toEqual([]);
  expect(result.payload.model).toMatchObject({
    method: 'gli_variable_thickness',
    first_layer: {
      mode: 'constant',
      weathering_velocity_m_s: 800,
    },
    bedrock_velocity_mode: 'solve_global',
    initial_bedrock_velocity_m_s: 2400,
  });
  expect(result.payload.conversion).toEqual({ mode: 't1lsst_1layer' });
});

test('two-layer preset builds public global V3/T2 request', () => {
  const ui = loadStaticCorrectionScript();
  setViewerTarget('line-preserved', 17, 21);
  selectedPickFile('pick-preserved.npz');

  selectModelPreset('two_layer_global');
  document.getElementById('staticCorrectionV3MaxOffsetM').value = '';
  const result = ui.buildStaticCorrectionRequest();

  expect(result.errors).toEqual([]);
  expect(document.getElementById('staticCorrectionV3LayerFields').hidden).toBe(false);
  expect(document.getElementById('staticCorrectionVsubLayerFields').hidden).toBe(true);
  expect(result.payload.file_id).toBe('line-preserved');
  expect(result.payload.key1_byte).toBe(17);
  expect(result.payload.key2_byte).toBe(21);
  expect(result.payload.pick_source).toEqual({ kind: 'uploaded_npz' });
  expect(result.payload.model).toMatchObject({
    method: 'multilayer_time_term',
    layers: [
      {
        kind: 'v2_t1',
        enabled: true,
        min_offset_m: 300,
        max_offset_m: 4000,
        velocity_mode: 'solve_global',
        initial_velocity_m_s: 2400,
      },
      {
        kind: 'v3_t2',
        enabled: true,
        min_offset_m: 4000,
        max_offset_m: null,
        velocity_mode: 'solve_global',
        initial_velocity_m_s: 3600,
      },
    ],
  });
  expect(result.payload.conversion).toEqual({
    mode: 't1lsst_multilayer',
    layer_count: 2,
  });
});

test('three-layer preset builds public global Vsub/T3 request', () => {
  const ui = loadStaticCorrectionScript();

  selectModelPreset('three_layer_global');
  const result = ui.buildStaticCorrectionRequest();

  expect(result.errors).toEqual([]);
  expect(document.getElementById('staticCorrectionV3LayerFields').hidden).toBe(false);
  expect(document.getElementById('staticCorrectionVsubLayerFields').hidden).toBe(false);
  expect(result.payload.model.layers.map((layer) => layer.kind)).toEqual([
    'v2_t1',
    'v3_t2',
    'vsub_t3',
  ]);
  expect(result.payload.model.layers[1]).toMatchObject({
    kind: 'v3_t2',
    velocity_mode: 'solve_global',
    min_offset_m: 4000,
    max_offset_m: 6000,
  });
  expect(result.payload.model.layers[2]).toMatchObject({
    kind: 'vsub_t3',
    velocity_mode: 'solve_global',
    min_offset_m: 6000,
    max_offset_m: null,
    initial_velocity_m_s: 5000,
  });
  expect(result.payload.conversion).toEqual({
    mode: 't1lsst_multilayer',
    layer_count: 3,
  });
});

test('cell 2D preset builds line-projected solve-cell V2 request', () => {
  const ui = loadStaticCorrectionScript();

  selectModelPreset('cell_v2_t1_line_2d');
  document.getElementById('staticCorrectionCellCountY').value = '9';
  document.getElementById('staticCorrectionLineOriginXM').value = '1000';
  document.getElementById('staticCorrectionLineOriginYM').value = '2000';
  document.getElementById('staticCorrectionLineAzimuthDeg').value = '45';
  const result = ui.buildStaticCorrectionRequest();

  expect(result.errors).toEqual([]);
  expect(document.getElementById('staticCorrectionCellFields').hidden).toBe(false);
  expect(document.getElementById('staticCorrectionLine2DFields').hidden).toBe(false);
  expect(document.getElementById('staticCorrectionBedrockVelocityMode').value).toBe('solve_cell');
  expect(document.getElementById('staticCorrectionBedrockVelocityMode').disabled).toBe(true);
  expect(result.payload.model).toMatchObject({
    method: 'gli_variable_thickness',
    bedrock_velocity_mode: 'solve_cell',
    refractor_cell: {
      coordinate_mode: 'line_2d_projected',
      number_of_cell_y: 1,
      size_of_cell_y_m: null,
      line_origin_x_m: 1000,
      line_origin_y_m: 2000,
      line_azimuth_deg: 45,
      min_observations_per_cell: 5,
      velocity_smoothing_weight: 0,
    },
  });
  expect(result.payload.conversion).toEqual({ mode: 't1lsst_1layer' });
});

test('cell 3D preset builds grid solve-cell V2 request', () => {
  const ui = loadStaticCorrectionScript();

  selectModelPreset('cell_v2_t1_grid_3d');
  document.getElementById('staticCorrectionCellCountY').value = '4';
  document.getElementById('staticCorrectionCellSizeYM').value = '750';
  const result = ui.buildStaticCorrectionRequest();

  expect(result.errors).toEqual([]);
  expect(document.getElementById('staticCorrectionCellFields').hidden).toBe(false);
  expect(document.getElementById('staticCorrectionLine2DFields').hidden).toBe(true);
  expect(document.getElementById('staticCorrectionBedrockVelocityMode').value).toBe('solve_cell');
  expect(result.payload.model.refractor_cell).toMatchObject({
    coordinate_mode: 'grid_3d',
    number_of_cell_x: 20,
    number_of_cell_y: 4,
    size_of_cell_x_m: 500,
    size_of_cell_y_m: 750,
  });
  expect(result.payload.model.refractor_cell).not.toHaveProperty('line_origin_x_m');
  expect(result.payload.model.refractor_cell).not.toHaveProperty('line_origin_y_m');
  expect(result.payload.model.refractor_cell).not.toHaveProperty('line_azimuth_deg');
});

test('unsupported V3 and Vsub cell velocity modes are not exposed', () => {
  loadStaticCorrectionScript();

  selectModelPreset('three_layer_global');

  expect(document.querySelector('[name="model.layers.v3_t2.velocity_mode"]')).toBeNull();
  expect(document.querySelector('[name="model.layers.vsub_t3.velocity_mode"]')).toBeNull();
  expect(document.getElementById('staticCorrectionV3LayerFields').textContent).not.toContain(
    'solve_cell'
  );
  expect(document.getElementById('staticCorrectionVsubLayerFields').textContent).not.toContain(
    'solve_cell'
  );
});

test('unchecked linkage skips linkage build when running static correction', async () => {
  const ui = loadStaticCorrectionScript();
  const { calls } = createStaticCorrectionFetchMock();

  await ui.runStaticCorrection();
  await flushAsyncWork();

  expect(calls.map((call) => call.url)).toEqual([
    '/statics/refraction/apply-with-picks',
    '/statics/job/static-job-a/status',
    '/statics/job/static-job-a/files',
  ]);
  expect(calls[0].body.linkage).toEqual({ mode: 'none' });
  expect(calls.map((call) => call.url)).not.toContain('/statics/refraction/apply');
  expect(calls[0].formData).toBeInstanceOf(FormData);
  expect(calls[0].options.headers).toBeUndefined();
});

test('static correction multipart form data contains request JSON and pick NPZ', async () => {
  const ui = loadStaticCorrectionScript();
  const pickFile = selectedPickFile('uploaded-picks.npz', 'pick-data');
  setViewerTarget('active-line', 9, 13);
  const { calls } = createStaticCorrectionFetchMock();

  await ui.runStaticCorrection();
  await flushAsyncWork();

  const applyCall = calls.find((call) => call.url === '/statics/refraction/apply-with-picks');
  const request = JSON.parse(applyCall.formData.get('request_json'));
  expect(request).toMatchObject({
    file_id: 'active-line',
    key1_byte: 9,
    key2_byte: 13,
    pick_source: { kind: 'uploaded_npz' },
    linkage: { mode: 'none' },
  });
  expect(applyCall.formData.get('pick_npz')).toBe(pickFile);
});

test('static correction submit failure returns to a retryable idle state', async () => {
  const ui = loadStaticCorrectionScript();
  const calls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    calls.push({ url: String(url), options });
    if (url === '/statics/refraction/apply-with-picks') {
      return jsonResponse({ detail: 'apply failed' }, 400);
    }
    throw new Error(`unexpected fetch ${url}`);
  }));

  await ui.runStaticCorrection();
  await flushAsyncWork();

  expect(calls.map((call) => call.url)).toEqual(['/statics/refraction/apply-with-picks']);
  expect(window.refractionStaticRunState.phase).toBe('idle');
  expect(document.getElementById('staticCorrectionRunButton').disabled).toBe(false);
  expect(document.getElementById('staticCorrectionError').textContent).toContain('apply failed');
  expect(document.getElementById('staticCorrectionStatus').textContent).toContain(
    'Static correction submission failed'
  );
});

test('static correction submit displays backend validation error', async () => {
  const ui = loadStaticCorrectionScript();
  vi.stubGlobal('fetch', vi.fn(async (url) => {
    if (url === '/statics/refraction/apply-with-picks') {
      return jsonResponse({ detail: 'request_json.file_id is invalid' }, 422);
    }
    throw new Error(`unexpected fetch ${url}`);
  }));

  await ui.runStaticCorrection();
  await flushAsyncWork();

  expect(document.getElementById('staticCorrectionError').hidden).toBe(false);
  expect(document.getElementById('staticCorrectionError').textContent).toContain(
    'request_json.file_id is invalid'
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
    '/statics/refraction/apply-with-picks',
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
  expect(calls.map((call) => call.url)).toContain('/statics/refraction/apply-with-picks');
});

test('failed linkage prevents refraction apply submit', async () => {
  const ui = loadStaticCorrectionScript();
  enableLinkage();
  const { calls } = createStaticCorrectionFetchMock([
    { state: 'error', message: 'linkage build failed', progress: 1 },
  ]);

  await ui.runStaticCorrection();
  await flushAsyncWork();

  expect(calls.map((call) => call.url)).not.toContain('/statics/refraction/apply-with-picks');
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

  const applyCall = calls.find((call) => call.url === '/statics/refraction/apply-with-picks');
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
