import { readFileSync } from 'node:fs';

import { beforeEach, expect, test } from 'vitest';

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
