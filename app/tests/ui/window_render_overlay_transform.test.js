import { afterEach, beforeEach, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const WINDOW_RENDER_SCRIPT = resolve(process.cwd(), 'static/viewer/window_render.js');

function installWindowRenderScript() {
  (0, eval)(`
    var defaultDt = 0.004;
    var latestWindowRender = null;
    var renderedStart = 10;
    var renderedEnd = 20;
  `);
  (0, eval)(readFileSync(WINDOW_RENDER_SCRIPT, 'utf8'));
}

function makePlotDiv() {
  const plotDiv = document.createElement('div');
  plotDiv._fullLayout = {
    _size: { l: 10, t: 20, w: 200, h: 100 },
    xaxis: { range: [0, 2], domain: [0, 1] },
    yaxis: { range: [0, 100], domain: [0, 1] },
  };
  plotDiv.getBoundingClientRect = () => ({
    left: 5,
    top: 7,
    width: 300,
    height: 200,
  });
  document.body.appendChild(plotDiv);
  return plotDiv;
}

beforeEach(() => {
  installWindowRenderScript();
});

afterEach(() => {
  document.body.innerHTML = '';
  delete window.ViewerOverlayTransform;
  delete window.buildOverlayTransformInputFromPlot;
  delete window.createOverlayTransformForPlot;
  delete window.viewerTraceTimeTranspose;
  delete window.viewerOverlayTranspose;
});

test('overlay transform input uses plot/render transpose metadata by default', () => {
  const plotDiv = makePlotDiv();
  plotDiv.__svTraceTimeTranspose = true;

  const input = window.buildOverlayTransformInputFromPlot(plotDiv);

  expect(input).toMatchObject({
    transpose: true,
    xRange: [0, 2],
    yRange: [0, 100],
    renderedStart: 10,
    renderedEnd: 20,
    renderedTimeStart: undefined,
    renderedTimeEnd: undefined,
  });
});

test('explicit overlay transpose option overrides plot metadata', () => {
  const plotDiv = makePlotDiv();
  plotDiv.__svTraceTimeTranspose = true;

  expect(window.buildOverlayTransformInputFromPlot(plotDiv, { transpose: false }).transpose).toBe(false);
  expect(window.buildOverlayTransformInputFromPlot(plotDiv, { transpose: '1' }).transpose).toBe(true);
});

test('createOverlayTransformForPlot passes the resolved transpose state', () => {
  const plotDiv = makePlotDiv();
  plotDiv.dataset.svOverlayTranspose = 'true';
  window.ViewerOverlayTransform = {
    createOverlayTransform: (input) => input,
  };

  const input = window.createOverlayTransformForPlot(plotDiv);

  expect(input.transpose).toBe(true);
});
