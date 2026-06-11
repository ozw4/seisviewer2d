import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import {
  initViewerOverlayLayer,
  syncViewerOverlayLayer,
} from '../../static/viewer/overlay_layer.js';

function makePlotDiv({ width = 320, height = 180 } = {}) {
  document.body.innerHTML = '<div id="plot"></div>';
  const plotDiv = document.getElementById('plot');
  plotDiv.getBoundingClientRect = () => ({
    left: 8,
    top: 12,
    width,
    height,
  });
  plotDiv._fullLayout = {
    _size: { l: 24, t: 18, w: width - 40, h: height - 36 },
    xaxis: { range: [10, 110], domain: [0, 1] },
    yaxis: { range: [1.2, 0], domain: [0, 1] },
  };
  return plotDiv;
}

beforeEach(() => {
  vi.stubGlobal('requestAnimationFrame', (callback) => {
    callback();
    return 1;
  });
  Object.defineProperty(window, 'devicePixelRatio', {
    configurable: true,
    value: 2,
  });
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    setTransform: vi.fn(),
    clearRect: vi.fn(),
  });
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  delete window.buildOverlayTransformInputFromPlot;
});

describe('viewer overlay layer', () => {
  test('creates root with separate manual and prediction canvases', () => {
    const plotDiv = makePlotDiv();

    const state = initViewerOverlayLayer({ plotDiv });

    expect(state.root).toBe(plotDiv.querySelector(':scope > .sv-viewer-overlay-root'));
    expect(state.manualPickCanvas).toBe(state.root.querySelector('.sv-viewer-manual-pick-overlay'));
    expect(state.predictionCanvas).toBe(state.root.querySelector('.sv-viewer-prediction-overlay'));
    expect(state.manualPickCanvas).not.toBe(state.predictionCanvas);
  });

  test('sizes canvas bitmaps with devicePixelRatio while keeping CSS size in plot pixels', () => {
    makePlotDiv({ width: 333.5, height: 201.25 });

    const state = syncViewerOverlayLayer('test-size');

    expect(state.root.style.width).toBe('333.5px');
    expect(state.root.style.height).toBe('201.25px');
    expect(state.manualPickCanvas.width).toBe(667);
    expect(state.manualPickCanvas.height).toBe(403);
    expect(state.predictionCanvas.width).toBe(667);
    expect(state.predictionCanvas.height).toBe(403);
    expect(state.manualPickCanvas.style.width).toBe('333.5px');
    expect(state.manualPickCanvas.style.height).toBe('201.25px');
    expect(state.transform.valid).toBe(true);
  });
});
