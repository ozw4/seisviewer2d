import { createOverlayTransform } from './overlay_transform.js';

const ROOT_CLASS = 'sv-viewer-overlay-root';
const CANVAS_CLASS = 'sv-viewer-overlay-canvas';
const MANUAL_PICK_CLASS = 'sv-viewer-manual-pick-overlay';
const PREDICTION_CLASS = 'sv-viewer-prediction-overlay';
const MIN_DEVICE_PIXEL_RATIO = 1;

let overlayState = null;
let syncRaf = 0;
const redrawCallbacks = new Set();

function finiteNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function cssPx(value) {
  const n = finiteNumber(value);
  return `${Math.max(0, n ?? 0)}px`;
}

function overlayAxisSuffix(index) {
  return index === 0 ? '' : String(index + 1);
}

function overlayAxisLayoutName(base, index) {
  return `${base}axis${overlayAxisSuffix(index)}`;
}

function normalizeDevicePixelRatio(value = window.devicePixelRatio) {
  const dpr = finiteNumber(value);
  return dpr && dpr > 0 ? dpr : MIN_DEVICE_PIXEL_RATIO;
}

function createCanvas(className) {
  const canvas = document.createElement('canvas');
  canvas.className = `${CANVAS_CLASS} ${className}`;
  canvas.setAttribute('aria-hidden', 'true');
  canvas.tabIndex = -1;
  return canvas;
}

function createOverlayRoot(plotDiv) {
  const root = document.createElement('div');
  root.className = ROOT_CLASS;
  root.setAttribute('aria-hidden', 'true');
  root.dataset.svRole = 'viewer-overlay-root';

  const manualPickCanvas = createCanvas(MANUAL_PICK_CLASS);
  const predictionCanvas = createCanvas(PREDICTION_CLASS);
  root.append(manualPickCanvas, predictionCanvas);
  plotDiv.appendChild(root);

  return { root, manualPickCanvas, predictionCanvas };
}

function ensureOverlayState(plotDiv = document.getElementById('plot')) {
  if (!plotDiv) return null;

  let root = plotDiv.querySelector(`:scope > .${ROOT_CLASS}`);
  let manualPickCanvas = root?.querySelector(`.${MANUAL_PICK_CLASS}`) || null;
  let predictionCanvas = root?.querySelector(`.${PREDICTION_CLASS}`) || null;

  if (!root || !manualPickCanvas || !predictionCanvas) {
    if (root) root.remove();
    ({ root, manualPickCanvas, predictionCanvas } = createOverlayRoot(plotDiv));
  }

  if (!overlayState || overlayState.plotDiv !== plotDiv || overlayState.root !== root) {
    overlayState = {
      plotDiv,
      root,
      manualPickCanvas,
      predictionCanvas,
      width: 0,
      height: 0,
      dpr: MIN_DEVICE_PIXEL_RATIO,
      transform: null,
      lastReason: 'init',
      resizeObserver: null,
    };
    installResizeObserver(overlayState);
  } else {
    overlayState.manualPickCanvas = manualPickCanvas;
    overlayState.predictionCanvas = predictionCanvas;
  }

  return overlayState;
}

function installResizeObserver(state) {
  if (state.resizeObserver || typeof ResizeObserver !== 'function') return;
  state.resizeObserver = new ResizeObserver(() => {
    scheduleViewerOverlaySync('resize-observer');
  });
  state.resizeObserver.observe(state.plotDiv);
}

function plotAreaFromLayout(plotDiv, panelIndex = 0) {
  const layout = plotDiv?._fullLayout;
  const size = layout?._size;
  if (!layout || !size) return null;

  const xAxisName = overlayAxisLayoutName('x', panelIndex);
  const yAxisName = overlayAxisLayoutName('y', panelIndex);
  const xa = layout[xAxisName];
  const ya = layout[yAxisName];
  if (!xa || !ya || !Array.isArray(xa.range) || !Array.isArray(ya.range)) return null;

  const xDomain = Array.isArray(xa.domain) && xa.domain.length === 2 ? xa.domain : [0, 1];
  const yDomain = Array.isArray(ya.domain) && ya.domain.length === 2 ? ya.domain : [0, 1];
  const w = finiteNumber(size.w) ?? 0;
  const h = finiteNumber(size.h) ?? 0;

  return {
    plotArea: {
      left: (finiteNumber(size.l) ?? 0) + (finiteNumber(xDomain[0]) ?? 0) * w,
      top: (finiteNumber(size.t) ?? 0) + (1 - (finiteNumber(yDomain[1]) ?? 1)) * h,
      width: Math.max(0, ((finiteNumber(xDomain[1]) ?? 1) - (finiteNumber(xDomain[0]) ?? 0)) * w),
      height: Math.max(0, ((finiteNumber(yDomain[1]) ?? 1) - (finiteNumber(yDomain[0]) ?? 0)) * h),
    },
    xRange: [xa.range[0], xa.range[1]],
    yRange: [ya.range[0], ya.range[1]],
    xAxisName,
    yAxisName,
    panelIndex,
  };
}

function buildTransformInput(plotDiv) {
  if (typeof window.buildOverlayTransformInputFromPlot === 'function') {
    const input = window.buildOverlayTransformInputFromPlot(plotDiv);
    if (input) return input;
  }

  const rect = plotDiv?.getBoundingClientRect?.();
  if (!rect) return null;
  const area = plotAreaFromLayout(plotDiv);
  if (!area) return null;

  return {
    containerRect: {
      left: finiteNumber(rect.left) ?? 0,
      top: finiteNumber(rect.top) ?? 0,
      width: finiteNumber(rect.width) ?? 0,
      height: finiteNumber(rect.height) ?? 0,
    },
    plotArea: area.plotArea,
    xRange: area.xRange,
    yRange: area.yRange,
    transpose: false,
    panelIndex: area.panelIndex,
    xAxisName: area.xAxisName,
    yAxisName: area.yAxisName,
  };
}

function resizeCanvas(canvas, width, height, dpr) {
  const bitmapWidth = Math.max(0, Math.round(width * dpr));
  const bitmapHeight = Math.max(0, Math.round(height * dpr));
  const changed = canvas.width !== bitmapWidth || canvas.height !== bitmapHeight;

  if (changed) {
    canvas.width = bitmapWidth;
    canvas.height = bitmapHeight;
  }
  canvas.style.width = cssPx(width);
  canvas.style.height = cssPx(height);

  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
  }
  return changed;
}

export function syncViewerOverlayLayer(reason = 'sync') {
  const state = ensureOverlayState();
  if (!state) return null;

  const rect = state.plotDiv.getBoundingClientRect();
  const width = finiteNumber(rect.width) ?? 0;
  const height = finiteNumber(rect.height) ?? 0;
  const dpr = normalizeDevicePixelRatio();

  state.root.style.width = cssPx(width);
  state.root.style.height = cssPx(height);
  const manualChanged = resizeCanvas(state.manualPickCanvas, width, height, dpr);
  const predictionChanged = resizeCanvas(state.predictionCanvas, width, height, dpr);

  const input = buildTransformInput(state.plotDiv);
  state.transform = input ? createOverlayTransform(input) : null;
  state.width = width;
  state.height = height;
  state.dpr = dpr;
  state.lastReason = reason;

  const payload = {
    reason,
    state,
    manualPickCanvas: state.manualPickCanvas,
    predictionCanvas: state.predictionCanvas,
    transform: state.transform,
    sizeChanged: manualChanged || predictionChanged,
  };
  for (const callback of redrawCallbacks) {
    try {
      callback(payload);
    } catch (err) {
      console.warn('[viewer overlay] redraw callback failed', err);
    }
  }
  window.viewerPerfMetrics?.recordOverlayRender?.({ reason });
  return state;
}

export function scheduleViewerOverlaySync(reason = 'scheduled') {
  if (syncRaf !== 0) return;
  syncRaf = requestAnimationFrame(() => {
    syncRaf = 0;
    syncViewerOverlayLayer(reason);
  });
}

export function onViewerOverlayRedraw(callback) {
  if (typeof callback !== 'function') return () => {};
  redrawCallbacks.add(callback);
  return () => redrawCallbacks.delete(callback);
}

export function initViewerOverlayLayer(options = {}) {
  const plotDiv = options.plotDiv || document.getElementById('plot');
  const state = ensureOverlayState(plotDiv);
  if (typeof options.onRedraw === 'function') {
    onViewerOverlayRedraw(options.onRedraw);
  }
  if (state) scheduleViewerOverlaySync('init');
  return state;
}

window.ViewerOverlayLayer = {
  init: initViewerOverlayLayer,
  sync: syncViewerOverlayLayer,
  scheduleSync: scheduleViewerOverlaySync,
  onRedraw: onViewerOverlayRedraw,
  getState: () => overlayState,
};
window.initViewerOverlayLayer = initViewerOverlayLayer;
window.syncViewerOverlayLayer = syncViewerOverlayLayer;
window.scheduleViewerOverlaySync = scheduleViewerOverlaySync;
window.onViewerOverlayRedraw = onViewerOverlayRedraw;

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => initViewerOverlayLayer(), { once: true });
} else {
  initViewerOverlayLayer();
}
