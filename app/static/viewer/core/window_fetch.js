import { renderWindowHeatmap, renderWindowWiggle, renderLatestView } from './render.js';

const WINDOW_MAX_POINTS = 1_200_000;
const WIGGLE_MAX_POINTS = 2_500_000;

function roundUpPowerOfTwo(value) {
  let v = Math.max(1, Math.floor(value));
  v -= 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

export function computeStepsForWindow({
  tracesVisible,
  samplesVisible,
  widthPx,
  heightPx,
  oversampleX = 1.2,
  oversampleY = 1.2,
  maxPoints = WINDOW_MAX_POINTS,
}) {
  const ratio = window.devicePixelRatio || 1;
  const effW = Math.max(1, Math.round(widthPx * ratio));
  const effH = Math.max(1, Math.round(heightPx * ratio));
  let stepX = Math.max(1, Math.ceil(tracesVisible / (effW * oversampleX)));
  let stepY = Math.max(1, Math.ceil(samplesVisible / (effH * oversampleY)));

  const tracesOut = () => Math.ceil(tracesVisible / stepX);
  const samplesOut = () => Math.ceil(samplesVisible / stepY);

  let guard = 0;
  while (tracesOut() * samplesOut() > maxPoints && guard < 512) {
    if (tracesOut() / effW > samplesOut() / effH) {
      stepX += 1;
    } else {
      stepY += 1;
    }
    guard += 1;
  }

  stepX = roundUpPowerOfTwo(stepX);
  stepY = roundUpPowerOfTwo(stepY);

  guard = 0;
  while (Math.ceil(tracesVisible / stepX) * Math.ceil(samplesVisible / stepY) > maxPoints && guard < 512) {
    if (tracesVisible / stepX > samplesVisible / stepY) {
      stepX = roundUpPowerOfTwo(stepX + 1);
    } else {
      stepY = roundUpPowerOfTwo(stepY + 1);
    }
    guard += 1;
  }

  return { step_x: stepX, step_y: stepY };
}

export function wantWiggleForWindow({ tracesVisible, samplesVisible, widthPx }) {
  const density = tracesVisible / Math.max(1, widthPx);
  const threshold = typeof window.WIGGLE_DENSITY_THRESHOLD === 'number' ? window.WIGGLE_DENSITY_THRESHOLD : 0.2;
  if (density >= threshold) return false;
  if ((tracesVisible * samplesVisible) > WIGGLE_MAX_POINTS) return false;
  return true;
}

export function currentVisibleWindow() {
  const shape = window.sectionShape;
  if (!shape || shape.length !== 2) return null;
  const [totalTraces, totalSamples] = shape;

  const forceFull = !!window.forceFullExtentOnce;
  let x0;
  let x1;
  if (forceFull) {
    x0 = 0;
    x1 = totalTraces - 1;
  } else if (Array.isArray(window.savedXRange) && window.savedXRange.length === 2) {
    const minX = Math.min(window.savedXRange[0], window.savedXRange[1]);
    const maxX = Math.max(window.savedXRange[0], window.savedXRange[1]);
    x0 = Math.floor(minX);
    x1 = Math.ceil(maxX);
  } else if (typeof window.renderedStart === 'number' && typeof window.renderedEnd === 'number') {
    x0 = window.renderedStart;
    x1 = window.renderedEnd;
  } else {
    x0 = 0;
    x1 = totalTraces - 1;
  }

  x0 = Math.max(0, Math.floor(x0));
  x1 = Math.min(totalTraces - 1, Math.ceil(x1));
  if (x1 < x0) [x0, x1] = [x1, x0];

  const spanX = Math.max(1, x1 - x0 + 1);
  const padX = (!forceFull && Array.isArray(window.savedXRange))
    ? Math.max(1, Math.floor(spanX * 0.5))
    : 0;
  x0 = Math.max(0, x0 - padX);
  x1 = Math.min(totalTraces - 1, x1 + padX);

  const dtBase = window.defaultDt ?? 0.002;
  let yMinSec;
  let yMaxSec;
  if (!forceFull && Array.isArray(window.savedYRange) && window.savedYRange.length === 2) {
    yMinSec = Math.min(window.savedYRange[0], window.savedYRange[1]);
    yMaxSec = Math.max(window.savedYRange[0], window.savedYRange[1]);
  } else {
    yMinSec = 0;
    yMaxSec = (totalSamples - 1) * dtBase;
  }

  let y0 = Math.floor(yMinSec / dtBase);
  let y1 = Math.ceil(yMaxSec / dtBase);
  y0 = Math.max(0, y0);
  y1 = Math.min(totalSamples - 1, y1);
  if (y1 < y0) [y0, y1] = [y1, y0];

  const spanY = Math.max(1, y1 - y0 + 1);
  const padY = (!forceFull && Array.isArray(window.savedYRange))
    ? Math.max(1, Math.floor(spanY * 0.1))
    : 0;
  y0 = Math.max(0, y0 - padY);
  y1 = Math.min(totalSamples - 1, y1 + padY);

  if (forceFull) {
    window.forceFullExtentOnce = false;
  }

  return {
    x0,
    x1,
    y0,
    y1,
    nTraces: x1 - x0 + 1,
    nSamples: y1 - y0 + 1,
  };
}

function decodeWindowPayload(obj) {
  const shapeRaw = Array.isArray(obj.shape) ? obj.shape : Array.from(obj.shape ?? []);
  if (shapeRaw.length !== 2) {
    console.warn('Unexpected window shape', obj.shape);
    return null;
  }
  const rows = Number(shapeRaw[0]);
  const cols = Number(shapeRaw[1]);
  if (!rows || !cols) return null;

  const buffer = obj.data?.buffer instanceof ArrayBuffer ? obj.data.buffer : null;
  if (!buffer) return null;
  const int8 = new Int8Array(buffer);
  const values = Float32Array.from(int8, (v) => v / obj.scale);
  return { rows, cols, values };
}

export async function fetchWindowAndPlot() {
  const fileId = window.currentFileId;
  const shape = window.sectionShape;
  if (!fileId || !shape) return;

  const slider = document.getElementById('key1_idx_slider');
  if (!slider) return;
  const idx = Number.parseInt(slider.value, 10);
  const key1Values = Array.isArray(window.key1Values) ? window.key1Values : [];
  const key1Val = key1Values[idx];
  if (key1Val === undefined) return;

  const windowInfo = currentVisibleWindow();
  if (!windowInfo) return;

  const plotDiv = document.getElementById('plot');
  if (!plotDiv) return;

  const widthPx = plotDiv.clientWidth || plotDiv.offsetWidth || 1;
  const heightPx = plotDiv.clientHeight || plotDiv.offsetHeight || 1;
  const sel = document.getElementById('layerSelect');
  const requestedLayer = sel ? sel.value : 'raw';
  const isFbLayer = requestedLayer === 'fbprob';
  const wantWiggle = !isFbLayer && wantWiggleForWindow({
    tracesVisible: windowInfo.nTraces,
    samplesVisible: windowInfo.nSamples,
    widthPx,
  });

  let step_x;
  let step_y;
  if (wantWiggle) {
    step_x = 1;
    step_y = 1;
  } else {
    ({ step_x, step_y } = computeStepsForWindow({
      tracesVisible: windowInfo.nTraces,
      samplesVisible: windowInfo.nSamples,
      widthPx,
      heightPx,
    }));
  }

  const pipelineKeyNow = window.latestPipelineKey || null;
  const mode = wantWiggle ? 'wiggle' : 'heatmap';

  let effectiveLayer = requestedLayer;
  let tapLabel = null;
  if (requestedLayer !== 'raw') {
    if (pipelineKeyNow) {
      tapLabel = requestedLayer;
    } else {
      effectiveLayer = 'raw';
    }
  } else {
    effectiveLayer = 'raw';
  }

  if (!wantWiggle && effectiveLayer === 'raw' && window.latestSeismicData && step_x === 1 && step_y === 1) {
    return;
  }
  if (!wantWiggle && tapLabel && window.latestTapData?.[requestedLayer] && step_x === 1 && step_y === 1) {
    window.latestSeismicData = window.latestTapData[requestedLayer];
    renderLatestView(windowInfo.x0, windowInfo.x1);
    return;
  }

  const params = new URLSearchParams({
    file_id: fileId,
    key1_idx: String(key1Val),
    key1_byte: String(window.currentKey1Byte),
    key2_byte: String(window.currentKey2Byte),
    x0: String(windowInfo.x0),
    x1: String(windowInfo.x1),
    y0: String(windowInfo.y0),
    y1: String(windowInfo.y1),
    step_x: String(step_x),
    step_y: String(step_y),
  });
  if (tapLabel && pipelineKeyNow) {
    params.set('pipeline_key', pipelineKeyNow);
    params.set('tap_label', tapLabel);
  }

  const requestId = (window.windowFetchToken = (window.windowFetchToken || 0) + 1);

  const prevCtrl = window.windowFetchCtrl;
  if (prevCtrl) {
    try { prevCtrl.abort(); } catch (_) { /* noop */ }
  }
  const ctrl = new AbortController();
  window.windowFetchCtrl = ctrl;

  try {
    const res = await fetch(`/get_section_window_bin?${params.toString()}`, { signal: ctrl.signal });
    if (!res.ok) {
      console.warn('Window fetch failed', res.status);
      return;
    }
    const bin = new Uint8Array(await res.arrayBuffer());
    if (requestId !== window.windowFetchToken) return;
    const obj = window.msgpack.decode(bin);
    if (typeof window.applyServerDt === 'function') {
      window.applyServerDt(obj);
    }
    const decoded = decodeWindowPayload(obj);
    if (!decoded) return;
    const { rows, cols, values } = decoded;

    const windowPayload = {
      key1: key1Val,
      requestedLayer,
      effectiveLayer,
      pipelineKey: tapLabel ? pipelineKeyNow : null,
      x0: windowInfo.x0,
      x1: windowInfo.x1,
      y0: windowInfo.y0,
      y1: windowInfo.y1,
      stepX: step_x,
      stepY: step_y,
      shape: [rows, cols],
      values,
      mode,
    };

    window.latestSeismicData = null;
    window.latestWindowRender = windowPayload;
    if (window.isRelayouting) {
      window.redrawPending = true;
      return;
    }
    if (mode === 'wiggle') {
      renderWindowWiggle(windowPayload);
    } else {
      renderWindowHeatmap(windowPayload);
    }
  } catch (err) {
    if (err && err.name === 'AbortError') {
      return;
    }
    if (requestId === window.windowFetchToken) {
      console.warn('Window fetch error', err);
    }
  } finally {
    if (window.windowFetchCtrl === ctrl) {
      window.windowFetchCtrl = null;
    }
  }
}

function debounce(fn, wait) {
  let t = null;
  return function debounced(...args) {
    if (t) clearTimeout(t);
    t = setTimeout(() => {
      t = null;
      fn.apply(this, args);
    }, wait);
  };
}

export const scheduleWindowFetch = debounce(() => {
  fetchWindowAndPlot().catch((err) => console.warn('Window fetch failed', err));
}, 120);
