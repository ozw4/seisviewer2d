import { setGrid } from './grid.js';
import { buildLayout, buildPickShapes } from './layout.js';

const AMP_LIMIT = 3.0;
const MAX_POINTS = 3_000_000;

function getPlotDiv() {
  return document.getElementById('plot');
}

function getPlotly() {
  return window.Plotly;
}

function withSuppressed(fn) {
  return typeof window.withSuppressedRelayout === 'function'
    ? window.withSuppressedRelayout(fn)
    : fn;
}

function scheduleResize(plotDiv) {
  const Plotly = getPlotly();
  if (!Plotly || !plotDiv) return;
  setTimeout(() => {
    try {
      withSuppressed(Plotly.Plots.resize(plotDiv));
    } catch (err) {
      console.warn('Plotly resize failed', err);
    }
  }, 50);
}

export function snapshotAxesRangesFromDOM() {
  const gd = getPlotDiv();
  const xa = gd?._fullLayout?.xaxis;
  const ya = gd?._fullLayout?.yaxis;
  if (xa && Array.isArray(xa.range) && xa.range.length === 2) {
    window.savedXRange = [xa.range[0], xa.range[1]];
  }
  if (ya && Array.isArray(ya.range) && ya.range.length === 2) {
    const [y0, y1] = ya.range;
    window.savedYRange = y0 > y1 ? [y0, y1] : [y1, y0];
  }
}

export function currentUiRevision() {
  const sel = document.getElementById('layerSelect');
  const layer = sel ? sel.value : 'raw';
  const slider = document.getElementById('key1_idx_slider');
  const idx = slider ? Number.parseInt(slider.value, 10) || 0 : 0;
  const key1Val = Array.isArray(window.key1Values) ? window.key1Values[idx] : undefined;
  const pipelineKey = window.latestPipelineKey || '';
  const nonce = typeof window.uiResetNonce === 'number' ? window.uiResetNonce : 0;
  const fileId = window.currentFileId || '';
  return `rev:${fileId}|${key1Val}|${layer}|${pipelineKey}|${nonce}`;
}

export function clickModeForCurrentState() {
  return window.isPickMode ? 'event' : 'event+select';
}

function applyPlotly(plotDiv, traces, layout) {
  const Plotly = getPlotly();
  if (!Plotly || !plotDiv) return;
  try {
    withSuppressed(Plotly.react(plotDiv, traces, layout, {
      responsive: true,
      editable: true,
      modeBarButtonsToAdd: ['eraseshape'],
      edits: { shapePosition: false },
    }));
  } catch (err) {
    console.error('Plotly.react failed', err);
    throw err;
  }
  scheduleResize(plotDiv);
  window.requestAnimationFrame(() => {
    if (typeof window.applyDragMode === 'function') {
      window.applyDragMode();
    }
  });
  if (typeof window.installPlotlyViewportHandlersOnce === 'function') {
    window.installPlotlyViewportHandlersOnce();
  }
  if (typeof window.attachPickListeners === 'function') {
    window.attachPickListeners(plotDiv);
  }
}

function buildShapes({ xMin, xMax }) {
  const manual = Array.isArray(window.picks) ? window.picks : [];
  const predicted = Array.isArray(window.predictedPicks) ? window.predictedPicks : [];
  const showPred = !!document.getElementById('showFbPred')?.checked;
  return buildPickShapes({
    manualPicks: manual,
    predicted: showPred ? predicted : [],
    xMin,
    xMax,
    showPredicted: showPred,
  });
}

function getGain() {
  const gainEl = document.getElementById('gain');
  const gain = gainEl ? Number.parseFloat(gainEl.value) : Number.NaN;
  return Number.isFinite(gain) ? gain : 1.0;
}

function ensureArrayBuffer(values) {
  if (values instanceof Float32Array) return values;
  if (ArrayBuffer.isView(values)) {
    return new Float32Array(values.buffer, values.byteOffset, values.length);
  }
  return Float32Array.from(values || []);
}

export function renderWindowWiggle(windowData) {
  if (window.isRelayouting) {
    window.latestWindowRender = windowData;
    window.redrawPending = true;
    return;
  }
  snapshotAxesRangesFromDOM();
  if (!windowData || (windowData.mode && windowData.mode !== 'wiggle')) return;

  const sel = document.getElementById('layerSelect');
  const currentLayer = sel ? sel.value : 'raw';
  if (windowData.requestedLayer !== currentLayer) return;

  const slider = document.getElementById('key1_idx_slider');
  const idx = slider ? Number.parseInt(slider.value, 10) || 0 : 0;
  const key1Val = Array.isArray(window.key1Values) ? window.key1Values[idx] : undefined;
  if (windowData.key1 !== key1Val) return;

  if (windowData.pipelineKey && (window.latestPipelineKey || null) !== (windowData.pipelineKey || null)) {
    return;
  }

  if (windowData.effectiveLayer === 'fbprob') return;

  const plotDiv = getPlotDiv();
  if (!plotDiv) return;

  const { values: rawValues, shape, x0, x1, y0, y1, stepX = 1, stepY = 1 } = windowData;
  const rows = Number(shape?.[0] ?? 0);
  const cols = Number(shape?.[1] ?? 0);
  if (!rows || !cols) return;

  const values = ensureArrayBuffer(rawValues);
  if (values.length !== rows * cols) return;

  setGrid({ x0, stepX: 1, y0, stepY: 1 });
  const dt = window.defaultDt ?? 0.002;
  const time = new Float32Array(rows);
  for (let r = 0; r < rows; r += 1) {
    time[r] = (y0 + r * stepY) * dt;
  }

  const traces = [];
  const gain = getGain();

  for (let c = 0; c < cols; c += 1) {
    const baseX = new Float32Array(rows);
    const shiftedFullX = new Float32Array(rows);
    const shiftedPosX = new Float32Array(rows);
    const traceIndex = x0 + c * stepX;
    for (let r = 0; r < rows; r += 1) {
      const idxVal = r * cols + c;
      let val = values[idxVal] * gain;
      if (val > AMP_LIMIT) val = AMP_LIMIT;
      if (val < -AMP_LIMIT) val = -AMP_LIMIT;

      baseX[r] = traceIndex;
      shiftedFullX[r] = traceIndex + val;
      shiftedPosX[r] = traceIndex + (val < 0 ? 0 : val);
    }

    traces.push({ type: 'scatter', mode: 'lines', x: baseX, y: time, line: { width: 0 }, hoverinfo: 'skip', showlegend: false });
    traces.push({ type: 'scatter', mode: 'lines', x: shiftedPosX, y: time, fill: 'tonextx', fillcolor: 'black', line: { width: 0 }, opacity: 0.6, hoverinfo: 'skip', showlegend: false });
    traces.push({ type: 'scatter', mode: 'lines', x: shiftedFullX, y: time, line: { color: 'black', width: 0.5 }, hoverinfo: 'x+y', showlegend: false });
  }

  window.downsampleFactor = 1;
  const endTrace = typeof x1 === 'number' ? x1 : x0 + cols - 1;
  window.renderedStart = x0;
  window.renderedEnd = endTrace;

  const totalSamples = Array.isArray(window.sectionShape) ? window.sectionShape[1] : rows;
  const layout = buildLayout({
    mode: 'wiggle',
    x0,
    x1: endTrace,
    y0,
    y1,
    stepX: 1,
    stepY: 1,
    totalSamples,
    dt: window.defaultDt ?? 0.002,
    savedXRange: window.savedXRange,
    savedYRange: window.savedYRange,
    clickmode: clickModeForCurrentState(),
    dragmode: typeof window.effectiveDragMode === 'function' ? window.effectiveDragMode() : 'zoom',
    uirevision: currentUiRevision(),
    fbTitle: null,
  });
  layout.shapes = buildShapes({ xMin: x0, xMax: endTrace });

  applyPlotly(plotDiv, traces, layout);
}

export function renderWindowHeatmap(windowData) {
  if (window.isRelayouting) {
    window.latestWindowRender = windowData;
    window.redrawPending = true;
    return;
  }
  snapshotAxesRangesFromDOM();
  if (!windowData || (windowData.mode && windowData.mode !== 'heatmap')) return;

  const sel = document.getElementById('layerSelect');
  const currentLayer = sel ? sel.value : 'raw';
  if (windowData.requestedLayer !== currentLayer) return;

  const slider = document.getElementById('key1_idx_slider');
  const idx = slider ? Number.parseInt(slider.value, 10) || 0 : 0;
  const key1Val = Array.isArray(window.key1Values) ? window.key1Values[idx] : undefined;
  if (windowData.key1 !== key1Val) return;

  if (windowData.pipelineKey && (window.latestPipelineKey || null) !== (windowData.pipelineKey || null)) {
    return;
  }

  const plotDiv = getPlotDiv();
  if (!plotDiv) return;

  const { values: rawValues, shape, x0, x1, y0, y1, stepX, stepY, effectiveLayer } = windowData;
  const rows = Number(shape?.[0] ?? 0);
  const cols = Number(shape?.[1] ?? 0);
  if (!rows || !cols) return;

  const values = ensureArrayBuffer(rawValues);
  if (values.length !== rows * cols) return;

  setGrid({ x0, stepX, y0, stepY });
  const gain = getGain();
  const fbMode = effectiveLayer === 'fbprob';
  const zData = new Array(rows);
  for (let r = 0; r < rows; r += 1) {
    const row = new Float32Array(cols);
    const offset = r * cols;
    for (let c = 0; c < cols; c += 1) {
      let val = values[offset + c];
      if (fbMode) {
        row[c] = val * 255;
      } else {
        val *= gain;
        if (val > AMP_LIMIT) val = AMP_LIMIT;
        else if (val < -AMP_LIMIT) val = -AMP_LIMIT;
        row[c] = val;
      }
    }
    zData[r] = row;
  }

  const xVals = new Float32Array(cols);
  for (let c = 0; c < cols; c += 1) xVals[c] = x0 + c * stepX;

  const baseDt = window.defaultDt ?? 0.002;
  const yVals = new Float32Array(rows);
  for (let r = 0; r < rows; r += 1) yVals[r] = (y0 + r * stepY) * baseDt;

  window.downsampleFactor = stepY || 1;
  window.renderedStart = x0;
  window.renderedEnd = x1;

  const cmName = document.getElementById('colormap')?.value || 'Greys';
  const reverse = !!document.getElementById('cmReverse')?.checked;
  const cmMap = window.COLORMAPS || {};
  const cm = cmMap[cmName] || 'Greys';
  const isDiv = cmName === 'RdBu' || cmName === 'BWR';
  const zMin = fbMode ? 0 : -AMP_LIMIT;
  const zMax = fbMode ? 255 : AMP_LIMIT;

  const traces = [{
    type: 'heatmap',
    x: xVals,
    y: yVals,
    z: zData,
    colorscale: cm,
    reversescale: reverse,
    zmin: zMin,
    zmax: zMax,
    ...(fbMode ? {} : (isDiv ? { zmid: 0 } : {})),
    showscale: false,
    hoverinfo: 'x+y',
    hovertemplate: '',
  }];

  const totalSamples = Array.isArray(window.sectionShape) ? window.sectionShape[1] : (y1 - y0 + 1);
  const layout = buildLayout({
    mode: 'heatmap',
    x0,
    x1,
    y0,
    y1,
    stepX,
    stepY,
    totalSamples,
    dt: window.defaultDt ?? 0.002,
    savedXRange: window.savedXRange,
    savedYRange: window.savedYRange,
    clickmode: clickModeForCurrentState(),
    dragmode: typeof window.effectiveDragMode === 'function' ? window.effectiveDragMode() : 'zoom',
    uirevision: currentUiRevision(),
    fbTitle: fbMode ? 'First-break Probability' : null,
  });
  layout.shapes = buildShapes({ xMin: x0, xMax: x1 });

  const plotDivRef = getPlotDiv();
  if (!plotDivRef) return;
  applyPlotly(plotDivRef, traces, layout);
}

function plotSeismicData(seismic, dt, startTrace = 0, endTrace = seismic.length - 1) {
  const plotDiv = getPlotDiv();
  if (!plotDiv || !seismic || !seismic.length) return;

  snapshotAxesRangesFromDOM();

  const totalTraces = seismic.length;
  const clampedStart = Math.max(0, Math.min(startTrace, totalTraces - 1));
  const clampedEnd = Math.max(clampedStart, Math.min(endTrace, totalTraces - 1));
  const nTraces = clampedEnd - clampedStart + 1;
  const nSamples = seismic[0]?.length ?? 0;
  if (!nSamples) return;

  const widthPx = plotDiv.clientWidth || 1;
  const visibleTraces = clampedEnd - clampedStart + 1;
  const density = visibleTraces / Math.max(1, widthPx);
  const modeSelect = document.getElementById('layerSelect');
  const mode = modeSelect ? modeSelect.value : 'raw';
  const fbMode = mode === 'fbprob';
  const threshold = typeof window.WIGGLE_DENSITY_THRESHOLD === 'number' ? window.WIGGLE_DENSITY_THRESHOLD : 0.2;
  const slider = document.getElementById('key1_idx_slider');
  const keyIdx = slider ? Number.parseInt(slider.value, 10) || 0 : 0;
  const key1Val = Array.isArray(window.key1Values) ? window.key1Values[keyIdx] : undefined;

  if (!fbMode && density < threshold) {
    const rows = nSamples;
    const cols = nTraces;
    const values = new Float32Array(rows * cols);
    for (let c = 0; c < cols; c += 1) {
      const trace = seismic[clampedStart + c];
      for (let r = 0; r < rows; r += 1) {
        values[r * cols + c] = trace[r];
      }
    }
    renderWindowWiggle({
      key1: key1Val,
      requestedLayer: mode,
      effectiveLayer: fbMode ? 'fbprob' : mode,
      pipelineKey: null,
      x0: clampedStart,
      x1: clampedEnd,
      y0: 0,
      y1: rows - 1,
      stepX: 1,
      stepY: 1,
      shape: [rows, cols],
      values,
      mode: 'wiggle',
    });
    return;
  }

  let factor = 1;
  while (Math.floor(nTraces / factor) * Math.floor(nSamples / factor) > MAX_POINTS) {
    factor += 1;
  }
  const cols = Math.max(1, Math.floor(nTraces / factor));
  const rows = Math.max(1, Math.floor(nSamples / factor));
  setGrid({ x0: clampedStart, stepX: factor, y0: 0, stepY: factor });

  const values = new Float32Array(rows * cols);
  for (let c = 0; c < cols; c += 1) {
    const trace = seismic[clampedStart + c * factor];
    for (let r = 0; r < rows; r += 1) {
      const sampleIdx = r * factor;
      values[r * cols + c] = trace?.[sampleIdx] ?? 0;
    }
  }

  renderWindowHeatmap({
    key1: key1Val,
    requestedLayer: mode,
    effectiveLayer: fbMode ? 'fbprob' : mode,
    pipelineKey: null,
    x0: clampedStart,
    x1: clampedStart + (cols - 1) * factor,
    y0: 0,
    y1: (rows - 1) * factor,
    stepX: factor,
    stepY: factor,
    shape: [rows, cols],
    values,
    mode: 'heatmap',
  });
}

export function renderLatestView(startOverride = undefined, endOverride = undefined) {
  const sel = document.getElementById('layerSelect');
  const layer = sel ? sel.value : 'raw';
  const slider = document.getElementById('key1_idx_slider');
  const idx = slider ? Number.parseInt(slider.value, 10) || 0 : 0;
  const key1Val = Array.isArray(window.key1Values) ? window.key1Values[idx] : undefined;

  if (window.latestSeismicData) {
    const startTrace = typeof startOverride === 'number'
      ? startOverride
      : (typeof window.renderedStart === 'number' ? window.renderedStart : 0);
    const endTrace = typeof endOverride === 'number'
      ? endOverride
      : (typeof window.renderedEnd === 'number'
        ? window.renderedEnd
        : window.latestSeismicData.length - 1);
    plotSeismicData(window.latestSeismicData, window.defaultDt ?? 0.002, startTrace, endTrace);
    return;
  }

  const latestWindow = window.latestWindowRender;
  if (
    latestWindow &&
    latestWindow.requestedLayer === layer &&
    latestWindow.key1 === key1Val
  ) {
    if (layer !== 'raw') {
      const pipelineKeyNow = window.latestPipelineKey || null;
      if ((latestWindow.pipelineKey || null) !== (pipelineKeyNow || null)) {
        return;
      }
    }
    if (latestWindow.mode === 'wiggle') {
      renderWindowWiggle(latestWindow);
    } else {
      renderWindowHeatmap(latestWindow);
    }
  }
}
