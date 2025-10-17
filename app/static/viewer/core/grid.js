// /static/viewer/core/grid.js
// Single source of truth for coordinate/grid math and DOM<->data helpers

export const Grid = {
  x0: 0,      // leftmost trace center
  stepX: 1,   // trace stride (1 for wiggle; step for heatmap)
  y0: 0,      // top sample index center
  stepY: 1,   // sample stride
  get dt() { return (window.defaultDt ?? 0.002) || 0.002; }
};

export function setGrid({ x0, stepX, y0, stepY }) {
  Grid.x0   = Number.isFinite(x0)   ? x0   : 0;
  Grid.stepX= Number.isFinite(stepX) ? stepX: 1;
  Grid.y0   = Number.isFinite(y0)   ? y0   : 0;
  Grid.stepY= Number.isFinite(stepY) ? stepY: 1;
}

export function getPlotEnv() {
  const gd  = document.getElementById('plot');
  const rect= gd?.getBoundingClientRect();
  const m   = gd?._fullLayout?._size;   // {l,t,w,h}
  const xa  = gd?._fullLayout?.xaxis;
  const ya  = gd?._fullLayout?.yaxis;
  if (!gd || !rect || !m || !xa || !ya) return null;
  return { gd, rect, m, xa, ya };
}

// Convert client pixels to data coords using Plotly axes
export function dataXYFromClient(clientX, clientY) {
  const env = getPlotEnv(); if (!env) return null;
  const { rect, m, xa, ya } = env;
  let x = Number.NaN, y = Number.NaN;
  if (Number.isFinite(clientX)) {
    const innerX = clientX - rect.left - m.l;
    x = xa.p2d(innerX);
  }
  if (Number.isFinite(clientY)) {
    const innerY = clientY - rect.top - m.t;
    y = ya.p2d(innerY);
  }
  return { x, y };
}

// Snap helpers relative to current Grid
export function snapTraceFromDataX(x) {
  const k = Math.round((x - Grid.x0) / Grid.stepX);
  return Grid.x0 + k * Grid.stepX;
}
export function snapTimeFromDataY(y) {
  const idx = y / Grid.dt;
  const k = Math.round((idx - Grid.y0) / Grid.stepY);
  const snappedIdx = Grid.y0 + k * Grid.stepY;
  return snappedIdx * Grid.dt;
}

// Public helpers used around the codebase
export function traceAtPixel(clientX) {
  const env = getPlotEnv(); if (!env) return Number.NaN;
  const { rect, m, xa } = env;
  const innerX = clientX - rect.left - m.l;
  const x = xa.p2d(innerX);
  if (!Number.isFinite(x)) return Number.NaN;
  return snapTraceFromDataX(x);
}

export function pixelForTrace(trace) {
  const env = getPlotEnv(); if (!env) return null;
  const { rect, m, xa } = env;
  const snapped = snapTraceFromDataX(trace);
  const inner = xa.d2p(snapped);
  return rect.left + m.l + inner; // clientX
}

export function timeAtPixel(clientY) {
  const env = getPlotEnv(); if (!env) return Number.NaN;
  const { rect, m, ya } = env;
  const innerY = clientY - rect.top - m.t;
  const y = ya.p2d(innerY);
  if (!Number.isFinite(y)) return Number.NaN;
  return snapTimeFromDataY(y);
}
