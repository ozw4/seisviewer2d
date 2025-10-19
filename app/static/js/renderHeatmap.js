// app/static/js/renderHeatmap.js
import {F32DoubleBuffer} from './pool.js';
import {buildLinearLUT, buildMulawLUT} from './quantLut.js';

const f32db = new F32DoubleBuffer();
let rowViews = [];
let rowViewsCap = 0;

const lutCache = new Map();
let poolGrowthCount = 0;
let lastCapA = 0;
let lastCapB = 0;

function reportPoolGrowth(pool) {
  const isA = pool === f32db.a;
  const prev = isA ? lastCapA : lastCapB;
  if (pool.cap > prev) {
    if (isA) lastCapA = pool.cap;
    else lastCapB = pool.cap;
    poolGrowthCount += 1;
    const maxCap = Math.max(lastCapA, lastCapB);
    console.info(`Heatmap F32 pool cap: ${maxCap} (elements), grows: ${poolGrowthCount}`);
  }
}

function getLUT(quant) {
  const {mode = 'linear', lo = 0, hi = 1, mu = 255} = quant || {};
  const k = `${mode}:${lo}:${hi}:${mu}`;
  let lut = lutCache.get(k);
  if (!lut) {
    lut = mode === 'mulaw' ? buildMulawLUT(lo, hi, mu) : buildLinearLUT(lo, hi);
    lutCache.set(k, lut);
  }
  return lut;
}

function ensureRowViews(rows, cols, backing) {
  if (rowViewsCap < rows) {
    for (let r = rowViewsCap; r < rows; r++) rowViews[r] = null;
    rowViewsCap = rows;
  }
  for (let r = 0; r < rows; r++) {
    rowViews[r] = backing.subarray(r * cols, (r + 1) * cols);
  }
  rowViews.length = rows;
  rowViews.backing = backing;
  rowViews.rows = rows;
  rowViews.cols = cols;
  return rowViews;
}

function cloneRowViews(source) {
  const clone = source.slice(0, source.length);
  clone.backing = source.backing;
  clone.rows = source.rows;
  clone.cols = source.cols;
  return clone;
}

function decodeIntoBuffer({i8, quant, out, length}) {
  const total = length ?? out.length;
  if (quant && ('lo' in quant) && ('hi' in quant)) {
    const lut = getLUT(quant);
    for (let p = 0; p < total; p++) {
      const q = (i8[p] + 128) & 0xff;
      out[p] = lut[q];
    }
  } else if (quant && ('scale' in quant)) {
    const inv = 1 / quant.scale;
    for (let p = 0; p < total; p++) out[p] = i8[p] * inv;
  } else {
    for (let p = 0; p < total; p++) out[p] = i8[p];
  }
  return out;
}

/**
 * Convert Int8 quantized data to Plotly heatmap z using a pooled Float32 backing buffer.
 */
export function toPlotlyHeatmapZ({i8, rows, cols, quant}) {
  const total = rows * cols;
  const out = f32db.acquire(total);
  const pool = f32db.flip ? f32db.a : f32db.b;
  reportPoolGrowth(pool);
  decodeIntoBuffer({i8, quant, out, length: total});
  return cloneRowViews(ensureRowViews(rows, cols, out));
}

export function getQuantLUT(quant) {
  return getLUT(quant);
}

export function getHeatmapPoolStats() {
  const cap = Math.max(lastCapA, lastCapB);
  return {cap, grows: poolGrowthCount};
}
