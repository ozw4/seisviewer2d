const EPSILON = 1e-9;

function finiteNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function normalizeRect(rect) {
  if (!rect || typeof rect !== 'object') return null;
  const left = finiteNumber(rect.left ?? rect.x) ?? 0;
  const top = finiteNumber(rect.top ?? rect.y) ?? 0;
  const width = finiteNumber(rect.width);
  const height = finiteNumber(rect.height);
  if (!(width > 0) || !(height > 0)) return null;
  return { left, top, width, height };
}

function normalizeRange(range) {
  if (!Array.isArray(range) || range.length !== 2) return null;
  const a = finiteNumber(range[0]);
  const b = finiteNumber(range[1]);
  if (a === null || b === null || Math.abs(b - a) <= EPSILON) return null;
  return [a, b];
}

function normalizeOptionalRange(a, b) {
  const start = finiteNumber(a);
  const end = finiteNumber(b);
  if (start === null || end === null) return null;
  return [Math.min(start, end), Math.max(start, end)];
}

function rangeMinMax(range) {
  return [Math.min(range[0], range[1]), Math.max(range[0], range[1])];
}

function intersectRanges(a, b) {
  if (!a) return null;
  if (!b) return a;
  const lo = Math.max(a[0], b[0]);
  const hi = Math.min(a[1], b[1]);
  if (lo > hi + EPSILON) return null;
  return [lo, hi];
}

function withinRange(value, range) {
  const n = finiteNumber(value);
  if (n === null || !range) return false;
  const [lo, hi] = rangeMinMax(range);
  return n >= lo - EPSILON && n <= hi + EPSILON;
}

function dataToPixel(value, range, pixelStart, pixelLength) {
  const n = finiteNumber(value);
  if (n === null || !range || !(pixelLength > 0)) return null;
  return pixelStart + ((n - range[0]) / (range[1] - range[0])) * pixelLength;
}

function pixelToData(pixel, range, pixelStart, pixelLength) {
  const n = finiteNumber(pixel);
  if (n === null || !range || !(pixelLength > 0)) return null;
  return range[0] + ((n - pixelStart) / pixelLength) * (range[1] - range[0]);
}

function normalizePlotArea(rect, plotArea) {
  const area = normalizeRect(plotArea);
  if (!area) {
    return { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
  }
  return {
    left: rect.left + area.left,
    top: rect.top + area.top,
    width: area.width,
    height: area.height,
  };
}

function makeInvalidTransform(input) {
  return {
    valid: false,
    input,
    traceTimeToPixel() {
      return null;
    },
    pixelToTraceTime() {
      return null;
    },
    isTraceTimeVisible() {
      return false;
    },
    visibleTraceRange() {
      return null;
    },
    visibleTimeRange() {
      return null;
    },
  };
}

export function createOverlayTransform(input = {}) {
  const rect = normalizeRect(input.containerRect ?? input.boundingRect ?? input.rect);
  const xRange = normalizeRange(input.xRange);
  const yRange = normalizeRange(input.yRange);
  if (!rect || !xRange || !yRange) return makeInvalidTransform(input);

  const plotArea = normalizePlotArea(rect, input.plotAreaRect ?? input.plotArea);
  const transpose = input.transpose === true || input.transpose === 'true';
  const traceAxisRange = transpose ? yRange : xRange;
  const timeAxisRange = transpose ? xRange : yRange;
  const traceRenderRange = normalizeOptionalRange(input.renderedStart, input.renderedEnd);
  const timeRenderRange =
    normalizeOptionalRange(input.renderedTimeStart, input.renderedTimeEnd) ??
    normalizeOptionalRange(input.renderedTimeRange?.[0], input.renderedTimeRange?.[1]);

  const tracePixelStart = transpose ? plotArea.top : plotArea.left;
  const tracePixelLength = transpose ? plotArea.height : plotArea.width;
  const timePixelStart = transpose ? plotArea.left : plotArea.top;
  const timePixelLength = transpose ? plotArea.width : plotArea.height;
  const visibleTrace = intersectRanges(rangeMinMax(traceAxisRange), traceRenderRange);
  const visibleTime = intersectRanges(rangeMinMax(timeAxisRange), timeRenderRange);

  return {
    valid: true,
    input,
    rect,
    plotArea,
    transpose,
    traceTimeToPixel(trace, timeS) {
      const tracePx = dataToPixel(trace, traceAxisRange, tracePixelStart, tracePixelLength);
      const timePx = dataToPixel(timeS, timeAxisRange, timePixelStart, timePixelLength);
      if (tracePx === null || timePx === null) return null;
      const clientX = transpose ? timePx : tracePx;
      const clientY = transpose ? tracePx : timePx;
      return {
        x: clientX,
        y: clientY,
        relativeX: clientX - rect.left,
        relativeY: clientY - rect.top,
      };
    },
    pixelToTraceTime(x, y) {
      const tracePixel = transpose ? y : x;
      const timePixel = transpose ? x : y;
      const trace = pixelToData(tracePixel, traceAxisRange, tracePixelStart, tracePixelLength);
      const timeS = pixelToData(timePixel, timeAxisRange, timePixelStart, timePixelLength);
      if (trace === null || timeS === null) return null;
      return { trace, timeS };
    },
    isTraceTimeVisible(trace, timeS) {
      return (
        withinRange(trace, traceAxisRange) &&
        withinRange(timeS, timeAxisRange) &&
        (!traceRenderRange || withinRange(trace, traceRenderRange)) &&
        (!timeRenderRange || withinRange(timeS, timeRenderRange))
      );
    },
    visibleTraceRange() {
      return visibleTrace ? [visibleTrace[0], visibleTrace[1]] : null;
    },
    visibleTimeRange() {
      return visibleTime ? [visibleTime[0], visibleTime[1]] : null;
    },
  };
}

export default createOverlayTransform;

if (typeof window !== 'undefined') {
  window.ViewerOverlayTransform = {
    createOverlayTransform,
  };
}
