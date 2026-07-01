(function () {
  'use strict';

  const AMP_LIMIT = 3.0;

  function payloadDt(payload) {
    const dt = Number(payload?.dt);
    if (Number.isFinite(dt) && dt > 0) return dt;
    const fallback = Number(window.defaultDt ?? defaultDt);
    return Number.isFinite(fallback) && fallback > 0 ? fallback : null;
  }

  function payloadShapeInfo(payload) {
    if (!payload || !Array.isArray(payload.shape) || payload.shape.length !== 2) return null;
    const rows = Number(payload.shape[0]);
    const cols = Number(payload.shape[1]);
    if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows <= 0 || cols <= 0) return null;
    return { rows, cols, total: rows * cols };
  }

  function payloadInvScale(payload) {
    const payloadScale = Number(payload?.scale);
    const quantScale = Number(payload?.quant?.scale);
    const scale = Number.isFinite(payloadScale) && payloadScale !== 0
      ? payloadScale
      : quantScale;
    return Number.isFinite(scale) && scale !== 0 ? 1 / scale : 1;
  }

  function payloadHasComputeValues(payload) {
    const shape = payloadShapeInfo(payload);
    if (!shape) return false;
    return (
      (payload.valuesI8 instanceof Int8Array && payload.valuesI8.length >= shape.total) ||
      (payload.values instanceof Float32Array && payload.values.length >= shape.total)
    );
  }

  function canUseCachedComparePayload(payload, source) {
    if (source?.domain !== 'probability') return true;
    return payloadHasComputeValues(payload);
  }

  function sourceDomain(options) {
    if (typeof options === 'string') return options;
    return options?.domain || '';
  }

  function payloadToF32(payload, options = {}) {
    const shape = payloadShapeInfo(payload);
    if (!shape) return null;
    const { rows, cols, total } = shape;
    let out = null;
    if (payload.valuesI8 instanceof Int8Array && payload.valuesI8.length >= total) {
      const invScale = payloadInvScale(payload);
      out = new Float32Array(total);
      for (let i = 0; i < total; i++) out[i] = payload.valuesI8[i] * invScale;
    } else if (payload.values instanceof Float32Array && payload.values.length >= total) {
      out = new Float32Array(payload.values.subarray(0, total));
    } else if (sourceDomain(options) === 'probability') {
      return null;
    } else if (payload.zBacking instanceof Float32Array && payload.zBacking.length >= total) {
      out = new Float32Array(payload.zBacking.subarray(0, total));
    } else if (Array.isArray(payload.zRows) && payload.zRows.length === rows) {
      out = new Float32Array(total);
      for (let r = 0; r < rows; r++) {
        const row = payload.zRows[r];
        if (!row || row.length < cols) return null;
        out.set(row.subarray ? row.subarray(0, cols) : Array.from(row).slice(0, cols), r * cols);
      }
    }
    return out;
  }

  function sameShape(a, b) {
    return Array.isArray(a?.shape) && Array.isArray(b?.shape) &&
      a.shape.length === 2 && b.shape.length === 2 &&
      Number(a.shape[0]) === Number(b.shape[0]) &&
      Number(a.shape[1]) === Number(b.shape[1]);
  }

  function sameGrid(a, b) {
    return ['x0', 'x1', 'y0', 'y1', 'stepX', 'stepY'].every((key) => Number(a?.[key]) === Number(b?.[key]));
  }

  function validateComparePair(a, b, sources) {
    if (!sameShape(a, b)) return { ok: false, reason: 'shape', message: 'A-B unavailable: source shapes are different.' };
    const dtA = payloadDt(a);
    const dtB = payloadDt(b);
    if (!(Number.isFinite(dtA) && Number.isFinite(dtB)) || Math.abs(dtA - dtB) > 1e-9) {
      return { ok: false, reason: 'dt', message: 'A-B unavailable: source sample intervals are different.' };
    }
    if (!sameGrid(a, b)) return { ok: false, reason: 'grid', message: 'A-B unavailable: source grids are different.' };
    if (sources.a.domain !== sources.b.domain) {
      return { ok: false, reason: 'domain', message: 'A-B unavailable: source domains are different.' };
    }
    return { ok: true, reason: '', message: '' };
  }

  function subtractF32(a, b) {
    if (!(a instanceof Float32Array) || !(b instanceof Float32Array) || a.length !== b.length) return null;
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
    return out;
  }

  function rowsFromF32(values, rows, cols) {
    const out = new Array(rows);
    for (let r = 0; r < rows; r++) out[r] = values.subarray(r * cols, (r + 1) * cols);
    return out;
  }

  function compareHeatmapScale(panel, gain) {
    const g = Math.max(Number(gain) || 1.0, 1e-9);
    if (panel?.kind === 'source' && panel.domain === 'probability') {
      return { zmin: 0, zmax: 1 / g, signed: false };
    }
    if (panel?.kind === 'diff' && panel.domain === 'probability') {
      return { zmin: -1 / g, zmax: 1 / g, signed: true };
    }
    return { zmin: -AMP_LIMIT / g, zmax: AMP_LIMIT / g, signed: true };
  }

  window.__svCompareData = Object.freeze({
    payloadDt,
    payloadShapeInfo,
    payloadInvScale,
    payloadHasComputeValues,
    canUseCachedComparePayload,
    sourceDomain,
    payloadToF32,
    sameShape,
    sameGrid,
    validateComparePair,
    subtractF32,
    rowsFromF32,
    compareHeatmapScale,
  });
})();
