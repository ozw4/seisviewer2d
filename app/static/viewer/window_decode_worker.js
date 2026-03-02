/* global importScripts, msgpack */
'use strict';

importScripts('/static/msgpack.min.js');

const canceledJobIds = new Set();
const lutCache = new Map();
const CANCEL_CHECK_MASK = 0x0fff; // every 4096 elements

function buildLinearLUT(lo, hi) {
  const lut = new Float32Array(256);
  const scale = (hi - lo) / 255.0;
  for (let i = 0; i < 256; i++) lut[i] = lo + i * scale;
  return lut;
}

function buildMulawLUT(lo, hi, mu = 255) {
  const lut = new Float32Array(256);
  const ln = Math.log(1 + mu);
  for (let i = 0; i < 256; i++) {
    const q = i / 255.0;
    const y = q * 2.0 - 1.0;
    const x = Math.sign(y) * (Math.exp(ln * Math.abs(y)) - 1.0) / mu;
    const x01 = (x + 1.0) * 0.5;
    lut[i] = lo + x01 * (hi - lo);
  }
  return lut;
}

function resolveShape(shapeRaw) {
  let shape = null;
  if (Array.isArray(shapeRaw)) {
    shape = shapeRaw;
  } else if (shapeRaw && typeof shapeRaw.length === 'number') {
    shape = [shapeRaw[0], shapeRaw[1]];
  }
  if (!shape || shape.length !== 2) return null;
  const rows = Number(shape[0]);
  const cols = Number(shape[1]);
  if (!Number.isFinite(rows) || !Number.isFinite(cols)) return null;
  const rowsInt = Math.trunc(rows);
  const colsInt = Math.trunc(cols);
  if (rowsInt <= 0 || colsInt <= 0) return null;
  return [rowsInt, colsInt];
}

function resolveQuantMeta(obj) {
  if (obj && obj.quant && typeof obj.quant === 'object') return obj.quant;
  if (obj && obj.lo !== undefined && obj.hi !== undefined) {
    return {
      mode: obj.method || 'linear',
      lo: obj.lo,
      hi: obj.hi,
      mu: obj.mu ?? 255,
    };
  }
  if (obj && obj.scale != null) {
    return { scale: obj.scale };
  }
  return null;
}

function resolveLUT(quantMeta) {
  if (!quantMeta || typeof quantMeta !== 'object') return null;
  const lo = Number(quantMeta.lo);
  const hi = Number(quantMeta.hi);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  let mu = Number(quantMeta.mu);
  if (!Number.isFinite(mu) || mu <= 0) mu = 255;
  const modeRaw = typeof quantMeta.mode === 'string'
    ? quantMeta.mode
    : (typeof quantMeta.method === 'string' ? quantMeta.method : 'linear');
  const mode = String(modeRaw).toLowerCase() === 'mulaw' ? 'mulaw' : 'linear';
  const key = `${mode}:${lo}:${hi}:${mu}`;
  let lut = lutCache.get(key);
  if (!lut) {
    lut = mode === 'mulaw' ? buildMulawLUT(lo, hi, mu) : buildLinearLUT(lo, hi);
    lutCache.set(key, lut);
  }
  return lut;
}

function resolveInvScale(quantMeta, scale) {
  let scaleVal = NaN;
  if (quantMeta && typeof quantMeta === 'object' && 'scale' in quantMeta) {
    scaleVal = Number(quantMeta.scale);
  }
  if (!Number.isFinite(scaleVal) || scaleVal === 0) {
    scaleVal = Number(scale);
  }
  if (!Number.isFinite(scaleVal) || scaleVal === 0) {
    scaleVal = 1;
  }
  return 1 / scaleVal;
}

function isCanceled(jobId) {
  return canceledJobIds.has(jobId);
}

function clearCanceled(jobId) {
  canceledJobIds.delete(jobId);
}

function postDecodeError(jobId, err) {
  const message = (err && err.message) ? err.message : String(err || 'decode_failed');
  self.postMessage({
    type: 'decoded',
    jobId,
    ok: false,
    error: message,
  });
}

function handleDecode(msg) {
  const jobId = Number(msg?.jobId);
  if (!Number.isInteger(jobId)) return;
  if (isCanceled(jobId)) {
    clearCanceled(jobId);
    return;
  }
  if (!(msg?.bin instanceof ArrayBuffer)) {
    clearCanceled(jobId);
    postDecodeError(jobId, 'invalid_bin');
    return;
  }

  let obj = null;
  try {
    if (!self.msgpack || typeof self.msgpack.decode !== 'function') {
      throw new Error('msgpack_decode_unavailable');
    }
    const u8 = new Uint8Array(msg.bin);
    obj = self.msgpack.decode(u8);
  } catch (err) {
    clearCanceled(jobId);
    postDecodeError(jobId, err);
    return;
  }

  const shape = resolveShape(obj?.shape);
  if (!shape) {
    clearCanceled(jobId);
    postDecodeError(jobId, 'invalid_shape');
    return;
  }
  const [rows, cols] = shape;
  const total = rows * cols;
  if (!Number.isSafeInteger(total) || total <= 0) {
    clearCanceled(jobId);
    postDecodeError(jobId, 'invalid_shape_size');
    return;
  }

  const dataView = obj?.data;
  if (!dataView || !ArrayBuffer.isView(dataView)) {
    clearCanceled(jobId);
    postDecodeError(jobId, 'invalid_data_view');
    return;
  }
  const valuesI8 = new Int8Array(dataView.buffer, dataView.byteOffset, dataView.byteLength);
  if (valuesI8.length < total) {
    clearCanceled(jobId);
    postDecodeError(jobId, 'insufficient_data_length');
    return;
  }

  const dtRaw = Number(obj?.dt);
  const dt = Number.isFinite(dtRaw) && dtRaw > 0 ? dtRaw : null;
  const scaleRaw = Number(obj?.scale);
  const scale = Number.isFinite(scaleRaw) ? scaleRaw : null;
  const quantMeta = resolveQuantMeta(obj);
  const wantZ = msg?.wantZ === true;
  const fbMode = msg?.fbMode === true;

  try {
    if (wantZ) {
      const zBacking = new Float32Array(total);
      const lut = resolveLUT(quantMeta);
      const invScale = resolveInvScale(quantMeta, scale);
      for (let i = 0; i < total; i++) {
        if ((i & CANCEL_CHECK_MASK) === 0 && isCanceled(jobId)) {
          clearCanceled(jobId);
          return;
        }
        let val;
        if (lut) {
          const q = (valuesI8[i] + 128) & 0xff;
          val = lut[q];
        } else {
          val = valuesI8[i] * invScale;
        }
        if (fbMode) val *= 255;
        zBacking[i] = val;
      }
      if (isCanceled(jobId)) {
        clearCanceled(jobId);
        return;
      }
      clearCanceled(jobId);
      self.postMessage({
        type: 'decoded',
        jobId,
        ok: true,
        dt,
        rows,
        cols,
        scale,
        quant: quantMeta,
        zBuf: zBacking.buffer,
      }, [zBacking.buffer]);
      return;
    }

    const out = new Int8Array(total);
    out.set(valuesI8.subarray(0, total));
    if (isCanceled(jobId)) {
      clearCanceled(jobId);
      return;
    }
    clearCanceled(jobId);
    self.postMessage({
      type: 'decoded',
      jobId,
      ok: true,
      dt,
      rows,
      cols,
      scale,
      quant: quantMeta,
      i8Buf: out.buffer,
    }, [out.buffer]);
  } catch (err) {
    clearCanceled(jobId);
    postDecodeError(jobId, err);
  }
}

self.onmessage = (event) => {
  const msg = event?.data;
  if (!msg || typeof msg !== 'object') return;
  if (msg.type === 'cancel') {
    const jobId = Number(msg.jobId);
    if (Number.isInteger(jobId)) canceledJobIds.add(jobId);
    return;
  }
  if (msg.type === 'decode') {
    handleDecode(msg);
  }
};
