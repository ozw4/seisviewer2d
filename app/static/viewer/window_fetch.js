    function maybeFetchIfOutOfWindow() {
      if (typeof isCompareModeEnabled === 'function' && isCompareModeEnabled()) {
        return;
      }
      if (!latestWindowRender || !sectionShape) {
        scheduleWindowFetch();
        return;
      }
      const { x0, x1, y0, y1 } = latestWindowRender;
      const win = currentVisibleWindow();
      if (!win) return;
      const guardX = Math.max(1, Math.floor((x1 - x0 + 1) * 0.05));
      const guardY = Math.max(1, Math.floor((y1 - y0 + 1) * 0.05));
      const insideX = win.x0 >= x0 + guardX && win.x1 <= x1 - guardX;
      const insideY = win.y0 >= y0 + guardY && win.y1 <= y1 - guardY;
      if (!(insideX && insideY)) scheduleWindowFetch();
    }
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

    function computeStepsForWindow({
      tracesVisible,
      samplesVisible,
      widthPx,
      heightPx,
      oversampleX = 1.2,
      oversampleY = 1.2,
      maxPoints, // ここは未指定OK
    }) {
      const ratio = window.devicePixelRatio || 1;
      const effW = Math.max(1, Math.round(widthPx * ratio));
      const effH = Math.max(1, Math.round(heightPx * ratio));
      // cfg 未初期化でも安全な閾値
      const MAX = Number.isFinite(maxPoints)
        ? Math.max(1, Math.floor(maxPoints))
        : ((typeof cfg === 'object' && Number.isFinite(cfg.WINDOW_MAX_POINTS))
          ? cfg.WINDOW_MAX_POINTS
          : 2_000_000);

      let stepX = Math.max(1, Math.ceil(tracesVisible / (effW * oversampleX)));
      let stepY = Math.max(1, Math.ceil(samplesVisible / (effH * oversampleY)));

      const tracesOut = () => Math.ceil(tracesVisible / stepX);
      const samplesOut = () => Math.ceil(samplesVisible / stepY);

      let guard = 0;
      while (tracesOut() * samplesOut() > MAX && guard < 512) {
        if (tracesOut() / effW > samplesOut() / effH) stepX += 1; else stepY += 1;
        guard += 1;
      }

      stepX = roundUpPowerOfTwo(stepX);
      stepY = roundUpPowerOfTwo(stepY);

      guard = 0;
      while (Math.ceil(tracesVisible / stepX) * Math.ceil(samplesVisible / stepY) > MAX && guard < 512) {
        if (tracesVisible / stepX > samplesVisible / stepY) stepX = roundUpPowerOfTwo(stepX + 1);
        else stepY = roundUpPowerOfTwo(stepY + 1);
        guard += 1;
      }

      return { step_x: stepX, step_y: stepY };
    }

    function buildWindowLoadingMessage({ mode, stepX, stepY }) {
      return `Loading window... Mode: ${mode}, stepX=${stepX}, stepY=${stepY}`;
    }

    function showLoading(message) {
      const overlay = document.getElementById('windowLoadingOverlay');
      if (!overlay) return;
      const messageEl = document.getElementById('windowLoadingMessage');
      if (messageEl) {
        messageEl.textContent = message || 'Loading window...';
      }
      overlay.style.pointerEvents = window.windowLoadingBlocksInput === true ? 'auto' : 'none';
      overlay.classList.add('show');
    }

    function hideLoading() {
      const overlay = document.getElementById('windowLoadingOverlay');
      if (!overlay) return;
      overlay.classList.remove('show');
    }

    function bumpWindowFetchId() {
      activeWindowFetchId += 1;
      windowFetchToken = activeWindowFetchId;
      return activeWindowFetchId;
    }

    const WINDOW_CACHE_DEFAULT_MAX_BYTES = 128 * 1024 * 1024;
    const WINDOW_CACHE_DEFAULT_MAX_ENTRIES = 24;
    const PREFETCH_DEFAULT_ENABLE = true;
    const PREFETCH_DEFAULT_MAX_INFLIGHT = 2;
    const PREFETCH_DEFAULT_MARGIN_RATIO = 0.15;
    const PREFETCH_DEFAULT_ENABLE_Y = false;
    const WINDOW_DECODE_USE_WORKER_DEFAULT = true;
    const WINDOW_DECODE_WORKER_PATH_DEFAULT = '/static/viewer/window_decode_worker.js';
    const DECODE_WORKER_KIND_MAIN = 'main';
    const DECODE_WORKER_KIND_PREFETCH = 'prefetch';
    const windowPayloadCache = new Map();
    const prefetchInflight = new Map();
    const decodeWorkerStates = {
      [DECODE_WORKER_KIND_MAIN]: { worker: null, pending: new Map() },
      [DECODE_WORKER_KIND_PREFETCH]: { worker: null, pending: new Map() },
    };
    let nextWorkerJobId = 1;
    let activeMainDecodeJobId = null;
    let hasWarnedDecodeFallback = false;
    const windowCacheStats = {
      hits: 0,
      misses: 0,
      evicts: 0,
      prefetchStarted: 0,
      prefetchDone: 0,
      bytes: 0,
      entries: 0,
    };

    function syncWindowCacheStats() {
      windowCacheStats.entries = windowPayloadCache.size;
      if (!Number.isFinite(windowCacheStats.bytes) || windowCacheStats.bytes < 0) {
        windowCacheStats.bytes = 0;
      }
    }

    function resetWindowCacheStats() {
      windowCacheStats.hits = 0;
      windowCacheStats.misses = 0;
      windowCacheStats.evicts = 0;
      windowCacheStats.prefetchStarted = 0;
      windowCacheStats.prefetchDone = 0;
      windowCacheStats.bytes = 0;
      windowCacheStats.entries = windowPayloadCache.size;
    }

    function readWindowCacheMaxBytes() {
      const configured = (typeof cfg === 'object' && cfg !== null)
        ? Number(cfg.WINDOW_CACHE_MAX_BYTES)
        : NaN;
      if (Number.isFinite(configured) && configured > 0) {
        return Math.floor(configured);
      }
      return WINDOW_CACHE_DEFAULT_MAX_BYTES;
    }

    function readWindowCacheMaxEntries() {
      const configured = (typeof cfg === 'object' && cfg !== null)
        ? Number(cfg.WINDOW_CACHE_MAX_ENTRIES)
        : NaN;
      if (Number.isFinite(configured) && configured > 0) {
        return Math.floor(configured);
      }
      return WINDOW_CACHE_DEFAULT_MAX_ENTRIES;
    }

    function readPrefetchEnabled() {
      const configured = (typeof cfg === 'object' && cfg !== null)
        ? cfg.PREFETCH_ENABLE
        : undefined;
      if (typeof configured === 'boolean') {
        return configured;
      }
      return PREFETCH_DEFAULT_ENABLE;
    }

    function readPrefetchMaxInflight() {
      const configured = (typeof cfg === 'object' && cfg !== null)
        ? Number(cfg.PREFETCH_MAX_INFLIGHT)
        : NaN;
      if (Number.isFinite(configured) && configured > 0) {
        return Math.floor(configured);
      }
      return PREFETCH_DEFAULT_MAX_INFLIGHT;
    }

    function readPrefetchMarginRatio() {
      const configured = (typeof cfg === 'object' && cfg !== null)
        ? Number(cfg.PREFETCH_MARGIN_RATIO)
        : NaN;
      if (Number.isFinite(configured) && configured > 0) {
        return Math.min(0.49, Math.max(0.01, configured));
      }
      return PREFETCH_DEFAULT_MARGIN_RATIO;
    }

    function readPrefetchEnableY() {
      const configured = (typeof cfg === 'object' && cfg !== null)
        ? cfg.PREFETCH_ENABLE_Y
        : undefined;
      if (typeof configured === 'boolean') {
        return configured;
      }
      return PREFETCH_DEFAULT_ENABLE_Y;
    }

    function readWindowDecodeUseWorker() {
      const configured = (typeof cfg === 'object' && cfg !== null)
        ? cfg.WINDOW_DECODE_USE_WORKER
        : undefined;
      if (typeof configured === 'boolean') {
        return configured;
      }
      return WINDOW_DECODE_USE_WORKER_DEFAULT;
    }

    function readWindowDecodeWorkerPath() {
      const configured = (typeof cfg === 'object' && cfg !== null)
        ? cfg.WINDOW_DECODE_WORKER_PATH
        : undefined;
      if (typeof configured === 'string' && configured.trim()) {
        return configured.trim();
      }
      return WINDOW_DECODE_WORKER_PATH_DEFAULT;
    }

    function warnDecodeFallback(reason) {
      if (hasWarnedDecodeFallback) return;
      hasWarnedDecodeFallback = true;
      console.warn('[FALLBACK] Window decode worker disabled; using main-thread decode', { reason });
    }

    function estimateWindowPayloadBytes(payload) {
      const valuesBytes = Number(payload?.valuesI8?.byteLength) || 0;
      const zBackingBytes = Number(payload?.zBacking?.byteLength) || 0;
      return valuesBytes + zBackingBytes + 512;
    }

    function evictWindowCacheIfNeeded() {
      const maxEntries = readWindowCacheMaxEntries();
      const maxBytes = readWindowCacheMaxBytes();
      while (windowPayloadCache.size > 0
        && (windowPayloadCache.size > maxEntries || windowCacheStats.bytes > maxBytes)) {
        const oldestKey = windowPayloadCache.keys().next().value;
        const oldestEntry = windowPayloadCache.get(oldestKey);
        windowPayloadCache.delete(oldestKey);
        windowCacheStats.bytes -= Number(oldestEntry?.bytes) || 0;
        windowCacheStats.evicts += 1;
      }
      syncWindowCacheStats();
    }

    function windowCacheGet(key) {
      const entry = windowPayloadCache.get(key);
      if (!entry) {
        windowCacheStats.misses += 1;
        syncWindowCacheStats();
        return null;
      }
      windowPayloadCache.delete(key);
      windowPayloadCache.set(key, entry);
      windowCacheStats.hits += 1;
      syncWindowCacheStats();
      return entry.payload;
    }

    function windowCacheSet(key, payload) {
      const bytes = estimateWindowPayloadBytes(payload);
      if (windowPayloadCache.has(key)) {
        const prev = windowPayloadCache.get(key);
        windowCacheStats.bytes -= Number(prev?.bytes) || 0;
        windowPayloadCache.delete(key);
      }
      windowPayloadCache.set(key, {
        payload,
        bytes,
        t: Date.now(),
      });
      windowCacheStats.bytes += bytes;
      evictWindowCacheIfNeeded();
      syncWindowCacheStats();
      return payload;
    }

    function windowCachePeek(key) {
      const entry = windowPayloadCache.get(key);
      return entry ? entry.payload : null;
    }

    function windowCacheClear() {
      windowPayloadCache.clear();
      resetWindowCacheStats();
      syncWindowCacheStats();
    }

    function getSafeLmoKey() {
      return typeof window.currentLmoKey === 'function'
        ? window.currentLmoKey()
        : 'lmo:off';
    }

    function getSafeLmoState() {
      if (typeof window.getCurrentLinearMoveout !== 'function') {
        return { enabled: false };
      }
      return window.getCurrentLinearMoveout();
    }

    function appendLinearMoveoutParams(params) {
      const lmo = getSafeLmoState();
      if (!lmo || !lmo.enabled) {
        params.set('lmo_enabled', 'false');
        return;
      }

      params.set('lmo_enabled', 'true');
      params.set('lmo_velocity_mps', String(lmo.velocityMps));
      params.set('lmo_offset_byte', String(lmo.offsetByte));
      params.set('lmo_offset_scale', String(lmo.offsetScale));
      params.set('lmo_offset_mode', lmo.offsetMode);
      params.set('lmo_ref_mode', lmo.refMode);
      params.set('lmo_ref_trace', String(lmo.refTrace));
      params.set('lmo_polarity', String(lmo.polarity));
    }

    function isCurrentLmoKey(lmoKey) {
      return lmoKey === getSafeLmoKey();
    }

    function buildWindowCacheKey({
      fileId,
      key1,
      key1Byte,
      key2Byte,
      x0,
      x1,
      y0,
      y1,
      stepX,
      stepY,
      requestedLayer,
      effectiveLayer,
      pipelineKey,
      tapLabel,
      referencePipelineKey,
      referenceTapLabel,
      scaling,
      transpose,
      mode,
      purpose,
      lmoKey,
    }) {
      const enc = (value) => encodeURIComponent(value == null ? '' : String(value));
      const parts = ['svwin'];
      if (purpose) parts.push(`purpose=${enc(purpose)}`);
      parts.push(
        `file=${enc(fileId)}`,
        `k1=${enc(key1)}`,
        `b1=${enc(key1Byte)}`,
        `b2=${enc(key2Byte)}`,
        `x=${enc(x0)}-${enc(x1)}`,
        `y=${enc(y0)}-${enc(y1)}`,
        `sx=${enc(stepX)}`,
        `sy=${enc(stepY)}`,
        `rql=${enc(requestedLayer)}`,
        `layer=${enc(effectiveLayer)}`,
        `pkey=${enc(pipelineKey)}`,
        `tap=${enc(tapLabel)}`,
        `rpkey=${enc(referencePipelineKey)}`,
        `rtap=${enc(referenceTapLabel)}`,
        `sc=${enc(scaling)}`,
        `tr=${enc(transpose)}`,
        `mode=${enc(mode)}`,
        `lmo=${enc(lmoKey || 'lmo:off')}`,
      );
      return parts.join('|');
    }

    function buildWindowRequestArtifacts({
      fileId,
      key1Val,
      key1Byte,
      key2Byte,
      windowInfo,
      stepX,
      stepY,
      requestedLayer,
      effectiveLayer,
      pipelineKey,
      tapLabel,
      referencePipelineKey,
      referenceTapLabel,
      scaling,
      transpose,
      mode,
      purpose,
    }) {
      const resolvedPipelineKey = (tapLabel && pipelineKey) ? pipelineKey : null;
      const resolvedTapLabel = (tapLabel && resolvedPipelineKey) ? tapLabel : null;
      const resolvedReferencePipelineKey = (referenceTapLabel && referencePipelineKey)
        ? referencePipelineKey
        : null;
      const resolvedReferenceTapLabel = (
        referenceTapLabel && resolvedReferencePipelineKey
      ) ? referenceTapLabel : null;
      const lmoKey = getSafeLmoKey();
      const params = new URLSearchParams({
        file_id: String(fileId),
        key1: String(key1Val),
        key1_byte: String(key1Byte),
        key2_byte: String(key2Byte),
        x0: String(windowInfo.x0),
        x1: String(windowInfo.x1),
        y0: String(windowInfo.y0),
        y1: String(windowInfo.y1),
        step_x: String(stepX),
        step_y: String(stepY),
      });
      if (resolvedPipelineKey && resolvedTapLabel) {
        params.set('pipeline_key', resolvedPipelineKey);
        params.set('tap_label', resolvedTapLabel);
      }
      if (resolvedReferencePipelineKey && resolvedReferenceTapLabel) {
        params.set('reference_pipeline_key', resolvedReferencePipelineKey);
        params.set('reference_tap_label', resolvedReferenceTapLabel);
      }
      params.set('transpose', transpose);
      params.set('scaling', scaling);
      appendLinearMoveoutParams(params);

      const resolvedRequestContext = {
        fileId,
        key1Val,
        key1Byte,
        key2Byte,
        windowInfo,
        stepX,
        stepY,
        requestedLayer,
        effectiveLayer,
        pipelineKey: resolvedPipelineKey,
        tapLabel: resolvedTapLabel,
        referencePipelineKey: resolvedReferencePipelineKey,
        referenceTapLabel: resolvedReferenceTapLabel,
        scaling,
        transpose,
        mode,
        purpose,
        lmoKey,
      };

      return {
        params,
        cacheKey: buildWindowCacheKey({
          fileId,
          key1: key1Val,
          key1Byte,
          key2Byte,
          x0: windowInfo.x0,
          x1: windowInfo.x1,
          y0: windowInfo.y0,
          y1: windowInfo.y1,
          stepX,
          stepY,
          requestedLayer,
          effectiveLayer,
          pipelineKey: resolvedPipelineKey,
          tapLabel: resolvedTapLabel,
          referencePipelineKey: resolvedReferencePipelineKey,
          referenceTapLabel: resolvedReferenceTapLabel,
          scaling,
          transpose,
          mode,
          purpose,
          lmoKey,
        }),
        requestContext: resolvedRequestContext,
        payloadMeta: {
          key1: key1Val,
          requestedLayer,
          effectiveLayer,
          pipelineKey: resolvedPipelineKey,
          referencePipelineKey: resolvedReferencePipelineKey,
          x0: windowInfo.x0,
          x1: windowInfo.x1,
          y0: windowInfo.y0,
          y1: windowInfo.y1,
          stepX,
          stepY,
          mode,
          purpose,
          lmoKey,
        },
      };
    }

    function resolveWindowShape(shapeRaw) {
      const shape = Array.isArray(shapeRaw) ? shapeRaw : Array.from(shapeRaw ?? []);
      if (shape.length !== 2) return null;
      const rows = Math.trunc(Number(shape[0]));
      const cols = Math.trunc(Number(shape[1]));
      if (!Number.isFinite(rows) || !Number.isFinite(cols)) return null;
      if (rows <= 0 || cols <= 0) return null;
      return [rows, cols];
    }

    function resolveWindowQuantMeta(obj) {
      return obj?.quant || (
        (obj?.lo !== undefined && obj?.hi !== undefined)
          ? { mode: obj.method || 'linear', lo: obj.lo, hi: obj.hi, mu: obj.mu ?? 255 }
          : (obj?.scale != null ? { scale: obj.scale } : null)
      );
    }

    function buildHeatmapRowsFromBacking(zBacking, rows, cols) {
      const zRows = new Array(rows);
      for (let r = 0; r < rows; r++) {
        zRows[r] = zBacking.subarray(r * cols, (r + 1) * cols);
      }
      return zRows;
    }

    function flushDecodeWorkerPending(kind, reason) {
      const state = decodeWorkerStates[kind];
      if (!state) return;
      for (const [, pending] of state.pending.entries()) {
        pending.reject(new Error(reason || 'decode_worker_error'));
      }
      state.pending.clear();
    }

    function ensureDecodeWorker(kind) {
      const state = decodeWorkerStates[kind];
      if (!state) throw new Error(`Unknown decode worker kind: ${kind}`);
      if (state.worker) return state.worker;
      if (typeof Worker !== 'function') {
        throw new Error('Worker is not supported in this browser');
      }
      const workerPath = readWindowDecodeWorkerPath();
      const worker = new Worker(workerPath);
      worker.onmessage = (event) => {
        const msg = event?.data;
        if (!msg || msg.type !== 'decoded') return;
        const jobId = Number(msg.jobId);
        if (!Number.isInteger(jobId)) return;
        const pending = state.pending.get(jobId);
        if (!pending) return;
        state.pending.delete(jobId);
        if (msg.ok) pending.resolve(msg);
        else pending.reject(new Error(msg.error || 'worker_decode_failed'));
      };
      worker.onerror = (event) => {
        console.warn('Window decode worker error', {
          kind,
          message: event?.message || null,
        });
        flushDecodeWorkerPending(kind, 'decode_worker_error');
        try {
          worker.terminate();
        } catch (_) {
          // ignore termination error
        }
        if (state.worker === worker) state.worker = null;
      };
      worker.onmessageerror = () => {
        console.warn('Window decode worker message error', { kind });
        flushDecodeWorkerPending(kind, 'decode_worker_message_error');
        try {
          worker.terminate();
        } catch (_) {
          // ignore termination error
        }
        if (state.worker === worker) state.worker = null;
      };
      state.worker = worker;
      return worker;
    }

    function enqueueDecodeJob(kind, { bin, mode, fbMode, wantZ }) {
      const worker = ensureDecodeWorker(kind);
      const state = decodeWorkerStates[kind];
      const jobId = nextWorkerJobId++;
      const promise = new Promise((resolve, reject) => {
        state.pending.set(jobId, { resolve, reject });
        try {
          worker.postMessage({
            type: 'decode',
            jobId,
            bin,
            mode,
            fbMode: fbMode === true,
            wantZ: wantZ === true,
          }, [bin]);
        } catch (err) {
          state.pending.delete(jobId);
          reject(err);
        }
      });
      return { jobId, promise };
    }

    function cancelDecodeJob(kind, jobId, { resolveDropped = false } = {}) {
      if (!Number.isInteger(jobId)) return;
      const state = decodeWorkerStates[kind];
      if (!state) return;
      if (state.worker) {
        try {
          state.worker.postMessage({ type: 'cancel', jobId });
        } catch (_) {
          // no-op: stale job cancellation best-effort
        }
      }
      const pending = state.pending.get(jobId);
      if (!pending) return;
      state.pending.delete(jobId);
      if (resolveDropped) {
        pending.resolve(null);
      } else {
        pending.reject(new Error('decode_job_canceled'));
      }
    }

    function cancelActiveMainDecodeJob() {
      if (!Number.isInteger(activeMainDecodeJobId)) return;
      cancelDecodeJob(DECODE_WORKER_KIND_MAIN, activeMainDecodeJobId, { resolveDropped: true });
      activeMainDecodeJobId = null;
    }

    function decodeWindowPayload(bin, payloadMeta, perfMeta, onInvalidShape, { applyDt = true } = {}) {
      const obj = msgpack.decode(bin);
      if (applyDt) applyServerDt(obj);
      const dataView = obj?.data;
      if (!dataView || !ArrayBuffer.isView(dataView)) {
        if (typeof onInvalidShape === 'function') {
          onInvalidShape({ reason: 'invalid_data_view' });
        }
        return null;
      }
      const shape = resolveWindowShape(obj?.shape);
      if (!shape) {
        if (typeof onInvalidShape === 'function') {
          onInvalidShape(obj?.shape);
        }
        return null;
      }
      const [rows, cols] = shape;
      const valuesI8 = new Int8Array(dataView.buffer, dataView.byteOffset, dataView.byteLength);
      const total = rows * cols;
      if (valuesI8.length < total) {
        if (typeof onInvalidShape === 'function') {
          onInvalidShape({ reason: 'insufficient_data_length', rows, cols, len: valuesI8.length });
        }
        return null;
      }
      const scaleRaw = Number(obj?.scale);
      const scale = Number.isFinite(scaleRaw) ? scaleRaw : null;
      const dtRaw = Number(obj?.dt);
      const dt = Number.isFinite(dtRaw) && dtRaw > 0 ? dtRaw : null;
      return {
        ...payloadMeta,
        shape: [rows, cols],
        valuesI8,
        scale,
        dt,
        quant: resolveWindowQuantMeta(obj),
        __perf: perfMeta || null,
      };
    }

    function buildWindowPayloadFromWorkerDecoded(decoded, payloadMeta, perfMeta, onInvalidShape) {
      if (!decoded || decoded.ok !== true) return null;
      const rows = Math.trunc(Number(decoded.rows));
      const cols = Math.trunc(Number(decoded.cols));
      if (!Number.isFinite(rows) || !Number.isFinite(cols) || rows <= 0 || cols <= 0) {
        if (typeof onInvalidShape === 'function') {
          onInvalidShape({ reason: 'invalid_shape', rows: decoded.rows, cols: decoded.cols });
        }
        return null;
      }
      const total = rows * cols;
      const scaleRaw = Number(decoded.scale);
      const dtRaw = Number(decoded.dt);
      const payload = {
        ...payloadMeta,
        shape: [rows, cols],
        scale: Number.isFinite(scaleRaw) ? scaleRaw : null,
        dt: Number.isFinite(dtRaw) && dtRaw > 0 ? dtRaw : null,
        quant: decoded.quant ?? null,
        __perf: perfMeta || null,
      };
      if (payloadMeta.mode === 'heatmap') {
        if (!(decoded.zBuf instanceof ArrayBuffer)) {
          if (typeof onInvalidShape === 'function') {
            onInvalidShape({ reason: 'missing_z_buffer' });
          }
          return null;
        }
        const zBacking = new Float32Array(decoded.zBuf);
        if (zBacking.length < total) {
          if (typeof onInvalidShape === 'function') {
            onInvalidShape({ reason: 'short_z_buffer', rows, cols, len: zBacking.length });
          }
          return null;
        }
        payload.zBacking = zBacking;
        payload.zRows = buildHeatmapRowsFromBacking(zBacking, rows, cols);
        return payload;
      }

      if (!(decoded.i8Buf instanceof ArrayBuffer)) {
        if (typeof onInvalidShape === 'function') {
          onInvalidShape({ reason: 'missing_i8_buffer' });
        }
        return null;
      }
      const valuesI8 = new Int8Array(decoded.i8Buf);
      if (valuesI8.length < total) {
        if (typeof onInvalidShape === 'function') {
          onInvalidShape({ reason: 'short_i8_buffer', rows, cols, len: valuesI8.length });
        }
        return null;
      }
      payload.valuesI8 = valuesI8;
      return payload;
    }

    function renderWindowPayload(windowPayload) {
      if (!windowPayload) return;
      if (windowPayload.mode === 'wiggle') renderWindowWiggle(windowPayload);
      else renderWindowHeatmap(windowPayload);
    }

    function clampWindowInfoToSectionBounds(windowInfo) {
      if (!Array.isArray(sectionShape) || sectionShape.length < 2) return null;
      const totalTraces = Number(sectionShape[0]);
      const totalSamples = Number(sectionShape[1]);
      if (!Number.isFinite(totalTraces) || totalTraces <= 0) return null;
      if (!Number.isFinite(totalSamples) || totalSamples <= 0) return null;

      let x0 = Math.floor(Math.min(Number(windowInfo.x0), Number(windowInfo.x1)));
      let x1 = Math.floor(Math.max(Number(windowInfo.x0), Number(windowInfo.x1)));
      let y0 = Math.floor(Math.min(Number(windowInfo.y0), Number(windowInfo.y1)));
      let y1 = Math.floor(Math.max(Number(windowInfo.y0), Number(windowInfo.y1)));
      if (!Number.isFinite(x0) || !Number.isFinite(x1) || !Number.isFinite(y0) || !Number.isFinite(y1)) {
        return null;
      }

      x0 = Math.max(0, x0);
      x1 = Math.min(totalTraces - 1, x1);
      y0 = Math.max(0, y0);
      y1 = Math.min(totalSamples - 1, y1);
      if (x1 < x0 || y1 < y0) return null;

      return {
        x0,
        x1,
        y0,
        y1,
        nTraces: x1 - x0 + 1,
        nSamples: y1 - y0 + 1,
      };
    }

    function abortOldestPrefetchIfNeeded() {
      const maxInflight = readPrefetchMaxInflight();
      if (!Number.isFinite(maxInflight) || maxInflight <= 0) return false;

      while (prefetchInflight.size >= maxInflight && prefetchInflight.size > 0) {
        let oldestKey = null;
        let oldestStartedAt = Infinity;
        for (const [key, inflight] of prefetchInflight.entries()) {
          const startedAt = Number(inflight?.startedAt) || 0;
          if (oldestKey == null || startedAt < oldestStartedAt) {
            oldestKey = key;
            oldestStartedAt = startedAt;
          }
        }
        if (oldestKey == null) break;
        const oldest = prefetchInflight.get(oldestKey);
        if (oldest?.ctrl) oldest.ctrl.abort();
        if (Number.isInteger(oldest?.decodeJobId)) {
          cancelDecodeJob(DECODE_WORKER_KIND_PREFETCH, oldest.decodeJobId, { resolveDropped: true });
        }
        prefetchInflight.delete(oldestKey);
      }
      return true;
    }

    function prefetchWindowByRequest(requestContext) {
      if (!readPrefetchEnabled()) return Promise.resolve(null);
      const maxInflight = readPrefetchMaxInflight();
      if (!Number.isFinite(maxInflight) || maxInflight <= 0) return Promise.resolve(null);
      const workerEnabled = readWindowDecodeUseWorker();

      const { params, cacheKey, payloadMeta } = buildWindowRequestArtifacts(requestContext);
      const cached = windowCachePeek(cacheKey);
      if (cached) return Promise.resolve(cached);

      const existing = prefetchInflight.get(cacheKey);
      if (existing?.promise) return existing.promise;

      abortOldestPrefetchIfNeeded();
      const ctrl = new AbortController();
      const startedAt = Date.now();
      windowCacheStats.prefetchStarted += 1;

      let promise = null;
      let decodeJobId = null;
      promise = (async () => {
        try {
          const res = await fetch(`/get_section_window_bin?${params.toString()}`, { signal: ctrl.signal });
          if (!res.ok) {
            console.debug('Prefetch window fetch failed', { status: res.status, cacheKey });
            return null;
          }
          const buf = await res.arrayBuffer();
          if (!isCurrentLmoKey(payloadMeta.lmoKey)) return null;
          let payload = null;
          if (workerEnabled) {
            const decodeJob = enqueueDecodeJob(DECODE_WORKER_KIND_PREFETCH, {
              bin: buf,
              mode: payloadMeta.mode,
              fbMode: payloadMeta.effectiveLayer === 'fbprob',
              wantZ: payloadMeta.mode === 'heatmap',
            });
            decodeJobId = decodeJob.jobId;
            const inflight = prefetchInflight.get(cacheKey);
            if (inflight && inflight.promise === promise) {
              inflight.decodeJobId = decodeJobId;
            }
            const decoded = await decodeJob.promise;
            decodeJobId = null;
            if (!decoded) return null;
            if (!isCurrentLmoKey(payloadMeta.lmoKey)) return null;
            payload = buildWindowPayloadFromWorkerDecoded(
              decoded,
              payloadMeta,
              null,
              (shape) => console.debug('Prefetch unexpected window shape', { cacheKey, shape }),
            );
          } else {
            warnDecodeFallback('cfg.WINDOW_DECODE_USE_WORKER=false');
            const bin = new Uint8Array(buf);
            payload = decodeWindowPayload(
              bin,
              payloadMeta,
              null,
              (shape) => console.debug('Prefetch unexpected window shape', { cacheKey, shape }),
              { applyDt: false },
            );
          }
          if (!payload) return null;
          if (!isCurrentLmoKey(payload.lmoKey)) return null;
          const cachePayload = payload.__perf ? { ...payload, __perf: null } : payload;
          windowCacheSet(cacheKey, cachePayload);
          windowCacheStats.prefetchDone += 1;
          return cachePayload;
        } catch (err) {
          if (Number.isInteger(decodeJobId)) {
            cancelDecodeJob(DECODE_WORKER_KIND_PREFETCH, decodeJobId, { resolveDropped: true });
            decodeJobId = null;
          }
          if (err && err.name === 'AbortError') {
            console.debug('Prefetch aborted', { cacheKey });
            return null;
          }
          console.debug('Prefetch window fetch error', { cacheKey, err });
          return null;
        } finally {
          const active = prefetchInflight.get(cacheKey);
          if (active && active.promise === promise) {
            prefetchInflight.delete(cacheKey);
          }
        }
      })();

      prefetchInflight.set(cacheKey, {
        promise,
        ctrl,
        startedAt,
        decodeJobId: null,
      });
      return promise;
    }

    function schedulePrefetchTask(task) {
      if (typeof task !== 'function') return;
      const run = () => {
        try {
          task();
        } catch (err) {
          console.debug('Prefetch schedule task error', err);
        }
      };
      if (typeof window.requestIdleCallback === 'function') {
        window.requestIdleCallback(() => run(), { timeout: 120 });
        return;
      }
      setTimeout(run, 0);
    }

    function queuePrefetchWindow(requestContext, nextWindowInfo, compareBaseWindow) {
      const baseWindow = clampWindowInfoToSectionBounds(compareBaseWindow || requestContext?.windowInfo);
      if (!baseWindow) return;
      const nextWindow = clampWindowInfoToSectionBounds(nextWindowInfo);
      if (!nextWindow) return;
      if (nextWindow.x0 === baseWindow.x0
        && nextWindow.x1 === baseWindow.x1
        && nextWindow.y0 === baseWindow.y0
        && nextWindow.y1 === baseWindow.y1) {
        return;
      }
      const prefetchContext = {
        ...requestContext,
        windowInfo: nextWindow,
      };
      schedulePrefetchTask(() => {
        void prefetchWindowByRequest(prefetchContext);
      });
    }

    function resolveViewportWindowForPrefetch(fallbackWindow) {
      const dtBase = window.defaultDt ?? defaultDt;
      let x0 = fallbackWindow.x0;
      let x1 = fallbackWindow.x1;
      if (Array.isArray(savedXRange) && savedXRange.length === 2) {
        const xA = Number(savedXRange[0]);
        const xB = Number(savedXRange[1]);
        if (Number.isFinite(xA) && Number.isFinite(xB)) {
          x0 = Math.floor(Math.min(xA, xB));
          x1 = Math.ceil(Math.max(xA, xB));
        }
      }

      let y0 = fallbackWindow.y0;
      let y1 = fallbackWindow.y1;
      if (Array.isArray(savedYRange) && savedYRange.length === 2 && Number.isFinite(dtBase) && dtBase > 0) {
        const yA = Number(savedYRange[0]);
        const yB = Number(savedYRange[1]);
        if (Number.isFinite(yA) && Number.isFinite(yB)) {
          y0 = Math.floor(Math.min(yA, yB) / dtBase);
          y1 = Math.ceil(Math.max(yA, yB) / dtBase);
        }
      }

      return clampWindowInfoToSectionBounds({ x0, x1, y0, y1 });
    }

    function maybePrefetchAroundCurrentViewport(requestContext) {
      if (!readPrefetchEnabled()) return;
      const latestWindow = latestWindowRender
        ? clampWindowInfoToSectionBounds({
          x0: latestWindowRender.x0,
          x1: latestWindowRender.x1,
          y0: latestWindowRender.y0,
          y1: latestWindowRender.y1,
        })
        : null;
      const baseWindow = latestWindow || clampWindowInfoToSectionBounds(requestContext?.windowInfo);
      if (!baseWindow) return;
      const viewportWindow = resolveViewportWindowForPrefetch(baseWindow);
      if (!viewportWindow) return;

      const marginRatio = readPrefetchMarginRatio();
      const spanX = Math.max(1, baseWindow.x1 - baseWindow.x0 + 1);
      const marginX = Math.max(1, Math.floor(spanX * marginRatio));
      const guardX = Math.max(1, Math.floor(spanX * 0.05));
      const shiftX = Math.max(1, spanX - 2 * guardX);
      const nearLeftX = Math.abs(viewportWindow.x0 - baseWindow.x0) <= marginX;
      const nearRightX = Math.abs(baseWindow.x1 - viewportWindow.x1) <= marginX;

      if (nearLeftX) {
        queuePrefetchWindow(requestContext, {
          x0: baseWindow.x0 - shiftX,
          x1: baseWindow.x1 - shiftX,
          y0: baseWindow.y0,
          y1: baseWindow.y1,
        }, baseWindow);
      }
      if (nearRightX) {
        queuePrefetchWindow(requestContext, {
          x0: baseWindow.x0 + shiftX,
          x1: baseWindow.x1 + shiftX,
          y0: baseWindow.y0,
          y1: baseWindow.y1,
        }, baseWindow);
      }

      if (!readPrefetchEnableY()) return;
      const spanY = Math.max(1, baseWindow.y1 - baseWindow.y0 + 1);
      const marginY = Math.max(1, Math.floor(spanY * marginRatio));
      const guardY = Math.max(1, Math.floor(spanY * 0.05));
      const shiftY = Math.max(1, spanY - 2 * guardY);
      const nearTopY = Math.abs(viewportWindow.y0 - baseWindow.y0) <= marginY;
      const nearBottomY = Math.abs(baseWindow.y1 - viewportWindow.y1) <= marginY;

      if (nearTopY) {
        queuePrefetchWindow(requestContext, {
          x0: baseWindow.x0,
          x1: baseWindow.x1,
          y0: baseWindow.y0 - shiftY,
          y1: baseWindow.y1 - shiftY,
        }, baseWindow);
      }
      if (nearBottomY) {
        queuePrefetchWindow(requestContext, {
          x0: baseWindow.x0,
          x1: baseWindow.x1,
          y0: baseWindow.y0 + shiftY,
          y1: baseWindow.y1 + shiftY,
        }, baseWindow);
      }
    }

    window.__svWindowCache = {
      get: windowCacheGet,
      peek: windowCachePeek,
      set: windowCacheSet,
      clear: windowCacheClear,
      stats: windowCacheStats,
    };

    async function fetchWindowAndPlot() {
      if (typeof isCompareModeEnabled === 'function' && isCompareModeEnabled()) {
        return fetchCompareAndPlot();
      }
      D('WINDOW@req', { key1: key1Values?.[parseInt(document.getElementById('key1_slider')?.value || '0', 10)] });
      if (!currentFileId) return;
      if (!sectionShape) {
        await fetchSectionMeta();
        if (!sectionShape) return;
      }
      const slider = document.getElementById('key1_slider');
      if (!slider) return;
      const idx = parseInt(slider.value, 10);
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

      let step_x, step_y;
      if (wantWiggle) {
        step_x = 1; step_y = 1;
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
      const workerEnabled = readWindowDecodeUseWorker();
      D('WINDOW@calc', { mode, step_x, step_y, x0: windowInfo.x0, x1: windowInfo.x1, y0: windowInfo.y0, y1: windowInfo.y1 });
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

      // ★ 加工後配列は保持しないため、上記ショートカットは全て削除
      const transpose = '1';
      const requestContext = {
        fileId: currentFileId,
        key1Val,
        key1Byte: currentKey1Byte,
        key2Byte: currentKey2Byte,
        windowInfo,
        stepX: step_x,
        stepY: step_y,
        requestedLayer,
        effectiveLayer,
        pipelineKey: pipelineKeyNow,
        tapLabel,
        scaling: currentScaling,
        transpose,
        mode,
      };
      const { params, cacheKey, payloadMeta, requestContext: resolvedRequestContext } = buildWindowRequestArtifacts(requestContext);
      cancelActiveMainDecodeJob();
      const cachedPayload = windowCacheGet(cacheKey);
      if (cachedPayload) {
        if (!isCurrentLmoKey(cachedPayload.lmoKey)) return;
        bumpWindowFetchId();
        if (windowFetchCtrl) {
          windowFetchCtrl.abort();
          windowFetchCtrl = null;
        }
        latestSeismicData = null;
        latestWindowRender = cachedPayload;
        hideLoading();
        if (isRelayouting) {
          redrawPending = true;
          return;
        }
        renderWindowPayload(cachedPayload);
        maybePrefetchAroundCurrentViewport(resolvedRequestContext);
        return;
      }

      const requestId = bumpWindowFetchId();
      const perfEnabled = window.SV_PERF === true;
      let tReq0 = null;
      let tRes = null;
      let tBuf = null;
      let tDec0 = null;
      let tDec1 = null;
      let bytes = null;
      let decodeJobId = null;
      showLoading(buildWindowLoadingMessage({
        mode,
        stepX: step_x,
        stepY: step_y,
      }));

      // ---- Abort older in-flight window fetch, then create a new controller
      if (windowFetchCtrl) {
        windowFetchCtrl.abort();
      }
      const ctrl = new AbortController();
      windowFetchCtrl = ctrl;

      try {
        if (perfEnabled) tReq0 = performance.now();
        const res = await fetch(`/get_section_window_bin?${params.toString()}`, { signal: ctrl.signal });
        if (perfEnabled) tRes = performance.now();
        if (!res.ok) {
          console.warn('Window fetch failed', res.status);
          return;
        }
        const buf = await res.arrayBuffer();
        if (perfEnabled) tBuf = performance.now();
        if (perfEnabled) bytes = buf.byteLength;
        if (requestId !== activeWindowFetchId) return; // stale
        if (!isCurrentLmoKey(payloadMeta.lmoKey)) return;

        if (perfEnabled) tDec0 = performance.now();
        let windowPayload = null;
        if (workerEnabled) {
          const decodeJob = enqueueDecodeJob(DECODE_WORKER_KIND_MAIN, {
            bin: buf,
            mode,
            fbMode: effectiveLayer === 'fbprob',
            wantZ: mode === 'heatmap',
          });
          decodeJobId = decodeJob.jobId;
          activeMainDecodeJobId = decodeJobId;
          const decoded = await decodeJob.promise;
          if (activeMainDecodeJobId === decodeJobId) {
            activeMainDecodeJobId = null;
          }
          decodeJobId = null;
          if (!decoded) return;
          if (requestId !== activeWindowFetchId) return;
          if (!isCurrentLmoKey(payloadMeta.lmoKey)) return;
          if (Number.isFinite(decoded?.dt) && decoded.dt > 0) {
            applyServerDt({ dt: decoded.dt });
          }
          windowPayload = buildWindowPayloadFromWorkerDecoded(
            decoded,
            payloadMeta,
            null,
            (shape) => console.warn('Unexpected window shape', shape),
          );
        } else {
          warnDecodeFallback('cfg.WINDOW_DECODE_USE_WORKER=false');
          const bin = new Uint8Array(buf);
          windowPayload = decodeWindowPayload(
            bin,
            payloadMeta,
            null,
            (shape) => console.warn('Unexpected window shape', shape),
          );
        }
        if (perfEnabled) tDec1 = performance.now();
        if (!windowPayload) return;
        if (!isCurrentLmoKey(windowPayload.lmoKey)) return;
        windowPayload.__perf = perfEnabled ? {
          id: requestId,
          mode,
          bytes,
          tReq0,
          tRes,
          tBuf,
          tDec0,
          tDec1,
        } : null;

        if (requestId !== activeWindowFetchId) return; // stale (decode/render phase)
        if (!isCurrentLmoKey(windowPayload.lmoKey)) return;
        const cachePayload = windowPayload.__perf ? { ...windowPayload, __perf: null } : windowPayload;
        windowCacheSet(cacheKey, cachePayload);
        latestSeismicData = null;
        latestWindowRender = windowPayload;
        if (isRelayouting) {      // ドラッグ中なら描画は保留
          redrawPending = true;
          return;
        }
        D('WINDOW@recv', { mode, shape: windowPayload.shape, stepX: windowPayload.stepX, stepY: windowPayload.stepY });
        renderWindowPayload(windowPayload);
        maybePrefetchAroundCurrentViewport(resolvedRequestContext);

      } catch (err) {
        if (Number.isInteger(decodeJobId)) {
          cancelDecodeJob(DECODE_WORKER_KIND_MAIN, decodeJobId, { resolveDropped: true });
          if (activeMainDecodeJobId === decodeJobId) {
            activeMainDecodeJobId = null;
          }
          decodeJobId = null;
        }
        if (err && err.name === 'AbortError') {
          cancelActiveMainDecodeJob();
          console.debug('Window fetch aborted', { requestId });
          return; // canceled on purpose
        }
        if (err && err.message === 'decode_job_canceled') return;
        if (requestId === activeWindowFetchId) console.warn('Window fetch error', err);
      } finally {
        if (Number.isInteger(decodeJobId)) {
          cancelDecodeJob(DECODE_WORKER_KIND_MAIN, decodeJobId, { resolveDropped: true });
          if (activeMainDecodeJobId === decodeJobId) {
            activeMainDecodeJobId = null;
          }
          decodeJobId = null;
        }
        if (windowFetchCtrl === ctrl) windowFetchCtrl = null;
        if (requestId === activeWindowFetchId) hideLoading();
      }
    }

    window.addEventListener('lmo:change', () => {
      if (typeof scheduleWindowFetch === 'function') {
        scheduleWindowFetch();
      }
      if (typeof schedulePickOverlayUpdate === 'function') {
        schedulePickOverlayUpdate();
      }
    });
