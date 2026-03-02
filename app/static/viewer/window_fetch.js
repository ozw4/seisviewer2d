    function maybeFetchIfOutOfWindow() {
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
    const windowPayloadCache = new Map();
    const windowCacheStats = {
      hits: 0,
      misses: 0,
      evicts: 0,
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

    function estimateWindowPayloadBytes(payload) {
      const valuesBytes = Number(payload?.valuesI8?.byteLength) || 0;
      return valuesBytes + 512;
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
      scaling,
      transpose,
      mode,
    }) {
      const enc = (value) => encodeURIComponent(value == null ? '' : String(value));
      return [
        'svwin',
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
        `sc=${enc(scaling)}`,
        `tr=${enc(transpose)}`,
        `mode=${enc(mode)}`,
      ].join('|');
    }

    window.__svWindowCache = {
      get: windowCacheGet,
      peek: windowCachePeek,
      set: windowCacheSet,
      clear: windowCacheClear,
      stats: windowCacheStats,
    };

    async function fetchWindowAndPlot() {
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
      const params = new URLSearchParams({
        file_id: currentFileId,
        key1: String(key1Val),
        key1_byte: String(currentKey1Byte),
        key2_byte: String(currentKey2Byte),
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
      const transpose = '1';
      params.set('transpose', transpose);
      params.set('scaling', currentScaling);
      const cacheKey = buildWindowCacheKey({
        fileId: currentFileId,
        key1: key1Val,
        key1Byte: currentKey1Byte,
        key2Byte: currentKey2Byte,
        x0: windowInfo.x0,
        x1: windowInfo.x1,
        y0: windowInfo.y0,
        y1: windowInfo.y1,
        stepX: step_x,
        stepY: step_y,
        requestedLayer,
        effectiveLayer,
        pipelineKey: tapLabel ? pipelineKeyNow : null,
        tapLabel,
        scaling: currentScaling,
        transpose,
        mode,
      });
      const cachedPayload = windowCacheGet(cacheKey);
      if (cachedPayload) {
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
        if (cachedPayload.mode === 'wiggle') renderWindowWiggle(cachedPayload);
        else renderWindowHeatmap(cachedPayload);
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
        const bin = new Uint8Array(buf);
        if (perfEnabled) bytes = bin.byteLength;
        if (requestId !== activeWindowFetchId) return; // stale

        if (perfEnabled) tDec0 = performance.now();
        const obj = msgpack.decode(bin);
        if (perfEnabled) tDec1 = performance.now();
        applyServerDt(obj);

        // Int8 のまま保持（Float32生成しない）
        const valuesI8 = new Int8Array(obj.data.buffer);
        const shapeRaw = Array.isArray(obj.shape) ? obj.shape : Array.from(obj.shape ?? []);
        if (shapeRaw.length !== 2) {
          console.warn('Unexpected window shape', obj.shape);
          return;
        }
        const rows = Number(shapeRaw[0]);
        const cols = Number(shapeRaw[1]);

        const quantMeta = obj.quant || (
          (obj.lo !== undefined && obj.hi !== undefined)
            ? { mode: obj.method || 'linear', lo: obj.lo, hi: obj.hi, mu: obj.mu ?? 255 }
            : (obj.scale != null ? { scale: obj.scale } : null)
        );

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
          valuesI8,          // Int8保持
          scale: obj.scale,  // 後段で /scale
          quant: quantMeta,
          mode,
          __perf: perfEnabled ? {
            id: requestId,
            mode,
            bytes,
            tReq0,
            tRes,
            tBuf,
            tDec0,
            tDec1,
          } : null,
        };

        if (requestId !== activeWindowFetchId) return; // stale (decode/render phase)
        const cachePayload = windowPayload.__perf ? { ...windowPayload, __perf: null } : windowPayload;
        windowCacheSet(cacheKey, cachePayload);
        latestSeismicData = null;
        latestWindowRender = windowPayload;
        if (isRelayouting) {      // ドラッグ中なら描画は保留
          redrawPending = true;
          return;
        }
        D('WINDOW@recv', { mode, shape: windowPayload.shape, stepX: windowPayload.stepX, stepY: windowPayload.stepY });
        if (mode === 'wiggle') renderWindowWiggle(windowPayload);
        else renderWindowHeatmap(windowPayload);

      } catch (err) {
        if (err && err.name === 'AbortError') {
          console.debug('Window fetch aborted', { requestId });
          return; // canceled on purpose
        }
        if (requestId === activeWindowFetchId) console.warn('Window fetch error', err);
      } finally {
        if (windowFetchCtrl === ctrl) windowFetchCtrl = null;
        if (requestId === activeWindowFetchId) hideLoading();
      }
    }
