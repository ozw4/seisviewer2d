



    let __dbg = { enabled: true };
    // 連続重複を抑制したいタグ（必要なら増やせる）
      const DEDUP_TAGS = new Set(['RENDER@wiggle:shapes']);
    const __lastLogKeyByTag = new Map();
    function __stableStringify(v) {
        if (v && typeof v === 'object') {
            if (Array.isArray(v)) return `[${v.map(__stableStringify).join(',')}]`;
            const keys = Object.keys(v).sort();
            return `{${keys.map(k => JSON.stringify(k) + ':' + __stableStringify(v[k])).join(',')}}`;
          }
        return JSON.stringify(v);
      }
    function D(tag, obj) {
      if (!__dbg.enabled) return;
      // 直前と同じ内容ならスキップ（指定タグのみ）
        if (DEDUP_TAGS.has(tag)) {
            let key;
            try { key = __stableStringify(obj); } catch { key = null; }
            const last = __lastLogKeyByTag.get(tag);
            if (last === key) return;
            __lastLogKeyByTag.set(tag, key);
          }
      try {
          console.debug(`🧭 ${tag}`, JSON.parse(JSON.stringify(obj)));
        } catch {
            console.debug(`🧭 ${tag}`, obj);
          }
    }


    function samplePicks(arr, k = 6) {
      return (arr || []).slice(0, k).map(p => ({ tr: Math.round(p.trace), t: +p.time }));
    }
    // 現在の可視Xレンジ（整数トレース）と viewer 状態サマリ
    function visibleXRng() {
      const gd = document.getElementById('plot');
      const xr = gd?._fullLayout?.xaxis?.range || savedXRange || [renderedStart ?? 0, renderedEnd ?? 0];
      const x0 = Math.floor(Math.min(xr[0], xr[1]));
      const x1 = Math.ceil(Math.max(xr[0], xr[1]));
      return [x0, x1];
    }
    function viewerState(tag = 'STATE') {
      const [x0, x1] = visibleXRng();
      const mode = latestWindowRender?.mode || 'none';
      const st = {
        key1: key1Values?.[parseInt(document.getElementById('key1_slider')?.value || '0', 10)],
        layer: document.getElementById('layerSelect')?.value,
        mode, stepX: latestWindowRender?.stepX, stepY: latestWindowRender?.stepY,
        x0, x1, renderedStart, renderedEnd, savedXRange, picksCount: picks?.length || 0
      };
      D(tag, st);
      return st;
    }
    // ローカル→サーバ疑い時に呼べる即席ダンプ

    var currentScaling = 'amax';
    var sectionShape = null;
    var renderedStart = null;
    var renderedEnd = null;
    var picks = [];
    var predictedPicks = [];
    const fbPredCache = new Map(); // key: "key1|layer|pipelineKey"
    var currentFbKey = null;
    var fbPredReqId = 0;
    var downsampleFactor = 1;
    var isPickMode = false;
    var linePickStart = null;
    var deleteRangeStart = null;

    let uiResetNonce = 0;
    function currentUiRevision() {
      const sel = document.getElementById('layerSelect');
      const layer = sel ? sel.value : 'raw';
      const slider = document.getElementById('key1_slider');
      const idx = slider ? parseInt(slider.value, 10) : 0;
      const key1Val = key1Values[idx];
      const pKey = window.latestPipelineKey || '';
      return `rev:${currentFileId}|${key1Val}|${layer}|${pKey}|${uiResetNonce}`;
    }


    // --- 追加：ハンドラ ---
    function onKey1Input() {
      updateKey1Display();
      fetchAndPlotDebounced();        // 入力が止まってから実行
    }

    async function onKey1Change() {
      updateKey1Display();
      const debounced = ensureFlushPickOpsDebounced();
      if (typeof debounced?.flush === 'function') {
        await debounced.flush();
      } else {
        await flushPickOps();
      }
      fetchAndPlotDebounced.flush();
    }



    function snapshotAxesRangesFromDOM() {
      const gd = document.getElementById('plot');
      const xa = gd?._fullLayout?.xaxis;
      const ya = gd?._fullLayout?.yaxis;
      if (xa && Array.isArray(xa.range) && xa.range.length === 2) {
        savedXRange = [xa.range[0], xa.range[1]];
      }
      if (ya && Array.isArray(ya.range) && ya.range.length === 2) {
        const y0 = ya.range[0], y1 = ya.range[1];
        // 上下逆レンジでも扱えるように max/min 順で保存
        savedYRange = y0 > y1 ? [y0, y1] : [y1, y0];
      }
    }

    function filenameFromContentDisposition(disposition) {
      if (!disposition) return null;
      const utfMatch = /filename\*=UTF-8''([^;\n]+)/i.exec(disposition);
      if (utfMatch && utfMatch[1]) {
        try {
          return decodeURIComponent(utfMatch[1]);
        } catch (err) {
          console.warn('Failed to decode UTF-8 filename from Content-Disposition:', err);
          return utfMatch[1];
        }
      }
      const match = /filename="?([^";]+)"?/i.exec(disposition);
      return match && match[1] ? match[1] : null;
    }

    // 手動ピック -> インデックスベクトル（-1は欠損）
    function buildPickIndexVector(picksArr, nTraces, nSamples, dt) {
      const last = new Map();
      for (const p of (picksArr || [])) {
        const tr = Math.round(p.trace);
        if (tr >= 0 && tr < nTraces && Number.isFinite(p.time)) last.set(tr, p.time);
      }
      const vec = new Int32Array(nTraces);
      vec.fill(-1);
      for (const [tr, t] of last) {
        let idx = Math.round(t / dt);
        if (!Number.isFinite(idx) || idx < 0 || idx >= nSamples) idx = -1;
        vec[tr] = idx;
      }
      return vec;
    }

    function saveBlob(uint8OrBlob, filename) {
        const blob = (uint8OrBlob instanceof Blob)
          ? uint8OrBlob
          : new Blob([uint8OrBlob], { type: 'application/octet-stream' });

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename || 'download.bin';
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }

    // .npy エンコーダ（1D/2D両対応）
    function npyEncode(data, shape, descr = '<i4') {
      const magic = new Uint8Array([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]); // \x93NUMPY
      const ver = new Uint8Array([0x01, 0x00]); // v1.0
      const shapeTxt = `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`;

      // v1.0 ヘッダは Python 辞書表現・末尾カンマなし・改行で終端
      let headerDict =
        "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shapeTxt + ", }";
      // ↑ 末尾カンマを削る
      headerDict = headerDict.replace(/,\s*}$/, ' }');

      // 最後に改行を必ず入れる
      let header = headerDict + '\n';

      // 16B アライン（magic+ver+hlen(2B)+header の合計が16の倍数になるようパディング）
      const fixed = magic.length + ver.length + 2;
      const pad = (16 - ((fixed + header.length) % 16)) % 16;
      header += ' '.repeat(pad);

      const hbytes = new TextEncoder().encode(header);
      const hlenLE = new Uint8Array(2);
      new DataView(hlenLE.buffer).setUint16(0, hbytes.length, true);

      const out = new Uint8Array(magic.length + ver.length + 2 + hbytes.length + data.byteLength);
      let o = 0;
      out.set(magic, o); o += magic.length;
      out.set(ver, o); o += ver.length;
      out.set(hlenLE, o); o += 2;
      out.set(hbytes, o); o += hbytes.length;
      out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), o);
      return out;
    }


    // ★唯一のエクスポータ：全 key1 を [K, Ntr] 行列で書き出し
    async function exportAllPickIndexMatrixNpy() {
      if (!sectionShape || sectionShape.length < 2) {
        alert('Section shape is unknown yet.');
        return;
      }
      if (!Array.isArray(key1Values) || key1Values.length === 0) {
        alert('key1 values not loaded.');
        return;
      }

      if (!currentFileId) {
        alert('file_id not loaded.');
        return;
      }

      const nTraces = sectionShape[0];
      const nSamples = sectionShape[1];
      const dt = (window.defaultDt ?? defaultDt);
      const K = key1Values.length;

      const buf = new Int32Array(K * nTraces);
      buf.fill(-1);

      for (let i = 0; i < K; i++) {
        const key1Val = key1Values[i];
        if (key1Val === undefined) continue;
        const url = `/picks?file_id=${encodeURIComponent(currentFileId)}&key1=${encodeURIComponent(key1Val)}&key1_byte=${currentKey1Byte}&key2_byte=${currentKey2Byte}`;
        const r = await fetch(url);
        if (!r.ok) continue;
        const j = await r.json();
        const arr = (j.picks || []).map(p => ({ trace: (p.trace | 0), time: +p.time }));
        const vec = buildPickIndexVector(arr, nTraces, nSamples, dt);
        buf.set(vec, i * nTraces);
      }

      const npy = npyEncode(buf, [K, nTraces], '<i4');
      const base = currentFileName || currentFileId || 'picks';
      saveBlob(npy, `manual_picks_idx_ALL_${base}.npy`);
    }

    let suppressRelayout = false;       // ignore relayouts we cause internally
    let isRelayouting = false;          // true while user is actively adjusting viewport
    let pendingResetFetch = false;      // reset event arrived while dragging; run once after settle
    let lastImmediateWindowFetchAt = 0; // immediate-reset fetch dedup
    const RESET_FETCH_DEDUP_MS = 100;
    let forceFullExtentOnce = false;    // next window calc uses full extent with no padding
    let pickOverlayRaf = 0;
    let pickOverlayDirty = false;

    // 追加：現在のFB計算に紐づくレイヤ/パイプラインキー
    let currentFbLayer = 'raw';
    let currentFbPipelineKey = null;

    let dragOverride = null; // 一時上書き（'pan' を入れる）

    function getBaseDragMode() {
      return (dragBase === 'pan') ? 'pan' : 'zoom';
    }

    function setBaseDragMode(mode) {
      const next = (mode === 'pan') ? 'pan' : 'zoom';
      if (dragBase !== next) {
        dragBase = next;
        try { localStorage.setItem(cfg.LS_KEYS.DRAG_BASE, next); } catch (_) { }
        applyDragMode();
      }
    }

    function getOverrideDragMode() {
      if (dragOverride) return dragOverride;
      if (isPickMode) return false;
      return null;
    }

    function effectiveDragMode() {
      const over = getOverrideDragMode();
      if (over !== null && over !== undefined) return over;
      return getBaseDragMode();
    }
    function applyDragMode() {
      const plotDiv = document.getElementById('plot');
      if (!plotDiv) return;

      // 既存関数：'zoom' or 'pan' を返す（ピック中は false を返していた）
      const dm = effectiveDragMode();

      //  - ピック中（dm===false）のときは dragmode は 'zoom' のままにして
      //    x/y の fixedrange を true にしてドラッグ無効化
      //  - Alt パンや通常時は fixedrange を false に戻す
      if (dm === false) {
        // ピック中（ドラッグ禁止）
        safeRelayout(plotDiv, {
          dragmode: 'zoom',                  // 値は何でもOKだが合法値を入れておく
          'xaxis.fixedrange': true,
          'yaxis.fixedrange': true,
        });
        applyDragMode._last = 'pick-locked';
        return;
      }

      // 通常 or Alt パン（ドラッグ可能）
      if (applyDragMode._last === dm) return; // 余計な relayout を避ける
      applyDragMode._last = dm;
      safeRelayout(plotDiv, {
        dragmode: dm,                        // 'zoom' or 'pan'
        'xaxis.fixedrange': false,
        'yaxis.fixedrange': false,
      });
    }

    function safeRelayout(gd, props) {
      suppressRelayout = true;
      try {
        const result = Plotly.relayout(gd, props);
        if (result && typeof result.finally === 'function') {
          return result.finally(() => {
            suppressRelayout = false;
          });
        }
        suppressRelayout = false;
        return result;
      } catch (err) {
        suppressRelayout = false;
        throw err;
      }
    }

    function schedulePickOverlayUpdate() {
      pickOverlayDirty = true;
      if (pickOverlayRaf !== 0) return;
      pickOverlayRaf = requestAnimationFrame(() => {
        pickOverlayRaf = 0;
        flushPickOverlayUpdate();
        if (pickOverlayDirty && !isRelayouting) {
          schedulePickOverlayUpdate();
        }
      });
    }

    function flushPickOverlayUpdate() {
      if (!pickOverlayDirty) return;
      if (isRelayouting) return;

      const plotDiv = document.getElementById('plot');
      if (!plotDiv) {
        pickOverlayDirty = false;
        return;
      }

      let xMin = null;
      let xMax = null;

      const xaRange = plotDiv?._fullLayout?.xaxis?.range;
      if (
        Array.isArray(xaRange) &&
        xaRange.length === 2 &&
        Number.isFinite(xaRange[0]) &&
        Number.isFinite(xaRange[1])
      ) {
        xMin = Math.floor(Math.min(xaRange[0], xaRange[1]));
        xMax = Math.ceil(Math.max(xaRange[0], xaRange[1]));
      }

      if (Number.isFinite(renderedStart) && Number.isFinite(renderedEnd)) {
        const rs = Math.floor(Math.min(renderedStart, renderedEnd));
        const re = Math.ceil(Math.max(renderedStart, renderedEnd));
        if (Number.isFinite(xMin) && Number.isFinite(xMax)) {
          xMin = Math.max(xMin, rs);
          xMax = Math.min(xMax, re);
        } else {
          xMin = rs;
          xMax = re;
        }
      }

      if (!(Number.isFinite(xMin) && Number.isFinite(xMax) && xMin <= xMax)) {
        const wx0 = Number(latestWindowRender?.x0);
        const wx1 = Number(latestWindowRender?.x1);
        if (Number.isFinite(wx0) && Number.isFinite(wx1)) {
          xMin = Math.floor(Math.min(wx0, wx1));
          xMax = Math.ceil(Math.max(wx0, wx1));
        }
      }

      if (!(Number.isFinite(xMin) && Number.isFinite(xMax) && xMin <= xMax)) {
        pickOverlayDirty = false;
        return;
      }

      const showPred = !!document.getElementById('showFbPred')?.checked;
      const newShapes = buildPickShapes({
        manualPicks: picks,
        predicted: showPred ? predictedPicks : [],
        xMin,
        xMax,
        showPredicted: showPred,
      });

      pickOverlayDirty = false;
      safeRelayout(plotDiv, { shapes: newShapes });
    }

    // 統一キー関数（FB予測キャッシュ用）
    function fbCacheKey(fileId, k1, layer, pKey) {
      return `${fileId}|${k1}|${layer}|${pKey ?? 'raw'}`;
    }

    // 一時上書きの適用
    function setAltPan(on) {
      const next = on ? 'pan' : null;
      if (dragOverride !== next) {
        dragOverride = next;
        applyDragMode();
      }
    }

    function currentDesiredMode() {
      const win = currentVisibleWindow();
      const plotDiv = document.getElementById('plot');
      if (!win || !plotDiv) return null;
      const wantWig = wantWiggleForWindow({
        tracesVisible: win.nTraces,
        samplesVisible: win.nSamples,
        widthPx: plotDiv.clientWidth || 1,
      });
      return wantWig ? 'wiggle' : 'heatmap';
    }

    function isResetRelayout(ev) {
      if (!ev || typeof ev !== 'object') return false;

      if (ev['xaxis.autorange'] === true || ev['yaxis.autorange'] === true) {
        return true;
      }

      for (const key of Object.keys(ev)) {
        if (!/^(xaxis|yaxis)(\d+)?\.autorange$/.test(key)) continue;
        if (ev[key] === true) return true;
      }

      return false;
    }

    function requestWindowFetch({ immediate = false } = {}) {
      const plotDiv = document.getElementById('plot');
      if (!plotDiv) return;

      if (!immediate) {
        scheduleWindowFetch();
        return;
      }

      const now = Date.now();
      if (now - lastImmediateWindowFetchAt < RESET_FETCH_DEDUP_MS) return;
      lastImmediateWindowFetchAt = now;

      if (typeof scheduleWindowFetch?.cancel === 'function') {
        scheduleWindowFetch.cancel();
      }
      if (typeof scheduleWindowFetch?.flush === 'function') {
        scheduleWindowFetch.flush();
        return;
      }

      const result = fetchWindowAndPlot();
      if (result && typeof result.catch === 'function') {
        result.catch((err) => console.warn('Window fetch failed', err));
      }
    }

    function flushPendingResetFetchIfNeeded() {
      if (!pendingResetFetch) return;
      if (suppressRelayout || isRelayouting) return;
      pendingResetFetch = false;
      checkModeFlipAndRefetch({ immediate: true });
    }

    function checkModeFlipAndRefetch({ immediate = false } = {}) {
        const desired = currentDesiredMode();
        if (!desired) return;
          const cur = (latestWindowRender && latestWindowRender.mode) || null;
        const plotDiv = document.getElementById('plot');
        const win = currentVisibleWindow();
        if (!plotDiv || !win) return;

          if (desired === 'wiggle') {
              // wiggle は step=1 前提。モード不一致 or step不一致 or ウィンドウ外なら再フェッチ
                const needFresh =
                    !latestWindowRender ||
                    cur !== 'wiggle' ||
                    latestWindowRender.stepX !== 1 ||
                    latestWindowRender.stepY !== 1 ||
                    latestWindowRender.x0 > win.x0 || latestWindowRender.x1 < win.x1 ||
                    latestWindowRender.y0 > win.y0 || latestWindowRender.y1 < win.y1;
              if (needFresh) requestWindowFetch({ immediate });
              return;
            }

          // heatmap の場合：現在の可視窓と描画サイズから必要 step を再計算して比較
          const { step_x, step_y } = computeStepsForWindow({
              tracesVisible: win.nTraces,
              samplesVisible: win.nSamples,
              widthPx: plotDiv.clientWidth || 1,
              heightPx: plotDiv.clientHeight || 1,
            });

        const needFresh =
            !latestWindowRender ||
            cur !== 'heatmap' ||
            latestWindowRender.stepX !== step_x ||
            latestWindowRender.stepY !== step_y ||
            latestWindowRender.x0 > win.x0 || latestWindowRender.x1 < win.x1 ||
            latestWindowRender.y0 > win.y0 || latestWindowRender.y1 < win.y1;

        if (needFresh) requestWindowFetch({ immediate });
    }

    // （任意：すでに入れているならそのままでOK）


    // 入力系にフォーカスがある時は無効
    function canUseGlobalHotkey() {
      const el = document.activeElement;
      const tag = el?.tagName;
      return !(tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el?.isContentEditable);
    }

    // Alt 押してる間だけ pan
    window.addEventListener('keydown', (e) => {
      if (!canUseGlobalHotkey()) return;
      if (e.key === 'Alt' || e.altKey) setAltPan(true);
    });
    window.addEventListener('keyup', (e) => {
      if (e.key === 'Alt' || !e.altKey) setAltPan(false);
    });

    // 取りこぼし対策
    window.addEventListener('blur', () => setAltPan(false));
    document.addEventListener('visibilitychange', () => { if (document.hidden) setAltPan(false); });
    window.addEventListener('pointerup', (e) => { if (!e.altKey) setAltPan(false); });

    function applyServerDt(obj) {
      const dtSec =
        obj && typeof obj.dt === 'number' && isFinite(obj.dt) && obj.dt > 0
          ? obj.dt
          : null;

      if (dtSec === null) return;

      const prev =
        typeof window.defaultDt === 'number' && isFinite(window.defaultDt)
          ? window.defaultDt
          : defaultDt;

      // dt が実際に変わった時だけ更新＆Yレンジ初期化
      if (!Number.isFinite(prev) || Math.abs(prev - dtSec) > 1e-12) {
        defaultDt = dtSec;
        window.defaultDt = dtSec;
        cfg.setDefaultDt(dtSec);
        savedYRange = null;
      }
    }

    const COLORMAPS = {
      Greys: 'Greys',
      RdBu: 'RdBu',
      BWR: [[0, 'blue'], [0.5, 'white'], [1, 'red']],
      Cividis: 'Cividis',
      Jet: 'Jet',
    };
    window.COLORMAPS = COLORMAPS;

    (function () {
      const saved = localStorage.getItem('gain');
      const el = document.getElementById('gain');
      const disp = document.getElementById('gain_display');
      if (el && disp) {
        const val = saved !== null ? parseFloat(saved) : parseFloat(el.value);
        el.value = val; disp.textContent = `${val}×`;
      }
    })();

    (function () {
      const sel = document.getElementById('colormap');
      const chk = document.getElementById('cmReverse');
      if (sel) {
        const saved = localStorage.getItem('colormap');
        if (saved) sel.value = saved;
      }
      if (chk) {
        const savedRev = localStorage.getItem('cmReverse');
        if (savedRev !== null) chk.checked = savedRev === 'true';
      }
    })();

    (function restoreFbUi() {
      const sig = localStorage.getItem('sigma_ms_max');
      if (sig !== null) {
        const s = document.getElementById('sigma_ms_max');
        if (s) s.value = parseFloat(sig);
      }
      const pm = localStorage.getItem('pick_method');
      if (pm) {
        const sel = document.getElementById('pick_method');
        if (sel) sel.value = pm;
      }
      const sh = localStorage.getItem('showFbPred');
      if (sh !== null) {
        const c = document.getElementById('showFbPred');
        if (c) c.checked = (sh === 'true');
      }
    })();

    (function restoreSnapUi() {
      const m = localStorage.getItem('snap_mode');
      if (m) {
        const el = document.getElementById('snap_mode');
        if (el) el.value = m;
      }
      const w = localStorage.getItem('snap_ms');
      if (w) {
        const el = document.getElementById('snap_ms');
        if (el) el.value = w;
      }
      const r = localStorage.getItem('snap_refine');
      if (r) {
        const el = document.getElementById('snap_refine');
        if (el) el.value = r;
      }
    })();

    // ★★★ FB確率取得：レイヤ/パイプライン対応（既存のジョブAPIを使用）
    function getCachedFbEntry(cacheKey, method, sigmaMsMax) {
      const entry = fbPredCache.get(cacheKey);
      if (!entry) return null;
      if (entry.method !== method) return null;
      if (Math.abs(entry.sigma_ms_max - sigmaMsMax) > 1e-9) return null;
      return entry;
    }

    function installUnifiedClickRouter(plotDiv) {
      if (!plotDiv || plotDiv.__unifiedClickRouter) return;

      const onClick = (e) => {
        if (e.detail > 1) return; // keep Plotly double-click reset
        if (e.button !== 0) return;
        if (!isPickMode) return;
        if (dragOverride === 'pan' || e.altKey) return;
        if (isRelayouting) return;

        const activeEl = document.activeElement;
        if (activeEl && activeEl !== document.body) {
          const tag = activeEl.tagName;
          if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || activeEl.isContentEditable) {
            return;
          }
        }

        const env = getPlotEnv();
        if (!env) return;
        const { rect, m } = env;
        const innerX = e.clientX - rect.left;
        const innerY = e.clientY - rect.top;
        if (innerX < m.l || innerX > m.l + m.w || innerY < m.t || innerY > m.t + m.h) {
          return;
        }

        const tr = traceAtPixel(e.clientX);
        const tSec = timeAtPixel(e.clientY);
        if (!Number.isFinite(tr) || !Number.isFinite(tSec)) return;

        e.stopImmediatePropagation();
        e.stopPropagation();
        e.preventDefault();

        const maybePromise = handlePickNormalized({
          trace: snapTraceFromDataX(tr),
          time: snapTimeFromDataY(tSec),
          shiftKey: !!e.shiftKey,
          ctrlKey: !!e.ctrlKey,
          altKey: !!e.altKey,
        });
        if (maybePromise && typeof maybePromise.catch === 'function') {
          maybePromise.catch(err => console.warn('handlePickNormalized failed', err));
        }
      };

      const remover = () => {
        plotDiv.removeEventListener('click', onClick, true);
        if (plotDiv.__unifiedClickRouter === remover) {
          plotDiv.__unifiedClickRouter = null;
        }
      };
      plotDiv.addEventListener('click', onClick, { capture: true });
      plotDiv.__unifiedClickRouter = remover;
    }

    function attachPickListeners(plotDiv) {
      // relayout ハンドラは一度だけインストール
      installPlotlyViewportHandlersOnce();

      // 既存の pick/hover ハンドラを外す
      plotDiv.removeAllListeners('plotly_click');
      plotDiv.removeAllListeners('plotly_hover');
      plotDiv.removeAllListeners('plotly_unhover');

      if (plotDiv._genericClickHandler) {
        plotDiv.removeEventListener('click', plotDiv._genericClickHandler);
        plotDiv._genericClickHandler = null;
      }
      if (plotDiv._captureShiftHandler) {
        plotDiv.removeEventListener('click', plotDiv._captureShiftHandler, true);
        plotDiv._captureShiftHandler = null;
      }

      // 黒いホバーボックスと同じ値を保持
      plotDiv.on('plotly_hover', (e) => {
        const p = e?.points?.[0];
        if (!p) return;
        lastHover = {
          x: p.x,
          y: p.y,
          meta: Number.isFinite(p.data?.meta) ? p.data.meta : null,
          t: performance.now(),
        };
      });
      plotDiv.on('plotly_unhover', () => { lastHover = null; });

      installUnifiedClickRouter(plotDiv);
    }

    function recomputeFbPicks() {
      const slider = document.getElementById('key1_slider');
      if (!slider) return;
      const idx0 = parseInt(slider.value, 10);
      const keyAtNow = key1Values[idx0];
      const layerNow = (document.getElementById('layerSelect')?.value) || 'raw';
      const pKeyNow = window.latestPipelineKey || null;
      const method = document.getElementById('pick_method').value;
      const sigma = Number(document.getElementById('sigma_ms_max').value) || 20;
      const cacheKeyStr = fbCacheKey(currentFileId, keyAtNow, layerNow, pKeyNow);
      const cached = getCachedFbEntry(cacheKeyStr, method, sigma);
      if (cached) {
        predictedPicks = (cached.picks || []).slice();
        currentFbKey = keyAtNow;
        currentFbLayer = layerNow;
        currentFbPipelineKey = pKeyNow;
        renderLatestView();
        return;
      }
      predictFromFb();
    }

    function onSigmaChange() {
      localStorage.setItem('sigma_ms_max', document.getElementById('sigma_ms_max').value);
      recomputeFbPicks();
    }
    function onPickMethodChange() {
      localStorage.setItem('pick_method', document.getElementById('pick_method').value);
      recomputeFbPicks();
    }

    async function predictFromFb() {
      const idx0 = parseInt(document.getElementById('key1_slider').value, 10);
      const keyAtStart = key1Values[idx0];
      const layerAtStart = (document.getElementById('layerSelect')?.value) || 'raw';
      const pipelineKeyAtStart = window.latestPipelineKey || null;
      const method = document.getElementById('pick_method').value;
      const sigmaMax = Number(document.getElementById('sigma_ms_max').value) || 20;
      const cacheKeyStr = fbCacheKey(currentFileId, keyAtStart, layerAtStart, pipelineKeyAtStart);

      const cached = getCachedFbEntry(cacheKeyStr, method, sigmaMax);
      const reqToken = ++fbPredReqId;
      const btn = document.getElementById('predictFbBtn');
      if (btn) btn.disabled = true;

      try {
        if (cached) {
          predictedPicks = (cached.picks || []).slice();
          currentFbKey = keyAtStart;
          currentFbLayer = layerAtStart;
          currentFbPipelineKey = pipelineKeyAtStart;
          renderLatestView();
          return;
        }

        const body = {
          file_id: currentFileId,
          key1: keyAtStart,
          key1_byte: currentKey1Byte,
          key2_byte: currentKey2Byte,
          method,
          sigma_ms_max: sigmaMax,
        };
        if (layerAtStart && layerAtStart !== 'raw' && pipelineKeyAtStart) {
          body.pipeline_key = pipelineKeyAtStart;
          body.tap_label = layerAtStart;
        }

        const res = await fetch('/fbpick_predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          let message = `fbpick_predict failed (${res.status})`;
          try {
            const text = await res.text();
            if (text) {
              try {
                const parsed = JSON.parse(text);
                if (parsed && typeof parsed.detail === 'string') {
                  message = parsed.detail;
                } else {
                  message = text;
                }
              } catch (parseErr) {
                message = text;
              }
            }
          } catch (readErr) {
            // ignore body read errors
          }
          throw new Error(message);
        }

        const data = await res.json();
        const picks = Array.isArray(data?.picks)
          ? data.picks.map(p => ({ trace: p.trace, time: p.time }))
          : [];

        const idxNow = parseInt(document.getElementById('key1_slider').value, 10);
        const keyNow = key1Values[idxNow];
        const layerNow = (document.getElementById('layerSelect')?.value) || 'raw';
        const pipelineKeyNow = window.latestPipelineKey || null;
        if (reqToken !== fbPredReqId ||
          keyNow !== keyAtStart ||
          layerNow !== layerAtStart ||
          pipelineKeyNow !== pipelineKeyAtStart) {
          return;
        }

        predictedPicks = picks.slice();
        currentFbKey = keyAtStart;
        currentFbLayer = layerAtStart;
        currentFbPipelineKey = pipelineKeyAtStart;
        fbPredCache.set(cacheKeyStr, {
          picks: picks.slice(),
          method,
          sigma_ms_max: sigmaMax,
        });

        renderLatestView();
      } finally {
        if (btn) btn.disabled = false;
      }
    }

    function onScalingChange() {
      const sel = document.getElementById('scalingMode');
      const rawValue = (sel && typeof sel.value === 'string') ? sel.value.toLowerCase() : 'amax';
      currentScaling = rawValue === 'tracewise' ? 'tracewise' : 'amax';
      if (sel) sel.value = currentScaling;
      try { localStorage.setItem('scaling_mode', currentScaling); } catch (_) {}
      if (windowFetchCtrl) {
        try { windowFetchCtrl.abort(); } catch (_) {}
        windowFetchCtrl = null;
      }
      scheduleWindowFetch();
    }

    function onWiggleDensityChange() {
      const el = document.getElementById('wiggle_density');
      const v = parseFloat(el?.value);
      if (!Number.isFinite(v) || v <= 0) return;
      const stored = cfg.setWiggleDensity(v);
      if (el) el.value = stored.toFixed(2);
      WIGGLE_DENSITY_THRESHOLD = stored;
      // Apply now: re-evaluate window mode and redraw
      renderLatestView();
      scheduleWindowFetch();
    }

    // Restore UI on load
    (function restoreWiggleUi() {
        const el = document.getElementById('wiggle_density');
        if (!el) return;
        const min = parseFloat(el.min) || 0.02;
        const max = parseFloat(el.max) || 0.30;
        let v;
        if (window.cfg && typeof window.cfg.getWiggleDensity === 'function') {
            v = window.cfg.getWiggleDensity();
          } else {
            const saved = localStorage.getItem('wiggle_density');
            const p = parseFloat(saved);
            v = Number.isFinite(p) ? p : (parseFloat(el.value) || 0.10);
          }
        v = Math.min(max, Math.max(min, v));
        el.value = v.toFixed(2);
        WIGGLE_DENSITY_THRESHOLD = v;
      })();

    (function hookFetchOnce() {
        if (window.__fetchHooked) return;
        window.__fetchHooked = true;
        const orig = window.fetch;
        window.fetch = async function (...args) {
          const url = String(args?.[0] || '');
          const method = (args?.[1]?.method || 'GET').toUpperCase();
          const t0 = performance.now();
          try {
            const res = await orig.apply(this, args);
            if (/\/picks(\?|$)/.test(url)) {
              console.debug(`📡 FETCH ${method} ${url} → ${res.status} (${(performance.now() - t0).toFixed(0)}ms)`);
              console.trace('  ↳ stack @picks'); // 重要：誰が呼んだか
            } else if (/get_section_window_bin/.test(url)) {
              console.debug(`📡 FETCH ${method} ${url} → ${res.status} (${(performance.now() - t0).toFixed(0)}ms)`);
            }
            return res;
          } catch (e) {
            console.debug(`📡 FETCH ${method} ${url} ✖`, e);
            throw e;
          }
        };
      })();


    function onGainChange() {
      const el = document.getElementById('gain');
      const val = el ? el.value : '1';
      document.getElementById('gain_display').textContent = `${parseFloat(val)}×`;
      localStorage.setItem('gain', val);
      scheduleRestyle();
    }

    function onColormapChange() {
      const sel = document.getElementById('colormap');
      const chk = document.getElementById('cmReverse');
      if (sel) localStorage.setItem('colormap', sel.value);
      if (chk) localStorage.setItem('cmReverse', chk.checked);
      scheduleRestyle();
    }

    function togglePickMode() {
      isPickMode = !isPickMode;
      const btn = document.getElementById('pickModeBtn');
      btn.textContent = isPickMode ? 'Pick Mode: ON' : 'Pick Mode: OFF';
      btn.classList.toggle('active', isPickMode);
      linePickStart = null;
      deleteRangeStart = null;
      const plotDiv = document.getElementById('plot');
      if (plotDiv) {
        const relayoutResult = safeRelayout(plotDiv, {
          clickmode: clickModeForCurrentState(),
        });
        if (relayoutResult && typeof relayoutResult.finally === 'function') {
          relayoutResult.finally(() => {
            applyDragMode();
          });
        } else {
          applyDragMode();
        }
      } else {
        applyDragMode();
      }
      schedulePickOverlayUpdate();
    }

    function getTraceSamplesForProcessingRange(trace, rowLoIncl, rowHiIncl) {
      const win = latestWindowRender;
      if (!win || win.effectiveLayer !== 'raw') return null;

      const rows = Number(win.shape?.[0] ?? 0);
      const cols = Number(win.shape?.[1] ?? 0);
      if (!rows || !cols) return null;

      const x0 = Number(win.x0);
      const stepX = Number(win.stepX) || 1;
      const col = Math.round((trace - x0) / stepX);
      if (col < 0 || col >= cols) return null;

      const loReq = Math.round(Number(rowLoIncl));
      const hiReq = Math.round(Number(rowHiIncl));
      if (!Number.isFinite(loReq) || !Number.isFinite(hiReq)) return null;
      const lo = Math.max(0, Math.min(rows - 1, loReq));
      const hi = Math.max(0, Math.min(rows - 1, hiReq));
      if (hi < lo) return null;

      const valuesI8 = win.valuesI8 instanceof Int8Array ? win.valuesI8 : null;
      const valuesF32 = !valuesI8 && win.values && win.values.length != null ? win.values : null;
      if (!valuesI8 && !valuesF32) return null;

      const total = rows * cols;
      if (valuesI8 && valuesI8.length < total) return null;
      if (valuesF32 && valuesF32.length < total) return null;

      const scale = Number(win.scale) || 1;
      const len = hi - lo + 1;
      const samples = new Float32Array(len);
      for (let r = lo; r <= hi; r++) {
        const idx = r * cols + col;
        samples[r - lo] = valuesI8 ? (valuesI8[idx] / scale) : valuesF32[idx];
      }

      const sampleStep = Number(win.stepY) || 1;

      return {
        samples,
        sampleStart: (Number(win.y0) || 0) + lo * sampleStep,
        sampleStep,
      };
    }

    // 3-point parabolic interpolation around index i (for peak/trough). Returns a float index.
    function parabolicRefine(arr, i) {
      const n = arr.length;
      const ii = Math.max(1, Math.min(n - 2, i | 0));
      const y1 = arr[ii - 1], y2 = arr[ii], y3 = arr[ii + 1];
      const denom = (y1 - 2 * y2 + y3);
      if (!Number.isFinite(denom) || Math.abs(denom) < 1e-12) return ii; // near-flat → skip
      const delta = 0.5 * (y1 - y3) / denom;          // typically within ~[-0.5, 0.5]
      if (!Number.isFinite(delta) || Math.abs(delta) > 0.6) return ii;   // outlier move → skip
      const xhat = ii + delta;
      if (!Number.isFinite(xhat)) return ii;
      return Math.max(0, Math.min(n - 1, xhat));
    }

    // Upward zero-crossing linear interpolation near i (for rise). Returns a float index.
    function zeroCrossRefine(arr, i) {
      const n = arr.length;
      let i0 = Math.max(0, Math.min(n - 2, i | 0));
      let i1 = i0 + 1;

      // Prefer a nearby upward zero-crossing if present
      if (!(arr[i0] <= 0 && arr[i1] > 0)) {
        if (i0 > 0 && (arr[i0 - 1] <= 0 && arr[i0] > 0)) { i0 = i0 - 1; i1 = i0 + 1; }
        else if (i1 < n - 1 && (arr[i1] <= 0 && arr[i1 + 1] > 0)) { i0 = i1; i1 = i0 + 1; }
        else return i; // no clear upward zero-cross nearby → keep integer index
      }

      const dy = (arr[i1] - arr[i0]);
      if (!Number.isFinite(dy) || Math.abs(dy) < 1e-12) return i; // near-flat → skip
      const frac = (0 - arr[i0]) / dy;                // ∈ [0, 1]
      const xhat = i0 + frac;
      if (!Number.isFinite(xhat)) return i;
      return Math.max(0, Math.min(n - 1, xhat));
    }

    // Adjust a pick's time (seconds) on a given trace to the nearest feature in ±window.
    // Snap is supported only when raw layer data is available in latestWindowRender.
    function adjustPickToFeature(trace, timeSec) {
      const mode = (document.getElementById('snap_mode')?.value) || 'none';
      if (mode === 'none') return timeSec;

      const refineMode = (document.getElementById('snap_refine')?.value) || 'none';

      const dt = (window.defaultDt ?? defaultDt);
      if (!Number.isFinite(dt) || dt <= 0) return timeSec;
      const win = latestWindowRender;
      if (!win || win.effectiveLayer !== 'raw') return timeSec;
      const rows = Number(win.shape?.[0] ?? 0);
      const cols = Number(win.shape?.[1] ?? 0);
      if (!rows || !cols) return timeSec;
      const sampleStartWin = Number(win.y0) || 0;
      const sampleStep = Number(win.stepY) || 1;
      const dtEff = dt * sampleStep;
      if (!Number.isFinite(dtEff) || dtEff <= 0) return timeSec;

      const i0Abs = Math.round(timeSec / dt);
      if (!Number.isFinite(i0Abs)) return timeSec;
      const i0Win = Math.round((i0Abs - sampleStartWin) / sampleStep);
      if (!Number.isFinite(i0Win) || i0Win < 0 || i0Win >= rows) return timeSec;

      // ±window in samples
      const ms = parseFloat(document.getElementById('snap_ms')?.value) || 4;
      const rad = Math.max(1, Math.round((ms / 1000) / dtEff));

      const loWin = Math.max(0, i0Win - rad);
      const hiWin = Math.min(rows - 1, i0Win + rad);
      const sliceLo = Math.max(0, loWin - 1);
      const sliceHi = Math.min(rows - 1, hiWin + 1);
      const traceView = getTraceSamplesForProcessingRange(trace, sliceLo, sliceHi);
      if (!traceView || !traceView.samples) return timeSec;
      const arr = traceView.samples;
      if (arr.length < 3) return timeSec;

      const i0 = i0Win - sliceLo;
      if (!Number.isFinite(i0) || i0 < 0 || i0 >= arr.length) return timeSec;

      // keep one-sample margin for central differences
      const lo = Math.max(1, loWin - sliceLo);
      const hi = Math.min(arr.length - 2, hiWin - sliceLo);
      if (lo > hi) return timeSec;

      let idx = i0;

      if (mode === 'peak') {
        // 近傍の「局所最大」(arr[i-1] <= arr[i] >= arr[i+1]) のうち i0 に最も近い点
        let best = null, bestDist = Infinity;
        for (let i = lo; i <= hi; i++) {
          if (arr[i] >= arr[i - 1] && arr[i] >= arr[i + 1]) {
            const d = Math.abs(i - i0);
            if (d < bestDist) { bestDist = d; best = i; }
          }
        }
        if (best != null) idx = best;
        else {
          // 局所最大がない時のフォールバック：従来どおり最大値へ
          let vmax = -Infinity;
          for (let i = lo; i <= hi; i++) { const v = arr[i]; if (v > vmax) { vmax = v; idx = i; } }
        }
      } else if (mode === 'trough') {
        // 近傍の「局所最小」(arr[i-1] >= arr[i] <= arr[i+1]) のうち最も近い点
        let best = null, bestDist = Infinity;
        for (let i = lo; i <= hi; i++) {
          if (arr[i] <= arr[i - 1] && arr[i] <= arr[i + 1]) {
            const d = Math.abs(i - i0);
            if (d < bestDist) { bestDist = d; best = i; }
          }
        }
        if (best != null) idx = best;
        else {
          // フォールバック：従来どおり最小値へ
          let vmin = Infinity;
          for (let i = lo; i <= hi; i++) { const v = arr[i]; if (v < vmin) { vmin = v; idx = i; } }
        }
      } else if (mode === 'rise') {
        // 「最も近い上りエッジ」を優先：ゼロクロス↑があれば最優先、なければ最大正勾配
        let best = null, bestDist = Infinity;
        for (let i = lo; i < hi; i++) {
          if (arr[i] <= 0 && arr[i + 1] > 0) {
            const cand = (Math.abs(arr[i]) < Math.abs(arr[i + 1])) ? i : i + 1;
            const d = Math.abs(cand - i0);
            if (d < bestDist) { bestDist = d; best = cand; }
          }
        }
        if (best != null) idx = best;
        else {
          let smax = -Infinity;
          for (let i = lo; i <= hi; i++) {
            const s = arr[i + 1] - arr[i - 1];
            if (s > 0 && s > smax) { smax = s; idx = i; }
          }
        }
      }

      let idxFloat = idx;
      if (mode === 'peak' || mode === 'trough') {
        if (refineMode === 'parabolic') idxFloat = parabolicRefine(arr, idx);
      } else if (mode === 'rise') {
        if (refineMode === 'zc') idxFloat = zeroCrossRefine(arr, idx);
      }

      const idxAbs = traceView.sampleStart + idxFloat * traceView.sampleStep;
      return idxAbs * dt;
    }

    function wantWiggleForWindow({ tracesVisible, samplesVisible, widthPx }) {
      const density = tracesVisible / Math.max(1, widthPx);
      if (density >= WIGGLE_DENSITY_THRESHOLD) return false;
      if ((tracesVisible * samplesVisible) > WIGGLE_MAX_POINTS) return false;
      return true;
    }

    function currentVisibleWindow() {
      if (!sectionShape) return null;
      const [totalTraces, totalSamples] = sectionShape;

      // X range
      let x0, x1;
      if (forceFullExtentOnce) {
        x0 = 0; x1 = totalTraces - 1;
      } else if (savedXRange && savedXRange.length === 2) {
        const minX = Math.min(savedXRange[0], savedXRange[1]);
        const maxX = Math.max(savedXRange[0], savedXRange[1]);
        x0 = Math.floor(minX);
        x1 = Math.ceil(maxX);
      } else if (typeof renderedStart === 'number' && typeof renderedEnd === 'number') {
        x0 = renderedStart;
        x1 = renderedEnd;
      } else {
        x0 = 0;
        x1 = totalTraces - 1;
      }

      x0 = Math.max(0, Math.floor(x0));
      x1 = Math.min(totalTraces - 1, Math.ceil(x1));
      if (x1 < x0) [x0, x1] = [x1, x0];

      const spanX = Math.max(1, x1 - x0 + 1);
      const padX = (!forceFullExtentOnce && !!savedXRange)
        ? Math.max(1, Math.floor(spanX * 0.5))
        : 0;
      x0 = Math.max(0, x0 - padX);
      x1 = Math.min(totalTraces - 1, x1 + padX);

      // Y range
      const dtBase = window.defaultDt ?? defaultDt;
      let yMinSec, yMaxSec;
      if (!forceFullExtentOnce && savedYRange && savedYRange.length === 2) {
        yMinSec = Math.min(savedYRange[0], savedYRange[1]);
        yMaxSec = Math.max(savedYRange[0], savedYRange[1]);
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
      const padY = (!forceFullExtentOnce && !!savedYRange)
        ? Math.max(1, Math.floor(spanY * 0.1))
        : 0;
      y0 = Math.max(0, y0 - padY);
      y1 = Math.min(totalSamples - 1, y1 + padY);

      // one-shot full-extent is consumed here
      if (forceFullExtentOnce) forceFullExtentOnce = false;

      return {
        x0, x1, y0, y1,
        nTraces: x1 - x0 + 1,
        nSamples: y1 - y0 + 1,
      };
    }

    function updateKey1Display() {
      const slider = document.getElementById('key1_slider');
      const display = document.getElementById('key1_val_display');
      const idx = parseInt(slider.value);
      display.value = key1Values[idx] ?? '';
    }

    function syncSliderWithInput() {
      const slider = document.getElementById('key1_slider');
      const display = document.getElementById('key1_val_display');
      const val = parseInt(display.value);
      const idx = key1Values.indexOf(val);
      slider.value = idx >= 0 ? idx : 0;
      display.value = key1Values[slider.value] ?? '';
    }

    function stepKey1(delta) {
      const slider = document.getElementById('key1_slider');
      let value = parseInt(slider.value) + delta;
      value = Math.max(slider.min, Math.min(slider.max, value));
      slider.value = value;
      updateKey1Display();
    }

    function setKey1SliderMax(max) {
      document.getElementById('key1_slider').max = max;
    }

    async function fetchKey1Values() {
      const res = await fetch(`/get_key1_values?file_id=${currentFileId}&key1_byte=${currentKey1Byte}&key2_byte=${currentKey2Byte}`);
      if (res.ok) {
        const data = await res.json();
        key1Values = data.values;
        setKey1SliderMax(key1Values.length - 1);
        document.getElementById('key1_val_display').min = key1Values[0];
        document.getElementById('key1_val_display').max = key1Values[key1Values.length - 1];
        document.getElementById('key1_slider').value = 0;
        updateKey1Display();
      }
    }

    async function loadSettings() {
      const params = new URLSearchParams(window.location.search);
      currentFileId = params.get('file_id') || localStorage.getItem('file_id') || '';
      currentKey1Byte = parseInt(params.get('key1_byte') || localStorage.getItem('key1_byte') || '189');
      currentKey2Byte = parseInt(params.get('key2_byte') || localStorage.getItem('key2_byte') || '193');
      const scalingParam = (params.get('scaling') || localStorage.getItem('scaling_mode') || 'amax').toLowerCase();
      currentScaling = scalingParam === 'tracewise' ? 'tracewise' : 'amax';
      try { localStorage.setItem('scaling_mode', currentScaling); } catch (_) {}
      const scalingSel = document.getElementById('scalingMode');
      if (scalingSel) scalingSel.value = currentScaling;
      document.getElementById('file_id').value = currentFileId;
      if (!currentFileId) {
        currentFileName = '';
        return;
      }
      localStorage.setItem('file_id', currentFileId);
      localStorage.setItem('key1_byte', currentKey1Byte);
      localStorage.setItem('key2_byte', currentKey2Byte);
      await fetchCurrentFileName();
      await fetchKey1Values();
      await fetchSectionMeta();
      await fetchAndPlot();
    }

    async function fetchPicks() {
      if (currentFileName) {
        await reloadPicksForCurrentSection();
        return;
      }
      if (!currentFileId) return;
      const idx = parseInt(document.getElementById('key1_slider').value);
      const key1Val = key1Values?.[idx];
      if (key1Val === undefined) return;
      try {
        const res = await fetch(`/picks?file_id=${currentFileId}&key1=${encodeURIComponent(key1Val)}&key1_byte=${currentKey1Byte}&key2_byte=${currentKey2Byte}`);
        if (res.ok) {
          const data = await res.json();
          picks = (data.picks || []).map(p => ({ trace: p.trace, time: p.time }));
        }
        const [x0, x1] = visibleXRng();
        D('PICKS@fetchPicks', { count: picks.length, vis: [x0, x1],
          visCount: picks.filter(p => Math.round(p.trace) >= x0 && Math.round(p.trace) <= x1).length,
          sample: samplePicks(picks) });
      } catch (e) { console.error('Failed to fetch picks', e); }
    }

    function postPick(trace, time) {
      queueUpsert(trace, time);
      D('PICKS@queuePost', { trace: (trace | 0), time: +time });
    }

    function deletePick(trace) {
      queueDelete(trace);
      D('PICKS@queueDelete', { trace: (trace | 0) });
    }

    // --- filename & picks loader (by filename) ---
    async function fetchSectionMeta() {
      if (!currentFileId) return null;
      try {
        const q = new URLSearchParams({
          file_id: currentFileId,
          key1_byte: String(currentKey1Byte),
          key2_byte: String(currentKey2Byte),
        });
        const res = await fetch(`/get_section_meta?${q.toString()}`);
        if (!res.ok) {
          console.warn('get_section_meta failed', res.status);
          return null;
        }
        const meta = await res.json();
        if (Array.isArray(meta.shape) && meta.shape.length === 2) {
          sectionShape = [Number(meta.shape[0]), Number(meta.shape[1])];
          applyServerDt(meta);
          console.info('[META] sectionShape=', sectionShape, 'dt=', meta.dt);
          return sectionShape;
        }
      } catch (e) {
        console.warn('get_section_meta error', e);
      }
      return null;
    }

    async function fetchCurrentFileName() {
      if (!currentFileId) {
        currentFileName = '';
        return;
      }
      try {
        const r = await fetch(`/file_info?file_id=${encodeURIComponent(currentFileId)}`);
        if (!r.ok) {
          currentFileName = '';
          return;
        }
        const j = await r.json();
        currentFileName = j.file_name || '';
      } catch (e) {
        currentFileName = '';
        console.warn('file_info failed', e);
      }
    }

    async function reloadPicksForCurrentSection(key1IdxOrVal) {
      if (!currentFileId) return [];

      const fileId = currentFileId;
      const idxRaw = Number.isInteger(key1IdxOrVal)
        ? key1IdxOrVal
        : parseInt(document.getElementById('key1_slider').value, 10);
      const idx = Number.isFinite(idxRaw) ? idxRaw : 0;
      const key1Val = key1Values?.[idx];
      if (key1Val === undefined) return [];
      try {
        const r = await fetch(`/picks?file_id=${encodeURIComponent(fileId)}&key1=${encodeURIComponent(key1Val)}&key1_byte=${currentKey1Byte}&key2_byte=${currentKey2Byte}`);
        if (!r.ok) return [];
        const j = await r.json();
        const arr = (j.picks || []).map(p => ({ trace: (p.trace | 0), time: +p.time }));
        picks = arr;
        if (typeof renderLatestView === 'function') {
          try { renderLatestView(); } catch (e) { console.warn('render after picks failed', e); }
        }
        const [x0, x1] = visibleXRng();
        D('PICKS@reload', {
          count: picks.length,
          vis: [x0, x1],
          visCount: picks.filter(p => Math.round(p.trace) >= x0 && Math.round(p.trace) <= x1).length,
          sample: samplePicks(picks)
        });
        return arr;
      } catch (e) {
        console.warn('reload picks failed', e);
        return [];
      }
    }

    async function fetchAndPlot() {
      snapshotAxesRangesFromDOM();
      console.log('--- fetchAndPlot start ---');
      console.time('Total fetchAndPlot');

      const index = parseInt(document.getElementById('key1_slider').value);
      const key1Val = key1Values[index];

      // ★ FB予測キャッシュ取得：レイヤ＆パイプラインキーでキー統一
      const layerCur = (document.getElementById('layerSelect')?.value) || 'raw';
      const pKeyCur = window.latestPipelineKey || null;
      const methodCur = document.getElementById('pick_method').value;
      const sigmaCur = Number(document.getElementById('sigma_ms_max').value) || 20;
      const cachedEntry = getCachedFbEntry(
        fbCacheKey(currentFileId, key1Val, layerCur, pKeyCur),
        methodCur,
        sigmaCur,
      );
      predictedPicks = cachedEntry && cachedEntry.picks ? cachedEntry.picks.slice() : [];

      await fetchPicks();

      latestWindowRender = null;
      windowFetchToken += 1;
      uiResetNonce++;
      if (window.pipelineEvents && typeof window.pipelineEvents.emit === 'function') {
        window.pipelineEvents.emit('section:prepare', {});
      } else if (window.pipelineUI && typeof window.pipelineUI.prepareForNewSection === 'function') {
        window.pipelineUI.prepareForNewSection();
      } else {
        latestTapData = {};
        latestPipelineKey = null;
        const sel = document.getElementById('layerSelect');
        if (sel) {
          sel.innerHTML = '';
          sel.appendChild(new Option('raw', 'raw'));
          sel.value = 'raw';
        }
      }

      latestSeismicData = null;
      renderLatestView();
      fetchWindowAndPlot();

      console.timeEnd('Total fetchAndPlot');
      console.log('--- fetchAndPlot end ---');
    }

    function drawSelectedLayer(start = null, end = null) {
      D('DRAW@selectLayer', { layer: document.getElementById('layerSelect')?.value, start, end });
      latestSeismicData = null;
      renderLatestView();
      scheduleWindowFetch();
    }


    function renderLatestView(startOverride = null, endOverride = null) {
      const sel = document.getElementById('layerSelect');
      const layer = sel ? sel.value : 'raw';
      const slider = document.getElementById('key1_slider');
      const idx = slider ? parseInt(slider.value, 10) : 0;
      const key1Val = key1Values[idx];

      if (latestSeismicData) {
        latestSeismicData = null;
      }

      if (
        latestWindowRender &&
        latestWindowRender.requestedLayer === layer &&
        latestWindowRender.key1 === key1Val
      ) {
        if (layer !== 'raw') {
          const pipelineKeyNow = window.latestPipelineKey || null;
          if ((latestWindowRender.pipelineKey || null) !== (pipelineKeyNow || null)) {
            return;
          }
        }
        if (latestWindowRender.mode === 'wiggle') {
          renderWindowWiggle(latestWindowRender);
        } else {
          renderWindowHeatmap(latestWindowRender);
        }
      }
    }
    function clickModeForCurrentState() {
      return isPickMode ? 'event' : 'event+select';
    }

    function pickOnTrace(trace) {
      return picks.findIndex(p => Math.round(p.trace) === trace);
    }

    function toTraceInt(t) {
      // マイナス防止も兼ねる。必要なら上限クリップもここで。
      return Math.max(0, Math.round(t));
    }

    async function handlePickNormalized({ trace, time, shiftKey, ctrlKey, altKey }) {
      if (isPickMode && dragOverride === 'pan') return;
      if (!Number.isFinite(trace) || !Number.isFinite(time)) return;
      const trInt = toTraceInt(trace);

      if (handlePickNormalized._busy) {
        handlePickNormalized._queued = { trace: trInt, time, shiftKey, ctrlKey, altKey };
        return;
      }
      handlePickNormalized._busy = true;

      try {
        if (!isPickMode) return;

        console.log('🔥 pick request', { trace, time, shiftKey, ctrlKey, altKey });

        if (ctrlKey) {
          if (deleteRangeStart === null) {
            deleteRangeStart = trInt;
            linePickStart = null;
            return;
          }
          const x0 = deleteRangeStart;
          deleteRangeStart = null;
          const x1 = trInt;
          const start = Math.min(x0, x1);
          const end = Math.max(x0, x1);
          const toDelete = picks.filter(p => Math.round(p.trace) >= start && Math.round(p.trace) <= end);
          const promises = toDelete.map(p => deletePick(Math.round(p.trace)));
          picks = picks.filter(p => Math.round(p.trace) < start || Math.round(p.trace) > end);
          await Promise.all(promises);
          D('PICKS@handlePickNormalized:line', { range: [start, end], count: picks.length });
          schedulePickOverlayUpdate();
          return;
        }

        if (shiftKey) {
          if (!linePickStart) {
            linePickStart = { trace: trInt, time };
            deleteRangeStart = null;
            return;
          }

          const { trace: x0, time: y0 } = linePickStart;
          linePickStart = { trace: trInt, time };
          const x1 = trInt;
          const y1 = time;
          const xStart = Math.round(Math.min(x0, x1));
          const xEnd = Math.round(Math.max(x0, x1));
          const slope = x1 === x0 ? 0 : (y1 - y0) / (x1 - x0);

          const promises = [];
          for (let x = xStart; x <= xEnd; x++) {
            const y = x1 === x0 ? y1 : y0 + slope * (x - x0);
            const snapped = snapTimeFromDataY(y);
            const tAdj = adjustPickToFeature(x, snapped);

            const idx = pickOnTrace(x);
            if (idx >= 0) {
              promises.push(deletePick(x));
              picks.splice(idx, 1);
            }
            picks.push({ trace: x, time: tAdj });
            promises.push(postPick(x, tAdj));
          }
          await Promise.all(promises);
          D('PICKS@handlePickNormalized:line', { range: [xStart, xEnd], count: picks.length });
          schedulePickOverlayUpdate();
          return;
        }

        linePickStart = null;
        deleteRangeStart = null;

        const idx = pickOnTrace(trInt);
        const promises = [];
        if (idx >= 0) {
          promises.push(deletePick(trInt));
          picks.splice(idx, 1);
        }
        const tAdj = adjustPickToFeature(trInt, time);
        picks.push({ trace: trInt, time: tAdj });
        promises.push(postPick(trInt, tAdj));
        await Promise.all(promises);
        D('PICKS@handlePickNormalized:single', { add: { trace: trInt, time: tAdj }, count: picks.length });
        schedulePickOverlayUpdate();
      } finally {
        handlePickNormalized._busy = false;
        const next = handlePickNormalized._queued;
        handlePickNormalized._queued = null;
        if (next) setTimeout(() => handlePickNormalized(next), 0);
      }
    }

    async function handleRelayout(ev) {
      if (suppressRelayout) return;
      if (!ev || typeof ev !== 'object') return;

      flushPendingResetFetchIfNeeded();

      D('RELAYOUT@begin', { keys: Object.keys(ev), isRelayouting, pickMode: isPickMode });

      const gd = document.getElementById('plot');
      if (!gd) return;

      // range 更新
      if ('xaxis.range[0]' in ev && 'xaxis.range[1]' in ev) {
        savedXRange = [ev['xaxis.range[0]'], ev['xaxis.range[1]']];
      }
      if ('yaxis.range[0]' in ev && 'yaxis.range[1]' in ev) {
        const y0 = ev['yaxis.range[0]'];
        const y1 = ev['yaxis.range[1]'];
        savedYRange = y0 > y1 ? [y0, y1] : [y1, y0];
      }

      // reset/autorange は debounce をバイパスして即 fetch（shape同期はしない）
      if (isResetRelayout(ev)) {
        await new Promise(r => requestAnimationFrame(r));
        snapshotAxesRangesFromDOM();
        if (isRelayouting) {
          pendingResetFetch = true;
          return;
        }
        pendingResetFetch = false;
        checkModeFlipAndRefetch({ immediate: true });
        return;
      }

      // ★ 本当に shape のプロパティが変わったときだけ同期（eraseshape 等の全置換は無視）
      const shapePropKeys = Object.keys(ev).filter(k => /^shapes\[\d+\]\.(x0|x1|y0|y1|line\.color|line\.width|visible)$/.test(k));
      if (shapePropKeys.length === 0 || !isPickMode) {
        D('RELAYOUT@skip(no-shape-prop or not pickMode)', viewerState());
        return;
      }

      // Plotly が layout 反映完了するのを待つ
      await new Promise(r => requestAnimationFrame(r));

      const fullShapes =
        (gd && gd._fullLayout && Array.isArray(gd._fullLayout.shapes))
          ? gd._fullLayout.shapes
          : [];

      // 可視Xレンジ
      const xa = gd?._fullLayout?.xaxis;
      const xr = (xa && Array.isArray(xa.range)) ? xa.range : [renderedStart ?? 0, renderedEnd ?? 0];
      const xVis0 = Math.floor(Math.min(xr[0], xr[1]));
      const xVis1 = Math.ceil(Math.max(xr[0], xr[1]));
      const inView = (t) => t >= xVis0 && t <= xVis1;

      // shapes → newMap（赤のみ・可視範囲のみ）
      const newMap = new Map();
      for (const s of fullShapes) {
        if (!s || !s.line || s.line.color !== 'red') continue;
        const tr = Math.round(((+s.x0) + (+s.x1)) / 2);
        if (!inView(tr)) continue;
        const time = ((+s.y0) + (+s.y1)) / 2;
        newMap.set(tr, { trace: tr, time });
      }

      // 旧ローカル → oldMap（可視範囲のみ）
      const oldMap = new Map();
      for (const p of (picks || [])) {
        const tr = Math.round(p.trace);
        if (!inView(tr)) continue;
        oldMap.set(tr, { trace: tr, time: +p.time });
      }

      // 差分計算
      const del = [];
      const add = [];
      const upd = [];

      for (const [tr, op] of oldMap) {
        if (!newMap.has(tr)) {
          del.push(tr);
        } else {
          const np = newMap.get(tr);
          if (Math.abs(np.time - op.time) > 1e-9) upd.push(np);
        }
      }
      for (const [tr, np] of newMap) {
        if (!oldMap.has(tr)) add.push(np);
      }

      // サーバ反映（可視範囲の差分のみ）
      await Promise.all([
        ...del.map(t => deletePick(t)),
        ...add.map(p => postPick(p.trace, p.time)),
        ...upd.map(p => postPick(p.trace, p.time)),
      ]);

      // ローカル更新：不可視はそのまま、可視は差分適用
      const kept = (picks || []).filter(p => {
        const tr = Math.round(p.trace);
        return tr < xVis0 || tr > xVis1 || !del.includes(tr);
      });
      const mergedMap = new Map(kept.map(p => [Math.round(p.trace), { trace: Math.round(p.trace), time: +p.time }]));
      for (const p of add) mergedMap.set(p.trace, p);
      for (const p of upd) mergedMap.set(p.trace, p);
      picks = Array.from(mergedMap.values()).sort((a, b) => a.trace - b.trace);

      D('RELAYOUT@sync', {
        shapePropKeys, vis: [xVis0, xVis1],
        del, add: add.map(p => p.trace), upd: upd.map(p => p.trace),
        count: picks.length, sample: samplePicks(picks)
      });
    }


    window.addEventListener('DOMContentLoaded', () => {
      const createProgressOverlay = window.createPipelineProgressOverlay;
      if (typeof createProgressOverlay !== 'function') return;
      const progressApi = createProgressOverlay({
        overlayId: 'ppOverlay',
        statusId: 'ppStatus',
        barId: 'ppBarInner',
        cancelId: 'ppCancelBtn',
        onCancel: () => {
          if (window.pipelineUI && typeof window.pipelineUI.cancel === 'function') {
            window.pipelineUI.cancel();
          }
        },
      });
      if (!progressApi) return;
      window.pipelineProgress = progressApi;
    });

    window.addEventListener('DOMContentLoaded', () => {
      console.info('[viewer] DOMContentLoaded hook');
      const fileIdEl = document.getElementById('file_id');
      const slider = document.getElementById('key1_slider');

      const boot = async () => {
        if (fileIdEl && fileIdEl.value && !currentFileId) {
          currentFileId = fileIdEl.value;
        }
        if (!currentFileId) {
          currentFileName = '';
          return;
        }
        if (!currentFileName) {
          await fetchCurrentFileName();
        }
      };

      loadSettings().catch((err) => console.warn('loadSettings failed', err)).finally(() => {
        boot().catch((err) => console.warn('initial picks load failed', err));
      });

      if (fileIdEl) {
        fileIdEl.addEventListener('change', async () => {
          currentFileId = fileIdEl.value || '';
          key1Values = [];
          sectionShape = null;
          savedXRange = null;
          savedYRange = null;
          renderedStart = null;
          renderedEnd = null;
          latestWindowRender = null;
          windowFetchToken += 1;
          latestSeismicData = null;
          latestTapData = {};
          fbPredReqId += 1;

          picks = [];
          predictedPicks = [];
          currentFbKey = null;
          currentFbLayer = 'raw';
          currentFbPipelineKey = null;
          fbPredCache.clear();
          latestPipelineKey = null;
          window.latestPipelineKey = null;
          if (windowFetchCtrl) {
            try { windowFetchCtrl.abort(); } catch (_) { }
            windowFetchCtrl = null;
          }


          await fetchCurrentFileName();
          await fetchKey1Values();
          await fetchSectionMeta();
          (typeof fetchAndPlotDebounced?.flush === 'function')
            ? fetchAndPlotDebounced.flush()
            : fetchAndPlot();
        });
      }
      if (slider) {
        slider.addEventListener('change', () => {
          // ピック読込は fetchAndPlot() 内に集約
          (typeof fetchAndPlotDebounced?.flush === 'function')
            ? fetchAndPlotDebounced.flush()
            : fetchAndPlot();
        });
      }
    });

    // Toggle between raw and first tap with the "n" key
    window.addEventListener('keydown', (e) => {
      if (
        e.key.toLowerCase() === 'n' &&
        !e.ctrlKey && !e.altKey && !e.metaKey &&
        !['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)
      ) {
        const sel = document.getElementById('layerSelect');
        if (!sel) return;
        if (sel.options.length > 1) {
          sel.value = sel.value === 'raw' ? sel.options[1].value : 'raw';
          drawSelectedLayer();
        }
      }
    });

    window.addEventListener('keyup', (e) => {
      if (e.key === 'Shift') {
        linePickStart = null;
      }
    });
