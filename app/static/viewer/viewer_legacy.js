    function installCustomDoubleClick(plotDiv) {
      if (!plotDiv || plotDiv.__customDbl) return;

      const onDbl = (e) => {
        // 既定の Plotly ダブルクリックは無効化しているので、ここで挙動を全部やる
        e.preventDefault();
        e.stopPropagation();

        // Shift+ダブルクリック → フル範囲へ
        if (e.shiftKey) {
          if (!sectionShape) return;
          const totalTraces = sectionShape[0] | 0;
          const totalSamples = sectionShape[1] | 0;
          const dt = window.defaultDt ?? defaultDt;
          savedXRange = [0, Math.max(0, totalTraces - 1)];
          savedYRange = [Math.max(0, (totalSamples - 1) * dt), 0]; // Yは上向きに大きい→降順で保存
          withSuppressedRelayout(Plotly.relayout(plotDiv, {
            'xaxis.autorange': false,
            'yaxis.autorange': false,
            'xaxis.range': savedXRange,
            'yaxis.range': savedYRange,
          }));
          checkModeFlipAndRefetch();
          return;
        }

        // 通常ダブルクリック → カーソル中心で 1/2ズーム（範囲2倍）
        const gd = plotDiv;
        const xa = gd?._fullLayout?.xaxis;
        const ya = gd?._fullLayout?.yaxis;
        if (!xa || !ya) return;

        // 現在のレンジ（Plotlyはyが上→下で降順のことが多いのでmin/max整える）
        const xr = Array.isArray(xa.range) ? xa.range.slice() : [renderedStart ?? 0, renderedEnd ?? 0];
        let xMin = Math.min(xr[0], xr[1]);
        let xMax = Math.max(xr[0], xr[1]);

        const yr = Array.isArray(ya.range) ? ya.range.slice() : [0, (sectionShape?.[1] ?? 1) * (window.defaultDt ?? defaultDt)];
        let yMin = Math.min(yr[0], yr[1]);
        let yMax = Math.max(yr[0], yr[1]);

        const totalTraces = sectionShape?.[0] ?? (latestSeismicData?.length ?? 0);
        const totalSamples = sectionShape?.[1] ?? (latestSeismicData?.[0]?.length ?? 0);
        if (!totalTraces || !totalSamples) return;
        const dt = window.defaultDt ?? defaultDt;
        const yAbsMin = 0;
        const yAbsMax = (totalSamples - 1) * dt;

        // 画面座標→データ座標
        const cx = Number.isFinite(traceAtPixel?.(e.clientX)) ? traceAtPixel(e.clientX) : (xMin + xMax) / 2;
        const cy = Number.isFinite(timeAtPixel?.(e.clientY)) ? timeAtPixel(e.clientY) : (yMin + yMax) / 2;

        // いまの幅/高さを2倍に
        const spanX = (xMax - xMin) * 2;
        const spanY = (yMax - yMin) * 2;

        // 新しい範囲（中心はカーソル）
        let nx0 = cx - spanX / 2;
        let nx1 = cx + spanX / 2;

        let ny0 = cy - spanY / 2;
        let ny1 = cy + spanY / 2;

        // 枠内に収める（はみ出した分は反対側に寄せて可能な限り中心を保つ）
        if (nx0 < 0) { nx1 += -nx0; nx0 = 0; }
        if (nx1 > totalTraces - 1) { const d = nx1 - (totalTraces - 1); nx0 -= d; nx1 = totalTraces - 1; }
        nx0 = Math.max(0, Math.floor(nx0));
        nx1 = Math.min(totalTraces - 1, Math.ceil(nx1));

        if (ny0 < yAbsMin) { ny1 += (yAbsMin - ny0); ny0 = yAbsMin; }
        if (ny1 > yAbsMax) { const d = ny1 - yAbsMax; ny0 -= d; ny1 = yAbsMax; }
        ny0 = Math.max(yAbsMin, ny0);
        ny1 = Math.min(yAbsMax, ny1);

        // 保存（Yは降順で保存している実装に合わせる）
        savedXRange = [nx0, nx1];
        savedYRange = [ny1, ny0];

        withSuppressedRelayout(Plotly.relayout(plotDiv, {
          'xaxis.autorange': false,
          'yaxis.autorange': false,
          'xaxis.range': savedXRange,
          'yaxis.range': savedYRange,
        }));

        // 表示密度の閾値を超えた/下回ったら自動モード切替も走らせる
        checkModeFlipAndRefetch();
      };

      plotDiv.addEventListener('dblclick', onDbl, { capture: true });
      plotDiv.__customDbl = onDbl;
    }


    function ensureFlushPickOpsDebounced() {
      if (!flushPickOpsDebounced) {
        const debFn = (typeof debounce === 'function') ? debounce : makeDebounced;
        flushPickOpsDebounced = debFn(flushPickOps, 120);
      }
      return flushPickOpsDebounced;
    }

    function _sliderKey1Idx() {
        const el = document.getElementById('key1_slider');
        const v = Number.parseInt(el?.value ?? '0', 10);
        return Number.isFinite(v) ? v : 0;
      }
        function _opKey(fileId, key1Val, key1Byte, trace) {
          return `${fileId}|${key1Val}|${key1Byte}|${trace | 0}`;
        }

    function queueUpsert(trace, time) {
      const tr = (trace | 0);
      const fileId = currentFileId;
      const key1Idx = _sliderKey1Idx();
      const key1Byte = currentKey1Byte;
      const key1Val = key1Values?.[key1Idx];
      if (key1Val === undefined) {
        console.warn('queueUpsert skipped: key1 value unavailable', { key1Idx });
        return;
      }
      const k = _opKey(fileId, key1Val, key1Byte, tr);
      __pickOps.set(k, { op: 'upsert', fileId, key1Val, key1Byte, trace: tr, time: +time });
      const debounced = ensureFlushPickOpsDebounced();
      debounced();
    }

    function queueDelete(trace) {
      const tr = (trace | 0);
      const fileId = currentFileId;
      const key1Idx = _sliderKey1Idx();
      const key1Byte = currentKey1Byte;
      const key1Val = key1Values?.[key1Idx];
      if (key1Val === undefined) {
        console.warn('queueDelete skipped: key1 value unavailable', { key1Idx });
        return;
      }
      const k = _opKey(fileId, key1Val, key1Byte, tr);
      __pickOps.set(k, { op: 'delete', fileId, key1Val, key1Byte, trace: tr });
      const debounced = ensureFlushPickOpsDebounced();
      debounced();
    }


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
      const mode = latestWindowRender?.mode || (latestSeismicData ? 'full' : 'none');
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
    window.debugDump = () => ({
      viewer: viewerState('DUMP'),
      picks: samplePicks(picks, 20),
      predicted: samplePicks(predictedPicks, 10)
  });

    var currentScaling = 'amax';

    function cacheKey(val, mode) {
      const scaleKey = mode === 'raw' ? currentScaling : mode;
      return `${val}|${mode}|${scaleKey}`;
    }
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

    function withSuppressedRelayout(promiseLike) {
        suppressRelayout = true;
        if (promiseLike && typeof promiseLike.finally === 'function') {
            return promiseLike.finally(() => { suppressRelayout = false; });
          }
        suppressRelayout = false;
        return promiseLike;
      }

    function logCurrentPicks(opts = {}) {
      const { includePredicted = false, onlyVisible = false } = opts;

      const dt = (window.defaultDt ?? defaultDt);
      const k1Idx = parseInt(document.getElementById('key1_slider').value, 10) | 0;
      const key1Val = key1Values?.[k1Idx];
      const layer = document.getElementById('layerSelect')?.value || 'raw';

      const start = (typeof renderedStart === 'number') ? renderedStart : 0;
      const end = (typeof renderedEnd === 'number') ? renderedEnd : (sectionShape ? sectionShape[0] - 1 : 0);

      const src = includePredicted ? [...picks, ...predictedPicks] : picks;

      const rows = src
        .filter(p => Number.isFinite(p.trace) && Number.isFinite(p.time))
        .map(p => ({
          trace: Math.round(p.trace),
          time_s: +p.time,
          sample_idx: Math.round(p.time / dt),
        }))
        .filter(r => !onlyVisible || (r.trace >= start && r.trace <= end))
        .sort((a, b) => a.trace - b.trace);

      console.log(`[Picks] key1=${key1Val} layer=${layer} count=${rows.length} visibleOnly=${!!onlyVisible}`);
      console.table(rows);
      return rows; // 必要なら呼び出し元で使えるように返す
    }

    // グローバルから呼べるように
    window.logCurrentPicks = logCurrentPicks;

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
    let forceFullExtentOnce = false;    // next window calc uses full extent with no padding

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

    function checkModeFlipAndRefetch() {
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
              if (needFresh) scheduleWindowFetch();
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

        if (needFresh) scheduleWindowFetch();
    }

    // （任意：すでに入れているならそのままでOK）
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

    function installPlotlyViewportHandlersOnce() {
      const plotDiv = document.getElementById('plot');
      if (!plotDiv || plotDiv.__viewportHandlersInstalled) return;
      plotDiv.__viewportHandlersInstalled = true;

      plotDiv.on('plotly_relayouting', () => {
        if (suppressRelayout) return;
        isRelayouting = true;
      });

      plotDiv.on('plotly_relayout', (ev) => {
        if (suppressRelayout) return;
        isRelayouting = false;
        const result = handleRelayout(ev);
        if (result && typeof result.catch === 'function') {
          result.catch((err) => console.warn('handleRelayout failed', err));
        }
        if (redrawPending) {
            redrawPending = false;
            try { renderLatestView(); } catch (e) { console.warn('deferred render failed', e); }
          }
        onViewportSettled();
      });

    }

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
      renderLatestView();
      applyDragMode();
    }

    function getSeismicForProcessing() {
      const sel = document.getElementById('layerSelect');
      const layer = sel?.value || 'raw';
      // ★ 加工レイヤは window API 経由のみ。配列は raw のみ保持。
      return (layer === 'raw' && Array.isArray(rawSeismicData)) ? rawSeismicData : null;
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
    // Uses the *original* data of the currently selected layer (raw or pipeline layer).
    function adjustPickToFeature(trace, timeSec) {
      const mode = (document.getElementById('snap_mode')?.value) || 'none';
      if (mode === 'none') return timeSec;

      const refineMode = (document.getElementById('snap_refine')?.value) || 'none';

      const seismic = getSeismicForProcessing(); // raw or current TAP layer
      if (!seismic || !seismic[trace]) return timeSec;

      const dt = (window.defaultDt ?? defaultDt);
      const arr = seismic[trace]; // Float32Array (one trace)
      if (!arr || !arr.length) return timeSec;

      const i0 = Math.round(timeSec / dt);

      // ±window in samples
      const ms = parseFloat(document.getElementById('snap_ms')?.value) || 4;
      const rad = Math.max(1, Math.round((ms / 1000) / dt));

      // keep one-sample margin for central differences
      const lo = Math.max(1, i0 - rad);
      const hi = Math.min(arr.length - 2, i0 + rad);

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

      return idxFloat * dt;
    }

    function putCacheF32(key, f32) {
      if (cache.size >= CACHE_LIMIT) {
        const oldestKey = cache.keys().next().value;
        const old = cache.get(oldestKey);
        if (old) old.f32 = null;
        cache.delete(oldestKey);
      }
      cache.set(key, { f32 });

      const canCheckMemory = performance.memory && performance.memory.usedJSHeapSize > 0;
      const used = canCheckMemory ? performance.memory.usedJSHeapSize : 0;

      const hardLimit = (typeof cfg === 'object' && Number.isFinite(cfg.HARD_LIMIT_BYTES))
        ? cfg.HARD_LIMIT_BYTES
        : 512 * 1024 * 1024;

      console.log('--- putCache called ---');
      console.log('Keys:', Array.from(cache.keys()));
      console.log('Cache size:', cache.size);
      if (canCheckMemory) {
        console.log(`used: ${(used / 1024 / 1024).toFixed(1)} MB / ${(hardLimit / 1024 / 1024).toFixed(0)} MB`);
      }

      if (canCheckMemory && used > hardLimit) {
        console.warn(`⚠ Memory limit exceeded! (${(used / 1024 / 1024).toFixed(1)} MB > ${(hardLimit / 1024 / 1024).toFixed(0)} MB)`);
        let removed = 0;
        while (cache.size > 1) {
          const removedKey = cache.keys().next().value;
          const entry = cache.get(removedKey);
          if (entry) entry.f32 = null;
          cache.delete(removedKey);
          removed++;
        }
        console.warn(`Evicted ${removed} items due to hard heap limit.`);
        console.log('Remaining keys:', Array.from(cache.keys()));
      }
      console.log('------------------------');
    }

    function getCacheF32(key) {
      if (!cache.has(key)) return null;
      const entry = cache.get(key);
      cache.delete(key); cache.set(key, entry); // refresh LRU
      return entry;
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

      // 🪄 Window-first ならフルraw構築をスキップして即ウィンドウ描画へ
      if (shouldPreferWindowFirst()) {
          latestWindowRender = null;
          windowFetchToken += 1;
          uiResetNonce++;
          if (window.pipelineUI && typeof window.pipelineUI.prepareForNewSection === 'function') {
              window.pipelineUI.prepareForNewSection();
            } else {
              latestTapData = { };
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
          return;
        }

      console.time('Cache lookup');
      const hit = getCacheF32(cacheKey(key1Val, 'raw'));
      console.timeEnd('Cache lookup');

      let traces;
      let f32 = hit ? hit.f32 : null;

      if (f32) {
        console.time('Decode from cache');
        const [nTraces, nSamples] = sectionShape;
        traces = new Array(nTraces);
        for (let i = 0; i < nTraces; i++) {
          traces[i] = f32.subarray(i * nSamples, (i + 1) * nSamples);
        }
        console.timeEnd('Decode from cache');
        rawSeismicData = traces;
      } else {
        console.time('Fetch binary');
        const meta = sectionShape || await fetchSectionMeta();
        if (!meta) {
          console.timeEnd('Fetch binary');
          alert('Failed to load section meta');
          return;
        }
        const [nTracesMeta, nSamplesMeta] = meta;
        const qFull = new URLSearchParams({
          file_id: currentFileId,
          key1: String(key1Val),
          key1_byte: String(currentKey1Byte),
          key2_byte: String(currentKey2Byte),
          x0: '0', x1: String(nTracesMeta - 1),
          y0: '0', y1: String(nSamplesMeta - 1),
          step_x: '1', step_y: '1',
          transpose: '0',
        });
        qFull.set('scaling', currentScaling);
        const res = await fetch(`/get_section_window_bin?${qFull.toString()}`);
        if (!res.ok) {
          console.timeEnd('Fetch binary');
          alert('Failed to load section');
          return;
        }
        const bin = new Uint8Array(await res.arrayBuffer());
        console.timeEnd('Fetch binary');

        console.time('Decode & cache');
        const obj = msgpack.decode(bin);
        applyServerDt(obj);
        const int8 = new Int8Array(obj.data.buffer);
        f32 = Float32Array.from(int8, (v) => v / obj.scale);
        putCacheF32(cacheKey(key1Val, 'raw'), f32);
        // Keep sectionShape from /get_section_meta only. Sanity-check payload shape if needed:
        if (Array.isArray(sectionShape) && Array.isArray(obj.shape)) {
          const [mt, ms] = sectionShape;
          const [pt0, pt1] = obj.shape;
          if ((pt0 * pt1) !== (mt * ms)) {
            console.warn('shape mismatch vs meta', { meta: sectionShape, payload: obj.shape });
          }
        }
        const [nTraces, nSamples] = meta;
        traces = new Array(nTraces);
        for (let i = 0; i < nTraces; i++) {
          traces[i] = f32.subarray(i * nSamples, (i + 1) * nSamples);
        }
        console.timeEnd('Decode & cache');
        rawSeismicData = traces;
      }

      latestWindowRender = null;
      windowFetchToken += 1;
      uiResetNonce++;
      if (window.pipelineUI && typeof window.pipelineUI.prepareForNewSection === 'function') {
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

      const totalTraces = rawSeismicData ? rawSeismicData.length : (sectionShape ? sectionShape[0] : 0);
      const [s, e] = totalTraces > 0
        ? (savedXRange ? visibleTraceIndices(savedXRange, totalTraces) : [0, totalTraces - 1])
        : [0, 0];
      drawSelectedLayer(s, e);

      console.timeEnd('Total fetchAndPlot');
      console.log('--- fetchAndPlot end ---');
    }

    function shouldPreferWindowFirst() {
      if (ALWAYS_WINDOW_FIRST) return true;
      const win = currentVisibleWindow();
      const plotDiv = document.getElementById('plot');
      if (!win || !plotDiv) return false;
      const { step_x, step_y } = computeStepsForWindow({
        tracesVisible: win.nTraces,
        samplesVisible: win.nSamples,
        widthPx: plotDiv.clientWidth || 1,
        heightPx: plotDiv.clientHeight || 1,
      });
      return (step_x > 1 || step_y > 1);
    }

    function drawSelectedLayer(start = null, end = null) {
      D('DRAW@selectLayer', { layer: document.getElementById('layerSelect')?.value, start, end });
      const sel = document.getElementById('layerSelect');
      const layer = sel ? sel.value : 'raw';
      // ★ 加工レイヤのフル配列は保持しない（window API に任せる）
      latestSeismicData = (layer === 'raw') ? rawSeismicData : null;

      if (shouldPreferWindowFirst()) {
        // 最初からウィンドウ版に任せる（フル描画をスキップ）
        latestSeismicData = null;
        renderLatestView();      // 既存：必要なら状態維持（スピナー等を出すならここ）
        fetchWindowAndPlot();    // 即時フェッチ
        return;
      }

      // 低負荷なら従来どおりフル描画
      const total = latestSeismicData ? latestSeismicData.length : (sectionShape ? sectionShape[0] : 0);
      const s = (typeof start === 'number') ? start : 0;
      const e = (typeof end === 'number') ? end : Math.max(0, total - 1);
      if (latestSeismicData) {
        plotSeismicData(latestSeismicData, defaultDt, s, e);
      } else {
        renderLatestView();
      }
      scheduleWindowFetch();
    }

    function renderWindowWiggle(windowData) {
      D('RENDER@wiggle', { key1: windowData.key1, x: [windowData.x0, windowData.x1],
        y: [windowData.y0, windowData.y1], step: [windowData.stepX, windowData.stepY],
        picksTotal: picks.length });
      if (isRelayouting) {           // ユーザーがドラッグ中
        latestWindowRender = windowData; // 最新結果だけ覚えて
        redrawPending = true;            // 終了後に再描画
        return;
      }
      snapshotAxesRangesFromDOM();
      if (!windowData || (windowData.mode && windowData.mode !== 'wiggle')) return;

      const sel = document.getElementById('layerSelect');
      const currentLayer = sel ? sel.value : 'raw';
      if (windowData.requestedLayer !== currentLayer) return;

      const slider = document.getElementById('key1_slider');
      const idx = slider ? parseInt(slider.value, 10) : 0;
      const key1Val = key1Values[idx];
      if (windowData.key1 !== key1Val) return;

      if (windowData.pipelineKey && (window.latestPipelineKey || null) !== (windowData.pipelineKey || null)) {
        return;
      }

      if (windowData.effectiveLayer === 'fbprob') return;

      const plotDiv = document.getElementById('plot');
      if (!plotDiv) return;

      const { shape, x0, x1, y0, y1 } = windowData;
      const rows = Number(shape?.[0] ?? 0);
      const cols = Number(shape?.[1] ?? 0);
      if (!rows || !cols) return;

      const useI8 = windowData.valuesI8 instanceof Int8Array;
      const useF32 = !useI8 && windowData.values && windowData.values.length != null;
      if (!useI8 && !useF32) return;

      const N = rows * cols;
      if (useI8 && windowData.valuesI8.length < N) return;
      if (useF32 && windowData.values.length < N) return;

      const scale = Number(windowData.scale) || 1;

      setGrid({ x0, stepX: 1, y0, stepY: 1 });
      const dt = window.defaultDt ?? defaultDt;
      const time = new Float32Array(rows);
      for (let r = 0; r < rows; r++) time[r] = (y0 + r * (windowData.stepY || 1)) * dt;

      const traces = [];
      const gain = parseFloat(document.getElementById('gain').value) || 1.0;
      const AMP_LIMIT = 3.0;

      const stepX = windowData.stepX || 1;
      for (let c = 0; c < cols; c++) {
        const baseX = new Float32Array(rows);
        const shiftedFullX = new Float32Array(rows);
        const shiftedPosX = new Float32Array(rows);
        const traceIndex = x0 + c * stepX;
        for (let r = 0; r < rows; r++) {
          const idxVal = r * cols + c;
          let val = (useI8 ? (windowData.valuesI8[idxVal] / scale)
            : windowData.values[idxVal]) * gain;
          if (val > AMP_LIMIT) val = AMP_LIMIT;
          if (val < -AMP_LIMIT) val = -AMP_LIMIT;

          baseX[r] = traceIndex;
          shiftedFullX[r] = traceIndex + val;
          shiftedPosX[r] = traceIndex + (val < 0 ? 0 : val);
        }

        traces.push({ type: 'scatter', mode: 'lines', x: baseX, y: time, line: { width: 0 }, hoverinfo: 'skip', showlegend: false });
        traces.push({ type: 'scatter', mode: 'lines', x: shiftedPosX, y: time, fill: 'tonextx', fillcolor: 'black', line: { width: 0 }, opacity: 0.6, hoverinfo: 'skip', showlegend: false });
        traces.push({ type: 'scatter', mode: 'lines', x: shiftedFullX, y: time, line: { color: 'black', width: 0.5 }, hoverinfo: 'x+y', showlegend: false });

        const [x0v, x1v] = visibleXRng();
        D('RENDER@wiggle:shapes', {
          manualInWin: picks.filter(p => p.trace >= windowData.x0 && p.trace <= windowData.x1).length,
          vis: [x0v, x1v],
        });
      }

      downsampleFactor = 1;
      const endTrace = typeof x1 === 'number' ? x1 : x0 + cols - 1;
      renderedStart = x0;
      renderedEnd = endTrace;

      const totalSamples = sectionShape ? sectionShape[1] : (typeof y1 === 'number' ? y1 - y0 + 1 : rows);
      const layout = buildLayout({
        mode: 'wiggle',
        x0,
        x1: endTrace,
        y0,
        y1,
        stepX: 1,
        stepY: 1,
        totalSamples,
        dt,
        savedXRange,
        savedYRange,
        clickmode: clickModeForCurrentState(),
        dragmode: effectiveDragMode(),
        uirevision: currentUiRevision(),
        fbTitle: null,
      });

      const showPred = !!document.getElementById('showFbPred')?.checked;
      layout.shapes = buildPickShapes({
        manualPicks: picks,
        predicted: showPred ? predictedPicks : [],
        xMin: x0,
        xMax: endTrace,
        showPredicted: showPred,
      });

      withSuppressedRelayout(Plotly.react(plotDiv, traces, layout, {
        responsive: true,
        editable: true,
        modeBarButtonsToAdd: ['eraseshape'],
        edits: { shapePosition: false },
        doubleClick: false,
        doubleClickDelay: 300,
      }));
      setTimeout(() => { withSuppressedRelayout(Plotly.Plots.resize(plotDiv)); }, 50);
      requestAnimationFrame(applyDragMode);
      installPlotlyViewportHandlersOnce();
      attachPickListeners(plotDiv);
      installCustomDoubleClick(plotDiv);
    }

    function renderWindowHeatmap(windowData) {
      D('RENDER@heatmap', { key1: windowData.key1, x: [windowData.x0, windowData.x1],
        y: [windowData.y0, windowData.y1], step: [windowData.stepX, windowData.stepY],
        picksTotal: picks.length });
      if (isRelayouting) {           // ユーザーがドラッグ中
        latestWindowRender = windowData; // 最新結果だけ覚えて
        redrawPending = true;            // 終了後に再描画
        return;
      }
      snapshotAxesRangesFromDOM();
      if (!windowData || (windowData.mode && windowData.mode !== 'heatmap')) return;

      const sel = document.getElementById('layerSelect');
      const currentLayer = sel ? sel.value : 'raw';
      if (windowData.requestedLayer !== currentLayer) return;

      const slider = document.getElementById('key1_slider');
      const idx = slider ? parseInt(slider.value, 10) : 0;
      const key1Val = key1Values[idx];
      if (windowData.key1 !== key1Val) return;

      if (windowData.pipelineKey && (window.latestPipelineKey || null) !== (windowData.pipelineKey || null)) {
        return;
      }

      const plotDiv = document.getElementById('plot');
      if (!plotDiv) return;

      const { shape, x0, x1, y0, y1, effectiveLayer } = windowData;
      let { stepX, stepY } = windowData;
      const rows = Number(shape?.[0] ?? 0);
      const cols = Number(shape?.[1] ?? 0);
      if (!rows || !cols) return;

      const useI8 = windowData.valuesI8 instanceof Int8Array;
      const useF32 = !useI8 && windowData.values && windowData.values.length != null;
      if (!useI8 && !useF32) {
        console.warn('renderWindowHeatmap: missing values');
        return;
      }
      const N = rows * cols;
      if (useI8 && windowData.valuesI8.length < N) return;
      if (useF32 && windowData.values.length < N) return;

      setGrid({ x0, stepX, y0, stepY });
      const gain = parseFloat(document.getElementById('gain').value) || 1.0;
      const AMP_LIMIT = 3.0;
      const fbMode = effectiveLayer === 'fbprob';
      const quantMeta = windowData.quant || (
        (windowData.lo !== undefined && windowData.hi !== undefined)
          ? { mode: windowData.method || 'linear', lo: windowData.lo, hi: windowData.hi, mu: windowData.mu ?? 255 }
          : (windowData.scale != null ? { scale: windowData.scale } : null)
      );
      const fallbackScale = Number(windowData.scale) || 1;

      let zData;
      const poolingCandidate = (
        USE_HEATMAP_POOLING &&
        useI8 &&
        window.SeisHeatmap &&
        typeof window.SeisHeatmap.toPlotlyHeatmapZ === 'function'
      );
      if (poolingCandidate) {
        zData = window.SeisHeatmap.toPlotlyHeatmapZ({
          i8: windowData.valuesI8,
          rows,
          cols,
          quant: quantMeta,
        });
        const backing = zData.backing;
        const total = rows * cols;
        if (fbMode) {
            for (let p = 0; p < total; p++) backing[p] = backing[p] * 255;
          } else {
          }
      }
      else {
        const zRows = new Array(rows);
        const hasLut = !!(quantMeta && 'lo' in quantMeta && 'hi' in quantMeta);
        const lut = hasLut && window.SeisHeatmap && typeof window.SeisHeatmap.getQuantLUT === 'function'
          ? window.SeisHeatmap.getQuantLUT(quantMeta)
          : null;
        let invScale = 1 / (fallbackScale || 1);
        if (quantMeta && 'scale' in (quantMeta || {})) {
          const scaleVal = Number(quantMeta.scale);
          if (Number.isFinite(scaleVal) && scaleVal !== 0) {
            invScale = 1 / scaleVal;
          }
        }
        const srcF32 = useF32 ? windowData.values : null;
        for (let r = 0; r < rows; r++) {
          const row = new Float32Array(cols);
          const offset = r * cols;
          for (let c = 0; c < cols; c++) {
            let rawValue;
            if (useI8) {
              const q = windowData.valuesI8[offset + c];
              if (lut) rawValue = lut[(q + 128) & 0xff];
              else rawValue = q * invScale;
            } else if (srcF32) {
              rawValue = srcF32[offset + c];
            } else {
              rawValue = 0;
            }
            if (fbMode) {
              row[c] = rawValue * 255;
            } else {
              row[c] = rawValue
            }
          }
          zRows[r] = row;
        }
        zData = zRows;
      }

      const xVals = new Float32Array(cols);
      for (let c = 0; c < cols; c++) xVals[c] = x0 + c * stepX;

      const baseDt = window.defaultDt ?? defaultDt;
      const yVals = new Float32Array(rows);
      for (let r = 0; r < rows; r++) yVals[r] = (y0 + r * stepY) * baseDt;

      downsampleFactor = stepY || 1;
      renderedStart = x0;
      renderedEnd = x1;

      const cmName = document.getElementById('colormap')?.value || 'Greys';
      const reverse = document.getElementById('cmReverse')?.checked || false;
      const cm = COLORMAPS[cmName] || 'Greys';
      const isDiv = cmName === 'RdBu' || cmName === 'BWR';
      const g = Math.max(gain, 1e-9);
      const zMin = fbMode ? 0 : -AMP_LIMIT / g;
      const zMax = fbMode ? 255 : AMP_LIMIT / g;
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

      const dt = window.defaultDt ?? defaultDt;
      const layout = buildLayout({
        mode: 'heatmap',
        x0,
        x1,
        y0,
        y1,
        stepX,
        stepY,
        totalSamples: sectionShape ? sectionShape[1] : (y1 - y0 + 1),
        dt,
        savedXRange,
        savedYRange,
        clickmode: clickModeForCurrentState(),
        dragmode: effectiveDragMode(),
        uirevision: currentUiRevision(),
        fbTitle: fbMode ? 'First-break Probability' : null,
      });

      const showPred = !!document.getElementById('showFbPred')?.checked;
      layout.shapes = buildPickShapes({
        manualPicks: picks,
        predicted: showPred ? predictedPicks : [],
        xMin: x0,
        xMax: x1,
        showPredicted: showPred,
      });

      withSuppressedRelayout(Plotly.react(plotDiv, traces, layout, {
        responsive: true,
        editable: true,
        modeBarButtonsToAdd: ['eraseshape'],
        edits: { shapePosition: false },
        doubleClick: false,
        doubleClickDelay: 300,
      }));
      setTimeout(() => { withSuppressedRelayout(Plotly.Plots.resize(plotDiv)); }, 50);
      requestAnimationFrame(applyDragMode);
      installPlotlyViewportHandlersOnce();
      attachPickListeners(plotDiv);
      installCustomDoubleClick(plotDiv);
      const [x0v, x1v] = visibleXRng();
       D('RENDER@wiggle:shapes', {
        manualInWin: picks.filter(p => p.trace >= windowData.x0 && p.trace <= windowData.x1).length,
        vis: [x0v, x1v],
       });
    }

    function renderLatestView(startOverride = null, endOverride = null) {
      const sel = document.getElementById('layerSelect');
      const layer = sel ? sel.value : 'raw';
      const slider = document.getElementById('key1_slider');
      const idx = slider ? parseInt(slider.value, 10) : 0;
      const key1Val = key1Values[idx];

      if (latestSeismicData) {
        const startTrace = typeof startOverride === 'number'
          ? startOverride
          : (typeof renderedStart === 'number' ? renderedStart : 0);
        const endTrace = typeof endOverride === 'number'
          ? endOverride
          : (typeof renderedEnd === 'number'
            ? renderedEnd
            : latestSeismicData.length - 1);
        plotSeismicData(latestSeismicData, defaultDt, startTrace, endTrace);
        return;
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
      params.set('transpose', '1');
      params.set('scaling', currentScaling);

      const requestId = ++windowFetchToken;

      // ---- Abort older in-flight window fetch, then create a new controller
      if (windowFetchCtrl) {
        try { windowFetchCtrl.abort(); } catch (_) { }
      }
      const ctrl = new AbortController();
      windowFetchCtrl = ctrl;

      try {
        const res = await fetch(`/get_section_window_bin?${params.toString()}`, { signal: ctrl.signal });
        if (!res.ok) {
          console.warn('Window fetch failed', res.status);
          return;
        }
        const bin = new Uint8Array(await res.arrayBuffer());
        if (requestId !== windowFetchToken) return; // stale

        const obj = msgpack.decode(bin);
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
        };

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
          return; // canceled on purpose
        }
        if (requestId === windowFetchToken) console.warn('Window fetch error', err);
      } finally {
        if (windowFetchCtrl === ctrl) windowFetchCtrl = null;
      }
    }

    function visibleTraceIndices(range, total) {
      let start = Math.floor(range[0]);
      let end = Math.ceil(range[1]);
      start = Math.max(0, start);
      end = Math.min(total - 1, end);
      return [start, end];
    }

    function clickModeForCurrentState() {
      return isPickMode ? 'event' : 'event+select';
    }


    function plotSeismicData(seismic, dt, startTrace = 0, endTrace = seismic.length - 1) {
      snapshotAxesRangesFromDOM();
      const totalTraces = seismic.length;
      startTrace = Math.max(0, startTrace);
      endTrace = Math.min(totalTraces - 1, endTrace);
      const nTraces = endTrace - startTrace + 1;
      const nSamples = seismic[0].length;
      const plotDiv = document.getElementById('plot');

      const widthPx = plotDiv.clientWidth || 1;
      const xRange = savedXRange ?? [0, totalTraces - 1];
      const visibleTraces = endTrace - startTrace + 1;
      const density = visibleTraces / widthPx;
      const mode = document.getElementById('layerSelect').value;
      const fbMode = mode === 'fbprob'; // 使っていないが残置可

      let traces = [];
      const gain = parseFloat(document.getElementById('gain').value) || 1.0;
      const AMP_LIMIT = 3.0;
      let defaultXRange;
      let defaultYRange;

      if (!fbMode && density < WIGGLE_DENSITY_THRESHOLD) {
        downsampleFactor = 1;
        setGrid({ x0: startTrace, stepX: 1, y0: 0, stepY: 1 });
        const time = new Float32Array(nSamples);
        for (let t = 0; t < nSamples; t++) time[t] = t * dt;
        for (let i = startTrace; i <= endTrace; i++) {
          const raw = seismic[i];
          const baseX = new Float32Array(nSamples);
          const shiftedFullX = new Float32Array(nSamples);
          const shiftedPosX = new Float32Array(nSamples);
          for (let j = 0; j < nSamples; j++) {
            let val = raw[j] * gain ;
            if (val > AMP_LIMIT) val = AMP_LIMIT;
            if (val < -AMP_LIMIT) val = -AMP_LIMIT;
            baseX[j] = i;
            shiftedFullX[j] = val + i;
            shiftedPosX[j] = (val < 0 ? 0 : val) + i;
          }

          traces.push({ type: 'scatter', mode: 'lines', x: baseX, y: time, line: { width: 0 }, hoverinfo: 'x+y', showlegend: false });
          traces.push({ type: 'scatter', mode: 'lines', x: shiftedPosX, y: time, fill: 'tonextx', fillcolor: 'black', line: { width: 0 }, opacity: 0.6, hoverinfo: 'skip', showlegend: false });
          traces.push({ type: 'scatter', mode: 'lines', x: shiftedFullX, y: time, line: { color: 'black', width: 0.5 }, hoverinfo: 'skip', showlegend: false });
        }
        defaultXRange = [startTrace, endTrace];
        defaultYRange = [nSamples * dt, 0];
      } else {
        const MAX_POINTS = 3_000_000;
        let factor = 1;
        while (Math.floor(nTraces / factor) * Math.floor(nSamples / factor) > MAX_POINTS) factor++;
        const nTracesDS = Math.floor(nTraces / factor);
        const nSamplesDS = Math.floor(nSamples / factor);
        console.log('Downsampling factor:', factor);
        console.log('Final dimensions:', nTracesDS, 'x', nSamplesDS);

        setGrid({ x0: startTrace, stepX: factor, y0: 0, stepY: factor });
        const time = new Float32Array(nSamplesDS);
        for (let t = 0; t < nSamplesDS; t++) time[t] = t * dt * factor;

        const zData = Array.from({ length: nSamplesDS }, () => new Float32Array(nTracesDS));

        for (let i = startTrace, col = 0; col < nTracesDS; i += factor, col++) {
          const trace = seismic[i];
          for (let j = 0, row = 0; row < nSamplesDS; j += factor, row++) {
            if (fbMode) {
              const val = trace[j] * 255;
              zData[row][col] = val;
            } else {
              zData[row][col] = trace[j];
            }
          }
        }

        const g = Math.max(gain, 1e-9);
        const zMin = fbMode ? 0 : -AMP_LIMIT / g;
        const zMax = fbMode ? 255 : AMP_LIMIT / g;

        const xVals = new Float32Array(nTracesDS);
        for (let i = 0; i < nTracesDS; i++) xVals[i] = startTrace + i * factor;
        downsampleFactor = factor;

        const cmName = document.getElementById('colormap')?.value || 'Greys';
        const reverse = document.getElementById('cmReverse')?.checked || false;
        const cm = COLORMAPS[cmName] || 'Greys';
        const isDiv = cmName === 'RdBu' || cmName === 'BWR';

        traces = [{
          type: 'heatmap',
          x: xVals,
          y: time,
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
        const halfX = factor * 0.5;
        const halfY = dt * factor * 0.5;
        defaultXRange = [startTrace - halfX, (startTrace + (nTracesDS - 1) * factor) + halfX];
        defaultYRange = [ (nSamplesDS * dt * factor) - halfY, 0 - halfY ];
      }

      const layout = {
        xaxis: {
          title: 'Trace', showgrid: false, tickfont: { color: '#000' }, titlefont: { color: '#000' },
          autorange: false, range: savedXRange ?? defaultXRange
        },
        yaxis: {
          title: 'Time (s)', showgrid: false, tickfont: { color: '#000' }, titlefont: { color: '#000' },
          autorange: false, range: savedYRange ?? defaultYRange
        },
        clickmode: clickModeForCurrentState(),
        uirevision: currentUiRevision(),
        paper_bgcolor: '#fff', plot_bgcolor: '#fff',
        margin: { t: 10, r: 10, l: 60, b: 40 },
        dragmode: effectiveDragMode(),
        ...(fbMode ? { title: 'First-break Probability' } : {}),
      };

      const manualShapes = picks.map(p => ({
        type: 'line',
        x0: p.trace - 0.4, x1: p.trace + 0.4,
        y0: p.time, y1: p.time,
        line: { color: 'red', width: 2 }
      }));

      const showPred = document.getElementById('showFbPred')?.checked;
      const predShapes = (showPred ? predictedPicks : [])
        .filter(p => p.trace >= startTrace && p.trace <= endTrace)
        .map(p => ({
          type: 'line',
          x0: p.trace - 0.4, x1: p.trace + 0.4,
          y0: p.time, y1: p.time,
          line: { color: '#1f77b4', width: 5, dash: 'dot' }
        }));

      layout.shapes = [...manualShapes, ...predShapes];

      withSuppressedRelayout(Plotly.react(plotDiv, traces, layout, {
        responsive: true,
        editable: true,
        modeBarButtonsToAdd: ['eraseshape'],
        edits: { shapePosition: false },
        doubleClick: false,
        doubleClickDelay: 300
      }));
      setTimeout(() => {
          withSuppressedRelayout(Plotly.Plots.resize(plotDiv));
        }, 50);
      requestAnimationFrame(applyDragMode);
      renderedStart = startTrace;
      renderedEnd = endTrace;
      console.log(`Rendered traces ${startTrace}-${endTrace}`);
      installPlotlyViewportHandlersOnce();
      attachPickListeners(plotDiv);
      installCustomDoubleClick(plotDiv);
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
          setTimeout(() => {
            try { renderLatestView(); } catch (e) { console.warn('renderLatestView failed', e); }
          }, 0);
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
          renderLatestView();
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
        renderLatestView();
      } finally {
        handlePickNormalized._busy = false;
        const next = handlePickNormalized._queued;
        handlePickNormalized._queued = null;
        if (next) setTimeout(() => handlePickNormalized(next), 0);
      }
    }

    async function handleRelayout(ev) {
      if (suppressRelayout) return;

      D('RELAYOUT@begin', { keys: Object.keys(ev), isRelayouting, pickMode: isPickMode });

      const gd = document.getElementById('plot');

      // range 更新
      if ('xaxis.range[0]' in ev && 'xaxis.range[1]' in ev) {
        savedXRange = [ev['xaxis.range[0]'], ev['xaxis.range[1]']];
      }
      if ('yaxis.range[0]' in ev && 'yaxis.range[1]' in ev) {
        const y0 = ev['yaxis.range[0]'];
        const y1 = ev['yaxis.range[1]'];
        savedYRange = y0 > y1 ? [y0, y1] : [y1, y0];
      }

      // autorange のときはレンジ再取得して終了（shape同期はしない）
      if ('xaxis.autorange' in ev || 'yaxis.autorange' in ev) {
        await new Promise(r => requestAnimationFrame(r));
        snapshotAxesRangesFromDOM();
        checkModeFlipAndRefetch();
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
      const overlay = document.getElementById('ppOverlay');
      const statusEl = document.getElementById('ppStatus');
      const barInner = document.getElementById('ppBarInner');
      const cancelBtn = document.getElementById('ppCancelBtn');

      if (!overlay || !statusEl || !barInner || !cancelBtn) {
        return;
      }

      let totalStepsCount = 1;

      function setText(msg) {
        statusEl.textContent = msg || '';
      }

      function setProgress(pct) {
        const clamped = Math.max(0, Math.min(100, Math.round(pct)));
        barInner.style.width = `${clamped}%`;
      }

      function openOverlay(totalSteps = 1, initialText = 'Preparing…') {
        totalStepsCount = Math.max(1, Number(totalSteps) || 1);
        overlay.classList.add('show');
        setText(initialText);
        setProgress(0);
      }

      function stepOverlay(index, name) {
        const idx = Math.max(0, Number(index) || 0);
        const pct = Math.round((idx / totalStepsCount) * 100);
        const label = name ? `(${idx}/${totalStepsCount}) ${name}` : `Step ${idx}/${totalStepsCount}`;
        setText(label);
        setProgress(pct);
      }

      function closeOverlay() {
        overlay.classList.remove('show');
      }

      function errorOverlay(message) {
        setText(`Error: ${message}`);
        setProgress(100);
        setTimeout(() => closeOverlay(), 1200);
      }

      const progressApi = {
        open: openOverlay,
        progress: setProgress,
        step: stepOverlay,
        text: setText,
        close: closeOverlay,
        error: errorOverlay,
      };

      window.pipelineProgress = progressApi;

      cancelBtn.addEventListener('click', () => {
        try {
          if (window.pipelineUI && typeof window.pipelineUI.cancel === 'function') {
            window.pipelineUI.cancel();
          }
        } catch (err) {
          console.warn('pipeline cancel threw', err);
        }
        closeOverlay();
      });

      function attachPipelineHandlers() {
        const ui = window.pipelineUI;
        if (!ui) return;

        if (typeof ui.on === 'function') {
          ui.on('run:start', (event) => {
            const total = event && typeof event.totalSteps === 'number' ? event.totalSteps : 1;
            openOverlay(total, 'Submitting…');
          });
          ui.on('run:step', (event) => {
            const idx = event && typeof event.index === 'number' ? event.index : 0;
            const name = event && typeof event.name === 'string' ? event.name : 'Processing…';
            stepOverlay(idx, name);
          });
          ui.on('run:finish', (event) => {
            const total = event && typeof event.totalSteps === 'number' ? event.totalSteps : totalStepsCount;
            stepOverlay(total, 'Done');
            setTimeout(() => closeOverlay(), 400);
          });
          ui.on('run:error', (event) => {
            const message = event && typeof event.message === 'string' ? event.message : 'failed';
            errorOverlay(message);
          });
          return;
        }

        if (typeof ui.run === 'function' && !ui.__progressWrapped) {
          const originalRun = ui.run.bind(ui);
          ui.__progressWrapped = true;
          ui.run = async function progressWrappedRun(...args) {
            openOverlay(1, 'Running pipeline…');
            try {
              const result = await originalRun(...args);
              setProgress(100);
              setTimeout(() => closeOverlay(), 400);
              return result;
            } catch (err) {
              errorOverlay((err && err.message) || 'failed');
              throw err;
            }
          };
        }
      }

      attachPipelineHandlers();
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
          rawSeismicData = null;
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

          cache.clear();


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
