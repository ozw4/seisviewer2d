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
    function withSuppressedRelayout(promiseLike) {
        suppressRelayout = true;
        if (promiseLike && typeof promiseLike.finally === 'function') {
            return promiseLike.finally(() => { suppressRelayout = false; });
          }
        suppressRelayout = false;
        return promiseLike;
      }
    function maybeResizePlot(plotDiv, force) {
      if (!plotDiv) return Promise.resolve(false);

      const w = Number(plotDiv.clientWidth);
      const h = Number(plotDiv.clientHeight);
      if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
        return Promise.resolve(false);
      }

      const prev = plotDiv.__svLastSize;
      const changed = !prev || prev.w !== w || prev.h !== h;
      if (!force && !changed) return Promise.resolve(false);

      plotDiv.__svLastSize = { w, h };
      return withSuppressedRelayout(Promise.resolve(Plotly.Plots.resize(plotDiv)));
    }
    function resolveWiggleTraceIndices(gd) {
      const data = Array.isArray(gd?.data) ? gd.data : [];
      const idxs = [];
      for (let i = 0; i < data.length; i++) {
        const tr = data[i];
        if (tr && tr.meta && tr.meta.svRole === 'wiggle') idxs.push(i);
      }
      return idxs;
    }
    function resolvePickTraceIndices(gd) {
      const data = Array.isArray(gd?.data) ? gd.data : [];
      const isPickIdx = (idx, kind) => (
        Number.isInteger(idx) &&
        idx >= 0 &&
        idx < data.length &&
        data[idx] &&
        data[idx].meta &&
        data[idx].meta.svRole === 'pick' &&
        data[idx].meta.svKind === kind
      );

      const cachedManual = gd ? gd.__svPickIdxManual : -1;
      const cachedPred = gd ? gd.__svPickIdxPred : -1;
      const cachedPending = gd ? gd.__svPickIdxPending : -1;
      if (
        isPickIdx(cachedManual, 'manual') &&
        isPickIdx(cachedPred, 'pred') &&
        isPickIdx(cachedPending, 'pending')
      ) {
        return { manualIdx: cachedManual, predIdx: cachedPred, pendingIdx: cachedPending };
      }

      let manualIdx = -1;
      let predIdx = -1;
      let pendingIdx = -1;
      for (let i = 0; i < data.length; i++) {
        const tr = data[i];
        if (!tr || !tr.meta || tr.meta.svRole !== 'pick') continue;
        if (tr.meta.svKind === 'manual' && manualIdx < 0) manualIdx = i;
        if (tr.meta.svKind === 'pred' && predIdx < 0) predIdx = i;
        if (tr.meta.svKind === 'pending' && pendingIdx < 0) pendingIdx = i;
      }
      if (gd) {
        gd.__svPickIdxManual = manualIdx;
        gd.__svPickIdxPred = predIdx;
        gd.__svPickIdxPending = pendingIdx;
      }
      return { manualIdx, predIdx, pendingIdx };
    }
    window.resolvePickTraceIndices = resolvePickTraceIndices;
    function makeWiggleSig(opts) {
      const styleKey = 'base:line0:skip|fill:toself:black:0.6:line0:skip|line:black:0.5:x+y';
      return JSON.stringify({
        displayTraceCount: opts.displayTraceCount,
        plotTraceCount: opts.plotTraceCount,
        pointsPerTrace: opts.pointsPerTrace,
        stepX: opts.stepX,
        stepY: opts.stepY,
        ampSource: opts.ampSource,
        styleKey,
      });
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
        const shouldRefreshPendingOverlay = (
          typeof window.hasPendingPickOverlayState === 'function' &&
          window.hasPendingPickOverlayState()
        );
        if (
          typeof schedulePickOverlayUpdate === 'function' &&
          (
            (typeof pickOverlayDirty !== 'undefined' && pickOverlayDirty) ||
            shouldRefreshPendingOverlay
          )
        ) {
          schedulePickOverlayUpdate();
        }
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
      const perf = windowData.__perf || null;
      const perfEnabled = window.SV_PERF === true;
      let tPrep0 = null;
      let tPrep1 = null;
      let tPlot0 = null;
      let plotPromise = null;

      const useI8 = windowData.valuesI8 instanceof Int8Array;
      const useF32 = !useI8 && windowData.values && windowData.values.length != null;
      if (!useI8 && !useF32) return;

      const N = rows * cols;
      if (useI8 && windowData.valuesI8.length < N) return;
      if (useF32 && windowData.values.length < N) return;

      const scale = Number(windowData.scale) || 1;
      const stepX = windowData.stepX || 1;
      const stepY = windowData.stepY || 1;
      const expectedTraceCount = 3;
      const wiggleSig = makeWiggleSig({
        displayTraceCount: cols,
        plotTraceCount: expectedTraceCount,
        pointsPerTrace: rows,
        stepX,
        stepY,
        ampSource: useI8 ? 'scaled-int8' : 'float32',
      });
      const prevWiggleSig = typeof plotDiv.__svWiggleSig === 'string' ? plotDiv.__svWiggleSig : null;
      const prevMode = plotDiv.__svPlotMode;
      const hasPlotData = Array.isArray(plotDiv.data) && plotDiv.data.length > 0;
      const wiggleIdxs = resolveWiggleTraceIndices(plotDiv);
      const pickTraceIdxs = resolvePickTraceIndices(plotDiv);
      const hasPickTraces = pickTraceIdxs.manualIdx >= 0 && pickTraceIdxs.predIdx >= 0 && pickTraceIdxs.pendingIdx >= 0;
      if (!hasPickTraces && hasPlotData && prevMode === 'wiggle') {
        console.warn('[RENDER@wiggle][PICKS] missing pick traces; forcing react init');
      }
      const needsReactInit = (
        !hasPlotData ||
        prevMode !== 'wiggle' ||
        prevWiggleSig !== wiggleSig ||
        wiggleIdxs.length !== expectedTraceCount ||
        !hasPickTraces
      );

      if (perfEnabled) tPrep0 = performance.now();
      setGrid({ x0, stepX: 1, y0, stepY: 1 });
      const dt = window.defaultDt ?? defaultDt;
      const time = new Float32Array(rows);
      for (let r = 0; r < rows; r++) time[r] = (y0 + r * stepY) * dt;

      const traces = needsReactInit ? [] : null;
      const gain = parseFloat(document.getElementById('gain').value) || 1.0;
      const AMP_LIMIT = 3.0;
      const lineSegLen = rows + 1;
      const fillSegLen = (2 * rows) + 2;
      const lenLine = cols * lineSegLen;
      const lenFill = cols * fillSegLen;
      const baseX = new Float32Array(lenLine);
      const baseY = new Float32Array(lenLine);
      const lineX = new Float32Array(lenLine);
      const lineY = new Float32Array(lenLine);
      const fillX = new Float32Array(lenFill);
      const fillY = new Float32Array(lenFill);

      for (let c = 0; c < cols; c++) {
        const traceIndex = x0 + c * stepX;
        const lineStart = c * lineSegLen;
        const fillStart = c * fillSegLen;
        for (let r = 0; r < rows; r++) {
          const idxVal = r * cols + c;
          let val = (useI8 ? (windowData.valuesI8[idxVal] / scale)
            : windowData.values[idxVal]) * gain;
          if (val > AMP_LIMIT) val = AMP_LIMIT;
          if (val < -AMP_LIMIT) val = -AMP_LIMIT;

          const lineIdx = lineStart + r;
          const fillBaseIdx = fillStart + r;
          const fillPosIdx = fillStart + rows + (rows - 1 - r);
          const posVal = val < 0 ? 0 : val;

          baseX[lineIdx] = traceIndex;
          baseY[lineIdx] = time[r];
          lineX[lineIdx] = traceIndex + val;
          lineY[lineIdx] = time[r];

          fillX[fillBaseIdx] = traceIndex;
          fillY[fillBaseIdx] = time[r];
          fillX[fillPosIdx] = traceIndex + posVal;
          fillY[fillPosIdx] = time[r];
        }

        const lineNanIdx = lineStart + rows;
        baseX[lineNanIdx] = NaN;
        baseY[lineNanIdx] = NaN;
        lineX[lineNanIdx] = NaN;
        lineY[lineNanIdx] = NaN;

        const fillCloseIdx = fillStart + (2 * rows);
        const fillNanIdx = fillCloseIdx + 1;
        fillX[fillCloseIdx] = traceIndex;
        fillY[fillCloseIdx] = time[0];
        fillX[fillNanIdx] = NaN;
        fillY[fillNanIdx] = NaN;
      }

      const wiggleX = [baseX, fillX, lineX];
      const wiggleY = [baseY, fillY, lineY];

      if (needsReactInit) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: baseX,
          y: baseY,
          line: { width: 0 },
          connectgaps: false,
          hoverinfo: 'skip',
          showlegend: false,
          meta: { svRole: 'wiggle' },
        });
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: fillX,
          y: fillY,
          fill: 'toself',
          fillcolor: 'black',
          line: { width: 0 },
          opacity: 0.6,
          connectgaps: false,
          hoverinfo: 'skip',
          showlegend: false,
          meta: { svRole: 'wiggle' },
        });
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: lineX,
          y: lineY,
          line: { color: 'black', width: 0.5 },
          connectgaps: false,
          hoverinfo: 'x+y',
          showlegend: false,
          meta: { svRole: 'wiggle' },
        });
      }

      downsampleFactor = 1;
      const endTrace = typeof x1 === 'number' ? x1 : x0 + cols - 1;
      renderedStart = x0;
      renderedEnd = endTrace;

      const showPred = !!document.getElementById('showFbPred')?.checked;
      const [manualPickTr, predPickTr] = buildPickMarkerTraces({
        manualPicks: picks,
        predicted: showPred ? predictedPicks : [],
        xMin: x0,
        xMax: endTrace,
        showPredicted: showPred,
      });
      const pendingPickTr = buildPendingPickMarkerTrace({
        pending: typeof window.getPendingPickOverlayState === 'function'
          ? window.getPendingPickOverlayState()
          : null,
        yMin: Math.min(time[0], time[rows - 1]),
        yMax: Math.max(time[0], time[rows - 1]),
      });
      const pickManualCount = manualPickTr.x ? manualPickTr.x.length : 0;
      const pickPredCount = predPickTr.x ? predPickTr.x.length : 0;
      const [x0v, x1v] = visibleXRng();
      D('RENDER@picks', {
        mode: 'wiggle',
        manualInWin: pickManualCount,
        predInWin: pickPredCount,
        vis: [x0v, x1v],
      });
      if (perfEnabled) tPrep1 = performance.now();
      if (needsReactInit) {
        traces.push(manualPickTr, predPickTr, pendingPickTr);
        plotDiv.__svPickIdxManual = traces.length - 3;
        plotDiv.__svPickIdxPred = traces.length - 2;
        plotDiv.__svPickIdxPending = traces.length - 1;

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

        if (perfEnabled) tPlot0 = performance.now();
        plotPromise = withSuppressedRelayout(Plotly.react(plotDiv, traces, layout, {
          responsive: true,
          doubleClick: false,
          doubleClickDelay: 300,
        }));
        setTimeout(() => { maybeResizePlot(plotDiv, true); }, 50);
      } else {
        const diffUpdatePromise = Promise.resolve()
          .then(() => Plotly.restyle(plotDiv, {
            x: wiggleX,
            y: wiggleY,
          }, wiggleIdxs))
          .then(() => {
            const { manualIdx, predIdx, pendingIdx } = resolvePickTraceIndices(plotDiv);
            if (manualIdx < 0 || predIdx < 0 || pendingIdx < 0) {
              console.warn('[RENDER@wiggle][PICKS] pick traces missing on restyle path');
              return;
            }
            return Plotly.restyle(plotDiv, {
              x: [manualPickTr.x, predPickTr.x, pendingPickTr.x],
              y: [manualPickTr.y, predPickTr.y, pendingPickTr.y],
              visible: [true, !!showPred, !!pendingPickTr.visible],
              mode: [manualPickTr.mode, predPickTr.mode, pendingPickTr.mode],
              marker: [manualPickTr.marker, predPickTr.marker, pendingPickTr.marker],
              line: [manualPickTr.line, predPickTr.line, pendingPickTr.line],
            }, [manualIdx, predIdx, pendingIdx]);
          });
        if (perfEnabled) tPlot0 = performance.now();
        plotPromise = withSuppressedRelayout(diffUpdatePromise);
      }
      if (perfEnabled && plotPromise && typeof plotPromise.then === 'function') {
        plotPromise.then(() => {
          const tDone = performance.now();
          window.svPerfLog({
            kind: 'window',
            mode: 'wiggle',
            plot: needsReactInit ? 'react' : 'restyle',
            rows,
            cols,
            stepX,
            stepY,
            pick_manual: pickManualCount,
            pick_pred: pickPredCount,
            fetch_ms: perf ? (perf.tBuf - perf.tReq0) : null,
            decode_ms: perf ? (perf.tDec1 - perf.tDec0) : null,
            prep_ms: tPrep1 - tPrep0,
            plotly_ms: tDone - tPlot0,
            total_ms: perf ? (tDone - perf.tReq0) : null,
            bytes: perf ? perf.bytes : null,
          });
        });
      }
      requestAnimationFrame(applyDragMode);
      installPlotlyViewportHandlersOnce();
      attachPickListeners(plotDiv);
      installCustomDoubleClick(plotDiv);
      plotDiv.__svPlotMode = 'wiggle';
      plotDiv.__svWiggleSig = wiggleSig;
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

      const resolveHeatmapTraceIndex = (gd) => {
        if (typeof window.heatmapTraceIndex === 'function') {
          const idxFromHelper = window.heatmapTraceIndex(gd);
          if (Number.isInteger(idxFromHelper) && idxFromHelper >= 0) return idxFromHelper;
        }
        const data = Array.isArray(gd?.data) ? gd.data : [];
        for (let i = 0; i < data.length; i++) {
          const tr = data[i];
          if (tr && tr.type === 'heatmap') return i;
        }
        return -1;
      };
      const heatIdx = resolveHeatmapTraceIndex(plotDiv);
      const prevMode = plotDiv.__svPlotMode;
      const pickTraceIdxs = resolvePickTraceIndices(plotDiv);
      const hasPickTraces = pickTraceIdxs.manualIdx >= 0 && pickTraceIdxs.predIdx >= 0 && pickTraceIdxs.pendingIdx >= 0;
      if (!hasPickTraces && Array.isArray(plotDiv.data) && plotDiv.data.length > 0 && prevMode === 'heatmap') {
        console.warn('[RENDER@heatmap][PICKS] missing pick traces; forcing react init');
      }
      const needsReactInit = heatIdx < 0 || prevMode !== 'heatmap' || !hasPickTraces;

      const { shape, x0, x1, y0, y1, effectiveLayer } = windowData;
      let { stepX, stepY } = windowData;
      const rows = Number(shape?.[0] ?? 0);
      const cols = Number(shape?.[1] ?? 0);
      if (!rows || !cols) return;
      const perf = windowData.__perf || null;
      const perfEnabled = window.SV_PERF === true;
      let tLut0 = null;
      let tLut1 = null;
      let tPlot0 = null;
      let plotPromise = null;

      const N = rows * cols;
      const hasWorkerRows = Array.isArray(windowData.zRows) && windowData.zRows.length === rows;
      const hasWorkerBacking = windowData.zBacking instanceof Float32Array && windowData.zBacking.length >= N;
      const useI8 = !hasWorkerRows && !hasWorkerBacking && windowData.valuesI8 instanceof Int8Array;
      const useF32 = !hasWorkerRows && !hasWorkerBacking && !useI8 && windowData.values && windowData.values.length != null;
      if (!hasWorkerRows && !hasWorkerBacking && !useI8 && !useF32) {
        console.warn('renderWindowHeatmap: missing values');
        return;
      }
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

      if (perfEnabled) tLut0 = performance.now();
      let zData;
      let poolingCandidate = false;
      if (hasWorkerRows) {
        zData = windowData.zRows;
      } else if (hasWorkerBacking) {
        const zRows = new Array(rows);
        const backing = windowData.zBacking;
        for (let r = 0; r < rows; r++) {
          zRows[r] = backing.subarray(r * cols, (r + 1) * cols);
        }
        windowData.zRows = zRows;
        zData = zRows;
      } else {
        poolingCandidate = (
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
          }
        } else {
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
                row[c] = rawValue;
              }
            }
            zRows[r] = row;
          }
          zData = zRows;
        }
      }
      if (perfEnabled) tLut1 = performance.now();

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
      const zMid = (!fbMode && isDiv) ? 0 : null;
      const fbTitle = fbMode ? 'First-break Probability' : null;

      const showPred = !!document.getElementById('showFbPred')?.checked;
      const [manualPickTr, predPickTr] = buildPickMarkerTraces({
        manualPicks: picks,
        predicted: showPred ? predictedPicks : [],
        xMin: x0,
        xMax: x1,
        showPredicted: showPred,
      });
      const pendingPickTr = buildPendingPickMarkerTrace({
        pending: typeof window.getPendingPickOverlayState === 'function'
          ? window.getPendingPickOverlayState()
          : null,
        yMin: Math.min(yVals[0], yVals[rows - 1]),
        yMax: Math.max(yVals[0], yVals[rows - 1]),
      });
      const pickManualCount = manualPickTr.x ? manualPickTr.x.length : 0;
      const pickPredCount = predPickTr.x ? predPickTr.x.length : 0;
      const [x0v, x1v] = visibleXRng();
      D('RENDER@picks', {
        mode: 'heatmap',
        manualInWin: pickManualCount,
        predInWin: pickPredCount,
        vis: [x0v, x1v],
      });

      if (needsReactInit) {
        const traces = [{
          type: 'heatmap',
          x: xVals,
          y: yVals,
          z: zData,
          colorscale: cm,
          reversescale: reverse,
          zmin: zMin,
          zmax: zMax,
          zmid: zMid,
          showscale: false,
          hoverinfo: 'x+y',
          hovertemplate: '',
        }];
        traces.push(manualPickTr, predPickTr, pendingPickTr);
        plotDiv.__svPickIdxManual = traces.length - 3;
        plotDiv.__svPickIdxPred = traces.length - 2;
        plotDiv.__svPickIdxPending = traces.length - 1;

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
          fbTitle,
        });

        if (perfEnabled) tPlot0 = performance.now();
        plotPromise = withSuppressedRelayout(Plotly.react(plotDiv, traces, layout, {
          responsive: true,
          doubleClick: false,
          doubleClickDelay: 300,
        }));
        setTimeout(() => { maybeResizePlot(plotDiv, true); }, 50);
      } else {
        const diffUpdatePromise = Promise.resolve()
          .then(() => Plotly.restyle(plotDiv, {
            x: [xVals],
            y: [yVals],
            z: [zData],
            colorscale: [cm],
            reversescale: [reverse],
            zmin: [zMin],
            zmax: [zMax],
            zmid: [zMid],
          }, [heatIdx]))
          .then(() => {
            const { manualIdx, predIdx, pendingIdx } = resolvePickTraceIndices(plotDiv);
            if (manualIdx < 0 || predIdx < 0 || pendingIdx < 0) {
              console.warn('[RENDER@heatmap][PICKS] pick traces missing on restyle path');
              return;
            }
            return Plotly.restyle(plotDiv, {
              x: [manualPickTr.x, predPickTr.x, pendingPickTr.x],
              y: [manualPickTr.y, predPickTr.y, pendingPickTr.y],
              visible: [true, !!showPred, !!pendingPickTr.visible],
              mode: [manualPickTr.mode, predPickTr.mode, pendingPickTr.mode],
              marker: [manualPickTr.marker, predPickTr.marker, pendingPickTr.marker],
              line: [manualPickTr.line, predPickTr.line, pendingPickTr.line],
            }, [manualIdx, predIdx, pendingIdx]);
          })
          .then(() => Plotly.relayout(plotDiv, {
            title: fbTitle ?? '',
          }));
        if (perfEnabled) tPlot0 = performance.now();
        plotPromise = withSuppressedRelayout(diffUpdatePromise);
      }
      if (perfEnabled && plotPromise && typeof plotPromise.then === 'function') {
        plotPromise.then(() => {
          const tDone = performance.now();
          window.svPerfLog({
            kind: 'window',
            mode: 'heatmap',
            plot: needsReactInit ? 'react' : 'restyle',
            pool: poolingCandidate,
            rows,
            cols,
            stepX,
            stepY,
            pick_manual: pickManualCount,
            pick_pred: pickPredCount,
            fetch_ms: perf ? (perf.tBuf - perf.tReq0) : null,
            decode_ms: perf ? (perf.tDec1 - perf.tDec0) : null,
            lut_ms: tLut1 - tLut0,
            plotly_ms: tDone - tPlot0,
            total_ms: perf ? (tDone - perf.tReq0) : null,
            bytes: perf ? perf.bytes : null,
          });
        });
      }
      requestAnimationFrame(applyDragMode);
      installPlotlyViewportHandlersOnce();
      attachPickListeners(plotDiv);
      installCustomDoubleClick(plotDiv);
      plotDiv.__svPlotMode = 'heatmap';
    }
