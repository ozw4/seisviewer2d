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
