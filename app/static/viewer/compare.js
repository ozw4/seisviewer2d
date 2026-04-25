(function () {
  const AXIS_MARGIN = 0.035;
  const AMP_LIMIT = 3.0;
  let latestCompareRender = null;
  let compareSyncing = false;

  class CompareFetchError extends Error {
    constructor(source, status, detail) {
      const label = source && source.role === 'A' ? 'A' : 'B';
      const suffix = detail ? `: ${detail}` : '';
      super(`${label} source fetch failed (${status})${suffix}`);
      this.name = 'CompareFetchError';
      this.source = source;
      this.status = status;
      this.detail = detail || '';
    }
  }

  function getCompareNodes() {
    return {
      toggle: document.getElementById('compareModeToggle'),
      sourceA: document.getElementById('compareSourceA'),
      sourceB: document.getElementById('compareSourceB'),
      showDiff: document.getElementById('compareShowDiff'),
      status: document.getElementById('compareStatus'),
    };
  }

  function setCompareStatus(message) {
    const { status } = getCompareNodes();
    if (!status) return;
    const text = String(message || '').trim();
    status.textContent = text;
    status.hidden = text.length === 0;
  }

  function isCompareModeEnabled() {
    return !!document.getElementById('compareModeToggle')?.checked;
  }

  function compareShowDiffEnabled() {
    return !!document.getElementById('compareShowDiff')?.checked;
  }

  function optionValues(select) {
    if (!select) return ['raw'];
    const values = Array.from(select.options || [])
      .map((opt) => opt.value || opt.textContent || '')
      .filter(Boolean);
    return values.length ? values : ['raw'];
  }

  function getLayerSourceOptions() {
    const layerSelect = document.getElementById('layerSelect');
    const values = optionValues(layerSelect);
    const unique = [];
    for (const value of ['raw', ...values]) {
      if (!unique.includes(value)) unique.push(value);
    }
    return unique;
  }

  function fillSourceSelect(select, values, preferred, fallback) {
    if (!select) return;
    const previous = preferred || select.value || '';
    select.innerHTML = '';
    for (const value of values) {
      select.appendChild(new Option(value, value));
    }
    const target = values.includes(previous)
      ? previous
      : (values.includes(fallback) ? fallback : values[0]);
    select.value = target || 'raw';
  }

  function updateCompareSourceOptions() {
    const { sourceA, sourceB } = getCompareNodes();
    const values = getLayerSourceOptions();
    const firstTap = values.find((value) => value !== 'raw') || 'raw';
    fillSourceSelect(sourceA, values, sourceA?.value || 'raw', 'raw');
    fillSourceSelect(sourceB, values, sourceB?.value || firstTap, firstTap);
  }

  function resolveSourceDomain(sourceId) {
    if (!sourceId || sourceId === 'raw') return 'amplitude';
    const tapData = window.latestTapData && window.latestTapData[sourceId];
    if (tapData && typeof tapData === 'object') {
      const meta = tapData.meta;
      if (meta && typeof meta.domain === 'string') return meta.domain;
      if (Object.prototype.hasOwnProperty.call(tapData, 'prob')) return 'probability';
    }
    const lowered = String(sourceId).toLowerCase();
    if (lowered.includes('fbpick') || lowered.includes('prob')) return 'probability';
    return 'amplitude';
  }

  function resolveCompareSource(select, role) {
    const value = select?.value || 'raw';
    const isRaw = value === 'raw';
    const pipelineKey = isRaw ? null : (window.latestPipelineKey || null);
    return {
      role,
      id: value,
      label: value,
      pipelineKey,
      tapLabel: isRaw ? null : value,
      domain: resolveSourceDomain(value),
      available: isRaw || !!pipelineKey,
    };
  }

  function getCompareSources() {
    updateCompareSourceOptions();
    const { sourceA, sourceB } = getCompareNodes();
    return {
      a: resolveCompareSource(sourceA, 'A'),
      b: resolveCompareSource(sourceB, 'B'),
    };
  }

  function currentCompareKey1() {
    const slider = document.getElementById('key1_slider');
    const idx = slider ? parseInt(slider.value, 10) : 0;
    return Array.isArray(key1Values) ? key1Values[idx] : undefined;
  }

  function sourcePairKey(sources) {
    return `${sources.a.id}|${sources.b.id}|${sources.a.pipelineKey || ''}|${sources.b.pipelineKey || ''}`;
  }

  function canAttemptDiff(sources) {
    return sources.a.domain === sources.b.domain;
  }

  function visibleComparePanelCount(sources) {
    return compareShowDiffEnabled() && canAttemptDiff(sources) ? 3 : 2;
  }

  function decideCompareWindowMode(windowInfo, plotDiv, sources) {
    const panelCount = visibleComparePanelCount(sources);
    const fullWidth = plotDiv.clientWidth || plotDiv.offsetWidth || 1;
    const widthPx = Math.max(1, fullWidth / panelCount);
    const heightPx = plotDiv.clientHeight || plotDiv.offsetHeight || 1;
    const probabilityInvolved = sources.a.domain === 'probability' || sources.b.domain === 'probability';
    const wantWiggle = !probabilityInvolved && wantWiggleForWindow({
      tracesVisible: windowInfo.nTraces,
      samplesVisible: windowInfo.nSamples,
      widthPx,
    });
    if (wantWiggle) {
      return {
        mode: 'wiggle',
        stepX: 1,
        stepY: 1,
        panelCount,
        panelWidth: widthPx,
        probabilityInvolved,
      };
    }
    const steps = computeStepsForWindow({
      tracesVisible: windowInfo.nTraces,
      samplesVisible: windowInfo.nSamples,
      widthPx,
      heightPx,
    });
    return {
      mode: 'heatmap',
      stepX: steps.step_x,
      stepY: steps.step_y,
      panelCount,
      panelWidth: widthPx,
      probabilityInvolved,
    };
  }

  function buildCompareRequest(source, referenceSource, key1Val, windowInfo, decision) {
    const effectiveLayer = source.id === 'raw' ? 'raw' : source.id;
    const tapLabel = source.id === 'raw' ? null : source.tapLabel;
    const referencePipelineKey = referenceSource?.id === 'raw'
      ? null
      : (referenceSource?.pipelineKey || null);
    const referenceTapLabel = referenceSource?.id === 'raw'
      ? null
      : (referenceSource?.tapLabel || null);
    const requestContext = {
      fileId: currentFileId,
      key1Val,
      key1Byte: currentKey1Byte,
      key2Byte: currentKey2Byte,
      windowInfo,
      stepX: decision.stepX,
      stepY: decision.stepY,
      requestedLayer: source.id,
      effectiveLayer,
      pipelineKey: source.pipelineKey,
      tapLabel,
      referencePipelineKey,
      referenceTapLabel,
      scaling: currentScaling,
      transpose: '1',
      mode: decision.mode,
    };
    const artifacts = buildWindowRequestArtifacts(requestContext);
    return { source, requestContext, ...artifacts };
  }

  async function fetchComparePayload(request, ctrl, requestId) {
    const cached = windowCacheGet(request.cacheKey);
    if (cached) return cached;

    const res = await fetch(`/get_section_window_bin?${request.params.toString()}`, { signal: ctrl.signal });
    if (!res.ok) {
      let detail = '';
      try {
        const contentType = res.headers.get('content-type') || '';
        if (contentType.includes('application/json')) {
          const json = await res.json();
          detail = typeof json?.detail === 'string' ? json.detail : '';
        } else {
          detail = await res.text();
        }
      } catch (_) {
        detail = '';
      }
      throw new CompareFetchError(request.source, res.status, detail);
    }
    const buf = await res.arrayBuffer();
    if (requestId !== activeWindowFetchId) return null;
    const payload = decodeWindowPayload(
      new Uint8Array(buf),
      request.payloadMeta,
      null,
      (shape) => console.warn('Unexpected compare window shape', shape),
    );
    if (!payload) return null;
    windowCacheSet(request.cacheKey, payload);
    return payload;
  }

  function payloadDt(payload) {
    const dt = Number(payload?.dt);
    if (Number.isFinite(dt) && dt > 0) return dt;
    const fallback = Number(window.defaultDt ?? defaultDt);
    return Number.isFinite(fallback) && fallback > 0 ? fallback : null;
  }

  function payloadToF32(payload) {
    if (!payload || !Array.isArray(payload.shape) || payload.shape.length !== 2) return null;
    const rows = Number(payload.shape[0]);
    const cols = Number(payload.shape[1]);
    if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows <= 0 || cols <= 0) return null;
    const total = rows * cols;
    let out = null;
    if (payload.zBacking instanceof Float32Array && payload.zBacking.length >= total) {
      out = new Float32Array(payload.zBacking.subarray(0, total));
    } else if (Array.isArray(payload.zRows) && payload.zRows.length === rows) {
      out = new Float32Array(total);
      for (let r = 0; r < rows; r++) {
        const row = payload.zRows[r];
        if (!row || row.length < cols) return null;
        out.set(row.subarray ? row.subarray(0, cols) : Array.from(row).slice(0, cols), r * cols);
      }
    } else if (payload.values instanceof Float32Array && payload.values.length >= total) {
      out = new Float32Array(payload.values.subarray(0, total));
    } else if (payload.valuesI8 instanceof Int8Array && payload.valuesI8.length >= total) {
      const scale = Number(payload.scale) || Number(payload.quant?.scale) || 1;
      const invScale = scale === 0 ? 1 : 1 / scale;
      out = new Float32Array(total);
      for (let i = 0; i < total; i++) out[i] = payload.valuesI8[i] * invScale;
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

  function axisSuffix(index) {
    return index === 0 ? '' : String(index + 1);
  }

  function axisRef(base, index) {
    return index === 0 ? base : `${base}${index + 1}`;
  }

  function axisLayoutName(base, index) {
    return `${base}axis${axisSuffix(index)}`;
  }

  function compareDomains(count) {
    const gapTotal = AXIS_MARGIN * Math.max(0, count - 1);
    const panelWidth = (1 - gapTotal) / count;
    const domains = [];
    let start = 0;
    for (let i = 0; i < count; i++) {
      const end = i === count - 1 ? 1 : start + panelWidth;
      domains.push([start, end]);
      start = end + AXIS_MARGIN;
    }
    return domains;
  }

  function panelTitle(panel) {
    if (panel.kind === 'diff') return `A-B: ${panel.left} - ${panel.right}`;
    return `${panel.role}: ${panel.label}`;
  }

  function buildCompareLayout(render, panels, xRange, yRange) {
    const domains = compareDomains(panels.length);
    const dt = Number(payloadDt(render.a.payload)) || Number(window.defaultDt ?? defaultDt) || 0;
    const yDefault = [(render.windowInfo.y1 * dt), (render.windowInfo.y0 * dt)];
    const layout = {
      clickmode: clickModeForCurrentState(),
      dragmode: effectiveDragMode(),
      uirevision: `${currentUiRevision()}:compare:${sourcePairKey(render.sources)}`,
      paper_bgcolor: '#fff',
      plot_bgcolor: '#fff',
      margin: { t: 38, r: 12, l: 58, b: 42 },
      annotations: [],
      showlegend: false,
    };
    for (let i = 0; i < panels.length; i++) {
      const xName = axisLayoutName('x', i);
      const yName = axisLayoutName('y', i);
      layout[xName] = {
        domain: domains[i],
        title: 'Trace',
        showgrid: false,
        tickfont: { color: '#000' },
        titlefont: { color: '#000' },
        autorange: false,
        range: xRange || [render.windowInfo.x0, render.windowInfo.x1],
      };
      layout[yName] = {
        domain: [0, 1],
        title: i === 0 ? 'Time (s)' : '',
        showgrid: false,
        tickfont: { color: '#000' },
        titlefont: { color: '#000' },
        autorange: false,
        range: yRange || yDefault,
      };
      layout.annotations.push({
        xref: 'paper',
        yref: 'paper',
        x: (domains[i][0] + domains[i][1]) / 2,
        y: 1.06,
        xanchor: 'center',
        yanchor: 'bottom',
        showarrow: false,
        text: panelTitle(panels[i]),
        font: { size: 13, color: '#111827' },
      });
    }
    return layout;
  }

  function buildCompareWiggleTraces(panel, axisIndex, render) {
    const { rows, cols, x0, stepX, y0, stepY } = render;
    const values = panel.values;
    const dt = Number(payloadDt(render.a.payload)) || Number(window.defaultDt ?? defaultDt) || 0;
    const gain = parseFloat(document.getElementById('gain')?.value) || 1.0;
    const lineSegLen = rows + 1;
    const fillSegLen = (2 * rows) + 2;
    const baseX = new Float32Array(cols * lineSegLen);
    const baseY = new Float32Array(cols * lineSegLen);
    const lineX = new Float32Array(cols * lineSegLen);
    const lineY = new Float32Array(cols * lineSegLen);
    const fillX = new Float32Array(cols * fillSegLen);
    const fillY = new Float32Array(cols * fillSegLen);
    for (let c = 0; c < cols; c++) {
      const traceIndex = x0 + c * stepX;
      const lineStart = c * lineSegLen;
      const fillStart = c * fillSegLen;
      for (let r = 0; r < rows; r++) {
        const t = (y0 + r * stepY) * dt;
        const idx = r * cols + c;
        let val = values[idx] * gain;
        if (val > AMP_LIMIT) val = AMP_LIMIT;
        if (val < -AMP_LIMIT) val = -AMP_LIMIT;
        const posVal = val < 0 ? 0 : val;
        const lineIdx = lineStart + r;
        const fillBaseIdx = fillStart + r;
        const fillPosIdx = fillStart + rows + (rows - 1 - r);
        baseX[lineIdx] = traceIndex;
        baseY[lineIdx] = t;
        lineX[lineIdx] = traceIndex + val;
        lineY[lineIdx] = t;
        fillX[fillBaseIdx] = traceIndex;
        fillY[fillBaseIdx] = t;
        fillX[fillPosIdx] = traceIndex + posVal;
        fillY[fillPosIdx] = t;
      }
      const lineNanIdx = lineStart + rows;
      baseX[lineNanIdx] = NaN;
      baseY[lineNanIdx] = NaN;
      lineX[lineNanIdx] = NaN;
      lineY[lineNanIdx] = NaN;
      const fillCloseIdx = fillStart + (2 * rows);
      const fillNanIdx = fillCloseIdx + 1;
      fillX[fillCloseIdx] = traceIndex;
      fillY[fillCloseIdx] = (y0 * dt);
      fillX[fillNanIdx] = NaN;
      fillY[fillNanIdx] = NaN;
    }
    const xaxis = axisRef('x', axisIndex);
    const yaxis = axisRef('y', axisIndex);
    return [
      {
        type: 'scatter',
        mode: 'lines',
        x: baseX,
        y: baseY,
        xaxis,
        yaxis,
        line: { width: 0 },
        connectgaps: false,
        hoverinfo: 'skip',
        showlegend: false,
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: fillX,
        y: fillY,
        xaxis,
        yaxis,
        fill: 'toself',
        fillcolor: 'black',
        line: { width: 0 },
        opacity: 0.6,
        connectgaps: false,
        hoverinfo: 'skip',
        showlegend: false,
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: lineX,
        y: lineY,
        xaxis,
        yaxis,
        line: { color: 'black', width: 0.5 },
        connectgaps: false,
        hoverinfo: 'x+y',
        showlegend: false,
      },
    ];
  }

  function buildCompareHeatmapTrace(panel, axisIndex, render) {
    const { rows, cols, x0, stepX, y0, stepY } = render;
    const xVals = new Float32Array(cols);
    for (let c = 0; c < cols; c++) xVals[c] = x0 + c * stepX;
    const dt = Number(payloadDt(render.a.payload)) || Number(window.defaultDt ?? defaultDt) || 0;
    const yVals = new Float32Array(rows);
    for (let r = 0; r < rows; r++) yVals[r] = (y0 + r * stepY) * dt;
    const gain = parseFloat(document.getElementById('gain')?.value) || 1.0;
    const cmName = document.getElementById('colormap')?.value || 'Greys';
    const reverse = !!document.getElementById('cmReverse')?.checked;
    const cm = (window.COLORMAPS && window.COLORMAPS[cmName]) || 'Greys';
    const scale = compareHeatmapScale(panel, gain);
    const isDiv = scale.signed && (cmName === 'RdBu' || cmName === 'BWR');
    return {
      type: 'heatmap',
      x: xVals,
      y: yVals,
      z: rowsFromF32(panel.values, rows, cols),
      xaxis: axisRef('x', axisIndex),
      yaxis: axisRef('y', axisIndex),
      colorscale: cm,
      reversescale: reverse,
      zmin: scale.zmin,
      zmax: scale.zmax,
      zmid: isDiv ? 0 : null,
      showscale: false,
      hoverinfo: 'x+y',
      hovertemplate: '',
    };
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

  function buildComparePanels(render) {
    const panels = [
      {
        kind: 'source',
        role: 'A',
        domain: render.sources.a.domain,
        label: render.sources.a.label,
        values: render.a.values,
      },
      {
        kind: 'source',
        role: 'B',
        domain: render.sources.b.domain,
        label: render.sources.b.label,
        values: render.b.values,
      },
    ];
    if (compareShowDiffEnabled() && render.diffAvailable && render.diffValues) {
      panels.push({
        kind: 'diff',
        role: 'A-B',
        domain: render.sources.a.domain,
        label: `${render.sources.a.label} - ${render.sources.b.label}`,
        left: render.sources.a.label,
        right: render.sources.b.label,
        values: render.diffValues,
      });
    }
    return panels;
  }

  function renderCompareLatestView() {
    if (!isCompareModeEnabled()) return false;
    const render = latestCompareRender;
    if (!render) return false;
    const key1Val = currentCompareKey1();
    const sources = getCompareSources();
    if (render.key1 !== key1Val || sourcePairKey(render.sources) !== sourcePairKey(sources)) return false;
    if (render.scaling !== currentScaling) return false;

    const plotDiv = document.getElementById('plot');
    if (!plotDiv) return false;
    const panels = buildComparePanels(render);
    if (compareShowDiffEnabled() && !render.diffAvailable) {
      setCompareStatus(render.diffMessage || 'A-B unavailable.');
    } else {
      setCompareStatus('');
    }

    const xRange = savedXRange || null;
    const yRange = savedYRange || null;
    const traces = [];
    for (let i = 0; i < panels.length; i++) {
      if (render.mode === 'wiggle') traces.push(...buildCompareWiggleTraces(panels[i], i, render));
      else traces.push(buildCompareHeatmapTrace(panels[i], i, render));
    }
    const layout = buildCompareLayout(render, panels, xRange, yRange);
    downsampleFactor = render.stepY || 1;
    renderedStart = render.x0;
    renderedEnd = render.x1;
    latestWindowRender = null;
    setGrid({ x0: render.x0, stepX: render.mode === 'wiggle' ? 1 : render.stepX, y0: render.y0, stepY: render.stepY });
    const promise = withSuppressedRelayout(Plotly.react(plotDiv, traces, layout, {
      responsive: true,
      doubleClick: false,
      doubleClickDelay: 300,
    }));
    plotDiv.__svPlotMode = `compare-${render.mode}`;
    plotDiv.__svComparePanelCount = panels.length;
    plotDiv.__svCompareMode = render.mode;
    if (promise && typeof promise.finally === 'function') {
      promise.finally(() => {
        if (typeof maybeResizePlot === 'function') maybeResizePlot(plotDiv, true);
      });
    } else if (typeof maybeResizePlot === 'function') {
      maybeResizePlot(plotDiv, true);
    }
    requestAnimationFrame(applyDragMode);
    if (typeof installPlotlyViewportHandlersOnce === 'function') installPlotlyViewportHandlersOnce();
    return true;
  }

  function buildCompareRender(aPayload, bPayload, sources, decision, validation, windowInfo) {
    const aValues = payloadToF32(aPayload);
    const bValues = payloadToF32(bPayload);
    if (!aValues || !bValues) {
      return null;
    }
    const rows = Number(aPayload.shape[0]);
    const cols = Number(aPayload.shape[1]);
    const diffValues = validation.ok ? subtractF32(aValues, bValues) : null;
    return {
      key1: aPayload.key1,
      sources,
      sourcePair: sourcePairKey(sources),
      scaling: currentScaling,
      mode: decision.mode,
      panelCount: decision.panelCount,
      stepX: decision.stepX,
      stepY: decision.stepY,
      x0: aPayload.x0,
      x1: aPayload.x1,
      y0: aPayload.y0,
      y1: aPayload.y1,
      rows,
      cols,
      windowInfo,
      a: { payload: aPayload, values: aValues },
      b: { payload: bPayload, values: bValues },
      diffAvailable: validation.ok && !!diffValues,
      diffMessage: validation.message,
      diffValues,
    };
  }

  function compareUnavailableMessage(sources) {
    if (!sources.a.available) return 'A-B unavailable: A tap is not available. Run pipeline first.';
    if (!sources.b.available) return 'A-B unavailable: B tap is not available. Run pipeline first.';
    if (compareShowDiffEnabled() && sources.a.domain !== sources.b.domain) {
      return 'A-B unavailable: source domains are different.';
    }
    return '';
  }

  async function fetchCompareAndPlot() {
    if (!isCompareModeEnabled()) return false;
    if (!currentFileId) return true;
    if (!sectionShape) {
      await fetchSectionMeta();
      if (!sectionShape) return true;
    }
    const key1Val = currentCompareKey1();
    if (key1Val === undefined) return true;
    const windowInfo = currentVisibleWindow();
    if (!windowInfo) return true;
    const plotDiv = document.getElementById('plot');
    if (!plotDiv) return true;

    const sources = getCompareSources();
    if (!sources.a.available || !sources.b.available) {
      setCompareStatus(compareUnavailableMessage(sources));
      return true;
    }
    const decision = decideCompareWindowMode(windowInfo, plotDiv, sources);
    const requestA = buildCompareRequest(sources.a, sources.a, key1Val, windowInfo, decision);
    const requestB = buildCompareRequest(sources.b, sources.a, key1Val, windowInfo, decision);

    const requestId = bumpWindowFetchId();
    if (windowFetchCtrl) windowFetchCtrl.abort();
    if (typeof cancelActiveMainDecodeJob === 'function') cancelActiveMainDecodeJob();
    const ctrl = new AbortController();
    windowFetchCtrl = ctrl;
    showLoading(buildWindowLoadingMessage({
      mode: `compare ${decision.mode}`,
      stepX: decision.stepX,
      stepY: decision.stepY,
    }));

    try {
      const aPromise = fetchComparePayload(requestA, ctrl, requestId);
      const bPromise = requestA.cacheKey === requestB.cacheKey
        ? aPromise
        : fetchComparePayload(requestB, ctrl, requestId);
      const [aPayload, bPayload] = await Promise.all([aPromise, bPromise]);
      if (requestId !== activeWindowFetchId || !aPayload || !bPayload) return true;

      const validation = validateComparePair(aPayload, bPayload, sources);
      const render = buildCompareRender(aPayload, bPayload, sources, decision, validation, windowInfo);
      if (!render) {
        setCompareStatus('A-B unavailable: source data could not be decoded.');
        return true;
      }
      latestCompareRender = render;
      latestSeismicData = null;
      renderCompareLatestView();
      return true;
    } catch (err) {
      if (err && err.name === 'AbortError') return true;
      if (err instanceof CompareFetchError) {
        const role = err.source?.role === 'A' ? 'A' : 'B';
        if (err.status === 409) {
          setCompareStatus(`A-B unavailable: ${role} tap is not available. Run pipeline first.`);
        } else {
          setCompareStatus(err.message);
        }
        return true;
      }
      console.warn('Compare window fetch error', err);
      setCompareStatus(err instanceof Error ? err.message : String(err));
      return true;
    } finally {
      if (windowFetchCtrl === ctrl) windowFetchCtrl = null;
      if (requestId === activeWindowFetchId) hideLoading();
    }
  }

  function compareCurrentDesiredMode() {
    if (!isCompareModeEnabled()) return null;
    const win = currentVisibleWindow();
    const plotDiv = document.getElementById('plot');
    if (!win || !plotDiv) return null;
    const sources = getCompareSources();
    return decideCompareWindowMode(win, plotDiv, sources).mode;
  }

  function compareNeedsFresh(decision, win, sources) {
    if (!latestCompareRender) return true;
    if (latestCompareRender.key1 !== currentCompareKey1()) return true;
    if (sourcePairKey(latestCompareRender.sources) !== sourcePairKey(sources)) return true;
    if (latestCompareRender.scaling !== currentScaling) return true;
    if (latestCompareRender.mode !== decision.mode) return true;
    if (latestCompareRender.stepX !== decision.stepX || latestCompareRender.stepY !== decision.stepY) return true;
    if (latestCompareRender.x0 > win.x0 || latestCompareRender.x1 < win.x1) return true;
    if (latestCompareRender.y0 > win.y0 || latestCompareRender.y1 < win.y1) return true;
    return false;
  }

  function requestCompareWindowFetch(immediate) {
    if (typeof requestWindowFetch === 'function') {
      requestWindowFetch({ immediate: immediate === true });
      return;
    }
    if (typeof scheduleWindowFetch === 'function') scheduleWindowFetch();
  }

  function checkCompareModeFlipAndRefetch({ immediate = false } = {}) {
    if (!isCompareModeEnabled()) return false;
    const win = currentVisibleWindow();
    const plotDiv = document.getElementById('plot');
    if (!win || !plotDiv) return false;
    const sources = getCompareSources();
    const unavailable = compareUnavailableMessage(sources);
    if (unavailable && (!sources.a.available || !sources.b.available || compareShowDiffEnabled())) {
      setCompareStatus(unavailable);
    }
    const decision = decideCompareWindowMode(win, plotDiv, sources);
    if (compareNeedsFresh(decision, win, sources)) {
      requestCompareWindowFetch(immediate);
      return true;
    }
    renderCompareLatestView();
    return false;
  }

  function readCompareAxisRange(ev, base, index) {
    if (typeof readAxisRange !== 'function') return null;
    return readAxisRange(ev, `${base}axis${axisSuffix(index)}`);
  }

  function firstEventAxisRange(ev, base) {
    for (let i = 0; i < 3; i++) {
      const range = readCompareAxisRange(ev, base, i);
      if (range) return range;
    }
    return null;
  }

  function firstFullLayoutAxisRange(plotDiv, base) {
    const layout = plotDiv?._fullLayout;
    if (!layout) return null;
    const count = Math.max(2, Number(plotDiv.__svComparePanelCount) || 2);
    for (let i = 0; i < count; i++) {
      const axis = layout[axisLayoutName(base, i)];
      const range = axis?.range;
      if (Array.isArray(range) && range.length === 2 && Number.isFinite(range[0]) && Number.isFinite(range[1])) {
        return [range[0], range[1]];
      }
    }
    return null;
  }

  function syncCompareAxes(plotDiv, xRange, yRange) {
    if (!plotDiv || compareSyncing || !window.Plotly || typeof window.Plotly.relayout !== 'function') return null;
    const count = Math.max(2, Number(plotDiv.__svComparePanelCount) || 2);
    const props = {};
    for (let i = 0; i < count; i++) {
      const xs = axisSuffix(i);
      if (xRange) props[`xaxis${xs}.range`] = [xRange[0], xRange[1]];
      if (yRange) props[`yaxis${xs}.range`] = [yRange[0], yRange[1]];
      props[`xaxis${xs}.autorange`] = false;
      props[`yaxis${xs}.autorange`] = false;
    }
    compareSyncing = true;
    const promise = withSuppressedRelayout(window.Plotly.relayout(plotDiv, props));
    if (promise && typeof promise.finally === 'function') {
      return promise.finally(() => { compareSyncing = false; });
    }
    compareSyncing = false;
    return promise;
  }

  async function handleCompareRelayout(ev) {
    if (!isCompareModeEnabled() || compareSyncing) return;
    const plotDiv = document.getElementById('plot');
    if (!plotDiv) return;
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const xRange = firstEventAxisRange(ev, 'x') || firstFullLayoutAxisRange(plotDiv, 'x');
    const yRangeRaw = firstEventAxisRange(ev, 'y') || firstFullLayoutAxisRange(plotDiv, 'y');
    if (xRange) savedXRange = [xRange[0], xRange[1]];
    let yRange = null;
    if (yRangeRaw) {
      yRange = yRangeRaw[0] > yRangeRaw[1]
        ? [yRangeRaw[0], yRangeRaw[1]]
        : [yRangeRaw[1], yRangeRaw[0]];
      savedYRange = yRange;
    }
    await syncCompareAxes(plotDiv, savedXRange, savedYRange);
    checkCompareModeFlipAndRefetch({ immediate: typeof isResetRelayout === 'function' && isResetRelayout(ev) });
  }

  function snapshotCompareAxesRangesFromDOM() {
    if (!isCompareModeEnabled()) return false;
    const plotDiv = document.getElementById('plot');
    const xRange = firstFullLayoutAxisRange(plotDiv, 'x');
    const yRangeRaw = firstFullLayoutAxisRange(plotDiv, 'y');
    if (xRange) savedXRange = [xRange[0], xRange[1]];
    if (yRangeRaw) {
      savedYRange = yRangeRaw[0] > yRangeRaw[1]
        ? [yRangeRaw[0], yRangeRaw[1]]
        : [yRangeRaw[1], yRangeRaw[0]];
    }
    return true;
  }

  function clearCompareRender() {
    latestCompareRender = null;
    setCompareStatus('');
  }

  function onCompareControlChange() {
    updateCompareSourceOptions();
    snapshotCompareAxesRangesFromDOM();
    if (!isCompareModeEnabled()) {
      clearCompareRender();
      if (typeof renderLatestView === 'function') renderLatestView();
      if (typeof scheduleWindowFetch === 'function') scheduleWindowFetch();
      return;
    }
    const requested = checkCompareModeFlipAndRefetch({ immediate: true });
    if (!requested && !latestCompareRender) requestCompareWindowFetch(true);
  }

  function initCompareControls() {
    updateCompareSourceOptions();
    const { toggle, sourceA, sourceB, showDiff } = getCompareNodes();
    for (const node of [toggle, sourceA, sourceB, showDiff]) {
      if (!node) continue;
      node.addEventListener('change', onCompareControlChange);
    }
  }

  window.isCompareModeEnabled = isCompareModeEnabled;
  window.compareShowDiffEnabled = compareShowDiffEnabled;
  window.updateCompareSourceOptions = updateCompareSourceOptions;
  window.fetchCompareAndPlot = fetchCompareAndPlot;
  window.renderCompareLatestView = renderCompareLatestView;
  window.compareCurrentDesiredMode = compareCurrentDesiredMode;
  window.checkCompareModeFlipAndRefetch = checkCompareModeFlipAndRefetch;
  window.handleCompareRelayout = handleCompareRelayout;
  window.snapshotCompareAxesRangesFromDOM = snapshotCompareAxesRangesFromDOM;
  window.clearCompareRender = clearCompareRender;
  window.__svCompare = {
    validateComparePair,
    subtractF32,
    payloadToF32,
    resolveSourceDomain,
    compareHeatmapScale,
    buildComparePanels,
  };

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', initCompareControls, { once: true });
  } else {
    initCompareControls();
  }
})();
