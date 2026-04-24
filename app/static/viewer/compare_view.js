(function () {
  const AMP_LIMIT = 3.0;
  const DIFF_NOTE = 'Diff is computed from decoded displayed window values.';
  const EPSILON_DIFF_LIMIT = 1e-6;
  const DOMAIN_MISMATCH_MESSAGE = 'Cannot compare sources: source domains differ.';

  const state = {
    mode: 'single',
    sourceA: { type: 'raw' },
    sourceB: { type: 'raw' },
    diffMode: 'b_minus_a',
    showRmsByTrace: true,
    sourceBAuto: true,
    lastResult: null,
    fetchToken: 0,
    fetchCtrl: null,
    decodeJobs: [],
    suppressRelayout: false,
  };

  function getUi() {
    return {
      mainPlot: document.getElementById('plot'),
      compareModeSelect: document.getElementById('compareModeSelect'),
      compareSourceASelect: document.getElementById('compareSourceASelect'),
      compareSourceBSelect: document.getElementById('compareSourceBSelect'),
      compareDiffModeSelect: document.getElementById('compareDiffModeSelect'),
      compareShowRms: document.getElementById('compareShowRms'),
      compareStatus: document.getElementById('compareStatus'),
      compareStatusViewport: document.getElementById('compareStatusViewport'),
      compareSurface: document.getElementById('compareSurface'),
      compareSideBySide: document.getElementById('compareSideBySide'),
      compareDiffView: document.getElementById('compareDiffView'),
      compareSourceALabel: document.getElementById('compareSourceALabel'),
      compareSourceBLabel: document.getElementById('compareSourceBLabel'),
      compareDiffExpression: document.getElementById('compareDiffExpression'),
      compareNotice: document.getElementById('compareNotice'),
      compareStatMean: document.getElementById('compareStatMean'),
      compareStatStd: document.getElementById('compareStatStd'),
      compareStatRms: document.getElementById('compareStatRms'),
      compareStatMaxAbs: document.getElementById('compareStatMaxAbs'),
      compareRmsSection: document.getElementById('compareRmsSection'),
      comparePlotA: document.getElementById('comparePlotA'),
      comparePlotB: document.getElementById('comparePlotB'),
      comparePlotDiff: document.getElementById('comparePlotDiff'),
      compareRmsPlot: document.getElementById('compareRmsPlot'),
      layerSelect: document.getElementById('layerSelect'),
    };
  }

  function cloneSource(source) {
    if (!source || source.type !== 'pipeline_tap') {
      return { type: 'raw' };
    }
    return {
      type: 'pipeline_tap',
      pipelineKey: source.pipelineKey || null,
      tapLabel: String(source.tapLabel || ''),
    };
  }

  function sameSource(a, b) {
    const sourceA = cloneSource(a);
    const sourceB = cloneSource(b);
    return (
      sourceA.type === sourceB.type &&
      (sourceA.pipelineKey || null) === (sourceB.pipelineKey || null) &&
      String(sourceA.tapLabel || '') === String(sourceB.tapLabel || '')
    );
  }

  function sourceLabel(source) {
    const next = cloneSource(source);
    if (next.type === 'pipeline_tap') return next.tapLabel || 'pipeline tap';
    return 'raw';
  }

  function sourceOptionValue(source) {
    const next = cloneSource(source);
    if (next.type !== 'pipeline_tap') return 'raw';
    return `pipeline_tap:${encodeURIComponent(next.pipelineKey || '')}:${encodeURIComponent(next.tapLabel || '')}`;
  }

  function parseSourceOptionValue(value) {
    if (!value || value === 'raw') return { type: 'raw' };
    if (!String(value).startsWith('pipeline_tap:')) return { type: 'raw' };
    const raw = String(value).slice('pipeline_tap:'.length);
    const splitIndex = raw.indexOf(':');
    if (splitIndex < 0) return { type: 'raw' };
    return {
      type: 'pipeline_tap',
      pipelineKey: decodeURIComponent(raw.slice(0, splitIndex)) || null,
      tapLabel: decodeURIComponent(raw.slice(splitIndex + 1)) || '',
    };
  }

  function inferSourceDomain(source) {
    const next = cloneSource(source);
    if (next.type !== 'pipeline_tap') return 'amplitude';
    const label = String(next.tapLabel || '').toLowerCase();
    if (label.includes('fbpick') || label.includes('fbprob') || label.includes('prob')) {
      return 'probability';
    }
    return 'amplitude';
  }

  function listAvailableSources() {
    const ui = getUi();
    const sources = [{ type: 'raw' }];
    const seen = new Set(['raw']);
    const pipelineKey = window.latestPipelineKey || null;
    const options = Array.from(ui.layerSelect?.options || []);
    for (const option of options) {
      const value = String(option?.value || '');
      if (!value || value === 'raw') continue;
      const source = {
        type: 'pipeline_tap',
        pipelineKey,
        tapLabel: value,
      };
      const key = sourceOptionValue(source);
      if (seen.has(key)) continue;
      seen.add(key);
      sources.push(source);
    }
    return sources;
  }

  function firstProcessedSource(sources) {
    return (Array.isArray(sources) ? sources : []).find((source) => cloneSource(source).type === 'pipeline_tap') || null;
  }

  function normalizeSource(source, sources, fallbackSource) {
    const next = cloneSource(source);
    const available = Array.isArray(sources) ? sources : [];
    const exact = available.find((candidate) => sameSource(candidate, next));
    if (exact) return cloneSource(exact);
    if (next.type === 'pipeline_tap') {
      const labelMatch = available.find((candidate) => (
        cloneSource(candidate).type === 'pipeline_tap' &&
        String(candidate.tapLabel || '') === String(next.tapLabel || '')
      ));
      if (labelMatch) return cloneSource(labelMatch);
    }
    return cloneSource(fallbackSource || { type: 'raw' });
  }

  function formatMetric(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return '-';
    const absValue = Math.abs(num);
    if ((absValue >= 1000) || (absValue > 0 && absValue < 0.001)) {
      return num.toExponential(3);
    }
    return num.toFixed(4);
  }

  function rowsFromBacking(backing, rows, cols) {
    const zRows = new Array(rows);
    for (let r = 0; r < rows; r += 1) {
      zRows[r] = backing.subarray(r * cols, (r + 1) * cols);
    }
    return zRows;
  }

  function getPlotSizeForCompareMode() {
    const ui = getUi();
    const fallbackHost = ui.compareSurface || ui.mainPlot;
    const host = state.mode === 'side_by_side'
      ? (ui.comparePlotA || fallbackHost)
      : (ui.comparePlotDiff || fallbackHost);
    const widthPx = host?.clientWidth || host?.offsetWidth || fallbackHost?.clientWidth || 1;
    const heightPx = host?.clientHeight || host?.offsetHeight || fallbackHost?.clientHeight || 1;
    return { widthPx, heightPx };
  }

  function getCompareDragMode() {
    const dragMode = (typeof window.effectiveDragMode === 'function')
      ? window.effectiveDragMode()
      : null;
    return dragMode === 'pan' ? 'pan' : 'zoom';
  }

  function getAxisRangeFromDiv(plotDiv, axisName) {
    const axis = plotDiv?._fullLayout?.[axisName];
    if (!axis || !Array.isArray(axis.range) || axis.range.length !== 2) return null;
    const a = Number(axis.range[0]);
    const b = Number(axis.range[1]);
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    return [a, b];
  }

  function readRelayoutRange(plotDiv, event, axisName) {
    if (typeof window.readAxisRange === 'function') {
      const range = window.readAxisRange(event, axisName);
      if (range) return range;
    }
    return getAxisRangeFromDiv(plotDiv, axisName);
  }

  function setCompareStatus(message, isError) {
    const ui = getUi();
    const text = typeof message === 'string' ? message.trim() : '';
    const nodes = [ui.compareStatus, ui.compareStatusViewport].filter(Boolean);
    for (const node of nodes) {
      node.textContent = text;
      node.hidden = text.length === 0;
      node.classList.toggle('is-error', !!text && isError === true);
    }
  }

  function clearCompareStatus() {
    setCompareStatus('', false);
  }

  function clearSummaryMetrics() {
    const ui = getUi();
    if (ui.compareStatMean) ui.compareStatMean.textContent = '-';
    if (ui.compareStatStd) ui.compareStatStd.textContent = '-';
    if (ui.compareStatRms) ui.compareStatRms.textContent = '-';
    if (ui.compareStatMaxAbs) ui.compareStatMaxAbs.textContent = '-';
  }

  function cancelCompareDecodeJobs(decodeJobs) {
    const jobs = Array.isArray(decodeJobs) ? decodeJobs : [];
    while (jobs.length > 0) {
      const jobId = jobs.pop();
      if (typeof window.cancelDecodeJob === 'function' && Number.isInteger(jobId)) {
        try {
          window.cancelDecodeJob('main', jobId, { resolveDropped: true });
        } catch (_) {
          // Best effort only.
        }
      }
    }
  }

  function cancelTrackedCompareFetch(ctrl, decodeJobs) {
    if (ctrl) ctrl.abort();
    cancelCompareDecodeJobs(decodeJobs);
  }

  function isCurrentCompareFetch(fetchToken, ctrl) {
    return fetchToken === state.fetchToken && state.fetchCtrl === ctrl;
  }

  function cancelActiveCompareFetch() {
    cancelTrackedCompareFetch(state.fetchCtrl, state.decodeJobs);
    state.fetchCtrl = null;
    state.decodeJobs = [];
    if (typeof window.hideLoading === 'function') window.hideLoading();
  }

  function updateSurfaceVisibility() {
    const ui = getUi();
    const active = state.mode !== 'single';
    if (ui.mainPlot) ui.mainPlot.hidden = active;
    if (ui.compareSurface) ui.compareSurface.hidden = !active;
    if (ui.compareSideBySide) ui.compareSideBySide.hidden = state.mode !== 'side_by_side';
    if (ui.compareDiffView) ui.compareDiffView.hidden = state.mode !== 'difference';
    if (ui.compareRmsSection) ui.compareRmsSection.hidden = !(state.mode === 'difference' && state.showRmsByTrace);
    if (ui.compareNotice) {
      ui.compareNotice.hidden = state.mode !== 'difference';
      ui.compareNotice.textContent = state.mode === 'difference' ? DIFF_NOTE : '';
    }
  }

  function updateControlState() {
    const ui = getUi();
    const active = state.mode !== 'single';
    const diffActive = state.mode === 'difference';
    if (ui.compareModeSelect) ui.compareModeSelect.value = state.mode;
    if (ui.compareDiffModeSelect) {
      ui.compareDiffModeSelect.value = state.diffMode;
      ui.compareDiffModeSelect.disabled = !diffActive;
    }
    if (ui.compareShowRms) {
      ui.compareShowRms.checked = !!state.showRmsByTrace;
      ui.compareShowRms.disabled = !diffActive;
    }
    if (ui.compareSourceASelect) ui.compareSourceASelect.disabled = !active;
    if (ui.compareSourceBSelect) ui.compareSourceBSelect.disabled = !active;
    updateSurfaceVisibility();
  }

  function syncSourceSelect(selectEl, sources, currentSource) {
    if (!selectEl) return;
    const nextSources = Array.isArray(sources) ? sources : [];
    const currentValue = sourceOptionValue(currentSource);
    selectEl.innerHTML = '';
    for (const source of nextSources) {
      selectEl.appendChild(new Option(sourceLabel(source), sourceOptionValue(source)));
    }
    selectEl.value = currentValue;
  }

  function syncCompareControls() {
    const ui = getUi();
    const prevSourceA = cloneSource(state.sourceA);
    const prevSourceB = cloneSource(state.sourceB);
    const sources = listAvailableSources();
    const rawSource = { type: 'raw' };
    const defaultB = firstProcessedSource(sources) || rawSource;
    state.sourceA = normalizeSource(state.sourceA, sources, rawSource);
    if (state.sourceBAuto) {
      state.sourceB = cloneSource(defaultB);
    } else {
      state.sourceB = normalizeSource(state.sourceB, sources, defaultB);
    }
    syncSourceSelect(ui.compareSourceASelect, sources, state.sourceA);
    syncSourceSelect(ui.compareSourceBSelect, sources, state.sourceB);
    updateControlState();
    return {
      sourceAChanged: !sameSource(prevSourceA, state.sourceA),
      sourceBChanged: !sameSource(prevSourceB, state.sourceB),
    };
  }

  function buildRequestContextForSource(source, requestBase) {
    const next = cloneSource(source);
    const requestedLayer = sourceLabel(next);
    if (next.type !== 'pipeline_tap') {
      return {
        ...requestBase,
        requestedLayer,
        effectiveLayer: 'raw',
        pipelineKey: null,
        tapLabel: null,
      };
    }
    if (!next.pipelineKey || !next.tapLabel) {
      throw new Error('Selected compare source is unavailable.');
    }
    return {
      ...requestBase,
      requestedLayer,
      effectiveLayer: next.tapLabel,
      pipelineKey: next.pipelineKey,
      tapLabel: next.tapLabel,
    };
  }

  async function readCompareErrorDetail(response) {
    if (!response) return '';
    try {
      const contentType = response.headers?.get('content-type') || '';
      if (contentType.includes('application/json')) {
        const payload = await response.json();
        if (payload && payload.detail != null) return String(payload.detail);
      }
      const text = await response.text();
      return text || '';
    } catch (_) {
      return '';
    }
  }

  function resolvePayloadDt(payload) {
    const payloadDt = Number(payload?.dt);
    if (Number.isFinite(payloadDt) && payloadDt > 0) return payloadDt;
    const defaultDt = Number(window.defaultDt);
    if (Number.isFinite(defaultDt) && defaultDt > 0) return defaultDt;
    return 0;
  }

  function ensureCompareHeatmapPayload(payload) {
    if (!payload) return null;
    if (typeof window.materializeHeatmapPayload === 'function') {
      return window.materializeHeatmapPayload(payload) ? payload : null;
    }
    return payload.zBacking instanceof Float32Array ? payload : null;
  }

  async function decodeComparePayload(buffer, payloadMeta, token, ctrl, decodeJobs) {
    const workerEnabled = typeof window.readWindowDecodeUseWorker === 'function'
      ? window.readWindowDecodeUseWorker()
      : true;
    if (workerEnabled && typeof window.enqueueDecodeJob === 'function') {
      const decodeJob = window.enqueueDecodeJob('main', {
        bin: buffer,
        mode: 'heatmap',
        fbMode: payloadMeta.effectiveLayer === 'fbprob',
        wantZ: true,
      });
      const jobId = Number(decodeJob?.jobId);
      if (Number.isInteger(jobId)) decodeJobs.push(jobId);
      try {
        const decoded = await decodeJob.promise;
        const jobIndex = decodeJobs.indexOf(jobId);
        if (jobIndex >= 0) decodeJobs.splice(jobIndex, 1);
        if (!isCurrentCompareFetch(token, ctrl) || !decoded) return null;
        const payload = window.buildWindowPayloadFromWorkerDecoded(
          decoded,
          payloadMeta,
          null,
          (shape) => console.warn('Unexpected compare window shape', shape)
        );
        return ensureCompareHeatmapPayload(payload);
      } catch (err) {
        const jobIndex = decodeJobs.indexOf(jobId);
        if (jobIndex >= 0) decodeJobs.splice(jobIndex, 1);
        throw err;
      }
    }

    const payload = window.decodeWindowPayload(
      new Uint8Array(buffer),
      payloadMeta,
      null,
      (shape) => console.warn('Unexpected compare window shape', shape),
      { applyDt: false }
    );
    return ensureCompareHeatmapPayload(payload);
  }

  async function fetchComparePayload(source, requestBase, fetchToken, cachePromises, signal, ctrl, decodeJobs) {
    const requestContext = buildRequestContextForSource(source, requestBase);
    const artifacts = window.buildWindowRequestArtifacts(requestContext);
    const cacheKey = artifacts.cacheKey;
    if (cachePromises.has(cacheKey)) return cachePromises.get(cacheKey);

    const promise = (async () => {
      const cachedPayload = window.windowCacheGet(cacheKey);
      if (cachedPayload) return ensureCompareHeatmapPayload(cachedPayload);

      const response = await fetch(`/get_section_window_bin?${artifacts.params.toString()}`, {
        signal,
      });
      if (!response.ok) {
        const detail = await readCompareErrorDetail(response);
        const suffix = detail ? `: ${detail}` : '';
        throw new Error(`Compare fetch failed (${response.status})${suffix}`);
      }
      const buffer = await response.arrayBuffer();
      if (!isCurrentCompareFetch(fetchToken, ctrl)) return null;
      const payload = await decodeComparePayload(buffer, artifacts.payloadMeta, fetchToken, ctrl, decodeJobs);
      if (!payload || !isCurrentCompareFetch(fetchToken, ctrl)) return null;
      const cachePayload = payload.__perf ? { ...payload, __perf: null } : payload;
      window.windowCacheSet(cacheKey, cachePayload);
      return cachePayload;
    })();

    cachePromises.set(cacheKey, promise);
    return promise;
  }

  function isMixedDomainDifference() {
    if (state.mode === 'single') return false;
    const domainA = inferSourceDomain(state.sourceA);
    const domainB = inferSourceDomain(state.sourceB);
    return domainA !== domainB;
  }

  function buildHeatmapTrace(payload, zRows, colors) {
    const rows = Number(payload.shape?.[0] ?? 0);
    const cols = Number(payload.shape?.[1] ?? 0);
    const xVals = new Float32Array(cols);
    for (let c = 0; c < cols; c += 1) xVals[c] = payload.x0 + (c * (payload.stepX || 1));
    const dt = resolvePayloadDt(payload);
    const yVals = new Float32Array(rows);
    for (let r = 0; r < rows; r += 1) yVals[r] = (payload.y0 + (r * (payload.stepY || 1))) * dt;
    return {
      trace: {
        type: 'heatmap',
        x: xVals,
        y: yVals,
        z: zRows,
        colorscale: colors.colorscale,
        reversescale: colors.reversescale,
        zmin: colors.zmin,
        zmax: colors.zmax,
        zmid: colors.zmid,
        showscale: false,
        hovertemplate: '',
      },
      dt,
    };
  }

  function buildHeatmapLayout(payload, title) {
    const dt = resolvePayloadDt(payload);
    return window.buildLayout({
      mode: 'heatmap',
      x0: payload.x0,
      x1: payload.x1,
      y0: payload.y0,
      y1: payload.y1,
      stepX: payload.stepX || 1,
      stepY: payload.stepY || 1,
      totalSamples: Array.isArray(window.sectionShape) ? window.sectionShape[1] : (payload.y1 - payload.y0 + 1),
      dt,
      savedXRange: window.savedXRange,
      savedYRange: window.savedYRange,
      clickmode: 'event',
      dragmode: getCompareDragMode(),
      uirevision: `compare:${state.mode}:${sourceOptionValue(state.sourceA)}:${sourceOptionValue(state.sourceB)}:${state.diffMode}`,
      fbTitle: title,
    });
  }

  function resolveStandardColors(payload) {
    const colormap = document.getElementById('colormap')?.value || 'Greys';
    const reverse = !!document.getElementById('cmReverse')?.checked;
    const gain = parseFloat(document.getElementById('gain')?.value || '1') || 1;
    const fbMode = payload?.effectiveLayer === 'fbprob';
    return {
      colorscale: (window.COLORMAPS && window.COLORMAPS[colormap]) || 'Greys',
      reversescale: reverse,
      zmin: fbMode ? 0 : (-AMP_LIMIT / Math.max(gain, 1e-9)),
      zmax: fbMode ? 255 : (AMP_LIMIT / Math.max(gain, 1e-9)),
      zmid: (!fbMode && (colormap === 'RdBu' || colormap === 'BWR')) ? 0 : undefined,
    };
  }

  function resolveDiffColors(limit) {
    const selected = document.getElementById('colormap')?.value || 'Greys';
    const diffMap = (selected === 'RdBu' || selected === 'BWR') ? selected : 'RdBu';
    return {
      colorscale: (window.COLORMAPS && window.COLORMAPS[diffMap]) || 'RdBu',
      reversescale: !!document.getElementById('cmReverse')?.checked,
      zmin: -limit,
      zmax: limit,
      zmid: 0,
    };
  }

  function ensureCompareHandlers(plotDiv, role) {
    if (!plotDiv || plotDiv.__svCompareRole === role) return;
    plotDiv.__svCompareRole = role;
    plotDiv.on('plotly_relayout', (event) => {
      if (state.mode === 'single' || state.suppressRelayout) return;
      const xRange = readRelayoutRange(plotDiv, event, 'xaxis');
      const yRange = role === 'rms' ? null : readRelayoutRange(plotDiv, event, 'yaxis');

      if (xRange) {
        window.savedXRange = [xRange[0], xRange[1]];
      }
      if (yRange) {
        const y0 = Number(yRange[0]);
        const y1 = Number(yRange[1]);
        if (Number.isFinite(y0) && Number.isFinite(y1)) {
          window.savedYRange = y0 > y1 ? [y0, y1] : [y1, y0];
        }
      }

      const ui = getUi();
      const updates = {};
      if (xRange) updates['xaxis.range'] = xRange;
      if (yRange) updates['yaxis.range'] = yRange;
      const syncTargets = [];
      const rmsReady = !!ui.compareRmsPlot?.data?.length;

      if (state.mode === 'side_by_side') {
        if (role === 'a' && ui.comparePlotB) syncTargets.push(ui.comparePlotB);
        if (role === 'b' && ui.comparePlotA) syncTargets.push(ui.comparePlotA);
      } else if (state.mode === 'difference') {
        if (role === 'diff' && xRange && state.showRmsByTrace && rmsReady) {
          syncTargets.push({ div: ui.compareRmsPlot, updates: { 'xaxis.range': xRange } });
        }
        if (role === 'rms' && xRange && ui.comparePlotDiff) {
          syncTargets.push({ div: ui.comparePlotDiff, updates: { 'xaxis.range': xRange } });
        }
      }

      if (syncTargets.length) {
        state.suppressRelayout = true;
        const relayouts = syncTargets.map((target) => {
          const div = target.div || target;
          const nextUpdates = target.updates || updates;
          return window.Plotly.relayout(div, nextUpdates);
        });
        Promise.allSettled(relayouts).finally(() => {
          state.suppressRelayout = false;
        });
      }

      if (typeof window.scheduleWindowFetch === 'function') {
        window.scheduleWindowFetch();
      }
    });
  }

  async function renderHeatmapInto(plotDiv, payload, title, colors, zBacking) {
    if (!plotDiv || !payload || !(zBacking instanceof Float32Array)) return;
    const rows = Number(payload.shape?.[0] ?? 0);
    const cols = Number(payload.shape?.[1] ?? 0);
    const zRows = rowsFromBacking(zBacking, rows, cols);
    const { trace } = buildHeatmapTrace(payload, zRows, colors);
    const layout = buildHeatmapLayout(payload, title);
    await window.Plotly.react(plotDiv, [trace], layout, { responsive: true });
  }

  async function renderRmsPlot(plotDiv, payload, rmsByTrace) {
    if (!plotDiv || !payload) return;
    const width = Number(payload.shape?.[1] ?? 0);
    const xVals = new Float32Array(width);
    for (let i = 0; i < width; i += 1) xVals[i] = payload.x0 + (i * (payload.stepX || 1));
    const layout = {
      xaxis: {
        title: 'Trace',
        showgrid: false,
        range: window.savedXRange || undefined,
      },
      yaxis: {
        title: 'RMS(B - A)',
        showgrid: false,
      },
      margin: { t: 10, r: 10, l: 60, b: 40 },
      paper_bgcolor: '#fff',
      plot_bgcolor: '#fff',
      dragmode: getCompareDragMode(),
      uirevision: `compare:rms:${sourceOptionValue(state.sourceA)}:${sourceOptionValue(state.sourceB)}`,
    };
    const trace = {
      type: 'scattergl',
      mode: 'lines',
      x: xVals,
      y: rmsByTrace,
      line: {
        color: '#0f172a',
        width: 1.5,
      },
      hovertemplate: 'Trace %{x}<br>RMS %{y:.4g}<extra></extra>',
    };
    await window.Plotly.react(plotDiv, [trace], layout, { responsive: true });
  }

  async function renderSideBySide(result) {
    const ui = getUi();
    const labelA = `A: ${sourceLabel(state.sourceA)}`;
    const labelB = `B: ${sourceLabel(state.sourceB)}`;
    if (ui.compareSourceALabel) ui.compareSourceALabel.textContent = labelA;
    if (ui.compareSourceBLabel) ui.compareSourceBLabel.textContent = labelB;
    clearSummaryMetrics();
    if (ui.compareNotice) ui.compareNotice.textContent = '';
    state.suppressRelayout = true;
    try {
      await Promise.all([
        renderHeatmapInto(ui.comparePlotA, result.payloadA, labelA, resolveStandardColors(result.payloadA), result.payloadA.zBacking),
        renderHeatmapInto(ui.comparePlotB, result.payloadB, labelB, resolveStandardColors(result.payloadB), result.payloadB.zBacking),
      ]);
    } finally {
      state.suppressRelayout = false;
    }
    ensureCompareHandlers(ui.comparePlotA, 'a');
    ensureCompareHandlers(ui.comparePlotB, 'b');
  }

  async function renderDifference(result) {
    const ui = getUi();
    const metricsApi = window.compareMetrics;
    if (!metricsApi) throw new Error('compareMetrics is unavailable');
    const payloadA = result.payloadA;
    const payloadB = result.payloadB;
    const a = payloadA.zBacking;
    const b = payloadB.zBacking;
    const diffBA = metricsApi.computeDiff(a, b, 'b_minus_a');
    const displayDiff = state.diffMode === 'b_minus_a'
      ? diffBA
      : metricsApi.computeDiff(a, b, 'a_minus_b');
    const stats = metricsApi.computeSummaryStats(diffBA);
    const rows = Number(payloadA.shape?.[0] ?? 0);
    const cols = Number(payloadA.shape?.[1] ?? 0);
    const rmsByTrace = metricsApi.computeRmsByTrace(a, b, rows, cols);
    const rawLimit = metricsApi.percentileAbs(displayDiff, 99);
    const noDifference = !Number.isFinite(rawLimit) || rawLimit <= 0;
    const diffLimit = noDifference ? EPSILON_DIFF_LIMIT : rawLimit;
    const expression = state.diffMode === 'a_minus_b' ? 'A - B' : 'B - A';

    if (ui.compareDiffExpression) ui.compareDiffExpression.textContent = expression;
    if (ui.compareNotice) {
      ui.compareNotice.textContent = noDifference
        ? `${DIFF_NOTE} No visible difference in the current window.`
        : DIFF_NOTE;
    }
    if (ui.compareStatMean) ui.compareStatMean.textContent = formatMetric(stats.mean);
    if (ui.compareStatStd) ui.compareStatStd.textContent = formatMetric(stats.std);
    if (ui.compareStatRms) ui.compareStatRms.textContent = formatMetric(stats.rms);
    if (ui.compareStatMaxAbs) ui.compareStatMaxAbs.textContent = formatMetric(stats.maxAbs);

    state.suppressRelayout = true;
    try {
      await renderHeatmapInto(
        ui.comparePlotDiff,
        payloadA,
        expression,
        resolveDiffColors(diffLimit),
        displayDiff
      );
      if (state.showRmsByTrace) {
        await renderRmsPlot(ui.compareRmsPlot, payloadA, rmsByTrace);
      } else if (ui.compareRmsPlot && typeof window.Plotly?.purge === 'function') {
        window.Plotly.purge(ui.compareRmsPlot);
      }
    } finally {
      state.suppressRelayout = false;
    }
    ensureCompareHandlers(ui.comparePlotDiff, 'diff');
    if (state.showRmsByTrace) ensureCompareHandlers(ui.compareRmsPlot, 'rms');
  }

  async function renderFromCache() {
    if (state.mode === 'single') {
      updateSurfaceVisibility();
      return;
    }
    if (!state.lastResult) return;
    updateSurfaceVisibility();
    clearCompareStatus();
    if (state.mode === 'side_by_side') {
      if (isMixedDomainDifference()) {
        purgeComparePlots();
        updateSurfaceVisibility();
        clearSummaryMetrics();
        setCompareStatus(DOMAIN_MISMATCH_MESSAGE, true);
        return;
      }
      await renderSideBySide(state.lastResult);
      return;
    }
    if (isMixedDomainDifference()) {
      purgeComparePlots();
      updateSurfaceVisibility();
      clearSummaryMetrics();
      setCompareStatus(DOMAIN_MISMATCH_MESSAGE, true);
      return;
    }
    await renderDifference(state.lastResult);
  }

  async function fetchWindowAndRender() {
    if (state.mode === 'single') return null;
    const metricsApi = window.compareMetrics;
    if (!metricsApi) return null;
    const ui = getUi();
    updateSurfaceVisibility();
    clearCompareStatus();
    if (isMixedDomainDifference()) {
      state.lastResult = null;
      purgeComparePlots();
      clearSummaryMetrics();
      setCompareStatus(DOMAIN_MISMATCH_MESSAGE, true);
      return null;
    }

    if (!window.currentFileId) {
      state.lastResult = null;
      return null;
    }
    if (!Array.isArray(window.sectionShape) || window.sectionShape.length < 2) {
      if (typeof window.fetchSectionMeta === 'function') {
        await window.fetchSectionMeta();
      }
      if (!Array.isArray(window.sectionShape) || window.sectionShape.length < 2) return null;
    }

    const slider = document.getElementById('key1_slider');
    const key1Index = slider ? parseInt(slider.value || '0', 10) : 0;
    const key1Val = Array.isArray(window.key1Values) ? window.key1Values[key1Index] : undefined;
    if (key1Val === undefined) return null;

    const windowInfo = typeof window.currentVisibleWindow === 'function'
      ? window.currentVisibleWindow()
      : null;
    if (!windowInfo) return null;

    const { widthPx, heightPx } = getPlotSizeForCompareMode();
    const { step_x, step_y } = window.computeStepsForWindow({
      tracesVisible: windowInfo.nTraces,
      samplesVisible: windowInfo.nSamples,
      widthPx,
      heightPx,
    });

    cancelActiveCompareFetch();
    state.fetchToken += 1;
    const token = state.fetchToken;
    const ctrl = new AbortController();
    const decodeJobs = [];
    state.fetchCtrl = ctrl;
    state.decodeJobs = decodeJobs;

    if (typeof window.showLoading === 'function') {
      window.showLoading(`Loading compare view... stepX=${step_x}, stepY=${step_y}`);
    }

    const requestBase = {
      fileId: window.currentFileId,
      key1Val,
      key1Byte: window.currentKey1Byte,
      key2Byte: window.currentKey2Byte,
      windowInfo,
      stepX: step_x,
      stepY: step_y,
      scaling: window.currentScaling || 'amax',
      transpose: '1',
      mode: 'heatmap',
    };

    const cachePromises = new Map();
    try {
      const [payloadA, payloadB] = await Promise.all([
        fetchComparePayload(state.sourceA, requestBase, token, cachePromises, ctrl.signal, ctrl, decodeJobs),
        fetchComparePayload(state.sourceB, requestBase, token, cachePromises, ctrl.signal, ctrl, decodeJobs),
      ]);
      if (!isCurrentCompareFetch(token, ctrl) || !payloadA || !payloadB) return null;

      const shapeA = Array.isArray(payloadA.shape) ? payloadA.shape.join('x') : '';
      const shapeB = Array.isArray(payloadB.shape) ? payloadB.shape.join('x') : '';
      if (shapeA !== shapeB) {
        state.lastResult = null;
        purgeComparePlots();
        clearSummaryMetrics();
        setCompareStatus('Cannot compare sources: decoded window shapes differ.', true);
        return null;
      }

      const dtA = resolvePayloadDt(payloadA);
      const dtB = resolvePayloadDt(payloadB);
      if (Number.isFinite(dtA) && Number.isFinite(dtB) && Math.abs(dtA - dtB) > 1e-12) {
        state.lastResult = null;
        purgeComparePlots();
        clearSummaryMetrics();
        setCompareStatus('Cannot compare sources: decoded window sample intervals differ.', true);
        return null;
      }

      state.lastResult = { payloadA, payloadB };
      await renderFromCache();
      return state.lastResult;
    } catch (err) {
      if (err && err.name === 'AbortError') return null;
      state.lastResult = null;
      purgeComparePlots();
      clearSummaryMetrics();
      setCompareStatus(err instanceof Error ? err.message : String(err), true);
      return null;
    } finally {
      cancelTrackedCompareFetch(ctrl, decodeJobs);
      if (state.fetchCtrl === ctrl) {
        state.fetchCtrl = null;
        state.decodeJobs = [];
        if (typeof window.hideLoading === 'function') window.hideLoading();
      }
    }
  }

  function purgeComparePlots() {
    const ui = getUi();
    const plots = [ui.comparePlotA, ui.comparePlotB, ui.comparePlotDiff, ui.compareRmsPlot];
    for (const plotDiv of plots) {
      if (!plotDiv || typeof window.Plotly?.purge !== 'function') continue;
      try {
        window.Plotly.purge(plotDiv);
      } catch (_) {
        // Ignore purge errors for empty divs.
      }
      plotDiv.innerHTML = '';
    }
  }

  function reset(options = {}) {
    cancelActiveCompareFetch();
    if (options.clearCache !== false) state.lastResult = null;
    if (options.keepStatus !== true) {
      clearCompareStatus();
      clearSummaryMetrics();
    }
    purgeComparePlots();
    if (options.forceHide === true) {
      const ui = getUi();
      if (ui.compareSurface) ui.compareSurface.hidden = true;
      if (ui.mainPlot) ui.mainPlot.hidden = false;
      return;
    }
    updateSurfaceVisibility();
  }

  function restylePlots() {
    if (state.mode === 'single') return;
    void renderFromCache().catch((err) => {
      setCompareStatus(err instanceof Error ? err.message : String(err), true);
    });
  }

  function handleModeChange() {
    const ui = getUi();
    state.mode = ui.compareModeSelect?.value || 'single';
    updateControlState();
    if (state.mode === 'single') {
      reset({ clearCache: false });
      if (typeof window.drawSelectedLayer === 'function') window.drawSelectedLayer();
      return;
    }
    void fetchWindowAndRender();
  }

  function handleSourceChange(which) {
    const ui = getUi();
    if (which === 'a') {
      state.sourceA = parseSourceOptionValue(ui.compareSourceASelect?.value || 'raw');
      syncCompareControls();
    } else {
      state.sourceBAuto = false;
      state.sourceB = parseSourceOptionValue(ui.compareSourceBSelect?.value || 'raw');
      syncCompareControls();
    }
    if (state.mode !== 'single') {
      void fetchWindowAndRender();
    }
  }

  function handleDiffModeChange() {
    const ui = getUi();
    state.diffMode = ui.compareDiffModeSelect?.value === 'a_minus_b' ? 'a_minus_b' : 'b_minus_a';
    if (state.mode === 'difference') {
      void renderFromCache();
    }
  }

  function handleShowRmsChange() {
    const ui = getUi();
    state.showRmsByTrace = !!ui.compareShowRms?.checked;
    updateControlState();
    if (state.mode === 'difference') {
      void renderFromCache();
    }
  }

  function bindControls() {
    const ui = getUi();
    if (ui.compareModeSelect) ui.compareModeSelect.addEventListener('change', handleModeChange);
    if (ui.compareSourceASelect) ui.compareSourceASelect.addEventListener('change', () => handleSourceChange('a'));
    if (ui.compareSourceBSelect) ui.compareSourceBSelect.addEventListener('change', () => handleSourceChange('b'));
    if (ui.compareDiffModeSelect) ui.compareDiffModeSelect.addEventListener('change', handleDiffModeChange);
    if (ui.compareShowRms) ui.compareShowRms.addEventListener('change', handleShowRmsChange);

    if (ui.layerSelect && typeof MutationObserver === 'function') {
      const observer = new MutationObserver(() => {
        const { sourceAChanged, sourceBChanged } = syncCompareControls();
        if (state.mode !== 'single' && (sourceAChanged || sourceBChanged)) {
          void fetchWindowAndRender();
        }
      });
      observer.observe(ui.layerSelect, {
        childList: true,
        subtree: true,
        attributes: true,
      });
    }

    const resizeTargets = [
      ui.compareSurface,
      ui.comparePlotA,
      ui.comparePlotB,
      ui.comparePlotDiff,
      ui.compareRmsPlot,
    ].filter(Boolean);
    if (typeof ResizeObserver === 'function' && resizeTargets.length) {
      const resizeObserver = new ResizeObserver(() => {
        if (state.mode === 'single') return;
        const plots = [ui.comparePlotA, ui.comparePlotB, ui.comparePlotDiff, ui.compareRmsPlot];
        for (const plotDiv of plots) {
          if (!plotDiv || typeof window.Plotly?.Plots?.resize !== 'function') continue;
          try {
            window.Plotly.Plots.resize(plotDiv);
          } catch (_) {
            // Ignore empty plot resize failures.
          }
        }
      });
      for (const target of resizeTargets) resizeObserver.observe(target);
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    bindControls();
    syncCompareControls();
    updateSurfaceVisibility();
  });

  window.compareView = {
    isActive() {
      return state.mode !== 'single';
    },
    fetchWindowAndRender,
    renderFromCache,
    reset,
    restylePlots,
    syncControls: syncCompareControls,
  };
})();
