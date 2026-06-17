let pickMapCanvasCleanup = null;

export function cleanupPickMapCanvasRenderer() {
  if (!pickMapCanvasCleanup) return;
  pickMapCanvasCleanup();
  pickMapCanvasCleanup = null;
}

export function renderPickMapView({
  root,
  viewConfig,
  viewState,
  clearNode,
  controllerActions,
  createKv,
  formatNumber,
  getCanvas2dContext,
  paddedRange,
  pickMapGatherNumber,
  pickMapTicks,
  plotHeight,
  toFiniteNumber,
}) {
  const state = viewState;

  function effectivePickMapDisplayMode(payload = state.pickMap) {
    if (payload && !payload.has_after_statics) return 'before';
    return state.pickMapDisplayMode === 'after' ? 'after' : 'before';
  }

  function renderPickMap(content, viewConfig = PICK_MAP_VIEWS.pick_map) {
    cleanupPickMapCanvasRenderer();
    const displayMode = effectivePickMapDisplayMode();
    const testIdPrefix = viewConfig.testIdPrefix;
  
    const controls = document.createElement('div');
    controls.className = 'refraction-qc-controls';
  
    const beforeButton = document.createElement('button');
    beforeButton.type = 'button';
    beforeButton.textContent = 'Before Statics';
    beforeButton.dataset.testid = `${testIdPrefix}-before`;
    beforeButton.className = displayMode === 'before' ? 'is-active' : '';
    beforeButton.addEventListener('click', () => {
      controllerActions.setPickMapDisplayMode('before');
    });
  
    const afterButton = document.createElement('button');
    afterButton.type = 'button';
    afterButton.textContent = 'After Statics';
    afterButton.dataset.testid = `${testIdPrefix}-after`;
    afterButton.disabled = !state.pickMap?.has_after_statics;
    afterButton.className = displayMode === 'after' ? 'is-active' : '';
    afterButton.addEventListener('click', () => {
      if (!state.pickMap?.has_after_statics) return;
      controllerActions.setPickMapDisplayMode('after');
    });
  
    const gatherStart = document.createElement('input');
    gatherStart.type = 'number';
    gatherStart.placeholder = 'Gather start';
    gatherStart.value = state.pickMapGatherStart;
    gatherStart.dataset.testid = `${testIdPrefix}-gather-start`;
    gatherStart.addEventListener('input', () => {
      controllerActions.setPickMapGatherRange('start', gatherStart.value);
    });
  
    const gatherEnd = document.createElement('input');
    gatherEnd.type = 'number';
    gatherEnd.placeholder = 'Gather end';
    gatherEnd.value = state.pickMapGatherEnd;
    gatherEnd.dataset.testid = `${testIdPrefix}-gather-end`;
    gatherEnd.addEventListener('input', () => {
      controllerActions.setPickMapGatherRange('end', gatherEnd.value);
    });
  
    const cachedButton = document.createElement('button');
    cachedButton.type = 'button';
    cachedButton.textContent = 'Load from Static Correction NPZ';
    cachedButton.dataset.testid = `${testIdPrefix}-load-cached`;
    cachedButton.disabled = !state.pickMapCachedFile;
    cachedButton.addEventListener('click', () => {
      controllerActions.loadPreStaticsPickMap(state.pickMapCachedFile);
    });
  
    const completedButton = document.createElement('button');
    completedButton.type = 'button';
    completedButton.textContent = `Load completed-job ${viewConfig.label}`;
    completedButton.dataset.testid = `${testIdPrefix}-load-job`;
    completedButton.disabled = !(state.qcBundle?.job_id || state.selectedJobId);
    completedButton.addEventListener('click', () => {
      controllerActions.loadCompletedPickMap();
    });
  
    controls.append(beforeButton, afterButton, gatherStart, gatherEnd, cachedButton, completedButton);
    content.appendChild(controls);
  
    if (state.pickMapCacheStatus) {
      const cacheStatus = document.createElement('p');
      cacheStatus.className = 'refraction-qc-note';
      cacheStatus.dataset.testid = `${testIdPrefix}-cache-status`;
      cacheStatus.textContent = state.pickMapCacheStatus;
      if (!state.pickMapCachedFile && controllerActions.activePickMapTarget()) {
        cacheStatus.appendChild(document.createTextNode(' '));
        const link = document.createElement('a');
        link.href = controllerActions.staticCorrectionLinkForTarget(controllerActions.activePickMapTarget());
        link.textContent = 'Open Static Correction';
        cacheStatus.appendChild(link);
      }
      content.appendChild(cacheStatus);
    }
    if (state.pickMapLoading) {
      const loading = document.createElement('p');
      loading.className = 'refraction-qc-placeholder';
      loading.textContent = 'Loading Pick Map...';
      content.appendChild(loading);
      return;
    }
    if (state.pickMapError) {
      const error = document.createElement('div');
      error.className = 'refraction-qc-error';
      error.dataset.testid = `${testIdPrefix}-error`;
      error.textContent = state.pickMapError;
      content.appendChild(error);
    }
    if (!state.pickMap) {
      const empty = document.createElement('p');
      empty.className = 'refraction-qc-placeholder';
      empty.textContent = state.qcBundle
        ? 'No Pick Map loaded for this static job.'
        : 'Load a completed job or a cached Static Correction NPZ.';
      content.appendChild(empty);
      return;
    }
  
    const status = document.createElement('p');
    status.className = 'refraction-qc-note';
    status.dataset.testid = `${testIdPrefix}-status`;
    status.textContent = state.pickMap.status_message || '';
    content.appendChild(status);
  
    const pointResult = filteredPickMapPoints(state.pickMap, viewConfig);
    const points = pointResult.points;
    content.appendChild(createKv([
      ['Mode', state.pickMap.mode],
      ['Receiver numbering', state.pickMap.receiver_number_mode],
      ['Gather min', state.pickMap.gather_range?.min],
      ['Gather max', state.pickMap.gather_range?.max],
      ['Displayed points', points.length],
    ]));
  
    const plot = document.createElement('div');
    plot.className = 'refraction-qc-plot refraction-qc-pick-map-plot';
    plot.dataset.testid = `${testIdPrefix}-plot`;
    plot.dataset.pointCount = String(points.length);
    plot.dataset.renderer = 'canvas';
    plot.dataset.xAxisTitle = viewConfig.xAxisTitle;
    plot.dataset.yAxisTitle = 'First-break pick time (ms)';
    plot.dataset.yAxisDirection = 'down';
    plot.dataset.yAxisAutorange = 'reversed';
    content.appendChild(plot);
  
    if (!points.length) {
      plot.textContent = pointResult.missingXCount > 0 ? viewConfig.emptyMessage : 'No Pick Map records match the current gather range.';
      return;
    }
    renderPickMapCanvas(plot, points, state.pickMap, viewConfig);
  }

  function filteredPickMapPoints(payload, viewConfig = PICK_MAP_VIEWS.pick_map) {
    const data = payload?.pick_map || {};
    const count = Array.isArray(data.pick_before_ms) ? data.pick_before_ms.length : 0;
    const start = toFiniteNumber(state.pickMapGatherStart);
    const end = toFiniteNumber(state.pickMapGatherEnd);
    const hasStart = Number.isFinite(start);
    const hasEnd = Number.isFinite(end);
    const points = [];
    let missingXCount = 0;
    for (let index = 0; index < count; index += 1) {
      const gather = data.gather_id?.[index];
      const gatherNumber = pickMapGatherNumber(gather);
      if (hasStart && Number.isFinite(gatherNumber) && gatherNumber < start) continue;
      if (hasEnd && Number.isFinite(gatherNumber) && gatherNumber > end) continue;
      const beforeMs = toFiniteNumber(data.pick_before_ms?.[index]);
      const afterMs = toFiniteNumber(data.pick_after_ms?.[index]);
      const y = effectivePickMapDisplayMode(payload) === 'after' ? afterMs : beforeMs;
      const x = toFiniteNumber(data[viewConfig.xField]?.[index]);
      if (!Number.isFinite(y)) continue;
      if (!Number.isFinite(x)) {
        missingXCount += 1;
        continue;
      }
      const used = data.used_in_statics?.[index] === true;
      points.push({
        x,
        y,
        gather,
        sourceId: data.source_id?.[index],
        receiverId: data.receiver_id?.[index],
        beforeMs,
        afterMs,
        offsetM: toFiniteNumber(data.offset_m?.[index]),
        offsetUsed: toFiniteNumber(data.offset_used?.[index]),
        used,
        appliedShiftMs: toFiniteNumber(data.applied_shift_ms?.[index]),
      });
    }
    return { points, missingXCount };
  }

  function renderPickMapCanvas(plot, points, payload, viewConfig = PICK_MAP_VIEWS.pick_map) {
    cleanupPickMapCanvasRenderer();
    clearNode(plot);
  
    const canvas = document.createElement('canvas');
    canvas.className = 'refraction-qc-pick-map-canvas';
    canvas.dataset.testid = `${viewConfig.testIdPrefix}-canvas`;
    canvas.setAttribute('role', 'img');
    canvas.setAttribute(
      'aria-label',
      `${viewConfig.label} scatter plot with ${viewConfig.xAxisTitle} on x and first-break pick time in milliseconds increasing downward.`
    );
    plot.appendChild(canvas);
  
    const pointStats = pickMapPointStats(points, payload, viewConfig);
    plot.dataset.usedPointCount = String(pointStats.used);
    plot.dataset.unusedPointCount = String(pointStats.unused);
    plot.dataset.offsetColorCount = String(pointStats.offsetColored);
  
    const draw = () => drawPickMapCanvas(plot, canvas, points, payload, viewConfig);
    draw();
  
    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(draw);
      observer.observe(plot);
      pickMapCanvasCleanup = () => observer.disconnect();
    } else {
      window.addEventListener('resize', draw);
      pickMapCanvasCleanup = () => window.removeEventListener('resize', draw);
    }
  }

  function pickMapPointStats(points, payload, viewConfig = PICK_MAP_VIEWS.pick_map) {
    if (payload.mode !== 'completed_job') {
      let offsetColored = 0;
      for (const point of points) {
        if (viewConfig.colorByOffset && Number.isFinite(point.offsetM)) offsetColored += 1;
      }
      return { used: points.length, unused: 0, offsetColored };
    }
    let used = 0;
    let unused = 0;
    let offsetColored = 0;
    for (const point of points) {
      if (point.used) {
        used += 1;
        if (viewConfig.colorByOffset && Number.isFinite(pickMapColorValue(point))) offsetColored += 1;
      } else {
        unused += 1;
      }
    }
    return { used, unused, offsetColored };
  }

  function drawPickMapCanvas(plot, canvas, points, payload, viewConfig = PICK_MAP_VIEWS.pick_map) {
    let context = null;
    try {
      context = getCanvas2dContext(canvas);
    } catch (_) {
      context = null;
    }
    if (!context) {
      plot.textContent = 'Canvas rendering is unavailable.';
      return;
    }
  
    const rect = plot.getBoundingClientRect();
    const cssWidth = Math.max(1, Math.floor(rect.width || plot.clientWidth || plot.offsetWidth || 640));
    const cssHeight = Math.max(plotHeight(340, 560), Math.floor(rect.height || plot.clientHeight || plot.offsetHeight || 0));
    const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
    const pixelWidth = Math.max(1, Math.floor(cssWidth * pixelRatio));
    const pixelHeight = Math.max(1, Math.floor(cssHeight * pixelRatio));
    if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
    if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
  
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, cssWidth, cssHeight);
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, cssWidth, cssHeight);
  
    const margin = { left: 66, right: 22, top: 34, bottom: 56 };
    const plotWidth = Math.max(1, cssWidth - margin.left - margin.right);
    const plotHeightCss = Math.max(1, cssHeight - margin.top - margin.bottom);
    const xRange = paddedRange(points, (point) => point.x);
    const yRange = paddedRange(points, (point) => point.y);
    const colorRange = paddedRange(points, (point) => (
      viewConfig.colorByOffset && !(payload.mode === 'completed_job' && !point.used)
        ? pickMapColorValue(point)
        : NaN
    ));
    const xScale = (value) => margin.left + ((value - xRange.min) / (xRange.max - xRange.min)) * plotWidth;
    const yScale = (value) => margin.top + ((value - yRange.min) / (yRange.max - yRange.min)) * plotHeightCss;
  
    drawPickMapGrid(context, margin, plotWidth, plotHeightCss, xRange, yRange, xScale, yScale, viewConfig);
  
    const title = `${effectivePickMapDisplayMode(payload) === 'after' ? 'After Statics' : 'Before Statics'} ${viewConfig.label}`;
    context.fillStyle = '#334155';
    context.font = '12px sans-serif';
    context.textAlign = 'left';
    context.textBaseline = 'top';
    context.fillText(title, margin.left, 10);
  
    if (payload.mode === 'completed_job') {
      drawPickMapPoints(context, points, xScale, yScale, colorRange, payload, viewConfig, (point) => !point.used);
      drawPickMapPoints(context, points, xScale, yScale, colorRange, payload, viewConfig, (point) => point.used);
    } else {
      drawPickMapPoints(context, points, xScale, yScale, colorRange, payload, viewConfig);
    }
  }

  function drawPickMapGrid(context, margin, plotWidth, plotHeightCss, xRange, yRange, xScale, yScale, viewConfig) {
    const right = margin.left + plotWidth;
    const bottom = margin.top + plotHeightCss;
    context.save();
    context.strokeStyle = '#e5e7eb';
    context.lineWidth = 1;
    context.font = '10px sans-serif';
    context.fillStyle = '#334155';
    context.textBaseline = 'middle';
  
    for (const tick of pickMapTicks(xRange)) {
      const x = xScale(tick);
      context.beginPath();
      context.moveTo(x, margin.top);
      context.lineTo(x, bottom);
      context.stroke();
      context.textAlign = 'center';
      context.fillText(formatNumber(tick, 0), x, bottom + 16);
    }
    for (const tick of pickMapTicks(yRange)) {
      const y = yScale(tick);
      context.beginPath();
      context.moveTo(margin.left, y);
      context.lineTo(right, y);
      context.stroke();
      context.textAlign = 'right';
      context.fillText(formatNumber(tick, 1), margin.left - 8, y);
    }
  
    context.strokeStyle = '#94a3b8';
    context.strokeRect(margin.left, margin.top, plotWidth, plotHeightCss);
    context.textAlign = 'center';
    context.textBaseline = 'bottom';
    context.fillText(viewConfig.xAxisTitle, margin.left + plotWidth / 2, margin.top + plotHeightCss + 44);
  
    context.save();
    context.translate(14, margin.top + plotHeightCss / 2);
    context.rotate(-Math.PI / 2);
    context.textBaseline = 'top';
    context.fillText('First-break pick time (ms)', 0, 0);
    context.restore();
    context.restore();
  }

  function drawPickMapPoints(context, points, xScale, yScale, colorRange, payload, viewConfig, includePoint = null) {
    for (const point of points) {
      if (includePoint && !includePoint(point)) continue;
      const x = xScale(point.x);
      const y = yScale(point.y);
      context.beginPath();
      context.arc(x, y, point.used || payload.mode !== 'completed_job' ? 3 : 2.5, 0, Math.PI * 2);
      context.fillStyle = pickMapPointColor(point, colorRange, payload, viewConfig);
      context.globalAlpha = payload.mode === 'completed_job' && !point.used ? 0.35 : 0.85;
      context.fill();
    }
    context.globalAlpha = 1;
  }

  function pickMapColorValue(point) {
    return Number.isFinite(point.offsetUsed) ? point.offsetUsed : point.offsetM;
  }

  function pickMapPointColor(point, colorRange, payload, viewConfig = PICK_MAP_VIEWS.pick_map) {
    if (payload.mode === 'completed_job' && !point.used) return '#94a3b8';
    if (!viewConfig.colorByOffset) return '#2563eb';
    const value = pickMapColorValue(point);
    if (!Number.isFinite(value) || !Number.isFinite(colorRange.min) || colorRange.max <= colorRange.min) {
      return '#2563eb';
    }
    const ratio = Math.max(0, Math.min(1, (value - colorRange.min) / (colorRange.max - colorRange.min)));
    const stops = [
      [68, 1, 84],
      [59, 82, 139],
      [33, 145, 140],
      [94, 201, 98],
      [253, 231, 37],
    ];
    const scaled = ratio * (stops.length - 1);
    const index = Math.min(stops.length - 2, Math.floor(scaled));
    const local = scaled - index;
    const start = stops[index];
    const end = stops[index + 1];
    const channel = (offset) => Math.round(start[offset] + (end[offset] - start[offset]) * local);
    return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
  }

  renderPickMap(root, viewConfig);
}
