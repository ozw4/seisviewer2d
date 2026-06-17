let stationStructureCanvasCleanup = null;

export function cleanupStationStructureCanvasRenderer() {
  if (!stationStructureCanvasCleanup) return;
  stationStructureCanvasCleanup();
  stationStructureCanvasCleanup = null;
}

export function renderStationStructureView({
  root,
  viewState,
  clearNode,
  controllerActions,
  createKv,
  formatNumber,
  getCanvas2dContext,
  paddedRange,
  pickMapTicks,
  plotHeight,
  toFiniteNumber,
}) {
  const state = viewState;

  function renderStationStructure(content) {
    cleanupStationStructureCanvasRenderer();
    content.appendChild(createStationStructureControls());
  
    if (state.stationStructureLoading) {
      const loading = document.createElement('p');
      loading.className = 'refraction-qc-placeholder';
      loading.textContent = 'Loading station-structure QC...';
      content.appendChild(loading);
      return;
    }
    if (state.stationStructureError) {
      const error = document.createElement('div');
      error.className = 'refraction-qc-error';
      error.dataset.testid = 'refraction-qc-station-structure-error';
      error.textContent = state.stationStructureError;
      content.appendChild(error);
    }
    if (!state.stationStructure) {
      const empty = document.createElement('p');
      empty.className = 'refraction-qc-placeholder';
      empty.textContent = state.qcBundle
        ? 'Load station-structure QC for this completed refraction job.'
        : 'Load a completed job before requesting station-structure QC.';
      content.appendChild(empty);
      return;
    }
  
    const payload = state.stationStructure;
    const status = document.createElement('p');
    status.className = 'refraction-qc-note';
    status.dataset.testid = 'refraction-qc-station-structure-filter-status';
    status.textContent = stationStructureFilterNote(payload);
    content.appendChild(status);
  
    const warnings = Array.isArray(payload.warnings) ? payload.warnings : [];
    for (const warningText of warnings) {
      const warning = document.createElement('p');
      warning.className = 'refraction-qc-note';
      warning.textContent = warningText;
      content.appendChild(warning);
    }
  
    content.appendChild(createKv([
      ['X axis', payload.x_axis_label || payload.x_axis],
      ['Velocity', String(payload.velocity?.field || '').toUpperCase()],
      ['Depth / structure', payload.depth?.label || payload.depth?.field],
      ['Source color', payload.colors?.source || 'cyan'],
      ['Receiver color', payload.colors?.receiver || 'red'],
    ]));
  
    const grid = document.createElement('div');
    grid.className = 'refraction-qc-plot-grid refraction-qc-station-structure-grid';
    content.appendChild(grid);
  
    const panels = [
      {
        key: 'time_term',
        title: 'Time-term distribution',
        yAxisTitle: 'Time term (ms)',
        testId: 'refraction-qc-station-structure-time-term',
      },
      {
        key: 'velocity',
        title: payload.velocity?.label || 'Velocity structure',
        yAxisTitle: 'Velocity (m/s)',
        testId: 'refraction-qc-station-structure-velocity',
      },
      {
        key: 'depth',
        title: payload.depth?.label || 'Depth / structure',
        yAxisTitle: stationStructureDepthYAxisTitle(payload.depth),
        testId: 'refraction-qc-station-structure-depth',
      },
    ];
  
    const cleanups = [];
    for (const panelConfig of panels) {
      const plot = document.createElement('div');
      plot.className = 'refraction-qc-plot refraction-qc-station-structure-plot';
      plot.dataset.testid = `${panelConfig.testId}-plot`;
      plot.dataset.renderer = 'canvas';
      plot.dataset.xAxisTitle = payload.x_axis_label || payload.x_axis || '';
      plot.dataset.yAxisTitle = panelConfig.yAxisTitle;
      const panel = payload[panelConfig.key] || {};
      const pointCount = stationStructurePointCount(panel);
      plot.dataset.pointCount = String(pointCount);
      grid.appendChild(plot);
      if (!pointCount) {
        plot.textContent = `No finite ${panelConfig.title.toLowerCase()} values are available for the selected range.`;
        continue;
      }
      cleanups.push(renderStationStructureCanvas(plot, payload, panel, panelConfig));
    }
    stationStructureCanvasCleanup = () => {
      for (const cleanup of cleanups) cleanup();
    };
  }

  function createStationStructureControls() {
    const controls = document.createElement('div');
    controls.className = 'refraction-qc-controls';
  
    const gatherStart = document.createElement('input');
    gatherStart.type = 'number';
    gatherStart.placeholder = 'Shot gather start';
    gatherStart.value = state.stationStructureGatherStart;
    gatherStart.dataset.testid = 'refraction-qc-station-structure-gather-start';
    gatherStart.addEventListener('input', () => {
      state.stationStructureGatherStart = gatherStart.value;
    });
  
    const gatherEnd = document.createElement('input');
    gatherEnd.type = 'number';
    gatherEnd.placeholder = 'Shot gather end';
    gatherEnd.value = state.stationStructureGatherEnd;
    gatherEnd.dataset.testid = 'refraction-qc-station-structure-gather-end';
    gatherEnd.addEventListener('input', () => {
      state.stationStructureGatherEnd = gatherEnd.value;
    });
  
    const velocity = document.createElement('select');
    velocity.dataset.testid = 'refraction-qc-station-structure-velocity-field';
    for (const [value, label] of [
      ['auto', 'Auto'],
      ['v1', 'V1'],
      ['v2', 'V2'],
      ['v3', 'V3'],
      ['vsub', 'Vsub'],
    ]) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      velocity.appendChild(option);
    }
    velocity.value = state.stationStructureVelocityField;
    velocity.addEventListener('change', () => {
      state.stationStructureVelocityField = velocity.value;
    });
  
    const depth = document.createElement('select');
    depth.dataset.testid = 'refraction-qc-station-structure-depth-field';
    for (const [value, label] of [
      ['auto', 'Auto'],
      ['sh1', 'Weathering thickness SH1'],
      ['sh2', 'Weathering thickness SH2'],
      ['sh3', 'Weathering thickness SH3'],
      ['refractor_depth', 'Refractor depth'],
      ['refractor_elevation', 'Refractor elevation'],
      ['layer1_base_elevation', 'Layer 1 base elevation'],
      ['layer2_base_elevation', 'Layer 2 base elevation'],
    ]) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      depth.appendChild(option);
    }
    depth.value = state.stationStructureDepthField;
    depth.addEventListener('change', () => {
      state.stationStructureDepthField = depth.value;
    });
  
    const loadButton = document.createElement('button');
    loadButton.type = 'button';
    loadButton.textContent = 'Load / Refresh';
    loadButton.dataset.testid = 'refraction-qc-station-structure-load';
    loadButton.disabled = !(state.qcBundle?.job_id || state.selectedJobId);
    loadButton.addEventListener('click', () => {
      controllerActions.loadStationStructureQc();
    });
  
    controls.append(gatherStart, gatherEnd, velocity, depth, loadButton);
    return controls;
  }

  function stationStructureFilterNote(payload) {
    const status = payload.filter_status || 'unknown';
    const start = payload.gather_range?.start ?? '';
    const end = payload.gather_range?.end ?? '';
    const rangeText = start === '' && end === '' ? 'all gathers' : `shot gathers ${start || 'min'} to ${end || 'max'}`;
    if (status === 'ok') return `Filter: ${rangeText}.`;
    if (status === 'unfiltered') return 'Filter: all available source and receiver endpoints.';
    if (status === 'receiver_participation_unavailable') {
      return `Filter: source endpoints use ${rangeText}; receiver participation is unavailable from artifacts.`;
    }
    return `Filter status: ${status}.`;
  }

  function stationStructureDepthYAxisTitle(panel) {
    const field = panel?.field || '';
    if (field === 'refractor_elevation' || field === 'layer1_base_elevation' || field === 'layer2_base_elevation') {
      return 'Elevation (m)';
    }
    return 'Depth / structure (m)';
  }

  function stationStructurePointCount(panel) {
    const source = Array.isArray(panel?.source?.x) ? panel.source.x.length : 0;
    const receiver = Array.isArray(panel?.receiver?.x) ? panel.receiver.x.length : 0;
    return source + receiver;
  }

  function renderStationStructureCanvas(plot, payload, panel, panelConfig) {
    clearNode(plot);
    const canvas = document.createElement('canvas');
    canvas.className = 'refraction-qc-station-structure-canvas';
    canvas.dataset.testid = `${panelConfig.testId}-canvas`;
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', `${panelConfig.title} station-structure scatter plot.`);
    plot.appendChild(canvas);
  
    const draw = () => drawStationStructureCanvas(plot, canvas, payload, panel, panelConfig);
    draw();
    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(draw);
      observer.observe(plot);
      return () => observer.disconnect();
    }
    window.addEventListener('resize', draw);
    return () => window.removeEventListener('resize', draw);
  }

  function drawStationStructureCanvas(plot, canvas, payload, panel, panelConfig) {
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
  
    const points = stationStructurePoints(panel, payload.colors || {});
    const rect = plot.getBoundingClientRect();
    const cssWidth = Math.max(1, Math.floor(rect.width || plot.clientWidth || plot.offsetWidth || 640));
    const cssHeight = Math.max(plotHeight(250, 300), Math.floor(rect.height || plot.clientHeight || plot.offsetHeight || 0));
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
  
    const margin = { left: 68, right: 22, top: 34, bottom: 52 };
    const plotWidth = Math.max(1, cssWidth - margin.left - margin.right);
    const plotHeightCss = Math.max(1, cssHeight - margin.top - margin.bottom);
    const xRange = paddedRange(points, (point) => point.x);
    const yRange = paddedRange(points, (point) => point.y);
    const xScale = (value) => margin.left + ((value - xRange.min) / (xRange.max - xRange.min)) * plotWidth;
    const yScale = (value) => margin.top + plotHeightCss - ((value - yRange.min) / (yRange.max - yRange.min)) * plotHeightCss;
  
    drawStationStructureGrid(context, margin, plotWidth, plotHeightCss, xRange, yRange, xScale, yScale, payload, panelConfig);
    for (const point of points) {
      context.beginPath();
      context.arc(xScale(point.x), yScale(point.y), point.status === 'ok' ? 3 : 2.5, 0, Math.PI * 2);
      context.globalAlpha = point.status === 'ok' ? 0.85 : 0.35;
      context.fillStyle = point.color;
      context.fill();
    }
    context.globalAlpha = 1;
  }

  function stationStructurePoints(panel, colors) {
    const points = [];
    for (const side of ['source', 'receiver']) {
      const series = panel?.[side] || {};
      const count = Array.isArray(series.x) ? series.x.length : 0;
      for (let index = 0; index < count; index += 1) {
        const x = toFiniteNumber(series.x[index]);
        const y = toFiniteNumber(series.y?.[index]);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        points.push({
          x,
          y,
          side,
          color: side === 'source' ? (colors.source || 'cyan') : (colors.receiver || 'red'),
          status: String(series.status?.[index] || 'ok'),
        });
      }
    }
    return points;
  }

  function drawStationStructureGrid(context, margin, plotWidth, plotHeightCss, xRange, yRange, xScale, yScale, payload, panelConfig) {
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
    context.textAlign = 'left';
    context.textBaseline = 'top';
    context.fillText(panelConfig.title, margin.left, 10);
    context.fillStyle = payload.colors?.source || 'cyan';
    context.fillText('source', right - 90, 10);
    context.fillStyle = payload.colors?.receiver || 'red';
    context.fillText('receiver', right - 48, 10);
    context.fillStyle = '#334155';
    context.textAlign = 'center';
    context.textBaseline = 'bottom';
    context.fillText(payload.x_axis_label || payload.x_axis || 'Station', margin.left + plotWidth / 2, bottom + 42);
    context.save();
    context.translate(14, margin.top + plotHeightCss / 2);
    context.rotate(-Math.PI / 2);
    context.textBaseline = 'top';
    context.fillText(panelConfig.yAxisTitle, 0, 0);
    context.restore();
    context.restore();
  }

  renderStationStructure(root);
}
