export function renderReducedTimeView({
  root,
  bundle,
  viewDef,
  viewState,
  LAYER_COLORS,
  createFirstBreakPlot,
  createKv,
  createTable,
  filteredReducedTimePoints,
  findDownsampling,
  findViewData,
  firstBreakXAxisDefinition,
  getPlotly,
  isUnavailable,
  layerLabel,
  plotHeight,
  plotlyNewPlot,
  plotlyUnavailableMessage,
  reducedTimeGateOverlays,
  reducedTimeHoverText,
  reducedTimeVelocitySummary,
  statusCounts,
}) {
  const state = viewState;

  function renderReducedTimePlot(content, bundle, viewDef) {
    const found = findViewData(bundle, viewDef);
    if (!found) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = isUnavailable(bundle, viewDef)
        ? 'This view is unavailable from the loaded QC bundle artifacts.'
        : 'No sampled reduced-time records are present for this view.';
      content.appendChild(missing);
      return;
    }
  
    const { key, view } = found;
    const points = filteredReducedTimePoints(view);
    const plottedPoints = points.filter((point) => Number.isFinite(point.reducedMs));
    const unavailablePoints = points.filter((point) => !Number.isFinite(point.reducedMs));
    const downsampling = findDownsampling(bundle, key, view);
    const downsamplingText = downsampling
      ? `${downsampling.returned_points || 0} of ${downsampling.total_points || 0}; ${downsampling.downsampled ? 'downsampled' : 'not downsampled'}${downsampling.method ? ` (${downsampling.method})` : ''}`
      : 'not reported';
    const gateOverlay = reducedTimeGateOverlays(bundle);
  
    content.appendChild(createKv([
      ['Bundle view', key],
      ['Artifact', view.artifact],
      ['Rows', `${view.returned_points || 0} of ${view.total_points || 0}`],
      ['Plotted points', `${plottedPoints.length}`],
      ['Layer filter', state.selectedLayerKind === 'all' ? 'all' : layerLabel(state.selectedLayerKind)],
      ['Unused picks', state.showRejectedFirstBreaks ? 'shown' : 'hidden'],
      ['Reduction velocity', reducedTimeVelocitySummary(points)],
      ['Unavailable rows', unavailablePoints.length ? `${unavailablePoints.length}; ${statusCounts(unavailablePoints, 'unavailableReason')}` : '0'],
    ]));
  
    const formulaNote = document.createElement('p');
    formulaNote.className = 'refraction-qc-note';
    formulaNote.textContent = 'Reduced time = observed first-break time - offset / reduction velocity, shown in ms.';
    formulaNote.dataset.testid = 'refraction-qc-reduced-time-formula-note';
    content.appendChild(formulaNote);
  
    const gateNote = document.createElement('p');
    gateNote.className = 'refraction-qc-note';
    gateNote.textContent = `Gate overlays: ${gateOverlay.label}`;
    gateNote.dataset.testid = 'refraction-qc-reduced-time-gates';
    content.appendChild(gateNote);
  
    const downsamplingNote = document.createElement('p');
    downsamplingNote.className = 'refraction-qc-note';
    downsamplingNote.textContent = `Downsampling: ${downsamplingText}`;
    downsamplingNote.dataset.testid = 'refraction-qc-reduced-time-downsampling';
    content.appendChild(downsamplingNote);
  
    if (!plottedPoints.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = unavailablePoints.length
        ? 'Reduced-time rows matched the filters, but none have an available reduced-time value.'
        : 'No plottable reduced-time records match the current filters.';
      content.appendChild(missing);
      if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
        content.appendChild(createTable(view));
      }
      return;
    }
  
    const plot = createFirstBreakPlot('refraction-qc-reduced-time-plot');
    plot.dataset.pointCount = String(plottedPoints.length);
    content.appendChild(plot);
  
    if (getPlotly()) {
      const xAxis = firstBreakXAxisDefinition();
      const groups = new Map();
      for (const point of plottedPoints) {
        const groupKey = `${point.layerKind}|${point.used ? 'used' : 'unused'}`;
        if (!groups.has(groupKey)) {
          groups.set(groupKey, {
            layerKind: point.layerKind,
            used: point.used,
            x: [],
            y: [],
            text: [],
          });
        }
        const group = groups.get(groupKey);
        group.x.push(point.x);
        group.y.push(point.reducedMs);
        group.text.push(reducedTimeHoverText(point));
      }
      const traces = Array.from(groups.values()).map((group) => ({
        name: `${layerLabel(group.layerKind)} ${group.used ? 'used' : 'unused'}`,
        type: 'scatter',
        mode: 'markers',
        x: group.x,
        y: group.y,
        text: group.text,
        hovertemplate: '%{text}<extra></extra>',
        marker: {
          color: LAYER_COLORS[group.layerKind] || LAYER_COLORS.unknown,
          symbol: group.used ? 'circle' : 'x',
          size: group.used ? 7 : 8,
          opacity: group.used ? 0.9 : 0.55,
        },
      }));
      plotlyNewPlot(plot, traces, {
        height: plotHeight(300, 480),
        margin: { l: 62, r: 14, t: 34, b: 50 },
        font: { size: 10, color: '#334155' },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        title: { text: 'Reduced-time first-break QC', font: { size: 12 } },
        xaxis: {
          title: { text: xAxis.label },
          zeroline: false,
          gridcolor: '#e5e7eb',
        },
        yaxis: {
          title: { text: 'Reduced time (ms)' },
          zeroline: true,
          zerolinecolor: '#94a3b8',
          gridcolor: '#e5e7eb',
        },
        legend: {
          orientation: 'h',
          x: 0,
          y: 1.16,
          xanchor: 'left',
          yanchor: 'top',
          font: { size: 10 },
        },
        shapes: gateOverlay.shapes,
      }, { displayModeBar: false, responsive: true });
    } else {
      plot.textContent = plotlyUnavailableMessage();
    }
  
    if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
      content.appendChild(createTable(view));
    }
  }

  renderReducedTimePlot(root, bundle, viewDef);
}
