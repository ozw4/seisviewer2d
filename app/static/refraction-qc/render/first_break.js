export function renderFirstBreakView({
  root,
  bundle,
  viewDef,
  viewState,
  FIRST_BREAK_SORT_CONTROL_OPTIONS,
  LAYER_COLORS,
  attachFirstBreakPickClickActions,
  createFirstBreakPickActions,
  createFirstBreakPlot,
  createKv,
  createTable,
  filteredFirstBreakPoints,
  findDownsampling,
  findViewData,
  firstBreakPickCustomData,
  firstBreakResidualThresholdMs,
  firstBreakXAxisDefinition,
  formatNumber,
  getPlotly,
  isUnavailable,
  layerLabel,
  optionLabel,
  plotHeight,
  plotHoverText,
  plotlyNewPlot,
  plotlyUnavailableMessage,
  selectedFirstBreakPick,
}) {
  const state = viewState;

  function renderFirstBreakPlots(content, bundle, viewDef) {
    const found = findViewData(bundle, viewDef);
    if (!found) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = isUnavailable(bundle, viewDef)
        ? 'This view is unavailable from the loaded QC bundle artifacts.'
        : 'No sampled first-break fit records are present for this view.';
      content.appendChild(missing);
      return;
    }
  
    const { key, view } = found;
    const points = filteredFirstBreakPoints(view);
    const downsampling = findDownsampling(bundle, key, view);
    const downsamplingText = downsampling
      ? `${downsampling.returned_points || 0} of ${downsampling.total_points || 0}; ${downsampling.downsampled ? 'downsampled' : 'not downsampled'}${downsampling.method ? ` (${downsampling.method})` : ''}`
      : 'not reported';
    const residualThreshold = firstBreakResidualThresholdMs();
  
    content.appendChild(createKv([
      ['Bundle view', key],
      ['Artifact', view.artifact],
      ['Rows', `${view.returned_points || 0} of ${view.total_points || 0}`],
      ['Plotted points', `${points.length}`],
      ['Layer filter', state.selectedLayerKind === 'all' ? 'all' : layerLabel(state.selectedLayerKind)],
      ['Rejected picks', state.showRejectedFirstBreaks ? 'shown' : 'hidden'],
      ['Residual threshold', residualThreshold === null ? 'none' : `${formatNumber(residualThreshold, 1)} ms`],
      ['Sort by', optionLabel(FIRST_BREAK_SORT_CONTROL_OPTIONS, state.firstBreakSortBy)],
    ]));
  
    const residualNote = document.createElement('p');
    residualNote.className = 'refraction-qc-note';
    residualNote.textContent = 'Residual = observed - modeled, shown in ms.';
    residualNote.dataset.testid = 'refraction-qc-first-break-residual-note';
    content.appendChild(residualNote);
  
    const downsamplingNote = document.createElement('p');
    downsamplingNote.className = 'refraction-qc-note';
    downsamplingNote.textContent = `Downsampling: ${downsamplingText}`;
    downsamplingNote.dataset.testid = 'refraction-qc-first-break-downsampling';
    content.appendChild(downsamplingNote);
  
    if (!points.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = 'No plottable observed/modeled first-break records match the current filters.';
      content.appendChild(missing);
      if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
        content.appendChild(createTable(view));
      }
      return;
    }
  
    const selectedPick = selectedFirstBreakPick(points);
    if (selectedPick) content.appendChild(createFirstBreakPickActions(selectedPick));
  
    const plotGrid = document.createElement('div');
    plotGrid.className = 'refraction-qc-plot-grid';
    const timePlot = createFirstBreakPlot('refraction-qc-first-break-time-plot');
    const residualPlot = createFirstBreakPlot('refraction-qc-first-break-residual-plot');
    timePlot.dataset.pointCount = String(points.length);
    residualPlot.dataset.pointCount = String(points.length);
    plotGrid.append(timePlot, residualPlot);
    content.appendChild(plotGrid);
  
    if (getPlotly()) {
      const xAxis = firstBreakXAxisDefinition();
      const hoverText = points.map(plotHoverText);
      const customdata = points.map(firstBreakPickCustomData);
      const commonLayout = {
        height: plotHeight(260, 440),
        margin: { l: 58, r: 14, t: 34, b: 50 },
        font: { size: 10, color: '#334155' },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        xaxis: {
          title: { text: xAxis.label },
          zeroline: false,
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
      };
      const config = { displayModeBar: false, responsive: true };
  
      Promise.resolve(plotlyNewPlot(timePlot, [
        {
          name: 'Observed',
          type: 'scatter',
          mode: 'markers',
          x: points.map((point) => point.x),
          y: points.map((point) => point.observedMs),
          text: hoverText,
          customdata,
          hovertemplate: '%{text}<extra>Observed</extra>',
          marker: {
            color: '#2563eb',
            size: 7,
            opacity: points.map((point) => point.opacity),
          },
        },
        {
          name: 'Modeled',
          type: 'scatter',
          mode: 'markers',
          x: points.map((point) => point.x),
          y: points.map((point) => point.modeledMs),
          text: hoverText,
          customdata,
          hovertemplate: '%{text}<extra>Modeled</extra>',
          marker: {
            color: '#f97316',
            symbol: 'x',
            size: 8,
            opacity: points.map((point) => point.opacity),
          },
        },
      ], {
        ...commonLayout,
        title: { text: 'Observed vs modeled first-break time', font: { size: 12 } },
        yaxis: {
          title: { text: 'First-break time (ms)' },
          zeroline: false,
          gridcolor: '#e5e7eb',
        },
      }, config)).then(() => attachFirstBreakPickClickActions(timePlot));
  
      const residualGroups = new Map();
      for (const point of points) {
        const groupKey = `${point.layerKind}|${point.status}`;
        if (!residualGroups.has(groupKey)) {
          residualGroups.set(groupKey, {
            layerKind: point.layerKind,
            status: point.status,
            x: [],
            y: [],
            text: [],
            customdata: [],
          });
        }
        const group = residualGroups.get(groupKey);
        group.x.push(point.x);
        group.y.push(point.residualMs);
        group.text.push(plotHoverText(point));
        group.customdata.push(firstBreakPickCustomData(point));
      }
      const residualTraces = Array.from(residualGroups.values()).map((group) => ({
        name: `${layerLabel(group.layerKind)} ${group.status}`,
        type: 'scatter',
        mode: 'markers',
        x: group.x,
        y: group.y,
        text: group.text,
        customdata: group.customdata,
        hovertemplate: '%{text}<extra></extra>',
        marker: {
          color: LAYER_COLORS[group.layerKind] || LAYER_COLORS.unknown,
          symbol: group.status === 'rejected' ? 'x' : 'circle',
          size: group.status === 'rejected' ? 8 : 7,
          opacity: group.status === 'rejected' ? 0.55 : 0.9,
        },
      }));
      Promise.resolve(plotlyNewPlot(residualPlot, residualTraces, {
        ...commonLayout,
        title: { text: 'First-break residuals', font: { size: 12 } },
        yaxis: {
          title: { text: 'Residual (ms)' },
          zeroline: true,
          zerolinecolor: '#94a3b8',
          gridcolor: '#e5e7eb',
        },
      }, config)).then(() => attachFirstBreakPickClickActions(residualPlot));
    } else {
      timePlot.textContent = plotlyUnavailableMessage();
      residualPlot.textContent = plotlyUnavailableMessage();
    }
  
    if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
      content.appendChild(createTable(view));
    }
  }

  renderFirstBreakPlots(root, bundle, viewDef);
}
