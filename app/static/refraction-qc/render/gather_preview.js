export function renderGatherPreviewView({
  root,
  bundle,
  viewState,
  GATHER_AXIS_LABELS,
  GATHER_DISPLAY_LABELS,
  createFirstBreakPlot,
  createGatherContextBanner,
  createGatherOverlayLegend,
  createKv,
  formatNumber,
  gatherEndpointSummary,
  gatherHasResidualScale,
  gatherOverlayTrace,
  gatherPreviewAxisContext,
  getPlotly,
  plotHeight,
  plotlyNewPlot,
  plotlyResize,
  plotlyUnavailableMessage,
  textOrDash,
}) {
  const state = viewState;

  function correctedStatusMessage(preview) {
    const ref = preview?.corrected_window_ref || {};
    const status = String(ref.status || '').trim();
    if (!status) return '';
    if (status === 'ok') {
      return `Corrected data source: ${textOrDash(preview.corrected_samples_source)}.`;
    }
    const message = String(ref.message || '').trim();
    return `Missing corrected data: status ${status}. ${message || `Preview samples source: ${textOrDash(preview.corrected_samples_source)}.`}`;
  }

  function renderGatherHeatmap(content, preview, options) {
    const plot = createFirstBreakPlot(options.testId);
    plot.classList.add('refraction-qc-gather-plot');
    plot.dataset.pointCount = String(preview?.shape?.[1] || 0);
    content.appendChild(plot);

    if (!getPlotly()) {
      plot.textContent = plotlyUnavailableMessage();
      return;
    }

    const axisContext = options.axisContext || gatherPreviewAxisContext(preview);
    const xAxis = axisContext.xAxis;
    const y = axisContext.yValues;
    const xRange = axisContext.xRange;
    const yRange = axisContext.yRange;
    const reversedYRange = Array.isArray(yRange) ? [yRange[1], yRange[0]] : undefined;
    const sideBySide = Boolean(options.sideBySide);
    const showAmplitudeScale = !sideBySide;
    const showResidualScale = !sideBySide || Boolean(options.corrected);
    const showOverlayLegend = !sideBySide;
    const observedField = options.corrected
      ? 'corrected_observed_pick_time_s'
      : 'observed_pick_time_s';
    const modeledField = options.corrected
      ? 'corrected_modeled_pick_time_s'
      : 'modeled_pick_time_s';
    const observedTrace = gatherOverlayTrace(preview, xAxis.x, {
      field: observedField,
      name: options.corrected ? 'Corrected observed first break' : 'Observed first break',
      color: '#2563eb',
      symbol: 'circle',
      residual: true,
      showResidualScale,
      showlegend: showOverlayLegend,
    });
    const hasResidualColorbar = Boolean(observedTrace?.marker?.showscale);
    const splitColorbars = showAmplitudeScale && hasResidualColorbar;
    const amplitudeColorbar = showAmplitudeScale
      ? {
          title: { text: 'Amplitude' },
          ...(splitColorbars ? { x: 1.02, xanchor: 'left', y: 0.76, yanchor: 'middle', len: 0.42 } : {}),
        }
      : undefined;
    if (splitColorbars && observedTrace?.marker?.colorbar) {
      observedTrace.marker.colorbar = {
        ...observedTrace.marker.colorbar,
        y: 0.24,
        yanchor: 'middle',
        len: 0.42,
      };
    }
    const showAnyColorbar = showAmplitudeScale || hasResidualColorbar;
    const sideBySideRightMargin = options.corrected && gatherHasResidualScale(preview) ? 86 : 24;
    const marginRight = sideBySide
      ? sideBySideRightMargin
      : splitColorbars ? 96 : showAnyColorbar ? 86 : 24;
    const traces = [
      {
        name: options.title,
        type: 'heatmap',
        x: xAxis.x,
        y,
        z: options.samples,
        colorscale: 'Greys',
        zsmooth: false,
        showscale: showAmplitudeScale,
        colorbar: amplitudeColorbar,
        hovertemplate: `${options.title}<br>${xAxis.label}: %{x}<br>Time: %{y:.4f} s<br>Amplitude: %{z:.4g}<extra></extra>`,
      },
    ];
    const modeledTrace = gatherOverlayTrace(preview, xAxis.x, {
      field: modeledField,
      name: options.corrected ? 'Corrected modeled first break' : 'Modeled first break',
      color: '#f97316',
      symbol: 'x',
      size: 9,
      showlegend: showOverlayLegend,
    });
    if (observedTrace) traces.push(observedTrace);
    if (modeledTrace) traces.push(modeledTrace);

    const promise = plotlyNewPlot(plot, traces, {
      height: plotHeight(320, 520),
      margin: { l: 62, r: marginRight, t: 64, b: 52 },
      font: { size: 10, color: '#334155' },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      showlegend: showOverlayLegend,
      title: { text: options.title, font: { size: 12 } },
      xaxis: {
        title: { text: xAxis.label },
        range: xRange,
        autorange: xRange ? false : true,
        zeroline: false,
        gridcolor: '#e5e7eb',
      },
      yaxis: {
        title: { text: 'Time (s)' },
        range: reversedYRange,
        autorange: reversedYRange ? false : 'reversed',
        zeroline: false,
        gridcolor: '#e5e7eb',
      },
      legend: {
        orientation: 'h',
        x: 0,
        y: 1.22,
        xanchor: 'left',
        yanchor: 'top',
        font: { size: 10 },
      },
    }, { displayModeBar: false, responsive: true });
    Promise.resolve(promise).then(() => {
      plotlyResize(plot);
    });
  }

  function renderGatherReducedTime(content, preview) {
    const plot = createFirstBreakPlot('refraction-qc-gather-reduced-plot');
    plot.classList.add('refraction-qc-gather-plot');
    content.appendChild(plot);

    if (!getPlotly()) {
      plot.textContent = plotlyUnavailableMessage();
      return;
    }

    const xAxis = gatherPreviewAxisContext(preview).xAxis;
    const traces = [];
    const observed = gatherOverlayTrace(preview, xAxis.x, {
      field: 'reduced_observed_time_s',
      name: 'Reduced observed first break',
      color: '#2563eb',
      symbol: 'circle',
      residual: true,
    });
    const modeled = gatherOverlayTrace(preview, xAxis.x, {
      field: 'reduced_modeled_time_s',
      name: 'Reduced modeled first break',
      color: '#f97316',
      symbol: 'x',
      size: 9,
    });
    if (observed) traces.push(observed);
    if (modeled) traces.push(modeled);

    if (!traces.length) {
      plot.textContent = 'Reduced-time values are unavailable for this preview.';
      return;
    }

    plotlyNewPlot(plot, traces, {
      height: plotHeight(320, 520),
      margin: { l: 62, r: 18, t: 38, b: 52 },
      font: { size: 10, color: '#334155' },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      title: { text: 'Reduced-time first-break preview', font: { size: 12 } },
      xaxis: {
        title: { text: xAxis.label },
        zeroline: false,
        gridcolor: '#e5e7eb',
      },
      yaxis: {
        title: { text: 'Reduced time (s)' },
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
    }, { displayModeBar: false, responsive: true });
  }

  function renderGatherPreviewPayload(content, preview) {
    const statusMessage = correctedStatusMessage(preview);
    content.appendChild(createGatherContextBanner(preview));
    content.appendChild(createKv([
      ['Job ID', preview.job_id],
      ['Gather', GATHER_AXIS_LABELS[preview.gather?.axis] || preview.gather?.axis],
      ['Station', preview.gather?.axis === 'section'
        ? 'Section window'
        : gatherEndpointSummary(preview.gather?.axis === 'receiver' ? 'receiver' : 'source', preview.gather?.endpoint_key)],
      ['Trace count', `${preview.window?.returned_trace_count || 0} of ${preview.window?.requested_trace_count || 0}`],
      ['Samples', `${preview.window?.returned_sample_count || 0} of ${preview.window?.requested_sample_count || 0}`],
      ['dt', `${formatNumber(Number(preview.dt_s), 6)} s`],
      ['Display', GATHER_DISPLAY_LABELS[state.gatherDisplayMode] || state.gatherDisplayMode],
      ['Corrected source', preview.corrected_samples_source],
    ]));

    const signNote = document.createElement('p');
    signNote.className = 'refraction-qc-note';
    signNote.textContent = `${preview.sign_convention}; positive shift_s delays displayed events and negative shift_s advances displayed events.`;
    signNote.dataset.testid = 'refraction-qc-gather-sign-note';
    content.appendChild(signNote);

    if (statusMessage) {
      const status = document.createElement('p');
      status.className = 'refraction-qc-note';
      status.textContent = statusMessage;
      status.dataset.testid = 'refraction-qc-gather-corrected-status';
      content.appendChild(status);
    }

    const overlayNote = document.createElement('p');
    overlayNote.className = 'refraction-qc-note';
    overlayNote.textContent = `First-break overlays: ${textOrDash(preview.overlay_status?.first_break_fit)}; residual colors use observed - modeled.`;
    overlayNote.dataset.testid = 'refraction-qc-gather-overlay-status';
    content.appendChild(overlayNote);

    if (state.gatherDisplayMode === 'reduced_time') {
      renderGatherReducedTime(content, preview);
      return;
    }

    const axisContext = gatherPreviewAxisContext(preview);
    const plotGrid = document.createElement('div');
    const sideBySide = state.gatherDisplayMode === 'side_by_side';
    if (sideBySide) {
      content.appendChild(createGatherOverlayLegend());
    }
    plotGrid.className = sideBySide
      ? 'refraction-qc-gather-grid refraction-qc-gather-grid-side-by-side'
      : 'refraction-qc-plot-grid';
    content.appendChild(plotGrid);

    if (state.gatherDisplayMode === 'raw' || state.gatherDisplayMode === 'side_by_side') {
      renderGatherHeatmap(plotGrid, preview, {
        testId: 'refraction-qc-gather-raw-plot',
        title: 'Raw gather',
        samples: preview.raw_samples,
        corrected: false,
        axisContext,
        sideBySide,
      });
    }
    if (state.gatherDisplayMode === 'corrected' || state.gatherDisplayMode === 'side_by_side') {
      renderGatherHeatmap(plotGrid, preview, {
        testId: 'refraction-qc-gather-corrected-plot',
        title: 'Corrected gather',
        samples: preview.corrected_samples,
        corrected: true,
        axisContext,
        sideBySide,
      });
    }
  }

  function renderGatherPreview(content, bundle) {
    if (state.gatherLoading) {
      const message = document.createElement('p');
      message.className = 'refraction-qc-placeholder';
      message.textContent = 'Loading bounded gather preview from the M6 API...';
      content.appendChild(message);
    }
    if (state.gatherError) {
      const error = document.createElement('p');
      error.className = 'refraction-qc-error';
      error.dataset.testid = 'refraction-qc-gather-error';
      error.textContent = state.gatherError;
      content.appendChild(error);
    }
    if (!state.gatherPreview && !state.gatherLoading && state.gatherEndpointKey) {
      content.appendChild(createGatherContextBanner());
    }
    if (!state.gatherPreview && !state.gatherLoading && !state.gatherError) {
      const message = document.createElement('p');
      message.className = 'refraction-qc-placeholder';
      message.textContent = bundle
        ? 'Choose a station and preview the gather.'
        : 'Load a QC bundle before requesting a gather preview.';
      content.appendChild(message);
    }
    if (state.gatherPreview) {
      renderGatherPreviewPayload(content, state.gatherPreview);
    }
  }

  renderGatherPreview(root, bundle);
}
