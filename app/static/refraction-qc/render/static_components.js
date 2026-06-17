export function renderStaticComponentsView({
  root,
  bundle,
  viewDef,
  viewState,
  STATIC_COMPONENT_DEFS,
  STATIC_ENDPOINT_VIEW_KEY,
  STATIC_TRACE_COMPONENT_DEFS,
  STATIC_TRACE_VIEW_KEY,
  buildStaticComponentRows,
  componentColor,
  createFirstBreakPlot,
  createKv,
  createStaticComponentTable,
  createStaticEndpointActions,
  firstDefined,
  getPlotly,
  isUnavailable,
  matchingLegacyStaticRecord,
  normalizedText,
  plotHeight,
  plotlyNewPlot,
  plotlyUnavailableMessage,
  selectedStaticEndpointRecord,
  selectedStaticTraceRecord,
  staticApplyToTraceShift,
  staticEndpointKey,
  staticEndpointKind,
  staticEndpointRecords,
  staticTraceIndex,
  staticTraceRecords,
  textOrDash,
  viewByKey,
}) {
  const state = viewState;

  function renderStaticComponentBars(content, rows, testId, title) {
    const finiteRows = rows.filter((row) => Number.isFinite(row.value));
    if (!finiteRows.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = `No finite ${title.toLowerCase()} values are available to plot.`;
      content.appendChild(missing);
      return;
    }

    const plot = createFirstBreakPlot(testId);
    plot.classList.add('refraction-qc-static-waterfall');
    plot.dataset.pointCount = String(finiteRows.length);
    content.appendChild(plot);

    if (getPlotly()) {
      plotlyNewPlot(plot, [
        {
          name: title,
          type: 'bar',
          orientation: 'h',
          y: finiteRows.map((row) => row.label),
          x: finiteRows.map((row) => row.value),
          text: finiteRows.map((row) => `${row.valueText}; ${row.direction}; ${row.status}`),
          hovertemplate: '%{y}<br>%{text}<extra></extra>',
          marker: {
            color: finiteRows.map((row) => componentColor(row.value)),
          },
        },
      ], {
        height: Math.max(plotHeight(260, 460), finiteRows.length * 30 + 90),
        margin: { l: 150, r: 18, t: 32, b: 44 },
        font: { size: 10, color: '#334155' },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        title: { text: title, font: { size: 12 } },
        xaxis: {
          title: { text: 'Shift (ms)' },
          zeroline: true,
          zerolinecolor: '#0f172a',
          gridcolor: '#e5e7eb',
        },
        yaxis: {
          automargin: true,
        },
        showlegend: false,
      }, { displayModeBar: false, responsive: true });
    } else {
      plot.textContent = plotlyUnavailableMessage();
    }
  }

  function renderStaticComponents(content, bundle, viewDef) {
    const endpointView = viewByKey(bundle, STATIC_ENDPOINT_VIEW_KEY);
    const traceView = viewByKey(bundle, STATIC_TRACE_VIEW_KEY);
    if (!endpointView || !traceView) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = isUnavailable(bundle, viewDef)
        ? 'Static component QC is unavailable from the loaded refraction_static_component_qc_* artifacts.'
        : 'Static component QC endpoint and trace records are required for this view.';
      content.appendChild(missing);
      return;
    }
  
    const endpointRecords = staticEndpointRecords(endpointView);
    const traceRecords = staticTraceRecords(traceView);
    const endpointRecord = selectedStaticEndpointRecord(endpointRecords);
    const endpointNoMatch = Boolean(normalizedText(state.selectedEndpoint) && !endpointRecord);
    const traceRecord = endpointNoMatch ? null : selectedStaticTraceRecord(traceRecords, endpointRecord);
    const legacyRecord = matchingLegacyStaticRecord(bundle, endpointRecord);
    let staticActionDisabledReason = '';
    if (!endpointRecords.length) {
      staticActionDisabledReason = 'Gather preview is disabled because no endpoint component rows are available.';
    } else if (endpointNoMatch) {
      staticActionDisabledReason = 'Gather preview is disabled because no endpoint matches the current endpoint selector.';
    } else if (!endpointRecord) {
      staticActionDisabledReason = 'Gather preview is disabled because no endpoint is selected.';
    }
    const endpointRows = buildStaticComponentRows(
      endpointRecord,
      legacyRecord,
      STATIC_COMPONENT_DEFS,
      ['static_status'],
    );
    const traceRows = buildStaticComponentRows(
      traceRecord,
      null,
      STATIC_TRACE_COMPONENT_DEFS,
      ['static_status'],
    );
    const signConvention = bundle.sign_convention || firstDefined(endpointRecord, ['sign_convention'])
      || firstDefined(traceRecord, ['sign_convention']);
  
    content.appendChild(createKv([
      ['Endpoint artifact', endpointView.artifact],
      ['Trace artifact', traceView.artifact],
      ['Endpoint rows', `${endpointView.returned_points || 0} of ${endpointView.total_points || 0}`],
      ['Trace rows', `${traceView.returned_points || 0} of ${traceView.total_points || 0}`],
      ['Selected endpoint kind', endpointRecord ? staticEndpointKind(endpointRecord) : state.selectedEndpointKind],
      ['Selected endpoint', endpointRecord ? staticEndpointKey(endpointRecord) : '-'],
      ['Selected trace', traceRecord ? staticTraceIndex(traceRecord) : '-'],
      ['Apply field shift', staticApplyToTraceShift(traceRecord || endpointRecord)],
      ['Endpoint status', textOrDash(firstDefined(endpointRecord, ['static_status']))],
      ['Trace status', textOrDash(firstDefined(traceRecord, ['static_status']))],
    ]));
    content.appendChild(createStaticEndpointActions(endpointRecord, staticActionDisabledReason));
  
    const signNote = document.createElement('p');
    signNote.className = 'refraction-qc-note';
    signNote.textContent = `${signConvention || 'corrected(t) = raw(t - shift_s)'}; positive shift_s delays displayed events and negative shift_s advances displayed events.`;
    signNote.dataset.testid = 'refraction-qc-static-sign-note';
    content.appendChild(signNote);
  
    if (!endpointRecord) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = 'No source/receiver endpoint component rows match the current endpoint selector.';
      content.appendChild(missing);
    } else {
      renderStaticComponentBars(
        content,
        endpointRows,
        'refraction-qc-static-waterfall',
        'Endpoint static components',
      );
      content.appendChild(createStaticComponentTable(
        endpointRows,
        'refraction-qc-static-component-list',
      ));
    }
  
    if (!traceRecord) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = 'No trace component row matches the current trace or endpoint selector.';
      content.appendChild(missing);
    } else {
      renderStaticComponentBars(
        content,
        traceRows,
        'refraction-qc-static-trace-waterfall',
        'Trace static components',
      );
      content.appendChild(createStaticComponentTable(
        traceRows,
        'refraction-qc-static-trace-component-list',
      ));
    }
  }

  renderStaticComponents(root, bundle, viewDef);
}
