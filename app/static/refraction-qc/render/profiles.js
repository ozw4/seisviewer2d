export function renderProfilesView({
  root,
  bundle,
  viewDef,
  viewState,
  PROFILE_SERIES_COLORS,
  attachProfileEndpointClickActions,
  createFirstBreakPlot,
  createKv,
  createProfileEndpointActions,
  createTable,
  filteredProfileRecords,
  findDownsampling,
  findViewData,
  getPlotly,
  isUnavailable,
  layerLabel,
  plotHeight,
  plotlyNewPlot,
  plotlyUnavailableMessage,
  profileAxisTitle,
  profileDisplayValue,
  profileEndpointCustomData,
  profileEndpointKey,
  profileHoverText,
  profileSeriesRawValue,
  profileTraceName,
  selectedProfileSeries,
  unavailableReason,
}) {
  const state = viewState;

  function renderProfilePlot(content, bundle, viewDef) {
    const found = findViewData(bundle, viewDef);
    if (!found) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      if (isUnavailable(bundle, viewDef)) {
        const reason = unavailableReason(bundle, viewDef);
        missing.textContent = reason
          ? `This view is unavailable from refraction_line_profile_qc_* artifacts: ${reason}.`
          : 'This view is unavailable from refraction_line_profile_qc_* artifacts.';
      } else {
        missing.textContent = 'No sampled line-profile records are present for this view.';
      }
      content.appendChild(missing);
      return;
    }
  
    const { key, view } = found;
    const records = filteredProfileRecords(view);
    const { group, available, unavailable } = selectedProfileSeries(records);
    const invalidCount = records.filter((record) => record.invalid).length;
    const downsampling = findDownsampling(bundle, key, view);
    const downsamplingText = downsampling
      ? `${downsampling.returned_points || 0} of ${downsampling.total_points || 0}; ${downsampling.downsampled ? 'downsampled' : 'not downsampled'}${downsampling.method ? ` (${downsampling.method})` : ''}`
      : 'not reported';
    const missingLabels = unavailable.map((series) => series.label).join(', ');
  
    content.appendChild(createKv([
      ['Bundle view', key],
      ['Artifact', view.artifact],
      ['Rows', `${view.returned_points || 0} of ${view.total_points || 0}`],
      ['Plotted endpoints', `${records.length}`],
      ['Profile group', group.label],
      ['Endpoint kind', state.selectedEndpointKind],
      ['Layer filter', state.selectedLayerKind === 'all' ? 'all' : layerLabel(state.selectedLayerKind)],
      ['Status filter', state.profileStatusFilter],
      ['Invalid endpoints', `${invalidCount}`],
      ['Y units', profileAxisTitle(group, available)],
      ['Unavailable fields', missingLabels || 'none'],
    ]));
  
    const signNote = document.createElement('p');
    signNote.className = 'refraction-qc-note';
    signNote.textContent = 'Static shifts follow corrected(t) = raw(t - shift_s); positive shift_s delays displayed events.';
    signNote.dataset.testid = 'refraction-qc-profile-sign-note';
    content.appendChild(signNote);
  
    const downsamplingNote = document.createElement('p');
    downsamplingNote.className = 'refraction-qc-note';
    downsamplingNote.textContent = `Downsampling: ${downsamplingText}`;
    downsamplingNote.dataset.testid = 'refraction-qc-profile-downsampling';
    content.appendChild(downsamplingNote);
  
    if (!records.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = 'No line-profile endpoints match the current endpoint and status filters.';
      content.appendChild(missing);
      if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
        content.appendChild(createTable(view));
      }
      return;
    }
  
    if (!available.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = 'No plottable profile fields match the current group and layer selection.';
      content.appendChild(missing);
      if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
        content.appendChild(createTable(view));
      }
      return;
    }
  
    const plot = createFirstBreakPlot('refraction-qc-profile-plot');
    plot.dataset.pointCount = String(records.length);
    const selectedProfileEndpoint = state.selectedProfileEndpoint;
    if (selectedProfileEndpoint) {
      const selectedKey = profileEndpointKey(selectedProfileEndpoint);
      const selectedRecord = records.find((record) => (
        profileEndpointKey({
          endpointKind: record.endpointKind,
          endpointKey: record.endpointKey,
        }) === selectedKey
      ));
      if (selectedRecord) content.appendChild(createProfileEndpointActions(selectedProfileEndpoint));
    }
    content.appendChild(plot);
  
    if (getPlotly()) {
      const traces = [];
      for (const series of available) {
        const grouped = new Map();
        for (const point of records) {
          const rawValue = profileSeriesRawValue(point.raw, series);
          const y = profileDisplayValue(rawValue, series, group);
          if (!Number.isFinite(y)) continue;
          if (!grouped.has(point.endpointKind)) {
            grouped.set(point.endpointKind, {
              endpointKind: point.endpointKind,
              x: [],
              y: [],
              text: [],
              customdata: [],
              invalid: [],
            });
          }
          const entry = grouped.get(point.endpointKind);
          entry.x.push(point.inline);
          entry.y.push(y);
          entry.text.push(profileHoverText(point, series, y, group));
          entry.customdata.push(profileEndpointCustomData(point));
          entry.invalid.push(point.invalid);
        }
        for (const entry of grouped.values()) {
          traces.push({
            name: profileTraceName(series, entry.endpointKind, group),
            type: 'scatter',
            mode: 'lines+markers',
            x: entry.x,
            y: entry.y,
            text: entry.text,
            customdata: entry.customdata,
            hovertemplate: '%{text}<extra></extra>',
            line: {
              color: PROFILE_SERIES_COLORS[series.key] || '#64748b',
              width: 2,
              dash: entry.endpointKind === 'receiver' ? 'dot' : 'solid',
            },
            marker: {
              color: PROFILE_SERIES_COLORS[series.key] || '#64748b',
              symbol: entry.invalid.map((invalid) => {
                if (invalid) return 'x';
                return entry.endpointKind === 'receiver' ? 'diamond' : 'circle';
              }),
              size: entry.invalid.map((invalid) => invalid ? 9 : 7),
              opacity: entry.invalid.map((invalid) => invalid ? 0.62 : 0.9),
            },
          });
        }
      }
  
      Promise.resolve(plotlyNewPlot(plot, traces, {
        height: plotHeight(320, 520),
        margin: { l: 62, r: 14, t: 34, b: 50 },
        font: { size: 10, color: '#334155' },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        title: { text: `${group.label} profile`, font: { size: 12 } },
        xaxis: {
          title: { text: 'Inline distance (m)' },
          zeroline: false,
          gridcolor: '#e5e7eb',
        },
        yaxis: {
          title: { text: profileAxisTitle(group, available) },
          zeroline: group.unit === 'ms' || group.unit === 'mixed',
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
      }, { displayModeBar: false, responsive: true })).then(() => attachProfileEndpointClickActions(plot));
    } else {
      plot.textContent = plotlyUnavailableMessage();
    }
  
    if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
      content.appendChild(createTable(view));
    }
  }

  renderProfilePlot(root, bundle, viewDef);
}
