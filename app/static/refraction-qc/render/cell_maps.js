export function renderCellMapsView({
  root,
  bundle,
  viewDef,
  viewState,
  buildCellMapMatrix,
  cellDrilldownTargetFromSelectedCell,
  cellEndpointDisplayName,
  cellMapHoverText,
  cellMapQuantityDefinition,
  cellMapQuantityValue,
  cellStatusCode,
  cellStatusColor,
  createEndpointActionButton,
  createFirstBreakPlot,
  createKv,
  createTable,
  domRef,
  drilldownMetric,
  drilldownResidualMs,
  filteredCellMapRecords,
  findDownsampling,
  findViewData,
  firstDefined,
  formatNumber,
  getPlotly,
  isUnavailable,
  layerLabel,
  loadQcDrilldown,
  plotHeight,
  plotlyNewPlot,
  plotlyUnavailableMessage,
  previewGatherForEndpoint,
  render,
  selectedCellLabel,
  selectedCellMapLayer,
  selectedCellMapPoint,
  statusCounts,
  textOrDash,
  toFiniteNumber,
}) {
  const state = viewState;
  const dom = domRef();

  function endpointCandidatesFromCellDrilldown(payload) {
    const records = Array.isArray(payload?.observations?.records)
      ? payload.observations.records
      : [];
    const byEndpoint = new Map();
    for (const record of records) {
      for (const endpointKind of ['source', 'receiver']) {
        const endpointKey = String(firstDefined(record, [`${endpointKind}_endpoint_key`, `${endpointKind}_id`]) || '').trim();
        if (!endpointKey) continue;
        const key = `${endpointKind}|${endpointKey}`;
        if (!byEndpoint.has(key)) {
          byEndpoint.set(key, {
            endpointKind,
            endpointKey,
            pickCount: 0,
            residualSquares: [],
          });
        }
        const candidate = byEndpoint.get(key);
        candidate.pickCount += 1;
        const residualMs = drilldownResidualMs(record);
        if (Number.isFinite(residualMs)) candidate.residualSquares.push(residualMs * residualMs);
      }
    }
    return Array.from(byEndpoint.values())
      .map((candidate) => ({
        endpointKind: candidate.endpointKind,
        endpointKey: candidate.endpointKey,
        pickCount: candidate.pickCount,
        residualRmsMs: candidate.residualSquares.length
          ? Math.sqrt(candidate.residualSquares.reduce((total, value) => total + value, 0) / candidate.residualSquares.length)
          : NaN,
      }))
      .sort((a, b) => (
        a.endpointKind.localeCompare(b.endpointKind)
        || b.pickCount - a.pickCount
        || a.endpointKey.localeCompare(b.endpointKey)
      ));
  }

  function createCellEndpointCandidateRow(candidate) {
    const item = document.createElement('li');
    const summary = document.createElement('span');
    const rms = Number.isFinite(candidate.residualRmsMs)
      ? ` · RMS ${formatNumber(candidate.residualRmsMs, 1)} ms`
      : '';
    summary.textContent = `${cellEndpointDisplayName(
      candidate.endpointKind,
      candidate.endpointKey,
    )} · picks ${candidate.pickCount}${rms}`;
    item.appendChild(summary);

    const action = createEndpointActionButton(
      candidate.endpointKind === 'receiver' ? 'Preview receiver gather' : 'Preview source gather',
      '',
      () => previewGatherForEndpoint(candidate.endpointKind, candidate.endpointKey),
      candidate.endpointKind === 'receiver'
        ? 'refraction-qc-cell-preview-receiver'
        : 'refraction-qc-cell-preview-source',
    );
    item.appendChild(action);
    return item;
  }

  function appendCellDrilldownObservations(panel, observations) {
    const records = Array.isArray(observations?.records) ? observations.records.slice(0, 8) : [];
    const title = document.createElement('h4');
    title.textContent = 'Contributing picks';
    panel.appendChild(title);

    if (!records.length) {
      const empty = document.createElement('p');
      empty.className = 'refraction-qc-placeholder';
      empty.textContent = 'No contributing pick records were returned for this cell.';
      panel.appendChild(empty);
      return;
    }

    const list = document.createElement('ul');
    list.className = 'refraction-qc-cell-drilldown-picks';
    list.dataset.testid = 'refraction-qc-cell-drilldown-picks';
    for (const record of records) {
      const item = document.createElement('li');
      const trace = textOrDash(firstDefined(record, ['trace_index_sorted', 'sorted_trace_index', 'trace_index', 'trace']));
      const source = textOrDash(firstDefined(record, ['source_endpoint_key', 'source_id']));
      const receiver = textOrDash(firstDefined(record, ['receiver_endpoint_key', 'receiver_id']));
      const residual = drilldownResidualMs(record);
      const residualText = Number.isFinite(residual)
        ? ` · residual ${residual >= 0 ? '+' : ''}${formatNumber(residual, 1)} ms`
        : '';
      item.textContent = `trace ${trace} · source ${source} · receiver ${receiver}${residualText}`;
      list.appendChild(item);
    }
    panel.appendChild(list);
  }

  function renderCellDrilldownPanel(content, payload) {
    const panel = document.createElement('section');
    panel.className = 'refraction-qc-cell-drilldown';
    panel.dataset.testid = 'refraction-qc-cell-drilldown';

    const title = document.createElement('h3');
    title.textContent = 'Cell drilldown';
    panel.appendChild(title);

    if (!state.selectedCell || state.selectedCell.cell_ix === undefined) {
      const empty = document.createElement('p');
      empty.className = 'refraction-qc-placeholder';
      empty.textContent = 'Click a cell to load contributing endpoints and picks.';
      panel.appendChild(empty);
      content.appendChild(panel);
      return;
    }

    if (state.qcDrilldownLoading) {
      const loading = document.createElement('p');
      loading.className = 'refraction-qc-note';
      loading.textContent = 'Loading cell drilldown...';
      panel.appendChild(loading);
    }

    if (state.qcDrilldownError) {
      const error = document.createElement('p');
      error.className = 'refraction-qc-error';
      error.dataset.testid = 'refraction-qc-cell-drilldown-error';
      error.textContent = state.qcDrilldownError;
      panel.appendChild(error);
    }

    if (!payload) {
      content.appendChild(panel);
      return;
    }

    const cell = payload.cell || {};
    const velocity = payload.velocity || cell.velocity || {};
    const fold = payload.fold || cell.fold || {};
    const residual = payload.residual_summary || cell.residual_summary || {};
    const endpointCounts = payload.endpoint_counts || cell.endpoint_counts || {};
    const observations = payload.observations || {};
    const row = cell.row || {};
    const layerKind = cell.layer_kind || payload.target?.layer_kind || state.qcDrilldownTarget?.layer_kind;
    const cellIx = firstDefined(cell, ['cell_ix']) ?? payload.target?.cell_ix ?? state.qcDrilldownTarget?.cell_ix;
    const cellIy = firstDefined(cell, ['cell_iy']) ?? payload.target?.cell_iy ?? state.qcDrilldownTarget?.cell_iy;
    const residualRms = firstDefined(residual, ['cell_residual_rms_ms', 'used_rms_ms', 'all_rms_ms']);

    panel.appendChild(createKv([
      ['Layer', layerLabel(layerKind)],
      ['Cell', `ix ${textOrDash(cellIx)}, iy ${textOrDash(cellIy)}`],
      ['Velocity', drilldownMetric(firstDefined(velocity, ['velocity_m_s', 'v2_m_s']), 2, 'm/s')],
      ['Fold', firstDefined(fold, ['n_observations', 'fold', 'observation_count'])],
      ['Used fold', firstDefined(fold, ['n_used_observations', 'used_fold'])],
      ['Residual RMS', drilldownMetric(residualRms, 1, 'ms')],
      ['Status', firstDefined(velocity, ['velocity_status', 'status']) || firstDefined(row, ['velocity_status', 'status'])],
      ['Contributing observations', `${observations.returned_count ?? 0} of ${observations.total_count ?? 0}`],
      ['Source endpoints', endpointCounts.source_count],
      ['Receiver endpoints', endpointCounts.receiver_count],
    ]));

    if (observations.capped) {
      const capped = document.createElement('p');
      capped.className = 'refraction-qc-note';
      capped.dataset.testid = 'refraction-qc-cell-drilldown-capped';
      capped.textContent = `Observation records are capped at ${observations.returned_count} of ${observations.total_count} by max_observations.`;
      panel.appendChild(capped);
    }

    const endpointTitle = document.createElement('h4');
    endpointTitle.textContent = 'Contributing endpoints';
    panel.appendChild(endpointTitle);
    const candidates = endpointCandidatesFromCellDrilldown(payload);
    if (candidates.length) {
      const list = document.createElement('ul');
      list.className = 'refraction-qc-cell-drilldown-endpoints';
      list.dataset.testid = 'refraction-qc-cell-drilldown-endpoints';
      for (const candidate of candidates) {
        list.appendChild(createCellEndpointCandidateRow(candidate));
      }
      panel.appendChild(list);
    } else {
      const empty = document.createElement('p');
      empty.className = 'refraction-qc-placeholder';
      empty.textContent = 'No source or receiver endpoint keys were returned for this cell.';
      panel.appendChild(empty);
    }

    appendCellDrilldownObservations(panel, observations);
    content.appendChild(panel);
  }

  function renderCellMapPlot(content, bundle, viewDef) {
    const found = findViewData(bundle, viewDef);
    if (!found) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = isUnavailable(bundle, viewDef)
        ? '3D cell maps are unavailable for global-velocity jobs or QC bundles without refraction_grid_map_qc.csv.'
        : 'No refraction_grid_map_qc rows are present for this QC bundle.';
      content.appendChild(missing);
      return;
    }
  
    const { key, view } = found;
    const records = filteredCellMapRecords(view);
    const layerKind = selectedCellMapLayer(records);
    const layerRecords = records.filter((record) => record.layerKind === layerKind);
    const quantity = cellMapQuantityDefinition();
    const downsampling = findDownsampling(bundle, key, view);
    const downsamplingText = downsampling
      ? `${downsampling.returned_points || 0} of ${downsampling.total_points || 0}; ${downsampling.downsampled ? 'downsampled' : 'not downsampled'}${downsampling.method ? ` (${downsampling.method})` : ''}`
      : 'not reported';
    const statusSummary = statusCounts(layerRecords, 'status') || 'none';
    const plottedValues = layerRecords
      .map((point) => cellMapQuantityValue(point, quantity))
      .filter((value) => Number.isFinite(value)).length;
  
    content.appendChild(createKv([
      ['Bundle view', key],
      ['Artifact', view.artifact],
      ['Rows', `${view.returned_points || 0} of ${view.total_points || 0}`],
      ['Coordinate mode', bundle.coordinate_mode],
      ['Layer filter', state.selectedLayerKind === 'all' ? 'all' : layerLabel(state.selectedLayerKind)],
      ['Plotted layer', layerKind ? layerLabel(layerKind) : '-'],
      ['Quantity', quantity.label],
      ['Cells', `${layerRecords.length}`],
      ['Plotted values', `${plottedValues}`],
      ['Status counts', statusSummary],
      ['Selected cell', selectedCellLabel(state.selectedCell)],
    ]));
  
    const downsamplingNote = document.createElement('p');
    downsamplingNote.className = 'refraction-qc-note';
    downsamplingNote.textContent = `Downsampling: ${downsamplingText}`;
    downsamplingNote.dataset.testid = 'refraction-qc-cell-map-downsampling';
    content.appendChild(downsamplingNote);
    renderCellDrilldownPanel(content, state.qcDrilldown);
  
    if (!records.length || !layerRecords.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = 'No grid-map cells match the current layer selection.';
      content.appendChild(missing);
      if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
        content.appendChild(createTable(view));
      }
      return;
    }
  
    const plot = createFirstBreakPlot('refraction-qc-cell-map-plot');
    plot.classList.add('refraction-qc-cell-map-plot');
    plot.dataset.pointCount = String(layerRecords.length);
    content.appendChild(plot);
  
    if (getPlotly()) {
      const matrix = buildCellMapMatrix(layerRecords, quantity);
      const traces = [
        {
          name: quantity.label,
          type: 'heatmap',
          x: matrix.x,
          y: matrix.y,
          z: matrix.z,
          text: matrix.text,
          customdata: matrix.customdata,
          hovertemplate: '%{text}<extra></extra>',
          colorscale: quantity.status
            ? [
              [0, '#dc2626'],
              [0.249, '#dc2626'],
              [0.25, '#cbd5e1'],
              [0.499, '#cbd5e1'],
              [0.5, '#f59e0b'],
              [0.749, '#f59e0b'],
              [0.75, '#2563eb'],
              [1, '#2563eb'],
            ]
            : quantity.colorscale,
          reversescale: Boolean(quantity.reversescale),
          zmin: quantity.status ? 0 : undefined,
          zmax: quantity.status ? 3 : undefined,
          colorbar: quantity.status
            ? {
              title: { text: 'Status' },
              tickmode: 'array',
              tickvals: [0, 1, 2, 3],
              ticktext: ['other', 'inactive', 'low_fold', 'solved'],
            }
            : { title: { text: quantity.axisLabel } },
        },
      ];
  
      const flagged = layerRecords.filter((point) => cellStatusCode(point.status) < 3);
      if (flagged.length) {
        traces.push({
          name: 'Flagged cells',
          type: 'scatter',
          mode: 'markers',
          x: flagged.map((point) => point.centerX),
          y: flagged.map((point) => point.centerY),
          text: flagged.map((point) => cellMapHoverText(
            point,
            quantity,
            cellMapQuantityValue(point, quantity),
          )),
          customdata: flagged.map((point) => ({
            cell_ix: point.cellIx,
            cell_iy: point.cellIy,
            layer_kind: point.layerKind,
            velocity_m_s: point.velocity,
            fold: point.fold,
            residual_rms_ms: point.residualRmsMs,
            status: point.status,
          })),
          hovertemplate: '%{text}<extra>Flagged</extra>',
          marker: {
            color: flagged.map((point) => cellStatusColor(point.status)),
            symbol: flagged.map((point) => point.status === 'low_fold' ? 'diamond-open' : 'x'),
            size: flagged.map((point) => point.status === 'low_fold' ? 12 : 10),
            line: { color: '#0f172a', width: 1 },
          },
        });
      }
  
      const selected = selectedCellMapPoint(layerRecords, layerKind);
      if (selected) {
        traces.push({
          name: 'Selected cell',
          type: 'scatter',
          mode: 'markers',
          x: [selected.centerX],
          y: [selected.centerY],
          text: [cellMapHoverText(selected, quantity, cellMapQuantityValue(selected, quantity))],
          customdata: [{
            cell_ix: selected.cellIx,
            cell_iy: selected.cellIy,
            layer_kind: selected.layerKind,
            velocity_m_s: selected.velocity,
            fold: selected.fold,
            residual_rms_ms: selected.residualRmsMs,
            status: selected.status,
          }],
          hovertemplate: '%{text}<extra>Selected</extra>',
          marker: {
            color: '#0f172a',
            symbol: 'square-open',
            size: 16,
            line: { color: '#0f172a', width: 2 },
          },
        });
      }
  
      plotlyNewPlot(plot, traces, {
        height: plotHeight(340, 560),
        margin: { l: 62, r: 14, t: 42, b: 56 },
        font: { size: 10, color: '#334155' },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        title: { text: `${layerLabel(layerKind)} ${quantity.label} map`, font: { size: 12 } },
        xaxis: {
          title: { text: 'Cell center X (m)' },
          zeroline: false,
          gridcolor: '#e5e7eb',
        },
        yaxis: {
          title: { text: 'Cell center Y (m)' },
          zeroline: false,
          gridcolor: '#e5e7eb',
          scaleanchor: matrix.x.length > 1 && matrix.y.length > 1 ? 'x' : undefined,
        },
        legend: {
          orientation: 'h',
          x: 0,
          y: 1.14,
          xanchor: 'left',
          yanchor: 'top',
          font: { size: 10 },
        },
      }, { displayModeBar: false, responsive: true }).then(() => {
        plot.on('plotly_click', (event) => {
          const point = event?.points?.[0];
          const cell = point?.customdata;
          if (!cell || cell.cell_ix === undefined || cell.cell_iy === undefined) return;
          state.selectedCell = {
            cell_ix: Number(cell.cell_ix),
            cell_iy: Number(cell.cell_iy),
            layer_kind: String(cell.layer_kind || layerKind),
            velocity_m_s: toFiniteNumber(cell.velocity_m_s),
            fold: toFiniteNumber(cell.fold),
            residual_rms_ms: toFiniteNumber(cell.residual_rms_ms),
            status: textOrDash(cell.status),
          };
          state.selectedObject = {
            kind: 'cell',
            key: selectedCellLabel(state.selectedCell),
            payload: state.selectedCell,
          };
          if (dom?.cell) dom.cell.value = `${state.selectedCell.cell_ix},${state.selectedCell.cell_iy}`;
          const drilldownTarget = cellDrilldownTargetFromSelectedCell(state.selectedCell);
          render();
          loadQcDrilldown(drilldownTarget);
        });
      });
    } else {
      plot.textContent = plotlyUnavailableMessage();
    }
  
    if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
      content.appendChild(createTable(view));
    }
  }

  renderCellMapPlot(root, bundle, viewDef);
}
