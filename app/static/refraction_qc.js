(function () {
  const RECENT_JOBS_KEY = 'sv.refraction_qc.recent_jobs';
  const MAX_RECENT_JOBS = 8;
  const DEFAULT_MAX_POINTS = 20000;

  const VIEW_DEFS = [
    {
      id: 'summary',
      include: 'summary',
      viewKeys: [],
      unavailableKeys: ['summary'],
    },
    {
      id: 'first_break_residuals',
      include: 'first_break',
      viewKeys: ['first_break_residual', 'first_break_fit'],
      unavailableKeys: ['first_break'],
    },
    {
      id: 'reduced_time',
      include: 'reduced_time',
      viewKeys: ['reduced_time'],
      unavailableKeys: ['reduced_time'],
    },
    {
      id: 'profiles_2d',
      include: 'profiles',
      viewKeys: ['line_profiles'],
      unavailableKeys: ['profiles'],
    },
    {
      id: 'cell_maps_3d',
      include: 'cells',
      viewKeys: ['refractor_cells', 'v3_refractor_cells', 'vsub_refractor_cells'],
      unavailableKeys: ['cells'],
    },
    {
      id: 'static_components',
      include: 'static_components',
      viewKeys: ['static_components'],
      unavailableKeys: ['static_components'],
    },
    {
      id: 'gather_preview',
      include: 'gather_preview',
      viewKeys: [],
      unavailableKeys: ['gather_preview'],
    },
  ];

  const INCLUDE_ALL = Array.from(new Set(VIEW_DEFS.map((view) => view.include)));

  const state = {
    selectedJobId: '',
    qcBundle: null,
    selectedView: 'summary',
    selectedLayerKind: 'v2_t1',
    selectedEndpointKind: 'source',
    selectedCell: null,
    selectedEndpoint: '',
    maxPoints: DEFAULT_MAX_POINTS,
    error: null,
    loading: false,
  };

  let dom = null;
  let requestSerial = 0;

  function readRecentJobs() {
    try {
      const raw = localStorage.getItem(RECENT_JOBS_KEY);
      const parsed = JSON.parse(raw || '[]');
      if (!Array.isArray(parsed)) return [];
      return parsed.filter((item) => typeof item === 'string' && item).slice(0, MAX_RECENT_JOBS);
    } catch (_) {
      return [];
    }
  }

  function writeRecentJob(jobId) {
    const cleanJobId = String(jobId || '').trim();
    if (!cleanJobId) return;
    const recent = readRecentJobs().filter((item) => item !== cleanJobId);
    recent.unshift(cleanJobId);
    try {
      localStorage.setItem(RECENT_JOBS_KEY, JSON.stringify(recent.slice(0, MAX_RECENT_JOBS)));
    } catch (_) {
    }
    renderRecentJobs();
  }

  function parsePositiveInteger(value, fallback) {
    const parsed = Number.parseInt(String(value), 10);
    if (!Number.isFinite(parsed) || parsed < 1) return fallback;
    return parsed;
  }

  function parseCell(value) {
    const text = String(value || '').trim();
    if (!text) return null;
    const parts = text.split(/[,\s]+/).filter(Boolean);
    if (parts.length !== 2) return { text };
    const ix = Number.parseInt(parts[0], 10);
    const iy = Number.parseInt(parts[1], 10);
    if (!Number.isInteger(ix) || !Number.isInteger(iy) || ix < 0 || iy < 0) {
      return { text };
    }
    return { cell_ix: ix, cell_iy: iy };
  }

  function textOrDash(value) {
    if (value === null || value === undefined || value === '') return '-';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  }

  function clearNode(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
  }

  function appendText(parent, text) {
    parent.appendChild(document.createTextNode(text));
  }

  function createKv(items) {
    const list = document.createElement('dl');
    list.className = 'refraction-qc-kv';
    for (const [label, value] of items) {
      const dt = document.createElement('dt');
      dt.textContent = label;
      const dd = document.createElement('dd');
      dd.textContent = textOrDash(value);
      list.append(dt, dd);
    }
    return list;
  }

  function createTable(view) {
    const wrap = document.createElement('div');
    wrap.className = 'refraction-qc-table-wrap';
    const table = document.createElement('table');
    table.className = 'refraction-qc-table';

    const columns = Array.isArray(view.columns) ? view.columns.slice(0, 5) : [];
    const head = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const column of columns) {
      const th = document.createElement('th');
      th.textContent = column;
      headRow.appendChild(th);
    }
    head.appendChild(headRow);
    table.appendChild(head);

    const body = document.createElement('tbody');
    const records = Array.isArray(view.records) ? view.records.slice(0, 5) : [];
    for (const record of records) {
      const row = document.createElement('tr');
      for (const column of columns) {
        const td = document.createElement('td');
        td.textContent = textOrDash(record ? record[column] : '');
        row.appendChild(td);
      }
      body.appendChild(row);
    }
    table.appendChild(body);
    wrap.appendChild(table);
    return wrap;
  }

  function findViewData(bundle, viewDef) {
    const views = bundle && typeof bundle.views === 'object' && bundle.views ? bundle.views : {};
    for (const key of viewDef.viewKeys) {
      if (views[key]) return { key, view: views[key] };
    }
    return null;
  }

  function isUnavailable(bundle, viewDef) {
    const unavailable = Array.isArray(bundle?.unavailable_views) ? bundle.unavailable_views : [];
    return viewDef.unavailableKeys.some((key) => unavailable.includes(key));
  }

  function renderSummary(content, bundle) {
    const summary = bundle && typeof bundle.summary === 'object' && bundle.summary ? bundle.summary : {};
    content.appendChild(createKv([
      ['Job ID', bundle.job_id],
      ['State', summary.job_state || summary.status],
      ['Workflow', summary.workflow],
      ['Method', summary.method],
      ['Conversion', summary.conversion_mode],
      ['Layer count', summary.layer_count],
      ['Coordinate mode', bundle.coordinate_mode],
      ['Available views', Array.isArray(bundle.available_views) ? bundle.available_views.join(', ') : ''],
      ['Unavailable views', Array.isArray(bundle.unavailable_views) ? bundle.unavailable_views.join(', ') : ''],
    ]));
  }

  function renderTabular(content, bundle, viewDef) {
    const found = findViewData(bundle, viewDef);
    if (!found) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = isUnavailable(bundle, viewDef)
        ? 'This view is unavailable from the loaded QC bundle artifacts.'
        : 'No sampled records are present for this view.';
      content.appendChild(missing);
      return;
    }

    const { key, view } = found;
    content.appendChild(createKv([
      ['Bundle view', key],
      ['Artifact', view.artifact],
      ['Rows', `${view.returned_points || 0} of ${view.total_points || 0}`],
      ['Downsampled', view.downsampled ? 'yes' : 'no'],
      ['Columns', Array.isArray(view.columns) ? view.columns.join(', ') : ''],
    ]));
    if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
      content.appendChild(createTable(view));
    }
  }

  function renderGatherPreview(content, bundle) {
    const message = document.createElement('p');
    message.className = 'refraction-qc-placeholder';
    if (isUnavailable(bundle, VIEW_DEFS[6])) {
      message.textContent = 'Gather preview is not included in the compact QC bundle.';
    } else {
      message.textContent = 'No gather preview payload loaded.';
    }
    content.appendChild(message);
    content.appendChild(createKv([
      ['Endpoint kind', state.selectedEndpointKind],
      ['Endpoint', state.selectedEndpoint],
      ['Cell', state.selectedCell ? textOrDash(state.selectedCell) : '-'],
      ['Layer', state.selectedLayerKind],
    ]));
  }

  function renderViewContent(viewDef) {
    if (!dom) return;
    const content = dom.viewContents.get(viewDef.id);
    if (!content) return;
    clearNode(content);
    content.className = 'refraction-qc-placeholder';

    if (state.loading) {
      appendText(content, 'Loading QC bundle...');
      return;
    }
    if (!state.qcBundle) {
      appendText(content, state.error ? 'QC bundle is not loaded.' : 'No QC bundle loaded.');
      return;
    }

    content.className = '';
    if (viewDef.id === 'summary') {
      renderSummary(content, state.qcBundle);
    } else if (viewDef.id === 'gather_preview') {
      renderGatherPreview(content, state.qcBundle);
    } else {
      renderTabular(content, state.qcBundle, viewDef);
    }
  }

  function renderRecentJobs() {
    if (!dom?.jobList) return;
    clearNode(dom.jobList);
    for (const jobId of readRecentJobs()) {
      const option = document.createElement('option');
      option.value = jobId;
      dom.jobList.appendChild(option);
    }
  }

  function render() {
    if (!dom) return;

    dom.loadButton.disabled = state.loading;
    dom.maxPoints.value = String(state.maxPoints);
    dom.layerKind.value = state.selectedLayerKind;
    dom.endpointKind.value = state.selectedEndpointKind;
    dom.endpoint.value = state.selectedEndpoint;
    dom.cell.value = state.selectedCell?.cell_ix !== undefined
      ? `${state.selectedCell.cell_ix},${state.selectedCell.cell_iy}`
      : (state.selectedCell?.text || '');

    if (state.loading) {
      dom.status.textContent = 'Loading QC bundle...';
    } else if (state.qcBundle) {
      const available = Array.isArray(state.qcBundle.available_views)
        ? state.qcBundle.available_views.length
        : 0;
      const unavailable = Array.isArray(state.qcBundle.unavailable_views)
        ? state.qcBundle.unavailable_views.length
        : 0;
      dom.status.textContent = `Loaded ${state.qcBundle.job_id}: ${available} available, ${unavailable} unavailable.`;
    } else {
      dom.status.textContent = 'No QC bundle loaded.';
    }

    dom.error.hidden = !state.error;
    dom.error.textContent = state.error || '';

    const signConvention = state.qcBundle?.sign_convention;
    dom.sign.hidden = !signConvention;
    dom.sign.textContent = signConvention ? `Sign: ${signConvention}` : '';

    for (const button of dom.viewButtons) {
      const active = button.dataset.view === state.selectedView;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-selected', active ? 'true' : 'false');
    }
    for (const panel of dom.viewPanels) {
      panel.hidden = panel.dataset.viewPanel !== state.selectedView;
    }
    for (const viewDef of VIEW_DEFS) {
      renderViewContent(viewDef);
    }
  }

  function setSelectedView(viewId) {
    if (!VIEW_DEFS.some((view) => view.id === viewId)) return;
    state.selectedView = viewId;
    render();
  }

  function activateSidebarTab(tabName) {
    if (!dom) return;
    const showQc = tabName === 'refraction_qc';
    dom.pipelineTab.classList.toggle('is-active', !showQc);
    dom.qcTab.classList.toggle('is-active', showQc);
    dom.pipelineTab.setAttribute('aria-selected', showQc ? 'false' : 'true');
    dom.qcTab.setAttribute('aria-selected', showQc ? 'true' : 'false');
    dom.pipelinePanel.hidden = showQc;
    dom.qcPanel.hidden = !showQc;
  }

  async function readError(response) {
    try {
      const payload = await response.json();
      if (payload && typeof payload.detail === 'string') return payload.detail;
      if (payload && payload.detail) return JSON.stringify(payload.detail);
    } catch (_) {
    }
    try {
      const text = await response.text();
      if (text) return text;
    } catch (_) {
    }
    return `QC bundle request failed with status ${response.status}`;
  }

  async function loadBundle() {
    if (!dom) return;
    const jobId = String(dom.jobId.value || '').trim();
    const maxPoints = parsePositiveInteger(dom.maxPoints.value, DEFAULT_MAX_POINTS);
    state.selectedJobId = jobId;
    state.maxPoints = maxPoints;
    if (!jobId) {
      state.error = 'Job ID is required.';
      state.qcBundle = null;
      render();
      return;
    }

    const serial = ++requestSerial;
    state.loading = true;
    state.error = null;
    render();

    try {
      const response = await fetch('/statics/refraction/qc', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          include: INCLUDE_ALL,
          max_points: maxPoints,
          coordinate_mode: 'auto',
        }),
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const bundle = await response.json();
      if (serial !== requestSerial) return;
      state.qcBundle = bundle;
      state.error = null;
      writeRecentJob(jobId);
    } catch (error) {
      if (serial !== requestSerial) return;
      state.qcBundle = null;
      state.error = error instanceof Error ? error.message : String(error);
    } finally {
      if (serial === requestSerial) {
        state.loading = false;
        render();
      }
    }
  }

  function init() {
    const pipelineTab = document.getElementById('pipelineSidebarTab');
    const qcTab = document.getElementById('refractionQcSidebarTab');
    const pipelinePanel = document.getElementById('pipelineTabPanel');
    const qcPanel = document.getElementById('refractionQcTabPanel');
    const form = document.getElementById('refractionQcForm');
    if (!pipelineTab || !qcTab || !pipelinePanel || !qcPanel || !form) return;

    dom = {
      pipelineTab,
      qcTab,
      pipelinePanel,
      qcPanel,
      form,
      jobId: document.getElementById('refractionQcJobId'),
      jobList: document.getElementById('refractionQcJobList'),
      maxPoints: document.getElementById('refractionQcMaxPoints'),
      loadButton: document.getElementById('refractionQcLoadButton'),
      status: document.getElementById('refractionQcStatus'),
      error: document.getElementById('refractionQcError'),
      sign: document.getElementById('refractionQcSign'),
      layerKind: document.getElementById('refractionQcLayerKind'),
      endpointKind: document.getElementById('refractionQcEndpointKind'),
      endpoint: document.getElementById('refractionQcEndpoint'),
      cell: document.getElementById('refractionQcCell'),
      viewButtons: Array.from(document.querySelectorAll('.refraction-qc-view-button')),
      viewPanels: Array.from(document.querySelectorAll('.refraction-qc-view')),
      viewContents: new Map(Array.from(document.querySelectorAll('[data-view-content]')).map(
        (node) => [node.dataset.viewContent, node],
      )),
    };

    if (!dom.jobId || !dom.maxPoints || !dom.loadButton || !dom.status || !dom.error || !dom.sign) return;
    if (!dom.layerKind || !dom.endpointKind || !dom.endpoint || !dom.cell) return;

    pipelineTab.addEventListener('click', () => activateSidebarTab('pipeline'));
    qcTab.addEventListener('click', () => activateSidebarTab('refraction_qc'));
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      loadBundle();
    });
    dom.jobId.addEventListener('input', () => {
      state.selectedJobId = dom.jobId.value.trim();
    });
    dom.maxPoints.addEventListener('input', () => {
      state.maxPoints = parsePositiveInteger(dom.maxPoints.value, DEFAULT_MAX_POINTS);
    });
    dom.layerKind.addEventListener('change', () => {
      state.selectedLayerKind = dom.layerKind.value;
      render();
    });
    dom.endpointKind.addEventListener('change', () => {
      state.selectedEndpointKind = dom.endpointKind.value;
      render();
    });
    dom.endpoint.addEventListener('input', () => {
      state.selectedEndpoint = dom.endpoint.value.trim();
      render();
    });
    dom.cell.addEventListener('input', () => {
      state.selectedCell = parseCell(dom.cell.value);
      render();
    });
    for (const button of dom.viewButtons) {
      button.addEventListener('click', () => setSelectedView(button.dataset.view));
    }

    const params = new URLSearchParams(window.location.search || '');
    const jobId = params.get('refraction_job_id') || params.get('refraction_qc_job_id') || '';
    if (jobId) {
      state.selectedJobId = jobId;
      dom.jobId.value = jobId;
    }

    renderRecentJobs();
    render();
  }

  window.refractionQcState = state;
  window.refractionQcUI = {
    loadBundle,
    setSelectedView,
    activateSidebarTab,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
