(function () {
  // === smoke log ===
  console.info('[pipeline] pipeline_ui.js loaded');

  const DEBUG_PIPELINE = true;
  const VALID_STEP_NAMES = new Set(['bandpass', 'denoise']);

  const pipelineState = {
    graph: [],
    desiredLayer: 'raw',
    inspectorStepId: null,
    cardsContainer: null,
    inspectorEl: null,
    inspectorForm: null,
    inspectorFieldsEl: null,
    inspectorTitleEl: null,
    addMenuEl: null,
    addButtonEl: null,
    draggedId: null,
    runToken: 0,
  };

  const PARAM_DEFS = {
    bandpass: [
      { key: 'low_hz', label: 'low_hz', type: 'number', step: '0.1' },
      { key: 'high_hz', label: 'high_hz', type: 'number', step: '0.1' },
      { key: 'dt', label: 'dt', type: 'number', step: '0.0001' },
      { key: 'taper', label: 'taper', type: 'number', step: '0.05' },
    ],
    denoise: [
      { key: 'chunk_h', label: 'chunk_h', type: 'number', step: '1', min: 1, parser: (v) => parseInt(v, 10) },
      { key: 'overlap', label: 'overlap', type: 'number', step: '1', min: 0, parser: (v) => parseInt(v, 10) },
      { key: 'mask_ratio', label: 'mask_ratio', type: 'number', step: '0.05', min: 0, max: 1 },
      { key: 'noise_std', label: 'noise_std', type: 'number', step: '0.1' },
      {
        key: 'mask_noise_mode',
        label: 'mask_noise_mode',
        type: 'select',
        options: [
          { value: 'replace', label: 'replace' },
          { value: 'add', label: 'add' },
        ],
      },
      { key: 'passes_batch', label: 'passes_batch', type: 'number', step: '1', min: 1, parser: (v) => parseInt(v, 10) },
    ],
  };

  function debounce(fn, delay) {
    let timer = null;
    return function debounced(...args) {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = null;
        fn.apply(this, args);
      }, delay);
    };
  }

  function formatStepName(name) {
    if (!name) return '';
    return name.charAt(0).toUpperCase() + name.slice(1);
  }

  function getDtFromUI() {
    const input = document.getElementById('dt');
    const candidate = input ? parseFloat(input.value) : NaN;
    if (!Number.isNaN(candidate) && isFinite(candidate) && candidate > 0) return candidate;
    if (typeof window.defaultDt === 'number' && isFinite(window.defaultDt)) return window.defaultDt;
    return 0.002;
  }

  function defaultParamsFor(name) {
    if (name === 'bandpass') {
      return { low_hz: 5, high_hz: 60, dt: getDtFromUI(), taper: 0.1 };
    }
    if (name === 'denoise') {
      return {
        chunk_h: 128,
        overlap: 32,
        mask_ratio: 0.5,
        noise_std: 1.0,
        mask_noise_mode: 'replace',
        passes_batch: 4,
      };
    }
    return {};
  }

  function applyDefaultParams(name, params) {
    const base = defaultParamsFor(name);
    const incoming = params && typeof params === 'object' ? params : {};
    return { ...base, ...incoming };
  }

  function labelOf(step) {
    return (step.label && step.label.trim()) || step.name;
  }

  function generateId() {
    return `step_${Math.random().toString(36).slice(2, 9)}_${Date.now().toString(36)}`;
  }

  function normalizeStep(raw) {
    if (!raw || typeof raw !== 'object') return null;
    if (!VALID_STEP_NAMES.has(raw.name)) return null;
    return {
      id: typeof raw.id === 'string' ? raw.id : generateId(),
      kind: raw.kind === 'analyzer' ? raw.kind : 'transform',
      name: raw.name,
      enabled: raw.enabled !== false,
      tap: !!raw.tap,
      label: typeof raw.label === 'string' ? raw.label : raw.name,
      params: applyDefaultParams(raw.name, raw.params),
    };
  }

  function createStep(name) {
    const normalized = name === 'bandpass' ? 'bandpass' : name === 'denoise' ? 'denoise' : null;
    if (!normalized) return null;
    return {
      id: generateId(),
      kind: 'transform',
      name: normalized,
      enabled: true,
      tap: false,
      label: normalized,
      params: defaultParamsFor(normalized),
    };
  }

  // taps は「ラベルの累積」＋ 'final' を返す
  function graphToSpec(graph) {
    const steps = [];
    const enabled = graph.filter((s) => s && s.enabled);

    for (const step of enabled) {
      steps.push({
        kind: step.kind || 'transform',
        name: step.name,
        params: { ...(step.params || {}) },
        label: labelOf(step),
      });
    }

    const taps = [];
    if (enabled.length > 0) {
      const path = [];
      for (const step of enabled) {
        path.push(labelOf(step));
        if (step.tap) taps.push(path.join('+')); // 例: "bandpass+denoise"
      }
      taps.push('final');
    }
    return { spec: { steps }, taps };
  }

  function specToGraph(spec, taps) {
    if (!spec || typeof spec !== 'object' || !Array.isArray(spec.steps)) return [];
    const tapSet = new Set(Array.isArray(taps) ? taps : []);
    const out = [];
    for (const step of spec.steps) {
      const normalized = normalizeStep({
        ...step,
        enabled: true,
        tap: tapSet.has(step.label || step.name),
        label: step.label || step.name,
      });
      if (normalized) out.push(normalized);
    }
    return out;
  }

  function loadPipelineFromLocalStorage() {
    try {
      const stored = localStorage.getItem('viewer.pipelineGraph');
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          const steps = parsed.map(normalizeStep).filter(Boolean);
          if (steps.length) return steps;
        }
      }
    } catch (e) {
      console.warn('Failed to parse viewer.pipelineGraph', e);
    }

    try {
      const specRaw = localStorage.getItem('viewer.pipelineSpec');
      const tapsRaw = localStorage.getItem('viewer.pipelineTaps');
      if (specRaw) {
        const spec = JSON.parse(specRaw);
        const taps = tapsRaw ? JSON.parse(tapsRaw) : [];
        const steps = specToGraph(spec, taps);
        if (steps.length) return steps;
      }
    } catch (e) {
      console.warn('Failed to parse stored pipeline spec/taps', e);
    }

    return [];
  }

  function savePipelineToLocalStorage(graph) {
    try {
      localStorage.setItem('viewer.pipelineGraph', JSON.stringify(graph));
    } catch (e) {
      console.warn('Failed to persist pipeline graph', e);
    }
    try {
      const { spec, taps } = graphToSpec(graph);
      localStorage.setItem('viewer.pipelineSpec', JSON.stringify(spec));
      localStorage.setItem('viewer.pipelineTaps', JSON.stringify(taps));
    } catch (e) {
      console.warn('Failed to persist pipeline spec/taps', e);
    }
  }

  function renderPipelineCards(graph) {
    const container = pipelineState.cardsContainer;
    if (!container) return;
    container.innerHTML = '';

    if (!graph.length) {
      const empty = document.createElement('div');
      empty.className = 'pipeline-empty';
      empty.textContent = 'No steps yet. Add a step to get started.';
      container.appendChild(empty);
      return;
    }

    for (const step of graph) {
      const card = document.createElement('div');
      card.className = 'pipeline-card';
      card.dataset.id = step.id;
      card.draggable = true;
      if (!step.enabled) card.classList.add('disabled');
      if (pipelineState.inspectorStepId === step.id) card.classList.add('inspecting');

      card.addEventListener('dragstart', (event) => {
        pipelineState.draggedId = step.id;
        card.classList.add('is-dragging');
        event.dataTransfer.effectAllowed = 'move';
      });
      card.addEventListener('dragend', () => {
        card.classList.remove('is-dragging');
        pipelineState.draggedId = null;
      });

      const header = document.createElement('div');
      header.className = 'pipeline-card-header';
      const titleWrap = document.createElement('div');
      titleWrap.className = 'pipeline-card-title';
      const handle = document.createElement('span');
      handle.className = 'drag-handle';
      handle.textContent = '⋮⋮';
      titleWrap.appendChild(handle);
      const title = document.createElement('span');
      title.textContent = formatStepName(step.name);
      titleWrap.appendChild(title);
      header.appendChild(titleWrap);

      const editBtn = document.createElement('button');
      editBtn.type = 'button';
      editBtn.textContent = 'Edit';
      editBtn.addEventListener('click', () => openInspector(step.id));
      header.appendChild(editBtn);
      // Delete button
        const delBtn = document.createElement('button');
      delBtn.type = 'button';
      delBtn.textContent = 'Delete';
      delBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          if (window.confirm('Delete this step?')) {
              deleteStep(step.id);
            }
        });
      header.appendChild(delBtn);
      card.appendChild(header);

      const controls = document.createElement('div');
      controls.className = 'pipeline-card-controls';

      const enableLabel = document.createElement('label');
      const enableInput = document.createElement('input');
      enableInput.type = 'checkbox';
      enableInput.checked = step.enabled;
      enableInput.addEventListener('change', () => {
        step.enabled = enableInput.checked;
        const labelValue = labelOf(step);
        if (!step.enabled && pipelineState.desiredLayer === labelValue) {
          pipelineState.desiredLayer = 'raw';
          const sel = document.getElementById('layerSelect');
          if (sel) sel.value = 'raw';
          if (typeof drawSelectedLayer === 'function') {
            const start = typeof renderedStart === 'number' ? renderedStart : 0;
            const end = typeof renderedEnd === 'number'
              ? renderedEnd
              : (Array.isArray(rawSeismicData) ? rawSeismicData.length - 1 : 0);
            drawSelectedLayer(start, end);
          }
        }
        notifyGraphChanged();
      });
      enableLabel.appendChild(enableInput);
      enableLabel.appendChild(document.createTextNode('Enable'));
      controls.appendChild(enableLabel);

      const tapLabel = document.createElement('label');
      const tapInput = document.createElement('input');
      tapInput.type = 'checkbox';
      tapInput.checked = step.tap;
      tapInput.addEventListener('change', () => {
        step.tap = tapInput.checked;

        // この step までの累積キーを作る
        const enabled = pipelineState.graph.filter((s) => s && s.enabled);
        const idx = enabled.findIndex((s) => s.id === step.id);
        const pathKey = enabled.slice(0, idx + 1).map(labelOf).join('+'); // ex) "bandpass+denoise"

        if (tapInput.checked) {
          pipelineState.desiredLayer = pathKey;
          const sel = document.getElementById('layerSelect');
          if (sel) sel.value = pathKey;
        } else {
          if (pipelineState.desiredLayer === pathKey) {
            pipelineState.desiredLayer = 'raw';
            const sel = document.getElementById('layerSelect');
            if (sel) sel.value = 'raw';
          }
        }

        if (typeof drawSelectedLayer === 'function') {
          const start = typeof renderedStart === 'number' ? renderedStart : 0;
          const end = typeof renderedEnd === 'number'
            ? renderedEnd
            : (Array.isArray(rawSeismicData) ? rawSeismicData.length - 1 : 0);
          drawSelectedLayer(start, end);
        }
        notifyGraphChanged();
      });
      tapLabel.appendChild(tapInput);
      tapLabel.appendChild(document.createTextNode('Tap'));
      controls.appendChild(tapLabel);

      card.appendChild(controls);

      const labelWrap = document.createElement('label');
      labelWrap.className = 'pipeline-label';
      const labelTitle = document.createElement('span');
      labelTitle.textContent = 'Label';
      labelWrap.appendChild(labelTitle);
      const labelInput = document.createElement('input');
      labelInput.type = 'text';
      labelInput.className = 'pipeline-label-input';
      labelInput.value = step.label || '';
      labelInput.placeholder = step.name;
      labelInput.addEventListener('change', () => {
        const previous = labelOf(step);
        const nextLabel = labelInput.value.trim();
        step.label = nextLabel;
        if (pipelineState.desiredLayer === previous) {
          pipelineState.desiredLayer = nextLabel || step.name;
        }
        notifyGraphChanged();
      });
      labelWrap.appendChild(labelInput);
      card.appendChild(labelWrap);

      container.appendChild(card);
    }

    if (pipelineState.inspectorStepId && !graph.some((step) => step.id === pipelineState.inspectorStepId)) {
      closeInspector();
    }
  }

  function addStep(name) {
    const step = createStep(name);
    if (!step) return;
    pipelineState.graph.push(step);
    notifyGraphChanged();
  }

  function deleteStep(stepId) {
      const idx = pipelineState.graph.findIndex((s) => s.id === stepId);
      if (idx === -1) return;
      const removed = pipelineState.graph[idx];
      const removedLabel = (removed.label && removed.label.trim()) || removed.name;
      // If currently viewing this tap, fall back to raw
        if (pipelineState.desiredLayer === removedLabel) {
            pipelineState.desiredLayer = 'raw';
            const sel = document.getElementById('layerSelect');
            if (sel) sel.value = 'raw';
          }
      // Remove and close inspector if needed
        pipelineState.graph.splice(idx, 1);
      if (pipelineState.inspectorStepId === stepId) {
          pipelineState.inspectorStepId = null;
          if (pipelineState.inspectorEl) pipelineState.inspectorEl.classList.add('hidden');
          if (pipelineState.inspectorFieldsEl) pipelineState.inspectorFieldsEl.innerHTML = '';
        }
      notifyGraphChanged();
  }
  function getDragAfterElement(container, y) {
    const cards = Array.from(container.querySelectorAll('.pipeline-card:not(.is-dragging)'));
    let closest = { offset: Number.NEGATIVE_INFINITY, element: null };
    for (const card of cards) {
      const box = card.getBoundingClientRect();
      const offset = y - box.top - box.height / 2;
      if (offset < 0 && offset > closest.offset) {
        closest = { offset, element: card };
      }
    }
    return closest.element;
  }

  function handleDragOver(event) {
    if (!pipelineState.draggedId) return;
    event.preventDefault();
    const container = pipelineState.cardsContainer;
    const draggingEl = container ? container.querySelector('.pipeline-card.is-dragging') : null;
    if (!container || !draggingEl) return;
    const afterElement = getDragAfterElement(container, event.clientY);
    if (!afterElement) {
      container.appendChild(draggingEl);
    } else if (afterElement !== draggingEl) {
      container.insertBefore(draggingEl, afterElement);
    }
  }

  function handleDrop(event) {
    if (!pipelineState.draggedId) return;
    event.preventDefault();
    const container = pipelineState.cardsContainer;
    if (!container) return;
    const orderedIds = Array.from(container.querySelectorAll('.pipeline-card')).map((el) => el.dataset.id);
    const idToStep = new Map(pipelineState.graph.map((step) => [step.id, step]));
    const reordered = orderedIds.map((id) => idToStep.get(id)).filter(Boolean);
    if (reordered.length === pipelineState.graph.length) {
      pipelineState.graph = reordered;
      notifyGraphChanged();
    } else {
      renderPipelineCards(pipelineState.graph);
    }
    pipelineState.draggedId = null;
  }

  function openInspector(stepId) {
    const step = pipelineState.graph.find((s) => s.id === stepId);
    if (!step || !pipelineState.inspectorEl || !pipelineState.inspectorFieldsEl || !pipelineState.inspectorTitleEl) return;
    pipelineState.inspectorStepId = stepId;
    pipelineState.inspectorTitleEl.textContent = `${formatStepName(step.name)} Parameters`;
    pipelineState.inspectorFieldsEl.innerHTML = '';
    pipelineState.inspectorEl.classList.remove('hidden');

    const defs = PARAM_DEFS[step.name] || [];
    for (const def of defs) {
      const wrapper = document.createElement('label');
      wrapper.textContent = '';
      wrapper.className = 'inspector-field';
      const title = document.createElement('span');
      title.textContent = def.label;
      wrapper.appendChild(title);

      if (def.type === 'select') {
        const select = document.createElement('select');
        select.name = def.key;
        for (const opt of def.options || []) {
          const option = document.createElement('option');
          option.value = opt.value;
          option.textContent = opt.label;
          select.appendChild(option);
        }
        select.value = step.params?.[def.key] ?? (def.options && def.options[0] ? def.options[0].value : '');
        wrapper.appendChild(select);
      } else {
        const input = document.createElement('input');
        input.type = 'number';
        input.name = def.key;
        if (def.step) input.step = def.step;
        if (def.min !== undefined) input.min = def.min;
        if (def.max !== undefined) input.max = def.max;
        const val = step.params?.[def.key];
        input.value = typeof val === 'number' && isFinite(val) ? val : '';
        wrapper.appendChild(input);
      }

      pipelineState.inspectorFieldsEl.appendChild(wrapper);
    }

    renderPipelineCards(pipelineState.graph);
  }

  function closeInspector() {
    pipelineState.inspectorStepId = null;
    if (pipelineState.inspectorEl) pipelineState.inspectorEl.classList.add('hidden');
    if (pipelineState.inspectorFieldsEl) pipelineState.inspectorFieldsEl.innerHTML = '';
    renderPipelineCards(pipelineState.graph);
  }

  function handleInspectorSubmit(event) {
    event.preventDefault();
    if (!pipelineState.inspectorForm) return;
    const step = pipelineState.graph.find((s) => s.id === pipelineState.inspectorStepId);
    if (!step) {
      closeInspector();
      return;
    }

    const defs = PARAM_DEFS[step.name] || [];
    const formData = new FormData(pipelineState.inspectorForm);
    const nextParams = { ...(step.params || {}) };
    for (const def of defs) {
      const value = formData.get(def.key);
      if (def.type === 'select') {
        nextParams[def.key] = value;
      } else {
        const parser = typeof def.parser === 'function' ? def.parser : (v) => parseFloat(v);
        const parsed = parser(value);
        if (!Number.isNaN(parsed) && isFinite(parsed)) {
          nextParams[def.key] = parsed;
        }
      }
    }
    step.params = nextParams;
    notifyGraphChanged();
    closeInspector();
  }

  function prepareForNewSection() {
    // グローバルをクリア（index.html側で定義）
    window.latestTapData = {};
    window.latestPipelineKey = null;
    const sel = document.getElementById('layerSelect');
    if (sel) {
      const current = sel.value;
      if (current && current !== 'raw') {
        pipelineState.desiredLayer = current;
      }
      sel.innerHTML = '';
      sel.appendChild(new Option('raw', 'raw'));
      sel.value = 'raw';
    }
  }

  function updateLayerSelect(tapMap) {
    const sel = document.getElementById('layerSelect');
    if (!sel) return;
    const names = Object.keys(tapMap || {}).sort((a, b) => a.localeCompare(b));
    const desired = pipelineState.desiredLayer;
    sel.innerHTML = '';
    sel.appendChild(new Option('raw', 'raw'));
    for (const name of names) sel.appendChild(new Option(name, name));
    let target = 'raw';
    if (desired && desired !== 'raw' && names.includes(desired)) {
      target = desired;
    } else if (names.length > 0) {
      target = names[0];
    }
    sel.value = target;
    pipelineState.desiredLayer = target;
    if (DEBUG_PIPELINE) {
      console.debug('[pipeline] updateLayerSelect: names=', names, 'selected=', target);
    }
  }

  async function runPipeline() {
    console.info('[pipeline] runPipeline() ENTER');
    console.info(
      '[pipeline] precheck: fileId=%o, key1Values.len=%o, slider? %o',
      window.currentFileId,
      Array.isArray(window.key1Values) ? window.key1Values.length : '(none)',
      !!document.getElementById('key1_idx_slider')
    );
    savePipelineToLocalStorage(pipelineState.graph);

    if (!window.currentFileId) {
      console.info('[pipeline] abort: no currentFileId');
      updateLayerSelect({});
      return;
    }
    const slider = document.getElementById('key1_idx_slider');
    if (!slider || !Array.isArray(window.key1Values) || !window.key1Values.length) {
      console.info('[pipeline] abort: slider/key1Values not ready', {
        sliderFound: !!slider,
        len: Array.isArray(window.key1Values) ? window.key1Values.length : null,
      });
      updateLayerSelect({});
      return;
    }
    const idx = parseInt(slider.value, 10);
    const key1Val = window.key1Values[idx];
    if (key1Val === undefined) {
      updateLayerSelect({});
      return;
    }
    const { spec, taps } = graphToSpec(pipelineState.graph);

    // 常に final を含めておく
    const tapsUnique = Array.from(new Set([...(taps || []), 'final']));
    console.info('[pipeline] runPipeline spec=%o taps=%o', spec, tapsUnique);
    if (DEBUG_PIPELINE) {
      console.debug('[pipeline] runPipeline spec=', spec, 'taps=', tapsUnique);
    }

    if (!spec.steps.length || !tapsUnique.length) {
      console.info('[pipeline] noop: no steps or taps');
      window.latestTapData = {};
      window.latestPipelineKey = null;
      updateLayerSelect({});
      if (typeof window.drawSelectedLayer === 'function') {
        const start = typeof window.renderedStart === 'number' ? window.renderedStart : 0;
        const end = typeof window.renderedEnd === 'number'
          ? window.renderedEnd
          : (Array.isArray(window.rawSeismicData) ? window.rawSeismicData.length - 1 : 0);
        window.drawSelectedLayer(start, end);
      }
      return;
    }

    const runId = ++pipelineState.runToken;
    try {
      const { taps: tapMap, pipelineKey } = await fetchSectionWithPipeline(
        window.currentFileId,
        key1Val,
        spec,
        tapsUnique,
        { key1Byte: window.currentKey1Byte, key2Byte: window.currentKey2Byte },
      );
      if (DEBUG_PIPELINE) {
        console.debug(
          '[pipeline] /pipeline/section OK pipelineKey=',
          pipelineKey,
          'tap keys=',
          Object.keys(tapMap || {})
        );
      }
      if (runId !== pipelineState.runToken) return;
      window.latestTapData = tapMap || {};
      window.latestPipelineKey = pipelineKey || null;
      updateLayerSelect(tapMap || {});
    } catch (error) {
      if (runId !== pipelineState.runToken) return;
      console.warn('pipeline/section failed', error);
      window.latestTapData = {};
      window.latestPipelineKey = null;
      updateLayerSelect({});
    }

    if (typeof window.drawSelectedLayer === 'function') {
      const start = typeof window.renderedStart === 'number' ? window.renderedStart : 0;
      const end = typeof window.renderedEnd === 'number'
        ? window.renderedEnd
        : (Array.isArray(window.rawSeismicData) ? window.rawSeismicData.length - 1 : 0);
      window.drawSelectedLayer(start, end);
    }
  }

  function notifyGraphChanged(options = {}) {
    savePipelineToLocalStorage(pipelineState.graph);
    renderPipelineCards(pipelineState.graph);
  }

  function initPipelineUI() {
    pipelineState.cardsContainer = document.getElementById('pipelineCards');
    pipelineState.inspectorEl = document.getElementById('pipelineInspector');
    pipelineState.inspectorForm = document.getElementById('pipelineInspectorForm');
    pipelineState.inspectorFieldsEl = document.getElementById('pipelineInspectorFields');
    pipelineState.inspectorTitleEl = document.getElementById('pipelineInspectorTitle');
    pipelineState.addMenuEl = document.getElementById('pipelineAddMenu');
    pipelineState.addButtonEl = document.getElementById('pipelineAddButton');

    pipelineState.graph = loadPipelineFromLocalStorage();
    if (DEBUG_PIPELINE) console.info('[pipeline] initPipelineUI(): graph=', pipelineState.graph);
    if (!Array.isArray(pipelineState.graph)) pipelineState.graph = [];

    renderPipelineCards(pipelineState.graph);
    savePipelineToLocalStorage(pipelineState.graph);

    const runButton = document.getElementById('pipelineRunButton');
    if (runButton) {
      runButton.addEventListener('click', () => runPipeline());
    }

    if (pipelineState.cardsContainer) {
      pipelineState.cardsContainer.addEventListener('dragover', handleDragOver);
      pipelineState.cardsContainer.addEventListener('drop', handleDrop);
    }

    if (pipelineState.inspectorForm) {
      pipelineState.inspectorForm.addEventListener('submit', handleInspectorSubmit);
    }
    const inspectorClose = document.getElementById('pipelineInspectorClose');
    if (inspectorClose) {
      inspectorClose.addEventListener('click', () => closeInspector());
    }

    if (pipelineState.addButtonEl && pipelineState.addMenuEl) {
      pipelineState.addButtonEl.addEventListener('click', (event) => {
        event.stopPropagation();
        pipelineState.addMenuEl.classList.toggle('open');
      });
      pipelineState.addMenuEl.querySelectorAll('button[data-step]').forEach((btn) => {
        btn.addEventListener('click', (event) => {
          event.stopPropagation();
          pipelineState.addMenuEl.classList.remove('open');
          addStep(btn.dataset.step);
        });
      });
      document.addEventListener('click', (event) => {
        if (!pipelineState.addMenuEl.contains(event.target) && event.target !== pipelineState.addButtonEl) {
          pipelineState.addMenuEl.classList.remove('open');
        }
      });
    }

    const layerSelect = document.getElementById('layerSelect');
    if (layerSelect) {
      pipelineState.desiredLayer = layerSelect.value || 'raw';
      layerSelect.addEventListener('change', () => {
        pipelineState.desiredLayer = layerSelect.value || 'raw';
      });
    }

  }

  const schedulePipelineRun = debounce(() => runPipeline(), 300);

  const pipelineUI = {
    initPipelineUI,
    renderPipelineCards,
    openInspector,
    closeInspector,
    graphToSpec,
    specToGraph,
    schedulePipelineRun,
    runPipeline,
    savePipelineToLocalStorage,
    loadPipelineFromLocalStorage,
    prepareForNewSection,
  };

  window.pipelineUI = pipelineUI;

  window.addEventListener('DOMContentLoaded', () => {
    console.info('[pipeline] DOMContentLoaded -> initPipelineUI()');
    initPipelineUI();
  });
})();

