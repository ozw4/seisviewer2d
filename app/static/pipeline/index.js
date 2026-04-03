(function () {
  const PUBLIC_API_KEYS = [
    'initPipelineUI',
    'renderPipelineCards',
    'openInspector',
    'closeInspector',
    'graphToSpec',
    'specToGraph',
    'schedulePipelineRun',
    'runPipeline',
    'savePipelineToLocalStorage',
    'loadPipelineFromLocalStorage',
    'prepareForNewSection',
    'on',
    'off',
    '_emit',
    'cancel',
  ];

  const DEBUG_PIPELINE = true;
  const VALID_STEP_NAMES = new Set(['bandpass', 'denoise']);

  const PARAM_DEFS = {
    bandpass: [
      { key: 'low_hz', label: 'low_hz', type: 'number', step: '0.1' },
      { key: 'high_hz', label: 'high_hz', type: 'number', step: '0.1' },
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

  let pipelineUIInstance = null;

  function ensureReady(cb) {
    if (typeof cb !== 'function') return;
    if (window.cfg && window.debounce) {
      cb();
      return;
    }
    (window.__viewerBootstrapQueue ||= []).push(cb);
  }

  if (typeof window.whenViewerBootstrapReady !== 'function') {
    window.whenViewerBootstrapReady = ensureReady;
  }

  function createFallbackEventBus() {
    const listeners = {};

    function on(eventName, handler) {
      if (typeof handler !== 'function') return;
      (listeners[eventName] ||= new Set()).add(handler);
    }

    function off(eventName, handler) {
      const bucket = listeners[eventName];
      if (!bucket) return;
      if (!handler) {
        bucket.clear();
        return;
      }
      bucket.delete(handler);
    }

    function emit(eventName, payload) {
      const bucket = listeners[eventName];
      if (!bucket) return;
      for (const handler of bucket) {
        try {
          handler(payload);
        } catch (err) {
          console.warn('[pipeline] listener for', eventName, 'threw', err);
        }
      }
    }

    return { on, off, emit };
  }

  function formatStepName(name) {
    if (!name) return '';
    return name.charAt(0).toUpperCase() + name.slice(1);
  }

  function defaultParamsFor(name) {
    if (name === 'bandpass') {
      return { low_hz: 5, high_hz: 60, taper: 0.1 };
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
    const incoming = params && typeof params === 'object' ? { ...params } : {};
    if (name === 'bandpass') {
      delete incoming.dt;
    }
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

  function getPipelineUI() {
    const ui = window.pipelineUI;
    if (!ui || typeof ui !== 'object') return null;
    return ui;
  }

  function createPipelineFacade(source) {
    const base = source && typeof source === 'object' ? source : getPipelineUI();
    if (!base) return null;

    const facade = {};
    for (const key of PUBLIC_API_KEYS) {
      if (Object.prototype.hasOwnProperty.call(base, key)) {
        facade[key] = base[key];
      }
    }
    return facade;
  }

  function createPipelineUI() {
    const eventBusFactory = typeof window.createPipelineEventBus === 'function'
      ? window.createPipelineEventBus
      : createFallbackEventBus;
    const existingBus = window.pipelineEvents;
    const eventBus = (
      existingBus &&
      typeof existingBus.on === 'function' &&
      typeof existingBus.off === 'function' &&
      typeof existingBus.emit === 'function'
    ) ? existingBus : eventBusFactory();
    window.pipelineEvents = eventBus;

    const stateFactory = window.createPipelineState;
    const storageFactory = window.createPipelineStorage;
    const cardsFactory = window.createPipelineCardsRenderer;
    const inspectorFactory = window.createPipelineInspectorRenderer;
    const runFactory = window.createPipelineRunner;

    if (typeof stateFactory !== 'function') {
      console.error('[pipeline][BLOCKER] missing createPipelineState()');
      return null;
    }
    if (typeof storageFactory !== 'function') {
      console.error('[pipeline][BLOCKER] missing createPipelineStorage()');
      return null;
    }
    if (typeof cardsFactory !== 'function') {
      console.error('[pipeline][BLOCKER] missing createPipelineCardsRenderer()');
      return null;
    }
    if (typeof inspectorFactory !== 'function') {
      console.error('[pipeline][BLOCKER] missing createPipelineInspectorRenderer()');
      return null;
    }
    if (typeof runFactory !== 'function') {
      console.error('[pipeline][BLOCKER] missing createPipelineRunner()');
      return null;
    }

    const state = stateFactory({ events: eventBus });

    let savePipelineToLocalStorage = function noopSave() {};
    let loadPipelineFromLocalStorage = function noopLoad() { return []; };
    let pipelineUI = null;

    const runner = runFactory({
      state,
      debugPipeline: DEBUG_PIPELINE,
      labelOf,
      normalizeStep,
      getPipelineUI: () => pipelineUIInstance || pipelineUI,
      getSavePipelineToLocalStorage: () => savePipelineToLocalStorage,
      events: eventBus,
    });
    eventBus.on('section:prepare', () => runner.prepareForNewSection());

    const storage = storageFactory({
      normalizeStep,
      graphToSpec: runner.graphToSpec,
      specToGraph: runner.specToGraph,
    });
    savePipelineToLocalStorage = storage.savePipelineToLocalStorage;
    loadPipelineFromLocalStorage = storage.loadPipelineFromLocalStorage;

    let renderPipelineCards = function noopRender() {};

    function notifyGraphChanged() {
      const st = state.getState();
      savePipelineToLocalStorage(st.graph);
      renderPipelineCards(st.graph);
    }

    const inspectorRenderer = inspectorFactory({
      state,
      paramDefs: PARAM_DEFS,
      formatStepName,
      notifyGraphChanged,
      requestRender: () => renderPipelineCards(state.getGraph()),
    });

    const cardsRenderer = cardsFactory({
      state,
      formatStepName,
      labelOf,
      openInspector: inspectorRenderer.openInspector,
      closeInspector: inspectorRenderer.closeInspector,
      notifyGraphChanged,
    });
    renderPipelineCards = cardsRenderer.renderPipelineCards;

    function addStep(name) {
      const step = createStep(name);
      if (!step) return;
      const st = state.getState();
      st.graph.push(step);
      notifyGraphChanged();
    }

    function initPipelineUI() {
      state.setDomRefs({
        cardsContainer: document.getElementById('pipelineCards'),
        inspectorEl: document.getElementById('pipelineInspector'),
        inspectorForm: document.getElementById('pipelineInspectorForm'),
        inspectorFieldsEl: document.getElementById('pipelineInspectorFields'),
        inspectorTitleEl: document.getElementById('pipelineInspectorTitle'),
        addMenuEl: document.getElementById('pipelineAddMenu'),
        addButtonEl: document.getElementById('pipelineAddButton'),
      });
      const st = state.getState();

      if (!state.isInitDiagnosticsLogged()) {
        runner.runInitDiagnostics();
        state.setInitDiagnosticsLogged(true);
      }

      state.setGraph(loadPipelineFromLocalStorage());
      if (DEBUG_PIPELINE) console.info('[pipeline] initPipelineUI(): graph=', st.graph);
      if (!Array.isArray(st.graph)) state.setGraph([]);

      renderPipelineCards(state.getGraph());
      savePipelineToLocalStorage(state.getGraph());

      const runButton = document.getElementById('pipelineRunButton');
      if (runButton) {
        runButton.addEventListener('click', () => runner.runPipeline());
      }
      const cancelButton = document.getElementById('pipelineCancelButton');
      if (cancelButton) {
        cancelButton.addEventListener('click', () => runner.cancel());
      }

      if (st.cardsContainer) {
        st.cardsContainer.addEventListener('dragover', cardsRenderer.handleDragOver);
        st.cardsContainer.addEventListener('drop', cardsRenderer.handleDrop);
      }

      if (st.inspectorForm) {
        st.inspectorForm.addEventListener('submit', inspectorRenderer.handleInspectorSubmit);
      }
      const inspectorClose = document.getElementById('pipelineInspectorClose');
      if (inspectorClose) {
        inspectorClose.addEventListener('click', () => inspectorRenderer.closeInspector());
      }

      if (st.addButtonEl && st.addMenuEl) {
        st.addButtonEl.addEventListener('click', (event) => {
          event.stopPropagation();
          st.addMenuEl.classList.toggle('open');
        });
        st.addMenuEl.querySelectorAll('button[data-step]').forEach((btn) => {
          btn.addEventListener('click', (event) => {
            event.stopPropagation();
            st.addMenuEl.classList.remove('open');
            addStep(btn.dataset.step);
          });
        });
        document.addEventListener('click', (event) => {
          if (!st.addMenuEl.contains(event.target) && event.target !== st.addButtonEl) {
            st.addMenuEl.classList.remove('open');
          }
        });
      }

      const layerSelect = document.getElementById('layerSelect');
      if (layerSelect) {
        state.setDesiredLayer(layerSelect.value || 'raw');
        layerSelect.addEventListener('change', () => {
          state.setDesiredLayer(layerSelect.value || 'raw');
        });
      }

      runner.ensureProgressOverlayBridge();
    }

    const debounceFn = typeof window.debounce === 'function'
      ? window.debounce
      : ((fn) => fn);
    const schedulePipelineRun = debounceFn(() => runner.runPipeline(), 300);

    pipelineUI = {
      initPipelineUI,
      renderPipelineCards,
      openInspector: inspectorRenderer.openInspector,
      closeInspector: inspectorRenderer.closeInspector,
      graphToSpec: runner.graphToSpec,
      specToGraph: runner.specToGraph,
      schedulePipelineRun,
      runPipeline: runner.runPipeline,
      cancel: runner.cancel,
      savePipelineToLocalStorage,
      loadPipelineFromLocalStorage,
      prepareForNewSection: runner.prepareForNewSection,
    };

    pipelineUI.on = pipelineUI.on || eventBus.on;
    pipelineUI.off = pipelineUI.off || eventBus.off;
    pipelineUI._emit = pipelineUI._emit || eventBus.emit;
    let initStarted = false;
    function startPipelineUI() {
      if (initStarted) return;
      initStarted = true;
      console.info('[pipeline] DOMContentLoaded -> initPipelineUI()');
      initPipelineUI();
    }

    if (document.readyState === 'loading') {
      window.addEventListener('DOMContentLoaded', startPipelineUI, { once: true });
    } else {
      startPipelineUI();
    }

    return pipelineUI;
  }

  function bootstrapPipelineUI() {
    if (pipelineUIInstance) return pipelineUIInstance;

    window.whenViewerBootstrapReady(() => {
      if (pipelineUIInstance) return;
      console.info('[pipeline] pipeline/index.js LOADED');
      const created = createPipelineUI();
      if (!created) return;
      pipelineUIInstance = created;
      window.pipelineUI = created;
    });

    return pipelineUIInstance;
  }

  window.pipelineIndex = {
    publicApiKeys: PUBLIC_API_KEYS.slice(),
    getPipelineUI,
    createPipelineFacade,
    createPipelineUI,
    bootstrap: bootstrapPipelineUI,
  };

  bootstrapPipelineUI();
  if (window.__pipelineLegacyPipelineUIBootstrapRequested) {
    bootstrapPipelineUI();
  }
})();
