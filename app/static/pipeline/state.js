(function () {
  function createPipelineState(options = {}) {
    const events = options.events;
    const state = {
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
      initDiagnosticsLogged: false,
      progressOverlayBound: false,
      progressOverlayApi: null,
    };

    function emitStateChange(field, value) {
      if (events && typeof events.emit === 'function') {
        events.emit('state:change', { field, value, state });
      }
    }

    function getState() {
      return state;
    }

    function getGraph() {
      return state.graph;
    }

    function setGraph(nextGraph) {
      state.graph = Array.isArray(nextGraph) ? nextGraph : [];
      emitStateChange('graph', state.graph);
    }

    function getDesiredLayer() {
      return state.desiredLayer;
    }

    function setDesiredLayer(nextLayer) {
      state.desiredLayer = nextLayer || 'raw';
      emitStateChange('desiredLayer', state.desiredLayer);
    }

    function getInspectorStepId() {
      return state.inspectorStepId;
    }

    function setInspectorStepId(stepId) {
      state.inspectorStepId = stepId;
      emitStateChange('inspectorStepId', state.inspectorStepId);
    }

    function getDraggedId() {
      return state.draggedId;
    }

    function setDraggedId(stepId) {
      state.draggedId = stepId;
      emitStateChange('draggedId', state.draggedId);
    }

    function getRunToken() {
      return state.runToken;
    }

    function nextRunToken() {
      state.runToken += 1;
      emitStateChange('runToken', state.runToken);
      return state.runToken;
    }

    function isInitDiagnosticsLogged() {
      return !!state.initDiagnosticsLogged;
    }

    function setInitDiagnosticsLogged(flag) {
      state.initDiagnosticsLogged = !!flag;
      emitStateChange('initDiagnosticsLogged', state.initDiagnosticsLogged);
    }

    function setDomRefs(refs = {}) {
      if (Object.prototype.hasOwnProperty.call(refs, 'cardsContainer')) {
        state.cardsContainer = refs.cardsContainer;
        emitStateChange('cardsContainer', state.cardsContainer);
      }
      if (Object.prototype.hasOwnProperty.call(refs, 'inspectorEl')) {
        state.inspectorEl = refs.inspectorEl;
        emitStateChange('inspectorEl', state.inspectorEl);
      }
      if (Object.prototype.hasOwnProperty.call(refs, 'inspectorForm')) {
        state.inspectorForm = refs.inspectorForm;
        emitStateChange('inspectorForm', state.inspectorForm);
      }
      if (Object.prototype.hasOwnProperty.call(refs, 'inspectorFieldsEl')) {
        state.inspectorFieldsEl = refs.inspectorFieldsEl;
        emitStateChange('inspectorFieldsEl', state.inspectorFieldsEl);
      }
      if (Object.prototype.hasOwnProperty.call(refs, 'inspectorTitleEl')) {
        state.inspectorTitleEl = refs.inspectorTitleEl;
        emitStateChange('inspectorTitleEl', state.inspectorTitleEl);
      }
      if (Object.prototype.hasOwnProperty.call(refs, 'addMenuEl')) {
        state.addMenuEl = refs.addMenuEl;
        emitStateChange('addMenuEl', state.addMenuEl);
      }
      if (Object.prototype.hasOwnProperty.call(refs, 'addButtonEl')) {
        state.addButtonEl = refs.addButtonEl;
        emitStateChange('addButtonEl', state.addButtonEl);
      }
    }

    function getProgressOverlay() {
      return state.progressOverlayApi;
    }

    function setProgressOverlay(api) {
      state.progressOverlayApi = api;
      state.progressOverlayBound = !!api;
      emitStateChange('progressOverlayApi', state.progressOverlayApi);
      emitStateChange('progressOverlayBound', state.progressOverlayBound);
    }

    function isProgressOverlayBound() {
      return !!state.progressOverlayBound;
    }

    return {
      getState,
      getGraph,
      setGraph,
      getDesiredLayer,
      setDesiredLayer,
      getInspectorStepId,
      setInspectorStepId,
      getDraggedId,
      setDraggedId,
      getRunToken,
      nextRunToken,
      isInitDiagnosticsLogged,
      setInitDiagnosticsLogged,
      setDomRefs,
      getProgressOverlay,
      setProgressOverlay,
      isProgressOverlayBound,
    };
  }

  window.createPipelineState = createPipelineState;
})();
