(function () {
  function createPipelineInspectorRenderer(options = {}) {
    const state = options.state;
    const paramDefs = options.paramDefs || {};
    const formatStepName = options.formatStepName;
    const notifyGraphChanged = options.notifyGraphChanged;
    const requestRender = options.requestRender;

    function getState() {
      return state && typeof state.getState === 'function' ? state.getState() : null;
    }

    function openInspector(stepId) {
      const st = getState();
      if (!st) return;
      const step = st.graph.find((s) => s.id === stepId);
      if (!step || !st.inspectorEl || !st.inspectorFieldsEl || !st.inspectorTitleEl) return;
      state.setInspectorStepId(stepId);
      st.inspectorTitleEl.textContent = `${formatStepName(step.name)} Parameters`;
      st.inspectorFieldsEl.innerHTML = '';
      st.inspectorEl.classList.remove('hidden');

      const defs = paramDefs[step.name] || [];
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
          select.value = step.params && Object.prototype.hasOwnProperty.call(step.params, def.key)
            ? step.params[def.key]
            : (def.options && def.options[0] ? def.options[0].value : '');
          wrapper.appendChild(select);
        } else {
          const input = document.createElement('input');
          input.type = 'number';
          input.name = def.key;
          if (def.step) input.step = def.step;
          if (def.min !== undefined) input.min = def.min;
          if (def.max !== undefined) input.max = def.max;
          const val = step.params ? step.params[def.key] : undefined;
          input.value = typeof val === 'number' && isFinite(val) ? val : '';
          wrapper.appendChild(input);
        }

        st.inspectorFieldsEl.appendChild(wrapper);
      }

      if (typeof requestRender === 'function') {
        requestRender();
      }
    }

    function closeInspector() {
      const st = getState();
      if (!st) return;
      state.setInspectorStepId(null);
      if (st.inspectorEl) st.inspectorEl.classList.add('hidden');
      if (st.inspectorFieldsEl) st.inspectorFieldsEl.innerHTML = '';
      if (typeof requestRender === 'function') {
        requestRender();
      }
    }

    function handleInspectorSubmit(event) {
      event.preventDefault();
      const st = getState();
      if (!st || !st.inspectorForm) return;
      const step = st.graph.find((s) => s.id === state.getInspectorStepId());
      if (!step) {
        closeInspector();
        return;
      }

      const defs = paramDefs[step.name] || [];
      const formData = new FormData(st.inspectorForm);
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
      if (typeof notifyGraphChanged === 'function') {
        notifyGraphChanged();
      }
      closeInspector();
    }

    return {
      openInspector,
      closeInspector,
      handleInspectorSubmit,
    };
  }

  window.createPipelineInspectorRenderer = createPipelineInspectorRenderer;
})();
