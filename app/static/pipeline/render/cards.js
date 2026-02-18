(function () {
  function createPipelineCardsRenderer(options = {}) {
    const state = options.state;
    const formatStepName = options.formatStepName;
    const labelOf = options.labelOf;
    const openInspector = options.openInspector;
    const closeInspector = options.closeInspector;
    const notifyGraphChanged = options.notifyGraphChanged;

    function getState() {
      return state && typeof state.getState === 'function' ? state.getState() : null;
    }

    function renderPipelineCards(graph) {
      const st = getState();
      if (!st) return;
      const container = st.cardsContainer;
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
        if (state.getInspectorStepId() === step.id) card.classList.add('inspecting');

        card.addEventListener('dragstart', (event) => {
          state.setDraggedId(step.id);
          card.classList.add('is-dragging');
          event.dataTransfer.effectAllowed = 'move';
        });
        card.addEventListener('dragend', () => {
          card.classList.remove('is-dragging');
          state.setDraggedId(null);
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

        const delBtn = document.createElement('button');
        delBtn.type = 'button';
        delBtn.textContent = 'Delete';
        delBtn.addEventListener('click', (event) => {
          event.stopPropagation();
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
          if (!step.enabled && state.getDesiredLayer() === labelValue) {
            state.setDesiredLayer('raw');
            const sel = document.getElementById('layerSelect');
            if (sel) sel.value = 'raw';
            if (typeof drawSelectedLayer === 'function') {
              const start = typeof window.renderedStart === 'number' ? window.renderedStart : 0;
              const end = typeof window.renderedEnd === 'number'
                ? window.renderedEnd
                : (Array.isArray(window.rawSeismicData) ? window.rawSeismicData.length - 1 : 0);
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

          const enabled = st.graph.filter((s) => s && s.enabled);
          const idx = enabled.findIndex((s) => s.id === step.id);
          const pathKey = enabled.slice(0, idx + 1).map(labelOf).join('+');

          if (tapInput.checked) {
            state.setDesiredLayer(pathKey);
            const sel = document.getElementById('layerSelect');
            if (sel) sel.value = pathKey;
          } else if (state.getDesiredLayer() === pathKey) {
            state.setDesiredLayer('raw');
            const sel = document.getElementById('layerSelect');
            if (sel) sel.value = 'raw';
          }

          if (typeof drawSelectedLayer === 'function') {
            const start = typeof window.renderedStart === 'number' ? window.renderedStart : 0;
            const end = typeof window.renderedEnd === 'number'
              ? window.renderedEnd
              : (Array.isArray(window.rawSeismicData) ? window.rawSeismicData.length - 1 : 0);
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
          if (state.getDesiredLayer() === previous) {
            state.setDesiredLayer(nextLabel || step.name);
          }
          notifyGraphChanged();
        });
        labelWrap.appendChild(labelInput);
        card.appendChild(labelWrap);

        container.appendChild(card);
      }

      const inspectorStepId = state.getInspectorStepId();
      if (inspectorStepId && !graph.some((step) => step.id === inspectorStepId)) {
        closeInspector();
      }
    }

    function deleteStep(stepId) {
      const st = getState();
      if (!st) return;
      const idx = st.graph.findIndex((s) => s.id === stepId);
      if (idx === -1) return;
      const removed = st.graph[idx];
      const removedLabel = (removed.label && removed.label.trim()) || removed.name;
      if (state.getDesiredLayer() === removedLabel) {
        state.setDesiredLayer('raw');
        const sel = document.getElementById('layerSelect');
        if (sel) sel.value = 'raw';
      }
      st.graph.splice(idx, 1);
      if (state.getInspectorStepId() === stepId) {
        state.setInspectorStepId(null);
        if (st.inspectorEl) st.inspectorEl.classList.add('hidden');
        if (st.inspectorFieldsEl) st.inspectorFieldsEl.innerHTML = '';
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
      if (!state.getDraggedId()) return;
      event.preventDefault();
      const st = getState();
      if (!st) return;
      const container = st.cardsContainer;
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
      if (!state.getDraggedId()) return;
      event.preventDefault();
      const st = getState();
      if (!st) return;
      const container = st.cardsContainer;
      if (!container) return;
      const orderedIds = Array.from(container.querySelectorAll('.pipeline-card')).map((el) => el.dataset.id);
      const idToStep = new Map(st.graph.map((step) => [step.id, step]));
      const reordered = orderedIds.map((id) => idToStep.get(id)).filter(Boolean);
      if (reordered.length === st.graph.length) {
        state.setGraph(reordered);
        notifyGraphChanged();
      } else {
        renderPipelineCards(st.graph);
      }
      state.setDraggedId(null);
    }

    return {
      renderPipelineCards,
      deleteStep,
      handleDragOver,
      handleDrop,
    };
  }

  window.createPipelineCardsRenderer = createPipelineCardsRenderer;
})();
