(function () {
  function createPipelineRunner(options = {}) {
    const state = options.state;
    const debugPipeline = !!options.debugPipeline;
    const labelOf = options.labelOf;
    const normalizeStep = options.normalizeStep;
    const getPipelineUI = options.getPipelineUI;
    const getSavePipelineToLocalStorage = options.getSavePipelineToLocalStorage;
    const events = options.events;
    const uiState = {
      activeAbortController: null,
      activeRunId: 0,
      cancelRequested: false,
    };

    function getState() {
      return state && typeof state.getState === 'function' ? state.getState() : null;
    }

    function emitPipelineEvent(eventName, payload) {
      const ui = typeof getPipelineUI === 'function' ? getPipelineUI() : window.pipelineUI;
      if (ui && typeof ui._emit === 'function') {
        ui._emit(eventName, payload);
        return;
      }
      if (events && typeof events.emit === 'function') {
        events.emit(eventName, payload);
      }
    }

    function getRunDom() {
      return {
        runButton: document.getElementById('pipelineRunButton'),
        addButton: document.getElementById('pipelineAddButton'),
        cancelButton: document.getElementById('pipelineCancelButton'),
        statusRoot: document.getElementById('pipelineRunStatus'),
        badge: document.getElementById('pipelineRunBadge'),
        summary: document.getElementById('pipelineRunSummary'),
        detail: document.getElementById('pipelineRunDetail'),
        note: document.getElementById('pipelineRunNote'),
      };
    }

    function setRunControlsBusy(isBusy) {
      const { runButton, addButton, cancelButton } = getRunDom();
      if (runButton) {
        runButton.disabled = !!isBusy;
        runButton.textContent = isBusy ? 'Running…' : '▶ Run';
      }
      if (addButton) {
        addButton.disabled = !!isBusy;
      }
      if (cancelButton) {
        cancelButton.hidden = !isBusy;
        cancelButton.disabled = !isBusy;
      }
    }

    function setRunStatus(status, summary, detail = '', note = '') {
      const { statusRoot, badge, summary: summaryEl, detail: detailEl, note: noteEl } = getRunDom();
      if (statusRoot) {
        statusRoot.classList.remove(
          'is-idle',
          'is-running',
          'is-success',
          'is-error',
          'is-cancelled',
          'is-cancel-requested'
        );
        const classByStatus = {
          idle: 'is-idle',
          running: 'is-running',
          success: 'is-success',
          error: 'is-error',
          cancelled: 'is-cancelled',
          cancel_requested: 'is-cancel-requested',
        };
        statusRoot.classList.add(classByStatus[status] || 'is-idle');
      }
      if (badge) {
        badge.textContent = status.replace('_', ' ');
      }
      if (summaryEl) {
        summaryEl.textContent = summary || '';
      }
      if (detailEl) {
        detailEl.textContent = detail || '';
        detailEl.classList.toggle('is-error', status === 'error');
        detailEl.classList.toggle('is-cancelled', status === 'cancelled');
      }
      if (noteEl) {
        noteEl.textContent = note || '';
      }
    }

    function formatStepList(steps) {
      return (Array.isArray(steps) ? steps : [])
        .map((step) => String(step.label || step.name || '').trim())
        .filter(Boolean)
        .join(' -> ');
    }

    function isAbortError(error) {
      return !!error && (error.name === 'AbortError' || error.code === 20);
    }

    function normalizePipelineRunError(error) {
      if (isAbortError(error)) {
        return 'The pipeline request was cancelled in the viewer.';
      }
      const text = String((error && error.message) || error || '').trim();
      const lower = text.toLowerCase();
      if (!text) {
        return 'The server returned an unexpected error. Check server logs.';
      }
      if (lower.includes('failed to fetch') || lower.includes('networkerror')) {
        return 'Network error while contacting the server. Check the connection and retry.';
      }
      if (lower.includes('fb pick model weights not found') || lower.includes('fbpick model not found')) {
        return 'FB model file is missing. Place the model weights under model/ and retry.';
      }
      if (lower.includes('file id not found') || lower.includes('no currentfileid')) {
        return 'No section is loaded yet. Load data before running pipeline.';
      }
      if (lower.includes('model_id must') || lower.includes('must be') || lower.includes('unprocessable')) {
        return `Some pipeline parameters are invalid. Review the step settings and retry.\n${text}`;
      }
      if (lower.includes('pipeline/section 4')) {
        return text;
      }
      if (lower.includes('pipeline/section 5')) {
        return `The server returned an unexpected error. Check server logs.\n${text}`;
      }
      return text;
    }

    function ensureProgressOverlayBridge() {
      if (state.isProgressOverlayBound()) return state.getProgressOverlay();
      const ui = typeof getPipelineUI === 'function' ? getPipelineUI() : window.pipelineUI;
      const eventSource = (
        window.pipelineEvents &&
        typeof window.pipelineEvents.on === 'function' &&
        typeof window.pipelineEvents.off === 'function'
      ) ? window.pipelineEvents : ui;
      if (window.pipelineProgress && typeof window.pipelineProgress.open === 'function') {
        if (typeof window.pipelineProgress.attachToPipelineUI === 'function' && eventSource) {
          window.pipelineProgress.attachToPipelineUI(eventSource);
        }
        state.setProgressOverlay(window.pipelineProgress);
        return window.pipelineProgress;
      }

      const createProgressOverlay = window.createPipelineProgressOverlay;
      if (typeof createProgressOverlay !== 'function') return null;
      const progressOverlay = createProgressOverlay({
        overlayId: 'ppOverlay',
        statusId: 'ppStatus',
        barId: 'ppBarInner',
        cancelId: 'ppCancelBtn',
        onCancel: () => {
          const pipelineUI = typeof getPipelineUI === 'function' ? getPipelineUI() : window.pipelineUI;
          if (pipelineUI && typeof pipelineUI.cancel === 'function') {
            pipelineUI.cancel();
          }
        },
      });
      if (!progressOverlay) return null;
      if (typeof progressOverlay.attachToPipelineUI === 'function' && eventSource) {
        progressOverlay.attachToPipelineUI(eventSource);
      }
      window.pipelineProgress = progressOverlay;
      state.setProgressOverlay(progressOverlay);
      return progressOverlay;
    }

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
          if (step.tap) taps.push(path.join('+'));
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

    function prepareForNewSection() {
      const st = getState();
      if (!st) return;

      cancel();

      window.latestTapData = {};
      window.latestPipelineKey = null;
      const sel = document.getElementById('layerSelect');
      if (sel) {
        const current = sel.value;
        if (current && current !== 'raw') {
          state.setDesiredLayer(current);
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
      const desired = state.getDesiredLayer();
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
      state.setDesiredLayer(target);
      if (debugPipeline) {
        console.debug('[pipeline] updateLayerSelect: names=', names, 'selected=', target);
      }
    }

    function cancel() {
      if (!uiState.activeAbortController) {
        return false;
      }
      uiState.cancelRequested = true;
      uiState.activeAbortController.abort();
      setRunStatus(
        'cancel_requested',
        'Cancelling the viewer request...',
        'The current request has been aborted locally. The server may still finish the in-flight work.',
        'Server-side step streaming is not available for this endpoint yet.'
      );
      emitPipelineEvent('run:cancel-requested', {});
      return true;
    }

    function runInitDiagnostics() {
      const required = [
        { id: 'pipelineRunButton', reason: 'RUN click handler cannot be attached' },
        { id: 'layerSelect', reason: 'Pipeline layer selection cannot be updated' },
        { id: 'key1_slider', reason: 'Primary key1 source is unavailable' },
        { id: 'key1_val_display', reason: 'Fallback key1 source by value is unavailable' },
      ];
      const missing = required
        .filter((entry) => !document.getElementById(entry.id))
        .map((entry) => ({ id: entry.id, reason: entry.reason }));
      if (missing.length) {
        console.error('[pipeline][BLOCKER] init diagnostics: missing critical DOM elements', { missing });
        return;
      }
      if (debugPipeline) {
        console.info('[pipeline] init diagnostics: critical DOM elements are present');
      }
    }

    function resolvePipelineRunKey1() {
      const key1Values = Array.isArray(window.key1Values) ? window.key1Values : null;
      const slider = document.getElementById('key1_slider');
      const display = document.getElementById('key1_val_display');
      const store = window.store && typeof window.store.get === 'function' ? window.store.get() : null;
      const storeIndexRaw = store && store.key1Index !== undefined ? store.key1Index : null;
      const storeIndex = Number.parseInt(storeIndexRaw, 10);
      const context = {
        key1ValuesLen: key1Values ? key1Values.length : null,
        sliderFound: !!slider,
        sliderValue: slider ? slider.value : null,
        displayFound: !!display,
        displayValue: display ? display.value : null,
        storeFound: !!store,
        storeIndexRaw,
        storeIndex: Number.isInteger(storeIndex) ? storeIndex : null,
      };

      if (!key1Values || !key1Values.length) {
        return { ok: false, context: { ...context, reason: 'key1Values unavailable' } };
      }

      if (slider) {
        const sliderIndex = Number.parseInt(slider.value, 10);
        if (Number.isInteger(sliderIndex) && sliderIndex >= 0 && sliderIndex < key1Values.length) {
          const key1Val = key1Values[sliderIndex];
          if (key1Val !== undefined) {
            return { ok: true, source: 'key1_slider', key1Val, key1Index: sliderIndex, context };
          }
        }
      }

      if (display) {
        const displayValue = Number(display.value);
        if (Number.isFinite(displayValue)) {
          const displayIndex = key1Values.indexOf(displayValue);
          if (displayIndex >= 0) {
            const key1Val = key1Values[displayIndex];
            if (key1Val !== undefined) {
              const fallbackContext = { ...context, displayIndex };
              console.warn('[pipeline][FALLBACK] key1 resolved via #key1_val_display', fallbackContext);
              return {
                ok: true,
                source: 'key1_val_display',
                key1Val,
                key1Index: displayIndex,
                context: fallbackContext,
              };
            }
          }
        }
      }

      if (Number.isInteger(storeIndex) && storeIndex >= 0 && storeIndex < key1Values.length) {
        const key1Val = key1Values[storeIndex];
        if (key1Val !== undefined) {
          const fallbackContext = { ...context, resolvedStoreIndex: storeIndex };
          console.warn('[pipeline][FALLBACK] key1 resolved via window.store.get().key1Index', fallbackContext);
          return {
            ok: true,
            source: 'store.key1Index',
            key1Val,
            key1Index: storeIndex,
            context: fallbackContext,
          };
        }
      }

      return { ok: false, context: { ...context, reason: 'no valid key1 source resolved' } };
    }

    async function runPipeline() {
      ensureProgressOverlayBridge();
      const st = getState();
      if (!st) return;

      console.info('[pipeline] runPipeline() ENTER');
      console.info(
        '[pipeline] precheck: fileId=%o, key1Values.len=%o, slider? %o, key1_val_display? %o',
        window.currentFileId,
        Array.isArray(window.key1Values) ? window.key1Values.length : '(none)',
        !!document.getElementById('key1_slider'),
        !!document.getElementById('key1_val_display')
      );

      const saveFn = typeof getSavePipelineToLocalStorage === 'function'
        ? getSavePipelineToLocalStorage()
        : null;
      if (typeof saveFn === 'function') {
        saveFn(st.graph);
      }

      if (!window.currentFileId) {
        console.info('[pipeline] abort: no currentFileId');
        updateLayerSelect({});
        setRunStatus('error', 'Pipeline could not start.', 'No section is loaded yet. Load data before running pipeline.');
        return;
      }
      const resolvedKey1 = resolvePipelineRunKey1();
      if (!resolvedKey1.ok) {
        console.error('[pipeline][BLOCKER] abort: key1 source not ready', resolvedKey1.context);
        updateLayerSelect({});
        setRunStatus('error', 'Pipeline could not start.', 'The current section is not ready yet. Wait for the viewer to finish loading and retry.');
        return;
      }
      const key1Val = resolvedKey1.key1Val;
      console.info(
        '[pipeline] runPipeline key1 resolved: value=%o index=%o source=%o',
        key1Val,
        resolvedKey1.key1Index,
        resolvedKey1.source
      );
      const converted = graphToSpec(st.graph);
      const spec = converted.spec;
      const taps = converted.taps;

      const tapsUnique = Array.from(new Set([...(taps || []), 'final']));
      console.info('[pipeline] runPipeline spec=%o taps=%o', spec, tapsUnique);
      if (debugPipeline) {
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
        setRunStatus('idle', 'No enabled pipeline steps.', 'Enable at least one step before running the pipeline.');
        return;
      }

      const steps = Array.isArray(spec.steps) ? spec.steps : [];
      const stepList = formatStepList(steps);
      const totalSteps = Math.max(steps.length, 1);
      const runId = state.nextRunToken();
      uiState.activeRunId = runId;
      uiState.cancelRequested = false;
      uiState.activeAbortController = new AbortController();
      setRunControlsBusy(true);
      setRunStatus(
        'running',
        `Running ${steps.length} step${steps.length === 1 ? '' : 's'} for key1=${key1Val}.`,
        stepList ? `Steps: ${stepList}` : 'The pipeline request is being sent to the server.',
        'Live per-step progress is not streamed by this endpoint yet.'
      );
      emitPipelineEvent('run:start', { totalSteps, stepNames: steps.map((step) => step.label || step.name) });
      try {
        const response = await fetchSectionWithPipeline(
          window.currentFileId,
          key1Val,
          spec,
          tapsUnique,
          {
            key1Byte: window.currentKey1Byte,
            key2Byte: window.currentKey2Byte,
            signal: uiState.activeAbortController.signal,
          }
        );
        const tapMap = response && response.taps ? response.taps : {};
        const pipelineKey = response && response.pipelineKey ? response.pipelineKey : null;

        if (debugPipeline) {
          console.debug(
            '[pipeline] /pipeline/section OK pipelineKey=',
            pipelineKey,
            'tap keys=',
            Object.keys(tapMap || {})
          );
        }
        if (runId !== state.getRunToken()) return;
        const activeKey1 = resolvePipelineRunKey1();
        if (!activeKey1.ok || activeKey1.key1Val !== key1Val) {
          console.warn(
            '[pipeline][FALLBACK] key1 changed during run; discarding stale pipeline result',
            {
              startedWith: key1Val,
              activeNow: activeKey1.ok ? activeKey1.key1Val : null,
              activeSource: activeKey1.ok ? activeKey1.source : null,
              resolverContext: activeKey1.context,
            }
          );
          window.latestTapData = {};
          window.latestPipelineKey = null;
          updateLayerSelect({});
          emitPipelineEvent('run:error', { message: 'key1 changed during run; result discarded' });
          setRunStatus('error', 'Pipeline result was discarded.', 'The selected section changed while the pipeline was running. Retry on the current section.');
          if (uiState.activeRunId === runId) {
            uiState.activeAbortController = null;
            uiState.cancelRequested = false;
            setRunControlsBusy(false);
          }
          return;
        }
        window.latestTapData = tapMap || {};
        window.latestPipelineKey = pipelineKey || null;
        updateLayerSelect(tapMap || {});
        const finalName = steps.length
          ? steps[steps.length - 1].label || steps[steps.length - 1].name || `Step ${steps.length}`
          : 'Complete';
        emitPipelineEvent('run:step', { index: totalSteps, name: finalName });
        emitPipelineEvent('run:finish', { totalSteps });
        setRunStatus(
          'success',
          'Pipeline completed.',
          stepList ? `Completed: ${stepList}` : 'The requested pipeline output is ready.',
          'Use the layer selector to inspect the generated tap output.'
        );
      } catch (error) {
        if (runId !== state.getRunToken()) return;
        console.warn('pipeline/section failed', error);
        window.latestTapData = {};
        window.latestPipelineKey = null;
        updateLayerSelect({});
        if (isAbortError(error)) {
          emitPipelineEvent('run:cancelled', { message: 'The pipeline request was cancelled in the viewer.' });
          setRunStatus(
            'cancelled',
            'Pipeline request cancelled.',
            'The request was stopped in the viewer. The server may still finish the current in-flight work.',
            'You can run the pipeline again on the current section.'
          );
        } else {
          const message = normalizePipelineRunError(error);
          emitPipelineEvent('run:error', { message });
          setRunStatus('error', 'Pipeline failed.', message, 'Review the message above, adjust inputs if needed, and retry.');
        }
      }

      if (uiState.activeRunId === runId) {
        uiState.activeAbortController = null;
        uiState.cancelRequested = false;
        setRunControlsBusy(false);
      }

      if (typeof window.drawSelectedLayer === 'function') {
        const start = typeof window.renderedStart === 'number' ? window.renderedStart : 0;
        const end = typeof window.renderedEnd === 'number'
          ? window.renderedEnd
          : (Array.isArray(window.rawSeismicData) ? window.rawSeismicData.length - 1 : 0);
        window.drawSelectedLayer(start, end);
      }
    }

    return {
      graphToSpec,
      specToGraph,
      prepareForNewSection,
      updateLayerSelect,
      runInitDiagnostics,
      resolvePipelineRunKey1,
      runPipeline,
      ensureProgressOverlayBridge,
      cancel,
    };
  }

  window.createPipelineRunner = createPipelineRunner;
})();
