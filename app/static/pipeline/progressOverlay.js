(function () {
  function createPipelineProgressOverlay(options = {}) {
    if (window.__pipelineProgressOverlayInstance) {
      return window.__pipelineProgressOverlayInstance;
    }

    const overlay = document.getElementById(options.overlayId || 'ppOverlay');
    const statusEl = document.getElementById(options.statusId || 'ppStatus');
    const barInner = document.getElementById(options.barId || 'ppBarInner');
    const cancelBtn = document.getElementById(options.cancelId || 'ppCancelBtn');

    if (!overlay || !statusEl || !barInner || !cancelBtn) {
      return null;
    }

    let totalStepsCount = 1;
    let attachedUI = null;
    let handlers = null;

    function setText(message) {
      statusEl.textContent = message || '';
    }

    function setProgress(percent) {
      const clamped = Math.max(0, Math.min(100, Math.round(percent)));
      barInner.style.width = `${clamped}%`;
    }

    function openOverlay(totalSteps = 1, initialText = 'Preparing…') {
      totalStepsCount = Math.max(1, Number(totalSteps) || 1);
      overlay.classList.add('show');
      setText(initialText);
      setProgress(0);
    }

    function stepOverlay(index, name) {
      const idx = Math.max(0, Number(index) || 0);
      const pct = Math.round((idx / totalStepsCount) * 100);
      const label = name ? `(${idx}/${totalStepsCount}) ${name}` : `Step ${idx}/${totalStepsCount}`;
      setText(label);
      setProgress(pct);
    }

    function closeOverlay() {
      overlay.classList.remove('show');
    }

    function errorOverlay(message) {
      setText(`Error: ${message}`);
      setProgress(100);
      setTimeout(() => closeOverlay(), 1200);
    }

    function attachToPipelineUI(pipelineUI) {
      if (!pipelineUI || typeof pipelineUI.on !== 'function') return;
      if (attachedUI === pipelineUI) return;

      if (attachedUI && typeof attachedUI.off === 'function' && handlers) {
        attachedUI.off('run:start', handlers.onStart);
        attachedUI.off('run:step', handlers.onStep);
        attachedUI.off('run:finish', handlers.onFinish);
        attachedUI.off('run:error', handlers.onError);
        attachedUI.off('run:cancel-requested', handlers.onCancelRequested);
        attachedUI.off('run:cancelled', handlers.onCancelled);
      }

      handlers = {
        onStart: (event) => {
          const total = event && typeof event.totalSteps === 'number' ? event.totalSteps : 1;
          openOverlay(total, 'Submitting…');
        },
        onStep: (event) => {
          const idx = event && typeof event.index === 'number' ? event.index : 0;
          const name = event && typeof event.name === 'string' ? event.name : 'Processing…';
          stepOverlay(idx, name);
        },
        onFinish: (event) => {
          const total = event && typeof event.totalSteps === 'number' ? event.totalSteps : totalStepsCount;
          stepOverlay(total, 'Done');
          setTimeout(() => closeOverlay(), 400);
        },
        onError: (event) => {
          const message = event && typeof event.message === 'string' ? event.message : 'failed';
          errorOverlay(message);
        },
        onCancelRequested: () => {
          setText('Cancelling…');
        },
        onCancelled: (event) => {
          const message = event && typeof event.message === 'string'
            ? event.message
            : 'The pipeline request was cancelled.';
          setText(message);
          setProgress(100);
          setTimeout(() => closeOverlay(), 700);
        },
      };

      pipelineUI.on('run:start', handlers.onStart);
      pipelineUI.on('run:step', handlers.onStep);
      pipelineUI.on('run:finish', handlers.onFinish);
      pipelineUI.on('run:error', handlers.onError);
      pipelineUI.on('run:cancel-requested', handlers.onCancelRequested);
      pipelineUI.on('run:cancelled', handlers.onCancelled);
      attachedUI = pipelineUI;
    }

    if (!cancelBtn.dataset.pipelineProgressBound) {
      cancelBtn.dataset.pipelineProgressBound = '1';
      cancelBtn.addEventListener('click', () => {
        if (typeof options.onCancel === 'function') {
          options.onCancel();
        }
      });
    }

    const api = {
      open: openOverlay,
      progress: setProgress,
      step: stepOverlay,
      text: setText,
      close: closeOverlay,
      error: errorOverlay,
      attachToPipelineUI,
    };

    window.__pipelineProgressOverlayInstance = api;
    return api;
  }

  window.createPipelineProgressOverlay = createPipelineProgressOverlay;
})();
