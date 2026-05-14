(function () {
  const state = {
    ready: false,
    message: 'Static correction setup is not available yet.',
    error: '',
  };

  let dom = null;

  function render() {
    if (!dom) return;
    dom.status.textContent = state.message;
    dom.error.hidden = !state.error;
    dom.error.textContent = state.error;
    dom.runButton.disabled = true;
  }

  function init() {
    const form = document.getElementById('staticCorrectionForm');
    const status = document.getElementById('staticCorrectionStatus');
    const error = document.getElementById('staticCorrectionError');
    const runButton = document.getElementById('staticCorrectionRunButton');
    if (!form || !status || !error || !runButton) return;

    dom = {
      form,
      status,
      error,
      runButton,
    };

    form.addEventListener('submit', (event) => {
      event.preventDefault();
    });
    runButton.addEventListener('click', (event) => {
      event.preventDefault();
    });

    render();
  }

  window.refractionStaticRunState = state;
  window.refractionStaticRunUI = {
    render,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
