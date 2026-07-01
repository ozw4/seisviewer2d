(function initControlsPanel() {
  const STORAGE_OPEN = 'sv.controls_panel.open';
  const STORAGE_HEIGHT = 'sv.controls_panel.height';
  const DEFAULT_HEIGHT = 270;
  const MIN_HEIGHT = 120;
  const MOBILE_BREAKPOINT = 640;

  function clampHeight(nextHeight) {
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || DEFAULT_HEIGHT;
    const maxHeight = Math.max(MIN_HEIGHT, Math.floor(viewportHeight * 0.6));
    return Math.min(maxHeight, Math.max(MIN_HEIGHT, Math.round(nextHeight)));
  }

  function isMobileLayout() {
    return window.innerWidth <= MOBILE_BREAKPOINT;
  }

  function schedulePlotResize() {
    const plotDiv = document.getElementById('plot');
    if (!plotDiv || typeof window.maybeResizePlot !== 'function') return;
    window.requestAnimationFrame(() => {
      window.maybeResizePlot(plotDiv, true).catch((err) => console.warn('controls panel resize failed', err));
    });
  }

  window.addEventListener('DOMContentLoaded', () => {
    const panel = document.getElementById('controlsPanel');
    const body = document.getElementById('controlsPanelBody');
    const toggle = document.getElementById('controlsPanelToggle');
    const handle = document.getElementById('controlsResizeHandle');
    if (!panel || !body || !toggle || !handle) return;

    let dragging = false;
    let dragStartY = 0;
    let dragStartHeight = DEFAULT_HEIGHT;
    let currentHeight = DEFAULT_HEIGHT;
    let isOpen = true;

    function persistState() {
      try {
        localStorage.setItem(STORAGE_OPEN, isOpen ? 'true' : 'false');
        localStorage.setItem(STORAGE_HEIGHT, String(currentHeight));
      } catch (_) {
      }
    }

    function applyHeight(nextHeight, options) {
      currentHeight = clampHeight(nextHeight);
      if (isMobileLayout()) {
        body.style.removeProperty('height');
      } else {
        body.style.height = currentHeight + 'px';
      }
      if (!options || options.persist !== false) persistState();
    }

    function applyOpenState(nextOpen, options) {
      isOpen = !!nextOpen;
      panel.classList.toggle('is-collapsed', !isOpen);
      toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      toggle.textContent = isOpen ? 'Hide Controls' : 'Show Controls';
      body.hidden = !isOpen;
      if (isOpen) {
        applyHeight(currentHeight, { persist: false });
      }
      if (!options || options.persist !== false) persistState();
      schedulePlotResize();
    }

    try {
      const storedOpen = localStorage.getItem(STORAGE_OPEN);
      if (storedOpen === 'false') isOpen = false;
      const storedHeight = Number(localStorage.getItem(STORAGE_HEIGHT));
      if (Number.isFinite(storedHeight) && storedHeight > 0) {
        currentHeight = clampHeight(storedHeight);
      }
    } catch (_) {
    }

    applyHeight(currentHeight, { persist: false });
    applyOpenState(isOpen, { persist: false });

    toggle.addEventListener('click', () => {
      applyOpenState(!isOpen);
    });

    handle.addEventListener('pointerdown', (event) => {
      if (!isOpen || isMobileLayout()) return;
      dragging = true;
      dragStartY = event.clientY;
      dragStartHeight = body.getBoundingClientRect().height;
      panel.classList.add('is-resizing');
      handle.setPointerCapture(event.pointerId);
      event.preventDefault();
    });

    handle.addEventListener('pointermove', (event) => {
      if (!dragging) return;
      const deltaY = event.clientY - dragStartY;
      applyHeight(dragStartHeight + deltaY, { persist: false });
      schedulePlotResize();
    });

    function finishDrag(event) {
      if (!dragging) return;
      dragging = false;
      panel.classList.remove('is-resizing');
      if (event && typeof handle.releasePointerCapture === 'function' && handle.hasPointerCapture(event.pointerId)) {
        handle.releasePointerCapture(event.pointerId);
      }
      persistState();
      schedulePlotResize();
    }

    handle.addEventListener('pointerup', finishDrag);
    handle.addEventListener('pointercancel', finishDrag);

    window.addEventListener('resize', () => {
      applyHeight(currentHeight, { persist: false });
      schedulePlotResize();
    });
  });
})();
