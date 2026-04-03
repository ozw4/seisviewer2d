(function () {
  const DEFAULT_DURATIONS = {
    info: 3200,
    success: 2600,
    warning: 4200,
    error: 6400,
  };
  const DEDUPE_WINDOW_MS = 2000;
  const activeToasts = new Map();
  const recentToastTimestamps = new Map();
  let initialized = false;
  let bannerEl = null;
  let bannerMessageEl = null;
  let toastStackEl = null;
  let politeLiveEl = null;
  let assertiveLiveEl = null;

  function ensureRoot(selector, className, tagName) {
    let el = document.querySelector(selector);
    if (!el) {
      el = document.createElement(tagName);
      if (className) {
        el.className = className;
      }
      if (selector === '[data-app-status-banner-host]') {
        el.setAttribute('data-app-status-banner-host', '');
        const anchor = document.querySelector('header') || document.body.firstElementChild;
        if (anchor && anchor.parentNode) {
          anchor.insertAdjacentElement('afterend', el);
        } else {
          document.body.prepend(el);
        }
      } else if (selector === '[data-app-status-toast-stack]') {
        el.setAttribute('data-app-status-toast-stack', '');
        document.body.appendChild(el);
      }
    }
    return el;
  }

  function announce(message, level) {
    const liveEl = level === 'error' || level === 'warning' ? assertiveLiveEl : politeLiveEl;
    if (!liveEl) {
      return;
    }
    liveEl.textContent = '';
    window.setTimeout(() => {
      liveEl.textContent = message;
    }, 20);
  }

  function levelLabel(level) {
    switch (level) {
      case 'success':
        return 'Success';
      case 'warning':
        return 'Warning';
      case 'error':
        return 'Error';
      default:
        return 'Info';
    }
  }

  function normalizeLevel(level) {
    return ['info', 'success', 'warning', 'error'].includes(level) ? level : 'info';
  }

  function ensureInitialized() {
    if (initialized || !document.body) {
      return;
    }

    const bannerHost = ensureRoot('[data-app-status-banner-host]', 'app-status-banner-host', 'div');
    if (!bannerEl) {
      bannerEl = document.createElement('div');
      bannerEl.className = 'app-status-banner';
      bannerEl.setAttribute('role', 'status');
      bannerEl.setAttribute('aria-atomic', 'true');
      bannerMessageEl = document.createElement('div');
      bannerMessageEl.className = 'app-status-banner__message';
      const bannerCloseEl = document.createElement('button');
      bannerCloseEl.type = 'button';
      bannerCloseEl.className = 'app-status-banner__close';
      bannerCloseEl.setAttribute('aria-label', 'Dismiss status');
      bannerCloseEl.textContent = '×';
      bannerCloseEl.addEventListener('click', () => {
        clearBanner();
      });
      bannerEl.append(bannerMessageEl, bannerCloseEl);
      bannerHost.appendChild(bannerEl);
    }

    toastStackEl = ensureRoot('[data-app-status-toast-stack]', 'app-status-toast-stack', 'div');

    if (!politeLiveEl) {
      politeLiveEl = document.createElement('div');
      politeLiveEl.className = 'app-status-sr-only';
      politeLiveEl.setAttribute('aria-live', 'polite');
      politeLiveEl.setAttribute('aria-atomic', 'true');
      document.body.appendChild(politeLiveEl);
    }

    if (!assertiveLiveEl) {
      assertiveLiveEl = document.createElement('div');
      assertiveLiveEl.className = 'app-status-sr-only';
      assertiveLiveEl.setAttribute('aria-live', 'assertive');
      assertiveLiveEl.setAttribute('aria-atomic', 'true');
      document.body.appendChild(assertiveLiveEl);
    }

    initialized = true;
  }

  function clearBanner() {
    ensureInitialized();
    if (!bannerEl || !bannerMessageEl) {
      return;
    }
    bannerMessageEl.textContent = '';
    bannerEl.classList.remove('is-visible');
    bannerEl.dataset.level = 'info';
  }

  function setBanner(message, level = 'info', options) {
    ensureInitialized();
    const normalizedLevel = normalizeLevel(level);
    const nextMessage = typeof message === 'string' ? message.trim() : '';
    if (!nextMessage) {
      clearBanner();
      return;
    }
    bannerEl.dataset.level = normalizedLevel;
    bannerMessageEl.textContent = nextMessage;
    bannerEl.classList.add('is-visible');
    announce(nextMessage, normalizedLevel);
    if (options && options.toast) {
      showToast(nextMessage, normalizedLevel, options.toastOptions || {});
    }
  }

  function removeToast(key) {
    const entry = activeToasts.get(key);
    if (!entry) {
      return;
    }
    if (entry.timeoutId) {
      window.clearTimeout(entry.timeoutId);
    }
    entry.el.remove();
    activeToasts.delete(key);
  }

  function scheduleToastRemoval(key, duration) {
    const entry = activeToasts.get(key);
    if (!entry) {
      return;
    }
    if (entry.timeoutId) {
      window.clearTimeout(entry.timeoutId);
    }
    if (duration <= 0) {
      entry.timeoutId = null;
      return;
    }
    entry.timeoutId = window.setTimeout(() => {
      removeToast(key);
    }, duration);
  }

  function showToast(message, level = 'info', options = {}) {
    ensureInitialized();
    const normalizedLevel = normalizeLevel(level);
    const text = typeof message === 'string' ? message.trim() : '';
    if (!text) {
      return null;
    }
    const key = `${normalizedLevel}:${text}`;
    const now = Date.now();
    const sticky = Boolean(options.sticky);
    const duration = sticky ? 0 : options.duration ?? DEFAULT_DURATIONS[normalizedLevel];
    const recentAt = recentToastTimestamps.get(key);
    if (recentAt && now - recentAt < (options.dedupeWindowMs ?? DEDUPE_WINDOW_MS)) {
      const existingEntry = activeToasts.get(key);
      if (existingEntry) {
        scheduleToastRemoval(key, duration);
      }
      return existingEntry ? existingEntry.el : null;
    }
    recentToastTimestamps.set(key, now);

    const existingEntry = activeToasts.get(key);
    if (existingEntry) {
      existingEntry.messageEl.textContent = text;
      existingEntry.el.dataset.level = normalizedLevel;
      scheduleToastRemoval(key, duration);
      announce(text, normalizedLevel);
      return existingEntry.el;
    }

    const toastEl = document.createElement('div');
    toastEl.className = 'app-status-toast';
    toastEl.dataset.level = normalizedLevel;
    toastEl.setAttribute('role', normalizedLevel === 'error' ? 'alert' : 'status');
    toastEl.setAttribute('aria-atomic', 'true');

    const bodyEl = document.createElement('div');
    bodyEl.className = 'app-status-toast__body';

    const levelEl = document.createElement('span');
    levelEl.className = 'app-status-toast__level';
    levelEl.textContent = levelLabel(normalizedLevel);

    const messageEl = document.createElement('div');
    messageEl.className = 'app-status-toast__message';
    messageEl.textContent = text;

    const closeEl = document.createElement('button');
    closeEl.type = 'button';
    closeEl.className = 'app-status-toast__close';
    closeEl.setAttribute('aria-label', 'Dismiss notification');
    closeEl.textContent = '×';
    closeEl.addEventListener('click', () => {
      removeToast(key);
    });

    bodyEl.append(levelEl, messageEl);
    toastEl.append(bodyEl, closeEl);
    toastStackEl.appendChild(toastEl);
    activeToasts.set(key, { el: toastEl, messageEl, timeoutId: null });
    scheduleToastRemoval(key, duration);
    announce(text, normalizedLevel);
    return toastEl;
  }

  window.appStatus = {
    clearBanner,
    setBanner,
    showToast,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ensureInitialized, { once: true });
  } else {
    ensureInitialized();
  }
})();
