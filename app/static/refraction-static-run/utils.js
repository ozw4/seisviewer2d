export function trimValue(value) {
  return String(value || '').trim();
}

export function safeLocalStorageValue(key) {
  try {
    return window.localStorage.getItem(key) || '';
  } catch (_) {
    return '';
  }
}

export function safeLocalStorageJson(key) {
  try {
    const value = window.localStorage.getItem(key);
    if (!value) return null;
    return JSON.parse(value);
  } catch (_) {
    return null;
  }
}

export function searchParamValue(key) {
  try {
    const params = new URLSearchParams(window.location.search || '');
    return params.get(key) || '';
  } catch (_) {
    return '';
  }
}

export function searchOrStorageValue(searchKey, storageKey, fallback = '') {
  return searchParamValue(searchKey) || safeLocalStorageValue(storageKey || searchKey) || fallback;
}

export function standaloneToolBaseUrl() {
  try {
    if (window.location && window.location.origin && window.location.origin !== 'null') {
      return window.location.origin;
    }
  } catch (_) {
  }
  return 'http://localhost';
}

export function setDefaultValue(element, value) {
  if (element && trimValue(element.value) === '') {
    element.value = value;
  }
}

export function setElementValue(element, value) {
  if (element && value !== undefined && value !== null) {
    element.value = String(value);
  }
}

export function setElementChecked(element, value) {
  if (element && value !== undefined && value !== null) {
    element.checked = Boolean(value);
  }
}

export function setHidden(element, hidden) {
  if (element) {
    element.hidden = hidden;
  }
}

export function setDisabled(elements, disabled) {
  for (const element of elements) {
    if (element) {
      element.disabled = disabled;
    }
  }
}

export function formatSavedAt(value) {
  const date = new Date(value || '');
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString([], {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export async function readResponseError(response, operation = 'request') {
  let detail = '';
  try {
    const contentType = response.headers && response.headers.get
      ? response.headers.get('content-type') || ''
      : '';
    if (contentType.includes('application/json')) {
      const payload = await response.json();
      detail = payload && typeof payload.detail === 'string'
        ? payload.detail
        : JSON.stringify(payload);
    } else {
      detail = await response.text();
    }
  } catch {
    detail = '';
  }
  return `${operation} ${response.status}${detail ? `: ${detail}` : ''}`;
}

export function delay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
