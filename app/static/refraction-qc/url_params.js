export function safeLocalStorageValue(key, storage = localStorage) {
  try {
    return storage.getItem(key) || '';
  } catch (_) {
    return '';
  }
}

export function safeLocalStorageJson(key, storage = localStorage) {
  try {
    const raw = storage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch (_) {
    return null;
  }
}

export function searchParamValue(key, location = window.location) {
  try {
    const params = new URLSearchParams(location.search || '');
    return params.get(key) || '';
  } catch (_) {
    return '';
  }
}

export function searchOrStorageValue(searchKey, storageKey, fallback = '') {
  return searchParamValue(searchKey) || safeLocalStorageValue(storageKey || searchKey) || fallback;
}

export function writeJobIdUrlParam(jobId) {
  if (!window.history || !window.location) return;
  try {
    const url = new URL(window.location.href);
    url.searchParams.set('refraction_job_id', jobId);
    url.searchParams.delete('refraction_qc_job_id');
    window.history.replaceState(window.history.state, '', url);
  } catch (_) {
  }
}
