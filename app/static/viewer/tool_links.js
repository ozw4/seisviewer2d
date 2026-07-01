const ACTIVE_VIEWER_TARGET_STORAGE_KEY = 'sv.active_viewer_target';

function safeLocalStorageGet(key) {
  try {
    return localStorage.getItem(key) || '';
  } catch (_) {
    return '';
  }
}

function safeLocalStorageSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (_) {
  }
}

function safeLocalStorageRemove(key) {
  try {
    localStorage.removeItem(key);
  } catch (_) {
  }
}

function toViewerToolInteger(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === 'string' && value.trim() === '') return null;
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function normalizeViewerToolTarget(candidate, options = {}) {
  if (!candidate || typeof candidate !== 'object') return null;
  if (options.requireLoaded && candidate.isFileLoaded === false) return null;
  const fileId = String(candidate.fileId ?? candidate.file_id ?? '').trim();
  const key1Byte = toViewerToolInteger(candidate.key1Byte ?? candidate.key1_byte);
  const key2Byte = toViewerToolInteger(candidate.key2Byte ?? candidate.key2_byte);
  if (!fileId || key1Byte === null || key2Byte === null) return null;
  const displayName = String(
    candidate.displayName ?? candidate.display_name ?? candidate.fileName ?? candidate.file_name ?? fileId
  ).trim() || fileId;
  return {
    fileId,
    displayName,
    key1Byte,
    key2Byte,
    isFileLoaded: candidate.isFileLoaded !== false,
  };
}

function readViewerToolTargetFromUrl() {
  const params = new URLSearchParams(window.location.search || '');
  return normalizeViewerToolTarget({
    fileId: params.get('file_id'),
    displayName: params.get('display_name'),
    key1Byte: params.get('key1_byte'),
    key2Byte: params.get('key2_byte'),
    isFileLoaded: true,
  });
}

function readViewerToolTargetFromStorage() {
  try {
    const parsed = JSON.parse(safeLocalStorageGet(ACTIVE_VIEWER_TARGET_STORAGE_KEY) || '{}');
    return normalizeViewerToolTarget(parsed, { requireLoaded: true });
  } catch (_) {
    return null;
  }
}

function readViewerToolTargetFromLegacyStorage() {
  return normalizeViewerToolTarget({
    fileId: safeLocalStorageGet('file_id'),
    displayName: safeLocalStorageGet('last_original_name'),
    key1Byte: safeLocalStorageGet('key1_byte') || safeLocalStorageGet('last_key1_byte'),
    key2Byte: safeLocalStorageGet('key2_byte') || safeLocalStorageGet('last_key2_byte'),
    isFileLoaded: true,
  });
}

function getActiveViewerToolTarget() {
  if (window.SeisViewerState && typeof window.SeisViewerState.getActiveFileTarget === 'function') {
    const active = normalizeViewerToolTarget(window.SeisViewerState.getActiveFileTarget(), { requireLoaded: true });
    if (active) return active;
  }
  if (window.SeisViewerState && typeof window.SeisViewerState.getActiveFileTargetState === 'function') {
    const stateTarget = normalizeViewerToolTarget(window.SeisViewerState.getActiveFileTargetState());
    if (stateTarget) return stateTarget;
  }
  const globalTarget = normalizeViewerToolTarget({
    fileId: window.currentFileId,
    displayName: window.currentFileName,
    key1Byte: window.currentKey1Byte,
    key2Byte: window.currentKey2Byte,
    isFileLoaded: Boolean(window.currentFileId),
  });
  return globalTarget
    || readViewerToolTargetFromUrl()
    || readViewerToolTargetFromStorage()
    || readViewerToolTargetFromLegacyStorage();
}

window.persistActiveViewerToolTarget = function persistActiveViewerToolTarget(candidate) {
  const target = normalizeViewerToolTarget(candidate, { requireLoaded: true });
  if (!target) {
    safeLocalStorageRemove(ACTIVE_VIEWER_TARGET_STORAGE_KEY);
    return null;
  }
  safeLocalStorageSet('file_id', target.fileId);
  safeLocalStorageSet('key1_byte', String(target.key1Byte));
  safeLocalStorageSet('key2_byte', String(target.key2Byte));
  safeLocalStorageSet('last_original_name', target.displayName);
  safeLocalStorageSet('last_key1_byte', String(target.key1Byte));
  safeLocalStorageSet('last_key2_byte', String(target.key2Byte));
  safeLocalStorageSet(ACTIVE_VIEWER_TARGET_STORAGE_KEY, JSON.stringify({
    ...target,
    updatedAt: new Date().toISOString(),
  }));
  return target;
};

function buildViewerToolUrl(path) {
  const target = new URL(path, window.location.origin);
  const activeTarget = getActiveViewerToolTarget();
  if (activeTarget) {
    target.searchParams.set('file_id', activeTarget.fileId);
    target.searchParams.set('key1_byte', String(activeTarget.key1Byte));
    target.searchParams.set('key2_byte', String(activeTarget.key2Byte));
    if (activeTarget.displayName && activeTarget.displayName !== activeTarget.fileId) {
      target.searchParams.set('display_name', activeTarget.displayName);
    }
  }
  return target;
}

window.refreshViewerToolLinks = function refreshViewerToolLinks() {
  const links = [
    ['staticCorrectionLink', '/static-correction'],
    ['refractionQcLink', '/refraction-qc'],
    ['batchApplyLink', '/batch'],
  ];
  for (const [id, path] of links) {
    const link = document.getElementById(id);
    if (link) link.href = buildViewerToolUrl(path).toString();
  }
};

window.addEventListener('DOMContentLoaded', () => {
  window.refreshViewerToolLinks();
});
