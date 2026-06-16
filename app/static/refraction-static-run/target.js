import {
  ACTIVE_VIEWER_TARGET_STORAGE_KEY,
  NO_ACTIVE_TARGET_ERROR,
} from './constants.js';
import {
  safeLocalStorageJson,
  safeLocalStorageValue,
  searchParamValue,
  standaloneToolBaseUrl,
  trimValue,
} from './utils.js';

export function normalizeStandaloneTargetCandidate(candidate, options = {}) {
  if (!candidate || typeof candidate !== 'object') return null;
  if (options.requireLoaded && candidate.isFileLoaded === false) return null;
  const fileId = trimValue(candidate.fileId ?? candidate.file_id);
  if (!fileId) return null;
  const key1Byte = candidate.key1Byte ?? candidate.key1_byte;
  const key2Byte = candidate.key2Byte ?? candidate.key2_byte;
  const displayName = trimValue(
    candidate.displayName ?? candidate.display_name ?? candidate.fileName ?? candidate.file_name
  ) || fileId;
  return {
    fileId,
    displayName,
    key1Byte,
    key2Byte,
    isFileLoaded: candidate.isFileLoaded !== false,
  };
}
function getStoredActiveViewerTargetState() {
  return normalizeStandaloneTargetCandidate(
    safeLocalStorageJson(ACTIVE_VIEWER_TARGET_STORAGE_KEY),
    { requireLoaded: true }
  );
}
function parseTargetPositiveInteger(value, label, errors) {
  if (value === null || value === undefined) {
    errors.push(`Active viewer target is missing ${label}.`);
    return null;
  }
  if (typeof value === 'string' && value.trim() === '') {
    errors.push(`Active viewer target is missing ${label}.`);
    return null;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    errors.push(`Active viewer target ${label} must be a positive integer.`);
    return null;
  }
  return parsed;
}
export function getStandaloneStaticCorrectionTargetState() {
  const storedTarget = getStoredActiveViewerTargetState();
  const urlTarget = normalizeStandaloneTargetCandidate({
    fileId: searchParamValue('file_id'),
    displayName: searchParamValue('display_name') || (storedTarget && storedTarget.displayName),
    key1Byte: searchParamValue('key1_byte') || (storedTarget && storedTarget.key1Byte),
    key2Byte: searchParamValue('key2_byte') || (storedTarget && storedTarget.key2Byte),
    isFileLoaded: true,
  });
  if (urlTarget) return urlTarget;
  if (storedTarget) return storedTarget;

  return normalizeStandaloneTargetCandidate({
    fileId: safeLocalStorageValue('file_id'),
    displayName: safeLocalStorageValue('last_original_name'),
    key1Byte: safeLocalStorageValue('key1_byte') || safeLocalStorageValue('last_key1_byte'),
    key2Byte: safeLocalStorageValue('key2_byte') || safeLocalStorageValue('last_key2_byte'),
    isFileLoaded: true,
  });
}
export function getStaticCorrectionTargetState() {
  const viewerState = window.SeisViewerState;
  if (!viewerState || typeof viewerState.getActiveFileTarget !== 'function') {
    return getStandaloneStaticCorrectionTargetState();
  }
  const targetState = typeof viewerState.getActiveFileTargetState === 'function'
    ? viewerState.getActiveFileTargetState()
    : viewerState.getActiveFileTarget();
  return targetState || getStandaloneStaticCorrectionTargetState();
}
export function validateStaticCorrectionTarget() {
  const target = getStaticCorrectionTargetState();
  const errors = [];
  if (!target || typeof target !== 'object') {
    return { target: null, errors: [NO_ACTIVE_TARGET_ERROR] };
  }
  if (target.isFileLoaded === false) {
    return { target: null, errors: [NO_ACTIVE_TARGET_ERROR] };
  }
  const fileId = trimValue(target.fileId);
  if (!fileId) {
    errors.push('Active viewer target is missing fileId.');
  }
  const key1Byte = parseTargetPositiveInteger(target.key1Byte, 'key1Byte', errors);
  const key2Byte = parseTargetPositiveInteger(target.key2Byte, 'key2Byte', errors);
  if (errors.length) {
    return { target: null, errors };
  }
  const displayName = trimValue(target.displayName) || fileId;
  return {
    target: {
      file_id: fileId,
      display_name: displayName,
      key1_byte: key1Byte,
      key2_byte: key2Byte,
    },
    errors,
  };
}
export function getStaticCorrectionTarget() {
  return validateStaticCorrectionTarget().target;
}
export function staticCorrectionTargetKey(target) {
  if (!target) return '';
  return `${trimValue(target.file_id)}:${target.key1_byte}:${target.key2_byte}`;
}
export function sameStaticCorrectionTarget(left, right) {
  if (!left || !right) return false;
  return (
    trimValue(left.file_id) === trimValue(right.file_id)
    && Number(left.key1_byte) === Number(right.key1_byte)
    && Number(left.key2_byte) === Number(right.key2_byte)
  );
}
export function buildRefractionQcUrl(jobId) {
  const url = new URL('/refraction-qc', standaloneToolBaseUrl());
  const safeJobId = trimValue(jobId);
  if (safeJobId) url.searchParams.set('refraction_job_id', safeJobId);
  const target = getStaticCorrectionTarget();
  if (target) {
    url.searchParams.set('file_id', target.file_id);
    url.searchParams.set('key1_byte', String(target.key1_byte));
    url.searchParams.set('key2_byte', String(target.key2_byte));
  }
  return url.toString();
}
