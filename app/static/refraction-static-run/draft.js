import { STATIC_DRAFT_STORAGE_KEY } from './constants.js';
import { collectPresetInputs, selectedPickNpzFile } from './form_collectors.js';
import {
  deletePickNpzFromIndexedDb,
  loadPickNpzFromIndexedDb,
  pickRecordMetadata,
} from './picks_db.js';
import { applyPresetValues } from './presets.js';
import { getStaticCorrectionDom, requestStaticCorrectionRender } from './runtime.js';
import { state } from './state.js';
import {
  getStaticCorrectionTarget,
  sameStaticCorrectionTarget,
  staticCorrectionTargetKey,
} from './target.js';
import { safeLocalStorageJson } from './utils.js';

export function readStaticCorrectionDraft() {
  const draft = safeLocalStorageJson(STATIC_DRAFT_STORAGE_KEY);
  if (!draft || draft.version !== 1 || !draft.form) return null;
  return draft;
}
function activeTargetMatchesPickMeta(meta, target) {
  if (!meta || !target) return false;
  return sameStaticCorrectionTarget(
    { file_id: meta.fileId, key1_byte: meta.key1Byte, key2_byte: meta.key2Byte },
    target
  );
}
export function saveStaticCorrectionDraft(options = {}) {
  const dom = getStaticCorrectionDom();
  if (!dom) return null;
  if (state.draftCleared) return null;
  const target = getStaticCorrectionTarget();
  if (!target) return null;
  const pickMeta = options.pickNpz !== undefined ? options.pickNpz : state.pickNpzDraftMeta;
  const draft = {
    version: 1,
    target: {
      file_id: target.file_id,
      key1_byte: target.key1_byte,
      key2_byte: target.key2_byte,
    },
    pickNpz: activeTargetMatchesPickMeta(pickMeta, target) ? pickMeta : null,
    form: collectPresetInputs(dom),
  };
  try {
    window.localStorage.setItem(STATIC_DRAFT_STORAGE_KEY, JSON.stringify(draft));
  } catch (_) {
    return null;
  }
  return draft;
}
function clearStoredDraftPickNpz() {
  const draft = readStaticCorrectionDraft();
  if (!draft) return;
  draft.pickNpz = null;
  try {
    window.localStorage.setItem(STATIC_DRAFT_STORAGE_KEY, JSON.stringify(draft));
  } catch (_) {
  }
}
function clearPickInput(targetDom = getStaticCorrectionDom()) {
  if (!targetDom || !targetDom.pickNpzFile) return;
  targetDom.pickNpzFile.value = '';
  try {
    const dt = new DataTransfer();
    targetDom.pickNpzFile.files = dt.files;
  } catch (_) {
    // Browsers that do not allow assigning an empty FileList still clear value.
  }
}
function setPickInputFile(targetDom, file) {
  if (!targetDom || !targetDom.pickNpzFile || !file) return false;
  try {
    const dt = new DataTransfer();
    dt.items.add(file);
    targetDom.pickNpzFile.files = dt.files;
    return targetDom.pickNpzFile.files && targetDom.pickNpzFile.files.length === 1;
  } catch (_) {
    return false;
  }
}
function showRestoredPickNpz(file, record) {
  state.pickNpzRestoreStatus = 'restored';
  state.pickNpzRestoreMessage = file && file.name ? file.name : record.filename;
  state.pickNpzRestoredSavedAt = record.savedAt || '';
}
function showPickRestoreWarning(message) {
  state.pickNpzRestoreStatus = 'warning';
  state.pickNpzRestoreMessage = message;
  state.pickNpzRestoredSavedAt = '';
}
async function restorePickNpzFileInput(targetDom, draft, activeTarget) {
  const meta = draft && draft.pickNpz;
  if (!meta || !meta.indexedDbRecordId || !targetDom || !targetDom.pickNpzFile) return null;
  if (!sameStaticCorrectionTarget(draft.target, activeTarget)) {
    showPickRestoreWarning('Saved NPZ belongs to a different viewer target. Replace or clear the saved NPZ.');
    return null;
  }
  const record = await loadPickNpzFromIndexedDb(meta.indexedDbRecordId);
  if (!record || !record.blob) {
    showPickRestoreWarning('Saved NPZ is no longer available. Please select the NPZ again.');
    return null;
  }
  const file = record.blob instanceof File
    ? record.blob
    : new File([record.blob], record.filename || meta.filename || 'first_breaks.npz', {
      type: record.type || meta.type || 'application/octet-stream',
      lastModified: record.lastModified || meta.lastModified || Date.now(),
    });
  const restored = setPickInputFile(targetDom, file);
  if (!restored) {
    showPickRestoreWarning('Saved NPZ could not be restored into the file input. Please select it again.');
    return null;
  }
  state.restoringPickInput = true;
  targetDom.pickNpzFile.dispatchEvent(new Event('change', { bubbles: true }));
  state.restoringPickInput = false;
  state.pickNpzDraftMeta = pickRecordMetadata(record, meta.indexedDbRecordId);
  showRestoredPickNpz(file, record);
  return file;
}
export async function restoreStaticCorrectionDraftIfAvailable() {
  const dom = getStaticCorrectionDom();
  if (!dom) return null;
  const activeTarget = getStaticCorrectionTarget();
  if (!activeTarget) return null;
  const targetKey = staticCorrectionTargetKey(activeTarget);
  if (state.draftRestoreAttemptedForTarget === targetKey) return null;
  state.draftRestoreAttemptedForTarget = targetKey;
  const draft = readStaticCorrectionDraft();
  if (!draft) return null;
  if (!sameStaticCorrectionTarget(draft.target, activeTarget)) {
    state.pickNpzDraftMeta = draft.pickNpz || null;
    showPickRestoreWarning('Saved NPZ belongs to a different viewer target. Replace or clear the saved NPZ.');
    requestStaticCorrectionRender();
    return null;
  }
  state.suppressPickChangeHandler = true;
  applyPresetValues(draft.form, dom);
  state.suppressPickChangeHandler = false;
  state.pickNpzDraftMeta = draft.pickNpz || null;
  try {
    await restorePickNpzFileInput(dom, draft, activeTarget);
  } catch (error) {
    showPickRestoreWarning(error instanceof Error ? error.message : String(error));
  }
  state.message = selectedPickNpzFile(dom)
    ? 'Restored Static Correction draft and pick NPZ.'
    : 'Restored Static Correction form draft.';
  requestStaticCorrectionRender();
  return draft;
}
export async function clearRestoredPickNpz() {
  const dom = getStaticCorrectionDom();
  if (!dom) return;
  const meta = state.pickNpzDraftMeta || (readStaticCorrectionDraft() || {}).pickNpz;
  state.suppressPickChangeHandler = true;
  clearPickInput(dom);
  dom.pickNpzFile.dispatchEvent(new Event('change', { bubbles: true }));
  state.suppressPickChangeHandler = false;
  state.pickNpzDraftMeta = null;
  state.pickNpzRestoreStatus = '';
  state.pickNpzRestoreMessage = '';
  state.draftCleared = true;
  if (meta && meta.indexedDbRecordId) {
    try {
      await deletePickNpzFromIndexedDb(meta.indexedDbRecordId);
    } catch (_) {
    }
  }
  saveStaticCorrectionDraft({ pickNpz: null });
  clearStoredDraftPickNpz();
  state.error = '';
  state.message = 'Cleared restored NPZ.';
  requestStaticCorrectionRender();
}
export async function clearStaticCorrectionDraft() {
  const dom = getStaticCorrectionDom();
  const draft = readStaticCorrectionDraft();
  const recordId = draft && draft.pickNpz && draft.pickNpz.indexedDbRecordId;
  state.draftCleared = true;
  state.suppressPickChangeHandler = true;
  clearPickInput(dom);
  if (dom && dom.pickNpzFile) {
    dom.pickNpzFile.dispatchEvent(new Event('change', { bubbles: true }));
  }
  state.suppressPickChangeHandler = false;
  state.pickNpzDraftMeta = null;
  state.pickNpzRestoreStatus = '';
  state.pickNpzRestoreMessage = '';
  try {
    window.localStorage.removeItem(STATIC_DRAFT_STORAGE_KEY);
  } catch (_) {
  }
  if (recordId) {
    try {
      await deletePickNpzFromIndexedDb(recordId);
    } catch (_) {
    }
  }
  state.error = '';
  state.message = 'Cleared Static Correction draft.';
  requestStaticCorrectionRender();
}
