import {
  STATIC_PICK_DB_NAME,
  STATIC_PICK_DB_VERSION,
  STATIC_PICK_MAX_AGE_MS,
  STATIC_PICK_MAX_RECORDS,
  STATIC_PICK_STORE,
} from './constants.js';

export function staticPickRecordId(target) {
  if (!target) return '';
  return `target:${target.file_id}:${target.key1_byte}:${target.key2_byte}:latest`;
}
function openStaticPickDb() {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      reject(new Error('IndexedDB is not available in this browser.'));
      return;
    }
    const request = window.indexedDB.open(STATIC_PICK_DB_NAME, STATIC_PICK_DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STATIC_PICK_STORE)) {
        db.createObjectStore(STATIC_PICK_STORE, { keyPath: 'id' });
      }
    };
    request.onerror = () => reject(request.error || new Error('Failed to open IndexedDB.'));
    request.onsuccess = () => resolve(request.result);
  });
}
function withStaticPickStore(mode, callback) {
  return openStaticPickDb().then((db) => new Promise((resolve, reject) => {
    const tx = db.transaction(STATIC_PICK_STORE, mode);
    const store = tx.objectStore(STATIC_PICK_STORE);
    let callbackResult;
    tx.oncomplete = () => {
      db.close();
      resolve(callbackResult);
    };
    tx.onerror = () => {
      db.close();
      reject(tx.error || new Error('IndexedDB transaction failed.'));
    };
    try {
      callbackResult = callback(store);
    } catch (error) {
      db.close();
      reject(error);
    }
  }));
}
function requestToPromise(request) {
  return new Promise((resolve, reject) => {
    request.onerror = () => reject(request.error || new Error('IndexedDB request failed.'));
    request.onsuccess = () => resolve(request.result);
  });
}
async function putIndexedDbRecord(record) {
  await withStaticPickStore('readwrite', (store) => {
    store.put(record);
  });
}
export async function loadPickNpzFromIndexedDb(recordId) {
  if (!recordId) return null;
  return withStaticPickStore('readonly', (store) => requestToPromise(store.get(recordId)));
}
export async function deletePickNpzFromIndexedDb(recordId) {
  if (!recordId) return;
  await withStaticPickStore('readwrite', (store) => {
    store.delete(recordId);
  });
}
async function cleanupStaticPickIndexedDb() {
  try {
    await withStaticPickStore('readwrite', (store) => {
      const request = store.getAll();
      request.onsuccess = () => {
        const records = Array.isArray(request.result) ? request.result : [];
        const now = Date.now();
        const sorted = records
          .filter((record) => record && record.id)
          .sort((left, right) => Date.parse(right.savedAt || '') - Date.parse(left.savedAt || ''));
        sorted.forEach((record, index) => {
          const savedAtMs = Date.parse(record.savedAt || '');
          const isExpired = Number.isFinite(savedAtMs) && now - savedAtMs > STATIC_PICK_MAX_AGE_MS;
          if (isExpired || index >= STATIC_PICK_MAX_RECORDS) {
            store.delete(record.id);
          }
        });
      };
    });
  } catch (_) {
    // Cleanup should not block normal Static Correction use.
  }
}
export async function savePickNpzToIndexedDb(recordId, file, target) {
  const savedAt = new Date().toISOString();
  const record = {
    id: recordId,
    filename: file.name,
    sizeBytes: file.size,
    type: file.type || 'application/octet-stream',
    lastModified: file.lastModified || Date.now(),
    savedAt,
    fileId: target.file_id,
    key1Byte: target.key1_byte,
    key2Byte: target.key2_byte,
    blob: file,
  };
  await putIndexedDbRecord(record);
  cleanupStaticPickIndexedDb();
  return record;
}
export function pickRecordMetadata(record, recordId = '') {
  if (!record) return null;
  return {
    indexedDbRecordId: record.id || recordId,
    filename: record.filename || 'first_breaks.npz',
    sizeBytes: Number(record.sizeBytes) || 0,
    type: record.type || 'application/octet-stream',
    lastModified: record.lastModified || Date.now(),
    savedAt: record.savedAt || new Date().toISOString(),
    fileId: record.fileId,
    key1Byte: record.key1Byte,
    key2Byte: record.key2Byte,
  };
}
