export function getStorageValue(key, fallback = '', storage = window.localStorage) {
  try {
    return storage.getItem(key) || fallback;
  } catch (_error) {
    return fallback;
  }
}

export function setStorageValue(key, value, storage = window.localStorage) {
  try {
    storage.setItem(key, value);
    return true;
  } catch (_error) {
    return false;
  }
}

export function removeStorageValue(key, storage = window.localStorage) {
  try {
    storage.removeItem(key);
    return true;
  } catch (_error) {
    return false;
  }
}
