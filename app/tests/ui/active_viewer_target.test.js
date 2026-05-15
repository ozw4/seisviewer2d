import { expect, test } from 'vitest';

import {
  createSeisViewerState,
  createStore,
  getActiveFileTargetFromState,
} from '../../static/viewer/store.js';

test('test_active_viewer_target_empty_before_file_load', () => {
  const store = createStore({
    fileId: '',
    displayName: '',
    key1Byte: 9,
    key2Byte: 13,
    isFileLoaded: false,
  });
  const viewerState = createSeisViewerState(store);

  expect(viewerState.getActiveFileTarget()).toBeNull();
  expect(getActiveFileTargetFromState({
    fileId: 'pending-file',
    displayName: '',
    key1Byte: 9,
    key2Byte: 13,
    isFileLoaded: false,
  })).toBeNull();
});

test('test_active_viewer_target_contains_file_id_and_sort_keys_after_load', () => {
  const store = createStore({
    fileId: 'file-a',
    displayName: 'LineA.sgy',
    key1Byte: '9',
    key2Byte: '13',
    isFileLoaded: true,
  });
  const viewerState = createSeisViewerState(store);

  expect(viewerState.getActiveFileTargetState()).toEqual({
    fileId: 'file-a',
    displayName: 'LineA.sgy',
    key1Byte: '9',
    key2Byte: '13',
    isFileLoaded: true,
  });
  expect(viewerState.getActiveFileTarget()).toEqual({
    fileId: 'file-a',
    displayName: 'LineA.sgy',
    key1Byte: 9,
    key2Byte: 13,
  });
});

test('test_active_viewer_target_rejects_null_sort_key_bytes', () => {
  expect(getActiveFileTargetFromState({
    fileId: 'file-a',
    displayName: 'LineA.sgy',
    key1Byte: null,
    key2Byte: 13,
    isFileLoaded: true,
  })).toBeNull();

  expect(getActiveFileTargetFromState({
    fileId: 'file-a',
    displayName: 'LineA.sgy',
    key1Byte: 9,
    key2Byte: null,
    isFileLoaded: true,
  })).toBeNull();
});

test('test_active_viewer_target_updates_when_new_file_loaded', () => {
  const store = createStore({
    fileId: 'file-a',
    displayName: 'LineA.sgy',
    key1Byte: 9,
    key2Byte: 13,
    isFileLoaded: true,
  });
  const viewerState = createSeisViewerState(store);

  store.patch({
    fileId: 'file-b',
    displayName: 'LineB.sgy',
    key1Byte: 21,
    key2Byte: 25,
    isFileLoaded: true,
  });

  expect(viewerState.getActiveFileTarget()).toEqual({
    fileId: 'file-b',
    displayName: 'LineB.sgy',
    key1Byte: 21,
    key2Byte: 25,
  });
});
