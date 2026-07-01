import { afterEach, beforeAll, expect, test, vi } from 'vitest';

beforeAll(async () => {
  await import('../../static/viewer/compare/api.js');
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function okJson(payload = {}) {
  return {
    ok: true,
    status: 200,
    headers: { get: () => 'application/json' },
    json: async () => payload,
  };
}

function errorJson(status, detail) {
  return {
    ok: false,
    status,
    headers: { get: () => 'application/json' },
    json: async () => ({ detail }),
  };
}

function errorText(status, text) {
  return {
    ok: false,
    status,
    headers: { get: () => 'text/plain' },
    text: async () => text,
  };
}

function comparePayloadHooks(overrides = {}) {
  return {
    isRequestCurrent: vi.fn(() => true),
    markStale: vi.fn(),
    isLmoCurrent: vi.fn(() => true),
    cacheGet: vi.fn(() => null),
    cacheSet: vi.fn(),
    canUseCachedPayload: vi.fn(() => true),
    decodePayload: vi.fn(() => ({ lmoKey: 'lmo:off' })),
    onUnexpectedShape: vi.fn(),
    ...overrides,
  };
}

function comparePayloadRequest() {
  return {
    source: { role: 'B', layerId: 'raw', fileId: 'file-b' },
    params: new URLSearchParams({
      file_id: 'file-b',
      normalization_file_id: 'file-a',
    }),
    cacheKey: 'cache:file-b:file-a',
    payloadMeta: { lmoKey: 'lmo:off' },
  };
}

test('raw validation sends A/B file ids and key bytes', async () => {
  const fetchImpl = vi.fn(async () => okJson({ ok: true, reason: '', message: '' }));

  await window.__svCompareApi.validateRawCompareSources({
    sources: {
      a: { fileId: 'file-a' },
      b: { fileId: 'file-b' },
    },
    key1Byte: 189,
    key2Byte: 193,
    fetchImpl,
  });

  expect(fetchImpl).toHaveBeenCalledTimes(1);
  const url = new URL(fetchImpl.mock.calls[0][0], 'http://localhost');
  expect(url.pathname).toBe('/compare/raw/validate');
  expect(url.searchParams.get('file_id_a')).toBe('file-a');
  expect(url.searchParams.get('file_id_b')).toBe('file-b');
  expect(url.searchParams.get('key1_byte')).toBe('189');
  expect(url.searchParams.get('key2_byte')).toBe('193');
});

test('response detail reader returns json and text details for status messages', async () => {
  await expect(window.__svCompareApi.readCompareResponseDetail(errorJson(400, 'json detail')))
    .resolves.toBe('json detail');
  await expect(window.__svCompareApi.readCompareResponseDetail(errorText(400, 'text detail')))
    .resolves.toBe('text detail');
});

test('raw reference baseline preflight fetches section meta for A source only', async () => {
  const fetchImpl = vi.fn(async () => okJson({ shape: [1, 1] }));

  await window.__svCompareApi.ensureRawCompareReferenceBaseline({
    sourceA: { fileId: 'file-a' },
    key1Byte: 189,
    key2Byte: 193,
    fetchImpl,
  });

  expect(fetchImpl).toHaveBeenCalledTimes(1);
  const url = new URL(fetchImpl.mock.calls[0][0], 'http://localhost');
  expect(url.pathname).toBe('/get_section_meta');
  expect(url.searchParams.get('file_id')).toBe('file-a');
  expect(url.searchParams.get('file_id')).not.toBe('file-b');
  expect(url.searchParams.get('key1_byte')).toBe('189');
  expect(url.searchParams.get('key2_byte')).toBe('193');
});

test('window payload fetch turns non-ok response into CompareFetchError', async () => {
  const fetchImpl = vi.fn(async () => errorJson(409, 'raw window unavailable'));

  await expect(window.__svCompareApi.fetchComparePayload({
    request: comparePayloadRequest(),
    requestId: 7,
    hooks: comparePayloadHooks(),
    fetchImpl,
  })).rejects.toMatchObject({
    name: 'CompareFetchError',
    status: 409,
    detail: 'raw window unavailable',
  });
});

test('B source import includes active key bytes in FormData', async () => {
  const fetchImpl = vi.fn(async () => okJson({ file_id: 'imported-file' }));

  await window.__svCompareApi.postCompareBSourceImport({
    file: new File(['sgy'], 'line-b.sgy'),
    key1Byte: 189,
    key2Byte: 193,
    fetchImpl,
  });

  expect(fetchImpl).toHaveBeenCalledTimes(1);
  expect(fetchImpl.mock.calls[0][0]).toBe('/compare/raw/import');
  const init = fetchImpl.mock.calls[0][1];
  expect(init.method).toBe('POST');
  expect(init.body).toBeInstanceOf(FormData);
  expect(init.body.get('key1_byte')).toBe('189');
  expect(init.body.get('key2_byte')).toBe('193');
  expect(init.body.get('file')).toBeInstanceOf(File);
});

test('B source import failure preserves backend error detail', async () => {
  const fetchImpl = vi.fn(async () => errorText(400, 'bad SEG-Y file'));

  await expect(window.__svCompareApi.postCompareBSourceImport({
    file: new File(['bad'], 'bad.sgy'),
    key1Byte: 189,
    key2Byte: 193,
    fetchImpl,
  })).rejects.toThrow('bad SEG-Y file');
});

test('fetch abort is still surfaced as AbortError', async () => {
  const abortError = new DOMException('The operation was aborted.', 'AbortError');
  const fetchImpl = vi.fn(async () => {
    throw abortError;
  });

  await expect(window.__svCompareApi.fetchComparePayload({
    request: comparePayloadRequest(),
    requestId: 7,
    signal: new AbortController().signal,
    hooks: comparePayloadHooks(),
    fetchImpl,
  })).rejects.toBe(abortError);
});
