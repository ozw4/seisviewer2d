(function () {
  'use strict';

  class CompareFetchError extends Error {
    constructor(source, status, detail) {
      const label = source && source.role === 'A' ? 'A' : 'B';
      const suffix = detail ? `: ${detail}` : '';
      super(`${label} source fetch failed (${status})${suffix}`);
      this.name = 'CompareFetchError';
      this.source = source;
      this.status = status;
      this.detail = detail || '';
    }
  }

  function resolveFetchImpl(fetchImpl) {
    if (typeof fetchImpl === 'function') return fetchImpl;
    if (typeof window.fetch === 'function') return window.fetch.bind(window);
    throw new Error('Fetch is not available.');
  }

  async function readCompareResponseDetail(response) {
    try {
      const contentType = response.headers?.get?.('content-type') || '';
      if (contentType.includes('application/json')) {
        const json = await response.json();
        return typeof json?.detail === 'string' ? json.detail : '';
      }
      return await response.text();
    } catch (_) {
      return '';
    }
  }

  async function loadCompareRecentDatasets({ fetchImpl } = {}) {
    const runFetch = resolveFetchImpl(fetchImpl);
    const response = await runFetch('/recent_datasets');
    if (!response.ok) throw new Error(`Recent datasets request failed (${response.status})`);
    const payload = await response.json();
    return Array.isArray(payload?.datasets) ? payload.datasets : [];
  }

  async function validateRawCompareSources({
    sources,
    key1Byte,
    key2Byte,
    signal,
    fetchImpl,
  }) {
    const runFetch = resolveFetchImpl(fetchImpl);
    const params = new URLSearchParams({
      file_id_a: String(sources.a.fileId),
      file_id_b: String(sources.b.fileId),
      key1_byte: String(key1Byte),
      key2_byte: String(key2Byte),
    });
    const response = await runFetch(`/compare/raw/validate?${params.toString()}`, { signal });
    if (!response.ok) {
      const detail = await readCompareResponseDetail(response);
      throw new Error(detail || `A-B validation failed (${response.status})`);
    }
    const payload = await response.json();
    const result = {
      ok: payload?.ok === true,
      reason: String(payload?.reason || ''),
      message: String(payload?.message || ''),
    };
    if (!result.ok && !result.message) {
      result.message = 'A-B unavailable: raw source grids are different.';
    }
    return result;
  }

  async function ensureRawCompareReferenceBaseline({
    sourceA,
    key1Byte,
    key2Byte,
    signal,
    fetchImpl,
  }) {
    const runFetch = resolveFetchImpl(fetchImpl);
    const params = new URLSearchParams({
      file_id: String(sourceA.fileId),
      key1_byte: String(key1Byte),
      key2_byte: String(key2Byte),
    });
    const response = await runFetch(`/get_section_meta?${params.toString()}`, { signal });
    if (!response.ok) {
      const detail = await readCompareResponseDetail(response);
      throw new Error(detail || `A normalization baseline is unavailable (${response.status})`);
    }
    return true;
  }

  async function fetchComparePayload({
    request,
    signal,
    requestId,
    hooks,
    fetchImpl,
  }) {
    const runFetch = resolveFetchImpl(fetchImpl);
    const state = hooks || {};
    if (!state.isRequestCurrent?.(requestId)) {
      state.markStale?.(requestId);
      return null;
    }
    if (!state.isLmoCurrent?.(request.payloadMeta?.lmoKey)) return null;
    const cached = state.cacheGet?.(request.cacheKey);
    if (
      cached
      && state.isLmoCurrent?.(cached.lmoKey)
      && state.canUseCachedPayload?.(cached, request.source)
    ) {
      if (!state.isRequestCurrent?.(requestId)) {
        state.markStale?.(requestId);
        return null;
      }
      return cached;
    }

    const response = await runFetch(`/get_section_window_bin?${request.params.toString()}`, { signal });
    if (!response.ok) {
      const detail = await readCompareResponseDetail(response);
      throw new CompareFetchError(request.source, response.status, detail);
    }
    const buf = await response.arrayBuffer();
    if (!state.isRequestCurrent?.(requestId)) {
      state.markStale?.(requestId);
      return null;
    }
    if (!state.isLmoCurrent?.(request.payloadMeta?.lmoKey)) return null;
    const payload = state.decodePayload?.(
      new Uint8Array(buf),
      request.payloadMeta,
      null,
      state.onUnexpectedShape,
    );
    if (!payload) return null;
    if (!state.isRequestCurrent?.(requestId)) {
      state.markStale?.(requestId);
      return null;
    }
    if (!state.isLmoCurrent?.(payload.lmoKey)) return null;
    state.cacheSet?.(request.cacheKey, payload);
    return payload;
  }

  async function postCompareBSourceImport({
    file,
    key1Byte,
    key2Byte,
    fetchImpl,
  }) {
    const runFetch = resolveFetchImpl(fetchImpl);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('key1_byte', String(key1Byte));
    formData.append('key2_byte', String(key2Byte));

    const response = await runFetch('/compare/raw/import', { method: 'POST', body: formData });
    if (!response.ok) {
      const detail = await readCompareResponseDetail(response);
      throw new Error(detail || `Import B source failed (${response.status})`);
    }
    return response.json();
  }

  async function importCompareBSourceFile({
    file,
    activeTarget,
    keyBytes,
    fetchImpl,
  }) {
    return postCompareBSourceImport({
      file,
      key1Byte: keyBytes?.key1Byte ?? activeTarget?.key1Byte,
      key2Byte: keyBytes?.key2Byte ?? activeTarget?.key2Byte,
      fetchImpl,
    });
  }

  window.__svCompareApi = Object.freeze({
    CompareFetchError,
    readCompareResponseDetail,
    loadCompareRecentDatasets,
    validateRawCompareSources,
    ensureRawCompareReferenceBaseline,
    fetchComparePayload,
    postCompareBSourceImport,
    importCompareBSourceFile,
  });
})();
