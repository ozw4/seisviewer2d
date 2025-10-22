async function fetchFbpickSectionBin(body) {
  const resp = await fetch('/fbpick_section_bin', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    // サーバ側のエラー内容をそのまま表示できるように
    const text = await resp.text().catch(() => '');
    throw new Error(`HTTP ${resp.status} ${resp.statusText} — ${text}`);
  }

  const buf = await resp.arrayBuffer();
  let u8 = new Uint8Array(buf);

  // --- gzip ヘッダをスニフ（0x1f, 0x8b） ---
  const isGzip = u8.length >= 2 && u8[0] === 0x1f && u8[1] === 0x8b;
  if (isGzip) {
    // 本当に gzip のときだけ解凍
    u8 = pako.ungzip(u8);
  }
  // ここまでで u8 は msgpack の生バイト
  const payload = msgpack.decode(u8);

  if (payload.dtype !== 'u8') {
    throw new Error(`Unexpected dtype: ${payload.dtype}`);
  }
  const { h, w, data, meta } = payload;
  const bytes = new Uint8Array(data);
  const probs = new Float32Array(h * w);
  for (let i = 0; i < bytes.length; i++) probs[i] = bytes[i] / 255.0;

  return { h, w, probs, meta };
}

const _pipelineSectionCache = new Map();

function cacheSet(key, value) {
  _pipelineSectionCache.set(key, value);
}

function cacheGet(key) {
  return _pipelineSectionCache.get(key);
}

function normalizeTapValue(value) {
  let mat = value;
  if (value && typeof value === 'object') {
    if (value.data) {
      mat = value.data;
    } else if (value.prob) {
      mat = value.prob;
    }
  }
  return Array.isArray(mat)
    ? mat.map((row) => Float32Array.from(row))
    : mat;
}

async function fetchSectionWithPipeline(
  fileId,
  key1Val,
  spec,
  taps,
  { key1Byte = 189, key2Byte = 193 } = {}
) {
  const url = new URL('/pipeline/section', location.origin);
  url.searchParams.set('file_id', fileId);
  url.searchParams.set('key1_val', String(key1Val));
  url.searchParams.set('key1_byte', String(key1Byte));
  url.searchParams.set('key2_byte', String(key2Byte));
  url.searchParams.set('list_only', '1');
  const r = await fetch(url.toString(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ spec, taps }),
  });
  if (!r.ok) throw new Error(`pipeline/section ${r.status}`);
  const json = await r.json();
  const out = {};
  for (const [name, val] of Object.entries(json.taps || {})) {
    try {
      out[name] = normalizeTapValue(val);
    } catch (e) {
      console.warn('normalizeTapValue failed', name, e);
    }
  }
  cacheSet(`${fileId}:${key1Val}:${json.pipeline_key}`, out);
  return { taps: out, pipelineKey: json.pipeline_key };
}

async function fetchRawBaselineStats({
  fileId,
  key1Val,
  key1Byte = 189,
  key2Byte = 193,
} = {}) {
  const url = new URL('/section/stats', location.origin);
  url.searchParams.set('file_id', fileId);
  url.searchParams.set('key1_val', String(key1Val));
  url.searchParams.set('baseline', 'raw');
  url.searchParams.set('key1_byte', String(key1Byte));
  url.searchParams.set('key2_byte', String(key2Byte));

  const resp = await fetch(url.toString());
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`section/stats ${resp.status} ${resp.statusText} — ${text}`);
  }

  const json = await resp.json();
  const stats = { ...json };
  stats.mu_traces = Float32Array.from(json.mu_traces ?? []);
  stats.sigma_traces = Float32Array.from(json.sigma_traces ?? []);
  const maskArray = Array.isArray(json.zero_var_mask)
    ? json.zero_var_mask
    : [];
  stats.zero_var_mask = Uint8Array.from(maskArray, (v) => (v ? 1 : 0));
  stats.mu_section = Number(json.mu_section ?? 0);
  stats.sigma_section = Number(json.sigma_section ?? 1);
  return stats;
}

window.fetchRawBaselineStats = fetchRawBaselineStats;
