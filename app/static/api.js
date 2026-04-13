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
  { key1Byte = 189, key2Byte = 193, signal } = {}
) {
  const url = new URL('/pipeline/section', location.origin);
  url.searchParams.set('file_id', fileId);
  url.searchParams.set('key1', String(key1Val));
  url.searchParams.set('key1_byte', String(key1Byte));
  url.searchParams.set('key2_byte', String(key2Byte));
  url.searchParams.set('list_only', '1');
  const r = await fetch(url.toString(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({ spec, taps }),
  });
  if (!r.ok) {
    let detail = '';
    try {
      const contentType = r.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        const payload = await r.json();
        detail = payload && typeof payload.detail === 'string' ? payload.detail : '';
      } else {
        detail = await r.text();
      }
    } catch {
      detail = '';
    }
    const suffix = detail ? `: ${detail}` : '';
    throw new Error(`pipeline/section ${r.status}${suffix}`);
  }
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
