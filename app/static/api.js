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
  key1Idx,
  spec,
  taps,
  { key1Byte = 189, key2Byte = 193 } = {}
) {
  const url = `/pipeline/section?file_id=${encodeURIComponent(fileId)}&key1_idx=${key1Idx}&key1_byte=${key1Byte}&key2_byte=${key2Byte}`;
  const r = await fetch(url, {
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
  cacheSet(`${fileId}:${key1Idx}:${json.pipeline_key}`, out);
  return { taps: out, pipelineKey: json.pipeline_key };
}
