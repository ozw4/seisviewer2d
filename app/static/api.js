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
