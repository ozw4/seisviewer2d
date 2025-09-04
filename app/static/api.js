async function fetchFbpickSectionBin(args) {
  const res = await fetch('/fbpick_section_bin', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(args),
  });
  if (!res.ok) {
    throw new Error(await res.text());
  }
  const buf = await res.arrayBuffer();
  const u8 = new Uint8Array(buf);
  const ungz = pako.ungzip(u8);
  const obj = msgpack.decode(ungz);
  if (obj.dtype === 'u8') {
    const arr = new Uint8Array(obj.data);
    const probs = Float32Array.from(arr, (v) => v / 255);
    return { h: obj.h, w: obj.w, probs, meta: obj.meta || {} };
  }
  throw new Error('Unsupported dtype');
}
