function makeZRowMajor(float1d, h, w) {
  const z = new Array(h);
  for (let i = 0; i < h; i++) {
    const base = i * w;
    const row = new Array(w);
    for (let j = 0; j < w; j++) row[j] = float1d[base + j];
    z[i] = row;
  }
  return z;
}

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
    const probs = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) probs[i] = arr[i] / 255;
    const z = makeZRowMajor(probs, obj.h, obj.w);
    return { h: obj.h, w: obj.w, probs, z, meta: obj.meta || {} };
  }
  throw new Error('Unsupported dtype');
}
