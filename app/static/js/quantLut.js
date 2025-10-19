// app/static/js/quantLut.js
export function buildLinearLUT(lo, hi) {
  const lut = new Float32Array(256);
  const s = (hi - lo) / 255.0, offs = lo;
  for (let i = 0; i < 256; i++) lut[i] = offs + i * s;
  return lut;
}

export function buildMulawLUT(lo, hi, mu = 255) {
  const lut = new Float32Array(256);
  const ln = Math.log(1 + mu);
  for (let i = 0; i < 256; i++) {
    const q = i / 255.0, y = q * 2.0 - 1.0;
    const x = Math.sign(y) * (Math.exp(ln * Math.abs(y)) - 1.0) / mu; // [-1,1]
    const x01 = (x + 1.0) * 0.5;
    lut[i] = lo + x01 * (hi - lo);
  }
  return lut;
}
