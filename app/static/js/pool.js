// app/static/js/pool.js
export class F32Pool {
  constructor() { this.buf = null; this.cap = 0; }
  acquire(minSize) {
    if (this.cap >= minSize) return this.buf;
    const newCap = Math.max(minSize, this.cap ? this.cap * 2 : 1 << 20); // ~1M elems start
    this.buf = new Float32Array(newCap);
    this.cap = newCap;
    return this.buf;
  }
}

export class F32DoubleBuffer {
  constructor(){ this.a = new F32Pool(); this.b = new F32Pool(); this.flip = false; }
  acquire(size){ this.flip = !this.flip; return (this.flip ? this.a : this.b).acquire(size); }
}
