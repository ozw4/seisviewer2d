import { beforeAll, expect, test } from 'vitest';

beforeAll(async () => {
  await import('../../static/viewer/compare/data.js');
});

function data() {
  return window.__svCompareData;
}

function basePayload(overrides = {}) {
  return {
    shape: [1, 2],
    dt: 0.002,
    x0: 10,
    x1: 20,
    y0: 0,
    y1: 4,
    stepX: 10,
    stepY: 4,
    ...overrides,
  };
}

function sources(overrides = {}) {
  return {
    a: { domain: 'amplitude' },
    b: { domain: 'amplitude' },
    ...overrides,
  };
}

function expectF32Values(actual, expected) {
  expect(actual).toBeInstanceOf(Float32Array);
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, index) => {
    expect(actual[index]).toBeCloseTo(value, 6);
  });
}

test('validateComparePair accepts matching shape, grid, dt, and source domain', () => {
  expect(data().validateComparePair(
    basePayload(),
    basePayload(),
    sources(),
  )).toMatchObject({ ok: true, reason: '', message: '' });
});

test('validateComparePair rejects shape mismatches', () => {
  expect(data().validateComparePair(
    basePayload(),
    basePayload({ shape: [2, 1] }),
    sources(),
  )).toMatchObject({ ok: false, reason: 'shape' });
});

test('validateComparePair rejects grid mismatches', () => {
  expect(data().validateComparePair(
    basePayload(),
    basePayload({ stepX: 5 }),
    sources(),
  )).toMatchObject({ ok: false, reason: 'grid' });
});

test('validateComparePair rejects source domain mismatches', () => {
  expect(data().validateComparePair(
    basePayload(),
    basePayload(),
    sources({ b: { domain: 'probability' } }),
  )).toMatchObject({ ok: false, reason: 'domain' });
});

test('subtractF32 subtracts same-length float arrays', () => {
  const diff = data().subtractF32(
    new Float32Array([3, 1]),
    new Float32Array([1, 4]),
  );

  expectF32Values(diff, [2, -3]);
});

test('payloadToF32 restores int8 payloads with the existing scale rule', () => {
  const values = data().payloadToF32({
    shape: [1, 3],
    valuesI8: new Int8Array([64, -32, 16]),
    scale: 128,
  }, { domain: 'probability' });

  expectF32Values(values, [0.5, -0.25, 0.125]);
});

test('canUseCachedComparePayload keeps probability compute-value requirement', () => {
  expect(data().canUseCachedComparePayload(
    { shape: [1, 1], zBacking: new Float32Array([0.5]) },
    { domain: 'amplitude' },
  )).toBe(true);
  expect(data().canUseCachedComparePayload(
    { shape: [1, 1], zBacking: new Float32Array([0.5]) },
    { domain: 'probability' },
  )).toBe(false);
  expect(data().canUseCachedComparePayload(
    { shape: [1, 1], valuesI8: new Int8Array([64]), scale: 128 },
    { domain: 'probability' },
  )).toBe(true);
  expect(data().canUseCachedComparePayload(
    { shape: [1, 1], values: new Float32Array([0.5]) },
    { domain: 'probability' },
  )).toBe(true);
});
