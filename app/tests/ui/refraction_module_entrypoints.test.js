import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { afterEach, expect, test, vi } from 'vitest';
import { initStaticCorrectionPage } from '../../static/refraction-static-run/main.js';
import { initRefractionQcPage } from '../../static/refraction-qc/main.js';

const STATIC_CORRECTION_HTML = readFileSync(
  resolve(process.cwd(), 'static/static_correction.html'),
  'utf8'
);
const REFRACTION_QC_HTML = readFileSync(
  resolve(process.cwd(), 'static/refraction_qc.html'),
  'utf8'
);
const STATIC_RUN_ENTRY = readFileSync(
  resolve(process.cwd(), 'static/refraction_static_run.js'),
  'utf8'
);
const REFRACTION_QC_ENTRY = readFileSync(
  resolve(process.cwd(), 'static/refraction_qc.js'),
  'utf8'
);
const VITE_CONFIG = readFileSync(resolve(process.cwd(), 'vite.config.ts'), 'utf8');

function moduleUrl(path) {
  return new URL(
    `${path}?test=${Date.now()}-${Math.random()}`,
    `${pathToFileURL(process.cwd()).href}/`
  ).href;
}

afterEach(() => {
  vi.unstubAllGlobals();
  delete window.__SEISVIEWER2D_DEV__;
  delete window.RefractionQc;
  delete window.StaticCorrection;
  delete window.refractionQcUI;
  delete window.refractionQcState;
  delete window.refractionStaticRunUI;
  delete window.refractionStaticRunState;
  document.body.innerHTML = '';
});

test('refraction HTML pages load native module entrypoints', () => {
  expect(STATIC_CORRECTION_HTML).toContain(
    '<script type="module" src="/static/refraction_static_run.js"></script>'
  );
  expect(REFRACTION_QC_HTML).toContain(
    '<script type="module" src="/static/refraction_qc.js"></script>'
  );
});

test('refraction entrypoint files have no legacy IIFE wrapper', () => {
  expect(STATIC_RUN_ENTRY).not.toMatch(/\(function\s*\(\)\s*\{/);
  expect(STATIC_RUN_ENTRY).not.toMatch(/\}\)\(\);/);
  expect(REFRACTION_QC_ENTRY).not.toMatch(/\(function\s*\(\)\s*\{/);
  expect(REFRACTION_QC_ENTRY).not.toMatch(/\}\)\(\);/);
});

test('refraction Vite build inputs include static run and QC entries', () => {
  expect(VITE_CONFIG).toContain("refraction_static_run: resolve(process.cwd(), 'static/refraction_static_run.js')");
  expect(VITE_CONFIG).toContain("refraction_qc: resolve(process.cwd(), 'static/refraction_qc.js')");
});

test('entrypoint imports are safe without required DOM or legacy globals', async () => {
  document.body.innerHTML = '';

  await import(moduleUrl('static/refraction_static_run.js'));
  await import(moduleUrl('static/refraction_qc.js'));

  expect(() => initStaticCorrectionPage()).not.toThrow();
  expect(() => initRefractionQcPage()).not.toThrow();
  expect(window.RefractionQc).toBeUndefined();
  expect(window.StaticCorrection).toBeUndefined();
  expect(window.refractionQcUI).toBeUndefined();
  expect(window.refractionStaticRunUI).toBeUndefined();
  expect(window.__SEISVIEWER2D_DEV__).toBeUndefined();
});
