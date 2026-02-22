import { expect, test } from 'vitest';

import { buildBatchApplyRequest } from '../../static/batch_apply.js';

function baseState() {
  return {
    fileId: 'file-a',
    key1Byte: '189',
    key2Byte: '193',
    enableBandpass: true,
    enableDenoise: true,
    enableFbpick: true,
    bandpass: {
      lowHz: '5',
      highHz: '60',
      taper: '0.1',
    },
    denoise: {
      chunkH: '128',
      overlap: '32',
      maskRatio: '0.5',
      noiseStd: '1',
      maskNoiseMode: 'replace',
      passesBatch: '4',
    },
    fbpick: {
      modelId: 'fbpick_edgenext_small.pth',
    },
    pick: {
      method: 'argmax',
      sigmaMsMax: '20',
      subsample: true,
      snapEnabled: true,
      snapMode: 'peak',
      snapRefine: 'parabolic',
      snapWindowMs: '20',
    },
    savePicks: true,
  };
}

test('buildBatchApplyRequest builds fixed-order steps and pick payload', () => {
  const payload = buildBatchApplyRequest(baseState());
  expect(payload.file_id).toBe('file-a');
  expect(payload.key1_byte).toBe(189);
  expect(payload.key2_byte).toBe(193);
  expect(payload.pipeline_spec.steps.map((s) => s.name)).toEqual([
    'bandpass',
    'denoise',
    'fbpick',
  ]);
  expect(payload.pipeline_spec.steps[2].params).toEqual({
    model_id: 'fbpick_edgenext_small.pth',
  });
  expect(payload.pick_options).toEqual({
    method: 'argmax',
    subsample: true,
    sigma_ms_max: 20,
    snap: {
      enabled: true,
      mode: 'peak',
      refine: 'parabolic',
      window_ms: 20,
    },
  });
  expect(payload.save_picks).toBe(true);
});

test('buildBatchApplyRequest forces save_picks=false when fbpick is disabled', () => {
  const state = baseState();
  state.enableFbpick = false;
  state.enableBandpass = false;
  state.enableDenoise = true;
  state.savePicks = true;
  const payload = buildBatchApplyRequest(state);
  expect(payload.pipeline_spec.steps.map((s) => s.name)).toEqual(['denoise']);
  expect(payload.save_picks).toBe(false);
});
