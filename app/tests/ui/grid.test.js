import { test, expect } from 'vitest';
import * as GridCore from '../../static/viewer/core/grid.js';

test('snapTraceFromDataX snaps to step', () => {
    GridCore.setGrid({ x0: 0, stepX: 5, y0: 0, stepY: 1 });
    const x = 12.2;
    const snapped = GridCore.snapTraceFromDataX(x);
    expect(snapped % 5).toBe(0);
});

test('snapTimeFromDataY uses dt & stepY', () => {
    // dt は window.defaultDt を参照する実装ならセット
    global.window = global.window || {};
    window.defaultDt = 0.002;
    GridCore.setGrid({ x0: 0, stepX: 1, y0: 0, stepY: 4 });
    const y = 0.011; // 5.5 サンプルくらい
    const t = GridCore.snapTimeFromDataY(y);
    // 4サンプル単位に丸められるはず
    expect(Math.abs(((t / 0.002) % 4))).toBeLessThan(1e-6);
});
