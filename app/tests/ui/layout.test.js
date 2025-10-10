import { test, expect } from 'vitest';
import { buildLayout } from '../../static/viewer/core/layout.js';
test('heatmap y-range centers include half cell', () => {
    const layout = buildLayout({
        mode: 'heatmap',
        x0: 0, x1: 9,
        y0: 0, y1: 99,
        stepX: 1, stepY: 2,
        totalSamples: 100,
        dt: 0.002,
        savedXRange: null, savedYRange: null,
        clickmode: 'event', dragmode: 'zoom', uirevision: 'test'
    });
    expect(layout.yaxis.range[0]).toBeGreaterThan(layout.yaxis.range[1]); // y軸が上向き
});
