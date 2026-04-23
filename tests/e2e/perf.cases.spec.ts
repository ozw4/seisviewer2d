import { test, expect, type Page } from '@playwright/test';
import { buildSvPerfArtifactMetadata, readSvPerfRows } from './helpers/perfArtifacts';
import {
	buildDatasetSkipReason,
	buildOpenPreparedStoreSkipReason,
	buildViewerPerfUrl,
	openPreparedStore,
	parseOptionalNumberEnv,
	readPerfDatasetConfig,
	resolveDataset,
} from './helpers/perfDataset';
import {
	buildPanAfterZoomRange,
	buildSkippedPerfScenarioResult,
	buildZoomedRange,
	findLatestWindowRow,
	measurePerfScenario,
	readPlotRanges,
	relayoutPlot,
	toFiniteNumber,
	type PerfScenarioResult,
	type PlotRanges,
} from './helpers/perfScenarios';

const DEFAULT_ZOOM_RATIO = 0.4;
const DEFAULT_PAN_SHIFT_RATIO = 0.9;

type ScenarioThresholdEnv = {
	total: string;
	plotly: string;
};

const THRESHOLD_ENV_BY_LABEL: Record<string, ScenarioThresholdEnv> = {
	'cold-initial': {
		total: 'SV_PERF_MAX_COLD_TOTAL_MS',
		plotly: 'SV_PERF_MAX_COLD_PLOTLY_MS',
	},
	'warm-initial': {
		total: 'SV_PERF_MAX_WARM_TOTAL_MS',
		plotly: 'SV_PERF_MAX_WARM_PLOTLY_MS',
	},
	'zoom-in': {
		total: 'SV_PERF_MAX_ZOOM_TOTAL_MS',
		plotly: 'SV_PERF_MAX_ZOOM_PLOTLY_MS',
	},
	'pan-after-zoom': {
		total: 'SV_PERF_MAX_PAN_TOTAL_MS',
		plotly: 'SV_PERF_MAX_PAN_PLOTLY_MS',
	},
};

function parseBoundedNumberEnv(
	name: string,
	fallback: number,
	options: { gt?: number; lt?: number } = {},
): number {
	const raw = process.env[name];
	if (!raw) {
		return fallback;
	}

	const value = Number(raw);
	if (!Number.isFinite(value)) {
		throw new Error(`${name} must be a number`);
	}
	if (options.gt !== undefined && !(value > options.gt)) {
		throw new Error(`${name} must be greater than ${options.gt}`);
	}
	if (options.lt !== undefined && !(value < options.lt)) {
		throw new Error(`${name} must be less than ${options.lt}`);
	}
	return value;
}

function assertMeasuredScenario(result: PerfScenarioResult): asserts result is Extract<
	PerfScenarioResult,
	{ status: 'measured' }
> {
	expect(result.status).toBe('measured');
}

function assertScenarioMetrics(result: Extract<PerfScenarioResult, { status: 'measured' }>): void {
	expect(result.deltaWindowRows.length).toBeGreaterThan(0);

	const latestWindowRow = findLatestWindowRow(result.deltaRows);
	expect(latestWindowRow).not.toBeNull();
	expect(result.latestWindowRow).toEqual(latestWindowRow);
	const totalMs = toFiniteNumber(latestWindowRow?.total_ms);
	const plotlyMs = toFiniteNumber(latestWindowRow?.plotly_ms);
	const thresholdEnv = THRESHOLD_ENV_BY_LABEL[result.label];
	const maxTotalMs = parseOptionalNumberEnv(thresholdEnv.total);
	const maxPlotlyMs = parseOptionalNumberEnv(thresholdEnv.plotly);

	if (result.label === 'cold-initial' || result.label === 'warm-initial') {
		expect(totalMs).not.toBeNull();
	} else {
		expect(plotlyMs).not.toBeNull();
	}

	if (maxTotalMs !== null && totalMs !== null) {
		expect(totalMs).toBeLessThanOrEqual(maxTotalMs);
	}

	if (maxPlotlyMs !== null) {
		expect(plotlyMs).not.toBeNull();
		expect(plotlyMs!).toBeLessThanOrEqual(maxPlotlyMs);
	}
}

function buildRangeSkipReason(label: string): string {
	return `${label} skipped because #plot axis ranges are unavailable.`;
}

async function openWarmViewerPage(coldPage: Page): Promise<Page> {
	const warmPage = await coldPage.context().newPage();
	return warmPage;
}

test('viewer perf cases attach JSON artifacts', async ({ page, request }, testInfo) => {
	const datasetConfig = readPerfDatasetConfig();
	const zoomRatio = parseBoundedNumberEnv('SV_PERF_ZOOM_RATIO', DEFAULT_ZOOM_RATIO, {
		gt: 0,
		lt: 1,
	});
	const panShiftRatio = parseBoundedNumberEnv(
		'SV_PERF_PAN_SHIFT_RATIO',
		DEFAULT_PAN_SHIFT_RATIO,
		{ gt: 0 },
	);

	const dataset = await resolveDataset(request, datasetConfig);
	test.skip(!dataset, buildDatasetSkipReason(datasetConfig));

	const openPayload = await openPreparedStore(request, dataset!);
	test.skip(!openPayload, buildOpenPreparedStoreSkipReason(dataset!));
	expect(openPayload?.file_id).toBeTruthy();

	const viewerUrl = buildViewerPerfUrl(openPayload!.file_id, dataset!);
	const cases: PerfScenarioResult[] = [];

	const coldInitial = await measurePerfScenario(page, 'cold-initial', async () => {
		await page.goto(viewerUrl, { waitUntil: 'domcontentloaded' });
	});
	cases.push(coldInitial);
	assertMeasuredScenario(coldInitial);
	assertScenarioMetrics(coldInitial);

	const warmPage = await openWarmViewerPage(page);
	try {
		const warmInitial = await measurePerfScenario(warmPage, 'warm-initial', async () => {
			await warmPage.goto(viewerUrl, { waitUntil: 'domcontentloaded' });
		});
		cases.push(warmInitial);
		assertMeasuredScenario(warmInitial);
		assertScenarioMetrics(warmInitial);

		const fullRanges = await readPlotRanges(warmPage);
		let zoomResult: PerfScenarioResult;
		let zoomedRanges: PlotRanges | null = null;

		if (!fullRanges) {
			zoomResult = buildSkippedPerfScenarioResult(
				'zoom-in',
				buildRangeSkipReason('zoom-in'),
				await readSvPerfRows(warmPage),
			);
		} else {
			const nextZoomX = buildZoomedRange(fullRanges.x, zoomRatio);
			const nextZoomY = buildZoomedRange(fullRanges.y, zoomRatio);
			if (!nextZoomX || !nextZoomY) {
				zoomResult = buildSkippedPerfScenarioResult(
					'zoom-in',
					'zoom-in skipped because the current viewport is too small to shrink safely.',
					await readSvPerfRows(warmPage),
				);
			} else {
				zoomResult = await measurePerfScenario(warmPage, 'zoom-in', async () => {
					await relayoutPlot(warmPage, { x: nextZoomX, y: nextZoomY });
				}, {
					skipOnMissingWindowRowReason:
						'zoom-in skipped because relayout did not produce a new window measurement.',
				});
				zoomedRanges = await readPlotRanges(warmPage);
			}
		}
		cases.push(zoomResult);
		if (zoomResult.status === 'measured') {
			assertScenarioMetrics(zoomResult);
		}

		let panResult: PerfScenarioResult;
		if (!fullRanges) {
			panResult = buildSkippedPerfScenarioResult(
				'pan-after-zoom',
				buildRangeSkipReason('pan-after-zoom'),
				await readSvPerfRows(warmPage),
			);
		} else if (zoomResult.status !== 'measured' || !zoomedRanges) {
			panResult = buildSkippedPerfScenarioResult(
				'pan-after-zoom',
				'pan-after-zoom skipped because zoom-in did not produce a reusable viewport.',
				await readSvPerfRows(warmPage),
			);
		} else {
			const nextPanX = buildPanAfterZoomRange(
				zoomedRanges.x,
				fullRanges.x,
				panShiftRatio,
			);
			if (!nextPanX) {
				panResult = buildSkippedPerfScenarioResult(
					'pan-after-zoom',
					'x range is too small to pan safely',
					await readSvPerfRows(warmPage),
				);
			} else {
				panResult = await measurePerfScenario(warmPage, 'pan-after-zoom', async () => {
					await relayoutPlot(warmPage, { x: nextPanX, y: zoomedRanges!.y });
				}, {
					skipOnMissingWindowRowReason:
						'pan-after-zoom skipped because relayout did not produce a new window measurement.',
				});
			}
		}
		cases.push(panResult);
		if (panResult.status === 'measured') {
			assertScenarioMetrics(panResult);
		}

		const metadata = buildSvPerfArtifactMetadata(warmPage, testInfo, 'viewer-perf-cases');
		const summary = {
			metadata: {
				...metadata,
				url: viewerUrl,
			},
			dataset: {
				original_name: dataset!.original_name,
				key1_byte: dataset!.key1_byte,
				key2_byte: dataset!.key2_byte,
			},
			openSegy: {
				reused_trace_store: openPayload!.reused_trace_store,
			},
			cases,
		};
		const rowsArtifact = {
			metadata: {
				...metadata,
				url: viewerUrl,
			},
			dataset: {
				original_name: dataset!.original_name,
				key1_byte: dataset!.key1_byte,
				key2_byte: dataset!.key2_byte,
			},
			cases: cases.map((scenario) => ({
				label: scenario.label,
				status: scenario.status,
				skipReason: scenario.status === 'skipped' ? scenario.skipReason : null,
				rows: scenario.deltaRows,
			})),
		};

		await testInfo.attach('viewer-perf-cases-summary.json', {
			body: Buffer.from(`${JSON.stringify(summary, null, 2)}\n`, 'utf-8'),
			contentType: 'application/json',
		});
		await testInfo.attach('viewer-perf-cases-sv-perf-rows.json', {
			body: Buffer.from(`${JSON.stringify(rowsArtifact, null, 2)}\n`, 'utf-8'),
			contentType: 'application/json',
		});
	} finally {
		await warmPage.close();
	}
});
