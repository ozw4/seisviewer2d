import { test, expect, type Page, type TestInfo } from '@playwright/test';
import {
	buildSvPerfArtifactMetadata,
	readSvPerfRows,
	writePerfArtifactJson,
} from './helpers/perfArtifacts';
import {
	buildDatasetSkipReason,
	buildOpenPreparedStoreSkipReason,
	buildViewerPerfUrl,
	openPreparedStore,
	readPerfDatasetConfig,
	resolveDataset,
} from './helpers/perfDataset';
import {
	buildPanAfterZoomRange,
	buildSkippedPerfScenarioResult,
	buildZoomedRange,
	findLatestWindowRow,
	findLatestWindowRowWithMetric,
	measurePerfScenario,
	readPlotRanges,
	relayoutPlot,
	toFiniteNumber,
	type PerfCaseLabel,
	type PerfScenarioResult,
	type PlotRanges,
} from './helpers/perfScenarios';
import {
	buildPerfThresholdFailureMessage,
	evaluatePerfThresholds,
} from './helpers/perfThresholds';

const DEFAULT_ZOOM_RATIO = 0.4;
const DEFAULT_PAN_SHIFT_RATIO = 0.9;
const PERF_CASES_TEST_TIMEOUT_MS = 180_000;
const OPTIONAL_RELAYOUT_WINDOW_ROW_TIMEOUT_MS = 5_000;

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
	const totalMs = toFiniteNumber(
		findLatestWindowRowWithMetric(result.deltaWindowRows, 'total_ms')?.total_ms,
	);
	const plotlyMs = toFiniteNumber(
		findLatestWindowRowWithMetric(result.deltaWindowRows, 'plotly_ms')?.plotly_ms,
	);

	if (result.label === 'cold-initial' || result.label === 'warm-initial') {
		expect(totalMs).not.toBeNull();
	} else {
		expect(plotlyMs).not.toBeNull();
	}
}

function buildRangeSkipReason(label: PerfCaseLabel): string {
	return `${label} skipped because #plot axis ranges are unavailable.`;
}

async function openWarmViewerPage(coldPage: Page): Promise<Page> {
	const warmPage = await coldPage.context().newPage();
	return warmPage;
}

async function attachViewerPerfArtifacts(
	testInfo: TestInfo,
	options: {
		page: Page;
		viewerUrl: string;
		dataset: {
			original_name: string;
			key1_byte: number;
			key2_byte: number;
		};
		openPayload: {
			reused_trace_store: boolean;
		};
		cases: PerfScenarioResult[];
	},
): Promise<string | null> {
	const metadata = buildSvPerfArtifactMetadata(options.page, testInfo, 'viewer-perf-cases');
	const summary = {
		metadata: {
			...metadata,
			url: options.viewerUrl,
		},
		dataset: {
			original_name: options.dataset.original_name,
			key1_byte: options.dataset.key1_byte,
			key2_byte: options.dataset.key2_byte,
		},
		openSegy: {
			reused_trace_store: options.openPayload.reused_trace_store,
		},
		cases: options.cases,
	};
	const rowsArtifact = {
		metadata: {
			...metadata,
			url: options.viewerUrl,
		},
		dataset: {
			original_name: options.dataset.original_name,
			key1_byte: options.dataset.key1_byte,
			key2_byte: options.dataset.key2_byte,
		},
		cases: options.cases.map((scenario) => ({
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

	const thresholdArtifact = evaluatePerfThresholds(options.cases, {
		...metadata,
		url: options.viewerUrl,
		label: 'viewer-perf-thresholds',
	});
	const datasetMetadata = {
		metadata: {
			...metadata,
			url: options.viewerUrl,
		},
		dataset: {
			original_name: options.dataset.original_name,
			key1_byte: options.dataset.key1_byte,
			key2_byte: options.dataset.key2_byte,
		},
		openSegy: {
			reused_trace_store: options.openPayload.reused_trace_store,
		},
	};
	const serverTiming = {
		metadata: {
			...metadata,
			url: options.viewerUrl,
		},
		cases: options.cases.map((scenario) => ({
			label: scenario.label,
			status: scenario.status,
			skipReason: scenario.status === 'skipped' ? scenario.skipReason : null,
			responses: scenario.responses,
		})),
	};
	await testInfo.attach('viewer-perf-threshold-results.json', {
		body: Buffer.from(`${JSON.stringify(thresholdArtifact, null, 2)}\n`, 'utf-8'),
		contentType: 'application/json',
	});
	await Promise.all([
		writePerfArtifactJson('sv-perf-summary.json', summary),
		writePerfArtifactJson('sv-perf-rows.json', rowsArtifact),
		writePerfArtifactJson('viewer-perf-threshold-results.json', thresholdArtifact),
		writePerfArtifactJson('dataset-metadata.json', datasetMetadata),
		writePerfArtifactJson('server-timing.json', serverTiming),
	]);

	return buildPerfThresholdFailureMessage(thresholdArtifact.failures);
}

test('viewer perf cases attach JSON artifacts', async ({ page, request }, testInfo) => {
	test.setTimeout(PERF_CASES_TEST_TIMEOUT_MS);

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
	let warmPage: Page | null = null;
	let thresholdFailureMessage: string | null = null;
	try {
		const coldInitial = await measurePerfScenario(page, 'cold-initial', async () => {
			await page.goto(viewerUrl, { waitUntil: 'domcontentloaded' });
		});
		cases.push(coldInitial);
		assertMeasuredScenario(coldInitial);
		assertScenarioMetrics(coldInitial);

		warmPage = await openWarmViewerPage(page);
		const warmViewerPage = warmPage;
		const warmInitial = await measurePerfScenario(warmViewerPage, 'warm-initial', async () => {
			await warmViewerPage.goto(viewerUrl, { waitUntil: 'domcontentloaded' });
		});
		cases.push(warmInitial);
		assertMeasuredScenario(warmInitial);
		assertScenarioMetrics(warmInitial);

		const fullRanges = await readPlotRanges(warmViewerPage);
		let zoomResult: PerfScenarioResult;
		let zoomedRanges: PlotRanges | null = null;

		if (!fullRanges) {
			zoomResult = buildSkippedPerfScenarioResult(
				'zoom-in',
				buildRangeSkipReason('zoom-in'),
				await readSvPerfRows(warmViewerPage),
			);
		} else {
			const nextZoomX = buildZoomedRange(fullRanges.x, zoomRatio);
			const nextZoomY = buildZoomedRange(fullRanges.y, zoomRatio);
			if (!nextZoomX || !nextZoomY) {
				zoomResult = buildSkippedPerfScenarioResult(
					'zoom-in',
					'zoom-in skipped because the current viewport is too small to shrink safely.',
					await readSvPerfRows(warmViewerPage),
				);
			} else {
				zoomResult = await measurePerfScenario(warmViewerPage, 'zoom-in', async () => {
					await relayoutPlot(warmViewerPage, { x: nextZoomX, y: nextZoomY });
				}, {
					windowRowTimeoutMs: OPTIONAL_RELAYOUT_WINDOW_ROW_TIMEOUT_MS,
					skipOnMissingWindowRowReason:
						'zoom-in skipped because relayout did not produce a new window measurement.',
				});
				zoomedRanges = await readPlotRanges(warmViewerPage);
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
				await readSvPerfRows(warmViewerPage),
			);
		} else if (zoomResult.status !== 'measured' || !zoomedRanges) {
			panResult = buildSkippedPerfScenarioResult(
				'pan-after-zoom',
				'pan-after-zoom skipped because zoom-in did not produce a reusable viewport.',
				await readSvPerfRows(warmViewerPage),
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
					await readSvPerfRows(warmViewerPage),
				);
			} else {
				panResult = await measurePerfScenario(warmViewerPage, 'pan-after-zoom', async () => {
					await relayoutPlot(warmViewerPage, { x: nextPanX, y: zoomedRanges!.y });
				}, {
					windowRowTimeoutMs: OPTIONAL_RELAYOUT_WINDOW_ROW_TIMEOUT_MS,
					skipOnMissingWindowRowReason:
						'pan-after-zoom skipped because relayout did not produce a new window measurement.',
				});
			}
		}
		cases.push(panResult);
		if (panResult.status === 'measured') {
			assertScenarioMetrics(panResult);
		}
	} finally {
		thresholdFailureMessage = await attachViewerPerfArtifacts(testInfo, {
			page: warmPage ?? page,
			viewerUrl,
			dataset: dataset!,
			openPayload: openPayload!,
			cases,
		});
		if (warmPage) {
			await warmPage.close();
		}
	}

	if (thresholdFailureMessage) {
		throw new Error(thresholdFailureMessage);
	}
});
