import {
	type Page,
	type Response,
} from '@playwright/test';
import { readSvPerfRows, type SvPerfRow } from './perfArtifacts';

export type PerfCaseLabel =
	| 'cold-initial'
	| 'warm-initial'
	| 'zoom-in'
	| 'pan-after-zoom';

export type WindowMetricName =
	| 'fetch_ms'
	| 'decode_ms'
	| 'lut_ms'
	| 'prep_ms'
	| 'plotly_ms'
	| 'total_ms'
	| 'bytes';

export type PlotAxisRange = [number, number];

export type PlotRanges = {
	x: PlotAxisRange;
	y: PlotAxisRange;
};

export type WindowResponseMetadata = {
	url: string;
	status: number;
	contentLength: number | null;
	requestTiming: {
		startTime: number | null;
		domainLookupStart: number | null;
		domainLookupEnd: number | null;
		connectStart: number | null;
		connectEnd: number | null;
		secureConnectionStart: number | null;
		requestStart: number | null;
		responseStart: number | null;
		responseEnd: number | null;
	};
	headers: {
		serverTiming: string | null;
		xSvCache: string | null;
		xSvServerMs: string | null;
		xSvBuildMs: string | null;
		xSvPackMs: string | null;
		xSvBytes: string | null;
	};
};

type MeasuredPerfScenarioResult = {
	label: PerfCaseLabel;
	status: 'measured';
	windowRowsBefore: number;
	windowRowsAfter: number;
	deltaRows: SvPerfRow[];
	deltaWindowRows: SvPerfRow[];
	latestWindowRow: SvPerfRow | null;
	responses: WindowResponseMetadata[];
};

type SkippedPerfScenarioResult = {
	label: PerfCaseLabel;
	status: 'skipped';
	skipReason: string;
	windowRowsBefore: number;
	windowRowsAfter: number;
	deltaRows: SvPerfRow[];
	deltaWindowRows: SvPerfRow[];
	latestWindowRow: SvPerfRow | null;
	responses: WindowResponseMetadata[];
};

export type PerfScenarioResult =
	| MeasuredPerfScenarioResult
	| SkippedPerfScenarioResult;

type MeasurePerfScenarioOptions = {
	skipOnMissingWindowRowReason?: string;
	windowRowTimeoutMs?: number;
};

function isFiniteNumber(value: unknown): value is number {
	return typeof value === 'number' && Number.isFinite(value);
}

export function toFiniteNumber(value: unknown): number | null {
	return isFiniteNumber(value) ? value : null;
}

export function countWindowRows(rows: SvPerfRow[]): number {
	return rows.filter((row) => row.kind === 'window').length;
}

export function findLatestWindowRow(rows: SvPerfRow[]): SvPerfRow | null {
	for (let index = rows.length - 1; index >= 0; index -= 1) {
		const row = rows[index];
		if (row?.kind === 'window') {
			return row;
		}
	}
	return null;
}

export function findLatestWindowRowWithMetric(
	rows: SvPerfRow[],
	metric: WindowMetricName,
): SvPerfRow | null {
	for (let index = rows.length - 1; index >= 0; index -= 1) {
		const row = rows[index];
		if (row?.kind !== 'window') {
			continue;
		}
		if (toFiniteNumber(row[metric]) !== null) {
			return row;
		}
	}
	return null;
}

function buildSvPerfRowSignature(row: SvPerfRow): string {
	return JSON.stringify(row) ?? 'null';
}

function buildSvPerfRowsSignature(rows: SvPerfRow[]): string {
	return rows.map((row) => buildSvPerfRowSignature(row)).join('\n');
}

function findDeltaRows(beforeRows: SvPerfRow[], afterRows: SvPerfRow[]): SvPerfRow[] {
	const beforeSignatures = beforeRows.map((row) => buildSvPerfRowSignature(row));
	const afterSignatures = afterRows.map((row) => buildSvPerfRowSignature(row));
	const maxOverlap = Math.min(beforeSignatures.length, afterSignatures.length);

	for (let overlap = maxOverlap; overlap >= 0; overlap -= 1) {
		let matches = true;
		for (let index = 0; index < overlap; index += 1) {
			const beforeIndex = beforeSignatures.length - overlap + index;
			if (beforeSignatures[beforeIndex] !== afterSignatures[index]) {
				matches = false;
				break;
			}
		}
		if (matches) {
			return afterRows.slice(overlap);
		}
	}

	return afterRows;
}

export async function waitForViewerPerfReady(page: Page): Promise<void> {
	await page.waitForFunction(
		() => (globalThis as { SV_PERF?: boolean }).SV_PERF === true,
		null,
		{ timeout: 60_000 },
	);
	await page.waitForFunction(
		() => document.getElementById('viewerEmptyState')?.hidden === true,
		null,
		{ timeout: 60_000 },
	);
}

export async function waitForSvPerfIdle(
	page: Page,
	quietMs = 1_000,
	timeoutMs = 60_000,
): Promise<void> {
	const startedAt = Date.now();
	let lastRowsSignature = buildSvPerfRowsSignature(await readSvPerfRows(page));
	let lastChangeAt = startedAt;
	const pollMs = Math.max(50, Math.min(250, Math.floor(quietMs / 4)));

	for (;;) {
		const now = Date.now();
		if (now - lastChangeAt >= quietMs) {
			return;
		}
		if (now - startedAt >= timeoutMs) {
			throw new Error(`Timed out waiting for SV_PERF_ROWS to stay idle for ${quietMs}ms.`);
		}

		await page.waitForTimeout(pollMs);
		const nextRowsSignature = buildSvPerfRowsSignature(await readSvPerfRows(page));
		if (nextRowsSignature !== lastRowsSignature) {
			lastRowsSignature = nextRowsSignature;
			lastChangeAt = Date.now();
		}
	}
}

export async function readWindowResponseMetadata(
	response: Response,
): Promise<WindowResponseMetadata> {
	await response.finished().catch(() => null);

	const headers = await response.allHeaders();
	const contentLengthHeader = headers['content-length'];
	const parsedContentLength = contentLengthHeader
		? Number.parseInt(contentLengthHeader, 10)
		: Number.NaN;
	const requestTiming = response.request().timing();

	return {
		url: response.url(),
		status: response.status(),
		contentLength: Number.isFinite(parsedContentLength) ? parsedContentLength : null,
		requestTiming: {
			startTime: toFiniteNumber(requestTiming.startTime),
			domainLookupStart: toFiniteNumber(requestTiming.domainLookupStart),
			domainLookupEnd: toFiniteNumber(requestTiming.domainLookupEnd),
			connectStart: toFiniteNumber(requestTiming.connectStart),
			connectEnd: toFiniteNumber(requestTiming.connectEnd),
			secureConnectionStart: toFiniteNumber(requestTiming.secureConnectionStart),
			requestStart: toFiniteNumber(requestTiming.requestStart),
			responseStart: toFiniteNumber(requestTiming.responseStart),
			responseEnd: toFiniteNumber(requestTiming.responseEnd),
		},
		headers: {
			serverTiming: headers['server-timing'] ?? null,
			xSvCache: headers['x-sv-cache'] ?? null,
			xSvServerMs: headers['x-sv-server-ms'] ?? null,
			xSvBuildMs: headers['x-sv-build-ms'] ?? null,
			xSvPackMs: headers['x-sv-pack-ms'] ?? null,
			xSvBytes: headers['x-sv-bytes'] ?? null,
		},
	};
}

export async function measurePerfScenario(
	page: Page,
	label: PerfCaseLabel,
	action: () => Promise<void>,
	options: MeasurePerfScenarioOptions = {},
): Promise<PerfScenarioResult> {
	const beforeRows = await readSvPerfRows(page);
	const beforeWindowRowCount = countWindowRows(beforeRows);
	const responseTasks: Array<Promise<WindowResponseMetadata>> = [];
	const windowRowTimeoutMs = options.windowRowTimeoutMs ?? 60_000;
	const onResponse = (response: Response) => {
		if (!response.url().includes('/get_section_window_bin')) {
			return;
		}
		responseTasks.push(readWindowResponseMetadata(response));
	};

	page.on('response', onResponse);
	let missingWindowRow = false;
	try {
		await action();
		await waitForViewerPerfReady(page);
		try {
			await page.waitForFunction(
				(countBefore) => {
					const rows = (globalThis as { SV_PERF_ROWS?: Array<{ kind?: string }> }).SV_PERF_ROWS;
					if (!Array.isArray(rows)) {
						return false;
					}
					let windowRowCount = 0;
					for (const row of rows) {
						if (row?.kind === 'window') {
							windowRowCount += 1;
						}
					}
					return windowRowCount > countBefore;
				},
				beforeWindowRowCount,
				{ timeout: windowRowTimeoutMs },
			);
		} catch (error) {
			if (!options.skipOnMissingWindowRowReason) {
				throw error;
			}
			if (!(error instanceof Error) || !error.message.includes('page.waitForFunction')) {
				throw error;
			}
			missingWindowRow = true;
		}

		if (!missingWindowRow) {
			await waitForSvPerfIdle(page);
		}
	} finally {
		page.off('response', onResponse);
	}

	const afterRows = await readSvPerfRows(page);
	const deltaRows = findDeltaRows(beforeRows, afterRows);
	const deltaWindowRows = deltaRows.filter((row) => row.kind === 'window');
	const responses = await Promise.all(responseTasks);

	if (missingWindowRow) {
		return {
			label,
			status: 'skipped',
			skipReason: options.skipOnMissingWindowRowReason!,
			windowRowsBefore: beforeWindowRowCount,
			windowRowsAfter: countWindowRows(afterRows),
			deltaRows,
			deltaWindowRows,
			latestWindowRow: findLatestWindowRow(deltaWindowRows),
			responses,
		};
	}

	return {
		label,
		status: 'measured',
		windowRowsBefore: beforeWindowRowCount,
		windowRowsAfter: countWindowRows(afterRows),
		deltaRows,
		deltaWindowRows,
		latestWindowRow: findLatestWindowRow(deltaWindowRows),
		responses,
	};
}

export function buildSkippedPerfScenarioResult(
	label: PerfCaseLabel,
	skipReason: string,
	rows: SvPerfRow[] = [],
): PerfScenarioResult {
	const windowRows = countWindowRows(rows);
	return {
		label,
		status: 'skipped',
		skipReason,
		windowRowsBefore: windowRows,
		windowRowsAfter: windowRows,
		deltaRows: [],
		deltaWindowRows: [],
		latestWindowRow: null,
		responses: [],
	};
}

export async function readPlotRanges(page: Page): Promise<PlotRanges | null> {
	await page.waitForSelector('#plot', { timeout: 60_000 });
	return page.evaluate(() => {
		const plot = document.getElementById('plot') as
			| (HTMLElement & {
					_fullLayout?: {
						xaxis?: { range?: unknown };
						yaxis?: { range?: unknown };
					};
			  })
			| null;
		const xRange = plot?._fullLayout?.xaxis?.range;
		const yRange = plot?._fullLayout?.yaxis?.range;

		if (
			!Array.isArray(xRange) ||
			xRange.length !== 2 ||
			!Number.isFinite(xRange[0]) ||
			!Number.isFinite(xRange[1]) ||
			!Array.isArray(yRange) ||
			yRange.length !== 2 ||
			!Number.isFinite(yRange[0]) ||
			!Number.isFinite(yRange[1])
		) {
			return null;
		}

		return {
			x: [Number(xRange[0]), Number(xRange[1])] as PlotAxisRange,
			y: [Number(yRange[0]), Number(yRange[1])] as PlotAxisRange,
		};
	});
}

function preserveRangeOrder(reference: PlotAxisRange, lo: number, hi: number): PlotAxisRange {
	return reference[0] <= reference[1] ? [lo, hi] : [hi, lo];
}

export function buildZoomedRange(
	range: PlotAxisRange,
	ratio: number,
): PlotAxisRange | null {
	const start = Number(range[0]);
	const end = Number(range[1]);
	if (!Number.isFinite(start) || !Number.isFinite(end)) {
		return null;
	}
	const span = Math.abs(end - start);
	if (!(span > 0) || !(ratio > 0) || !(ratio < 1)) {
		return null;
	}

	const center = (start + end) * 0.5;
	const halfNextSpan = span * ratio * 0.5;
	return preserveRangeOrder(range, center - halfNextSpan, center + halfNextSpan);
}

export function buildPanAfterZoomRange(
	currentRange: PlotAxisRange,
	fullRange: PlotAxisRange,
	shiftRatio: number,
): PlotAxisRange | null {
	const currentLo = Math.min(currentRange[0], currentRange[1]);
	const currentHi = Math.max(currentRange[0], currentRange[1]);
	const fullLo = Math.min(fullRange[0], fullRange[1]);
	const fullHi = Math.max(fullRange[0], fullRange[1]);
	const span = currentHi - currentLo;
	const shift = span * shiftRatio;

	if (!(span > 0) || !(shift > 0) || !(fullHi > fullLo)) {
		return null;
	}

	const rightLo = currentLo + shift;
	const rightHi = currentHi + shift;
	if (rightHi <= fullHi) {
		return preserveRangeOrder(currentRange, rightLo, rightHi);
	}

	const leftLo = currentLo - shift;
	const leftHi = currentHi - shift;
	if (leftLo >= fullLo) {
		return preserveRangeOrder(currentRange, leftLo, leftHi);
	}

	return null;
}

export async function relayoutPlot(
	page: Page,
	ranges: { x?: PlotAxisRange; y?: PlotAxisRange },
): Promise<void> {
	await page.waitForSelector('#plot', { timeout: 60_000 });
	await page.evaluate(async ({ nextXRange, nextYRange }) => {
		const plot = document.getElementById('plot');
		const relayout = (window as Window & {
			Plotly?: { relayout?: (target: Element, props: Record<string, unknown>) => Promise<void> };
		}).Plotly?.relayout;
		if (!plot || typeof relayout !== 'function') {
			throw new Error('Plotly relayout is unavailable');
		}

		const props: Record<string, unknown> = {};
		if (Array.isArray(nextXRange)) {
			props['xaxis.range'] = nextXRange;
		}
		if (Array.isArray(nextYRange)) {
			props['yaxis.range'] = nextYRange;
		}
		await relayout(plot, props);
	}, {
		nextXRange: ranges.x ?? null,
		nextYRange: ranges.y ?? null,
	});
}
