import { test, expect } from '@playwright/test';
import {
	attachSvPerfRows,
	attachSvPerfSummary,
	type SvPerfRow,
} from './helpers/perfArtifacts';

type RecentDataset = {
	original_name: string;
	key1_byte: number;
	key2_byte: number;
};

type RecentDatasetsResponse = {
	datasets?: RecentDataset[];
};

type OpenSegyResponse = {
	file_id: string;
	reused_trace_store: boolean;
};

const DEFAULT_KEY1_BYTE = 189;
const DEFAULT_KEY2_BYTE = 193;

function parseIntEnv(name: string, fallback: number): number {
	const raw = process.env[name];
	if (!raw) return fallback;
	const value = Number.parseInt(raw, 10);
	if (!Number.isInteger(value)) {
		throw new Error(`${name} must be an integer`);
	}
	return value;
}

function parseOptionalNumberEnv(name: string): number | null {
	const raw = process.env[name];
	if (!raw) return null;
	const value = Number(raw);
	if (!Number.isFinite(value)) {
		throw new Error(`${name} must be a number`);
	}
	return value;
}

function toFiniteNumber(value: unknown): number | null {
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function findLatestWindowRow(rows: SvPerfRow[]): SvPerfRow | null {
	for (let index = rows.length - 1; index >= 0; index -= 1) {
		const row = rows[index];
		if (row?.kind === 'window') {
			return row;
		}
	}
	return null;
}

test('perf artifact helper stores empty rows when SV_PERF_ROWS is missing', async (
	{ page },
	testInfo,
) => {
	await page.goto('/upload');

	const rows = await attachSvPerfRows(page, testInfo, 'upload-no-perf');
	await attachSvPerfSummary(page, testInfo, 'upload-no-perf');

	expect(rows).toEqual([]);
});

test('viewer initial render perf attaches JSON artifact', async ({ page, request }, testInfo) => {
	const originalName = process.env.SV_PERF_ORIGINAL_NAME?.trim() || null;
	const key1Byte = parseIntEnv('SV_PERF_KEY1_BYTE', DEFAULT_KEY1_BYTE);
	const key2Byte = parseIntEnv('SV_PERF_KEY2_BYTE', DEFAULT_KEY2_BYTE);
	const maxTotalMs = parseOptionalNumberEnv('SV_PERF_MAX_TOTAL_MS');

	const recentResponse = await request.get('/recent_datasets');
	expect(recentResponse.ok()).toBeTruthy();

	const recentPayload = (await recentResponse.json()) as RecentDatasetsResponse;
	const datasets = Array.isArray(recentPayload.datasets) ? recentPayload.datasets : [];
	const matchingDatasets = datasets.filter(
		(dataset) =>
			dataset.original_name &&
			dataset.key1_byte === key1Byte &&
			dataset.key2_byte === key2Byte,
	);
	const dataset = originalName
		? matchingDatasets.find((item) => item.original_name === originalName)
		: matchingDatasets[0];

	test.skip(
		!dataset,
		originalName
			? `No reusable TraceStore matched ${originalName} with key1_byte=${key1Byte} key2_byte=${key2Byte}.`
			: `No reusable TraceStore matched key1_byte=${key1Byte} key2_byte=${key2Byte}.`,
	);

	const openResponse = await request.post('/open_segy', {
		multipart: {
			original_name: dataset.original_name,
			key1_byte: String(key1Byte),
			key2_byte: String(key2Byte),
		},
	});
	test.skip(
		openResponse.status() === 404,
		`TraceStore ${dataset.original_name} is no longer available for perf measurement.`,
	);
	expect(openResponse.ok()).toBeTruthy();

	const openPayload = (await openResponse.json()) as OpenSegyResponse;
	expect(openPayload.file_id).toBeTruthy();

	const viewerParams = new URLSearchParams({
		file_id: openPayload.file_id,
		key1_byte: String(key1Byte),
		key2_byte: String(key2Byte),
		perf: '1',
	});
	const windowResponsePromise = page.waitForResponse(
		(response) =>
			response.url().includes('/get_section_window_bin') && response.status() === 200,
		{ timeout: 60_000 },
	);

	await page.goto(`/?${viewerParams.toString()}`, { waitUntil: 'domcontentloaded' });
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

	const windowResponse = await windowResponsePromise;
	await windowResponse.finished();

	await page.waitForFunction(
		() => {
			const rows = (globalThis as { SV_PERF_ROWS?: Array<{ kind?: string }> }).SV_PERF_ROWS;
			return Array.isArray(rows) && rows.some((row) => row?.kind === 'window');
		},
		null,
		{ timeout: 60_000 },
	);

	const rows = await attachSvPerfRows(page, testInfo, 'viewer-initial-render');
	await attachSvPerfSummary(page, testInfo, 'viewer-initial-render');
	expect(rows.some((row) => row.kind === 'window')).toBeTruthy();

	const perfRow = findLatestWindowRow(rows);

	expect(perfRow).not.toBeNull();
	expect(perfRow?.kind).toBe('window');
	expect(toFiniteNumber(perfRow?.total_ms)).not.toBeNull();

	const responseHeaders = await windowResponse.allHeaders();
	const contentLengthHeader = responseHeaders['content-length'];
	const parsedContentLength = contentLengthHeader
		? Number.parseInt(contentLengthHeader, 10)
		: Number.NaN;
	const requestTiming = windowResponse.request().timing();
	const artifact = {
		dataset: {
			original_name: dataset.original_name,
			key1_byte: key1Byte,
			key2_byte: key2Byte,
		},
		open_segy: {
			reused_trace_store: openPayload.reused_trace_store,
		},
		get_section_window_bin: {
			url: windowResponse.url(),
			status: windowResponse.status(),
			content_length: Number.isFinite(parsedContentLength) ? parsedContentLength : null,
			request_timing: {
				startTime: toFiniteNumber(requestTiming.startTime),
				domainLookupStart: toFiniteNumber(requestTiming.domainLookupStart),
				domainLookupEnd: toFiniteNumber(requestTiming.domainLookupEnd),
				connectStart: toFiniteNumber(requestTiming.connectStart),
				secureConnectionStart: toFiniteNumber(requestTiming.secureConnectionStart),
				connectEnd: toFiniteNumber(requestTiming.connectEnd),
				requestStart: toFiniteNumber(requestTiming.requestStart),
				responseStart: toFiniteNumber(requestTiming.responseStart),
				responseEnd: toFiniteNumber(requestTiming.responseEnd),
			},
		},
		sv_perf_window: {
			mode: perfRow?.mode ?? null,
			plot: perfRow?.plot ?? null,
			rows: perfRow?.rows ?? null,
			cols: perfRow?.cols ?? null,
			stepX: perfRow?.stepX ?? null,
			stepY: perfRow?.stepY ?? null,
			fetch_ms: perfRow?.fetch_ms ?? null,
			decode_ms: perfRow?.decode_ms ?? null,
			lut_ms: perfRow?.lut_ms ?? null,
			prep_ms: perfRow?.prep_ms ?? null,
			plotly_ms: perfRow?.plotly_ms ?? null,
			total_ms: perfRow?.total_ms ?? null,
			bytes: perfRow?.bytes ?? null,
		},
	};

	await testInfo.attach('viewer-initial-perf', {
		body: Buffer.from(`${JSON.stringify(artifact, null, 2)}\n`, 'utf-8'),
		contentType: 'application/json',
	});

	if (maxTotalMs !== null) {
		expect(toFiniteNumber(perfRow?.total_ms)).not.toBeNull();
		expect(perfRow!.total_ms!).toBeLessThanOrEqual(maxTotalMs);
	}
});
