import { test, expect } from '@playwright/test';
import {
	attachSvPerfRows,
	attachSvPerfSummary,
} from './helpers/perfArtifacts';
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
	findLatestWindowRow,
	readWindowResponseMetadata,
	toFiniteNumber,
	waitForViewerPerfReady,
} from './helpers/perfScenarios';

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
	const datasetConfig = readPerfDatasetConfig();
	const maxTotalMs = parseOptionalNumberEnv('SV_PERF_MAX_TOTAL_MS');

	const dataset = await resolveDataset(request, datasetConfig);

	test.skip(
		!dataset,
		buildDatasetSkipReason(datasetConfig),
	);

	const openPayload = await openPreparedStore(request, dataset!);
	test.skip(
		!openPayload,
		buildOpenPreparedStoreSkipReason(dataset!),
	);
	expect(openPayload?.file_id).toBeTruthy();

	const windowResponsePromise = page.waitForResponse(
		(response) =>
			response.url().includes('/get_section_window_bin') && response.status() === 200,
		{ timeout: 60_000 },
	);

	await page.goto(buildViewerPerfUrl(openPayload!.file_id, dataset!), {
		waitUntil: 'domcontentloaded',
	});
	await waitForViewerPerfReady(page);

	const windowResponse = await windowResponsePromise;

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

	const responseMetadata = await readWindowResponseMetadata(windowResponse);
	const artifact = {
		dataset: {
			original_name: dataset.original_name,
			key1_byte: dataset.key1_byte,
			key2_byte: dataset.key2_byte,
		},
		open_segy: {
			reused_trace_store: openPayload.reused_trace_store,
		},
		get_section_window_bin: {
			url: responseMetadata.url,
			status: responseMetadata.status,
			content_length: responseMetadata.contentLength,
			request_timing: responseMetadata.requestTiming,
			headers: responseMetadata.headers,
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
