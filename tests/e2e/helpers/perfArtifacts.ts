import { type Page, type TestInfo } from '@playwright/test';

export type SvPerfRow = {
	kind?: string;
	mode?: string;
	plot?: string;
	rows?: number;
	cols?: number;
	stepX?: number;
	stepY?: number;
	fetch_ms?: number | null;
	decode_ms?: number | null;
	lut_ms?: number | null;
	prep_ms?: number | null;
	plotly_ms?: number | null;
	total_ms?: number | null;
	bytes?: number | null;
};

export type SvPerfArtifactMetadata = {
	label: string;
	url: string;
	browserName: string;
	viewport: { width: number; height: number } | null;
	createdAt: string;
};

export function buildSvPerfArtifactMetadata(
	page: Page,
	testInfo: TestInfo,
	label: string,
): SvPerfArtifactMetadata {
	return {
		label,
		url: page.url(),
		browserName: testInfo.project.use.browserName,
		viewport: page.viewportSize(),
		createdAt: new Date().toISOString(),
	};
}

export async function readSvPerfRows(page: Page): Promise<SvPerfRow[]> {
	const rows = await page.evaluate(() => {
		const value = (globalThis as { SV_PERF_ROWS?: unknown }).SV_PERF_ROWS;
		return Array.isArray(value) ? value : [];
	});

	return rows as SvPerfRow[];
}

export async function attachSvPerfRows(
	page: Page,
	testInfo: TestInfo,
	label: string,
): Promise<SvPerfRow[]> {
	const rows = await readSvPerfRows(page);
	const metadata = buildSvPerfArtifactMetadata(page, testInfo, label);
	const body = JSON.stringify({ metadata, rows }, null, 2);

	await testInfo.attach(`${label}-sv-perf-rows.json`, {
		body: Buffer.from(`${body}\n`, 'utf-8'),
		contentType: 'application/json',
	});

	return rows;
}

export async function attachSvPerfSummary(
	page: Page,
	testInfo: TestInfo,
	label: string,
): Promise<void> {
	const rows = await readSvPerfRows(page);
	const windowRows = rows.filter((row) => row.kind === 'window');
	const latestWindowRow =
		windowRows.length > 0 ? windowRows[windowRows.length - 1] : null;
	const summary = {
		...buildSvPerfArtifactMetadata(page, testInfo, label),
		rowCount: rows.length,
		windowRowCount: windowRows.length,
		latestWindowRow,
	};

	await testInfo.attach(`${label}-sv-perf-summary.json`, {
		body: Buffer.from(`${JSON.stringify(summary, null, 2)}\n`, 'utf-8'),
		contentType: 'application/json',
	});
}
