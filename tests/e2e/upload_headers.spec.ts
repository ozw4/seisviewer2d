import { expect, test, type Page } from '@playwright/test';

const BASE_URL = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:8000';

function headerQcPayload(stagedId: string, recommendedPairs = [
	{
		key1_byte: 189,
		key1_name: 'INLINE_3D',
		key2_byte: 193,
		key2_name: 'CROSSLINE_3D',
		score: 0.94,
		confidence: 'high',
		reasons: ['key1 has 120 sections with median 200 traces/section'],
		warnings: [],
	},
	{
		key1_byte: 21,
		key1_name: 'CDP',
		key2_byte: 25,
		key2_name: 'CDP_TRACE',
		score: 0.82,
		confidence: 'medium',
		reasons: ['key2 is mostly unique within key1 sections'],
		warnings: ['small sections detected'],
	},
]) {
	return {
		staged_id: stagedId,
		file: {
			original_name: 'line001.sgy',
			safe_name: 'line001.sgy',
			size: 4096,
			sha256: 'abc123',
		},
		segy: {
			n_traces: 24000,
			n_samples: 1500,
			dt: 0.002,
		},
		recommended_pairs: recommendedPairs,
		headers: [
			{
				byte: 189,
				name: 'INLINE_3D',
				available: true,
				min: 1001,
				max: 1120,
				unique_count: 120,
				unique_ratio: 0.005,
				key1_score: 0.91,
				group_size: {
					min: 200,
					p05: 200,
					p50: 200,
					p95: 200,
					max: 200,
				},
				warnings: [],
			},
		],
		warnings: [],
	};
}

async function gotoUpload(page: Page) {
	await page.route('**/recent_datasets', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({ datasets: [] }),
		});
	});
	await page.goto(`${BASE_URL}/upload`);
}

async function selectSegyFile(page: Page, name = 'line001.sgy') {
	await page.setInputFiles('#upload_segy', {
		name,
		mimeType: 'application/octet-stream',
		buffer: Buffer.from('mock segy'),
	});
}

function expectMultipartValue(body: string, field: string, value: string) {
	expect(body).toContain(`name="${field}"`);
	expect(body).toContain(value);
}

test('upload page analyzes headers and opens staged SEG-Y', async ({ page }) => {
	let stageCalls = 0;
	let ingestBody = '';

	await page.route('**/stage_segy', async (route) => {
		stageCalls += 1;
		expect(route.request().method()).toBe('POST');
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(headerQcPayload('staged-123')),
		});
	});
	await page.route('**/ingest_staged_segy', async (route) => {
		ingestBody = route.request().postData() ?? '';
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({ file_id: 'viewer-file', reused_trace_store: false }),
		});
	});
	await page.route(/\/\?file_id=/, async (route) => {
		await route.fulfill({ status: 200, contentType: 'text/html', body: '<html></html>' });
	});

	await gotoUpload(page);

	await expect(page.locator('#analyzeHeadersBtn')).toBeDisabled();
	await expect(page.locator('#headerQcPanel')).toBeHidden();

	await selectSegyFile(page);

	await expect(page.locator('#analyzeHeadersBtn')).toBeEnabled();
	await expect(page.locator('#upload_btn')).toBeDisabled();
	expect(stageCalls).toBe(0);

	await page.locator('#analyzeHeadersBtn').click();

	await expect(page.locator('#headerQcPanel')).toBeVisible();
	await expect(page.getByText('INLINE_3D byte 189 / CROSSLINE_3D byte 193')).toBeVisible();
	await expect(page.locator('#key1_byte')).toHaveValue('189');
	await expect(page.locator('#key2_byte')).toHaveValue('193');
	await expect(page.locator('#upload_btn')).toBeEnabled();

	const cdpPair = page.getByTestId('recommended-pair').filter({ hasText: 'CDP byte 21' });
	await cdpPair.getByTestId('use-recommended-pair').click();
	await expect(page.locator('#key1_byte')).toHaveValue('21');
	await expect(page.locator('#key2_byte')).toHaveValue('25');

	await page.locator('#upload_btn').click();
	await expect.poll(() => ingestBody).toContain('staged-123');
	expectMultipartValue(ingestBody, 'key1_byte', '21');
	expectMultipartValue(ingestBody, 'key2_byte', '25');
	await page.waitForURL('**/?file_id=viewer-file&key1_byte=21&key2_byte=25');
});

test('changing selected file resets staged QC state', async ({ page }) => {
	await page.route('**/stage_segy', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(headerQcPayload('staged-reset')),
		});
	});

	await gotoUpload(page);
	await selectSegyFile(page, 'line-a.sgy');
	await page.locator('#analyzeHeadersBtn').click();

	await expect(page.locator('#headerQcPanel')).toBeVisible();
	await expect(page.locator('#upload_btn')).toBeEnabled();

	await selectSegyFile(page, 'line-b.sgy');

	await expect(page.locator('#headerQcPanel')).toBeHidden();
	await expect(page.locator('#upload_btn')).toBeDisabled();
	await expect(page.locator('#analyzeHeadersBtn')).toBeEnabled();
	await expect(page.locator('#status_summary')).toContainText('line-b.sgy selected.');
});

test('stale analyze response is ignored after selected file changes', async ({ page }) => {
	let releaseStageResponse!: () => void;
	const stageCanFinish = new Promise<void>((resolve) => {
		releaseStageResponse = resolve;
	});
	let markStageStarted!: () => void;
	const stageStarted = new Promise<void>((resolve) => {
		markStageStarted = resolve;
	});
	let ingestCalls = 0;

	await page.route('**/stage_segy', async (route) => {
		markStageStarted();
		await stageCanFinish;
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(headerQcPayload('staged-old')),
		});
	});
	await page.route('**/ingest_staged_segy', async (route) => {
		ingestCalls += 1;
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({ file_id: 'old-file', reused_trace_store: false }),
		});
	});

	await gotoUpload(page);
	await selectSegyFile(page, 'line-a.sgy');
	await page.locator('#analyzeHeadersBtn').click();
	await stageStarted;
	await selectSegyFile(page, 'line-b.sgy');

	const staleStageResponse = page.waitForResponse('**/stage_segy');
	releaseStageResponse();
	await staleStageResponse;
	await page.evaluate(() => new Promise((resolve) => window.setTimeout(resolve, 0)));

	await expect(page.locator('#headerQcPanel')).toBeHidden();
	await expect(page.locator('#upload_btn')).toBeDisabled();
	await expect(page.locator('#status_summary')).toContainText('line-b.sgy selected.');
	await expect.poll(() => ingestCalls).toBe(0);
});

test('empty recommended pairs still allow manual staged open', async ({ page }) => {
	let ingestBody = '';

	await page.route('**/stage_segy', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(headerQcPayload('staged-empty', [])),
		});
	});
	await page.route('**/ingest_staged_segy', async (route) => {
		ingestBody = route.request().postData() ?? '';
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({ file_id: 'manual-file', reused_trace_store: false }),
		});
	});
	await page.route(/\/\?file_id=/, async (route) => {
		await route.fulfill({ status: 200, contentType: 'text/html', body: '<html></html>' });
	});

	await gotoUpload(page);
	await selectSegyFile(page);
	await page.locator('#analyzeHeadersBtn').click();

	await expect(page.locator('#recommendedPairs')).toContainText('No confident key pair found. Please choose key bytes manually.');
	await expect(page.locator('#upload_btn')).toBeEnabled();

	await page.selectOption('#key1_byte', '189');
	await page.selectOption('#key2_byte', '193');
	await page.locator('#upload_btn').click();

	await expect.poll(() => ingestBody).toContain('staged-empty');
	expectMultipartValue(ingestBody, 'key1_byte', '189');
	expectMultipartValue(ingestBody, 'key2_byte', '193');
	await page.waitForURL('**/?file_id=manual-file&key1_byte=189&key2_byte=193');
});

test('analyze failure shows error and keeps open disabled', async ({ page }) => {
	await page.route('**/stage_segy', async (route) => {
		await route.fulfill({
			status: 500,
			contentType: 'application/json',
			body: JSON.stringify({ detail: 'header scanner failed' }),
		});
	});

	await gotoUpload(page);
	await selectSegyFile(page);
	await page.locator('#analyzeHeadersBtn').click();

	await expect(page.locator('#status_summary')).toHaveText('Failed to analyze SEG-Y headers.');
	await expect(page.locator('#status_detail')).toHaveText('header scanner failed');
	await expect(page.locator('#headerQcPanel')).toBeHidden();
	await expect(page.locator('#upload_btn')).toBeDisabled();
});

test('same key byte validation prevents staged ingest', async ({ page }) => {
	let ingestCalls = 0;

	await page.route('**/stage_segy', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(headerQcPayload('staged-validation')),
		});
	});
	await page.route('**/ingest_staged_segy', async (route) => {
		ingestCalls += 1;
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({ file_id: 'should-not-open', reused_trace_store: false }),
		});
	});

	await gotoUpload(page);
	await selectSegyFile(page);
	await page.locator('#analyzeHeadersBtn').click();
	await expect(page.locator('#upload_btn')).toBeEnabled();

	await page.selectOption('#key1_byte', '189');
	await page.selectOption('#key2_byte', '189');
	await page.locator('#upload_btn').click();

	await expect(page.locator('#status_summary')).toHaveText('key1_byte and key2_byte must differ.');
	await expect.poll(() => ingestCalls).toBe(0);
});

test('expired staged SEG-Y error resets staged open state', async ({ page }) => {
	await page.route('**/stage_segy', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(headerQcPayload('staged-expired')),
		});
	});
	await page.route('**/ingest_staged_segy', async (route) => {
		await route.fulfill({
			status: 404,
			contentType: 'application/json',
			body: JSON.stringify({ detail: 'Staged SEG-Y not found' }),
		});
	});

	await gotoUpload(page);
	await selectSegyFile(page);
	await page.locator('#analyzeHeadersBtn').click();
	await expect(page.locator('#upload_btn')).toBeEnabled();

	await page.locator('#upload_btn').click();

	await expect(page.locator('#status_summary')).toHaveText('Failed to open staged SEG-Y.');
	await expect(page.locator('#status_detail')).toHaveText('Staged SEG-Y expired. Please run Analyze headers again.');
	await expect(page.locator('#headerQcPanel')).toBeHidden();
	await expect(page.locator('#upload_btn')).toBeDisabled();
	await expect(page.locator('#analyzeHeadersBtn')).toBeEnabled();
});
