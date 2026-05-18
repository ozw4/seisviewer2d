import { expect, test, type Route } from '@playwright/test';

declare global {
	interface Window {
		SeisViewerState?: {
			syncActiveFileTarget?: (state: {
				fileId: string;
				displayName: string;
				key1Byte: number;
				key2Byte: number;
				isFileLoaded: boolean;
			}) => unknown;
		};
	}
}

async function fulfillJson(route: Route, payload: unknown, status = 200) {
	await route.fulfill({
		status,
		contentType: 'application/json',
		body: JSON.stringify(payload),
	});
}

test('direct NPZ Static Correction posts multipart request and auto-loads Refraction QC', async ({
	page,
}) => {
	const calls: string[] = [];
	let multipartBody = '';
	let qcRequestBody = '';

	await page.route('**/statics/**', async (route) => {
		const request = route.request();
		const url = new URL(request.url());
		calls.push(`${request.method()} ${url.pathname}`);

		if (request.method() === 'POST' && url.pathname === '/statics/refraction/apply-with-picks') {
			multipartBody = request.postData() || '';
			await fulfillJson(route, { job_id: 'static-job-e2e', state: 'queued' });
			return;
		}
		if (request.method() === 'GET' && url.pathname === '/statics/job/static-job-e2e/status') {
			await fulfillJson(route, {
				state: 'ready',
				message: 'finished',
				progress: 1,
			});
			return;
		}
		if (request.method() === 'GET' && url.pathname === '/statics/job/static-job-e2e/files') {
			await fulfillJson(route, {
				files: [
					{ name: 'refraction_static_qc.json', size_bytes: 512 },
					{ name: 'refraction_static_artifacts.json', size_bytes: 256 },
				],
			});
			return;
		}
		if (request.method() === 'POST' && url.pathname === '/statics/refraction/qc') {
			qcRequestBody = request.postData() || '';
			await fulfillJson(route, {
				job_id: 'static-job-e2e',
				summary: { status: 'ready', workflow: 'refraction' },
				available_views: ['summary'],
				unavailable_views: [],
				coordinate_mode: 'auto',
			});
			return;
		}

		throw new Error(`Unexpected Static Correction request: ${request.method()} ${url.pathname}`);
	});

	await page.goto('/');
	await page.waitForFunction(() => Boolean(window.SeisViewerState?.syncActiveFileTarget));
	await page.evaluate(() => {
		window.SeisViewerState?.syncActiveFileTarget?.({
			fileId: 'viewer-line',
			displayName: 'viewer-line.sgy',
			key1Byte: 9,
			key2Byte: 13,
			isFileLoaded: true,
		});
	});

	await page.getByTestId('static-correction-tab').click();
	await expect(page.locator('#staticCorrectionTargetFile')).toContainText('viewer-line');
	await page.locator('#staticCorrectionPickNpz').setInputFiles({
		name: 'uploaded-picks.npz',
		mimeType: 'application/octet-stream',
		buffer: Buffer.from('npz-bytes'),
	});
	await page.locator('#staticCorrectionRunButton').click();

	await expect(page.getByTestId('refraction-qc-tab')).toHaveAttribute('aria-selected', 'true');
	await expect(page.locator('#refractionQcJobId')).toHaveValue('static-job-e2e');
	await expect(page.locator('#refractionQcStatus')).toContainText('Loaded static-job-e2e');

	expect(calls).toEqual([
		'POST /statics/refraction/apply-with-picks',
		'GET /statics/job/static-job-e2e/status',
		'GET /statics/job/static-job-e2e/files',
		'POST /statics/refraction/qc',
	]);
	expect(multipartBody).toContain('name="request_json"');
	expect(multipartBody).toContain('"file_id":"viewer-line"');
	expect(multipartBody).toContain('"key1_byte":9');
	expect(multipartBody).toContain('"key2_byte":13');
	expect(multipartBody).toContain('"pick_source":{"kind":"uploaded_npz"}');
	expect(multipartBody).not.toContain('batch_predicted_npz');
	expect(multipartBody).toContain('"linkage":{"mode":"none"}');
	expect(multipartBody).toContain('name="pick_npz"; filename="uploaded-picks.npz"');
	expect(qcRequestBody).toContain('"job_id":"static-job-e2e"');
});

test('Static Correction Validate Inputs uses current viewer target and displays filter counts', async ({
	page,
}) => {
	const calls: string[] = [];
	let multipartBody = '';

	await page.route('**/statics/**', async (route) => {
		const request = route.request();
		const url = new URL(request.url());
		calls.push(`${request.method()} ${url.pathname}`);

		if (request.method() === 'POST' && url.pathname === '/statics/refraction/validate-with-picks') {
			multipartBody = request.postData() || '';
			await fulfillJson(route, {
				status: 'ok',
				target: { file_id: 'viewer-validate-line', key1_byte: 9, key2_byte: 13 },
				pick_npz: { selected_key: 'pick_time_s', shape: [4], keys: ['pick_time_s'] },
				diagnostics: {
					n_total_traces: 4,
					n_finite_picks: 4,
					n_valid_picks: 4,
					n_used_for_inversion: 3,
					n_unique_source_endpoints: 2,
					n_unique_receiver_endpoints: 4,
					offset_m: { min: 100, median: 250, max: 400 },
					filter_reason_counts: { offset_gate: 1, missing_pick: 0 },
				},
				warnings: [],
				errors: [],
			});
			return;
		}

		throw new Error(`Unexpected Static Correction request: ${request.method()} ${url.pathname}`);
	});

	await page.goto('/');
	await page.waitForFunction(() => Boolean(window.SeisViewerState?.syncActiveFileTarget));
	await page.evaluate(() => {
		window.SeisViewerState?.syncActiveFileTarget?.({
			fileId: 'viewer-validate-line',
			displayName: 'viewer-validate-line.sgy',
			key1Byte: 9,
			key2Byte: 13,
			isFileLoaded: true,
		});
	});

	await page.getByTestId('static-correction-tab').click();
	await page.locator('#staticCorrectionPickNpz').setInputFiles({
		name: 'validate-picks.npz',
		mimeType: 'application/octet-stream',
		buffer: Buffer.from('npz-bytes'),
	});
	await page.getByTestId('static-correction-validate').click();

	const diagnostics = page.getByTestId('static-correction-validation-diagnostics');
	await expect(diagnostics).toContainText('Validation passed.');
	await expect(diagnostics).toContainText('viewer-validate-line.sgy');
	await expect(diagnostics).toContainText('9/13');
	await expect(diagnostics).toContainText('pick_time_s');
	await expect(diagnostics).toContainText('n_total_traces');
	await expect(diagnostics).toContainText('n_finite_picks');
	await expect(diagnostics).toContainText('n_used_for_inversion');
	await expect(diagnostics).toContainText('"offset_gate":1');
	expect(calls).toEqual(['POST /statics/refraction/validate-with-picks']);
	expect(multipartBody).toContain('name="request_json"');
	expect(multipartBody).toContain('"file_id":"viewer-validate-line"');
	expect(multipartBody).toContain('"pick_source":{"kind":"uploaded_npz"}');
	expect(multipartBody).toContain('name="pick_npz"; filename="validate-picks.npz"');
});

test('Static Correction Validate Inputs does not call apply endpoint on validation errors', async ({
	page,
}) => {
	const calls: string[] = [];

	await page.route('**/statics/**', async (route) => {
		const request = route.request();
		const url = new URL(request.url());
		calls.push(`${request.method()} ${url.pathname}`);

		if (request.method() === 'POST' && url.pathname === '/statics/refraction/validate-with-picks') {
			await fulfillJson(route, {
				status: 'error',
				target: { file_id: 'viewer-bad-picks', key1_byte: 9, key2_byte: 13 },
				pick_npz: { selected_key: 'pick_time_s', shape: [3], keys: ['pick_time_s'] },
				diagnostics: {
					n_total_traces: 0,
					n_finite_picks: 0,
					n_valid_picks: 0,
					n_used_for_inversion: 0,
					n_unique_source_endpoints: 0,
					n_unique_receiver_endpoints: 0,
					offset_m: { min: null, median: null, max: null },
					filter_reason_counts: {},
				},
				warnings: [],
				errors: ['Invalid npz pick source: n_traces mismatch'],
			});
			return;
		}

		throw new Error(`Unexpected Static Correction request: ${request.method()} ${url.pathname}`);
	});

	await page.goto('/');
	await page.waitForFunction(() => Boolean(window.SeisViewerState?.syncActiveFileTarget));
	await page.evaluate(() => {
		window.SeisViewerState?.syncActiveFileTarget?.({
			fileId: 'viewer-bad-picks',
			displayName: 'viewer-bad-picks.sgy',
			key1Byte: 9,
			key2Byte: 13,
			isFileLoaded: true,
		});
	});

	await page.getByTestId('static-correction-tab').click();
	await page.locator('#staticCorrectionPickNpz').setInputFiles({
		name: 'bad-picks.npz',
		mimeType: 'application/octet-stream',
		buffer: Buffer.from('npz-bytes'),
	});
	await page.getByTestId('static-correction-validate').click();

	const diagnostics = page.getByTestId('static-correction-validation-diagnostics');
	await expect(diagnostics).toContainText('Validation found input issues.');
	await expect(diagnostics).toContainText('n_traces mismatch');
	await expect(page.getByTestId('static-correction-status')).toContainText(
		'Static correction was not submitted',
	);
	expect(calls).toEqual(['POST /statics/refraction/validate-with-picks']);
});

test('direct NPZ Static Correction builds checked linkage before multipart apply', async ({
	page,
}) => {
	const calls: string[] = [];
	let linkageBuildBody = '';
	let multipartBody = '';
	let qcRequestBody = '';

	await page.route('**/statics/**', async (route) => {
		const request = route.request();
		const url = new URL(request.url());
		calls.push(`${request.method()} ${url.pathname}`);

		if (request.method() === 'POST' && url.pathname === '/statics/linkage/build') {
			linkageBuildBody = request.postData() || '';
			await fulfillJson(route, { job_id: 'linkage-job-e2e', state: 'queued' });
			return;
		}
		if (request.method() === 'GET' && url.pathname === '/statics/job/linkage-job-e2e/status') {
			await fulfillJson(route, {
				job_id: 'linkage-job-e2e',
				state: 'done',
				message: 'linked',
				progress: 1,
			});
			return;
		}
		if (request.method() === 'POST' && url.pathname === '/statics/refraction/apply-with-picks') {
			multipartBody = request.postData() || '';
			await fulfillJson(route, { job_id: 'static-job-linked-e2e', state: 'queued' });
			return;
		}
		if (request.method() === 'GET' && url.pathname === '/statics/job/static-job-linked-e2e/status') {
			await fulfillJson(route, {
				state: 'ready',
				message: 'finished',
				progress: 1,
			});
			return;
		}
		if (request.method() === 'GET' && url.pathname === '/statics/job/static-job-linked-e2e/files') {
			await fulfillJson(route, {
				files: [
					{ name: 'refraction_static_qc.json', size_bytes: 512 },
					{ name: 'refraction_static_artifacts.json', size_bytes: 256 },
				],
			});
			return;
		}
		if (request.method() === 'POST' && url.pathname === '/statics/refraction/qc') {
			qcRequestBody = request.postData() || '';
			await fulfillJson(route, {
				job_id: 'static-job-linked-e2e',
				summary: { status: 'ready', workflow: 'refraction' },
				available_views: ['summary'],
				unavailable_views: [],
				coordinate_mode: 'auto',
			});
			return;
		}

		throw new Error(`Unexpected Static Correction request: ${request.method()} ${url.pathname}`);
	});

	await page.goto('/');
	await page.waitForFunction(() => Boolean(window.SeisViewerState?.syncActiveFileTarget));
	await page.evaluate(() => {
		window.SeisViewerState?.syncActiveFileTarget?.({
			fileId: 'viewer-linked-line',
			displayName: 'viewer-linked-line.sgy',
			key1Byte: 17,
			key2Byte: 21,
			isFileLoaded: true,
		});
	});

	await page.getByTestId('static-correction-tab').click();
	await page.locator('#staticCorrectionPickNpz').setInputFiles({
		name: 'uploaded-linked-picks.npz',
		mimeType: 'application/octet-stream',
		buffer: Buffer.from('npz-bytes'),
	});
	await page.getByTestId('static-correction-enable-linkage').check();
	await page.getByTestId('static-correction-linkage-threshold-m').fill('12.5');
	await page.locator('#staticCorrectionRunButton').click();

	await expect(page.getByTestId('refraction-qc-tab')).toHaveAttribute('aria-selected', 'true');
	await expect(page.locator('#refractionQcJobId')).toHaveValue('static-job-linked-e2e');
	await expect(page.locator('#refractionQcStatus')).toContainText('Loaded static-job-linked-e2e');

	expect(calls).toEqual([
		'POST /statics/linkage/build',
		'GET /statics/job/linkage-job-e2e/status',
		'POST /statics/refraction/apply-with-picks',
		'GET /statics/job/static-job-linked-e2e/status',
		'GET /statics/job/static-job-linked-e2e/files',
		'POST /statics/refraction/qc',
	]);
	expect(JSON.parse(linkageBuildBody)).toMatchObject({
		file_id: 'viewer-linked-line',
		key1_byte: 17,
		key2_byte: 21,
		linkage: { mode: 'auto_threshold', threshold_m: 12.5 },
	});
	expect(multipartBody).toContain('"file_id":"viewer-linked-line"');
	expect(multipartBody).toContain('"key1_byte":17');
	expect(multipartBody).toContain('"key2_byte":21');
	expect(multipartBody).toContain('"pick_source":{"kind":"uploaded_npz"}');
	expect(multipartBody).not.toContain('batch_predicted_npz');
	expect(multipartBody).toContain(
		'"linkage":{"mode":"required","job_id":"linkage-job-e2e","artifact_name":"geometry_linkage.npz"}',
	);
	expect(multipartBody).toContain('name="pick_npz"; filename="uploaded-linked-picks.npz"');
	expect(qcRequestBody).toContain('"job_id":"static-job-linked-e2e"');
});
