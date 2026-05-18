import { expect, test, type Page, type Route } from '@playwright/test';

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

async function openStandaloneStaticCorrection(
	page: Page,
	{ fileId = 'restore-line', key1Byte = 9, key2Byte = 13 } = {},
) {
	await page.goto(`/static-correction?file_id=${fileId}&key1_byte=${key1Byte}&key2_byte=${key2Byte}`);
	await expect(page.getByTestId('static-correction-panel')).toBeVisible();
	await expect(page.getByTestId('static-correction-target-status')).toHaveText('Ready');
}

async function selectStandalonePickNpz(
	page: Page,
	name = 'restored-picks.npz',
	buffer = Buffer.from('npz-restore-bytes'),
) {
	await page.getByTestId('static-correction-pick-npz').setInputFiles({
		name,
		mimeType: 'application/octet-stream',
		buffer,
	});
	await expect(page.getByTestId('static-correction-pick-npz-summary')).toContainText(name);
	await expect.poll(async () => page.evaluate(() => (
		JSON.parse(window.localStorage.getItem('sv.static_correction.form_draft.v1') || '{}').pickNpz?.filename || ''
	))).toBe(name);
}

async function staticCorrectionPickInputSnapshot(page: Page) {
	return page.getByTestId('static-correction-pick-npz').evaluate((input) => {
		const files = (input as HTMLInputElement).files;
		return {
			length: files?.length ?? 0,
			name: files?.[0]?.name ?? '',
			size: files?.[0]?.size ?? 0,
		};
	});
}

function multipartRequestBody(route: Route): string {
	return route.request().postData() || '';
}

test('Static Correction restores form draft and NPZ file input from IndexedDB', async ({ page }) => {
	await openStandaloneStaticCorrection(page, { fileId: 'restore-line', key1Byte: 9, key2Byte: 13 });
	await selectStandalonePickNpz(page, 'restored-picks.npz', Buffer.from('restored-npz'));
	await page.getByTestId('static-correction-min-offset').fill('475');
	await page.getByTestId('static-correction-register-corrected-file').check();

	await page.goto('/refraction-qc');
	await openStandaloneStaticCorrection(page, { fileId: 'restore-line', key1Byte: 9, key2Byte: 13 });

	await expect(page.getByTestId('static-correction-pick-npz-summary')).toContainText(
		'Restored NPZ: restored-picks.npz',
	);
	await expect(page.getByTestId('static-correction-pick-npz-summary')).toContainText(
		'This restored NPZ is loaded into the file input and will be submitted.',
	);
	await expect(page.getByTestId('static-correction-min-offset')).toHaveValue('475');
	await expect(page.getByTestId('static-correction-register-corrected-file')).toBeChecked();
	await expect.poll(async () => staticCorrectionPickInputSnapshot(page)).toMatchObject({
		length: 1,
		name: 'restored-picks.npz',
		size: Buffer.byteLength('restored-npz'),
	});
});

test('Static Correction Run submits restored file input NPZ', async ({ page }) => {
	let multipartBody = '';
	await openStandaloneStaticCorrection(page, { fileId: 'submit-restored-line', key1Byte: 9, key2Byte: 13 });
	await selectStandalonePickNpz(page, 'submit-restored-picks.npz', Buffer.from('submit-restored-npz'));
	await page.goto('/refraction-qc');
	await openStandaloneStaticCorrection(page, { fileId: 'submit-restored-line', key1Byte: 9, key2Byte: 13 });
	await expect.poll(async () => staticCorrectionPickInputSnapshot(page)).toMatchObject({
		length: 1,
		name: 'submit-restored-picks.npz',
	});

	await page.route('**/statics/refraction/apply-with-picks', async (route) => {
		multipartBody = multipartRequestBody(route);
		await fulfillJson(route, { state: 'ready' });
	});
	await page.getByTestId('static-correction-run').click();
	await expect.poll(() => multipartBody).toContain('name="pick_npz"; filename="submit-restored-picks.npz"');
	expect(multipartBody).toContain('"file_id":"submit-restored-line"');
	expect(multipartBody).toContain('"pick_source":{"kind":"uploaded_npz"}');
});

test('Static Correction selected NPZ replaces restored NPZ and clear removes cache', async ({ page }) => {
	await openStandaloneStaticCorrection(page, { fileId: 'replace-restored-line', key1Byte: 9, key2Byte: 13 });
	await selectStandalonePickNpz(page, 'old-picks.npz', Buffer.from('old-npz'));
	await page.goto('/refraction-qc');
	await openStandaloneStaticCorrection(page, { fileId: 'replace-restored-line', key1Byte: 9, key2Byte: 13 });
	await expect(page.getByTestId('static-correction-pick-npz-summary')).toContainText('Restored NPZ: old-picks.npz');

	await selectStandalonePickNpz(page, 'new-picks.npz', Buffer.from('new-npz'));
	await expect(page.getByTestId('static-correction-pick-npz-summary')).toContainText('new-picks.npz');
	await expect(page.getByTestId('static-correction-pick-npz-summary')).not.toContainText('Restored NPZ');

	await page.getByTestId('static-correction-clear-pick-npz').click();
	await expect(page.getByTestId('static-correction-pick-npz-summary')).toContainText('No NPZ file selected.');
	await expect.poll(async () => staticCorrectionPickInputSnapshot(page)).toMatchObject({ length: 0 });
	await expect.poll(async () => page.evaluate(() => (
		JSON.parse(window.localStorage.getItem('sv.static_correction.form_draft.v1') || '{}').pickNpz
	))).toBeNull();
});

test('Static Correction clear draft removes draft and cached NPZ record', async ({ page }) => {
	await openStandaloneStaticCorrection(page, { fileId: 'clear-draft-line', key1Byte: 9, key2Byte: 13 });
	await selectStandalonePickNpz(page, 'clear-draft-picks.npz', Buffer.from('clear-draft'));
	const recordId = await page.evaluate(() => (
		JSON.parse(window.localStorage.getItem('sv.static_correction.form_draft.v1') || '{}')
			.pickNpz?.indexedDbRecordId
	));

	await page.getByTestId('static-correction-clear-draft').click();

	await expect(page.getByTestId('static-correction-pick-npz-summary')).toContainText('No NPZ file selected.');
	await expect.poll(async () => staticCorrectionPickInputSnapshot(page)).toMatchObject({ length: 0 });
	await expect.poll(async () => page.evaluate(() => (
		window.localStorage.getItem('sv.static_correction.form_draft.v1')
	))).toBeNull();
	await expect.poll(async () => page.evaluate(async (recordIdArg) => {
		const record = await (window as any).refractionStaticRunUI.loadPickNpzFromIndexedDb(recordIdArg);
		return record || null;
	}, recordId)).toBeNull();
	await page.goto('/refraction-qc');
	await expect.poll(async () => page.evaluate(() => (
		window.localStorage.getItem('sv.static_correction.form_draft.v1')
	))).toBeNull();
});

test('Static Correction does not restore cached NPZ for a different target', async ({ page }) => {
	await openStandaloneStaticCorrection(page, { fileId: 'target-a', key1Byte: 9, key2Byte: 13 });
	await selectStandalonePickNpz(page, 'target-a-picks.npz', Buffer.from('target-a'));

	await openStandaloneStaticCorrection(page, { fileId: 'target-b', key1Byte: 9, key2Byte: 13 });
	await expect(page.getByTestId('static-correction-pick-npz-summary')).toContainText(
		'Saved NPZ belongs to a different viewer target',
	);
	await expect.poll(async () => staticCorrectionPickInputSnapshot(page)).toMatchObject({ length: 0 });
});

test('Static Correction reports missing IndexedDB NPZ blob', async ({ page }) => {
	await openStandaloneStaticCorrection(page, { fileId: 'missing-blob-line', key1Byte: 9, key2Byte: 13 });
	await selectStandalonePickNpz(page, 'missing-blob-picks.npz', Buffer.from('missing-blob'));
	await page.evaluate(async () => {
		const draft = JSON.parse(window.localStorage.getItem('sv.static_correction.form_draft.v1') || '{}');
		const recordId = draft.pickNpz?.indexedDbRecordId;
		await new Promise<void>((resolve, reject) => {
			const request = window.indexedDB.open('seisviewer2d-static-correction', 1);
			request.onerror = () => reject(request.error);
			request.onsuccess = () => {
				const db = request.result;
				const tx = db.transaction('pick_npz_blobs', 'readwrite');
				tx.objectStore('pick_npz_blobs').delete(recordId);
				tx.oncomplete = () => {
					db.close();
					resolve();
				};
				tx.onerror = () => {
					db.close();
					reject(tx.error);
				};
			};
		});
	});

	await page.goto('/refraction-qc');
	await openStandaloneStaticCorrection(page, { fileId: 'missing-blob-line', key1Byte: 9, key2Byte: 13 });
	await expect(page.getByTestId('static-correction-pick-npz-summary')).toContainText(
		'Saved NPZ is no longer available',
	);
	await expect.poll(async () => staticCorrectionPickInputSnapshot(page)).toMatchObject({ length: 0 });
});

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

	await openStandaloneStaticCorrection(page, { fileId: 'viewer-line', key1Byte: 9, key2Byte: 13 });
	await expect(page.locator('#staticCorrectionTargetFile')).toContainText('viewer-line');
	await page.locator('#staticCorrectionPickNpz').setInputFiles({
		name: 'uploaded-picks.npz',
		mimeType: 'application/octet-stream',
		buffer: Buffer.from('npz-bytes'),
	});
	await page.locator('#staticCorrectionRunButton').click();

	await expect(page).toHaveURL(/\/refraction-qc\?/);
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

	await openStandaloneStaticCorrection(page, { fileId: 'viewer-validate-line', key1Byte: 9, key2Byte: 13 });
	await page.locator('#staticCorrectionPickNpz').setInputFiles({
		name: 'validate-picks.npz',
		mimeType: 'application/octet-stream',
		buffer: Buffer.from('npz-bytes'),
	});
	await page.getByTestId('static-correction-validate').click();

	const diagnostics = page.getByTestId('static-correction-validation-diagnostics');
	await expect(diagnostics).toContainText('Validation passed.');
	await expect(diagnostics).toContainText('viewer-validate-line');
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

	await openStandaloneStaticCorrection(page, { fileId: 'viewer-bad-picks', key1Byte: 9, key2Byte: 13 });
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

	await openStandaloneStaticCorrection(page, { fileId: 'viewer-linked-line', key1Byte: 17, key2Byte: 21 });
	await page.locator('#staticCorrectionPickNpz').setInputFiles({
		name: 'uploaded-linked-picks.npz',
		mimeType: 'application/octet-stream',
		buffer: Buffer.from('npz-bytes'),
	});
	await page.getByTestId('static-correction-enable-linkage').check();
	await page.getByTestId('static-correction-linkage-threshold-m').fill('12.5');
	await page.locator('#staticCorrectionRunButton').click();

	await expect(page).toHaveURL(/\/refraction-qc\?/);
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
