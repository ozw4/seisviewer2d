import { expect, type APIRequestContext } from '@playwright/test';

export type PerfDataset = {
	original_name: string;
	key1_byte: number;
	key2_byte: number;
};

type RecentDatasetsResponse = {
	datasets?: PerfDataset[];
};

export type OpenPreparedStoreResult = {
	file_id: string;
	reused_trace_store: boolean;
};

export type PerfDatasetConfig = {
	originalName: string | null;
	key1Byte: number;
	key2Byte: number;
};

export const DEFAULT_KEY1_BYTE = 189;
export const DEFAULT_KEY2_BYTE = 193;

export function parseIntEnv(name: string, fallback: number): number {
	const raw = process.env[name];
	if (!raw) return fallback;
	const value = Number.parseInt(raw, 10);
	if (!Number.isInteger(value)) {
		throw new Error(`${name} must be an integer`);
	}
	return value;
}

export function parseOptionalNumberEnv(name: string): number | null {
	const raw = process.env[name];
	if (!raw) return null;
	const value = Number(raw);
	if (!Number.isFinite(value)) {
		throw new Error(`${name} must be a number`);
	}
	return value;
}

export function readPerfDatasetConfig(): PerfDatasetConfig {
	return {
		originalName: process.env.SV_PERF_ORIGINAL_NAME?.trim() || null,
		key1Byte: parseIntEnv('SV_PERF_KEY1_BYTE', DEFAULT_KEY1_BYTE),
		key2Byte: parseIntEnv('SV_PERF_KEY2_BYTE', DEFAULT_KEY2_BYTE),
	};
}

export function buildDatasetSkipReason(config: PerfDatasetConfig): string {
	return config.originalName
		? `No reusable TraceStore matched ${config.originalName} with key1_byte=${config.key1Byte} key2_byte=${config.key2Byte}.`
		: `No reusable TraceStore matched key1_byte=${config.key1Byte} key2_byte=${config.key2Byte}.`;
}

export function buildOpenPreparedStoreSkipReason(dataset: PerfDataset): string {
	return `TraceStore ${dataset.original_name} is no longer available for perf measurement.`;
}

export async function resolveDataset(
	request: APIRequestContext,
	config: PerfDatasetConfig = readPerfDatasetConfig(),
): Promise<PerfDataset | null> {
	const recentResponse = await request.get('/recent_datasets');
	expect(recentResponse.ok()).toBeTruthy();

	const recentPayload = (await recentResponse.json()) as RecentDatasetsResponse;
	const datasets = Array.isArray(recentPayload.datasets) ? recentPayload.datasets : [];
	const matchingDatasets = datasets.filter(
		(dataset) =>
			dataset.original_name &&
			dataset.key1_byte === config.key1Byte &&
			dataset.key2_byte === config.key2Byte,
	);

	return config.originalName
		? matchingDatasets.find((item) => item.original_name === config.originalName) ?? null
		: matchingDatasets[0] ?? null;
}

export async function openPreparedStore(
	request: APIRequestContext,
	dataset: PerfDataset,
): Promise<OpenPreparedStoreResult | null> {
	const openResponse = await request.post('/open_segy', {
		multipart: {
			original_name: dataset.original_name,
			key1_byte: String(dataset.key1_byte),
			key2_byte: String(dataset.key2_byte),
		},
	});
	if (openResponse.status() === 404) {
		return null;
	}

	expect(openResponse.ok()).toBeTruthy();
	return (await openResponse.json()) as OpenPreparedStoreResult;
}

export function buildViewerPerfUrl(
	fileId: string,
	dataset: Pick<PerfDataset, 'key1_byte' | 'key2_byte'>,
): string {
	const viewerParams = new URLSearchParams({
		file_id: fileId,
		key1_byte: String(dataset.key1_byte),
		key2_byte: String(dataset.key2_byte),
		perf: '1',
	});
	return `/?${viewerParams.toString()}`;
}
