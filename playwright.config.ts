import { defineConfig, devices } from '@playwright/test';

const baseURL = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:8000';
const webServerReadyURL = `${baseURL}/upload`;
const isPerfBenchmark = process.env.PLAYWRIGHT_PERF_BENCHMARK === '1';

export default defineConfig({
	testDir: './tests/e2e',
	timeout: 60_000,
	expect: {
		timeout: 10_000,
	},
	fullyParallel: false,
	forbidOnly: !!process.env.CI,
	retries: process.env.CI ? 2 : 0,
	workers: process.env.CI ? 2 : 1,
	reporter: process.env.CI ? [['html'], ['list']] : [['list'], ['html']],
	use: {
		baseURL,
		headless: true,
		trace: isPerfBenchmark ? 'retain-on-failure' : 'on-first-retry',
		screenshot: 'only-on-failure',
		video: isPerfBenchmark ? 'off' : 'retain-on-failure',
	},
	webServer: {
		command: 'uvicorn app.main:app --host 0.0.0.0 --port 8000',
		url: webServerReadyURL,
		reuseExistingServer: !process.env.CI,
		timeout: 120_000,
	},
	projects: [
		{
			name: 'chromium',
			use: {
				...devices['Desktop Chrome'],
				browserName: 'chromium',
			},
		},
	],
});
