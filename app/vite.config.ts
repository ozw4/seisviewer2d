import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
    base: '/static/',
    build: {
        outDir: 'static/assets',
        assetsDir: '',
        emptyOutDir: false,
        rollupOptions: {
            input: {
                main: resolve(process.cwd(), 'web/src/main.js'),
                refraction_static_run: resolve(process.cwd(), 'static/refraction_static_run.js'),
                refraction_qc: resolve(process.cwd(), 'static/refraction_qc.js')
            },
            output: {
                entryFileNames: (chunkInfo) => (
                    chunkInfo.name === 'main' ? 'main.js' : '[name].js'
                ),
                chunkFileNames: 'chunk-[hash].js',
                assetFileNames: 'asset-[hash][extname]'
            }
        },

    }
});
