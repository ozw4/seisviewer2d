import { defineConfig } from 'vite';

export default defineConfig({
    base: '/static/',
    build: {
        outDir: 'static/assets',
        assetsDir: '',
        emptyOutDir: false,
        rollupOptions: {
            input: 'web/src/main.js',
            output: {
                entryFileNames: 'main.js',
                chunkFileNames: 'chunk-[hash].js',
                assetFileNames: 'asset-[hash][extname]'
            }
        },

    }
});
